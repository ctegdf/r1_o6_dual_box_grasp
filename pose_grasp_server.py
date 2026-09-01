#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R1-o6 pose + grasp interactive server.

Human-in-the-loop system: user sends end-effector pose commands, cuRobo plans
and executes arm trajectories; user then controls dexterous hand for grasping.

Usage (on the Isaac Lab server):
    See docs/OPERATIONS.md for portable startup commands.

Client (local):
    python pose_grasp_cli.py --host <server-ip> --port 5560

Protocol: JSONL over TCP (one JSON object per line).
Commands: move_to_pose, set_hand, get_state, home, revert, stop, auto_grasp,
          auto_clamp
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_clamp import (
    AUTO_CLAMP_OBJECT_KEYS,
    BimanualLiftState,
    ClampConfig,
    advance_inward_alpha,
    aggregate_hand_normal_force,
    bounded_inward_alpha,
    build_clamp_targets,
    evaluate_force_control,
    make_clamp_config,
    palms_ready_for_squeeze,
    path_progress_from_actual_y,
    palm_alignment,
    squeeze_contact_guard_should_pause,
    squeeze_seat_detected,
    update_bimanual_lift,
    updated_squeeze_compression,
)

DEFAULT_PORT = 5560
STATE_BROADCAST_HZ = 10
LOG_PREFIX = "[pose-grasp]"

TOOL_FRAMES = {"left": "lh_hand_base_link", "right": "rh_hand_base_link"}
DEFAULT_LEFT_CONFIG = "r1_o6_left.yml"
DEFAULT_RIGHT_CONFIG = "r1_o6_right.yml"
LOCK_JOINT_REBUILD_TOL = 1e-2  # inactive arm micro-jitter < 0.01 rad
CONTACT_FORCE_EPS_N = 1e-4

# Auto-clamp hand posture.  The four finger MCP joints and thumb CMC yaw stay
# at zero so the palm remains flat and the thumb does not cant the contact
# surface.  A separate curled posture is used only while raising/retrying for
# table clearance.
# Pure flat-palm contact: thumb CMC yaw/pitch are both neutral, so the thumb
# is not used as an inward clamp actuator when yaw is zero.
# Four fingers are curled only 0.15 rad (8.6 deg), within the requested 10°
# limit, to bring their distal contact surfaces onto the carton wall.
CLAMP_FLAT_HAND_VALUES = (
    0.0, 0.0,
    math.radians(10.0), math.radians(10.0),
    math.radians(10.0), math.radians(10.0),
)
CLAMP_TRANSIT_HAND_VALUES = (0.5, 0.4, 0.8, 0.8, 0.8, 0.8)


def _clamp_flat_hand(config) -> tuple[float, ...]:
    """Return flat-palm hand values, optionally overriding thumb and finger posture."""
    thumb_yaw = getattr(config, "flat_hand_thumb_yaw_rad", 0.0)
    thumb_pitch = getattr(config, "flat_hand_thumb_pitch_rad", 0.0)
    finger_mcp = getattr(config, "flat_hand_finger_mcp_rad", 0.0)
    result = CLAMP_FLAT_HAND_VALUES
    if thumb_yaw or thumb_pitch:
        result = (thumb_yaw, thumb_pitch) + result[2:]
    if finger_mcp:
        result = result[:2] + (finger_mcp,) * 4
    return result

# ---- Auto-grasp common constants ----
DEFAULT_ORIENTATION_TOLERANCE = 3.14
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FOUR_FINGER_NAMES = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class GraspProfile:
    """Per-object parameters for force-guided lateral auto-grasp.

    Flow: OPEN_HAND → RAISE → APPROACH → DESCEND → SETTLE → FORCE_CLOSE
          → VERIFY_GRASP → LIFT → VERIFY_HOLD.
    """
    half_height: float              # object half-height (m)
    y_offset: float                 # lateral approach distance from object center (m)
    x_backoff: float                # robot-side offset from object center (m)
    grasp_z_offset: float = 0.02    # grasp height = object_center_z + offset (m)
    pre_z_clearance: float = 0.0    # approach height above grasp point (m)
    lift_z: float = 0.05            # lift height after stable grasp (m)
    # Multi-candidate approach sampling
    approach_x_delta: float = 0.03  # x-direction perturbation per sample (m)
    approach_z_retry: float = 0.02  # z offset for second round of candidates (m)
    # Hand control
    # Transit position: fingers curled during arm motion to avoid table collision.
    # At q=0 fingers extend ~10cm below hand_base and hit the table surface.
    transit_values: tuple = (0.5, 0.4, 0.8, 0.8, 0.8, 0.8)
    open_values: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    force_close_step: float = 0.1   # finger angle increment per sim tick (rad)
    force_close_max: float = 2.0    # max finger angle before giving up (rad)
    force_close_indices: tuple = (1, 2, 3, 4, 5)  # thumb_pitch + 4 fingers
    # Tactile thresholds
    contact_threshold_n: float = 0.05   # per-finger contact detection (N)
    max_contact_force_n: float = 5.0    # safety limit per finger (N)
    min_total_contacts: int = 2         # minimum total contacting fingers
    min_four_finger_contacts: int = 1   # minimum four-finger contacts (excl thumb)
    # Wrist roll: if set, actively command wrist_roll during grasp phases
    # instead of floating to actual (needed for cup-like side grasps).
    wrist_roll_rad: float | None = None
    # End-effector orientation for approach/descend (base frame, wxyz).
    # If None, cuRobo uses unconstrained orientation (default 3.14 tolerance).
    grasp_quat_wxyz: tuple[float, float, float, float] | None = None
    grasp_orientation_tolerance: float = 0.4  # rad, only used when grasp_quat_wxyz is set
    # --- Cup-style side-raise approach ---
    use_side_raise: bool = False           # raise arm laterally before approach
    side_raise_pitch_rad: float = 0.0      # shoulder_pitch retraction during side raise
    side_raise_roll_rad: float = 1.0       # shoulder_roll target for side raise
    # Pre-grasp hand shape: thumb opposed, fingers open.
    # If set, used during transit instead of transit_values.
    pregrasp_hand: tuple | None = None
    # Pre-grasp Y offset (further from cup); grasp uses y_offset.
    pregrasp_y_offset: float | None = None
    # FK search orientation constraints (like auto_clamp).
    # Palm inward: palm +X dot onto inward direction (≥0 = facing object).
    # Finger down: local +Z dot onto world -Z (≥0 = fingers pointing down).
    fk_min_palm_inward: float = -1.0       # no constraint by default
    fk_min_finger_down: float = -1.0       # no constraint by default
    # Differentiated force close: support fingers close slower and cap earlier.
    force_support_indices: tuple = ()      # e.g. (3, 4, 5) for mid/ring/pinky
    force_support_step: float = 0.02       # slower step for support fingers
    force_support_max_rad: float = 0.6     # max angle for support fingers


# ---- Per-object grasp profiles ----
# Each profile defines geometry offsets + force-guided closing parameters.
# force_close_max is the key per-object tuning knob (narrower → higher value).
GRASP_PROFILES: dict[str, GraspProfile] = {
    # 中号快递纸箱 0.30×0.22×0.20 (half_width_y=0.11)
    "courier_box_m": GraspProfile(
        half_height=0.10, y_offset=0.15, x_backoff=0.04,
        grasp_z_offset=-0.10,  # grasp at table surface (z=-0.10 base)
        approach_z_retry=0.01,
    ),
    # 小号快递纸箱 0.20×0.14×0.10
    "courier_box_s": GraspProfile(
        half_height=0.05, y_offset=0.10, x_backoff=0.06,
        force_close_max=1.6,
    ),
    # 快递文件软包 0.32×0.24×0.03
    "poly_mailer": GraspProfile(
        half_height=0.015, y_offset=0.15, x_backoff=0.08,
        grasp_z_offset=0.005, force_close_max=1.6,
    ),
    # EPS泡沫保温箱 0.35×0.24×0.25
    "foam_box": GraspProfile(
        half_height=0.125, y_offset=0.15, x_backoff=0.12,
        force_close_max=1.1,
    ),
    # PP一次性打包餐盒 0.17×0.12×0.06
    "meal_box": GraspProfile(
        half_height=0.03, y_offset=0.09, x_backoff=0.06,
        grasp_z_offset=0.01, force_close_max=1.6,
    ),
    # 奶茶杯 r=0.045 h=0.16 — 湿滑表面, 需多接触包裹
    "bubble_tea_cup": GraspProfile(
        half_height=0.08,
        y_offset=0.055,             # grasp: 5.5cm from cup center
        x_backoff=0.02,
        grasp_z_offset=0.00,
        pre_z_clearance=0.18,          # approach 18cm above grasp (clear forearm over cup)
        # Transit hand: thumb opposed + curled, four fingers curled to reduce sweep
        transit_values=(1.22, 0.4, 0.8, 0.8, 0.8, 0.8),
        open_values=(1.22, 0.0, 0.0, 0.0, 0.0, 0.0),
        force_close_step=0.04,
        force_close_max=1.2,
        force_close_indices=(1, 2),  # thumb_pitch + index only (primary)
        contact_threshold_n=0.05,
        max_contact_force_n=3.0,
        min_total_contacts=2,
        min_four_finger_contacts=1,
        # Side raise to clear table
        use_side_raise=True,
        side_raise_pitch_rad=-0.5,
        side_raise_roll_rad=1.5,
        # APPROACH goes directly to grasp position (y_offset=0.055).
        # No separate MOVE_GRASP phase — avoids sweeping through cup.
        pregrasp_hand=(1.22, 0.0, 0.0, 0.0, 0.0, 0.0),
        pregrasp_y_offset=None,     # disabled: APPROACH → OPEN_PREGRASP → SETTLE
        # FK search: palm must face inward, fingers pointing down
        fk_min_palm_inward=0.5,
        fk_min_finger_down=0.2,
        # Support fingers: light contact only
        force_support_indices=(3, 4, 5),  # middle, ring, pinky
        force_support_step=0.02,
        force_support_max_rad=0.6,
    ),
    # 纸质咖啡杯 r=0.043 h=0.12
    "coffee_cup": GraspProfile(
        half_height=0.06, y_offset=0.08, x_backoff=0.05,
        force_close_max=1.7,
    ),
    # 披萨盒 0.33×0.33×0.045
    "pizza_box": GraspProfile(
        half_height=0.0225, y_offset=0.20, x_backoff=0.12,
        grasp_z_offset=0.005, force_close_max=1.1,
    ),
    # 汤桶 r=0.055 h=0.09
    "soup_container": GraspProfile(
        half_height=0.045, y_offset=0.09, x_backoff=0.05,
        grasp_z_offset=0.01, force_close_max=1.5,
    ),
    # PET饮料瓶 r=0.033 h=0.22
    "pet_bottle": GraspProfile(
        half_height=0.11, y_offset=0.07, x_backoff=0.04,
        force_close_max=1.8,
    ),
    # 小号圆柱杯 r=0.0165 h=0.08 — 拇指对掌power grasp
    "small_cup_33": GraspProfile(
        half_height=0.04,
        y_offset=0.05,
        x_backoff=0.04,
        grasp_z_offset=0.06,
        lift_z=0.05,
        approach_x_delta=0.02,
        approach_z_retry=0.015,
        transit_values=(1.2, 0.3, 1.0, 1.0, 1.0, 1.0),
        open_values=(1.2, 0.0, 0.6, 0.6, 0.6, 0.6),
        force_close_step=0.04,
        force_close_max=1.2,
        force_close_indices=(1, 2, 3, 4, 5),
        contact_threshold_n=0.05,
        max_contact_force_n=2.0,
        min_total_contacts=2,
        min_four_finger_contacts=1,
    ),
}

HAND_ACTIVE_SUFFIXES = (
    "thumb_cmc_yaw", "thumb_cmc_pitch",
    "index_mcp_pitch", "middle_mcp_pitch",
    "ring_mcp_pitch", "pinky_mcp_pitch",
)


# ============================================================
# JSONL TCP server (network threads only enqueue commands)
# ============================================================

class JsonLineServer:
    """TCP hub: accept clients, receive JSONL commands, broadcast responses."""

    def __init__(self, host: str, port: int, inbox: queue.Queue):
        self._host = host
        self._port = port
        self._inbox = inbox
        self._hello: dict | None = None
        self._sock: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def set_hello(self, payload: dict) -> None:
        self._hello = payload

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.listen(4)
        sock.settimeout(0.5)
        self._sock = sock
        threading.Thread(target=self._accept_loop, daemon=True).start()
        _log(f"Listening on {self._host}:{self._port}")

    def stop(self) -> None:
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._clients.clear()

    def publish(self, msg: dict) -> None:
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        with self._lock:
            dead = []
            for c in self._clients:
                try:
                    c.sendall(data)
                except OSError:
                    dead.append(c)
            for c in dead:
                self._clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                client, addr = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            client.settimeout(0.5)
            with self._lock:
                self._clients.append(client)
            _log(f"Client connected: {addr}")
            if self._hello:
                try:
                    client.sendall(
                        (json.dumps(self._hello, separators=(",", ":")) + "\n").encode()
                    )
                except OSError:
                    pass
            threading.Thread(
                target=self._read_loop, args=(client, addr), daemon=True
            ).start()

    def _read_loop(self, client: socket.socket, addr: tuple) -> None:
        buf = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = client.recv(8192)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict):
                        self._inbox.put(msg)
        finally:
            with self._lock:
                if client in self._clients:
                    self._clients.remove(client)
            client.close()
            _log(f"Client disconnected: {addr}")


# ============================================================
# Trajectory execution state
# ============================================================

class ActiveTrajectory:
    __slots__ = ("request_id", "command", "arm", "joint_names",
                 "q_waypoints", "steps_per_wp", "settle_steps",
                 "wp_idx", "hold_count", "settle_count",
                 "t_start", "suppress_done")

    def __init__(self, request_id: str | None, command: str,
                 arm: str, joint_names: list[str],
                 q_waypoints, steps_per_wp: int,
                 settle_steps: int = 0,
                 suppress_done: bool = False):
        self.request_id = request_id
        self.command = command
        self.arm = arm
        self.joint_names = joint_names
        self.q_waypoints = q_waypoints  # [T, len(joint_names)] tensor
        self.steps_per_wp = steps_per_wp
        self.settle_steps = settle_steps
        self.wp_idx = 0
        self.hold_count = 0
        self.settle_count = 0
        self.t_start = time.monotonic()
        self.suppress_done = suppress_done

    @property
    def num_waypoints(self) -> int:
        return int(self.q_waypoints.shape[0])

    @property
    def settling(self) -> bool:
        return self.wp_idx >= self.num_waypoints and self.settle_count < self.settle_steps

    @property
    def done(self) -> bool:
        return self.wp_idx >= self.num_waypoints and self.settle_count >= self.settle_steps


class DualArmTrajectory:
    """Synchronous two-arm trajectory that explicitly controls both wrists."""

    __slots__ = ("request_id", "command", "joint_names", "q_waypoints",
                 "steps_per_wp", "settle_steps", "wp_idx", "hold_count",
                 "settle_count", "t_start")

    def __init__(self, request_id: str | None, command: str,
                 joint_names: list[str], q_waypoints, steps_per_wp: int,
                 settle_steps: int = 0):
        self.request_id = request_id
        self.command = command
        self.joint_names = joint_names
        self.q_waypoints = q_waypoints
        self.steps_per_wp = steps_per_wp
        self.settle_steps = settle_steps
        self.wp_idx = 0
        self.hold_count = 0
        self.settle_count = 0
        self.t_start = time.monotonic()

    @property
    def num_waypoints(self) -> int:
        return int(self.q_waypoints.shape[0])

    @property
    def settling(self) -> bool:
        return self.wp_idx >= self.num_waypoints and self.settle_count < self.settle_steps

    @property
    def done(self) -> bool:
        return self.wp_idx >= self.num_waypoints and self.settle_count >= self.settle_steps


class AutoGraspPhase(Enum):
    OPEN_HAND = auto()
    SIDE_RAISE = auto()      # single-arm shoulder_roll raise to clear table
    VERIFY_CLEARANCE = auto()  # check fingertips above table, retry if not
    RAISE = auto()
    APPROACH = auto()        # FK search move to pre-grasp position
    OPEN_PREGRASP = auto()   # open hand to pregrasp shape after reaching position
    MOVE_GRASP = auto()      # small FK move from pre-grasp to grasp
    DESCEND = auto()
    SETTLE = auto()
    FORCE_CLOSE = auto()
    VERIFY_GRASP = auto()
    LIFT = auto()
    VERIFY_HOLD = auto()


@dataclass
class AutoGraspContext:
    request_id: str | None
    arm: str
    object_key: str
    profile: GraspProfile
    phase: AutoGraspPhase = AutoGraspPhase.OPEN_HAND
    step_counter: int = 0
    move_started: bool = False
    hand_opened: bool = False
    # Multi-candidate approach
    candidates: list[dict[str, Any]] = field(default_factory=list)
    candidate_idx: int = 0
    attempts: int = 0
    selected: dict[str, Any] | None = None
    raised_done: bool = False
    # Force-close state
    last_forces: dict[str, Any] = field(default_factory=dict)
    t_start: float = field(default_factory=time.monotonic)


class AutoClampPhase(Enum):
    PREPARE_HANDS = auto()
    SIDE_RAISE = auto()
    OPEN_HANDS = auto()
    VERIFY_CLEARANCE = auto()
    SOLVE_PREGRASP_LEFT = auto()
    SOLVE_PREGRASP_RIGHT = auto()
    MOVE_PREGRASP = auto()
    VERIFY_PREGRASP = auto()
    SOLVE_CLAMP_LEFT = auto()
    SOLVE_CLAMP_RIGHT = auto()
    FORCE_CLAMP = auto()
    SOLVE_LIFT_LEFT = auto()
    SOLVE_LIFT_RIGHT = auto()
    MOVE_LIFT = auto()
    VERIFY_LIFT = auto()


@dataclass
class AutoClampContext:
    request_id: str | None
    object_key: str
    config: ClampConfig
    targets: dict[str, dict[str, tuple[float, float, float]]]
    object_snapshot_w: tuple[float, float, float]
    object_snapshot_quat_wxyz: tuple[float, float, float, float]
    phase: AutoClampPhase = AutoClampPhase.PREPARE_HANDS
    t_start: float = field(default_factory=time.monotonic)
    total_sim_steps: int = 0
    phase_sim_steps: int = 0
    step_counter: int = 0
    move_started: bool = False
    side_raise_roll_rad: float = 1.0
    solutions: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {"left": {}, "right": {}}
    )
    metrics: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {"left": {}, "right": {}}
    )
    clamp_alpha: dict[str, float] = field(
        default_factory=lambda: {"left": 0.0, "right": 0.0}
    )
    command_alpha: dict[str, float] = field(
        default_factory=lambda: {"left": 0.0, "right": 0.0}
    )
    squeeze_anchor_q: dict[str, Any] = field(default_factory=dict)
    squeeze_jacobian_y: dict[str, Any] = field(default_factory=dict)
    squeeze_compression_m: dict[str, float] = field(
        default_factory=lambda: {"left": 0.0, "right": 0.0}
    )
    squeeze_seating_complete: bool = False
    squeeze_seat_unload_counter: int = 0
    stable_frames: dict[str, int] = field(
        default_factory=lambda: {"left": 0, "right": 0}
    )
    contact_seen: dict[str, bool] = field(
        default_factory=lambda: {"left": False, "right": False}
    )
    contact_alpha: dict[str, float | None] = field(
        default_factory=lambda: {"left": None, "right": None}
    )
    limit_frames: dict[str, int] = field(
        default_factory=lambda: {"left": 0, "right": 0}
    )
    broad_force_limit_frames: int = 0
    squeeze_contact_loss_frames: int = 0
    lift_force_drop_frames: dict[str, int] = field(
        default_factory=lambda: {"left": 0, "right": 0}
    )
    lift_hard_limit_frames: int = 0
    lift_verify_frames: int = 0
    lift_start_object_z: float | None = None
    lift_squeeze_baseline_m: dict[str, float] = field(default_factory=dict)
    lift_coordinator: BimanualLiftState | None = None
    lift_nominal_q: dict[str, Any] = field(default_factory=dict)
    lift_anchor_features: dict[str, Any] = field(default_factory=dict)
    lift_start_hand_base_z: dict[str, float] = field(default_factory=dict)
    lift_last_command: dict[str, Any] = field(default_factory=dict)
    lift_settle_counter: int = 0
    lift_verify_attempt_frames: int = 0
    last_forces: dict[str, Any] = field(default_factory=dict)
    clearance_world_z: dict[str, float] = field(default_factory=dict)


# ============================================================
# Math helpers
# ============================================================

def _normalize_quat(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-12:
        raise ValueError("zero-length quaternion")
    return [v / norm for v in q]


def _rpy_to_quat_wxyz(rpy: list[float]) -> list[float]:
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return _normalize_quat([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw]


def _quat_inv(q):
    q = _normalize_quat(q)
    return [q[0], -q[1], -q[2], -q[3]]


def _quat_rotate(q, v):
    r = _quat_mul(_quat_mul(q, [0.0, v[0], v[1], v[2]]), _quat_inv(q))
    return r[1:4]


def _as_floats(raw, n: int, name: str) -> list[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != n:
        raise ValueError(f"{name} must be length-{n} list")
    vals = [float(v) for v in raw]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} contains non-finite value")
    return vals


def _log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R1-o6 pose + grasp server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--object", default="courier_box_m")
    parser.add_argument("--robot-config", default="r1_o6.yml",
                        help="Dual-arm config (initial/default pose only)")
    parser.add_argument("--left-config", default=DEFAULT_LEFT_CONFIG,
                        help="Single-arm cuRobo config for left arm")
    parser.add_argument("--right-config", default=DEFAULT_RIGHT_CONFIG,
                        help="Single-arm cuRobo config for right arm")
    parser.add_argument("--plan-attempts", type=int, default=5)
    parser.add_argument("--steps-per-waypoint", type=int, default=2,
                        help="Sim steps to hold each trajectory waypoint")
    parser.add_argument("--settle-steps", type=int, default=20,
                        help="Extra sim steps to hold final waypoint for PD convergence")
    parser.add_argument(
        "--record-dir", default=None,
        help="Save auto-clamp RGB frames here (converted to MP4 after the run)",
    )
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument(
        "--robot-root-x-offset-m", type=float, default=0.0,
        help="Shift the fixed robot root in world X for reach experiments",
    )
    parser.add_argument(
        "--clamp-x-offset-m", type=float, default=0.0,
        help="Shift pregrasp and clamp targets from the object centre in base X",
    )
    parser.add_argument(
        "--clamp-force-target-n", type=float, default=None,
        help=(
            "Override the selected auto-clamp object's per-side filtered "
            "normal-force target"
        ),
    )
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    if args.record_dir:
        from isaaclab.sensors import CameraCfg

    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState, Pose

    from r1_o6_scene_cfg import (
        ARM_JOINT_NAMES,
        HAND_ACTIVE_JOINT_NAMES,
        MIMIC_JOINTS,
        O6_ALL_TACTILE_BODIES,
        O6_FINGERTIP_BODIES,
        O6_OBJECT_CONTACT_SENSOR_NAMES,
        R1_O6_ACTIVE_DOF_NAMES,
        R1_O6_ALL_JOINT_NAMES,
        R1_O6_PASSIVE_DOF_NAMES,
        R1_O6_USD_PATH,
        ROBOT_ROOT_HEIGHT,
        TABLE_CENTER_X,
        TABLE_SURFACE_Z,
        TABLE_TOP_SIZE,
        TABLE_TOP_THICKNESS,
        R1O6DeliverySceneCfg,
        make_delivery_object_cfg,
    )

    if not Path(R1_O6_USD_PATH).is_file():
        raise FileNotFoundError(f"USD not found: {R1_O6_USD_PATH}")

    device = args.device if hasattr(args, "device") else "cuda:0"

    # ---- Scene setup ----
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        device=device,
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = R1O6DeliverySceneCfg(
        num_envs=1, env_spacing=3.0, replicate_physics=True
    )
    scene_cfg.robot.init_state.pos = (
        args.robot_root_x_offset_m,
        0.0,
        ROBOT_ROOT_HEIGHT,
    )
    scene_cfg.object = make_delivery_object_cfg(args.object)
    if args.record_dir:
        scene_cfg.record_camera = CameraCfg(
            prim_path="/World/RecordCamera",
            offset=CameraCfg.OffsetCfg(
                pos=(1.8, 1.2, 1.5),
                # Camera local +X/+Y/+Z are forward/left/up in the world
                # convention; this basis looks from the table toward the box.
                rot=(-0.33415332, -0.15570849, -0.05607491, 0.92787501),
                convention="world",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                clipping_range=(0.1, 10.0),
            ),
            width=640,
            height=360,
        )
    sim.set_camera_view(eye=(1.8, 1.2, 1.5), target=(0.35, 0.0, 0.85))
    scene = InteractiveScene(scene_cfg)
    robot = scene["robot"]
    object_asset = scene["object"]
    contact_hands = scene["contact_hands"]
    object_contact_sensors = {
        body: scene[sensor_name]
        for body, sensor_name in O6_OBJECT_CONTACT_SENSOR_NAMES.items()
    }
    palm_object_sensors = {
        "left": object_contact_sensors["lh_hand_base_link"],
        "right": object_contact_sensors["rh_hand_base_link"],
    }
    fingertip_object_sensors = {
        body: object_contact_sensors[body]
        for body in O6_FINGERTIP_BODIES
    }
    record_camera = scene["record_camera"] if args.record_dir else None

    _log("sim.reset()...")
    sim.reset()
    scene.update(sim_cfg.dt)
    _log("sim.reset() done.")

    actual = set(robot.joint_names)
    expected = set(R1_O6_ALL_JOINT_NAMES)
    if actual != expected:
        raise RuntimeError(
            f"Joint mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )

    name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}

    # ---- Teleport robot to cuRobo default pose (q=0 has self-collision) ----
    # cuRobo's default pose is collision-free and a good starting config
    _log("Loading cuRobo config for initial pose...")
    _kin_cfg_init = KinematicsCfg.from_robot_yaml_file(args.robot_config)
    _kin_init = Kinematics(_kin_cfg_init)
    _default_arm_q = _kin_init.default_joint_position.to(device=device).reshape(-1)
    _default_arm_names = list(_kin_init.joint_names)

    _init_pos = robot.data.default_joint_pos.clone()
    _init_vel = torch.zeros_like(_init_pos)
    for i, name in enumerate(_default_arm_names):
        _init_pos[:, name_to_idx[name]] = _default_arm_q[i]

    # Log pre-teleport state
    _obj_pos_pre = object_asset.data.root_pos_w[0].detach().cpu().tolist()
    _log(f"Pre-teleport: object_pos={[round(v,4) for v in _obj_pos_pre]}")

    robot.write_joint_state_to_sim(_init_pos, _init_vel)
    robot.set_joint_position_target(_init_pos)
    for _ in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

    # Log post-teleport state
    _obj_pos_post = object_asset.data.root_pos_w[0].detach().cpu().tolist()
    _arm_q_post = robot.data.joint_pos[0].detach().cpu()
    _lwr = float(_arm_q_post[name_to_idx["left_wrist_roll_joint"]])
    _log(f"Post-teleport: object_pos={[round(v,4) for v in _obj_pos_post]}, "
         f"left_wrist_roll={_lwr:.4f}")

    # Force-reset object to configured init position if it drifted
    from r1_o6_scene_cfg import _object_init_pos
    _expected_obj_pos = _object_init_pos(args.object)
    _obj_drift = abs(_obj_pos_post[0] - _expected_obj_pos[0])
    if _obj_drift > 0.01:
        _log(f"Object drifted {_obj_drift:.3f}m, resetting to {_expected_obj_pos}")
        _reset_pos = torch.tensor(
            [[_expected_obj_pos[0], _expected_obj_pos[1], _expected_obj_pos[2]]],
            device=device, dtype=torch.float32,
        )
        _reset_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32,
        )
        object_asset.write_root_pose_to_sim(
            torch.cat([_reset_pos, _reset_quat], dim=-1)
        )
        object_asset.write_root_velocity_to_sim(
            torch.zeros((1, 6), device=device, dtype=torch.float32)
        )
        for _ in range(5):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_cfg.dt)
        _obj_pos_final = object_asset.data.root_pos_w[0].detach().cpu().tolist()
        _log(f"Object reset: final_pos={[round(v,4) for v in _obj_pos_final]}")

    _log("Robot teleported to cuRobo default pose.")

    # ---- cuRobo setup (single-arm planners, built lazily) ----
    import yaml

    _log("cuRobo planners will be built lazily per arm.")
    # All 5 arm joints per side (for home, state, etc.)
    arm_joint_names_map = {
        "left": [n for n in ARM_JOINT_NAMES if n.startswith("left_")],
        "right": [n for n in ARM_JOINT_NAMES if n.startswith("right_")],
    }
    # 4 planner joints per side (wrist_roll locked at 0.0 in cuRobo).
    # Wrist roll is dangerous in trajectory planning — collision sphere
    # rotation causes unpredictable self-collision.  FK-based grasp search
    # (like auto_clamp) uses all 5 joints separately.
    arm_planner_joint_names_map = {
        k: [n for n in v if "wrist_roll" not in n]
        for k, v in arm_joint_names_map.items()
    }
    all_arm_joint_names = list(ARM_JOINT_NAMES)
    arm_config_map = {
        "left": args.left_config,
        "right": args.right_config,
    }
    # Cache: {(arm, orientation_tolerance): {"planner", "kin", ...}}
    _planner_cache: dict[tuple[str, float], dict[str, Any]] = {}

    # World collision model in base (pelvis) frame.
    # Only the floor is modeled. The table is NOT modeled because the arm's
    # collision spheres (forearm/upper arm) overlap the table volume when
    # reaching forward, making all targets above the table IK-infeasible.
    # Instead, the approach trajectory uses a raised intermediate waypoint
    # to keep the elbow above the table surface.
    _floor_z_base = -ROBOT_ROOT_HEIGHT  # ground level in base frame
    _table_surface_base = TABLE_SURFACE_Z - ROBOT_ROOT_HEIGHT  # for approach planning
    _world_scene = {
        "cuboid": {
            "floor": {
                "pose": [0.0, 0.0, _floor_z_base,
                         1.0, 0.0, 0.0, 0.0],
                "dims": [5.0, 5.0, 0.01],
            },
        },
    }
    _collision_cache = {"cuboid": 8}
    _log(f"World collision: floor at base z={_floor_z_base:.3f} "
         f"(table surface at base z={_table_surface_base:.3f}, "
         f"not modeled — using raised approach waypoints instead)")

    def _other_arm(arm: str) -> str:
        return "right" if arm == "left" else "left"

    def _resolve_config(name: str) -> Path:
        raw = Path(name)
        candidates = ([raw] if raw.is_absolute() else [
            Path.cwd() / raw,
            Path(__file__).resolve().parent / raw,
            Path(__file__).resolve().parent / "configs" / raw,
        ])
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(f"config not found: {name}")

    _tmp_dirs: dict[tuple[str, float], Path] = {}

    def _write_locked_config(arm: str, cache_key: tuple[str, float] | None = None) -> str:
        """Create temp YAML with inactive arm locked at current positions."""
        src = _resolve_config(arm_config_map[arm])
        with src.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        lock = data["robot_cfg"]["kinematics"].setdefault("lock_joints", {})
        q_all = robot.data.joint_pos[0]
        # Update all existing locked joints (inactive arm, waist, etc.)
        for jname in list(lock):
            if jname in name_to_idx:
                lock[jname] = float(q_all[name_to_idx[jname]].detach().cpu())
        # Lock active arm's wrist_roll at current value for 4DOF planning.
        wrist_name = f"{arm}_wrist_roll_joint"
        if wrist_name not in lock:
            lock[wrist_name] = float(q_all[name_to_idx[wrist_name]].detach().cpu())

        # Reuse or replace the temp dir for this cache key
        _ck = cache_key if cache_key is not None else (arm, DEFAULT_ORIENTATION_TOLERANCE)
        old = _tmp_dirs.get(_ck)
        if old is not None and old.exists():
            import shutil
            shutil.rmtree(old, ignore_errors=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"curobo_{arm}_"))
        _tmp_dirs[_ck] = tmp
        out = tmp / src.name
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return str(out)

    def build_planner(
        arm: str,
        orientation_tolerance: float = DEFAULT_ORIENTATION_TOLERANCE,
    ) -> dict[str, Any]:
        """Build or reuse a single-arm planner for the given orientation tolerance."""
        orientation_tolerance = round(float(orientation_tolerance), 6)
        if not math.isfinite(orientation_tolerance) or orientation_tolerance <= 0.0:
            raise ValueError("orientation_tolerance must be a positive finite value")
        inactive_q = get_arm_q(_other_arm(arm), full=True).detach().clone()
        cache_key = (arm, orientation_tolerance)
        cache = _planner_cache.setdefault(cache_key, {})
        prev_q = cache.get("inactive_q")
        # Only rebuild when inactive arm positions change significantly.
        needs_rebuild = (cache.get("planner") is None
                         or prev_q is None
                         or torch.max(torch.abs(prev_q - inactive_q)).item() > LOCK_JOINT_REBUILD_TOL)
        if not needs_rebuild:
            return cache

        cfg_path = _write_locked_config(arm, cache_key)
        kin_cfg = KinematicsCfg.from_robot_yaml_file(cfg_path)
        kin_arm = Kinematics(kin_cfg)
        pl_cfg = MotionPlannerCfg.create(
            robot=cfg_path,
            position_tolerance=0.05,
            orientation_tolerance=orientation_tolerance,
            num_ik_seeds=64,
            ik_optimizer_configs=[
                "ik/particle_ik.yml",
                "ik/lbfgs_ik.yml",
            ],
            scene_model=_world_scene,
            collision_cache=_collision_cache,
        )
        pl = MotionPlanner(pl_cfg)
        pl.warmup(enable_graph=True, num_warmup_iterations=3)

        jnames = list(pl.joint_names)
        tframes = list(pl.tool_frames)
        expected_jnames = arm_planner_joint_names_map[arm]
        if jnames != expected_jnames:
            raise RuntimeError(
                f"{arm} planner joints {jnames} != {expected_jnames}")
        if tframes != [TOOL_FRAMES[arm]]:
            raise RuntimeError(
                f"{arm} planner tools {tframes} != [{TOOL_FRAMES[arm]}]")

        cache.clear()
        cache.update({
            "planner": pl, "kin": kin_arm,
            "joint_names": jnames,
            "tool_frame": TOOL_FRAMES[arm],
            "inactive_q": inactive_q,
            "orientation_tolerance": orientation_tolerance,
        })
        _log(f"Built {arm} planner (orient_tol={orientation_tolerance:.3f}, "
             f"5DOF, "
             f"inactive {_other_arm(arm)} locked at {inactive_q.cpu().tolist()})")
        return cache

    # ---- Helper closures ----
    def get_arm_q(arm: str, full: bool = False):
        """Current joint positions for one arm.

        full=False (default): 5 planner DOF.
        full=True: all 5 DOF (same as False; kept for API compat)."""
        q_all = robot.data.joint_pos[0]
        names = arm_joint_names_map[arm] if full else arm_planner_joint_names_map[arm]
        return torch.stack(
            [q_all[name_to_idx[n]] for n in names]
        ).to(device=device)

    def compute_tool_pose(arm: str, q_arm, pinfo: dict | None = None):
        """FK for one arm -> (frame_name, Pose)."""
        if pinfo is None:
            pinfo = build_planner(arm)
        if q_arm.ndim == 1:
            q_arm = q_arm.unsqueeze(0)
        js = JointState.from_position(q_arm, joint_names=pinfo["joint_names"])
        fk = pinfo["kin"].compute_kinematics(js)
        frame = pinfo["tool_frame"]
        p = fk.tool_poses.get_link_pose(frame)
        return frame, Pose(position=p.position.clone(),
                           quaternion=p.quaternion.clone())

    def _extract_plan_q(traj, n_dof: int):
        """Extract [T, n_dof] positions from planner trajectory."""
        q = traj.position.detach()
        while q.ndim > 2:
            q = q[0]
        if q.shape[-1] > n_dof:
            q = q[:, :n_dof]
        if q.shape[-1] != n_dof:
            raise RuntimeError(
                f"trajectory DOF mismatch: got {q.shape[-1]}, expected {n_dof}")
        return q

    def serialize_pose(pose: Pose) -> dict:
        pos = pose.position.detach().reshape(-1).cpu().tolist()
        quat = pose.quaternion.detach().reshape(-1).cpu().tolist()
        return {
            "xyz": [round(v, 6) for v in pos],
            "quat_wxyz": [round(v, 6) for v in quat],
        }

    def sync_mimic():
        target = robot.data.joint_pos_target.clone()
        lim = robot.data.soft_joint_pos_limits
        for child, parent, mult, offset in MIMIC_JOINTS:
            ci, pi = name_to_idx[child], name_to_idx[parent]
            target[:, ci] = (target[:, pi] * mult + offset).clamp(
                min=lim[:, ci, 0], max=lim[:, ci, 1],
            )
        robot.set_joint_position_target(target)

    def world_to_base(xyz_w, quat_w):
        """Transform world-frame pose to robot base (pelvis) frame."""
        root_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
        root_quat = robot.data.root_quat_w[0].detach().cpu().tolist()
        inv_q = _quat_inv(root_quat)
        rel = [xyz_w[i] - root_pos[i] for i in range(3)]
        base_xyz = _quat_rotate(inv_q, rel)
        base_quat = _normalize_quat(_quat_mul(inv_q, quat_w))
        return base_xyz, base_quat

    def current_world_tool_pose(frame_name: str) -> dict:
        """Get a tool frame's current pose in world coordinates."""
        body_idx = robot.body_names.index(frame_name)
        xyz = robot.data.body_pos_w[0, body_idx].detach().cpu().tolist()
        quat = robot.data.body_quat_w[0, body_idx].detach().cpu().tolist()
        return {"xyz": xyz, "quat_wxyz": quat}

    def parse_target_pose(msg: dict, current_base_pose: dict,
                          current_world_pose: dict):
        """Parse pose from message, return (xyz, quat_wxyz, has_orientation) in base frame."""
        pose_data = msg.get("pose", {})
        if not isinstance(pose_data, dict):
            raise ValueError("pose must be an object")

        frame = str(msg.get("frame", "base")).lower()
        if frame not in ("base", "world"):
            raise ValueError(f"frame must be 'base' or 'world', got '{frame}'")

        # Use world-frame defaults if frame=world, base-frame defaults otherwise
        defaults = current_world_pose if frame == "world" else current_base_pose

        # Position: use provided or keep current (in the same frame)
        if "xyz" in pose_data:
            xyz = _as_floats(pose_data["xyz"], 3, "xyz")
        else:
            xyz = list(defaults["xyz"])

        # Orientation: quat > rpy > keep current
        has_orientation = False
        if "quat_wxyz" in pose_data:
            quat = _normalize_quat(_as_floats(pose_data["quat_wxyz"], 4, "quat_wxyz"))
            has_orientation = True
        elif "rpy" in pose_data:
            quat = _rpy_to_quat_wxyz(_as_floats(pose_data["rpy"], 3, "rpy"))
            has_orientation = True
        else:
            quat = _normalize_quat(list(defaults["quat_wxyz"]))

        # Convert to base frame if needed
        if frame == "world":
            xyz, quat = world_to_base(xyz, quat)

        return xyz, quat, has_orientation

    # ---- Build joint limits dict ----
    soft_lim = robot.data.soft_joint_pos_limits[0].cpu()
    limits_dict = {}
    for name in R1_O6_ACTIVE_DOF_NAMES:
        idx = name_to_idx[name]
        limits_dict[name] = {
            "lower": round(float(soft_lim[idx, 0]), 5),
            "upper": round(float(soft_lim[idx, 1]), 5),
        }

    # ---- TCP server ----
    inbox: queue.Queue = queue.Queue()
    server = JsonLineServer(args.host, args.port, inbox)
    server.set_hello({
        "type": "hello",
        "protocol": "r1_o6_pose_grasp.v1",
        "commands": ["move_to_pose", "set_hand", "get_state", "home",
                     "revert", "stop", "auto_grasp", "auto_clamp"],
        "arms": TOOL_FRAMES,
        "object": str(args.object),
        "auto_clamp_objects": list(AUTO_CLAMP_OBJECT_KEYS),
        "quat_convention": "wxyz",
        "arm_joints": all_arm_joint_names,
        "hand_joints": {
            "left": [f"lh_{s}" for s in HAND_ACTIVE_SUFFIXES],
            "right": [f"rh_{s}" for s in HAND_ACTIVE_SUFFIXES],
        },
        "limits": limits_dict,
    })
    server.start()

    # ---- State ----
    active_traj: ActiveTrajectory | None = None
    dual_arm_traj: DualArmTrajectory | None = None
    last_trajectory: dict[str, Any] | None = None
    pending_home: dict[str, Any] | None = None
    auto_grasp_ctx: AutoGraspContext | None = None
    auto_clamp_ctx: AutoClampContext | None = None

    # Optional RGB recording is activated only for an auto-clamp request.
    record_dir = Path(args.record_dir).expanduser() if args.record_dir else None
    if record_dir is not None:
        record_dir.mkdir(parents=True, exist_ok=True)
    record_active = False
    record_tail_frames = 0
    record_frame_index = 0
    record_frame_accumulator = 0.0
    record_frame_period = 1.0 / max(1, int(args.record_fps))
    record_sensor_path = (
        record_dir / "sensor_data.jsonl" if record_dir is not None else None
    )

    def _capture_record_frame(dt: float) -> None:
        nonlocal record_tail_frames, record_frame_index, record_frame_accumulator
        if record_camera is None or record_dir is None:
            return
        if not record_active and record_tail_frames <= 0:
            return
        record_frame_accumulator += dt
        if record_frame_accumulator + 1e-9 < record_frame_period:
            return
        record_frame_accumulator -= record_frame_period
        image = record_camera.data.output["rgb"][0].detach().cpu().numpy()
        from PIL import Image
        Image.fromarray(image[..., :3].astype("uint8"), mode="RGB").save(
            record_dir / f"frame_{record_frame_index:06d}.png"
        )
        if record_sensor_path is not None:
            hand_base_pose = {}
            for arm, frame in TOOL_FRAMES.items():
                world_pose = current_world_tool_pose(frame)
                base_xyz, base_quat = world_to_base(
                    world_pose["xyz"], world_pose["quat_wxyz"]
                )
                hand_base_pose[arm] = {
                    "xyz": [round(value, 6) for value in base_xyz],
                    "quat_wxyz": [round(value, 6) for value in base_quat],
                }
            sensor_sample = {
                "frame": record_frame_index,
                "phase": (
                    auto_clamp_ctx.phase.name
                    if auto_clamp_ctx is not None else None
                ),
                "clamp_alpha": (
                    dict(auto_clamp_ctx.clamp_alpha)
                    if auto_clamp_ctx is not None else None
                ),
                "bimanual_lift": (
                    {
                        "filtered_left_n": round(
                            auto_clamp_ctx.lift_coordinator.filtered_left_n, 5
                        ),
                        "filtered_right_n": round(
                            auto_clamp_ctx.lift_coordinator.filtered_right_n, 5
                        ),
                        "advance_filtered_left_n": round(
                            auto_clamp_ctx.lift_coordinator.advance_filtered_left_n,
                            5,
                        ),
                        "advance_filtered_right_n": round(
                            auto_clamp_ctx.lift_coordinator.advance_filtered_right_n,
                            5,
                        ),
                        "squeeze_offset_m": round(
                            auto_clamp_ctx.lift_coordinator.squeeze_offset_m, 7
                        ),
                        "center_offset_m": round(
                            auto_clamp_ctx.lift_coordinator.center_offset_m, 7
                        ),
                        "progress_m": round(
                            auto_clamp_ctx.lift_coordinator.progress_m, 6
                        ),
                        "recovery_frames": (
                            auto_clamp_ctx.lift_coordinator.recovery_frames
                        ),
                        "stable_frames": (
                            auto_clamp_ctx.lift_coordinator.stable_frames
                        ),
                        "command": dict(auto_clamp_ctx.lift_last_command),
                    }
                    if (
                        auto_clamp_ctx is not None
                        and auto_clamp_ctx.lift_coordinator is not None
                    ) else None
                ),
                "object_pose": build_object_pose(),
                "hand_base_pose": hand_base_pose,
                "contact_forces": build_contact_forces(),
            }
            with record_sensor_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sensor_sample, ensure_ascii=False) + "\n")
        record_frame_index += 1
        if not record_active:
            record_tail_frames -= 1

    def msg_id(msg: dict) -> str | None:
        raw = msg.get("id")
        return str(raw) if raw is not None else None

    def reply_error(msg: dict, code: str, message: str):
        server.publish({
            "type": "error", "id": msg_id(msg),
            "command": msg.get("type"), "code": code, "message": message,
        })

    def reply_ack(msg: dict, **extra):
        server.publish({
            "type": "ack", "id": msg_id(msg), "command": msg.get("type"), **extra,
        })

    def reply_done(msg_or_traj, **extra):
        if isinstance(msg_or_traj, ActiveTrajectory):
            server.publish({
                "type": "done", "id": msg_or_traj.request_id,
                "command": msg_or_traj.command, **extra,
            })
        else:
            server.publish({
                "type": "done", "id": msg_id(msg_or_traj),
                "command": msg_or_traj.get("type"), **extra,
            })

    # ---- Command handlers ----

    def handle_move_to_pose(msg: dict) -> bool:
        nonlocal active_traj, last_trajectory
        internal = bool(msg.get("_internal"))
        quiet = bool(msg.get("_quiet"))

        def _fail(code: str, message: str) -> bool:
            if not quiet:
                reply_error(msg, code, message)
            return False

        if active_traj is not None or dual_arm_traj is not None:
            return _fail("busy", "trajectory in progress, send 'stop' first")
        if (auto_grasp_ctx is not None or auto_clamp_ctx is not None) and not internal:
            return _fail("busy", "automatic motion in progress, send 'stop' first")

        arm = str(msg.get("arm", "")).lower()
        if arm not in TOOL_FRAMES:
            return _fail("bad_request", "arm must be 'left' or 'right'")

        if not quiet:
            reply_ack(msg, status="planning")
        tool_frame = TOOL_FRAMES[arm]

        try:
            try:
                orient_tol = float(msg.get("orientation_tolerance",
                                           DEFAULT_ORIENTATION_TOLERANCE))
            except (TypeError, ValueError):
                return _fail("bad_request",
                             "orientation_tolerance must be numeric")
            if not math.isfinite(orient_tol) or orient_tol <= 0.0:
                return _fail("bad_request",
                             "orientation_tolerance must be positive finite")
            pinfo = build_planner(arm, orientation_tolerance=orient_tol)
            jnames = pinfo["joint_names"]

            start_q = get_arm_q(arm)
            frame, cur_pose = compute_tool_pose(arm, start_q, pinfo)
            current_base = serialize_pose(cur_pose)
            current_world = current_world_tool_pose(tool_frame)

            target_xyz, target_quat, has_orientation = parse_target_pose(
                msg, current_base, current_world
            )

            goal_pose = Pose(
                position=torch.tensor([target_xyz], device=device, dtype=torch.float32),
                quaternion=torch.tensor([target_quat], device=device, dtype=torch.float32),
            )
            goal = GoalToolPose.from_poses(
                {tool_frame: goal_pose},
                ordered_tool_frames=[tool_frame],
                num_goalset=1,
            )
            start_js = JointState.from_position(
                start_q.unsqueeze(0), joint_names=jnames,
            )
            max_att = int(msg.get("max_attempts", args.plan_attempts))
            _log(f"Planning: arm={arm}, target_xyz={target_xyz}, "
                 f"has_orientation={has_orientation}, "
                 f"start_q={start_q.cpu().tolist()}")
            result = pinfo["planner"].plan_pose(goal, start_js, max_attempts=max_att)

            ok = (result is not None
                  and hasattr(result, "success")
                  and result.success.detach().any().item())
            if not ok:
                diag = "None (IK failed)" if result is None else "success=False"
                if result is not None and hasattr(result, "position_error"):
                    diag += f" pos_err={result.position_error.min().item():.4f}"
                _log(f"Planning FAILED: {diag}")
                return _fail("planning_failed",
                             f"cuRobo plan_pose failed ({diag})")

            traj = result.get_interpolated_plan()
            if traj is None or not hasattr(traj, "position"):
                return _fail("planning_failed", "no interpolated trajectory")

            q_arm = _extract_plan_q(traj, len(jnames))

            # Compute achieved end-effector position
            q_final = q_arm[-1:].to(device=device)
            _, final_pose = compute_tool_pose(arm, q_final, pinfo)
            achieved = serialize_pose(final_pose)

            cmd_label = "auto_grasp" if internal else "move_to_pose"
            active_traj = ActiveTrajectory(
                request_id=msg_id(msg),
                command=cmd_label,
                arm=arm,
                joint_names=list(jnames),
                q_waypoints=q_arm.to(device=device),
                steps_per_wp=max(1, int(msg.get("steps_per_waypoint",
                                                 args.steps_per_waypoint))),
                settle_steps=args.settle_steps,
                suppress_done=quiet,
            )
            if not internal:
                last_trajectory = {
                    "arm": arm,
                    "joint_names": list(jnames),
                    "start_q": start_q.detach().clone(),
                    "q_waypoints": active_traj.q_waypoints.detach().clone(),
                }
            _log(f"Planning OK: {active_traj.num_waypoints} waypoints, "
                 f"achieved={achieved['xyz']}")
            if not quiet:
                reply_ack(msg, status="executing",
                          num_waypoints=active_traj.num_waypoints,
                          target={"arm": arm, "xyz": target_xyz, "quat_wxyz": target_quat},
                          achieved=achieved)
            return True

        except Exception as exc:
            return _fail("exception", str(exc))

    def handle_set_hand(msg: dict) -> bool:
        quiet = bool(msg.get("_quiet"))
        internal = bool(msg.get("_internal"))
        if (not internal and (
                active_traj is not None or dual_arm_traj is not None
                or auto_grasp_ctx is not None or auto_clamp_ctx is not None)):
            if not quiet:
                reply_error(msg, "busy", "automatic motion in progress, send 'stop' first")
            return False
        hand = str(msg.get("hand", "both")).lower()
        if hand not in ("left", "right", "both"):
            if not quiet:
                reply_error(msg, "bad_request", "hand must be 'left', 'right', or 'both'")
            return False

        target = robot.data.joint_pos_target.clone()
        count = 0

        if isinstance(msg.get("joints"), dict):
            # Named joints
            for raw_name, raw_val in msg["joints"].items():
                name = str(raw_name)
                if name not in name_to_idx:
                    # Try adding prefix
                    for prefix in ("lh", "rh"):
                        candidate = f"{prefix}_{name}"
                        if candidate in name_to_idx:
                            name = candidate
                            break
                if name in set(HAND_ACTIVE_JOINT_NAMES) and name in name_to_idx:
                    lo = limits_dict[name]["lower"]
                    hi = limits_dict[name]["upper"]
                    target[:, name_to_idx[name]] = max(lo, min(hi, float(raw_val)))
                    count += 1

        elif isinstance(msg.get("values"), (list, tuple)):
            try:
                values = [float(v) for v in msg["values"]]
            except (TypeError, ValueError):
                if not quiet:
                    reply_error(msg, "bad_request", "values must be numeric")
                return False
            if not all(math.isfinite(v) for v in values):
                if not quiet:
                    reply_error(msg, "bad_request", "values contain non-finite value")
                return False
            sides = [hand] if hand != "both" else ["left", "right"]
            if hand == "both" and len(values) == 12:
                # left 6 + right 6
                for i, suffix in enumerate(HAND_ACTIVE_SUFFIXES):
                    for si, side in enumerate(("left", "right")):
                        name = f"{'lh' if side == 'left' else 'rh'}_{suffix}"
                        val = values[si * 6 + i]
                        lo = limits_dict[name]["lower"]
                        hi = limits_dict[name]["upper"]
                        target[:, name_to_idx[name]] = max(lo, min(hi, val))
                        count += 1
            elif len(values) == 6:
                for side in sides:
                    prefix = "lh" if side == "left" else "rh"
                    for i, suffix in enumerate(HAND_ACTIVE_SUFFIXES):
                        name = f"{prefix}_{suffix}"
                        val = values[i]
                        lo = limits_dict[name]["lower"]
                        hi = limits_dict[name]["upper"]
                        target[:, name_to_idx[name]] = max(lo, min(hi, val))
                        count += 1
            else:
                if not quiet:
                    reply_error(msg, "bad_request",
                                f"values: expected 6 (one hand) or 12 (both), got {len(values)}")
                return False
        else:
            if not quiet:
                reply_error(msg, "bad_request", "set_hand requires 'joints' dict or 'values' list")
            return False

        robot.set_joint_position_target(target)
        if not quiet:
            reply_done(msg, updated=count)
        return True

    def _plan_home_arm(msg: dict, arm: str):
        """Plan one arm back to default. Returns [T, 5] tensor or None if already home."""
        pinfo = build_planner(arm)
        jnames = pinfo["joint_names"]
        start_q = get_arm_q(arm)
        goal_q = pinfo["kin"].default_joint_position.to(device=device).reshape(-1)

        if torch.max(torch.abs(start_q - goal_q)).item() < 0.02:
            return None

        frame, goal_pose = compute_tool_pose(arm, goal_q, pinfo)
        goal = GoalToolPose.from_poses(
            {frame: goal_pose},
            ordered_tool_frames=[frame],
            num_goalset=1,
        )
        start_js = JointState.from_position(
            start_q.unsqueeze(0), joint_names=jnames,
        )
        result = pinfo["planner"].plan_pose(
            goal, start_js,
            max_attempts=int(msg.get("max_attempts", args.plan_attempts)),
        )
        ok = (result is not None
              and hasattr(result, "success")
              and result.success.detach().any().item())
        if ok:
            traj = result.get_interpolated_plan()
            if traj is not None and hasattr(traj, "position"):
                return _extract_plan_q(traj, len(jnames))

        _log(f"Home plan_pose failed for {arm}, using linear fallback")
        steps = max(2, int(msg.get("steps", 80)))
        alpha = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
        return start_q.unsqueeze(0) + alpha * (goal_q - start_q).unsqueeze(0)

    def _start_next_home_segment(hstate: dict):
        """Start the next arm segment of a home command."""
        nonlocal active_traj, pending_home

        msg = hstate["msg"]
        while hstate["remaining"]:
            arm = hstate["remaining"].pop(0)
            q_arm = _plan_home_arm(msg, arm)
            if q_arm is None:
                hstate["done_arms"].append(arm)
                continue

            active_traj = ActiveTrajectory(
                request_id=msg_id(msg), command="home",
                arm=arm,
                joint_names=list(arm_planner_joint_names_map[arm]),
                q_waypoints=q_arm.to(device=device),
                steps_per_wp=hstate["steps_per_wp"],
                settle_steps=args.settle_steps,
            )
            hstate["total_wp"] += active_traj.num_waypoints
            pending_home = hstate
            reply_ack(msg, status="executing", arm=arm,
                      num_waypoints=active_traj.num_waypoints)
            return

        # All arms done
        elapsed = time.monotonic() - hstate["t_start"]
        status = "already_home" if hstate["total_wp"] == 0 else "complete"
        pending_home = None
        reply_done(msg, status=status, arms=hstate["done_arms"],
                   num_waypoints=hstate["total_wp"],
                   elapsed_sec=round(elapsed, 3))

    def handle_home(msg: dict):
        nonlocal active_traj, last_trajectory, pending_home
        if active_traj is not None or dual_arm_traj is not None:
            reply_error(msg, "busy", "trajectory in progress")
            return
        if auto_grasp_ctx is not None or auto_clamp_ctx is not None:
            reply_error(msg, "busy", "automatic motion in progress, send 'stop' first")
            return

        reply_ack(msg, status="planning")

        try:
            # Zero all hand joints
            target = robot.data.joint_pos_target.clone()
            for name in HAND_ACTIVE_JOINT_NAMES:
                target[:, name_to_idx[name]] = 0.0
            robot.set_joint_position_target(target)

            last_trajectory = None
            _start_next_home_segment({
                "msg": msg,
                "remaining": ["left", "right"],
                "done_arms": [],
                "total_wp": 0,
                "steps_per_wp": max(1, int(msg.get("steps_per_waypoint",
                                                    args.steps_per_waypoint))),
                "t_start": time.monotonic(),
            })

        except Exception as exc:
            pending_home = None
            reply_error(msg, "exception", str(exc))

    def handle_revert(msg: dict):
        nonlocal active_traj, last_trajectory
        if active_traj is not None or dual_arm_traj is not None:
            reply_error(msg, "busy", "trajectory in progress, send 'stop' first")
            return
        if auto_grasp_ctx is not None or auto_clamp_ctx is not None:
            reply_error(msg, "busy", "automatic motion in progress, send 'stop' first")
            return
        if last_trajectory is None:
            reply_error(msg, "no_trajectory", "no trajectory available to revert")
            return

        try:
            arm = last_trajectory["arm"]
            jnames = last_trajectory["joint_names"]
            q_waypoints = last_trajectory["q_waypoints"].to(device=device)
            start_q = last_trajectory["start_q"].to(device=device)

            # Reverse the waypoints
            revert_waypoints = q_waypoints.flip(0).clone()

            # Ensure last waypoint matches the original start_q exactly
            if torch.max(torch.abs(revert_waypoints[-1] - start_q)).item() > 1e-6:
                revert_waypoints = torch.cat(
                    [revert_waypoints, start_q.unsqueeze(0)], dim=0,
                )

            active_traj = ActiveTrajectory(
                request_id=msg_id(msg),
                command="revert",
                arm=arm,
                joint_names=list(jnames),
                q_waypoints=revert_waypoints,
                steps_per_wp=max(1, int(msg.get("steps_per_waypoint",
                                                 args.steps_per_waypoint))),
                settle_steps=args.settle_steps,
            )
            last_trajectory = None
            _log(f"Reverting last trajectory: "
                 f"{active_traj.num_waypoints} waypoints")
            reply_ack(msg, status="executing",
                      num_waypoints=active_traj.num_waypoints)

        except Exception as exc:
            reply_error(msg, "exception", str(exc))

    def handle_stop(msg: dict):
        nonlocal active_traj, dual_arm_traj, last_trajectory, pending_home
        nonlocal auto_grasp_ctx, auto_clamp_ctx
        grasp_request_id = (
            auto_grasp_ctx.request_id if auto_grasp_ctx is not None else None
        )
        clamp_request_id = (
            auto_clamp_ctx.request_id if auto_clamp_ctx is not None else None
        )
        if active_traj is not None:
            _log(f"Stopping trajectory: {active_traj.command} arm={active_traj.arm}")
        if dual_arm_traj is not None:
            _log(f"Stopping dual trajectory: {dual_arm_traj.command}")
        if (active_traj is not None or dual_arm_traj is not None
                or auto_grasp_ctx is not None or auto_clamp_ctx is not None):
            _hold_all_arm_joints()
        active_traj = None
        dual_arm_traj = None
        pending_home = None
        last_trajectory = None
        if auto_grasp_ctx is not None:
            _log("Auto-grasp cancelled by stop")
            auto_grasp_ctx = None
            if grasp_request_id:
                server.publish({
                    "type": "error", "id": grasp_request_id,
                    "command": "auto_grasp", "code": "cancelled",
                    "message": "auto_grasp cancelled by stop",
                })
        if auto_clamp_ctx is not None:
            _log("Auto-clamp cancelled by stop")
            auto_clamp_ctx = None
            if clamp_request_id:
                server.publish({
                    "type": "error", "id": clamp_request_id,
                    "command": "auto_clamp", "code": "cancelled",
                    "message": "auto_clamp cancelled by stop",
                })
        reply_done(msg, status="stopped")

    def handle_get_state(msg: dict):
        server.publish(build_state(reply_to=msg_id(msg)))

    # ---- Trajectory execution (called each sim step) ----

    def step_trajectory():
        nonlocal active_traj, pending_home
        if active_traj is None:
            return

        # Always command the current (or final) waypoint
        q = active_traj.q_waypoints[
            min(active_traj.wp_idx, active_traj.num_waypoints - 1)
        ]
        target = robot.data.joint_pos_target.clone()
        q_actual = robot.data.joint_pos[0]
        for i, name in enumerate(active_traj.joint_names):
            target[:, name_to_idx[name]] = q[i]
        # Wrist roll: if it's a planned joint (in joint_names), the loop above
        # already set it from the trajectory. Otherwise, float to actual.
        arm = active_traj.arm
        wrist_name = f"{'left' if arm == 'left' else 'right'}_wrist_roll_joint"
        if wrist_name in name_to_idx and wrist_name not in active_traj.joint_names:
            target[:, name_to_idx[wrist_name]] = q_actual[name_to_idx[wrist_name]]
        robot.set_joint_position_target(target)

        # Settle phase: keep commanding final waypoint for extra steps
        if active_traj.settling:
            active_traj.settle_count += 1
            return

        active_traj.hold_count += 1
        if active_traj.hold_count < active_traj.steps_per_wp:
            return
        active_traj.hold_count = 0
        active_traj.wp_idx += 1

        if active_traj.done:
            elapsed = time.monotonic() - active_traj.t_start
            done_traj = active_traj
            _log(f"{done_traj.command} segment done: arm={done_traj.arm}, "
                 f"{done_traj.num_waypoints} wp, {elapsed:.2f}s")

            # --- FK diagnostic: planned vs actual ---
            try:
                _diag_arm = done_traj.arm
                _diag_planned_q = done_traj.q_waypoints[-1]
                _diag_actual_q = get_arm_q(_diag_arm)
                _diag_q_err = (_diag_planned_q - _diag_actual_q).abs()
                _log(f"FK-DIAG planned_q: {_diag_planned_q.cpu().tolist()}")
                _log(f"FK-DIAG actual_q:  {_diag_actual_q.cpu().tolist()}")
                _log(f"FK-DIAG q_error:   {_diag_q_err.cpu().tolist()} "
                     f"max={_diag_q_err.max().item():.4f}")
                # cuRobo FK for planned joints
                _diag_pinfo = build_planner(_diag_arm)
                _, _diag_fk_planned = compute_tool_pose(
                    _diag_arm, _diag_planned_q, _diag_pinfo)
                _diag_fk_p_xyz = _diag_fk_planned.position.reshape(-1).cpu().tolist()
                # cuRobo FK for actual joints
                _, _diag_fk_actual = compute_tool_pose(
                    _diag_arm, _diag_actual_q, _diag_pinfo)
                _diag_fk_a_xyz = _diag_fk_actual.position.reshape(-1).cpu().tolist()
                # Isaac Lab actual body position
                _diag_tool = TOOL_FRAMES[_diag_arm]
                _diag_world = current_world_tool_pose(_diag_tool)
                _diag_base, _ = world_to_base(
                    _diag_world["xyz"], _diag_world["quat_wxyz"])
                _log(f"FK-DIAG curobo_fk(planned): {[round(v,4) for v in _diag_fk_p_xyz]}")
                _log(f"FK-DIAG curobo_fk(actual):  {[round(v,4) for v in _diag_fk_a_xyz]}")
                _log(f"FK-DIAG isaac_body(world):   {[round(v,4) for v in _diag_world['xyz']]}")
                _log(f"FK-DIAG isaac_body(base):    {[round(v,4) for v in _diag_base]}")
                _fk_vs_isaac = [abs(_diag_fk_a_xyz[i] - _diag_base[i]) for i in range(3)]
                _log(f"FK-DIAG fk_vs_isaac_delta:   {[round(v,4) for v in _fk_vs_isaac]}")
            except Exception as _diag_exc:
                _log(f"FK-DIAG error: {_diag_exc}")

            # Home command: advance to next arm segment.
            # Flush the completed arm's final target into sim so that the
            # next planner locks the correct position (not stale sim data).
            if done_traj.command == "home" and pending_home is not None:
                pending_home["done_arms"].append(done_traj.arm)
                # Force sim to see the final waypoint before building next planner
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_cfg.dt)
                active_traj = None
                try:
                    _start_next_home_segment(pending_home)
                except Exception as exc:
                    reply_error({"id": done_traj.request_id, "type": "home"},
                                "exception", str(exc))
                    pending_home = None
                return

            active_traj = None
            if not done_traj.suppress_done:
                reply_done(done_traj,
                           status="complete",
                           num_waypoints=done_traj.num_waypoints,
                           elapsed_sec=round(elapsed, 3))

    def _start_dual_linear(
        ctx: AutoClampContext,
        command: str,
        goals_by_arm: dict[str, Any],
        steps: int,
        settle_steps: int = 20,
        starts_by_arm: dict[str, Any] | None = None,
    ) -> None:
        """Start a synchronized 5DOF-per-arm joint interpolation."""
        nonlocal dual_arm_traj
        if active_traj is not None or dual_arm_traj is not None:
            raise RuntimeError("cannot start dual trajectory while another trajectory is active")

        q_start = torch.stack([
            robot.data.joint_pos[0, name_to_idx[name]]
            for name in all_arm_joint_names
        ]).to(device=device)
        if starts_by_arm is not None:
            for arm, q_arm_raw in starts_by_arm.items():
                q_arm = q_arm_raw.to(device=device).reshape(-1)
                names = arm_joint_names_map[arm]
                if q_arm.numel() != len(names):
                    raise RuntimeError(
                        f"{arm} dual start has {q_arm.numel()} joints, "
                        f"expected {len(names)}"
                    )
                for i, name in enumerate(names):
                    q_start[all_arm_joint_names.index(name)] = q_arm[i]
        q_goal = q_start.clone()
        for arm, q_arm_raw in goals_by_arm.items():
            q_arm = q_arm_raw.to(device=device).reshape(-1)
            names = arm_joint_names_map[arm]
            if q_arm.numel() != len(names):
                raise RuntimeError(
                    f"{arm} dual goal has {q_arm.numel()} joints, expected {len(names)}"
                )
            for i, name in enumerate(names):
                q_goal[all_arm_joint_names.index(name)] = q_arm[i]

        alpha = torch.linspace(0.0, 1.0, max(2, int(steps)), device=device).unsqueeze(1)
        q_waypoints = q_start.unsqueeze(0) + alpha * (q_goal - q_start).unsqueeze(0)
        dual_arm_traj = DualArmTrajectory(
            request_id=ctx.request_id,
            command=command,
            joint_names=list(all_arm_joint_names),
            q_waypoints=q_waypoints,
            steps_per_wp=1,
            settle_steps=max(0, int(settle_steps)),
        )
        _log(f"Auto-clamp: started {command}, {dual_arm_traj.num_waypoints} waypoints")

    def _staged_pregrasp_arm_path(arm: str, q_start, q_end):
        """Keep shoulder-roll raised until the other four joints are in place."""
        roll_idx = arm_joint_names_map[arm].index(f"{arm}_shoulder_roll_joint")
        q_mid = q_end.clone()
        q_mid[roll_idx] = q_start[roll_idx]
        alpha_first = torch.linspace(0.0, 1.0, 141, device=device).unsqueeze(1)
        alpha_second = torch.linspace(0.0, 1.0, 121, device=device).unsqueeze(1)
        first = q_start.unsqueeze(0) + alpha_first * (q_mid - q_start).unsqueeze(0)
        second = q_mid.unsqueeze(0) + alpha_second * (q_end - q_mid).unsqueeze(0)
        return torch.cat([first, second[1:]], dim=0)

    def _start_dual_staged_pregrasp(
        ctx: AutoClampContext, goals_by_arm: dict[str, Any], settle_steps: int = 30,
    ) -> None:
        """Start the synchronized roll-last route from side-raise to pregrasp."""
        nonlocal dual_arm_traj
        if active_traj is not None or dual_arm_traj is not None:
            raise RuntimeError("cannot start staged pregrasp while trajectory is active")
        arm_paths = {}
        for arm in ("left", "right"):
            q_start = get_arm_q(arm, full=True).detach().clone()
            q_end = goals_by_arm[arm].to(device=device).reshape(-1)
            arm_paths[arm] = _staged_pregrasp_arm_path(arm, q_start, q_end)

        q_waypoints = torch.empty(
            (arm_paths["left"].shape[0], len(all_arm_joint_names)),
            device=device, dtype=arm_paths["left"].dtype,
        )
        for arm in ("left", "right"):
            for i, name in enumerate(arm_joint_names_map[arm]):
                q_waypoints[:, all_arm_joint_names.index(name)] = arm_paths[arm][:, i]
        dual_arm_traj = DualArmTrajectory(
            request_id=ctx.request_id,
            command="auto_clamp_pregrasp",
            joint_names=list(all_arm_joint_names),
            q_waypoints=q_waypoints,
            steps_per_wp=1,
            settle_steps=max(0, int(settle_steps)),
        )
        _log(
            f"Auto-clamp: started staged pregrasp, "
            f"{dual_arm_traj.num_waypoints} waypoints (roll held, then lowered)"
        )

    def step_dual_trajectory() -> None:
        """Advance the current two-arm trajectory by one simulation tick."""
        nonlocal dual_arm_traj
        traj = dual_arm_traj
        if traj is None:
            return

        q = traj.q_waypoints[min(traj.wp_idx, traj.num_waypoints - 1)]
        target = robot.data.joint_pos_target.clone()
        for i, name in enumerate(traj.joint_names):
            target[:, name_to_idx[name]] = q[i]
        robot.set_joint_position_target(target)

        if traj.settling:
            traj.settle_count += 1
        else:
            traj.hold_count += 1
            if traj.hold_count >= traj.steps_per_wp:
                traj.hold_count = 0
                traj.wp_idx += 1

        if traj.done:
            elapsed = time.monotonic() - traj.t_start
            _log(f"Auto-clamp: {traj.command} done in {elapsed:.2f}s")
            dual_arm_traj = None

    # ---- Object & contact data ----

    def build_object_pose() -> dict:
        xyz = object_asset.data.root_pos_w[0].detach().cpu().tolist()
        quat = object_asset.data.root_quat_w[0].detach().cpu().tolist()
        return {
            "xyz": [round(v, 5) for v in xyz],
            "quat_wxyz": [round(v, 5) for v in quat],
        }

    def _filtered_force_vector(sensor):
        """Return one object-filtered sensor's net world-frame force vector."""
        fm = getattr(sensor.data, "force_matrix_w", None)
        if fm is None or fm.numel() == 0:
            return torch.zeros(3, device=device, dtype=torch.float32)
        return fm[0].reshape(-1, 3).sum(dim=0)

    def _auto_clamp_forces() -> dict[str, Any]:
        """Per-hand object-filtered tactile forces for auto-clamp control."""
        result: dict[str, Any] = {}
        broad_vectors = contact_hands.data.net_forces_w[0].detach()
        for arm, prefix in (("left", "lh"), ("right", "rh")):
            side_bodies = [
                body for body in O6_ALL_TACTILE_BODIES
                if body.startswith(prefix + "_")
            ]
            vectors = {
                body: _filtered_force_vector(object_contact_sensors[body])
                for body in side_bodies
            }
            raw_axis_y = {
                body: float(vector[1].detach().cpu().item())
                for body, vector in vectors.items()
            }
            structural_flag = (
                auto_clamp_ctx is not None
                and auto_clamp_ctx.config.thumb_is_structural
            )
            normal = aggregate_hand_normal_force(
                raw_axis_y, arm,
                thumb_is_structural=structural_flag,
            )
            per_body_axis_y_n = normal["per_body_n"]
            per_body_norm_n = {
                body: float(torch.linalg.norm(vector).detach().cpu().item())
                for body, vector in vectors.items()
            }
            palm_body = f"{prefix}_hand_base_link"
            fingertip_by_name = {
                finger: per_body_norm_n[f"{prefix}_{finger}_distal"]
                for finger in FINGER_NAMES
            }
            four_finger_distal_axis_y_n = sum(
                per_body_axis_y_n[f"{prefix}_{finger}_distal"]
                for finger in FOUR_FINGER_NAMES
            )
            broad_non_thumb_by_name = {}
            for i, body in enumerate(contact_hands.body_names):
                if not body.startswith(prefix + "_") or "thumb" in body:
                    continue
                axis_n = float(torch.abs(broad_vectors[i, 1]).cpu().item())
                broad_non_thumb_by_name[body] = axis_n
            broad_non_thumb_axis_y_n = float(
                sum(broad_non_thumb_by_name.values())
            )
            result[arm] = {
                "palm_n": per_body_norm_n[palm_body],
                "palm_axis_y_n": normal["palm_n"],
                "fingertip_sum_n": float(sum(
                    fingertip_by_name[finger] for finger in FOUR_FINGER_NAMES
                )),
                "fingertip_axis_y_n": four_finger_distal_axis_y_n,
                "fingertip_by_name_n": fingertip_by_name,
                "thumb_axis_y_n": normal["thumb_n"],
                "non_thumb_axis_y_n": normal["non_thumb_n"],
                "object_contact_body_count": normal["contact_body_count"],
                "object_by_body_axis_y_n": per_body_axis_y_n,
                "object_by_body_norm_n": per_body_norm_n,
                "broad_non_thumb_axis_y_n": broad_non_thumb_axis_y_n,
                "broad_non_thumb_by_name_n": broad_non_thumb_by_name,
                # All Object-filtered structural contacts, including passive
                # thumb load, drive contact detection and the per-side profile
                # controller. The broad sensor remains diagnostic/safety-only.
                "object_filtered_axis_y_n": normal["total_n"],
                "object_axis_y_n": normal["total_n"],
            }
        return result

    def _store_auto_clamp_forces(
        ctx: AutoClampContext, forces: dict[str, Any],
    ) -> None:
        """Copy live tactile values into JSON-safe state/recording fields."""

        ctx.last_forces = {
            arm: {
                key: (
                    {name: round(force, 4) for name, force in value.items()}
                    if isinstance(value, dict) else round(value, 4)
                )
                for key, value in values.items()
            }
            for arm, values in forces.items()
        }

    def build_contact_forces() -> dict:
        # Broad sensor summary: how many bodies in contact + max force
        forces = contact_hands.data.net_forces_w[0].detach()
        force_norms = torch.linalg.norm(forces, dim=-1)
        contact_count = int((force_norms > CONTACT_FORCE_EPS_N).sum().cpu().item())
        max_force = round(float(force_norms.max().cpu().item()), 4)
        broad_by_body = {
            body: {
                "fx": round(float(forces[i, 0].cpu().item()), 4),
                "fy": round(float(forces[i, 1].cpu().item()), 4),
                "fz": round(float(forces[i, 2].cpu().item()), 4),
                "norm": round(float(force_norms[i].cpu().item()), 4),
            }
            for i, body in enumerate(contact_hands.body_names)
        }

        # Per-fingertip filtered forces against object
        fingertip_forces = {}
        fingertip_raw = {}
        for body, sensor in fingertip_object_sensors.items():
            fm = getattr(sensor.data, "force_matrix_w", None)
            if fm is not None and fm.numel() > 0:
                vectors = fm[0].reshape(-1, 3)
                summed = vectors.sum(dim=0)
                val = float(torch.linalg.norm(vectors, dim=-1).max().cpu().item())
                fingertip_raw[body] = {
                    "fx": round(float(summed[0].cpu().item()), 4),
                    "fy": round(float(summed[1].cpu().item()), 4),
                    "fz": round(float(summed[2].cpu().item()), 4),
                    "norm": round(float(torch.linalg.norm(summed).cpu().item()), 4),
                    "matrix_contacts": int(vectors.shape[0]),
                }
            else:
                val = 0.0
                fingertip_raw[body] = {
                    "fx": 0.0, "fy": 0.0, "fz": 0.0,
                    "norm": 0.0, "matrix_contacts": 0,
                }
            fingertip_forces[body] = round(val, 4)

        clamp_forces = _auto_clamp_forces()
        palm_raw = {}
        for arm, sensor in palm_object_sensors.items():
            vec = _filtered_force_vector(sensor)
            palm_raw[arm] = {
                "fx": round(float(vec[0].cpu().item()), 4),
                "fy": round(float(vec[1].cpu().item()), 4),
                "fz": round(float(vec[2].cpu().item()), 4),
                "norm": round(float(torch.linalg.norm(vec).cpu().item()), 4),
            }

        return {
            "contact_hands": {
                "contact_body_count": contact_count,
                "max_force_n": max_force,
                "by_body_w": broad_by_body,
            },
            "fingertips_object_n": fingertip_forces,
            "fingertips_object_raw_w": fingertip_raw,
            "palms_object_raw_w": palm_raw,
            "palms_object_n": {
                arm: round(values["palm_n"], 4)
                for arm, values in clamp_forces.items()
            },
            "auto_clamp_hands": {
                arm: {
                    key: (
                        {name: round(force, 4) for name, force in value.items()}
                        if isinstance(value, dict) else round(value, 4)
                    )
                    for key, value in values.items()
                }
                for arm, values in clamp_forces.items()
            },
        }

    # ---- Auto-grasp state machine (Phase 2: force-guided) ----

    def _to_base_xyz(world_xyz: list[float]) -> list[float]:
        """Convert world XYZ to base frame (discard orientation)."""
        base_xyz, _ = world_to_base(world_xyz, [1.0, 0.0, 0.0, 0.0])
        return base_xyz

    def _build_approach_candidates(
        arm: str, profile: GraspProfile,
    ) -> list[dict[str, Any]]:
        """Generate 6 approach/descend/lift candidates (3 x-samples × 2 z-rounds)."""
        obj = build_object_pose()
        ox, oy, oz = obj["xyz"]
        obj_top = oz + profile.half_height
        sign = 1.0 if arm == "left" else -1.0
        x_deltas = (0.0, profile.approach_x_delta, -profile.approach_x_delta)
        z_deltas = (0.0, profile.approach_z_retry)
        # Convert profile orientation to base frame (if specified).
        base_quat = None
        if profile.grasp_quat_wxyz is not None:
            world_q = list(profile.grasp_quat_wxyz)
            # Mirror Y for right arm: negate qx and qz to flip palm direction.
            if arm == "right":
                world_q = [world_q[0], -world_q[1], world_q[2], -world_q[3]]
            _, base_quat = world_to_base([0, 0, 0], world_q)

        candidates: list[dict[str, Any]] = []
        for z_round, z_d in enumerate(z_deltas):
            for x_sample, x_d in enumerate(x_deltas):
                x_back = max(0.01, profile.x_backoff + x_d)
                grasp_z = oz + profile.grasp_z_offset + z_d
                x_target = ox - x_back
                # Pre-grasp uses pregrasp_y_offset (further from object),
                # grasp uses y_offset (closer to object surface).
                pregrasp_y = profile.pregrasp_y_offset or profile.y_offset
                y_pregrasp = oy + sign * pregrasp_y
                y_grasp = oy + sign * profile.y_offset
                approach_w = [x_target, y_pregrasp,
                              grasp_z + profile.pre_z_clearance]
                descend_w = [x_target, y_pregrasp, grasp_z]
                grasp_w = [x_target, y_grasp, grasp_z]
                lift_w = [x_target, y_grasp, grasp_z + profile.lift_z]
                # Raised waypoint: close to the body, above table.
                # Must use arm's natural lateral y (±0.191) to avoid
                # pelvis self-collision at small y_target values.
                # x=0.10 keeps within reliable workspace, z=+6cm above table.
                raised_x_world = 0.10
                raised_z_world = TABLE_SURFACE_Z + 0.06
                raised_y_world = sign * 0.19
                raised_w = [raised_x_world, raised_y_world, raised_z_world]
                candidates.append({
                    "index": len(candidates),
                    "round": z_round + 1,
                    "sample": x_sample + 1,
                    "x_backoff": round(x_back, 5),
                    "x_delta": round(x_d, 5),
                    "z_delta": round(z_d, 5),
                    "raised": _to_base_xyz(raised_w),
                    "approach": _to_base_xyz(approach_w),
                    "descend": _to_base_xyz(descend_w),
                    "grasp": _to_base_xyz(grasp_w),
                    "lift": _to_base_xyz(lift_w),
                    "quat_wxyz": base_quat,
                })
        return candidates[:6]

    def _candidate_summary(c: dict[str, Any] | None) -> dict | None:
        if c is None:
            return None
        return {k: c[k] for k in ("index", "round", "sample",
                                   "x_backoff", "x_delta", "z_delta",
                                   "raised", "approach", "descend",
                                   "grasp", "lift")}

    def _start_auto_move(
        arm: str, xyz: list[float],
        quat_wxyz: tuple | list | None = None,
        orientation_tolerance: float | None = None,
    ) -> bool:
        """Issue an internal move_to_pose for auto-grasp."""
        pose: dict[str, Any] = {"xyz": xyz}
        if quat_wxyz is not None:
            pose["quat_wxyz"] = list(quat_wxyz)
        msg: dict[str, Any] = {
            "type": "move_to_pose",
            "arm": arm,
            "pose": pose,
            "_internal": True,
            "_quiet": True,
            # Orientation-constrained trajectories need more time per waypoint
            # for the PD controller to track aggressive joint changes.
            "steps_per_waypoint": 8 if quat_wxyz is not None else args.steps_per_waypoint,
        }
        if orientation_tolerance is not None:
            msg["orientation_tolerance"] = orientation_tolerance
        return handle_move_to_pose(msg)

    def _start_fk_linear_move(
        arm: str, base_xyz: list[float],
        min_palm_inward: float = -1.0,
        min_finger_down: float = -1.0,
        label: str = "fk_move",
        lock_wrist_roll: bool = False,
        max_wrist_delta: float = math.pi / 2,
    ) -> bool:
        """FK-based IK search + joint-space linear interpolation.

        Like ``_solve_clamp_pose`` but for single-arm auto-grasp.  Searches
        5-DOF joint space (including wrist_roll) for a configuration that
        reaches ``base_xyz`` while satisfying optional palm orientation
        constraints, then linearly interpolates in joint space.

        Args:
            min_palm_inward: minimum dot product of palm +X onto the inward
                direction (-Y for left, +Y for right).  Set >=0 to require
                palm facing the object.
            min_finger_down: minimum dot product of local +Z onto world -Z.
                Set >=0 to require fingers pointing downward.
            lock_wrist_roll: if True, keep wrist_roll at its current value
                during both search and interpolation.
            max_wrist_delta: maximum wrist_roll change from current value
                (default π/2 = 90°). Bounds are intersected with joint limits.
        """
        nonlocal active_traj
        if active_traj is not None or dual_arm_traj is not None:
            return False
        try:
            info = _clamp_kinematics(arm)
            seed_q = get_arm_q(arm, full=True).detach().to(
                device=device, dtype=torch.float32).reshape(-1)
            target = torch.tensor(
                base_xyz, device=device, dtype=torch.float32)
            lower, upper = info["lower"], info["upper"]
            span = torch.clamp(upper - lower, min=1e-4)
            generator = torch.Generator(device=device)
            generator.manual_seed(7187 + (0 if arm == "left" else 1009))
            # Find wrist_roll column — pin it or locally bound it.
            wrist_col: int | None = None
            wrist_name = f"{arm}_wrist_roll_joint"
            if wrist_name in info["joint_names"]:
                wrist_col = info["joint_names"].index(wrist_name)
                if not lock_wrist_roll:
                    # Limit wrist rotation to ±max_wrist_delta from current.
                    lower = lower.clone()
                    upper = upper.clone()
                    lower[wrist_col] = torch.maximum(
                        lower[wrist_col],
                        seed_q[wrist_col] - max_wrist_delta)
                    upper[wrist_col] = torch.minimum(
                        upper[wrist_col],
                        seed_q[wrist_col] + max_wrist_delta)

            best_q = torch.clamp(seed_q, lower, upper).detach().clone()
            best_rank = (2, float("inf"))

            def _evaluate(samples):
                nonlocal best_q, best_rank
                pose = _clamp_fk_batch(arm, samples, info)
                pos_err = torch.linalg.norm(
                    pose.position - target, dim=-1)
                seed_dist = torch.mean(
                    ((samples - seed_q) / span) ** 2, dim=-1)
                # Palm orientation: local +X projected onto Y axis
                quat = pose.quaternion
                w, x, y, z = (quat[:, i] for i in range(4))
                local_x_y = 2.0 * (x * y + w * z)
                inward_dot = (-local_x_y if arm == "left" else local_x_y)
                # Finger direction: local +Z projected onto world -Z
                local_z_z = 1.0 - 2.0 * (x * x + y * y)
                down_dot = -local_z_z

                valid = ((inward_dot >= min_palm_inward)
                         & (down_dot >= min_finger_down))
                if bool(valid.any().item()):
                    score = (pos_err
                             + 0.04 * (1.0 - inward_dot)
                             + 0.01 * (1.0 - down_dot)
                             + 0.001 * seed_dist)
                    score = torch.where(
                        valid, score, torch.full_like(score, float("inf")))
                    idx = int(torch.argmin(score).item())
                    rank = (0, float(score[idx].detach().cpu().item()))
                else:
                    score = (pos_err
                             + 0.08 * torch.relu(min_palm_inward - inward_dot)
                             + 0.05 * torch.relu(min_finger_down - down_dot)
                             + 0.001 * seed_dist)
                    idx = int(torch.argmin(score).item())
                    rank = (1, float(score[idx].detach().cpu().item()))
                if rank < best_rank:
                    best_rank = rank
                    best_q = samples[idx].detach().clone()

            # Global search (large batch for orientation-constrained IK)
            samples = lower + torch.rand(
                (16384, len(info["joint_names"])),
                device=device, generator=generator) * (upper - lower)
            samples[0] = best_q
            if wrist_col is not None:
                samples[:, wrist_col] = seed_q[wrist_col]
            _evaluate(samples)
            # Local refinement rounds (progressively tighter)
            for radius in (0.5, 0.25, 0.12, 0.05, 0.02):
                centre = best_q.clone()
                lo = torch.maximum(lower, centre - radius)
                hi = torch.minimum(upper, centre + radius)
                samples = lo + torch.rand(
                    (4096, len(info["joint_names"])),
                    device=device, generator=generator) * (hi - lo)
                samples[0] = centre
                if wrist_col is not None:
                    samples[:, wrist_col] = seed_q[wrist_col]
                _evaluate(samples)

            # Verify solution quality
            achieved_pose = _clamp_fk_batch(arm, best_q, info)
            achieved_pos = achieved_pose.position[0]
            pos_err = float(torch.linalg.norm(
                achieved_pos - target).detach().cpu().item())
            aq = achieved_pose.quaternion[0]
            aw, ax, ay, az = (float(aq[i].detach().cpu().item()) for i in range(4))
            a_inward = -(2.0 * (ax * ay + aw * az)) if arm == "left" \
                else (2.0 * (ax * ay + aw * az))
            a_down = -(1.0 - 2.0 * (ax * ax + ay * ay))

            if pos_err > 0.025:
                _log(f"Auto-grasp: {label} FK search failed, "
                     f"pos_err={pos_err:.4f}m")
                return False
            if best_rank[0] > 0:
                _log(f"Auto-grasp: {label} FK search — no orientation-valid "
                     f"solution (inward={a_inward:.3f} need>={min_palm_inward:.2f}, "
                     f"down={a_down:.3f} need>={min_finger_down:.2f})")
                return False

            max_delta = float(torch.max(
                torch.abs(best_q - seed_q)).detach().cpu().item())
            steps = max(2, int(math.ceil(max_delta / 0.02)) + 1)
            alpha = torch.linspace(
                0.0, 1.0, steps, device=device).unsqueeze(1)
            q_waypoints = (seed_q.unsqueeze(0)
                           + alpha * (best_q - seed_q).unsqueeze(0))
            active_traj = ActiveTrajectory(
                request_id=(auto_grasp_ctx.request_id
                            if auto_grasp_ctx is not None else None),
                command="auto_grasp",
                arm=arm,
                joint_names=list(info["joint_names"]),
                q_waypoints=q_waypoints,
                steps_per_wp=max(1, args.steps_per_waypoint),
                settle_steps=args.settle_steps,
                suppress_done=True,
            )
            joint_deltas = {n: round(float((best_q[i] - seed_q[i]).cpu()), 3)
                           for i, n in enumerate(info["joint_names"])}
            _log(f"Auto-grasp: {label} FK solution, "
                 f"pos_err={pos_err:.4f}m, inward={a_inward:.3f}, "
                 f"down={a_down:.3f}, waypoints={steps}, "
                 f"joint_delta={joint_deltas}")
            return True
        except Exception as exc:
            _log(f"Auto-grasp: {label} FK error: {exc}")
            return False

    def _auto_set_hand(arm: str, values: list | tuple) -> bool:
        """Issue an internal set_hand for auto-grasp."""
        return handle_set_hand({
            "type": "set_hand",
            "hand": arm,
            "values": list(values),
            "_internal": True,
            "_quiet": True,
        })

    def _compute_palm_normal(arm: str) -> list[float]:
        """Compute O6 palm normal in world frame (hand-base local +X)."""
        tool_frame = TOOL_FRAMES[arm]
        body_idx = robot.body_names.index(tool_frame)
        quat = robot.data.body_quat_w[0, body_idx].detach().cpu().tolist()
        w, x, y, z = quat
        # Rotation matrix column for local X.  O6 fingers flex around local Y,
        # so the palm face is the YZ plane (not the XZ plane).
        nx_x = 1.0 - 2.0 * (y * y + z * z)
        nx_y = 2.0 * (x * y + w * z)
        nx_z = 2.0 * (x * z - w * y)
        return [nx_x, nx_y, nx_z]

    def _log_palm_info(arm: str):
        """Log palm normal and hand/object positions for diagnostics."""
        palm_n = _compute_palm_normal(arm)
        tool_frame = TOOL_FRAMES[arm]
        body_idx = robot.body_names.index(tool_frame)
        hand_xyz = robot.data.body_pos_w[0, body_idx].detach().cpu().tolist()
        obj = build_object_pose()
        _log(f"Auto-grasp: hand_world={[round(v,3) for v in hand_xyz]}, "
             f"palm_N=[{palm_n[0]:.3f},{palm_n[1]:.3f},{palm_n[2]:.3f}], "
             f"obj_world={obj['xyz']}")

    def _auto_fingertip_forces(arm: str) -> dict[str, float]:
        """Return per-finger fingertip→object force magnitudes (N)."""
        prefix = "lh" if arm == "left" else "rh"
        forces: dict[str, float] = {}
        for finger in FINGER_NAMES:
            key = f"{prefix}_{finger}_distal"
            sensor = fingertip_object_sensors.get(key)
            val = 0.0
            if sensor is not None:
                fm = getattr(sensor.data, "force_matrix_w", None)
                if fm is not None and fm.numel() > 0:
                    val = float(torch.linalg.norm(
                        fm[0].reshape(-1, 3), dim=-1).max().cpu().item())
            forces[finger] = val
        return forces

    def _check_grasp_stability(
        forces: dict[str, float], profile: GraspProfile,
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate per-finger forces against stable grasp criteria."""
        thr = profile.contact_threshold_n
        contacts = {f: forces.get(f, 0.0) > thr for f in FINGER_NAMES}
        total_contacts = sum(contacts.values())
        four_contacts = sum(contacts[f] for f in FOUR_FINGER_NAMES)
        max_force = max(forces.values(), default=0.0)
        thumb_ok = forces.get("thumb", 0.0) > thr
        stable = (
            thumb_ok
            and four_contacts >= profile.min_four_finger_contacts
            and total_contacts >= profile.min_total_contacts
            and max_force < profile.max_contact_force_n
        )
        metrics = {
            "stable": stable,
            "thumb_n": forces.get("thumb", 0.0),
            "four_contacts": four_contacts,
            "total_contacts": total_contacts,
            "max_n": max_force,
            "per_finger": {k: round(v, 4) for k, v in forces.items()},
        }
        return stable, metrics

    def _hand_joint_names(arm: str) -> list[str]:
        prefix = "lh" if arm == "left" else "rh"
        return [f"{prefix}_{s}" for s in HAND_ACTIVE_SUFFIXES]

    def _step_force_close(arm: str, profile: GraspProfile) -> list[float]:
        """Increment force-close joints by one step. Returns new target values."""
        target = robot.data.joint_pos_target.clone()
        names = _hand_joint_names(arm)
        primary_set = set(profile.force_close_indices)
        support_set = set(profile.force_support_indices)
        values: list[float] = []
        for i, name in enumerate(names):
            idx = name_to_idx[name]
            lo = limits_dict[name]["lower"]
            hi = limits_dict[name]["upper"]
            cur = float(target[0, idx].detach().cpu())
            if i in primary_set:
                hi = min(hi, profile.force_close_max)
                val = min(hi, cur + profile.force_close_step)
            elif i in support_set:
                hi = min(hi, profile.force_support_max_rad)
                val = min(hi, cur + profile.force_support_step)
            else:
                val = float(profile.open_values[i])
            val = max(lo, min(hi, val))
            target[:, idx] = val
            values.append(val)
        robot.set_joint_position_target(target)
        return values

    def _force_close_at_limit(arm: str, profile: GraspProfile) -> bool:
        """Check if all primary force-close joints have reached their max angle."""
        target = robot.data.joint_pos_target[0]
        names = _hand_joint_names(arm)
        for i in profile.force_close_indices:
            name = names[i]
            hi = min(limits_dict[name]["upper"], profile.force_close_max)
            cur = float(target[name_to_idx[name]].detach().cpu())
            if cur < hi - 1e-4:
                return False
        return True

    def _fail_auto_grasp(code: str, message: str):
        nonlocal auto_grasp_ctx
        ctx = auto_grasp_ctx
        auto_grasp_ctx = None
        phase = ctx.phase.name if ctx is not None else "UNKNOWN"
        _log(f"Auto-grasp FAILED at {phase}: {message}")
        if ctx is not None and ctx.request_id:
            server.publish({
                "type": "error", "id": ctx.request_id,
                "command": "auto_grasp", "code": code, "message": message,
                "phase": phase,
                "candidate": _candidate_summary(ctx.selected),
                "forces": ctx.last_forces,
            })

    def _finish_auto_grasp(status: str = "complete"):
        nonlocal auto_grasp_ctx
        ctx = auto_grasp_ctx
        auto_grasp_ctx = None
        elapsed = time.monotonic() - ctx.t_start
        _log(f"Auto-grasp {status}: {elapsed:.2f}s")
        server.publish({
            "type": "done", "id": ctx.request_id,
            "command": "auto_grasp", "status": status,
            "elapsed_sec": round(elapsed, 3),
            "candidate": _candidate_summary(ctx.selected),
            "forces": ctx.last_forces,
        })

    def handle_auto_grasp(msg: dict):
        nonlocal auto_grasp_ctx, record_active
        if (active_traj is not None or dual_arm_traj is not None
                or auto_grasp_ctx is not None or auto_clamp_ctx is not None):
            reply_error(msg, "busy", "robot busy, send 'stop' first")
            return

        arm = str(msg.get("arm", "left")).lower()
        if arm not in TOOL_FRAMES:
            reply_error(msg, "bad_request", "arm must be 'left' or 'right'")
            return

        obj_key = str(args.object)
        profile = GRASP_PROFILES.get(obj_key)
        if profile is None:
            available = ", ".join(sorted(GRASP_PROFILES))
            reply_error(msg, "no_profile",
                        f"no grasp profile for '{obj_key}'. "
                        f"Available: {available}")
            return

        candidates = _build_approach_candidates(arm, profile)
        if not candidates:
            reply_error(msg, "no_candidates",
                        "failed to generate approach candidates")
            return

        auto_grasp_ctx = AutoGraspContext(
            request_id=msg_id(msg),
            arm=arm, object_key=obj_key, profile=profile,
            candidates=candidates,
        )

        _log(f"Auto-grasp START: arm={arm}, object={obj_key}, "
             f"profile=(y_off={profile.y_offset}, x_back={profile.x_backoff}, "
             f"force_max={profile.force_close_max}), "
             f"candidates={len(candidates)}")
        # Enable recording for auto_grasp if record_dir is configured
        if record_dir is not None:
            record_active = True
            _log("Auto-grasp: recording enabled")
        reply_ack(msg, status="auto_grasp_started",
                  object=obj_key, phase="OPEN_HAND",
                  candidates=[_candidate_summary(c) for c in candidates])

    def step_auto_grasp():
        """Advance the auto-grasp state machine by one tick."""
        nonlocal auto_grasp_ctx, active_traj
        ctx = auto_grasp_ctx
        if ctx is None:
            return
        if active_traj is not None:
            return

        # Continuously command wrist_roll during non-trajectory phases
        # (SETTLE, FORCE_CLOSE, VERIFY, etc.) when profile specifies it.
        if ctx.profile.wrist_roll_rad is not None:
            wrist_name = f"{ctx.arm}_wrist_roll_joint"
            if wrist_name in name_to_idx:
                target = robot.data.joint_pos_target.clone()
                target[:, name_to_idx[wrist_name]] = ctx.profile.wrist_roll_rad
                robot.set_joint_position_target(target)

        phase = ctx.phase

        # ── Phase 1: OPEN_HAND (set hand to transit/pregrasp shape) ──
        if phase == AutoGraspPhase.OPEN_HAND:
            if not ctx.hand_opened:
                if ctx.profile.use_side_raise:
                    # Curl fingers for transit — reduces collision envelope
                    # during side raise and approach. Fingers open later
                    # in OPEN_PREGRASP after arm reaches position.
                    _auto_set_hand(ctx.arm, ctx.profile.transit_values)
                    _log("Auto-grasp: hand set to transit (curled)")
                else:
                    hand_vals = ctx.profile.pregrasp_hand or ctx.profile.transit_values
                    _auto_set_hand(ctx.arm, hand_vals)
                    _log(f"Auto-grasp: hand set to "
                         f"{'pregrasp' if ctx.profile.pregrasp_hand else 'transit'}")
                ctx.hand_opened = True
            if ctx.profile.use_side_raise:
                ctx.phase = AutoGraspPhase.SIDE_RAISE
                ctx.move_started = False
                _log("Auto-grasp: OPEN_HAND -> SIDE_RAISE")
            else:
                ctx.phase = AutoGraspPhase.APPROACH
                ctx.move_started = False
                _log("Auto-grasp: OPEN_HAND -> APPROACH (skip RAISE)")

        # ── Phase 1.2: SIDE_RAISE (shoulder_roll up to clear table) ──
        elif phase == AutoGraspPhase.SIDE_RAISE:
            if not ctx.move_started:
                arm = ctx.arm
                sign = 1.0 if arm == "left" else -1.0
                q_start = get_arm_q(arm, full=True).detach().clone()
                q_goal = q_start.clone()
                pitch_name = f"{arm}_shoulder_pitch_joint"
                roll_name = f"{arm}_shoulder_roll_joint"
                pitch_idx = arm_joint_names_map[arm].index(pitch_name)
                roll_idx = arm_joint_names_map[arm].index(roll_name)
                desired_pitch = ctx.profile.side_raise_pitch_rad
                roll_rad = getattr(ctx, '_retry_roll', ctx.profile.side_raise_roll_rad)
                desired_roll = sign * roll_rad
                q_goal[pitch_idx] = max(
                    limits_dict[pitch_name]["lower"],
                    min(limits_dict[pitch_name]["upper"], desired_pitch),
                )
                q_goal[roll_idx] = max(
                    limits_dict[roll_name]["lower"],
                    min(limits_dict[roll_name]["upper"], desired_roll),
                )
                steps = 80
                alpha = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
                q_traj = q_start.unsqueeze(0) + alpha * (q_goal - q_start).unsqueeze(0)
                active_traj = ActiveTrajectory(
                    request_id=ctx.request_id,
                    command="auto_grasp",
                    arm=arm,
                    joint_names=list(arm_joint_names_map[arm]),
                    q_waypoints=q_traj.to(device=device),
                    steps_per_wp=3,
                    settle_steps=20,
                    suppress_done=True,
                )
                ctx.move_started = True
                _log(f"Auto-grasp: SIDE_RAISE pitch/roll -> "
                     f"{desired_pitch:.3f}/{desired_roll:.3f} rad ({steps} steps)")
            else:
                ctx.phase = AutoGraspPhase.VERIFY_CLEARANCE
                ctx.move_started = False
                ctx.step_counter = 0
                _log("Auto-grasp: SIDE_RAISE done -> VERIFY_CLEARANCE")

        # ── Phase 1.3: VERIFY_CLEARANCE (check fingertips above table) ──
        elif phase == AutoGraspPhase.VERIFY_CLEARANCE:
            clearance = _fingertip_min_world_z()
            arm_z = clearance.get(ctx.arm, 0.0)
            required_z = TABLE_SURFACE_Z + 0.03  # 3cm clearance
            if arm_z >= required_z:
                _log(f"Auto-grasp: fingertip clearance passed "
                     f"({ctx.arm}={arm_z:.3f}m >= {required_z:.3f}m)")
                ctx.phase = AutoGraspPhase.APPROACH
                ctx.move_started = False
                ctx.step_counter = 0
            else:
                # Increment roll and retry side raise
                new_roll = ctx.profile.side_raise_roll_rad + 0.10
                max_roll = 2.2  # near joint limit
                if new_roll > max_roll:
                    _fail_auto_grasp(
                        "clearance_failed",
                        f"fingertip {ctx.arm}={arm_z:.3f}m below "
                        f"{required_z:.3f}m at max roll")
                    return
                _log(f"Auto-grasp: clearance insufficient "
                     f"({ctx.arm}={arm_z:.3f}m < {required_z:.3f}m); "
                     f"retry roll={new_roll:.2f}rad")
                # Re-curl fingers and retry
                _auto_set_hand(ctx.arm, ctx.profile.transit_values)
                # Update the profile's roll for retry — store on ctx
                ctx._retry_roll = new_roll
                ctx.phase = AutoGraspPhase.SIDE_RAISE
                ctx.move_started = False

        # ── Phase 1.5: RAISE (lift arm above table to clear elbow) ──
        # All candidates share the same RAISE target (near-body, natural y).
        # Once raised, skip re-raising on APPROACH/DESCEND fallback.
        elif phase == AutoGraspPhase.RAISE:
            if not ctx.move_started:
                if ctx.candidate_idx >= len(ctx.candidates):
                    _fail_auto_grasp(
                        "planning_failed",
                        f"all {ctx.attempts} approach candidates exhausted")
                    return
                cand = ctx.candidates[ctx.candidate_idx]
                # If already raised, skip directly to APPROACH
                if getattr(ctx, "raised_done", False):
                    ctx.selected = cand
                    ctx.candidate_idx += 1
                    ctx.attempts += 1
                    ctx.phase = AutoGraspPhase.APPROACH
                    ctx.move_started = False
                    _log(f"Auto-grasp: already raised, APPROACH candidate "
                         f"{cand['index']}")
                    return
                ctx.candidate_idx += 1
                ctx.attempts += 1
                _log(f"Auto-grasp: RAISE candidate {cand['index']} "
                     f"(round={cand['round']}, sample={cand['sample']}, "
                     f"x_back={cand['x_backoff']:.3f}, z_d={cand['z_delta']:.3f})")
                ok = _start_auto_move(ctx.arm, cand["raised"])
                if not ok:
                    _fail_auto_grasp(
                        "planning_failed",
                        "RAISE target unreachable")
                    return
                ctx.selected = cand
                ctx.move_started = True
            else:
                _log_palm_info(ctx.arm)
                ctx.raised_done = True
                ctx.phase = AutoGraspPhase.APPROACH
                ctx.move_started = False
                _log("Auto-grasp: RAISE done -> APPROACH")

        # ── Phase 2: APPROACH (cuRobo move to grasp approach position) ──
        elif phase == AutoGraspPhase.APPROACH:
            if not ctx.move_started:
                # Pick next candidate if none selected
                if ctx.selected is None:
                    if ctx.candidate_idx >= len(ctx.candidates):
                        _fail_auto_grasp(
                            "planning_failed",
                            f"all {ctx.attempts} approach candidates exhausted")
                        return
                    ctx.selected = ctx.candidates[ctx.candidate_idx]
                    ctx.candidate_idx += 1
                    ctx.attempts += 1
                if ctx.profile.use_side_raise:
                    # After side raise, cuRobo can't plan from the extreme
                    # configuration. Use FK search + joint-space interpolation
                    # with palm orientation constraints.
                    ok = _start_fk_linear_move(
                        ctx.arm, ctx.selected["approach"],
                        min_palm_inward=-1.0,   # relaxed: wrist locked
                        min_finger_down=-1.0,   # relaxed: wrist locked
                        label="APPROACH",
                        lock_wrist_roll=True)
                else:
                    cand_quat = ctx.selected.get("quat_wxyz")
                    cand_orient_tol = (ctx.profile.grasp_orientation_tolerance
                                       if cand_quat is not None else None)
                    ok = _start_auto_move(ctx.arm, ctx.selected["approach"],
                                          quat_wxyz=cand_quat,
                                          orientation_tolerance=cand_orient_tol)
                if not ok:
                    _log(f"Auto-grasp: approach candidate {ctx.selected['index']} "
                         f"failed, trying next")
                    ctx.selected = None
                    ctx.move_started = False
                    return  # retry APPROACH with next candidate
                ctx.move_started = True
            else:
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_cfg.dt)
                _log_palm_info(ctx.arm)
                if ctx.profile.use_side_raise and ctx.profile.pregrasp_y_offset is not None:
                    # Keep fingers curled — go straight to MOVE_GRASP,
                    # open hand only after reaching grasp position.
                    ctx.phase = AutoGraspPhase.MOVE_GRASP
                    ctx.move_started = False
                    _log("Auto-grasp: APPROACH done -> MOVE_GRASP (fingers curled)")
                elif ctx.profile.use_side_raise and ctx.profile.pregrasp_hand:
                    # Arm at position with wrist locked — now do in-place
                    # wrist adjustment via DESCEND (FK with wrist free).
                    ctx.phase = AutoGraspPhase.DESCEND
                    ctx.move_started = False
                    _log("Auto-grasp: APPROACH done -> DESCEND (wrist adjust)")
                elif ctx.profile.pregrasp_y_offset is not None:
                    ctx.phase = AutoGraspPhase.MOVE_GRASP
                    ctx.move_started = False
                    _log("Auto-grasp: APPROACH done -> MOVE_GRASP")
                else:
                    # Skip DESCEND if approach and descend are at same height
                    a_z = ctx.selected["approach"][2]
                    d_z = ctx.selected["descend"][2]
                    if abs(a_z - d_z) < 0.005:
                        ctx.phase = AutoGraspPhase.SETTLE
                        ctx.step_counter = 0
                        _log("Auto-grasp: APPROACH done -> SETTLE (skip DESCEND)")
                    else:
                        ctx.phase = AutoGraspPhase.DESCEND
                        _log("Auto-grasp: APPROACH done -> DESCEND")
                ctx.move_started = False

        # ── Phase 2.3: OPEN_PREGRASP (open hand after arm is in position) ──
        elif phase == AutoGraspPhase.OPEN_PREGRASP:
            if ctx.step_counter == 0:
                hand_vals = ctx.profile.pregrasp_hand or ctx.profile.open_values
                _auto_set_hand(ctx.arm, hand_vals)
                _log("Auto-grasp: hand opened to pregrasp shape")
            ctx.step_counter += 1
            if ctx.step_counter >= 20:  # ~0.4s settle for fingers
                ctx.phase = AutoGraspPhase.SETTLE
                ctx.step_counter = 0
                _log("Auto-grasp: OPEN_PREGRASP done -> SETTLE")

        # ── Phase 2.5: MOVE_GRASP (small move from pre-grasp to grasp point) ──
        elif phase == AutoGraspPhase.MOVE_GRASP:
            if ctx.selected is None:
                _fail_auto_grasp("internal_error", "no selected candidate")
                return
            if not ctx.move_started:
                # FK-based joint-space interpolation with orientation.
                ok = _start_fk_linear_move(
                    ctx.arm, ctx.selected["grasp"],
                    min_palm_inward=ctx.profile.fk_min_palm_inward,
                    min_finger_down=ctx.profile.fk_min_finger_down,
                    label="MOVE_GRASP")
                if not ok:
                    _fail_auto_grasp("planning_failed",
                                     "MOVE_GRASP FK search failed")
                    return
                ctx.move_started = True
            else:
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_cfg.dt)
                _log_palm_info(ctx.arm)
                ctx.move_started = False
                if ctx.profile.use_side_raise and ctx.profile.pregrasp_hand:
                    # Now at grasp position — open fingers.
                    ctx.phase = AutoGraspPhase.OPEN_PREGRASP
                    ctx.step_counter = 0
                    _log("Auto-grasp: MOVE_GRASP done -> OPEN_PREGRASP")
                else:
                    ctx.phase = AutoGraspPhase.SETTLE
                    ctx.step_counter = 0
                    _log("Auto-grasp: MOVE_GRASP done -> SETTLE")

        # ── Phase 3: DESCEND (move to grasp height) ──
        elif phase == AutoGraspPhase.DESCEND:
            if ctx.selected is None:
                _fail_auto_grasp("internal_error", "no selected candidate")
                return

            if not ctx.move_started:
                if ctx.profile.use_side_raise:
                    # FK-based descent (cuRobo can't plan from extreme config)
                    ok = _start_fk_linear_move(
                        ctx.arm, ctx.selected["descend"],
                        min_palm_inward=ctx.profile.fk_min_palm_inward,
                        min_finger_down=ctx.profile.fk_min_finger_down,
                        label="DESCEND")
                else:
                    cand_quat = ctx.selected.get("quat_wxyz")
                    cand_orient_tol = (ctx.profile.grasp_orientation_tolerance
                                       if cand_quat is not None else None)
                    ok = _start_auto_move(ctx.arm, ctx.selected["descend"],
                                          quat_wxyz=cand_quat,
                                          orientation_tolerance=cand_orient_tol)
                if not ok:
                    _log("Auto-grasp: descend planning failed, trying next candidate")
                    ctx.selected = None
                    ctx.move_started = False
                    ctx.phase = AutoGraspPhase.APPROACH
                    return
                ctx.move_started = True
            else:
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_cfg.dt)
                _log_palm_info(ctx.arm)
                ctx.move_started = False
                if ctx.profile.use_side_raise and ctx.profile.pregrasp_hand:
                    ctx.phase = AutoGraspPhase.OPEN_PREGRASP
                    ctx.step_counter = 0
                    _log("Auto-grasp: DESCEND done -> OPEN_PREGRASP")
                else:
                    ctx.phase = AutoGraspPhase.SETTLE
                    ctx.step_counter = 0
                    _log("Auto-grasp: DESCEND done -> SETTLE")

        # ── Phase 4: SETTLE (open hand + wait for physics to stabilize) ──
        elif phase == AutoGraspPhase.SETTLE:
            if ctx.step_counter == 0:
                # Open hand from transit position for grasping
                _auto_set_hand(ctx.arm, ctx.profile.open_values)
                _log("Auto-grasp: hand opened for grasp")
            ctx.step_counter += 1
            if ctx.step_counter >= 15:
                ctx.phase = AutoGraspPhase.FORCE_CLOSE
                ctx.step_counter = 0
                _log("Auto-grasp: SETTLE done -> FORCE_CLOSE")

        # ── Phase 5: FORCE_CLOSE (incremental force-guided finger closing) ──
        elif phase == AutoGraspPhase.FORCE_CLOSE:
            forces = _auto_fingertip_forces(ctx.arm)
            stable, metrics = _check_grasp_stability(forces, ctx.profile)
            ctx.last_forces = metrics

            if metrics["max_n"] >= ctx.profile.max_contact_force_n:
                _fail_auto_grasp(
                    "force_limit",
                    f"max force {metrics['max_n']:.3f}N >= "
                    f"limit {ctx.profile.max_contact_force_n}N")
                return

            if stable:
                ctx.phase = AutoGraspPhase.VERIFY_GRASP
                _log(f"Auto-grasp: contact detected at step {ctx.step_counter} "
                     f"metrics={metrics}")
                return

            if _force_close_at_limit(ctx.arm, ctx.profile):
                _fail_auto_grasp(
                    "grasp_failed",
                    f"force close reached {ctx.profile.force_close_max:.2f}rad "
                    f"without stable grasp: {metrics}")
                return

            vals = _step_force_close(ctx.arm, ctx.profile)
            ctx.step_counter += 1
            if ctx.step_counter % 5 == 0:
                _log(f"Auto-grasp: force close step={ctx.step_counter} "
                     f"vals={[round(v, 2) for v in vals]} metrics={metrics}")

        # ── Phase 6: VERIFY_GRASP (check stability before lift) ──
        elif phase == AutoGraspPhase.VERIFY_GRASP:
            forces = _auto_fingertip_forces(ctx.arm)
            stable, metrics = _check_grasp_stability(forces, ctx.profile)
            ctx.last_forces = metrics
            _log(f"Auto-grasp: pre-lift metrics={metrics}")
            if not stable:
                _fail_auto_grasp(
                    "grasp_failed",
                    f"unstable grasp before lift: {metrics}")
                return
            ctx.phase = AutoGraspPhase.LIFT
            ctx.move_started = False
            _log("Auto-grasp: VERIFY_GRASP passed -> LIFT")

        # ── Phase 7: LIFT ──
        elif phase == AutoGraspPhase.LIFT:
            if ctx.selected is None:
                _fail_auto_grasp("internal_error", "no selected candidate")
                return

            if not ctx.move_started:
                cand_quat = ctx.selected.get("quat_wxyz")
                cand_orient_tol = (ctx.profile.grasp_orientation_tolerance
                                   if cand_quat is not None else None)
                ok = _start_auto_move(ctx.arm, ctx.selected["lift"],
                                      quat_wxyz=cand_quat,
                                      orientation_tolerance=cand_orient_tol)
                if not ok:
                    _fail_auto_grasp("planning_failed", "lift move failed")
                    return
                ctx.move_started = True
            else:
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_cfg.dt)
                ctx.phase = AutoGraspPhase.VERIFY_HOLD
                ctx.move_started = False
                _log("Auto-grasp: LIFT done -> VERIFY_HOLD")

        # ── Phase 8: VERIFY_HOLD ──
        elif phase == AutoGraspPhase.VERIFY_HOLD:
            forces = _auto_fingertip_forces(ctx.arm)
            stable, metrics = _check_grasp_stability(forces, ctx.profile)
            ctx.last_forces = metrics
            _log(f"Auto-grasp: post-lift metrics={metrics}")
            if not stable:
                _fail_auto_grasp(
                    "hold_failed",
                    f"object not held after lift: {metrics}")
                return
            _finish_auto_grasp("complete")

    # ---- Dual-arm, force-guided large-box clamp ----

    _clamp_kin_cache: dict[str, dict[str, Any]] = {}

    def _set_clamp_phase(ctx: AutoClampContext, phase: AutoClampPhase) -> None:
        ctx.phase = phase
        ctx.phase_sim_steps = 0
        ctx.step_counter = 0
        ctx.move_started = False
        _log(f"Auto-clamp: phase -> {phase.name}")

    def _hold_all_arm_joints() -> None:
        target = robot.data.joint_pos_target.clone()
        q_actual = robot.data.joint_pos[0]
        for name in all_arm_joint_names:
            target[:, name_to_idx[name]] = q_actual[name_to_idx[name]]
        robot.set_joint_position_target(target)

    def _fail_auto_clamp(code: str, message: str) -> None:
        nonlocal auto_clamp_ctx, dual_arm_traj, record_active, record_tail_frames
        ctx = auto_clamp_ctx
        dual_arm_traj = None
        _hold_all_arm_joints()
        auto_clamp_ctx = None
        if record_dir is not None:
            record_active = False
            record_tail_frames = max(record_tail_frames, int(args.record_fps * 2))
        _log(f"Auto-clamp FAILED [{code}]: {message}")
        if ctx is not None:
            server.publish({
                "type": "error",
                "id": ctx.request_id,
                "command": "auto_clamp",
                "code": code,
                "message": message,
                "phase": ctx.phase.name,
                "forces": ctx.last_forces,
            })

    def _finish_auto_clamp() -> None:
        nonlocal auto_clamp_ctx, record_active, record_tail_frames
        nonlocal record_frame_accumulator, record_frame_index
        ctx = auto_clamp_ctx
        if ctx is None:
            return
        elapsed = time.monotonic() - ctx.t_start
        auto_clamp_ctx = None
        if record_dir is not None:
            record_active = False
            record_tail_frames = max(record_tail_frames, int(args.record_fps * 2))
            _log(f"Auto-clamp recording tail: {record_tail_frames} frames")
        lift_metrics = ctx.metrics.get("lift", {})
        _log(
            f"Auto-clamp COMPLETE (lifted) in {elapsed:.2f}s, "
            f"forces={ctx.last_forces}, lift={lift_metrics}"
        )
        server.publish({
            "type": "done",
            "id": ctx.request_id,
            "command": "auto_clamp",
            "status": "lifted",
            "elapsed_sec": round(elapsed, 3),
            "forces": ctx.last_forces,
            "clamp_alpha": {
                arm: round(value, 5) for arm, value in ctx.clamp_alpha.items()
            },
            "metrics": ctx.metrics,
        })

    def _fingertip_min_world_z() -> dict[str, float]:
        """Lowest estimated physical fingertip, including distal mesh length."""
        result: dict[str, float] = {}
        tip_offset = (
            auto_clamp_ctx.config.fingertip_tip_offset_m
            if auto_clamp_ctx is not None else ClampConfig().fingertip_tip_offset_m
        )
        for arm, prefix in (("left", "lh_"), ("right", "rh_")):
            indices = [
                robot.body_names.index(body)
                for body in O6_FINGERTIP_BODIES if body.startswith(prefix)
            ]
            positions = robot.data.body_pos_w[0, indices]
            quaternions = robot.data.body_quat_w[0, indices]
            qx = quaternions[:, 1]
            qy = quaternions[:, 2]
            local_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
            tip_world_z = positions[:, 2] + tip_offset * local_z_world_z
            result[arm] = float(tip_world_z.min().detach().cpu().item())
        return result

    def _clamp_kinematics(arm: str) -> dict[str, Any]:
        cached = _clamp_kin_cache.get(arm)
        if cached is not None:
            return cached
        cfg_path = str(_resolve_config(arm_config_map[arm]))
        kin = Kinematics(KinematicsCfg.from_robot_yaml_file(cfg_path))
        joint_names = list(kin.joint_names)
        expected = arm_joint_names_map[arm]
        if joint_names != expected:
            raise RuntimeError(
                f"{arm} clamp kinematics joints {joint_names} != {expected}"
            )
        lower = torch.tensor(
            [limits_dict[name]["lower"] for name in joint_names],
            device=device, dtype=torch.float32,
        )
        upper = torch.tensor(
            [limits_dict[name]["upper"] for name in joint_names],
            device=device, dtype=torch.float32,
        )
        cached = {
            "kin": kin,
            "joint_names": joint_names,
            "lower": lower,
            "upper": upper,
            "tool_frame": TOOL_FRAMES[arm],
        }
        _clamp_kin_cache[arm] = cached
        _log(f"Built {arm} 5DOF clamp FK model")
        return cached

    def _clamp_fk_batch(arm: str, q_batch, info: dict[str, Any] | None = None):
        if info is None:
            info = _clamp_kinematics(arm)
        if q_batch.ndim == 1:
            q_batch = q_batch.unsqueeze(0)
        js = JointState.from_position(q_batch, joint_names=info["joint_names"])
        fk = info["kin"].compute_kinematics(js)
        return fk.tool_poses.get_link_pose(info["tool_frame"])

    def _squeeze_y_jacobian(arm: str, anchor_q):
        """Numerically linearise hand-base Y at the measured contact pose."""

        info = _clamp_kinematics(arm)
        anchor_q = anchor_q.to(device=device, dtype=torch.float32).reshape(-1)
        epsilon = 1e-3
        q_batch = anchor_q.repeat(len(anchor_q) + 1, 1)
        for index in range(len(anchor_q)):
            q_batch[index + 1, index] += epsilon
        pose = _clamp_fk_batch(arm, q_batch, info)
        jacobian_y = (pose.position[1:, 1] - pose.position[0, 1]) / epsilon
        if float(torch.linalg.norm(jacobian_y).detach().cpu().item()) < 1e-5:
            raise RuntimeError(f"{arm} contact pose has singular TCP-Y Jacobian")
        return jacobian_y.detach()

    def _squeeze_joint_command(ctx: AutoClampContext, arm: str):
        """Convert virtual normal compression into a contact-local joint target."""

        anchor_q = ctx.squeeze_anchor_q[arm]
        jacobian_y = ctx.squeeze_jacobian_y[arm]
        direction = -1.0 if arm == "left" else 1.0
        desired_delta_y = (
            direction
            * ctx.squeeze_compression_m[arm]
            * ctx.config.squeeze_joint_target_gain
        )
        damping = 1e-5
        delta_q = (
            jacobian_y * desired_delta_y
            / (torch.dot(jacobian_y, jacobian_y) + damping)
        )
        info = _clamp_kinematics(arm)
        return torch.maximum(
            info["lower"], torch.minimum(info["upper"], anchor_q + delta_q)
        )

    def _lift_force_offset_command(
        ctx: AutoClampContext, arm: str, nominal_q,
    ):
        """Add only the force-feedback delta to a moving lift waypoint."""

        jacobian_y = _squeeze_y_jacobian(arm, nominal_q)
        direction = -1.0 if arm == "left" else 1.0
        compression_delta = (
            ctx.squeeze_compression_m[arm]
            - ctx.lift_squeeze_baseline_m[arm]
        )
        desired_delta_y = (
            direction
            * compression_delta
            * ctx.config.squeeze_joint_target_gain
        )
        damping = 1e-5
        delta_q = (
            jacobian_y * desired_delta_y
            / (torch.dot(jacobian_y, jacobian_y) + damping)
        )
        info = _clamp_kinematics(arm)
        return torch.maximum(
            info["lower"], torch.minimum(info["upper"], nominal_q + delta_q)
        )

    def _lift_task_features(arm: str, q_batch):
        """Return Y/Z and the two independent palm-normal components.

        Rotation about the palm normal is intentionally free.  Locking that
        roll as a fifth task consumes every DOF and leaves the 5DOF arm unable
        to move vertically at the clamped pose, even though the contact plane
        can remain facing the carton.
        """

        pose = _clamp_fk_batch(arm, q_batch)
        quat = pose.quaternion
        w, x, y, z = (quat[:, index] for index in range(4))
        local_x_x = 1.0 - 2.0 * (y * y + z * z)
        local_x_z = 2.0 * (x * z - w * y)
        return torch.stack(
            (pose.position[:, 1], pose.position[:, 2],
             local_x_x, local_x_z),
            dim=1,
        )

    def _lift_task_jacobian(arm: str, q):
        """Numerically linearise the four coupled lift task features."""

        q = q.to(device=device, dtype=torch.float32).reshape(-1)
        epsilon = 1e-3
        batch = q.repeat(len(q) + 1, 1)
        for index in range(len(q)):
            batch[index + 1, index] += epsilon
        features = _lift_task_features(arm, batch)
        return features[0], (features[1:] - features[0]).transpose(0, 1) / epsilon

    def _advance_unified_lift_nominal(
        ctx: AutoClampContext, arm: str, full_task: bool,
    ) -> None:
        """Advance only the shared lift path while preserving clamp Y."""

        q = ctx.lift_nominal_q[arm]
        current, jacobian = _lift_task_jacobian(arm, q)
        target = ctx.lift_anchor_features[arm].clone()
        if full_task:
            target[1] += ctx.lift_coordinator.progress_m
        else:
            # During contact recovery move only along the grasp normal.  Do
            # not let a pending Z/orientation error peel the remaining patch.
            target[1:] = current[1:]
        error = target - current
        error[0] = torch.clamp(error[0], -0.001, 0.001)
        error[1] = torch.clamp(error[1], -0.001, 0.001)
        error[2:] = torch.clamp(error[2:], -0.01, 0.01)

        # Strict task priority: carton-normal Y is the force-producing axis
        # and must never be traded against lift or orientation objectives.
        # Solve it first, then project Z and palm-normal corrections into its
        # joint-space nullspace.
        jacobian_y = jacobian[0]
        primary_damping = 1e-5
        primary_denominator = (
            torch.dot(jacobian_y, jacobian_y) + primary_damping
        )
        delta_primary = jacobian_y * error[0] / primary_denominator
        identity = torch.eye(len(q), device=device)
        nullspace = identity - torch.outer(
            jacobian_y, jacobian_y
        ) / primary_denominator

        secondary_jacobian = jacobian[1:] @ nullspace
        secondary_damping = 1e-3
        secondary_system = (
            secondary_jacobian @ secondary_jacobian.transpose(0, 1)
        )
        delta_secondary = nullspace @ secondary_jacobian.transpose(0, 1) @ (
            torch.linalg.solve(
                secondary_system + secondary_damping * torch.eye(
                    secondary_system.shape[0], device=device
                ),
                error[1:],
            )
        )
        delta_q = delta_primary + delta_secondary
        max_delta = float(torch.max(torch.abs(delta_q)).detach().cpu().item())
        if max_delta > 0.003:
            delta_q = delta_q * (0.003 / max_delta)
        info = _clamp_kinematics(arm)
        ctx.lift_nominal_q[arm] = torch.maximum(
            info["lower"], torch.minimum(info["upper"], q + delta_q)
        ).detach()

    def _unified_lift_joint_command(
        ctx: AutoClampContext, arm: str,
    ):
        """Overlay tactile normal correction without drifting the lift path.

        Force feedback is intentionally not integrated into ``lift_nominal_q``.
        A normal correction that cannot be realised under contact therefore
        stays a bounded position-control error instead of recursively changing
        the posture used by the next vertical IK step.
        """

        nominal_q = ctx.lift_nominal_q[arm]
        jacobian_y = _squeeze_y_jacobian(arm, nominal_q)
        coordinator = ctx.lift_coordinator
        desired_delta_y = coordinator.center_offset_m + (
            -coordinator.squeeze_offset_m
            if arm == "left" else coordinator.squeeze_offset_m
        )
        damping = 1e-5
        delta_q = (
            jacobian_y * desired_delta_y
            / (torch.dot(jacobian_y, jacobian_y) + damping)
        )
        info = _clamp_kinematics(arm)
        return torch.maximum(
            info["lower"], torch.minimum(info["upper"], nominal_q + delta_q)
        )

    def _begin_unified_lift(ctx: AutoClampContext) -> None:
        """Capture one paired contact frame and start tactile-gated local IK."""

        forces = ctx.last_forces
        for arm in ("left", "right"):
            # Preserve actuator-target continuity: start from the exact
            # force-producing squeeze command.  Its complete task feature is
            # the zero point, so the first unified command is exactly
            # continuous without physically releasing palm contact.
            nominal_q = _squeeze_joint_command(ctx, arm).detach().clone()
            ctx.lift_nominal_q[arm] = nominal_q
            ctx.lift_anchor_features[arm] = _lift_task_features(
                arm, nominal_q.unsqueeze(0)
            )[0].detach().clone()
            world_pose = current_world_tool_pose(TOOL_FRAMES[arm])
            base_xyz, _ = world_to_base(
                world_pose["xyz"], world_pose["quat_wxyz"]
            )
            ctx.lift_start_hand_base_z[arm] = base_xyz[2]
        ctx.lift_coordinator = BimanualLiftState(
            filtered_left_n=forces["left"]["object_axis_y_n"],
            filtered_right_n=forces["right"]["object_axis_y_n"],
            squeeze_offset_m=0.0,
            center_offset_m=0.0,
        )
        ctx.lift_start_object_z = build_object_pose()["xyz"][2]
        ctx.lift_settle_counter = 0
        ctx.lift_verify_attempt_frames = 0
        _log(
            "Auto-clamp: unified local lift started "
            f"from object z={ctx.lift_start_object_z:.5f}, "
            "relative squeeze/center offsets reset to zero"
        )
        _set_clamp_phase(ctx, AutoClampPhase.MOVE_LIFT)

    def _normalise_wrist_near_seed(arm: str, q, seed, info: dict[str, Any]):
        """Use an equivalent 2pi wrist angle nearest the current pose."""
        q = q.clone()
        wrist_name = f"{arm}_wrist_roll_joint"
        wrist_idx = info["joint_names"].index(wrist_name)
        raw = float(q[wrist_idx].detach().cpu().item())
        reference = float(seed[wrist_idx].detach().cpu().item())
        lo = float(info["lower"][wrist_idx].detach().cpu().item())
        hi = float(info["upper"][wrist_idx].detach().cpu().item())
        candidates = [
            raw + turns * 2.0 * math.pi for turns in range(-3, 4)
            if lo <= raw + turns * 2.0 * math.pi <= hi
        ]
        if candidates:
            q[wrist_idx] = min(candidates, key=lambda value: abs(value - reference))
        return q

    def _validate_clamp_joint_path(
        arm: str, q_start, q_end, label: str, config: ClampConfig,
    ) -> dict[str, float]:
        """Reject low or non-inward joint interpolations before execution."""
        if label == "pregrasp":
            q_path = _staged_pregrasp_arm_path(arm, q_start, q_end)
        else:
            alpha = torch.linspace(0.0, 1.0, 121, device=device).unsqueeze(1)
            q_path = q_start.unsqueeze(0) + alpha * (q_end - q_start).unsqueeze(0)
        pose = _clamp_fk_batch(arm, q_path)
        positions = pose.position
        quat = pose.quaternion
        w, x, y, z = (quat[:, i] for i in range(4))
        local_x_y = 2.0 * (x * y + w * z)
        inward = -local_x_y if arm == "left" else local_x_y
        down = -(1.0 - 2.0 * (x * x + y * y))
        y_step = positions[1:, 1] - positions[:-1, 1]
        backward = (
            torch.relu(y_step).max() if arm == "left"
            else torch.relu(-y_step).max()
        )
        metrics = {
            "min_hand_base_z": float(positions[:, 2].min().detach().cpu().item()),
            "max_backward_y_step_m": float(backward.detach().cpu().item()),
            "min_palm_inward_dot": float(inward.min().detach().cpu().item()),
            "min_fingers_down_dot": float(down.min().detach().cpu().item()),
        }
        if metrics["min_hand_base_z"] < _table_surface_base + 0.10:
            raise RuntimeError(f"{arm} {label} path too low: {metrics}")
        if label == "clamp":
            if metrics["max_backward_y_step_m"] > 0.002:
                raise RuntimeError(f"{arm} clamp path is not inward-monotonic: {metrics}")
            if metrics["min_palm_inward_dot"] < config.min_palm_inward_dot - 0.10:
                raise RuntimeError(f"{arm} clamp path loses palm orientation: {metrics}")
            if metrics["min_fingers_down_dot"] < config.min_finger_down_dot - 0.10:
                raise RuntimeError(f"{arm} clamp path loses finger orientation: {metrics}")
        return metrics

    def _solve_clamp_pose(arm: str, target_xyz, seed_q, label: str):
        """Deterministic GPU Monte-Carlo FK search with local refinement."""
        info = _clamp_kinematics(arm)
        seed_q = seed_q.to(device=device, dtype=torch.float32).reshape(-1)
        config = auto_clamp_ctx.config if auto_clamp_ctx else ClampConfig()
        target = torch.tensor(target_xyz, device=device, dtype=torch.float32)
        lower, upper = info["lower"], info["upper"]
        span = torch.clamp(upper - lower, min=1e-4)
        generator = torch.Generator(device=device)
        generator.manual_seed(4417 + (0 if arm == "left" else 1009)
                              + (0 if label == "pregrasp" else 211))
        required_inward_dot = (
            config.lift_min_palm_inward_dot
            if label == "lift" else config.min_palm_inward_dot
        )
        required_finger_down_dot = (
            config.lift_min_finger_down_dot
            if label == "lift" else config.min_finger_down_dot
        )

        best_q = seed_q.clone()
        best_rank = (2, float("inf"))
        best_metrics: dict[str, float] = {}

        def evaluate(samples) -> None:
            nonlocal best_q, best_rank, best_metrics
            pose = _clamp_fk_batch(arm, samples, info)
            position_delta = pose.position - target
            position_error = torch.linalg.norm(position_delta, dim=-1)
            if label == "lift":
                # A structural O6 clamp needs vertical/normal accuracy; X is
                # free to drift together on both hands to satisfy 5-DOF IK.
                task_position_error = torch.sqrt(
                    (0.20 * position_delta[:, 0]) ** 2
                    + position_delta[:, 1] ** 2
                    + position_delta[:, 2] ** 2
                )
            else:
                task_position_error = position_error
            quat = pose.quaternion
            w, x, y, z = (quat[:, i] for i in range(4))
            local_x_y = 2.0 * (x * y + w * z)
            inward_dot = (-local_x_y if arm == "left" else local_x_y)
            local_z_z = 1.0 - 2.0 * (x * x + y * y)
            down_dot = -local_z_z
            seed_distance = torch.mean(((samples - seed_q) / span) ** 2, dim=-1)
            valid = ((inward_dot >= required_inward_dot)
                     & (down_dot >= required_finger_down_dot))
            if bool(valid.any().item()):
                score = (task_position_error
                         + 0.04 * (1.0 - inward_dot)
                         + 0.01 * (1.0 - down_dot)
                         + 0.001 * seed_distance)
                score = torch.where(valid, score, torch.full_like(score, float("inf")))
                index = int(torch.argmin(score).item())
                rank = (0, float(score[index].detach().cpu().item()))
            else:
                score = (task_position_error
                         + 0.08 * torch.relu(required_inward_dot - inward_dot)
                         + 0.05 * torch.relu(required_finger_down_dot - down_dot)
                         + 0.001 * seed_distance)
                index = int(torch.argmin(score).item())
                rank = (1, float(score[index].detach().cpu().item()))
            if rank < best_rank:
                best_rank = rank
                best_q = samples[index].detach().clone()
                best_metrics = {
                    "position_error_m": float(position_error[index].detach().cpu().item()),
                    "palm_inward_dot": float(inward_dot[index].detach().cpu().item()),
                    "fingers_down_dot": float(down_dot[index].detach().cpu().item()),
                }

        global_samples = lower + torch.rand(
            (500_000, len(info["joint_names"])),
            device=device, generator=generator,
        ) * (upper - lower)
        global_samples[0] = seed_q.clamp(lower, upper)
        evaluate(global_samples)
        del global_samples

        for radius, count in ((0.35, 150_000), (0.10, 150_000)):
            centre = best_q.clone()
            local_lower = torch.maximum(lower, centre - radius)
            local_upper = torch.minimum(upper, centre + radius)
            samples = local_lower + torch.rand(
                (count, len(info["joint_names"])),
                device=device, generator=generator,
            ) * (local_upper - local_lower)
            samples[0] = centre
            evaluate(samples)
            del samples

        best_q = _normalise_wrist_near_seed(arm, best_q, seed_q, info)
        final_pose = _clamp_fk_batch(arm, best_q, info)
        final_pos = final_pose.position[0].detach().cpu().tolist()
        final_quat = final_pose.quaternion[0].detach().cpu().tolist()
        inward_dot, down_dot = palm_alignment(final_quat, arm)
        position_error = math.dist(final_pos, list(target_xyz))
        best_metrics = {
            "position_error_m": position_error,
            "position_error_xyz_m": [
                round(final_pos[i] - float(target_xyz[i]), 6)
                for i in range(3)
            ],
            "palm_inward_dot": inward_dot,
            "fingers_down_dot": down_dot,
            "achieved_xyz": [round(value, 6) for value in final_pos],
            "target_xyz": [round(float(value), 6) for value in target_xyz],
        }
        if label == "lift":
            dx, dy, dz = (
                abs(final_pos[i] - float(target_xyz[i])) for i in range(3)
            )
            position_valid = (
                dx <= config.lift_max_x_drift_m
                and dy <= config.lift_max_normal_error_m
                and dz <= config.lift_max_z_error_m
            )
        else:
            position_valid = position_error <= config.max_pose_error_m
        if (not position_valid
                or inward_dot < required_inward_dot
                or down_dot < required_finger_down_dot):
            raise RuntimeError(
                f"{arm} {label} FK search outside tolerance: {best_metrics}"
            )
        _log(f"Auto-clamp: {arm} {label} solution {best_metrics}")
        return best_q, best_metrics

    def _verify_pregrasp_actual(ctx: AutoClampContext) -> tuple[bool, str]:
        for arm in ("left", "right"):
            frame_pose = current_world_tool_pose(TOOL_FRAMES[arm])
            base_xyz, base_quat = world_to_base(
                frame_pose["xyz"], frame_pose["quat_wxyz"]
            )
            target_xyz = ctx.targets[arm]["pregrasp"]
            position_error = math.dist(base_xyz, target_xyz)
            inward_dot, down_dot = palm_alignment(base_quat, arm)
            actual_metrics = {
                "position_error_m": position_error,
                "palm_inward_dot": inward_dot,
                "fingers_down_dot": down_dot,
                "actual_xyz": [round(value, 6) for value in base_xyz],
            }
            ctx.metrics[arm]["pregrasp_actual"] = actual_metrics
            if position_error > ctx.config.max_pose_error_m:
                return False, f"{arm} pregrasp position error {position_error:.4f}m"
            if inward_dot < ctx.config.min_palm_inward_dot:
                return False, f"{arm} palm inward dot {inward_dot:.3f}"
            if down_dot < ctx.config.min_finger_down_dot:
                return False, f"{arm} fingers-down dot {down_dot:.3f}"
        return True, ""

    def handle_auto_clamp(msg: dict) -> None:
        nonlocal auto_clamp_ctx, record_active, record_tail_frames
        nonlocal record_frame_accumulator, record_frame_index
        if (active_traj is not None or dual_arm_traj is not None
                or auto_grasp_ctx is not None or auto_clamp_ctx is not None):
            reply_error(msg, "busy", "robot busy, send 'stop' first")
            return
        object_key = str(msg.get("object", args.object))
        if object_key not in AUTO_CLAMP_OBJECT_KEYS:
            reply_error(
                msg, "no_profile",
                f"auto_clamp has no profile for {object_key!r}; supported: "
                f"{', '.join(AUTO_CLAMP_OBJECT_KEYS)}",
            )
            return
        if str(args.object) != object_key:
            reply_error(
                msg, "object_mismatch",
                f"auto_clamp requested {object_key!r}, but the server loaded "
                f"{str(args.object)!r}",
            )
            return

        config_overrides: dict[str, Any] = {}
        if record_dir is not None:
            config_overrides.update(
                total_timeout_s=1200.0,
                force_phase_timeout_s=1100.0,
                max_alpha_difference=1.0,
            )
        if args.clamp_force_target_n is not None:
            config_overrides["force_target_n"] = args.clamp_force_target_n
        config = make_clamp_config(object_key, **config_overrides)
        if record_dir is not None:
            record_active = True
            record_tail_frames = 0
            record_frame_accumulator = 0.0
            record_frame_index = 0
            if record_sensor_path is not None:
                record_sensor_path.write_text("", encoding="utf-8")
            _log(f"Auto-clamp recording started: {record_dir}")
        object_pose = build_object_pose()
        object_base = _to_base_xyz(object_pose["xyz"])
        clamp_center_base = (
            object_base[0] + args.clamp_x_offset_m,
            object_base[1],
            object_base[2],
        )
        targets = build_clamp_targets(clamp_center_base, config)
        _log(
            "Auto-clamp frame offsets: "
            f"robot_root_x={args.robot_root_x_offset_m:.4f}m, "
            f"object_base_x={object_base[0]:.4f}m, "
            f"clamp_x_offset={args.clamp_x_offset_m:.4f}m, "
            f"target_base_x={clamp_center_base[0]:.4f}m, "
            f"box_size={config.box_size_xyz_m}, "
            f"force_target={config.force_target_n:.2f}N"
        )
        auto_clamp_ctx = AutoClampContext(
            request_id=msg_id(msg),
            object_key=object_key,
            config=config,
            targets=targets,
            object_snapshot_w=tuple(object_pose["xyz"]),
            object_snapshot_quat_wxyz=tuple(object_pose["quat_wxyz"]),
            side_raise_roll_rad=config.initial_side_raise_roll_rad,
        )
        _log(f"Auto-clamp START: object={object_key}, targets={targets}")
        reply_ack(
            msg,
            status="auto_clamp_started",
            object=object_key,
            phase=auto_clamp_ctx.phase.name,
            targets=targets,
            box_size_xyz_m=config.box_size_xyz_m,
            force_target_n=config.force_target_n,
            force_band_n=[config.force_lower_n, config.force_upper_n],
            lift_height_m=config.lift_height_m,
            lift_min_object_rise_m=config.lift_min_object_rise_m,
        )

    def step_auto_clamp() -> None:
        """Advance the asynchronous side-raise/pregrasp/force-clamp chain."""
        ctx = auto_clamp_ctx
        if ctx is None:
            return
        ctx.total_sim_steps += 1
        ctx.phase_sim_steps += 1
        total_sim_s = ctx.total_sim_steps * sim_cfg.dt
        phase_sim_s = ctx.phase_sim_steps * sim_cfg.dt
        if total_sim_s > ctx.config.total_timeout_s:
            _fail_auto_clamp("timeout", "total auto-clamp timeout")
            return
        phase_timeout = (
            ctx.config.force_phase_timeout_s
            if ctx.phase in (
                AutoClampPhase.FORCE_CLAMP,
                AutoClampPhase.MOVE_LIFT,
            )
            else ctx.config.phase_timeout_s
        )
        if phase_sim_s > phase_timeout:
            _fail_auto_clamp("phase_timeout", f"{ctx.phase.name} timeout")
            return

        phase = ctx.phase
        if phase == AutoClampPhase.PREPARE_HANDS:
            if ctx.step_counter == 0:
                _auto_set_hand("both", CLAMP_TRANSIT_HAND_VALUES)
                _log("Auto-clamp: both hands set to transit curl")
            ctx.step_counter += 1
            if ctx.step_counter >= 20:
                _set_clamp_phase(ctx, AutoClampPhase.SIDE_RAISE)

        elif phase == AutoClampPhase.SIDE_RAISE:
            if not ctx.move_started:
                goals = {}
                for arm, sign in (("left", 1.0), ("right", -1.0)):
                    q_goal = get_arm_q(arm, full=True).detach().clone()
                    roll_idx = arm_joint_names_map[arm].index(
                        f"{arm}_shoulder_roll_joint"
                    )
                    desired = sign * ctx.side_raise_roll_rad
                    joint_name = f"{arm}_shoulder_roll_joint"
                    q_goal[roll_idx] = max(
                        limits_dict[joint_name]["lower"],
                        min(limits_dict[joint_name]["upper"], desired),
                    )
                    goals[arm] = q_goal
                _start_dual_linear(ctx, "auto_clamp_side_raise", goals, steps=120)
                ctx.move_started = True
            elif dual_arm_traj is None:
                _set_clamp_phase(ctx, AutoClampPhase.OPEN_HANDS)

        elif phase == AutoClampPhase.OPEN_HANDS:
            if ctx.step_counter == 0:
                _auto_set_hand("both", _clamp_flat_hand(ctx.config))
                _log("Auto-clamp: both hands opened")
            ctx.step_counter += 1
            if ctx.step_counter >= 30:
                _set_clamp_phase(ctx, AutoClampPhase.VERIFY_CLEARANCE)

        elif phase == AutoClampPhase.VERIFY_CLEARANCE:
            ctx.clearance_world_z = _fingertip_min_world_z()
            required_z = TABLE_SURFACE_Z + ctx.config.fingertip_table_clearance_m
            if all(value >= required_z for value in ctx.clearance_world_z.values()):
                _log(
                    f"Auto-clamp: fingertip clearance passed "
                    f"{ctx.clearance_world_z}, required={required_z:.3f}"
                )
                _auto_set_hand("both", _clamp_flat_hand(ctx.config))
                _log("Auto-clamp: hands flattened for pregrasp and palm contact")
                _set_clamp_phase(ctx, AutoClampPhase.SOLVE_PREGRASP_LEFT)
                return
            next_roll = ctx.side_raise_roll_rad + ctx.config.side_raise_increment_rad
            if next_roll > ctx.config.max_side_raise_roll_rad + 1e-9:
                _fail_auto_clamp(
                    "clearance_failed",
                    f"fingertips {ctx.clearance_world_z} below {required_z:.3f}m",
                )
                return
            ctx.side_raise_roll_rad = next_roll
            _auto_set_hand("both", CLAMP_TRANSIT_HAND_VALUES)
            _log(
                f"Auto-clamp: clearance insufficient {ctx.clearance_world_z}; "
                f"curl hands and retry shoulder roll ±{next_roll:.2f}rad"
            )
            _set_clamp_phase(ctx, AutoClampPhase.SIDE_RAISE)

        elif phase == AutoClampPhase.SOLVE_PREGRASP_LEFT:
            q, metrics = _solve_clamp_pose(
                "left", ctx.targets["left"]["pregrasp"],
                get_arm_q("left", full=True), "pregrasp",
            )
            ctx.solutions["left"]["pregrasp"] = q
            ctx.metrics["left"]["pregrasp_fk"] = metrics
            _set_clamp_phase(ctx, AutoClampPhase.SOLVE_PREGRASP_RIGHT)

        elif phase == AutoClampPhase.SOLVE_PREGRASP_RIGHT:
            q, metrics = _solve_clamp_pose(
                "right", ctx.targets["right"]["pregrasp"],
                get_arm_q("right", full=True), "pregrasp",
            )
            ctx.solutions["right"]["pregrasp"] = q
            ctx.metrics["right"]["pregrasp_fk"] = metrics
            for arm in ("left", "right"):
                ctx.metrics[arm]["side_to_pregrasp_path"] = (
                    _validate_clamp_joint_path(
                        arm,
                        get_arm_q(arm, full=True),
                        ctx.solutions[arm]["pregrasp"],
                        "pregrasp",
                        ctx.config,
                    )
                )
            _set_clamp_phase(ctx, AutoClampPhase.MOVE_PREGRASP)

        elif phase == AutoClampPhase.MOVE_PREGRASP:
            if not ctx.move_started:
                _start_dual_staged_pregrasp(
                    ctx,
                    {arm: ctx.solutions[arm]["pregrasp"]
                     for arm in ("left", "right")},
                    settle_steps=30,
                )
                ctx.move_started = True
            elif dual_arm_traj is not None:
                clearance = _fingertip_min_world_z()
                if min(clearance.values()) < TABLE_SURFACE_Z - 0.005:
                    _fail_auto_clamp(
                        "table_clearance",
                        f"pregrasp path fingertip clearance lost at "
                        f"waypoint={dual_arm_traj.wp_idx}/"
                        f"{dual_arm_traj.num_waypoints}: {clearance}",
                    )
            else:
                _set_clamp_phase(ctx, AutoClampPhase.VERIFY_PREGRASP)

        elif phase == AutoClampPhase.VERIFY_PREGRASP:
            ok, reason = _verify_pregrasp_actual(ctx)
            if not ok:
                _fail_auto_clamp("pregrasp_mismatch", reason)
                return
            current_object = build_object_pose()["xyz"]
            if math.dist(current_object, ctx.object_snapshot_w) > 0.02:
                _fail_auto_clamp(
                    "object_moved", "object moved more than 2cm before clamping"
                )
                return
            baseline = _auto_clamp_forces()
            if max(values["object_axis_y_n"] for values in baseline.values()) > 0.5:
                _fail_auto_clamp(
                    "unexpected_contact", f"pregrasp hand force is not near zero: {baseline}"
                )
                return
            for arm in ("left", "right"):
                ctx.solutions[arm]["force_start"] = get_arm_q(
                    arm, full=True
                ).detach().clone()
            _set_clamp_phase(ctx, AutoClampPhase.SOLVE_CLAMP_LEFT)

        elif phase == AutoClampPhase.SOLVE_CLAMP_LEFT:
            q, metrics = _solve_clamp_pose(
                "left", ctx.targets["left"]["clamp"],
                ctx.solutions["left"]["pregrasp"], "clamp",
            )
            ctx.solutions["left"]["clamp"] = q
            ctx.metrics["left"]["clamp_fk"] = metrics
            _set_clamp_phase(ctx, AutoClampPhase.SOLVE_CLAMP_RIGHT)

        elif phase == AutoClampPhase.SOLVE_CLAMP_RIGHT:
            q, metrics = _solve_clamp_pose(
                "right", ctx.targets["right"]["clamp"],
                ctx.solutions["right"]["pregrasp"], "clamp",
            )
            ctx.solutions["right"]["clamp"] = q
            ctx.metrics["right"]["clamp_fk"] = metrics
            for arm in ("left", "right"):
                ctx.metrics[arm]["pregrasp_to_clamp_path"] = (
                    _validate_clamp_joint_path(
                        arm,
                        ctx.solutions[arm]["force_start"],
                        ctx.solutions[arm]["clamp"],
                        "clamp",
                        ctx.config,
                    )
                )
            _set_clamp_phase(ctx, AutoClampPhase.FORCE_CLAMP)

        elif phase == AutoClampPhase.FORCE_CLAMP:
            forces = _auto_clamp_forces()
            left_force = forces["left"]["object_axis_y_n"]
            right_force = forces["right"]["object_axis_y_n"]
            contact_seen_before = dict(ctx.contact_seen)
            decision = evaluate_force_control(
                left_force,
                right_force,
                ctx.stable_frames["left"],
                ctx.stable_frames["right"],
                ctx.config,
                ctx.contact_seen["left"],
                ctx.contact_seen["right"],
            )
            ctx.contact_seen["left"] = decision.left_contact_seen
            ctx.contact_seen["right"] = decision.right_contact_seen
            newly_contacted = []
            for arm in ("left", "right"):
                if ctx.contact_seen[arm] and not contact_seen_before[arm]:
                    contact_alpha = ctx.clamp_alpha[arm]
                    ctx.contact_alpha[arm] = contact_alpha
                    newly_contacted.append(arm)
                    ctx.metrics[arm]["contact_probe_alpha"] = round(
                        contact_alpha, 6
                    )
                    _log(
                        f"Auto-clamp: {arm} contact latched at "
                        f"alpha={contact_alpha:.5f}; maintaining continuous "
                        "low preload"
                    )
            if newly_contacted and all(ctx.contact_seen.values()):
                _log(
                    "Auto-clamp: both hands registered object contact; "
                    "entering continuous bilateral squeeze"
                )
            ctx.stable_frames["left"] = decision.left_stable_frames
            ctx.stable_frames["right"] = decision.right_stable_frames
            _store_auto_clamp_forces(ctx, forces)
            broad_norms = torch.linalg.norm(
                contact_hands.data.net_forces_w[0].detach(), dim=-1
            )
            broad_max_n = float(broad_norms.max().cpu().item())
            palm_total_max_n = max(values["palm_n"] for values in forces.values())
            ctx.broad_force_limit_frames = (
                ctx.broad_force_limit_frames + 1
                if broad_max_n >= ctx.config.force_hard_limit_n
                else 0
            )
            # A stiff position actuator can produce a one-frame collision
            # impulse at the exact probe threshold.  Force control commands
            # bounded outward relief immediately; fail only if overload
            # persists for the configured 3-frame (25 ms at 120 Hz) window.
            # The broad sensor includes the palm, so one temporal guard covers
            # both filtered palm load and unfiltered hand-body impacts.
            if (ctx.broad_force_limit_frames
                    >= ctx.config.broad_force_limit_frames):
                _fail_auto_clamp(
                    "force_limit",
                    f"total tactile force exceeded {ctx.config.force_hard_limit_n:.1f}N "
                    f"(broad={broad_max_n:.3f}N for "
                    f"{ctx.broad_force_limit_frames} frames, "
                    f"palm={palm_total_max_n:.3f}N)",
                )
                return
            if decision.error:
                _fail_auto_clamp(
                    decision.error,
                    f"hand force reached safety limit: {ctx.last_forces}",
                )
                return

            actions = {"left": decision.left_action, "right": decision.right_action}
            if (all(ctx.contact_seen.values())
                    and len(ctx.squeeze_anchor_q) < 2):
                for arm in ("left", "right"):
                    # Use the alpha-path command position, not the actual,
                    # so that compression=0 reproduces the same target the
                    # arm was tracking.  Switching to actual would lose the
                    # tracking-error preload and drop contact force.
                    q_start = ctx.solutions[arm]["force_start"]
                    q_end = ctx.solutions[arm]["clamp"]
                    alpha = ctx.clamp_alpha[arm]
                    anchor_q = (q_start + alpha * (q_end - q_start)).detach().clone()
                    ctx.squeeze_anchor_q[arm] = anchor_q
                    ctx.squeeze_jacobian_y[arm] = _squeeze_y_jacobian(
                        arm, anchor_q
                    )
                _log(
                    "Auto-clamp: bilateral object contact; "
                    "contact-local TCP-Y squeeze anchors captured"
                )
            squeeze_active = len(ctx.squeeze_anchor_q) == 2
            squeeze_reanchored = False

            # Two-phase squeeze: detect palm-seating force surge and unload.
            # During the unload countdown, suppress the contact-loss guard
            # because zero-error targets intentionally reduce contact force.
            seating_unloading = (
                squeeze_active
                and ctx.config.squeeze_contact_guard_enabled
                and not ctx.squeeze_seating_complete
                and ctx.squeeze_seat_unload_counter > 0
            )
            if seating_unloading:
                ctx.squeeze_seat_unload_counter -= 1
                actions = {"left": "hold", "right": "hold"}
                if ctx.squeeze_seat_unload_counter == 0:
                    # Unload complete. Re-anchor at measured joints.
                    for arm in ("left", "right"):
                        anchor_q = get_arm_q(
                            arm, full=True
                        ).detach().clone()
                        ctx.squeeze_anchor_q[arm] = anchor_q
                        ctx.squeeze_jacobian_y[arm] = (
                            _squeeze_y_jacobian(arm, anchor_q)
                        )
                        ctx.squeeze_compression_m[arm] = 0.0
                    ctx.squeeze_seating_complete = True
                    ctx.squeeze_contact_loss_frames = 0
                    ctx.stable_frames["left"] = 0
                    ctx.stable_frames["right"] = 0
                    squeeze_reanchored = True
                    _log(
                        "Auto-clamp: palm seating unload complete; "
                        "anchors reset at actual joints for "
                        "normal-gain loading"
                    )
            elif (squeeze_active
                    and ctx.config.squeeze_contact_guard_enabled
                    and not ctx.squeeze_seating_complete):
                object_force_by_arm = {
                    arm: forces[arm]["object_axis_y_n"]
                    for arm in ("left", "right")
                }
                if squeeze_seat_detected(object_force_by_arm, ctx.config):
                    # Seating surge detected — start unload countdown.
                    ctx.squeeze_seat_unload_counter = (
                        ctx.config.squeeze_seat_unload_frames
                    )
                    actions = {"left": "hold", "right": "hold"}
                    ctx.stable_frames["left"] = 0
                    ctx.stable_frames["right"] = 0
                    max_force = max(object_force_by_arm.values())
                    _log(
                        f"Auto-clamp: palm seating surge detected "
                        f"(max={max_force:.1f}N); freezing for "
                        f"{ctx.config.squeeze_seat_unload_frames} frames"
                    )
                elif decision.success:
                    # Forces reached the target band smoothly at the
                    # seating gain without a detectable surge.  The palm
                    # is stably loaded — transition to full-gain loading
                    # via the same re-anchor.
                    for arm in ("left", "right"):
                        anchor_q = get_arm_q(
                            arm, full=True
                        ).detach().clone()
                        ctx.squeeze_anchor_q[arm] = anchor_q
                        ctx.squeeze_jacobian_y[arm] = (
                            _squeeze_y_jacobian(arm, anchor_q)
                        )
                        ctx.squeeze_compression_m[arm] = 0.0
                    ctx.squeeze_seating_complete = True
                    ctx.stable_frames["left"] = 0
                    ctx.stable_frames["right"] = 0
                    squeeze_reanchored = True
                    _log(
                        "Auto-clamp: seating complete (smooth force "
                        "convergence); re-anchored for normal-gain loading"
                    )

            squeeze_paused = False
            if (squeeze_active
                    and not seating_unloading
                    and squeeze_contact_guard_should_pause(
                        {arm: forces[arm]["object_axis_y_n"]
                         for arm in ("left", "right")},
                        ctx.squeeze_compression_m,
                        ctx.config,
                    )):
                squeeze_paused = True
                actions = {"left": "hold", "right": "hold"}
                ctx.stable_frames["left"] = 0
                ctx.stable_frames["right"] = 0
                ctx.squeeze_contact_loss_frames += 1
                if (ctx.squeeze_contact_loss_frames
                        >= ctx.config.squeeze_contact_loss_frames):
                    _fail_auto_clamp(
                        "squeeze_contact_lost",
                        "palm squeeze contact remained below "
                        f"{ctx.config.squeeze_contact_loss_n:.3f}N for "
                        f"{ctx.squeeze_contact_loss_frames} frames: "
                        f"{ctx.last_forces}",
                    )
                    return
            elif (squeeze_active
                    and not seating_unloading
                    and ctx.squeeze_contact_loss_frames):
                # Transient sensor dropout recovered. Re-anchor at the
                # held physical pose to prevent the old virtual-compression
                # target from releasing stored energy as a snap impulse.
                for arm in ("left", "right"):
                    anchor_q = get_arm_q(arm, full=True).detach().clone()
                    ctx.squeeze_anchor_q[arm] = anchor_q
                    ctx.squeeze_jacobian_y[arm] = _squeeze_y_jacobian(
                        arm, anchor_q
                    )
                    ctx.squeeze_compression_m[arm] = 0.0
                ctx.squeeze_contact_loss_frames = 0
                squeeze_reanchored = True
                _log("Auto-clamp: squeeze contact recovered; anchors reset")
            squeeze_frozen = (
                squeeze_paused
                or seating_unloading
            )
            rendezvous_active = (
                any(ctx.contact_seen.values())
                and not all(ctx.contact_seen.values())
            )
            for arm, action in actions.items():
                q_start = ctx.solutions[arm]["force_start"]
                q_end = ctx.solutions[arm]["clamp"]
                current_alpha = ctx.clamp_alpha[arm]
                pre_xyz = ctx.metrics[arm]["pregrasp_actual"]["actual_xyz"]
                end_xyz = ctx.metrics[arm]["clamp_fk"]["achieved_xyz"]
                command_alpha = current_alpha
                ctx.command_alpha[arm] = command_alpha
                current_command = q_start + command_alpha * (q_end - q_start)
                q_actual = get_arm_q(arm, full=True)
                joint_tracking_error = float(torch.max(torch.abs(
                    q_actual - current_command
                )).item())
                path_delta_q = q_end - q_start
                path_norm_sq = torch.dot(path_delta_q, path_delta_q)
                actual_joint_progress = float(torch.clamp(
                    torch.dot(q_actual - q_start, path_delta_q) / path_norm_sq,
                    0.0,
                    1.0,
                ).item())
                expected_xyz = [
                    pre_xyz[i] + command_alpha * (end_xyz[i] - pre_xyz[i])
                    for i in range(3)
                ]
                world_pose = current_world_tool_pose(TOOL_FRAMES[arm])
                actual_xyz, actual_quat = world_to_base(
                    world_pose["xyz"], world_pose["quat_wxyz"]
                )
                actual_progress = path_progress_from_actual_y(
                    pre_xyz[1], end_xyz[1], actual_xyz[1]
                )
                position_tracking_error = math.dist(actual_xyz, expected_xyz)
                inward_dot, down_dot = palm_alignment(actual_quat, arm)
                ctx.metrics[arm]["force_actual_tcp_alpha"] = actual_progress
                ctx.metrics[arm]["force_actual_joint_alpha"] = actual_joint_progress
                ctx.metrics[arm]["force_progress_mode"] = (
                    ctx.config.approach_progress_mode
                )
                ctx.metrics[arm]["force_tracking_error_rad"] = joint_tracking_error
                ctx.metrics[arm]["force_tracking_position_error_m"] = (
                    position_tracking_error
                )
                if squeeze_active:
                    if not squeeze_reanchored and not squeeze_frozen:
                        ctx.squeeze_compression_m[arm] = (
                            updated_squeeze_compression(
                                ctx.squeeze_compression_m[arm],
                                action, ctx.config,
                            )
                        )
                    ctx.metrics[arm]["squeeze_compression_m"] = (
                        ctx.squeeze_compression_m[arm]
                    )
                    continue
                if action == "inward":
                    # A 5-DOF arm cannot eliminate every Cartesian tracking
                    # residual while preserving orientation. Advance along the
                    # fixed, known-object path only a small amount ahead of the
                    # measured TCP; this prevents command alpha from running
                    # inward while a structural thumb contact blocks the hand.
                    if (inward_dot >= ctx.config.hard_min_palm_inward_dot
                            and down_dot >= ctx.config.hard_min_finger_down_dot):
                        requested_step = (
                            ctx.config.rendezvous_alpha_step
                            if rendezvous_active
                            else
                            ctx.config.contact_zone_alpha_step
                            if ctx.clamp_alpha[arm]
                            >= ctx.config.contact_zone_alpha
                            else ctx.config.clamp_alpha_step
                        )
                        ctx.clamp_alpha[arm] = advance_inward_alpha(
                            ctx.clamp_alpha[arm],
                            actual_progress,
                            requested_step,
                            (
                                ctx.config.contact_compression_alpha_lead
                                if all(ctx.contact_seen.values())
                                else ctx.config.max_command_alpha_lead
                            ),
                            ctx.config.approach_progress_mode,
                            joint_tracking_error,
                            ctx.config.tracking_joint_tolerance_rad,
                            ctx.config.tracking_joint_hard_limit_rad,
                            ctx.config.lagging_joint_alpha_step,
                        )
                elif action == "outward":
                    force_n = forces[arm]["object_axis_y_n"]
                    ctx.clamp_alpha[arm] = max(
                        0.0, ctx.clamp_alpha[arm] - (
                            ctx.config.emergency_relief_alpha_step
                            if force_n >= ctx.config.force_hard_limit_n
                            else
                            ctx.config.unilateral_relief_alpha_step
                            if rendezvous_active
                            else ctx.config.relief_alpha_step
                        )
                    )

            alpha_gap = ctx.clamp_alpha["left"] - ctx.clamp_alpha["right"]
            if alpha_gap > ctx.config.max_alpha_difference:
                ctx.clamp_alpha["left"] = (
                    ctx.clamp_alpha["right"] + ctx.config.max_alpha_difference
                )
            elif alpha_gap < -ctx.config.max_alpha_difference:
                ctx.clamp_alpha["right"] = (
                    ctx.clamp_alpha["left"] + ctx.config.max_alpha_difference
                )

            current_object = build_object_pose()
            object_displacement = math.dist(
                current_object["xyz"], ctx.object_snapshot_w
            )
            quat_dot = abs(sum(
                current_object["quat_wxyz"][i] * ctx.object_snapshot_quat_wxyz[i]
                for i in range(4)
            ))
            object_rotation = 2.0 * math.acos(max(-1.0, min(1.0, quat_dot)))
            if (object_displacement > ctx.config.max_object_displacement_m
                    or object_rotation > ctx.config.max_object_rotation_rad):
                _fail_auto_clamp(
                    "object_moved",
                    f"object moved during clamp: displacement={object_displacement:.4f}m, "
                    f"rotation={math.degrees(object_rotation):.2f}deg",
                )
                return

            fingertip_z = _fingertip_min_world_z()
            if min(fingertip_z.values()) < TABLE_SURFACE_Z - 0.005:
                _fail_auto_clamp(
                    "table_clearance",
                    f"force-clamp fingertip entered table region: {fingertip_z}",
                )
                return

            if ctx.step_counter % 30 == 0:
                for arm in ("left", "right"):
                    world_pose = current_world_tool_pose(TOOL_FRAMES[arm])
                    _, base_quat = world_to_base(
                        world_pose["xyz"], world_pose["quat_wxyz"]
                    )
                    inward_dot, down_dot = palm_alignment(base_quat, arm)
                    if (inward_dot < ctx.config.hard_min_palm_inward_dot
                            or down_dot < ctx.config.hard_min_finger_down_dot):
                        _fail_auto_clamp(
                            "orientation_lost",
                            f"{arm} orientation drifted during clamp: "
                            f"inward={inward_dot:.3f}, down={down_dot:.3f}",
                        )
                        return

            for arm, action in actions.items():
                q_actual = get_arm_q(arm, full=True)
                q_end = ctx.solutions[arm]["clamp"]
                joint_error = float(torch.max(torch.abs(q_actual - q_end)).item())
                ctx.metrics[arm]["force_joint_error_rad"] = joint_error
                at_actual_endpoint = (
                    action == "inward"
                    and ctx.clamp_alpha[arm] >= 1.0
                    and ctx.metrics[arm]["force_tracking_position_error_m"]
                    <= ctx.config.endpoint_position_tolerance_m
                )
                ctx.limit_frames[arm] = (
                    ctx.limit_frames[arm] + 1 if at_actual_endpoint else 0
                )

            if max(ctx.limit_frames.values()) >= ctx.config.endpoint_no_contact_frames:
                _fail_auto_clamp(
                    "no_contact",
                    f"clamp path exhausted before both hands reached force band: "
                    f"{ctx.last_forces}",
                )
                return

            target = robot.data.joint_pos_target.clone()
            for arm in ("left", "right"):
                if squeeze_frozen:
                    q_command = get_arm_q(arm, full=True).detach().clone()
                elif squeeze_active:
                    q_command = _squeeze_joint_command(ctx, arm)
                else:
                    q_start = ctx.solutions[arm]["force_start"]
                    q_end = ctx.solutions[arm]["clamp"]
                    alpha = ctx.clamp_alpha[arm]
                    ctx.command_alpha[arm] = alpha
                    q_command = q_start + alpha * (q_end - q_start)
                for i, name in enumerate(arm_joint_names_map[arm]):
                    target[:, name_to_idx[name]] = q_command[i]
            robot.set_joint_position_target(target)

            ctx.step_counter += 1
            if ctx.step_counter % 30 == 0:
                seating_status = (
                    "done" if ctx.squeeze_seating_complete else "phase1"
                )
                if not ctx.config.squeeze_contact_guard_enabled:
                    seating_status = "disabled"
                _log(
                    f"Auto-clamp force: L={left_force:.2f}N "
                    f"R={right_force:.2f}N alpha={ctx.clamp_alpha} "
                    f"command_alpha={ctx.command_alpha} "
                    f"compression={ctx.squeeze_compression_m} "
                    f"stable={ctx.stable_frames} "
                    f"seating={seating_status}"
                )
            if (decision.success
                    and squeeze_active
                    and not squeeze_frozen
                    and not squeeze_reanchored
                    and (not ctx.config.squeeze_contact_guard_enabled
                         or ctx.squeeze_seating_complete)):
                _log(
                    f"Auto-clamp: bilateral {ctx.config.force_target_n:.1f}N "
                    "clamp stable; "
                    "starting unified tactile-gated lift"
                )
                _begin_unified_lift(ctx)

        elif phase == AutoClampPhase.SOLVE_LIFT_LEFT:
            world_pose = current_world_tool_pose(TOOL_FRAMES["left"])
            base_xyz, _ = world_to_base(
                world_pose["xyz"], world_pose["quat_wxyz"]
            )
            target_xyz = (
                base_xyz[0],
                base_xyz[1],
                base_xyz[2] + ctx.config.lift_height_m,
            )
            q, metrics = _solve_clamp_pose(
                "left", target_xyz, get_arm_q("left", full=True), "lift",
            )
            ctx.solutions["left"]["lift"] = q
            ctx.metrics["left"]["lift_fk"] = metrics
            _set_clamp_phase(ctx, AutoClampPhase.SOLVE_LIFT_RIGHT)

        elif phase == AutoClampPhase.SOLVE_LIFT_RIGHT:
            world_pose = current_world_tool_pose(TOOL_FRAMES["right"])
            base_xyz, _ = world_to_base(
                world_pose["xyz"], world_pose["quat_wxyz"]
            )
            target_xyz = (
                ctx.metrics["left"]["lift_fk"]["achieved_xyz"][0],
                base_xyz[1],
                base_xyz[2] + ctx.config.lift_height_m,
            )
            q, metrics = _solve_clamp_pose(
                "right", target_xyz, get_arm_q("right", full=True), "lift",
            )
            ctx.solutions["right"]["lift"] = q
            ctx.metrics["right"]["lift_fk"] = metrics
            _start_dual_linear(
                ctx,
                "auto_clamp_vertical_lift",
                {
                    "left": ctx.solutions["left"]["lift"],
                    "right": ctx.solutions["right"]["lift"],
                },
                steps=ctx.config.lift_steps,
                settle_steps=ctx.config.lift_settle_steps,
                starts_by_arm=ctx.squeeze_anchor_q,
            )
            _set_clamp_phase(ctx, AutoClampPhase.MOVE_LIFT)

        elif phase == AutoClampPhase.MOVE_LIFT:
            forces = _auto_clamp_forces()
            _store_auto_clamp_forces(ctx, forces)
            coordinator = ctx.lift_coordinator
            if coordinator is None:
                _fail_auto_clamp("lift_state", "missing bimanual lift coordinator")
                return
            actual_hand_rise_m = {}
            for arm in ("left", "right"):
                world_pose = current_world_tool_pose(TOOL_FRAMES[arm])
                base_xyz, _ = world_to_base(
                    world_pose["xyz"], world_pose["quat_wxyz"]
                )
                actual_hand_rise_m[arm] = (
                    base_xyz[2] - ctx.lift_start_hand_base_z[arm]
                )
            minimum_actual_rise_m = min(actual_hand_rise_m.values())
            vertical_tracking_ok = (
                coordinator.progress_m
                <= minimum_actual_rise_m
                + ctx.config.lift_max_progress_lead_m
            )
            command = update_bimanual_lift(
                coordinator,
                forces["left"]["object_axis_y_n"],
                forces["right"]["object_axis_y_n"],
                forces["left"]["non_thumb_axis_y_n"],
                forces["right"]["non_thumb_axis_y_n"],
                forces["left"]["object_contact_body_count"],
                forces["right"]["object_contact_body_count"],
                ctx.config,
                vertical_tracking_ok=vertical_tracking_ok,
            )
            ctx.lift_last_command = {
                "advance": command.advance,
                "balanced": command.balanced,
                "delta_z_m": round(command.delta_z_m, 7),
                "left_contact_ok": command.left_contact_ok,
                "right_contact_ok": command.right_contact_ok,
                "vertical_tracking_ok": command.vertical_tracking_ok,
                "soft_rebalance": command.soft_rebalance,
                "hard_limit_frames": ctx.lift_hard_limit_frames,
                "actual_hand_rise_m": {
                    arm: round(value, 6)
                    for arm, value in actual_hand_rise_m.items()
                },
                "error": command.error,
            }
            ctx.lift_hard_limit_frames = (
                ctx.lift_hard_limit_frames + 1
                if command.error is not None else 0
            )
            ctx.lift_last_command["hard_limit_frames"] = (
                ctx.lift_hard_limit_frames
            )
            if (ctx.lift_hard_limit_frames
                    >= ctx.config.lift_hard_limit_frames):
                _fail_auto_clamp(
                    command.error,
                    "raw object-filtered side force persistently exceeded "
                    f"{ctx.config.force_hard_limit_n:.1f}N for "
                    f"{ctx.lift_hard_limit_frames} frames during unified "
                    f"lift: {ctx.last_forces}",
                )
                return
            if coordinator.recovery_frames >= ctx.config.lift_force_drop_frames:
                _fail_auto_clamp(
                    "contact_degraded",
                    "palm/non-thumb contact envelope did not recover during "
                    f"unified lift: {ctx.last_forces}",
                )
                return

            if command.left_contact_ok or command.right_contact_ok:
                full_task = (
                    command.error is None
                    and command.left_contact_ok
                    and command.right_contact_ok
                    and ctx.config.force_lower_n
                    <= command.advance_filtered_left_n
                    <= ctx.config.effective_advance_force_upper_n
                    and ctx.config.force_lower_n
                    <= command.advance_filtered_right_n
                    <= ctx.config.effective_advance_force_upper_n
                    and abs(
                        command.advance_filtered_left_n
                        - command.advance_filtered_right_n
                    ) <= ctx.config.advance_force_balance_tolerance_n
                )
                for arm in ("left", "right"):
                    _advance_unified_lift_nominal(
                        ctx, arm, full_task=full_task
                    )
            target = robot.data.joint_pos_target.clone()
            for arm in ("left", "right"):
                q_command = _unified_lift_joint_command(ctx, arm)
                for index, name in enumerate(arm_joint_names_map[arm]):
                    target[:, name_to_idx[name]] = q_command[index]
            robot.set_joint_position_target(target)

            ctx.step_counter += 1
            if ctx.step_counter % 12 == 0:
                _log(
                    "Auto-clamp unified lift: "
                    f"raw=L{forces['left']['object_axis_y_n']:.2f}/"
                    f"R{forces['right']['object_axis_y_n']:.2f}N "
                    f"filtered=L{command.filtered_left_n:.2f}/"
                    f"R{command.filtered_right_n:.2f}N "
                    f"adv=L{command.advance_filtered_left_n:.2f}/"
                    f"R{command.advance_filtered_right_n:.2f}N "
                    f"progress={coordinator.progress_m:.4f}m "
                    f"squeeze={coordinator.squeeze_offset_m:.5f}m "
                    f"center={coordinator.center_offset_m:.5f}m "
                    f"dir={coordinator.squeeze_direction_counter} "
                    f"advance={command.advance}"
                )
            if coordinator.progress_m >= ctx.config.lift_height_m - 1e-6:
                ctx.lift_settle_counter = (
                    ctx.lift_settle_counter + 1 if command.balanced else 0
                )
                if ctx.lift_settle_counter >= ctx.config.lift_settle_steps:
                    _set_clamp_phase(ctx, AutoClampPhase.VERIFY_LIFT)

        elif phase == AutoClampPhase.VERIFY_LIFT:
            forces = _auto_clamp_forces()
            _store_auto_clamp_forces(ctx, forces)
            object_pose = build_object_pose()
            start_z = ctx.lift_start_object_z
            if start_z is None:
                _fail_auto_clamp("lift_state", "missing lift start object height")
                return
            object_rise = object_pose["xyz"][2] - start_z
            coordinator = ctx.lift_coordinator
            retained = (
                coordinator is not None
                and ctx.config.force_lower_n
                <= coordinator.advance_filtered_left_n
                <= ctx.config.effective_advance_force_upper_n
                and ctx.config.force_lower_n
                <= coordinator.advance_filtered_right_n
                <= ctx.config.effective_advance_force_upper_n
                and all(
                    forces[arm]["non_thumb_axis_y_n"]
                    >= ctx.config.lift_min_non_thumb_force_n
                    for arm in ("left", "right")
                )
            )
            lifted = object_rise >= ctx.config.lift_min_object_rise_m
            ctx.metrics["lift"] = {
                "object_start_z": round(start_z, 5),
                "object_current_z": round(object_pose["xyz"][2], 5),
                "object_rise_m": round(object_rise, 5),
                "retained": retained,
            }
            ctx.lift_verify_frames = (
                ctx.lift_verify_frames + 1 if retained and lifted else 0
            )
            ctx.lift_verify_attempt_frames += 1
            if ctx.step_counter % 15 == 0:
                _log(
                    f"Auto-clamp lift verify: rise={object_rise:.4f}m "
                    f"forces=L{forces['left']['object_axis_y_n']:.2f}/"
                    f"R{forces['right']['object_axis_y_n']:.2f}N "
                    f"stable={ctx.lift_verify_frames}"
                )
            ctx.step_counter += 1
            if ctx.lift_verify_frames >= ctx.config.lift_verify_frames:
                _finish_auto_clamp()
            elif ctx.lift_verify_attempt_frames >= 180:
                _fail_auto_clamp(
                    "lift_not_retained",
                    "unified lift reached its commanded height but did not "
                    f"retain the box at >= {ctx.config.lift_min_object_rise_m:.2f}m: "
                    f"{ctx.metrics['lift']}",
                )

    # ---- State builder ----

    def build_state(reply_to: str | None = None) -> dict:
        pos = robot.data.joint_pos[0].detach().cpu()
        ee = {}
        for arm, frame in TOOL_FRAMES.items():
            body_idx = robot.body_names.index(frame)
            world_xyz = robot.data.body_pos_w[0, body_idx].detach().cpu().tolist()
            world_quat = robot.data.body_quat_w[0, body_idx].detach().cpu().tolist()
            base_xyz, base_quat = world_to_base(world_xyz, world_quat)
            ee[arm] = {
                "base": {
                    "xyz": [round(v, 6) for v in base_xyz],
                    "quat_wxyz": [round(v, 6) for v in base_quat],
                },
                "world": {
                    "xyz": [round(v, 5) for v in world_xyz],
                    "quat_wxyz": [round(v, 5) for v in world_quat],
                },
            }

        return {
            "type": "state",
            "id": reply_to,
            "busy": (
                active_traj is not None or dual_arm_traj is not None
                or auto_grasp_ctx is not None or auto_clamp_ctx is not None
            ),
            "active_command": (
                "auto_clamp" if auto_clamp_ctx is not None
                else "auto_grasp" if auto_grasp_ctx is not None
                else (dual_arm_traj.command if dual_arm_traj is not None
                      else active_traj.command if active_traj is not None
                      else None)
            ),
            "auto_grasp_phase": (
                auto_grasp_ctx.phase.name if auto_grasp_ctx is not None
                else None
            ),
            "auto_clamp_phase": (
                auto_clamp_ctx.phase.name if auto_clamp_ctx is not None
                else None
            ),
            "auto_clamp": (
                {
                    "targets": auto_clamp_ctx.targets,
                    "forces": auto_clamp_ctx.last_forces,
                    "clamp_alpha": auto_clamp_ctx.clamp_alpha,
                    "stable_frames": auto_clamp_ctx.stable_frames,
                    "squeeze_seating_complete": (
                        auto_clamp_ctx.squeeze_seating_complete
                    ),
                    "contact_seen": auto_clamp_ctx.contact_seen,
                    "contact_alpha": auto_clamp_ctx.contact_alpha,
                    "lift_force_drop_frames": (
                        auto_clamp_ctx.lift_force_drop_frames
                    ),
                    "lift_verify_frames": auto_clamp_ctx.lift_verify_frames,
                    "broad_force_limit_frames": (
                        auto_clamp_ctx.broad_force_limit_frames
                    ),
                    "fingertip_min_world_z": auto_clamp_ctx.clearance_world_z,
                    "metrics": auto_clamp_ctx.metrics,
                }
                if auto_clamp_ctx is not None else None
            ),
            "arm_joints": {
                n: round(float(pos[name_to_idx[n]]), 5) for n in all_arm_joint_names
            },
            "hand_joints": {
                n: round(float(pos[name_to_idx[n]]), 5) for n in HAND_ACTIVE_JOINT_NAMES
            },
            "ee": ee,
            "object_pose": build_object_pose(),
            "contact_forces": build_contact_forces(),
        }

    # ---- Command dispatch ----
    dispatch = {
        "move_to_pose": handle_move_to_pose,
        "set_hand": handle_set_hand,
        "home": handle_home,
        "revert": handle_revert,
        "stop": handle_stop,
        "get_state": handle_get_state,
        "auto_grasp": handle_auto_grasp,
        "auto_clamp": handle_auto_clamp,
    }

    # ---- Main simulation loop ----
    last_broadcast = 0.0
    broadcast_interval = 1.0 / STATE_BROADCAST_HZ

    _log("Entering main loop. Waiting for client commands...")

    try:
        while simulation_app.is_running():
            # Process incoming commands (drain queue each step)
            for _ in range(32):
                try:
                    msg = inbox.get_nowait()
                except queue.Empty:
                    break
                cmd = msg.get("type", "")
                handler = dispatch.get(cmd)
                if handler:
                    try:
                        handler(msg)
                    except Exception as exc:
                        reply_error(msg, "exception", str(exc))
                        _log(f"Command {cmd} failed: {exc}")
                else:
                    reply_error(msg, "unknown_command", f"unknown: {cmd}")

            # Execute trajectory waypoint
            try:
                step_trajectory()
            except Exception as exc:
                _log(f"Trajectory execution error: {exc}")
                if active_traj is not None:
                    reply_error(
                        {"id": active_traj.request_id, "type": active_traj.command},
                        "execution_failed", str(exc))
                    active_traj = None
                    pending_home = None
                if auto_grasp_ctx is not None:
                    _fail_auto_grasp("execution_failed",
                                     f"trajectory error: {exc}")

            try:
                step_dual_trajectory()
            except Exception as exc:
                _log(f"Dual trajectory execution error: {exc}")
                if auto_clamp_ctx is not None:
                    _fail_auto_clamp("execution_failed", str(exc))
                else:
                    dual_arm_traj = None

            # Advance auto-grasp state machine
            try:
                step_auto_grasp()
            except Exception as exc:
                _log(f"Auto-grasp error: {exc}")
                if auto_grasp_ctx is not None:
                    _fail_auto_grasp("exception", str(exc))

            try:
                step_auto_clamp()
            except Exception as exc:
                _log(f"Auto-clamp error: {exc}")
                if auto_clamp_ctx is not None:
                    _fail_auto_clamp("exception", str(exc))

            # Sync mimic joints
            sync_mimic()

            # Sim step
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_cfg.dt)
            _capture_record_frame(sim_cfg.dt)

            # Periodic state broadcast
            now = time.monotonic()
            if now - last_broadcast >= broadcast_interval:
                try:
                    server.publish(build_state())
                except Exception as exc:
                    _log(f"State broadcast error: {exc}")
                last_broadcast = now

    finally:
        server.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
