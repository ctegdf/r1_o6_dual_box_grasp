#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI client for R1-o6 pose + grasp interactive server.

Interactive commands to control robot arm poses and dexterous hand grasping.

Usage:
    python pose_grasp_cli.py --host <server-ip> --port 5560

Commands:
    state                       Show current robot state
    move left  0.3 0.2 0.1      Move left hand to xyz (keep orientation)
    move right 0.3 -0.2 0.1 --rpy 0 1.57 0    Move with orientation
    move left  0.3 0.2 0.1 --frame world       Move in world frame
    hand left open              Open left hand
    hand left close             Close left hand
    hand right close 0.7        Close right hand to scale 0.7
    hand left set 0 0.3 0.8 0.8 0.8 0.8        Set 6 finger joint values
    clamp [object]             Two-sided large-box clamp (server object by default)
    home                        Return to default pose
    revert                      Follow last trajectory in reverse
    stop                        Stop current trajectory
    help                        Show this help
    quit                        Exit
"""

from __future__ import annotations

import argparse
import json
import math
import readline  # noqa: F401 — enables input history/editing
import socket
import sys
import threading
import time
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5560

# Predefined hand poses (6 values: thumb_yaw, thumb_pitch, index, middle, ring, pinky)
HAND_PRESETS = {
    "open":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "close":    [0.0, 0.5, 1.4, 1.4, 1.4, 1.4],
    "pregrasp": [0.0, 0.2, 0.6, 0.6, 0.6, 0.6],
    "pinch":    [0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
    "point":    [0.0, 0.4, 0.0, 1.4, 1.4, 1.4],
}


class PoseGraspClient:
    """TCP JSONL client for the pose_grasp_server."""

    def __init__(self, host: str, port: int, verbose: bool = False):
        self.host = host
        self.port = port
        self.verbose = verbose
        self._sock: socket.socket | None = None
        self._buf = b""
        self._hello: dict | None = None
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, dict] = {}
        self._pending_lock = threading.Lock()
        self._last_state: dict | None = None
        self._last_response: dict | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._msg_id = 0

    def connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect((self.host, self.port))
        sock.settimeout(0.5)
        self._sock = sock
        self._stop.clear()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        # Wait for hello
        time.sleep(0.5)
        if self._hello:
            print(f"Connected to {self.host}:{self.port}")
            print(f"  Protocol: {self._hello.get('protocol')}")
            print(f"  Commands: {self._hello.get('commands')}")
        else:
            print(f"Connected to {self.host}:{self.port} (no hello received)")

    def close(self):
        self._stop.set()
        if self._sock:
            self._sock.close()
            self._sock = None

    @property
    def loaded_object(self) -> str:
        """Object selected when the connected server was launched."""

        if self._hello is None:
            return "courier_box_m"
        return str(self._hello.get("object", "courier_box_m"))

    def _next_id(self) -> str:
        self._msg_id += 1
        return f"cli-{self._msg_id}"

    def send(self, msg: dict, wait: bool = True, timeout: float = 30.0) -> dict | None:
        if self._sock is None:
            print("Send error: not connected")
            return None
        mid = self._next_id()
        msg["id"] = mid

        evt = threading.Event() if wait else None
        if wait:
            with self._pending_lock:
                self._pending[mid] = evt

        if self.verbose:
            print(f"  >> {json.dumps(msg)}")

        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        try:
            self._sock.sendall(data)
        except OSError as e:
            if wait:
                with self._pending_lock:
                    self._pending.pop(mid, None)
            print(f"Send error: {e}")
            return None

        if not wait:
            return None

        if evt.wait(timeout):
            with self._pending_lock:
                self._pending.pop(mid, None)
                return self._responses.pop(mid, self._last_response)
        else:
            with self._pending_lock:
                self._pending.pop(mid, None)
                self._responses.pop(mid, None)
            print(f"  Timeout waiting for response (id={mid})")
            return None

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(8192)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._handle_message(msg)

    def _handle_message(self, msg: dict):
        msg_type = msg.get("type")

        if msg_type == "hello":
            self._hello = msg
            return

        if msg_type == "state":
            self._last_state = msg
            mid = msg.get("id")
            if mid:
                with self._pending_lock:
                    if mid in self._pending:
                        self._last_response = msg
                        self._responses[mid] = msg
                        self._pending[mid].set()
            return

        if msg_type == "ack":
            if self.verbose:
                status = msg.get("status", "")
                print(f"  << ack: {msg.get('command')} → {status}")
            return

        if msg_type in ("done", "error"):
            mid = msg.get("id")
            if msg_type == "error":
                print(f"  Error: [{msg.get('code')}] {msg.get('message')}")
            elif msg_type == "done":
                status = msg.get("status", "")
                extra = ""
                if "num_waypoints" in msg:
                    extra += f", {msg['num_waypoints']} waypoints"
                if "elapsed_sec" in msg:
                    extra += f", {msg['elapsed_sec']}s"
                print(f"  Done: {msg.get('command')} → {status}{extra}")
            with self._pending_lock:
                self._last_response = msg
                if mid and mid in self._pending:
                    self._responses[mid] = msg
                    self._pending[mid].set()
            return


def format_pose(pose_dict: dict, label: str = "") -> str:
    xyz = pose_dict.get("xyz", [0, 0, 0])
    quat = pose_dict.get("quat_wxyz", [1, 0, 0, 0])
    prefix = f"{label}: " if label else ""
    return (f"{prefix}xyz=({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f})  "
            f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")


def print_state(state: dict):
    if not state:
        print("  No state available")
        return
    busy = state.get("busy", False)
    active = state.get("active_command")
    ag_phase = state.get("auto_grasp_phase")
    ac_phase = state.get("auto_clamp_phase")
    status = "BUSY" if busy else "idle"
    if active:
        status += f" ({active})"
    if ag_phase:
        status += f" [{ag_phase}]"
    if ac_phase:
        status += f" [{ac_phase}]"
    print(f"  Status: {status}")

    ee = state.get("ee", {})
    for arm in ("left", "right"):
        if arm in ee:
            base_pose = ee[arm].get("base", {})
            world_pose = ee[arm].get("world", {})
            print(f"  {arm.upper()} hand:")
            print(f"    {format_pose(base_pose, 'base ')}")
            print(f"    {format_pose(world_pose, 'world')}")

    obj_pose = state.get("object_pose", {})
    if obj_pose:
        print(f"  Object: {format_pose(obj_pose, 'world')}")

    contact = state.get("contact_forces", {})
    if contact:
        hands = contact.get("contact_hands", {})
        if hands:
            cnt = hands.get("contact_body_count", 0)
            maxf = hands.get("max_force_n", 0.0)
            print(f"  Contact: {cnt} bodies, max={maxf:.3f} N")

        ft = contact.get("fingertips_object_n", {})
        if ft:
            _fingers = ("thumb", "index", "middle", "ring", "pinky")

            def _side(prefix):
                return ", ".join(
                    f"{f}={ft.get(f'{prefix}_{f}_distal', 0.0):.2f}"
                    for f in _fingers
                )

            print(f"  Fingertip→Object L (N): {_side('lh')}")
            print(f"  Fingertip→Object R (N): {_side('rh')}")

        palms = contact.get("auto_clamp_hands", {})
        if palms:
            left = palms.get("left", {})
            right = palms.get("right", {})
            print(
                "  Hand→Object clamp-axis total (N): "
                f"L={left.get('object_axis_y_n', 0.0):.2f}, "
                f"R={right.get('object_axis_y_n', 0.0):.2f} "
                f"(palm L/R={left.get('palm_axis_y_n', 0.0):.2f}/"
                f"{right.get('palm_axis_y_n', 0.0):.2f})"
            )

    hand = state.get("hand_joints", {})
    if hand:
        left_vals = [hand.get(f"lh_{s}", 0) for s in
                     ("thumb_cmc_yaw", "thumb_cmc_pitch", "index_mcp_pitch",
                      "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch")]
        right_vals = [hand.get(f"rh_{s}", 0) for s in
                      ("thumb_cmc_yaw", "thumb_cmc_pitch", "index_mcp_pitch",
                       "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch")]
        print(f"  Hand L: [{', '.join(f'{v:.2f}' for v in left_vals)}]")
        print(f"  Hand R: [{', '.join(f'{v:.2f}' for v in right_vals)}]")


def show_help():
    print("""
Commands:
  state                             Show current robot state
  move <arm> <x> <y> <z>            Move arm to xyz (base frame, keep orientation)
  move <arm> <x> <y> <z> --rpy <r> <p> <y>    With orientation (roll pitch yaw)
  move <arm> <x> <y> <z> --frame world         In world frame
  hand <side> open                  Open hand (preset)
  hand <side> close [scale]         Close hand (preset, optional scale 0-1)
  hand <side> pregrasp              Pre-grasp pose (preset)
  hand <side> pinch                 Pinch pose (preset)
  hand <side> set <v1> ... <v6>     Set 6 finger values directly
  grasp [arm] [object]              Auto-grasp sequence (default: left, unknown)
  clamp [object]                    Two-sided large-box force-controlled clamp
                                      (default: object loaded by server)
  home                              Return arms to default, open hands
  revert                            Follow last arm trajectory in reverse
  stop                              Stop trajectory or cancel automatic motion
  verbose                           Toggle verbose mode
  help                              Show this help
  quit / exit                       Exit

Arms: left, right
Sides: left, right, both

Coordinate frames:
  base  = pelvis_link local frame (default)
  world = simulation world frame

Hand joints (per hand, 6 values):
  thumb_cmc_yaw, thumb_cmc_pitch, index_mcp_pitch,
  middle_mcp_pitch, ring_mcp_pitch, pinky_mcp_pitch
""")


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Pose + grasp CLI client")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main():
    args = parse_cli_args()
    client = PoseGraspClient(args.host, args.port, verbose=args.verbose)

    try:
        client.connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    show_help()

    try:
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("quit", "exit", "q"):
                break

            elif cmd == "help":
                show_help()

            elif cmd == "verbose":
                client.verbose = not client.verbose
                print(f"  Verbose: {'on' if client.verbose else 'off'}")

            elif cmd == "state":
                result = client.send({"type": "get_state"})
                if result:
                    print_state(result)

            elif cmd == "stop":
                client.send({"type": "stop"})

            elif cmd == "home":
                steps = 80
                if len(parts) > 1:
                    try:
                        steps = int(parts[1])
                    except ValueError:
                        pass
                client.send({"type": "home", "steps": steps})

            elif cmd == "revert":
                client.send({"type": "revert"})

            elif cmd == "move":
                if len(parts) < 5:
                    print("  Usage: move <arm> <x> <y> <z> [--rpy r p y] [--frame base|world]")
                    continue
                arm = parts[1].lower()
                try:
                    xyz = [float(parts[2]), float(parts[3]), float(parts[4])]
                except ValueError:
                    print("  Invalid xyz coordinates")
                    continue

                msg: dict[str, Any] = {
                    "type": "move_to_pose",
                    "arm": arm,
                    "pose": {"xyz": xyz},
                }

                # Parse optional flags
                i = 5
                while i < len(parts):
                    if parts[i] == "--rpy" and i + 3 < len(parts):
                        try:
                            rpy = [float(parts[i+1]), float(parts[i+2]), float(parts[i+3])]
                            msg["pose"]["rpy"] = rpy
                        except ValueError:
                            print("  Invalid rpy values")
                        i += 4
                    elif parts[i] == "--quat" and i + 4 < len(parts):
                        try:
                            q = [float(parts[i+j]) for j in range(1, 5)]
                            msg["pose"]["quat_wxyz"] = q
                        except ValueError:
                            print("  Invalid quaternion values")
                        i += 5
                    elif parts[i] == "--frame" and i + 1 < len(parts):
                        msg["frame"] = parts[i+1]
                        i += 2
                    else:
                        i += 1

                client.send(msg)

            elif cmd == "hand":
                if len(parts) < 3:
                    print("  Usage: hand <side> <preset|set v1..v6>")
                    continue
                side = parts[1].lower()
                subcmd = parts[2].lower()

                if subcmd == "set":
                    if len(parts) < 9:
                        print("  Usage: hand <side> set <v1> <v2> <v3> <v4> <v5> <v6>")
                        continue
                    try:
                        values = [float(parts[3+i]) for i in range(6)]
                    except (ValueError, IndexError):
                        print("  Invalid values")
                        continue
                    client.send({"type": "set_hand", "hand": side, "values": values},
                                wait=True, timeout=5)

                elif subcmd in HAND_PRESETS:
                    values = list(HAND_PRESETS[subcmd])
                    # Optional scale factor for close
                    if subcmd == "close" and len(parts) > 3:
                        try:
                            scale = float(parts[3])
                            values = [v * scale for v in values]
                        except ValueError:
                            pass
                    client.send({"type": "set_hand", "hand": side, "values": values},
                                wait=True, timeout=5)

                else:
                    print(f"  Unknown hand preset: {subcmd}")
                    print(f"  Available: {', '.join(HAND_PRESETS.keys())}, set")

            elif cmd == "grasp":
                arm = parts[1].lower() if len(parts) > 1 else "left"
                obj_key = parts[2] if len(parts) > 2 else "unknown"
                result = client.send(
                    {"type": "auto_grasp", "arm": arm, "object": obj_key},
                    wait=True, timeout=120)
                if result and result.get("type") == "done":
                    print(f"  Grasp complete: {result.get('status')}")

            elif cmd in ("clamp", "auto_clamp"):
                object_key = parts[1] if len(parts) > 1 else client.loaded_object
                result = client.send(
                    {"type": "auto_clamp", "object": object_key},
                    # Camera recording plus deterministic GPU IK can exceed
                    # two wall-clock minutes even though simulated motion is
                    # short.  Keep the client attached for the full chain.
                    wait=True, timeout=1300,
                )
                if result and result.get("type") == "done":
                    forces = result.get("forces", {})
                    print(f"  Clamp complete: {result.get('status')} forces={forces}")

            else:
                print(f"  Unknown command: {cmd}. Type 'help' for usage.")

    finally:
        client.close()
        print("Disconnected.")


if __name__ == "__main__":
    main()
