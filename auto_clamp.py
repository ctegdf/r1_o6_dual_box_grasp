#!/usr/bin/env python3
"""Pure geometry and force-control rules for large-box bimanual clamping.

This module intentionally has no Isaac Lab or CUDA imports.  The simulator
server owns FK, sensing, and joint commands; this file keeps the hard-coded
contract small enough to test offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal, Mapping, Sequence


ForceAction = Literal["inward", "hold", "outward"]


@dataclass(frozen=True)
class ClampConfig:
    """Geometry and control parameters for one large-box clamp profile."""

    box_size_xyz_m: tuple[float, float, float] = (0.30, 0.22, 0.20)
    pregrasp_clearance_m: float = 0.07
    # The O6 hand-base frame sits behind the physical palm face by roughly
    # 30 mm; use a bounded inward bias so a flat hand reaches the carton.
    clamp_surface_margin_m: float = -0.069
    palm_center_offset_z_m: float = 0.10
    max_clamp_travel_m: float = 0.15
    fingertip_table_clearance_m: float = 0.03
    fingertip_tip_offset_m: float = 0.055
    initial_side_raise_roll_rad: float = 0.75
    max_side_raise_roll_rad: float = 1.40
    side_raise_increment_rad: float = 0.10
    min_palm_inward_dot: float = 0.95
    min_finger_down_dot: float = 0.75
    hard_min_palm_inward_dot: float = 0.70
    hard_min_finger_down_dot: float = 0.60
    max_pose_error_m: float = 0.025
    # Thumb CMC overrides for the flat-palm posture.  Higher values curl the
    # thumb out of the approach path so that the palm face contacts the box
    # first, preventing early thumb-only contact on light objects.
    flat_hand_thumb_yaw_rad: float = 0.0
    flat_hand_thumb_pitch_rad: float = 0.0
    # Finger MCP override for the flat-palm posture.  Higher values curl
    # the four fingers inward against the box side, increasing the contact
    # area and therefore the friction available during vertical lift.
    flat_hand_finger_mcp_rad: float = 0.0
    # When the curled thumb is the primary clamping surface, count all
    # thumb bodies as structural (non-thumb) for the lift contact-quality
    # gate so that the force split between thumb_distal and
    # thumb_metacarpals does not cause spurious contact_degraded failures.
    thumb_is_structural: bool = False
    force_target_n: float = 10.0
    force_tolerance_n: float = 1.0
    # Before both palms have touched the box, keep the first-contact side at a
    # light preload instead of driving it to the profile's final force target.
    # A full unilateral load slides the free box away from the other palm.
    contact_detect_n: float = 0.2
    unilateral_preload_upper_n: float = 0.8
    force_hard_limit_n: float = 15.0
    lift_soft_force_limit_n: float = 12.5
    force_stable_frames: int = 12
    clamp_alpha_step: float = 0.0004
    contact_zone_alpha: float = 0.50
    contact_zone_alpha_step: float = 0.0001
    max_command_alpha_lead: float = 0.025
    approach_progress_mode: Literal["tcp_y", "joint_path"] = "tcp_y"
    # Once both structural hand envelopes touch, measured TCP position should
    # stop at the rigid box.  A larger but bounded command lead then acts as
    # virtual spring compression so the restored arm actuators can build load.
    contact_compression_alpha_lead: float = 0.20
    squeeze_compression_step_m: float = 0.00002
    squeeze_relief_step_m: float = 0.0005
    max_squeeze_compression_m: float = 0.03
    squeeze_joint_target_gain: float = 6.0
    # When enabled, the server detects the palm-seating force surge (a
    # one-frame jump from ~2N to ~100N when the palm face suddenly contacts
    # the box) and immediately freezes both arms at their measured joints.
    # After the unload countdown the server re-anchors at the measured pose
    # with zero compression, so subsequent squeeze builds force smoothly
    # against the already-seated palm.
    squeeze_seat_detect_n: float = 12.0
    squeeze_seat_unload_frames: int = 10
    # When enabled, squeeze anchors are not captured until both palms report
    # force >= contact_detect_n, and squeeze pauses (both arms frozen at
    # their measured joint positions) when either side drops below
    # squeeze_contact_loss_n after compression exceeds the startup minimum.
    # Persistent loss beyond squeeze_contact_loss_frames triggers failure.
    squeeze_contact_guard_enabled: bool = False
    squeeze_contact_loss_n: float = 0.05
    squeeze_contact_loss_min_compression_m: float = 0.001
    squeeze_contact_loss_frames: int = 30
    rendezvous_alpha_step: float = 0.0002
    unilateral_relief_alpha_step: float = 0.0001
    relief_alpha_step: float = 0.004
    emergency_relief_alpha_step: float = 0.04
    broad_force_limit_frames: int = 3
    phase_timeout_s: float = 30.0
    force_phase_timeout_s: float = 70.0
    total_timeout_s: float = 90.0
    endpoint_joint_tolerance_rad: float = 0.08
    tracking_joint_tolerance_rad: float = 0.08
    tracking_joint_hard_limit_rad: float = 0.12
    lagging_joint_alpha_step: float = 0.00004
    # The 5-DOF arm settles with ~15.2 mm task-space error while preserving
    # the strict palm orientation. Allow that measured tracking envelope so
    # force-guided inward motion does not deadlock halfway.
    tracking_position_tolerance_m: float = 0.022
    tracking_min_palm_inward_dot: float = 0.75
    tracking_min_finger_down_dot: float = 0.65
    endpoint_position_tolerance_m: float = 0.020
    endpoint_no_contact_frames: int = 120
    # Joint-space progress is not geometrically symmetric: the two arms reach
    # the same box face with different IK solutions and contact compliance.
    # Leave enough room for the lagging hand to close while object-pose and
    # force hard limits remain the authoritative safety constraints.
    max_alpha_difference: float = 0.20
    max_object_displacement_m: float = 0.02
    max_object_rotation_rad: float = math.radians(10.0)
    lift_height_m: float = 0.16
    lift_steps: int = 800
    lift_settle_steps: int = 60
    lift_verify_frames: int = 30
    lift_min_object_rise_m: float = 0.15
    lift_min_force_n: float = 4.0
    lift_force_drop_frames: int = 60
    lift_hard_limit_frames: int = 3
    lift_squeeze_inward_step_m: float = 0.00004
    lift_squeeze_outward_step_m: float = 0.00008
    force_filter_alpha: float = 0.08
    # Preserve the responsive force estimate for squeeze/centre control while
    # allowing the lift-progress gate to reject long-period contact resonance.
    # The default matches the primary filter to preserve the validated courier
    # profile; resonance-prone profiles opt into a slower gate filter.
    advance_force_filter_alpha: float = 0.08
    # None preserves the target-band upper edge.  Profiles with a measured
    # higher lift-force equilibrium may widen the slow-average gate without
    # weakening the responsive soft-limit controller.
    advance_force_upper_n: float | None = None
    # Begin pair-centre load transfer before a visible unilateral overload
    # develops.  The contact model reacts one to two rendered frames after a
    # TCP correction, so a wide dead band makes the later correction chase
    # stale force samples.
    lift_force_balance_tolerance_n: float = 0.5
    # Gate-only tolerance for the slow advance force estimates.  Separate from
    # the responsive centre controller's dead band because the slow filter's
    # anti-phase residual can exceed the centre controller's tolerance.
    advance_force_balance_tolerance_n: float = 0.5
    lift_raw_force_imbalance_soft_n: float = 2.0
    # At 120 Hz, eight stable samples leave one full 30 Hz observation
    # interval between 0.2 mm height increments.  This prevents consecutive
    # lift targets from accumulating before the contact solver has exposed a
    # lateral load-transfer impulse.
    lift_progress_stable_frames: int = 8
    lift_squeeze_persist_frames: int = 180
    lift_squeeze_slew_m: float = 0.00004
    lift_center_slew_m: float = 0.00002
    lift_emergency_center_slew_multiplier: float = 2.0
    lift_emergency_squeeze_relief_m: float = 0.002
    lift_emergency_center_step_m: float = 0.0002
    lift_max_squeeze_offset_m: float = 0.075
    lift_max_release_offset_m: float = 0.02
    lift_max_center_offset_m: float = 0.02
    # Position-controlled arms need about one centimetre of target lead to
    # generate upward effort while supporting the box.  The observed
    # right-arm steady-state lag is 9.6 mm near full height, so retain a small
    # margin while force/contact gates still stop progress independently.
    lift_max_progress_lead_m: float = 0.012
    lift_min_non_thumb_force_n: float = 3.0
    lift_recovery_min_non_thumb_force_n: float = 0.5
    lift_min_contact_body_count: int = 2
    lift_min_palm_inward_dot: float = 0.85
    lift_min_finger_down_dot: float = 0.20
    # During lift, vertical height and the box-normal coordinate matter more
    # than preserving the exact fore/aft TCP coordinate.  The 5-DOF arm has
    # no exact 16 cm Cartesian solution at the clamped orientation, so permit
    # a common X drift while keeping Y contact and Z lift tightly bounded.
    lift_max_x_drift_m: float = 0.065
    lift_max_normal_error_m: float = 0.02
    lift_max_z_error_m: float = 0.01

    def __post_init__(self) -> None:
        if (
            len(self.box_size_xyz_m) != 3
            or not all(
                math.isfinite(float(value)) and float(value) > 0.0
                for value in self.box_size_xyz_m
            )
        ):
            raise ValueError("box_size_xyz_m must contain three positive dimensions")
        if not isinstance(self.force_stable_frames, int) or self.force_stable_frames < 1:
            raise ValueError("force_stable_frames must be a positive integer")
        if (not isinstance(self.broad_force_limit_frames, int)
                or self.broad_force_limit_frames < 1):
            raise ValueError("broad_force_limit_frames must be a positive integer")
        for name, value in (
            ("lift_steps", self.lift_steps),
            ("lift_settle_steps", self.lift_settle_steps),
            ("lift_verify_frames", self.lift_verify_frames),
            ("lift_force_drop_frames", self.lift_force_drop_frames),
            ("lift_hard_limit_frames", self.lift_hard_limit_frames),
            ("lift_min_contact_body_count", self.lift_min_contact_body_count),
            ("lift_progress_stable_frames", self.lift_progress_stable_frames),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.force_tolerance_n <= 0.0 or self.force_target_n <= self.force_tolerance_n:
            raise ValueError("force target/tolerance define an invalid force band")
        if self.force_hard_limit_n <= self.force_upper_n:
            raise ValueError("force hard limit must be above the target band")
        if not self.force_upper_n < self.lift_soft_force_limit_n < self.force_hard_limit_n:
            raise ValueError("lift soft force limit must sit below the hard limit")
        if self.advance_force_upper_n is not None and not (
            math.isfinite(float(self.advance_force_upper_n))
            and self.force_upper_n
            <= float(self.advance_force_upper_n)
            < self.force_hard_limit_n
        ):
            raise ValueError(
                "advance force upper bound must extend the target band and "
                "stay below the hard limit"
            )
        if not 0.0 < self.contact_detect_n < self.unilateral_preload_upper_n:
            raise ValueError("unilateral contact thresholds must be ordered")
        if self.unilateral_preload_upper_n >= self.force_lower_n:
            raise ValueError("unilateral preload must stay below the final force band")
        if not self.relief_alpha_step < self.emergency_relief_alpha_step < 1.0:
            raise ValueError("emergency relief must exceed normal relief")
        if not 0.0 < self.max_command_alpha_lead < 1.0:
            raise ValueError("max command alpha lead must be between zero and one")
        if self.approach_progress_mode not in ("tcp_y", "joint_path"):
            raise ValueError("approach progress mode must be 'tcp_y' or 'joint_path'")
        if not (
            0.0 < self.tracking_joint_tolerance_rad
            < self.tracking_joint_hard_limit_rad
        ):
            raise ValueError("joint tracking thresholds must be positive and ordered")
        if not 0.0 < self.lagging_joint_alpha_step <= self.clamp_alpha_step:
            raise ValueError("lagging joint alpha step must fit the normal alpha step")
        if not self.max_command_alpha_lead < self.contact_compression_alpha_lead < 1.0:
            raise ValueError("contact compression lead must exceed free-space lead")
        if not (
            0.0 < self.squeeze_compression_step_m
            < self.squeeze_relief_step_m
            < self.max_squeeze_compression_m
        ):
            raise ValueError("squeeze compression and relief steps must be ordered")
        if not math.isfinite(self.squeeze_joint_target_gain) or self.squeeze_joint_target_gain <= 0.0:
            raise ValueError("squeeze joint target gain must be positive")
        if not (
            math.isfinite(self.squeeze_seat_detect_n)
            and self.contact_detect_n
            < self.squeeze_seat_detect_n
            < self.force_hard_limit_n
        ):
            raise ValueError(
                "squeeze seat detect threshold must lie between contact "
                "detection and the force hard limit"
            )
        if (
            not isinstance(self.squeeze_seat_unload_frames, int)
            or isinstance(self.squeeze_seat_unload_frames, bool)
            or self.squeeze_seat_unload_frames < 1
        ):
            raise ValueError(
                "squeeze seat unload frames must be a positive integer"
            )
        if not isinstance(self.squeeze_contact_guard_enabled, bool):
            raise ValueError("squeeze contact guard flag must be boolean")
        if (not isinstance(self.squeeze_contact_loss_frames, int)
                or isinstance(self.squeeze_contact_loss_frames, bool)
                or self.squeeze_contact_loss_frames < 1):
            raise ValueError("squeeze contact loss frames must be a positive integer")
        if not (
            math.isfinite(self.squeeze_contact_loss_n)
            and self.squeeze_contact_loss_n > 0.0
            and 0.0 < self.squeeze_contact_loss_min_compression_m
            < self.max_squeeze_compression_m
        ):
            raise ValueError("squeeze contact guard thresholds are invalid")
        if not 0.0 < self.lift_min_object_rise_m <= self.lift_height_m:
            raise ValueError("lift rise threshold must fit inside lift height")
        if not 0.0 < self.lift_min_force_n < self.force_lower_n:
            raise ValueError("lift retention force must stay below clamp force band")
        if not (
            0.0 < self.lift_squeeze_inward_step_m
            <= self.lift_squeeze_outward_step_m
            < self.max_squeeze_compression_m
        ):
            raise ValueError("lift squeeze steps must be positive and ordered")
        for name, value in (
            ("force_filter_alpha", self.force_filter_alpha),
            ("advance_force_filter_alpha", self.advance_force_filter_alpha),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be inside (0, 1]")
        if self.lift_squeeze_persist_frames < 1:
            raise ValueError("squeeze persist frames must be positive")
        if not 0.0 < self.lift_force_balance_tolerance_n < self.force_target_n:
            raise ValueError("lift force balance tolerance is invalid")
        if not 0.0 < self.advance_force_balance_tolerance_n:
            raise ValueError("advance force balance tolerance must be positive")
        if not (
            self.lift_force_balance_tolerance_n
            < self.lift_raw_force_imbalance_soft_n
            < self.force_target_n
        ):
            raise ValueError("raw lift imbalance threshold is invalid")
        if not (
            0.0 < self.lift_squeeze_slew_m < self.lift_max_squeeze_offset_m
            and 0.0 < self.lift_center_slew_m < self.lift_max_center_offset_m
            and self.lift_squeeze_slew_m < self.lift_max_release_offset_m
        ):
            raise ValueError("bimanual lift offset limits are invalid")
        if not 0.0 < self.lift_max_progress_lead_m < self.lift_height_m:
            raise ValueError("lift progress lead is invalid")
        if self.lift_emergency_center_slew_multiplier <= 1.0:
            raise ValueError("emergency centre slew multiplier must exceed one")
        if not (
            self.lift_squeeze_outward_step_m
            < self.lift_emergency_squeeze_relief_m
            < self.lift_max_release_offset_m
            and self.lift_center_slew_m
            < self.lift_emergency_center_step_m
            < self.lift_max_center_offset_m
        ):
            raise ValueError("emergency lift relief steps are invalid")
        if not 0.0 < self.lift_min_non_thumb_force_n < self.force_lower_n:
            raise ValueError("lift non-thumb contact threshold is invalid")
        if not (
            0.0 < self.lift_recovery_min_non_thumb_force_n
            < self.lift_min_non_thumb_force_n
        ):
            raise ValueError("lift recovery contact threshold is invalid")
        if not (
            self.hard_min_palm_inward_dot
            <= self.tracking_min_palm_inward_dot
            <= self.min_palm_inward_dot
        ):
            raise ValueError("palm orientation thresholds must be ordered")
        if not (
            self.hard_min_finger_down_dot
            <= self.tracking_min_finger_down_dot
            <= self.min_finger_down_dot
        ):
            raise ValueError("finger orientation thresholds must be ordered")
        if not self.hard_min_palm_inward_dot <= self.lift_min_palm_inward_dot <= 1.0:
            raise ValueError("lift palm orientation must stay inside the hard limit")
        if not -1.0 <= self.lift_min_finger_down_dot <= 1.0:
            raise ValueError("lift finger orientation threshold is invalid")
        if not (
            0.0 < self.lift_max_z_error_m
            <= self.lift_max_normal_error_m
            <= self.lift_max_x_drift_m
        ):
            raise ValueError("lift Cartesian tolerances must be positive and ordered")

    @property
    def force_lower_n(self) -> float:
        return self.force_target_n - self.force_tolerance_n

    @property
    def force_upper_n(self) -> float:
        return self.force_target_n + self.force_tolerance_n

    @property
    def effective_advance_force_upper_n(self) -> float:
        """Upper slow-average force admitted by the lift advance gate."""

        return (
            self.force_upper_n
            if self.advance_force_upper_n is None
            else float(self.advance_force_upper_n)
        )


AUTO_CLAMP_OBJECT_KEYS = ("courier_box_m", "foam_box")

# Keep the validated courier-box V50 controller intact.  Object profiles vary
# only where simulator evidence justifies tuning.  The EPS foam profile raises
# contact detection to 1.0 N to reject free-space sensor noise (~0.6 N) and uses
# a slower, longer contact-local squeeze for progressive palm seating.
AUTO_CLAMP_PROFILES: Mapping[str, ClampConfig] = MappingProxyType({
    "courier_box_m": ClampConfig(
        box_size_xyz_m=(0.30, 0.22, 0.20),
        palm_center_offset_z_m=0.10,
        force_target_n=10.5,
    ),
    "foam_box": ClampConfig(
        box_size_xyz_m=(0.35, 0.24, 0.25),
        # Y=0.24 keeps the foam box only 1 cm wider per side than the
        # courier box, giving >34 mm elbow clearance.  The original 0.28 m
        # width left only ~8 mm and failed during contact-patch compression.
        palm_center_offset_z_m=0.10,
        force_target_n=11.0,
        force_tolerance_n=2.0,
        # Wider band (9-13 N) keeps forces at the upper edge (~12 N) in-band
        # so the lift advance gate fires reliably.  The soft force limit sits
        # well above the upper band to allow headroom.
        lift_soft_force_limit_n=14.0,
        # The asymmetric palm-seating snap displaces the box ~30 mm.
        # Allow 50 mm so the opposite palm can seat and partially reverse
        # the initial shift.
        max_object_displacement_m=0.05,
        # Free-space sensor noise reached ~0.6 N on EPS foam; require 1.0 N
        # before latching contact and capturing squeeze anchors.  A 3.0 N
        # unilateral preload stays below the final force band (9-13 N).
        contact_detect_n=1.0,
        unilateral_preload_upper_n=3.0,
        # Slight finger curl distributes contact across fingertips before
        # the palm seats, helping progressive contact buildup.
        flat_hand_finger_mcp_rad=0.3,
        # Low-gain squeeze: gain=1.0 with 100 mm range approaches the palm
        # seating point (~81 mm) at 0.02 mm/step = 2.4 mm/s at 120 Hz.
        # The object-filtered palm force at seating is only ~2 N, but the
        # broad (unfiltered) sensor reads ~94 N from hand self-collision.
        # A 100 N / 15-frame broad guard rides out the 5-frame transient;
        # lift safety relies on the 12.5 N soft limit and the object-
        # filtered hard limit at 3-frame debounce.
        squeeze_joint_target_gain=1.0,
        max_squeeze_compression_m=0.10,
        force_hard_limit_n=100.0,
        broad_force_limit_frames=15,
        # Filtered-force direction integrator: the foam-box oscillation
        # period is ~120 steps (1 s).  A threshold of 20 gives ~0.17 s
        # response latency, fast enough to track the filtered average
        # without chasing raw noise.  Counter clamping limits overshoot.
        lift_squeeze_persist_frames=20,
        # At the 120-step resonance, alpha=0.005 passes about 9.5% of the
        # oscillatory component.  A separate gate-only balance tolerance
        # admits the residual anti-phase differential without weakening the
        # responsive 0.5 N pair-centre controller.
        # The oscillation low-phase can drop non_thumb below 3.0 N for 60-90
        # steps per cycle.  Extend the recovery timeout to 1.5 oscillation
        # periods so transient contact dips do not trigger contact_degraded.
        lift_force_drop_frames=180,
        advance_force_filter_alpha=0.005,
        advance_force_upper_n=35.0,
        advance_force_balance_tolerance_n=10.0,
        # Squeeze takes ~34 s, plus force convergence and tactile-gated
        # lift; extend the global budget while retaining the 70 s per-phase
        # limit.
        total_timeout_s=130.0,
        lift_height_m=0.20,
    ),
})


def make_clamp_config(object_key: str, **overrides) -> ClampConfig:
    """Return an isolated validated config for a supported large box."""

    key = str(object_key)
    prototype = AUTO_CLAMP_PROFILES.get(key)
    if prototype is None:
        supported = ", ".join(AUTO_CLAMP_OBJECT_KEYS)
        raise ValueError(
            f"unsupported auto-clamp object {key!r}; supported: {supported}"
        )
    return replace(prototype, **overrides)


@dataclass
class BimanualLiftState:
    """State of the coupled mean/differential lift-force controller."""

    filtered_left_n: float
    filtered_right_n: float
    squeeze_offset_m: float
    center_offset_m: float
    progress_m: float = 0.0
    recovery_frames: int = 0
    stable_frames: int = 0
    # Direction persistence counter for the squeeze integrator.  Positive
    # values count consecutive frames requesting squeeze; negative counts
    # relief.  The integrator only acts once the counter magnitude reaches
    # the configured threshold, filtering out oscillation half-cycles.
    squeeze_direction_counter: int = 0
    # Slow-filtered force estimates for the advance gate, seeded from the
    # primary filter at lift entry to avoid cold-start delay.
    advance_filtered_left_n: float = field(init=False)
    advance_filtered_right_n: float = field(init=False)

    def __post_init__(self) -> None:
        self.advance_filtered_left_n = float(self.filtered_left_n)
        self.advance_filtered_right_n = float(self.filtered_right_n)


@dataclass(frozen=True)
class BimanualLiftCommand:
    filtered_left_n: float
    filtered_right_n: float
    advance_filtered_left_n: float
    advance_filtered_right_n: float
    squeeze_offset_m: float
    center_offset_m: float
    delta_z_m: float
    advance: bool
    balanced: bool
    left_contact_ok: bool
    right_contact_ok: bool
    vertical_tracking_ok: bool
    soft_rebalance: bool
    error: str | None = None


def update_bimanual_lift(
    state: BimanualLiftState,
    left_raw_n: float,
    right_raw_n: float,
    left_non_thumb_n: float,
    right_non_thumb_n: float,
    left_contact_body_count: int,
    right_contact_body_count: int,
    config: ClampConfig,
    vertical_tracking_ok: bool = True,
) -> BimanualLiftCommand:
    """Update one atomic command for both hands during vertical lifting.

    Average force changes symmetric squeeze, while left-minus-right force
    shifts the pair centre.  Vertical progress is a common-mode command and
    is emitted only while both contact patches are balanced and valid.
    """

    numeric = (
        left_raw_n,
        right_raw_n,
        left_non_thumb_n,
        right_non_thumb_n,
    )
    if not all(math.isfinite(float(value)) and float(value) >= 0.0
               for value in numeric):
        raise ValueError("bimanual lift forces must be finite and non-negative")
    if left_contact_body_count < 0 or right_contact_body_count < 0:
        raise ValueError("contact body counts must be non-negative")

    alpha = config.force_filter_alpha
    state.filtered_left_n += alpha * (left_raw_n - state.filtered_left_n)
    state.filtered_right_n += alpha * (right_raw_n - state.filtered_right_n)
    advance_alpha = config.advance_force_filter_alpha
    state.advance_filtered_left_n += advance_alpha * (
        left_raw_n - state.advance_filtered_left_n
    )
    state.advance_filtered_right_n += advance_alpha * (
        right_raw_n - state.advance_filtered_right_n
    )

    left_contact_ok = (
        left_non_thumb_n >= config.lift_min_non_thumb_force_n
        and left_contact_body_count >= config.lift_min_contact_body_count
    )
    right_contact_ok = (
        right_non_thumb_n >= config.lift_min_non_thumb_force_n
        and right_contact_body_count >= config.lift_min_contact_body_count
    )
    left_recovery_patch_ok = (
        left_non_thumb_n >= config.lift_recovery_min_non_thumb_force_n
        and left_contact_body_count >= config.lift_min_contact_body_count
    )
    right_recovery_patch_ok = (
        right_non_thumb_n >= config.lift_recovery_min_non_thumb_force_n
        and right_contact_body_count >= config.lift_min_contact_body_count
    )
    error = (
        "force_limit"
        if max(left_raw_n, right_raw_n) >= config.force_hard_limit_n
        else None
    )
    soft_rebalance = (
        error is not None
        or (
            max(left_raw_n, right_raw_n) >= config.lift_soft_force_limit_n
            or abs(left_raw_n - right_raw_n)
            >= config.lift_raw_force_imbalance_soft_n
        )
    )
    if error is not None:
        # The sensor reports the impulse only after the physics step that
        # generated it.  Stop vertical progress and immediately unload the
        # pair while the server's short temporal guard distinguishes a
        # one-step contact impulse from a persistent unsafe load.
        state.squeeze_offset_m = max(
            -config.lift_max_release_offset_m,
            state.squeeze_offset_m
            - config.lift_emergency_squeeze_relief_m,
        )
        if left_raw_n > right_raw_n:
            state.center_offset_m = min(
                config.lift_max_center_offset_m,
                state.center_offset_m + config.lift_emergency_center_step_m,
            )
        elif right_raw_n > left_raw_n:
            state.center_offset_m = max(
                -config.lift_max_center_offset_m,
                state.center_offset_m - config.lift_emergency_center_step_m,
            )

    bilateral_contact_ok = left_contact_ok and right_contact_ok
    bilateral_recovery_patch_ok = (
        left_recovery_patch_ok and right_recovery_patch_ok
    )
    if error is None and (bilateral_contact_ok or bilateral_recovery_patch_ok):
        # Direction persistence filter for symmetric squeeze.  PhysX contact
        # resonances (~1 s period, ~60-step half-cycles) swing raw force
        # between 3 and 42 N.  Using filtered forces (EWMA) for the
        # direction decision smooths these oscillations: the filter
        # represents the sustained average, so the integrator sees a
        # consistent signal even when raw samples alternate wildly.
        weaker_filtered = min(state.filtered_left_n, state.filtered_right_n)
        stronger_filtered = max(state.filtered_left_n, state.filtered_right_n)
        if (weaker_filtered < config.force_lower_n
                and stronger_filtered <= config.force_upper_n):
            state.squeeze_direction_counter += 1
        elif (stronger_filtered > config.force_upper_n
                and weaker_filtered >= config.force_lower_n):
            state.squeeze_direction_counter -= 1
        # else: filtered forces are in-band or straddling — no directional
        # signal.  Leave the counter unchanged.

        threshold = config.lift_squeeze_persist_frames
        # Clamp counter to ±threshold so that a single opposing frame
        # immediately halts the squeeze/relief action.  Without clamping,
        # accumulated evidence beyond the threshold creates a reversal
        # delay that overshoots (e.g. 3.5 mm of relief crashes forces
        # from 16 N to 4 N on foam).
        state.squeeze_direction_counter = max(
            -threshold, min(threshold, state.squeeze_direction_counter)
        )
        if state.squeeze_direction_counter >= threshold:
            state.squeeze_offset_m = min(
                config.lift_max_squeeze_offset_m,
                state.squeeze_offset_m + config.lift_squeeze_slew_m,
            )
            # Reset after acting so the direction must be re-confirmed
            # before the next step.  This limits the squeeze rate to one
            # slew step per threshold frames, preventing force overshoot
            # from continuous actuation during the filter response delay.
            state.squeeze_direction_counter = 0
        elif state.squeeze_direction_counter <= -threshold:
            state.squeeze_offset_m = max(
                -config.lift_max_release_offset_m,
                state.squeeze_offset_m - config.lift_squeeze_slew_m,
            )
            state.squeeze_direction_counter = 0

        if bilateral_contact_ok:
            state.recovery_frames = 0
        else:
            # Both palm/non-thumb patches still exist, but their load has
            # fallen below the lift-quality threshold.  Continue bounded
            # symmetric squeeze while the recovery timeout remains active.
            state.recovery_frames += 1
    else:
        # A thumb-only patch is a geometry failure, not evidence that more
        # normal compression is safe.  Freeze squeeze until the palm and
        # non-thumb contact envelope has returned.
        state.recovery_frames += 1

    # Pair-centre motion transfers load from the strong hand to the weak hand
    # without shrinking the gap.  Keep this recovery degree of freedom active
    # when exactly one palm patch has degraded; it is the safe way to rebuild
    # contact without squeeze-integrator windup.
    force_difference = (
        left_raw_n - right_raw_n
        if soft_rebalance
        else state.filtered_left_n - state.filtered_right_n
    )
    if error is None and (left_contact_ok or right_contact_ok):
        center_slew = config.lift_center_slew_m * (
            config.lift_emergency_center_slew_multiplier
            if soft_rebalance else 1.0
        )
        if force_difference > config.lift_force_balance_tolerance_n:
            state.center_offset_m = min(
                config.lift_max_center_offset_m,
                state.center_offset_m + center_slew,
            )
        elif force_difference < -config.lift_force_balance_tolerance_n:
            state.center_offset_m = max(
                -config.lift_max_center_offset_m,
                state.center_offset_m - center_slew,
            )

    balanced = (
        error is None
        and left_contact_ok
        and right_contact_ok
        and config.force_lower_n
        <= state.advance_filtered_left_n
        <= config.effective_advance_force_upper_n
        and config.force_lower_n
        <= state.advance_filtered_right_n
        <= config.effective_advance_force_upper_n
        and abs(
            state.advance_filtered_left_n - state.advance_filtered_right_n
        ) <= config.advance_force_balance_tolerance_n
        and vertical_tracking_ok
    )
    state.stable_frames = state.stable_frames + 1 if balanced else 0
    advance = (
        balanced
        and state.stable_frames >= config.lift_progress_stable_frames
    )
    delta_z_m = (
        config.lift_height_m / config.lift_steps if advance else 0.0
    )
    if advance:
        state.progress_m = min(
            config.lift_height_m, state.progress_m + delta_z_m
        )
        # One vertical micro-step consumes the stability window. Require a
        # fresh bilateral hold before issuing the next step so tactile
        # feedback can observe and rebalance the changed contact.
        state.stable_frames = 0
    return BimanualLiftCommand(
        filtered_left_n=state.filtered_left_n,
        filtered_right_n=state.filtered_right_n,
        advance_filtered_left_n=state.advance_filtered_left_n,
        advance_filtered_right_n=state.advance_filtered_right_n,
        squeeze_offset_m=state.squeeze_offset_m,
        center_offset_m=state.center_offset_m,
        delta_z_m=delta_z_m,
        advance=advance,
        balanced=balanced,
        left_contact_ok=left_contact_ok,
        right_contact_ok=right_contact_ok,
        vertical_tracking_ok=vertical_tracking_ok,
        soft_rebalance=soft_rebalance,
        error=error,
    )


def filtered_normal_force(axis_forces_y: Sequence[float], arm: str) -> float:
    """Sum object-filtered compressive force along the grasp normal.

    Isaac reports the force acting on each hand body.  The box pushes the
    left hand toward +Y and the right hand toward -Y.  Opposite-direction
    samples are not clamp force and therefore do not contribute to the
    profile-specific target.
    """

    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    sign = 1.0 if arm == "left" else -1.0
    total = 0.0
    for raw_force in axis_forces_y:
        force = float(raw_force)
        if not math.isfinite(force):
            raise ValueError("axis force samples must be finite")
        total += max(0.0, sign * force)
    return total


def aggregate_hand_normal_force(
    axis_force_y_by_body: Mapping[str, float],
    arm: str,
    contact_threshold_n: float = 0.05,
    thumb_is_structural: bool = False,
) -> dict[str, object]:
    """Aggregate every object-filtered hand contact into one side load.

    The O6 thumb cannot be made coplanar with the palm.  It is therefore a
    passive structural contact rather than an ignored actuator: its load is
    included in the physical side-force total and reported separately for load
    distribution diagnostics.

    When *thumb_is_structural* is True, all thumb bodies are excluded from
    the thumb subtotal and counted as structural (non-thumb) force.  This is
    used when the curled thumb serves as the primary clamping surface.
    """

    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    if not math.isfinite(contact_threshold_n) or contact_threshold_n < 0.0:
        raise ValueError("contact threshold must be finite and non-negative")
    sign = 1.0 if arm == "left" else -1.0
    per_body_n: dict[str, float] = {}
    for body, raw_force in axis_force_y_by_body.items():
        force = float(raw_force)
        if not math.isfinite(force):
            raise ValueError("axis force samples must be finite")
        per_body_n[str(body)] = max(0.0, sign * force)
    total_n = sum(per_body_n.values())

    thumb_n = (
        0.0 if thumb_is_structural
        else sum(force for body, force in per_body_n.items() if "thumb" in body)
    )
    palm_n = per_body_n.get(f"{'lh' if arm == 'left' else 'rh'}_hand_base_link", 0.0)
    return {
        "total_n": total_n,
        "thumb_n": thumb_n,
        "palm_n": palm_n,
        "non_thumb_n": total_n - thumb_n,
        "contact_body_count": sum(
            force >= contact_threshold_n for force in per_body_n.values()
        ),
        "per_body_n": per_body_n,
    }


def path_progress_from_actual_y(start_y: float, end_y: float, actual_y: float) -> float:
    """Project measured TCP Y onto a fixed inward path without object truth."""

    start_y, end_y, actual_y = map(float, (start_y, end_y, actual_y))
    if not all(math.isfinite(value) for value in (start_y, end_y, actual_y)):
        raise ValueError("path coordinates must be finite")
    travel = end_y - start_y
    if abs(travel) < 1e-9:
        raise ValueError("path must have non-zero Y travel")
    return max(0.0, min(1.0, (actual_y - start_y) / travel))


def bounded_inward_alpha(
    current_alpha: float,
    actual_progress: float,
    requested_step: float,
    max_lead: float,
) -> float:
    """Advance a command only while the measured TCP remains close behind."""

    values = tuple(map(float, (
        current_alpha, actual_progress, requested_step, max_lead,
    )))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("alpha control values must be finite")
    current_alpha, actual_progress, requested_step, max_lead = values
    if not 0.0 <= current_alpha <= 1.0 or not 0.0 <= actual_progress <= 1.0:
        raise ValueError("alpha values must be inside [0, 1]")
    if requested_step <= 0.0 or max_lead <= 0.0:
        raise ValueError("alpha step and lead must be positive")
    candidate = min(1.0, current_alpha + requested_step)
    # Never pull a command inward solely because contact compliance made the
    # measured TCP lag.  Freeze it until the robot catches up or force control
    # explicitly requests outward relief.
    lead_limit = max(current_alpha, min(1.0, actual_progress + max_lead))
    return min(candidate, lead_limit)


def advance_inward_alpha(
    current_alpha: float,
    actual_tcp_progress: float,
    requested_step: float,
    max_lead: float,
    progress_mode: Literal["tcp_y", "joint_path"],
    joint_tracking_error_rad: float,
    joint_tracking_tolerance_rad: float,
    joint_tracking_hard_limit_rad: float,
    lagging_joint_alpha_step: float,
) -> float:
    """Advance one approach step with the profile's measured-progress gate."""

    tcp_bounded_alpha = bounded_inward_alpha(
        current_alpha,
        actual_tcp_progress,
        requested_step,
        max_lead,
    )
    joint_error = float(joint_tracking_error_rad)
    joint_tolerance = float(joint_tracking_tolerance_rad)
    joint_hard_limit = float(joint_tracking_hard_limit_rad)
    lagging_step = float(lagging_joint_alpha_step)
    if not math.isfinite(joint_error) or joint_error < 0.0:
        raise ValueError("joint tracking error must be finite and non-negative")
    if not (
        math.isfinite(joint_tolerance)
        and math.isfinite(joint_hard_limit)
        and 0.0 < joint_tolerance < joint_hard_limit
    ):
        raise ValueError("joint tracking thresholds must be finite and ordered")
    if not math.isfinite(lagging_step) or not 0.0 < lagging_step <= requested_step:
        raise ValueError("lagging alpha step must fit the requested step")
    if progress_mode == "joint_path":
        if joint_error >= joint_hard_limit:
            return float(current_alpha)
        step = requested_step if joint_error <= joint_tolerance else lagging_step
        return min(1.0, float(current_alpha) + float(step))
    if progress_mode == "tcp_y":
        return tcp_bounded_alpha
    raise ValueError("progress mode must be 'tcp_y' or 'joint_path'")


def updated_squeeze_compression(
    current_m: float,
    action: ForceAction,
    config: ClampConfig,
) -> float:
    """Update contact-local virtual compression from the force action."""

    current_m = float(current_m)
    if not math.isfinite(current_m) or not 0.0 <= current_m <= config.max_squeeze_compression_m:
        raise ValueError("squeeze compression is outside its configured range")
    if action == "inward":
        return min(
            config.max_squeeze_compression_m,
            current_m + config.squeeze_compression_step_m,
        )
    if action == "outward":
        return max(0.0, current_m - config.squeeze_relief_step_m)
    if action == "hold":
        return current_m
    raise ValueError(f"unknown force action: {action}")


def squeeze_seat_detected(
    object_axis_y_by_arm_n: Mapping[str, float],
    config: ClampConfig,
) -> bool:
    """Detect the palm-seating force surge during low-gain squeeze.

    Returns True when the guard is enabled, seating has not yet been
    absorbed, and either side's total object contact force exceeds the
    seating detection threshold.  The caller is responsible for tracking
    whether seating has already been handled.
    """

    if not config.squeeze_contact_guard_enabled:
        return False
    return any(
        float(object_axis_y_by_arm_n.get(arm, 0.0))
        >= config.squeeze_seat_detect_n
        for arm in ("left", "right")
    )


def palms_ready_for_squeeze(
    palm_axis_y_by_arm_n: Mapping[str, float],
    config: ClampConfig,
) -> bool:
    """Require bilateral palm-face load before capturing squeeze anchors.

    When the squeeze contact guard is disabled (e.g. courier_box_m), this
    always returns True so anchor capture follows the legacy path.
    """

    if not config.squeeze_contact_guard_enabled:
        return True
    return all(
        float(palm_axis_y_by_arm_n.get(arm, 0.0)) >= config.contact_detect_n
        for arm in ("left", "right")
    )


def squeeze_contact_guard_should_pause(
    palm_axis_y_by_arm_n: Mapping[str, float],
    compression_by_arm_m: Mapping[str, float],
    config: ClampConfig,
) -> bool:
    """Detect palm-force loss after squeeze has stored displacement.

    Returns True when the guard is enabled, compression has passed the
    startup minimum, and either side's palm Y-force has dropped below the
    loss threshold.
    """

    if not config.squeeze_contact_guard_enabled:
        return False
    past_startup = max(
        float(compression_by_arm_m.get(arm, 0.0))
        for arm in ("left", "right")
    ) > config.squeeze_contact_loss_min_compression_m
    return past_startup and any(
        float(palm_axis_y_by_arm_n.get(arm, 0.0))
        < config.squeeze_contact_loss_n
        for arm in ("left", "right")
    )


@dataclass(frozen=True)
class ForceDecision:
    left_action: ForceAction
    right_action: ForceAction
    left_stable_frames: int
    right_stable_frames: int
    success: bool
    error: str | None = None
    left_contact_seen: bool = False
    right_contact_seen: bool = False


def build_clamp_targets(
    object_center_base_xyz: Sequence[float],
    config: ClampConfig,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Build mirrored hand-base targets around the object's Y faces.

    The hand frame sits above the palm contact region while local Z points
    down; the configured positive Z offset also keeps curled fingertips clear
    of the tabletop throughout the joint-space approach.
    """

    if len(object_center_base_xyz) != 3:
        raise ValueError("object_center_base_xyz must contain exactly 3 values")
    ox, oy, oz = (float(value) for value in object_center_base_xyz)
    if not all(math.isfinite(value) for value in (ox, oy, oz)):
        raise ValueError("object centre contains a non-finite value")

    half_width_y = 0.5 * config.box_size_xyz_m[1]
    pre_offset = half_width_y + config.pregrasp_clearance_m
    clamp_offset = half_width_y + config.clamp_surface_margin_m
    for label, offset in (("left", clamp_offset), ("right", clamp_offset)):
        travel = pre_offset - offset
        if offset <= 0.0:
            raise ValueError(f"{label} clamp offset must be positive")
        if travel <= 0.0 or travel > config.max_clamp_travel_m + 0.05:
            raise ValueError(
                f"{label} clamp travel {travel:.4f} m exceeds configured bound"
            )

    hand_base_z = oz + config.palm_center_offset_z_m
    return {
        "left": {
            "pregrasp": (ox, oy + pre_offset, hand_base_z),
            "clamp": (ox, oy + clamp_offset, hand_base_z),
        },
        "right": {
            "pregrasp": (ox, oy - pre_offset, hand_base_z),
            "clamp": (ox, oy - clamp_offset, hand_base_z),
        },
    }


def quaternion_palm_axes_wxyz(
    quat_wxyz: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return world/base directions of the O6 palm's local X and local Z.

    O6 finger flexion is around local Y, so local +X is the palm-face normal
    and local +Z follows the extended fingers.  Treating local Y as the palm
    normal rotates the requested grasp orientation by 90 degrees.
    """

    if len(quat_wxyz) != 4:
        raise ValueError("quat_wxyz must contain exactly 4 values")
    w, x, y, z = (float(value) for value in quat_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError("quat_wxyz must have a finite non-zero norm")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    local_x = (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    )
    local_z = (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )
    return local_x, local_z


def palm_alignment(quat_wxyz: Sequence[float], arm: str) -> tuple[float, float]:
    """Return (palm-facing-box dot, fingers-down dot) for one arm."""

    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    local_x, local_z = quaternion_palm_axes_wxyz(quat_wxyz)
    inward_y = -1.0 if arm == "left" else 1.0
    inward_dot = local_x[1] * inward_y
    fingers_down_dot = -local_z[2]
    return inward_dot, fingers_down_dot


def evaluate_force_control(
    left_force_n: float,
    right_force_n: float,
    left_stable_frames: int,
    right_stable_frames: int,
    config: ClampConfig,
    left_contact_seen: bool = False,
    right_contact_seen: bool = False,
) -> ForceDecision:
    """Choose clamp-axis actions with a low-force two-palm rendezvous.

    The box is free to slide on the table.  If one palm reaches full target
    force while the other is still in free space, it moves the box away from the
    lagging palm.  The first-contact palm therefore holds a light preload (or
    relieves excess preload) until both palms report contact.  Only then does
    the controller regulate each side into the final force band.
    """

    left_force_n = float(left_force_n)
    right_force_n = float(right_force_n)
    if not all(math.isfinite(value) and value >= 0.0
               for value in (left_force_n, right_force_n)):
        raise ValueError("contact forces must be finite and non-negative")

    left_seen = left_contact_seen or left_force_n >= config.contact_detect_n
    right_seen = right_contact_seen or right_force_n >= config.contact_detect_n

    if left_seen != right_seen:
        if left_seen:
            left_action = (
                "inward"
                if left_force_n < config.contact_detect_n
                else "hold"
                if left_force_n <= config.unilateral_preload_upper_n
                else "outward"
            )
            return ForceDecision(
                left_action, "inward", 0, 0, False,
                left_contact_seen=True,
                right_contact_seen=False,
            )
        right_action = (
            "inward"
            if right_force_n < config.contact_detect_n
            else "hold"
            if right_force_n <= config.unilateral_preload_upper_n
            else "outward"
        )
        return ForceDecision(
            "inward", right_action, 0, 0, False,
            left_contact_seen=False,
            right_contact_seen=True,
        )

    def decide(force_n: float, stable_frames: int) -> tuple[ForceAction, int]:
        if force_n < config.force_lower_n:
            return "inward", 0
        if force_n <= config.force_upper_n:
            return "hold", stable_frames + 1
        return "outward", 0

    left_action, left_stable = decide(left_force_n, left_stable_frames)
    right_action, right_stable = decide(right_force_n, right_stable_frames)
    success = (
        left_stable >= config.force_stable_frames
        and right_stable >= config.force_stable_frames
    )
    return ForceDecision(
        left_action,
        right_action,
        left_stable,
        right_stable,
        success,
        left_contact_seen=left_seen,
        right_contact_seen=right_seen,
    )
