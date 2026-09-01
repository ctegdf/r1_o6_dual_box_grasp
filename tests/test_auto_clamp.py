#!/usr/bin/env python3
"""Offline regression tests for dual-arm large-box clamp logic."""

import unittest

from auto_clamp import (
    AUTO_CLAMP_OBJECT_KEYS,
    AUTO_CLAMP_PROFILES,
    BimanualLiftState,
    ClampConfig,
    advance_inward_alpha,
    aggregate_hand_normal_force,
    bounded_inward_alpha,
    build_clamp_targets,
    evaluate_force_control,
    filtered_normal_force,
    make_clamp_config,
    palms_ready_for_squeeze,
    path_progress_from_actual_y,
    palm_alignment,
    squeeze_contact_guard_should_pause,
    squeeze_seat_detected,
    update_bimanual_lift,
    updated_squeeze_compression,
)


class ClampObjectProfileTests(unittest.TestCase):
    def test_supported_large_box_profiles_are_explicit(self):
        self.assertEqual(
            AUTO_CLAMP_OBJECT_KEYS,
            ("courier_box_m", "foam_box"),
        )
        self.assertEqual(set(AUTO_CLAMP_PROFILES), set(AUTO_CLAMP_OBJECT_KEYS))

    def test_courier_profile_preserves_v50_runtime_parameters(self):
        cfg = make_clamp_config("courier_box_m")

        self.assertEqual(cfg.box_size_xyz_m, (0.30, 0.22, 0.20))
        self.assertEqual(cfg.force_target_n, 10.5)
        self.assertEqual((cfg.force_lower_n, cfg.force_upper_n), (9.5, 11.5))
        self.assertIsNone(cfg.advance_force_upper_n)
        self.assertEqual(cfg.effective_advance_force_upper_n, 11.5)
        self.assertEqual(
            (cfg.contact_detect_n, cfg.unilateral_preload_upper_n),
            (0.2, 0.8),
        )
        self.assertFalse(cfg.squeeze_contact_guard_enabled)
        self.assertEqual(cfg.approach_progress_mode, "tcp_y")
        self.assertEqual(cfg.max_command_alpha_lead, 0.025)
        self.assertEqual(cfg.lift_height_m, 0.16)
        self.assertEqual(cfg.lift_min_object_rise_m, 0.15)

    def test_foam_profile_uses_tuned_contact_rendezvous(self):
        cfg = make_clamp_config("foam_box")
        targets = build_clamp_targets((0.36, 0.0, 0.145), cfg)

        self.assertEqual(cfg.box_size_xyz_m, (0.35, 0.24, 0.25))
        self.assertEqual(cfg.force_target_n, 11.0)
        self.assertEqual(cfg.force_tolerance_n, 2.0)
        self.assertEqual((cfg.force_lower_n, cfg.force_upper_n), (9.0, 13.0))
        self.assertEqual(cfg.palm_center_offset_z_m, 0.10)
        self.assertEqual(cfg.approach_progress_mode, "tcp_y")
        self.assertEqual(cfg.contact_compression_alpha_lead, 0.20)
        self.assertEqual(cfg.flat_hand_thumb_yaw_rad, 0.0)
        self.assertEqual(cfg.flat_hand_thumb_pitch_rad, 0.0)
        self.assertEqual(cfg.max_command_alpha_lead, 0.025)
        self.assertEqual(cfg.contact_detect_n, 1.0)
        self.assertEqual(cfg.unilateral_preload_upper_n, 3.0)
        self.assertFalse(cfg.squeeze_contact_guard_enabled)
        self.assertEqual(cfg.flat_hand_finger_mcp_rad, 0.3)
        self.assertEqual(cfg.force_hard_limit_n, 100.0)
        self.assertEqual(cfg.broad_force_limit_frames, 15)
        self.assertEqual(cfg.squeeze_compression_step_m, 0.00002)
        self.assertEqual(cfg.squeeze_relief_step_m, 0.0005)
        self.assertEqual(cfg.max_squeeze_compression_m, 0.10)
        self.assertEqual(cfg.squeeze_joint_target_gain, 1.0)
        self.assertEqual(cfg.squeeze_seat_detect_n, 12.0)
        self.assertEqual(cfg.total_timeout_s, 130.0)
        self.assertAlmostEqual(targets["left"]["pregrasp"][1], 0.19)
        self.assertAlmostEqual(targets["right"]["pregrasp"][1], -0.19)
        self.assertAlmostEqual(targets["left"]["clamp"][1], 0.051)
        self.assertAlmostEqual(targets["right"]["clamp"][1], -0.051)
        self.assertAlmostEqual(targets["left"]["pregrasp"][2], 0.245)
        self.assertEqual(cfg.lift_height_m, 0.20)
        self.assertEqual(cfg.lift_min_object_rise_m, 0.15)
        self.assertEqual(cfg.lift_progress_stable_frames, 8)
        self.assertEqual(cfg.lift_force_balance_tolerance_n, 0.5)
        self.assertEqual(cfg.advance_force_filter_alpha, 0.005)
        self.assertEqual(cfg.advance_force_upper_n, 35.0)
        self.assertEqual(cfg.effective_advance_force_upper_n, 35.0)
        self.assertEqual(cfg.advance_force_balance_tolerance_n, 10.0)
        self.assertFalse(cfg.thumb_is_structural)

    def test_foam_squeeze_budget_covers_observed_seating_travel(self):
        cfg = make_clamp_config("foam_box")
        seating_travel_m = 0.081
        commanded_range_m = (
            cfg.max_squeeze_compression_m * cfg.squeeze_joint_target_gain
        )
        commanded_rate_m_s = (
            cfg.squeeze_compression_step_m
            * cfg.squeeze_joint_target_gain
            * 120.0
        )
        seating_time_s = seating_travel_m / commanded_rate_m_s

        self.assertGreater(commanded_range_m, seating_travel_m)
        self.assertAlmostEqual(commanded_rate_m_s, 0.0024)
        self.assertAlmostEqual(seating_time_s, 33.75)
        self.assertLess(seating_time_s, cfg.force_phase_timeout_s)

    def test_runtime_override_does_not_mutate_profile(self):
        overridden = make_clamp_config(
            "foam_box",
            force_target_n=8.5,
            total_timeout_s=1200.0,
        )

        self.assertEqual(overridden.force_target_n, 8.5)
        self.assertEqual(overridden.total_timeout_s, 1200.0)
        self.assertEqual(AUTO_CLAMP_PROFILES["foam_box"].force_target_n, 11.0)
        self.assertFalse(AUTO_CLAMP_PROFILES["foam_box"].thumb_is_structural)

    def test_unknown_clamp_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported auto-clamp object"):
            make_clamp_config("pizza_box")


class ClampGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ClampConfig()

    def test_pregrasp_targets_are_mirrored_about_object_center(self):
        targets = build_clamp_targets((0.27, 0.02, 0.005), self.cfg)

        self.assertAlmostEqual(targets["left"]["pregrasp"][0], 0.27)
        self.assertAlmostEqual(targets["right"]["pregrasp"][0], 0.27)
        self.assertAlmostEqual(targets["left"]["pregrasp"][1], 0.20)
        self.assertAlmostEqual(targets["right"]["pregrasp"][1], -0.16)
        self.assertAlmostEqual(targets["left"]["pregrasp"][2], 0.105)
        self.assertAlmostEqual(targets["right"]["pregrasp"][2], 0.105)

    def test_clamp_targets_move_each_hand_inward_with_bounded_travel(self):
        targets = build_clamp_targets((0.27, 0.0, 0.0), self.cfg)

        left_pre = targets["left"]["pregrasp"][1]
        right_pre = targets["right"]["pregrasp"][1]
        left_clamp = targets["left"]["clamp"][1]
        right_clamp = targets["right"]["clamp"][1]
        self.assertAlmostEqual(left_clamp, 0.041)
        self.assertAlmostEqual(right_clamp, -0.041)
        self.assertLess(left_clamp, left_pre)
        self.assertGreater(right_clamp, right_pre)
        self.assertLessEqual(left_pre - left_clamp, self.cfg.max_clamp_travel_m)
        self.assertLessEqual(
            right_clamp - right_pre, self.cfg.max_clamp_travel_m
        )

    def test_ideal_palm_orientations_face_box_and_fingers_down(self):
        root_half = 2.0 ** -0.5
        left_inward, left_down = palm_alignment(
            (0.0, -root_half, root_half, 0.0), "left"
        )
        right_inward, right_down = palm_alignment(
            (0.0, -root_half, -root_half, 0.0), "right"
        )

        self.assertAlmostEqual(left_inward, 1.0)
        self.assertAlmostEqual(right_inward, 1.0)
        self.assertAlmostEqual(left_down, 1.0)
        self.assertAlmostEqual(right_down, 1.0)

        wrong_left, _ = palm_alignment(
            (0.0, -root_half, -root_half, 0.0), "left"
        )
        wrong_right, _ = palm_alignment(
            (0.0, -root_half, root_half, 0.0), "right"
        )
        self.assertAlmostEqual(wrong_left, -1.0)
        self.assertAlmostEqual(wrong_right, -1.0)

    def test_invalid_object_center_or_excessive_travel_is_rejected(self):
        with self.assertRaises(ValueError):
            build_clamp_targets((0.27, float("nan"), 0.0), self.cfg)
        with self.assertRaises(ValueError):
            build_clamp_targets(
                (0.27, 0.0, 0.0),
                ClampConfig(pregrasp_clearance_m=0.20),
            )

    def test_actual_tcp_progress_is_mirrored_and_bounded(self):
        self.assertAlmostEqual(path_progress_from_actual_y(0.18, 0.04, 0.11), 0.5)
        self.assertAlmostEqual(path_progress_from_actual_y(-0.18, -0.04, -0.11), 0.5)
        self.assertEqual(path_progress_from_actual_y(0.18, 0.04, 0.20), 0.0)
        self.assertEqual(path_progress_from_actual_y(0.18, 0.04, 0.00), 1.0)

    def test_inward_command_cannot_run_ahead_of_actual_tcp(self):
        self.assertAlmostEqual(bounded_inward_alpha(0.30, 0.30, 0.01, 0.025), 0.31)
        self.assertAlmostEqual(bounded_inward_alpha(0.30, 0.20, 0.01, 0.025), 0.30)
        self.assertAlmostEqual(bounded_inward_alpha(0.98, 0.99, 0.05, 0.025), 1.0)

    def test_joint_path_mode_uses_tracking_error_not_nonlinear_tcp_alpha(self):
        self.assertEqual(
            advance_inward_alpha(
                0.24, 0.139, 0.0004, 0.025, "tcp_y",
                0.013, 0.05, 0.10, 0.00004,
            ),
            0.24,
        )
        self.assertGreater(
            advance_inward_alpha(
                0.24, 0.139, 0.0004, 0.025, "joint_path",
                0.030, 0.05, 0.10, 0.00004,
            ),
            0.24,
        )
        self.assertAlmostEqual(
            advance_inward_alpha(
                0.24, 0.139, 0.0004, 0.025, "joint_path",
                0.051, 0.05, 0.10, 0.00004,
            ),
            0.24004,
        )
        self.assertEqual(
            advance_inward_alpha(
                0.24, 0.139, 0.0004, 0.025, "joint_path",
                0.10, 0.05, 0.10, 0.00004,
            ),
            0.24,
        )

    def test_contact_squeeze_compression_is_bounded_and_reversible(self):
        cfg = ClampConfig(
            squeeze_compression_step_m=0.002,
            squeeze_relief_step_m=0.005,
            max_squeeze_compression_m=0.03,
        )
        self.assertAlmostEqual(updated_squeeze_compression(0.01, "inward", cfg), 0.012)
        self.assertAlmostEqual(updated_squeeze_compression(0.029, "inward", cfg), 0.03)
        self.assertAlmostEqual(updated_squeeze_compression(0.01, "outward", cfg), 0.005)
        self.assertAlmostEqual(updated_squeeze_compression(0.01, "hold", cfg), 0.01)

    def test_opt_in_squeeze_guard_requires_bilateral_palm_contact(self):
        cfg = make_clamp_config(
            "foam_box", squeeze_contact_guard_enabled=True,
        )

        self.assertFalse(palms_ready_for_squeeze(
            {"left": 1.0, "right": 0.999}, cfg,
        ))
        self.assertTrue(palms_ready_for_squeeze(
            {"left": 1.0, "right": 1.0}, cfg,
        ))
        # The foam profile leaves the guard disabled, so it does not gate anchors.
        self.assertTrue(palms_ready_for_squeeze(
            {}, make_clamp_config("foam_box"),
        ))

    def test_opt_in_squeeze_guard_pauses_only_after_startup(self):
        cfg = make_clamp_config(
            "foam_box", squeeze_contact_guard_enabled=True,
        )
        low_right = {"left": 1.0, "right": 0.049}

        # Below startup compression — no pause
        self.assertFalse(squeeze_contact_guard_should_pause(
            low_right, {"left": 0.001, "right": 0.001}, cfg,
        ))
        # Above startup compression with low right — pause
        self.assertTrue(squeeze_contact_guard_should_pause(
            low_right, {"left": 0.00101, "right": 0.001}, cfg,
        ))
        # Above startup but both forces OK — no pause
        self.assertFalse(squeeze_contact_guard_should_pause(
            {"left": 0.05, "right": 0.05},
            {"left": 0.00101, "right": 0.001},
            cfg,
        ))
        # The foam profile leaves the guard disabled, so it never pauses.
        self.assertFalse(squeeze_contact_guard_should_pause(
            {"left": 0.0, "right": 0.0},
            {"left": 0.01, "right": 0.01},
            make_clamp_config("foam_box"),
        ))

    def test_opt_in_squeeze_seat_detected_triggers_on_force_surge(self):
        cfg = make_clamp_config(
            "foam_box", squeeze_contact_guard_enabled=True,
        )
        threshold_n = cfg.squeeze_seat_detect_n
        # Below threshold — not detected
        self.assertFalse(squeeze_seat_detected(
            {"left": threshold_n - 0.001, "right": 2.0}, cfg,
        ))
        # Either side at threshold — detected
        self.assertTrue(squeeze_seat_detected(
            {"left": threshold_n, "right": 2.0}, cfg,
        ))
        self.assertTrue(squeeze_seat_detected(
            {"left": 2.0, "right": threshold_n + 5.0}, cfg,
        ))
        # The default foam profile disables seat detection.
        self.assertFalse(squeeze_seat_detected(
            {"left": 50.0, "right": 50.0},
            make_clamp_config("foam_box"),
        ))

    def test_seat_detect_n_remains_valid_when_guard_is_disabled(self):
        cfg = make_clamp_config("foam_box")
        self.assertGreater(cfg.squeeze_seat_detect_n, cfg.contact_detect_n)
        self.assertLess(cfg.squeeze_seat_detect_n, cfg.force_hard_limit_n)


class ClampForceControlTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ClampConfig(force_stable_frames=3)

    def test_11n_target_fits_existing_lift_safety_limits(self):
        cfg = ClampConfig(force_target_n=11.0)

        self.assertEqual(cfg.force_lower_n, 10.0)
        self.assertEqual(cfg.force_upper_n, 12.0)
        self.assertLess(cfg.force_upper_n, cfg.lift_soft_force_limit_n)
        self.assertLess(cfg.lift_soft_force_limit_n, cfg.force_hard_limit_n)

    def test_foam_profile_ignores_recorded_free_space_force_noise(self):
        cfg = make_clamp_config("foam_box")

        noise = evaluate_force_control(0.6, 0.6, 0, 0, cfg)
        self.assertEqual((noise.left_action, noise.right_action), ("inward", "inward"))
        self.assertFalse(noise.left_contact_seen)
        self.assertFalse(noise.right_contact_seen)

        contact = evaluate_force_control(1.0, 0.6, 0, 0, cfg)
        self.assertEqual(
            (contact.left_action, contact.right_action),
            ("hold", "inward"),
        )
        self.assertTrue(contact.left_contact_seen)
        self.assertFalse(contact.right_contact_seen)

    def test_both_hands_must_independently_reach_force_band(self):
        decision = evaluate_force_control(10.0, 0.0, 2, 0, self.cfg)

        self.assertEqual(decision.left_action, "outward")
        self.assertEqual(decision.right_action, "inward")
        self.assertEqual(decision.left_stable_frames, 0)
        self.assertFalse(decision.success)

    def test_first_contact_holds_only_a_light_preload(self):
        decision = evaluate_force_control(1.0, 0.0, 2, 2, self.cfg)

        self.assertEqual(decision.left_action, "outward")
        self.assertEqual(decision.right_action, "inward")
        self.assertEqual(decision.left_stable_frames, 0)
        self.assertEqual(decision.right_stable_frames, 0)
        self.assertTrue(decision.left_contact_seen)

    def test_seen_contact_rebuilds_preload_after_force_drops(self):
        decision = evaluate_force_control(
            0.0, 0.0, 0, 0, self.cfg,
            left_contact_seen=True,
            right_contact_seen=False,
        )

        self.assertEqual(decision.left_action, "inward")
        self.assertEqual(decision.right_action, "inward")
        self.assertTrue(decision.left_contact_seen)
        self.assertFalse(decision.right_contact_seen)

    def test_first_contact_holds_inside_preload_band(self):
        decision = evaluate_force_control(0.5, 0.0, 0, 0, self.cfg)

        self.assertEqual(decision.left_action, "hold")
        self.assertEqual(decision.right_action, "inward")
        self.assertTrue(decision.left_contact_seen)
        self.assertFalse(decision.right_contact_seen)

    def test_preload_is_mirrored_when_right_hand_contacts_first(self):
        decision = evaluate_force_control(0.0, 2.0, 0, 0, self.cfg)

        self.assertEqual(decision.left_action, "inward")
        self.assertEqual(decision.right_action, "outward")
        self.assertFalse(decision.success)

    def test_success_requires_consecutive_stable_frames(self):
        left_stable = right_stable = 0
        for expected in (1, 2, 3):
            decision = evaluate_force_control(
                10.0, 10.0, left_stable, right_stable, self.cfg
            )
            left_stable = decision.left_stable_frames
            right_stable = decision.right_stable_frames
            self.assertEqual(left_stable, expected)
            self.assertEqual(right_stable, expected)

        self.assertTrue(decision.success)

    def test_force_above_target_band_commands_outward_relief(self):
        decision = evaluate_force_control(12.0, 10.0, 2, 2, self.cfg)

        self.assertEqual(decision.left_action, "outward")
        self.assertEqual(decision.left_stable_frames, 0)
        self.assertEqual(decision.right_action, "hold")
        self.assertFalse(decision.success)

    def test_hard_force_sample_commands_relief_while_temporal_guard_counts(self):
        decision = evaluate_force_control(15.0, 10.0, 0, 0, self.cfg)

        self.assertEqual(decision.left_action, "outward")
        self.assertEqual(decision.right_action, "hold")
        self.assertIsNone(decision.error)
        self.assertFalse(decision.success)

    def test_stable_frame_requirement_must_be_positive(self):
        with self.assertRaises(ValueError):
            ClampConfig(force_stable_frames=0)
        with self.assertRaises(ValueError):
            ClampConfig(broad_force_limit_frames=0)
        with self.assertRaises(ValueError):
            ClampConfig(contact_detect_n=2.0, unilateral_preload_upper_n=1.5)
        with self.assertRaises(ValueError):
            ClampConfig(contact_compression_alpha_lead=0.02)
        with self.assertRaises(ValueError):
            ClampConfig(squeeze_joint_target_gain=0.0)
        with self.assertRaises(ValueError):
            ClampConfig(squeeze_contact_guard_enabled=1)
        with self.assertRaises(ValueError):
            ClampConfig(squeeze_contact_loss_frames=0)
        with self.assertRaises(ValueError):
            ClampConfig(squeeze_contact_loss_min_compression_m=0.03)
        self.assertGreater(ClampConfig().lift_height_m, 0.15)
        self.assertGreaterEqual(ClampConfig().lift_min_object_rise_m, 0.15)
        with self.assertRaises(ValueError):
            ClampConfig(lift_min_object_rise_m=0.17)

    def test_lift_tolerances_prioritise_vertical_and_normal_accuracy(self):
        cfg = ClampConfig()
        self.assertLess(cfg.lift_max_z_error_m, cfg.lift_max_normal_error_m)
        self.assertLess(cfg.lift_max_normal_error_m, cfg.lift_max_x_drift_m)
        self.assertGreaterEqual(cfg.lift_progress_stable_frames, 8)
        with self.assertRaises(ValueError):
            ClampConfig(lift_max_z_error_m=0.03)
        with self.assertRaises(ValueError):
            ClampConfig(lift_squeeze_inward_step_m=0.001,
                        lift_squeeze_outward_step_m=0.0005)
        with self.assertRaises(ValueError):
            ClampConfig(lift_emergency_squeeze_relief_m=0.00001)
        with self.assertRaises(ValueError):
            ClampConfig(advance_force_filter_alpha=0.0)
        with self.assertRaises(ValueError):
            ClampConfig(advance_force_balance_tolerance_n=0.0)
        with self.assertRaises(ValueError):
            ClampConfig(advance_force_upper_n=15.0)

    def test_force_drop_resets_only_that_sides_stability(self):
        decision = evaluate_force_control(8.99, 10.0, 2, 2, self.cfg)

        self.assertEqual(decision.left_stable_frames, 0)
        self.assertEqual(decision.right_stable_frames, 3)
        self.assertFalse(decision.success)

    def test_orientation_thresholds_keep_tracking_inside_hard_limits(self):
        with self.assertRaises(ValueError):
            ClampConfig(tracking_min_palm_inward_dot=0.69)
        with self.assertRaises(ValueError):
            ClampConfig(tracking_min_finger_down_dot=0.76)
        with self.assertRaises(ValueError):
            ClampConfig(lift_min_palm_inward_dot=0.69)

    def test_invalid_force_samples_are_rejected(self):
        for invalid in (-0.01, float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    evaluate_force_control(invalid, 10.0, 0, 0, self.cfg)

    def test_filtered_normal_force_uses_only_inward_object_load(self):
        self.assertAlmostEqual(
            filtered_normal_force((4.0, 2.0, -1.0), "left"), 6.0
        )
        self.assertAlmostEqual(
            filtered_normal_force((-4.0, -2.0, 1.0), "right"), 6.0
        )

    def test_filtered_normal_force_rejects_opposite_and_invalid_samples(self):
        self.assertEqual(filtered_normal_force((-3.0, -2.0), "left"), 0.0)
        self.assertEqual(filtered_normal_force((3.0, 2.0), "right"), 0.0)
        with self.assertRaises(ValueError):
            filtered_normal_force((1.0,), "center")
        with self.assertRaises(ValueError):
            filtered_normal_force((float("nan"),), "left")

    def test_all_structural_contacts_contribute_to_side_force(self):
        forces = aggregate_hand_normal_force(
            {
                "lh_hand_base_link": 4.0,
                "lh_thumb_metacarpals_base2": 1.0,
                "lh_thumb_metacarpals": 0.5,
                "lh_thumb_distal": 1.5,
                "lh_index_proximal": 2.0,
                "lh_index_distal": 1.0,
                "lh_ring_distal": -20.0,
            },
            "left",
        )

        self.assertAlmostEqual(forces["total_n"], 10.0)
        self.assertAlmostEqual(forces["thumb_n"], 3.0)
        self.assertAlmostEqual(forces["palm_n"], 4.0)
        self.assertAlmostEqual(forces["non_thumb_n"], 7.0)
        self.assertEqual(forces["contact_body_count"], 6)

    def test_right_structural_contacts_use_mirrored_force_sign(self):
        forces = aggregate_hand_normal_force(
            {
                "rh_hand_base_link": -5.0,
                "rh_thumb_distal": -2.0,
                "rh_middle_proximal": -3.0,
                "rh_pinky_distal": 9.0,
            },
            "right",
        )

        self.assertAlmostEqual(forces["total_n"], 10.0)
        self.assertAlmostEqual(forces["thumb_n"], 2.0)
        self.assertAlmostEqual(forces["non_thumb_n"], 8.0)


class BimanualLiftCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ClampConfig(
            force_filter_alpha=1.0,
            advance_force_filter_alpha=1.0,
            lift_squeeze_persist_frames=1,
            lift_progress_stable_frames=1,
        )
        self.state = BimanualLiftState(
            filtered_left_n=10.0,
            filtered_right_n=10.0,
            squeeze_offset_m=0.04,
            center_offset_m=0.0,
        )

    def step(self, left, right, left_non_thumb=8.0, right_non_thumb=8.0,
             left_bodies=3, right_bodies=3):
        return update_bimanual_lift(
            self.state,
            left,
            right,
            left_non_thumb,
            right_non_thumb,
            left_bodies,
            right_bodies,
            self.cfg,
        )

    def test_balanced_contact_advances_only_the_common_vertical_axis(self):
        command = self.step(10.0, 10.0)

        self.assertTrue(command.advance)
        self.assertTrue(command.balanced)
        self.assertGreater(command.delta_z_m, 0.0)
        self.assertEqual(command.center_offset_m, 0.0)
        self.assertEqual(command.squeeze_offset_m, 0.04)

    def test_foam_advance_accepts_balanced_force_above_target_band(self):
        cfg = make_clamp_config(
            "foam_box",
            force_filter_alpha=1.0,
            advance_force_filter_alpha=1.0,
            lift_progress_stable_frames=1,
        )
        state = BimanualLiftState(17.0, 18.0, 0.0, 0.0)

        command = update_bimanual_lift(
            state, 17.0, 18.0, 8.0, 8.0, 3, 3, cfg,
        )

        self.assertTrue(command.balanced)
        self.assertTrue(command.advance)

    def test_foam_advance_rejects_force_above_sustained_ceiling(self):
        cfg = make_clamp_config(
            "foam_box",
            force_filter_alpha=1.0,
            advance_force_filter_alpha=1.0,
            lift_progress_stable_frames=1,
        )
        state = BimanualLiftState(36.0, 36.0, 0.0, 0.0)

        command = update_bimanual_lift(
            state, 36.0, 36.0, 8.0, 8.0, 3, 3, cfg,
        )

        self.assertIsNone(command.error)
        self.assertFalse(command.balanced)
        self.assertFalse(command.advance)

    def test_foam_advance_rejects_large_sustained_imbalance(self):
        cfg = make_clamp_config(
            "foam_box",
            force_filter_alpha=1.0,
            advance_force_filter_alpha=1.0,
            lift_progress_stable_frames=1,
        )
        state = BimanualLiftState(12.0, 34.0, 0.0, 0.0)

        command = update_bimanual_lift(
            state, 12.0, 34.0, 8.0, 8.0, 3, 3, cfg,
        )

        self.assertFalse(command.balanced)
        self.assertFalse(command.advance)

    def test_force_difference_moves_pair_center_without_extra_squeeze(self):
        command = self.step(12.0, 8.0)

        self.assertFalse(command.advance)
        self.assertGreater(command.center_offset_m, 0.0)
        self.assertEqual(command.squeeze_offset_m, 0.04)

    def test_small_filtered_imbalance_rebalances_before_vertical_progress(self):
        command = self.step(10.0, 9.4)

        self.assertFalse(command.advance)
        self.assertGreater(command.center_offset_m, 0.0)
        self.assertEqual(command.squeeze_offset_m, 0.04)

    def test_weak_thumb_only_side_recovers_via_center_not_squeeze(self):
        squeeze_before = self.state.squeeze_offset_m
        command = self.step(
            12.0, 2.0, right_non_thumb=0.0, right_bodies=1,
        )

        self.assertFalse(command.advance)
        self.assertGreater(command.center_offset_m, 0.0)
        self.assertEqual(command.squeeze_offset_m, squeeze_before)

    def test_bilateral_low_force_increases_only_bounded_squeeze(self):
        previous = self.state.squeeze_offset_m
        command = self.step(6.0, 6.0)

        self.assertFalse(command.advance)
        self.assertGreater(command.squeeze_offset_m, previous)
        for _ in range(5000):
            command = self.step(0.0, 0.0)
        self.assertLessEqual(
            command.squeeze_offset_m, self.cfg.lift_max_squeeze_offset_m
        )

    def test_weak_side_below_band_closes_mean_force_dead_zone(self):
        previous = self.state.squeeze_offset_m
        command = self.step(8.95, 9.56)

        self.assertFalse(command.advance)
        self.assertGreater(command.squeeze_offset_m, previous)

    def test_straddled_force_band_balances_center_before_squeezing(self):
        squeeze_before = self.state.squeeze_offset_m
        command = self.step(8.0, 12.0)

        self.assertFalse(command.advance)
        self.assertLess(command.center_offset_m, 0.0)
        self.assertEqual(command.squeeze_offset_m, squeeze_before)

    def test_thumb_only_contact_freezes_lift_without_integrator_windup(self):
        before = self.state.squeeze_offset_m
        command = self.step(
            2.0, 10.0, left_non_thumb=0.0, left_bodies=1,
        )

        self.assertFalse(command.advance)
        self.assertFalse(command.left_contact_ok)
        self.assertEqual(command.squeeze_offset_m, before)

    def test_bilateral_marginal_palm_patches_keep_recovery_squeeze_active(self):
        before = self.state.squeeze_offset_m
        command = self.step(
            4.0, 5.0,
            left_non_thumb=2.0,
            right_non_thumb=2.5,
            left_bodies=3,
            right_bodies=3,
        )

        self.assertFalse(command.left_contact_ok)
        self.assertFalse(command.right_contact_ok)
        self.assertGreater(command.squeeze_offset_m, before)
        self.assertEqual(self.state.recovery_frames, 1)

    def test_raw_force_in_band_stops_squeeze_despite_filter_lag(self):
        cfg = ClampConfig(force_filter_alpha=0.08)
        state = BimanualLiftState(
            filtered_left_n=8.5,
            filtered_right_n=8.6,
            squeeze_offset_m=0.013,
            center_offset_m=0.0,
        )

        command = update_bimanual_lift(
            state, 9.3, 9.7, 7.0, 7.0, 3, 3, cfg,
        )

        self.assertEqual(command.squeeze_offset_m, 0.013)
        self.assertFalse(command.advance)

    def test_filtered_force_direction_tracks_average_not_raw_noise(self):
        """When raw forces oscillate wildly but the filtered average stays
        above the band, the direction counter consistently accumulates
        relief — it does not chase raw half-cycles."""
        cfg = make_clamp_config("foam_box")
        state = BimanualLiftState(
            filtered_left_n=13.5,
            filtered_right_n=13.5,
            squeeze_offset_m=0.0,
            center_offset_m=0.0,
        )
        # Alternating raw spikes and dips — but filtered starts above 13
        # and stays there due to the high average.
        for _ in range(cfg.lift_squeeze_persist_frames):
            for left_raw, right_raw in (
                (20.0, 42.0),
                (9.0, 8.0),
            ):
                command = update_bimanual_lift(
                    state, left_raw, right_raw,
                    min(left_raw, 8.0), min(right_raw, 8.0),
                    3, 3, cfg,
                )
        # Relief should have triggered (squeeze offset decreased) because
        # filtered forces stayed consistently above the upper band.
        self.assertLess(command.squeeze_offset_m, 0.0)

    def test_in_band_forces_preserve_direction_counter(self):
        """In-band samples leave the counter unchanged — only explicit
        squeeze/relief conditions modify it."""
        cfg = make_clamp_config("foam_box")
        state = BimanualLiftState(
            filtered_left_n=11.0,
            filtered_right_n=11.0,
            squeeze_offset_m=0.0,
            center_offset_m=0.0,
            squeeze_direction_counter=-5,
        )
        update_bimanual_lift(state, 11.0, 11.0, 8.0, 8.0, 3, 3, cfg)
        self.assertEqual(state.squeeze_direction_counter, -5)

    def test_straddling_forces_preserve_direction_counter(self):
        """Band-straddling samples (one below lower, one above upper) also
        leave the counter unchanged."""
        cfg = make_clamp_config("foam_box")
        state = BimanualLiftState(
            filtered_left_n=11.0,
            filtered_right_n=11.0,
            squeeze_offset_m=0.0,
            center_offset_m=0.0,
            squeeze_direction_counter=-5,
        )
        # 8N < force_lower_n(9) and 20N > force_upper_n(13) → straddle
        update_bimanual_lift(state, 8.0, 20.0, 8.0, 8.0, 3, 3, cfg)
        self.assertEqual(state.squeeze_direction_counter, -5)

    def test_sustained_filtered_overforce_accumulates_relief(self):
        """When filtered forces stay consistently above the band, the
        direction counter accumulates relief regardless of raw oscillation."""
        cfg = make_clamp_config("foam_box")
        state = BimanualLiftState(
            filtered_left_n=15.0,
            filtered_right_n=15.0,
            squeeze_offset_m=0.0,
            center_offset_m=0.0,
        )
        # Both filtered forces well above upper band (13N) → relief each
        # frame.  Use consistent raw overforce so filtered stays above band.
        for _ in range(cfg.lift_squeeze_persist_frames):
            command = update_bimanual_lift(
                state, 15.0, 15.0, 8.0, 8.0, 3, 3, cfg,
            )
        # Relief should have triggered (squeeze_offset decreased).
        self.assertLess(command.squeeze_offset_m, 0.0)

    def test_sustained_overforce_triggers_relief_after_persist_threshold(self):
        """After enough consecutive relief-direction frames, the squeeze
        integrator must start relieving."""
        cfg = make_clamp_config("foam_box")
        state = BimanualLiftState(
            filtered_left_n=11.0,
            filtered_right_n=11.0,
            squeeze_offset_m=0.0,
            center_offset_m=0.0,
        )
        # Run enough frames for filtered forces to converge above band
        # and trigger at least one relief step.
        for _ in range(200):
            command = update_bimanual_lift(
                state, 14.0, 14.0, 8.0, 8.0, 3, 3, cfg,
            )
        self.assertLess(command.squeeze_offset_m, 0.0)

    def test_bilateral_overforce_can_release_below_lift_baseline(self):
        self.state.squeeze_offset_m = 0.0

        command = self.step(11.4, 11.3)

        self.assertLess(command.squeeze_offset_m, 0.0)
        for _ in range(1000):
            command = self.step(11.4, 11.3)
        self.assertGreaterEqual(
            command.squeeze_offset_m,
            -self.cfg.lift_max_release_offset_m,
        )

    def test_raw_force_limit_fails_before_filtered_force_can_hide_it(self):
        squeeze_before = self.state.squeeze_offset_m
        center_before = self.state.center_offset_m
        command = self.step(16.0, 10.0)

        self.assertEqual(command.error, "force_limit")
        self.assertFalse(command.advance)
        self.assertTrue(command.soft_rebalance)
        self.assertAlmostEqual(
            command.squeeze_offset_m,
            squeeze_before - self.cfg.lift_emergency_squeeze_relief_m,
        )
        self.assertAlmostEqual(
            command.center_offset_m,
            center_before + self.cfg.lift_emergency_center_step_m,
        )

    def test_soft_force_limit_uses_fast_raw_force_center_rebalance(self):
        before = self.state.center_offset_m
        command = self.step(4.0, 13.2)

        self.assertTrue(command.soft_rebalance)
        self.assertLess(
            command.center_offset_m,
            before - self.cfg.lift_center_slew_m,
        )
        self.assertIsNone(command.error)

    def test_vertical_progress_waits_for_actual_tcp_tracking(self):
        command = update_bimanual_lift(
            self.state, 10.0, 10.0, 8.0, 8.0, 3, 3, self.cfg,
            vertical_tracking_ok=False,
        )

        self.assertFalse(command.advance)
        self.assertFalse(command.balanced)
        self.assertEqual(command.delta_z_m, 0.0)
        self.assertEqual(self.state.progress_m, 0.0)

    def test_slow_advance_filter_passes_high_mean_foam_resonance(self):
        """Slow-average lift force may exceed the clamp target but stays bounded."""
        cfg = make_clamp_config("foam_box")
        state = BimanualLiftState(11.0, 11.0, 0.0, 0.0)
        saw_fast_outside_band_during_advance = False

        # Anti-phase 7-42 N pulses with a ~16.9 N cycle mean, approximating
        # the asymmetric 120-step PhysX resonance observed during lift.
        for step in range(round(cfg.force_phase_timeout_s * 120.0)):
            phase = step % 120
            left_raw_n = 42.0 if phase < 34 else 7.0
            right_raw_n = 42.0 if 60 <= phase < 94 else 7.0
            command = update_bimanual_lift(
                state,
                left_raw_n,
                right_raw_n,
                min(left_raw_n, 8.0),
                min(right_raw_n, 8.0),
                3,
                3,
                cfg,
            )
            if command.advance:
                saw_fast_outside_band_during_advance |= not (
                    cfg.force_lower_n
                    <= command.filtered_left_n
                    <= cfg.force_upper_n
                    and cfg.force_lower_n
                    <= command.filtered_right_n
                    <= cfg.force_upper_n
                )
            if state.progress_m >= cfg.lift_height_m - 1e-9:
                break

        self.assertGreaterEqual(state.progress_m, cfg.lift_height_m - 1e-9)
        self.assertTrue(saw_fast_outside_band_during_advance)

    def test_each_vertical_micro_step_requires_a_fresh_stability_window(self):
        cfg = ClampConfig(
            force_filter_alpha=1.0,
            advance_force_filter_alpha=1.0,
            lift_progress_stable_frames=2,
        )
        state = BimanualLiftState(10.0, 10.0, 0.0, 0.0)

        first = update_bimanual_lift(
            state, 10.0, 10.0, 8.0, 8.0, 3, 3, cfg,
        )
        second = update_bimanual_lift(
            state, 10.0, 10.0, 8.0, 8.0, 3, 3, cfg,
        )
        third = update_bimanual_lift(
            state, 10.0, 10.0, 8.0, 8.0, 3, 3, cfg,
        )

        self.assertFalse(first.advance)
        self.assertTrue(second.advance)
        self.assertFalse(third.advance)


if __name__ == "__main__":
    unittest.main()
