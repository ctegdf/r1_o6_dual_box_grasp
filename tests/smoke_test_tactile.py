#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O6 触觉传感器 smoke test — 将物体放置于左手掌心, 闭合手指, 验证接触力。

在 Isaac Lab 服务器上运行，项目路径和命令见 docs/OPERATIONS.md。

输出保存到: tests/tactile_smoke_output.txt (脚本同目录)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_PATH = Path(__file__).resolve().parent / "tactile_smoke_output.txt"

# Left hand q=0 world-frame position: [0.037, 0.191, 0.755]
# Place a small object slightly forward (+x) and lower (-z) so fingers wrap around it.
OBJECT_AT_PALM_POS = (0.08, 0.191, 0.74)

# Left hand close preset: (joint_name, target_value)
# From pose_grasp_cli.py HAND_PRESETS["close"]
LEFT_HAND_CLOSE = (
    ("lh_thumb_cmc_yaw", 0.0),
    ("lh_thumb_cmc_pitch", 0.5),
    ("lh_index_mcp_pitch", 1.4),
    ("lh_middle_mcp_pitch", 1.4),
    ("lh_ring_mcp_pitch", 1.4),
    ("lh_pinky_mcp_pitch", 1.4),
)

# (body_name, scene_sensor_key)
FINGERTIP_SENSOR_KEYS = (
    ("lh_thumb_distal", "lh_thumb_object_contact"),
    ("lh_index_distal", "lh_index_object_contact"),
    ("lh_middle_distal", "lh_middle_object_contact"),
    ("lh_ring_distal", "lh_ring_object_contact"),
    ("lh_pinky_distal", "lh_pinky_object_contact"),
    ("rh_thumb_distal", "rh_thumb_object_contact"),
    ("rh_index_distal", "rh_index_object_contact"),
    ("rh_middle_distal", "rh_middle_object_contact"),
    ("rh_ring_distal", "rh_ring_object_contact"),
    ("rh_pinky_distal", "rh_pinky_object_contact"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="O6 tactile sensor smoke test (finger close)")
    parser.add_argument(
        "--object", default="courier_box_m", help="Supported box object key",
    )
    parser.add_argument(
        "--object-pos", nargs=3, type=float, default=OBJECT_AT_PALM_POS,
        metavar=("X", "Y", "Z"),
        help="Object init position in world frame (default: left palm)",
    )
    parser.add_argument("--steps", type=int, default=300, help="Total simulation steps")
    parser.add_argument("--close-steps", type=int, default=30,
                        help="Steps to ramp hand from open to closed")
    parser.add_argument("--interval", type=int, default=50, help="Print interval (steps)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output file")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path: Path = args.output
    log_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.scene import InteractiveScene

        from r1_o6_scene_cfg import (
            MIMIC_JOINTS,
            O6_ALL_TACTILE_BODIES,
            O6_FINGERTIP_BODIES,
            OBJECT_KEYS,
            R1_O6_ALL_JOINT_NAMES,
            R1_O6_USD_PATH,
            R1O6DeliverySceneCfg,
            make_delivery_object_cfg,
        )

        if args.object not in OBJECT_KEYS:
            raise ValueError(f"Unknown --object {args.object!r}. Available: {', '.join(OBJECT_KEYS)}")
        if not Path(R1_O6_USD_PATH).is_file():
            raise FileNotFoundError(f"R1-o6 USD not found: {R1_O6_USD_PATH}")

        # --- Build scene: object placed at left palm ---
        object_pos = tuple(float(v) for v in args.object_pos)
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            device=getattr(args, "device", "cuda:0"),
        )
        sim = sim_utils.SimulationContext(sim_cfg)

        scene_cfg = R1O6DeliverySceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
        scene_cfg.object = make_delivery_object_cfg(args.object).replace(
            init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos),
        )
        sim.set_camera_view(eye=(0.8, 0.6, 1.1), target=object_pos)
        scene = InteractiveScene(scene_cfg)

        log(f"[setup] object={args.object}, pos={object_pos}")
        log(f"[setup] steps={args.steps}, close_steps={args.close_steps}, dt={sim_cfg.dt}")

        sim.reset()
        scene.reset()
        scene.update(sim_cfg.dt)
        log("[setup] sim.reset() done")

        # --- Validate joints ---
        robot = scene["robot"]
        actual = set(robot.joint_names)
        expected = set(R1_O6_ALL_JOINT_NAMES)
        if actual != expected:
            raise RuntimeError(
                f"Joint mismatch: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )

        # --- Validate sensors ---
        contact_hands = scene["contact_hands"]
        body_names = tuple(contact_hands.body_names)
        resolved = set(body_names)
        expected_bodies = set(O6_ALL_TACTILE_BODIES)
        if resolved != expected_bodies:
            raise RuntimeError(
                f"contact_hands body mismatch: "
                f"missing={sorted(expected_bodies - resolved)}, "
                f"extra={sorted(resolved - expected_bodies)}"
            )
        log(f"[layout] contact_hands: {len(body_names)} bodies resolved OK")
        log(f"[layout] net_forces_w shape: {tuple(contact_hands.data.net_forces_w.shape)}")

        filtered_sensors = {}
        for body_name, sensor_key in FINGERTIP_SENSOR_KEYS:
            sensor = scene[sensor_key]
            filtered_sensors[sensor_key] = sensor
            fm = getattr(sensor.data, "force_matrix_w", None)
            fm_shape = tuple(fm.shape) if fm is not None else "N/A"
            log(f"[layout] {sensor_key}: body={tuple(sensor.body_names)}, force_matrix_w={fm_shape}")

        # --- Joint control helpers ---
        name_to_idx = {name: i for i, name in enumerate(robot.joint_names)}

        def sync_mimic() -> None:
            target = robot.data.joint_pos_target.clone()
            limits = robot.data.soft_joint_pos_limits
            for child, parent, mult, offset in MIMIC_JOINTS:
                ci, pi = name_to_idx[child], name_to_idx[parent]
                target[:, ci] = (target[:, pi] * mult + offset).clamp(
                    min=limits[:, ci, 0], max=limits[:, ci, 1],
                )
            robot.set_joint_position_target(target)

        def set_left_hand_target(fraction: float) -> None:
            """Set left hand joint targets to fraction * close preset."""
            fraction = max(0.0, min(1.0, fraction))
            target = robot.data.joint_pos_target.clone()
            for joint_name, joint_value in LEFT_HAND_CLOSE:
                target[:, name_to_idx[joint_name]] = fraction * joint_value
            robot.set_joint_position_target(target)

        # --- Simulation loop ---
        max_net = 0.0
        max_filtered = 0.0

        def snapshot(step: int) -> None:
            nonlocal max_net, max_filtered

            forces = contact_hands.data.net_forces_w  # (1, B, 3)
            if not torch.isfinite(forces).all():
                raise RuntimeError(f"Step {step}: net_forces_w has non-finite values")

            mag = torch.linalg.norm(forces[0], dim=-1)  # (B,)
            step_max_net = float(mag.max())
            active_count = int((mag > 1e-4).sum())

            step_max_filtered = 0.0
            filtered_vals: dict[str, float] = {}
            for body_name, sensor_key in FINGERTIP_SENSOR_KEYS:
                sensor = filtered_sensors[sensor_key]
                fm = getattr(sensor.data, "force_matrix_w", None)
                if fm is not None and fm.numel() > 0:
                    if not torch.isfinite(fm).all():
                        raise RuntimeError(f"Step {step}: {sensor_key}.force_matrix_w non-finite")
                    val = float(torch.linalg.norm(fm[0].reshape(-1, 3), dim=-1).max())
                else:
                    val = 0.0
                filtered_vals[body_name] = val
                step_max_filtered = max(step_max_filtered, val)

            max_net = max(max_net, step_max_net)
            max_filtered = max(max_filtered, step_max_filtered)

            air_time = getattr(contact_hands.data, "current_air_time", None)

            log("")
            log(f"[step {step:04d}] max_net={step_max_net:.4f} N, active={active_count}/{len(body_names)}, max_object={step_max_filtered:.4f} N")
            log(f"  {'body':<28} {'net_N':>8}  {'air_s':>6}  {'object_N':>8}")
            log(f"  {'─' * 54}")
            for i, bname in enumerate(body_names):
                net_n = float(mag[i])
                if air_time is not None and air_time.ndim >= 2:
                    air_s = f"{float(air_time[0, i]):6.3f}"
                else:
                    air_s = "   n/a"
                obj_n = filtered_vals.get(bname)
                obj_text = f"{obj_n:8.4f}" if obj_n is not None else "       -"
                marker = " *" if (net_n > 0.01 or (obj_n is not None and obj_n > 0.01)) else ""
                log(f"  {bname:<28} {net_n:8.4f}  {air_s}  {obj_text}{marker}")

        log("")
        log(f"[action] Closing left hand over {args.close_steps} steps, then holding...")
        for step in range(args.steps + 1):
            if step % args.interval == 0 or step == args.steps:
                snapshot(step)

            if step == args.steps:
                break
            if not simulation_app.is_running():
                log(f"[warn] Simulation stopped at step {step}")
                break

            # Ramp hand close over close_steps
            if args.close_steps > 0:
                fraction = min(1.0, float(step + 1) / float(args.close_steps))
            else:
                fraction = 1.0
            set_left_hand_target(fraction)
            sync_mimic()
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_cfg.dt)

        log("")
        log("[result]")
        log(f"  max_contact_hands_net_N = {max_net:.4f}")
        log(f"  max_filtered_object_N   = {max_filtered:.4f}")
        if max_filtered > 1e-4:
            log("  PASS: fingertip-object contact detected")
        else:
            log("  WARN: no fingertip-object contact detected (object may have fallen or missed)")
            log("  Broad sensor still validated: all tensors finite, shapes correct")

    except Exception as exc:
        log("")
        log(f"[FAIL] {exc}")
        log(traceback.format_exc())
        raise
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"\n[smoke] output saved to {output_path}", flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
