#!/usr/bin/env python3
"""Smoke test: R1-o6 cuRobo v2 config (FK alignment, IK, motion planning).

Run on the Isaac Lab server after preparing .runtime/configs.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_log_file = None

def log(msg: str = "") -> None:
    print(msg, flush=True)
    if _log_file is not None:
        _log_file.write(msg + "\n")
        _log_file.flush()


ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)
TOOL_FRAMES = ("lh_hand_base_link", "rh_hand_base_link")

# Conservative ranges for sampling (narrower than URDF hard limits)
ARM_SAMPLE_RANGES = {
    "left_shoulder_pitch_joint": (-0.90, 0.35),
    "left_shoulder_roll_joint": (0.07547, 1.07547),
    "left_shoulder_yaw_joint": (-0.65, 0.65),
    "left_elbow_joint": (-1.75, -0.45),
    "left_wrist_roll_joint": (-1.20, 1.20),
    "right_shoulder_pitch_joint": (-0.90, 0.35),
    "right_shoulder_roll_joint": (-1.07547, -0.07547),
    "right_shoulder_yaw_joint": (-0.65, 0.65),
    "right_elbow_joint": (-1.75, -0.45),
    "right_wrist_roll_joint": (-1.20, 1.20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R1-o6 cuRobo v2 smoke test")
    parser.add_argument(
        "--robot-config",
        default=str(PROJECT_ROOT / ".runtime" / "configs" / "r1_o6.yml"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fk-samples", type=int, default=8)
    parser.add_argument("--fk-tolerance", type=float, default=0.01,
                        help="FK alignment tolerance in meters")
    parser.add_argument("--ik-trials", type=int, default=32)
    parser.add_argument("--ik-threshold", type=float, default=0.70,
                        help="Minimum IK success rate")
    parser.add_argument("--plan-attempts", type=int, default=5)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def sample_arm_configs(torch_mod, count: int, seed: int, device: str):
    """Sample random arm joint configurations within safe ranges."""
    lo = torch_mod.tensor(
        [ARM_SAMPLE_RANGES[n][0] for n in ARM_JOINT_NAMES], dtype=torch_mod.float32
    )
    hi = torch_mod.tensor(
        [ARM_SAMPLE_RANGES[n][1] for n in ARM_JOINT_NAMES], dtype=torch_mod.float32
    )
    rng = torch_mod.Generator(device="cpu")
    rng.manual_seed(seed)
    q = lo + torch_mod.rand((count, len(ARM_JOINT_NAMES)), generator=rng) * (hi - lo)
    return q.to(device=device)


# ---- Test 1: cuRobo config loading + FK ----

def test_curobo_load_and_fk(torch_mod, Kinematics, KinematicsCfg, JointState,
                            robot_config: str, device: str):
    """Load cuRobo config, verify joint names and tool frames, run FK."""
    log("\n[Test 1] cuRobo config loading + FK")

    cfg = KinematicsCfg.from_robot_yaml_file(robot_config)
    kin = Kinematics(cfg)

    assert list(kin.joint_names) == list(ARM_JOINT_NAMES), (
        f"Joint order mismatch: {kin.joint_names} vs {list(ARM_JOINT_NAMES)}"
    )
    assert tuple(kin.tool_frames) == TOOL_FRAMES, (
        f"Tool frames mismatch: {kin.tool_frames} vs {TOOL_FRAMES}"
    )

    # FK at default position
    q_default = kin.default_joint_position.to(device=device)
    if q_default.ndim == 1:
        q_default = q_default.unsqueeze(0)
    js = JointState.from_position(q_default, joint_names=list(kin.joint_names))
    fk_result = kin.compute_kinematics(js)

    for frame in TOOL_FRAMES:
        pose = fk_result.tool_poses.get_link_pose(frame)
        pos = pose.position.detach().cpu()
        quat = pose.quaternion.detach().cpu()
        log(f"  {frame}: pos={pos.tolist()}, quat={quat.tolist()}")

    log("  [PASS] cuRobo config loaded, FK works")
    return kin


# ---- Test 2: FK alignment (cuRobo vs Isaac Lab) ----

def test_fk_alignment(torch_mod, scene, sim, dt, robot, kin, JointState,
                      samples, tolerance: float):
    """Compare cuRobo FK with Isaac Lab FK for same joint states."""
    log(f"\n[Test 2] FK alignment ({len(samples)} samples, tol={tolerance}m)")

    from r1_o6_scene_cfg import HAND_JOINT_NAMES

    name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}

    max_error = 0.0
    worst_info = ""

    for i, q_arm in enumerate(samples):
        # Write arm joints to Isaac Lab sim
        joint_pos = robot.data.default_joint_pos.clone()
        for src_i, jname in enumerate(ARM_JOINT_NAMES):
            joint_pos[:, name_to_idx[jname]] = q_arm[src_i].item()
        for jname in HAND_JOINT_NAMES:
            if jname in name_to_idx:
                joint_pos[:, name_to_idx[jname]] = 0.0
        joint_vel = torch_mod.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)

        # Step sim to let state settle
        for _ in range(3):
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

        # cuRobo FK
        q_arm_batch = q_arm.unsqueeze(0)
        js = JointState.from_position(q_arm_batch, joint_names=list(kin.joint_names))
        fk_result = kin.compute_kinematics(js)

        # Compare each tool frame
        for frame in TOOL_FRAMES:
            curobo_pos = fk_result.tool_poses.get_link_pose(frame).position.detach()
            # curobo_pos shape: [1, 3] (after get_link_pose)

            # Isaac Lab: get body position in root frame
            body_idx = robot.body_names.index(frame)
            isaac_body_pos = robot.data.body_pos_w[0, body_idx]
            isaac_root_pos = robot.data.root_pos_w[0]
            isaac_root_quat = robot.data.root_quat_w[0]  # wxyz

            # Transform to root frame
            rel_pos = isaac_body_pos - isaac_root_pos
            # quaternion inverse rotation (wxyz convention)
            w, x, y, z = isaac_root_quat[0], isaac_root_quat[1], isaac_root_quat[2], isaac_root_quat[3]
            t = 2.0 * torch_mod.cross(
                torch_mod.stack([x, y, z]),
                rel_pos, dim=-1
            )
            isaac_pos_local = rel_pos - w * t + torch_mod.cross(
                torch_mod.stack([x, y, z]), t, dim=-1
            )

            curobo_pos_flat = curobo_pos.reshape(3).to(isaac_pos_local.device)
            error = torch_mod.linalg.norm(curobo_pos_flat - isaac_pos_local).item()

            if error > max_error:
                max_error = error
                worst_info = (
                    f"sample={i}, frame={frame}, "
                    f"curobo={curobo_pos_flat.cpu().tolist()}, "
                    f"isaac={isaac_pos_local.cpu().tolist()}, "
                    f"error={error:.5f}m"
                )

    log(f"  Max FK error: {max_error:.5f} m")
    if worst_info:
        log(f"  Worst case: {worst_info}")

    if max_error > tolerance:
        log(f"  [FAIL] FK error {max_error:.4f}m exceeds tolerance {tolerance}m")
        return False
    log(f"  [PASS] FK aligned within {tolerance}m")
    return True


# ---- Test 3: IK success rate ----

def test_ik(torch_mod, InverseKinematics, InverseKinematicsCfg, GoalToolPose,
            JointState, Pose, kin, samples, robot_config: str, threshold: float):
    """Test IK success rate on random reachable poses."""
    log(f"\n[Test 3] IK success rate ({len(samples)} trials, threshold={threshold:.0%})")

    # Get FK poses for the samples (these are guaranteed reachable)
    js = JointState.from_position(samples, joint_names=list(kin.joint_names))
    fk_result = kin.compute_kinematics(js)

    # Build goal poses from FK results
    goal_poses = {}
    for frame in kin.tool_frames:
        pose = fk_result.tool_poses.get_link_pose(frame)
        goal_poses[frame] = Pose(position=pose.position.clone(),
                                 quaternion=pose.quaternion.clone())

    goal = GoalToolPose.from_poses(
        goal_poses,
        ordered_tool_frames=list(kin.tool_frames),
        num_goalset=1,
    )

    # Diagnostic: seed IK with the exact FK joint positions (should be trivial)
    log("  --- Diagnostic: exact-seed IK (no optimizer) ---")
    ik_cfg_diag = InverseKinematicsCfg.create(
        robot=robot_config, num_seeds=1,
        max_batch_size=len(samples), self_collision_check=False,
    )
    ik_diag = InverseKinematics(ik_cfg_diag)
    seed = samples.unsqueeze(1)  # [B, 1, DOF]
    diag_result = ik_diag.solve_pose(goal, seed_config=seed, run_optimizer=False)
    _log_ik_diagnostics(diag_result, "  exact-seed no-opt no-coll")

    # Test A: IK without self-collision (easier)
    log("  --- IK without self-collision ---")
    ik_cfg_nocoll = InverseKinematicsCfg.create(
        robot=robot_config, num_seeds=64,
        max_batch_size=len(samples), self_collision_check=False,
    )
    ik_nocoll = InverseKinematics(ik_cfg_nocoll)
    result_nocoll = ik_nocoll.solve_pose(goal)
    rate_nocoll = _log_ik_diagnostics(result_nocoll, "  no-coll")

    # Test B: IK with self-collision
    log("  --- IK with self-collision ---")
    ik_cfg = InverseKinematicsCfg.create(
        robot=robot_config, num_seeds=64,
        max_batch_size=len(samples), self_collision_check=True,
    )
    ik_solver = InverseKinematics(ik_cfg)
    result = ik_solver.solve_pose(goal)
    rate = _log_ik_diagnostics(result, "  with-coll")

    # Use the better of the two rates for pass/fail
    best_rate = max(rate, rate_nocoll)
    if best_rate < threshold:
        log(f"  [FAIL] Best IK rate {best_rate:.1%} below threshold {threshold:.0%}")
        return False
    log(f"  [PASS] IK success rate above threshold")
    return True


def _log_ik_diagnostics(result, prefix: str) -> float:
    """Log IK result diagnostics and return success rate."""
    success = result.success.detach().reshape(-1).float()
    rate = success.mean().item()
    n_ok = int(success.sum().item())
    n_total = len(success)
    log(f"{prefix}: success={n_ok}/{n_total} ({rate:.1%})")

    if hasattr(result, "feasible"):
        feas = result.feasible.detach().reshape(-1).float()
        log(f"{prefix}: feasible={int(feas.sum().item())}/{n_total}")

    if hasattr(result, "position_error"):
        pe = result.position_error.detach().reshape(-1)
        log(f"{prefix}: pos_err min={pe.min().item():.5f} mean={pe.mean().item():.5f} max={pe.max().item():.5f}")

    if hasattr(result, "rotation_error"):
        re = result.rotation_error.detach().reshape(-1)
        log(f"{prefix}: rot_err min={re.min().item():.5f} mean={re.mean().item():.5f} max={re.max().item():.5f}")

    return rate


# ---- Test 4: Motion planning ----

def test_plan_pose(torch_mod, MotionPlanner, MotionPlannerCfg, GoalToolPose,
                   JointState, Pose, kin, samples, robot_config: str,
                   max_attempts: int):
    """Test plan_pose for left-only, right-only, and dual-arm goals."""
    log(f"\n[Test 4] Motion planning (max_attempts={max_attempts})")

    cfg = MotionPlannerCfg.create(robot=robot_config)
    planner = MotionPlanner(cfg)
    log("  Warming up planner...")
    planner.warmup(enable_graph=True, num_warmup_iterations=3)

    q_start = samples[0]
    q_goals = [samples[1], samples[2], samples[3]]

    # Get FK poses
    def get_poses(q):
        js = JointState.from_position(q.unsqueeze(0), joint_names=list(kin.joint_names))
        fk = kin.compute_kinematics(js)
        return {
            frame: Pose(
                position=fk.tool_poses.get_link_pose(frame).position.clone(),
                quaternion=fk.tool_poses.get_link_pose(frame).quaternion.clone(),
            )
            for frame in kin.tool_frames
        }

    start_poses = get_poses(q_start)
    start_js = JointState.from_position(
        q_start.unsqueeze(0), joint_names=list(planner.joint_names)
    )

    cases = [
        ("left_arm_only", {
            "lh_hand_base_link": get_poses(q_goals[0])["lh_hand_base_link"],
            "rh_hand_base_link": start_poses["rh_hand_base_link"],
        }),
        ("right_arm_only", {
            "lh_hand_base_link": start_poses["lh_hand_base_link"],
            "rh_hand_base_link": get_poses(q_goals[1])["rh_hand_base_link"],
        }),
        ("dual_arm", {
            "lh_hand_base_link": get_poses(q_goals[2])["lh_hand_base_link"],
            "rh_hand_base_link": get_poses(q_goals[2])["rh_hand_base_link"],
        }),
    ]

    all_pass = True
    for case_name, goal_poses in cases:
        goal = GoalToolPose.from_poses(
            goal_poses,
            ordered_tool_frames=list(planner.tool_frames),
            num_goalset=1,
        )
        result = planner.plan_pose(goal, start_js, max_attempts=max_attempts)

        if result is not None and hasattr(result, "success"):
            ok = bool(result.success.detach().any().item())
        else:
            ok = False

        waypoints = "?"
        if ok and result is not None and hasattr(result, "get_interpolated_plan"):
            traj = result.get_interpolated_plan()
            if traj is not None and hasattr(traj, "position"):
                waypoints = traj.position.shape[-2]

        status = "PASS" if ok else "FAIL"
        log(f"  [{status}] {case_name}: waypoints={waypoints}")
        if not ok:
            all_pass = False

    return all_pass


# ---- Main ----

def main() -> None:
    # Early debug marker
    global _log_file
    _log_file = open("/tmp/smoke_test_results.log", "w")
    _log_file.write("=== File opened ===\n")
    _log_file.flush()

    try:
        args = parse_args()
    except SystemExit as e:
        _log_file.write(f"parse_args SystemExit: {e}\n")
        _log_file.flush()
        raise
    except Exception as e:
        _log_file.write(f"parse_args error: {e}\n")
        import traceback
        _log_file.write(traceback.format_exc())
        _log_file.flush()
        raise
    _log_file.write("args parsed OK\n")
    _log_file.flush()

    from isaaclab.app import AppLauncher
    try:
        _log_file.write("Creating AppLauncher...\n")
        _log_file.flush()
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        _log_file.write("AppLauncher OK\n")
        _log_file.flush()
    except Exception as e:
        _log_file.write(f"AppLauncher error: {e}\n")
        import traceback
        _log_file.write(traceback.format_exc())
        _log_file.flush()
        raise

    try:
        log("Importing torch...")
        import torch
        log("Importing isaaclab...")
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass
        log("Importing cuRobo...")
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import GoalToolPose, JointState, Pose
        log("Importing r1_o6_scene_cfg...")
        from r1_o6_scene_cfg import R1_O6_CFG, R1_O6_USD_PATH
        log("All imports OK")

        if not Path(R1_O6_USD_PATH).is_file():
            raise FileNotFoundError(f"USD not found: {R1_O6_USD_PATH}")

        # -- Isaac Lab scene (minimal, robot only) --
        @configclass
        class SmokeSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/GroundPlane",
                spawn=sim_utils.GroundPlaneCfg(),
            )
            dome_light = AssetBaseCfg(
                prim_path="/World/DomeLight",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85)),
            )
            robot: ArticulationCfg = R1_O6_CFG

        device = args.device if hasattr(args, "device") else "cuda:0"
        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=device)
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(SmokeSceneCfg(num_envs=1, env_spacing=3.0))
        sim.reset()
        scene.update(sim_cfg.dt)
        robot = scene["robot"]

        # Generate samples
        n_samples = max(args.fk_samples, args.ik_trials, 4)
        samples = sample_arm_configs(torch, n_samples, args.seed, device)

        # Test 1: Load config + FK
        kin = test_curobo_load_and_fk(
            torch, Kinematics, KinematicsCfg, JointState,
            args.robot_config, device,
        )

        # Test 2: FK alignment
        fk_ok = test_fk_alignment(
            torch, scene, sim, sim_cfg.dt, robot, kin, JointState,
            samples[:args.fk_samples], args.fk_tolerance,
        )

        # Test 3: IK
        ik_ok = test_ik(
            torch, InverseKinematics, InverseKinematicsCfg, GoalToolPose,
            JointState, Pose, kin, samples[:args.ik_trials],
            args.robot_config, args.ik_threshold,
        )

        # Test 4: Motion planning
        plan_ok = test_plan_pose(
            torch, MotionPlanner, MotionPlannerCfg, GoalToolPose,
            JointState, Pose, kin, samples,
            args.robot_config, args.plan_attempts,
        )

        # Summary
        log("\n" + "=" * 50)
        results = {
            "Config loading + FK": True,
            "FK alignment": fk_ok,
            "IK success rate": ik_ok,
            "Motion planning": plan_ok,
        }
        all_pass = all(results.values())
        for name, ok in results.items():
            log(f"  {'PASS' if ok else 'FAIL'}: {name}")
        log("=" * 50)
        log(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise
    finally:
        if _log_file is not None:
            _log_file.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
