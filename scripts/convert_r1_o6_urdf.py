#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert R1-o6 URDF to USD for Isaac Lab, locking non-arm/hand joints.

This script:
1. Reads the original R1-o6.urdf
2. Converts all non-arm/hand revolute joints to fixed joints
3. Writes a reduced URDF
4. Uses Isaac Lab's UrdfConverter to produce a USD asset

O6 hand DOF structure (per hand):
  Active  (6): thumb_cmc_yaw, thumb_cmc_pitch, index/middle/ring/pinky_mcp_pitch
  Passive (5): thumb_ip, index/middle/ring/pinky_dip  (mimic joints)
  Total active DOFs: 10 arm + 12 hand_active + 10 hand_passive = 32

Run with Isaac Lab's Python launcher. See docs/ASSETS.md.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


ROBOT_DIR = Path(__file__).resolve().parent.parent
INPUT_URDF = ROBOT_DIR / "R1-o6.urdf"
ASSET_DIR = ROBOT_DIR / "assets" / "R1-o6"
REDUCED_URDF_NAME = "R1-o6_arms_hands_fixed.urdf"
USD_NAME = "r1_o6_arms_hands_fixed.usd"

# ---- Joints that remain active (arms + hands) ----
ARM_JOINTS = {
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
}

# Hand active joints: 6 per hand, directly controlled by policy
_HAND_ACTIVE_SUFFIXES = {
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
}

# Hand passive joints: 5 per hand, mimic-coupled to active joints.
# In URDF these have <mimic> tags; convert_mimic_joints_to_normal_joints=True
# converts them to normal revolute joints in USD, and coupling is enforced
# at runtime via _sync_mimic_targets().
_HAND_PASSIVE_SUFFIXES = {
    "thumb_ip",
    "index_dip",
    "middle_dip",
    "ring_dip",
    "pinky_dip",
}

_HAND_PREFIXES = ("lh", "rh")

HAND_ACTIVE_JOINTS = {
    f"{p}_{s}" for p in _HAND_PREFIXES for s in _HAND_ACTIVE_SUFFIXES
}
HAND_PASSIVE_JOINTS = {
    f"{p}_{s}" for p in _HAND_PREFIXES for s in _HAND_PASSIVE_SUFFIXES
}
HAND_ALL_JOINTS = HAND_ACTIVE_JOINTS | HAND_PASSIVE_JOINTS

# 10 arm + 12 hand_active + 10 hand_passive = 32 movable joints
EXPECTED_MOVABLE_COUNT = len(ARM_JOINTS) + len(HAND_ALL_JOINTS)

# Mimic relationships to validate from the original URDF
EXPECTED_MIMIC = {
    "lh_thumb_ip":   ("lh_thumb_cmc_pitch",  2.29),
    "lh_index_dip":  ("lh_index_mcp_pitch",  0.89),
    "lh_middle_dip": ("lh_middle_mcp_pitch", 0.89),
    "lh_ring_dip":   ("lh_ring_mcp_pitch",   0.89),
    "lh_pinky_dip":  ("lh_pinky_mcp_pitch",  0.89),
    "rh_thumb_ip":   ("rh_thumb_cmc_pitch",  1.86),
    "rh_index_dip":  ("rh_index_mcp_pitch",  0.89),
    "rh_middle_dip": ("rh_middle_mcp_pitch", 0.89),
    "rh_ring_dip":   ("rh_ring_mcp_pitch",   0.89),
    "rh_pinky_dip":  ("rh_pinky_mcp_pitch",  0.89),
}

MOVABLE_JOINT_TYPES = {"revolute", "continuous", "prismatic"}


def _is_kept_joint(name: str) -> bool:
    """Return True if this joint should remain movable (not locked to fixed)."""
    return name in ARM_JOINTS or name in HAND_ALL_JOINTS


def _lock_joint(joint_elem: ET.Element) -> None:
    """Convert a revolute/continuous/prismatic joint to fixed."""
    joint_elem.set("type", "fixed")
    for tag in ("axis", "limit", "dynamics", "mimic", "safety_controller", "calibration"):
        child = joint_elem.find(tag)
        if child is not None:
            joint_elem.remove(child)


def write_reduced_urdf(input_path: Path, output_path: Path) -> tuple[list[str], list[str]]:
    """Lock all non-arm/hand movable joints and write a new URDF."""
    tree = ET.parse(input_path)
    root = tree.getroot()

    kept, locked = [], []
    observed_mimic: dict[str, tuple[str, float]] = {}

    for joint in root.findall("joint"):
        name = joint.get("name", "")
        jtype = joint.get("type", "")

        # Record mimic relationships for validation
        mimic_elem = joint.find("mimic")
        if mimic_elem is not None:
            observed_mimic[name] = (
                mimic_elem.get("joint", ""),
                float(mimic_elem.get("multiplier", "1.0")),
            )

        if jtype in MOVABLE_JOINT_TYPES and not _is_kept_joint(name):
            _lock_joint(joint)
            locked.append(name)
        elif jtype in MOVABLE_JOINT_TYPES:
            kept.append(name)

    # Validate mimic joints match expectations
    for name, (expected_parent, expected_mult) in EXPECTED_MIMIC.items():
        if name not in observed_mimic:
            raise RuntimeError(f"Expected mimic joint '{name}' not found in URDF")
        parent, mult = observed_mimic[name]
        if parent != expected_parent or abs(mult - expected_mult) > 0.01:
            raise RuntimeError(
                f"Mimic mismatch for '{name}': "
                f"expected ({expected_parent}, {expected_mult}), "
                f"got ({parent}, {mult})"
            )

    if len(kept) != EXPECTED_MOVABLE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_MOVABLE_COUNT} movable joints, got {len(kept)}: {kept}"
        )

    # Classify kept joints for reporting
    kept_arm = [j for j in kept if j in ARM_JOINTS]
    kept_hand_active = [j for j in kept if j in HAND_ACTIVE_JOINTS]
    kept_hand_passive = [j for j in kept if j in HAND_PASSIVE_JOINTS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return kept_arm, kept_hand_active, kept_hand_passive, locked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-urdf", type=Path, default=INPUT_URDF)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--force", action="store_true", help="Force USD regeneration even if file exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1: Write reduced URDF (lock body joints)
    reduced_urdf = args.asset_dir / REDUCED_URDF_NAME
    kept_arm, kept_hand_active, kept_hand_passive, locked = write_reduced_urdf(
        args.input_urdf, reduced_urdf,
    )
    print(f"[convert] Reduced URDF written: {reduced_urdf}")
    print(f"[convert] Arm joints ({len(kept_arm)}): {sorted(kept_arm)}")
    print(f"[convert] Hand active joints ({len(kept_hand_active)}): {sorted(kept_hand_active)}")
    print(f"[convert] Hand passive/mimic joints ({len(kept_hand_passive)}): {sorted(kept_hand_passive)}")
    print(f"[convert] Body joints locked ({len(locked)}): {sorted(locked)}")

    # Step 2: Copy mesh directory so the URDF can find its meshes
    import shutil
    mesh_src = args.input_urdf.parent / "meshes"
    mesh_dst = args.asset_dir / "meshes"
    if not mesh_src.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_src}")
    shutil.copytree(mesh_src, mesh_dst, dirs_exist_ok=True)
    print(f"[convert] Synced meshes: {mesh_src} -> {mesh_dst}")

    # Step 3: Isaac Lab URDF -> USD conversion (requires Omniverse runtime)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    converter_cfg = UrdfConverterCfg(
        asset_path=str(reduced_urdf),
        usd_dir=str(args.asset_dir),
        usd_file_name=USD_NAME,
        force_usd_conversion=args.force,
        make_instanceable=True,
        fix_base=True,
        merge_fixed_joints=True,
        # Mimic joints (5 passive per hand) are converted to normal revolute joints.
        # Coupling is enforced at runtime via _sync_mimic_targets() in the scene runner.
        convert_mimic_joints_to_normal_joints=True,
        collision_from_visuals=False,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    )
    converter = UrdfConverter(converter_cfg)
    print(f"[convert] USD written: {converter.usd_path}")
    print(f"[convert] Total movable joints in USD: {EXPECTED_MOVABLE_COUNT}")
    print(f"[convert]   = {len(kept_arm)} arm + {len(kept_hand_active)} hand_active + {len(kept_hand_passive)} hand_passive(mimic→normal)")
    print("[convert] Done. Use this USD path in your scene config.")

    simulation_app.close()


if __name__ == "__main__":
    main()
