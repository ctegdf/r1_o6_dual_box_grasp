# -*- coding: utf-8 -*-
"""Isaac Lab scene config: fixed-base R1-o6 with tabletop delivery objects.

Robot: R1-o6 humanoid, legs/waist/head locked, arms + O6 dexterous hands active.
Scene: parallel environments with one selected delivery object type on a table.

This standalone project exposes only courier_box_m and foam_box.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
from isaaclab.utils import configclass

from delivery_objects_cfg import DELIVERY_OBJECTS

# ============================================================
# Robot configuration
# ============================================================

# USD produced by scripts/convert_r1_o6_urdf.py
_ROBOT_DIR = Path(__file__).resolve().parent
R1_O6_USD_PATH = str(_ROBOT_DIR / "assets" / "R1-o6" / "r1_o6_arms_hands_fixed.usd")

# Zero-pose pelvis height. Shoulder roll has ±10deg baked into URDF origin,
# so q=0 already has arms slightly apart (no cuRobo self-collision).
ROBOT_ROOT_HEIGHT = 0.82

# ---- Explicit joint name lists (safer than regex with re.fullmatch) ----

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

# ---- Hand joints: 6 active + 5 passive (mimic) per hand ----
# Active: directly controlled by policy
_HAND_ACTIVE_SUFFIXES = (
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)
# Passive (mimic): follow active parents via _sync_mimic_targets()
_HAND_PASSIVE_SUFFIXES = (
    "thumb_ip",
    "index_dip",
    "middle_dip",
    "ring_dip",
    "pinky_dip",
)

HAND_ACTIVE_JOINT_NAMES = tuple(
    f"{prefix}_{s}" for prefix in ("lh", "rh") for s in _HAND_ACTIVE_SUFFIXES
)
HAND_PASSIVE_JOINT_NAMES = tuple(
    f"{prefix}_{s}" for prefix in ("lh", "rh") for s in _HAND_PASSIVE_SUFFIXES
)
HAND_JOINT_NAMES = HAND_ACTIVE_JOINT_NAMES + HAND_PASSIVE_JOINT_NAMES

# Policy action space: 10 arm + 12 active hand = 22 DOFs
R1_O6_ACTIVE_DOF_NAMES = ARM_JOINT_NAMES + HAND_ACTIVE_JOINT_NAMES
# Passive mimic: 10 DOFs (not in policy action space)
R1_O6_PASSIVE_DOF_NAMES = HAND_PASSIVE_JOINT_NAMES
# All USD articulation joints: 32 total
R1_O6_ALL_JOINT_NAMES = R1_O6_ACTIVE_DOF_NAMES + R1_O6_PASSIVE_DOF_NAMES
# Backward-compatible alias used by _validate_robot_joints()
R1_O6_ACTIVE_JOINT_NAMES = R1_O6_ALL_JOINT_NAMES

# Mimic relationships from URDF (child, parent, multiplier, offset).
# USD converts mimic to normal joints (convert_mimic_joints_to_normal_joints=True),
# so coupling must be enforced at runtime via _sync_mimic_targets().
MIMIC_JOINTS = (
    ("lh_thumb_ip",   "lh_thumb_cmc_pitch",  2.29, 0.0),
    ("lh_index_dip",  "lh_index_mcp_pitch",  0.89, 0.0),
    ("lh_middle_dip", "lh_middle_mcp_pitch", 0.89, 0.0),
    ("lh_ring_dip",   "lh_ring_mcp_pitch",   0.89, 0.0),
    ("lh_pinky_dip",  "lh_pinky_mcp_pitch",  0.89, 0.0),
    ("rh_thumb_ip",   "rh_thumb_cmc_pitch",  1.86, 0.0),
    ("rh_index_dip",  "rh_index_mcp_pitch",  0.89, 0.0),
    ("rh_middle_dip", "rh_middle_mcp_pitch", 0.89, 0.0),
    ("rh_ring_dip",   "rh_ring_mcp_pitch",   0.89, 0.0),
    ("rh_pinky_dip",  "rh_pinky_mcp_pitch",  0.89, 0.0),
)

# ============================================================
# O6 tactile sensor body definitions
# ============================================================
# The real Linxin O6 hand has distributed capacitive pressure arrays:
#   - Each fingertip (distal phalanx): high-density tactile pad, ~0.01N threshold
#   - Each finger proximal phalanx: lateral contact surface
#   - Palm (hand_base_link): large-area pressure sensing
#   - Thumb metacarpal base: CMC area contact
#
# Isaac Lab approximation: PhysX ContactReporter per rigid body link.
# Reports net normal contact force per body at physics rate (120Hz).

# Fingertip distal links — primary tactile surfaces (high-density sensing on real O6)
O6_FINGERTIP_BODIES = (
    "lh_thumb_distal", "lh_index_distal", "lh_middle_distal",
    "lh_ring_distal", "lh_pinky_distal",
    "rh_thumb_distal", "rh_index_distal", "rh_middle_distal",
    "rh_ring_distal", "rh_pinky_distal",
)

# Proximal phalanx links — lateral/wrap contact (medium-density on real O6)
O6_PROXIMAL_BODIES = (
    "lh_thumb_metacarpals", "lh_index_proximal", "lh_middle_proximal",
    "lh_ring_proximal", "lh_pinky_proximal",
    "rh_thumb_metacarpals", "rh_index_proximal", "rh_middle_proximal",
    "rh_ring_proximal", "rh_pinky_proximal",
)

# Palm links — large-area pressure sensing
O6_PALM_BODIES = ("lh_hand_base_link", "rh_hand_base_link")

# Thumb CMC base links — optional, contact during power grasp
O6_THUMB_BASE_BODIES = ("lh_thumb_metacarpals_base2", "rh_thumb_metacarpals_base2")

# All tactile-relevant bodies (24 total: 10 fingertip + 10 proximal + 2 palm + 2 thumb base)
O6_ALL_TACTILE_BODIES = (
    O6_FINGERTIP_BODIES + O6_PROXIMAL_BODIES + O6_PALM_BODIES + O6_THUMB_BASE_BODIES
)


def _object_contact_sensor_name(body: str) -> str:
    if body.endswith("_hand_base_link"):
        return body[:2] + "_palm_object_contact"
    if body.endswith("_thumb_metacarpals_base2"):
        return body[:2] + "_thumb_base_object_contact"
    if body.endswith("_distal"):
        return body.removesuffix("_distal") + "_object_contact"
    return body + "_object_contact"


O6_OBJECT_CONTACT_SENSOR_NAMES = {
    body: _object_contact_sensor_name(body) for body in O6_ALL_TACTILE_BODIES
}

# Regex for ContactSensorCfg prim_path: matches all O6 hand bodies under Robot
_O6_TACTILE_BODY_REGEX = "(" + "|".join(O6_ALL_TACTILE_BODIES) + ")"


def _object_contact_sensor_cfg(body: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        force_threshold=0.05,
    )


R1_O6_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=R1_O6_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=2,
            fix_root_link=True,
        ),
    ),
    # URDF zero-pose: elbows bent 90°, shoulder_roll offset ±10° (arms slightly apart),
    # wrists rotated ±90° (palms facing inward), hands below pelvis.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, ROBOT_ROOT_HEIGHT),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        # 10 arm joints (5 per side)
        "arms": ImplicitActuatorCfg(
            joint_names_expr=list(ARM_JOINT_NAMES),
            effort_limit_sim=60.0,
            velocity_limit_sim=20.0,
            stiffness=400.0,
            damping=40.0,
        ),
        # 12 active hand joints (6 per side) — policy-controlled
        "hands_active": ImplicitActuatorCfg(
            joint_names_expr=list(HAND_ACTIVE_JOINT_NAMES),
            effort_limit_sim=20.0,
            velocity_limit_sim=3.0,
            stiffness=30.0,
            damping=3.0,
        ),
        # 10 passive hand joints (5 per side) — mimic, driven by _sync_mimic_targets()
        # Higher stiffness ensures they track targets tightly
        "hands_passive": ImplicitActuatorCfg(
            joint_names_expr=list(HAND_PASSIVE_JOINT_NAMES),
            effort_limit_sim=20.0,
            velocity_limit_sim=8.0,
            stiffness=60.0,
            damping=6.0,
        ),
    },
)


# ============================================================
# Table configuration
# ============================================================

TABLE_SURFACE_Z = 0.72
TABLE_CENTER_X = 0.50  # near edge at x=0.10, away from robot body at x=0
TABLE_TOP_THICKNESS = 0.04
TABLE_TOP_SIZE = (0.80, 1.40, TABLE_TOP_THICKNESS)
TABLE_TOP_POS = (TABLE_CENTER_X, 0.0, TABLE_SURFACE_Z - 0.5 * TABLE_TOP_THICKNESS)

TABLE_LEG_RADIUS = 0.025
TABLE_LEG_HEIGHT = TABLE_SURFACE_Z - TABLE_TOP_THICKNESS
TABLE_LEG_Z = 0.5 * TABLE_LEG_HEIGHT
TABLE_LEG_DX = 0.34  # half-length offset from table center
TABLE_LEG_DY = 0.62  # half-width offset from table center

TABLE_COLOR_TOP = (0.45, 0.38, 0.30)
TABLE_COLOR_LEG = (0.32, 0.28, 0.22)

TABLE_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.8,
    dynamic_friction=0.65,
    restitution=0.02,
    friction_combine_mode="average",
    restitution_combine_mode="min",
)


def _table_top_cfg() -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=TABLE_TOP_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=TABLE_PHYSICS_MATERIAL,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=TABLE_COLOR_TOP, roughness=0.75, metallic=0.0,
        ),
    )


def _table_leg_cfg() -> sim_utils.CylinderCfg:
    return sim_utils.CylinderCfg(
        radius=TABLE_LEG_RADIUS,
        height=TABLE_LEG_HEIGHT,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=TABLE_PHYSICS_MATERIAL,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=TABLE_COLOR_LEG, roughness=0.6, metallic=0.1,
        ),
    )


# ============================================================
# Object configuration — one selected object type for all envs
# ============================================================

OBJECT_KEYS = tuple(DELIVERY_OBJECTS.keys())
NUM_OBJECTS = len(OBJECT_KEYS)
DEFAULT_OBJECT_KEY = "courier_box_m"

# Small gap above table surface to avoid initial penetration
OBJECT_INIT_CLEARANCE = 0.005
# Object centers recorded by the validated server runs.  In particular, the
# successful foam-box flat-palm v19 run started at world X=0.295 m.
VALIDATED_BOX_INIT_X = {
    "courier_box_m": 0.27,
    "foam_box": 0.295,
}
# Clamped per-object so items stay fully on the table (both near and far edges).
OBJECT_TABLE_EDGE_CLEARANCE = 0.005


def _object_half_height(object_key: str) -> float:
    """Object half-height from delivery_objects_cfg (z value = height/2)."""
    return float(DELIVERY_OBJECTS[object_key].init_state.pos[2])


def _object_half_extent_x(object_key: str) -> float:
    """Half extent along x for table-edge clamping."""
    spawn_cfg = DELIVERY_OBJECTS[object_key].spawn
    size = getattr(spawn_cfg, "size", None)
    if size is not None:
        return 0.5 * float(size[0])
    radius = getattr(spawn_cfg, "radius", None)
    if radius is not None:
        return float(radius)
    return 0.0


def _object_init_pos(object_key: str) -> tuple[float, float, float]:
    """Compute per-object init position: near robot side of table, above surface.

    Clamped to keep the object fully on the table (both near and far edges).
    """
    half_x = _object_half_extent_x(object_key)
    table_near_edge = TABLE_CENTER_X - 0.5 * TABLE_TOP_SIZE[0]
    table_far_edge = TABLE_CENTER_X + 0.5 * TABLE_TOP_SIZE[0]
    x_min = table_near_edge + half_x + OBJECT_TABLE_EDGE_CLEARANCE
    x_max = table_far_edge - half_x - OBJECT_TABLE_EDGE_CLEARANCE
    x = max(x_min, min(VALIDATED_BOX_INIT_X[object_key], x_max))
    z = TABLE_SURFACE_Z + _object_half_height(object_key) + OBJECT_INIT_CLEARANCE
    return (x, 0.0, z)


def make_delivery_object_cfg(object_key: str = DEFAULT_OBJECT_KEY) -> RigidObjectCfg:
    """Create a RigidObjectCfg for a single delivery object."""
    if object_key not in DELIVERY_OBJECTS:
        raise ValueError(
            f"Unknown object '{object_key}'. Available: {', '.join(OBJECT_KEYS)}"
        )
    return DELIVERY_OBJECTS[object_key].replace(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_object_init_pos(object_key)),
    )


def make_delivery_objects_cfg(selected_keys: tuple[str, ...]) -> RigidObjectCfg:
    """Create a RigidObjectCfg for one or more delivery objects.

    Single object: uses its spawn config directly (replicate_physics=True safe).
    Multiple objects: uses MultiAssetSpawnerCfg(random_choice=False) to assign
    one object per env by index (requires replicate_physics=False).
    """
    unknown = [k for k in selected_keys if k not in DELIVERY_OBJECTS]
    if unknown:
        raise ValueError(
            f"Unknown object(s): {', '.join(unknown)}. "
            f"Available: {', '.join(OBJECT_KEYS)}"
        )

    if len(selected_keys) == 1:
        return make_delivery_object_cfg(selected_keys[0])

    # Use the tallest object's clearance for the shared init z
    max_half_h = max(_object_half_height(k) for k in selected_keys)
    z = TABLE_SURFACE_Z + max_half_h + OBJECT_INIT_CLEARANCE

    # Use the widest object for x clamping (both edges)
    max_half_x = max(_object_half_extent_x(k) for k in selected_keys)
    table_near_edge = TABLE_CENTER_X - 0.5 * TABLE_TOP_SIZE[0]
    table_far_edge = TABLE_CENTER_X + 0.5 * TABLE_TOP_SIZE[0]
    x_min = table_near_edge + max_half_x + OBJECT_TABLE_EDGE_CLEARANCE
    x_max = table_far_edge - max_half_x - OBJECT_TABLE_EDGE_CLEARANCE
    preferred_x = max(VALIDATED_BOX_INIT_X[k] for k in selected_keys)
    x = max(x_min, min(preferred_x, x_max))

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=MultiAssetSpawnerCfg(
            assets_cfg=[DELIVERY_OBJECTS[k].spawn for k in selected_keys],
            random_choice=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, 0.0, z)),
    )


OBJECT_CFG = make_delivery_object_cfg(DEFAULT_OBJECT_KEY)


# ============================================================
# Scene configuration
# ============================================================

@configclass
class R1O6DeliverySceneCfg(InteractiveSceneCfg):
    """Fixed-base R1-o6 with a tabletop and one selected delivery object."""

    # -- Lighting & ground --
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85)),
    )

    # -- Robot --
    robot: ArticulationCfg = R1_O6_CFG

    # Enabled only by pose_grasp_server.py --record-dir.  Keeping it optional
    # avoids rendering overhead for normal control sessions.
    record_camera: CameraCfg | None = None

    # -- Table (static colliders) --
    table_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        spawn=_table_top_cfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_TOP_POS),
    )

    table_leg_fl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFL",
        spawn=_table_leg_cfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER_X + TABLE_LEG_DX, TABLE_LEG_DY, TABLE_LEG_Z),
        ),
    )
    table_leg_fr = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFR",
        spawn=_table_leg_cfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER_X + TABLE_LEG_DX, -TABLE_LEG_DY, TABLE_LEG_Z),
        ),
    )
    table_leg_bl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegBL",
        spawn=_table_leg_cfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER_X - TABLE_LEG_DX, TABLE_LEG_DY, TABLE_LEG_Z),
        ),
    )
    table_leg_br = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegBR",
        spawn=_table_leg_cfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER_X - TABLE_LEG_DX, -TABLE_LEG_DY, TABLE_LEG_Z),
        ),
    )

    # -- O6 tactile contact sensors --
    # Broad sensor: all 24 hand bodies, unfiltered. Reports net normal contact
    # force per body against ANY collider (table, object, other fingers, etc.).
    # Data shape: net_forces_w = (num_envs, 24, 3)
    # Use sensor.find_bodies(O6_FINGERTIP_BODIES) to index fingertips only.
    contact_hands = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + _O6_TACTILE_BODY_REGEX,
        update_period=0.0,       # every physics step (120Hz ≈ real O6 ~100-500Hz)
        history_length=6,        # ~50ms history at 120Hz for event detection
        track_air_time=True,     # contact/release timing
        force_threshold=0.05,    # 50mN — real O6 capacitive sensor ~10mN
        debug_vis=False,
    )

    # One Object-filtered sensor per collision-bearing hand link.  The O6
    # cannot make its thumb coplanar with the palm, so omitting thumb bases or
    # proximal links would hide real normal load from the 10 N controller.
    # Data shape per sensor: force_matrix_w = (num_envs, 1, 1, 3)
    lh_palm_object_contact = _object_contact_sensor_cfg("lh_hand_base_link")
    lh_thumb_base_object_contact = _object_contact_sensor_cfg("lh_thumb_metacarpals_base2")
    lh_thumb_metacarpals_object_contact = _object_contact_sensor_cfg("lh_thumb_metacarpals")
    lh_thumb_object_contact = _object_contact_sensor_cfg("lh_thumb_distal")
    lh_index_proximal_object_contact = _object_contact_sensor_cfg("lh_index_proximal")
    lh_index_object_contact = _object_contact_sensor_cfg("lh_index_distal")
    lh_middle_proximal_object_contact = _object_contact_sensor_cfg("lh_middle_proximal")
    lh_middle_object_contact = _object_contact_sensor_cfg("lh_middle_distal")
    lh_ring_proximal_object_contact = _object_contact_sensor_cfg("lh_ring_proximal")
    lh_ring_object_contact = _object_contact_sensor_cfg("lh_ring_distal")
    lh_pinky_proximal_object_contact = _object_contact_sensor_cfg("lh_pinky_proximal")
    lh_pinky_object_contact = _object_contact_sensor_cfg("lh_pinky_distal")
    rh_palm_object_contact = _object_contact_sensor_cfg("rh_hand_base_link")
    rh_thumb_base_object_contact = _object_contact_sensor_cfg("rh_thumb_metacarpals_base2")
    rh_thumb_metacarpals_object_contact = _object_contact_sensor_cfg("rh_thumb_metacarpals")
    rh_thumb_object_contact = _object_contact_sensor_cfg("rh_thumb_distal")
    rh_index_proximal_object_contact = _object_contact_sensor_cfg("rh_index_proximal")
    rh_index_object_contact = _object_contact_sensor_cfg("rh_index_distal")
    rh_middle_proximal_object_contact = _object_contact_sensor_cfg("rh_middle_proximal")
    rh_middle_object_contact = _object_contact_sensor_cfg("rh_middle_distal")
    rh_ring_proximal_object_contact = _object_contact_sensor_cfg("rh_ring_proximal")
    rh_ring_object_contact = _object_contact_sensor_cfg("rh_ring_distal")
    rh_pinky_proximal_object_contact = _object_contact_sensor_cfg("rh_pinky_proximal")
    rh_pinky_object_contact = _object_contact_sensor_cfg("rh_pinky_distal")

    # -- Selected delivery object (override via make_delivery_object_cfg) --
    object: RigidObjectCfg = OBJECT_CFG
