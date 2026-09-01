# -*- coding: utf-8 -*-
"""Isaac Lab rigid-body definitions for the two supported box profiles."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


MAT_CARDBOARD = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.60,
    dynamic_friction=0.50,
    restitution=0.05,
    friction_combine_mode="average",
    restitution_combine_mode="min",
)

MAT_EPS_FOAM = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.70,
    dynamic_friction=0.60,
    restitution=0.10,
    friction_combine_mode="average",
    restitution_combine_mode="min",
)


def _rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=1,
        max_depenetration_velocity=1.0,
        disable_gravity=False,
    )


def _cuboid(
    size: tuple[float, float, float],
    mass: float,
    physics_material: sim_utils.RigidBodyMaterialCfg,
    color: tuple[float, float, float],
    position: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/Object",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=physics_material,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.9,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


DELIVERY_OBJECTS: dict[str, RigidObjectCfg] = {
    "courier_box_m": _cuboid(
        size=(0.30, 0.22, 0.20),
        mass=1.0,
        physics_material=MAT_CARDBOARD,
        color=(0.72, 0.55, 0.35),
        position=(0.0, 0.0, 0.10),
    ),
    "foam_box": _cuboid(
        size=(0.35, 0.24, 0.25),
        mass=0.5,
        physics_material=MAT_EPS_FOAM,
        color=(0.95, 0.95, 0.95),
        position=(0.0, 0.0, 0.125),
    ),
}
