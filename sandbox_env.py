"""
Phase 1 sandbox environment for granular terrain interaction in SAPIEN 3.0.

This script creates:
1) A kinematic sandbox container (bottom + 4 walls)
2) 2500 small dynamic spheres as sand particles
3) A real-time viewer loop showing free-fall and pile-up under gravity
"""

from __future__ import annotations

import numpy as np
import sapien
from sapien.utils import Viewer

# Sand-like contact preset (dry, highly dissipative granular behavior).
SAND_STATIC_FRICTION = 1.75
SAND_DYNAMIC_FRICTION = 1.45
SAND_RESTITUTION = 0.0
SAND_DENSITY = 1700.0
SAND_LINEAR_DAMPING = 0.24
SAND_ANGULAR_DAMPING = 0.30
SAND_SOLVER_POS_ITERS = 12
SAND_SOLVER_VEL_ITERS = 4


def create_scene(timestep: float = 1 / 240.0, prefer_gpu: bool = True) -> sapien.Scene:
    """Create a SAPIEN 3.0 scene and prefer GPU PhysX when available."""
    scene: sapien.Scene

    if prefer_gpu:
        try:
            physx_system = sapien.physx.PhysxGpuSystem("cuda")
            render_system = sapien.render.RenderSystem()
            scene = sapien.Scene([physx_system, render_system])
            print("[Info] Using PhysX GPU system (cuda).")
        except Exception as exc:  # noqa: BLE001
            # Fallback keeps the script runnable on setups without GPU PhysX support.
            print(f"[Warn] GPU PhysX unavailable, fallback to CPU PhysX: {exc}")
            scene = sapien.Scene()
    else:
        scene = sapien.Scene()

    scene.set_timestep(timestep)
    return scene


def build_sandbox(scene: sapien.Scene) -> sapien.Entity:
    """Build a kinematic sandbox actor using one ActorBuilder."""
    # Inner usable region for particles.
    inner_half_x = 0.45
    inner_half_y = 0.35
    wall_thickness = 0.03
    wall_height = 0.30
    bottom_thickness = 0.04

    # High-friction material for container contact.
    sandbox_physical_material = scene.create_physical_material(
        static_friction=1.90,
        dynamic_friction=1.60,
        restitution=0.0,
    )

    # Slightly desaturated warm gray-brown for container visuals.
    sandbox_visual_material = sapien.render.RenderMaterial(
        base_color=[0.48, 0.41, 0.33, 1.0],
        specular=0.10,
        roughness=0.85,
        metallic=0.0,
    )

    builder = scene.create_actor_builder()

    # Bottom plate: top surface is aligned to z = 0.
    bottom_half_size = [inner_half_x + wall_thickness, inner_half_y + wall_thickness, bottom_thickness]
    bottom_pose = sapien.Pose(p=[0.0, 0.0, -bottom_thickness])
    builder.add_box_collision(
        pose=bottom_pose,
        half_size=bottom_half_size,
        material=sandbox_physical_material,
    )
    builder.add_box_visual(
        pose=bottom_pose,
        half_size=bottom_half_size,
        material=sandbox_visual_material,
        name="sandbox_bottom",
    )

    # +X wall / -X wall.
    wall_x_half_size = [wall_thickness, inner_half_y + wall_thickness, wall_height / 2.0]
    wall_x_offset = inner_half_x + wall_thickness
    wall_center_z = wall_height / 2.0
    for sign in (+1.0, -1.0):
        wall_pose = sapien.Pose(p=[sign * wall_x_offset, 0.0, wall_center_z])
        builder.add_box_collision(
            pose=wall_pose,
            half_size=wall_x_half_size,
            material=sandbox_physical_material,
        )
        builder.add_box_visual(
            pose=wall_pose,
            half_size=wall_x_half_size,
            material=sandbox_visual_material,
        )

    # +Y wall / -Y wall.
    wall_y_half_size = [inner_half_x, wall_thickness, wall_height / 2.0]
    wall_y_offset = inner_half_y + wall_thickness
    for sign in (+1.0, -1.0):
        wall_pose = sapien.Pose(p=[0.0, sign * wall_y_offset, wall_center_z])
        builder.add_box_collision(
            pose=wall_pose,
            half_size=wall_y_half_size,
            material=sandbox_physical_material,
        )
        builder.add_box_visual(
            pose=wall_pose,
            half_size=wall_y_half_size,
            material=sandbox_visual_material,
        )

    sandbox = builder.build_kinematic(name="sandbox")
    return sandbox


def spawn_sand_particles(
    scene: sapien.Scene,
    particle_count: int = 2500,
    particle_radius: float = 0.015,
) -> list[sapien.Entity]:
    """Spawn dynamic spheres above the sandbox as granular particles."""
    if particle_count != 2500:
        raise ValueError("This phase is configured for exactly 2500 particles.")

    # Sand-like contact behavior: high friction, very low restitution.
    sand_physical_material = scene.create_physical_material(
        static_friction=SAND_STATIC_FRICTION,
        dynamic_friction=SAND_DYNAMIC_FRICTION,
        restitution=SAND_RESTITUTION,
    )

    # Brown-yellow, rough, non-metallic appearance.
    sand_visual_material = sapien.render.RenderMaterial(
        base_color=[0.76, 0.66, 0.42, 1.0],
        specular=0.06,
        roughness=0.95,
        metallic=0.0,
    )

    particle_builder = scene.create_actor_builder()
    particle_builder.add_sphere_collision(
        radius=particle_radius,
        material=sand_physical_material,
        density=SAND_DENSITY,
    )
    particle_builder.add_sphere_visual(
        radius=particle_radius,
        material=sand_visual_material,
    )

    # 25 x 20 x 5 = 2500 particles arranged in a compact cloud above the sandbox.
    nx, ny, nz = 25, 20, 5
    spacing = particle_radius * 2.05

    x_coords = (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * spacing
    y_coords = (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * spacing
    z_start = 0.50
    z_coords = z_start + np.arange(nz, dtype=np.float32) * spacing

    gx, gy, gz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    initial_positions = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    # Slight random jitter reduces perfect lattice artifacts during pile formation.
    rng = np.random.default_rng(seed=42)
    initial_positions += rng.uniform(
        low=-0.12 * particle_radius,
        high=0.12 * particle_radius,
        size=initial_positions.shape,
    ).astype(np.float32)

    particles: list[sapien.Entity] = []
    for pos in initial_positions:
        particle_builder.set_initial_pose(sapien.Pose(p=pos.tolist()))
        particle = particle_builder.build()

        # Per-body damping and solver iterations improve granular stability.
        rigid = particle.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        rigid.set_linear_damping(SAND_LINEAR_DAMPING)
        rigid.set_angular_damping(SAND_ANGULAR_DAMPING)
        rigid.set_solver_position_iterations(SAND_SOLVER_POS_ITERS)
        rigid.set_solver_velocity_iterations(SAND_SOLVER_VEL_ITERS)

        particles.append(particle)

    return particles


def configure_lighting(scene: sapien.Scene) -> None:
    """Set practical lighting for observing particles and pile shape."""
    scene.set_ambient_light([0.35, 0.35, 0.35])
    scene.add_directional_light(
        direction=[0.3, -0.2, -1.0],
        color=[0.95, 0.92, 0.88],
        shadow=True,
    )
    scene.add_point_light(
        position=[0.0, 0.0, 1.8],
        color=[0.55, 0.52, 0.48],
        shadow=False,
    )


def run_viewer(scene: sapien.Scene) -> None:
    """Launch SAPIEN viewer loop and focus camera on sandbox center."""
    viewer = Viewer()
    viewer.set_scene(scene)

    # Camera roughly looks toward sandbox center.
    viewer.set_camera_xyz(x=-1.60, y=0.0, z=0.95)
    viewer.set_camera_rpy(r=0.0, p=-0.45, y=0.0)
    viewer.window.set_camera_parameters(near=0.01, far=20.0, fovy=np.deg2rad(60.0))

    print("[Info] Simulation started. Close the viewer window to exit.")
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()


def main() -> None:
    scene = create_scene(timestep=1 / 240.0, prefer_gpu=True)
    configure_lighting(scene)
    build_sandbox(scene)
    spawn_sand_particles(scene, particle_count=2500, particle_radius=0.015)
    run_viewer(scene)


if __name__ == "__main__":
    main()
