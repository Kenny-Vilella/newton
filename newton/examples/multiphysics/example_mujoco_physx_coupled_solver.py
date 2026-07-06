# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MuJoCo-PhysX Coupled Solver
#
# nv_humanoid (mjwarp source) standing on a synthetic static-prop scene
# owned by SolverPhysX (the destination). SolverPhysX hosts compliant-
# contact "mirror" prims that mirror each humanoid link; PhysX's contact
# pipeline computes reaction forces on the mirrors from the static props,
# and the framework propagates those forces back to the mjwarp humanoid
# as ``body_f`` on the next coupled iteration.
#
# Mirror USD is generated programmatically from the MJCF in setUp by
# walking the humanoid's bounding boxes — no proprietary geometry, no
# external asset dependency beyond ``newton.examples.get_asset``.
#
# Command: python -m newton.examples mujoco_physx_coupled_solver
#
###########################################################################

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import (
    PhysxParseInfo,  # noqa: F401
    SolverCoupledProxy,
    SolverPhysX,
)

import newton
import newton.examples
from newton._src.geometry.types import GeoType
from newton.solvers import SolverMuJoCo

HUMANOID_MJCF = newton.examples.get_asset("nv_humanoid.xml")
HUMANOID_INITIAL_XFORM = wp.transform((0.0, 0.0, 1.4), wp.quat_identity())

# Mirror prims live under this scope inside the scene USD. They are
# park-positioned below the floor at construction time; the framework's
# proxy mechanism teleports them to the humanoid body poses each step.
MIRRORS_ROOT = "/World/Mirrors"

PYRAMID_COUNT = 5
PYRAMID_BASE_N = 6
PYRAMID_RING_RADIUS = 3.0
PYRAMID_CUBE_SIZE = 0.2
PYRAMID_CUBE_MASS = 0.5


def _humanoid_mirror_descriptions(mjcf_path: str) -> list[dict]:
    """Parse ``mjcf_path`` into a temporary ``ModelBuilder``, walk each
    body's shapes, and return per-body axis-aligned bounding boxes
    plus the shape-local transforms. Used to author one rigid-body
    mirror prim per humanoid body in the scene USD.
    """
    tb = newton.ModelBuilder()
    tb.add_mjcf(
        mjcf_path,
        ignore_names=["floor", "ground"],
        enable_self_collisions=False,
        parse_sites=False,
    )
    tb.approximate_meshes("bounding_box")

    shapes_by_body: dict[int, list[int]] = {}
    for s in range(tb.shape_count):
        body = int(tb.shape_body[s])
        if body < 0:
            continue
        shapes_by_body.setdefault(body, []).append(s)

    descriptions: list[dict] = []
    for body_idx, shape_idxs in shapes_by_body.items():
        # Skip light fingertip-style bodies for coupling stability
        if float(tb.body_mass[body_idx]) < 0.1:
            continue
        label = tb.body_label[body_idx] if body_idx < len(tb.body_label) else f"body_{body_idx}"
        body_short = (label or "").rsplit("/", 1)[-1] or f"body_{body_idx}"

        per_shape: list[tuple[np.ndarray, wp.transform]] = []
        for s in shape_idxs:
            stype = int(tb.shape_type[s])
            sz = np.asarray(tb.shape_scale[s], dtype=np.float64)
            if stype == int(GeoType.BOX):
                size = sz * 2.0
            elif stype == int(GeoType.SPHERE):
                r = float(sz[0])
                size = np.array([2 * r, 2 * r, 2 * r])
            elif stype == int(GeoType.CAPSULE):
                r, h_half = float(sz[0]), float(sz[1])
                size = np.array([2 * r, 2 * r, 2 * (r + h_half)])
            else:
                continue
            if float(size.min()) <= 0.0:
                continue
            per_shape.append((size, tb.shape_transform[s]))

        if not per_shape:
            continue
        descriptions.append({"body_short": body_short, "shapes": per_shape})
    return descriptions


def _pyramid_usda(pyramid_idx: int, center_x: float, center_y: float) -> str:
    """Author a single ``PYRAMID_BASE_N``-base pyramid of dynamic cubes
    centered at ``(center_x, center_y, 0)``. Each cube is its own
    rigid body; PhysX defaults are used (no compliant binding) so the
    stack behaves like classic rigid blocks rather than springs."""
    s = PYRAMID_CUBE_SIZE
    cubes = []
    for layer in range(PYRAMID_BASE_N):
        n = PYRAMID_BASE_N - layer
        offset = (n - 1) * s / 2.0
        z = (layer + 0.5) * s
        for i in range(n):
            for j in range(n):
                x = center_x - offset + i * s
                y = center_y - offset + j * s
                cubes.append(f"""
            def Cube "L{layer}_R{i}_C{j}" (
                prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI", "PhysxContactReportAPI"]
            )
            {{
                double size = 1
                float3 xformOp:scale = ({s}, {s}, {s})
                double3 xformOp:translate = ({x:.4f}, {y:.4f}, {z:.4f})
                quatf xformOp:orient = (1, 0, 0, 0)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
                float physics:mass = {PYRAMID_CUBE_MASS}
            }}""")
    return f"""
        def Xform "Pyramid_{pyramid_idx}"
        {{{"".join(cubes)}
        }}"""


def _format_quat_xyzw(t: wp.transform) -> tuple[float, float, float, float]:
    arr = np.asarray(t).reshape(-1)
    return float(arr[3]), float(arr[4]), float(arr[5]), float(arr[6])


def _format_translation(t: wp.transform) -> tuple[float, float, float]:
    arr = np.asarray(t).reshape(-1)
    return float(arr[0]), float(arr[1]), float(arr[2])


def _build_scene_usda(descriptions: list[dict]) -> str:
    """Author the full scene USDA: PhysicsScene + ground floor + a few
    static props + one rigid-body mirror per humanoid body. Mirrors
    are parked far below the floor at construction time; the framework
    teleports them to humanoid poses each step.
    """
    # Compliant-contact material — bound on each mirror's collider.
    # PhysxMaterialAPI must be applied via apiSchemas in the USD.
    material = """
    def Material "CompliantMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
    )
    {
        float physics:dynamicFriction = 0.7
        float physics:staticFriction = 0.7
        float physics:restitution = 0.0
        bool physxMaterial:compliantContactAccelerationSpring = 1
        float physxMaterial:compliantContactStiffness = 2500
        float physxMaterial:compliantContactDamping = 500
    }
"""

    # Static ground plane (bound to the compliant-contact material).
    static_props = """
    def Plane "Floor" (prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI", "PhysxContactReportAPI"])
    {
        uniform token axis = "Z"
        double3 xformOp:translate = (0, 0, 0)
        quatf xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        rel material:binding:physics = </World/Mirrors/CompliantMaterial>
    }
"""

    # Mirror prims. Each is parked at (-30, 0, -5+i*0.5) — far below the
    # floor so PhysX doesn't bake any source-pose contact at construction.
    # The framework teleports them per step.
    #
    # Apply self-collision filter.
    mirror_paths_for_group: list[str] = []
    mirror_blocks = []
    for i, desc in enumerate(descriptions):
        body_short = desc["body_short"]
        mirror_name = f"{body_short}_Mirror"
        mirror_paths_for_group.append(f"{MIRRORS_ROOT}/{mirror_name}")
        park_z = -5.0 + 0.5 * i

        shape_blocks = []
        for j, (size, xform) in enumerate(desc["shapes"]):
            tx, ty, tz = _format_translation(xform)
            qx, qy, qz, qw = _format_quat_xyzw(xform)
            sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
            shape_blocks.append(f"""
            def Cube "shape_{j}" (
                prepend apiSchemas = [
                    "PhysicsCollisionAPI",
                    "PhysxContactReportAPI",
                    "MaterialBindingAPI",
                ]
            )
            {{
                double size = 1
                float3 xformOp:scale = ({sx}, {sy}, {sz})
                double3 xformOp:translate = ({tx}, {ty}, {tz})
                quatf xformOp:orient = ({qw}, {qx}, {qy}, {qz})
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
                rel material:binding:physics = <{MIRRORS_ROOT}/CompliantMaterial>
            }}""")

        shapes_text = "".join(shape_blocks)
        mirror_blocks.append(f"""
        def Xform "{mirror_name}" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxContactReportAPI"]
        )
        {{
            float physics:mass = 1.0
            double3 xformOp:translate = (-30.0, 0.0, {park_z})
            uniform token[] xformOpOrder = ["xformOp:translate"]
{shapes_text}
        }}""")

    mirrors_text = "".join(mirror_blocks)

    # Stacked pyramids of dynamic cubes, ring-arranged around origin.
    pyramid_blocks = []
    for i in range(PYRAMID_COUNT):
        theta = 2.0 * math.pi * i / PYRAMID_COUNT
        cx = PYRAMID_RING_RADIUS * math.cos(theta)
        cy = PYRAMID_RING_RADIUS * math.sin(theta)
        pyramid_blocks.append(_pyramid_usda(i, cx, cy))
    pyramids_text = "".join(pyramid_blocks)

    group_path = f"{MIRRORS_ROOT}/NoSelfCollideGroup"
    includes_text = ",\n            ".join(f"<{p}>" for p in mirror_paths_for_group)
    self_filter = f"""
        def PhysicsCollisionGroup "NoSelfCollideGroup" (
            prepend apiSchemas = ["CollectionAPI:colliders"]
        )
        {{
            uniform token collection:colliders:expansionRule = "expandPrims"
            rel collection:colliders:includes = [
            {includes_text}
            ]
            rel physics:filteredGroups = <{group_path}>
        }}"""

    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World"
{{
    def PhysicsScene "physicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}
{static_props}
    def Xform "Mirrors"
    {{
{material}
{self_filter}
{mirrors_text}
    }}
    def Xform "Pyramids"
    {{
{pyramids_text}
    }}
}}
"""


def _write_usda(content: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="mjwarp_physx_")
    path = os.path.join(tmpdir, "scene.usda")
    with open(path, "w") as f:
        f.write(content)
    return path


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        # ---- Mirror descriptions from the humanoid MJCF ---------------
        descriptions = _humanoid_mirror_descriptions(HUMANOID_MJCF)
        scene_usda = _build_scene_usda(descriptions)
        usd_path = _write_usda(scene_usda)

        # ---- Build the coupled scene ----------------------------------
        # Build the humanoid FIRST so its bodies and joints occupy a contiguous
        # prefix [0, N), letting the framework restrict mjwarp's view via the fast
        # prefix path. (Shapes are kept by global id, so their order is moot.)
        # The PhysX scene (mirrors + pyramid cubes + floor) is appended afterwards.
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        builder.add_mjcf(
            HUMANOID_MJCF,
            xform=HUMANOID_INITIAL_XFORM,
            ignore_names=["floor", "ground"],
            enable_self_collisions=False,
            parse_sites=False,
        )
        self.humanoid_body_indices = list(range(builder.body_count))
        humanoid_joint_count = builder.joint_count
        builder.approximate_meshes("bounding_box", keep_visual_shapes=True)

        parse_info = SolverPhysX.parse_usd(builder, usd_path, skip_mesh_approximation=True)
        self.parse_info = parse_info

        # Maps humanoid body short-name to mirror Newton body index.
        # SolverPhysX-owned bodies are the mirrors; everything else
        # (the floor) is a static collider with no Newton body.
        mirror_path_by_short = {d["body_short"]: f"{MIRRORS_ROOT}/{d['body_short']}_Mirror" for d in descriptions}
        mirror_idx_by_short = {
            short: parse_info.path_body_map[path]
            for short, path in mirror_path_by_short.items()
            if path in parse_info.path_body_map
        }

        # Pair humanoid bodies with their mirrors by suffix match.
        body_labels = list(builder.body_label or [])
        self.proxy_source_bodies: list[int] = []
        self.proxy_dest_bodies: list[int] = []
        for short, mirror_idx in mirror_idx_by_short.items():
            matches = [i for i in self.humanoid_body_indices if body_labels[i] and body_labels[i].endswith(short)]
            if len(matches) != 1:
                continue
            self.proxy_source_bodies.append(matches[0])
            self.proxy_dest_bodies.append(mirror_idx)
            # Override the mirror's model-level mass/inertia to match the
            # source body. The viewer's pick force computation reads
            # ``model.body_mass`` directly — at 1 kg (the default in the
            # mirror USD) the max pick force is tiny.
            builder.body_mass[mirror_idx] = builder.body_mass[matches[0]]
            builder.body_inertia[mirror_idx] = builder.body_inertia[matches[0]]
            if builder.body_mass[matches[0]] > 0.0:
                builder.body_inv_mass[mirror_idx] = 1.0 / builder.body_mass[matches[0]]

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        ovphysx_body_indices = list(parse_info.path_body_map.values())
        config = SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="mjwarp",
                    destination="physx",
                    bodies=self.proxy_source_bodies,
                    proxy_bodies=self.proxy_dest_bodies,
                    mass_scale=1.0,
                    mode="lagged",
                    # Mirrors are real PhysX-owned bodies driven kinematically by
                    # the humanoid each step, so the proxy targets coincide with
                    # the destination's owned set.
                    destination_owned=True,
                    collision_pipeline=lambda _m: None,
                ),
            ],
            iterations=1,
        )

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mjwarp",
                    solver=lambda v: SolverMuJoCo(
                        model=v,
                        use_mujoco_contacts=False,
                        njmax=200,
                    ),
                    bodies=self.humanoid_body_indices,
                    joints=list(range(humanoid_joint_count)),
                    # 2 inner steps per outer call: mjwarp at 400 Hz
                    # while PhysX stays at the 200 Hz outer rate.
                    substeps=2,
                ),
                SolverCoupledProxy.Entry(
                    name="physx",
                    solver=lambda v: SolverPhysX(v, parse_info=parse_info),
                    bodies=ovphysx_body_indices,
                ),
            ],
            coupling=config,
        )

        # ``finalize_coupling`` pushes source-body mass / inertia / COM
        # onto the PhysX-side mirrors so the viewer's pick force and
        # the contact dynamics see correct rigid-body inertials.
        self.physx_solver = self.solver.solver("physx")
        self.physx_solver.finalize_coupling(config.proxies)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.model.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            # The viewer's pick raycast can hit a mirror prim (which sits
            # at the same world location as its source humanoid body).
            # Redirect any pick force that landed on a mirror to its
            # source body.
            # This is a trick needed for user interaction only.
            body_f_np = self.state_0.body_f.numpy()
            redirected = False
            for src_idx, mirror_idx in zip(self.proxy_source_bodies, self.proxy_dest_bodies, strict=True):
                f = body_f_np[mirror_idx]
                if float(np.linalg.norm(f)) > 0.0:
                    body_f_np[src_idx] += f
                    body_f_np[mirror_idx] = 0.0
                    redirected = True
            if redirected:
                self.state_0.body_f.assign(body_f_np)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Root body should still be above the floor with no explosion
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "humanoid root stays above the floor",
            lambda q, qd: q[2] > -0.5 and q[2] < 5.0,
            indices=[self.humanoid_body_indices[0]],
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
