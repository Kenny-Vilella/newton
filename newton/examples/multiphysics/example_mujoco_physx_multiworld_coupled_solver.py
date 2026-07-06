# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MuJoCo-PhysX Multi-World Coupled Solver
#
# N parallel worlds of nv_humanoid (mjwarp source) standing on per-world
# static floors with a pyramid of dynamic cubes (both owned by SolverPhysX,
# the destination). SolverPhysX hosts compliant-contact "mirror" prims that
# mirror each humanoid link; PhysX's contact pipeline computes reaction
# forces on the mirrors against the floor and the pyramid, and the framework
# propagates those forces back to each mjwarp humanoid as ``body_f`` on the
# next coupled iteration.
#
# The per-env scene is authored once under ``SolverPhysX.TEMPLATE_ROOT``
# and instantiated N times via ``physx.clone``; the humanoid is added to
# the same template via ``pre_replicate_extend`` so mjwarp and PhysX
# bodies share env world tags. Envs overlap at the world origin physically
# — PhysX env_id and Newton ``body_world`` isolate contacts; the viewer
# applies an XY grid for visual separation only.
#
# Command: python -m newton.examples mujoco_physx_multiworld_coupled_solver
#
###########################################################################

from __future__ import annotations

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

# Per-world content lives under this scope inside the scene USD. Mirror
# prims are park-positioned below the floor at construction time; the
# framework's proxy mechanism teleports them to the humanoid body poses
# each step. ``SolverPhysX.parse_usd`` clones this scope into
# ``/World/envs/env_<i>`` for ``world_count`` parallel instances.
TEMPLATE_ROOT = SolverPhysX.TEMPLATE_ROOT
MIRRORS_SCOPE = f"{TEMPLATE_ROOT}/Mirrors_NV"
PYRAMID_SCOPE = f"{TEMPLATE_ROOT}/Pyramid"

PYRAMID_CUBE_SIZE = 0.15
PYRAMID_CUBE_MASS = 0.3
PYRAMID_CENTER = (0.0, 0.9, 0.0)


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


def _format_quat_xyzw(t: wp.transform) -> tuple[float, float, float, float]:
    arr = np.asarray(t).reshape(-1)
    return float(arr[3]), float(arr[4]), float(arr[5]), float(arr[6])


def _format_translation(t: wp.transform) -> tuple[float, float, float]:
    arr = np.asarray(t).reshape(-1)
    return float(arr[0]), float(arr[1]), float(arr[2])


def _pyramid_usda(base_n: int) -> str:
    """Author a single pyramid of dynamic cubes under ``PYRAMID_SCOPE``.

    Each layer ``L`` has ``(base_n - L)^2`` cubes arranged in a square; layer 0
    is the bottom. Each cube is its own rigid body with PhysX defaults.
    """
    cx, cy, _cz = PYRAMID_CENTER
    s = PYRAMID_CUBE_SIZE
    blocks = []
    for layer in range(base_n):
        n = base_n - layer
        if n <= 0:
            break
        row_offset = (n - 1) * s / 2.0
        z = (layer + 0.5) * s
        for i in range(n):
            for j in range(n):
                x = cx - row_offset + i * s
                y = cy - row_offset + j * s
                blocks.append(f"""
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
    cubes_text = "".join(blocks)
    return f"""
            def Xform "Pyramid"
            {{
{cubes_text}
            }}"""


def _build_template_usda(descriptions: list[dict], pyramid_base: int) -> str:
    """Author the per-env template USDA: a single ``PhysicsScene`` at
    ``/World`` (shared across envs) plus a static floor, one rigid-body
    mirror per humanoid body, and an optional pyramid of dynamic cubes —
    all under ``TEMPLATE_ROOT``. Mirrors are parked far below the floor
    at construction time; the framework teleports them to humanoid
    poses each step.
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
    static_floor = f"""
        def Plane "Floor" (prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI", "PhysxContactReportAPI"])
        {{
            uniform token axis = "Z"
            double3 xformOp:translate = (0, 0, 0)
            quatf xformOp:orient = (1, 0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
            rel material:binding:physics = <{MIRRORS_SCOPE}/CompliantMaterial>
        }}
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
        mirror_paths_for_group.append(f"{MIRRORS_SCOPE}/{mirror_name}")
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
                    rel material:binding:physics = <{MIRRORS_SCOPE}/CompliantMaterial>
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
    group_path = f"{MIRRORS_SCOPE}/NoSelfCollideGroup"
    includes_text = ",\n                ".join(f"<{p}>" for p in mirror_paths_for_group)
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

    # Single pyramid of dynamic cubes per env, planted in front of the
    # humanoid. ``pyramid_base == 0`` opts out for stress-free scenes.
    pyramid_block = _pyramid_usda(pyramid_base) if pyramid_base > 0 else ""

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
    def Xform "envs"
    {{
        def Xform "env_template"
        {{
{static_floor}
            def Xform "Mirrors_NV"
            {{
{material}
{self_filter}
{mirrors_text}
            }}
{pyramid_block}
        }}
    }}
}}
"""


def _write_usda(content: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="mjwarp_physx_multienv_")
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
        self.world_count = max(1, int(args.world_count))
        self.pyramid_base = int(getattr(args, "pyramid_base", 3) or 0)
        self._env_spacing = (5.0, 5.0, 0.0)

        # ---- Mirror descriptions from the humanoid MJCF ---------------
        descriptions = _humanoid_mirror_descriptions(HUMANOID_MJCF)
        scene_usda = _build_template_usda(descriptions, self.pyramid_base)
        usd_path = _write_usda(scene_usda)

        # ---- Build the coupled scene ----------------------------------
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        def _add_humanoid_to_template(template_builder: newton.ModelBuilder) -> None:
            SolverMuJoCo.register_custom_attributes(template_builder)
            template_builder.add_mjcf(
                HUMANOID_MJCF,
                xform=HUMANOID_INITIAL_XFORM,
                ignore_names=["floor", "ground"],
                enable_self_collisions=False,
                parse_sites=False,
            )
            template_builder.approximate_meshes("bounding_box", keep_visual_shapes=True)

        parse_info = SolverPhysX.parse_usd(
            builder,
            usd_path,
            num_envs=self.world_count,
            pre_replicate_extend=_add_humanoid_to_template,
            skip_mesh_approximation=True,
        )
        self.parse_info = parse_info

        # Pair humanoid bodies with PhysX mirror bodies, per env. Mirror prim
        # paths in parse_info.path_body_map carry the env scope (e.g.
        # /World/envs/env_0/Mirrors_NV/torso_Mirror); humanoid body labels keep
        # the original MJCF short names (e.g. "torso") since add_mjcf does not
        # prefix labels. We match by short-name suffix within the same world.
        # At ``world_count == 1`` the parse runs single-env and everything lands
        # at ``body_world == -1``; we treat that as the implicit "the env" so
        # the pairing logic below works uniformly at any N.
        body_world = np.asarray(builder.body_world, dtype=np.int64)
        body_labels = list(builder.body_label or [])

        mirror_set = set(parse_info.path_body_map.values())
        mirror_idx_by_world_short: dict[tuple[int, str], int] = {}
        for path, body_idx in parse_info.path_body_map.items():
            tail = path.rsplit("/", 1)[-1]
            if not tail.endswith("_Mirror"):
                continue
            short = tail[: -len("_Mirror")]
            mirror_idx_by_world_short[(int(body_world[body_idx]), short)] = body_idx

        humanoid_bodies: list[int] = []
        proxy_source_bodies: list[int] = []
        proxy_dest_bodies: list[int] = []
        humanoid_pose_pairs: list[tuple[int, int]] = [] 

        for i, label in enumerate(body_labels):
            if i in mirror_set:
                continue
            if not label:
                continue
            world = int(body_world[i])
            short = label.rsplit("/", 1)[-1]
            mirror_idx = mirror_idx_by_world_short.get((world, short))
            if mirror_idx is None:
                # Humanoid body without a matching mirror (e.g. weight-skipped
                # fingertip body) — keep it on the mjwarp side, just unpaired.
                humanoid_bodies.append(i)
                continue
            humanoid_bodies.append(i)
            proxy_source_bodies.append(i)
            proxy_dest_bodies.append(mirror_idx)
            humanoid_pose_pairs.append((i, mirror_idx))

        # Override the mirror's model-level mass/inertia to match the
        # source body. The viewer's pick force computation reads
        # ``model.body_mass`` directly — at 1 kg (the default in the
        # mirror USD) the max pick force is tiny.
        for h, m in humanoid_pose_pairs:
            if float(builder.body_mass[h]) > 0.0:
                builder.body_mass[m] = builder.body_mass[h]
                builder.body_inertia[m] = builder.body_inertia[h]
                builder.body_inv_mass[m] = 1.0 / builder.body_mass[h]

        # All joints with a humanoid child are mjwarp-owned; everything else
        # (mirrors, pyramid cubes) is PhysX-owned and rigid.
        humanoid_body_set = set(humanoid_bodies)
        humanoid_joints = [
            j for j in range(builder.joint_count) if int(builder.joint_child[j]) in humanoid_body_set
        ]

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.humanoid_body_indices = humanoid_bodies
        self.proxy_source_bodies = proxy_source_bodies
        self.proxy_dest_bodies = proxy_dest_bodies
        self.ovphysx_body_indices = sorted(mirror_set)

        config = SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="mjwarp",
                    destination="physx",
                    bodies=proxy_source_bodies,
                    proxy_bodies=proxy_dest_bodies,
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
                        njmax=512,
                        nconmax=2048,
                    ),
                    bodies=humanoid_bodies,
                    joints=humanoid_joints,
                    # 2 inner steps per outer call: mjwarp at 400 Hz
                    # while PhysX stays at the 200 Hz outer rate.
                    substeps=2,
                ),
                SolverCoupledProxy.Entry(
                    name="physx",
                    solver=lambda v: SolverPhysX(v, parse_info=parse_info),
                    bodies=self.ovphysx_body_indices,
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
        # Envs overlap at the origin physically; lay them out on an XY grid
        # in the viewer purely for visual separation.
        if self.world_count > 1 and any(self._env_spacing):
            self.viewer.set_world_offsets(self._env_spacing)

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
        # Humanoid bodies must stay above the floor with no explosion in any env.
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "humanoid bodies stay above the floor in each env",
            lambda q, qd: q[2] > -0.5 and q[2] < 5.0,
            indices=self.humanoid_body_indices,
        )
        # Mirrors should be tracking their source bodies within ~0.5 m
        # (lagged proxy => 1-step delay).
        body_q = self.state_0.body_q.numpy()
        for s, m in zip(self.proxy_source_bodies, self.proxy_dest_bodies, strict=True):
            dist = float(np.linalg.norm(body_q[s, :3] - body_q[m, :3]))
            assert dist < 0.5, f"mirror lag too large: src={s} mirror={m} dist={dist:.3f}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=4)
        parser.add_argument(
            "--pyramid-base",
            type=int,
            default=8,
            help="Pyramid base size per world (0 disables pyramids).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
