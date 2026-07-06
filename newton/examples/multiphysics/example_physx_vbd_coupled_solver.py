# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhysX-VBD Coupled Solver
#
# Three rigid boxes (SolverPhysX) drop onto a 60x60 VBD cloth patch
# (SolverVBD) via ``SolverCoupledProxy``. PhysX simulates the boxes;
# VBD simulates the cloth; the framework propagates box poses into VBD
# each step and feeds VBD's harvested reaction force back into PhysX
# on the next iteration.
#
# Command: python -m newton.examples physx_vbd_coupled_solver
#
###########################################################################

from __future__ import annotations

import os
import tempfile

import warp as wp
from newton.solvers.experimental.coupled import (
    PhysxParseInfo,  # noqa: F401
    SolverCoupledProxy,
    SolverPhysX,
)

import newton
import newton.examples
from newton.solvers import SolverVBD

_THREE_BOXES_USDA = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)
def Xform "World"
{
    def PhysicsScene "PhysicsScene"
    {
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }
    def Cube "Box0" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"])
    {
        double size = 1.0
        float3 xformOp:scale = (0.30, 0.30, 0.30)
        double3 xformOp:translate = (0, 0, 2.0)
        quatf xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        float physics:mass = 10.0
    }
    def Cube "Box1" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"])
    {
        double size = 1.0
        float3 xformOp:scale = (0.20, 0.40, 0.20)
        double3 xformOp:translate = (0.3, 0.1, 2.5)
        quatf xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        float physics:mass = 5.0
    }
    def Cube "Box2" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"])
    {
        double size = 1.0
        float3 xformOp:scale = (0.24, 0.24, 0.24)
        double3 xformOp:translate = (-0.2, -0.1, 3.0)
        quatf xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        float physics:mass = 8.0
    }
}
"""


def _write_usda(content: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="physx_vbd_")
    path = os.path.join(tmpdir, "boxes.usda")
    with open(path, "w") as f:
        f.write(content)
    return path


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()

        usd_path = _write_usda(_THREE_BOXES_USDA)
        parse_info = SolverPhysX.parse_usd(builder, usd_path)
        self.parse_info = parse_info
        self.box_indices = [parse_info.path_body_map[f"/World/Box{i}"] for i in range(3)]

        # 60x60 cloth pinned on left and right edges.
        builder.add_cloth_grid(
            pos=wp.vec3(-1.0, -1.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            fix_left=True,
            fix_right=True,
            dim_x=60,
            dim_y=60,
            cell_x=1.0 / 30.0,
            cell_y=1.0 / 30.0,
            mass=0.1,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e-2,
            edge_ke=0.01,
            edge_kd=1e-2,
            particle_radius=0.01,
        )

        builder.color()
        self.model = builder.finalize()

        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_mu = 0.5
        self.model.shape_material_kd.fill_(0.0)

        vbd_kwargs = {
            "iterations": 10,
            "friction_epsilon": 0.01,
            "particle_enable_self_contact": True,
            "particle_self_contact_radius": 0.01,
            "particle_self_contact_margin": 0.01,
            "rigid_contact_hard": False,
            "rigid_avbd_beta": 1.0e5,
            "rigid_avbd_gamma": 0.99,
            "rigid_contact_k_start": 1.0e2,
            "rigid_joint_linear_k_start": 1.0e4,
            "rigid_joint_angular_k_start": 1.0e1,
            "rigid_joint_linear_ke": 1.0e9,
            "rigid_joint_angular_ke": 1.0e9,
            "rigid_joint_linear_kd": 1.0e-2,
            "rigid_joint_angular_kd": 0.0,
        }

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="physx",
                    solver=lambda v: SolverPhysX(v, parse_info=parse_info),
                    bodies=self.box_indices,
                ),
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(model=v, **vbd_kwargs),
                    bodies=[],
                    particles=list(range(self.model.particle_count)),
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="physx",
                        destination="vbd",
                        bodies=self.box_indices,
                        mass_scale=1.0,
                        mode="lagged",
                        collision_pipeline=lambda m: newton.CollisionPipeline(m, broad_phase="explicit"),
                        collide_interval=1,
                    ),
                ],
                iterations=1,
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self._frame_count = 0

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._frame_count += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Boxes should rest on the cloth (cloth pinned at z=1.0) — not fall
        # through to negative z, and no coupling rocket-launch.
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "boxes rest on the cloth",
            lambda q, qd: q[2] > 0.0 and q[2] < 5.0,
            indices=self.box_indices,
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
