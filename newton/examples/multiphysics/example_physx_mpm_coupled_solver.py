# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhysX-MPM Coupled Solver
#
# A rigid box (SolverPhysX) drops onto an MPM sand bed (SolverImplicitMPM)
# via ``SolverCoupledProxy``. PhysX simulates the box; MPM simulates the
# sand particles; the framework propagates the box pose into MPM each
# step and feeds MPM's harvested reaction force back into PhysX's
# ``body_f`` on the next iteration.
#
# Command: python -m newton.examples physx_mpm_coupled_solver
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
from newton.solvers import SolverImplicitMPM

_BOX_USDA = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World"
{
    def PhysicsScene "physicsScene"
    {
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }
    def Plane "groundPlane" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {
        token axis = "Z"
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "Box" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"])
    {
        double size = 1.0
        float3 xformOp:scale = (0.5, 0.5, 0.5)
        double3 xformOp:translate = (0, 0, 2.0)
        quatf xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        float physics:mass = 75.0
    }
}
"""


def _write_usda(content: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="physx_mpm_")
    path = os.path.join(tmpdir, "box.usda")
    with open(path, "w") as f:
        f.write(content)
    return path


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.5
        SolverImplicitMPM.register_custom_attributes(builder)

        usd_path = _write_usda(_BOX_USDA)
        parse_info = SolverPhysX.parse_usd(builder, usd_path)
        self.box_idx = parse_info.path_body_map["/World/Box"]
        self.parse_info = parse_info

        # ---- MPM sand bed ---------------------------------------------
        voxel_size = 0.05
        self.voxel_size = voxel_size
        particles_per_cell = 3.0
        density = 2500.0

        bed_lo = np.array([-1.0, -1.0, 0.0])
        bed_hi = np.array([1.0, 1.0, 0.5])
        bed_res = np.array(np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size), dtype=int)
        cell_size = (bed_hi - bed_lo) / bed_res
        cell_volume = float(np.prod(cell_size))
        radius = float(np.max(cell_size) * 0.5)
        mass = float(cell_volume * density)

        builder.add_particle_grid(
            pos=wp.vec3(bed_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(bed_res[0]) + 1,
            dim_y=int(bed_res[1]) + 1,
            dim_z=int(bed_res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            custom_attributes={"mpm:friction": 0.75},
        )

        self.model = builder.finalize()

        # ---- MPM config -----------------------------------------------
        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = voxel_size
        mpm_config.grid_type = "fixed"
        mpm_config.grid_padding = 50
        mpm_config.max_active_cell_count = 1 << 15
        mpm_config.strain_basis = "P0"
        mpm_config.max_iterations = 50
        mpm_config.critical_fraction = 0.0

        # ---- Coupled solver -------------------------------------------
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="physx",
                    solver=lambda v: SolverPhysX(v, parse_info=parse_info),
                    bodies=[self.box_idx],
                    # PhysX internal substeps per coupled step.
                    substeps=4,
                ),
                SolverCoupledProxy.Entry(
                    name="mpm",
                    solver=lambda v: SolverImplicitMPM(model=v, config=mpm_config),
                    particles=list(range(self.model.particle_count)),
                    in_place=True,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="physx",
                        destination="mpm",
                        bodies=[self.box_idx],
                        mass_scale=1.0,
                        mode="lagged",
                        collision_pipeline=lambda _model: None,
                    ),
                ],
                iterations=1,
            ),
        )

        self.state_0 = self.model.state()
        self.rigid_collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.rigid_collision_pipeline.contacts()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.particle_render_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.7, 0.6, 0.4),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self._frame_count = 0

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.rigid_collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._frame_count += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_points(
            "/sand",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()

    def test_final(self):
        # Box should rest on the sand (~z=0.7), not punch through to the
        # ground or rocket-launch.
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "box rests on the sand",
            lambda q, qd: q[2] > 0.0 and q[2] < 1.5,
            indices=[self.box_idx],
        )
        voxel_size = self.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
