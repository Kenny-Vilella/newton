# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhysX-MPM Anymal Coupled Solver
#
# An Anymal quadruped articulation (SolverPhysX) stands on an MPM sand bed
# (SolverImplicitMPM) via ``SolverCoupledProxy``, holding its standing pose with
# PD drives so its feet settle into and displace the sand. PhysX simulates the
# articulation; MPM simulates the sand; the framework propagates the foot poses
# into MPM each step and harvests MPM's reaction force back onto the PhysX foot
# links (two-way coupling).
#
# Command: python -m newton.examples physx_mpm_anymal_coupled_solver
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

# Articulation root prim path inside the Anymal USD
ARTICULATION_ROOT = "/anymal/base"
FOOT_NAMES = ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]

# Standing pose for the 12 actuated joints, keyed by joint short name.
INITIAL_Q = {
    "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
    "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
    "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
    "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
}  # fmt: skip

# PD drive gains.
DRIVE_STIFFNESS = 400.0
DRIVE_DAMPING = 15.0


def _wrapper_usda(anymal_usd_path: str) -> str:
    """Sublayer the downloaded Anymal USD, place the robot at standing height,
    and add a static PhysX ground so it is supported regardless of MPM.
    """
    return f"""#usda 1.0
(
    subLayers = [@{anymal_usd_path}@]
    defaultPrim = "anymal"
    metersPerUnit = 1
    upAxis = "Z"
)
over "anymal"
{{
    double3 xformOp:translate = (0, 0, 0.62)
    quatf xformOp:orient = (1, 0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    # The asset requests 16 articulation velocity iterations; PhysX's TGS solver
    # warns above 4 (extra velocity iterations are wasteful in TGS). Cap it.
    over "base"
    {{
        int physxArticulation:solverVelocityIterationCount = 4
    }}
}}
def Cube "Ground" (prepend apiSchemas = ["PhysicsCollisionAPI"])
{{
    double size = 1.0
    float3 xformOp:scale = (20.0, 20.0, 0.05)
    double3 xformOp:translate = (0, 0, -0.025)
    quatf xformOp:orient = (1, 0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
}}
"""


def _write_usda(content: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="physx_mpm_anymal_")
    path = os.path.join(tmpdir, "anymal_with_ground.usda")
    with open(path, "w") as f:
        f.write(content)
    return path


def _body_index_by_suffix(model: newton.Model, name: str) -> int:
    """Newton body index whose label ends with ``/<name>``."""
    for i, label in enumerate(model.body_label or []):
        if label and label.endswith("/" + name):
            return i
    raise ValueError(f"Anymal body {name!r} not found in the parsed model.")


def _dof_indices_by_name(model: newton.Model, names: list[str]) -> dict[str, int]:
    """Map each joint short name to its Newton ``joint_qd`` (DOF) index."""
    joint_label = list(model.joint_label or [])
    joint_qd_start = model.joint_qd_start.numpy()
    out: dict[str, int] = {}
    for name in names:
        match = -1
        for i, label in enumerate(joint_label):
            if label and (label.endswith("/" + name) or label == name):
                match = int(joint_qd_start[i])
                break
        if match < 0:
            raise ValueError(f"Anymal joint {name!r} not found in the parsed model.")
        out[name] = match
    return out


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        # ---- Anymal USD (downloaded asset + ground/pose wrapper) ------
        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        anymal_usd = str(asset_path / "usd" / "anymal_c.usda")
        usd_path = _write_usda(_wrapper_usda(anymal_usd))

        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        parse_info = SolverPhysX.parse_usd(builder, usd_path)
        self.parse_info = parse_info

        # ---- MPM sand bed under the robot -----------------------------
        voxel_size = 0.03
        self.voxel_size = voxel_size
        particles_per_cell = 3.0
        density = 2500.0

        bed_lo = np.array([-1.0, -0.6, 0.0])
        bed_hi = np.array([1.0, 0.6, 0.05])
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

        builder.add_ground_plane()
        self.model = builder.finalize()

        # ---- Newton-side index lookups --------------------------------
        self.base_idx = _body_index_by_suffix(self.model, "base")
        foot_indices = [_body_index_by_suffix(self.model, n) for n in FOOT_NAMES]
        dof_indices = _dof_indices_by_name(self.model, list(INITIAL_Q.keys()))
        anymal_body_indices = list(parse_info.path_body_map.values())

        # ---- MPM config -----------------------------------------------
        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = voxel_size
        mpm_config.grid_type = "fixed"
        mpm_config.grid_padding = 30
        mpm_config.max_active_cell_count = 1 << 16
        mpm_config.strain_basis = "P0"
        mpm_config.max_iterations = 30
        mpm_config.critical_fraction = 0.0

        # ---- Coupled solver -------------------------------------------
        # Two-way: the feet are proxy source bodies whose poses drive MPM, and
        # MPM's harvested reaction force flows back onto the PhysX foot links.
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="physx",
                    solver=lambda v: SolverPhysX(v, parse_info=parse_info),
                    bodies=anymal_body_indices,
                    joints=list(range(self.model.joint_count)),
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
                        bodies=foot_indices,
                        mass_scale=1.0,
                        mode="lagged",
                        collision_pipeline=lambda _model: None,
                    ),
                ],
                iterations=1,
            ),
        )

        # The Anymal USD ships drives with zero gains, so install PD gains;
        # otherwise the position targets do nothing and the robot collapses.
        self.physx_solver = self.solver.solver("physx")
        self.physx_solver.set_articulation_drive_gains(
            ARTICULATION_ROOT, stiffness=DRIVE_STIFFNESS, damping=DRIVE_DAMPING
        )

        # ---- State / control ------------------------------------------
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()
        self.control = self.model.control()

        # Hold the standing pose for the whole run (the quadruped stance is
        # statically stable, so fixed targets keep the robot upright).
        target = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        for name, value in INITIAL_Q.items():
            target[dof_indices[name]] = value
        self.control.joint_target_q.assign(target)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.particle_render_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.7, 0.6, 0.4),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self._init_particle_z = self.state_0.particle_q.numpy()[:, 2].copy()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.model.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

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
        # Robot should still be standing (base near its 0.62 m spawn height
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "anymal base stays standing",
            lambda q, qd: q[2] > 0.3 and q[2] < 1.0,
            indices=[self.base_idx],
        )
        # The feet should have moved some particles
        dz = self.state_0.particle_q.numpy()[:, 2] - self._init_particle_z
        moved = int((np.abs(dz) > 0.005).sum())
        assert moved > 0, "no sand particles moved — coupling did not fire"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
