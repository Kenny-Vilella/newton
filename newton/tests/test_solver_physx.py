# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverPhysX.

Convention follows ``newton/tests/test_coupled_solver.py``:
``unittest.TestCase`` subclasses, ``@unittest.skipUnless`` for GPU,
``setUp`` + ``self.skipTest`` for the ovphysx optional dependency.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, is_dataclass

import warp as wp

from newton import ModelBuilder, StateFlags
from newton._src.solvers.coupled.interface import CouplingInterface
from newton._src.solvers.solver import SolverBase
from newton.solvers.experimental.coupled import (
    PhysxParseInfo,
    SolverPhysX,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ovphysx_available() -> bool:
    try:
        import ovphysx  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


_OVPHYSX = _ovphysx_available()
_CUDA = wp.is_cuda_available()


_TINY_USDA = """#usda 1.0
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
    def Cube "Floor" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {
        double size = 1
        float3 xformOp:scale = (10.0, 10.0, 0.05)
        double3 xformOp:translate = (0, 0, -0.025)
        quatf xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }
    def Cube "DynamicBox" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI", "PhysxContactReportAPI"])
    {
        double size = 1
        float3 xformOp:scale = (0.2, 0.2, 0.2)
        double3 xformOp:translate = (0, 0, 5.0)
        quatf xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        float physics:mass = 1.0
    }
}
"""


def _write_usda(content: str) -> str:
    """Write a USDA string to a temp file and return its path."""
    tmpdir = tempfile.mkdtemp(prefix="solver_physx_test_")
    path = os.path.join(tmpdir, "scene.usda")
    with open(path, "w") as f:
        f.write(content)
    return path


# Process isolation for ovphysx tests.
#
# ovphysx imposes a process-singleton constraint on its ``PhysX``
# instance: only one can be constructed per process over the process's
# lifetime (Carbonite cannot finalize-and-reinit, so ``release()`` does
# not lift the limit). That breaks ``unittest``'s "one process, many
# tests" model for every class that constructs ``SolverPhysX``.
#
# ``_PhysxIsolatedTestCase`` resolves this by running each test method in
# its own subprocess: the normally-invoked (parent) ``run`` re-dispatches
# the single test into a fresh ``python -m unittest <id>`` process, which
# constructs exactly one PhysX. Skips for missing CUDA / ovphysx are
# decided in the parent so no subprocess is spawned and reporting stays
# accurate. This lets the full suite run under ``python -m newton.tests``
# (and the parallel runner) instead of one-test-at-a-time by hand.

_IN_SUBPROCESS_ENV = "NEWTON_PHYSX_TEST_ISOLATED"

# Importable dotted path to this module, derived from its location so the
# subprocess dispatch works regardless of how the parent test harness named
# the module (the parallel runner uses the bare ``test_solver_physx``, which
# ``python -m unittest`` cannot import). ``newton/tests/test_solver_physx.py``
# -> ``newton.tests.test_solver_physx``.
_MODULE_DOTTED = ".".join(pathlib.Path(__file__).resolve().with_suffix("").parts[-3:])


class _PhysxIsolatedTestCase(unittest.TestCase):
    """Base for tests that construct a ``SolverPhysX`` (directly or via a
    coupled solver). Each test method runs in its own subprocess; see the
    module comment above for why."""

    def run(self, result=None):
        # Child process: the env var is set, so run the real test body.
        if os.environ.get(_IN_SUBPROCESS_ENV) == "1":
            return super().run(result)

        # Parent process: decide skips here, then dispatch to a subprocess.
        if result is None:
            result = self.defaultTestResult()
        result.startTest(self)
        try:
            if not _CUDA:
                result.addSkip(self, "Requires CUDA")
                return result
            if not _OVPHYSX:
                result.addSkip(self, "ovphysx is not installed")
                return result

            test_id = f"{_MODULE_DOTTED}.{self.__class__.__qualname__}.{self._testMethodName}"
            env = os.environ.copy()
            env[_IN_SUBPROCESS_ENV] = "1"
            proc = subprocess.run(
                [sys.executable, "-m", "unittest", "-v", test_id],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            if proc.returncode == 0:
                result.addSuccess(self)
            else:
                msg = (
                    f"isolated subprocess for {test_id} exited with code "
                    f"{proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
                    f"--- stderr ---\n{proc.stderr}"
                )
                try:
                    raise AssertionError(msg)
                except AssertionError:
                    result.addFailure(self, sys.exc_info())
        finally:
            result.stopTest(self)
        return result


# ---------------------------------------------------------------------------
# Layer 1: API-surface contracts — no ovphysx, no GPU, runs everywhere.
# ---------------------------------------------------------------------------


class TestPhysxParseInfo(unittest.TestCase):
    """``PhysxParseInfo`` dataclass shape."""

    def test_is_dataclass(self):
        self.assertTrue(is_dataclass(PhysxParseInfo))

    def test_is_frozen(self):
        info = PhysxParseInfo(
            usd_path="/tmp/scene.usda",
            prim_paths=frozenset({"/World/Cube"}),
            path_body_map={"/World/Cube": 0},
        )
        with self.assertRaises(FrozenInstanceError):
            info.usd_path = "/tmp/other.usda"  # type: ignore[misc]

    def test_required_fields_present(self):
        fields = {f.name for f in PhysxParseInfo.__dataclass_fields__.values()}
        self.assertEqual(
            fields,
            {
                "usd_path",
                "prim_paths",
                "path_body_map",
                "articulation_roots",
                "num_envs",
                "template_root",
                "env_roots",
            },
        )

    def test_construction_with_explicit_fields(self):
        info = PhysxParseInfo(
            usd_path="/tmp/scene.usda",
            prim_paths=frozenset({"/World/Cube"}),
            path_body_map={"/World/Cube": 0},
        )
        self.assertEqual(info.usd_path, "/tmp/scene.usda")
        self.assertIn("/World/Cube", info.prim_paths)
        self.assertEqual(info.path_body_map["/World/Cube"], 0)


class TestSolverPhysXInheritance(unittest.TestCase):
    """``SolverPhysX`` must inherit from ``SolverBase`` and
    ``CouplingInterface`` so the framework's discovery / dispatch paths
    pick it up correctly."""

    def test_inherits_solver_base(self):
        self.assertTrue(issubclass(SolverPhysX, SolverBase))

    def test_inherits_coupling_interface(self):
        self.assertTrue(issubclass(SolverPhysX, CouplingInterface))


def _hook_raises_not_implemented(method, *args, **kwargs) -> bool:
    """Return whether a ``SolverPhysX`` coupling hook is declared unsupported.

    The framework dropped the ``CouplingInterface.Hook`` / ``coupling_unsupported``
    registry; a hook is now "unsupported" iff it raises ``NotImplementedError``.
    Calls the unbound hook with a placeholder ``self`` so this stays a no-GPU,
    no-ovphysx API-surface check: a ``NotImplementedError`` means unsupported,
    while any other error (or no error) from the placeholder arguments means
    the hook is implemented.
    """

    class _Dummy:
        pass

    try:
        method(_Dummy(), *args, **kwargs)
    except NotImplementedError:
        return True
    except Exception:
        return False
    return False


class TestCouplingUnsupportedDeclaration(unittest.TestCase):
    """Which coupling hooks ``SolverPhysX`` supports, per the raise-to-opt-out contract."""

    def test_particle_proxy_harvest_declared_unsupported(self):
        # PhysX has no particles: the particle-harvest hook must opt out.
        self.assertTrue(_hook_raises_not_implemented(SolverPhysX.coupling_harvest_proxy_particle_forces, None, None))

    def test_body_proxy_harvest_not_declared_unsupported(self):
        self.assertFalse(_hook_raises_not_implemented(SolverPhysX.coupling_harvest_proxy_wrenches, None, None))

    def test_effective_mass_block_not_declared_unsupported(self):
        self.assertFalse(
            _hook_raises_not_implemented(SolverPhysX.coupling_eval_effective_mass_block, None, None, None, None)
        )

    def test_notify_input_state_update_not_declared_unsupported(self):
        self.assertFalse(_hook_raises_not_implemented(SolverPhysX.coupling_notify_input_state_update, None, 0))


class TestApiSurface(unittest.TestCase):
    """Method-shape contracts that the framework relies on."""

    def test_parse_usd_is_classmethod(self):
        self.assertIsInstance(
            inspect.getattr_static(SolverPhysX, "parse_usd"),
            classmethod,
        )

    def test_finalize_coupling_is_instance_method(self):
        # Should be a regular function on the class, not a classmethod
        # or staticmethod.
        attr = inspect.getattr_static(SolverPhysX, "finalize_coupling")
        self.assertNotIsInstance(attr, classmethod)
        self.assertNotIsInstance(attr, staticmethod)
        self.assertTrue(callable(attr))

    def test_coupling_notify_signature_matches_framework_contract(self):
        # The framework calls:
        #   solver.coupling_notify_input_state_update(state, flags, *, iteration_restart=..., dt=...)
        sig = inspect.signature(SolverPhysX.coupling_notify_input_state_update)
        params = list(sig.parameters.values())
        # self, state, flags, *, iteration_restart, dt
        self.assertEqual(params[0].name, "self")
        self.assertEqual(params[1].name, "state")
        self.assertEqual(params[2].name, "flags")
        self.assertEqual(params[3].name, "iteration_restart")
        self.assertEqual(params[3].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(params[4].name, "dt")
        self.assertEqual(params[4].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_effective_mass_block_signature_matches_framework_contract(self):
        sig = inspect.signature(SolverPhysX.coupling_eval_effective_mass_block)
        params = list(sig.parameters.values())
        self.assertEqual(
            [p.name for p in params],
            ["self", "endpoint_kind", "endpoint_index", "endpoint_local_pos", "out_mass", "out_inertia"],
        )
        # out_inertia has a default of None
        self.assertIsNone(params[5].default)

    def test_init_signature(self):
        sig = inspect.signature(SolverPhysX.__init__)
        params = list(sig.parameters.values())
        self.assertEqual(params[0].name, "self")
        self.assertEqual(params[1].name, "model")
        kw_names = {p.name for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY}
        self.assertEqual(kw_names, {"parse_info", "direct_gpu"})
        # parse_info is required (no default), others have defaults
        parse_info_param = next(p for p in params if p.name == "parse_info")
        self.assertIs(parse_info_param.default, inspect.Parameter.empty)


# ---------------------------------------------------------------------------
# Layer 2: Construction & parse_usd behaviour.
# Requires ovphysx; GPU-only.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_CUDA, "Requires CUDA")
class TestParseUsdSingleEnv(_PhysxIsolatedTestCase):
    """``parse_usd`` against a single-env USD."""

    def setUp(self):
        if not _OVPHYSX:
            self.skipTest("ovphysx is not installed")
        self.usd_path = _write_usda(_TINY_USDA)
        self.builder = ModelBuilder()

    def test_returns_physx_parse_info(self):
        info = SolverPhysX.parse_usd(self.builder, self.usd_path)
        self.assertIsInstance(info, PhysxParseInfo)

    def test_prim_paths_contains_authored_dynamic_prim(self):
        info = SolverPhysX.parse_usd(self.builder, self.usd_path)
        self.assertIn("/World/DynamicBox", info.prim_paths)

    def test_path_body_map_resolves_to_valid_body_index(self):
        info = SolverPhysX.parse_usd(self.builder, self.usd_path)
        idx = info.path_body_map["/World/DynamicBox"]
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, self.builder.body_count)

    def test_marks_added_shapes_as_collision_group_zero(self):
        pre_shape_count = self.builder.shape_count
        SolverPhysX.parse_usd(self.builder, self.usd_path)
        for s in range(pre_shape_count, self.builder.shape_count):
            self.assertEqual(self.builder.shape_collision_group[s], 0)

    def test_does_not_attach_ovphysx_stamp(self):
        SolverPhysX.parse_usd(self.builder, self.usd_path)
        # The prototype carried `_ovphysx_stamp` as a side effect; the
        # production API must not.
        self.assertFalse(hasattr(self.builder, "_ovphysx_stamp"))


@unittest.skipUnless(_CUDA, "Requires CUDA")
class TestConstruction(_PhysxIsolatedTestCase):
    """``SolverPhysX.__init__`` validation and parent-model handling."""

    def setUp(self):
        if not _OVPHYSX:
            self.skipTest("ovphysx is not installed")
        usd_path = _write_usda(_TINY_USDA)
        self.builder = ModelBuilder()
        self.parse_info = SolverPhysX.parse_usd(self.builder, usd_path)
        self.model = self.builder.finalize()

    def test_constructs_with_valid_parse_info(self):
        solver = SolverPhysX(self.model, parse_info=self.parse_info)
        self.assertIsNotNone(solver)

    def test_raises_when_parse_info_does_not_match_model(self):
        # Build a different model with no overlap with parse_info.prim_paths.
        other_builder = ModelBuilder()
        other_builder.add_body(xform=wp.transform_identity(), mass=1.0)
        other_model = other_builder.finalize()
        with self.assertRaises(ValueError):
            SolverPhysX(other_model, parse_info=self.parse_info)


# ---------------------------------------------------------------------------
# Layer 2 (continued): Coupling-hook conditional presence.
# Requires ovphysx; GPU-only.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Layer 2 (continued): ``step`` behaviour.
# Requires ovphysx; GPU-only.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_CUDA, "Requires CUDA")
class TestStepSingleEnv(_PhysxIsolatedTestCase):
    """``step`` against a single-env scene with one dynamic body."""

    def setUp(self):
        if not _OVPHYSX:
            self.skipTest("ovphysx is not installed")
        usd_path = _write_usda(_TINY_USDA)
        self.builder = ModelBuilder()
        self.parse_info = SolverPhysX.parse_usd(self.builder, usd_path)
        self.model = self.builder.finalize()
        self.solver = SolverPhysX(self.model, parse_info=self.parse_info)
        self.box_idx = self.parse_info.path_body_map["/World/DynamicBox"]

    def _state_pair(self):
        return self.model.state(), self.model.state()

    def test_dynamic_body_falls_under_gravity(self):
        state_in, state_out = self._state_pair()
        initial_z = float(state_in.body_q.numpy()[self.box_idx, 2])
        for _ in range(10):
            self.solver.step(state_in, state_out, None, None, 1.0 / 60.0)
            state_in, state_out = state_out, state_in
        final_z = float(state_in.body_q.numpy()[self.box_idx, 2])
        # Gravity should have pulled the body downward by at least ~10 cm
        # over 10 steps starting from rest at 60 Hz.
        self.assertLess(final_z, initial_z - 0.05)

    def test_post_step_pose_is_finite(self):
        state_in, state_out = self._state_pair()
        for _ in range(5):
            self.solver.step(state_in, state_out, None, None, 1.0 / 60.0)
            state_in, state_out = state_out, state_in
        pose = state_in.body_q.numpy()[self.box_idx]
        self.assertTrue(all(abs(v) < 1e6 for v in pose))
        self.assertFalse(any(v != v for v in pose))  # no NaN

    def test_post_step_velocity_is_downward(self):
        state_in, state_out = self._state_pair()
        for _ in range(5):
            self.solver.step(state_in, state_out, None, None, 1.0 / 60.0)
            state_in, state_out = state_out, state_in
        # body_qd is a spatial vector (v, omega) in Newton's convention —
        # linear velocity is the first three components, angular the last
        # three (see state.py).
        qd = state_in.body_qd.numpy()[self.box_idx]
        v_z = float(qd[2])
        self.assertLess(v_z, 0.0)


@unittest.skipUnless(_CUDA, "Requires CUDA")
class TestCouplingNotify(_PhysxIsolatedTestCase):
    """``coupling_notify_input_state_update`` flag-by-flag behaviour."""

    def setUp(self):
        if not _OVPHYSX:
            self.skipTest("ovphysx is not installed")
        usd_path = _write_usda(_TINY_USDA)
        self.builder = ModelBuilder()
        self.parse_info = SolverPhysX.parse_usd(self.builder, usd_path)
        self.model = self.builder.finalize()
        self.solver = SolverPhysX(self.model, parse_info=self.parse_info)
        self.box_idx = self.parse_info.path_body_map["/World/DynamicBox"]

    def _state_pair(self):
        return self.model.state(), self.model.state()

    def _apply_z_force(self, state, fz: float) -> None:
        """Write a linear +Z force on the DynamicBox. The convention is
        body_f spatial-vector indices [0..2] = linear force, [3..5] =
        torque, matching the prototype's ``_gather_body_wrench`` kernel
        (``spatial_top`` = force, ``spatial_bottom`` = torque)."""
        body_f = state.body_f.numpy()
        body_f[self.box_idx] = [0.0, 0.0, fz, 0.0, 0.0, 0.0]
        state.body_f.assign(body_f)

    def test_body_f_push_lifts_dynamic_body_when_notified(self):
        """+30 N upward, gravity 9.81 N → net +20 N → body rises."""
        state_in, state_out = self._state_pair()
        initial_z = float(state_in.body_q.numpy()[self.box_idx, 2])
        for _ in range(30):
            self._apply_z_force(state_in, 30.0)
            self.solver.coupling_notify_input_state_update(state_in, StateFlags.BODY_F, dt=1.0 / 60.0)
            self.solver.step(state_in, state_out, None, None, 1.0 / 60.0)
            state_in, state_out = state_out, state_in
        final_z = float(state_in.body_q.numpy()[self.box_idx, 2])
        self.assertGreater(final_z, initial_z + 0.5)

    def test_body_f_not_pushed_without_notify_call(self):
        """body_f set on state but notify never called → force ignored,
        body falls under gravity."""
        state_in, state_out = self._state_pair()
        initial_z = float(state_in.body_q.numpy()[self.box_idx, 2])
        for _ in range(30):
            self._apply_z_force(state_in, 30.0)
            # Deliberately skip the notify call.
            self.solver.step(state_in, state_out, None, None, 1.0 / 60.0)
            state_in, state_out = state_out, state_in
        final_z = float(state_in.body_q.numpy()[self.box_idx, 2])
        # Gravity-only behaviour: body should fall, not rise.
        self.assertLess(final_z, initial_z)

    def test_body_f_not_pushed_when_other_flag_set(self):
        """body_f set, notify called with BODY_Q (not BODY_F) → force
        ignored. Sanity check that the flag gate is per-flag, not "any
        notify call pushes everything"."""
        state_in, state_out = self._state_pair()
        initial_z = float(state_in.body_q.numpy()[self.box_idx, 2])
        for _ in range(30):
            self._apply_z_force(state_in, 30.0)
            self.solver.coupling_notify_input_state_update(state_in, StateFlags.BODY_Q, dt=1.0 / 60.0)
            self.solver.step(state_in, state_out, None, None, 1.0 / 60.0)
            state_in, state_out = state_out, state_in
        final_z = float(state_in.body_q.numpy()[self.box_idx, 2])
        self.assertLess(final_z, initial_z)

    def test_particle_flags_silently_ignored(self):
        """Particle-source coupling is unsupported; calling notify with
        particle flags must not crash. The framework may issue these
        flags in mixed-solver scenes; SolverPhysX is expected to no-op."""
        state_in, _ = self._state_pair()
        self.solver.coupling_notify_input_state_update(state_in, StateFlags.PARTICLE_Q, dt=1.0 / 60.0)
        self.solver.coupling_notify_input_state_update(state_in, StateFlags.PARTICLE_QD, dt=1.0 / 60.0)
        self.solver.coupling_notify_input_state_update(state_in, StateFlags.PARTICLE_F, dt=1.0 / 60.0)

    def test_body_qd_does_not_crash(self):
        """BODY_QD is a no-op (standalone-body velocity is pushed during
        step()). Must not raise."""
        state_in, _ = self._state_pair()
        self.solver.coupling_notify_input_state_update(state_in, StateFlags.BODY_QD, dt=1.0 / 60.0)

    def test_combined_flags_processed_independently(self):
        """``flags = BODY_Q | BODY_F`` should still process BODY_F (push
        wrench) even though BODY_Q is a no-op."""
        state_in, state_out = self._state_pair()
        initial_z = float(state_in.body_q.numpy()[self.box_idx, 2])
        combined = StateFlags.BODY_Q | StateFlags.BODY_F
        for _ in range(30):
            self._apply_z_force(state_in, 30.0)
            self.solver.coupling_notify_input_state_update(state_in, combined, dt=1.0 / 60.0)
            self.solver.step(state_in, state_out, None, None, 1.0 / 60.0)
            state_in, state_out = state_out, state_in
        final_z = float(state_in.body_q.numpy()[self.box_idx, 2])
        # BODY_F still gets processed when combined with BODY_Q.
        self.assertGreater(final_z, initial_z + 0.5)


# ---------------------------------------------------------------------------
# Layer 2 (continued): ``coupling_eval_effective_mass_block`` (OSI hook).
# Requires ovphysx; GPU-only.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_CUDA, "Requires CUDA")
class TestCouplingEffectiveMassBlock(_PhysxIsolatedTestCase):
    """``coupling_eval_effective_mass_block`` against a standalone body.

    The standalone-body code path is the simpler half of the OSI hook:
    no articulation, no Jacobian, no mass matrix — the docstring says
    ``out_mass`` / ``out_inertia`` should fall back to the intrinsic
    ``body_mass`` / ``body_inertia`` from the model.

    Articulation-OSI tests (which require an actual articulated robot
    USD) are deferred to the Layer-3 batch.
    """

    def setUp(self):
        if not _OVPHYSX:
            self.skipTest("ovphysx is not installed")
        usd_path = _write_usda(_TINY_USDA)
        self.builder = ModelBuilder()
        self.parse_info = SolverPhysX.parse_usd(self.builder, usd_path)
        self.model = self.builder.finalize()
        self.solver = SolverPhysX(self.model, parse_info=self.parse_info)
        self.box_idx = self.parse_info.path_body_map["/World/DynamicBox"]

    def _one_endpoint(self, body_idx: int, local_pos=(0.0, 0.0, 0.0)):
        """Build the per-endpoint argument arrays for a single body."""
        kinds = wp.array([int(CouplingInterface.EndpointKind.BODY)], dtype=wp.int32)
        indices = wp.array([body_idx], dtype=wp.int32)
        positions = wp.array([list(local_pos)], dtype=wp.float32)
        out_mass = wp.zeros(1, dtype=wp.float32)
        out_inertia = wp.zeros(1, dtype=wp.mat33)
        return kinds, indices, positions, out_mass, out_inertia

    def test_standalone_body_returns_intrinsic_mass(self):
        kinds, indices, positions, out_mass, _ = self._one_endpoint(self.box_idx)
        self.solver.coupling_eval_effective_mass_block(kinds, indices, positions, out_mass)
        expected = float(self.model.body_mass.numpy()[self.box_idx])
        self.assertAlmostEqual(float(out_mass.numpy()[0]), expected, places=5)

    def test_standalone_body_returns_intrinsic_inertia_when_requested(self):
        kinds, indices, positions, out_mass, out_inertia = self._one_endpoint(self.box_idx)
        self.solver.coupling_eval_effective_mass_block(kinds, indices, positions, out_mass, out_inertia)
        expected = self.model.body_inertia.numpy()[self.box_idx]
        result = out_inertia.numpy()[0]
        # mat33 comparison element-wise to a small tolerance.
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(float(result[i, j]), float(expected[i, j]), places=4)

    def test_non_zero_local_pos_raises_not_implemented(self):
        kinds, indices, positions, out_mass, _ = self._one_endpoint(self.box_idx, local_pos=(0.1, 0.0, 0.0))
        with self.assertRaises(NotImplementedError):
            self.solver.coupling_eval_effective_mass_block(kinds, indices, positions, out_mass)

    def test_particle_endpoint_kind_is_skipped(self):
        """The hook iterates endpoints by kind; non-BODY entries should
        be left untouched in the output. For a single PARTICLE endpoint,
        ``out_mass`` should remain at its initialized value (zero)."""
        kinds = wp.array([int(CouplingInterface.EndpointKind.PARTICLE)], dtype=wp.int32)
        indices = wp.array([0], dtype=wp.int32)
        positions = wp.array([[0.0, 0.0, 0.0]], dtype=wp.float32)
        out_mass = wp.zeros(1, dtype=wp.float32)
        self.solver.coupling_eval_effective_mass_block(kinds, indices, positions, out_mass)
        self.assertEqual(float(out_mass.numpy()[0]), 0.0)


# ---------------------------------------------------------------------------
# Layer 3: end-to-end coupling.
# Requires ovphysx + mjwarp; GPU-only.
#
# These are smoke-style tests: they exercise the full SolverCoupledProxy
# pipeline with SolverPhysX as the destination, but the assertions are
# strictly structural (no exceptions, finite state, hooks activated as
# expected). Empirical "settled cleanly" behaviour is covered by visual
# examples, not by gated tests.
# ---------------------------------------------------------------------------


_MJWARP_OVPHYSX_USDA = """#usda 1.0
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
    def Cube "Floor" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {
        double size = 1
        float3 xformOp:scale = (10.0, 10.0, 0.05)
        double3 xformOp:translate = (0, 0, -0.025)
        quatf xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }
    def Xform "Mirrors"
    {
        def Cube "ball_Mirror" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI", "PhysxContactReportAPI"])
        {
            double size = 1
            float3 xformOp:scale = (0.1, 0.1, 0.1)
            double3 xformOp:translate = (0, 0, 5.0)
            quatf xformOp:orient = (1, 0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
            float physics:mass = 1.0
        }
    }
}
"""


_BALL_MJCF = """<mujoco>
  <worldbody>
    <body name="ball" pos="0 0 5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Articulated source: a floating base carrying a hinged child. The base link's
# articulated (operational-space) effective mass exceeds its 1 kg intrinsic
# mass because moving the base must also accelerate the child. Used to check
# that the coupling framework's effective-mass override reaches ovphysx.
_CHAIN_MJCF = """<mujoco>
  <worldbody>
    <body name="base" pos="0 0 5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body name="child" pos="0.4 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.1" mass="3"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _mujoco_available() -> bool:
    try:
        from newton.solvers import SolverMuJoCo  # noqa: F401, PLC0415

        return True
    except Exception:
        return False


_MUJOCO = _mujoco_available()


@unittest.skipUnless(_CUDA, "Requires CUDA")
class TestLayer3MjwarpOvphysxCoupling(_PhysxIsolatedTestCase):
    """End-to-end smoke test of mjwarp → ovphysx coupling.

    Builds a tiny synthetic scene: ovphysx owns a floor (static
    collider) and a dynamic ball mirror; mjwarp owns one free
    dynamic ball authored via MJCF. ``SolverCoupledProxy`` wires the
    mjwarp ball as the source of the ovphysx-side ball mirror. The
    expected destination-mode pipeline runs through every coupling
    hook on the SolverPhysX side.

    Assertions are structural only (no NaN / exceptions / harvest hook
    activated). Physical correctness (does the ball settle on the
    floor?) is out of scope for this test layer — see the
    ``mujoco_physx_coupled_solver`` example for visual validation.
    """

    def setUp(self):
        if not _OVPHYSX:
            self.skipTest("ovphysx is not installed")
        if not _MUJOCO:
            self.skipTest("mujoco / SolverMuJoCo is not installed")

        # Author the ovphysx USD with floor + ball mirror.
        usd_path = _write_usda(_MJWARP_OVPHYSX_USDA)

        # Author the mjwarp MJCF inline (temp file because add_mjcf
        # takes a file path).
        mjcf_dir = tempfile.mkdtemp(prefix="solver_physx_mjcf_")
        mjcf_path = os.path.join(mjcf_dir, "ball.mjcf")
        with open(mjcf_path, "w") as f:
            f.write(_BALL_MJCF)

        # Build the model: ovphysx-owned floor + mirror via parse_usd,
        # mjwarp-owned ball via add_mjcf into the same builder.
        builder = ModelBuilder()
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        SolverMuJoCo.register_custom_attributes(builder)

        self.parse_info = SolverPhysX.parse_usd(builder, usd_path)
        pre_mjcf_body_count = builder.body_count
        builder.add_mjcf(mjcf_path)
        self.mjwarp_body_indices = list(range(pre_mjcf_body_count, builder.body_count))
        self.assertEqual(len(self.mjwarp_body_indices), 1, "MJCF should add one body")

        self.mirror_body_idx = self.parse_info.path_body_map["/World/Mirrors/ball_Mirror"]
        self.source_body_idx = self.mjwarp_body_indices[0]
        self.ovphysx_body_indices = list(self.parse_info.path_body_map.values())

        self.model = builder.finalize()

        # Build the coupling framework.
        from newton.solvers.experimental.coupled import SolverCoupledProxy  # noqa: PLC0415

        proxy = SolverCoupledProxy.Proxy(
            source="mjwarp",
            destination="ovphysx",
            bodies=[self.source_body_idx],
            proxy_bodies=[self.mirror_body_idx],
            mass_scale=1.0,
            mode="lagged",
            # Mirror is an ovphysx-owned body driven kinematically by the
            # mjwarp source, so the proxy target coincides with the
            # destination's owned set.
            destination_owned=True,
        )
        config = SolverCoupledProxy.Config(proxies=[proxy], iterations=1)

        self.coupled_solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mjwarp",
                    solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=16),
                    bodies=self.mjwarp_body_indices,
                    joints=list(range(self.model.joint_count)),
                ),
                SolverCoupledProxy.Entry(
                    name="ovphysx",
                    solver=lambda v: SolverPhysX(v, parse_info=self.parse_info),
                    bodies=self.ovphysx_body_indices,
                ),
            ],
            coupling=config,
        )

        # finalize_coupling pushes mass / inertia / COM onto the
        # ovphysx-side mirror.
        self.ovphysx_solver = self.coupled_solver._entries["ovphysx"].solver
        self.ovphysx_solver.finalize_coupling(config.proxies)

    def test_one_step_runs_without_exception(self):
        state_in = self.model.state()
        state_out = self.model.state()
        contacts = self.model.contacts()
        control = self.model.control()
        # Should run cleanly with no exceptions raised by either solver
        # nor by the framework's harvest / notify dispatch.
        self.coupled_solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)

    def test_short_run_state_remains_finite(self):
        state_in = self.model.state()
        state_out = self.model.state()
        contacts = self.model.contacts()
        control = self.model.control()
        for _ in range(20):
            self.coupled_solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
            state_in, state_out = state_out, state_in
        body_q = state_in.body_q.numpy()
        body_qd = state_in.body_qd.numpy()
        # No NaN, no inf.
        self.assertTrue((body_q == body_q).all(), "body_q contains NaN")
        self.assertTrue((body_qd == body_qd).all(), "body_qd contains NaN")
        # Positions within a generous sanity bound — no runaway explosion.
        self.assertTrue((abs(body_q) < 1e3).all(), "body_q out of bounds")


@unittest.skipUnless(_CUDA, "Requires CUDA")
class TestLayer3EffectiveMassReachesOvphysx(_PhysxIsolatedTestCase):
    """Regression: the coupling framework's effective-mass override must reach
    the ovphysx destination.

    An articulated mjwarp source (floating base + hinged child) has a base-link
    effective mass that differs from its intrinsic mass. The framework installs
    that effective mass on the destination view and calls ``notify_model_changed``;
    :meth:`SolverPhysX.finalize_coupling` and
    :meth:`SolverPhysX.notify_model_changed` must forward it onto the ovphysx
    mirror. Without this, the pose-driven mirror runs at the wrong impedance and
    the explicit coupling loop diverges under hard contact (the mirror instead
    keeps its raw intrinsic mass).
    """

    def setUp(self):
        if not _OVPHYSX:
            self.skipTest("ovphysx is not installed")
        if not _MUJOCO:
            self.skipTest("mujoco / SolverMuJoCo is not installed")

        usd_path = _write_usda(_MJWARP_OVPHYSX_USDA)
        mjcf_dir = tempfile.mkdtemp(prefix="solver_physx_chain_")
        mjcf_path = os.path.join(mjcf_dir, "chain.mjcf")
        with open(mjcf_path, "w") as f:
            f.write(_CHAIN_MJCF)

        builder = ModelBuilder()
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415

        SolverMuJoCo.register_custom_attributes(builder)
        self.parse_info = SolverPhysX.parse_usd(builder, usd_path)
        pre = builder.body_count
        builder.add_mjcf(mjcf_path)
        self.mjwarp_body_indices = list(range(pre, builder.body_count))
        self.base_body_idx = self.mjwarp_body_indices[0]  # floating base link
        self.mirror_body_idx = self.parse_info.path_body_map["/World/Mirrors/ball_Mirror"]
        self.ovphysx_body_indices = list(self.parse_info.path_body_map.values())
        self.model = builder.finalize()
        self.raw_base_mass = float(self.model.body_mass.numpy()[self.base_body_idx])

        from newton.solvers.experimental.coupled import SolverCoupledProxy  # noqa: PLC0415

        proxy = SolverCoupledProxy.Proxy(
            source="mjwarp",
            destination="ovphysx",
            bodies=[self.base_body_idx],
            proxy_bodies=[self.mirror_body_idx],
            mass_scale=1.0,
            mode="lagged",
            destination_owned=True,
        )
        self.config = SolverCoupledProxy.Config(proxies=[proxy], iterations=1)
        self.coupled_solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mjwarp",
                    solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=16),
                    bodies=self.mjwarp_body_indices,
                    joints=list(range(self.model.joint_count)),
                ),
                SolverCoupledProxy.Entry(
                    name="ovphysx",
                    solver=lambda v: SolverPhysX(v, parse_info=self.parse_info),
                    bodies=self.ovphysx_body_indices,
                ),
            ],
            coupling=self.config,
        )
        self.ovphysx_solver = self.coupled_solver._entries["ovphysx"].solver

        # The framework installs the effective mass on the destination view
        # during construction; capture it and the mirror's view-local index.
        view = self.ovphysx_solver.model
        self.mirror_local = int(view.coupled_index_maps.body_global_to_local.numpy()[self.mirror_body_idx])
        self.effective_mass = float(view.body_mass.numpy()[self.mirror_local])

    def _read_ovphysx_mirror_mass(self) -> float:
        from ovphysx.types import TensorType  # noqa: PLC0415

        buf = wp.zeros(1, dtype=wp.float32, device=self.ovphysx_solver.device)
        with self.ovphysx_solver._physx.create_tensor_binding(
            prim_paths=["/World/Mirrors/ball_Mirror"],
            tensor_type=TensorType.RIGID_BODY_MASS,
            raise_if_empty=True,
        ) as b:
            b.read(buf)
        return float(buf.numpy()[0])

    def test_finalize_pushes_effective_mass_to_ovphysx(self):
        # Sanity: articulation makes the effective mass differ from intrinsic;
        # otherwise the test could not distinguish the two code paths.
        self.assertGreater(
            abs(self.effective_mass - self.raw_base_mass),
            1e-3,
            "articulated effective mass should differ from intrinsic base mass",
        )
        self.ovphysx_solver.finalize_coupling(self.config.proxies)
        mirror_mass = self._read_ovphysx_mirror_mass()
        # ovphysx must carry the effective mass, not the raw intrinsic mass.
        self.assertAlmostEqual(mirror_mass, self.effective_mass, places=3)
        self.assertNotAlmostEqual(mirror_mass, self.raw_base_mass, places=3)

    def test_notify_model_changed_forwards_new_inertia(self):
        from newton import ModelFlags  # noqa: PLC0415

        self.ovphysx_solver.finalize_coupling(self.config.proxies)
        # Mimic the framework installing a fresh effective mass on the view and
        # notifying the solver: the update must propagate to ovphysx.
        view = self.ovphysx_solver.model
        new_mass = 7.5
        indices = wp.array([self.mirror_local], dtype=wp.int32, device=view.device)
        masses = wp.array([new_mass], dtype=wp.float32, device=view.device)
        inertias = wp.array(
            [wp.mat33(0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05)],
            dtype=wp.mat33,
            device=view.device,
        )
        view.set_body_inertial_properties(indices, masses, inertias)
        self.ovphysx_solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
        self.assertAlmostEqual(self._read_ovphysx_mirror_mass(), new_mass, places=3)


_BOX_FOR_MPM_USDA = """#usda 1.0
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
    def Cube "Box" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"])
    {
        double size = 1
        float3 xformOp:scale = (0.3, 0.3, 0.3)
        double3 xformOp:translate = (0, 0, 1.5)
        quatf xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        float physics:mass = 1.0
    }
}
"""


def _mpm_available() -> bool:
    try:
        from newton.solvers import SolverImplicitMPM  # noqa: F401, PLC0415

        return True
    except Exception:
        return False


def _vbd_available() -> bool:
    try:
        from newton.solvers import SolverVBD  # noqa: F401, PLC0415

        return True
    except Exception:
        return False


_MPM = _mpm_available()
_VBD = _vbd_available()


@unittest.skipUnless(_CUDA, "Requires CUDA")
class TestLayer3OvphysxMpmCoupling(_PhysxIsolatedTestCase):
    """End-to-end smoke test of ovphysx → MPM coupling (source mode).

    ovphysx owns the rigid box; SolverImplicitMPM owns a small sand
    bed. The framework's shared-body proxy mechanism propagates the
    box pose into MPM each step; MPM's default momentum-difference
    harvest feeds reaction force back into the box's ``body_f``.

    SolverPhysX-side surface exercised:

    - :meth:`step` advances the PhysX scene with the box.
    - :meth:`coupling_notify_input_state_update` is called with
      BODY_F by the framework to inject MPM's harvested force.

    Both are already covered deterministically by Layer-2 tests; this
    test verifies the full pipeline composes without crashing.
    """

    def setUp(self):
        if not _OVPHYSX:
            self.skipTest("ovphysx is not installed")
        if not _MPM:
            self.skipTest("SolverImplicitMPM is not installed")

        from newton.solvers import SolverImplicitMPM  # noqa: PLC0415
        from newton.solvers.experimental.coupled import SolverCoupledProxy  # noqa: PLC0415

        usd_path = _write_usda(_BOX_FOR_MPM_USDA)
        builder = ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        self.parse_info = SolverPhysX.parse_usd(builder, usd_path)
        self.box_idx = self.parse_info.path_body_map["/World/Box"]

        # Tiny sand bed (a 3x3x2 grid of particles) — enough to
        # exercise body↔particle interaction without large-scene
        # overhead.
        builder.add_particle_grid(
            pos=wp.vec3(-0.2, -0.2, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=3,
            dim_y=3,
            dim_z=2,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            mass=0.1,
            jitter=0.0,
            radius_mean=0.04,
        )

        self.model = builder.finalize()

        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = 0.1
        mpm_config.grid_type = "fixed"
        mpm_config.grid_padding = 10
        mpm_config.max_active_cell_count = 1 << 12
        mpm_config.strain_basis = "P0"
        mpm_config.max_iterations = 10

        self.coupled_solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="ovphysx",
                    solver=lambda v: SolverPhysX(v, parse_info=self.parse_info),
                    bodies=[self.box_idx],
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
                        source="ovphysx",
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

    def test_one_step_runs_without_exception(self):
        state_in = self.model.state()
        state_out = self.model.state()
        contacts = self.model.contacts()
        control = self.model.control()
        self.coupled_solver.step(state_in, state_out, control, contacts, 1.0 / 200.0)

    def test_short_run_state_remains_finite(self):
        state_in = self.model.state()
        state_out = self.model.state()
        contacts = self.model.contacts()
        control = self.model.control()
        for _ in range(10):
            self.coupled_solver.step(state_in, state_out, control, contacts, 1.0 / 200.0)
            state_in, state_out = state_out, state_in
        body_q = state_in.body_q.numpy()
        self.assertTrue((body_q == body_q).all(), "body_q contains NaN")


@unittest.skipUnless(_CUDA, "Requires CUDA")
class TestLayer3OvphysxVbdCoupling(_PhysxIsolatedTestCase):
    """End-to-end smoke test of ovphysx → VBD coupling (source mode).

    Mirrors the MPM test but with a small VBD cloth patch underneath
    the box. Same SolverPhysX-side coverage profile (already covered
    by Layer-2 tests); this test composes the full pipeline.
    """

    def setUp(self):
        if not _OVPHYSX:
            self.skipTest("ovphysx is not installed")
        if not _VBD:
            self.skipTest("SolverVBD is not installed")

        from newton.solvers import SolverVBD  # noqa: PLC0415
        from newton.solvers.experimental.coupled import SolverCoupledProxy  # noqa: PLC0415

        usd_path = _write_usda(_BOX_FOR_MPM_USDA)
        builder = ModelBuilder()
        self.parse_info = SolverPhysX.parse_usd(builder, usd_path)
        self.box_idx = self.parse_info.path_body_map["/World/Box"]

        # Tiny cloth patch: 4x4 vertices in the X-Y plane at z=0.
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=3,
            dim_y=3,
            cell_x=0.25,
            cell_y=0.25,
            mass=0.05,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=10.0,
        )
        cloth_particle_ids = list(range(builder.particle_count))
        builder.color()

        self.model = builder.finalize()

        self.coupled_solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="ovphysx",
                    solver=lambda v: SolverPhysX(v, parse_info=self.parse_info),
                    bodies=[self.box_idx],
                ),
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(model=v, iterations=4),
                    particles=cloth_particle_ids,
                    in_place=True,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="ovphysx",
                        destination="vbd",
                        bodies=[self.box_idx],
                        mass_scale=1.0,
                        mode="lagged",
                        collision_pipeline=lambda _model: None,
                    ),
                ],
                iterations=1,
            ),
        )

    def test_one_step_runs_without_exception(self):
        state_in = self.model.state()
        state_out = self.model.state()
        contacts = self.model.contacts()
        control = self.model.control()
        self.coupled_solver.step(state_in, state_out, control, contacts, 1.0 / 200.0)

    def test_short_run_state_remains_finite(self):
        state_in = self.model.state()
        state_out = self.model.state()
        contacts = self.model.contacts()
        control = self.model.control()
        for _ in range(10):
            self.coupled_solver.step(state_in, state_out, control, contacts, 1.0 / 200.0)
            state_in, state_out = state_out, state_in
        body_q = state_in.body_q.numpy()
        self.assertTrue((body_q == body_q).all(), "body_q contains NaN")
