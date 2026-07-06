# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverPhysX — PhysX-backed Newton solver."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton import BodyFlags, ModelFlags, StateFlags
from newton._src.solvers.coupled.interface import CouplingInterface
from newton._src.solvers.solver import SolverBase

if TYPE_CHECKING:
    from newton.solvers.experimental.coupled import SolverCoupledProxy

    from newton import Contacts, Control, Model, ModelBuilder, State


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _diagonalize_body_inertia(
    inertia_3x3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a 3x3 symmetric inertia tensor into ``(principal_moments,
    principal_axes_quat_xyzw)``.

    PhysX stores only diagonal mass-space inertia; the principal-axes
    rotation is supplied via the COM-pose orientation. Pushing a non-
    diagonal matrix with an identity COM quaternion silently drops the
    rotation and the simulated rotational dynamics are wrong.
    """
    inertia = np.asarray(inertia_3x3, dtype=np.float64).reshape(3, 3)
    eigvals, eigvecs = np.linalg.eigh(inertia)
    # eigh returns orthonormal eigenvectors but may form a left-handed
    # frame; flip one axis so the principal-axes matrix is a proper
    # rotation before converting it to a quaternion.
    if np.linalg.det(eigvecs) < 0.0:
        eigvecs[:, 0] = -eigvecs[:, 0]
    quat = np.array(wp.quat_from_matrix(wp.mat33(eigvecs.astype(np.float32).flatten())), dtype=np.float32)
    return eigvals.astype(np.float32), quat


@wp.kernel
def _gather_body_q(
    body_q: wp.array[wp.transform],
    indices: wp.array[wp.int32],
    out: wp.array[wp.transform],
):
    """Gather the selected PhysX bodies' poses from Newton's global ``body_q``
    into the dense and binding-ordered ``out`` buffer that is then written to PhysX.
    """
    tid = wp.tid()
    out[tid] = body_q[indices[tid]]


@wp.kernel
def _scatter_body_q(
    src: wp.array[wp.transform],
    indices: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
):
    """Scatter the selected PhysX bodies' poses from the dense and binding-ordered
    ``src`` buffer (read back from PhysX) into Newton's global ``body_q``.
    """
    tid = wp.tid()
    body_q[indices[tid]] = src[tid]


@wp.kernel
def _gather_body_qd(
    body_qd: wp.array[wp.spatial_vector],
    indices: wp.array[wp.int32],
    out: wp.array[wp.spatial_vector],
):
    """Gather the selected PhysX bodies' spatial velocities from Newton's global
    ``body_qd`` into the dense and binding-ordered ``out`` buffer that is then
    written to PhysX.
    """
    tid = wp.tid()
    out[tid] = body_qd[indices[tid]]


@wp.kernel
def _scatter_body_qd(
    src: wp.array[wp.spatial_vector],
    indices: wp.array[wp.int32],
    body_qd: wp.array[wp.spatial_vector],
):
    """Scatter the selected PhysX bodies' spatial velocities from the dense and
    binding-ordered ``src`` buffer (read back from PhysX) into Newton's global
    ``body_qd``.
    """
    tid = wp.tid()
    body_qd[indices[tid]] = src[tid]


@wp.kernel
def _scatter_articulation_link_q(
    src: wp.array2d[wp.transform],
    indices: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
):
    """Scatter one articulation's link poses from the binding-ordered ``src``
    buffer (read back from PhysX) into Newton's global ``body_q``.
    Links with no Newton body (index ``-1``) are skipped.
    """
    link = wp.tid()
    newton_idx = indices[link]
    if newton_idx >= 0:
        body_q[newton_idx] = src[0, link]


@wp.kernel
def _scatter_articulation_link_qd(
    src: wp.array2d[wp.spatial_vector],
    indices: wp.array[wp.int32],
    body_qd: wp.array[wp.spatial_vector],
):
    """Scatter one articulation's link spatial velocities from the binding-ordered
    ``src`` buffer (read back from PhysX) into Newton's global ``body_qd``.
    Links with no Newton body (index ``-1``) are skipped.
    """
    link = wp.tid()
    newton_idx = indices[link]
    if newton_idx >= 0:
        body_qd[newton_idx] = src[0, link]


@wp.kernel
def _gather_articulation_dof(
    src: wp.array[float],
    indices: wp.array[wp.int32],
    out: wp.array2d[float],
):
    """Gather one articulation's per-DOF values (drive targets or gains) from a
    Newton-DOF-indexed ``src`` array into the dense, binding-ordered ``out``
    buffer that is then written to PhysX.
    """
    tid = wp.tid()
    out[0, tid] = src[indices[tid]]


@wp.kernel
def _gather_articulation_link_wrench(
    body_f: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    out: wp.array3d[float],
):
    """Gather one articulation's link wrenches from Newton's global ``body_f`` into
    the dense and binding-ordered ``out`` buffer that is then written to PhysX
    (ovphysx's ``ARTICULATION_LINK_WRENCH`` layout ``[fx,fy,fz, tx,ty,tz, px,py,pz]``).
    Links with no Newton body (index ``-1``) are written as a zero wrench.

    Convention: Newton ``body_f`` packs the linear force first (``wp.spatial_top``)
    and the torque second (``wp.spatial_bottom``), both in world frame. The PhysX
    application point ``p`` is world-frame and is set to the link's world-frame COM
    to match Newton's convention.
    """
    link = wp.tid()
    newton_idx = indices[link]
    if newton_idx < 0:
        for c in range(9):
            out[0, link, c] = 0.0
        return
    spv = body_f[newton_idx]
    force = wp.spatial_top(spv)
    torque = wp.spatial_bottom(spv)
    pos = wp.transform_point(body_q[newton_idx], body_com[newton_idx])
    for c in range(3):
        out[0, link, c] = force[c]
        out[0, link, c + 3] = torque[c]
        out[0, link, c + 6] = pos[c]


@wp.kernel
def _gather_body_wrench(
    body_f: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    out: wp.array2d[wp.float32],
):
    """Gather the selected PhysX bodies' external wrenches from Newton's global
    ``body_f`` into the dense and binding-ordered ``out`` buffer that is then
    written to PhysX (ovphysx's ``RIGID_BODY_WRENCH`` layout
    ``[fx,fy,fz, tx,ty,tz, px,py,pz]``).

    Convention: Newton ``body_f`` packs the linear force first (``wp.spatial_top``)
    and the torque second (``wp.spatial_bottom``), both in world frame. The PhysX
    application point ``p`` is world-frame and is set to the body's world-frame COM
    to match Newton's convention.
    """
    tid = wp.tid()
    newton_idx = indices[tid]
    spv = body_f[newton_idx]
    force = wp.spatial_top(spv)
    torque = wp.spatial_bottom(spv)
    pos = wp.transform_point(body_q[newton_idx], body_com[newton_idx])
    for c in range(3):
        out[tid, c] = force[c]
        out[tid, c + 3] = torque[c]
        out[tid, c + 6] = pos[c]


@wp.kernel
def _gather_intrinsic_inertia(
    endpoint_ids: wp.array[wp.int32],
    body_ids: wp.array[wp.int32],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    out_mass: wp.array[float],
    out_inertia: wp.array[wp.mat33],
):
    """Write a body's intrinsic mass / inertia to its endpoint slot, for
    standalone (non-articulation) endpoints and articulation fallbacks.
    """
    k = wp.tid()
    bid = body_ids[k]
    if bid >= 0 and bid < body_mass.shape[0]:
        ep = endpoint_ids[k]
        out_mass[ep] = body_mass[bid]
        out_inertia[ep] = body_inertia[bid]


@wp.kernel(enable_backward=False)
def _harvest_proxy_wrenches_kernel(
    dt: float,
    body_local_to_proxy_global: wp.array[int],
    qd_before: wp.array[wp.spatial_vector],
    qd_after: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    body_gravity_acceleration: wp.array[wp.vec3],
    out_coupling_forces: wp.array[wp.spatial_vector],
):
    """Estimate proxy feedback from the PhysX-side momentum change, subtracting
    gravity and external force inputs so the result is the contact reaction.

    Writes the absolute reaction by assignment (not accumulation): PhysX proxy
    mappings are 1:1 (each source body drives exactly one mirror), so each
    global id is written once. This makes ``coupling_forces`` the fresh
    per-step reaction, matching the velocity-rewind lagged scheme; the
    framework's accumulating ``atomic_add`` default is for Newton-native
    destinations driven by the gravity-compensated rewind, which PhysX cannot
    honor (see :meth:`SolverPhysX.coupling_harvest_proxy_wrenches`).
    """
    local_id = wp.tid()
    global_id = body_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    dv = wp.spatial_top(qd_after[local_id]) - wp.spatial_top(qd_before[local_id])
    dw = wp.spatial_bottom(qd_after[local_id]) - wp.spatial_bottom(qd_before[local_id])

    m = body_mass[local_id]
    I_body = body_inertia[local_id]
    r = wp.transform_get_rotation(body_q[local_id])
    g = body_gravity_acceleration[local_id]

    f_ext = body_f[local_id]

    f = m * dv / dt - m * g - wp.spatial_top(f_ext)
    tau = wp.quat_rotate(r, I_body * wp.quat_rotate_inv(r, dw)) / dt - wp.spatial_bottom(f_ext)

    out_coupling_forces[global_id] = wp.spatial_vector(f, tau)


@functools.cache
def _make_effective_mass_kernel(dof: int):
    """Make a kernel computing an articulation link's operational-space inertia.

    One tile row per ``(endpoint, link)``: from the link Jacobian ``J`` and mass
    matrix ``M`` it inverts the translational and rotational 3x3 blocks of
    ``J·M⁻¹·Jᵀ`` into ``Λ_v`` and ``Λ_ω``, then writes at the endpoint slot:

    - ``out_mass`` receives ``trace(Λ_v) / 3``, the scalar isotropic effective mass.
    - ``out_inertia`` receives ``Λ_ω`` rotated into the link's body frame.

    A non-finite or non-positive mass (e.g. a rank-deficient Jacobian) falls back
    to the body's intrinsic mass / inertia.
    """
    DOF = wp.constant(dof)

    @wp.kernel(enable_backward=False)
    def effective_mass_kernel(
        mass_matrix: wp.array2d[float],
        jacobian: wp.array2d[float],
        link_pose: wp.array2d[wp.transform],
        link_ids: wp.array[wp.int32],
        endpoint_ids: wp.array[wp.int32],
        body_ids: wp.array[wp.int32],
        body_mass: wp.array[float],
        body_inertia: wp.array[wp.mat33],
        out_mass: wp.array[float],
        out_inertia: wp.array[wp.mat33],
    ):
        k = wp.tid()
        link = link_ids[k]
        ep = endpoint_ids[k]

        chol = wp.tile_cholesky(wp.tile_load(mass_matrix, shape=(DOF, DOF)))
        # Jᵀ columns are the link Jacobian's rows; forward-solve each so that
        # Y = L⁻¹·Jᵀ and YᵀY = J·M⁻¹·Jᵀ.
        jac_t = wp.tile_transpose(wp.tile_load(jacobian, shape=(6, DOF), offset=(6 * link, 0)))
        y = wp.tile_zeros(shape=(DOF, 6), dtype=float)
        for c in range(6):
            col = wp.tile_zeros(shape=(DOF,), dtype=float)
            for r in range(DOF):
                col[r] = jac_t[r, c]
            solved = wp.tile_lower_solve(chol, col)
            for r in range(DOF):
                y[r, c] = solved[r]
        a = wp.tile_zeros(shape=(6, 6), dtype=float)
        wp.tile_matmul(wp.tile_transpose(y), y, a)

        lam_v = wp.inverse(wp.mat33(a[0, 0], a[0, 1], a[0, 2], a[1, 0], a[1, 1], a[1, 2], a[2, 0], a[2, 1], a[2, 2]))
        mass = (lam_v[0, 0] + lam_v[1, 1] + lam_v[2, 2]) / 3.0
        if mass > 0.0 and mass < 1.0e30:
            out_mass[ep] = mass
            lam_w = wp.inverse(
                wp.mat33(a[3, 3], a[3, 4], a[3, 5], a[4, 3], a[4, 4], a[4, 5], a[5, 3], a[5, 4], a[5, 5])
            )
            rot = wp.quat_to_matrix(wp.transform_get_rotation(link_pose[0, link]))
            out_inertia[ep] = wp.transpose(rot) * lam_w * rot
        else:
            bid = body_ids[k]
            out_mass[ep] = body_mass[bid]
            out_inertia[ep] = body_inertia[bid]

    return effective_mass_kernel


# ---------------------------------------------------------------------------
# Articulation-link / DOF index resolution.
# ---------------------------------------------------------------------------


def _resolve_link_indices(
    root_path: str,
    link_names: Sequence[str],
    body_labels: Sequence[str],
    scope_prefix: str,
) -> list[int]:
    """Resolve each PhysX link to its Newton body index by matching the
    ``body_label`` that ends in ``/<link_name>`` within ``scope_prefix``
    (``-1`` if unmatched, raises if ambiguous).

    Assumes labels are the USD prim paths and link names are their trailing
    segments, and that link names are unique within ``scope_prefix`` (which must
    bound this one articulation); link order is never relied on.
    """
    link_indices: list[int] = []
    for name in link_names:
        suffix = "/" + name
        matches = [
            i for i, lbl in enumerate(body_labels) if lbl and lbl.startswith(scope_prefix) and lbl.endswith(suffix)
        ]
        if len(matches) == 1:
            link_indices.append(matches[0])
        elif len(matches) == 0:
            link_indices.append(-1)
        else:
            raise ValueError(
                f"Articulation link {name!r} under {root_path!r} resolves "
                f"to multiple Newton bodies: "
                f"{[body_labels[i] for i in matches]}. Make link names "
                "unique within the scope."
            )
    return link_indices


def _resolve_dof_indices(
    root_path: str,
    dof_names: Sequence[str],
    joint_labels: Sequence[str],
    joint_qd_start: np.ndarray,
    scope_prefix: str,
) -> list[int]:
    """Resolve each PhysX DOF to its Newton ``joint_qd`` index by matching the
    *joint* label that ends in ``/<dof_name>`` within ``scope_prefix`` and
    taking its ``joint_qd_start`` (``-1`` if unmatched, raises if ambiguous).

    Assumes joint labels are the USD prim paths with the DOF name as their
    trailing segment, names are unique within ``scope_prefix``, and there is one
    DOF per joint — it returns the joint's first DOF, so only single-axis joints
    (revolute / prismatic, where the DOF name equals the joint name) resolve;
    multi-axis joints (e.g. D6) won't match.
    """
    dof_indices: list[int] = []
    for name in dof_names:
        suffix = "/" + name
        matches = [
            j for j, lbl in enumerate(joint_labels) if lbl and lbl.startswith(scope_prefix) and lbl.endswith(suffix)
        ]
        if len(matches) == 1:
            dof_indices.append(int(joint_qd_start[matches[0]]))
        elif len(matches) == 0:
            dof_indices.append(-1)
        else:
            raise ValueError(
                f"Articulation DOF {name!r} under {root_path!r} resolves "
                f"to multiple Newton joints: "
                f"{[joint_labels[j] for j in matches]}."
            )
    return dof_indices


# ---------------------------------------------------------------------------
# SolverPhysX
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysxParseInfo:
    """Data produced by :meth:`SolverPhysX.parse_usd` and consumed by
    :meth:`SolverPhysX.__init__`.

    The constructor needs to know things that only the parse
    calls know: which USD file was loaded and which Newton bodies came
    from it. ``PhysxParseInfo`` is the explicit handoff between the two
    calls.

    Attributes:
        usd_path: Path to the USD file that was loaded.
        prim_paths: USD prim paths whose corresponding Newton bodies
            are owned by ``SolverPhysX``. ``__init__`` filters its
            auto-classification (dynamic / articulation link) against
            this set so other solvers' bodies in the same model are
            left alone.
        path_body_map: ``{prim_path: newton_body_index}`` from Newton's
            underlying ``parse_usd``. Exposed for callers that need to
            look up specific prims (e.g. to wire proxy mappings).
        articulation_roots: Prim paths carrying ``PhysicsArticulationRootAPI``
            in the parsed (per-env) scene — i.e. every articulation PhysX
            will simulate, with per-env paths when ``num_envs > 1``.
        num_envs: Number of envs the parse produced. ``1`` for single-world
            (no ``physx.clone`` issued); ``> 1`` for multi-world.
        template_root: Template prim scope to clone in ovphysx. ``None`` for
            single-world.
        env_roots: Per-env target prim scopes for ``physx.clone``. Empty for
            single-world; length ``num_envs`` for multi-world.
    """

    usd_path: str
    prim_paths: frozenset[str]
    path_body_map: Mapping[str, int]
    articulation_roots: tuple[str, ...] = ()
    num_envs: int = 1
    template_root: str | None = None
    env_roots: tuple[str, ...] = ()


class SolverPhysX(SolverBase, CouplingInterface):
    """PhysX-backed solver for rigid bodies and articulations, run as an entry
    in :class:`SolverCoupledProxy`.

    Wraps NVIDIA ovphysx. PhysX simulates the bodies and articulations authored
    in a USD; each :meth:`step` exchanges pose, velocity, wrench, and DOF-target
    data between Newton state and ovphysx tensor bindings. It can act as a coupling
    *source* (its bodies drive a proxy) or *destination* (proxy poses drive its
    bodies and the contact reactions are harvested back).

    Setup is parse -> finalize -> construct:

    .. code-block:: python

        parse_info = SolverPhysX.parse_usd(builder, "scene.usda")
        model = builder.finalize()
        solver = SolverPhysX(model, parse_info=parse_info)

    The :class:`PhysxParseInfo` returned by :meth:`parse_usd` must be handed to
    the constructor. As a coupling destination, call :meth:`finalize_coupling` to push
    the source bodies' mass / inertia / COM onto the proxy bodies.

    Limitations:

    - Single-world only. Multi-world coupling is structurally blocked
      at this stage; ``SolverPhysX.__init__`` raises
      ``NotImplementedError`` if ``model.world_count > 1``.
    - One USD per process (ovphysx is single-stage).
    - One scene-wide set of global parameters (gravity, dt, solver
      iterations).
    - No runtime body creation; every prim must be authored into the
      USD before construction.
    - Particle-proxy harvest is unsupported. Particle-source solvers
      (VBD cloth, MPM material points) cannot couple into SolverPhysX
      as destinations.
    """

    TEMPLATE_ROOT: str = "/World/envs/env_template"

    @classmethod
    def parse_usd(
        cls,
        builder: ModelBuilder,
        usd_path: str,
        *,
        num_envs: int = 1,
        pre_replicate_extend: Callable[[ModelBuilder], None] | None = None,
        **parse_kwargs,
    ) -> PhysxParseInfo:
        """Load the PhysX scene in ``usd_path`` into ``builder`` and return the
        :class:`PhysxParseInfo` the constructor needs.

        The function parses the PhysX scene into ``builder`` as ordinary Newton bodies
        and shapes. Those bodies are simulated by ovphysx; Newton only holds a mirror
        of them (for state I/O, rendering, and proxy pose reads).

        Beyond a plain Newton ``parse_usd``, this does exactly three things:

        - Captures the prim-path -> Newton-body-index map into the returned
          :class:`PhysxParseInfo`.
        - Records every ``PhysicsArticulationRootAPI`` prim so the constructor can
          bind all articulations without the caller enumerating them.
        - Sets ``shape_collision_group = 0`` on the shapes it adds; shapes already in
          ``builder`` are left untouched. Group 0 disables Newton's rigid-rigid broadphase
          for a shape, so ovphysx stays the sole owner of these bodies' contacts.
          This affects rigid-rigid pairing only; rigid-vs-particle (soft) contacts are
          unaffected, so proxy coupling into particle solvers (MPM, VBD) works.

        Multi-env mode is selected when ``num_envs > 1``. The USD must
        contain a prim at :attr:`TEMPLATE_ROOT` (``/World/envs/env_template``)
        in that case; the template scene is parsed once into a temporary
        builder, then added to ``builder`` ``num_envs`` times (each as its own
        world). Body, joint, and shape labels of the replicas are rewritten
        to per-env prim paths (``TEMPLATE_ROOT`` -> ``/World/envs/env_<i>``)
        so each Newton body's label matches its ovphysx counterpart after
        ``physx.clone`` runs in :meth:`__init__`. Bodies/joints/shapes/
        articulations not under ``TEMPLATE_ROOT`` (e.g. a shared ground
        plane) are also replicated per world; they participate in PhysX
        collisions only inside their env.

        At ``num_envs == 1`` the USD is parsed directly into ``builder`` and
        no cloning happens — even if the template scope exists. Body labels
        keep their authored prim path. The ``pre_replicate_extend`` callback,
        if provided, still runs against ``builder`` so single-env scenes can
        share the same extension code used at higher N. Passing ``num_envs > 1``
        without a template prim raises ``ValueError``.

        Multi-env physical layout: all env clones share the world origin in
        both Newton and ovphysx; per-env collision isolation is handled by
        ``body_world`` on the Newton side and PhysX ``env_id`` on the ovphysx
        side. For visual separation, call
        :meth:`newton.viewer.Viewer.set_world_offsets` on the viewer.

        Args:
            builder: ``ModelBuilder`` to receive the parsed bodies, joints,
                shapes, and articulations.
            usd_path: Path to the USD file to parse.
            num_envs: Number of parallel envs to instantiate. ``1`` (default)
                parses the USD directly into ``builder`` unless the USD
                contains the template scope, in which case it clones once.
                ``> 1`` requires the template scope to be present.
            pre_replicate_extend: Optional callback invoked on the per-env
                template ``ModelBuilder`` *after* PhysX parsing but *before*
                replication. Use it to add destination-side
                bodies/joints/shapes that must share the env world tag with
                the PhysX bodies (e.g. proxy targets for
                ``SolverCoupledProxy``). Only invoked when the template prim
                exists.
            **parse_kwargs: Forwarded to Newton's ``parse_usd`` (e.g.
                ``ignore_names``, ``parse_sites``, ``skip_mesh_approximation``).

        Returns:
            :class:`PhysxParseInfo` to hand to :meth:`__init__`.
        """
        from pxr import Usd, UsdPhysics

        from newton._src.utils.import_usd import parse_usd as _newton_parse_usd  # noqa: PLC0415

        stage = Usd.Stage.Open(usd_path)
        template_prim = stage.GetPrimAtPath(cls.TEMPLATE_ROOT)
        template_exists = bool(template_prim) and template_prim.IsValid()

        num_envs = max(1, int(num_envs))
        if num_envs > 1 and not template_exists:
            raise ValueError(
                f"parse_usd: num_envs={num_envs} requires a template scope at "
                f"{cls.TEMPLATE_ROOT!r}. Author the per-env content under that "
                "prim, or call parse_usd with num_envs=1 to parse the USD as a "
                "single-env scene."
            )
        single_env = num_envs == 1

        # Single-env parses straight into the caller's builder. Multi-env
        # parses into a temporary template builder, lets the caller extend it,
        # then replicates the template N times into ``builder`` with per-env
        # label rewriting so each replica's body/joint labels match what
        # ``physx.clone`` will create on the ovphysx side.
        if single_env:
            parse_target = builder
        else:
            import newton as _newton  # noqa: PLC0415

            parse_target = _newton.ModelBuilder()
            parse_target.up_axis = builder.up_axis

        pre_main_shape_count = builder.shape_count
        parse_result = _newton_parse_usd(parse_target, usd_path, **parse_kwargs)
        template_path_body_map = dict(parse_result.get("path_body_map", {}))

        usd_articulation_roots = tuple(
            prim.GetPath().pathString for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )

        if single_env:
            if pre_replicate_extend is not None:
                pre_replicate_extend(builder)
            for s in range(pre_main_shape_count, builder.shape_count):
                builder.shape_collision_group[s] = 0
            return PhysxParseInfo(
                usd_path=usd_path,
                prim_paths=frozenset(template_path_body_map.keys()),
                path_body_map=template_path_body_map,
                articulation_roots=usd_articulation_roots,
                num_envs=1,
                template_root=None,
                env_roots=(),
            )

        if pre_replicate_extend is not None:
            pre_replicate_extend(parse_target)

        template_root = cls.TEMPLATE_ROOT
        parent_scope = template_root.rsplit("/", 1)[0] + "/"
        env_roots = tuple(f"{parent_scope}env_{i}" for i in range(num_envs))

        # Replicate the template N times with per-env label rewriting. All envs
        # share the world origin physically; the viewer's ``set_world_offsets``
        # handles visual separation, while ``body_world`` (Newton) and
        # ``env_id`` (PhysX) handle collision isolation.
        path_body_map: dict[str, int] = {}
        env_articulation_roots: list[str] = []
        for env_root in env_roots:
            before_body_count = builder.body_count
            before_joint_count = builder.joint_count
            builder.add_world(parse_target)
            for new_b in range(before_body_count, builder.body_count):
                old_label = builder.body_label[new_b]
                if not old_label or not old_label.startswith(template_root):
                    continue
                new_label = env_root + old_label[len(template_root) :]
                builder.body_label[new_b] = new_label
                if old_label in template_path_body_map:
                    path_body_map[new_label] = new_b
            for new_j in range(before_joint_count, builder.joint_count):
                old_jl = builder.joint_label[new_j]
                if not old_jl or not old_jl.startswith(template_root):
                    continue
                builder.joint_label[new_j] = env_root + old_jl[len(template_root) :]
            for art_root in usd_articulation_roots:
                if art_root.startswith(template_root):
                    env_articulation_roots.append(env_root + art_root[len(template_root) :])

        for s in range(pre_main_shape_count, builder.shape_count):
            builder.shape_collision_group[s] = 0

        return PhysxParseInfo(
            usd_path=usd_path,
            prim_paths=frozenset(path_body_map.keys()),
            path_body_map=path_body_map,
            articulation_roots=tuple(env_articulation_roots),
            num_envs=num_envs,
            template_root=template_root,
            env_roots=env_roots,
        )

    def __init__(
        self,
        model: Model,
        *,
        parse_info: PhysxParseInfo,
        direct_gpu: bool = True,
    ) -> None:
        """Build the PhysX scene from ``parse_info`` and wire up the tensor
        bindings for the PhysX-owned bodies and articulations.

        Loads the USD recorded in ``parse_info`` into an ovphysx PhysX instance
        (the actual simulation), then creates the pose / velocity / wrench /
        DOF-target tensor bindings used to exchange state each :meth:`step`.

        Args:
            model: Newton ``Model`` produced by ``builder.finalize()`` after
                :meth:`parse_usd` ran against ``builder``.
            parse_info: The :class:`PhysxParseInfo` returned by :meth:`parse_usd`.
                Every articulation it recorded is bound automatically.
            direct_gpu: When ``True``, ovphysx keeps tensor bindings GPU-resident
                (DirectGPU). Set ``False`` only for diagnostics that need
                CPU-readable bindings.

        Raises:
            ValueError: If ``parse_info`` does not match ``model`` (its prim paths
                don't resolve to body labels in the model), or if a multi-env
                ``parse_info`` (``num_envs > 1``) is paired with a single-world
                model or vice versa.
            ImportError: If ovphysx is not installed.
            RuntimeError: If ovphysx's ``add_usd`` or ``clone`` fails.
        """
        # parse_info and model must agree on env count: multi-env parse_info
        # implies a replicated Newton model; conversely a single-env parse_info
        # cannot describe a multi-world model.
        model_world_count = int(getattr(model, "world_count", 1) or 1)
        parent = getattr(model, "parent", model)
        parent_world_count = int(getattr(parent, "world_count", 1) or 1)
        if int(parse_info.num_envs) != parent_world_count and parent_world_count > 1:
            raise ValueError(
                f"parse_info.num_envs={parse_info.num_envs} does not match parent "
                f"model.world_count={parent_world_count}; build the model with "
                "the same num_envs you pass to parse_usd()."
            )
        del model_world_count

        from ovphysx import PhysX, PhysXConfig  # noqa: PLC0415
        from ovphysx.types import TensorType  # noqa: PLC0415

        super().__init__(model)
        self._TensorType = TensorType
        self._parse_info = parse_info
        self._usd_path = parse_info.usd_path
        self._owned_prims: frozenset[str] = parse_info.prim_paths
        self._articulation_roots: list[str] = list(parse_info.articulation_roots)

        # Resolve names against the full (parent) model.
        self._parent_model = getattr(model, "parent", model)
        body_labels = list(getattr(self._parent_model, "body_label", None) or [])

        # Under SolverCoupled the state/control arrays handed to the hooks are
        # entry-local (compacted), so resolved global indices must be mapped into
        # that local space. The framework exposes the entry's inverse maps on the
        # view; they are identity for a standalone solver or a prefix-owned entry.
        _imaps = getattr(model, "coupled_index_maps", None)
        _body_g2l = _imaps.body_global_to_local.numpy() if _imaps is not None else None
        _dof_g2l = _imaps.joint_dof_global_to_local.numpy() if _imaps is not None else None
        # Parent (global) -> view-local body map, retained so coupling hooks can
        # read the framework's per-mirror effective-mass override off the view.
        self._body_g2l = _body_g2l
        # {mirror_label: (view_local_body_idx, source_com)}, populated by
        # :meth:`finalize_coupling`; drives :meth:`_push_proxy_inertials`.
        self._mirror_push_info: dict[str, tuple[int, np.ndarray]] = {}

        def _to_local_bodies(global_ids):
            if _body_g2l is None:
                return list(global_ids)
            return [int(_body_g2l[g]) if g >= 0 else -1 for g in global_ids]

        def _to_local_dofs(global_dofs):
            if _dof_g2l is None:
                return list(global_dofs)
            return [int(_dof_g2l[d]) if d >= 0 else -1 for d in global_dofs]

        # Check that the provided model has been built with parse_usd
        label_set = set(filter(None, body_labels))
        missing = [p for p in self._owned_prims if p not in label_set]
        if missing:
            raise ValueError(
                f"parse_info does not match model: prim paths "
                f"{missing[:3]}{'...' if len(missing) > 3 else ''} are not "
                "present as body labels in the model. Did you pass the wrong "
                "parse_info, or a model from a different builder?"
            )

        body_flags = self._parent_model.body_flags.numpy() if self._parent_model.body_flags is not None else None

        # Construct ovphysx PhysX scene and load the USD.
        if direct_gpu:
            self._physx = PhysX(
                device="gpu",
                config=PhysXConfig(
                    carbonite_overrides={
                        "/physics/suppressReadback": True,
                        "/physics/suppressFabricUpdate": True,
                    },
                ),
            )
            self._is_directgpu = True
        else:
            self._physx = PhysX(device="gpu")
            self._is_directgpu = False
        usd_handle, add_op = self._physx.add_usd(self._usd_path)
        self._physx.wait_op(add_op)
        self._usd_handle = usd_handle

        if parse_info.template_root is not None and parse_info.env_roots:
            clone_op = self._physx.clone(
                source_path=parse_info.template_root,
                target_paths=list(parse_info.env_roots),
            )
            self._physx.wait_op(clone_op)

        # --- Create articulation metadata and tensor bindings, one per root ---
        joint_labels = list(getattr(self._parent_model, "joint_label", None) or [])
        joint_qd_start = (
            self._parent_model.joint_qd_start.numpy() if self._parent_model.joint_qd_start is not None else None
        )
        self._articulations: list[dict] = []
        articulation_link_indices: set[int] = set()
        self._body_to_articulation: dict[int, tuple[int, int]] = {}
        for root_path in self._articulation_roots:
            pose_b = self._physx.create_tensor_binding(
                prim_paths=[root_path],
                tensor_type=TensorType.ARTICULATION_LINK_POSE,
                raise_if_empty=True,
            )
            vel_b = self._physx.create_tensor_binding(
                prim_paths=[root_path],
                tensor_type=TensorType.ARTICULATION_LINK_VELOCITY,
                raise_if_empty=True,
            )
            wrench_b = self._physx.create_tensor_binding(
                prim_paths=[root_path],
                tensor_type=TensorType.ARTICULATION_LINK_WRENCH,
                raise_if_empty=True,
            )
            link_names = list(pose_b.body_names)
            L = pose_b.body_count
            # Prim-path prefix shared by all of this articulation's links.
            # Either use the parent scope or the root path itself.
            root_short = root_path.rsplit("/", 1)[-1]
            if root_short in link_names:
                scope_prefix = root_path.rsplit("/", 1)[0] + "/"
            else:
                scope_prefix = root_path.rstrip("/") + "/"
            link_indices = _resolve_link_indices(root_path, link_names, body_labels, scope_prefix)
            articulation_link_indices.update(v for v in link_indices if v >= 0)
            a_idx = len(self._articulations)
            for link_idx, newton_body_idx in enumerate(link_indices):
                if newton_body_idx >= 0:
                    self._body_to_articulation[newton_body_idx] = (a_idx, link_idx)

            dof_target_b = self._physx.create_tensor_binding(
                prim_paths=[root_path],
                tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET,
                raise_if_empty=True,
            )
            D = dof_target_b.dof_count
            dof_buf: wp.array | None = None
            dof_newton_indices_warp: wp.array | None = None
            # Persistent drive-gain bindings + reusable buffers, keyed by gain name
            gain_types = {
                "stiffness": TensorType.ARTICULATION_DOF_STIFFNESS,
                "damping": TensorType.ARTICULATION_DOF_DAMPING,
                "max_force": TensorType.ARTICULATION_DOF_MAX_FORCE,
            }
            dof_gain_bindings: dict = {}
            dof_gain_bufs: dict = {}
            if D > 0:
                dof_buf = wp.zeros((1, D), dtype=wp.float32, device=self.device)
                if joint_qd_start is None:
                    raise ValueError(
                        f"Articulation {root_path!r} has {D} DOFs but Newton "
                        "model lacks joint_qd_start; cannot auto-derive DOF "
                        "mapping."
                    )
                dof_names = list(dof_target_b.dof_names)
                dof_newton_indices = _resolve_dof_indices(
                    root_path, dof_names, joint_labels, joint_qd_start, scope_prefix
                )
                dof_newton_indices_warp = wp.array(
                    _to_local_dofs(dof_newton_indices), dtype=wp.int32, device=self.device
                )
                dof_gain_bindings = {
                    name: self._physx.create_tensor_binding(prim_paths=[root_path], tensor_type=t, raise_if_empty=True)
                    for name, t in gain_types.items()
                }
                dof_gain_bufs = {name: wp.zeros((1, D), dtype=wp.float32, device=self.device) for name in gain_types}
            else:
                dof_target_b.destroy()
                dof_target_b = None

            self._articulations.append(
                {
                    "root_path": root_path,
                    "link_names": list(link_names),
                    "link_count": L,
                    "pose_binding": pose_b,
                    "vel_binding": vel_b,
                    "wrench_binding": wrench_b,
                    "link_indices": wp.array(_to_local_bodies(link_indices), dtype=wp.int32, device=self.device),
                    "pose_buf": wp.zeros((1, L), dtype=wp.transform, device=self.device),
                    "vel_buf": wp.zeros((1, L), dtype=wp.spatial_vector, device=self.device),
                    "wrench_buf": wp.zeros((1, L, 9), dtype=wp.float32, device=self.device),
                    "dof_target_binding": dof_target_b,
                    "dof_buf": dof_buf,
                    "dof_newton_indices": dof_newton_indices_warp,
                    "dof_count": (dof_target_b.dof_count if dof_target_b is not None else 0),
                    "dof_gain_bindings": dof_gain_bindings,
                    "dof_gain_bufs": dof_gain_bufs,
                }
            )

        # --- Create tensor bindings for rigid bodies
        self._simulated_body_mapping: dict[int, str] = {}
        if body_flags is not None:
            for i, lbl in enumerate(body_labels):
                if not lbl or lbl not in self._owned_prims:
                    continue
                if i in articulation_link_indices:
                    continue
                if int(body_flags[i]) & int(BodyFlags.DYNAMIC):
                    self._simulated_body_mapping[i] = lbl

        self._sim_pose_binding = None
        self._sim_vel_binding = None
        self._sim_force_binding = None
        self._sim_indices_warp = None
        self._sim_pose_buf = None
        self._sim_vel_buf = None
        self._sim_force_buf = None
        if self._simulated_body_mapping:
            sim_sorted_ids = sorted(self._simulated_body_mapping.keys())
            sim_sorted_paths = [self._simulated_body_mapping[i] for i in sim_sorted_ids]
            n_sim = len(sim_sorted_ids)
            self._sim_pose_binding = self._physx.create_tensor_binding(
                prim_paths=sim_sorted_paths,
                tensor_type=TensorType.RIGID_BODY_POSE,
                raise_if_empty=True,
            )
            self._sim_vel_binding = self._physx.create_tensor_binding(
                prim_paths=sim_sorted_paths,
                tensor_type=TensorType.RIGID_BODY_VELOCITY,
                raise_if_empty=True,
            )
            self._sim_force_binding = self._physx.create_tensor_binding(
                prim_paths=sim_sorted_paths,
                tensor_type=TensorType.RIGID_BODY_WRENCH,
                raise_if_empty=True,
            )
            self._sim_indices_warp = wp.array(_to_local_bodies(sim_sorted_ids), dtype=wp.int32, device=self.device)
            self._sim_pose_buf = wp.zeros(n_sim, dtype=wp.transform, device=self.device)
            self._sim_vel_buf = wp.zeros(n_sim, dtype=wp.spatial_vector, device=self.device)
            self._sim_force_buf = wp.zeros((n_sim, 9), dtype=wp.float32, device=self.device)

        # ovphysx time tracker.
        self._sim_time = 0.0

        # Cached per-body gravity acceleration for the proxy harvest override.
        self._proxy_gravity_accel: wp.array | None = None

    def _proxy_body_gravity_acceleration(self, count: int) -> wp.array:
        """Per-body gravity acceleration over the view, cached. Filled from the
        mixin default (reads ``model.gravity`` per world)."""
        if self._proxy_gravity_accel is None or self._proxy_gravity_accel.shape[0] != count:
            self._proxy_gravity_accel = wp.zeros(count, dtype=wp.vec3, device=self.device)
            CouplingInterface.coupling_eval_gravity_acceleration(
                self, out_body_acceleration=self._proxy_gravity_accel, out_particle_acceleration=None
            )
        return self._proxy_gravity_accel

    def coupling_rewind_proxy_body(
        self,
        body_local_to_proxy_global: wp.array[int],
        state: State,
        coupling_forces: wp.array[wp.spatial_vector],
        body_gravity_acceleration: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """No-op rewind: let PhysX apply full gravity to the mirror bodies.

        The framework default rewind pushes a gravity- and feedback-compensating
        wrench into the destination's ``body_f`` before its solve, expecting the
        destination to integrate that wrench so the harvested velocity change is
        already gravity-free. PhysX *does* apply such a pushed wrench, so leaving
        the default in place double-counts gravity once
        :meth:`coupling_harvest_proxy_wrenches` also removes it analytically —
        the mirror is over-supported and the coupled body diverges.

        Overriding to a no-op leaves ``body_f`` cleared (PhysX applies plain
        gravity) and the synced velocity untouched; the harvest then subtracts
        gravity itself. This is the stable pairing for a PhysX destination.
        """
        del body_local_to_proxy_global, state, coupling_forces, body_gravity_acceleration, dt

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        body_qd_before: wp.array[wp.spatial_vector] | None = None,
        state: State | None = None,
        state_out: State | None = None,
        contacts: Contacts | None = None,
        dt: float = 0.0,
    ) -> None:
        """Harvest the contact reaction from the PhysX-side momentum change.

        PhysX integrates the mirror under full gravity (see
        :meth:`coupling_rewind_proxy_body`), so the momentum residual
        ``m·(qd_after − qd_before)/dt`` contains gravity; subtract it
        analytically to recover the contact reaction. ``qd_before`` is the
        framework's pre-solve synced velocity. The result is written by
        assignment (not accumulation): PhysX proxy mappings are 1:1, so each
        global id is written once and ``coupling_forces`` holds the fresh
        per-step reaction.
        """
        del contacts
        n = body_local_to_proxy_global.shape[0]
        if n == 0:
            return
        if state_out is None or state_out.body_qd is None or body_qd_before is None:
            raise ValueError("SolverPhysX proxy harvest requires body_qd_before and state_out.body_qd")
        if dt <= 0.0:
            raise ValueError("SolverPhysX proxy harvest requires dt > 0")
        model = self.model
        wp.launch(
            _harvest_proxy_wrenches_kernel,
            dim=n,
            inputs=[
                float(dt),
                body_local_to_proxy_global,
                body_qd_before,
                state_out.body_qd,
                state.body_f,
                model.body_mass,
                model.body_inertia,
                state_out.body_q,
                self._proxy_body_gravity_acceleration(int(state_out.body_qd.shape[0])),
                out_body_f,
            ],
            device=model.device,
        )

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the PhysX scene by ``dt`` and read the result into ``state_out``.

        Pushes the per-articulation DOF position targets from ``control``, steps
        ovphysx, then reads the post-step pose and velocity of both standalone
        bodies and articulation links into ``state_out``.

        Args:
            state_in: Unused; PhysX advances from its own internal state
                updated through :meth:`coupling_notify_input_state_update`.
            state_out: Receives the post-step pose and velocity of both
                standalone bodies and articulation links.
            control: Per-articulation DOF position targets (``joint_target_q``);
                ``None`` leaves the current targets unchanged.
            contacts: Ignored — ovphysx owns its contact pipeline.
            dt: Time step [s].
        """
        del state_in, contacts

        # Per-articulation DOF position targets (when control provides them)
        if control is not None and getattr(control, "joint_target_q", None) is not None:
            for artic in self._articulations:
                if artic["dof_target_binding"] is None:
                    continue
                wp.launch(
                    _gather_articulation_dof,
                    dim=artic["dof_count"],
                    inputs=[control.joint_target_q, artic["dof_newton_indices"]],
                    outputs=[artic["dof_buf"]],
                    device=self.device,
                )
                artic["dof_target_binding"].write(artic["dof_buf"])

        # Step ovphysx
        self._physx.step_sync(dt, self._sim_time)
        self._sim_time += dt

        # Read post-step state back
        for artic in self._articulations:
            artic["pose_binding"].read(artic["pose_buf"])
            wp.launch(
                _scatter_articulation_link_q,
                dim=artic["link_count"],
                inputs=[artic["pose_buf"], artic["link_indices"]],
                outputs=[state_out.body_q],
                device=self.device,
            )
            artic["vel_binding"].read(artic["vel_buf"])
            wp.launch(
                _scatter_articulation_link_qd,
                dim=artic["link_count"],
                inputs=[artic["vel_buf"], artic["link_indices"]],
                outputs=[state_out.body_qd],
                device=self.device,
            )
        if self._sim_pose_binding is not None:
            self._sim_pose_binding.read(self._sim_pose_buf)
            wp.launch(
                _scatter_body_q,
                dim=self._sim_indices_warp.shape[0],
                inputs=[self._sim_pose_buf, self._sim_indices_warp],
                outputs=[state_out.body_q],
                device=self.device,
            )
            self._sim_vel_binding.read(self._sim_vel_buf)
            wp.launch(
                _scatter_body_qd,
                dim=self._sim_indices_warp.shape[0],
                inputs=[self._sim_vel_buf, self._sim_indices_warp],
                outputs=[state_out.body_qd],
                device=self.device,
            )

    def coupling_notify_input_state_update(
        self,
        state: State,
        flags: StateFlags,
        *,
        iteration_restart: bool = False,
        dt: float = 0.0,
    ) -> None:
        """Sync Newton-side state changes into ovphysx.

        Called by ``SolverCoupledProxy`` after the framework writes
        into a state field. Per-flag behavior:

        - ``BODY_Q`` / ``BODY_QD``: push standalone-body pose and velocity into
          PhysX — this is how a destination drive's imposed pose reaches PhysX.
          Articulation-link poses are joint-constrained and not pushed.
        - ``BODY_F``: push wrench (force at COM + torque) to articulation
          links and PhysX-simulated standalone bodies.
        - other flags: ignored. (Particle flags are not handled —
          particle-source coupling is unsupported; see
          :meth:`coupling_harvest_proxy_particle_forces`.)
        """
        del iteration_restart, dt
        if flags & (StateFlags.BODY_Q | StateFlags.BODY_QD):
            # Source mode never raises this notify, so no guard is needed here.
            if self._sim_pose_binding is not None:
                wp.launch(
                    _gather_body_q,
                    dim=self._sim_indices_warp.shape[0],
                    inputs=[state.body_q, self._sim_indices_warp],
                    outputs=[self._sim_pose_buf],
                    device=self.device,
                )
                self._sim_pose_binding.write(self._sim_pose_buf)
                wp.launch(
                    _gather_body_qd,
                    dim=self._sim_indices_warp.shape[0],
                    inputs=[state.body_qd, self._sim_indices_warp],
                    outputs=[self._sim_vel_buf],
                    device=self.device,
                )
                self._sim_vel_binding.write(self._sim_vel_buf)
        if flags & StateFlags.BODY_F:
            # Articulations and dynamic standalone bodies (``_sim_*``) consume
            # ``body_f`` as an external wrench.
            for artic in self._articulations:
                wp.launch(
                    _gather_articulation_link_wrench,
                    dim=artic["link_count"],
                    inputs=[state.body_f, state.body_q, self.model.body_com, artic["link_indices"]],
                    outputs=[artic["wrench_buf"]],
                    device=self.device,
                )
                artic["wrench_binding"].write(artic["wrench_buf"])
            if self._sim_force_binding is not None:
                wp.launch(
                    _gather_body_wrench,
                    dim=self._sim_indices_warp.shape[0],
                    inputs=[
                        state.body_f,
                        state.body_q,
                        self.model.body_com,
                        self._sim_indices_warp,
                    ],
                    outputs=[self._sim_force_buf],
                    device=self.device,
                )
                self._sim_force_binding.write(self._sim_force_buf)

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global: wp.array[int],
        out_particle_f: wp.array[wp.vec3],
        *,
        particle_qd_before: wp.array[wp.vec3] | None = None,
        state: State | None = None,
        state_out: State | None = None,
        contacts: Contacts | None = None,
        dt: float = 0.0,
    ) -> None:
        """Reject proxy-particle feedback harvesting — PhysX has no particles."""
        del particle_local_to_proxy_global, out_particle_f
        del particle_qd_before, state, state_out, contacts, dt
        raise NotImplementedError("SolverPhysX does not support proxy particle harvest")

    def coupling_eval_effective_mass_block(
        self,
        endpoint_kind: wp.array,
        endpoint_index: wp.array,
        endpoint_local_pos: wp.array,
        out_mass: wp.array,
        out_inertia: wp.array | None = None,
    ) -> None:
        """Return the operational-space inertia at the requested
        endpoints.

        For articulation links: returns ``Λ_v = (J·M⁻¹·Jᵀ)⁻¹``
        evaluated at the current articulation configuration.
        ``out_mass`` receives ``trace(Λ_v) / 3`` (scalar isotropic
        mass); ``out_inertia`` (if provided) receives ``Λ_ω`` rotated
        into the link's body frame.

        For standalone bodies or zero-DOF articulations: returns the
        intrinsic ``body_mass`` / ``body_inertia`` from the model.

        **Assumption:** the articulation effective mass is configuration-dependent (``J = J(q)``),
        so in principle it should be recomputed every step. It is instead evaluated once
        at the construction pose and held fixed as the joints move, so the proxy mass is
        an approximation.

        **Performance:** setup-time only — ``SolverCoupledProxy`` calls this once at
        construction and again only on a ``BODY_INERTIAL_PROPERTIES`` model change,
        never in the step loop.

        Raises:
            NotImplementedError: If ``endpoint_local_pos`` is non-zero.
                Offset application points are not supported in this
                version.
        """
        if np.any(np.abs(endpoint_local_pos.numpy()) > 1e-9):
            raise NotImplementedError(
                "SolverPhysX.coupling_eval_effective_mass_block: non-zero "
                "endpoint_local_pos not supported. The effective mass is computed "
                "at the link's body origin only."
            )

        kinds = endpoint_kind.numpy()
        indices = endpoint_index.numpy()
        out_mass.zero_()
        # The effective-mass kernel always writes inertia
        # alias a scratch buffer when the caller wants mass only.
        inertia = (
            out_inertia
            if out_inertia is not None
            else wp.zeros(int(indices.shape[0]), dtype=wp.mat33, device=out_mass.device)
        )
        inertia.zero_()
        body_mass = self.model.body_mass
        body_inertia = self.model.body_inertia

        # Bucket each body endpoint under its owning articulation
        # Each endpoint is recorded as (output slot, articulation link index, body index)
        # ``groups[a_idx]`` the list of endpoints to solve for articulation ``a_idx``.
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for slot in range(int(indices.shape[0])):
            if int(kinds[slot]) != int(CouplingInterface.EndpointKind.BODY):
                continue
            body_idx = int(indices[slot])
            a_idx, link_idx = self._body_to_articulation.get(body_idx, (-1, -1))
            groups.setdefault(a_idx, []).append((slot, link_idx, body_idx))

        # Articulation path: compute the operational-space inertia
        intrinsic_slots: list[int] = []
        intrinsic_bodies: list[int] = []
        for a_idx, endpoints in groups.items():
            slots, links, bodies = (list(col) for col in zip(*endpoints, strict=True))
            # Skip for bodies that are not part of an articulation
            if a_idx < 0:
                intrinsic_slots += slots
                intrinsic_bodies += bodies
                continue
            artic = self._articulations[a_idx]
            # Read the Jacobian (6*L, dof) and mass matrix (dof, dof) into device
            # buffers; the bindings carry a leading length-1 axis, dropped below.
            with (
                self._physx.create_tensor_binding(
                    prim_paths=[artic["root_path"]],
                    tensor_type=self._TensorType.ARTICULATION_JACOBIAN,
                    raise_if_empty=True,
                ) as jb,
                self._physx.create_tensor_binding(
                    prim_paths=[artic["root_path"]],
                    tensor_type=self._TensorType.ARTICULATION_MASS_MATRIX,
                    raise_if_empty=True,
                ) as mb,
            ):
                _, j_rows, j_cols = jb.shape
                _, m_rows, _ = mb.shape
                if j_rows == 0 or j_cols == 0:
                    intrinsic_slots += slots
                    intrinsic_bodies += bodies
                    continue
                jacobian = wp.zeros((1, j_rows, j_cols), dtype=wp.float32, device=self.device)
                mass_matrix = wp.zeros((1, m_rows, m_rows), dtype=wp.float32, device=self.device)
                jb.read(jacobian)
                mb.read(mass_matrix)
            # Specialize on the mass-matrix dimension — the generalized DOF count,
            # which includes the floating-base DOFs, not the actuated dof_count.
            artic["pose_binding"].read(artic["pose_buf"])
            wp.launch_tiled(
                _make_effective_mass_kernel(int(m_rows)),
                dim=[len(endpoints)],
                inputs=[
                    mass_matrix.reshape((m_rows, m_rows)),
                    jacobian.reshape((j_rows, j_cols)),
                    artic["pose_buf"],
                    wp.array(links, dtype=wp.int32, device=self.device),
                    wp.array(slots, dtype=wp.int32, device=self.device),
                    wp.array(bodies, dtype=wp.int32, device=self.device),
                    body_mass,
                    body_inertia,
                ],
                outputs=[out_mass, inertia],
                block_dim=64,
                device=self.device,
            )

        # Rigid body path: use their intrinsic mass / inertia
        if intrinsic_slots:
            wp.launch(
                _gather_intrinsic_inertia,
                dim=len(intrinsic_slots),
                inputs=[
                    wp.array(intrinsic_slots, dtype=wp.int32, device=self.device),
                    wp.array(intrinsic_bodies, dtype=wp.int32, device=self.device),
                    body_mass,
                    body_inertia,
                ],
                outputs=[out_mass, inertia],
                device=self.device,
            )

    def finalize_coupling(
        self,
        proxies: Sequence[SolverCoupledProxy.Proxy],
    ) -> None:
        """Record each mirror's coupling inertials and push them to PhysX.

        For each ``(source_idx, mirror_idx)`` pair in ``proxies`` whose
        ``mirror_idx`` names a SolverPhysX-owned body, records the mirror's
        view-local body index and the source-body COM, then writes the mirror
        inertials via :meth:`_push_proxy_inertials`. Mass and inertia are read
        from ``self.model`` — the coupling framework installs the source body's
        *effective* (articulated, operational-space) mass there — so the
        pose-driven mirror matches the source's interface impedance rather than
        its raw link mass. COM comes from the source body, since the framework's
        inertia override does not touch ``body_com``.

        Matching interface impedance is what keeps the explicit
        source→mirror→source coupling loop stable under hard contact; using the
        raw link mass leaves the loop under-damped and prone to blow-up.

        Pairs whose mirror is not SolverPhysX-owned are skipped (they describe
        other solvers' destinations). Must be called after :meth:`__init__`
        (once the framework has applied the effective-mass override) and before
        the first step. Source COM is read from the parent ``Model`` — not from
        ``self.model``, which does not contain the source-side bodies.
        """
        parent = self._parent_model
        body_labels = list(getattr(parent, "body_label", None) or [])
        model_com = parent.body_com.numpy() if parent.body_com is not None else None

        # Walk every (source_idx, mirror_idx) pair across all proxies; keep only
        # the ones whose mirror is SolverPhysX-owned, resolving each mirror to
        # its view-local index so the harvest can read the framework override.
        self._mirror_push_info = {}
        for proxy in proxies:
            for raw_source_idx, raw_mirror_idx in zip(proxy.bodies, proxy.proxy_bodies, strict=True):
                source_idx = int(raw_source_idx)
                mirror_idx = int(raw_mirror_idx)
                if not (0 <= mirror_idx < len(body_labels)):
                    continue
                mirror_label = body_labels[mirror_idx]
                if not mirror_label or mirror_label not in self._owned_prims:
                    continue
                if model_com is None or source_idx < 0 or source_idx >= len(model_com):
                    continue
                local_idx = int(self._body_g2l[mirror_idx]) if self._body_g2l is not None else mirror_idx
                if local_idx < 0:
                    continue
                self._mirror_push_info[mirror_label] = (local_idx, np.asarray(model_com[source_idx], dtype=np.float32))

        self._push_proxy_inertials()

    def _push_proxy_inertials(self) -> None:
        """Write each mirror's effective mass / inertia / COM onto its PhysX prim.

        Mass and inertia are read fresh from ``self.model`` (the framework's
        effective-mass override, already scaled by ``proxy.mass_scale``); COM
        comes from the source body cached in :attr:`_mirror_push_info`. Inertia
        is diagonalized into PhysX's principal-moments + COM-pose-orientation
        form. Idempotent, so it can re-run whenever the override changes (see
        :meth:`notify_model_changed`).
        """
        if not self._mirror_push_info:
            return
        view_mass = self.model.body_mass.numpy() if self.model.body_mass is not None else None
        view_inertia = self.model.body_inertia.numpy() if self.model.body_inertia is not None else None
        if view_mass is None or view_inertia is None:
            return
        for mirror_label, (local_idx, com) in self._mirror_push_info.items():
            if not (0 <= local_idx < len(view_mass)):
                continue
            mass = float(view_mass[local_idx])
            inertia = np.asarray(view_inertia[local_idx])
            moments, quat = _diagonalize_body_inertia(inertia)
            diag_inertia = np.zeros(9, dtype=np.float32)
            diag_inertia[0] = moments[0]
            diag_inertia[4] = moments[1]
            diag_inertia[8] = moments[2]
            com_pose = np.asarray(
                [com[0], com[1], com[2], quat[0], quat[1], quat[2], quat[3]],
                dtype=np.float32,
            )
            mass_buf = wp.array([mass], dtype=wp.float32, device=self.device)
            inertia_buf = wp.array([diag_inertia], dtype=wp.float32, device=self.device)
            com_pose_buf = wp.array([com_pose], dtype=wp.float32, device=self.device)
            # Write order matters: inertia first (resets COM orientation), COM-pose last.
            self._one_shot_write(mirror_label, self._TensorType.RIGID_BODY_MASS, mass_buf)
            self._one_shot_write(mirror_label, self._TensorType.RIGID_BODY_INERTIA, inertia_buf)
            self._one_shot_write(mirror_label, self._TensorType.RIGID_BODY_COM_POSE, com_pose_buf)

    def notify_model_changed(self, flags: int) -> None:
        """Forward the framework's effective-mass override to the PhysX solver.

        The coupling framework writes the source body's effective mass onto the
        mirror bodies in ``self.model`` and calls this with
        :attr:`~newton.ModelFlags.BODY_INERTIAL_PROPERTIES`; re-pushing those
        inertials (see :meth:`_push_proxy_inertials`) is what delivers the
        impedance match to ovphysx. A no-op before :meth:`finalize_coupling` has
        recorded the mirror set.
        """
        if int(flags) & int(ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._push_proxy_inertials()

    def set_articulation_drive_gains(
        self,
        articulation_root_path: str,
        *,
        stiffness: float | wp.array[wp.float32] | None = None,
        damping: float | wp.array[wp.float32] | None = None,
        max_force: float | wp.array[wp.float32] | None = None,
    ) -> None:
        """Write stiffness, damping, and/or maximum force to an articulation's
        per-DOF position drives.

        Each provided gain is a scalar (broadcast to every DOF) or a ``wp.array``
        indexed like ``control.joint_target_q`` (by Newton DOF index); a gain
        left ``None`` is not written.

        Args:
            articulation_root_path: USD prim path of the articulation root, one
                of the roots bound at construction (auto-derived or explicit).
            stiffness: Proportional gain [N·m/rad].
            damping: Derivative gain [N·m·s/rad].
            max_force: Maximum drive force [N·m].

        Raises:
            ValueError: If ``articulation_root_path`` is not a bound articulation
                root, or the articulation has no actuated DOFs.
        """
        artic = None
        for a in self._articulations:
            if a["root_path"] == articulation_root_path:
                artic = a
                break
        if artic is None:
            raise ValueError(
                f"{articulation_root_path!r} is not a bound articulation root; "
                "it must be a PhysicsArticulationRootAPI prim in the loaded USD."
            )
        n_dofs = int(artic["dof_count"])
        if n_dofs == 0:
            raise ValueError(f"Articulation {articulation_root_path!r} has no actuated DOFs.")

        for name, value in (("stiffness", stiffness), ("damping", damping), ("max_force", max_force)):
            if value is None:
                continue
            buf = artic["dof_gain_bufs"][name]
            if isinstance(value, wp.array):
                wp.launch(
                    _gather_articulation_dof,
                    dim=n_dofs,
                    inputs=[value, artic["dof_newton_indices"]],
                    outputs=[buf],
                    device=self.device,
                )
            else:
                buf.fill_(float(value))
            artic["dof_gain_bindings"][name].write(buf)

    def release(self) -> None:
        """Release the ovphysx resources held by this solver.

        Destroys every tensor binding and the underlying PhysX instance,
        freeing those resources deterministically rather than at
        garbage-collection time. The solver is unusable afterward.

        Only one PhysX instance may exist per process, so this does *not*
        enable constructing a second solver.
        """
        for attr in ("_sim_pose_binding", "_sim_vel_binding", "_sim_force_binding"):
            b = getattr(self, attr, None)
            if b is not None:
                b.destroy()
                setattr(self, attr, None)
        for artic in getattr(self, "_articulations", []):
            for key in (
                "pose_binding",
                "vel_binding",
                "wrench_binding",
                "dof_target_binding",
            ):
                b = artic.get(key)
                if b is not None:
                    b.destroy()
                    artic[key] = None
            for name, b in artic.get("dof_gain_bindings", {}).items():
                if b is not None:
                    b.destroy()
                    artic["dof_gain_bindings"][name] = None
        self._articulations = []
        if getattr(self, "_physx", None) is not None:
            self._physx.release()
            self._physx = None

    def __del__(self) -> None:
        # Last-chance cleanup. Swallow failures because __del__ cannot
        # safely propagate exceptions (interpreter teardown, partial
        # initialization, etc.).
        try:
            self.release()
        except Exception:
            pass

    def _one_shot_write(self, prim_path: str, tensor_type, in_buf) -> None:
        """Create a one-shot tensor binding, write ``in_buf`` to it,
        destroy the binding."""
        with self._physx.create_tensor_binding(
            prim_paths=[prim_path],
            tensor_type=tensor_type,
            raise_if_empty=True,
        ) as b:
            b.write(in_buf)
