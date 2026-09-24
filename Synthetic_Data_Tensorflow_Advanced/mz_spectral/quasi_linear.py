#!/usr/bin/env python3
"""
Quasi-linear (QL) closure for RRU backscatter onto resolved modes.

We approximate the unresolved contribution to the resolved equation by a
single scalar eddy coefficient ν_QL(t) multiplying a resolved-only operator:

    B_QL(φ_R) ≈ ν_QL(t) · P_R @ D₂ @ φ_R

where D₂ is the same second-derivative stencil as in Dupire's spatial operator
(without ηk² prefactor). ν is fitted by least squares against the Markovian
MZ residual computed from truth trajectories.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def ql_backscatter_term(nu: float, P_R: np.ndarray, D2: np.ndarray, phi_R: np.ndarray) -> np.ndarray:
    """ν · P_R @ D₂ @ φ_R (same shape as φ)."""
    return nu * (P_R @ (D2 @ phi_R))


def fit_nu_ql(
    phi_truth: np.ndarray,
    t_tilde: np.ndarray,
    L_RR_rows: List[np.ndarray],
    P_R: np.ndarray,
    D2: np.ndarray,
    *,
    ridge: float = 1e-8,
    nu_clip: float = 5e-4,
    nonneg: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Global ν and per-time ν(t) via ridge least squares on interior FD slices.

    The fitted ν multiplies the QL backscatter term +ν·P_R·D₂·φ that is added to
    the resolved RHS downstream (see ``ql_backscatter_term`` / ``integrate_rrr_ql``).
    With ν>0 this is a positive eddy-diffusion — the physically-correct sign for a
    Mori–Zwanzig memory from integrating out the high-frequency modes. With ν<0 it
    is anti-diffusion (a backward-heat term), which is ill-posed and destabilizing.

    By default (``nonneg=True``) ν is therefore constrained to be non-negative.
    Because the closure has a single scalar coefficient, the constrained optimum is
    just the projection of the unconstrained ridge-LS optimum ν*=⟨lhs,q⟩/⟨q,q⟩ onto
    [0, nu_clip] — i.e. ``max(0, ν*)`` clipped above — which is exactly the
    non-negative least-squares (NNLS) solution. The closure can no longer pick an
    anti-diffusive coefficient.

    Set ``nonneg=False`` to recover the legacy symmetric clip ν ∈ [−nu_clip, nu_clip]
    (permits the anti-diffusive sign; retained only for before/after diagnostics).
    """
    nu_lo = 0.0 if nonneg else -nu_clip
    n_t, n_k = phi_truth.shape
    numers = []
    denoms = []
    nu_steps = []
    for i in range(1, n_t - 1):
        dt = t_tilde[i + 1] - t_tilde[i - 1]
        dphi = (phi_truth[i + 1] - phi_truth[i - 1]) / dt
        phi_R = P_R @ phi_truth[i]
        lhs = P_R @ dphi - L_RR_rows[i] @ phi_R
        q = P_R @ (D2 @ phi_R)
        num = float(np.dot(lhs, q))
        den = float(np.dot(q, q)) + ridge
        nu_i = num / den
        nu_i = float(np.clip(nu_i, nu_lo, nu_clip))
        nu_steps.append(nu_i)
        numers.append(num)
        denoms.append(den + ridge)
    nu_global_raw = float(sum(numers) / (sum(denoms) + 1e-16))
    nu_global = float(np.clip(nu_global_raw, nu_lo, nu_clip))
    return nu_global, np.asarray(nu_steps, dtype=float)
