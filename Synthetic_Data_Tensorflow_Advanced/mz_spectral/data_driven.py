#!/usr/bin/env python3
"""
Data-driven MZ setup: build L_RR from Dupire-inverted σ(φ_data), integrate RRR/QL, optional σ fixed-point.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mz_spectral.fourier_dupire import TKGrid
from mz_spectral.mz_decomposition import apply_projector, low_pass_projector_matrix
from mz_spectral.quasi_linear import fit_nu_ql
from mz_spectral.sigma_from_phi import (
    build_eta_sigma_from_phi,
    relative_sigma_change,
)
from mz_spectral.validation import (
    integrate_rrr_only,
    integrate_rrr_ql,
    precompute_L_and_blocks,
)


def build_L_rr_from_phi_data(
    phi_data: np.ndarray,
    grid: TKGrid,
    D2: np.ndarray,
    T_max: float,
    keep: int,
    *,
    filter_phi: bool = True,
    filter_phi_adaptive: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Dupire-invert φ_data → σ, then precompute L_RR rows and P_R.

    Returns ``(L_RR_rows, P_R, sigma_used, eta_used)``.
    """
    eta, sigma = build_eta_sigma_from_phi(
        phi_data,
        grid,
        D2,
        T_max,
        keep,
        filter_phi=filter_phi,
        filter_phi_adaptive=filter_phi_adaptive,
        regularize=True,
    )
    _, P_R, _, L_RR_rows = precompute_L_and_blocks(grid, sigma, T_max, keep)
    return L_RR_rows, P_R, sigma, eta


def run_data_driven_rom_integration(
    phi_data: np.ndarray,
    grid: TKGrid,
    D2: np.ndarray,
    config: Any,
    keep: int,
    *,
    vol_iterate: int = 1,
    vol_iterate_tol: float = 1e-2,
    vol_blend_alpha: float = 0.35,
    filter_phi_for_sigma: bool = False,
    filter_phi_adaptive: bool = True,
) -> Dict[str, Any]:
    """
    Data-driven ROM: σ from φ_data, optional fixed-point on σ, RRR + RRR+QL integration.

    IC: ``P_R @ phi_data[0]``. QL ν fitted against ``phi_data`` (not oracle-only truth label).
    """
    T_max = float(config.T_max)
    n_k = len(grid.k_tilde)
    P_R = low_pass_projector_matrix(n_k, keep)

    sigma_history: List[np.ndarray] = []
    phi_for_L = phi_data

    L_RR_rows: List[np.ndarray] = []
    sigma_gen = np.zeros_like(grid.K)
    eta_gen = np.zeros_like(grid.K)

    for k in range(max(1, int(vol_iterate))):
        L_RR_rows, P_R, sigma_gen, eta_gen = build_L_rr_from_phi_data(
            phi_for_L,
            grid,
            D2,
            T_max,
            keep,
            filter_phi=filter_phi_for_sigma,
            filter_phi_adaptive=filter_phi_adaptive,
        )
        sigma_history.append(np.array(sigma_gen, copy=True))

        if k > 0 and len(sigma_history) >= 2:
            rel = relative_sigma_change(sigma_history[-2], sigma_history[-1])
            if np.isfinite(rel) and rel < vol_iterate_tol:
                break

        if k < vol_iterate - 1:
            phi0_probe = apply_projector(P_R, phi_data[0])
            phi_probe = integrate_rrr_only(phi0_probe, grid.t_tilde, L_RR_rows, P_R)
            _, sigma_new = build_eta_sigma_from_phi(
                phi_probe,
                grid,
                D2,
                T_max,
                keep,
                filter_phi=filter_phi_for_sigma,
                filter_phi_adaptive=filter_phi_adaptive,
                regularize=True,
            )
            phi_for_L = phi_probe
            sigma_gen = (1.0 - vol_blend_alpha) * sigma_gen + vol_blend_alpha * sigma_new

    nu_g, nu_steps = fit_nu_ql(phi_data, grid.t_tilde, L_RR_rows, P_R, D2)
    phi0 = apply_projector(P_R, phi_data[0])
    phi_rrr = integrate_rrr_only(phi0, grid.t_tilde, L_RR_rows, P_R)
    phi_ql = integrate_rrr_ql(phi0, grid.t_tilde, L_RR_rows, P_R, D2, nu_g)

    _, sigma_data = build_eta_sigma_from_phi(
        phi_data,
        grid,
        D2,
        T_max,
        keep,
        filter_phi=filter_phi_for_sigma,
        filter_phi_adaptive=filter_phi_adaptive,
        regularize=True,
    )

    return {
        "phi_rrr": phi_rrr,
        "phi_ql": phi_ql,
        "phi0": phi0,
        "nu_ql_global": float(nu_g),
        "nu_ql_steps": nu_steps,
        "P_R": P_R,
        "L_RR_rows": L_RR_rows,
        "sigma_data": sigma_data,
        "sigma_generator": sigma_gen,
        "sigma_history": sigma_history,
        "vol_iterations_run": len(sigma_history),
    }
