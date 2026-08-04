#!/usr/bin/env python3
"""
Sweep resolved Fourier-mode cutoff `keep` (low-pass index in mz_decomposition)
and score RRR-only / RRR+QL errors vs truth on φ and PDF.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from mz_spectral.fourier_dupire import (
    build_L_operator,
    build_second_derivative_matrix,
    eta_tilde_from_sigma,
    verify_ode_residual,
)
from mz_spectral.quasi_linear import fit_nu_ql
from mz_spectral.validation import (
    integrate_rrr_ql,
    integrate_rrr_only,
    payoff_phi_tilde,
    pdf_from_phi_tilde,
    pdf_metrics,
    phi_l2_error,
    precompute_L_and_blocks,
    uuu_gap_metric,
)


def default_keep_values(n_k: int) -> List[int]:
    """
    Diagnostic sweep cutoffs including a moderate truncation (32) and Nyquist-1.

    Values are clamped to ``[1, n_k//2 - 1]``.
    """
    keep_max = max(1, n_k // 2 - 1)
    candidates = {4, 8, 16, 32, keep_max}
    if n_k < 16:
        candidates = {1, 2, 3, max(1, keep_max // 2), keep_max}
    out = sorted({int(c) for c in candidates if 1 <= int(c) <= keep_max})
    return out if out else [1]


def interpolate_phi_from_training_npz(
    npz_path: str,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    S0: float = 1000.0,
) -> np.ndarray:
    """
    Interpolate call price C from flattened MC training samples onto (T_i, K_ij).

    Returns φ̃ = C / S0 on same shape as K_grid.
    """
    data = np.load(npz_path)
    T_flat = data["T"].astype(float).ravel()
    K_flat = data["K"].astype(float).ravel()
    phi_flat = data["phi"].astype(float).ravel()
    if "S0" in data.files:
        S0 = float(data["S0"])
    pts = np.column_stack([T_flat, K_flat])
    interp = LinearNDInterpolator(pts, phi_flat, fill_value=np.nan)
    Ti, Ki = np.broadcast_arrays(T_grid[:, None], K_grid)
    phi = interp(Ti.ravel(), Ki.ravel()).reshape(K_grid.shape)
    if np.any(np.isnan(phi)):
        phi = np.nan_to_num(phi, nan=np.nanmean(phi_flat))
    return phi / S0


def sweep_k_trunc(
    phi_truth: np.ndarray,
    grid: Any,
    sigma: np.ndarray,
    config: Any,
    keep_values: Sequence[int],
    *,
    pdf_row_index: int = -1,
) -> List[Dict[str, Any]]:
    """
    For each `keep`, integrate RRR-only and RRR+QL; report errors vs truth.

    pdf_row_index: which maturity row to score PDF (default last).
    """
    rows: List[Dict[str, Any]] = []
    T_max = float(config.T_max)
    r = float(config.r)
    S0 = float(config.S0)
    K_max = float(config.K_max)
    t_tilde = grid.t_tilde
    k_tilde = grid.k_tilde
    K_mat = grid.K

    n_t, n_k = phi_truth.shape
    D2 = build_second_derivative_matrix(n_k, grid.dk)
    L_stack = np.zeros((n_t, n_k, n_k))
    for i in range(n_t):
        eta_row = eta_tilde_from_sigma(sigma[i : i + 1], T_max)[0]
        L_stack[i] = build_L_operator(eta_row, k_tilde, D2)

    mean_ode, _ = verify_ode_residual(phi_truth, t_tilde, L_stack)

    i_pdf = pdf_row_index % n_t
    T_sel = float(grid.T[i_pdf])
    K_sel = K_mat[i_pdf]

    f_truth = pdf_from_phi_tilde(
        phi_truth[i_pdf], T_sel, r, S0, K_max, K_sel, D2, normalize=True
    )

    for keep in keep_values:
        D2b, P_R, _L_rows, L_RR_rows = precompute_L_and_blocks(grid, sigma, T_max, int(keep))
        assert np.allclose(D2, D2b)

        phi0 = payoff_phi_tilde(k_tilde, float(S0), float(K_max))
        phi_rrr = integrate_rrr_only(phi0, t_tilde, L_RR_rows, P_R)
        err_phi_rrr = phi_l2_error(phi_rrr, phi_truth, k_tilde)

        nu_g, _ = fit_nu_ql(phi_truth, t_tilde, L_RR_rows, P_R, D2)
        phi_ql = integrate_rrr_ql(phi0, t_tilde, L_RR_rows, P_R, D2, nu_g)
        err_phi_ql = phi_l2_error(phi_ql, phi_truth, k_tilde)

        f_rrr = pdf_from_phi_tilde(
            phi_rrr[i_pdf], T_sel, r, S0, K_max, K_sel, D2, normalize=True
        )
        f_ql = pdf_from_phi_tilde(
            phi_ql[i_pdf], T_sel, r, S0, K_max, K_sel, D2, normalize=True
        )
        m_rrr = pdf_metrics(f_rrr, f_truth, K_sel)
        m_ql = pdf_metrics(f_ql, f_truth, K_sel)

        rows.append(
            {
                "keep": int(keep),
                "mean_relative_ode_error": float(mean_ode),
                "l2_phi_rrr": err_phi_rrr,
                "l2_phi_rrr_ql": err_phi_ql,
                "l2_pdf_rrr": m_rrr["l2_pdf"],
                "l2_pdf_rrr_ql": m_ql["l2_pdf"],
                "nu_ql_global": float(nu_g),
                "uuu_gap_pdf": uuu_gap_metric(m_rrr["l2_pdf"], m_ql["l2_pdf"]),
            }
        )

    return rows


def pick_keep_pareto(rows: List[Dict[str, Any]], pdf_tol: float = 0.05) -> int:
    """Smallest keep with RRR-only PDF L² error ≤ pdf_tol * first-row error."""
    if not rows:
        return 1
    base = max(rows[0]["l2_pdf_rrr"], 1e-9)
    thr = pdf_tol * base
    for r in rows:
        if r["l2_pdf_rrr"] <= thr:
            return int(r["keep"])
    return int(rows[-1]["keep"])


def save_truncation_metadata(
    out_dir: str,
    case: str,
    rows: List[Dict[str, Any]],
    chosen_keep: int,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "case": case,
        "sweep": rows,
        "chosen_keep": chosen_keep,
        "extra": extra or {},
    }
    path = os.path.join(out_dir, "mz_truncation_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
