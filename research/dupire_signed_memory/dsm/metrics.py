"""Scoring metrics in the source's own coordinates (no clipping, no discarding)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from dsm import operators as op
from dsm import repo

_trapz = getattr(np, "trapezoid", None) or np.trapz


def interior_masks(grid: op.Grid) -> np.ndarray:
    """(n_t, n_k) boolean interior band (5–95 % strike quantiles, ``interior_k_mask``)."""
    return np.array([repo.interior_k_mask(grid.K_row(T)) for T in grid.T_out])


def rms(err: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    err = np.asarray(err, float)
    if mask is None:
        return float(np.sqrt(np.mean(err ** 2)))
    return float(np.sqrt(np.mean((err ** 2)[mask])))


def density_rows(phi_rows: np.ndarray, grid: op.Grid, D2: np.ndarray) -> np.ndarray:
    """Breeden–Litzenberger f(K) = e^{rT} S0 ∂²φ̃/∂K², ∂²/∂K² = e^{-2rT}/K_max² ∂²/∂k̃²
    (the repository's ``pdf_from_phi_tilde`` formula without its max(f,0) clip and renormalisation)."""
    out = np.empty_like(phi_rows)
    for i, T in enumerate(grid.T_out):
        d2k = D2 @ phi_rows[i]
        out[i] = np.exp(grid.r * T) * grid.S0 * d2k * (np.exp(-grid.r * T) / grid.K_max) ** 2
    return out


def density_metrics(f: np.ndarray, f_ref: np.ndarray, grid: op.Grid, interior: np.ndarray) -> Dict[str, float]:
    l2, negmass, mass = [], [], []
    for i, T in enumerate(grid.T_out):
        K = grid.K_row(T)
        m = interior[i]
        l2.append(float(np.sqrt(_trapz((f[i][m] - f_ref[i][m]) ** 2, K[m]))))
        negmass.append(float(_trapz(np.maximum(-f[i], 0.0), K)))
        mass.append(float(_trapz(f[i], K)))
    return {"density_l2_interior_mean": float(np.mean(l2)), "density_l2_interior_max": float(np.max(l2)),
            "negative_mass_max": float(np.max(negmass)), "negative_mass_T_max": negmass[-1],
            "mass_T_max": mass[-1]}


def time_derivative_rows(phi_fine: np.ndarray, out_idx: np.ndarray, h: float) -> np.ndarray:
    """∂φ̃/∂t̃ at the output rows by central differences on the fine grid (one-sided at the ends)."""
    rows = []
    N = phi_fine.shape[0]
    for i in out_idx:
        if 0 < i < N - 1:
            rows.append((phi_fine[i + 1] - phi_fine[i - 1]) / (2 * h))
        elif i == 0:
            rows.append((phi_fine[1] - phi_fine[0]) / h)
        else:
            rows.append((phi_fine[-1] - phi_fine[-2]) / h)
    return np.array(rows)


def validity_metrics(phi_rows: np.ndarray, phi_t_rows: np.ndarray, grid: op.Grid, D2: np.ndarray,
                     interior: np.ndarray, sigma_true_rows: Optional[np.ndarray] = None,
                     rel_tol: float = 1e-3) -> Dict[str, float]:
    """Monotonicity, butterfly (φ̃_kk ≥ 0), calendar (φ̃_t ≥ 0) and inverse-volatility validity,
    fractions over the interior band; violations are counted below -rel_tol·max|·| per row so
    round-off never counts."""
    n_t, n_k = phi_rows.shape
    but, cal, mono, inv_invalid, sig_relerr = [], [], [], [], []
    for i in range(n_t):
        m = interior[i]
        pk = np.gradient(phi_rows[i], grid.dk)
        pkk = D2 @ phi_rows[i]
        pt = phi_t_rows[i]
        tol_kk = rel_tol * np.abs(pkk[m]).max()
        tol_t = rel_tol * np.abs(pt[m]).max()
        but.append(float(np.mean(pkk[m] < -tol_kk)))
        cal.append(float(np.mean(pt[m] < -tol_t)))
        mono.append(float(np.mean(pk[m] > rel_tol * np.abs(pk[m]).max())))
        denom = grid.T_max * grid.k ** 2 * pkk
        with np.errstate(divide="ignore", invalid="ignore"):
            sig2 = 2.0 * pt / denom
        undefined = pkk <= tol_kk
        invalid = undefined | ~np.isfinite(sig2) | (sig2 <= 0)
        inv_invalid.append(float(np.mean(invalid[m])))
        if sigma_true_rows is not None:
            ok = m & ~invalid
            if ok.any():
                sig_relerr.append(float(np.median(np.abs(np.sqrt(sig2[ok]) - sigma_true_rows[i][ok]) / sigma_true_rows[i][ok])))
            else:
                sig_relerr.append(float("nan"))
    out = {"butterfly_violation_frac_mean": float(np.mean(but)), "butterfly_violation_frac_max": float(np.max(but)),
           "calendar_violation_frac_mean": float(np.mean(cal)), "calendar_violation_frac_max": float(np.max(cal)),
           "monotonicity_violation_frac_mean": float(np.mean(mono)),
           "inverse_vol_invalid_frac_mean": float(np.mean(inv_invalid)), "inverse_vol_invalid_frac_max": float(np.max(inv_invalid)),
           "min_phi_rows": float(phi_rows.min()), "max_phi_rows": float(phi_rows.max())}
    if sigma_true_rows is not None:
        out["inverse_vol_median_relerr_mean"] = float(np.nanmean(sig_relerr)) if len(sig_relerr) else float("nan")
    return out


def sigma_true_rows(case, grid: op.Grid) -> np.ndarray:
    return np.array([np.broadcast_to(np.asarray(case.sigma_fn(float(T), grid.K_row(T) / grid.S0), float), grid.k.shape)
                     for T in grid.T_out])
