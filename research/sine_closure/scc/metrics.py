"""Scoring metrics (PLAN.md section 8, API_CONTRACT.md 'scc/metrics.py').

All quantities are computed from ``ctx.T_rows`` (physical maturities), never from ``grid.T_out``
directly, so the early-time panel can reuse everything with its own maturities.  No clipping,
renormalisation or smoothing anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from dsm import metrics as _dsm_metrics
from dsm import operators as op
from dsm import repo

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ----------------------------------------------------------------------------- context
@dataclass
class EvalContext:
    grid: op.Grid
    D2_repo: np.ndarray
    D2_lin: np.ndarray
    interior: np.ndarray   # (n_t, n) bool, repository 5-95 % strike band
    atm: np.ndarray        # (n_t, n) bool, |K/S0 - 1| <= 0.10
    wing: np.ndarray       # (n_t, n) bool, K/S0 <= 0.7 or >= 1.5
    K_rows: np.ndarray     # (n_t, n)
    T_rows: np.ndarray     # (n_t,)


def make_context(grid: op.Grid, T_rows: Optional[np.ndarray] = None) -> EvalContext:
    """Build the evaluation context: masks from ``interior_k_mask(K_row(T))`` per row,
    ATM ``|K/S0-1|<=0.1``, wings ``K/S0<=0.7`` or ``>=1.5``."""
    T_rows = np.asarray(grid.T_out if T_rows is None else T_rows, float).reshape(-1)
    n = grid.n_k
    K_rows = np.array([grid.K_row(float(T)) for T in T_rows])
    interior = np.array([repo.interior_k_mask(K) for K in K_rows])
    x = K_rows / grid.S0
    atm = np.abs(x - 1.0) <= 0.10
    wing = (x <= 0.7) | (x >= 1.5)
    return EvalContext(grid=grid, D2_repo=op.d2_repo(n, grid.dk), D2_lin=op.d2_linearity(n, grid.dk),
                       interior=interior, atm=atm, wing=wing, K_rows=K_rows, T_rows=T_rows)


# ----------------------------------------------------------------------------- price
def _rms(err: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    e2 = np.asarray(err, float) ** 2
    if mask is not None:
        e2 = e2[mask]
    if e2.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(e2)))


def price_metrics(phi_rows, ref_rows, ctx: EvalContext, bs_rows=None) -> Dict:
    """RMS = sqrt(mean over masked entries of (phi - ref)^2); masks interior/full/atm/wing;
    price_physical_interior_rms = interior RMS * S0 (phi~ = C/S0)."""
    err = np.asarray(phi_rows, float) - np.asarray(ref_rows, float)
    out = {
        "price_interior_rms": _rms(err, ctx.interior),
        "price_full_rms": _rms(err),
        "price_max_abs": float(np.max(np.abs(err))),
        "price_atm_rms": _rms(err, ctx.atm),
        "price_wing_rms": _rms(err, ctx.wing),
        "price_per_maturity_rms": [_rms(err[i], ctx.interior[i]) for i in range(err.shape[0])],
    }
    out["price_physical_interior_rms"] = out["price_interior_rms"] * float(ctx.grid.S0)
    if bs_rows is not None:
        out["price_interior_rms_vs_bs"] = _rms(np.asarray(phi_rows, float) - np.asarray(bs_rows, float), ctx.interior)
    return out


def resolved_error(c_rows, c_ref_rows, V, interior) -> Dict:
    """e = (c - c_ref) V^T (n_t, n); interior RMS, full RMS, and RMS over all coefficients."""
    dc = np.asarray(c_rows, float) - np.asarray(c_ref_rows, float)
    e = dc @ np.asarray(V, float).T
    return {"resolved_interior_rms": _rms(e, interior), "resolved_full_rms": _rms(e),
            "resolved_coeff_rms": _rms(dc)}


# ----------------------------------------------------------------------------- density
def _fd_weights_d2(offsets: Sequence[int]) -> np.ndarray:
    """Weights w such that sum_j w_j f(x0 + o_j h) = h^2 f''(x0) is exact for polynomials of degree < len(offsets)."""
    o = np.asarray(offsets, float)
    m = len(o)
    A = np.vander(o, m, increasing=True).T          # A[p, j] = o_j^p
    rhs = np.zeros(m)
    rhs[2] = 2.0                                      # d^2/dx^2 of x^p at 0 is 2 for p=2
    return np.linalg.solve(A, rhs)


def d2_fd4_matrix(n: int, dk: float, D2_repo: Optional[np.ndarray] = None) -> np.ndarray:
    """Fourth-order second-derivative matrix: central (-1,16,-30,16,-1)/12 on points 2..n-3, six-point
    one-sided fourth-order rows at points 1 and n-2 (offsets -1..4 and -4..1), repo rows at 0 and n-1."""
    if D2_repo is None:
        D2_repo = op.d2_repo(n, dk)
    D = np.zeros((n, n))
    D[0] = D2_repo[0]
    D[-1] = D2_repo[-1]
    cen = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
    for i in range(2, n - 2):
        D[i, i - 2:i + 3] = cen
    w_fwd = _fd_weights_d2([-1, 0, 1, 2, 3, 4])
    D[1, 0:6] = w_fwd
    D[n - 2, n - 6:n] = w_fwd[::-1]
    D[1:-1] /= dk ** 2
    return D


def density_rows(phi_rows, ctx: EvalContext, method: str = "d2_repo") -> np.ndarray:
    """f = exp(rT) S0 d2k (exp(-rT)/K_max)^2 per row, d2k = d^2 phi/dk^2 by 'd2_repo' | 'd2_lin' | 'fd4'."""
    phi_rows = np.asarray(phi_rows, float)
    g = ctx.grid
    if method == "d2_repo":
        D = ctx.D2_repo
    elif method == "d2_lin":
        D = ctx.D2_lin
    elif method == "fd4":
        D = d2_fd4_matrix(g.n_k, g.dk, ctx.D2_repo)
    else:
        raise ValueError(f"unknown density method {method!r}")
    d2k = phi_rows @ D.T
    T = ctx.T_rows[:, None]
    return np.exp(g.r * T) * g.S0 * d2k * (np.exp(-g.r * T) / g.K_max) ** 2


def density_rows_spectral(c_rows, V_full, lam_V, ctx: EvalContext) -> np.ndarray:
    """Analytic sine second derivative: d2k = (c * lam_V) @ V_full^T, same prefactor as density_rows."""
    g = ctx.grid
    d2k = (np.asarray(c_rows, float) * np.asarray(lam_V, float)[None, :]) @ np.asarray(V_full, float).T
    T = ctx.T_rows[:, None]
    return np.exp(g.r * T) * g.S0 * d2k * (np.exp(-g.r * T) / g.K_max) ** 2


def density_metrics(f, f_ref, ctx: EvalContext, post_initial_from: int = 0) -> Dict:
    """Trapezoid-in-K density diagnostics (see PLAN.md section 8)."""
    f = np.asarray(f, float)
    f_ref = np.asarray(f_ref, float)
    n_t = f.shape[0]
    l2, l1, l2f, neg, sm, mom, tail = [], [], [], [], [], [], []
    for i in range(n_t):
        K = ctx.K_rows[i]
        m = ctx.interior[i]
        d = f[i] - f_ref[i]
        l2.append(float(np.sqrt(_trapz(d[m] ** 2, K[m]))))
        l1.append(float(_trapz(np.abs(d[m]), K[m])))
        l2f.append(float(np.sqrt(_trapz(d ** 2, K))))
        neg.append(float(_trapz(np.maximum(-f[i], 0.0), K)))
        sm.append(float(_trapz(f[i], K)))
    err = np.abs(f - f_ref)
    K = ctx.K_rows[-1]
    out_band = ~ctx.interior[-1]
    return {
        "density_l2_interior_mean": float(np.mean(l2)),
        "density_l2_interior_max": float(np.max(l2)),
        "density_l1_interior_mean": float(np.mean(l1)),
        "density_l2_full_mean": float(np.mean(l2f)),
        "density_max_abs_err": float(err.max()),
        "density_q99_abs_err": float(np.quantile(err, 0.99)),
        "negative_mass": neg,
        "negative_mass_max_post_initial": float(np.max(neg[post_initial_from:])),
        "negative_mass_final": neg[-1],
        "signed_mass": sm,
        "mass_final": sm[-1],
        "mass_ref_final": float(_trapz(f_ref[-1], K)),
        "first_moment_final": float(_trapz(K * f[-1], K)),
        "first_moment_ref_final": float(_trapz(K * f_ref[-1], K)),
        "tail_mass_outside_interior_final": float(_trapz(np.where(out_band, f[-1], 0.0), K)),
        "tail_mass_ref_outside_interior_final": float(_trapz(np.where(out_band, f_ref[-1], 0.0), K)),
    }


# ----------------------------------------------------------------------------- arbitrage
def arbitrage_metrics(phi_rows, phi_t_rows, ctx: EvalContext) -> Dict:
    """Violation fractions for positivity (phi<-g), monotonicity (phi_k>+g), butterfly (phi_kk<-g),
    calendar (phi_t<-g); g = 1e-12 (raw) or 1e-3 (repo) times the row max |quantity|."""
    phi_rows = np.asarray(phi_rows, float)
    phi_t_rows = np.asarray(phi_t_rows, float)
    dk = ctx.grid.dk
    pk = np.array([np.gradient(r, dk) for r in phi_rows])
    pkk = phi_rows @ ctx.D2_repo.T
    quant = {"positivity": phi_rows, "monotonicity": pk, "butterfly": pkk, "calendar": phi_t_rows}
    out = {}
    for check, q in quant.items():
        scale = np.max(np.abs(q), axis=1, keepdims=True)
        for rule, rel in (("raw", 1e-12), ("repo", 1e-3)):
            g = rel * scale
            viol = (q > g) if check == "monotonicity" else (q < -g)
            for region, mask in (("interior", ctx.interior), ("full", np.ones_like(ctx.interior))):
                fr = np.array([viol[i][mask[i]].mean() if mask[i].any() else np.nan for i in range(q.shape[0])])
                out[f"{check}_{rule}_{region}_frac_mean"] = float(np.nanmean(fr))
                out[f"{check}_{rule}_{region}_frac_max"] = float(np.nanmax(fr))
    return out


# ----------------------------------------------------------------------------- inverse volatility
def identifiable_mask(ref_rows, ctx: EvalContext, rel_tol: float = 1e-3) -> np.ndarray:
    """interior & (D2_repo ref > rel_tol * rowmax |D2_repo ref|)."""
    pkk = np.asarray(ref_rows, float) @ ctx.D2_repo.T
    thr = rel_tol * np.max(np.abs(pkk), axis=1, keepdims=True)
    return ctx.interior & (pkk > thr)


def inverse_vol_metrics(phi_rows, phi_t_rows, ctx: EvalContext, ident_mask, sigma_true_rows,
                        rel_tol_panel=(1e-2, 1e-3, 1e-4)) -> Dict:
    """sigma_hat^2 = 2 phi_t / (T_max k^2 phi_kk); undefined <=> phi_kk <= rel_tol*rowmax|phi_kk|;
    invalid <=> undefined or nonfinite or sigma_hat^2 <= 0."""
    phi_rows = np.asarray(phi_rows, float)
    phi_t_rows = np.asarray(phi_t_rows, float)
    sig_true = np.asarray(sigma_true_rows, float)
    ident = np.asarray(ident_mask, bool)
    g = ctx.grid
    pkk = phi_rows @ ctx.D2_repo.T
    with np.errstate(divide="ignore", invalid="ignore"):
        sig2 = 2.0 * phi_t_rows / (g.T_max * g.k[None, :] ** 2 * pkk)
    finite_pos = np.isfinite(sig2) & (sig2 > 0)
    rowmax = np.max(np.abs(pkk), axis=1, keepdims=True)
    out: Dict = {}
    for tol in rel_tol_panel:
        undefined = pkk <= tol * rowmax
        invalid = undefined | ~finite_pos
        inter = ctx.interior
        n_int = inter.sum()
        n_id = int(ident.sum())
        ok = ident & ~invalid
        if ok.any():
            rel = np.abs(np.sqrt(sig2[ok]) - sig_true[ok]) / sig_true[ok]
            med, rms = float(np.median(rel)), float(np.sqrt(np.mean(rel ** 2)))
        else:
            med = rms = float("nan")
        sub = {
            "undefined_frac_interior": float((undefined & inter).sum() / n_int),
            "negative_or_nonfinite_frac_interior": float((~undefined & ~finite_pos & inter).sum() / n_int),
            "invalid_frac_full": float(invalid.mean()),
            "coverage": float(ok.sum() / n_id) if n_id else float("nan"),
            "sigma_relerr_median": med,
            "sigma_relerr_rms": rms,
            "n_identifiable": n_id,
        }
        out[f"{tol:.0e}"] = sub
    if "1e-03" in out:
        for key in ("undefined_frac_interior", "negative_or_nonfinite_frac_interior", "invalid_frac_full",
                    "coverage", "sigma_relerr_median", "sigma_relerr_rms"):
            out[f"invvol_{key}"] = out["1e-03"][key]
    return out


def sigma_true_rows(case, grid: op.Grid, T_rows) -> np.ndarray:
    """sigma_fn(T, K_row(T)/S0) broadcast to the k grid, one row per T in T_rows."""
    return np.array([np.broadcast_to(np.asarray(case.sigma_fn(float(T), grid.K_row(float(T)) / grid.S0), float),
                                     grid.k.shape) for T in np.asarray(T_rows, float)])


def time_derivative_rows_fd(phi_fine, idx, h):
    """Central differences of the fine-grid solution (dsm.metrics.time_derivative_rows)."""
    return _dsm_metrics.time_derivative_rows(phi_fine, idx, h)


# ----------------------------------------------------------------------------- model time derivatives
def rhs_time_derivative_reduced(c_rows, Mhat_rows, A_list, V_full) -> np.ndarray:
    """phi_t rows = (A_i c_i + Mhat_i) @ V_full^T; A_list: one (m,m) per row or a single matrix."""
    c_rows = np.asarray(c_rows, float)
    Mhat_rows = np.asarray(Mhat_rows, float)
    if isinstance(A_list, np.ndarray) and A_list.ndim == 2:
        cdot = c_rows @ A_list.T + Mhat_rows
    else:
        cdot = np.array([np.asarray(A, float) @ c + M for A, c, M in zip(A_list, c_rows, Mhat_rows)])
    return cdot @ np.asarray(V_full, float).T


def rhs_time_derivative_full(phi_rows, L_list) -> np.ndarray:
    """phi_t rows = L_i @ phi_i (L_list: one (n,n) per row or a single matrix)."""
    phi_rows = np.asarray(phi_rows, float)
    if isinstance(L_list, np.ndarray) and L_list.ndim == 2:
        return phi_rows @ L_list.T
    return np.array([np.asarray(L, float) @ p for L, p in zip(L_list, phi_rows)])
