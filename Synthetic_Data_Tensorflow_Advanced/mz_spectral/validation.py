#!/usr/bin/env python3
"""
Metrics, PDF extraction, time integration (RRR / RRR+QL), and optional NN comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

_trapz = getattr(np, "trapezoid", np.trapz)


def payoff_phi_tilde(k_tilde: np.ndarray, S0: float, K_max: float) -> np.ndarray:
    """
    Normalized call payoff at t̃ = 0 on the k̃ grid: φ̃(0, k̃) = max(0, 1 − K_max·k̃/S₀).

    Matches ``loss_dupire_cal`` / ``loss_phi_cal`` terminal shape (ReLU digital in k̃).
    """
    k_tilde = np.asarray(k_tilde, dtype=float)
    return np.maximum(0.0, 1.0 - (K_max / S0) * k_tilde)


def interior_k_mask(K_row: np.ndarray, qlo: float = 0.05, qhi: float = 0.95) -> np.ndarray:
    """Boolean mask over strike index for interior band (quantiles of K)."""
    K_row = np.asarray(K_row, dtype=float)
    lo, hi = np.quantile(K_row, [qlo, qhi])
    return (K_row >= lo) & (K_row <= hi)


from mz_spectral.fourier_dupire import (
    TKGrid,
    build_L_operator,
    build_second_derivative_matrix,
    build_uniform_tk_grid,
    eta_tilde_from_sigma,
)
from mz_spectral.mz_decomposition import apply_projector, low_pass_projector_matrix, split_operator_blocks
from mz_spectral.quasi_linear import fit_nu_ql


def d2_wrt_K_from_k(phi_row: np.ndarray, T: float, r: float, K_max: float, D2: np.ndarray) -> np.ndarray:
    """∂²φ̃/∂K² from samples on uniform k̃, using ∂²φ/∂k² (D2 @ phi) and dk/dK = exp(-rT)/K_max."""
    dk2_phi = D2 @ phi_row
    dk_dK = np.exp(-r * T) / K_max
    return dk2_phi * (dk_dK**2)


def pdf_from_phi_tilde(
    phi_row: np.ndarray,
    T: float,
    r: float,
    S0: float,
    K_max: float,
    K_row: np.ndarray,
    D2: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Risk-neutral density f(K) from scaled call price φ̃ = C/S₀ on a k̃ row.

    f(K) = exp(rT) · ∂²C/∂K² = exp(rT) · S₀ · ∂²φ̃/∂K²
    """
    d2K = d2_wrt_K_from_k(phi_row, T, r, K_max, D2)
    f = np.exp(r * T) * S0 * d2K
    f = np.maximum(f, 0.0)
    if normalize and len(K_row) > 1:
        area = _trapz(f, K_row)
        if area > 0:
            f /= area
    return f


def spectral_pdf_filter(phi_row: np.ndarray, keep: int) -> np.ndarray:
    """Low-pass φ in k̃ (Hermitian FFT) before PDF differentiation reduces UUU noise."""
    n = len(phi_row)
    mask = np.zeros(n)
    if keep <= 0:
        mask[0] = 1.0
    elif 2 * keep >= n:
        mask[:] = 1.0
    else:
        mask[: keep + 1] = 1.0
        mask[n - keep :] = 1.0
    spec = np.fft.fft(phi_row, norm="ortho")
    return np.fft.ifft(spec * mask, norm="ortho").real


def pdf_metrics(
    f: np.ndarray,
    f_ref: np.ndarray,
    K: np.ndarray,
    *,
    tail_q: float = 0.9,
    k_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    """L² error, tail mass ratio, excess kurtosis difference."""
    f = np.asarray(f, dtype=float)
    f_ref = np.asarray(f_ref, dtype=float)
    K = np.asarray(K, dtype=float)
    if k_mask is not None:
        km = np.asarray(k_mask, dtype=bool)
        f = f[km]
        f_ref = f_ref[km]
        K = K[km]
        if K.size < 2:
            return {
                "l2_pdf": float("nan"),
                "tail_mass_ratio": float("nan"),
                "excess_kurtosis_diff": float("nan"),
            }
    err = float(np.sqrt(_trapz((f - f_ref) ** 2, K)))
    thr = np.quantile(K, tail_q)
    tail = K >= thr
    m_t = float(_trapz(f[tail], K[tail]) + 1e-12)
    m_r = float(_trapz(f_ref[tail], K[tail]) + 1e-12)
    def _moments(d):
        m0 = _trapz(d, K)
        if m0 <= 0:
            return 0.0, 1.0, 0.0
        p = d / (m0 + 1e-16)
        ex = _trapz(K * p, K)
        ex2 = _trapz((K**2) * p, K)
        var = max(ex2 - ex**2, 1e-16)
        Kc = K - ex
        mu4 = _trapz((Kc**4) * p, K)
        kurt = mu4 / (var**2) - 3.0
        return ex, var, kurt

    _, _, kf = _moments(f)
    _, _, kr = _moments(f_ref)
    return {
        "l2_pdf": err,
        "tail_mass_ratio": m_t / m_r,
        "excess_kurtosis_diff": float(kf - kr),
    }


def precompute_L_and_blocks(
    grid: TKGrid,
    sigma: np.ndarray,
    T_max: float,
    keep: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """D2, P_R, list of L[i], list of L_RR[i]."""
    n_k = len(grid.k_tilde)
    D2 = build_second_derivative_matrix(n_k, grid.dk)
    P_R = low_pass_projector_matrix(n_k, keep)
    eta = eta_tilde_from_sigma(sigma, T_max)
    L_rows = []
    L_RR_rows = []
    for i in range(len(grid.T)):
        L_i = build_L_operator(eta[i], grid.k_tilde, D2)
        L_rows.append(L_i)
        L_RR, _, _, _, _ = split_operator_blocks(L_i, P_R)
        L_RR_rows.append(L_RR)
    return D2, P_R, L_rows, L_RR_rows


def integrate_rrr_only(
    phi0: np.ndarray,
    t_tilde: np.ndarray,
    L_RR_rows: List[np.ndarray],
    P_R: np.ndarray,
) -> np.ndarray:
    """Backward Euler: (I - Δt̃ L_RR) φ^{n+1} = φ^n, φ^0 = P_R φ0 (stiff-stable)."""
    n_t = len(t_tilde)
    n_k = len(phi0)
    out = np.zeros((n_t, n_k))
    out[0] = P_R @ phi0
    I = np.eye(n_k)
    for i in range(n_t - 1):
        dt = t_tilde[i + 1] - t_tilde[i]
        A = I - dt * L_RR_rows[i]
        out[i + 1] = apply_projector(P_R, np.linalg.solve(A, out[i]))
    return out


def integrate_rrr_ql(
    phi0: np.ndarray,
    t_tilde: np.ndarray,
    L_RR_rows: List[np.ndarray],
    P_R: np.ndarray,
    D2: np.ndarray,
    nu: float,
) -> np.ndarray:
    """
    Implicit L_RR + explicit QL (IMEX): (I - Δt̃ L_RR) φ^{n+1} = φ^n + Δt̃ ν P_R D₂ φ^n.
    """
    n_t = len(t_tilde)
    n_k = len(phi0)
    out = np.zeros((n_t, n_k))
    out[0] = P_R @ phi0
    I = np.eye(n_k)
    for i in range(n_t - 1):
        dt = t_tilde[i + 1] - t_tilde[i]
        ql = nu * (P_R @ (D2 @ out[i]))
        rhs = out[i] + dt * ql
        out[i + 1] = apply_projector(P_R, np.linalg.solve(I - dt * L_RR_rows[i], rhs))
    return out


def phi_l2_error(a: np.ndarray, b: np.ndarray, k_vec: np.ndarray) -> float:
    """Mean over time of L² in k."""
    errs = []
    for i in range(a.shape[0]):
        errs.append(_trapz((a[i] - b[i]) ** 2, k_vec))
    return float(np.mean(np.sqrt(np.maximum(errs, 0.0))))


def compare_to_nn_pdf(
    model_dir: str,
    config: Any,
    T_list: List[float],
    K_grid: np.ndarray,
) -> Dict[float, np.ndarray]:
    """
    NN-implied PDFs using DupirePipeline.PDFAnalyzer (requires TensorFlow).

    Returns mapping T -> f_NN(K) normalized on K_grid.
    """
    from dupire_pipeline import PDFAnalyzer, load_trained_models

    nn_phi, nn_eta, meta = load_trained_models(model_dir)
    analyzer = PDFAnalyzer(nn_phi, nn_eta, config, meta)
    out = {}
    for T in T_list:
        out[float(T)] = analyzer.extract_model_implied_density(float(T), K_grid)
    return out


def build_truth_phi_bs(
    grid: TKGrid,
    S0: float,
    r: float,
    sigma_const: float,
) -> np.ndarray:
    """Black–Scholes φ̃ = C/S₀ on (T,K) grid (constant vol)."""
    from analytical_solutions import black_scholes_call

    phi = np.zeros_like(grid.K)
    for i, Ti in enumerate(grid.T):
        phi[i, :] = black_scholes_call(S0, grid.K[i, :], float(Ti), r, sigma_const) / S0
    return phi


def uuu_gap_metric(err_rrr: float, err_rrr_ql: float) -> float:
    """Δ_UUU style: improvement from QL (positive means QL helped vs RRR-only)."""
    return float(err_rrr - err_rrr_ql)
