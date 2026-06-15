#!/usr/bin/env python3
"""
Semidiscrete Dupire dynamics in scaled (t̃, k̃) coordinates.

PDE (matches DupireNeuralModel.loss_dupire_cal):

    ∂φ̃/∂t̃ = η̃(t̃, k̃) · k̃² · ∂²φ̃/∂k̃²

with η̃ = (T_max/2) σ²(T, K), K = k̃ · K_max · exp(rT), T = t̃ · T_max.

Semidiscrete ODE on a uniform k̃ grid at each t̃:

    dφ/dt̃ = L(t̃) φ,   L = diag(η · k²) @ D₂

where D₂ is a consistent second-derivative matrix in k̃.

Fourier diagnostics: for periodic extension, k²∂ₖₖ would not diagonalize when k
varies; we verify the physical-space ODE and optionally report rFFT spectra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TKGrid:
    """Uniform (t̃, k̃) grid with physical (T, K) maps."""

    t_tilde: np.ndarray  # shape (n_t,)
    k_tilde: np.ndarray  # shape (n_k,) uniform
    T: np.ndarray  # shape (n_t,)
    K: np.ndarray  # shape (n_t, n_k) physical strikes
    dk: float


def build_uniform_tk_grid(
    *,
    S0: float,
    r: float,
    T_max: float,
    K_max: float,
    K_min: float,
    T_min: float,
    n_t: int,
    n_k: int,
    vol_config,
) -> Tuple[TKGrid, np.ndarray]:
    """
    Build uniform t̃ and k̃ grids; return (grid, sigma_grid).

    k̃ envelope covers [exp(-rT)K_min/K_max, exp(-rT)K_max/K_max] over T in [T_min, T_max].
    """
    T = np.linspace(T_min, T_max, n_t)
    t_tilde = T / T_max

    k_lo = min(np.exp(-r * t) * K_min / K_max for t in T)
    k_hi = max(np.exp(-r * t) * K_max / K_max for t in T)
    k_tilde = np.linspace(k_lo, k_hi, n_k)
    dk = (k_hi - k_lo) / (n_k - 1) if n_k > 1 else 1.0

    K = np.zeros((n_t, n_k))
    for i, Ti in enumerate(T):
        K[i, :] = k_tilde * K_max * np.exp(r * Ti)

    grid = TKGrid(
        t_tilde=t_tilde,
        k_tilde=k_tilde,
        T=T,
        K=K,
        dk=float(dk),
    )

    # sigma with correct S0 for moneyness x = K/S0
    sigma = np.zeros_like(K)
    for i in range(n_t):
        t = float(T[i])
        x = K[i, :] / S0
        if vol_config.model_type == "dupire_exact":
            y = np.sqrt(x + vol_config.x_shift) * (t + vol_config.t_shift)
            sigma[i, :] = vol_config.sigma_base + y * np.exp(-y)
        elif vol_config.model_type == "custom":
            sigma[i, :] = np.asarray(vol_config.custom_volatility_func(t, x), dtype=float)
        else:
            raise ValueError(f"Unsupported model_type={vol_config.model_type}")

    return grid, sigma


def eta_tilde_from_sigma(sigma: np.ndarray, T_max: float) -> np.ndarray:
    """η̃ = (T_max/2) σ²."""
    return (T_max / 2.0) * (sigma ** 2)


def build_second_derivative_matrix(n: int, dk: float) -> np.ndarray:
    """
    Central second derivative on interior; one-sided second order at boundaries.

    Returns n×n matrix D₂ with (D₂ φ)_i ≈ ∂²φ/∂k² at uniform spacing dk.
    """
    if n < 4:
        raise ValueError("Need at least n_k=4 for stable boundary stencils")
    D = np.zeros((n, n))
    inv_dk2 = 1.0 / (dk * dk)
    for i in range(1, n - 1):
        D[i, i - 1] = inv_dk2
        D[i, i] = -2.0 * inv_dk2
        D[i, i + 1] = inv_dk2
    # forward at 0: (2f0 - 5f1 + 4f2 - f3)/dk^2
    D[0, 0] = 2.0 * inv_dk2
    D[0, 1] = -5.0 * inv_dk2
    D[0, 2] = 4.0 * inv_dk2
    D[0, 3] = -1.0 * inv_dk2
    # backward at n-1
    j = n - 1
    D[j, j] = 2.0 * inv_dk2
    D[j, j - 1] = -5.0 * inv_dk2
    D[j, j - 2] = 4.0 * inv_dk2
    D[j, j - 3] = -1.0 * inv_dk2
    return D


def build_L_operator(
    eta_row: np.ndarray,
    k_row: np.ndarray,
    D2: np.ndarray,
) -> np.ndarray:
    """L = diag(η k²) @ D₂ for one maturity row (length n_k)."""
    coeff = eta_row * (k_row ** 2)
    return np.diag(coeff) @ D2


def apply_generator(phi: np.ndarray, L: np.ndarray) -> np.ndarray:
    """L @ phi."""
    return L @ phi


def verify_ode_residual(
    phi: np.ndarray,
    t_tilde: np.ndarray,
    L_rows: np.ndarray,
    *,
    relative: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Compare ∂φ/∂t̃ (FD in t) to L(t)φ using same k̃ columns.

    phi shape (n_t, n_k). L_rows shape (n_t, n_k, n_k).

    Returns (mean_relative_error, residuals_per_slice) where residuals are
    ‖φ̇ - Lφ‖₂ per interior time index.
    """
    n_t, n_k = phi.shape
    residuals = []
    errs = []
    for i in range(1, n_t - 1):
        dt = t_tilde[i + 1] - t_tilde[i - 1]
        dphi_dt = (phi[i + 1] - phi[i - 1]) / dt
        rhs = L_rows[i] @ phi[i]
        r = dphi_dt - rhs
        nrm = np.linalg.norm(r)
        residuals.append(nrm)
        denom = max(np.linalg.norm(dphi_dt), 1e-12)
        errs.append(nrm / denom if relative else nrm)
    mean_e = float(np.mean(errs)) if errs else 0.0
    return mean_e, np.asarray(residuals)


def rfft_energy_spectrum(phi_row: np.ndarray) -> np.ndarray:
    """|rFFT(φ)|² (unnormalized) for diagnostics."""
    return np.abs(np.fft.rfft(phi_row)) ** 2


def fourier_rhs_consistency(phi_row: np.ndarray, L_phi_row: np.ndarray) -> float:
    """
    Diagnostic: ‖rFFT(Lφ)‖ / ‖Lφ‖ (mode mixing proxy when L is not circulant).
    """
    _ = phi_row
    a = np.fft.rfft(L_phi_row)
    return float(np.linalg.norm(a) / (np.linalg.norm(L_phi_row) + 1e-12))
