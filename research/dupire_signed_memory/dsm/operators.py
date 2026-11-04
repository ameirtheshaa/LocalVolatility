"""Grid, stencils, operators and projectors for the two operator variants.

OP-A (repository-actual): production one-sided stencil, Fourier projector on raw φ̃.
OP-B (boundary-consistent): linearity boundary rows (φ̃_kk = 0 at both ends), Fourier
projector on the detrended variable ψ = φ̃ - ℓ(k̃).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from dsm import repo


@dataclass
class Grid:
    k: np.ndarray          # uniform k̃ grid (n_k,)
    dk: float
    T_out: np.ndarray      # repository output maturities (n_t,)
    t_out: np.ndarray      # = T_out / T_max
    S0: float
    r: float
    T_max: float
    K_max: float
    K_min: float
    T_min: float

    @property
    def n_k(self) -> int:
        return len(self.k)

    def K_row(self, T: float) -> np.ndarray:
        """Physical strikes at maturity T (K = k̃ K_max e^{rT})."""
        return self.k * self.K_max * np.exp(self.r * T)

    def payoff(self) -> np.ndarray:
        return repo.payoff_phi_tilde(self.k, float(self.S0), float(self.K_max))


def build_grid(n_k: int = 256, n_t: int = 24, *, K_min: Optional[float] = None,
               K_max: Optional[float] = None) -> Grid:
    """Repository grid (``build_uniform_tk_grid``) with the production constants."""
    cfg = repo.DupirePipelineConfig()
    K_min = cfg.K_min if K_min is None else K_min
    K_max = cfg.K_max if K_max is None else K_max
    vc = repo.VolatilityConfig.custom(lambda t, x: np.ones_like(np.asarray(x, float)))
    g, _ = repo.build_uniform_tk_grid(S0=cfg.S0, r=cfg.r, T_max=cfg.T_max, K_max=K_max,
                                      K_min=K_min, T_min=cfg.T_min, n_t=n_t, n_k=n_k,
                                      vol_config=vc)
    return Grid(k=np.asarray(g.k_tilde, float), dk=float(g.dk), T_out=np.asarray(g.T, float),
                t_out=np.asarray(g.t_tilde, float), S0=float(cfg.S0), r=float(cfg.r),
                T_max=float(cfg.T_max), K_max=float(K_max), K_min=float(K_min),
                T_min=float(cfg.T_min))


# ----------------------------------------------------------------------------- stencils
def d2_repo(n: int, dk: float) -> np.ndarray:
    """Production stencil: central interior, one-sided (2,-5,4,-1)/dk² rows at both ends."""
    return repo.build_second_derivative_matrix(n, dk)


def d2_linearity(n: int, dk: float) -> np.ndarray:
    """Same interior stencil, boundary rows zeroed (linearity condition φ̃_kk = 0)."""
    D = repo.build_second_derivative_matrix(n, dk)
    D[0, :] = 0.0
    D[-1, :] = 0.0
    return D


def d2_dirichlet_interior(n: int, dk: float):
    """Interior (n-2)×(n-2) central stencil and the boundary-coupling columns for OP-C.

    Returns (D_int, D_bnd) with D_bnd shape (n-2, 2) acting on (φ_0, φ_{n-1}).
    """
    D = repo.build_second_derivative_matrix(n, dk)
    D_int = D[1:-1, 1:-1].copy()
    D_bnd = np.stack([D[1:-1, 0], D[1:-1, -1]], axis=1)
    return D_int, D_bnd


# ----------------------------------------------------------------------------- volatility
def eta_row(sigma_fn: Callable[[float, np.ndarray], np.ndarray], grid: Grid, T: float) -> np.ndarray:
    """η̃(T, k̃) = (T_max/2) σ(T, K/S0)² on the k̃ grid (repository convention)."""
    x = grid.K_row(T) / grid.S0
    sig = np.asarray(sigma_fn(float(T), x), dtype=float)
    sig = np.broadcast_to(sig, grid.k.shape)
    return repo.eta_tilde_from_sigma(sig, grid.T_max)


def build_L(eta: np.ndarray, grid: Grid, D2: np.ndarray) -> np.ndarray:
    """L = diag(η̃ k̃²) D2 (``build_L_operator``)."""
    return repo.build_L_operator(eta, grid.k, D2)


# ----------------------------------------------------------------------------- projectors
def fourier_basis(n: int, keep: int) -> np.ndarray:
    """Orthonormal real basis V (n × m), m = 2·keep+1, of range(P_R) for the repository mask.

    Columns: DC, cos(2π j i/n), sin(2π j i/n) for j = 1..keep, normalised. V Vᵀ equals the
    repository ``low_pass_projector_matrix(n, keep)`` (checked in tests).
    """
    if 2 * keep >= n:
        return np.eye(n)
    i = np.arange(n)
    cols = [np.ones(n) / np.sqrt(n)]
    for j in range(1, keep + 1):
        c = np.cos(2 * np.pi * j * i / n)
        s = np.sin(2 * np.pi * j * i / n)
        cols.append(c / np.linalg.norm(c))
        cols.append(s / np.linalg.norm(s))
    return np.stack(cols, axis=1)


def projector(V: np.ndarray) -> np.ndarray:
    return V @ V.T


def eigen_basis(L_sym_weight: np.ndarray, D2: np.ndarray, m: int):
    """Weighted eigenbasis projector (diagnostic only): slowest m eigenvectors of
    A = √D D2 √D with D = diag(η̃ k̃²). Returns (V_w, evals, sqrtD)."""
    sq = np.sqrt(L_sym_weight)
    A = sq[:, None] * D2 * sq[None, :]
    A = 0.5 * (A + A.T)
    ev, V = np.linalg.eigh(A)
    order = np.argsort(np.abs(ev))
    return V[:, order[:m]], ev[order], sq


# ----------------------------------------------------------------------------- trend
def linear_trend(k: np.ndarray, left: float, right: float) -> np.ndarray:
    return left + (right - left) * (k - k[0]) / (k[-1] - k[0])


def detrend_rows(phi: np.ndarray, k: np.ndarray) -> np.ndarray:
    """ψ = φ̃ - ℓ(k̃) row-wise with ℓ the linear interpolant of the two endpoint values."""
    phi = np.atleast_2d(phi)
    out = np.empty_like(phi)
    for i, row in enumerate(phi):
        out[i] = row - linear_trend(k, row[0], row[-1])
    return out
