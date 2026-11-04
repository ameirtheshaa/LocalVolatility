"""Full semidiscrete reference solutions (the 'truth' for the MZ decomposition).

Time grid: uniform step h on [0, 1] chosen so that the 24 repository output maturities
t_out ∈ [0.2, 1] lie on the grid: `sub` fine steps per output interval, h = 0.8/(23·sub).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy.linalg import expm

from dsm import operators as op
from dsm import repo
from dsm.cases import Case

CACHE = os.path.join(os.environ.get("DSM_OUTPUT_ROOT", repo.STUDY_ROOT), "results", "cache")


@dataclass
class FineGrid:
    t: np.ndarray        # (N+1,) fine times from 0 to 1
    h: float
    i0: int              # index of t̃0 = t_out[0]
    out_idx: np.ndarray  # indices of the 24 output maturities


def fine_grid(grid: op.Grid, sub: int = 16) -> FineGrid:
    if sub % 4 != 0:
        raise ValueError(f"sub must be a multiple of 4 so that t̃0 = 0.2 is a grid point (got {sub})")
    n_out = len(grid.t_out)
    h = (grid.t_out[-1] - grid.t_out[0]) / ((n_out - 1) * sub)
    n0 = int(round(grid.t_out[0] / h))
    h0 = grid.t_out[0] / n0 if n0 > 0 else h
    assert abs(h0 - h) < 1e-12 * max(1.0, 1 / h), "t̃0 must be a multiple of h"
    N = n0 + (n_out - 1) * sub
    t = np.arange(N + 1) * h
    out_idx = n0 + sub * np.arange(n_out)
    assert np.allclose(t[out_idx], grid.t_out)
    return FineGrid(t=t, h=float(h), i0=n0, out_idx=out_idx)


def operator_builder(case: Case, grid: op.Grid, variant: str) -> Callable[[float], np.ndarray]:
    """Returns L(t̃) for the variant ('A' production stencil, 'B'/'C' linearity rows)."""
    n = grid.n_k
    D2 = op.d2_repo(n, grid.dk) if variant == "A" else op.d2_linearity(n, grid.dk)
    if not case.time_dependent:
        eta = op.eta_row(case.sigma_fn, grid, grid.T_max)   # any T gives the same row
        L = op.build_L(eta, grid, D2)
        return lambda t: L
    cache: Dict[float, np.ndarray] = {}

    def L_of_t(t: float) -> np.ndarray:
        key = round(float(t), 12)
        if key not in cache:
            eta = op.eta_row(case.sigma_fn, grid, key * grid.T_max)
            cache[key] = op.build_L(eta, grid, D2)
        return cache[key]
    return L_of_t


def solve_full(case: Case, grid: op.Grid, variant: str, fg: FineGrid,
               phi0: Optional[np.ndarray] = None) -> np.ndarray:
    """φ̃ on the fine grid, exact `expm` steps (midpoint Magnus when L depends on t)."""
    L_of_t = operator_builder(case, grid, variant)
    phi = grid.payoff() if phi0 is None else np.asarray(phi0, float).copy()
    out = np.empty((len(fg.t), grid.n_k))
    out[0] = phi
    if not case.time_dependent:
        E = expm(fg.h * L_of_t(0.0))
        for n in range(len(fg.t) - 1):
            phi = E @ phi
            out[n + 1] = phi
    else:
        for n in range(len(fg.t) - 1):
            tm = 0.5 * (fg.t[n] + fg.t[n + 1])
            phi = expm(fg.h * L_of_t(tm)) @ phi
            out[n + 1] = phi
    return out


def bs_truth(grid: op.Grid, sigma: float, t: np.ndarray) -> np.ndarray:
    """Black–Scholes φ̃ = C/S0 at scaled times t (rows) on the k̃ grid."""
    out = np.zeros((len(t), grid.n_k))
    for i, ti in enumerate(t):
        T = ti * grid.T_max
        if T <= 0:
            out[i] = grid.payoff()
        else:
            out[i] = repo.black_scholes_call(grid.S0, grid.K_row(T), float(T), grid.r, sigma) / grid.S0
    return out


def state_variable(phi: np.ndarray, grid: op.Grid, variant: str) -> np.ndarray:
    """The variable the projector acts on: raw φ̃ (OP-A) or detrended ψ (OP-B/C)."""
    if variant == "A":
        return np.asarray(phi, float)
    return op.detrend_rows(phi, grid.k)


def trend_rows(phi: np.ndarray, grid: op.Grid) -> np.ndarray:
    phi = np.atleast_2d(phi)
    return np.stack([op.linear_trend(grid.k, row[0], row[-1]) for row in phi])


def reference_key(case: Case, grid: op.Grid, variant: str, fg: FineGrid) -> str:
    return f"ref_{case.id}_{variant}_nk{grid.n_k}_Kmin{grid.K_min:g}_Kmax{grid.K_max:g}_N{len(fg.t)-1}_{case.config_hash()}"


def get_reference(case: Case, grid: op.Grid, variant: str, fg: FineGrid, *, force: bool = False) -> np.ndarray:
    """Cached full solve on the fine grid (results/cache, regenerable, not committed)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, reference_key(case, grid, variant, fg) + ".npz")
    if os.path.isfile(path) and not force:
        with np.load(path) as z:
            if np.allclose(z["t"], fg.t):
                return z["phi"]
    phi = solve_full(case, grid, variant, fg)
    np.savez_compressed(path + ".tmp.npz", phi=phi, t=fg.t)
    os.replace(path + ".tmp.npz", path)
    return phi
