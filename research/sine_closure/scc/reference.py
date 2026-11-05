"""Full semidiscrete references on nested grids: dense expm, sparse expm_multiply, banded Crank-Nicolson.

All solvers advance phi~ on the fine time grid ``fg`` with the OP-B operator
L(t) = diag(eta~(t) k~^2) D2, D2 = ``dsm.operators.d2_linearity`` (boundary rows zero, so the two endpoint
values are conserved), operator frozen at the step midpoint (midpoint Magnus) when the case is time dependent.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from scipy.sparse.linalg import expm_multiply

from dsm import operators as dop
from dsm import reference as dref
from scc import identity

# Autonomous (time_dependent=False) cases: dsm.reference.operator_builder evaluates the coefficient row at
# T = T_max (scaled t = 1) once. For family S this matters, because x = K/S0 with K = k K_max e^{rT} makes
# the k-grid row depend on T; every solver here uses the same convention so that all agree with the dense one.
from scc.conventions import AUTONOMOUS_T  # single source of the autonomous-row convention

fine_grid = dref.fine_grid
bs_rows = dref.bs_truth


def make_grid(n: int) -> dop.Grid:
    """Repository grid with n strike points: dsm.operators.build_grid(n_k=n)."""
    return dop.build_grid(n_k=int(n))


def nested_index(n_fine: int, n_coarse: int) -> np.ndarray:
    """Indices of the coarse grid points inside the fine grid: arange(0, n_fine, (n_fine-1)//(n_coarse-1)).
    Requires (n_fine-1) % (n_coarse-1) == 0."""
    assert (n_fine - 1) % (n_coarse - 1) == 0, f"grids {n_fine} and {n_coarse} are not nested"
    step = (n_fine - 1) // (n_coarse - 1)
    return np.arange(0, n_fine, step)


def nested_index_grids(grid_fine: dop.Grid, grid_coarse: dop.Grid) -> np.ndarray:
    """nested_index plus the check that the fine k values at those indices equal the coarse k to 1e-12."""
    idx = nested_index(grid_fine.n_k, grid_coarse.n_k)
    err = float(np.max(np.abs(grid_fine.k[idx] - grid_coarse.k)))
    assert err < 1e-12, f"nested k grids differ by {err:.3e}"
    return idx


def restrict_rows(rows: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """rows[:, idx] (also accepts a 1-D array)."""
    rows = np.asarray(rows)
    return rows[..., idx]


# ----------------------------------------------------------------------------- operators
def _row_time(case, t: float) -> float:
    """Scaled time at which the coefficient row is evaluated: AUTONOMOUS_T for time-independent cases
    (the dsm convention, see PLAN.md Addendum 1), the requested t otherwise."""
    return AUTONOMOUS_T if not bool(getattr(case, "time_dependent", True)) else float(t)


def coefficient_full(case, grid: dop.Grid, t: float) -> np.ndarray:
    """d(t) = eta~(t*T_max, k~) k~^2 on the full grid (endpoints included), t scaled time
    (time-independent cases: the row at AUTONOMOUS_T regardless of t)."""
    return dop.eta_row(case.sigma_fn, grid, _row_time(case, t) * grid.T_max) * grid.k ** 2


def L_full_dense(case, grid: dop.Grid, t: float) -> np.ndarray:
    """OP-B: diag(eta~ k~^2) @ d2_linearity(n, dk), dense (n,n)."""
    eta = dop.eta_row(case.sigma_fn, grid, _row_time(case, t) * grid.T_max)
    return dop.build_L(eta, grid, dop.d2_linearity(grid.n_k, grid.dk))


def L_full_sparse(case, grid: dop.Grid, t: float) -> sp.csr_matrix:
    """Same matrix as L_full_dense as CSR: interior rows d_i (1,-2,1)/dk^2, boundary rows empty."""
    n = grid.n_k
    c = coefficient_full(case, grid, t)[1:-1] / grid.dk ** 2
    i = np.arange(1, n - 1)
    rows = np.concatenate([i, i, i])
    cols = np.concatenate([i - 1, i, i + 1])
    vals = np.concatenate([c, -2.0 * c, c])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ----------------------------------------------------------------------------- solvers
def solve_dense(case, grid: dop.Grid, fg, phi0: Optional[np.ndarray] = None) -> np.ndarray:
    """Dense midpoint-Magnus expm stepping == dsm.reference.solve_full(case, grid, 'B', fg, phi0)."""
    return dref.solve_full(case, grid, "B", fg, phi0)


def solve_sparse(case, grid: dop.Grid, fg, phi0: Optional[np.ndarray] = None) -> np.ndarray:
    """Same stepping with scipy.sparse.linalg.expm_multiply.

    Time-independent (operator evaluated at AUTONOMOUS_T = 1, the dsm convention): ONE call expm_multiply(L, phi0, start=0, stop=t[-1], num=N+1, endpoint=True).
    Time-dependent: one call per step with h*L(t_mid). scipy's expm_multiply has no tolerance argument
    (it targets full double precision), so no ``tol`` parameter is offered (contract signature minus tol).
    """
    phi = grid.payoff() if phi0 is None else np.asarray(phi0, float).copy()
    N = len(fg.t) - 1
    if not case.time_dependent:
        L = L_full_sparse(case, grid, AUTONOMOUS_T)
        out = expm_multiply(L, phi, start=0.0, stop=float(fg.t[-1]), num=N + 1, endpoint=True)
        out = np.asarray(out)
        out[0] = phi
        return out
    out = np.empty((N + 1, grid.n_k))
    out[0] = phi
    for s in range(N):
        tm = 0.5 * (fg.t[s] + fg.t[s + 1])
        phi = expm_multiply(fg.h * L_full_sparse(case, grid, tm), phi)
        out[s + 1] = phi
    return out


def _cn_step(phi, c, dt, theta, dk2inv):
    """One theta-step (theta=1/2 CN, 1 backward Euler) with interior coefficient vector c = d/dk^2
    (zero at the two endpoints, whose values therefore stay fixed)."""
    n = len(phi)
    ab = np.zeros((3, n))
    ab[1] = 1.0 + 2.0 * theta * dt * c
    ab[0, 1:] = -theta * dt * c[:-1]       # upper diagonal, row i col i+1
    ab[2, :-1] = -theta * dt * c[1:]       # lower diagonal, row i col i-1
    rhs = phi.copy()
    if theta < 1.0:
        lap = np.zeros(n)
        lap[1:-1] = phi[:-2] - 2.0 * phi[1:-1] + phi[2:]
        rhs = rhs + (1.0 - theta) * dt * c * lap
    return solve_banded((1, 1), ab, rhs)


def solve_banded_cn(case, grid: dop.Grid, fg, substeps: int, rannacher: int = 2,
                    phi0: Optional[np.ndarray] = None) -> np.ndarray:
    """Crank-Nicolson, ``substeps`` substeps per fine step, operator frozen at the substep midpoint time.
    The first ``rannacher`` substeps of the whole run are each replaced by two backward-Euler half-substeps.
    Tridiagonal solves via scipy.linalg.solve_banded; boundary rows are identity. Returns values at fg.t."""
    phi = grid.payoff() if phi0 is None else np.asarray(phi0, float).copy()
    N = len(fg.t) - 1
    dt = fg.h / substeps
    dk2inv = 1.0 / grid.dk ** 2
    out = np.empty((N + 1, grid.n_k))
    out[0] = phi
    c_const = None
    if not case.time_dependent:
        c_const = np.zeros(grid.n_k)
        c_const[1:-1] = coefficient_full(case, grid, AUTONOMOUS_T)[1:-1] * dk2inv
    count = 0
    for s in range(N):
        for j in range(substeps):
            if c_const is not None:
                c = c_const
            else:
                tm = fg.t[s] + (j + 0.5) * dt
                c = np.zeros(grid.n_k)
                c[1:-1] = coefficient_full(case, grid, tm)[1:-1] * dk2inv
            if count < rannacher:
                phi = _cn_step(phi, c, 0.5 * dt, 1.0, dk2inv)
                phi = _cn_step(phi, c, 0.5 * dt, 1.0, dk2inv)
            else:
                phi = _cn_step(phi, c, dt, 0.5, dk2inv)
            count += 1
        out[s + 1] = phi
    return out


def solver_check(case, grid: dop.Grid, fg) -> dict:
    """max |dense - sparse| and max |phi|; also |dense - CN(8 substeps)|."""
    a = solve_dense(case, grid, fg)
    b = solve_sparse(case, grid, fg)
    cn = solve_banded_cn(case, grid, fg, 8)
    return {"max_abs_diff_dense_sparse": float(np.max(np.abs(a - b))),
            "max_abs_phi": float(np.max(np.abs(a))),
            "max_abs_diff_dense_cn_substeps8": float(np.max(np.abs(a - cn)))}


# ----------------------------------------------------------------------------- cached references
def reference_identity(case, n: int, sub: int, solver: str, phi0_tag: str = "payoff", **extra) -> dict:
    """Identity of one reference solve (the grid domain is the production K_min/K_max)."""
    g = make_grid(n)
    return identity.make_identity(kind="reference", case=case.config_hash(), case_id=case.id, split=case.split,
                                  n=int(n), sub=int(sub), solver=solver, K_min=g.K_min, K_max=g.K_max,
                                  phi0=phi0_tag, **extra)


def _run_solver(solver, case, grid, fg, phi0):
    if solver == "dense":
        return solve_dense(case, grid, fg, phi0)
    if solver == "sparse":
        return solve_sparse(case, grid, fg, phi0)
    if solver.startswith("cn"):
        return solve_banded_cn(case, grid, fg, int(solver[2:]), phi0=phi0)
    raise ValueError(f"unknown solver {solver!r} (dense | sparse | cn<substeps>)")


def get_reference(case, n: int, sub: int, solver: str = "dense", phi0: Optional[np.ndarray] = None,
                  phi0_tag: str = "payoff", force: bool = False) -> dict:
    """Cached reference solve (kind 'reference'). Returns dict(phi, t, h, i0, out_idx, identity, from_cache, seconds).
    Calls identity.assert_phase_allows(case.split) first; phi0 given => phi0_tag must be given explicitly."""
    identity.assert_phase_allows(case.split)
    if phi0 is not None and phi0_tag == "payoff":
        raise ValueError("phi0 given: pass a descriptive phi0_tag (it is part of the cache identity)")
    t0 = time.perf_counter()
    ident = reference_identity(case, n, sub, solver, phi0_tag)
    if not force:
        z = identity.cache_load("reference", ident)
        if z is not None and {"phi", "t", "h", "i0", "out_idx"} <= set(z):
            return {"phi": z["phi"], "t": z["t"], "h": float(z["h"]), "i0": int(z["i0"]),
                    "out_idx": z["out_idx"], "identity": ident, "from_cache": True,
                    "seconds": time.perf_counter() - t0}
    grid = make_grid(n)
    fg = fine_grid(grid, sub)
    phi = _run_solver(solver, case, grid, fg, phi0)
    identity.cache_store("reference", ident, phi=phi, t=fg.t, h=np.array(fg.h), i0=np.array(fg.i0),
                         out_idx=fg.out_idx)
    return {"phi": phi, "t": fg.t, "h": float(fg.h), "i0": int(fg.i0), "out_idx": fg.out_idx,
            "identity": ident, "from_cache": False, "seconds": time.perf_counter() - t0}
