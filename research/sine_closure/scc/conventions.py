"""Boundary-adapted sine (DST-I) basis, interior operator system and block conventions.

Everything here is pure NumPy/SciPy.  The state is psi = phi - ell (ell = affine interpolant of the
endpoint values); only interior rows 1..n-2 are dynamic.  See docs/API_CONTRACT.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm

import scc  # noqa: F401  (puts dsm on sys.path)
from dsm import operators as _op
from dsm.mz import phi1_step as _dsm_phi1_step  # noqa: F401  (reference semantics)

# Scaled time at which a time-independent case's coefficient row is evaluated. The earlier study's
# ``dsm.reference.operator_builder`` uses T = T_max (t = 1); the strike-smile family depends on T
# through the forward factor in x = K/S0, so this choice is part of the operator definition and must
# be shared by the reference solvers (scc.reference imports it from here).
AUTONOMOUS_T = 1.0


# ----------------------------------------------------------------------------- basis
def _sine_columns(n: int, j: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / (n - 1)) * np.sin(np.pi * np.outer(rows, j) / (n - 1))


def dst_basis(n: int, m: int) -> np.ndarray:
    """V[i, j-1] = sqrt(2/(n-1)) sin(pi j i/(n-1)), i=0..n-1, j=1..m; rows 0 and n-1 exactly 0."""
    if not (1 <= m <= n - 2):
        raise ValueError(f"need 1 <= m <= n-2, got n={n}, m={m}")
    V = _sine_columns(n, np.arange(1, m + 1), np.arange(n))
    V[0, :] = 0.0
    V[-1, :] = 0.0
    return np.ascontiguousarray(V)


def dst_interior(n: int, m: int) -> np.ndarray:
    """Interior rows of dst_basis, shape (n-2, m)."""
    return np.ascontiguousarray(dst_basis(n, m)[1:-1])


def dst_complement_interior(n: int, m: int) -> np.ndarray:
    """DST columns j = m+1..n-2 on the interior rows, shape (n-2, n-2-m)."""
    if not (1 <= m <= n - 2):
        raise ValueError(f"need 1 <= m <= n-2, got n={n}, m={m}")
    return np.ascontiguousarray(_sine_columns(n, np.arange(m + 1, n - 1), np.arange(1, n - 1)))


def sine_eigenvalues(n: int, dk: float, j) -> np.ndarray:
    """lam_j = -(4/dk^2) sin^2(pi j / (2 (n-1))): eigenvalues of the interior (1,-2,1)/dk^2 block."""
    j = np.asarray(j, dtype=float)
    return -(4.0 / dk ** 2) * np.sin(np.pi * j / (2.0 * (n - 1))) ** 2


# ----------------------------------------------------------------------------- lift
def lift_rows(phi_rows: np.ndarray, k: np.ndarray) -> np.ndarray:
    """ell = affine interpolant of each row's endpoint values on k (1-D or 2-D input)."""
    phi = np.asarray(phi_rows, float)
    one = phi.ndim == 1
    P = np.atleast_2d(phi)
    s = (k - k[0]) / (k[-1] - k[0])
    out = P[:, :1] + (P[:, -1:] - P[:, :1]) * s[None, :]
    return out[0] if one else out


def detrend_rows(phi_rows: np.ndarray, k: np.ndarray) -> np.ndarray:
    """psi = phi - lift_rows(phi) (1-D or 2-D)."""
    phi = np.asarray(phi_rows, float)
    return phi - lift_rows(phi, k)


def payoff_state(grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(phi0, ell, psi0) with phi0 = grid.payoff()."""
    phi0 = np.asarray(grid.payoff(), float)
    ell = lift_rows(phi0, grid.k)
    return phi0, ell, phi0 - ell


# ----------------------------------------------------------------------------- interior system
@dataclass
class InteriorSystem:
    """Interior operator L_int(t) = diag(d(t)) D2_int with d = eta_tilde k^2 (OP-B, interior rows)."""
    grid: object
    case: object
    n: int
    k_int: np.ndarray
    D2_int: np.ndarray
    D2_bnd: np.ndarray
    time_dependent: bool
    _cache: Optional[np.ndarray] = None

    def d_full(self, t: float) -> np.ndarray:
        """eta_tilde(t T_max, k) * k^2 on the full grid (incl. endpoints)."""
        eta = _op.eta_row(self.case.sigma_fn, self.grid, float(t) * self.grid.T_max)
        return eta * self.grid.k ** 2

    def d(self, t: float) -> np.ndarray:
        """Interior coefficient d(t) = eta_row(...)[1:-1] * k_int^2 (time-independent cases: the row at AUTONOMOUS_T)."""
        if not self.time_dependent:
            if self._cache is None:
                self._cache = self.d_full(AUTONOMOUS_T)[1:-1].copy()
            return self._cache
        return self.d_full(t)[1:-1]

    def L_int(self, t: float) -> np.ndarray:
        return self.d(t)[:, None] * self.D2_int


def interior_system(case, grid) -> InteriorSystem:
    n = grid.n_k
    D2 = _op.d2_repo(n, grid.dk)
    return InteriorSystem(grid=grid, case=case, n=n, k_int=np.asarray(grid.k[1:-1], float),
                          D2_int=np.ascontiguousarray(D2[1:-1, 1:-1]),
                          D2_bnd=np.ascontiguousarray(D2[1:-1][:, [0, n - 1]]),
                          time_dependent=bool(case.time_dependent))


def blocks(sys: InteriorSystem, t: float, V_int: np.ndarray, U_int: np.ndarray) -> Dict[str, np.ndarray]:
    """Dense blocks of W^T L_int(t) W, W = [V U]: A=V^T L V, B=V^T L U, C=U^T L V, D=U^T L U."""
    L = sys.L_int(t)
    LV = L @ V_int
    LU = L @ U_int
    return {"A": V_int.T @ LV, "B": V_int.T @ LU, "C": U_int.T @ LV, "D": U_int.T @ LU}


def S_matrix(sys: InteriorSystem, t: float, V_int: np.ndarray) -> np.ndarray:
    """S = V^T diag(d) V (symmetric positive definite)."""
    return V_int.T @ (sys.d(t)[:, None] * V_int)


def A_structured(sys: InteriorSystem, t: float, V_int: np.ndarray, lam_V: np.ndarray) -> np.ndarray:
    """A = S diag(lam): (V^T diag(d) V) * lam[None, :] since D2_int V = V diag(lam)."""
    return S_matrix(sys, t, V_int) * np.asarray(lam_V)[None, :]


def forcing_g(sys: InteriorSystem, t: float, ell: np.ndarray, ell_dot: Optional[np.ndarray] = None,
              b_int: Optional[np.ndarray] = None) -> np.ndarray:
    """g = d (D2_int ell_int + D2_bnd ell_(0,n-1)) + b_int - ell_dot_int (absent terms are zero)."""
    ell = np.asarray(ell, float)
    g = sys.d(t) * (sys.D2_int @ ell[1:-1] + sys.D2_bnd @ ell[[0, -1]])
    if b_int is not None:
        g = g + np.asarray(b_int, float)
    if ell_dot is not None:
        g = g - np.asarray(ell_dot, float)[1:-1]
    return g


def memory_target(sys: InteriorSystem, t: float, V_int: np.ndarray, U_int: np.ndarray,
                  psi_int_rows: np.ndarray) -> np.ndarray:
    """M = V^T L_int (I - V V^T) psi, row-wise; w = psi - (psi V) V^T, Lw = d * (D2_int w), M = Lw V.

    Equals B(t) (psi U) for every row (U spans the complement of V in the interior).
    """
    psi = np.atleast_2d(np.asarray(psi_int_rows, float))
    w = psi - (psi @ V_int) @ V_int.T
    Lw = (sys.D2_int @ w.T).T * sys.d(t)[None, :]
    return Lw @ V_int


def resolved_blocks_timewise(sys: InteriorSystem, V_int: np.ndarray, U_int: np.ndarray,
                             times) -> List[Dict[str, np.ndarray]]:
    """blocks() at each time (a single entry when the system is time-independent)."""
    times = list(np.atleast_1d(times))
    if not sys.time_dependent:
        return [blocks(sys, times[0] if times else 0.0, V_int, U_int)]
    return [blocks(sys, float(t), V_int, U_int) for t in times]


# ----------------------------------------------------------------------------- exponentials
def expm_h(M: np.ndarray, h: float) -> np.ndarray:
    """exp(h M)."""
    return expm(h * np.asarray(M, float))


def phi1_step(M: np.ndarray, h: float, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(e^{hM}, h phi1(hM) g) = (e^{hM}, int_0^h e^{(h-s)M} g ds) via expm([[hM, g],[0, 0]])."""
    p = M.shape[0]
    Mx = np.zeros((p + 1, p + 1))
    Mx[:p, :p] = h * M
    Mx[:p, p] = g
    E = expm(Mx)
    return E[:p, :p], h * E[:p, p]
