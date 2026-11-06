"""Training targets along a full reference: resolved state c, unresolved z, memory target M = B z.

Pure NumPy.  See docs/API_CONTRACT.md ("scc/targets.py") and PLAN.md section 7.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

import scc  # noqa: F401
from scc import conventions as cv


def projected_state(psi_int_rows: np.ndarray, V_int: np.ndarray) -> np.ndarray:
    """c = psi_int V (rows)."""
    return np.atleast_2d(np.asarray(psi_int_rows, float)) @ V_int


def unresolved_state(psi_int_rows: np.ndarray, U_int: np.ndarray) -> np.ndarray:
    """z = psi_int U (rows)."""
    return np.atleast_2d(np.asarray(psi_int_rows, float)) @ U_int


def build_targets(case, ref: dict, grid, m: int) -> dict:
    """Targets along ``ref`` (keys phi (N+1,n), t, h, i0, out_idx) for a resolved dimension ``m``.

    Returns dict(c, z, M, t, rms_M, ell, psi0, V_int, U_int, lam_V, sys, case_id).
    ``rms_M = sqrt(mean_n ||M_n||^2 / m)``.
    """
    n = grid.n_k
    phi = np.asarray(ref["phi"], float)
    t = np.asarray(ref["t"], float)
    psi = cv.detrend_rows(phi, grid.k)
    psi_int = psi[:, 1:-1]
    V_int = cv.dst_interior(n, m)
    U_int = cv.dst_complement_interior(n, m)
    lam_V = cv.sine_eigenvalues(n, grid.dk, np.arange(1, m + 1))
    sys_ = cv.interior_system(case, grid)
    c = projected_state(psi_int, V_int)
    z = unresolved_state(psi_int, U_int)
    if sys_.time_dependent:
        M = np.empty((len(t), m))
        for i in range(len(t)):
            M[i] = cv.memory_target(sys_, float(t[i]), V_int, U_int, psi_int[i:i + 1])[0]
    else:
        M = cv.memory_target(sys_, float(t[0]), V_int, U_int, psi_int)
    rms_M = float(np.sqrt(np.mean(np.sum(M ** 2, axis=1) / m)))
    ell = cv.lift_rows(phi[0], grid.k)
    psi0 = psi[0].copy()
    return {"c": c, "z": z, "M": M, "t": t, "rms_M": rms_M, "ell": ell, "psi0": psi0,
            "V_int": V_int, "U_int": U_int, "lam_V": lam_V, "sys": sys_,
            "case_id": getattr(case, "id", None)}


def trajectory_weights(rms_by_case: Dict[str, float], n_rows_by_case: Dict[str, int]) -> Dict[str, float]:
    """w_case = 1 / (n_rows * max(rms, 0.1 * median(rms values))**2)."""
    med = float(np.median(list(rms_by_case.values())))
    out = {}
    for k, r in rms_by_case.items():
        out[k] = 1.0 / (n_rows_by_case[k] * max(float(r), 0.1 * med) ** 2)
    return out


def d_extremes(cases: Iterable, grid, times) -> Tuple[float, float]:
    """(d_min, d_max) of the interior coefficient d(t) over cases x times x interior points."""
    lo, hi = np.inf, -np.inf
    for case in cases:
        s = cv.interior_system(case, grid)
        for t in np.atleast_1d(times):
            d = s.d(float(t))
            lo = min(lo, float(d.min()))
            hi = max(hi, float(d.max()))
    return lo, hi
