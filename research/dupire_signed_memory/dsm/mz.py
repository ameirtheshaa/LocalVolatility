"""Exact Mori–Zwanzig pieces for a fixed block operator and the resolved-coordinate forms."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from scipy.linalg import expm


def resolved_blocks(L: np.ndarray, V: np.ndarray) -> Dict[str, np.ndarray]:
    """Blocks of L for the orthogonal projector P_R = V Vᵀ (V orthonormal, n×m).

    L̂_RR = Vᵀ L V (m×m) acts on resolved coefficients c = Vᵀ φ;
    L̂_RU = Vᵀ L P_U (m×n); L_UR = P_U L V (n×m); L_UU = P_U L P_U (n×n).
    """
    n = L.shape[0]
    P_U = np.eye(n) - V @ V.T
    return {
        "L_RR": V.T @ L @ V,
        "L_RU": V.T @ L @ P_U,
        "L_UR": P_U @ L @ V,
        "L_UU": P_U @ L @ P_U,
        "P_U": P_U,
    }


def memory_term(L: np.ndarray, V: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Exact memory/feedback term in resolved coordinates along a trajectory:
    M̂(t) = Vᵀ L P_U φ(t) = Vᵀ L (φ - V Vᵀ φ).  phi: (n_times, n)."""
    phi = np.atleast_2d(phi)
    phi_U = phi - (phi @ V) @ V.T
    return phi_U @ (V.T @ L).T


def ql_regressor_matrix(D2: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Q̂ = Vᵀ D2 V so that P_R D2 φ_R = V (Q̂ c)."""
    return V.T @ D2 @ V


def pooled_nu(Mhat: np.ndarray, c: np.ndarray, Qhat: np.ndarray, ridge: float = 1e-8) -> Tuple[float, float, float]:
    """Pooled least-squares scalar: ν* = Σ⟨M̂, Q̂c⟩ / (Σ‖Q̂c‖² + ridge). Returns (ν*, num, den)."""
    q = c @ Qhat.T
    num = float(np.sum(Mhat * q))
    den = float(np.sum(q * q))
    return num / (den + ridge), num, den


def phi1_step(L: np.ndarray, h: float, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(e^{hL}, ∫₀ʰ e^{(h-τ)L} g dτ = h φ₁(hL) g) via one augmented expm.

    expm([[hL, g],[0,0]]) = [[e^{hL}, φ₁(hL) g],[0,1]]; the integral needs the extra factor h.
    """
    n = L.shape[0]
    Mx = np.zeros((n + 1, n + 1))
    Mx[:n, :n] = h * L
    Mx[:n, n] = g
    E = expm(Mx)
    return E[:n, :n], h * E[:n, n]


def block_phi1(L: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """(e^{hL}, h φ₁(hL)) from one 2n×2n augmented expm; h φ₁(hL) g = ∫₀ʰ e^{(h-τ)L} g dτ."""
    n = L.shape[0]
    Mx = np.zeros((2 * n, 2 * n))
    Mx[:n, :n] = h * L
    Mx[:n, n:] = np.eye(n)
    E = expm(Mx)
    return E[:n, :n], h * E[:n, n:]


def variation_of_constants(L_UU_of_t: Callable[[float], np.ndarray], forcing: np.ndarray,
                           u0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """u(t) = U(t,t0) u0 + ∫ U(t,s) g(s) ds for u' = L_UU(t) u + g(t), g given on the grid
    (midpoint rule for g, midpoint Magnus for U). Exact for constant L_UU and piecewise-
    constant g; second order otherwise."""
    u = np.asarray(u0, float).copy()
    out = np.empty((len(t), len(u)))
    out[0] = u
    for n in range(len(t) - 1):
        h = t[n + 1] - t[n]
        tm = 0.5 * (t[n] + t[n + 1])
        gm = 0.5 * (forcing[n] + forcing[n + 1])
        E, v = phi1_step(L_UU_of_t(tm), h, gm)
        u = E @ u + v
        out[n + 1] = u
    return out


def kernel_norms(blocks: Dict[str, np.ndarray], taus: np.ndarray) -> np.ndarray:
    """Frobenius norm of K(τ) = L̂_RU e^{τ L_UU} L_UR for the listed lags (constant L only).
    Overflow (divergent orthogonal dynamics) is reported as inf, never raised."""
    out = []
    for tau in taus:
        with np.errstate(all="ignore"):
            try:
                K = blocks["L_RU"] @ expm(tau * blocks["L_UU"]) @ blocks["L_UR"]
                v = float(np.linalg.norm(K)) if np.all(np.isfinite(K)) else float("inf")
            except Exception:
                v = float("inf")
        out.append(v)
    return np.asarray(out)


def kernel_quadratic_range(blocks: Dict[str, np.ndarray], tau: float) -> Tuple[float, float]:
    """(min, max) eigenvalue of sym(K̂(τ)), K̂(τ) = L̂_RU e^{τ L_UU} L_UR in resolved coordinates."""
    with np.errstate(all="ignore"):
        K = blocks["L_RU"] @ expm(tau * blocks["L_UU"]) @ blocks["L_UR"]
    if not np.all(np.isfinite(K)):
        return float("nan"), float("nan")
    ev = np.linalg.eigvalsh(0.5 * (K + K.T))
    return float(ev.min()), float(ev.max())


def spectral_abscissa(M: np.ndarray) -> float:
    return float(np.max(np.linalg.eigvals(M).real))


def numerical_abscissa(M: np.ndarray) -> float:
    return float(np.max(np.linalg.eigvalsh(0.5 * (M + M.T))))
