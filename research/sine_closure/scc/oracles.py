"""Exact-identity oracles for the block (Mori-Zwanzig) decomposition in the sine basis."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from scipy.linalg import expm

import scc  # noqa: F401
from scc import conventions as cv


def projected_reconstruction(psi_int_rows: np.ndarray, V_int: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """phi_hat = ell + [0, V c, 0] with c = psi_int V; returns (N, n)."""
    psi = np.atleast_2d(psi_int_rows)
    c = psi @ V_int
    out = np.zeros((psi.shape[0], psi.shape[1] + 2))
    out[:, 1:-1] = c @ V_int.T
    return out + np.asarray(ell, float)[None, :]


def teacher_forced_rollout(A_mid: Callable[[int], np.ndarray], M_ref: np.ndarray, t: np.ndarray,
                           c0: np.ndarray, time_dependent: bool) -> np.ndarray:
    """c_{n+1} = E c_n + h phi1(h A_mid)(M_n + M_{n+1})/2, one augmented expm per step.

    When ``time_dependent`` is False the pair (E, h phi1) is computed once per distinct step size and reused.
    """
    N = len(t) - 1
    c = np.empty((N + 1, len(c0)))
    c[0] = c0
    cache: Dict[float, tuple] = {}
    for n in range(N):
        h = float(t[n + 1] - t[n])
        g = 0.5 * (M_ref[n] + M_ref[n + 1])
        if time_dependent:
            E, v = cv.phi1_step(np.asarray(A_mid(n), float), h, g)
        else:
            key = round(h, 14)
            if key not in cache:
                A = np.asarray(A_mid(n), float)
                p = A.shape[0]
                Mx = np.zeros((2 * p, 2 * p))
                Mx[:p, :p] = h * A
                Mx[:p, p:] = np.eye(p)
                X = expm(Mx)
                cache[key] = (X[:p, :p], h * X[:p, p:])
            E, Ph = cache[key]
            v = Ph @ g
        c[n + 1] = E @ c[n] + v
    return c


def _rel(err: float, scale: float) -> float:
    return float(err / scale) if scale > 0 else float(err)


def block_identity_autonomous(sys, V_int, U_int, psi_int_rows, t, i0, check_times=None) -> Dict[str, float]:
    """Exact identity z(tau) = e^{tau D} z0 + [top-right of expm(tau [[D,C'],[0,L_rot]])] x0, C'=[C,0].

    L_rot = W^T L_int W = [[A,B],[C,D]], x = psi_int W.  Valid for time-independent L.
    """
    W = np.hstack([V_int, U_int])
    m = V_int.shape[1]
    nU = U_int.shape[1]
    n_int = W.shape[0]
    L = sys.L_int(t[i0])
    Lr = W.T @ L @ W
    A, B, Cb, D = Lr[:m, :m], Lr[:m, m:], Lr[m:, :m], Lr[m:, m:]
    x = np.asarray(psi_int_rows, float) @ W
    N1 = x.shape[0]
    if check_times is None:
        check_times = np.unique(np.linspace(i0 + 1, N1 - 1, 6).round().astype(int))
    check_times = np.asarray(check_times, int)
    x0 = x[i0]
    z0 = x0[m:]
    Cp = np.hstack([Cb, np.zeros((nU, nU))])
    Maug = np.zeros((nU + n_int, nU + n_int))
    Maug[:nU, :nU] = D
    Maug[:nU, nU:] = Cp
    Maug[nU:, nU:] = Lr
    Lx_scale = max(np.max(np.linalg.norm(x @ Lr.T, axis=1)), 1e-300)
    z_scale = max(np.max(np.linalg.norm(x[:, m:], axis=1)), 1e-300)
    ez = ec = 0.0
    for j in check_times:
        tau = float(t[j] - t[i0])
        E = expm(tau * Maug)
        z_rec = E[:nU, :nU] @ z0 + E[:nU, nU:] @ x0
        ez = max(ez, np.linalg.norm(z_rec - x[j, m:]))
        cdot_rec = A @ x[j, :m] + B @ z_rec
        cdot_ref = (Lr @ x[j])[:m]
        ec = max(ec, np.linalg.norm(cdot_rec - cdot_ref))
    j = check_times[-1]
    tau = float(t[j] - t[i0])
    E = expm(tau * Maug)
    return {"max_rel_err_z": float(ez / z_scale), "max_rel_err_cdot": float(ec / Lx_scale),
            "norm_ic_term": float(np.linalg.norm(B @ (E[:nU, :nU] @ z0))),
            "norm_memory_term": float(np.linalg.norm(B @ (E[:nU, nU:] @ x0))),
            "n_checks": int(len(check_times))}


def block_identity_timeordered(sys, V_int, U_int, psi_int_rows, t, i0, order: str = "forward") -> Dict[str, float]:
    """z_{n+1} = e^{h D_mid} z_n + h phi1(h D_mid) C_mid (c_n + c_{n+1})/2 (midpoint operators).

    order='reversed' uses the step operators of step (N-1-n) at step n (control that must fail).
    """
    if order not in ("forward", "reversed"):
        raise ValueError(order)
    psi = np.asarray(psi_int_rows, float)
    c = psi @ V_int
    z_ref = psi @ U_int
    N = len(t) - 1
    z = np.empty_like(z_ref)
    z[: i0 + 1] = z_ref[: i0 + 1]
    zc = z_ref[i0].copy()
    cache: Dict[int, tuple] = {}

    def step_ops(s: int):
        if s not in cache:
            h = float(t[s + 1] - t[s])
            bl = cv.blocks(sys, 0.5 * (t[s] + t[s + 1]), V_int, U_int)
            cache[s] = (h, bl["D"], bl["C"])
        return cache[s]

    for n in range(i0, N):
        s = n if order == "forward" else (N - 1 - n + i0)
        h, D, Cb = step_ops(s)
        g = Cb @ (0.5 * (c[n] + c[n + 1]))
        E, v = cv.phi1_step(D, h, g)
        zc = E @ zc + v
        z[n + 1] = zc
    scale = max(np.max(np.linalg.norm(z_ref[i0:], axis=1)), 1e-300)
    err = np.linalg.norm(z[i0:] - z_ref[i0:], axis=1)
    return {"max_rel_err_z": float(err.max() / scale), "final_rel_err_z": float(err[-1] / scale)}


def b_zero_control(n: int, m: int, dk: float, d_const: float) -> Dict[str, float]:
    """L = d_const D2_int is diagonal in the sine basis => B = 0, memory target 0, TF rollout == RRR."""
    from dsm import operators as _op
    D2 = _op.d2_repo(n, dk)[1:-1, 1:-1]
    L = d_const * D2
    V = cv.dst_interior(n, m)
    U = cv.dst_complement_interior(n, m)
    B = V.T @ L @ U
    A = V.T @ L @ V
    b_ratio = float(np.max(np.abs(B)) / np.max(np.abs(A)))
    assert b_ratio <= 1e-12, f"B not zero: {b_ratio}"
    rng = np.random.default_rng(0)
    psi = rng.standard_normal((7, n - 2))
    w = psi - (psi @ V) @ V.T
    M = (L @ w.T).T @ V
    m_ratio = float(np.max(np.abs(M)) / np.max(np.abs(L @ psi.T)))
    assert m_ratio <= 1e-12, f"memory target not zero: {m_ratio}"
    t = np.linspace(0.0, 1.0, 9)
    c0 = psi[0] @ V
    c_tf = teacher_forced_rollout(lambda s: A, np.zeros((len(t), m)), t, c0, False)
    c_rrr = np.empty_like(c_tf)
    c_rrr[0] = c0
    E = expm(float(t[1] - t[0]) * A)
    for s in range(len(t) - 1):
        c_rrr[s + 1] = E @ c_rrr[s]
    tf_diff = float(np.max(np.abs(c_tf - c_rrr)))
    assert tf_diff <= 1e-12 * max(1.0, np.max(np.abs(c_rrr))), f"TF != RRR: {tf_diff}"
    return {"max_B_over_max_A": b_ratio, "memory_target_rel": m_ratio, "tf_vs_rrr_max_abs": tf_diff}


def decomposition(phi_ref_rows, ell, V_full, c_model_rows) -> Dict[str, object]:
    """phi_ref - (ell + V c_model) = (I-P) psi_ref + V (c_ref - c_model), P = V V^T (full-grid Euclidean)."""
    phi = np.atleast_2d(np.asarray(phi_ref_rows, float))
    ell = np.asarray(ell, float)
    V = np.asarray(V_full, float)
    psi = phi - ell[None, :]
    c_ref = psi @ V
    proj = psi - c_ref @ V.T
    dyn = (c_ref - np.asarray(c_model_rows, float)) @ V.T
    num = np.abs(np.sum(proj * dyn, axis=1))
    den = np.linalg.norm(proj, axis=1) * np.linalg.norm(dyn, axis=1)
    ok = den > 0
    defect = float(np.max(num[ok] / den[ok])) if ok.any() else 0.0
    resid = float(np.max(np.abs(phi - (ell[None, :] + np.asarray(c_model_rows, float) @ V.T) - proj - dyn)))
    return {"proj_rows": proj, "dyn_rows": dyn, "orthogonality_defect": defect, "reconstruction_residual": resid}
