"""Linear closures of the resolved sine-basis equation c' = A c + M-hat: RRR, QL, LM, LR.

Pure NumPy/SciPy.  See docs/API_CONTRACT.md ("scc/closures.py") and PLAN.md sections 5-8.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

import scc  # noqa: F401
from scc import conventions as cv


# ----------------------------------------------------------------------------- spec
@dataclass
class ModelSpec:
    name: str
    m: int
    closure: str                       # 'RRR' | 'QL' | 'LM' | 'LR'
    params: dict = field(default_factory=dict)
    n_h: int = 0
    hidden_init: str = "H0"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.closure not in ("RRR", "QL", "LM", "LR"):
            raise ValueError(f"unknown closure {self.closure!r}")

    def has_bias(self) -> bool:
        return self.closure == "LM" and self.params.get("b") is not None

    def total_state(self) -> int:
        return self.m + self.n_h + (1 if self.has_bias() else 0)

    def param_count(self) -> Dict[str, int]:
        m, nh = self.m, self.n_h
        if self.closure == "RRR":
            return {"fitted": 0, "fixed": 0}
        if self.closure == "QL":
            return {"fitted": 1, "fixed": 0}
        if self.closure == "LM":
            return {"fitted": m * m + (m if self.has_bias() else 0), "fixed": 0}
        learned = bool(self.meta.get("G_learned"))
        fitted = m * m + m * nh + (nh * m if learned else 0)
        fixed = nh + (0 if learned else nh * m)
        if self.params.get("K_init") is not None:
            fitted += nh * (m + 1)
        return {"fitted": int(fitted), "fixed": int(fixed)}


def closed_matrix(A: np.ndarray, spec: ModelSpec, lam_V: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    m = A.shape[0]
    cl = spec.closure
    if cl == "RRR":
        return A
    if cl == "QL":
        return A + float(spec.params["nu"]) * np.diag(np.asarray(lam_V, float))
    if cl == "LM":
        H = np.asarray(spec.params["H"], float)
        b = spec.params.get("b")
        if b is None:
            return A + H
        Mc = np.zeros((m + 1, m + 1))
        Mc[:m, :m] = A + H
        Mc[:m, m] = np.asarray(b, float)
        return Mc
    F = np.asarray(spec.params["F"], float)
    G = np.asarray(spec.params["G"], float)
    H = np.asarray(spec.params["H"], float)
    R = np.asarray(spec.params["R"], float)
    return np.block([[A + H, R], [G, F]])


# ----------------------------------------------------------------------------- rollout
def _initial_state(spec: ModelSpec, c0: np.ndarray, h0: Optional[np.ndarray]) -> np.ndarray:
    c0 = np.asarray(c0, float)
    if spec.closure == "LR":
        h = np.zeros(spec.n_h) if h0 is None else np.asarray(h0, float)
        return np.concatenate([c0, h])
    if spec.has_bias():
        return np.concatenate([c0, [1.0]])
    return c0.copy()


def _step_matrices(spec, A_mid, t, lam_V, time_dependent):
    """Generator of (n, E) with E = expm(h * closed_matrix(A_mid(n)))."""
    N = len(t) - 1
    cache: Dict[float, np.ndarray] = {}
    for n in range(N):
        h = float(t[n + 1] - t[n])
        if time_dependent:
            yield n, expm(h * closed_matrix(A_mid(n), spec, lam_V)), True
        else:
            key = round(h, 14)
            fresh = key not in cache
            if fresh:
                cache[key] = expm(h * closed_matrix(A_mid(n), spec, lam_V))
            yield n, cache[key], fresh


def rollout(spec: ModelSpec, A_mid: Callable[[int], np.ndarray], t: np.ndarray, c0: np.ndarray,
            lam_V: np.ndarray, h0: Optional[np.ndarray] = None, blowup: float = 10.0,
            ref_norm_max: Optional[float] = None, time_dependent: bool = True) -> dict:
    """Free rollout x_{n+1} = expm(h Mc(t_mid)) x_n with x = [c; h] (or [c; 1] for LM with bias)."""
    t = np.asarray(t, float)
    N = len(t) - 1
    m, nh = spec.m, spec.n_h
    x = _initial_state(spec, c0, h0)
    c_out = np.full((N + 1, m), np.nan)
    h_out = np.full((N + 1, nh), np.nan)
    c_out[0] = x[:m]
    if nh:
        h_out[0] = x[m:m + nh]
    ref0 = float(np.linalg.norm(c0)) if ref_norm_max is None else float(ref_norm_max)
    limit = blowup * ref0
    unstable, n_last, n_expm = False, N, 0
    t0 = _time.perf_counter()
    for n, E, fresh in _step_matrices(spec, A_mid, t, lam_V, time_dependent):
        n_expm += int(fresh)
        x = E @ x
        c_out[n + 1] = x[:m]
        if nh:
            h_out[n + 1] = x[m:m + nh]
        if (not np.all(np.isfinite(x))) or np.linalg.norm(x[:m]) > limit:
            unstable, n_last = True, n + 1
            break
    seconds = _time.perf_counter() - t0
    return {"c": c_out, "h": h_out, "unstable": unstable, "n_last": int(n_last),
            "seconds": float(seconds), "n_expm": int(n_expm)}


def admissible_bound(systems: Sequence, V_int: np.ndarray, times) -> float:
    """nu_adm = -min over systems x times of lambda_min(V^T diag(d) V) (a negative number)."""
    lo = np.inf
    for s in systems:
        for t in np.atleast_1d(times):
            lo = min(lo, float(np.linalg.eigvalsh(cv.S_matrix(s, float(t), V_int))[0]))
    return -lo


# ----------------------------------------------------------------------------- weights helpers
def _wlist(targets_list: Sequence[dict], weights) -> List[float]:
    if isinstance(weights, dict):
        if all(tg.get("case_id") in weights for tg in targets_list):
            return [float(weights[tg["case_id"]]) for tg in targets_list]
        vals = list(weights.values())
        if len(vals) != len(targets_list):
            raise ValueError("weights do not match targets_list")
        return [float(v) for v in vals]
    return [float(v) for v in weights]


def _stack(targets_list, weights, key: str) -> Tuple[np.ndarray, np.ndarray]:
    w = _wlist(targets_list, weights)
    X = np.vstack([tg[key] for tg in targets_list])
    wr = np.concatenate([np.full(len(tg[key]), wi) for tg, wi in zip(targets_list, w)])
    return X, wr


# ----------------------------------------------------------------------------- QL
def fit_nu(targets_list: Sequence[dict], weights, nu_adm: float) -> dict:
    w = _wlist(targets_list, weights)
    num = den = 0.0
    for tg, wi in zip(targets_list, w):
        lc = tg["c"] * np.asarray(tg["lam_V"])[None, :]
        num += wi * float(np.sum(tg["M"] * lc))
        den += wi * float(np.sum(lc * lc))
    nu = num / den if den > 0 else float("nan")
    half = 0.5 * nu_adm
    return {"nu_unclipped": nu, "num": num, "den": den, "nu_adm": float(nu_adm),
            "nu_plus": max(nu, 0.0), "nu_pm": nu if nu > half else half}


# ----------------------------------------------------------------------------- ridge / LM
def _ridge_abs(X, Y, w, lam):
    Xw = X * w[:, None]
    Gm = X.T @ Xw
    W = np.linalg.solve(Gm + lam * np.eye(Gm.shape[0]), Xw.T @ Y)
    return W.T, Gm


def ridge_rel(X: np.ndarray, Y: np.ndarray, w_rows: np.ndarray, lam_rel: float) -> np.ndarray:
    """min sum_n w_n ||W x_n - y_n||^2 + lam ||W||_F^2, lam = lam_rel tr(X^T W X)/n_feat; returns (out, feat)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    w = np.asarray(w_rows, float)
    Gm = X.T @ (X * w[:, None])
    lam = lam_rel * np.trace(Gm) / X.shape[1]
    return np.linalg.solve(Gm + lam * np.eye(X.shape[1]), (X * w[:, None]).T @ Y).T


def fit_LM(targets_list, weights, lam_rel: float, intercept: bool = False):
    X, w = _stack(targets_list, weights, "c")
    Y, _ = _stack(targets_list, weights, "M")
    m = X.shape[1]
    Xf = np.hstack([X, np.ones((len(X), 1))]) if intercept else X
    Gm = Xf.T @ (Xf * w[:, None])
    lam = lam_rel * np.trace(Gm) / Xf.shape[1]
    W = ridge_rel(Xf, Y, w, lam_rel)
    H = W[:, :m].copy()
    b = W[:, m].copy() if intercept else None
    res = Y - Xf @ W.T
    info = {"lam": float(lam), "gram_cond": float(np.linalg.cond(Gm)), "n_rows": int(len(X)),
            "weighted_M_sq": float(np.sum(w[:, None] * Y ** 2)),
            "weighted_resid_sq": float(np.sum(w[:, None] * res ** 2))}
    info["resid_ratio"] = info["weighted_resid_sq"] / max(info["weighted_M_sq"], 1e-300)
    return H, b, info


# ----------------------------------------------------------------------------- LR
def rate_bank(n_h: int, bank: str, r1: float, ratio: float) -> np.ndarray:
    if bank == "N":
        lo, hi = 0.5 * r1, 16.0 * r1 * ratio
    elif bank == "W":
        lo, hi = 0.25 * r1, 64.0 * r1 * ratio
    else:
        raise ValueError(f"unknown bank {bank!r}")
    return np.logspace(np.log10(lo), np.log10(hi), n_h)


def filtered_states(c_rows: np.ndarray, t: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Exact exponential filtering of the piecewise-linear input; returns q (N+1, n_h, m), q[0] = 0."""
    c = np.asarray(c_rows, float)
    t = np.asarray(t, float)
    a = np.asarray(alphas, float)[:, None]
    N = len(t) - 1
    q = np.zeros((N + 1, a.shape[0], c.shape[1]))
    cache: Dict[float, tuple] = {}
    for n in range(N):
        h = float(t[n + 1] - t[n])
        key = round(h, 14)
        if key not in cache:
            x = a * h
            E = np.exp(-x)
            I0 = -np.expm1(-x) / a
            series = x < 1e-2
            xs = np.where(series, x, 0.0)
            I1_ser = h * h * (0.5 - xs / 6 + xs ** 2 / 24 - xs ** 3 / 120 + xs ** 4 / 720)
            I1 = np.where(series, I1_ser, h / a - (-np.expm1(-x)) / a ** 2)
            cache[key] = (E, I0, I1 / h)
        E, I0, I1h = cache[key]
        q[n + 1] = E * q[n] + I0 * c[n][None, :] + I1h * (c[n + 1] - c[n])[None, :]
    return q


def _lr_spec(name, m, n_h, alphas, G, H, R, learned):
    return ModelSpec(name=name, m=m, closure="LR", n_h=n_h,
                     params={"F": -np.diag(np.asarray(alphas, float)), "G": np.array(G), "H": np.array(H),
                             "R": np.array(R), "K_init": None},
                     meta={"G_learned": bool(learned), "alphas": np.asarray(alphas, float).copy()})


def fit_LR(targets_list, weights, n_h: int, alphas, lam_rel: float, G0=None, als_iters: int = 30,
           tol: float = 1e-6, seed: int = 0):
    t0 = _time.perf_counter()
    alphas = np.asarray(alphas, float)
    wl = _wlist(targets_list, weights)
    m = targets_list[0]["c"].shape[1]
    Qs = [filtered_states(tg["c"], tg["t"], alphas) for tg in targets_list]
    Q = np.concatenate(Qs, axis=0)                      # (rows, n_h, m)
    C, w = _stack(targets_list, weights, "c")
    Y, _ = _stack(targets_list, weights, "M")
    rows = len(C)
    Qf = Q.reshape(rows, n_h * m)
    QQ = Qf.T @ (Qf * w[:, None])
    G = (np.random.default_rng(seed).standard_normal((n_h, m)) / np.sqrt(m)) if G0 is None else np.array(G0, float)

    def hidden(Gm):
        return np.einsum("ki,nki->nk", Gm, Q)

    def readout(Gm, lam=None):
        X = np.hstack([C, hidden(Gm)])
        Gram = X.T @ (X * w[:, None])
        if lam is None:
            lam = lam_rel * np.trace(Gram) / X.shape[1]
        W = np.linalg.solve(Gram + lam * np.eye(X.shape[1]), (X * w[:, None]).T @ Y).T
        return W[:, :m], W[:, m:], lam, Gram

    H, R, lam, gram1 = readout(G)
    spec0 = _lr_spec("LR0", m, n_h, alphas, G, H, R, False)
    # lam_G fixed from the first G-step Gram so the objective is a fixed function.
    lam_G = lam_rel * np.trace(QQ * np.kron(R.T @ R, np.ones((m, m)))) / (n_h * m)

    def objective(Gm, H, R):
        pred = C @ H.T + hidden(Gm) @ R.T
        return float(np.sum(w[:, None] * (pred - Y) ** 2) + lam * (np.sum(H ** 2) + np.sum(R ** 2))
                     + lam_G * np.sum(Gm ** 2))

    J = [objective(G, H, R)]
    best = (J[0], G.copy(), H.copy(), R.copy())
    conds = [float(np.linalg.cond(gram1))]
    iters = 0
    for _ in range(als_iters):
        iters += 1
        Yp = Y - C @ H.T
        Gram = QQ * np.kron(R.T @ R, np.ones((m, m)))
        rhs = np.einsum("nki,nk->ki", Q * w[:, None, None], Yp @ R).reshape(-1)
        Gn = np.linalg.solve(Gram + lam_G * np.eye(n_h * m), rhs).reshape(n_h, m)
        Hn, Rn, _, gram = readout(Gn, lam)
        Jn = objective(Gn, Hn, Rn)
        if Jn > J[-1]:                      # never accept an increase (roundoff-level only)
            break
        J.append(Jn)
        conds.append(float(np.linalg.cond(gram)))
        G, H, R = Gn, Hn, Rn
        best = (Jn, G.copy(), H.copy(), R.copy())
        if (J[-2] - J[-1]) / max(J[-2], 1e-300) < tol:
            break
    _, G, H, R = best
    spec = _lr_spec("LR", m, n_h, alphas, G, H, R, True)
    info = {"objective": J, "iterations": iters, "lam": float(lam), "lam_G": float(lam_G),
            "gram_cond": conds, "seconds": float(_time.perf_counter() - t0), "n_rows": int(rows)}
    return spec0, spec, info


def fit_initializer(spec: ModelSpec, targets_list, weights, i0: int, lam_rel: float) -> np.ndarray:
    """K (n_h, m+1): teacher-forced h(t_i0) regressed on [c(t_i0); 1]."""
    G = np.asarray(spec.params["G"], float)
    alphas = -np.diag(np.asarray(spec.params["F"], float))
    w = np.array(_wlist(targets_list, weights))
    X = np.empty((len(targets_list), spec.m + 1))
    Yh = np.empty((len(targets_list), spec.n_h))
    for j, tg in enumerate(targets_list):
        q = filtered_states(tg["c"][: i0 + 1], tg["t"][: i0 + 1], alphas)[i0]
        Yh[j] = np.einsum("ki,ki->k", G, q)
        X[j, : spec.m] = tg["c"][i0]
        X[j, spec.m] = 1.0
    return ridge_rel(X, Yh, w, lam_rel)


def hidden_init_state(spec: ModelSpec, protocol: str, c0: np.ndarray) -> np.ndarray:
    if protocol in ("H0", "W0"):
        return np.zeros(spec.n_h)
    if protocol == "WI":
        K = spec.params.get("K_init")
        if K is None:
            raise ValueError("WI needs spec.params['K_init']")
        return np.asarray(K, float) @ np.concatenate([np.asarray(c0, float), [1.0]])
    if protocol == "WU":
        raise ValueError("WU is handled by the scoring script")
    raise ValueError(f"unknown protocol {protocol!r}")


# ----------------------------------------------------------------------------- stability
def abscissae(Mc: np.ndarray) -> Tuple[float, float]:
    Mc = np.asarray(Mc, float)
    return float(np.max(np.linalg.eigvals(Mc).real)), float(np.max(np.linalg.eigvalsh(0.5 * (Mc + Mc.T))))


def stability_report(spec: ModelSpec, A_list_out: Sequence[np.ndarray], lam_V: np.ndarray) -> dict:
    sp, nu = [], []
    for A in A_list_out:
        s, nn = abscissae(closed_matrix(A, spec, lam_V))
        sp.append(s)
        nu.append(nn)
    return {"spectral_abscissa_max": float(max(sp)), "numerical_abscissa_max": float(max(nu)),
            "spectral_abscissa_per_time": sp, "numerical_abscissa_per_time": nu}


def perturbation_growth(spec, A_mid, t, c0, lam_V, eps: float = 1e-6, seed: int = 0, **kw) -> dict:
    c0 = np.asarray(c0, float)
    v = np.random.default_rng(seed).standard_normal(len(c0))
    v /= np.linalg.norm(v)
    d0 = eps * np.linalg.norm(c0)
    base = rollout(spec, A_mid, t, c0, lam_V, **kw)
    pert = rollout(spec, A_mid, t, c0 + d0 * v, lam_V, **kw)
    if base["unstable"] or pert["unstable"]:
        return {"max_growth_ratio": float("inf"), "final_growth_ratio": float("inf")}
    ratio = np.linalg.norm(pert["c"] - base["c"], axis=1) / d0
    return {"max_growth_ratio": float(np.max(ratio)), "final_growth_ratio": float(ratio[-1])}


def propagator_growth(spec, A_mid, t, lam_V, time_dependent: bool = True) -> float:
    t = np.asarray(t, float)
    p = spec.total_state()
    Phi = np.eye(p)
    best = 1.0
    for _, E, _f in _step_matrices(spec, A_mid, t, lam_V, time_dependent):
        Phi = E @ Phi
        nrm = np.linalg.norm(Phi, 2)
        if not np.isfinite(nrm):
            return float("inf")
        best = max(best, float(nrm))
    return best


def long_horizon(spec, A_end, lam_V, t_end: float = 3.0, h: float = 0.8 / 368, t_start: float = 1.0) -> dict:
    m = spec.m
    n_steps = int(round((t_end - t_start) / h))
    E = expm(h * closed_matrix(A_end, spec, lam_V))
    c = np.random.default_rng(0).standard_normal(m)
    c /= np.linalg.norm(c)
    x = _initial_state(spec, c, None)
    k = m + spec.n_h
    n0 = np.linalg.norm(x[:k])
    mx = 1.0
    for _ in range(n_steps):
        x = E @ x
        nr = np.linalg.norm(x[:k]) / n0
        if not np.isfinite(nr):
            return {"bounded": False, "max_norm_ratio": float("inf")}
        mx = max(mx, float(nr))
    return {"bounded": bool(mx <= 10.0), "max_norm_ratio": mx}
