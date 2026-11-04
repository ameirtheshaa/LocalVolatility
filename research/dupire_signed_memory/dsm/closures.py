"""Reduced-model rollouts in resolved coordinates and the closures of the frozen plan.

State: c = Vᵀψ (m resolved Fourier coefficients). All closures use the same exponential
time-differencing scheme (ETD2, Cox–Matthews Adams–Bashforth form)

    c_{n+1} = e^{hL} c_n + h φ₁(hL) M_n + h φ₂(hL) (M_n - M_{n-1}),

with L = L̂_RR(t_{n+1/2}) and M_n the closure's feedback term at step n. The exact-memory
oracle uses M_n = Vᵀ L Q ψ_ref(t_n), so its rollout error is the scheme's own floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm

from dsm import mz, operators as op, reference
from dsm.cases import Case


# ----------------------------------------------------------------------------- integrator
def etd2_matrices(L: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(e^{hL}, h φ₁(hL), h φ₂(hL)) from one 3m×3m augmented expm."""
    m = L.shape[0]
    Mx = np.zeros((3 * m, 3 * m))
    Mx[:m, :m] = h * L
    Mx[:m, m:2 * m] = np.eye(m)
    Mx[m:2 * m, 2 * m:] = np.eye(m)
    E = expm(Mx)
    return E[:m, :m], h * E[:m, m:2 * m], h * E[:m, 2 * m:]


@dataclass
class RomSetup:
    case_id: str
    variant: str
    keep: int
    V: np.ndarray            # (n, m)
    t: np.ndarray            # fine times from t̃0 (window)
    h: float
    out_rel: np.ndarray      # output indices relative to the window
    c_ref: np.ndarray        # (N+1, m) reference resolved coefficients
    M_ref: np.ndarray        # (N+1, m) exact feedback Vᵀ L Q ψ along the reference
    psi_ref: np.ndarray      # (N+1, n) reference state variable on the window
    trend: np.ndarray        # (n,) time-independent trend ℓ (zeros for OP-A)
    Qhat: np.ndarray         # (m, m) QL regressor Vᵀ D2 V
    LRR_mid: List[np.ndarray]      # per-step L̂_RR(t_{n+1/2}) (one entry if constant)
    steps: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]  # per-step ETD2 matrices (one if constant)
    constant: bool

    def step_mats(self, n: int):
        return self.steps[0] if self.constant else self.steps[n]

    def LRR(self, n: int):
        return self.LRR_mid[0] if self.constant else self.LRR_mid[n]


def build_setup(case: Case, grid: op.Grid, fg: reference.FineGrid, variant: str, keep: int,
                phi_fine: Optional[np.ndarray] = None) -> RomSetup:
    if phi_fine is None:
        phi_fine = reference.get_reference(case, grid, variant, fg)
    psi = reference.state_variable(phi_fine, grid, variant)
    n = grid.n_k
    V = op.fourier_basis(n, keep)
    D2v = op.d2_repo(n, grid.dk) if variant == "A" else op.d2_linearity(n, grid.dk)
    L_of_t = reference.operator_builder(case, grid, variant)
    i0 = fg.i0
    t = fg.t[i0:]
    h = fg.h
    psi_w = psi[i0:]
    c_ref = psi_w @ V
    if case.time_dependent:
        M_ref = np.empty_like(c_ref)
        LRR_mid, steps = [], []
        for j in range(len(t)):
            Lj = L_of_t(t[j])
            M_ref[j] = mz.memory_term(Lj, V, psi_w[j:j + 1])[0]
            if j < len(t) - 1:
                Lm = L_of_t(0.5 * (t[j] + t[j + 1]))
                LRRm = V.T @ Lm @ V
                LRR_mid.append(LRRm)
                steps.append(etd2_matrices(LRRm, h))
        constant = False
    else:
        L = L_of_t(0.0)
        M_ref = mz.memory_term(L, V, psi_w)
        LRRm = V.T @ L @ V
        LRR_mid, steps, constant = [LRRm], [etd2_matrices(LRRm, h)], True
    trend = np.zeros(n) if variant == "A" else op.linear_trend(grid.k, phi_fine[0][0], phi_fine[0][-1])
    return RomSetup(case_id=case.id, variant=variant, keep=keep, V=V, t=t, h=h,
                    out_rel=fg.out_idx - i0, c_ref=c_ref, M_ref=M_ref, psi_ref=psi_w, trend=trend,
                    Qhat=mz.ql_regressor_matrix(D2v, V), LRR_mid=LRR_mid, steps=steps, constant=constant)


# ----------------------------------------------------------------------------- closures
class Closure:
    """M_n = feedback(n, c_n); stateful closures update internal state in `advance`."""
    name = "base"

    def reset(self, setup: RomSetup) -> None:
        pass

    def feedback(self, n: int, c: np.ndarray) -> np.ndarray:
        return np.zeros_like(c)

    def advance(self, n: int, c: np.ndarray) -> None:
        pass


class RRR(Closure):
    name = "RRR"


class QL(Closure):
    def __init__(self, nu: float, name: str):
        self.nu, self.name = float(nu), name

    def reset(self, setup):
        self.Q = setup.Qhat

    def feedback(self, n, c):
        return self.nu * (self.Q @ c)


class Oracle(Closure):
    name = "ORACLE"

    def reset(self, setup):
        self.M = setup.M_ref

    def feedback(self, n, c):
        return self.M[n]


class LinearMarkov(Closure):
    name = "LM"

    def __init__(self, C: np.ndarray, bias: np.ndarray, in_scale: float, out_scale: float):
        self.C, self.bias, self.in_scale, self.out_scale = C, bias, in_scale, out_scale

    def feedback(self, n, c):
        return self.out_scale * (self.C @ (c / self.in_scale) + self.bias)


class Reservoir(Closure):
    """Leaky reservoir s_{n+1} = (1-a) s_n + a f(A s_n + B x_n), x = c/in_scale, readout
    M = out_scale · W [s; x; 1]. f = tanh (ESN) or identity (LR). A, B regenerated from the seed."""

    def __init__(self, keep_dim: int, n_res: int = 200, rho: float = 0.9, leak: float = 0.5,
                 in_scale_w: float = 1.0, seed: int = 0, nonlinear: bool = True, name: str = "ESN"):
        self.name = name
        self.n_res, self.rho, self.leak, self.seed, self.nonlinear = n_res, rho, leak, seed, nonlinear
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n_res, n_res))
        A *= rho / np.max(np.abs(np.linalg.eigvals(A)))
        self.A = A
        self.B = rng.uniform(-0.5, 0.5, size=(n_res, keep_dim)) * in_scale_w
        self.W: Optional[np.ndarray] = None
        self.in_scale = 1.0
        self.out_scale = 1.0

    def features(self, s: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.concatenate([s, x, [1.0]])

    def reset(self, setup):
        self.s = np.zeros(self.n_res)

    def update_state(self, s: np.ndarray, x: np.ndarray) -> np.ndarray:
        pre = self.A @ s + self.B @ x
        f = np.tanh(pre) if self.nonlinear else pre
        return (1.0 - self.leak) * s + self.leak * f

    def feedback(self, n, c):
        return self.out_scale * (self.W @ self.features(self.s, c / self.in_scale))

    def advance(self, n, c):
        self.s = self.update_state(self.s, c / self.in_scale)

    def teacher_features(self, c_ref: np.ndarray) -> np.ndarray:
        """Features along a reference trajectory (s_0 = 0), one row per time step."""
        s = np.zeros(self.n_res)
        rows = []
        for n in range(c_ref.shape[0]):
            x = c_ref[n] / self.in_scale
            rows.append(self.features(s, x))
            s = self.update_state(s, x)
        return np.array(rows)


# ----------------------------------------------------------------------------- rollout
def rollout(setup: RomSetup, closure: Closure, blowup_factor: float = 10.0) -> Dict[str, object]:
    N = len(setup.t) - 1
    m = setup.V.shape[1]
    c = np.full((N + 1, m), np.nan)
    c[0] = setup.c_ref[0]
    closure.reset(setup)
    limit = blowup_factor * np.linalg.norm(setup.c_ref, axis=1).max()
    M_prev = None
    unstable = False
    n_last = N
    for n in range(N):
        M = closure.feedback(n, c[n])
        E, P1, P2 = setup.step_mats(n)
        if M_prev is None:
            c[n + 1] = E @ c[n] + P1 @ M
        else:
            c[n + 1] = E @ c[n] + P1 @ M + P2 @ (M - M_prev)
        closure.advance(n, c[n])
        M_prev = M
        if not np.all(np.isfinite(c[n + 1])) or np.linalg.norm(c[n + 1]) > limit:
            unstable = True
            n_last = n + 1
            c[n + 1:] = np.nan
            break
    return {"c": c, "unstable": unstable, "n_last": n_last}


def closed_operator_abscissa(setup: RomSetup, closure: Closure) -> float:
    """Spectral abscissa of the autonomous closed operator at the final time (QL / LM only)."""
    L = setup.LRR(len(setup.steps) - 1)
    if isinstance(closure, QL):
        return mz.spectral_abscissa(L + closure.nu * setup.Qhat)
    if isinstance(closure, LinearMarkov):
        return mz.spectral_abscissa(L + closure.out_scale / closure.in_scale * closure.C)
    return mz.spectral_abscissa(L)


# ----------------------------------------------------------------------------- fitting
def fit_nu_pooled(setups: List[RomSetup], ridge: float = 1e-8) -> Dict[str, float]:
    num = den = 0.0
    for s in setups:
        _, a, b = mz.pooled_nu(s.M_ref, s.c_ref, s.Qhat, ridge=0.0)
        num += a
        den += b
    nu = num / (den + ridge)
    return {"nu_unclipped": nu, "num": num, "den": den}


def ridge_solve(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """W minimising ‖X Wᵀ - Y‖² + λ‖W‖²  (W: out × feat)."""
    G = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(G, X.T @ Y).T


def fit_linear_markov(setups: List[RomSetup], lam: float) -> LinearMarkov:
    in_scale = max(np.abs(s.c_ref).max() for s in setups)
    out_scale = max(np.abs(s.M_ref).max() for s in setups)
    X = np.concatenate([np.hstack([s.c_ref / in_scale, np.ones((s.c_ref.shape[0], 1))]) for s in setups])
    Y = np.concatenate([s.M_ref / out_scale for s in setups])
    W = ridge_solve(X, Y, lam)
    return LinearMarkov(C=W[:, :-1], bias=W[:, -1], in_scale=in_scale, out_scale=out_scale)


def fit_reservoir(setups: List[RomSetup], lam: float, nonlinear: bool, **kw) -> Reservoir:
    m = setups[0].V.shape[1]
    res = Reservoir(keep_dim=m, nonlinear=nonlinear, name="ESN" if nonlinear else "LR", **kw)
    res.in_scale = max(np.abs(s.c_ref).max() for s in setups)
    res.out_scale = max(np.abs(s.M_ref).max() for s in setups)
    X = np.concatenate([res.teacher_features(s.c_ref) for s in setups])
    Y = np.concatenate([s.M_ref / res.out_scale for s in setups])
    res.W = ridge_solve(X, Y, lam)
    return res


def reconstruct_rows(setup: RomSetup, c: np.ndarray) -> np.ndarray:
    """Full-field φ̂ at the output maturities: V c + ℓ."""
    return c[setup.out_rel] @ setup.V.T + setup.trend[None, :]
