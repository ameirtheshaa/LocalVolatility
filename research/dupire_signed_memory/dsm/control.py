"""Analytic control: eliminate y from x' = -2x + y, y' = x - 2y.

Full generator eigenvalues -1, -3 (dissipative). Resolved equation for x:
    x'(t) = -2 x(t) + e^{-2t} y(0) + ∫_0^t e^{-2(t-s)} x(s) ds,
memory kernel K(τ) = L_RU e^{τ L_UU} L_UR = e^{-2τ} > 0.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def exact(x0: float, y0: float, t: np.ndarray) -> Dict[str, np.ndarray]:
    a = 0.5 * (x0 + y0)       # slow mode (eigenvalue -1), eigenvector (1, 1)
    b = 0.5 * (x0 - y0)       # fast mode (eigenvalue -3), eigenvector (1, -1)
    x = a * np.exp(-t) + b * np.exp(-3 * t)
    y = a * np.exp(-t) - b * np.exp(-3 * t)
    return {"x": x, "y": y}


def memory_integral(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """∫_0^t e^{-2(t-s)} x(s) ds by exact stepping for piecewise-linear x."""
    out = np.zeros_like(x)
    for n in range(len(t) - 1):
        h = t[n + 1] - t[n]
        # ∫_0^h e^{-2(h-τ)} (x_n + (x_{n+1}-x_n) τ/h) dτ
        e = np.exp(-2 * h)
        i0 = (1 - e) / 2.0
        i1 = (h / 2.0 - (1 - e) / 4.0) / h
        out[n + 1] = e * out[n] + x[n] * (i0 - i1) + x[n + 1] * i1
    return out


def check(x0: float = 1.0, y0: float = 0.5, n: int = 4001, t_end: float = 4.0) -> Dict[str, float]:
    t = np.linspace(0.0, t_end, n)
    ex = exact(x0, y0, t)
    x, y = ex["x"], ex["y"]
    mem = memory_integral(x, t)
    ic_term = np.exp(-2 * t) * y0
    rhs = -2 * x + ic_term + mem                    # resolved equation right-hand side
    xdot = -0.5 * (x0 + y0) * np.exp(-t) - 1.5 * (x0 - y0) * np.exp(-3 * t)
    resid = float(np.max(np.abs(xdot - rhs)))
    # the unresolved reconstruction y = e^{-2t} y0 + ∫ e^{-2(t-s)} x(s) ds must equal the true y
    y_rec = ic_term + mem
    # Markovian fit of the memory term on the "diffusion-like" regressor q = -x
    M = ic_term + mem
    q = -x
    nu_star = float(np.sum(M * q) / np.sum(q * q))     # sign: negative = anti-damping
    work = M * x                                        # instantaneous resolved work ⟨x, M⟩
    closed_rate = -2.0 - nu_star                        # x' = (-2 + ν* (-1)·(-1))... see below
    return {
        "max_resolved_equation_residual": resid,
        "max_unresolved_reconstruction_error": float(np.max(np.abs(y_rec - y))),
        "kernel_positive": True,
        "full_eigenvalues": [-1.0, -3.0],
        "nu_star_on_minus_x": nu_star,
        "fraction_time_work_positive": float(np.mean(work > 0)),
        "closed_model_rate": -2.0 + (-nu_star),  # x' = -2x + ν* q = (-2 - ν*) x
        "closed_model_stable": bool((-2.0 - nu_star) < 0),
        "exact_markovian_limit_rate": -1.0,
    }
