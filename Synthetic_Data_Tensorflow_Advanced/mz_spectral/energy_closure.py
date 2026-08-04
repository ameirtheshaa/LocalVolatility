#!/usr/bin/env python3
r"""
MZ x Dupire — density-side energy-budget closure (Step 2 of the synthesis).

Implements §5 of ``docs/MZ_DUPIRE_SYNTHESIS.md``: a Gaussian/log-normal resolved
core plus a single stable tail-energy scalar ``E_tail`` (the "tail energy"), whose
maturity evolution obeys a balance ODE, coupled back through a structure-preserving
map that makes the risk-neutral density VALID BY CONSTRUCTION:

  * ``∫ f dK = 1``                — exact, free (Hermite orthogonality, §5.4)
  * ``E[S_T] = S0 e^{rT}``        — exact, free (martingale-pinned core mean, §5.5)
  * ``f ≥ 0``                     — NOT free: the raw Gram-Charlier expansion goes
                                    negative in the wings (the classical Edgeworth
                                    failure = SLGG's sign-structure theorem). The
                                    CURE is a max-entropy / exponential tilt (§5.6),
                                    manifestly positive.

Standardized log-return convention (§5.1). Let ``X_T = ln(S_T/S0)`` with risk-neutral
mean ``μ_T`` and std ``s_T``; ``Y = (X_T - μ_T)/s_T`` is the standardized log return.
The density of ``Y`` is expanded in probabilists' Hermite functions:

    g(y) = φ_N(y) · [ 1 + Σ_{n>=3} c_n He_n(y) ],   φ_N(y) = exp(-y^2/2)/sqrt(2π).

With the leading non-Gaussian channels ``c_3 = Skew/6``, ``c_4 = ExKurt/24``.

Pure NumPy + ``numpy.polynomial.hermite_e`` (the probabilists' He_n ladder, for which
``E[He_n^2] = n!`` under N(0,1)) + ``scipy.optimize`` (the max-entropy λ solve).
TensorFlow-free at import — keep it that way.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial.hermite_e import hermegauss, hermeval
from scipy import optimize

# numpy.trapz was renamed numpy.trapezoid in newer numpy; support both.
_trapz = getattr(np, "trapezoid", np.trapz)

# Default leading non-Gaussian Hermite orders used everywhere (skew + kurtosis
# channels). §8 Step 1 found the standardized skew/kurtosis 99% collinear (PCA
# dim 1) so a single E_tail scalar suffices; we still carry both He_3 and He_4
# coefficients because E_tail mixes them (E_tail = 6 c_3^2 + 24 c_4^2).
DEFAULT_ORDERS: Tuple[int, ...] = (3, 4)

SQRT_2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Hermite helpers (probabilists' He_n)
# ---------------------------------------------------------------------------
def _he(n: int, x: np.ndarray) -> np.ndarray:
    """Probabilists' Hermite polynomial He_n evaluated at x."""
    c = np.zeros(n + 1)
    c[n] = 1.0
    return hermeval(np.asarray(x, dtype=float), c)


def phi_normal(y: np.ndarray) -> np.ndarray:
    """Standard normal pdf φ_N(y) = exp(-y^2/2)/sqrt(2π)."""
    y = np.asarray(y, dtype=float)
    return np.exp(-0.5 * y * y) / SQRT_2PI


# ---------------------------------------------------------------------------
# 1. Log-return moment extractor (the "missing extractor", §5.1)
# ---------------------------------------------------------------------------
def log_return_moments(
    f: np.ndarray, K: np.ndarray, S0: float
) -> Tuple[float, float, float, float]:
    r"""
    Map a strike-space density f(K) to the standardized log-return moments §5.1 needs.

    Treats f(K) as the density of ``S_T`` on the strike grid K, forms the log-return
    ``X = ln(S/S0)`` density by the change of variables ``f_X(x) = f(K) K`` (with
    ``K = S0 e^x``), and returns

        μ_T   = E[X]
        s2_T  = Var[X]
        skew_Y = E[((X-μ)/s)^3]            (standardized skewness)
        exkurt_Y = E[((X-μ)/s)^4] - 3       (standardized excess kurtosis)

    so that ``Y = (X - μ_T)/s_T`` has unit variance, ``skew_Y``, and ``exkurt_Y``.

    Returns ``(μ_T, s2_T, skew_Y, exkurt_Y)``; NaNs if the density is degenerate.
    """
    f = np.clip(np.asarray(f, dtype=float), 0.0, None)
    K = np.asarray(K, dtype=float)
    pos = K > 0
    f = f[pos]
    K = K[pos]
    if K.size < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    area = _trapz(f, K)
    if not np.isfinite(area) or area <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    f = f / area
    x = np.log(K / float(S0))  # log-return support
    # Moments of X under f(K) dK (== f_X(x) dx, mass-preserving change of var)
    mu = _trapz(x * f, K)
    var = _trapz((x - mu) ** 2 * f, K)
    if not np.isfinite(var) or var <= 0:
        return float(mu), float("nan"), float("nan"), float("nan")
    sd = math.sqrt(var)
    m3 = _trapz((x - mu) ** 3 * f, K)
    m4 = _trapz((x - mu) ** 4 * f, K)
    skew = m3 / sd ** 3
    exkurt = m4 / var ** 2 - 3.0
    return float(mu), float(var), float(skew), float(exkurt)


def log_return_moments_from_samples(
    samples: np.ndarray, S0: float
) -> Tuple[float, float, float, float]:
    r"""
    Standardized log-return moments straight from MC terminal samples ``S_T`` (the most
    accurate truth — no grid, no KDE bandwidth, no truncation bias). Returns
    ``(μ_T, s2_T, skew_Y, exkurt_Y)`` of ``X = ln(S_T/S0)``.

    Preferred over ``log_return_moments`` on a strike grid when raw paths are available:
    a truncated K-grid chops the tails that carry the skew/kurtosis and badly biases the
    grid-extracted moments (a const-σ lognormal on [500,3000] spuriously reads skew≈0.14).
    """
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size < 20:
        return float("nan"), float("nan"), float("nan"), float("nan")
    x = np.log(s / float(S0))
    mu = float(x.mean())
    var = float(x.var())
    if var <= 0:
        return mu, float("nan"), float("nan"), float("nan")
    sd = math.sqrt(var)
    z = (x - mu) / sd
    skew = float((z ** 3).mean())
    exkurt = float((z ** 4).mean() - 3.0)
    return mu, var, skew, exkurt


def coeffs_from_skew_kurt(
    skew_Y: float, exkurt_Y: float, orders: Sequence[int] = DEFAULT_ORDERS
) -> Dict[int, float]:
    r"""Gram-Charlier coefficients from standardized skew/kurtosis: c_3=skew/6, c_4=exkurt/24."""
    mapping = {3: skew_Y / 6.0, 4: exkurt_Y / 24.0}
    return {n: float(mapping.get(n, 0.0)) for n in orders}


# ---------------------------------------------------------------------------
# 2. Raw Gram-Charlier density (CAN go negative — the failure mode)
# ---------------------------------------------------------------------------
def gram_charlier_density(
    y: np.ndarray,
    c3: float = 0.0,
    c4: float = 0.0,
    *,
    coeffs: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    r"""
    Raw Gram-Charlier / Edgeworth density of the standardized variable Y:

        g(y) = φ_N(y) · [ 1 + Σ_{n} c_n He_n(y) ].

    ``coeffs`` (an ``{order: c_n}`` dict) overrides the (c3, c4) scalar arguments.
    NOT clipped: this expansion CAN go negative in the wings for large |c3|,|c4|
    (the classical Edgeworth failure — the sign-theorem failure mode we exhibit).
    """
    y = np.asarray(y, dtype=float)
    if coeffs is None:
        coeffs = {3: c3, 4: c4}
    bracket = np.ones_like(y)
    for n, cn in coeffs.items():
        if cn != 0.0:
            bracket = bracket + cn * _he(int(n), y)
    return phi_normal(y) * bracket


# ---------------------------------------------------------------------------
# 3. Max-entropy / exponential-tilt density (manifestly >= 0 — the cure, §5.6)
# ---------------------------------------------------------------------------
def _maxent_eval(y: np.ndarray, lambdas: Dict[int, float], log_Z: float) -> np.ndarray:
    """g(y) = exp[ Σ λ_n He_n(y) − y^2/2 − log√(2π) − log Z ], computed in ONE exp.

    Folding the −y^2/2 Gaussian damping into the exponent keeps it finite even where
    the tilt polynomial Σ λ_n He_n(y) ~ +y^4 is large (the wings), so no overflow.
    """
    y = np.asarray(y, dtype=float)
    s = np.zeros_like(y)
    for n, lam in lambdas.items():
        s = s + lam * _he(int(n), y)
    expo = s - 0.5 * y * y - math.log(SQRT_2PI) - log_Z
    return np.exp(expo)


def _tilt_log_Z(lambdas: Dict[int, float], y_max: float = 12.0, n_grid: int = 8000) -> float:
    r"""
    log of the tilt normalization ``Z = ∫ φ_N(y) exp(Σ λ_n He_n(y)) dy`` by TRAPEZOIDAL
    integration on a fine uniform y-grid (NOT Gauss-Hermite): a polynomial tilt makes a
    single GH tail node grow super-Gaussian and dominate the GH sum, giving a node-count-
    dependent Z. The uniform-grid integral of the (bounded, Gaussian-cored) integrand is
    robust. Computed via log-sum-exp on the log-integrand for numerical safety.
    """
    y = np.linspace(-float(y_max), float(y_max), int(n_grid))
    log_core = np.zeros_like(y)
    for n, lam in lambdas.items():
        log_core = log_core + lam * _he(int(n), y)
    log_integrand = log_core - 0.5 * y * y - math.log(SQRT_2PI)
    m = float(np.max(log_integrand))
    Z = _trapz(np.exp(log_integrand - m), y)
    return float(m + math.log(max(Z, 1e-300)))


# exponent clamp: keep a transient bad iterate (e.g. a momentarily non-normalizable
# quartic) from poisoning the quadrature with inf; the clamp only bites off the
# optimization path, never the solved optimum (where the tilt is normalizable).
_EXP_CLAMP = 30.0

# Gauss-Hermite node count used by the λ-SOLVE specifically. Kept deliberately MODEST
# (max node ≈ 8) because a fine grid puts He_4 ~ y^4 at extreme nodes, which ill-conditions
# the nonlinear moment solve near the feasibility boundary. The Z/MGF/density EVALUATION
# can use a finer grid (the `n_quad` arguments) safely once the bounded λ are fixed.
_SOLVE_QUAD = 40


def _tilt_raw_moments(
    lam_full: np.ndarray, He_full: np.ndarray, nodes: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    r"""
    Raw power moments ``E_g[Y^k]`` for k=1..4 under the tilt g = φ_N exp(Σ_m λ_m He_m)/Z,
    by Gauss-Hermite quadrature. ``He_full`` are He_1..He_M evaluated at ``nodes``.
    Log-sum-exp + clamp stabilized.
    """
    s = (lam_full[:, None] * He_full).sum(axis=0)
    s = np.minimum(s - float(np.max(s)), _EXP_CLAMP)
    e = np.exp(s) * weights  # ∝ g at the nodes (× e^{-y^2/2}/√(2π) already in weights)
    Z = e.sum()
    return np.array([(nodes ** k * e).sum() / Z for k in (1, 2, 3, 4)])


def _maxent_moment_system(
    lam_vec: np.ndarray,
    He_full: np.ndarray,
    targets: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    r"""
    Residual of the four standardized-moment constraints for the tilt:

        E_g[Y]=0,  E_g[Y^2]=1,  E_g[Y^3]=skew,  E_g[Y^4]=exkurt+3 ,

    matched over λ = (λ_1, λ_2, λ_3, λ_4) so the tilt's mean, variance, skew, and
    kurtosis equal the targets. (Power moments are far better conditioned for the
    nonlinear solve than the Hermite-coefficient residual; the two are equivalent
    at the optimum.) ``targets`` = [0, 1, skew, exkurt+3].
    """
    return _tilt_raw_moments(lam_vec, He_full, nodes, weights) - targets


def maxent_tilt_density(
    y: np.ndarray,
    lambdas: Optional[Dict[int, float]] = None,
    *,
    target_coeffs: Optional[Dict[int, float]] = None,
    target_moments: Optional[Tuple[float, float]] = None,
    orders: Sequence[int] = DEFAULT_ORDERS,
    n_quad: int = 48,
    return_lambdas: bool = False,
):
    r"""
    Maximum-entropy / exponential-tilt density (§5.6), manifestly nonnegative:

        g(y) = φ_N(y) · exp[ Σ_{n=1}^{4} λ_n He_n(y) ] / Z,   Z by Gauss-Hermite quad.

    Call modes:
      * ``lambdas`` given                 -> evaluate directly (Z computed internally).
      * ``target_moments=(skew, exkurt)`` -> solve λ from the four STANDARDIZED-moment
        constraints E_g[Y]=0, E_g[Y^2]=1, E_g[Y^3]=skew, E_g[Y^4]=exkurt+3.
      * ``target_coeffs`` (``{order: c_n}``) -> same, with skew=6 c_3, exkurt=24 c_4
        (the Gram-Charlier coefficient convention).

    The λ-solve is ``scipy.optimize.least_squares`` (Trust-Region Reflective) seeded at
    the small-cumulant Gram-Charlier values, with the quartic coefficient BOX-BOUNDED to
    the normalizable cone (λ_4 ≤ 0 ⇔ tails no fatter than Gaussian at infinity). Matching
    the raw-GC moments while forcing positivity is the density-side analog of SLGG's
    structure-preserving projection — build positivity into the representation, never clip.

    NOTE (the §5.4 sign theorem, made quantitative): a 4-moment He-tilt with λ_4 ≤ 0
    spans skew with negative-to-moderately-positive excess kurtosis (the bounded-λ cone),
    but a LARGE positive excess kurtosis combined with large skew is OUTSIDE the He-tilt
    feasible region (the quartic max-entropy density cannot have fat-enough tails). In that
    regime the solve falls back to the closest-feasible tilt and ``info["solved"]`` is False
    with ``info["max_moment_residual"]`` reporting the gap — an honest statement of the
    boundary, not a silent failure.

    Returns ``g(y)`` (>= 0). If ``return_lambdas`` also returns ``(lambdas, info)``.
    """
    # full ladder He_1..He_4 (standardization needs all four). The λ-solve and the Z/MGF
    # all use robust grids internally (modest GH for the solve, uniform-trapezoid for Z);
    # the eval n_quad argument is retained for API compatibility but the density is placed
    # on the caller's y points directly.
    solve_orders = (1, 2, 3, 4)

    info: Dict[str, object] = {"solved": False}
    if lambdas is None:
        if target_moments is not None:
            skew_t, exk_t = float(target_moments[0]), float(target_moments[1])
        elif target_coeffs is not None:
            skew_t = 6.0 * float(target_coeffs.get(3, 0.0))
            exk_t = 24.0 * float(target_coeffs.get(4, 0.0))
        else:
            raise ValueError("provide lambdas, target_moments, or target_coeffs")
        targets = np.array([0.0, 1.0, skew_t, exk_t + 3.0])
        # SOLVE on the modest fixed grid (well-conditioned), independent of eval n_quad
        s_nodes_arr, s_w = hermegauss(_SOLVE_QUAD)
        s_weights = s_w / SQRT_2PI
        He_full = np.array([_he(n, s_nodes_arr) for n in solve_orders])  # (4, _SOLVE_QUAD)
        # seed: small-cumulant GC values (λ_1=λ_2≈0, λ_3≈skew/6, λ_4≈exkurt/24, clipped<=0)
        lam0 = np.array([0.0, 0.0, skew_t / 6.0, min(exk_t / 24.0, -1e-6)])
        # box bounds: keep the tilt in a well-conditioned, normalizable cone.
        #   λ_4 ≤ 0  -> tails no fatter than Gaussian at infinity (normalizable);
        #   |λ_3| ≤ 3, |λ_4| ≤ 3 -> any sane near-Gaussian tilt; bounding the cubic
        #   stops the solver from chasing an INFEASIBLE (skew, +large exkurt) target
        #   with a runaway λ_3 (the §5.4 sign-theorem boundary, made finite).
        lb = np.array([-10.0, -10.0, -3.0, -3.0])
        ub = np.array([10.0, 10.0, 3.0, 0.0])
        sol = optimize.least_squares(
            _maxent_moment_system, lam0, args=(He_full, targets, s_nodes_arr, s_weights),
            bounds=(lb, ub), xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000,
        )
        lam_vec = sol.x
        achieved = _tilt_raw_moments(lam_vec, He_full, s_nodes_arr, s_weights)
        # Infeasibility guard: if the solver wandered into a meaningless region (absurd
        # achieved moments — the §5.4 sign-theorem boundary where a fat-tailed target is
        # outside the bounded-λ He-tilt cone), fall back to λ = 0 (the standard Gaussian,
        # which IS a valid normalizable density). This makes the boundary an HONEST
        # closest-VALID answer with a reported moment gap, never a NaN/inf blow-up.
        # (A cubic-only seed is NOT a safe fallback — an odd tilt is non-normalizable.)
        if (not np.all(np.isfinite(achieved))) or np.max(np.abs(achieved)) > 50.0:
            lam_vec = np.zeros(4)
            achieved = _tilt_raw_moments(lam_vec, He_full, s_nodes_arr, s_weights)
        lambdas = {int(n): float(l) for n, l in zip(solve_orders, lam_vec)}
        resid = achieved - targets
        info = {
            "solved": bool(sol.success) and float(np.max(np.abs(resid))) < 1e-6,
            "max_moment_residual": float(np.max(np.abs(resid))),
            "lambdas": dict(lambdas),
            "target_skew": skew_t, "target_exkurt": exk_t,
            "achieved_mean": float(achieved[0]), "achieved_var": float(achieved[1]),
            "achieved_skew": float(achieved[2]), "achieved_exkurt": float(achieved[3] - 3.0),
        }

    # normalization constant via robust trapezoidal integration (see _tilt_log_Z)
    log_Z = _tilt_log_Z(lambdas)
    g = _maxent_eval(y, lambdas, log_Z)
    info["log_Z"] = log_Z
    if return_lambdas:
        return g, lambdas, info
    return g


# ---------------------------------------------------------------------------
# 4. Tail-energy scalar + balance ODE (the transplant, §5.3-5.4)
# ---------------------------------------------------------------------------
def E_tail(coeffs: Dict[int, float]) -> float:
    r"""
    Tail-energy scalar ``E_tail = Σ_{n>=3} n! c_n^2`` (§5.3): the squared χ²-distance
    of g from Gaussian in the L^2(1/φ_N) metric. Nonnegative by construction (the
    density analog of SLGG's E_U >= 0). For (c3,c4): E_tail = 6 c3^2 + 24 c4^2.
    """
    return float(sum(math.factorial(int(n)) * (cn ** 2) for n, cn in coeffs.items()))


def balance_ode_step(
    E: float, S2: float, gamma_in: float, gamma_c: float, dT: float
) -> float:
    r"""
    One backward-Euler step of the §5.4 tail-energy balance ODE

        dE_tail/dT = γ_in · S(T)^2 − γ_c · E_tail .

    Backward-Euler (unconditionally stable, matches the resolved BE integrator):

        E^{n+1} = (E^n + dT γ_in S2) / (1 + dT γ_c).

    With γ_in, S2, γ_c > 0 the unique attracting fixed point is
    ``E* = (γ_in/γ_c) S2`` and E_tail cannot blow up — §5.4 unconditional stability.
    """
    return float((E + dT * gamma_in * float(S2)) / (1.0 + dT * gamma_c))


def integrate_balance_ode(
    S2_of_T: np.ndarray,
    T_grid: np.ndarray,
    gamma_in: float,
    gamma_c: float,
    E0: float = 0.0,
) -> np.ndarray:
    """Roll the §5.4 balance ODE over a maturity grid by backward-Euler; returns E_tail(T)."""
    S2_of_T = np.asarray(S2_of_T, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)
    n = len(T_grid)
    E = np.zeros(n)
    E[0] = float(E0)
    for i in range(n - 1):
        dT = T_grid[i + 1] - T_grid[i]
        E[i + 1] = balance_ode_step(E[i], S2_of_T[i + 1], gamma_in, gamma_c, dT)
    return E


def coeffs_from_E_tail(
    E_target: float,
    skew_kurt_ratio: float,
    orders: Sequence[int] = DEFAULT_ORDERS,
) -> Dict[int, float]:
    r"""
    Recover Gram-Charlier coefficients from a target ``E_tail`` along a fixed
    skew/kurtosis direction (the §8 Step-1 dim-1 finding: skew & kurtosis 99%
    collinear, so one scalar + one direction parameterizes the tail).

    Parameterize ``c_4 = ratio · c_3`` (the empirical direction), then
    ``E_tail = 6 c_3^2 + 24 c_4^2 = (6 + 24 ratio^2) c_3^2`` fixes |c_3|; sign(c_3)
    is taken negative (left-skewed equity default). Used by the balance-ODE path
    in Phase B as the §5.7 coupling ``{c_n} <- E_tail``.
    """
    denom = 6.0 + 24.0 * skew_kurt_ratio ** 2
    c3 = -math.sqrt(max(E_target, 0.0) / denom) if denom > 0 else 0.0
    c4 = skew_kurt_ratio * c3
    base = {3: c3, 4: c4}
    return {int(n): float(base.get(n, 0.0)) for n in orders}


# ---------------------------------------------------------------------------
# 5. Martingale-pinned core mean (§5.5) — makes E[S_T]=S0 e^{rT} EXACT
# ---------------------------------------------------------------------------
def _gc_mgf_bracket(coeffs: Dict[int, float], s_T: float) -> float:
    r"""The bracket ``1 + Σ_{n} c_n s_T^n`` appearing in the Gram-Charlier MGF."""
    return 1.0 + sum(float(cn) * (float(s_T) ** int(n)) for n, cn in coeffs.items())


def martingale_core_mean(
    s2_T: float, coeffs: Dict[int, float], r: float, T: float
) -> float:
    r"""
    Martingale-pinned core mean (§5.5). With the Gram-Charlier MGF
    ``M_Y(u)=e^{u^2/2}[1+Σ c_n u^n]`` and ``S_T = S0 exp(μ_T + s_T Y)``,

        E[S_T] = S0 e^{μ_T + s_T^2/2} [ 1 + Σ_n c_n s_T^n ].

    Defining

        μ_T = r T − ½ s_T^2 − ln[ 1 + Σ_n c_n s_T^n ]

    makes ``E[S_T] = S0 e^{rT}`` EXACT for ANY skew/kurtosis {c_n}. This is the
    closed-form version of the φ̃(T,0)=1 boundary condition the PINN enforces softly.
    """
    s_T = math.sqrt(max(float(s2_T), 0.0))
    bracket = _gc_mgf_bracket(coeffs, s_T)
    if bracket <= 0:
        # tilt bracket should stay positive for sensible tail energy; guard anyway
        bracket = 1e-12
    return float(r * T - 0.5 * float(s2_T) - math.log(bracket))


def martingale_core_mean_tilt(
    s2_T: float, lambdas: Dict[int, float], r: float, T: float, *,
    y_max: float = 12.0, n_grid: int = 8000
) -> float:
    r"""
    Martingale-pinned core mean for the MAX-ENTROPY TILT (§5.5, tilt version).

    The tilt's MGF is not the closed-form Gram-Charlier bracket, so we compute the
    log-MGF ``ln E_g[e^{s_T Y}]`` and set

        μ_T = r T − ln E_g[ e^{s_T Y} ]   ⇒   E[S_T] = S0 e^{μ_T} E_g[e^{s_T Y}] = S0 e^{rT}

    EXACT for the tilt, for any λ. (For the raw GC density use ``martingale_core_mean``.)

    The expectation is taken by TRAPEZOIDAL integration on a fine UNIFORM y-grid, NOT
    Gauss-Hermite quadrature: the MGF integrand ``e^{s_T y} g(y)`` is smooth and bounded
    (the Gaussian core of g decays faster than e^{s_T y} grows), but a polynomial tilt
    ``λ_3 He_3(y)`` makes one GH tail node grow super-Gaussian and dominate the GH sum —
    so GH gives a node-count-dependent, unreliable MGF here. The uniform grid is robust.
    """
    s_T = math.sqrt(max(float(s2_T), 0.0))
    y = np.linspace(-float(y_max), float(y_max), int(n_grid))
    # log g(y) up to the (μ-independent) normalization, computed in one exp for stability
    log_core = np.zeros_like(y)
    for n, lam in lambdas.items():
        log_core = log_core + lam * _he(int(n), y)
    log_g_un = log_core - 0.5 * y * y - math.log(SQRT_2PI)  # un-normalized log density
    g_un = np.exp(log_g_un - float(np.max(log_g_un)))  # shift for stability (cancels in ratio)
    Z = _trapz(g_un, y)
    mgf = _trapz(np.exp(s_T * y) * g_un, y) / max(Z, 1e-300)
    return float(r * T - math.log(max(mgf, 1e-300)))


# ---------------------------------------------------------------------------
# 6. y -> K change of variables: place the closure density on the strike grid
# ---------------------------------------------------------------------------
def density_on_K_grid(
    K: np.ndarray,
    mu_T: float,
    s2_T: float,
    coeffs: Dict[int, float],
    S0: float,
    *,
    kind: str = "maxent",
    orders: Sequence[int] = DEFAULT_ORDERS,
    n_quad: int = 64,
    lambdas: Optional[Dict[int, float]] = None,
    return_aux: bool = False,
):
    r"""
    Place the closure density on the strike grid K (§5.4 ``y -> K`` change of variables).

    With ``S = S0 exp(μ_T + s_T y)`` (so ``K`` plays the role of ``S``), the standardized
    variable is ``y = (ln(K/S0) - μ_T)/s_T`` and the strike density is

        f(K) = g(y) · |dy/dK| = g(y) / (s_T · K),

    where ``g`` is either the raw Gram-Charlier (``kind="raw"``) or the max-entropy tilt
    (``kind="maxent"``). Mass ``∫f dK = ∫g dy = 1`` is exact (Hermite orthogonality);
    the martingale ``∫K f dK = S0 e^{rT}`` is exact when ``μ_T`` comes from
    ``martingale_core_mean`` for the SAME g (raw bracket for raw, tilt bracket for maxent).

    Returns ``f(K)`` (NOT renormalized — the invariants must hold by construction, so
    we never launder them with a clip+renormalize). If ``return_aux`` also returns a
    dict with the solved λ's / moment residual (maxent) for diagnostics.
    """
    K = np.asarray(K, dtype=float)
    s_T = math.sqrt(max(float(s2_T), 0.0))
    if s_T <= 0:
        out = np.zeros_like(K)
        return (out, {}) if return_aux else out
    y = (np.log(np.maximum(K, 1e-300) / float(S0)) - float(mu_T)) / s_T

    aux: Dict[str, object] = {}
    if kind == "raw":
        g = gram_charlier_density(y, coeffs=coeffs)
    elif kind == "maxent":
        if lambdas is not None:
            g = maxent_tilt_density(y, lambdas=lambdas, n_quad=n_quad)
            aux = {"lambdas": dict(lambdas)}
        else:
            g, lam, info = maxent_tilt_density(
                y, target_coeffs=coeffs, orders=orders, n_quad=n_quad,
                return_lambdas=True,
            )
            aux = info
    else:
        raise ValueError(f"kind must be 'raw' or 'maxent', got {kind!r}")

    f = g / (s_T * np.maximum(K, 1e-300))
    if return_aux:
        return f, aux
    return f


def maxent_lambdas_for_coeffs(
    coeffs: Dict[int, float],
    orders: Sequence[int] = DEFAULT_ORDERS,
    n_quad: int = 64,
) -> Tuple[Dict[int, float], Dict[str, object]]:
    """Solve the max-entropy tilt λ's matching the Gram-Charlier ``coeffs``; (λ dict, info)."""
    _, lam, info = maxent_tilt_density(
        np.array([0.0]), target_coeffs=coeffs, orders=orders, n_quad=n_quad,
        return_lambdas=True,
    )
    return lam, info


def _pin_mu_on_grid(
    K: np.ndarray, s2_T: float, S0: float, r: float, T: float,
    shape_fn, mu_seed: float,
) -> float:
    r"""
    Solve for the core mean μ_T so the martingale ``∫ K f(K) dK = S0 e^{rT}`` holds EXACTLY
    on the *given* trapezoidal K-grid (not on a separate analytic grid). With
    ``f(K) = g((ln(K/S0)-μ)/s_T)/(s_T K)``, shifting μ only translates the density in
    log-space, so this is a smooth 1-D root in μ. ``shape_fn(y)`` returns the standardized
    density g(y) (raw GC or tilt). Anchors the by-construction martingale to the SAME
    discretization the invariants are measured on, eliminating grid-extent mismatch.
    """
    K = np.asarray(K, dtype=float)
    s_T = math.sqrt(max(float(s2_T), 0.0))
    lnK = np.log(np.maximum(K, 1e-300) / float(S0))
    target = float(S0 * math.exp(r * T))

    def mean_minus_target(mu):
        y = (lnK - mu) / s_T
        f = shape_fn(y) / (s_T * np.maximum(K, 1e-300))
        return float(_trapz(K * f, K) - target)

    # Adaptively widen the bracket until it straddles a root (a very fat-tailed density
    # whose mass spills past the K-grid can need a large μ shift to satisfy the martingale
    # ON the truncated grid). Fall back to mu_seed only if no sign change exists at all.
    for half in (3.0, 6.0, 10.0, 16.0):
        a, b = mu_seed - half, mu_seed + half
        try:
            if mean_minus_target(a) * mean_minus_target(b) <= 0:
                return float(optimize.brentq(mean_minus_target, a, b,
                                             xtol=1e-14, rtol=1e-15, maxiter=200))
        except Exception:
            break
    return float(mu_seed)


def closure_density_on_K(
    K: np.ndarray,
    s2_T: float,
    coeffs: Dict[int, float],
    S0: float,
    r: float,
    T: float,
    *,
    kind: str = "maxent",
    orders: Sequence[int] = DEFAULT_ORDERS,
    n_quad: int = 64,
    pin_on_grid: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    r"""
    Full §5 closure density on the strike grid, with the martingale-pinned core mean
    (§5.5) wired to the chosen density ``kind`` so ``E[S_T]=S0 e^{rT}`` holds EXACTLY:

      * ``kind="raw"``    : μ_T from ``martingale_core_mean``     (Gram-Charlier bracket)
      * ``kind="maxent"`` : λ solved from ``coeffs`` (max-entropy tilt), μ_T from
                            ``martingale_core_mean_tilt`` (tilt log-MGF, uniform grid)

    If ``pin_on_grid`` (default), μ_T is additionally refined by ``_pin_mu_on_grid`` so the
    martingale holds to machine precision on the *given* K-grid (removes the small residual
    from integrating the analytic μ_T over a truncated K-grid). Returns ``(f_K, aux)``.
    """
    if kind == "raw":
        mu_T = martingale_core_mean(s2_T, coeffs, r, T)
        shape_fn = lambda y: gram_charlier_density(y, coeffs=coeffs)
        if pin_on_grid:
            mu_T = _pin_mu_on_grid(K, s2_T, S0, r, T, shape_fn, mu_T)
        f = density_on_K_grid(K, mu_T, s2_T, coeffs, S0, kind="raw")
        aux: Dict[str, object] = {"mu_T": mu_T, "kind": "raw", "coeffs": dict(coeffs)}
        return f, aux
    if kind == "maxent":
        lam, info = maxent_lambdas_for_coeffs(coeffs, orders=orders, n_quad=n_quad)
        mu_T = martingale_core_mean_tilt(s2_T, lam, r, T)
        log_Z = _tilt_log_Z(lam)
        shape_fn = lambda y: _maxent_eval(y, lam, log_Z)
        if pin_on_grid:
            mu_T = _pin_mu_on_grid(K, s2_T, S0, r, T, shape_fn, mu_T)
        f = density_on_K_grid(K, mu_T, s2_T, coeffs, S0, kind="maxent",
                              lambdas=lam, n_quad=n_quad)
        info = dict(info)
        info.update({"mu_T": mu_T, "kind": "maxent"})
        return f, info
    raise ValueError(f"kind must be 'raw' or 'maxent', got {kind!r}")


# ---------------------------------------------------------------------------
# Invariant diagnostics (pure NumPy; mirrors run_mz_uq phase_A snippet)
# ---------------------------------------------------------------------------
def density_invariants(
    f: np.ndarray, K: np.ndarray, S0: float, r: float, T: float
) -> Dict[str, float]:
    r"""
    Pure-NumPy mass + martingale check (the run_mz_uq.phase_A pattern):
    ``mass = ∫ f dK``, ``mean = ∫ K f dK``, compared to the forward ``S0 e^{rT}``.

    Does NOT clip or renormalize — these must hold by construction for the closure.
    """
    f = np.asarray(f, dtype=float)
    K = np.asarray(K, dtype=float)
    mass = float(_trapz(f, K))
    mean = float(_trapz(K * f, K))
    target = float(S0 * math.exp(r * T))
    return {
        "mass": mass,
        "mass_defect": float(abs(mass - 1.0)),
        "mean": mean,
        "S0_erT": target,
        "mean_rel_err": float(abs(mean - target) / max(target, 1e-300)),
        "min_f": float(np.min(f)),
    }
