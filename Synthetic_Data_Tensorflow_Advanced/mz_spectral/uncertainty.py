#!/usr/bin/env python3
"""
Epistemic uncertainty quantification (UQ) for the MZ–Dupire calibration.

Turns POINT estimates of the local volatility sigma(T,K) and the risk-neutral
density f(K,T)=e^{rT} d^2C/dK^2 into CALIBRATED confidence bands by propagating a
data-estimable input price-surface covariance Sigma_phi through:
  - the Breeden-Litzenberger map  f = A . phi~   (LINEAR -> exact covariance), and
  - the Dupire inverse  sigma = sqrt(2 eta~/T_max), eta~ = phi~_t / (k~^2 phi~_kk)
    (NONLINEAR -> MC-over-input + closed-form delta-method).

Two design ideas are ported from the jet-space reconstruction paper
("Jet-Space Reconstruction of Dynamical Systems from Scalar Time Series",
Shaa & Guet), where they tame noise at a closure degeneracy:

  (1) WEAK-FORM / integration-by-parts propagation.  A density functional
        int f(K) psi(K) dK = e^{rT} int C(K) psi''(K) dK
      moves BOTH K-derivatives onto a smooth, compactly-supported mollifier psi.
      The price covariance is then propagated through psi'' (bounded) instead of
      through the ill-conditioned pointwise d^2/dK^2 (= D2 acting on noisy phi~).
      Localized mollifiers give well-conditioned *pointwise* bands whose width is
      set by the mollification scale h (the regularizer), separating genuine
      statistical tail-widening from spurious numerical blow-up.  This mirrors the
      repo's existing weak-form mean check (compute_mean_diagnostics, PR #2).

  (2) POLE-BENIGNITY feasibility gate.  The inverse has a pole where the price
      convexity phi~_kk -> 0 (the deep tails).  Rather than the crude hard
      reliability mask (sigma_from_phi.sigma_reliability_mask) that NaNs those
      cells, we flag them and let the band quantify the (large, honest)
      uncertainty there.

All pure NumPy.  The Dupire generator is dissipative (de-risk Step 0a), so the
unresolved-mode contribution propagates through a *decaying* map and the bands
stay finite -- the well-posedness that the Hasegawa-Wakatani analog lacked.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .validation import pdf_from_phi_tilde, interior_k_mask
from .sigma_from_phi import implied_sigma_from_phi, sigma_reliability_mask


def smooth_phi_k(phi, smooth_k):
    """Gaussian-smooth phi~ along the strike (k~) axis by `smooth_k` grid points.

    This is the inversion-side regularizer: it is the same mollification that
    conditions the weak-form density (gaussian_mollifier_bank), applied to the
    surface BEFORE the Dupire second derivative, so the pointwise inverse stops
    amplifying price noise.  smooth_k=0 returns phi unchanged.
    """
    if not smooth_k:
        return phi
    return gaussian_filter1d(np.asarray(phi, dtype=float), float(smooth_k),
                             axis=-1, mode="nearest")


# ----------------------------------------------------------------------------
# Input covariance Sigma_phi
# ----------------------------------------------------------------------------
def input_covariance_const(phi, *, abs_noise=None, rel_noise=None, floor=1e-12):
    """Diagonal input std for phi~ on an (n_t,n_k) grid.

    abs_noise : homoscedastic per-cell std (in phi~ units).
    rel_noise : heteroscedastic std = rel_noise * |phi~| (a fractional price noise).
    Returns an (n_t,n_k) array of per-cell standard deviations (NOT variances).
    """
    phi = np.asarray(phi, dtype=float)
    se = np.zeros_like(phi)
    if abs_noise is not None:
        se += float(abs_noise)
    if rel_noise is not None:
        se = np.maximum(se, float(rel_noise) * np.abs(phi))
    return np.maximum(se, floor)


# ----------------------------------------------------------------------------
# Breeden-Litzenberger linear density map  f = A . phi~   (per maturity row)
# ----------------------------------------------------------------------------
def density_linear_operator(T, r, S0, K_max, D2):
    """A (n_k,n_k) s.t. f_raw_row = A @ phi_row  (raw, unclipped density in K).

    f = e^{rT} d^2C/dK^2,  C = S0 phi~,  dk~/dK = e^{-rT}/K_max (constant in K) =>
        d^2/dK^2 = (e^{-rT}/K_max)^2 d^2/dk~^2,  so
        f = S0 e^{-rT}/K_max^2 * (D2 @ phi~).
    """
    c = float(S0) * np.exp(-float(r) * float(T)) / (float(K_max) ** 2)
    return c * np.asarray(D2, dtype=float)


def density_band(phi_row, se_phi_row, T, r, S0, K_max, D2, *, z=1.96):
    """Pointwise raw-density band (noise-amplified baseline).

    Linear map => exact Gaussian covariance for diagonal input std `se_phi_row`:
        f = A phi~,  diag(Cov f) = (A**2) @ se^2.
    Returns (f_mean, f_lo, f_hi).  f_mean is the RAW (pre-clip) linear density.
    """
    A = density_linear_operator(T, r, S0, K_max, D2)
    phi_row = np.asarray(phi_row, dtype=float)
    se2 = np.asarray(se_phi_row, dtype=float) ** 2
    f_mean = A @ phi_row
    var = (A ** 2) @ se2
    sd = np.sqrt(np.maximum(var, 0.0))
    return f_mean, f_mean - z * sd, f_mean + z * sd


# ----------------------------------------------------------------------------
# WEAK-FORM density band (jet-space IBP idea) -- well-conditioned
# ----------------------------------------------------------------------------
def gaussian_mollifier_bank(K_row, h):
    """Rows = K-normalized Gaussian bumps centered at each grid strike (width h)."""
    K_row = np.asarray(K_row, dtype=float)
    dK = K_row[:, None] - K_row[None, :]                  # (center, eval)
    Psi = np.exp(-0.5 * (dK / h) ** 2) / (np.sqrt(2.0 * np.pi) * h)
    # renormalize each bump to unit mass on the actual (possibly nonuniform) K grid
    mass = np.trapz(Psi, K_row, axis=1)
    mass = np.where(mass > 0, mass, 1.0)
    return Psi / mass[:, None]


def weak_form_density_band(phi_row, se_phi_row, K_row, T, r, S0, K_max, D2,
                           *, h_factor=3.0, z=1.96):
    """Mollified density curve + band via the IBP weak form.

    int f psi_i dK = c * sum_m phi~_m (D2^T (w . psi~_i))_m,  c = S0 e^{-rT}/K_max^2,
    w_j = (dK/dk~) dk~ = trapezoid weights in K.  The functional is linear in phi~
    with the derivative moved onto the SMOOTH mollifier (D2^T psi), so the variance
    propagates through bounded weights -- no 1/dk^2 amplification.

    Returns (f_moll, f_lo, f_hi, h).  Resolution ~ h; h = h_factor * median(dK).
    """
    K_row = np.asarray(K_row, dtype=float)
    phi_row = np.asarray(phi_row, dtype=float)
    se2 = np.asarray(se_phi_row, dtype=float) ** 2
    h = float(h_factor) * float(np.median(np.abs(np.diff(K_row))))
    Psi = gaussian_mollifier_bank(K_row, h)               # (n_center, n_k)

    c = float(S0) * np.exp(-float(r) * float(T)) / (float(K_max) ** 2)
    # K-trapezoid weights (so sum_j g_ij phi_j ~ int f psi_i dK)
    w = np.gradient(K_row)                                 # ~ dK per node
    D2 = np.asarray(D2, dtype=float)
    # g_i = c * D2^T (w . psi_i);  G has rows g_i -> F = G @ phi~
    G = c * (Psi * w[None, :]) @ D2                        # (n_center, n_k)
    F = G @ phi_row
    var = (G ** 2) @ se2
    sd = np.sqrt(np.maximum(var, 0.0))
    return F, F - z * sd, F + z * sd, h


# ----------------------------------------------------------------------------
# Dupire-inverse sigma bands
# ----------------------------------------------------------------------------
def sigma_point(phi, grid, D2, T_max, **inv_kwargs):
    """Point-estimate sigma(T,K) via the existing Dupire inverter (NaN where masked)."""
    return implied_sigma_from_phi(phi, grid, D2, T_max, **inv_kwargs)


def _phi_kk(phi, D2):
    return phi @ np.asarray(D2, dtype=float).T            # (D2 @ phi_i) per row


def sigma_band_delta(phi, se_phi, grid, D2, T_max, *, z=1.96, sigma_cap=5.0):
    """Closed-form delta-method sigma band for diagonal input std (cheap, analytic).

    eta~ = num/den,  num = d phi~/d t~ (central diff in t~),  den = k~^2 (D2 phi~).
    num uses time-neighbours at column j; den uses space-neighbours at row i; under
    a diagonal Sigma_phi they are independent, so
        Var[eta~] = Var[num]/den^2 + (num/den^2)^2 Var[den],
        Var[num]_ij = (1/(2 dt))^2 (se_{i+1,j}^2 + se_{i-1,j}^2),
        Var[den]_ij = k~_j^4 sum_m D2_{jm}^2 se_{i,m}^2,
    and sigma = sqrt(2 eta~/T_max) => Var[sigma] = Var[eta~]/(T_max sigma)^2.
    The (num/den^2)^2 term blows up as den = k~^2 phi~_kk -> 0 (the tail pole):
    the calibrated widening.  Returns (sigma_mean, sigma_lo, sigma_hi).
    """
    phi = np.asarray(phi, dtype=float)
    se2 = np.asarray(se_phi, dtype=float) ** 2
    D2 = np.asarray(D2, dtype=float)
    t = np.asarray(grid.t_tilde, dtype=float)
    k = np.asarray(grid.k_tilde, dtype=float)
    n_t, n_k = phi.shape

    phi_kk = _phi_kk(phi, D2)                              # (n_t,n_k)
    den = (k[None, :] ** 2) * phi_kk
    # central time derivative of phi~ (interior rows); one-sided at the ends
    num = np.gradient(phi, t, axis=0)

    # Var[num]: gradient is central interior with spacing (t_{i+1}-t_{i-1}); use a
    # uniform-dt approximation dt = mean spacing for the variance bookkeeping.
    dt = float(np.mean(np.diff(t)))
    var_num = np.zeros_like(phi)
    var_num[1:-1, :] = (1.0 / (2.0 * dt)) ** 2 * (se2[2:, :] + se2[:-2, :])
    var_num[0, :] = (1.0 / dt) ** 2 * (se2[0, :] + se2[1, :])
    var_num[-1, :] = (1.0 / dt) ** 2 * (se2[-1, :] + se2[-2, :])

    # Var[den]_ij = k_j^4 * sum_m D2_jm^2 se_{i,m}^2
    D2sq = D2 ** 2
    var_den = (k[None, :] ** 4) * (se2 @ D2sq.T)

    with np.errstate(divide="ignore", invalid="ignore"):
        eta = num / den
        var_eta = var_num / den ** 2 + (num / den ** 2) ** 2 * var_den
        sigma = np.sqrt(np.clip(2.0 * eta / T_max, 0.0, None))
        var_sigma = var_eta / (T_max * sigma) ** 2
        sd = np.sqrt(np.clip(var_sigma, 0.0, None))

    sigma = np.clip(sigma, 0.0, sigma_cap)
    lo = np.clip(sigma - z * sd, 0.0, sigma_cap)
    hi = np.clip(sigma + z * sd, 0.0, sigma_cap)
    # where den<=0 (no valid convex density) or eta<0, the estimate is infeasible
    bad = ~np.isfinite(sigma) | (den <= 0)
    sigma[bad] = np.nan
    lo[bad] = np.nan
    hi[bad] = np.nan
    return sigma, lo, hi


def sigma_band_mc(phi, se_phi, grid, D2, T_max, *, n_samples=200, seed=0,
                  lo_q=2.5, hi_q=97.5, smooth_k=0.0, **inv_kwargs):
    """Robust sigma band by Monte-Carlo over the input price noise.

    Draws phi^(m) = phi + eps^(m), eps ~ N(0, diag(se^2)), optionally regularizes
    each draw by smoothing along k~ (`smooth_k` grid points), inverts via the real
    Dupire inverter, and returns (median, lo_q, hi_q) percentiles over samples
    (NaN-aware).  With smooth_k=0 the RAW pointwise inverse is ill-posed at realistic
    price noise (bands fail to cover); smooth_k>0 is the calibrated, regularized band
    at resolution ~smooth_k grid points -- the inversion-side analog of the weak-form
    density mollifier.
    """
    phi = np.asarray(phi, dtype=float)
    se = np.asarray(se_phi, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples,) + phi.shape, dtype=float)
    for m in range(n_samples):
        eps = rng.standard_normal(phi.shape) * se
        samples[m] = implied_sigma_from_phi(smooth_phi_k(phi + eps, smooth_k),
                                            grid, D2, T_max, **inv_kwargs)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(samples, axis=0)
        lo = np.nanpercentile(samples, lo_q, axis=0)
        hi = np.nanpercentile(samples, hi_q, axis=0)
    return med, lo, hi


# ----------------------------------------------------------------------------
# Pole-benignity feasibility gate (upgrades the hard tau_kk mask)
# ----------------------------------------------------------------------------
def pole_feasibility(phi, grid, D2, *, tau_kk=1e-3):
    """Per-cell feasibility of the Dupire inverse.

    Returns dict with:
      den         : k~^2 phi~_kk  (the inverse denominator; pole where ->0)
      reliable    : existing |phi~_kk| >= tau_kk*rowmax screen (sigma_reliability_mask)
      convex      : phi~_kk > 0   (valid risk-neutral density / no butterfly arb)
      feasible    : reliable AND convex
    Unlike the hard mask, infeasible cells are FLAGGED (their band carries the
    large, honest uncertainty) rather than discarded.
    """
    phi = np.asarray(phi, dtype=float)
    k = np.asarray(grid.k_tilde, dtype=float)
    phi_kk = _phi_kk(phi, D2)
    den = (k[None, :] ** 2) * phi_kk
    reliable = sigma_reliability_mask(phi, grid, D2, tau_kk=tau_kk)
    convex = phi_kk > 0.0
    return {"den": den, "reliable": reliable, "convex": convex,
            "feasible": reliable & convex}


# ----------------------------------------------------------------------------
# Coverage diagnostic
# ----------------------------------------------------------------------------
def band_coverage(truth, lo, hi, mask=None):
    """Fraction of cells where truth in [lo,hi] (NaN-aware), overall and in `mask`."""
    truth = np.asarray(truth, dtype=float)
    inside = (truth >= lo) & (truth <= hi)
    valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(truth)
    out = {"overall": float(inside[valid].mean()) if valid.any() else float("nan"),
           "n_valid": int(valid.sum())}
    if mask is not None:
        mv = valid & mask
        out["masked"] = float(inside[mv].mean()) if mv.any() else float("nan")
        out["n_masked"] = int(mv.sum())
    return out


# ----------------------------------------------------------------------------
# MC-grounded input covariance (Phase B; dupire_paper / const_vol)
# ----------------------------------------------------------------------------
def input_covariance_from_mc(mc_path, grid, r, S0, *, max_paths=None):
    """Per-(T,K) phi~ standard error from raw MC terminal-price paths.

    Reads data_mc.npz lazily (mmap): keys 'S_matrix' (n_steps,n_paths), 't_all'
    (n_steps,1).  For each grid maturity row i, finds the nearest MC time index and
    computes se(T,K) = std_paths[(S_T-K)^+ e^{-rT}] / (S0 sqrt(N)) over the K-row,
    looping over strikes to bound memory.  WARNING: the MC time grid typically ends
    at T~1.0; rows with grid.T > t_all.max() snap to the last MC step.
    Returns (se_tilde (n_t,n_k), C_tilde (n_t,n_k) MC phi~ point estimate, t_used).
    """
    z = np.load(mc_path, mmap_mode="r")
    S_matrix = z["S_matrix"]
    t_all = np.asarray(z["t_all"]).ravel()
    n_t, n_k = grid.K.shape
    se = np.zeros((n_t, n_k))
    Ct = np.zeros((n_t, n_k))
    t_used = np.zeros(n_t)
    for i in range(n_t):
        Ti = float(grid.T[i])
        idx = int(np.argmin(np.abs(t_all - Ti)))
        t_used[i] = float(t_all[idx])
        S_T = np.asarray(S_matrix[idx, :], dtype=np.float64)
        if max_paths is not None and max_paths < S_T.size:
            S_T = S_T[:max_paths]
        N = S_T.size
        disc = np.exp(-float(r) * t_used[i])
        Krow = grid.K[i]
        for j in range(n_k):
            payoff = np.maximum(S_T - Krow[j], 0.0) * disc
            Ct[i, j] = payoff.mean() / S0
            se[i, j] = payoff.std() / (np.sqrt(N) * S0)
    return se, Ct, t_used


__all__ = [
    "smooth_phi_k",
    "input_covariance_const",
    "input_covariance_from_mc",
    "density_linear_operator",
    "density_band",
    "gaussian_mollifier_bank",
    "weak_form_density_band",
    "sigma_point",
    "sigma_band_delta",
    "sigma_band_mc",
    "pole_feasibility",
    "band_coverage",
]
