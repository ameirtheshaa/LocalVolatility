#!/usr/bin/env python3
r"""
FORWARD-CONSISTENCY TEST for the Dupire local-volatility PINN.
==============================================================

WHAT THIS TESTS
---------------
Calibrating on date t0 gives a local-vol surface sigma_0(tau, S) for tau measured
from t0.  Under the LV model this surface determines the price of EVERY European
option at EVERY future time and spot -- that is the model's strong claim, and it
is *not* tested by anything we have done so far:

  * fit-at-market RMSE   -> in-sample, same date
  * MC reprice RMSE      -> in-sample, same date, and circular (routes through
                            the model's own eta back to the same quotes)
  * K=0 boundary         -> a single structural constraint

Here we instead ask: given the t0-calibrated surface and the *realised* spot on a
later date t1, does the model predict the option prices actually quoted on t1?

    C_pred(K, E) = e^{-r (tau_E - tau_1)} E[ (S_{tau_E} - K)^+ | S_{tau_1} = S_1 ]
    dS = r S dtau + sigma_0(tau, S) S dW      (sigma_0 from the t0 calibration ONLY)

and compare against the quotes observed on t1 at the SAME calendar expiries E.
Nothing from t1 enters the model except the single number S_1.

WHY 7/8/9 AUG WORKS AS A FORWARD TEST WITH NO NEW TRAINING
----------------------------------------------------------
The three DAX files quote the same five calendar expiries with the same strike
sets; their maturities differ by exactly the calendar gap (1 and 2 days).  So
(7aug -> 8aug) and (7aug -> 9aug) and (8aug -> 9aug) are genuine out-of-sample
forward tests at horizons of 1 and 2 days over spot moves of -2.40%, -4.18% and
-1.82%.  Those are large one-day moves, which is exactly what makes the test
discriminating: the benchmarks below disagree materially.

BENCHMARKS (a bare RMSE is meaningless without them)
----------------------------------------------------
  floor    in-sample MC reprice of the t0 model against the t0 quotes.  This is
           the noise/consistency floor of the same MC + eta machinery.  No
           forward prediction can beat it.
  static   C_pred = C_market(t0).  "prices did not move."  Zero-model baseline.
  ss-iv    sticky-STRIKE implied vol: sigma_imp stays put in K, reprice by
           Black-Scholes at the new spot and shorter maturity.
  sm-iv    sticky-MONEYNESS (sticky-delta) implied vol: sigma_imp stays put in
           K/S, reprice by BS at the new spot.  The standard trader heuristic.
  theta    the t0 NN surface read at the shorter maturity but the OLD spot --
           isolates "time decay only, no spot move".
  lvm      DIAGNOSTIC, not a model: the same LV surface re-anchored in moneyness,
           sigma_0(tau, S * S_0/S_1).  Contrasting it with lv isolates how much of
           the forward error is caused by the LV surface being pinned to absolute
           strikes -- i.e. it tests the mechanism, not just the size, of the error.

Hagan et al. (2002) argue an LV model gets the *sign* of smile dynamics wrong
relative to sm-iv, so LV losing to sm-iv would be an expected, documented
failure -- not a bug.  It is reported either way.

WHAT IS COMPARED
----------------
  L1  prices        RMSE / MAE / bias at the shared quotes, overall and per expiry.
  L1b implied vol   the same predictions inverted to BS vol at the t1 state, since
                    price RMSE is dominated by low strikes.
  L2  distribution  P(S_{tau_E} > K): fraction of simulated paths above K, vs the
                    market survival function -e^{r tau} dC/dK from the t1 quotes.
                    The survival function -- not the density -- is used because on
                    this sparse strike grid the quoted call curve is not perfectly
                    convex, so the raw second difference e^{r tau} d2C/dK2 is
                    13-27% negative and cannot serve as ground truth.  The count of
                    non-monotone market points is reported so that limitation is
                    visible rather than hidden.

HONESTY CONTROLS
----------------
  * MC standard error is reported per arm, and the noise-removed RMSE
    sqrt(RMSE^2 - mean SE^2), so "better" is only claimed above MC noise.
  * The in-sample floor's BIAS is reported as a control: if the model were already
    biased on its own date, a forward bias would say nothing about dynamics.
  * Common random numbers across every MC arm (same seed, same dW), so lv vs lvm is
    an exactly paired comparison on identical Brownian paths.
  * A martingale check E[S_tau] vs S_1 e^{r(tau-tau_1)} quantifies Euler bias.
  * Strikes needing extrapolation of the t0 IV curve, quotes that fail IV
    inversion, and deep-OTM strikes with no simulated path above K are all counted
    and printed rather than silently dropped.

Run:  python3 forward_test.py                 # M=1e5 paths (headline)
      python3 forward_test.py --m 4000        # fast smoke
"""
import os
import sys
import json
import time
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                       # nn_boundary_deficit_results
# FWD_MODELS / FWD_DATA let the same file run unchanged on an offload backend,
# where the models and CSVs land in a flat staging directory.
MODELS = os.environ.get("FWD_MODELS", os.path.join(BASE, "dax_expk", "models"))
DATA = os.environ.get("FWD_DATA", os.path.join(BASE, "dax_expk", "data"))

SEED = 42
KMAX = 10000.0
R = 0.04
DT = 1e-3
N_T = 1000

# Tmax is NOT hardcoded: the pipeline sets T_max = max quoted maturity, so it is
# derived from each file (verified to reproduce 0.871 / 0.868 / 0.866 for 7/8/9 Aug).
# model=None means "no calibrated model yet" -- usable as a forward TARGET only.
DAYS = {
    "7may": dict(date="2001-05-07", S0=6122.62, csv="7_5_2001_dataTrain.csv", model=None),
    "8jun": dict(date="2001-06-08", S0=6187.21, csv="8_6_2001_dataTrain.csv", model="fwd_expk_8_6_2001"),
    "4jul": dict(date="2001-07-04", S0=6015.72, csv="4_7_2001_dataTrain.csv", model="fwd_expk_4_7_2001"),
    "6jul": dict(date="2001-07-06", S0=5862.10, csv="6_7_2001_dataTrain.csv", model=None),
    "7aug": dict(date="2001-08-07", S0=5752.51, csv="dataTrain_7_August_2001.csv", model="nbexpk7"),
    "8aug": dict(date="2001-08-08", S0=5614.51, csv="dataTrain_8_August_2001.csv", model="nbexpk8"),
    "9aug": dict(date="2001-08-09", S0=5512.28, csv="dataTrain_9_August_2001.csv", model="nbexpk9"),
}
PAIRS = [("7aug", "8aug"), ("7aug", "9aug"), ("8aug", "9aug")]
LONG_PAIRS = [("4jul", "7aug"), ("4jul", "8aug"), ("4jul", "9aug"),
              ("8jun", "7aug"), ("8jun", "8aug"), ("8jun", "9aug")]
_TMAX_CACHE = {}


# ============================================================
# model builders / pricing -- identical to the validated build_overlay.py
# ============================================================
def _resblock(t, u=64, act="tanh"):
    d = tf.keras.layers.Dense(u, use_bias=False)(t)
    b = tf.keras.layers.BatchNormalization()(d)
    a = tf.keras.layers.Activation(act)(b)
    d = tf.keras.layers.Dense(u, use_bias=False)(a)
    b = tf.keras.layers.BatchNormalization()(d)
    a = tf.keras.layers.Activation(act)(b)
    return tf.keras.layers.Add()([t, a])


def _build_arch(out_act):
    inp = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.GaussianNoise(0.5)(inp)
    x = tf.keras.layers.Dense(64, activation="tanh", use_bias=False)(x)
    for _ in range(3):
        x = _resblock(x)
    x = tf.keras.layers.Dense(64, activation="tanh", use_bias=True)(x)
    o = tf.keras.layers.Dense(1, activation=out_act, use_bias=True, dtype="float32")(x)
    return tf.keras.models.Model(inp, o)


def load_models(day):
    d = os.path.join(MODELS, DAYS[day]["model"])
    phi = _build_arch(None)
    phi.load_weights(os.path.join(d, "NN_phi_final.keras"))
    eta = _build_arch("softplus")
    eta.load_weights(os.path.join(d, "NN_eta_final.keras"))
    return phi, eta


def _col(T_arr, K_arr):
    T = tf.reshape(tf.cast(tf.convert_to_tensor(T_arr), tf.float32), [-1, 1])
    K = tf.reshape(tf.cast(tf.convert_to_tensor(K_arr), tf.float32), [-1, 1])
    return T, K


def rescale(T_arr, K_arr, Tmax):
    T, K = _col(T_arr, K_arr)
    k = tf.exp(-R * T) * K / KMAX
    t = tf.broadcast_to(T / Tmax, tf.shape(k))
    return t, k


def price_expk(phi, S0, Tmax, T_arr, K_arr):
    """phi~ = exp(-k~ * softplus(a + k~*raw)),  a = y + log1p(-exp(-y)), y=Kmax/S0."""
    t, k = rescale(T_arr, K_arr, Tmax)
    raw = phi(tf.concat([t, k], axis=1), training=False)
    y = KMAX / S0
    a = y + np.log1p(-np.exp(-y))
    return S0 * tf.exp(-k * tf.nn.softplus(a + k * raw))


def sigma_of(eta, Tmax, T_arr, K_arr):
    t, k = rescale(T_arr, K_arr, Tmax)
    return tf.sqrt(2.0 * eta(tf.concat([t, k], axis=1), training=False) / Tmax)


# ============================================================
# market data
# ============================================================
def load_quotes(day, opt_type=1):
    df = pd.read_csv(os.path.join(DATA, DAYS[day]["csv"]))
    df.columns = [str(c).replace(chr(10), " ").strip() for c in df.columns]
    tc = [c for c in df.columns if "type" in c.lower()][0]
    pc = [c for c in df.columns if "price" in c.lower()][0]
    d = df[df[tc] == opt_type][["Maturity", "Strike", pc]].copy()
    d.columns = ["T", "K", "P"]
    return d.astype(float).sort_values(["T", "K"]).reset_index(drop=True)


def tmax_of(day):
    """T_max = max quoted call maturity, exactly as the training pipeline sets it."""
    if day not in _TMAX_CACHE:
        _TMAX_CACHE[day] = float(load_quotes(day)["T"].max())
    return _TMAX_CACHE[day]


def calendar_gap(base, target):
    """dtau in years between the two quote dates (ACT/365.25, matching the files)."""
    import datetime as dt
    d0 = dt.date.fromisoformat(DAYS[base]["date"])
    d1 = dt.date.fromisoformat(DAYS[target]["date"])
    assert d1 > d0, f"{target} must be after {base}"
    return (d1 - d0).days / 365.25, (d1 - d0).days


def match_expiries(q0, q1, dtau, tol=0.005):
    """Pair the two dates' maturities by CALENDAR EXPIRY.

    Expiry E seen from t0 sits at tau = T0; seen from t1 at T1 = T0 - dtau.  So a
    shared expiry is any (T0, T1) with T0 - T1 ~= dtau.  Maturities are rounded to
    3 dp in the files (+-0.0005 each, so +-0.001 on the difference) while adjacent
    expiries are >= 0.03 apart, making tol=0.005 unambiguous.  Each T0 is used at
    most once.  Expiries quoted on only one of the two dates simply drop out.
    """
    m0 = sorted(float(x) for x in q0["T"].unique())
    m1 = sorted(float(x) for x in q1["T"].unique())
    pairs, used = [], set()
    for b in m1:
        cand = [a for a in m0 if a not in used and abs(a - b - dtau) < tol]
        if cand:
            a = min(cand, key=lambda x: abs(x - b - dtau))
            used.add(a)
            pairs.append((a, b))
    assert pairs, (f"no shared expiries: t0 maturities {m0}, t1 maturities {m1}, "
                   f"dtau={dtau:.4f}")
    resid = [abs(a - b - dtau) for a, b in pairs]
    return pairs, float(np.mean(resid)), float(np.max(resid))


# ============================================================
# Black-Scholes + implied vol (r, q=0; forward F = S e^{r tau})
# ============================================================
def bs_call(S, K, tau, sig, r=R):
    """Black-Scholes call, q=0. Branch-free so it works for scalars and arrays."""
    S, K, tau, sig = np.broadcast_arrays(*(np.asarray(v, float) for v in (S, K, tau, sig)))
    v = np.where((sig > 0) & (tau > 0), sig * np.sqrt(np.maximum(tau, 0.0)), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * tau) / v
        bs = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d1 - v)
    intrinsic = np.maximum(S - K * np.exp(-r * tau), 0.0)
    return np.where(np.isfinite(bs), bs, intrinsic)


def implied_vol(P, S, K, tau, r=R):
    """Brent inversion. NaN when the quote is outside the no-arbitrage bounds."""
    lo = max(S - K * np.exp(-r * tau), 0.0)
    if not (lo + 1e-8 < P < S - 1e-8):
        return np.nan
    f = lambda s: float(bs_call(S, K, tau, s, r)) - P
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-10, maxiter=200)
    except ValueError:
        return np.nan


# ============================================================
# MC: Euler-Maruyama from an arbitrary start (tau_start, S_start).
# Memory-light: only the requested expiry rows are retained.
# Common random numbers: dW is generated once per (M, n_steps, seed).
# ============================================================
def _dW(M, n, seed, dt=DT, antithetic=True):
    """Brownian increments.  With antithetic=True the second half of the paths is
    the exact mirror of the first, which is unbiased for any payoff and cuts the
    variance of the mean substantially.  M must then be even."""
    rng = np.random.default_rng(seed)
    if antithetic:
        assert M % 2 == 0, "antithetic sampling needs an even path count"
        h = rng.standard_normal((n, M // 2))
        z = np.concatenate([h, -h], axis=1)
    else:
        z = rng.standard_normal((n, M))
    return (z * np.sqrt(dt)).astype(np.float32)


def mc_paths(eta, Tmax, S_start, tau_start, tau_targets, M, seed=SEED, dt=DT, dW=None,
             k_scale=1.0, keep_every=None):
    """Simulate dS = rS dtau + sigma(tau,S) S dW from (tau_start, S_start).

    sigma is evaluated at the RUNNING time tau and the RUNNING level S -- i.e.
    exactly the Dupire LV identification, as in the validated reprice engine.

    k_scale is a DIAGNOSTIC knob, not part of the model.  With k_scale = S_0/S_1 the
    surface is read at sigma_0(tau, S * S_0/S_1), i.e. re-anchored in MONEYNESS
    rather than absolute strike, so a path sitting at the new spot sees the
    calibration date's at-the-money vol.  Comparing k_scale=1 (true Dupire LV) with
    k_scale=S_0/S_1 isolates how much of the forward error is caused by the LV
    surface being pinned to absolute strikes.

    keep_every retains the WHOLE path every keep_every steps, in addition to the expiry
    snapshots -- needed for anything that is a functional of the path rather than of the
    terminal value (occupation measures, per-path statistics).  Default None keeps the
    original memory-light behaviour, so existing callers are unaffected.  The retained
    array lands in diag["path_S"], shape (M, n_kept), with times in diag["path_tau"];
    keep_every = round(1/(252*dt)) is one sample per trading day.

    Returns {tau_target: S_samples[M]} plus diagnostics.
    """
    steps = [int(round((tt - tau_start) / dt)) for tt in tau_targets]
    n = max(steps)
    assert n <= N_T, f"need {n} steps > N_T={N_T}"
    if dW is None:
        dW = _dW(M, n, seed, dt)
    tau_grid = tau_start + np.arange(n + 1, dtype=np.float64) * dt

    ks = tf.constant(float(k_scale), tf.float32)

    @tf.function(reduce_retracing=True)
    def step(t_now, S_now, dW_now):
        sig = sigma_of(eta, Tmax, t_now, S_now * ks)
        return S_now + R * S_now * dt + sig * S_now * dW_now

    S = tf.constant(np.full((M, 1), S_start, dtype=np.float32))
    want = {s: tt for s, tt in zip(steps, tau_targets)}
    out, t0 = {}, time.time()
    if 0 in want:
        out[want[0]] = S.numpy().flatten().copy()
    keep_S, keep_j = ([S.numpy().flatten().copy()], [0]) if keep_every else (None, None)
    for j in range(n):
        S = step(tf.constant(tau_grid[j], tf.float32),
                 S, tf.constant(dW[j].reshape(-1, 1), tf.float32))
        if (j + 1) in want:
            out[want[j + 1]] = S.numpy().flatten().copy()
        if keep_every and (j + 1) % keep_every == 0:
            keep_S.append(S.numpy().flatten().copy())
            keep_j.append(j + 1)
    diag = {}
    for tt, Ss in out.items():
        fwd = S_start * np.exp(R * (tt - tau_start))
        diag[round(float(tt), 6)] = dict(
            mean=float(Ss.mean()), forward=float(fwd),
            mart_err_pct=float(100 * (Ss.mean() - fwd) / fwd))
    extra = {}
    if keep_every:
        extra = dict(path_S=np.stack(keep_S, axis=1).astype(np.float32),
                     path_tau=tau_grid[np.asarray(keep_j)])
    return out, dict(steps=steps, n_steps=n, wall_s=time.time() - t0, martingale=diag,
                     **extra)


def mc_price(S_samples, K, tau_pay, r=R, antithetic=True):
    """C = e^{-r tau_pay} E[(S-K)^+] and its MC standard error.

    Under antithetic sampling the i-th and (i+M/2)-th paths are NOT independent,
    so the SE must be taken over the M/2 *pair averages* -- using the raw
    per-path std would overstate it.
    """
    pay = np.maximum(S_samples.reshape(-1, 1) - np.asarray(K).reshape(1, -1), 0.0)
    disc = np.exp(-r * tau_pay)
    if antithetic:
        h = pay.shape[0] // 2
        unit = 0.5 * (pay[:h] + pay[h:2 * h])       # one i.i.d. draw per pair
    else:
        unit = pay
    n = unit.shape[0]
    return disc * unit.mean(axis=0), disc * unit.std(axis=0, ddof=1) / np.sqrt(n)


def mc_survival(S_samples, K, antithetic=True):
    """P(S > K) and its MC standard error, with the same antithetic pairing."""
    ind = (S_samples.reshape(-1, 1) > np.asarray(K).reshape(1, -1)).astype(np.float64)
    if antithetic:
        h = ind.shape[0] // 2
        unit = 0.5 * (ind[:h] + ind[h:2 * h])
    else:
        unit = ind
    n = unit.shape[0]
    return unit.mean(axis=0), unit.std(axis=0, ddof=1) / np.sqrt(n)


# ============================================================
# benchmarks
# ============================================================
def iv_curve(q0, T0, S0):
    """Implied vols of the t0 quotes at expiry T0, sorted by strike."""
    g = q0[np.isclose(q0["T"], T0)].sort_values("K")
    K = g["K"].to_numpy(float)
    iv = np.array([implied_vol(p, S0, k, T0) for p, k in zip(g["P"], K)])
    ok = np.isfinite(iv)
    return K[ok], iv[ok], g["P"].to_numpy(float)[ok], int((~ok).sum())


def interp_flag(x, xp, fp):
    """Linear interp in xp; returns (values, n_extrapolated) with flat extension."""
    v = np.interp(x, xp, fp)
    n_ex = int(np.sum((x < xp[0]) | (x > xp[-1])))
    return v, n_ex


# ============================================================
# main per-pair evaluation
# ============================================================
def run_pair(base, target, M, seed=SEED, verbose=True):
    cb, ct = DAYS[base], DAYS[target]
    if cb["model"] is None:
        raise SystemExit(f"no calibrated model for base date {base}")
    S0, S1 = cb["S0"], ct["S0"]
    Tmax0 = tmax_of(base)
    q0, q1 = load_quotes(base), load_quotes(target)
    dtau, gap_days = calendar_gap(base, target)
    expiry_pairs, res_mean, res_max = match_expiries(q0, q1, dtau)
    phi0, eta0 = load_models(base)

    if verbose:
        print("=" * 78)
        print(f"FORWARD TEST  {base} ({cb['date']})  ->  {target} ({ct['date']})")
        print(f"  S({base}) = {S0:.2f}   S({target}) = {S1:.2f}   "
              f"move = {100*(S1/S0-1):+.2f}%")
        print(f"  horizon = {gap_days} calendar days = {dtau:.5f} yr    "
              f"T_max({base}) = {Tmax0:.3f} (from data)")
        print(f"  {len(expiry_pairs)} shared calendar expiries of "
              f"{len(q0['T'].unique())}({base}) x {len(q1['T'].unique())}({target}); "
              f"expiry-match residual mean {res_mean:.4f} max {res_max:.4f} yr")
        for a, b in expiry_pairs:
            print(f"      tau {a:.3f} (from {base})  ->  {b:.3f} (from {target})   "
                  f"({a*365.25:6.1f} d -> {b*365.25:6.1f} d)")
        print()

    # --- align the MC time axis to the t0 clock -------------------------------
    # tau is measured from t0.  Expiry E sits at tau_E = T0 (the t0 maturity).
    # The forward MC starts at tau_1 = dtau with S = S1.  Snap tau_E onto the
    # dt grid started at dtau so that (tau_E - dtau)/dt is an exact integer.
    tau_E = [a for a, _ in expiry_pairs]
    steps_fwd = [int(round((a - dtau) / DT)) for a in tau_E]
    tau_E_snap = [dtau + s * DT for s in steps_fwd]
    tau_pay = [s * DT for s in steps_fwd]          # time from t1 to expiry
    n_fwd = max(steps_fwd)

    # common random numbers shared by the forward arm and the in-sample floor
    dW_common = _dW(M, max(n_fwd, max(int(round(a / DT)) for a in tau_E)), seed)

    # --- arm 1: FORWARD prediction (the actual test) -------------------------
    fwd_S, fwd_diag = mc_paths(eta0, Tmax0, S1, dtau, tau_E_snap, M,
                               dW=dW_common[:n_fwd])
    # --- arm 1b: DIAGNOSTIC -- same LV surface re-anchored in moneyness -------
    fwdm_S, _ = mc_paths(eta0, Tmax0, S1, dtau, tau_E_snap, M,
                         dW=dW_common[:n_fwd], k_scale=S0 / S1)
    # --- arm 2: in-sample FLOOR (same machinery, t0 spot, t0 quotes) ---------
    tau_E0_steps = [int(round(a / DT)) for a in tau_E]
    tau_E0_snap = [s * DT for s in tau_E0_steps]
    ck = (base, M, seed)
    if ck not in _FLOOR_CACHE:                       # identical for both 7aug pairs
        _FLOOR_CACHE[ck] = mc_paths(eta0, Tmax0, S0, 0.0, tau_E0_snap, M,
                                    dW=dW_common[:max(tau_E0_steps)])
    ins_S, ins_diag = _FLOOR_CACHE[ck]
    ins_key = dict(zip(tau_E, tau_E0_snap))   # t0 maturity -> snapped MC grid time

    rows = []
    for (T0, T1), tE, tp in zip(expiry_pairs, tau_E_snap, tau_pay):
        g1 = q1[np.isclose(q1["T"], T1)].sort_values("K")
        K = g1["K"].to_numpy(float)
        Pm = g1["P"].to_numpy(float)                       # ACTUAL t1 quotes

        # --- LV forward ---
        C_lv, se_lv = mc_price(fwd_S[tE], K, tp)
        C_lvm, se_lvm = mc_price(fwdm_S[tE], K, tp)      # moneyness-re-anchored

        # --- static: t0 price at the same strike & expiry ---
        g0 = q0[np.isclose(q0["T"], T0)].sort_values("K")
        C_static, n_ex_st = interp_flag(K, g0["K"].to_numpy(float), g0["P"].to_numpy(float))

        # --- IV benchmarks off the t0 smile ---
        K0, iv0, _, n_bad = iv_curve(q0, T0, S0)
        lm0 = np.log(K0 / S0)                              # t0 log-moneyness
        iv_ss, n_ex_ss = interp_flag(np.log(K / S0), lm0, iv0)   # sticky strike
        iv_sm, n_ex_sm = interp_flag(np.log(K / S1), lm0, iv0)   # sticky moneyness
        C_ss = bs_call(S1, K, T1, iv_ss)
        C_sm = bs_call(S1, K, T1, iv_sm)

        # --- theta-only: t0 NN surface at the shorter maturity, OLD spot ---
        C_theta = price_expk(phi0, S0, Tmax0, np.full_like(K, T1), K).numpy().flatten()

        # --- in-sample floor: t0 model repriced against t0 quotes ---
        C_ins, se_ins = mc_price(ins_S[ins_key[T0]], g0["K"].to_numpy(float), T0)
        floor_err = C_ins - g0["P"].to_numpy(float)
        floor_rmse = float(np.sqrt(np.mean(floor_err ** 2)))
        floor_bias = float(np.mean(floor_err))
        floor_se2 = float(np.mean(se_ins ** 2))

        # --- implied-vol view: price RMSE is dominated by ATM/low strikes, IV is
        #     the metric a practitioner reads.  Inverted at the TARGET state.
        iv_mkt = np.array([implied_vol(p, S1, k, T1) for p, k in zip(Pm, K)])
        iv_arm = {}
        for key, C in (("lv", C_lv), ("lvm", C_lvm), ("sm", C_sm), ("ss", C_ss),
                       ("theta", C_theta), ("static", C_static)):
            iv_arm[key] = np.array([implied_vol(p, S1, k, T1) for p, k in zip(C, K)])

        rows.append(dict(
            T0=float(T0), T1=float(T1), n=len(K), K=K, market=Pm,
            lv=C_lv, lv_se=se_lv, lvm=C_lvm, lvm_se=se_lvm,
            static=C_static, ss=C_ss, sm=C_sm, theta=C_theta,
            floor_rmse=floor_rmse, floor_bias=floor_bias, floor_se2=floor_se2, n_iv_fail=n_bad,
            iv0_K=K0, iv0=iv0, iv_ss=iv_ss, iv_sm=iv_sm,
            iv_mkt=iv_mkt, iv_arm=iv_arm,
            n_mc_zero=int(np.sum(C_lv <= 0.0)),
            n_extrap=dict(static=n_ex_st, ss=n_ex_ss, sm=n_ex_sm),
            fwd_S=fwd_S[tE],
        ))
    return dict(base=base, target=target, S0=S0, S1=S1, dtau=dtau,
                gap_days=gap_days, res_mean=res_mean, res_max=res_max,
                rows=rows, fwd_diag=fwd_diag, ins_diag=ins_diag, M=M)


_FLOOR_CACHE = {}   # (base_day, M, seed) -> in-sample MC; 7aug is base for 2 pairs

ARMS = [("lv", "LV forward (model)"),
        ("lvm", "LV, vol re-anchored in K/S"),      # diagnostic, not the model
        ("sm", "sticky-moneyness IV"),
        ("ss", "sticky-strike IV"), ("theta", "theta only (old spot)"),
        ("static", "static (no change)")]


def summarise(res, verbose=True):
    rows = res["rows"]
    allm = np.concatenate([r["market"] for r in rows])
    stats, per_exp = {}, {}
    for key, lab in ARMS:
        pred = np.concatenate([r[key] for r in rows])
        e = pred - allm
        stats[key] = dict(label=lab, rmse=float(np.sqrt(np.mean(e ** 2))),
                          mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                          max_abs=float(np.max(np.abs(e))))
        per_exp[key] = [float(np.sqrt(np.mean((r[key] - r["market"]) ** 2))) for r in rows]
    # MC noise: subtract it in quadrature to get the deterministic (model) error.
    # E[RMSE^2] = model_err^2 + mean(SE^2), so rmse_deb = sqrt(RMSE^2 - mean(SE^2)).
    se2 = float(np.mean(np.concatenate([r["lv_se"] for r in rows]) ** 2))
    stats["lv"]["mc_se_rms"] = float(np.sqrt(se2))
    stats["lv"]["rmse_denoised"] = float(np.sqrt(max(stats["lv"]["rmse"] ** 2 - se2, 0.0)))
    w = np.concatenate([np.full(r["n"], 1.0) for r in rows])
    floor = float(np.sqrt(np.average(
        np.concatenate([np.full(r["n"], r["floor_rmse"] ** 2) for r in rows]), weights=w)))
    floor_se2 = float(np.average(
        np.concatenate([np.full(r["n"], r["floor_se2"]) for r in rows]), weights=w))
    floor_deb = float(np.sqrt(max(floor ** 2 - floor_se2, 0.0)))
    # CONTROL: is the model already biased on its OWN date?  If the in-sample bias were
    # comparable to the forward bias, the forward result would say nothing about dynamics.
    floor_bias = float(np.average(
        np.concatenate([np.full(r["n"], r["floor_bias"]) for r in rows]), weights=w))

    if verbose:
        print(f"  --- L1  PRICES   ({len(allm)} quotes on {res['target']}, "
              f"mean price {allm.mean():.1f}) ---")
        print(f"  {'arm':<26} {'RMSE':>8} {'MAE':>8} {'bias':>8} {'maxabs':>8}   per-expiry RMSE")
        for key, lab in ARMS:
            s = stats[key]
            pe = " ".join(f"{v:6.1f}" for v in per_exp[key])
            print(f"  {lab:<26} {s['rmse']:8.2f} {s['mae']:8.2f} "
                  f"{s['bias']:+8.2f} {s['max_abs']:8.1f}   {pe}")
        print(f"  {'[in-sample floor]':<26} {floor:8.2f} {'':>8} {floor_bias:+8.2f}"
              f"           <- CONTROL: same MC+eta, t0 spot vs t0 quotes")
        print(f"  MC standard error (rms over quotes) = {stats['lv']['mc_se_rms']:.2f}"
              f"   (M={res['M']:,} paths, antithetic)")
        print(f"  noise-removed:  LV forward = {stats['lv']['rmse_denoised']:.2f}"
              f"   in-sample floor = {floor_deb:.2f}"
              f"   [sqrt(RMSE^2 - mean SE^2)]")
        mart = res["fwd_diag"]["martingale"]
        worst = max(abs(v["mart_err_pct"]) for v in mart.values())
        print(f"  Euler/martingale check: max |E[S_tau]/F - 1| = {worst:.3f}%")
        nz = sum(r["n_mc_zero"] for r in rows)
        if nz:
            print(f"  note: {nz}/{len(allm)} deep-OTM quotes have NO simulated path "
                  f"above K (MC price exactly 0); their market prices are "
                  f"{np.concatenate([r['market'][r['lv'] <= 0] for r in rows]).max():.2f} "
                  f"or less, so their RMSE weight is negligible")

    # ---- implied-vol view (vol points) ----
    ivm = np.concatenate([r["iv_mkt"] for r in rows])
    iv_stats = {}
    for key, lab in ARMS:
        iva = np.concatenate([r["iv_arm"][key] for r in rows])
        ok = np.isfinite(iva) & np.isfinite(ivm)
        e = 100.0 * (iva[ok] - ivm[ok])
        iv_stats[key] = dict(label=lab, n=int(ok.sum()), n_fail=int((~ok).sum()),
                             rmse_volpts=float(np.sqrt(np.mean(e ** 2))),
                             mae_volpts=float(np.mean(np.abs(e))),
                             bias_volpts=float(np.mean(e)))
    if verbose:
        print(f"  --- L1b IMPLIED VOL  (vol points; inverted at the {res['target']} "
              f"state; {int(np.isfinite(ivm).sum())}/{len(ivm)} market quotes invertible) ---")
        print(f"  {'arm':<26} {'RMSE':>8} {'MAE':>8} {'bias':>8} {'n':>5} {'n_fail':>7}")
        for key, lab in ARMS:
            s = iv_stats[key]
            print(f"  {lab:<26} {s['rmse_volpts']:8.2f} {s['mae_volpts']:8.2f} "
                  f"{s['bias_volpts']:+8.2f} {s['n']:5d} {s['n_fail']:7d}")

    return dict(stats=stats, per_expiry=per_exp, floor_rmse=floor,
                floor_rmse_denoised=floor_deb, floor_bias=floor_bias, iv_stats=iv_stats,
                n_quotes=int(len(allm)), mean_price=float(allm.mean()))


def cdf_check(res, verbose=True):
    """L2: model survival function vs the market's, from the t1 quotes.

    P(S_tau > K) = -e^{r tau} dC/dK.  Central differences on the quoted strikes;
    monotone and far more stable than the second difference.
    """
    out = []
    for r in res["rows"]:
        K, Pm, tau = r["K"], r["market"], r["T1"]
        # market survival function from central differences of the t1 quotes
        dCdK = np.gradient(Pm, K)
        surv_mkt = np.clip(-np.exp(R * tau) * dCdK, 0.0, 1.0)
        surv_mod, se = mc_survival(r["fwd_S"], K)
        m = np.isfinite(surv_mkt) & np.isfinite(surv_mod)
        out.append(dict(T1=float(tau), n=int(m.sum()),
                        corr=float(np.corrcoef(surv_mkt[m], surv_mod[m])[0, 1]),
                        rms_pp=float(100 * np.sqrt(np.mean((surv_mkt[m] - surv_mod[m]) ** 2))),
                        max_pp=float(100 * np.max(np.abs(surv_mkt[m] - surv_mod[m]))),
                        mc_se_pp=float(100 * np.sqrt(np.mean(se ** 2))),
                        n_nonmono=int(np.sum(np.diff(surv_mkt[m]) > 1e-9))))
    if verbose:
        print(f"  --- L2  RISK-NEUTRAL SURVIVAL FUNCTION  P(S>K) ---")
        print(f"  {'T1':>7} {'n':>4} {'corr':>7} {'RMS(pp)':>9} {'max(pp)':>9} "
              f"{'MC se(pp)':>10} {'mkt non-mono':>13}")
        for d in out:
            print(f"  {d['T1']:7.3f} {d['n']:4d} {d['corr']:7.4f} {d['rms_pp']:9.2f} "
                  f"{d['max_pp']:9.2f} {d['mc_se_pp']:10.3f} {d['n_nonmono']:13d}")
    return out


def make_figure(res, L2, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["rows"]
    n = len(rows)
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 10.2), facecolor="white")
    if n == 1:
        axes = axes.reshape(3, 1)
    COL = dict(lv="#d62728", lvm="#ff9d3a", sm="#1f77b4", ss="#2ca02c", static="#888888")
    LBL = dict(lv="LV forward", lvm="LV re-anchored $K/S$", sm="sticky-moneyness",
               ss="sticky-strike", static="static")

    for j, r in enumerate(rows):
        K, Pm, S1 = r["K"], r["market"], res["S1"]
        # ---- row 0: predicted vs actual price ----
        ax = axes[0, j]
        ax.plot(K, Pm, "k.", ms=5, label=f"{res['target']} quotes (actual)")
        for key in ("lv", "lvm", "sm", "ss"):
            ax.plot(K, r[key], "-", color=COL[key], lw=1.4, label=LBL[key])
        ax.axvline(S1, color="0.7", ls=":", lw=0.9)
        ax.set_yscale("log")
        ax.set_title(f"$\\tau$ = {r['T1']:.3f}  ({r['T1']*365.25:.0f} d)", fontsize=10)
        ax.grid(alpha=0.25)
        if j == 0:
            ax.set_ylabel("call price (log scale)")
            ax.legend(fontsize=7)

        # ---- row 1: signed pricing error ----
        ax = axes[1, j]
        for key in ("static", "sm", "ss", "lvm", "lv"):
            ax.plot(K, r[key] - Pm, "-", color=COL[key], lw=1.5 if key == "lv" else 1.1,
                    label=LBL[key])
        ax.fill_between(K, -2 * r["lv_se"], 2 * r["lv_se"], color=COL["lv"], alpha=0.18,
                        lw=0, label="$\\pm2\\,$MC s.e.")
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(S1, color="0.7", ls=":", lw=0.9)
        ax.grid(alpha=0.25)
        if j == 0:
            ax.set_ylabel("predicted $-$ actual")
            ax.legend(fontsize=7)

        # ---- row 2: risk-neutral survival function ----
        ax = axes[2, j]
        surv_mod, se = mc_survival(r["fwd_S"], K)
        surv_mkt = np.clip(-np.exp(R * r["T1"]) * np.gradient(Pm, K), 0.0, 1.0)
        ax.plot(K, surv_mkt, "k.", ms=5, label="market $-e^{r\\tau}\\partial_K C$")
        ax.plot(K, surv_mod, "-", color=COL["lv"], lw=1.5, label="LV forward MC")
        ax.axvline(S1, color="0.7", ls=":", lw=0.9)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("strike $K$")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.set_ylabel("$P(S_\\tau > K)$")
            ax.legend(fontsize=7)
        ax.text(0.03, 0.06, f"RMS {L2[j]['rms_pp']:.1f} pp\ncorr {L2[j]['corr']:.4f}",
                transform=ax.transAxes, fontsize=7.5, va="bottom")

    fig.suptitle(
        f"FORWARD test  {res['base']} $\\to$ {res['target']}   "
        f"$S$: {res['S0']:.0f} $\\to$ {res['S1']:.0f} ({100*(res['S1']/res['S0']-1):+.2f}%), "
        f"horizon {res['dtau']*365.25:.0f} d, M={res['M']:,} antithetic paths\n"
        f"the LV surface is calibrated on {res['base']} ONLY; the single number "
        f"$S_1$={res['S1']:.2f} is the only {res['target']} input",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out_dir, f"forward_{res['base']}_to_{res['target']}.png")
    fig.savefig(p, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=100000)
    ap.add_argument("--pairs", default="all",
                    help="'all' (1-2 day August pairs), 'long' (multi-week), "
                         "'every', or e.g. '4jul>7aug,8jun>9aug'")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    if a.pairs == "all":
        pairs = PAIRS
    elif a.pairs == "long":
        pairs = LONG_PAIRS
    elif a.pairs == "every":
        pairs = PAIRS + LONG_PAIRS
    else:
        pairs = [tuple(p.split(">")) for p in a.pairs.split(",")]
    pairs = [p for p in pairs if DAYS[p[0]]["model"] is not None
             and os.path.isdir(os.path.join(MODELS, DAYS[p[0]]["model"]))]
    print(f"pairs to run: {['->'.join(p) for p in pairs]}\n")

    t0 = time.time()
    payload = {}
    for b, t in pairs:
        res = run_pair(b, t, a.m)
        s = summarise(res)
        c = cdf_check(res)
        make_figure(res, c, a.out)
        print()
        payload[f"{b}->{t}"] = dict(
            base=b, target=t, base_date=DAYS[b]["date"], target_date=DAYS[t]["date"],
            S_base=res["S0"], S_target=res["S1"],
            spot_move_pct=100 * (res["S1"] / res["S0"] - 1),
            dtau_yr=res["dtau"], dtau_days=res["gap_days"],
            expiry_match_resid_mean=res["res_mean"], expiry_match_resid_max=res["res_max"],
            n_mc_zero=[r["n_mc_zero"] for r in res["rows"]],
            M=a.m, L1=s, L2=c,
            martingale=res["fwd_diag"]["martingale"],
            mc_wall_s=res["fwd_diag"]["wall_s"] + res["ins_diag"]["wall_s"],
            n_extrap=[r["n_extrap"] for r in res["rows"]],
            n_iv_fail=[r["n_iv_fail"] for r in res["rows"]],
            expiries=[dict(T_base=r["T0"], T_target=r["T1"], n=r["n"]) for r in res["rows"]],
        )
        np.savez_compressed(
            os.path.join(a.out, f"raw_{b}_to_{t}.npz"),
            **{f"{k}_{i}": r[k] for i, r in enumerate(res["rows"])
               for k in ("K", "market", "lv", "lv_se", "lvm", "static", "ss", "sm", "theta")},
            **{f"fwdS_{i}": r["fwd_S"][::20] for i, r in enumerate(res["rows"])})

    with open(os.path.join(a.out, "forward_test_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {os.path.join(a.out, 'forward_test_results.json')}")
    print(f"total wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
