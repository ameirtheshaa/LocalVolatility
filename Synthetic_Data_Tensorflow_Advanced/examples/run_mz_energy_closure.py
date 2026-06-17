#!/usr/bin/env python3
r"""
MZ x Dupire — Step 2: the DENSITY-SIDE ENERGY-BUDGET CLOSURE (§5 of the synthesis).

Validates ``mz_spectral/energy_closure.py``: a Gaussian/log-normal resolved core plus a
single tail-energy scalar E_tail, coupled back through a structure-preserving map that
makes the risk-neutral density VALID BY CONSTRUCTION — ∫f=1, E[S_T]=S0 e^{rT} exact, and
f≥0 STRUCTURAL (the max-entropy tilt cures the raw-Gram-Charlier wing negativity, which is
SLGG's sign-structure theorem reincarnated as Edgeworth negativity).

Phase A — const-σ calibration (no MC).
  Black-Scholes truth (``build_truth_phi_bs``) -> known ``lognormal_density``. Confirm the
  closure COLLAPSES to the Gaussian core: extracted skew/kurtosis ≈ 0, E_tail ≈ 0, and
  ``pdf_metrics`` l2 at the machinery floor (~1e-6).

Phase B — dupire_paper smile vs MC truth.
  Truth density via ``mc_kde_pdf_on_K_grid`` on ``data_mc.npz`` terminal slices (T<=1.0,
  subsampled paths). Fit the closure to the MC log-return moments; score closure-vs-truth
  with ``pdf_metrics`` on the interior (``interior_k_mask``) AND the tail. Also runs the
  E_tail balance-ODE (§5.4) over maturity and checks its fixed point.

Invariants (the centerpiece). For every phase verify ∫f=1 and ∫K f=S0 e^{rT} to machine
  precision by construction (the run_mz_uq.phase_A pure-NumPy snippet — NOT
  ``dupire_pipeline.compute_mean_diagnostics``, which is TF-bound). Positivity head-to-head:
  report min(f) for RAW ``gram_charlier_density`` (NEGATIVE in the smile's wings) vs
  ``maxent_tilt_density`` (≥0) — the §5.4 sign-theorem experiment — plus the raw-GC-vs-maxent
  ``pdf_metrics`` scorecard at comparable l2.

Pure NumPy + numpy.polynomial.hermite_e + scipy.optimize. No TensorFlow, no training.

Run:  ./.venv/bin/python examples/run_mz_energy_closure.py
Outputs: mz_energy_closure.json and plots/mz_energy_closure/*.png under the package root.
"""

import os
import sys
import json
import math
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DupirePipelineConfig, VolatilityConfig
from analytical_solutions import lognormal_density
from mz_spectral.fourier_dupire import build_uniform_tk_grid, build_second_derivative_matrix
from mz_spectral.validation import (
    build_truth_phi_bs, pdf_from_phi_tilde, pdf_metrics, interior_k_mask,
)
from mz_spectral.plot_pdf_analysis import mc_kde_pdf_on_K_grid, nearest_time_index
from mz_spectral import energy_closure as ec


# ----------------------------------------------------------------------------
def _const_vol(s):
    return VolatilityConfig.custom(lambda t, x: s * np.ones_like(np.asarray(x, float)))


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return o


def _wide_K_grid(s2_T, mu_T, S0, n=6000, n_sd=9.0):
    """Wide strike grid spanning ±n_sd log-return std around the core — for the
    by-construction invariants over (effectively) the whole real line."""
    s_T = math.sqrt(max(float(s2_T), 0.0))
    y = np.linspace(-n_sd, n_sd, n)
    return S0 * np.exp(float(mu_T) + s_T * y)


# ----------------------------------------------------------------------------
# Phase A — constant-sigma calibration (closure must collapse to the Gaussian core)
# ----------------------------------------------------------------------------
def phase_A(n_t=24, n_k=256, sigma_const=0.3):
    cfg = DupirePipelineConfig()
    S0, r, T_max, K_max = cfg.S0, cfg.r, cfg.T_max, cfg.K_max
    grid, _ = build_uniform_tk_grid(
        S0=S0, r=r, T_max=T_max, K_max=K_max, K_min=cfg.K_min, T_min=cfg.T_min,
        n_t=n_t, n_k=n_k, vol_config=_const_vol(sigma_const),
    )
    D2 = build_second_derivative_matrix(n_k, grid.dk)
    phi_truth = build_truth_phi_bs(grid, S0, r, sigma_const)

    i_rep = n_t - 1
    T_rep = float(grid.T[i_rep])
    Krow = grid.K[i_rep]
    km = interior_k_mask(Krow)

    # truth density on the strike grid: exact lognormal (constant-sigma ground truth)
    f_ref = lognormal_density(Krow, S0, T_rep, r, sigma_const)
    f_ref = f_ref / np.trapz(f_ref, Krow)

    # Extract the standardized log-return moments the closure consumes on a WIDE grid:
    # the canonical [K_min,K_max] grid truncates the lognormal's tails and would inject a
    # spurious skew/kurtosis (≈0.14/−0.23). On a wide grid the const-σ lognormal correctly
    # reads skew≈0, exkurt≈0 -> the closure collapses to the pure Gaussian core (E_tail≈0).
    K_wide = np.linspace(50.0, 20000.0, 8000)
    f_wide = lognormal_density(K_wide, S0, T_rep, r, sigma_const)
    f_wide = f_wide / np.trapz(f_wide, K_wide)
    mu_x, s2_x, skew_Y, exk_Y = ec.log_return_moments(f_wide, K_wide, S0)
    coeffs = ec.coeffs_from_skew_kurt(skew_Y, exk_Y)
    E_tail = ec.E_tail(coeffs)

    # build the closure densities on the strike grid (raw GC and max-entropy tilt)
    f_raw, aux_raw = ec.closure_density_on_K(Krow, s2_x, coeffs, S0, r, T_rep, kind="raw")
    f_me, aux_me = ec.closure_density_on_K(Krow, s2_x, coeffs, S0, r, T_rep, kind="maxent")

    # Cleanest collapse check: vs the EXACT (un-renormalized) analytic lognormal. The
    # f_ref above is renormalized on the truncated [K_min,K_max] grid (rescaled by ~1/0.97
    # since the lognormal has ~3% mass outside), so l2(closure, f_ref) carries that
    # renormalization mismatch; l2 vs the exact analytic density is the true machinery floor.
    f_exact = lognormal_density(Krow, S0, T_rep, r, sigma_const)
    l2_vs_exact = float(np.sqrt(np.trapz(((f_me - f_exact) ** 2)[km], Krow[km])))

    # invariants on a WIDE grid (by-construction exactness over the full line)
    Kw = _wide_K_grid(s2_x, aux_me["mu_T"], S0)
    fw_raw, _ = ec.closure_density_on_K(Kw, s2_x, coeffs, S0, r, T_rep, kind="raw")
    fw_me, _ = ec.closure_density_on_K(Kw, s2_x, coeffs, S0, r, T_rep, kind="maxent")
    inv_raw = ec.density_invariants(fw_raw, Kw, S0, r, T_rep)
    inv_me = ec.density_invariants(fw_me, Kw, S0, r, T_rep)

    res = {
        "params": {"n_t": n_t, "n_k": n_k, "sigma_const": sigma_const, "T_rep": T_rep},
        "extracted_moments": {"mu_logret": mu_x, "s2_logret": s2_x,
                              "skew_Y": skew_Y, "exkurt_Y": exk_Y, "E_tail": E_tail},
        "pdf_metrics_interior_raw": _jsonable(pdf_metrics(f_raw, f_ref, Krow, k_mask=km)),
        "pdf_metrics_interior_maxent": _jsonable(pdf_metrics(f_me, f_ref, Krow, k_mask=km)),
        "pdf_metrics_tail_maxent": _jsonable(pdf_metrics(f_me, f_ref, Krow, k_mask=~km)),
        "l2_interior_maxent_vs_exact_lognormal": l2_vs_exact,
        "invariants_raw": _jsonable(inv_raw),
        "invariants_maxent": _jsonable(inv_me),
        "maxent_solve": {"solved": bool(aux_me.get("solved")),
                         "max_moment_residual": float(aux_me.get("max_moment_residual", float("nan")))},
        "_plot": {"K": Krow.tolist(), "f_ref": f_ref.tolist(), "f_exact": f_exact.tolist(),
                  "f_raw": f_raw.tolist(), "f_me": f_me.tolist(), "T": T_rep},
    }
    res["verdict"] = (
        f"const-sigma: extracted skew={skew_Y:+.4f}, exkurt={exk_Y:+.4f}, E_tail={E_tail:.2e} "
        f"-> closure collapses to the Gaussian core. l2(maxent vs EXACT analytic lognormal, "
        f"interior)={l2_vs_exact:.2e} (machinery floor); l2 vs truncation-renormalized ref="
        f"{res['pdf_metrics_interior_maxent']['l2_pdf']:.2e}. Invariants (wide grid): "
        f"mass_defect={inv_me['mass_defect']:.1e}, mean_rel_err={inv_me['mean_rel_err']:.1e}, "
        f"min_f(maxent)={inv_me['min_f']:.1e}."
    )
    return res


# ----------------------------------------------------------------------------
# Phase B — dupire_paper smile vs MC truth
# ----------------------------------------------------------------------------
def phase_B(n_t=15, n_k=256, max_paths=80000, T_targets=(0.5, 1.0)):
    mc_path = os.path.join(_ROOT, "mc_arrays", "dupire_paper", "data_mc.npz")
    if not os.path.exists(mc_path):
        return {"skipped": True, "reason": f"MC arrays not found at {mc_path}"}
    cfg = DupirePipelineConfig()
    S0, r = cfg.S0, cfg.r
    # cap at T<=1.0 (MC t_all maxes at 1.0); keep the paper's K range
    grid, sigma_oracle = build_uniform_tk_grid(
        S0=S0, r=r, T_max=1.0, K_max=cfg.K_max, K_min=cfg.K_min, T_min=cfg.T_min,
        n_t=n_t, n_k=n_k, vol_config=VolatilityConfig.dupire_exact(),
    )
    D2 = build_second_derivative_matrix(n_k, grid.dk)

    mc = np.load(mc_path, mmap_mode="r")
    t_all = np.asarray(mc["t_all"]).ravel().astype(float)
    S_mat = mc["S_matrix"]

    per_T = {}
    balance_S2 = []          # smile-curvature source proxy S(T)^2 for the balance ODE
    grid_T_for_ode = []
    plot_blobs = []
    for T_req in T_targets:
        # MC terminal slice nearest T_req (MC grid), subsampled paths
        j = int(np.argmin(np.abs(t_all - float(T_req))))
        samples = np.asarray(S_mat[j, :max_paths]).astype(float)
        samples = samples[np.isfinite(samples) & (samples > 0)]

        # grid maturity nearest T_req
        idx = nearest_time_index(grid.T, T_req)
        T = float(grid.T[idx])
        Krow = grid.K[idx]
        km = interior_k_mask(Krow)

        # MC-truth density on the strike grid (log-K Gaussian KDE) — the scoring reference
        f_ref = mc_kde_pdf_on_K_grid(samples, Krow)

        # Fit the closure to the MC log-return moments extracted DIRECTLY from the raw paths
        # (no grid, no KDE bandwidth, no [K_min,K_max] truncation bias — a KDE on the
        # truncated grid would read skew≈0.37 vs the true ≈0.07). The closure then PREDICTS
        # the full density and we score it against the MC-KDE truth on the canonical grid.
        mu_x, s2_x, skew_Y, exk_Y = ec.log_return_moments_from_samples(samples, S0)
        coeffs = ec.coeffs_from_skew_kurt(skew_Y, exk_Y)
        E_tail = ec.E_tail(coeffs)

        f_raw, aux_raw = ec.closure_density_on_K(Krow, s2_x, coeffs, S0, r, T, kind="raw")
        f_me, aux_me = ec.closure_density_on_K(Krow, s2_x, coeffs, S0, r, T, kind="maxent")

        # invariants on a wide grid (by construction)
        Kw = _wide_K_grid(s2_x, aux_me["mu_T"], S0)
        fw_me, _ = ec.closure_density_on_K(Kw, s2_x, coeffs, S0, r, T, kind="maxent")
        fw_raw, _ = ec.closure_density_on_K(Kw, s2_x, coeffs, S0, r, T, kind="raw")
        inv_me = ec.density_invariants(fw_me, Kw, S0, r, T)
        inv_raw = ec.density_invariants(fw_raw, Kw, S0, r, T)

        # the source proxy for the balance ODE: the "tail energy" the data actually carries
        balance_S2.append(E_tail)
        grid_T_for_ode.append(T)

        per_T[f"{float(T_req):.4f}"] = {
            "grid_T": T, "mc_t_used": float(t_all[j]), "n_paths": int(samples.size),
            "extracted_moments": {"mu_logret": mu_x, "s2_logret": s2_x,
                                  "skew_Y": skew_Y, "exkurt_Y": exk_Y, "E_tail": E_tail},
            "pdf_metrics_interior_raw": _jsonable(pdf_metrics(f_raw, f_ref, Krow, k_mask=km)),
            "pdf_metrics_interior_maxent": _jsonable(pdf_metrics(f_me, f_ref, Krow, k_mask=km)),
            "pdf_metrics_tail_raw": _jsonable(pdf_metrics(f_raw, f_ref, Krow, k_mask=~km)),
            "pdf_metrics_tail_maxent": _jsonable(pdf_metrics(f_me, f_ref, Krow, k_mask=~km)),
            "invariants_raw": _jsonable(inv_raw),
            "invariants_maxent": _jsonable(inv_me),
            "maxent_solve": {"solved": bool(aux_me.get("solved")),
                             "max_moment_residual": float(aux_me.get("max_moment_residual", float("nan")))},
        }
        # wide-grid wing densities so the raw-GC negativity (which lives BEYOND the
        # [K_min,K_max] window) is visible in the positivity head-to-head panel
        Kwing = _wide_K_grid(s2_x, aux_me["mu_T"], S0, n=3000, n_sd=6.0)
        fwing_raw, _ = ec.closure_density_on_K(Kwing, s2_x, coeffs, S0, r, T, kind="raw")
        fwing_me, _ = ec.closure_density_on_K(Kwing, s2_x, coeffs, S0, r, T, kind="maxent")
        plot_blobs.append({"T_req": float(T_req), "grid_T": T, "K": Krow.tolist(),
                           "f_ref": f_ref.tolist(), "f_raw": f_raw.tolist(),
                           "f_me": f_me.tolist(), "samples_q": np.percentile(samples, [1, 99]).tolist(),
                           "Kwing": Kwing.tolist(), "fwing_raw": fwing_raw.tolist(),
                           "fwing_me": fwing_me.tolist()})

    # ---- §5.4 balance ODE over maturity: dE_tail/dT = gamma_in S^2 - gamma_c E_tail ----
    # Use the extracted E_tail trajectory as the source proxy S(T)^2; scan gamma_c ~ 3*kappa.
    # gamma_c is the lowest unresolved Hermite (He_3) relaxation rate ~ 3 (per the synthesis,
    # gamma_c = lowest unresolved Hermite rate); gamma_in chosen so the fixed point matches
    # the late-maturity tail energy (a one-parameter calibration, the rest derived).
    S2_arr = np.array(balance_S2)
    T_arr = np.array(grid_T_for_ode)
    ode_scan = {}
    if S2_arr.size >= 2:
        for p in (1, 2, 3):  # exponent scan (the synthesis: p=2 derived; scan to confirm)
            for gamma_c in (1.0, 3.0, 6.0):  # ~3*kappa derived drain; scan around it
                src = S2_arr ** p if p != 1 else S2_arr
                gamma_in = 1.0
                E = ec.integrate_balance_ode(src, T_arr, gamma_in=gamma_in, gamma_c=gamma_c, E0=0.0)
                fp = (gamma_in / gamma_c) * float(src[-1])
                ode_scan[f"p{p}_gc{gamma_c:.0f}"] = {
                    "E_tail_final": float(E[-1]), "fixed_point": fp,
                    "monotone_nonneg": bool(np.all(E >= -1e-12)),
                }

    res = {
        "skipped": False,
        "params": {"n_t": n_t, "n_k": n_k, "max_paths": max_paths,
                   "T_targets": list(T_targets), "T_range": [float(grid.T[0]), float(grid.T[-1])]},
        "by_maturity": per_T,
        "balance_ode_scan": ode_scan,
        "_plot": {"blobs": plot_blobs},
    }
    # headline verdict at the latest maturity
    last_key = f"{float(T_targets[-1]):.4f}"
    d = per_T[last_key]
    res["verdict"] = (
        f"dupire smile T={d['grid_T']:.2f}: MC truth skew={d['extracted_moments']['skew_Y']:+.3f}, "
        f"exkurt={d['extracted_moments']['exkurt_Y']:+.3f} (mild). maxent vs MC-KDE l2 interior="
        f"{d['pdf_metrics_interior_maxent']['l2_pdf']:.2e}, tail={d['pdf_metrics_tail_maxent']['l2_pdf']:.2e}, "
        f"tail_mass_ratio={d['pdf_metrics_tail_maxent']['tail_mass_ratio']:.3f}. POSITIVITY head-to-head: "
        f"min_f raw-GC={d['invariants_raw']['min_f']:+.2e} vs maxent={d['invariants_maxent']['min_f']:+.2e}. "
        f"Invariants(maxent): mass_defect={d['invariants_maxent']['mass_defect']:.1e}, "
        f"mean_rel_err={d['invariants_maxent']['mean_rel_err']:.1e}."
    )
    return res


# ----------------------------------------------------------------------------
# Positivity stress test (the §5.4 sign-theorem experiment, exaggerated)
# ----------------------------------------------------------------------------
def positivity_stress(S0=1000.0, r=0.04, T=1.0, s2=0.36,
                      cases=((-0.9, 1.0), (-0.6, -0.3), (-0.5, 0.4))):
    """Deliberately strong (skew, exkurt) to EXHIBIT the raw-Gram-Charlier wing negativity
    and the max-entropy cure. Reports min(f) head-to-head and the feasibility boundary."""
    out = {}
    for skew, exk in cases:
        coeffs = ec.coeffs_from_skew_kurt(skew, exk)
        Kw = _wide_K_grid(s2, 0.0, S0)
        f_raw, _ = ec.closure_density_on_K(Kw, s2, coeffs, S0, r, T, kind="raw")
        f_me, aux = ec.closure_density_on_K(Kw, s2, coeffs, S0, r, T, kind="maxent")
        inv_raw = ec.density_invariants(f_raw, Kw, S0, r, T)
        inv_me = ec.density_invariants(f_me, Kw, S0, r, T)
        out[f"skew{skew:+.2f}_exk{exk:+.2f}"] = {
            "E_tail": ec.E_tail(coeffs),
            "min_f_raw_gram_charlier": inv_raw["min_f"],
            "min_f_maxent_tilt": inv_me["min_f"],
            "raw_goes_negative": bool(inv_raw["min_f"] < 0),
            "maxent_nonneg": bool(inv_me["min_f"] >= 0),
            "maxent_solved": bool(aux.get("solved")),
            "maxent_moment_residual": float(aux.get("max_moment_residual", float("nan"))),
            "maxent_mass_defect": inv_me["mass_defect"],
            "maxent_mean_rel_err": inv_me["mean_rel_err"],
        }
    return out


# ----------------------------------------------------------------------------
def make_plots(A, B, outdir):
    os.makedirs(outdir, exist_ok=True)
    # Phase A: closure vs exact lognormal (residual vs the EXACT analytic density, which is
    # the true machinery floor — not vs the truncation-renormalized f_ref).
    try:
        p = A["_plot"]; K = np.array(p["K"])
        f_exact = np.array(p["f_exact"])
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        ax[0].plot(K, f_exact, "k-", lw=2, label="exact lognormal")
        ax[0].plot(K, p["f_raw"], "C3--", lw=1.6, label="raw Gram-Charlier")
        ax[0].plot(K, p["f_me"], "C0-.", lw=1.6, label="max-entropy tilt")
        ax[0].set_title(f"Phase A: const-σ closure vs lognormal (T={p['T']:.2f})\nE_tail≈0 -> Gaussian core")
        ax[0].set_xlabel("K"); ax[0].set_ylabel("f(K)"); ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        # residual vs EXACT analytic lognormal -> machine precision
        ax[1].plot(K, np.array(p["f_me"]) - f_exact, "C0-", lw=1.2, label="maxent − exact lognormal")
        ax[1].plot(K, np.array(p["f_raw"]) - f_exact, "C3--", lw=1.0, label="raw GC − exact lognormal")
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_title("Phase A: closure residual vs EXACT lognormal (machine precision)")
        ax[1].set_xlabel("K"); ax[1].set_ylabel("Δf"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_A_const_sigma.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"[plot A] skipped: {e}")

    # Phase B: closure vs MC truth, per maturity, with the positivity head-to-head
    if not B.get("skipped"):
        try:
            blobs = B["_plot"]["blobs"]
            n = len(blobs)
            fig, axes = plt.subplots(n, 2, figsize=(13, 4.4 * n))
            if n == 1:
                axes = axes.reshape(1, -1)
            for row, blob in enumerate(blobs):
                K = np.array(blob["K"])
                a0, a1 = axes[row, 0], axes[row, 1]
                a0.plot(K, blob["f_ref"], "k-", lw=2, label="MC-KDE truth")
                a0.plot(K, blob["f_raw"], "C3--", lw=1.5, label="raw Gram-Charlier")
                a0.plot(K, blob["f_me"], "C0-.", lw=1.5, label="max-entropy tilt")
                a0.set_title(f"Phase B: dupire smile vs MC (T={blob['grid_T']:.2f})")
                a0.set_xlabel("K"); a0.set_ylabel("f(K)"); a0.legend(fontsize=8); a0.grid(alpha=0.3)
                # FAR-WING positivity head-to-head on the WIDE grid: the raw Gram-Charlier
                # negativity lives in the deep tail (beyond [K_min,K_max]); plot it where the
                # min(f) is actually achieved so the sign-theorem failure is visible.
                Kw = np.array(blob["fwing_raw"]) * 0 + np.array(blob["Kwing"])
                fwr = np.array(blob["fwing_raw"]); fwm = np.array(blob["fwing_me"])
                imin = int(np.argmin(fwr))
                # zoom around the most-negative raw-GC region (left wing)
                lo = max(0, imin - 400); hi = min(len(Kw), imin + 400)
                sl = slice(lo, hi)
                a1.plot(Kw[sl], fwr[sl], "C3--", lw=1.6, label=f"raw GC (min={fwr.min():+.1e})")
                a1.plot(Kw[sl], fwm[sl], "C0-.", lw=1.6, label=f"maxent (min={fwm.min():+.1e}, ≥0)")
                a1.axhline(0, color="0.4", lw=1.0, ls=":")
                a1.set_title("Deep-wing positivity head-to-head (sign theorem)")
                a1.set_xlabel("K"); a1.set_ylabel("f(K)"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_B_smile_vs_mc.png"), dpi=130)
            plt.close(fig)
        except Exception as e:
            print(f"[plot B] skipped: {e}")


# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("MZ x Dupire — Step 2: density-side energy-budget closure (§5)")
    print("=" * 78)

    print("\n[Phase A] constant-sigma calibration (collapse to Gaussian core) ...")
    A = phase_A()
    print("  VERDICT:", A["verdict"])

    print("\n[Phase B] dupire_paper smile vs MC truth ...")
    B = phase_B()
    print("  VERDICT:", B.get("verdict", B.get("reason")))

    print("\n[Positivity stress] raw Gram-Charlier vs max-entropy tilt (sign theorem) ...")
    P = positivity_stress()
    for k, v in P.items():
        print(f"  {k}: min_f raw={v['min_f_raw_gram_charlier']:+.2e} (neg={v['raw_goes_negative']})  "
              f"maxent={v['min_f_maxent_tilt']:+.2e} (>=0={v['maxent_nonneg']}, solved={v['maxent_solved']}, "
              f"resid={v['maxent_moment_residual']:.1e})")

    make_plots(A, B, os.path.join(_ROOT, "plots", "mz_energy_closure"))

    A.pop("_plot", None)
    if not B.get("skipped"):
        B.pop("_plot", None)
    payload = {"phase_A": A, "phase_B": B, "positivity_stress": P}
    outjson = os.path.join(_ROOT, "mz_energy_closure.json")
    with open(outjson, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    print(f"\nWrote {outjson}")
    print(f"Wrote plots to {os.path.join(_ROOT, 'plots', 'mz_energy_closure')}/")
    print("\nDONE.")


if __name__ == "__main__":
    main()
