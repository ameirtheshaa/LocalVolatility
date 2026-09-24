#!/usr/bin/env python3
"""
MZ x Dupire — epistemic-UQ extension: calibrated confidence bands on sigma(T,K)
and the risk-neutral density f(K,T), validating the machinery in
mz_spectral/uncertainty.py.

Phase A (decisive, no MC): constant-sigma Black-Scholes exact case.
  Known truth sigma_oracle=const and exact lognormal density => clean CALIBRATION
  test. Inject a data-estimable input price noise Sigma_phi, propagate it, and check:
    * 95% band coverage of the truth (interior vs tail);
    * tail-widening: sigma-band width grows as the convexity phi~_kk -> 0;
    * weak-form (IBP) density band is well-conditioned in the tails where the
      pointwise delta-method band blows up;
    * delta-method vs MC-over-input sigma bands agree in the interior;
    * the point density preserves mass (int f = 1) and the martingale (int K f =
      S0 e^{rT}).

Phase B (realistic, MC-grounded): dupire_paper smile, Sigma_phi from the MC price
  standard error (mc_arrays/dupire_paper/data_mc.npz). Bands around the noisy
  MC-implied surface; coverage of the TRUE dupire-exact sigma where feasible; the
  pole-benignity feasibility map. Restricted to T<=1.0 (MC time-grid coverage).

Run:  ./.venv/bin/python examples/run_mz_uq.py
Outputs: mz_uq.json and plots/mz_uq/*.png under the package root.
"""

import os
import sys
import json
import math
import warnings

import numpy as np

# numpy.trapz was renamed numpy.trapezoid in numpy 2.0 and removed in 2.4; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

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
from mz_spectral.validation import build_truth_phi_bs, pdf_from_phi_tilde, interior_k_mask
from mz_spectral import uncertainty as uq


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


def _row_masks(grid):
    n_t = grid.K.shape[0]
    interior = np.array([interior_k_mask(grid.K[i]) for i in range(n_t)])
    return interior, ~interior


# ----------------------------------------------------------------------------
# Phase A — constant-sigma exact calibration test
# ----------------------------------------------------------------------------
def phase_A(n_t=24, n_k=256, sigma_const=0.3, rel_noise=0.01, R=300, z=1.96, seed=0):
    cfg = DupirePipelineConfig()
    S0, r, T_max, K_max = cfg.S0, cfg.r, cfg.T_max, cfg.K_max
    grid, _ = build_uniform_tk_grid(
        S0=S0, r=r, T_max=T_max, K_max=K_max, K_min=cfg.K_min, T_min=cfg.T_min,
        n_t=n_t, n_k=n_k, vol_config=_const_vol(sigma_const),
    )
    D2 = build_second_derivative_matrix(n_k, grid.dk)
    phi_truth = build_truth_phi_bs(grid, S0, r, sigma_const)
    se = uq.input_covariance_const(phi_truth, rel_noise=rel_noise)  # (n_t,n_k) std
    interior, tail = _row_masks(grid)
    sig_truth = np.full((n_t, n_k), sigma_const)
    nominal = 2.0 * (0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))) - 1.0  # e.g. 0.95 for z=1.96

    # --- sigma-band coverage: cheap delta band (per realization) vs the calibrated
    #     MC-over-input band. The NONLINEAR ill-posed inverse breaks the first-order
    #     delta linearization (under-coverage) -> MC-over-input is the band to use.
    rng = np.random.default_rng(seed)
    d_int, d_tail = [], []
    for _ in range(R):
        phi_obs = phi_truth + rng.standard_normal(phi_truth.shape) * se
        _, lo, hi = uq.sigma_band_delta(phi_obs, se, grid, D2, T_max, z=z)
        d_int.append(uq.band_coverage(sig_truth, lo, hi, mask=interior).get("masked", np.nan))
        d_tail.append(uq.band_coverage(sig_truth, lo, hi, mask=tail).get("masked", np.nan))
    lo_q, hi_q = 100.0 * (1 - nominal) / 2.0, 100.0 * (1 + nominal) / 2.0
    R_mc, n_samp, smooth_k = 40, 100, 5.0
    rng2 = np.random.default_rng(seed + 10)
    m_int, m_tail = [], []        # raw pointwise MC band (ill-posed)
    g_int, g_tail = [], []        # regularized (smoothed) MC band (calibrated)
    for rr in range(R_mc):
        phi_obs = phi_truth + rng2.standard_normal(phi_truth.shape) * se
        _, lo, hi = uq.sigma_band_mc(phi_obs, se, grid, D2, T_max, n_samples=n_samp,
                                     seed=seed + 100 + rr, lo_q=lo_q, hi_q=hi_q)
        m_int.append(uq.band_coverage(sig_truth, lo, hi, mask=interior).get("masked", np.nan))
        m_tail.append(uq.band_coverage(sig_truth, lo, hi, mask=tail).get("masked", np.nan))
        _, glo, ghi = uq.sigma_band_mc(phi_obs, se, grid, D2, T_max, n_samples=n_samp,
                                       seed=seed + 100 + rr, lo_q=lo_q, hi_q=hi_q, smooth_k=smooth_k)
        g_int.append(uq.band_coverage(sig_truth, glo, ghi, mask=interior).get("masked", np.nan))
        g_tail.append(uq.band_coverage(sig_truth, glo, ghi, mask=tail).get("masked", np.nan))
    sigma_cov = {"nominal": float(nominal),
                 "delta_interior": float(np.nanmean(d_int)), "delta_tail": float(np.nanmean(d_tail)),
                 "mc_raw_interior": float(np.nanmean(m_int)), "mc_raw_tail": float(np.nanmean(m_tail)),
                 "mc_reg_interior": float(np.nanmean(g_int)), "mc_reg_tail": float(np.nanmean(g_tail)),
                 "smooth_k": smooth_k, "R_delta": R, "R_mc": R_mc, "n_samples_mc": n_samp}

    # --- density-band coverage sweep at a representative maturity ---
    i_rep = n_t - 1
    T_rep, Krow = float(grid.T[i_rep]), grid.K[i_rep]
    f_oracle = lognormal_density(Krow, S0, T_rep, r, sigma_const)
    km = interior_k_mask(Krow)
    dcov_pt_i, dcov_pt_t, dcov_wf_i, dcov_wf_t = [], [], [], []
    for _ in range(R):
        phi_obs = phi_truth + rng.standard_normal(phi_truth.shape) * se
        fm, flo, fhi = uq.density_band(phi_obs[i_rep], se[i_rep], T_rep, r, S0, K_max, D2, z=z)
        wfm, wlo, whi, h = uq.weak_form_density_band(
            phi_obs[i_rep], se[i_rep], Krow, T_rep, r, S0, K_max, D2, z=z)
        cpt = uq.band_coverage(f_oracle, flo, fhi, mask=km)
        cwf = uq.band_coverage(f_oracle, wlo, whi, mask=km)
        dcov_pt_i.append(cpt["masked"]); dcov_pt_t.append(uq.band_coverage(f_oracle, flo, fhi, mask=~km)["masked"])
        dcov_wf_i.append(cwf["masked"]); dcov_wf_t.append(uq.band_coverage(f_oracle, wlo, whi, mask=~km)["masked"])
    density_cov = {
        "pointwise_interior": float(np.nanmean(dcov_pt_i)), "pointwise_tail": float(np.nanmean(dcov_pt_t)),
        "weakform_interior": float(np.nanmean(dcov_wf_i)), "weakform_tail": float(np.nanmean(dcov_wf_t)),
        "mollifier_h": float(h),
    }

    # --- single-realization bands (for plotting): delta, raw MC, regularized MC ---
    phi_obs = phi_truth + np.random.default_rng(seed + 1).standard_normal(phi_truth.shape) * se
    s_med_d, s_lo_d, s_hi_d = uq.sigma_band_delta(phi_obs, se, grid, D2, T_max, z=z)
    _, s_lo_m, s_hi_m = uq.sigma_band_mc(phi_obs, se, grid, D2, T_max, n_samples=200,
                                         seed=seed + 2, lo_q=lo_q, hi_q=hi_q)
    s_med_g, s_lo_g, s_hi_g = uq.sigma_band_mc(phi_obs, se, grid, D2, T_max, n_samples=200,
                                               seed=seed + 2, lo_q=lo_q, hi_q=hi_q, smooth_k=smooth_k)

    # tail widening: sigma band width vs |phi~_kk| at the representative maturity
    phi_kk_rep = (phi_truth @ D2.T)[i_rep]
    width_rep = (s_hi_d - s_lo_d)[i_rep]
    finite = np.isfinite(width_rep) & (np.abs(phi_kk_rep) > 0)
    # Spearman-ish: width should increase as |phi_kk| decreases -> negative corr(width, |phi_kk|)
    tail_widening_corr = float(np.corrcoef(width_rep[finite], np.abs(phi_kk_rep)[finite])[0, 1]) \
        if finite.sum() > 2 else float("nan")

    # weak-form vs pointwise conditioning in the tails (band sd ratio)
    fm, flo, fhi = uq.density_band(phi_obs[i_rep], se[i_rep], T_rep, r, S0, K_max, D2, z=z)
    wfm, wlo, whi, _ = uq.weak_form_density_band(phi_obs[i_rep], se[i_rep], Krow, T_rep, r, S0, K_max, D2, z=z)
    sd_pt = 0.5 * (fhi - flo); sd_wf = 0.5 * (whi - wlo)
    tailm = ~km
    conditioning = {
        "pointwise_tail_sd_median": float(np.median(sd_pt[tailm])),
        "weakform_tail_sd_median": float(np.median(sd_wf[tailm])),
        "tail_sd_ratio_pt_over_wf": float(np.median(sd_pt[tailm]) / max(np.median(sd_wf[tailm]), 1e-300)),
    }

    # martingale / mass of the point density (clipped+normalized estimate)
    f_pt = pdf_from_phi_tilde(phi_truth[i_rep], T_rep, r, S0, K_max, Krow, D2, normalize=False)
    mass = float(_trapz(f_pt, Krow))
    mean = float(_trapz(Krow * f_pt, Krow))
    martingale = {"int_f": mass, "int_Kf": mean, "S0_erT": float(S0 * np.exp(r * T_rep)),
                  "mean_rel_err": float(abs(mean - S0 * np.exp(r * T_rep)) / (S0 * np.exp(r * T_rep)))}

    res = {
        "params": {"n_t": n_t, "n_k": n_k, "sigma_const": sigma_const,
                   "rel_noise": rel_noise, "R": R, "z": z, "T_rep": T_rep},
        "sigma_coverage": sigma_cov,
        "density_coverage": density_cov,
        "tail_widening_corr_width_vs_absphikk": tail_widening_corr,
        "conditioning": conditioning,
        "martingale": martingale,
        "_plot": {
            "k_tilde": grid.k_tilde.tolist(), "K_rep": Krow.tolist(),
            "sig_lo_m": s_lo_m[i_rep].tolist(), "sig_hi_m": s_hi_m[i_rep].tolist(),
            "sig_med_g": s_med_g[i_rep].tolist(), "sig_lo_g": s_lo_g[i_rep].tolist(), "sig_hi_g": s_hi_g[i_rep].tolist(),
            "f_oracle": f_oracle.tolist(),
            "f_pt": fm.tolist(), "f_pt_lo": flo.tolist(), "f_pt_hi": fhi.tolist(),
            "f_wf": wfm.tolist(), "f_wf_lo": wlo.tolist(), "f_wf_hi": whi.tolist(),
            "phi_kk_rep": phi_kk_rep.tolist(), "width_rep": width_rep.tolist(),
            "sigma_const": sigma_const,
        },
    }
    # verdict
    res["verdict"] = (
        f"sigma coverage (nominal {nominal:.2f}): RAW pointwise inverse is ILL-POSED at "
        f"{rel_noise:.0%} price noise -> neither delta {sigma_cov['delta_interior']:.2f} nor raw MC "
        f"{sigma_cov['mc_raw_interior']:.2f} covers; REGULARIZED (smooth_k={sigma_cov['smooth_k']:.0f}) MC band "
        f"interior {sigma_cov['mc_reg_interior']:.2f} is calibrated. Density LINEAR -> exact bands: "
        f"weak-form tail coverage {density_cov['weakform_tail']:.2f} (= pointwise "
        f"{density_cov['pointwise_tail']:.2f}) but {conditioning['tail_sd_ratio_pt_over_wf']:.0f}x narrower "
        f"in the tails. Martingale rel-err {martingale['mean_rel_err']:.1e}."
    )
    return res


# ----------------------------------------------------------------------------
# Phase B — MC-grounded Sigma_phi on the dupire_paper smile
# ----------------------------------------------------------------------------
def phase_B(n_t=15, n_k=128, max_paths=200000, z=1.96):
    mc_path = os.path.join(_ROOT, "mc_arrays", "dupire_paper", "data_mc.npz")
    if not os.path.exists(mc_path):
        return {"skipped": True, "reason": f"MC arrays not found at {mc_path}"}
    cfg = DupirePipelineConfig()
    S0, r = cfg.S0, cfg.r
    # restrict to T<=1.0 (MC time-grid coverage); keep K range from the paper case
    grid, sigma_oracle = build_uniform_tk_grid(
        S0=S0, r=r, T_max=1.0, K_max=cfg.K_max, K_min=cfg.K_min, T_min=cfg.T_min,
        n_t=n_t, n_k=n_k, vol_config=VolatilityConfig.dupire_exact(),
    )
    D2 = build_second_derivative_matrix(n_k, grid.dk)
    se, Ct, t_used = uq.input_covariance_from_mc(mc_path, grid, r, S0, max_paths=max_paths)

    feas = uq.pole_feasibility(Ct, grid, D2)
    s_med, s_lo, s_hi = uq.sigma_band_delta(Ct, se, grid, D2, 1.0, z=z)
    interior, tail = _row_masks(grid)
    cov_feas = uq.band_coverage(sigma_oracle, s_lo, s_hi, mask=feas["feasible"])
    cov_int = uq.band_coverage(sigma_oracle, s_lo, s_hi, mask=interior)

    i_rep = n_t - 1
    Krow = grid.K[i_rep]
    fm, flo, fhi = uq.density_band(Ct[i_rep], se[i_rep], float(grid.T[i_rep]), r, S0, cfg.K_max, D2, z=z)
    return {
        "skipped": False,
        "params": {"n_t": n_t, "n_k": n_k, "max_paths": max_paths, "z": z,
                   "T_range": [float(grid.T[0]), float(grid.T[-1])],
                   "mc_t_used_range": [float(t_used.min()), float(t_used.max())]},
        "se_phi_median": float(np.median(se)),
        "frac_feasible": float(feas["feasible"].mean()),
        "frac_reliable": float(feas["reliable"].mean()),
        "frac_convex": float(feas["convex"].mean()),
        "sigma_coverage_feasible": _jsonable(cov_feas),
        "sigma_coverage_interior": _jsonable(cov_int),
        "_plot": {
            "k_tilde": grid.k_tilde.tolist(), "K_rep": Krow.tolist(),
            "sig_oracle_rep": sigma_oracle[i_rep].tolist(),
            "sig_med_rep": s_med[i_rep].tolist(), "sig_lo_rep": s_lo[i_rep].tolist(),
            "sig_hi_rep": s_hi[i_rep].tolist(),
            "feasible_rep": feas["feasible"][i_rep].tolist(),
            "f_rep": fm.tolist(), "f_lo_rep": flo.tolist(), "f_hi_rep": fhi.tolist(),
            "T_rep": float(grid.T[i_rep]),
        },
        "verdict": (f"MC-grounded: {100*feas['feasible'].mean():.0f}% of cells feasible; "
                    f"sigma-band covers true dupire sigma on {cov_feas['masked']:.2f} of "
                    f"feasible cells (nominal {2*(0.5*(1+math.erf(z/np.sqrt(2))))-1:.2f})."),
    }


# ----------------------------------------------------------------------------
def make_plots(A, B, outdir):
    os.makedirs(outdir, exist_ok=True)
    try:
        p = A["_plot"]; k = np.array(p["k_tilde"])
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
        ax[0].axhline(p["sigma_const"], color="k", lw=1.5, ls="-", label="oracle sigma")
        ax[0].fill_between(k, p["sig_lo_m"], p["sig_hi_m"], color="0.6", alpha=0.35,
                           label="raw pointwise MC band (ill-posed)")
        ax[0].plot(k, p["sig_med_g"], "C0-", lw=1, label="regularized median")
        ax[0].fill_between(k, p["sig_lo_g"], p["sig_hi_g"], color="C0", alpha=0.30,
                           label="regularized 95% band (calibrated)")
        ax[0].set_title("Phase A: sigma band (const-sigma) — raw is ill-posed; regularize")
        ax[0].set_xlabel("k~"); ax[0].set_ylabel("sigma"); ax[0].set_ylim(0, 1.2); ax[0].legend(fontsize=7)
        K = np.array(p["K_rep"])
        ax[1].plot(K, p["f_oracle"], "k-", lw=1.5, label="exact lognormal")
        ax[1].fill_between(K, p["f_pt_lo"], p["f_pt_hi"], color="C0", alpha=0.25, label="pointwise band")
        ax[1].fill_between(K, p["f_wf_lo"], p["f_wf_hi"], color="C2", alpha=0.35, label="weak-form band")
        ax[1].set_title("Phase A: density bands — weak-form is well-conditioned")
        ax[1].set_xlabel("K"); ax[1].set_ylabel("f(K)"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_A_bands.png"), dpi=130); plt.close(fig)
    except Exception as e:
        print(f"[plot A] skipped: {e}")

    if not B.get("skipped"):
        try:
            p = B["_plot"]; K = np.array(p["K_rep"])
            fig, ax = plt.subplots(figsize=(7.5, 4.3))
            ax.plot(K, p["sig_oracle_rep"], "k-", lw=1.5, label="true dupire sigma")
            ax.plot(K, p["sig_med_rep"], "C0.", ms=3, label="MC-implied (delta) median")
            ax.fill_between(K, p["sig_lo_rep"], p["sig_hi_rep"], color="C0", alpha=0.25, label="95% band")
            feas = np.array(p["feasible_rep"], dtype=bool)
            ax.plot(K[~feas], p["sig_oracle_rep"] if False else np.array(p["sig_oracle_rep"])[~feas],
                    "rx", ms=4, label="infeasible (pole gate)")
            ax.set_title(f"Phase B: MC-grounded sigma band, dupire smile (T={p['T_rep']:.2f})")
            ax.set_xlabel("K"); ax.set_ylabel("sigma"); ax.set_ylim(0, 1.5); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_B_mc_sigma.png"), dpi=130); plt.close(fig)
        except Exception as e:
            print(f"[plot B] skipped: {e}")


def main():
    print("=" * 78)
    print("MZ x Dupire — epistemic-UQ bands")
    print("=" * 78)
    print("\n[Phase A] constant-sigma exact calibration ...")
    A = phase_A()
    print("  VERDICT:", A["verdict"])
    print("\n[Phase B] MC-grounded Sigma_phi (dupire_paper smile) ...")
    B = phase_B()
    print("  VERDICT:", B.get("verdict", B.get("reason")))

    make_plots(A, B, os.path.join(_ROOT, "plots", "mz_uq"))
    A.pop("_plot", None); B.pop("_plot", None)
    with open(os.path.join(_ROOT, "mz_uq.json"), "w") as fh:
        json.dump(_jsonable({"phase_A": A, "phase_B": B}), fh, indent=2)
    print(f"\nWrote {os.path.join(_ROOT, 'mz_uq.json')}")
    print(f"Wrote plots to {os.path.join(_ROOT, 'plots', 'mz_uq')}/")
    print("\nDONE.")


if __name__ == "__main__":
    main()
