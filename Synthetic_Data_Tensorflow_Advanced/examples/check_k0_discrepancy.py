#!/usr/bin/env python3
"""
Why does the K=0 call deficit differ between the constant-σ and Dupire models?

The trained price net gives C_NN(0,T) = φ̃(0,T)·S0 with φ̃ = 1 − exp(−N_c). By
no-arbitrage C(0,T) = S0 (a zero-strike call is the asset), so φ̃(0,T) should be 1.
It is not, and the shortfall is larger for the Dupire model. This script quantifies
the three ingredients of that behaviour, side by side for both models:

  (1) Extrapolation: both nets see only K in [K_min, K_max]; K=0 (k̃=0) is outside.
      We report the lowest trained k̃ and the in-sample fit residual there (K=K_min),
      to separate "bad extrapolation" from "bad fit".
  (2) Boundary slope / lost mass: the network-implied survival probability
      P_NN(S_T>K) = −(S0/K_max)·∂φ̃/∂k̃ must be 1 at K=0. We read off P_NN(S_T>0).
  (3) Curve steepness / local vol: σ_loc(T,K) = sqrt(2·NN_eta/t_max). A steeper
      low-strike curve (skew) gives a sharper k̃=0 corner that a smooth net undershoots.

Outputs a comparison table, a JSON, and a 3-panel overlay figure.

Usage:
    python examples/check_k0_discrepancy.py [--maturities 0.25 0.5 0.75 1.0]
"""

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig
from dupire_pipeline import PDFAnalyzer, load_trained_models

NTU_NAVY = "#00205B"
NTU_RED = "#C02026"

MODELS = [
    ("const-σ", "models/runs/synthetic_paper_large_dataset_constant_vol", NTU_NAVY),
    ("Dupire", "models/runs/synthetic_paper_large_dataset_dupire_exact", NTU_RED),
]


def build(model_dir, T_values):
    config = DupirePipelineConfig.analysis_only(model_dir)
    config.analysis_config.T_analysis = T_values
    nn_phi, nn_eta, metadata = load_trained_models(model_dir)
    return PDFAnalyzer(nn_phi, nn_eta, config, metadata, phi_mapping='transformed')


def phi_of_ktilde(an, T, k_tilde_grid):
    """φ̃ on a k̃ grid: invert k̃ -> K, push through the production transform."""
    r = float(an.config.r)
    K = k_tilde_grid * float(an.k_max) * np.exp(r * T)
    t_t, k_t = an.prepare_data_for_nn(T, K)
    phi = an.phi_tilde_from_nn(t_t, k_t).numpy().flatten()
    return phi


def sigma_loc(an, T, K):
    """Local vol σ(T,K) = sqrt(2·NN_eta([t̃,k̃])/t_max)."""
    t_t, k_t = an.prepare_data_for_nn(T, np.asarray(K, dtype=np.float64))
    eta = an.nn_eta(tf.concat([t_t, k_t], axis=1)).numpy().flatten()
    return np.sqrt(2.0 * eta / float(an.t_max))


def insample_resid_at_Kmin(an, model_dir):
    """NN call vs training-target call price at the lowest trained strike."""
    d = np.load(os.path.join(model_dir, "training_data.npz"), allow_pickle=True)
    T = d["T"].flatten(); K = d["K"].flatten(); C = d["phi"].flatten()
    S0 = float(an.config.S0)
    Kmin = float(np.min(K))
    sel = np.isclose(K, Kmin, rtol=0, atol=1e-6)
    rows = []
    for Ti, Ci in zip(T[sel], C[sel]):
        t_t, k_t = an.prepare_data_for_nn(float(Ti), np.array([Kmin]))
        C_nn = float(an.phi_tilde_from_nn(t_t, k_t).numpy().flatten()[0]) * S0
        rows.append((float(Ti), Kmin, float(Ci), C_nn, (C_nn - Ci) / Ci * 100.0))
    return Kmin, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maturities', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument('--output-dir', default='plots/nn_k0_check')
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)
    T_values = sorted(args.maturities)
    os.makedirs(args.output_dir, exist_ok=True)

    out = {}
    analyzers = {}
    for tag, mdir, _c in MODELS:
        an = build(mdir, T_values)
        analyzers[tag] = an
        S0, r, kmax = float(an.config.S0), float(an.config.r), float(an.k_max)
        k_min_tilde = lambda T: np.exp(-r * T) * float(an.k_min) / kmax
        print("=" * 100)
        print(f"MODEL: {tag}   ({mdir})")
        print(f"  S0={S0:g} r={r:g} K_min={an.k_min:g} K_max={kmax:g} t_max={an.t_max:g}")
        # in-sample fit at K_min
        Kmin, rows = insample_resid_at_Kmin(an, mdir)
        res = [abs(x[4]) for x in rows]
        print(f"  In-sample fit at lowest trained strike K={Kmin:g} "
              f"(k̃≈{k_min_tilde(0.5):.3f}): |C_NN-C_data|/C_data = "
              f"{np.mean(res):.3f}% mean, {np.max(res):.3f}% max over {len(rows)} maturities")
        # boundary table
        print(f"\n  {'T':>5} | {'k̃_min':>6} | {'φ̃(k̃_min)':>9} | {'φ̃(0)':>7} | "
              f"{'N_c(0)':>7} | {'deficit%':>8} | {'dφ̃/dk̃|0':>9} | {'P_NN(S>0)':>9}")
        print("  " + "-" * 86)
        out[tag] = {"model_dir": mdir, "S0": S0, "r": r, "K_min": float(an.k_min),
                    "K_max": kmax, "insample_Kmin_pcterr_mean": float(np.mean(res)),
                    "insample_Kmin_pcterr_max": float(np.max(res)), "rows": {}}
        for T in T_values:
            kg = np.linspace(0.0, 0.03, 61)            # fine grid at the corner
            phi = phi_of_ktilde(an, T, kg)
            phi0 = float(phi[0])
            slope0 = float(np.polyfit(kg[:15], phi[:15], 1)[0])   # dφ̃/dk̃ at k̃->0
            P0 = -(S0 / kmax) * slope0                   # implied P(S_T>0); target 1
            km = float(k_min_tilde(T))
            phi_km = float(phi_of_ktilde(an, T, np.array([km]))[0])
            Nc0 = -np.log1p(-phi0)
            deficit = (1.0 - phi0) * 100.0
            print(f"  {T:>5.2f} | {km:>6.3f} | {phi_km:>9.4f} | {phi0:>7.4f} | "
                  f"{Nc0:>7.3f} | {deficit:>7.3f}% | {slope0:>9.3f} | {P0:>9.4f}")
            out[tag]["rows"][f"{T:.2f}"] = dict(k_min_tilde=km, phi_at_kmin=phi_km,
                phi0=phi0, N_c0=float(Nc0), deficit_pct=deficit,
                dphi_dk_at0=slope0, P_NN_ST_gt_0=float(P0))
        # local-vol skew at a representative maturity
        Kg = np.array([500, 750, 1000, 1500, 2000, 2500, 3000], dtype=np.float64)
        sig = sigma_loc(an, 0.5, Kg)
        print(f"\n  σ_loc(T=0.5, K) over K={Kg.astype(int).tolist()}:")
        print("    " + "  ".join(f"{s:.3f}" for s in sig)
              + f"   (skew = σ(500)/σ(2500) = {sig[0]/sig[-2]:.3f})")
        out[tag]["sigma_loc_T0.5"] = {int(k): float(s) for k, s in zip(Kg, sig)}
        print()

    # ---------- overlay figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor="white")
    kg = np.linspace(0.0, 0.55, 220)
    for tag, mdir, c in MODELS:
        an = analyzers[tag]
        for T, ls in [(0.25, "-"), (1.0, "--")]:
            phi = phi_of_ktilde(an, T, kg)
            axes[0].plot(kg, phi, color=c, ls=ls, lw=2, label=f"{tag}, T={T:g}")
            P = -(float(an.config.S0) / float(an.k_max)) * np.gradient(phi, kg)
            axes[1].plot(kg, P, color=c, ls=ls, lw=2, label=f"{tag}, T={T:g}")
        Kg = np.linspace(500, 3000, 120)
        axes[2].plot(Kg, sigma_loc(an, 0.5, Kg), color=c, lw=2.2, label=f"{tag}")

    an0 = analyzers["const-σ"]
    km_line = np.exp(-float(an0.config.r) * 0.5) * float(an0.k_min) / float(an0.k_max)
    for ax in axes[:2]:
        ax.axvline(km_line, color="0.6", ls=":", lw=1.3)
        ax.text(km_line + .005, ax.get_ylim()[0], " lowest trained K", color="0.4",
                fontsize=8, rotation=90, va="bottom")
    axes[0].axhline(1.0, color="0.4", ls="--", lw=1.2)
    axes[0].plot([0, 0.10], [1.0, 1.0 - 3 * 0.10], color="0.2", lw=1.0, alpha=0.7)
    axes[0].text(0.06, 0.80, "no-arb corner\n(φ̃=1, slope −3)", fontsize=8, color="0.3")
    axes[0].set(xlabel=r"$\tilde k = e^{-rT}K/K_{\max}$", ylabel=r"$\tilde\varphi(k,T)$",
                title="Normalized call near K=0\n(should reach 1 at the corner)", ylim=(0.0, 1.05))
    axes[1].axhline(1.0, color="0.4", ls="--", lw=1.2)
    axes[1].set(xlabel=r"$\tilde k$", ylabel=r"$P_{NN}(S_T>K)=-\frac{S_0}{K_{\max}}\partial_{\tilde k}\tilde\varphi$",
                title="Network-implied survival prob\n(should be 1 at K=0)", ylim=(0.0, 1.1))
    axes[2].set(xlabel="strike K", ylabel=r"$\sigma_{loc}$ (scaled NN units; real $\approx \div3$)",
                title="Local vol (T=0.5): flat vs mildly skewed")
    for ax in axes:
        ax.grid(True, ls="--", alpha=0.4); ax.legend(fontsize=8, framealpha=0.9)
    fig.suptitle("K=0 deficit: constant-σ vs Dupire — mechanism", fontweight="bold", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        p = os.path.join(args.output_dir, f"k0_discrepancy.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white"); print(f"Saved: {p}")
    with open(os.path.join(args.output_dir, "k0_discrepancy.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved: {os.path.join(args.output_dir, 'k0_discrepancy.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
