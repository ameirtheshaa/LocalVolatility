#!/usr/bin/env python3
"""
Direct check of the K=0 boundary hypothesis: evaluate the trained price network
NN_phi at the two domain boundaries and read off the raw output and the price.

The implemented call-price ansatz is

    phi_tilde(k,t) = 1 - exp(-N_c(k,t)),     N_c = NN_phi  (softplus output, > 0)

so phi_tilde ranges over (0, 1), OPEN at 1. The two boundary conditions are:

    K -> 0   (k_tilde = 0):  C_NN(0,T) = S0   <=>  phi_tilde = 1  <=>  N_c -> +inf
    K -> inf (k_tilde = 1):  C_NN(inf,T) = 0  <=>  phi_tilde = 0  <=>  N_c -> 0

The K=inf BC is reachable (N_c = 0 is attainable by softplus); the K=0 BC is NOT
(it needs N_c = +inf). So C_NN(0,T) < S0 structurally, and M_2 = e^{rT} C_NN(0,T)
sits below the forward by the resulting deficit.

This script evaluates N_c, phi_tilde and C_NN at k_tilde = 0 and k_tilde = 1 for
each maturity, cross-checks M_2 = e^{rT} C_NN(0,T) against the known IBP table, and
produces a k-sweep figure of phi_tilde(k) and N_c(k) over k in [0,1].

It uses the production evaluation path (prepare_data_for_nn + phi_tilde_from_nn);
nothing about the model or its data is modified.

Usage:
    python examples/check_nn_at_k0.py \\
        [--model-dir models/example_pretrained] \\
        [--maturities 0.5 1.0 1.5] \\
        [--output-dir plots/nn_k0_check]
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


# NTU palette (matches the deck)
NTU_NAVY = "#00205B"
NTU_RED = "#C02026"
NTU_GOLD = "#E8A33D"
T_COLORS = (NTU_NAVY, NTU_RED, NTU_GOLD, "#2E7D32", "#6A1B9A")

# Known IBP three-way M_2 (= e^{rT} C_NN(0,T)) for models/example_pretrained,
# used only as an independent cross-check that we are evaluating correctly.
M2_TABLE = {0.5: 948.93, 1.0: 967.74, 1.5: 982.95}


def resolve_model_dir(raw_path: str) -> str:
    """Accept absolute paths, repo-relative paths, or a bare run name."""
    if os.path.isabs(raw_path):
        return os.path.normpath(raw_path)
    normalized = raw_path.replace('\\', '/')
    candidates = [raw_path]
    if '/' not in normalized:
        candidates.append(os.path.join('models', 'runs', normalized))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    return os.path.normpath(raw_path)


def eval_point(analyzer, T, k_tilde_target):
    """Evaluate raw N_c, phi_tilde and C_NN at a single (T, k_tilde) point.

    We invert k_tilde = exp(-rT) K / k_max  ->  K = k_tilde * k_max * exp(rT)
    so that prepare_data_for_nn reproduces exactly the requested k_tilde, and we
    exercise the production feature transform rather than re-deriving it.
    """
    r = float(analyzer.config.r)
    S0 = float(analyzer.config.S0)
    k_max = float(analyzer.k_max)

    K = np.array([k_tilde_target * k_max * np.exp(r * T)], dtype=np.float64)
    t_tilde, k_tilde = analyzer.prepare_data_for_nn(T, K)

    # raw network output N_c (pre-exponential), and the transformed phi_tilde
    N_c = float(analyzer.nn_phi(tf.concat([t_tilde, k_tilde], axis=1)).numpy().flatten()[0])
    phi_tilde = float(analyzer.phi_tilde_from_nn(t_tilde, k_tilde).numpy().flatten()[0])

    # self-consistency: phi_tilde must equal 1 - exp(-N_c) in 'transformed' mode
    recon = 1.0 - np.exp(-N_c)
    assert abs(recon - phi_tilde) < 1e-5, (
        f"phi_tilde mismatch at T={T}, k_tilde={k_tilde_target}: "
        f"{phi_tilde} vs 1-exp(-N_c)={recon}"
    )

    C_NN = phi_tilde * S0
    M2 = float(np.exp(r * T) * C_NN)
    return {
        "k_tilde": float(k_tilde_target),
        "K": float(K[0]),
        "N_c": N_c,
        "phi_tilde": phi_tilde,
        "C_NN": C_NN,
        "M2": M2,
    }


def sweep(analyzer, T, n=200):
    """phi_tilde(k) and N_c(k) over a uniform k_tilde grid in [0,1]."""
    r = float(analyzer.config.r)
    k_max = float(analyzer.k_max)
    k_grid = np.linspace(0.0, 1.0, n, dtype=np.float64)
    K = k_grid * k_max * np.exp(r * T)
    t_tilde, k_tilde = analyzer.prepare_data_for_nn(T, K)
    N_c = analyzer.nn_phi(tf.concat([t_tilde, k_tilde], axis=1)).numpy().flatten()
    phi_tilde = analyzer.phi_tilde_from_nn(t_tilde, k_tilde).numpy().flatten()
    return k_grid, phi_tilde, N_c


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', default='models/example_pretrained')
    parser.add_argument('--maturities', type=float, nargs='+', default=[0.5, 1.0, 1.5])
    parser.add_argument('--output-dir', default='plots/nn_k0_check')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    model_dir = resolve_model_dir(args.model_dir)
    if not os.path.exists(model_dir):
        print(f"ERROR: model directory not found: {model_dir}", file=sys.stderr)
        return 1

    T_values = sorted(args.maturities)
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 96)
    print('DIRECT NN BOUNDARY CHECK  (price network NN_phi at k_tilde = 0 and k_tilde = 1)')
    print('=' * 96)
    print(f'Model      : {model_dir}')
    print(f'Maturities : {T_values}')

    config = DupirePipelineConfig.analysis_only(model_dir)
    config.analysis_config.T_analysis = T_values
    nn_phi, nn_eta, metadata = load_trained_models(model_dir)
    analyzer = PDFAnalyzer(nn_phi, nn_eta, config, metadata, phi_mapping='transformed')

    S0 = float(config.S0)
    r = float(config.r)
    print(f'S0 = {S0:g},  r = {r:g},  K_max = {analyzer.k_max:g},  T_max = {analyzer.t_max:g}')
    print(f'Ansatz: phi_tilde = 1 - exp(-N_c),  N_c = softplus output (> 0)  =>  phi_tilde in (0,1)')
    print()

    # ---- per-maturity evaluation at both boundaries ----
    header = (
        f"  {'T':>4} | {'N_c(0,T)':>9} | {'phi~(0,T)':>9} | {'C_NN(0,T)':>10} | "
        f"{'M2=e^rT*C':>10} | {'M2(table)':>9} | {'dM2':>7} | {'deficit%':>8} || "
        f"{'N_c(1,T)':>9} | {'phi~(1,T)':>9} | {'C_NN(1,T)':>10}"
    )
    print(header)
    print('  ' + '-' * (len(header) - 2))

    results = {}
    for T in T_values:
        zero = eval_point(analyzer, T, 0.0)   # k_tilde = 0  (K = 0)
        inf = eval_point(analyzer, T, 1.0)    # k_tilde = 1  (K = K_max e^{rT})
        deficit_pct = (S0 - zero["C_NN"]) / S0 * 100.0
        m2_tab = M2_TABLE.get(round(T, 4))
        if m2_tab is not None:
            dM2 = zero["M2"] - m2_tab
            m2_tab_s, dM2_s = f"{m2_tab:9.2f}", f"{dM2:+7.3f}"
        else:
            m2_tab_s, dM2_s = f"{'--':>9}", f"{'--':>7}"

        print(
            f"  {T:>4.2f} | {zero['N_c']:>9.4f} | {zero['phi_tilde']:>9.5f} | "
            f"{zero['C_NN']:>10.3f} | {zero['M2']:>10.3f} | {m2_tab_s} | {dM2_s} | "
            f"{deficit_pct:>7.3f}% || "
            f"{inf['N_c']:>9.5f} | {inf['phi_tilde']:>9.6f} | {inf['C_NN']:>10.5f}"
        )

        results[f"{T:.4f}"] = {
            "T": T,
            "ref_forward": float(S0 * np.exp(r * T)),
            "k0": zero,
            "kinf": inf,
            "deficit_pct": float(deficit_pct),
            "M2_table_crosscheck": m2_tab,
        }

    print('  ' + '-' * (len(header) - 2))
    print("\n  Reading: at k_tilde=0 the network outputs a FINITE N_c (~2.6), so")
    print("  phi_tilde = 1 - exp(-N_c) ~ 0.93 < 1  =>  C_NN(0,T) < S0 (the deficit).")
    print("  For the BC C_NN(0,T)=S0 we would need N_c -> +inf, which softplus cannot reach.")
    print("  At k_tilde=1 the network drives N_c -> 0, so phi_tilde -> 0, C_NN -> 0: that")
    print("  boundary IS reachable. The asymmetry is the whole story.")

    # ---- k-sweep figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for i, T in enumerate(T_values):
        k_grid, phi, N_c = sweep(analyzer, T)
        c = T_COLORS[i % len(T_COLORS)]
        ax1.plot(k_grid, phi, color=c, lw=2, label=rf"$T={T:g}$")
        ax2.plot(k_grid, N_c, color=c, lw=2, label=rf"$T={T:g}$")

    ax1.axhline(1.0, ls="--", color="0.4", lw=1.2)
    ax1.text(0.02, 1.0, r"$\tilde\varphi=1$ (K=0 target, unreachable)",
             color="0.35", fontsize=8, va="bottom", ha="left")
    ax1.set_xlabel(r"$k = e^{-rT}K/K_{\max}$")
    ax1.set_ylabel(r"$\tilde\varphi(k,T) = 1-e^{-N_c}$")
    ax1.set_title(r"Normalized price $\tilde\varphi$: lands $\approx 0.93$ at $k{=}0$, $\to 0$ at $k{=}1$",
                  fontsize=9)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(loc="center right", fontsize=8, framealpha=0.9)

    ax2.axhline(0.0, ls="--", color="0.4", lw=1.2)
    ax2.set_xlabel(r"$k = e^{-rT}K/K_{\max}$")
    ax2.set_ylabel(r"raw network output $N_c(k,T)$")
    ax2.set_title(r"$N_c$: finite plateau $\approx 2.6$ at $k{=}0$ (needs $\infty$), $\to 0$ at $k{=}1$",
                  fontsize=9)
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("Direct check: trained price network at the domain boundaries "
                 f"({model_dir})", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    for ext in ("png", "pdf"):
        out = os.path.join(args.output_dir, f"nn_at_k0.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
        print(f"\n  Saved: {out}")
    plt.close(fig)

    json_path = os.path.join(args.output_dir, "nn_at_k0.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"model_dir": model_dir, "S0": S0, "r": r,
                   "k_max": float(analyzer.k_max), "t_max": float(analyzer.t_max),
                   "results": results}, fh, indent=2)
    print(f"  Saved: {json_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
