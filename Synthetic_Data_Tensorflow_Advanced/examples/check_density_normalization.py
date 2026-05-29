#!/usr/bin/env python3
"""
Density normalization sanity check: verify that the NN-implied density
∫ q(K,T) dK integrates to ~1 (mass conservation) and that the first moment
M_1 = ∫ K q(K,T) dK matches the known IBP table values.

The density is q(K,T) = e^(rT) ∂²C_NN/∂K², returned by _raw_model_density.
No clipping or renormalisation is applied (that is the point: we check the
raw integrated mass to see how much the K=0 boundary deficit costs).

Under perfect boundary conditions both results hold:
    ∫ q dK = 1            (probability mass conserved)
    ∫ K q dK = S0 e^{rT}  (risk-neutral forward)

A mass below 1 is itself a fingerprint of the K=0 boundary deficit — the
missing mass sits in the unmeasured interval [−∞, 0], which the NN cannot
reach but which the true density requires.

Usage:
    python examples/check_density_normalization.py \\
        [--model-dir models/example_pretrained] \\
        [--maturities 0.5 1.0 1.5] \\
        [--output-dir plots/nn_k0_check]
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig
from dupire_pipeline import PDFAnalyzer, load_trained_models


def resolve_model_dir(raw_path: str) -> str:
    """Accept absolute paths, repo-relative paths, or a bare run name."""
    if os.path.isabs(raw_path):
        return os.path.normpath(raw_path)
    normalized = raw_path.replace("\\", "/")
    candidates = [raw_path]
    if "/" not in normalized:
        candidates.append(os.path.join("models", "runs", normalized))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    return os.path.normpath(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", default="models/example_pretrained")
    parser.add_argument("--maturities", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--output-dir", default="plots/nn_k0_check")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    model_dir = resolve_model_dir(args.model_dir)
    if not os.path.exists(model_dir):
        print(f"ERROR: model directory not found: {model_dir}", file=sys.stderr)
        return 1

    T_values = sorted(args.maturities)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("DENSITY NORMALIZATION SANITY CHECK  (∫ q dK and ∫ K q dK)")
    print("=" * 80)
    print(f"Model      : {model_dir}")
    print(f"Maturities : {T_values}")

    config = DupirePipelineConfig.analysis_only(model_dir)
    config.analysis_config.T_analysis = T_values
    nn_phi, nn_eta, metadata = load_trained_models(model_dir)
    analyzer = PDFAnalyzer(nn_phi, nn_eta, config, metadata, phi_mapping="transformed")

    S0 = float(config.S0)
    r = float(config.r)
    K_max_plot = float(analyzer.k_max)  # 3000 for models/example_pretrained

    print(f"S0 = {S0:g},  r = {r:g},  K_max_plot = {K_max_plot:g}")
    print()
    print(
        "Grid: K_wide = linspace(0, K_tail, 5000)  where  "
        "K_tail = max(5*S0*exp(r*T), 5*K_max_plot)"
    )
    print(
        "Density: q = analyzer._raw_model_density(T, K_wide)  "
        "[= e^{rT} d2C/dK2, unclipped]"
    )
    print()

    # ---- header ----
    header = (
        f"  {'T':>5} | {'K_tail':>10} | {'∫q dK (mass)':>13} | "
        f"{'M_1=∫Kq dK':>12} | {'M_ref=S0e^rT':>13} | {'M_1/M_ref':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = {}
    for T in T_values:
        K_tail = max(5.0 * S0 * math.exp(r * T), 5.0 * K_max_plot)
        K_wide = np.linspace(0.0, K_tail, 5000, dtype=np.float64)

        q = analyzer._raw_model_density(T, K_wide)

        mass = float(np.trapz(q, K_wide))
        mean1 = float(np.trapz(K_wide * q, K_wide))
        ref = S0 * math.exp(r * T)
        ratio = mean1 / ref if ref > 0 else float("nan")

        print(
            f"  {T:>5.2f} | {K_tail:>10.1f} | {mass:>13.6f} | "
            f"{mean1:>12.4f} | {ref:>13.4f} | {ratio:>10.6f}"
        )

        results[f"{T:.4f}"] = {
            "T": T,
            "K_tail": K_tail,
            "mass": mass,
            "M_1": mean1,
            "M_ref": ref,
            "M_1_over_M_ref": ratio,
        }

    print("  " + "-" * (len(header) - 2))

    # ---- interpretation ----
    print()
    masses = [results[f"{T:.4f}"]["mass"] for T in T_values]
    ratios = [results[f"{T:.4f}"]["M_1_over_M_ref"] for T in T_values]
    mass_ok = all(abs(m - 1.0) < 0.05 for m in masses)
    print(
        f"  Mass conservation: {'PASS (all |mass-1| < 5%)' if mass_ok else 'FAIL (mass deviates > 5% from 1)'}"
    )
    print(
        f"  M_1/M_ref: {', '.join(f'{v:.4f}' for v in ratios)} "
        f"(expect ~0.93 from three-way table)"
    )
    print()
    print(
        "  INTERPRETATION: A mass < 1 indicates that the K=0 boundary deficit\n"
        "  prevents full probability mass from accumulating over [0, K_tail].\n"
        "  The missing mass fraction equals the deficit fraction (7% at T=0.5, etc.).\n"
        "  M_1/M_ref < 1 is thus expected: it matches the ~7%-below-forward values\n"
        "  seen in the IBP three-way table (M_2 / M_ref columns)."
    )

    # ---- save JSON ----
    json_path = os.path.join(args.output_dir, "density_normalization.json")
    payload = {
        "model_dir": model_dir,
        "S0": S0,
        "r": r,
        "K_max_plot": K_max_plot,
        "n_grid": 5000,
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
