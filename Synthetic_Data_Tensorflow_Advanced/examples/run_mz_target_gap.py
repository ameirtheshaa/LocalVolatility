#!/usr/bin/env python3
r"""
MZ-Dupire framing-4 — Step-4c EVIDENCE GATE (Codex's gate; RUN LATER).

Given an ALREADY-TRAINED vanilla PINN (NN_phi/NN_eta) + the real SPX surface,
this script answers the only question that decides whether the Step-4c L_MZ
energy-closure loss is worth wiring:

    Do the PINN-implied per-maturity log-return tail moments (excess kurtosis
    κ4, tail-energy E_tail) DISAGREE with what the Step-2 structure-preserving
    closure (mz_spectral/energy_closure.py) produces on the SAME maturities,
    and is a LARGER disagreement associated with WORSE held-out IV/price fit?

If the gap is ~0 everywhere, the PINN already lives on the closure manifold and
L_MZ buys nothing. If the gap is large where the holdout fit is bad, L_MZ has a
target to pull toward — 4c is worth building.

Two modes
---------
(default) target-gap:
    For each analysis maturity T:
      * PINN density  f_PINN(K) = e^{rT} ∂²C_NN/∂K²  (the differentiable core,
        reused read-only via PDFAnalyzer._raw_model_density).
      * PINN moments  (μ, s², skew, κ4) = energy_closure.log_return_moments,
        and E_tail_PINN = E_tail(coeffs_from_skew_kurt(skew, κ4)).
      * Step-2 MZ target: feed those moments through the closure (maxent tilt,
        the §5.6 manifestly-≥0 density), re-extract its moments → κ4_MZ,
        E_tail_MZ. The gap |κ4_PINN−κ4_MZ| / |E_tail_PINN−E_tail_MZ| measures
        how far the PINN tail is from the closure's reproducible manifold.
      * Holdout fit: per-maturity price RMSE on the held-out SPX CSV.
    Reports Pearson corr(gap, holdout_rmse) across maturities.

--check-step2-stability:
    Runs the closure's max-entropy moment-solve on EACH real SPX maturity's
    PINN moments and reports info['solved'] / max_moment_residual. Codex's
    caveat: if the closure solve FAILS exactly where the PINN tail needs the
    most help (large κ4), 4c is not ready — the target itself is unreliable
    there.

Emits mz_target_gap.json. Read-only w.r.t. energy_closure (imported as ec).

Run (AFTER a vanilla PINN exists):
  ./.venv/bin/python examples/run_mz_target_gap.py \
      --model-dir models/runs/mz_spx_vanilla \
      --market-csv ../SPX_Tensorflow/trainingDataSet.csv \
      --holdout-csv ../SPX_Tensorflow/testingDataSet.csv \
      --maturities 0.25 0.5 1.0
"""

import os
import sys
import json
import math
import argparse

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from config import DupirePipelineConfig
from dupire_pipeline import load_trained_models, PDFAnalyzer
from mz_spectral import energy_closure as ec


def _pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float("nan")
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _load_market(csv_path, option_type):
    """Minimal market loader for the holdout fit (Maturity, Strike, Option price,
    Option type). Mirrors DataGenerator.from_market_csv's column resolution."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    def _find(cands):
        norm = {c.replace("\n", " ").strip().lower(): c for c in df.columns}
        for cand in cands:
            k = cand.replace("\n", " ").strip().lower()
            if k in norm:
                return norm[k]
        return None

    col_T = _find(["Maturity"])
    col_K = _find(["Strike"])
    col_phi = _find(["Option price", "Option\nprice"])
    col_type = _find(["Option type", "Option\ntype"])
    if col_type is not None:
        m = df[col_type].astype("float64").round().astype("int64") == option_type
        df = df.loc[m].reset_index(drop=True)
    return (df[col_T].to_numpy(float), df[col_K].to_numpy(float),
            df[col_phi].to_numpy(float))


def _wide_K_grid(T, S0, r, n=4000, span=5.0):
    """Wide [~0, span·forward] strike grid for the PINN density / moments."""
    K_hi = max(span * S0 * math.exp(r * T), span * S0)
    return np.linspace(K_hi / n, K_hi, n)


def main():
    p = argparse.ArgumentParser(description="MZ-Dupire Step-4c evidence gate")
    p.add_argument("--model-dir", required=True,
                   help="dir with NN_phi_final.keras / NN_eta_final.keras / metadata.json")
    p.add_argument("--market-csv", default="../SPX_Tensorflow/trainingDataSet.csv")
    p.add_argument("--holdout-csv", default="../SPX_Tensorflow/testingDataSet.csv")
    p.add_argument("--option-type", type=int, default=2)
    p.add_argument("--maturities", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0],
                   help="maturities at which to compare PINN vs MZ-closure moments")
    p.add_argument("--skew-kurt-ratio", type=float, default=None,
                   help="optional c4/c3 direction for coeffs_from_E_tail (diagnostic)")
    p.add_argument("--check-step2-stability", action="store_true",
                   help="run the closure moment-solve per maturity; report convergence")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    nn_phi, nn_eta, metadata = load_trained_models(args.model_dir)

    # Reconstruct a config whose S0/r/scaling match the trained model.
    config = DupirePipelineConfig.full_training()
    scaling = metadata.get("scaling", {}) if isinstance(metadata, dict) else {}
    S0 = float(scaling.get("S0", config.S0))
    r = float(scaling.get("r", config.r))
    config.S0, config.r = S0, r
    if "t_max" in scaling:
        config.T_max = float(scaling["t_max"])
    if "k_max" in scaling:
        config.K_max = float(scaling["k_max"])
    if "k_min" in scaling:
        config.K_min = float(scaling["k_min"])

    analyzer = PDFAnalyzer(nn_phi, nn_eta, config, metadata, phi_mapping="transformed")

    # ---- holdout market data for the fit-correlation leg ----
    T_h, K_h, phi_h = _load_market(args.holdout_csv, args.option_type)

    per_T = {}
    gaps_k4, gaps_Etail, holdout_rmse = [], [], []
    stability = {}

    for T in args.maturities:
        T = float(T)
        K_wide = _wide_K_grid(T, S0, r)

        # --- PINN density via the differentiable core (read-only) ---
        f_pinn = analyzer._raw_model_density(T, K_wide)
        mu_p, s2_p, skew_p, k4_p = ec.log_return_moments(f_pinn, K_wide, S0)
        coeffs_p = ec.coeffs_from_skew_kurt(skew_p, k4_p)
        Etail_p = ec.E_tail(coeffs_p)

        # --- Step-2 MZ target: closure round-trip on the PINN moments ---
        # Build the manifestly-≥0 maxent tilt matching (skew_p, k4_p), place it
        # on the same grid, re-extract its moments. The closure's reproducible
        # tail is (skew_MZ, k4_MZ); the gap is what L_MZ would have to close.
        f_mz, aux_mz = ec.closure_density_on_K(
            K_wide, s2_p, coeffs_p, S0, r, T, kind="maxent")
        mu_m, s2_m, skew_m, k4_m = ec.log_return_moments(f_mz, K_wide, S0)
        coeffs_m = ec.coeffs_from_skew_kurt(skew_m, k4_m)
        Etail_m = ec.E_tail(coeffs_m)

        gap_k4 = abs(k4_p - k4_m) if (np.isfinite(k4_p) and np.isfinite(k4_m)) else float("nan")
        gap_Et = abs(Etail_p - Etail_m) if (np.isfinite(Etail_p) and np.isfinite(Etail_m)) else float("nan")

        # --- holdout fit at this maturity (nearest maturity bucket) ---
        if T_h.size:
            j = np.isclose(T_h, T, atol=max(1e-3, 0.02 * max(T, 1e-6)))
            if not j.any():  # nearest single maturity
                j = np.zeros_like(T_h, bool)
                j[int(np.argmin(np.abs(T_h - T)))] = True
            Kj, phij = K_h[j], phi_h[j]
            t_tilde, k_tilde = analyzer.prepare_data_for_nn(T, Kj.astype(np.float32))
            phi_tilde_nn = analyzer.phi_tilde_from_nn(t_tilde, k_tilde)
            c_nn = (S0 * phi_tilde_nn.numpy().ravel())
            rmse = float(np.sqrt(np.mean((c_nn - phij) ** 2))) if phij.size else float("nan")
            n_h = int(phij.size)
        else:
            rmse, n_h = float("nan"), 0

        gaps_k4.append(gap_k4)
        gaps_Etail.append(gap_Et)
        holdout_rmse.append(rmse)

        rec = {
            "T": T,
            "pinn_moments": {"mu": mu_p, "s2": s2_p, "skew": skew_p, "kappa4": k4_p,
                             "E_tail": Etail_p},
            "mz_closure_moments": {"mu": mu_m, "s2": s2_m, "skew": skew_m, "kappa4": k4_m,
                                   "E_tail": Etail_m},
            "gap_kappa4": gap_k4,
            "gap_E_tail": gap_Et,
            "holdout_price_rmse": rmse,
            "holdout_n_points": n_h,
            "maxent_solved": bool(aux_mz.get("solved", False)),
            "maxent_max_moment_residual": float(aux_mz.get("max_moment_residual", float("nan"))),
        }
        per_T[f"{T:.4f}"] = rec
        print(f"  T={T:.3f} | κ4_PINN={k4_p:+.4f} κ4_MZ={k4_m:+.4f} "
              f"|Δκ4|={gap_k4:.4f} | E_tail_PINN={Etail_p:.3e} E_tail_MZ={Etail_m:.3e} "
              f"|ΔE|={gap_Et:.3e} | holdout RMSE={rmse:.4f} (n={n_h}) | "
              f"maxent_solved={rec['maxent_solved']}")

        # ---- stability mode: explicit closure-solve convergence report ----
        if args.check_step2_stability:
            # solve the tilt λ's directly on the PINN target moments
            lam, info = ec.maxent_lambdas_for_coeffs(coeffs_p)
            stability[f"{T:.4f}"] = {
                "target_skew": skew_p, "target_kappa4": k4_p,
                "solved": bool(info.get("solved", False)),
                "max_moment_residual": float(info.get("max_moment_residual", float("nan"))),
                "achieved_skew": float(info.get("achieved_skew", float("nan"))),
                "achieved_exkurt": float(info.get("achieved_exkurt", float("nan"))),
                "lambdas": {str(k): float(v) for k, v in lam.items()},
            }
            s = stability[f"{T:.4f}"]
            print(f"      [stability] solved={s['solved']} "
                  f"resid={s['max_moment_residual']:.2e} "
                  f"(target κ4={k4_p:+.3f}, achieved κ4={s['achieved_exkurt']:+.3f})")

    corr_k4 = _pearson(gaps_k4, holdout_rmse)
    corr_Et = _pearson(gaps_Etail, holdout_rmse)

    # decision summary (NOT a verdict — that's for the human after the runs)
    finite_k4 = [g for g in gaps_k4 if np.isfinite(g)]
    summary = {
        "model_dir": args.model_dir,
        "S0": S0, "r": r,
        "maturities": [float(t) for t in args.maturities],
        "per_maturity": per_T,
        "corr_gap_kappa4_vs_holdout_rmse": corr_k4,
        "corr_gap_Etail_vs_holdout_rmse": corr_Et,
        "max_abs_gap_kappa4": (max(finite_k4) if finite_k4 else float("nan")),
        "mean_abs_gap_kappa4": (float(np.mean(finite_k4)) if finite_k4 else float("nan")),
        "all_maxent_solved": all(
            v.get("maxent_solved", False) for v in per_T.values()),
        "stability": stability if args.check_step2_stability else None,
        "note": ("Gate reading: 4c is worth wiring iff the κ4/E_tail gap is "
                 "MATERIAL and positively correlated with holdout RMSE AND the "
                 "closure solve converges where κ4 is large. Near-zero gap or "
                 "non-convergence-where-needed => 4c not ready / not worth it."),
    }
    out = args.output or os.path.join(args.model_dir, "mz_target_gap.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  corr(|Δκ4|, holdout RMSE)   = {corr_k4:+.4f}")
    print(f"  corr(|ΔE_tail|, holdout RMSE) = {corr_Et:+.4f}")
    print(f"  max |Δκ4| over maturities    = {summary['max_abs_gap_kappa4']:.4f}")
    print(f"  all maxent solves converged  = {summary['all_maxent_solved']}")
    print(f"  wrote {out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
