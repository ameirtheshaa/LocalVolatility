#!/usr/bin/env python3
"""
MZ-Dupire Step 4 — GPU smoke train (gpu-laptop MCP).

Paired arms: Arm A (lambda_mart=0, lambda_pos=0) vs Arm B (+martingale+positivity).
2 seeds × 200 synthetic epochs each.

Success criteria:
  - No NaN losses in either arm
  - L_mart > 0 in Arm B (non-trivial martingale penalty)
  - L_pos > 0 in Arm B  (non-trivial positivity penalty)
  - Arm A baseline: L_mart == 0, L_pos == 0 (strict-extension control)

Self-contained — all imports from C:/mcp_jobs/scripts/ where dupire_pipeline.py
and config.py are co-located (uploaded via gpu-laptop MCP upload_script).
"""

import os, sys, json, time
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx("float32")

# Import config first, then pipeline
from config import DupirePipelineConfig
from dupire_pipeline import (
    DataGenerator, DupireNeuralModel, ModelTrainer, save_metadata, data_type,
)

N_SEEDS  = 2
N_EPOCHS = 200
PRINT_EVERY = 50

# ---------------------------------------------------------------------------

def _f(x):
    """Safely convert TF tensor or float to Python float."""
    try:
        return float(x.numpy()) if hasattr(x, "numpy") else float(x)
    except Exception:
        return float("nan")


def _build_cfg(lm: float, lp: float) -> DupirePipelineConfig:
    cfg = DupirePipelineConfig.full_training()
    cfg.mode = "train"
    cfg.real_data = False
    cfg.num_epochs = N_EPOCHS
    cfg.print_epochs = PRINT_EVERY
    cfg.save_epochs = N_EPOCHS
    cfg.skip_if_exists = False
    cfg.save_training_data = False
    # Smoke scale-down
    cfg.M_train    = 400
    cfg.N_maturities = 4
    cfg.N_strikes  = 16
    # Step-4 knobs
    cfg.lambda_mart = lm
    cfg.lambda_pos  = lp
    cfg.lambda_mz   = 0.0
    cfg.mart_kgrid_n = 64
    cfg.mart_kgrid_kmax_mult = 1.5
    return cfg


def run_paired_seed(seed: int):
    """Train Arm A then Arm B from the same init weights. Return summary dict."""
    tf.random.set_seed(seed * 7919 + 1)
    np.random.seed(seed * 7919 + 1)

    # --- Generate data once (same for both arms) ---
    cfg0 = _build_cfg(0.0, 0.0)
    dg = DataGenerator(cfg0)
    T_nn, K_nn, phi_ref = dg.get_training_data()  # runs MC, returns tensors
    t_tilde, k_tilde = dg.scale_data(T_nn, K_nn)
    phi_tilde_ref = phi_ref / cfg0.S0

    t_min = float(tf.reduce_min(t_tilde).numpy())
    t_max = float(tf.reduce_max(t_tilde).numpy())
    k_min = float(tf.reduce_min(k_tilde).numpy())
    k_max = float(tf.reduce_max(k_tilde).numpy())

    # --- Build reference model → capture init weights ---
    model0 = DupireNeuralModel(cfg0, dg)
    model0.build_models()
    # Capture sub-network weights for deterministic paired-arm comparison
    init_phi_ws = model0.NN_phi_tilde.get_weights()
    init_eta_ws = model0.NN_eta_tilde.get_weights()
    del model0

    arm_results = {}
    for arm, (lm, lp) in [("A", (0.0, 0.0)), ("B", (1.0, 1.0))]:
        cfg = _build_cfg(lm, lp)
        model = DupireNeuralModel(cfg, dg)
        model.build_models()
        # Restore same init weights so both arms start identically
        model.NN_phi_tilde.set_weights(init_phi_ws)
        model.NN_eta_tilde.set_weights(init_eta_ws)

        trainer = ModelTrainer(model, cfg)
        t0 = time.perf_counter()
        trainer.train(
            T_nn, K_nn, phi_ref,
            t_tilde, k_tilde, phi_tilde_ref,
            t_min, t_max, k_min, k_max,
            output_dir=".",
        )
        elapsed = time.perf_counter() - t0

        # One final eval for clean loss values
        lp_tf = tf.constant(cfg.lambda_pde, dtype=data_type)
        lr_tf = tf.constant(cfg.lambda_reg, dtype=data_type)
        pos_scale = tf.constant(1.0, dtype=data_type)
        f_phi, f_dup, f_reg, f_mart, f_pos = model.train_step(
            t_tilde, k_tilde, phi_tilde_ref,
            t_min, t_max, k_min, k_max,
            lp_tf, lr_tf, float(lm), float(lp),
            lambda_pos_scale=pos_scale,
        )
        arm_results[arm] = {
            "lambda_mart": lm, "lambda_pos": lp,
            "loss_phi":  _f(f_phi),
            "loss_dup":  _f(f_dup),
            "loss_mart": _f(f_mart),
            "loss_pos":  _f(f_pos),
            "mart_nonzero": abs(_f(f_mart)) > 1e-12,
            "pos_nonzero":  abs(_f(f_pos))  > 1e-12,
            "any_nan": any(
                not np.isfinite(_f(x))
                for x in [f_phi, f_dup, f_reg, f_mart, f_pos]
            ),
            "elapsed_s": elapsed,
        }
        print(
            f"  Seed {seed} Arm {arm}: "
            f"phi={_f(f_phi):.4e} dup={_f(f_dup):.4e} "
            f"mart={_f(f_mart):.4e} pos={_f(f_pos):.4e} "
            f"({elapsed:.1f}s)"
        )

    return {
        "seed": seed,
        "arm_A": arm_results["A"],
        "arm_B": arm_results["B"],
        "mart_nonzero_B":   arm_results["B"]["mart_nonzero"],
        "pos_nonzero_B":    arm_results["B"]["pos_nonzero"],
        "no_nan_A":         not arm_results["A"]["any_nan"],
        "no_nan_B":         not arm_results["B"]["any_nan"],
        "baseline_arm_A_ok": (
            not arm_results["A"]["mart_nonzero"] and
            not arm_results["A"]["pos_nonzero"]
        ),
    }


def main():
    print("=" * 70)
    print("MZ-Dupire Step 4 — GPU smoke train")
    print(f"  N_SEEDS={N_SEEDS}  N_EPOCHS={N_EPOCHS}")
    print(f"  GPU: {tf.config.list_physical_devices('GPU')}")
    print("=" * 70)

    per_seed = []
    for s in range(N_SEEDS):
        print(f"\n=== Seed {s} ===")
        per_seed.append(run_paired_seed(s))

    agg = {
        "n_seeds":            N_SEEDS,
        "n_epochs":           N_EPOCHS,
        "all_mart_nonzero_B": all(r["mart_nonzero_B"]   for r in per_seed),
        "all_pos_nonzero_B":  all(r["pos_nonzero_B"]    for r in per_seed),
        "all_no_nan_A":       all(r["no_nan_A"]          for r in per_seed),
        "all_no_nan_B":       all(r["no_nan_B"]          for r in per_seed),
        "all_baseline_arm_A": all(r["baseline_arm_A_ok"] for r in per_seed),
        "per_seed":           per_seed,
    }
    agg["PASS"] = all([
        agg["all_mart_nonzero_B"],
        agg["all_pos_nonzero_B"],
        agg["all_no_nan_A"],
        agg["all_no_nan_B"],
        agg["all_baseline_arm_A"],
    ])

    out = "mz_pinn_step4_smoke.json"
    with open(out, "w") as fh:
        json.dump(agg, fh, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"OVERALL PASS = {agg['PASS']}")
    for k in ["all_mart_nonzero_B", "all_pos_nonzero_B",
              "all_no_nan_A", "all_no_nan_B", "all_baseline_arm_A"]:
        print(f"  {k} = {agg[k]}")
    print(f"Written: {out}")
    print("=" * 70)

    print("=== JSON_RESULTS_BEGIN ===")
    print(json.dumps(agg, default=str))
    print("=== JSON_RESULTS_END ===")


if __name__ == "__main__":
    main()
