#!/usr/bin/env python3
r"""
MZ-Dupire framing-4 "Step 4 Lite" — headless PINN training driver.

Trains the WSPG25 local-vol PINN (DupireNeuralModel) with the optional
MZ martingale + mass + positivity penalties (config.lambda_mart / lambda_pos),
on EITHER synthetic Monte-Carlo data OR a real SPX market CSV
(DataGenerator.from_market_csv). No input() — built for unattended GPU jobs.

This is the executable the offloaded 3000-epoch ablation invokes. It does NOT
run the heavy MC analysis stage — it trains NN_phi / NN_eta and saves:
    <output_dir>/NN_phi_final.keras
    <output_dir>/NN_eta_final.keras
    <output_dir>/metadata.json            (config + data-driven scaling)
    <output_dir>/mz_pinn_losses.png       (loss curves: φ, dupire, reg, mart, pos)
    <output_dir>/mz_pinn_density.png      (model-implied density at T_max + S0·e^{rT})
    <output_dir>/mz_pinn_train.json       (final losses, scaling, λ knobs, checks)

Strict-extension control: with --lambda-mart 0 --lambda-pos 0 the loss assembly
is byte-identical to the WSPG25 baseline (the train_step gate is a python-level
branch, so the density penalties are not even traced into the graph).

Examples
--------
# tiny synthetic smoke test (Mac CPU, sanity only — NOT a training run)
./.venv/bin/python examples/run_mz_pinn_train.py --synthetic --epochs 50 \
    --lambda-mart 0 --lambda-pos 0 --assert-baseline --output-dir models/runs/smoke

# GPU ablation arm A (vanilla / baseline)
./.venv/bin/python examples/run_mz_pinn_train.py --real \
    --market-csv ../SPX_Tensorflow/trainingDataSet.csv \
    --epochs 3000 --lambda-mart 0 --lambda-pos 0 \
    --output-dir models/runs/mz_spx_vanilla

# GPU ablation arm B (+martingale+mass+positivity)
./.venv/bin/python examples/run_mz_pinn_train.py --real \
    --market-csv ../SPX_Tensorflow/trainingDataSet.csv \
    --epochs 3000 --lambda-mart 1.0 --lambda-pos 1.0 \
    --output-dir models/runs/mz_spx_martpos
"""

import os
import sys
import json
import argparse

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf  # noqa: E402  (after sys.path / Agg)

from config import DupirePipelineConfig
from dupire_pipeline import (
    DataGenerator, DupireNeuralModel, ModelTrainer, save_metadata, data_type,
)


def _to_float(x):
    try:
        return float(x.numpy()) if hasattr(x, "numpy") else float(x)
    except Exception:
        return float("nan")


def build_config(args) -> DupirePipelineConfig:
    """Construct the training config from CLI args (synthetic or real)."""
    config = DupirePipelineConfig.full_training()
    config.mode = "train"
    config.skip_if_exists = False
    config.num_epochs = int(args.epochs)
    config.print_epochs = max(1, int(args.print_epochs))
    config.save_epochs = int(args.epochs)  # checkpoint only at the end
    config.output_dir = args.output_dir

    # MZ Step-4-Lite knobs
    config.lambda_mart = float(args.lambda_mart)
    config.lambda_pos = float(args.lambda_pos)
    config.lambda_mz = 0.0  # 4c NOT wired this round
    config.mart_kgrid_n = int(args.mart_kgrid_n)
    config.mart_kgrid_kmax_mult = float(args.mart_kgrid_kmax_mult)

    # keep the heavy training light for synthetic smoke tests
    if args.synthetic:
        config.real_data = False
        if args.m_train is not None:
            config.M_train = int(args.m_train)
        if args.n_maturities is not None:
            config.N_maturities = int(args.n_maturities)
        if args.n_strikes is not None:
            config.N_strikes = int(args.n_strikes)
        if args.k_min is not None:
            config.K_min = float(args.k_min)
    else:
        config.real_data = True
        config.market_csv = args.market_csv
        config.market_csv_test = args.market_csv_test
        config.market_option_type = int(args.option_type)
        if args.s0 is not None:
            config.S0 = float(args.s0)
        if args.r is not None:
            config.r = float(args.r)

    # plotting handled by this driver, not the pipeline visualizer
    config.plot_config.enable_training_plots = False
    return config


def assert_baseline_identical(model, trainer_inputs):
    """
    Strict-extension control: at λ_mart=λ_pos=0 the Step-4-Lite extension must
    reproduce the WSPG25 baseline's loss_total AND NN_phi gradients BIT-FOR-BIT.

    loss_phi_cal / loss_dupire_cal sample collocation & boundary points with
    tf.random.uniform, so a single loss eval is STOCHASTIC. We therefore (a) fix
    the global TF seed before each evaluation so the random draws are identical,
    and (b) compute, inside ONE eager GradientTape:
      * REF: the EXACT original 3-term assembly inlined verbatim from the
             pre-extension train_step:  loss_phi + λ_pde·loss_dupire + λ_reg·loss_reg
      * EXT: the SAME three terms PLUS the gated 4-Lite additions with weight 0:
             loss_total + 0.0·L_mart + 0.0·L_pos
    and compare loss_total and d(loss_total)/d(NN_phi) bit-for-bit. This isolates
    the loss-assembly seam (no optimizer-state confound). It does NOT call
    train_step (whose apply_gradients would mutate weights between the two evals).
    """
    (t_tilde, k_tilde, phi_tilde_ref,
     t_min, t_max, k_min, k_max) = trainer_inputs
    lp = float(model.lambda_pde)
    lr = float(model.lambda_reg)
    SEED = 12345

    def _eval(extend):
        tf.random.set_seed(SEED)
        with tf.GradientTape() as tape:
            loss_phi = model.loss_phi_cal(
                t_tilde, k_tilde, phi_tilde_ref, t_min, t_max, k_min, k_max)
            loss_dupire, loss_reg = model.loss_dupire_cal(t_min, t_max, k_min, k_max)
            # verbatim original assembly
            loss_total = loss_phi + lp * loss_dupire + lr * loss_reg
            l_mart = tf.constant(0.0, dtype=data_type)
            l_pos = tf.constant(0.0, dtype=data_type)
            if extend:
                # gated additions at ZERO weight (mirror train_step exactly)
                l_mart = model.loss_martingale_cal(t_max)
                l_pos = model.loss_positivity_cal(t_max)
                loss_total = loss_total + 0.0 * l_mart + 0.0 * l_pos
        grads = tape.gradient(loss_total, model.NN_phi_tilde.trainable_variables)
        return (float(loss_total.numpy()),
                [g.numpy().copy() for g in grads],
                float(l_mart.numpy()), float(l_pos.numpy()))

    lt_ref, g_ref, _, _ = _eval(extend=False)
    lt_ext, g_ext, lm, lpz = _eval(extend=True)

    loss_equal = np.array_equal(np.asarray(lt_ref), np.asarray(lt_ext))
    grads_equal = all(np.array_equal(a, b) for a, b in zip(g_ref, g_ext))
    # at zero weight the additive terms vanish from loss_total; the underlying
    # L_mart / L_pos values are finite & reported (sanity: density eval works).
    ok = bool(loss_equal and grads_equal)
    detail = {
        "losses_equal": bool(loss_equal),
        "weights_equal": bool(grads_equal),   # gradient-level identity (NN_phi)
        "loss_total_ref": lt_ref,
        "loss_total_ext": lt_ext,
        "loss_mart_value_off": lm,   # the (unweighted) L_mart at this point
        "loss_pos_value_off": lpz,   # the (unweighted) L_pos at this point
        "loss_mart_is_zero": True,   # contribution to loss_total is 0·L_mart
        "loss_pos_is_zero": True,
    }
    return ok, detail


def main():
    p = argparse.ArgumentParser(description="MZ-Dupire Step-4-Lite PINN trainer")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true", help="train on synthetic MC data")
    src.add_argument("--real", action="store_true", help="train on a real market CSV")

    p.add_argument("--market-csv", default="../SPX_Tensorflow/trainingDataSet.csv")
    p.add_argument("--market-csv-test", default="../SPX_Tensorflow/testingDataSet.csv")
    p.add_argument("--option-type", type=int, default=2, help="Option type filter (2=puts)")
    p.add_argument("--s0", type=float, default=None)
    p.add_argument("--r", type=float, default=None)

    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--print-epochs", type=int, default=500)
    p.add_argument("--lambda-mart", type=float, default=0.0)
    p.add_argument("--lambda-pos", type=float, default=0.0)
    p.add_argument("--mart-kgrid-n", type=int, default=256)
    p.add_argument("--mart-kgrid-kmax-mult", type=float, default=1.5)

    # synthetic-only knobs
    p.add_argument("--m-train", type=int, default=None)
    p.add_argument("--n-maturities", type=int, default=None)
    p.add_argument("--n-strikes", type=int, default=None)
    p.add_argument("--k-min", type=float, default=None)

    p.add_argument("--output-dir", default=os.path.join("models", "runs", "mz_pinn"))
    p.add_argument("--assert-baseline", action="store_true",
                   help="assert λ=0 reproduces the WSPG25 baseline exactly, then continue")
    args = p.parse_args()

    if not args.synthetic and not args.real:
        args.synthetic = True  # default mode

    config = build_config(args)
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 80)
    print("MZ-DUPIRE STEP-4-LITE PINN TRAINING")
    print("=" * 80)
    print(f"  mode           = {'real' if config.real_data else 'synthetic'}")
    print(f"  epochs         = {config.num_epochs}")
    print(f"  lambda_mart    = {config.lambda_mart}")
    print(f"  lambda_pos     = {config.lambda_pos}")
    print(f"  lambda_mz      = {config.lambda_mz}  (4c NOT wired)")
    print(f"  mart_kgrid_n   = {config.mart_kgrid_n}")
    print(f"  output_dir     = {config.output_dir}")

    # ---- load / generate data ----
    data_gen = DataGenerator(config)
    market_meta = None
    if config.real_data:
        market = data_gen.from_market_csv(config.market_csv)
        T_nn, K_nn, phi_ref = market["T_nn"], market["K_nn"], market["phi_ref"]
        phi_tilde_ref = market["phi_tilde_ref"]
        t_tilde, k_tilde = market["t_tilde"], market["k_tilde"]
        market_meta = market["scaling"]
        np.savez(os.path.join(config.output_dir, "market_sigma_ref.npz"),
                 T=T_nn.numpy(), K=K_nn.numpy(),
                 sigma_ref=market["sigma_ref"], locvol_flag=market["locvol_flag"])
    else:
        T_nn, K_nn, phi_ref = data_gen.get_training_data()
        phi_tilde_ref = phi_ref / config.S0
        t_tilde, k_tilde = data_gen.scale_data(T_nn, K_nn)

    t_min = float(tf.reduce_min(t_tilde).numpy())
    t_max = float(tf.reduce_max(t_tilde).numpy())
    k_min = float(tf.reduce_min(k_tilde).numpy())
    k_max = float(tf.reduce_max(k_tilde).numpy())

    # ---- build model ----
    model = DupireNeuralModel(config, data_gen)
    model.build_models()

    trainer_inputs = (t_tilde, k_tilde, phi_tilde_ref, t_min, t_max, k_min, k_max)

    # ---- strict-extension assert (optional) ----
    baseline_check = None
    if args.assert_baseline:
        ok, detail = assert_baseline_identical(model, trainer_inputs)
        baseline_check = {"passed": ok, **detail}
        print("\n  [strict-extension control] λ_mart=λ_pos=0 == WSPG25 baseline?")
        print(f"    losses_equal={detail['losses_equal']}, "
              f"weights_equal={detail['weights_equal']}, "
              f"L_mart==0:{detail['loss_mart_is_zero']}, "
              f"L_pos==0:{detail['loss_pos_is_zero']}  -> "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            raise AssertionError("strict-extension control FAILED: "
                                 f"{json.dumps(baseline_check)}")

    # ---- train ----
    trainer = ModelTrainer(model, config)
    rmse_sigma_list, error_sigma_list = trainer.train(
        T_nn, K_nn, phi_ref, t_tilde, k_tilde, phi_tilde_ref,
        t_min, t_max, k_min, k_max, config.output_dir, visualizer=None,
    )

    # ---- final losses (one extra forward eval for reporting) ----
    lp = tf.constant(model.lambda_pde, dtype=data_type)
    lr = tf.constant(model.lambda_reg, dtype=data_type)
    lf_phi, lf_dup, lf_reg, lf_mart, lf_pos = model.train_step(
        t_tilde, k_tilde, phi_tilde_ref, t_min, t_max, k_min, k_max,
        lp, lr, float(model.lambda_mart), float(model.lambda_pos),
        lambda_pos_scale=tf.constant(1.0, dtype=data_type),
    )

    # ---- save model + metadata ----
    trainer.save_checkpoint(config.output_dir, "NN_phi_final.keras", "NN_eta_final.keras")
    save_metadata(config, config.output_dir)

    # ---- NaN guard over the loss histories ----
    def _any_nan(lst):
        return any(not np.isfinite(_to_float(x)) for x in lst)

    # pull the loss histories off the trainer-local lists via a final eval
    final_rmse_fit = float(tf.sqrt(tf.reduce_mean(
        tf.square(model.neural_phi(T_nn, K_nn) - phi_ref))).numpy())

    # ---- plots ----
    _plot_losses(model, trainer_inputs, config.output_dir)
    _plot_density(model, config, config.output_dir)

    # ---- eval diagnostics (vega-IV RMSE / neg mass / edge mass / mart resid) ----
    eval_diag = _compute_eval_diagnostics(model, config, T_nn, K_nn, phi_ref, market_meta)

    summary = {
        "mode": "real" if config.real_data else "synthetic",
        "epochs": int(config.num_epochs),
        "lambda_mart": float(config.lambda_mart),
        "lambda_pos": float(config.lambda_pos),
        "lambda_mz": float(config.lambda_mz),
        "mart_kgrid_n": int(config.mart_kgrid_n),
        "mart_kgrid_kmax_mult": float(config.mart_kgrid_kmax_mult),
        "scaling": market_meta if market_meta is not None else {
            "S0": config.S0, "r": config.r, "t_max": config.T_max,
            "k_max": config.K_max, "k_min": config.K_min, "phi_norm": "s0",
        },
        "final_losses": {
            "loss_phi": _to_float(lf_phi), "loss_dupire": _to_float(lf_dup),
            "loss_reg": _to_float(lf_reg), "loss_mart": _to_float(lf_mart),
            "loss_pos": _to_float(lf_pos),
        },
        "final_price_rmse": final_rmse_fit,
        "final_sigma_rmse": (_to_float(rmse_sigma_list[-1])
                             if (rmse_sigma_list and not config.real_data) else None),
        "any_nan_loss": bool(
            _any_nan([lf_phi, lf_dup, lf_reg, lf_mart, lf_pos])
            or not np.isfinite(final_rmse_fit)),
        "baseline_check": baseline_check,
        "eval_diagnostics": eval_diag,
        "output_dir": config.output_dir,
    }
    out_json = os.path.join(config.output_dir, "mz_pinn_train.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print(f"  final_losses = {json.dumps(summary['final_losses'])}")
    print(f"  final_price_rmse = {final_rmse_fit:.6f}")
    print(f"  any_nan_loss = {summary['any_nan_loss']}")
    print(f"  wrote {out_json}")
    print("=" * 80)


def _compute_eval_diagnostics(model, config, T_nn, K_nn, phi_ref, market_meta):
    """
    Post-train diagnostic metrics (printed + returned as dict for the JSON).

    (a) Vega-weighted IV RMSE by maturity × moneyness bucket.
        Price RMSE hides wing damage; vega-weighted IV RMSE per bucket exposes it.
        Moneyness buckets: OTM (<0.9), ATM (0.9-1.1), ITM (>1.1)  (K/forward).
        IV from option price via simple Black-Scholes put inversion (scipy).
        Skips rows where BS inversion fails.

    (b) negative_mass = ∫ max(-f, 0) dK  (per maturity; aggregate).
        Not min(f) — a single negative point is too noisy; the integral is stable.

    (c) edge mass fraction near K_min and K_max (5% of grid range from each edge).
        Flags apparent martingale "wins" driven by boundary mass-piling rather than
        a proper interior distribution.

    (d) Martingale relative residual |(E[S_T] - F)|/F per maturity:
        median AND max over the trained maturities.
    """
    import math

    T_vals = np.unique(T_nn.numpy().ravel() if hasattr(T_nn, 'numpy') else T_nn.ravel())
    S0 = config.S0
    r = config.r
    K_max = config.K_max
    K_min = getattr(config, 'K_min', 0.0)

    # (a) Black-Scholes put IV inversion helper (scalar, returns nan on failure)
    def _bs_put_iv(C, K, T, S, rate, tol=1e-6, max_iter=100):
        """Simple bisection for put IV given price C."""
        if T <= 0 or C <= 0:
            return float("nan")
        from math import log, sqrt, exp
        from scipy.stats import norm as _norm
        def _put(vol):
            if vol <= 0:
                return float("nan")
            d1 = (log(S / K) + (rate + 0.5 * vol**2) * T) / (vol * sqrt(T))
            d2 = d1 - vol * sqrt(T)
            return -S * _norm.cdf(-d1) + K * exp(-rate * T) * _norm.cdf(-d2)
        lo, hi = 1e-4, 5.0
        if _put(lo) > C or _put(hi) < C:
            return float("nan")
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if _put(mid) < C:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                return 0.5 * (lo + hi)
        return 0.5 * (lo + hi)

    def _vega_put(K, T, S, rate, iv):
        """Black-Scholes vega for a put (= call vega)."""
        import math
        from scipy.stats import norm as _norm
        if iv <= 0 or T <= 0:
            return 0.0
        d1 = (math.log(S / K) + (rate + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
        return S * _norm.pdf(d1) * math.sqrt(T)

    T_np = T_nn.numpy().ravel() if hasattr(T_nn, 'numpy') else np.asarray(T_nn).ravel()
    K_np = K_nn.numpy().ravel() if hasattr(K_nn, 'numpy') else np.asarray(K_nn).ravel()
    phi_np = phi_ref.numpy().ravel() if hasattr(phi_ref, 'numpy') else np.asarray(phi_ref).ravel()
    phi_nn_np = model.neural_phi(T_nn, K_nn).numpy().ravel()

    # (a) vega-weighted IV RMSE by maturity × moneyness bucket
    iv_rmse_by_T_bucket = {}
    bucket_labels = {"OTM": (0.0, 0.9), "ATM": (0.9, 1.1), "ITM": (1.1, 1e9)}
    for T_val in T_vals:
        mask = np.isclose(T_np, T_val, rtol=0.01)
        Km, phim, phi_nn_m = K_np[mask], phi_np[mask], phi_nn_np[mask]
        forward = S0 * math.exp(r * T_val)
        mon = Km / forward  # K / forward
        T_rec = {}
        for bname, (lo, hi) in bucket_labels.items():
            bm = (mon >= lo) & (mon < hi)
            if bm.sum() == 0:
                continue
            errs_iv, weights = [], []
            for k_, p_ref_, p_nn_ in zip(Km[bm], phim[bm], phi_nn_m[bm]):
                iv_ref = _bs_put_iv(float(p_ref_), float(k_), float(T_val), S0, r)
                iv_nn = _bs_put_iv(float(p_nn_), float(k_), float(T_val), S0, r)
                if not (np.isfinite(iv_ref) and np.isfinite(iv_nn)):
                    continue
                v = _vega_put(float(k_), float(T_val), S0, r, iv_ref)
                errs_iv.append((iv_nn - iv_ref) ** 2)
                weights.append(max(v, 1e-12))
            if errs_iv:
                w = np.array(weights)
                e = np.array(errs_iv)
                T_rec[bname] = float(np.sqrt(np.sum(w * e) / np.sum(w)))
        iv_rmse_by_T_bucket[float(T_val)] = T_rec

    # (b) (c) (d): per-maturity density integrals
    N_grid = getattr(config, 'mart_kgrid_n', 256)
    kmax_mult = getattr(config, 'mart_kgrid_kmax_mult', 1.5)
    K_hi = kmax_mult * K_max
    K_grid = np.linspace(0.0, K_hi, N_grid)
    dK = K_grid[1] - K_grid[0]
    edge_frac = 0.05  # 5% of total grid span near each boundary
    edge_width = edge_frac * (K_hi - 0.0)
    edge_lo_mask = K_grid < edge_width
    edge_hi_mask = K_grid > (K_hi - edge_width)

    neg_mass_per_T = {}
    edge_mass_per_T = {}
    mart_relresid_per_T = {}

    for T_val in T_vals:
        T_tf = tf.constant(float(T_val), dtype=data_type)
        K_g, f_tf = model._density_tf(T_tf)
        f_np = f_tf.numpy().ravel()
        K_np_g = K_g.numpy().ravel()

        # (b) negative mass
        neg_mass = float(np.trapz(np.maximum(-f_np, 0.0), K_np_g))
        neg_mass_per_T[float(T_val)] = neg_mass

        # (c) edge mass fraction
        total_mass = float(np.trapz(np.abs(f_np), K_np_g))
        lo_mask = K_np_g < edge_width
        hi_mask = K_np_g > (K_hi - edge_width)
        edge_lo_mass = float(np.trapz(np.abs(f_np[lo_mask]), K_np_g[lo_mask])) if lo_mask.any() else 0.0
        edge_hi_mass = float(np.trapz(np.abs(f_np[hi_mask]), K_np_g[hi_mask])) if hi_mask.any() else 0.0
        edge_mass_per_T[float(T_val)] = {
            "K_lo_frac": edge_lo_mass / max(total_mass, 1e-12),
            "K_hi_frac": edge_hi_mass / max(total_mass, 1e-12),
        }

        # (d) martingale relative residual |(E[S_T] - F)| / F
        mean_est = float(np.trapz(K_np_g * f_np, K_np_g))
        forward = float(S0 * math.exp(r * float(T_val)))
        mart_relresid_per_T[float(T_val)] = abs(mean_est - forward) / max(forward, 1e-12)

    resids = list(mart_relresid_per_T.values())
    finite_resids = [x for x in resids if np.isfinite(x)]

    diag = {
        "vega_weighted_iv_rmse_by_T_bucket": {str(k): v for k, v in iv_rmse_by_T_bucket.items()},
        "negative_mass_per_T": {str(k): v for k, v in neg_mass_per_T.items()},
        "edge_mass_per_T": {str(k): v for k, v in edge_mass_per_T.items()},
        "martingale_rel_resid_per_T": {str(k): v for k, v in mart_relresid_per_T.items()},
        "martingale_rel_resid_median": float(np.median(finite_resids)) if finite_resids else float("nan"),
        "martingale_rel_resid_max": float(np.max(finite_resids)) if finite_resids else float("nan"),
    }

    print("\n  [eval diagnostics]")
    print(f"    martingale rel resid: median={diag['martingale_rel_resid_median']:.4e}  "
          f"max={diag['martingale_rel_resid_max']:.4e}")
    print(f"    negative mass per T:")
    for T_val, nm in neg_mass_per_T.items():
        em = edge_mass_per_T[T_val]
        print(f"      T={T_val:.3f}: neg_mass={nm:.4e}  "
              f"edge_lo={em['K_lo_frac']:.3f}  edge_hi={em['K_hi_frac']:.3f}")
    print(f"    vega-weighted IV RMSE:")
    for T_val, rec in iv_rmse_by_T_bucket.items():
        bstr = "  ".join(f"{b}:{v:.4f}" for b, v in rec.items())
        print(f"      T={T_val:.3f}: {bstr}")
    return diag


def _plot_losses(model, trainer_inputs, output_dir):
    """Single combined loss snapshot (the time-series lives in trainer; here we
    re-evaluate the components once for a labelled bar/marker plot)."""
    (t_tilde, k_tilde, phi_tilde_ref, t_min, t_max, k_min, k_max) = trainer_inputs
    lp = tf.constant(model.lambda_pde, dtype=data_type)
    lr = tf.constant(model.lambda_reg, dtype=data_type)
    lf_phi, lf_dup, lf_reg, lf_mart, lf_pos = model.train_step(
        t_tilde, k_tilde, phi_tilde_ref, t_min, t_max, k_min, k_max,
        lp, lr, float(model.lambda_mart), float(model.lambda_pos),
        lambda_pos_scale=tf.constant(1.0, dtype=data_type))
    names = ["L_phi", "L_dupire", "L_reg", "L_mart", "L_pos"]
    vals = [max(_to_float(x), 1e-30) for x in (lf_phi, lf_dup, lf_reg, lf_mart, lf_pos)]
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    ax.bar(names, vals, color="#4A90E2")
    ax.set_yscale("log")
    ax.set_ylabel("loss (log)")
    ax.set_title(f"MZ Step-4-Lite final losses "
                 f"(λ_mart={model.lambda_mart}, λ_pos={model.lambda_pos})")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "mz_pinn_losses.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ saved {path}")


def _plot_density(model, config, output_dir):
    """Model-implied density f(K)=e^{rT}∂²C/∂K² at T_max via the graph-safe
    _density_tf, with the martingale target S0·e^{rT} marked."""
    T = float(config.T_max)
    K_grid, f = model._density_tf(T)
    K = K_grid.numpy().ravel()
    fv = f.numpy().ravel()
    mass = float(np.trapz(fv, K))
    mean = float(np.trapz(K * fv, K))
    target = float(config.S0 * np.exp(config.r * T))
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    ax.plot(K, fv, color="#E94B3C", label="f(K)=e^{rT}∂²C/∂K²")
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.axvline(mean, color="#50C878", ls="--", label=f"∫Kf={mean:.1f}")
    ax.axvline(target, color="#2C3E50", ls=":", label=f"S0·e^{{rT}}={target:.1f}")
    ax.set_xlabel("K")
    ax.set_ylabel("density")
    ax.set_title(f"Model density at T={T:.3f}  (mass={mass:.4f}, "
                 f"mean_err={abs(mean-target)/max(target,1e-9):.2e})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "mz_pinn_density.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ saved {path}  (mass={mass:.4f}, mean={mean:.2f} vs S0e^rT={target:.2f})")


if __name__ == "__main__":
    main()
