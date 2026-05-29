#!/usr/bin/env python3
"""
Re-generate Monte Carlo path data for a trained Dupire local-volatility model
under two source volatilities — and save both to disk.

  data MC      : paths under the GROUND-TRUTH sigma surface
                 (dupire_exact analytical formula, or constant sigma=1.0).
  repriced MC  : paths under the LEARNED sigma_NN (NN_eta wrapped as a callable
                 and dispatched through DataGenerator's 'custom' volatility
                 branch).

Both branches go through the same `DataGenerator.run_mc()` Euler-Maruyama loop
(dupire_pipeline.py:366) with `np.random.seed(42)` and `n_steps = N_t_train`,
which bypasses the simulator bug in `simulate_paths_with_nn_volatility`
(where `n_steps = int(n_paths)`).

Output is written as `<basename>.npz` with keys `S_matrix [N_t, M]` (float32)
and `t_all [N_t, 1]` (float32) — the exact schema the existing plot driver
`run_reproduce_user_plots.py --mc-source data --training-data PATH` expects.

Usage:
    python examples/run_regenerate_mc.py --model-dir PATH [...]

CLI:
    --model-dir PATH               (required)
    --mc-sources data repriced     (default: both)
    --volatility {auto,dupire_exact,constant_vol}   (default: auto)
    --m-data N                     (default: 1_000_000)
    --m-repriced N                 (default: 100_000)
    --dt FLOAT                     (default: 0.001 or metadata.dt_train)
    --n-t N                        (default: from metadata or 1000)
    --output-dir DIR               (default: <model-dir>/regenerated_mc)
"""

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig, VolatilityConfig
from dupire_pipeline import DataGenerator, load_trained_models


def resolve_model_dir(raw_path: str) -> str:
    if os.path.isabs(raw_path):
        return os.path.normpath(raw_path)
    normalized = raw_path.replace('\\', '/')
    candidates = [raw_path]
    if '/' not in normalized:
        candidates.append(os.path.join('models', 'runs', normalized))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    if '/' not in normalized:
        return os.path.normpath(os.path.join('models', 'runs', normalized))
    return os.path.normpath(raw_path)


def constant_vol(t, x):
    """Constant volatility sigma=1.0 (Black-Scholes baseline).

    Reconstructed locally because the original callable in the paper-run
    metadata.json was serialised as the string '<function constant_vol at ...>'
    and is unrecoverable.
    """
    if hasattr(x, 'shape'):
        return np.ones_like(x, dtype=np.float32)
    return np.float32(1.0)


def make_nn_sigma(nn_eta, S0: float, r: float, t_max: float, k_max: float):
    """Wrap NN_eta as a callable matching the VolatilityConfig.custom signature
    (t, x=S/S0 -> sigma)."""

    def nn_sigma(t, x):
        x_arr = np.asarray(x, dtype=np.float32)
        if np.isscalar(t):
            t_arr = np.full_like(x_arr, float(t), dtype=np.float32)
        else:
            t_arr = np.asarray(t, dtype=np.float32)
            if t_arr.shape != x_arr.shape:
                t_arr = np.broadcast_to(t_arr, x_arr.shape).astype(np.float32)
        K = x_arr * np.float32(S0)
        t_tilde = (t_arr / np.float32(t_max)).reshape(-1, 1)
        k_tilde = (np.exp(-np.float32(r) * t_arr) * K / np.float32(k_max)).reshape(-1, 1)
        feat = tf.concat([
            tf.constant(t_tilde, dtype=tf.float32),
            tf.constant(k_tilde, dtype=tf.float32),
        ], axis=1)
        eta_tilde = nn_eta(feat).numpy().reshape(x_arr.shape).astype(np.float32)
        return np.sqrt(2.0 * eta_tilde / np.float32(t_max))

    return nn_sigma


def detect_vol_kind(metadata: dict, model_dir: str, override: str) -> str:
    if override != 'auto':
        return override
    vol = metadata.get('volatility', {}) or metadata.get('config', {}).get('volatility_config', {})
    mtype = vol.get('model_type', '')
    if mtype == 'dupire_exact':
        return 'dupire_exact'
    if mtype == 'custom':
        # paper run convention: directory name carries "constant_vol"
        if 'constant_vol' in os.path.basename(os.path.normpath(model_dir)).lower():
            return 'constant_vol'
        # otherwise no way to recover the callable
        raise RuntimeError(
            "metadata says model_type='custom' but the original callable cannot "
            "be recovered. Pass --volatility {dupire_exact,constant_vol} explicitly."
        )
    raise RuntimeError(
        f"Unsupported volatility model_type={mtype!r} in metadata. "
        "Pass --volatility explicitly."
    )


def build_vol_config(kind: str) -> VolatilityConfig:
    if kind == 'dupire_exact':
        return VolatilityConfig.dupire_exact(sigma_base=0.3, t_shift=0.1, x_shift=0.1)
    if kind == 'constant_vol':
        return VolatilityConfig.custom(constant_vol)
    raise ValueError(f"Unknown vol kind: {kind!r}")


def _dupire_exact_sigma(sigma_base: float, t_shift: float, x_shift: float):
    """Inline dupire_exact callable, matching DataGenerator.exact_sigma's formula."""
    def f(t, x):
        x_arr = np.asarray(x, dtype=np.float32)
        if np.isscalar(t):
            t_val = np.float32(t)
            y = (t_val + np.float32(t_shift)) * np.sqrt(x_arr + np.float32(x_shift))
        else:
            t_arr = np.asarray(t, dtype=np.float32)
            if t_arr.shape != x_arr.shape:
                t_arr = np.broadcast_to(t_arr, x_arr.shape).astype(np.float32)
            y = (t_arr + np.float32(t_shift)) * np.sqrt(x_arr + np.float32(x_shift))
        return np.float32(sigma_base) + y * np.exp(-y)
    return f


def _resolve_callable(vol_cfg: VolatilityConfig):
    """Return a fully numpy-vectorised sigma(t, x) callable that respects vol_cfg.

    Bypasses DataGenerator.exact_sigma — that method routes through TF tensors
    and is slow when called repeatedly from a Python loop.
    """
    if vol_cfg.model_type == 'dupire_exact':
        return _dupire_exact_sigma(
            sigma_base=float(vol_cfg.sigma_base),
            t_shift=float(vol_cfg.t_shift),
            x_shift=float(vol_cfg.x_shift),
        )
    if vol_cfg.model_type == 'custom':
        return vol_cfg.custom_volatility_func
    raise ValueError(
        f"Unsupported model_type for inline loop: {vol_cfg.model_type!r}"
    )


def run_branch(branch: str, config: DupirePipelineConfig, vol_cfg: VolatilityConfig,
               M: int, N_t: int, dt: float, output_path: str) -> None:
    """Run a vectorised Euler-Maruyama loop and save to .npz.

    Bypasses DataGenerator.run_mc to avoid its per-path Python loop for the
    dW tensor (slow for M > ~10k). Uses a single `np.random.normal` call
    sized [N_t, M].
    """
    print('=' * 80)
    print(f"  Branch: {branch}")
    print(f"  vol     : model_type={vol_cfg.model_type}")
    print(f"  M paths : {M:,}")
    print(f"  N_t     : {N_t}   dt={dt}   -> simulated horizon {N_t*dt:.4f} y")
    print(f"  output  : {output_path}")
    print('=' * 80)

    sigma_func = _resolve_callable(vol_cfg)
    S0 = np.float32(config.S0)
    r = np.float32(config.r)
    dt_f = np.float32(dt)
    sqrt_dt = np.sqrt(dt_f)
    M = int(M)
    N_t = int(N_t)

    # Time grid (matches DataGenerator: linspace(0, N_t*dt, N_t))
    t_all = np.linspace(0.0, N_t * dt_f, N_t, dtype=np.float32).reshape(-1, 1)

    t0 = time.time()
    print(f"    Allocating dW [{N_t}, {M:,}] = {N_t*M*4/1e9:.2f} GB ...")
    np.random.seed(42)
    dW = np.random.normal(0.0, 1.0, size=(N_t, M)).astype(np.float32) * sqrt_dt
    print(f"    dW done in {time.time()-t0:.1f}s; starting Euler loop ...")

    S = np.empty((N_t, M), dtype=np.float32)
    S[0] = S0
    loop_t0 = time.time()
    for i in range(N_t - 1):
        t_now = t_all[i, 0]
        S_now = S[i]
        sigma_local = np.asarray(sigma_func(t_now, S_now / S0), dtype=np.float32)
        # Euler-Maruyama
        S[i + 1] = S_now + r * S_now * dt_f + sigma_local * S_now * dW[i]
        if (i + 1) % 200 == 0:
            print(f"    step {i+1}/{N_t-1}  "
                  f"S range [{S[i+1].min():.1f}, {S[i+1].max():.1f}]  "
                  f"({time.time()-loop_t0:.1f}s)")
    elapsed = time.time() - t0

    print(f"  -> S_matrix shape {S.shape}, t_all shape {t_all.shape}")
    print(f"  -> S at t_0  : [{S[0].min():.2f}, {S[0].max():.2f}]  "
          f"mean {S[0].mean():.2f}")
    print(f"  -> S at t_end: [{S[-1].min():.2f}, {S[-1].max():.2f}]  "
          f"mean {S[-1].mean():.2f}")
    print(f"  -> wall-clock: {elapsed:.1f}s")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(output_path, S_matrix=S, t_all=t_all)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  -> saved: {output_path}  ({size_mb:.1f} MB)")
    # Free the big arrays before the next branch.
    del S, dW


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-generate data MC and/or repriced MC for a trained model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mc-sources', nargs='+', default=['data', 'repriced'],
                        choices=['data', 'repriced'])
    parser.add_argument('--volatility', default='auto',
                        choices=['auto', 'dupire_exact', 'constant_vol'])
    parser.add_argument('--m-data', type=int, default=1_000_000)
    parser.add_argument('--m-repriced', type=int, default=100_000)
    parser.add_argument('--dt', type=float, default=None,
                        help="Override metadata dt_train (default: read from metadata).")
    parser.add_argument('--n-t', type=int, default=None,
                        help="Override metadata N_t_train (default: read from metadata).")
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    model_dir = resolve_model_dir(args.model_dir)
    if not os.path.exists(model_dir):
        print(f"ERROR: model_dir not found: {model_dir}", file=sys.stderr)
        return 1

    # Build a config and load metadata via the analysis_only preset.
    config = DupirePipelineConfig.analysis_only(model_dir)

    # Metadata-driven N_t and dt (with CLI overrides if provided)
    import json
    meta_path = os.path.join(model_dir, 'metadata.json')
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as fh:
            metadata = json.load(fh)
    cfg_meta = metadata.get('config', {}) or {}
    dt = args.dt if args.dt is not None else float(cfg_meta.get('dt_train', 0.001))
    N_t = args.n_t if args.n_t is not None else int(cfg_meta.get('N_t_train', 1000))

    vol_kind = detect_vol_kind(metadata, model_dir, args.volatility)
    print(f"\nResolved model_dir : {model_dir}")
    print(f"Volatility kind    : {vol_kind}")
    print(f"S0={config.S0}  r={config.r}  T_max={config.T_max}  "
          f"K_min={config.K_min}  K_max={config.K_max}")
    print(f"N_t={N_t}  dt={dt}  -> horizon {N_t*dt:.4f} y")

    out_dir = args.output_dir or os.path.join(model_dir, 'regenerated_mc')
    os.makedirs(out_dir, exist_ok=True)

    if 'data' in args.mc_sources:
        vol_cfg_data = build_vol_config(vol_kind)
        run_branch('data MC (ground-truth sigma)', config, vol_cfg_data,
                   M=args.m_data, N_t=N_t, dt=dt,
                   output_path=os.path.join(out_dir, 'data_mc.npz'))

    if 'repriced' in args.mc_sources:
        nn_phi, nn_eta, _meta = load_trained_models(model_dir)
        nn_sigma = make_nn_sigma(
            nn_eta,
            S0=float(config.S0),
            r=float(config.r),
            t_max=float(config.T_max),
            k_max=float(config.K_max),
        )
        vol_cfg_nn = VolatilityConfig.custom(nn_sigma)
        run_branch('repriced MC (NN-implied sigma)', config, vol_cfg_nn,
                   M=args.m_repriced, N_t=N_t, dt=dt,
                   output_path=os.path.join(out_dir, 'repriced_mc.npz'))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
