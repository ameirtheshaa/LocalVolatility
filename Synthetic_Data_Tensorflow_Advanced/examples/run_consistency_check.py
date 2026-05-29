#!/usr/bin/env python3
"""
Dupire NN consistency check — three-way mean comparison.

For each maturity T, computes:

    M_1 = e^(rT) * integral_0^inf K * d^2 C_NN / dK^2 dK     (price moment via IBP)
    M_2 = e^(rT) * C_NN(K=0, T)                              (forward via call at zero)
    M_3 = E[S_T] from MC with sigma_NN                       (vol-network forward)

All three must equal the theoretical forward S_0 * e^(rT).
  Delta_M12 = |M_1 - M_2| / M_2 tests the IBP identity (numerical d^2/dK^2 health).
  Delta_M13 = |M_1 - M_3| / M_3 tests cross-network Dupire consistency.

Defaults to the synthetic-volatility baseline at models/example_pretrained
(shipped NN_phi/NN_eta trained on the default dupire_exact synthetic data).
Re-usable: point --model-dir at any directory holding NN_phi*.keras,
NN_eta*.keras, and metadata.json.

Scaling is delegated to the existing primitives in dupire_pipeline.py:
  - phi_tilde_from_nn (transformed mode: phi_tilde = 1 - exp(-NN_phi))
  - prepare_data_for_nn (t_tilde = T/T_max, k_tilde = e^(-rT)*K/K_max)
  - _raw_model_density (chain-rule Jacobian d k_tilde / dK = e^(-rT) / K_max)
  - simulate_paths_with_nn_volatility (Euler-Maruyama; np.random.seed(42))

Usage:
    python examples/run_consistency_check.py
    python examples/run_consistency_check.py --model-dir models/runs/k0_bc_retrain
    python examples/run_consistency_check.py --maturities 0.25 0.5 1.0 --n-paths 50000
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig
from dupire_pipeline import PDFAnalyzer, load_trained_models


def resolve_model_dir(raw_path: str) -> str:
    """Accept abs paths, repo-relative paths, or a bare run name under models/runs/."""
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


def build_plot_K_grid(analyzer: PDFAnalyzer, mc_samples: np.ndarray) -> np.ndarray:
    """Reproduce the K-grid that create_enhanced_pdf_analysis uses (150 points
    on [max(k_min, q1), min(k_max, q99)]) so K_tail in compute_mean_diagnostics
    matches what run_ibp_comparison_plots.py produced for the existing JSONs."""
    if mc_samples.size > 0:
        domain_mask = (mc_samples >= analyzer.k_min) & (mc_samples <= analyzer.k_max)
        trimmed = mc_samples[domain_mask]
        mc_used = trimmed if trimmed.size >= 100 else mc_samples
    else:
        mc_used = mc_samples
    if mc_used.size >= 2:
        q1, q99 = np.percentile(mc_used, [1, 99])
        K_min_plot = max(analyzer.k_min, q1)
        K_max_plot = min(analyzer.k_max, q99)
    else:
        K_min_plot, K_max_plot = analyzer.k_min, analyzer.k_max
    if (not np.isfinite(K_min_plot) or not np.isfinite(K_max_plot)
            or K_max_plot <= K_min_plot):
        K_min_plot, K_max_plot = analyzer.k_min, analyzer.k_max
    return np.linspace(K_min_plot, K_max_plot, 150, dtype=np.float64)


def classify(diag: dict) -> str:
    """Per-maturity interpretation line per the spec's rule set."""
    M1 = diag['mean_nn_density_wide']
    M2 = diag['spot_call']
    M3 = diag['mean_mc']
    ref = diag['theoretical_ref']
    if not np.isfinite(M3) or M3 == 0:
        return 'MC mean unavailable — cannot evaluate Delta_M13.'
    notes = []
    if all(abs(v - ref) / ref < 0.02 for v in (M1, M2, M3)):
        notes.append('all three within 2% of forward → price and vol Dupire-consistent')
    dM12 = abs(M1 - M2) / abs(M2) if M2 != 0 else float('inf')
    if dM12 > 0.01:
        notes.append('Delta_M12 > 1% → numerical d^2 C/dK^2 may be ill-conditioned')
    if abs(M1 - M3) / abs(M3) > 0.02:
        notes.append('Delta_M13 > 2% → soft Dupire constraint is too weak')
    ratio = M1 / M3
    if ratio < 0.99:
        notes.append('M_1 << M_3 → density left-shifted vs MC; vol network implies higher mean')
    elif ratio > 1.01:
        notes.append('M_1 >> M_3 → density right-shifted vs MC; vol network implies lower mean')
    return '; '.join(notes) if notes else '(no anomaly above thresholds)'


def print_table(T: float, diag: dict) -> None:
    M1 = diag['mean_nn_density_wide']
    M2 = diag['spot_call']
    M3 = diag['mean_mc']
    ref = diag['theoretical_ref']
    pct = lambda v: (v - ref) / ref * 100.0 if ref != 0 else float('nan')
    dM12 = abs(M1 - M2) / abs(M2) * 100.0 if M2 != 0 else float('nan')
    if np.isfinite(M3) and M3 != 0:
        dM13 = abs(M1 - M3) / abs(M3) * 100.0
    else:
        dM13 = float('nan')
    print()
    print(f'  T = {T:.2f}   (reference forward  S_0*e^(rT) = {ref:.4f})')
    print(f'  Value                       | Estimate     | % Error vs M_ref')
    print(f'  ----------------------------+--------------+-----------------')
    print(f'  M_1 (Price moment, IBP)     | {M1:12.4f} | {pct(M1):+8.3f} %')
    print(f'  M_2 (Forward via C_NN(0))   | {M2:12.4f} | {pct(M2):+8.3f} %')
    print(f'  M_3 (MC mean from sigma_NN) | {M3:12.4f} | {pct(M3):+8.3f} %')
    print(f'  Reference (S_0*e^(rT))      | {ref:12.4f} |   0.000 %')
    print(f'  Delta_M12 = |M1-M2| / M2    |              | {dM12:8.3f} %   '
          '(IBP identity check)')
    print(f'  Delta_M13 = |M1-M3| / M3    |              | {dM13:8.3f} %   '
          '(price ↔ vol Dupire consistency)')
    print(f"  [tail-mass check: K*f at K_tail={diag['K_tail']:.0f} = "
          f"{diag['tail_mass_check']:.2e}  (good if << M_ref)]")
    print(f'  Interpretation: {classify(diag)}')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Three-way consistency check for trained Dupire NN models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Default: synthetic-volatility baseline (models/example_pretrained)
  python examples/run_consistency_check.py

  # Re-usable for any model directory with NN_phi*.keras, NN_eta*.keras, metadata.json
  python examples/run_consistency_check.py --model-dir models/runs/k0_bc_retrain

  # Custom maturities and MC budget
  python examples/run_consistency_check.py --maturities 0.25 0.5 1.0 --n-paths 50000
""")
    parser.add_argument('--model-dir', default='models/example_pretrained',
                        help='Directory with NN_phi/NN_eta keras files and metadata.json '
                             '(default: models/example_pretrained, the synthetic baseline).')
    parser.add_argument('--maturities', type=float, nargs='+', default=[0.5, 1.0, 1.5],
                        help='Maturities to evaluate (default: 0.5 1.0 1.5).')
    parser.add_argument('--n-paths', type=int, default=25000,
                        help='Number of MC paths for M_3 (default: 25000).')
    parser.add_argument('--output-dir', default=None,
                        help='Where to write consistency_check.json '
                             '(default: <model-dir>/consistency_check).')
    parser.add_argument('--phi-mapping', choices=['transformed', 'legacy'], default='transformed',
                        help="phi_tilde mapping; 'transformed' = 1-exp(-NN_phi) matches "
                             "training (default and required for valid IBP).")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    model_dir = resolve_model_dir(args.model_dir)
    if not os.path.exists(model_dir):
        print(f'ERROR: model directory not found: {model_dir}', file=sys.stderr)
        return 1

    T_values = sorted(args.maturities)
    out_dir = args.output_dir or os.path.join(model_dir, 'consistency_check')
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 80)
    print('DUPIRE NN CONSISTENCY CHECK')
    print('=' * 80)
    print(f'Model:        {model_dir}')
    print(f'phi mapping:  {args.phi_mapping}')
    print(f'Maturities:   {T_values}')
    print(f'MC paths:     {args.n_paths:,}  (seed = 42, fixed inside simulator)')
    print(f'Output dir:   {out_dir}')
    print('=' * 80)

    config = DupirePipelineConfig.analysis_only(model_dir)
    config.analysis_config.n_paths_analysis = args.n_paths

    nn_phi, nn_eta, metadata = load_trained_models(model_dir)
    analyzer = PDFAnalyzer(nn_phi, nn_eta, config, metadata,
                           phi_mapping=args.phi_mapping)

    mc_data = analyzer.simulate_paths_with_nn_volatility(T_values, verbose=True)

    per_T = {}
    for T in T_values:
        K_grid = build_plot_K_grid(analyzer, mc_data[T])
        diag = analyzer.compute_mean_diagnostics(T, K_grid, mc_data[T])
        print_table(T, diag)
        ref = diag['theoretical_ref']
        M1 = diag['mean_nn_density_wide']
        M2 = diag['spot_call']
        M3 = diag['mean_mc']
        per_T[f'{T:.4f}'] = {
            'M_1':            M1,
            'M_2':            M2,
            'M_3':            M3,
            'M_ref':          ref,
            'pct_err_M1':     (M1 - ref) / ref * 100.0 if ref != 0 else float('nan'),
            'pct_err_M2':     (M2 - ref) / ref * 100.0 if ref != 0 else float('nan'),
            'pct_err_M3':     (M3 - ref) / ref * 100.0 if ref != 0 else float('nan'),
            'delta_M12_pct':  abs(M1 - M2) / abs(M2) * 100.0 if M2 != 0 else float('nan'),
            'delta_M13_pct':  (abs(M1 - M3) / abs(M3) * 100.0
                               if (np.isfinite(M3) and M3 != 0) else float('nan')),
            'K_tail':              diag['K_tail'],
            'tail_mass_check':     diag['tail_mass_check'],
            'mean_nn_density_plot': diag['mean_nn_density_plot'],
            'interpretation':      classify(diag),
        }

    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    header = (f"  {'T':>5} | {'M_ref':>9} | {'M_1':>9} | {'%err':>7} | "
              f"{'M_2':>9} | {'%err':>7} | {'M_3':>9} | {'%err':>7} | "
              f"{'dM12 %':>7} | {'dM13 %':>7}")
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for T in T_values:
        r = per_T[f'{T:.4f}']
        print(f"  {T:>5.2f} | {r['M_ref']:>9.2f} | "
              f"{r['M_1']:>9.2f} | {r['pct_err_M1']:>+7.2f} | "
              f"{r['M_2']:>9.2f} | {r['pct_err_M2']:>+7.2f} | "
              f"{r['M_3']:>9.2f} | {r['pct_err_M3']:>+7.2f} | "
              f"{r['delta_M12_pct']:>7.3f} | {r['delta_M13_pct']:>7.3f}")

    payload = {
        'model_dir':   model_dir,
        'phi_mapping': args.phi_mapping,
        'n_paths':     args.n_paths,
        'mc_seed':     42,
        'S0':          float(config.S0),
        'r':           float(config.r),
        'maturities':  T_values,
        'per_maturity': per_T,
    }
    out_json = os.path.join(out_dir, 'consistency_check.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print()
    print(f'Saved: {out_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
