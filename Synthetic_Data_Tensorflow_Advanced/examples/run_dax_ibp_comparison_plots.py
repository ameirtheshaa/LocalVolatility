#!/usr/bin/env python3
"""
Generate DAX IBP comparison PDF analysis plots (legacy vs transformed phi):

  plots/ibp_comparison_dax/{7aug,8aug,9aug}/
    pre_correction/  - legacy phi mapping, legacy DAX keras weights
    pretrained/      - transformed phi, same weights + same MC
    retrained/       - placeholder only (not run yet)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt

_examples_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.dirname(_examples_dir)
sys.path.insert(0, _pkg_root)
sys.path.insert(0, _examples_dir)

from config import DupirePipelineConfig, PlotConfig
from dax_analysis_common import DAX_RUNS, RISK_FREE, _read_market_grid
from dupire_pipeline import PDFAnalyzer

import tensorflow as tf

DAY_LABELS = {
    '7aug': 'DAX 7 Aug 2001',
    '8aug': 'DAX 8 Aug 2001',
    '9aug': 'DAX 9 Aug 2001',
}


def save_figure(fig, out_dir: str, dpi: int = 450) -> None:
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, 'pdf_analysis.png')
    pdf_path = os.path.join(out_dir, 'pdf_analysis.pdf')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def run_plot_set(analyzer: PDFAnalyzer, mc_data: dict, T_values: list,
                 out_dir: str, figure_label: str, dpi: int) -> dict:
    print(f"\n{'='*80}")
    print(f"Plot set: {figure_label}")
    print(f"Output: {out_dir}")
    print(f"phi_mapping: {analyzer.phi_mapping}")
    print(f"{'='*80}")

    fig, results = analyzer.create_enhanced_pdf_analysis(
        mc_data, T_values=T_values, figure_label=figure_label)
    save_figure(fig, out_dir, dpi=dpi)

    summary_path = os.path.join(out_dir, 'diagnostics.json')
    serializable = {
        float(T): {k: (float(v) if isinstance(v, (int, float)) else v)
                   for k, v in diag.items()}
        for T, diag in results.items()
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {summary_path}")
    return results


def format_table(day_label: str, name: str, results: dict) -> str:
    lines = [f"### {day_label} — {name}", "",
             "| T | ref | (i) wide | (ii) C_NN(0) | (iii) MC |",
             "|---:|---:|---:|---:|---:|"]
    for T in sorted(results.keys()):
        d = results[T]
        ref = d.get('theoretical_ref', float('nan'))
        lines.append(
            f"| {T:.3f} | {ref:.2f} | {d['mean_nn_density_wide']:.2f} | "
            f"{d['spot_call']:.2f} | {d['mean_mc']:.2f} |")
    return '\n'.join(lines) + '\n'


def write_retrained_stub(day_dir: str) -> None:
    stub_dir = os.path.join(day_dir, 'retrained')
    os.makedirs(stub_dir, exist_ok=True)
    path = os.path.join(stub_dir, 'README.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(
            "# Retrained (deferred)\n\n"
            "DAX K=0 boundary retrain has not been run yet. "
            "This folder is reserved for a future `k0_bc_retrain` analogue.\n"
        )
    print(f"  Wrote stub: {path}")


def write_readme(base_dir: str, tables: Dict[str, Dict[str, dict]]) -> None:
    readme = """# DAX IBP three-way comparison plots

Three-panel PDF figures (K-space / log-space / Gaussian) with Nicolas IBP mean
diagnostics in the yellow box on Panel 3.

Per day (`7aug`, `8aug`, `9aug`):

| Folder | Model | phi mapping |
|--------|-------|-------------|
| `pre_correction/` | legacy DAX keras | legacy (raw NN_phi as phi_tilde) |
| `pretrained/` | same weights, same MC | transformed: 1 - exp(-NN_phi) |
| `retrained/` | — | not run yet (placeholder) |

MC: 25,000 paths with sigma_NN(t,S). Maturities from market CSV quantiles (0.25, 0.5, 0.75).

## Diagnostic tables

"""
    for day_key in sorted(tables.keys()):
        day_tables = tables[day_key]
        day_label = DAY_LABELS.get(day_key, day_key)
        if 'pre_correction' in day_tables:
            readme += format_table(day_label, 'Pre-correction (legacy phi)',
                                   day_tables['pre_correction'])
        if 'pretrained' in day_tables:
            readme += format_table(day_label, 'Pretrained (corrected phi)',
                                   day_tables['pretrained'])

    readme += """## Interpretation

1. (i) and (ii) agree closely in every run — IBP identity holds in code.
2. Pre-correction inflates (i)/(ii) via the legacy phi bug.
3. Pretrained (corrected): residual gap between (i)/(ii) and forward localizes NN_phi at K=0.
4. `retrained/` is deferred until a DAX K=0 boundary retrain exists.

Regenerate: `python3 examples/run_dax_ibp_comparison_plots.py`
"""
    path = os.path.join(base_dir, 'README.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"\nWrote {path}")


def _resolve_paths(day_key: str) -> tuple:
    spec = DAX_RUNS[day_key]
    model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), spec['model_dir']))
    csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), spec['csv']))
    return spec, model_dir, csv_path


def _load_dax_keras(model_dir: str) -> tuple:
    """Load legacy DAX keras weights (bypass empty metadata.json placeholders)."""
    phi_path = os.path.join(model_dir, 'NN_phi.keras')
    eta_path = os.path.join(model_dir, 'NN_eta.keras')
    print(f"\nLoading DAX models from: {model_dir}")
    print(f"  NN_phi: {phi_path}")
    nn_phi = tf.keras.models.load_model(phi_path)
    print(f"    loaded {nn_phi.count_params():,} parameters")
    print(f"  NN_eta: {eta_path}")
    nn_eta = tf.keras.models.load_model(eta_path)
    print(f"    loaded {nn_eta.count_params():,} parameters")
    return nn_phi, nn_eta


def _check_models(model_dir: str) -> None:
    for fname in ('NN_phi.keras', 'NN_eta.keras'):
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(
                f"DAX models in {model_dir} are empty or missing ({fname}). "
                "Make the DAX_Tensorflow run folders available offline, then retry."
            )


def _build_dax_analyzer_config(
    spec: dict,
    t_min: float,
    t_max: float,
    k_min: float,
    k_max: float,
    n_paths: int,
) -> tuple:
    config = DupirePipelineConfig()
    config.S0 = spec['S0']
    config.r = RISK_FREE
    config.T_min = t_min
    config.T_max = t_max
    config.K_min = k_min
    config.K_max = k_max
    config.analysis_config.n_paths_analysis = n_paths
    config.analysis_config.reuse_training_mc = False
    config.analysis_config.run_mc_with_nn_volatility = True
    config.plot_config = PlotConfig(dpi=450)

    metadata = {
        'scaling': {
            'S0': spec['S0'],
            'r': RISK_FREE,
            't_max': t_max,
            'k_min': k_min,
            'k_max': k_max,
        },
        'source': 'dax_ibp_comparison',
        'date_label': spec['date_label'],
    }
    return config, metadata


def run_day(day_key: str, base_dir: str, n_paths: int) -> Dict[str, dict]:
    spec, model_dir, csv_path = _resolve_paths(day_key)
    _check_models(model_dir)

    t_min, t_max, k_min, k_max, maturities = _read_market_grid(csv_path)
    T_values = sorted(maturities)
    day_label = DAY_LABELS[day_key]
    day_dir = os.path.join(base_dir, day_key)

    print(f"\n{'#'*80}")
    print(f"DAY: {day_label}  S0={spec['S0']:.2f}  maturities={T_values}")
    print(f"Model: {model_dir}")
    print(f"{'#'*80}")

    config, metadata = _build_dax_analyzer_config(
        spec, t_min, t_max, k_min, k_max, n_paths)

    nn_phi, nn_eta = _load_dax_keras(model_dir)

    analyzer_mc = PDFAnalyzer(nn_phi, nn_eta, config, metadata, phi_mapping='transformed')
    analyzer_mc.training_T_min = t_min
    analyzer_mc.training_T_max = t_max
    mc_data = analyzer_mc.simulate_paths_with_nn_volatility(T_values, verbose=True)

    analyzer_legacy = PDFAnalyzer(
        nn_phi, nn_eta, config, metadata,
        phi_mapping='legacy',
        figure_label=f'{day_label} — Pre-correction (legacy phi_tilde mapping)')
    analyzer_legacy.training_T_min = t_min
    analyzer_legacy.training_T_max = t_max
    res_pre = run_plot_set(
        analyzer_legacy, mc_data, T_values,
        os.path.join(day_dir, 'pre_correction'),
        f'{day_label} — Pre-correction (legacy phi_tilde mapping)',
        config.plot_config.dpi)

    analyzer_pretrained = PDFAnalyzer(
        nn_phi, nn_eta, config, metadata,
        phi_mapping='transformed',
        figure_label=f'{day_label} — Pretrained (corrected phi_tilde mapping)')
    analyzer_pretrained.training_T_min = t_min
    analyzer_pretrained.training_T_max = t_max
    res_pt = run_plot_set(
        analyzer_pretrained, mc_data, T_values,
        os.path.join(day_dir, 'pretrained'),
        f'{day_label} — Pretrained (corrected phi_tilde mapping)',
        config.plot_config.dpi)

    write_retrained_stub(day_dir)

    return {'pre_correction': res_pre, 'pretrained': res_pt}


def main():
    parser = argparse.ArgumentParser(
        description='Generate DAX IBP comparison plots (pre_correction + pretrained)')
    parser.add_argument('--output-dir', default='plots/ibp_comparison_dax')
    parser.add_argument('--n-paths', type=int, default=25000)
    parser.add_argument('--days', nargs='+', default=['7aug', '8aug', '9aug'],
                        choices=sorted(DAX_RUNS.keys()))
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    print('=' * 80)
    print('DAX IBP COMPARISON PLOTS')
    print('=' * 80)
    print(f'Output base: {args.output_dir}')
    print(f'Days: {args.days}')
    print(f'MC paths: {args.n_paths:,}')

    all_tables: Dict[str, Dict[str, dict]] = {}
    for day_key in args.days:
        all_tables[day_key] = run_day(day_key, args.output_dir, args.n_paths)

    write_readme(args.output_dir, all_tables)

    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)


if __name__ == '__main__':
    main()
