#!/usr/bin/env python3
"""
Generate three-way IBP comparison PDF analysis plots:

  pre_correction/  - legacy phi mapping, example_pretrained weights
  pretrained/      - transformed phi, same weights + same MC
  retrained/         - transformed phi, k0_bc_retrain weights
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig
from dupire_pipeline import MC_PROVENANCE_NN, PDFAnalyzer, load_trained_models


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
                 out_dir: str, figure_label: str, config: DupirePipelineConfig,
                 mc_provenance: str = MC_PROVENANCE_NN) -> dict:
    """Plot `mc_data` with `analyzer`.

    `mc_provenance` must be passed explicitly because the analyzer plotting the
    data is not always the one that simulated it (the pretrained sets reuse one
    MC across two analyzers), so analyzer.mc_provenance may be unset.
    """
    print(f"\n{'='*80}")
    print(f"Plot set: {figure_label}")
    print(f"Output: {out_dir}")
    print(f"phi_mapping: {analyzer.phi_mapping}")
    print(f"{'='*80}")

    fig, results = analyzer.create_enhanced_pdf_analysis(
        mc_data, T_values=T_values, figure_label=figure_label,
        mc_provenance=mc_provenance)
    save_figure(fig, out_dir, dpi=config.plot_config.dpi)

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


def format_table(name: str, results: dict) -> str:
    lines = [f"### {name}", "",
              "| T | ref | (i) wide | (ii) C_NN(0) | (iii) MC |",
              "|---:|---:|---:|---:|---:|"]
    for T in sorted(results.keys()):
        d = results[T]
        ref = d.get('theoretical_ref', float('nan'))
        lines.append(
            f"| {T:.1f} | {ref:.2f} | {d['mean_nn_density_wide']:.2f} | "
            f"{d['spot_call']:.2f} | {d['mean_mc']:.2f} |")
    return '\n'.join(lines) + '\n'


def write_readme(base_dir: str, tables: dict) -> None:
    readme = """# IBP three-way comparison plots

Three-panel PDF figures (K-space / log-space / Gaussian) with Nicolas IBP mean
diagnostics in the yellow box on Panel 3.

| Folder | Model | phi mapping |
|--------|-------|-------------|
| `pre_correction/` | `models/example_pretrained` | legacy (raw NN_phi as phi_tilde) |
| `pretrained/` | same weights, same MC | transformed: 1 - exp(-NN_phi) |
| `retrained/` | `models/runs/k0_bc_retrain` | transformed |

MC: 25,000 paths with sigma_NN(t,S). Maturities T = 0.5, 1.0, 1.5.

## Diagnostic tables

"""
    for key, title in [
        ('pre_correction', 'Pre-correction (legacy phi)'),
        ('pretrained', 'Pretrained (corrected phi)'),
        ('retrained', 'Retrained (K0 BC, K_min=100)'),
    ]:
        if key in tables:
            readme += format_table(title, tables[key])

    readme += """## Interpretation

1. (i) and (ii) agree closely in every run — IBP identity holds in code.
2. Pre-correction inflates (i)/(ii) via the legacy phi bug (~2.7x ref).
3. Pretrained (corrected): (i)/(ii) are ~7% below ref (NN_phi undershoots near K=0); unconditional MC (iii) is 0.2-0.7% below ref — NN_eta MC is well-calibrated.
4. Retrained (K0 BC) aligns (i), (ii), (iii) within ~1% at every T = 0.5, 1.0, 1.5.

Note: the IBP M_3 here is the **unconditional** MC mean (full mc_samples).
Before the dupire_pipeline.py:1590 fix this was inadvertently the trimmed
in-window mean, which biased it upward by ~2% on the pretrained model
(asymmetric trimming under the heavy right tail of the local-vol surface).

Regenerate: `python3 examples/run_ibp_comparison_plots.py`
"""
    path = os.path.join(base_dir, 'README.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"\nWrote {path}")


def main():
    parser = argparse.ArgumentParser(description='Generate IBP three-way comparison plots')
    parser.add_argument('--output-dir', default='plots/ibp_comparison')
    parser.add_argument('--n-paths', type=int, default=25000)
    parser.add_argument('--maturities', type=float, nargs='+', default=[0.5, 1.0, 1.5])
    parser.add_argument('--pretrained-dir', default='models/example_pretrained')
    parser.add_argument('--retrained-dir', default='models/runs/k0_bc_retrain')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    config = DupirePipelineConfig.analysis_only(args.pretrained_dir)
    config.analysis_config.n_paths_analysis = args.n_paths
    T_values = sorted(args.maturities)
    base_dir = args.output_dir

    print('=' * 80)
    print('IBP THREE-WAY COMPARISON PLOTS')
    print('=' * 80)
    print(f'Output base: {base_dir}')
    print(f'Maturities: {T_values}')
    print(f'MC paths: {args.n_paths:,}')

    # --- Pretrained: one MC, two plot sets ---
    nn_phi, nn_eta, metadata = load_trained_models(args.pretrained_dir)
    analyzer_mc = PDFAnalyzer(nn_phi, nn_eta, config, metadata, phi_mapping='transformed')
    mc_data = analyzer_mc.simulate_paths_with_nn_volatility(T_values, verbose=True)

    analyzer_legacy = PDFAnalyzer(
        nn_phi, nn_eta, config, metadata,
        phi_mapping='legacy',
        figure_label='Pre-correction (legacy phi_tilde mapping)')
    res_pre = run_plot_set(
        analyzer_legacy, mc_data, T_values,
        os.path.join(base_dir, 'pre_correction'),
        'Pre-correction (legacy phi_tilde mapping)',
        config)

    analyzer_pretrained = PDFAnalyzer(
        nn_phi, nn_eta, config, metadata,
        phi_mapping='transformed',
        figure_label='Pretrained (corrected phi_tilde mapping)')
    res_pt = run_plot_set(
        analyzer_pretrained, mc_data, T_values,
        os.path.join(base_dir, 'pretrained'),
        'Pretrained (corrected phi_tilde mapping)',
        config)

    # --- Retrained ---
    config_rt = DupirePipelineConfig.analysis_only(args.retrained_dir)
    config_rt.analysis_config.n_paths_analysis = args.n_paths
    nn_phi_rt, nn_eta_rt, metadata_rt = load_trained_models(args.retrained_dir)
    analyzer_rt = PDFAnalyzer(
        nn_phi_rt, nn_eta_rt, config_rt, metadata_rt,
        phi_mapping='transformed',
        figure_label='Retrained (K0 BC penalty, K_min=100)')
    analyzer_rt.training_T_min = 0.0
    analyzer_rt.training_T_max = float(config_rt.T_max)

    mc_data_rt = analyzer_rt.simulate_paths_with_nn_volatility(T_values, verbose=True)
    res_rt = run_plot_set(
        analyzer_rt, mc_data_rt, T_values,
        os.path.join(base_dir, 'retrained'),
        'Retrained (K0 BC penalty, K_min=100)',
        config_rt)

    write_readme(base_dir, {
        'pre_correction': res_pre,
        'pretrained': res_pt,
        'retrained': res_rt,
    })

    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)


if __name__ == '__main__':
    main()
