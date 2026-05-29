#!/usr/bin/env python3
"""
Reproduce the 4-maturity PDF-analysis figures the user produced earlier
(constant-σ and dupire_exact paper models, T=[0.25, 0.5, 0.75, 1.0],
with green-dashed "Analytical" KDE-of-MC overlays on all three panels).

Two MC sources:
  --mc-source repriced  (default)  Generate fresh paths under
                                   dS = r*S*dt + sigma_NN(t,S)*S*dW.
  --mc-source data                 Read paths from training_data.npz
                                   (paths generated during data
                                   generation with the ground-truth σ).

Usage:
    python examples/run_reproduce_user_plots.py \\
        --model-dir <PATH> \\
        --mc-source {repriced,data} \\
        [--training-data PATH] \\
        [--maturities 0.25 0.5 0.75 1.0] \\
        [--n-paths 25000] \\
        [--output-dir DIR]

The script writes pdf_analysis.{png,pdf} and diagnostics.json into the
output directory. It does not modify the trained model or its training
data; only reads them.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig
from dupire_pipeline import PDFAnalyzer, load_trained_models


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
    if '/' not in normalized:
        return os.path.normpath(os.path.join('models', 'runs', normalized))
    return os.path.normpath(raw_path)


def _check_data_file(path: str) -> None:
    """Fail with a clear message if the training-data file is missing or
    is a 0-byte Dropbox placeholder."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"training_data.npz not found at:\n  {path}\n"
            "Use --training-data PATH to point at one explicitly, or "
            "pick a model directory that has it."
        )
    size = os.path.getsize(path)
    if size < 1024:
        raise FileNotFoundError(
            f"training_data.npz is {size} bytes (Dropbox placeholder?):\n"
            f"  {path}\n"
            "Force-sync the file in Finder (right-click → 'Always keep "
            "on this device') and retry."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce 4-maturity PDF-analysis figures with KDE-of-MC "
            "'Analytical' overlays."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model-dir', required=True,
                        help="Directory containing NN_phi*.keras + NN_eta*.keras "
                             "(and metadata.json if available).")
    parser.add_argument('--mc-source', choices=['repriced', 'data'], default='repriced',
                        help="repriced (default) = simulate_paths_with_nn_volatility; "
                             "data = extract_mc_samples_from_training_data.")
    parser.add_argument('--training-data', default=None,
                        help="Optional explicit path to training_data.npz. "
                             "Defaults to <model-dir>/training_data.npz.")
    parser.add_argument('--maturities', type=float, nargs='+',
                        default=[0.25, 0.5, 0.75, 1.0],
                        help='Maturities (default: 0.25 0.5 0.75 1.0).')
    parser.add_argument('--n-paths', type=int, default=25000,
                        help='MC paths for repriced source (default 25000).')
    parser.add_argument('--output-dir', default=None,
                        help="Where to save outputs (default: "
                             "<model-dir>/reproduce_user_plots).")
    parser.add_argument('--phi-mapping', choices=['transformed', 'legacy'],
                        default='transformed',
                        help="phi_tilde mapping (default 'transformed' = "
                             "matches training).")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    model_dir = resolve_model_dir(args.model_dir)
    if not os.path.exists(model_dir):
        print(f"ERROR: model directory not found: {model_dir}", file=sys.stderr)
        return 1

    T_values = sorted(args.maturities)
    out_dir = args.output_dir or os.path.join(model_dir, 'reproduce_user_plots')
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 80)
    print('REPRODUCE USER PDF-ANALYSIS FIGURES')
    print('=' * 80)
    print(f'Model        : {model_dir}')
    print(f'MC source    : {args.mc_source}')
    print(f'phi_mapping  : {args.phi_mapping}')
    print(f'Maturities   : {T_values}')
    if args.mc_source == 'repriced':
        print(f'MC paths     : {args.n_paths:,}  (seed=42)')
    print(f'Output dir   : {out_dir}')
    print('=' * 80)

    config = DupirePipelineConfig.analysis_only(model_dir)
    config.analysis_config.n_paths_analysis = args.n_paths
    config.analysis_config.T_analysis = T_values

    nn_phi, nn_eta, metadata = load_trained_models(model_dir)
    analyzer = PDFAnalyzer(nn_phi, nn_eta, config, metadata,
                           phi_mapping=args.phi_mapping)

    if args.mc_source == 'repriced':
        mc_data = analyzer.simulate_paths_with_nn_volatility(T_values, verbose=True)
    else:
        data_path = args.training_data or os.path.join(model_dir, 'training_data.npz')
        _check_data_file(data_path)
        print(f"\nReading training-data MC from: {data_path}")
        mc_data = analyzer.extract_mc_samples_from_training_data(
            data_path, T_values, verbose=True)

    print()
    fig, results = analyzer.create_enhanced_pdf_analysis(
        mc_data, T_values=T_values,
        figure_label=f'MC source: {args.mc_source}')

    dpi = getattr(config.plot_config, 'dpi', 300)
    png_path = os.path.join(out_dir, 'pdf_analysis.png')
    pdf_path = os.path.join(out_dir, 'pdf_analysis.pdf')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")

    json_path = os.path.join(out_dir, 'diagnostics.json')

    def _serialise(value):
        try:
            float_val = float(value)
            if float_val != float_val:  # NaN
                return None
            return float_val
        except (TypeError, ValueError):
            if isinstance(value, dict):
                return {str(k): _serialise(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_serialise(v) for v in value]
            return str(value)

    payload = {
        'model_dir':   model_dir,
        'mc_source':   args.mc_source,
        'phi_mapping': args.phi_mapping,
        'n_paths':     args.n_paths,
        'maturities':  T_values,
        'results':     {f"{T:.4f}": _serialise(d) for T, d in results.items()},
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {json_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
