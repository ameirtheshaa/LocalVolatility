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

The figure subtitle states the SDE the paths were simulated under, so the
source must be labelled correctly. For --mc-source data that is normally the
ground-truth σ, but run_regenerate_mc.py also writes NN-repriced paths in the
same .npz schema (repriced_mc.npz); --data-vol resolves which.

Usage:
    python examples/run_reproduce_user_plots.py \\
        --model-dir <PATH> \\
        --mc-source {repriced,data} \\
        [--training-data PATH] \\
        [--data-vol {auto,ground-truth,nn}] \\
        [--maturities 0.25 0.5 0.75 1.0] \\
        [--n-paths 25000] \\
        [--output-dir DIR]

--n-paths sizes the repriced simulation only; --mc-source data takes whatever
sample count the .npz holds. diagnostics.json therefore reports n_mc_samples,
measured from the MC arrays, as the count backing the figures, and keeps the
requested knob separately as n_paths_requested (null when unused).

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
from dupire_pipeline import (MC_PROVENANCE_NN, MC_PROVENANCE_TRUTH, PDFAnalyzer,
                             load_trained_models)


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


def resolve_data_provenance(data_path: str, choice: str) -> str:
    """Which volatility surface generated the paths in `data_path`.

    'auto' keys off the filename: run_regenerate_mc.py writes NN-repriced paths
    to repriced_mc.npz, while training_data.npz and data_mc.npz hold
    ground-truth paths. Pass --data-vol explicitly for other filenames.
    """
    if choice == 'nn':
        return MC_PROVENANCE_NN
    if choice == 'ground-truth':
        return MC_PROVENANCE_TRUTH
    if 'repriced' in os.path.basename(data_path).lower():
        return MC_PROVENANCE_NN
    return MC_PROVENANCE_TRUTH


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
    parser.add_argument('--data-vol', choices=['auto', 'ground-truth', 'nn'],
                        default='auto',
                        help="For --mc-source data: which volatility surface "
                             "generated the .npz paths, used for the figure's "
                             "SDE subtitle. auto (default) = nn if the filename "
                             "contains 'repriced' (run_regenerate_mc.py's "
                             "repriced_mc.npz), else ground-truth.")
    parser.add_argument('--maturities', type=float, nargs='+',
                        default=[0.25, 0.5, 0.75, 1.0],
                        help='Maturities (default: 0.25 0.5 0.75 1.0).')
    parser.add_argument('--n-paths', type=int, default=25000,
                        help='MC paths to simulate, --mc-source repriced only '
                             '(default 25000). Ignored by --mc-source data, '
                             'whose sample count is fixed by the .npz; '
                             'diagnostics.json records the count actually used '
                             'as n_mc_samples either way.')
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
        mc_provenance = MC_PROVENANCE_NN
    else:
        data_path = args.training_data or os.path.join(model_dir, 'training_data.npz')
        _check_data_file(data_path)
        mc_provenance = resolve_data_provenance(data_path, args.data_vol)
        print(f"\nReading training-data MC from: {data_path}")
        print(f"  Path provenance: {mc_provenance}  (--data-vol {args.data_vol})")
        mc_data = analyzer.extract_mc_samples_from_training_data(
            data_path, T_values, verbose=True, provenance=mc_provenance)

    # Sample count actually backing the figures, measured from the returned
    # arrays rather than taken from --n-paths. --n-paths only feeds the repriced
    # simulator (via analysis_config.n_paths_analysis); on the --mc-source data
    # path it is never consulted, so recording it there asserts a count the .npz
    # does not have -- run_regenerate_mc.py writes 1,000,000 paths to
    # data_mc.npz and 100,000 to repriced_mc.npz against a 25,000 default.
    samples_by_T = {f"{T:.4f}": int(len(S_T)) for T, S_T in mc_data.items()}
    distinct_counts = set(samples_by_T.values())
    n_mc_samples = distinct_counts.pop() if len(distinct_counts) == 1 else None
    if n_mc_samples is not None:
        print(f'\nMC samples   : {n_mc_samples:,} per maturity')
    else:
        print(f'\nMC samples   : {samples_by_T}  (varies by maturity)')

    print()
    fig, results = analyzer.create_enhanced_pdf_analysis(
        mc_data, T_values=T_values,
        figure_label=f'MC source: {args.mc_source}',
        mc_provenance=mc_provenance)

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

    # n_mc_samples is measured from the MC arrays, so it describes the figures
    # for either source. n_paths_requested is the --n-paths knob and is null
    # unless the repriced simulator actually consumed it; the two were one
    # 'n_paths' field, which recorded the 25,000 default on --mc-source data
    # runs whose .npz held 1,000,000 samples.
    payload = {
        'model_dir':                model_dir,
        'mc_source':                args.mc_source,
        'mc_provenance':            mc_provenance,
        'phi_mapping':              args.phi_mapping,
        'n_mc_samples':             n_mc_samples,
        'n_mc_samples_by_maturity': samples_by_T,
        'n_paths_requested':        args.n_paths if args.mc_source == 'repriced' else None,
        'maturities':               T_values,
        'results':                  {f"{T:.4f}": _serialise(d) for T, d in results.items()},
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {json_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
