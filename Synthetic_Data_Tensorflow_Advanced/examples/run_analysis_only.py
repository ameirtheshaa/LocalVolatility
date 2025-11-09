#!/usr/bin/env python3
"""
Analysis Only Example - Dupire Local Volatility Pipeline

This script loads pre-trained models and runs PDF analysis.
Use this when you:
- Already have trained models
- Want to re-run analysis with different parameters
- Need to generate additional plots
- Want to test different maturities

Usage:
    python run_analysis_only.py --model-dir path/to/trained/models
    python run_analysis_only.py --model-dir models/runs/synthetic_data_3resblock_20250129_143022

Optional arguments:
    --n-paths N          : Number of MC paths (default: 25000)
    --maturities T1 T2   : Maturities to analyze (default: 0.5 1.0 1.5)
    --no-plots           : Skip plot generation (only compute statistics)
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig, AnalysisConfig
from dupire_pipeline import DupirePipeline


def resolve_model_dir(raw_path: str) -> str:
    """Resolve model directory, defaulting relative names into models/runs/."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Run PDF analysis on pre-trained Dupire models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_analysis_only.py --model-dir models/runs/synthetic_data_3resblock_20250129_143022

  # High-quality analysis with more paths
  python run_analysis_only.py --model-dir models/runs/my_models --n-paths 50000

  # Custom maturities
  python run_analysis_only.py --model-dir models/runs/my_models --maturities 0.25 0.5 0.75 1.0

  # Statistics only (no plots)
  python run_analysis_only.py --model-dir models/runs/my_models --no-plots
        """
    )

    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing trained models (NN_phi_final.keras, NN_eta_final.keras). '
                            'Relative names are resolved under models/runs/.')

    parser.add_argument('--n-paths', type=int, default=25000,
                       help='Number of Monte Carlo paths for validation (default: 25000)')

    parser.add_argument('--maturities', type=float, nargs='+', default=[0.5, 1.0, 1.5],
                       help='Maturities to analyze in years (default: 0.5 1.0 1.5)')

    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation (only compute statistics)')

    args = parser.parse_args()

    print("=" * 80)
    print("ANALYSIS ONLY - DUPIRE LOCAL VOLATILITY PIPELINE")
    print("=" * 80)
    print()
    model_dir = resolve_model_dir(args.model_dir)
    print(f"Model directory: {model_dir}")
    print(f"MC paths: {args.n_paths:,}")
    print(f"Maturities: {args.maturities}")
    print(f"Generate plots: {not args.no_plots}")
    print("=" * 80)
    print()

    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"✗ Error: Model directory not found: {model_dir}")
        print()
        print("Please provide a valid directory containing:")
        print("  - NN_phi_final.keras (or NN_phi.keras)")
        print("  - NN_eta_final.keras (or NN_eta.keras)")
        print("  - metadata.json (optional)")
        sys.exit(1)

    # Check for model files
    phi_exists = (
        os.path.exists(os.path.join(model_dir, 'NN_phi_final.keras')) or
        os.path.exists(os.path.join(model_dir, 'NN_phi.keras'))
    )
    eta_exists = (
        os.path.exists(os.path.join(model_dir, 'NN_eta_final.keras')) or
        os.path.exists(os.path.join(model_dir, 'NN_eta.keras'))
    )

    if not phi_exists or not eta_exists:
        print(f"✗ Error: Model files not found in {model_dir}")
        print()
        print("Required files:")
        print("  - NN_phi_final.keras (or NN_phi.keras)")
        print("  - NN_eta_final.keras (or NN_eta.keras)")
        print()
        print("Found in directory:")
        for f in os.listdir(model_dir):
            if f.endswith('.keras'):
                print(f"  ✓ {f}")
        sys.exit(1)

    print("✓ Model files found!")
    print()

    # Create analysis-only configuration
    config = DupirePipelineConfig.analysis_only(model_dir)

    # Customize analysis parameters
    config.analysis_config.n_paths_analysis = args.n_paths
    config.analysis_config.T_analysis = args.maturities

    # Control plot generation
    if args.no_plots:
        config.plot_config.enable_pdf_plots = False
        print("Plot generation disabled (--no-plots flag)")
    else:
        config.plot_config.enable_pdf_plots = True
        config.plot_config.save_png = True
        config.plot_config.save_pdf = True

    print("=" * 80)
    print("Running analysis...")
    print("=" * 80)
    print()

    # Run the pipeline
    pipeline = DupirePipeline(config)
    pipeline.run()

    print()
    print("=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {pipeline.output_dir}")
    print()

    if not args.no_plots:
        print("Generated plots:")
        for f in os.listdir(pipeline.output_dir):
            if f.startswith('pdf_analysis_') and (f.endswith('.png') or f.endswith('.pdf')):
                print(f"  - {f}")
        print()

    print("Next steps:")
    print("  - Review PDF plots for each maturity")
    print("  - Check Gaussian space plots for log-normal fit quality")
    print("  - Compare MC histogram vs NN model curves")
    print("  - Look at skewness and kurtosis statistics")
    print()
    print("To re-run with different parameters:")
    print(f"  python {sys.argv[0]} --model-dir {model_dir} --n-paths 50000")
    print("=" * 80)


if __name__ == "__main__":
    main()
