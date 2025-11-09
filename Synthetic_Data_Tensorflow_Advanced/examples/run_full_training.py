#!/usr/bin/env python3
"""
Full Training Example - Dupire Local Volatility Pipeline

This script runs the complete training pipeline with production parameters.
Use this for:
- Final results
- Publication-quality outputs
- Research experiments

Expected runtime:
- CPU: ~30-60 minutes
- GPU: ~10-15 minutes

Expected results:
- Full convergence (30,000 epochs)
- Volatility RMSE: <1% (excellent fit)
- High-quality PDF agreement
- Publication-ready plots
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig, VolatilityConfig
from dupire_pipeline import DupirePipeline


def resolve_output_dir(raw_path: str) -> str:
    """Map simple directory names into models/runs/ for consistency."""
    if not raw_path:
        return raw_path
    return DupirePipelineConfig.normalize_output_dir(raw_path)


def main():
    print("=" * 80)
    print("FULL TRAINING - DUPIRE LOCAL VOLATILITY PIPELINE")
    print("=" * 80)
    print()
    print("This will run the complete pipeline with production parameters:")
    print("  - 30,000 training epochs")
    print("  - 10,000 Monte Carlo paths for training")
    print("  - 25,000 paths for validation")
    print("  - Full PDF analysis")
    print()
    print("⏱  Expected runtime:")
    print("     CPU: ~30-60 minutes")
    print("     GPU: ~10-15 minutes")
    print("=" * 80)
    print()

    # Use Dupire exact volatility model (synthetic)
    print("Using DUPIRE EXACT volatility model (synthetic)")
    print("  σ(t,x) = 0.3 + y·exp(-y) where y = (t + 0.1)·√(x + 0.1)")
    print()

    config = DupirePipelineConfig.full_training()

    # Display configuration
    print()
    print("Current configuration:")
    print(f"  - Epochs: {config.num_epochs:,}")
    print(f"  - Training paths: {config.M_train:,}")
    print(f"  - λ_PDE: {config.lambda_pde}")
    print(f"  - Residual blocks: {config.num_res_blocks}")
    print(f"  - Output: {config.output_dir}")
    print()

    # Optional: customize output directory
    raw_custom_dir = input("Custom output directory (press Enter to use default): ").strip()
    if raw_custom_dir:
        custom_dir = resolve_output_dir(raw_custom_dir)
        config.output_dir = custom_dir
        print(f"  → Output directory set to: {config.output_dir}")
        if custom_dir != raw_custom_dir:
            print("    (relative names are stored under models/runs/)")

    print()
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    print()

    # Run the pipeline
    pipeline = DupirePipeline(config)
    pipeline.run()

    print()
    print("=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {pipeline.output_dir}")
    print()
    print("Generated files:")
    print(f"  - {pipeline.output_dir}/NN_phi_final.keras")
    print(f"  - {pipeline.output_dir}/NN_eta_final.keras")
    print(f"  - {pipeline.output_dir}/metadata.json")
    print(f"  - {pipeline.output_dir}/training_data.npz")
    print(f"  - {pipeline.output_dir}/pdf_analysis_*.png")
    print()
    print("Next steps:")
    print("  - Check PDF analysis plots for validation")
    print("  - Verify volatility RMSE < 1%")
    print("  - Re-run analysis: python examples/run_analysis_only.py --model-dir", pipeline.output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
