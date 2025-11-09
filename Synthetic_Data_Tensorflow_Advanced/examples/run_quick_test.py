#!/usr/bin/env python3
"""
Quick Test Example - Dupire Local Volatility Pipeline

This script runs a quick test of the pipeline with reduced parameters.
Perfect for:
- Testing the installation
- Quick debugging
- Experimenting with parameters

Expected runtime:
- CPU: ~2 minutes
- GPU: ~30 seconds

Expected results:
- Training will NOT fully converge (only 100 epochs)
- Volatility RMSE: ~10-20% (rough approximation)
- PDF plots will be somewhat noisy

For production results, use run_full_training.py instead.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig
from dupire_pipeline import DupirePipeline


def main():
    print("=" * 80)
    print("QUICK TEST - DUPIRE LOCAL VOLATILITY PIPELINE")
    print("=" * 80)
    print()
    print("This quick test will:")
    print("  - Generate training data (1,000 MC paths)")
    print("  - Train for 100 epochs (~2 minutes)")
    print("  - Analyze and generate PDF plots")
    print()
    print("Note: Results will NOT be publication-quality!")
    print("Use run_full_training.py for final results.")
    print("=" * 80)
    print()

    # Use the quick test preset
    config = DupirePipelineConfig.quick_test()

    # You can customize further if needed:
    # config.output_dir = 'my_quick_test'
    # config.num_epochs = 200  # Double the epochs for better results
    # config.lambda_pde = 2.0  # Increase PDE weight

    # Run the pipeline
    pipeline = DupirePipeline(config)
    pipeline.run()

    print()
    print("=" * 80)
    print("✓ QUICK TEST COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {pipeline.output_dir}")
    print()
    print("Next steps:")
    print("  - Check PDF plots in the output directory")
    print("  - Look at loss curves (should be decreasing but not converged)")
    print("  - For better results, run: python examples/run_full_training.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
