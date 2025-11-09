#!/usr/bin/env python3
"""
Simple Custom Configuration Runner

Import defaults from global config and override specific parameters.
Edit the parameters below and run: python examples/run_custom_config.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig, VolatilityConfig
from dupire_pipeline import DupirePipeline


def main():
    # ========================================================================
    # CONFIGURE YOUR CUSTOM PARAMETERS HERE
    # ========================================================================

    # Start with default configuration
    config = DupirePipelineConfig.quick_test()

    # Override volatility: Custom constant volatility sigma(t,x) = 1.0
    config.volatility_config = VolatilityConfig.custom(lambda t, x: 1.0)

    # Override training parameters
    config.num_epochs = 100
    config.output_dir = DupirePipelineConfig.normalize_output_dir('my_custom_run2')

    # Optional: Override other parameters as needed
    # config.lambda_pde = 2.0
    # config.num_res_blocks = 4
    # config.M_train = 10000
    # config.learning_rate = 5e-4

    # ========================================================================
    # RUN PIPELINE
    # ========================================================================

    print(f"Running pipeline with custom volatility (sigma=1.0), epochs={config.num_epochs}")

    pipeline = DupirePipeline(config)
    pipeline.run()

    print(f"\n✓ Complete! Results saved to: {pipeline.output_dir}")


if __name__ == "__main__":
    main()
