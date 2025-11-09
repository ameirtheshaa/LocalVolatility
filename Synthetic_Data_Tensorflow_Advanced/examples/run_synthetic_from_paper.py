#!/usr/bin/env python3
"""
Synthetic Case Configuration from Paper

Reproduces the synthetic data experiment from:
Wang et al., "Deep self-consistent learning of local volatility"
Journal of Computational Finance 29(2), 2025, Section 4.1

This uses the large dataset (10×20 grid = 200 training points).
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig, VolatilityConfig
from dupire_pipeline import DupirePipeline


# def synthetic_volatility_paper(t, x):
#     """
#     Exact volatility function from paper equation (4.1):

#     σ(x,t) = 0.3 + y·exp(-y)
#     where y = (t + 0.1)·√(x + 0.1)
#     """
#     y = (t + 0.1) * np.sqrt(x + 0.1)
#     return 0.3 + y * np.exp(-y)

def constant_vol(t, x):
    """
    Exact volatility function from paper equation (4.1):

    σ(x,t) = 1
    """
    y = 1
    return y


def main():
    print("=" * 80)
    print("SYNTHETIC CASE FROM PAPER (Section 4.1)")
    print("=" * 80)
    print()

    # ========================================================================
    # CONFIGURATION - Only override what differs from defaults
    # ========================================================================

    # Start with defaults
    config = DupirePipelineConfig()

    # Override only paper-specific parameters
    config.volatility_config = VolatilityConfig.custom(constant_vol)
    config.lr_phi = 1e-3              # Paper uses 10^-3 (default is 10^-4)
    config.lr_eta = 1e-3              # Same learning rate for both networks
    config.M_train = 10**6            # Paper uses 10^6 MC paths (Section 4.1)
    config.output_dir = DupirePipelineConfig.normalize_output_dir(
        'synthetic_paper_large_dataset_constant_vol'
    )
    # config.num_epochs = 100

    # ========================================================================
    # Note: These are already defaults, so no need to set:
    #   - S0 = 1000
    #   - r = 0.04
    #   - T_max = 1.5
    #   - K_min = 500, K_max = 3000
    #   - T_min = 0.3
    #   - N_strikes = 20, N_maturities = 10
    #   - num_epochs = 30000
    #   - num_res_blocks = 3
    #   - units = 64
    #   - lambda_pde = 1.0
    #   - lr_decay_rate = 1.1, lr_decay_steps = 2000
    # ========================================================================

    print("Configuration (non-defaults only):")
    print(f"  Volatility: σ(x,t) = 0.3 + y·exp(-y), y = (t+0.1)√(x+0.1)")
    print(f"  Learning rates: {config.lr_phi}")
    print(f"  M_train (MC paths): {config.M_train:,}")
    print(f"  Output: {config.output_dir}")
    print()

    # Run the pipeline
    pipeline = DupirePipeline(config)
    pipeline.run()

    print()
    print("=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)
    print(f"Results: {pipeline.output_dir}")


if __name__ == "__main__":
    main()
