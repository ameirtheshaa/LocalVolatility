#!/usr/bin/env python3
"""Train from scratch with K=0 BC penalty and extended low-K grid."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig
from dupire_pipeline import DupirePipeline


def main():
    config = DupirePipelineConfig.full_training()
    config.mode = 'all'
    config.skip_if_exists = False
    config.K_min = 100.0
    config.num_epochs = 3000
    config.lambda_k0 = 10.0
    config.print_epochs = 500
    config.save_epochs = 3000
    config.output_dir = os.path.join('models', 'runs', 'k0_bc_retrain')
    config.analysis_config.T_analysis = [0.5, 1.0, 1.5]
    config.analysis_config.n_paths_analysis = 25000
    config.analysis_config.reuse_training_mc = False
    config.analysis_config.run_mc_with_nn_volatility = True

    print(f"K_min={config.K_min}, lambda_k0={config.lambda_k0}, epochs={config.num_epochs}")
    DupirePipeline(config).run()


if __name__ == '__main__':
    main()
