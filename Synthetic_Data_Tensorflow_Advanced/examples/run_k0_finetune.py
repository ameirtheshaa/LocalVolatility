#!/usr/bin/env python3
"""Fine-tune pretrained models with strong K=0 call boundary penalty."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DupirePipelineConfig
from dupire_pipeline import DupirePipeline


def main():
    config = DupirePipelineConfig.full_training()
    config.mode = 'all'
    config.skip_if_exists = False
    config.num_epochs = 3000
    config.lambda_k0 = 100.0
    config.print_epochs = 500
    config.save_epochs = 3000
    config.output_dir = os.path.join('models', 'runs', 'k0_bc_finetune')
    config.init_model_dir = os.path.join('models', 'example_pretrained')
    config.analysis_config.n_paths_analysis = 25000

    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_epochs}, lambda_k0: {config.lambda_k0}")
    DupirePipeline(config).run()


if __name__ == '__main__':
    main()
