#!/usr/bin/env python3
"""
Shared helpers for DAX PDF analysis example scripts.

Two entry styles:
- pipeline: DupirePipeline.stage3 via analysis_only config (output prefix pdf_analysis_)
- direct: PDFAnalyzer invoked directly (output prefix pdf_analysis_direct_)
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Parent package (Synthetic_Data_Tensorflow_Advanced)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AnalysisConfig, DupirePipelineConfig, PlotConfig
from dupire_pipeline import DupirePipeline, PDFAnalyzer, load_trained_models, save_metadata

# Legacy DAX training folders (ldup=1 runs from Nov 2024)
DAX_RUNS = {
    '7aug': {
        'date_label': '7_August_2001',
        'S0': 5752.51,
        'model_dir': os.path.join('..', '..', 'DAX_Tensorflow', '7aug_3resblock_2024_11_14_17_37'),
        'csv': os.path.join('..', '..', 'DAX_Tensorflow', 'dataTrain_7_August_2001.csv'),
        'output_pipeline': 'dax_7aug_pdf_analysis',
        'output_direct': 'dax_7aug_pdf_analysis_direct',
    },
    '8aug': {
        'date_label': '8_August_2001',
        'S0': 5614.51,
        'model_dir': os.path.join('..', '..', 'DAX_Tensorflow', '8aug_3resblock_2024_11_15_12_49'),
        'csv': os.path.join('..', '..', 'DAX_Tensorflow', 'dataTrain_8_August_2001.csv'),
        'output_pipeline': 'dax_8aug_pdf_analysis',
        'output_direct': 'dax_8aug_pdf_analysis_direct',
    },
    '9aug': {
        'date_label': '9_August_2001',
        'S0': 5512.28,
        'model_dir': os.path.join('..', '..', 'DAX_Tensorflow', '9aug_3resblock_2024_11_15_02_22'),
        'csv': os.path.join('..', '..', 'DAX_Tensorflow', 'dataTrain_9_August_2001.csv'),
        'output_pipeline': 'dax_9aug_pdf_analysis',
        'output_direct': 'dax_9aug_pdf_analysis_direct',
    },
}

RISK_FREE = 0.04


def _repo_relative(*parts: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(base, *parts))


def _runs_dir(name: str) -> str:
    return _repo_relative('models', 'runs', name)


def _read_market_grid(csv_path: str) -> Tuple[float, float, float, float, List[float]]:
    """Return t_min, t_max, k_min, k_max and suggested analysis maturities from DAX CSV."""
    df = pd.read_csv(csv_path)
    # Call options only (same filter as legacy tf_NN_call_DAX.py)
    mask = df['Option\ntype'] == 1
    T = df.loc[mask, 'Maturity'].astype(float).values
    K = df.loc[mask, 'Strike'].astype(float).values
    t_min, t_max = float(T.min()), float(T.max())
    k_min, k_max = float(K.min()), float(K.max())
    # Use three interior maturities for PDF panels
    qs = [0.25, 0.5, 0.75]
    maturities = sorted({float(np.quantile(T, q)) for q in qs})
    return t_min, t_max, k_min, k_max, maturities


def _ensure_models_in_output(model_dir: str, output_dir: str) -> None:
    """Copy legacy NN_phi/NN_eta.keras into analysis output dir if missing."""
    os.makedirs(output_dir, exist_ok=True)
    for name in ('NN_phi.keras', 'NN_eta.keras', 'NN_phi_final.keras', 'NN_eta_final.keras'):
        src = os.path.join(model_dir, name.replace('_final', ''))
        if not os.path.exists(src):
            continue
        dst = os.path.join(output_dir, name if 'final' in name else name)
        if not os.path.exists(dst) or os.path.getsize(dst) == 0:
            shutil.copy2(src, dst)
    # Pipeline loader prefers *_final names
    for base in ('NN_phi', 'NN_eta'):
        plain = os.path.join(output_dir, f'{base}.keras')
        final = os.path.join(output_dir, f'{base}_final.keras')
        if os.path.exists(plain) and (not os.path.exists(final) or os.path.getsize(final) == 0):
            shutil.copy2(plain, final)


def _build_dax_config(
    spec: Dict,
    output_dir: str,
    maturities: List[float],
) -> DupirePipelineConfig:
    t_min, t_max, k_min, k_max, _ = _read_market_grid(
        os.path.normpath(os.path.join(os.path.dirname(__file__), spec['csv']))
    )
    config = DupirePipelineConfig.analysis_only(output_dir)
    config.S0 = spec['S0']
    config.r = RISK_FREE
    config.T_min = t_min
    config.T_max = t_max
    config.K_min = k_min
    config.K_max = k_max
    config.analysis_config = AnalysisConfig(
        T_analysis=maturities,
        n_paths_analysis=25000,
        reuse_training_mc=False,
        run_mc_with_nn_volatility=True,
    )
    config.plot_config = PlotConfig(
        enable_pdf_plots=True,
        save_png=True,
        save_pdf=True,
        dpi=450,
    )
    # Persist scaling for PDFAnalyzer (matches legacy DAX normalization)
    config.output_dir = output_dir
    save_metadata(config, output_dir)
    return config


def run_pipeline_analysis(day_key: str) -> str:
    """Run Stage 3 through DupirePipeline (filenames: pdf_analysis_<timestamp>)."""
    spec = DAX_RUNS[day_key]
    model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), spec['model_dir']))
    output_dir = _runs_dir(spec['output_pipeline'])
    _ensure_models_in_output(model_dir, output_dir)

    if os.path.getsize(os.path.join(output_dir, 'NN_phi_final.keras')) == 0:
        raise FileNotFoundError(
            f"DAX models in {model_dir} are empty (Dropbox placeholders). "
            "Make the DAX_Tensorflow run folders available offline, then retry."
        )

    _, _, _, _, maturities = _read_market_grid(
        os.path.normpath(os.path.join(os.path.dirname(__file__), spec['csv']))
    )
    config = _build_dax_config(spec, output_dir, maturities)
    DupirePipeline(config).run()
    return output_dir


def run_direct_analysis(day_key: str) -> str:
    """Call PDFAnalyzer directly (filenames: pdf_analysis_direct_<timestamp>)."""
    import matplotlib.pyplot as plt

    spec = DAX_RUNS[day_key]
    model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), spec['model_dir']))
    output_dir = _runs_dir(spec['output_direct'])
    os.makedirs(output_dir, exist_ok=True)

    if any(os.path.getsize(os.path.join(model_dir, f)) == 0 for f in ('NN_phi.keras', 'NN_eta.keras')):
        raise FileNotFoundError(
            f"DAX models in {model_dir} are empty (Dropbox placeholders). "
            "Make the DAX_Tensorflow run folders available offline, then retry."
        )

    csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), spec['csv']))
    t_min, t_max, k_min, k_max, maturities = _read_market_grid(csv_path)

    config = DupirePipelineConfig()
    config.S0 = spec['S0']
    config.r = RISK_FREE
    config.T_min = t_min
    config.T_max = t_max
    config.K_min = k_min
    config.K_max = k_max
    config.analysis_config.T_analysis = maturities
    config.analysis_config.n_paths_analysis = 25000
    config.analysis_config.reuse_training_mc = False
    config.analysis_config.run_mc_with_nn_volatility = True

    metadata = {
        'scaling': {
            'S0': spec['S0'],
            'r': RISK_FREE,
            't_max': t_max,
            'k_min': k_min,
            'k_max': k_max,
        },
        'source': 'dax_direct_analysis',
        'model_dir': model_dir,
    }

    nn_phi, nn_eta, _ = load_trained_models(model_dir)
    analyzer = PDFAnalyzer(nn_phi, nn_eta, config, metadata)
    analyzer.training_T_min = t_min
    analyzer.training_T_max = t_max

    mc_data = analyzer.simulate_paths_with_nn_volatility(maturities, verbose=True)
    fig, results = analyzer.create_enhanced_pdf_analysis(mc_data, maturities)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base = f'pdf_analysis_direct_{timestamp}'
    png_path = os.path.join(output_dir, f'{base}.png')
    pdf_path = os.path.join(output_dir, f'{base}.pdf')
    fig.savefig(png_path, dpi=config.plot_config.dpi, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=config.plot_config.dpi, bbox_inches='tight')
    plt.close(fig)

    summary_path = os.path.join(output_dir, f'analysis_summary_{timestamp}.json')
    with open(summary_path, 'w') as fh:
        json.dump({str(T): res for T, res in results.items()}, fh, indent=2, default=str)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    return output_dir
