# DAX PDF Analysis Scripts

December 2025 workflows for validating legacy DAX call-option models (Aug 7–9, 2001) with the Advanced pipeline’s PDF machinery.

## Two paths

| Path | Scripts | Output prefix | Mechanism |
|------|---------|---------------|-----------|
| **Pipeline** | `run_dax_{7,8,9}aug.py` | `pdf_analysis_<timestamp>` | `DupirePipeline` + `DupirePipelineConfig.analysis_only()` → `stage3_pdf_analysis()` |
| **Direct** | `run_dax_{7,8,9}aug_direct.py` | `pdf_analysis_direct_<timestamp>` | `load_trained_models()` + `PDFAnalyzer` directly; no pipeline orchestration |

Batch entry: `run_dax_market_data.py` (`--pipeline`, `--direct`, or both).

**IBP comparison (legacy vs corrected φ̃):** `examples/run_dax_ibp_comparison_plots.py` → `plots/ibp_comparison_dax/{7aug,8aug,9aug}/{pre_correction,pretrained}/` (retrained deferred).

Shared logic: [`examples/dax_analysis_common.py`](../Synthetic_Data_Tensorflow_Advanced/examples/dax_analysis_common.py).

## Forward-looking density result (7 Aug 2001)

[`presentation/journal_forward_density.tex`](../Synthetic_Data_Tensorflow_Advanced/presentation/journal_forward_density.tex) / `.pdf` — a finished,
self-contained 11-slide result, distinct from the pipeline/direct validation
scripts above: recovers the full risk-neutral density of the DAX at every
horizon from a single day's option quotes (7 Aug 2001, 217 calls, 5
maturities) via Breeden–Litzenberger applied to the self-consistent
exp_k-ansatz Dupire PINN calibration `nbexpk7` — no historical prices, no
forecasting model.

Validated three independent ways (figures in
`presentation/figures/forward_density/`):

| Check | Figure | Result |
|---|---|---|
| Extraction (autodiff vs. central difference on the network's own output) | `fig3_extraction` | agree to 2.2e-3 of peak; both theoretical error exponents (−2 roundoff, +2 truncation) recovered |
| Self-consistency (analytic density vs. a 40,000-path MC reprice of the same learned local vol) | `fig6_selfconsistency` | KS ≤ 0.0047 across all five maturities |
| Shape vs. realised path (Nicolas's discounted-DAX method) | `fig5_realised` | both left-skewed in the same direction; explicitly not a formal test (one ℙ-path vs. a ℚ-density) |

Explicit scope, per the deck's own closing slide: this is the market's
risk-neutral view on that one day, not a ℙ-measure forecast.

## Model and data sources

| Date | S₀ | Legacy model dir | CSV |
|------|-----|------------------|-----|
| 7 Aug 2001 | 5752.51 | `DAX_Tensorflow/7aug_3resblock_2024_11_14_17_37/` | `dataTrain_7_August_2001.csv` |
| 8 Aug 2001 | 5614.51 | `DAX_Tensorflow/8aug_3resblock_2024_11_15_12_49/` | `dataTrain_8_August_2001.csv` |
| 9 Aug 2001 | 5512.28 | `DAX_Tensorflow/9aug_3resblock_2024_11_15_02_22/` | `dataTrain_9_August_2001.csv` |

Analysis outputs go under `Synthetic_Data_Tensorflow_Advanced/models/runs/`:

- `dax_7aug_pdf_analysis` / `dax_7aug_pdf_analysis_direct`
- `dax_8aug_pdf_analysis` / `dax_8aug_pdf_analysis_direct`
- `dax_9aug_pdf_analysis` / `dax_9aug_pdf_analysis_direct`

## Pipeline path details

1. Copies `NN_phi.keras` / `NN_eta.keras` from the legacy folder into the run output dir (as `*_final.keras` for the loader).
2. Builds market-scaled metadata from the CSV (`T_min`/`T_max`, `K_min`/`K_max`, `S₀`, `r=0.04`).
3. Sets `reuse_training_mc=False` (no synthetic `training_data.npz` for DAX).
4. Runs MC with NN-predicted volatility, then saves 3-panel PDF plots via `DupirePipeline`.

## Direct path details

1. Loads models in place from `DAX_Tensorflow/...` (no copy).
2. Builds a minimal `metadata['scaling']` dict for `PDFAnalyzer`.
3. Calls `simulate_paths_with_nn_volatility` and `create_enhanced_pdf_analysis` explicitly.
4. Writes `pdf_analysis_direct_*` plus `analysis_summary_<timestamp>.json`.

## Usage

```bash
cd Synthetic_Data_Tensorflow_Advanced
source .venv/bin/activate   # Python 3.12 + TensorFlow 2.16

python examples/run_dax_7aug.py
python examples/run_dax_7aug_direct.py
python examples/run_dax_market_data.py --all
```

**Prerequisite:** DAX `NN_phi.keras` / `NN_eta.keras` must be non-empty (Dropbox offline). If placeholders are 0 bytes, sync `DAX_Tensorflow/*aug_3resblock_*` first.

## Historical runs (Dec 2025)

Existing artifacts (when synced from Dropbox):

- `dax_7aug_pdf_analysis/pdf_analysis_20251212_230526.*`
- `dax_7aug_pdf_analysis_direct/pdf_analysis_direct_20251212_235509.*`
- Similar timestamps for 8 Aug and 9 Aug (9 Aug has two pipeline runs: Dec 11 and Dec 12).
