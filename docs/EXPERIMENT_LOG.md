# Experiment Log — LocalVolatility

Chronological record of major runs. Full repo holds artifacts; source mirror: `LocalVolatility-lite` on GitHub.

## 2024 Q4 — Legacy TensorFlow (market + synthetic)

| ID | Subproject | Description | Outputs |
|----|------------|-------------|---------|
| dax-7aug | `DAX_Tensorflow` | DAX calls, 7 Aug 2001, ldup=1, 3 resblocks | `7aug_3resblock_2024_11_14_17_37/` |
| dax-8aug | `DAX_Tensorflow` | DAX calls, 8 Aug 2001 | `8aug_3resblock_2024_11_15_12_49/` |
| dax-9aug | `DAX_Tensorflow` | DAX calls, 9 Aug 2001 | `9aug_3resblock_2024_11_15_02_22/` |
| spx-19may | `SPX_Tensorflow` | SPX puts, 19 May 2019 | `19may_spx_3resblock_2024_11_18_11_11/` |
| syn-3res | `Synthetic_Data_Tensorflow` | Synthetic MC, full grid | `synthetic_data_3resblock_2024_11_18_11_00/` |
| syn-small | `Synthetic_Data_Tensorflow` | Synthetic MC, small dataset | `synthetic_data_small_dataset_3resblock_2024_11_18_11_13/` |

## 2025 Jan — Advanced package

Documented Dupire pipeline (`Synthetic_Data_Tensorflow_Advanced`): `config.py`, `dupire_pipeline.py`, examples, `docs/MATHEMATICAL_TREATMENT.md`, supervisor `PACKAGE_SUMMARY.md`.

## 2025 Nov — Paper-style synthetic (constant σ = 1)

| Run dir | Config highlights | Notes |
|---------|-------------------|-------|
| `synthetic_paper_large_dataset_constant_vol` | `M_train=10⁶`, 30k epochs, `lr=1e-3`, σ=1 | Main run; NN vs BS comparison Nov 2 |
| `synthetic_paper_large_dataset_constant_vol_HPC` | Same + HPC plots | `nn_vs_bs_*`, vol comparison Nov 1 |
| `synthetic_paper_large_dataset_constant_vol_new` | Rerun | Dec 9 PDF |

**NN vs BS metrics** (`comparison_metrics_20251102_161011.json`): option price RMSE ≈ 1.26; local vol RMSE ≈ 0.0098 vs σ=1.

**Verified 2026-05-21:** `run_analysis_only` on constant_vol run (5k MC paths, T=0.5,1.0) → `pdf_analysis_20260521_143540.png/pdf`.

## 2025 Nov–Dec — Dupire-exact synthetic

| Run dir | Volatility | Notes |
|---------|------------|-------|
| `synthetic_paper_large_dataset_dupire_exact` | σ(t,x)=0.3+y·exp(-y) | `training_data.npz` present locally (~40 MB) |

## 2025 Dec — DAX PDF analysis (Advanced)

| Run dir | Script family | Sample artifact |
|---------|---------------|-----------------|
| `dax_7aug_pdf_analysis` | pipeline | `pdf_analysis_20251212_230526.png` |
| `dax_7aug_pdf_analysis_direct` | direct | `pdf_analysis_direct_20251212_235509.png` |
| `dax_8aug_pdf_analysis` / `_direct` | both | Dec 12 23:05–23:56 |
| `dax_9aug_pdf_analysis` | pipeline | Two runs (Dec 11, Dec 12) |
| `dax_9aug_pdf_analysis_direct` | direct | Dec 12 23:57 |

See [DAX_ANALYSIS.md](DAX_ANALYSIS.md) for pipeline vs direct behavior.
