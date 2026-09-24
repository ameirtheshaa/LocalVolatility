# Agent Handoff — LocalVolatility

Last updated: 2026-05-21 (onboarding restore).

## Active blocker

- **Dropbox placeholders:** Many artifacts under `models/runs/` and `DAX_Tensorflow/*aug_*/` may still be 0 bytes locally. Source code and constant-vol keras were restored from `LocalVolatility-lite` + GitHub; force offline sync for plots and DAX keras before re-running DAX scripts.

## Continuation priority

1. Sync remaining Dropbox artifacts (DAX `NN_*.keras`, historical `pdf_analysis_*.png`).
2. Re-run DAX analysis after sync: `cd Synthetic_Data_Tensorflow_Advanced && .venv/bin/python examples/run_dax_7aug.py`.
3. Optional: train Dupire-exact run to completion if only `training_data.npz` exists without hydrated models.

## Working environment

- Python **3.12** venv: `Synthetic_Data_Tensorflow_Advanced/.venv` (TensorFlow 2.16.2).
- Quick validation: `.venv/bin/python examples/run_analysis_only.py --model-dir synthetic_paper_large_dataset_constant_vol`

## Do not repeat

- Do not assume `run_synthetic_from_paper.py` uses Dupire-exact vol — it uses **constant σ=1**; exact vol is `synthetic_paper_large_dataset_dupire_exact`.
- Do not edit against 0-byte files; check `wc -c` first.
- DAX scripts were reconstructed Dec 2025 → May 2026 in `examples/dax_analysis_common.py`; confirm against Dropbox originals if they hydrate.

## Key paths

| What | Where |
|------|--------|
| Main pipeline | `Synthetic_Data_Tensorflow_Advanced/dupire_pipeline.py` |
| Config | `Synthetic_Data_Tensorflow_Advanced/config.py` |
| Best-documented runs | `models/runs/synthetic_paper_large_dataset_constant_vol/` |
| DAX docs | `docs/DAX_ANALYSIS.md` |
| Paper PDF | `jcf_privault_online_early.pdf` |
| Lite source mirror | `../LocalVolatility-lite` |

## Git

`.git` restored from `github.com/ameirtheshaa/LocalVolatility` (May 2026). Prior Dropbox copy had empty `HEAD`/`index`.
