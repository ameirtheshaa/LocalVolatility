# File Catalog — LocalVolatility

## Root

| File | Role |
|------|------|
| `README.md` | Project overview |
| `requirements.txt` | Root dependencies |
| `jcf_privault_online_early.pdf` | Wang et al. JCF 2025 paper |
| `LICENSE` | MIT |

## Subprojects

| Directory | Purpose | Main entry |
|-----------|---------|------------|
| `DAX_Tensorflow/` | DAX call options Aug 2001 | `tf_NN_call_DAX.py` |
| `SPX_Tensorflow/` | SPX puts May 2019 | `tf_NN_put_SPX.py` |
| `Synthetic_Data_Tensorflow/` | Early synthetic MC NNs | `tf_NN_call_MC.py` |
| `Synthetic_Data_PyTorch/` | PyTorch port | `run.py` |
| `Synthetic_Data_Tensorflow_Advanced/` | **Production Dupire pipeline** | `dupire_pipeline.py` |

## Advanced pipeline (primary)

| Path | Role |
|------|------|
| `config.py` | `DupirePipelineConfig`, presets |
| `dupire_pipeline.py` | `DataGenerator`, `DupireNeuralModel`, `PDFAnalyzer`, `DupirePipeline` |
| `analytical_solutions.py` | Black–Scholes benchmarks |
| `compare_nn_vs_analytical.py` | NN vs BS for constant vol |
| `examples/run_quick_test.py` | 100-epoch smoke test |
| `examples/run_full_training.py` | 30k-epoch training |
| `examples/run_analysis_only.py` | PDF analysis on saved models |
| `examples/run_synthetic_from_paper.py` | Constant σ=1 paper-style run |
| `examples/dax_analysis_common.py` | Shared DAX PDF helpers |
| `examples/run_dax_*.py` | DAX market PDF (pipeline / direct) |
| `docs/*.md` | Math, setup, quick start |
| `models/example_pretrained/` | Small pretrained demo models |
| `models/runs/<experiment>/` | Training + analysis artifacts |

## Experiment run directories

Under `Synthetic_Data_Tensorflow_Advanced/models/runs/`:

- `synthetic_paper_large_dataset_constant_vol` — main constant-vol study
- `synthetic_paper_large_dataset_constant_vol_HPC` — HPC comparison plots
- `synthetic_paper_large_dataset_constant_vol_new` — Dec 2025 rerun
- `synthetic_paper_large_dataset_dupire_exact` — Dupire-exact σ(t,x)
- `dax_{7,8,9}aug_pdf_analysis` — DAX pipeline PDF outputs
- `dax_{7,8,9}aug_pdf_analysis_direct` — DAX direct PDF outputs

## Onboarding docs

| File | Role |
|------|------|
| `docs/EXPERIMENT_LOG.md` | Run history |
| `docs/AGENT_HANDOFF.md` | Continuation notes |
| `docs/DAX_ANALYSIS.md` | DAX script semantics |
| `docs/ENVIRONMENT.md` | Python / venv |
| `docs/project_backfill_profile.yaml` | Agent backfill config |
