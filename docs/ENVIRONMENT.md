# Environment — LocalVolatility

## Python

- **Recommended:** Python 3.12 (Anaconda: `/opt/anaconda3/bin/python3.12`)
- TensorFlow **2.10–2.16** (tested with **2.16.2**)
- Python 3.14+ is not supported by TensorFlow on macOS

## Advanced pipeline venv

```bash
cd Synthetic_Data_Tensorflow_Advanced
/opt/anaconda3/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `tensorflow`, `numpy`, `scipy`, `matplotlib`, `scikit-learn` (see `requirements.txt`).

## Verify install

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python examples/run_analysis_only.py \
  --model-dir synthetic_paper_large_dataset_constant_vol \
  --n-paths 5000 --maturities 0.5 1.0
```

Expect new `pdf_analysis_<timestamp>.png` under `models/runs/synthetic_paper_large_dataset_constant_vol/`.

## Apple Silicon

For GPU acceleration on M-series Macs, consider `tensorflow-macos` + `tensorflow-metal` (see TensorFlow install docs). CPU runs are sufficient for analysis-only workflows.

## Dropbox note

If `pip` or `python` succeed but `load_model` fails, check model files are not 0-byte placeholders (`wc -c models/runs/.../NN_phi_final.keras`).
