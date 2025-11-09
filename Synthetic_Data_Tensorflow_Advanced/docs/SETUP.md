# Setup Guide - Dupire Local Volatility Pipeline

**Complete installation and setup instructions for supervisors and new users**

---

## ⚡ Fastest Path: Analyze Pre-trained Models (30 seconds)

**Just want to see it work?** Skip the full setup and analyze pre-trained models immediately:

```bash
cd dupire_clean
pip install -r requirements.txt
python examples/run_analysis_only.py --model-dir models/example_pretrained
```

**Done!** Check `models/example_pretrained/` for PDF validation plots.

This skips:
- ❌ Virtual environment setup (not needed for quick test)
- ❌ GPU configuration (CPU is fine for analysis)
- ❌ Model training (pre-trained models included)

**Result**: Publication-quality plots in ~30 seconds

---

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.7 or higher (3.9+ recommended)
- **RAM**: 8 GB
- **Storage**: 1 GB free space
- **Processor**: Any modern CPU

### Recommended Requirements
- **OS**: Linux or macOS
- **Python**: 3.10+
- **RAM**: 16 GB
- **Storage**: 5 GB (for multiple experiments)
- **GPU**: NVIDIA GPU with CUDA support (optional but significantly faster)

---

## Installation

### Step 1: Verify Python Installation

```bash
python3 --version
# Should show Python 3.7 or higher
```

If Python is not installed:
- **macOS**: `brew install python3`
- **Ubuntu/Debian**: `sudo apt-get install python3 python3-pip`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

### Step 2: Create Virtual Environment (Recommended)

A virtual environment keeps dependencies isolated:

```bash
# Navigate to the dupire_clean directory
cd dupire_clean

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Your prompt should now show (venv)
```

**Note**: You'll need to activate the virtual environment each time you open a new terminal.

### Step 3: Install Dependencies

```bash
# Ensure you're in dupire_clean with venv activated
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

This will install:
- TensorFlow (neural network framework)
- NumPy (numerical computations)
- Matplotlib (plotting)
- SciPy (scientific computing)
- scikit-learn (machine learning tools)

**Installation time**: ~2-5 minutes depending on internet speed

### Step 4: Verify Installation

```bash
# Test TensorFlow installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# Check GPU availability (optional)
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Expected output**:
```
TensorFlow version: 2.15.0 (or similar)
GPUs: [] (empty if no GPU, or list of GPU devices if available)
```

---

## Quick Start Test

Verify everything works with a quick test:

```bash
# Should take ~2 minutes on CPU, ~30 seconds on GPU
python examples/run_quick_test.py
```

**Expected output**:
```
================================================================================
QUICK TEST - DUPIRE LOCAL VOLATILITY PIPELINE
================================================================================
...
[Training progress messages]
...
✓ QUICK TEST COMPLETE!
================================================================================
```

**Output files** will be in a directory like:
```
models/runs/synthetic_data_3resblock_20250129_143022/
```

Check that directory for:
- `NN_phi_final.keras` - Trained model
- `NN_eta_final.keras` - Trained model
- `pdf_analysis_*.png` - Validation plots

---

## GPU Setup (Optional but Recommended)

### For NVIDIA GPUs

**1. Check GPU compatibility**:
```bash
nvidia-smi
```

If this command works, you have NVIDIA drivers installed.

**2. Install CUDA and cuDNN** (if not already installed):

Visit: https://www.tensorflow.org/install/pip#linux

Follow TensorFlow's official GPU setup guide for your OS.

**3. Verify GPU is detected**:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Should show:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### For Apple Silicon (M1/M2/M3)

macOS uses **Metal** instead of CUDA:

```bash
# Install TensorFlow for Apple Silicon
pip install tensorflow-macos tensorflow-metal
```

Verify:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

---

## Directory Structure Overview

After installation, your directory should look like:

```
dupire_clean/
├── README.md                          # Main documentation
├── docs/                              # Supplementary guides
│   ├── SETUP.md                       # Installation walkthrough
│   ├── QUICK_START.md                 # Runbook for experiments
│   ├── PACKAGE_SUMMARY.md             # Supervisor summary
│   └── MATHEMATICAL_TREATMENT.md      # Mathematical derivations
├── requirements.txt                   # Python dependencies
│
├── config.py                          # Configuration module
├── dupire_pipeline.py                 # Main pipeline code
│
├── examples/
│   ├── run_quick_test.py             # Quick test script
│   ├── run_full_training.py          # Full training script
│   └── run_analysis_only.py          # Analysis-only script
│
├── models/
│   ├── README.md                      # Models directory documentation
│   ├── example_pretrained/            # Pre-trained example models
│   │   ├── NN_phi_final.keras
│   │   └── NN_eta_final.keras
│   └── runs/                          # Organized experiment outputs
│       └── ...                        # Timestamped/custom run folders
│
└── venv/                              # Virtual environment (if created)
```

---

## Running Your First Experiment

### Option 1: Quick Test (Recommended First)

```bash
python examples/run_quick_test.py
```

- **Time**: 2 minutes (CPU) or 30 seconds (GPU)
- **Purpose**: Verify installation, test workflow
- **Output**: Basic results (not publication-quality)

### Option 2: Full Training

```bash
python examples/run_full_training.py
```

- **Time**: 30-60 minutes (CPU) or 10-15 minutes (GPU)
- **Purpose**: Production-quality results
- **Output**: Publication-ready plots and models

### Option 3: Analyze Pre-trained Models

```bash
python examples/run_analysis_only.py --model-dir models/example_pretrained
```

- **Time**: 1-2 minutes
- **Purpose**: Test analysis pipeline, learn outputs
- **Output**: PDF validation plots

---

## Customizing Parameters

### Method 1: Python Script

Create a file `my_experiment.py`:

```python
from config import DupirePipelineConfig, VolatilityConfig
from dupire_pipeline import DupirePipeline

# Create custom configuration
config = DupirePipelineConfig(
    # Volatility model
    volatility_config=VolatilityConfig.dupire_exact(sigma_base=0.25),

    # Training parameters
    num_epochs=10000,
    lambda_pde=2.0,
    num_res_blocks=4,

    # Output
    output_dir='models/runs/my_experiment'
)

# Run pipeline
pipeline = DupirePipeline(config)
pipeline.run()
```

Run with:
```bash
python my_experiment.py
```

### Method 2: Command Line

```bash
python dupire_pipeline.py \
    --num-epochs 10000 \
    --ldup 2.0 \
    --num-res-blocks 4 \
    --output-dir models/runs/my_experiment
```

See `python dupire_pipeline.py --help` for all options.

---

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError: No module named 'config'

**Problem**: Python can't find the config module

**Solution**:
```bash
# Make sure you're in the dupire_clean directory
cd dupire_clean

# Run from this directory
python examples/run_quick_test.py
```

### Issue 2: ImportError: DLL load failed (Windows)

**Problem**: Missing C++ redistributables

**Solution**:
Download and install:
- Microsoft Visual C++ Redistributable
- Visit: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Issue 3: Out of memory errors

**Problem**: Not enough RAM for training

**Solution**:
```python
# Reduce batch size or number of paths
config = DupirePipelineConfig(
    M_train=5000,  # Reduce from 10,000
    analysis_config=AnalysisConfig(n_paths_analysis=10000)  # Reduce from 25,000
)
```

### Issue 4: Training is very slow

**Problem**: Running on CPU instead of GPU

**Solution**:
- Verify GPU setup (see GPU Setup section above)
- Or reduce epochs for testing: `--num-epochs 1000`

### Issue 5: NaN losses during training

**Problem**: Learning rate too high or numerical instability

**Solution**:
```bash
# Reduce learning rate
python dupire_pipeline.py --lr 1e-5

# Reduce PDE weight
python dupire_pipeline.py --ldup 0.5
```

### Issue 6: Plots not generated

**Problem**: matplotlib backend issue

**Solution**:
```bash
# On headless servers, use Agg backend
export MPLBACKEND=Agg
python examples/run_full_training.py
```

---

## Deactivating Virtual Environment

When you're done:

```bash
deactivate
```

To reactivate later:

```bash
cd dupire_clean
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

---

## Uninstallation

To completely remove the installation:

```bash
# Deactivate virtual environment if active
deactivate

# Remove virtual environment
rm -rf venv

# Optionally remove downloaded dependencies cache
pip cache purge
```

To keep the code but remove generated outputs:

```bash
# Remove all experiment outputs
rm -rf models/runs/*/
```

---

## Getting Help

### Documentation

1. **README.md** - Overview and quick start
2. **docs/SETUP.md** - Installation and environment troubleshooting
3. **docs/MATHEMATICAL_TREATMENT.md** - Complete mathematical theory
4. **config.py** - Detailed parameter descriptions (run `python config.py`)
5. **models/README.md** - Working with trained models

### Command-line Help

```bash
python dupire_pipeline.py --help
python examples/run_quick_test.py --help
python examples/run_full_training.py --help
python examples/run_analysis_only.py --help
```

### Testing Configuration

```bash
# Print example configurations
python config.py
```

### Troubleshooting Checklist

- [ ] Python 3.7+ installed: `python3 --version`
- [ ] Virtual environment activated: prompt shows `(venv)`
- [ ] Dependencies installed: `pip list | grep tensorflow`
- [ ] In correct directory: `ls` should show `config.py` and `dupire_pipeline.py`
- [ ] No import errors: `python -c "import config; print('OK')"`

---

## Next Steps

After successful installation:

1. **Run quick test**: `python examples/run_quick_test.py`
2. **Review outputs**: Check the generated folder under `models/runs/`
3. **Read mathematical treatment**: Open `docs/MATHEMATICAL_TREATMENT.md`
4. **Customize parameters**: Edit a script or use command-line options
5. **Run full training**: `python examples/run_full_training.py`

---

## Support

For issues, questions, or suggestions:
- Check this SETUP.md file
- Review README.md
- Inspect error messages carefully
- Verify all steps in "Troubleshooting Checklist"

---

**You're all set! Happy modeling!** 🚀
