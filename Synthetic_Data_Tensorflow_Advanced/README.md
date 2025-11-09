# Dupire Local Volatility Neural Network Pipeline

**A professional implementation of Dupire's local volatility model using deep neural networks for option pricing and calibration.**

---

## Overview

This package implements a complete pipeline for:

1. **Data Generation**: Monte Carlo simulation with known local volatility
2. **Model Training**: Neural network calibration using Dupire PDE
3. **PDF Analysis**: Validation through probability density function comparison

### Key Features

- **Physics-informed neural networks** that satisfy Dupire's PDE
- **Flexible volatility models** (Dupire exact synthetic model, or custom)
- **Comprehensive PDF analysis** with Monte Carlo validation
- **Publication-quality plots** for research presentations
- **Modular architecture** with easy configuration

---

## ⚡ Instant Analysis (No Training - 30 seconds)

**Want to see results immediately?** Analyze the included pre-trained models without any training:

```bash
# Step 1: Install dependencies (one time only)
cd Synthetic_Data_Tensorflow_Advanced
pip install -r requirements.txt

# Step 2: Run analysis on pre-trained models
python examples/run_analysis_only.py --model-dir models/example_pretrained
```

**That's it!** In ~30 seconds, you'll have publication-quality PDF validation plots.

### What You Get

The analysis generates plots in `models/example_pretrained/`:

- **pdf_analysis_T_0.50.png** - Probability density at T=0.5 years
- **pdf_analysis_T_1.00.png** - Probability density at T=1.0 years
- **pdf_analysis_T_1.50.png** - Probability density at T=1.5 years

Each plot contains **three panels** showing the same distribution in different coordinate systems:

**Panel 1 - Strike Space (K):**
- Shows the probability density of terminal stock price in original units
- **X-axis**: Strike price K (e.g., $500 to $3000)
- **Y-axis**: Probability density f(K)
- **Blue histogram**: Monte Carlo simulation using NN-predicted volatility
- **Red curve**: Density extracted from NN option prices using f(K) = e^(rT) × ∂²C/∂K²
- **What to look for**: Agreement between histogram and curve validates the NN

**Panel 2 - Log-Space (ln K):**
- Shows the distribution of log-returns: ln(S_T / S_0)
- **X-axis**: Natural log of strike, ln(K)
- **Y-axis**: Density f(ln K), with Jacobian correction applied: f(ln K) = f(K) × K
- **Purpose**: Tests if K follows a log-normal distribution (should appear Gaussian here)
- **What to look for**: If the distribution looks like a bell curve (Gaussian), then K is log-normal

**Panel 3 - Gaussian Standardized Space:**
- Shows the standardized distribution: x = (ln K - μ) / σ
- **X-axis**: Standardized variable x (dimensionless)
- **Y-axis**: Density g(x), should match standard normal N(0,1) if perfectly log-normal
- **Dotted black line**: Perfect standard normal for comparison
- **Statistics shown**: Skewness and excess kurtosis (both should be ≈ 0 for log-normal)
- **Correlation coefficient**: Measures agreement between MC and NN
- **What to look for**: How closely the distribution matches the dotted normal curve

**Interpretation:**
- Good agreement in Panel 1 → NN learned correct option prices
- Gaussian shape in Panel 2 → Distribution is approximately log-normal
- Close match in Panel 3 → Distribution is nearly perfectly log-normal

### Understanding the PDF Plots

**Why three panels?**

The three panels show a progression from raw data to theoretical fit:

1. **Panel 1** answers: "Does the neural network reproduce the correct probability distribution?"
2. **Panel 2** answers: "Is the distribution approximately log-normal?" (standard assumption in finance)
3. **Panel 3** answers: "How close is it to a perfect log-normal distribution?"

**Mathematical Note:**

When transforming probability densities between coordinate systems, we must apply a Jacobian correction:
- From K to ln(K): multiply by K (Panel 2)
- From ln(K) to standardized x: multiply by σ×K (Panel 3)

This ensures the density integrates to 1 in each coordinate system. The code handles these transformations automatically.

**What indicates good results?**
- ✅ Histogram and curve overlap in all three panels
- ✅ Panel 2 looks roughly Gaussian (bell-shaped)
- ✅ Panel 3 closely matches the dotted normal curve
- ✅ Skewness ≈ 0, excess kurtosis ≈ 0 (shown in Panel 3 legend)
- ✅ High correlation (> 0.95) between MC and NN

---

## Quick Start (Training)

### Installation

```bash
# Clone or navigate to the Synthetic_Data_Tensorflow_Advanced directory
cd Synthetic_Data_Tensorflow_Advanced

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Training

```bash
# Quick test (100 epochs, ~2 minutes)
python examples/run_quick_test.py

# Full training (30,000 epochs, ~30-60 minutes)
python examples/run_full_training.py
```

---

## What This Code Does

### The Mathematical Problem

The **Dupire local volatility model** describes how asset prices evolve:

```
dS_t = r S_t dt + σ(t, S_t) S_t dW_t
```

where:
- `S_t`: Stock price at time t
- `r`: Risk-free interest rate
- `σ(t, S)`: **Local volatility function** (time and price dependent)
- `dW_t`: Brownian motion increment

**Goal**: Given option prices C(K,T), recover the local volatility surface σ(t,K)

### The Neural Network Approach

Instead of traditional finite difference methods, we use **two neural networks**:

1. **NN_phi**: Learns option prices C(T,K)
2. **NN_eta**: Learns local volatility squared σ²(T,K)

These networks are trained to satisfy:
- **Dupire's PDE**: ∂C/∂T = (1/2)K²σ²(K,T) ∂²C/∂K²
- **Boundary conditions**: C(K,0) = (S₀ - K)⁺
- **Arbitrage-free constraints**: ∂C/∂T ≥ 0, ∂²C/∂K² ≥ 0

### The Three-Stage Pipeline

#### Stage 1: Data Generation
- Run Monte Carlo simulation with **exact local volatility** σ(t,x) = 0.3 + y·exp(-y)
- Generate training data: (T, K, C_exact) tuples
- Save for reproducibility

#### Stage 2: Model Training
- Train neural networks using:
  - **Data loss**: Fit to observed option prices
  - **PDE loss**: Satisfy Dupire equation
  - **Regularization**: Prevent arbitrage
- Save trained models

#### Stage 3: PDF Analysis
- Extract **risk-neutral density**: f(K) = e^(rT) ∂²C/∂K²
- Run Monte Carlo with **NN-predicted volatility**
- Compare distributions in three spaces:
  - Strike space (K)
  - Log space (ln K)
  - Standardized Gaussian space

---

## Configuration

All parameters are controlled through [config.py](config.py). Here are the most important settings:

### Basic Parameters

```python
from config import DupirePipelineConfig, VolatilityConfig

config = DupirePipelineConfig(
    # Market parameters
    S0=1000.0,           # Initial stock price
    r=0.04,              # Risk-free rate (4%)
    K_min=500.0,         # Minimum strike
    K_max=3000.0,        # Maximum strike
    T_max=1.5,           # Maximum maturity (years)

    # Training parameters
    num_epochs=30000,    # Number of training iterations
    M_train=10000,       # Monte Carlo paths for training data

    # Neural network architecture
    num_res_blocks=3,    # Number of residual blocks
    lambda_pde=1.0,      # Weight for PDE loss
)
```

### Volatility Models

**Option 1: Dupire Exact (Default - Synthetic Model)**

```python
config.volatility_config = VolatilityConfig.dupire_exact()
# σ(t,x) = 0.3 + y·exp(-y) where y = (t + 0.1)·√(x + 0.1)
```

This is the standard model for synthetic data generation and testing. It has known properties:
- Smooth, arbitrage-free surface
- Analytically tractable for validation
- Non-constant volatility (captures local volatility effects)

**Option 2: Custom Volatility (Advanced)**

```python
def my_volatility(t, x):
    """Custom local volatility function"""
    return 0.2 + 0.1 * np.exp(-t) * (x - 1.0)**2

config.volatility_config = VolatilityConfig.custom(my_volatility)
```

Use this for your own volatility models or real market data calibration.

### Running Modes

```python
# Run all stages
config.mode = 'all'          # Generate → Train → Analyze

# Run individual stages
config.mode = 'generate'     # Only generate training data
config.mode = 'train'        # Only train models
config.mode = 'analyze'      # Only analyze existing models
```

### Preset Configurations

```python
# Quick test (100 epochs, reduced data)
config = DupirePipelineConfig.quick_test()

# Full training (30,000 epochs, production settings)
config = DupirePipelineConfig.full_training()

# Analysis only (load existing models)
config = DupirePipelineConfig.analysis_only(model_dir='path/to/models')
```

---

## Command-Line Interface

You can also run the pipeline from the command line:

```bash
# Quick test
python dupire_pipeline.py --preset quick

# Full training
python dupire_pipeline.py --preset full

# Custom parameters
python dupire_pipeline.py \
    --num-epochs 5000 \
    --ldup 2.0 \
    --num-res-blocks 4 \
    --lr 1e-4

# Analysis only
python dupire_pipeline.py \
    --mode analyze \
    --output-dir path/to/trained/models

# Disable plots (faster)
python dupire_pipeline.py \
    --preset quick \
    --no-training-plots \
    --no-pdf-plots
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--preset quick` | Quick test (100 epochs) | - |
| `--preset full` | Full training (30k epochs) | - |
| `--mode {all,generate,train,analyze}` | Pipeline stage | `all` |
| `--num-epochs N` | Training epochs | 30000 |
| `--ldup LAMBDA` | PDE loss weight | 1.0 |
| `--num-res-blocks N` | Residual blocks | 3 |
| `--lr RATE` | Learning rate | 1e-4 |
| `--M-train N` | MC paths | 10000 |
| `--output-dir DIR` | Output directory | auto |
| `--no-training-plots` | Disable training plots | - |
| `--no-pdf-plots` | Disable PDF plots | - |

---

## Output Files

After running the pipeline, you'll find:

```
output_directory/
├── metadata.json              # Configuration and scaling parameters
├── training_data.npz          # Monte Carlo data (T, K, C)
├── NN_phi_final.keras         # Trained option price network
├── NN_eta_final.keras         # Trained volatility network
├── losses_*.png               # Training loss curves
└── pdf_analysis_*.png         # PDF comparison plots
```

### Output Plots

**Training Plots**:
- Loss curves (data loss, PDE loss, regularization)
- Volatility error over epochs

**PDF Analysis Plots** (3-panel for each maturity):
1. **Strike Space**: Histogram vs NN model density
2. **Log Space**: Log-returns distribution
3. **Gaussian Space**: Standardized distribution with statistics

---

## Examples

### Example 1: Quick Test

```python
from config import DupirePipelineConfig
from dupire_pipeline import DupirePipeline

# Run a quick test (100 epochs, ~2 minutes)
config = DupirePipelineConfig.quick_test()
pipeline = DupirePipeline(config)
pipeline.run()
```

### Example 2: Custom Volatility Surface

```python
import numpy as np
from config import DupirePipelineConfig, VolatilityConfig
from dupire_pipeline import DupirePipeline

# Define custom volatility
def heston_like_vol(t, x):
    """Heston-inspired local volatility"""
    kappa = 2.0    # Mean reversion
    theta = 0.04   # Long-term variance
    v0 = 0.04      # Initial variance

    # Mean-reverting variance
    v_t = theta + (v0 - theta) * np.exp(-kappa * t)

    # Leverage effect (volatility increases when price drops)
    leverage = 1.0 + 0.5 * (1.0 - x)

    return np.sqrt(v_t) * leverage

# Configure pipeline
config = DupirePipelineConfig(
    volatility_config=VolatilityConfig.custom(heston_like_vol),
    num_epochs=10000,
    output_dir='heston_volatility'
)

# Run
pipeline = DupirePipeline(config)
pipeline.run()
```

### Example 3: Analyze Pre-trained Models

```python
from config import DupirePipelineConfig
from dupire_pipeline import DupirePipeline

# Load existing models and run analysis
config = DupirePipelineConfig.analysis_only(
    model_dir='path/to/trained/models'
)

# Override analysis settings
config.analysis_config.n_paths_analysis = 50000  # More paths
config.analysis_config.T_analysis = [0.5, 1.0, 1.5]  # Three maturities

pipeline = DupirePipeline(config)
pipeline.run()
```

### Example 4: Parameter Sweep

```python
from config import DupirePipelineConfig
from dupire_pipeline import DupirePipeline

# Test different PDE weights
for lambda_pde in [0.5, 1.0, 2.0, 5.0]:
    config = DupirePipelineConfig(
        num_epochs=5000,
        lambda_pde=lambda_pde,
        output_dir=f'sweep_lambda_{lambda_pde}'
    )

    pipeline = DupirePipeline(config)
    pipeline.run()

    print(f"\nCompleted lambda_pde = {lambda_pde}")
```

---

## Technical Details

### Neural Network Architecture

Both NN_phi and NN_eta use the same architecture:

```
Input (t_tilde, k_tilde)
    ↓
Gaussian Noise Layer
    ↓
Dense(64, tanh)
    ↓
Residual Block × N
    ↓
Dense(64, tanh)
    ↓
Dense(1, softplus)
    ↓
Output (φ_tilde or η_tilde)
```

**Residual Block**:
```
Input → Dense(64) → BatchNorm → Activation → Dense(64) → BatchNorm → Activation → Add(Input)
```

### Loss Function

```python
L_total = L_data + λ_PDE × L_PDE + λ_reg × L_arbitrage

where:
  L_data = MSE(φ_NN, φ_exact) + MSE(φ_NN(t=0), (S₀ - K)⁺)
  L_PDE = MSE(∂φ/∂t - η k² ∂²φ/∂k², 0)
  L_arbitrage = MSE(ReLU(−∂φ/∂t), 0)
```

### Coordinate Transformations

To improve numerical stability, we use scaled coordinates:

```
Original → Scaled:
  t_tilde = T / T_max
  k_tilde = exp(−rT) × K / K_max

Scaled → Original:
  T = t_tilde × T_max
  K = exp(rT) × k_tilde × K_max / S₀
```

---

## Hardware Requirements

### Minimum Requirements
- **CPU**: Any modern CPU (Intel i5 or equivalent)
- **RAM**: 8 GB
- **Storage**: 1 GB free space
- **Time**:
  - Quick test: ~2 minutes (CPU)
  - Full training: ~60 minutes (CPU) or ~15 minutes (GPU)

### Recommended Requirements
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 3060+)
- **RAM**: 16 GB
- **Storage**: 5 GB (for multiple experiments)
- **Time**:
  - Quick test: ~30 seconds (GPU)
  - Full training: ~10-15 minutes (GPU)

### GPU Support

TensorFlow will automatically detect and use your GPU if available:

```python
# Check GPU availability
import tensorflow as tf
print("GPUs available:", tf.config.list_physical_devices('GPU'))
```

If no GPU is detected, the code will automatically fall back to CPU.

---

## Troubleshooting

### Common Issues

**1. Import Error: No module named 'config'**

```bash
# Make sure you're in the Synthetic_Data_Tensorflow_Advanced directory
cd Synthetic_Data_Tensorflow_Advanced
python dupire_pipeline.py
```

**2. TensorFlow GPU not detected**

```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**3. Out of memory errors**

Reduce batch size or number of MC paths:

```python
config = DupirePipelineConfig(
    M_train=5000,  # Reduce from default 10000
    analysis_config=AnalysisConfig(n_paths_analysis=10000)  # Reduce from 25000
)
```

**4. Training is very slow**

- Enable GPU acceleration
- Reduce `num_epochs` for testing
- Use `--preset quick` for initial tests

**5. NaN losses during training**

- Reduce learning rate: `--lr 1e-5`
- Reduce PDE weight: `--ldup 0.5`
- Check data quality: ensure MC simulation completed successfully

---

## Project Structure

```
Synthetic_Data_Tensorflow_Advanced/
│
├── README.md                          # This file
├── docs/                              # Supplementary guides
│   ├── SETUP.md                       # Installation and troubleshooting
│   ├── QUICK_START.md                 # 30-second validation runbook
│   ├── PACKAGE_SUMMARY.md             # Supervisor-facing overview
│   └── MATHEMATICAL_TREATMENT.md      # Complete mathematical derivations
├── requirements.txt                   # Python dependencies
│
├── config.py                          # Configuration classes
├── dupire_pipeline.py                 # Main pipeline implementation
│
├── examples/
│   ├── run_quick_test.py             # Quick 100-epoch test
│   ├── run_full_training.py          # Full 30k-epoch training
│   └── run_analysis_only.py          # Analyze existing models
│
└── models/                            # Models and experiment outputs
    ├── README.md                      # Working with trained models
    ├── example_pretrained/            # Pre-trained reference models
    └── runs/                          # Organized experiment directories
```

---

## Mathematical Documentation

For a complete mathematical treatment including:
- Dupire PDE derivation
- Neural network loss function formulation
- PDF extraction methodology
- Coordinate transformations
- Arbitrage-free constraints

See **[MATHEMATICAL_TREATMENT.md](docs/MATHEMATICAL_TREATMENT.md)**

---

## References

### Key Papers

1. **Dupire, B. (1994)**. "Pricing with a smile." Risk Magazine, 7(1), 18-20.
   - Original formulation of local volatility model

2. **Wang, Z., et al. (2025)**. "Deep self-consistent learning of local volatility."
   - Neural network approach to Dupire calibration

3. **Privault, N. (2022)**. "Introduction to Stochastic Finance" (2nd ed).
   - Comprehensive mathematical background

### Additional Resources

- **Dupire PDE**: https://en.wikipedia.org/wiki/Local_volatility
- **TensorFlow Documentation**: https://www.tensorflow.org/guide
- **Physics-Informed Neural Networks**: Raissi et al. (2019), "Physics-informed neural networks"

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dupire_neural_pipeline,
  title={Dupire Local Volatility Neural Network Pipeline},
  author={Wang, Zhe and Shaa, Ameir and Privault, Nicolas and Guet, Claude},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: ameirshaa.akberali@ntu.edu.sg

---

## Acknowledgments

- Nicolas Privault and Claude Guet (Nanyang Technological University)
- National Research Foundation Singapore (CREATE/DesCartes program)
- Energy Research Institute @ NTU
- CNRS@CREATE (French National Centre for Scientific Research)
- TensorFlow and SciPy communities

---

**Happy modeling!** 🚀
