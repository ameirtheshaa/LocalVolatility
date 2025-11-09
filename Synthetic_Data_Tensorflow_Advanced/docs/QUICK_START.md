# Quick Start: Analyzing Pre-trained Models

**Get results in 30 seconds without training**

This guide walks through analyzing the included pre-trained neural network models to generate validation plots immediately.

---

## Purpose

This quick start demonstrates the Dupire local volatility pipeline by analyzing pre-trained models without requiring any training. This approach allows you to:

- Verify the installation
- Understand the output format
- Evaluate the validation methodology
- See publication-quality results immediately

For training new models, see [README.md](../README.md).

---

## Prerequisites

- **Python**: Version 3.7 or higher
- **Disk space**: Approximately 500 MB for dependencies
- **Time required**: 2-3 minutes total
- **Hardware**: Any CPU (no GPU required for analysis)

---

## Installation

Navigate to the package directory and install dependencies:

```bash
cd dupire_clean
pip install -r requirements.txt
```

**Expected duration**: 1-2 minutes

**Dependencies installed**:
- TensorFlow (neural network framework)
- NumPy (numerical computations)
- Matplotlib (visualization)
- SciPy (scientific computing)
- scikit-learn (density estimation)

---

## Running the Analysis

Execute the analysis script with the pre-trained models:

```bash
python examples/run_analysis_only.py --model-dir models/example_pretrained
```

**Expected duration**: 20-30 seconds

**What happens**:
1. Loads pre-trained neural networks (NN_phi and NN_eta)
2. Runs Monte Carlo simulation with NN-predicted local volatility
3. Extracts probability densities from option prices
4. Generates three-panel validation plots for each maturity
5. Saves plots to the model directory

---

## Output Files

After successful execution, the following files are generated in `models/example_pretrained/`:

### PDF Analysis Plots

- **pdf_analysis_T_0.50.png** - Validation at T = 0.5 years (6 months)
- **pdf_analysis_T_1.00.png** - Validation at T = 1.0 years (12 months)
- **pdf_analysis_T_1.50.png** - Validation at T = 1.5 years (18 months)

Each plot contains three panels showing the probability density in different coordinate systems.

---

## Understanding the Results

### Three-Panel Plot Structure

Each PDF analysis plot contains three panels:

**Panel 1: Strike Space (K)**
- Shows probability density in original strike price units
- Blue histogram: Monte Carlo samples
- Red curve: Neural network model prediction
- Validates that the NN correctly learned option prices

**Panel 2: Log-Space (ln K)**
- Shows distribution of log-returns
- Tests if prices follow a log-normal distribution
- Applies Jacobian correction: f(ln K) = f(K) × K
- Should appear approximately Gaussian

**Panel 3: Gaussian Standardized Space**
- Shows standardized distribution: x = (ln K - μ) / σ
- Compares against standard normal N(0,1) (dotted line)
- Displays skewness and excess kurtosis statistics
- Ultimate test of log-normality

For detailed interpretation, see the "Understanding the PDF Plots" section in [README.md](../README.md#understanding-the-pdf-plots).

---

## Verification

### Expected Results

Good validation is indicated by:

- **Panel 1**: Blue histogram and red curve closely overlap
- **Panel 2**: Distribution appears approximately Gaussian (symmetric, bell-shaped)
- **Panel 3**: Distribution closely follows the dotted N(0,1) reference line
- **Statistics**: Skewness ≈ 0, excess kurtosis ≈ 0
- **Correlation**: Value > 0.95 (shown in Panel 3 legend)

### Key Metrics

The pre-trained models should exhibit:
- Volatility RMSE < 1% (printed to console)
- High correlation between MC and NN densities (> 0.95)
- Minimal skewness and excess kurtosis in standardized space

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'tensorflow'`
- **Solution**: Ensure dependencies are installed: `pip install -r requirements.txt`

**Issue**: `FileNotFoundError: [Errno 2] No such file or directory: 'models/example_pretrained'`
- **Solution**: Verify you are in the `dupire_clean` directory
- **Check**: Run `ls models/example_pretrained/` to confirm models exist

**Issue**: Plots not generated or empty output directory
- **Solution**: Check console output for error messages
- **Verify**: Model files exist: `NN_phi_final.keras` and `NN_eta_final.keras`

**Issue**: Warning about GPU not found
- **Solution**: This is normal; analysis runs efficiently on CPU
- **Action**: No action required, execution will proceed

### Getting Help

For additional support:
- Detailed installation: See [SETUP.md](SETUP.md)
- Full documentation: See [README.md](../README.md)
- Mathematical background: See [MATHEMATICAL_TREATMENT.md](MATHEMATICAL_TREATMENT.md)
- Contact: ameirshaa.akberali@ntu.edu.sg

---

## Next Steps

### Train Your Own Models

To train new models with custom parameters:

```bash
# Quick test (100 epochs, ~2 minutes)
python examples/run_quick_test.py

# Full training (30,000 epochs, ~30-60 minutes)
python examples/run_full_training.py
```

See [README.md](../README.md) for comprehensive training options.

### Customize Parameters

Modify volatility models, training hyperparameters, or analysis settings:

```python
from config import DupirePipelineConfig, VolatilityConfig

# Customize configuration
config = DupirePipelineConfig(
    num_epochs=10000,
    lambda_pde=2.0,
    volatility_config=VolatilityConfig.dupire_exact(sigma_base=0.25)
)
```

See [config.py](config.py) for all available parameters.

### Explore the Mathematics

For complete mathematical derivations:
- Dupire PDE formulation
- Neural network loss functions
- PDF extraction methodology
- Coordinate transformations

See [MATHEMATICAL_TREATMENT.md](MATHEMATICAL_TREATMENT.md).

---

## Summary

You have successfully:
- Installed the Dupire pipeline dependencies
- Analyzed pre-trained neural network models
- Generated PDF validation plots
- Verified the results

The pre-trained models demonstrate that the neural network approach correctly learns local volatility and reproduces the expected probability distributions.

For further exploration, consult the comprehensive documentation in [README.md](../README.md) or the mathematical treatment in [MATHEMATICAL_TREATMENT.md](MATHEMATICAL_TREATMENT.md).

---

## Technical Details

### Pre-trained Model Specifications

The included models were trained with:
- **Architecture**: 3 residual blocks, 64 units per layer
- **Training epochs**: 30,000
- **Training data**: 10,000 Monte Carlo paths
- **Volatility model**: Dupire exact (σ = 0.3 + y·exp(-y))
- **Market parameters**: S₀ = 1000, r = 0.04

### Analysis Parameters

The analysis script uses:
- **Monte Carlo paths**: 25,000 (for validation)
- **Time step**: 10⁻³ years
- **Maturities analyzed**: T ∈ {0.5, 1.0, 1.5} years

### Computational Requirements

- **RAM**: ~2 GB during execution
- **Time**: ~30 seconds on modern CPU
- **Output size**: ~3-5 MB (three PNG files)

---

**Documentation Version**: 1.0
**Last Updated**: January 2025
**Authors**: Wang, Z., Shaa, A., Privault, N., & Guet, C.
