# Local Volatility Modeling with Neural Networks

This repository contains implementations for modeling local volatility using neural networks. The approach leverages physics-based loss functions and Monte Carlo methods to provide accurate, arbitrage-free option prices. It includes scripts for training neural networks on DAX call options, SPX put options, and general volatility modeling using synthetic data.

## Repository Structure

### 1. `DAX_Call_Option_Pricing`

This folder contains scripts and resources for training a neural network to price call options on the DAX index. The model applies a physics-based loss function, including consistency with the Dupire equation, to maintain accuracy and adhere to financial constraints.

#### Features

- **Physics-Based Loss Functions**: Includes `loss_phi_cal` and `loss_dupire_cal`, ensuring adherence to financial modeling constraints.
- **Monte Carlo Estimation**: Uses Monte Carlo methods for computing option prices based on maturity and strike ranges for repricing.
- **Configurable Parameters**: Allows setting ranges for maturities and strike prices, providing flexible control for put option pricing.
- **Automatic Differentiation**: Utilizes TensorFlow’s automatic differentiation to backpropagate through the custom loss functions.

#### Usage

```bash
cd DAX_Call_Option_Pricing
python tf_NN_call_DAX.py
```

Configure parameters for maturity (`t_min`, `t_max`) and strike prices (`k_min`, `k_max`) directly in the script. The output will display training progress and the predicted option prices.

### 2. `SPX_Put_Option_Pricing`

This folder includes resources for modeling and pricing put options on the SPX index. Using a similar structure to the DAX model, this model incorporates custom loss functions and Monte Carlo simulations to predict put option prices reliably.

#### Features

- **Physics-Based Loss Functions**: Implements `loss_phi_cal` and `loss_dupire_cal` with lambda-weighted training steps.
- **Monte Carlo Estimation**: Uses Monte Carlo methods for computing option prices based on maturity and strike ranges for repricing.
- **Configurable Parameters**: Allows setting ranges for maturities and strike prices, providing flexible control for put option pricing.
- **Automatic Differentiation**: Utilizes TensorFlow’s automatic differentiation to backpropagate through the custom loss functions.

#### Usage

```bash
cd SPX_Put_Option_Pricing
python tf_NN_put_SPX.py
```

Key parameters, including `t_min`, `t_max`, `k_min`, and `k_max`, can be adjusted within the script for optimal configuration.

### 3. Synthetic Data

This folder provides the resources for training a generalized local volatility model, designed to predict volatility surfaces based on strike and maturity inputs modeled with synthetic data obtained from Monte Carlo. The model employs neural networks with residual blocks and physics-informed loss functions to refine volatility predictions.

#### Features

- **Neural Network with Residual Blocks**: Implements a custom neural network with residual blocks for improved training stability and accuracy.
- **Physics-Informed Loss Functions**: Applies `loss_dupire` to ensure consistency with local volatility constraints.
- **Monte Carlo Estimation**: Uses Monte Carlo simulation to validate the accuracy of predicted volatilities.
- **Cosine Annealing Scheduler**: Adjusts learning rate during training for efficient convergence.

#### Usage

```bash
cd LocalVolatility_Model
python local_vol_model.py
```

Set parameters for maturity, strike, and volatility constraints in the script to customize model predictions. The output provides real-time tracking of training metrics and loss values.

### 4. `Synthetic_Data_Tensorflow_Advanced`

This folder contains a sophisticated, production-ready implementation of the Dupire local volatility pipeline using deep neural networks. This is an advanced version that provides a complete end-to-end pipeline with comprehensive analysis tools, analytical validation, and extensive documentation.

#### Features

- **Complete Pipeline**: Modular architecture with data generation, model training, and PDF analysis stages
- **Physics-Informed Neural Networks**: Neural networks that satisfy Dupire's PDE constraints
- **Analytical Validation**: Comparison tools against Black-Scholes analytical solutions
- **Comprehensive PDF Analysis**: Three-panel validation (strike space, log space, Gaussian space)
- **Flexible Configuration**: Extensive configuration system supporting multiple volatility models
- **Example Scripts**: Ready-to-use examples for common use cases
- **Pre-trained Models**: Included models for quick validation without training

#### Quick Start

**Instant Analysis (No Training - 30 seconds):**

Analyze pre-trained models immediately:

```bash
cd Synthetic_Data_Tensorflow_Advanced
pip install -r requirements.txt
python examples/run_analysis_only.py --model-dir models/example_pretrained
```

**Training:**

```bash
# Quick test (100 epochs, ~2 minutes)
python examples/run_quick_test.py

# Full training (30,000 epochs, ~30-60 minutes)
python examples/run_full_training.py
```

#### What This Code Does

The pipeline implements a complete workflow for Dupire local volatility calibration:

1. **Data Generation**: Monte Carlo simulation with known local volatility
2. **Model Training**: Neural network calibration using Dupire PDE
3. **PDF Analysis**: Validation through probability density function comparison

The implementation uses two neural networks:
- **NN_phi**: Learns option prices C(T,K)
- **NN_eta**: Learns local volatility squared σ²(T,K)

These networks are trained to satisfy Dupire's PDE: ∂C/∂T = (1/2)K²σ²(K,T) ∂²C/∂K²

#### Documentation

For comprehensive documentation, examples, configuration options, and mathematical details, see [Synthetic_Data_Tensorflow_Advanced/README.md](Synthetic_Data_Tensorflow_Advanced/README.md).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ameirtheshaa/LocalVolatility.git
   cd LocalVolatility
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## References

This repository builds upon foundational work in local volatility modeling and stochastic finance. We acknowledge the following key references:

1. **Dupire, B. (1994)**. "Pricing with a smile." *Risk Magazine*, 7(1), 18-20.
   - Original formulation of the local volatility model and Dupire's partial differential equation

2. **Privault, N. (2022)**. "Introduction to Stochastic Finance" (2nd ed).
   - Comprehensive mathematical background on stochastic processes and financial mathematics

3. **Wang, Z., et al. (2025)**. "Deep self-consistent learning of local volatility."
   - Neural network approach to Dupire local volatility calibration using physics-informed deep learning

### Citation

If you use this code in your research, please cite:

```bibtex
@software{local_volatility_neural_networks,
  title={Local Volatility Modeling with Neural Networks},
  author={Wang, Zhe and Shaa, Ameir and Privault, Nicolas and Guet, Claude},
  year={2025},
  url={https://github.com/ameirtheshaa/LocalVolatility}
}
```

## Contributing

Contributions are welcome! Feel free to fork the repository, submit issues, or create pull requests to improve the functionality or accuracy of the models.

## License

This project is licensed under the MIT License.