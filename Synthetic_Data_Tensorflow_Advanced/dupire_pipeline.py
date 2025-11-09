#!/usr/bin/env python3
"""
================================================================================
CONSOLIDATED DUPIRE LOCAL VOLATILITY PIPELINE
================================================================================

End-to-end pipeline for training and analyzing neural network models
for Dupire local volatility calibration.

STAGES:
    [1] DATA GENERATION - Monte Carlo simulation with exact local volatility
    [2] MODEL TRAINING - Neural network training using Dupire PDE
    [3] PDF ANALYSIS - Validation through probability density functions

ARCHITECTURE:
    Modular design with registry pattern for easy customization
    All plots and analyses can be independently enabled/disabled

USAGE:
    python consolidated_dupire_pipeline.py --mode all
    python consolidated_dupire_pipeline.py --preset quick
    python consolidated_dupire_pipeline.py --mode analyze --output-dir path/to/models

Author: Consolidated from tf_NN_call_MC.py and consolidated_synthetic_nn_analysis.py
Date: 2025
================================================================================

MATHEMATICAL SUMMARY (Pure Mathematics - No Neural Networks)
================================================================================

This script implements the Dupire local volatility model for option pricing.
The mathematical framework is based on the following principles:

1. ASSET PRICE DYNAMICS
   The underlying asset price S_t evolves according to a stochastic differential
   equation (SDE) with local volatility:

       dS_t = r S_t dt + σ(t, S_t) S_t dB_t,    S_0 given

   where:
   - S_t: asset price at time t
   - r: constant risk-free interest rate
   - σ(t, S): local volatility function (deterministic, time and price dependent)
   - B_t: standard Brownian motion

2. EUROPEAN OPTION PRICING
   European call and put option prices are given by risk-neutral expectations:

       C(K,T) = e^{-rT} E[(S_T - K)^+]  (call)
       P(K,T) = e^{-rT} E[(K - S_T)^+]  (put)

   where:
   - K: strike price
   - T: maturity
   - (x)^+ = max(x, 0)
   - E[·]: expectation under risk-neutral measure

3. DUPIRE'S PARTIAL DIFFERENTIAL EQUATION
   The option price π(K,T) satisfies Dupire's PDE:

       ∂π/∂T + rK ∂π/∂K - (1/2)K²σ²(K,T) ∂²π/∂K² = 0

   with initial and boundary conditions:

   Call options:
       π^c(K,0) = (S_0 - K)^+    (initial condition)
       π^c(∞,T) = 0              (boundary condition)

   Put options:
       π^p(K,0) = (K - S_0)^+    (initial condition)
       π^p(0,T) = 0              (boundary condition)

4. DUPIRE'S FORMULA
   Given observed option prices π(K,T), the local volatility can be recovered:

       σ²(K,T) = (2 ∂π/∂T + 2rK ∂π/∂K) / (K² ∂²π/∂K²)

   This provides the link between market option prices and local volatility.

5. COORDINATE TRANSFORMATION AND SCALING
   To ensure numerical stability, we apply the change of variables:

       k = e^{-rT} K / K_max     (scaled strike)
       t = T / T_max             (scaled maturity)
       η(k,t) = (T_max/2) σ²(K,T)   (scaled squared volatility)

   The Dupire PDE in scaled coordinates becomes:

       ∂π/∂t - η(k,t) k² ∂²π/∂k² = 0

6. ARBITRAGE-FREE CONDITIONS
   To ensure economic consistency, option prices must satisfy:

   (a) Calendar spread: ∂π/∂T ≥ 0
   (b) Butterfly spread: ∂²π/∂K² ≥ 0

   These conditions guarantee:
   - Options with longer maturities are more expensive
   - Risk-neutral density is non-negative

7. MONTE CARLO SIMULATION
   Asset paths are simulated using Euler-Maruyama discretization:

       S_{n+1} = S_n + r S_n Δt + σ(t_n, S_n) S_n ΔW_n

   where:
   - Δt: time step
   - ΔW_n ~ N(0, Δt): Brownian increment

   Option prices are estimated as sample averages:

       C(K,T) ≈ e^{-rT} (1/M) Σ_{i=1}^M (S_T^{(i)} - K)^+

8. RISK-NEUTRAL DENSITY EXTRACTION
   The probability density function f(K) of S_T is related to option prices:

       f(K) = e^{rT} ∂²C/∂K² (K,T)

   Properties:
   - f(K) ≥ 0 for all K
   - ∫_0^∞ f(K) dK = 1
   - E[S_T] = e^{rT} S_0 (risk-neutral drift)

9. LOG-NORMAL DISTRIBUTION ANALYSIS
   If ln(S_T) ~ N(μ, σ²), then S_T has a log-normal distribution.
   Moments:

       E[S_T] = e^{μ + σ²/2}
       Var[S_T] = e^{2μ + σ²}(e^{σ²} - 1)

   Parameter estimation from empirical density:

       mean_K = E[K]
       var_K = E[K²] - E[K]²

       σ² = ln(var_K/mean_K² + 1)
       μ = ln(mean_K) - σ²/2

10. COORDINATE TRANSFORMATIONS FOR DENSITY
    To analyze the density in Gaussian space:

    (a) Log-transform: y = ln(K)
        f_Y(y) = f_K(e^y) × e^y  (Jacobian transformation)

    (b) Standardization: x = (ln(K) - μ)/σ
        If K ~ LogNormal(μ,σ²), then x ~ N(0,1) (standard normal)

11. EXACT LOCAL VOLATILITY (SYNTHETIC DATA)
    For validation purposes, we use the exact formula:

        σ(t,x) = 0.3 + y e^{-y}

    where:
        y = (t + 0.1)√(x + 0.1)
        x = S/S_0

    This provides a known ground truth for testing the calibration.

12. STATISTICAL MEASURES
    To compare distributions:

    - Skewness: γ_1 = E[(X-μ)³]/σ³
    - Excess Kurtosis: γ_2 = E[(X-μ)⁴]/σ⁴ - 3
    - For standard normal: γ_1 = 0, γ_2 = 0

REFERENCES:
    - Dupire, B. (1994). "Pricing with a smile." Risk Magazine.
    - Privault, N. (2022). "Introduction to Stochastic Finance" (2nd ed).
    - Wang, Z. et al. (2025). "Deep self-consistent learning of local volatility."

================================================================================
"""

import os
import sys
import argparse
import datetime
import time
import json
import warnings
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import asdict

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import resource_variable_ops
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from scipy import interpolate
from scipy.stats import norm, skew, kurtosis, gaussian_kde
from sklearn.neighbors import KernelDensity

# Import configurations from separate config module
from config import (
    VolatilityConfig,
    PlotConfig,
    AnalysisConfig,
    DupirePipelineConfig
)

warnings.filterwarnings('ignore')

# =============================================================================
# [0] TENSORFLOW AND MATPLOTLIB SETUP
# =============================================================================

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Configure GPU/CPU
VariableSpec = resource_variable_ops.VariableSpec
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

if len(gpus) != 0:
    device = gpus[0]
    tf.config.set_visible_devices(device, 'GPU')
    tf.config.experimental.set_memory_growth(device, True)
    print(f'Running on GPU: {device.name}')
else:
    device = cpus[0]
    tf.config.set_visible_devices(device, 'CPU')
    print(f'Running on CPU: {device.name}')

# Set data types
data_type = tf.float32
data_type_nn = tf.float32
tf.keras.backend.set_floatx('float32')
tf.random.set_seed(42)

# Configure matplotlib for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Professional color palette
COLORS = {
    'mc_hist': '#4A90E2',      # Soft blue
    'model': '#E94B3C',        # Warm red
    'kde': '#50C878',          # Emerald green
    'normal': '#2C3E50',       # Dark blue-gray
    'text': '#34495E',         # Darker gray
    'background': '#FAFAFA',   # Light gray
}


# =============================================================================
# [1] UTILITY FUNCTIONS
# =============================================================================

def save_metadata(config: DupirePipelineConfig, output_dir: str):
    """Save configuration and scaling metadata to JSON"""
    # Convert config to dict, handling nested dataclasses
    config_dict = asdict(config)

    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': config_dict,
        'scaling': {
            'S0': config.S0,
            'r': config.r,
            't_max': config.T_max,
            'k_min': config.K_min,
            'k_max': config.K_max,
        },
        'volatility': {
            'model_type': config.volatility_config.model_type,
            'parameters': asdict(config.volatility_config)
        }
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)  # default=str handles non-serializable objects

    print(f"  Saved metadata: {metadata_path}")


def load_metadata(output_dir: str) -> Dict:
    """Load metadata from JSON"""
    metadata_path = os.path.join(output_dir, 'metadata.json')

    if not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


# =============================================================================
# [2] STAGE 1: DATA GENERATION
# =============================================================================

class DataGenerator:
    """
    Generates synthetic training data using Monte Carlo simulation

    The ground truth local volatility σ(t,x) is known, allowing validation.
    This class:
    1. Runs Monte Carlo with exact σ(t,x)
    2. Extracts (T, K, φ_exact) training samples
    3. Saves data for reproducibility

    Mathematical Details:
    ---------------------
    Exact local volatility: σ(t,x) = 0.3 + y·exp(-y)
    where y = (t + 0.1)·√(x + 0.1) and x = S/S₀

    SDE: dS_t = r·S_t·dt + σ(t,S_t/S₀)·S_t·dB_t
    """

    def __init__(self, config: DupirePipelineConfig):
        self.config = config
        self.S_matrix = None
        self.t_all = None

    def exact_sigma(self, t: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Ground truth local volatility: σ(t, x)

        Uses the volatility model specified in config.volatility_config

        Args:
            t: Time (can be scalar or tensor)
            x: Normalized stock price S/S₀

        Returns:
            Local volatility σ(t,x)
        """
        vol_config = self.config.volatility_config

        if vol_config.model_type == 'dupire_exact':
            # Model: σ = σ_base + y·exp(-y) where y = (t + t_shift)·√(x + x_shift)
            y = tf.sqrt(x + vol_config.x_shift) * (t + vol_config.t_shift)
            sigma = vol_config.sigma_base + y * tf.exp(-y)

        elif vol_config.model_type == 'smile_surface':
            # Model: σ = σ_base + time_factor·exp(-decay_rate·t) + smile_factor·(x - 1)²
            time_component = vol_config.time_factor * tf.exp(-vol_config.decay_rate * t)
            smile_component = vol_config.smile_factor * tf.square(x - 1.0)
            sigma = vol_config.sigma_base + time_component + smile_component
            sigma = tf.maximum(sigma, vol_config.min_volatility)

        elif vol_config.model_type == 'custom':
            # Use custom function (convert to numpy, apply function, convert back)
            t_np = t.numpy() if hasattr(t, 'numpy') else float(t)
            x_np = x.numpy() if hasattr(x, 'numpy') else x
            sigma_np = vol_config.custom_volatility_func(t_np, x_np)
            sigma = tf.constant(sigma_np, dtype=data_type)

        else:
            raise ValueError(f"Unknown volatility model type: {vol_config.model_type}")

        return sigma

    def run_mc(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Run Monte Carlo simulation with exact local volatility

        Simulates: dS_t = r·S_t·dt + σ(t,S_t/S₀)·S_t·dB_t

        Returns:
            (S_matrix, t_all): Stock price paths and time grid
            S_matrix shape: [N_t, M] where M = number of paths
        """
        print("\n  Generating Monte Carlo paths with exact local volatility...")

        M = self.config.M_train
        N_t = self.config.N_t_train
        dt = self.config.dt_train
        S0 = self.config.S0
        r = self.config.r

        # Time grid
        t_all = tf.cast(tf.reshape(np.linspace(0, N_t*dt, N_t), [-1,1]), dtype=data_type)

        # Initialize paths
        S_list = [tf.cast(tf.reshape(np.full(M, S0), [1,M]), dtype=data_type)]

        # Brownian increments
        np.random.seed(42)
        dW_list = tf.cast(
            tf.concat([np.random.normal(0, 1, size=[N_t,1]) * np.sqrt(dt) for _ in range(M)], axis=1),
            dtype=data_type
        )

        print(f"    Simulating {M:,} paths with {N_t} time steps (dt={dt})...")
        start_time = time.time()

        for i in range(N_t-1):
            t_now = t_all[i]
            S_now = S_list[-1]

            # Compute local volatility σ(t, S/S₀)
            sigma_local = self.exact_sigma(t_now, S_now / S0)

            # Euler-Maruyama step
            S_new = S_now + r * S_now * dt + sigma_local * S_now * dW_list[i]
            S_list.append(S_new)

        S_matrix = tf.concat(S_list, axis=0)

        elapsed = time.time() - start_time
        print(f"    ✓ MC completed in {elapsed:.2f}s")
        print(f"    Final S range: [{tf.reduce_min(S_matrix[-1]):.1f}, {tf.reduce_max(S_matrix[-1]):.1f}]")

        self.S_matrix = S_matrix
        self.t_all = t_all

        return S_matrix, t_all

    def get_training_data(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Extract (T, K, φ_exact) training samples from MC paths

        Creates a grid of (T,K) points and computes option prices via:
            φ(T,K) = e^{-rT} · E[(S_T - K)^+]

        Returns:
            (T_nn, K_nn, phi_ref): Maturities, strikes, option prices
        """
        if self.S_matrix is None:
            self.run_mc()

        print("\n  Extracting training data from MC paths...")

        N = self.config.N_maturities
        m = self.config.N_strikes
        N_t = self.config.N_t_train

        # Select N maturities from the MC time grid
        indices = tf.cast(tf.linspace(30, N_t-1, N), tf.int32)
        t_all_T = tf.gather(self.t_all, indices)
        S_T = tf.gather(self.S_matrix, indices, axis=0)

        # Create (T,K) grid
        T = tf.repeat(tf.reshape(t_all_T, [-1,1]), m, axis=1)
        K = tf.cast(
            tf.repeat(
                tf.reshape(np.linspace(self.config.K_min, self.config.K_max, m), [1,-1]),
                len(T),
                axis=0
            ),
            dtype=data_type
        )

        print(f"    Grid: {N} maturities × {m} strikes = {N*m} data points")

        # Compute exact option prices via MC expectation
        def exact_phi(T, K):
            """Compute option price: e^{-rT} · E[(S_T - K)^+]"""
            E_payoff = tf.concat([
                tf.reshape(
                    tf.reduce_mean(
                        tf.nn.relu(tf.expand_dims(S_T[i], axis=0) - tf.expand_dims(K[i], axis=1)),
                        axis=1
                    ),
                    [1,-1]
                )
                for i in range(N)
            ], axis=0)

            phi = tf.exp(-self.config.r * T) * E_payoff
            return phi

        phi_exact = exact_phi(T, K)

        # Reshape to vectors
        T_nn = tf.reshape(T, [-1,1])
        K_nn = tf.reshape(K, [-1,1])
        phi_ref = tf.reshape(phi_exact, [-1,1])

        print(f"    Price range: [{tf.reduce_min(phi_ref):.2f}, {tf.reduce_max(phi_ref):.2f}]")

        return T_nn, K_nn, phi_ref

    def save_training_data(self, output_dir: str):
        """Save training data to disk"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        T_nn, K_nn, phi_ref = self.get_training_data()

        data_path = os.path.join(output_dir, 'training_data.npz')
        np.savez(
            data_path,
            T=T_nn.numpy(),
            K=K_nn.numpy(),
            phi=phi_ref.numpy(),
            S_matrix=self.S_matrix.numpy(),
            t_all=self.t_all.numpy(),
        )

        print(f"  ✓ Saved training data: {data_path}")

    def load_training_data(self, output_dir: str) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Load training data from disk"""
        data_path = os.path.join(output_dir, 'training_data.npz')

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found: {data_path}")

        data = np.load(data_path)

        T_nn = tf.constant(data['T'], dtype=data_type)
        K_nn = tf.constant(data['K'], dtype=data_type)
        phi_ref = tf.constant(data['phi'], dtype=data_type)

        self.S_matrix = tf.constant(data['S_matrix'], dtype=data_type)
        self.t_all = tf.constant(data['t_all'], dtype=data_type)

        print(f"  ✓ Loaded training data: {data_path}")
        print(f"    {len(T_nn)} data points")

        return T_nn, K_nn, phi_ref

    def scale_data(self, T_nn: tf.Tensor, K_nn: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Scale data to normalized coordinates

        Transformations:
            t_tilde = T / T_max
            k_tilde = exp(-r·T) · K / K_max

        Args:
            T_nn: Maturities
            K_nn: Strike prices

        Returns:
            (t_tilde, k_tilde): Scaled coordinates
        """
        t_tilde = T_nn / self.config.T_max
        k_tilde = tf.exp(-self.config.r * T_nn) * K_nn / self.config.K_max

        return t_tilde, k_tilde


# =============================================================================
# [3] STAGE 2: MODEL TRAINING
# =============================================================================

class DupireNeuralModel(tf.keras.Model):
    """
    Neural network model for Dupire local volatility

    Networks:
        NN_phi_tilde: (t_tilde, k_tilde) → normalized option price
        NN_eta_tilde: (t_tilde, k_tilde) → normalized local volatility²

    Architecture:
        - Input: (t_tilde, k_tilde)
        - Gaussian noise layer
        - Dense layer (64 units, tanh)
        - N residual blocks (2 × Dense-BatchNorm-Activation with skip connection)
        - Dense layer (64 units, tanh)
        - Output layer (1 unit, softplus for positivity)

    Loss Functions:
        - loss_phi_cal(): Data fitting + boundary condition
        - loss_dupire_cal(): Dupire PDE residual + arbitrage penalty
    """

    def __init__(self, config: DupirePipelineConfig, data_generator: DataGenerator):
        super(DupireNeuralModel, self).__init__()
        self.config = config
        self.data = data_generator
        self.lambda_pde = config.lambda_pde
        self.lambda_reg = config.lambda_reg
        self.num_res_blocks = config.num_res_blocks
        self.activation = config.activation
        self.gaussian_phi = config.gaussian_noise_phi
        self.gaussian_eta = config.gaussian_noise_eta

        # Will be set by build_models()
        self.NN_phi_tilde = None
        self.NN_eta_tilde = None
        self.optimizer_NN_phi = None
        self.optimizer_NN_eta = None

    def residual_block(self, input_tensor, units=64, activation='tanh'):
        """
        Single residual block with two Dense-BatchNorm-Activation layers
        and a residual (skip) connection

        Architecture:
            input → Dense → BatchNorm → Activation → Dense → BatchNorm → Activation → Add(input) → output
        """
        dense_1 = tf.keras.layers.Dense(units, use_bias=False)(input_tensor)
        batchnorm_1 = tf.keras.layers.BatchNormalization()(dense_1)
        activation_1 = tf.keras.layers.Activation(activation)(batchnorm_1)

        dense_2 = tf.keras.layers.Dense(units, use_bias=False)(activation_1)
        batchnorm_2 = tf.keras.layers.BatchNormalization()(dense_2)
        activation_2 = tf.keras.layers.Activation(activation)(batchnorm_2)

        # Residual connection (skip connection)
        added = tf.keras.layers.Add()([input_tensor, activation_2])
        return added

    def net_phi_tilde(self, num_res_blocks=3, units=64, activation='tanh'):
        """
        Build neural network for normalized option price

        Maps (t_tilde, k_tilde) → φ_tilde ∈ R+
        """
        input_ = tf.keras.Input(shape=(2,))

        noisy_input = tf.keras.layers.GaussianNoise(self.gaussian_phi)(input_)
        dense_in = tf.keras.layers.Dense(units, activation=activation, use_bias=False)(noisy_input)

        # Apply residual blocks
        x = dense_in
        for _ in range(num_res_blocks):
            x = self.residual_block(x, units, activation)

        # Final dense layers
        dense_out = tf.keras.layers.Dense(units, activation=activation, use_bias=True)(x)
        output_ = tf.keras.layers.Dense(1, activation='softplus', use_bias=True, dtype="float32")(dense_out)

        model = tf.keras.models.Model(inputs=input_, outputs=output_)
        return model

    def net_eta_tilde(self, num_res_blocks=3, units=64, activation='tanh'):
        """
        Build neural network for normalized local volatility squared

        Maps (t_tilde, k_tilde) → η_tilde ∈ R+
        """
        input_ = tf.keras.Input(shape=(2,))

        noisy_input = tf.keras.layers.GaussianNoise(self.gaussian_eta)(input_)
        dense_in = tf.keras.layers.Dense(units, activation='tanh', use_bias=False)(noisy_input)

        # Apply residual blocks
        x = dense_in
        for _ in range(num_res_blocks):
            x = self.residual_block(x, units, activation)

        # Final dense layers
        dense_out = tf.keras.layers.Dense(units, activation=activation, use_bias=True)(x)
        output_ = tf.keras.layers.Dense(1, activation='softplus', use_bias=True, dtype="float32")(dense_out)

        model = tf.keras.models.Model(inputs=input_, outputs=output_)
        return model

    def build_models(self):
        """Build both neural networks and optimizers"""
        print("\n  Building neural networks...")

        self.NN_phi_tilde = self.net_phi_tilde(
            num_res_blocks=self.num_res_blocks,
            units=64,
            activation=self.activation
        )
        self.NN_eta_tilde = self.net_eta_tilde(
            num_res_blocks=self.num_res_blocks,
            units=64,
            activation=self.activation
        )

        self.optimizer_NN_phi = tf.keras.optimizers.Adam(learning_rate=self.config.lr_phi)
        self.optimizer_NN_eta = tf.keras.optimizers.Adam(learning_rate=self.config.lr_eta)

        phi_params = self.NN_phi_tilde.count_params()
        eta_params = self.NN_eta_tilde.count_params()

        print(f"    NN_phi_tilde: {phi_params:,} parameters")
        print(f"    NN_eta_tilde: {eta_params:,} parameters")
        print(f"    Total: {phi_params + eta_params:,} parameters")

    def neural_phi_tilde(self, t_tilde, k_tilde):
        """
        Compute normalized option price from NN

        φ_tilde = 1 - exp(-NN_φ)
        """
        phi_nn = self.NN_phi_tilde(tf.concat([t_tilde, k_tilde], axis=1))
        return (1 - tf.exp(-phi_nn))

    def neural_eta_tilde(self, t_tilde, k_tilde):
        """Compute normalized local volatility squared from NN"""
        eta_nn = self.NN_eta_tilde(tf.concat([t_tilde, k_tilde], axis=1))
        return eta_nn

    def exact_eta_tilde(self, t_tilde, k_tilde):
        """Compute exact normalized local volatility squared (for comparison)"""
        T = self.config.T_max * t_tilde
        K = tf.exp(self.config.r * T) * self.config.K_max * k_tilde / self.config.S0
        eta_tilde = 0.5 * self.config.T_max * self.data.exact_sigma(T, K)**2
        return eta_tilde

    def neural_phi(self, T, K):
        """Compute option price in original coordinates"""
        T_nn = tf.cast(tf.reshape(T, [-1,1]), dtype=data_type)
        K_nn = tf.cast(tf.reshape(K, [-1,1]), dtype=data_type)
        t_tilde, k_tilde = self.data.scale_data(T_nn, K_nn)
        phi_nn = self.config.S0 * self.neural_phi_tilde(t_tilde, k_tilde)
        return phi_nn

    def neural_sigma(self, T, K):
        """Compute local volatility in original coordinates"""
        T_nn = tf.cast(tf.reshape(T, [-1,1]), dtype=data_type)
        K_nn = tf.cast(tf.reshape(K, [-1,1]), dtype=data_type)
        t_tilde, k_tilde = self.data.scale_data(T_nn, K_nn)
        sigma_nn = tf.sqrt(2 * self.neural_eta_tilde(t_tilde, k_tilde) / self.config.T_max)
        return tf.squeeze(sigma_nn)

    def clip(self, y):
        """Adaptive clipping for weight function"""
        x = tf.stop_gradient(tf.reduce_mean(y**2) / y**2)
        return tf.clip_by_value(x, clip_value_min=0.1, clip_value_max=10)

    def weight(self, y):
        """Adaptive weight function to balance loss contributions"""
        return 1 + self.clip(y) / tf.reduce_mean(self.clip(y))

    def loss_phi_cal(self, t_tilde, k_tilde, phi_tilde_ref, k_min_random, k_max_random):
        """
        Data fitting loss + boundary condition loss

        L_phi = L_data + L_bc

        where:
        - L_data: Fit NN to observed option prices
        - L_bc: Enforce initial condition φ(k, 0) = (S₀ - K)^+
        """
        # Data fitting loss
        phi_tilde_nn = self.neural_phi_tilde(t_tilde, k_tilde)
        loss_phi = tf.reduce_mean(self.weight(phi_tilde_ref) * tf.square(phi_tilde_nn - phi_tilde_ref))

        # Boundary condition loss
        M1 = 128
        t_tilde_0 = tf.zeros([M1, 1], dtype=data_type)
        k_tilde_0 = tf.random.uniform(shape=[M1, 1], minval=k_min_random, maxval=k_max_random, dtype=data_type)
        phi_tilde_0 = tf.nn.relu(1 - (1/self.config.S0) * self.config.K_max * k_tilde_0)
        loss_bc = tf.reduce_mean(
            self.weight(phi_tilde_0) * tf.square(self.neural_phi_tilde(t_tilde_0, k_tilde_0) - phi_tilde_0)
        )

        return loss_phi + loss_bc

    def loss_dupire_cal(self, t_min_random, t_max_random, k_min_random, k_max_random):
        """
        Dupire PDE loss + arbitrage penalty

        L_dup = L_PDE + L_arb

        where:
        - L_PDE: Residual of Dupire's equation
        - L_arb: Penalty for arbitrage violations
        """
        M2 = 128

        # Sample collocation points
        t_tilde_0 = tf.fill([M2, 1], t_min_random)
        t_tilde_1 = tf.fill([M2, 1], t_max_random)
        k_tilde_0 = tf.fill([M2, 1], k_min_random)
        k_tilde_1 = tf.fill([M2, 1], k_max_random)
        t_tilde_bulk = tf.random.uniform(shape=[M2*M2, 1], minval=t_min_random, maxval=t_max_random, dtype=data_type)
        k_tilde_bulk = tf.random.uniform(shape=[M2*M2, 1], minval=k_min_random, maxval=k_max_random, dtype=data_type)

        t_tilde_random = tf.concat([t_tilde_0, t_tilde_1, t_tilde_bulk], axis=0)
        k_tilde_random = tf.concat([k_tilde_bulk, k_tilde_0, k_tilde_1], axis=0)

        # Compute gradients using automatic differentiation
        with tf.GradientTape(persistent=True) as tape_2:
            tape_2.watch(k_tilde_random)
            with tf.GradientTape(persistent=True) as tape_1:
                tape_1.watch(t_tilde_random)
                tape_1.watch(k_tilde_random)

                phi_tilde = self.neural_phi_tilde(t_tilde_random, k_tilde_random)

            grad_phi_t_tilde = tape_1.gradient(phi_tilde, t_tilde_random)
            grad_phi_k_tilde = tape_1.gradient(phi_tilde, k_tilde_random)

        grad_phi_kk_tilde = tape_2.gradient(grad_phi_k_tilde, k_tilde_random)

        # Dupire PDE residual
        eta_tilde = self.neural_eta_tilde(t_tilde_random, k_tilde_random)
        dupire_eqn = grad_phi_t_tilde - eta_tilde * k_tilde_random**2 * grad_phi_kk_tilde

        loss_dupire = tf.reduce_mean(self.weight(grad_phi_t_tilde) * tf.square(dupire_eqn))

        # Arbitrage penalty
        arb_eqn = grad_phi_t_tilde - self.config.r * self.config.T_max * k_tilde_random * tf.nn.relu(grad_phi_k_tilde)
        loss_reg = tf.reduce_mean(self.weight(grad_phi_t_tilde) * tf.square(tf.nn.relu(-arb_eqn)))

        return loss_dupire, loss_reg

    @tf.function
    def train_step(self, t_tilde, k_tilde, phi_tilde_ref,
                   t_min_random, t_max_random, k_min_random, k_max_random,
                   lambda_pde=None, lambda_reg=None):
        """
        Single training step

        Updates both NN_phi and NN_eta networks
        """
        if lambda_pde is None:
            lambda_pde = self.lambda_pde
        if lambda_reg is None:
            lambda_reg = self.lambda_reg

        with tf.GradientTape(persistent=True) as tape:
            loss_phi = self.loss_phi_cal(t_tilde, k_tilde, phi_tilde_ref, k_min_random, k_max_random)
            loss_dupire, loss_reg = self.loss_dupire_cal(t_min_random, t_max_random, k_min_random, k_max_random)
            loss_total = loss_phi + lambda_pde * loss_dupire + lambda_reg * loss_reg

        grads_NN_phi = tape.gradient(loss_total, self.NN_phi_tilde.trainable_variables)
        grads_NN_eta = tape.gradient(loss_dupire, self.NN_eta_tilde.trainable_variables)

        self.optimizer_NN_phi.apply_gradients(zip(grads_NN_phi, self.NN_phi_tilde.trainable_variables))
        self.optimizer_NN_eta.apply_gradients(zip(grads_NN_eta, self.NN_eta_tilde.trainable_variables))

        return loss_phi, loss_dupire, loss_reg


class ModelTrainer:
    """
    Handles training loop with checkpointing and metrics tracking

    Methods:
        train(): Main training loop with LR scheduling
        save_checkpoint(): Save model state
        compute_metrics(): Track RMSE and relative error
    """

    def __init__(self, model: DupireNeuralModel, config: DupirePipelineConfig):
        self.model = model
        self.config = config

    def train(self, T_nn, K_nn, phi_ref, t_tilde, k_tilde, phi_tilde_ref,
              t_min_random, t_max_random, k_min_random, k_max_random,
              output_dir: str, visualizer=None):
        """
        Main training loop

        Args:
            T_nn, K_nn, phi_ref: Training data in original coordinates
            t_tilde, k_tilde, phi_tilde_ref: Training data in scaled coordinates
            t_min_random, t_max_random, k_min_random, k_max_random: Sampling bounds
            output_dir: Directory to save checkpoints
            visualizer: Optional TrainingVisualizer for plotting

        Returns:
            (rmse_sigma_list, error_sigma_list): Training metrics
        """
        print(f"\n{'='*80}")
        print("TRAINING NEURAL NETWORKS")
        print(f"{'='*80}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Lambda PDE: {self.config.lambda_pde}")
        print(f"Lambda Reg: {self.config.lambda_reg}")
        print(f"Learning rate (phi): {self.config.lr_phi}")
        print(f"Learning rate (eta): {self.config.lr_eta}")

        learning_rate = self.config.lr_phi
        self.model.optimizer_NN_phi.learning_rate.assign(learning_rate)
        self.model.optimizer_NN_eta.learning_rate.assign(learning_rate / 10)

        loss_phi_list = []
        loss_dupire_list = []
        loss_reg_list = []
        error_sigma_list = []
        rmse_sigma_list = []

        lambda_pde = tf.constant(self.config.lambda_pde, dtype=data_type)
        lambda_reg = tf.constant(self.config.lambda_reg, dtype=data_type)

        start_time = time.time()

        for iter_ in range(self.config.num_epochs + 1):
            # Training step
            loss_phi, loss_dupire, loss_reg = self.model.train_step(
                t_tilde, k_tilde, phi_tilde_ref,
                t_min_random, t_max_random, k_min_random, k_max_random,
                lambda_pde, lambda_reg
            )

            loss_phi_list.append(loss_phi)
            loss_dupire_list.append(loss_dupire)
            loss_reg_list.append(loss_reg)

            # Compute relative error of local volatility
            sigma_exact = tf.sqrt(2 * self.model.exact_eta_tilde(t_tilde, k_tilde) / self.config.T_max)
            sigma_nn = tf.sqrt(2 * self.model.neural_eta_tilde(t_tilde, k_tilde) / self.config.T_max)
            error_sigma = tf.reduce_mean(tf.abs(sigma_exact - sigma_nn) / sigma_exact)
            rmse_sigma = tf.sqrt(tf.reduce_mean(tf.square(1 - sigma_nn / sigma_exact)))
            error_sigma_list.append(error_sigma)
            rmse_sigma_list.append(rmse_sigma)

            # Print progress
            if iter_ % self.config.print_epochs == 0:
                rmse_fit = tf.sqrt(tf.reduce_mean(tf.square(self.model.neural_phi(T_nn, K_nn) - phi_ref)))
                elapsed = time.time() - start_time
                print(f"  Epoch {iter_:5d} | L_phi: {loss_phi:.4f} | L_dup: {loss_dupire:.4f} | " +
                      f"σ_err: {error_sigma:.4f} | RMSE: {rmse_fit:.4f} | {elapsed:.1f}s")

            # Learning rate decay
            if iter_ % self.config.lr_decay_steps == 0 and iter_ != 0:
                learning_rate /= self.config.lr_decay_rate
                self.model.optimizer_NN_phi.learning_rate.assign(learning_rate)
                self.model.optimizer_NN_eta.learning_rate.assign(learning_rate / 10)

            # Save checkpoints and visualize
            if iter_ % self.config.save_epochs == 0 and iter_ != 0:
                if visualizer is not None:
                    visualizer.plot_progress(
                        loss_phi_list, loss_dupire_list, loss_reg_list, error_sigma_list,
                        T_nn, K_nn, phi_ref, t_tilde, k_tilde, phi_tilde_ref,
                        iter_
                    )

                if iter_ > int(self.config.num_epochs - 1):
                    self.save_checkpoint(output_dir, f'NN_phi_{iter_}.keras', f'NN_eta_{iter_}.keras')

        elapsed_total = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

        return rmse_sigma_list, error_sigma_list

    def save_checkpoint(self, output_dir: str, phi_filename: str, eta_filename: str):
        """Save model checkpoints"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        phi_path = os.path.join(output_dir, phi_filename)
        eta_path = os.path.join(output_dir, eta_filename)

        tf.keras.models.save_model(self.model.NN_phi_tilde, filepath=phi_path, overwrite=True)
        tf.keras.models.save_model(self.model.NN_eta_tilde, filepath=eta_path, overwrite=True)

        print(f"  ✓ Saved checkpoint: {phi_filename}, {eta_filename}")


class TrainingVisualizer:
    """
    Visualizes training progress

    Generates:
    - Option price surfaces (NN vs exact)
    - Local volatility surfaces (NN vs exact)
    - PDE gradients
    - Loss curves
    """

    def __init__(self, config: DupirePipelineConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_progress(self, loss_phi_list, loss_dupire_list, loss_reg_list, error_sigma_list,
                     T_nn, K_nn, phi_ref, t_tilde, k_tilde, phi_tilde_ref, step):
        """
        Generate all training plots

        Only generates plots enabled in config.plot_config
        """
        if not self.config.plot_config.enable_training_plots:
            return

        # Generate visualizations based on config
        if self.config.plot_config.plot_loss_curves:
            self._plot_losses(loss_phi_list, loss_dupire_list, loss_reg_list, error_sigma_list, step)

    def _plot_losses(self, loss_phi_list, loss_dupire_list, loss_reg_list, error_sigma_list, step):
        """Plot loss curves"""
        fig, ax = plt.subplots(1, 2, figsize=[12, 3], dpi=self.config.plot_config.dpi)

        # Plot primary losses in log scale
        ax[0].semilogy(loss_phi_list, label='loss_phi', color=COLORS['model'])
        ax[0].semilogy(loss_dupire_list, label='loss_dupire', color=COLORS['kde'])
        ax[0].semilogy(loss_reg_list, label='loss_reg', color=COLORS['normal'])
        ax[0].legend(loc='upper right')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Loss (log scale)')
        ax[0].set_title('Training Losses', fontweight='bold')
        ax[0].grid(True, alpha=0.3)

        # Plot relative error in volatility
        ax[1].plot(error_sigma_list, label='relative error σ', color=COLORS['mc_hist'])
        ax[1].legend(loc='upper right')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Relative Error')
        ax[1].set_title('Local Volatility Error', fontweight='bold')
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f'losses_{step}.png')
        plt.savefig(output_path, dpi=self.config.plot_config.dpi, bbox_inches='tight')
        plt.close()


# =============================================================================
# [4] STAGE 3: PDF ANALYSIS
# =============================================================================

def load_trained_models(model_dir: str) -> Tuple[tf.keras.Model, tf.keras.Model, Dict]:
    """
    Load pre-trained NN_phi and NN_eta models

    Args:
        model_dir: Directory containing NN_phi_final.keras and NN_eta_final.keras

    Returns:
        (nn_phi, nn_eta, metadata): Loaded models and metadata
    """
    print(f"\nLoading trained models from: {model_dir}")

    # Try multiple possible filenames
    phi_filenames = ['NN_phi_final.keras', 'NN_phi.keras']
    eta_filenames = ['NN_eta_final.keras', 'NN_eta.keras']

    phi_path = None
    eta_path = None

    for fname in phi_filenames:
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            phi_path = path
            break

    for fname in eta_filenames:
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            eta_path = path
            break

    if phi_path is None:
        raise FileNotFoundError(f"NN_phi model not found in {model_dir}")
    if eta_path is None:
        raise FileNotFoundError(f"NN_eta model not found in {model_dir}")

    # Load models
    print(f"  Loading NN_phi from: {os.path.basename(phi_path)}")
    nn_phi = tf.keras.models.load_model(phi_path)
    print(f"    ✓ Loaded with {nn_phi.count_params():,} parameters")

    print(f"  Loading NN_eta from: {os.path.basename(eta_path)}")
    nn_eta = tf.keras.models.load_model(eta_path)
    print(f"    ✓ Loaded with {nn_eta.count_params():,} parameters")

    # Load metadata if available
    metadata = load_metadata(model_dir)
    if not metadata:
        metadata = {
            'model_dir': model_dir,
            'phi_params': nn_phi.count_params(),
            'eta_params': nn_eta.count_params(),
            'loaded_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    print("  ✓ Models loaded successfully!")
    return nn_phi, nn_eta, metadata


class PDFAnalyzer:
    """
    Analyzes probability density functions from trained models

    Key Methods:
        simulate_paths_with_nn_volatility(): MC using NN predictions
        extract_model_implied_density(): Get PDF from option prices
        fit_lognormal_corrected_method(): Fit log-normal distribution
        create_enhanced_pdf_analysis(): Generate three-panel plots
    """

    def __init__(self, nn_phi: tf.keras.Model, nn_eta: tf.keras.Model,
                 config: DupirePipelineConfig, metadata: Dict):
        self.nn_phi = nn_phi
        self.nn_eta = nn_eta
        self.config = config
        self.metadata = metadata

        # Extract scaling parameters from metadata
        if 'scaling' in metadata:
            self.t_max = metadata['scaling']['t_max']
            self.k_max = metadata['scaling']['k_max']
            self.k_min = metadata['scaling']['k_min']
        else:
            self.t_max = config.T_max
            self.k_max = config.K_max
            self.k_min = config.K_min

    def prepare_data_for_nn(self, T: float, K: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Scale T and K to neural network's normalized space

        Transformations:
            t_tilde = T / t_max
            k_tilde = exp(-r*T) * K / k_max
        """
        T_arr = np.full_like(K, T, dtype=np.float32)

        t_tilde = T_arr / self.t_max
        k_tilde = np.exp(-self.config.r * T_arr) * K / self.k_max

        t_tilde_tensor = tf.reshape(tf.constant(t_tilde, dtype=data_type), [-1, 1])
        k_tilde_tensor = tf.reshape(tf.constant(k_tilde, dtype=data_type), [-1, 1])

        return t_tilde_tensor, k_tilde_tensor

    def get_neural_local_volatility(self, t: float, S: np.ndarray) -> np.ndarray:
        """
        Query neural network to get local volatility at (t, S)

        The NN predicts eta_tilde, which relates to local volatility by:
            σ(t,S) = sqrt(2 * eta_tilde / t_max)
        """
        K = S  # Use S as strike K for local volatility query

        t_tilde, k_tilde = self.prepare_data_for_nn(t, K)

        eta_tilde = self.nn_eta(tf.concat([t_tilde, k_tilde], axis=1))

        sigma = tf.sqrt(2.0 * eta_tilde / self.t_max)

        return tf.squeeze(sigma).numpy()

    def extract_mc_samples_from_training_data(self, data_path: str,
                                             T_values: List[float],
                                             verbose: bool = True) -> Dict[float, np.ndarray]:
        """
        Extract Monte Carlo samples from saved training data

        This reuses the exact MC paths generated during training, which:
        1. Is instant (no new simulation needed)
        2. Uses all M_train paths (typically 1,000,000)
        3. Compares against exact σ(t,x) used in training

        Args:
            data_path: Path to training_data.npz file
            T_values: List of maturities to extract
            verbose: Print progress

        Returns:
            Dictionary mapping maturity T -> final stock prices S_T
        """
        if verbose:
            print(f"\n{'='*80}")
            print("EXTRACTING MC SAMPLES FROM TRAINING DATA")
            print(f"{'='*80}")
            print(f"  Loading from: {data_path}")

        # Load training data
        data = np.load(data_path)
        S_matrix = data['S_matrix']  # Shape: [N_t, M]
        t_all = data['t_all'].flatten()  # Shape: [N_t]

        M = S_matrix.shape[1]  # Number of paths
        N_t = S_matrix.shape[0]  # Number of time steps

        if verbose:
            print(f"  Loaded MC data: {N_t} time steps × {M:,} paths")
            print(f"  Time range: [{t_all.min():.3f}, {t_all.max():.3f}]")
            print(f"  Extracting samples at maturities: {T_values}")

        results = {}

        for T in T_values:
            # Find closest time index
            idx = np.argmin(np.abs(t_all - T))
            actual_T = t_all[idx]

            # Extract stock prices at this maturity
            S_T = S_matrix[idx, :]
            results[T] = S_T

            if verbose:
                print(f"    ✓ T={T:.3f} (actual={actual_T:.3f}): "
                      f"S ∈ [{S_T.min():.1f}, {S_T.max():.1f}], "
                      f"{M:,} samples")

        if verbose:
            print(f"  ✓ Extraction complete!")

        return results

    def simulate_paths_with_nn_volatility(self, T_values: List[float],
                                         n_paths: Optional[int] = None,
                                         dt: Optional[float] = None,
                                         verbose: bool = True) -> Dict[float, np.ndarray]:
        """
        Monte Carlo simulation using neural network's local volatility

        Simulates: dS_t = r S_t dt + σ_net(t,S) S_t dB_t

        Args:
            T_values: List of maturities to simulate to
            n_paths: Number of MC paths
            dt: Time step
            verbose: Print progress

        Returns:
            Dictionary mapping maturity T -> final stock prices S_T
        """
        if n_paths is None:
            n_paths = self.config.analysis_config.n_paths_analysis
        if dt is None:
            dt = self.config.analysis_config.dt_analysis

        if verbose:
            print(f"\n{'='*80}")
            print("MONTE CARLO SIMULATION WITH NN LOCAL VOLATILITY")
            print(f"{'='*80}")
            print(f"  Maturities: {T_values}")
            print(f"  Number of paths: {n_paths:,}")
            print(f"  Time step: dt = {dt}")
            print(f"  SDE: dS_t = r*S_t*dt + σ_NN(t,S)*S_t*dW_t")

        T_max_sim = max(T_values)
        n_steps = int(n_paths)

        results = {}

        # Initialize paths
        S_current = np.full(n_paths, self.config.S0, dtype=np.float32)

        # Generate Brownian increments
        np.random.seed(42)
        dW = np.random.normal(0, np.sqrt(dt), size=(n_steps, n_paths))

        saved_maturities = set()

        if verbose:
            print(f"  Simulating {n_steps} time steps...")
            start_time = time.time()

        for i in range(n_steps):
            t_current = i * dt

            # Query NN for local volatility at current (t, S)
            sigma_local = self.get_neural_local_volatility(t_current, S_current)

            # Euler-Maruyama step
            drift = self.config.r * S_current * dt
            diffusion = sigma_local * S_current * dW[i]
            S_current = S_current + drift + diffusion

            # Ensure positive stock prices
            S_current = np.maximum(S_current, 0.01)

            # Check if we've reached any target maturities
            for T in T_values:
                if T not in saved_maturities and t_current >= T - dt/2:
                    results[T] = S_current.copy()
                    saved_maturities.add(T)
                    if verbose:
                        print(f"    ✓ T={T:.3f}: S ∈ [{S_current.min():.1f}, {S_current.max():.1f}]")

        # Ensure all maturities are saved
        for T in T_values:
            if T not in results:
                results[T] = S_current.copy()
                if verbose:
                    print(f"    ✓ T={T:.3f} (final): S ∈ [{S_current.min():.1f}, {S_current.max():.1f}]")

        if verbose:
            elapsed = time.time() - start_time
            print(f"  ✓ MC simulation complete in {elapsed:.1f}s")

        return results

    def extract_model_implied_density(self, T: float, K_grid: np.ndarray) -> np.ndarray:
        """
        Extract model-implied density from NN option prices

        Risk-neutral density: f(K) = exp(rT) * ∂²C/∂K²

        Uses automatic differentiation to compute second derivatives.
        """
        t_tilde, k_tilde = self.prepare_data_for_nn(T, K_grid)

        # Compute second derivatives using automatic differentiation
        with tf.GradientTape(persistent=True) as tape_outer:
            tape_outer.watch(k_tilde)
            with tf.GradientTape(persistent=True) as tape_inner:
                tape_inner.watch(k_tilde)
                # Get option price from neural network
                phi_tilde = self.nn_phi(tf.concat([t_tilde, k_tilde], axis=1))

            # First derivative
            grad_phi_k_tilde = tape_inner.gradient(phi_tilde, k_tilde)

        # Second derivative
        grad_phi_kk_tilde = tape_outer.gradient(grad_phi_k_tilde, k_tilde)

        # Transform back to original coordinates using chain rule
        dk_tilde_dK = np.exp(-self.config.r * T) / self.k_max

        # Chain rule: ∂²φ/∂K² = (∂²φ̃/∂k̃²) * (∂k̃/∂K)² * S₀
        grad_phi_KK = grad_phi_kk_tilde * (dk_tilde_dK ** 2) * self.config.S0

        # Apply discount factor
        risk_neutral_density = np.exp(self.config.r * T) * grad_phi_KK.numpy().flatten()

        # Ensure positivity
        density = np.maximum(risk_neutral_density, 0)

        # Normalize
        if len(K_grid) > 1 and np.sum(density) > 0:
            integral = np.trapz(density, K_grid)
            if integral > 0:
                density = density / integral

        return density

    def fit_lognormal_corrected_method(self, K_grid: np.ndarray,
                                      f_vals: np.ndarray) -> Tuple:
        """
        Fit log-normal distribution using corrected method of moments

        CORRECTED transformations:
        - Proper variance: Var[K] = E[K²] - E[K]²
        - Correct Jacobian: g(x) = f(K) * σ * K

        Returns:
            (mu, sigma, mean_K, var_K, x_vals, g_vals)
        """
        if len(K_grid) < 2:
            return None, None, None, None, None, None

        integral = np.trapz(f_vals, K_grid)
        if integral <= 0:
            return None, None, None, None, None, None

        f_vals_norm = f_vals / integral

        # Compute moments CORRECTLY
        mean_K = np.trapz(K_grid * f_vals_norm, K_grid)
        second_moment = np.trapz((K_grid**2) * f_vals_norm, K_grid)
        var_K = second_moment - mean_K**2

        if var_K <= 0 or mean_K <= 0:
            return None, None, mean_K, var_K, None, None

        # Fit log-normal parameters
        cv_squared = var_K / (mean_K**2)
        if cv_squared <= 0:
            return None, None, mean_K, var_K, None, None

        sigma_squared = np.log(cv_squared + 1)
        mu = np.log(mean_K) - sigma_squared/2
        sigma = np.sqrt(sigma_squared)

        # Apply CORRECT transformation with Jacobian
        x_vals = (np.log(K_grid) - mu) / sigma
        jacobian = sigma * K_grid  # |dK/dx| = σ * K
        g_vals = f_vals_norm * jacobian

        # Normalize in x-space
        # Use trapz for proper integration with non-uniform spacing
        if len(x_vals) > 1:
            integral_x = np.trapz(g_vals, x_vals)
            if integral_x > 0:
                g_vals = g_vals / integral_x

        return mu, sigma, mean_K, var_K, x_vals, g_vals

    def create_enhanced_pdf_analysis(self, mc_data: Dict[float, np.ndarray],
                                    T_values: Optional[List[float]] = None) -> Tuple[Figure, Dict]:
        """
        Generate publication-quality PDF analysis plots

        Creates three-panel plots for each maturity:
        - Panel 1: Strike price space (K-space)
        - Panel 2: Log-space (ln K-space)
        - Panel 3: Gaussian standardized space

        Args:
            mc_data: Monte Carlo data (dict mapping T -> stock prices)
            T_values: List of maturities to plot

        Returns:
            (fig, results): Figure object and results dictionary
        """
        if T_values is None:
            T_values = sorted(mc_data.keys())

        print(f"\n{'='*80}")
        print("GENERATING PDF ANALYSIS PLOTS")
        print(f"{'='*80}")
        print(f"Analyzing {len(T_values)} maturities: {T_values}")

        # Create figure
        fig, axes = plt.subplots(len(T_values), 3, figsize=(18, 5*len(T_values)))
        fig.patch.set_facecolor('white')

        # Title
        fig.suptitle(
            'PDF Analysis: Neural Network Synthetic Local Volatility Model\n' +
            r'Monte Carlo with $dS_t = rS_t dt + \sigma_{NN}(t,S)S_t dW_t$',
            fontsize=18, fontweight='bold', color=COLORS['text'], y=0.98
        )

        if len(T_values) == 1:
            axes = axes.reshape(1, -1)

        results = {}

        for i, T in enumerate(T_values):
            print(f"\n--- Analyzing T = {T:.3f} ---")

            # Get Monte Carlo samples
            raw_samples = mc_data[T]
            mc_samples = raw_samples[np.isfinite(raw_samples) & (raw_samples > 0)]

            if mc_samples.size == 0:
                print("  ✗ No finite Monte Carlo samples for this maturity. Skipping row.")
                for ax in axes[i]:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=14, fontweight='bold')
                    ax.set_axis_off()
                results[T] = {}
                continue

            print(f"  MC samples: {len(mc_samples):,}, "
                  f"range: [{mc_samples.min():.1f}, {mc_samples.max():.1f}]")

            domain_mask = (mc_samples >= self.k_min) & (mc_samples <= self.k_max)
            trimmed_samples = mc_samples[domain_mask]

            use_trimmed = trimmed_samples.size >= 100
            if use_trimmed and trimmed_samples.size < mc_samples.size:
                trimmed_out = mc_samples.size - trimmed_samples.size
                print(f"  - Trimmed {trimmed_out:,} samples outside "
                      f"[{self.k_min:.1f}, {self.k_max:.1f}]")
            elif not use_trimmed and trimmed_samples.size == 0:
                print(f"  ⚠️  All MC samples lie outside strike range "
                      f"[{self.k_min:.1f}, {self.k_max:.1f}]. Using untrimmed data "
                      "for diagnostics.")
            elif not use_trimmed and trimmed_samples.size < mc_samples.size:
                print(f"  ⚠️  Fewer than 100 samples inside strike range "
                      f"[{self.k_min:.1f}, {self.k_max:.1f}]; using untrimmed data.")

            mc_used = trimmed_samples if use_trimmed else mc_samples

            if mc_used.size >= 2:
                q1, q99 = np.percentile(mc_used, [1, 99])
                K_min_plot = max(self.k_min, q1)
                K_max_plot = min(self.k_max, q99)
            else:
                K_min_plot, K_max_plot = self.k_min, self.k_max

            if not np.isfinite(K_min_plot) or not np.isfinite(K_max_plot) or K_max_plot <= K_min_plot:
                K_min_plot, K_max_plot = self.k_min, self.k_max

            K_grid = np.linspace(K_min_plot, K_max_plot, 150, dtype=np.float64)

            # Extract model-implied density
            print(f"  Extracting NN model-implied density...")
            model_density = self.extract_model_implied_density(T, K_grid)

            # --- PANEL 1: K-SPACE ---
            ax1 = axes[i, 0]
            ax1.hist(mc_used, bins=50, density=True, alpha=0.7,
                    color=COLORS['mc_hist'], edgecolor='white', linewidth=0.5,
                    label='Monte Carlo')
            ax1.plot(K_grid, model_density, color=COLORS['model'], linewidth=3,
                    label='NN Model', zorder=10)
            ax1.set_xlabel('Strike Price $K$', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Density $f(K)$', fontsize=12, fontweight='bold')
            ax1.set_title(f'Strike Distribution (T = {T:.2f})',
                         fontsize=14, fontweight='bold', pad=15)
            ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.4, linestyle='--')
            ax1.set_facecolor(COLORS['background'])

            # --- PANEL 2: LOG-SPACE ---
            ax2 = axes[i, 1]
            log_samples = np.log(mc_used)
            ax2.hist(log_samples, bins=50, density=True, alpha=0.7,
                    color=COLORS['kde'], edgecolor='white', linewidth=0.5,
                    label='MC Log Returns')

            # Transform model density to log-space with Jacobian
            log_K = np.log(K_grid)
            log_density = model_density * K_grid  # Jacobian |dK/d(ln K)| = K
            ax2.plot(log_K, log_density, color=COLORS['model'], linewidth=3,
                    label='NN Model', zorder=10)
            ax2.set_xlabel(r'$\ln$(Strike Price)', fontsize=12, fontweight='bold')
            ax2.set_ylabel(r'Density $f(\ln K)$', fontsize=12, fontweight='bold')
            ax2.set_title(f'Log-Normal Distribution (T = {T:.2f})',
                         fontsize=14, fontweight='bold', pad=15)
            ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.4, linestyle='--')
            ax2.set_facecolor(COLORS['background'])

            # --- PANEL 3: GAUSSIAN SPACE ---
            ax3 = axes[i, 2]

            # Fit log-normal to FULL MC DATA (not truncated)
            print(f"  Fitting log-normal to full MC distribution...")
            ln_mc_full = np.log(mc_samples)
            mu_mc = ln_mc_full.mean()
            sigma_mc = ln_mc_full.std()

            # Also fit to model density for comparison
            mu_model, sigma_model, mean_K, var_K, x_model, g_model = self.fit_lognormal_corrected_method(
                K_grid, model_density)

            if mu_mc is not None and sigma_mc is not None and sigma_mc > 1e-8:
                # Standardize MC using its OWN fitted parameters
                x_mc = (ln_mc_full - mu_mc) / sigma_mc

                # KDE of MC in x-space
                x_range = np.linspace(-3, 3, 100)
                if len(x_mc) > 20:
                    kde = KernelDensity(kernel='gaussian', bandwidth=0.12)
                    kde.fit(x_mc.reshape(-1, 1))
                    kde_vals = np.exp(kde.score_samples(x_range.reshape(-1, 1)))
                else:
                    kde_vals = np.zeros_like(x_range)

                # Standard normal reference
                std_normal = norm.pdf(x_range)

                # Plot
                ax3.hist(x_mc, bins=30, density=True, alpha=0.6,
                        color=COLORS['mc_hist'], edgecolor='white', linewidth=0.5,
                        label='MC (standardized)')
                ax3.plot(x_range, kde_vals, color=COLORS['kde'], linewidth=3,
                        label='KDE of MC', zorder=10)

                # Transform model density to MC coordinate system (mu_mc, sigma_mc)
                # This ensures model and MC are in the same coordinate system
                if len(K_grid) > 5 and len(model_density) == len(K_grid):
                    # Transform K_grid points to x-space using MC parameters
                    x_model_mc_space = (np.log(K_grid) - mu_mc) / sigma_mc
                    
                    # Transform model density with Jacobian: g(x) = f(K) * sigma_mc * K
                    g_model_mc_space = model_density * sigma_mc * K_grid
                    
                    # Normalize in x-space
                    if len(x_model_mc_space) > 1:
                        integral_g = np.trapz(g_model_mc_space, x_model_mc_space)
                        if integral_g > 0:
                            g_model_mc_space = g_model_mc_space / integral_g
                    
                    # Interpolate to x_range for plotting
                    # Only use points within reasonable range to avoid extrapolation issues
                    valid_mask = (x_model_mc_space >= -4) & (x_model_mc_space <= 4)
                    if np.sum(valid_mask) > 5:
                        x_valid = x_model_mc_space[valid_mask]
                        g_valid = g_model_mc_space[valid_mask]
                        # Sort for interpolation
                        sort_idx = np.argsort(x_valid)
                        g_interp = np.interp(x_range, x_valid[sort_idx], g_valid[sort_idx],
                                            left=np.nan, right=np.nan)
                        # Only plot where we have valid interpolation
                        valid_interp = np.isfinite(g_interp)
                        if np.any(valid_interp):
                            ax3.plot(x_range[valid_interp], g_interp[valid_interp], 
                                    color=COLORS['model'], linestyle='--',
                                    linewidth=3, label='NN Model g(x)', zorder=9)
                        else:
                            g_interp = None
                    else:
                        g_interp = None
                else:
                    g_interp = None

                ax3.plot(x_range, std_normal, color=COLORS['normal'], linestyle=':',
                        linewidth=3, label='Standard Normal', zorder=8)

                # Statistics
                skew_mc = skew(x_mc)
                kurt_mc = kurtosis(x_mc, fisher=True)

                # Correlation
                try:
                    if g_interp is not None:
                        corr = np.corrcoef(kde_vals, g_interp)[0, 1]
                    else:
                        corr = np.nan
                except:
                    corr = np.nan

                # Add validation of standardization
                x_mc_mean = x_mc.mean()
                x_mc_std = x_mc.std()

                # Build title showing both MC and model parameters
                if mu_model is not None and sigma_model is not None:
                    title_str = (f'Gaussian Space (T = {T:.2f})\n' +
                                f'MC Standardization: μ_MC={mu_mc:.2f}, σ_MC={sigma_mc:.2f} → x_MC ~ N(0,1)\n' +
                                f'Model Fit: μ_model={mu_model:.2f}, σ_model={sigma_model:.2f}\n' +
                                f'MC Stats: μ={x_mc_mean:.2f}, σ={x_mc_std:.2f}, Skew={skew_mc:.2f}, ExKurt={kurt_mc:.2f}')
                else:
                    title_str = (f'Gaussian Space (T = {T:.2f})\n' +
                                f'MC Standardization: μ_MC={mu_mc:.2f}, σ_MC={sigma_mc:.2f} → x_MC ~ N(0,1)\n' +
                                f'MC Stats: μ={x_mc_mean:.2f}, σ={x_mc_std:.2f}, Skew={skew_mc:.2f}, ExKurt={kurt_mc:.2f}')
                
                ax3.set_title(title_str, fontsize=13, fontweight='bold', pad=15)
                ax3.set_xlabel(r'$x = \frac{\ln K - \mu_{MC}}{\sigma_{MC}}$', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Density $g(x)$', fontsize=12, fontweight='bold')
                ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
                ax3.set_xlim(-3, 3)
                ax3.grid(True, alpha=0.4, linestyle='--')
                ax3.set_facecolor(COLORS['background'])

                # Store both MC and model parameters for completeness
                results[T] = {
                    'mu_mc': mu_mc, 'sigma_mc': sigma_mc,
                    'mu_model': mu_model, 'sigma_model': sigma_model,
                    'mean_K': mean_K, 'var_K': var_K,
                    'correlation': corr, 'skewness': skew_mc, 'excess_kurtosis': kurt_mc,
                    'x_mc_mean': x_mc_mean, 'x_mc_std': x_mc_std
                }

                # Validation check
                if abs(x_mc_mean) > 0.1 or abs(x_mc_std - 1.0) > 0.1:
                    print(f"    ⚠️  Standardization check: x_mc μ={x_mc_mean:.3f}, σ={x_mc_std:.3f} (should be ~0, ~1)")

                # Print summary with both MC and model parameters
                if mu_model is not None and sigma_model is not None:
                    print(f"    ✓ MC fit: μ_MC={mu_mc:.3f}, σ_MC={sigma_mc:.3f} | Model fit: μ_model={mu_model:.3f}, σ_model={sigma_model:.3f}")
                    print(f"      MC standardized: μ={x_mc_mean:.3f}, σ={x_mc_std:.3f} | corr={corr:.3f}, skew={skew_mc:.3f}")
                else:
                    print(f"    ✓ MC fit: μ_MC={mu_mc:.3f}, σ_MC={sigma_mc:.3f} | MC standardized: μ={x_mc_mean:.3f}, σ={x_mc_std:.3f} | corr={corr:.3f}, skew={skew_mc:.3f}")

            else:
                ax3.text(0.5, 0.5, f'T = {T:.2f}\nLog-normal fitting failed',
                        ha='center', va='center', transform=ax3.transAxes,
                        fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
                ax3.set_facecolor(COLORS['background'])

        # Improve layout
        plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=3.0, w_pad=2.0)

        return fig, results


# =============================================================================
# [5] PIPELINE ORCHESTRATOR
# =============================================================================

class DupirePipeline:
    """
    Main orchestrator - coordinates all stages

    Usage:
        pipeline = DupirePipeline(config)
        pipeline.run()

    Stages:
        1. Data Generation: Monte Carlo with exact σ(t,x)
        2. Model Training: Neural network calibration
        3. PDF Analysis: Validation through density comparison
    """

    def __init__(self, config: DupirePipelineConfig):
        self.config = config
        self.output_dir = self._setup_output_dir()

    def _setup_output_dir(self) -> str:
        """Create output directory if it doesn't exist"""
        output_dir = self.config.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        return output_dir

    def run(self):
        """Execute enabled stages"""

        print(f"\n{'='*80}")
        print("DUPIRE LOCAL VOLATILITY PIPELINE")
        print(f"{'='*80}")
        print(f"Mode: {self.config.mode}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}\n")

        # Stage 1: Data Generation
        if self.config.data_generation and self.config.mode in ['all', 'generate']:
            self.stage1_generate_data()

        # Stage 2: Model Training
        if self.config.training and self.config.mode in ['all', 'train']:
            self.stage2_train_models()

        # Stage 3: PDF Analysis
        if self.config.analysis and self.config.mode in ['all', 'analyze']:
            self.stage3_pdf_analysis()

        print(f"\n{'='*80}")
        print("✓ PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}\n")

    def stage1_generate_data(self):
        """Stage 1: Generate Monte Carlo training data"""
        print(f"\n{'='*80}")
        print("STAGE 1: DATA GENERATION")
        print(f"{'='*80}")

        data_gen = DataGenerator(self.config)

        # Check if data already exists
        data_path = os.path.join(self.output_dir, 'training_data.npz')
        if self.config.skip_if_exists and os.path.exists(data_path):
            print(f"  Training data already exists: {data_path}")
            print("  Skipping data generation (use --skip-if-exists false to regenerate)")
            return

        # Generate data
        T_nn, K_nn, phi_ref = data_gen.get_training_data()

        # Save data
        if self.config.save_training_data:
            data_gen.save_training_data(self.output_dir)

        # Save metadata
        save_metadata(self.config, self.output_dir)

        print(f"\n  ✓ Stage 1 complete: {len(T_nn)} training samples generated")

    def stage2_train_models(self):
        """Stage 2: Train neural networks"""
        print(f"\n{'='*80}")
        print("STAGE 2: MODEL TRAINING")
        print(f"{'='*80}")

        # Check if models already exist
        phi_path = os.path.join(self.output_dir, 'NN_phi_final.keras')
        eta_path = os.path.join(self.output_dir, 'NN_eta_final.keras')

        if self.config.skip_if_exists and os.path.exists(phi_path) and os.path.exists(eta_path):
            print(f"  Trained models already exist in {self.output_dir}")
            print("  Skipping training (use --skip-if-exists false to retrain)")
            return

        # Load or generate data
        data_gen = DataGenerator(self.config)
        data_path = os.path.join(self.output_dir, 'training_data.npz')

        if os.path.exists(data_path):
            T_nn, K_nn, phi_ref = data_gen.load_training_data(self.output_dir)
        else:
            print("  Training data not found, generating...")
            T_nn, K_nn, phi_ref = data_gen.get_training_data()
            if self.config.save_training_data:
                data_gen.save_training_data(self.output_dir)

        # Scale data
        phi_tilde_ref = phi_ref / self.config.S0
        t_tilde, k_tilde = data_gen.scale_data(T_nn, K_nn)

        # Get random sampling bounds
        t_min = tf.reduce_min(t_tilde).numpy()
        t_max = tf.reduce_max(t_tilde).numpy()
        k_min = tf.reduce_min(k_tilde).numpy()
        k_max = tf.reduce_max(k_tilde).numpy()

        # Build model
        model = DupireNeuralModel(self.config, data_gen)
        model.build_models()

        # Setup trainer and visualizer
        trainer = ModelTrainer(model, self.config)
        visualizer = TrainingVisualizer(self.config, self.output_dir) if self.config.plot_config.enable_training_plots else None

        # Train
        rmse_sigma_list, error_sigma_list = trainer.train(
            T_nn, K_nn, phi_ref,
            t_tilde, k_tilde, phi_tilde_ref,
            t_min, t_max, k_min, k_max,
            self.output_dir,
            visualizer
        )

        # Save final models
        print(f"\n  Saving final models...")
        trainer.save_checkpoint(self.output_dir, 'NN_phi_final.keras', 'NN_eta_final.keras')

        # Save metadata
        save_metadata(self.config, self.output_dir)

        print(f"\n  ✓ Stage 2 complete: Models saved to {self.output_dir}")

    def stage3_pdf_analysis(self):
        """Stage 3: PDF analysis and validation"""
        print(f"\n{'='*80}")
        print("STAGE 3: PDF ANALYSIS")
        print(f"{'='*80}")

        # Load models
        try:
            nn_phi, nn_eta, metadata = load_trained_models(self.output_dir)
        except FileNotFoundError as e:
            print(f"  ✗ Error: {e}")
            print("  Please run training first (--mode train or --mode all)")
            return

        # Determine analysis maturities within the training domain
        requested_T = list(self.config.analysis_config.T_analysis)
        train_T_min = None
        train_T_max = None
        data_path = os.path.join(self.output_dir, 'training_data.npz')

        if os.path.exists(data_path):
            try:
                with np.load(data_path) as train_data:
                    T_array = np.asarray(train_data['T'], dtype=np.float64).reshape(-1)
                    if T_array.size > 0:
                        train_T_min = float(np.min(T_array))
                        train_T_max = float(np.max(T_array))
            except Exception as exc:
                print(f"  ⚠️  Unable to infer training maturities from {data_path}: {exc}")

        if train_T_min is None or train_T_max is None:
            # Fall back to configuration bounds if training data unavailable
            train_T_min = 0.0
            train_T_max = float(self.config.T_max)

        tol = 1e-6
        valid_T = []
        skipped_T = []
        for T in requested_T:
            if T < train_T_min - tol or T > train_T_max + tol:
                skipped_T.append(T)
                continue
            valid_T.append(T)

        if skipped_T:
            print(f"  ⚠️  Skipping maturities outside training range "
                  f"[{train_T_min:.3f}, {train_T_max:.3f}]: {skipped_T}")

        if not valid_T:
            print("  ✗ No analysis maturities remain within the training domain. "
                  "Adjust analysis_config.T_analysis and retry.")
            return

        # Ensure deterministic ordering
        valid_T = sorted(valid_T)

        # Create analyzer
        analyzer = PDFAnalyzer(nn_phi, nn_eta, self.config, metadata)
        analyzer.training_T_min = train_T_min
        analyzer.training_T_max = train_T_max

        # Determine MC data source: reuse training data or run new simulation
        mc_data = None

        if self.config.analysis_config.reuse_training_mc and os.path.exists(data_path):
            # Reuse training MC paths (exact volatility, all M_train samples)
            print(f"\n  Using training MC data (exact σ(t,x), reusing saved paths)")
            print(f"  Config: reuse_training_mc=True")
            try:
                mc_data = analyzer.extract_mc_samples_from_training_data(
                    data_path,
                    valid_T,
                    verbose=True
                )
            except Exception as e:
                print(f"  ⚠️  Failed to extract training data: {e}")
                print(f"  Falling back to new MC simulation with NN volatility")
                mc_data = None

        if mc_data is None and self.config.analysis_config.run_mc_with_nn_volatility:
            # Run new MC simulation with NN-predicted volatility
            print(f"\n  Running new MC simulation with NN volatility σ_NN(t,S)")
            if self.config.analysis_config.reuse_training_mc:
                print(f"  (Training data not available or extraction failed)")
            else:
                print(f"  Config: reuse_training_mc=False")
            mc_data = analyzer.simulate_paths_with_nn_volatility(
                valid_T,
                verbose=True
            )
        elif mc_data is None:
            print("  Skipping MC simulation (disabled in config)")
            return

        # Generate PDF analysis plots
        if self.config.plot_config.enable_pdf_plots:
            fig, results = analyzer.create_enhanced_pdf_analysis(
                mc_data,
                valid_T
            )

            # Save plots
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f'pdf_analysis_{timestamp}'

            if self.config.plot_config.save_png:
                png_file = os.path.join(self.output_dir, f'{base_filename}.png')
                fig.savefig(png_file, dpi=self.config.plot_config.dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none', pad_inches=0.2)
                print(f"\n  ✓ Saved: {png_file}")

            if self.config.plot_config.save_pdf:
                pdf_file = os.path.join(self.output_dir, f'{base_filename}.pdf')
                fig.savefig(pdf_file, dpi=self.config.plot_config.dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none', pad_inches=0.2)
                print(f"  ✓ Saved: {pdf_file}")

            plt.close(fig)

            # Print results summary
            print(f"\n  Results Summary:")
            print(f"  {'-'*70}")
            for T, res in results.items():
                mu_mc = res.get('mu_mc', 'N/A')
                sigma_mc = res.get('sigma_mc', 'N/A')
                mu_model = res.get('mu_model', 'N/A')
                sigma_model = res.get('sigma_model', 'N/A')
                
                # Format values
                mu_mc_str = f"{mu_mc:.3f}" if isinstance(mu_mc, (int, float)) else str(mu_mc)
                sigma_mc_str = f"{sigma_mc:.3f}" if isinstance(sigma_mc, (int, float)) else str(sigma_mc)
                mu_model_str = f"{mu_model:.3f}" if isinstance(mu_model, (int, float)) and mu_model is not None else "N/A"
                sigma_model_str = f"{sigma_model:.3f}" if isinstance(sigma_model, (int, float)) and sigma_model is not None else "N/A"
                
                print(f"    T={T:.2f}: μ_MC={mu_mc_str}, σ_MC={sigma_mc_str}, " +
                      f"μ_model={mu_model_str}, σ_model={sigma_model_str}")
                print(f"            corr={res['correlation']:.3f}, skew={res['skewness']:.3f}, " +
                      f"ex.kurt={res['excess_kurtosis']:.3f}")
            print(f"  {'-'*70}")

        print(f"\n  ✓ Stage 3 complete: PDF analysis saved to {self.output_dir}")


# =============================================================================
# [6] MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point with argument parsing"""

    parser = argparse.ArgumentParser(
        description="Dupire Local Volatility Pipeline - End-to-End Training and Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (100 epochs, reduced data)
  python consolidated_dupire_pipeline.py --preset quick

  # Full training (30,000 epochs)
  python consolidated_dupire_pipeline.py --preset full

  # Analysis only (requires existing models)
  python consolidated_dupire_pipeline.py --mode analyze --output-dir path/to/models

  # Custom configuration
  python consolidated_dupire_pipeline.py --num-epochs 5000 --ldup 2.0 --num-res-blocks 4

  # Disable specific plots
  python consolidated_dupire_pipeline.py --no-training-plots
        """
    )

    # Preset configurations
    parser.add_argument('--preset', choices=['quick', 'full'],
                       help='Use preset configuration')

    # Mode control
    parser.add_argument('--mode', choices=['all', 'generate', 'train', 'analyze'],
                       default='all', help='Pipeline stage to run (default: all)')

    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of training epochs (default: 30000)')
    parser.add_argument('--ldup', type=float, default=None,
                       help='Lambda for Dupire PDE loss (default: 1.0)')
    parser.add_argument('--num-res-blocks', type=int, default=None,
                       help='Number of residual blocks (default: 3)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: 1e-4)')

    # Data generation parameters
    parser.add_argument('--M-train', type=int, default=None,
                       help='Number of MC paths for training (default: 10000)')

    # Plot toggles
    parser.add_argument('--no-training-plots', action='store_true',
                       help='Disable all training plots')
    parser.add_argument('--no-pdf-plots', action='store_true',
                       help='Disable all PDF analysis plots')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-generated with timestamp)')
    parser.add_argument('--skip-if-exists', action='store_true', default=True,
                       help='Skip stages if output already exists (default: True)')
    parser.add_argument('--no-skip-if-exists', dest='skip_if_exists',
                       action='store_false',
                       help='Always run stages even if output exists')

    args = parser.parse_args()

    # Build configuration
    if args.preset == 'quick':
        config = DupirePipelineConfig.quick_test()
        print("Using QUICK TEST preset (100 epochs, reduced data)")
    elif args.preset == 'full':
        config = DupirePipelineConfig.full_training()
        print("Using FULL TRAINING preset (30,000 epochs)")
    else:
        config = DupirePipelineConfig()

    # Override with command-line args
    config.mode = args.mode
    config.skip_if_exists = args.skip_if_exists

    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.ldup is not None:
        config.lambda_pde = args.ldup
    if args.num_res_blocks is not None:
        config.num_res_blocks = args.num_res_blocks
    if args.lr is not None:
        config.lr_phi = args.lr
        config.lr_eta = args.lr / 10
    if args.M_train is not None:
        config.M_train = args.M_train
    if args.output_dir is not None:
        resolved_output_dir = DupirePipelineConfig.normalize_output_dir(args.output_dir)
        config.output_dir = resolved_output_dir
        if resolved_output_dir != args.output_dir:
            print(f"Resolved output directory to: {config.output_dir}")

    # Apply plot toggles
    if args.no_training_plots:
        config.plot_config.enable_training_plots = False
    if args.no_pdf_plots:
        config.plot_config.enable_pdf_plots = False

    # Run pipeline
    try:
        pipeline = DupirePipeline(config)
        pipeline.run()

        print(f"\n{'='*80}")
        print("✨ SUCCESS ✨")
        print(f"{'='*80}")
        print(f"All outputs saved to: {pipeline.output_dir}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n{'='*80}")
        print("✗ ERROR")
        print(f"{'='*80}")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
