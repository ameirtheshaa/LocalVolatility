#!/usr/bin/env python3
"""
================================================================================
NEURAL NETWORK vs ANALYTICAL SOLUTION COMPARISON
================================================================================

Compare trained neural network models against analytical Black-Scholes solution
for constant volatility σ=1.

This script validates the NN-based Dupire calibration by comparing against
known closed-form solutions when local volatility is constant.

COMPARISON METRICS:
------------------
1. Option Prices: C_NN(K,T) vs C_BS(K,T)
2. Local Volatility: σ_NN(K,T) vs σ=1
3. Risk-Neutral Density: f_NN(K) vs f_lognormal(K)
4. Statistical Moments: Mean, Variance, Skewness, Kurtosis

MODELS UNDER TEST:
-----------------
- Trained with constant σ=1.0
- Located in: models/runs/synthetic_paper_large_dataset_constant_vol/
- 1,000,000 MC paths, 30,000 epochs
- Parameters: S₀=1000, r=0.04, T∈[0.3,1.5], K∈[500,3000]

Author: Validation Tool
Date: 2025-11-02
================================================================================
"""

import os
import sys
import json
import datetime
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm, skew, kurtosis

# Import analytical solutions
from analytical_solutions import (
    black_scholes_call,
    lognormal_density,
    local_volatility_constant,
    validate_lognormal_moments
)

# Import pipeline components
from dupire_pipeline import load_trained_models, PDFAnalyzer
from config import DupirePipelineConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model directory
MODEL_DIR = 'models/runs/synthetic_paper_large_dataset_constant_vol'

# Analytical parameters (must match training)
SIGMA_ANALYTICAL = 1.0  # Constant volatility used in training

# Plotting configuration
COLORS = {
    'nn': '#E94B3C',          # Neural Network (warm red)
    'analytical': '#2C3E50',  # Analytical (dark blue-gray)
    'error': '#F39C12',       # Error (orange)
    'background': '#FAFAFA',  # Light gray background
}

DPI = 450  # High-quality output


# =============================================================================
# LOAD MODELS AND METADATA
# =============================================================================

def load_models_and_config(model_dir: str) -> Tuple:
    """
    Load trained NN models and extract configuration

    Returns:
        (nn_phi, nn_eta, metadata, config)
    """
    print(f"\n{'='*80}")
    print("LOADING TRAINED MODELS")
    print(f"{'='*80}")

    # Load Keras models
    nn_phi, nn_eta, metadata = load_trained_models(model_dir)

    # Extract configuration
    if 'scaling' in metadata:
        S0 = metadata['scaling']['S0']
        r = metadata['scaling']['r']
        T_max = metadata['scaling']['t_max']
        K_min = metadata['scaling']['k_min']
        K_max = metadata['scaling']['k_max']
    else:
        # Fallback to defaults
        print("  ⚠️  Scaling parameters not found in metadata, using defaults")
        S0 = 1000.0
        r = 0.04
        T_max = 1.5
        K_min = 500.0
        K_max = 3000.0

    # Create minimal config for PDFAnalyzer
    config = DupirePipelineConfig()
    config.S0 = S0
    config.r = r
    config.T_max = T_max
    config.K_min = K_min
    config.K_max = K_max

    print(f"  Model parameters:")
    print(f"    S₀ = {S0}")
    print(f"    r  = {r}")
    print(f"    T_max = {T_max}")
    print(f"    K ∈ [{K_min}, {K_max}]")

    return nn_phi, nn_eta, metadata, config


# =============================================================================
# COMPARISON 1: OPTION PRICES
# =============================================================================

def compare_option_prices(nn_phi: tf.keras.Model, config: DupirePipelineConfig,
                         analyzer: PDFAnalyzer, sigma: float = 1.0) -> Dict:
    """
    Compare NN option prices against Black-Scholes analytical solution

    Args:
        nn_phi: Trained option price network
        config: Pipeline configuration
        analyzer: PDF analyzer instance
        sigma: Analytical constant volatility

    Returns:
        Dictionary with comparison results and errors
    """
    print(f"\n{'='*80}")
    print("COMPARISON 1: OPTION PRICES (C_NN vs C_BS)")
    print(f"{'='*80}")

    # Create (K, T) grid
    N_K = 50
    N_T = 30
    K_grid = np.linspace(config.K_min, config.K_max, N_K)
    T_grid = np.linspace(0.3, config.T_max, N_T)
    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

    # Flatten for vectorized computation
    K_flat = K_mesh.flatten()
    T_flat = T_mesh.flatten()

    # Compute NN prices
    print("  Computing NN option prices...")
    t_tilde_list = []
    k_tilde_list = []
    for i in range(len(K_flat)):
        t_tilde, k_tilde = analyzer.prepare_data_for_nn(T_flat[i], np.array([K_flat[i]]))
        t_tilde_list.append(t_tilde.numpy()[0,0])
        k_tilde_list.append(k_tilde.numpy()[0,0])

    t_tilde_tensor = tf.constant(np.array(t_tilde_list).reshape(-1,1), dtype=tf.float32)
    k_tilde_tensor = tf.constant(np.array(k_tilde_list).reshape(-1,1), dtype=tf.float32)

    phi_tilde_nn = nn_phi(tf.concat([t_tilde_tensor, k_tilde_tensor], axis=1))
    C_nn = config.S0 * (1 - tf.exp(-phi_tilde_nn)).numpy().flatten()

    # Compute analytical Black-Scholes prices
    print("  Computing Black-Scholes analytical prices...")
    C_bs = black_scholes_call(config.S0, K_flat, T_flat, config.r, sigma)

    # Reshape to grid
    C_nn_grid = C_nn.reshape(N_T, N_K)
    C_bs_grid = C_bs.reshape(N_T, N_K)

    # Compute errors
    abs_error = np.abs(C_nn_grid - C_bs_grid)
    rel_error = abs_error / (C_bs_grid + 1e-10)  # Avoid division by zero

    rmse = np.sqrt(np.mean((C_nn - C_bs)**2))
    max_abs_error = np.max(abs_error)
    mean_rel_error = np.mean(rel_error)

    print(f"\n  Error Metrics:")
    print(f"    RMSE:              {rmse:.6f}")
    print(f"    Max absolute error: {max_abs_error:.6f}")
    print(f"    Mean relative error: {mean_rel_error:.4%}")

    return {
        'K_grid': K_grid,
        'T_grid': T_grid,
        'K_mesh': K_mesh,
        'T_mesh': T_mesh,
        'C_nn': C_nn_grid,
        'C_bs': C_bs_grid,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'rmse': rmse,
        'max_abs_error': max_abs_error,
        'mean_rel_error': mean_rel_error
    }


# =============================================================================
# COMPARISON 2: LOCAL VOLATILITY
# =============================================================================

def compare_local_volatility(nn_eta: tf.keras.Model, config: DupirePipelineConfig,
                             analyzer: PDFAnalyzer, sigma: float = 1.0) -> Dict:
    """
    Compare NN local volatility against constant analytical solution

    Args:
        nn_eta: Trained volatility network
        config: Pipeline configuration
        analyzer: PDF analyzer instance
        sigma: Analytical constant volatility

    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*80}")
    print("COMPARISON 2: LOCAL VOLATILITY (σ_NN vs σ=const)")
    print(f"{'='*80}")

    # Create (K, T) grid
    N_K = 50
    N_T = 30
    K_grid = np.linspace(config.K_min, config.K_max, N_K)
    T_grid = np.linspace(0.3, config.T_max, N_T)
    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

    K_flat = K_mesh.flatten()
    T_flat = T_mesh.flatten()

    # Compute NN volatility
    print("  Computing NN local volatility...")
    t_tilde_list = []
    k_tilde_list = []
    for i in range(len(K_flat)):
        t_tilde, k_tilde = analyzer.prepare_data_for_nn(T_flat[i], np.array([K_flat[i]]))
        t_tilde_list.append(t_tilde.numpy()[0,0])
        k_tilde_list.append(k_tilde.numpy()[0,0])

    t_tilde_tensor = tf.constant(np.array(t_tilde_list).reshape(-1,1), dtype=tf.float32)
    k_tilde_tensor = tf.constant(np.array(k_tilde_list).reshape(-1,1), dtype=tf.float32)

    eta_tilde_nn = nn_eta(tf.concat([t_tilde_tensor, k_tilde_tensor], axis=1))
    sigma_nn = tf.sqrt(2.0 * eta_tilde_nn / config.T_max).numpy().flatten()

    # Analytical volatility (constant)
    sigma_analytical = local_volatility_constant(K_flat, T_flat, sigma)

    # Reshape to grid
    sigma_nn_grid = sigma_nn.reshape(N_T, N_K)
    sigma_analytical_grid = sigma_analytical.reshape(N_T, N_K)

    # Compute errors
    abs_error = np.abs(sigma_nn_grid - sigma_analytical_grid)
    rel_error = abs_error / sigma

    rmse = np.sqrt(np.mean((sigma_nn - sigma_analytical)**2))
    max_abs_error = np.max(abs_error)
    mean_rel_error = np.mean(rel_error)

    print(f"\n  Error Metrics:")
    print(f"    RMSE:               {rmse:.6f}")
    print(f"    Max absolute error: {max_abs_error:.6f}")
    print(f"    Mean relative error: {mean_rel_error:.4%}")

    return {
        'K_grid': K_grid,
        'T_grid': T_grid,
        'K_mesh': K_mesh,
        'T_mesh': T_mesh,
        'sigma_nn': sigma_nn_grid,
        'sigma_analytical': sigma_analytical_grid,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'rmse': rmse,
        'max_abs_error': max_abs_error,
        'mean_rel_error': mean_rel_error
    }


# =============================================================================
# COMPARISON 3: RISK-NEUTRAL DENSITY
# =============================================================================

def compare_densities(nn_phi: tf.keras.Model, config: DupirePipelineConfig,
                     analyzer: PDFAnalyzer, T_values: list,
                     sigma: float = 1.0) -> Dict:
    """
    Compare NN-implied density against analytical lognormal density

    Uses automatic differentiation to extract NN density:
        f_NN(K) = e^(rT) · ∂²C_NN/∂K²

    Args:
        nn_phi: Trained option price network
        config: Pipeline configuration
        analyzer: PDF analyzer instance
        T_values: List of maturities to compare
        sigma: Analytical constant volatility

    Returns:
        Dictionary with density comparisons for each maturity
    """
    print(f"\n{'='*80}")
    print("COMPARISON 3: RISK-NEUTRAL DENSITY (f_NN vs f_lognormal)")
    print(f"{'='*80}")
    print(f"  Maturities: {T_values}")

    results = {}

    for T in T_values:
        print(f"\n  --- T = {T:.2f} ---")

        # Define strike grid
        K_grid = np.linspace(config.K_min, config.K_max, 200)

        # Extract NN-implied density using automatic differentiation
        print(f"    Extracting NN density via ∂²C/∂K²...")
        density_nn = analyzer.extract_model_implied_density(T, K_grid)

        # Compute analytical lognormal density
        print(f"    Computing analytical lognormal density...")
        density_analytical = lognormal_density(K_grid, config.S0, T, config.r, sigma)

        # Compute error metrics
        abs_error = np.abs(density_nn - density_analytical)
        # Relative error where density is significant
        mask = density_analytical > 1e-6
        rel_error = np.zeros_like(abs_error)
        rel_error[mask] = abs_error[mask] / density_analytical[mask]

        rmse = np.sqrt(np.mean((density_nn - density_analytical)**2))
        max_abs_error = np.max(abs_error)
        mean_rel_error = np.mean(rel_error[mask]) if np.any(mask) else 0.0

        # Verify moments
        moments = validate_lognormal_moments(K_grid, config.S0, T, config.r, sigma)

        print(f"      RMSE: {rmse:.6e}")
        print(f"      Max abs error: {max_abs_error:.6e}")
        print(f"      Mean rel error: {mean_rel_error:.4%}")
        print(f"      Analytical normalization: {moments['normalization']:.6f}")

        results[T] = {
            'K_grid': K_grid,
            'density_nn': density_nn,
            'density_analytical': density_analytical,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'rmse': rmse,
            'max_abs_error': max_abs_error,
            'mean_rel_error': mean_rel_error,
            'moments': moments
        }

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(price_results: Dict, vol_results: Dict, density_results: Dict,
                   output_dir: str, sigma: float = 1.0):
    """
    Generate comprehensive comparison plots

    Creates a multi-panel figure with:
    - Panel 1: Option price surfaces (NN vs BS)
    - Panel 2: Local volatility surfaces (NN vs constant)
    - Panel 3: Risk-neutral densities for multiple maturities
    - Panel 4: Error heatmaps

    Args:
        price_results: Option price comparison results
        vol_results: Local volatility comparison results
        density_results: Density comparison results
        output_dir: Directory to save plots
        sigma: Analytical volatility
    """
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*80}")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f'Neural Network vs Analytical Solution Comparison (σ = {sigma})\n' +
        'Trained Model vs Black-Scholes',
        fontsize=16, fontweight='bold', y=0.98
    )

    # =========================================================================
    # ROW 1: OPTION PRICES
    # =========================================================================

    # Panel 1a: NN Option Prices
    ax1 = plt.subplot(2, 4, 1, projection='3d')
    surf1 = ax1.plot_surface(price_results['K_mesh'], price_results['T_mesh'],
                             price_results['C_nn'], cmap='viridis', alpha=0.9)
    ax1.set_xlabel('Strike K')
    ax1.set_ylabel('Maturity T')
    ax1.set_zlabel('Call Price')
    ax1.set_title('NN Option Prices C_NN(K,T)', fontweight='bold')
    plt.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Panel 1b: Analytical Option Prices
    ax2 = plt.subplot(2, 4, 2, projection='3d')
    surf2 = ax2.plot_surface(price_results['K_mesh'], price_results['T_mesh'],
                             price_results['C_bs'], cmap='viridis', alpha=0.9)
    ax2.set_xlabel('Strike K')
    ax2.set_ylabel('Maturity T')
    ax2.set_zlabel('Call Price')
    ax2.set_title('Black-Scholes C_BS(K,T)', fontweight='bold')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # Panel 1c: Option Price Error
    ax3 = plt.subplot(2, 4, 3)
    im1 = ax3.contourf(price_results['K_mesh'], price_results['T_mesh'],
                       price_results['abs_error'], levels=20, cmap='hot')
    ax3.set_xlabel('Strike K')
    ax3.set_ylabel('Maturity T')
    ax3.set_title(f'Absolute Error |C_NN - C_BS|\nRMSE = {price_results["rmse"]:.4f}',
                  fontweight='bold')
    plt.colorbar(im1, ax=ax3)

    # Panel 1d: Relative Error
    ax4 = plt.subplot(2, 4, 4)
    im2 = ax4.contourf(price_results['K_mesh'], price_results['T_mesh'],
                       100 * price_results['rel_error'], levels=20, cmap='hot')
    ax4.set_xlabel('Strike K')
    ax4.set_ylabel('Maturity T')
    ax4.set_title(f'Relative Error (%)\nMean = {100*price_results["mean_rel_error"]:.2f}%',
                  fontweight='bold')
    plt.colorbar(im2, ax=ax4)

    # =========================================================================
    # ROW 2: LOCAL VOLATILITY & DENSITIES
    # =========================================================================

    # Panel 2a: NN Local Volatility
    ax5 = plt.subplot(2, 4, 5, projection='3d')
    surf3 = ax5.plot_surface(vol_results['K_mesh'], vol_results['T_mesh'],
                             vol_results['sigma_nn'], cmap='plasma', alpha=0.9)
    ax5.set_xlabel('Strike K')
    ax5.set_ylabel('Maturity T')
    ax5.set_zlabel('Volatility')
    ax5.set_title('NN Local Volatility σ_NN(K,T)', fontweight='bold')
    plt.colorbar(surf3, ax=ax5, shrink=0.5, aspect=5)

    # Panel 2b: Analytical Constant Volatility
    ax6 = plt.subplot(2, 4, 6, projection='3d')
    surf4 = ax6.plot_surface(vol_results['K_mesh'], vol_results['T_mesh'],
                             vol_results['sigma_analytical'], cmap='plasma', alpha=0.9)
    ax6.set_xlabel('Strike K')
    ax6.set_ylabel('Maturity T')
    ax6.set_zlabel('Volatility')
    ax6.set_title(f'Constant Volatility σ = {sigma}', fontweight='bold')
    plt.colorbar(surf4, ax=ax6, shrink=0.5, aspect=5)

    # Panel 2c: Volatility Error
    ax7 = plt.subplot(2, 4, 7)
    im3 = ax7.contourf(vol_results['K_mesh'], vol_results['T_mesh'],
                       vol_results['abs_error'], levels=20, cmap='hot')
    ax7.set_xlabel('Strike K')
    ax7.set_ylabel('Maturity T')
    ax7.set_title(f'Volatility Error |σ_NN - σ|\nRMSE = {vol_results["rmse"]:.4f}',
                  fontweight='bold')
    plt.colorbar(im3, ax=ax7)

    # Panel 2d: Density Comparisons
    ax8 = plt.subplot(2, 4, 8)
    for T, result in density_results.items():
        K = result['K_grid']
        ax8.plot(K, result['density_analytical'], '--', linewidth=2,
                label=f'T={T} (Analytical)', alpha=0.7)
        ax8.plot(K, result['density_nn'], '-', linewidth=2,
                label=f'T={T} (NN)', alpha=0.9)
    ax8.set_xlabel('Strike K')
    ax8.set_ylabel('Density f(K)')
    ax8.set_title('Risk-Neutral Densities', fontweight='bold')
    ax8.legend(fontsize=8, ncol=2)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    png_file = os.path.join(output_dir, f'comparison_nn_vs_bs_{timestamp}.png')
    pdf_file = os.path.join(output_dir, f'comparison_nn_vs_bs_{timestamp}.pdf')

    fig.savefig(png_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_file, dpi=DPI, bbox_inches='tight', facecolor='white')

    print(f"  ✓ Saved: {png_file}")
    print(f"  ✓ Saved: {pdf_file}")

    plt.close(fig)


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def save_summary_report(price_results: Dict, vol_results: Dict,
                       density_results: Dict, output_dir: str, sigma: float = 1.0):
    """
    Save numerical comparison results to JSON

    Args:
        price_results: Option price comparison
        vol_results: Volatility comparison
        density_results: Density comparison
        output_dir: Output directory
        sigma: Analytical volatility
    """
    print(f"\n{'='*80}")
    print("SAVING SUMMARY REPORT")
    print(f"{'='*80}")

    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'analytical_volatility': sigma,
        'option_prices': {
            'rmse': float(price_results['rmse']),
            'max_abs_error': float(price_results['max_abs_error']),
            'mean_rel_error': float(price_results['mean_rel_error'])
        },
        'local_volatility': {
            'rmse': float(vol_results['rmse']),
            'max_abs_error': float(vol_results['max_abs_error']),
            'mean_rel_error': float(vol_results['mean_rel_error'])
        },
        'densities': {}
    }

    for T, result in density_results.items():
        report['densities'][f'T={T}'] = {
            'rmse': float(result['rmse']),
            'max_abs_error': float(result['max_abs_error']),
            'mean_rel_error': float(result['mean_rel_error']),
            'normalization': float(result['moments']['normalization']),
            'mean_error': float(result['moments']['mean_error']),
            'var_error': float(result['moments']['var_error'])
        }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(output_dir, f'comparison_metrics_{timestamp}.json')

    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ Saved: {json_file}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main comparison workflow"""
    print(f"\n{'='*80}")
    print("NEURAL NETWORK vs ANALYTICAL SOLUTION COMPARISON")
    print(f"{'='*80}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Analytical volatility: σ = {SIGMA_ANALYTICAL}")

    # Load models and configuration
    nn_phi, nn_eta, metadata, config = load_models_and_config(MODEL_DIR)

    # Create analyzer
    analyzer = PDFAnalyzer(nn_phi, nn_eta, config, metadata)

    # Comparison 1: Option Prices
    price_results = compare_option_prices(nn_phi, config, analyzer, SIGMA_ANALYTICAL)

    # Comparison 2: Local Volatility
    vol_results = compare_local_volatility(nn_eta, config, analyzer, SIGMA_ANALYTICAL)

    # Comparison 3: Risk-Neutral Densities
    T_values = [0.5, 1.0, 1.5]  # Selected maturities
    density_results = compare_densities(nn_phi, config, analyzer, T_values, SIGMA_ANALYTICAL)

    # Generate plots
    plot_comparison(price_results, vol_results, density_results, MODEL_DIR, SIGMA_ANALYTICAL)

    # Save summary report
    save_summary_report(price_results, vol_results, density_results, MODEL_DIR, SIGMA_ANALYTICAL)

    print(f"\n{'='*80}")
    print("✓ COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {MODEL_DIR}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
