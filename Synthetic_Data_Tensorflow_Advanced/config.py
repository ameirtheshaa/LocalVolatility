#!/usr/bin/env python3
"""
================================================================================
DUPIRE PIPELINE CONFIGURATION
================================================================================

Configuration dataclasses for the Dupire local volatility pipeline.

This module contains all configuration settings separated from the main code:
- VolatilityConfig: Local volatility model parameters
- PlotConfig: Plotting options and output formats
- AnalysisConfig: PDF analysis and Monte Carlo settings
- DupirePipelineConfig: Master configuration combining all settings

QUICK START:
    >>> from config import DupirePipelineConfig
    >>> config = DupirePipelineConfig.quick_test()  # 100 epochs, ~2 minutes
    >>> config = DupirePipelineConfig.full_training()  # 30k epochs, production

CUSTOM VOLATILITY:
    >>> import numpy as np
    >>> def my_vol(t, x):
    ...     return 0.2 + 0.1 * np.sin(t) * x
    >>> from config import VolatilityConfig
    >>> vol_config = VolatilityConfig.custom(my_vol)
    >>> config = DupirePipelineConfig(volatility_config=vol_config)

Author: Enhanced for supervisor review
Date: 2025
================================================================================
"""

import datetime
import os
from typing import List, Optional, Callable
from dataclasses import dataclass, field


# =============================================================================
# [1] VOLATILITY CONFIGURATION
# =============================================================================

@dataclass
class VolatilityConfig:
    """
    Configuration for local volatility models σ(t, x)

    The local volatility function determines how volatility varies with time
    and stock price. This is the core of the Dupire model.

    SUPPORTED MODELS:
    -----------------
    1. 'dupire_exact': Original synthetic model (default)
       Formula: σ(t,x) = σ_base + y·exp(-y)
       where y = (t + t_shift)·√(x + x_shift)

       This model has:
       - Smooth surface (no arbitrage)
       - Known analytical form (easy to validate)
       - Non-constant volatility (captures local volatility effects)

       This is the standard model for synthetic data generation and testing.

    2. 'custom': Provide your own callable function
       Signature: σ(t, x) -> float
       where t = time, x = S/S₀ (normalized stock price)

       Use this for advanced customization or real market data calibration.

    PARAMETERS:
    -----------
    For all models:
        x = S/S₀     : Normalized stock price (moneyness)
        t            : Time in years
        σ_base       : Base volatility level (e.g., 0.3 = 30% annualized)

    EXAMPLE USAGE:
    --------------
    # Use default Dupire exact model
    >>> vol_config = VolatilityConfig.dupire_exact()

    # Customize parameters
    >>> vol_config = VolatilityConfig.dupire_exact(sigma_base=0.25, t_shift=0.2)

    # Custom volatility
    >>> import numpy as np
    >>> def my_vol(t, x):
    ...     return 0.2 + 0.1 * np.exp(-t) * (x - 1.0)**2
    >>> vol_config = VolatilityConfig.custom(my_vol)
    """

    model_type: str = 'dupire_exact'  # 'dupire_exact' or 'custom'

    # =========================================================================
    # DUPIRE EXACT MODEL PARAMETERS
    # =========================================================================
    # Model: σ = σ_base + y·exp(-y) where y = (t + t_shift)·√(x + x_shift)
    #
    # This is a synthetic model with known properties:
    # - No arbitrage (smooth and positive)
    # - Non-constant (captures local volatility effects)
    # - Analytically tractable (easy to compute exact option prices via MC)

    sigma_base: float = 0.3         # Base volatility (30% annualized)
                                     # Typical range: 0.15 - 0.50 (15% - 50%)

    t_shift: float = 0.1            # Time shift to avoid singularity at t=0
                                     # Ensures y > 0 even at t=0
                                     # Typical range: 0.05 - 0.20

    x_shift: float = 0.1            # Moneyness shift to avoid singularity at x=0
                                     # Ensures y > 0 even for deep OTM
                                     # Typical range: 0.05 - 0.20

    # =========================================================================
    # CUSTOM MODEL
    # =========================================================================
    custom_volatility_func: Optional[Callable] = None
    # Custom function with signature: σ(t, x) -> float
    #
    # Requirements:
    # - Must return positive values (volatility > 0)
    # - Should be smooth (no jumps or discontinuities)
    # - Should avoid arbitrage (monotonic in strike for fixed time)
    #
    # Example:
    #   def my_vol(t, x):
    #       return 0.2 + 0.05 * np.exp(-t) + 0.1 * (x - 1.0)**2

    def __post_init__(self):
        """Validate configuration after initialization"""
        valid_models = ['dupire_exact', 'custom']
        if self.model_type not in valid_models:
            raise ValueError(
                f"model_type must be one of {valid_models}, got '{self.model_type}'"
            )

        if self.model_type == 'custom' and self.custom_volatility_func is None:
            raise ValueError(
                "custom_volatility_func must be provided when model_type='custom'\n"
                "Example: VolatilityConfig.custom(lambda t, x: 0.2 + 0.1*x)"
            )

        # Validate parameters
        if self.sigma_base <= 0:
            raise ValueError(f"sigma_base must be positive, got {self.sigma_base}")

    @classmethod
    def dupire_exact(cls, sigma_base: float = 0.3, t_shift: float = 0.1, x_shift: float = 0.1):
        """
        Preset: Original Dupire exact model (synthetic data)

        Formula: σ(t,x) = 0.3 + y·exp(-y) where y = (t + 0.1)·√(x + 0.1)

        This is the default model for synthetic data generation and testing.
        It has a known closed form and produces smooth volatility surfaces.

        Parameters:
        -----------
        sigma_base : float, default=0.3
            Base volatility level (30% annualized)
        t_shift : float, default=0.1
            Time shift to avoid singularity at t=0
        x_shift : float, default=0.1
            Moneyness shift to avoid singularity at x=0

        Returns:
        --------
        VolatilityConfig configured for Dupire exact model
        """
        return cls(
            model_type='dupire_exact',
            sigma_base=sigma_base,
            t_shift=t_shift,
            x_shift=x_shift
        )

    @classmethod
    def custom(cls, volatility_func: Callable):
        """
        Preset: Custom volatility function

        Use this when you have your own volatility model.

        Parameters:
        -----------
        volatility_func : Callable
            Function with signature σ(t, x) -> float
            where:
                t : float or np.ndarray - time in years
                x : float or np.ndarray - normalized stock price S/S₀
                returns: volatility (must be positive)

        Example:
        --------
        >>> import numpy as np
        >>> def heston_vol(t, x):
        ...     # Heston-inspired local volatility
        ...     kappa, theta, v0 = 2.0, 0.04, 0.04
        ...     v_t = theta + (v0 - theta) * np.exp(-kappa * t)
        ...     leverage = 1.0 + 0.5 * (1.0 - x)  # Vol increases when price drops
        ...     return np.sqrt(v_t) * leverage
        >>>
        >>> vol_config = VolatilityConfig.custom(heston_vol)

        Returns:
        --------
        VolatilityConfig configured with custom function
        """
        return cls(
            model_type='custom',
            custom_volatility_func=volatility_func
        )


# =============================================================================
# [2] PLOT CONFIGURATION
# =============================================================================

@dataclass
class PlotConfig:
    """
    Control which plots to generate (granular control)

    Set any flag to False to disable that plot category.
    This is useful for:
    - Speed (skip expensive visualizations)
    - Storage (reduce output file size)
    - Focus (only generate relevant plots)

    CATEGORIES:
    -----------
    1. Training plots - Generated during model training
    2. PDF analysis plots - Generated during analysis stage
    3. Output formats - PNG, PDF, SVG

    EXAMPLE USAGE:
    --------------
    # Disable all training plots for faster training
    >>> plot_config = PlotConfig(enable_training_plots=False)

    # Only generate PDF analysis plots
    >>> plot_config = PlotConfig(
    ...     enable_training_plots=False,
    ...     enable_pdf_plots=True
    ... )

    # High-quality plots for publication
    >>> plot_config = PlotConfig(
    ...     dpi=600,
    ...     save_pdf=True,
    ...     save_svg=True
    ... )
    """

    # =========================================================================
    # TRAINING PLOTS (generated during model training)
    # =========================================================================
    enable_training_plots: bool = True
    # Master switch for all training plots
    # Set to False to skip all training visualizations (faster training)

    # Option price surface plots
    plot_option_surfaces: bool = True
    # 3D surface plots of option prices C(T,K)
    # Shows learned option price surface vs exact

    plot_option_error: bool = True
    # Error surface: |C_NN - C_exact|
    # Useful for identifying regions where NN struggles

    # Local volatility surface plots
    plot_volatility_surfaces: bool = True
    # 3D surface plots of local volatility σ(T,K)
    # Compares NN prediction vs true volatility

    plot_volatility_error: bool = True
    # Error surface: |σ_NN - σ_exact|
    # Key metric for calibration quality

    # PDE and gradient plots
    plot_dupire_gradients: bool = True
    # Visualization of PDE terms: ∂C/∂T, ∂²C/∂K², etc.
    # Useful for debugging PDE enforcement

    # Training metrics
    plot_loss_curves: bool = True
    # Loss over epochs (data loss, PDE loss, regularization)
    # Essential for monitoring training progress

    plot_error_metrics: bool = True
    # RMSE and relative error over epochs
    # Shows convergence behavior

    # =========================================================================
    # PDF ANALYSIS PLOTS (generated during analysis stage)
    # =========================================================================
    enable_pdf_plots: bool = True
    # Master switch for all PDF analysis plots
    # Set to False to skip density analysis (faster)

    # Distribution comparison plots (three-panel plots)
    plot_pdf_k_space: bool = True
    # Panel 1: Density in strike space f(K)
    # Compares MC histogram vs NN-implied density

    plot_pdf_log_space: bool = True
    # Panel 2: Density in log-strike space f(ln K)
    # Better for seeing log-normal behavior

    plot_pdf_gaussian_space: bool = True
    # Panel 3: Density in standardized Gaussian space g(x)
    # Tests if distribution is truly log-normal

    # =========================================================================
    # OUTPUT OPTIONS
    # =========================================================================
    save_png: bool = True
    # Save plots as PNG (raster format)
    # Good for: presentations, quick viewing, smaller file size

    save_pdf: bool = True
    # Save plots as PDF (vector format)
    # Good for: publications, LaTeX documents, scalable graphics

    save_svg: bool = False
    # Save plots as SVG (vector format)
    # Good for: web, editing in Inkscape/Illustrator
    # Note: Larger file size than PDF

    dpi: int = 450
    # Resolution for raster formats (PNG)
    # Typical values:
    #   - 150: Draft quality (fast, small files)
    #   - 300: Print quality (standard)
    #   - 450: High quality (recommended for papers)
    #   - 600: Publication quality (Nature, Science)

    # Plotting style
    style: str = 'publication'
    # Options: 'publication', 'presentation'
    # Future enhancement: different color schemes and layouts


# =============================================================================
# [3] ANALYSIS CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """
    Control which analyses to run during Stage 3 (PDF Analysis)

    This stage validates the trained model by:
    1. Extracting risk-neutral density from NN option prices
    2. Running Monte Carlo with NN-predicted local volatility
    3. Comparing densities in multiple coordinate systems
    4. Computing statistical moments (mean, variance, skewness, kurtosis)

    WHAT EACH ANALYSIS DOES:
    ------------------------
    - extract_density_from_option_prices:
        Uses automatic differentiation to compute f(K) = e^(rT) ∂²C/∂K²
        This is the risk-neutral probability density

    - fit_lognormal_distribution:
        Fits a log-normal distribution to the extracted density
        Used to test if the model produces realistic distributions

    - compute_higher_moments:
        Computes skewness and excess kurtosis
        Standard normal has skew=0, ex.kurtosis=0
        Deviations indicate non-Gaussian behavior

    - run_mc_with_nn_volatility:
        Runs Monte Carlo simulation using σ_NN(t,S) from the neural network
        This is the ultimate validation: does the NN produce correct prices?

    MONTE CARLO PARAMETERS:
    -----------------------
    - n_paths_analysis: Number of MC paths (more = smoother density)
        Typical: 25,000 (good balance), 50,000 (high quality), 100,000 (research)

    - dt_analysis: Time step for MC simulation
        Smaller = more accurate but slower
        Typical: 1e-3 (0.001 years ≈ 0.365 days)

    - T_analysis: List of maturities to analyze
        Default: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0] years
        Choose maturities within training range
    """

    # =========================================================================
    # DENSITY EXTRACTION
    # =========================================================================
    extract_density_from_option_prices: bool = True
    # Extract f(K) = e^(rT) ∂²C/∂K² using automatic differentiation
    # This is the model-implied risk-neutral density
    # Required for PDF comparison plots

    # =========================================================================
    # STATISTICAL FITTING
    # =========================================================================
    fit_lognormal_distribution: bool = True
    # Fit log-normal parameters (μ, σ) from extracted density
    # Uses corrected method of moments:
    #   σ² = ln(CV² + 1) where CV = StdDev/Mean
    #   μ = ln(Mean) - σ²/2

    compute_higher_moments: bool = True
    # Compute skewness and excess kurtosis
    # Useful for detecting deviations from log-normal assumption
    # Standard normal: skewness=0, excess kurtosis=0

    # =========================================================================
    # MONTE CARLO VALIDATION
    # =========================================================================
    run_mc_with_nn_volatility: bool = True
    # Run MC simulation with NN-predicted local volatility σ_NN(t,S)
    # This validates that NN produces correct option prices
    # SDE: dS_t = r S_t dt + σ_NN(t, S_t) S_t dW_t

    # Monte Carlo parameters
    n_paths_analysis: int = 10**6
    # Number of Monte Carlo paths for analysis
    # Default: 10**6 (1,000,000) - matches M_train for consistency
    # More paths = smoother density estimate
    # Recommended:
    #   - Quick test: 5,000
    #   - Standard: 25,000
    #   - High quality: 50,000
    #   - Research: 100,000+
    # Note: When reuse_training_mc=True, this is ignored and M_train is used instead

    dt_analysis: float = 1e-3
    # Time step for MC simulation (in years)
    # Smaller = more accurate but slower
    # 1e-3 years ≈ 0.365 days (good balance)
    # For high accuracy: 1e-4 years ≈ 0.0365 days

    T_analysis: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    # Maturities to analyze (in years)
    # Should be within training range [T_min, T_max]
    # Default: 3 months, 6 months, 9 months, 1 year, 1.5 years, 2 years
    # Customize as needed:
    #   Short-term: [0.1, 0.25, 0.5]
    #   Long-term: [1.0, 2.0, 3.0]

    reuse_training_mc: bool = True
    # If True and training_data.npz exists, reuse the exact MC paths from training
    # This is much faster and uses M_train paths (typically 1,000,000)
    # If False, always run new MC simulation with NN volatility using n_paths_analysis
    # Benefits of reusing:
    #   - Instant (no new simulation needed)
    #   - More samples (M_train >> n_paths_analysis typically)
    #   - Direct comparison against exact σ(t,x) used in training


# =============================================================================
# [4] MASTER PIPELINE CONFIGURATION
# =============================================================================

@dataclass
class DupirePipelineConfig:
    """
    Master configuration - combines all settings

    This is the main configuration class that controls the entire pipeline.
    It combines settings for:
    - Market parameters (S0, r, strike/maturity ranges)
    - Volatility model (Dupire exact, smile surface, or custom)
    - Data generation (Monte Carlo parameters)
    - Model training (neural network architecture, optimization)
    - Analysis (PDF extraction, validation)
    - Plotting (output formats, quality)

    PRESET CONFIGURATIONS:
    ----------------------
    Use these for common scenarios:

    1. Quick Test (100 epochs, ~2 minutes):
       >>> config = DupirePipelineConfig.quick_test()

    2. Full Training (30,000 epochs, ~30-60 minutes):
       >>> config = DupirePipelineConfig.full_training()

    3. Analysis Only (load existing models):
       >>> config = DupirePipelineConfig.analysis_only('path/to/models')

    EXECUTION MODES:
    ----------------
    - 'all': Run all stages (Generate → Train → Analyze)
    - 'generate': Only generate training data
    - 'train': Only train models (requires existing data)
    - 'analyze': Only analyze models (requires trained models)

    EXAMPLE CUSTOMIZATION:
    ----------------------
    >>> from config import DupirePipelineConfig, VolatilityConfig
    >>>
    >>> # Custom configuration with smile surface
    >>> config = DupirePipelineConfig(
    ...     volatility_config=VolatilityConfig.smile_surface(),
    ...     num_epochs=10000,
    ...     lambda_pde=2.0,
    ...     output_dir='my_experiment'
    ... )
    """

    # =========================================================================
    # EXECUTION CONTROL
    # =========================================================================
    mode: str = 'all'
    # Pipeline execution mode
    # Options:
    #   'all' - Run all stages: Generate → Train → Analyze
    #   'generate' - Only generate training data (Stage 1)
    #   'train' - Only train models (Stage 2, requires data)
    #   'analyze' - Only analyze existing models (Stage 3, requires models)

    skip_if_exists: bool = True
    # Skip stages if output already exists
    # True: Don't regenerate data/models if they exist (faster)
    # False: Always regenerate (useful for testing parameter changes)

    verbose: bool = True
    # Print detailed progress information
    # True: Show all status messages
    # False: Minimal output

    # =========================================================================
    # MARKET PARAMETERS
    # =========================================================================
    S0: float = 1000.0
    # Initial stock price (arbitrary units)
    # This sets the scale for strikes and option prices
    # Typical values: 100.0, 1000.0, or match real market (e.g., S&P 500 ≈ 4500)

    r: float = 0.04
    # Risk-free interest rate (annualized)
    # 0.04 = 4% per year (typical historical value)
    # Adjust based on current market:
    #   Low rate environment: 0.01 - 0.02
    #   Normal rates: 0.03 - 0.05
    #   High rate environment: 0.05+

    # =========================================================================
    # VOLATILITY MODEL
    # =========================================================================
    volatility_config: VolatilityConfig = field(default_factory=VolatilityConfig.dupire_exact)
    # Local volatility model σ(t, x)
    # Default: Dupire exact model (synthetic data)
    #
    # To customize:
    #   - Custom: VolatilityConfig.custom(your_function)

    # =========================================================================
    # STAGE 1: DATA GENERATION
    # =========================================================================
    data_generation: bool = True
    # Enable/disable data generation stage
    # Set to False if you only want to train on existing data

    M_train: int = 10**6  # = 10,000
    # Number of Monte Carlo paths for training data generation
    # More paths = more accurate option prices, but slower
    # Recommended:
    #   Quick test: 1,000
    #   Standard: 10,000
    #   High quality: 50,000

    N_t_train: int = 1000
    # Number of time steps in Monte Carlo simulation
    # More steps = more accurate SDE discretization
    # 1000 steps with dt=1e-3 → max time = 1.0 years

    dt_train: float = 1e-3
    # Time step size for MC simulation (in years)
    # 1e-3 years ≈ 0.365 days
    # Smaller = more accurate but slower

    N_strikes: int = 20
    # Number of strikes in training grid
    # These are uniformly spaced between K_min and K_max
    # More strikes = more training data, better calibration

    N_maturities: int = 10
    # Number of maturities in training grid
    # These are selected from the MC time grid
    # More maturities = better temporal coverage

    K_min: float = 500.0
    # Minimum strike price
    # Should be well below S0 to capture deep ITM calls

    K_max: float = 3000.0
    # Maximum strike price
    # Should be well above S0 to capture deep OTM calls

    T_min: float = 0.3
    # Minimum maturity (years)
    # Avoid very short maturities (numerical instability)

    T_max: float = 1.5
    # Maximum maturity (years)
    # Also used for coordinate scaling: t_tilde = T / T_max

    save_training_data: bool = True
    # Save training data to disk (training_data.npz)
    # Allows reuse without regenerating

    # =========================================================================
    # STAGE 2: NEURAL NETWORK TRAINING
    # =========================================================================
    training: bool = True
    # Enable/disable training stage
    # Set to False for analysis-only mode

    num_epochs: int = 30000
    # Number of training iterations
    # More epochs = better convergence, but diminishing returns
    # Recommended:
    #   Quick test: 100
    #   Development: 1,000 - 5,000
    #   Production: 30,000
    #   Research: 50,000+

    print_epochs: int = 2500
    # Print progress every N epochs
    # Useful for monitoring without cluttering output

    save_epochs: int = 10000
    # Save checkpoints every N epochs
    # Useful for long training runs (can resume if interrupted)

    # -------------------------------------------------------------------------
    # Neural Network Architecture
    # -------------------------------------------------------------------------
    num_res_blocks: int = 3
    # Number of residual blocks in each network
    # More blocks = more expressive, but slower and risk overfitting
    # Recommended: 2-5

    units: int = 64
    # Number of neurons per layer
    # More units = more capacity
    # Typical: 32, 64, 128

    activation: str = 'tanh'
    # Activation function
    # Options: 'tanh', 'relu', 'elu', 'swish'
    # 'tanh' is standard for this problem

    gaussian_noise_phi: float = 0.5
    # Gaussian noise added to option price network input
    # Acts as regularization (prevents overfitting)
    # Typical: 0.1 - 1.0

    gaussian_noise_eta: float = 0.5
    # Gaussian noise added to volatility network input
    # Typical: 0.1 - 1.0

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------
    lr_phi: float = 1e-4
    # Learning rate for option price network (NN_phi)
    # 1e-4 = 0.0001 (good default)
    # If training is unstable: reduce to 1e-5
    # If training is too slow: increase to 5e-4

    lr_eta: float = 1e-4
    # Learning rate for volatility network (NN_eta)
    # Often set to lr_phi / 10 (volatility is harder to learn)

    lr_decay_rate: float = 1.1
    # Learning rate decay factor
    # Every lr_decay_steps, lr = lr / lr_decay_rate
    # 1.1 = 10% reduction

    lr_decay_steps: int = 2000
    # Apply learning rate decay every N epochs
    # Helps convergence in later stages

    # -------------------------------------------------------------------------
    # Loss Function Weights
    # -------------------------------------------------------------------------
    lambda_pde: float = 1.0
    # Weight for Dupire PDE loss
    # Controls how strongly we enforce PDE: ∂C/∂T = (1/2)K²σ² ∂²C/∂K²
    #
    # Higher values:
    #   + Better PDE satisfaction
    #   - May hurt data fitting
    #
    # Recommended:
    #   Start with 1.0
    #   If PDE residual is high: increase to 2.0 - 5.0
    #   If data fit is poor: decrease to 0.5

    lambda_reg: float = 1.0
    # Weight for arbitrage regularization loss
    # Penalizes violations of ∂C/∂T ≥ 0 (calendar spread arbitrage)
    #
    # Higher values:
    #   + Stronger arbitrage-free constraints
    #   - May hurt flexibility
    #
    # Recommended: 1.0 (usually sufficient)

    # =========================================================================
    # STAGE 3: ANALYSIS
    # =========================================================================
    analysis: bool = True
    # Enable/disable analysis stage
    # Set to False if you only want to generate data and train

    analysis_config: AnalysisConfig = field(default_factory=AnalysisConfig)
    # Configuration for PDF analysis and validation
    # See AnalysisConfig class for details

    plot_config: PlotConfig = field(default_factory=PlotConfig)
    # Configuration for plot generation
    # See PlotConfig class for details

    # =========================================================================
    # OUTPUT PATHS
    # =========================================================================
    output_dir: str = None
    # Directory for saving all outputs
    # If None, auto-generated with timestamp under models/runs/:
    #   models/runs/synthetic_data_3resblock_20250129_143022
    #
    # Manually specify for organized experiments:
    #   output_dir='models/runs/experiment_1_dupire_exact'

    @staticmethod
    def normalize_output_dir(path: Optional[str]) -> Optional[str]:
        """Normalize output directories to live under models/runs when relative."""
        if path is None:
            return None
        if os.path.isabs(path):
            return os.path.normpath(path)
        normalized = path.replace('\\', '/')
        if '/' not in normalized:
            path = os.path.join('models', 'runs', normalized)
        return os.path.normpath(path)

    def __post_init__(self):
        """Set default output directory if not specified"""
        if self.output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f'synthetic_data_{self.num_res_blocks}resblock_{timestamp}'
            self.output_dir = os.path.join('models', 'runs', run_name)
        self.output_dir = self.normalize_output_dir(self.output_dir)

    # =========================================================================
    # PRESET CONFIGURATIONS
    # =========================================================================

    @classmethod
    def quick_test(cls):
        """
        Preset for quick testing (reduced parameters)

        Use this for:
        - Testing code changes
        - Debugging
        - Quick experimentation

        Parameters:
        -----------
        - 100 epochs (~2 minutes on CPU, ~30 seconds on GPU)
        - 1,000 MC paths (vs 10,000 in full)
        - 6 strikes × 3 maturities (vs 20 × 10 in full)
        - 5,000 analysis paths (vs 25,000 in full)
        - Fewer plots (only essential ones)

        Expected results:
        -----------------
        - Volatility RMSE: ~10-20% (rough approximation)
        - Loss will not fully converge
        - PDF plots will be noisy

        This is NOT suitable for final results!
        """
        return cls(
            num_epochs=100,
            M_train=1000,
            N_strikes=6,
            N_maturities=3,
            print_epochs=50,
            save_epochs=100,
            analysis_config=AnalysisConfig(
                n_paths_analysis=5000,
                T_analysis=[0.5, 1.0, 1.5],
            ),
            plot_config=PlotConfig(
                plot_option_error=False,
                plot_dupire_gradients=False,
            )
        )

    @classmethod
    def full_training(cls):
        """
        Preset for full production training

        Use this for:
        - Final results
        - Publication-quality outputs
        - Research experiments

        Parameters:
        -----------
        - 30,000 epochs (~30-60 minutes on CPU, ~10-15 minutes on GPU)
        - 10,000 MC paths
        - 20 strikes × 10 maturities
        - 25,000 analysis paths
        - All plots enabled

        Expected results:
        -----------------
        - Volatility RMSE: <1% (excellent fit)
        - Smooth convergence
        - High-quality PDF agreement
        """
        return cls(
            num_epochs=30000,
            M_train=10**4,
        )

    @classmethod
    def analysis_only(cls, model_dir: str):
        """
        Preset for analysis only (requires existing trained models)

        Use this when you have trained models and want to:
        - Re-run analysis with different parameters
        - Generate additional plots
        - Test different maturities

        Parameters:
        -----------
        model_dir : str
            Path to directory containing NN_phi_final.keras and NN_eta_final.keras

        Example:
        --------
        >>> config = DupirePipelineConfig.analysis_only(
        ...     'synthetic_data_3resblock_20250129_143022'
        ... )
        >>>
        >>> # Customize analysis
        >>> config.analysis_config.T_analysis = [0.5, 1.0, 1.5, 2.0]
        >>> config.analysis_config.n_paths_analysis = 50000  # Higher quality
        """
        return cls(
            mode='analyze',
            data_generation=False,
            training=False,
            analysis=True,
            output_dir=model_dir,
        )


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DUPIRE PIPELINE CONFIGURATION - USAGE EXAMPLES")
    print("=" * 80)

    # Example 1: Default configuration
    print("\n[Example 1] Default configuration (Dupire exact):")
    print("-" * 80)
    config1 = DupirePipelineConfig()
    print(f"Volatility model: {config1.volatility_config.model_type}")
    print(f"σ_base = {config1.volatility_config.sigma_base}")
    print(f"Epochs = {config1.num_epochs:,}")
    print(f"Output: {config1.output_dir}")

    # Example 2: Custom volatility function
    print("\n[Example 2] Custom volatility configuration:")
    print("-" * 80)
    def my_volatility(t, x):
        """Custom volatility: constant 20%"""
        import numpy as np
        return 0.20 * np.ones_like(t)

    config2_custom = DupirePipelineConfig(
        volatility_config=VolatilityConfig.custom(my_volatility)
    )
    print(f"Volatility model: {config2_custom.volatility_config.model_type}")
    print(f"Custom function: {config2_custom.volatility_config.custom_volatility_func.__name__}")

    # Example 3: Quick test preset
    print("\n[Example 3] Quick test preset:")
    print("-" * 80)
    config4 = DupirePipelineConfig.quick_test()
    print(f"Epochs: {config4.num_epochs}")
    print(f"Training paths: {config4.M_train:,}")
    print(f"Analysis paths: {config4.analysis_config.n_paths_analysis:,}")
    print(f"Strikes × Maturities: {config4.N_strikes} × {config4.N_maturities}")

    # Example 4: Production configuration
    print("\n[Example 4] Full production configuration:")
    print("-" * 80)
    config5 = DupirePipelineConfig.full_training()
    print(f"Epochs: {config5.num_epochs:,}")
    print(f"Training paths: {config5.M_train:,}")
    print(f"λ_PDE = {config5.lambda_pde}")
    print(f"Residual blocks: {config5.num_res_blocks}")

    print("\n" + "=" * 80)
    print("✓ Configuration module loaded successfully!")
    print("=" * 80)
    print("\nTo use in your code:")
    print("  from config import DupirePipelineConfig")
    print("  config = DupirePipelineConfig.quick_test()")
    print("  # or")
    print("  config = DupirePipelineConfig.full_training()")
    print("=" * 80 + "\n")
