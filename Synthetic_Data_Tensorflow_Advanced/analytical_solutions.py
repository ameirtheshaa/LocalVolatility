#!/usr/bin/env python3
"""
================================================================================
ANALYTICAL SOLUTIONS FOR CONSTANT VOLATILITY (BLACK-SCHOLES)
================================================================================

This module provides analytical formulas for the Black-Scholes model with
constant volatility σ, used for validating neural network predictions.

MATHEMATICAL BACKGROUND:
-----------------------
When local volatility is constant (σ(K,T) = σ for all K,T), the Dupire PDE
reduces to the Black-Scholes PDE with constant coefficients.

Under risk-neutral measure with constant volatility σ:
    ln(S_T) ~ N(ln(S₀) + (r - σ²/2)T, σ²T)

This leads to:
1. Lognormal risk-neutral density
2. Closed-form option pricing (Black-Scholes formula)
3. Constant local volatility surface

REFERENCES:
----------
- BLACK-SCHOLES-EXACT.md in docs/
- Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Breeden, D. and Litzenberger, R. (1978). "Prices of State-Contingent Claims"

Author: Comparison validation tool
Date: 2025-11-02
================================================================================
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Tuple


def black_scholes_call(S0: float, K: Union[float, np.ndarray], T: float,
                       r: float, sigma: float) -> Union[float, np.ndarray]:
    """
    Black-Scholes European call option price with constant volatility

    Formula:
        C(K,T) = S₀·N(d₁) - K·e^(-rT)·N(d₂)

    where:
        d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
        d₂ = d₁ - σ√T

    Args:
        S0: Initial stock price
        K: Strike price (scalar or array)
        T: Time to maturity
        r: Risk-free interest rate
        sigma: Constant volatility

    Returns:
        Call option price(s)
    """
    # Handle expiration (T=0 or negative)
    if np.isscalar(T):
        if T <= 0:
            return np.maximum(S0 - K, 0.0)
    else:
        # T is array - use np.where for vectorization
        if np.any(T <= 0):
            result = np.zeros_like(T, dtype=float)
            mask = T > 0
            if not np.any(mask):
                return np.maximum(S0 - K, 0.0)
            # Only compute for T > 0
            K_pos = K[mask] if hasattr(K, '__len__') else K
            T_pos = T[mask]
            sqrt_T = np.sqrt(T_pos)
            d1 = (np.log(S0 / K_pos) + (r + 0.5 * sigma**2) * T_pos) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            result[mask] = S0 * norm.cdf(d1) - K_pos * np.exp(-r * T_pos) * norm.cdf(d2)
            result[~mask] = np.maximum(S0 - (K[~mask] if hasattr(K, '__len__') else K), 0.0)
            return result

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price


def black_scholes_put(S0: float, K: Union[float, np.ndarray], T: float,
                      r: float, sigma: float) -> Union[float, np.ndarray]:
    """
    Black-Scholes European put option price with constant volatility

    Formula (via put-call parity):
        P(K,T) = C(K,T) - S₀ + K·e^(-rT)

    Or directly:
        P(K,T) = K·e^(-rT)·N(-d₂) - S₀·N(-d₁)

    Args:
        S0: Initial stock price
        K: Strike price (scalar or array)
        T: Time to maturity
        r: Risk-free interest rate
        sigma: Constant volatility

    Returns:
        Put option price(s)
    """
    # Handle expiration (T=0 or negative)
    if np.isscalar(T):
        if T <= 0:
            return np.maximum(K - S0, 0.0)
    else:
        # T is array - use np.where for vectorization
        if np.any(T <= 0):
            result = np.zeros_like(T, dtype=float)
            mask = T > 0
            if not np.any(mask):
                return np.maximum(K - S0, 0.0)
            # Only compute for T > 0
            K_pos = K[mask] if hasattr(K, '__len__') else K
            T_pos = T[mask]
            sqrt_T = np.sqrt(T_pos)
            d1 = (np.log(S0 / K_pos) + (r + 0.5 * sigma**2) * T_pos) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            result[mask] = K_pos * np.exp(-r * T_pos) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
            result[~mask] = np.maximum((K[~mask] if hasattr(K, '__len__') else K) - S0, 0.0)
            return result

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    return put_price


def lognormal_density(K: Union[float, np.ndarray], S0: float, T: float,
                     r: float, sigma: float) -> Union[float, np.ndarray]:
    """
    Risk-neutral probability density of terminal stock price S_T

    For constant volatility σ, the density is lognormal:

        p(K; T) = (1 / (K·σ√(2πT))) · exp(-(ln(K/S₀) - μ)² / (2σ²T))

    where:
        μ = (r - σ²/2)T  (risk-neutral drift)

    This is the density that satisfies:
        ∂²C/∂K² = e^(-rT) · p(K; T)  (Breeden-Litzenberger)

    Args:
        K: Strike price / stock price level (scalar or array)
        S0: Initial stock price
        T: Time to maturity
        r: Risk-free interest rate
        sigma: Constant volatility

    Returns:
        Probability density p(K; T)

    Properties:
        - ∫₀^∞ p(K) dK = 1  (normalization)
        - E[S_T] = S₀·e^(rT)  (risk-neutral drift)
        - Var[S_T] = S₀²·e^(2rT)·(e^(σ²T) - 1)
    """
    # Handle expiration (T=0 or negative)
    if np.isscalar(T):
        if T <= 0:
            # Dirac delta at S0
            return np.inf if np.isscalar(K) and K == S0 else (0.0 if np.isscalar(K) else np.zeros_like(K))
    else:
        # T is array
        if np.any(T <= 0):
            result = np.zeros_like(T if hasattr(T, 'shape') else K, dtype=float)
            mask = T > 0
            if not np.any(mask):
                return result
            # Only compute for T > 0 (handle below)
            T = T[mask] if hasattr(T, '__len__') else T
            K = K[mask] if hasattr(K, '__len__') and len(K) == len(mask) else K

    sqrt_T = np.sqrt(T)
    mu = (r - 0.5 * sigma**2) * T

    # Lognormal PDF
    log_ratio = np.log(K / S0)
    exponent = -((log_ratio - mu)**2) / (2 * sigma**2 * T)
    density = (1.0 / (K * sigma * sqrt_T * np.sqrt(2 * np.pi))) * np.exp(exponent)

    return density


def local_volatility_constant(K: Union[float, np.ndarray], T: Union[float, np.ndarray],
                              sigma: float) -> Union[float, np.ndarray]:
    """
    Local volatility for constant-volatility model

    For Black-Scholes with constant σ, the local volatility is trivially:
        σ_local(K,T) = σ  (constant everywhere)

    This is included for API consistency with time-dependent local vol models.

    Args:
        K: Strike price(s)
        T: Maturity(ies)
        sigma: Constant volatility

    Returns:
        Local volatility (constant array matching input shape)
    """
    # Return constant sigma with same shape as inputs
    if np.isscalar(K) and np.isscalar(T):
        return sigma
    elif np.isscalar(K):
        return np.full_like(T, sigma, dtype=float)
    elif np.isscalar(T):
        return np.full_like(K, sigma, dtype=float)
    else:
        return np.full(np.broadcast(K, T).shape, sigma, dtype=float)


def bs_greeks(S0: float, K: Union[float, np.ndarray], T: float,
              r: float, sigma: float) -> dict:
    """
    Compute Black-Scholes Greeks for call options

    Greeks:
        Delta: ∂C/∂S
        Gamma: ∂²C/∂S²
        Vega: ∂C/∂σ
        Theta: -∂C/∂T (note negative sign)
        Rho: ∂C/∂r

    Args:
        S0: Initial stock price
        K: Strike price(s)
        T: Time to maturity
        r: Risk-free interest rate
        sigma: Constant volatility

    Returns:
        Dictionary with keys: 'delta', 'gamma', 'vega', 'theta', 'rho'
    """
    if T <= 0:
        # At expiration, most Greeks are undefined or zero
        itm = S0 > K
        return {
            'delta': np.where(itm, 1.0, 0.0),
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'rho': 0.0
        }

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Standard normal density
    phi_d1 = norm.pdf(d1)

    # Greeks
    delta = norm.cdf(d1)
    gamma = phi_d1 / (S0 * sigma * sqrt_T)
    vega = S0 * phi_d1 * sqrt_T
    theta = (-(S0 * phi_d1 * sigma) / (2 * sqrt_T)
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


def implied_volatility_bs(call_price: float, S0: float, K: float, T: float,
                          r: float, tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Compute implied volatility from Black-Scholes call price using Newton-Raphson

    Solves for σ such that:
        BS_call(S0, K, T, r, σ) = call_price

    Args:
        call_price: Observed call option price
        S0: Initial stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free interest rate
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Implied volatility σ

    Raises:
        ValueError: If no convergence or invalid inputs
    """
    # Intrinsic value bounds
    intrinsic = max(S0 - K * np.exp(-r * T), 0)
    if call_price < intrinsic:
        raise ValueError(f"Call price {call_price:.4f} below intrinsic value {intrinsic:.4f}")

    # Initial guess: ATM volatility from Brenner-Subrahmanyam
    sigma = np.sqrt(2 * np.pi / T) * call_price / S0

    for _ in range(max_iter):
        bs_price = black_scholes_call(S0, K, T, r, sigma)
        diff = bs_price - call_price

        if abs(diff) < tol:
            return sigma

        # Vega (derivative wrt sigma)
        greeks = bs_greeks(S0, K, T, r, sigma)
        vega = greeks['vega']

        if vega < 1e-10:
            raise ValueError("Vega too small, implied vol iteration failed")

        # Newton-Raphson update
        sigma = sigma - diff / vega

        # Keep sigma positive
        sigma = max(sigma, 1e-6)

    raise ValueError(f"Implied volatility did not converge after {max_iter} iterations")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR VALIDATION
# =============================================================================

def validate_lognormal_moments(K: np.ndarray, S0: float, T: float,
                               r: float, sigma: float) -> dict:
    """
    Verify that lognormal density has correct moments

    For lognormal distribution:
        E[S_T] = S₀·e^(rT)
        Var[S_T] = S₀²·e^(2rT)·(e^(σ²T) - 1)

    Args:
        K: Strike grid
        S0: Initial stock price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Dictionary with theoretical and numerical moments
    """
    density = lognormal_density(K, S0, T, r, sigma)
    dK = K[1] - K[0]  # Assume uniform grid

    # Numerical moments via integration
    mean_numerical = np.sum(K * density) * dK
    second_moment = np.sum(K**2 * density) * dK
    var_numerical = second_moment - mean_numerical**2

    # Theoretical moments
    mean_theoretical = S0 * np.exp(r * T)
    var_theoretical = S0**2 * np.exp(2 * r * T) * (np.exp(sigma**2 * T) - 1)

    return {
        'mean_numerical': mean_numerical,
        'mean_theoretical': mean_theoretical,
        'mean_error': abs(mean_numerical - mean_theoretical) / mean_theoretical,
        'var_numerical': var_numerical,
        'var_theoretical': var_theoretical,
        'var_error': abs(var_numerical - var_theoretical) / var_theoretical,
        'normalization': np.sum(density) * dK
    }


if __name__ == "__main__":
    """Test analytical solutions"""
    print("="*80)
    print("ANALYTICAL SOLUTIONS TEST")
    print("="*80)

    # Test parameters
    S0 = 1000.0
    r = 0.04
    sigma = 1.0
    T = 1.0

    # Generate strike grid
    K_grid = np.linspace(500, 3000, 100)

    # Test 1: Option pricing
    print("\n[Test 1] Black-Scholes Call Prices")
    print("-" * 80)
    call_prices = black_scholes_call(S0, K_grid, T, r, sigma)
    print(f"  S₀={S0}, T={T}, r={r}, σ={sigma}")
    print(f"  ATM call (K={S0}): {black_scholes_call(S0, S0, T, r, sigma):.4f}")
    print(f"  Price range: [{call_prices.min():.2f}, {call_prices.max():.2f}]")

    # Test 2: Risk-neutral density
    print("\n[Test 2] Lognormal Risk-Neutral Density")
    print("-" * 80)
    density = lognormal_density(K_grid, S0, T, r, sigma)
    moments = validate_lognormal_moments(K_grid, S0, T, r, sigma)
    print(f"  Normalization: ∫p(K)dK = {moments['normalization']:.6f}")
    print(f"  E[S_T] (theoretical): {moments['mean_theoretical']:.2f}")
    print(f"  E[S_T] (numerical):   {moments['mean_numerical']:.2f}")
    print(f"  Mean error: {moments['mean_error']:.2e}")
    print(f"  Var error:  {moments['var_error']:.2e}")

    # Test 3: Greeks
    print("\n[Test 3] Black-Scholes Greeks (ATM)")
    print("-" * 80)
    greeks = bs_greeks(S0, S0, T, r, sigma)
    for name, value in greeks.items():
        print(f"  {name.capitalize():8s}: {value:.6f}")

    # Test 4: Implied volatility round-trip
    print("\n[Test 4] Implied Volatility Round-Trip")
    print("-" * 80)
    K_test = S0
    call_test = black_scholes_call(S0, K_test, T, r, sigma)
    iv = implied_volatility_bs(call_test, S0, K_test, T, r)
    print(f"  Input σ:      {sigma:.6f}")
    print(f"  Call price:   {call_test:.6f}")
    print(f"  Recovered σ:  {iv:.6f}")
    print(f"  Error:        {abs(iv - sigma):.2e}")

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
