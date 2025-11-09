# Mathematical Treatment: Dupire Local Volatility Model with Neural Networks

**A complete mathematical derivation of the Dupire local volatility model and its neural network implementation**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [The Dupire Local Volatility Model](#2-the-dupire-local-volatility-model)
3. [Dupire's Partial Differential Equation](#3-dupires-partial-differential-equation)
4. [Coordinate Transformations and Scaling](#4-coordinate-transformations-and-scaling)
5. [Neural Network Formulation](#5-neural-network-formulation)
6. [Loss Function Design](#6-loss-function-design)
7. [Risk-Neutral Density Extraction](#7-risk-neutral-density-extraction)
8. [Monte Carlo Validation](#8-monte-carlo-validation)
9. [Statistical Analysis](#9-statistical-analysis)
10. [Arbitrage-Free Constraints](#10-arbitrage-free-constraints)
11. [References](#11-references)

---

## 1. Introduction

### 1.1 Motivation

In the **Black-Scholes model**, volatility is assumed to be constant:

```
dS_t = μ S_t dt + σ S_t dW_t
```

However, market-observed option prices exhibit the **volatility smile**: implied volatility varies with strike K and maturity T. The **Dupire local volatility model** (Dupire, 1994) extends Black-Scholes by allowing volatility to depend on both time and stock price:

```
dS_t = r S_t dt + σ(t, S_t) S_t dW_t
```

where σ(t, S) is the **local volatility function**.

### 1.2 The Calibration Problem

**Given**: Market option prices C_market(K, T) for various strikes K and maturities T

**Find**: The local volatility function σ(t, S) that reproduces these prices

**Challenge**: This is an ill-posed inverse problem (non-unique, unstable to noise)

### 1.3 Neural Network Approach

Instead of traditional finite difference methods, we use **physics-informed neural networks (PINNs)**:

- Two neural networks: NN_φ for option prices, NN_η for local volatility
- Trained to satisfy Dupire's PDE as a soft constraint
- Automatic differentiation provides derivatives
- No discretization grid needed

---

## 2. The Dupire Local Volatility Model

### 2.1 Asset Price Dynamics

The stock price S_t evolves according to the stochastic differential equation (SDE):

$$
dS_t = r S_t \, dt + \sigma(t, S_t) S_t \, dW_t, \quad S_0 \text{ given}
$$

where:
- **S_t**: Stock price at time t
- **r**: Risk-free interest rate (constant)
- **σ(t, S)**: Local volatility function (deterministic, time and price dependent)
- **W_t**: Standard Brownian motion under the risk-neutral measure ℚ

**Key Properties**:
1. **Markovian**: Future evolution depends only on current state (t, S_t)
2. **Complete market**: All derivatives can be perfectly hedged
3. **Risk-neutral pricing**: Drift term is r (no arbitrage condition)

### 2.2 European Option Pricing

European call and put options have payoffs:

$$
\text{Call payoff: } \max(S_T - K, 0) = (S_T - K)^+
$$

$$
\text{Put payoff: } \max(K - S_T, 0) = (K - S_T)^+
$$

Under the risk-neutral measure, option prices are:

$$
C(K, T) = e^{-rT} \mathbb{E}^\mathbb{Q}[(S_T - K)^+]
$$

$$
P(K, T) = e^{-rT} \mathbb{E}^\mathbb{Q}[(K - S_T)^+]
$$

where:
- **K**: Strike price
- **T**: Maturity (time to expiration)
- **𝔼^ℚ[·]**: Expectation under risk-neutral measure

### 2.3 Put-Call Parity

European calls and puts satisfy:

$$
C(K, T) - P(K, T) = e^{-rT}(F_0 - K)
$$

where **F_0 = S_0 e^{rT}** is the forward price. This holds for **any** volatility model (model-independent).

---

## 3. Dupire's Partial Differential Equation

### 3.1 The Fokker-Planck Equation

The transition probability density p(S, T | S_0, 0) satisfies the **Fokker-Planck equation**:

$$
\frac{\partial p}{\partial T} = -r \frac{\partial}{\partial S}(S p) + \frac{1}{2} \frac{\partial^2}{\partial S^2}(\sigma^2(T, S) S^2 p)
$$

This describes the evolution of the probability distribution of S_T.

### 3.2 Derivation of Dupire's PDE

**Theorem (Dupire, 1994)**: The option price π(K, T) ≡ C(K, T) satisfies:

$$
\frac{\partial \pi}{\partial T} = \frac{1}{2} K^2 \sigma^2(T, K) \frac{\partial^2 \pi}{\partial K^2} - r K \frac{\partial \pi}{\partial K}
$$

with initial and boundary conditions:

**Call options**:
$$
\pi^c(K, 0) = (S_0 - K)^+ \quad \text{(initial condition)}
$$

$$
\lim_{K \to 0} \pi^c(K, T) = S_0 e^{-rT} \quad \text{(deep ITM)}
$$

$$
\lim_{K \to \infty} \pi^c(K, T) = 0 \quad \text{(deep OTM)}
$$

**Put options**:
$$
\pi^p(K, 0) = (K - S_0)^+ \quad \text{(initial condition)}
$$

$$
\lim_{K \to 0} \pi^p(K, T) = 0 \quad \text{(deep OTM)}
$$

$$
\lim_{K \to \infty} \pi^p(K, T) = K e^{-rT} \quad \text{(deep ITM)}
$$

### 3.3 Proof Sketch

The key insight is that strike K plays the role of a "space variable" in the Fokker-Planck equation. By the **Breeden-Litzenberger formula**, the risk-neutral density is:

$$
f(K) = e^{rT} \frac{\partial^2 C}{\partial K^2}(K, T)
$$

Differentiating this with respect to T and using the Fokker-Planck equation yields Dupire's PDE.

**Detailed steps**:

1. Start with the pricing formula:
   $$
   C(K, T) = e^{-rT} \int_K^\infty (S - K) p(S, T | S_0, 0) \, dS
   $$

2. Differentiate with respect to T:
   $$
   \frac{\partial C}{\partial T} = -r C + e^{-rT} \int_K^\infty (S - K) \frac{\partial p}{\partial T} \, dS
   $$

3. Substitute the Fokker-Planck equation for ∂p/∂T

4. Integration by parts (twice) yields:
   $$
   \frac{\partial C}{\partial T} = -r C + r K \frac{\partial C}{\partial K} + \frac{1}{2} K^2 \sigma^2(T, K) \frac{\partial^2 C}{\partial K^2}
   $$

5. Rearranging gives Dupire's PDE

### 3.4 Dupire's Formula (Inverse Problem)

Solving for σ²(T, K) from observed option prices:

$$
\sigma^2(T, K) = \frac{2 \left( \frac{\partial C}{\partial T} + r K \frac{\partial C}{\partial K} \right)}{K^2 \frac{\partial^2 C}{\partial K^2}}
$$

**Challenges**:
1. Requires second derivatives (sensitive to noise)
2. Denominator can be small (numerical instability)
3. Implied volatility surface may not be smooth

This is why neural networks offer an advantage: automatic differentiation computes derivatives exactly (no finite differences).

---

## 4. Coordinate Transformations and Scaling

### 4.1 Motivation for Scaling

To improve numerical stability and convergence, we transform to normalized coordinates:

**Problems with original coordinates**:
- Stock price S can vary over large ranges (e.g., 500 to 3000)
- Option prices vary by orders of magnitude
- Gradients can have vastly different scales

**Solution**: Scale all variables to [0, 1] or similar bounded ranges

### 4.2 Scaled Variables

Define the transformations:

$$
t = \frac{T}{T_{\max}}, \quad k = \frac{e^{-rT} K}{K_{\max}}, \quad \phi = \frac{\pi}{S_0}
$$

where:
- **t ∈ [0, 1]**: Scaled time (dimensionless)
- **k**: Scaled and discounted strike (approximately ∈ [0, 1])
- **φ**: Scaled option price (approximately ∈ [0, 1])

For the local volatility squared, define:

$$
\eta(t, k) = \frac{T_{\max}}{2} \sigma^2(T, K)
$$

### 4.3 Dupire PDE in Scaled Coordinates

**Theorem**: In scaled coordinates, Dupire's PDE becomes:

$$
\frac{\partial \phi}{\partial t} = \eta(t, k) k^2 \frac{\partial^2 \phi}{\partial k^2}
$$

**Proof**:

Starting from:
$$
\frac{\partial \pi}{\partial T} = \frac{1}{2} K^2 \sigma^2(T, K) \frac{\partial^2 \pi}{\partial K^2} - r K \frac{\partial \pi}{\partial K}
$$

Substitute π = S₀ φ, T = T_max t, K = (K_max / e^{-rT}) k:

1. **Time derivative**:
   $$
   \frac{\partial \pi}{\partial T} = S_0 \frac{\partial \phi}{\partial t} \frac{dt}{dT} = \frac{S_0}{T_{\max}} \frac{\partial \phi}{\partial t}
   $$

2. **First spatial derivative**:
   $$
   \frac{\partial \pi}{\partial K} = S_0 \frac{\partial \phi}{\partial k} \frac{dk}{dK} = S_0 \frac{e^{-rT}}{K_{\max}} \frac{\partial \phi}{\partial k}
   $$

3. **Second spatial derivative**:
   $$
   \frac{\partial^2 \pi}{\partial K^2} = S_0 \frac{e^{-2rT}}{K_{\max}^2} \frac{\partial^2 \phi}{\partial k^2} + S_0 \frac{r e^{-rT}}{K_{\max}} \frac{\partial \phi}{\partial k}
   $$

4. Substitute into Dupire PDE and simplify:
   $$
   \frac{S_0}{T_{\max}} \frac{\partial \phi}{\partial t} = \frac{1}{2} K^2 \sigma^2 S_0 \frac{e^{-2rT}}{K_{\max}^2} \frac{\partial^2 \phi}{\partial k^2} + \text{(terms that cancel)}
   $$

5. Using K = (K_max / e^{-rT}) k and η = (T_max/2) σ²:
   $$
   \frac{\partial \phi}{\partial t} = \eta k^2 \frac{\partial^2 \phi}{\partial k^2}
   $$

**Boundary conditions** in scaled coordinates:

$$
\phi(k, 0) = \max\left(1 - \frac{K_{\max}}{S_0} k, 0\right)
$$

This is much simpler and numerically stable!

### 4.4 Benefits of Scaling

1. **Bounded variables**: t, k, φ ≈ [0, 1] → better NN training
2. **Simplified PDE**: No drift term (rK ∂φ/∂K term disappears)
3. **Balanced gradients**: All terms have similar magnitudes
4. **Universal scaling**: Same NN architecture works for different (S₀, K_max, T_max)

---

## 5. Neural Network Formulation

### 5.1 Network Architecture

We use two neural networks:

1. **NN_φ_tilde(t, k)**: Predicts normalized option price
2. **NN_η_tilde(t, k)**: Predicts normalized local volatility squared

**Common architecture** (shared by both networks):

```
Input: (t, k) ∈ ℝ²
    ↓
Gaussian Noise Layer (regularization)
    ↓
Dense(64, activation='tanh')
    ↓
Residual Block 1
    ↓
Residual Block 2
    ↓
...
    ↓
Residual Block N
    ↓
Dense(64, activation='tanh')
    ↓
Dense(1, activation='softplus')
    ↓
Output: φ̃ or η̃ ∈ ℝ₊
```

### 5.2 Residual Block Details

Each residual block has the structure:

```
Input x
    ↓
Dense(64, use_bias=False) → x₁
    ↓
BatchNormalization() → x₂
    ↓
Activation (tanh) → x₃
    ↓
Dense(64, use_bias=False) → x₄
    ↓
BatchNormalization() → x₅
    ↓
Activation (tanh) → x₆
    ↓
Add([x, x₆]) → Output
```

**Mathematical representation**:

$$
\text{ResBlock}(x) = x + \text{Activation}(\text{BN}(\text{Dense}(\text{Activation}(\text{BN}(\text{Dense}(x))))))
$$

**Benefits**:
- **Skip connections**: Gradient flows directly (mitigates vanishing gradients)
- **Batch normalization**: Stabilizes training, allows higher learning rates
- **Deep networks**: Can stack many blocks without degradation

### 5.3 Output Transformation

Raw network output is transformed to ensure positivity:

$$
\phi_{\text{tilde}}(t, k) = 1 - \exp(-\text{NN}_\phi(t, k))
$$

This ensures:
- **φ_tilde ∈ (0, 1)**: Bounded option prices
- **Smooth**: Exponential is infinitely differentiable
- **Boundary**: φ_tilde → 0 as NN_φ → 0, φ_tilde → 1 as NN_φ → ∞

For volatility:

$$
\eta_{\text{tilde}}(t, k) = \text{NN}_\eta(t, k)
$$

The softplus activation already ensures η_tilde > 0.

### 5.4 Parameter Count

For a network with N residual blocks:

```
Parameters per residual block = 2 × (64 × 64) + 2 × (64 × 2) [BatchNorm parameters]
                                ≈ 8,448 parameters

Total parameters ≈ 64 × 2 [input layer]
                   + N × 8,448 [residual blocks]
                   + 64 × 64 [pre-output layer]
                   + 64 × 1 [output layer]
                   ≈ 128 + 8,448N + 4,160

For N = 3: ≈ 29,632 parameters (NN_φ)
           ≈ 29,632 parameters (NN_η)
           ≈ 59,264 total
```

This is relatively small (modern CNNs have millions of parameters), making training fast.

---

## 6. Loss Function Design

### 6.1 Overall Loss Function

The total loss is a weighted sum of three components:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\phi} + \lambda_{\text{PDE}} \mathcal{L}_{\text{Dupire}} + \lambda_{\text{reg}} \mathcal{L}_{\text{arb}}
$$

where:
- **𝓛_φ**: Data fitting loss (match observed option prices)
- **𝓛_Dupire**: PDE residual loss (enforce Dupire's equation)
- **𝓛_arb**: Arbitrage penalty (no calendar spread arbitrage)
- **λ_PDE, λ_reg**: Hyperparameters controlling trade-offs

### 6.2 Data Fitting Loss

$$
\mathcal{L}_{\phi} = \frac{1}{N} \sum_{i=1}^N w_i \left| \phi_{\text{NN}}(t_i, k_i) - \phi_{\text{ref}}(t_i, k_i) \right|^2 + \mathcal{L}_{\text{BC}}
$$

where:
- **N**: Number of training data points
- **w_i**: Adaptive weights (balances different price scales)
- **φ_ref**: Reference option prices (from Monte Carlo or market)
- **𝓛_BC**: Boundary condition loss

**Adaptive weighting**:

$$
w_i = 1 + \frac{\overline{\phi_{\text{ref}}^2}}{\phi_{\text{ref},i}^2}
$$

This upweights small option prices (deep OTM) which would otherwise be ignored.

### 6.3 Boundary Condition Loss

At t = 0, the option price must match the payoff:

$$
\mathcal{L}_{\text{BC}} = \frac{1}{M} \sum_{j=1}^M w_j \left| \phi_{\text{NN}}(0, k_j) - \max\left(1 - \frac{K_{\max}}{S_0} k_j, 0\right) \right|^2
$$

where k_j are randomly sampled strikes in [k_min, k_max].

**Why random sampling?**
- Covers the domain more thoroughly than a fixed grid
- Acts as data augmentation
- Different samples each epoch → better generalization

### 6.4 Dupire PDE Loss

The PDE residual is:

$$
\mathcal{R}_{\text{Dupire}}(t, k) = \frac{\partial \phi}{\partial t} - \eta(t, k) k^2 \frac{\partial^2 \phi}{\partial k^2}
$$

The loss is:

$$
\mathcal{L}_{\text{Dupire}} = \frac{1}{M^2} \sum_{i,j} w_{ij} \left| \mathcal{R}_{\text{Dupire}}(t_i, k_j) \right|^2
$$

where (t_i, k_j) are **collocation points** randomly sampled in the domain.

**Computing derivatives** using TensorFlow's automatic differentiation:

```python
with tf.GradientTape(persistent=True) as tape_2:
    tape_2.watch(k)
    with tf.GradientTape(persistent=True) as tape_1:
        tape_1.watch(t)
        tape_1.watch(k)

        phi = NN_phi([t, k])

    grad_phi_t = tape_1.gradient(phi, t)    # ∂φ/∂t
    grad_phi_k = tape_1.gradient(phi, k)    # ∂φ/∂k

grad_phi_kk = tape_2.gradient(grad_phi_k, k)  # ∂²φ/∂k²
```

This gives **exact derivatives** (up to floating-point precision), unlike finite differences.

### 6.5 Arbitrage Penalty

To prevent calendar spread arbitrage, we enforce:

$$
\frac{\partial \phi}{\partial t} \geq 0
$$

Economically: longer-maturity options must be more expensive (more time value).

The penalty is:

$$
\mathcal{L}_{\text{arb}} = \frac{1}{M^2} \sum_{i,j} w_{ij} \left[ \max\left(0, -\mathcal{A}(t_i, k_j)\right) \right]^2
$$

where:

$$
\mathcal{A}(t, k) = \frac{\partial \phi}{\partial t} - r T_{\max} k \cdot \text{ReLU}\left(\frac{\partial \phi}{\partial k}\right)
$$

The ReLU ensures we only penalize when the arbitrage is violated.

### 6.6 Optimization

We use **Adam optimizer** (Kingma & Ba, 2014) with:

- **Learning rate**: 10⁻⁴ (standard for PINNs)
- **Decay schedule**: Divide by 1.1 every 2000 epochs
- **Separate optimizers**: NN_φ and NN_η updated independently

**Update rule** (simplified):

```python
# Compute gradients
grads_phi = tape.gradient(L_total, NN_phi.weights)
grads_eta = tape.gradient(L_Dupire, NN_eta.weights)

# Apply updates
optimizer_phi.apply_gradients(zip(grads_phi, NN_phi.weights))
optimizer_eta.apply_gradients(zip(grads_eta, NN_eta.weights))
```

**Why separate optimizers?**
- NN_η appears only in PDE loss (harder to train)
- Can use different learning rates (typically lr_η = lr_φ / 10)

---

## 7. Risk-Neutral Density Extraction

### 7.1 Breeden-Litzenberger Formula

The **risk-neutral probability density function** of S_T is:

$$
f_{\mathbb{Q}}(K; T) = e^{rT} \frac{\partial^2 C}{\partial K^2}(K, T)
$$

**Derivation**:

Starting from the pricing formula:

$$
C(K, T) = e^{-rT} \int_K^\infty (S - K) f(S) \, dS
$$

Differentiate once with respect to K:

$$
\frac{\partial C}{\partial K} = -e^{-rT} \int_K^\infty f(S) \, dS = -e^{-rT} \mathbb{P}(S_T > K)
$$

Differentiate again:

$$
\frac{\partial^2 C}{\partial K^2} = e^{-rT} f(K)
$$

Rearranging gives the Breeden-Litzenberger formula.

### 7.2 Computing Density from Neural Network

Given the trained NN_φ, we compute:

$$
f(K; T) = e^{rT} \frac{\partial^2 C}{\partial K^2}(K, T)
$$

**Step-by-step**:

1. **Scale coordinates**:
   $$
   t = \frac{T}{T_{\max}}, \quad k = \frac{e^{-rT} K}{K_{\max}}
   $$

2. **Evaluate NN**:
   $$
   \phi(t, k) = \text{NN}_\phi(t, k)
   $$

3. **Compute second derivative** using automatic differentiation:
   ```python
   with tf.GradientTape(persistent=True) as tape_outer:
       tape_outer.watch(k)
       with tf.GradientTape(persistent=True) as tape_inner:
           tape_inner.watch(k)
           phi = NN_phi([t, k])

       grad_phi_k = tape_inner.gradient(phi, k)

   grad_phi_kk = tape_outer.gradient(grad_phi_k, k)
   ```

4. **Chain rule** to transform from scaled to original coordinates:
   $$
   \frac{\partial k}{\partial K} = \frac{e^{-rT}}{K_{\max}}
   $$

   $$
   \frac{\partial^2 \phi}{\partial K^2} = \frac{\partial^2 \phi}{\partial k^2} \left( \frac{\partial k}{\partial K} \right)^2 \cdot S_0
   $$

5. **Apply discount factor**:
   $$
   f(K) = e^{rT} \frac{\partial^2 C}{\partial K^2}
   $$

### 7.3 Ensuring Valid Density

The extracted density must satisfy:

1. **Non-negativity**: f(K) ≥ 0 for all K
2. **Normalization**: ∫₀^∞ f(K) dK = 1
3. **Finite moments**: E[S_T^n] < ∞ for relevant n

**Numerical fixes**:

```python
# Ensure positivity
density = np.maximum(density, 0)

# Normalize
integral = np.trapezoid(density, K_grid)
if integral > 0:
    density = density / integral
```

If these conditions are violated, it indicates:
- Insufficient training
- Wrong hyperparameters (λ_PDE too low)
- Arbitrage in the NN predictions

---

## Quick Reference: Understanding the Three-Panel PDF Plots

**For readers who want to understand the output plots without reading the full mathematical derivation:**

The pipeline generates three-panel plots for each maturity showing the same probability distribution in different coordinate systems. Each transformation tests a different aspect of the model.

### Panel 1: Strike Space (K-space)

**Formula**: f(K) in original strike units

**What it shows**:
- Direct comparison of Monte Carlo histogram vs neural network density
- No coordinate transformations applied

**Purpose**: Validates that the NN correctly learned the option price distribution

**Good result**: Blue histogram and red curve overlap closely

---

### Panel 2: Log-Space (ln K)

**Formula**: f(ln K) = f(K) × K (Jacobian-corrected)

**What it shows**:
- Distribution of log-returns: Y = ln(S_T / S_0)
- The Jacobian factor K accounts for the coordinate transformation

**Purpose**: Tests if K follows a log-normal distribution
- If K is log-normal, then ln(K) is Gaussian (normal distribution)
- Panel 2 should look like a bell curve if the assumption holds

**Mathematical justification**:
When changing variables from K to Y = ln(K), probability conservation requires:
```
∫ f_K(K) dK = ∫ f_Y(Y) dY
```
Since dK/dY = K, we have f_Y(Y) = f_K(K) × K

**Good result**: Distribution appears approximately Gaussian (bell-shaped, symmetric)

---

### Panel 3: Gaussian Standardized Space

**Formula**: g(x) = f(K) × σ × K where x = (ln K - μ) / σ (doubly-corrected)

**What it shows**:
- Standardized variable: x = (ln K - μ) / σ
- Should match standard normal N(0, 1) if K is perfectly log-normal
- Dotted black line shows the perfect N(0, 1) for comparison

**Purpose**: Ultimate test of log-normality
- Measures deviation from perfect log-normal via skewness and kurtosis
- Perfect log-normal has skewness = 0, excess kurtosis = 0

**Additional statistics**:
- **Correlation**: Measures linear relationship between MC and NN (should be > 0.95)
- **Skewness**: Asymmetry measure (0 = symmetric)
- **Excess Kurtosis**: Tail heaviness (0 = normal tails, >0 = heavy tails, <0 = light tails)

**Good result**:
- Distribution closely follows the dotted N(0,1) curve
- Skewness ≈ 0, excess kurtosis ≈ 0
- High correlation (> 0.95)

---

### Interpretation Guide

| Observation | Interpretation |
|-------------|----------------|
| Good agreement in Panel 1 | ✅ NN correctly learned option prices |
| Gaussian shape in Panel 2 | ✅ Distribution is approximately log-normal |
| Close match to N(0,1) in Panel 3 | ✅ Distribution is nearly perfectly log-normal |
| Skewness ≈ 0 | ✅ Distribution is symmetric |
| Excess kurtosis ≈ 0 | ✅ Tails match Gaussian (not too heavy or light) |
| Correlation > 0.95 | ✅ Strong linear relationship between MC and NN |

---

**For complete mathematical derivations, see:**
- Section 7: Risk-Neutral Density Extraction
- Section 8: Monte Carlo Validation
- Section 9: Statistical Analysis

---

## 8. Monte Carlo Validation

### 8.1 Simulation with NN Local Volatility

To validate the trained model, we run **Monte Carlo simulation** using the NN-predicted local volatility:

$$
dS_t = r S_t \, dt + \sigma_{\text{NN}}(t, S_t) S_t \, dW_t
$$

**Algorithm** (Euler-Maruyama discretization):

```
Initialize: S₀ = initial stock price
For each time step i = 0, 1, ..., N-1:
    1. Query NN: σᵢ = σ_NN(tᵢ, Sᵢ)
    2. Sample Brownian increment: ΔWᵢ ~ N(0, √Δt)
    3. Update: Sᵢ₊₁ = Sᵢ + r Sᵢ Δt + σᵢ Sᵢ ΔWᵢ
    4. Enforce positivity: Sᵢ₊₁ = max(Sᵢ₊₁, ε)
```

**Parameters**:
- **M paths**: Typically 10,000 - 50,000
- **Time step**: Δt = 10⁻³ years (≈ 0.365 days)
- **Maturities**: T ∈ {0.5, 1.0, 1.5} years

### 8.2 Extracting Monte Carlo Density

From M simulated paths, we estimate the density:

**Method 1: Histogram**

```python
hist, bins = np.histogram(S_T, bins=50, density=True)
```

**Method 2: Kernel Density Estimation (KDE)**

$$
\hat{f}(x) = \frac{1}{M h} \sum_{i=1}^M K\left( \frac{x - S_T^{(i)}}{h} \right)
$$

where:
- **K**: Kernel function (typically Gaussian)
- **h**: Bandwidth (controls smoothness)

```python
from sklearn.neighbors import KernelDensity
kde = KernelDensity(kernel='gaussian', bandwidth=h)
kde.fit(S_T.reshape(-1, 1))
density = np.exp(kde.score_samples(K_grid.reshape(-1, 1)))
```

**Choosing bandwidth h**:
- Scott's rule: h = σ̂ M^(-1/5)
- Silverman's rule: h = 0.9 σ̂ M^(-1/5)
- Cross-validation

### 8.3 Comparison Metrics

To quantify agreement between NN-implied density and MC density:

**1. Visual comparison**:
- Overlay histograms and density curves
- Check all three spaces: K, ln K, standardized

**2. Kolmogorov-Smirnov statistic**:

$$
D = \sup_K |F_{\text{NN}}(K) - F_{\text{MC}}(K)|
$$

where F is the cumulative distribution function.

**3. Moment comparison**:

$$
\text{Error}_n = \left| \mathbb{E}_{\text{NN}}[S_T^n] - \mathbb{E}_{\text{MC}}[S_T^n] \right|
$$

for n = 1 (mean), 2 (variance), 3 (skewness), 4 (kurtosis).

**4. Option price RMSE**:

For strikes K₁, ..., K_L:

$$
\text{RMSE} = \sqrt{\frac{1}{L} \sum_{j=1}^L \left| C_{\text{NN}}(K_j, T) - C_{\text{MC}}(K_j, T) \right|^2}
$$

where:

$$
C_{\text{MC}}(K_j, T) = e^{-rT} \frac{1}{M} \sum_{i=1}^M (S_T^{(i)} - K_j)^+
$$

---

## 9. Statistical Analysis

### 9.1 Log-Normal Hypothesis

Under the **Black-Scholes model** with constant volatility, S_T is log-normally distributed:

$$
\ln S_T \sim \mathcal{N}\left( \ln S_0 + \left(r - \frac{\sigma^2}{2}\right)T, \sigma^2 T \right)
$$

For **local volatility models**, S_T is generally **not** log-normal. However, we can test "how close" it is.

### 9.2 Parameter Estimation

Given the extracted density f(K), estimate log-normal parameters (μ, σ).

**Method of moments** (corrected):

1. **Compute moments of K**:
   $$
   \mathbb{E}[K] = \int_0^\infty K f(K) \, dK
   $$

   $$
   \mathbb{E}[K^2] = \int_0^\infty K^2 f(K) \, dK
   $$

2. **Variance**:
   $$
   \text{Var}[K] = \mathbb{E}[K^2] - \mathbb{E}[K]^2
   $$

3. **Coefficient of variation**:
   $$
   \text{CV}^2 = \frac{\text{Var}[K]}{\mathbb{E}[K]^2}
   $$

4. **Log-normal parameters**:
   $$
   \sigma^2 = \ln(1 + \text{CV}^2)
   $$

   $$
   \mu = \ln(\mathbb{E}[K]) - \frac{\sigma^2}{2}
   $$

**Verification**: If K ~ LogNormal(μ, σ²), then:

$$
\mathbb{E}[K] = e^{\mu + \sigma^2/2}
$$

$$
\text{Var}[K] = e^{2\mu + \sigma^2}(e^{\sigma^2} - 1)
$$

### 9.3 Transformation to Gaussian Space

Define the standardized variable:

$$
X = \frac{\ln K - \mu}{\sigma}
$$

**If K ~ LogNormal(μ, σ²)**, then **X ~ N(0, 1)** (standard normal).

**Jacobian transformation**:

The density of X is related to the density of K by:

$$
g(x) = f_K(K(x)) \left| \frac{dK}{dx} \right|
$$

where:

$$
K(x) = e^{\mu + \sigma x}
$$

$$
\frac{dK}{dx} = \sigma e^{\mu + \sigma x} = \sigma K
$$

Therefore:

$$
g(x) = f_K(K) \cdot \sigma K
$$

**Normalization**:

$$
\int_{-\infty}^\infty g(x) \, dx = 1
$$

### 9.4 Higher Moments

**Skewness**:

$$
\gamma_1 = \frac{\mathbb{E}[(X - \mu_X)^3]}{\sigma_X^3}
$$

For standard normal: γ₁ = 0

**Excess kurtosis**:

$$
\gamma_2 = \frac{\mathbb{E}[(X - \mu_X)^4]}{\sigma_X^4} - 3
$$

For standard normal: γ₂ = 0

**Interpretation**:
- **γ₁ > 0**: Right-skewed (long tail on right)
- **γ₁ < 0**: Left-skewed (long tail on left)
- **γ₂ > 0**: Heavy-tailed (more extreme values than Gaussian)
- **γ₂ < 0**: Light-tailed (fewer extreme values than Gaussian)

For log-normal distribution with parameters (μ, σ):

$$
\gamma_1 = (e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}
$$

$$
\gamma_2 = e^{4\sigma^2} + 2e^{3\sigma^2} + 3e^{2\sigma^2} - 6
$$

**Deviations from these values** indicate non-log-normal behavior.

---

## 10. Arbitrage-Free Constraints

### 10.1 Necessary Conditions

For option prices to be arbitrage-free:

**1. Calendar spread arbitrage**:

$$
\frac{\partial C}{\partial T}(K, T) \geq 0
$$

Longer maturity options must be more expensive (more time value).

**2. Butterfly spread arbitrage**:

$$
\frac{\partial^2 C}{\partial K^2}(K, T) \geq 0
$$

The risk-neutral density must be non-negative.

**3. Call price bounds**:

$$
(S_0 - e^{-rT} K)^+ \leq C(K, T) \leq S_0
$$

**4. Convexity**:

$$
C(K_1, T) \geq \lambda C(K_2, T) + (1-\lambda) C(K_3, T)
$$

for K₂ = λK₁ + (1-λ)K₃, 0 ≤ λ ≤ 1.

### 10.2 Enforcement in Neural Networks

**Soft constraints** (penalty terms in loss function):

$$
\mathcal{L}_{\text{arb}} = \mathcal{L}_{\text{calendar}} + \mathcal{L}_{\text{butterfly}}
$$

where:

$$
\mathcal{L}_{\text{calendar}} = \frac{1}{M} \sum_i \left[ \text{ReLU}\left( -\frac{\partial C}{\partial T}(K_i, T_i) \right) \right]^2
$$

$$
\mathcal{L}_{\text{butterfly}} = \frac{1}{M} \sum_i \left[ \text{ReLU}\left( -\frac{\partial^2 C}{\partial K^2}(K_i, T_i) \right) \right]^2
$$

**Hard constraints** (architecture design):

- Use **softplus** or **ReLU** activations for final layer (ensures C > 0)
- Use **monotonic networks** (Wehenkel & Louppe, 2019) to enforce ∂C/∂T ≥ 0

**Post-processing**:

If constraints are violated, project to nearest arbitrage-free surface:

$$
C_{\text{corrected}} = \arg\min_{C' \in \mathcal{A}} \| C' - C_{\text{NN}} \|^2
$$

where 𝒜 is the set of arbitrage-free option prices.

---

## 11. References

### Primary References

**Dupire, B. (1994)**. "Pricing with a smile." *Risk Magazine*, 7(1), 18-20.
- Original formulation of local volatility model and Dupire's formula

**Breeden, D. T., & Litzenberger, R. H. (1978)**. "Prices of state-contingent claims implicit in option prices." *Journal of Business*, 51(4), 621-651.
- Extraction of risk-neutral density from option prices

**Gatheral, J. (2006)**. *The Volatility Surface: A Practitioner's Guide*. Wiley.
- Comprehensive treatment of local volatility, implied volatility, and stochastic volatility

### Neural Network Methods

**Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019)**. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.
- Foundation of physics-informed neural networks (PINNs)

**Wang, Z., et al. (2025)**. "Deep self-consistent learning of local volatility." *arXiv preprint*.
- Neural network approach to Dupire local volatility calibration
- Inspiration for this implementation

**Horvath, B., Muguruza, A., & Tomas, M. (2021)**. "Deep learning volatility: A deep neural network perspective on pricing and calibration in (rough) volatility models." *Quantitative Finance*, 21(1), 11-27.
- Deep learning for volatility surface calibration

### Stochastic Calculus and Mathematical Finance

**Privault, N. (2022)**. *Introduction to Stochastic Finance with Market Examples* (2nd ed.). CRC Press.
- Rigorous treatment of stochastic calculus, martingale pricing, and local volatility

**Shreve, S. E. (2004)**. *Stochastic Calculus for Finance II: Continuous-Time Models*. Springer.
- Graduate-level textbook on continuous-time finance

**Björk, T. (2009)**. *Arbitrage Theory in Continuous Time* (3rd ed.). Oxford University Press.
- Complete treatment of arbitrage-free pricing and PDE methods

### Numerical Methods

**Glasserman, P. (2003)**. *Monte Carlo Methods in Financial Engineering*. Springer.
- Comprehensive reference on Monte Carlo simulation for finance

**Kingma, D. P., & Ba, J. (2014)**. "Adam: A method for stochastic optimization." *arXiv preprint arXiv:1412.6980*.
- Adam optimizer used in training

**Silverman, B. W. (1986)**. *Density Estimation for Statistics and Data Analysis*. Chapman and Hall.
- Kernel density estimation theory and methods

---

## Appendix: Notation Reference

| Symbol | Description | Units |
|--------|-------------|-------|
| **S_t** | Stock price at time t | Currency |
| **K** | Strike price | Currency |
| **T** | Maturity (time to expiration) | Years |
| **r** | Risk-free interest rate | Year⁻¹ |
| **σ(t, S)** | Local volatility function | Year⁻½ |
| **C(K, T)** | Call option price | Currency |
| **P(K, T)** | Put option price | Currency |
| **f(K)** | Risk-neutral density | Currency⁻¹ |
| **t** | Scaled time = T / T_max | Dimensionless |
| **k** | Scaled strike = e^(-rT) K / K_max | Dimensionless |
| **φ** | Scaled option price = π / S₀ | Dimensionless |
| **η** | Scaled volatility squared = (T_max/2) σ² | Dimensionless |
| **NN_φ** | Neural network for option prices | - |
| **NN_η** | Neural network for local volatility | - |
| **𝔼^ℚ[·]** | Expectation under risk-neutral measure | - |
| **W_t** | Standard Brownian motion | Years½ |

---

**End of Mathematical Treatment**

For implementation details, see the code in [dupire_pipeline.py](dupire_pipeline.py) and configuration options in [config.py](config.py).
