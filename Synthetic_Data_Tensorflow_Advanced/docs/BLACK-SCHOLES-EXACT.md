Dupire Equation and Risk-Neutral Density (σ = 1)

1. Dupire Equation Overview

The Dupire forward PDE for call option prices is:
\frac{\partial C}{\partial T} = \frac{1}{2} \sigma^2(K,T) K^2 \frac{\partial^2 C}{\partial K^2} - (r - q) K \frac{\partial C}{\partial K} + q C.

When the local volatility is constant (\sigma = 1), the PDE simplifies to the constant-volatility Black–Scholes PDE:
\frac{\partial C}{\partial T} = \frac{1}{2} K^2 \frac{\partial^2 C}{\partial K^2} - (r - q) K \frac{\partial C}{\partial K} + q C.
This corresponds to a lognormal diffusion process:
dS_t = S_t[(r - q)dt + dW_t].

⸻

2. Breeden–Litzenberger Transformation

Breeden and Litzenberger (1978) showed that the risk-neutral probability density of the terminal price S_T can be obtained from the second derivative of the call price surface:
\frac{\partial^2 C}{\partial K^2}(K,T) = e^{-rT} f_{S_T}(K).
	•	First derivative: \frac{\partial C}{\partial K} = -e^{-rT} P(S_T > K)
	•	Second derivative: \frac{\partial^2 C}{\partial K^2} = e^{-rT} f_{S_T}(K)

This links the Dupire PDE (in K) to the forward Fokker–Planck equation for f_{S_T}(K).

⸻

3. Analytical Density for \sigma = 1

Under the risk-neutral measure:
\ln S_T \sim \mathcal{N}\big(\ln S_0 + (r - q - \tfrac{1}{2})T,\; T\big).
Thus, the exact analytical density of S_T (and equivalently of the strike variable K) is:

\boxed{
p(K;T) = \frac{1}{K \sqrt{2\pi T}} \exp\left( -\frac{\big(\ln(K/S_0) - (r - q - \tfrac{1}{2})T\big)^2}{2T} \right)
}

This is the lognormal density of the terminal price for constant volatility.

⸻

4. Density from the Call Surface

By the Breeden–Litzenberger relation, the second derivative of the call price is:
C_{KK}(K,T) = e^{-rT} p(K;T) = \frac{e^{-rT}}{K \sqrt{2\pi T}} \exp\left( -\frac{(\ln(K/S_0) - (r - q - \tfrac{1}{2})T)^2}{2T} \right).

⸻

5. Example: Python Implementation

import numpy as np
import matplotlib.pyplot as plt

S0, r, q, T = 100, 0.03, 0.0, 1.0
K = np.linspace(1, 300, 500)
mu = np.log(S0) + (r - q - 0.5)*T
pdf = 1/(K*np.sqrt(2*np.pi*T)) * np.exp(-(np.log(K) - mu)**2/(2*T))

plt.plot(K, pdf)
plt.xlabel('Strike K')
plt.ylabel('Risk-neutral density p(K;T)')
plt.title('Lognormal Density from Dupire (σ=1)')
plt.show()


⸻

6. Comparison: Constant vs Non-Constant Volatility

When \sigma(K,T) varies with strike and maturity, the Dupire equation no longer reduces to a constant-coefficient heat equation. Instead:
	•	The diffusion term \frac{1}{2}\sigma^2(K,T)K^2 C_{KK} introduces space–time-dependent coefficients, destroying closed-form solvability.
	•	The corresponding density p(K,T) = e^{rT} C_{KK} satisfies a nonlinear Fokker–Planck equation:
\frac{\partial p}{\partial T} = -\frac{\partial}{\partial K}\big[(r - q)Kp\big] + \frac{1}{2}\frac{\partial^2}{\partial K^2}\big[\sigma^2(K,T)K^2 p\big].
	•	This requires numerical integration and can generate implied-volatility smiles and skews.

In contrast, when \sigma=1, the coefficients are constant, yielding a pure lognormal diffusion and an analytical density.

⸻

7. Summary Table

Case	PDE Type	Density	Comments
\sigma = 1	Constant-coefficient	Lognormal, analytic	Equivalent to Black–Scholes
\sigma(K,T) \neq 1	Variable coefficients	Requires numerical solution	Produces volatility smiles and skews


⸻

In summary: When \sigma = 1, Dupire reduces to the Black–Scholes constant-volatility case, and the option surface’s curvature C_{KK} directly encodes a lognormal density with variance T. For non-constant \sigma(K,T), the problem becomes a full parabolic PDE for the evolving density.