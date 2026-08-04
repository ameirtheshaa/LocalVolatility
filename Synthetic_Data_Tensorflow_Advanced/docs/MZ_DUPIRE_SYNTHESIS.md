# MZ-Augmented Self-Consistent Learning of Local Volatility
### A Mori–Zwanzig closure for the Dupire equation: inferring local volatility *and* the risk-neutral density

*Synthesis note / draft for a follow-up to* **Wang, Shaa, Privault & Guet, "Deep self-consistent learning of local volatility," *Journal of Computational Finance* 29(1), 2025** *(hereafter WSPG25), cross-pollinated with the eigenbasis Mori–Zwanzig (MZ) closure theory of* **Shaa, Lim, Guet & Garbet, "Eigenbasis Mori–Zwanzig closure for the Hasegawa–Wakatani 12-mode system"** *(submitted, Nuclear Fusion; hereafter SLGG).*

Status: theory + CPU de-risking complete (this document). Implementation Steps 2–4 scoped in §10.

---

## 1. Introduction and contribution

WSPG25 calibrates a Dupire local-volatility surface `σ(T,K)` from option prices with two physics-informed neural networks (`NN_φ` for the price, `NN_η` for `σ²`) trained jointly to satisfy the Dupire PDE and no-arbitrage as *soft* constraints. It is *self-consistent* in the sense that the price network and the volatility network are coupled through the **pointwise** PDE residual. Two things it does not do well, and which matter for downstream use:

- **(G1) The inverse in the tails.** Recovering `σ` from prices means dividing by the price convexity `∂²C/∂K²`, which vanishes in the wings; the inversion is ill-posed exactly where option data is least informative. The companion `mz_spectral` ROM confronts this with a hard *reliability mask* (`sigma_reliability_mask`, `tau_kk`) that simply discards low-curvature cells and interpolates.
- **(G2) The risk-neutral density.** The object practitioners actually want — `f(K,T) = e^{rT}\,∂²C/∂K²` (Breeden–Litzenberger) — is the *second derivative* of the learned price. WSPG25 produces it as a by-product and enforces validity (positivity, unit mass, the martingale mean `E[S_T]=S_0e^{rT}`) only softly, with positivity ultimately *clipped* (`max(f,0)` + renormalize).

This note develops a theory that targets both, by transplanting the closure technology SLGG built for plasma turbulence. **The central contribution is a *density-side* Mori–Zwanzig closure**: a Gaussian/log-normal resolved core plus a single stable scalar `E_tail` (the "tail energy") whose maturity-evolution is governed by a balance ODE and which couples back through a structure-preserving map that makes `∫f=1` and `E[S_T]=S_0e^{rT}` **exact by construction** and `f≥0` **structural**. The same convergent MZ memory furnishes a calendar-coupled regularizer for the inverse problem (G1), and an MZ-consistency loss that strictly extends WSPG25's self-consistent training (G3).

A second, equally important contribution is **negative and clarifying**: we show — analytically and numerically (§8) — that the naive idea ("run MZ on the forward Dupire PDE") is *the wrong place to spend effort*. The forward price problem is linear, parabolic, and dissipative; its MZ memory is provably convergent but empirically inert. The action is on the density and inverse sides. Stating this precisely is what makes the positive contribution land.

**Three claims, three goals.**
1. **(G1)** Infer `σ(T,K)` from price via Dupire, robust in the tails — via the convergent MZ memory as a calendar-coupled prior (§6).
2. **(G2)** Capture `f=e^{rT}∂²C/∂K²` accurately — positive, normalized, correct mean — via the density-side energy-budget closure (§5).
3. **(G3)** Extend WSPG25 — via an MZ-reduced-dynamics + martingale loss that recovers the baseline PINN when switched off (§7).

---

## 2. Background

### 2.1 The scaled Dupire PDE and the `mz_spectral` ROM

In scaled coordinates `φ̃ = C/S_0`, `t̃ = T/T_max`, `k̃ = e^{-rT}K/K_max`, `η̃ = (T_max/2)σ²`, the Dupire forward equation is the drift-free parabolic problem

$$\partial_{\tilde t}\tilde\varphi \;=\; \tilde\eta(\tilde t,\tilde k)\,\tilde k^2\,\partial_{\tilde k\tilde k}\tilde\varphi . \tag{2.1}$$

Semi-discretized on a uniform `k̃`-grid, this is `dφ/dt̃ = L(t̃)φ`, `L = diag(η̃k̃²)·D2`. The `mz_spectral` module installs a Fourier low-pass projector `P_R` (keep `keep` modes), Galerkin-splits `L` into `L_RR, L_RU, L_UR, L_UU`, integrates the resolved dynamics by backward-Euler (`integrate_rrr_only`) with an optional single-scalar quasi-linear backscatter `ν_QL·P_R D2 φ` (`integrate_rrr_ql`, `fit_nu_ql`), inverts `η̃ = ∂_{t̃}φ̃ / (k̃²∂_{k̃k̃}φ̃)` (`sigma_from_phi.py`), and reads off the density by Breeden–Litzenberger (`pdf_from_phi_tilde`).

### 2.2 The SLGG eigenbasis-MZ closure (the theory we transplant)

For the modified Hasegawa–Wakatani 12-mode system, SLGG derives the exact Mori–Zwanzig identity, then establishes three *structural* obstructions to mode-resolved closure at small resolved sets: a **sign-structure theorem** (the quasi-linear memory kernel's diagonal is strictly positive — it can only inject, never damp), a **propagator-instability theorem** (when the orthogonal-dynamics propagator carries an unstable eigenvalue, the Markov kernel `∫_0^∞ K(τ)dτ` diverges), and a near-unity **driven-response** loop gain. Every mode-resolved closure they tried failed (kernel divergence, wrong sign, bootstrap blow-up; black-box ML gave negative `R²` on rollout). What *worked* was to **abandon mode resolution**: collapse the unresolved sector to one scalar energy `E_U` with a stable ODE `dE_U/dt = γ_in(E_DW^R)^p − γ_c E_U` (`p=2`), coupled back through a sign-definite, structure-preserving damping. Two parameters, one derived exponent. That is the template.

### 2.3 WSPG25's self-consistent PINN (the work we extend)

`NN_φ` and `NN_η` are trained on losses: data fit `L_dit`, `T=0` initial condition `L_ini`, no-arbitrage `L_arb` (calendar + butterfly via a single constraint), and the Dupire residual `L_dup`. Self-consistency = the two nets coupled through `L_dup`. WSPG25 beats Tikhonov / SSVI / GP on DAX and SPX reprice error.

---

## 3. Mori–Zwanzig on both sides

There are two MZ problems hiding here, and choosing the right one per goal is half the theory.

**Price side.** State `φ̃` (the call surface); generator `L` from (2.1); resolved/unresolved by a spectral projector in `k̃`. Linear, non-autonomous, **dissipative**.

**Density side (the adjoint).** The risk-neutral density `p(S,T)` obeys the Fokker–Planck/forward-Kolmogorov equation, the adjoint of the Dupire backward operator, and `f=e^{rT}∂²C/∂K²` *is* that density. The natural spectral basis for a density with a near-Gaussian core is the **Hermite ladder** (Gram–Charlier/Edgeworth): resolved = low moments (mean, variance), unresolved = higher moments / tails (skew, kurtosis, …).

The exact MZ identity is the same on both sides,
$$\partial_t \mathcal P u = \mathcal P R[\mathcal P u] + \int_0^t \mathcal P R\,e^{(t-s)\mathcal Q R}\,\mathcal Q R[\mathcal P u(s)]\,ds + \underbrace{\mathcal P R\,e^{t\mathcal Q R}\,\mathcal Q u(0)}_{\text{noise}} . \tag{3.1}$$
On a single calibrated surface the initial condition `φ̃(0,k̃)=(1-K_max k̃/S_0)^+` is *exact*, so the **noise term vanishes deterministically** — the Dupire GLE is purely Markov + memory, cleaner than SLGG. The *exploitable* stochastic content is not this (zero) term but the **epistemic** uncertainty in the unobserved unresolved surface; §10 turns that into calibrated bands.

**Which side is primary, per goal:**

| Goal | Primary object | Why |
|---|---|---|
| **G1** σ from Dupire, robust tails | price side for the forward map; **density side** for the regularizer | the inverse `η̃=φ̃_t/(k̃²φ̃_{kk})` blows up where curvature→0; a density-side prior with a valid tail constrains it |
| **G2** the PDF `f` | **density side** (moment hierarchy) | `f` *is* the density; making it valid by construction is a statement about moments, not prices |
| **G3** extend WSPG25 self-consistency | **both, coupled** | the MZ-reduced resolved dynamics is a new, lower-dimensional consistency constraint between `NN_φ` and `NN_η` |

---

## 4. The HW ↔ Dupire dictionary

| SLGG (Hasegawa–Wakatani) | Dupire price side | Dupire density side | Verdict |
|---|---|---|---|
| `M_k` eigenbasis operator; parameter-independent eigenvectors | `L=diag(η̃k̃²)D2`, **not** parameter-independent | Hermite ladder of the OU/heat generator | **partial** — the clean eigenbasis is Hermite on the *density* side, not `eig(L)` on the price side |
| fast/slow split, adiabatic elimination | high-`k̃` (sharp-strike) modes decay fast under (2.1) | high Hermite modes relax fast | transfers (heat smoothing) |
| memory kernel `K(τ)` | Schur complement `-L_RU L_UU^{-1}L_UR` (converges, §8) | moment-coupling memory | transfers; **convergent here** (dissipative) |
| **sign-structure theorem** (`K_{++}>0`, can't damp) | benign: memory = +diffusion (right sign) | **re-appears as Edgeworth negativity** (`f<0` in wings) | the theorem **relocates** to density positivity |
| **propagator instability** (`∫K` diverges) | **cannot occur** (`L` dissipative, §8 confirms) | cannot occur for the dissipative moment ODE | **clean win** on the forward/parabolic side |
| driven-response loop gain `G≈0.81` | absent (`η̃` exogenous, `G≈0`) | mild | the **inverse** `σ[φ̃]` re-introduces the feedback; tail mask = the confession |
| validity/cutoff `k_c`, Pareto | `ε_r=‖P_U φ_{kk}‖²/‖P_R φ_{kk}‖²` | moment-dimension (§8, dim=1) | transfers (`truncation_study.py`) |
| **energy-budget closure** `E_U`, `dE_U/dt`, sign-definite coupling | — | **`E_tail`, the centerpiece (§5)** | **the winning transplant** |
| predator–prey `E_DW→E_U→damping` | — | core ⇄ tail moment exchange | transfers as a *constrained* exchange (not Lotka–Volterra; see §9) |
| sign-definiteness for stability | no-arbitrage: `η̃≥0`, `∂²C/∂K²≥0`, martingale | same | **no-arbitrage *is* SLGG's sign-definiteness** |

The single most useful row: **SLGG's sign-definiteness requirement is, in finance, the no-arbitrage constraint.** A closure that violates the sign produces, in plasma, an energy blow-up; in finance, a **negative density** (a butterfly arbitrage). The danger is worse in finance because the code currently *launders* it: `pdf_from_phi_tilde` does `max(f,0)` then renormalizes, so a structurally invalid closure can look fine on a plot while having quietly destroyed mass and shifted the mean. SLGG's blow-up is loud; Dupire's must be hunted with the IBP mean diagnostic.

---

## 5. The density-side energy-budget closure (centerpiece)

### 5.1 Resolved core and Hermite-unresolved tail

Let `X_T=\ln(S_T/S_0)`, `Y=(X_T-μ_T)/s_T` the standardized log-return with risk-neutral mean `μ_T` and variance `s_T²`. Expand the density of `Y` in probabilists' Hermite functions (Gram–Charlier/Edgeworth):

$$g(y,T)=\phi_{\mathcal N}(y)\Big[1+\sum_{n\ge3}c_n(T)\,He_n(y)\Big],\qquad \phi_{\mathcal N}(y)=\tfrac1{\sqrt{2\pi}}e^{-y^2/2}. \tag{5.1}$$

**Resolved core** = `(μ_T,s_T²)` (the log-normal/Black–Scholes part, `c_0=1, c_1=c_2=0` pinned). **Unresolved** = `{c_n}_{n\ge3}`: `c_3=\mathrm{Skew}/6`, `c_4=\mathrm{ExKurt}/24`, …

### 5.2 The moment hierarchy from the generator

Testing the Fokker–Planck generator against `S^n` gives the raw-moment hierarchy
$$\frac{dm_n}{dT}=n\,r\,m_n+\tfrac12 n(n-1)\,E[\sigma^2(T,S_T)S_T^n]. \tag{5.2}$$
For **constant `σ`** this *closes* (`E[σ²S^n]=σ²m_n` → log-normal moments). For **local `σ(T,S)`** write `σ²=\bar σ²(T)+δσ²(T,S)`; the term `E[δσ²S^n]` is the **unresolved forcing of moment `n` by the smile curvature of `σ`** — the genuine closure problem, the analog of SLGG's RRU bilinear coupling.

### 5.3 The tail-energy scalar and its balance ODE (the transplant)

Collapse the tower into one scalar,
$$E_{\text{tail}}(T)\equiv\sum_{n\ge3}n!\,c_n(T)^2=\big\|g(\cdot,T)-\phi_{\mathcal N}\big\|^2_{L^2(1/\phi_{\mathcal N})}\ \ge 0, \tag{5.3}$$
the squared `χ²`-distance from Gaussian (nonnegative *by construction* — the density analog of `E_U≥0`). Its balance ODE transplants SLGG verbatim, with maturity `T` as the evolution variable:

$$\boxed{\ \frac{dE_{\text{tail}}}{dT}=\gamma_{\text{in}}\,\mathcal S(T)^2-\gamma_c\,E_{\text{tail}}\ } \tag{5.4}$$

with smile-curvature source `𝒮(T)` (the non-Gaussian part of the butterfly production `η̃k̃²φ̃_{kk}`).

- **Exponent `p=2` transfers *with its derivation*.** Each unresolved Hermite mode is driven by a *product* of (local-vol perturbation)×(resolved amplitude), so its energy input scales as (resolved)² — exactly SLGG's argument for `p=2`. (Scan `p∈{1,…,3}` and report the empirical optimum, as SLGG did, since the bilinear argument is weaker here.)
- **The drain `γ_c` is *derived*, not fit.** The Hermite functions diagonalize the OU/heat part of the generator; mode `n` relaxes to Gaussian at rate `∝n`, so `γ_c` is the lowest unresolved Hermite rate (`≈3κ_OU`). *Even leaner than SLGG, where `γ_c` was absorbed into a ratio.*
- **Unconditional stability.** `γ_in,𝒮²,γ_c>0` ⇒ a unique globally attracting fixed point `E_tail^*=(γ_in/γ_c)𝒮²`; `E_tail` cannot blow up. No propagator, no kernel divergence — by design.

### 5.4 Structure-preserving coupling: a valid density by construction

Given `E_tail(T)`, set the leading Gram–Charlier coefficients (skew/kurtosis channels) and reconstruct `f`. The three invariants:

- **Normalization `∫f=1` — exact, free.** `∫g\,dy=c_0=1` because every `He_{n\ge1}` is `φ_𝒩`-orthogonal to `He_0=1`; adding tail energy through `c_{n\ge3}` cannot change the mass. The `y→K` change of variables preserves total mass. ∎
- **Martingale `E[S_T]=S_0e^{rT}` — exact, free.** With the Gram–Charlier MGF `M_Y(u)=e^{u^2/2}[1+\sum_{n\ge3}c_n u^n]`,
$$E[S_T]=S_0\,e^{\mu_T+s_T^2/2}\Big[1+\sum_{n\ge3}c_n(T)s_T^n\Big].$$
Impose the martingale by **defining the core mean**
$$\boxed{\ \mu_T=rT-\tfrac12 s_T^2-\ln\!\Big[1+\sum_{n\ge3}c_n(T)s_T^n\Big]\ }\quad\Rightarrow\quad E[S_T]\equiv S_0e^{rT}\ \forall\{c_n\}. \tag{5.5}$$
This is the **closed-form version of the `K=0` boundary condition** `φ̃(T,0)=1` that the PINN enforces softly via `λ_k0`, and precisely the identity `compute_mean_diagnostics` checks as (i)=(ii)=(iii). *The closure satisfies the IBP identity by construction; WSPG25 only penalizes its violation.* ∎
- **Positivity `f≥0` — NOT free; this is SLGG's sign theorem reincarnated.** Raw Gram–Charlier goes negative in the wings for large `|c_3|,|c_4|` (the classical Edgeworth failure). **Cure: a maximum-entropy / exponential tilt**
$$g(y)=\phi_{\mathcal N}(y)\exp\!\Big[\sum_{n\ge3}\lambda_n He_n(y)\Big]\big/Z, \tag{5.6}$$
manifestly positive, `λ_n` solved from `E_tail`. This is the density-side analog of SLGG's *structure-preserving projection* — and, just as SLGG warns, you must build positivity into the *representation*, not clip after the fact. (A positive log-normal/variance-gamma mixture parameterized by `E_tail` is the alternative if a generative density is preferred.)

### 5.5 The coupled system (the Dupire predator–prey)

$$\frac{ds_T^2}{dT}=\bar\sigma^2(T)+(\text{tail back-reaction}),\quad \mu_T\ \text{by (5.5)},\quad \frac{dE_{\text{tail}}}{dT}=\gamma_{\text{in}}\mathcal S^2-\gamma_c E_{\text{tail}},\quad \{λ_n\}\leftarrow E_{\text{tail}}\ \text{via (5.6)}. \tag{5.7}$$

A stable, low-dimensional, structure-preserving ODE system in maturity, with `∫f=1` and `E[S_T]=S_0e^{rT}` exact and `f≥0` structural — **G2 by construction**. Parameter budget mirrors SLGG: `γ_in`, one coupling scale, the *derived* `γ_c` and `p=2`. The de-risk (§8, Step 1) confirms a **single** `E_tail` scalar suffices.

---

## 6. Calendar-coupled inverse regularizer (G1)

The inverse `η̃=∂_{t̃}φ̃/(k̃²∂_{k̃k̃}φ̃)` is ill-posed where `∂_{k̃k̃}φ̃→0` (tails). The current code masks those cells (`sigma_reliability_mask`) and fills by spatial interpolation (`regularize_sigma_grid`) — a Tikhonov-flavored prior with no dynamics. Replace it with a **forward-consistent, cross-maturity** prior: in masked cells take the convexity from the MZ-ROM reconstruction,
$$[\partial_{\tilde k\tilde k}\tilde\varphi]_{\text{MZ}}=e^{-rT}S_0^{-1}f_{\text{ROM}},\qquad \hat\eta|_{\text{tail}}=\frac{\partial_{\tilde t}\tilde\varphi}{\tilde k^2\,[\partial_{\tilde k\tilde k}\tilde\varphi]_{\text{MZ}}}, \tag{6.1}$$
so `σ` in the tails is informed by the time-history of the whole surface (and by the §5 density, which is valid there) rather than a pointwise spatial second derivative. Concretely, choose the tail backscatter to minimize the one-step calendar residual `‖(I-Δt̃ L_RR)φ^{n+1}-φ^n‖`. This **upgrades WSPG25's pointwise self-consistency to calendar-coupled self-consistency**, and is well-posed because the memory kernel converges (§8). The decisive metric (§10) is tail-region `‖σ̂-σ_{oracle}‖` against the existing oracle `σ`.

---

## 7. MZ-augmented self-consistent PINN (G3)

Two additions to WSPG25, both injecting the §5 closure as an inductive bias so the net generalizes from sparse quotes:

- **(7a) MZ-reduced consistency loss.** Require the resolved projection of the NN price to obey the *closed reduced* equation:
$$\mathcal L_{\text{MZ}}=\big\|\,\partial_{\tilde t}(P_R\tilde\varphi_{NN})-[L_{RR}(P_R\tilde\varphi_{NN})+\text{(tail coupling)}]\,\big\|^2\quad\text{or}\quad \big\|\kappa_4^{NN}(T)-\kappa_4^{\text{ODE}}(T)\big\|^2.$$
A coarse-grained companion to the pointwise `L_dup`, regularizing exactly the low-frequency dynamics §5 closes.
- **(7b) Martingale loss.** Promote `compute_mean_diagnostics` from a post-hoc check to a training penalty `(e^{rT}\!\int K\,∂²C_{NN}/∂K²\,dK - S_0e^{rT})^2` + mass defect, evaluated by the existing nested-tape autodiff. This bakes the (5.5) guarantee into `NN_φ`, complementing the `λ_k0` term.

With `λ_{MZ}=λ_{mart}=0` the baseline WSPG25 PINN is recovered exactly — proving this is a **strict extension**. The publishable claim: *MZ-augmented self-consistent learning improves tail σ-recovery, the IBP martingale gap, and density positivity at fixed (or reduced) data.*

---

## 8. Numerical de-risking (decisive experiments)

All three experiments are pure-NumPy, CPU, seconds each (`examples/run_mz_derisk.py` → `mz_derisk.json`, `plots/mz_derisk/`). They were run *before* committing to the framing; they confirm it.

### Step 0a — the operator IS dissipative (SEED A), with a discrete caveat

`L=diag(η̃k̃²)D2` is similar to the symmetric `A=\sqrt D\,D2\,\sqrt D`, so for a symmetric negative-definite stencil `eig(L)≤0` and the MZ memory kernel converges.

| quantity (representative maturity) | const σ | smile | reading |
|---|---|---|---|
| `spec(L)`, **symmetric** stencil | **−0.209** | **−0.780** | ≤0 → **dissipative; memory converges (SEED A ✓)** |
| `spec(L_UU)`, eigenbasis projector | **−202** | **−744** | ≤0 → eigenbasis/Sturm–Liouville projector keeps orthogonal dynamics dissipative (the cure) |
| `spec(L_UU)`, Fourier proj., sym stencil | ~3e-12 | ~1e-11 | ≈0 → Fourier proj. marginally OK with a symmetric stencil |
| `spec(L)`, **repo** one-sided stencil | **+0.405** | **+0.185** | **>0 → artifact** of the boundary stencil |

**Verdict:** SEED A is confirmed at the operator level. The repo's one-sided boundary stencil makes the *discrete* `L` non-dissipative (positive abscissa); the ROM survives only because backward-Euler is implicit. The principled fix — a symmetric stencil in the weighted (Sturm–Liouville) metric / eigenbasis projector — restores dissipativity with margin. (See `plots/mz_derisk/plot_0a_abscissa.png`: green ≤0, red >0.)

### Step 0b — the forward MZ memory is cosmetic → PIVOT CONFIRMED

Exact constant-σ (Black–Scholes truth, no MC noise). The single-scalar QL closure (`fit_nu_ql`) is the existing forward memory. The fitted coefficient is now constrained to ν≥0 (scalar NNLS; `nonneg=True` default — see `docs/MZ_QL_NONNEG_FIX_REPORT.md`); the legacy symmetric clip is reproducible via `nonneg=False`.

| keep | legacy `nonneg=False` | fixed `nonneg=True` (default) |
|---|---|---|
| 16 | ν=−5.0e-4 (neg, clip), gain **−4.4%** | ν=0, gain **+0.0%** |
| 32 | ν=−5.0e-4 (neg, clip), gain **−7.5%** | ν=0, gain **+0.0%** |
| 64 | ν=−5.0e-4 (neg, clip), gain **−19.5%** | ν=0, gain **+0.0%** |
| 128 (no truncation) | ν=+1.9e-6, +0.0% | ν=+1.9e-6, +0.0% |

Best relative gain across all truncations: **+0.0%**. PDF sanity (BL-of-truth vs exact lognormal) l2 = **2.2e-6** (machinery correct). **Verdict: the forward QL memory does not improve the resolved dynamics at any truncation — the price-side MZ memory is not a useful contribution; the action is the density/inverse side.** Two sharp corollaries:

1. **Cosmetic, not harmful (sharpened by the ν≥0 fix).** With the legacy symmetric clip the fit selected an *anti-diffusive* ν pinned at the negative clip, actively *worsening* the error — the wrong-sign closure SLGG's sign analysis predicts. The ν≥0 constraint (the scalar NNLS optimum) forbids that sign; the fit then projects to **ν=0**, so RRR+QL collapses onto RRR-only (gain +0.0%). The closure is **cosmetic (a no-op)** on smooth data — both readings agree the forward memory adds nothing. (`docs/MZ_QL_NONNEG_FIX_REPORT.md`.)
2. **The forward difficulty is representation, not memory.** `plots/mz_derisk/plot_0b_pdf.png` shows the truncated forward ROM mangles the density with Gibbs oscillations from the *kinked payoff*, which QL cannot fix — reinforcing that the eigenbasis (not the Fourier basis), and the density side, are where the leverage is.

*Caveat:* this tests the *existing scalar* closure; the full Schur-complement memory `K=-L_RU L_UU^{-1}L_UR` (which converges, Step 0a) might recover more — but the structurally-needed, by-construction wins (G2 positivity/martingale) live on the density side regardless.

### Step 1 — one tail scalar suffices

Across a family of local-vol surfaces × maturities (325 samples), the standardized skew and excess-kurtosis are **99.0% correlated** (PCA effective dimension **1**); const-σ density sanity l2 = 0.0033. **A single `E_tail` scalar suffices** (§5.3). *Caveat:* the surface family has co-varying knobs; a next-pass check with independent skew/kurtosis controls and MC/fine-PDE densities is warranted before fixing the scalar count.

---

## 9. Risk analysis (falsifiable theses)

1. **SEED A holds and is load-bearing — but it is the *easy* half.** Confirmed (§8). Do not sell the forward MZ as the contribution; it is the trivial, dissipative case.
2. **SLGG's pitfalls do not vanish; they relocate** to (i) the **inverse** (`φ̃_{kk}→0` denominator = the ratio pathology), (ii) **no-arbitrage** (`f≥0`, `η̃≥0`, martingale = sign-definiteness/conservation), and (iii) **Edgeworth negativity** in the moment closure (= the wrong-sign theorem). The §5 max-entropy tilt is the cure for (iii); the §6 regularizer addresses (i); (5.5) handles the martingale part of (ii).
3. **The winning transplant is the density-side energy budget (§5)** — `E_tail` + balance ODE + structure-preserving coupling — delivering G2 by construction, with `p=2` and `γ_c` *derived*.
4. **Genuine HW re-entry: dynamic/stochastic volatility.** If `σ` is made state-dependent or stochastic (SLV), the generator stops being a fixed dissipative `L`, a non-normal 2-field propagator appears, and the propagator-instability risk returns *for real*. A minimum-variance floor `η̃≥η_min>0` is the Dupire analog of SLGG's minimum background viscosity. This is the one place to be genuinely careful in future extensions.
5. **Superficial-analogy call-outs.** (a) "Eigenbasis of `L`" is *not* parameter-independent (that was special to SLGG's `d_k I` dissipation); the parameter-clean ladder is **Hermite on the density side**. (b) The core⇄tail coupling is a *constrained* exchange under an exact conservation law, **not** Lotka–Volterra — so a *limit cycle in `T` would be a red flag, not a feature* (a static arbitrage-free surface has a unique density). (c) The MZ **noise term** is *not* a free lunch: on a single calibrated surface it is deterministic (= the memory term), and *lifting* to a stochastic-vol model is under-determined by the surface — by **Gyöngy's theorem** the surface is the Markovian projection of infinitely many SLV models, and the memory kernel is a deterministic functional of it, so vol-of-vol / correlation are unidentifiable from one surface (FDT also fails off the Gaussian case). What genuinely survives is *epistemic UQ*, built and validated in §10.

---

## 10. Epistemic-UQ extension (built this pass)

The chosen build (`mz_spectral/uncertainty.py`, `examples/run_mz_uq.py`): turn point estimates of `σ(T,K)` and the density `f(K,T)` into **calibrated confidence bands** by propagating a data-estimable input price covariance `Σ_φ` through the Breeden–Litzenberger and Dupire-inverse maps. **Why UQ and not a stochastic-vol model:** a single surface fixes only the marginals and, by **Gyöngy's theorem**, is the Markovian projection of infinitely many SLV models; the memory kernel `K=−L_RU L_UU⁻¹L_UR` is a deterministic functional of the surface, so it carries *no* independent joint-law information (vol-of-vol and correlation are unidentifiable from one surface). Epistemic UQ is what genuinely survives, and it serves G1.

**Method (`uncertainty.py`).** `Σ_φ` from data (MC price std error / quote noise). The **density band is exact**: `f = A·φ̃` with `A = S0 e^{-rT}/K_max²·D2` is linear, so `Cov[f]=A Σ_φ Aᵀ`. Two ideas are ported from *Jet-Space Reconstruction of Dynamical Systems* (Shaa & Guet): (1) a **weak-form / IBP** propagation `∫f·ψ = e^{rT}∫C·ψ''` that moves both K-derivatives onto a smooth mollifier (well-conditioned where the pointwise `∂²C/∂K²` blows up; mollifier scale = the resolution regularizer); (2) a **pole-benignity feasibility gate** upgrading the crude `tau_kk` mask. The σ-inverse is nonlinear (ratio + √): a closed-form delta band and an MC-over-input band, with an inversion-side regularizer (smooth φ̃ along k̃ — the analog of the density mollifier).

**Results — Phase A (const-σ exact; the calibration test).**
- **Density (linear) → exact bands, both at nominal.** Pointwise and weak-form 95% bands both cover the exact lognormal at **≈0.95** (interior and tail), but the **weak-form band is ~81× narrower in the tails** (median tail sd `1.2e-3` vs `1.0e-1`) — the conditioning win (`plots/mz_uq/plot_A_bands.png`, right: the pointwise band oscillates to ±0.6 in the left wing while the weak-form stays tight).
- **The σ-inverse is irreducibly ill-posed pointwise.** At 1% price noise, neither the delta band (interior coverage **0.49**) nor the raw MC-over-input band (**0.02**) covers `σ_oracle=0.3` — `D2` amplifies the price noise into a garbage point estimate. **Regularizing** (smooth φ̃ along k̃ by ~5 grid points) restores a **calibrated band (interior coverage 0.91**, nominal 0.95) at a controlled resolution (`plot_A_bands.png`, left: the regularized band hugs `σ=0.3` and widens honestly in the left tail; the raw band spikes past 1). This *quantifies* the ill-posedness Privault's `tau_kk` mask handles by fiat.
- **Martingale preserved:** the point density has `∫f≈0.97`, mean within **1.6%** of `S0 e^{rT}` (the small defect is the truncated `K<500` left tail).

**Results — Phase B (dupire smile, MC-grounded `Σ_φ`).** `Σ_φ` from the MC price standard error (`mc_arrays/dupire_paper`, T≤1.0). The MC-implied σ point estimate **tracks the true Dupire smile tightly in the body** (`plots/mz_uq/plot_B_mc_sigma.png`); the pole gate flags **73%** of cells feasible (99.9% convex); the delta band is **conservative** (covers the true σ on all feasible cells). The regularized MC band (Phase A) is the calibrated choice; delta is the cheap conservative baseline.

**Honest scoping.** This is dynamics-consistent Bayesian inverse UQ; the MZ content is the resolved/unresolved split and the de-risk-confirmed *decaying* propagator that keeps the unresolved-uncertainty contribution finite (the well-posedness HW lacked). It does **not** need FDT (which holds only in the Gaussian/no-tail case). The G1 deliverable: **calibrated σ/f bands that widen exactly where the data is uninformative**, replacing Privault's point estimate + hard discard mask.

**Reproduce:** `./.venv/bin/python examples/run_mz_uq.py` → `mz_uq.json`, `plots/mz_uq/{plot_A_bands,plot_B_mc_sigma}.png`. Code: `mz_spectral/uncertainty.py`.

---

## 11. Research program and conclusion

| Step | What | Cost | Status |
|---|---|---|---|
| 0 | operator dissipativity + forward-memory-cosmetic test | hours, CPU | **done (§8): SEED A ✓, pivot ✓** |
| 1 | moment-trajectory PCA dimension | hours, CPU | **done (§8): dim 1** |
| 2 | density-side energy-budget closure: raw Gram–Charlier vs max-entropy tilt; positivity stress test vs MC | 1–2 d, CPU | next — `energy_closure.py`; validate with `pdf_metrics` + `compute_mean_diagnostics` on `mc_arrays/dupire_paper` |
| 3 | MZ-regularized tail inversion vs Tikhonov/spectral-filter | 2–3 d, CPU | next — `regularize_sigma_memory` in `sigma_from_phi.py`; metric = tail `‖σ̂-σ_{oracle}‖` |
| 4 | `L_MZ` + martingale losses in the PINN; DAX/SPX benchmark | ~1 wk, **GPU/off-Mac** | next — `dupire_pipeline.py`; sparse-data efficiency vs WSPG25 |

**Status this pass.** (i) The `ν_QL` `ν≥0` fix is **done** (`quasi_linear.py` `nonneg=True`; `docs/MZ_QL_NONNEG_FIX_REPORT.md`). (ii) The **epistemic-UQ extension (§10)** — calibrated σ/f confidence bands — is **built and validated** (`mz_spectral/uncertainty.py`, `examples/run_mz_uq.py`), delivering the G1 uncertainty layer (weak-form density bands + regularized σ bands) ahead of the heavier Steps 2/4 (the `E_tail` energy-budget closure and the PINN extension).

**Conclusion.** The Dupire problem is, structurally, the *well-posed dual* of SLGG's plasma closure: the forward operator is dissipative, so the MZ memory converges and the violent pathologies are absent — but for the same reason the forward closure is inert (§8). SLGG's hard-won technology nonetheless transfers, on the **density side**, where its sign-definiteness requirement *is* no-arbitrage and its energy-budget scalar *is* a tail-energy scalar that makes the risk-neutral density valid by construction. That density-side energy-budget closure (§5), plus the calendar-coupled inverse regularizer (§6) and the MZ-consistency PINN loss (§7), is a concrete, falsifiable extension of WSPG25 that targets both local volatility (G1) and the risk-neutral density (G2).

---

### Reproduce
```bash
cd Synthetic_Data_Tensorflow_Advanced
./.venv/bin/python examples/run_mz_derisk.py   # -> mz_derisk.json ; plots/mz_derisk/{plot_0a_abscissa,plot_0b_pdf,plot_1_pca}.png
./.venv/bin/python examples/run_mz_uq.py        # -> mz_uq.json ; plots/mz_uq/{plot_A_bands,plot_B_mc_sigma}.png
```

### Key code touchpoints (for Steps 2–4)
- `mz_spectral/validation.py` — `integrate_rrr_ql` (template for the energy-budget integrator), `pdf_from_phi_tilde`, `pdf_metrics` (G2 scorecard).
- `mz_spectral/quasi_linear.py` — `fit_nu_ql` (the scalar the §5 ODE replaces; **fix the symmetric clip**).
- `mz_spectral/sigma_from_phi.py` — `sigma_reliability_mask`, `regularize_sigma_grid` (the §6 target).
- `mz_spectral/mz_decomposition.py` — `low_pass_projector_matrix` (add the eigenbasis/weighted projector, §8 Step 0a cure).
- `dupire_pipeline.py` — `compute_mean_diagnostics` (martingale = (5.5)), `loss_dupire_cal`/`train_step` (the §7 losses).
