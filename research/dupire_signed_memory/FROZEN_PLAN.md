# Frozen experiment plan — signed memory in the Dupire reduced model

Frozen 2026-11-04 before any closure was scored. Later edits are only allowed as dated
addenda at the end of this file; nothing above the "Addenda" heading changes.

## Question

Does the Fourier-truncated Dupire reduced model in `Synthetic_Data_Tensorflow_Advanced/mz_spectral`
need **signed, stateful** feedback from the eliminated modes, or is the nonnegative
scalar eddy-diffusion restriction (`fit_nu_ql(nonneg=True)`) justified on
variable-volatility problems?

Hypotheses tested (each can fail independently):

* **H-sign** — "eliminated modes must add diffusion" (the report's premise): the exact
  memory term, projected on the QL regressor `q = P_R D2 φ_R`, has a nonnegative coefficient.
* **H-scalar** — a single scalar Markovian coefficient (either sign) captures a useful
  fraction of the exact-memory gap.
* **H-state** — a compact stateful surrogate driven only by the resolved state closes a
  larger fraction of the gap than any scalar closure, out of sample.

## Operator variants (fixed)

* **OP-A, repository-actual.** `D2` with the production one-sided boundary rows,
  `L = diag(η̃ k̃²) D2`, Euclidean Fourier low-pass projector applied to the raw `φ̃` row.
  This is the model whose restriction is under test; its artifacts are reported, not repaired.
* **OP-B, boundary-consistent.** Same interior stencil, boundary rows replaced by the
  linearity condition `∂²φ̃/∂k̃² = 0` at both ends (standard call-pricing boundary; boundary
  values are then constant in time), projector applied to the detrended variable
  `ψ = φ̃ - ℓ(k̃)` with `ℓ` the linear interpolant of the two boundary values. `ψ` vanishes at
  both ends, `ψ_t = Lψ` exactly, and the orthogonal block is dissipative. This is the
  minimal change that makes the memory a convergent object; it is used to answer the
  scientific question on a well-posed footing.
* **OP-C, Dirichlet-forced (Gate 1–2 only).** OP-B stencil with time-dependent boundary data
  from Black–Scholes, giving the boundary-forcing term `b(t)` of the contract. Used only
  to verify the equivalence identity including boundary forcing.

## Case matrix (fixed; all deterministic from committed code)

Family C (constant σ, exact BS reference): σ ∈ {0.20, 0.25, 0.30, 0.35, 0.40}.
Family S (smooth strike-dependent, time-independent): `σ(x) = clip(b(1 + s(1-x) + c(x-1)²), 0.05, 2)`,
`x = K/S0`, parameters drawn from `numpy.random.default_rng(seed)` with
`b ~ U(0.2,0.4)`, `s ~ U(0,0.10)`, `c ~ U(0,0.05)`, seeds 0–9.
Family D (repository smile, time- and strike-dependent): `VolatilityConfig.dupire_exact(sigma_base=b)`,
`b ∈ {0.25, 0.30, 0.35}`.

Splits (complete trajectories stay together):
* train: C{0.20, 0.30, 0.40}, S seeds {0,1,2,3,4,5}, D{0.30}
* validation (ridge selection only): C{0.25}, S seeds {6,7}, D{0.25}
* final held-out (untouched until final scoring): C{0.35}, S seeds {8,9}, D{0.35}

Truncations: `keep ∈ {16, 32, 64}` (`dim R = 33, 65, 129`); `n_k = 256` primary,
`n_k = 512` refinement on C{0.30}, S seed 8, D{0.30} for the uncertainty band.
Repository constants throughout: `S0=1000, r=0.04, K_min=500, K_max=3000, T_min=0.3, T_max=1.5`.

Time handling: the reference (full) solution starts from the payoff at `t̃=0` (the time-origin
offset of the existing drivers is not reproduced). All reduced models start at `t̃0 = 0.2`
from the projected reference state `P_R φ(t̃0)` with the nonzero unresolved remainder
`P_U φ(t̃0)` deliberately discarded (so the unresolved initial-condition term is exercised),
and run to `t̃ = 1` with 400 uniform steps; outputs are scored at the 24 repository maturities.
Reference integration: exact `expm` steps for time-independent `L` (C, S), second-order Magnus
(midpoint `expm`) with 800 steps for D (convergence checked against 400 steps in Gate 2).

## Closures compared (identical inputs and initial conditions)

| id | description | parameters fitted |
|---|---|---|
| RRR | resolved-only `c' = L̂_RR c` | none |
| QL+ | existing nonnegative QL, `ν ∈ [0, 5e-4]` | `ν` |
| QL± | otherwise identical signed QL, `ν ∈ [-5e-4, 5e-4]` | `ν` |
| QL* | unclipped least-squares `ν` (reported for sign/stability analysis; scored like the others) | `ν` |
| ORACLE | exact memory term `M(t) = P_R L P_U φ(t)` from the full solve, same integrator | none (not deployable) |
| LM | linear Markovian ridge map `M ≈ C c` (matched non-neural, stateless) | `C` |
| LR | linear reservoir: `s_{n+1} = (1-a)s_n + a(A s_n + B c_n)`, `M ≈ W s`, fixed random `A,B` | `W` |
| ESN | same as LR with `tanh`, the compact deployable reservoir surrogate | `W` |

LM/LR/ESN are added only if Gate 3 shows a reproducible ORACLE gap (RRR error above the refinement
band on every training case). Fitting targets come from the operator (`M(t)` on the fine grid),
never from finite-differenced trajectories. Surrogates are driven by the reference resolved
state during fitting (teacher forcing) and by their own state during scoring; the true unresolved
state is never an online input. `ν` for QL variants is fitted by pooled ridge LS over the training
trajectories (same formula as `fit_nu_ql`, `ridge=1e-8`), applied unchanged to validation and
held-out cases. Fitting budget: reservoir size 200, spectral radius 0.9, leak 0.5, input scale
1.0, seed 0; ridge for LM/LR/ESN chosen on the validation split from {1e-8, 1e-6, 1e-4, 1e-2};
no other hyperparameter is varied. Normalisation statistics come from the training split only.

## Metrics (per case, per keep; aggregated across cases, never across time steps)

* resolved rollout error: RMS of `V_R c_model - P_R φ_ref` on the repository interior band
  (5–95 % strike quantiles, `interior_k_mask`) and on the full domain
* price error: RMS of the reconstructed `φ̃` against the full semidiscrete solution, and against
  Black–Scholes for family C
* density: Breeden–Litzenberger `f = e^{rT} S0 ∂²φ̃/∂K²` with the repository `D2`, L2 error vs
  reference, negative mass `∫ max(-f,0) dK`, reported without clipping
* no-arbitrage: fraction of interior points with butterfly `φ̃_kk < 0` and calendar `φ̃_t < 0`
* inverse volatility: fraction of interior points where `σ̂² = 2 φ̃_t / (T_max k̃² φ̃_kk)` is
  undefined or negative
* stability: non-finite values or `‖c‖` growth > 10× the reference norm ⇒ `unstable=True`,
  scored as failure (no clipping, no discarding); spectral abscissa of the closed autonomous
  operator `L̂_RR + ν P̂_R D2 P̂_R` reported separately
* sign diagnostics: fitted `ν`, operator-based `ν*`, instantaneous resolved work
  `⟨c, M⟩` sign statistics in the Euclidean and `D⁻¹`-weighted metrics

Refinement uncertainty: `u = |err_RRR(n_k=256) - err_RRR(n_k=512)|` on the refinement cases;
a difference between closures smaller than `u` is reported as unresolved.

## Prespecified acceptance criteria and verdict rules

* A closure *helps* at a given `keep` if, on **every** held-out case, it is stable and its
  resolved-interior error is below RRR's by more than `u`, and it does not increase negative
  density mass or invalid inverse-vol fraction relative to RRR.
* **Restriction justified**: QL± never helps where QL+ does not, and QL* is unstable or harmful
  whenever `ν* < 0`.
* **Restriction overly restrictive**: QL± (or QL*) helps on OP-B at some `keep` with `ν < 0`,
  stable rollouts, and no no-arbitrage degradation.
* **Unresolved**: anything else, including gains inside the refinement band.
* H-state is supported only if ESN or LR helps where every scalar closure does not, on the
  held-out split, with the fixed budget above.

## Compute and cloud budget

CPU-only NumPy/SciPy (`n_k ≤ 512`, matrices ≤ 512×512). Target wall time under 30 minutes for
the full matrix. Cloud credit accounting: see `BUDGET.md`.

## Out of scope

Market data, trading, the completed P9 tail-energy arbitrage detector, plasma/DNS claims,
TensorFlow/GPU training, changes to production defaults in `mz_spectral`.

## Addenda
(none yet)

### Addendum 1 (2026-11-04, before any closure was scored)

* Time stepping: the reduced models use 368 = 23·16 uniform steps on `[0.2, 1]` (not 400) so that
  the 24 repository maturities lie exactly on the fine grid; the reference uses the same step.
* Diagnostic only, not a scored closure: `scripts/gate2b_basis_diagnostic.py` reports the
  resolved-only (Galerkin) error of OP-B at matched dimension `2·keep+1` for three bases —
  the repository Fourier modes, the Dirichlet sine modes `sin(jπ(k̃-k̃_lo)/(k̃_hi-k̃_lo))`, and the
  weighted eigenbasis of `L(t̃=1)` — to quantify how much of the Fourier "memory" is the
  wrap-around derivative mismatch of a periodic basis applied to a non-periodic function.
  The scored comparison (Gate 3) stays with the repository projector as frozen above.
* `mz.kernel_norms` reports overflow of the divergent OP-A orthogonal propagator as `inf`
  instead of raising (robustness fix found during the Gate 2 run; no change to any number).
