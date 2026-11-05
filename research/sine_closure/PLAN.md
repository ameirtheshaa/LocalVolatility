# Frozen plan v1 — closure test for small boundary-adapted sine models

Frozen 2026-11-05 at source commit `c028ae45ec94e490b87d269df41769fff71b7d57` (the PR #9 merge,
verified as the checkout SHA) **before any new held-out trajectory was scored**. Everything above
the "Addenda" heading is binding. A later change that affects test scoring requires a dated,
versioned addendum *and* either a new test set or an explicit "exploratory" label on the result.

Study directory: `research/sine_closure/` (package `scc`). The earlier study
`research/dupire_signed_memory/` (Gates 0–5) is imported read-only; nothing there is rerun into
its own result files, superseded or overwritten. Production defaults in
`Synthetic_Data_Tensorflow_Advanced` are untouched.

## 0. Question and scope

Does a deployable closure let a 9- or 17-mode boundary-adapted sine model (OP-B operator,
detrended price) match or improve the useful accuracy of the 33-mode unclosed sine model, at
lower online cost, without losing density validity? Improvements are reported separately for
resolved dynamics, reconstructed price, reconstructed density and inverse local volatility.
A negative result is a result: if closure adds complexity without a reliable benefit, the
recommendation stays with the unclosed 33-mode model.

Not claimed: market generalisation (all data are synthetic semidiscrete references), anything
about the production Fourier model beyond regression rows, and anything about nonlinear or
neural closures (not run; see §5).

## 1. Frozen scientific questions

Primary (each answered per dimension m ∈ {9, 17}, per closure, per start protocol):

* **Q1 resolved dynamics.** Does the free-running closed model reduce the resolved-coordinate
  error beyond numerical uncertainty?
* **Q2 reconstruction.** Does that survive in the reconstructed price *and* density, including
  negative mass and arbitrage checks?
* **Q3 compression.** Does a closed 9/17-mode model meet the prespecified 33-mode
  accuracy/validity target at genuinely lower measured online cost (hidden states counted)?
* **Q4 stateful vs stateless.** Does the linear state-space closure (LR) beat the matched
  stateless linear map (LM) by enough to justify hidden state, training and runtime?

Secondary, separate and not needed for Q1–Q4: **S1** does a 33-mode closed model (LM-33, QL-33
only) improve the density beyond its refined numerical uncertainty?

## 2. Mathematical contract (fixed)

Scaled variables and operator exactly as `dsm.operators` (production constants `S0=1000`,
`r=0.04`, `K_min=500`, `K_max=3000`, `T_min=0.3`, `T_max=1.5`, 24 output maturities
`T ∈ linspace(0.3, 1.5, 24)`, `t̃ = T/T_max ∈ [0.2, 1]`):

    φ'(t) = L(t) φ(t) + b(t),   L(t) = diag(η̃(t,k̃_i) k̃_i²) D2,   η̃ = (T_max/2) σ(T, K/S0)²,
    K = k̃ K_max e^{rT},  k̃ ∈ [e^{-r T_max} K_min/K_max, e^{-r T_min}] = [0.15696, 0.98807].

* **Operator OP-B** (`dsm.operators.d2_linearity`): central interior second differences, zero
  boundary rows, endpoint values fixed; `b ≡ 0`.
* **Lift.** `ℓ` = affine interpolant of the two endpoint values of the payoff (constant in time
  for OP-B), `ψ = φ − ℓ`, so `ψ' = Lψ + g` with `g = Lℓ + b − ℓ'`. For OP-B `g` must vanish; this
  is **asserted by a test** (`‖g‖ ≤ 1e-12 · ‖diag(d)‖·‖D2‖·‖ℓ‖`), never compensated by an
  intercept. The time-dependent Dirichlet variant (Black–Scholes boundary data, OP-C of the
  earlier study) is run only as a boundary-sensitivity check in the verification stage, with
  its nonzero forcing `g = −ℓ'` derived and retained (`Lℓ = 0` still holds).
* **Basis (DST-I, deterministic sign).** On `n` uniform points,
  `V[i, j] = sqrt(2/(n−1)) · sin(π j i/(n−1))`, `i = 0..n−1`, `j = 1..m`; endpoint rows are
  exactly zero; the first interior entry of every column is positive. No QR. `VᵀV = I` exactly
  (to roundoff) because the sampled sines are the DST-I eigenvectors of the interior second
  difference: `D2_int V_int = V_int Λ_V`, `λ_j = −(4/dk²) sin²(π j/(2(n−1)))`. Therefore the QL
  regressor `Q̂ = Vᵀ D2 V = Λ_V` is diagonal. The Gate 5 basis (QR of the same sampled sines)
  spans the same subspace; Galerkin rollouts are invariant to the orthonormal re-basing, which
  is what the Gate 5 reproduction checks.
* **Interior state space and explicit complement.** `ψ_int ∈ ℝ^{n−2}`; `V_int` = DST columns
  `j = 1..m`, `U_int` = columns `j = m+1..n−2`; `[V U]` is orthogonal on the interior. Blocks:
  `A = Vᵀ diag(d) V Λ_V`, `B = Vᵀ diag(d) U Λ_U`, `C = Uᵀ diag(d) V Λ_V`, `D = Uᵀ diag(d) U Λ_U`
  with `d_i = η̃(t,k̃_i) k̃_i²` on interior points. `A` is assembled by the structured
  contraction `(Vᵀ diag(d) V) Λ_V` (cost `O(n m²)`), never by a dense `n×n` product; the dense
  product is used only in tests. The full-space embedding `Q = I − VVᵀ` of the earlier study is
  not used for block calculations (its two artificial zero eigenvalues are not physical modes).
* **Exact resolved equation** (contract every closure approximates), `c = Vᵀψ`, `z = Uᵀψ`:

      c' = A c + g_R + B Φ_D(t,t0) z0 + ∫ B(t) Φ_D(t,s) C(s) c(s) ds + ∫ B(t) Φ_D(t,s) g_U(s) ds,

  `Φ_D` the time-ordered propagator of `z' = D(t) z` (`e^{(t−s)D}` when autonomous). The
  operator target along a full reference is `M(t) = Vᵀ L(t)(I − VVᵀ) ψ_ref(t) = B(t) z_ref(t)`;
  it contains the unresolved-initial-condition, memory and forcing contributions and is never
  estimated by time differencing.
* **Time stepping.** Uniform `h = 0.8/(23·sub)` on `t̃ ∈ [0, 1]` so that the 24 maturities and
  `t̃0 = 0.2` are grid points; `sub = 16` primary (`h = 2.1739e-3`, 460 steps), `sub = 32, 64`
  for step refinement. Full solves: midpoint exponential (second-order Magnus), identical to
  `dsm.reference.solve_full` variant "B". Reduced models and every linear closure: one exact
  exponential of the closed matrix at `t_mid` per step (one in total when σ is
  time-independent, cached identically for every model). Oracle rows that carry a forcing use
  the φ₁ midpoint rule.

## 3. Grids, references, refinement and uncertainty

* Model grid `n = 256`. Refinement grids are **nested**: `n = 511` (every second point is a
  256-grid point) and `n = 1021` (every fourth); verified to coincide to 0.0. Refined references
  are restricted to the 256 grid by index, so no interpolation enters the primary scores. This
  deviates from Gate 5 (512 points + cubic spline); the Gate 5 regression check runs the
  original script unchanged, and the spline-vs-restriction difference is reported once on
  development data as a diagnostic.
* Reference solver: dense `expm` midpoint stepping for `n ∈ {256, 511}`; sparse
  `expm_multiply` with the same midpoint scheme (tolerance 1e-13) for `n = 1021` and for
  `sub = 64`. Dense and sparse must agree to `1e-11` (sup-norm, relative to `max|φ|`) at
  `n = 256` on two development cases before any sparse result is used.
* Reference levels per case: `Full256@16` (same-grid truth: isolates truncation/closure error;
  also the fitting source), `Full511@16` (end-to-end truth), `Full256@32` (step uncertainty).
  On all test-interpolation cases and on two development cases per family: `Full1021@16` and
  `Full256@64` (convergence order). Dense `expm` is a reference solver, not a production
  solver (§9).
* **Uncertainty per quantity `q` and case** (frozen): `u_q = max(u_q^grid, u_q^step)` where
  `u_q^grid` is `q` scored with `Full256@16` as the model and `Full511@16` as truth (for
  error-type `q`: the error of the same-grid full solve itself; for fractions, masses and
  moments: the absolute difference between the two levels), and `u_q^step` is the same with
  `Full256@32` as model and `Full256@16` as truth. Reported, separately, for price (interior
  RMS), density (interior L2, L1), negative mass, each validity fraction and the inverse-vol
  error. Where the 1021/64 levels exist the observed convergence order is reported; `u` is
  **not** reduced by it. For the resolved-coordinate error (Q1) the uncertainty is
  `u_dyn = max(interior RMS of V(c_ref@16 − c_ref@32), e_TF)` with `e_TF` the teacher-forced
  oracle's own rollout error (integrator floor).
* Domain check (development cases only, diagnostic, not folded into `u`): strike domain
  `K ∈ [250, 6000]` with `n = 587` points (equal physical spacing to the base grid), compared
  with the base-domain reference on the common region `K ∈ [500, 3000]`; the interpolation of
  the wide-domain solution onto the base grid is done twice (cubic spline and monotone PCHIP)
  and both numbers are reported with their difference.
* Roundoff guard `γ` for the negative-mass rule: determined on development data as `10 ×` the
  largest paired `|Δ negative mass|` between two mathematically identical but
  roundoff-different rollouts (basis with all column signs flipped; summation in reversed
  order), recorded in `results/verify/roundoff_guard.json` before test scoring.

## 4. Cases and splits

All cases are generated deterministically by `scc/cases.py` from the formulas and seeds below;
`results/manifests/CASES.json` records ids, parameters, splits and the sha256 of the sorted
(id, family, params, split, generator) list. Whole trajectories are the split unit. Physical
maturity `T = t̃ T_max` and moneyness `x = K/S0` enter σ; σ is clipped to `[0.05, 2.0]` where
stated, and the manifest discloses per case: min/max σ on the grid, the fraction of grid
points where the clip is active, and the largest relative σ jump between neighbouring strike
points and between neighbouring output maturities (nonsmoothness disclosure).

| family | σ(T, x) | interpolation ranges (train / val / test-interp) | OOD stress (test-ood) |
|---|---|---|---|
| C constant | σ | train {0.18, 0.22, 0.27, 0.33, 0.38, 0.44}; val {0.24, 0.36}; test {0.205, 0.31, 0.415} | {0.12, 0.55} |
| S strike smile | `clip(b(1 + s(1−x) + c(x−1)²), 0.05, 2)` | `b~U(0.18,0.42), s~U(0,0.25), c~U(0,0.15)`; seeds train 100–109, val 110–113, test 120–125 | `b~U(0.15,0.5), s~U(0.25,0.5), c~U(0.15,0.4)`; seeds 130–133 |
| D repository-type time+strike | `base + y e^{−y}`, `y = sqrt(x + x_s)(T + t_s)` | `base~U(0.2,0.35), t_s~U(0.05,0.3), x_s~U(0.02,0.2)`; seeds train 200–205, val 210–211, test 220–223 | `base~U(0.35,0.5), t_s~U(0.3,0.6), x_s~U(0.2,0.5)`; seeds 230–231 |
| M temporally modulated smile | `clip(b(1 + a sin(2π f T/T_max + φ))(1 + s(1−x) + c(x−1)²), 0.05, 2)` | `b~U(0.2,0.4), a~U(0.1,0.3), f~U(0.5,1.5), φ~U(0,2π), s~U(0,0.2), c~U(0,0.1)`; seeds train 300–305, val 310–311, test 320–323 | `a~U(0.3,0.5), f~U(1.5,3), s~U(0.2,0.4), c~U(0.1,0.3)` (b, φ as interp.); seeds 330–331 |

Parameters are drawn from `numpy.random.default_rng(seed)` in the column order listed, rounded
to 6 decimals and stored. Counts: train 28, val 10, test-interp 17, test-ood 10. The three
Gate 5 cases (`C_sig0.30`, `S_seed9`, `F_fresh_base0.28_t0.2_x0.05`) are **development
regression rows** (split `dev_regression`), never training data and never held-out evidence.
Short maturities are covered by the payoff-start early-time panel (§6); the operator stays
well posed for every listed range (σ > 0 and bounded).

Leakage rules: normalisation statistics, ridge / hidden-size / rate-bank selection, the
roundoff guard, the representation-limited stopping rule and every tolerance are computed from
`train`, `val` and `dev_regression` only. Test references are generated by a separate script
into a separate cache namespace; the fitting and selection scripts run with `SCC_PHASE=fit`,
under which the cache layer refuses to open any `test_*` reference.

## 5. Dimensions, models and budgets

Closed models at `m ∈ {9, 17}`; unclosed RRR at `m ∈ {9, 13, 17, 25, 33, 65}` (13 and 25 are
the intermediate unclosed dimensions needed for total-state comparisons: LR totals are
9+8 = 17, 9+16 = 25, 17+8 = 25, 17+16 = 33). Full rows: `Full256` (dense reference scheme),
`Full256-sparse` and `Full256-CN` (cost candidates, §9), `Full511`, selected `Full1021`.

| id | model | fitted | parameter count |
|---|---|---|---|
| RRR | `c' = A c` | none | 0 |
| QL+ | `c' = A c + ν Λ_V c`, `ν = max(ν*, 0)` | ν | 1 |
| QL± | same, `ν = ν*` if `ν* > ν_adm/2` else `ν_adm/2` | ν | 1 |
| QL* | same, unclipped `ν*` (scored like the others; may be unstable) | ν | 1 |
| LM | `M̂ = H c` | H | m² |
| LM+b | `M̂ = H c + b` (declared intercept ablation only) | H, b | m² + m |
| LR0 | `h' = F h + G₀ c`, `M̂ = H c + R h`, F = −diag(rate bank), G₀ fixed | H, R | m² + m·n_h (+ m·n_h fixed) |
| LR | as LR0 with G learned by alternating least squares | H, R, G | m² + 2 m·n_h |
| WI | hidden initialiser `h(t0) = K [c(t0); 1]` for the warm start | K | n_h (m+1) |
| PROJ | exact projected reconstruction `ℓ + V Vᵀ ψ_ref` (oracle) | — | — |
| BLOCK | exact block/MZ reconstruction with the true `z0` (correctness oracle) | — | — |
| TF | teacher-forced rollout with `M_ref(t)` (integrator floor, diagnostic) | — | — |

`ν*` is the weighted pooled least-squares fit `Σ w ⟨M, Λ c⟩ / Σ w ‖Λ c‖²` on the training set;
`ν_adm = −min_{train, t} λ_min(Vᵀ diag(d(t)) V)` is the admissible bound (the closed matrix
`(S + νI)Λ_V` has a negative real spectrum iff `S + νI ≻ 0`). The unclipped fit, the bound, the
selected values and a sensitivity panel `ν ∈ {0, ¼, ½, 1, 2}·ν*` and `−ν*` on the validation
split are all reported; the Fourier-era clip `5e-4` is not transplanted.

Hyperparameter budget (selected on validation only): ridge `λ ∈ {1e-10, 1e-8, 1e-6, 1e-4, 1e-2} ×
tr(Gram)/dim` (the same relative grid for H, R, G and K); hidden sizes `n_h ∈ {8, 16}`; rate banks
"N" (`[0.5 r₁, 16 r₁ d_max/d_min]`) and "W" (`[0.25 r₁, 64 r₁ d_max/d_min]`), log-spaced,
`r₁ = d_min |λ_{m+1}|`, with `d_min, d_max` the extremes of `d_i(t)` over the training set
(recorded constants). ALS: at most 30 iterations, stop when the relative objective decrease is
below 1e-6; `G₀` = standard normal / √m with seed 0. Nothing else is varied. Selection criterion:
geometric mean over validation cases and the protocols {P0, P1-W0} of the free-rollout
resolved interior RMS `e_dyn`; any instability disqualifies a configuration; ties go to the
smaller `n_h`, then the larger `λ`. Selected configurations and weights are locked in
`models/LOCK.json` (sha256 of every weight file) before the test-scoring script is run.

No nonlinear reservoir / ESN / neural closure is run in this plan. Escalation to one requires
(i) LR showing a robust held-out gain over LM under §10 and (ii) diagnostics indicating
unresolved structure that linear models miss, and would be pre-registered separately.

## 6. Start protocols and panels

* **P0 payoff start** (`t̃ = 0 → 1`): `c(0) = Vᵀ ψ(0)`, `ψ(0) = payoff − ℓ`; hidden state
  `h(0) = 0` (the payoff is identical for every case, so a learned initialiser is a constant and
  is not run). Scored at the 24 maturities and at the early-time panel `t̃ ∈ {0.05, 0.10, 0.15}`
  (fine-grid indices `23·sub/16, 46·sub/16, 69·sub/16`). No pointwise density is evaluated at
  `t̃ = 0`.
* **P1 warm start** (`t̃0 = 0.2`): `c(t0) = Vᵀ ψ_ref,256@16(t0)` (same-grid full reference; the
  discarded `z0` is never given to a deployable model). Hidden-state protocols, reported as
  separate rows: `W0` zero; `WI` learned initialiser `K[c(t0); 1]`; `WU` causal warm-up (run
  the closed model from the payoff on `[0, t0]` with its own state, keep `h(t0)`, reset `c` to
  the projection). Scored at the 24 maturities; row 1 is the shared projected initial
  condition, so the negative-mass maximum is taken over rows 2–24 ("post-initial") and also
  reported at the final row.
* Oracle rows: `PROJ` and `TF` under both protocols; `BLOCK` with the true `z0` under both. The
  secondary identity test with `z0 = 0` is compared with a full solve started from `ℓ + V c0`,
  never with the unprojected reference.

## 7. Fitting

Targets: for every training trajectory, `c(t_n) = Vᵀ ψ_ref,256@16(t_n)`, `z(t_n)`, and
`M(t_n) = B(t_n) z(t_n)` at every fine step from `t̃ = 0` (461 rows). Trajectory-balanced
weights `w_case = 1 / (N_rows · max(RMS_case(M), 0.1 · median_cases RMS(M))²)` (so neither long
nor high-amplitude paths dominate, and near-zero-memory cases do not blow up the weights).
Hidden states for LR fitting come from exact exponential filtering of the piecewise-linear
`c_ref` (teacher forcing); the true `z` is never an online input. `WI` is fitted by weighted
ridge from the teacher-forced `h(t0)` against `[c(t0); 1]` over the training trajectories.
Training diagnostics (objective per ALS iteration, condition numbers, effective ranks) and
per-case fit residuals are saved with the weights.

## 8. Metrics (definitions frozen)

Evaluation grid: the 256 model grid at the 24 output maturities (plus the early panel for P0).
Interior band = the repository `interior_k_mask` (5–95 % strike quantiles per maturity). ATM
band `|K/S0 − 1| ≤ 0.10`; wings `K/S0 ≤ 0.7` or `≥ 1.5`.

* **Price.** Interior RMS (primary), full-domain RMS, max abs, per-maturity RMS, ATM and wing
  RMS, physical-price RMS (× S0), resolved-coordinate error `V(c − c_ref)` (interior RMS and
  coefficient RMS), Black–Scholes analytic truth for family C. Reported against the same-grid
  (`Full256@16`) and the refined (`Full511@16`) truth; the primary verdict quantity for Q2/Q3
  is the refined-grid interior RMS.
* **Density.** Primary definition: Breeden–Litzenberger `f = e^{rT} S0 ∂²φ̃/∂K²` with the
  repository `D2` (one-sided boundary rows) on the 256 grid for every row (references computed
  on their own grid, then restricted). Diagnostic definitions: zero-boundary-row `D2`,
  analytic sine second derivatives (spectral, models only) and a fourth-order central stencil;
  their mutual differences are reported as consistency diagnostics, not as alternative truths.
  Metrics: interior L2 (primary) and L1, full-domain L2, max and 99 %-quantile abs error,
  signed total mass per maturity, negative mass `∫ max(−f, 0) dK` **at every maturity**, its
  maximum over post-initial rows and its final-row value, restricted-domain mass versus the
  reference mass, first moment `∫ K f dK` versus reference, tail mass outside the interior
  band. No clipping, renormalisation, smoothing or filtering anywhere in primary scores.
* **Arbitrage** (in the scaled variables, where `∂φ̃/∂t̃ ≥ 0`, `∂φ̃/∂k̃ ≤ 0`, `∂²φ̃/∂k̃² ≥ 0`,
  `φ̃ ≥ 0` are the no-arbitrage conditions). Raw violation fractions with a declared roundoff
  guard `1e-12 × row max` (positivity, monotonicity, butterfly, calendar) over the interior band
  and over the whole domain, plus the repository's `1e-3`-relative version for comparability.
  Time derivatives for the calendar check and for inverse volatility come from the model's own
  right-hand side (`V(A c + M̂)` for reduced models, `Lφ` for full solves) and are compared with
  central differences on the fine grid at `sub = 16` and `32` (diagnostic).
* **Inverse volatility.** `σ̂² = 2 ∂_t̃ φ̃ / (T_max k̃² ∂²_k̃ φ̃)`. Fixed identifiable mask shared by
  every model: interior-band points where the refined reference's curvature exceeds
  `rel_tol × row max` (`rel_tol = 1e-3`, the repository metric). Reported: undefined fraction
  (model curvature below threshold), negative/nonfinite fraction, whole-domain invalid fraction
  (undefined ∪ negative ∪ nonfinite over the whole grid), coverage (fraction of identifiable
  points where the model's `σ̂` is valid), σ relative error (median and RMS) on identifiable ∩
  valid points against the known σ(T, K/S0) — always with the coverage beside it — and a
  threshold-sensitivity panel `rel_tol ∈ {1e-2, 1e-3, 1e-4}`.
* **Stability.** Instability flag (non-finite state or `‖c‖ > 10 ×` the reference maximum ⇒
  scored as failure, kept in every table), spectral and numerical abscissae of the closed
  matrix at every output time (maxima reported), maximum state and output amplification,
  perturbation response (rollout from `c0 + 1e-6 ‖c0‖ v`, fixed unit `v`, seed 0; maximal
  growth ratio of the difference), propagator growth `max_t ‖Φ_closed(t, t0)‖₂` for
  time-dependent cases, and a long-horizon stress rollout to `t̃ = 3` with the operator held at
  its `t̃ = 1` value beyond the horizon. A stable `F` alone is never cited as closure stability.

## 9. Cost protocol

Offline (reported once per model): reference generation, target construction, fitting and
hyperparameter search wall time. Online (per query, reported per model and case): initialisation
(projection of the payoff / warm-start state, hidden initialiser), per-step operator assembly
(`Vᵀ diag(d) V`), exponentiation, stepping, reconstruction at the 24 maturities, density
evaluation and inverse-volatility evaluation, each timed separately and summed. Autonomous
cases cache the single exponential for every model alike; time-dependent cases pay assembly and
exponentiation per step. Full-solver cost candidates: dense `expm` (reference scheme), sparse
`expm_multiply` (same scheme), banded Crank–Nicolson with two Rannacher start steps whose
substep count is the smallest power of two that brings its price error against `Full256@16`
below the case's `u_price` (equivalent convergence target). Machine: the GPU-laptop CPU lane,
run exclusively (no other job), threads pinned to 1 (`OMP/MKL/OPENBLAS_NUM_THREADS`, plus
`threadpoolctl` when available), 7 repeats (median, IQR, min), cold = first call in a fresh
process, warm = after two untimed calls, model order randomised per repeat (seeded), hardware,
BLAS, versions and precision captured. A cost difference counts only when the medians differ by
more than the larger of the two IQRs. Deliverables: error-versus-runtime and
error-versus-total-state Pareto frontiers, and the break-even deployment count
`N* = offline cost / (online cost of RRR-33 − online cost of the closed model)`.

## 10. Decision rules (frozen before test scoring)

Multipliers: a difference is **beyond uncertainty** when `|Δq| > 2 u_q,case`; model `X` is
**noninferior** to `Y` in `q` when `q_X ≤ q_Y + u_q,case`; **negative mass**: strict paired
non-increase, `negmass_X ≤ negmass_Y + γ` with the verified roundoff guard `γ` of §3 (no 5 %
relaxation anywhere; a second, application-tolerance assessment is not part of this plan and
would need its own addendum); **validity fractions**: `frac_X ≤ frac_Y + max(u, 1/N_interior)`.
Primary held-out evidence: the `test_interp` cases under both start protocols; `test_ood` cases
are reported in full but enter only the "selected regimes" wording.

Classification of each (m, closure, hidden protocol):

* **Resolved-dynamics improvement** — on every test-interp case and both protocols: stable and
  `e_dyn(closure) < e_dyn(RRR_m) − 2 u_dyn`. Not deployment success by itself.
* **Useful closure at fixed m** — resolved-dynamics improvement, *and* the refined-grid price
  interior RMS improves beyond uncertainty on every case, *and* noninferiority in density L2,
  negative mass (strict), every arbitrage fraction, the inverse-vol invalid fraction and
  coverage on every case. Every exception is listed per case; no averaging hides a regression.
* **Successful compression** — useful closure *and*, on every test-interp case, price ≤
  `RRR-33 + u_price`, density L2 ≤ `RRR-33 + u_dens`, negative mass ≤ `RRR-33 + γ`, inverse-vol
  invalid fraction ≤ `RRR-33 + max(u, 1/N_interior)`, *and* online cost below RRR-33 beyond
  timing variability, counting hidden states, initialiser and per-step assembly. If only the
  price condition holds: **price-only compression**.
* **Stateful advantage** — the selected LR beats the selected LM at the same `m` on `e_dyn`
  beyond uncertainty on every case, is noninferior elsewhere, and this holds against the unclosed
  model of equal total state (`m + n_h`), not only against RRR-m.
* **No supported benefit** — gains within uncertainty, or the accuracy gain costs more than
  adding sine modes (an unclosed model of equal total state is at least as good), or validity
  deteriorates. Recommendation: unclosed 33 modes.
* **Representation-limited** — stopping rule, evaluated on development data before fitting at
  dimension `m`: if `PROJ` fails the 33-mode density target (`RRR-33 + u_dens`) on at least 2/3
  of the development cases *and* `e_dyn(RRR_m) − e_dyn(TF) ≤ 2 u_dyn` on at least 2/3 of them,
  then `m` is representation-limited: LR escalation at that `m` is skipped (QL and LM still run
  as cheap checks) and the verdict says so. If a fitted model lowers the density error by
  moving resolved coefficients away from `c_ref` (price up, density down), this is reported as
  output-oriented compensation with the trade-off shown, never as recovered memory.
* **Inconclusive** — the gain lies below `2u` on some but not all cases, or the reference/step
  levels disagree on the sign of the gain; the memo names the next discriminating experiment.

The memo states, for `m = 9` and `m = 17` separately, whether closure helps (and which
closure), does not help, or helps only in selected regimes (named), and whether the unclosed
33-mode model remains the recommendation.

## 11. Verification before any scoring (results/verify)

Asserted checks with recorded tolerances: `VᵀV = I`, `UᵀU = I`, `VᵀU = 0`, zero endpoint rows,
`D2_int V = V Λ`; lift residual `g = 0` for OP-B and the nonzero forcing for the Dirichlet
variant; reconstruction identity `ψ = V c + U z` on every reference row; exact autonomous block
elimination (augmented-exponential form, machine precision); nonautonomous propagator ordering
(forward order converges at second order under step refinement, reversed order fails);
`B = 0` control operator gives identically zero `M` and `TF ≡ RRR`; structured versus dense
block assembly; dense versus sparse full solver; dense versus analytic Black–Scholes on family
C; Gate 5 reproduced from the unchanged original script into `results/gate5_reproduction/`
with every non-timing metric within `rel 1e-5 + abs 1e-9` (fractions: abs 0.01) of the
committed values. The roundoff guard `γ` is measured here.

## 12. Deliverables and reproducible command plan

`docs/MEMO.md` (decision memo), `results/tables/*.md` (complete per-case model table, including
failed and unstable models), `results/decomposition/` (projection / dynamics / density
decomposition per dimension), `results/panels/` (early-time and per-maturity panels),
`results/figures/` (validity and coverage, runtime and total-state Pareto), `results/verify/`,
`results/manifests/` (cases, splits, environment, source SHA and file hashes, weights hashes,
timing environment), `models/` (locked weights and `LOCK.json`). Commands, in order, from the
repository root (`SCC_OUTPUT_ROOT` redirects outputs; jobs longer than a minute run on the
GPU-laptop CPU lane):

    python3 research/sine_closure/scripts/s0_verify.py
    python3 research/sine_closure/scripts/s1_reproduce_gate5.py
    python3 research/sine_closure/scripts/s2_references.py --split train val dev_regression
    python3 research/sine_closure/scripts/s3_diagnose.py
    python3 research/sine_closure/scripts/s4_fit.py            # SCC_PHASE=fit, locks models/
    python3 research/sine_closure/scripts/s2_references.py --split test_interp test_ood
    python3 research/sine_closure/scripts/s5_score.py
    python3 research/sine_closure/scripts/s6_cost.py
    python3 research/sine_closure/scripts/s7_report.py

Each script is idempotent and resumable; every result file carries an identity (numerical-code
hash, basis/sign convention, lift, operator, grid/domain, case hash, split, step, model,
initialiser, metric version) and is reused only when the identity matches; stale or corrupted
files are superseded, never silently accepted.

## Addenda

(none)

### Addendum 1 (2026-11-05, before any scoring; clarification, no change to any rule)

* Time-independent families (C, S) evaluate the coefficient row `d_i = η̃ k̃_i²` once at `T = T_max`
  (`t̃ = 1`), exactly as `dsm.reference.operator_builder` does for Gates 2–5. For the strike-smile
  family the moneyness `x = K/S0 = k̃ K_max e^{rT}/S0` depends on `T` through the forward factor, so
  "time-independent" means "frozen at `T_max`" by that convention; the same row is used by the
  reference solvers, the interior system, the oracles and every reduced model
  (`scc.conventions.AUTONOMOUS_T`). Families D and M evaluate the row at every midpoint time.
* The refinement levels are `Full256@16`, `Full511@16`, `Full1021@16` (sparse action of the same
  midpoint exponential, same step) and `Full256@32`, `Full256@64` (step refinement); the 1021 level
  therefore does not mix step error into the grid comparison.
* The forcing input `J g_R` of the LR closure is identically zero for OP-B (`g ≡ 0`, asserted) and
  is not a fitted quantity; it would appear only in the Dirichlet boundary-sensitivity check.

### Addendum 2 (2026-11-05, after the development diagnostics, before any fitting or test scoring)

Numeric values of every frozen rule, computed from development data only (`train`, `val`,
`dev_regression`; 41 cases; `results/diagnose`, `results/uncertainty`, `results/verify`). Nothing in
§3–§10 changes; this addendum fixes the numbers those rules refer to and adds three rules whose
need became visible in the diagnostics. Test references had not been generated when this was written.

**Uncertainties (per case; dev ranges min / median / max).** `u_price` (interior RMS) 2.6e-6 / 3.3e-6 /
4.6e-6, entirely grid-dominated (`u_step` ≤ 1e-15: the midpoint exponential is exact in time for
autonomous operators and second-order with 1e-13-level step error for the time-dependent ones);
`u_dens` (interior L2) 6.5e-7 / 3.0e-6 / 1.2e-5; negative mass (post-initial max) 1.2e-8 / 4.8e-7 /
6.1e-7; inverse-vol invalid fraction 0 / 0 / 4.9e-4; coverage 0; σ relative error (median) 5.6e-6 /
3.1e-5 / 1.7e-4; raw arbitrage fractions 0 (the reference has no violations at either level, so the
`1/N_interior` floor of §10 is the operative tolerance). Observed grid convergence orders on the eight
cases with the 1021 level: price 2.55–2.81, density 2.15–2.34 (second-order stencil plus payoff kink);
step orders are exactly 2.0 for the time-dependent cases and undefined (roundoff ratios) otherwise.
`u_dyn` (resolved-coordinate uncertainty) is a **same-grid** quantity by construction (step
sensitivity of `c_ref` plus the teacher-forced oracle floor): 5e-9 / 1.8e-8 / 6.4e-8 at m = 9,
1e-9 / 3.6e-9 / 6.4e-8 at m = 17; it is not comparable with `u_price`, and Q1 (resolved dynamics) is
therefore decidable on every case at both m (RRR resolved errors are ≥ 5e-6, ≥ 100 `u_dyn`).

**Roundoff guard.** γ = 1.536e-12 (= 10 × the largest paired |Δ negative mass| over 41 cases × m ∈
{9, 17, 33} × both protocols; the only variant that perturbs the exponential arithmetic is the rotated
coefficient basis, 1.5e-13; sign flips are bitwise exact; `results/verify/roundoff_guard.json`).

**Verification outcome.** All asserted checks of §11 pass at n = 256 (`results/verify/verify.json`);
dense vs sparse solvers 1.3e-13; Black–Scholes vs the 256-point solve 2.03e-5 on `C_sig0.30` (as in
Gate 5). Cross-check against the earlier study on the three Gate 5 cases
(`results/verify/gate5_crosscheck.json`): projection errors identical to 1e-10 relative, autonomous
abscissae to 1e-12, time-dependent abscissae 1e-3 apart by definition (last-step midpoint vs t = 1).
Stopping rule (§10): cond_A (PROJ density fails the 33-mode target) holds on 41/41 dev cases at both
m; cond_B (unclosed dynamics already at the oracle) holds on 0/41; the conjunction is false, so LR
fitting proceeds at m = 9 and m = 17. Family checks (`results/diagnose/family_checks.json`; first train case per family, interior RMS on the common strike region `K ∈ [500, 3000]`, wide domain `K ∈ [250, 6000]` with 587 points): C (C_sig0.180): wide-domain vs base 4.7e-06 (cubic) / 4.7e-06 (PCHIP), interpolation difference 3.3e-08; 512-spline vs 511-restriction price 7.5e-07, density 2.9e-07; S (S2_seed100): wide-domain vs base 2.1e-04 (cubic) / 2.1e-04 (PCHIP), interpolation difference 4.6e-09; 512-spline vs 511-restriction price 5.2e-07, density 5.1e-08; D (D2_seed200): wide-domain vs base 6.1e-03 (cubic) / 6.1e-03 (PCHIP), interpolation difference 7.6e-10; 512-spline vs 511-restriction price 4.2e-07, density 3.3e-08; M (M_seed300): wide-domain vs base 6.3e-05 (cubic) / 6.3e-05 (PCHIP), interpolation difference 5.0e-09; 512-spline vs 511-restriction price 5.5e-07, density 5.2e-08. The spline-versus-restriction difference is ≤ 0.25 `u_price` everywhere, so the nested-grid choice does not move any Gate 5-comparable number beyond uncertainty.

**Rule A — PROJ is a lower bound only in one norm; beating it is compensation.** `V` is orthonormal
and `D2_int V = V Λ` with `D2_int` symmetric, so in the same-grid, full-interior, uniform-weight
Euclidean norm `c_ref = Vᵀψ_ref` minimises both the price error and the `D2` density error over all
coefficient vectors; no m-mode reconstruction can beat PROJ there. In the interior-band, refined-grid
or spectral-density norms the bound is only approximate. Consequently: (i) the 33-mode density target
is unreachable at m = 9 and m = 17 on every development case (PROJ_17 density L2 exceeds RRR33 + u_dens
on 41/41 cases, by 3–20×; PROJ_9 by 30–150×), so density compression is excluded by representation
before any closure is fitted, and the memo says so; (ii) any closed model whose same-grid price
interior RMS is below PROJ_m's by more than `u_price`, or whose refined density L2 is below PROJ_m's by
more than `u_dens`, is flagged **output-oriented compensation** in the tables and is never described
as recovered memory. Negative mass is not bounded by PROJ and is judged by the strict rule alone.

**Rule B — compression regime, fixed from reference-derived quantities.** A (case, protocol) cell is
*eligible* for the compression classes at dimension m iff
`PROJ_m price < RRR33 price + u_price` **and** `RRR_m price − PROJ_m price > 2 u_price` (refined-grid
interior RMS; PROJ and RRR are unfitted). `successful_compression` and `price_only_compression` are
evaluated on the eligible cells only, must hold on every eligible cell, and are reported as
`not_testable_regime_empty` when no test cell is eligible; the verdict names the eligible cases
("price-only compression in the eligible regime: …"). Development counts: m = 17: 2/41 cells
eligible under P0 (`C_sig0.270`, `C_sig0.240`; low σ); m = 9: 0/41 (PROJ_9 is 20–100× above the
target). The headroom `RRR_m − PROJ_m` exceeds `2 u_price` on 41/41 dev cases at m = 9 and on 10/41 at
m = 17 (low-σ and strong-smile cases); on the D family at m = 17 the unclosed rollout is already
*below* the exact projection by ≈ 1e-6 (< u), so a closure cannot show a refined-price gain there.
Expected outcome shape, stated now: useful-closure-at-fixed-m is testable at both m; price-only
compression is testable only in a small low-σ regime at m = 17; "inconclusive by construction" applies
to the price questions on cells whose headroom is below `2 u_price`, never to Q1.

**Rule C — physical-truth caveat (domain truncation).** For family C the analytic Black–Scholes gap of
the semidiscrete reference (`results/verify/bs_vs_u_family_C.json`) is 0.7–5e-6 for σ ≤ 0.27 but
6e-5 (σ = 0.33), 1.5e-4 (0.36), 2.4e-4 (0.38) and 8.3e-4 (0.44), unchanged between 256 and 511
points: it is domain truncation plus the linearity boundary rows, common to every model and reference
on the fixed strike domain. The independent domain refinement gives the same picture per family: the wide-domain solution differs from the base-domain reference by 4.7e-06 (C), 2.1e-04 (S), 6.1e-03 (D) and 6.3e-05 (M) interior RMS (cubic and PCHIP interpolations agree to 1e-8), i.e. at the `u` level for the constant-σ and strike-smile families but 10³× `u` for the two time-dependent families (the earlier study's Gate 2 reported 5.8e-3 for the repository smile). `u_domain` is therefore reported per family as a diagnostic and is NOT folded into `u` (§3 unchanged): folding it in would make every model indistinguishable on the D and M families and would compare models against a quantity none of them is asked to approximate. Verdicts remain statements about the semidiscrete
problem (as in Gate 5); cases with `|Full511 − BS| > 2 u_price` (family C, σ ≥ 0.33 on development
data; the test-ood case σ = 0.55 and test-interp σ = 0.415 will be flagged by the same rule) carry a
`domain_limited_physical` annotation in every table, and no physical-price claim is made on them.

**Early-time panel (P0), for the record.** At t̃ = 0.05 the payoff kink gives interior density L2 of
2e-2 (m = 9), 6e-3 (m = 17), 1e-4 (m = 33) with negative mass 0.2 / 0.07 / 1e-3 for the exact
projection itself; the free rollouts are within a factor 1.5 of PROJ there. Early-time density is
representation-limited at every m ≤ 17 and is reported, not classified.

**Warm start (P1).** The unresolved-initial-condition term `B Φ_D(t,t0) z0` that no deployable model
receives is, at the final time, 0.1 % of the memory term at m = 9 and 1e-9 at m = 17 on `C_sig0.30`
(BLOCK rows); its early-time share is not recorded. W0/WI/WU remain separate rows.

**Provenance.** s3 output: 41 diagnose files, 41 uncertainty files, `family_checks.json`,
`summary.json`; sha256 over the sorted (file, sha256) list = `65b891385393487c (84 files)`; code identity
`5db723e3f2651764` (full) / `d7167e8921f3025c` (reference); laptop: Python 3.11.9, NumPy 2.2.6, SciPy 1.13.1,
Windows, 16 cores.
