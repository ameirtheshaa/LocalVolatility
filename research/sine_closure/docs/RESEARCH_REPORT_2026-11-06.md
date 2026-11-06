# Closure test for small boundary-adapted sine models: implementation, verdict and session record

_LocalVolatility · `research/sine_closure` · work of 2026-11-05 (PR opened 2026-11-06) · branch `claude/research-brief-implementation-8bee67` → [PR #10](https://github.com/ameirtheshaa/LocalVolatility/pull/10) · prepared with Claude Code for Ameir_

---

## 1. Overview & Context

**Where this sits.** The earlier study `research/dupire_signed_memory` (Gates 0–5) built a sine-basis reduced-order model of the scaled Dupire forward equation and ended, at Gate 5, with an *unclosed* 33-mode boundary-adapted sine model as the recommendation. The research brief implemented here (`RESEARCH_BRIEF.md`, transcribed verbatim into the binding `PLAN.md`) asked one follow-up question:

> Does a deployable closure let a 9- or 17-mode boundary-adapted sine model (OP-B operator, detrended price) match or improve the useful accuracy of the 33-mode unclosed sine model, at lower online cost, without losing density validity?

The brief's constraints were explicit: preserve every existing result, never rerun, supersede or overwrite Gates 0–5, implement in a separate research directory and branch, and leave production defaults untouched. It demanded a frozen, pre-registered plan before any held-out trajectory was scored, an exact mathematical contract, hash-recorded case sets with trajectory-level splits and leakage rules, a numerical-uncertainty protocol, a fixed comparison matrix (RRR, scalar-ν controls, stateless and stateful linear closures, oracles, full solvers), a cost protocol and a frozen set of decision classes. A negative result was declared a result in advance.

**What was done.** A complete study package (`scc`, ≈ 2 900 lines in 12 modules, plus ≈ 5 000 lines of pipeline, launcher and diagnostic scripts and a pytest suite of 69 tests) was written, verified, run end to end on the GPU-laptop CPU lane, and reported. Frozen questions Q1–Q4 and S1 were answered under rules fixed in `PLAN.md` §10 and its two dated addenda, with the model lock (`models/LOCK.json`) and the numeric addendum committed *before* any test reference existed.

**Outcome.** Under the frozen rules no closure reaches "useful closure at fixed m" at m = 9 or m = 17. Density compression is excluded at both m by a projection bound that holds before any closure is fitted. The m = 17 price question is inconclusive because the attainable gain (≈ 4e-6) is below twice the numerical uncertainty. At m = 9 only the linear state-space closure with a learned or causal hidden initialiser reaches "resolved-dynamics improvement", which is not a deployment benefit. Scalar-ν controls never help. Adding sine modes beats closing at equal total state. A banded Crank–Nicolson full solver on the production grid is faster than every reduced model, which also changes how Gate 5's speed-up claim should be read. The unclosed 33-mode sine model remains the recommendation.

The branch carries 18 commits and is open for review as PR #10 (no CI configured). Everything outside `research/sine_closure` is byte-identical to `main`.

---

## 2. Theoretical Motivation

### 2.1 The semidiscrete problem (contract fixed in `PLAN.md` §2)

Scaled variables and operator are exactly those of `dsm.operators` with the production constants $S_0 = 1000$, $r = 0.04$, $K_{\min} = 500$, $K_{\max} = 3000$, $T_{\min} = 0.3$, $T_{\max} = 1.5$, 24 output maturities $T \in \mathrm{linspace}(0.3, 1.5, 24)$ and $\tilde t = T/T_{\max} \in [0.2, 1]$:

$$\varphi'(t) = L(t)\varphi(t) + b(t), \qquad L(t) = \mathrm{diag}\big(\tilde\eta(t,\tilde k_i)\,\tilde k_i^2\big)\, D_2, \qquad \tilde\eta = \tfrac{T_{\max}}{2}\,\sigma(T, K/S_0)^2,$$

with $K = \tilde k K_{\max} e^{rT}$ and $\tilde k \in [0.15696, 0.98807]$. **OP-B** is the operator with central interior second differences, zero boundary rows and fixed endpoint values, $b \equiv 0$. The **lift** $\ell$ is the affine interpolant of the two payoff endpoints (constant in time for OP-B); $\psi = \varphi - \ell$ obeys $\psi' = L\psi + g$ with $g = L\ell + b - \ell'$. For OP-B $g$ must vanish, and this is *asserted by a test* ($\|g\| \le 10^{-12}\,\|\mathrm{diag}(d)\|\,\|D_2\|\,\|\ell\|$), never compensated by an intercept.

### 2.2 Basis, state space and blocks

The basis is the **DST-I** sine basis with a deterministic sign, on $n$ uniform points:

$$V[i,j] = \sqrt{\tfrac{2}{n-1}}\,\sin\!\Big(\tfrac{\pi j i}{n-1}\Big),\quad i = 0..n-1,\ j = 1..m,$$

endpoint rows exactly zero, first interior entry of every column positive, no QR. Because the sampled sines are eigenvectors of the interior second difference, $D_{2,\mathrm{int}} V_{\mathrm{int}} = V_{\mathrm{int}}\Lambda_V$ with $\lambda_j = -(4/dk^2)\sin^2\!\big(\pi j/(2(n-1))\big)$, so $V^\top V = I$ to roundoff and the QL regressor $\hat Q = V^\top D_2 V = \Lambda_V$ is diagonal. The Gate 5 basis (QR of the same sampled sines) spans the same subspace; Galerkin rollouts are invariant to the re-basing, which the Gate 5 reproduction checks.

The interior state $\psi_{\mathrm{int}} \in \mathbb{R}^{n-2}$ is split with an **explicit complement**: $V_{\mathrm{int}}$ = DST columns $1..m$, $U_{\mathrm{int}}$ = columns $m+1..n-2$, $[V\ U]$ orthogonal on the interior. With $d_i = \tilde\eta(t,\tilde k_i)\tilde k_i^2$:

$$A = V^\top \mathrm{diag}(d) V \Lambda_V,\quad B = V^\top \mathrm{diag}(d) U \Lambda_U,\quad C = U^\top \mathrm{diag}(d) V \Lambda_V,\quad D = U^\top \mathrm{diag}(d) U \Lambda_U .$$

$A$ is assembled by the structured contraction $(V^\top \mathrm{diag}(d) V)\Lambda_V$ at cost $O(nm^2)$; the dense product is used only in tests. The full-space embedding $Q = I - VV^\top$ of the earlier study is not used for block calculations.

### 2.3 Exact resolved equation and the memory target

With $c = V^\top\psi$ and $z = U^\top\psi$ the resolved coordinates obey exactly

$$c' = Ac + g_R + B\,\Phi_D(t,t_0)\,z_0 + \int_{t_0}^{t} B(t)\,\Phi_D(t,s)\,C(s)\,c(s)\,ds + \int_{t_0}^{t} B(t)\,\Phi_D(t,s)\,g_U(s)\,ds,$$

$\Phi_D$ being the time-ordered propagator of $z' = D(t)z$ ($e^{(t-s)D}$ when autonomous). Every closure approximates this. The operator-based target along a full reference is

$$M(t) = V^\top L(t)(I - VV^\top)\psi_{\mathrm{ref}}(t) = B(t)\,z_{\mathrm{ref}}(t),$$

which contains the unresolved-initial-condition, memory and forcing contributions together and is never estimated by time differencing. The augmented-exponential identity $\exp\!\big(\tau\,[[D, C'],[0, L_{\mathrm{rot}}]]\big)$ gives the autonomous block elimination to machine precision, and the time-ordered propagator (forward order second-order convergent, reversed order failing) is the nonautonomous control; both are verification checks in `s0_verify`.

### 2.4 Closure families and hidden-state protocols

| id | model | fitted |
|---|---|---|
| RRR | $c' = Ac$ | none |
| QL+ / QL± / QL* | $c' = Ac + \nu\Lambda_V c$ with $\nu = \max(\nu^*,0)$ / $\nu^*$ if $\nu^* > \nu_{\mathrm{adm}}/2$ else $\nu_{\mathrm{adm}}/2$ / unclipped $\nu^*$ | $\nu$ |
| LM / LM+b | $\hat M = Hc$ / $Hc + b$ (declared intercept ablation only) | $H$ (, $b$) |
| LR0 / LR | $h' = Fh + G c$, $\hat M = Hc + Rh$, $F = -\mathrm{diag}(\text{rate bank})$; $G_0$ fixed (LR0) or learned by alternating least squares (LR) | $H, R$ (, $G$) |
| WI | hidden initialiser $h(t_0) = K[c(t_0);1]$ for the warm start | $K$ |
| PROJ / BLOCK / TF | exact projected reconstruction $\ell + VV^\top\psi_{\mathrm{ref}}$ / exact block reconstruction with the true $z_0$ / teacher-forced rollout with $M_{\mathrm{ref}}(t)$ | oracles |

$\nu^*$ is the weighted pooled least-squares fit $\sum w\langle M,\Lambda c\rangle / \sum w\|\Lambda c\|^2$ on the training set; $\nu_{\mathrm{adm}} = -\min_{\mathrm{train},t}\lambda_{\min}(V^\top\mathrm{diag}(d(t))V)$ is the admissible bound (the closed matrix $(S+\nu I)\Lambda_V$ has negative real spectrum iff $S + \nu I \succ 0$). LR hidden states for fitting come from exact exponential filtering of the piecewise-linear $c_{\mathrm{ref}}$ (teacher forcing); the true $z$ is never an online input. Two start protocols are scored: **P0** payoff start ($\tilde t = 0 \to 1$, $h(0) = 0$) and **P1** warm start at $\tilde t_0 = 0.2$ with $c(t_0) = V^\top\psi_{\mathrm{ref}}(t_0)$ and hidden-state rows **W0** (zero), **WI** (learned initialiser) and **WU** (causal warm-up: run the closed model from the payoff on $[0,t_0]$, keep $h(t_0)$, reset $c$ to the projection). The discarded $z_0$ is never given to a deployable model. All deployable closures are integrated by one exact exponential of the closed $(m+n_h)$ matrix per step.

No nonlinear, reservoir or neural closure was run; the plan made escalation conditional on a robust held-out LR gain over LM, which did not occur (§5).

### 2.5 Three rules added by Addendum 2 (before fitting or test scoring)

* **Rule A (projection bound ⇒ compensation flag).** $V$ is orthonormal and $D_{2,\mathrm{int}}$ is symmetric with $V, U$ as eigenvectors, so in the same-grid, full-interior Euclidean norm $c_{\mathrm{ref}} = V^\top\psi_{\mathrm{ref}}$ minimises both the price error and the $D_2$-density error over all coefficient vectors; no $m$-mode reconstruction can beat PROJ there. Any closed model that beats PROJ$_m$ by more than $u$ in the same-grid price or in the refined density is flagged *output-oriented compensation* and never described as recovered memory.
* **Rule B (compression regime).** A (case, protocol) cell is eligible for the compression classes at dimension $m$ iff $\mathrm{PROJ}_m\ \text{price} < \mathrm{RRR33}\ \text{price} + u_{\mathrm{price}}$ and $\mathrm{RRR}_m - \mathrm{PROJ}_m > 2u_{\mathrm{price}}$; an empty regime is reported as `not_testable_regime_empty`.
* **Rule C (domain truncation).** Verdicts are statements about the semidiscrete OP-B problem on the production strike domain; family-C cases whose Black–Scholes gap exceeds $2u_{\mathrm{price}}$ ($\sigma \ge 0.33$) carry a `domain_limited_physical` annotation and no physical-price claim.

---

## 3. Implementation

### 3.1 Package `scc` (binding contract in `docs/API_CONTRACT.md`)

| module | role |
|---|---|
| `conventions.py` | DST-I basis and complement, sine eigenvalues, lift/detrend, `InteriorSystem` (coefficient row $d(t)$, cached at `AUTONOMOUS_T = 1.0` for time-independent families), blocks, structured $A$, forcing $g$, memory target, `expm_h`, `phi1_step` |
| `cases.py` | deterministic case generation for families C, S, D, M (+ the three Gate 5 regression cases), splits, σ-clip disclosure, sha256 manifest |
| `identity.py` | code identity (`full` and narrow `reference` scopes), identity-hashed npz cache, atomic `replace_retry`, `RunManifest`, Windows-safe liveness probe, `SCC_PHASE=fit` leakage guard |
| `reference.py` | nested grids 256/511/1021, dense `expm` midpoint solver (delegates to `dsm.reference.solve_full` variant "B"), sparse `expm_multiply` solver, banded Crank–Nicolson with Rannacher start, cached `get_reference` |
| `targets.py` | $c, z, M$ along a reference, trajectory-balanced weights, $d$ extremes |
| `closures.py` | `ModelSpec`, closed matrix, rollout with blow-up flag, `fit_nu`, `fit_LM`, rate banks, exact exponential filtering, `fit_LR` (ALS), `fit_initializer`, stability reports (abscissae, perturbation growth, propagator growth, long-horizon stress) |
| `oracles.py` | PROJ, TF, autonomous and time-ordered block identities, $B = 0$ control, projection/dynamics decomposition |
| `metrics.py` | price, resolved error, density (repository $D_2$ primary; zero-row $D_2$, fourth-order stencil and spectral as diagnostics), negative mass per maturity, raw and repository arbitrage fractions, inverse-vol with threshold panel |
| `cost.py` | thread pinning, timing environment capture, `time_callable` (median/IQR/min/cold) |
| `modelio.py` | weight files, `LOCK.json` with sha256 per file, locked loading |
| `evaluate.py` | reference levels, `EvalBundle`, trajectory scoring, uncertainty, oracle/block/rollout rows under P0 and P1-W0/WI/WU |

### 3.2 Pipeline (all idempotent, identity-checked, resumable)

1. **`s0_verify`** — asserted checks a–j: basis orthonormality and eigen-relation, lift residual, structured vs dense assembly, reconstruction identity on every reference row, autonomous block elimination, time-ordered propagator (forward converges, reversed fails), $B = 0$ control, dense vs sparse solvers, dense vs analytic Black–Scholes on family C, roundoff guard γ (five basis/summation variants including a rotated coefficient basis). Writes `results/verify/*`.
2. **`s1_reproduce_gate5`** — runs the *unchanged* original Gate 5 script into the separate root `results/gate5_reproduction/` and compares every non-timing metric with the committed values.
3. **`s2_references --split train val dev_regression`** — generates `Full256@16`, `Full511@16`, `Full256@32` (and `Full1021@16`, `Full256@64` where required) into the identity-hashed cache.
4. **`s3_diagnose`** — per development case: PROJ/RRR/TF/BLOCK rows at $m \in \{9,13,17,25,33,65\}$, uncertainty files, projection/dynamics decomposition, family checks (wide-domain and spline-vs-restriction), the representation-limited stopping rule, and the roundoff-guard value. Its output hash was recorded in Addendum 2.
5. **`s4_fit`** (`SCC_PHASE=fit`) — fits QL (three clip rules), LM, LM+b, LR0, LR over the frozen hyperparameter grid; selects by the geometric mean of the free-rollout resolved error over validation cases × {P0, P1-W0}; writes validation tables, `selection.json`, the QL sensitivity panel, offline costs, 18 weight files, `models/LOCK.json` and `LOCK_NOTES.json` (budget completeness, degenerate and edge selections, selection margins, leakage evidence, pre-registered expectations).
6. **`s2_references --split test_interp test_ood`** — only after the lock commit.
7. **`s5_score`** — 68 cases × 80 rows each (RRR at six dimensions, PROJ, TF, BLOCK, FULL, locked closures under P0 and P1-W0/WI/WU) plus stability diagnostics and per-maturity arrays.
8. **`s6_cost`** — exclusive run on the laptop, pinned threads, 7 repeats after 2 warm-ups, randomised model order, one held-out case per family, full-solver candidates (dense, sparse, banded CN).
9. **`s7_report`** — tables, `decision.json`/`decision.md` implementing §10 and Rules A/B/C, decomposition, panels, figures, and the AUTOGEN blocks of `docs/MEMO.md`.

Diagnostic extras: `x_gate5_crosscheck` (projection errors and abscissae against the earlier study), `x_ic_term` (share of the unresolved-initial-condition term in $M$ after $t_0$), `x_ql_pertraj` (per-trajectory scalar fits), `x_cell_counts` (per-cell condition counts behind the classes). `_laptop_job.py` is the launcher used on the GPU laptop; `_migrate_cache_identity.py` re-keys cache entries across identity-scheme changes with a roundoff-tolerant staleness guard; `run_all.py --resume` chains the stages.

### 3.3 Provenance and leakage controls

Every result file carries an identity (code hash, basis `dst1_v1`, lift `affine_endpoints_v1`, operator `OP-B`, grid, step, case hash, split, model, initialiser, schema) and is reused only when it matches. The cache refuses to open any `test_*` reference while `SCC_PHASE=fit`; `LOCK_NOTES.json` records that at lock time the reference index held only 28 train, 10 val and 3 dev-regression references (index sha256 `a6b6f0be…`). The plan was frozen at source commit `c028ae45` (the PR #9 merge); Addendum 2 records the s3 output hash `65b891385393487c` (84 files) and the code identities `5db723e3f2651764` (full) / `d7167e8921f3025c` (reference). The earlier study is imported read-only as `dsm`; `git diff main -- research/dupire_signed_memory` is empty.

### 3.4 Compute

Jobs longer than a minute ran on the GPU-laptop CPU lane (Windows 10 build 26200, 16 cores, Python 3.11.9, NumPy 2.2.6 with OpenBLAS 0.3.29, SciPy 1.13.1) through `scripts/_laptop_job.py`, with the Dropbox-synced worktree as the working copy and `SCC_CACHE_ROOT=C:/mcp_jobs/scc_cache`; results synced back through Dropbox. The Mac ran only tests, quick smoke modes (`--quick`, written to `results/*_quick*` and git-ignored) and the report stage. The session execution log (jobs, failures, fixes) is in Appendix A.

---

## 4. Parameters & Design Choices

| parameter | value | rationale |
|---|---|---|
| model grid $n$ | 256 | the production grid; frozen in the plan |
| refinement grids | 511, 1021 (nested: every second / fourth point is a 256-grid point; coincidence verified to 0.0) | refined references restricted by index, no interpolation in primary scores (deviates from Gate 5's 512 + cubic spline; difference reported ≤ 0.25 $u_{\mathrm{price}}$) |
| time step | $h = 0.8/(23\cdot\mathrm{sub})$, `sub` = 16 primary ($h = 2.1739\times10^{-3}$, 460 steps), 32 and 64 for step refinement | 24 maturities and $\tilde t_0 = 0.2$ are grid points |
| `AUTONOMOUS_T` | 1.0 (coefficient row of time-independent families evaluated at $T = T_{\max}$) | matches `dsm.reference.operator_builder` for Gates 2–5 (Addendum 1); the S family depends on $T$ through the forward factor |
| reference solver | dense `expm` midpoint stepping (second-order Magnus) for 256/511; sparse `expm_multiply` (tol 1e-13) for 1021 and sub 64 | dense vs sparse agreement required to 1e-11 before any sparse result is used |
| uncertainty | $u_q = \max(u_q^{\mathrm{grid}}, u_q^{\mathrm{step}})$ per quantity and case; $u_{\mathrm{dyn}} = \max(\text{interior RMS of } V(c_{\mathrm{ref}@16} - c_{\mathrm{ref}@32}),\ e_{\mathrm{TF}})$ | never reduced by observed convergence order |
| roundoff guard γ | $1.536\times10^{-12}$ (10 × largest paired \|Δ negative mass\| between the base RRR rollout and four roundoff-different variants — sign-flipped basis, reversed summation, alternating column signs, rotated coefficient basis — over 41 cases × 3 m × 2 protocols) | strict negative-mass rule without any percentage relaxation |
| decision multipliers | beyond uncertainty: $|\Delta q| > 2u$; noninferior: $q_X \le q_Y + u$; fractions: $\max(u, 1/N_{\mathrm{interior}})$ with $N_{\mathrm{interior}} = 230 \times 24$ | frozen in §10 |
| case families | C constant σ; S strike smile $\mathrm{clip}(b(1+s(1-x)+c(x-1)^2), 0.05, 2)$; D repository-type $base + ye^{-y}$, $y = \sqrt{x+x_s}(T+t_s)$; M temporally modulated smile | ranges and seeds in `PLAN.md` §4; parameters from `default_rng(seed)` in column order, rounded to 6 decimals |
| splits | train 28, val 10, test-interp 17, test-ood 10, dev-regression 3 (the Gate 5 cases) | whole trajectories are the split unit; manifest sha256 `51e0626f…` |
| closed dimensions | $m \in \{9, 17\}$; S1 at 33 (LM, QL only) | unclosed RRR at $m \in \{9,13,17,25,33,65\}$ so that equal-total-state comparisons exist (9+8, 9+16, 17+8, 17+16) |
| hidden sizes | $n_h \in \{8, 16\}$ | both grid values are edges; selected $n_h = 16$ at m = 9 and 8 at m = 17 |
| ridge | $\lambda \in \{10^{-10}, 10^{-8}, 10^{-6}, 10^{-4}, 10^{-2}\} \times \mathrm{tr}(\mathrm{Gram})/\dim$ | same relative grid for $H, R, G, K$ |
| rate banks | "N" $[0.5r_1, 16 r_1 d_{\max}/d_{\min}]$, "W" $[0.25 r_1, 64 r_1 d_{\max}/d_{\min}]$, log-spaced; $r_1 = d_{\min}|\lambda_{m+1}|$ | recorded constants: $d_{\min} = 6.238\times10^{-4}$, $d_{\max} = 0.3731$, ratio 598.2; $r_1 = 0.890$ (m 9), 2.876 (m 17), 10.15 (m 33) |
| ALS | ≤ 30 iterations, stop at relative decrease < 1e-6, $G_0$ standard normal/$\sqrt m$ seed 0 | — |
| training weights | $w_{\mathrm{case}} = 1/(N_{\mathrm{rows}}\max(\mathrm{RMS}_{\mathrm{case}}(M), 0.1\cdot\mathrm{median}\,\mathrm{RMS}(M))^2)$, 461 rows per trajectory from $\tilde t = 0$ | neither long nor high-amplitude paths dominate |
| selection | min geometric mean $e_{\mathrm{dyn}}$ over val × {P0, P1-W0}; instability disqualifies; ties → smaller $n_h$, then larger λ | selected: m 9 LM λ 1e-6; LR bank N λ 1e-4 $n_h$ 16; LR0 bank W λ 1e-4 $n_h$ 16; m 17 LM λ 1e-6; LR/LR0 bank W λ 1e-6 $n_h$ 8; m 33 LM λ 1e-10 (edge) |
| scalar ν | $\nu^* = +3.06\times10^{-5}$ (m 9, $\nu_{\mathrm{adm}} = -1.35\times10^{-3}$), $-3.59\times10^{-5}$ (m 17, $\nu_{\mathrm{adm}} = -9.68\times10^{-4}$; QL+ clips to 0 and equals RRR), $+1.27\times10^{-6}$ (m 33) | the Fourier-era clip 5e-4 was not transplanted |
| metrics | interior band = repository `interior_k_mask` (5–95 % strike quantiles per maturity); ATM $|K/S_0 - 1| \le 0.10$; wings $\le 0.7$ or $\ge 1.5$; density = Breeden–Litzenberger with the repository $D_2$; raw arbitrage guard $10^{-12}\times$ row max (plus the repository $10^{-3}$ version); inverse-vol identifiable mask at `rel_tol` 1e-3 with panel {1e-2, 1e-3, 1e-4}; instability = non-finite or $\|c\| > 10\times$ reference maximum | definitions frozen in §8; no clipping, smoothing or renormalisation anywhere |
| cost protocol | 7 repeats, 2 untimed warm-ups, cold = first call, threads pinned to 1, model order randomised per repeat, exclusive run; CN substeps = smallest power of two reaching $u_{\mathrm{price}}$ (= 1 on every timed case), Rannacher 2 | a difference counts only when medians differ by more than the larger IQR |
| wide-domain check | $K \in [250, 6000]$, $n = 587$, cubic and PCHIP interpolation back to the base grid | diagnostic only, not folded into $u$ |

---

## 5. Results

### 5.1 Verification before scoring (`results/verify`)

All asserted checks pass at $n = 256$, `sub` = 16 (`verify.json`, wall 443 s on the laptop). Dense and sparse solvers agree to 1.3e-13 (`C_sig0.30`) and 2.3e-13 (`D2_seed200`) in max-abs, 2.5e-13 and 4.3e-13 relative to $\max|\varphi|$ (tolerance 1e-11); the 256-point solve differs from analytic Black–Scholes by 2.03e-5 interior RMS on `C_sig0.30`, as in Gate 5; the banded CN candidate's observed orders are 0.95–0.98 for the first substep doubling and 2.0–2.4 thereafter (informational). The roundoff guard is γ = 1.536e-12; the only variant that perturbs the exponential arithmetic is the rotated coefficient basis (1.5e-13), sign flips are bitwise exact.

**Gate 5 regression.** The unchanged original script reproduces 975 committed metrics with 0 failures; the maximum relative deviation is 5.04e-7. The cross-check of the new code against the earlier study on the three Gate 5 cases gives projection errors identical to 1e-10 relative and autonomous abscissae to 1e-12; time-dependent abscissae differ by 1e-3 by definition (Gate 5 uses the last-step midpoint operator, this study the $t = 1$ operator).

### 5.2 Development diagnostics (41 cases, before fitting)

* **Stopping rule.** cond_A (PROJ density misses the 33-mode target) holds on 41/41 cases at both m; cond_B (unclosed dynamics already at the oracle) on 0/41. The conjunction is false, so LR fitting proceeded at both m; neither m is "representation-limited" by the frozen conjunction. But cond_A alone already settles density compression: PROJ$_{17}$ exceeds RRR33 + $u_{\mathrm{dens}}$ by 3–20× and PROJ$_9$ by 30–150× on every case.
* **Compression eligibility (Rule B).** m = 17: 2/41 development cells (`C_sig0.270`, `C_sig0.240`, low σ); m = 9: 0/41. The headroom $\mathrm{RRR}_m - \mathrm{PROJ}_m$ exceeds $2u_{\mathrm{price}}$ on 41/41 cases at m = 9 but only 10/41 at m = 17; on the D family at m = 17 the unclosed rollout is already ≈ 1e-6 *below* the exact projection, so no refined-price gain is possible there.
* **Uncertainties.** $u_{\mathrm{price}}$ 2.6e-6 / 3.3e-6 / 4.6e-6 (min / median / max), entirely grid-dominated ($u^{\mathrm{step}} \le 10^{-15}$ for autonomous operators, 1e-13-level for time-dependent ones); $u_{\mathrm{dens}}$ 6.5e-7 / 3.0e-6 / 1.2e-5; negative mass 1.2e-8 / 4.8e-7 / 6.1e-7; raw arbitrage fractions 0 at both reference levels. Observed grid convergence orders: price 2.55–2.81, density 2.15–2.34. $u_{\mathrm{dyn}}$ 5e-9 / 1.8e-8 / 6.4e-8 (m 9), 1e-9 / 3.6e-9 / 6.4e-8 (m 17), so Q1 is decidable everywhere (RRR resolved errors ≥ 5e-6).
* **Domain truncation (Rule C).** Black–Scholes gap of the reference on family C: 7.4e-7 (σ 0.18), 6.9e-7 (0.22), 5.0e-6 (0.27), 5.9e-5 (0.33), 1.5e-4 (0.36), 2.4e-4 (0.38), 8.3e-4 (0.44), unchanged between 256 and 511 points. Wide-domain refinement moves the base reference by 4.7e-6 (C), 2.1e-4 (S), 6.1e-3 (D), 6.3e-5 (M) interior RMS; cubic and PCHIP agree to 1e-8. Spline-vs-restriction differences are ≤ 7.5e-7 in price, i.e. ≤ 0.25 $u_{\mathrm{price}}$.
* **Projection/dynamics decomposition** (`results/decomposition`). At m = 9 the projection part dominates the price error (e.g. `C_sig0.180`, P0: 5.29e-4 projection vs 2.50e-4 dynamics); at m = 17 the two are comparable (3.27e-5 vs 4.29e-5), and from m = 25 the dynamics part exceeds the projection part (8.7e-7 vs 6.8e-6 at m 25). The parts are orthogonal only in the full-grid Euclidean norm (orthogonality defect ≤ 3e-7 for m ≤ 33, up to 3e-2 at m = 65 where both parts are at roundoff).
* **Early-time panel (P0).** At $\tilde t = 0.05$ the payoff kink gives interior density L2 of 2e-2 (m 9), 6e-3 (m 17), 1e-4 (m 33) with negative mass 0.2 / 0.07 / 1e-3 for the exact projection itself (full solve: 1e-13); free rollouts are within a factor 1.5 of PROJ. Early-time density is representation-limited at every m ≤ 17 and is reported, not classified.
* **Warm-start initial-condition term** (`x_ic_term`, four development cases). Just after $t_0$ the term $B\Phi_D(t,t_0)z_0$ that no deployable model receives is 47–91 % of $M$; integrated over $[t_0, 1]$ it is 1.4–13.8 % (m 9) and 0.07–0.96 % (m 17). A teacher-forced rollout that receives only $M$ minus this term has resolved error 3.4e-6–1.3e-5 at m 9 (full-$M$ oracle 2.6e-9–1.8e-8; RRR 5e-5–1.4e-4), i.e. even a perfect memory model started from a zero hidden state keeps a floor only 4–40× below RRR and 10³× above the full-$M$ oracle, which is why W0 is a separate row and WI/WU exist.

### 5.3 Fitting and selection (validation data only)

Validation geometric-mean resolved error relative to RRR$_m$ (over val cases × {P0, P1-W0}): m 9 — LM 0.55, LM+b 0.56, LR 0.60, LR0 0.63, QL 1.36; m 17 — LM 0.52, LM+b 0.52, LR0 0.75, LR 0.88, QL± 12.7 (QL+ ≡ RRR); m 33 — LM 0.55. The stateful models beat LM under the payoff start but lose under the warm start with zero hidden state, which the selection criterion averages over; with the learned initialiser (test set, below) the order reverses.

**Scalar ν is misspecified, not mis-clipped.** The validation sensitivity panel $\nu \in \{0, \tfrac14, \tfrac12, 1, 2\}\nu^*$ and $-\nu^*$ shows $\nu = 0$ (RRR) as the best point at every m; both signs of ν raise the resolved error (m 9: 1.20e-4 at 0 → 1.63e-4 at $\nu^*$, 1.45e-4 at $-\nu^*$; m 17: 8.48e-6 → 1.08e-4 at $\nu^*$, 1.08e-4 at $-\nu^*$). Per-trajectory fits are sign-consistent within each m (24/28 positive at m 9, 28/28 negative at m 17, 26/28 positive at m 33) and the sign flips between m.

Selection margins (`LOCK_NOTES.json`): LM/LM+b are clear at every m (log-gap to the runner-up λ 0.14 at m 9, 0.18 at m 17, 0.80 at m 33); LR/LR0 are flagged "arbitrary" at both m because the two rate banks differ by only 0.2–2.4 % (log-gap 0.002–0.024). Degenerate entry: QL+ at m 17 equals RRR because $\nu^* < 0$. Edge selections: $n_h$ at both m (the grid has only edges), λ = 1e-10 for LM-33.

**Pre-registered expectations vs outcome** (`LOCK_NOTES.json`, written after the lock commit and before any test reference existed):

| expectation logged before test scoring | held-out outcome |
|---|---|
| QL: no benefit at any m | confirmed |
| density: no compression at either m (Rule A) | confirmed |
| m 9: LM *may* qualify as useful closure at fixed m (resolved and price gains beyond 2u on all cells) | **not reached** — resolved gain on 33/34 cells, price on 29/34, and validity (negative mass, coverage, arbitrage fractions) is not noninferior; class *inconclusive* |
| m 17: LM may qualify on cells with headroom; price-only compression possible only on the 2/41 low-σ cells and marginal there; eligible test set may be empty | **not reached**; eligible test set empty (0/34); RDI fails on 2/34 cells |
| stateful advantage by the frozen rule not expected; P0-only advantage will be reported as such | confirmed |
| m 33 secondary: expected inconclusive | confirmed |

> _Note: the pre-lock expectation that the stateless map could reach "useful closure at fixed m" was superseded by the held-out validity results; the final diagnosis is that closures import the exact projection's non-validity while improving the resolved coordinates (§5.4)._

### 5.4 Held-out verdict (17 interpolation cases × {P0, P1} = 34 cells per entry)

**Classification (frozen §10, `results/decision.json`).**

| entry | primary class |
|---|---|
| QL+/QL±/QL* m 9 and m 17 | no_supported_benefit |
| LM, LM+b m 9 | inconclusive |
| LR0, LR m 9 W0 | no_supported_benefit |
| LR0, LR m 9 WI and WU | **resolved_dynamics_improvement** (34/34 cells) — not a deployment benefit by itself |
| LM, LM+b, LR0, LR m 17 (all hidden protocols) | inconclusive |
| compression, both m | not_testable_regime_empty (0/34 eligible cells) |
| S1 (m 33 density gain beyond 2$u_{\mathrm{dens}}$) | no for QL+, QL±, QL*, LM |

Output-oriented compensation is flagged for every closed entry at both m. The two bound assertions behind the flags hold with 0 violations: no closed model beats the exact projection in the same-grid full-domain price norm (2992 cells) or in the same-grid full-domain $D_2$ density norm (3808 cells, including every compensation-flagged cell). Every density improvement over PROJ is therefore a refined-truth, interior-window effect, of two kinds: *trade-type* (price worse than the unclosed model, density better than the projection: the positive-ν controls at m 9) and *norm-effect* (price and density both slightly better, e.g. LM-17 on `D2_seed220`).

**Per-cell counts, test_interp** (closed model vs RRR at the same m; `results/tables/cell_counts.md`):

| entry | resolved > 2u | price > 2u | density nonin. | neg. mass strict | invvol nonin. | coverage nonin. | GM resolved / RRR | GM price / RRR | GM density / RRR |
|---|---|---|---|---|---|---|---|---|---|
| QL (any clip), m 9 | 0 | 0 | 30 | 34 | 34 | 34 | 1.40 | 1.193 | 0.998 |
| LM m 9 | 33 | 29 | 19 | 14 | 8 | 8 | 0.47 | 0.869 | 0.999 |
| LR m 9 h16 / W0 | 22 | 21 | 13 | 23 | 2 | 2 | 0.60 | 0.925 | 1.029 |
| LR m 9 h16 / WI | 34 | 33 | 24 | 9 | 0 | 0 | 0.36 | 0.840 | 0.983 |
| QL+ m 17 (≡ RRR) | 0 | 0 | 34 | 34 | 34 | 34 | 1.00 | 1.000 | 1.000 |
| QL± / QL* m 17 | 0 | 0 | 0 | 4 | 17 | 15 | 18.08 | 12.972 | 1.185 |
| LM m 17 | 32 | 0 | 6 | 16 | 34 | 1 | 0.44 | 1.048 | 1.137 |
| LR m 17 h8 / W0 | 21 | 2 | 7 | 9 | 30 | 1 | 0.91 | 1.360 | 1.421 |
| LR m 17 h8 / WI | 32 | 2 | 9 | 15 | 34 | 2 | 0.35 | 1.028 | 1.111 |
| LM m 33 | 32 | 0 | 6 | 8 | 25 | 9 | 0.32 | 0.996 | 1.279 |
| QL* m 33 | 0 | 0 | 34 | 34 | 34 | 34 | 15.09 | 0.654 | 0.988 |

Geometric means here are over both protocols. The memo's payoff-start-only figures are 0.84–0.87× price at m 9 and 0.90–0.91× at m 17; under the warm start LM-17 and LR-17 are *worse* than RRR-17 in price (7.54e-6 and 7.55e-6 vs 6.24e-6) and density (1.37e-4 vs 1.11e-4), which is why the both-protocol ratios exceed 1. The two m = 17 resolved-dynamics misses (`C_sig0.310`, `M_seed322`, warm start) are cells where the unclosed model is already accurate (2.3e-6 and 6.8e-7 against 5e-6–4.3e-5 on development data): a closure fitted to an average memory term adds error where the true term is small.

**Headline numbers (test_interp, P0, geometric means over 17 cases; `summary_by_split.md`).**

| model | price int. RMS (refined) | density L2 int. | neg. mass max post-init. | coverage |
|---|---|---|---|---|
| FULL256 (same-grid truth vs 511) | 3.22e-06 | 2.38e-06 | 1.59e-07 | 1.000 |
| PROJ m 9 / m 17 | 1.25e-04 / 7.35e-06 | 8.78e-04 / 1.25e-04 | 1.28e-02 / 3.85e-05 | 0.952 / 0.993 |
| RRR m 9 | 1.61e-04 | 9.06e-04 | 1.25e-03 | 0.972 |
| LM m 9 | 1.37e-04 | 9.09e-04 | 1.30e-03 | 0.954 |
| LR m 9 h16 | 1.31e-04 | 8.93e-04 | 9.25e-03 | 0.955 |
| RRR m 13 | 2.57e-05 | 2.76e-04 | 5.38e-04 | 0.991 |
| RRR m 17 | 9.40e-06 | 1.36e-04 | 1.79e-05 | 0.997 |
| LM m 17 | 8.54e-06 | 1.43e-04 | 1.93e-05 | 0.991 |
| LR m 17 h8 | 8.21e-06 | 1.38e-04 | 1.98e-05 | 0.991 |
| QL± m 17 | 1.20e-04 | 1.62e-04 | 1.84e-05 | 0.996 |
| RRR m 25 | 4.05e-06 | 4.21e-05 | 1.58e-06 | 0.999 |
| RRR m 33 | 3.33e-06 | 1.81e-05 | 1.90e-07 | 1.000 |

The m = 17 price gains (≈ 4e-6 at best) sit below $2u_{\mathrm{price}}$ (5.2–8.6e-6 over the interpolation cases; LM 0/34 and LR 2/34 cells beyond it): *inconclusive*, as logged before scoring. On the high-σ cells price does not discriminate 17 from 33 modes at all (even RRR-17 → RRR-33 is within $2u_{\mathrm{price}}$); RRR-25 is within $u$ of RRR-33 in price on 17/17 cases.

**Adding modes beats closing.** RRR-25 (17 + 8 states) has 2.0× lower price error, 3.3× lower density error and 12× lower negative mass than the 25-state closed model LR-17/h8; RRR-13 beats every closed 9-mode model by ~5× in price and ~3× in density. The matched stateful-advantage test (LR vs LM at the same m *and* vs RRR at $m + n_h$, both protocols, every cell) fails, so the m = 9 resolved-dynamics class is not evidence for memory.

**Validity.** The strict negative-mass rule fails on 20–25 of 34 cells at m 9 and 15–19 at m 17. Where the extra negative mass sits (per-maturity arrays in `results/score`): at m 9 the closures *lower* it at the first two maturities, where the unclosed model's payoff ringing dominates (2.8e-2 → 2.3e-2 at T = 0.30), and *raise* it from the fourth maturity on by 2e-4–1e-3 absolute, where the unclosed model is nearly valid; at m 17 the excess is at the first maturities (+4e-4 on 1.2e-3 at T = 0.30), consistent with transient growth of the non-normal coupled system after initialisation. Raw arbitrage-violation fractions rise by a median of 2–3 interior grid points per maturity at m 9 (max 12) and 0.5 at m 17 (max 10); inverse-vol coverage drops by a median of 4 points (m 9) and 1 point (m 17). LR-9 raises the geometric-mean negative mass about 10× because it moves the coefficients towards the exact projection, whose own negative mass is 13× the unclosed model's.

**Stability and out-of-distribution stress (10 cases).** No blow-ups anywhere; every closed matrix keeps a negative spectral abscissa (≤ −0.028). The coupled stateful systems are non-normal: numerical abscissa up to +3.9, propagator growth up to 4.1, long-horizon stress ratio ≤ 1.17 (bounded). Out of distribution the stateless map loses its gain entirely (LM-9 resolved ratio 1.16 vs RRR, price 1.75e-4 vs 1.65e-4; LM-17 worst-case negative mass 0.44 vs 0.12), as expected for a constant $H$ shared across volatility levels; the stateful closure keeps a resolved-error gain (0.65× at m 9, 0.70× at m 17 over both protocols) and beats LM in resolved error on 20/20 cells at m 9 (14/20 at m 17) with price noninferior on 20/20, but density on 17/20 (3/20) and strict negative mass on 13/20 (12/20). This is a stress observation, not a verdict.

**Extrapolation disclosure.** 10 (case, parameter) pairs of the 17 interpolation cases lie outside the span realised by the 28 training draws although inside the pre-registered ranges (among them the worst held-out case `M_seed322`, modulation frequency 0.78 against a realised span 0.85–1.50); labels are kept as frozen. Development and held-out behaviour agree: validation geometric means of the locked models are reproduced on the test set within 10 %.

### 5.5 Cost (`results/cost/cost.json`, GPU laptop, one held-out case per family)

| solver / model | online warm median [ms] | note |
|---|---|---|
| dense `expm` reference (Gate 5's "un-reduced solve") | 6 673 | |
| sparse `expm_multiply` | 1 910 | |
| **banded Crank–Nicolson, 1 substep, Rannacher 2** | **20.9** | converged to $u_{\mathrm{price}}$ on every timed case; density 2.4e-7 / 4.6e-8 vs dense, negative mass and butterfly identical to the reference's (`cn_solver_density_check.json`) |
| RRR 9 / 13 / 17 / 25 / 33 / 65 (P0) | 26.1 / 28.2 / 32.9 / 36.4 / 47.6 / 107 | |
| LM 9 / 17 (P0) | 27.8 / 31.7 | nothing measurable over RRR at the same m |
| LR0, LR 9/h16 and 17/h8 (P0) | 50.7–52.4 | as much as RRR-33 |
| same with hidden initialiser (P1-WI) | 38.7–43.2 | |

IQRs are 7–29 ms, so RRR-9 and the banded full solver are within variability of each other and RRR-33 is slower than the banded solver by about one IQR. Consequences: (i) no closed model has an online-cost advantage over the unclosed model of equal total state, so even a positive accuracy result would not have been a compression; (ii) Gate 5's "5–17× faster than the un-reduced solve" referred to the dense reference solver; against a banded production-style solver of equivalent accuracy the sine models give no speed-up at this problem size. Break-even counts (3e3–4e6 queries) are reported but moot. Offline cost per closed model is 119 s (LM, QL) to 136 s (LR), dominated by generating the 28 training references (119 s); the fits themselves take milliseconds (LM) to about a second per candidate (LR). Gate 5's report is not rewritten; a future addendum would carry the annotation.

### 5.6 Negative results, stated explicitly

* Scalar ν (all three clip rules), m 9, 17, 33: no benefit; the pooled fit is in the wrong direction on validation at every m, and the per-trajectory sign flips with m. At m 33 QL lowers the refined price error (0.65×) while raising the resolved error 15×: an output effect, not dynamics.
* Stateless linear map LM and its intercept ablation LM+b: resolved gain on 32–33/34 cells at both m, but validity worse (negative mass, arbitrage fractions, coverage) and at m 17 price/density worse under the warm start; class *inconclusive*. The intercept changes nothing (the forcing is identically zero and asserted so).
* Stateful closures with zero hidden state (W0): 2–5× worse resolved error than with WI/WU; no supported benefit.
* Stateful advantage (frozen rule): fails. Under the payoff start and out of distribution LR beats LM in resolved error, at the price of non-normal transient growth and higher negative mass.
* Compression (price-only or full): not testable, eligible regime empty at both m; density compression excluded by the projection bound.
* S1 (33-mode density): inconclusive; LM-33 raises the interior density error by 26–28 % on average.
* Reduced models vs a banded full solver: no speed-up.

### 5.7 Figures and panels

`results/figures/`: `density_vs_state.png`, `early_panel.png`, `negative_mass.png`, `pareto_runtime.png`, `pareto_state.png`, `per_maturity.png`, `validity_coverage.png`. `results/panels/early_time.md` (P0 at $\tilde t$ = 0.05, 0.10, 0.15) and `maturity.md` (per-maturity price RMS). One reporting caveat is recorded in `results/tables/WARNINGS.md`: the report script's case discovery lists 12 (case, quantity) pairs with missing $u$ that are the manifest/summary files, not cases; they are treated as 0 and do not touch any classification.

---

## 6. Files & Artifacts

All paths under `research/sine_closure/` unless stated; branch `claude/research-brief-implementation-8bee67`, commits `421c4cd` … `0bd6f44` (18), PR https://github.com/ameirtheshaa/LocalVolatility/pull/10 (open, ready for review, no CI).

* `PLAN.md` — frozen plan v1 (§0–§12) + Addendum 1 (conventions) + Addendum 2 (numeric rules, Rules A/B/C, provenance)
* `docs/MEMO.md` — decision memo (verdict narrative, AUTOGEN tables, uncertainty note, "what this does not show", cost reading)
* `docs/API_CONTRACT.md`, `docs/RESULT_SCHEMA.md` — binding module and file contracts (model keys `<closure>_m<m>[_h<n_h>]__P0|P1[_W0|WI|WU]`)
* `README.md` — reading order, layout, run commands, outcome
* `scc/` — package (12 modules); `tests/` — pytest suite (8 files); `scripts/` — `s0_verify.py` … `s7_report.py`, `run_all.py`, `_laptop_job.py`, `_migrate_cache_identity.py`, `x_gate5_crosscheck.py`, `x_ic_term.py`, `x_ql_pertraj.py`, `x_cell_counts.py`, `smoke_eval.py`, `_fake_results.py`
* `results/verify/` — `verify.json`, `roundoff_guard.json` (γ), `solver_check.json`, `gate5_crosscheck.json`, `bs_vs_u_family_C.json`, `ic_term_warm_start.json`, `cn_solver_density_check.json`
* `results/gate5_reproduction/` — unchanged Gate 5 script output, checkpoints, `comparison.json` (975 metrics, 0 failed)
* `results/diagnose/` — 41 development diagnostics, `summary.json` (stopping rule), `family_checks.json`; `results/uncertainty/` — per-case $u$ files (68 cases)
* `results/fit/` — `validation_m{9,17,33}.json`, `selection.json`, `ql_sensitivity_m*.json`, `ql_pertrajectory.json`, `offline_costs.json`
* `models/` — 18 locked weight files under `m9/`, `m17/`, `m33/`; `LOCK.json` (created 2026-11-05T12:21:40Z, lock sha256 `26bdf1dc…`, cases sha256 `51e0626f…`, code identity `5db723e3f2651764`); `LOCK_NOTES.json`
* `results/score/` — 68 case files, 80 rows each; `results/cost/` — `cost.json`, four `timing_*.json`; `results/manifests/` — `CASES.json`, `ENV.json`, `TIMING_ENV.json`
* `results/tables/` — `decision.md` (every rule with every exception), `per_case_models.md` (complete), `summary_by_split.md`, `cell_counts.md`/`.json`, `extrapolation_disclosure.md`, `WARNINGS.md`; `results/decision.json`; `results/decomposition/`; `results/panels/`; `results/figures/`
* Regenerable, git-ignored: `results/cache/` (laptop copy at `C:/mcp_jobs/scc_cache`), `results/*_quick*`
* Earlier study (read-only import): `research/dupire_signed_memory/` (`dsm`, Gate 5 script `scripts/gate5_sine_rom.py`)
* Session bookkeeping: unified checkpoint `ckpt_01M6NWAGHRNK845BMQX76WTCA0`; Keep follow-ups `FU-20261105-2100a` (push/PR — done), `…b` (n = 511 rerun), `…c` (positivity-preserving decoder), `…d` (optional LR OOD stress study); auto-memory `sine_closure_study.md`, `sine_closure_outcome.md`

---

## 7. Future Work

The verdict is not to be re-litigated without a new pre-registered addendum and a *fresh* held-out set: the present test cases have been scored and are now development data.

1. **The m = 17 price question (immediate, pre-registration needed).** Repeat the comparison on a 511-point model grid with 1021/2041-point references ($u_{\mathrm{price}}$ halves twice, ≈ /6.5) restricted to the low-σ regime where the Rule-B eligibility holds, with a new test set. Not run here because the plan froze 256 as the production grid. (Keep `FU-20261105-2100b`.)
2. **Density-valid small models (medium term).** The validity failure is a reconstruction question, not a dynamics one. A positivity-preserving output decoder, counted as a separate model with its own cost, is the only route identified; it must be pre-registered as such. (Keep `FU-20261105-2100c`.)
3. **Optional LR out-of-distribution stress study.** The stateful closure's OOD resolved-error gain (0.65–0.70×) with worse density and negative mass was a stress observation; a dedicated, pre-registered study would be needed to make it a claim. (Keep `FU-20261105-2100d`.)
4. **Annotate the Gate 5 speed-up claim.** Gate 5's "5–17× faster than the un-reduced solve" needs a dated addendum noting that the comparison was against the dense exponential solver; against the banded CN solver (21 ms) there is no speed-up at $n = 256$.
5. **Application-tolerance negative-mass assessment.** The plan allows only the strict paired rule; a tolerance-based assessment would need its own addendum.

Not warranted by this study: nonlinear, reservoir or neural closures (escalation required a robust held-out LR gain over LM, which did not occur), and any market-data or production-Fourier-model claim (all data are synthetic semidiscrete references; verdicts are statements about the OP-B problem on the fixed strike domain, with Rule C's domain caveat on high-σ cases).

---

## Appendix A. Session execution record

_This appendix records how the study was run, for reproducibility of the process rather than of the numbers (the numbers are reproducible from the scripts and identities above)._

**Order of operations (pre-registration preserved).** Plan frozen and committed → `scc` modules and tests → s0 verification and Gate 5 reproduction → development references → s3 diagnostics → Addendum 2 committed → s4 fit and lock committed, `LOCK_NOTES` expectations written → test references → s5 scoring → s6 cost (run alone) → s7 report and memo → README, memory, checkpoint, Keep follow-ups → push and PR #10.

**Failures encountered and fixes adopted.**

* The interior system initially evaluated the coefficient row at $t = 0$ while the earlier study uses $T = T_{\max}$ for autonomous cases; found in verification against `dsm`, fixed by `conventions.AUTONOMOUS_T = 1.0` and shared by every operator helper (Addendum 1), before any scoring.
* `PermissionError` on `os.replace` of `RUN_MANIFEST.json` / `index.json` under Dropbox file locks on Windows killed the first s0 and s2 jobs; fixed by `identity.replace_retry` (exponential backoff), also used by `modelio`.
* Three of four parallel s3 group jobs died silently with exit 0 and near-empty logs: the liveness probe `os.kill(pid, 0)` *terminates* the target process on Windows, so the groups killed each other; replaced by a ctypes `OpenProcess`/`GetExitCodeProcess` probe, and s3 was rerun as one sequential job over all 41 cases (≈ 5 min; 2–21 s per case, D and M families slowest).
* The quick-mode s0 run failed its time-ordered-propagator check (8 asserted failures) before the rollout was patched; the full-size verification then passed.
* The cache-identity migration first reported "0 re-keyed" because the old scheme hashed bare module names (`6941aa686ecce5d1`, still visible in the s0 identities) rather than relative paths; a `--list` option and `--old` argument were added; the bitwise staleness guard failed on 1e-15 thread-count differences and was given a 1e-12 tolerance.
* Quick-mode cache artifacts reached the first commit; removed with `git rm --cached` and ignore patterns (`results/cache_quick/`, `results/verify_quick/`, `results/**/*_quick.json`, `results/*_quick/`, `results/references_quick/`), commit amended.
* The s7 case discovery picked up `RUN_MANIFEST`/`summary`/`family_checks` files as cases; globs patched (the residual `WARNINGS.md` caveat is that artefact).
* The Gate 5 cross-check "failed" only on the time-dependent abscissa, a definitional difference (last-step midpoint $t = 1 - h/2$ vs $t = 1$); per-class tolerances (1e-4 autonomous, 5e-3 time-dependent) and a note were added, check passes.
* Two gpu-laptop MCP `run_job` submissions timed out at the MCP layer with no job registered; resubmitted on request.
* Minor: `timeout` absent on macOS; a zsh glob in `--include=*.py` needed quoting; `unified-checkpoint` requires absolute artefact paths and a `--did` text ≤ 500 characters; the PR-body write was declined when the app quit mid-action and was redone from a heredoc after restart.

**Jobs and runtimes** (GPU-laptop CPU lane, `_laptop_job.py`, job ids from the `gpu-laptop` queue; UTC times of 2026-11-05).

| stage | job id | threads | outcome |
|---|---|---|---|
| s1 Gate 5 reproduction | `693274f1` | — | done, 88 s (two earlier submissions timed out at the MCP layer; `qstat` showed no job registered before the resubmit) |
| s0 verify, first | `014d41bd` | 2 | failed after 1.5 s (`PermissionError` on `RUN_MANIFEST.json`) |
| s2 dev references, first | `d318a9b8` | 1 (×4 parallel) | failed after 14 s (`PermissionError` on `index.json`) |
| s0 verify, retry | `ba0d482f` | 2 | pass, wall 442.6 s (time-ordered propagator check 254 s, reconstruction 9.7 s, solvers 9.9 s, autonomous block 4.9 s) |
| s2 dev references, retry | `74ffa307` | 1 (×4 parallel) | ≈ 15.6 min for 41 cases; dense 511-point solves up to 158 s per case, sparse 1021-point 8–30 s |
| cache identity migration | `0e11a86a` → `441b2d48` (`--list`) → `0deee501` | 4 | 0 re-keyed → old code hash identified → 145 references re-keyed, bitwise staleness guard tripped on 4 cases at 1e-15 (exit 3) |
| s3 diagnose, four parallel groups | `5e9345b2`, `78d0e264`, `903c34f4` (killed, exit 0, 349–445-byte logs), `3de121b3` (10 cases, 92 s) | 4 | liveness-probe bug |
| s3 diagnose, sequential | `a785d6bd` | 8 | 41 cases in ≈ 5 min; family checks D 88 s, M 82 s |
| s4 fit and lock | `05df60e7` | 8 | 120.4 s (lock created 12:21:40) |
| x_ic_term | `6870c024` | 2 | 20.6 s |
| s2 test references | `e1cbfd05` | 2 (×6 parallel) | ≈ 8 min for 27 references (12:23–12:31) |
| s5 score, development splits | `82064e14` | 4 | ≈ 15 min, 41 cases × 80 rows (4.6–57 s per case) |
| x_ql_pertraj | `c6f045b6` | 2 | 7.4 s |
| s5 score, held-out splits | `67cc2b97` | 4 | ≈ 9.4 min, 27 cases (done 12:41) |
| s6 cost, exclusive | `a6b278d9` | 1 | 453.9 s |

The report stage, cross-checks and quick-mode smoke runs ran on the Mac (seconds each). The session began at 10:32 UTC on 2026-11-05 with the brief; the memo commits landed the same day; the branch was pushed and PR #10 opened at about 05:03 UTC on 2026-11-06.

**Tests.** The pytest suite (`research/sine_closure/tests`, 8 files) was run seven times during the session: 52 passed at 11:40 UTC (before the evaluate core existed), then 69 passed in ≈ 2 s on every run from 11:48 UTC onward, including after each infrastructure fix (retrying replace, narrowed reference identity, liveness probe). No failing run is recorded.

**Checkpoints.** Five `unified-checkpoint` records were written during the session; the final one is `ckpt_01M6NWAGHRNK845BMQX76WTCA0`.
