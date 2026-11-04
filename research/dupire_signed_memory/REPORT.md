# Signed memory in the Dupire reduced model — report

_Study directory `research/dupire_signed_memory/`, branch `claude/brave-brahmagupta-twxtst`,
base commit `9a679cb` of `ameirtheshaa/LocalVolatility`. All numbers below are in
`results/` (machine-readable) and `results/gate4/TABLES.md`; the derivation is in
`docs/DERIVATION.md`; the plan was frozen in `FROZEN_PLAN.md` before any closure was scored._

## 1. Question and short answer

**Question.** Does the Fourier-truncated Dupire reduced model in `mz_spectral` need signed,
stateful feedback from the eliminated modes, or is the nonnegative scalar eddy-diffusion
restriction of `fit_nu_ql` (`docs/MZ_QL_NONNEG_FIX_REPORT.md`) justified on variable-volatility
problems?

**Short answer** (updated after the wrap-up, §6a and §9). The restriction is not justified by the physics and not harmful in practice.
The exact feedback of the eliminated modes projects *negatively* on the eddy-diffusion regressor
in every tested case (constant, strike-dependent and time-dependent volatility, both operator
variants, all truncations), so "eliminated modes must add diffusion" is false; but a scalar
coefficient of either sign is useless — with `ν ≥ 0` the closure is a no-op, with `ν` free it is
a sub-resolution gain or an instability. The feedback is a basis artifact: with boundary-adapted
modes the truncation is exact to `1e-7`–`1e-10`, and within the Fourier split a stateless linear
map on the resolved state removes 98–99 % of the resolved error. Signed *and* stateful
feedback is therefore neither needed nor sufficient; what is needed is the basis. Re-scoring
under the strict negative-mass rule (§6a) leaves every verdict unchanged but removes the one
formal "stateful helps" result; no closure improves the density in the Fourier split. A
33-mode boundary-adapted (sine) model needs no closure and matches the un-reduced solve to grid
accuracy with valid densities on a constant-σ control, a held-out strike smile and a fresh
time-dependent smile (§9); that is the practical recommendation (§10).

## 2. What was tested (Gate 0: inventory and frozen plan)

* Operator and projector exactly as implemented: `L = diag(η̃ k̃²) D2` on the repository
  `(t̃, k̃)` grid, `D2` with one-sided boundary rows, Euclidean Fourier low-pass projector on the
  raw scaled call price (**OP-A**). A minimal boundary-consistent variant (**OP-B**: linearity
  boundary rows, projector on the detrended price) was added because OP-A's orthogonal
  dynamics turn out to be divergent (§4), and only a convergent memory can answer a sign
  question.
* 18 deterministic cases: constant σ (5), smooth strike-dependent smiles (10 seeds), the
  repository's time-dependent `dupire_exact` smile (3 levels); train/validation/held-out
  splits by whole trajectories; truncations `keep ∈ {16, 32, 64}`; `n_k = 256` with 512
  refinement; references from `expm` / second-order Magnus with 368 steps on `t̃ ∈ [0.2, 1]`.
* Closures: resolved-only (RRR), the existing nonnegative QL (QL+), the matched signed QL
  (QL±), the unclipped scalar fit (QL*), the exact-memory oracle, and — because the oracle gap
  is reproducible — a linear Markovian ridge map (LM), a linear reservoir (LR) and a compact
  echo-state reservoir (ESN), all driven only by the resolved state.

## 3. Gate 1 — the mathematical contract (`results/gate1/gate1_contract.json`)

The exact resolved equation of the discrete system has five terms (DERIVATION §3): the
Markovian part `L̂_RR c`, the resolved boundary forcing, the unresolved initial-condition term
`L̂_RU e^{(t-t0)L_UU} u(t0)`, the memory `∫K(t-s)c(s)ds` with `K(τ) = L̂_RU e^{τ L_UU} L_UR`, and
the boundary forcing routed through the orthogonal dynamics. For a time-dependent operator the
exponentials become time-ordered propagators.

| check | result |
|---|---|
| analytic control `x' = -2x+y, y' = x-2y` | resolved-equation residual `4.2e-8`; kernel `e^{-2τ} > 0`; resolved work `> 0` at every instant; fitted coefficient on the damping regressor `ν* = -0.70` (anti-damping sign); closed model stable, rate `-1.30` between the truncated `-2` and the exact Markovian limit `-1` |
| OP-B identity, full horizon, `u(t0) ≠ 0` | unresolved reconstruction rel. error `6.4e-5`, resolved `1.2e-5`; unresolved-IC feedback decays from `0.81` to `8e-19`; memory and Markovian terms both `≈ 8.2` at `t̃ = 1` |
| OP-C identity with Black–Scholes Dirichlet data | boundary values track BS to `4e-9`; rel. errors `3.5e-5 / 6.8e-6`; forced solve matches BS to `3.8e-6` (vs `2.0e-5` OP-B, `3.2e-5` OP-A), so domain truncation + boundary treatment is a `2e-5` effect at σ = 0.3 |
| OP-A identity | exact for 2–5 steps (`6e-8`, `6e-5`), then `7.4` at 10 steps, `1e11` at 20, overflow by 160: `max Re eig(L_UU) = +1.07e3` at `keep = 32` while `max Re eig(L) = +0.405`; numerical abscissa of `L` is `2.0e4` (one-sided rows) |
| sign theorem numerics (interior Dirichlet system) | `W`-orthogonal projector: `sym(K̃(τ))` has `min eig -6.5e-11` (PSD, theorem holds); Euclidean Fourier projector: `min eig -1.3e-2` against `max eig 5.3e5` (indefinite but overwhelmingly positive) |
| eigenbasis projector | `‖L̂_RU‖ = 5.9e-11` vs `‖L̂_RR‖ = 2.8e3`: no memory for constant `L` |

The control and the theorem together refute the premise that eliminated modes must act as
extra diffusion: for any self-adjoint dissipative system with a compatible projector the kernel
is positive semidefinite and the Markovian truncation over-damps.

## 4. Gate 2 — exact references and artifact diagnosis (`results/gate2/`)

**Reference validity.** The OP-B references are valid everywhere (min φ̃ = 0, negative density
mass ≤ `6e-7`, no butterfly or calendar violations). The production-stencil (OP-A) references
are not: negative scaled prices in every family (down to `-1.0e-2` at σ = 0.4, `-2.7e-2` on the
worst strike smile, `-0.32` on the repository smile), negative density mass up to `1.0e-2`,
`4.2e-2` and `0.27`, and 20 % butterfly/calendar violations on the repository smile. The
one-sided boundary rows make `L` non-dissipative (`max Re eig = +0.4` at σ = 0.3, growing with
σ); on the repository smile the growing boundary mode dominates the solution at the production
grid.

**Refinement and truncation.** Grid 256 → 512 changes the interior reference by `4.6e-6` (C),
`4.4e-6` (S), `3.0e-6` (OP-B smile) and `4.4e-4` (OP-A smile, the unstable mode); the Magnus
step by `1e-7`; the strike-domain truncation `[500, 3000]` by `3.2e-5` (C), `4e-6` (S) but
`5.8e-3` (OP-B) / `2.8e-2` (OP-A) on the repository smile, whose diffusion (σ up to 0.67) is not
contained by the production strike range. Projection errors at `keep = 16/32/64` are
`7.8e-3 / 3.9e-3 / 2.2e-3` (OP-A, raw price) and `3.2e-4 / 8.7e-5 / 2.2e-5` (OP-B, detrended).

**Closure targets from the operator.** With `M(t) = Vᵀ L Q φ(t)` computed along the exact
reference (no time differencing):

| variant | keep | ν* range over 18 cases | cases with ν* < 0 | repository `fit_nu_ql` (nonneg) | repository FD estimate, unclipped | weighted work > 0 (fraction of time, mean) | max Re eig `L_UU` |
|---|---|---|---|---|---|---|---|
| A | 16 | `-1.3e-1 … -1.3e-2` | 18/18 | `0` | `-1.3e-1 … -1.3e-2` | 1.00 | `+5.9e2` |
| A | 32 | `-1.4e-1 … -1.4e-2` | 18/18 | `0` | same | 1.00 | `+3.2e3` |
| A | 64 | `-1.4e-1 … -1.5e-2` | 18/18 | `0` | same | 0.04 | `+1.5e4` |
| B | 16 | `-1.4e-1 … -1.1e-2` | 18/18 | `0` | same | 0.95 | `≈ 0` |
| B | 32 | `-1.1e-1 … -8.3e-3` | 18/18 | `0` | same | 1.00 | `≈ 0` |
| B | 64 | `-2.1e-2 … -1.7e-3` | 18/18 | `0` | same | 1.00 | `≈ 0` |

Three facts follow. (i) The sign restriction binds on every case, truncation and variant: the
nonnegative closure is identically the resolved-only model (`ν = 0`). (ii) The repository's
central-difference estimator agrees with the operator-based coefficient to three digits, so the
sign is not a finite-difference artifact. (iii) The coefficient scales with σ² and shrinks with
`keep` on OP-B but not on OP-A, where the feedback is as large as the Markovian term
(`‖L̂_RU u‖/‖L̂_RR c‖ ≈ 1.00` in every case) because it is the projection artifact of a periodic
basis applied to a price that jumps from `≈ 0.85` to `0` across the periodic seam. The
instantaneous resolved work is metric- and truncation-dependent (Euclidean: all negative at
`keep = 16`, all positive at 32/64 on OP-B; weighted: positive except OP-A at `keep = 64`),
which is why neither it nor the fitted sign can stand in for the other.

**Basis diagnostic (Gate 2b, `results/gate2/basis/`, diagnostic only).** Resolved-only
(Galerkin) error on OP-B at matched dimension `2·keep+1`, geometric mean over the cases of each
family (projection error in parentheses):

| family | keep | repository Fourier modes | Dirichlet sine modes | weighted eigenbasis of `L(t̃=1)` |
|---|---|---|---|---|
| C (5) | 16 / 32 / 64 | `2.1e-2 / 1.8e-2 / 1.4e-2` (`3.2e-4 / 8.7e-5 / 2.2e-5`) | `7.9e-8 / 3.9e-9 / 1.7e-10` (`1.4e-7 / 7.1e-9 / 2.9e-10`) | `5e-15` (`≤ 3e-15`) |
| S (10) | 16 / 32 / 64 | `2.3e-2 / 2.0e-2 / 1.6e-2` (`3.2e-4 / 8.7e-5 / 2.2e-5`) | `1.1e-7 / 4.9e-9 / 2.0e-10` (`1.9e-7 / 9.1e-9 / 3.7e-10`) | `4e-15` |
| D (3) | 16 / 32 / 64 | `6.1e-2 / 5.5e-2 / 4.9e-2` (`2.5e-4 / 6.7e-5 / 1.7e-5`) | `3.0e-7 / 1.1e-8 / 4.1e-10` (`4.0e-7 / 1.7e-8 / 6.6e-10`) | `3.0e-10 / 7.6e-12 / 2.2e-13` |

With a basis that satisfies the boundary conditions, Galerkin truncation is exact to `1e-7`–`1e-10`
at 33 modes; in the eigenbasis the only memory left is the one generated by the time dependence
of the repository smile, `3e-10`. The Fourier "memory" the QL closure is asked to model is
therefore, to five orders of magnitude, the wrap-around derivative mismatch of a periodic basis
applied to a non-periodic function. It does not change with the sign convention of any closure.

## 5. Gate 3 — frozen comparison (`results/gate3/blocks/`)

All closures start from the same projected reference state at `t̃0 = 0.2` (unresolved remainder
discarded), use the same ETD2 integrator (oracle floor `1e-7`–`2e-6`), and are scored on the
held-out split only after fitting on the training split with ridge chosen on validation.
`u` is the refinement band (RRR at `n_k = 256` vs `512`). "Helps" is the prespecified rule: on
every held-out case, stable, below RRR by more than `u`, no increase in negative density mass or
invalid inverse-volatility fraction.

**OP-A (production split).** Pooled `ν_unclipped = -4.3e-2 / -4.5e-2 / …` (`keep = 16 / 32 / 64`),
so QL+ is `ν = 0 ≡ RRR` and QL± sits at the clip `-5e-4`.

| keep | u | RRR test | QL+ | QL± | QL* (`ν_unclipped`) | ORACLE | LM | LR | ESN |
|---|---|---|---|---|---|---|---|---|---|
| 16 | `1.4e-3` | `1.03e-1` | `= RRR` | `1.05e-1` (worse) | unstable 0/4 (abscissa `+5.0e2`) | `1.8e-6` | unstable (abscissa `+4.8e5`) | unstable | unstable |
| 32 | `2.2e-3` | `1.10e-1` | `= RRR` | `1.13e-1` (worse) | unstable (abscissa `+2.2e3`) | `2.1e-6` | unstable | unstable | unstable |

Even the exact resolved part of the price (ORACLE) yields an invalid density on OP-A
(negative mass `35–78`, inverse-vol invalid on `52 %` of the interior band): at these
truncations the resolved Fourier field of the raw price cannot represent the density at all.
Nothing deployable improves OP-A; the only cure is the basis/boundary treatment.

**OP-B (boundary-consistent split).** Pooled `ν_unclipped = -3.3e-2 / -2.5e-2 / -5.0e-3`
(`keep = 16 / 32 / 64`); QL+ is again `ν = 0 ≡ RRR`, QL± sits at the clip `-5e-4`. Held-out
resolved-interior RMS (mean over the 4 held-out cases; stability as stable/4):

| keep | u | RRR | QL± (`-5e-4`) | QL* (`ν_unclipped`) | ORACLE | LM (stateless linear map) | LR (linear reservoir) | ESN (tanh reservoir) |
|---|---|---|---|---|---|---|---|---|
| 16 | `2.2e-3` | `3.58e-2` | `3.45e-2` (4/4) | unstable 0/4 | `1.1e-7` | `4.7e-4` (4/4) | `1.2e-4` (4/4) | `1.07e-1` (3/4) |
| 32 | `2.8e-3` | `3.20e-2` | `3.04e-2` (4/4) | unstable 0/4 | `8.6e-8` | `4.4e-4` (4/4) | unstable 0/4 | `6.2e-3` (4/4) |
| 64 | `6.2e-3` | `2.75e-2` | `2.59e-2` (4/4) | 1/4 stable, `4.2e-2` where stable | `6.9e-8` | `4.0e-4` (4/4) | `5.4e-4` (4/4) | `7.3e-4` (4/4) |

* **Signed scalar closure.** QL± improves on RRR in every one of the 18 cases at every keep
  (held-out gains `+0.6e-3` to `+1.9e-3`, i.e. 2–13 %), but every gain is inside the refinement
  band `u`, so by the frozen rule it does not help. The unclipped coefficient makes the closed
  operator unstable (`max Re eig = +3.6e2 / +1.1e3 / +4e2`), exactly as the control predicts
  when the Markovian limit is over-shot: the sign is right, the ansatz is wrong.
* **Non-scalar closures.** A full linear Markovian map `M ≈ C c + b` (LM, `33²`–`129²`
  coefficients, ridge `1e-4`) reduces the held-out resolved error 60–76× at every keep, is
  stable on every case, and its closed autonomous operator has spectral abscissa `-0.09` to
  `-0.15`. The linear reservoir is better than LM at `keep = 16` (`1.2e-4`), unstable for every
  ridge at `keep = 32`, and slightly worse than LM at 64; the tanh reservoir is never better
  than its linear twin. State and nonlinearity therefore add nothing robust: the feedback is
  effectively a Markovian linear functional of the resolved state, which is what one expects
  of a basis artifact (the Fourier tail of a smooth non-periodic function is fixed by its
  boundary derivatives, a linear functional of the resolved coefficients).
* **Density side.** The price-side gains do not reach the density. Even the exact resolved
  field (ORACLE) has density L2 error `2.0e-2`, negative mass `1.2` and an invalid
  inverse-volatility estimate on 28 % of the interior band, the same as RRR, because the second
  derivative of a 33–129-mode Fourier reconstruction rings. LM/LR/ESN raise the negative mass
  by 5–40 % (`1.25 / 1.41 / 1.73` vs `1.19` at `keep = 64`), which is why the frozen rule
  rejects them at `keep = 32, 64`; only LM and LR pass at `keep = 16`.

## 6. Gate 4 — verdict under the prespecified rules (`results/gate4/`)

| hypothesis | outcome |
|---|---|
| **H-sign** — eliminated modes add diffusion (`ν* ≥ 0`) | **rejected in 108/108** case × keep × variant combinations; the operator-based coefficient is negative everywhere, the repository's own finite-difference estimator agrees to three digits, and the weighted resolved work is positive (energy injected) at every instant on OP-B |
| **H-scalar** — a scalar coefficient of either sign closes a useful fraction of the oracle gap | **not supported**: QL+ is identically RRR; QL± gains are consistent but inside the refinement band; QL* is unstable |
| **H-state** — a compact stateful surrogate helps where scalar closures do not | formally met at one block (`B_keep16`, LR), but the matched stateless baseline LM helps there too and is more robust elsewhere, so **state is not the operative ingredient** |
| Restriction verdict per block (rule of `FROZEN_PLAN.md`) | `justified` in all six blocks, in the narrow sense the rule defines: relaxing the sign inside the scalar eddy-diffusion ansatz never clears the refinement band and the unclipped value destabilises |

**Answer to the question.** Within the tested scope the nonnegative restriction is *justified
only as a safety clamp on an ansatz that cannot represent the feedback*: with `ν ≥ 0` the
closure is a no-op (`ν = 0` on every case), with `ν` free it is either a sub-resolution gain or an
instability. As a physical statement — that the eliminated modes act as extra diffusion — the
restriction is wrong on every tested constant-, strike- and time-dependent-volatility case, in
line with the analytic control and the sign theorem. What the reduced model actually needs is
not a signed scalar but (i) a basis that respects the boundary conditions, after which the
Galerkin truncation is exact to `1e-7`–`1e-10` and no closure is needed, or, if the Fourier
split is kept, (ii) a full linear Markovian map on the resolved state, which is deployable and
stable but does not repair the density. On the production split (OP-A) nothing deployable
helps, and the production stencil itself produces negative prices and invalid densities on the
repository smile at the production grid. The existing report's conclusion that the forward QL
memory is "cosmetic" stands; its explanation (anti-diffusion is unphysical) does not.

## 6a. Erratum and re-scoring (2026-11-04, after the first release)

**What was wrong.** The frozen rule says a closure must not *increase* negative density mass
relative to RRR; the first `gate4_evaluate.py` allowed a 5 % increase. The first scoring is
preserved unchanged as `results/gate4/evaluation_v1_tolerant.json` / `TABLES_v1_tolerant.md`;
the corrected scoring (any increase beyond a `1e-9` round-off guard) is now the default and the
old behaviour is reproducible with `--rule tolerant --u-source all`
(`evaluation_v1check.json`). A second variant scores the negative mass at the last maturity
instead of the row maximum (`evaluation_Tmax.json`), because the row maximum is usually set by
the shared initial row and is blind to the closure (FROZEN_PLAN Addendum 2).

**Disclosure.** `REFINEMENT_CASES` contained the held-out case `S_seed8`; it entered the
refinement band `u` (never a fitted parameter). Recomputing `u` from the training refinement
cases only gives the same `u` at every block (`S_seed8`'s band was never the maximum), so no
number changes, but the held-out split was not fully untouched and is reported as such.

**What changes.** Held-out "helps" decisions (every case stable, gain > u, no increase of
negative mass or invalid inverse-vol fraction):

| block | closure | v1 (5 % tolerance) | strict, row-max negative mass | strict, last-row negative mass |
|---|---|---|---|---|
| B_keep16 | LM | helps | no (negative mass increases on 1 case) | no |
| B_keep16 | LR | helps | no (increase on 2 cases) | no |
| B_keep16/32/64 | ORACLE | helps | helps | no (last-row negative mass higher than RRR on all 4 cases) |
| all blocks | QL+, QL±, QL* | no | no | no |
| B_keep32/64 | LM, LR, ESN | no | no | no |

The per-block verdicts ("justified" everywhere) and H-sign/H-scalar are unchanged;
`H_state_supported_anywhere` becomes **False**. The exact resolved field itself fails the
last-row density rule, which shows that the density component of the rule measures the
Fourier projection's ringing, not the closure; it cannot discriminate between closures in this
split and should not be read as evidence against the linear map's price-side gain.

**Support classification of each conclusion.**

| conclusion | evidence | status |
|---|---|---|
| Eliminated modes do not add diffusion (`ν* < 0`) | 108/108 combinations, operator-based and finite-difference estimators agree, analytic control and theorem | supported |
| The nonnegative clip makes QL a no-op | `ν = 0` fitted on every case | supported |
| The signed clip (`-5e-4`) improves the resolved error | direction consistent on 18/18 cases and 4/4 held-out at every keep, but every gain is 0.3–0.9 of `u` | **uncertain** (inside refinement uncertainty) |
| The unclipped coefficient destabilises | abscissa `+3.6e2` to `+2.2e3`; unstable on 12/12 held-out rollouts at `keep = 16, 32`, 3/4 at 64 | supported |
| A stateless linear map removes 98–99 % of the resolved error | held-out gains 60–76× `u` at every OP-B block, stable, abscissa `< 0` | supported for the resolved price; **not** for the density (fails the strict rule, and so does the exact resolved field) |
| State (reservoirs) adds nothing robust | LR unstable at `keep = 32`, never better than LM except `keep = 16`; ESN never better than LR | supported within the fixed budget only |
| The feedback is a basis artifact; boundary-adapted bases need no closure | Gate 2b: Galerkin error `1e-7`–`1e-10` at 33 modes in sine/eigen bases vs `1e-2` Fourier | supported for the resolved price; density, cost and stability tested in §9 |
| OP-A production references are unphysical | negative prices and negative mass in every family at the production grid | supported |

## 9. Gate 5 — a small boundary-adapted reduced model (`results/gate5/`)

The most promising practical improvement identified in §4 was tested as a deployable model:
Dirichlet sine modes on the detrended price with the boundary-consistent stencil (`B_sine`,
17/33/65 degrees of freedom, resolved-only Galerkin, no closure), against the production Fourier
model (`A_fourier`), the boundary-consistent Fourier model (`B_fourier`), the un-reduced 256-point
solves, and the refined 512-point boundary-consistent solve as reference (Black–Scholes for the
control). Three cases: the constant-σ control `C_sig0.30`, the held-out strike smile `S_seed9`, and
one fresh time- and strike-dependent smile never used before
(`F_fresh`: `σ = 0.28 + y e^{-y}`, `y = (t+0.2)√(x+0.05)`, defined only in `gate5_sine_rom.py`).
Two start protocols: from the projected reference at `t̃0 = 0.2` (as in Gate 3) and from the
payoff at `t̃ = 0` (deployment). Numerical uncertainty `u_grid` is the 256-vs-512 interior RMS
difference of the full solve (`4.0e-6`, `3.6e-6`, `3.0e-6`); for the fresh case the step
refinement adds `5.8e-8`. Interior RMS of the scaled price at the 24 repository maturities,
start from the payoff (the projected-start numbers are equal or smaller):

| case | model | dof | price RMS | density L2 | negative mass (max row) | inverse-vol invalid | stable | rollout cost |
|---|---|---|---|---|---|---|---|---|
| C_sig0.30 | A_fourier (production) | 33 | `9.9e-2` | `4.4e-2` | `1.5` | 0.154 | yes | 7 ms |
| | B_fourier | 33 | `3.0e-2` | `2.1e-2` | `0.94` | 0.539 | yes | 7 ms |
| | **B_sine** | **17** | `1.0e-5` | `1.0e-4` | `6e-4` | 0.169 | yes (abscissa `-0.21`) | 8 ms |
| | **B_sine** | **33** | `4.2e-6` | `1.4e-5` | `1.9e-6` | 0.169 | yes | 8 ms |
| | B_full256 (un-reduced) | 256 | `4.0e-6` | `3.3e-6` | `6e-7` | 0.169 | yes | 140 ms |
| S_seed9 | A_fourier (production) | 33 | `1.1e-1` | `3.9e-2` | `1.3` | 0.209 | yes | 7 ms |
| | B_fourier | 33 | `4.4e-2` | `2.2e-2` | `0.87` | 0.523 | yes | 19 ms |
| | **B_sine** | **33** | `3.8e-6` | `2.1e-5` | `1.8e-6` | 0.075 | yes (`-0.33`) | 24 ms |
| | B_full256 | 256 | `3.6e-6` | `2.5e-6` | `4e-7` | 0.076 | yes | 138 ms |
| F_fresh | A_full256 (production, un-reduced) | 256 | `1.6e-2` | `2.8e-3` | `9.0e-2` | 0.191 | yes | 48 s |
| | A_fourier (production) | 33 | `1.2e-1` | `2.4e-2` | `0.92` | 0.267 | yes | 8.8 s |
| | B_fourier | 33 | `7.8e-2` | `1.8e-2` | `0.69` | 0.425 | yes | 8.6 s |
| | **B_sine** | **17** | `8.1e-6` | `1.6e-4` | `7e-7` | 0.005 | yes (`-0.71`) | 8.7 s |
| | **B_sine** | **33** | `3.1e-6` | `2.6e-5` | `2.3e-7` | 0.004 | yes | 9.4 s |
| | B_full256 | 256 | `3.0e-6` | `9.0e-7` | `2.7e-8` | 0.004 | yes | 46 s |

(The fresh case has a time-dependent operator, so every model pays one small matrix
exponential per step; the constant-coefficient cases pay one exponential in total. Costs are
medians of three rollouts on the session's CPU and are only meaningful relative to each other.)

**Claims that exceed numerical uncertainty** (`results/gate5/comparison.json`): on every case
and under both start protocols, the 33-mode sine model's price error is within `u_grid` of the
un-reduced 256-point solve (it is numerically indistinguishable from not reducing at all), its
gain over the 33-mode Fourier models is `2.2e-2`–`1.2e-1`, i.e. `10³`–`10⁴` times `u_grid`, its
density error is `10³` times smaller than the Fourier models' and the gain exceeds the density
uncertainty, its negative density mass drops from `0.7–45` to `≤ 2e-6`, and its invalid
inverse-volatility fraction equals that of the full solve. The 17-mode model is within 2–3
`u_grid` of the full solve. Rollout cost is 5–17× below the un-reduced solve at equal time step.
The production stencil's own un-reduced solve on the fresh smile has price error `1.6e-2`,
negative mass `0.09` and 19 % invalid inverse-volatility estimates, so on such smiles the
boundary-adapted 17-mode model is more accurate than the production full solve by four
orders of magnitude.

**What this does not show.** Three cases, one grid family, one time horizon and the study's
own semidiscrete references; no noisy or market data; the sine basis relies on the
boundary-consistent (linearity) stencil, whose domain-truncation error on the repository smile
is `5.8e-3` (Gate 2), common to all models here; the 512-point reference shares the stencil
and the strike domain with the models, so agreement with it measures discretisation, not
physical truth, except on the control where Black–Scholes confirms it (`4e-6`).

## 10. Recommendation

1. Do not use the Fourier-truncated model (either stencil) for densities or local-volatility
   inversion: at 33–129 modes its density error, negative mass and inverse-volatility validity
   are those of the projection, not of any closure, and no tested closure repairs them.
2. If a reduced model is wanted, replace the Fourier modes by Dirichlet sine modes on the
   detrended price with the linearity boundary rows: 33 modes reproduce the 256-point solve
   to grid accuracy on all three test cases, with valid densities and no closure, at 5–17× lower
   rollout cost (§9). This is a change of basis and boundary treatment, not of the closure.
3. Drop the sign restriction debate: the eliminated-mode feedback is anti-diffusive in every
   tested case, but a scalar coefficient of either sign is useless (§5–§6a). If the Fourier
   model must be kept for the resolved price only, a stateless linear map on the resolved state
   is the only closure that helped beyond uncertainty, and it does not repair the density.
4. Treat the production one-sided boundary rows as a defect to fix before any further
   reduced-model work: they make the operator non-dissipative and produce negative prices and
   invalid densities on the repository smile at the production grid (§4, §9).
5. Widen the strike domain for strong smiles (domain truncation `5.8e-3` on the repository
   smile, §4).

## 7. Limitations (first release; see also §9 for Gate 5)

* Synthetic, noise-free, semidiscrete references only; nothing here bears on market
  performance, on the P9 tail-energy detector, or on plasma/DNS closures.
* One grid family (`n_k = 256`, refinement 512), one time horizon (`t̃ ∈ [0.2, 1]`), three
  truncations, 18 cases; the refinement band `u` (`1.4e-3`–`6.2e-3`) is coarse compared with
  the QL± gains, so a finer study could promote QL± from "inside the band" to "helps" without
  changing the sign conclusion or the 60× gap to LM.
* The surrogate budget was fixed in advance (one reservoir size, one spectral radius, one leak,
  four ridge values); the ESN's poor showing at `keep = 16` is a property of that budget, not a
  theorem.
* Density metrics are computed with the repository's one-sided `D2`; the "invalid inverse-vol"
  fraction counts points whose curvature is below `1e-3` of the row maximum as undefined, which
  is why even valid references show 18–20 % on the constant-σ family (0.9 % on the smile).
* Family D's strike domain `[500, 3000]` truncates the repository smile (`5.8e-3` interior
  change with a wider domain); all closure comparisons are relative to the truncated-domain
  semidiscrete system, so this does not affect the MZ identities but bounds physical fidelity.
* Noisy-observation fits (secondary in the plan) were not run.
* The reference commit named in the brief (`f539f41`) is not in this repository; see
  `SOURCE_INVENTORY.md`.
* The held-out case `S_seed8` was one of the three refinement cases that define `u` (§6a).

## 8. Release and recovery

* Code: `dsm/`, `scripts/`, `tests/` (12 tests, ~10 s). Dependencies: `requirements.txt`
  (NumPy 2.5.3, SciPy 1.18.1, matplotlib, pytest; Python 3.13 verified).
* Results: `results/gate1/gate1_contract.json`, `results/gate2/*.json` (+ `basis/`),
  `results/gate3/blocks/*.json`, `results/gate4/{evaluation.json,TABLES.md}`; surrogate weights
  in `fixtures/*.npz` with sha256 in the Gate 3 block files; checkpoints with Git commit, source
  hashes, environment, seeds, case status, commands, exit codes and result checksums in
  `checkpoints/`; `NEXT_ACTIONS.md` names the one safe next command.
* Reproduce / resume (repository root, Mac/Linux, no account): see `README.md`;
  `python3 research/dupire_signed_memory/scripts/run_all.py --resume` re-runs only what is
  missing. Interruption/resume is exercised by `tests/test_resume.py` and by
  `scripts/restore_rehearsal.sh` (fresh clone, venv, tests, Gate 1 recomputed and compared
  numerically with the committed JSON, interrupted + resumed Gate 2 case). Rehearsal outcome
  on 2026-11-04 from a fresh clone of commit `f68a9ce`: 12 tests passed, Gate 1 and the resumed
  Gate 2 case agreed with the committed results on every numeric field (`verify_results.py`).
  Two earlier rehearsal runs failed on exactly this comparison and caught stale diagnostic
  fields (`kernel_norms` computed with the spectral norm before the Frobenius change); the
  affected 16 result files were regenerated with the committed code.
* Failed attempts preserved in Git history: the first Gate 2 run (kernel-norm overflow on the
  production split, commit `90f2a2b`) and the first Gate 3 run (progress-print bug that
  overwrote finished blocks, fixed in `9fd4120`).
* Budget: cloud credit accounting was unavailable in the session (`BUDGET.md`); all pushes
  succeeded, so no export bundle was needed.
