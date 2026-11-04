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

**Short answer.** _(filled in §6 after Gates 3–4)_

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

_(Basis diagnostic, Gate 2b, and Gates 3–4 follow.)_
