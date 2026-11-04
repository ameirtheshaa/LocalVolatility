# Mathematical contract: exact resolved dynamics of the discrete Dupire operator

Source-grounded derivation for the operator, projector and conventions actually implemented in
`Synthetic_Data_Tensorflow_Advanced/mz_spectral` (commit `9a679cb`). Notation follows the code.

## 1. The semidiscrete system

`fourier_dupire.py` discretises the scaled Dupire equation `∂φ̃/∂t̃ = η̃(t̃,k̃) k̃² ∂²φ̃/∂k̃²` on a
uniform `k̃` grid of `n` points as

    φ'(t) = L(t) φ(t) + b(t),        L(t) = D(t) D2,   D(t) = diag(η̃(t, k̃_i) k̃_i²) > 0.      (1)

* `D2` is the second-difference matrix. **OP-A** (production): central interior rows and the
  one-sided rows `(2,-5,4,-1)/dk²` at both ends; the system is closed, `b ≡ 0`, and the boundary
  values evolve by extrapolation. **OP-B**: boundary rows zero (linearity condition `φ̃_kk = 0`),
  `b ≡ 0`, boundary values constant. **OP-C**: OP-B stencil with prescribed Dirichlet data
  `φ_0 = g_lo(t)`, `φ_{n-1} = g_hi(t)`, which is (1) with `b(t) = e_0 ġ_lo(t) + e_{n-1} ġ_hi(t)`.
* For OP-B/C the projector is applied to the detrended variable `ψ = φ - ℓ`, `ℓ` the linear
  interpolant of the two boundary values. Since `D2 ℓ = 0` in the interior and the boundary rows
  vanish, `Lℓ = 0`; hence `ψ' = Lψ` (OP-B, `ℓ̇ = 0`) and `ψ' = Lψ + b - ℓ̇` (OP-C). The forcing
  `b - ℓ̇` is the boundary forcing seen by the projected system.
* Initial condition `φ(0) = max(0, 1 - K_max k̃/S0)` (`payoff_phi_tilde`), at `t̃ = 0`, not at
  `t̃ = T_min/T_max` as the existing drivers do.

## 2. Projection and blocks

`mz_decomposition.py` uses the orthogonal low-pass projector `P = V Vᵀ` in the Euclidean metric,
`V ∈ ℝ^{n×m}` the orthonormal real Fourier modes (`m = 2·keep+1`; `V Vᵀ` equals
`low_pass_projector_matrix` to machine precision), `Q = I - P`. Resolved coefficients `c = Vᵀφ`,
unresolved part `u = Qφ`. With `L̂_RR = VᵀLV`, `L̂_RU = VᵀLQ`, `L_UR = QLV`, `L_UU = QLQ`
(`split_operator_blocks` in coefficient form), (1) is exactly

    c' = L̂_RR c + L̂_RU u + Vᵀ b,                                                                (2a)
    u' = L_UU u + L_UR c + Q b.                                                                 (2b)

## 3. Exact resolved equation, time-independent `L`

Variation of constants on (2b) from `t0`:

    u(t) = e^{(t-t0) L_UU} u(t0) + ∫_{t0}^{t} e^{(t-s) L_UU} [ L_UR c(s) + Q b(s) ] ds.              (3)

Substituting into (2a) gives the **contract** that any closure must approximate:

    c'(t) =  L̂_RR c(t)                                   Markovian (what RRR keeps)
           + Vᵀ b(t)                                     resolved boundary forcing
           + L̂_RU e^{(t-t0) L_UU} u(t0)                  unresolved initial-condition term
           + ∫_{t0}^{t} K(t-s) c(s) ds                    memory, K(τ) = L̂_RU e^{τ L_UU} L_UR
           + ∫_{t0}^{t} L̂_RU e^{(t-s) L_UU} Q b(s) ds     boundary forcing routed through U.     (4)

All five terms are needed for equivalence with (1); `scripts/gate1_contract.py` verifies (3)–(4)
numerically on the actual operator (OP-B over the full horizon with `u(t0) ≠ 0`; OP-C with
Black–Scholes boundary data; OP-A only over short horizons, see §7). `e^{τ L_UU}` acts on
`range(Q)` (it is the identity on `range(P)`, which never receives a `Q` vector).

The exact feedback along the true trajectory is `M(t) := L̂_RU u(t) = Vᵀ L Q φ(t)`; it is computable
from the operator and the full solution without any time differencing, and it equals the sum of
the last three lines of (4). It is the closure target used in Gates 2–4.

## 4. Time-dependent operator

If `η̃` depends on `t̃` (`VolatilityConfig.dupire_exact`), `L_UU(t)` does not commute with itself at
different times and `e^{(t-s)L_UU}` must be replaced by the time-ordered propagator
`U_UU(t,s)`, `∂_t U_UU = L_UU(t) U_UU`, `U_UU(s,s) = I`. The kernel becomes
`K(t,s) = L̂_RU(t) U_UU(t,s) L_UR(s)`, no longer a function of `t-s`. Numerically `U_UU` is the
ordered product of midpoint exponentials (second-order Magnus), which is what `reference.py`
uses for the full solve as well.

## 5. Metrics and the three distinct sign notions

**Full-system dissipation.** With a symmetric interior stencil, `L = D D2` is self-adjoint and
negative semidefinite in the weighted inner product `⟨u,v⟩_W = uᵀ D⁻¹ v`:
`⟨u, Lv⟩_W = uᵀ D2 v`. Hence `E_W = ½‖φ‖²_W` satisfies `dE_W/dt = φᵀ D2 φ ≤ 0` (closed stencil).
In the Euclidean metric `L` is not normal; with the production one-sided rows
`max Re eig(L) > 0` and the numerical abscissa is `O(d_{n-1}/dk²)` (boundary rows), as the
repository's own Step 0a records.

**Instantaneous resolved work.** `W(t) = ⟨c(t), M(t)⟩` (Euclidean) or `⟨Vc, W·VᵀᵀM⟩`-type weighted
versions: the rate at which the eliminated modes add (`W>0`) or remove (`W<0`) resolved energy
*relative to the Markovian truncation* at that instant. It may change sign along a trajectory
and between metrics.

**Fitted scalar coefficient.** The QL ansatz `M ≈ ν Q̂ c`, `Q̂ = Vᵀ D2 V` (`P D2 P φ_R = V Q̂ c`),
fitted by pooled least squares `ν* = Σ⟨M, Q̂c⟩ / Σ‖Q̂c‖²`. Since `Q̂` is (nearly) negative definite,
`ν > 0` damps resolved modes (eddy diffusion) and `ν < 0` anti-damps them. `fit_nu_ql` estimates
`M` by central differences of `n_t = 24` trajectory slices instead of from the operator, and
clips to `[0, 5e-4]` (`nonneg=True`) or `[-5e-4, 5e-4]`.

**Autonomous stability of the closed model.** Spectral abscissa and transient growth of
`L̂_RR + ν Q̂`. `ν < 0` is stable as long as `|ν|` stays below the local diffusion scale
`η̃ k̃²` (the clip `5e-4` is below `η̃ k̃_lo² ≈ 1.7e-3` at `σ = 0.3`); `ν > 0` guarantees nothing
about accuracy.

None of the four implies another. The analytic control (§6) has a positive kernel, positive
work at every instant, a negative fitted coefficient on the damping regressor, and a stable
closed model.

## 6. A sign theorem, and why it is only a hypothesis for the actual split

**Proposition.** Let `L` be self-adjoint and `≤ 0` in an inner product `⟨·,·⟩_W`, and let `P` be
`W`-orthogonal. Then `L_UR = (L̂_RU)^{*W}` and, for every `τ ≥ 0`,
`K(τ) = L̂_RU e^{τ L_UU} (L̂_RU)^{*W}` is `W`-positive semidefinite, because `L_UU` is `W`-self-adjoint
so `e^{τ L_UU}` is `W`-positive definite. Consequently the memory integral injects resolved
energy whenever `c` varies slowly (`⟨c(t), ∫K(t-s)c(s)ds⟩_W ≈ ∫⟨c,K(t-s)c⟩_W ds > 0`): the Markovian
truncation `L̂_RR` *over-damps*, and the eliminated modes act *against* diffusion, not as extra
diffusion. The control `x' = -2x + y, y' = x - 2y` is the 2×2 instance: full eigenvalues `-1, -3`,
kernel `e^{-2τ} > 0`, exact Markovian long-time rate `-1` versus truncated rate `-2`.

The "eliminated modes must add diffusion" premise is therefore false in exactly the class of
systems where the full dynamics are cleanly dissipative. For the actual Dupire split none of
the hypotheses holds verbatim: the Fourier projector is Euclidean-orthogonal, not `W`-orthogonal;
the production stencil is not symmetric; OP-B is symmetric only in the interior with constant
boundary coupling. So the sign must be **computed**. Gate 1 does so on the interior Dirichlet
system: with the `W`-orthogonalised Fourier subspace `sym(K̃(τ))` is numerically PSD (theorem),
and with the Euclidean Fourier projector it is indefinite but overwhelmingly positive
(`min eig ≈ -8e-3` against `max eig ≈ 6e3` at `τ = 0.01`).

**Eigenbasis projector.** If `P` projects on invariant subspaces of `L` (the `W`-eigenbasis
of a time-independent `L`), `L̂_RU = 0` identically: there is no memory at all and Galerkin is
exact. Memory in such a basis arises only from time dependence of `L`.

## 7. What the production split does to the contract

For OP-A the orthogonal generator `L_UU = Q L Q` has large positive real eigenvalues
(`+196, +1073, +4975` at `keep = 16, 32, 64`, `n = 256`, `σ = 0.3`) although `max Re eig(L) =
+0.405`. Eigenvalues of a compression lie in the numerical range of `L`, and the one-sided boundary
rows push the numerical abscissa of `L` to `O(10⁴)`. Consequences:

* `e^{τ L_UU}` and hence `K(τ)` grow like `e^{5·10³ τ}`; the three unresolved terms in (4) are
  individually astronomically large and cancel. Identity (4) is still exact in exact arithmetic,
  but no floating-point reconstruction survives more than a few steps (Gate 1 measures the
  growth of the reconstruction error with horizon).
* The decomposition itself is dominated by non-periodicity: `φ̃(k̃_lo) ≈ 0.85`, `φ̃(k̃_hi) ≈ 0`, so
  the Fourier remainder `Qφ` is `O(0.2–0.5)` and `L̂_RU u` is as large as `L̂_RR c` with opposite
  sign (ratio ≈ 1 along the reference). The resolved-only model drops a term as large as the one
  it keeps; this is an artifact of the basis, not evidence of physical memory.

OP-B removes both pathologies with two minimal changes (linearity boundary rows, detrending),
after which `max Re eig(L_UU) ≈ 0`, the kernel decays, and the resolved-only error is of the
order of the projection error. The scientific comparison of closures is therefore run on
both variants, but only OP-B can answer whether a *sign* restriction is justified.

## 8. Discrete-time relation to `fit_nu_ql`

With exact trajectories, `(Pφ)' = L_RR Pφ + L_RU Qφ`, so the central-difference residual
`P(φ_{i+1}-φ_{i-1})/(2Δt) - L_RR Pφ_i` used by `fit_nu_ql` equals `M(t_i) + O(Δt²)`, with
`Δt̃ = 0.8/23 ≈ 0.035` on the repository grid. Gate 3 reports both the operator-based `ν*` and the
repository's finite-difference `ν` (with and without the sign restriction) on the same data.
