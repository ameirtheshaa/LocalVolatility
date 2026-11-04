# Source inventory — Dupire signed-memory study

Recorded 2026-11-04 at the start of the study. Everything below is verifiable from the
Git repository alone.

## Repository identity

| Item | Value |
|---|---|
| Checkout | `https://github.com/ameirtheshaa/LocalVolatility` (public) |
| Starting commit (`main` = `origin/main` = session HEAD) | `9a679cbfda11d272a6b4e88f3f213d9848a8315f` (Merge pull request #7, `claude/lv-followups-cleanup`) |
| Working branch | `claude/brave-brahmagupta-twxtst` (pre-created by the session, identical to `main` at start) |
| Survey reference commit named in the brief | `f539f41f989220b4a2f17982d1f6c2536e566434` — **not present in this repository** (`git cat-file` fails; `git fetch origin <sha>` answers "not our ref"). The brief names the repository `ameirtheshaa/LocalVolatility-experiments` (private; listed by the session's repo listing but not attached to this session). The session was started on `ameirtheshaa/LocalVolatility`, whose `main` contains every source file the brief names. Changes since the reference commit therefore cannot be enumerated here; the per-file identities below are the substitute. |
| Repository instructions | No `CLAUDE.md`, `AGENTS.md` or `.claude/` directory. `Synthetic_Data_Tensorflow_Advanced/pytest.ini` deselects `slow` tests by default. |

## Source files read (commit of last change, sha256 prefix at the starting commit)

| File | Last commit | sha256[:16] | Role |
|---|---|---|---|
| `Synthetic_Data_Tensorflow_Advanced/mz_spectral/mz_decomposition.py` | a6cc384 (2026-06-15) | 466fdb95748f7dc1 | Fourier low-pass projector `P_R`, Galerkin blocks `L_RR, L_RU, L_UR, L_UU` |
| `Synthetic_Data_Tensorflow_Advanced/mz_spectral/quasi_linear.py` | f966d04 (2026-09-24) | e541b041ab0a6bdf | QL closure `+ν P_R D2 φ_R`, `fit_nu_ql` (ridge LS on central-difference residuals, `nonneg=True` default, clip `5e-4`) |
| `Synthetic_Data_Tensorflow_Advanced/mz_spectral/validation.py` | 54d7de9 (2026-09-24) | 6d2c3d3319f701c4 | Backward-Euler RRR integrator, IMEX RRR+QL integrator, payoff IC, Breeden–Litzenberger density (`max(f,0)` then renormalise), BS truth |
| `Synthetic_Data_Tensorflow_Advanced/mz_spectral/fourier_dupire.py` | a6cc384 | 97a7a59c07a67aa3 | Grid `(t̃,k̃)`, `η̃=(T_max/2)σ²`, `D2` (central interior, one-sided 2nd-order at both ends), `L = diag(η̃ k̃²) D2` |
| `Synthetic_Data_Tensorflow_Advanced/docs/MZ_QL_NONNEG_FIX_REPORT.md` | a6cc384 | 45882050ff393ace | The report under test (validates ν≥0 on constant-σ BS; leaves smile/noisy open) |
| `Synthetic_Data_Tensorflow_Advanced/examples/run_mz_derisk.py` | 54d7de9 | a0e710abb6abfa21 | Steps 0a/0b/1 driver; Step 0b is the ν-sign experiment |
| `Synthetic_Data_Tensorflow_Advanced/analytical_solutions.py` | c4adea5 (2025-11-09) | 5a07c411599c25d2 | `black_scholes_call`, `lognormal_density` |
| `Synthetic_Data_Tensorflow_Advanced/config.py` | 7144912 (2026-09-24) | 7342318d6298f384 | Defaults: `S0=1000, r=0.04, K_min=500, K_max=3000, T_min=0.3, T_max=1.5`; `VolatilityConfig.dupire_exact`: `σ=0.3+y e^{-y}, y=(t+0.1)√(x+0.1)` |

Also read: `mz_spectral/mz_synthetic_setup.py`, `mz_spectral/data_driven.py`, `mz_spectral/truncation_study.py` (the other `fit_nu_ql` callers), `mz_spectral/__init__.py`, `tests/*.py`, `docs/MZ_DUPIRE_SYNTHESIS.md` §8 (headings only, to avoid re-doing the P9 tail-energy work).

## Coordinates and conventions actually used by the source

* `t̃ = T/T_max`, `k̃ = K e^{-rT}/K_max`, `φ̃ = C/S0`. Dupire's PDE with zero dividends becomes
  `∂φ̃/∂t̃ = η̃(t̃,k̃) k̃² ∂²φ̃/∂k̃²`, `η̃ = (T_max/2) σ²(T,K)`; the `-rK ∂_K C` drift is absorbed by the discounted strike.
* Initial condition at `t̃=0`: `φ̃(0,k̃) = max(0, 1 - K_max k̃/S0)` (`payoff_phi_tilde`).
* Density: `f(K) = e^{rT} S0 ∂²φ̃/∂K²`, `∂²/∂K² = e^{-2rT}/K_max² ∂²/∂k̃²`.
* Grid: `n_t` maturities uniform in `[T_min, T_max]` (so `t̃ ∈ [0.2, 1]`); `n_k` uniform `k̃` points covering `[e^{-rT_max}K_min/K_max, e^{-rT_min}]`.
* Projector: Hermitian low-pass mask on the unitary FFT of the *raw, non-periodic* `φ̃` row; `keep` positive frequencies retained (`dim R = 2·keep+1`).
* `D2`: second-order central stencil in the interior, `(2,-5,4,-1)/dk²` one-sided rows at both ends (no boundary data; the boundary values evolve by extrapolation).

## Observations made during the inventory (numerical probes, constant σ=0.3, `n_t=24`, `n_k=256`)

1. **Time-origin offset in the existing drivers.** `integrate_rrr_only` / `integrate_rrr_ql` / `mz_synthetic_setup` place the `T=0` payoff at the first grid time `T_min=0.3` and compare with truth at `T`. The full semidiscrete solve with the payoff at `t̃=0` matches Black–Scholes to `3.2e-5` interior RMS; with the payoff placed at `t̃=0.2` the error is `1.0e-2`. This offset is common to RRR and RRR+QL in the existing report, so it does not change their *relative* ranking, but it inflates every absolute error in the report.
2. **Projection artifact dominates.** The resolved-only (RRR) resolved-interior error is `8.6e-2 / 9.4e-2 / 9.8e-2` at `keep=16/32/64` (exact-in-time integration, correct timing), while the projection error `‖P_R φ - φ‖` is only `7.8e-3 / 3.9e-3 / 2.2e-3`. The unresolved content `‖P_U φ‖` is `0.47 / 0.33 / 0.21` because `φ̃` is non-periodic (`φ̃(k_lo) ≈ 0.85`, `φ̃(k_hi) ≈ 0`); subtracting the linear trend between the endpoints reduces it 40–140×.
3. **Unstable orthogonal dynamics.** With the production one-sided stencil, `max Re eig(L_UU) = +196 / +1073 / +4975` at `keep=16/32/64` (the full operator has `max Re eig(L) = +0.405`). The exact memory kernel `L_RU e^{τ L_UU} L_UR` is therefore exponentially divergent in this split; only its action on the actual trajectory, `L_RU φ_U(t) = P_R L P_U φ(t)`, is finite.
4. **Sign of the fitted coefficient.** The least-squares coefficient of the *operator-computed* memory term on the QL regressor is `ν* = -2.9e-2 / -3.1e-2 / -3.3e-2` (negative, 60× beyond the `5e-4` clip). The repository's central-difference fit on BS truth with `nonneg=False` returns `-5e-4` (pinned at the clip), reproducing the report's "before" column.
5. **Well-posed variants exist.** A symmetric/boundary-consistent stencil gives `max Re eig(L_UU) ≈ 0` and RRR errors `5.0e-3 / 2.1e-3 / 8.0e-4`; the weighted eigenbasis projector has `L_RU = 0` exactly for time-independent `L` (no memory at all).
6. **Smile case with the production stencil.** The full solve of `VolatilityConfig.dupire_exact()` on the production grid reaches `min φ̃ = -0.156` at `T_max` (negative scaled call prices), to be diagnosed in Gate 2 before any closure is scored.

These observations fix the plan in `FROZEN_PLAN.md`.
