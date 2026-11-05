# Result files and naming (binding for scripts s0–s7)

All paths are under `OUTPUT_ROOT` (= `research/sine_closure` unless `SCC_OUTPUT_ROOT` is set);
`results/cache/` (or `SCC_CACHE_ROOT`) holds regenerable `.npz` references and is never committed.
Every JSON result written by a stage carries `"status": "done" | "failed"`, `"identity"` (from
`scc.identity.make_identity`), `"identity_hash"`, `"stage"`, `"wall_seconds"` and, on failure,
`"traceback"`. Stages are idempotent: a result is reused only when `identity.result_status` says
`done`; stale/corrupted/failed files are superseded (kept as `*.superseded.<utc>.json`).
Scripts accept `--resume` (reuse done results), `--max-cases N` (deliberate early stop) and
`--cases ID [ID ...]` (subset); jobs longer than a minute are launched on the GPU-laptop CPU lane
by the coordinator — scripts must not assume the Mac.

## Grid levels and reference keys
Level names: `n256_s16` (model grid, primary), `n511_s16`, `n1021_s16`, `n256_s32`, `n256_s64`.
Solver per level: dense for `n256_*s16`, `n511_s16`, `n256_s32`; sparse for `n1021_s16`, `n256_s64`.
`scc.reference.get_reference(case, n, sub, solver)` is the only way to obtain a reference.

## Model keys (used in every table)
`<closure>_m<m>[_h<n_h>][_<variant>]__<protocol>[_<hidden>]` where
* closure ∈ `RRR, QLp, QLpm, QLs, LM, LMb, LR0, LR, PROJ, BLOCK, TF, FULL256, FULL256sp, FULL256cn, FULL511, FULL1021`
  (`QLp` = ν⁺, `QLpm` = ν±, `QLs` = unclipped ν*, `LMb` = LM with intercept);
* `_h<n_h>` only for LR0/LR; `_<variant>` reserved (e.g. bank id) — the locked configuration is implied, so omit it in tables;
* protocol ∈ `P0` (payoff start), `P1` (warm start at t̃0 = 0.2);
* hidden ∈ `W0, WI, WU` for LR0/LR under `P1` only (under `P0` the hidden state starts at 0, no suffix).
Examples: `RRR_m17__P0`, `LR_m9_h16__P1_WI`, `FULL256__P0`, `PROJ_m9__P1`.

## Stage outputs
* `results/manifests/CASES.json` (s2; `scc.cases.write_manifest` with sigma checks on the 256 grid),
  `ENV.json` (python/numpy/scipy/platform/git head/source hashes), `TIMING_ENV.json` (s6).
* `results/verify/verify.json` (s0): every check of PLAN.md §11 as `{name: {value(s), tolerance, pass}}`
  plus `roundoff_guard.json` (γ and the paired differences it came from) and `solver_check.json`.
* `results/gate5_reproduction/` (s1): the unchanged Gate 5 outputs + `results/gate5/comparison.json`.
* `results/references/index.json` (s2): `{case_id: {level: {identity_hash, seconds, from_cache, npz_path}}}`.
* `results/diagnose/<case_id>.json` (s3) with, per `m` in `{9, 13, 17, 25, 33, 65}` and protocol:
  `PROJ`, `RRR`, `TF` metric dicts (price vs same-grid and vs refined, resolved error, density, negative mass),
  `BLOCK` identity errors, `decomposition` norms (interior RMS of the projection part, of the dynamics part,
  orthogonality defect), density-definition comparison (`d2_repo` vs `d2_lin` vs `spectral` vs `fd4`: pairwise interior L2),
  and `results/diagnose/summary.json` with the §10 stopping-rule evaluation per `m` (`representation_limited: bool`,
  the fractions of development cases failing each condition, the numbers behind them).
* `results/uncertainty/<case_id>.json` (s3 for development cases, s5 for test cases): `u_q^grid`, `u_q^step`, `u_q`
  for every quantity named in PLAN.md §3, observed convergence orders where the 1021/64 levels exist.
* `models/m<m>/<closure>[_h<n_h>]_<config>.npz` + sidecar `.json` (s4): weights, config (λ, bank, n_h), training
  diagnostics (objective per iteration, condition numbers, in-sample residuals per case), parameter counts, offline seconds;
  `results/fit/validation_m<m>.json`: every configuration × validation case × protocol with `e_dyn` and stability;
  `results/fit/selection.json`: the selected configuration per (m, closure) and why; `models/LOCK.json`: sha256 of every
  selected weight file, the code identity, the manifest sha256, the UTC time — written before s5 runs.
* `results/score/<case_id>.json` (s5): `{case, family, split, params, uncertainty: {...}, models: {model_key: row}}`,
  each row `{metrics: {price_* vs refined, price_same_* (same-grid), resolved_*, density_*, <arbitrage keys>, invvol_* and the
  threshold panel, stability: {...}, early_panel: {t: {...}} (P0 only), phi_t_check: {rms_rhs_vs_fd16, rms_fd16_vs_fd32}},
  unstable, n_last, total_state, param_count, online_seconds: {init, assembly_and_expm, stepping, reconstruction, density, inverse_vol, total},
  n_expm}`. Failed/unstable models keep their row with `unstable: true` and whatever metrics are finite.
* `results/cost/cost.json` (s6): per model key (one row per case family representative) the §9 timing dict
  (cold, warm median/IQR/min, components), the full-solver candidates, offline costs, break-even counts, and the Pareto data.
* `results/tables/*.md`, `results/figures/*.png`, `results/decomposition/*.json|md`, `results/panels/*.json|md` (s7), and
  `results/decision.json` (s7): the §10 classification per (m, closure, hidden protocol) with every exception listed.
