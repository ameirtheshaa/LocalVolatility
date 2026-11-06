# `scc` API contract v1 (binding for every module; see PLAN.md for the science)

Package `research/sine_closure/scc`. Pure NumPy/SciPy, float64, no global mutable state except
on-disk caches. The earlier study is imported read-only as `dsm` (`scc/__init__.py` inserts
`research/dupire_signed_memory` into `sys.path`); never edit anything under `dsm`. Every public
function has a docstring stating the formula it implements. Tests live in
`research/sine_closure/tests` (pytest, `conftest.py` redirects `SCC_OUTPUT_ROOT` to a temp dir).
All arrays are C-ordered `np.ndarray`; "rows" means time along axis 0.

Shared conventions:
* `grid` is `dsm.operators.Grid` (fields `k, dk, T_out, t_out, S0, r, T_max, K_max, K_min, T_min`,
  properties `n_k`, `K_row(T)`, `payoff()`). Build with `scc.reference.make_grid(n)`.
* `fg` is `dsm.reference.FineGrid` (`t (N+1,), h, i0, out_idx (24,)`), from `scc.reference.fine_grid(grid, sub)`.
* Scaled time `t` means `t̃`; physical maturity is `T = t * grid.T_max`.
* Interior index range is `1:-1` of the full grid; `n_int = n - 2`.

## scc/__init__.py (exists)
`STUDY_ROOT` (= research/sine_closure), `DSM_ROOT`, `REPO_ROOT`, `OUTPUT_ROOT` (env `SCC_OUTPUT_ROOT`
or `STUDY_ROOT`), `CACHE_ROOT` (env `SCC_CACHE_ROOT` or `OUTPUT_ROOT/results/cache`), `RESULTS`.

## scc/conventions.py
* `dst_basis(n, m) -> (n, m)`: `V[i, j-1] = sqrt(2/(n-1)) * sin(pi*j*i/(n-1))`, `i=0..n-1`, `j=1..m`;
  rows 0 and n-1 set to exactly 0.0. Requires `1 <= m <= n-2`.
* `dst_interior(n, m) -> (n-2, m)` = `dst_basis(n, m)[1:-1]`.
* `dst_complement_interior(n, m) -> (n-2, n-2-m)`: DST columns `j = m+1..n-2` (same formula, interior rows).
* `sine_eigenvalues(n, dk, j) -> ndarray`: `-(4/dk**2) * sin(pi*j/(2*(n-1)))**2` for integer array `j`.
* `lift_rows(phi_rows, k) -> ell_rows`: affine interpolant of each row's endpoints
  (`dsm.operators.linear_trend` semantics); accepts 1-D or 2-D.
* `detrend_rows(phi_rows, k) -> psi_rows` = `phi_rows - lift_rows(phi_rows, k)`.
* `payoff_state(grid) -> (phi0 (n,), ell (n,), psi0 (n,))`.
* `class InteriorSystem` (dataclass): `grid`, `case`, `n`, `k_int (n-2,)`, `D2_int (n-2,n-2)`,
  `D2_bnd (n-2, 2)` (columns 0 and n-1 of the interior rows of `dsm.operators.d2_repo(n, dk)`;
  identical for `d2_linearity`), `time_dependent: bool`; methods `d(t) -> (n-2,)` =
  `dsm.operators.eta_row(case.sigma_fn, grid, t*grid.T_max)[1:-1] * k_int**2`,
  `L_int(t) -> (n-2,n-2)` = `d(t)[:, None] * D2_int`, and `d_full(t) -> (n,)` (incl. endpoints).
  Constructor `interior_system(case, grid)`. For time-independent cases `d(t)` may cache its one value.
* `blocks(sys, t, V_int, U_int) -> dict(A, B, C, D)`: dense blocks of `W^T L_int(t) W`, `W=[V U]`.
* `A_structured(sys, t, V_int, lam_V) -> (m,m)` = `(V_int.T @ (d[:,None]*V_int)) * lam_V[None,:]`
  (equals `blocks(...)['A']` to 1e-12 relative; tested).
* `S_matrix(sys, t, V_int) -> (m,m)` = `V_int.T @ (d[:,None]*V_int)` (symmetric positive definite).
* `forcing_g(sys, t, ell (n,), ell_dot=None, b_int=None) -> (n-2,)` =
  `d(t) * (D2_int @ ell[1:-1] + D2_bnd @ ell[[0,-1]]) + b_int - ell_dot[1:-1]` (missing terms = 0).
* `memory_target(sys, t, V_int, U_int, psi_int_rows (N, n-2)) -> (N, m)`: `M = B(t) z`, `z = psi_int @ U`;
  implemented as `((psi_int - (psi_int @ V) @ V.T) * d[None,:]) ...` — i.e. `M = ((psi_int @ U) @ (U.T * d[None,:].T ...))`;
  whichever form is used, the test checks it equals `(d[:,None]*(D2_int @ (psi_int - (psi_int@V)@V.T).T)).T @ V` to 1e-12.
  Note `L_int = diag(d) D2_int`, so `V^T L_int w = ((d * (D2_int @ w)) @ V)`.
* `resolved_blocks_timewise(sys, V_int, U_int, times) -> list of dict` convenience (time-dependent or one entry).
* `expm_h(M, h) -> expm(h*M)` and `phi1_step(M, h, g) -> (E, v)` with `(e^{hM}, h φ1(hM) g)` as `dsm.mz.phi1_step`.

## scc/cases.py
* `@dataclass Case(id: str, family: str, params: dict, split: str, time_dependent: bool, sigma_fn, generator: str)`,
  `config_hash()` = sha256[:16] of sorted JSON of `{id, family, params, time_dependent, generator}`.
  `sigma_fn(T: float, x: ndarray) -> ndarray` (physical maturity T, moneyness x = K/S0), broadcastable.
* Splits: `'train' | 'val' | 'test_interp' | 'test_ood' | 'dev_regression'`.
* `family_C(), family_S(), family_D(), family_M()` per PLAN.md §4 (exact seeds/ranges; rng draw order =
  column order in the table; `round(x, 6)` stored in `params`; the stored rounded values are what `sigma_fn` uses).
  `gate5_regression()` returns the three Gate 5 cases built from `dsm.cases` formulas (ids unchanged,
  split `dev_regression`, generator `'gate5'`): `C_sig0.30`, `S_seed9`, `F_fresh_base0.28_t0.2_x0.05`.
* `all_cases()`, `cases_by_split(split)`, `get_case(case_id)`, `split_counts()`.
* `sigma_checks(case, grid, n_T=49) -> dict(sigma_min, sigma_max, clip_active_fraction, max_rel_jump_k, max_rel_jump_T)`
  on the grid × `linspace(0, T_max, n_T)` (clip-active = unclipped formula outside [0.05, 2]).
* `manifest(grid=None) -> dict` with `cases: [{id, family, params, split, generator, config_hash, time_dependent, checks?}]`,
  `counts`, `sha256` over the sorted `(id, family, params, split, generator)` JSON; `write_manifest(path, grid=None)`.

## scc/identity.py
* `code_identity() -> str` sha256[:16] over `scc/{conventions,cases,reference,targets,closures,oracles,metrics}.py`
  (name + bytes; missing files hashed as 'missing'). Env override `SCC_CODE_IDENTITY_OVERRIDE` (tests only).
* `make_identity(**fields) -> dict` = `{schema: 1, code, basis: 'dst1_v1', lift: 'affine_endpoints_v1', operator: 'OP-B', ...fields}`;
  `identity_hash(ident)` sha256[:16] of sorted JSON (default=str).
* npz cache: `cache_path(kind, ident)` → `CACHE_ROOT/<kind>/<identity_hash>.npz`; `cache_store(kind, ident, **arrays)`
  writes `.npz.tmp` + sidecar `.json` (identity, created_utc, numpy/scipy versions, array shapes) then `os.replace`;
  `cache_load(kind, ident) -> dict | None` (None when missing, unreadable, sidecar identity hash mismatch, or shapes mismatch).
* `write_json(path, obj)` (atomic, numpy-safe, sort_keys, indent=1) and `read_json(path)`; `sha256_file(path)`;
  `env_identity()`; `git_head()`; `source_hashes()` over `scc/*.py`, `scripts/*.py` and the `dsm` numerical modules.
* `class RunManifest(stage, planned)` with `start/done/failed/finish` and `previous_interrupted` (same semantics as
  `dsm.checkpoint.RunManifest`, file `RESULTS/<stage>/RUN_MANIFEST.json`).
* `result_status(path, ident) -> (status, detail)` and `supersede(path)` as in `dsm.checkpoint`.
* Leakage guard: `assert_phase_allows(split)`: raises `RuntimeError` if `os.environ.get('SCC_PHASE') == 'fit'` and
  `split.startswith('test')`.

## scc/reference.py
* `make_grid(n) -> Grid` (`dsm.operators.build_grid(n_k=n)`); `fine_grid(grid, sub)`.
* `nested_index(n_fine, n_coarse) -> ndarray`: asserts `(n_fine-1) % (n_coarse-1) == 0` and returns `arange(0, n_fine, step)`;
  `restrict_rows(rows, idx) -> rows[:, idx]`.
* `coefficient_full(case, grid, t) -> (n,)` = `eta_row(...)*k**2` (incl. endpoints).
* `L_full_dense(case, grid, t) -> (n,n)` OP-B (`dsm.operators.build_L(eta, grid, d2_linearity)`).
* `L_full_sparse(case, grid, t) -> scipy.sparse.csr_matrix` same matrix.
* `solve_dense(case, grid, fg, phi0=None) -> (N+1, n)`: identical to `dsm.reference.solve_full(case, grid, 'B', fg, phi0)`
  (midpoint `expm`, one `expm` when time-independent).
* `solve_sparse(case, grid, fg, phi0=None, tol=1e-13) -> (N+1, n)`: same stepping with
  `scipy.sparse.linalg.expm_multiply(h*L_mid, phi)`; for time-independent cases build the operator once.
* `solve_banded_cn(case, grid, fg, substeps: int, rannacher: int = 2) -> (N+1, n)`: Crank–Nicolson with `substeps`
  substeps per fine step, the first `rannacher` substeps of the whole run replaced by two half-size backward-Euler
  steps each; banded LU via `scipy.linalg.solve_banded` (tridiagonal, boundary rows identity); returns values at `fg.t`.
* `solver_check(case, grid, fg) -> dict(max_abs_diff_dense_sparse, max_abs_phi)` on a given case.
* `bs_rows(grid, sigma, t_rows)` = `dsm.reference.bs_truth`.
* `reference_identity(case, n, sub, solver, phi0_tag='payoff', **extra) -> dict` with fields
  `kind='reference', case=config_hash, case_id, split, n, sub, solver, K_min, K_max, phi0=phi0_tag`.
* `get_reference(case, n, sub, solver='dense', phi0=None, phi0_tag='payoff', force=False) -> dict(phi, t, h, i0, out_idx, identity, from_cache: bool, seconds: float)`
  cached under kind `reference`; calls `identity.assert_phase_allows(case.split)` first; `phi0` given ⇒ `phi0_tag` must be given.

## scc/targets.py
* `projected_state(psi_int_rows, V_int) -> (N, m)` = `psi_int_rows @ V_int`; `unresolved_state(psi_int_rows, U_int) -> (N, n-2-m)`.
* `build_targets(case, ref: dict, grid, m) -> dict(c (N+1,m), z (N+1,n-2-m), M (N+1,m), t (N+1,), rms_M: float, ell (n,), psi0 (n,))`:
  `psi = detrend_rows(ref['phi'], grid.k)`, interior slice, `M` via `conventions.memory_target` row by row for
  time-dependent cases (one call per time), once for autonomous cases.
* `trajectory_weights(rms_by_case: dict[str,float], n_rows_by_case: dict[str,int]) -> dict[str,float]` per PLAN.md §7.
* `d_extremes(cases, grid, times) -> (d_min, d_max)` over cases × times × interior points.

## scc/closures.py
* `@dataclass ModelSpec(name: str, m: int, closure: str, params: dict, n_h: int = 0, hidden_init: str = 'H0', meta: dict = field(default_factory=dict))`
  closure ∈ {'RRR','QL','LM','LR'}; params: QL `{'nu': float}`; LM `{'H': (m,m), 'b': (m,) | None}`;
  LR `{'F': (n_h,n_h) diagonal, 'G': (n_h,m), 'H': (m,m), 'R': (m,n_h), 'K_init': (n_h, m+1) | None}`.
  `total_state()` = m + n_h (+1 when LM has a bias, counted as the affine augmentation). `param_count() -> dict(fitted, fixed)`.
* `closed_matrix(A (m,m), spec, lam_V) -> square`: RRR `A`; QL `A + nu*diag(lam_V)`; LM `A + H` (with bias: augmented
  `[[A+H, b],[0, 0]]` on state `[c; 1]`); LR `[[A+H, R],[G, F]]`.
* `rollout(spec, A_mid, t, c0, lam_V, h0=None, blowup=10.0, ref_norm_max=None, time_dependent=True) -> dict(c (N+1,m), h (N+1,n_h), unstable, n_last, seconds, n_expm)`:
  `A_mid` is a callable `step_index -> A(t_mid)` (or any object with `__call__`); when `time_dependent` is False one
  exponential is computed and reused; states after a blow-up are NaN; uses `scipy.linalg.expm`.
* Fitting (all weighted; `targets_list` = list of `build_targets` outputs, `weights` = dict case_id → w; inputs are `c`, outputs `M`):
  `fit_nu(targets_list, weights, S_min_eig: float) -> dict(nu_unclipped, num, den, nu_adm, nu_plus, nu_pm)`;
  `fit_LM(targets_list, weights, lam_rel, intercept=False) -> (H, b, info)`;
  `rate_bank(n_h, bank: str, r1: float, ratio: float) -> alphas (n_h,)` with bank 'N' `[0.5 r1, 16 r1 ratio]`, 'W' `[0.25 r1, 64 r1 ratio]`, log-spaced;
  `filtered_states(c_rows, t, alphas) -> q (N+1, n_h, m)` exact exponential filtering of the piecewise-linear input
  (`q[n+1,k,:] = e^{-a h} q[n,k,:] + c_n I0 + (c_{n+1}-c_n) I1/h`, `I0 = (1-e^{-ah})/a`, `I1 = h/a - (1-e^{-ah})/a**2`);
  `fit_LR(targets_list, weights, n_h, alphas, lam_rel, G0=None, als_iters=30, tol=1e-6, seed=0) -> (spec_LR0, spec_LR, info)`;
  `fit_initializer(spec, targets_list, weights, i0, lam_rel) -> K (n_h, m+1)` from teacher-forced `h(t0)`.
  `ridge_rel(X, Y, W_rows, lam_rel)` weighted ridge with `lam = lam_rel * trace(XᵀWX)/X.shape[1]`.
* Stability: `abscissae(Mc) -> (spectral, numerical)`; `stability_report(spec, A_list_out, lam_V) -> dict(spectral_abscissa_max, numerical_abscissa_max, per_time lists)`;
  `perturbation_growth(spec, A_mid, t, c0, lam_V, eps=1e-6, seed=0, **kw) -> dict(max_growth_ratio, final_growth_ratio)`;
  `propagator_growth(spec, A_mid, t, lam_V, time_dependent) -> float` (max over t of ‖Φ(t,t0)‖₂, product of step matrices);
  `long_horizon(spec, A_end, lam_V, t_end=3.0, h) -> dict(bounded: bool, max_norm_ratio)`.
* Warm-up / initialisers: `hidden_init_state(spec, protocol, c0, ...)`: 'H0'/'W0' zeros; 'WI' `K @ [c0; 1]`;
  'WU' is implemented by the scoring script (rollout from payoff then reset c).

## scc/oracles.py
* `projected_reconstruction(psi_int_rows, V_int, ell) -> (N, n)`: `ell + [0, V c, 0]`.
* `teacher_forced_rollout(A_mid, M_ref, t, c0, time_dependent) -> c (N+1, m)`: `c_{n+1} = E c_n + hφ1(hA_mid)(M_n+M_{n+1})/2`.
* `block_identity_autonomous(sys, V_int, U_int, psi_int_rows, t, i0, check_times=None) -> dict`:
  at each check time `τ = t_j - t_i0` compute `[[D, C'],[0, L_rot]]` exponential with `L_rot = W^T L_int W`,
  `C' = [C, 0]`; `z(τ) = e^{τD} z0 + (top-right block) @ x0`; residuals `max_rel_err_z` vs `z_ref`, and
  `max_rel_err_cdot` of `A c + B z` vs `[L_rot x_ref]_c`; also `norm_ic_term`, `norm_memory_term` at the last time.
* `block_identity_timeordered(sys, V_int, U_int, psi_int_rows, t, i0, order='forward') -> dict(max_rel_err_z, final_rel_err_z)`:
  propagate `z` with the φ1 midpoint rule using `D(t_mid)`, forcing `C(t_mid) (c_n + c_{n+1})/2`; `order='reversed'`
  uses the step operators in reversed time order (control that must fail).
* `b_zero_control(n, m, dk, d_const) -> dict`: synthetic operator `L = d_const * D2_int` (B = 0 exactly in the sine
  basis because `D2_int` is diagonal in it); asserts `max|B| < 1e-12·max|A|`, `max|M| = 0` for a random `psi`, and TF == RRR.
* `decomposition(phi_ref_rows, ell, V (n,m), c_model_rows) -> dict(proj_rows (N,n), dyn_rows (N,n), orthogonality_defect)`:
  `phi_ref - (ell + V c_model) = (I-P)psi_ref + V(c_ref - c_model)`; defect = `max |<proj, dyn>| / (‖proj‖‖dyn‖)` over rows (full grid).

## scc/metrics.py
* `@dataclass EvalContext(grid, D2_repo, D2_lin, interior (n_t,n) bool, atm (n_t,n), wing (n_t,n), K_rows (n_t,n), T_rows (n_t,))`,
  `make_context(grid, T_rows=None)` (default `grid.T_out`; the early panel passes its own `T_rows`).
* `price_metrics(phi_rows, ref_rows, ctx, bs_rows=None) -> dict` keys exactly:
  `price_interior_rms, price_full_rms, price_max_abs, price_atm_rms, price_wing_rms, price_per_maturity_rms (list),
   price_physical_interior_rms, price_interior_rms_vs_bs (only with bs_rows)`.
* `resolved_error(c_rows, c_ref_rows, V, interior) -> dict(resolved_interior_rms, resolved_full_rms, resolved_coeff_rms)`
  (`V` full (n,m); error field `(c-c_ref) @ V.T`).
* `density_rows(phi_rows, ctx, method='d2_repo') -> f_rows`, methods `'d2_repo' | 'd2_lin' | 'fd4'` (fourth-order central,
  one-sided fourth-order at the two points next to each end, repo rows at the ends);
  `density_rows_spectral(c_rows, V_full, lam_V, ctx)` = `e^{rT} S0 e^{-2rT}/K_max² · (c*lam) @ V.T`.
  Formula: `f = exp(rT) S0 d2k · (exp(-rT)/K_max)²` with `d2k = ∂²φ̃/∂k̃²`.
* `density_metrics(f, f_ref, ctx, post_initial_from=0) -> dict` keys exactly:
  `density_l2_interior_mean, density_l2_interior_max, density_l1_interior_mean, density_l2_full_mean, density_max_abs_err,
   density_q99_abs_err, negative_mass (list per row), negative_mass_max_post_initial, negative_mass_final,
   signed_mass (list), mass_final, mass_ref_final, first_moment_final, first_moment_ref_final, tail_mass_outside_interior_final,
   tail_mass_ref_outside_interior_final` (trapezoid in K along each row; L2 = sqrt(∫ (f-f_ref)² dK) on the band).
* `arbitrage_metrics(phi_rows, phi_t_rows, ctx) -> dict`: for each of `positivity, monotonicity, butterfly, calendar`
  and each of `raw` (guard `1e-12 × row max abs`) and `repo` (`1e-3 × row max abs`), fractions over `interior` and `full`:
  keys `f'{check}_{rule}_{region}_frac_mean'` and `..._frac_max` (over rows). Derivatives: `np.gradient(row, dk)` for the
  k-derivative, `D2_repo` for curvature, `phi_t_rows` given.
* `identifiable_mask(ref_rows, ctx, rel_tol=1e-3) -> (n_t, n) bool` = interior ∧ `D2_repo ref > rel_tol·rowmax|D2_repo ref|`.
* `inverse_vol_metrics(phi_rows, phi_t_rows, ctx, ident_mask, sigma_true_rows, rel_tol_panel=(1e-2, 1e-3, 1e-4)) -> dict`
  with, for each `rel_tol` as string key (`'1e-03'` style), a sub-dict
  `undefined_frac_interior, negative_or_nonfinite_frac_interior, invalid_frac_full, coverage, sigma_relerr_median, sigma_relerr_rms, n_identifiable`
  and the top-level copies for `1e-3` prefixed `invvol_` (e.g. `invvol_invalid_frac_full`).
  `σ̂² = 2 φ_t / (T_max k̃² φ_kk)`; undefined ⇔ `φ_kk ≤ rel_tol·rowmax`; invalid ⇔ undefined ∨ nonfinite ∨ σ̂² ≤ 0.
* `sigma_true_rows(case, grid, T_rows)`; `time_derivative_rows_fd(phi_fine, idx, h)` (= `dsm.metrics.time_derivative_rows`).
* `rhs_time_derivative_reduced(c_rows, Mhat_rows, A_list, V_full) -> phi_t_rows` = `(A c + M̂) @ V.T`;
  `rhs_time_derivative_full(phi_rows, L_list) -> phi_t_rows`.

## scc/cost.py
* `pin_threads()` context manager (env vars + `threadpoolctl.threadpool_limits(1)` when importable).
* `timing_env() -> dict(platform, machine, cpu_count, python, numpy, scipy, blas (from numpy.show_config dict when available), threads)`.
* `time_callable(fn, repeats=7, warmup=2) -> dict(median, iqr, min, max, samples (list), cold)` where `cold` is the very first call.
