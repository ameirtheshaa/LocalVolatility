# Decision classification (PLAN section 10)

Classification per (m, closure, hidden protocol) on the `test_interp` cases under both start protocols. `test_ood` is reported separately and enters only the selected-regimes wording. Rules as implemented:

* **beyond uncertainty**: |dq| > 2 u_q,case (improvement direction: q_Y - q_X > 2 u)
* **noninferior**: q_X <= q_Y + u_q,case
* **negative mass (strict)**: negmass_X <= negmass_Y + gamma, gamma from results/verify/roundoff_guard.json; quantity = max over post-initial rows
* **validity fractions**: frac_X <= frac_Y + max(u, 1/N_interior), N_interior = interior points per row x 24 (230 x 24 when not in the score file)
* **coverage**: cov_X >= cov_Y - max(u, 1/N_interior)
* **resolved_dynamics_improvement**: every test_interp case, both protocols: stable AND e_dyn(X) < e_dyn(RRR_m) - 2 u_dyn
* **useful_closure**: RDI AND refined price RMS(X) < RMS(RRR_m) - 2u on every case/protocol AND noninferior to RRR_m in density L2, negmass (strict), positivity/monotonicity/butterfly/calendar raw interior fractions, invvol invalid fraction (full) and coverage on every case/protocol
* **successful_compression**: useful_closure AND on every ELIGIBLE case/protocol cell (Addendum 2): price <= RRR33 + u_price, density <= RRR33 + u_dens, negmass <= RRR33 + gamma, invvol invalid <= RRR33 + max(u, 1/N) AND online warm median(X) < median(RRR33) - max(IQR_X, IQR_RRR33) (cost.json)
* **compression eligibility**: E_m(case, protocol) = PROJ_m price < RRR33 price + u AND RRR_m price - PROJ_m price > 2u; compression classes use only eligible cells, empty regime = not_testable_regime_empty
* **price_only_compression**: not successful_compression, but RDI AND price improves beyond 2u over RRR_m AND price <= RRR33 + u_price on every case/protocol
* **stateful_advantage**: LR/LR0 only: vs LM at same m (same protocol) e_dyn(LM) - e_dyn(LR) > 2 u_dyn on every case AND noninferior in price, density, negmass, arbitrage fractions, invvol invalid, coverage; AND the same two requirements against RRR at m + n_h
* **representation_limited**: results/diagnose/summary.json (PLAN section 10 stopping rule, evaluated on development data); only used when no positive class holds
* **inconclusive**: no positive class, and either (a) for e_dyn or price some cells have gain > 2u and others 0 < gain <= 2u, or (b) the sign of the price gain differs between the same-grid and refined truths in some cell
* **no_supported_benefit**: fallback: none of the above; reasons (gains within uncertainty / equal-state unclosed model at least as good / validity deteriorated) are listed

**Caveats (inputs missing, classification is provisional):**
* uncertainty u missing for 12 (case, quantity) pairs (treated as 0, fractions floored at 1/N_interior): RUN_MANIFEST:density_l2_interior_mean, RUN_MANIFEST:invvol_invalid_frac_full, RUN_MANIFEST:negative_mass_max_post_initial, RUN_MANIFEST:price_interior_rms, RUN_MANIFEST:u_dyn_m17, RUN_MANIFEST:u_dyn_m9, summary:density_l2_interior_mean, summary:invvol_invalid_frac_full, summary:negative_mass_max_post_initial, summary:price_interior_rms, summary:u_dyn_m17, summary:u_dyn_m9

gamma = 1.54e-12; test_interp cases: C_sig0.205, C_sig0.310, C_sig0.415, D2_seed220, D2_seed221, D2_seed222, D2_seed223, M_seed320, M_seed321, M_seed322, M_seed323, S2_seed120, S2_seed121, S2_seed122, S2_seed123, S2_seed124, S2_seed125; test_ood cases: C_sig0.120, C_sig0.550, D2_seed230, D2_seed231, M_seed330, M_seed331, S2_seed130, S2_seed131, S2_seed132, S2_seed133.

## Verdict per dimension

* m = 9: the best result is resolved-dynamics improvement (LR0_m9_h16/WI, LR0_m9_h16/WU, LR_m9_h16/WI, LR_m9_h16/WU), which is not a deployment benefit by itself (the useful-closure conditions fail). Compression not testable on this test set (eligible regime empty: PROJ_m already exceeds the 33-mode target or the headroom RRR_m - PROJ_m is within 2u). Output-oriented compensation flagged for: QLp_m9, QLpm_m9, QLs_m9, LM_m9, LMb_m9, LR0_m9_h16/W0, LR0_m9_h16/WI, LR0_m9_h16/WU, LR_m9_h16/W0, LR_m9_h16/WI, LR_m9_h16/WU. The unclosed 33-mode model remains the recommendation.
* m = 17: closure does not help (no entry reaches resolved-dynamics improvement on every held-out interpolation case). Compression not testable on this test set (eligible regime empty: PROJ_m already exceeds the 33-mode target or the headroom RRR_m - PROJ_m is within 2u). Output-oriented compensation flagged for: QLp_m17, QLpm_m17, QLs_m17, LM_m17, LMb_m17, LR0_m17_h8/W0, LR0_m17_h8/WI, LR0_m17_h8/WU, LR_m17_h8/W0, LR_m17_h8/WI, LR_m17_h8/WU. The unclosed 33-mode model remains the recommendation.

## Compression-eligibility regime (PLAN Addendum 2)

E_m(case, protocol) = PROJ_m price < RRR_m33 price + u_price AND RRR_m price - PROJ_m price > 2 u_price (refined price interior RMS, per protocol). Compression classes are evaluated only over eligible cells; an empty regime is reported as `not_testable_regime_empty`.

* m = 9: test_interp eligible cells 0 of 34; development data (P0): 0 eligible dev cases / 41 diagnosed
* m = 17: test_interp eligible cells 0 of 34; development data (P0): 2 eligible dev cases / 41 diagnosed

## Domain annotation (family C, results/verify/bs_vs_u_family_C.json)

* C_sig0.180: domain_limited_physical: false (full511_vs_bs 7.40e-07, u_grid_price 4.59e-06)
* C_sig0.220: domain_limited_physical: false (full511_vs_bs 6.95e-07, u_grid_price 4.13e-06)
* C_sig0.270: domain_limited_physical: false (full511_vs_bs 5.02e-06, u_grid_price 3.70e-06)
* C_sig0.330: domain_limited_physical: true (full511_vs_bs 5.93e-05, u_grid_price 3.34e-06)
* C_sig0.380: domain_limited_physical: true (full511_vs_bs 2.42e-04, u_grid_price 3.13e-06)
* C_sig0.440: domain_limited_physical: true (full511_vs_bs 8.27e-04, u_grid_price 2.96e-06)
* C_sig0.240: domain_limited_physical: false (full511_vs_bs 1.06e-06, u_grid_price 3.94e-06)
* C_sig0.360: domain_limited_physical: true (full511_vs_bs 1.45e-04, u_grid_price 3.20e-06)

## Output-oriented compensation (beats PROJ_m beyond u: same-grid price or refined density)

Reported as compensation, never as recovered memory: in the full-grid Euclidean norm c_ref is the best coefficient vector, so beating PROJ means the fitted model moved resolved coefficients away from c_ref to improve an output.

* **QLp_m9** compensation: true (20 cells)
  * D2_seed220 (test_interp) P0: same-grid price 1.91e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.17e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 1.70e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.02e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 1.92e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.88e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 1.66e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.73e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 1.86e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.52e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 1.63e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.36e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 1.91e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.05e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 1.72e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 6.92e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * C_sig0.120 (test_ood) P0: same-grid price 1.36e-03 vs PROJ 1.36e-03 (u 5.66e-06); density 1.28e-02 vs PROJ 1.29e-02 (u 3.17e-05)
  * C_sig0.120 (test_ood) P1: same-grid price 1.36e-03 vs PROJ 1.36e-03 (u 5.66e-06); density 1.28e-02 vs PROJ 1.29e-02 (u 3.17e-05)
  * C_sig0.550 (test_ood) P0: same-grid price 1.83e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.44e-04 vs PROJ 7.74e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 1.58e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.33e-04 vs PROJ 7.74e-04 (u 7.86e-07)
* **QLpm_m9** compensation: true (20 cells)
  * D2_seed220 (test_interp) P0: same-grid price 1.91e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.17e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 1.70e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.02e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 1.92e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.88e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 1.66e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.73e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 1.86e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.52e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 1.63e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.36e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 1.91e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.05e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 1.72e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 6.92e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * C_sig0.120 (test_ood) P0: same-grid price 1.36e-03 vs PROJ 1.36e-03 (u 5.66e-06); density 1.28e-02 vs PROJ 1.29e-02 (u 3.17e-05)
  * C_sig0.120 (test_ood) P1: same-grid price 1.36e-03 vs PROJ 1.36e-03 (u 5.66e-06); density 1.28e-02 vs PROJ 1.29e-02 (u 3.17e-05)
  * C_sig0.550 (test_ood) P0: same-grid price 1.83e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.44e-04 vs PROJ 7.74e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 1.58e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.33e-04 vs PROJ 7.74e-04 (u 7.86e-07)
* **QLs_m9** compensation: true (20 cells)
  * D2_seed220 (test_interp) P0: same-grid price 1.91e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.17e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 1.70e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.02e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 1.92e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.88e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 1.66e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.73e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 1.86e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.52e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 1.63e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.36e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 1.91e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.05e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 1.72e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 6.92e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * C_sig0.120 (test_ood) P0: same-grid price 1.36e-03 vs PROJ 1.36e-03 (u 5.66e-06); density 1.28e-02 vs PROJ 1.29e-02 (u 3.17e-05)
  * C_sig0.120 (test_ood) P1: same-grid price 1.36e-03 vs PROJ 1.36e-03 (u 5.66e-06); density 1.28e-02 vs PROJ 1.29e-02 (u 3.17e-05)
  * C_sig0.550 (test_ood) P0: same-grid price 1.83e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.44e-04 vs PROJ 7.74e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 1.58e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.33e-04 vs PROJ 7.74e-04 (u 7.86e-07)
* **LM_m9** compensation: true (11 cells)
  * D2_seed220 (test_interp) P0: same-grid price 1.15e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.20e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 1.12e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.03e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 1.20e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 8.06e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 1.15e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.90e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P1: same-grid price 1.11e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.71e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 1.13e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.11e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 1.12e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 6.96e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * C_sig0.550 (test_ood) P0: same-grid price 1.04e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.47e-04 vs PROJ 7.74e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 1.02e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.38e-04 vs PROJ 7.74e-04 (u 7.86e-07)
  * D2_seed231 (test_ood) P0: same-grid price 9.74e-05 vs PROJ 8.17e-05 (u 2.39e-06); density 5.55e-04 vs PROJ 5.69e-04 (u 3.41e-07)
  * D2_seed231 (test_ood) P1: same-grid price 8.98e-05 vs PROJ 8.17e-05 (u 2.39e-06); density 5.60e-04 vs PROJ 5.69e-04 (u 3.41e-07)
* **LMb_m9** compensation: true (7 cells)
  * D2_seed220 (test_interp) P0: same-grid price 1.15e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.34e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 1.12e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.17e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P1: same-grid price 1.16e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.98e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed223 (test_interp) P0: same-grid price 1.14e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.27e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 1.12e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.11e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * C_sig0.550 (test_ood) P0: same-grid price 1.05e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.52e-04 vs PROJ 7.74e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 1.02e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.43e-04 vs PROJ 7.74e-04 (u 7.86e-07)
* **LR0_m9_h16/W0** compensation: true (6 cells)
  * D2_seed220 (test_interp) P0: same-grid price 1.18e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.31e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 1.12e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.34e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed223 (test_interp) P0: same-grid price 1.16e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.25e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 1.10e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.22e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * M_seed320 (test_interp) P0: same-grid price 1.35e-04 vs PROJ 1.35e-04 (u 3.49e-06); density 9.71e-04 vs PROJ 9.80e-04 (u 3.82e-06)
  * C_sig0.550 (test_ood) P0: same-grid price 1.06e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.57e-04 vs PROJ 7.74e-04 (u 7.86e-07)
* **LR0_m9_h16/WI** compensation: true (10 cells)
  * C_sig0.415 (test_interp) P1: same-grid price 1.13e-04 vs PROJ 1.12e-04 (u 3.02e-06); density 7.93e-04 vs PROJ 7.97e-04 (u 1.83e-06)
  * D2_seed220 (test_interp) P0: same-grid price 1.18e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.31e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 1.15e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.10e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P1: same-grid price 1.18e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.98e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed223 (test_interp) P0: same-grid price 1.16e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.25e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 1.15e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.05e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * M_seed320 (test_interp) P0: same-grid price 1.35e-04 vs PROJ 1.35e-04 (u 3.49e-06); density 9.71e-04 vs PROJ 9.80e-04 (u 3.82e-06)
  * M_seed320 (test_interp) P1: same-grid price 1.35e-04 vs PROJ 1.35e-04 (u 3.49e-06); density 9.71e-04 vs PROJ 9.80e-04 (u 3.82e-06)
  * C_sig0.550 (test_ood) P0: same-grid price 1.06e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.57e-04 vs PROJ 7.74e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 1.03e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.38e-04 vs PROJ 7.74e-04 (u 7.86e-07)
* **LR0_m9_h16/WU** compensation: true (10 cells)
  * C_sig0.415 (test_interp) P1: same-grid price 1.13e-04 vs PROJ 1.12e-04 (u 3.02e-06); density 7.93e-04 vs PROJ 7.97e-04 (u 1.83e-06)
  * D2_seed220 (test_interp) P0: same-grid price 1.18e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.31e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 1.16e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.09e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P1: same-grid price 1.18e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.97e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed223 (test_interp) P0: same-grid price 1.16e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.25e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 1.16e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.05e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * M_seed320 (test_interp) P0: same-grid price 1.35e-04 vs PROJ 1.35e-04 (u 3.49e-06); density 9.71e-04 vs PROJ 9.80e-04 (u 3.82e-06)
  * M_seed320 (test_interp) P1: same-grid price 1.35e-04 vs PROJ 1.35e-04 (u 3.49e-06); density 9.71e-04 vs PROJ 9.80e-04 (u 3.82e-06)
  * C_sig0.550 (test_ood) P0: same-grid price 1.06e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.57e-04 vs PROJ 7.74e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 1.03e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.38e-04 vs PROJ 7.74e-04 (u 7.86e-07)
* **LR_m9_h16/W0** compensation: true (10 cells)
  * D2_seed220 (test_interp) P0: same-grid price 1.09e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.08e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 1.08e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.26e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 1.15e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.94e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 1.11e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.78e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 1.08e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.01e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 1.07e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.16e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * M_seed320 (test_interp) P0: same-grid price 1.35e-04 vs PROJ 1.35e-04 (u 3.49e-06); density 9.69e-04 vs PROJ 9.80e-04 (u 3.82e-06)
  * C_sig0.550 (test_ood) P0: same-grid price 1.03e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.50e-04 vs PROJ 7.74e-04 (u 7.86e-07)
  * D2_seed230 (test_ood) P0: same-grid price 9.70e-05 vs PROJ 9.02e-05 (u 2.52e-06); density 6.55e-04 vs PROJ 6.56e-04 (u 4.43e-07)
  * D2_seed231 (test_ood) P0: same-grid price 9.28e-05 vs PROJ 8.17e-05 (u 2.39e-06); density 5.53e-04 vs PROJ 5.69e-04 (u 3.41e-07)
* **LR_m9_h16/WI** compensation: true (15 cells)
  * D2_seed220 (test_interp) P0: same-grid price 1.09e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.08e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 1.10e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 6.95e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 1.15e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.94e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 1.12e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.80e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 1.11e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.78e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 1.09e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.65e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 1.08e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.01e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 1.10e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 6.91e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * M_seed320 (test_interp) P0: same-grid price 1.35e-04 vs PROJ 1.35e-04 (u 3.49e-06); density 9.69e-04 vs PROJ 9.80e-04 (u 3.82e-06)
  * M_seed320 (test_interp) P1: same-grid price 1.35e-04 vs PROJ 1.35e-04 (u 3.49e-06); density 9.72e-04 vs PROJ 9.80e-04 (u 3.82e-06)
  * C_sig0.550 (test_ood) P0: same-grid price 1.03e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.50e-04 vs PROJ 7.74e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 1.03e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.45e-04 vs PROJ 7.74e-04 (u 7.86e-07)
* **LR_m9_h16/WU** compensation: true (15 cells)
  * D2_seed220 (test_interp) P0: same-grid price 1.09e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 7.08e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 1.09e-04 vs PROJ 1.03e-04 (u 2.63e-06); density 6.95e-04 vs PROJ 7.42e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 1.15e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.94e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 1.12e-04 vs PROJ 1.10e-04 (u 2.72e-06); density 7.81e-04 vs PROJ 8.08e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 1.11e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.78e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 1.09e-04 vs PROJ 1.06e-04 (u 2.69e-06); density 7.65e-04 vs PROJ 7.79e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 1.08e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 7.01e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 1.09e-04 vs PROJ 1.02e-04 (u 2.60e-06); density 6.90e-04 vs PROJ 7.36e-04 (u 7.16e-07)
  * M_seed320 (test_interp) P0: same-grid price 1.35e-04 vs PROJ 1.35e-04 (u 3.49e-06); density 9.69e-04 vs PROJ 9.80e-04 (u 3.82e-06)
  * M_seed320 (test_interp) P1: same-grid price 1.35e-04 vs PROJ 1.35e-04 (u 3.49e-06); density 9.72e-04 vs PROJ 9.80e-04 (u 3.82e-06)
  * C_sig0.550 (test_ood) P0: same-grid price 1.03e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.50e-04 vs PROJ 7.74e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 1.02e-04 vs PROJ 1.04e-04 (u 2.76e-06); density 7.43e-04 vs PROJ 7.74e-04 (u 7.86e-07)
* **QLp_m17** compensation: true (36 cells)
  * C_sig0.310 (test_interp) P1: same-grid price 3.28e-06 vs PROJ 3.90e-06 (u 3.44e-06); density 7.41e-05 vs PROJ 8.13e-05 (u 3.77e-06)
  * C_sig0.415 (test_interp) P0: same-grid price 7.09e-06 vs PROJ 7.30e-06 (u 3.02e-06); density 1.48e-04 vs PROJ 1.67e-04 (u 1.83e-06)
  * C_sig0.415 (test_interp) P1: same-grid price 5.46e-06 vs PROJ 7.30e-06 (u 3.02e-06); density 1.42e-04 vs PROJ 1.67e-04 (u 1.83e-06)
  * D2_seed220 (test_interp) P0: same-grid price 6.88e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.54e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 6.73e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.52e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 7.11e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.59e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 6.42e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.57e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 7.03e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.61e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 6.41e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.58e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 6.90e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.55e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 6.82e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.53e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * M_seed320 (test_interp) P1: same-grid price 4.03e-06 vs PROJ 5.09e-06 (u 3.49e-06); density 9.40e-05 vs PROJ 1.06e-04 (u 3.82e-06)
* **QLpm_m17** compensation: true (22 cells)
  * C_sig0.415 (test_interp) P0: same-grid price 1.16e-04 vs PROJ 7.30e-06 (u 3.02e-06); density 1.65e-04 vs PROJ 1.67e-04 (u 1.83e-06)
  * C_sig0.415 (test_interp) P1: same-grid price 7.94e-05 vs PROJ 7.30e-06 (u 3.02e-06); density 1.56e-04 vs PROJ 1.67e-04 (u 1.83e-06)
  * D2_seed220 (test_interp) P0: same-grid price 7.85e-05 vs PROJ 7.67e-06 (u 2.63e-06); density 1.63e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 4.82e-05 vs PROJ 7.67e-06 (u 2.63e-06); density 1.60e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 8.85e-05 vs PROJ 7.86e-06 (u 2.72e-06); density 1.70e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 5.61e-05 vs PROJ 7.86e-06 (u 2.72e-06); density 1.66e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 8.69e-05 vs PROJ 7.86e-06 (u 2.69e-06); density 1.72e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 5.49e-05 vs PROJ 7.86e-06 (u 2.69e-06); density 1.67e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 7.67e-05 vs PROJ 7.70e-06 (u 2.60e-06); density 1.64e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 4.68e-05 vs PROJ 7.70e-06 (u 2.60e-06); density 1.61e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * S2_seed120 (test_interp) P1: same-grid price 8.38e-05 vs PROJ 6.43e-06 (u 3.13e-06); density 1.37e-04 vs PROJ 1.42e-04 (u 2.23e-06)
  * S2_seed122 (test_interp) P1: same-grid price 8.44e-05 vs PROJ 6.03e-06 (u 3.16e-06); density 1.29e-04 vs PROJ 1.33e-04 (u 2.30e-06)
* **QLs_m17** compensation: true (22 cells)
  * C_sig0.415 (test_interp) P0: same-grid price 1.16e-04 vs PROJ 7.30e-06 (u 3.02e-06); density 1.65e-04 vs PROJ 1.67e-04 (u 1.83e-06)
  * C_sig0.415 (test_interp) P1: same-grid price 7.94e-05 vs PROJ 7.30e-06 (u 3.02e-06); density 1.56e-04 vs PROJ 1.67e-04 (u 1.83e-06)
  * D2_seed220 (test_interp) P0: same-grid price 7.85e-05 vs PROJ 7.67e-06 (u 2.63e-06); density 1.63e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 4.82e-05 vs PROJ 7.67e-06 (u 2.63e-06); density 1.60e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 8.85e-05 vs PROJ 7.86e-06 (u 2.72e-06); density 1.70e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 5.61e-05 vs PROJ 7.86e-06 (u 2.72e-06); density 1.66e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 8.69e-05 vs PROJ 7.86e-06 (u 2.69e-06); density 1.72e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 5.49e-05 vs PROJ 7.86e-06 (u 2.69e-06); density 1.67e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 7.67e-05 vs PROJ 7.70e-06 (u 2.60e-06); density 1.64e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 4.68e-05 vs PROJ 7.70e-06 (u 2.60e-06); density 1.61e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * S2_seed120 (test_interp) P1: same-grid price 8.38e-05 vs PROJ 6.43e-06 (u 3.13e-06); density 1.37e-04 vs PROJ 1.42e-04 (u 2.23e-06)
  * S2_seed122 (test_interp) P1: same-grid price 8.44e-05 vs PROJ 6.03e-06 (u 3.16e-06); density 1.29e-04 vs PROJ 1.33e-04 (u 2.30e-06)
* **LM_m17** compensation: true (14 cells)
  * D2_seed220 (test_interp) P0: same-grid price 6.93e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.71e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 6.51e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.69e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 7.61e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.83e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 6.99e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.81e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 7.94e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.89e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 7.20e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.86e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 7.02e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.73e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 6.57e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.71e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * C_sig0.550 (test_ood) P0: same-grid price 7.74e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.91e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 7.24e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.90e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * D2_seed230 (test_ood) P0: same-grid price 6.36e-06 vs PROJ 6.67e-06 (u 2.52e-06); density 1.60e-04 vs PROJ 1.62e-04 (u 4.43e-07)
  * D2_seed230 (test_ood) P1: same-grid price 6.26e-06 vs PROJ 6.67e-06 (u 2.52e-06); density 1.61e-04 vs PROJ 1.62e-04 (u 4.43e-07)
* **LMb_m17** compensation: true (12 cells)
  * D2_seed220 (test_interp) P0: same-grid price 7.03e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.74e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 6.61e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.72e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 7.68e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.84e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 7.06e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.83e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 8.01e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.90e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 7.27e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.88e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 7.12e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.76e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 6.67e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.74e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * C_sig0.550 (test_ood) P0: same-grid price 7.78e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.92e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 7.29e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.91e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * D2_seed231 (test_ood) P0: same-grid price 5.45e-06 vs PROJ 6.02e-06 (u 2.39e-06); density 1.38e-04 vs PROJ 1.40e-04 (u 3.41e-07)
  * D2_seed231 (test_ood) P1: same-grid price 5.64e-06 vs PROJ 6.02e-06 (u 2.39e-06); density 1.40e-04 vs PROJ 1.40e-04 (u 3.41e-07)
* **LR0_m17_h8/W0** compensation: true (13 cells)
  * D2_seed220 (test_interp) P0: same-grid price 6.91e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.72e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 6.66e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.72e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 7.56e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.82e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 7.15e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.85e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 7.86e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.88e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 7.35e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.89e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 6.99e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.74e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 6.72e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.73e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * C_sig0.550 (test_ood) P0: same-grid price 7.79e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.93e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 7.64e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.96e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * D2_seed230 (test_ood) P0: same-grid price 6.38e-06 vs PROJ 6.67e-06 (u 2.52e-06); density 1.61e-04 vs PROJ 1.62e-04 (u 4.43e-07)
  * D2_seed231 (test_ood) P0: same-grid price 5.31e-06 vs PROJ 6.02e-06 (u 2.39e-06); density 1.34e-04 vs PROJ 1.40e-04 (u 3.41e-07)
* **LR0_m17_h8/WI** compensation: true (13 cells)
  * D2_seed220 (test_interp) P0: same-grid price 6.91e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.72e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 6.55e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.70e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 7.56e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.82e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 7.01e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.81e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 7.86e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.88e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 7.21e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.86e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 6.99e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.74e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 6.61e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.72e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * C_sig0.550 (test_ood) P0: same-grid price 7.79e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.93e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 7.38e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.92e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * D2_seed230 (test_ood) P0: same-grid price 6.38e-06 vs PROJ 6.67e-06 (u 2.52e-06); density 1.61e-04 vs PROJ 1.62e-04 (u 4.43e-07)
  * D2_seed231 (test_ood) P0: same-grid price 5.31e-06 vs PROJ 6.02e-06 (u 2.39e-06); density 1.34e-04 vs PROJ 1.40e-04 (u 3.41e-07)
* **LR0_m17_h8/WU** compensation: true (13 cells)
  * D2_seed220 (test_interp) P0: same-grid price 6.91e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.72e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 6.55e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.70e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P0: same-grid price 7.56e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.82e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed221 (test_interp) P1: same-grid price 7.00e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.81e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed222 (test_interp) P0: same-grid price 7.86e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.88e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed222 (test_interp) P1: same-grid price 7.21e-06 vs PROJ 7.86e-06 (u 2.69e-06); density 1.86e-04 vs PROJ 1.92e-04 (u 8.67e-07)
  * D2_seed223 (test_interp) P0: same-grid price 6.99e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.74e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 6.61e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.72e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * C_sig0.550 (test_ood) P0: same-grid price 7.79e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.93e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 7.39e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.92e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * D2_seed230 (test_ood) P0: same-grid price 6.38e-06 vs PROJ 6.67e-06 (u 2.52e-06); density 1.61e-04 vs PROJ 1.62e-04 (u 4.43e-07)
  * D2_seed231 (test_ood) P0: same-grid price 5.31e-06 vs PROJ 6.02e-06 (u 2.39e-06); density 1.34e-04 vs PROJ 1.40e-04 (u 3.41e-07)
* **LR_m17_h8/W0** compensation: true (6 cells)
  * D2_seed220 (test_interp) P0: same-grid price 7.38e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.77e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 6.99e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.77e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed223 (test_interp) P0: same-grid price 7.45e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.79e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 7.05e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.78e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * C_sig0.550 (test_ood) P0: same-grid price 8.31e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.98e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * D2_seed231 (test_ood) P0: same-grid price 5.84e-06 vs PROJ 6.02e-06 (u 2.39e-06); density 1.38e-04 vs PROJ 1.40e-04 (u 3.41e-07)
* **LR_m17_h8/WI** compensation: true (8 cells)
  * D2_seed220 (test_interp) P0: same-grid price 7.38e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.77e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 6.87e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.74e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed221 (test_interp) P1: same-grid price 7.56e-06 vs PROJ 7.86e-06 (u 2.72e-06); density 1.88e-04 vs PROJ 1.90e-04 (u 9.88e-07)
  * D2_seed223 (test_interp) P0: same-grid price 7.45e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.79e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 6.92e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.76e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * C_sig0.550 (test_ood) P0: same-grid price 8.31e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.98e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 7.92e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.96e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * D2_seed231 (test_ood) P0: same-grid price 5.84e-06 vs PROJ 6.02e-06 (u 2.39e-06); density 1.38e-04 vs PROJ 1.40e-04 (u 3.41e-07)
* **LR_m17_h8/WU** compensation: true (8 cells)
  * D2_seed220 (test_interp) P0: same-grid price 7.38e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.77e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed220 (test_interp) P1: same-grid price 6.92e-06 vs PROJ 7.67e-06 (u 2.63e-06); density 1.75e-04 vs PROJ 1.85e-04 (u 7.83e-07)
  * D2_seed223 (test_interp) P0: same-grid price 7.45e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.79e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * D2_seed223 (test_interp) P1: same-grid price 6.94e-06 vs PROJ 7.70e-06 (u 2.60e-06); density 1.77e-04 vs PROJ 1.87e-04 (u 7.16e-07)
  * C_sig0.550 (test_ood) P0: same-grid price 8.31e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.98e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * C_sig0.550 (test_ood) P1: same-grid price 7.91e-06 vs PROJ 7.87e-06 (u 2.76e-06); density 1.96e-04 vs PROJ 1.99e-04 (u 7.86e-07)
  * D2_seed231 (test_ood) P0: same-grid price 5.84e-06 vs PROJ 6.02e-06 (u 2.39e-06); density 1.38e-04 vs PROJ 1.40e-04 (u 3.41e-07)
  * D2_seed231 (test_ood) P1: same-grid price 5.86e-06 vs PROJ 6.02e-06 (u 2.39e-06); density 1.39e-04 vs PROJ 1.40e-04 (u 3.41e-07)

## Classification table (test_interp)

| entry | primary class | all classes holding | cost beats RRR33 | #exc RDI | #exc useful | #exc compression | compression status | eligible/cells | compensation | reasons (no supported benefit) |
|---|---|---|---|---|---|---|---|---|---|---|
| QLp_m9 | no_supported_benefit | – | no | 34 | 73 | 0 | not_testable_regime_empty | 0/34 | yes | gains within uncertainty (no cell with a gain beyond 2u in e_dyn or price); validity deteriorated (a noninferiority condition fails in some cell) |
| QLpm_m9 | no_supported_benefit | – | no | 34 | 73 | 0 | not_testable_regime_empty | 0/34 | yes | gains within uncertainty (no cell with a gain beyond 2u in e_dyn or price); validity deteriorated (a noninferiority condition fails in some cell) |
| QLs_m9 | no_supported_benefit | – | no | 34 | 73 | 0 | not_testable_regime_empty | 0/34 | yes | gains within uncertainty (no cell with a gain beyond 2u in e_dyn or price); validity deteriorated (a noninferiority condition fails in some cell) |
| LM_m9 | inconclusive | – | no | 1 | 196 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LMb_m9 | inconclusive | – | no | 1 | 196 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LR0_m9_h16/W0 | no_supported_benefit | – | no | 12 | 212 | 0 | not_testable_regime_empty | 0/34 | yes | unclosed model of equal total state is at least as good (e_dyn and price, every cell); validity deteriorated (a noninferiority condition fails in some cell) |
| LR0_m9_h16/WI | resolved_dynamics_improvement | resolved_dynamics_improvement | no | 0 | 189 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LR0_m9_h16/WU | resolved_dynamics_improvement | resolved_dynamics_improvement | no | 0 | 189 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LR_m9_h16/W0 | no_supported_benefit | – | no | 12 | 220 | 0 | not_testable_regime_empty | 0/34 | yes | unclosed model of equal total state is at least as good (e_dyn and price, every cell); validity deteriorated (a noninferiority condition fails in some cell) |
| LR_m9_h16/WI | resolved_dynamics_improvement | resolved_dynamics_improvement | no | 0 | 213 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LR_m9_h16/WU | resolved_dynamics_improvement | resolved_dynamics_improvement | no | 0 | 210 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| QLp_m17 | no_supported_benefit | – | no | 34 | 68 | 0 | not_testable_regime_empty | 0/34 | yes | gains within uncertainty (no cell with a gain beyond 2u in e_dyn or price) |
| QLpm_m17 | no_supported_benefit | – | no | 34 | 201 | 0 | not_testable_regime_empty | 0/34 | yes | gains within uncertainty (no cell with a gain beyond 2u in e_dyn or price); validity deteriorated (a noninferiority condition fails in some cell) |
| QLs_m17 | no_supported_benefit | – | no | 34 | 201 | 0 | not_testable_regime_empty | 0/34 | yes | gains within uncertainty (no cell with a gain beyond 2u in e_dyn or price); validity deteriorated (a noninferiority condition fails in some cell) |
| LM_m17 | inconclusive | – | no | 2 | 186 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LMb_m17 | inconclusive | – | no | 2 | 188 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LR0_m17_h8/W0 | inconclusive | – | no | 12 | 217 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LR0_m17_h8/WI | inconclusive | – | no | 2 | 175 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LR0_m17_h8/WU | inconclusive | – | no | 2 | 174 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LR_m17_h8/W0 | inconclusive | – | no | 13 | 225 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LR_m17_h8/WI | inconclusive | – | no | 2 | 182 | 0 | not_testable_regime_empty | 0/34 | yes |  |
| LR_m17_h8/WU | inconclusive | – | no | 2 | 181 | 0 | not_testable_regime_empty | 0/34 | yes |  |

## Exceptions (first 30 per entry and class; complete lists in decision.json)

### QLp_m9 — not `resolved_dynamics_improvement` (34 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=2.51e-04, Y=2.14e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-3.68e-05
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.49e-04, Y=1.04e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-4.53e-05
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=1.75e-04, Y=1.10e-04, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-6.53e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.20e-04, Y=9.75e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-2.27e-05
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.04e-04, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-5.90e-05
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=1.13e-04, Y=8.81e-05, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-2.45e-05
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=1.84e-04, Y=1.41e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.27e-05
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=1.35e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-2.59e-05
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=1.80e-04, Y=1.35e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-4.48e-05
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=1.50e-04, Y=1.24e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-2.61e-05
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=1.76e-04, Y=1.30e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-4.50e-05
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=1.51e-04, Y=1.24e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-2.70e-05
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=1.83e-04, Y=1.41e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-4.22e-05
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.37e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-2.58e-05
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=1.77e-04, Y=1.21e-04, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-5.62e-05
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.27e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-2.59e-05
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.72e-04, Y=1.18e-04, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-5.46e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=9.90e-05, Y=7.06e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-2.84e-05
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=1.91e-04, Y=1.22e-04, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-6.95e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.24e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-3.67e-05
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=2.12e-04, Y=1.58e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-5.42e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.47e-04, Y=1.14e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-3.27e-05
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=9.82e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-6.21e-05
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=1.06e-04, Y=8.30e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-2.32e-05
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.68e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-6.69e-05
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.08e-04, Y=8.38e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-2.44e-05
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=1.58e-04, Y=9.39e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-6.43e-05
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=7.83e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-2.42e-05
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.86e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-6.55e-05
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.70e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-2.16e-05
* ... 4 more

### QLp_m9 — not `useful_closure` (73 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=2.51e-04, Y=2.14e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-3.68e-05
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.49e-04, Y=1.04e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-4.53e-05
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=1.75e-04, Y=1.10e-04, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-6.53e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.20e-04, Y=9.75e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-2.27e-05
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.04e-04, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-5.90e-05
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=1.13e-04, Y=8.81e-05, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-2.45e-05
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=1.84e-04, Y=1.41e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.27e-05
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=1.35e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-2.59e-05
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=1.80e-04, Y=1.35e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-4.48e-05
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=1.50e-04, Y=1.24e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-2.61e-05
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=1.76e-04, Y=1.30e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-4.50e-05
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=1.51e-04, Y=1.24e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-2.70e-05
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=1.83e-04, Y=1.41e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-4.22e-05
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.37e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-2.58e-05
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=1.77e-04, Y=1.21e-04, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-5.62e-05
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.27e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-2.59e-05
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.72e-04, Y=1.18e-04, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-5.46e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=9.90e-05, Y=7.06e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-2.84e-05
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=1.91e-04, Y=1.22e-04, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-6.95e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.24e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-3.67e-05
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=2.12e-04, Y=1.58e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-5.42e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.47e-04, Y=1.14e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-3.27e-05
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=9.82e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-6.21e-05
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=1.06e-04, Y=8.30e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-2.32e-05
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.68e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-6.69e-05
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.08e-04, Y=8.38e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-2.44e-05
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=1.58e-04, Y=9.39e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-6.43e-05
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=7.83e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-2.42e-05
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.86e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-6.55e-05
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.70e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-2.16e-05
* ... 43 more

### QLpm_m9 — not `resolved_dynamics_improvement` (34 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=2.51e-04, Y=2.14e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-3.68e-05
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.49e-04, Y=1.04e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-4.53e-05
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=1.75e-04, Y=1.10e-04, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-6.53e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.20e-04, Y=9.75e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-2.27e-05
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.04e-04, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-5.90e-05
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=1.13e-04, Y=8.81e-05, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-2.45e-05
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=1.84e-04, Y=1.41e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.27e-05
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=1.35e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-2.59e-05
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=1.80e-04, Y=1.35e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-4.48e-05
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=1.50e-04, Y=1.24e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-2.61e-05
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=1.76e-04, Y=1.30e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-4.50e-05
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=1.51e-04, Y=1.24e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-2.70e-05
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=1.83e-04, Y=1.41e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-4.22e-05
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.37e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-2.58e-05
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=1.77e-04, Y=1.21e-04, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-5.62e-05
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.27e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-2.59e-05
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.72e-04, Y=1.18e-04, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-5.46e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=9.90e-05, Y=7.06e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-2.84e-05
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=1.91e-04, Y=1.22e-04, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-6.95e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.24e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-3.67e-05
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=2.12e-04, Y=1.58e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-5.42e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.47e-04, Y=1.14e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-3.27e-05
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=9.82e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-6.21e-05
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=1.06e-04, Y=8.30e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-2.32e-05
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.68e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-6.69e-05
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.08e-04, Y=8.38e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-2.44e-05
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=1.58e-04, Y=9.39e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-6.43e-05
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=7.83e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-2.42e-05
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.86e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-6.55e-05
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.70e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-2.16e-05
* ... 4 more

### QLpm_m9 — not `useful_closure` (73 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=2.51e-04, Y=2.14e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-3.68e-05
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.49e-04, Y=1.04e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-4.53e-05
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=1.75e-04, Y=1.10e-04, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-6.53e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.20e-04, Y=9.75e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-2.27e-05
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.04e-04, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-5.90e-05
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=1.13e-04, Y=8.81e-05, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-2.45e-05
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=1.84e-04, Y=1.41e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.27e-05
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=1.35e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-2.59e-05
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=1.80e-04, Y=1.35e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-4.48e-05
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=1.50e-04, Y=1.24e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-2.61e-05
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=1.76e-04, Y=1.30e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-4.50e-05
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=1.51e-04, Y=1.24e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-2.70e-05
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=1.83e-04, Y=1.41e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-4.22e-05
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.37e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-2.58e-05
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=1.77e-04, Y=1.21e-04, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-5.62e-05
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.27e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-2.59e-05
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.72e-04, Y=1.18e-04, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-5.46e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=9.90e-05, Y=7.06e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-2.84e-05
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=1.91e-04, Y=1.22e-04, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-6.95e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.24e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-3.67e-05
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=2.12e-04, Y=1.58e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-5.42e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.47e-04, Y=1.14e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-3.27e-05
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=9.82e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-6.21e-05
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=1.06e-04, Y=8.30e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-2.32e-05
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.68e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-6.69e-05
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.08e-04, Y=8.38e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-2.44e-05
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=1.58e-04, Y=9.39e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-6.43e-05
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=7.83e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-2.42e-05
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.86e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-6.55e-05
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.70e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-2.16e-05
* ... 43 more

### QLs_m9 — not `resolved_dynamics_improvement` (34 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=2.51e-04, Y=2.14e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-3.68e-05
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.49e-04, Y=1.04e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-4.53e-05
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=1.75e-04, Y=1.10e-04, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-6.53e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.20e-04, Y=9.75e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-2.27e-05
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.04e-04, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-5.90e-05
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=1.13e-04, Y=8.81e-05, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-2.45e-05
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=1.84e-04, Y=1.41e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.27e-05
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=1.35e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-2.59e-05
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=1.80e-04, Y=1.35e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-4.48e-05
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=1.50e-04, Y=1.24e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-2.61e-05
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=1.76e-04, Y=1.30e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-4.50e-05
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=1.51e-04, Y=1.24e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-2.70e-05
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=1.83e-04, Y=1.41e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-4.22e-05
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.37e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-2.58e-05
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=1.77e-04, Y=1.21e-04, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-5.62e-05
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.27e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-2.59e-05
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.72e-04, Y=1.18e-04, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-5.46e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=9.90e-05, Y=7.06e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-2.84e-05
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=1.91e-04, Y=1.22e-04, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-6.95e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.24e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-3.67e-05
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=2.12e-04, Y=1.58e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-5.42e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.47e-04, Y=1.14e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-3.27e-05
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=9.82e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-6.21e-05
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=1.06e-04, Y=8.30e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-2.32e-05
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.68e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-6.69e-05
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.08e-04, Y=8.38e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-2.44e-05
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=1.58e-04, Y=9.39e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-6.43e-05
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=7.83e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-2.42e-05
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.86e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-6.55e-05
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.70e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-2.16e-05
* ... 4 more

### QLs_m9 — not `useful_closure` (73 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=2.51e-04, Y=2.14e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-3.68e-05
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.49e-04, Y=1.04e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-4.53e-05
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=1.75e-04, Y=1.10e-04, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-6.53e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.20e-04, Y=9.75e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-2.27e-05
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.04e-04, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-5.90e-05
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=1.13e-04, Y=8.81e-05, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-2.45e-05
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=1.84e-04, Y=1.41e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.27e-05
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=1.35e-04, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-2.59e-05
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=1.80e-04, Y=1.35e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-4.48e-05
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=1.50e-04, Y=1.24e-04, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-2.61e-05
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=1.76e-04, Y=1.30e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-4.50e-05
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=1.51e-04, Y=1.24e-04, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-2.70e-05
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=1.83e-04, Y=1.41e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-4.22e-05
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=1.63e-04, Y=1.37e-04, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-2.58e-05
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=1.77e-04, Y=1.21e-04, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-5.62e-05
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.27e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-2.59e-05
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.72e-04, Y=1.18e-04, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-5.46e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=9.90e-05, Y=7.06e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-2.84e-05
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=1.91e-04, Y=1.22e-04, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-6.95e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.24e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-3.67e-05
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=2.12e-04, Y=1.58e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-5.42e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.47e-04, Y=1.14e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-3.27e-05
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=1.60e-04, Y=9.82e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-6.21e-05
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=1.06e-04, Y=8.30e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-2.32e-05
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.68e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-6.69e-05
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.08e-04, Y=8.38e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-2.44e-05
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=1.58e-04, Y=9.39e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-6.43e-05
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=7.83e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-2.42e-05
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=1.64e-04, Y=9.86e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-6.55e-05
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.09e-04, Y=8.70e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-2.16e-05
* ... 43 more

### LM_m9 — not `resolved_dynamics_improvement` (1 exceptions)

* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-1.53e-05

### LM_m9 — not `useful_closure` (196 exceptions)

* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-1.53e-05
* C_sig0.205 / P0 / price_interior_rms [price_beyond]: X=4.05e-04, Y=3.99e-04, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=-6.00e-06
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=3.70e-04, Y=3.72e-04, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=1.43e-06
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=1.55e-04, Y=1.60e-04, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=5.33e-06
* M_seed322 / P0 / price_interior_rms [price_beyond]: X=1.77e-04, Y=1.79e-04, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=1.52e-06
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=1.71e-04, Y=1.49e-04, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-2.22e-05
* C_sig0.205 / P0 / density_l2_interior_mean [noninf]: X=2.91e-03, Y=2.85e-03, u=9.06e-06, tol=9.06e-06, threshold=2.86e-03
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=2.70e-01, Y=2.66e-01, u=0.00e+00, tol=1.81e-04, threshold=2.66e-01
* C_sig0.205 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=2.70e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P0 / invvol_invalid_frac_full [noninf]: X=4.20e-01, Y=3.52e-01, u=1.63e-04, tol=1.81e-04, threshold=3.53e-01
* C_sig0.205 / P0 / invvol_coverage [noninf]: X=8.64e-01, Y=8.85e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.205 / P1 / positivity_raw_interior_frac_mean [noninf]: X=2.40e-01, Y=2.35e-01, u=0.00e+00, tol=1.81e-04, threshold=2.35e-01
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=2.73e-01, Y=2.63e-01, u=0.00e+00, tol=1.81e-04, threshold=2.64e-01
* C_sig0.205 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=2.70e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P1 / calendar_raw_interior_frac_mean [noninf]: X=2.14e-01, Y=2.11e-01, u=0.00e+00, tol=1.81e-04, threshold=2.11e-01
* C_sig0.205 / P1 / invvol_invalid_frac_full [noninf]: X=4.24e-01, Y=3.75e-01, u=1.63e-04, tol=1.81e-04, threshold=3.75e-01
* C_sig0.205 / P1 / invvol_coverage [noninf]: X=8.65e-01, Y=8.84e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.310 / P0 / positivity_raw_interior_frac_mean [noninf]: X=7.36e-02, Y=7.17e-02, u=0.00e+00, tol=1.81e-04, threshold=7.19e-02
* C_sig0.310 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=9.91e-02, Y=8.77e-02, u=0.00e+00, tol=1.81e-04, threshold=8.79e-02
* C_sig0.310 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=1.13e-01, Y=8.61e-02, u=0.00e+00, tol=1.81e-04, threshold=8.62e-02
* C_sig0.310 / P0 / calendar_raw_interior_frac_mean [noninf]: X=5.65e-02, Y=5.60e-02, u=0.00e+00, tol=1.81e-04, threshold=5.62e-02
* C_sig0.310 / P0 / invvol_invalid_frac_full [noninf]: X=1.75e-01, Y=1.50e-01, u=1.63e-04, tol=1.81e-04, threshold=1.50e-01
* C_sig0.310 / P0 / invvol_coverage [noninf]: X=9.34e-01, Y=9.65e-01, u=0.00e+00, tol=1.81e-04, threshold=9.65e-01
* C_sig0.310 / P1 / negative_mass_max_post_initial [noninf]: X=1.48e-02, Y=1.29e-02, u=1.54e-12, tol=1.54e-12, threshold=1.29e-02
* C_sig0.310 / P1 / positivity_raw_interior_frac_mean [noninf]: X=7.34e-02, Y=6.76e-02, u=0.00e+00, tol=1.81e-04, threshold=6.78e-02
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=9.89e-02, Y=8.68e-02, u=0.00e+00, tol=1.81e-04, threshold=8.70e-02
* C_sig0.310 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=1.13e-01, Y=8.46e-02, u=0.00e+00, tol=1.81e-04, threshold=8.48e-02
* C_sig0.310 / P1 / calendar_raw_interior_frac_mean [noninf]: X=5.51e-02, Y=5.38e-02, u=0.00e+00, tol=1.81e-04, threshold=5.40e-02
* C_sig0.310 / P1 / invvol_invalid_frac_full [noninf]: X=1.71e-01, Y=1.51e-01, u=1.63e-04, tol=1.81e-04, threshold=1.52e-01
* C_sig0.310 / P1 / invvol_coverage [noninf]: X=9.33e-01, Y=9.66e-01, u=0.00e+00, tol=1.81e-04, threshold=9.66e-01
* ... 166 more

### LMb_m9 — not `resolved_dynamics_improvement` (1 exceptions)

* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.02e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-1.41e-05

### LMb_m9 — not `useful_closure` (196 exceptions)

* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.02e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-1.41e-05
* C_sig0.205 / P0 / price_interior_rms [price_beyond]: X=4.05e-04, Y=3.99e-04, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=-6.78e-06
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=3.70e-04, Y=3.72e-04, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=1.59e-06
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=1.54e-04, Y=1.60e-04, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=5.80e-06
* M_seed322 / P0 / price_interior_rms [price_beyond]: X=1.77e-04, Y=1.79e-04, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=2.04e-06
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=1.70e-04, Y=1.49e-04, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-2.14e-05
* C_sig0.205 / P0 / density_l2_interior_mean [noninf]: X=2.91e-03, Y=2.85e-03, u=9.06e-06, tol=9.06e-06, threshold=2.86e-03
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=2.70e-01, Y=2.66e-01, u=0.00e+00, tol=1.81e-04, threshold=2.66e-01
* C_sig0.205 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=2.71e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P0 / invvol_invalid_frac_full [noninf]: X=4.19e-01, Y=3.52e-01, u=1.63e-04, tol=1.81e-04, threshold=3.53e-01
* C_sig0.205 / P0 / invvol_coverage [noninf]: X=8.64e-01, Y=8.85e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.205 / P1 / positivity_raw_interior_frac_mean [noninf]: X=2.39e-01, Y=2.35e-01, u=0.00e+00, tol=1.81e-04, threshold=2.35e-01
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=2.73e-01, Y=2.63e-01, u=0.00e+00, tol=1.81e-04, threshold=2.64e-01
* C_sig0.205 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=2.71e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P1 / calendar_raw_interior_frac_mean [noninf]: X=2.15e-01, Y=2.11e-01, u=0.00e+00, tol=1.81e-04, threshold=2.11e-01
* C_sig0.205 / P1 / invvol_invalid_frac_full [noninf]: X=4.24e-01, Y=3.75e-01, u=1.63e-04, tol=1.81e-04, threshold=3.75e-01
* C_sig0.205 / P1 / invvol_coverage [noninf]: X=8.65e-01, Y=8.84e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.310 / P0 / positivity_raw_interior_frac_mean [noninf]: X=7.32e-02, Y=7.17e-02, u=0.00e+00, tol=1.81e-04, threshold=7.19e-02
* C_sig0.310 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=9.86e-02, Y=8.77e-02, u=0.00e+00, tol=1.81e-04, threshold=8.79e-02
* C_sig0.310 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=1.13e-01, Y=8.61e-02, u=0.00e+00, tol=1.81e-04, threshold=8.62e-02
* C_sig0.310 / P0 / calendar_raw_interior_frac_mean [noninf]: X=5.67e-02, Y=5.60e-02, u=0.00e+00, tol=1.81e-04, threshold=5.62e-02
* C_sig0.310 / P0 / invvol_invalid_frac_full [noninf]: X=1.75e-01, Y=1.50e-01, u=1.63e-04, tol=1.81e-04, threshold=1.50e-01
* C_sig0.310 / P0 / invvol_coverage [noninf]: X=9.33e-01, Y=9.65e-01, u=0.00e+00, tol=1.81e-04, threshold=9.65e-01
* C_sig0.310 / P1 / negative_mass_max_post_initial [noninf]: X=1.48e-02, Y=1.29e-02, u=1.54e-12, tol=1.54e-12, threshold=1.29e-02
* C_sig0.310 / P1 / positivity_raw_interior_frac_mean [noninf]: X=7.32e-02, Y=6.76e-02, u=0.00e+00, tol=1.81e-04, threshold=6.78e-02
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=9.86e-02, Y=8.68e-02, u=0.00e+00, tol=1.81e-04, threshold=8.70e-02
* C_sig0.310 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=1.12e-01, Y=8.46e-02, u=0.00e+00, tol=1.81e-04, threshold=8.48e-02
* C_sig0.310 / P1 / calendar_raw_interior_frac_mean [noninf]: X=5.53e-02, Y=5.38e-02, u=0.00e+00, tol=1.81e-04, threshold=5.40e-02
* C_sig0.310 / P1 / invvol_invalid_frac_full [noninf]: X=1.71e-01, Y=1.51e-01, u=1.63e-04, tol=1.81e-04, threshold=1.52e-01
* C_sig0.310 / P1 / invvol_coverage [noninf]: X=9.33e-01, Y=9.66e-01, u=0.00e+00, tol=1.81e-04, threshold=9.66e-01
* ... 166 more

### LR0_m9_h16/W0 — not `resolved_dynamics_improvement` (12 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.49e-04, Y=1.04e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-4.50e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.31e-04, Y=9.75e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-3.34e-05
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=1.16e-04, Y=8.27e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-3.34e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=1.04e-04, Y=7.06e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-3.39e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.33e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-4.51e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.22e-04, Y=1.14e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-7.88e-06
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=1.01e-04, Y=8.30e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-1.79e-05
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.15e-04, Y=8.38e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-3.16e-05
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=9.93e-05, Y=7.83e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-2.10e-05
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.19e-04, Y=8.70e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-3.20e-05
* S2_seed124 / P1 / resolved_interior_rms [edyn_beyond]: X=8.16e-05, Y=5.72e-05, u=1.50e-08, tol=1.50e-08, threshold=3.00e-08, gain=-2.43e-05
* S2_seed125 / P1 / resolved_interior_rms [edyn_beyond]: X=8.45e-05, Y=6.49e-05, u=1.77e-08, tol=1.77e-08, threshold=3.54e-08, gain=-1.96e-05

### LR0_m9_h16/W0 — not `useful_closure` (212 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.49e-04, Y=1.04e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-4.50e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.31e-04, Y=9.75e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-3.34e-05
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=1.16e-04, Y=8.27e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-3.34e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=1.04e-04, Y=7.06e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-3.39e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.33e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-4.51e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.22e-04, Y=1.14e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-7.88e-06
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=1.01e-04, Y=8.30e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-1.79e-05
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.15e-04, Y=8.38e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-3.16e-05
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=9.93e-05, Y=7.83e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-2.10e-05
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.19e-04, Y=8.70e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-3.20e-05
* S2_seed124 / P1 / resolved_interior_rms [edyn_beyond]: X=8.16e-05, Y=5.72e-05, u=1.50e-08, tol=1.50e-08, threshold=3.00e-08, gain=-2.43e-05
* S2_seed125 / P1 / resolved_interior_rms [edyn_beyond]: X=8.45e-05, Y=6.49e-05, u=1.77e-08, tol=1.77e-08, threshold=3.54e-08, gain=-1.96e-05
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=4.00e-04, Y=3.72e-04, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=-2.83e-05
* C_sig0.310 / P1 / price_interior_rms [price_beyond]: X=1.73e-04, Y=1.53e-04, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=-2.02e-05
* C_sig0.415 / P1 / price_interior_rms [price_beyond]: X=1.42e-04, Y=1.28e-04, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-1.37e-05
* M_seed320 / P1 / price_interior_rms [price_beyond]: X=1.73e-04, Y=1.52e-04, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=-2.13e-05
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=1.82e-04, Y=1.60e-04, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=-2.16e-05
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=1.78e-04, Y=1.49e-04, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-2.96e-05
* M_seed323 / P1 / price_interior_rms [price_beyond]: X=2.69e-04, Y=2.61e-04, u=3.83e-06, tol=3.83e-06, threshold=7.66e-06, gain=-7.80e-06
* S2_seed120 / P1 / price_interior_rms [price_beyond]: X=1.47e-04, Y=1.28e-04, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-1.93e-05
* S2_seed121 / P1 / price_interior_rms [price_beyond]: X=1.50e-04, Y=1.29e-04, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=-2.10e-05
* S2_seed122 / P1 / price_interior_rms [price_beyond]: X=1.43e-04, Y=1.23e-04, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-2.00e-05
* S2_seed123 / P1 / price_interior_rms [price_beyond]: X=1.59e-04, Y=1.37e-04, u=3.28e-06, tol=3.28e-06, threshold=6.57e-06, gain=-2.23e-05
* S2_seed124 / P1 / price_interior_rms [price_beyond]: X=1.16e-04, Y=9.56e-05, u=3.25e-06, tol=3.25e-06, threshold=6.50e-06, gain=-1.99e-05
* S2_seed125 / P1 / price_interior_rms [price_beyond]: X=1.27e-04, Y=1.07e-04, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-2.05e-05
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=2.73e-01, Y=2.66e-01, u=0.00e+00, tol=1.81e-04, threshold=2.66e-01
* C_sig0.205 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=2.67e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P0 / invvol_invalid_frac_full [noninf]: X=4.12e-01, Y=3.52e-01, u=1.63e-04, tol=1.81e-04, threshold=3.53e-01
* C_sig0.205 / P0 / invvol_coverage [noninf]: X=8.71e-01, Y=8.85e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.205 / P1 / density_l2_interior_mean [noninf]: X=2.92e-03, Y=2.85e-03, u=9.06e-06, tol=9.06e-06, threshold=2.86e-03
* ... 182 more

### LR0_m9_h16/W0 — not `stateful_advantage` (435 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [lm_edyn]: X=1.49e-04, Y=5.64e-05, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-9.23e-05
* C_sig0.310 / P1 / resolved_interior_rms [lm_edyn]: X=1.31e-04, Y=4.65e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-8.44e-05
* C_sig0.415 / P0 / resolved_interior_rms [lm_edyn]: X=4.83e-05, Y=4.21e-05, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-6.18e-06
* C_sig0.415 / P1 / resolved_interior_rms [lm_edyn]: X=8.75e-05, Y=3.02e-05, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-5.73e-05
* D2_seed220 / P0 / resolved_interior_rms [lm_edyn]: X=8.17e-05, Y=7.74e-05, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.35e-06
* D2_seed221 / P0 / resolved_interior_rms [lm_edyn]: X=8.00e-05, Y=7.22e-05, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-7.86e-06
* D2_seed221 / P1 / resolved_interior_rms [lm_edyn]: X=6.94e-05, Y=6.06e-05, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-8.75e-06
* D2_seed222 / P0 / resolved_interior_rms [lm_edyn]: X=7.37e-05, Y=6.46e-05, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-9.11e-06
* D2_seed222 / P1 / resolved_interior_rms [lm_edyn]: X=5.97e-05, Y=5.77e-05, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-2.03e-06
* D2_seed223 / P0 / resolved_interior_rms [lm_edyn]: X=7.95e-05, Y=7.60e-05, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-3.49e-06
* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=2.29e-05, Y=1.94e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-3.52e-06
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=1.16e-04, Y=1.92e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-9.68e-05
* M_seed321 / P1 / resolved_interior_rms [lm_edyn]: X=1.04e-04, Y=4.10e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-6.35e-05
* M_seed322 / P1 / resolved_interior_rms [lm_edyn]: X=1.33e-04, Y=1.03e-04, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-2.97e-05
* M_seed323 / P1 / resolved_interior_rms [lm_edyn]: X=1.22e-04, Y=4.80e-05, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-7.41e-05
* S2_seed120 / P0 / resolved_interior_rms [lm_edyn]: X=5.16e-05, Y=4.69e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-4.73e-06
* S2_seed120 / P1 / resolved_interior_rms [lm_edyn]: X=1.01e-04, Y=3.81e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-6.28e-05
* S2_seed121 / P0 / resolved_interior_rms [lm_edyn]: X=4.95e-05, Y=4.57e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-3.81e-06
* S2_seed121 / P1 / resolved_interior_rms [lm_edyn]: X=1.15e-04, Y=4.00e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-7.54e-05
* S2_seed122 / P0 / resolved_interior_rms [lm_edyn]: X=4.70e-05, Y=4.26e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-4.39e-06
* S2_seed122 / P1 / resolved_interior_rms [lm_edyn]: X=9.93e-05, Y=3.41e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-6.52e-05
* S2_seed123 / P0 / resolved_interior_rms [lm_edyn]: X=5.02e-05, Y=4.95e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-7.05e-07
* S2_seed123 / P1 / resolved_interior_rms [lm_edyn]: X=1.19e-04, Y=4.61e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-7.29e-05
* S2_seed124 / P1 / resolved_interior_rms [lm_edyn]: X=8.16e-05, Y=2.02e-05, u=1.50e-08, tol=1.50e-08, threshold=3.00e-08, gain=-6.14e-05
* S2_seed125 / P1 / resolved_interior_rms [lm_edyn]: X=8.45e-05, Y=2.30e-05, u=1.77e-08, tol=1.77e-08, threshold=3.54e-08, gain=-6.15e-05
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.73e-01, Y=2.70e-01, u=0.00e+00, tol=1.81e-04, threshold=2.71e-01
* C_sig0.205 / P1 / price_interior_rms [lm_noninf]: X=4.00e-04, Y=3.70e-04, u=4.29e-06, tol=4.29e-06, threshold=3.75e-04
* C_sig0.205 / P1 / density_l2_interior_mean [lm_noninf]: X=2.92e-03, Y=2.83e-03, u=9.06e-06, tol=9.06e-06, threshold=2.84e-03
* C_sig0.205 / P1 / negative_mass_max_post_initial [lm_noninf]: X=9.02e-02, Y=8.87e-02, u=1.54e-12, tol=1.54e-12, threshold=8.87e-02
* C_sig0.205 / P1 / calendar_raw_interior_frac_mean [lm_noninf]: X=2.14e-01, Y=2.14e-01, u=0.00e+00, tol=1.81e-04, threshold=2.14e-01
* ... 405 more

### LR0_m9_h16/WI — not `useful_closure` (189 exceptions)

* M_seed321 / P1 / price_interior_rms [price_beyond]: X=1.54e-04, Y=1.60e-04, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=6.57e-06
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=1.50e-04, Y=1.49e-04, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-1.56e-06
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=2.73e-01, Y=2.66e-01, u=0.00e+00, tol=1.81e-04, threshold=2.66e-01
* C_sig0.205 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=2.67e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P0 / invvol_invalid_frac_full [noninf]: X=4.12e-01, Y=3.52e-01, u=1.63e-04, tol=1.81e-04, threshold=3.53e-01
* C_sig0.205 / P0 / invvol_coverage [noninf]: X=8.71e-01, Y=8.85e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.205 / P1 / positivity_raw_interior_frac_mean [noninf]: X=2.39e-01, Y=2.35e-01, u=0.00e+00, tol=1.81e-04, threshold=2.35e-01
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=2.72e-01, Y=2.63e-01, u=0.00e+00, tol=1.81e-04, threshold=2.64e-01
* C_sig0.205 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=2.67e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P1 / invvol_invalid_frac_full [noninf]: X=4.10e-01, Y=3.75e-01, u=1.63e-04, tol=1.81e-04, threshold=3.75e-01
* C_sig0.205 / P1 / invvol_coverage [noninf]: X=8.71e-01, Y=8.84e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.310 / P0 / positivity_raw_interior_frac_mean [noninf]: X=7.54e-02, Y=7.17e-02, u=0.00e+00, tol=1.81e-04, threshold=7.19e-02
* C_sig0.310 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=1.03e-01, Y=8.77e-02, u=0.00e+00, tol=1.81e-04, threshold=8.79e-02
* C_sig0.310 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=1.11e-01, Y=8.61e-02, u=0.00e+00, tol=1.81e-04, threshold=8.62e-02
* C_sig0.310 / P0 / invvol_invalid_frac_full [noninf]: X=1.76e-01, Y=1.50e-01, u=1.63e-04, tol=1.81e-04, threshold=1.50e-01
* C_sig0.310 / P0 / invvol_coverage [noninf]: X=9.41e-01, Y=9.65e-01, u=0.00e+00, tol=1.81e-04, threshold=9.65e-01
* C_sig0.310 / P1 / negative_mass_max_post_initial [noninf]: X=1.43e-02, Y=1.29e-02, u=1.54e-12, tol=1.54e-12, threshold=1.29e-02
* C_sig0.310 / P1 / positivity_raw_interior_frac_mean [noninf]: X=7.50e-02, Y=6.76e-02, u=0.00e+00, tol=1.81e-04, threshold=6.78e-02
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=1.03e-01, Y=8.68e-02, u=0.00e+00, tol=1.81e-04, threshold=8.70e-02
* C_sig0.310 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=1.12e-01, Y=8.46e-02, u=0.00e+00, tol=1.81e-04, threshold=8.48e-02
* C_sig0.310 / P1 / calendar_raw_interior_frac_mean [noninf]: X=5.42e-02, Y=5.38e-02, u=0.00e+00, tol=1.81e-04, threshold=5.40e-02
* C_sig0.310 / P1 / invvol_invalid_frac_full [noninf]: X=1.71e-01, Y=1.51e-01, u=1.63e-04, tol=1.81e-04, threshold=1.52e-01
* C_sig0.310 / P1 / invvol_coverage [noninf]: X=9.41e-01, Y=9.66e-01, u=0.00e+00, tol=1.81e-04, threshold=9.66e-01
* C_sig0.415 / P0 / negative_mass_max_post_initial [noninf]: X=4.69e-03, Y=3.76e-03, u=1.54e-12, tol=1.54e-12, threshold=3.76e-03
* C_sig0.415 / P0 / positivity_raw_interior_frac_mean [noninf]: X=1.74e-02, Y=1.59e-02, u=0.00e+00, tol=1.81e-04, threshold=1.61e-02
* C_sig0.415 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=2.79e-02, Y=2.25e-02, u=0.00e+00, tol=1.81e-04, threshold=2.26e-02
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=3.12e-02, Y=2.28e-02, u=0.00e+00, tol=1.81e-04, threshold=2.30e-02
* C_sig0.415 / P0 / calendar_raw_interior_frac_mean [noninf]: X=9.60e-03, Y=9.06e-03, u=0.00e+00, tol=1.81e-04, threshold=9.24e-03
* C_sig0.415 / P0 / invvol_invalid_frac_full [noninf]: X=5.71e-02, Y=5.45e-02, u=0.00e+00, tol=1.81e-04, threshold=5.47e-02
* C_sig0.415 / P0 / invvol_coverage [noninf]: X=9.88e-01, Y=9.91e-01, u=0.00e+00, tol=1.81e-04, threshold=9.91e-01
* ... 159 more

### LR0_m9_h16/WI — not `stateful_advantage` (403 exceptions)

* C_sig0.415 / P0 / resolved_interior_rms [lm_edyn]: X=4.83e-05, Y=4.21e-05, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-6.18e-06
* D2_seed220 / P0 / resolved_interior_rms [lm_edyn]: X=8.17e-05, Y=7.74e-05, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.35e-06
* D2_seed220 / P1 / resolved_interior_rms [lm_edyn]: X=8.08e-05, Y=7.40e-05, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-6.84e-06
* D2_seed221 / P0 / resolved_interior_rms [lm_edyn]: X=8.00e-05, Y=7.22e-05, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-7.86e-06
* D2_seed221 / P1 / resolved_interior_rms [lm_edyn]: X=6.87e-05, Y=6.06e-05, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-8.09e-06
* D2_seed222 / P0 / resolved_interior_rms [lm_edyn]: X=7.37e-05, Y=6.46e-05, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-9.11e-06
* D2_seed222 / P1 / resolved_interior_rms [lm_edyn]: X=6.77e-05, Y=5.77e-05, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-1.00e-05
* D2_seed223 / P0 / resolved_interior_rms [lm_edyn]: X=7.95e-05, Y=7.60e-05, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-3.49e-06
* D2_seed223 / P1 / resolved_interior_rms [lm_edyn]: X=8.17e-05, Y=7.51e-05, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-6.60e-06
* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=2.29e-05, Y=1.94e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-3.52e-06
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=2.33e-05, Y=1.92e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-4.01e-06
* S2_seed120 / P0 / resolved_interior_rms [lm_edyn]: X=5.16e-05, Y=4.69e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-4.73e-06
* S2_seed121 / P0 / resolved_interior_rms [lm_edyn]: X=4.95e-05, Y=4.57e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-3.81e-06
* S2_seed122 / P0 / resolved_interior_rms [lm_edyn]: X=4.70e-05, Y=4.26e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-4.39e-06
* S2_seed123 / P0 / resolved_interior_rms [lm_edyn]: X=5.02e-05, Y=4.95e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-7.05e-07
* S2_seed124 / P1 / resolved_interior_rms [lm_edyn]: X=2.64e-05, Y=2.02e-05, u=1.50e-08, tol=1.50e-08, threshold=3.00e-08, gain=-6.12e-06
* S2_seed125 / P1 / resolved_interior_rms [lm_edyn]: X=2.79e-05, Y=2.30e-05, u=1.77e-08, tol=1.77e-08, threshold=3.54e-08, gain=-4.89e-06
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.73e-01, Y=2.70e-01, u=0.00e+00, tol=1.81e-04, threshold=2.71e-01
* C_sig0.205 / P1 / negative_mass_max_post_initial [lm_noninf]: X=8.99e-02, Y=8.87e-02, u=1.54e-12, tol=1.54e-12, threshold=8.87e-02
* C_sig0.310 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=7.54e-02, Y=7.36e-02, u=0.00e+00, tol=1.81e-04, threshold=7.37e-02
* C_sig0.310 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.03e-01, Y=9.91e-02, u=0.00e+00, tol=1.81e-04, threshold=9.93e-02
* C_sig0.310 / P0 / invvol_invalid_frac_full [lm_noninf]: X=1.76e-01, Y=1.75e-01, u=1.63e-04, tol=1.81e-04, threshold=1.75e-01
* C_sig0.310 / P1 / positivity_raw_interior_frac_mean [lm_noninf]: X=7.50e-02, Y=7.34e-02, u=0.00e+00, tol=1.81e-04, threshold=7.36e-02
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.03e-01, Y=9.89e-02, u=0.00e+00, tol=1.81e-04, threshold=9.91e-02
* C_sig0.415 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.74e-02, Y=1.63e-02, u=0.00e+00, tol=1.81e-04, threshold=1.65e-02
* C_sig0.415 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.79e-02, Y=2.50e-02, u=0.00e+00, tol=1.81e-04, threshold=2.52e-02
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=3.12e-02, Y=2.79e-02, u=0.00e+00, tol=1.81e-04, threshold=2.81e-02
* C_sig0.415 / P1 / negative_mass_max_post_initial [lm_noninf]: X=4.10e-03, Y=3.94e-03, u=1.54e-12, tol=1.54e-12, threshold=3.94e-03
* C_sig0.415 / P1 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.83e-02, Y=1.65e-02, u=0.00e+00, tol=1.81e-04, threshold=1.67e-02
* C_sig0.415 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.92e-02, Y=2.55e-02, u=0.00e+00, tol=1.81e-04, threshold=2.57e-02
* ... 373 more

### LR0_m9_h16/WU — not `useful_closure` (189 exceptions)

* M_seed321 / P1 / price_interior_rms [price_beyond]: X=1.54e-04, Y=1.60e-04, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=6.59e-06
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=1.50e-04, Y=1.49e-04, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-1.43e-06
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=2.73e-01, Y=2.66e-01, u=0.00e+00, tol=1.81e-04, threshold=2.66e-01
* C_sig0.205 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=2.67e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P0 / invvol_invalid_frac_full [noninf]: X=4.12e-01, Y=3.52e-01, u=1.63e-04, tol=1.81e-04, threshold=3.53e-01
* C_sig0.205 / P0 / invvol_coverage [noninf]: X=8.71e-01, Y=8.85e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.205 / P1 / positivity_raw_interior_frac_mean [noninf]: X=2.39e-01, Y=2.35e-01, u=0.00e+00, tol=1.81e-04, threshold=2.35e-01
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=2.73e-01, Y=2.63e-01, u=0.00e+00, tol=1.81e-04, threshold=2.64e-01
* C_sig0.205 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=2.67e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P1 / invvol_invalid_frac_full [noninf]: X=4.10e-01, Y=3.75e-01, u=1.63e-04, tol=1.81e-04, threshold=3.75e-01
* C_sig0.205 / P1 / invvol_coverage [noninf]: X=8.71e-01, Y=8.84e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.310 / P0 / positivity_raw_interior_frac_mean [noninf]: X=7.54e-02, Y=7.17e-02, u=0.00e+00, tol=1.81e-04, threshold=7.19e-02
* C_sig0.310 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=1.03e-01, Y=8.77e-02, u=0.00e+00, tol=1.81e-04, threshold=8.79e-02
* C_sig0.310 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=1.11e-01, Y=8.61e-02, u=0.00e+00, tol=1.81e-04, threshold=8.62e-02
* C_sig0.310 / P0 / invvol_invalid_frac_full [noninf]: X=1.76e-01, Y=1.50e-01, u=1.63e-04, tol=1.81e-04, threshold=1.50e-01
* C_sig0.310 / P0 / invvol_coverage [noninf]: X=9.41e-01, Y=9.65e-01, u=0.00e+00, tol=1.81e-04, threshold=9.65e-01
* C_sig0.310 / P1 / negative_mass_max_post_initial [noninf]: X=1.43e-02, Y=1.29e-02, u=1.54e-12, tol=1.54e-12, threshold=1.29e-02
* C_sig0.310 / P1 / positivity_raw_interior_frac_mean [noninf]: X=7.52e-02, Y=6.76e-02, u=0.00e+00, tol=1.81e-04, threshold=6.78e-02
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=1.03e-01, Y=8.68e-02, u=0.00e+00, tol=1.81e-04, threshold=8.70e-02
* C_sig0.310 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=1.12e-01, Y=8.46e-02, u=0.00e+00, tol=1.81e-04, threshold=8.48e-02
* C_sig0.310 / P1 / calendar_raw_interior_frac_mean [noninf]: X=5.40e-02, Y=5.38e-02, u=0.00e+00, tol=1.81e-04, threshold=5.40e-02
* C_sig0.310 / P1 / invvol_invalid_frac_full [noninf]: X=1.71e-01, Y=1.51e-01, u=1.63e-04, tol=1.81e-04, threshold=1.52e-01
* C_sig0.310 / P1 / invvol_coverage [noninf]: X=9.41e-01, Y=9.66e-01, u=0.00e+00, tol=1.81e-04, threshold=9.66e-01
* C_sig0.415 / P0 / negative_mass_max_post_initial [noninf]: X=4.69e-03, Y=3.76e-03, u=1.54e-12, tol=1.54e-12, threshold=3.76e-03
* C_sig0.415 / P0 / positivity_raw_interior_frac_mean [noninf]: X=1.74e-02, Y=1.59e-02, u=0.00e+00, tol=1.81e-04, threshold=1.61e-02
* C_sig0.415 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=2.79e-02, Y=2.25e-02, u=0.00e+00, tol=1.81e-04, threshold=2.26e-02
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=3.12e-02, Y=2.28e-02, u=0.00e+00, tol=1.81e-04, threshold=2.30e-02
* C_sig0.415 / P0 / calendar_raw_interior_frac_mean [noninf]: X=9.60e-03, Y=9.06e-03, u=0.00e+00, tol=1.81e-04, threshold=9.24e-03
* C_sig0.415 / P0 / invvol_invalid_frac_full [noninf]: X=5.71e-02, Y=5.45e-02, u=0.00e+00, tol=1.81e-04, threshold=5.47e-02
* C_sig0.415 / P0 / invvol_coverage [noninf]: X=9.88e-01, Y=9.91e-01, u=0.00e+00, tol=1.81e-04, threshold=9.91e-01
* ... 159 more

### LR0_m9_h16/WU — not `stateful_advantage` (403 exceptions)

* C_sig0.415 / P0 / resolved_interior_rms [lm_edyn]: X=4.83e-05, Y=4.21e-05, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-6.18e-06
* D2_seed220 / P0 / resolved_interior_rms [lm_edyn]: X=8.17e-05, Y=7.74e-05, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.35e-06
* D2_seed220 / P1 / resolved_interior_rms [lm_edyn]: X=8.12e-05, Y=7.40e-05, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-7.20e-06
* D2_seed221 / P0 / resolved_interior_rms [lm_edyn]: X=8.00e-05, Y=7.22e-05, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-7.86e-06
* D2_seed221 / P1 / resolved_interior_rms [lm_edyn]: X=6.89e-05, Y=6.06e-05, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-8.24e-06
* D2_seed222 / P0 / resolved_interior_rms [lm_edyn]: X=7.37e-05, Y=6.46e-05, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-9.11e-06
* D2_seed222 / P1 / resolved_interior_rms [lm_edyn]: X=6.80e-05, Y=5.77e-05, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-1.03e-05
* D2_seed223 / P0 / resolved_interior_rms [lm_edyn]: X=7.95e-05, Y=7.60e-05, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-3.49e-06
* D2_seed223 / P1 / resolved_interior_rms [lm_edyn]: X=8.21e-05, Y=7.51e-05, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-7.03e-06
* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=2.29e-05, Y=1.94e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-3.52e-06
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=2.34e-05, Y=1.92e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-4.13e-06
* S2_seed120 / P0 / resolved_interior_rms [lm_edyn]: X=5.16e-05, Y=4.69e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-4.73e-06
* S2_seed121 / P0 / resolved_interior_rms [lm_edyn]: X=4.95e-05, Y=4.57e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-3.81e-06
* S2_seed122 / P0 / resolved_interior_rms [lm_edyn]: X=4.70e-05, Y=4.26e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-4.39e-06
* S2_seed123 / P0 / resolved_interior_rms [lm_edyn]: X=5.02e-05, Y=4.95e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-7.05e-07
* S2_seed124 / P1 / resolved_interior_rms [lm_edyn]: X=2.64e-05, Y=2.02e-05, u=1.50e-08, tol=1.50e-08, threshold=3.00e-08, gain=-6.16e-06
* S2_seed125 / P1 / resolved_interior_rms [lm_edyn]: X=2.79e-05, Y=2.30e-05, u=1.77e-08, tol=1.77e-08, threshold=3.54e-08, gain=-4.94e-06
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.73e-01, Y=2.70e-01, u=0.00e+00, tol=1.81e-04, threshold=2.71e-01
* C_sig0.205 / P1 / negative_mass_max_post_initial [lm_noninf]: X=8.99e-02, Y=8.87e-02, u=1.54e-12, tol=1.54e-12, threshold=8.87e-02
* C_sig0.310 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=7.54e-02, Y=7.36e-02, u=0.00e+00, tol=1.81e-04, threshold=7.37e-02
* C_sig0.310 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.03e-01, Y=9.91e-02, u=0.00e+00, tol=1.81e-04, threshold=9.93e-02
* C_sig0.310 / P0 / invvol_invalid_frac_full [lm_noninf]: X=1.76e-01, Y=1.75e-01, u=1.63e-04, tol=1.81e-04, threshold=1.75e-01
* C_sig0.310 / P1 / positivity_raw_interior_frac_mean [lm_noninf]: X=7.52e-02, Y=7.34e-02, u=0.00e+00, tol=1.81e-04, threshold=7.36e-02
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.03e-01, Y=9.89e-02, u=0.00e+00, tol=1.81e-04, threshold=9.91e-02
* C_sig0.415 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.74e-02, Y=1.63e-02, u=0.00e+00, tol=1.81e-04, threshold=1.65e-02
* C_sig0.415 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.79e-02, Y=2.50e-02, u=0.00e+00, tol=1.81e-04, threshold=2.52e-02
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=3.12e-02, Y=2.79e-02, u=0.00e+00, tol=1.81e-04, threshold=2.81e-02
* C_sig0.415 / P1 / negative_mass_max_post_initial [lm_noninf]: X=4.09e-03, Y=3.94e-03, u=1.54e-12, tol=1.54e-12, threshold=3.94e-03
* C_sig0.415 / P1 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.83e-02, Y=1.65e-02, u=0.00e+00, tol=1.81e-04, threshold=1.67e-02
* C_sig0.415 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.92e-02, Y=2.55e-02, u=0.00e+00, tol=1.81e-04, threshold=2.57e-02
* ... 373 more

### LR_m9_h16/W0 — not `resolved_dynamics_improvement` (12 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.67e-04, Y=1.04e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-6.38e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.40e-04, Y=9.75e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-4.24e-05
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=1.22e-04, Y=8.27e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-3.91e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=1.12e-04, Y=7.06e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-4.18e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.28e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-4.09e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.33e-04, Y=1.14e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-1.92e-05
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=9.81e-05, Y=8.30e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-1.51e-05
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.20e-04, Y=8.38e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-3.57e-05
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=9.76e-05, Y=7.83e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-1.93e-05
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.22e-04, Y=8.70e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-3.52e-05
* S2_seed124 / P1 / resolved_interior_rms [edyn_beyond]: X=8.11e-05, Y=5.72e-05, u=1.50e-08, tol=1.50e-08, threshold=3.00e-08, gain=-2.38e-05
* S2_seed125 / P1 / resolved_interior_rms [edyn_beyond]: X=8.34e-05, Y=6.49e-05, u=1.77e-08, tol=1.77e-08, threshold=3.54e-08, gain=-1.85e-05

### LR_m9_h16/W0 — not `useful_closure` (220 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.67e-04, Y=1.04e-04, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-6.38e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.40e-04, Y=9.75e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-4.24e-05
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=1.22e-04, Y=8.27e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-3.91e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=1.12e-04, Y=7.06e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-4.18e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.28e-04, Y=8.75e-05, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-4.09e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.33e-04, Y=1.14e-04, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-1.92e-05
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=9.81e-05, Y=8.30e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-1.51e-05
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.20e-04, Y=8.38e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-3.57e-05
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=9.76e-05, Y=7.83e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-1.93e-05
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.22e-04, Y=8.70e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-3.52e-05
* S2_seed124 / P1 / resolved_interior_rms [edyn_beyond]: X=8.11e-05, Y=5.72e-05, u=1.50e-08, tol=1.50e-08, threshold=3.00e-08, gain=-2.38e-05
* S2_seed125 / P1 / resolved_interior_rms [edyn_beyond]: X=8.34e-05, Y=6.49e-05, u=1.77e-08, tol=1.77e-08, threshold=3.54e-08, gain=-1.85e-05
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=4.08e-04, Y=3.72e-04, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=-3.62e-05
* C_sig0.310 / P1 / price_interior_rms [price_beyond]: X=1.81e-04, Y=1.53e-04, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=-2.83e-05
* C_sig0.415 / P1 / price_interior_rms [price_beyond]: X=1.43e-04, Y=1.28e-04, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-1.48e-05
* M_seed320 / P1 / price_interior_rms [price_beyond]: X=1.79e-04, Y=1.52e-04, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=-2.65e-05
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=1.89e-04, Y=1.60e-04, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=-2.91e-05
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=1.73e-04, Y=1.49e-04, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-2.38e-05
* M_seed323 / P1 / price_interior_rms [price_beyond]: X=2.74e-04, Y=2.61e-04, u=3.83e-06, tol=3.83e-06, threshold=7.66e-06, gain=-1.28e-05
* S2_seed120 / P1 / price_interior_rms [price_beyond]: X=1.49e-04, Y=1.28e-04, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-2.13e-05
* S2_seed121 / P1 / price_interior_rms [price_beyond]: X=1.55e-04, Y=1.29e-04, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=-2.61e-05
* S2_seed122 / P1 / price_interior_rms [price_beyond]: X=1.45e-04, Y=1.23e-04, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-2.24e-05
* S2_seed123 / P1 / price_interior_rms [price_beyond]: X=1.64e-04, Y=1.37e-04, u=3.28e-06, tol=3.28e-06, threshold=6.57e-06, gain=-2.75e-05
* S2_seed124 / P1 / price_interior_rms [price_beyond]: X=1.18e-04, Y=9.56e-05, u=3.25e-06, tol=3.25e-06, threshold=6.50e-06, gain=-2.19e-05
* S2_seed125 / P1 / price_interior_rms [price_beyond]: X=1.29e-04, Y=1.07e-04, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-2.28e-05
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=2.73e-01, Y=2.66e-01, u=0.00e+00, tol=1.81e-04, threshold=2.66e-01
* C_sig0.205 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=2.66e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P0 / invvol_invalid_frac_full [noninf]: X=4.09e-01, Y=3.52e-01, u=1.63e-04, tol=1.81e-04, threshold=3.53e-01
* C_sig0.205 / P0 / invvol_coverage [noninf]: X=8.72e-01, Y=8.85e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.205 / P1 / density_l2_interior_mean [noninf]: X=2.94e-03, Y=2.85e-03, u=9.06e-06, tol=9.06e-06, threshold=2.86e-03
* ... 190 more

### LR_m9_h16/W0 — not `stateful_advantage` (460 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [lm_edyn]: X=1.67e-04, Y=5.64e-05, u=7.20e-09, tol=7.20e-09, threshold=1.44e-08, gain=-1.11e-04
* C_sig0.310 / P1 / resolved_interior_rms [lm_edyn]: X=1.40e-04, Y=4.65e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-9.34e-05
* C_sig0.415 / P1 / resolved_interior_rms [lm_edyn]: X=8.18e-05, Y=3.02e-05, u=2.27e-08, tol=2.27e-08, threshold=4.54e-08, gain=-5.15e-05
* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=2.42e-05, Y=1.94e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-4.81e-06
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=1.22e-04, Y=1.92e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.03e-04
* M_seed321 / P1 / resolved_interior_rms [lm_edyn]: X=1.12e-04, Y=4.10e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-7.14e-05
* M_seed322 / P1 / resolved_interior_rms [lm_edyn]: X=1.28e-04, Y=1.03e-04, u=1.58e-08, tol=1.58e-08, threshold=3.15e-08, gain=-2.55e-05
* M_seed323 / P1 / resolved_interior_rms [lm_edyn]: X=1.33e-04, Y=4.80e-05, u=2.05e-08, tol=2.05e-08, threshold=4.10e-08, gain=-8.54e-05
* S2_seed120 / P1 / resolved_interior_rms [lm_edyn]: X=9.81e-05, Y=3.81e-05, u=1.91e-08, tol=1.91e-08, threshold=3.82e-08, gain=-6.00e-05
* S2_seed121 / P0 / resolved_interior_rms [lm_edyn]: X=4.57e-05, Y=4.57e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-1.96e-09
* S2_seed121 / P1 / resolved_interior_rms [lm_edyn]: X=1.20e-04, Y=4.00e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-7.95e-05
* S2_seed122 / P1 / resolved_interior_rms [lm_edyn]: X=9.76e-05, Y=3.41e-05, u=1.80e-08, tol=1.80e-08, threshold=3.60e-08, gain=-6.35e-05
* S2_seed123 / P1 / resolved_interior_rms [lm_edyn]: X=1.22e-04, Y=4.61e-05, u=1.59e-08, tol=1.59e-08, threshold=3.19e-08, gain=-7.61e-05
* S2_seed124 / P1 / resolved_interior_rms [lm_edyn]: X=8.11e-05, Y=2.02e-05, u=1.50e-08, tol=1.50e-08, threshold=3.00e-08, gain=-6.08e-05
* S2_seed125 / P1 / resolved_interior_rms [lm_edyn]: X=8.34e-05, Y=2.30e-05, u=1.77e-08, tol=1.77e-08, threshold=3.54e-08, gain=-6.04e-05
* C_sig0.205 / P0 / negative_mass_max_post_initial [lm_noninf]: X=1.11e-01, Y=1.10e-01, u=1.54e-12, tol=1.54e-12, threshold=1.10e-01
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.73e-01, Y=2.70e-01, u=0.00e+00, tol=1.81e-04, threshold=2.71e-01
* C_sig0.205 / P1 / price_interior_rms [lm_noninf]: X=4.08e-04, Y=3.70e-04, u=4.29e-06, tol=4.29e-06, threshold=3.75e-04
* C_sig0.205 / P1 / density_l2_interior_mean [lm_noninf]: X=2.94e-03, Y=2.83e-03, u=9.06e-06, tol=9.06e-06, threshold=2.84e-03
* C_sig0.205 / P1 / negative_mass_max_post_initial [lm_noninf]: X=9.24e-02, Y=8.87e-02, u=1.54e-12, tol=1.54e-12, threshold=8.87e-02
* C_sig0.310 / P0 / negative_mass_max_post_initial [lm_noninf]: X=2.32e-02, Y=2.20e-02, u=1.54e-12, tol=1.54e-12, threshold=2.20e-02
* C_sig0.310 / P0 / calendar_raw_interior_frac_mean [lm_noninf]: X=5.76e-02, Y=5.65e-02, u=0.00e+00, tol=1.81e-04, threshold=5.67e-02
* C_sig0.310 / P1 / price_interior_rms [lm_noninf]: X=1.81e-04, Y=1.32e-04, u=3.44e-06, tol=3.44e-06, threshold=1.35e-04
* C_sig0.310 / P1 / density_l2_interior_mean [lm_noninf]: X=1.01e-03, Y=8.78e-04, u=3.77e-06, tol=3.77e-06, threshold=8.82e-04
* C_sig0.310 / P1 / calendar_raw_interior_frac_mean [lm_noninf]: X=6.54e-02, Y=5.51e-02, u=0.00e+00, tol=1.81e-04, threshold=5.53e-02
* C_sig0.310 / P1 / invvol_invalid_frac_full [lm_noninf]: X=1.86e-01, Y=1.71e-01, u=1.63e-04, tol=1.81e-04, threshold=1.72e-01
* C_sig0.310 / P1 / invvol_coverage [lm_noninf]: X=9.28e-01, Y=9.33e-01, u=0.00e+00, tol=1.81e-04, threshold=9.33e-01
* C_sig0.415 / P0 / negative_mass_max_post_initial [lm_noninf]: X=6.13e-03, Y=5.20e-03, u=1.54e-12, tol=1.54e-12, threshold=5.20e-03
* C_sig0.415 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.83e-02, Y=1.63e-02, u=0.00e+00, tol=1.81e-04, threshold=1.65e-02
* C_sig0.415 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=3.03e-02, Y=2.50e-02, u=0.00e+00, tol=1.81e-04, threshold=2.52e-02
* ... 430 more

### LR_m9_h16/WI — not `useful_closure` (213 exceptions)

* M_seed321 / P1 / price_interior_rms [price_beyond]: X=1.56e-04, Y=1.60e-04, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=4.26e-06
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=2.73e-01, Y=2.66e-01, u=0.00e+00, tol=1.81e-04, threshold=2.66e-01
* C_sig0.205 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=2.66e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P0 / invvol_invalid_frac_full [noninf]: X=4.09e-01, Y=3.52e-01, u=1.63e-04, tol=1.81e-04, threshold=3.53e-01
* C_sig0.205 / P0 / invvol_coverage [noninf]: X=8.72e-01, Y=8.85e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.205 / P1 / positivity_raw_interior_frac_mean [noninf]: X=2.38e-01, Y=2.35e-01, u=0.00e+00, tol=1.81e-04, threshold=2.35e-01
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=2.72e-01, Y=2.63e-01, u=0.00e+00, tol=1.81e-04, threshold=2.64e-01
* C_sig0.205 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=2.67e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P1 / invvol_invalid_frac_full [noninf]: X=4.07e-01, Y=3.75e-01, u=1.63e-04, tol=1.81e-04, threshold=3.75e-01
* C_sig0.205 / P1 / invvol_coverage [noninf]: X=8.72e-01, Y=8.84e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.310 / P0 / positivity_raw_interior_frac_mean [noninf]: X=7.30e-02, Y=7.17e-02, u=0.00e+00, tol=1.81e-04, threshold=7.19e-02
* C_sig0.310 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=9.91e-02, Y=8.77e-02, u=0.00e+00, tol=1.81e-04, threshold=8.79e-02
* C_sig0.310 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=1.07e-01, Y=8.61e-02, u=0.00e+00, tol=1.81e-04, threshold=8.62e-02
* C_sig0.310 / P0 / calendar_raw_interior_frac_mean [noninf]: X=5.76e-02, Y=5.60e-02, u=0.00e+00, tol=1.81e-04, threshold=5.62e-02
* C_sig0.310 / P0 / invvol_invalid_frac_full [noninf]: X=1.71e-01, Y=1.50e-01, u=1.63e-04, tol=1.81e-04, threshold=1.50e-01
* C_sig0.310 / P0 / invvol_coverage [noninf]: X=9.37e-01, Y=9.65e-01, u=0.00e+00, tol=1.81e-04, threshold=9.65e-01
* C_sig0.310 / P1 / negative_mass_max_post_initial [noninf]: X=1.53e-02, Y=1.29e-02, u=1.54e-12, tol=1.54e-12, threshold=1.29e-02
* C_sig0.310 / P1 / positivity_raw_interior_frac_mean [noninf]: X=7.28e-02, Y=6.76e-02, u=0.00e+00, tol=1.81e-04, threshold=6.78e-02
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=9.91e-02, Y=8.68e-02, u=0.00e+00, tol=1.81e-04, threshold=8.70e-02
* C_sig0.310 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=1.07e-01, Y=8.46e-02, u=0.00e+00, tol=1.81e-04, threshold=8.48e-02
* C_sig0.310 / P1 / calendar_raw_interior_frac_mean [noninf]: X=5.62e-02, Y=5.38e-02, u=0.00e+00, tol=1.81e-04, threshold=5.40e-02
* C_sig0.310 / P1 / invvol_invalid_frac_full [noninf]: X=1.68e-01, Y=1.51e-01, u=1.63e-04, tol=1.81e-04, threshold=1.52e-01
* C_sig0.310 / P1 / invvol_coverage [noninf]: X=9.37e-01, Y=9.66e-01, u=0.00e+00, tol=1.81e-04, threshold=9.66e-01
* C_sig0.415 / P0 / density_l2_interior_mean [noninf]: X=8.15e-04, Y=8.06e-04, u=1.83e-06, tol=1.83e-06, threshold=8.08e-04
* C_sig0.415 / P0 / negative_mass_max_post_initial [noninf]: X=6.13e-03, Y=3.76e-03, u=1.54e-12, tol=1.54e-12, threshold=3.76e-03
* C_sig0.415 / P0 / positivity_raw_interior_frac_mean [noninf]: X=1.83e-02, Y=1.59e-02, u=0.00e+00, tol=1.81e-04, threshold=1.61e-02
* C_sig0.415 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=3.03e-02, Y=2.25e-02, u=0.00e+00, tol=1.81e-04, threshold=2.26e-02
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=3.50e-02, Y=2.28e-02, u=0.00e+00, tol=1.81e-04, threshold=2.30e-02
* C_sig0.415 / P0 / calendar_raw_interior_frac_mean [noninf]: X=1.18e-02, Y=9.06e-03, u=0.00e+00, tol=1.81e-04, threshold=9.24e-03
* C_sig0.415 / P0 / invvol_invalid_frac_full [noninf]: X=6.45e-02, Y=5.45e-02, u=0.00e+00, tol=1.81e-04, threshold=5.47e-02
* ... 183 more

### LR_m9_h16/WI — not `stateful_advantage` (440 exceptions)

* C_sig0.310 / P1 / resolved_interior_rms [lm_edyn]: X=4.76e-05, Y=4.65e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-1.03e-06
* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=2.42e-05, Y=1.94e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-4.81e-06
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=2.44e-05, Y=1.92e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-5.13e-06
* M_seed321 / P1 / resolved_interior_rms [lm_edyn]: X=4.41e-05, Y=4.10e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-3.07e-06
* S2_seed121 / P0 / resolved_interior_rms [lm_edyn]: X=4.57e-05, Y=4.57e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-1.96e-09
* S2_seed121 / P1 / resolved_interior_rms [lm_edyn]: X=4.25e-05, Y=4.00e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-2.42e-06
* C_sig0.205 / P0 / negative_mass_max_post_initial [lm_noninf]: X=1.11e-01, Y=1.10e-01, u=1.54e-12, tol=1.54e-12, threshold=1.10e-01
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.73e-01, Y=2.70e-01, u=0.00e+00, tol=1.81e-04, threshold=2.71e-01
* C_sig0.205 / P1 / negative_mass_max_post_initial [lm_noninf]: X=9.07e-02, Y=8.87e-02, u=1.54e-12, tol=1.54e-12, threshold=8.87e-02
* C_sig0.310 / P0 / negative_mass_max_post_initial [lm_noninf]: X=2.32e-02, Y=2.20e-02, u=1.54e-12, tol=1.54e-12, threshold=2.20e-02
* C_sig0.310 / P0 / calendar_raw_interior_frac_mean [lm_noninf]: X=5.76e-02, Y=5.65e-02, u=0.00e+00, tol=1.81e-04, threshold=5.67e-02
* C_sig0.310 / P1 / negative_mass_max_post_initial [lm_noninf]: X=1.53e-02, Y=1.48e-02, u=1.54e-12, tol=1.54e-12, threshold=1.48e-02
* C_sig0.310 / P1 / calendar_raw_interior_frac_mean [lm_noninf]: X=5.62e-02, Y=5.51e-02, u=0.00e+00, tol=1.81e-04, threshold=5.53e-02
* C_sig0.415 / P0 / negative_mass_max_post_initial [lm_noninf]: X=6.13e-03, Y=5.20e-03, u=1.54e-12, tol=1.54e-12, threshold=5.20e-03
* C_sig0.415 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.83e-02, Y=1.63e-02, u=0.00e+00, tol=1.81e-04, threshold=1.65e-02
* C_sig0.415 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=3.03e-02, Y=2.50e-02, u=0.00e+00, tol=1.81e-04, threshold=2.52e-02
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=3.50e-02, Y=2.79e-02, u=0.00e+00, tol=1.81e-04, threshold=2.81e-02
* C_sig0.415 / P0 / calendar_raw_interior_frac_mean [lm_noninf]: X=1.18e-02, Y=1.11e-02, u=0.00e+00, tol=1.81e-04, threshold=1.12e-02
* C_sig0.415 / P0 / invvol_invalid_frac_full [lm_noninf]: X=6.45e-02, Y=5.79e-02, u=0.00e+00, tol=1.81e-04, threshold=5.81e-02
* C_sig0.415 / P0 / invvol_coverage [lm_noninf]: X=9.76e-01, Y=9.83e-01, u=0.00e+00, tol=1.81e-04, threshold=9.83e-01
* C_sig0.415 / P1 / negative_mass_max_post_initial [lm_noninf]: X=5.02e-03, Y=3.94e-03, u=1.54e-12, tol=1.54e-12, threshold=3.94e-03
* C_sig0.415 / P1 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.88e-02, Y=1.65e-02, u=0.00e+00, tol=1.81e-04, threshold=1.67e-02
* C_sig0.415 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=3.04e-02, Y=2.55e-02, u=0.00e+00, tol=1.81e-04, threshold=2.57e-02
* C_sig0.415 / P1 / butterfly_raw_interior_frac_mean [lm_noninf]: X=3.62e-02, Y=2.93e-02, u=0.00e+00, tol=1.81e-04, threshold=2.95e-02
* C_sig0.415 / P1 / calendar_raw_interior_frac_mean [lm_noninf]: X=1.67e-02, Y=1.63e-02, u=0.00e+00, tol=1.81e-04, threshold=1.65e-02
* C_sig0.415 / P1 / invvol_invalid_frac_full [lm_noninf]: X=6.66e-02, Y=6.07e-02, u=0.00e+00, tol=1.81e-04, threshold=6.09e-02
* C_sig0.415 / P1 / invvol_coverage [lm_noninf]: X=9.75e-01, Y=9.82e-01, u=0.00e+00, tol=1.81e-04, threshold=9.82e-01
* D2_seed220 / P0 / negative_mass_max_post_initial [lm_noninf]: X=6.48e-04, Y=9.76e-05, u=1.54e-12, tol=1.54e-12, threshold=9.76e-05
* D2_seed220 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=2.17e-03, Y=1.27e-03, u=0.00e+00, tol=1.81e-04, threshold=1.45e-03
* D2_seed220 / P0 / invvol_invalid_frac_full [lm_noninf]: X=1.66e-02, Y=1.30e-02, u=0.00e+00, tol=1.81e-04, threshold=1.32e-02
* ... 410 more

### LR_m9_h16/WU — not `useful_closure` (210 exceptions)

* M_seed321 / P1 / price_interior_rms [price_beyond]: X=1.56e-04, Y=1.60e-04, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=4.09e-06
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=2.73e-01, Y=2.66e-01, u=0.00e+00, tol=1.81e-04, threshold=2.66e-01
* C_sig0.205 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=2.66e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P0 / invvol_invalid_frac_full [noninf]: X=4.09e-01, Y=3.52e-01, u=1.63e-04, tol=1.81e-04, threshold=3.53e-01
* C_sig0.205 / P0 / invvol_coverage [noninf]: X=8.72e-01, Y=8.85e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.205 / P1 / positivity_raw_interior_frac_mean [noninf]: X=2.38e-01, Y=2.35e-01, u=0.00e+00, tol=1.81e-04, threshold=2.35e-01
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=2.72e-01, Y=2.63e-01, u=0.00e+00, tol=1.81e-04, threshold=2.64e-01
* C_sig0.205 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=2.67e-01, Y=2.54e-01, u=0.00e+00, tol=1.81e-04, threshold=2.54e-01
* C_sig0.205 / P1 / invvol_invalid_frac_full [noninf]: X=4.08e-01, Y=3.75e-01, u=1.63e-04, tol=1.81e-04, threshold=3.75e-01
* C_sig0.205 / P1 / invvol_coverage [noninf]: X=8.72e-01, Y=8.84e-01, u=0.00e+00, tol=1.81e-04, threshold=8.84e-01
* C_sig0.310 / P0 / positivity_raw_interior_frac_mean [noninf]: X=7.30e-02, Y=7.17e-02, u=0.00e+00, tol=1.81e-04, threshold=7.19e-02
* C_sig0.310 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=9.91e-02, Y=8.77e-02, u=0.00e+00, tol=1.81e-04, threshold=8.79e-02
* C_sig0.310 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=1.07e-01, Y=8.61e-02, u=0.00e+00, tol=1.81e-04, threshold=8.62e-02
* C_sig0.310 / P0 / calendar_raw_interior_frac_mean [noninf]: X=5.76e-02, Y=5.60e-02, u=0.00e+00, tol=1.81e-04, threshold=5.62e-02
* C_sig0.310 / P0 / invvol_invalid_frac_full [noninf]: X=1.71e-01, Y=1.50e-01, u=1.63e-04, tol=1.81e-04, threshold=1.50e-01
* C_sig0.310 / P0 / invvol_coverage [noninf]: X=9.37e-01, Y=9.65e-01, u=0.00e+00, tol=1.81e-04, threshold=9.65e-01
* C_sig0.310 / P1 / negative_mass_max_post_initial [noninf]: X=1.54e-02, Y=1.29e-02, u=1.54e-12, tol=1.54e-12, threshold=1.29e-02
* C_sig0.310 / P1 / positivity_raw_interior_frac_mean [noninf]: X=7.28e-02, Y=6.76e-02, u=0.00e+00, tol=1.81e-04, threshold=6.78e-02
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [noninf]: X=9.93e-02, Y=8.68e-02, u=0.00e+00, tol=1.81e-04, threshold=8.70e-02
* C_sig0.310 / P1 / butterfly_raw_interior_frac_mean [noninf]: X=1.07e-01, Y=8.46e-02, u=0.00e+00, tol=1.81e-04, threshold=8.48e-02
* C_sig0.310 / P1 / calendar_raw_interior_frac_mean [noninf]: X=5.54e-02, Y=5.38e-02, u=0.00e+00, tol=1.81e-04, threshold=5.40e-02
* C_sig0.310 / P1 / invvol_invalid_frac_full [noninf]: X=1.67e-01, Y=1.51e-01, u=1.63e-04, tol=1.81e-04, threshold=1.52e-01
* C_sig0.310 / P1 / invvol_coverage [noninf]: X=9.37e-01, Y=9.66e-01, u=0.00e+00, tol=1.81e-04, threshold=9.66e-01
* C_sig0.415 / P0 / density_l2_interior_mean [noninf]: X=8.15e-04, Y=8.06e-04, u=1.83e-06, tol=1.83e-06, threshold=8.08e-04
* C_sig0.415 / P0 / negative_mass_max_post_initial [noninf]: X=6.13e-03, Y=3.76e-03, u=1.54e-12, tol=1.54e-12, threshold=3.76e-03
* C_sig0.415 / P0 / positivity_raw_interior_frac_mean [noninf]: X=1.83e-02, Y=1.59e-02, u=0.00e+00, tol=1.81e-04, threshold=1.61e-02
* C_sig0.415 / P0 / monotonicity_raw_interior_frac_mean [noninf]: X=3.03e-02, Y=2.25e-02, u=0.00e+00, tol=1.81e-04, threshold=2.26e-02
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [noninf]: X=3.50e-02, Y=2.28e-02, u=0.00e+00, tol=1.81e-04, threshold=2.30e-02
* C_sig0.415 / P0 / calendar_raw_interior_frac_mean [noninf]: X=1.18e-02, Y=9.06e-03, u=0.00e+00, tol=1.81e-04, threshold=9.24e-03
* C_sig0.415 / P0 / invvol_invalid_frac_full [noninf]: X=6.45e-02, Y=5.45e-02, u=0.00e+00, tol=1.81e-04, threshold=5.47e-02
* ... 180 more

### LR_m9_h16/WU — not `stateful_advantage` (437 exceptions)

* C_sig0.310 / P1 / resolved_interior_rms [lm_edyn]: X=4.72e-05, Y=4.65e-05, u=1.43e-08, tol=1.43e-08, threshold=2.86e-08, gain=-6.33e-07
* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=2.42e-05, Y=1.94e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-4.81e-06
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=2.40e-05, Y=1.92e-05, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-4.76e-06
* M_seed321 / P1 / resolved_interior_rms [lm_edyn]: X=4.44e-05, Y=4.10e-05, u=2.39e-08, tol=2.39e-08, threshold=4.79e-08, gain=-3.46e-06
* S2_seed121 / P0 / resolved_interior_rms [lm_edyn]: X=4.57e-05, Y=4.57e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-1.96e-09
* S2_seed121 / P1 / resolved_interior_rms [lm_edyn]: X=4.20e-05, Y=4.00e-05, u=1.41e-08, tol=1.41e-08, threshold=2.81e-08, gain=-1.97e-06
* C_sig0.205 / P0 / negative_mass_max_post_initial [lm_noninf]: X=1.11e-01, Y=1.10e-01, u=1.54e-12, tol=1.54e-12, threshold=1.10e-01
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.73e-01, Y=2.70e-01, u=0.00e+00, tol=1.81e-04, threshold=2.71e-01
* C_sig0.205 / P1 / negative_mass_max_post_initial [lm_noninf]: X=9.06e-02, Y=8.87e-02, u=1.54e-12, tol=1.54e-12, threshold=8.87e-02
* C_sig0.310 / P0 / negative_mass_max_post_initial [lm_noninf]: X=2.32e-02, Y=2.20e-02, u=1.54e-12, tol=1.54e-12, threshold=2.20e-02
* C_sig0.310 / P0 / calendar_raw_interior_frac_mean [lm_noninf]: X=5.76e-02, Y=5.65e-02, u=0.00e+00, tol=1.81e-04, threshold=5.67e-02
* C_sig0.310 / P1 / negative_mass_max_post_initial [lm_noninf]: X=1.54e-02, Y=1.48e-02, u=1.54e-12, tol=1.54e-12, threshold=1.48e-02
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=9.93e-02, Y=9.89e-02, u=0.00e+00, tol=1.81e-04, threshold=9.91e-02
* C_sig0.310 / P1 / calendar_raw_interior_frac_mean [lm_noninf]: X=5.54e-02, Y=5.51e-02, u=0.00e+00, tol=1.81e-04, threshold=5.53e-02
* C_sig0.415 / P0 / negative_mass_max_post_initial [lm_noninf]: X=6.13e-03, Y=5.20e-03, u=1.54e-12, tol=1.54e-12, threshold=5.20e-03
* C_sig0.415 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.83e-02, Y=1.63e-02, u=0.00e+00, tol=1.81e-04, threshold=1.65e-02
* C_sig0.415 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=3.03e-02, Y=2.50e-02, u=0.00e+00, tol=1.81e-04, threshold=2.52e-02
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=3.50e-02, Y=2.79e-02, u=0.00e+00, tol=1.81e-04, threshold=2.81e-02
* C_sig0.415 / P0 / calendar_raw_interior_frac_mean [lm_noninf]: X=1.18e-02, Y=1.11e-02, u=0.00e+00, tol=1.81e-04, threshold=1.12e-02
* C_sig0.415 / P0 / invvol_invalid_frac_full [lm_noninf]: X=6.45e-02, Y=5.79e-02, u=0.00e+00, tol=1.81e-04, threshold=5.81e-02
* C_sig0.415 / P0 / invvol_coverage [lm_noninf]: X=9.76e-01, Y=9.83e-01, u=0.00e+00, tol=1.81e-04, threshold=9.83e-01
* C_sig0.415 / P1 / negative_mass_max_post_initial [lm_noninf]: X=4.93e-03, Y=3.94e-03, u=1.54e-12, tol=1.54e-12, threshold=3.94e-03
* C_sig0.415 / P1 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.88e-02, Y=1.65e-02, u=0.00e+00, tol=1.81e-04, threshold=1.67e-02
* C_sig0.415 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=3.03e-02, Y=2.55e-02, u=0.00e+00, tol=1.81e-04, threshold=2.57e-02
* C_sig0.415 / P1 / butterfly_raw_interior_frac_mean [lm_noninf]: X=3.57e-02, Y=2.93e-02, u=0.00e+00, tol=1.81e-04, threshold=2.95e-02
* C_sig0.415 / P1 / invvol_invalid_frac_full [lm_noninf]: X=6.62e-02, Y=6.07e-02, u=0.00e+00, tol=1.81e-04, threshold=6.09e-02
* C_sig0.415 / P1 / invvol_coverage [lm_noninf]: X=9.76e-01, Y=9.82e-01, u=0.00e+00, tol=1.81e-04, threshold=9.82e-01
* D2_seed220 / P0 / negative_mass_max_post_initial [lm_noninf]: X=6.48e-04, Y=9.76e-05, u=1.54e-12, tol=1.54e-12, threshold=9.76e-05
* D2_seed220 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=2.17e-03, Y=1.27e-03, u=0.00e+00, tol=1.81e-04, threshold=1.45e-03
* D2_seed220 / P0 / invvol_invalid_frac_full [lm_noninf]: X=1.66e-02, Y=1.30e-02, u=0.00e+00, tol=1.81e-04, threshold=1.32e-02
* ... 407 more

### QLp_m17 — not `resolved_dynamics_improvement` (34 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=2.94e-05, Y=2.94e-05, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=0.00e+00
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=7.81e-06, Y=7.81e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=0.00e+00
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=7.76e-06, Y=7.76e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=0.00e+00
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=2.28e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=0.00e+00
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=6.16e-06, Y=6.16e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=0.00e+00
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=4.91e-06, Y=4.91e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=0.00e+00
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=6.82e-06, Y=6.82e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=0.00e+00
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=7.08e-06, Y=7.08e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=0.00e+00
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=6.84e-06, Y=6.84e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=0.00e+00
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=6.60e-06, Y=6.60e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=0.00e+00
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=6.67e-06, Y=6.67e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=0.00e+00
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=6.56e-06, Y=6.56e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=0.00e+00
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=6.83e-06, Y=6.83e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=0.00e+00
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=7.16e-06, Y=7.16e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=0.00e+00
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=8.60e-06, Y=8.60e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=0.00e+00
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=3.02e-06, Y=3.02e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=0.00e+00
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.03e-05, Y=1.03e-05, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=0.00e+00
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=3.04e-06, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=0.00e+00
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=6.30e-06, Y=6.30e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=0.00e+00
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=6.78e-07, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=0.00e+00
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=1.93e-05, Y=1.93e-05, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=0.00e+00
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=3.56e-06, Y=3.56e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=0.00e+00
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=6.08e-06, Y=6.08e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=0.00e+00
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=4.20e-06, Y=4.20e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=0.00e+00
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=6.12e-06, Y=6.12e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=0.00e+00
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=2.67e-06, Y=2.67e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=0.00e+00
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=5.73e-06, Y=5.73e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=0.00e+00
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=3.85e-06, Y=3.85e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=0.00e+00
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=6.53e-06, Y=6.53e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=0.00e+00
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=3.24e-06, Y=3.24e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=0.00e+00
* ... 4 more

### QLp_m17 — not `useful_closure` (68 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=2.94e-05, Y=2.94e-05, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=0.00e+00
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=7.81e-06, Y=7.81e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=0.00e+00
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=7.76e-06, Y=7.76e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=0.00e+00
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=2.28e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=0.00e+00
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=6.16e-06, Y=6.16e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=0.00e+00
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=4.91e-06, Y=4.91e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=0.00e+00
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=6.82e-06, Y=6.82e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=0.00e+00
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=7.08e-06, Y=7.08e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=0.00e+00
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=6.84e-06, Y=6.84e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=0.00e+00
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=6.60e-06, Y=6.60e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=0.00e+00
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=6.67e-06, Y=6.67e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=0.00e+00
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=6.56e-06, Y=6.56e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=0.00e+00
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=6.83e-06, Y=6.83e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=0.00e+00
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=7.16e-06, Y=7.16e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=0.00e+00
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=8.60e-06, Y=8.60e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=0.00e+00
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=3.02e-06, Y=3.02e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=0.00e+00
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.03e-05, Y=1.03e-05, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=0.00e+00
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=3.04e-06, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=0.00e+00
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=6.30e-06, Y=6.30e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=0.00e+00
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=6.78e-07, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=0.00e+00
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=1.93e-05, Y=1.93e-05, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=0.00e+00
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=3.56e-06, Y=3.56e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=0.00e+00
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=6.08e-06, Y=6.08e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=0.00e+00
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=4.20e-06, Y=4.20e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=0.00e+00
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=6.12e-06, Y=6.12e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=0.00e+00
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=2.67e-06, Y=2.67e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=0.00e+00
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=5.73e-06, Y=5.73e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=0.00e+00
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=3.85e-06, Y=3.85e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=0.00e+00
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=6.53e-06, Y=6.53e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=0.00e+00
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=3.24e-06, Y=3.24e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=0.00e+00
* ... 38 more

### QLpm_m17 — not `resolved_dynamics_improvement` (34 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=1.66e-04, Y=2.94e-05, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-1.36e-04
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.15e-04, Y=7.81e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-1.08e-04
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=1.35e-04, Y=7.76e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-1.27e-04
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=9.44e-05, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-9.21e-05
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=1.15e-04, Y=6.16e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-1.09e-04
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=7.88e-05, Y=4.91e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-7.39e-05
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=7.81e-05, Y=6.82e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-7.13e-05
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=4.78e-05, Y=7.08e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.07e-05
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=8.81e-05, Y=6.84e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-8.13e-05
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=5.56e-05, Y=6.60e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-4.90e-05
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=8.64e-05, Y=6.67e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-7.98e-05
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=5.44e-05, Y=6.56e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-4.78e-05
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=7.63e-05, Y=6.83e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-6.95e-05
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=4.64e-05, Y=7.16e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-3.92e-05
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=1.29e-04, Y=8.60e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.20e-04
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=8.89e-05, Y=3.02e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-8.58e-05
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.28e-04, Y=1.03e-05, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-1.18e-04
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=8.82e-05, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-8.51e-05
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=1.45e-04, Y=6.30e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.39e-04
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.02e-04
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=1.48e-04, Y=1.93e-05, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.29e-04
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.04e-04, Y=3.56e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.00e-04
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=1.21e-04, Y=6.08e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-1.15e-04
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=8.33e-05, Y=4.20e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-7.91e-05
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=1.31e-04, Y=6.12e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-1.25e-04
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=9.09e-05, Y=2.67e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-8.82e-05
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=1.22e-04, Y=5.73e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-1.16e-04
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=8.40e-05, Y=3.85e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-8.01e-05
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=1.28e-04, Y=6.53e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-1.22e-04
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=8.93e-05, Y=3.24e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-8.60e-05
* ... 4 more

### QLpm_m17 — not `useful_closure` (201 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=1.66e-04, Y=2.94e-05, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-1.36e-04
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.15e-04, Y=7.81e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-1.08e-04
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=1.35e-04, Y=7.76e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-1.27e-04
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=9.44e-05, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-9.21e-05
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=1.15e-04, Y=6.16e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-1.09e-04
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=7.88e-05, Y=4.91e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-7.39e-05
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=7.81e-05, Y=6.82e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-7.13e-05
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=4.78e-05, Y=7.08e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.07e-05
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=8.81e-05, Y=6.84e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-8.13e-05
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=5.56e-05, Y=6.60e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-4.90e-05
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=8.64e-05, Y=6.67e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-7.98e-05
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=5.44e-05, Y=6.56e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-4.78e-05
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=7.63e-05, Y=6.83e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-6.95e-05
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=4.64e-05, Y=7.16e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-3.92e-05
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=1.29e-04, Y=8.60e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.20e-04
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=8.89e-05, Y=3.02e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-8.58e-05
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.28e-04, Y=1.03e-05, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-1.18e-04
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=8.82e-05, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-8.51e-05
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=1.45e-04, Y=6.30e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.39e-04
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.02e-04
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=1.48e-04, Y=1.93e-05, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.29e-04
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.04e-04, Y=3.56e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.00e-04
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=1.21e-04, Y=6.08e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-1.15e-04
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=8.33e-05, Y=4.20e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-7.91e-05
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=1.31e-04, Y=6.12e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-1.25e-04
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=9.09e-05, Y=2.67e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-8.82e-05
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=1.22e-04, Y=5.73e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-1.16e-04
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=8.40e-05, Y=3.85e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-8.01e-05
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=1.28e-04, Y=6.53e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-1.22e-04
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=8.93e-05, Y=3.24e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-8.60e-05
* ... 171 more

### QLs_m17 — not `resolved_dynamics_improvement` (34 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=1.66e-04, Y=2.94e-05, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-1.36e-04
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.15e-04, Y=7.81e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-1.08e-04
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=1.35e-04, Y=7.76e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-1.27e-04
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=9.44e-05, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-9.21e-05
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=1.15e-04, Y=6.16e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-1.09e-04
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=7.88e-05, Y=4.91e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-7.39e-05
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=7.81e-05, Y=6.82e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-7.13e-05
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=4.78e-05, Y=7.08e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.07e-05
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=8.81e-05, Y=6.84e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-8.13e-05
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=5.56e-05, Y=6.60e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-4.90e-05
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=8.64e-05, Y=6.67e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-7.98e-05
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=5.44e-05, Y=6.56e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-4.78e-05
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=7.63e-05, Y=6.83e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-6.95e-05
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=4.64e-05, Y=7.16e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-3.92e-05
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=1.29e-04, Y=8.60e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.20e-04
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=8.89e-05, Y=3.02e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-8.58e-05
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.28e-04, Y=1.03e-05, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-1.18e-04
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=8.82e-05, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-8.51e-05
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=1.45e-04, Y=6.30e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.39e-04
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.02e-04
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=1.48e-04, Y=1.93e-05, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.29e-04
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.04e-04, Y=3.56e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.00e-04
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=1.21e-04, Y=6.08e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-1.15e-04
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=8.33e-05, Y=4.20e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-7.91e-05
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=1.31e-04, Y=6.12e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-1.25e-04
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=9.09e-05, Y=2.67e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-8.82e-05
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=1.22e-04, Y=5.73e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-1.16e-04
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=8.40e-05, Y=3.85e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-8.01e-05
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=1.28e-04, Y=6.53e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-1.22e-04
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=8.93e-05, Y=3.24e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-8.60e-05
* ... 4 more

### QLs_m17 — not `useful_closure` (201 exceptions)

* C_sig0.205 / P0 / resolved_interior_rms [edyn_beyond]: X=1.66e-04, Y=2.94e-05, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-1.36e-04
* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.15e-04, Y=7.81e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-1.08e-04
* C_sig0.310 / P0 / resolved_interior_rms [edyn_beyond]: X=1.35e-04, Y=7.76e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-1.27e-04
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=9.44e-05, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-9.21e-05
* C_sig0.415 / P0 / resolved_interior_rms [edyn_beyond]: X=1.15e-04, Y=6.16e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-1.09e-04
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=7.88e-05, Y=4.91e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-7.39e-05
* D2_seed220 / P0 / resolved_interior_rms [edyn_beyond]: X=7.81e-05, Y=6.82e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-7.13e-05
* D2_seed220 / P1 / resolved_interior_rms [edyn_beyond]: X=4.78e-05, Y=7.08e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-4.07e-05
* D2_seed221 / P0 / resolved_interior_rms [edyn_beyond]: X=8.81e-05, Y=6.84e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-8.13e-05
* D2_seed221 / P1 / resolved_interior_rms [edyn_beyond]: X=5.56e-05, Y=6.60e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-4.90e-05
* D2_seed222 / P0 / resolved_interior_rms [edyn_beyond]: X=8.64e-05, Y=6.67e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-7.98e-05
* D2_seed222 / P1 / resolved_interior_rms [edyn_beyond]: X=5.44e-05, Y=6.56e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-4.78e-05
* D2_seed223 / P0 / resolved_interior_rms [edyn_beyond]: X=7.63e-05, Y=6.83e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-6.95e-05
* D2_seed223 / P1 / resolved_interior_rms [edyn_beyond]: X=4.64e-05, Y=7.16e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-3.92e-05
* M_seed320 / P0 / resolved_interior_rms [edyn_beyond]: X=1.29e-04, Y=8.60e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.20e-04
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=8.89e-05, Y=3.02e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-8.58e-05
* M_seed321 / P0 / resolved_interior_rms [edyn_beyond]: X=1.28e-04, Y=1.03e-05, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-1.18e-04
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=8.82e-05, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-8.51e-05
* M_seed322 / P0 / resolved_interior_rms [edyn_beyond]: X=1.45e-04, Y=6.30e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.39e-04
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-04, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.02e-04
* M_seed323 / P0 / resolved_interior_rms [edyn_beyond]: X=1.48e-04, Y=1.93e-05, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.29e-04
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.04e-04, Y=3.56e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.00e-04
* S2_seed120 / P0 / resolved_interior_rms [edyn_beyond]: X=1.21e-04, Y=6.08e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-1.15e-04
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=8.33e-05, Y=4.20e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-7.91e-05
* S2_seed121 / P0 / resolved_interior_rms [edyn_beyond]: X=1.31e-04, Y=6.12e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-1.25e-04
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=9.09e-05, Y=2.67e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-8.82e-05
* S2_seed122 / P0 / resolved_interior_rms [edyn_beyond]: X=1.22e-04, Y=5.73e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-1.16e-04
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=8.40e-05, Y=3.85e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-8.01e-05
* S2_seed123 / P0 / resolved_interior_rms [edyn_beyond]: X=1.28e-04, Y=6.53e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-1.22e-04
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=8.93e-05, Y=3.24e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-8.60e-05
* ... 171 more

### LM_m17 — not `resolved_dynamics_improvement` (2 exceptions)

* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=2.46e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-1.80e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.31e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-6.36e-07

### LM_m17 — not `useful_closure` (186 exceptions)

* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=2.46e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-1.80e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.31e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-6.36e-07
* C_sig0.205 / P0 / price_interior_rms [price_beyond]: X=2.71e-05, Y=3.13e-05, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=4.18e-06
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=1.93e-05, Y=1.91e-05, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=-1.14e-07
* C_sig0.310 / P0 / price_interior_rms [price_beyond]: X=7.39e-06, Y=9.64e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=2.25e-06
* C_sig0.310 / P1 / price_interior_rms [price_beyond]: X=6.56e-06, Y=4.80e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=-1.76e-06
* C_sig0.415 / P0 / price_interior_rms [price_beyond]: X=9.66e-06, Y=8.41e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-1.25e-06
* C_sig0.415 / P1 / price_interior_rms [price_beyond]: X=8.70e-06, Y=6.30e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-2.40e-06
* D2_seed220 / P0 / price_interior_rms [price_beyond]: X=7.79e-06, Y=7.31e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-4.85e-07
* D2_seed220 / P1 / price_interior_rms [price_beyond]: X=6.68e-06, Y=6.17e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-5.15e-07
* D2_seed221 / P0 / price_interior_rms [price_beyond]: X=8.62e-06, Y=7.86e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-7.61e-07
* D2_seed221 / P1 / price_interior_rms [price_beyond]: X=7.41e-06, Y=6.19e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-1.22e-06
* D2_seed222 / P0 / price_interior_rms [price_beyond]: X=9.02e-06, Y=7.84e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.18e-06
* D2_seed222 / P1 / price_interior_rms [price_beyond]: X=7.60e-06, Y=6.14e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.46e-06
* D2_seed223 / P0 / price_interior_rms [price_beyond]: X=7.91e-06, Y=7.31e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-6.01e-07
* D2_seed223 / P1 / price_interior_rms [price_beyond]: X=6.71e-06, Y=6.20e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-5.05e-07
* M_seed320 / P0 / price_interior_rms [price_beyond]: X=6.31e-06, Y=1.01e-05, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=3.76e-06
* M_seed320 / P1 / price_interior_rms [price_beyond]: X=6.15e-06, Y=5.38e-06, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=-7.64e-07
* M_seed321 / P0 / price_interior_rms [price_beyond]: X=8.26e-06, Y=1.14e-05, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=3.12e-06
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=8.06e-06, Y=6.41e-06, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=-1.65e-06
* M_seed322 / P0 / price_interior_rms [price_beyond]: X=4.48e-06, Y=7.66e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=3.18e-06
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=4.30e-06, Y=3.90e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-3.97e-07
* M_seed323 / P0 / price_interior_rms [price_beyond]: X=1.33e-05, Y=2.01e-05, u=3.83e-06, tol=3.83e-06, threshold=7.66e-06, gain=6.80e-06
* M_seed323 / P1 / price_interior_rms [price_beyond]: X=1.14e-05, Y=1.14e-05, u=3.83e-06, tol=3.83e-06, threshold=7.66e-06, gain=-3.31e-08
* S2_seed120 / P0 / price_interior_rms [price_beyond]: X=9.03e-06, Y=8.43e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-6.05e-07
* S2_seed120 / P1 / price_interior_rms [price_beyond]: X=8.01e-06, Y=5.95e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-2.07e-06
* S2_seed121 / P0 / price_interior_rms [price_beyond]: X=7.29e-06, Y=8.31e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=1.02e-06
* S2_seed121 / P1 / price_interior_rms [price_beyond]: X=6.26e-06, Y=4.98e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=-1.28e-06
* S2_seed122 / P0 / price_interior_rms [price_beyond]: X=8.53e-06, Y=8.12e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-4.12e-07
* S2_seed122 / P1 / price_interior_rms [price_beyond]: X=7.60e-06, Y=5.73e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-1.87e-06
* ... 156 more

### LMb_m17 — not `resolved_dynamics_improvement` (2 exceptions)

* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=2.50e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-2.15e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.40e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-7.26e-07

### LMb_m17 — not `useful_closure` (188 exceptions)

* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=2.50e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-2.15e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.40e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-7.26e-07
* C_sig0.205 / P0 / price_interior_rms [price_beyond]: X=2.71e-05, Y=3.13e-05, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=4.18e-06
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=1.93e-05, Y=1.91e-05, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=-1.26e-07
* C_sig0.310 / P0 / price_interior_rms [price_beyond]: X=7.41e-06, Y=9.64e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=2.24e-06
* C_sig0.310 / P1 / price_interior_rms [price_beyond]: X=6.58e-06, Y=4.80e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=-1.78e-06
* C_sig0.415 / P0 / price_interior_rms [price_beyond]: X=9.62e-06, Y=8.41e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-1.21e-06
* C_sig0.415 / P1 / price_interior_rms [price_beyond]: X=8.66e-06, Y=6.30e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-2.36e-06
* D2_seed220 / P0 / price_interior_rms [price_beyond]: X=7.88e-06, Y=7.31e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-5.66e-07
* D2_seed220 / P1 / price_interior_rms [price_beyond]: X=6.78e-06, Y=6.17e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-6.09e-07
* D2_seed221 / P0 / price_interior_rms [price_beyond]: X=8.68e-06, Y=7.86e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-8.17e-07
* D2_seed221 / P1 / price_interior_rms [price_beyond]: X=7.47e-06, Y=6.19e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-1.28e-06
* D2_seed222 / P0 / price_interior_rms [price_beyond]: X=9.08e-06, Y=7.84e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.24e-06
* D2_seed222 / P1 / price_interior_rms [price_beyond]: X=7.67e-06, Y=6.14e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.53e-06
* D2_seed223 / P0 / price_interior_rms [price_beyond]: X=8.00e-06, Y=7.31e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-6.89e-07
* D2_seed223 / P1 / price_interior_rms [price_beyond]: X=6.81e-06, Y=6.20e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-6.08e-07
* M_seed320 / P0 / price_interior_rms [price_beyond]: X=6.30e-06, Y=1.01e-05, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=3.77e-06
* M_seed320 / P1 / price_interior_rms [price_beyond]: X=6.13e-06, Y=5.38e-06, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=-7.49e-07
* M_seed321 / P0 / price_interior_rms [price_beyond]: X=8.20e-06, Y=1.14e-05, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=3.18e-06
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=8.00e-06, Y=6.41e-06, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=-1.59e-06
* M_seed322 / P0 / price_interior_rms [price_beyond]: X=4.51e-06, Y=7.66e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=3.14e-06
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=4.34e-06, Y=3.90e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-4.41e-07
* M_seed323 / P0 / price_interior_rms [price_beyond]: X=1.33e-05, Y=2.01e-05, u=3.83e-06, tol=3.83e-06, threshold=7.66e-06, gain=6.77e-06
* M_seed323 / P1 / price_interior_rms [price_beyond]: X=1.15e-05, Y=1.14e-05, u=3.83e-06, tol=3.83e-06, threshold=7.66e-06, gain=-7.57e-08
* S2_seed120 / P0 / price_interior_rms [price_beyond]: X=9.01e-06, Y=8.43e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-5.86e-07
* S2_seed120 / P1 / price_interior_rms [price_beyond]: X=7.99e-06, Y=5.95e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-2.05e-06
* S2_seed121 / P0 / price_interior_rms [price_beyond]: X=7.29e-06, Y=8.31e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=1.01e-06
* S2_seed121 / P1 / price_interior_rms [price_beyond]: X=6.27e-06, Y=4.98e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=-1.29e-06
* S2_seed122 / P0 / price_interior_rms [price_beyond]: X=8.51e-06, Y=8.12e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-3.93e-07
* S2_seed122 / P1 / price_interior_rms [price_beyond]: X=7.58e-06, Y=5.73e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-1.85e-06
* ... 158 more

### LR0_m17_h8/W0 — not `resolved_dynamics_improvement` (12 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.64e-05, Y=7.81e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-8.59e-06
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=8.56e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-6.28e-06
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=9.08e-06, Y=3.02e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-6.06e-06
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=7.42e-06, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-4.39e-06
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-05, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-9.60e-06
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=7.18e-06, Y=3.56e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-3.62e-06
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=4.99e-06, Y=4.20e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-7.91e-07
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=6.32e-06, Y=2.67e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-3.65e-06
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=5.16e-06, Y=3.85e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-1.30e-06
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=6.64e-06, Y=3.24e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-3.40e-06
* S2_seed124 / P1 / resolved_interior_rms [edyn_beyond]: X=5.22e-06, Y=2.81e-06, u=1.31e-09, tol=1.31e-09, threshold=2.63e-09, gain=-2.41e-06
* S2_seed125 / P1 / resolved_interior_rms [edyn_beyond]: X=5.33e-06, Y=3.46e-06, u=1.57e-09, tol=1.57e-09, threshold=3.15e-09, gain=-1.87e-06

### LR0_m17_h8/W0 — not `useful_closure` (217 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=1.64e-05, Y=7.81e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-8.59e-06
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=8.56e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-6.28e-06
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=9.08e-06, Y=3.02e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-6.06e-06
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=7.42e-06, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-4.39e-06
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=1.03e-05, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-9.60e-06
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=7.18e-06, Y=3.56e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-3.62e-06
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=4.99e-06, Y=4.20e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-7.91e-07
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=6.32e-06, Y=2.67e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-3.65e-06
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=5.16e-06, Y=3.85e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-1.30e-06
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=6.64e-06, Y=3.24e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-3.40e-06
* S2_seed124 / P1 / resolved_interior_rms [edyn_beyond]: X=5.22e-06, Y=2.81e-06, u=1.31e-09, tol=1.31e-09, threshold=2.63e-09, gain=-2.41e-06
* S2_seed125 / P1 / resolved_interior_rms [edyn_beyond]: X=5.33e-06, Y=3.46e-06, u=1.57e-09, tol=1.57e-09, threshold=3.15e-09, gain=-1.87e-06
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=2.57e-05, Y=1.91e-05, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=-6.59e-06
* C_sig0.310 / P0 / price_interior_rms [price_beyond]: X=7.44e-06, Y=9.64e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=2.20e-06
* C_sig0.310 / P1 / price_interior_rms [price_beyond]: X=1.04e-05, Y=4.80e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=-5.62e-06
* C_sig0.415 / P0 / price_interior_rms [price_beyond]: X=9.70e-06, Y=8.41e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-1.29e-06
* C_sig0.415 / P1 / price_interior_rms [price_beyond]: X=9.80e-06, Y=6.30e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-3.50e-06
* D2_seed220 / P0 / price_interior_rms [price_beyond]: X=7.75e-06, Y=7.31e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-4.40e-07
* D2_seed220 / P1 / price_interior_rms [price_beyond]: X=6.78e-06, Y=6.17e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-6.09e-07
* D2_seed221 / P0 / price_interior_rms [price_beyond]: X=8.56e-06, Y=7.86e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-7.01e-07
* D2_seed221 / P1 / price_interior_rms [price_beyond]: X=7.52e-06, Y=6.19e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-1.32e-06
* D2_seed222 / P0 / price_interior_rms [price_beyond]: X=8.93e-06, Y=7.84e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.09e-06
* D2_seed222 / P1 / price_interior_rms [price_beyond]: X=7.69e-06, Y=6.14e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.56e-06
* D2_seed223 / P0 / price_interior_rms [price_beyond]: X=7.85e-06, Y=7.31e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-5.47e-07
* D2_seed223 / P1 / price_interior_rms [price_beyond]: X=6.80e-06, Y=6.20e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-6.03e-07
* M_seed320 / P0 / price_interior_rms [price_beyond]: X=6.26e-06, Y=1.01e-05, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=3.81e-06
* M_seed320 / P1 / price_interior_rms [price_beyond]: X=1.08e-05, Y=5.38e-06, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=-5.44e-06
* M_seed321 / P0 / price_interior_rms [price_beyond]: X=8.00e-06, Y=1.14e-05, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=3.37e-06
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=1.07e-05, Y=6.41e-06, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=-4.29e-06
* M_seed322 / P0 / price_interior_rms [price_beyond]: X=4.82e-06, Y=7.66e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=2.83e-06
* ... 187 more

### LR0_m17_h8/W0 — not `stateful_advantage` (452 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [lm_edyn]: X=1.64e-05, Y=4.75e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-1.16e-05
* C_sig0.310 / P0 / resolved_interior_rms [lm_edyn]: X=3.53e-06, Y=3.47e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-5.63e-08
* C_sig0.310 / P1 / resolved_interior_rms [lm_edyn]: X=8.56e-06, Y=2.46e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-6.10e-06
* C_sig0.415 / P1 / resolved_interior_rms [lm_edyn]: X=4.41e-06, Y=1.16e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-3.25e-06
* D2_seed220 / P1 / resolved_interior_rms [lm_edyn]: X=4.09e-06, Y=3.42e-06, u=6.40e-08, tol=6.40e-08, threshold=1.28e-07, gain=-6.71e-07
* D2_seed221 / P1 / resolved_interior_rms [lm_edyn]: X=3.38e-06, Y=2.71e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-6.65e-07
* D2_seed222 / P1 / resolved_interior_rms [lm_edyn]: X=3.59e-06, Y=2.67e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-9.18e-07
* D2_seed223 / P1 / resolved_interior_rms [lm_edyn]: X=4.24e-06, Y=3.46e-06, u=6.31e-08, tol=6.31e-08, threshold=1.26e-07, gain=-7.76e-07
* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=1.14e-06, Y=1.03e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.08e-07
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=9.08e-06, Y=6.13e-07, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-8.46e-06
* M_seed321 / P1 / resolved_interior_rms [lm_edyn]: X=7.42e-06, Y=2.85e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-4.57e-06
* M_seed322 / P0 / resolved_interior_rms [lm_edyn]: X=2.33e-06, Y=1.62e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-7.17e-07
* M_seed322 / P1 / resolved_interior_rms [lm_edyn]: X=1.03e-05, Y=1.31e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-8.96e-06
* M_seed323 / P1 / resolved_interior_rms [lm_edyn]: X=7.18e-06, Y=2.65e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-4.53e-06
* S2_seed120 / P1 / resolved_interior_rms [lm_edyn]: X=4.99e-06, Y=1.21e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-3.79e-06
* S2_seed121 / P1 / resolved_interior_rms [lm_edyn]: X=6.32e-06, Y=1.21e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-5.11e-06
* S2_seed122 / P1 / resolved_interior_rms [lm_edyn]: X=5.16e-06, Y=1.15e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-4.01e-06
* S2_seed123 / P0 / resolved_interior_rms [lm_edyn]: X=3.31e-06, Y=3.31e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-2.79e-10
* S2_seed123 / P1 / resolved_interior_rms [lm_edyn]: X=6.64e-06, Y=1.80e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-4.84e-06
* S2_seed124 / P1 / resolved_interior_rms [lm_edyn]: X=5.22e-06, Y=9.34e-07, u=1.31e-09, tol=1.31e-09, threshold=2.63e-09, gain=-4.29e-06
* S2_seed125 / P1 / resolved_interior_rms [lm_edyn]: X=5.33e-06, Y=1.22e-06, u=1.57e-09, tol=1.57e-09, threshold=3.15e-09, gain=-4.11e-06
* C_sig0.205 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.15e-01, Y=1.10e-01, u=0.00e+00, tol=1.81e-04, threshold=1.10e-01
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.50e-01, Y=1.41e-01, u=0.00e+00, tol=1.81e-04, threshold=1.41e-01
* C_sig0.205 / P0 / calendar_raw_interior_frac_mean [lm_noninf]: X=1.24e-01, Y=1.22e-01, u=0.00e+00, tol=1.81e-04, threshold=1.22e-01
* C_sig0.205 / P1 / price_interior_rms [lm_noninf]: X=2.57e-05, Y=1.93e-05, u=4.29e-06, tol=4.29e-06, threshold=2.35e-05
* C_sig0.205 / P1 / density_l2_interior_mean [lm_noninf]: X=3.28e-04, Y=2.48e-04, u=9.06e-06, tol=9.06e-06, threshold=2.57e-04
* C_sig0.205 / P1 / negative_mass_max_post_initial [lm_noninf]: X=2.33e-02, Y=1.28e-02, u=1.54e-12, tol=1.54e-12, threshold=1.28e-02
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.46e-01, Y=1.43e-01, u=0.00e+00, tol=1.81e-04, threshold=1.43e-01
* C_sig0.205 / P1 / butterfly_raw_interior_frac_mean [lm_noninf]: X=1.71e-01, Y=1.70e-01, u=0.00e+00, tol=1.81e-04, threshold=1.70e-01
* C_sig0.205 / P1 / calendar_raw_interior_frac_mean [lm_noninf]: X=1.24e-01, Y=1.18e-01, u=0.00e+00, tol=1.81e-04, threshold=1.18e-01
* ... 422 more

### LR0_m17_h8/WI — not `resolved_dynamics_improvement` (2 exceptions)

* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=2.68e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-3.95e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=2.28e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.60e-06

### LR0_m17_h8/WI — not `useful_closure` (175 exceptions)

* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=2.68e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-3.95e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=2.28e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.60e-06
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=1.86e-05, Y=1.91e-05, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=5.75e-07
* C_sig0.310 / P0 / price_interior_rms [price_beyond]: X=7.44e-06, Y=9.64e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=2.20e-06
* C_sig0.310 / P1 / price_interior_rms [price_beyond]: X=6.70e-06, Y=4.80e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=-1.90e-06
* C_sig0.415 / P0 / price_interior_rms [price_beyond]: X=9.70e-06, Y=8.41e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-1.29e-06
* C_sig0.415 / P1 / price_interior_rms [price_beyond]: X=8.88e-06, Y=6.30e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-2.58e-06
* D2_seed220 / P0 / price_interior_rms [price_beyond]: X=7.75e-06, Y=7.31e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-4.40e-07
* D2_seed220 / P1 / price_interior_rms [price_beyond]: X=6.75e-06, Y=6.17e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-5.82e-07
* D2_seed221 / P0 / price_interior_rms [price_beyond]: X=8.56e-06, Y=7.86e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-7.01e-07
* D2_seed221 / P1 / price_interior_rms [price_beyond]: X=7.45e-06, Y=6.19e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-1.26e-06
* D2_seed222 / P0 / price_interior_rms [price_beyond]: X=8.93e-06, Y=7.84e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.09e-06
* D2_seed222 / P1 / price_interior_rms [price_beyond]: X=7.64e-06, Y=6.14e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.50e-06
* D2_seed223 / P0 / price_interior_rms [price_beyond]: X=7.85e-06, Y=7.31e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-5.47e-07
* D2_seed223 / P1 / price_interior_rms [price_beyond]: X=6.77e-06, Y=6.20e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-5.74e-07
* M_seed320 / P0 / price_interior_rms [price_beyond]: X=6.26e-06, Y=1.01e-05, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=3.81e-06
* M_seed320 / P1 / price_interior_rms [price_beyond]: X=6.09e-06, Y=5.38e-06, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=-7.03e-07
* M_seed321 / P0 / price_interior_rms [price_beyond]: X=8.00e-06, Y=1.14e-05, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=3.37e-06
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=7.91e-06, Y=6.41e-06, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=-1.50e-06
* M_seed322 / P0 / price_interior_rms [price_beyond]: X=4.82e-06, Y=7.66e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=2.83e-06
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=4.74e-06, Y=3.90e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-8.37e-07
* M_seed323 / P1 / price_interior_rms [price_beyond]: X=1.16e-05, Y=1.14e-05, u=3.83e-06, tol=3.83e-06, threshold=7.66e-06, gain=-1.45e-07
* S2_seed120 / P0 / price_interior_rms [price_beyond]: X=9.09e-06, Y=8.43e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-6.62e-07
* S2_seed120 / P1 / price_interior_rms [price_beyond]: X=8.20e-06, Y=5.95e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-2.25e-06
* S2_seed121 / P0 / price_interior_rms [price_beyond]: X=7.31e-06, Y=8.31e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=9.93e-07
* S2_seed121 / P1 / price_interior_rms [price_beyond]: X=6.40e-06, Y=4.98e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=-1.42e-06
* S2_seed122 / P0 / price_interior_rms [price_beyond]: X=8.57e-06, Y=8.12e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-4.54e-07
* S2_seed122 / P1 / price_interior_rms [price_beyond]: X=7.76e-06, Y=5.73e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-2.03e-06
* S2_seed123 / P0 / price_interior_rms [price_beyond]: X=8.38e-06, Y=8.80e-06, u=3.28e-06, tol=3.28e-06, threshold=6.57e-06, gain=4.21e-07
* S2_seed123 / P1 / price_interior_rms [price_beyond]: X=7.50e-06, Y=5.42e-06, u=3.28e-06, tol=3.28e-06, threshold=6.57e-06, gain=-2.08e-06
* ... 145 more

### LR0_m17_h8/WI — not `stateful_advantage` (356 exceptions)

* C_sig0.310 / P0 / resolved_interior_rms [lm_edyn]: X=3.53e-06, Y=3.47e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-5.63e-08
* C_sig0.310 / P1 / resolved_interior_rms [lm_edyn]: X=2.68e-06, Y=2.46e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-2.16e-07
* C_sig0.415 / P1 / resolved_interior_rms [lm_edyn]: X=1.31e-06, Y=1.16e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-1.48e-07
* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=1.14e-06, Y=1.03e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.08e-07
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=7.13e-07, Y=6.13e-07, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.00e-07
* M_seed322 / P0 / resolved_interior_rms [lm_edyn]: X=2.33e-06, Y=1.62e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-7.17e-07
* M_seed322 / P1 / resolved_interior_rms [lm_edyn]: X=2.28e-06, Y=1.31e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-9.63e-07
* M_seed323 / P1 / resolved_interior_rms [lm_edyn]: X=3.20e-06, Y=2.65e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-5.48e-07
* S2_seed120 / P1 / resolved_interior_rms [lm_edyn]: X=1.44e-06, Y=1.21e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-2.32e-07
* S2_seed121 / P1 / resolved_interior_rms [lm_edyn]: X=1.47e-06, Y=1.21e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-2.58e-07
* S2_seed122 / P1 / resolved_interior_rms [lm_edyn]: X=1.34e-06, Y=1.15e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-1.89e-07
* S2_seed123 / P0 / resolved_interior_rms [lm_edyn]: X=3.31e-06, Y=3.31e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-2.79e-10
* S2_seed123 / P1 / resolved_interior_rms [lm_edyn]: X=2.07e-06, Y=1.80e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-2.75e-07
* C_sig0.205 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.15e-01, Y=1.10e-01, u=0.00e+00, tol=1.81e-04, threshold=1.10e-01
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.50e-01, Y=1.41e-01, u=0.00e+00, tol=1.81e-04, threshold=1.41e-01
* C_sig0.205 / P0 / calendar_raw_interior_frac_mean [lm_noninf]: X=1.24e-01, Y=1.22e-01, u=0.00e+00, tol=1.81e-04, threshold=1.22e-01
* C_sig0.205 / P1 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.32e-01, Y=1.29e-01, u=0.00e+00, tol=1.81e-04, threshold=1.29e-01
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.51e-01, Y=1.43e-01, u=0.00e+00, tol=1.81e-04, threshold=1.43e-01
* C_sig0.205 / P1 / calendar_raw_interior_frac_mean [lm_noninf]: X=1.22e-01, Y=1.18e-01, u=0.00e+00, tol=1.81e-04, threshold=1.18e-01
* C_sig0.310 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=5.69e-02, Y=4.84e-02, u=0.00e+00, tol=1.81e-04, threshold=4.86e-02
* C_sig0.310 / P1 / density_l2_interior_mean [lm_noninf]: X=1.13e-04, Y=1.08e-04, u=3.77e-06, tol=3.77e-06, threshold=1.12e-04
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.61e-02, Y=2.55e-02, u=0.00e+00, tol=1.81e-04, threshold=2.57e-02
* C_sig0.310 / P1 / butterfly_raw_interior_frac_mean [lm_noninf]: X=5.53e-02, Y=4.73e-02, u=0.00e+00, tol=1.81e-04, threshold=4.75e-02
* C_sig0.415 / P0 / density_l2_interior_mean [lm_noninf]: X=1.83e-04, Y=1.80e-04, u=1.83e-06, tol=1.83e-06, threshold=1.82e-04
* C_sig0.415 / P0 / negative_mass_max_post_initial [lm_noninf]: X=3.81e-05, Y=2.58e-06, u=1.54e-12, tol=1.54e-12, threshold=2.58e-06
* C_sig0.415 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.63e-03, Y=0.00e+00, u=0.00e+00, tol=1.81e-04, threshold=1.81e-04
* C_sig0.415 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.63e-03, Y=0.00e+00, u=0.00e+00, tol=1.81e-04, threshold=1.81e-04
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=2.72e-03, Y=1.63e-03, u=0.00e+00, tol=1.81e-04, threshold=1.81e-03
* C_sig0.415 / P0 / invvol_invalid_frac_full [lm_noninf]: X=6.79e-02, Y=6.74e-02, u=0.00e+00, tol=1.81e-04, threshold=6.76e-02
* C_sig0.415 / P0 / invvol_coverage [lm_noninf]: X=9.93e-01, Y=9.94e-01, u=0.00e+00, tol=1.81e-04, threshold=9.94e-01
* ... 326 more

### LR0_m17_h8/WU — not `resolved_dynamics_improvement` (2 exceptions)

* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=2.68e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-3.97e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=2.27e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.59e-06

### LR0_m17_h8/WU — not `useful_closure` (174 exceptions)

* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=2.68e-06, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-3.97e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=2.27e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.59e-06
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=1.86e-05, Y=1.91e-05, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=5.74e-07
* C_sig0.310 / P0 / price_interior_rms [price_beyond]: X=7.44e-06, Y=9.64e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=2.20e-06
* C_sig0.310 / P1 / price_interior_rms [price_beyond]: X=6.70e-06, Y=4.80e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=-1.90e-06
* C_sig0.415 / P0 / price_interior_rms [price_beyond]: X=9.70e-06, Y=8.41e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-1.29e-06
* C_sig0.415 / P1 / price_interior_rms [price_beyond]: X=8.88e-06, Y=6.30e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-2.58e-06
* D2_seed220 / P0 / price_interior_rms [price_beyond]: X=7.75e-06, Y=7.31e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-4.40e-07
* D2_seed220 / P1 / price_interior_rms [price_beyond]: X=6.75e-06, Y=6.17e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-5.82e-07
* D2_seed221 / P0 / price_interior_rms [price_beyond]: X=8.56e-06, Y=7.86e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-7.01e-07
* D2_seed221 / P1 / price_interior_rms [price_beyond]: X=7.45e-06, Y=6.19e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-1.26e-06
* D2_seed222 / P0 / price_interior_rms [price_beyond]: X=8.93e-06, Y=7.84e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.09e-06
* D2_seed222 / P1 / price_interior_rms [price_beyond]: X=7.64e-06, Y=6.14e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.50e-06
* D2_seed223 / P0 / price_interior_rms [price_beyond]: X=7.85e-06, Y=7.31e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-5.47e-07
* D2_seed223 / P1 / price_interior_rms [price_beyond]: X=6.78e-06, Y=6.20e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-5.75e-07
* M_seed320 / P0 / price_interior_rms [price_beyond]: X=6.26e-06, Y=1.01e-05, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=3.81e-06
* M_seed320 / P1 / price_interior_rms [price_beyond]: X=6.09e-06, Y=5.38e-06, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=-7.03e-07
* M_seed321 / P0 / price_interior_rms [price_beyond]: X=8.00e-06, Y=1.14e-05, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=3.37e-06
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=7.92e-06, Y=6.41e-06, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=-1.51e-06
* M_seed322 / P0 / price_interior_rms [price_beyond]: X=4.82e-06, Y=7.66e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=2.83e-06
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=4.73e-06, Y=3.90e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-8.32e-07
* M_seed323 / P1 / price_interior_rms [price_beyond]: X=1.16e-05, Y=1.14e-05, u=3.83e-06, tol=3.83e-06, threshold=7.66e-06, gain=-1.58e-07
* S2_seed120 / P0 / price_interior_rms [price_beyond]: X=9.09e-06, Y=8.43e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-6.62e-07
* S2_seed120 / P1 / price_interior_rms [price_beyond]: X=8.20e-06, Y=5.95e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-2.25e-06
* S2_seed121 / P0 / price_interior_rms [price_beyond]: X=7.31e-06, Y=8.31e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=9.93e-07
* S2_seed121 / P1 / price_interior_rms [price_beyond]: X=6.40e-06, Y=4.98e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=-1.42e-06
* S2_seed122 / P0 / price_interior_rms [price_beyond]: X=8.57e-06, Y=8.12e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-4.54e-07
* S2_seed122 / P1 / price_interior_rms [price_beyond]: X=7.76e-06, Y=5.73e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-2.03e-06
* S2_seed123 / P0 / price_interior_rms [price_beyond]: X=8.38e-06, Y=8.80e-06, u=3.28e-06, tol=3.28e-06, threshold=6.57e-06, gain=4.21e-07
* S2_seed123 / P1 / price_interior_rms [price_beyond]: X=7.50e-06, Y=5.42e-06, u=3.28e-06, tol=3.28e-06, threshold=6.57e-06, gain=-2.08e-06
* ... 144 more

### LR0_m17_h8/WU — not `stateful_advantage` (359 exceptions)

* C_sig0.310 / P0 / resolved_interior_rms [lm_edyn]: X=3.53e-06, Y=3.47e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-5.63e-08
* C_sig0.310 / P1 / resolved_interior_rms [lm_edyn]: X=2.68e-06, Y=2.46e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-2.18e-07
* C_sig0.415 / P1 / resolved_interior_rms [lm_edyn]: X=1.31e-06, Y=1.16e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-1.51e-07
* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=1.14e-06, Y=1.03e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.08e-07
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=7.12e-07, Y=6.13e-07, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-9.93e-08
* M_seed322 / P0 / resolved_interior_rms [lm_edyn]: X=2.33e-06, Y=1.62e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-7.17e-07
* M_seed322 / P1 / resolved_interior_rms [lm_edyn]: X=2.27e-06, Y=1.31e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-9.55e-07
* M_seed323 / P1 / resolved_interior_rms [lm_edyn]: X=3.23e-06, Y=2.65e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-5.81e-07
* S2_seed120 / P1 / resolved_interior_rms [lm_edyn]: X=1.44e-06, Y=1.21e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-2.32e-07
* S2_seed121 / P1 / resolved_interior_rms [lm_edyn]: X=1.47e-06, Y=1.21e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-2.58e-07
* S2_seed122 / P1 / resolved_interior_rms [lm_edyn]: X=1.34e-06, Y=1.15e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-1.89e-07
* S2_seed123 / P0 / resolved_interior_rms [lm_edyn]: X=3.31e-06, Y=3.31e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-2.79e-10
* S2_seed123 / P1 / resolved_interior_rms [lm_edyn]: X=2.07e-06, Y=1.80e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-2.75e-07
* C_sig0.205 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.15e-01, Y=1.10e-01, u=0.00e+00, tol=1.81e-04, threshold=1.10e-01
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.50e-01, Y=1.41e-01, u=0.00e+00, tol=1.81e-04, threshold=1.41e-01
* C_sig0.205 / P0 / calendar_raw_interior_frac_mean [lm_noninf]: X=1.24e-01, Y=1.22e-01, u=0.00e+00, tol=1.81e-04, threshold=1.22e-01
* C_sig0.205 / P1 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.32e-01, Y=1.29e-01, u=0.00e+00, tol=1.81e-04, threshold=1.29e-01
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.51e-01, Y=1.43e-01, u=0.00e+00, tol=1.81e-04, threshold=1.43e-01
* C_sig0.205 / P1 / calendar_raw_interior_frac_mean [lm_noninf]: X=1.23e-01, Y=1.18e-01, u=0.00e+00, tol=1.81e-04, threshold=1.18e-01
* C_sig0.310 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=5.69e-02, Y=4.84e-02, u=0.00e+00, tol=1.81e-04, threshold=4.86e-02
* C_sig0.310 / P1 / density_l2_interior_mean [lm_noninf]: X=1.13e-04, Y=1.08e-04, u=3.77e-06, tol=3.77e-06, threshold=1.12e-04
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.61e-02, Y=2.55e-02, u=0.00e+00, tol=1.81e-04, threshold=2.57e-02
* C_sig0.310 / P1 / butterfly_raw_interior_frac_mean [lm_noninf]: X=5.53e-02, Y=4.73e-02, u=0.00e+00, tol=1.81e-04, threshold=4.75e-02
* C_sig0.415 / P0 / density_l2_interior_mean [lm_noninf]: X=1.83e-04, Y=1.80e-04, u=1.83e-06, tol=1.83e-06, threshold=1.82e-04
* C_sig0.415 / P0 / negative_mass_max_post_initial [lm_noninf]: X=3.81e-05, Y=2.58e-06, u=1.54e-12, tol=1.54e-12, threshold=2.58e-06
* C_sig0.415 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.63e-03, Y=0.00e+00, u=0.00e+00, tol=1.81e-04, threshold=1.81e-04
* C_sig0.415 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.63e-03, Y=0.00e+00, u=0.00e+00, tol=1.81e-04, threshold=1.81e-04
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=2.72e-03, Y=1.63e-03, u=0.00e+00, tol=1.81e-04, threshold=1.81e-03
* C_sig0.415 / P0 / invvol_invalid_frac_full [lm_noninf]: X=6.79e-02, Y=6.74e-02, u=0.00e+00, tol=1.81e-04, threshold=6.76e-02
* C_sig0.415 / P0 / invvol_coverage [lm_noninf]: X=9.93e-01, Y=9.94e-01, u=0.00e+00, tol=1.81e-04, threshold=9.94e-01
* ... 329 more

### LR_m17_h8/W0 — not `resolved_dynamics_improvement` (13 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=4.17e-05, Y=7.81e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-3.39e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.63e-05, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-1.41e-05
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=6.67e-06, Y=4.91e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-1.76e-06
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=2.06e-05, Y=3.02e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.75e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=1.65e-05, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-1.35e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=2.36e-05, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-2.29e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.80e-05, Y=3.56e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.44e-05
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=8.11e-06, Y=4.20e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-3.90e-06
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.20e-05, Y=2.67e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-9.28e-06
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=8.56e-06, Y=3.85e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-4.70e-06
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.16e-05, Y=3.24e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-8.37e-06
* S2_seed124 / P1 / resolved_interior_rms [edyn_beyond]: X=9.27e-06, Y=2.81e-06, u=1.31e-09, tol=1.31e-09, threshold=2.63e-09, gain=-6.46e-06
* S2_seed125 / P1 / resolved_interior_rms [edyn_beyond]: X=8.73e-06, Y=3.46e-06, u=1.57e-09, tol=1.57e-09, threshold=3.15e-09, gain=-5.26e-06

### LR_m17_h8/W0 — not `useful_closure` (225 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [edyn_beyond]: X=4.17e-05, Y=7.81e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-3.39e-05
* C_sig0.310 / P1 / resolved_interior_rms [edyn_beyond]: X=1.63e-05, Y=2.28e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-1.41e-05
* C_sig0.415 / P1 / resolved_interior_rms [edyn_beyond]: X=6.67e-06, Y=4.91e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-1.76e-06
* M_seed320 / P1 / resolved_interior_rms [edyn_beyond]: X=2.06e-05, Y=3.02e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.75e-05
* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=1.65e-05, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-1.35e-05
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=2.36e-05, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-2.29e-05
* M_seed323 / P1 / resolved_interior_rms [edyn_beyond]: X=1.80e-05, Y=3.56e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.44e-05
* S2_seed120 / P1 / resolved_interior_rms [edyn_beyond]: X=8.11e-06, Y=4.20e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-3.90e-06
* S2_seed121 / P1 / resolved_interior_rms [edyn_beyond]: X=1.20e-05, Y=2.67e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-9.28e-06
* S2_seed122 / P1 / resolved_interior_rms [edyn_beyond]: X=8.56e-06, Y=3.85e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-4.70e-06
* S2_seed123 / P1 / resolved_interior_rms [edyn_beyond]: X=1.16e-05, Y=3.24e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-8.37e-06
* S2_seed124 / P1 / resolved_interior_rms [edyn_beyond]: X=9.27e-06, Y=2.81e-06, u=1.31e-09, tol=1.31e-09, threshold=2.63e-09, gain=-6.46e-06
* S2_seed125 / P1 / resolved_interior_rms [edyn_beyond]: X=8.73e-06, Y=3.46e-06, u=1.57e-09, tol=1.57e-09, threshold=3.15e-09, gain=-5.26e-06
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=4.70e-05, Y=1.91e-05, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=-2.79e-05
* C_sig0.310 / P0 / price_interior_rms [price_beyond]: X=5.90e-06, Y=9.64e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=3.74e-06
* C_sig0.310 / P1 / price_interior_rms [price_beyond]: X=1.65e-05, Y=4.80e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=-1.17e-05
* C_sig0.415 / P0 / price_interior_rms [price_beyond]: X=9.24e-06, Y=8.41e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-8.28e-07
* C_sig0.415 / P1 / price_interior_rms [price_beyond]: X=1.02e-05, Y=6.30e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-3.91e-06
* D2_seed220 / P0 / price_interior_rms [price_beyond]: X=8.37e-06, Y=7.31e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-1.06e-06
* D2_seed220 / P1 / price_interior_rms [price_beyond]: X=7.42e-06, Y=6.17e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-1.25e-06
* D2_seed221 / P0 / price_interior_rms [price_beyond]: X=9.30e-06, Y=7.86e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-1.44e-06
* D2_seed221 / P1 / price_interior_rms [price_beyond]: X=8.29e-06, Y=6.19e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-2.10e-06
* D2_seed222 / P0 / price_interior_rms [price_beyond]: X=9.66e-06, Y=7.84e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.82e-06
* D2_seed222 / P1 / price_interior_rms [price_beyond]: X=8.59e-06, Y=6.14e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-2.45e-06
* D2_seed223 / P0 / price_interior_rms [price_beyond]: X=8.47e-06, Y=7.31e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-1.16e-06
* D2_seed223 / P1 / price_interior_rms [price_beyond]: X=7.47e-06, Y=6.20e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-1.27e-06
* M_seed320 / P0 / price_interior_rms [price_beyond]: X=6.80e-06, Y=1.01e-05, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=3.27e-06
* M_seed320 / P1 / price_interior_rms [price_beyond]: X=2.11e-05, Y=5.38e-06, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=-1.58e-05
* M_seed321 / P0 / price_interior_rms [price_beyond]: X=9.06e-06, Y=1.14e-05, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=2.31e-06
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=1.81e-05, Y=6.41e-06, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=-1.17e-05
* ... 195 more

### LR_m17_h8/W0 — not `stateful_advantage` (457 exceptions)

* C_sig0.205 / P1 / resolved_interior_rms [lm_edyn]: X=4.17e-05, Y=4.75e-06, u=3.64e-09, tol=3.64e-09, threshold=7.29e-09, gain=-3.70e-05
* C_sig0.310 / P1 / resolved_interior_rms [lm_edyn]: X=1.63e-05, Y=2.46e-06, u=2.60e-09, tol=2.60e-09, threshold=5.19e-09, gain=-1.39e-05
* C_sig0.415 / P1 / resolved_interior_rms [lm_edyn]: X=6.67e-06, Y=1.16e-06, u=2.91e-09, tol=2.91e-09, threshold=5.82e-09, gain=-5.50e-06
* D2_seed221 / P1 / resolved_interior_rms [lm_edyn]: X=3.08e-06, Y=2.71e-06, u=6.02e-08, tol=6.02e-08, threshold=1.20e-07, gain=-3.67e-07
* D2_seed222 / P1 / resolved_interior_rms [lm_edyn]: X=3.14e-06, Y=2.67e-06, u=5.83e-08, tol=5.83e-08, threshold=1.17e-07, gain=-4.67e-07
* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=1.37e-06, Y=1.03e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-3.44e-07
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=2.06e-05, Y=6.13e-07, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-1.99e-05
* M_seed321 / P0 / resolved_interior_rms [lm_edyn]: X=3.95e-06, Y=3.14e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-8.15e-07
* M_seed321 / P1 / resolved_interior_rms [lm_edyn]: X=1.65e-05, Y=2.85e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-1.37e-05
* M_seed322 / P0 / resolved_interior_rms [lm_edyn]: X=3.57e-06, Y=1.62e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.95e-06
* M_seed322 / P1 / resolved_interior_rms [lm_edyn]: X=2.36e-05, Y=1.31e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-2.23e-05
* M_seed323 / P0 / resolved_interior_rms [lm_edyn]: X=6.69e-06, Y=5.61e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.08e-06
* M_seed323 / P1 / resolved_interior_rms [lm_edyn]: X=1.80e-05, Y=2.65e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.53e-05
* S2_seed120 / P1 / resolved_interior_rms [lm_edyn]: X=8.11e-06, Y=1.21e-06, u=2.72e-09, tol=2.72e-09, threshold=5.44e-09, gain=-6.90e-06
* S2_seed121 / P1 / resolved_interior_rms [lm_edyn]: X=1.20e-05, Y=1.21e-06, u=2.27e-09, tol=2.27e-09, threshold=4.53e-09, gain=-1.07e-05
* S2_seed122 / P1 / resolved_interior_rms [lm_edyn]: X=8.56e-06, Y=1.15e-06, u=2.41e-09, tol=2.41e-09, threshold=4.82e-09, gain=-7.41e-06
* S2_seed123 / P1 / resolved_interior_rms [lm_edyn]: X=1.16e-05, Y=1.80e-06, u=2.53e-09, tol=2.53e-09, threshold=5.05e-09, gain=-9.81e-06
* S2_seed124 / P1 / resolved_interior_rms [lm_edyn]: X=9.27e-06, Y=9.34e-07, u=1.31e-09, tol=1.31e-09, threshold=2.63e-09, gain=-8.33e-06
* S2_seed125 / P1 / resolved_interior_rms [lm_edyn]: X=8.73e-06, Y=1.22e-06, u=1.57e-09, tol=1.57e-09, threshold=3.15e-09, gain=-7.50e-06
* C_sig0.205 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.13e-01, Y=1.10e-01, u=0.00e+00, tol=1.81e-04, threshold=1.10e-01
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.44e-01, Y=1.41e-01, u=0.00e+00, tol=1.81e-04, threshold=1.41e-01
* C_sig0.205 / P1 / price_interior_rms [lm_noninf]: X=4.70e-05, Y=1.93e-05, u=4.29e-06, tol=4.29e-06, threshold=2.35e-05
* C_sig0.205 / P1 / density_l2_interior_mean [lm_noninf]: X=6.79e-04, Y=2.48e-04, u=9.06e-06, tol=9.06e-06, threshold=2.57e-04
* C_sig0.205 / P1 / negative_mass_max_post_initial [lm_noninf]: X=5.03e-02, Y=1.28e-02, u=1.54e-12, tol=1.54e-12, threshold=1.28e-02
* C_sig0.205 / P1 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.78e-01, Y=1.29e-01, u=0.00e+00, tol=1.81e-04, threshold=1.29e-01
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.93e-01, Y=1.43e-01, u=0.00e+00, tol=1.81e-04, threshold=1.43e-01
* C_sig0.205 / P1 / butterfly_raw_interior_frac_mean [lm_noninf]: X=2.01e-01, Y=1.70e-01, u=0.00e+00, tol=1.81e-04, threshold=1.70e-01
* C_sig0.205 / P1 / calendar_raw_interior_frac_mean [lm_noninf]: X=1.25e-01, Y=1.18e-01, u=0.00e+00, tol=1.81e-04, threshold=1.18e-01
* C_sig0.205 / P1 / invvol_coverage [lm_noninf]: X=9.51e-01, Y=9.78e-01, u=0.00e+00, tol=1.81e-04, threshold=9.78e-01
* C_sig0.310 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=2.41e-02, Y=2.17e-02, u=0.00e+00, tol=1.81e-04, threshold=2.19e-02
* ... 427 more

### LR_m17_h8/WI — not `resolved_dynamics_improvement` (2 exceptions)

* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=3.34e-06, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-2.99e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=3.40e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-2.72e-06

### LR_m17_h8/WI — not `useful_closure` (182 exceptions)

* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=3.34e-06, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-2.99e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=3.40e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-2.72e-06
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=1.87e-05, Y=1.91e-05, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=4.94e-07
* C_sig0.310 / P0 / price_interior_rms [price_beyond]: X=5.90e-06, Y=9.64e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=3.74e-06
* C_sig0.310 / P1 / price_interior_rms [price_beyond]: X=5.12e-06, Y=4.80e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=-3.20e-07
* C_sig0.415 / P0 / price_interior_rms [price_beyond]: X=9.24e-06, Y=8.41e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-8.28e-07
* C_sig0.415 / P1 / price_interior_rms [price_beyond]: X=8.59e-06, Y=6.30e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-2.28e-06
* D2_seed220 / P0 / price_interior_rms [price_beyond]: X=8.37e-06, Y=7.31e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-1.06e-06
* D2_seed220 / P1 / price_interior_rms [price_beyond]: X=7.33e-06, Y=6.17e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-1.16e-06
* D2_seed221 / P0 / price_interior_rms [price_beyond]: X=9.30e-06, Y=7.86e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-1.44e-06
* D2_seed221 / P1 / price_interior_rms [price_beyond]: X=8.19e-06, Y=6.19e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-2.00e-06
* D2_seed222 / P0 / price_interior_rms [price_beyond]: X=9.66e-06, Y=7.84e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.82e-06
* D2_seed222 / P1 / price_interior_rms [price_beyond]: X=8.43e-06, Y=6.14e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-2.29e-06
* D2_seed223 / P0 / price_interior_rms [price_beyond]: X=8.47e-06, Y=7.31e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-1.16e-06
* D2_seed223 / P1 / price_interior_rms [price_beyond]: X=7.36e-06, Y=6.20e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-1.16e-06
* M_seed320 / P0 / price_interior_rms [price_beyond]: X=6.80e-06, Y=1.01e-05, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=3.27e-06
* M_seed320 / P1 / price_interior_rms [price_beyond]: X=6.67e-06, Y=5.38e-06, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=-1.28e-06
* M_seed321 / P0 / price_interior_rms [price_beyond]: X=9.06e-06, Y=1.14e-05, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=2.31e-06
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=8.88e-06, Y=6.41e-06, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=-2.47e-06
* M_seed322 / P0 / price_interior_rms [price_beyond]: X=4.92e-06, Y=7.66e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=2.73e-06
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=4.78e-06, Y=3.90e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-8.81e-07
* M_seed323 / P1 / price_interior_rms [price_beyond]: X=1.12e-05, Y=1.14e-05, u=3.83e-06, tol=3.83e-06, threshold=7.66e-06, gain=1.93e-07
* S2_seed120 / P0 / price_interior_rms [price_beyond]: X=8.30e-06, Y=8.43e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=1.29e-07
* S2_seed120 / P1 / price_interior_rms [price_beyond]: X=7.55e-06, Y=5.95e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-1.60e-06
* S2_seed121 / P0 / price_interior_rms [price_beyond]: X=6.51e-06, Y=8.31e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=1.80e-06
* S2_seed121 / P1 / price_interior_rms [price_beyond]: X=5.64e-06, Y=4.98e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=-6.55e-07
* S2_seed122 / P0 / price_interior_rms [price_beyond]: X=7.87e-06, Y=8.12e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=2.48e-07
* S2_seed122 / P1 / price_interior_rms [price_beyond]: X=7.19e-06, Y=5.73e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-1.46e-06
* S2_seed123 / P0 / price_interior_rms [price_beyond]: X=7.14e-06, Y=8.80e-06, u=3.28e-06, tol=3.28e-06, threshold=6.57e-06, gain=1.66e-06
* S2_seed123 / P1 / price_interior_rms [price_beyond]: X=6.34e-06, Y=5.42e-06, u=3.28e-06, tol=3.28e-06, threshold=6.57e-06, gain=-9.23e-07
* ... 152 more

### LR_m17_h8/WI — not `stateful_advantage` (360 exceptions)

* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=1.37e-06, Y=1.03e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-3.44e-07
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=1.18e-06, Y=6.13e-07, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-5.70e-07
* M_seed321 / P0 / resolved_interior_rms [lm_edyn]: X=3.95e-06, Y=3.14e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-8.15e-07
* M_seed321 / P1 / resolved_interior_rms [lm_edyn]: X=3.34e-06, Y=2.85e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-4.85e-07
* M_seed322 / P0 / resolved_interior_rms [lm_edyn]: X=3.57e-06, Y=1.62e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.95e-06
* M_seed322 / P1 / resolved_interior_rms [lm_edyn]: X=3.40e-06, Y=1.31e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-2.09e-06
* M_seed323 / P0 / resolved_interior_rms [lm_edyn]: X=6.69e-06, Y=5.61e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.08e-06
* C_sig0.205 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.13e-01, Y=1.10e-01, u=0.00e+00, tol=1.81e-04, threshold=1.10e-01
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.44e-01, Y=1.41e-01, u=0.00e+00, tol=1.81e-04, threshold=1.41e-01
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.47e-01, Y=1.43e-01, u=0.00e+00, tol=1.81e-04, threshold=1.43e-01
* C_sig0.205 / P1 / calendar_raw_interior_frac_mean [lm_noninf]: X=1.18e-01, Y=1.18e-01, u=0.00e+00, tol=1.81e-04, threshold=1.18e-01
* C_sig0.310 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=2.41e-02, Y=2.17e-02, u=0.00e+00, tol=1.81e-04, threshold=2.19e-02
* C_sig0.310 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.99e-02, Y=2.64e-02, u=0.00e+00, tol=1.81e-04, threshold=2.66e-02
* C_sig0.310 / P1 / positivity_raw_interior_frac_mean [lm_noninf]: X=2.26e-02, Y=1.96e-02, u=0.00e+00, tol=1.81e-04, threshold=1.97e-02
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=3.03e-02, Y=2.55e-02, u=0.00e+00, tol=1.81e-04, threshold=2.57e-02
* C_sig0.415 / P0 / negative_mass_max_post_initial [lm_noninf]: X=1.34e-05, Y=2.58e-06, u=1.54e-12, tol=1.54e-12, threshold=2.58e-06
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=2.17e-03, Y=1.63e-03, u=0.00e+00, tol=1.81e-04, threshold=1.81e-03
* C_sig0.415 / P0 / calendar_raw_interior_frac_mean [lm_noninf]: X=7.25e-04, Y=0.00e+00, u=0.00e+00, tol=1.81e-04, threshold=1.81e-04
* C_sig0.415 / P0 / invvol_invalid_frac_full [lm_noninf]: X=6.88e-02, Y=6.74e-02, u=0.00e+00, tol=1.81e-04, threshold=6.76e-02
* C_sig0.415 / P0 / invvol_coverage [lm_noninf]: X=9.93e-01, Y=9.94e-01, u=0.00e+00, tol=1.81e-04, threshold=9.94e-01
* C_sig0.415 / P1 / negative_mass_max_post_initial [lm_noninf]: X=1.36e-05, Y=3.89e-06, u=1.54e-12, tol=1.54e-12, threshold=3.89e-06
* C_sig0.415 / P1 / invvol_invalid_frac_full [lm_noninf]: X=6.88e-02, Y=6.75e-02, u=0.00e+00, tol=1.81e-04, threshold=6.77e-02
* C_sig0.415 / P1 / invvol_coverage [lm_noninf]: X=9.93e-01, Y=9.94e-01, u=0.00e+00, tol=1.81e-04, threshold=9.94e-01
* D2_seed220 / P0 / density_l2_interior_mean [lm_noninf]: X=1.77e-04, Y=1.71e-04, u=7.83e-07, tol=7.83e-07, threshold=1.72e-04
* D2_seed220 / P0 / invvol_coverage [lm_noninf]: X=9.99e-01, Y=9.99e-01, u=0.00e+00, tol=1.81e-04, threshold=9.99e-01
* D2_seed220 / P1 / density_l2_interior_mean [lm_noninf]: X=1.74e-04, Y=1.69e-04, u=7.83e-07, tol=7.83e-07, threshold=1.70e-04
* D2_seed221 / P0 / density_l2_interior_mean [lm_noninf]: X=1.91e-04, Y=1.83e-04, u=9.88e-07, tol=9.88e-07, threshold=1.83e-04
* D2_seed221 / P0 / invvol_coverage [lm_noninf]: X=9.98e-01, Y=9.99e-01, u=0.00e+00, tol=1.81e-04, threshold=9.99e-01
* D2_seed221 / P1 / density_l2_interior_mean [lm_noninf]: X=1.88e-04, Y=1.81e-04, u=9.88e-07, tol=9.88e-07, threshold=1.82e-04
* D2_seed221 / P1 / invvol_coverage [lm_noninf]: X=9.99e-01, Y=9.99e-01, u=0.00e+00, tol=1.81e-04, threshold=9.99e-01
* ... 330 more

### LR_m17_h8/WU — not `resolved_dynamics_improvement` (2 exceptions)

* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=3.45e-06, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-4.09e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=3.51e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-2.83e-06

### LR_m17_h8/WU — not `useful_closure` (181 exceptions)

* M_seed321 / P1 / resolved_interior_rms [edyn_beyond]: X=3.45e-06, Y=3.04e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-4.09e-07
* M_seed322 / P1 / resolved_interior_rms [edyn_beyond]: X=3.51e-06, Y=6.78e-07, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-2.83e-06
* C_sig0.205 / P1 / price_interior_rms [price_beyond]: X=1.86e-05, Y=1.91e-05, u=4.29e-06, tol=4.29e-06, threshold=8.58e-06, gain=5.09e-07
* C_sig0.310 / P0 / price_interior_rms [price_beyond]: X=5.90e-06, Y=9.64e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=3.74e-06
* C_sig0.310 / P1 / price_interior_rms [price_beyond]: X=5.13e-06, Y=4.80e-06, u=3.44e-06, tol=3.44e-06, threshold=6.89e-06, gain=-3.22e-07
* C_sig0.415 / P0 / price_interior_rms [price_beyond]: X=9.24e-06, Y=8.41e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-8.28e-07
* C_sig0.415 / P1 / price_interior_rms [price_beyond]: X=8.57e-06, Y=6.30e-06, u=3.02e-06, tol=3.02e-06, threshold=6.04e-06, gain=-2.27e-06
* D2_seed220 / P0 / price_interior_rms [price_beyond]: X=8.37e-06, Y=7.31e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-1.06e-06
* D2_seed220 / P1 / price_interior_rms [price_beyond]: X=7.39e-06, Y=6.17e-06, u=2.63e-06, tol=2.63e-06, threshold=5.26e-06, gain=-1.22e-06
* D2_seed221 / P0 / price_interior_rms [price_beyond]: X=9.30e-06, Y=7.86e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-1.44e-06
* D2_seed221 / P1 / price_interior_rms [price_beyond]: X=8.28e-06, Y=6.19e-06, u=2.72e-06, tol=2.72e-06, threshold=5.45e-06, gain=-2.08e-06
* D2_seed222 / P0 / price_interior_rms [price_beyond]: X=9.66e-06, Y=7.84e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-1.82e-06
* D2_seed222 / P1 / price_interior_rms [price_beyond]: X=8.47e-06, Y=6.14e-06, u=2.69e-06, tol=2.69e-06, threshold=5.38e-06, gain=-2.33e-06
* D2_seed223 / P0 / price_interior_rms [price_beyond]: X=8.47e-06, Y=7.31e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-1.16e-06
* D2_seed223 / P1 / price_interior_rms [price_beyond]: X=7.39e-06, Y=6.20e-06, u=2.60e-06, tol=2.60e-06, threshold=5.20e-06, gain=-1.19e-06
* M_seed320 / P0 / price_interior_rms [price_beyond]: X=6.80e-06, Y=1.01e-05, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=3.27e-06
* M_seed320 / P1 / price_interior_rms [price_beyond]: X=6.67e-06, Y=5.38e-06, u=3.49e-06, tol=3.49e-06, threshold=6.98e-06, gain=-1.29e-06
* M_seed321 / P0 / price_interior_rms [price_beyond]: X=9.06e-06, Y=1.14e-05, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=2.31e-06
* M_seed321 / P1 / price_interior_rms [price_beyond]: X=8.95e-06, Y=6.41e-06, u=3.48e-06, tol=3.48e-06, threshold=6.96e-06, gain=-2.54e-06
* M_seed322 / P0 / price_interior_rms [price_beyond]: X=4.92e-06, Y=7.66e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=2.73e-06
* M_seed322 / P1 / price_interior_rms [price_beyond]: X=4.84e-06, Y=3.90e-06, u=3.69e-06, tol=3.69e-06, threshold=7.37e-06, gain=-9.45e-07
* M_seed323 / P1 / price_interior_rms [price_beyond]: X=1.13e-05, Y=1.14e-05, u=3.83e-06, tol=3.83e-06, threshold=7.66e-06, gain=1.52e-07
* S2_seed120 / P0 / price_interior_rms [price_beyond]: X=8.30e-06, Y=8.43e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=1.29e-07
* S2_seed120 / P1 / price_interior_rms [price_beyond]: X=7.54e-06, Y=5.95e-06, u=3.13e-06, tol=3.13e-06, threshold=6.25e-06, gain=-1.60e-06
* S2_seed121 / P0 / price_interior_rms [price_beyond]: X=6.51e-06, Y=8.31e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=1.80e-06
* S2_seed121 / P1 / price_interior_rms [price_beyond]: X=5.66e-06, Y=4.98e-06, u=3.37e-06, tol=3.37e-06, threshold=6.74e-06, gain=-6.81e-07
* S2_seed122 / P0 / price_interior_rms [price_beyond]: X=7.87e-06, Y=8.12e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=2.48e-07
* S2_seed122 / P1 / price_interior_rms [price_beyond]: X=7.19e-06, Y=5.73e-06, u=3.16e-06, tol=3.16e-06, threshold=6.32e-06, gain=-1.46e-06
* S2_seed123 / P0 / price_interior_rms [price_beyond]: X=7.14e-06, Y=8.80e-06, u=3.28e-06, tol=3.28e-06, threshold=6.57e-06, gain=1.66e-06
* S2_seed123 / P1 / price_interior_rms [price_beyond]: X=6.34e-06, Y=5.42e-06, u=3.28e-06, tol=3.28e-06, threshold=6.57e-06, gain=-9.21e-07
* ... 151 more

### LR_m17_h8/WU — not `stateful_advantage` (359 exceptions)

* M_seed320 / P0 / resolved_interior_rms [lm_edyn]: X=1.37e-06, Y=1.03e-06, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-3.44e-07
* M_seed320 / P1 / resolved_interior_rms [lm_edyn]: X=1.18e-06, Y=6.13e-07, u=2.38e-08, tol=2.38e-08, threshold=4.77e-08, gain=-5.70e-07
* M_seed321 / P0 / resolved_interior_rms [lm_edyn]: X=3.95e-06, Y=3.14e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-8.15e-07
* M_seed321 / P1 / resolved_interior_rms [lm_edyn]: X=3.45e-06, Y=2.85e-06, u=2.40e-08, tol=2.40e-08, threshold=4.80e-08, gain=-5.95e-07
* M_seed322 / P0 / resolved_interior_rms [lm_edyn]: X=3.57e-06, Y=1.62e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-1.95e-06
* M_seed322 / P1 / resolved_interior_rms [lm_edyn]: X=3.51e-06, Y=1.31e-06, u=7.77e-09, tol=7.77e-09, threshold=1.55e-08, gain=-2.20e-06
* M_seed323 / P0 / resolved_interior_rms [lm_edyn]: X=6.69e-06, Y=5.61e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-1.08e-06
* M_seed323 / P1 / resolved_interior_rms [lm_edyn]: X=2.68e-06, Y=2.65e-06, u=2.06e-08, tol=2.06e-08, threshold=4.12e-08, gain=-2.53e-08
* C_sig0.205 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=1.13e-01, Y=1.10e-01, u=0.00e+00, tol=1.81e-04, threshold=1.10e-01
* C_sig0.205 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.44e-01, Y=1.41e-01, u=0.00e+00, tol=1.81e-04, threshold=1.41e-01
* C_sig0.205 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=1.47e-01, Y=1.43e-01, u=0.00e+00, tol=1.81e-04, threshold=1.43e-01
* C_sig0.310 / P0 / positivity_raw_interior_frac_mean [lm_noninf]: X=2.41e-02, Y=2.17e-02, u=0.00e+00, tol=1.81e-04, threshold=2.19e-02
* C_sig0.310 / P0 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=2.99e-02, Y=2.64e-02, u=0.00e+00, tol=1.81e-04, threshold=2.66e-02
* C_sig0.310 / P1 / positivity_raw_interior_frac_mean [lm_noninf]: X=2.26e-02, Y=1.96e-02, u=0.00e+00, tol=1.81e-04, threshold=1.97e-02
* C_sig0.310 / P1 / monotonicity_raw_interior_frac_mean [lm_noninf]: X=3.04e-02, Y=2.55e-02, u=0.00e+00, tol=1.81e-04, threshold=2.57e-02
* C_sig0.415 / P0 / negative_mass_max_post_initial [lm_noninf]: X=1.34e-05, Y=2.58e-06, u=1.54e-12, tol=1.54e-12, threshold=2.58e-06
* C_sig0.415 / P0 / butterfly_raw_interior_frac_mean [lm_noninf]: X=2.17e-03, Y=1.63e-03, u=0.00e+00, tol=1.81e-04, threshold=1.81e-03
* C_sig0.415 / P0 / calendar_raw_interior_frac_mean [lm_noninf]: X=7.25e-04, Y=0.00e+00, u=0.00e+00, tol=1.81e-04, threshold=1.81e-04
* C_sig0.415 / P0 / invvol_invalid_frac_full [lm_noninf]: X=6.88e-02, Y=6.74e-02, u=0.00e+00, tol=1.81e-04, threshold=6.76e-02
* C_sig0.415 / P0 / invvol_coverage [lm_noninf]: X=9.93e-01, Y=9.94e-01, u=0.00e+00, tol=1.81e-04, threshold=9.94e-01
* C_sig0.415 / P1 / negative_mass_max_post_initial [lm_noninf]: X=1.44e-05, Y=3.89e-06, u=1.54e-12, tol=1.54e-12, threshold=3.89e-06
* C_sig0.415 / P1 / invvol_invalid_frac_full [lm_noninf]: X=6.88e-02, Y=6.75e-02, u=0.00e+00, tol=1.81e-04, threshold=6.77e-02
* C_sig0.415 / P1 / invvol_coverage [lm_noninf]: X=9.93e-01, Y=9.94e-01, u=0.00e+00, tol=1.81e-04, threshold=9.94e-01
* D2_seed220 / P0 / density_l2_interior_mean [lm_noninf]: X=1.77e-04, Y=1.71e-04, u=7.83e-07, tol=7.83e-07, threshold=1.72e-04
* D2_seed220 / P0 / invvol_coverage [lm_noninf]: X=9.99e-01, Y=9.99e-01, u=0.00e+00, tol=1.81e-04, threshold=9.99e-01
* D2_seed220 / P1 / density_l2_interior_mean [lm_noninf]: X=1.75e-04, Y=1.69e-04, u=7.83e-07, tol=7.83e-07, threshold=1.70e-04
* D2_seed220 / P1 / invvol_coverage [lm_noninf]: X=9.98e-01, Y=9.99e-01, u=0.00e+00, tol=1.81e-04, threshold=9.98e-01
* D2_seed221 / P0 / density_l2_interior_mean [lm_noninf]: X=1.91e-04, Y=1.83e-04, u=9.88e-07, tol=9.88e-07, threshold=1.83e-04
* D2_seed221 / P0 / invvol_coverage [lm_noninf]: X=9.98e-01, Y=9.99e-01, u=0.00e+00, tol=1.81e-04, threshold=9.99e-01
* D2_seed221 / P1 / density_l2_interior_mean [lm_noninf]: X=1.89e-04, Y=1.81e-04, u=9.88e-07, tol=9.88e-07, threshold=1.82e-04
* ... 329 more

## test_ood outcome (reported separately; selected-regimes wording)

| entry | test_ood primary class | classes holding | cases where useful_closure holds (interp+ood, both protocols) |
|---|---|---|---|
| QLp_m9 | no_supported_benefit | – | – |
| QLpm_m9 | no_supported_benefit | – | – |
| QLs_m9 | no_supported_benefit | – | – |
| LM_m9 | inconclusive | – | – |
| LMb_m9 | inconclusive | – | – |
| LR0_m9_h16/W0 | no_supported_benefit | – | – |
| LR0_m9_h16/WI | no_supported_benefit | – | – |
| LR0_m9_h16/WU | no_supported_benefit | – | – |
| LR_m9_h16/W0 | no_supported_benefit | – | – |
| LR_m9_h16/WI | inconclusive | – | – |
| LR_m9_h16/WU | inconclusive | – | – |
| QLp_m17 | no_supported_benefit | – | – |
| QLpm_m17 | no_supported_benefit | – | – |
| QLs_m17 | no_supported_benefit | – | – |
| LM_m17 | inconclusive | – | – |
| LMb_m17 | inconclusive | – | – |
| LR0_m17_h8/W0 | inconclusive | – | – |
| LR0_m17_h8/WI | inconclusive | – | – |
| LR0_m17_h8/WU | inconclusive | – | – |
| LR_m17_h8/W0 | inconclusive | – | – |
| LR_m17_h8/WI | inconclusive | – | – |
| LR_m17_h8/WU | inconclusive | – | – |

## Secondary S1 (m = 33 closed models, density vs RRR33 beyond u_dens on test_interp)

* QLp_m33: density improvement beyond 2u on every case and protocol: no (exceptions: C_sig0.205/P0, C_sig0.205/P1, C_sig0.310/P0, C_sig0.310/P1, C_sig0.415/P0, C_sig0.415/P1, D2_seed220/P0, D2_seed220/P1 ...)
* QLpm_m33: density improvement beyond 2u on every case and protocol: no (exceptions: C_sig0.205/P0, C_sig0.205/P1, C_sig0.310/P0, C_sig0.310/P1, C_sig0.415/P0, C_sig0.415/P1, D2_seed220/P0, D2_seed220/P1 ...)
* QLs_m33: density improvement beyond 2u on every case and protocol: no (exceptions: C_sig0.205/P0, C_sig0.205/P1, C_sig0.310/P0, C_sig0.310/P1, C_sig0.415/P0, C_sig0.415/P1, D2_seed220/P0, D2_seed220/P1 ...)
* LM_m33: density improvement beyond 2u on every case and protocol: no (exceptions: C_sig0.205/P0, C_sig0.205/P1, C_sig0.310/P0, C_sig0.310/P1, C_sig0.415/P0, C_sig0.415/P1, D2_seed220/P0, D2_seed220/P1 ...)
