"""
Regression tests for two mz_spectral bugs:

1. quasi_linear.fit_nu_ql double-counted `ridge` in the pooled/global
   least-squares denominator (see docs/MZ_QL_NONNEG_FIX_REPORT.md §2 for the
   intended closed-form nu* = sum(lhs.q) / (sum(q.q) + ridge)).
2. data_driven.run_data_driven_rom_integration's `vol_blend_alpha` had no
   effect on the output because the blended sigma was discarded before the
   next vol-iteration rebuilt the generator from scratch.

All fixtures are tiny (n_t<=8, n_k<=16) so each test runs in well under a
second on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import DupirePipelineConfig
from mz_spectral.fourier_dupire import build_second_derivative_matrix, build_uniform_tk_grid
from mz_spectral.data_driven import run_data_driven_rom_integration
from mz_spectral.quasi_linear import fit_nu_ql
from mz_spectral.validation import build_truth_phi_bs, precompute_L_and_blocks


class _ConstVolConfig:
    """Minimal stand-in for VolatilityConfig.custom(...) — flat sigma(t, x)."""

    model_type = "custom"

    def __init__(self, sigma_const: float):
        self._sigma_const = sigma_const

    @property
    def custom_volatility_func(self):
        s = self._sigma_const
        return lambda t, x: s * np.ones_like(np.asarray(x, dtype=float))


def _tiny_grid(n_t: int, n_k: int, sigma_const: float = 0.3):
    cfg = DupirePipelineConfig()
    grid, sigma = build_uniform_tk_grid(
        S0=cfg.S0,
        r=cfg.r,
        T_max=cfg.T_max,
        K_max=cfg.K_max,
        K_min=cfg.K_min,
        T_min=cfg.T_min,
        n_t=n_t,
        n_k=n_k,
        vol_config=_ConstVolConfig(sigma_const),
    )
    phi_truth = build_truth_phi_bs(grid, cfg.S0, cfg.r, sigma_const)
    D2 = build_second_derivative_matrix(n_k, grid.dk)
    return cfg, grid, sigma, phi_truth, D2


# ---------------------------------------------------------------------------
# (a) fit_nu_ql: ridge counted exactly once in the global denominator.
# ---------------------------------------------------------------------------
def test_fit_nu_ql_global_ridge_counted_once():
    n_t, n_k, keep = 6, 16, 4
    # Large enough (comparable to sum(q.q) on this tiny grid, ~1.7e5) that a
    # double count of ridge in the pooled denominator is clearly detectable.
    ridge = 1e5
    _, grid, sigma, phi_truth, D2 = _tiny_grid(n_t, n_k)
    _, P_R, _, L_RR_rows = precompute_L_and_blocks(grid, sigma, grid.T[-1], keep)

    # Unclipped, unconstrained fit (disable clipping/nonneg to expose the raw ratio).
    nu_g, nu_steps = fit_nu_ql(
        phi_truth, grid.t_tilde, L_RR_rows, P_R, D2,
        ridge=ridge, nu_clip=1e6, nonneg=False,
    )

    numers = []
    denoms = []
    for i in range(1, n_t - 1):
        dt = grid.t_tilde[i + 1] - grid.t_tilde[i - 1]
        dphi = (phi_truth[i + 1] - phi_truth[i - 1]) / dt
        phi_R = P_R @ phi_truth[i]
        lhs = P_R @ dphi - L_RR_rows[i] @ phi_R
        q = P_R @ (D2 @ phi_R)
        numers.append(float(np.dot(lhs, q)))
        denoms.append(float(np.dot(q, q)))

    # Closed-form expected value: ridge added exactly once in the pooled sum.
    expected_correct = sum(numers) / (sum(denoms) + ridge)
    # What the old (buggy) code computed: ridge added once per time-slice
    # AND once more globally.
    expected_buggy = sum(numers) / (sum(d + ridge for d in denoms) + ridge)

    assert nu_g == pytest.approx(expected_correct, rel=1e-10, abs=1e-12)
    # The two formulas must meaningfully disagree at this ridge, otherwise the
    # test would not be discriminating.
    assert abs(expected_correct - expected_buggy) > 1e-3
    assert nu_g != pytest.approx(expected_buggy, rel=1e-3)


# ---------------------------------------------------------------------------
# (b)/(c) vol_blend_alpha actually drives the vol-iteration when
# vol_iterate > 1, and has zero effect when vol_iterate == 1 (default).
# ---------------------------------------------------------------------------
def test_vol_blend_alpha_affects_output_when_iterating():
    n_t, n_k, keep = 6, 16, 4
    cfg, grid, _, phi_truth, D2 = _tiny_grid(n_t, n_k)

    out_lo = run_data_driven_rom_integration(
        phi_truth, grid, D2, cfg, keep, vol_iterate=2, vol_blend_alpha=0.0,
    )
    out_hi = run_data_driven_rom_integration(
        phi_truth, grid, D2, cfg, keep, vol_iterate=2, vol_blend_alpha=1.0,
    )

    assert not np.allclose(out_lo["phi_ql"], out_hi["phi_ql"])
    assert not np.allclose(out_lo["sigma_generator"], out_hi["sigma_generator"])


def test_vol_blend_alpha_is_noop_at_default_vol_iterate():
    n_t, n_k, keep = 6, 16, 4
    cfg, grid, _, phi_truth, D2 = _tiny_grid(n_t, n_k)

    out_lo = run_data_driven_rom_integration(
        phi_truth, grid, D2, cfg, keep, vol_iterate=1, vol_blend_alpha=0.0,
    )
    out_hi = run_data_driven_rom_integration(
        phi_truth, grid, D2, cfg, keep, vol_iterate=1, vol_blend_alpha=1.0,
    )

    # With vol_iterate=1 the blend branch (k < vol_iterate - 1) never
    # executes, so the output must be bit-for-bit unchanged.
    np.testing.assert_array_equal(out_lo["phi_ql"], out_hi["phi_ql"])
    np.testing.assert_array_equal(out_lo["phi_rrr"], out_hi["phi_rrr"])
    np.testing.assert_array_equal(out_lo["sigma_generator"], out_hi["sigma_generator"])
