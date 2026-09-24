"""
Smoke tests for the mz_spectral package.

Scope (see tests/README.md for the running convention):
  - Every mz_spectral submodule imports cleanly (parametrized).
  - A tiny synthetic setup (mz_synthetic_setup.setup_mz_synthetic_case, the
    'constant' Black-Scholes case, n_t/n_k in the tens) runs end-to-end
    through the RRR / RRR+QL integration and produces finite output of the
    expected shape.
  - The documented non-negativity invariant on the QL closure coefficient
    (docs/MZ_QL_NONNEG_FIX_REPORT.md: nu_global >= 0 and every per-step
    nu_i >= 0 after the QL non-negativity fix in quasi_linear.fit_nu_ql).

Does NOT test fit_nu_ql's ridge term specifically (covered by another agent
in tests/test_mz_spectral_regressions.py) and does not touch data_driven.py's
vol_blend_alpha (same reason).

All setups here use case='constant' (closed-form Black-Scholes truth via
build_truth_phi_bs, no Monte Carlo) with n_t<=12, n_k<=32, so the whole file
runs in a couple of seconds on CPU.
"""

from __future__ import annotations

import importlib
import tempfile

import numpy as np
import pytest

MZ_SUBMODULES = [
    "mz_decomposition",
    "quasi_linear",
    "validation",
    "sigma_from_phi",
    "fourier_dupire",
    "energy_closure",
    "uncertainty",
    "uuu_dimension",
    "truncation_study",
    "data_driven",
    "plot_pdf_analysis",
    "plot_vol_surfaces",
    "export_panels",
    "mz_synthetic_setup",
]


@pytest.mark.parametrize("modname", MZ_SUBMODULES)
def test_mz_spectral_submodule_imports(modname):
    mod = importlib.import_module(f"mz_spectral.{modname}")
    assert mod is not None


def _tiny_synthetic_setup(tmp_path, n_t=8, n_k=32, chosen_keep=8):
    from mz_spectral.mz_synthetic_setup import setup_mz_synthetic_case

    return setup_mz_synthetic_case(
        "constant",
        fast=True,
        n_t=n_t,
        n_k=n_k,
        out_root=str(tmp_path),
        chosen_keep=chosen_keep,
    )


def test_tiny_synthetic_setup_runs_end_to_end(tmp_path):
    result = _tiny_synthetic_setup(tmp_path)

    n_t, n_k = 8, 32
    assert result["phi_truth"].shape == (n_t, n_k)
    assert result["phi_rrr"].shape == (n_t, n_k)
    assert result["phi_ql"].shape == (n_t, n_k)

    assert np.all(np.isfinite(result["phi_truth"]))
    assert np.all(np.isfinite(result["phi_rrr"]))
    assert np.all(np.isfinite(result["phi_ql"]))
    assert np.isfinite(result["nu_ql_global"])


def test_tiny_synthetic_setup_data_mode_runs_end_to_end(tmp_path):
    """Same tiny setup but through the data-driven (vol_mode='data') path,
    which builds sigma from phi via Dupire inversion rather than using the
    config's oracle sigma directly."""
    from mz_spectral.mz_synthetic_setup import setup_mz_synthetic_case

    result = setup_mz_synthetic_case(
        "constant",
        fast=True,
        n_t=8,
        n_k=32,
        out_root=str(tmp_path),
        chosen_keep=8,
        vol_mode="data",
    )

    assert result["phi_rrr"].shape == (8, 32)
    assert result["phi_ql"].shape == (8, 32)
    assert np.all(np.isfinite(result["phi_ql"]))
    assert np.isfinite(result["nu_ql_global"])


# ---------------------------------------------------------------------------
# QL non-negativity invariant (docs/MZ_QL_NONNEG_FIX_REPORT.md)
# ---------------------------------------------------------------------------

def test_fit_nu_ql_default_is_nonnegative_on_tiny_synthetic_setup(tmp_path):
    """nu_global from setup_mz_synthetic_case (oracle mode, nonneg default) must be >= 0."""
    result = _tiny_synthetic_setup(tmp_path)
    assert result["nu_ql_global"] >= 0.0


def test_fit_nu_ql_per_step_and_global_are_nonnegative_by_default():
    """Direct call to fit_nu_ql (nonneg=True default): both nu_global and every
    per-step nu_i must be >= 0 -- the closed-form NNLS projection documented in
    docs/MZ_QL_NONNEG_FIX_REPORT.md."""
    from config import DupirePipelineConfig
    from mz_spectral.fourier_dupire import build_second_derivative_matrix, build_uniform_tk_grid
    from mz_spectral.quasi_linear import fit_nu_ql
    from mz_spectral.validation import build_truth_phi_bs, precompute_L_and_blocks

    class _ConstVolConfig:
        model_type = "custom"

        @property
        def custom_volatility_func(self):
            return lambda t, x: np.ones_like(np.asarray(x, dtype=float))

    cfg = DupirePipelineConfig()
    n_t, n_k, keep = 8, 32, 8
    grid, sigma = build_uniform_tk_grid(
        S0=cfg.S0, r=cfg.r, T_max=cfg.T_max, K_max=cfg.K_max, K_min=cfg.K_min,
        T_min=cfg.T_min, n_t=n_t, n_k=n_k, vol_config=_ConstVolConfig(),
    )
    phi_truth = build_truth_phi_bs(grid, cfg.S0, cfg.r, 1.0)
    D2 = build_second_derivative_matrix(n_k, grid.dk)
    _, P_R, _, L_RR_rows = precompute_L_and_blocks(grid, sigma, grid.T[-1], keep)

    nu_global, nu_steps = fit_nu_ql(phi_truth, grid.t_tilde, L_RR_rows, P_R, D2)

    assert nu_global >= 0.0
    assert np.all(nu_steps >= 0.0)


def test_fit_nu_ql_nonneg_false_escape_hatch_can_go_negative():
    """The nonneg=False escape hatch (kept for before/after diagnostics per
    docs/MZ_QL_NONNEG_FIX_REPORT.md) is not itself constrained -- this just
    documents that the default (nonneg=True) is an actual constraint, not a
    no-op, by checking the two branches can disagree in sign on data crafted
    so the unconstrained optimum is negative."""
    from mz_spectral.quasi_linear import fit_nu_ql

    n_k = 8
    n_t = 5
    rng = np.random.default_rng(0)
    P_R = np.eye(n_k)
    D2 = -np.eye(n_k)  # q = P_R @ D2 @ phi_R = -phi_R
    t_tilde = np.linspace(0.0, 1.0, n_t)
    # Construct phi_truth and a matching L_RR of zeros so lhs = P_R @ dphi.
    # Pick phi growing with time so dphi and q=-phi have opposite-sign dot
    # products almost surely, driving the unconstrained nu* negative.
    phi_truth = np.tile(np.linspace(1.0, 2.0, n_k), (n_t, 1)) * np.linspace(1.0, 3.0, n_t)[:, None]
    L_RR_rows = [np.zeros((n_k, n_k)) for _ in range(n_t)]

    nu_default, _ = fit_nu_ql(phi_truth, t_tilde, L_RR_rows, P_R, D2, nonneg=True)
    nu_legacy, _ = fit_nu_ql(phi_truth, t_tilde, L_RR_rows, P_R, D2, nonneg=False)

    assert nu_default >= 0.0
    assert nu_legacy <= 0.0
    assert nu_legacy < nu_default
