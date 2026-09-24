#!/usr/bin/env python3
"""
Shared grid build, truncation sweep (optional), and RRR/QL integration for MZ synthetic drivers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Optional

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from config import DupirePipelineConfig, VolatilityConfig
from dupire_pipeline import DataGenerator
from mz_spectral.fourier_dupire import build_second_derivative_matrix, build_uniform_tk_grid
from mz_spectral.quasi_linear import fit_nu_ql
from mz_spectral.truncation_study import (
    default_keep_values,
    pick_keep_pareto,
    save_truncation_metadata,
    sweep_k_trunc,
)
from mz_spectral.data_driven import run_data_driven_rom_integration
from mz_spectral.validation import (
    build_truth_phi_bs,
    integrate_rrr_only,
    integrate_rrr_ql,
    payoff_phi_tilde,
    precompute_L_and_blocks,
)


def truth_from_mc_on_grid(config: DupirePipelineConfig, grid: Any) -> np.ndarray:
    """Monte Carlo call prices on (T, K) grid → φ̃ = C/S0."""
    dg = DataGenerator(config)
    dg.run_mc()
    T_nn, K_nn, phi_ref = dg.get_training_data()
    T_flat = T_nn.numpy().ravel().astype(float)
    K_flat = K_nn.numpy().ravel().astype(float)
    phi_flat = phi_ref.numpy().ravel().astype(float)
    interp = LinearNDInterpolator(
        np.column_stack([T_flat, K_flat]), phi_flat, fill_value=np.nan
    )
    Ti, Ki = np.broadcast_arrays(grid.T[:, None], grid.K)
    phi = interp(Ti.ravel(), Ki.ravel()).reshape(grid.K.shape)
    phi = np.nan_to_num(phi, nan=np.nanmean(phi_flat))
    return phi / config.S0


def load_chosen_keep_from_artifacts(out_dir: str) -> Optional[int]:
    """Read ``chosen_keep`` from a prior ``mz_validation_summary.json`` or ``mz_truncation_metadata.json``."""
    summ = os.path.join(out_dir, "mz_validation_summary.json")
    if os.path.isfile(summ):
        with open(summ, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "chosen_keep" in data:
            return int(data["chosen_keep"])
    meta = os.path.join(out_dir, "mz_truncation_metadata.json")
    if os.path.isfile(meta):
        with open(meta, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "chosen_keep" in data:
            return int(data["chosen_keep"])
    return None


def _validate_chosen_keep(n_k: int, chosen_keep: int) -> None:
    if chosen_keep < 1:
        raise ValueError(f"chosen_keep must be >= 1, got {chosen_keep}")
    if 2 * chosen_keep >= n_k:
        raise ValueError(
            f"Require 2*chosen_keep < n_k for truncation headroom (alias / Nyquist); "
            f"got n_k={n_k}, chosen_keep={chosen_keep}"
        )


def setup_mz_synthetic_case(
    case: str,
    *,
    fast: bool,
    n_t: int,
    n_k: int,
    out_root: str,
    chosen_keep: int = 256,
    chosen_keep_override: Optional[int] = None,
    skip_truncation_sweep: bool = False,
    run_truncation_sweep: bool = False,
    use_pareto_chosen: bool = False,
    estimate_uuu_dim: bool = False,
    uuu_fnn_max_d: int = 128,
    uuu_fnn_epochs: int = 120,
    uuu_pca_only: bool = False,
    m_train: Optional[int] = None,
    vol_mode: Literal["oracle", "data"] = "oracle",
    vol_iterate: int = 1,
    vol_iterate_tol: float = 1e-2,
    phi_data_override: Optional[np.ndarray] = None,
) -> dict:
    """
    Build config, grid, φ_truth, truncation choice, and integrated φ_rrr / φ_ql.

    IC for integration is payoff-consistent ``payoff_phi_tilde`` (not ``phi_truth[0]``).

    ``chosen_keep`` (default 256) is used for ``P_R`` / integration unless
    ``chosen_keep_override`` is set (e.g. ``--replot-only`` loading prior JSON).

    If ``run_truncation_sweep`` is True, runs ``sweep_k_trunc`` over
    ``default_keep_values(n_k)`` and writes ``mz_truncation_metadata.json``.
    The integration ``keep`` stays ``chosen_keep`` / override unless
    ``use_pareto_chosen`` is True (then ``pick_keep_pareto`` replaces it).

    ``skip_truncation_sweep`` with ``chosen_keep_override``: skip sweep and metadata
    rewrite (replot path); ``chosen_used = chosen_keep_override``.

    ``m_train`` overrides ``DupirePipelineConfig.M_train`` when set.

    ``vol_mode="data"``: build ``L_RR`` from Dupire-inverted ``σ(φ_data)`` only;
    IC ``P_R @ φ_data[0]``; ``fit_nu_ql`` on ``φ_data``. Oracle mode unchanged
    (config ``σ`` in ``L``, payoff IC, ``ν`` from ``φ_truth``).
    """
    config = DupirePipelineConfig()
    if fast:
        config.M_train = 8000
        config.N_t_train = 250
        config.dt_train = config.T_max / max(config.N_t_train - 1, 1)
    if m_train is not None:
        config.M_train = int(m_train)

    if case == "constant":
        config.volatility_config = VolatilityConfig.custom(lambda t, x: 1.0)
        sigma_const: Optional[float] = 1.0
    elif case == "paper":
        config.volatility_config = VolatilityConfig.dupire_exact()
        sigma_const = None
    else:
        raise ValueError(case)

    grid, sigma = build_uniform_tk_grid(
        S0=config.S0,
        r=config.r,
        T_max=config.T_max,
        K_max=config.K_max,
        K_min=config.K_min,
        T_min=config.T_min,
        n_t=n_t,
        n_k=n_k,
        vol_config=config.volatility_config,
    )

    if case == "constant":
        phi_truth = build_truth_phi_bs(grid, config.S0, config.r, float(sigma_const))
    else:
        phi_truth = truth_from_mc_on_grid(config, grid)

    phi_data = (
        np.array(phi_data_override, copy=True)
        if phi_data_override is not None
        else np.array(phi_truth, copy=True)
    )

    n_k_eff = len(grid.k_tilde)
    out_dir = os.path.join(out_root, f"mz_synthetic_{case}")
    meta_file = os.path.join(out_dir, "mz_truncation_metadata.json")

    if skip_truncation_sweep and chosen_keep_override is not None:
        chosen_used = int(chosen_keep_override)
        _validate_chosen_keep(n_k_eff, chosen_used)
        rows: list = []
        meta_path = meta_file if os.path.isfile(meta_file) else None
    else:
        chosen_used = int(chosen_keep_override) if chosen_keep_override is not None else int(chosen_keep)
        _validate_chosen_keep(n_k_eff, chosen_used)

        rows = []
        if run_truncation_sweep:
            keep_values = default_keep_values(n_k_eff)
            rows = sweep_k_trunc(phi_truth, grid, sigma, config, keep_values)
            if use_pareto_chosen:
                chosen_used = int(pick_keep_pareto(rows, pdf_tol=0.08))
                _validate_chosen_keep(n_k_eff, chosen_used)

        meta_path = save_truncation_metadata(
            out_dir,
            case,
            rows,
            chosen_used,
            extra={
                "n_t": n_t,
                "n_k": n_k,
                "fast": fast,
                "run_truncation_sweep": bool(run_truncation_sweep),
                "use_pareto_chosen": bool(use_pareto_chosen),
                "chosen_keep_cli": int(chosen_keep),
            },
        )

    D2 = build_second_derivative_matrix(n_k_eff, grid.dk)
    sigma_oracle = np.array(sigma, copy=True)
    sigma_generator = sigma_oracle
    sigma_data: Optional[np.ndarray] = None
    sigma_history: list = []
    vol_iterations_run = 0

    if vol_mode == "oracle":
        _, P_R, _, L_RR_rows = precompute_L_and_blocks(grid, sigma, config.T_max, chosen_used)
        phi0 = payoff_phi_tilde(grid.k_tilde, float(config.S0), float(config.K_max))
        phi_rrr = integrate_rrr_only(phi0, grid.t_tilde, L_RR_rows, P_R)
        nu_g, _ = fit_nu_ql(phi_truth, grid.t_tilde, L_RR_rows, P_R, D2)
        phi_ql = integrate_rrr_ql(phi0, grid.t_tilde, L_RR_rows, P_R, D2, nu_g)
    elif vol_mode == "data":
        # Exact BS φ is already smooth; spectral low-pass before Dupire destroys flat σ.
        filter_phi_for_sigma = case != "constant"
        dd = run_data_driven_rom_integration(
            phi_data,
            grid,
            D2,
            config,
            chosen_used,
            vol_iterate=vol_iterate,
            vol_iterate_tol=vol_iterate_tol,
            filter_phi_for_sigma=filter_phi_for_sigma,
        )
        phi_rrr = dd["phi_rrr"]
        phi_ql = dd["phi_ql"]
        P_R = dd["P_R"]
        L_RR_rows = dd["L_RR_rows"]
        nu_g = dd["nu_ql_global"]
        sigma_data = dd["sigma_data"]
        sigma_generator = dd["sigma_generator"]
        sigma_history = dd["sigma_history"]
        vol_iterations_run = int(dd["vol_iterations_run"])
    else:
        raise ValueError(f"vol_mode must be 'oracle' or 'data', got {vol_mode!r}")

    uuu_dim_path: Optional[str] = None
    uuu_dim_report: Optional[dict] = None
    if estimate_uuu_dim:
        from mz_spectral.uuu_dimension import estimate_uuu_dimension, save_uuu_dimension_report

        uuu_dim_report = estimate_uuu_dimension(
            phi_truth,
            grid,
            sigma,
            config.T_max,
            chosen_used,
            fnn_max_d=uuu_fnn_max_d,
            fnn_epochs=uuu_fnn_epochs,
            run_fnn=not uuu_pca_only,
            verbose=0,
        )
        uuu_dim_path = save_uuu_dimension_report(out_dir, uuu_dim_report)

    return {
        "case": case,
        "config": config,
        "grid": grid,
        "sigma": sigma_oracle,
        "sigma_oracle": sigma_oracle,
        "sigma_generator": sigma_generator,
        "sigma_data": sigma_data,
        "sigma_history": sigma_history,
        "vol_mode": vol_mode,
        "vol_iterate": int(vol_iterate),
        "vol_iterations_run": vol_iterations_run,
        "phi_truth": phi_truth,
        "phi_data": phi_data,
        "chosen_keep": chosen_used,
        "D2": D2,
        "P_R": P_R,
        "L_RR_rows": L_RR_rows,
        "phi_rrr": phi_rrr,
        "phi_ql": phi_ql,
        "nu_ql_global": float(nu_g),
        "sigma_const": sigma_const,
        "out_dir": out_dir,
        "truncation_metadata": meta_path,
        "truncation_rows": rows,
        "uuu_dimension_json": uuu_dim_path,
        "uuu_dimension": uuu_dim_report,
    }
