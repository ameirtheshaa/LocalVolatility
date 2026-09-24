"""
Mori–Zwanzig / spectral closure utilities for the scaled Dupire PDE.

See docs/MATHEMATICAL_TREATMENT.md §12 and module docstrings.
"""

from mz_spectral.fourier_dupire import (
    build_uniform_tk_grid,
    build_second_derivative_matrix,
    build_L_operator,
    apply_generator,
    verify_ode_residual,
)
from mz_spectral.mz_decomposition import (
    apply_projector,
    low_pass_projector_matrix,
    resolved_forcing_from_unresolved,
    split_operator_blocks,
)
from mz_spectral.plot_pdf_analysis import (
    bl_pdf_from_phi_row,
    create_mz_pdf_analysis,
    mc_kde_pdf_on_K_grid,
    nearest_time_index,
    pdf_comparison_table,
    pdf_reference_is_mc_kde,
    reference_pdf,
)
from mz_spectral.data_driven import (
    build_L_rr_from_phi_data,
    run_data_driven_rom_integration,
)
from mz_spectral.plot_vol_surfaces import (
    plot_mz_volatility_surfaces,
    plot_mz_volatility_surfaces_triple,
)
from mz_spectral.sigma_from_phi import (
    build_eta_sigma_from_phi,
    implied_sigma_from_phi,
    regularize_sigma_grid,
    regularize_sigma_memory,
)
from mz_spectral.quasi_linear import fit_nu_ql, ql_backscatter_term
from mz_spectral.uuu_dimension import (
    ambient_dim_u,
    estimate_uuu_dimension,
    fnn_bottleneck_scan,
    pca_effective_dimension,
    save_uuu_dimension_report,
)
from mz_spectral.truncation_study import (
    default_keep_values,
    pick_keep_pareto,
    save_truncation_metadata,
    sweep_k_trunc,
)
from mz_spectral.validation import (
    build_truth_phi_bs,
    compare_to_nn_pdf,
    integrate_rrr_only,
    integrate_rrr_ql,
    interior_k_mask,
    payoff_phi_tilde,
    pdf_from_phi_tilde,
    pdf_metrics,
    phi_l2_error,
    precompute_L_and_blocks,
    spectral_pdf_filter,
    uuu_gap_metric,
)

__all__ = [
    "build_uniform_tk_grid",
    "build_second_derivative_matrix",
    "build_L_operator",
    "apply_generator",
    "verify_ode_residual",
    "low_pass_projector_matrix",
    "split_operator_blocks",
    "apply_projector",
    "resolved_forcing_from_unresolved",
    "build_L_rr_from_phi_data",
    "run_data_driven_rom_integration",
    "build_eta_sigma_from_phi",
    "regularize_sigma_grid",
    "regularize_sigma_memory",
    "implied_sigma_from_phi",
    "plot_mz_volatility_surfaces",
    "plot_mz_volatility_surfaces_triple",
    "create_mz_pdf_analysis",
    "mc_kde_pdf_on_K_grid",
    "bl_pdf_from_phi_row",
    "nearest_time_index",
    "pdf_comparison_table",
    "pdf_reference_is_mc_kde",
    "reference_pdf",
    "fit_nu_ql",
    "ql_backscatter_term",
    "sweep_k_trunc",
    "default_keep_values",
    "pick_keep_pareto",
    "save_truncation_metadata",
    "pdf_from_phi_tilde",
    "pdf_metrics",
    "interior_k_mask",
    "payoff_phi_tilde",
    "integrate_rrr_only",
    "integrate_rrr_ql",
    "compare_to_nn_pdf",
    "spectral_pdf_filter",
    "build_truth_phi_bs",
    "phi_l2_error",
    "precompute_L_and_blocks",
    "uuu_gap_metric",
    "ambient_dim_u",
    "estimate_uuu_dimension",
    "pca_effective_dimension",
    "fnn_bottleneck_scan",
    "save_uuu_dimension_report",
]
