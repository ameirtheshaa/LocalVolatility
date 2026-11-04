"""Access to the committed production code (no copies, no edits)."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(STUDY_ROOT))
PKG_ROOT = os.path.join(REPO_ROOT, "Synthetic_Data_Tensorflow_Advanced")

if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from config import DupirePipelineConfig, VolatilityConfig  # noqa: E402
from analytical_solutions import black_scholes_call, lognormal_density  # noqa: E402
from mz_spectral.fourier_dupire import (  # noqa: E402
    build_L_operator,
    build_second_derivative_matrix,
    build_uniform_tk_grid,
    eta_tilde_from_sigma,
)
from mz_spectral.mz_decomposition import (  # noqa: E402
    low_pass_projector_matrix,
    split_operator_blocks,
)
from mz_spectral.quasi_linear import fit_nu_ql  # noqa: E402
from mz_spectral.validation import (  # noqa: E402
    build_truth_phi_bs,
    integrate_rrr_only,
    integrate_rrr_ql,
    interior_k_mask,
    payoff_phi_tilde,
    pdf_from_phi_tilde,
    precompute_L_and_blocks,
)

__all__ = [
    "DupirePipelineConfig", "VolatilityConfig", "black_scholes_call", "lognormal_density",
    "build_L_operator", "build_second_derivative_matrix", "build_uniform_tk_grid",
    "eta_tilde_from_sigma", "low_pass_projector_matrix", "split_operator_blocks", "fit_nu_ql",
    "build_truth_phi_bs", "integrate_rrr_only", "integrate_rrr_ql", "interior_k_mask",
    "payoff_phi_tilde", "pdf_from_phi_tilde", "precompute_L_and_blocks",
    "REPO_ROOT", "PKG_ROOT", "STUDY_ROOT",
]
