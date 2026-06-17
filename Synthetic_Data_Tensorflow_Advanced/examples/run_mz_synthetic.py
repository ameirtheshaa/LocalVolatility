#!/usr/bin/env python3
"""
Run Mori–Zwanzig / spectral Dupire validation on synthetic benchmarks.

  python examples/run_mz_synthetic.py --case constant --fast
  python examples/run_mz_synthetic.py --case paper --fast
  python examples/run_mz_synthetic.py --case both --fast --plot-surfaces --plot-pdf \\
      --model-dir models/runs/synthetic_paper_large_dataset_constant_vol

Requires execution from repo root or with PYTHONPATH including
Synthetic_Data_Tensorflow_Advanced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Package root: Synthetic_Data_Tensorflow_Advanced
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mz_spectral.mz_synthetic_setup import setup_mz_synthetic_case  # noqa: E402
from mz_spectral.plot_pdf_analysis import (  # noqa: E402
    create_mz_pdf_analysis,
    mc_samples_at_maturities,
    nearest_time_index,
    pdf_reference_is_mc_kde,
    reference_pdf_row,
)
from mz_spectral.plot_vol_surfaces import plot_mz_volatility_surfaces  # noqa: E402
from mz_spectral.validation import (  # noqa: E402
    compare_to_nn_pdf,
    interior_k_mask,
    pdf_from_phi_tilde,
    pdf_metrics,
    phi_l2_error,
    spectral_pdf_filter,
    uuu_gap_metric,
)


def run_one_case(
    case: str,
    *,
    fast: bool,
    model_dir: str | None,
    out_root: str,
    n_t: int,
    n_k: int,
    chosen_keep: int = 256,
    run_truncation_sweep: bool = False,
    use_pareto_chosen: bool = False,
    estimate_uuu_dim: bool = False,
    uuu_fnn_max_d: int = 128,
    uuu_fnn_epochs: int = 120,
    uuu_pca_only: bool = False,
    plot_surfaces: bool = False,
    plot_pdf: bool = False,
    pdf_maturities: list[float] | None = None,
    vol_mode: str = "oracle",
    vol_iterate: int = 1,
    vol_iterate_tol: float = 1e-2,
) -> dict:
    pdf_maturities = list(pdf_maturities or [0.5, 1.0, 1.5])

    st = setup_mz_synthetic_case(
        case,
        fast=fast,
        n_t=n_t,
        n_k=n_k,
        out_root=out_root,
        chosen_keep=chosen_keep,
        chosen_keep_override=None,
        skip_truncation_sweep=False,
        run_truncation_sweep=run_truncation_sweep,
        use_pareto_chosen=use_pareto_chosen,
        estimate_uuu_dim=estimate_uuu_dim,
        uuu_fnn_max_d=uuu_fnn_max_d,
        uuu_fnn_epochs=uuu_fnn_epochs,
        uuu_pca_only=uuu_pca_only,
        vol_mode=vol_mode,
        vol_iterate=vol_iterate,
        vol_iterate_tol=vol_iterate_tol,
    )
    config = st["config"]
    grid = st["grid"]
    sigma_oracle = st["sigma_oracle"]
    sigma = sigma_oracle
    phi_truth = st["phi_truth"]
    phi_data = st["phi_data"]
    sigma_data = st.get("sigma_data")
    sigma_generator = st.get("sigma_generator")
    chosen = st["chosen_keep"]
    D2 = st["D2"]
    phi_rrr = st["phi_rrr"]
    phi_ql = st["phi_ql"]
    nu_g = st["nu_ql_global"]
    sigma_const = st["sigma_const"]
    out_dir = st["out_dir"]
    meta_path = st["truncation_metadata"]
    uuu_dim = st.get("uuu_dimension")

    i_last = len(grid.T) - 1
    T_sel = float(grid.T[i_last])
    K_sel = grid.K[i_last]
    km = interior_k_mask(K_sel)

    mc_slice = None
    if case == "paper":
        mc_map = mc_samples_at_maturities(config, [T_sel])
        mc_slice = mc_map.get(float(T_sel))

    f_truth = reference_pdf_row(
        case,
        T_sel,
        K_sel,
        phi_truth[i_last],
        float(config.r),
        float(config.S0),
        float(config.K_max),
        D2,
        sigma_const,
        mc_samples=mc_slice,
    )
    f_rrr = pdf_from_phi_tilde(
        phi_rrr[i_last], T_sel, config.r, config.S0, config.K_max, K_sel, D2
    )
    f_ql = pdf_from_phi_tilde(
        phi_ql[i_last], T_sel, config.r, config.S0, config.K_max, K_sel, D2
    )
    phi_filt = spectral_pdf_filter(phi_rrr[i_last], chosen)
    f_rrr_filt = pdf_from_phi_tilde(
        phi_filt, T_sel, config.r, config.S0, config.K_max, K_sel, D2
    )

    pdf_ref = (
        "mc_kde"
        if case == "paper" and pdf_reference_is_mc_kde("paper", mc_slice, K_sel)
        else ("lognormal" if case == "constant" else "bl_phi_truth")
    )

    f_bl_data = pdf_from_phi_tilde(
        spectral_pdf_filter(phi_data[i_last], chosen),
        T_sel,
        float(config.r),
        float(config.S0),
        float(config.K_max),
        K_sel,
        D2,
    )
    pdf_metrics_bl_phi_data = (
        pdf_metrics(f_bl_data, f_truth, K_sel, k_mask=km)
        if vol_mode == "data"
        else None
    )

    summary: dict = {
        "case": case,
        "vol_mode": vol_mode,
        "vol_iterate": int(vol_iterate),
        "vol_iterations_run": st.get("vol_iterations_run", 0),
        "chosen_keep": chosen,
        "nu_ql_global": float(nu_g),
        "l2_phi_rrr_vs_truth": phi_l2_error(phi_rrr, phi_truth, grid.k_tilde),
        "l2_phi_rrr_ql_vs_truth": phi_l2_error(phi_ql, phi_truth, grid.k_tilde),
        "pdf_reference": pdf_ref,
        "pdf_metrics_bl_phi_data_vs_ref": pdf_metrics_bl_phi_data,
        "pdf_metrics_rrr_vs_ref": pdf_metrics(f_rrr, f_truth, K_sel),
        "pdf_metrics_ql_vs_ref": pdf_metrics(f_ql, f_truth, K_sel),
        "pdf_metrics_rrr": pdf_metrics(f_rrr, f_truth, K_sel),
        "pdf_metrics_ql": pdf_metrics(f_ql, f_truth, K_sel),
        "pdf_metrics_rrr_spectral_filtered": pdf_metrics(f_rrr_filt, f_truth, K_sel),
        "pdf_metrics_rrr_interior_K": pdf_metrics(f_rrr, f_truth, K_sel, k_mask=km),
        "pdf_metrics_ql_interior_K": pdf_metrics(f_ql, f_truth, K_sel, k_mask=km),
        "uuu_gap_pdf": uuu_gap_metric(
            pdf_metrics(f_rrr, f_truth, K_sel)["l2_pdf"],
            pdf_metrics(f_ql, f_truth, K_sel)["l2_pdf"],
        ),
        "truncation_metadata": meta_path,
        "pdf_slice_T": T_sel,
    }
    if uuu_dim is not None:
        summary["uuu_dimension_json"] = st.get("uuu_dimension_json")
        summary["ambient_dim_u"] = uuu_dim.get("ambient_dim_u")
        summary["uuu_dim_pca_phi_u"] = uuu_dim["pca_phi_u"]["dim"]
        summary["uuu_dim_pca_l_uu"] = uuu_dim["pca_l_uu_phi_u"]["dim"]
        fnn = uuu_dim.get("fnn_autoencoder", {})
        if not fnn.get("skipped"):
            summary["uuu_dim_fnn"] = fnn.get("dim")
        summary["uuu_d_eff_recommended"] = uuu_dim.get("d_eff_recommended")
        summary["uuu_keep_suggest"] = uuu_dim.get("keep_suggest_from_d_eff")

    if model_dir and os.path.isdir(model_dir):
        try:
            f_nn_map = compare_to_nn_pdf(model_dir, config, [T_sel], K_sel)
            f_nn = f_nn_map[float(T_sel)]
            summary["pdf_metrics_nn"] = pdf_metrics(f_nn, f_truth, K_sel)
        except Exception as e:
            summary["nn_error"] = str(e)

    if vol_mode == "data" and sigma_data is not None:
        summary["sigma_data_mean_interior"] = float(
            np.nanmean(sigma_data[np.isfinite(sigma_data)])
        )

    if plot_surfaces:
        if vol_mode == "data" and sigma_data is not None:
            sigma_ref = sigma_data
            sigma_ref_label = "σ from data (Dupire)"
        else:
            sigma_ref = sigma_oracle
            sigma_ref_label = "Exact σ(K,T)"
        vol_art = plot_mz_volatility_surfaces(
            grid,
            sigma_ref,
            phi_rrr,
            phi_ql,
            D2,
            float(config.T_max),
            out_dir,
            case_label=case,
            chosen_keep=chosen,
            sigma_ref_label=sigma_ref_label,
            sigma_oracle=sigma_oracle if vol_mode == "data" else None,
            vol_mode=vol_mode,
        )
        summary.update(vol_art)

    if plot_pdf:
        mc_data = mc_samples_at_maturities(config, pdf_maturities)
        nn_f_by_T = None
        if model_dir and os.path.isdir(model_dir):
            nn_f_by_T = {}
            for T_req in pdf_maturities:
                idx = nearest_time_index(grid.T, T_req)
                T_act = float(grid.T[idx])
                K_row = grid.K[idx]
                try:
                    m = compare_to_nn_pdf(model_dir, config, [T_act], K_row)
                    nn_f_by_T[float(T_req)] = m[float(T_act)]
                except Exception:
                    pass
            if not nn_f_by_T:
                nn_f_by_T = None

        fig, pdf_res = create_mz_pdf_analysis(
            grid,
            phi_data if vol_mode == "data" else phi_truth,
            phi_rrr,
            phi_ql,
            D2,
            config,
            pdf_maturities,
            chosen,
            case,
            sigma_const,
            out_dir,
            case_label=case,
            mc_data=mc_data,
            nn_f_by_T=nn_f_by_T,
            vol_mode=vol_mode,
            phi_data=phi_data if vol_mode == "data" else None,
        )
        plt.close(fig)
        summary["pdf_maturities"] = pdf_maturities
        summary["pdf_analysis_png"] = pdf_res["pdf_analysis_png"]
        summary["pdf_analysis_pdf"] = pdf_res["pdf_analysis_pdf"]
        summary["pdf_analysis_summary_json"] = pdf_res["pdf_analysis_summary_json"]
        summary["pdf_metrics_by_maturity_interior_K"] = pdf_res[
            "pdf_metrics_by_maturity_interior_K"
        ]

    summ_path = os.path.join(out_dir, "mz_validation_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    summary["summary_json"] = summ_path
    return summary


def main():
    p = argparse.ArgumentParser(description="MZ / spectral Dupire synthetic validation")
    p.add_argument("--case", choices=["constant", "paper", "both"], default="constant")
    p.add_argument("--fast", action="store_true", help="Smaller MC for paper case")
    p.add_argument("--n-t", type=int, default=24)
    p.add_argument("--n-k", type=int, default=1024)
    p.add_argument(
        "--chosen-keep",
        type=int,
        default=256,
        help="Low-pass truncation index for P_R / spectral PDF filter (require 2*keep < n_k)",
    )
    p.add_argument(
        "--sweep-truncation",
        action="store_true",
        help="Run k-truncation sweep; writes mz_truncation_metadata.json (does not change keep unless --use-pareto-chosen)",
    )
    p.add_argument(
        "--use-pareto-chosen",
        action="store_true",
        help="After sweep, replace chosen keep with Pareto pick (off by default)",
    )
    p.add_argument(
        "--estimate-uuu-dim",
        action="store_true",
        help="PCA + FNN autoencoder scan on φ_U; writes mz_uuu_dimension.json",
    )
    p.add_argument(
        "--uuu-fnn-max-d",
        type=int,
        default=128,
        help="Max bottleneck width in FNN dimension scan",
    )
    p.add_argument(
        "--uuu-fnn-epochs",
        type=int,
        default=120,
        help="Training epochs per bottleneck (early stopping)",
    )
    p.add_argument(
        "--uuu-pca-only",
        action="store_true",
        help="Skip FNN scan; PCA / SVD on φ_U and L_UU φ_U only",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Optional NN checkpoint dir for PDF benchmark / PDF plots",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default=os.path.join(_ROOT, "models", "runs"),
    )
    p.add_argument(
        "--vol-mode",
        choices=["oracle", "data"],
        default="oracle",
        help="oracle: config σ in L, payoff IC; data: Dupire σ(φ_data), P_R φ_data IC",
    )
    p.add_argument(
        "--vol-iterate",
        type=int,
        default=1,
        help="Data mode only: fixed-point iterations on σ (default 1 = single pass)",
    )
    p.add_argument(
        "--vol-iterate-tol",
        type=float,
        default=1e-2,
        help="Data mode: stop σ fixed-point when mean relative change < tol",
    )
    p.add_argument(
        "--plot-surfaces",
        action="store_true",
        help="Write σ(K,T) PNG/PDF (reference, RRR, RRR+QL) under mz_synthetic_<case>/",
    )
    p.add_argument(
        "--plot-pdf",
        action="store_true",
        help="Write pdf_analysis_mz_* PNG/PDF + summary JSON; merges into mz_validation_summary.json",
    )
    p.add_argument(
        "--maturities",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
        help="Physical maturities for PDF panels (nearest grid rows)",
    )
    args = p.parse_args()

    plot_surfaces = args.plot_surfaces
    plot_pdf = args.plot_pdf
    if args.vol_mode == "data" and not (plot_surfaces or plot_pdf):
        plot_surfaces = True
        plot_pdf = True

    cases = ["constant", "paper"] if args.case == "both" else [args.case]
    for c in cases:
        print("=" * 72)
        print(f"Case: {c}")
        s = run_one_case(
            c,
            fast=args.fast,
            model_dir=args.model_dir,
            out_root=args.out_root,
            n_t=args.n_t,
            n_k=args.n_k,
            chosen_keep=args.chosen_keep,
            run_truncation_sweep=args.sweep_truncation,
            use_pareto_chosen=args.use_pareto_chosen,
            estimate_uuu_dim=args.estimate_uuu_dim,
            uuu_fnn_max_d=args.uuu_fnn_max_d,
            uuu_fnn_epochs=args.uuu_fnn_epochs,
            uuu_pca_only=args.uuu_pca_only,
            plot_surfaces=plot_surfaces,
            plot_pdf=plot_pdf,
            pdf_maturities=args.maturities,
            vol_mode=args.vol_mode,
            vol_iterate=args.vol_iterate,
            vol_iterate_tol=args.vol_iterate_tol,
        )
        print(json.dumps(s, indent=2, default=str))
    print("Done.")


if __name__ == "__main__":
    main()
