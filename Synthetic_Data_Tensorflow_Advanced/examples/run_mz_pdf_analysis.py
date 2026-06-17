#!/usr/bin/env python3
"""
Standalone MZ Breeden–Litzenberger PDF analysis (multi-maturity three-panel figures).

  cd Synthetic_Data_Tensorflow_Advanced
  MPLBACKEND=Agg python examples/run_mz_pdf_analysis.py --case constant --fast \\
      --maturities 0.5 1.0 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mz_spectral.mz_synthetic_setup import (  # noqa: E402
    load_chosen_keep_from_artifacts,
    setup_mz_synthetic_case,
)
from mz_spectral.plot_pdf_analysis import (  # noqa: E402
    create_mz_pdf_analysis,
    mc_samples_at_maturities,
    nearest_time_index,
)
from mz_spectral.validation import compare_to_nn_pdf  # noqa: E402


def _mc_from_training_npz(
    data_path: str,
    maturities: list[float],
) -> dict[float, np.ndarray]:
    data = np.load(data_path)
    S_matrix = data["S_matrix"]
    t_all = data["t_all"].flatten()
    out: dict[float, np.ndarray] = {}
    for T in maturities:
        idx = int(np.argmin(np.abs(t_all - T)))
        out[float(T)] = np.asarray(S_matrix[idx, :], dtype=float)
    return out


def run_pdf_case(
    case: str,
    *,
    fast: bool,
    n_t: int,
    n_k: int,
    chosen_keep: int,
    chosen_keep_cli: int | None,
    run_truncation_sweep: bool,
    use_pareto_chosen: bool,
    out_root: str,
    maturities: list[float],
    replot_only: bool,
    training_data: str | None,
    with_nn: bool,
    model_dir: str | None,
    m_train: int | None,
) -> dict:
    out_dir = os.path.join(out_root, f"mz_synthetic_{case}")
    skip = False
    chosen_override = None
    if replot_only:
        chosen_override = load_chosen_keep_from_artifacts(out_dir)
        if chosen_override is None:
            raise SystemExit(
                f"--replot-only: no chosen_keep in {out_dir}/mz_validation_summary.json "
                f"or mz_truncation_metadata.json"
            )
        skip = True
        if (
            chosen_keep_cli is not None
            and int(chosen_keep_cli) != int(chosen_override)
        ):
            print(
                f"Warning [{case}]: --replot-only uses chosen_keep={chosen_override} from "
                f"artifacts; ignoring --chosen-keep {chosen_keep_cli} (re-run full driver to "
                f"regenerate with a different keep).",
                file=sys.stderr,
            )

    st = setup_mz_synthetic_case(
        case,
        fast=fast,
        n_t=n_t,
        n_k=n_k,
        out_root=out_root,
        chosen_keep=chosen_keep,
        chosen_keep_override=chosen_override,
        skip_truncation_sweep=skip,
        run_truncation_sweep=run_truncation_sweep and not skip,
        use_pareto_chosen=use_pareto_chosen and not skip,
        m_train=m_train,
    )
    config = st["config"]
    grid = st["grid"]
    phi_truth = st["phi_truth"]
    phi_rrr = st["phi_rrr"]
    phi_ql = st["phi_ql"]
    chosen = st["chosen_keep"]
    D2 = st["D2"]
    sigma_const = st["sigma_const"]
    out_dir = st["out_dir"]

    if training_data and os.path.isfile(training_data):
        mc_data = _mc_from_training_npz(training_data, maturities)
    else:
        mc_data = mc_samples_at_maturities(config, maturities)

    nn_f_by_T = None
    if with_nn and model_dir and os.path.isdir(model_dir):
        nn_f_by_T = {}
        for T_req in maturities:
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
        phi_truth,
        phi_rrr,
        phi_ql,
        D2,
        config,
        maturities,
        chosen,
        case,
        sigma_const,
        out_dir,
        case_label=case,
        mc_data=mc_data,
        nn_f_by_T=nn_f_by_T,
    )
    plt.close(fig)

    out = {**pdf_res, "out_dir": out_dir, "case": case}
    sidecar = os.path.join(out_dir, "mz_pdf_driver_last.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    out["driver_sidecar_json"] = sidecar
    return out


def main():
    p = argparse.ArgumentParser(description="MZ PDF analysis (multi-maturity BL)")
    p.add_argument("--case", choices=["constant", "paper", "both"], default="constant")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--n-t", type=int, default=24)
    p.add_argument("--n-k", type=int, default=1024)
    p.add_argument(
        "--chosen-keep",
        type=int,
        default=None,
        help="Low-pass keep for P_R / PDF filter (default 256; ignored under --replot-only except for this warning)",
    )
    p.add_argument(
        "--sweep-truncation",
        action="store_true",
        help="Optional truncation sweep (writes metadata; keep fixed unless --use-pareto-chosen)",
    )
    p.add_argument(
        "--use-pareto-chosen",
        action="store_true",
        help="After sweep, use Pareto-picked keep instead of --chosen-keep",
    )
    p.add_argument("--maturities", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    p.add_argument(
        "--out-root",
        type=str,
        default=os.path.join(_ROOT, "models", "runs"),
    )
    p.add_argument(
        "--replot-only",
        action="store_true",
        help="Skip k-truncation sweep; load chosen_keep from prior mz artifacts",
    )
    p.add_argument(
        "--training-data",
        type=str,
        default=None,
        help="Optional path to training_data.npz for MC histograms",
    )
    p.add_argument("--n-paths", type=int, default=None, help="Override M_train (MC paths)")
    p.add_argument("--with-nn", action="store_true", help="Overlay NN density when weights load")
    p.add_argument("--model-dir", type=str, default=None)
    args = p.parse_args()

    chosen_keep = int(args.chosen_keep) if args.chosen_keep is not None else 256

    cases = ["constant", "paper"] if args.case == "both" else [args.case]
    for c in cases:
        summ = run_pdf_case(
            c,
            fast=bool(args.fast),
            n_t=args.n_t,
            n_k=args.n_k,
            chosen_keep=chosen_keep,
            chosen_keep_cli=args.chosen_keep,
            run_truncation_sweep=bool(args.sweep_truncation),
            use_pareto_chosen=bool(args.use_pareto_chosen),
            out_root=args.out_root,
            maturities=list(args.maturities),
            replot_only=args.replot_only,
            training_data=args.training_data,
            with_nn=args.with_nn,
            model_dir=args.model_dir,
            m_train=args.n_paths,
        )
        print(json.dumps({k: summ[k] for k in summ if k != "pdf_metrics_by_maturity_interior_K"}, indent=2, default=str))
    print("Done.")


if __name__ == "__main__":
    main()
