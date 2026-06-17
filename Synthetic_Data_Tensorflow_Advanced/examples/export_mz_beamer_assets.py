#!/usr/bin/env python3
"""
Export high-res individual MZ panels and copy into HW/presentation/figures/mz_local_vol/.

  cd Synthetic_Data_Tensorflow_Advanced
  MPLBACKEND=Agg python examples/export_mz_beamer_assets.py --fast
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mz_spectral.export_panels import (
    export_pdf_panels,
    export_vol_surface_panels,
    export_vol_surfaces_triple_panel,
)
from mz_spectral.mz_synthetic_setup import load_chosen_keep_from_artifacts, setup_mz_synthetic_case
from mz_spectral.plot_pdf_analysis import mc_samples_at_maturities

_DEFAULT_HW_FIG = os.path.normpath(
    os.path.join(
        _ROOT,
        "..",
        "..",
        "HW",
        "presentation",
        "figures",
        "mz_local_vol",
    )
)


def main():
    p = argparse.ArgumentParser(description="Export MZ Beamer panel figures")
    p.add_argument("--case", choices=["constant", "paper", "both"], default="both")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--n-t", type=int, default=24)
    p.add_argument("--n-k", type=int, default=1024)
    p.add_argument("--chosen-keep", type=int, default=256)
    p.add_argument(
        "--sweep-truncation",
        action="store_true",
        help="Optional truncation sweep (metadata only unless --use-pareto-chosen)",
    )
    p.add_argument(
        "--use-pareto-chosen",
        action="store_true",
        help="After sweep, use Pareto-picked keep",
    )
    p.add_argument("--maturities", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    p.add_argument("--out-root", type=str, default=os.path.join(_ROOT, "models", "runs"))
    p.add_argument(
        "--hw-figures-dir",
        type=str,
        default=_DEFAULT_HW_FIG,
        help="Copy PNGs here for Beamer (HW/presentation/figures/mz_local_vol)",
    )
    p.add_argument("--replot-only", action="store_true")
    args = p.parse_args()

    cases = ["constant", "paper"] if args.case == "both" else [args.case]
    manifest: dict = {"cases": {}, "hw_figures_dir": args.hw_figures_dir}

    for case in cases:
        out_dir_run = os.path.join(args.out_root, f"mz_synthetic_{case}", "panels_hires")
        chosen_override = None
        skip = False
        if args.replot_only:
            chosen_override = load_chosen_keep_from_artifacts(
                os.path.join(args.out_root, f"mz_synthetic_{case}")
            )
            if chosen_override is None:
                raise SystemExit(f"No chosen_keep artifacts for {case}")
            skip = True
            if int(args.chosen_keep) != int(chosen_override):
                print(
                    f"Warning [{case}]: --replot-only uses chosen_keep={chosen_override} from "
                    f"artifacts; ignoring --chosen-keep {args.chosen_keep}.",
                    file=sys.stderr,
                )

        st = setup_mz_synthetic_case(
            case,
            fast=args.fast,
            n_t=args.n_t,
            n_k=args.n_k,
            out_root=args.out_root,
            chosen_keep=args.chosen_keep,
            chosen_keep_override=chosen_override,
            skip_truncation_sweep=skip,
            run_truncation_sweep=bool(args.sweep_truncation) and not skip,
            use_pareto_chosen=bool(args.use_pareto_chosen) and not skip,
        )
        mc_data = mc_samples_at_maturities(st["config"], list(args.maturities))
        ck = int(st["chosen_keep"])
        vol_paths = export_vol_surface_panels(
            st["grid"],
            st["sigma"],
            st["phi_rrr"],
            st["phi_ql"],
            st["D2"],
            float(st["config"].T_max),
            out_dir_run,
            case,
            chosen_keep=ck,
        )
        triple = export_vol_surfaces_triple_panel(
            st["grid"],
            st["sigma"],
            st["phi_rrr"],
            st["phi_ql"],
            st["D2"],
            float(st["config"].T_max),
            out_dir_run,
            case,
            chosen_keep=ck,
        )
        vol_paths["surfaces_triple"] = triple
        pdf_paths = export_pdf_panels(
            st["grid"],
            st["phi_truth"],
            st["phi_rrr"],
            st["phi_ql"],
            st["D2"],
            st["config"],
            list(args.maturities),
            st["chosen_keep"],
            case,
            st["sigma_const"],
            out_dir_run,
            mc_data=mc_data,
        )
        manifest["cases"][case] = {
            "source_dir": out_dir_run,
            "vol_panels": vol_paths,
            "pdf_panels": pdf_paths,
        }

    os.makedirs(args.hw_figures_dir, exist_ok=True)
    copied = []
    for case, blob in manifest["cases"].items():
        src = blob["source_dir"]
        for fn in os.listdir(src):
            if not fn.endswith(".png"):
                continue
            dest = os.path.join(args.hw_figures_dir, fn)
            shutil.copy2(os.path.join(src, fn), dest)
            copied.append(fn)
    manifest["copied_png_count"] = len(copied)
    manifest_path = os.path.join(args.hw_figures_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"manifest": manifest_path, "n_png": len(copied)}, indent=2))


if __name__ == "__main__":
    main()
