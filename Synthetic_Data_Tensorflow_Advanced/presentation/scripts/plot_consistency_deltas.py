#!/usr/bin/env python3
"""Plot three-way consistency-check deltas (DeltaM_12, DeltaM_13) vs maturity T.

Reads one or more JSON files produced by the consistency-check pipeline
and overlays them on a single log-y axis. ΔM_12 is always drawn in NTU
navy, ΔM_13 always in NTU red; different models are distinguished by
linestyle. Writes PNG + PDF into the chosen output directory.

Defaults assume the script lives in `presentation/scripts/` of the
Synthetic_Data_Tensorflow_Advanced repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt


# NTU palette
NTU_NAVY = "#00205B"
NTU_RED = "#C02026"

# Linestyles cycled per input series.
LINESTYLES: Sequence[str] = ("-", "--", ":", "-.")
MARKERS: Sequence[Tuple[str, str]] = (
    ("o", "s"), ("D", "^"), ("v", "P"), ("X", "*"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot consistency-check deltas vs maturity (single or multi-model overlay)."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["models/example_pretrained/consistency_check/consistency_check.json"],
        help=(
            "Path(s) to consistency_check.json (relative to repo root). "
            "Pass multiple --input PATHs to overlay them on the same axes."
        ),
    )
    parser.add_argument(
        "--label",
        nargs="*",
        default=None,
        help=(
            "Optional label per --input (same count). If omitted, labels "
            "are derived from each JSON's model_dir field (basename)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="presentation/figures",
        help="Output directory for the PNG/PDF (relative to repo root).",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help=(
            "Basename for the output files (without extension). "
            "Defaults to 'ibp_deltas_vs_T' for one input or "
            "'ibp_deltas_vs_T_compare' for multiple."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Override the figure title.",
    )
    return parser.parse_args()


def load_series(json_path: str) -> Tuple[str, List[Tuple[float, float, float]]]:
    """Return (model_dir_basename, sorted [(T, delta_M12_pct, delta_M13_pct), ...])."""
    with open(json_path, "r") as fh:
        payload = json.load(fh)

    per_maturity = payload["per_maturity"]
    series: List[Tuple[float, float, float]] = []
    for T_key, entry in per_maturity.items():
        T_val = float(T_key)
        series.append(
            (T_val, float(entry["delta_M12_pct"]), float(entry["delta_M13_pct"]))
        )
    series.sort(key=lambda row: row[0])

    raw_model_dir = str(payload.get("model_dir", os.path.dirname(json_path)))
    model_label = os.path.basename(os.path.normpath(raw_model_dir)) or raw_model_dir
    return model_label, series


def main() -> int:
    args = parse_args()

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    os.chdir(repo_root)

    inputs: List[str] = args.input
    if args.label is not None and len(args.label) != len(inputs):
        print(
            f"ERROR: --label count ({len(args.label)}) must match "
            f"--input count ({len(inputs)}).",
            file=sys.stderr,
        )
        return 1

    print(f"Repo root      : {repo_root}")
    print(f"Inputs ({len(inputs)})  :")
    for p in inputs:
        print(f"  - {p}")

    loaded: List[Tuple[str, List[Tuple[float, float, float]]]] = []
    for idx, path in enumerate(inputs):
        if not os.path.isfile(path):
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
            return 1
        derived_label, series = load_series(path)
        if not series:
            print(f"ERROR: no per_maturity entries in {path}.", file=sys.stderr)
            return 1
        label = args.label[idx] if args.label else derived_label
        loaded.append((label, series))
        print(f"\nLoaded '{label}' from {path}:")
        for T_val, m12, m13 in series:
            print(
                f"  T = {T_val:.4f}  ->  DeltaM_12 = {m12:.6f} %,  "
                f"DeltaM_13 = {m13:.6f} %"
            )

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(6.0, 3.6), facecolor="white")

    for idx, (label, series) in enumerate(loaded):
        Ts = [row[0] for row in series]
        d12 = [row[1] for row in series]
        d13 = [row[2] for row in series]
        ls = LINESTYLES[idx % len(LINESTYLES)]
        marker_12, marker_13 = MARKERS[idx % len(MARKERS)]
        label_12 = rf"$\Delta M_{{12}}$ -- {label}"
        label_13 = rf"$\Delta M_{{13}}$ -- {label}"
        ax.plot(
            Ts, d12,
            color=NTU_NAVY, linestyle=ls, marker=marker_12,
            linewidth=2, markersize=7, label=label_12,
        )
        ax.plot(
            Ts, d13,
            color=NTU_RED, linestyle=ls, marker=marker_13,
            linewidth=2, markersize=7, label=label_13,
        )

    # Reference thresholds (in percent).
    ax.axhline(1.0, ls="--", color="0.5", lw=1)
    ax.axhline(2.0, ls=":", color="0.5", lw=1)
    ax.text(
        0.99, 0.34, "IBP threshold 1%",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="0.35",
    )
    ax.text(
        0.99, 0.42, "Dupire threshold 2%",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="0.35",
    )

    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e2)

    # Tight x-axis padding across all maturities.
    all_Ts = [T for _, series in loaded for T, _, _ in series]
    if len(set(all_Ts)) >= 2:
        span = max(all_Ts) - min(all_Ts)
        pad = max(0.05, 0.08 * span)
        ax.set_xlim(min(all_Ts) - pad, max(all_Ts) + pad)

    ax.set_xlabel(r"Maturity $T$ (years)")
    ax.set_ylabel(r"$|\,\mathrm{discrepancy}\,|$  (%)")

    if args.title is not None:
        title = args.title
    elif len(loaded) == 1:
        title = (
            "Three-way consistency vs maturity\n"
            rf"({loaded[0][0]}, transformed $\tilde{{\varphi}}$)"
        )
    else:
        title = (
            "Three-way consistency vs maturity\n"
            "(constant vs manuscript $\\sigma$, repriced MC)"
        )
    ax.set_title(title, fontsize=10)

    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="center right", fontsize=7, framealpha=0.9)

    plt.tight_layout()

    # ------------------------------------------------------------------- save
    if args.basename is not None:
        basename = args.basename
    elif len(loaded) == 1:
        basename = "ibp_deltas_vs_T"
    else:
        basename = "ibp_deltas_vs_T_compare"

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{basename}.{ext}")
        fig.savefig(
            path,
            dpi=450,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.1,
        )
        print(f"  Saved: {path}")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
