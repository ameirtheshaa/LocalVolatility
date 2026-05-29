#!/usr/bin/env python3
"""Compute the IBP three-way consistency diagnostic (DeltaM_12, DeltaM_13) from
one or more diagnostics.json files produced by `run_reproduce_user_plots.py`,
emit a comparison table to stdout, a comparison figure (PNG + PDF), and a
flat summary JSON.

Per maturity T, given M_1=mean_nn_density_wide, M_2=spot_call,
M_3=mean_mc, M_ref=theoretical_ref:

    DeltaM_12 = |M_1 - M_2| / M_2 * 100 %     # IBP identity check
    DeltaM_13 = |M_1 - M_3| / M_3 * 100 %     # price-vol Dupire check

Usage:
    python examples/compute_ibp_deltas.py \\
        --input PATH [PATH ...] \\
        --label STR [STR ...] \\
        [--output-dir DIR]
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt


# Match plot_consistency_deltas.py
NTU_NAVY = "#00205B"
NTU_RED = "#C02026"
LINESTYLES = ("-", "--", ":", "-.")
MARKERS = (("o", "s"), ("D", "^"), ("v", "P"), ("X", "*"))


def load_results(path: str):
    with open(path, "r") as fh:
        payload = json.load(fh)
    if "results" in payload:
        results = payload["results"]
    elif "per_maturity" in payload:
        results = payload["per_maturity"]
    else:
        raise KeyError(f"{path} has neither 'results' nor 'per_maturity' key.")

    rows = []
    for T_key, res in results.items():
        try:
            T = float(T_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(res, dict):
            continue
        M1 = res.get("mean_nn_density_wide") or res.get("M_1")
        M2 = res.get("spot_call") or res.get("M_2")
        M3 = res.get("mean_mc") or res.get("M_3")
        Mref = res.get("theoretical_ref") or res.get("M_ref")
        if None in (M1, M2, M3, Mref):
            continue
        M1, M2, M3, Mref = float(M1), float(M2), float(M3), float(Mref)
        rows.append({
            "T": T,
            "M_ref": Mref,
            "M_1": M1, "M_2": M2, "M_3": M3,
            "pct_err_M1": (M1 - Mref) / Mref * 100.0,
            "pct_err_M2": (M2 - Mref) / Mref * 100.0,
            "pct_err_M3": (M3 - Mref) / Mref * 100.0,
            "delta_M12_pct": abs(M1 - M2) / abs(M2) * 100.0 if M2 else float("nan"),
            "delta_M13_pct": abs(M1 - M3) / abs(M3) * 100.0 if M3 else float("nan"),
        })
    rows.sort(key=lambda r: r["T"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True,
                        help="diagnostics.json paths.")
    parser.add_argument("--label", nargs="*", default=None,
                        help="Optional label per --input.")
    parser.add_argument("--output-dir", default="plots/ibp_deltas_regen")
    parser.add_argument("--basename", default="ibp_deltas_regen")
    args = parser.parse_args()

    if args.label is not None and len(args.label) != len(args.input):
        print(
            f"ERROR: --label count ({len(args.label)}) must match "
            f"--input count ({len(args.input)}).",
            file=sys.stderr,
        )
        return 1

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    loaded = []
    for i, path in enumerate(args.input):
        if not os.path.isfile(path):
            print(f"ERROR: not found: {path}", file=sys.stderr)
            return 1
        label = args.label[i] if args.label else os.path.basename(os.path.dirname(os.path.abspath(path)))
        rows = load_results(path)
        if not rows:
            print(f"ERROR: no usable per-maturity rows in {path}", file=sys.stderr)
            return 1
        loaded.append({"label": label, "path": path, "rows": rows})

    # -------- table --------
    header = (
        f"  {'label':<22} | {'T':>5} | {'M_ref':>9} | "
        f"{'M_1':>9} | {'%err':>6} | {'M_2':>9} | {'%err':>6} | "
        f"{'M_3':>9} | {'%err':>6} | {'dM12 %':>8} | {'dM13 %':>8}"
    )
    print()
    print("=" * len(header))
    print("IBP three-way diagnostic — recomputed on regenerated MC")
    print("=" * len(header))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for entry in loaded:
        for r in entry["rows"]:
            print(
                f"  {entry['label']:<22} | {r['T']:>5.2f} | "
                f"{r['M_ref']:>9.2f} | "
                f"{r['M_1']:>9.2f} | {r['pct_err_M1']:>+6.2f} | "
                f"{r['M_2']:>9.2f} | {r['pct_err_M2']:>+6.2f} | "
                f"{r['M_3']:>9.2f} | {r['pct_err_M3']:>+6.2f} | "
                f"{r['delta_M12_pct']:>8.4f} | {r['delta_M13_pct']:>8.4f}"
            )
        print("  " + "-" * (len(header) - 2))

    # -------- figure --------
    os.makedirs(args.output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="white")
    for i, entry in enumerate(loaded):
        Ts = [r["T"] for r in entry["rows"]]
        d12 = [r["delta_M12_pct"] for r in entry["rows"]]
        d13 = [r["delta_M13_pct"] for r in entry["rows"]]
        ls = LINESTYLES[i % len(LINESTYLES)]
        m12, m13 = MARKERS[i % len(MARKERS)]
        ax.plot(Ts, d12, color=NTU_NAVY, linestyle=ls, marker=m12,
                linewidth=2, markersize=7,
                label=rf"$\Delta M_{{12}}$ -- {entry['label']}")
        ax.plot(Ts, d13, color=NTU_RED, linestyle=ls, marker=m13,
                linewidth=2, markersize=7,
                label=rf"$\Delta M_{{13}}$ -- {entry['label']}")

    ax.axhline(1.0, ls="--", color="0.5", lw=1)
    ax.axhline(2.0, ls=":", color="0.5", lw=1)
    ax.text(0.99, 0.32, "IBP threshold 1%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="0.35")
    ax.text(0.99, 0.4, "Dupire threshold 2%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="0.35")

    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e2)
    ax.set_xlabel(r"Maturity $T$ (years)")
    ax.set_ylabel(r"$|\,\mathrm{discrepancy}\,|$  (%)")
    ax.set_title(
        "Three-way IBP consistency vs maturity\n"
        "(regenerated data MC and repriced MC, both paper models)",
        fontsize=10,
    )
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9, ncol=2)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        out_path = os.path.join(args.output_dir, f"{args.basename}.{ext}")
        fig.savefig(out_path, dpi=450, bbox_inches="tight",
                    facecolor="white", pad_inches=0.1)
        print(f"\n  Saved: {out_path}")
    plt.close(fig)

    # -------- JSON --------
    summary = {
        "inputs": [{"label": e["label"], "path": e["path"]} for e in loaded],
        "results": {
            e["label"]: {f"{r['T']:.4f}": r for r in e["rows"]} for e in loaded
        },
    }
    json_path = os.path.join(args.output_dir, f"{args.basename}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
