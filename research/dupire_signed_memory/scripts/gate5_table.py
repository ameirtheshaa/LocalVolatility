#!/usr/bin/env python3
"""Tabulate Gate 5 (boundary-adapted reduced model) with uncertainty-aware comparisons.
Writes results/gate5/TABLE.md and results/gate5/comparison.json."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsm import checkpoint  # noqa: E402

OUT = os.path.join(checkpoint.RESULTS, "gate5")


def fmt(x, p=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{p}e}"


def main():
    md = ["# Gate 5 — boundary-adapted reduced model vs Fourier model vs refined full solve", "",
          "Reference: boundary-consistent full solve at n_k = 512 (interpolated to the 256 grid); for the constant-σ control also Black–Scholes. "
          "u_grid = interior RMS difference between the 256 and 512 full solves (numerical uncertainty); an improvement is claimed only when it exceeds u_grid.", ""]
    comp = {}
    for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        d = json.load(open(p))
        if d.get("status") != "done":
            continue
        cid = d["case"]
        u = d["uncertainty"]
        ug = u["grid_256_vs_512_interior_rms"]
        md += [f"## {cid} ({d['family']}, {d['params']})", "",
               f"u_grid = {fmt(ug)}; density u = {fmt(u['grid_256_vs_512_density_l2'])}" + (f"; step u = {fmt(u['step_sub16_vs_sub32_interior_rms'])}" if "step_sub16_vs_sub32_interior_rms" in u else "") + (f"; full512 vs BS = {fmt(u['full512_vs_bs_interior_rms'])}" if "full512_vs_bs_interior_rms" in u else ""), "",
               "| model | start | dof | price int. RMS | price vs BS | density L2 | neg. mass max | neg. mass T_max | inverse-vol invalid | butterfly viol. | stable | reduced abscissa | proj. err | rollout s |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = {}
        for name, m in d["models"].items():
            if "price_interior_rms" not in m:
                md.append(f"| {name} | {m.get('start','')} | {m.get('dof','')} | – | – | – | – | – | – | – | {'no' if m.get('unstable') else 'yes'} | {fmt(m.get('reduced_operator_abscissa'))} | – | {fmt(m.get('wall_seconds_rollout_median', m.get('wall_seconds_rollout')), 3)} |")
                continue
            rows[name] = m
            md.append(f"| {name.split('__')[0]} | {m.get('start','')} | {m['dof']} | {fmt(m['price_interior_rms'])} | {fmt(m.get('price_interior_rms_vs_bs'))} | {fmt(m['density_l2_interior_mean'])} | {fmt(m['negative_mass_max'])} | {fmt(m['negative_mass_T_max'])} | {m['inverse_vol_invalid_frac_mean']:.3f} | {m['butterfly_violation_frac_mean']:.3f} | {'no' if m['unstable'] else 'yes'} | {fmt(m.get('reduced_operator_abscissa'))} | {fmt(m.get('projection_err_interior_rms'))} | {fmt(m.get('wall_seconds_rollout_median', m.get('wall_seconds_rollout')), 3)} |")
        md.append("")
        # uncertainty-aware comparisons
        c = {"u_grid": ug, "claims": []}
        full = rows.get("B_full256", {})
        for proto in ("from_projected_t0.2", "from_payoff_t0"):
            for m_s in (17, 33, 65):
                s = rows.get(f"B_sine_m{m_s}__{proto}")
                if not s:
                    continue
                for m_f in (33, 65):
                    for var in ("A", "B"):
                        f = rows.get(f"{var}_fourier_m{m_f}__{proto}")
                        if not f or f["unstable"]:
                            continue
                        gain = f["price_interior_rms"] - s["price_interior_rms"]
                        c["claims"].append({"protocol": proto, "sine_dof": m_s, "fourier": f"{var}_m{m_f}",
                                            "price_gain": gain, "exceeds_u": bool(gain > ug),
                                            "density_gain": f["density_l2_interior_mean"] - s["density_l2_interior_mean"],
                                            "density_exceeds_u": bool(f["density_l2_interior_mean"] - s["density_l2_interior_mean"] > u["grid_256_vs_512_density_l2"]),
                                            "negmass_fourier": f["negative_mass_max"], "negmass_sine": s["negative_mass_max"],
                                            "sine_within_u_of_full256": bool(abs(s["price_interior_rms"] - full.get("price_interior_rms", np.nan)) <= ug) if full else None,
                                            "speedup_vs_full256": (full.get("wall_seconds_rollout", np.nan) / s["wall_seconds_rollout_median"]) if full else None})
        comp[cid] = c
        md += ["Uncertainty-aware claims (price interior RMS):", ""]
        for cl in c["claims"]:
            md.append(f"- {cl['protocol']}: sine m={cl['sine_dof']} vs Fourier {cl['fourier']}: gain {fmt(cl['price_gain'])} ({'exceeds' if cl['exceeds_u'] else 'inside'} u_grid); "
                      f"density gain {fmt(cl['density_gain'])} ({'exceeds' if cl['density_exceeds_u'] else 'inside'} density u); negative mass {fmt(cl['negmass_fourier'])} → {fmt(cl['negmass_sine'])}; "
                      f"sine within u of full-256: {cl['sine_within_u_of_full256']}; speed-up vs full-256 rollout: {cl['speedup_vs_full256']:.1f}×" if cl['speedup_vs_full256'] else "")
        md.append("")
    checkpoint.write_json(os.path.join(OUT, "comparison.json"), comp)
    with open(os.path.join(OUT, "TABLE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {OUT}/TABLE.md and comparison.json")


if __name__ == "__main__":
    main()
