#!/usr/bin/env python3
"""Per-cell pass counts behind the PLAN §10 classes (memo support): for each closed-model entry and split,
how many (case, protocol) cells pass each individual condition (resolved beyond 2u_dyn vs RRR_m; price beyond
2u_price vs RRR_m; noninferiority in density, negative mass (strict + gamma), the four raw arbitrage fractions,
inverse-vol invalid fraction and coverage), plus the geometric-mean ratios to RRR_m. Reads results/score and
results/uncertainty; writes results/tables/cell_counts.md and results/tables/cell_counts.json."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import scc  # noqa: E402
from scc import cases as cs, identity  # noqa: E402

ENTRIES = [("QLp", 9, 0, None), ("QLpm", 9, 0, None), ("QLs", 9, 0, None), ("LM", 9, 0, None), ("LMb", 9, 0, None),
           ("LR0", 9, 16, "W0"), ("LR0", 9, 16, "WI"), ("LR0", 9, 16, "WU"), ("LR", 9, 16, "W0"), ("LR", 9, 16, "WI"), ("LR", 9, 16, "WU"),
           ("QLp", 17, 0, None), ("QLpm", 17, 0, None), ("QLs", 17, 0, None), ("LM", 17, 0, None), ("LMb", 17, 0, None),
           ("LR0", 17, 8, "W0"), ("LR0", 17, 8, "WI"), ("LR0", 17, 8, "WU"), ("LR", 17, 8, "W0"), ("LR", 17, 8, "WI"), ("LR", 17, 8, "WU"),
           ("LM", 33, 0, None), ("QLs", 33, 0, None)]
FRACS = ["positivity_raw_interior_frac_mean", "monotonicity_raw_interior_frac_mean", "butterfly_raw_interior_frac_mean",
         "calendar_raw_interior_frac_mean"]


def key(cl, m, nh, proto, hidden):
    k = f"{cl}_m{m}" + (f"_h{nh}" if nh else "") + f"__{proto}"
    if hidden and proto == "P1":
        k += f"_{hidden}"
    return k


def main():
    gamma = (identity.read_json(os.path.join(scc.RESULTS, "verify", "roundoff_guard.json")) or {}).get("gamma", 0.0)
    out = {}
    lines = ["# Per-cell condition counts (cells = cases x protocols {P0, P1}; closed model vs RRR at the same m)", "",
             f"gamma = {gamma:.2e}; tolerances: beyond = 2u, noninferior = u, fractions = max(u, 1/N_interior), coverage >= RRR - tol.", ""]
    for split in ("test_interp", "test_ood", "val", "train"):
        ids = [c.id for c in cs.cases_by_split(split)]
        lines += [f"## {split} ({len(ids)} cases)", "",
                  "| entry | cells | resolved>2u | price>2u | dens nonin | negmass strict | pos | mono | butt | cal | invvol nonin | cov nonin | gm resolved/RRR | gm price/RRR | gm dens/RRR | unstable |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for cl, m, nh, hidden in ENTRIES:
            name = f"{cl}_m{m}" + (f"_h{nh}" if nh else "") + (f"/{hidden}" if hidden else "")
            cnt = {k: 0 for k in ("cells", "res", "price", "dens", "neg", "pos", "mono", "butt", "cal", "inv", "cov", "unst")}
            lr, lp, ld = [], [], []
            for cid in ids:
                p = os.path.join(scc.RESULTS, "score", cid + ".json")
                if not os.path.isfile(p):
                    continue
                d = json.load(open(p))
                U = d["uncertainty"]["per_quantity"]
                for proto in ("P0", "P1"):
                    row = d["models"].get(key(cl, m, nh, proto, hidden))
                    base = d["models"].get(f"RRR_m{m}__{proto}")
                    if row is None or base is None:
                        continue
                    cnt["cells"] += 1
                    if row.get("unstable"):
                        cnt["unst"] += 1
                        continue
                    x, y = row["metrics"], base["metrics"]
                    udyn = d["uncertainty"]["u_dyn"][str(m)]
                    up, ud = U["price_interior_rms"]["u"], U["density_l2_interior_mean"]["u"]
                    cnt["res"] += int(x["resolved_interior_rms"] < y["resolved_interior_rms"] - 2 * udyn)
                    cnt["price"] += int(x["price_interior_rms"] < y["price_interior_rms"] - 2 * up)
                    cnt["dens"] += int(x["density_l2_interior_mean"] <= y["density_l2_interior_mean"] + ud)
                    cnt["neg"] += int(x["negative_mass_max_post_initial"] <= y["negative_mass_max_post_initial"] + gamma)
                    n_int = 230 * 24
                    for f, k in zip(FRACS, ("pos", "mono", "butt", "cal")):
                        tol = max(U.get(f, {}).get("u", 0.0), 1.0 / n_int)
                        cnt[k] += int(x[f] <= y[f] + tol)
                    tol = max(U["invvol_invalid_frac_full"]["u"], 1.0 / n_int)
                    cnt["inv"] += int(x["invvol_invalid_frac_full"] <= y["invvol_invalid_frac_full"] + tol)
                    tol = max(U["invvol_coverage"]["u"], 1.0 / n_int)
                    cnt["cov"] += int(x["invvol_coverage"] >= y["invvol_coverage"] - tol)
                    lr.append(x["resolved_interior_rms"] / y["resolved_interior_rms"])
                    lp.append(x["price_interior_rms"] / y["price_interior_rms"])
                    ld.append(x["density_l2_interior_mean"] / y["density_l2_interior_mean"])
            gm = lambda L: float(np.exp(np.mean(np.log(L)))) if L else float("nan")
            out.setdefault(split, {})[name] = {**cnt, "gm_resolved_ratio": gm(lr), "gm_price_ratio": gm(lp), "gm_density_ratio": gm(ld)}
            n = cnt["cells"]
            lines.append(f"| {name} | {n} | {cnt['res']} | {cnt['price']} | {cnt['dens']} | {cnt['neg']} | {cnt['pos']} | {cnt['mono']} | {cnt['butt']} | {cnt['cal']} | {cnt['inv']} | {cnt['cov']} | {gm(lr):.2f} | {gm(lp):.3f} | {gm(ld):.3f} | {cnt['unst']} |")
        lines.append("")
    os.makedirs(os.path.join(scc.RESULTS, "tables"), exist_ok=True)
    with open(os.path.join(scc.RESULTS, "tables", "cell_counts.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    identity.write_json(os.path.join(scc.RESULTS, "tables", "cell_counts.json"), out)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
