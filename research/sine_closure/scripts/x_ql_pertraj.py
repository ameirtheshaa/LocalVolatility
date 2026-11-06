#!/usr/bin/env python3
"""QL sensitivity panel (PLAN.md §5): per-trajectory scalar fits nu*_case = <M, Lam c>/||Lam c||^2 on the training
trajectories, their sign counts and spread, next to the pooled fit; plus the validation sensitivity grid written by s4
(results/fit/ql_sensitivity_m<m>.json) when present. Writes results/fit/ql_pertrajectory.json. Development data only."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("SCC_PHASE", "fit")
import scc  # noqa: E402
from scc import cases as cs, identity, reference as ref, targets  # noqa: E402


def main():
    grid = ref.make_grid(256)
    train = cs.cases_by_split("train")
    out = {}
    for m in (9, 17, 33):
        per = {}
        num = den = 0.0
        for c in train:
            r = ref.get_reference(c, 256, 16, "dense")
            tg = targets.build_targets(c, r, grid, m)
            lc = tg["c"] * tg["lam_V"][None, :]
            a = float(np.sum(tg["M"] * lc)); b = float(np.sum(lc * lc))
            # time-resolved sign: fraction of rows where <M_n, Lam c_n> > 0
            rows = np.sum(tg["M"] * lc, axis=1)
            per[c.id] = {"nu_case": a / b, "num": a, "den": b, "frac_rows_positive": float(np.mean(rows > 0)),
                         "nu_early_t_le_0.2": float(np.sum(rows[: r["i0"] + 1]) / max(np.sum((lc * lc)[: r["i0"] + 1]), 1e-300)),
                         "nu_late_t_gt_0.2": float(np.sum(rows[r["i0"] + 1:]) / max(np.sum((lc * lc)[r["i0"] + 1:]), 1e-300)),
                         "family": c.family}
            num += a; den += b
        vals = np.array([v["nu_case"] for v in per.values()])
        out[f"m{m}"] = {"per_case": per, "pooled_unweighted": num / den,
                        "n_positive": int(np.sum(vals > 0)), "n_negative": int(np.sum(vals < 0)),
                        "min": float(vals.min()), "median": float(np.median(vals)), "max": float(vals.max()),
                        "by_family_median": {f: float(np.median([v["nu_case"] for v in per.values() if v["family"] == f]))
                                             for f in sorted({v["family"] for v in per.values()})}}
        sens = identity.read_json(os.path.join(scc.RESULTS, "fit", f"ql_sensitivity_m{m}.json"))
        if sens is not None:
            out[f"m{m}"]["validation_sensitivity"] = sens
        o = out[f"m{m}"]
        print(f"m={m}: pooled(unweighted) {o['pooled_unweighted']:+.3e}; per-case nu: {o['n_positive']} positive / {o['n_negative']} negative, "
              f"min {o['min']:+.2e} median {o['median']:+.2e} max {o['max']:+.2e}; by family {o['by_family_median']}", flush=True)
    path = os.path.join(scc.RESULTS, "fit", "ql_pertrajectory.json")
    identity.write_json(path, out)
    print("wrote", path)


if __name__ == "__main__":
    main()
