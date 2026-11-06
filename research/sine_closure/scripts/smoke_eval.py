#!/usr/bin/env python3
"""Smoke test for scc.evaluate + s3_diagnose + s5_score on tiny grids (n=64, sub=4; refinement n=127 / 253, sub 8 / 16).

Everything goes to a temporary SCC_OUTPUT_ROOT (default: a fresh temp dir).  Steps: generate the five nested
references for one autonomous (C) and one time-dependent (D) training case; fit tiny closures (LM, LMb, QL*, LR0/LR with
n_h=2) on the C case for m in {9, 17, 33}; save them with scc.modelio.save_spec and write a temporary LOCK (one entry,
m17/LMb, is deliberately left out to exercise the `skipped` path); run s3 --quick, s3 on the D case (all m, P0+P1,
BLOCK time-ordered), s5 --quick, s5 on the D case, and s5 --quick without a LOCK (synthetic path).

Usage: python3 smoke_eval.py [--root DIR]      (all work is seconds-scale; no full-size run is possible here)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = a.root or tempfile.mkdtemp(prefix="scc_smoke_")
    os.makedirs(root, exist_ok=True)
    os.environ["SCC_OUTPUT_ROOT"] = root
    os.environ["SCC_CACHE_ROOT"] = os.path.join(root, "results", "cache_quick")
    sys.path.insert(0, STUDY)
    import scc                                                  # noqa: E402
    from scc import cases as cs, evaluate as ev, modelio, reference as rf  # noqa: E402
    print("temp root:", root, flush=True)
    n, sub = 64, 4
    grid = rf.make_grid(n)
    t0 = time.time()
    c_case = [c for c in cs.cases_by_split("train") if c.family == "C"][0]
    d_case = [c for c in cs.cases_by_split("train") if c.family == "D"][0]
    for case in (c_case, d_case):
        refs = ev.load_levels(case, None, generate=True, n=n, sub=sub)
        print(f"refs {case.id}: {sorted(refs)} ({time.time() - t0:.1f}s)", flush=True)
    ref_c = ev.load_levels(c_case, ["same"], n=n, sub=sub)["n64_s4"]
    selected = {}
    for m, names in ((9, ["QLp", "QLpm", "QLs", "LM", "LMb", "LR0", "LR"]),
                     (17, ["QLp", "QLpm", "QLs", "LM", "LR0", "LR"]),                 # LMb deliberately missing
                     (33, ["LM", "QLp", "QLpm", "QLs"])):
        specs = ev.fit_tiny_specs(c_case, grid, ref_c, m, n_h=2)
        selected[f"m{m}"] = {}
        for name in names:
            sp = specs[name]
            p = modelio.spec_path(m, name, sp.n_h, "")
            modelio.save_spec(sp, p, config={"tiny": True})
            selected[f"m{m}"][name] = {"file": p, "config": {"tiny": True}}
    lock = modelio.write_lock(selected, cs.manifest(grid)["sha256"])
    print("lock:", lock, flush=True)

    env = dict(os.environ)
    py = sys.executable
    runs = [
        ("s3 --quick", [py, os.path.join(HERE, "s3_diagnose.py"), "--quick", "--n", str(n), "--sub", str(sub)]),
        ("s3 D case full", [py, os.path.join(HERE, "s3_diagnose.py"), "--cases", d_case.id, "--n", str(n),
                            "--sub", str(sub), "--skip-family-checks"]),
        ("s5 --quick", [py, os.path.join(HERE, "s5_score.py"), "--quick", "--split", "train", "--n", str(n),
                        "--sub", str(sub)]),
        ("s5 D case full", [py, os.path.join(HERE, "s5_score.py"), "--cases", d_case.id, "--n", str(n),
                            "--sub", str(sub)]),
    ]
    rc = 0
    for name, cmd in runs:
        t1 = time.time()
        print(f"\n===== {name} =====", flush=True)
        r = subprocess.run(cmd, env=env)
        print(f"===== {name}: rc={r.returncode} in {time.time() - t1:.1f}s", flush=True)
        rc = rc or r.returncode
    # synthetic-lock path
    os.remove(lock)
    t1 = time.time()
    print("\n===== s5 --quick without LOCK (synthetic) =====", flush=True)
    r = subprocess.run([py, os.path.join(HERE, "s5_score.py"), "--quick", "--split", "train", "--n", str(n),
                        "--sub", str(sub)], env=env)
    print(f"===== synthetic: rc={r.returncode} in {time.time() - t1:.1f}s", flush=True)
    r2 = subprocess.run([py, os.path.join(HERE, "s5_score.py"), "--cases", c_case.id, "--n", str(n), "--sub",
                         str(sub)], env=env)
    print(f"s5 without LOCK and without --quick must refuse: rc={r2.returncode} (expected 2)", flush=True)
    print(f"\nTOTAL {time.time() - t0:.1f}s; outputs under {root}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
