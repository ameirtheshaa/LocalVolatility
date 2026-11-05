#!/usr/bin/env python3
"""s1 — reproduce Gate 5 of the earlier study into a separate output location and compare.

Runs the *unchanged* `research/dupire_signed_memory/scripts/gate5_sine_rom.py` with
`DSM_OUTPUT_ROOT` pointing at `research/sine_closure/results/gate5_reproduction/` (so the
committed `research/dupire_signed_memory/results/gate5/*.json` are never touched), then compares
every non-timing metric with the committed values under the recorded tolerances of PLAN.md §11
(`rel 1e-5 + abs 1e-9`; fraction-type metrics `abs 0.01`). Writes `comparison.json` next to the
reproduced results. Exit code 0 when every compared metric passes, 2 otherwise.

Usage: python3 s1_reproduce_gate5.py [--sub 16] [--compare-only] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)                                   # research/sine_closure
DSM_STUDY = os.path.join(os.path.dirname(STUDY), "dupire_signed_memory")
COMMITTED = os.path.join(DSM_STUDY, "results", "gate5")
CASE_IDS = ["C_sig0.30", "S_seed9", "F_fresh_base0.28_t0.2_x0.05"]
SKIP_KEYS = ("wall_seconds", "wall_seconds_rollout", "wall_seconds_rollout_median")
REL, ABS, FRAC_ABS = 1e-5, 1e-9, 0.01


def _walk(d, prefix=""):
    for k, v in d.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _walk(v, p + ".")
        else:
            yield p, v


def compare(out_dir):
    rep_dir = os.path.join(out_dir, "results", "gate5")
    summary = {"tolerances": {"rel": REL, "abs": ABS, "frac_abs": FRAC_ABS}, "cases": {}, "n_compared": 0,
               "n_failed": 0, "max_rel_dev": 0.0, "max_rel_dev_key": None}
    for cid in CASE_IDS:
        a = json.load(open(os.path.join(COMMITTED, cid + ".json")))
        pb = os.path.join(rep_dir, cid + ".json")
        if not os.path.isfile(pb):
            summary["cases"][cid] = {"status": "missing"}
            summary["n_failed"] += 1
            continue
        b = json.load(open(pb))
        rows, failed = {}, []
        fa = dict(_walk({"models": a["models"], "uncertainty": a["uncertainty"]}))
        fb = dict(_walk({"models": b["models"], "uncertainty": b["uncertainty"]}))
        for key, va in fa.items():
            if any(key.endswith(s) for s in SKIP_KEYS) or not isinstance(va, (int, float)) or isinstance(va, bool):
                continue
            vb = fb.get(key)
            if vb is None or not isinstance(vb, (int, float)):
                failed.append(key)
                rows[key] = {"committed": va, "reproduced": vb, "pass": False}
                continue
            va, vb = float(va), float(vb)
            if math.isnan(va) and math.isnan(vb):
                ok, dev = True, 0.0
            else:
                diff = abs(va - vb)
                if "frac" in key:
                    ok, dev = diff <= FRAC_ABS, diff
                else:
                    tol = ABS + REL * max(abs(va), abs(vb))
                    ok = diff <= tol
                    dev = diff / max(abs(va), abs(vb), 1e-300)
                if dev > summary["max_rel_dev"] and "frac" not in key:
                    summary["max_rel_dev"], summary["max_rel_dev_key"] = dev, f"{cid}:{key}"
            rows[key] = {"committed": va, "reproduced": vb, "pass": bool(ok)}
            summary["n_compared"] += 1
            if not ok:
                failed.append(key)
        summary["cases"][cid] = {"status": "done" if not failed else "deviations", "failed_keys": failed,
                                 "n_failed": len(failed), "metrics": rows,
                                 "reproduced_wall_seconds": b.get("wall_seconds"),
                                 "committed_identity": a.get("identity"), "reproduced_identity": b.get("identity")}
        summary["n_failed"] += len(failed)
    summary["pass"] = summary["n_failed"] == 0
    path = os.path.join(rep_dir, "comparison.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    return summary, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=16)
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(STUDY, "results", "gate5_reproduction"))
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    os.environ["DSM_OUTPUT_ROOT"] = out                     # before any dsm import
    sys.path.insert(0, DSM_STUDY)
    sys.path.insert(0, os.path.join(DSM_STUDY, "scripts"))
    import numpy, scipy  # noqa: E402
    print(f"python {sys.version.split()[0]} numpy {numpy.__version__} scipy {scipy.__version__}", flush=True)
    print(f"DSM_OUTPUT_ROOT={out}", flush=True)
    if not args.compare_only:
        import gate5_sine_rom as g5  # noqa: E402  (unchanged original script)
        assert os.path.abspath(g5.OUT).startswith(out), g5.OUT
        t0 = time.time()
        sys.argv = ["gate5_sine_rom.py", "--sub", str(args.sub)]
        try:
            g5.main()
        except SystemExit as e:
            print(f"gate5 exit code {e.code}", flush=True)
        print(f"gate5 reproduction finished in {time.time() - t0:.0f}s", flush=True)
    summary, path = compare(out)
    print(f"compared {summary['n_compared']} metrics, failed {summary['n_failed']}, "
          f"max rel dev {summary['max_rel_dev']:.2e} at {summary['max_rel_dev_key']}; wrote {path}", flush=True)
    for cid, c in summary["cases"].items():
        print(f"  {cid}: {c['status']} {c.get('failed_keys', [])[:10]}", flush=True)
    sys.exit(0 if summary["pass"] else 2)


if __name__ == "__main__":
    main()
