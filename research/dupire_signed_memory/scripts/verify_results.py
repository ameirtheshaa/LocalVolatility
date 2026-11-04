#!/usr/bin/env python3
"""Independent check: compare a freshly recomputed result JSON with the committed one on every
numeric field (relative tolerance), ignoring timing/identity fields.
    python3 scripts/verify_results.py committed.json fresh.json [--rtol 1e-6]
Exit code 0 when all numeric fields agree, 1 otherwise (differences are listed)."""
from __future__ import annotations

import argparse
import json
import math
import sys

IGNORE = {"wall_seconds", "timestamp_utc", "traceback", "command", "git_head", "git_dirty_tracked_areas",
          "environment", "source_identities", "result_checksums", "path", "stamped",
          "wall_seconds_rollout", "wall_seconds_rollout_median"}


def walk(a, b, path, rtol, atol, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: missing on one side")
                continue
            walk(a[k], b[k], f"{path}/{k}", rtol, atol, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", rtol, atol, diffs)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        fa, fb = float(a), float(b)
        if (math.isnan(fa) and math.isnan(fb)) or (math.isinf(fa) and math.isinf(fb) and (fa > 0) == (fb > 0)):
            return
        if not math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol):
            diffs.append(f"{path}: {fa!r} vs {fb!r}")
    else:
        if a != b:
            diffs.append(f"{path}: {a!r} vs {b!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("committed")
    ap.add_argument("fresh")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-12)
    args = ap.parse_args()
    a = json.load(open(args.committed))
    b = json.load(open(args.fresh))
    diffs = []
    walk(a, b, "", args.rtol, args.atol, diffs)
    if diffs:
        print(f"{len(diffs)} differences:")
        for d in diffs[:50]:
            print("  ", d)
        sys.exit(1)
    print("all numeric fields agree")


if __name__ == "__main__":
    main()
