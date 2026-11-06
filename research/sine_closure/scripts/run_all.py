#!/usr/bin/env python3
"""Run the sine-closure pipeline stages s0..s7 in the PLAN section 12 order; stop at the first failure.

    python research/sine_closure/scripts/run_all.py [--from s4] [--only s7] [--dry-run] [--no-resume]

Stages (PLAN.md section 12), in order:
    s0  s0_verify.py
    s1  s1_reproduce_gate5.py
    s2  s2_references.py --split train val dev_regression
    s3  s3_diagnose.py
    s4  s4_fit.py                       (SCC_PHASE=fit; locks models/)
    s2b s2_references.py --split test_interp test_ood
    s5  s5_score.py
    s6  s6_cost.py
    s7  s7_report.py

`--from sN` starts at stage N (`--from s2` = the first s2 call, `--from s2b` = the test-reference call).
`--resume` is forwarded to every script that declares it (default on; `--no-resume` disables). SCC_OUTPUT_ROOT and
SCC_CACHE_ROOT are inherited by every stage (and s1's --out is pointed into SCC_OUTPUT_ROOT when it is set).
NOTE: jobs longer than a minute (s2, s3, s4, s5, s6) belong on the GPU-laptop CPU lane (user rule); this runner executes
locally and is meant for the lane itself, for SCC_OUTPUT_ROOT smoke runs, or for the final s7. Use --dry-run to print the commands.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(STUDY))

STAGES = [
    ("s0", "s0_verify.py", [], {}),
    ("s1", "s1_reproduce_gate5.py", [], {}),
    ("s2", "s2_references.py", ["--split", "train", "val", "dev_regression"], {}),
    ("s3", "s3_diagnose.py", [], {}),
    ("s4", "s4_fit.py", [], {"SCC_PHASE": "fit"}),
    ("s2b", "s2_references.py", ["--split", "test_interp", "test_ood"], {}),
    ("s5", "s5_score.py", [], {}),
    ("s6", "s6_cost.py", [], {}),
    ("s7", "s7_report.py", [], {}),
]


def declares(script, flag):
    try:
        with open(os.path.join(HERE, script), encoding="utf-8") as f:
            return flag in f.read()
    except OSError:
        return False


def build_cmd(stage, script, args, resume):
    cmd = [sys.executable, os.path.join(HERE, script)] + list(args)
    if resume and declares(script, "--resume") and "--resume" not in cmd:
        cmd.append("--resume")
    out_root = os.environ.get("SCC_OUTPUT_ROOT")
    if out_root and stage == "s1" and declares(script, "--out") and "--out" not in cmd:
        cmd += ["--out", os.path.join(out_root, "results", "gate5_reproduction")]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="start", default="s0", help="first stage to run (s0..s7, s2b)")
    ap.add_argument("--only", default=None, help="run exactly this stage")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--no-resume", action="store_true", help="do not forward --resume")
    ap.add_argument("--resume", action="store_true", help="(default) forward --resume to scripts that support it")
    a = ap.parse_args()
    ids = [s[0] for s in STAGES]
    if a.start not in ids or (a.only and a.only not in ids):
        sys.exit(f"unknown stage; choose from {ids}")
    todo = [s for s in STAGES if s[0] == a.only] if a.only else STAGES[ids.index(a.start):]
    print(f"SCC_OUTPUT_ROOT={os.environ.get('SCC_OUTPUT_ROOT', '(study directory)')}", flush=True)
    for stage, script, args, env_extra in todo:
        cmd = build_cmd(stage, script, args, not a.no_resume)
        shown = " ".join(shlex.quote(c) for c in ([os.path.relpath(cmd[1], REPO)] + cmd[2:]))
        prefix = "".join(f"{k}={v} " for k, v in env_extra.items())
        print(f"\n=== {stage}: {prefix}python3 {shown}", flush=True)
        if a.dry_run:
            continue
        if not os.path.isfile(os.path.join(HERE, script)):
            print(f"STOP: {script} does not exist yet.", flush=True)
            print(f"next command: python3 {os.path.relpath(os.path.abspath(__file__), REPO)} --from {stage}", flush=True)
            return 3
        env = dict(os.environ)
        env.update(env_extra)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=REPO, env=env)
        print(f"--- {stage} finished rc={rc} in {time.time() - t0:.1f}s", flush=True)
        if rc != 0:
            print(f"STOP: stage {stage} failed (rc={rc}). Fix it, then resume with:", flush=True)
            print(f"  python3 {os.path.relpath(os.path.abspath(__file__), REPO)} --from {stage}", flush=True)
            return rc
    if not a.dry_run:
        print("\nall requested stages finished", flush=True)
        if a.only is None or a.only == "s7":
            print("next: review results/tables/WARNINGS.md, then write the narrative at the '<!-- coordinator: narrative -->' markers in docs/MEMO.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
