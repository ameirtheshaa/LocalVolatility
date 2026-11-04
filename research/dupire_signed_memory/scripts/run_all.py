#!/usr/bin/env python3
"""Run (or resume) the whole study in order: gate1 → gate2 → gate3 → gate4. Each gate is
idempotent; the first failing gate stops the run and NEXT_ACTIONS.md names the exact command."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # repository root
GATES = [
    ("gate1", ["research/dupire_signed_memory/scripts/gate1_contract.py"]),
    ("gate2", ["research/dupire_signed_memory/scripts/gate2_reference.py", "--resume"]),
    ("gate3", ["research/dupire_signed_memory/scripts/gate3_compare.py", "--resume"]),
    ("gate4", ["research/dupire_signed_memory/scripts/gate4_evaluate.py"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="(default behaviour; kept for the documented command)")
    ap.add_argument("--from", dest="start", default="gate1", choices=[g for g, _ in GATES])
    args = ap.parse_args()
    started = False
    for name, argv in GATES:
        if name == args.start:
            started = True
        if not started:
            continue
        print(f"=== {name}: {' '.join(argv)}", flush=True)
        rc = subprocess.call([sys.executable, *argv], cwd=ROOT)
        if rc != 0:
            print(f"{name} failed with exit code {rc}; see research/dupire_signed_memory/NEXT_ACTIONS.md", flush=True)
            sys.exit(rc)
    print("all gates complete")


if __name__ == "__main__":
    main()
