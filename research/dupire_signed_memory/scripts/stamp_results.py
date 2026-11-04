#!/usr/bin/env python3
"""Attribute legacy (schema-1) result files to the code that produced them, without recomputing.

A legacy result carries only the case hash. This script verifies that every numerical module at
HEAD is byte-identical to the one at the commit the results were produced with (default
7d7753c, the first release) and, if so, writes the schema-2 identity into each legacy result so
`result_status()` can reuse it. If any numerical module differs, nothing is stamped and the
script exits 1: the results must then be recomputed (or stamped against the right commit).
Stamping only adds fields; the original file content is otherwise untouched and remains in Git.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsm import cases, checkpoint  # noqa: E402
from dsm.repo import REPO_ROOT, STUDY_ROOT  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(STUDY_ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def modules_match_commit(commit: str):
    diffs = []
    for rel in checkpoint.NUMERICAL_MODULES:
        repo_rel = os.path.relpath(os.path.join(STUDY_ROOT, rel), REPO_ROOT)
        try:
            old = subprocess.check_output(["git", "-C", REPO_ROOT, "show", f"{commit}:{repo_rel}"])
        except subprocess.CalledProcessError:
            diffs.append(f"{rel}: not found at {commit}")
            continue
        with open(os.path.join(STUDY_ROOT, rel), "rb") as f:
            new = f.read()
        if hashlib.sha256(old).hexdigest() != hashlib.sha256(new).hexdigest():
            diffs.append(f"{rel}: differs from {commit}")
    return diffs


def _stamp(d, ident, commit, path, dry):
    d["identity"], d["identity_hash"] = ident, checkpoint.identity_hash(ident)
    d["stamped"] = {"attributed_commit": commit, "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not dry:
        checkpoint.write_json(path, d)


def stamp_gate2(dry, commit):
    g2 = _load("gate2_reference")
    n = 0
    for p in sorted(glob.glob(os.path.join(checkpoint.RESULTS, "gate2", "*__*.json"))):
        d = checkpoint.read_json(p)
        if not d or d.get("status") != "done" or "identity_hash" in d:
            continue
        case = cases.get_case(d["case"])
        if case.config_hash() != d.get("config_hash"):
            print(f"skip {p}: case hash mismatch")
            continue
        _stamp(d, g2.case_identity(case, d["variant"], d["n_k"], d["sub"], "refinement" in d, cases.KEEPS), commit, p, dry)
        n += 1
    return n


def stamp_gate2b(dry, commit, sub=16):
    g = _load("gate2b_basis_diagnostic")
    n = 0
    for p in sorted(glob.glob(os.path.join(checkpoint.RESULTS, "gate2", "basis", "*__B.json"))):
        d = checkpoint.read_json(p)
        if not d or d.get("status") != "done" or "identity_hash" in d:
            continue
        case = cases.get_case(d["case"])
        if case.config_hash() != d.get("config_hash"):
            print(f"skip {p}: case hash mismatch")
            continue
        _stamp(d, g.case_identity(case, d["n_k"], sub, cases.KEEPS), commit, p, dry)
        n += 1
    return n


def stamp_gate3(dry, commit):
    g3 = _load("gate3_compare")
    n = 0
    for p in sorted(glob.glob(os.path.join(checkpoint.RESULTS, "gate3", "blocks", "*.json"))):
        d = checkpoint.read_json(p)
        if not d or d.get("status") != "done" or "identity_hash" in d:
            continue
        refine = bool(d.get("refinement_band", {}).get("per_case"))
        _stamp(d, g3.block_identity(d["variant"], d["keep"], d["n_k"], d["sub"], refine), commit, p, dry)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribute-to", default="7d7753c", help="commit whose numerical modules produced the legacy results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    diffs = modules_match_commit(args.attribute_to)
    if diffs:
        print("refusing to stamp: numerical modules differ from the attributed commit:")
        for d in diffs:
            print("  ", d)
        sys.exit(1)
    n2, n2b, n3 = stamp_gate2(args.dry_run, args.attribute_to), stamp_gate2b(args.dry_run, args.attribute_to), stamp_gate3(args.dry_run, args.attribute_to)
    print(f"{'would stamp' if args.dry_run else 'stamped'} gate2={n2} gate2b={n2b} gate3={n3} results, attributed to {args.attribute_to}; code identity {checkpoint.numerical_code_identity()}")


if __name__ == "__main__":
    main()
