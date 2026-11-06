#!/usr/bin/env python3
"""Stage s2 -- generate and cache the full semidiscrete references (PLAN.md section 3, RESULT_SCHEMA.md).

Per case and level it calls ``scc.reference.get_reference`` (the only sanctioned way to obtain a reference), which
caches ``.npz`` files under the cache root, and records the result in ``results/references/index.json``
({case_id: {level: {identity_hash, seconds, from_cache, npz_path, n, sub, solver}}}), merged with the existing
content and rewritten atomically after every case (parent process only).  Also writes
results/manifests/CASES.json and ENV.json.  --quick uses tiny grids and writes *_quick files in a quick cache.

When SCC_PHASE=fit the cache layer refuses test splits; the resulting RuntimeError is propagated (not caught).
Heavy at full size: run on the remote CPU box, not on the Mac.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback


def _argval(flag):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


if "--quick" in sys.argv:                       # smoke runs never touch the real cache
    os.environ.setdefault("SCC_CACHE_ROOT", os.path.join(
        os.environ.get("SCC_OUTPUT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "cache_quick"))
try:
    if int(_argval("--parallel") or 1) > 1:     # one BLAS thread per worker process
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(_v, "1")
except ValueError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scc                                                    # noqa: E402
from scc import cases as cs                                   # noqa: E402
from scc import identity                                      # noqa: E402
from scc import reference as ref                              # noqa: E402

STAGE = "s2_references"
# level name -> (n, sub, solver)
LEVELS_FULL = {"n256_s16": (256, 16, "dense"), "n511_s16": (511, 16, "dense"), "n256_s32": (256, 32, "dense"),
               "n1021_s16": (1021, 16, "sparse"), "n256_s64": (256, 64, "sparse")}
# --quick: nested coarse analogues (64 -> 127 -> 253 are nested), sub in {4, 8, 16}; names are kept
LEVELS_QUICK = {"n256_s16": (64, 4, "dense"), "n511_s16": (127, 4, "dense"), "n256_s32": (64, 8, "dense"),
                "n1021_s16": (253, 4, "sparse"), "n256_s64": (64, 16, "sparse")}
BASE_LEVELS = ["n256_s16", "n511_s16", "n256_s32"]
EXTRA_LEVELS = ["n1021_s16", "n256_s64"]


def npz_rel(path):
    try:
        rel = os.path.relpath(path, scc.OUTPUT_ROOT)
        return path if rel.startswith("..") else rel
    except ValueError:
        return path


def level_ident(case, spec):
    n, sub, solver = spec
    return ref.reference_identity(case, n, sub, solver)


def case_levels(case, extra_ids, override):
    if override:
        return list(override)
    lv = list(BASE_LEVELS)
    if case.split == "test_interp" or case.id in extra_ids:
        lv += EXTRA_LEVELS
    return lv


def work(args):
    """Worker (own process in --parallel mode): compute/cache every requested level for one case."""
    cid, levels, table = args
    case = cs.get_case(cid)
    out = {}
    try:
        for lv in levels:
            n, sub, solver = table[lv]
            r = ref.get_reference(case, n, sub, solver)
            ident = r["identity"]
            out[lv] = {"identity_hash": identity.identity_hash(ident), "seconds": float(r["seconds"]),
                       "from_cache": bool(r["from_cache"]), "npz_path": npz_rel(identity.cache_path("reference", ident)),
                       "n": int(n), "sub": int(sub), "solver": solver}
    except RuntimeError as e:                    # leakage guard (SCC_PHASE=fit): propagate, never swallow
        if "SCC_PHASE" in str(e):
            raise
        return cid, out, traceback.format_exc()
    except Exception:
        return cid, out, traceback.format_exc()
    return cid, out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["train", "val", "dev_regression"],
                    choices=["train", "val", "dev_regression", "test_interp", "test_ood"])
    ap.add_argument("--levels", nargs="+", default=None, choices=list(LEVELS_FULL), help="override the per-case level list")
    ap.add_argument("--resume", action="store_true", help="skip (case, level) pairs already in the index whose cache entry loads")
    ap.add_argument("--max-cases", type=int, default=0, help="stop after this many cases that needed work")
    ap.add_argument("--cases", nargs="+", default=None, help="explicit case ids (ignores --split)")
    ap.add_argument("--quick", action="store_true", help="tiny grids (n 64/127/253, sub 4/8/16) and *_quick output files; smoke test only")
    ap.add_argument("--parallel", type=int, default=1, help="worker processes over cases (index is merged in the parent only)")
    args = ap.parse_args()
    table = LEVELS_QUICK if args.quick else LEVELS_FULL
    sfx = "_quick" if args.quick else ""
    t_start = time.time()

    if args.cases:
        todo = [cs.get_case(c) for c in args.cases]
    else:
        todo = [c for sp in args.split for c in cs.cases_by_split(sp)]
    # leakage guard: raises RuntimeError under SCC_PHASE=fit for test splits (propagated on purpose)
    for c in todo:
        identity.assert_phase_allows(c.split)

    extra_ids = set()
    for fam in ("C", "S", "D", "M"):
        extra_ids.update(c.id for c in [x for x in cs.cases_by_split("train") if x.family == fam][:2])

    out_ref = os.path.join(scc.RESULTS, "references")
    index_path = os.path.join(out_ref, f"index{sfx}.json")
    man_dir = os.path.join(scc.RESULTS, "manifests")
    grid_for_manifest = ref.make_grid(64 if args.quick else 256)
    cases_path = os.path.join(man_dir, f"CASES{sfx}.json")
    os.makedirs(man_dir, exist_ok=True)
    cs.write_manifest(cases_path, grid=grid_for_manifest)
    identity.write_json(os.path.join(man_dir, f"ENV{sfx}.json"), {
        "env": identity.env_identity(), "git_head": identity.git_head(), "source_hashes": identity.source_hashes(),
        "cases_manifest_sha256": cs.manifest()["sha256"], "cases_json_sha256": identity.sha256_file(cases_path),
        "stage": STAGE, "quick": bool(args.quick), "scc_phase": os.environ.get("SCC_PHASE")})

    index = identity.read_json(index_path) or {}
    plan = []                                    # (case, levels still needed)
    for c in todo:
        lv_all = case_levels(c, extra_ids, args.levels)
        need = []
        for lv in lv_all:
            if args.resume:
                e = index.get(c.id, {}).get(lv)
                ident = level_ident(c, table[lv])
                if (e and e.get("identity_hash") == identity.identity_hash(ident)
                        and identity.cache_load("reference", ident) is not None):
                    continue
            need.append(lv)
        if need:
            plan.append((c, need))
        else:
            print(f"[{c.id}] up to date ({len(lv_all)} levels), skipped", flush=True)
    if args.max_cases:
        plan = plan[: args.max_cases]
    manifest = identity.RunManifest("references_quick" if args.quick else "references", [c.id for c, _ in plan])
    if manifest.previous_interrupted:
        print(f"previous run interrupted: {manifest.previous_interrupted}", flush=True)
    print(f"s2_references: {len(plan)} cases to process, parallel={args.parallel}, quick={args.quick}, index={index_path}",
          flush=True)

    failed = []
    pending_ids = [c.id for c, _ in plan]

    def absorb(cid, res, tb):
        if res:
            index.setdefault(cid, {}).update(res)
            identity.write_json(index_path, index)
        for lv, e in res.items():
            print(f"[{cid}] {lv:10s} n={e['n']:<5d} sub={e['sub']:<3d} {e['solver']:6s} {e['seconds']:8.2f}s"
                  f"{' (cache)' if e['from_cache'] else ''}", flush=True)
        pending_ids.remove(cid)
        if tb:
            failed.append(cid)
            manifest.failed(cid)
            print(f"[{cid}] FAILED\n{tb}", flush=True)
        else:
            manifest.done(cid)
        manifest.data["in_progress"] = pending_ids[: max(1, args.parallel)]
        manifest._write()

    jobs = [(c.id, need, table) for c, need in plan]
    if args.parallel <= 1:
        for j in jobs:
            manifest.start(j[0])
            absorb(*work(j))
    elif jobs:
        manifest.data["in_progress"] = pending_ids[: args.parallel]
        manifest._write()
        with mp.get_context("spawn").Pool(processes=args.parallel) as pool:
            for r in pool.imap_unordered(work, jobs):
                absorb(*r)
    manifest.finish()
    print(f"s2_references: {len(plan) - len(failed)} cases done, {len(failed)} failed, wall {time.time() - t_start:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
