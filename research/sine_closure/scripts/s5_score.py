#!/usr/bin/env python3
"""s5 -- score the locked models (models/LOCK.json) on held-out cases; writes results/score/<case_id>.json,
results/uncertainty/<case_id>.json and results/score/summary.json.

Rows per case (RESULT_SCHEMA keys): RRR at m in {9,13,17,25,33,65}; PROJ at every m; TF at {9,17,33};
BLOCK identity at {9,17}; closures QLp, QLpm, QLs, LM, LMb, LR0, LR at m in {9,17} and the secondary
LM, QLp, QLpm, QLs at m=33 (a (m, closure) absent from the lock is skipped and listed under `skipped`);
FULL256 (same-grid full solve vs the refined truth) and FULL511 (vs n1021 when present, else null).
Protocols: P0 payoff start; P1 warm start at t~0=0.2 with hidden protocols W0 / WI / WU for LR0, LR.

REFUSES to run when models/LOCK.json is missing (except --quick, which fits tiny models in memory and says so).
Needs all reference levels n256_s16, n511_s16, n256_s32 for every scored case (n1021_s16 / n256_s64 optional).
Heavy at full size: run on the remote CPU box, not on the Mac.

Usage: s5_score.py [--split test_interp test_ood] [--cases ID ...] [--max-cases N] [--resume] [--quick]
                   [--n 256] [--sub 16] [--generate]
"""
from __future__ import annotations

import argparse
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


_N_ARG = int(_argval("--n") or 256)
if _N_ARG != 256:
    os.environ.setdefault("SCC_CACHE_ROOT", os.path.join(
        os.environ.get("SCC_OUTPUT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "cache_quick"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402

import scc                                                    # noqa: E402
from scc import cases as cs                                   # noqa: E402
from scc import evaluate as ev                                # noqa: E402
from scc import identity                                      # noqa: E402
from scc import modelio                                       # noqa: E402
from scc import reference as rf                               # noqa: E402
from scc.closures import ModelSpec                            # noqa: E402

STAGE = "score"
SCRIPT = os.path.abspath(__file__)
M_RRR = [9, 13, 17, 25, 33, 65]
M_TF = (9, 17, 33)
M_BLOCK = (9, 17)
M_PRIMARY = (9, 17)
M_SECONDARY = 33
CL_PRIMARY = ["QLp", "QLpm", "QLs", "LM", "LMb", "LR0", "LR"]
CL_SECONDARY = ["LM", "QLp", "QLpm", "QLs"]
ALL_SPLITS = ["train", "val", "dev_regression", "test_interp", "test_ood"]


class LockSpecs:
    """Specs from models/LOCK.json (hash-verified by modelio.load_locked)."""
    def __init__(self, lock):
        self.lock = lock
        self.kind = "lock"

    def get(self, m, closure):
        return modelio.load_locked(self.lock, m, closure)


class MemorySpecs:
    """Quick mode without a LOCK: tiny models fitted in memory on the case's own same-grid reference."""
    kind = "synthetic"

    def __init__(self, specs_by_m):
        self.d = specs_by_m

    def get(self, m, closure):
        return self.d[m][closure]


def out_dirs(small):
    sfx = "_quick" if small else ""
    return os.path.join(scc.RESULTS, STAGE + sfx), os.path.join(scc.RESULTS, "uncertainty" + sfx)


def hidden_list(spec, protocol):
    if spec.closure != "LR":
        return [None]
    if protocol == "P0":
        return ["W0"]
    hs = ["W0"]
    if spec.params.get("K_init") is not None:
        hs.append("WI")
    hs.append("WU")
    return hs


def score_case(case, args, provider, small):
    t0 = time.time()
    n, sub = args.n, args.sub
    grid = rf.make_grid(n)
    roles = ["same", "refined", "step"] if args.quick else None
    refs = ev.load_levels(case, roles, generate=(args.generate or args.quick), n=n, sub=sub)
    bundle = ev.make_bundle(case, grid, refs, sub)
    protocols = ["P0"] if args.quick else ["P0", "P1"]
    m_rrr = [9] if args.quick else [m for m in M_RRR if m <= n - 3]
    unc = ev.uncertainty(bundle, m_rrr)
    models, skipped = {}, []

    if args.quick and provider.kind == "synthetic":
        provider.d[9] = ev.fit_tiny_specs(case, grid, bundle.refs["same"], 9)

    for protocol in protocols:
        for m in m_rrr:
            row, _ = ev.run_rollout_row(bundle, ModelSpec(name="RRR", m=m, closure="RRR"), protocol, "W0", True)
            models[ev.model_key("RRR", m, 0, protocol)] = row
            models[ev.model_key("PROJ", m, 0, protocol)] = ev.run_oracle_row(bundle, "PROJ", m, protocol)
            if m in M_TF:
                models[ev.model_key("TF", m, 0, protocol)] = ev.run_oracle_row(bundle, "TF", m, protocol)
            if m in M_BLOCK:
                models[ev.model_key("BLOCK", m, 0, protocol)] = ev.run_block_row(bundle, m, protocol)
    plan = [(m, CL_PRIMARY) for m in M_PRIMARY if m in m_rrr] + \
           ([(M_SECONDARY, CL_SECONDARY)] if M_SECONDARY in m_rrr else [])
    for m, names in plan:
        for name in names:
            try:
                spec = provider.get(m, name)
            except (KeyError, FileNotFoundError) as e:
                skipped.append({"m": m, "closure": name, "reason": f"not in lock ({type(e).__name__}: {e})"})
                continue
            for protocol in protocols:
                for hid in hidden_list(spec, protocol):
                    key = ev.model_key(name, m, spec.n_h, protocol, hid if protocol == "P1" else None)
                    try:
                        row, _ = ev.run_rollout_row(bundle, spec, protocol, hid or "W0", True)
                        row["hidden"] = hid
                        models[key] = row
                    except Exception:
                        models[key] = {"metrics": None, "unstable": True, "failed": True,
                                       "traceback": traceback.format_exc()}
    # full solves
    models["FULL256__P0"] = {"metrics": ev.score_level_rows(bundle, "same", "refined"), "unstable": False,
                             "n_last": bundle.N, "total_state": int(n), "param_count": None,
                             "online_seconds": None, "n_expm": None}
    if "refined2" in bundle.rows:
        models["FULL511__P0"] = {"metrics": ev.score_level_rows(bundle, "refined", "refined2"), "unstable": False,
                                 "n_last": bundle.N, "total_state": int(2 * n - 1), "param_count": None,
                                 "online_seconds": None, "n_expm": None}
    else:
        models["FULL511__P0"] = {"metrics": None, "reason": "n1021_s16 reference not available",
                                 "unstable": False, "total_state": int(2 * n - 1)}
    return bundle, unc, models, skipped, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["test_interp", "test_ood"], choices=ALL_SPLITS)
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true", help="1 case (first autonomous family-C), m=9, P0 only")
    ap.add_argument("--n", type=int, default=256, help="model grid size (smoke tests only)")
    ap.add_argument("--sub", type=int, default=16, help="fine substeps per output interval (smoke only)")
    ap.add_argument("--generate", action="store_true", help="compute missing reference levels instead of failing")
    args = ap.parse_args()
    small = args.quick or args.n != 256
    out_dir, unc_dir = out_dirs(small)

    lock_path = os.path.join(scc.MODELS, "LOCK.json")
    if os.path.isfile(lock_path):
        lock = modelio.read_lock()
        provider = LockSpecs(lock)
        lock_sha = identity.sha256_file(lock_path)
    elif args.quick:
        print("WARNING: models/LOCK.json missing; --quick fits tiny models in memory (synthetic LOCK).", flush=True)
        provider, lock_sha = MemorySpecs({}), "synthetic"
    else:
        print("REFUSING to score: models/LOCK.json is missing (run s4_fit.py first).", file=sys.stderr)
        return 2

    if args.cases:
        todo = [cs.get_case(c) for c in args.cases]
    else:
        todo = [c for sp in args.split for c in cs.cases_by_split(sp)]
    if args.quick:
        cand = [c for c in todo if c.family == "C" and not c.time_dependent] or todo
        todo = cand[:1]
    if args.max_cases:
        todo = todo[:args.max_cases]
    man = identity.RunManifest(STAGE + ("_quick" if small else ""), [c.id for c in todo])
    if man.previous_interrupted:
        print("previous interrupted run:", man.previous_interrupted, flush=True)
    t_all = time.time()
    done, failed, skipped_cases, skipped_models = [], [], [], {}
    for case in todo:
        path = os.path.join(out_dir, case.id + ".json")
        upath = os.path.join(unc_dir, case.id + ".json")
        sig = ev.levels_signature(case, args.n, args.sub)
        ident = ev.result_ident("score", case, args.n, args.sub, SCRIPT, quick=bool(args.quick), lock=lock_sha,
                                levels=sig)
        uident = dict(ident, kind="uncertainty")
        if ev.prepare_target(path, ident, args.resume) == "skip" and \
                identity.result_status(upath, uident)[0] == "done":
            print(f"[{case.id}] skip (done)", flush=True)
            skipped_cases.append(case.id)
            continue
        ev.prepare_target(upath, uident, False)
        man.start(case.id)
        t0 = time.time()
        try:
            bundle, unc, models, skipped, wall = score_case(case, args, provider, small)
            head = dict(case=case.id, case_id=case.id, family=case.family, split=case.split, params=case.params,
                        n=args.n, sub=args.sub, levels_present=bundle.levels_present, lock_sha256=lock_sha,
                        lock_kind=provider.kind)
            ev.write_result(upath, uident, "uncertainty", wall, uncertainty=unc, **head)
            ev.write_result(path, ident, STAGE, time.time() - t0, uncertainty=unc, models=models, skipped=skipped,
                            **head)
            man.done(case.id)
            done.append(case.id)
            skipped_models[case.id] = skipped
            print(f"[{case.id}] done: {len(models)} rows, {len(skipped)} skipped, {time.time() - t0:.1f}s",
                  flush=True)
        except Exception:
            tb = traceback.format_exc()
            ev.write_result(path, ident, STAGE, time.time() - t0, status="failed", traceback_text=tb,
                            case_id=case.id, case=case.id)
            man.failed(case.id)
            failed.append(case.id)
            print(f"[{case.id}] FAILED\n{tb}", flush=True)
    sident = identity.make_identity(kind="score_summary", n=args.n, sub=args.sub, cases=sorted(done + failed),
                                    lock=lock_sha, evaluate=ev.evaluate_identity_hash(), script=ev.sha_of_file(SCRIPT))
    ev.write_result(os.path.join(out_dir, "summary.json"), sident, STAGE, time.time() - t_all, done=done,
                    failed=failed, skipped_cases=skipped_cases, skipped_models=skipped_models, lock_sha256=lock_sha,
                    lock_kind=provider.kind)
    man.finish()
    print(f"done={len(done)} failed={len(failed)} skipped={len(skipped_cases)} wall={time.time() - t_all:.1f}s",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
