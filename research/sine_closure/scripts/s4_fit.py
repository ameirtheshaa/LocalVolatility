#!/usr/bin/env python3
"""Stage s4 -- fitting, validation-only selection and model locking (PLAN.md sections 5-7, 10).

Runs with ``SCC_PHASE=fit``: the reference cache then refuses every ``test_*`` split (leakage guard).
Fits QL (three variants), LM, LM+b, LR0 and LR (with the warm-start initialiser K_init) at m in {9, 17, 33}
on the *train* split only, selects hyperparameters on the *val* split only (free rollouts, protocols P0 and
P1-W0, geometric mean of the resolved interior RMS ``e_dyn``), saves every candidate under ``models/m<m>/`` and
writes ``models/LOCK.json``.

Outputs (under OUTPUT_ROOT = SCC_OUTPUT_ROOT or research/sine_closure):
  models/m<m>/<closure>[_h<n_h>]_<config>.npz + .json  (every candidate, not only the selected ones)
  models/LOCK.json
  results/fit/validation_m<m>.json, results/fit/ql_sensitivity_m<m>.json,
  results/fit/selection.json, results/fit/offline_costs.json

``--quick``: n=64, sub=4, first 2 train + 1 val case, n_h in {2}, bank N, two ridge values; all outputs go to a
temporary SCC_OUTPUT_ROOT (created when the variable is unset), so it never touches the real models/ or results/.
Full size is heavy: run it on the remote CPU box, not on the Mac.
"""
from __future__ import annotations

import os

os.environ["SCC_PHASE"] = "fit"      # leakage guard: must precede any scc import

import argparse                                                # noqa: E402
import sys                                                     # noqa: E402
import tempfile                                                # noqa: E402
import time                                                    # noqa: E402
import traceback                                               # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if "--quick" in sys.argv:                                      # smoke runs never touch the real outputs/cache
    if not os.environ.get("SCC_OUTPUT_ROOT"):
        os.environ["SCC_OUTPUT_ROOT"] = tempfile.mkdtemp(prefix="scc_quick_")
    os.environ.setdefault("SCC_CACHE_ROOT", os.path.join(os.environ["SCC_OUTPUT_ROOT"], "results", "cache_quick"))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np                                             # noqa: E402

import scc                                                     # noqa: E402
from scc import cases as cs                                    # noqa: E402
from scc import closures as cl                                 # noqa: E402
from scc import conventions as cv                              # noqa: E402
from scc import identity                                       # noqa: E402
from scc import metrics as mt                                  # noqa: E402
from scc import modelio                                        # noqa: E402
from scc import reference as ref                               # noqa: E402
from scc import targets as tgm                                 # noqa: E402

STAGE = "s4_fit"
LAMS_FULL = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
LAMS_QUICK = [1e-6, 1e-4]
NHS_FULL, BANKS_FULL = [8, 16], ["N", "W"]
NHS_QUICK, BANKS_QUICK = [2], ["N"]
SENS_MULTS = [0.0, 0.25, 0.5, 1.0, 2.0, -1.0]       # multiples of nu_unclipped (PLAN section 5: {0,1/4,1/2,1,2} and -nu*)
PROTOCOLS = ("P0", "P1")
FAMILIES = ("LM", "LMb", "LR0", "LR")
LOCK_NAMES = ("QLp", "QLpm", "QLs", "LM", "LMb", "LR0", "LR")


# ----------------------------------------------------------------------------- small helpers
def lam_tag(lam: float) -> str:
    return f"lam{lam:g}"


def now() -> float:
    return time.perf_counter()


def fin(x) -> bool:
    return x is not None and np.isfinite(x)


def geomean(vals):
    vals = [v for v in vals]
    if not vals or any((not fin(v)) or v <= 0 for v in vals):
        return None
    return float(np.exp(np.mean(np.log(vals))))


def representation_limited(m: int):
    """(flag or None, source) from results/diagnose/summary.json (written by s3); layout-tolerant."""
    p = os.path.join(scc.RESULTS, "diagnose", "summary.json")
    d = identity.read_json(p) if os.path.isfile(p) else None
    if not isinstance(d, dict):
        return None, "summary.json absent"
    stack = [d]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for key in (f"m{m}", str(m), f"m_{m}"):
                v = cur.get(key)
                if isinstance(v, dict) and "representation_limited" in v:
                    return bool(v["representation_limited"]), p
            stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            for e in cur:
                if isinstance(e, dict) and e.get("m") == m and "representation_limited" in e:
                    return bool(e["representation_limited"]), p
                stack.append(e)
    return None, "no entry for this m in summary.json"


# ----------------------------------------------------------------------------- validation machinery
class ValCase:
    """Everything needed for the free rollouts of one validation case at one m."""

    def __init__(self, case, tg, fg, grid, ctx):
        self.id = case.id
        self.td = bool(case.time_dependent)
        self.t = tg["t"]
        self.i0 = int(fg.i0)
        self.out_idx = np.asarray(fg.out_idx)
        self.c_ref = tg["c"]
        self.lam = tg["lam_V"]
        self.V_int = tg["V_int"]
        self.V_full = cv.dst_basis(grid.n_k, self.V_int.shape[1])
        self.ctx = ctx
        sys_ = tg["sys"]
        if self.td:
            tm = 0.5 * (self.t[:-1] + self.t[1:])
            self.A = np.array([cv.A_structured(sys_, float(x), self.V_int, self.lam) for x in tm])
            self.Am = lambda i: self.A[i]
        else:
            A0 = cv.A_structured(sys_, 0.0, self.V_int, self.lam)
            self.Am = lambda i: A0
        self.c0_P0 = cv.payoff_state(grid)[2][1:-1] @ self.V_int
        self.norm_all = float(np.max(np.linalg.norm(self.c_ref, axis=1)))
        self.norm_win = float(np.max(np.linalg.norm(self.c_ref[self.i0:], axis=1)))
        self.c_ref_out = self.c_ref[self.out_idx]

    def e_dyn(self, spec, proto: str):
        i0 = self.i0
        if proto == "P0":
            t, c0, Am, rows, nm = self.t, self.c0_P0, self.Am, self.out_idx, self.norm_all
        else:
            t, c0, rows, nm = self.t[i0:], self.c_ref[i0], self.out_idx - i0, self.norm_win
            Am = (lambda i, f=self.Am, o=i0: f(i + o))
        roll = cl.rollout(spec, Am, t, c0, self.lam, ref_norm_max=nm, time_dependent=self.td)
        c_rows = roll["c"][rows]
        bad = bool(roll["unstable"]) or not np.all(np.isfinite(c_rows))
        if bad:
            return None, True
        e = mt.resolved_error(c_rows, self.c_ref_out, self.V_full, self.ctx.interior)["resolved_interior_rms"]
        if not np.isfinite(e):
            return None, True
        return float(e), False


def evaluate(spec, vcs):
    """{case: {P0: {e_dyn, unstable}, P1: {...}}}, geometric mean (None if any instability), n_unstable."""
    out, vals, nbad = {}, [], 0
    for vc in vcs:
        out[vc.id] = {}
        for pr in PROTOCOLS:
            e, bad = vc.e_dyn(spec, pr)
            out[vc.id][pr] = {"e_dyn": e, "unstable": bad}
            vals.append(e if not bad else None)
            nbad += int(bad)
    gm = None if nbad else geomean(vals)
    return out, gm, nbad


def pick_best(cands):
    """Minimum geometric mean; ties (rel 1e-9) -> smaller n_h, then larger lambda."""
    ok = [c for c in cands if c["gm"] is not None]
    if not ok:
        return None
    best = min(c["gm"] for c in ok)
    tied = [c for c in ok if c["gm"] <= best * (1 + 1e-9)]
    return min(tied, key=lambda c: (c["config"].get("n_h", 0), -c["config"].get("lam_rel", 0.0)))


# ----------------------------------------------------------------------------- in-sample residuals
def insample_residuals(spec, tgs, qcache):
    """Relative RMS of M - M_hat (teacher-forced inputs) per training case; M_hat from the fitted closure."""
    out = {}
    for tg in tgs:
        c, M = tg["c"], tg["M"]
        if spec.closure == "QL":
            Mh = float(spec.params["nu"]) * c * tg["lam_V"][None, :]
        elif spec.closure == "LM":
            Mh = c @ np.asarray(spec.params["H"]).T
            if spec.params.get("b") is not None:
                Mh = Mh + np.asarray(spec.params["b"])[None, :]
        else:
            alphas = np.asarray(spec.meta["alphas"], float)
            key = (tg["case_id"], tuple(np.round(alphas, 12)))
            if key not in qcache:
                qcache[key] = cl.filtered_states(c, tg["t"], alphas)
            h = np.einsum("ki,nki->nk", np.asarray(spec.params["G"]), qcache[key])
            Mh = c @ np.asarray(spec.params["H"]).T + h @ np.asarray(spec.params["R"]).T
        r = float(np.sqrt(np.mean(np.sum((M - Mh) ** 2, axis=1) / M.shape[1])))
        out[tg["case_id"]] = {"resid_rms": r, "rms_M": float(tg["rms_M"]), "resid_rel": r / max(tg["rms_M"], 1e-300)}
    return out


# ----------------------------------------------------------------------------- candidate store / resume
def try_resume(path, extra_key, resume):
    if not resume or not os.path.isfile(path) or not os.path.isfile(path[:-4] + ".json"):
        return None
    side = identity.read_json(path[:-4] + ".json")
    if not side or side.get("code_identity") != identity.code_identity():
        return None
    if side.get("extra", {}).get("fit_key") != extra_key:
        return None
    try:
        spec = modelio.load_spec(path)
    except Exception:
        return None
    return spec, side


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="n=64, sub=4, 2 train + 1 val, tiny budgets")
    ap.add_argument("--resume", action="store_true", help="reuse model files with matching code identity")
    ap.add_argument("--max-cases", type=int, default=None, help="deliberate early stop: first N train and val cases")
    ap.add_argument("--cases", nargs="+", default=None, help="restrict to these case ids (train/val)")
    ap.add_argument("--m", nargs="+", type=int, default=[9, 17, 33])
    args = ap.parse_args(argv)
    quick = args.quick
    t_wall = now()

    n, sub = (64, 4) if quick else (256, 16)
    LAMS = LAMS_QUICK if quick else LAMS_FULL
    NHS = NHS_QUICK if quick else NHS_FULL
    BANKS = BANKS_QUICK if quick else BANKS_FULL
    grid = ref.make_grid(n)
    fg = ref.fine_grid(grid, sub)
    train, val = cs.cases_by_split("train"), cs.cases_by_split("val")
    if quick and not args.cases:
        train, val = train[:2], val[:1]
    if args.cases:
        train = [c for c in train if c.id in args.cases]
        val = [c for c in val if c.id in args.cases]
    if args.max_cases:
        train, val = train[:args.max_cases], val[:args.max_cases]
    assert train and val, "need at least one train and one val case"
    assert all(c.split == "train" for c in train) and all(c.split == "val" for c in val)
    print(f"[s4] n={n} sub={sub} train={len(train)} val={len(val)} out={scc.OUTPUT_ROOT} phase={os.environ['SCC_PHASE']}")

    # -- references (n256_s16 dense; cached by s2, generated here when absent -- same identity)
    refs, ref_seconds = {}, {}
    for c in train + val:
        r = ref.get_reference(c, n, sub, "dense")
        refs[c.id] = r
        ref_seconds[c.id] = {"seconds": r["seconds"], "from_cache": r["from_cache"]}
    ctx = mt.make_context(grid)
    i0 = int(fg.i0)
    cases_sha = cs.manifest()["sha256"]
    ident = identity.make_identity(stage=STAGE, n=n, sub=sub, quick=quick, cases_sha256=cases_sha,
                                   train=[c.id for c in train], val=[c.id for c in val], ms=list(args.m),
                                   lams=LAMS, n_h=NHS, banks=BANKS)
    ihash = identity.identity_hash(ident)
    fit_key = {"n": n, "sub": sub, "train": [c.id for c in train], "lams": LAMS}

    def dump(path, obj):
        obj = dict(obj)
        obj.update({"status": "done", "stage": STAGE, "identity": ident, "identity_hash": ihash,
                    "wall_seconds": now() - t_wall, "quick": quick})
        identity.write_json(path, obj)

    fit_dir = os.path.join(scc.RESULTS, "fit")
    os.makedirs(fit_dir, exist_ok=True)
    manifest = identity.RunManifest("fit", [f"m{m}" for m in args.m])

    constants, nu_report, rep_report = {}, {}, {}
    selection, lock_sel = {}, {}
    fit_seconds, search_total, target_seconds, val_seconds = {}, {}, {}, {}
    skipped = {}
    times_dmin = np.concatenate([[0.0], fg.t[fg.out_idx]])

    for m in args.m:
        manifest.start(f"m{m}")
        try:
            tm0 = now()
            tgs = [tgm.build_targets(c, refs[c.id], grid, m) for c in train]
            target_seconds[f"m{m}"] = {"total": now() - tm0, "per_case": (now() - tm0) / len(train)}
            vtg = [tgm.build_targets(c, refs[c.id], grid, m) for c in val]
            rms = {tg["case_id"]: tg["rms_M"] for tg in tgs}
            nrows = {tg["case_id"]: len(tg["t"]) for tg in tgs}
            w = tgm.trajectory_weights(rms, nrows)
            d_min, d_max = tgm.d_extremes(train, grid, times_dmin)
            lam_next = float(cv.sine_eigenvalues(n, grid.dk, m + 1))
            r1 = d_min * abs(lam_next)
            ratio = d_max / d_min
            constants[f"m{m}"] = {"d_min": d_min, "d_max": d_max, "ratio": ratio, "lam_m_plus_1": lam_next, "r1": r1,
                                  "train_times": "24 output times + 0"}
            print(f"[s4] m={m}: d_min={d_min:.4g} d_max={d_max:.4g} ratio={ratio:.4g} r1={r1:.4g}")

            V_int = tgs[0]["V_int"]
            lam_V = tgs[0]["lam_V"]
            vcs = [ValCase(c, tg, fg, grid, ctx) for c, tg in zip(val, vtg)]
            qcache = {}
            cands = []          # dict(family, name, spec, path, config, gm, ...)
            m_models = {}

            # RRR baseline (validation reference numbers only)
            rrr_spec = cl.ModelSpec(name=f"RRR_m{m}", m=m, closure="RRR")
            rrr_eval, rrr_gm, rrr_bad = evaluate(rrr_spec, vcs)

            def register(family, spec, path, config, train_info, secs, extra_fit=None):
                tr = {"fit": train_info, "insample": insample_residuals(spec, tgs, qcache),
                      "weights": w, "family": family}
                sha = modelio.save_spec(spec, path, config=config, training=tr, offline_seconds=secs,
                                        extra={"fit_key": fit_key, "family": family})
                return sha

            # ---------------- QL
            t0 = now()
            nu_adm = cl.admissible_bound([tg["sys"] for tg in tgs], V_int, times_dmin)
            fitq = cl.fit_nu(tgs, w, nu_adm)
            sec_ql = now() - t0
            nu_report[f"m{m}"] = fitq
            search_total[f"m{m}"] = {"QL": sec_ql}
            ql = {}
            for nm, key in (("QLp", "nu_plus"), ("QLpm", "nu_pm"), ("QLs", "nu_unclipped")):
                spec = cl.ModelSpec(name=nm, m=m, closure="QL", params={"nu": float(fitq[key])})
                path = modelio.spec_path(m, nm)
                register(nm, spec, path, {"nu": float(fitq[key]), "rule": key}, fitq, sec_ql / 3)
                ev, gm, nbad = evaluate(spec, vcs)
                ql[nm] = {"file": path, "config": {"nu": float(fitq[key]), "rule": key}, "gm": gm, "eval": ev,
                          "n_unstable": nbad, "spec": spec}
                fit_seconds[os.path.basename(path)[:-4] + f"@m{m}"] = sec_ql / 3
            sens = []
            nus = fitq["nu_unclipped"]
            for mult in SENS_MULTS:
                spec = cl.ModelSpec(name="QLsens", m=m, closure="QL", params={"nu": float(mult * nus)})
                ev, gm, nbad = evaluate(spec, vcs)
                sens.append({"multiple_of_nu_unclipped": mult, "nu": float(mult * nus), "gm_e_dyn": gm,
                             "n_unstable": nbad, "e_dyn": ev})
            dump(os.path.join(fit_dir, f"ql_sensitivity_m{m}.json"),
                 {"m": m, "note": "validation sensitivity panel; NOT used for selection", "nu_fit": fitq,
                  "rrr_gm_e_dyn": rrr_gm, "entries": sens})

            # ---------------- LM / LMb
            lm_sec = {"LM": 0.0, "LMb": 0.0}
            for fam, icpt in (("LM", False), ("LMb", True)):
                if fam == "LMb" and m == 33:
                    skipped.setdefault(f"m{m}", {})["LMb"] = "m=33 is secondary: only LM and QL are run"
                    continue
                for lam in LAMS:
                    path = modelio.spec_path(m, fam, 0, lam_tag(lam))
                    cfg = {"lam_rel": lam, "bank": None, "n_h": 0, "alphas": None, "intercept": icpt}
                    res = try_resume(path, fit_key, args.resume)
                    if res is not None:
                        spec, side = res
                        secs = float(side["offline_seconds"])
                        print(f"[s4]   resume {os.path.basename(path)}")
                    else:
                        t0 = now()
                        H, b, info = cl.fit_LM(tgs, w, lam, intercept=icpt)
                        secs = now() - t0
                        spec = cl.ModelSpec(name=fam, m=m, closure="LM", params={"H": H, "b": b})
                        register(fam, spec, path, cfg, info, secs)
                    lm_sec[fam] += secs
                    fit_seconds[os.path.basename(path)[:-4] + f"@m{m}"] = secs
                    t1 = now()
                    ev, gm, nbad = evaluate(spec, vcs)
                    val_seconds[f"m{m}"] = val_seconds.get(f"m{m}", 0.0) + now() - t1
                    cands.append({"family": fam, "file": path, "config": cfg, "gm": gm, "eval": ev,
                                  "n_unstable": nbad, "offline_seconds": secs})
            search_total[f"m{m}"].update(lm_sec)

            # ---------------- LR
            rl, rl_src = (None, "quick: ignored") if quick else representation_limited(m)
            skip_lr = None
            if m == 33:
                skip_lr = "m=33 is secondary: only LM and QL are run"
            elif rl:
                skip_lr = f"representation_limited=True at m={m} ({rl_src}): LR escalation skipped"
            rep_report[f"m{m}"] = {"representation_limited": rl, "source": rl_src, "lr_skipped": skip_lr}
            lr_sec = 0.0
            if skip_lr:
                skipped.setdefault(f"m{m}", {})["LR"] = skip_lr
                print(f"[s4] m={m}: {skip_lr}")
            else:
                for nh in NHS:
                    for bank in BANKS:
                        alphas = cl.rate_bank(nh, bank, r1, ratio)
                        for lam in LAMS:
                            tag = f"{lam_tag(lam)}_bank{bank}"
                            cfg = {"lam_rel": lam, "bank": bank, "n_h": nh, "alphas": alphas.tolist()}
                            p0 = modelio.spec_path(m, "LR0", nh, tag)
                            p1 = modelio.spec_path(m, "LR", nh, tag)
                            r0, r1_ = try_resume(p0, fit_key, args.resume), try_resume(p1, fit_key, args.resume)
                            if r0 is not None and r1_ is not None:
                                spec0, spec = r0[0], r1_[0]
                                secs = float(r1_[1]["offline_seconds"])
                                print(f"[s4]   resume {os.path.basename(p1)}")
                            else:
                                t0 = now()
                                try:
                                    spec0, spec, info = cl.fit_LR(tgs, w, nh, alphas, lam, als_iters=30, tol=1e-6, seed=0)
                                    for sp_ in (spec0, spec):
                                        sp_.params["K_init"] = cl.fit_initializer(sp_, tgs, w, i0, lam)
                                except Exception:
                                    print(f"[s4]   LR fit FAILED nh={nh} bank={bank} lam={lam:g}\n" + traceback.format_exc())
                                    cands.append({"family": "LR", "file": p1, "config": cfg, "gm": None, "eval": {},
                                                  "n_unstable": -1, "offline_seconds": now() - t0, "failed": True})
                                    continue
                                secs = now() - t0
                                register("LR0", spec0, p0, cfg, {"note": "G fixed (G0); shared fit with LR", **{
                                    k: info[k] for k in ("lam", "n_rows")}, "objective_first": info["objective"][0]}, secs)
                                register("LR", spec, p1, cfg, info, secs)
                            lr_sec += secs
                            for fam, sp_, pth in (("LR0", spec0, p0), ("LR", spec, p1)):
                                fit_seconds[os.path.basename(pth)[:-4] + f"@m{m}"] = secs
                                t1 = now()
                                ev, gm, nbad = evaluate(sp_, vcs)
                                val_seconds[f"m{m}"] = val_seconds.get(f"m{m}", 0.0) + now() - t1
                                cands.append({"family": fam, "file": pth, "config": cfg, "gm": gm, "eval": ev,
                                              "n_unstable": nbad, "offline_seconds": secs})
            search_total[f"m{m}"]["LR0"] = lr_sec
            search_total[f"m{m}"]["LR"] = lr_sec
            search_total[f"m{m}"]["LR_note"] = "LR0 and LR come from one shared fit per config; the figure is the total"

            # ---------------- selection (validation only)
            sel_m, lock_m = {}, {}
            for nm in ("QLp", "QLpm", "QLs"):
                q = ql[nm]
                lock_m[nm] = {"file": q["file"], "config": q["config"]}
                sel_m[nm] = {"file": os.path.relpath(q["file"], scc.MODELS), "config": q["config"],
                             "gm_e_dyn": q["gm"], "n_unstable": q["n_unstable"], "rrr_gm_e_dyn": rrr_gm,
                             "rule": "no hyperparameters; all three variants selected"}
            for fam in FAMILIES:
                fc = [c for c in cands if c["family"] == fam]
                if not fc:
                    sel_m[fam] = None
                    continue
                best = pick_best(fc)
                if best is None:
                    sel_m[fam] = {"selected": None, "reason": "all candidates unstable or non-finite",
                                  "n_candidates": len(fc)}
                    continue
                lock_m[fam] = {"file": best["file"], "config": best["config"]}
                sel_m[fam] = {"file": os.path.relpath(best["file"], scc.MODELS), "config": best["config"],
                              "gm_e_dyn": best["gm"],
                              "e_dyn_per_case": {cid: {p: v["e_dyn"] for p, v in d.items()}
                                                 for cid, d in best["eval"].items()},
                              "n_candidates": len(fc), "n_disqualified": int(sum(c["gm"] is None for c in fc)),
                              "rrr_gm_e_dyn": rrr_gm,
                              "rule": "min geometric mean e_dyn over val cases x {P0,P1-W0}; instability disqualifies; "
                                      "ties -> smaller n_h, then larger lambda"}
            selection[f"m{m}"] = sel_m
            lock_sel[f"m{m}"] = lock_m

            # ---------------- validation file
            vfile = {"m": m, "val_cases": [c.id for c in val], "protocols": list(PROTOCOLS),
                     "criterion": "geometric mean of resolved interior RMS e_dyn over val cases x {P0, P1-W0}",
                     "rrr_baseline": {"gm_e_dyn": rrr_gm, "n_unstable": rrr_bad, "e_dyn": rrr_eval},
                     "candidates": {}}
            for nm, q in ql.items():
                vfile["candidates"][os.path.basename(q["file"])[:-4]] = {
                    "family": nm, "config": q["config"], "gm_e_dyn": q["gm"], "n_unstable": q["n_unstable"],
                    "e_dyn": q["eval"]}
            for c in cands:
                vfile["candidates"][os.path.basename(c["file"])[:-4]] = {
                    "family": c["family"], "config": c["config"], "gm_e_dyn": c["gm"],
                    "n_unstable": c["n_unstable"], "e_dyn": c["eval"], "offline_seconds": c["offline_seconds"],
                    **({"failed": True} if c.get("failed") else {})}
            dump(os.path.join(fit_dir, f"validation_m{m}.json"), vfile)
            manifest.done(f"m{m}")
        except Exception:
            manifest.failed(f"m{m}")
            raise

    # ---------------- lock, selection.json, offline_costs.json
    lock_path = modelio.write_lock(lock_sel, cases_sha256=cases_sha,
                                   extra={"validation_criterion": "geometric mean over val cases x {P0, P1-W0} of "
                                          "resolved interior RMS e_dyn; instability disqualifies; ties -> smaller n_h, "
                                          "larger lambda", "phase": "fit", "quick": quick, "n": n, "sub": sub,
                                          "train": [c.id for c in train], "val": [c.id for c in val]})
    # selection.json mirrors the lock (file paths now relative to models/)
    dump(os.path.join(fit_dir, "selection.json"),
         {"constants": constants, "nu": nu_report, "representation_limited": rep_report, "skipped": skipped,
          "selected": selection, "lock_file": os.path.relpath(lock_path, scc.OUTPUT_ROOT)})
    dump(os.path.join(fit_dir, "offline_costs.json"),
         {"fit_seconds_per_candidate": fit_seconds, "search_total_seconds": search_total,
          "target_construction_seconds": target_seconds, "validation_seconds": val_seconds,
          "reference_seconds_train_val": ref_seconds,
          "note": "seconds are wall time on the machine that ran s4; LR0 and LR share one fit per config"})
    manifest.finish()

    # ---------------- compact table
    print("\n[s4] selected configurations (validation geometric-mean e_dyn)")
    print(f"{'m':>3} {'closure':<6} {'n_h':>3} {'lam':>8} {'bank':>4} {'gm e_dyn':>11} {'RRR gm':>11}  file")
    for mk, d in selection.items():
        for nm in LOCK_NAMES:
            e = d.get(nm)
            if not e or e.get("file") is None:
                reason = (e or {}).get("reason") or skipped.get(mk, {}).get(nm) or skipped.get(mk, {}).get("LR", "skipped")
                if nm in ("LR0", "LR") or e:
                    print(f"{mk[1:]:>3} {nm:<6} {'-':>3} {'-':>8} {'-':>4} {'-':>11} {'-':>11}  ({reason})")
                continue
            c = e["config"]
            gm = e.get("gm_e_dyn")
            rg = e.get("rrr_gm_e_dyn")
            print(f"{mk[1:]:>3} {nm:<6} {c.get('n_h', 0) or '-':>3} {c.get('lam_rel', float('nan')) if 'lam_rel' in c else float('nan'):>8.1e} "
                  f"{str(c.get('bank') or '-'):>4} {gm if gm is not None else float('nan'):>11.4e} "
                  f"{rg if rg is not None else float('nan'):>11.4e}  {e['file']}")
    print(f"[s4] lock: {lock_path}   wall {now() - t_wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
