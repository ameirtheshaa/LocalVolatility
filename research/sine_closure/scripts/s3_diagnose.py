#!/usr/bin/env python3
"""s3 -- diagnostics on DEVELOPMENT data only (train, val, dev_regression): PROJ / RRR / TF rows, BLOCK identity,
projection-vs-dynamics decomposition, density-definition comparison, per-case numerical uncertainty, the
once-per-family spline and domain checks, and the PLAN.md section 10 representation-limited stopping rule.

Outputs (under OUTPUT_ROOT/results; '_quick' directories when --quick or --n != 256):
  diagnose/<case_id>.json, diagnose/family_checks.json, diagnose/summary.json,
  uncertainty/<case_id>.json.
Requires the references written by s2_references (levels n256_s16, n511_s16, n256_s32 mandatory; n1021_s16 and
n256_s64 used for convergence orders when present).  --generate lets this script compute missing levels itself
(slow at full size).  Heavy at full size: run on the remote CPU box, not on the Mac.

Usage: s3_diagnose.py [--split train val dev_regression] [--cases ID ...] [--max-cases N] [--resume] [--quick]
                      [--n 256] [--sub 16] [--generate] [--skip-family-checks]
"""
from __future__ import annotations

import argparse
import glob
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
if _N_ARG != 256:                              # small-grid smoke runs never touch the real reference cache
    os.environ.setdefault("SCC_CACHE_ROOT", os.path.join(
        os.environ.get("SCC_OUTPUT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "cache_quick"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402

import scc                                                    # noqa: E402
from scc import cases as cs                                   # noqa: E402
from scc import evaluate as ev                                # noqa: E402
from scc import identity                                      # noqa: E402
from scc import metrics as mt                                 # noqa: E402
from scc import oracles as orc                                # noqa: E402
from scc import reference as rf                               # noqa: E402
from scc.closures import ModelSpec                            # noqa: E402

STAGE = "diagnose"
SCRIPT = os.path.abspath(__file__)
M_ALL = [9, 13, 17, 25, 33, 65]
M_BLOCK = (9, 17)
M_RULE = (9, 17)
DEV_SPLITS = ["train", "val", "dev_regression"]
DENSITY_DEFS = ("d2_repo", "d2_lin", "spectral", "fd4")


def out_dirs(small):
    sfx = "_quick" if small else ""
    return (os.path.join(scc.RESULTS, STAGE + sfx), os.path.join(scc.RESULTS, "uncertainty" + sfx))


# ----------------------------------------------------------------------------- per-case work
def decomposition_row(bundle, m, traj):
    b = bundle.basis(m)
    s0 = traj["start"]
    c_out = traj["c"][bundle.out_idx - s0]
    d = orc.decomposition(bundle.rows["same"]["phi"], bundle.ell, b["V_full"], c_out)
    inter = bundle.ctx24.interior
    return {"proj_interior_rms": ev.interior_rms(d["proj_rows"], inter),
            "dyn_interior_rms": ev.interior_rms(d["dyn_rows"], inter),
            "orthogonality_defect": d["orthogonality_defect"],
            "reconstruction_residual": d["reconstruction_residual"]}


def density_definitions_row(bundle, m, traj):
    b = bundle.basis(m)
    ctx = bundle.ctx24
    c_out = traj["c"][bundle.out_idx - traj["start"]]
    phi = bundle.ell[None, :] + c_out @ b["V_full"].T
    f = {"d2_repo": mt.density_rows(phi, ctx, "d2_repo"), "d2_lin": mt.density_rows(phi, ctx, "d2_lin"),
         "spectral": mt.density_rows_spectral(c_out, b["V_full"], b["lam_V"], ctx),
         "fd4": mt.density_rows(phi, ctx, "fd4")}
    out = {}
    for i, a in enumerate(DENSITY_DEFS):
        for bb in DENSITY_DEFS[i + 1:]:
            l2 = ev.interior_l2_rows(f[a], f[bb], ctx)
            out[f"{a}__{bb}"] = {"final": float(l2[-1]), "mean": float(np.mean(l2))}
    return out


def process_case(case, args, small):
    t0 = time.time()
    n, sub = args.n, args.sub
    grid = rf.make_grid(n)
    roles = ["same", "refined", "step"] if args.quick else None
    refs = ev.load_levels(case, roles, generate=(args.generate or args.quick), n=n, sub=sub)
    bundle = ev.make_bundle(case, grid, refs, sub)
    m_list = [9] if args.quick else [m for m in M_ALL if m <= n - 3]
    protocols = ["P0"] if args.quick else ["P0", "P1"]
    unc = ev.uncertainty(bundle, m_list)
    models, decomp, dens = {}, {}, {}
    for protocol in protocols:
        for m in m_list:
            models[ev.model_key("PROJ", m, 0, protocol)] = ev.run_oracle_row(bundle, "PROJ", m, protocol)
            row, traj = ev.run_rollout_row(bundle, ModelSpec(name="RRR", m=m, closure="RRR"), protocol, "W0", True)
            key = ev.model_key("RRR", m, 0, protocol)
            models[key] = row
            decomp[key] = decomposition_row(bundle, m, traj)
            dens[key] = density_definitions_row(bundle, m, traj)
            models[ev.model_key("TF", m, 0, protocol)] = ev.run_oracle_row(bundle, "TF", m, protocol)
            if m in M_BLOCK:
                models[ev.model_key("BLOCK", m, 0, protocol)] = ev.run_block_row(bundle, m, protocol)
    wall = time.time() - t0
    return bundle, unc, models, decomp, dens, m_list, protocols, wall


# ----------------------------------------------------------------------------- family checks (spline, domain)
def spline_vs_restriction(case, bundle, grid, sub):
    from scipy.interpolate import CubicSpline
    r512 = rf.get_reference(case, 512, sub, "dense")
    grid512 = rf.make_grid(512)
    phi512 = np.asarray(r512["phi"])[np.asarray(r512["out_idx"], int)]
    phi_sp = CubicSpline(grid512.k, phi512, axis=1)(grid.k)
    f_sp = mt.density_rows(phi_sp, bundle.ctx24, "d2_repo")
    R = bundle.rows["refined"]
    inter = bundle.ctx24.interior
    return {"n_spline_source": 512, "phi_interior_rms_diff": ev.interior_rms(phi_sp - R["phi"], inter),
            "density_interior_rms_diff": ev.interior_rms(f_sp - R["f"], inter),
            "density_interior_l2_mean": float(np.mean(ev.interior_l2_rows(f_sp, R["f"], bundle.ctx24))),
            "phi_interior_rms_rel": ev.interior_rms(phi_sp - R["phi"], inter) /
            max(ev.interior_rms(R["phi"], inter), 1e-300)}


def domain_check(case, bundle, grid, sub):
    from scipy.interpolate import CubicSpline, PchipInterpolator
    from dsm import operators as dop
    from dsm import reference as dref
    grid_w = dop.build_grid(n_k=587, K_min=250.0, K_max=6000.0)
    fg_w = dref.fine_grid(grid_w, sub)
    phi_w = dref.solve_full(case, grid_w, "B", fg_w)[fg_w.out_idx]
    base = bundle.rows["same"]["phi"]
    inter = bundle.ctx24.interior
    cub = np.empty_like(base)
    pch = np.empty_like(base)
    for i, T in enumerate(grid.T_out):
        Kw, Kb = grid_w.K_row(float(T)), grid.K_row(float(T))
        cub[i] = CubicSpline(Kw, phi_w[i])(Kb)
        pch[i] = PchipInterpolator(Kw, phi_w[i])(Kb)
    a, b = ev.interior_rms(cub - base, inter), ev.interior_rms(pch - base, inter)
    return {"n_wide": 587, "K_min_wide": 250.0, "K_max_wide": 6000.0, "interior_rms_cubic_vs_base": a,
            "interior_rms_pchip_vs_base": b, "difference_cubic_minus_pchip": a - b,
            "interior_rms_cubic_vs_pchip_field": ev.interior_rms(cub - pch, inter)}


def family_checks(args, small):
    out_dir, _ = out_dirs(small)
    path = os.path.join(out_dir, "family_checks.json")
    prev = identity.read_json(path) if os.path.isfile(path) else None
    res = (prev or {}).get("families", {}) if args.resume else {}
    grid = rf.make_grid(args.n)
    for fam in ("C", "S", "D", "M"):
        case = [c for c in cs.cases_by_split("train") if c.family == fam][0]
        if res.get(fam, {}).get("status") == "done" and res[fam].get("case_id") == case.id:
            print(f"[family {fam}] reuse", flush=True)
            continue
        t0 = time.time()
        try:
            refs = ev.load_levels(case, ["same", "refined", "step"], generate=args.generate, n=args.n, sub=args.sub)
            bundle = ev.make_bundle(case, grid, refs, args.sub)
            res[fam] = {"status": "done", "case_id": case.id,
                        "spline_vs_restriction": spline_vs_restriction(case, bundle, grid, args.sub),
                        "domain_check": domain_check(case, bundle, grid, args.sub),
                        "wall_seconds": time.time() - t0}
            print(f"[family {fam}] {case.id} done in {time.time() - t0:.1f}s", flush=True)
        except Exception:
            res[fam] = {"status": "failed", "case_id": case.id, "traceback": traceback.format_exc()}
            print(f"[family {fam}] FAILED\n{res[fam]['traceback']}", flush=True)
        ident = identity.make_identity(kind="family_checks", n=args.n, sub=args.sub,
                                       evaluate=ev.evaluate_identity_hash(), script=ev.sha_of_file(SCRIPT))
        ev.write_result(path, ident, STAGE, 0.0, status="done", families=res)


# ----------------------------------------------------------------------------- summary (stopping rule)
def build_summary(args, small):
    out_dir, unc_dir = out_dirs(small)
    per_case, notes = {}, []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        base = os.path.basename(p)
        if base in ("summary.json", "family_checks.json", "RUN_MANIFEST.json") or ".superseded." in base:
            continue
        d = identity.read_json(p)
        if not d or d.get("status") != "done" or "models" not in d:
            continue
        cid = d["case_id"]
        u = identity.read_json(os.path.join(unc_dir, cid + ".json"))
        if not u or u.get("status") != "done":
            notes.append(f"{cid}: uncertainty file missing")
            continue
        U = u["uncertainty"]
        u_dens = U["per_quantity"]["density_l2_interior_mean"]["u"]
        mods = d["models"]
        rec = {"u_dens": u_dens, "split": d.get("split"), "family": d.get("family")}
        r33 = mods.get("RRR_m33__P0")
        for m in M_RULE:
            proj, rrr, tf = (mods.get(f"PROJ_m{m}__P0"), mods.get(f"RRR_m{m}__P0"), mods.get(f"TF_m{m}__P0"))
            if not (proj and rrr and tf and r33):
                rec[f"m{m}"] = {"skipped": "needed rows (PROJ/RRR/TF at m, RRR_m33) not present"}
                continue
            dens_proj = proj["metrics"]["density_l2_interior_mean"]
            dens_33 = r33["metrics"]["density_l2_interior_mean"]
            e_rrr = rrr["metrics"]["resolved_interior_rms"]
            e_tf = tf["metrics"]["resolved_interior_rms"]
            u_dyn = U["u_dyn"][str(m)]
            rec[f"m{m}"] = {"proj_density_l2": dens_proj, "rrr33_density_l2": dens_33, "u_dens": u_dens,
                            "cond_A": bool(dens_proj > dens_33 + u_dens),
                            "e_dyn_rrr": e_rrr, "e_dyn_tf": e_tf, "u_dyn": u_dyn,
                            "cond_B": bool(e_rrr - e_tf <= 2.0 * u_dyn)}
        per_case[cid] = rec
    rules = {}
    for m in M_RULE:
        rows = [(cid, r[f"m{m}"]) for cid, r in per_case.items() if "cond_A" in r.get(f"m{m}", {})]
        nA = sum(r["cond_A"] for _, r in rows)
        nB = sum(r["cond_B"] for _, r in rows)
        N = len(rows)
        fA, fB = (nA / N, nB / N) if N else (None, None)
        rules[f"m{m}"] = {"n_cases": N, "fraction_cond_A": fA, "fraction_cond_B": fB,
                          "representation_limited": bool(N and fA >= 2 / 3 and fB >= 2 / 3),
                          "cond_A_definition": "PROJ density_l2_interior_mean (refined, P0) > RRR33 + u_dens",
                          "cond_B_definition": "resolved_interior_rms(RRR_m,P0) - resolved_interior_rms(TF_m,P0) <= 2 u_dyn(m)"}
    ident = identity.make_identity(kind="diagnose_summary", n=args.n, sub=args.sub, cases=sorted(per_case),
                                   evaluate=ev.evaluate_identity_hash(), script=ev.sha_of_file(SCRIPT))
    ev.write_result(os.path.join(out_dir, "summary.json"), ident, STAGE, 0.0, rule_per_m=rules, per_case=per_case,
                    notes=notes, rule="PLAN.md section 10 representation-limited stopping rule")
    return rules


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=DEV_SPLITS, choices=DEV_SPLITS)
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true", help="1 case (first autonomous family-C), m=9, P0 only")
    ap.add_argument("--n", type=int, default=256, help="model grid size (smoke tests only; default 256)")
    ap.add_argument("--sub", type=int, default=16, help="fine substeps per output interval (smoke only)")
    ap.add_argument("--generate", action="store_true", help="compute missing reference levels instead of failing")
    ap.add_argument("--skip-family-checks", action="store_true")
    args = ap.parse_args()
    small = args.quick or args.n != 256
    out_dir, unc_dir = out_dirs(small)

    if args.cases:
        todo = [cs.get_case(c) for c in args.cases]
    else:
        todo = [c for sp in args.split for c in cs.cases_by_split(sp)]
    for c in todo:
        if c.split.startswith("test"):
            raise SystemExit(f"s3 is development-only; refusing split {c.split}")
    if args.quick:
        cand = [c for c in todo if c.family == "C" and not c.time_dependent] or todo
        todo = cand[:1]
    if args.max_cases:
        todo = todo[:args.max_cases]
    man = identity.RunManifest(STAGE + ("_quick" if small else ""), [c.id for c in todo])
    if man.previous_interrupted:
        print("previous interrupted run:", man.previous_interrupted, flush=True)
    t_all = time.time()
    done, failed, skipped = [], [], []
    for case in todo:
        path = os.path.join(out_dir, case.id + ".json")
        upath = os.path.join(unc_dir, case.id + ".json")
        sig = ev.levels_signature(case, args.n, args.sub)
        m_list = [9] if args.quick else [m for m in M_ALL if m <= args.n - 3]
        protos = ["P0"] if args.quick else ["P0", "P1"]
        ident = ev.result_ident("diagnose", case, args.n, args.sub, SCRIPT, m_list=m_list, protocols=protos,
                                quick=bool(args.quick), levels=sig)
        if ev.prepare_target(path, ident, args.resume) == "skip" and \
                identity.result_status(upath, dict(ident, kind="uncertainty"))[0] == "done":
            print(f"[{case.id}] skip (done)", flush=True)
            skipped.append(case.id)
            continue
        ev.prepare_target(upath, dict(ident, kind="uncertainty"), False)
        man.start(case.id)
        t0 = time.time()
        try:
            bundle, unc, models, decomp, dens, m_list, protos, wall = process_case(case, args, small)
            head = dict(case=case.id, case_id=case.id, family=case.family, split=case.split, params=case.params,
                        n=args.n, sub=args.sub, levels_present=bundle.levels_present)
            ev.write_result(upath, dict(ident, kind="uncertainty"), "uncertainty", wall, uncertainty=unc, **head)
            ev.write_result(path, ident, STAGE, time.time() - t0, models=models, decomposition=decomp,
                            density_definitions=dens, u_dyn=unc["u_dyn"], e_TF=unc["e_TF"],
                            uncertainty_file=os.path.relpath(upath, scc.OUTPUT_ROOT), **head)
            man.done(case.id)
            done.append(case.id)
            print(f"[{case.id}] done: {len(models)} rows in {time.time() - t0:.1f}s", flush=True)
        except Exception:
            tb = traceback.format_exc()
            ev.write_result(path, ident, STAGE, time.time() - t0, status="failed", traceback_text=tb,
                            case_id=case.id, case=case.id)
            man.failed(case.id)
            failed.append(case.id)
            print(f"[{case.id}] FAILED\n{tb}", flush=True)
    if not args.quick and not args.skip_family_checks and not small and not args.cases:
        family_checks(args, small)
    rules = build_summary(args, small)
    man.finish()
    print(f"summary rules: {rules}", flush=True)
    print(f"done={len(done)} failed={len(failed)} skipped={len(skipped)} wall={time.time() - t_all:.1f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
