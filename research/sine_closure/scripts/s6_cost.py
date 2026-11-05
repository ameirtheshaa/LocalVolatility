#!/usr/bin/env python3
"""Stage s6 -- online/offline cost protocol (PLAN.md section 9).

RUN THIS ON THE REMOTE CPU BOX ONLY, EXCLUSIVELY (no other job), with the threads pinned *before* Python starts:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/s6_cost.py
(``scc.cost.pin_threads`` also pins at run time and records the pins, but BLAS libraries read the environment
at import.)  Timing on the Mac or on a loaded machine is meaningless; the script does not enforce this.

Inputs:  models/LOCK.json (s4), results/references/index.json (s2), results/uncertainty/<case>.json (u_price; when
         absent u_price = interior RMS of FULL256@16 vs FULL511@16 restricted to the 256 grid, computed on the fly),
         results/fit/offline_costs.json (s4), results/score/<case>.json (optional: price errors for the Pareto data).
Outputs: results/cost/cost.json, results/manifests/TIMING_ENV.json.

Protocol: for every representative case and every model (RRR m in {9,13,17,25,33,65}; locked closures at m in
{9,17}; protocols P0 and P1-W0, plus P1-WI for LR0/LR; FULL256 dense, FULL256sp sparse, FULL256cn Crank-Nicolson) the
online components init / stepping (rollout incl. per-step assembly and expm) / reconstruction / density /
inverse_vol are timed separately and summed.  Per case: one cold pass (first call, in this process), 2 untimed warm-up
passes, then 7 timed repeats in which the model order is re-randomised with ``default_rng(repeat)`` (this is why the
loop is interleaved instead of calling ``scc.cost.time_callable`` per model; the statistics are the same:
median / IQR / min / max / samples / cold).

``--quick``: one C and one D train case, n=64, sub=4, a temporary lock built from tiny fits (s4_fit.py --quick in a
subprocess), 2 repeats; everything goes to a temporary SCC_OUTPUT_ROOT.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if "--quick" in sys.argv:
    if not os.environ.get("SCC_OUTPUT_ROOT"):
        os.environ["SCC_OUTPUT_ROOT"] = tempfile.mkdtemp(prefix="scc_quick_")
    os.environ.setdefault("SCC_CACHE_ROOT", os.path.join(os.environ["SCC_OUTPUT_ROOT"], "results", "cache_quick"))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np                                             # noqa: E402

import scc                                                     # noqa: E402
from scc import cases as cs                                    # noqa: E402
from scc import closures as cl                                 # noqa: E402
from scc import conventions as cv                              # noqa: E402
from scc import cost                                           # noqa: E402
from scc import identity                                       # noqa: E402
from scc import metrics as mt                                  # noqa: E402
from scc import modelio                                        # noqa: E402
from scc import reference as ref                               # noqa: E402
from scc import targets as tgm                                 # noqa: E402

STAGE = "s6_cost"
COMPONENTS = ("init", "stepping", "reconstruction", "density", "inverse_vol")
RRR_MS = (9, 13, 17, 25, 33, 65)
CLOSED_MS = (9, 17)
CLOSURE_ORDER = ("QLp", "QLpm", "QLs", "LM", "LMb", "LR0", "LR")
CN_SUBSTEPS = (1, 2, 4, 8, 16, 32, 64)
FAMILY_OF = {"QLp": "QL", "QLpm": "QL", "QLs": "QL", "LM": "LM", "LMb": "LMb", "LR0": "LR0", "LR": "LR"}


def now():
    return time.perf_counter()


def summarize(samples, cold):
    s = np.asarray(samples, float)
    q75, q25 = np.percentile(s, [75, 25])
    return {"median": float(np.median(s)), "iqr": float(q75 - q25), "min": float(s.min()), "max": float(s.max()),
            "samples": [float(x) for x in s], "cold": float(cold)}


# ----------------------------------------------------------------------------- u_price
def _find_u_price(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("u_price"), (int, float)):
            return float(obj["u_price"])
        u = obj.get("u")
        if isinstance(u, dict) and isinstance(u.get("price_interior_rms"), (int, float)):
            return float(u["price_interior_rms"])
        for k, v in obj.items():
            if "price_interior" in str(k) and isinstance(v, dict) and isinstance(v.get("u"), (int, float)):
                return float(v["u"])
        for v in obj.values():
            r = _find_u_price(v)
            if r is not None:
                return r
    return None


def get_u_price(case, grid, fg, ctx, phi256, sub):
    p = os.path.join(scc.RESULTS, "uncertainty", f"{case.id}.json")
    d = identity.read_json(p) if os.path.isfile(p) else None
    u = _find_u_price(d) if d else None
    if u is not None:
        return {"value": u, "source": os.path.relpath(p, scc.OUTPUT_ROOT)}
    n = grid.n_k
    n2 = 2 * n - 1
    r2 = ref.get_reference(case, n2, sub, "dense")
    idx = ref.nested_index(n2, n)
    fine = ref.restrict_rows(r2["phi"][fg.out_idx], idx)
    u = mt.price_metrics(phi256[fg.out_idx], fine, ctx)["price_interior_rms"]
    return {"value": float(u), "source": f"computed on the fly: FULL{n}@s{sub} vs FULL{n2}@s{sub} restricted (grid component only)"}


# ----------------------------------------------------------------------------- jobs
class Job:
    def __init__(self, key, comps, meta):
        self.key, self.comps, self.meta, self.st = key, comps, meta, {}

    def run(self):
        out = {}
        for name, fn in self.comps:
            t0 = now()
            fn(self.st)
            out[name] = now() - t0
        out["total"] = sum(out.values())
        return out


class CaseCtx:
    def __init__(self, case, n, sub):
        self.case = case
        self.n, self.sub = n, sub
        self.grid = ref.make_grid(n)
        self.fg = ref.fine_grid(self.grid, sub)
        self.ctx = mt.make_context(self.grid)
        self.ref = ref.get_reference(case, n, sub, "dense")
        self.phi_ref = self.ref["phi"]
        self.sys = cv.interior_system(case, self.grid)
        self.td = bool(case.time_dependent)
        self.phi0, self.ell, _ = cv.payoff_state(self.grid)
        self.ident = mt.identifiable_mask(self.phi_ref[self.fg.out_idx], self.ctx)
        self.sig_true = mt.sigma_true_rows(case, self.grid, self.grid.T_out)
        self.t_out = self.fg.t[self.fg.out_idx]

    def t_op(self, t):
        return t if self.td else cv.AUTONOMOUS_T


def reduced_job(cc: CaseCtx, key, spec, m, proto, hidden):
    grid, fg, ctx, case = cc.grid, cc.fg, cc.ctx, cc.case
    n, i0 = cc.n, int(fg.i0)
    V_int, V_full = cv.dst_interior(n, m), cv.dst_basis(n, m)
    lam = cv.sine_eigenvalues(n, grid.dk, np.arange(1, m + 1))
    if proto == "P0":
        t, rows = fg.t, np.asarray(fg.out_idx)
        start = 0
    else:
        t, rows = fg.t[i0:], np.asarray(fg.out_idx) - i0
        start = i0
    psi_win = cv.detrend_rows(cc.phi_ref[start:], grid.k)[:, 1:-1] @ V_int
    refmax = float(np.max(np.linalg.norm(psi_win, axis=1)))
    sys_ = cc.sys
    out_times = cc.t_out
    ell = cc.ell
    spec_eff = spec if spec is not None else cl.ModelSpec(name=f"RRR_m{m}", m=m, closure="RRR")

    def A_mid(i):
        return cv.A_structured(sys_, 0.5 * (t[i] + t[i + 1]), V_int, lam)

    def f_init(st):
        if proto == "P0":
            phi0 = grid.payoff()
            c0 = cv.detrend_rows(phi0, grid.k)[1:-1] @ V_int
        else:
            row = cc.phi_ref[i0]
            c0 = cv.detrend_rows(row, grid.k)[1:-1] @ V_int
        st["c0"] = c0
        st["h0"] = cl.hidden_init_state(spec_eff, "WI", c0) if hidden == "WI" else None

    def f_step(st):
        st["roll"] = cl.rollout(spec_eff, A_mid, t, st["c0"], lam, h0=st["h0"], ref_norm_max=refmax,
                                time_dependent=cc.td)

    def f_recon(st):
        st["c_rows"] = st["roll"]["c"][rows]
        st["phi"] = st["c_rows"] @ V_full.T + ell[None, :]

    def f_dens(st):
        st["f"] = mt.density_rows(st["phi"], ctx, "d2_repo")

    def f_inv(st):
        c = st["c_rows"]
        if cc.td:
            A_list = [cv.A_structured(sys_, float(tt), V_int, lam) for tt in out_times]
        else:
            A_list = cv.A_structured(sys_, 0.0, V_int, lam)
        cl_ = spec_eff.closure
        if cl_ == "RRR":
            Mh = np.zeros_like(c)
        elif cl_ == "QL":
            Mh = float(spec_eff.params["nu"]) * c * lam[None, :]
        elif cl_ == "LM":
            Mh = c @ np.asarray(spec_eff.params["H"]).T
            if spec_eff.params.get("b") is not None:
                Mh = Mh + np.asarray(spec_eff.params["b"])[None, :]
        else:
            h = st["roll"]["h"][rows]
            Mh = c @ np.asarray(spec_eff.params["H"]).T + h @ np.asarray(spec_eff.params["R"]).T
        phi_t = mt.rhs_time_derivative_reduced(c, Mh, A_list, V_full)
        st["iv"] = mt.inverse_vol_metrics(st["phi"], phi_t, ctx, cc.ident, cc.sig_true)

    comps = [("init", f_init), ("stepping", f_step), ("reconstruction", f_recon), ("density", f_dens),
             ("inverse_vol", f_inv)]
    meta = {"model": key, "m": m, "protocol": proto, "hidden": hidden,
            "total_state": int(spec_eff.total_state()), "param_count": spec_eff.param_count(),
            "closure": spec_eff.closure}
    job = Job(key, comps, meta)
    job.assembly_probe = lambda: A_mid(0)
    return job


def full_job(cc: CaseCtx, key, kind, substeps=None):
    grid, fg, ctx, case = cc.grid, cc.fg, cc.ctx, cc.case
    out_idx = np.asarray(fg.out_idx)
    out_times = cc.t_out

    def f_init(st):
        st["phi0"] = grid.payoff()

    def f_step(st):
        if kind == "dense":
            st["full"] = ref.solve_dense(case, grid, fg, st["phi0"])
        elif kind == "sparse":
            st["full"] = ref.solve_sparse(case, grid, fg, st["phi0"])
        else:
            st["full"] = ref.solve_banded_cn(case, grid, fg, int(substeps), phi0=st["phi0"])

    def f_recon(st):
        st["phi"] = st["full"][out_idx]

    def f_dens(st):
        st["f"] = mt.density_rows(st["phi"], ctx, "d2_repo")

    def f_inv(st):
        if kind == "dense":
            if cc.td:
                L_list = [ref.L_full_dense(case, grid, float(tt)) for tt in out_times]
            else:
                L_list = ref.L_full_dense(case, grid, cv.AUTONOMOUS_T)
            phi_t = mt.rhs_time_derivative_full(st["phi"], L_list)
        else:
            if cc.td:
                phi_t = np.array([ref.L_full_sparse(case, grid, float(tt)) @ p for tt, p in zip(out_times, st["phi"])])
            else:
                Ls = ref.L_full_sparse(case, grid, cv.AUTONOMOUS_T)
                phi_t = np.array([Ls @ p for p in st["phi"]])
        st["iv"] = mt.inverse_vol_metrics(st["phi"], phi_t, ctx, cc.ident, cc.sig_true)

    comps = [("init", f_init), ("stepping", f_step), ("reconstruction", f_recon), ("density", f_dens),
             ("inverse_vol", f_inv)]
    meta = {"model": key, "protocol": "P0", "hidden": None, "total_state": int(grid.n_k),
            "param_count": {"fitted": 0, "fixed": 0}, "closure": key.split("__")[0],
            **({"substeps": int(substeps)} if substeps else {})}
    return Job(key, comps, meta)


# ----------------------------------------------------------------------------- scheduler
def run_schedule(jobs, repeats, warmup):
    names = list(jobs)
    cold = {k: jobs[k].run() for k in names}
    for _ in range(warmup):
        for k in names:
            jobs[k].run()
    samples = {k: {c: [] for c in COMPONENTS + ("total",)} for k in names}
    for r in range(repeats):
        for i in np.random.default_rng(r).permutation(len(names)):
            k = names[i]
            res = jobs[k].run()
            for c, v in res.items():
                samples[k][c].append(v)
    out = {}
    for k in names:
        comp = {c: summarize(samples[k][c], cold[k][c]) for c in COMPONENTS}
        tot = summarize(samples[k]["total"], cold[k]["total"])
        st = jobs[k].st
        roll = st.get("roll")
        row = {**jobs[k].meta, "cold": tot["cold"], "warm": {x: tot[x] for x in ("median", "iqr", "min", "max")},
               "warm_samples": tot["samples"], "components": comp, "online_total_median": tot["median"],
               "total_state_for_pareto": jobs[k].meta["total_state"]}
        if roll is not None:
            row["unstable"] = bool(roll["unstable"])
            row["n_expm"] = int(roll["n_expm"])
            row["rollout_internal_seconds_last"] = float(roll["seconds"])
        probe = getattr(jobs[k], "assembly_probe", None)
        if probe is not None:
            t0 = now()
            for _ in range(5):
                probe()
            row["assembly_per_call_seconds_diagnostic"] = (now() - t0) / 5
        out[k] = row
    return out


# ----------------------------------------------------------------------------- one case
def model_key(label, m, proto, hidden, n_h=0):
    base = f"{label}_m{m}" + (f"_h{n_h}" if n_h else "")
    if proto == "P0":
        return f"{base}__P0"
    return f"{base}__P1" + (f"_{hidden}" if label in ("LR0", "LR") else "")


def time_case(case, n, sub, lock, repeats, warmup, rrr_ms, closed_ms):
    cc = CaseCtx(case, n, sub)
    jobs = {}
    for m in rrr_ms:
        if m > n - 2:
            continue
        for proto in ("P0", "P1"):
            jobs[model_key("RRR", m, proto, None)] = reduced_job(cc, model_key("RRR", m, proto, None), None, m, proto, "W0")
    locked = {}
    for m in closed_ms:
        if f"m{m}" not in lock["selected"]:
            continue
        for label in CLOSURE_ORDER:
            if label not in lock["selected"][f"m{m}"]:
                continue
            spec = modelio.load_locked(lock, m, label)
            locked[(m, label)] = spec
            hid = ["W0", "WI"] if label in ("LR0", "LR") else ["W0"]
            for proto in ("P0", "P1"):
                for h in (hid if proto == "P1" else ["W0"]):
                    k = model_key(label, m, proto, h, spec.n_h)
                    jobs[k] = reduced_job(cc, k, spec, m, proto, h)
    # full solvers: CN substep search first (error vs the dense 256@sub reference, same 24 maturities)
    uinfo = get_u_price(case, cc.grid, cc.fg, cc.ctx, cc.phi_ref, sub)
    u = uinfo["value"]
    search, chosen = [], None
    for s in CN_SUBSTEPS:
        t0 = now()
        phi = ref.solve_banded_cn(case, cc.grid, cc.fg, s)
        sec = now() - t0
        err = mt.price_metrics(phi[cc.fg.out_idx], cc.phi_ref[cc.fg.out_idx], cc.ctx)["price_interior_rms"]
        search.append({"substeps": s, "price_interior_rms": float(err), "passes": bool(err <= u), "solve_seconds": sec})
        if err <= u:
            chosen = s
            break
    converged = chosen is not None
    if chosen is None:
        chosen = min(search, key=lambda r: r["price_interior_rms"])["substeps"]
    jobs["FULL256__P0"] = full_job(cc, "FULL256__P0", "dense")
    jobs["FULL256sp__P0"] = full_job(cc, "FULL256sp__P0", "sparse")
    jobs["FULL256cn__P0"] = full_job(cc, "FULL256cn__P0", "cn", chosen)
    print(f"[s6] {case.id}: {len(jobs)} jobs, u_price={u:.3e} ({uinfo['source'][:40]}), CN substeps={chosen} converged={converged}")
    rows = run_schedule(jobs, repeats, warmup)
    rows["FULL256cn__P0"].update({"substeps": int(chosen), "converged": bool(converged)})
    return {"case": case.id, "family": case.family, "split": case.split, "time_dependent": bool(case.time_dependent),
            "u_price": uinfo, "cn_substep_search": search, "cn_converged": converged, "cn_substeps": int(chosen),
            "models": rows}


# ----------------------------------------------------------------------------- offline
def offline_section(lock, n, sub, train_ids):
    oc_p = os.path.join(scc.RESULTS, "fit", "offline_costs.json")
    oc = identity.read_json(oc_p) if os.path.isfile(oc_p) else {}
    idx_p = os.path.join(scc.RESULTS, "references", "index.json")
    idx = identity.read_json(idx_p) if os.path.isfile(idx_p) else None
    level = f"n{n}_s{sub}"
    ref_secs, from_cache, missing = {}, 0, []
    if idx:
        for cid in train_ids:
            e = (idx.get(cid) or {}).get(level)
            if e and "seconds" in e:
                ref_secs[cid] = float(e["seconds"])
                from_cache += int(bool(e.get("from_cache")))
            else:
                missing.append(cid)
    ref_source = "results/references/index.json"
    case0 = cs.get_case(train_ids[0])
    grid = ref.make_grid(n)
    fg = ref.fine_grid(grid, sub)
    if not idx or missing:
        t0 = now()
        ref.solve_dense(case0, grid, fg)
        per = now() - t0
        for cid in (train_ids if not idx else missing):
            ref_secs[cid] = per
        ref_source = f"estimated: one timed dense solve of {case0.id} x number of train cases (index absent/incomplete)"
    ref_total = float(sum(ref_secs.values()))
    r0 = ref.get_reference(case0, n, sub, "dense")
    tgt = {}
    for m in CLOSED_MS + (33,):
        if m > n - 2:
            continue
        tc = cost.time_callable(lambda: tgm.build_targets(case0, r0, grid, m), repeats=3, warmup=1)
        tgt[f"m{m}"] = {"per_case_median_seconds": tc["median"], "n_train_cases": len(train_ids),
                        "total_estimate_seconds": tc["median"] * len(train_ids)}
    per_model = {}
    for mk, d in lock["selected"].items():
        m = int(mk[1:])
        for label, e in d.items():
            side = identity.read_json(os.path.join(scc.MODELS, e["file"])[:-4] + ".json") or {}
            fam = FAMILY_OF[label]
            search = ((oc.get("search_total_seconds") or {}).get(mk) or {}).get(fam)
            tgt_total = (tgt.get(mk) or {}).get("total_estimate_seconds")
            if tgt_total is None:
                tgt_total = 0.0
            tot = ref_total + tgt_total + float(search or 0.0)
            per_model[f"{label}_m{m}"] = {"own_fit_seconds": side.get("offline_seconds"),
                                          "search_total_seconds": search, "target_construction_seconds": tgt_total,
                                          "reference_generation_seconds": ref_total, "offline_total_seconds": tot}
    return {"reference_generation": {"per_case_seconds": ref_secs, "total_seconds": ref_total, "source": ref_source,
                                     "n_from_cache_in_index": from_cache},
            "target_construction": tgt, "fit_offline_costs_file": os.path.relpath(oc_p, scc.OUTPUT_ROOT) if oc else None,
            "per_model": per_model, "note": "offline total = train-reference generation + target construction (all train cases, "
                                            "from one timed case) + fit/hyperparameter-search wall time of the model family"}


def break_even(per_case, offline):
    out = {}
    for cid, res in per_case.items():
        rows = res["models"]
        out[cid] = {}
        for k, r in rows.items():
            if k.startswith(("RRR", "FULL")):
                continue
            label = k.split("_m")[0]
            m = int(k.split("_m")[1].split("_")[0].split("__")[0])
            base_k = "RRR_m33__P0" if "__P0" in k else "RRR_m33__P1"
            if base_k not in rows:
                continue
            b = rows[base_k]
            diff = b["online_total_median"] - r["online_total_median"]
            off = offline["per_model"].get(f"{label}_m{m}", {}).get("offline_total_seconds")
            sig = abs(diff) > max(b["warm"]["iqr"], r["warm"]["iqr"])
            out[cid][k] = {"offline_total_seconds": off, "online_rrr33_median": b["online_total_median"],
                           "online_model_median": r["online_total_median"], "saving_per_query": diff,
                           "difference_beyond_iqr": bool(sig),
                           "N_star": (off / diff) if (off is not None and diff > 0) else None}
    return out


def pareto_rows(per_case):
    rows = []
    for cid, res in per_case.items():
        p = os.path.join(scc.RESULTS, "score", f"{cid}.json")
        sc = identity.read_json(p) if os.path.isfile(p) else None
        for k, r in res["models"].items():
            err = None
            if sc:
                try:
                    err = sc["models"][k]["metrics"]["price_interior_rms"]
                except Exception:
                    err = None
            rows.append({"model_key": k, "case": cid, "total_state": r["total_state"],
                         "online_total_median": r["online_total_median"], "price_interior_rms": err})
    return rows


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="PLAN section 9 cost protocol (remote CPU box only)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse per-case timing files with matching identity")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args(argv)
    quick = args.quick
    t_wall = now()
    n, sub = (64, 4) if quick else (256, 16)
    repeats = args.repeats or (2 if quick else 7)
    warmup = 1 if quick else args.warmup

    if quick:
        tr = cs.cases_by_split("train")
        c_tr = [c for c in tr if c.family == "C"][0]
        d_tr = [c for c in tr if c.family == "D"][0]
        d_val = [c for c in cs.cases_by_split("val") if c.family == "D"][0]
        sel_cases = [c_tr, d_tr]
        env = dict(os.environ)
        env["SCC_OUTPUT_ROOT"] = scc.OUTPUT_ROOT
        env["SCC_CACHE_ROOT"] = scc.CACHE_ROOT
        subprocess.run([sys.executable, os.path.join(_HERE, "s4_fit.py"), "--quick", "--cases", c_tr.id, d_tr.id,
                        d_val.id], check=True, env=env, stdout=subprocess.DEVNULL)
    else:
        ti = cs.cases_by_split("test_interp")
        sel_cases = []
        for fam in ("C", "S", "D", "M"):
            f = [c for c in ti if c.family == fam]
            if f:
                sel_cases.append(f[0])
        if args.cases:
            sel_cases = [cs.get_case(c) for c in args.cases]
    if args.max_cases:
        sel_cases = sel_cases[:args.max_cases]

    lock = modelio.read_lock()
    if bool(lock.get("extra", {}).get("quick")) != quick:
        raise SystemExit("models/LOCK.json was built " + ("quick" if lock.get("extra", {}).get("quick") else "full-size")
                         + " -- mismatch with this run")
    train_ids = lock.get("extra", {}).get("train") or [c.id for c in cs.cases_by_split("train")]
    cases_sha = cs.manifest()["sha256"]
    ident = identity.make_identity(stage=STAGE, n=n, sub=sub, quick=quick, cases=[c.id for c in sel_cases],
                                   repeats=repeats, warmup=warmup, lock_cases_sha=lock.get("cases_sha256"),
                                   cases_sha256=cases_sha)
    ihash = identity.identity_hash(ident)
    out_dir = os.path.join(scc.RESULTS, "cost")
    os.makedirs(out_dir, exist_ok=True)
    manifest = identity.RunManifest("cost", [c.id for c in sel_cases])
    per_case = {}
    with cost.pin_threads():
        tenv = cost.timing_env()
        identity.write_json(os.path.join(scc.RESULTS, "manifests", "TIMING_ENV.json"), {
            "status": "done", "stage": STAGE, "identity": ident, "identity_hash": ihash, "timing_env": tenv,
            "repeats": repeats, "warmup": warmup, "thread_pins": {k: os.environ.get(k) for k in
                                                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "cases": [c.id for c in sel_cases], "n": n, "sub": sub, "quick": quick,
            "cold_definition": "first call of each component in this process (models of a case are first-called in "
                               "sequence); not a fresh interpreter per model",
            "order_randomisation": "per repeat r the models of a case are permuted with numpy default_rng(r)",
            "run_exclusively_on_remote_box": True})
        for c in sel_cases:
            part = os.path.join(out_dir, f"timing_{c.id}.json")
            if args.resume and identity.result_status(part, ident)[0] == "done":
                per_case[c.id] = identity.read_json(part)["result"]
                print(f"[s6] resume {c.id}")
                continue
            manifest.start(c.id)
            try:
                res = time_case(c, n, sub, lock, repeats, warmup, RRR_MS, CLOSED_MS)
            except Exception:
                manifest.failed(c.id)
                identity.write_json(part, {"status": "failed", "identity": ident, "identity_hash": ihash,
                                           "traceback": traceback.format_exc()})
                raise
            per_case[c.id] = res
            identity.write_json(part, {"status": "done", "stage": STAGE, "identity": ident, "identity_hash": ihash,
                                       "result": res})
            manifest.done(c.id)
        offline = offline_section(lock, n, sub, train_ids)

    be = break_even(per_case, offline)
    full = {cid: {"u_price": r["u_price"], "cn_substep_search": r["cn_substep_search"],
                  "cn_converged": r["cn_converged"], "cn_substeps": r["cn_substeps"],
                  **{k: r["models"][k] for k in r["models"] if k.startswith("FULL")}} for cid, r in per_case.items()}
    identity.write_json(os.path.join(out_dir, "cost.json"), {
        "status": "done", "stage": STAGE, "identity": ident, "identity_hash": ihash, "wall_seconds": now() - t_wall,
        "quick": quick, "n": n, "sub": sub, "repeats": repeats, "warmup": warmup,
        "cases": [c.id for c in sel_cases],
        "models": {cid: {k: v for k, v in r["models"].items() if not k.startswith("FULL")} for cid, r in per_case.items()},
        "full_solvers": full, "offline": offline, "break_even": be, "pareto": pareto_rows(per_case),
        "lock_sha_cases": lock.get("cases_sha256"), "lock_created_utc": lock.get("created_utc")})
    manifest.finish()

    print("\n[s6] example rows (warm median, ms)")
    cid = sel_cases[0].id
    hdr = f"{'model':<26}{'state':>6}" + "".join(f"{c[:9]:>11}" for c in COMPONENTS) + f"{'total':>11}"
    print(f"case {cid}\n{hdr}")
    for k, r in per_case[cid]["models"].items():
        if k.startswith(("RRR_m9__", "RRR_m33__", "LR_m9", "LM_m9", "FULL")) and "P1_WI" not in k:
            print(f"{k:<26}{r['total_state']:>6}" + "".join(f"{1e3 * r['components'][c]['median']:>11.3f}" for c in COMPONENTS)
                  + f"{1e3 * r['online_total_median']:>11.3f}")
    print(f"[s6] wrote {os.path.join(out_dir, 'cost.json')}  wall {now() - t_wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
