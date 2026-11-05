#!/usr/bin/env python3
"""Warm-start diagnostic: how much of the exact feedback M(t) after t0 = 0.2 is the unresolved-initial-condition
term B(t) Φ_D(t,t0) z0 that no deployable closure receives (PLAN.md §6, Addendum 2)?

For each case and m: z_ic(t) propagated by the same midpoint exponentials as the reference (autonomous: one
expm of h D), ic(t) = B(t) z_ic(t); memory(t) = M(t) - ic(t). Reports ||ic||/||M|| at t0+h, t=0.25, t=0.30,
the time-integrated ratio over [t0, 1], and the resolved-coordinate error (interior RMS of V(c - c_ref) at the
24 maturities) of a teacher-forced rollout that receives M - ic (the "memory-only floor", i.e. what a closure
reproducing the memory exactly but starting from a zero hidden state could at best achieve), next to the TF
oracle (full M) and RRR. Writes results/verify/ic_term_warm_start.json.

    python x_ic_term.py [--cases ID ...] [--m 9 17]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import scc  # noqa: E402
from scc import cases as cs, conventions as cv, identity, metrics as mt, oracles, reference as ref, targets  # noqa: E402


def run(case, m, grid, r):
    sysm = cv.interior_system(case, grid)
    tg = targets.build_targets(case, r, grid, m)
    V, U, lam, t, i0 = tg["V_int"], tg["U_int"], tg["lam_V"], tg["t"], int(r["i0"])
    N = len(t) - 1
    h = float(r["h"])
    out_idx = np.asarray(r["out_idx"], int)
    td = sysm.time_dependent
    z0 = tg["z"][i0]
    z_ic = np.zeros_like(tg["z"])
    z_ic[i0] = z0
    ic = np.zeros_like(tg["M"])
    bl = cv.blocks(sysm, t[i0], V, U)
    ic[i0] = bl["B"] @ z0
    E = None
    for n in range(i0, N):
        tm = 0.5 * (t[n] + t[n + 1])
        if td or E is None:
            blm = cv.blocks(sysm, tm, V, U)
            E = cv.expm_h(blm["D"], h)
        z_ic[n + 1] = E @ z_ic[n]
        Bn = cv.blocks(sysm, t[n + 1], V, U)["B"] if td else bl["B"]
        ic[n + 1] = Bn @ z_ic[n + 1]
    M = tg["M"]
    nic = np.linalg.norm(ic, axis=1)
    nM = np.linalg.norm(M, axis=1)
    def ratio_at(tt):
        j = int(round(tt / h))
        return float(nic[j] / max(nM[j], 1e-300))
    integ = float(np.trapezoid(nic[i0:], t[i0:]) / max(np.trapezoid(nM[i0:], t[i0:]), 1e-300))
    # resolved-coordinate floors (warm start): TF with full M, TF with M - ic, RRR
    A_mid = lambda n: cv.A_structured(sysm, 0.5 * (t[i0 + n] + t[i0 + n + 1]), V, lam)
    tw = t[i0:]
    c0 = tg["c"][i0]
    c_tf = oracles.teacher_forced_rollout(A_mid, M[i0:], tw, c0, td)
    c_mem = oracles.teacher_forced_rollout(A_mid, (M - ic)[i0:], tw, c0, td)
    c_rrr = oracles.teacher_forced_rollout(A_mid, np.zeros_like(M[i0:]), tw, c0, td)
    ctx = mt.make_context(grid)
    Vf = cv.dst_basis(grid.n_k, m)
    rel = out_idx - i0
    cref = tg["c"][out_idx]
    e = lambda c: mt.resolved_error(c[rel], cref, Vf, ctx.interior)["resolved_interior_rms"]
    return {"ratio_ic_over_M_at_t0_plus_h": ratio_at(t[i0] + h), "ratio_at_t0.25": ratio_at(0.25), "ratio_at_t0.30": ratio_at(0.30),
            "ratio_at_t1": float(nic[-1] / max(nM[-1], 1e-300)), "integrated_ratio_t0_to_1": integ,
            "norm_ic_t0": float(nic[i0]), "norm_M_t0": float(nM[i0]),
            "resolved_interior_rms_TF_full_M": e(c_tf), "resolved_interior_rms_TF_memory_only": e(c_mem),
            "resolved_interior_rms_RRR": e(c_rrr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["C_sig0.30", "S2_seed100", "D2_seed200", "M_seed300"])
    ap.add_argument("--m", nargs="+", type=int, default=[9, 17])
    args = ap.parse_args()
    grid = ref.make_grid(256)
    out = {}
    for cid in args.cases:
        case = cs.get_case(cid)
        r = ref.get_reference(case, 256, 16, "dense")
        out[cid] = {}
        for m in args.m:
            res = run(case, m, grid, r)
            out[cid][f"m{m}"] = res
            print(f"[{cid}] m={m}: |ic|/|M| at t0+h {res['ratio_ic_over_M_at_t0_plus_h']:.3f}, t=0.25 {res['ratio_at_t0.25']:.3f}, "
                  f"t=0.30 {res['ratio_at_t0.30']:.3f}, t=1 {res['ratio_at_t1']:.1e}, integrated {res['integrated_ratio_t0_to_1']:.3f} | "
                  f"resolved err: TF {res['resolved_interior_rms_TF_full_M']:.2e}, memory-only {res['resolved_interior_rms_TF_memory_only']:.2e}, "
                  f"RRR {res['resolved_interior_rms_RRR']:.2e}", flush=True)
    path = os.path.join(scc.RESULTS, "verify", "ic_term_warm_start.json")
    identity.write_json(path, {"results": out, "definition": __doc__})
    print("wrote", path)


if __name__ == "__main__":
    main()
