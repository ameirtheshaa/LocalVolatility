#!/usr/bin/env python3
"""Stage s0 -- verification before any scoring (PLAN.md section 11, docs/RESULT_SCHEMA.md).

Writes results/verify/{verify,roundoff_guard,solver_check}.json (``results/verify_quick/`` under --quick).
Checks a-h are asserted (verify.json ``pass`` = AND of them); i (Black-Scholes) is informational; j measures
the roundoff guard gamma.  Heavy at full size (n=256): run on the remote CPU box, not on the Mac.
Pure NumPy/SciPy.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

if "--quick" in sys.argv:                       # smoke runs never touch the real cache
    os.environ.setdefault("SCC_CACHE_ROOT", os.path.join(
        os.environ.get("SCC_OUTPUT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "cache_quick"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402
from scipy.linalg import expm                                 # noqa: E402

import scc                                                    # noqa: E402,F401
from scc import cases as cs                                   # noqa: E402
from scc import conventions as cv                             # noqa: E402
from scc import identity                                      # noqa: E402
from scc import metrics as mt                                 # noqa: E402
from scc import oracles as orc                                # noqa: E402
from scc import reference as ref                              # noqa: E402
from dsm import reference as dref                             # noqa: E402

STAGE = "s0_verify"
_trapz = getattr(np, "trapezoid", None) or np.trapz
M_LIST = (9, 17, 33)


# ----------------------------------------------------------------------------- helpers
def rel(err, scale):
    return float(err / scale) if scale > 0 else float(err)


def psi_int_rows(phi, grid):
    return cv.detrend_rows(phi, grid.k)[:, 1:-1]


def first_train(family):
    return [c for c in cs.cases_by_split("train") if c.family == family][0]


def setup(case, n, m):
    grid = ref.make_grid(n)
    sys_ = cv.interior_system(case, grid)
    return grid, sys_, cv.dst_interior(n, m), cv.dst_complement_interior(n, m)


# ----------------------------------------------------------------------------- a. basis
def check_basis(n):
    case = cs.get_case("C_sig0.30")
    grid = ref.make_grid(n)
    sys_ = cv.interior_system(case, grid)
    out, ok = {}, True
    for m in (9, 17, 33):
        Vf = cv.dst_basis(n, m)
        V = cv.dst_interior(n, m)
        U = cv.dst_complement_interior(n, m)
        lam = cv.sine_eigenvalues(n, grid.dk, np.arange(1, m + 1))
        e_vv = float(np.max(np.abs(V.T @ V - np.eye(m))))
        e_uu = float(np.max(np.abs(U.T @ U - np.eye(U.shape[1]))))
        e_vu = float(np.max(np.abs(V.T @ U)))
        end = float(max(np.max(np.abs(Vf[0])), np.max(np.abs(Vf[-1]))))
        R = sys_.D2_int @ V
        e_eig = rel(np.max(np.abs(R - V * lam[None, :])), np.max(np.abs(R)))
        p = bool(e_vv <= 1e-13 and e_uu <= 1e-13 and e_vu <= 1e-13 and end == 0.0 and e_eig <= 1e-10)
        out[f"m{m}"] = {"VtV_err": e_vv, "UtU_err": e_uu, "VtU_err": e_vu, "endpoint_rows_max_abs": end,
                        "D2V_eq_VLambda_rel_err": e_eig, "pass": p}
        ok &= p
    return {"values": out, "tolerance": {"orthogonality": 1e-13, "endpoint_rows": 0.0, "eigen_rel": 1e-10}, "pass": ok}


# ----------------------------------------------------------------------------- b. lift
def check_lift(n):
    out, ok = {}, True
    m = 9
    # affine payoff lift: autonomous C case and a time-dependent M case
    for case in (cs.get_case("C_sig0.30"), first_train("M")):
        grid, sys_, V, U = setup(case, n, m)
        _, ell, _ = cv.payoff_state(grid)
        rec = {}
        for t in (0.2, 0.5, 1.0):
            g = cv.forcing_g(sys_, t, ell)
            scale = float(np.max(np.abs(sys_.d(t))) * np.max(np.abs(sys_.D2_int)) * np.max(np.abs(ell)))
            tol = 1e-12 * scale
            e = float(np.max(np.abs(g)))
            rec[f"t{t}"] = {"max_abs_g": e, "tolerance": tol, "pass": bool(e <= tol)}
            ok &= e <= tol
        out[f"payoff_lift_{case.id}"] = rec
    # Dirichlet sensitivity variant: time-dependent affine lift from Black-Scholes endpoint values
    case = cs.get_case("C_sig0.30")
    grid, sys_, V, U = setup(case, n, m)
    bs = dref.bs_truth(grid, 0.3, grid.t_out)
    ell_rows = cv.lift_rows(bs, grid.k)
    ell_dot = np.gradient(ell_rows, grid.t_out, axis=0)       # central differences (one-sided at the two ends)
    rec = {"rows": []}
    worst_L, worst_exact, worst_dev = 0.0, 0.0, 0.0
    gR_max = gU_max = 0.0
    tol_ok = True
    for j in range(len(grid.t_out)):
        t = float(grid.t_out[j])
        g0 = cv.forcing_g(sys_, t, ell_rows[j])                 # L ell only
        g1 = cv.forcing_g(sys_, t, ell_rows[j], ell_dot[j])
        scale = float(np.max(np.abs(sys_.d(t))) * np.max(np.abs(sys_.D2_int)) * np.max(np.abs(ell_rows[j])))
        tol = 1e-12 * scale
        worst_L = max(worst_L, float(np.max(np.abs(g0))) / max(scale, 1e-300))
        exact = bool(np.array_equal(g1, g0 - ell_dot[j][1:-1]))
        dev = float(np.max(np.abs(g1 + ell_dot[j][1:-1])))
        worst_exact = max(worst_exact, 0.0 if exact else 1.0)
        worst_dev = max(worst_dev, dev)
        tol_ok &= bool(np.max(np.abs(g0)) <= tol) and exact and dev <= tol
        gR_max = max(gR_max, float(np.linalg.norm(V.T @ g1)))
        gU_max = max(gU_max, float(np.linalg.norm(U.T @ g1)))
    rec.update({"max_rel_L_ell": worst_L, "g_equals_g0_minus_ell_dot_bitwise": bool(worst_exact == 0.0),
                "max_abs_g_plus_ell_dot": worst_dev, "norm_g_R_max_over_maturities": gR_max,
                "norm_g_U_max_over_maturities": gU_max, "m": m,
                "note": "gR and gU magnitudes are reported, not asserted against zero",
                "tolerance": "1e-12 * max|d| * max|D2_int| * max|ell|", "pass": bool(tol_ok)})
    del rec["rows"]
    out["dirichlet_variant_C_sig0.30"] = rec
    ok &= tol_ok
    return {"values": out, "tolerance": "1e-12 * max|d| * max|D2_int| * max|ell|", "pass": bool(ok)}


# ----------------------------------------------------------------------------- c. structured vs dense
def check_assembly(n):
    out, ok = {}, True
    ids = ["C_sig0.30", first_train("S").id, first_train("D").id, first_train("M").id]
    for cid in ids:
        case = cs.get_case(cid)
        grid = ref.make_grid(n)
        sys_ = cv.interior_system(case, grid)
        rec = {}
        for m in (9, 17):
            V, U = cv.dst_interior(n, m), cv.dst_complement_interior(n, m)
            Vf = cv.dst_basis(n, m)
            lam = cv.sine_eigenvalues(n, grid.dk, np.arange(1, m + 1))
            for t in (0.2, 0.6, 1.0):
                As = cv.A_structured(sys_, t, V, lam)
                Ab = cv.blocks(sys_, t, V, U)["A"]
                # ref.L_full_dense evaluates the row at the t it is given; autonomous cases are frozen at
                # AUTONOMOUS_T by convention (solvers and InteriorSystem.d do this), so pass that time.
                Lf = ref.L_full_dense(case, grid, t if case.time_dependent else cv.AUTONOMOUS_T)
                Ad = Vf.T @ Lf @ Vf
                sc = float(np.max(np.abs(Ab)))
                e1 = rel(np.max(np.abs(As - Ab)), sc)
                e2 = rel(np.max(np.abs(As - Ad)), sc)
                p = bool(e1 <= 1e-12 and e2 <= 1e-12)
                rec[f"m{m}_t{t}"] = {"structured_vs_blocks_rel": e1, "structured_vs_full_dense_rel": e2, "pass": p}
                ok &= p
        out[cid] = rec
    return {"values": out, "tolerance": 1e-12, "pass": bool(ok)}


# ----------------------------------------------------------------------------- d. reconstruction
def check_reconstruction(n, sub):
    out, ok = {}, True
    for case in (cs.get_case("C_sig0.30"), first_train("M")):
        r = ref.get_reference(case, n, sub, "dense")
        grid = ref.make_grid(n)
        psi = psi_int_rows(r["phi"], grid)
        rec = {}
        for m in (9, 33):
            V, U = cv.dst_interior(n, m), cv.dst_complement_interior(n, m)
            c, z = psi @ V, psi @ U
            e = rel(np.max(np.abs(c @ V.T + z @ U.T - psi)), np.max(np.abs(psi)))
            rec[f"m{m}"] = {"max_rel_err_all_rows": e, "n_rows": int(psi.shape[0]), "pass": bool(e <= 1e-12)}
            ok &= e <= 1e-12
        out[case.id] = rec
    return {"values": out, "tolerance": 1e-12, "pass": bool(ok)}


# ----------------------------------------------------------------------------- e. autonomous block identity
def check_autonomous(n, sub):
    out, ok = {}, True
    for case in (cs.get_case("C_sig0.30"), first_train("S")):
        grid = ref.make_grid(n)
        sys_ = cv.interior_system(case, grid)
        r = ref.get_reference(case, n, sub, "dense")
        t, i0 = r["t"], r["i0"]
        psi = psi_int_rows(r["phi"], grid)
        _, ell, _ = cv.payoff_state(grid)
        rec = {}
        for m in (9, 17):
            V, U = cv.dst_interior(n, m), cv.dst_complement_interior(n, m)
            r1 = orc.block_identity_autonomous(sys_, V, U, psi, t, i0)
            p1 = bool(r1["max_rel_err_z"] <= 1e-9 and r1["max_rel_err_cdot"] <= 1e-9)
            # secondary identity: full solve started from phi0 = ell + V c0 (z(0) = 0 exactly), not the
            # unprojected reference.  c0 = V^T psi_ref(t0); the oracle is applied from index 0 of that run.
            c0 = psi[i0] @ V
            phi0 = ell + cv.dst_basis(n, m) @ c0
            r2ref = ref.get_reference(case, n, sub, "dense", phi0=phi0, phi0_tag=f"Vc0_m{m}")
            psi2 = psi_int_rows(r2ref["phi"], grid)
            r2 = orc.block_identity_autonomous(sys_, V, U, psi2, r2ref["t"], 0)
            p2 = bool(r2["max_rel_err_z"] <= 1e-9 and r2["max_rel_err_cdot"] <= 1e-9
                      and r2["norm_ic_term"] <= 1e-10)
            rec[f"m{m}"] = {"ordinary_reference_from_i0": {**r1, "i0": int(i0), "tolerance": 1e-9, "pass": p1},
                            "z0_equals_zero_run": {**r2, "start_index": 0, "tolerance": 1e-9,
                                                   "ic_term_abs_tolerance": 1e-10, "pass": p2,
                                                   "note": ("compares against the full solve started from "
                                                            "phi0 = ell + V c0 at t=0 (so z(0)=0 and the "
                                                            "identity is applied from index 0), NOT the "
                                                            "unprojected ordinary reference")}}
            ok &= p1 and p2
        out[case.id] = rec
    return {"values": out, "tolerance": 1e-9, "pass": bool(ok)}


# ----------------------------------------------------------------------------- f. time-ordered propagator
def check_timeordered(n, sub, quick):
    m = 9
    levels = [sub, 2 * sub] + ([] if quick else [4 * sub])
    out, ok = {}, True
    for case in (first_train("D"), first_train("M")):
        grid = ref.make_grid(n)
        sys_ = cv.interior_system(case, grid)
        V, U = cv.dst_interior(n, m), cv.dst_complement_interior(n, m)
        fwd, rev = {}, {}
        for s in levels:
            r = ref.get_reference(case, n, s, "dense")
            psi = psi_int_rows(r["phi"], grid)
            fwd[s] = orc.block_identity_timeordered(sys_, V, U, psi, r["t"], r["i0"], "forward")["max_rel_err_z"]
            rev[s] = orc.block_identity_timeordered(sys_, V, U, psi, r["t"], r["i0"], "reversed")["max_rel_err_z"]
        orders = {f"{a}->{b}": float(np.log2(fwd[a] / fwd[b])) if fwd[b] > 0 else float("inf")
                  for a, b in zip(levels[:-1], levels[1:])}
        p_ord = all(1.5 <= o <= 2.5 for o in orders.values())
        ratios = {str(s): (rev[s] / fwd[s] if fwd[s] > 0 else float("inf")) for s in levels}
        p_rev = all(rev[s] >= 10.0 * fwd[s] for s in levels)
        out[case.id] = {"m": m, "levels_sub": levels, "forward_err": {str(s): fwd[s] for s in levels},
                        "reversed_err": {str(s): rev[s] for s in levels}, "observed_order": orders,
                        "order_window": [1.5, 2.5], "reversed_over_forward": ratios,
                        "pass_order": bool(p_ord), "pass_reversed": bool(p_rev), "pass": bool(p_ord and p_rev)}
        ok &= p_ord and p_rev
    return {"values": out, "tolerance": {"order": [1.5, 2.5], "reversed_over_forward_min": 10.0}, "pass": bool(ok)}


# ----------------------------------------------------------------------------- g. B = 0 control
def check_bzero(n):
    grid = ref.make_grid(n)
    r = orc.b_zero_control(n, 9, grid.dk, d_const=0.05)     # raises AssertionError on failure
    return {"values": r, "tolerance": {"B_over_A": 1e-12, "M_rel": 1e-12, "TF_vs_RRR": 1e-12}, "pass": True}


# ----------------------------------------------------------------------------- h. solver checks
def check_solvers(n, sub):
    grid = ref.make_grid(n)
    fg = ref.fine_grid(grid, sub)
    out, ok = {}, True
    for case in (cs.get_case("C_sig0.30"), first_train("D")):
        t0 = time.time()
        dense = ref.get_reference(case, n, sub, "dense")["phi"]
        sparse = ref.solve_sparse(case, grid, fg)
        mx = float(np.max(np.abs(dense)))
        d = float(np.max(np.abs(dense - sparse)))
        p = bool(d < 1e-11 * mx)
        cn = {}
        for k in (1, 2, 4, 8):
            ph = ref.solve_banded_cn(case, grid, fg, k)
            ctx_in = mt.make_context(grid)
            rows_c, rows_d = ph[fg.out_idx], dense[fg.out_idx]
            cn[str(k)] = {"max_abs_err": float(np.max(np.abs(ph - dense))),
                          "output_rows_interior_rms": float(np.sqrt(np.mean((rows_c - rows_d)[ctx_in.interior] ** 2)))}
        ks = [1, 2, 4, 8]
        cn_orders = {f"{a}->{b}": float(np.log2(cn[str(a)]["max_abs_err"] / cn[str(b)]["max_abs_err"]))
                     for a, b in zip(ks[:-1], ks[1:]) if cn[str(b)]["max_abs_err"] > 0}
        out[case.id] = {"dense_vs_sparse_max_abs": d, "max_abs_phi": mx, "relative": d / mx,
                        "tolerance": 1e-11, "pass_dense_sparse": p, "cn_vs_dense": cn,
                        "cn_observed_orders_informational": cn_orders, "seconds": time.time() - t0}
        ok &= p
    return {"values": out, "tolerance": "max|dense-sparse| < 1e-11 * max|phi|", "pass": bool(ok)}


# ----------------------------------------------------------------------------- i. Black-Scholes
def check_bs(n, sub):
    case = cs.get_case("C_sig0.30")
    grid = ref.make_grid(n)
    fg = ref.fine_grid(grid, sub)
    r = ref.get_reference(case, n, sub, "dense")
    rows = r["phi"][fg.out_idx]
    bs = dref.bs_truth(grid, 0.3, grid.t_out)
    ctx = mt.make_context(grid)
    pm = mt.price_metrics(rows, bs, ctx)
    return {"values": {"price_interior_rms_vs_bs": pm["price_interior_rms"], "price_full_rms_vs_bs": pm["price_full_rms"],
                       "price_max_abs_vs_bs": pm["price_max_abs"], "n": n, "sub": sub,
                       "note": "informational; ~2e-5 expected at n=256"},
            "tolerance": None, "pass": None}


# ----------------------------------------------------------------------------- j. roundoff guard
def _rollout(sys_, V_int, lam, t, i0, c0, td):
    """Plain Galerkin rollout. ``lam`` is the diagonal eigenvalue vector (A = S diag(lam)) or, for a rotated
    coefficient basis V Q, the full matrix Q^T diag(lam) Q (then A = S_rot @ lam, S_rot = (VQ)^T diag(d) (VQ))."""
    m = V_int.shape[1]
    N = len(t) - 1
    h = float(t[1] - t[0])
    lam = np.asarray(lam, float)

    def A_of(tt):
        if lam.ndim == 2:
            return cv.S_matrix(sys_, tt, V_int) @ lam
        return cv.A_structured(sys_, tt, V_int, lam)

    c = np.full((N + 1, m), np.nan)
    c[i0] = c0
    E = None if td else expm(h * A_of(1.0))
    for s in range(i0, N):
        if td:
            E = expm(h * A_of(0.5 * (t[s] + t[s + 1])))
        c[s + 1] = E @ c[s]
    return c


def _neg_mass(c_rows, Vf, ell, ctx, reverse=False):
    if reverse:
        pad = np.flip(c_rows, axis=1) @ np.flip(Vf.T, axis=0)
    else:
        pad = c_rows @ Vf.T
    phi = ell[None, :] + pad
    f = mt.density_rows(phi, ctx, "d2_repo")
    return np.array([float(_trapz(np.maximum(-f[i], 0.0), ctx.K_rows[i])) for i in range(f.shape[0])])


def roundoff_guard(n, sub, quick):
    grid = ref.make_grid(n)
    fg = ref.fine_grid(grid, sub)
    ctx = mt.make_context(grid)
    _, ell, psi0 = cv.payoff_state(grid)
    dev = cs.cases_by_split("train") + cs.cases_by_split("val") + cs.cases_by_split("dev_regression")
    if quick:
        pick = ["C_sig0.30", "S_seed9", "F_fresh_base0.28_t0.2_x0.05", first_train("D").id, first_train("M").id, "C_sig0.240"]
        dev = [cs.get_case(p) for p in pick]
    t = fg.t
    pairs, gmax = {}, 0.0
    per_variant_max = {"flipped_all_signs": 0.0, "reversed_reconstruction_order": 0.0, "alternating_signs": 0.0,
                       "rotated_coefficient_basis": 0.0}
    base_nm_max = 0.0
    for case in dev:
        sys_ = cv.interior_system(case, grid)
        td = bool(case.time_dependent)
        r = ref.get_reference(case, n, sub, "dense")
        psi_ref0 = cv.detrend_rows(r["phi"][r["i0"]], grid.k)[1:-1]
        for m in M_LIST:
            V = cv.dst_interior(n, m)
            Vf = cv.dst_basis(n, m)
            lam = cv.sine_eigenvalues(n, grid.dk, np.arange(1, m + 1))
            sgn = np.where(np.arange(m) % 2 == 0, 1.0, -1.0)
            # (5) rotated coefficient basis: V Q with Q orthogonal (QR of a seeded Gaussian), c0 -> Q^T c0.
            # Same subspace, same reconstruction in exact arithmetic, but the exponential and every matvec
            # are evaluated on different floating-point numbers (sign flips alone are bitwise exact).
            Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((m, m)))
            for proto, i0, c0 in (("P0", 0, psi0[1:-1] @ V), ("P1", r["i0"], psi_ref0 @ V)):
                idx = fg.out_idx
                c1 = _rollout(sys_, V, lam, t, i0, c0, td)[idx]
                nm1 = _neg_mass(c1, Vf, ell, ctx)
                c2 = _rollout(sys_, -V, lam, t, i0, -c0, td)[idx]
                nm2 = _neg_mass(c2, -Vf, ell, ctx)
                nm3 = _neg_mass(c1, Vf, ell, ctx, reverse=True)
                c4 = _rollout(sys_, V * sgn[None, :], lam, t, i0, c0 * sgn, td)[idx]
                nm4 = _neg_mass(c4, Vf * sgn[None, :], ell, ctx)
                c5 = _rollout(sys_, V @ Q, Q.T @ np.diag(lam) @ Q, t, i0, Q.T @ c0, td)[idx]
                nm5 = _neg_mass(c5, Vf @ Q, ell, ctx)
                d2 = float(np.max(np.abs(nm2 - nm1)))
                d3 = float(np.max(np.abs(nm3 - nm1)))
                d4 = float(np.max(np.abs(nm4 - nm1)))
                d5 = float(np.max(np.abs(nm5 - nm1)))
                pairs[f"{case.id}|m{m}|{proto}"] = {"flipped_all_signs": d2, "reversed_reconstruction_order": d3,
                                                    "alternating_signs": d4, "rotated_coefficient_basis": d5,
                                                    "max_negative_mass_base": float(nm1.max())}
                per_variant_max["flipped_all_signs"] = max(per_variant_max["flipped_all_signs"], d2)
                per_variant_max["reversed_reconstruction_order"] = max(per_variant_max["reversed_reconstruction_order"], d3)
                per_variant_max["alternating_signs"] = max(per_variant_max["alternating_signs"], d4)
                per_variant_max["rotated_coefficient_basis"] = max(per_variant_max["rotated_coefficient_basis"], d5)
                base_nm_max = max(base_nm_max, float(nm1.max()))
                gmax = max(gmax, d2, d3, d4, d5)
        print(f"  roundoff {case.id}: running max paired |d neg mass| = {gmax:.3e}", flush=True)
    gamma = max(10.0 * gmax, 1e-16)
    procedure = ("For each development case (train+val+dev_regression), m in {9,17,33} and protocol P0 (payoff start "
                 "from t=0, c0=V^T psi0) / P1 (warm start at t0 index, c0=V^T psi_ref(t0) from the Full-n@sub dense "
                 "reference): plain RRR Galerkin rollout c_{k+1}=expm(h A_mid) c_k, A_mid=A_structured at the step "
                 "midpoint (one exponential for autonomous cases), reconstruction phi=ell+c V^T on the 24 output "
                 "rows, density by d2_repo, negative mass = trapezoid in K of max(-f,0). Variants: (1) basis V; "
                 "(2) basis -V (all column signs flipped, c0 -> -c0); (3) variant (1) with the reconstruction sum "
                 "evaluated with both factors reversed along the summed (column) axis; (4, extra, more conservative "
                 "than the plan) alternating column signs; (5, extra) rotated coefficient basis V Q with Q a seeded "
                 "random orthogonal matrix (same subspace; the exponential and all matvecs act on different "
                 "floating-point numbers). Paired difference = max over the 24 rows of "
                 "|negmass_variant - negmass_(1)|; gamma = 10 * max over all cases/m/protocols/variants "
                 "(floor 1e-16).")
    return {"gamma": gamma, "max_paired_abs_diff": gmax, "per_variant_max": per_variant_max,
            "gamma_from_plan_variants_only": max(10.0 * max(per_variant_max["flipped_all_signs"],
                                                            per_variant_max["reversed_reconstruction_order"]), 1e-16),
            "max_negative_mass_base": base_nm_max, "n_cases": len(dev), "m_list": list(M_LIST),
            "paired_differences": pairs, "procedure": procedure, "n": n, "sub": sub}


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--sub", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="n=64, sub=4, 6 roundoff cases (smoke test only; writes results/verify_quick)")
    ap.add_argument("--resume", action="store_true", help="skip when all three result files are done with matching identity")
    args = ap.parse_args()
    n, sub = (64, 4) if args.quick else (args.n, args.sub)
    outdir = os.path.join(scc.RESULTS, "verify_quick" if args.quick else "verify")
    paths = {k: os.path.join(outdir, k + ".json") for k in ("verify", "roundoff_guard", "solver_check")}
    idents = {k: identity.make_identity(stage=STAGE, file=k, n=n, sub=sub, quick=bool(args.quick)) for k in paths}
    if args.resume and all(identity.result_status(paths[k], idents[k])[0] == "done" for k in paths):
        print("s0_verify: all results done with matching identity; nothing to do")
        return 0
    for k in paths:
        st, detail = identity.result_status(paths[k], idents[k])
        if st != "missing":
            old = identity.supersede(paths[k])
            print(f"[{k}] previous result {st} ({detail}) kept as {old}")
    t_start = time.time()
    manifest = identity.RunManifest("verify_quick" if args.quick else "verify", ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    checks, timings = {}, {}
    steps = [("a_basis", lambda: check_basis(n)), ("b_lift", lambda: check_lift(n)),
             ("c_assembly", lambda: check_assembly(n)), ("d_reconstruction", lambda: check_reconstruction(n, sub)),
             ("e_autonomous_block", lambda: check_autonomous(n, sub)),
             ("f_timeordered", lambda: check_timeordered(n, sub, args.quick)),
             ("g_b_zero_control", lambda: check_bzero(n)), ("h_solvers", lambda: check_solvers(n, sub)),
             ("i_black_scholes", lambda: check_bs(n, sub))]
    for name, fn in steps:
        manifest.start(name)
        t0 = time.time()
        try:
            checks[name] = fn()
            manifest.done(name)
        except Exception:
            checks[name] = {"values": None, "tolerance": None, "pass": False if not name.startswith("i_") else None,
                            "traceback": traceback.format_exc()}
            manifest.failed(name)
            print(checks[name]["traceback"], file=sys.stderr)
        timings[name] = time.time() - t0
        p = checks[name]["pass"]
        print(f"{name:22s} {'PASS' if p else ('info' if p is None else 'FAIL'):5s} {timings[name]:8.1f}s", flush=True)
    asserted = [v["pass"] for k, v in checks.items() if v["pass"] is not None]
    allpass = bool(all(asserted))
    # roundoff guard (j)
    manifest.start("j_roundoff")
    t0 = time.time()
    rg_status, rg = "done", None
    try:
        rg = roundoff_guard(n, sub, args.quick)
        manifest.done("j_roundoff")
    except Exception:
        rg_status, rg = "failed", {"traceback": traceback.format_exc()}
        manifest.failed("j_roundoff")
        print(rg["traceback"], file=sys.stderr)
    timings["j_roundoff"] = time.time() - t0
    print(f"{'j_roundoff_guard':22s} {'done' if rg_status == 'done' else 'FAIL':5s} {timings['j_roundoff']:8.1f}s"
          + (f"  gamma={rg['gamma']:.3e}" if rg_status == "done" else ""), flush=True)
    manifest.finish()
    wall = time.time() - t_start

    def stamp(k, status, body, w):
        d = {"status": status, "identity": idents[k], "identity_hash": identity.identity_hash(idents[k]),
             "stage": STAGE, "wall_seconds": w}
        d.update(body)
        identity.write_json(paths[k], d)

    stamp("verify", "done" if all(v.get("traceback") is None for v in checks.values()) else "failed",
          {"pass": allpass, "checks": checks, "check_seconds": timings, "n": n, "sub": sub, "quick": bool(args.quick),
           "summary": {k: v["pass"] for k, v in checks.items()}}, wall)
    stamp("roundoff_guard", rg_status, rg if rg_status == "done" else rg, timings["j_roundoff"])
    sc = checks.get("h_solvers", {})
    stamp("solver_check", "done" if sc.get("values") is not None else "failed",
          {"pass": sc.get("pass"), "dense_vs_sparse_and_cn": sc.get("values"), "black_scholes": checks.get("i_black_scholes", {}).get("values")},
          timings.get("h_solvers", 0.0))
    print(f"s0_verify: asserted checks {'ALL PASS' if allpass else 'FAILURES'} ({len(asserted)} asserted) "
          f"wall {wall:.1f}s -> {outdir}")
    return 0 if allpass and rg_status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
