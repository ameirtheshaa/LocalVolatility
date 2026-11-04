#!/usr/bin/env python3
"""Gate 2 — exact-reference benchmark and artifact diagnosis.

For every case of the frozen matrix and both operator variants: full semidiscrete reference on
the fine grid (cached), validity of the reference itself (negative prices/densities, no-arbitrage,
inverse-vol), Black–Scholes error (family C), and per-`keep` projection/memory diagnostics with
closure targets computed from the operator. Refinement cases add grid (n_k=512), time (sub=32,
family D) and domain-truncation (wider K range) checks. Idempotent: `--resume` skips cases whose
stored config hash matches. Writes results/gate2/*.json and checkpoints/CHECKPOINT_gate2.json.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsm import cases, checkpoint, metrics, mz, operators as op, reference, repo  # noqa: E402

OUT = os.path.join(checkpoint.RESULTS, "gate2")
ALGO_VERSION = 1   # bump when the numerics of this script change (identity component)


def case_identity(case: cases.Case, variant: str, n_k: int, sub: int, refine: bool, keeps) -> dict:
    cfg = repo.DupirePipelineConfig()
    return checkpoint.make_identity(gate="gate2", algo=ALGO_VERSION, case=case.config_hash(), case_id=case.id,
                                    variant=variant, n_k=int(n_k), sub=int(sub), K_min=float(cfg.K_min),
                                    K_max=float(cfg.K_max), keeps=list(keeps), refine=bool(refine))


def case_diagnostics(case: cases.Case, variant: str, n_k: int, sub: int, keeps) -> dict:
    grid = op.build_grid(n_k=n_k)
    fg = reference.fine_grid(grid, sub=sub)
    phi = reference.get_reference(case, grid, variant, fg)
    psi = reference.state_variable(phi, grid, variant)
    rows = phi[fg.out_idx]
    interior = metrics.interior_masks(grid)
    D2m = op.d2_repo(grid.n_k, grid.dk)            # density / no-arb always in the source's convention
    phi_t_rows = metrics.time_derivative_rows(phi, fg.out_idx, fg.h)
    sig_true = metrics.sigma_true_rows(case, grid)
    res = {"case": case.id, "family": case.family, "params": case.params, "split": case.split,
           "variant": variant, "n_k": n_k, "sub": sub, "h": fg.h, "n_fine": int(len(fg.t) - 1),
           "config_hash": case.config_hash(), "status": "running"}
    res["reference_validity"] = metrics.validity_metrics(rows, phi_t_rows, grid, D2m, interior, sig_true)
    f_ref = metrics.density_rows(rows, grid, D2m)
    res["reference_density"] = metrics.density_metrics(f_ref, f_ref, grid, interior)
    res["reference_validity"]["finite"] = bool(np.all(np.isfinite(phi)))
    if case.family == "C":
        bs = reference.bs_truth(grid, case.params["sigma"], grid.t_out)
        f_bs = metrics.density_rows(bs, grid, D2m)
        res["bs_comparison"] = {
            "interior_rms": metrics.rms(rows - bs, interior), "full_rms": metrics.rms(rows - bs),
            "density_l2_interior_mean": metrics.density_metrics(f_ref, f_bs, grid, interior)["density_l2_interior_mean"],
        }
    # operator along the fine grid (constant for C/S)
    L_of_t = reference.operator_builder(case, grid, variant)
    L_end = L_of_t(1.0)
    res["operator"] = {"abscissa_L_end": mz.spectral_abscissa(L_end), "numerical_abscissa_L_end": mz.numerical_abscissa(L_end)}
    # closure targets from the operator, per keep
    per_keep = {}
    tt = fg.t[fg.i0:]
    psi_w = psi[fg.i0:]
    D2v = op.d2_repo(grid.n_k, grid.dk) if variant == "A" else op.d2_linearity(grid.n_k, grid.dk)
    for keep in keeps:
        V = op.fourier_basis(grid.n_k, keep)
        B = mz.resolved_blocks(L_end, V)
        c = psi_w @ V
        Qhat = mz.ql_regressor_matrix(D2v, V)
        if case.time_dependent:
            Mhat = np.empty_like(c)
            mark = np.empty_like(c)
            for j, tj in enumerate(tt):
                Lj = L_of_t(tj)
                Mhat[j] = mz.memory_term(Lj, V, psi_w[j:j + 1])[0]
                mark[j] = (V.T @ Lj @ V) @ c[j]
        else:
            Mhat = mz.memory_term(L_end, V, psi_w)
            mark = c @ B["L_RR"].T
        nu_star, num, den = mz.pooled_nu(Mhat, c, Qhat)
        work = np.einsum("ij,ij->i", c, Mhat)
        # weighted work: <V c, W V Mhat> with W = diag(1/(η k²)) at t_end
        dvec = op.eta_row(case.sigma_fn, grid, grid.T_max) * grid.k ** 2
        Wv = V.T @ (V / dvec[:, None])
        work_w = np.einsum("ij,ij->i", c, Mhat @ Wv.T)
        rec = {
            "dim_R": int(V.shape[1]),
            "projection_err_interior_rms": metrics.rms((psi[fg.out_idx] @ V) @ V.T - psi[fg.out_idx], interior),
            "unresolved_norm_mean": float(np.linalg.norm(psi_w - c @ V.T, axis=1).mean()),
            "abscissa_L_UU_end": mz.spectral_abscissa(B["L_UU"]),
            "abscissa_L_RR_end": mz.spectral_abscissa(B["L_RR"]),
            "memory_to_markovian_ratio_median": float(np.median(np.linalg.norm(Mhat, axis=1) / (np.linalg.norm(mark, axis=1) + 1e-300))),
            "nu_star_operator": nu_star, "nu_star_num": num, "nu_star_den": den,
            "work_euclid_frac_positive": float(np.mean(work > 0)), "work_euclid_mean": float(work.mean()),
            "work_weighted_frac_positive": float(np.mean(work_w > 0)), "work_weighted_mean": float(work_w.mean()),
            "net_feedback_norm_end": float(np.linalg.norm(Mhat[-1])), "markovian_norm_end": float(np.linalg.norm(mark[-1])),
        }
        if not case.time_dependent:
            taus = np.array([1e-3, 1e-2, 0.1, 0.4])
            rec["kernel_norms"] = dict(zip([str(t) for t in taus], mz.kernel_norms(B, taus).tolist()))
        # repository finite-difference fit on the 24 output slices (what fit_nu_ql actually does)
        P_R = V @ V.T
        L_RR_rows = []
        for T in grid.T_out:
            Lt = L_of_t(T / grid.T_max)
            L_RR_rows.append(P_R @ Lt @ P_R)
        nu_fd_pos, steps_pos = repo.fit_nu_ql(psi[fg.out_idx], grid.t_out, L_RR_rows, P_R, D2v, nonneg=True)
        nu_fd_sgn, steps_sgn = repo.fit_nu_ql(psi[fg.out_idx], grid.t_out, L_RR_rows, P_R, D2v, nonneg=False)
        # same FD estimator without any clip (raw pooled ratio), for the sign only
        nu_fd_raw, _ = repo.fit_nu_ql(psi[fg.out_idx], grid.t_out, L_RR_rows, P_R, D2v, nonneg=False, nu_clip=1e9)
        rec["nu_fit_repo_nonneg"] = float(nu_fd_pos)
        rec["nu_fit_repo_signed_clip"] = float(nu_fd_sgn)
        rec["nu_fit_repo_unclipped"] = float(nu_fd_raw)
        rec["nu_fit_repo_steps_frac_negative_unclipped"] = float(np.mean(
            repo.fit_nu_ql(psi[fg.out_idx], grid.t_out, L_RR_rows, P_R, D2v, nonneg=False, nu_clip=1e9)[1] < 0))
        per_keep[str(keep)] = rec
    res["per_keep"] = per_keep
    return res


def refinement_checks(case: cases.Case, variant: str, sub: int, keeps) -> dict:
    """Grid refinement (512), time refinement (D only) and domain truncation (wider K range)."""
    out = {}
    g256 = op.build_grid(n_k=256)
    fg256 = reference.fine_grid(g256, sub=sub)
    phi256 = reference.get_reference(case, g256, variant, fg256)
    interior = metrics.interior_masks(g256)
    # grid refinement: n_k = 512 interpolated onto the 256 grid
    g512 = op.build_grid(n_k=512)
    fg512 = reference.fine_grid(g512, sub=sub)
    phi512 = reference.get_reference(case, g512, variant, fg512)
    r256 = phi256[fg256.out_idx]
    r512 = np.array([CubicSpline(g512.k, phi512[i])(g256.k) for i in fg512.out_idx])
    out["grid_refinement_interior_rms_256_vs_512"] = metrics.rms(r256 - r512, interior)
    out["grid_refinement_full_rms_256_vs_512"] = metrics.rms(r256 - r512)
    # projection errors at 512 for the same keeps (dim R fixed, so the unresolved part changes)
    psi512 = reference.state_variable(phi512, g512, variant)
    int512 = metrics.interior_masks(g512)
    out["projection_err_interior_rms_512"] = {}
    out["nu_star_operator_512"] = {}
    L512 = reference.operator_builder(case, g512, variant)(1.0)
    D2v = op.d2_repo(512, g512.dk) if variant == "A" else op.d2_linearity(512, g512.dk)
    for keep in keeps:
        V = op.fourier_basis(512, keep)
        out["projection_err_interior_rms_512"][str(keep)] = metrics.rms((psi512[fg512.out_idx] @ V) @ V.T - psi512[fg512.out_idx], int512)
        if not case.time_dependent:
            c = psi512[fg512.i0:] @ V
            Mhat = mz.memory_term(L512, V, psi512[fg512.i0:])
            out["nu_star_operator_512"][str(keep)] = mz.pooled_nu(Mhat, c, mz.ql_regressor_matrix(D2v, V))[0]
            B = mz.resolved_blocks(L512, V)
            out.setdefault("abscissa_L_UU_512", {})[str(keep)] = mz.spectral_abscissa(B["L_UU"])
    # time refinement (Magnus step) for the time-dependent family
    if case.time_dependent:
        fg32 = reference.fine_grid(g256, sub=2 * sub)
        phi32 = reference.get_reference(case, g256, variant, fg32)
        out["time_refinement_interior_rms_sub16_vs_sub32"] = metrics.rms(r256 - phi32[fg32.out_idx], interior)
    # domain truncation: wider strike range at the same physical resolution, compared on common strikes
    cfg = repo.DupirePipelineConfig()
    dK = g256.dk * g256.K_max
    K_min_w, K_max_w = 0.5 * cfg.K_min, 2.0 * cfg.K_max
    k_lo_w = np.exp(-cfg.r * cfg.T_max) * K_min_w / K_max_w
    k_hi_w = np.exp(-cfg.r * cfg.T_min)
    n_w = int(round((k_hi_w - k_lo_w) * K_max_w / dK)) + 1
    gw = op.build_grid(n_k=n_w, K_min=K_min_w, K_max=K_max_w)
    fgw = reference.fine_grid(gw, sub=sub)
    phiw = reference.get_reference(case, gw, variant, fgw)
    errs = []
    for j, i in enumerate(fg256.out_idx):
        T = g256.T_out[j]
        Kc = g256.K_row(T)
        Kw = gw.K_row(T)
        cs = CubicSpline(Kw, phiw[fgw.out_idx[j]])
        m = interior[j] & (Kc >= Kw[0]) & (Kc <= Kw[-1])
        errs.append((phi256[i][m] - cs(Kc[m])) ** 2)
    out["domain_truncation_interior_rms"] = float(np.sqrt(np.mean(np.concatenate(errs))))
    out["wide_domain"] = {"K_min": K_min_w, "K_max": K_max_w, "n_k": n_w}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_k", type=int, default=256)
    ap.add_argument("--sub", type=int, default=16)
    ap.add_argument("--variants", default="A,B")
    ap.add_argument("--only", default="", help="comma-separated case ids")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse results whose identity (code, inputs, grid, step, refinement) matches")
    ap.add_argument("--max-cases", type=int, default=0, help="stop after this many newly computed cases (interruption test)")
    args = ap.parse_args()
    cmd = "python3 research/dupire_signed_memory/scripts/gate2_reference.py --resume"
    keeps = cases.KEEPS
    variants = args.variants.split(",")
    todo = cases.all_cases()
    if args.only:
        ids = set(args.only.split(","))
        todo = [c for c in todo if c.id in ids]
    planned = [f"{case.id}__{variant}" for case in todo for variant in variants]
    manifest = checkpoint.RunManifest("gate2", planned)
    if manifest.previous_interrupted:
        print(f"previous run interrupted: {manifest.previous_interrupted}", flush=True)
    done, failed, pending, reused = [], [], [], []
    invalidated = {}
    before = {}
    computed = 0
    t0 = time.time()
    for case in todo:
        for variant in variants:
            cid = f"{case.id}__{variant}"
            path = os.path.join(OUT, cid + ".json")
            refine = (not args.no_refine) and case.id in cases.REFINEMENT_CASES
            ident = case_identity(case, variant, args.n_k, args.sub, refine, keeps)
            status, detail = checkpoint.result_status(path, ident)
            before[cid] = status
            if status == "done" and args.resume:
                done.append(cid)
                reused.append(cid)
                continue
            if status in ("stale", "corrupted", "failed", "incomplete"):
                old = checkpoint.supersede(path)
                invalidated[cid] = {"status": status, "detail": detail, "kept_as": os.path.relpath(old, checkpoint.OUTPUT_ROOT) if old else None}
                print(f"[{cid}] {status}: {detail} -> recomputing (old file kept)", flush=True)
            if args.max_cases and computed >= args.max_cases:
                pending.append(cid)
                continue
            manifest.start(cid)
            try:
                t1 = time.time()
                res = case_diagnostics(case, variant, args.n_k, args.sub, keeps)
                if refine:
                    res["refinement"] = refinement_checks(case, variant, args.sub, keeps)
                res["status"] = "done"
                res["identity"] = ident
                res["identity_hash"] = checkpoint.identity_hash(ident)
                res["wall_seconds"] = time.time() - t1
                checkpoint.write_json(path, res)
                manifest.done(cid)
                done.append(cid)
                computed += 1
                pk = res["per_keep"]
                print(f"[{cid}] {res['wall_seconds']:.1f}s min_phi={res['reference_validity']['min_phi_rows']:+.3e} "
                      f"negmass={res['reference_density']['negative_mass_max']:.2e} "
                      + " ".join(f"k{k}:nu*={v['nu_star_operator']:+.2e},LUU={v['abscissa_L_UU_end']:+.1e},proj={v['projection_err_interior_rms']:.1e}" for k, v in pk.items())
                      + (f" | BS int {res['bs_comparison']['interior_rms']:.2e}" if "bs_comparison" in res else ""), flush=True)
            except Exception:
                tb = traceback.format_exc()
                checkpoint.write_json(path, {"case": case.id, "variant": variant, "config_hash": case.config_hash(),
                                             "identity": ident, "status": "failed", "traceback": tb})
                manifest.failed(cid)
                failed.append(cid)
                print(f"[{cid}] FAILED\n{tb}", flush=True)
    manifest.finish()
    counts = {k: sum(1 for v in before.values() if v == k) for k in ("done", "missing", "stale", "corrupted", "failed", "incomplete")}
    summary = {"done": done, "failed": failed, "pending": pending, "reused": reused, "computed": computed,
               "invalidated": invalidated, "status_before_run": counts, "previous_interrupted": manifest.previous_interrupted,
               "wall_seconds": time.time() - t0, "config": vars(args),
               "identity_components": {"code": checkpoint.numerical_code_identity(), "algo": ALGO_VERSION}}
    checkpoint.write_json(os.path.join(OUT, "summary.json"), summary)
    exit_code = 0 if not failed else 1
    checkpoint.save_checkpoint("gate2", command=cmd, exit_code=exit_code, done=done, failed=failed, pending=pending,
                               seeds={"family_S": list(cases.SPLIT_S.keys())},
                               config={**vars(args), "algo_version": ALGO_VERSION, "code_identity": checkpoint.numerical_code_identity()},
                               extra={"reused": len(reused), "computed": computed, "invalidated": invalidated,
                                      "status_before_run": counts, "previous_interrupted": manifest.previous_interrupted},
                               next_command=("python3 research/dupire_signed_memory/scripts/gate2_reference.py --resume"
                                             if (pending or failed) else
                                             "python3 research/dupire_signed_memory/scripts/gate3_compare.py --resume"))
    print(f"gate2: done={len(done)} (reused {len(reused)}, computed {computed}) failed={len(failed)} pending={len(pending)} "
          f"invalidated={len(invalidated)} before={counts} in {summary['wall_seconds']:.0f}s")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
