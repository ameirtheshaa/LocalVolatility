#!/usr/bin/env python3
"""Gate 2b — basis diagnostic (not a scored closure; see FROZEN_PLAN.md Addendum 1).

For OP-B and every case/keep, compare the resolved-only Galerkin error at matched dimension
m = 2·keep+1 for (i) the repository Fourier basis, (ii) Dirichlet sine modes, (iii) the weighted
eigenbasis of L(t̃=1). Rollouts: c_{n+1} = e^{h Vᵀ L(t_mid) V} c_n from Vᵀ x(t̃0); error measured
on the repository interior band at the 24 maturities, in ψ coordinates.
Writes results/gate2/basis/<case>__B.json and results/gate2/basis/summary.json.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsm import cases, checkpoint, metrics, operators as op, reference  # noqa: E402

OUT = os.path.join(checkpoint.RESULTS, "gate2", "basis")
ALGO_VERSION = 1


def case_identity(case, n_k, sub, keeps):
    from dsm import repo
    cfg = repo.DupirePipelineConfig()
    return checkpoint.make_identity(gate="gate2b", algo=ALGO_VERSION, case=case.config_hash(), case_id=case.id, variant="B",
                                    n_k=int(n_k), sub=int(sub), K_min=float(cfg.K_min), K_max=float(cfg.K_max), keeps=list(keeps))


def galerkin_rollout(L_of_t, V, x_ref, t, constant):
    c = np.empty((len(t), V.shape[1]))
    c[0] = V.T @ x_ref[0]
    h = t[1] - t[0]
    if constant:
        E = expm(h * (V.T @ L_of_t(t[0]) @ V))
        for n in range(len(t) - 1):
            c[n + 1] = E @ c[n]
    else:
        for n in range(len(t) - 1):
            Lm = L_of_t(0.5 * (t[n] + t[n + 1]))
            c[n + 1] = expm(h * (V.T @ Lm @ V)) @ c[n]
    return c


def sine_basis(n: int, m: int) -> np.ndarray:
    i = np.arange(n)
    cols = [np.sin(np.pi * j * i / (n - 1)) for j in range(1, m + 1)]
    V, _ = np.linalg.qr(np.stack(cols, axis=1))
    return V


def run_case(case: cases.Case, n_k: int, sub: int, keeps):
    grid = op.build_grid(n_k=n_k)
    fg = reference.fine_grid(grid, sub=sub)
    phi = reference.get_reference(case, grid, "B", fg)
    psi = reference.state_variable(phi, grid, "B")
    interior = metrics.interior_masks(grid)
    L_of_t = reference.operator_builder(case, grid, "B")
    t = fg.t[fg.i0:]
    out_rel = fg.out_idx - fg.i0
    psi_w = psi[fg.i0:]
    # interior system for the eigenbasis (ψ vanishes at both ends, boundary columns act on zeros)
    sl = slice(1, -1)
    dvec = op.eta_row(case.sigma_fn, grid, grid.T_max) * grid.k ** 2
    S = 1.0 / np.sqrt(dvec[sl])                      # w = S ψ_int, L̃ = S L_int S⁻¹ symmetric for constant L
    L1 = L_of_t(1.0)[sl, sl]
    Lt1 = S[:, None] * L1 / S[None, :]
    ev, Vt = np.linalg.eigh(0.5 * (Lt1 + Lt1.T))
    order = np.argsort(np.abs(ev))
    w_ref = psi_w[:, sl] * S[None, :]

    def Lt_of_t(tt):
        Lm = L_of_t(tt)[sl, sl]
        return S[:, None] * Lm / S[None, :]

    res = {"case": case.id, "family": case.family, "split": case.split, "variant": "B", "n_k": n_k,
           "config_hash": case.config_hash(), "per_keep": {}}
    for keep in keeps:
        m = 2 * keep + 1
        rec = {"dim": m}
        for name, V, x_ref, L_fn, back in (
            ("fourier", op.fourier_basis(grid.n_k, keep), psi_w, L_of_t, None),
            ("sine", sine_basis(grid.n_k, m), psi_w, L_of_t, None),
            ("eigen", Vt[:, order[:m]], w_ref, Lt_of_t, S),
        ):
            c = galerkin_rollout(L_fn, V, x_ref, t, not case.time_dependent)
            rec_rows = c[out_rel] @ V.T
            ref_rows = (x_ref[out_rel] @ V) @ V.T
            proj_rows = ref_rows - x_ref[out_rel]
            if back is not None:                     # map w → ψ on the interior, pad the zero ends
                def to_psi(rows):
                    full = np.zeros((rows.shape[0], grid.n_k))
                    full[:, sl] = rows / back[None, :]
                    return full
                rec_rows, ref_rows = to_psi(rec_rows), to_psi(ref_rows)
                proj_full = np.zeros((proj_rows.shape[0], grid.n_k))
                proj_full[:, sl] = proj_rows / back[None, :]
                proj_rows = proj_full
            rec[name] = {"rrr_resolved_interior_rms": metrics.rms(rec_rows - ref_rows, interior),
                         "projection_err_interior_rms": metrics.rms(proj_rows, interior),
                         "finite": bool(np.all(np.isfinite(c)))}
        res["per_keep"][str(keep)] = rec
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_k", type=int, default=256)
    ap.add_argument("--sub", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    done, failed = [], []
    invalidated = {}
    for case in cases.all_cases():
        path = os.path.join(OUT, f"{case.id}__B.json")
        ident = case_identity(case, args.n_k, args.sub, cases.KEEPS)
        status, detail = checkpoint.result_status(path, ident)
        if args.resume and status == "done":
            done.append(case.id)
            continue
        if status in ("stale", "corrupted", "failed", "incomplete"):
            old = checkpoint.supersede(path)
            invalidated[case.id] = {"status": status, "detail": detail, "kept_as": old}
            print(f"[{case.id}] {status}: {detail} -> recomputing (old file kept)", flush=True)
        try:
            res = run_case(case, args.n_k, args.sub, cases.KEEPS)
            res["status"] = "done"
            res["identity"] = ident
            res["identity_hash"] = checkpoint.identity_hash(ident)
            checkpoint.write_json(path, res)
            done.append(case.id)
            pk = res["per_keep"]
            print(f"[{case.id}] " + " ".join(f"k{k}: F={v['fourier']['rrr_resolved_interior_rms']:.1e}/{v['fourier']['projection_err_interior_rms']:.1e} "
                                             f"S={v['sine']['rrr_resolved_interior_rms']:.1e}/{v['sine']['projection_err_interior_rms']:.1e} "
                                             f"E={v['eigen']['rrr_resolved_interior_rms']:.1e}/{v['eigen']['projection_err_interior_rms']:.1e}" for k, v in pk.items()), flush=True)
        except Exception as e:  # pragma: no cover
            import traceback
            checkpoint.write_json(path, {"case": case.id, "status": "failed", "config_hash": case.config_hash(), "traceback": traceback.format_exc()})
            failed.append(case.id)
            print(f"[{case.id}] FAILED {e}", flush=True)
    checkpoint.write_json(os.path.join(OUT, "summary.json"), {"done": done, "failed": failed, "invalidated": invalidated,
                                                              "wall_seconds": time.time() - t0,
                                                              "identity_components": {"code": checkpoint.numerical_code_identity(), "algo": ALGO_VERSION}})
    print(f"gate2b: done={len(done)} failed={len(failed)} in {time.time()-t0:.0f}s")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
