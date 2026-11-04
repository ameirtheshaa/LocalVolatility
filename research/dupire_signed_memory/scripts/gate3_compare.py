#!/usr/bin/env python3
"""Gate 3 — frozen comparison of closures (see FROZEN_PLAN.md).

Per (variant, keep) block: pooled ν fits on the training split (QL+, QL±, QL*), rollouts of
RRR / QL+ / QL± / QL* / ORACLE on every case, refinement band u from RRR at n_k = 256 vs 512 on
the refinement cases, oracle-gap criterion on the training split, and — if the gap is
reproducible — LM / LR / ESN fitted on train with ridge chosen on val, rolled out everywhere.
Idempotent per block (`--resume`). Writes results/gate3/blocks/<variant>_keep<keep>.json,
fixtures/*.npz (weights + hashes) and checkpoints/CHECKPOINT_gate3.json.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsm import cases, checkpoint, closures as cz, metrics, operators as op, reference, repo  # noqa: E402

OUT = os.path.join(checkpoint.RESULTS, "gate3")
FIXTURES = os.path.join(checkpoint.OUTPUT_ROOT, "fixtures")
NU_CLIP = 5e-4
RIDGE_GRID = [1e-8, 1e-6, 1e-4, 1e-2]
RES_KW = dict(n_res=200, rho=0.9, leak=0.5, in_scale_w=1.0, seed=0)


def score_rollout(case: cases.Case, setup: cz.RomSetup, grid: op.Grid, fg: reference.FineGrid,
                  result: dict, closure: cz.Closure, interior: np.ndarray, D2m: np.ndarray,
                  sig_true: np.ndarray, f_ref: np.ndarray, phi_ref_rows: np.ndarray,
                  bs_rows=None) -> dict:
    c = result["c"]
    out = {"unstable": bool(result["unstable"]), "n_last": int(result["n_last"])}
    try:
        out["closed_abscissa_end"] = cz.closed_operator_abscissa(setup, closure)
    except Exception:
        out["closed_abscissa_end"] = float("nan")
    if result["unstable"]:
        for k in ("resolved_interior_rms", "resolved_full_rms", "price_interior_rms", "price_full_rms",
                  "density_l2_interior_mean", "negative_mass_max"):
            out[k] = float("nan")
        return out
    err_rows = (c[setup.out_rel] - setup.c_ref[setup.out_rel]) @ setup.V.T
    out["resolved_interior_rms"] = metrics.rms(err_rows, interior)
    out["resolved_full_rms"] = metrics.rms(err_rows)
    phat_rows = cz.reconstruct_rows(setup, c)
    out["price_interior_rms"] = metrics.rms(phat_rows - phi_ref_rows, interior)
    out["price_full_rms"] = metrics.rms(phat_rows - phi_ref_rows)
    if bs_rows is not None:
        out["price_interior_rms_vs_bs"] = metrics.rms(phat_rows - bs_rows, interior)
    f_hat = metrics.density_rows(phat_rows, grid, D2m)
    out.update(metrics.density_metrics(f_hat, f_ref, grid, interior))
    phat_fine = c @ setup.V.T + setup.trend[None, :]
    phat_t_rows = metrics.time_derivative_rows(phat_fine, setup.out_rel, setup.h)
    out.update(metrics.validity_metrics(phat_rows, phat_t_rows, grid, D2m, interior, sig_true))
    return out


def run_block(variant: str, keep: int, n_k: int, sub: int, u_refine: bool = True) -> dict:
    t_block = time.time()
    grid = op.build_grid(n_k=n_k)
    fg = reference.fine_grid(grid, sub=sub)
    interior = metrics.interior_masks(grid)
    D2m = op.d2_repo(grid.n_k, grid.dk)
    all_cases = cases.all_cases()
    setups: Dict[str, cz.RomSetup] = {}
    aux: Dict[str, dict] = {}
    for case in all_cases:
        phi = reference.get_reference(case, grid, variant, fg)
        S = cz.build_setup(case, grid, fg, variant, keep, phi_fine=phi)
        setups[case.id] = S
        rows = phi[fg.out_idx]
        aux[case.id] = {"phi_rows": rows, "f_ref": metrics.density_rows(rows, grid, D2m),
                        "sig_true": metrics.sigma_true_rows(case, grid),
                        "bs_rows": reference.bs_truth(grid, case.params["sigma"], grid.t_out) if case.family == "C" else None}
    train = [c for c in all_cases if c.split == "train"]
    val = [c for c in all_cases if c.split == "val"]
    # ---- scalar fits (pooled over the training split, operator-based targets)
    fit = cz.fit_nu_pooled([setups[c.id] for c in train])
    nu = fit["nu_unclipped"]
    nus = {"QL+": float(np.clip(nu, 0.0, NU_CLIP)), "QL±": float(np.clip(nu, -NU_CLIP, NU_CLIP)), "QL*": float(nu)}
    fit["nu_by_closure"] = nus
    fit["nu_star_per_train_case"] = {c.id: cz.fit_nu_pooled([setups[c.id]])["nu_unclipped"] for c in train}
    base = [cz.RRR(), cz.QL(nus["QL+"], "QL+"), cz.QL(nus["QL±"], "QL±"), cz.QL(nus["QL*"], "QL*"), cz.Oracle()]
    per_case: Dict[str, dict] = {c.id: {} for c in all_cases}
    for case in all_cases:
        S = setups[case.id]
        a = aux[case.id]
        for cl in base:
            r = cz.rollout(S, cl)
            per_case[case.id][cl.name] = score_rollout(case, S, grid, fg, r, cl, interior, D2m, a["sig_true"],
                                                       a["f_ref"], a["phi_rows"], a["bs_rows"])
    # ---- refinement band u: RRR resolved-interior error at n_k=256 vs 512 (same keep)
    u_cases = {}
    if u_refine:
        g512 = op.build_grid(n_k=512)
        fg512 = reference.fine_grid(g512, sub=sub)
        int512 = metrics.interior_masks(g512)
        for cid in cases.REFINEMENT_CASES:
            case = cases.get_case(cid)
            S512 = cz.build_setup(case, g512, fg512, variant, keep)
            r = cz.rollout(S512, cz.RRR())
            err_rows = (r["c"][S512.out_rel] - S512.c_ref[S512.out_rel]) @ S512.V.T
            e512 = metrics.rms(err_rows, int512) if not r["unstable"] else float("nan")
            e256 = per_case[cid]["RRR"]["resolved_interior_rms"]
            u_cases[cid] = {"rrr_256": e256, "rrr_512": e512, "u": abs(e256 - e512)}
    u = max(v["u"] for v in u_cases.values()) if u_cases else float("nan")
    # ---- oracle-gap criterion on the training split
    gaps = {c.id: per_case[c.id]["RRR"]["resolved_interior_rms"] - per_case[c.id]["ORACLE"]["resolved_interior_rms"] for c in train}
    gap_reproducible = bool(np.all([g > u for g in gaps.values()])) if np.isfinite(u) else False
    block = {"variant": variant, "keep": keep, "n_k": n_k, "sub": sub, "dim_R": int(setups[train[0].id].V.shape[1]),
             "fit": fit, "refinement_band": {"u": u, "per_case": u_cases},
             "oracle_gap_train": gaps, "gap_reproducible": gap_reproducible, "ridge_selection": {}}
    # ---- surrogates
    if gap_reproducible:
        train_setups = [setups[c.id] for c in train]
        models = {}
        for name in ("LM", "LR", "ESN"):
            best = None
            for lam in RIDGE_GRID:
                if name == "LM":
                    model = cz.fit_linear_markov(train_setups, lam)
                else:
                    model = cz.fit_reservoir(train_setups, lam, nonlinear=(name == "ESN"), **RES_KW)
                errs = []
                for case in val:
                    r = cz.rollout(setups[case.id], model)
                    if r["unstable"]:
                        errs.append(np.inf)
                    else:
                        S = setups[case.id]
                        err_rows = (r["c"][S.out_rel] - S.c_ref[S.out_rel]) @ S.V.T
                        errs.append(metrics.rms(err_rows, interior))
                score = float(np.mean(errs))
                block["ridge_selection"].setdefault(name, {})[str(lam)] = score
                if best is None or score < best[0]:
                    best = (score, lam, model)
            block["ridge_selection"][name]["chosen"] = best[1]
            models[name] = best[2]
        for name, model in models.items():
            for case in all_cases:
                S = setups[case.id]
                a = aux[case.id]
                r = cz.rollout(S, model)
                per_case[case.id][name] = score_rollout(case, S, grid, fg, r, model, interior, D2m, a["sig_true"],
                                                         a["f_ref"], a["phi_rows"], a["bs_rows"])
            # persist weights (compact fixtures) with hashes
            os.makedirs(FIXTURES, exist_ok=True)
            fx = os.path.join(FIXTURES, f"gate3_{variant}_keep{keep}_{name}.npz")
            if name == "LM":
                np.savez_compressed(fx, C=model.C.astype(np.float32), bias=model.bias.astype(np.float32),
                                    in_scale=model.in_scale, out_scale=model.out_scale)
            else:
                np.savez_compressed(fx, W=model.W.astype(np.float32), in_scale=model.in_scale, out_scale=model.out_scale,
                                    seed=model.seed, n_res=model.n_res, rho=model.rho, leak=model.leak,
                                    nonlinear=model.nonlinear, keep_dim=model.B.shape[1])
            block.setdefault("fixtures", {})[name] = {"path": os.path.relpath(fx, checkpoint.OUTPUT_ROOT),
                                                      "sha256": checkpoint.sha256_file(fx)}
    block["per_case"] = per_case
    block["splits"] = {c.id: c.split for c in all_cases}
    block["wall_seconds"] = time.time() - t_block
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_k", type=int, default=256)
    ap.add_argument("--sub", type=int, default=16)
    ap.add_argument("--variants", default="A,B")
    ap.add_argument("--keeps", default="16,32,64")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--no-refine", action="store_true")
    args = ap.parse_args()
    cmd = "python3 research/dupire_signed_memory/scripts/gate3_compare.py --resume"
    keeps = [int(k) for k in args.keeps.split(",")]
    variants = args.variants.split(",")
    done, failed, pending = [], [], []
    computed = 0
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "blocks"), exist_ok=True)
    for variant in variants:
        for keep in keeps:
            bid = f"{variant}_keep{keep}"
            path = os.path.join(OUT, "blocks", bid + ".json")
            cfg_hash = checkpoint.hashlib.sha256(f"{bid}|{args.n_k}|{args.sub}|{NU_CLIP}|{RIDGE_GRID}|{RES_KW}".encode()).hexdigest()[:16]
            if args.resume and checkpoint.result_is_done(path, cfg_hash):
                done.append(bid)
                continue
            if args.max_blocks and computed >= args.max_blocks:
                pending.append(bid)
                continue
            try:
                block = run_block(variant, keep, args.n_k, args.sub, u_refine=not args.no_refine)
                block["status"] = "done"
                block["config_hash"] = cfg_hash
                checkpoint.write_json(path, block)
                done.append(bid)
                computed += 1
                pc = block["per_case"]
                def mean_split(cl, split):
                    vals = [pc[c]["resolved_interior_rms"] for c in pc if block["splits"][c] == split and cl in pc[c]]
                    vals = [v for v in vals if np.isfinite(v)]
                    return float(np.mean(vals)) if vals else float("nan")
                names = [n for n in ("RRR", "QL+", "QL±", "QL*", "ORACLE", "LM", "LR", "ESN") if n in pc[next(iter(pc))]]
                print(f"[{bid}] {block['wall_seconds']:.0f}s nu*={block['fit']['nu_unclipped']:+.3e} u={block['refinement_band']['u']:.2e} gap={block['gap_reproducible']} | "
                      + " ".join(f"{n}:{mean_split(n,'train'):.2e}/{mean_split(n,'test'):.2e}" for n in names), flush=True)
            except Exception:
                tb = traceback.format_exc()
                checkpoint.write_json(path, {"block": bid, "status": "failed", "config_hash": cfg_hash, "traceback": tb})
                failed.append(bid)
                print(f"[{bid}] FAILED\n{tb}", flush=True)
    checkpoint.write_json(os.path.join(OUT, "summary.json"), {"done": done, "failed": failed, "pending": pending,
                                                              "wall_seconds": time.time() - t0, "config": vars(args)})
    exit_code = 0 if not failed else 1
    checkpoint.save_checkpoint("gate3", command=cmd, exit_code=exit_code, done=done, failed=failed, pending=pending,
                               seeds={"reservoir_seed": RES_KW["seed"], "family_S": list(cases.SPLIT_S.keys())},
                               config={**vars(args), "nu_clip": NU_CLIP, "ridge_grid": RIDGE_GRID, "reservoir": RES_KW},
                               next_command=(cmd if (pending or failed) else
                                             "python3 research/dupire_signed_memory/scripts/gate4_evaluate.py"))
    print(f"gate3: done={len(done)} failed={len(failed)} pending={len(pending)} in {time.time()-t0:.0f}s")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
