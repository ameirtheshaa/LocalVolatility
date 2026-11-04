#!/usr/bin/env python3
"""Gate 5 — practical test of a small boundary-adapted reduced model.

Compares, on a constant-volatility control, a held-out strike-dependent smile and one fresh
time- and strike-dependent smile (defined only here, never used before):

  A_fourier_m{33,65}  production Fourier model (one-sided stencil, raw price), resolved-only
  B_fourier_m{33,65}  boundary-consistent Fourier model (linearity rows, detrended), resolved-only
  B_sine_m{17,33,65}  boundary-adapted model: Dirichlet sine modes on the detrended price, Galerkin
  A_full256, B_full256  un-reduced semidiscrete solves (cost/accuracy baselines)

against the refined full calculation B_full512 (and Black–Scholes for the control), with two
start protocols: from the projected reference state at t̃0 = 0.2 (as in Gate 3) and from the
payoff at t̃ = 0 (deployment). Metrics: price error, density error, negative density mass,
inverse-volatility validity, stability, wall-clock cost. Numerical uncertainty is the
256-vs-512 grid difference (and the step refinement for the time-dependent case); an
improvement is claimed only when it exceeds it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsm import cases, checkpoint, metrics, mz, operators as op, reference, repo  # noqa: E402

OUT = os.path.join(checkpoint.RESULTS, "gate5")
ALGO_VERSION = 1
FRESH = cases.Case(id="F_fresh_base0.28_t0.2_x0.05", family="F", params={"sigma_base": 0.28, "t_shift": 0.2, "x_shift": 0.05},
                   split="fresh", time_dependent=True, sigma_fn=cases._smile_D(0.28, t_shift=0.2, x_shift=0.05))
CASE_IDS = ["C_sig0.30", "S_seed9", FRESH.id]


def get_case(cid):
    return FRESH if cid == FRESH.id else cases.get_case(cid)


def sine_basis(n, m):
    i = np.arange(n)
    V, _ = np.linalg.qr(np.stack([np.sin(np.pi * j * i / (n - 1)) for j in range(1, m + 1)], axis=1))
    return V


def rollout(L_of_t, V, x0, t, constant):
    """Galerkin rollout c_{n+1} = exp(h Vᵀ L(t_mid) V) c_n; returns (c, seconds, abscissa)."""
    m = V.shape[1]
    c = np.empty((len(t), m))
    c[0] = V.T @ x0
    h = t[1] - t[0]
    tic = time.perf_counter()
    if constant:
        Lr = V.T @ L_of_t(t[0]) @ V
        E = expm(h * Lr)
        for n in range(len(t) - 1):
            c[n + 1] = E @ c[n]
    else:
        Lr = None
        for n in range(len(t) - 1):
            Lr = V.T @ L_of_t(0.5 * (t[n] + t[n + 1])) @ V
            c[n + 1] = expm(h * Lr) @ c[n]
    sec = time.perf_counter() - tic
    return c, sec, mz.spectral_abscissa(Lr)


def full_rollout(L_of_t, x0, t, constant):
    x = np.empty((len(t), len(x0)))
    x[0] = x0
    h = t[1] - t[0]
    tic = time.perf_counter()
    if constant:
        E = expm(h * L_of_t(t[0]))
        for n in range(len(t) - 1):
            x[n + 1] = E @ x[n]
    else:
        for n in range(len(t) - 1):
            x[n + 1] = expm(h * L_of_t(0.5 * (t[n] + t[n + 1]))) @ x[n]
    return x, time.perf_counter() - tic


def interp_rows(k_src, rows, k_dst):
    return np.array([CubicSpline(k_src, r)(k_dst) for r in rows])


def score(phi_rows, phi_t_rows, ref_rows, f_ref, grid, interior, D2m, sig_true, bs_rows=None):
    out = {"price_interior_rms": metrics.rms(phi_rows - ref_rows, interior), "price_full_rms": metrics.rms(phi_rows - ref_rows)}
    if bs_rows is not None:
        out["price_interior_rms_vs_bs"] = metrics.rms(phi_rows - bs_rows, interior)
    f = metrics.density_rows(phi_rows, grid, D2m)
    out.update(metrics.density_metrics(f, f_ref, grid, interior))
    out.update(metrics.validity_metrics(phi_rows, phi_t_rows, grid, D2m, interior, sig_true))
    return out


def run_case(cid, sub, dims_fourier=(16, 32), dims_sine=(17, 33, 65), repeats=3):
    case = get_case(cid)
    g = op.build_grid(n_k=256)
    g512 = op.build_grid(n_k=512)
    fg = reference.fine_grid(g, sub=sub)
    fg512 = reference.fine_grid(g512, sub=sub)
    interior = metrics.interior_masks(g)
    D2m = op.d2_repo(g.n_k, g.dk)
    sig_true = metrics.sigma_true_rows(case, g)
    t = fg.t
    i0 = fg.i0
    out_idx = fg.out_idx
    res = {"case": cid, "family": case.family, "params": case.params, "sub": sub, "models": {}, "uncertainty": {}}
    # ---- references
    LB = reference.operator_builder(case, g, "B")
    LA = reference.operator_builder(case, g, "A")
    LB512 = reference.operator_builder(case, g512, "B")
    phiB, secB = full_rollout(LB, g.payoff(), t, not case.time_dependent)
    phiA, secA = full_rollout(LA, g.payoff(), t, not case.time_dependent)
    phi512, sec512 = full_rollout(LB512, g512.payoff(), fg512.t, not case.time_dependent)
    ref_rows = interp_rows(g512.k, phi512[fg512.out_idx], g.k)
    ref_t_rows = interp_rows(g512.k, metrics.time_derivative_rows(phi512, fg512.out_idx, fg512.h), g.k)
    f_ref = interp_rows(g512.k, metrics.density_rows(phi512[fg512.out_idx], g512, op.d2_repo(512, g512.dk)), g.k)
    bs_rows = reference.bs_truth(g, case.params["sigma"], g.t_out) if case.family == "C" else None
    res["uncertainty"]["grid_256_vs_512_interior_rms"] = metrics.rms(phiB[out_idx] - ref_rows, interior)
    res["uncertainty"]["grid_256_vs_512_density_l2"] = metrics.density_metrics(metrics.density_rows(phiB[out_idx], g, D2m), f_ref, g, interior)["density_l2_interior_mean"]
    if case.time_dependent:
        fg32 = reference.fine_grid(g, sub=2 * sub)
        phi32, _ = full_rollout(LB, g.payoff(), fg32.t, False)
        res["uncertainty"]["step_sub16_vs_sub32_interior_rms"] = metrics.rms(phiB[out_idx] - phi32[fg32.out_idx], interior)
    if bs_rows is not None:
        res["uncertainty"]["full512_vs_bs_interior_rms"] = metrics.rms(ref_rows - bs_rows, interior)
    res["models"]["B_full512_reference"] = {"dof": 512, "wall_seconds_rollout": sec512}
    for name, phi, sec, n in (("B_full256", phiB, secB, 256), ("A_full256", phiA, secA, 256)):
        m = score(phi[out_idx], metrics.time_derivative_rows(phi, out_idx, fg.h), ref_rows, f_ref, g, interior, D2m, sig_true, bs_rows)
        m.update({"dof": n, "wall_seconds_rollout": sec, "unstable": not bool(np.all(np.isfinite(phi))), "start": "payoff_t0"})
        res["models"][name] = m
    # ---- reduced models, two start protocols
    ellB = op.linear_trend(g.k, phiB[0][0], phiB[0][-1])
    psiB = phiB - ellB[None, :]
    protocols = {"from_projected_t0.2": (t[i0:], i0), "from_payoff_t0": (t, 0)}
    specs = []
    for keep in dims_fourier:
        specs.append((f"A_fourier_m{2*keep+1}", "A", op.fourier_basis(256, keep)))
        specs.append((f"B_fourier_m{2*keep+1}", "B", op.fourier_basis(256, keep)))
    for m in dims_sine:
        specs.append((f"B_sine_m{m}", "B", sine_basis(256, m)))
    for name, variant, V in specs:
        for pname, (tt, start) in protocols.items():
            L_of_t = LA if variant == "A" else LB
            x_src = phiA if variant == "A" else psiB
            secs = []
            for _ in range(repeats):
                c, sec, absc = rollout(L_of_t, V, x_src[start], tt, not case.time_dependent)
                secs.append(sec)
            rows_rel = out_idx - start
            x_rows = c[rows_rel] @ V.T
            x_fine = c @ V.T
            if variant == "B":
                x_rows = x_rows + ellB[None, :]
                x_fine = x_fine + ellB[None, :]
            stable = bool(np.all(np.isfinite(c)))
            m = score(x_rows, metrics.time_derivative_rows(x_fine, rows_rel, fg.h), ref_rows, f_ref, g, interior, D2m, sig_true, bs_rows) if stable else {}
            proj = metrics.rms(((x_src[out_idx] @ V) @ V.T) - x_src[out_idx], interior)
            m.update({"dof": int(V.shape[1]), "wall_seconds_rollout_median": float(np.median(secs)), "unstable": not stable,
                      "reduced_operator_abscissa": absc, "projection_err_interior_rms": proj, "start": pname})
            res["models"][f"{name}__{pname}"] = m
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    manifest = checkpoint.RunManifest("gate5", CASE_IDS)
    done, failed = [], []
    t0 = time.time()
    for cid in CASE_IDS:
        case = get_case(cid)
        path = os.path.join(OUT, cid + ".json")
        cfg = repo.DupirePipelineConfig()
        ident = checkpoint.make_identity(gate="gate5", algo=ALGO_VERSION, case=case.config_hash(), case_id=cid, sub=args.sub,
                                         n_k=256, n_k_ref=512, K_min=float(cfg.K_min), K_max=float(cfg.K_max),
                                         dims_fourier=[16, 32], dims_sine=[17, 33, 65])
        status, detail = checkpoint.result_status(path, ident)
        if args.resume and status == "done":
            done.append(cid)
            continue
        if status in ("stale", "corrupted", "failed", "incomplete"):
            checkpoint.supersede(path)
        manifest.start(cid)
        try:
            t1 = time.time()
            res = run_case(cid, args.sub)
            res.update({"status": "done", "identity": ident, "identity_hash": checkpoint.identity_hash(ident), "wall_seconds": time.time() - t1})
            checkpoint.write_json(path, res)
            manifest.done(cid)
            done.append(cid)
            u = res["uncertainty"]
            print(f"[{cid}] {res['wall_seconds']:.0f}s u_grid={u['grid_256_vs_512_interior_rms']:.2e}", flush=True)
            for k, v in res["models"].items():
                if "price_interior_rms" in v:
                    print(f"   {k:40s} dof={v['dof']:4d} price={v['price_interior_rms']:.2e} dens={v['density_l2_interior_mean']:.2e} negmass={v['negative_mass_max']:.2e} invvol={v['inverse_vol_invalid_frac_mean']:.3f} t={v.get('wall_seconds_rollout_median', v.get('wall_seconds_rollout')):.3f}s absc={v.get('reduced_operator_abscissa', float('nan')):+.1e}", flush=True)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            checkpoint.write_json(path, {"case": cid, "status": "failed", "identity": ident, "traceback": tb})
            manifest.failed(cid)
            failed.append(cid)
            print(f"[{cid}] FAILED\n{tb}", flush=True)
    manifest.finish()
    checkpoint.write_json(os.path.join(OUT, "summary.json"), {"done": done, "failed": failed, "wall_seconds": time.time() - t0,
                                                              "fresh_case": {"id": FRESH.id, "params": FRESH.params, "config_hash": FRESH.config_hash()}})
    checkpoint.save_checkpoint("gate5", command="python3 research/dupire_signed_memory/scripts/gate5_sine_rom.py --resume",
                               exit_code=0 if not failed else 1, done=done, failed=failed, pending=[], seeds={},
                               config={**vars(args), "algo_version": ALGO_VERSION, "code_identity": checkpoint.numerical_code_identity()},
                               next_command="python3 research/dupire_signed_memory/scripts/gate5_table.py")
    print(f"gate5: done={len(done)} failed={len(failed)} in {time.time()-t0:.0f}s")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
