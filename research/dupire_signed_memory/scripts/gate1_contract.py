#!/usr/bin/env python3
"""Gate 1 — mathematical contract: analytic control + exact-reconstruction identities on the
actual discrete Dupire operator (OP-A production, OP-B boundary-consistent, OP-C Dirichlet-forced)
and the sign-theorem numerics. Writes results/gate1/gate1_contract.json and a checkpoint."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsm import cases, checkpoint, control, mz, operators as op, reference, repo  # noqa: E402


def reconstruct_constant_L(L, V, psi, t, i0, forcing_full=None):
    """Exact-reconstruction identities for constant L from index i0 onward.

    Returns dict of max errors of (1) unresolved variation-of-constants vs P_U ψ,
    (2) resolved equation integrated with the reconstructed feedback vs Vᵀψ, and the
    separate magnitudes of the unresolved-IC term, the memory convolution and boundary pieces.
    """
    B = mz.resolved_blocks(L, V)
    P_U, L_UU, L_UR, L_RU, L_RR = B["P_U"], B["L_UU"], B["L_UR"], B["L_RU"], B["L_RR"]
    h = float(t[1] - t[0])
    tt = t[i0:]
    c_ref = psi[i0:] @ V                     # resolved coefficients along the reference
    u_ref = psi[i0:] @ P_U.T                 # unresolved part along the reference
    E_U, Phi_U = mz.block_phi1(L_UU, h)
    E_R, Phi_R = mz.block_phi1(L_RR, h)
    # forcing on the unresolved equation: g(s) = L_UR c(s) + P_U b(s)
    g = c_ref @ L_UR.T
    bU = bR = None
    if forcing_full is not None:
        bU = forcing_full[i0:] @ P_U.T
        bR = forcing_full[i0:] @ V
        g = g + bU
    # (1) u(t) = e^{(t-t0)L_UU} u0 + ∫ e^{(t-s)L_UU} g(s) ds  (midpoint rule for g)
    u_ic = np.empty_like(u_ref)
    u_mem = np.zeros_like(u_ref)
    u_ic[0] = u_ref[0]
    for n in range(len(tt) - 1):
        u_ic[n + 1] = E_U @ u_ic[n]
        u_mem[n + 1] = E_U @ u_mem[n] + Phi_U @ (0.5 * (g[n] + g[n + 1]))
    u_rec = u_ic + u_mem
    err_u = np.linalg.norm(u_rec - u_ref, axis=1)
    # (2) c' = L_RR c + L_RU u_rec(t) + Vᵀ b(t), integrated with the same exponential midpoint rule
    feedback = u_rec @ L_RU.T
    if bR is not None:
        feedback = feedback + bR
    c = np.empty_like(c_ref)
    c[0] = c_ref[0]
    for n in range(len(tt) - 1):
        c[n + 1] = E_R @ c[n] + Phi_R @ (0.5 * (feedback[n] + feedback[n + 1]))
    err_c = np.linalg.norm(c - c_ref, axis=1)
    ic_term = u_ic @ L_RU.T              # L_RU e^{(t-t0)L_UU} u0  (unresolved-IC feedback)
    mem_term = u_mem @ L_RU.T            # ∫ K(t-s) c(s) ds (+ boundary routed through U)
    out = {
        "max_rel_err_unresolved_reconstruction": float(err_u.max() / (np.linalg.norm(u_ref, axis=1).max() + 1e-300)),
        "max_rel_err_resolved_reconstruction": float(err_c.max() / (np.linalg.norm(c_ref, axis=1).max() + 1e-300)),
        "final_rel_err_resolved": float(err_c[-1] / (np.linalg.norm(c_ref[-1]) + 1e-300)),
        "norm_unresolved_ic_feedback_t0": float(np.linalg.norm(ic_term[1])),
        "norm_unresolved_ic_feedback_end": float(np.linalg.norm(ic_term[-1])),
        "norm_memory_feedback_end": float(np.linalg.norm(mem_term[-1])),
        "norm_markovian_term_end": float(np.linalg.norm(L_RR @ c_ref[-1])),
        "norm_total_feedback_end": float(np.linalg.norm(feedback[-1])),
        "finite": bool(np.all(np.isfinite(c)) and np.all(np.isfinite(u_rec))),
    }
    if bR is not None:
        out["norm_resolved_boundary_forcing_end"] = float(np.linalg.norm(bR[-1]))
        out["norm_unresolved_boundary_forcing_end"] = float(np.linalg.norm(bU[-1]))
    return out, B


def sign_theorem_numerics(L_full: np.ndarray, D2: np.ndarray, dvec: np.ndarray, V: np.ndarray, taus):
    """Interior Dirichlet system: W-self-adjoint negative L_int. Compare the kernel's quadratic
    form for (a) the W-orthogonal projector on the Fourier subspace and (b) the Euclidean one."""
    L_int = L_full[1:-1, 1:-1]
    Vi = V[1:-1, :]
    Vi, _ = np.linalg.qr(Vi)                      # Euclidean-orthonormal basis of the restricted subspace
    w = 1.0 / dvec[1:-1]                          # W = diag(1/(η k²))
    S = np.sqrt(w)                                # similarity: L̃ = S L S⁻¹ symmetric
    Lt = S[:, None] * L_int / S[None, :]
    sym_defect = float(np.abs(Lt - Lt.T).max() / np.abs(Lt).max())
    # (a) W-orthogonal projector onto span(Vi): in S-coordinates the subspace is span(S Vi)
    Vw, _ = np.linalg.qr(S[:, None] * Vi)
    Bw = mz.resolved_blocks(0.5 * (Lt + Lt.T), Vw)
    # (b) Euclidean Fourier projector on the interior system (as the repository does)
    Be = mz.resolved_blocks(L_int, Vi)
    rows = []
    for tau in taus:
        Kw = Bw["L_RU"] @ expm(tau * Bw["L_UU"]) @ Bw["L_UR"]
        ew = np.linalg.eigvalsh(0.5 * (Kw + Kw.T))
        Ke = Be["L_RU"] @ expm(tau * Be["L_UU"]) @ Be["L_UR"]
        ee = np.linalg.eigvalsh(0.5 * (Ke + Ke.T))
        rows.append({"tau": float(tau),
                     "W_orthogonal": {"min_eig_symK": float(ew.min()), "max_eig_symK": float(ew.max())},
                     "euclidean_fourier": {"min_eig_symK": float(ee.min()), "max_eig_symK": float(ee.max())}})
    return {"symmetry_defect_of_similarity_transformed_L": sym_defect,
            "L_UU_abscissa_W_orthogonal": mz.spectral_abscissa(Bw["L_UU"]),
            "L_UU_abscissa_euclidean": mz.spectral_abscissa(Be["L_UU"]),
            "kernel_quadratic_form": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_k", type=int, default=256)
    ap.add_argument("--keep", type=int, default=32)
    ap.add_argument("--sub", type=int, default=16)
    args = ap.parse_args()
    t_start = time.time()
    cmd = "python3 research/dupire_signed_memory/scripts/gate1_contract.py" + \
        (f" --n_k {args.n_k} --keep {args.keep} --sub {args.sub}")
    res = {"config": vars(args), "control": control.check()}

    grid = op.build_grid(n_k=args.n_k)
    fg = reference.fine_grid(grid, sub=args.sub)
    V = op.fourier_basis(grid.n_k, args.keep)
    case = cases.get_case("C_sig0.30")
    sigma = case.params["sigma"]

    # ---------------- OP-B: boundary-consistent, full horizon, nonzero unresolved IC
    LB = reference.operator_builder(case, grid, "B")(0.0)
    phiB = reference.solve_full(case, grid, "B", fg)
    psiB = reference.state_variable(phiB, grid, "B")
    recB, BB = reconstruct_constant_L(LB, V, psiB, fg.t, fg.i0)
    recB["abscissa_L_UU"] = mz.spectral_abscissa(BB["L_UU"])
    recB["abscissa_L"] = mz.spectral_abscissa(LB)
    recB["numerical_abscissa_L"] = mz.numerical_abscissa(LB)
    taus = [0.0, 1e-3, 1e-2, 0.1, 0.4, 0.8]
    recB["kernel_norms"] = dict(zip([str(t) for t in taus], mz.kernel_norms(BB, np.array(taus)).tolist()))
    recB["kernel_sym_eigrange"] = {str(t): mz.kernel_quadratic_range(BB, t) for t in taus[1:]}
    res["OP_B"] = recB

    # ---------------- OP-A: production stencil; identity only over short horizons
    LA = reference.operator_builder(case, grid, "A")(0.0)
    phiA = reference.solve_full(case, grid, "A", fg)
    BA = mz.resolved_blocks(LA, V)
    recA = {"abscissa_L_UU": mz.spectral_abscissa(BA["L_UU"]), "abscissa_L": mz.spectral_abscissa(LA),
            "numerical_abscissa_L": mz.numerical_abscissa(LA), "horizons": {}}
    for n_steps in (2, 5, 10, 20, 40, 80, 160, 368):
        i1 = fg.i0 + n_steps
        r, _ = reconstruct_constant_L(LA, V, phiA[: i1 + 1], fg.t[: i1 + 1], fg.i0)
        recA["horizons"][str(n_steps)] = {k: r[k] for k in ("max_rel_err_resolved_reconstruction",
                                                           "max_rel_err_unresolved_reconstruction", "finite")}
    recA["memory_vs_markovian_ratio_along_reference"] = float(np.median(
        np.linalg.norm(mz.memory_term(LA, V, phiA[fg.i0:]), axis=1) /
        np.linalg.norm((phiA[fg.i0:] @ V) @ BA["L_RR"].T, axis=1)))
    res["OP_A"] = recA

    # ---------------- OP-C: Dirichlet-forced with Black–Scholes boundary data (boundary forcing)
    t = fg.t
    gl = np.array([reference.bs_truth(grid, sigma, np.array([ti]))[0][0] for ti in t])
    gh = np.array([reference.bs_truth(grid, sigma, np.array([ti]))[0][-1] for ti in t])
    # time derivatives of the boundary data by fine central differences (step h/64)
    eps = fg.h / 64
    def bs_bnd(ti):
        row = reference.bs_truth(grid, sigma, np.array([max(ti, 0.0)]))[0]
        return row[0], row[-1]
    dgl = np.empty_like(t); dgh = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti - eps < 0:
            a0, b0 = bs_bnd(ti); a1, b1 = bs_bnd(ti + eps)
            dgl[i], dgh[i] = (a1 - a0) / eps, (b1 - b0) / eps
        else:
            a0, b0 = bs_bnd(ti - eps); a1, b1 = bs_bnd(ti + eps)
            dgl[i], dgh[i] = (a1 - a0) / (2 * eps), (b1 - b0) / (2 * eps)
    forcing = np.zeros((len(t), grid.n_k))
    forcing[:, 0] = dgl
    forcing[:, -1] = dgh
    # full forced solve: φ' = L_B φ + b(t), midpoint rule on b
    E_B, Phi_B = mz.block_phi1(LB, fg.h)
    phiC = np.empty_like(phiA)
    phiC[0] = grid.payoff()
    for n in range(len(t) - 1):
        phiC[n + 1] = E_B @ phiC[n] + Phi_B @ (0.5 * (forcing[n] + forcing[n + 1]))
    bs = reference.bs_truth(grid, sigma, t[fg.out_idx])
    interior = np.array([repo.interior_k_mask(grid.K_row(T)) for T in grid.T_out])
    def rms(m, mask):
        return float(np.sqrt((m ** 2)[mask].mean()))
    recC_acc = {
        "boundary_values_match_bs_max": float(max(np.abs(phiC[:, 0] - gl).max(), np.abs(phiC[:, -1] - gh).max())),
        "interior_rms_vs_bs_OP_C": rms(phiC[fg.out_idx] - bs, interior),
        "interior_rms_vs_bs_OP_B": rms(phiB[fg.out_idx] - bs, interior),
        "interior_rms_vs_bs_OP_A": rms(phiA[fg.out_idx] - bs, interior),
        "full_rms_vs_bs_OP_C": rms(phiC[fg.out_idx] - bs, np.ones_like(interior)),
        "full_rms_vs_bs_OP_B": rms(phiB[fg.out_idx] - bs, np.ones_like(interior)),
        "full_rms_vs_bs_OP_A": rms(phiA[fg.out_idx] - bs, np.ones_like(interior)),
    }
    # the projector acts on the detrended variable; the trend is now time dependent, so the
    # forcing seen by ψ is b - ℓ̇ + L ℓ (= b - ℓ̇ since L ℓ = 0 for the linearity stencil)
    ell = reference.trend_rows(phiC, grid)
    psiC = phiC - ell
    dell = np.stack([op.linear_trend(grid.k, dgl[i], dgh[i]) for i in range(len(t))])
    forcing_psi = forcing - dell
    assert np.abs(LB @ ell.T).max() < 1e-8 * np.abs(LB).max(), "L ℓ must vanish for the linearity stencil"
    recC, BC = reconstruct_constant_L(LB, V, psiC, t, fg.i0, forcing_full=forcing_psi)
    recC.update(recC_acc)
    recC["abscissa_L_UU"] = mz.spectral_abscissa(BC["L_UU"])
    res["OP_C"] = recC

    # ---------------- sign theorem numerics on OP-B (interior Dirichlet system)
    D2B = op.d2_linearity(grid.n_k, grid.dk)
    dvec = op.eta_row(case.sigma_fn, grid, grid.T_max) * grid.k ** 2
    res["sign_theorem"] = sign_theorem_numerics(LB, D2B, dvec, V, [1e-3, 1e-2, 0.1, 0.4])

    # ---------------- eigenbasis projector (interior Dirichlet system): no memory for constant L
    L_int = LB[1:-1, 1:-1]
    w = 1.0 / dvec[1:-1]
    S = np.sqrt(w)
    Lt = 0.5 * ((S[:, None] * L_int / S[None, :]) + (S[:, None] * L_int / S[None, :]).T)
    ev, Vt = np.linalg.eigh(Lt)
    order = np.argsort(np.abs(ev))
    Ve = Vt[:, order[: 2 * args.keep + 1]]
    Beig = mz.resolved_blocks(Lt, Ve)
    res["eigenbasis"] = {"norm_L_RU": float(np.linalg.norm(Beig["L_RU"])), "norm_L_RR": float(np.linalg.norm(Beig["L_RR"])),
                         "note": "L_RU = 0: Galerkin in the weighted eigenbasis is exact for time-independent L"}

    res["wall_seconds"] = time.time() - t_start
    out = os.path.join(checkpoint.RESULTS, "gate1", "gate1_contract.json")
    res["status"] = "done"
    res["config_hash"] = case.config_hash()
    sha = checkpoint.write_json(out, res)
    print(f"wrote {out} sha256={sha[:16]} in {res['wall_seconds']:.1f}s")
    for k in ("OP_B", "OP_C"):
        r = res[k]
        print(f"  {k}: rel err unresolved {r['max_rel_err_unresolved_reconstruction']:.2e} resolved {r['max_rel_err_resolved_reconstruction']:.2e} | L_UU abscissa {r.get('abscissa_L_UU', float('nan')):+.3e}")
    print("  OP_A horizons:", {k: f"{v['max_rel_err_resolved_reconstruction']:.1e}" for k, v in res["OP_A"]["horizons"].items()})
    print("  OP_C accuracy:", {k: f"{v:.2e}" for k, v in recC_acc.items()})
    print("  sign theorem:", res["sign_theorem"]["kernel_quadratic_form"][1])
    checkpoint.save_checkpoint(
        "gate1", command=cmd, exit_code=0, done=["control", "OP_B", "OP_A", "OP_C", "sign_theorem"],
        failed=[], pending=[], seeds={}, config=vars(args),
        next_command="python3 research/dupire_signed_memory/scripts/gate2_reference.py --resume")


if __name__ == "__main__":
    main()
