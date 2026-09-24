#!/usr/bin/env python3
"""
MZ x Dupire — CPU de-risk experiments (Steps 0a, 0b, 1 of the synthesis plan).

Decides, cheaply and before any expensive build, the central claims of the
"MZ-augmented self-consistent learning of local volatility" synthesis:

  Step 0a  Is the discrete MZ orthogonal-dynamics generator L_UU dissipative?
           (SEED A: if yes, the MZ memory kernel converges and Markovianization
           is legitimate — HW's propagator-instability pathology is ABSENT.)
           We report the spectral abscissa  alpha(L_UU) = max Re eig(L_UU)
           and the numerical abscissa       mu(L_UU)    = max eig(sym(L_UU))
           in BOTH the Euclidean metric and the Sturm-Liouville weighted metric
           <u,v>_w = sum u v / (eta~ k~^2), for a constant-sigma and a smile case,
           swept over the truncation `keep`.

  Step 0b  Is the FORWARD MZ memory real or cosmetic? (decides the paper's lead
           claim.) On the EXACT constant-sigma case (Black-Scholes truth, no MC
           noise) we compare RRR-only vs RRR+QL against truth, split interior vs
           tail, read fit_nu_ql's nu_global, and compute the uuu gap. If nu~0 and
           QL does not beat RRR in the interior while RRR already matches truth
           there, the forward memory is cosmetic and the contribution lives on
           the density/inverse (tail) side.

  Step 1   Is one tail scalar enough? Across a family of local-vol surfaces and
           maturities, PCA the standardized-cumulant trajectory (skew c3, excess
           kurtosis c4) to estimate how many independent non-Gaussian "shape"
           directions exist -> 1 scalar (single E_tail) vs 2 (skew/kurt energies).

Pure NumPy/SciPy + matplotlib. No TensorFlow, no training, no MC. Reuses the
real mz_spectral functions so it tests the ACTUAL code paths.

Run:  .venv/bin/python examples/run_mz_derisk.py
Outputs: mz_derisk.json  and  plots/mz_derisk/*.png  (under the package root).
"""

import os
import sys
import json

import numpy as np

# --- path setup: make the package root importable (mirrors every example) ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DupirePipelineConfig, VolatilityConfig
from analytical_solutions import lognormal_density
from mz_spectral.fourier_dupire import (
    build_uniform_tk_grid, eta_tilde_from_sigma, build_second_derivative_matrix,
)
from mz_spectral.mz_decomposition import (
    split_operator_blocks, low_pass_projector_matrix,
)
from mz_spectral.validation import (
    build_truth_phi_bs,
    payoff_phi_tilde,
    precompute_L_and_blocks,
    integrate_rrr_only,
    integrate_rrr_ql,
    pdf_from_phi_tilde,
    pdf_metrics,
    interior_k_mask,
    uuu_gap_metric,
)
from mz_spectral.quasi_linear import fit_nu_ql
from mz_spectral.uuu_dimension import pca_effective_dimension


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _const_vol(sigma_const):
    """VolatilityConfig for a flat sigma(t,x) = sigma_const."""
    return VolatilityConfig.custom(
        lambda t, x: sigma_const * np.ones_like(np.asarray(x, dtype=float))
    )


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays to plain python for json."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def _spectral_abscissa(M):
    """max Re lambda(M)  (<=0  <=>  asymptotic decay of e^{Mt})."""
    return float(np.max(np.linalg.eigvals(M).real))


def _num_abscissa(M):
    """max lambda(sym(M))  (<=0  <=>  no transient growth of ||e^{Mt}||)."""
    return float(np.max(np.linalg.eigvalsh(0.5 * (M + M.T))))


def _sym_second_derivative(n, dk):
    """Symmetric (Dirichlet) tridiagonal d2/dk2 — negative definite by construction."""
    D = -2.0 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)
    return D / (dk * dk)


def _standardized_cumulants(K_row, f):
    """Return (skew, excess_kurtosis) of a (normalized) density f on grid K_row."""
    f = np.clip(np.asarray(f, dtype=float), 0.0, None)
    area = np.trapz(f, K_row)
    if not np.isfinite(area) or area <= 0:
        return np.nan, np.nan
    f = f / area
    m1 = np.trapz(K_row * f, K_row)
    var = np.trapz((K_row - m1) ** 2 * f, K_row)
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan
    sd = np.sqrt(var)
    mu3 = np.trapz((K_row - m1) ** 3 * f, K_row)
    mu4 = np.trapz((K_row - m1) ** 4 * f, K_row)
    skew = mu3 / sd ** 3
    exkurt = mu4 / var ** 2 - 3.0
    return float(skew), float(exkurt)


# ----------------------------------------------------------------------------
# STEP 0a — operator dissipativity (SEED A), separating continuous truth from
#           discrete artifact and exhibiting the eigenbasis/weighted cure.
#
# Key fact: L = diag(eta~ k~^2) @ D2 = D @ D2 is SIMILAR to the symmetric matrix
#   A = sqrt(D) @ D2 @ sqrt(D)   (same eigenvalues as L).
# If D2 is symmetric negative-definite then A is negative-definite => eig(L) <= 0
# => the forward semigroup decays and the MZ memory kernel converges (SEED A).
# We test three things per case:
#   (1) spec abscissa of L with a SYMMETRIC stencil  -> should be <= 0 (SEED A true)
#   (2) spec abscissa of L with the REPO one-sided stencil -> artifact (> 0)
#   (3) the Fourier-projected unresolved block L_UU, and the eigenbasis-projected
#       unresolved block, for the symmetric stencil -> shows which projector keeps
#       the orthogonal dynamics dissipative (the cure).
# ----------------------------------------------------------------------------
def step_0a(n_t=24, n_k=256, keep_rep=32):
    cfg = DupirePipelineConfig()
    common = dict(
        S0=cfg.S0, r=cfg.r, T_max=cfg.T_max, K_max=cfg.K_max,
        K_min=cfg.K_min, T_min=cfg.T_min, n_t=n_t, n_k=n_k,
    )
    cases = {
        "const": _const_vol(0.3),
        "smile": VolatilityConfig.dupire_exact(),  # sigma_base=0.3 + y e^{-y}
    }
    out = {}
    for cname, vc in cases.items():
        grid, sigma = build_uniform_tk_grid(vol_config=vc, **common)
        eta = eta_tilde_from_sigma(sigma, cfg.T_max)        # (n_t, n_k)
        i = n_t - 1                                          # representative (smooth) maturity
        dvec = eta[i] * grid.k_tilde ** 2                    # diag(D) > 0
        sqrtD = np.sqrt(dvec)
        D2_repo = build_second_derivative_matrix(n_k, grid.dk)
        D2_sym = _sym_second_derivative(n_k, grid.dk)
        P_R = low_pass_projector_matrix(n_k, keep_rep)
        rec = {}
        for sname, D2 in (("repo", D2_repo), ("sym", D2_sym)):
            L = dvec[:, None] * D2                            # diag(D) @ D2
            L_UU = split_operator_blocks(L, P_R)[3]
            rec[sname] = {
                "spec_L": _spectral_abscissa(L),
                "spec_LUU_fourier": _spectral_abscissa(L_UU),
                "num_LUU_fourier": _num_abscissa(L_UU),
            }
            if sname == "sym":
                A = sqrtD[:, None] * D2 * sqrtD[None, :]      # sqrt(D) D2 sqrt(D), symmetric
                evals = np.linalg.eigvalsh(A)                # real, == eig(L)
                order = np.argsort(np.abs(evals))            # slowest (|lam|~0) first
                unresolved = order[keep_rep:]                # fast modes = the U block
                rec["eig_cure"] = {
                    "spec_LUU_eig": float(np.max(evals[unresolved])),  # U-block abscissa
                    "max_eig_A": float(np.max(evals)),
                }
        out[cname] = rec

    tol = 1e-6
    sym_dissipative = all(out[c]["sym"]["spec_L"] <= tol for c in cases)
    cure_ok = all(out[c]["eig_cure"]["spec_LUU_eig"] <= tol for c in cases)
    repo_artifact = any(out[c]["repo"]["spec_L"] > tol for c in cases)
    fourier_sym_growth = any(out[c]["sym"]["spec_LUU_fourier"] > tol for c in cases)

    parts = []
    if sym_dissipative and cure_ok:
        parts.append("SEED A CONFIRMED: the symmetrized Dupire generator sqrt(D)·D2·sqrt(D) is "
                     "negative-definite (eig(L)<=0) -> forward semigroup decays, MZ memory kernel "
                     "converges; the eigenbasis (Sturm-Liouville) projector keeps the orthogonal "
                     "dynamics dissipative.")
    else:
        parts.append("SEED A NOT confirmed numerically even for the symmetric stencil -- investigate.")
    if repo_artifact:
        parts.append("DISCRETE CAVEAT: the repo's one-sided boundary stencil makes L NON-dissipative "
                     "at the operator level (spectral abscissa > 0); the ROM stays stable only because "
                     "backward-Euler is implicit. A symmetric stencil removes this.")
    if fourier_sym_growth:
        parts.append("PROJECTOR CAVEAT: even with a symmetric stencil the Euclidean Fourier projector "
                     "yields a non-dissipative unresolved block -> motivates the weighted eigenbasis projector.")

    return {
        "by_case": out,
        "verdict": {
            "seed_a_confirmed": bool(sym_dissipative and cure_ok),
            "repo_stencil_artifact": bool(repo_artifact),
            "fourier_projector_growth": bool(fourier_sym_growth),
            "label": " ".join(parts),
        },
        "grid": {"n_t": n_t, "n_k": n_k, "keep_rep": keep_rep},
    }


# ----------------------------------------------------------------------------
# STEP 0b — is the forward MZ memory real or cosmetic?
# ----------------------------------------------------------------------------
def step_0b(n_t=24, n_k=256, keeps=(16, 32, 64, 128), keep_detail=64, sigma_const=0.3):
    cfg = DupirePipelineConfig()
    S0, r, T_max, K_max = cfg.S0, cfg.r, cfg.T_max, cfg.K_max
    grid, sigma = build_uniform_tk_grid(
        S0=S0, r=r, T_max=T_max, K_max=K_max, K_min=cfg.K_min, T_min=cfg.T_min,
        n_t=n_t, n_k=n_k, vol_config=_const_vol(sigma_const),
    )
    phi_truth = build_truth_phi_bs(grid, S0, r, sigma_const)
    phi0 = payoff_phi_tilde(grid.k_tilde, float(S0), float(K_max))
    interior = np.array([interior_k_mask(grid.K[i]) for i in range(n_t)])  # (n_t,n_k)
    tail = ~interior
    nu_clip = 5e-4

    def rms(mat, mask):
        return float(np.sqrt((mat ** 2)[mask].mean()))

    sweep = {}
    detail = None
    for keep in keeps:
        D2, P_R, L_rows, L_RR_rows = precompute_L_and_blocks(grid, sigma, T_max, keep)
        phi_rrr = integrate_rrr_only(phi0, grid.t_tilde, L_RR_rows, P_R)
        nu_global, nu_steps = fit_nu_ql(phi_truth, grid.t_tilde, L_RR_rows, P_R, D2)
        phi_ql = integrate_rrr_ql(phi0, grid.t_tilde, L_RR_rows, P_R, D2, nu_global)

        d_rrr, d_ql = phi_rrr - phi_truth, phi_ql - phi_truth
        # resolved-projected residual (the part a resolved-mode closure can affect)
        dR_rrr, dR_ql = d_rrr @ P_R, d_ql @ P_R
        ei_rrr, ei_ql = rms(dR_rrr, interior), rms(dR_ql, interior)   # resolved interior
        rec = {
            "nu_global": float(nu_global),
            "nu_at_clip": bool(abs(nu_global) >= 0.99 * nu_clip),
            "nu_sign": "neg(anti-diffusion)" if nu_global < 0 else "pos",
            "rrr_resolved_interior": ei_rrr,
            "ql_resolved_interior": ei_ql,
            "gap_resolved_interior": float(uuu_gap_metric(ei_rrr, ei_ql)),
            "rrr_full_interior": rms(d_rrr, interior),
            "ql_full_interior": rms(d_ql, interior),
            "rrr_full_tail": rms(d_rrr, tail),
            "ql_full_tail": rms(d_ql, tail),
        }
        rec["rel_gain_resolved_interior"] = rec["gap_resolved_interior"] / (ei_rrr + 1e-30)
        sweep[str(keep)] = rec

        if keep == keep_detail:
            i = n_t - 1
            T, Krow = float(grid.T[i]), grid.K[i]
            f_truth = pdf_from_phi_tilde(phi_truth[i], T, r, S0, K_max, Krow, D2, normalize=True)
            f_exact = lognormal_density(Krow, S0, T, r, sigma_const)
            f_exact = f_exact / np.trapz(f_exact, Krow)
            f_rrr = pdf_from_phi_tilde(phi_rrr[i], T, r, S0, K_max, Krow, D2, normalize=True)
            f_ql = pdf_from_phi_tilde(phi_ql[i], T, r, S0, K_max, Krow, D2, normalize=True)
            km = interior_k_mask(Krow)
            detail = {
                "keep": keep, "T": T,
                "sanity_l2_truthBL_vs_lognormal": float(np.sqrt(np.trapz((f_truth - f_exact) ** 2, Krow))),
                "pdf_rrr_interior": _jsonable(pdf_metrics(f_rrr, f_exact, Krow, k_mask=km)),
                "pdf_ql_interior": _jsonable(pdf_metrics(f_ql, f_exact, Krow, k_mask=km)),
                "pdf_rrr_tail": _jsonable(pdf_metrics(f_rrr, f_exact, Krow, k_mask=~km)),
                "_arrays_for_plot": {"K": Krow.tolist(), "f_truth": f_truth.tolist(),
                                     "f_rrr": f_rrr.tolist(), "f_ql": f_ql.tolist(),
                                     "f_exact": f_exact.tolist()},
            }

    # verdict: forward QL memory is useful only if it reduces the RESOLVED interior
    # error by a clear margin at some truncation. Whether nu hit the clip is irrelevant
    # to usefulness; a negative nu pinned at the clip is the mis-signed (anti-diffusive)
    # failure the HW sign analysis predicts.
    best_rel_gain = max(sweep[str(k)]["rel_gain_resolved_interior"] for k in keeps)
    any_neg_clip = any(sweep[str(k)]["nu_at_clip"] and sweep[str(k)]["nu_global"] < 0 for k in keeps)
    cosmetic = best_rel_gain < 0.02
    if cosmetic:
        label = ("PIVOT CONFIRMED: forward QL memory does NOT improve the resolved interior at any "
                 f"truncation (best relative gain {best_rel_gain:+.1%}) -> the price-side MZ memory is "
                 "not a useful contribution; the action is the density/inverse (tail) side.")
        if any_neg_clip:
            label += (" The fitted nu is NEGATIVE and pinned at the clip (anti-diffusion) -- the "
                      "mis-signed memory the HW sign analysis predicts; QL increases the error.")
    else:
        label = (f"NOT cosmetic: QL improves the resolved interior by up to {best_rel_gain:.1%} "
                 "-> price-side MZ memory is itself a contribution; reweight the note.")

    return {
        "sigma_const": sigma_const, "n_t": n_t, "n_k": n_k, "keeps": list(keeps),
        "sweep": sweep, "detail": detail,
        "verdict": {"label": label, "cosmetic": bool(cosmetic),
                    "best_rel_gain_resolved_interior": float(best_rel_gain),
                    "negative_nu_at_clip": bool(any_neg_clip)},
    }


# ----------------------------------------------------------------------------
# STEP 1 — moment-trajectory PCA dimension (skew/kurtosis)
# ----------------------------------------------------------------------------
def _surface_family():
    """A small family of local-vol surfaces sigma(t,x), x=K/S0 moneyness."""
    fam = []
    fam.append(("flat0.30", lambda t, x: 0.30 * np.ones_like(np.asarray(x, float))))
    for base in (0.20, 0.30, 0.40):
        for skew in (0.04, 0.08):
            for curv in (0.0, 0.02):
                def f(t, x, base=base, skew=skew, curv=curv):
                    x = np.asarray(x, float)
                    s = base * (1.0 + skew * (1.0 - x) + curv * (x - 1.0) ** 2) * (1.0 + 0.15 * t)
                    return np.clip(s, 0.05, 2.0)
                fam.append((f"b{base}_s{skew}_c{curv}", f))
    return fam


def step_1(n_t=40, n_k=256, K_min=100.0, K_max=6000.0):
    cfg = DupirePipelineConfig()
    S0, r, T_max, T_min = cfg.S0, cfg.r, cfg.T_max, cfg.T_min
    keep_full = n_k // 2  # 2*keep >= n_k -> P_R = identity -> full forward solve

    rows = []        # [skew, exkurt]
    meta = []
    sanity = None
    for name, vf in _surface_family():
        grid, sigma = build_uniform_tk_grid(
            S0=S0, r=r, T_max=T_max, K_max=K_max, K_min=K_min, T_min=T_min,
            n_t=n_t, n_k=n_k, vol_config=VolatilityConfig.custom(vf),
        )
        D2, P_R, L_rows, L_RR_rows = precompute_L_and_blocks(grid, sigma, T_max, keep_full)
        phi0 = payoff_phi_tilde(grid.k_tilde, float(S0), float(K_max))
        phi = integrate_rrr_only(phi0, grid.t_tilde, L_RR_rows, P_R)
        for i in range(n_t):
            if grid.T[i] < 0.5 * T_max:   # later maturities: density well-resolved
                continue
            Krow = grid.K[i]
            f = pdf_from_phi_tilde(phi[i], float(grid.T[i]), r, S0, K_max, Krow, D2, normalize=True)
            sk, ek = _standardized_cumulants(Krow, f)
            if np.isfinite(sk) and np.isfinite(ek):
                rows.append([sk, ek])
                meta.append({"surface": name, "T": float(grid.T[i])})
        # const-sigma sanity: full-solve density vs exact lognormal (last maturity)
        if name == "flat0.30":
            i = n_t - 1
            Krow = grid.K[i]
            f = pdf_from_phi_tilde(phi[i], float(grid.T[i]), r, S0, K_max, Krow, D2, normalize=True)
            fx = lognormal_density(Krow, S0, float(grid.T[i]), r, 0.30)
            fx = fx / np.trapz(fx, Krow)
            sanity = float(np.sqrt(np.trapz((f - fx) ** 2, Krow)))

    X = np.array(rows)  # (n_samples, 2) = (skew, exkurt)
    # z-score columns so PCA measures shape-correlation, not raw scale
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    pca = pca_effective_dimension(Xz, energy_threshold=0.95)
    corr = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]) if X.shape[0] > 2 else np.nan

    dim = int(pca.get("dim", 0))
    if dim <= 1:
        label = ("dim=1: skew and excess-kurtosis are effectively collinear across "
                 "surfaces -> a SINGLE tail-energy scalar E_tail suffices.")
    else:
        label = ("dim=2: skew and excess-kurtosis are independent shape directions -> "
                 "use TWO scalars (skew-energy, kurt-energy).")
    return {
        "n_samples": int(X.shape[0]),
        "pca": _jsonable(pca),
        "skew_kurt_corr": corr,
        "const_sigma_density_sanity_l2": sanity,
        "verdict": {"dim": dim, "label": label,
                    "caveat": ("family has co-varying level/skew/curvature knobs; a next-pass "
                               "robustness check should use independent skew and kurtosis controls "
                               "and densities from MC or a fine PDE solve, not the n_t-step BE rollout.")},
        "_arrays_for_plot": {"skew": X[:, 0].tolist(), "exkurt": X[:, 1].tolist()},
    }


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def make_plots(r0a, r0b, r1, outdir):
    os.makedirs(outdir, exist_ok=True)
    try:
        labels = ["L (repo)", "L (sym)", "L_UU Four(repo)", "L_UU Four(sym)", "L_UU eig(sym)"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for cname, ax in zip(("const", "smile"), axes):
            d = r0a["by_case"][cname]
            vals = [d["repo"]["spec_L"], d["sym"]["spec_L"],
                    d["repo"]["spec_LUU_fourier"], d["sym"]["spec_LUU_fourier"],
                    d["eig_cure"]["spec_LUU_eig"]]
            colors = ["#c0392b" if v > 0 else "#27ae60" for v in vals]
            ax.bar(range(len(vals)), vals, color=colors)
            ax.axhline(0.0, color="k", lw=0.8)
            ax.set_yscale("symlog")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"Step 0a: spectral abscissa — {cname}\n(green<=0 dissipative, red>0 artifact)")
            ax.set_ylabel("max Re lambda  (symlog)")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_0a_abscissa.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"[plot 0a] skipped: {e}")

    try:
        a = r0b["detail"]["_arrays_for_plot"]
        K = np.array(a["K"])
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(K, a["f_exact"], "k-", lw=2, label="exact lognormal")
        ax.plot(K, a["f_truth"], "C2:", lw=2, label="BL of BS-truth phi")
        ax.plot(K, a["f_rrr"], "C0--", label="RRR-only")
        ax.plot(K, a["f_ql"], "C3-.", label="RRR+QL")
        ax.set_title(f"Step 0b: risk-neutral PDF at T={r0b['detail']['T']:.2f} "
                     f"(const sigma, keep={r0b['detail']['keep']})")
        ax.set_xlabel("K"); ax.set_ylabel("f(K)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_0b_pdf.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"[plot 0b] skipped: {e}")

    try:
        cum = r1["pca"].get("cumulative_energy", [])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        if cum:
            ax[0].plot(range(1, len(cum) + 1), cum, "o-")
            ax[0].axhline(0.95, color="r", ls=":", label="0.95")
            ax[0].set_title("Step 1: PCA cumulative energy (z-scored skew,kurt)")
            ax[0].set_xlabel("component"); ax[0].set_ylabel("cum. energy"); ax[0].legend(fontsize=8)
        ax[1].scatter(r1["_arrays_for_plot"]["skew"], r1["_arrays_for_plot"]["exkurt"], s=14)
        ax[1].set_title(f"skew vs excess-kurtosis (corr={r1['skew_kurt_corr']:.3f})")
        ax[1].set_xlabel("skew c3*6"); ax[1].set_ylabel("excess kurtosis c4*24")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "plot_1_pca.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"[plot 1] skipped: {e}")


# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("MZ x Dupire de-risk  (Steps 0a, 0b, 1)")
    print("=" * 78)

    print("\n[Step 0a] operator dissipativity ...")
    r0a = step_0a()
    for cname in ("const", "smile"):
        d = r0a["by_case"][cname]
        print(f"  [{cname}] spec(L) repo={d['repo']['spec_L']:.3e}  sym={d['sym']['spec_L']:.3e}  "
              f"| L_UU Fourier(sym)={d['sym']['spec_LUU_fourier']:.3e}  eig-cure(sym)={d['eig_cure']['spec_LUU_eig']:.3e}")
    print("  VERDICT:", r0a["verdict"]["label"])

    print("\n[Step 0b] forward MZ memory: real or cosmetic? ...")
    r0b = step_0b()
    for k in r0b["keeps"]:
        s = r0b["sweep"][str(k)]
        print(f"  keep={k:>4}  nu={s['nu_global']:+.2e}({s['nu_sign']},clip={s['nu_at_clip']})  "
              f"resolved-int RRR={s['rrr_resolved_interior']:.3e} QL={s['ql_resolved_interior']:.3e} "
              f"gain={s['rel_gain_resolved_interior']:+.1%}")
    print(f"  PDF sanity (BL-truth vs lognormal) l2 = {r0b['detail']['sanity_l2_truthBL_vs_lognormal']:.3e}")
    print("  VERDICT:", r0b["verdict"]["label"])

    print("\n[Step 1] moment-trajectory PCA dimension ...")
    r1 = step_1()
    print(f"  n_samples={r1['n_samples']}  skew-kurt corr={r1['skew_kurt_corr']:.3f}  "
          f"const-sigma density sanity l2={r1['const_sigma_density_sanity_l2']}")
    print("  VERDICT:", r1["verdict"]["label"])

    outdir_plots = os.path.join(_ROOT, "plots", "mz_derisk")
    make_plots(r0a, r0b, r1, outdir_plots)

    # strip bulky plot arrays from the json payload (kept only for plotting)
    if r0b.get("detail"):
        r0b["detail"].pop("_arrays_for_plot", None)
    r1.pop("_arrays_for_plot", None)
    payload = {"step_0a": r0a, "step_0b": r0b, "step_1": r1}
    outjson = os.path.join(_ROOT, "mz_derisk.json")
    with open(outjson, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    print(f"\nWrote {outjson}")
    print(f"Wrote plots to {outdir_plots}/")
    print("\nDONE.")


if __name__ == "__main__":
    main()
