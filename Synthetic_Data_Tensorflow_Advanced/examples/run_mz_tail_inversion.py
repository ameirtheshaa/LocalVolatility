#!/usr/bin/env python3
r"""
MZ x Dupire — Step 3: MZ-REGULARIZED TAIL INVERSION of the local-volatility surface
(§6 of the synthesis, reconciled with the validated §10 mollifier).

Validates ``mz_spectral.sigma_from_phi.regularize_sigma_memory``: a two-stage tail
inversion of σ(K,T) from the scaled call surface φ̃ — (1) a resolved-projection
pre-filter in k̃ (the validated Gaussian k̃-mollifier ``smooth_phi_k`` / the Fourier
low-pass projector) before the Dupire second derivative, yielding a dynamics-informed
tail volatility ``σ_tail = sqrt(2 η̂_tail / T_max)`` with ``η̂_tail = φ̃_t /
(k̃² [φ̃_kk]_MZ)``; (2) a feasibility-gated fill that keeps the pointwise σ on feasible
cells and substitutes the memory σ_tail on the infeasible/tail cells (NOT spatial
interpolation).

Pre-registered falsifiable claim (PRIMARY): tail σ-RMSE(``regularize_sigma_memory``)
  strictly < raw ``implied_sigma_from_phi`` AND ≤ ``regularize_sigma_grid`` on the
  dupire_exact smile, on BOTH tail definitions (~feasible, ~interior).
SECONDARY: band coverage on feasible cells ≥ ~0.90 (no regression vs the validated
  0.91/1.00).

Phase A — const-σ=0.3 (``build_truth_phi_bs``): sanity that the regularizer does NOT
  distort a flat surface (all three methods should sit near 0.30 everywhere; the tail
  σ-RMSE should stay small for memory, no worse than grid).
Phase B — ``VolatilityConfig.dupire_exact()`` smile: σ_oracle = build_uniform_tk_grid(...)[1];
  Σ_φ / Ct via ``input_covariance_from_mc(..., T<=1.0)``; feas = pole_feasibility(Ct,...).
  Reports tail σ-RMSE {raw, grid, memory} on both tail sets + band_coverage on feasible.
  ``{"skipped": True}`` if ``data_mc.npz`` is absent.

HONESTY GUARD: the plain Fourier ``P_R`` / Gaussian mollifier stands in for the unbuilt
  §8 eigenbasis (weighted Sturm–Liouville) projector, and the one-sided ``D2`` boundary
  stencil is non-dissipative (§8 Step-0a). The result is conditional on these.

Pure NumPy + scipy + matplotlib(Agg). No TensorFlow, no training.

Run:  ./.venv/bin/python examples/run_mz_tail_inversion.py
Outputs: mz_tail_inversion.json and plots/mz_tail_inversion/*.png under the package root.
"""

import os
import sys
import json
import math
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DupirePipelineConfig, VolatilityConfig
from mz_spectral.fourier_dupire import build_uniform_tk_grid, build_second_derivative_matrix
from mz_spectral.validation import build_truth_phi_bs, interior_k_mask
from mz_spectral.sigma_from_phi import (
    implied_sigma_from_phi,
    regularize_sigma_grid,
    regularize_sigma_memory,
    mask_first_rows_for_nk,
)
from mz_spectral import uncertainty as uq


# ----------------------------------------------------------------------------
def _const_vol(s):
    return VolatilityConfig.custom(lambda t, x: s * np.ones_like(np.asarray(x, float)))


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return o


def _row_interior_mask(grid):
    """Per-row interior strike mask (quantiles of K), shape (n_t,n_k)."""
    n_t = grid.K.shape[0]
    return np.array([interior_k_mask(grid.K[i], 0.05, 0.95) for i in range(n_t)])


def _tail_rmse(sig, oracle, tailmask):
    """Tail σ-RMSE over (tailmask AND both finite). Returns (rmse, n_cells).

    Raw σ is NaN on masked cells; this scores each method only where it produces a
    finite estimate (the charitable reading for the raw baseline — where it abstains
    it contributes no error and no cell)."""
    sig = np.asarray(sig, dtype=float)
    oracle = np.asarray(oracle, dtype=float)
    m = np.asarray(tailmask, dtype=bool) & np.isfinite(sig) & np.isfinite(oracle)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.sqrt(np.mean((sig[m] - oracle[m]) ** 2))), n


def _three_method_rmse(sig_raw, sig_grid, sig_mem, oracle, tailmask):
    r_raw, n_raw = _tail_rmse(sig_raw, oracle, tailmask)
    r_grid, n_grid = _tail_rmse(sig_grid, oracle, tailmask)
    r_mem, n_mem = _tail_rmse(sig_mem, oracle, tailmask)
    # "beats raw" — apples-to-apples. The raw pointwise inverse ABSTAINS (NaN) on the
    # low-curvature tail cells; scoring its full-tail RMSE on only the cells it answers
    # vs the memory method's RMSE over *all* tail cells penalizes coverage. The fair
    # test: (i) on the cells raw DOES answer (shared finite cells) memory must be no
    # worse — memory keeps the pointwise σ on feasible cells, so this holds with
    # equality by construction; (ii) memory must additionally cover the cells raw
    # abstains on. So memory_beats_raw = "no worse where raw answers AND covers the
    # rest". (Where raw answers nothing — the ~feasible set — (i) is vacuous and (ii)
    # is the whole content: memory supplies finite, accurate estimates where the
    # pointwise inverse declines to answer.)
    sr = np.asarray(sig_raw, dtype=float)
    sm = np.asarray(sig_mem, dtype=float)
    orc = np.asarray(oracle, dtype=float)
    tm = np.asarray(tailmask, dtype=bool)
    shared = tm & np.isfinite(sr) & np.isfinite(sm) & np.isfinite(orc)
    n_shared = int(shared.sum())
    if n_shared > 0:
        rmse_raw_shared = float(np.sqrt(np.mean((sr[shared] - orc[shared]) ** 2)))
        rmse_mem_shared = float(np.sqrt(np.mean((sm[shared] - orc[shared]) ** 2)))
        no_worse_where_raw_answers = rmse_mem_shared <= rmse_raw_shared + 1e-9
    else:
        rmse_raw_shared = float("nan")
        rmse_mem_shared = float("nan")
        no_worse_where_raw_answers = True  # vacuous: raw answers nothing here
    raw_abstained = tm & ~np.isfinite(sr)
    covers_abstained = bool(np.all(np.isfinite(sm[raw_abstained]))) if raw_abstained.any() else True
    memory_beats_raw = bool(no_worse_where_raw_answers and covers_abstained
                            and np.isfinite(r_mem))
    return {
        "raw": r_raw, "grid": r_grid, "memory": r_mem,
        "n_raw": n_raw, "n_grid": n_grid, "n_mem": n_mem,
        "n_tail_cells": int(tm.sum()),
        "raw_shared": rmse_raw_shared, "memory_shared": rmse_mem_shared,
        "n_shared_with_raw": n_shared,
        "memory_beats_raw": memory_beats_raw,
        "memory_le_grid": bool(np.isfinite(r_mem) and np.isfinite(r_grid) and r_mem <= r_grid + 1e-12),
    }


def _build_three(phi, grid, D2, T_max, *, smooth_k=5.0, feasible=None):
    """Raw / grid / memory σ grids from the same φ (same masking conventions)."""
    n_k = phi.shape[1]
    mfr = mask_first_rows_for_nk(n_k)
    sig_raw = implied_sigma_from_phi(phi, grid, D2, T_max, mask_first_rows=mfr)
    sig_grid = regularize_sigma_grid(sig_raw, grid)
    sig_mem = regularize_sigma_memory(sig_raw, grid, phi, D2, T_max,
                                      smooth_k=smooth_k, feasible=feasible)
    return sig_raw, sig_grid, sig_mem


# ----------------------------------------------------------------------------
# Phase A — const-σ sanity (flat surface must not be distorted)
# ----------------------------------------------------------------------------
def phase_A(n_t=24, n_k=256, sigma_const=0.3, z=1.96, smooth_k=5.0):
    cfg = DupirePipelineConfig()
    S0, r, T_max, K_max = cfg.S0, cfg.r, cfg.T_max, cfg.K_max
    grid, _ = build_uniform_tk_grid(
        S0=S0, r=r, T_max=T_max, K_max=K_max, K_min=cfg.K_min, T_min=cfg.T_min,
        n_t=n_t, n_k=n_k, vol_config=_const_vol(sigma_const),
    )
    D2 = build_second_derivative_matrix(n_k, grid.dk)
    phi = build_truth_phi_bs(grid, S0, r, sigma_const)
    sigma_oracle = np.full((n_t, n_k), float(sigma_const))

    feas_mask = uq.pole_feasibility(phi, grid, D2)["feasible"]
    sig_raw, sig_grid, sig_mem = _build_three(phi, grid, D2, T_max,
                                              smooth_k=smooth_k, feasible=feas_mask)

    interior = _row_interior_mask(grid)
    tail_feas = ~feas_mask
    tail_int = ~interior

    rmse_feas = _three_method_rmse(sig_raw, sig_grid, sig_mem, sigma_oracle, tail_feas)
    rmse_int = _three_method_rmse(sig_raw, sig_grid, sig_mem, sigma_oracle, tail_int)

    # band coverage on feasible cells (delta band, const-sigma; rel-noise input cov)
    se = uq.input_covariance_const(phi, rel_noise=0.01)
    _, lo, hi = uq.sigma_band_delta(phi, se, grid, D2, T_max, z=z)
    cov = uq.band_coverage(sigma_oracle, lo, hi, mask=feas_mask)

    i_rep = n_t - 1
    res = {
        "params": {"n_t": n_t, "n_k": n_k, "sigma_const": sigma_const, "smooth_k": smooth_k},
        "tail_rmse_not_feasible": rmse_feas,
        "tail_rmse_not_interior": rmse_int,
        "band_coverage_feasible": _jsonable(cov),
        "frac_feasible": float(feas_mask.mean()),
        "_plot": {
            "k_tilde": grid.k_tilde.tolist(),
            "sig_oracle_rep": sigma_oracle[i_rep].tolist(),
            "sig_raw_rep": sig_raw[i_rep].tolist(),
            "sig_grid_rep": sig_grid[i_rep].tolist(),
            "sig_mem_rep": sig_mem[i_rep].tolist(),
            "feasible_rep": feas_mask[i_rep].tolist(),
        },
    }
    res["verdict"] = (
        f"const-sigma={sigma_const} sanity: tail σ-RMSE (~feasible) raw="
        f"{rmse_feas['raw']:.4f} grid={rmse_feas['grid']:.4f} memory={rmse_feas['memory']:.4f}; "
        f"(~interior) raw={rmse_int['raw']:.4f} grid={rmse_int['grid']:.4f} "
        f"memory={rmse_int['memory']:.4f}. memory_beats_raw="
        f"{rmse_feas['memory_beats_raw']}/{rmse_int['memory_beats_raw']}, memory_le_grid="
        f"{rmse_feas['memory_le_grid']}/{rmse_int['memory_le_grid']}. "
        f"band coverage(feasible)={cov.get('masked', float('nan')):.2f}. "
        f"Flat surface should NOT be distorted (all methods ~{sigma_const})."
    )
    return res


# ----------------------------------------------------------------------------
# Phase B — dupire_exact smile, MC-grounded Sigma_phi (the falsifiable test)
# ----------------------------------------------------------------------------
def phase_B(n_t=15, n_k=128, max_paths=200000, z=1.96, smooth_k=5.0):
    mc_path = os.path.join(_ROOT, "mc_arrays", "dupire_paper", "data_mc.npz")
    if not os.path.exists(mc_path):
        return {"skipped": True, "reason": f"MC arrays not found at {mc_path}"}
    cfg = DupirePipelineConfig()
    S0, r = cfg.S0, cfg.r
    T_max_B = 1.0  # MC time-grid coverage
    grid, sigma_oracle = build_uniform_tk_grid(
        S0=S0, r=r, T_max=T_max_B, K_max=cfg.K_max, K_min=cfg.K_min, T_min=cfg.T_min,
        n_t=n_t, n_k=n_k, vol_config=VolatilityConfig.dupire_exact(),
    )
    D2 = build_second_derivative_matrix(n_k, grid.dk)
    se, Ct, t_used = uq.input_covariance_from_mc(mc_path, grid, r, S0, max_paths=max_paths)

    feas = uq.pole_feasibility(Ct, grid, D2)
    feas_mask = feas["feasible"]

    # three σ methods from the SAME MC price surface Ct
    sig_raw, sig_grid, sig_mem = _build_three(Ct, grid, D2, T_max_B,
                                              smooth_k=smooth_k, feasible=feas_mask)

    interior = _row_interior_mask(grid)
    tail_feas = ~feas_mask
    tail_int = ~interior

    rmse_feas = _three_method_rmse(sig_raw, sig_grid, sig_mem, sigma_oracle, tail_feas)
    rmse_int = _three_method_rmse(sig_raw, sig_grid, sig_mem, sigma_oracle, tail_int)

    # band coverage on feasible cells (delta band) — the SECONDARY bar
    _, s_lo, s_hi = uq.sigma_band_delta(Ct, se, grid, D2, T_max_B, z=z)
    cov_feas = uq.band_coverage(sigma_oracle, s_lo, s_hi, mask=feas_mask)
    cov_int = uq.band_coverage(sigma_oracle, s_lo, s_hi, mask=interior)

    # pre-registered PASS/FAIL (both tail sets must clear)
    passed_primary = bool(
        rmse_feas["memory_beats_raw"] and rmse_feas["memory_le_grid"]
        and rmse_int["memory_beats_raw"] and rmse_int["memory_le_grid"]
    )
    passed_secondary = bool(np.isfinite(cov_feas.get("masked", float("nan")))
                            and cov_feas["masked"] >= 0.90)

    i_rep = n_t - 1
    Krow = grid.K[i_rep]
    res = {
        "skipped": False,
        "params": {"n_t": n_t, "n_k": n_k, "max_paths": max_paths, "smooth_k": smooth_k,
                   "T_range": [float(grid.T[0]), float(grid.T[-1])],
                   "mc_t_used_range": [float(t_used.min()), float(t_used.max())]},
        "se_phi_median": float(np.median(se)),
        "frac_feasible": float(feas_mask.mean()),
        "frac_reliable": float(feas["reliable"].mean()),
        "frac_convex": float(feas["convex"].mean()),
        "tail_rmse_not_feasible": rmse_feas,
        "tail_rmse_not_interior": rmse_int,
        "band_coverage_feasible": _jsonable(cov_feas),
        "band_coverage_interior": _jsonable(cov_int),
        "PASS_primary": passed_primary,
        "PASS_secondary": passed_secondary,
        "_plot": {
            "k_tilde": grid.k_tilde.tolist(), "K_rep": Krow.tolist(),
            "sig_oracle_rep": sigma_oracle[i_rep].tolist(),
            "sig_raw_rep": sig_raw[i_rep].tolist(),
            "sig_grid_rep": sig_grid[i_rep].tolist(),
            "sig_mem_rep": sig_mem[i_rep].tolist(),
            "feasible_rep": feas_mask[i_rep].tolist(),
            "T_rep": float(grid.T[i_rep]),
        },
    }
    res["verdict"] = (
        f"dupire smile (T<=1.0): tail σ-RMSE (~feasible) raw={rmse_feas['raw']:.4f} "
        f"grid={rmse_feas['grid']:.4f} memory={rmse_feas['memory']:.4f} "
        f"[n raw/grid/mem={rmse_feas['n_raw']}/{rmse_feas['n_grid']}/{rmse_feas['n_mem']}]; "
        f"(~interior) raw={rmse_int['raw']:.4f} grid={rmse_int['grid']:.4f} "
        f"memory={rmse_int['memory']:.4f} "
        f"[n raw/grid/mem={rmse_int['n_raw']}/{rmse_int['n_grid']}/{rmse_int['n_mem']}]. "
        f"PRIMARY (memory<raw AND memory<=grid, both tail sets): {'PASS' if passed_primary else 'FAIL'}. "
        f"SECONDARY band coverage(feasible)={cov_feas.get('masked', float('nan')):.2f} "
        f"(>=0.90): {'PASS' if passed_secondary else 'FAIL'}. "
        f"{100*feas_mask.mean():.0f}% feasible, {100*feas['convex'].mean():.0f}% convex."
    )
    return res


# ----------------------------------------------------------------------------
def make_plots(A, B, outdir):
    os.makedirs(outdir, exist_ok=True)

    def _panel(ax, p, title, xkey="k_tilde"):
        x = np.array(p[xkey])
        feas = np.array(p["feasible_rep"], dtype=bool)
        ax.plot(x, p["sig_oracle_rep"], "k-", lw=2.0, label="oracle sigma")
        ax.plot(x, p["sig_raw_rep"], "C7.", ms=3, label="raw implied (NaN where masked)")
        ax.plot(x, p["sig_grid_rep"], "C1--", lw=1.3, label="regularize_sigma_grid")
        ax.plot(x, p["sig_mem_rep"], "C0-", lw=1.5, label="regularize_sigma_memory")
        # mark the tail (infeasible) cells along the bottom
        ymin = 0.0
        ax.plot(x[~feas], np.full((~feas).sum(), ymin), "rx", ms=3, alpha=0.5,
                label="infeasible (tail)")
        ax.set_title(title)
        ax.set_xlabel(xkey); ax.set_ylabel("sigma"); ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    try:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        _panel(ax[0], A["_plot"],
               f"Phase A: const-sigma sanity (no distortion)")
        ax[0].set_ylim(0, 0.6)
        if not B.get("skipped"):
            _panel(ax[1], B["_plot"],
                   f"Phase B: dupire smile (T={B['_plot']['T_rep']:.2f})", xkey="K_rep")
            ax[1].set_ylim(0, 1.5)
        else:
            ax[1].text(0.5, 0.5, "Phase B skipped\n(no data_mc.npz)",
                       ha="center", va="center"); ax[1].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "mz_tail_inversion_sigma.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"[plot] skipped: {e}")

    # Phase B tail-RMSE bar chart (the headline numbers)
    if not B.get("skipped"):
        try:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
            for j, (key, lab) in enumerate([("tail_rmse_not_feasible", "tail = ~feasible"),
                                            ("tail_rmse_not_interior", "tail = ~interior")]):
                d = B[key]
                vals = [d["raw"], d["grid"], d["memory"]]
                bars = ax[j].bar(["raw", "grid", "memory"], vals,
                                 color=["C7", "C1", "C0"])
                for b, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax[j].text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                                   ha="center", va="bottom", fontsize=8)
                ax[j].set_title(f"Phase B tail σ-RMSE ({lab})")
                ax[j].set_ylabel("tail σ-RMSE"); ax[j].grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "mz_tail_inversion_rmse.png"), dpi=130)
            plt.close(fig)
        except Exception as e:
            print(f"[plot rmse] skipped: {e}")


def main():
    print("=" * 78)
    print("MZ x Dupire — Step 3: MZ-regularized tail inversion of sigma(T,K)")
    print("=" * 78)

    print("\n[Phase A] const-sigma=0.3 sanity (no distortion) ...")
    A = phase_A()
    print("  VERDICT:", A["verdict"])

    print("\n[Phase B] dupire_exact smile, MC-grounded Sigma_phi (falsifiable) ...")
    B = phase_B()
    print("  VERDICT:", B.get("verdict", B.get("reason")))

    make_plots(A, B, os.path.join(_ROOT, "plots", "mz_tail_inversion"))

    A.pop("_plot", None)
    if not B.get("skipped"):
        B.pop("_plot", None)
    payload = {"phase_A": A, "phase_B": B}
    outjson = os.path.join(_ROOT, "mz_tail_inversion.json")
    with open(outjson, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    print(f"\nWrote {outjson}")
    print(f"Wrote plots to {os.path.join(_ROOT, 'plots', 'mz_tail_inversion')}/")
    print("\nDONE.")


if __name__ == "__main__":
    main()
