#!/usr/bin/env python3
"""
3D local volatility σ(K,T) surfaces: exact vs MZ-implied from φ̃ (RRR / RRR+QL).

Inversion (scaled Dupire, §4.3): ∂φ̃/∂t̃ = η̃ k̃² ∂²φ̃/∂k̃²  ⇒  η̃ = (∂φ̃/∂t̃) / (k̃² ∂²φ̃/∂k̃²).

Physical vol: σ = sqrt(2 η̃ / T_max), matching PDFAnalyzer / NN convention.

Unreliable cells (|φ_kk| small vs row scale, first time rows with one-sided ∂φ̃/∂t̃)
are masked to NaN so 3D colorbars are not dominated by wing spikes.
"""

from __future__ import annotations

import datetime
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from matplotlib import cm

from mz_spectral.fourier_dupire import TKGrid
from mz_spectral.sigma_from_phi import (
    implied_sigma_from_phi,
    mask_first_rows_for_nk,
    phi_for_sigma_inversion,
    sigma_reliability_mask,
)

# Match compare_nn_vs_analytical.py
DPI = 450


def _interior_sigma_mask(
    K_mesh: np.ndarray,
    T_mesh: np.ndarray,
    *,
    qk: Tuple[float, float] = (0.05, 0.95),
    qt: Tuple[float, float] = (0.1, 0.9),
) -> np.ndarray:
    k_lo, k_hi = np.quantile(K_mesh.ravel(), qk)
    t_lo, t_hi = np.quantile(T_mesh.ravel(), qt)
    return (K_mesh >= k_lo) & (K_mesh <= k_hi) & (T_mesh >= t_lo) & (T_mesh <= t_hi)


def _sigma_rmse(a: np.ndarray, b: np.ndarray, extra_mask: np.ndarray | None = None) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if extra_mask is not None:
        m &= extra_mask
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _shared_sigma_zlim(
    sigma_exact: np.ndarray,
    sigma_rrr: np.ndarray,
    sigma_ql: np.ndarray,
    *,
    pct: Tuple[float, float] = (2.0, 98.0),
) -> Tuple[float, float]:
    vals = []
    for z in (sigma_exact, sigma_rrr, sigma_ql):
        v = z[np.isfinite(z)]
        if v.size:
            vals.append(v)
    if not vals:
        return 0.0, 1.5
    v = np.concatenate(vals)
    zlo, zhi = np.percentile(v, pct)
    span = max(zhi - zlo, 1e-6)
    pad = 0.05 * span
    return float(zlo - pad), float(zhi + pad)


def plot_mz_volatility_surfaces_triple(
    grid: TKGrid,
    sigma_exact: np.ndarray,
    sigma_rrr: np.ndarray,
    sigma_ql: np.ndarray,
    out_dir: str,
    *,
    case_label: str = "",
    zlim: Tuple[float, float] | None = None,
    dpi: int = DPI,
) -> Dict[str, str]:
    """
    Save a single 1×3 figure: exact / RRR / RRR+QL σ(K,T) with shared z-limits.
    """
    os.makedirs(out_dir, exist_ok=True)
    if zlim is None:
        zlo, zhi = _shared_sigma_zlim(sigma_exact, sigma_rrr, sigma_ql)
    else:
        zlo, zhi = zlim

    K_mesh = grid.K
    T_mesh = np.broadcast_to(grid.T[:, np.newaxis], grid.K.shape)
    panels = [
        (sigma_exact, "Exact σ(K,T)"),
        (sigma_rrr, "RRR implied σ(K,T)"),
        (sigma_ql, "RRR+QL implied σ(K,T)"),
    ]

    fig = plt.figure(figsize=(20, 6.5))
    title = "MZ local volatility σ(K,T): exact vs RRR vs RRR+QL"
    if case_label:
        title = f"{title} — {case_label}"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    for i, (Z, subtitle) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        Zm = ma.masked_invalid(Z)
        surf = ax.plot_surface(
            K_mesh, T_mesh, Zm, cmap=cm.plasma, alpha=0.92, linewidth=0, antialiased=True
        )
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Maturity T")
        ax.set_zlabel("σ")
        ax.set_zlim(zlo, zhi)
        ax.set_title(subtitle, fontweight="bold")
        fig.colorbar(surf, ax=ax, shrink=0.55, aspect=10)

    plt.tight_layout()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"vol_surfaces_triple_{case_label}_" if case_label else "vol_surfaces_triple_"
    png = os.path.join(out_dir, f"{prefix}{ts}.png")
    pdf = os.path.join(out_dir, f"{prefix}{ts}.pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"vol_surfaces_triple_png": png, "vol_surfaces_triple_pdf": pdf}


def plot_mz_volatility_surfaces(
    grid: TKGrid,
    sigma_ref: np.ndarray,
    phi_rrr: np.ndarray,
    phi_ql: np.ndarray,
    D2: np.ndarray,
    T_max: float,
    out_dir: str,
    *,
    case_label: str = "",
    chosen_keep: int | None = None,
    sigma_ref_label: str = "Exact σ(K,T)",
    sigma_oracle: np.ndarray | None = None,
    vol_mode: str = "oracle",
) -> Dict[str, object]:
    """
    Save PNG/PDF with 3×3 layout: row1 reference/RRR/QL surfaces; row2 error contours.

    RRR/QL surfaces use NaN holes (masked arrays) instead of forcing zeros into spikes.
    If ``chosen_keep`` is set, applies ``spectral_pdf_filter`` per time row before inversion.
    In data mode, ``sigma_ref`` is typically Dupire-inverted ``σ(φ_data)``.
    Returns paths, full-grid and interior σ RMSEs vs reference on finite cells.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_k_dim = int(phi_rrr.shape[1])
    mfr = mask_first_rows_for_nk(n_k_dim)
    phi_r = phi_for_sigma_inversion(phi_rrr, chosen_keep)
    phi_q = phi_for_sigma_inversion(phi_ql, chosen_keep)
    sigma_rrr = implied_sigma_from_phi(phi_r, grid, D2, T_max, mask_first_rows=mfr)
    sigma_ql = implied_sigma_from_phi(phi_q, grid, D2, T_max, mask_first_rows=mfr)
    sigma_exact = sigma_ref

    K_mesh = grid.K
    T_mesh = np.broadcast_to(grid.T[:, np.newaxis], grid.K.shape)
    interior = _interior_sigma_mask(K_mesh, T_mesh)

    err_rrr = np.abs(sigma_rrr - sigma_exact)
    err_ql = np.abs(sigma_ql - sigma_exact)
    err_cross = np.abs(sigma_ql - sigma_rrr)

    rmse_rrr = _sigma_rmse(sigma_rrr, sigma_exact)
    rmse_ql = _sigma_rmse(sigma_ql, sigma_exact)
    rmse_rrr_i = _sigma_rmse(sigma_rrr, sigma_exact, interior)
    rmse_ql_i = _sigma_rmse(sigma_ql, sigma_exact, interior)

    zlo, zhi = _shared_sigma_zlim(sigma_exact, sigma_rrr, sigma_ql)

    fig = plt.figure(figsize=(18, 10))
    if vol_mode == "data":
        title = "MZ local volatility σ(K,T): data Dupire vs RRR vs RRR+QL"
    else:
        title = "MZ local volatility σ(K,T): exact vs RRR vs RRR+QL"
    if case_label:
        title = f"{title} — {case_label}"
    if sigma_oracle is not None and vol_mode == "data":
        title = f"{title} (dashed overlay: generating σ)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    def surf(ax_pos, Z, zlabel, title_txt):
        ax = fig.add_subplot(2, 3, ax_pos, projection="3d")
        Zm = ma.masked_invalid(Z)
        surf_ = ax.plot_surface(
            K_mesh,
            T_mesh,
            Zm,
            cmap=cm.plasma,
            alpha=0.92,
            linewidth=0,
            antialiased=True,
        )
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Maturity T")
        ax.set_zlabel(zlabel)
        ax.set_zlim(zlo, zhi)
        ax.set_title(title_txt, fontweight="bold")
        fig.colorbar(surf_, ax=ax, shrink=0.55, aspect=8)
        return ax

    def cont(ax_pos, Z, title_txt):
        ax = fig.add_subplot(2, 3, ax_pos)
        Zm = ma.masked_invalid(Z)
        im = ax.contourf(K_mesh, T_mesh, Zm, levels=20, cmap="hot")
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Maturity T")
        ax.set_title(title_txt, fontweight="bold")
        fig.colorbar(im, ax=ax)
        return ax

    surf(1, sigma_exact, "σ", sigma_ref_label)
    surf(2, sigma_rrr, "σ", "RRR implied σ(K,T)")
    surf(3, sigma_ql, "σ", "RRR+QL implied σ(K,T)")

    ref_short = "σ_ref" if vol_mode == "data" else "σ_exact"
    cont(4, err_rrr, f"|σ_RRR − {ref_short}|  (RMSE={rmse_rrr:.4f}, interior={rmse_rrr_i:.4f})")
    cont(5, err_ql, f"|σ_QL − {ref_short}|  (RMSE={rmse_ql:.4f}, interior={rmse_ql_i:.4f})")
    cont(6, err_cross, "|σ_QL − σ_RRR|")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png = os.path.join(out_dir, f"vol_surfaces_{ts}.png")
    pdf = os.path.join(out_dir, f"vol_surfaces_{ts}.pdf")
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    triple = plot_mz_volatility_surfaces_triple(
        grid,
        sigma_exact,
        sigma_rrr,
        sigma_ql,
        out_dir,
        case_label=case_label,
        zlim=(zlo, zhi),
        dpi=DPI,
    )

    return {
        "vol_surface_png": png,
        "vol_surface_pdf": pdf,
        **triple,
        "sigma_rmse_rrr_vs_ref": rmse_rrr,
        "sigma_rmse_ql_vs_ref": rmse_ql,
        "sigma_rmse_rrr_vs_ref_interior": rmse_rrr_i,
        "sigma_rmse_ql_vs_ref_interior": rmse_ql_i,
        "sigma_rmse_rrr_vs_exact": rmse_rrr,
        "sigma_rmse_ql_vs_exact": rmse_ql,
        "sigma_rmse_rrr_vs_exact_interior": rmse_rrr_i,
        "sigma_rmse_ql_vs_exact_interior": rmse_ql_i,
        "sigma_zlim_shared": [zlo, zhi],
        "vol_mode": vol_mode,
        "sigma_ref_label": sigma_ref_label,
    }
