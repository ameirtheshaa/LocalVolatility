#!/usr/bin/env python3
"""Export each MZ vol-surface / PDF panel as a standalone high-resolution figure."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from matplotlib import cm
from scipy.stats import kurtosis, norm, skew

from mz_spectral.fourier_dupire import TKGrid
from mz_spectral.plot_pdf_analysis import (
    COLORS,
    _density_to_mc_x_space,
    _mc_samples_for_maturity,
    reference_pdf_row,
)
from mz_spectral.plot_vol_surfaces import (
    _interior_sigma_mask,
    _shared_sigma_zlim,
    _sigma_rmse,
    plot_mz_volatility_surfaces_triple,
)
from mz_spectral.sigma_from_phi import (
    implied_sigma_from_phi,
    mask_first_rows_for_nk,
    phi_for_sigma_inversion,
)
from mz_spectral.validation import pdf_from_phi_tilde, spectral_pdf_filter

PANEL_DPI = 600


def _save_fig(fig: plt.Figure, path_base: str) -> Dict[str, str]:
    os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
    png = f"{path_base}.png"
    pdf = f"{path_base}.pdf"
    fig.savefig(png, dpi=PANEL_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, dpi=PANEL_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def export_vol_surface_panels(
    grid: TKGrid,
    sigma_exact: np.ndarray,
    phi_rrr: np.ndarray,
    phi_ql: np.ndarray,
    D2: np.ndarray,
    T_max: float,
    out_dir: str,
    case: str,
    *,
    chosen_keep: int | None = None,
) -> Dict[str, Dict[str, str]]:
    """Six standalone figures: 3× σ surfaces + 3× error contours."""
    os.makedirs(out_dir, exist_ok=True)
    n_k_dim = int(phi_rrr.shape[1])
    mfr = mask_first_rows_for_nk(n_k_dim)
    phi_r = phi_for_sigma_inversion(phi_rrr, chosen_keep)
    phi_q = phi_for_sigma_inversion(phi_ql, chosen_keep)
    sigma_rrr = implied_sigma_from_phi(phi_r, grid, D2, T_max, mask_first_rows=mfr)
    sigma_ql = implied_sigma_from_phi(phi_q, grid, D2, T_max, mask_first_rows=mfr)
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

    paths: Dict[str, Dict[str, str]] = {}

    def _surf(Z: np.ndarray, title: str, stem: str) -> None:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        Zm = ma.masked_invalid(Z)
        s = ax.plot_surface(
            K_mesh, T_mesh, Zm, cmap=cm.plasma, alpha=0.92, linewidth=0, antialiased=True
        )
        ax.set_xlabel("Strike K", fontsize=12)
        ax.set_ylabel("Maturity T", fontsize=12)
        ax.set_zlabel("σ", fontsize=12)
        ax.set_zlim(zlo, zhi)
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.colorbar(s, ax=ax, shrink=0.6, aspect=12)
        paths[stem] = _save_fig(fig, os.path.join(out_dir, f"{case}_vol_{stem}"))

    def _cont(Z: np.ndarray, title: str, stem: str) -> None:
        fig, ax = plt.subplots(figsize=(11, 7))
        Zm = ma.masked_invalid(Z)
        im = ax.contourf(K_mesh, T_mesh, Zm, levels=20, cmap="hot")
        ax.set_xlabel("Strike K", fontsize=12)
        ax.set_ylabel("Maturity T", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax)
        paths[stem] = _save_fig(fig, os.path.join(out_dir, f"{case}_vol_{stem}"))

    _surf(sigma_exact, "Exact σ(K,T)", "exact_sigma")
    _surf(sigma_rrr, "RRR implied σ(K,T)", "rrr_sigma")
    _surf(sigma_ql, "RRR+QL implied σ(K,T)", "ql_sigma")
    _cont(
        err_rrr,
        f"|σ_RRR − σ_exact|  (RMSE={rmse_rrr:.4f}, interior={rmse_rrr_i:.4f})",
        "err_rrr",
    )
    _cont(
        err_ql,
        f"|σ_QL − σ_exact|  (RMSE={rmse_ql:.4f}, interior={rmse_ql_i:.4f})",
        "err_ql",
    )
    _cont(err_cross, "|σ_QL − σ_RRR|", "err_cross")
    return paths


def export_vol_surfaces_triple_panel(
    grid: TKGrid,
    sigma_exact: np.ndarray,
    phi_rrr: np.ndarray,
    phi_ql: np.ndarray,
    D2: np.ndarray,
    T_max: float,
    out_dir: str,
    case: str,
    *,
    chosen_keep: int | None = None,
) -> Dict[str, str]:
    """Single 1×3 figure: exact / RRR / RRR+QL σ(K,T) (600 dpi, Beamer-friendly name)."""
    n_k_dim = int(phi_rrr.shape[1])
    mfr = mask_first_rows_for_nk(n_k_dim)
    phi_r = phi_for_sigma_inversion(phi_rrr, chosen_keep)
    phi_q = phi_for_sigma_inversion(phi_ql, chosen_keep)
    sigma_rrr = implied_sigma_from_phi(phi_r, grid, D2, T_max, mask_first_rows=mfr)
    sigma_ql = implied_sigma_from_phi(phi_q, grid, D2, T_max, mask_first_rows=mfr)
    paths = plot_mz_volatility_surfaces_triple(
        grid,
        sigma_exact,
        sigma_rrr,
        sigma_ql,
        out_dir,
        case_label=case,
        dpi=PANEL_DPI,
    )
    # Stable symlink-style copy for Beamer: {case}_vol_surfaces_triple.png
    import shutil

    src = paths["vol_surfaces_triple_png"]
    dest = os.path.join(out_dir, f"{case}_vol_surfaces_triple.png")
    shutil.copy2(src, dest)
    paths["vol_surfaces_triple_png_beamer"] = dest
    return paths


def export_pdf_panels(
    grid: TKGrid,
    phi_truth: np.ndarray,
    phi_rrr: np.ndarray,
    phi_ql: np.ndarray,
    D2: np.ndarray,
    config: Any,
    maturities: List[float],
    chosen_keep: int,
    case: str,
    sigma_const: Optional[float],
    out_dir: str,
    *,
    mc_data: Optional[Mapping[float, np.ndarray]] = None,
) -> Dict[str, Dict[str, str]]:
    """One figure per (maturity × K / logK / Gaussian)."""
    from sklearn.neighbors import KernelDensity

    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, Dict[str, str]] = {}
    x_range = np.linspace(-3, 3, 100)
    std_normal = norm.pdf(x_range)
    truth_label = "Truth ref. (MC KDE)" if case == "paper" else "Truth ref."

    for T_req in maturities:
        idx = int(np.argmin(np.abs(grid.T - float(T_req))))
        T = float(grid.T[idx])
        K_row = grid.K[idx]
        samples = _mc_samples_for_maturity(mc_data, T_req)
        use = samples
        if samples is not None and samples.size >= 2:
            q1, q99 = np.percentile(samples, [1, 99])
            use = samples[(samples >= q1) & (samples <= q99)]
            if use.size < 50:
                use = samples

        f_truth = reference_pdf_row(
            case,
            T,
            K_row,
            phi_truth[idx],
            float(config.r),
            float(config.S0),
            float(config.K_max),
            D2,
            sigma_const,
            mc_samples=samples,
        )
        f_rrr = pdf_from_phi_tilde(
            spectral_pdf_filter(phi_rrr[idx], chosen_keep),
            T,
            float(config.r),
            float(config.S0),
            float(config.K_max),
            K_row,
            D2,
        )
        f_ql = pdf_from_phi_tilde(
            spectral_pdf_filter(phi_ql[idx], chosen_keep),
            T,
            float(config.r),
            float(config.S0),
            float(config.K_max),
            K_row,
            D2,
        )
        ttag = f"T{float(T_req):.2f}".replace(".", "p")
        log_K = np.log(np.maximum(K_row, 1e-16))

        # K-space
        fig, ax = plt.subplots(figsize=(11, 7))
        if use is not None and use.size >= 2:
            ax.hist(
                use,
                bins=50,
                density=True,
                alpha=0.55,
                color=COLORS["mc_hist"],
                edgecolor="white",
                linewidth=0.5,
                label="MC",
            )
        ax.plot(K_row, f_truth, color=COLORS["truth"], linewidth=2.5, label=truth_label)
        ax.plot(K_row, f_rrr, color=COLORS["rrr"], linewidth=2, label="RRR (filtered)")
        ax.plot(K_row, f_ql, color=COLORS["ql"], linewidth=2, label="RRR+QL (filtered)")
        ax.set_xlabel(r"Strike $K$", fontsize=12)
        ax.set_ylabel(r"$f(K)$", fontsize=12)
        ax.set_title(
            f"{case}: K-space  ($T_\\mathrm{{req}}$={T_req:.2f}, grid $T$={T:.3f})",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.35)
        paths[f"{ttag}_K"] = _save_fig(fig, os.path.join(out_dir, f"{case}_pdf_{ttag}_K"))

        # log-K
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.plot(log_K, f_truth * K_row, color=COLORS["truth"], linewidth=2.5, label=truth_label)
        ax.plot(log_K, f_rrr * K_row, color=COLORS["rrr"], linewidth=2, label="RRR")
        ax.plot(log_K, f_ql * K_row, color=COLORS["ql"], linewidth=2, label="RRR+QL")
        if use is not None and use.size >= 2:
            ax.hist(
                np.log(use),
                bins=50,
                density=True,
                alpha=0.45,
                color=COLORS["kde"],
                edgecolor="white",
                linewidth=0.5,
                label="MC log K",
            )
        ax.set_xlabel(r"$\ln K$", fontsize=12)
        ax.set_ylabel(r"$f(\ln K)$", fontsize=12)
        ax.set_title(f"{case}: log-strike density ($T$={T_req:.2f})", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.35)
        paths[f"{ttag}_logK"] = _save_fig(fig, os.path.join(out_dir, f"{case}_pdf_{ttag}_logK"))

        # Gaussian
        fig, ax = plt.subplots(figsize=(11, 7))
        if samples is not None and samples.size >= 20:
            ln = np.log(samples[samples > 0])
            mu_mc = float(ln.mean())
            sigma_mc = float(max(ln.std(), 1e-8))
            x_mc = (ln - mu_mc) / sigma_mc
            kde_vals = np.zeros_like(x_range)
            try:
                kde = KernelDensity(kernel="gaussian", bandwidth=0.12)
                kde.fit(x_mc.reshape(-1, 1))
                kde_vals = np.exp(kde.score_samples(x_range.reshape(-1, 1)))
            except Exception:
                pass
            ax.hist(
                x_mc,
                bins=30,
                density=True,
                alpha=0.5,
                color=COLORS["mc_hist"],
                edgecolor="white",
                linewidth=0.5,
                label="MC standardized",
            )
            ax.plot(x_range, kde_vals, color=COLORS["kde"], linewidth=2, label="MC KDE", zorder=9)
        else:
            ln_t = np.log(np.maximum(K_row, 1e-16))
            w = f_truth / (np.sum(f_truth) + 1e-16)
            mu_mc = float(np.sum(w * ln_t))
            sigma_mc = float(max(np.sqrt(np.sum(w * (ln_t - mu_mc) ** 2)), 1e-8))
            x_mc = (ln_t - mu_mc) / sigma_mc

        g_truth = _density_to_mc_x_space(K_row, f_truth, mu_mc, sigma_mc, x_range)
        g_rrr = _density_to_mc_x_space(K_row, f_rrr, mu_mc, sigma_mc, x_range)
        g_ql = _density_to_mc_x_space(K_row, f_ql, mu_mc, sigma_mc, x_range)
        vt, vr, vq = np.isfinite(g_truth), np.isfinite(g_rrr), np.isfinite(g_ql)
        ax.plot(
            x_range[vt],
            g_truth[vt],
            color=COLORS["truth"],
            linewidth=2.2,
            label="MC KDE g(x)" if case == "paper" else "Truth g(x)",
        )
        ax.plot(x_range[vr], g_rrr[vr], color=COLORS["rrr"], linewidth=2, label="RRR g(x)")
        ax.plot(x_range[vq], g_ql[vq], color=COLORS["ql"], linewidth=2, label="RRR+QL g(x)")
        ax.plot(x_range, std_normal, color=COLORS["normal"], linestyle=":", linewidth=2, label="N(0,1)")
        sk = skew(x_mc) if samples is not None and samples.size >= 20 else float("nan")
        ku = kurtosis(x_mc, fisher=True) if samples is not None and samples.size >= 20 else float("nan")
        ax.set_xlim(-3, 3)
        ax.set_xlabel(r"$x=(\ln K-\mu)/\sigma$", fontsize=12)
        ax.set_ylabel(r"$g(x)$", fontsize=12)
        ax.set_title(
            f"{case}: Gaussian coords ($T$={T_req:.2f}, "
            f"$\\mu$={mu_mc:.2f}, $\\sigma$={sigma_mc:.2f}, skew={sk:.2f})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.35)
        paths[f"{ttag}_gauss"] = _save_fig(fig, os.path.join(out_dir, f"{case}_pdf_{ttag}_gauss"))

    return paths
