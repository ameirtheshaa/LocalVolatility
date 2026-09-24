#!/usr/bin/env python3
"""
Multi-maturity PDF analysis for MZ synthetic validation (Breeden–Litzenberger).

Layout mirrors ``PDFAnalyzer.create_enhanced_pdf_analysis`` (K / log-K / Gaussian panels).
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, norm, skew

from mz_spectral.fourier_dupire import TKGrid
from mz_spectral.validation import (
    interior_k_mask,
    pdf_from_phi_tilde,
    pdf_metrics,
    spectral_pdf_filter,
)

DPI = 450

# Align with dupire_pipeline.COLORS (avoid importing TF stack)
COLORS = {
    "mc_hist": "#4A90E2",
    "truth": "#2C3E50",
    "rrr": "#E94B3C",
    "ql": "#50C878",
    "nn": "#9B59B6",
    "normal": "#2C3E50",
    "kde": "#50C878",
    "text": "#34495E",
    "background": "#FAFAFA",
}


def nearest_time_index(T_column: np.ndarray, T_target: float) -> int:
    """Index of grid maturity closest to ``T_target``."""
    T_column = np.asarray(T_column, dtype=float).ravel()
    return int(np.argmin(np.abs(T_column - float(T_target))))


def _normalize_pdf(f: np.ndarray, K_row: np.ndarray) -> np.ndarray:
    f = np.maximum(np.asarray(f, dtype=float), 0.0)
    if len(K_row) > 1:
        area = np.trapz(f, K_row)
        if area > 0:
            f /= area
    return f


def bl_pdf_from_phi_row(
    phi_truth_row: np.ndarray,
    T: float,
    r: float,
    S0: float,
    K_max: float,
    K_row: np.ndarray,
    D2: np.ndarray,
) -> np.ndarray:
    """BL density from a ``phi_truth`` row (can be noisy on interpolated wings)."""
    f = pdf_from_phi_tilde(
        phi_truth_row, float(T), r, S0, K_max, K_row, D2, normalize=True
    )
    return _normalize_pdf(f, K_row)


def mc_kde_pdf_on_K_grid(
    samples: np.ndarray,
    K_row: np.ndarray,
    *,
    bandwidth: float = 0.12,
) -> np.ndarray:
    """
    Risk-neutral terminal density f(K) from MC samples via log-K Gaussian KDE.

    f_K(K) = f_{ln K}(ln K) / K with f_{ln K} from ``KernelDensity`` on ln(S_T).
    """
    from sklearn.neighbors import KernelDensity

    K_row = np.asarray(K_row, dtype=float)
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size < 20:
        return np.zeros_like(K_row)

    log_s = np.log(s)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(log_s.reshape(-1, 1))
    log_K = np.log(np.maximum(K_row, 1e-16))
    f_lnK = np.exp(kde.score_samples(log_K.reshape(-1, 1))).ravel()
    f_K = np.maximum(f_lnK / K_row, 0.0)
    return _normalize_pdf(f_K, K_row)


def reference_pdf_row(
    case: str,
    T: float,
    K_row: np.ndarray,
    phi_truth_row: np.ndarray,
    r: float,
    S0: float,
    K_max: float,
    D2: np.ndarray,
    sigma_const: Optional[float],
    *,
    mc_samples: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reference density on ``K_row`` for metrics and the solid truth curve.

    - **constant:** analytical lognormal.
    - **paper:** MC KDE on ``K_row`` when ``mc_samples`` has **≥20** usable paths and KDE mass normalizes;
      otherwise falls back to BL from ``phi_truth`` (legacy / no-MC path).
    """
    K_row = np.asarray(K_row, dtype=float)
    if case == "constant":
        if sigma_const is None:
            raise ValueError("constant case requires sigma_const")
        from analytical_solutions import lognormal_density

        f = np.asarray(
            lognormal_density(K_row, S0, float(T), r, float(sigma_const)), dtype=float
        )
        return _normalize_pdf(f, K_row)

    if mc_samples is not None:
        f_mc = mc_kde_pdf_on_K_grid(mc_samples, K_row)
        if np.trapz(f_mc, K_row) > 0:
            return f_mc
    return bl_pdf_from_phi_row(phi_truth_row, T, r, S0, K_max, K_row, D2)


# Plan / public API name
reference_pdf = reference_pdf_row


def pdf_reference_is_mc_kde(
    case: str, samples: Optional[np.ndarray], K_row: np.ndarray
) -> bool:
    """True when ``reference_pdf_row`` uses MC KDE (paper case, enough paths, positive mass)."""
    if case != "paper" or samples is None:
        return False
    f_mc = mc_kde_pdf_on_K_grid(samples, K_row)
    return float(np.trapz(f_mc, K_row)) > 0.0


def _mc_samples_for_maturity(
    mc_data: Optional[Mapping[float, np.ndarray]],
    T_req: float,
) -> Optional[np.ndarray]:
    if mc_data is None:
        return None
    raw = mc_data.get(float(T_req))
    if raw is None and mc_data:
        nearest_T = min(mc_data.keys(), key=lambda x: abs(float(x) - float(T_req)))
        raw = mc_data[nearest_T]
    if raw is None:
        return None
    return raw[np.isfinite(raw) & (raw > 0)]


def pdf_comparison_table(
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
    *,
    mc_data: Optional[Mapping[float, np.ndarray]] = None,
    k_interior: Tuple[float, float] = (0.05, 0.95),
    vol_mode: str = "oracle",
    phi_data: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Per maturity: RRR / QL vs reference (MC KDE for paper; interior K band)."""
    out: Dict[str, Any] = {}
    qlo, qhi = k_interior
    for T_req in maturities:
        idx = nearest_time_index(grid.T, T_req)
        T = float(grid.T[idx])
        K_row = grid.K[idx]
        km = interior_k_mask(K_row, qlo, qhi)
        samples = _mc_samples_for_maturity(mc_data, T_req)
        f_ref = reference_pdf_row(
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
        key = f"{float(T_req):.4f}"
        ref_kind = (
            "mc_kde"
            if pdf_reference_is_mc_kde(case, samples, K_row)
            else ("lognormal" if case == "constant" else "bl_phi_truth")
        )
        row_out: Dict[str, Any] = {
            "grid_T_row": T,
            "grid_row_index": idx,
            "reference": ref_kind,
            "rrr": pdf_metrics(f_rrr, f_ref, K_row, k_mask=km),
            "ql": pdf_metrics(f_ql, f_ref, K_row, k_mask=km),
        }
        if vol_mode == "data" and phi_data is not None:
            f_bl_data = pdf_from_phi_tilde(
                spectral_pdf_filter(phi_data[idx], chosen_keep),
                T,
                float(config.r),
                float(config.S0),
                float(config.K_max),
                K_row,
                D2,
            )
            row_out["bl_phi_data"] = pdf_metrics(f_bl_data, f_ref, K_row, k_mask=km)
        out[key] = row_out
    return out


def _density_to_mc_x_space(
    K_grid: np.ndarray,
    f_K: np.ndarray,
    mu_mc: float,
    sigma_mc: float,
    x_range: np.ndarray,
) -> np.ndarray:
    """Map f(K) to g(x) with x = (ln K - mu_mc) / sigma_mc, same Jacobian as dupire_pipeline."""
    if len(K_grid) < 5 or len(f_K) != len(K_grid):
        return np.full_like(x_range, np.nan, dtype=float)
    x_model = (np.log(K_grid) - mu_mc) / sigma_mc
    g = f_K * sigma_mc * K_grid
    if len(x_model) > 1:
        integral_g = np.trapz(g, x_model)
        if integral_g > 0:
            g = g / integral_g
    valid = (x_model >= -4) & (x_model <= 4)
    if np.sum(valid) < 5:
        return np.full_like(x_range, np.nan, dtype=float)
    xv = x_model[valid]
    gv = g[valid]
    order = np.argsort(xv)
    return np.interp(x_range, xv[order], gv[order], left=np.nan, right=np.nan)


def mc_samples_at_maturities(config: Any, maturity_list: List[float]) -> Dict[float, np.ndarray]:
    """Terminal stock samples at MC time rows nearest each requested maturity."""
    from dupire_pipeline import DataGenerator

    dg = DataGenerator(config)
    dg.run_mc()
    t_all = dg.t_all.numpy().ravel().astype(float)
    S_mat = dg.S_matrix.numpy()
    out: Dict[float, np.ndarray] = {}
    for T in maturity_list:
        j = int(np.argmin(np.abs(t_all - float(T))))
        out[float(T)] = np.asarray(S_mat[j, :], dtype=float)
    return out


def create_mz_pdf_analysis(
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
    case_label: str = "",
    mc_data: Optional[Mapping[float, np.ndarray]] = None,
    nn_f_by_T: Optional[Mapping[float, np.ndarray]] = None,
    vol_mode: str = "oracle",
    phi_data: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Build figure (len(maturities) × 3) and a results dict with metrics + artifact paths.

    RRR / RRR+QL curves use ``spectral_pdf_filter`` before BL differentiation.
    """
    from sklearn.neighbors import KernelDensity

    os.makedirs(out_dir, exist_ok=True)
    n_rows = len(maturities)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    fig.patch.set_facecolor("white")
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    if vol_mode == "data":
        title = "MZ PDF analysis (Breeden–Litzenberger): ref vs data BL vs RRR vs RRR+QL"
    else:
        title = "MZ PDF analysis (Breeden–Litzenberger): truth vs RRR vs RRR+QL"
    if case_label:
        title = f"{title} — {case_label}"
    fig.suptitle(title, fontsize=16, fontweight="bold", color=COLORS["text"], y=0.98)

    metrics_table = pdf_comparison_table(
        grid,
        phi_truth,
        phi_rrr,
        phi_ql,
        D2,
        config,
        maturities,
        chosen_keep,
        case,
        sigma_const,
        mc_data=mc_data,
        vol_mode=vol_mode,
        phi_data=phi_data,
    )

    x_range = np.linspace(-3, 3, 100)
    std_normal = norm.pdf(x_range)

    for row, T_req in enumerate(maturities):
        idx = nearest_time_index(grid.T, T_req)
        T = float(grid.T[idx])
        K_row = grid.K[idx]

        samples = _mc_samples_for_maturity(mc_data, T_req)
        use_mc_ref = pdf_reference_is_mc_kde(case, samples, K_row)

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

        ax1, ax2, ax3 = axes[row, 0], axes[row, 1], axes[row, 2]

        if case == "constant":
            truth_label = "Truth ref."
        elif use_mc_ref:
            truth_label = "MC reference"
        elif vol_mode == "data":
            truth_label = "BL from φ_data (sanity)" if not use_mc_ref else "MC reference"
        else:
            truth_label = "BL from φ_truth (sanity)"

        f_bl_data = None
        if vol_mode == "data" and phi_data is not None:
            f_bl_data = pdf_from_phi_tilde(
                spectral_pdf_filter(phi_data[idx], chosen_keep),
                T,
                float(config.r),
                float(config.S0),
                float(config.K_max),
                K_row,
                D2,
            )

        # --- K-space ---
        if samples is not None and samples.size >= 2:
            q1, q99 = np.percentile(samples, [1, 99])
            use = samples[(samples >= q1) & (samples <= q99)]
            if use.size < 50:
                use = samples
            ax1.hist(
                use,
                bins=50,
                density=True,
                alpha=0.55,
                color=COLORS["mc_hist"],
                edgecolor="white",
                linewidth=0.5,
                label="MC",
            )
        ax1.plot(K_row, f_truth, color=COLORS["truth"], linewidth=2.5, label=truth_label, zorder=8)
        if f_bl_data is not None and use_mc_ref:
            ax1.plot(
                K_row,
                f_bl_data,
                color=COLORS["truth"],
                linewidth=1.8,
                linestyle="--",
                alpha=0.75,
                label="BL from φ_data (filtered)",
                zorder=7,
            )
        ax1.plot(K_row, f_rrr, color=COLORS["rrr"], linewidth=2, label="RRR (filtered)", zorder=9)
        ax1.plot(K_row, f_ql, color=COLORS["ql"], linewidth=2, label="RRR+QL (filtered)", zorder=9)
        if nn_f_by_T is not None:
            f_nn = nn_f_by_T.get(float(T_req))
            if f_nn is not None and len(f_nn) == len(K_row):
                ax1.plot(K_row, f_nn, color=COLORS["nn"], linewidth=2, linestyle="--", label="NN", zorder=10)
        ax1.set_xlabel(r"Strike $K$", fontsize=11, fontweight="bold")
        ax1.set_ylabel(r"$f(K)$", fontsize=11, fontweight="bold")
        ax1.set_title(f"K-space (T_req={T_req:.2f}, grid T={T:.3f})", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.35)
        ax1.set_facecolor(COLORS["background"])

        # --- log-K ---
        log_K = np.log(np.maximum(K_row, 1e-16))
        ax2.plot(log_K, f_truth * K_row, color=COLORS["truth"], linewidth=2.5, label=truth_label)
        ax2.plot(log_K, f_rrr * K_row, color=COLORS["rrr"], linewidth=2, label="RRR (filtered)")
        ax2.plot(log_K, f_ql * K_row, color=COLORS["ql"], linewidth=2, label="RRR+QL (filtered)")
        if samples is not None and samples.size >= 2:
            ax2.hist(
                np.log(use),
                bins=50,
                density=True,
                alpha=0.45,
                color=COLORS["kde"],
                edgecolor="white",
                linewidth=0.5,
                label="MC log K",
            )
        if nn_f_by_T is not None:
            f_nn = nn_f_by_T.get(float(T_req))
            if f_nn is not None and len(f_nn) == len(K_row):
                ax2.plot(log_K, f_nn * K_row, color=COLORS["nn"], linewidth=2, linestyle="--", label="NN")
        ax2.set_xlabel(r"$\ln K$", fontsize=11, fontweight="bold")
        ax2.set_ylabel(r"$f(\ln K)$", fontsize=11, fontweight="bold")
        ax2.set_title("Log-strike density", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.35)
        ax2.set_facecolor(COLORS["background"])

        # --- Gaussian (standardize using MC if available, else truth) ---
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
            ax3.hist(
                x_mc,
                bins=30,
                density=True,
                alpha=0.5,
                color=COLORS["mc_hist"],
                edgecolor="white",
                linewidth=0.5,
                label="MC standardized",
            )
            ax3.plot(x_range, kde_vals, color=COLORS["kde"], linewidth=2, label="MC KDE", zorder=9)
        else:
            ln_t = np.log(np.maximum(K_row, 1e-16))
            w = f_truth / (np.sum(f_truth) + 1e-16)
            mu_mc = float(np.sum(w * ln_t))
            sigma_mc = float(max(np.sqrt(np.sum(w * (ln_t - mu_mc) ** 2)), 1e-8))
            x_mc = (ln_t - mu_mc) / sigma_mc  # for title stats only
            kde_vals = np.zeros_like(x_range)

        g_truth = _density_to_mc_x_space(K_row, f_truth, mu_mc, sigma_mc, x_range)
        g_rrr = _density_to_mc_x_space(K_row, f_rrr, mu_mc, sigma_mc, x_range)
        g_ql = _density_to_mc_x_space(K_row, f_ql, mu_mc, sigma_mc, x_range)
        vt = np.isfinite(g_truth)
        if case == "constant":
            g_truth_label = "Truth g(x)"
        elif use_mc_ref:
            g_truth_label = "MC ref. g(x)"
        else:
            g_truth_label = "BL g(x) (sanity)"
        ax3.plot(
            x_range[vt],
            g_truth[vt],
            color=COLORS["truth"],
            linewidth=2.2,
            label=g_truth_label,
        )
        vr = np.isfinite(g_rrr)
        ax3.plot(x_range[vr], g_rrr[vr], color=COLORS["rrr"], linewidth=2, label="RRR g(x)")
        vq = np.isfinite(g_ql)
        ax3.plot(x_range[vq], g_ql[vq], color=COLORS["ql"], linewidth=2, label="RRR+QL g(x)")
        if nn_f_by_T is not None:
            f_nn = nn_f_by_T.get(float(T_req))
            if f_nn is not None and len(f_nn) == len(K_row):
                g_nn = _density_to_mc_x_space(K_row, f_nn, mu_mc, sigma_mc, x_range)
                vn = np.isfinite(g_nn)
                ax3.plot(x_range[vn], g_nn[vn], color=COLORS["nn"], linewidth=2, linestyle="--", label="NN g(x)")
        ax3.plot(x_range, std_normal, color=COLORS["normal"], linestyle=":", linewidth=2, label="N(0,1)")
        ax3.set_xlim(-3, 3)
        ax3.set_xlabel(r"$x = (\ln K - \mu)/\sigma$", fontsize=11, fontweight="bold")
        ax3.set_ylabel(r"$g(x)$", fontsize=11, fontweight="bold")
        sk = skew(x_mc) if samples is not None and samples.size >= 20 else float("nan")
        ku = kurtosis(x_mc, fisher=True) if samples is not None and samples.size >= 20 else float("nan")
        ax3.set_title(
            f"Gaussian coords (μ={mu_mc:.3f}, σ={sigma_mc:.3f}, skew={sk:.2f}, ex.kurt={ku:.2f})",
            fontsize=11,
            fontweight="bold",
        )
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(True, alpha=0.35)
        ax3.set_facecolor(COLORS["background"])

    plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=2.5, w_pad=1.8)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png = os.path.join(out_dir, f"pdf_analysis_mz_{ts}.png")
    pdf_path = os.path.join(out_dir, f"pdf_analysis_mz_{ts}.pdf")
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", facecolor="white")

    json_path = os.path.join(out_dir, f"pdf_analysis_summary_{ts}.json")
    blob = {
        "case": case,
        "case_label": case_label,
        "maturities_requested": [float(x) for x in maturities],
        "chosen_keep": int(chosen_keep),
        "pdf_metrics_by_maturity_interior_K": metrics_table,
        "pdf_analysis_png": png,
        "pdf_analysis_pdf": pdf_path,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)

    results: Dict[str, Any] = {
        **blob,
        "pdf_analysis_summary_json": json_path,
    }
    return fig, results
