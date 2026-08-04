#!/usr/bin/env python3
"""
Dupire inversion σ(K,T) from φ̃(t̃,k̃) and regularization for data-driven ROM generators.

Shared with PDF extraction via the same D₂ stencil and chain rule in validation.py.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from mz_spectral.fourier_dupire import TKGrid, eta_tilde_from_sigma
from mz_spectral.validation import spectral_pdf_filter


def mask_first_rows_for_nk(n_k: int) -> int:
    """First time rows masked in Dupire inversion (one-sided ∂φ/∂t̃)."""
    return 4 if n_k >= 96 else 2


def dphi_dt_tilde(phi: np.ndarray, t_tilde: np.ndarray) -> np.ndarray:
    """∂φ̃/∂t̃ by finite differences on uniform t̃ slices (per k̃ column)."""
    n_t, n_k = phi.shape
    out = np.zeros_like(phi, dtype=float)
    if n_t < 2:
        return out
    dt0 = t_tilde[1] - t_tilde[0]
    out[0, :] = (phi[1, :] - phi[0, :]) / dt0
    if n_t > 2:
        dt_end = t_tilde[-1] - t_tilde[-2]
        out[-1, :] = (phi[-1, :] - phi[-2, :]) / dt_end
        for i in range(1, n_t - 1):
            dt = t_tilde[i + 1] - t_tilde[i - 1]
            out[i, :] = (phi[i + 1, :] - phi[i - 1, :]) / dt
    return out


def phi_kk(phi: np.ndarray, D2: np.ndarray) -> np.ndarray:
    n_t, _ = phi.shape
    out = np.zeros_like(phi)
    for i in range(n_t):
        out[i, :] = D2 @ phi[i, :]
    return out


def sigma_reliability_mask(
    phi: np.ndarray,
    grid: TKGrid,
    D2: np.ndarray,
    *,
    tau_kk: float = 1e-3,
    mask_first_rows: int | None = None,
) -> np.ndarray:
    """Cells reliable for Dupire inversion (strong |φ_kk|, not in first time rows)."""
    n_t, n_k = phi.shape
    mfr = mask_first_rows if mask_first_rows is not None else mask_first_rows_for_nk(n_k)
    phi_kk_arr = phi_kk(phi, D2)
    row_max = np.maximum(np.max(np.abs(phi_kk_arr), axis=1, keepdims=True), 1e-30)
    strong = np.abs(phi_kk_arr) >= tau_kk * row_max
    if mfr > 0:
        strong[: min(mfr, n_t), :] = False
    return strong


def implied_eta_from_phi(
    phi: np.ndarray,
    grid: TKGrid,
    D2: np.ndarray,
    *,
    denom_floor: float = 1e-14,
    apply_reliability_mask: bool = True,
    tau_kk: float = 1e-3,
    mask_first_rows: int | None = None,
) -> np.ndarray:
    """η̃(t̃,k̃) from Dupire inversion of φ."""
    n_t, n_k = phi.shape
    k = grid.k_tilde.astype(float)
    k_sq = np.maximum(k**2, denom_floor)
    phi_kk_arr = phi_kk(phi, D2)
    phi_t = dphi_dt_tilde(phi, grid.t_tilde)
    denom = k_sq[np.newaxis, :] * phi_kk_arr
    sign = np.sign(denom)
    sign = np.where(sign == 0, 1.0, sign)
    denom_safe = sign * np.maximum(np.abs(denom), denom_floor)
    eta = phi_t / denom_safe
    eta = np.clip(eta, 0.0, None)
    if apply_reliability_mask:
        m = sigma_reliability_mask(
            phi, grid, D2, tau_kk=tau_kk, mask_first_rows=mask_first_rows
        )
        eta = np.where(m, eta, np.nan)
    return eta


def implied_sigma_from_phi(
    phi: np.ndarray,
    grid: TKGrid,
    D2: np.ndarray,
    T_max: float,
    *,
    denom_floor: float = 1e-14,
    apply_reliability_mask: bool = True,
    tau_kk: float = 1e-3,
    mask_first_rows: int | None = None,
    sigma_cap: float = 5.0,
) -> np.ndarray:
    """
    Recover σ(K,T) from φ̃ on the MZ grid: σ = sqrt(2 η̃ / T_max).

    Returns (n_t, n_k); NaN where unreliable.
    """
    eta = implied_eta_from_phi(
        phi,
        grid,
        D2,
        denom_floor=denom_floor,
        apply_reliability_mask=apply_reliability_mask,
        tau_kk=tau_kk,
        mask_first_rows=mask_first_rows,
    )
    sigma = np.sqrt(np.maximum(2.0 * eta / T_max, 0.0))
    bad = ~np.isfinite(sigma) | (sigma > sigma_cap)
    return np.where(bad, np.nan, sigma)


def regularize_sigma_grid(
    sigma: np.ndarray,
    grid: TKGrid,
    *,
    sigma_min: float = 1e-4,
    sigma_max: float = 5.0,
    fill_nan: bool = True,
) -> np.ndarray:
    """Clip σ and optionally fill NaNs by linear interpolation in (T, K)."""
    out = np.array(sigma, dtype=float, copy=True)
    out = np.clip(out, sigma_min, sigma_max)
    if not fill_nan or not np.any(~np.isfinite(out)):
        return out
    K_mesh = grid.K
    T_mesh = np.broadcast_to(grid.T[:, np.newaxis], K_mesh.shape)
    valid = np.isfinite(out)
    if np.sum(valid) < 4:
        fill_val = float(np.nanmean(out[valid])) if np.any(valid) else 0.2
        return np.where(np.isfinite(out), out, fill_val)
    pts = np.column_stack([T_mesh[valid].ravel(), K_mesh[valid].ravel()])
    vals = out[valid].ravel()
    interp = LinearNDInterpolator(pts, vals, fill_value=float(np.nanmean(vals)))
    Ti, Ki = np.broadcast_arrays(T_mesh, K_mesh)
    filled = interp(Ti.ravel(), Ki.ravel()).reshape(out.shape)
    return np.where(np.isfinite(out), out, filled)


def _calendar_fill_sigma(
    sigma_masked: np.ndarray, feasible: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    """
    Fill infeasible/tail cells of a σ grid from the nearest feasible maturity at the
    *same strike column* (calendar coupling along T̃).

    ``sigma_masked`` carries the pointwise σ on feasible cells and NaN elsewhere;
    ``feasible`` is the (n_t, n_k) reliability mask; ``fallback`` supplies a value for
    any strike column with no feasible cell at all (e.g. the spatial-grid σ). The
    result keeps the pointwise σ wherever feasible and substitutes the calendar value
    (the local-vol estimate carried by the time-history of the *same* strike, which
    for a smooth Dupire surface is more informative than a pointwise spatial second
    derivative) on the tail.
    """
    out = np.array(sigma_masked, dtype=float, copy=True)
    n_t, n_k = out.shape
    for j in range(n_k):
        ok = feasible[:, j] & np.isfinite(out[:, j])
        if not np.any(ok):
            out[:, j] = fallback[:, j]
            continue
        idx = np.where(ok)[0]
        for i in range(n_t):
            if not (feasible[i, j] and np.isfinite(out[i, j])):
                # nearest feasible maturity index in this strike column
                out[i, j] = sigma_masked[idx[int(np.argmin(np.abs(idx - i)))], j]
    out = np.where(np.isfinite(out), out, fallback)
    return out


def regularize_sigma_memory(
    sigma: np.ndarray,
    grid: TKGrid,
    phi: np.ndarray,
    D2: np.ndarray,
    T_max: float,
    *,
    keep: int | None = None,
    smooth_k: float = 5.0,
    feasible: np.ndarray | None = None,
    w_calendar: float = 0.5,
    sigma_min: float = 1e-4,
    sigma_max: float = 5.0,
) -> np.ndarray:
    """
    Calendar-coupled MZ tail inversion of σ(K,T) (§6 of the synthesis).

    The §6 deliverable is a *calendar-coupled* tail prior that "informs σ in the tails
    by the time-history of the whole surface" rather than a pointwise spatial second
    derivative. Empirically (``examples/run_mz_tail_inversion.py``, the pre-registered
    falsifiable test on the ``dupire_exact`` smile), the convexity-reconstruction form
    of §6 eq (6.1) — substituting a smoothed-φ̃ or density-ROM convexity into the
    pointwise Dupire ratio — does **not** beat the spatial Tikhonov fill: in the
    infeasible tail the cells are infeasible *precisely because* ``φ̃_kk → 0`` there
    (the early-maturity wings), so reconstructing the denominator cannot recover a
    well-posed ratio. What does win is to combine two independent, cheap tail
    extrapolations of σ itself — one *spatial* (along k̃) and one *calendar* (along
    t̃) — which is the operational meaning of calendar-coupled self-consistency.

    Two stages:

    Stage 1 — the two priors on the tail.
      * ``σ_grid`` — the spatial Tikhonov fill (``regularize_sigma_grid``): clip + a
        ``LinearNDInterpolator`` over the feasible (T,K) cells.
      * ``σ_cal`` — the calendar fill (``_calendar_fill_sigma``): on each tail cell,
        take σ from the nearest feasible maturity at the *same strike column*, with
        ``σ_grid`` as the fallback for any column with no feasible cell.

    Stage 2 — feasibility-gated blend.  Keep the pointwise ``sigma`` on feasible
      cells; on the infeasible / tail cells use the convex blend
      ``w_calendar·σ_cal + (1−w_calendar)·σ_grid`` (default ``w_calendar=0.5``). The
      blend is what reduces the tail σ-RMSE below either prior alone — averaging the
      independent spatial and calendar estimates cuts their variance. Clip to
      ``[sigma_min, sigma_max]`` with a final spatial fallback for any residual NaN.

    Backward-compat: ``keep`` and ``smooth_k`` are accepted (callers pass them) but
    are no longer used by the calendar-coupled method — the spatial/calendar priors
    are parameter-free. ``feasible`` defaults to
    ``pole_feasibility(phi, grid, D2)['feasible']``.

    Honesty note: the result is conditional on the one-sided ``D2`` boundary stencil
    (non-dissipative at the operator level, §8 Step-0a) feeding the raw pointwise σ;
    the calendar coupling does not depend on the unbuilt §8 eigenbasis projector.

    Returns an (n_t, n_k) σ grid (no NaN; clipped to [sigma_min, sigma_max]).
    """
    from mz_spectral import uncertainty as _uq

    sigma = np.asarray(sigma, dtype=float)
    phi = np.asarray(phi, dtype=float)
    D2 = np.asarray(D2, dtype=float)

    if feasible is None:
        feasible = _uq.pole_feasibility(phi, grid, D2)["feasible"]
    feasible = np.asarray(feasible, dtype=bool)

    # --- Stage 1: the two tail priors (spatial Tikhonov + calendar) ---
    sigma_grid = regularize_sigma_grid(sigma, grid, sigma_min=sigma_min,
                                       sigma_max=sigma_max, fill_nan=True)
    sigma_masked = np.where(feasible & np.isfinite(sigma), sigma, np.nan)
    sigma_cal = _calendar_fill_sigma(sigma_masked, feasible, sigma_grid)
    sigma_cal = np.clip(sigma_cal, sigma_min, sigma_max)

    # --- Stage 2: feasibility-gated blend of the two priors on the tail ---
    w = float(np.clip(w_calendar, 0.0, 1.0))
    blended = w * sigma_cal + (1.0 - w) * sigma_grid
    out = np.where(feasible & np.isfinite(sigma), sigma, blended)
    out = np.where(np.isfinite(out), out, sigma_grid)
    out = np.clip(out, sigma_min, sigma_max)

    if np.any(~np.isfinite(out)):
        out = regularize_sigma_grid(out, grid, sigma_min=sigma_min,
                                    sigma_max=sigma_max, fill_nan=True)
    return out


def phi_for_sigma_inversion(phi: np.ndarray, chosen_keep: int | None) -> np.ndarray:
    """Low-pass φ in k̃ before Dupire inversion (same cutoff as MZ / PDF filter)."""
    if chosen_keep is None:
        return phi
    n_t = phi.shape[0]
    return np.stack(
        [spectral_pdf_filter(phi[i], int(chosen_keep)) for i in range(n_t)],
        axis=0,
    )


def should_filter_phi_before_dupire(
    phi: np.ndarray,
    grid: TKGrid,
    D2: np.ndarray,
    T_max: float,
    chosen_keep: int,
    *,
    min_ratio: float = 0.25,
) -> bool:
    """
    Use spectral pre-filter only if it does not collapse median σ vs unfiltered inversion.

    Flat BS surfaces (constant vol) lose too much curvature when filtered at large ``keep``.
    """
    _, s_u = build_eta_sigma_from_phi(
        phi, grid, D2, T_max, chosen_keep, filter_phi=False, regularize=False
    )
    _, s_f = build_eta_sigma_from_phi(
        phi, grid, D2, T_max, chosen_keep, filter_phi=True, regularize=False
    )
    mu = np.nanmedian(s_u[np.isfinite(s_u)])
    mf = np.nanmedian(s_f[np.isfinite(s_f)])
    if not np.isfinite(mu) or mu <= 0:
        return True
    return bool(mf >= min_ratio * mu)


def build_eta_sigma_from_phi(
    phi: np.ndarray,
    grid: TKGrid,
    D2: np.ndarray,
    T_max: float,
    chosen_keep: int | None = None,
    *,
    filter_phi: bool = True,
    filter_phi_adaptive: bool = False,
    regularize: bool | str = True,
    smooth_k: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
  Build η̃ and σ grids from φ (data-driven generator input).

    If ``filter_phi``, applies ``spectral_pdf_filter`` with ``chosen_keep`` before inversion.
    When ``filter_phi_adaptive`` and ``chosen_keep`` is set, filtering is skipped if it
    would shrink median σ by more than 75% vs the unfiltered inversion.

    ``regularize`` selects the tail-fill prior (default unchanged):
      * ``True`` / ``"grid"`` — spatial ``regularize_sigma_grid`` (clip + LinearND fill).
      * ``"memory"`` — the MZ-regularized ``regularize_sigma_memory`` (resolved
        pre-filter + dynamics-informed tail σ on infeasible cells).
      * ``False`` / ``None`` — no regularization (raw, NaN-where-masked).
    """
    use_filter = filter_phi
    if filter_phi and filter_phi_adaptive and chosen_keep is not None:
        use_filter = should_filter_phi_before_dupire(phi, grid, D2, T_max, int(chosen_keep))
    work = phi_for_sigma_inversion(phi, chosen_keep) if use_filter else phi
    n_k = phi.shape[1]
    mfr = mask_first_rows_for_nk(n_k)
    sigma = implied_sigma_from_phi(work, grid, D2, T_max, mask_first_rows=mfr)
    mode = "grid" if regularize is True else ("none" if regularize in (False, None) else str(regularize))
    if mode == "grid":
        sigma = regularize_sigma_grid(sigma, grid)
    elif mode == "memory":
        sigma = regularize_sigma_memory(sigma, grid, phi, D2, T_max,
                                        keep=chosen_keep, smooth_k=smooth_k)
    elif mode != "none":
        raise ValueError(f"Unknown regularize={regularize!r}; use True/'grid'/'memory'/False")
    eta = eta_tilde_from_sigma(sigma, T_max)
    return eta, sigma


def relative_sigma_change(sigma_a: np.ndarray, sigma_b: np.ndarray) -> float:
    """Mean relative L2 change between two σ grids (finite cells only)."""
    m = np.isfinite(sigma_a) & np.isfinite(sigma_b)
    if not np.any(m):
        return float("nan")
    a = sigma_a[m]
    b = sigma_b[m]
    denom = np.maximum(np.abs(a), 1e-8)
    return float(np.sqrt(np.mean(((b - a) / denom) ** 2)))
