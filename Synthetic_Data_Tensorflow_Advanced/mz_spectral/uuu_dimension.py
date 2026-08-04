#!/usr/bin/env python3
"""
Estimate effective dimension of the unresolved (UUU) subspace from truth φ̃ trajectories.

Pipeline:
  1. Build φ_U = P_U φ on the k̃ grid (ambient dim(U) = n_k − rank(P_R)).
  2. PCA / SVD on state snapshots → intrinsic dimension (energy threshold).
  3. FNN = feedforward autoencoder bottleneck scan (TensorFlow/Keras) on normalized φ_U.
  4. PCA on UUU forcing vectors L_UU φ_U (Markovian unresolved RHS).

Results are written to ``mz_uuu_dimension.json`` for use before fitting a UUU closure.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mz_spectral.fourier_dupire import (
    TKGrid,
    build_L_operator,
    build_second_derivative_matrix,
    eta_tilde_from_sigma,
)
from mz_spectral.mz_decomposition import low_pass_projector_matrix, split_operator_blocks


def ambient_dim_u(n_k: int, keep: int) -> int:
    """Number of unresolved Fourier modes in the Hermitian low-pass split."""
    if keep <= 0:
        return max(0, n_k - 1)
    if 2 * keep >= n_k:
        return 0
    return int(n_k - (2 * keep + 1))


def unresolved_orthonormal_basis(P_R: np.ndarray) -> np.ndarray:
    """
    Orthonormal columns spanning range(P_U) = ker(P_R) for symmetric projectors.

    Returns ``Q`` with shape (n_k, dim_u).
    """
    n_k = P_R.shape[0]
    P_U = np.eye(n_k) - P_R
    w, V = np.linalg.eigh(P_U)
    idx = w > 0.5
    Q = V[:, idx]
    if Q.shape[1] == 0:
        return np.zeros((n_k, 0))
    # QR for numerical orthonormality
    Q, _ = np.linalg.qr(Q)
    return Q


def phi_u_rows(phi_truth: np.ndarray, P_R: np.ndarray) -> np.ndarray:
    """Unresolved component φ_U = P_U φ for each time row (full grid, n_k)."""
    n_t, n_k = phi_truth.shape
    P_U = np.eye(n_k) - P_R
    out = np.zeros_like(phi_truth)
    for i in range(n_t):
        out[i] = P_U @ phi_truth[i]
    return out


def phi_u_compact_coords(phi_truth: np.ndarray, P_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coordinates of φ_U in an orthonormal unresolved basis.

    Returns ``(X, Q)`` with ``X`` shape (n_t, dim_u) and ``Q`` shape (n_k, dim_u).
    """
    Q = unresolved_orthonormal_basis(P_R)
    phi_u = phi_u_rows(phi_truth, P_R)
    if Q.shape[1] == 0:
        return np.zeros((phi_truth.shape[0], 0)), Q
    return phi_u @ Q, Q


def _dphi_dt_rows(phi: np.ndarray, t_tilde: np.ndarray) -> np.ndarray:
    """∂φ/∂t̃ by FD on uniform t̃ (same stencil as plot_vol_surfaces)."""
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


def precompute_l_uu_rows(
    grid: TKGrid,
    sigma: np.ndarray,
    T_max: float,
    keep: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """D2, P_R, and per-time L_UU blocks."""
    n_k = len(grid.k_tilde)
    D2 = build_second_derivative_matrix(n_k, grid.dk)
    P_R = low_pass_projector_matrix(n_k, keep)
    eta = eta_tilde_from_sigma(sigma, T_max)
    L_UU_rows: List[np.ndarray] = []
    for i in range(len(grid.T)):
        L_i = build_L_operator(eta[i], grid.k_tilde, D2)
        _, _, _, L_UU, _ = split_operator_blocks(L_i, P_R)
        L_UU_rows.append(L_UU)
    return D2, P_R, L_UU_rows


def pca_effective_dimension(
    X: np.ndarray,
    *,
    energy_threshold: float = 0.95,
    ridge: float = 1e-12,
) -> Dict[str, Any]:
    """
    SVD on centered snapshots X (n_samples × n_features).

    Returns rank at which cumulative squared singular values exceed ``energy_threshold``.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 2 or X.shape[1] < 1:
        return {
            "dim": 0,
            "ambient": int(X.shape[1]),
            "energy_threshold": energy_threshold,
            "singular_values_top": [],
            "cumulative_energy": [],
        }
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    e = s**2
    tot = float(np.sum(e)) + ridge
    cum = np.cumsum(e) / tot
    dim = int(np.searchsorted(cum, energy_threshold) + 1)
    dim = max(1, min(dim, len(s)))
    return {
        "dim": dim,
        "ambient": int(X.shape[1]),
        "energy_threshold": energy_threshold,
        "singular_values_top": s[: min(32, len(s))].tolist(),
        "cumulative_energy": cum[: min(32, len(cum))].tolist(),
    }


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    sig = np.where(sig < 1e-14, 1.0, sig)
    return (X - mu) / sig, mu, sig


def _bottleneck_candidates(ambient: int, max_d: int) -> List[int]:
    if ambient <= 1:
        return [1]
    raw = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    cap = min(max_d, ambient - 1, ambient)
    ds = sorted({d for d in raw if 1 <= d <= cap})
    if cap not in ds:
        ds.append(cap)
    return ds


def fnn_bottleneck_scan(
    X: np.ndarray,
    *,
    max_d: int = 128,
    energy_threshold: float = 0.95,
    rel_mse_tol: float = 0.05,
    epochs: int = 120,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 0,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Feedforward autoencoder (FNN) scan: smallest bottleneck d with val MSE within
    ``rel_mse_tol`` of the best (largest d) model.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)

    X = np.asarray(X, dtype=np.float32)
    n, ambient = X.shape
    if n < 4 or ambient < 2:
        return {
            "dim": max(1, min(ambient, 1)),
            "ambient": ambient,
            "skipped": True,
            "reason": "too_few_samples_or_features",
            "curve": {},
        }

    Xn, mu, sig = _standardize(X.astype(np.float64))
    Xn = Xn.astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(2, int(n * val_fraction))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    X_tr, X_val = Xn[tr_idx], Xn[val_idx]

    candidates = _bottleneck_candidates(ambient, max_d)
    curve: Dict[str, float] = {}
    best_mse = float("inf")
    best_d = candidates[-1]

    for d in candidates:
        inp = tf.keras.Input(shape=(ambient,), name="phi_u")
        h = tf.keras.layers.Dense(min(512, max(4 * d, 16)), activation="tanh")(inp)
        z = tf.keras.layers.Dense(d, activation="tanh", name="bottleneck")(h)
        h2 = tf.keras.layers.Dense(min(512, max(4 * d, 16)), activation="tanh")(z)
        out = tf.keras.layers.Dense(ambient, name="recon")(h2)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True
        )
        model.fit(
            X_tr,
            X_tr,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=min(batch_size, len(X_tr)),
            verbose=verbose,
            callbacks=[cb],
        )
        pred = model.predict(X_val, verbose=0)
        mse = float(np.mean((pred - X_val) ** 2))
        curve[str(d)] = mse
        if mse < best_mse:
            best_mse = mse
            best_d = d

    threshold = best_mse * (1.0 + rel_mse_tol)
    chosen = best_d
    for d in candidates:
        if curve[str(d)] <= threshold:
            chosen = d
            break

    return {
        "dim": int(chosen),
        "ambient": ambient,
        "best_bottleneck": int(best_d),
        "best_val_mse": best_mse,
        "rel_mse_tol": rel_mse_tol,
        "curve": curve,
        "epochs": epochs,
        "skipped": False,
    }


def _grid_to_compact(rows: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q.shape[1] == 0:
        return np.zeros((rows.shape[0], 0))
    return rows @ Q


def uuu_forcing_rows(
    phi_truth: np.ndarray,
    t_tilde: np.ndarray,
    L_UU_rows: List[np.ndarray],
    P_R: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """Markovian UUU forcing L_UU φ_U in compact unresolved coordinates."""
    phi_u = phi_u_rows(phi_truth, P_R)
    n_t = phi_truth.shape[0]
    full = np.zeros_like(phi_truth)
    for i in range(n_t):
        full[i] = L_UU_rows[i] @ phi_u[i]
    return _grid_to_compact(full, Q)


def uuu_markov_residual_rows(
    phi_truth: np.ndarray,
    t_tilde: np.ndarray,
    L_UU_rows: List[np.ndarray],
    P_R: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """P_U (dφ/dt̃) − L_UU φ_U in compact unresolved coordinates."""
    phi_u = phi_u_rows(phi_truth, P_R)
    dphi = _dphi_dt_rows(phi_truth, t_tilde)
    n_k = phi_truth.shape[1]
    P_U = np.eye(n_k) - P_R
    n_t = phi_truth.shape[0]
    full = np.zeros_like(phi_truth)
    for i in range(n_t):
        full[i] = P_U @ dphi[i] - L_UU_rows[i] @ phi_u[i]
    return _grid_to_compact(full, Q)


def estimate_uuu_dimension(
    phi_truth: np.ndarray,
    grid: TKGrid,
    sigma: np.ndarray,
    T_max: float,
    keep: int,
    *,
    energy_threshold: float = 0.95,
    fnn_max_d: int = 128,
    fnn_epochs: int = 120,
    run_fnn: bool = True,
    rel_mse_tol: float = 0.05,
    seed: int = 0,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Full UUU dimension report: ambient size, PCA state/forcing ranks, optional FNN bottleneck.
    """
    n_k = phi_truth.shape[1]
    keep = int(keep)
    d_amb_formula = ambient_dim_u(n_k, keep)
    _, P_R, L_UU_rows = precompute_l_uu_rows(grid, sigma, T_max, keep)

    X_state, Q = phi_u_compact_coords(phi_truth, P_R)
    d_amb = int(X_state.shape[1])
    pca_state = pca_effective_dimension(X_state, energy_threshold=energy_threshold)

    X_force = uuu_forcing_rows(phi_truth, grid.t_tilde, L_UU_rows, P_R, Q)
    pca_force = pca_effective_dimension(X_force, energy_threshold=energy_threshold)

    X_res = uuu_markov_residual_rows(phi_truth, grid.t_tilde, L_UU_rows, P_R, Q)
    pca_residual = pca_effective_dimension(X_res, energy_threshold=energy_threshold)

    fnn: Dict[str, Any]
    if run_fnn and d_amb >= 2 and X_state.shape[0] >= 4:
        fnn = fnn_bottleneck_scan(
            X_state,
            max_d=fnn_max_d,
            energy_threshold=energy_threshold,
            rel_mse_tol=rel_mse_tol,
            epochs=fnn_epochs,
            seed=seed,
            verbose=verbose,
        )
    else:
        fnn = {
            "dim": pca_state["dim"],
            "ambient": d_amb,
            "skipped": True,
            "reason": "fnn_disabled_or_degenerate_U",
        }

    # Heuristic keep suggestion: retain enough R modes that dim(U) ~ 2–4× effective rank
    d_eff = int(
        max(
            pca_state["dim"],
            fnn.get("dim", 0) if not fnn.get("skipped") else 0,
            pca_force["dim"],
        )
    )
    keep_suggest = max(keep, min(n_k // 2 - 1, max(4, (n_k - 2 * d_eff - 1) // 2)))

    return {
        "n_k": n_k,
        "chosen_keep": keep,
        "ambient_dim_u": d_amb,
        "ambient_dim_u_formula": d_amb_formula,
        "pca_phi_u": pca_state,
        "pca_l_uu_phi_u": pca_force,
        "pca_markov_residual_u": pca_residual,
        "fnn_autoencoder": fnn,
        "d_eff_recommended": d_eff,
        "keep_suggest_from_d_eff": int(keep_suggest),
        "note": (
            "d_eff is max(PCA state, FNN bottleneck, PCA L_UU phi_U). "
            "keep_suggest is heuristic only; does not override CLI chosen_keep."
        ),
    }


def save_uuu_dimension_report(out_dir: str, report: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mz_uuu_dimension.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path
