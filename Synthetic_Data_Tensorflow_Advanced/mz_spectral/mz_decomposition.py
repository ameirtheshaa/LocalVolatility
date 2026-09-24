#!/usr/bin/env python3
"""
Mori–Zwanzig split on a uniform k̃ grid using an orthogonal low-pass projector
in the unitary FFT basis.

Resolved (R): frequencies with index |m| ≤ cutoff (Hermitian-symmetric mask).
Unresolved (U): complement.

Operator blocks (Galerkin / Liouville partitioning):

    L_RR = P_R L P_R,   L_RU = P_R L P_U,   L_UR = P_U L P_R,   L_UU = P_U L P_U

RRR dynamics (Markovian resolved-only): dφ_R/dt ≈ L_RR φ_R when φ_U is neglected.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _hermitian_low_pass_mask(n: int, keep: int) -> np.ndarray:
    """
    Binary mask for unitary FFT, length n, symmetric around DC.

    `keep` is the number of *positive* frequency bins to retain (excluding DC),
    i.e. retain indices 0,1,...,keep and n-keep,...,n-1 (when n > 2*keep).
    """
    if keep < 0:
        raise ValueError("keep must be non-negative")
    if keep == 0:
        out = np.zeros(n)
        out[0] = 1.0
        return out
    if 2 * keep >= n:
        return np.ones(n)
    mask = np.zeros(n)
    mask[: keep + 1] = 1.0
    mask[n - keep :] = 1.0
    return mask


def low_pass_projector_matrix(n: int, keep: int) -> np.ndarray:
    """
    Orthogonal projector P_R onto low Fourier modes (unitary FFT).

    `keep`: retain DC and `keep` lowest positive frequencies and their negatives.
    """
    mask = _hermitian_low_pass_mask(n, keep)
    P = np.zeros((n, n), dtype=float)
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        col = np.fft.ifft(np.fft.fft(ej, norm="ortho") * mask, norm="ortho").real
        P[:, j] = col
    return P


def split_operator_blocks(
    L: np.ndarray, P_R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (L_RR, L_RU, L_UR, L_UU, P_U) with P_U = I - P_R.
    """
    n = L.shape[0]
    I = np.eye(n)
    P_U = I - P_R
    L_RR = P_R @ L @ P_R
    L_RU = P_R @ L @ P_U
    L_UR = P_U @ L @ P_R
    L_UU = P_U @ L @ P_U
    return L_RR, L_RU, L_UR, L_UU, P_U


def apply_projector(P: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return P @ phi


def resolved_forcing_from_unresolved(L_RU: np.ndarray, phi_U: np.ndarray) -> np.ndarray:
    """P_R L P_U φ_U = L_RU φ_U (backscatter onto resolved subspace)."""
    return L_RU @ phi_U
