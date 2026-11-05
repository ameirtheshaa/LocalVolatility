"""Shared evaluation core for s3 (diagnose), s5 (score) and later s6 (cost).

Pure NumPy/SciPy.  Implements PLAN.md section 3 (uncertainty), section 6 (protocols), section 8 (metrics) on top of
``scc.metrics``: level loading, reference rows on the 256 model grid (own-grid density and RHS, then restricted),
the ``EvalBundle`` that caches everything a case needs, ``score_trajectory`` (reduced models and oracles),
``score_level_rows`` (full solves), ``uncertainty`` and the shared row runners used by both scripts.

Conventions
-----------
* Level roles: ``same`` = Full(n)@sub (same-grid truth), ``refined`` = Full(2n-1)@sub, ``step`` = Full(n)@2sub,
  ``refined2`` = Full(4n-3)@sub (sparse), ``step2`` = Full(n)@4sub (sparse).  For n=256, sub=16 these are
  n256_s16, n511_s16, n256_s32, n1021_s16, n256_s64.  Smoke runs use n=64, sub=4 (64/127/253, 4/8/16).
* ``same_``-prefixed metric keys are the scores against the same-grid truth; unprefixed ones are against the refined
  truth (the primary verdict quantity).
* Windows: P0 = fine indices 0..N, P1 = fine indices i0..N.  ``score_trajectory`` receives the window rows; the 24
  output indices inside the window are ``out_idx - start``.
"""
from __future__ import annotations

import os
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import scc
from scc import conventions as cv
from scc import identity
from scc import metrics as mt
from scc import oracles as orc
from scc import reference as rf
from scc import targets as tgt
from scc import closures as cl
from scc.closures import ModelSpec
from dsm import operators as dop

METRIC_VERSION = "evaluate_v1"
EARLY_FRACTIONS = (0.05, 0.10, 0.15)
EARLY_LABELS = ("0.05", "0.10", "0.15")
ROLES = ("same", "refined", "step", "refined2", "step2")
REQUIRED_ROLES = ("same", "refined", "step")


# ----------------------------------------------------------------------------- levels
def level_table(n: int = 256, sub: int = 16) -> Dict[str, dict]:
    """role -> dict(name, n, sub, solver).  Names follow RESULT_SCHEMA ('n256_s16', ...)."""
    spec = {"same": (n, sub, "dense"), "refined": (2 * n - 1, sub, "dense"), "step": (n, 2 * sub, "dense"),
            "refined2": (4 * (n - 1) + 1, sub, "sparse"), "step2": (n, 4 * sub, "sparse")}
    return {r: {"name": f"n{a}_s{b}", "n": a, "sub": b, "solver": c} for r, (a, b, c) in spec.items()}


def _cached_reference(case, n: int, sub: int, solver: str) -> Optional[dict]:
    """Cache lookup without computing (mirrors the return value of ``reference.get_reference``)."""
    identity.assert_phase_allows(case.split)
    ident = rf.reference_identity(case, n, sub, solver)
    z = identity.cache_load("reference", ident)
    if z is None or not {"phi", "t", "h", "i0", "out_idx"} <= set(z):
        return None
    return {"phi": z["phi"], "t": z["t"], "h": float(z["h"]), "i0": int(z["i0"]), "out_idx": z["out_idx"],
            "identity": ident, "from_cache": True, "seconds": 0.0}


def load_levels(case, levels: Optional[Sequence[str]] = None, generate: bool = False, n: int = 256,
                sub: int = 16) -> Dict[str, dict]:
    """Return {level_name: reference dict}.  ``levels`` may list level names ('n511_s16') or roles ('refined');
    default = all five.  Levels missing from the cache are skipped unless ``generate`` (then
    ``scc.reference.get_reference`` computes and caches them).  Each returned dict gains keys role, n, sub."""
    tab = level_table(n, sub)
    by_name = {v["name"]: (r, v) for r, v in tab.items()}
    want = list(levels) if levels is not None else list(ROLES)
    out: Dict[str, dict] = {}
    for item in want:
        if item in tab:
            role, v = item, tab[item]
        elif item in by_name:
            role, v = by_name[item]
        else:
            raise KeyError(f"unknown level {item!r}")
        ref = (rf.get_reference(case, v["n"], v["sub"], v["solver"]) if generate
               else _cached_reference(case, v["n"], v["sub"], v["solver"]))
        if ref is None:
            continue
        ref = dict(ref)
        ref.update(role=role, n=v["n"], sub=v["sub"])
        out[v["name"]] = ref
    return out


def _role_of(ref: dict, n: int, sub: int) -> Optional[str]:
    n_own = ref["phi"].shape[1]
    sub_own = int(ref["out_idx"][1] - ref["out_idx"][0])
    if n_own == n and sub_own == sub:
        return "same"
    if n_own == 2 * n - 1 and sub_own == sub:
        return "refined"
    if n_own == n and sub_own == 2 * sub:
        return "step"
    if n_own == 4 * (n - 1) + 1 and sub_own == sub:
        return "refined2"
    if n_own == n and sub_own == 4 * sub:
        return "step2"
    return None


def reference_rows(case, ref: dict, grid256, fine_idx: Sequence[int]) -> Dict[str, np.ndarray]:
    """Rows of one reference at ``fine_idx`` (indices on THAT reference's own fine-time axis), on the 256 grid.

    phi = ref phi restricted by nested index; f = Breeden-Litzenberger density computed on the reference's own grid
    (own ``d2_repo`` through ``metrics.density_rows``) then restricted; phi_t = own-grid RHS ``L_full_dense(case,
    grid_n, t) @ phi`` (autonomous cases: the AUTONOMOUS_T operator) then restricted.
    Time-index mapping: levels with the base sub share indices; sub*2 -> 2*i, sub*4 -> 4*i (done by the caller).
    """
    fine_idx = np.asarray(fine_idx, int)
    phi_fine = ref["phi"]
    n_own = phi_fine.shape[1]
    grid_own = grid256 if n_own == grid256.n_k else rf.make_grid(n_own)
    idx = rf.nested_index(n_own, grid256.n_k)
    t_rows = np.asarray(ref["t"], float)[fine_idx]
    T_rows = t_rows * grid_own.T_max
    phi_own = phi_fine[fine_idx]
    ctx_own = mt.make_context(grid_own, T_rows=T_rows)
    f_own = mt.density_rows(phi_own, ctx_own, "d2_repo")
    if not case.time_dependent:
        L = rf.L_full_dense(case, grid_own, cv.AUTONOMOUS_T)
        phit_own = phi_own @ L.T
    else:
        phit_own = np.array([rf.L_full_dense(case, grid_own, float(t)) @ p for t, p in zip(t_rows, phi_own)])
    return {"phi": phi_own[:, idx], "f": f_own[:, idx], "phi_t": phit_own[:, idx], "t": t_rows}


# ----------------------------------------------------------------------------- small helpers
def interior_rms(x: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(x, float)
    sel = x[mask]
    return float(np.sqrt(np.mean(sel ** 2))) if sel.size else float("nan")


def interior_l2_rows(fa: np.ndarray, fb: np.ndarray, ctx: mt.EvalContext) -> np.ndarray:
    """sqrt(int_interior (fa-fb)^2 dK) per row (trapezoid in K), same convention as density_metrics."""
    out = []
    for i in range(fa.shape[0]):
        K, m = ctx.K_rows[i], ctx.interior[i]
        out.append(float(np.sqrt(mt._trapz((fa[i] - fb[i])[m] ** 2, K[m]))))
    return np.array(out)


def _neg_mass(f: np.ndarray, ctx: mt.EvalContext) -> np.ndarray:
    return np.array([float(mt._trapz(np.maximum(-f[i], 0.0), ctx.K_rows[i])) for i in range(f.shape[0])])


def _prefix(d: dict, p: str) -> dict:
    return {p + k: v for k, v in d.items()}


def _log2_ratio(a: float, b: float) -> Optional[float]:
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return None
    return float(np.log2(a / b))


def sha_of_file(path: str) -> str:
    return identity.sha256_file(path)[:16]


def evaluate_identity_hash() -> str:
    return sha_of_file(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- A(t) factories
def A_mid_factory(sys, V_int: np.ndarray, lam_V: np.ndarray, t: np.ndarray) -> Callable[[int], np.ndarray]:
    """step_index -> A_structured at t_mid = (t[n]+t[n+1])/2 (cached; autonomous: assembled once)."""
    t = np.asarray(t, float)
    cache: Dict[int, np.ndarray] = {}

    def f(n: int) -> np.ndarray:
        key = n if sys.time_dependent else 0
        if key not in cache:
            cache[key] = cv.A_structured(sys, 0.5 * (t[n] + t[n + 1]), V_int, lam_V)
        return cache[key]
    return f


def A_at_times(sys, V_int: np.ndarray, lam_V: np.ndarray, times: Sequence[float]) -> List[np.ndarray]:
    """List of A_structured(t) at the given scaled times (autonomous: one assembly repeated)."""
    times = list(np.atleast_1d(times))
    if not sys.time_dependent:
        A = cv.A_structured(sys, float(times[0]), V_int, lam_V)
        return [A for _ in times]
    return [cv.A_structured(sys, float(t), V_int, lam_V) for t in times]


# ----------------------------------------------------------------------------- model keys / feedback
def model_key(closure: str, m: int, n_h: int = 0, protocol: str = "P0", hidden: Optional[str] = None) -> str:
    """'<closure>_m<m>[_h<n_h>]__<protocol>[_<hidden>]' (RESULT_SCHEMA); _h only for LR0/LR, hidden only for LR*
    under P1."""
    s = f"{closure}_m{m}"
    if closure in ("LR0", "LR") and n_h:
        s += f"_h{n_h}"
    s += f"__{protocol}"
    if hidden and closure in ("LR0", "LR") and protocol == "P1":
        s += f"_{hidden}"
    return s


def feedback_fn(spec: ModelSpec, lam_V: Optional[np.ndarray] = None) -> Callable:
    """(c_rows, h_rows) -> M_hat rows: RRR 0, QL nu*lam*c, LM c H^T (+b), LR c H^T + h R^T."""
    cl_ = spec.closure
    if cl_ == "RRR":
        return lambda c, h: np.zeros_like(c)
    if cl_ == "QL":
        nu = float(spec.params["nu"])
        lam = np.asarray(lam_V, float)
        return lambda c, h: nu * c * lam[None, :]
    if cl_ == "LM":
        H = np.asarray(spec.params["H"], float)
        b = spec.params.get("b")
        bb = None if b is None else np.asarray(b, float)
        return lambda c, h: c @ H.T + (0.0 if bb is None else bb[None, :])
    H = np.asarray(spec.params["H"], float)
    R = np.asarray(spec.params["R"], float)
    return lambda c, h: c @ H.T + h @ R.T


# ----------------------------------------------------------------------------- bundle
@dataclass
class EvalBundle:
    case: object
    grid: object
    n: int
    sub: int
    t: np.ndarray
    h: float
    i0: int
    N: int
    out_idx: np.ndarray
    early_idx: np.ndarray
    ctx24: mt.EvalContext
    ctx_early: mt.EvalContext
    sys: cv.InteriorSystem
    ell: np.ndarray
    psi_int: np.ndarray                 # (N+1, n-2) detrended same-grid reference, interior
    refs: Dict[str, dict]               # role -> reference dict (full fine rows on own grid)
    rows: Dict[str, dict]               # role -> {phi,f,phi_t,t} at the 24 outputs on the 256 grid
    rows_early: Dict[str, dict]         # role -> same at the early indices (same, refined only)
    ident_mask: np.ndarray
    sigma24: np.ndarray
    bs24: Optional[np.ndarray]
    fd16_vs_fd32_rms: float
    levels_present: List[str]
    _basis: dict = field(default_factory=dict)
    _targets: dict = field(default_factory=dict)
    _Afull: dict = field(default_factory=dict)
    _tf0: dict = field(default_factory=dict)
    _fine: dict = field(default_factory=dict)

    # ---- basis / reference states
    def basis(self, m: int, need_U: bool = False) -> dict:
        b = self._basis.get(m)
        if b is None:
            V_full = cv.dst_basis(self.n, m)
            b = {"V_full": V_full, "V_int": np.ascontiguousarray(V_full[1:-1]),
                 "lam_V": cv.sine_eigenvalues(self.n, self.grid.dk, np.arange(1, m + 1))}
            self._basis[m] = b
        if need_U and "U_int" not in b:
            b["U_int"] = cv.dst_complement_interior(self.n, m)
        return b

    def c_ref(self, m: int) -> np.ndarray:
        """(N+1, m) = psi_int @ V_int at every fine time of the same-grid reference."""
        b = self.basis(m)
        if "c_ref" not in b:
            b["c_ref"] = self.psi_int @ b["V_int"]
        return b["c_ref"]

    def cref_norm_max(self, m: int) -> float:
        return float(np.max(np.linalg.norm(self.c_ref(m), axis=1)))

    def targets(self, m: int) -> dict:
        if m not in self._targets:
            self._targets[m] = tgt.build_targets(self.case, self.refs_same, self.grid, m)
        return self._targets[m]

    @property
    def refs_same(self) -> dict:
        return self.refs["same"]

    def A_mid(self, m: int, start: int = 0) -> Callable[[int], np.ndarray]:
        """Cached midpoint A(t) for the fine steps (index n+start); autonomous: one assembly."""
        b = self.basis(m)
        if not self.sys.time_dependent:
            if "A_aut" not in b:
                b["A_aut"] = cv.A_structured(self.sys, float(self.t[0]), b["V_int"], b["lam_V"])
            A = b["A_aut"]
            return lambda n: A
        if m not in self._Afull:
            self._Afull[m] = [cv.A_structured(self.sys, 0.5 * (self.t[n] + self.t[n + 1]), b["V_int"], b["lam_V"])
                              for n in range(self.N)]
        L = self._Afull[m]
        return lambda n: L[n + start]

    def A_at(self, m: int) -> Callable[[Sequence[float]], List[np.ndarray]]:
        b = self.basis(m)
        return lambda times: A_at_times(self.sys, b["V_int"], b["lam_V"], times)

    def tf_p0(self, m: int) -> np.ndarray:
        """Teacher-forced rollout from the payoff (P0), cached."""
        if m not in self._tf0:
            tg_ = self.targets(m)
            self._tf0[m] = orc.teacher_forced_rollout(self.A_mid(m, 0), tg_["M"], self.t, tg_["c"][0],
                                                      self.sys.time_dependent)
        return self._tf0[m]

    def level_fine256(self, role: str) -> tuple:
        """(fine phi rows restricted to the 256 grid, base-sub output indices on that level's axis, level h)."""
        if role not in self._fine:
            ref = self.refs[role]
            n_own = ref["phi"].shape[1]
            idx = rf.nested_index(n_own, self.n)
            r = int(ref["out_idx"][1] - ref["out_idx"][0]) // int(self.out_idx[1] - self.out_idx[0])
            self._fine[role] = (ref["phi"][:, idx], self.out_idx * r, float(ref["h"]))
        return self._fine[role]


def make_bundle(case, grid256, refs: Dict[str, dict], sub: int = 16) -> EvalBundle:
    """Assemble the EvalBundle.  ``refs`` = output of ``load_levels`` (name -> ref); needs same, refined, step."""
    n = grid256.n_k
    by_role: Dict[str, dict] = {}
    for ref in refs.values():
        role = ref.get("role") or _role_of(ref, n, sub)
        if role is None:
            role = _role_of(ref, n, sub)
        if role is not None:
            by_role[role] = ref
    missing = [r for r in REQUIRED_ROLES if r not in by_role]
    if missing:
        raise RuntimeError(f"required reference levels missing: {missing} (run s2_references first)")
    same = by_role["same"]
    t = np.asarray(same["t"], float)
    h = float(same["h"])
    out_idx = np.asarray(same["out_idx"], int)
    i0 = int(same["i0"])
    early_idx = np.rint(np.array(EARLY_FRACTIONS) / h).astype(int)
    ctx24 = mt.make_context(grid256)
    ctx_early = mt.make_context(grid256, T_rows=t[early_idx] * grid256.T_max)
    sys_ = cv.interior_system(case, grid256)
    phi_same = np.asarray(same["phi"], float)
    ell = cv.lift_rows(phi_same[0], grid256.k)
    psi_int = cv.detrend_rows(phi_same, grid256.k)[:, 1:-1]
    rows, rows_early = {}, {}
    for role, ref in by_role.items():
        r = int(ref["out_idx"][1] - ref["out_idx"][0]) // int(out_idx[1] - out_idx[0])
        rows[role] = reference_rows(case, ref, grid256, out_idx * r)
        if role in ("same", "refined"):
            rows_early[role] = reference_rows(case, ref, grid256, early_idx)
    ident = mt.identifiable_mask(rows["refined"]["phi"], ctx24, 1e-3)
    sigma24 = mt.sigma_true_rows(case, grid256, ctx24.T_rows)
    bs24 = None
    if case.family == "C" and "sigma" in case.params:
        bs24 = rf.bs_rows(grid256, float(case.params["sigma"]), t[out_idx])
    # FD16 vs FD32 of the same-grid reference (interior RMS over rows 1..-2 to avoid one-sided end rows)
    step = by_role["step"]
    fd16 = mt.time_derivative_rows_fd(phi_same, out_idx, h)
    r32 = int(step["out_idx"][1] - step["out_idx"][0]) // int(out_idx[1] - out_idx[0])
    fd32 = mt.time_derivative_rows_fd(np.asarray(step["phi"], float), out_idx * r32, float(step["h"]))
    fdr = interior_rms((fd16 - fd32)[1:-1], ctx24.interior[1:-1])
    return EvalBundle(case=case, grid=grid256, n=n, sub=sub, t=t, h=h, i0=i0, N=len(t) - 1, out_idx=out_idx,
                      early_idx=early_idx, ctx24=ctx24, ctx_early=ctx_early, sys=sys_, ell=ell, psi_int=psi_int,
                      refs=by_role, rows=rows, rows_early=rows_early, ident_mask=ident, sigma24=sigma24, bs24=bs24,
                      fd16_vs_fd32_rms=fdr, levels_present=sorted(by_role))


# ----------------------------------------------------------------------------- scoring core
def _score_core(bundle: EvalBundle, phi: np.ndarray, f: np.ndarray, phi_t: np.ndarray, post_from: int,
                truth: Optional[dict], same: Optional[dict]) -> dict:
    ctx = bundle.ctx24
    out: dict = {}
    if truth is not None:
        out.update(mt.price_metrics(phi, truth["phi"], ctx, bs_rows=bundle.bs24))
        out.update(mt.density_metrics(f, truth["f"], ctx, post_from))
    if same is not None:
        out.update(_prefix(mt.price_metrics(phi, same["phi"], ctx), "same_"))
        out.update(_prefix(mt.density_metrics(f, same["f"], ctx, post_from), "same_"))
    if truth is None:                       # still report the model's own negative / signed mass
        neg = _neg_mass(f, ctx)
        out["negative_mass"] = neg.tolist()
        out["negative_mass_max_post_initial"] = float(np.max(neg[post_from:]))
        out["negative_mass_final"] = float(neg[-1])
    out.update(mt.arbitrage_metrics(phi, phi_t, ctx))
    iv = mt.inverse_vol_metrics(phi, phi_t, ctx, bundle.ident_mask, bundle.sigma24)
    out["invvol_panel"] = {k: v for k, v in iv.items() if not k.startswith("invvol_")}
    out.update({k: v for k, v in iv.items() if k.startswith("invvol_")})
    return out


def _fd_check(bundle: EvalBundle, phi_t_rhs: np.ndarray, phi_fine: np.ndarray, out_w: np.ndarray, h: float) -> dict:
    """interior RMS (rows 1..-2) of RHS phi_t minus central differences of the model's own fine rows."""
    fd = mt.time_derivative_rows_fd(phi_fine, out_w, h)
    d = (phi_t_rhs - fd)[1:-1]
    return {"rms_rhs_vs_fd16": interior_rms(d, bundle.ctx24.interior[1:-1]),
            "rms_fd16_vs_fd32_ref": bundle.fd16_vs_fd32_rms}


def _early_panel(bundle: EvalBundle, phi_e: np.ndarray) -> dict:
    ctx = bundle.ctx_early
    ref = bundle.rows_early["refined"]
    f_e = mt.density_rows(phi_e, ctx, "d2_repo")
    err = phi_e - ref["phi"]
    l2 = interior_l2_rows(f_e, ref["f"], ctx)
    neg = _neg_mass(f_e, ctx)
    out = {}
    for i, lab in enumerate(EARLY_LABELS):
        sm = float(mt._trapz(f_e[i], ctx.K_rows[i]))
        out[lab] = {"price_interior_rms": interior_rms(err[i], ctx.interior[i]), "density_l2_interior": float(l2[i]),
                    "negative_mass": float(neg[i]), "signed_mass": sm,
                    "t_actual": float(bundle.t[bundle.early_idx[i]])}
    return out


def score_trajectory(bundle: EvalBundle, m: int, c_rows_fine: np.ndarray, M_hat_fn: Optional[Callable],
                     protocol: str, A_at: Callable, V_full: np.ndarray, spec: Optional[ModelSpec] = None) -> dict:
    """Score a resolved trajectory on its fine window (P0: indices 0..N, P1: indices i0..N).

    ``M_hat_fn(window_indices) -> (len, m)`` gives the closure feedback at those window rows (None => zeros).
    ``A_at(times) -> list of (m,m)`` gives A(t) at the output times.  ``spec`` is accepted for bookkeeping only.
    Keys: refined-truth metrics unprefixed, same-grid truth ``same_*``; ``resolved_*`` vs the same-grid c_ref;
    arbitrage / inverse-vol with phi_t = V(A c + M_hat); ``phi_t_check``; ``early_panel`` (P0 only).
    """
    s0 = 0 if protocol == "P0" else bundle.i0
    post = 0 if protocol == "P0" else 1
    out_w = bundle.out_idx - s0
    c_win = np.asarray(c_rows_fine, float)
    c_out = c_win[out_w]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        phi = bundle.ell[None, :] + c_out @ V_full.T
        f = mt.density_rows(phi, bundle.ctx24, "d2_repo")
        Mh = np.zeros_like(c_out) if M_hat_fn is None else np.asarray(M_hat_fn(out_w), float)
        A_list = A_at(bundle.t[bundle.out_idx])
        phi_t = mt.rhs_time_derivative_reduced(c_out, Mh, A_list, V_full)
        out = _score_core(bundle, phi, f, phi_t, post, bundle.rows["refined"], bundle.rows["same"])
        out.update(mt.resolved_error(c_out, bundle.c_ref(m)[bundle.out_idx], V_full, bundle.ctx24.interior))
        phi_fine = bundle.ell[None, :] + c_win @ V_full.T
        out["phi_t_check"] = _fd_check(bundle, phi_t, phi_fine, out_w, bundle.h)
        if protocol == "P0":
            phi_e = bundle.ell[None, :] + c_win[bundle.early_idx] @ V_full.T
            out["early_panel"] = _early_panel(bundle, phi_e)
    out["post_initial_from"] = post
    return out


def score_level_rows(bundle: EvalBundle, role: str, truth_role: Optional[str], include_same: bool = False) -> dict:
    """Score a full-solve level (restricted to the 256 grid, own density / RHS) against ``truth_role`` rows."""
    rows = bundle.rows[role]
    truth = bundle.rows.get(truth_role) if truth_role else None
    same = bundle.rows["same"] if include_same else None
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        out = _score_core(bundle, rows["phi"], rows["f"], rows["phi_t"], 0, truth, same)
        fine, out_w, h = bundle.level_fine256(role)
        out["phi_t_check"] = _fd_check(bundle, rows["phi_t"], fine, out_w, h)
        if role == "same" and "same" in bundle.rows_early:
            out["early_panel"] = _early_panel(bundle, bundle.rows_early[role]["phi"])
    out["post_initial_from"] = 0
    return out


# ----------------------------------------------------------------------------- uncertainty
ERROR_TYPE = ("price_interior_rms", "density_l2_interior_mean", "density_l1_interior_mean")
VALUE_TYPE = ("negative_mass_max_post_initial", "negative_mass_final", "invvol_invalid_frac_full", "invvol_coverage",
              "invvol_sigma_relerr_median", "butterfly_raw_interior_frac_mean", "calendar_raw_interior_frac_mean",
              "monotonicity_raw_interior_frac_mean", "positivity_raw_interior_frac_mean")
QUANTITIES = ERROR_TYPE + VALUE_TYPE


def _level_values(bundle: EvalBundle, rows: dict) -> Dict[str, float]:
    """Own values of the value-type quantities for one level's rows (24 outputs on the 256 grid)."""
    ctx = bundle.ctx24
    neg = _neg_mass(rows["f"], ctx)
    arb = mt.arbitrage_metrics(rows["phi"], rows["phi_t"], ctx)
    iv = mt.inverse_vol_metrics(rows["phi"], rows["phi_t"], ctx, bundle.ident_mask, bundle.sigma24)
    out = {"negative_mass_max_post_initial": float(np.max(neg)), "negative_mass_final": float(neg[-1])}
    for k in VALUE_TYPE[2:5]:
        out[k] = float(iv[k])
    for k in VALUE_TYPE[5:]:
        out[k] = float(arb[k])
    return out


def _error_values(bundle: EvalBundle, model: dict, truth: dict) -> Dict[str, float]:
    ctx = bundle.ctx24
    dm = mt.density_metrics(model["f"], truth["f"], ctx)
    return {"price_interior_rms": interior_rms(model["phi"] - truth["phi"], ctx.interior),
            "density_l2_interior_mean": dm["density_l2_interior_mean"],
            "density_l1_interior_mean": dm["density_l1_interior_mean"]}


def _pair_u(bundle: EvalBundle, model: dict, truth: dict) -> Dict[str, float]:
    u = _error_values(bundle, model, truth)
    vm, vt = _level_values(bundle, model), _level_values(bundle, truth)
    for k in VALUE_TYPE:
        u[k] = abs(vm[k] - vt[k])
    return u


def read_gamma() -> dict:
    """Roundoff guard from results/verify/roundoff_guard.json (None when absent)."""
    d = identity.read_json(os.path.join(scc.RESULTS, "verify", "roundoff_guard.json"))
    if not d:
        return {"gamma": None, "gamma_note": "results/verify/roundoff_guard.json not found"}
    g = d.get("gamma")
    if g is None:
        for k, v in d.items():
            if "gamma" in k.lower() and isinstance(v, (int, float)):
                g = v
                break
    return {"gamma": None if g is None else float(g), "gamma_note": "read from results/verify/roundoff_guard.json"}


def uncertainty(bundle: EvalBundle, m_list: Sequence[int]) -> dict:
    """PLAN.md section 3.  per_quantity[q] = {type, u_grid, u_step, u, [order_grid], [order_step]}; u_dyn / e_TF per m."""
    R = bundle.rows
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        ug = _pair_u(bundle, R["same"], R["refined"])
        us = _pair_u(bundle, R["step"], R["same"])
        per: Dict[str, dict] = {}
        for q in QUANTITIES:
            ent = {"type": "error" if q in ERROR_TYPE else "value", "u_grid": float(ug[q]), "u_step": float(us[q])}
            ent["u"] = float(np.nanmax([ug[q], us[q]])) if np.isfinite([ug[q], us[q]]).any() else float("nan")
            per[q] = ent
        have2 = "refined2" in R
        haves2 = "step2" in R
        orders = {}
        for q in ("price_interior_rms", "density_l2_interior_mean", "density_l1_interior_mean"):
            if have2:
                e1 = _error_values(bundle, R["same"], R["refined"])[q]
                e2 = _error_values(bundle, R["refined"], R["refined2"])[q]
                per[q]["order_grid"] = _log2_ratio(e1, e2)
                orders[f"{q}_grid"] = {"e_256_vs_511": e1, "e_511_vs_1021": e2}
            if haves2:
                e1 = _error_values(bundle, R["step"], R["same"])[q]
                e2 = _error_values(bundle, R["step2"], R["step"])[q]
                per[q]["order_step"] = _log2_ratio(e1, e2)
                orders[f"{q}_step"] = {"e_s32_vs_s16": e1, "e_s64_vs_s32": e2}
        # resolved-coordinate uncertainty
        step_ref = bundle.refs["step"]
        r32 = int(step_ref["out_idx"][1] - step_ref["out_idx"][0]) // int(bundle.out_idx[1] - bundle.out_idx[0])
        psi32 = cv.detrend_rows(np.asarray(step_ref["phi"], float), bundle.grid.k)[:, 1:-1]
        idx32 = bundle.out_idx * r32
        u_dyn, e_tf, parts = {}, {}, {}
        for m in m_list:
            b = bundle.basis(m)
            c16 = bundle.c_ref(m)[bundle.out_idx]
            c32 = psi32[idx32] @ b["V_int"]
            a = mt.resolved_error(c16, c32, b["V_full"], bundle.ctx24.interior)["resolved_interior_rms"]
            ctf = bundle.tf_p0(m)[bundle.out_idx]
            e = mt.resolved_error(ctf, c16, b["V_full"], bundle.ctx24.interior)["resolved_interior_rms"]
            u_dyn[str(m)] = float(np.nanmax([a, e]))
            e_tf[str(m)] = float(e)
            parts[str(m)] = {"step_diff_resolved_rms": float(a), "e_TF": float(e)}
    out = {"per_quantity": per, "u_dyn": u_dyn, "e_TF": e_tf, "u_dyn_parts": parts, "orders_detail": orders,
           "levels_present": list(bundle.levels_present), "quantities": list(QUANTITIES)}
    out.update(read_gamma())
    return out


# ----------------------------------------------------------------------------- online components (single timing)
def _sigma_hat(phi, phi_t, grid, ctx):
    pkk = phi @ ctx.D2_repo.T
    with np.errstate(all="ignore"):
        return 2.0 * phi_t / (grid.T_max * grid.k[None, :] ** 2 * pkk)


# ----------------------------------------------------------------------------- shared row runners
def _fin(x):
    return None if x is None else x


def run_oracle_row(bundle: EvalBundle, kind: str, m: int, protocol: str) -> dict:
    """PROJ (c = c_ref, M_hat = M_ref so phi_t = V V^T L psi_ref) or TF (teacher-forced rollout, M_hat = M_ref)."""
    tg_ = bundle.targets(m)
    b = bundle.basis(m)
    s0 = 0 if protocol == "P0" else bundle.i0
    if kind == "PROJ":
        c = tg_["c"][s0:]
    elif kind == "TF":
        if protocol == "P0":
            c = bundle.tf_p0(m)
        else:
            c = orc.teacher_forced_rollout(bundle.A_mid(m, s0), tg_["M"][s0:], bundle.t[s0:], tg_["c"][s0],
                                           bundle.sys.time_dependent)
    else:
        raise ValueError(kind)
    M = tg_["M"][s0:]
    metrics = score_trajectory(bundle, m, c, lambda w: M[w], protocol, bundle.A_at(m), b["V_full"])
    fin = bool(np.all(np.isfinite(c)))
    return {"metrics": metrics, "unstable": not fin, "n_last": int(len(c) - 1), "total_state": int(m),
            "param_count": {"fitted": 0, "fixed": 0}, "online_seconds": None, "n_expm": None}


def run_block_row(bundle: EvalBundle, m: int, protocol: str) -> dict:
    """Exact block identity with the true z0 (autonomous: augmented exponential; time-dependent: time-ordered phi1)."""
    b = bundle.basis(m, need_U=True)
    s0 = 0 if protocol == "P0" else bundle.i0
    if not bundle.sys.time_dependent:
        r = orc.block_identity_autonomous(bundle.sys, b["V_int"], b["U_int"], bundle.psi_int, bundle.t, s0)
        kind = "autonomous"
    else:
        r = orc.block_identity_timeordered(bundle.sys, b["V_int"], b["U_int"], bundle.psi_int, bundle.t, s0,
                                           "forward")
        kind = "timeordered_forward"
    return {"metrics": dict(r, identity_kind=kind), "unstable": False, "n_last": int(bundle.N), "total_state": int(m),
            "param_count": {"fitted": 0, "fixed": 0}, "online_seconds": None, "n_expm": None}


def run_rollout_row(bundle: EvalBundle, spec: ModelSpec, protocol: str, hidden: str = "W0",
                    stability: bool = True) -> tuple:
    """Free rollout of a deployable model from the payoff (P0) or the projected warm start (P1).

    ``hidden`` in W0 / WI / WU applies to LR specs under P1 (under P0 the hidden state starts at 0).
    Returns (row, traj) with traj = dict(c, h, start).  Online components are timed once with perf_counter
    (the proper protocol is s6): init, stepping (rollout['seconds'], which includes lazy A assembly and the
    exponentials), reconstruction, density, inverse_vol.
    """
    m = spec.m
    b = bundle.basis(m)
    V_int, V_full, lam = b["V_int"], b["V_full"], b["lam_V"]
    td = bundle.sys.time_dependent
    ref_norm = bundle.cref_norm_max(m)
    s0 = 0 if protocol == "P0" else bundle.i0
    clock = _time.perf_counter
    n_expm = 0
    t_init0 = clock()
    c0 = (bundle.psi_int[s0] @ V_int)
    h0 = None
    warm_unstable = False
    step_seconds = 0.0
    if spec.closure == "LR":
        if protocol == "P0" or hidden == "W0":
            h0 = np.zeros(spec.n_h)
        elif hidden == "WI":
            h0 = cl.hidden_init_state(spec, "WI", c0)
        elif hidden == "WU":
            t1 = bundle.t[: bundle.i0 + 1]
            ro1 = cl.rollout(spec, A_mid_factory(bundle.sys, V_int, lam, t1), t1, bundle.psi_int[0] @ V_int, lam,
                             h0=np.zeros(spec.n_h), ref_norm_max=ref_norm, time_dependent=td)
            h0 = ro1["h"][-1].copy()
            warm_unstable = bool(ro1["unstable"])
            step_seconds += ro1["seconds"]
            n_expm += ro1["n_expm"]
        else:
            raise ValueError(f"unknown hidden protocol {hidden!r}")
    t_init = clock() - t_init0
    tw = bundle.t[s0:]
    A_mid = A_mid_factory(bundle.sys, V_int, lam, tw)
    ro = cl.rollout(spec, A_mid, tw, c0, lam, h0=h0, ref_norm_max=ref_norm, time_dependent=td)
    step_seconds += ro["seconds"]
    n_expm += ro["n_expm"]
    fb = feedback_fn(spec, lam)
    c, hh = ro["c"], ro["h"]
    metrics = score_trajectory(bundle, m, c, lambda w: fb(c[w], hh[w]), protocol, bundle.A_at(m), V_full, spec)
    out_w = bundle.out_idx - s0
    # ---- timing of reconstruction / density / inverse volatility (single pass; not the s6 protocol)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        t2 = clock()
        phi = bundle.ell[None, :] + c[out_w] @ V_full.T
        t_rec = clock() - t2
        t3 = clock()
        mt.density_rows(phi, bundle.ctx24, "d2_repo")
        t_den = clock() - t3
        t4 = clock()
        phit = mt.rhs_time_derivative_reduced(c[out_w], fb(c[out_w], hh[out_w]), bundle.A_at(m)(bundle.t[bundle.out_idx]),
                                              V_full)
        _sigma_hat(phi, phit, bundle.grid, bundle.ctx24)
        t_inv = clock() - t4
    total = t_init + step_seconds + t_rec + t_den + t_inv
    online = {"init": t_init, "assembly_and_expm": None, "stepping": step_seconds, "reconstruction": t_rec,
              "density": t_den, "inverse_vol": t_inv, "total": total,
              "note": "assembly and exponentials are included in 'stepping'"}
    if stability:
        A_out = bundle.A_at(m)(bundle.t[bundle.out_idx])
        st = cl.stability_report(spec, A_out, lam)
        kw = dict(h0=h0, ref_norm_max=ref_norm, time_dependent=td)
        st["perturbation"] = cl.perturbation_growth(spec, A_mid, tw, c0, lam, **kw)
        if td:
            st["propagator_growth"] = cl.propagator_growth(spec, A_mid, tw, lam, time_dependent=True)
        else:
            st["propagator_growth"] = None
        A_end = cv.A_structured(bundle.sys, 1.0, V_int, lam)
        st["long_horizon"] = cl.long_horizon(spec, A_end, lam, t_end=3.0, h=bundle.h)
        metrics["stability"] = st
    if warm_unstable:
        metrics["warmup_unstable"] = True
    row = {"metrics": metrics, "unstable": bool(ro["unstable"] or warm_unstable), "n_last": int(ro["n_last"]),
           "total_state": int(spec.total_state()), "param_count": spec.param_count(), "online_seconds": online,
           "n_expm": int(n_expm)}
    return row, {"c": c, "h": hh, "start": s0}


# ----------------------------------------------------------------------------- tiny models (smoke tests / quick mode)
def fit_tiny_specs(case, grid, ref: dict, m: int, n_h: int = 2, lam_rel: float = 1e-6) -> Dict[str, ModelSpec]:
    """Tiny linear closures fitted on ONE reference (smoke tests only): QLp, QLpm, QLs, LM, LMb, LR0, LR."""
    tg_ = tgt.build_targets(case, ref, grid, m)
    sys_ = tg_["sys"]
    w = tgt.trajectory_weights({case.id: tg_["rms_M"]}, {case.id: len(tg_["t"])})
    V_int, lam = tg_["V_int"], tg_["lam_V"]
    nu_adm = cl.admissible_bound([sys_], V_int, [0.2, 0.6, 1.0])
    fit = cl.fit_nu([tg_], w, nu_adm)
    specs: Dict[str, ModelSpec] = {}
    for name, val in (("QLp", fit["nu_plus"]), ("QLpm", fit["nu_pm"]), ("QLs", fit["nu_unclipped"])):
        specs[name] = ModelSpec(name=name, m=m, closure="QL", params={"nu": float(val)},
                                meta={"nu_adm": float(nu_adm)})
    H, b, _ = cl.fit_LM([tg_], w, lam_rel)
    specs["LM"] = ModelSpec(name="LM", m=m, closure="LM", params={"H": H, "b": None})
    H2, b2, _ = cl.fit_LM([tg_], w, lam_rel, intercept=True)
    specs["LMb"] = ModelSpec(name="LMb", m=m, closure="LM", params={"H": H2, "b": b2})
    d_lo, d_hi = tgt.d_extremes([case], grid, np.linspace(0.2, 1.0, 5))
    lam_next = float(cv.sine_eigenvalues(grid.n_k, grid.dk, np.array([m + 1]))[0])
    alphas = cl.rate_bank(n_h, "N", d_lo * abs(lam_next), d_hi / d_lo)
    spec0, spec, _ = cl.fit_LR([tg_], w, n_h, alphas, lam_rel)
    i0 = int(ref["i0"])
    K = cl.fit_initializer(spec, [tg_], w, i0, lam_rel)
    spec.params["K_init"] = K
    spec0.params["K_init"] = cl.fit_initializer(spec0, [tg_], w, i0, lam_rel)
    specs["LR0"], specs["LR"] = spec0, spec
    return specs


# ----------------------------------------------------------------------------- result files (shared by s3 / s5 / s6)
def levels_signature(case, n: int, sub: int, roles: Optional[Sequence[str]] = None) -> Dict[str, Optional[str]]:
    """{level_name: identity_hash of the reference if its cache file exists else None} (no array loading)."""
    out = {}
    tab = level_table(n, sub)
    for r in (roles or ROLES):
        v = tab[r]
        ident = rf.reference_identity(case, v["n"], v["sub"], v["solver"])
        ok = os.path.isfile(identity.cache_path("reference", ident))
        out[v["name"]] = identity.identity_hash(ident) if ok else None
    return out


def result_ident(kind: str, case, n: int, sub: int, script_path: str, **extra) -> dict:
    """Identity of a per-case result file: code identity + evaluate.py hash + script hash + metric version + extras."""
    return identity.make_identity(kind=kind, case=case.config_hash(), case_id=case.id, split=case.split, n=int(n),
                                  sub=int(sub), evaluate=evaluate_identity_hash(), script=sha_of_file(script_path),
                                  metric_version=METRIC_VERSION, **extra)


def write_result(path: str, ident: dict, stage: str, wall_seconds: float, status: str = "done",
                 traceback_text: Optional[str] = None, **content) -> str:
    """JSON with status / identity / identity_hash / stage / wall_seconds (+ traceback when failed)."""
    obj = {"status": status, "identity": ident, "identity_hash": identity.identity_hash(ident), "stage": stage,
           "wall_seconds": float(wall_seconds)}
    if traceback_text:
        obj["traceback"] = traceback_text
    obj.update(content)
    return identity.write_json(path, obj)


def prepare_target(path: str, ident: dict, resume: bool) -> str:
    """Resume logic: returns 'skip' when a matching done result exists and resume is set; otherwise supersedes
    stale / corrupted / failed files (kept under *.superseded.<utc>.json) and returns 'run'."""
    status, detail = identity.result_status(path, ident)
    if status == "done" and resume:
        return "skip"
    if status in ("stale", "corrupted", "failed", "incomplete"):
        identity.supersede(path)
    return "run"
