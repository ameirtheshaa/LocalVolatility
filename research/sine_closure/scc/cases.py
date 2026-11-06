"""Deterministic case families and splits (PLAN.md section 4).

Case ids, parameters, splits and generator tags are fixed by the seeds and ranges below. Parameters are
drawn from ``numpy.random.default_rng(seed)`` in table-column order, rounded to 6 decimals and stored;
the stored rounded values are what ``sigma_fn`` uses.

``sigma_fn(T, x)``: ``T`` physical maturity (float), ``x = K/S0`` moneyness array; returns an array
broadcastable to ``x``.

Clip disclosure: every ``sigma_fn`` carries an attribute ``unclipped(T, x)`` giving the formula before the
clip to [SIGMA_MIN, SIGMA_MAX] (identical to ``sigma_fn`` for the families C and D, which are not clipped).
``sigma_checks`` uses it to report the clip-active fraction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from scc import identity

T_MAX = 1.5            # = dsm.repo.DupirePipelineConfig().T_max (asserted in tests)
SIGMA_MIN, SIGMA_MAX = 0.05, 2.0
SPLITS = ("train", "val", "test_interp", "test_ood", "dev_regression")


@dataclass
class Case:
    id: str
    family: str            # "C", "S", "D", "M" (gate-5 cases keep their dsm family: "C", "S", "F")
    params: Dict[str, float]
    split: str
    time_dependent: bool
    sigma_fn: Callable[[float, np.ndarray], np.ndarray] = field(repr=False)
    generator: str = ""

    def config_hash(self) -> str:
        """sha256[:16] of the sorted JSON of {id, family, params, time_dependent, generator}."""
        s = json.dumps({"id": self.id, "family": self.family, "params": self.params,
                        "time_dependent": self.time_dependent, "generator": self.generator}, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- sigma formulas
def _attach(f, unclipped):
    f.unclipped = unclipped
    return f


def _const(sig: float):
    def raw(T, x):
        return sig * np.ones_like(np.asarray(x, dtype=float))
    return _attach(raw, raw)


def _smile_S_raw(b, s, c):
    def raw(T, x):
        x = np.asarray(x, dtype=float)
        return b * (1.0 + s * (1.0 - x) + c * (x - 1.0) ** 2)
    return raw


def _smile_S(b, s, c):
    raw = _smile_S_raw(b, s, c)
    def f(T, x):
        return np.clip(raw(T, x), SIGMA_MIN, SIGMA_MAX)
    return _attach(f, raw)


def _smile_D(sigma_base, t_shift, x_shift):
    """sigma = base + y exp(-y), y = sqrt(x + x_shift)(T + t_shift); same formula as dsm.cases._smile_D."""
    def f(T, x):
        x = np.asarray(x, dtype=float)
        y = np.sqrt(x + x_shift) * (T + t_shift)
        return sigma_base + y * np.exp(-y)
    return _attach(f, f)


def _smile_M(b, a, fr, phi, s, c):
    def raw(T, x):
        x = np.asarray(x, dtype=float)
        mod = 1.0 + a * np.sin(2.0 * np.pi * fr * T / T_MAX + phi)
        return b * mod * (1.0 + s * (1.0 - x) + c * (x - 1.0) ** 2)
    def f(T, x):
        return np.clip(raw(T, x), SIGMA_MIN, SIGMA_MAX)
    return _attach(f, raw)


def _r6(v) -> float:
    return round(float(v), 6)


# ----------------------------------------------------------------------------- families
_C_SIGMAS = {
    "train": [0.18, 0.22, 0.27, 0.33, 0.38, 0.44],
    "val": [0.24, 0.36],
    "test_interp": [0.205, 0.31, 0.415],
    "test_ood": [0.12, 0.55],
}


def family_C() -> List[Case]:
    out = []
    for sp, sigs in _C_SIGMAS.items():
        for sig in sigs:
            out.append(Case(id=f"C_sig{sig:.3f}", family="C", params={"sigma": sig}, split=sp,
                            time_dependent=False, sigma_fn=_const(sig), generator="C_v1"))
    return out


_S_SEEDS = [(range(100, 110), "train", "interp"), (range(110, 114), "val", "interp"),
            (range(120, 126), "test_interp", "interp"), (range(130, 134), "test_ood", "ood")]
_S_RANGES = {"interp": ((0.18, 0.42), (0.0, 0.25), (0.0, 0.15)),
             "ood": ((0.15, 0.5), (0.25, 0.5), (0.15, 0.4))}


def family_S() -> List[Case]:
    out = []
    for seeds, sp, kind in _S_SEEDS:
        rb, rs, rc = _S_RANGES[kind]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            b = _r6(rng.uniform(*rb)); s = _r6(rng.uniform(*rs)); c = _r6(rng.uniform(*rc))
            out.append(Case(id=f"S2_seed{seed}", family="S", params={"seed": seed, "b": b, "s": s, "c": c},
                            split=sp, time_dependent=False, sigma_fn=_smile_S(b, s, c), generator="S_v1"))
    return out


_D_SEEDS = [(range(200, 206), "train", "interp"), (range(210, 212), "val", "interp"),
            (range(220, 224), "test_interp", "interp"), (range(230, 232), "test_ood", "ood")]
_D_RANGES = {"interp": ((0.2, 0.35), (0.05, 0.3), (0.02, 0.2)),
             "ood": ((0.35, 0.5), (0.3, 0.6), (0.2, 0.5))}


def family_D() -> List[Case]:
    out = []
    for seeds, sp, kind in _D_SEEDS:
        rb, rt, rx = _D_RANGES[kind]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            sb = _r6(rng.uniform(*rb)); ts = _r6(rng.uniform(*rt)); xs = _r6(rng.uniform(*rx))
            out.append(Case(id=f"D2_seed{seed}", family="D",
                            params={"seed": seed, "sigma_base": sb, "t_shift": ts, "x_shift": xs},
                            split=sp, time_dependent=True, sigma_fn=_smile_D(sb, ts, xs), generator="D_v1"))
    return out


_M_SEEDS = [(range(300, 306), "train", "interp"), (range(310, 312), "val", "interp"),
            (range(320, 324), "test_interp", "interp"), (range(330, 332), "test_ood", "ood")]
# order: b, a, f, phi, s, c
_M_RANGES = {"interp": ((0.2, 0.4), (0.1, 0.3), (0.5, 1.5), (0.0, 2 * np.pi), (0.0, 0.2), (0.0, 0.1)),
             "ood": ((0.2, 0.4), (0.3, 0.5), (1.5, 3.0), (0.0, 2 * np.pi), (0.2, 0.4), (0.1, 0.3))}


def family_M() -> List[Case]:
    out = []
    for seeds, sp, kind in _M_SEEDS:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            b, a, fr, phi, s, c = (_r6(rng.uniform(*r)) for r in _M_RANGES[kind])
            out.append(Case(id=f"M_seed{seed}", family="M",
                            params={"seed": seed, "b": b, "a": a, "f": fr, "phi": phi, "s": s, "c": c},
                            split=sp, time_dependent=True, sigma_fn=_smile_M(b, a, fr, phi, s, c),
                            generator="M_v1"))
    return out


def gate5_regression() -> List[Case]:
    """The three Gate 5 cases (ids unchanged) rebuilt from the dsm formulas; split 'dev_regression'."""
    from dsm import cases as dc
    out = []
    for cid in ("C_sig0.30", "S_seed9"):
        c = dc.get_case(cid)
        out.append(Case(id=c.id, family=c.family, params=dict(c.params), split="dev_regression",
                        time_dependent=c.time_dependent, sigma_fn=c.sigma_fn, generator="gate5"))
    out.append(Case(id="F_fresh_base0.28_t0.2_x0.05", family="F",
                    params={"sigma_base": 0.28, "t_shift": 0.2, "x_shift": 0.05}, split="dev_regression",
                    time_dependent=True, sigma_fn=dc._smile_D(0.28, t_shift=0.2, x_shift=0.05),
                    generator="gate5"))
    return out


def all_cases() -> List[Case]:
    """Order: C, S, D, M, gate5."""
    return family_C() + family_S() + family_D() + family_M() + gate5_regression()


def cases_by_split(split: str) -> List[Case]:
    return [c for c in all_cases() if c.split == split]


def get_case(case_id: str) -> Case:
    for c in all_cases():
        if c.id == case_id:
            return c
    raise KeyError(case_id)


def split_counts() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in all_cases():
        out[c.split] = out.get(c.split, 0) + 1
    return out


# ----------------------------------------------------------------------------- disclosure
def _unclipped_fn(case: Case):
    return getattr(case.sigma_fn, "unclipped", case.sigma_fn)


def sigma_checks(case: Case, grid, n_T: int = 49) -> dict:
    """sigma statistics on grid.K_row(T)/S0 for T in linspace(0, T_max, n_T).

    clip_active_fraction: fraction of (T,k) points where the unclipped formula is outside
    [0.05, 2.0] (0.0 for the unclipped families C and D). max_rel_jump_k / _T: largest
    |sigma[i+1]-sigma[i]| / max(sigma[i], 1e-12) between neighbouring strikes / neighbouring maturities.
    """
    Ts = np.linspace(0.0, grid.T_max, n_T)
    sig = np.empty((n_T, grid.n_k)); raw = np.empty_like(sig)
    unc = _unclipped_fn(case)
    for i, T in enumerate(Ts):
        x = grid.K_row(float(T)) / grid.S0
        sig[i] = np.broadcast_to(np.asarray(case.sigma_fn(float(T), x), float), x.shape)
        raw[i] = np.broadcast_to(np.asarray(unc(float(T), x), float), x.shape)
    dk = np.abs(np.diff(sig, axis=1)) / np.maximum(sig[:, :-1], 1e-12)
    dT = np.abs(np.diff(sig, axis=0)) / np.maximum(sig[:-1], 1e-12)
    return {"sigma_min": float(sig.min()), "sigma_max": float(sig.max()),
            "clip_active_fraction": float(np.mean((raw < SIGMA_MIN) | (raw > SIGMA_MAX))),
            "max_rel_jump_k": float(dk.max()), "max_rel_jump_T": float(dT.max())}


def _case_record(c: Case) -> dict:
    return {"id": c.id, "family": c.family, "params": c.params, "split": c.split,
            "generator": c.generator}


def manifest(grid=None) -> dict:
    """{cases: [...], counts, sha256}; sha256 over the JSON (sort_keys) of the sorted
    (id, family, params, split, generator) list. With a grid each case gets a 'checks' entry."""
    cases = all_cases()
    recs = []
    for c in cases:
        r = _case_record(c)
        r["config_hash"] = c.config_hash()
        r["time_dependent"] = c.time_dependent
        if grid is not None:
            r["checks"] = sigma_checks(c, grid)
        recs.append(r)
    key = sorted((c.id, c.family, c.params, c.split, c.generator) for c in cases)
    sha = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()
    return {"cases": recs, "counts": split_counts(), "sha256": sha}


def write_manifest(path: str, grid=None) -> str:
    """Atomically write manifest(grid) as JSON; returns the file sha256."""
    return identity.write_json(path, manifest(grid))
