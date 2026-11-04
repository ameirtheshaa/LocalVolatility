"""Deterministic case families and splits (see FROZEN_PLAN.md)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np


@dataclass
class Case:
    id: str
    family: str            # "C", "S", "D"
    params: Dict[str, float]
    split: str             # "train", "val", "test"
    time_dependent: bool
    sigma_fn: Callable[[float, np.ndarray], np.ndarray] = field(repr=False)

    def config_hash(self) -> str:
        s = json.dumps({"id": self.id, "family": self.family, "params": self.params,
                        "time_dependent": self.time_dependent}, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def _const(sig: float):
    def f(t, x):
        return sig * np.ones_like(np.asarray(x, dtype=float))
    return f


def _smile_S(b: float, s: float, c: float):
    def f(t, x):
        x = np.asarray(x, dtype=float)
        return np.clip(b * (1.0 + s * (1.0 - x) + c * (x - 1.0) ** 2), 0.05, 2.0)
    return f


def _smile_D(sigma_base: float, t_shift: float = 0.1, x_shift: float = 0.1):
    """Identical formula to ``build_uniform_tk_grid`` for model_type='dupire_exact'."""
    def f(t, x):
        x = np.asarray(x, dtype=float)
        y = np.sqrt(x + x_shift) * (t + t_shift)
        return sigma_base + y * np.exp(-y)
    return f


SPLIT_C = {0.20: "train", 0.30: "train", 0.40: "train", 0.25: "val", 0.35: "test"}
SPLIT_S = {**{s: "train" for s in range(6)}, 6: "val", 7: "val", 8: "test", 9: "test"}
SPLIT_D = {0.30: "train", 0.25: "val", 0.35: "test"}


def family_C() -> List[Case]:
    return [Case(id=f"C_sig{sig:.2f}", family="C", params={"sigma": sig}, split=sp,
                 time_dependent=False, sigma_fn=_const(sig)) for sig, sp in SPLIT_C.items()]


def family_S() -> List[Case]:
    out = []
    for seed, sp in SPLIT_S.items():
        rng = np.random.default_rng(seed)
        b = float(rng.uniform(0.2, 0.4))
        s = float(rng.uniform(0.0, 0.10))
        c = float(rng.uniform(0.0, 0.05))
        out.append(Case(id=f"S_seed{seed}", family="S",
                        params={"seed": seed, "b": round(b, 6), "s": round(s, 6), "c": round(c, 6)},
                        split=sp, time_dependent=False, sigma_fn=_smile_S(b, s, c)))
    return out


def family_D() -> List[Case]:
    return [Case(id=f"D_base{b:.2f}", family="D", params={"sigma_base": b}, split=sp,
                 time_dependent=True, sigma_fn=_smile_D(b)) for b, sp in SPLIT_D.items()]


def all_cases() -> List[Case]:
    return family_C() + family_S() + family_D()


def cases_by_split(split: str) -> List[Case]:
    return [c for c in all_cases() if c.split == split]


def get_case(case_id: str) -> Case:
    for c in all_cases():
        if c.id == case_id:
            return c
    raise KeyError(case_id)


REFINEMENT_CASES = ["C_sig0.30", "S_seed8", "D_base0.30"]
KEEPS = [16, 32, 64]
