"""Locked model files: one .npz (weights) + sidecar .json (config, training diagnostics, hashes).

File name: models/m<m>/<closure>[_h<n_h>]_<config_tag>.npz, e.g. models/m17/LR_h16_lam1e-06_bankN.npz,
models/m9/LM_lam1e-08.npz, models/m9/QL.npz (QL stores nu_plus / nu_pm / nu_unclipped in one file).
models/LOCK.json: {"selected": {"m9": {"LM": {"file": ..., "sha256": ..., "config": {...}}, ...}, ...},
                   "code_identity": ..., "cases_sha256": ..., "created_utc": ..., "n_files": ...}
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np

import scc
from scc import identity
from scc.closures import ModelSpec

ARRAY_KEYS = ("H", "b", "F", "G", "R", "K_init")


def spec_path(m: int, closure: str, n_h: int = 0, config_tag: str = "") -> str:
    name = closure + (f"_h{n_h}" if n_h else "") + (f"_{config_tag}" if config_tag else "")
    return os.path.join(scc.MODELS, f"m{m}", name + ".npz")


def save_spec(spec: ModelSpec, path: str, config: Optional[dict] = None, training: Optional[dict] = None,
              offline_seconds: float = 0.0, extra: Optional[dict] = None) -> str:
    """Write the weights (.npz, atomic) and the sidecar .json; returns the npz sha256."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {}
    for k, v in spec.params.items():
        if v is None:
            continue
        arrays[k] = np.asarray(v, float)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    identity.replace_retry(tmp, path)
    sha = identity.sha256_file(path)
    side = {"name": spec.name, "m": int(spec.m), "closure": spec.closure, "n_h": int(spec.n_h),
            "hidden_init": spec.hidden_init,
            "meta": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in spec.meta.items()},
            "config": config or {}, "training": training or {}, "param_count": spec.param_count(),
            "total_state": spec.total_state(), "offline_seconds": float(offline_seconds),
            "npz_sha256": sha, "code_identity": identity.code_identity(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "extra": extra or {}}
    identity.write_json(path[:-4] + ".json", side)
    return sha


def load_spec(path: str, verify_sha: Optional[str] = None) -> ModelSpec:
    """Rebuild a ModelSpec from a .npz + sidecar; optionally verify the npz sha256 (RuntimeError on mismatch)."""
    side = identity.read_json(path[:-4] + ".json")
    if side is None:
        raise FileNotFoundError(path[:-4] + ".json")
    if verify_sha is not None and identity.sha256_file(path) != verify_sha:
        raise RuntimeError(f"weight file hash mismatch: {path}")
    with np.load(path) as z:
        params = {k: np.array(z[k]) for k in z.files}
    for k in ARRAY_KEYS:
        params.setdefault(k, None)
    if side["closure"] == "QL":
        params = {"nu": float(params["nu"])} if "nu" in params else params
    elif side["closure"] == "LM":
        params = {"H": params["H"], "b": params.get("b")}
    elif side["closure"] == "LR":
        params = {"F": params["F"], "G": params["G"], "H": params["H"], "R": params["R"],
                  "K_init": params.get("K_init")}
    elif side["closure"] == "RRR":
        params = {}
    meta = dict(side.get("meta", {}))
    if "alphas" in meta:
        meta["alphas"] = np.asarray(meta["alphas"], float)
    return ModelSpec(name=side["name"], m=int(side["m"]), closure=side["closure"], params=params,
                     n_h=int(side["n_h"]), hidden_init=side.get("hidden_init", "H0"), meta=meta)


def write_lock(selected: Dict[str, Dict[str, dict]], cases_sha256: str, extra: Optional[dict] = None) -> str:
    """selected = {"m9": {"LM": {"file": <abs or models-relative path>, "config": {...}}, ...}, ...};
    the sha256 of every file is computed here. Written to models/LOCK.json."""
    n = 0
    for mkey, d in selected.items():
        for cl, e in d.items():
            p = e["file"] if os.path.isabs(e["file"]) else os.path.join(scc.MODELS, e["file"])
            e["sha256"] = identity.sha256_file(p)
            e["file"] = os.path.relpath(p, scc.MODELS)
            n += 1
    lock = {"selected": selected, "code_identity": identity.code_identity(), "cases_sha256": cases_sha256,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_files": n, "extra": extra or {}}
    path = os.path.join(scc.MODELS, "LOCK.json")
    identity.write_json(path, lock)
    return path


def read_lock() -> dict:
    lock = identity.read_json(os.path.join(scc.MODELS, "LOCK.json"))
    if lock is None:
        raise FileNotFoundError("models/LOCK.json (run s4_fit.py first)")
    return lock


def load_locked(lock: dict, m: int, closure: str) -> ModelSpec:
    """Load a locked model and verify its hash; KeyError when (m, closure) is not in the lock."""
    e = lock["selected"][f"m{m}"][closure]
    return load_spec(os.path.join(scc.MODELS, e["file"]), verify_sha=e["sha256"])
