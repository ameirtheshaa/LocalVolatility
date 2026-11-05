"""Run identities, npz cache, atomic JSON, run manifests and the test-leakage guard (API_CONTRACT)."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import scc
from scc import CACHE_ROOT, DSM_ROOT, REPO_ROOT, RESULTS, STUDY_ROOT

IDENTITY_SCHEMA = 1
CODE_MODULES = ["conventions", "cases", "reference", "targets", "closures", "oracles", "metrics"]
# A cached *reference* depends only on the case formulas, the operator conventions and the solvers (plus the
# production code they call); hashing the closure/metric modules into it would throw away every expensive
# reference on an unrelated edit.  Result files (scores etc.) use the full hash.
REFERENCE_FILES = [os.path.join(STUDY_ROOT, "scc", n + ".py") for n in ("conventions", "cases", "reference")] + \
    [os.path.join(DSM_ROOT, "dsm", n + ".py") for n in ("operators", "reference", "cases", "repo")] + \
    [os.path.join(REPO_ROOT, "Synthetic_Data_Tensorflow_Advanced", n) for n in
     ("config.py", "analytical_solutions.py", "mz_spectral/fourier_dupire.py", "mz_spectral/validation.py")]


def _hash_files(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(os.path.relpath(p, REPO_ROOT).replace(os.sep, "/").encode())
        if os.path.isfile(p):
            with open(p, "rb") as f:
                h.update(f.read())
        else:
            h.update(b"missing")
    return h.hexdigest()[:16]


def code_identity(scope: str = "full") -> str:
    """sha256[:16] over the numerical code. scope='full': scc/{conventions,cases,reference,targets,closures,
    oracles,metrics}.py (name + bytes; missing files hashed as b'missing'); scope='reference': REFERENCE_FILES
    (what a cached reference actually depends on). Env SCC_CODE_IDENTITY_OVERRIDE replaces both (tests only)."""
    ov = os.environ.get("SCC_CODE_IDENTITY_OVERRIDE")
    if ov:
        return ov
    if scope == "reference":
        return _hash_files(REFERENCE_FILES)
    return _hash_files([os.path.join(STUDY_ROOT, "scc", name + ".py") for name in CODE_MODULES])


def make_identity(**fields) -> Dict[str, Any]:
    """{schema: 1, code, basis: 'dst1_v1', lift: 'affine_endpoints_v1', operator: 'OP-B', **fields};
    kind='reference' uses the narrow reference-code hash (see REFERENCE_FILES)."""
    scope = "reference" if fields.get("kind") == "reference" else "full"
    ident = {"schema": IDENTITY_SCHEMA, "code": code_identity(scope), "basis": "dst1_v1",
             "lift": "affine_endpoints_v1", "operator": "OP-B"}
    ident.update(fields)
    return ident


def identity_hash(ident: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- JSON / hashes
def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def replace_retry(src: str, dst: str, attempts: int = 25, first_delay: float = 0.05) -> None:
    """os.replace with retries. On Windows a file under Dropbox can be briefly locked by the sync client
    right after it is written, and renaming over it raises PermissionError (WinError 5/32); retry with
    exponential backoff (total wait ~30 s) before giving up."""
    import time as _t
    delay = first_delay
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            _t.sleep(delay)
            delay = min(2.0 * delay, 2.0)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, obj: Any) -> str:
    """Atomic (tmp + os.replace), numpy-safe, sort_keys, indent=1. Returns the file sha256."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_default)
    replace_retry(tmp, path)
    return sha256_file(path)


def read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def env_identity() -> Dict[str, str]:
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "machine": platform.machine()}


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def source_hashes() -> Dict[str, str]:
    """sha256 of scc/*.py, scripts/*.py and the dsm numerical modules (dsm.checkpoint.NUMERICAL_MODULES)."""
    from dsm import checkpoint as dck
    out = {}
    paths = sorted(glob.glob(os.path.join(STUDY_ROOT, "scc", "*.py")) +
                   glob.glob(os.path.join(STUDY_ROOT, "scripts", "*.py")))
    for p in paths:
        out[os.path.relpath(p, REPO_ROOT)] = sha256_file(p)
    for rel in dck.NUMERICAL_MODULES:
        p = os.path.join(DSM_ROOT, rel)
        out[os.path.relpath(p, REPO_ROOT)] = sha256_file(p) if os.path.isfile(p) else "missing"
    return out


# ----------------------------------------------------------------------------- npz cache
def cache_path(kind: str, ident: Dict[str, Any]) -> str:
    return os.path.join(CACHE_ROOT, kind, identity_hash(ident) + ".npz")


def cache_store(kind: str, ident: Dict[str, Any], **arrays) -> str:
    """Write arrays to CACHE_ROOT/<kind>/<hash>.npz plus a sidecar .json (identity, hash, created_utc,
    numpy/scipy versions, shapes). Both are written to tmp names then os.replace'd: sidecar first, npz last,
    so a visible npz always has its sidecar.
    """
    import scipy
    path = cache_path(kind, ident)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrs = {k: np.asarray(v) for k, v in arrays.items()}
    side = {"identity": ident, "identity_hash": identity_hash(ident),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "shapes": {k: list(v.shape) for k, v in arrs.items()}}
    write_json(path[:-4] + ".json", side)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrs)
    replace_retry(tmp, path)
    return path


def cache_load(kind: str, ident: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """Arrays, or None when missing/unreadable/sidecar identity-hash mismatch/shape mismatch."""
    path = cache_path(kind, ident)
    try:
        side = read_json(path[:-4] + ".json")
        if not side or side.get("identity_hash") != identity_hash(ident):
            return None
        if side.get("identity") is None or identity_hash(side["identity"]) != side["identity_hash"]:
            return None
        if not os.path.isfile(path):
            return None
        with np.load(path, allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
        shapes = side.get("shapes", {})
        if set(shapes) != set(arrs) or any(list(arrs[k].shape) != list(shapes[k]) for k in arrs):
            return None
        return arrs
    except Exception:
        return None


# ----------------------------------------------------------------------------- result files
def result_status(path: str, ident: Dict[str, Any]) -> Tuple[str, str]:
    """('done' | 'missing' | 'corrupted' | 'stale' | 'failed' | 'incomplete', detail)."""
    if not os.path.isfile(path):
        return "missing", ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return "corrupted", f"unreadable JSON: {e}"
    if not isinstance(d, dict) or "status" not in d:
        return "corrupted", "no status field"
    if d["status"] == "failed":
        return "failed", str(d.get("traceback", ""))[-200:]
    if d["status"] != "done":
        return "incomplete", f"status={d['status']}"
    stored = d.get("identity")
    if stored is None or d.get("identity_hash") is None:
        return "stale", "result without identity"
    if d["identity_hash"] != identity_hash(ident):
        diff = sorted(k for k in set(stored) | set(ident) if stored.get(k) != ident.get(k))
        return "stale", f"identity differs in {diff}"
    if d["identity_hash"] != identity_hash(stored):
        return "corrupted", "identity_hash does not match the stored identity"
    return "done", ""


def supersede(path: str) -> Optional[str]:
    """Keep a stale/corrupted result under a timestamped name (evidence is never deleted)."""
    if not os.path.isfile(path):
        return None
    new = path.replace(".json", "") + ".superseded." + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".json"
    replace_retry(path, new)
    return new


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


class RunManifest:
    """Interrupted-run detection; file RESULTS/<stage>/RUN_MANIFEST.json."""

    def __init__(self, stage: str, planned: List[str]):
        self.path = os.path.join(RESULTS, stage, "RUN_MANIFEST.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.previous_interrupted: Dict[str, Any] = {}
        old = None
        try:
            old = read_json(self.path)
        except Exception:
            old = None
        if old and old.get("in_progress") and not _pid_alive(old.get("pid")):
            self.previous_interrupted = {"started_utc": old.get("started_utc"), "pid": old.get("pid"),
                                         "in_progress": old.get("in_progress"),
                                         "completed": old.get("completed", [])}
        self.data = {"stage": stage, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "pid": os.getpid(), "planned": list(planned), "in_progress": [], "completed": [],
                     "failed": [], "finished": False, "previous_interrupted": self.previous_interrupted}
        self._write()

    def _write(self):
        write_json(self.path, self.data)

    def start(self, cid: str):
        self.data["in_progress"] = [cid]
        self._write()

    def done(self, cid: str):
        self.data["in_progress"] = []
        self.data["completed"].append(cid)
        self._write()

    def failed(self, cid: str):
        self.data["in_progress"] = []
        self.data["failed"].append(cid)
        self._write()

    def finish(self):
        self.data["in_progress"] = []
        self.data["finished"] = True
        self._write()


def assert_phase_allows(split: str) -> None:
    """Leakage guard: RuntimeError if SCC_PHASE == 'fit' and the split starts with 'test'."""
    if os.environ.get("SCC_PHASE") == "fit" and str(split).startswith("test"):
        raise RuntimeError(f"SCC_PHASE=fit forbids access to split '{split}'")
