"""Durable, idempotent checkpoints (identities, seeds, case status, commands, checksums)."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from dsm.repo import REPO_ROOT, STUDY_ROOT

# DSM_OUTPUT_ROOT redirects results/, checkpoints/ and NEXT_ACTIONS.md (used by the tests and by
# the fresh-directory restore rehearsal); default is the study directory itself.
OUTPUT_ROOT = os.environ.get("DSM_OUTPUT_ROOT", STUDY_ROOT)
RESULTS = os.path.join(OUTPUT_ROOT, "results")
CHECKPOINTS = os.path.join(OUTPUT_ROOT, "checkpoints")
SOURCE_FILES = [
    "Synthetic_Data_Tensorflow_Advanced/mz_spectral/mz_decomposition.py",
    "Synthetic_Data_Tensorflow_Advanced/mz_spectral/quasi_linear.py",
    "Synthetic_Data_Tensorflow_Advanced/mz_spectral/validation.py",
    "Synthetic_Data_Tensorflow_Advanced/mz_spectral/fourier_dupire.py",
    "Synthetic_Data_Tensorflow_Advanced/analytical_solutions.py",
    "Synthetic_Data_Tensorflow_Advanced/config.py",
]


NUMERICAL_MODULES = ["dsm/operators.py", "dsm/cases.py", "dsm/reference.py", "dsm/mz.py",
                     "dsm/closures.py", "dsm/metrics.py"]
IDENTITY_SCHEMA = 2


def numerical_code_identity() -> str:
    """sha256 over the modules that determine every number (not the plumbing in this file or the
    gate scripts, whose numerical choices are captured by their ALGO_VERSION). The environment
    variable DSM_CODE_IDENTITY_OVERRIDE exists only for tests."""
    ov = os.environ.get("DSM_CODE_IDENTITY_OVERRIDE")
    if ov:
        return ov
    h = hashlib.sha256()
    for rel in NUMERICAL_MODULES:
        p = os.path.join(STUDY_ROOT, rel)
        h.update(rel.encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def make_identity(**fields) -> Dict[str, Any]:
    """Everything a result depends on: numerical code, inputs (case hash), grid, time step,
    refinement settings and algorithm version. Stored in the result and compared on reuse."""
    ident = {"schema": IDENTITY_SCHEMA, "code": numerical_code_identity()}
    ident.update(fields)
    return ident


def identity_hash(identity: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:16]


def result_status(path: str, identity: Dict[str, Any]):
    """('done' | 'missing' | 'corrupted' | 'stale' | 'failed' | 'incomplete', detail)."""
    if not os.path.isfile(path):
        return "missing", ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:  # unreadable or truncated
        return "corrupted", f"unreadable JSON: {e}"
    if not isinstance(d, dict) or "status" not in d:
        return "corrupted", "no status field"
    if d["status"] == "failed":
        return "failed", d.get("traceback", "")[-200:]
    if d["status"] != "done":
        return "incomplete", f"status={d['status']}"
    stored = d.get("identity")
    if stored is None or d.get("identity_hash") is None:
        return "stale", "legacy result without identity (schema 1); run scripts/stamp_results.py to attribute it or recompute"
    if d["identity_hash"] != identity_hash(identity):
        diff = sorted(k for k in set(stored) | set(identity) if stored.get(k) != identity.get(k))
        return "stale", f"identity differs in {diff}"
    if d["identity_hash"] != identity_hash(stored):
        return "corrupted", "identity_hash does not match the stored identity"
    return "done", ""


def supersede(path: str) -> Optional[str]:
    """Keep a stale/corrupted result on disk under a timestamped name (evidence is never deleted)."""
    if not os.path.isfile(path):
        return None
    new = path.replace(".json", "") + ".superseded." + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".json"
    os.replace(path, new)
    return new


class RunManifest:
    """Detects interrupted runs: in-progress case IDs are recorded before computing and cleared
    after the result is written; a manifest left with in-progress IDs and a dead pid means the
    previous run died there."""

    def __init__(self, gate: str, planned: List[str]):
        self.path = os.path.join(RESULTS, gate, "RUN_MANIFEST.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.previous_interrupted: Dict[str, Any] = {}
        old = read_json(self.path)
        if old and old.get("in_progress") and not _pid_alive(old.get("pid")):
            self.previous_interrupted = {"started_utc": old.get("started_utc"), "pid": old.get("pid"),
                                         "in_progress": old.get("in_progress"), "completed": old.get("completed", [])}
        self.data = {"gate": gate, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(),
                     "planned": planned, "in_progress": [], "completed": [], "failed": [], "finished": False,
                     "previous_interrupted": self.previous_interrupted}
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


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unavailable"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "-C", REPO_ROOT, "status", "--porcelain",
                                       "--", "Synthetic_Data_Tensorflow_Advanced", "research"], text=True)
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def source_identities() -> Dict[str, str]:
    ids = {}
    for rel in SOURCE_FILES:
        p = os.path.join(REPO_ROOT, rel)
        ids[rel] = sha256_file(p) if os.path.isfile(p) else "missing"
    for p in sorted(glob.glob(os.path.join(STUDY_ROOT, "dsm", "*.py")) +
                    glob.glob(os.path.join(STUDY_ROOT, "scripts", "*.py"))):
        ids[os.path.relpath(p, REPO_ROOT)] = sha256_file(p)
    return ids


def env_identity() -> Dict[str, str]:
    import numpy, scipy
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "machine": platform.machine()}


def write_json(path: str, obj: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_default)
    os.replace(tmp, path)
    return sha256_file(path)


def _default(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def result_is_done(path: str, config_hash: str) -> bool:
    """Legacy check (schema 1): case hash + status only. Superseded by result_status()."""
    d = read_json(path)
    return bool(d) and d.get("config_hash") == config_hash and d.get("status") == "done"


def result_checksums(subdir: str) -> Dict[str, str]:
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, subdir, "**", "*.json"), recursive=True)):
        out[os.path.relpath(p, OUTPUT_ROOT)] = sha256_file(p)
    return out


def save_checkpoint(gate: str, *, command: str, exit_code: int, done: List[str], failed: List[str],
                    pending: List[str], seeds: Dict[str, Any], config: Dict[str, Any],
                    extra: Optional[Dict[str, Any]] = None, next_command: str = "") -> str:
    ck = {
        "gate": gate,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "git_dirty_tracked_areas": git_dirty(),
        "source_identities": source_identities(),
        "environment": env_identity(),
        "config": config,
        "seeds": seeds,
        "cases": {"done": done, "failed": failed, "pending": pending},
        "command": command,
        "exit_code": exit_code,
        "result_checksums": result_checksums(gate),
        "next_command": next_command,
        "extra": extra or {},
    }
    path = os.path.join(CHECKPOINTS, f"CHECKPOINT_{gate}.json")
    write_json(path, ck)
    write_json(os.path.join(CHECKPOINTS, "CHECKPOINT_LATEST.json"), ck)
    write_next_actions(gate, next_command, pending, failed)
    return path


def write_next_actions(gate: str, next_command: str, pending: List[str], failed: List[str]) -> None:
    lines = [
        "# NEXT_ACTIONS",
        "",
        f"Last completed checkpoint: `{gate}` (see `checkpoints/CHECKPOINT_LATEST.json`).",
        "",
        "Exactly one safe next command (idempotent; from the repository root, Mac/Linux, no account needed):",
        "",
        "```bash",
        next_command or "python3 research/dupire_signed_memory/scripts/run_all.py --resume",
        "```",
        "",
        f"Pending case IDs: {pending if pending else 'none'}",
        f"Failed case IDs: {failed if failed else 'none'}",
        "",
        "Re-running the command re-uses every result whose stored config hash matches and recomputes only",
        "missing or mismatched cases; it never overwrites a committed earlier checkpoint file.",
    ]
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "NEXT_ACTIONS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
