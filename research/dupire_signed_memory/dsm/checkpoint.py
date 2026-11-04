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

RESULTS = os.path.join(STUDY_ROOT, "results")
CHECKPOINTS = os.path.join(STUDY_ROOT, "checkpoints")
SOURCE_FILES = [
    "Synthetic_Data_Tensorflow_Advanced/mz_spectral/mz_decomposition.py",
    "Synthetic_Data_Tensorflow_Advanced/mz_spectral/quasi_linear.py",
    "Synthetic_Data_Tensorflow_Advanced/mz_spectral/validation.py",
    "Synthetic_Data_Tensorflow_Advanced/mz_spectral/fourier_dupire.py",
    "Synthetic_Data_Tensorflow_Advanced/analytical_solutions.py",
    "Synthetic_Data_Tensorflow_Advanced/config.py",
]


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
    d = read_json(path)
    return bool(d) and d.get("config_hash") == config_hash and d.get("status") == "done"


def result_checksums(subdir: str) -> Dict[str, str]:
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, subdir, "**", "*.json"), recursive=True)):
        out[os.path.relpath(p, STUDY_ROOT)] = sha256_file(p)
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
    with open(os.path.join(STUDY_ROOT, "NEXT_ACTIONS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
