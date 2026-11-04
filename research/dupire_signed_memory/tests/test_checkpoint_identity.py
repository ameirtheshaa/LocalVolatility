"""Checkpoint reuse must depend on inputs, grid, time step, refinement, code identity and file
integrity; missing, corrupted and interrupted cases must be detected and recomputed."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from conftest import OUTPUT_ROOT  # noqa: F401  (sets DSM_OUTPUT_ROOT before dsm is imported)

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(STUDY))
SCRIPT = os.path.join(STUDY, "scripts", "gate2_reference.py")
BASE = ["--only", "C_sig0.40", "--variants", "B", "--no-refine", "--resume"]


def run(extra, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([sys.executable, SCRIPT, *BASE, *extra], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.load(open(os.path.join(os.environ["DSM_OUTPUT_ROOT"], "results", "gate2", "summary.json")))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("identity_root"))
    os.environ["DSM_OUTPUT_ROOT"] = out          # fresh root for this module's subprocess runs
    return {"res": os.path.join(out, "results", "gate2", "C_sig0.40__B.json"),
            "manifest": os.path.join(out, "results", "gate2", "RUN_MANIFEST.json"),
            "dir": os.path.join(out, "results", "gate2")}


def test_reuse_then_invalidation_chain(paths):
    s1 = run(["--n_k", "64", "--sub", "4"])
    assert s1["computed"] == 1 and s1["status_before_run"]["missing"] == 1
    h1 = sha(paths["res"])
    d = json.load(open(paths["res"]))
    assert d["identity_hash"] and d["identity"]["n_k"] == 64 and d["identity"]["sub"] == 4
    # identical configuration: reused, file untouched
    s2 = run(["--n_k", "64", "--sub", "4"])
    assert s2["computed"] == 0 and s2["reused"] == ["C_sig0.40__B"] and sha(paths["res"]) == h1
    # grid change: stale, recomputed, old file kept
    s3 = run(["--n_k", "48", "--sub", "4"])
    assert s3["computed"] == 1 and s3["status_before_run"]["stale"] == 1
    assert "n_k" in s3["invalidated"]["C_sig0.40__B"]["detail"]
    assert any(f.startswith("C_sig0.40__B.superseded.") for f in os.listdir(paths["dir"]))
    # time-step change: stale again
    s4 = run(["--n_k", "48", "--sub", "8"])
    assert s4["computed"] == 1 and "sub" in s4["invalidated"]["C_sig0.40__B"]["detail"]
    # code identity change (test-only override): stale
    s5 = run(["--n_k", "48", "--sub", "8"], env_extra={"DSM_CODE_IDENTITY_OVERRIDE": "deadbeefdeadbeef"})
    assert s5["computed"] == 1 and "code" in s5["invalidated"]["C_sig0.40__B"]["detail"]
    # back to the real code: the override result is not reusable either
    s6 = run(["--n_k", "48", "--sub", "8"])
    assert s6["computed"] == 1
    # corrupted file (truncated JSON): detected and recomputed
    txt = open(paths["res"]).read()
    with open(paths["res"], "w") as f:
        f.write(txt[:200])
    s7 = run(["--n_k", "48", "--sub", "8"])
    assert s7["status_before_run"]["corrupted"] == 1 and s7["computed"] == 1
    # tampered identity: detected (stale or corrupted), recomputed
    d = json.load(open(paths["res"]))
    d["identity"]["n_k"] = 999
    json.dump(d, open(paths["res"], "w"))
    s8 = run(["--n_k", "48", "--sub", "8"])
    assert s8["status_before_run"]["corrupted"] + s8["status_before_run"]["stale"] == 1 and s8["computed"] == 1
    # missing file: counted as missing and recomputed
    os.remove(paths["res"])
    s9 = run(["--n_k", "48", "--sub", "8"])
    assert s9["status_before_run"]["missing"] == 1 and s9["computed"] == 1
    # simulated interruption: manifest left with an in-progress case and a dead pid
    man = json.load(open(paths["manifest"]))
    man["in_progress"] = ["C_sig0.40__B"]
    man["pid"] = 2 ** 22 - 1
    man["finished"] = False
    json.dump(man, open(paths["manifest"], "w"))
    os.remove(paths["res"])
    s10 = run(["--n_k", "48", "--sub", "8"])
    assert s10["previous_interrupted"]["in_progress"] == ["C_sig0.40__B"] and s10["computed"] == 1
    man = json.load(open(paths["manifest"]))
    assert man["finished"] and man["in_progress"] == []
    # legacy schema-1 file (no identity): stale, recomputed, old file kept
    d = json.load(open(paths["res"]))
    d.pop("identity")
    d.pop("identity_hash")
    json.dump(d, open(paths["res"], "w"))
    s11 = run(["--n_k", "48", "--sub", "8"])
    assert s11["status_before_run"]["stale"] == 1 and "legacy" in s11["invalidated"]["C_sig0.40__B"]["detail"]
    # a run with --max-cases 0 after everything: nothing recomputed
    s12 = run(["--n_k", "48", "--sub", "8"])
    assert s12["computed"] == 0 and s12["reused"] == ["C_sig0.40__B"]
