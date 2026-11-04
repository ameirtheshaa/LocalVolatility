import glob
import json
import os
import subprocess
import sys

from conftest import OUTPUT_ROOT, STUDY

REPO_ROOT = os.path.dirname(os.path.dirname(STUDY))
SCRIPT = os.path.join(STUDY, "scripts", "gate2_reference.py")
ARGS = ["--only", "C_sig0.20,C_sig0.30", "--variants", "B", "--no-refine", "--n_k", "64",
        "--sub", "4", "--resume"]


def run(max_cases):
    env = dict(os.environ, DSM_OUTPUT_ROOT=OUTPUT_ROOT)
    p = subprocess.run([sys.executable, SCRIPT, *ARGS, "--max-cases", str(max_cases)],
                       cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    return p


def files():
    return sorted(glob.glob(os.path.join(OUTPUT_ROOT, "results", "gate2", "C_sig*.json")))


def sha(path):
    from dsm.checkpoint import sha256_file
    return sha256_file(path)


def ckpt():
    with open(os.path.join(OUTPUT_ROOT, "checkpoints", "CHECKPOINT_gate2.json")) as f:
        return json.load(f)["cases"]


def test_interrupt_resume_idempotent():
    assert not STUDY.startswith(OUTPUT_ROOT) and not OUTPUT_ROOT.startswith(STUDY)
    run(1)
    fs = files()
    assert len(fs) == 1
    with open(fs[0]) as f:
        assert json.load(f)["status"] == "done"
    c = ckpt()
    assert len(c["done"]) == 1 and len(c["pending"]) == 1
    na = os.path.join(OUTPUT_ROOT, "NEXT_ACTIONS.md")
    assert os.path.isfile(na)
    with open(na) as f:
        assert any(l.startswith("python3 research/dupire_signed_memory/scripts/") for l in f.read().splitlines())
    first, h1 = fs[0], sha(fs[0])

    run(0)
    fs = files()
    assert len(fs) == 2
    for p in fs:
        with open(p) as f:
            assert json.load(f)["status"] == "done"
    assert sha(first) == h1
    c = ckpt()
    assert len(c["done"]) == 2 and c["pending"] == []
    before = {p: sha(p) for p in fs}

    run(0)
    assert {p: sha(p) for p in files()} == before
    with open(os.path.join(OUTPUT_ROOT, "results", "gate2", "summary.json")) as f:
        assert len(json.load(f)["done"]) == 2
