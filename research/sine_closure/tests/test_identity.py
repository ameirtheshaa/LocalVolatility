import json
import os

import numpy as np
import pytest

from scc import identity as I


def test_make_identity_and_hash():
    a = I.make_identity(x=1, y="a")
    assert a["schema"] == 1 and a["basis"] == "dst1_v1" and a["lift"] == "affine_endpoints_v1" and a["operator"] == "OP-B"
    assert I.identity_hash(a) == I.identity_hash(dict(reversed(list(a.items()))))
    assert I.identity_hash(a) != I.identity_hash(I.make_identity(x=2, y="a"))


def test_code_identity_override(monkeypatch):
    base = I.code_identity()
    assert len(base) == 16
    monkeypatch.setenv("SCC_CODE_IDENTITY_OVERRIDE", "abc")
    assert I.code_identity() == "abc"


def test_cache_round_trip():
    ident = I.make_identity(test="roundtrip", n=3)
    a, b = np.arange(6.0).reshape(2, 3), np.array([1, 2, 3])
    p = I.cache_store("unit", ident, a=a, b=b)
    assert p == I.cache_path("unit", ident) and os.path.isfile(p)
    side = json.load(open(p[:-4] + ".json"))
    assert side["identity_hash"] == I.identity_hash(ident) and side["shapes"] == {"a": [2, 3], "b": [3]}
    assert {"created_utc", "numpy", "scipy"} <= set(side)
    z = I.cache_load("unit", ident)
    assert np.array_equal(z["a"], a) and np.array_equal(z["b"], b)
    assert I.cache_load("unit", I.make_identity(test="other")) is None
    assert I.cache_load("nokind", ident) is None


def test_cache_hash_mismatch_and_corruption():
    ident = I.make_identity(test="mismatch")
    p = I.cache_store("unit", ident, a=np.ones(4))
    sp = p[:-4] + ".json"
    side = json.load(open(sp))
    side["identity_hash"] = "0" * 16
    json.dump(side, open(sp, "w"))
    assert I.cache_load("unit", ident) is None
    open(sp, "w").write("{not json")
    assert I.cache_load("unit", ident) is None
    # shape mismatch
    I.cache_store("unit", ident, a=np.ones(4))
    side = json.load(open(sp)); side["shapes"]["a"] = [5]; json.dump(side, open(sp, "w"))
    assert I.cache_load("unit", ident) is None
    # corrupted npz
    I.cache_store("unit", ident, a=np.ones(4))
    open(p, "wb").write(b"garbage")
    assert I.cache_load("unit", ident) is None


def test_write_read_json(tmp_path):
    p = str(tmp_path / "sub" / "x.json")
    h = I.write_json(p, {"a": np.float64(1.5), "b": np.arange(3), "c": np.bool_(True)})
    assert h == I.sha256_file(p) and I.read_json(p) == {"a": 1.5, "b": [0, 1, 2], "c": True}
    assert I.read_json(str(tmp_path / "none.json")) is None
    assert not os.path.exists(p + ".tmp")


def test_result_status_and_supersede(tmp_path):
    p = str(tmp_path / "r.json")
    ident = I.make_identity(t=1)
    assert I.result_status(p, ident)[0] == "missing"
    open(p, "w").write("{")
    assert I.result_status(p, ident)[0] == "corrupted"
    I.write_json(p, {"status": "done", "identity": ident, "identity_hash": I.identity_hash(ident)})
    assert I.result_status(p, ident)[0] == "done"
    assert I.result_status(p, I.make_identity(t=2))[0] == "stale"
    I.write_json(p, {"status": "failed", "traceback": "boom"})
    assert I.result_status(p, ident)[0] == "failed"
    I.write_json(p, {"status": "running"})
    assert I.result_status(p, ident)[0] == "incomplete"
    new = I.supersede(p)
    assert new and os.path.isfile(new) and not os.path.exists(p)
    assert I.supersede(p) is None


def test_run_manifest():
    m = I.RunManifest("unit_stage", ["a", "b"])
    m.start("a"); m.done("a"); m.start("b")
    d = I.read_json(m.path)
    assert d["in_progress"] == ["b"] and d["completed"] == ["a"]
    m.failed("b"); m.finish()
    d = I.read_json(m.path)
    assert d["failed"] == ["b"] and d["finished"] and d["in_progress"] == []
    # simulate a dead previous run
    dead = I.read_json(m.path); dead.update(in_progress=["z"], pid=2 ** 22 + 12345)
    I.write_json(m.path, dead)
    assert I.RunManifest("unit_stage", ["a"]).previous_interrupted["in_progress"] == ["z"]


def test_phase_guard(monkeypatch):
    monkeypatch.setenv("SCC_PHASE", "fit")
    with pytest.raises(RuntimeError):
        I.assert_phase_allows("test_interp")
    with pytest.raises(RuntimeError):
        I.assert_phase_allows("test_ood")
    I.assert_phase_allows("train")
    monkeypatch.delenv("SCC_PHASE")
    I.assert_phase_allows("test_interp")


def test_misc_identities():
    assert isinstance(I.git_head(), str)
    assert {"python", "numpy", "scipy"} <= set(I.env_identity())
    sh = I.source_hashes()
    assert any(k.endswith("scc/identity.py") for k in sh) and any(k.endswith("dsm/operators.py") for k in sh)
