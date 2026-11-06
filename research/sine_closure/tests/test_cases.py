import copy
import hashlib
import json
import re

import numpy as np

from scc import cases as C
from scc import reference as R
from dsm import cases as dc
from dsm import repo

GRID = R.make_grid(64)


def test_split_counts():
    assert C.split_counts() == {"train": 28, "val": 10, "test_interp": 17, "test_ood": 10, "dev_regression": 3}
    assert len(C.all_cases()) == 68
    assert len({c.id for c in C.all_cases()}) == 68


def test_tmax_matches_repo():
    assert C.T_MAX == repo.DupirePipelineConfig().T_max


def test_determinism():
    a, b = C.all_cases(), C.all_cases()
    assert [(c.id, c.params, c.config_hash()) for c in a] == [(c.id, c.params, c.config_hash()) for c in b]
    assert C.manifest()["sha256"] == C.manifest()["sha256"]


def test_ids_and_lookup():
    assert C.get_case("C_sig0.180").params == {"sigma": 0.18}
    assert C.get_case("S2_seed100").split == "train"
    assert C.get_case("D2_seed231").split == "test_ood"
    assert C.get_case("M_seed331").split == "test_ood"
    assert C.get_case("M_seed331").time_dependent and not C.get_case("S2_seed100").time_dependent
    try:
        C.get_case("nope")
        assert False
    except KeyError:
        pass
    assert {c.id for c in C.cases_by_split("dev_regression")} == {"C_sig0.30", "S_seed9", "F_fresh_base0.28_t0.2_x0.05"}


def test_params_in_ranges_and_rounded():
    for c in C.family_S():
        lo, hi = (C._S_RANGES["ood"] if c.split == "test_ood" else C._S_RANGES["interp"])[0]
        assert lo <= c.params["b"] <= hi
        assert all(round(v, 6) == v for k, v in c.params.items() if k != "seed")
    for c in C.family_M():
        assert 0 <= c.params["phi"] <= 2 * np.pi


def test_sigma_positive_all_cases():
    for c in C.all_cases():
        ch = C.sigma_checks(c, GRID)
        assert ch["sigma_min"] > 0 and ch["sigma_max"] <= 2.0 + 1e-12, c.id
        assert 0.0 <= ch["clip_active_fraction"] <= 1.0
        if c.family in ("C", "D"):
            assert ch["clip_active_fraction"] == 0.0


def test_D_matches_dsm_formula():
    c = C.get_case("D2_seed200")
    p = c.params
    ref = dc._smile_D(p["sigma_base"], p["t_shift"], p["x_shift"])
    x = np.linspace(0.3, 2.5, 17)
    for T in (0.0, 0.4, 1.5):
        assert np.array_equal(c.sigma_fn(T, x), ref(T, x))


def test_M_reduces_to_S_when_a_zero():
    p = dict(C.get_case("M_seed300").params)
    s_fn = C._smile_S(p["b"], p["s"], p["c"])
    m_fn = C._smile_M(p["b"], 0.0, p["f"], p["phi"], p["s"], p["c"])
    x = np.linspace(0.3, 2.5, 17)
    for T in (0.0, 0.7, 1.5):
        assert np.allclose(m_fn(T, x), s_fn(T, x), rtol=0, atol=1e-14)
    # and the real M case varies in time
    c = C.get_case("M_seed300")
    assert not np.allclose(c.sigma_fn(0.0, x), c.sigma_fn(0.4, x))


def test_gate5_regression_matches_gate5_definitions():
    g = {c.id: c for c in C.gate5_regression()}
    assert list(g) == ["C_sig0.30", "S_seed9", "F_fresh_base0.28_t0.2_x0.05"]
    assert g["C_sig0.30"].params == dc.get_case("C_sig0.30").params
    assert g["S_seed9"].params == dc.get_case("S_seed9").params
    assert g["F_fresh_base0.28_t0.2_x0.05"].params == {"sigma_base": 0.28, "t_shift": 0.2, "x_shift": 0.05}
    assert g["F_fresh_base0.28_t0.2_x0.05"].time_dependent and not g["S_seed9"].time_dependent
    assert all(c.split == "dev_regression" and c.generator == "gate5" for c in g.values())
    x = np.linspace(0.3, 2.5, 9)
    fresh = dc._smile_D(0.28, t_shift=0.2, x_shift=0.05)
    assert np.array_equal(g["F_fresh_base0.28_t0.2_x0.05"].sigma_fn(0.6, x), fresh(0.6, x))
    assert np.array_equal(g["S_seed9"].sigma_fn(0.6, x), dc.get_case("S_seed9").sigma_fn(0.6, x))


def test_manifest_sha_and_sensitivity(tmp_path):
    m = C.manifest(GRID)
    assert re.fullmatch(r"[0-9a-f]{64}", m["sha256"])
    assert m["counts"] == C.split_counts()
    assert all("checks" in r for r in m["cases"])
    assert set(m["cases"][0]) >= {"id", "family", "params", "split", "generator", "config_hash", "time_dependent"}
    # change a param -> sha changes
    orig = C.family_S

    def patched():
        out = orig()
        out[0].params = dict(out[0].params, b=out[0].params["b"] + 1e-6)
        return out
    C.family_S = patched
    try:
        assert C.manifest()["sha256"] != m["sha256"]
    finally:
        C.family_S = orig
    assert C.manifest()["sha256"] == m["sha256"]
    p = tmp_path / "CASES.json"
    C.write_manifest(str(p), GRID)
    assert json.loads(p.read_text())["sha256"] == m["sha256"]


def test_config_hash_depends_on_generator():
    c = C.get_case("C_sig0.180")
    c2 = copy.copy(c); c2.generator = "other"
    assert c.config_hash() != c2.config_hash()
