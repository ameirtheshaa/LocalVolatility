import numpy as np
import pytest

from scc import reference as R
from scc import cases as C
from dsm import cases as dc


@pytest.fixture(scope="module")
def g64():
    g = R.make_grid(64)
    return g, R.fine_grid(g, 4)


def test_nested_index_and_k():
    assert np.array_equal(R.nested_index(511, 256), np.arange(0, 511, 2))
    assert np.array_equal(R.nested_index(1021, 256), np.arange(0, 1021, 4))
    g_f, g_c = R.make_grid(511), R.make_grid(256)
    idx = R.nested_index_grids(g_f, g_c)
    assert np.max(np.abs(g_f.k[idx] - g_c.k)) < 1e-12
    with pytest.raises(AssertionError):
        R.nested_index(500, 256)
    rows = np.arange(2 * 511.0).reshape(2, 511)
    assert R.restrict_rows(rows, idx).shape == (2, 256)


def test_operators_dense_sparse_agree(g64):
    g, _ = g64
    for c in (dc.get_case("C_sig0.30"), dc.get_case("D_base0.30")):
        Ld, Ls = R.L_full_dense(c, g, 0.5), R.L_full_sparse(c, g, 0.5).toarray()
        assert np.max(np.abs(Ld - Ls)) < 1e-11 * np.max(np.abs(Ld))
        assert np.all(Ls[0] == 0) and np.all(Ls[-1] == 0)
    d = R.coefficient_full(c, g, 0.5)
    assert d.shape == (g.n_k,)


@pytest.mark.parametrize("case_id", ["C_sig0.30", "D_base0.30"])
def test_dense_vs_sparse(g64, case_id):
    g, fg = g64
    c = dc.get_case(case_id)
    a, b = R.solve_dense(c, g, fg), R.solve_sparse(c, g, fg)
    assert a.shape == b.shape == (len(fg.t), 64)
    assert np.max(np.abs(a - b)) < 1e-11
    chk = R.solver_check(c, g, fg)
    assert chk["max_abs_diff_dense_sparse"] < 1e-11 and chk["max_abs_phi"] > 0


def test_scc_cases_dense_vs_sparse(g64):
    g, fg = g64
    for cid in ("S2_seed100", "M_seed300"):
        c = C.get_case(cid)
        assert np.max(np.abs(R.solve_dense(c, g, fg) - R.solve_sparse(c, g, fg))) < 1e-11


def test_cn_converges_to_dense(g64):
    g, fg = g64
    for c in (dc.get_case("C_sig0.30"), dc.get_case("D_base0.30")):
        ref = R.solve_dense(c, g, fg)
        errs = [np.max(np.abs(R.solve_banded_cn(c, g, fg, s) - ref)) for s in (1, 2, 4)]
        assert errs[0] > errs[1] > errs[2]
        assert errs[1] / errs[2] > 3.0           # roughly second order
        # endpoints stay fixed
        cn = R.solve_banded_cn(c, g, fg, 2)
        assert np.all(cn[:, 0] == cn[0, 0]) and np.all(cn[:, -1] == cn[0, -1])


def test_dense_vs_black_scholes(g64):
    g, fg = g64
    c = dc.get_case("C_sig0.30")
    err = np.max(np.abs(R.solve_dense(c, g, fg) - R.bs_rows(g, 0.30, fg.t)))
    assert np.isfinite(err) and err < 5e-3


def test_get_reference_cache():
    c = dc.get_case("C_sig0.30")
    a = R.get_reference(c, 64, 4)
    assert a["from_cache"] is False and a["phi"].shape == (len(a["t"]), 64) and len(a["out_idx"]) == 24
    b = R.get_reference(c, 64, 4)
    assert b["from_cache"] is True and np.array_equal(a["phi"], b["phi"]) and b["i0"] == a["i0"] and b["h"] == a["h"]
    d = R.get_reference(c, 64, 4, force=True)
    assert d["from_cache"] is False and np.array_equal(a["phi"], d["phi"])
    s = R.get_reference(c, 64, 4, solver="sparse")
    assert s["from_cache"] is False and s["identity"]["solver"] == "sparse"
    assert np.max(np.abs(s["phi"] - a["phi"])) < 1e-11


def test_get_reference_phi0_tag_and_guard(monkeypatch):
    c = dc.get_case("C_sig0.30")
    g = R.make_grid(64)
    with pytest.raises(ValueError):
        R.get_reference(c, 64, 4, phi0=g.payoff())
    r = R.get_reference(c, 64, 4, phi0=g.payoff() * 0.5, phi0_tag="half")
    assert r["identity"]["phi0"] == "half"
    t = C.get_case("C_sig0.205")
    monkeypatch.setenv("SCC_PHASE", "fit")
    with pytest.raises(RuntimeError):
        R.get_reference(t, 64, 4)
    R.get_reference(C.get_case("C_sig0.180"), 64, 4)
