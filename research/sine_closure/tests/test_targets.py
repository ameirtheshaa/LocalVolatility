import numpy as np
import pytest

import scc  # noqa: F401
from scc import conventions as cv
from scc import oracles as orc
from scc import targets as tgt
from dsm import operators as op
from dsm import reference as ref
from dsm.cases import get_case

N_K, SUB = 64, 4


def make_ref(case_id, m, sub=SUB):
    grid = op.build_grid(n_k=N_K)
    case = get_case(case_id)
    fg = ref.fine_grid(grid, sub=sub)
    phi = ref.solve_full(case, grid, "B", fg)
    r = {"phi": phi, "t": fg.t, "h": fg.h, "i0": fg.i0, "out_idx": fg.out_idx}
    return grid, case, fg, r, tgt.build_targets(case, r, grid, m)


@pytest.fixture(scope="module")
def C_setup():
    return make_ref("C_sig0.30", 8)


def test_shapes(C_setup):
    grid, case, fg, r, T = C_setup
    Np1 = len(fg.t)
    assert T["c"].shape == (Np1, 8) and T["M"].shape == (Np1, 8)
    assert T["z"].shape == (Np1, N_K - 2 - 8)
    assert T["t"].shape == (Np1,) and T["ell"].shape == (N_K,) and T["psi0"].shape == (N_K,)
    assert T["lam_V"].shape == (8,) and T["V_int"].shape == (N_K - 2, 8)
    expect = np.sqrt(np.mean(np.sum(T["M"] ** 2, axis=1) / 8))
    assert T["rms_M"] == pytest.approx(expect, rel=1e-14)


def test_M_equals_Bz(C_setup):
    grid, case, fg, r, T = C_setup
    B = cv.blocks(T["sys"], 0.0, T["V_int"], T["U_int"])["B"]
    Mz = T["z"] @ B.T
    assert np.max(np.abs(Mz - T["M"])) <= 1e-12 * np.max(np.abs(T["M"]))


def test_projected_unresolved_helpers(C_setup):
    grid, case, fg, r, T = C_setup
    psi = cv.detrend_rows(r["phi"], grid.k)[:, 1:-1]
    assert np.array_equal(tgt.projected_state(psi, T["V_int"]), T["c"])
    assert np.array_equal(tgt.unresolved_state(psi, T["U_int"]), T["z"])


def test_trajectory_weights_formula():
    rms = {"a": 1.0, "b": 2.0, "c": 1e-6}
    nr = {"a": 100, "b": 50, "c": 10}
    w = tgt.trajectory_weights(rms, nr)
    med = 1.0
    assert w["a"] == pytest.approx(1 / (100 * 1.0 ** 2))
    assert w["b"] == pytest.approx(1 / (50 * 4.0))
    assert w["c"] == pytest.approx(1 / (10 * (0.1 * med) ** 2))


def test_d_extremes():
    grid = op.build_grid(n_k=N_K)
    cases = [get_case("C_sig0.30"), get_case("D_base0.30")]
    lo, hi = tgt.d_extremes(cases, grid, [0.2, 0.6, 1.0])
    assert 0 < lo < hi
    s = cv.interior_system(cases[0], grid)
    d = s.d(0.2)
    assert lo <= d.min() and hi >= d.max()


def test_time_dependent_targets():
    grid, case, fg, r, T = make_ref("D_base0.30", 6)
    assert T["M"].shape == (len(fg.t), 6)
    i = len(fg.t) // 2
    B = cv.blocks(T["sys"], float(fg.t[i]), T["V_int"], T["U_int"])["B"]
    assert np.max(np.abs(B @ T["z"][i] - T["M"][i])) <= 1e-12 * np.max(np.abs(T["M"]))
