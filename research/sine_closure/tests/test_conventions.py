import numpy as np
import pytest

import scc  # noqa: F401
from scc import conventions as cv
from dsm import operators as op
from dsm.cases import get_case

N, M = 64, 8


@pytest.fixture(scope="module")
def setup():
    grid = op.build_grid(n_k=N)
    case = get_case("C_sig0.30")
    sys_ = cv.interior_system(case, grid)
    V = cv.dst_interior(N, M)
    U = cv.dst_complement_interior(N, M)
    lam = cv.sine_eigenvalues(N, grid.dk, np.arange(1, M + 1))
    return grid, case, sys_, V, U, lam


def test_orthonormality_and_endpoints(setup):
    grid, _, _, V, U, _ = setup
    assert np.max(np.abs(V.T @ V - np.eye(M))) < 1e-13
    assert np.max(np.abs(U.T @ U - np.eye(N - 2 - M))) < 1e-13
    assert np.max(np.abs(V.T @ U)) < 1e-13
    full = cv.dst_basis(N, M)
    assert np.all(full[0] == 0.0) and np.all(full[-1] == 0.0)
    assert np.array_equal(full[1:-1], V)
    with pytest.raises(ValueError):
        cv.dst_basis(N, N - 1)


def test_eigen_relation(setup):
    grid, _, sys_, V, _, lam = setup
    lhs = sys_.D2_int @ V
    rhs = V * lam[None, :]
    assert np.max(np.abs(lhs - rhs)) < 1e-10 * np.max(np.abs(rhs))
    d2l = op.d2_linearity(N, grid.dk)[1:-1, 1:-1]
    assert np.array_equal(d2l, sys_.D2_int)
    assert sys_.D2_bnd.shape == (N - 2, 2)


def test_A_structured_and_S(setup):
    _, _, sys_, V, U, lam = setup
    for t in (0.0, 0.5):
        bl = cv.blocks(sys_, t, V, U)
        As = cv.A_structured(sys_, t, V, lam)
        assert np.max(np.abs(As - bl["A"])) < 1e-12 * np.max(np.abs(bl["A"]))
        S = cv.S_matrix(sys_, t, V)
        assert np.allclose(S, S.T) and np.linalg.eigvalsh(S).min() > 0


def test_forcing_affine_vanishes(setup):
    grid, _, sys_, _, _, _ = setup
    phi0, ell, psi0 = cv.payoff_state(grid)
    assert np.array_equal(phi0, grid.payoff())
    assert np.allclose(ell + psi0, phi0)
    g = cv.forcing_g(sys_, 0.3, ell)
    tol = 1e-12 * np.max(np.abs(sys_.d(0.3))) * np.max(np.abs(sys_.D2_int)) * np.max(np.abs(ell))
    assert np.max(np.abs(g)) <= tol


def test_forcing_time_dependent_lift(setup):
    grid, _, sys_, _, _, _ = setup
    ell = 1.0 + 2.0 * grid.k
    ell_dot = 0.3 + 0.7 * grid.k
    g = cv.forcing_g(sys_, 0.3, ell, ell_dot=ell_dot)
    assert np.max(np.abs(g)) > 1e-3
    g0 = cv.forcing_g(sys_, 0.3, ell)
    assert np.allclose(g, g0 - ell_dot[1:-1], atol=0, rtol=0)
    b = np.arange(N - 2) * 0.01
    assert np.allclose(cv.forcing_g(sys_, 0.3, ell, b_int=b), g0 + b)


def test_lift_detrend():
    k = np.linspace(0, 1, 9)
    rows = np.random.default_rng(1).standard_normal((3, 9))
    L = cv.lift_rows(rows, k)
    assert np.allclose(L[:, 0], rows[:, 0]) and np.allclose(L[:, -1], rows[:, -1])
    assert np.allclose(L, op.detrend_rows(rows, k) * -1 + rows)
    assert np.allclose(cv.detrend_rows(rows[0], k), op.detrend_rows(rows, k)[0])
    assert cv.lift_rows(rows[0], k).shape == (9,)


def test_memory_target_and_reconstruction(setup):
    grid, case, sys_, V, U, _ = setup
    fg = __import__("dsm.reference", fromlist=["x"]).fine_grid(grid, sub=4)
    from dsm import reference as ref
    phi = ref.solve_full(case, grid, "B", fg)
    psi_int = op.detrend_rows(phi, grid.k)[:, 1:-1]
    c, z = psi_int @ V, psi_int @ U
    assert np.max(np.abs(psi_int - c @ V.T - z @ U.T)) < 1e-12 * np.max(np.abs(psi_int))
    Mt = cv.memory_target(sys_, 0.5, V, U, psi_int)
    B = cv.blocks(sys_, 0.5, V, U)["B"]
    alt = (B @ z.T).T
    assert np.max(np.abs(Mt - alt)) < 1e-12 * np.max(np.abs(alt))
    w = psi_int - (psi_int @ V) @ V.T
    alt2 = (sys_.d(0.5)[:, None] * (sys_.D2_int @ w.T)).T @ V
    assert np.max(np.abs(Mt - alt2)) < 1e-12 * np.max(np.abs(alt2))
    assert cv.memory_target(sys_, 0.5, V, U, psi_int[5]).shape == (1, M)


def test_time_dependent_d_and_timewise():
    grid = op.build_grid(n_k=N)
    case = get_case("D_base0.30")
    s = cv.interior_system(case, grid)
    assert s.time_dependent
    assert np.max(np.abs(s.d(0.2) - s.d(0.9))) > 1e-6
    assert s.d_full(0.5).shape == (N,)
    V, U = cv.dst_interior(N, M), cv.dst_complement_interior(N, M)
    assert len(cv.resolved_blocks_timewise(s, V, U, [0.2, 0.5, 0.9])) == 3
    sc = cv.interior_system(get_case("C_sig0.30"), grid)
    assert len(cv.resolved_blocks_timewise(sc, V, U, [0.2, 0.5, 0.9])) == 1


def test_phi1_step():
    from dsm.mz import phi1_step as ref_step
    rng = np.random.default_rng(2)
    A = rng.standard_normal((5, 5)) - 3 * np.eye(5)
    g = rng.standard_normal(5)
    E, v = cv.phi1_step(A, 0.1, g)
    E2, v2 = ref_step(A, 0.1, g)
    assert np.allclose(E, E2, atol=1e-14) and np.allclose(v, v2, atol=1e-14)
    assert np.allclose(cv.expm_h(A, 0.1), E)
