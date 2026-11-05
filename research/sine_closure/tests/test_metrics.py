import numpy as np
import pytest

import scc  # noqa: F401
from dsm import operators as op
from dsm import repo
from dsm.reference import bs_truth
from scc import metrics as M
from scc import cost

SIG = 0.3
_trapz = getattr(np, "trapezoid", None) or np.trapz


@pytest.fixture(scope="module")
def setup():
    grid = op.build_grid(n_k=128)
    ctx = M.make_context(grid)
    phi = bs_truth(grid, SIG, grid.t_out)
    return grid, ctx, phi


def test_price_zero(setup):
    grid, ctx, phi = setup
    out = M.price_metrics(phi, phi, ctx, bs_rows=phi)
    for k in ("price_interior_rms", "price_full_rms", "price_max_abs", "price_atm_rms", "price_wing_rms",
              "price_physical_interior_rms", "price_interior_rms_vs_bs"):
        assert out[k] == 0.0
    assert len(out["price_per_maturity_rms"]) == len(grid.T_out)
    out2 = M.price_metrics(phi + 1e-3, phi, ctx)
    assert out2["price_interior_rms"] == pytest.approx(1e-3)
    assert out2["price_physical_interior_rms"] == pytest.approx(1e-3 * grid.S0)
    assert "price_interior_rms_vs_bs" not in out2


def test_density_bs(setup):
    grid, ctx, phi = setup
    f = M.density_rows(phi, ctx, "d2_repo")
    assert f[:, ctx.interior[0]].min() > -1e-12 * np.abs(f).max() or (f * ctx.interior).min() >= -1e-9
    assert (np.where(ctx.interior, f, 0.0)).min() >= -1e-9
    # lognormal mass over band
    T = ctx.T_rows[-1]
    K = ctx.K_rows[-1]
    m = ctx.interior[-1]
    ln = repo.lognormal_density(K, grid.S0, T, grid.r, SIG)
    band_ln = _trapz(ln[m], K[m])
    band_f = _trapz(f[-1][m], K[m])
    assert band_f == pytest.approx(band_ln, rel=0.05)
    dm = M.density_metrics(f, f, ctx)
    assert 0.85 < dm["mass_final"] < 1.0
    assert dm["density_l2_interior_mean"] == 0.0
    assert dm["negative_mass_final"] >= 0.0
    assert len(dm["negative_mass"]) == len(grid.T_out) and len(dm["signed_mass"]) == len(grid.T_out)
    assert 0.0 <= dm["tail_mass_outside_interior_final"] < 0.2
    test_density_bs.numbers = (dm["mass_final"],)


def test_arbitrage_raw_bs(setup):
    grid, ctx, phi = setup
    dt = 1e-4
    phi_t = (bs_truth(grid, SIG, grid.t_out + dt) - bs_truth(grid, SIG, grid.t_out - dt)) / (2 * dt)
    out = M.arbitrage_metrics(phi, phi_t, ctx)
    for check in ("positivity", "monotonicity", "butterfly", "calendar"):
        assert out[f"{check}_raw_interior_frac_mean"] == 0.0
        assert out[f"{check}_raw_interior_frac_max"] == 0.0
    assert "butterfly_repo_full_frac_max" in out and len(out) == 4 * 2 * 2 * 2


def test_inverse_vol_bs(setup):
    grid, ctx, phi = setup
    dt = 1e-4
    phi_t = (bs_truth(grid, SIG, grid.t_out + dt) - bs_truth(grid, SIG, grid.t_out - dt)) / (2 * dt)
    ident = M.identifiable_mask(phi, ctx)
    assert ident.any() and not (ident & ~ctx.interior).any()
    sig = np.full(phi.shape, SIG)
    out = M.inverse_vol_metrics(phi, phi_t, ctx, ident, sig)
    sub = out["1e-03"]
    assert set(out) >= {"1e-02", "1e-03", "1e-04", "invvol_coverage", "invvol_sigma_relerr_median"}
    assert sub["coverage"] > 0.8
    assert sub["sigma_relerr_median"] < 2e-2
    assert out["invvol_coverage"] == sub["coverage"]
    test_inverse_vol_bs.numbers = (sub["coverage"], sub["sigma_relerr_median"])


def test_fd4_polynomial():
    n, dk = 40, 0.03
    k = np.arange(n) * dk
    D = M.d2_fd4_matrix(n, dk)
    p = 1.0 + 2 * k - 3 * k ** 2 + 0.7 * k ** 3
    d2 = -6.0 + 4.2 * k
    res = D @ p - d2
    # rows 1..n-2 exact to roundoff (fourth order exact for cubics); endpoint repo rows also exact for cubics? no: check 1..n-2
    r = np.abs(res[1:-1]).max()
    assert r < 1e-8
    test_fd4_polynomial.r = r
    # one-sided rows exact for degree-5
    q = k ** 5
    assert np.abs((D @ q - 20 * k ** 3)[[1, n - 2]]).max() < 1e-7
    # density_rows('fd4') agrees with d2_repo in the bulk for smooth BS
    grid = op.build_grid(n_k=128)
    ctx = M.make_context(grid)
    phi = bs_truth(grid, SIG, grid.t_out)
    f4 = M.density_rows(phi, ctx, "fd4")
    f2 = M.density_rows(phi, ctx, "d2_repo")
    assert np.abs(f4 - f2)[ctx.interior].max() < 0.05 * np.abs(f2).max()


def test_spectral_vs_lin(setup):
    grid, ctx, _ = setup
    n = grid.n_k
    j = 3
    i = np.arange(n)
    v = np.sin(np.pi * j * i / (n - 1))
    lam = -((np.pi * j / (n - 1)) / grid.dk) ** 2
    c = np.ones((len(grid.T_out), 1))
    f_spec = M.density_rows_spectral(c, v[:, None], np.array([lam]), ctx)
    f_lin = M.density_rows(np.tile(v, (len(grid.T_out), 1)), ctx, "d2_lin")
    sl = slice(1, -1)
    rel = np.abs(f_spec - f_lin)[:, sl].max() / np.abs(f_spec).max()
    assert rel < 1e-2


def test_resolved_and_context(setup):
    grid, ctx, _ = setup
    n, m, nt = grid.n_k, 5, len(grid.T_out)
    V = np.random.default_rng(0).standard_normal((n, m))
    c = np.random.default_rng(1).standard_normal((nt, m))
    out = M.resolved_error(c, c, V, ctx.interior)
    assert out == {"resolved_interior_rms": 0.0, "resolved_full_rms": 0.0, "resolved_coeff_rms": 0.0}
    assert M.resolved_error(c + 1.0, c, V, ctx.interior)["resolved_coeff_rms"] == pytest.approx(1.0)
    T = np.array([0.01, 0.05, 0.1])
    c2 = M.make_context(grid, T)
    assert c2.interior.shape == (3, n) and c2.K_rows.shape == (3, n) and np.allclose(c2.T_rows, T)
    assert np.allclose(c2.K_rows[1], grid.K_row(0.05))
    phi = bs_truth(grid, SIG, T / grid.T_max)
    assert M.density_rows(phi, c2).shape == (3, n)
    assert np.isnan(M._rms(np.ones((2, 2)), np.zeros((2, 2), bool)))


def test_rhs_helpers(setup):
    grid, ctx, phi = setup
    L = np.eye(grid.n_k) * 2.0
    assert np.allclose(M.rhs_time_derivative_full(phi, L), 2 * phi)
    assert np.allclose(M.rhs_time_derivative_full(phi, [L] * len(phi)), 2 * phi)
    m = 3
    V = np.random.default_rng(0).standard_normal((grid.n_k, m))
    c = np.random.default_rng(1).standard_normal((4, m))
    A = np.random.default_rng(2).standard_normal((m, m))
    Mh = np.random.default_rng(3).standard_normal((4, m))
    a = M.rhs_time_derivative_reduced(c, Mh, A, V)
    b = M.rhs_time_derivative_reduced(c, Mh, [A] * 4, V)
    assert np.allclose(a, b) and np.allclose(a, (c @ A.T + Mh) @ V.T)


def test_time_callable_and_env():
    with cost.pin_threads():
        import os
        assert os.environ["OMP_NUM_THREADS"] == "1"
        r = cost.time_callable(lambda: sum(range(1000)))
    assert len(r["samples"]) == 7 and r["min"] <= r["median"] <= r["max"] and r["cold"] > 0
    env = cost.timing_env()
    assert {"platform", "machine", "cpu_count", "python", "numpy", "scipy", "blas", "threads_env"} <= set(env)
