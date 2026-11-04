import numpy as np
import pytest

from dsm import cases, operators as op, repo


@pytest.mark.parametrize("n,keep", [(64, 5), (256, 32)])
def test_fourier_basis_matches_projector(n, keep):
    V = op.fourier_basis(n, keep)
    P = repo.low_pass_projector_matrix(n, keep)
    assert np.abs(V @ V.T - P).max() < 1e-12
    assert np.abs(V.T @ V - np.eye(2 * keep + 1)).max() < 1e-12


@pytest.mark.parametrize("case", cases.family_D(), ids=lambda c: c.id)
def test_family_D_sigma_matches_repo(case):
    cfg = repo.DupirePipelineConfig()
    vc = repo.VolatilityConfig.dupire_exact(sigma_base=case.params["sigma_base"])
    g, sigma = repo.build_uniform_tk_grid(S0=cfg.S0, r=cfg.r, T_max=cfg.T_max, K_max=cfg.K_max,
                                          K_min=cfg.K_min, T_min=cfg.T_min, n_t=24, n_k=64,
                                          vol_config=vc)
    sigma = np.asarray(sigma)
    T = np.asarray(g.T)
    K = np.asarray(g.K)
    assert sigma.shape == K.shape == (24, 64)
    for i in range(24):
        ref = case.sigma_fn(T[i], K[i, :] / cfg.S0)
        assert np.abs(sigma[i, :] - ref).max() < 1e-12


def test_d2_linearity_vs_repo():
    grid = op.build_grid(n_k=64)
    Dl = op.d2_linearity(64, grid.dk)
    Dr = op.d2_repo(64, grid.dk)
    assert not Dl[0].any() and not Dl[-1].any()
    assert np.array_equal(Dl[1:-1], Dr[1:-1])


def test_linear_trend_in_kernel_of_L():
    grid = op.build_grid(n_k=64)
    case = cases.get_case("C_sig0.30")
    L = op.build_L(op.eta_row(case.sigma_fn, grid, grid.T_max), grid, op.d2_linearity(64, grid.dk))
    out = L @ op.linear_trend(grid.k, 0.3, 0.0)
    assert np.abs(out).max() <= 1e-10 * np.abs(L).max()
