import os
import sys

import numpy as np
import pytest

from dsm import cases, closures, mz, operators as op, reference

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from gate1_contract import reconstruct_constant_L  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    case = cases.get_case("C_sig0.30")
    grid = op.build_grid(n_k=64)
    fg = reference.fine_grid(grid, sub=4)
    phi = reference.solve_full(case, grid, "B", fg)
    return case, grid, fg, phi


def test_exact_reconstruction(ctx):
    case, grid, fg, phi = ctx
    psi = reference.state_variable(phi, grid, "B")
    L = reference.operator_builder(case, grid, "B")(0.0)
    V = op.fourier_basis(64, 8)
    out, _ = reconstruct_constant_L(L, V, psi, fg.t, fg.i0)
    assert out["max_rel_err_unresolved_reconstruction"] < 1e-3
    assert out["max_rel_err_resolved_reconstruction"] < 1e-3
    assert out["finite"] is True


def test_oracle_vs_rrr_rollout(ctx):
    case, grid, fg, phi = ctx
    S = closures.build_setup(case, grid, fg, "B", 8, phi_fine=phi)

    def err(r):
        d = (r["c"][S.out_rel] - S.c_ref[S.out_rel]) @ S.V.T
        return float(np.sqrt(np.mean(d ** 2)))

    ro = closures.rollout(S, closures.Oracle())
    assert not ro["unstable"]
    e_or = err(ro)
    assert e_or < 1e-6
    rr = closures.rollout(S, closures.RRR())
    assert err(rr) > e_or


def test_weighted_eigenbasis_has_no_memory(ctx):
    case, grid, fg, phi = ctx
    L = reference.operator_builder(case, grid, "B")(0.0)
    eta = op.eta_row(case.sigma_fn, grid, grid.T_max)
    dvec = eta * grid.k ** 2
    L_int = L[1:-1, 1:-1]
    S = np.sqrt(1.0 / dvec[1:-1])
    M = S[:, None] * L_int / S[None, :]
    Lt = 0.5 * (M + M.T)
    ev, evec = np.linalg.eigh(Lt)
    Ve = evec[:, np.argsort(np.abs(ev))[:17]]
    B = mz.resolved_blocks(Lt, Ve)
    assert np.linalg.norm(B["L_RU"]) < 1e-8 * np.linalg.norm(B["L_RR"])
