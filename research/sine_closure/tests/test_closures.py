import numpy as np
import pytest
from scipy.linalg import expm

import scc  # noqa: F401
from scc import conventions as cv
from scc import oracles as orc
from scc import targets as tgt
from scc import closures as cl
from dsm import operators as op
from dsm import reference as ref
from dsm.cases import get_case

N_K, SUB, M = 64, 4, 8
PRINT = {}


def make(case_id, m, sub=SUB):
    grid = op.build_grid(n_k=N_K)
    case = get_case(case_id)
    fg = ref.fine_grid(grid, sub=sub)
    phi = ref.solve_full(case, grid, "B", fg)
    r = {"phi": phi, "t": fg.t, "h": fg.h, "i0": fg.i0, "out_idx": fg.out_idx}
    T = tgt.build_targets(case, r, grid, m)
    return grid, case, fg, T


@pytest.fixture(scope="module")
def S():
    grid, case, fg, T = make("C_sig0.30", M)
    A = cv.A_structured(T["sys"], 0.0, T["V_int"], T["lam_V"])
    w = tgt.trajectory_weights({"C": T["rms_M"]}, {"C": len(fg.t)})
    return dict(grid=grid, case=case, fg=fg, T=T, A=A, A_mid=lambda n: A, w=w, lam=T["lam_V"])


def rrr(m=M):
    return cl.ModelSpec("RRR", m, "RRR")


def test_rrr_rollout_exact(S):
    T, fg, A = S["T"], S["fg"], S["A"]
    r = cl.rollout(rrr(), S["A_mid"], T["t"], T["c"][0], S["lam"], time_dependent=False)
    E = expm(fg.h * A)
    c = np.empty_like(r["c"])
    c[0] = T["c"][0]
    for n in range(len(fg.t) - 1):
        c[n + 1] = E @ c[n]
    assert np.array_equal(r["c"], c)
    assert r["n_expm"] == 1 and not r["unstable"] and r["h"].shape == (len(fg.t), 0)
    # earlier-study style: dense Galerkin matrix V^T L V
    Ad = T["V_int"].T @ T["sys"].L_int(0.0) @ T["V_int"]
    assert np.max(np.abs(Ad - A)) <= 1e-9 * np.max(np.abs(A))
    cd = np.empty_like(c)
    cd[0] = T["c"][0]
    Ed = expm(fg.h * Ad)
    for n in range(len(fg.t) - 1):
        cd[n + 1] = Ed @ cd[n]
    assert np.max(np.abs(cd - c)) <= 1e-9 * np.max(np.abs(c))


def test_teacher_forced_beats_rrr(S):
    T, fg = S["T"], S["fg"]
    tf = orc.teacher_forced_rollout(S["A_mid"], T["M"], T["t"], T["c"][0], False)
    rr = cl.rollout(rrr(), S["A_mid"], T["t"], T["c"][0], S["lam"], time_dependent=False)["c"]
    scale = np.max(np.linalg.norm(T["c"], axis=1))
    oi = fg.out_idx
    e_tf = np.max(np.linalg.norm(tf[oi] - T["c"][oi], axis=1)) / scale
    e_rr = np.max(np.linalg.norm(rr[oi] - T["c"][oi], axis=1)) / scale
    PRINT["tf"], PRINT["rrr"] = e_tf, e_rr
    print(f"\nTF err {e_tf:.3e}  RRR err {e_rr:.3e}")
    # trapezoid-in-M integrator floor is O(h^2): ~1e-6 at sub=4 (see convergence test below)
    assert e_tf < 1e-5
    assert e_tf < 1e-2 * e_rr


def test_teacher_forced_second_order():
    errs = []
    for sub in (4, 8):
        grid, case, fg, T = make("C_sig0.30", M, sub=sub)
        A = cv.A_structured(T["sys"], 0.0, T["V_int"], T["lam_V"])
        tf = orc.teacher_forced_rollout(lambda n: A, T["M"], T["t"], T["c"][0], False)
        scale = np.max(np.linalg.norm(T["c"], axis=1))
        errs.append(np.max(np.linalg.norm(tf[fg.out_idx] - T["c"][fg.out_idx], axis=1)) / scale)
    print("\nTF err sub4/sub8:", errs)
    assert errs[1] < errs[0] / 3.0


def test_filtered_states_exact():
    t = np.linspace(0.0, 1.0, 41)
    al = np.array([0.5, 5.0, 80.0, 0.05])
    c1 = np.ones((41, 3)) * np.array([1.0, -2.0, 0.5])
    q = cl.filtered_states(c1, t, al)
    assert q.shape == (41, 4, 3)
    assert np.all(q[0] == 0)
    exp = c1[:, None, :] * ((1 - np.exp(-al[None, :] * t[:, None])) / al[None, :])[:, :, None]
    assert np.max(np.abs(q - exp)) <= 1e-12 * max(1.0, np.max(np.abs(exp)))
    c2 = np.repeat(t[:, None], 3, axis=1)
    q = cl.filtered_states(c2, t, al)
    a = al[None, :]
    exp = (t[:, None] / a - (1 - np.exp(-a * t[:, None])) / a ** 2)[:, :, None] * np.ones((1, 1, 3))
    assert np.max(np.abs(q - exp)) <= 1e-12 * max(1.0, np.max(np.abs(exp)))
    # non-uniform steps
    tn = np.cumsum(np.r_[0.0, np.random.default_rng(1).uniform(0.01, 0.05, 30)])
    q = cl.filtered_states(np.repeat(tn[:, None], 2, axis=1), tn, al)
    exp = (tn[:, None] / a - (1 - np.exp(-a * tn[:, None])) / a ** 2)[:, :, None] * np.ones((1, 1, 2))
    assert np.max(np.abs(q - exp)) <= 1e-12 * max(1.0, np.max(np.abs(exp)))


def test_ridge_rel_and_fit_LM(S):
    T, w = S["T"], S["w"]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    Wt = rng.standard_normal((3, 4))
    W = cl.ridge_rel(X, X @ Wt.T, np.ones(50), 1e-12)
    assert np.max(np.abs(W - Wt)) < 1e-8
    H, b, info = cl.fit_LM([T], w, 1e-8)
    assert H.shape == (M, M) and b is None
    ws = list(w.values())[0]
    res0 = ws * np.sum(T["M"] ** 2)
    res1 = ws * np.sum((T["M"] - T["c"] @ H.T) ** 2)
    PRINT["lm_ratio"] = res1 / res0
    print(f"\nLM in-sample residual ratio {res1/res0:.3e}")
    assert res1 < 0.5 * res0
    assert info["n_rows"] == len(T["c"]) and info["weighted_resid_sq"] < info["weighted_M_sq"]
    H2, b2, _ = cl.fit_LM([T], w, 1e-8, intercept=True)
    assert b2.shape == (M,) and H2.shape == (M, M)


def test_fit_nu_and_bound(S):
    T, w = S["T"], S["w"]
    nu_adm = cl.admissible_bound([T["sys"]], T["V_int"], [0.2, 1.0])
    assert nu_adm < 0
    r = cl.fit_nu([T], w, nu_adm)
    print("\nnu:", r)
    assert all(np.isfinite(r[k]) for k in ("nu_unclipped", "num", "den", "nu_plus", "nu_pm"))
    assert r["nu_pm"] > nu_adm and r["nu_plus"] >= 0
    # clip branches
    assert cl.fit_nu([T], w, 10.0)["nu_pm"] == pytest.approx(5.0) or r["nu_unclipped"] > 5.0


def test_LR(S):
    T, w, A = S["T"], S["w"], S["A"]
    r1 = abs(float(cv.sine_eigenvalues(N_K, S["grid"].dk, M + 1))) * 0.1
    al = cl.rate_bank(4, "N", r1, 3.0)
    assert al.shape == (4,) and np.all(np.diff(al) > 0)
    assert al[0] == pytest.approx(0.5 * r1) and al[-1] == pytest.approx(16 * r1 * 3.0)
    assert cl.rate_bank(4, "W", r1, 3.0)[0] == pytest.approx(0.25 * r1)
    s0, s, info = cl.fit_LR([T], w, 4, al, 1e-8, als_iters=5)
    J = info["objective"]
    print("\nLR objective:", J)
    PRINT["J"] = J
    assert all(J[i + 1] <= J[i] * (1 + 1e-12) for i in range(len(J) - 1))
    for sp in (s0, s):
        assert sp.params["H"].shape == (M, M) and sp.params["R"].shape == (M, 4)
        assert sp.params["G"].shape == (4, M) and sp.params["F"].shape == (4, 4)
        assert sp.total_state() == M + 4
    assert s0.meta["G_learned"] is False and s.meta["G_learned"] is True
    assert s0.param_count() == {"fitted": M * M + M * 4, "fixed": 4 + 4 * M}
    assert s.param_count() == {"fitted": M * M + 2 * M * 4, "fixed": 4}
    rr = cl.rollout(s, S["A_mid"], T["t"], T["c"][0], S["lam"], time_dependent=False)
    assert not rr["unstable"] and np.all(np.isfinite(rr["c"])) and rr["h"].shape == (len(T["t"]), 4)
    rep = cl.stability_report(s, [A], S["lam"])
    assert rep["spectral_abscissa_max"] < 1e-6 * abs(float(np.min(S["lam"])))
    err = np.linalg.norm(rr["c"] - T["c"], axis=1).max() / np.linalg.norm(T["c"], axis=1).max()
    print("LR free-rollout err", err)
    assert np.isfinite(err)
    # initialiser and hidden init
    K = cl.fit_initializer(s, [T], w, S["fg"].i0, 1e-8)
    assert K.shape == (4, M + 1)
    s.params["K_init"] = K
    assert s.param_count()["fitted"] == M * M + 2 * M * 4 + 4 * (M + 1)
    c0 = T["c"][S["fg"].i0]
    assert np.allclose(cl.hidden_init_state(s, "WI", c0), K @ np.r_[c0, 1.0])
    assert np.all(cl.hidden_init_state(s, "H0", c0) == 0) and np.all(cl.hidden_init_state(s, "W0", c0) == 0)
    with pytest.raises(ValueError):
        cl.hidden_init_state(s, "WU", c0)
    r = cl.rollout(s, S["A_mid"], T["t"][S["fg"].i0:], c0, S["lam"], h0=cl.hidden_init_state(s, "WI", c0),
                   time_dependent=False)
    assert np.all(np.isfinite(r["c"]))


def test_closed_matrix_shapes_and_counts(S):
    A = S["A"]
    lam = S["lam"]
    rng = np.random.default_rng(0)
    assert np.array_equal(cl.closed_matrix(A, rrr(), lam), A)
    ql = cl.ModelSpec("QL", M, "QL", {"nu": 0.3})
    assert np.allclose(cl.closed_matrix(A, ql, lam), A + 0.3 * np.diag(lam))
    assert ql.param_count() == {"fitted": 1, "fixed": 0}
    H = rng.standard_normal((M, M))
    lm = cl.ModelSpec("LM", M, "LM", {"H": H, "b": None})
    assert cl.closed_matrix(A, lm, lam).shape == (M, M) and lm.total_state() == M
    assert lm.param_count() == {"fitted": M * M, "fixed": 0}
    b = rng.standard_normal(M)
    lmb = cl.ModelSpec("LM+b", M, "LM", {"H": H, "b": b})
    Mc = cl.closed_matrix(A, lmb, lam)
    assert Mc.shape == (M + 1, M + 1) and np.all(Mc[M] == 0) and np.allclose(Mc[:M, M], b)
    assert lmb.total_state() == M + 1 and lmb.param_count() == {"fitted": M * M + M, "fixed": 0}
    r = cl.rollout(lmb, S["A_mid"], S["T"]["t"][:20], S["T"]["c"][0], lam, time_dependent=False)
    assert r["c"].shape == (20, M) and np.all(np.isfinite(r["c"]))
    nh = 3
    lr = cl.ModelSpec("LR", M, "LR", {"F": -np.diag([1.0, 2, 3]), "G": rng.standard_normal((nh, M)),
                                      "H": H, "R": rng.standard_normal((M, nh)), "K_init": None}, n_h=nh)
    assert cl.closed_matrix(A, lr, lam).shape == (M + nh, M + nh)


def test_instability_flag(S):
    bad = cl.ModelSpec("QL", M, "QL", {"nu": -50.0})   # strongly anti-diffusive
    T = S["T"]
    r = cl.rollout(bad, S["A_mid"], T["t"], T["c"][0], S["lam"], time_dependent=False)
    assert r["unstable"] and np.isnan(r["c"][-1]).all() and 0 < r["n_last"] < len(T["t"]) - 1
    assert not np.isnan(r["c"][r["n_last"]]).all()


def test_stability_tools(S):
    T, A, lam = S["T"], S["A"], S["lam"]
    rep = cl.stability_report(rrr(), [A, A], lam)
    assert set(rep) == {"spectral_abscissa_max", "numerical_abscissa_max", "spectral_abscissa_per_time",
                        "numerical_abscissa_per_time"}
    assert rep["spectral_abscissa_max"] < 0
    assert len(rep["spectral_abscissa_per_time"]) == 2
    sp, nu = cl.abscissae(A)
    assert nu >= sp - 1e-9
    pg = cl.perturbation_growth(rrr(), S["A_mid"], T["t"], T["c"][0], lam, time_dependent=False)
    assert set(pg) == {"max_growth_ratio", "final_growth_ratio"} and np.isfinite(pg["max_growth_ratio"])
    p = cl.propagator_growth(rrr(), S["A_mid"], T["t"], lam, False)
    assert np.isfinite(p) and p >= 1.0
    lh = cl.long_horizon(rrr(), A, lam)
    assert lh["bounded"] and lh["max_norm_ratio"] <= 10


def test_time_dependent_smoke():
    grid, case, fg, T = make("D_base0.30", 6)
    sys_ = T["sys"]
    t = T["t"]

    def A_mid(n):
        return cv.A_structured(sys_, 0.5 * (t[n] + t[n + 1]), T["V_int"], T["lam_V"])
    r = cl.rollout(rrr(6), A_mid, t, T["c"][0], T["lam_V"], time_dependent=True)
    assert r["n_expm"] == len(t) - 1 and not r["unstable"] and np.all(np.isfinite(r["c"]))
    assert r["seconds"] > 0
    mid = [A_mid(n) for n in (0, 10)]
    assert np.isfinite(cl.propagator_growth(rrr(6), A_mid, t[:12], T["lam_V"], True))
    assert cl.stability_report(rrr(6), mid, T["lam_V"])["spectral_abscissa_max"] < 0
