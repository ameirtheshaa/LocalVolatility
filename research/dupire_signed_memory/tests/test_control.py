from dsm import control


def test_control_check():
    r = control.check()
    assert r["max_resolved_equation_residual"] < 1e-6
    assert r["max_unresolved_reconstruction_error"] < 1e-6
    assert r["nu_star_on_minus_x"] < 0
    assert r["closed_model_stable"] is True
    assert r["fraction_time_work_positive"] == 1.0
    assert -2 < r["closed_model_rate"] < -1
