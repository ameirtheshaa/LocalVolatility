"""
Smoke / regression tests for config.py and dupire_pipeline.py.

Scope (see tests/README.md for the running convention):
  - DupirePipelineConfig defaults construct without error and default to the
    'exp_k' ansatz.
  - The exp_k call-price ansatz guarantees C(K=0,T) = S0 (phi_tilde(k=0) = 1)
    on a freshly-built, untrained, tiny-width model — this is the "K=0
    boundary deficit" fix documented in dupire_pipeline.py around
    neural_phi_tilde() / loss_phi_cal().
  - PDFAnalyzer.phi_mapping argument validation (4 valid values, ValueError
    on a bad one, and ValueError on an explicit value that disagrees with
    the model's metadata ansatz tag).
  - DupireNeuralModel._validate_training_ansatz() rejects the
    'one_minus_exp_1mk' ansatz (TRAINABLE_ANSATZ guard).
  - Metadata ansatz tag -> phi_mapping auto-resolution.
  - Black-Scholes analytical sanity: the lognormal density integrates to ~1
    and its mean equals the risk-neutral forward S0*exp(rT).

Everything here uses tiny networks (units=4, num_res_blocks=1) and/or pure
analytical formulas — no data generation, no training loops, no Monte Carlo.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from scipy import integrate

from config import DupirePipelineConfig
from dupire_pipeline import DataGenerator, DupireNeuralModel, PDFAnalyzer, data_type
from analytical_solutions import black_scholes_call, lognormal_density


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

def test_default_config_constructs_without_error():
    config = DupirePipelineConfig()
    assert config.output_dir is not None


def test_default_ansatz_is_exp_k():
    config = DupirePipelineConfig()
    assert config.ansatz == "exp_k"


def test_quick_test_preset_constructs():
    config = DupirePipelineConfig.quick_test()
    assert config.num_epochs == 100
    assert config.ansatz == "exp_k"


# ---------------------------------------------------------------------------
# exp_k ansatz: C(K=0, T) = S0  <=>  phi_tilde(k_tilde=0) = 1
# ---------------------------------------------------------------------------

def _tiny_model(config: DupirePipelineConfig) -> DupireNeuralModel:
    """Build a DupireNeuralModel with a tiny NN_phi_tilde, no data generation/training."""
    dg = DataGenerator(config)
    model = DupireNeuralModel(config, dg)
    model.NN_phi_tilde = model.net_phi_tilde(
        num_res_blocks=1, units=4, activation=config.activation
    )
    return model


def test_exp_k_ansatz_gives_unit_mass_at_k0():
    """Fresh (random-weight) exp_k model: phi_tilde(t, k=0) == 1 exactly for any t."""
    config = DupirePipelineConfig.quick_test()
    assert config.ansatz == "exp_k"
    model = _tiny_model(config)

    t_tilde = tf.constant([[0.0], [0.3], [0.7], [1.0]], dtype=data_type)
    k_tilde = tf.zeros_like(t_tilde)
    phi_tilde = model.neural_phi_tilde(t_tilde, k_tilde).numpy().ravel()

    np.testing.assert_allclose(phi_tilde, 1.0, atol=1e-5)


def test_exp_k_ansatz_call_price_at_k0_equals_s0():
    """neural_phi(T, K=0) == S0 for the exp_k ansatz (unit-mass boundary in original units)."""
    config = DupirePipelineConfig.quick_test()
    model = _tiny_model(config)

    T = tf.constant([0.5, 1.0], dtype=data_type)
    K = tf.zeros_like(T)
    C = model.neural_phi(T, K).numpy().ravel()

    np.testing.assert_allclose(C, config.S0, rtol=1e-5)


def test_one_minus_exp_ansatz_does_not_hardwire_k0_boundary():
    """Sanity check that the exp_k guarantee is ansatz-specific: 'one_minus_exp' has no
    such structural guarantee (phi_tilde(k=0) need not be 1 for random weights)."""
    config = DupirePipelineConfig.quick_test()
    config.ansatz = "one_minus_exp"
    model = _tiny_model(config)

    t_tilde = tf.constant([[0.5]], dtype=data_type)
    k_tilde = tf.zeros_like(t_tilde)
    phi_tilde = float(model.neural_phi_tilde(t_tilde, k_tilde).numpy().ravel()[0])

    # Not asserting a specific value -- only that it is a real, unconstrained
    # network output. If this ever becomes exactly 1.0 to machine precision it
    # would suggest an accidental structural coupling worth investigating.
    assert np.isfinite(phi_tilde)


# ---------------------------------------------------------------------------
# DupireNeuralModel.TRAINABLE_ANSATZ / _validate_training_ansatz
# ---------------------------------------------------------------------------

def test_trainable_ansatz_rejects_one_minus_exp_1mk():
    config = DupirePipelineConfig.quick_test()
    config.ansatz = "one_minus_exp_1mk"
    dg = DataGenerator(config)
    model = DupireNeuralModel(config, dg)

    with pytest.raises(ValueError, match="one_minus_exp_1mk"):
        model.net_phi_tilde(num_res_blocks=1, units=4, activation=config.activation)


@pytest.mark.parametrize("ansatz", ["one_minus_exp", "exp_k"])
def test_trainable_ansatz_accepts_known_values(ansatz):
    config = DupirePipelineConfig.quick_test()
    config.ansatz = ansatz
    dg = DataGenerator(config)
    model = DupireNeuralModel(config, dg)
    net = model.net_phi_tilde(num_res_blocks=1, units=4, activation=config.activation)
    assert net is not None


# ---------------------------------------------------------------------------
# PDFAnalyzer: phi_mapping validation + metadata ansatz resolution
# ---------------------------------------------------------------------------

def _dummy_analyzer_inputs(config: DupirePipelineConfig):
    dg = DataGenerator(config)
    model = DupireNeuralModel(config, dg)
    model.NN_phi_tilde = model.net_phi_tilde(num_res_blocks=1, units=4, activation=config.activation)
    model.NN_eta_tilde = model.net_eta_tilde(num_res_blocks=1, units=4, activation=config.activation)
    return model.NN_phi_tilde, model.NN_eta_tilde


def test_pdf_analyzer_rejects_bad_phi_mapping():
    config = DupirePipelineConfig.quick_test()
    nn_phi, nn_eta = _dummy_analyzer_inputs(config)
    metadata = {"ansatz": "exp_k"}

    with pytest.raises(ValueError, match="phi_mapping must be one of"):
        PDFAnalyzer(nn_phi, nn_eta, config, metadata, phi_mapping="bogus")


@pytest.mark.parametrize("phi_mapping", ["transformed", "legacy", "exp_k", "one_minus_exp_1mk"])
def test_pdf_analyzer_accepts_all_valid_phi_mappings(phi_mapping):
    config = DupirePipelineConfig.quick_test()
    nn_phi, nn_eta = _dummy_analyzer_inputs(config)
    # 'legacy' is always allowed; the others must not conflict with metadata,
    # so leave metadata's ansatz unset (None) to isolate the argument-validation path.
    metadata = {}
    analyzer = PDFAnalyzer(nn_phi, nn_eta, config, metadata, phi_mapping=phi_mapping)
    assert analyzer.phi_mapping == phi_mapping


def test_pdf_analyzer_explicit_phi_mapping_conflicting_with_metadata_raises():
    config = DupirePipelineConfig.quick_test()
    nn_phi, nn_eta = _dummy_analyzer_inputs(config)
    metadata = {"ansatz": "exp_k"}  # resolves to phi_mapping='exp_k'

    with pytest.raises(ValueError, match="disagrees with model metadata"):
        PDFAnalyzer(nn_phi, nn_eta, config, metadata, phi_mapping="transformed")


@pytest.mark.parametrize(
    "meta_ansatz, expected_mapping",
    [
        ("exp_k", "exp_k"),
        ("one_minus_exp", "transformed"),
    ],
)
def test_pdf_analyzer_metadata_ansatz_resolves_to_phi_mapping(meta_ansatz, expected_mapping):
    config = DupirePipelineConfig.quick_test()
    nn_phi, nn_eta = _dummy_analyzer_inputs(config)
    metadata = {"ansatz": meta_ansatz}

    analyzer = PDFAnalyzer(nn_phi, nn_eta, config, metadata)
    assert analyzer.phi_mapping == expected_mapping


def test_pdf_analyzer_unknown_metadata_ansatz_raises():
    config = DupirePipelineConfig.quick_test()
    nn_phi, nn_eta = _dummy_analyzer_inputs(config)
    metadata = {"ansatz": "totally_unknown_ansatz"}

    with pytest.raises(ValueError, match="has no price mapping"):
        PDFAnalyzer(nn_phi, nn_eta, config, metadata)


def test_pdf_analyzer_absent_metadata_ansatz_defaults_to_transformed():
    config = DupirePipelineConfig.quick_test()
    nn_phi, nn_eta = _dummy_analyzer_inputs(config)
    analyzer = PDFAnalyzer(nn_phi, nn_eta, config, {})
    assert analyzer.phi_mapping == "transformed"


# ---------------------------------------------------------------------------
# Analytical sanity: Black-Scholes constant-vol case
# ---------------------------------------------------------------------------

def test_bs_lognormal_density_integrates_to_one():
    S0, r, sigma, T = 1000.0, 0.04, 0.3, 1.0

    def density(K):
        return lognormal_density(K, S0, T, r, sigma)

    mass, _ = integrate.quad(density, 1e-6, S0 * 20, limit=200)
    assert mass == pytest.approx(1.0, abs=1e-4)


def test_bs_lognormal_density_mean_equals_forward():
    S0, r, sigma, T = 1000.0, 0.04, 0.3, 1.0
    forward = S0 * np.exp(r * T)

    def weighted(K):
        return K * lognormal_density(K, S0, T, r, sigma)

    mean, _ = integrate.quad(weighted, 1e-6, S0 * 20, limit=200)
    assert mean == pytest.approx(forward, rel=1e-4)


def test_bs_call_price_matches_discounted_density_expectation_grid():
    """Cross-check: Breeden-Litzenberger-consistent BS call price vs a coarse
    trapezoidal expectation of the discounted payoff under the same lognormal
    density (grid-based, not the closed form, to exercise both helpers)."""
    S0, r, sigma, T = 1000.0, 0.04, 0.3, 1.0
    K_strike = 1000.0

    K_grid = np.linspace(1.0, S0 * 10, 20000)
    density = lognormal_density(K_grid, S0, T, r, sigma)
    payoff = np.maximum(K_grid - K_strike, 0.0)
    price_numeric = np.exp(-r * T) * np.trapezoid(payoff * density, K_grid)

    price_closed_form = black_scholes_call(S0, K_strike, T, r, sigma)

    assert price_numeric == pytest.approx(price_closed_form, rel=5e-3)
