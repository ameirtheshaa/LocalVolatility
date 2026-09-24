# Tests

First automated test suite for the Dupire local-volatility pipeline. Smoke +
regression tests only -- fast enough to run on every change.

## Running

From the `Synthetic_Data_Tensorflow_Advanced/` directory:

```bash
python -m pytest tests -q
```

(`tests/conftest.py` puts `Synthetic_Data_Tensorflow_Advanced/` on `sys.path`,
so this also works run from the repo root as `python -m pytest
Synthetic_Data_Tensorflow_Advanced/tests -q`.)

The whole suite is designed to run in well under a minute on a Mac CPU: no
real training loops, no large Monte Carlo runs, only tiny (units=4,
1-residual-block) networks and small (n_t, n_k in the tens) synthetic grids.

## Slow tests

`pytest.ini` sets `addopts = -m "not slow"`, so anything marked
`@pytest.mark.slow` is skipped by default. Run those explicitly with:

```bash
python -m pytest tests -q -m slow
```

or run everything (slow included) with:

```bash
python -m pytest tests -q -m ""
```

## Files

- `conftest.py` -- puts the package root on `sys.path`.
- `test_pipeline_smoke.py` -- `config.py` defaults, the `exp_k` call-price
  ansatz's `C(K=0,T)=S0` boundary identity, the trainer's
  `TRAINABLE_ANSATZ` guard, `PDFAnalyzer.phi_mapping` validation and
  metadata-driven auto-resolution, and a Black-Scholes analytical sanity
  check (density integrates to 1, mean equals the forward).
- `test_mz_spectral_smoke.py` -- imports every `mz_spectral` submodule and
  runs a tiny constant-vol synthetic case end-to-end through the RRR /
  RRR+QL integrators, checking finite outputs and the documented `nu_QL >=
  0` invariant (see `docs/MZ_QL_NONNEG_FIX_REPORT.md`).
- `test_mz_spectral_regressions.py` -- regression tests for the `fit_nu_ql`
  ridge-double-counting fix and `data_driven.py`'s `vol_blend_alpha` fix
  (owned separately; see that file's docstring).
