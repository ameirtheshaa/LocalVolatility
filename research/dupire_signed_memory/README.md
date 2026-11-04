# Signed memory in the Dupire reduced model — bounded research study

Question: does the Fourier-truncated Dupire reduced model in
`Synthetic_Data_Tensorflow_Advanced/mz_spectral` need signed, stateful feedback from the
eliminated modes, or is the nonnegative scalar eddy-diffusion restriction of `fit_nu_ql`
justified on variable-volatility problems?

Read in this order: `SOURCE_INVENTORY.md` → `FROZEN_PLAN.md` → `docs/DERIVATION.md` →
`REPORT.md` (conclusion, limitations) → `results/gate4/TABLES.md` (every number).

Everything here is isolated from production code: the study imports the committed
`Synthetic_Data_Tensorflow_Advanced` modules read-only and never changes their defaults.

## Local setup (Mac/Linux, no account needed)

```bash
git clone https://github.com/ameirtheshaa/LocalVolatility.git
cd LocalVolatility
git checkout claude/brave-brahmagupta-twxtst
python3 -m venv .venv-dsm && . .venv-dsm/bin/activate
pip install -r research/dupire_signed_memory/requirements.txt
python3 -m pytest research/dupire_signed_memory/tests -q -p no:cacheprovider
```

Python ≥ 3.11 with NumPy/SciPy is enough (pins in `requirements.txt` are the verified versions).
No TensorFlow, GPU, network access or data download is needed; all reference problems are
generated from committed code.

## Reproduce / resume

All gates are idempotent and resumable; run from the repository root:

```bash
python3 research/dupire_signed_memory/scripts/run_all.py --resume
```

Individual gates (same order):

```bash
python3 research/dupire_signed_memory/scripts/gate1_contract.py
python3 research/dupire_signed_memory/scripts/gate2_reference.py --resume
python3 research/dupire_signed_memory/scripts/gate3_compare.py --resume
python3 research/dupire_signed_memory/scripts/gate4_evaluate.py
```

* `NEXT_ACTIONS.md` always names the one exact next command after an interruption.
* `checkpoints/CHECKPOINT_<gate>.json` records the Git commit, source hashes, environment,
  seeds, done/failed/pending case IDs, the command, its exit code and result checksums.
* Per-case results live in `results/gate2/*.json` and per-block results in
  `results/gate3/blocks/*.json`; re-running recomputes only missing or mismatched entries.
  `--max-cases N` / `--max-blocks N` deliberately stop early (interruption rehearsal).
* Reference solutions are cached in `results/cache/` (regenerable, not committed).
* Surrogate weights are in `fixtures/*.npz` with sha256 hashes in the Gate 3 block files.
* `DSM_OUTPUT_ROOT=/some/dir` redirects all outputs (used by the tests and the rehearsal).

## Independent checks that can falsify the implementation

```bash
python3 -m pytest research/dupire_signed_memory/tests -q -p no:cacheprovider
python3 research/dupire_signed_memory/scripts/verify_results.py \
    research/dupire_signed_memory/results/gate1/gate1_contract.json <fresh gate1 json>
bash research/dupire_signed_memory/scripts/restore_rehearsal.sh
```

The tests cover the analytic control, the projector identity with the production code, the
exact-reconstruction identity (unresolved initial data + memory), the oracle floor of the
integrator, the eigenbasis no-memory property and interruption/resume idempotence. The
rehearsal clones the repository into a fresh directory, installs the pinned dependencies,
runs the tests, recomputes Gate 1 and compares it numerically with the committed result.

## Layout

```
dsm/            library (operators, cases, reference solver, MZ pieces, closures, metrics, checkpoints)
scripts/        gate1_contract.py, gate2_reference.py, gate3_compare.py, gate4_evaluate.py,
                run_all.py, verify_results.py, restore_rehearsal.sh
tests/          pytest suite
results/        machine-readable results (gate1..gate4), cache/ ignored
checkpoints/    durable checkpoints
fixtures/       surrogate weights (compact, hashed)
docs/           DERIVATION.md
```
