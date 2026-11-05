# Closure test for small boundary-adapted sine models

Follow-up to `research/dupire_signed_memory` (Gates 0–5). Question: does a deployable closure let
a 9- or 17-mode boundary-adapted sine model match or improve the useful accuracy of the 33-mode
unclosed sine model at lower online cost without losing density validity?

Read in this order: `PLAN.md` (frozen plan, binding) → `docs/API_CONTRACT.md` (module contract)
→ `docs/MEMO.md` (decision memo, written last) → `results/tables/` (every number).

Everything here is isolated: the earlier study is imported read-only as `dsm`; its results are
never rerun into their own files (the Gate 5 regression goes to `results/gate5_reproduction/`);
production defaults are untouched.

## Layout

```
scc/        conventions (DST-I basis, lift, interior blocks), cases, identity/cache, reference solvers,
            targets, closures (RRR/QL/LM/LR), oracles, metrics, cost
scripts/    s0_verify … s7_report, run_all
tests/      pytest suite (conftest redirects SCC_OUTPUT_ROOT to a temp dir)
results/    machine-readable results; cache/ is regenerable and ignored
models/     locked fitted weights + LOCK.json
docs/       API contract, memo
```

## Run

From the repository root (jobs longer than a minute go to the GPU-laptop CPU lane, see PLAN.md §12):

```bash
python3 -m pytest research/sine_closure/tests -q -p no:cacheprovider
python3 research/sine_closure/scripts/run_all.py --resume
```

`SCC_OUTPUT_ROOT` redirects outputs, `SCC_CACHE_ROOT` the reference cache, `SCC_PHASE=fit` makes
the cache refuse test references (leakage guard).
