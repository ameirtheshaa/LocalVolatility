# Closure test for small boundary-adapted sine models

Follow-up to `research/dupire_signed_memory` (Gates 0–5). Question: does a deployable closure let
a 9- or 17-mode boundary-adapted sine model match or improve the useful accuracy of the 33-mode
unclosed sine model at lower online cost without losing density validity?

Read in this order: `docs/MEMO.md` (decision memo: verdict, numbers, caveats) → `PLAN.md` (frozen
plan and its two addenda; binding) → `results/tables/decision.md` (every §10 rule with every exception)
→ `results/tables/per_case_models.md`, `cell_counts.md`, `summary_by_split.md` (every number) →
`docs/API_CONTRACT.md`, `docs/RESULT_SCHEMA.md` (code and file contracts).

**Outcome (2026-11-05).** No closure gives a supported benefit at 9 or 17 modes under the frozen rules;
the unclosed 33-mode sine model remains the recommendation, and adding sine modes beats closing at
equal total state. See `docs/MEMO.md`.

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

From the repository root (jobs longer than a minute go to the GPU-laptop CPU lane, see PLAN.md §12;
`scripts/_laptop_job.py <stage> [args]` is the launcher used there):

```bash
python3 -m pytest research/sine_closure/tests -q -p no:cacheprovider
python3 research/sine_closure/scripts/run_all.py --resume
```

Stages in order: `s0_verify` (identities, roundoff guard), `s1_reproduce_gate5` (separate output root),
`s2_references --split train val dev_regression`, `s3_diagnose`, `s4_fit` (SCC_PHASE=fit; writes
`models/LOCK.json`), `s2_references --split test_interp test_ood`, `s5_score`, `s6_cost` (run alone),
`s7_report`. Diagnostic extras: `x_gate5_crosscheck`, `x_ic_term`, `x_ql_pertraj`, `x_cell_counts`.
Everything is idempotent and identity-checked; `--quick` smoke modes write to `results/*_quick*`.

`SCC_OUTPUT_ROOT` redirects outputs, `SCC_CACHE_ROOT` the reference cache, `SCC_PHASE=fit` makes
the cache refuse test references (leakage guard).
