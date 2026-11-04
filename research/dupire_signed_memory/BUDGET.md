# Budget accounting

* Brief: approximately USD 43 of promotional Claude credits remaining, expiring 2026-11-05;
  reserve roughly USD 5 for integration/export.
* This cloud session has **no view of the account's eligible balance, usage or expiry**: no
  billing or usage tool is available in the session, and the agent cannot read the account
  page. Accounting is therefore recorded as **unavailable**; the figures above are the user's
  report, not a verification.
* Policy applied: work is organised so that every gate ends with a committed, pushed,
  resumable checkpoint; the final export (report, hashes, bundle) is prepared before any
  optional experiment; no purchases, overage or account changes are made; nothing is spent on
  repeating completed checks.
* Entries (one per useful block; wall-clock only, no token-price guesses):

| block | status | note |
|---|---|---|
| inventory + probes + frozen plan | done | ~20 min wall clock |
| Gate 1 (derivation, control, identities) | done | ~15 min wall clock; one φ₁-scaling bug found and fixed by the identity check itself |
| Gate 2 (36 reference cases + refinement) | done | 18 min first run + 4 min resume (15 production-stencil cases failed on a kernel-norm overflow, now reported as inf); ~55 min of idle wall clock lost to a background wait that expired |
| Gate 2b basis diagnostic | done | ~20 min wall clock under CPU contention |
| Gate 3 (6 closure blocks) | done | 445 s wall clock after the restart; first attempt discarded (progress-print bug overwrote finished blocks) |
| Gate 4 + report | done | ~15 min wall clock |
| Gate 5 release (restore rehearsal, final push) | done | see README.md / scripts/restore_rehearsal.sh |
| Wrap-up A: Gate 4 scoring correction, v1 preserved, S_seed8 disclosure | done | ~10 min wall clock |
| Wrap-up B: result identity / corruption / interruption checks, stamping, tests | done | ~20 min wall clock |
| Wrap-up C: boundary-adapted reduced model (Gate 5) | done | 726 s compute (fresh time-dependent case with 512-point reference dominates) |
| Wrap-up D: report, recommendation, rehearsal, export | done | |
