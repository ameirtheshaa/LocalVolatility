# NEXT_ACTIONS

Last completed checkpoint: `gate2` (see `checkpoints/CHECKPOINT_LATEST.json`).

Exactly one safe next command (idempotent; from the repository root, Mac/Linux, no account needed):

```bash
python3 research/dupire_signed_memory/scripts/gate3_compare.py --resume
```

Pending case IDs: none
Failed case IDs: none

Re-running the command re-uses every result whose stored config hash matches and recomputes only
missing or mismatched cases; it never overwrites a committed earlier checkpoint file.
