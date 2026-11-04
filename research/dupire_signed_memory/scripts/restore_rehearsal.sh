#!/usr/bin/env bash
# Fresh-directory restore rehearsal: clone the repository into a new temporary directory, create a
# virtual environment with the pinned dependencies, run the tests, recompute Gate 1 and compare it
# numerically with the committed result, then rehearse an interrupted Gate 2 case and its resume.
# Usage: bash research/dupire_signed_memory/scripts/restore_rehearsal.sh [branch]
set -euo pipefail
BRANCH="${1:-claude/brave-brahmagupta-twxtst}"
SRC="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK="$(mktemp -d)"
echo "[rehearsal] source repo: $SRC  branch: $BRANCH  workdir: $WORK"
git clone -q --branch "$BRANCH" "$SRC" "$WORK/clone"
cd "$WORK/clone"
echo "[rehearsal] HEAD $(git rev-parse HEAD)"
python3 -m venv "$WORK/venv"
# shellcheck disable=SC1091
. "$WORK/venv/bin/activate"
pip install -q -r research/dupire_signed_memory/requirements.txt
python3 -m pytest research/dupire_signed_memory/tests -q -p no:cacheprovider
export DSM_OUTPUT_ROOT="$WORK/out"
python3 research/dupire_signed_memory/scripts/gate1_contract.py
python3 research/dupire_signed_memory/scripts/verify_results.py \
    research/dupire_signed_memory/results/gate1/gate1_contract.json "$WORK/out/results/gate1/gate1_contract.json"
# interrupted Gate 2 (one case computed, one left pending), then resume
python3 research/dupire_signed_memory/scripts/gate2_reference.py --only C_sig0.20,C_sig0.25 --variants B --no-refine --max-cases 1 --resume
test -f "$WORK/out/NEXT_ACTIONS.md" && grep -q "gate2_reference.py --resume" "$WORK/out/NEXT_ACTIONS.md"
python3 research/dupire_signed_memory/scripts/gate2_reference.py --only C_sig0.20,C_sig0.25 --variants B --no-refine --resume
python3 research/dupire_signed_memory/scripts/verify_results.py \
    research/dupire_signed_memory/results/gate2/C_sig0.20__B.json "$WORK/out/results/gate2/C_sig0.20__B.json"
echo "[rehearsal] OK — fresh checkout reproduces Gate 1 and a resumed Gate 2 case; outputs in $WORK/out"
