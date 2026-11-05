"""Closure test for small boundary-adapted sine models (research/sine_closure).

Pure NumPy/SciPy. Imports the earlier study `research/dupire_signed_memory` (package `dsm`) read-only
for the grid, the operator conventions and the production code access; nothing there is modified.
See PLAN.md (frozen plan) and docs/API_CONTRACT.md (module contract).
"""
from __future__ import annotations

import os
import sys

__version__ = "0.1.0"

STUDY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # research/sine_closure
RESEARCH_ROOT = os.path.dirname(STUDY_ROOT)                                       # research/
REPO_ROOT = os.path.dirname(RESEARCH_ROOT)
DSM_ROOT = os.path.join(RESEARCH_ROOT, "dupire_signed_memory")
if DSM_ROOT not in sys.path:
    sys.path.insert(0, DSM_ROOT)

OUTPUT_ROOT = os.environ.get("SCC_OUTPUT_ROOT", STUDY_ROOT)
RESULTS = os.path.join(OUTPUT_ROOT, "results")
CACHE_ROOT = os.environ.get("SCC_CACHE_ROOT", os.path.join(RESULTS, "cache"))
MODELS = os.path.join(OUTPUT_ROOT, "models")

import dsm  # noqa: E402,F401  (fails loudly if the earlier study is missing)
