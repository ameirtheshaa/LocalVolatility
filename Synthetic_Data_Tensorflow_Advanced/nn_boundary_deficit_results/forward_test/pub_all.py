#!/usr/bin/env python3
r"""Regenerate every forward-density deck figure and its caption macros, in one call.

Two-stage pipeline (see pubstyle.py's module docstring): pub_compute.py does the one
TensorFlow-and-Monte-Carlo pass and caches its results in pub_data.npz / pub_stats.json;
everything below reads only that cache, so a restyle -- the common case -- costs
seconds, not the ~76s CPU of a full recompute.

Usage:
    python3 pub_all.py              # restyle only: 5 renders + captions (default, fast)
    python3 pub_all.py --recompute  # also re-run pub_compute.py first (slow, TF)

This exists because the deck's own header comment used to enumerate the render scripts
by hand and had drifted (it omitted pub_quotes.py) -- see journal_forward_density.tex's
top-of-file comment, which now points here instead of re-listing them.

--recompute is NOT reproducible from this repo's tracked contents alone. pub_compute.py
itself imports three more untracked siblings in this directory (d2surface.py,
forward_test.py, realised.py) and reads model checkpoints + CSV quotes under
dax_expk/ and forward_test/data/ -- all of it deliberately local-only, same as the
pub_data.npz/pub_stats.json cache it produces (this repo's .gitignore excludes *.npz
generally). Only the default restyle path above is guaranteed to run from what's
tracked plus that cache. --recompute has not been exercised since the figures were
last restyled; treat it as documentation of intent, not a tested code path.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RENDER_SCRIPTS = [
    "pub_evolution.py",    # fig1_evolution
    "pub_quotes.py",       # fig2_quotes
    "pub_extraction.py",   # fig3_extraction, fig4_stepsize
    "pub_realised.py",     # fig5_realised
    "pub_consistency.py",  # fig6_selfconsistency, figA1_residual
]


def run(script):
    print(f"=== {script} ===")
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed (exit {result.returncode}) -- stopping")


def main():
    if "--recompute" in sys.argv:
        run("pub_compute.py")
    for script in RENDER_SCRIPTS:
        run(script)
    run("pub_captions.py")
    print("\nAll figures + fd_numbers.tex regenerated. Recompile the deck:")
    print("  cd ../../presentation && pdflatex journal_forward_density.tex (twice)")


if __name__ == "__main__":
    main()
