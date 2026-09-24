#!/usr/bin/env python3
"""DAX 9 Aug 2001 — PDF analysis via DupirePipeline."""

from dax_analysis_common import run_pipeline_analysis

if __name__ == '__main__':
    out = run_pipeline_analysis('9aug')
    print(f"Done. Results: {out}")
