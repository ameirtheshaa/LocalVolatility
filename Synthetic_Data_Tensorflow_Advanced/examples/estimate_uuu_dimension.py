#!/usr/bin/env python3
"""
Estimate effective UUU subspace dimension (PCA + optional FNN autoencoder).

  cd Synthetic_Data_Tensorflow_Advanced
  MPLBACKEND=Agg python examples/estimate_uuu_dimension.py --case paper --fast
  MPLBACKEND=Agg python examples/estimate_uuu_dimension.py --case both --uuu-pca-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mz_spectral.mz_synthetic_setup import setup_mz_synthetic_case  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="UUU dimension: PCA + FNN on φ_U")
    p.add_argument("--case", choices=["constant", "paper", "both"], default="paper")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--n-t", type=int, default=24)
    p.add_argument("--n-k", type=int, default=1024)
    p.add_argument("--chosen-keep", type=int, default=256)
    p.add_argument("--out-root", type=str, default=os.path.join(_ROOT, "models", "runs"))
    p.add_argument("--uuu-fnn-max-d", type=int, default=128)
    p.add_argument("--uuu-fnn-epochs", type=int, default=120)
    p.add_argument("--uuu-pca-only", action="store_true")
    args = p.parse_args()

    cases = ["constant", "paper"] if args.case == "both" else [args.case]
    for case in cases:
        print("=" * 72)
        print(f"Case: {case}")
        st = setup_mz_synthetic_case(
            case,
            fast=args.fast,
            n_t=args.n_t,
            n_k=args.n_k,
            out_root=args.out_root,
            chosen_keep=args.chosen_keep,
            skip_truncation_sweep=True,
            estimate_uuu_dim=True,
            uuu_fnn_max_d=args.uuu_fnn_max_d,
            uuu_fnn_epochs=args.uuu_fnn_epochs,
            uuu_pca_only=args.uuu_pca_only,
        )
        rep = st["uuu_dimension"]
        print(json.dumps(rep, indent=2))
        print(f"Saved: {st['uuu_dimension_json']}")
    print("Done.")


if __name__ == "__main__":
    main()
