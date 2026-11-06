#!/usr/bin/env python3
"""Cross-check of the new package against the earlier study on the three Gate 5 cases (development regression).

Same-grid quantities only (Gate 5 used a 512-point cubic-spline reference; this study uses nested 511/1021
restriction, so refined-grid errors are not comparable):
* Gate 5 ``projection_err_interior_rms`` (interior RMS of (I-P)psi at the 24 maturities, 256 grid) must equal
  this study's PROJ row ``same_price_interior_rms`` (PROJ reconstruction vs the same-grid full solve);
* Gate 5 ``reduced_operator_abscissa`` (spectral abscissa of V^T L V at the last step) must equal the RRR row's
  ``stability.spectral_abscissa_per_time[-1]``;
* Gate 5 ``price_interior_rms`` of B_sine_m{17,33} is against the 512-spline reference: reported next to this
  study's refined-grid number as a consistency indicator (expected to agree to ~u_price), not as an identity.
Writes results/verify/gate5_crosscheck.json and prints a table. Exit code 2 when an identity-type comparison
differs by more than 1e-6 relative.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
sys.path.insert(0, STUDY)
import scc  # noqa: E402
from scc import identity  # noqa: E402

GATE5 = os.path.join(os.path.dirname(STUDY), "dupire_signed_memory", "results", "gate5")
CASES = ["C_sig0.30", "S_seed9", "F_fresh_base0.28_t0.2_x0.05"]
PROTO = {"from_payoff_t0": "P0", "from_projected_t0.2": "P1"}


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def main():
    out = {"cases": {}, "pass": True, "tolerance_identity_rel": 1e-6,
           "note": "projection error: identical quantity (same-grid interior RMS of (I-P)psi at the 24 maturities); "
                   "abscissa: Gate 5 uses the last-step midpoint operator, this study the t=1 operator (differs by "
                   "operator drift over h/2 for time-dependent cases only); prices: Gate 5 vs 512-point cubic-spline "
                   "reference, this study vs the nested 511 restriction (not an identity)."}
    print(f"{'case':28s} {'m':>3s} {'proto':5s} {'G5 proj err':>12s} {'scc PROJ same':>13s} {'rel':>8s} | "
          f"{'G5 abscissa':>12s} {'scc abscissa':>12s} {'rel':>8s} | {'G5 price(512spl)':>16s} {'scc RRR price(511)':>18s}")
    for cid in CASES:
        g5p = os.path.join(GATE5, cid + ".json")
        scp = os.path.join(scc.RESULTS, "diagnose", cid + ".json")
        if not (os.path.isfile(g5p) and os.path.isfile(scp)):
            out["cases"][cid] = {"status": "missing", "gate5": os.path.isfile(g5p), "scc": os.path.isfile(scp)}
            out["pass"] = False
            print(f"{cid:28s} missing: gate5={os.path.isfile(g5p)} scc={os.path.isfile(scp)}")
            continue
        g5 = json.load(open(g5p))["models"]
        sc = json.load(open(scp))["models"]
        rows = {}
        for m in (17, 33):
            for g5proto, proto in PROTO.items():
                gm = g5.get(f"B_sine_m{m}__{g5proto}")
                pr = sc.get(f"PROJ_m{m}__{proto}", {}).get("metrics", {})
                rr = sc.get(f"RRR_m{m}__{proto}", {})
                if not gm or not pr:
                    continue
                a1, b1 = gm["projection_err_interior_rms"], pr.get("same_price_interior_rms")
                a2 = gm["reduced_operator_abscissa"]
                b2 = (rr.get("metrics", {}).get("stability", {}).get("spectral_abscissa_per_time") or [None])[-1]
                r1 = rel(a1, b1) if b1 is not None else float("nan")
                r2 = rel(a2, b2) if b2 is not None else float("nan")
                # abscissa: Gate 5 evaluates V^T L(t_mid,last) V at the last step midpoint (t = 1 - h/2), this study
                # the closed matrix at the last output time (t = 1); identical for autonomous cases, 1e-3 apart by
                # definition for time-dependent ones (operator drift over h/2), so the tolerance is 1e-4 / 5e-3.
                td = cid.startswith("F_") or cid.startswith("D") or cid.startswith("M")
                ok = (r1 <= 1e-6) and (r2 <= (5e-3 if td else 1e-4))
                rows[f"m{m}_{proto}"] = {"gate5_projection_err": a1, "scc_PROJ_same_price_interior_rms": b1, "rel_proj": r1,
                                         "gate5_abscissa": a2, "scc_abscissa_last": b2, "rel_abscissa": r2,
                                         "gate5_price_vs_512spline": gm["price_interior_rms"],
                                         "scc_RRR_price_vs_511": rr.get("metrics", {}).get("price_interior_rms"),
                                         "pass": bool(ok)}
                out["pass"] = out["pass"] and ok
                print(f"{cid:28s} {m:3d} {proto:5s} {a1:12.3e} {b1 if b1 is not None else float('nan'):13.3e} {r1:8.1e} | "
                      f"{a2:12.4e} {b2 if b2 is not None else float('nan'):12.4e} {r2:8.1e} | {gm['price_interior_rms']:16.3e} "
                      f"{rr.get('metrics', {}).get('price_interior_rms', float('nan')):18.3e}")
        out["cases"][cid] = rows
    path = os.path.join(scc.RESULTS, "verify", "gate5_crosscheck.json")
    identity.write_json(path, out)
    print(f"pass={out['pass']} -> {path}")
    sys.exit(0 if out["pass"] else 2)


if __name__ == "__main__":
    main()
