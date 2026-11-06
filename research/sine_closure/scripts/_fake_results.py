#!/usr/bin/env python3
"""DEV TOOL ONLY -- writes a SYNTHETIC, schema-shaped results tree to test scripts/s7_report.py.

The numbers are invented (deterministic, seed 0) and mean NOTHING scientifically. This script is never
part of the pipeline (not called by run_all.py) and refuses to write into the real study output unless
--root is given explicitly and differs from the study directory.

Usage:  python scripts/_fake_results.py --root /tmp/fake_scc [--scenario realistic|strong]
        SCC_OUTPUT_ROOT=/tmp/fake_scc python scripts/s7_report.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = [
    ("C_test_0.205", "C", "test_interp", {"sigma": 0.205}),
    ("S_seed120", "S", "test_interp", {"b": 0.31, "s": 0.1, "c": 0.05}),
    ("D_seed230", "D", "test_ood", {"base": 0.42, "t_s": 0.4, "x_s": 0.3}),
    ("C_sig0.30", "C", "dev_regression", {"sigma": 0.3}),
    ("S_seed9", "S", "dev_regression", {"b": 0.25, "s": 0.1, "c": 0.05}),
]
MS = (9, 13, 17, 25, 33, 65)
T_OUT = np.linspace(0.3, 1.5, 24)


def wj(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


def build(root, scenario):
    rng = np.random.default_rng(0)
    R = os.path.join(root, "results")
    strong = scenario == "strong"

    def price_rrr(m):
        return (3.5e-6 + (4.5e-5 if strong else 6.5e-6) * (9.0 / m) ** 2)

    def edyn_rrr(m):
        return 2e-6 + 4e-5 * (9.0 / m) ** 1.6

    def dens_rrr(m):
        return 1.2e-5 + 1.2e-4 * (9.0 / m) ** 1.3

    def neg_rrr(m):
        return 0.0 if m >= 33 else 6e-4 * (9.0 / m) ** 2.5

    def metrics(price, edyn, dens, neg, invvol=0.17, cov=0.97, butter=0.002, unstable=False, early=False, rng=rng, sigma=1.0):
        pm = price * (0.4 + 1.2 * np.linspace(0, 1, 24) ** 1.5)
        met = {
            "price_interior_rms": price, "price_full_rms": price * 1.4, "price_max_abs": price * 5, "price_atm_rms": price * 0.7,
            "price_wing_rms": price * 1.8, "price_per_maturity_rms": pm.tolist(), "price_physical_interior_rms": price * 1000,
            "same_price_interior_rms": price * 0.93,
            "resolved_interior_rms": edyn, "resolved_full_rms": edyn * 1.3, "resolved_coeff_rms": edyn * 2.0,
            "density_l2_interior_mean": dens, "density_l2_interior_max": dens * 1.8, "density_l1_interior_mean": dens * 0.8,
            "density_l2_full_mean": dens * 1.5, "density_max_abs_err": dens * 6, "density_q99_abs_err": dens * 3,
            "negative_mass": [neg * (0.1 + i / 23) for i in range(24)], "negative_mass_max_post_initial": neg, "negative_mass_final": neg * 0.8,
            "mass_final": 1.0, "mass_ref_final": 1.0, "first_moment_final": 1000.0, "first_moment_ref_final": 1000.0,
            "positivity_raw_interior_frac_mean": 0.0, "monotonicity_raw_interior_frac_mean": 0.0,
            "butterfly_raw_interior_frac_mean": butter, "calendar_raw_interior_frac_mean": 0.01 * (1 + butter * 50),
            "positivity_raw_interior_frac_max": 0.0, "butterfly_raw_interior_frac_max": butter * 2,
            "invvol_invalid_frac_full": invvol, "invvol_coverage": cov, "invvol_sigma_relerr_median": 1e-3 * price / 1e-5,
            "invvol_sigma_relerr_rms": 2e-3, "invvol_undefined_frac_interior": invvol * 0.8,
            "invvol_negative_or_nonfinite_frac_interior": invvol * 0.2,
            "stability": {"spectral_abscissa_max": -0.2 - 0.1 * rng.random(), "numerical_abscissa_max": 0.5, "max_growth_ratio": 1.1,
                          "final_growth_ratio": 0.9, "propagator_growth": 1.0, "long_horizon_bounded": True},
            "phi_t_check": {"rms_rhs_vs_fd16": 1e-8, "rms_fd16_vs_fd32": 2e-9},
        }
        met["invvol_panel"] = {}
        for tol, sc in (("1e-02", 1.2), ("1e-03", 1.0), ("1e-04", 0.8)):
            met["invvol_panel"][tol] = {"undefined_frac_interior": invvol * 0.8 * sc, "negative_or_nonfinite_frac_interior": invvol * 0.2,
                        "invalid_frac_full": invvol * sc, "coverage": min(1.0, cov / sc ** 0.1), "sigma_relerr_median": 1e-3,
                        "sigma_relerr_rms": 2e-3, "n_identifiable": 4000}
        if early:
            met["early_panel"] = {f"{t:.2f}": {"price_interior_rms": price * (1.5 - 4 * t), "density_l2_interior": dens * (1.4 - 3 * t),
                                              "negative_mass": neg * (1.2 - 2 * t), "signed_mass": 1.0, "t_actual": t}
                                  for t in (0.05, 0.10, 0.15)}
        if unstable:      # NaN-based metrics: fractions read as 0 (looks like a pass), must be treated as failures
            met["price_interior_rms"] = price * 1e3
            met["negative_mass_max_post_initial"] = 0.0
            met["butterfly_raw_interior_frac_mean"] = 0.0
            met["invvol_invalid_frac_full"] = 0.0
        return met

    def row(met, state, params, online, unstable=False, early=False):
        t = {"init": 1e-4 * state / 9, "assembly_and_expm": 2e-4 * state, "stepping": 1e-4, "reconstruction": 3e-4,
             "density": 2e-4, "inverse_vol": 2e-4}
        t = {k: v * online for k, v in t.items()}
        t["total"] = float(sum(t.values()))
        return {"metrics": met, "unstable": unstable, "n_last": 24 if not unstable else 11, "total_state": state,
                "param_count": params, "online_seconds": t, "n_expm": 1}

    for cid, fam, split, params in CASES:
        cf = 1.0 + 0.2 * rng.standard_normal()
        cf = float(np.clip(cf, 0.7, 1.4))
        models = {}
        u_price, u_dens, u_neg = 4e-6 * cf, 3e-6 * cf, 2e-8
        for m in MS:
            for P in ("P0", "P1"):
                pf = 1.0 if P == "P0" else 0.8
                models[f"RRR_m{m}__{P}"] = row(metrics(price_rrr(m) * cf * pf, edyn_rrr(m) * cf * pf, dens_rrr(m) * cf * pf, neg_rrr(m) * cf,
                                                       butter=0.004 * 9.0 / m, early=(P == "P0")), m, 0, 1.0 + m / 40.0)
                models[f"PROJ_m{m}__{P}"] = row(metrics(price_rrr(m) * cf * (0.2 if strong else 0.6), 1e-9, dens_rrr(m) * cf * 0.9, neg_rrr(m) * cf * 0.5), m, 0, 0.0)
                models[f"TF_m{m}__{P}"] = row(metrics(price_rrr(m) * cf * 0.7, 2e-7, dens_rrr(m) * cf * 0.95, neg_rrr(m) * cf * 0.7), m, 0, 0.0)
        for P in ("P0", "P1"):
            models[f"FULL256__{P}"] = row(metrics(4e-6 * cf, 0.0, 3e-6 * cf, 5e-7 * cf, early=(P == "P0")), 254, 0, 40.0)
        models["FULL511__P0"] = row(metrics(1e-6 * cf, 0.0, 1e-6 * cf, 1e-7 * cf), 509, 0, 150.0)
        clos_defs = []
        for m in (9, 17, 33):
            for c, pf_, ef_, df_, nf_ in (("QLp", 0.97, 0.95, 0.97, 1.0), ("QLpm", 0.96, 0.93, 0.96, 1.0), ("QLs", 0.95, 0.9, 0.95, 1.1),
                                          ("LM", 0.85, 0.7, 0.9, 1.2), ("LMb", 0.86, 0.72, 0.9, 1.2)):
                if m == 33 and c not in ("LM", "QLp"):
                    continue
                clos_defs.append((c, m, 0, pf_, ef_, df_, nf_))
        for m in (9, 17):
            for c, h, pf_, ef_, df_, nf_ in (("LR0", 8, 0.8, 0.6, 0.88, 1.1), ("LR", 8, 0.75, 0.5, 0.85, 1.1), ("LR", 16, 0.7, 0.4, 0.8, 0.9),
                                             ("LR0", 16, 0.78, 0.55, 0.86, 1.0)):
                clos_defs.append((c, m, h, pf_, ef_, df_, nf_))
        for (c, m, h, pf_, ef_, df_, nf_) in clos_defs:
            if strong and c == "LR" and h == 16 and m == 9:
                pf_, ef_, df_, nf_ = 0.12, 0.1, 0.5, 0.6
            if strong and c == "LR" and h == 8 and m == 17:
                pf_, ef_, df_, nf_ = 0.1, 0.05, 0.3, 0.3
            if strong and c == "LM" and m == 9:
                pf_, ef_ = 0.2, 0.15
            for P in ("P0", "P1", "P1_W0", "P1_WI", "P1_WU"):
                if h == 0 and P not in ("P0", "P1"):
                    continue
                if h and P == "P1":
                    continue
                base = "P1" if P.startswith("P1") else "P0"
                pf = 1.0 if base == "P0" else 0.8
                hid_adj = {"P1_WI": 0.97, "P1_WU": 0.95}.get(P, 1.0)
                unst = (c == "QLs" and m == 17 and cid == "S_seed120" and base == "P1")
                neg = neg_rrr(m) * cf * pf * nf_
                if c == "LM" and m == 17 and cid == "C_test_0.205":
                    neg *= 3.0     # an injected validity exception
                met = metrics(price_rrr(m) * cf * pf * pf_ * hid_adj, edyn_rrr(m) * cf * pf * ef_ * hid_adj, dens_rrr(m) * cf * pf * df_, neg,
                              cov=0.97 - (0.02 if c == "LMb" else 0.0), butter=0.004 * 9.0 / m, unstable=unst, early=(P == "P0"))
                name = f"{c}_m{m}" + (f"_h{h}" if h else "") + f"__{P}"
                models[name] = row(met, m + h + (1 if c == "LMb" else 0), (m * m if c in ("LM", "LMb") else (1 if c.startswith("QL") else m * m + 2 * m * h)),
                                   1.0 + (m + h) / 40.0 + (0.2 if h else 0.0), unstable=unst, early=(P == "P0"))
        models["QLs_m9__P1"] = {"metrics": None, "unstable": True, "failed": True, "traceback": "synthetic failure"}
        score = {"case": cid, "family": fam, "split": split, "params": params, "status": "done", "n_interior_per_row": 230,
                 "uncertainty": {}, "models": models}
        per_q = {"price_interior_rms": u_price, "density_l2_interior_mean": u_dens, "density_l1_interior_mean": u_dens * 0.8,
                 "negative_mass_max_post_initial": u_neg, "negative_mass_final": u_neg, "invvol_invalid_frac_full": 0.002,
                 "invvol_coverage": 0.002, "invvol_sigma_relerr_median": 1e-4}
        for k in ("positivity", "monotonicity", "butterfly", "calendar"):
            per_q[f"{k}_raw_interior_frac_mean"] = 0.001
        unc = {"per_quantity": {q: {"type": "value", "u_grid": v, "u_step": v * 0.3, "u": v} for q, v in per_q.items()},
               "u_dyn": {str(m): 1.5e-6 * cf for m in (9, 17, 33)}, "e_TF": {}, "gamma": 3.0e-9}
        score["uncertainty"] = unc
        wj(os.path.join(R, "score", f"{cid}.json"), score)
        wj(os.path.join(R, "uncertainty", f"{cid}.json"), {"case": cid, "status": "done", "stage": "uncertainty", "uncertainty": unc})
        if split in ("dev_regression",):
            dm, decomp = {}, {}
            for m in MS:
                for P in ("P0", "P1"):
                    for c in ("PROJ", "RRR", "TF"):
                        dm[f"{c}_m{m}__{P}"] = models[f"{c}_m{m}__{P}"]
                    decomp[f"RRR_m{m}__{P}"] = {"proj_interior_rms": price_rrr(m) * cf * 0.6, "dyn_interior_rms": edyn_rrr(m) * cf,
                                               "orthogonality_defect": 0.02, "reconstruction_residual": 1e-15}
            wj(os.path.join(R, "diagnose", f"{cid}.json"), {"case": cid, "split": split, "status": "done", "models": dm, "decomposition": decomp,
                                                            "u_dyn": unc["u_dyn"]})
    wj(os.path.join(R, "diagnose", "summary.json"), {"status": "done", "rule_per_m": {f"m{m}": {"n_cases": 2, "fraction_cond_A": 0.5, "fraction_cond_B": 0.5,
                                                                                               "representation_limited": False} for m in (9, 17, 33)}})
    wj(os.path.join(R, "verify", "verify.json"), {"V_orthonormal": {"value": 1e-15, "tolerance": 1e-12, "pass": True}})
    wj(os.path.join(R, "verify", "roundoff_guard.json"), {"gamma": 3.0e-9, "max_paired_diff": 3.0e-10, "factor": 10})
    # cost (s6 layout: models {case: {key: row}}, full_solvers {case: {...}}, break_even {case: {key: {...}}})
    def crow(med, iqr=1e-4, cold=None):
        return {"cold": cold or med * 1.3, "warm": {"median": med, "iqr": iqr, "min": med * 0.95, "max": med * 1.1}, "online_total_median": med,
                "components": {"stepping": {"median": med * 0.5, "iqr": iqr / 2}}, "total_state": 0}
    cm, fs, be = {}, {}, {}
    for cid in ("C_test_0.205", "S_seed120"):
        r = {}
        for m in MS:
            for P in ("P0", "P1"):
                r[f"RRR_m{m}__{P}"] = crow(2e-3 + m * 1e-4 * (1 + 0.1 * rng.random()))
        for k in ("QLp", "QLpm", "QLs", "LM", "LMb"):
            for m in (9, 17, 33):
                for P in ("P0", "P1"):
                    r[f"{k}_m{m}__{P}"] = crow(1.8e-3 + m * 8e-5)
        for c, h in (("LR0", 8), ("LR", 8), ("LR", 16), ("LR0", 16)):
            for m in (9, 17):
                r[f"{c}_m{m}_h{h}__P0"] = crow(2.0e-3 + (m + h) * 9e-5, 1.2e-4)
                for hid in ("W0", "WI", "WU"):
                    r[f"{c}_m{m}_h{h}__P1_{hid}"] = crow(2.0e-3 + (m + h) * 9e-5, 1.2e-4)
        cm[cid] = r
        fs[cid] = {"u_price": {"value": 4e-6}, "cn_substeps": 4, "FULL256__P0": crow(0.14, 0.01), "FULL256sp__P0": crow(0.30, 0.02),
                   "FULL256cn__P0": crow(0.02, 0.002)}
        be[cid] = {"LM_m9__P0": {"N_star": 1200.0}, "LR_m9_h16__P0": {"N_star": 40000.0}}
    cost = {"status": "done", "models": cm, "full_solvers": fs, "offline": {"per_model": {"LM_m9": {"offline_total_seconds": 12.0},
                                                                                       "LR_m9": {"offline_total_seconds": 180.0}}},
            "break_even": be, "pareto": []}
    wj(os.path.join(R, "cost", "cost.json"), cost)
    wj(os.path.join(R, "fit", "selection.json"), {"status": "done", "selected": {"9": {"LM": {"lam_rel": 1e-6}}}})
    wj(os.path.join(root, "models", "LOCK.json"), {"stub": True, "note": "synthetic"})
    wj(os.path.join(R, "manifests", "CASES.json"), {"cases": [{"id": c, "family": f, "split": s, "params": p} for c, f, s, p in CASES],
                                                    "counts": {"test_interp": 2, "test_ood": 1, "dev_regression": 2}, "sha256": "synthetic"})
    wj(os.path.join(R, "manifests", "ENV.json"), {"python": sys.version.split()[0], "numpy": np.__version__, "synthetic": True})
    wj(os.path.join(R, "verify", "bs_vs_u_family_C.json"), [{"case": "C_test_0.205", "full511_vs_bs": 2.0e-5, "u_grid_price": 4.1e-6},
                                                           {"case": "C_sig0.30", "full511_vs_bs": 5.0e-6, "u_grid_price": 4.0e-6}])
    cmp_ = {"pass": True, "n_compared": 123, "n_failed": 0, "max_rel_dev": 3.4e-9, "max_rel_dev_key": "C_sig0.30:x",
            "cases": {"C_sig0.30": {"status": "done", "metrics": {
                "models.a.price_interior_rms": {"committed": 1.0e-5, "reproduced": 1.0000001e-5, "pass": True},
                "models.a.tiny": {"committed": 1e-12, "reproduced": 3e-12, "pass": True},
                "models.a.negative_mass_frac": {"committed": 0.1, "reproduced": 0.1001, "pass": True}}}}}
    wj(os.path.join(R, "gate5_reproduction", "results", "gate5", "comparison.json"), cmp_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenario", choices=("realistic", "strong"), default="realistic")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    if root == STUDY:
        sys.exit("refusing to write synthetic data into the real study directory")
    build(root, a.scenario)
    print(f"synthetic ({a.scenario}) results tree written under {root}")


if __name__ == "__main__":
    main()
