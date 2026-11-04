#!/usr/bin/env python3
"""Gate 4 — scientific evaluation under the prespecified rules of FROZEN_PLAN.md.

Reads results/gate2/*.json and results/gate3/blocks/*.json, aggregates across independent cases
(never across time steps), applies the acceptance criteria, and writes
results/gate4/evaluation.json + results/gate4/TABLES.md and a checkpoint.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsm import cases, checkpoint  # noqa: E402

G2 = os.path.join(checkpoint.RESULTS, "gate2")
G3 = os.path.join(checkpoint.RESULTS, "gate3", "blocks")
OUT = os.path.join(checkpoint.RESULTS, "gate4")
CLOSURES = ["RRR", "QL+", "QL±", "QL*", "ORACLE", "LM", "LR", "ESN"]
SPLITS = ["train", "val", "test"]


def fmt(x, p=2):
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    return f"{x:.{p}e}"


def agg(vals):
    vals = [v for v in vals if v is not None]
    fin = [v for v in vals if np.isfinite(v)]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "n_finite": len(fin), "mean": float(np.mean(fin)) if fin else float("nan"),
            "min": float(np.min(fin)) if fin else float("nan"), "max": float(np.max(fin)) if fin else float("nan")}


RULE = "strict"          # "strict": any increase of negative mass counts (frozen plan); "tolerant": the v1 5 % allowance
NEGMASS_KEY = "negative_mass_max"   # or "negative_mass_T_max" (last output row), see FROZEN_PLAN Addendum 2
U_SOURCE = "train"       # "train": refinement band from training refinement cases only; "all": v1 (includes S_seed8)


def negmass_increased(a_val, r_val):
    if RULE == "tolerant":
        return a_val > 1.05 * r_val + 1e-12
    return a_val > r_val * (1.0 + 1e-9) + 1e-15


def helps(block, closure, split, u):
    """Prespecified: on EVERY case of the split, stable, interior error below RRR by more than u,
    no increase of negative density mass or invalid inverse-vol fraction relative to RRR."""
    pc, sp = block["per_case"], block["splits"]
    ids = [c for c in pc if sp[c] == split and closure in pc[c]]
    if not ids:
        return {"helps": None, "reason": "closure absent"}
    reasons = []
    for c in ids:
        a, r = pc[c][closure], pc[c]["RRR"]
        if a["unstable"] or not np.isfinite(a["resolved_interior_rms"]):
            reasons.append(f"{c}: unstable")
            continue
        if not (a["resolved_interior_rms"] < r["resolved_interior_rms"] - u):
            reasons.append(f"{c}: gain {r['resolved_interior_rms'] - a['resolved_interior_rms']:+.2e} not > u={u:.2e}")
        if negmass_increased(a[NEGMASS_KEY], r[NEGMASS_KEY]):
            reasons.append(f"{c}: {NEGMASS_KEY} {a[NEGMASS_KEY]:.4e} > RRR {r[NEGMASS_KEY]:.4e}")
        if a["inverse_vol_invalid_frac_mean"] > r["inverse_vol_invalid_frac_mean"] + 1e-12:
            reasons.append(f"{c}: invalid inverse-vol {a['inverse_vol_invalid_frac_mean']:.3f} > RRR {r['inverse_vol_invalid_frac_mean']:.3f}")
    return {"helps": len(reasons) == 0, "n_cases": len(ids), "reasons": reasons}


def refinement_band(block):
    """u from the stored per-case RRR refinement numbers; 'train' excludes held-out cases."""
    per = block["refinement_band"]["per_case"]
    sp = block["splits"]
    use = {c: v for c, v in per.items() if (U_SOURCE == "all" or sp.get(c) == "train")}
    return (max(v["u"] for v in use.values()) if use else float("nan")), sorted(use)


def support_summary(block, u):
    """Direction/magnitude evidence per closure on the held-out split, for the support table."""
    pc, sp = block["per_case"], block["splits"]
    ids = [c for c in pc if sp[c] == "test"]
    out = {}
    for cl in CLOSURES:
        if cl in ("RRR",) or not all(cl in pc[c] for c in ids):
            continue
        gains = [pc[c]["RRR"]["resolved_interior_rms"] - pc[c][cl]["resolved_interior_rms"] for c in ids if not pc[c][cl]["unstable"]]
        out[cl] = {"n_stable": len(gains), "n": len(ids),
                   "frac_improved": float(np.mean([g > 0 for g in gains])) if gains else float("nan"),
                   "min_gain_over_u": float(min(gains) / u) if gains and np.isfinite(u) and u > 0 else float("nan"),
                   "negmass_max_increased_cases": [c for c in ids if not pc[c][cl]["unstable"] and negmass_increased(pc[c][cl]["negative_mass_max"], pc[c]["RRR"]["negative_mass_max"])],
                   "negmass_Tmax_increased_cases": [c for c in ids if not pc[c][cl]["unstable"] and negmass_increased(pc[c][cl]["negative_mass_T_max"], pc[c]["RRR"]["negative_mass_T_max"])]}
    return out


def check_inputs(n_k: int, sub: int, keeps, variants=("A", "B")):
    """Every expected Gate 2 case and Gate 3 block must be present, uncorrupted and computed under
    the current identity (code, inputs, grid, step, refinement); otherwise the evaluation is refused."""
    import importlib.util
    def load(name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    g2, g3 = load("gate2_reference"), load("gate3_compare")
    problems = []
    for case in cases.all_cases():
        for v in variants:
            refine = case.id in cases.REFINEMENT_CASES
            st, detail = checkpoint.result_status(os.path.join(G2, f"{case.id}__{v}.json"), g2.case_identity(case, v, n_k, sub, refine, keeps))
            if st != "done":
                problems.append(f"gate2 {case.id}__{v}: {st} {detail}")
    for v in variants:
        for k in keeps:
            st, detail = checkpoint.result_status(os.path.join(G3, f"{v}_keep{k}.json"), g3.block_identity(v, k, n_k, sub, True))
            if st != "done":
                problems.append(f"gate3 {v}_keep{k}: {st} {detail}")
    return problems


def main():
    global RULE, NEGMASS_KEY, U_SOURCE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["strict", "tolerant"], default="strict")
    ap.add_argument("--negmass", choices=["negative_mass_max", "negative_mass_T_max"], default="negative_mass_max")
    ap.add_argument("--u-source", choices=["train", "all"], default="train")
    ap.add_argument("--suffix", default="", help="output file suffix (e.g. _Tmax)")
    ap.add_argument("--n_k", type=int, default=256)
    ap.add_argument("--sub", type=int, default=16)
    ap.add_argument("--allow-partial", action="store_true", help="score whatever is present (problems are listed in the output)")
    args = ap.parse_args()
    RULE, NEGMASS_KEY, U_SOURCE = args.rule, args.negmass, args.u_source
    t0 = time.time()
    problems = check_inputs(args.n_k, args.sub, cases.KEEPS)
    if problems:
        print(f"input check: {len(problems)} problem(s):")
        for pr in problems:
            print("  ", pr)
        if not args.allow_partial:
            print("refusing to evaluate (use --allow-partial to score what is present)")
            sys.exit(2)
    g2 = {}
    for p in sorted(glob.glob(os.path.join(G2, "*__*.json"))):
        d = json.load(open(p))
        if d.get("status") == "done":
            g2[(d["case"], d["variant"])] = d
    blocks = {}
    for p in sorted(glob.glob(os.path.join(G3, "*.json"))):
        d = json.load(open(p))
        if d.get("status") == "done":
            blocks[(d["variant"], d["keep"])] = d
    ev = {"n_gate2_cases": len(g2), "n_gate3_blocks": len(blocks), "reference": {}, "sign": {}, "closures": {},
          "verdict": {}}
    md = ["# Gate 4 tables (auto-generated by scripts/gate4_evaluate.py)", ""]

    # ---------------- reference validity (Gate 2)
    md += ["## Reference validity (full semidiscrete solves, per variant and family)", "",
           "| variant | family | n | min φ̃ | neg. mass max | butterfly viol. (mean frac) | calendar viol. | inverse-vol invalid (mean frac) | inverse-vol median rel.err | BS interior RMS (C) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for variant in ("A", "B"):
        for fam in ("C", "S", "D"):
            rows = [d for (cid, v), d in g2.items() if v == variant and d["family"] == fam]
            if not rows:
                continue
            rv = [r["reference_validity"] for r in rows]
            rec = {"n": len(rows), "min_phi": min(x["min_phi_rows"] for x in rv),
                   "negative_mass_max": max(r["reference_density"]["negative_mass_max"] for r in rows),
                   "butterfly_frac_mean": float(np.mean([x["butterfly_violation_frac_mean"] for x in rv])),
                   "calendar_frac_mean": float(np.mean([x["calendar_violation_frac_mean"] for x in rv])),
                   "invvol_invalid_mean": float(np.mean([x["inverse_vol_invalid_frac_mean"] for x in rv])),
                   "invvol_relerr_mean": float(np.nanmean([x.get("inverse_vol_median_relerr_mean", np.nan) for x in rv])),
                   "bs_interior_rms": agg([r["bs_comparison"]["interior_rms"] for r in rows if "bs_comparison" in r])}
            ev["reference"][f"{variant}_{fam}"] = rec
            md.append(f"| {variant} | {fam} | {rec['n']} | {fmt(rec['min_phi'])} | {fmt(rec['negative_mass_max'])} | {rec['butterfly_frac_mean']:.3f} | {rec['calendar_frac_mean']:.3f} | {rec['invvol_invalid_mean']:.3f} | {fmt(rec['invvol_relerr_mean'])} | {fmt(rec['bs_interior_rms'].get('mean')) if rec['bs_interior_rms']['n'] else '–'} |")
    md.append("")
    md += ["## Refinement / truncation checks (Gate 2 refinement cases)", "",
           "| variant | case | grid 256 vs 512 interior RMS | time sub16 vs sub32 (D) | domain truncation interior RMS |", "|---|---|---|---|---|"]
    for (cid, v), d in g2.items():
        if "refinement" in d:
            r = d["refinement"]
            ev["reference"].setdefault("refinement", {})[f"{v}_{cid}"] = r
            md.append(f"| {v} | {cid} | {fmt(r['grid_refinement_interior_rms_256_vs_512'])} | {fmt(r.get('time_refinement_interior_rms_sub16_vs_sub32'))} | {fmt(r['domain_truncation_interior_rms'])} |")
    md.append("")

    # ---------------- sign diagnostics (Gate 2 per keep)
    md += ["## Sign diagnostics per variant and truncation (across all cases)", "",
           "| variant | keep | ν* operator: min / max | frac ν*<0 | repo FD ν (nonneg) mean | repo FD ν unclipped: min / max | work>0 frac (Eucl.) | work>0 frac (weighted) | max Re eig L_UU (max over cases) | proj. err interior (mean) | memory/Markov ratio (median) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for variant in ("A", "B"):
        for keep in cases.KEEPS:
            rows = [d["per_keep"][str(keep)] for (cid, v), d in g2.items() if v == variant]
            if not rows:
                continue
            nu = [r["nu_star_operator"] for r in rows]
            rec = {"n": len(rows), "nu_star_min": min(nu), "nu_star_max": max(nu), "frac_nu_star_negative": float(np.mean(np.array(nu) < 0)),
                   "nu_repo_nonneg_mean": float(np.mean([r["nu_fit_repo_nonneg"] for r in rows])),
                   "nu_repo_unclipped_min": min(r["nu_fit_repo_unclipped"] for r in rows),
                   "nu_repo_unclipped_max": max(r["nu_fit_repo_unclipped"] for r in rows),
                   "work_euclid_pos_frac_mean": float(np.mean([r["work_euclid_frac_positive"] for r in rows])),
                   "work_weighted_pos_frac_mean": float(np.mean([r["work_weighted_frac_positive"] for r in rows])),
                   "abscissa_L_UU_max": max(r["abscissa_L_UU_end"] for r in rows),
                   "proj_err_mean": float(np.mean([r["projection_err_interior_rms"] for r in rows])),
                   "mem_ratio_median": float(np.median([r["memory_to_markovian_ratio_median"] for r in rows]))}
            ev["sign"][f"{variant}_keep{keep}"] = rec
            md.append(f"| {variant} | {keep} | {fmt(rec['nu_star_min'])} / {fmt(rec['nu_star_max'])} | {rec['frac_nu_star_negative']:.2f} | {fmt(rec['nu_repo_nonneg_mean'])} | {fmt(rec['nu_repo_unclipped_min'])} / {fmt(rec['nu_repo_unclipped_max'])} | {rec['work_euclid_pos_frac_mean']:.2f} | {rec['work_weighted_pos_frac_mean']:.2f} | {fmt(rec['abscissa_L_UU_max'])} | {fmt(rec['proj_err_mean'])} | {rec['mem_ratio_median']:.3f} |")
    md.append("")

    # ---------------- closure scores (Gate 3)
    verdicts = {}
    for (variant, keep), b in sorted(blocks.items()):
        key = f"{variant}_keep{keep}"
        u, u_cases = refinement_band(b)
        pc, sp = b["per_case"], b["splits"]
        present = [c for c in CLOSURES if any(c in pc[x] for x in pc)]
        rec = {"u": u, "u_cases": u_cases, "u_v1_all_cases": b["refinement_band"]["u"], "rule": RULE, "negmass_key": NEGMASS_KEY,
               "support": support_summary(b, u), "nu": b["fit"]["nu_by_closure"], "nu_unclipped": b["fit"]["nu_unclipped"],
               "gap_reproducible": b["gap_reproducible"], "ridge": {k: v.get("chosen") for k, v in b["ridge_selection"].items()},
               "scores": {}, "helps_test": {}, "helps_val": {}}
        md += [f"## Closure scores — variant {variant}, keep {keep} (dim R = {b['dim_R']}, u = {fmt(u)} from {u_cases} [v1 with S_seed8: {fmt(b['refinement_band']['u'])}], rule = {RULE} on {NEGMASS_KEY}, ν_unclipped = {fmt(b['fit']['nu_unclipped'])}, oracle gap reproducible = {b['gap_reproducible']})", "",
               "| closure | ν / ridge | split | n | stable | resolved interior RMS mean [min, max] | price interior RMS mean | density L2 mean | neg. mass max | inverse-vol invalid frac mean | butterfly viol. mean | closed abscissa max | helps (rule) |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for cl in present:
            for split in SPLITS:
                ids = [c for c in pc if sp[c] == split and cl in pc[c]]
                if not ids:
                    continue
                rows = [pc[c][cl] for c in ids]
                stable = sum(1 for r in rows if not r["unstable"])
                s = {"n": len(ids), "stable": stable,
                     "resolved_interior": agg([r["resolved_interior_rms"] for r in rows]),
                     "price_interior": agg([r["price_interior_rms"] for r in rows]),
                     "density_l2": agg([r.get("density_l2_interior_mean") for r in rows]),
                     "negative_mass_max": agg([r.get("negative_mass_max") for r in rows]),
                     "invvol_invalid": agg([r.get("inverse_vol_invalid_frac_mean") for r in rows]),
                     "butterfly": agg([r.get("butterfly_violation_frac_mean") for r in rows]),
                     "closed_abscissa": agg([r.get("closed_abscissa_end") for r in rows])}
                rec["scores"].setdefault(cl, {})[split] = s
                h = helps(b, cl, split, u) if cl != "RRR" else {"helps": None}
                if split == "test":
                    rec["helps_test"][cl] = h
                if split == "val":
                    rec["helps_val"][cl] = h
                par = fmt(b["fit"]["nu_by_closure"].get(cl)) if cl.startswith("QL") else (fmt(rec["ridge"].get(cl), 0) if cl in rec["ridge"] else "–")
                ri = s["resolved_interior"]
                md.append(f"| {cl} | {par} | {split} | {s['n']} | {stable}/{s['n']} | {fmt(ri.get('mean'))} [{fmt(ri.get('min'))}, {fmt(ri.get('max'))}] | {fmt(s['price_interior'].get('mean'))} | {fmt(s['density_l2'].get('mean'))} | {fmt(s['negative_mass_max'].get('max'))} | {s['invvol_invalid'].get('mean', float('nan')):.3f} | {s['butterfly'].get('mean', float('nan')):.3f} | {fmt(s['closed_abscissa'].get('max'))} | {('yes' if h['helps'] else 'no') if h.get('helps') is not None else '–'} |")
        md.append("")
        md += [f"### Per-case resolved interior RMS on the held-out split — {key}", "",
               "| case | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
        for c in [x for x in pc if sp[x] == "test"]:
            md.append(f"| {c} | " + " | ".join((fmt(pc[c][cl]["resolved_interior_rms"]) + (" (unstable)" if pc[c][cl]["unstable"] else "")) if cl in pc[c] else "–" for cl in present) + " |")
        md.append("")
        ev["closures"][key] = rec
        ht = rec["helps_test"]
        nu_u = b["fit"]["nu_unclipped"]
        qlpm_helps = bool(ht.get("QL±", {}).get("helps"))
        qlp_helps = bool(ht.get("QL+", {}).get("helps"))
        qls_helps = bool(ht.get("QL*", {}).get("helps"))
        qls_unstable = any(pc[c]["QL*"]["unstable"] for c in pc if sp[c] == "test") if "QL*" in present else None
        state_helps = any(bool(ht.get(n, {}).get("helps")) for n in ("LR", "ESN"))
        scalar_helps = qlpm_helps or qlp_helps or qls_helps
        if (qlpm_helps or qls_helps) and nu_u < 0:
            v = "overly restrictive"
        elif (not qlpm_helps or qlp_helps) and (nu_u >= 0 or (not qls_helps)):
            v = "justified"
        else:
            v = "unresolved"
        verdicts[key] = {"verdict": v, "nu_unclipped": nu_u, "QL+_helps": qlp_helps, "QL±_helps": qlpm_helps,
                         "QL*_helps": qls_helps, "QL*_unstable_on_test": qls_unstable, "H_state_supported": bool(state_helps and not scalar_helps),
                         "any_scalar_helps": scalar_helps, "state_helps": state_helps}
    ev["verdict"]["per_block"] = verdicts
    sign_B = [ev["sign"][k]["frac_nu_star_negative"] for k in ev["sign"] if k.startswith("B_")]
    sign_A = [ev["sign"][k]["frac_nu_star_negative"] for k in ev["sign"] if k.startswith("A_")]
    ev["verdict"]["H_sign_rejected_OP_B"] = bool(sign_B and min(sign_B) == 1.0)
    ev["verdict"]["H_sign_rejected_OP_A"] = bool(sign_A and min(sign_A) == 1.0)
    ev["verdict"]["H_scalar_supported_anywhere"] = any(v["any_scalar_helps"] for v in verdicts.values())
    ev["verdict"]["H_state_supported_anywhere"] = any(v["H_state_supported"] for v in verdicts.values())
    ev["verdict"]["summary_by_variant"] = {var: sorted({v["verdict"] for k, v in verdicts.items() if k.startswith(var + "_")}) for var in ("A", "B")}
    md += ["## Verdicts (prespecified rules)", "", "| block | ν_unclipped | QL+ helps | QL± helps | QL* helps | QL* unstable on test | stateful helps where scalar does not | verdict |", "|---|---|---|---|---|---|---|---|"]
    for k, v in verdicts.items():
        md.append(f"| {k} | {fmt(v['nu_unclipped'])} | {v['QL+_helps']} | {v['QL±_helps']} | {v['QL*_helps']} | {v['QL*_unstable_on_test']} | {v['H_state_supported']} | **{v['verdict']}** |")
    md += ["", f"H-sign (eliminated modes add diffusion, ν* ≥ 0) rejected on OP-B at every keep: {ev['verdict']['H_sign_rejected_OP_B']}; on OP-A: {ev['verdict']['H_sign_rejected_OP_A']}.",
           f"H-scalar supported in at least one block: {ev['verdict']['H_scalar_supported_anywhere']}.",
           f"H-state supported in at least one block: {ev['verdict']['H_state_supported_anywhere']}.", ""]
    ev["scoring"] = {"rule": RULE, "negmass_key": NEGMASS_KEY, "u_source": U_SOURCE}
    ev["input_check"] = {"n_k": args.n_k, "sub": args.sub, "problems": problems, "allow_partial": bool(args.allow_partial)}
    md.insert(1, f"Scoring: rule = {RULE}, negative-mass metric = {NEGMASS_KEY}, refinement band from = {U_SOURCE} refinement cases (see FROZEN_PLAN.md Addendum 2).")
    ev["wall_seconds"] = time.time() - t0
    ev["status"] = "done"
    os.makedirs(OUT, exist_ok=True)
    checkpoint.write_json(os.path.join(OUT, f"evaluation{args.suffix}.json"), ev)
    with open(os.path.join(OUT, f"TABLES{args.suffix}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    checkpoint.save_checkpoint("gate4", command=f"python3 research/dupire_signed_memory/scripts/gate4_evaluate.py --rule {RULE} --negmass {NEGMASS_KEY} --u-source {U_SOURCE}" + (f" --suffix {args.suffix}" if args.suffix else ""), exit_code=0,
                               done=list(verdicts.keys()), failed=[], pending=[], seeds={}, config={},
                               next_command="python3 research/dupire_signed_memory/scripts/run_all.py --resume")
    print(json.dumps(ev["verdict"]["per_block"], indent=None))
    print({k: v for k, v in ev["verdict"].items() if k != "per_block"})
    for key, rec in ev["closures"].items():
        print(key, "u=%.2e (v1 %.2e)" % (rec["u"], rec["u_v1_all_cases"]), {cl: (h["helps"], len(h.get("reasons", []))) for cl, h in rec["helps_test"].items()})
    print(f"wrote {OUT}/evaluation{args.suffix}.json and TABLES{args.suffix}.md")


if __name__ == "__main__":
    main()
