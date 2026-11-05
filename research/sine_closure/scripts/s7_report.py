#!/usr/bin/env python3
"""s7: reporting. Reads the results tree (SCC_OUTPUT_ROOT aware) and writes tables, decision, decomposition,
panels, figures and the memo skeleton. Pure Python + NumPy + matplotlib (Agg). Never runs any model.

Outputs (under scc.RESULTS unless noted):
  tables/per_case_models.md, tables/summary_by_split.md, tables/decision.md, tables/WARNINGS.md,
  decision.json, decomposition/decomposition.{md,json}, panels/early_time.md, panels/maturity.md (+ .json),
  figures/*.png, and docs/MEMO.md (under scc.OUTPUT_ROOT/docs).

Every sentence written to the memo is computed from the files; nothing here invents a conclusion.
The decision logic follows PLAN.md section 10 literally; see the `classify` function and RULES below.
Missing optional files give a "–" cell and an entry in tables/WARNINGS.md; they never crash the script.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import re
import sys
from collections import OrderedDict, defaultdict

import numpy as np

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STUDY)
import scc  # noqa: E402  (SCC_OUTPUT_ROOT aware)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = scc.RESULTS
MODELS = scc.MODELS
DOCS = os.path.join(scc.OUTPUT_ROOT, "docs")
WARN: list = []

CLOSURE_ORDER = ["FULL1021", "FULL511", "FULL256", "FULL256sp", "FULL256cn", "BLOCK", "PROJ", "TF", "RRR",
                 "QLp", "QLpm", "QLs", "LM", "LMb", "LR0", "LR"]
DECISION_CLOSURES = ["QLp", "QLpm", "QLs", "LM", "LMb", "LR0", "LR"]
HIDDEN_CLOSURES = ("LR0", "LR")
ORACLES = ("PROJ", "TF", "BLOCK")
DIMS_DECISION = (9, 17)
KEY_RE = re.compile(r"^(?P<clos>[A-Za-z0-9]+)(?:_m(?P<m>\d+))?(?:_h(?P<h>\d+))?__(?P<proto>P0|P1)(?:_(?P<hid>W0|WI|WU))?$")

# metric keys (docs/API_CONTRACT.md, scc/metrics.py)
Q_PRICE = "price_interior_rms"
Q_PRICE_SAME = "same_price_interior_rms"      # scc.evaluate prefixes same-grid metrics with "same_"
Q_PRICE_SAME_ALT = "price_same_interior_rms"
Q_EDYN = "resolved_interior_rms"
Q_DENS = "density_l2_interior_mean"
Q_NEG = "negative_mass_max_post_initial"
Q_NEGF = "negative_mass_final"
Q_INV = "invvol_invalid_frac_full"
Q_COV = "invvol_coverage"
ARB_KEYS = [f"{c}_raw_interior_frac_mean" for c in ("positivity", "monotonicity", "butterfly", "calendar")]


# ----------------------------------------------------------------------------- small helpers
def warn(msg):
    if msg not in WARN:
        WARN.append(msg)


def read_json(path, optional=True):
    if not os.path.isfile(path):
        warn(f"missing file: {os.path.relpath(path, scc.OUTPUT_ROOT)}")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        warn(f"unreadable file {os.path.relpath(path, scc.OUTPUT_ROOT)}: {e}")
        return None


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    return o


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def fin(x):
    return x is not None and isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def fe(x, p=2):
    return f"{x:.{p}e}" if fin(x) else "–"


def ff(x, p=3):
    return f"{x:.{p}f}" if fin(x) else "–"


def fb(x):
    return "–" if x is None else ("yes" if x else "no")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def gmean(vals, floor=1e-16):
    v = [x for x in vals if fin(x)]
    if not v:
        return None
    return float(math.exp(sum(math.log(max(x, floor)) for x in v) / len(v)))


def parse_key(k):
    m = KEY_RE.match(k)
    if not m:
        return None
    d = m.groupdict()
    return {"clos": d["clos"], "m": int(d["m"]) if d["m"] else None, "h": int(d["h"]) if d["h"] else None,
            "proto": d["proto"], "hid": d["hid"]}


def key_sort(k):
    p = parse_key(k)
    if not p:
        return (99, 0, 0, k)
    ci = CLOSURE_ORDER.index(p["clos"]) if p["clos"] in CLOSURE_ORDER else 50
    return (ci, p["m"] or 0, p["h"] or 0, p["proto"] + (p["hid"] or ""))


def short(k):
    return k.replace("__P0", "").replace("__", "|")


# ----------------------------------------------------------------------------- loading
class Data:
    def __init__(self):
        self.scores = OrderedDict()      # case -> score dict
        self.unc = {}                    # case -> uncertainty dict
        self.diag = {}                   # case -> diagnose dict
        self.diag_summary = None
        self.gamma = None
        self.cost = None
        self.selection = None
        self.manifest = None
        self.verify = None
        self.gate5 = None
        self.missing_u = set()
        self.load()

    def load(self):
        for p in sorted(glob.glob(os.path.join(RESULTS, "score", "*.json"))):
            d = read_json(p)
            if not d:
                continue
            if d.get("status") == "failed":
                warn(f"score file for {os.path.basename(p)} has status failed; skipped")
                continue
            cid = d.get("case") or os.path.splitext(os.path.basename(p))[0]
            self.scores[cid] = d
        if not self.scores:
            warn("no score files found in results/score")
        for p in sorted(glob.glob(os.path.join(RESULTS, "uncertainty", "*.json"))):
            d = read_json(p)
            if d:
                self.unc[d.get("case") or os.path.splitext(os.path.basename(p))[0]] = d
        for p in sorted(glob.glob(os.path.join(RESULTS, "diagnose", "*.json"))):
            if os.path.basename(p) == "summary.json":
                continue
            d = read_json(p)
            if d:
                self.diag[d.get("case") or os.path.splitext(os.path.basename(p))[0]] = d
        self.diag_summary = read_json(os.path.join(RESULTS, "diagnose", "summary.json"))
        rg = read_json(os.path.join(RESULTS, "verify", "roundoff_guard.json"))
        if rg:
            g = rg.get("gamma")
            if isinstance(g, dict):
                g = g.get("value", g.get("gamma"))
            self.gamma = float(g) if fin(g) else None
        if self.gamma is None:
            warn("roundoff guard gamma unavailable (results/verify/roundoff_guard.json); negative-mass rule uses gamma = 0")
        self.cost = read_json(os.path.join(RESULTS, "cost", "cost.json"))
        self.selection = read_json(os.path.join(RESULTS, "fit", "selection.json"))
        self.manifest = read_json(os.path.join(RESULTS, "manifests", "CASES.json"))
        self.verify = read_json(os.path.join(RESULTS, "verify", "verify.json"))
        self.gate5 = read_json(os.path.join(RESULTS, "gate5_reproduction", "results", "gate5", "comparison.json"))

    # -- accessors
    def split_of(self, cid):
        return self.scores[cid].get("split")

    def cases(self, split):
        if split == "dev":
            return [c for c in self.scores if self.split_of(c) in ("dev", "dev_regression")]
        return [c for c in self.scores if self.split_of(c) == split]

    def models(self, cid):
        return self.scores[cid].get("models", {})

    def row(self, cid, key):
        return self.models(cid).get(key)

    def n_interior(self, cid):
        s = self.scores[cid]
        for k in ("n_interior", "n_interior_total"):
            if fin(s.get(k)):
                return float(s[k])
        if fin(s.get("n_interior_per_row")):
            return float(s["n_interior_per_row"]) * 24
        return 230.0 * 24

    def gam(self):
        return self.gamma if self.gamma is not None else 0.0

    @staticmethod
    def _uroot(src):
        """uncertainty file layout: {..., 'uncertainty': {per_quantity, u_dyn, ...}} (s3/s5) or the dict itself."""
        if isinstance(src, dict) and isinstance(src.get("uncertainty"), dict):
            return src["uncertainty"]
        return src

    def u(self, cid, q):
        for src in (self._uroot(self.unc.get(cid)), self._uroot(self.scores.get(cid, {}).get("uncertainty"))):
            if not isinstance(src, dict):
                continue
            pq = src.get("per_quantity", src)
            e = pq.get(q) if isinstance(pq, dict) else None
            if isinstance(e, dict):
                e = e.get("u")
            if fin(e):
                return float(e)
        self.missing_u.add(f"{cid}:{q}")
        return None

    def u_dyn(self, cid, m):
        for src in (self._uroot(self.unc.get(cid)), self._uroot(self.scores.get(cid, {}).get("uncertainty"))):
            if not isinstance(src, dict):
                continue
            ud = src.get("u_dyn")
            if isinstance(ud, dict):
                e = ud.get(str(m))
                if isinstance(e, dict):
                    e = e.get("u", max([v for v in e.values() if fin(v)] or [None]) if any(fin(v) for v in e.values()) else None)
                if fin(e):
                    return float(e)
        self.missing_u.add(f"{cid}:u_dyn_m{m}")
        return None

    def all_keys(self, cases):
        ks = set()
        for c in cases:
            ks |= set(self.models(c))
        return sorted(ks, key=key_sort)


def val(row, name):
    """Metric lookup (dotted path allowed) in row['metrics'] then row; NaN -> None."""
    if row is None:
        return None
    for root in (row.get("metrics", {}), row):
        cur = root
        for part in name.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur is not None:
            if isinstance(cur, float) and not math.isfinite(cur):
                return None
            return cur
    return None


def vf(row, name):
    v = val(row, name)
    return float(v) if fin(v) else None


def vsame(row):
    v = vf(row, Q_PRICE_SAME)
    return v if v is not None else vf(row, Q_PRICE_SAME_ALT)


def online_total(row):
    if row is None:
        return None
    o = row.get("online_seconds")
    if isinstance(o, dict):
        return o.get("total")
    return o if fin(o) else None


# ----------------------------------------------------------------------------- 1. per-case table
PER_CASE_HEAD = ["model", "total state", "params", "price int. RMS (refined)", "price same-grid", "resolved int. RMS",
                 "density L2 int.", "neg. mass max post-init.", "neg. mass final", "invvol invalid full", "coverage",
                 "sigma relerr median", "butterfly raw int.", "calendar raw int.", "spectral abscissa max", "stable",
                 "online total s"]


def per_case_row(row, key):
    r = per_case_row_raw(row, key)
    if row.get("unstable") or row.get("failed"):
        for i in (7, 8, 9, 10, 12, 13):      # neg. mass, invvol, coverage, arbitrage fractions are not valid for unstable rows
            r[i] = "–"
    return r


def per_case_row_raw(row, key):
    pc = row.get("param_count")
    if isinstance(pc, dict):
        pc = pc.get("fitted", pc.get("total"))
    stab = val(row, "stability.spectral_abscissa_max")
    if stab is None:
        stab = val(row, "stability.spectral_abscissa")
    return [key, str(row.get("total_state", "–")), str(pc if pc is not None else "–"),
            fe(vf(row, Q_PRICE)), fe(vsame(row)), fe(vf(row, Q_EDYN)), fe(vf(row, Q_DENS)), fe(vf(row, Q_NEG)),
            fe(vf(row, Q_NEGF)), ff(vf(row, Q_INV)), ff(vf(row, Q_COV)), fe(vf(row, "invvol_sigma_relerr_median")),
            ff(vf(row, "butterfly_raw_interior_frac_mean")), ff(vf(row, "calendar_raw_interior_frac_mean")),
            fe(stab), "failed" if row.get("failed") else ("no" if row.get("unstable") else "yes"), fe(online_total(row), 3)]


def case_header(D, cid):
    s = D.scores[cid]
    ud = {m: D.u_dyn(cid, m) for m in (9, 17)}
    h = [f"## {cid} ({s.get('family', '?')}, split {s.get('split', '?')})", ""]
    if s.get("params"):
        h += [f"Parameters: `{json.dumps(s['params'], sort_keys=True)}`", ""]
    h += [f"Numerical uncertainty u: price {fe(D.u(cid, Q_PRICE))}; density L2 {fe(D.u(cid, Q_DENS))}; "
          f"negative mass {fe(D.u(cid, Q_NEG))}; invvol invalid {fe(D.u(cid, Q_INV))}; "
          f"u_dyn (m=9) {fe(ud[9])}, (m=17) {fe(ud[17])}.", ""]
    return h


def report_per_case(D):
    md = ["# Complete per-case model table", "",
          "Every model row is listed, including unstable and failed ones (stable = no). Price columns are interior RMS of the scaled "
          "price against the refined (Full511@16) truth, except the same-grid column (Full256@16). Numbers: `%.2e`, fractions `%.3f`; "
          "`–` = not available. Rows are keyed `<closure>_m<m>[_h<n_h>]__<protocol>[_<hidden>]`.", ""]
    order = sorted(D.scores, key=lambda c: ({"test_interp": 0, "test_ood": 1}.get(D.split_of(c), 2), c))
    for cid in order:
        md += case_header(D, cid)
        rows = [per_case_row(D.row(cid, k), k) for k in sorted(D.models(cid), key=key_sort)]
        md += table(PER_CASE_HEAD, rows) + [""]
    write_text(os.path.join(RESULTS, "tables", "per_case_models.md"), "\n".join(md) + "\n")


# ----------------------------------------------------------------------------- 2. summary by split
def proto_group(key):
    p = parse_key(key)
    return p["proto"] if p else "?"


def baseline33_key(key):
    p = parse_key(key)
    return f"RRR_m33__{p['proto']}" if p else None


def report_summary_by_split(D):
    md = ["# Summary by split and protocol", "",
          "For each model key: geometric mean (GM) and worst case over the cases of the split. Worst = maximum, except coverage (minimum). "
          "Zeros are floored at 1e-16 in geometric means. `within u of RRR33` counts cases where `|q - q(RRR33)| <= u_q,case` "
          "(same protocol, price = refined interior RMS, density = interior L2). Primary held-out evidence is `test_interp`; "
          "`test_ood` enters only the selected-regimes wording; `dev` = development regression cases.", ""]
    for split in ("test_interp", "test_ood", "dev"):
        cases = D.cases(split)
        if not cases:
            md += [f"## {split}", "", "No cases available.", ""]
            warn(f"summary_by_split: no cases in split {split}")
            continue
        for pg in ("P0", "P1"):
            keys = [k for k in D.all_keys(cases) if proto_group(k) == pg]
            if not keys:
                continue
            md += [f"## {split} — protocol {pg} ({len(cases)} cases: {', '.join(cases)})", ""]
            rows = []
            for k in keys:
                have = [c for c in cases if D.row(c, k) is not None]
                g = lambda q: [None if (D.row(c, k).get("unstable") or D.row(c, k).get("failed")) else vf(D.row(c, k), q) for c in have]  # noqa: E731
                worst = lambda xs, mn=False: (min if mn else max)([x for x in xs if fin(x)] or [None]) if any(fin(x) for x in xs) else None  # noqa: E731
                b33 = baseline33_key(k)
                wp = wd = nn = 0
                for c in have:
                    r33 = D.row(c, b33)
                    if r33 is None:
                        continue
                    nn += 1
                    for q, flag in ((Q_PRICE, "p"), (Q_DENS, "d")):
                        x, y, u = (None if (D.row(c, k).get("unstable") or D.row(c, k).get("failed")) else vf(D.row(c, k), q)), vf(r33, q), D.u(c, q)
                        if x is not None and y is not None and u is not None and abs(x - y) <= u:
                            if flag == "p":
                                wp += 1
                            else:
                                wd += 1
                nuns = sum(1 for c in have if (D.row(c, k).get("unstable") or D.row(c, k).get("failed")))
                rows.append([k, f"{len(have)}", fe(gmean(g(Q_PRICE))), fe(worst(g(Q_PRICE))), fe(gmean(g(Q_DENS))), fe(worst(g(Q_DENS))),
                             fe(gmean(g(Q_NEG))), fe(worst(g(Q_NEG))), fe(gmean(g(Q_INV))), fe(worst(g(Q_INV))),
                             ff(gmean(g(Q_COV))), ff(worst(g(Q_COV), True)),
                             f"{wp}/{nn}" if nn else "–", f"{wd}/{nn}" if nn else "–", str(nuns)])
            md += table(["model", "n", "price GM", "price worst", "density L2 GM", "density L2 worst", "negmass post-init GM",
                         "negmass worst", "invvol invalid GM", "invvol invalid worst", "coverage GM", "coverage worst (min)",
                         "price within u of RRR33", "density within u of RRR33", "unstable cases"], rows) + [""]
    write_text(os.path.join(RESULTS, "tables", "summary_by_split.md"), "\n".join(md) + "\n")


# ----------------------------------------------------------------------------- 3. decision (PLAN section 10)
RULES = OrderedDict([
    ("beyond uncertainty", "|dq| > 2 u_q,case (improvement direction: q_Y - q_X > 2 u)"),
    ("noninferior", "q_X <= q_Y + u_q,case"),
    ("negative mass (strict)", "negmass_X <= negmass_Y + gamma, gamma from results/verify/roundoff_guard.json; quantity = max over post-initial rows"),
    ("validity fractions", "frac_X <= frac_Y + max(u, 1/N_interior), N_interior = interior points per row x 24 (230 x 24 when not in the score file)"),
    ("coverage", "cov_X >= cov_Y - max(u, 1/N_interior)"),
    ("resolved_dynamics_improvement", "every test_interp case, both protocols: stable AND e_dyn(X) < e_dyn(RRR_m) - 2 u_dyn"),
    ("useful_closure", "RDI AND refined price RMS(X) < RMS(RRR_m) - 2u on every case/protocol AND noninferior to RRR_m in density L2, negmass (strict), "
                       "positivity/monotonicity/butterfly/calendar raw interior fractions, invvol invalid fraction (full) and coverage on every case/protocol"),
    ("successful_compression", "useful_closure AND on every case/protocol: price <= RRR33 + u_price, density <= RRR33 + u_dens, negmass <= RRR33 + gamma, "
                               "invvol invalid <= RRR33 + max(u, 1/N) AND online warm median(X) < median(RRR33) - max(IQR_X, IQR_RRR33) (cost.json)"),
    ("price_only_compression", "not successful_compression, but RDI AND price improves beyond 2u over RRR_m AND price <= RRR33 + u_price on every case/protocol"),
    ("stateful_advantage", "LR/LR0 only: vs LM at same m (same protocol) e_dyn(LM) - e_dyn(LR) > 2 u_dyn on every case AND noninferior in price, density, negmass, "
                           "arbitrage fractions, invvol invalid, coverage; AND the same two requirements against RRR at m + n_h"),
    ("representation_limited", "results/diagnose/summary.json (PLAN section 10 stopping rule, evaluated on development data); only used when no positive class holds"),
    ("inconclusive", "no positive class, and either (a) for e_dyn or price some cells have gain > 2u and others 0 < gain <= 2u, or (b) the sign of the price gain differs "
                     "between the same-grid and refined truths in some cell"),
    ("no_supported_benefit", "fallback: none of the above; reasons (gains within uncertainty / equal-state unclosed model at least as good / validity deteriorated) are listed"),
])

NONINF_SPEC = [(Q_DENS, "u"), (Q_NEG, "gamma")] + [(k, "frac") for k in ARB_KEYS] + [(Q_INV, "frac"), (Q_COV, "cov")]


def mkcheck(rule, q, cid, proto, xk, yk, x, y, u, tol, label=None):
    c = {"case": cid, "protocol": proto, "quantity": q, "x_model": xk, "y_model": yk, "x": x, "y": y, "u": u, "tol": tol,
         "rule": rule, "group": label}
    if x is None or y is None:
        c.update(ok=False, missing=True, note="value missing or non-finite")
        return c
    c["missing"] = False
    if rule == "beyond":
        c["gain"] = y - x
        c["threshold"] = 2 * (tol if tol is not None else 0.0)
        c["ok"] = bool(y - x > 2 * (tol or 0.0))
    elif rule == "noninf":
        c["threshold"] = y + (tol or 0.0)
        c["ok"] = bool(x <= y + (tol or 0.0))
    elif rule == "noninf_ge":
        c["threshold"] = y - (tol or 0.0)
        c["ok"] = bool(x >= y - (tol or 0.0))
    return c


def tol_for(D, cid, q, kind):
    u = D.u(cid, q)
    if kind == "u":
        return u, (u or 0.0)
    if kind == "gamma":
        return D.gam(), D.gam()
    if kind in ("frac", "cov"):
        return u, max(u or 0.0, 1.0 / D.n_interior(cid))
    raise ValueError(kind)


def entry_key(e, proto):
    k = f"{e['clos']}_m{e['m']}" + (f"_h{e['h']}" if e["h"] else "") + f"__{proto}"
    if proto == "P1" and e["hid"]:
        k += f"_{e['hid']}"
    return k


def rrr_key(m, proto):
    return f"RRR_m{m}__{proto}"


def noninf_checks(D, cid, proto, xk, yk, X, Y, label, include_price=False):
    out = []
    spec = list(NONINF_SPEC)
    if include_price:
        spec = [(Q_PRICE, "u")] + spec
    for q, kind in spec:
        u, tol = tol_for(D, cid, q, kind)
        rule = "noninf_ge" if kind == "cov" else "noninf"
        out.append(mkcheck(rule, q, cid, proto, xk, yk, vf(X, q), vf(Y, q), u, tol, label))
    return out


class Cost:
    """cost.json (s6 layout): models {case: {key: row}}, full_solvers {case: {..., FULL key: row}}; row['warm'] = {median, iqr, min, max}.
    A flat {key: row} layout is accepted too (case label 'all')."""

    def __init__(self, D):
        self.by = defaultdict(dict)       # key -> {case: (median, iqr, cold)}
        c = D.cost if isinstance(D.cost, dict) else {}
        for src in (c.get("models"), c.get("full_solvers")):
            if not isinstance(src, dict):
                continue
            for k, v in src.items():
                if parse_key(k) and isinstance(v, dict):
                    self._ingest("all", k, v)
                elif isinstance(v, dict):
                    for kk, r in v.items():
                        if parse_key(kk) and isinstance(r, dict):
                            self._ingest(k, kk, r)

    def _ingest(self, case, key, r):
        w = r.get("warm") if isinstance(r.get("warm"), dict) else r
        if isinstance(w, dict) and fin(w.get("median")):
            self.by[key][case] = (float(w["median"]), float(w.get("iqr") or 0.0), r.get("cold") if fin(r.get("cold")) else None)

    def resolve(self, key):
        """exact key, else the P0 timing of the same model (noted)."""
        if key in self.by:
            return key, ""
        alt = re.sub(r"__P1.*$", "__P0", key)
        if alt in self.by:
            return alt, f"P0 timing used for {key}"
        return None, ""

    def __call__(self, key):
        """aggregate over measured cases: (median of case medians, max IQR, note) or None."""
        k, note = self.resolve(key)
        if k is None:
            return None
        v = list(self.by[k].values())
        return float(np.median([x[0] for x in v])), float(max(x[1] for x in v)), note

    def compare(self, xk, yk):
        """per-case checks median_X < median_Y - max(IQR_X, IQR_Y) on the cases timed for both; None when not possible."""
        kx, nx = self.resolve(xk)
        ky, ny = self.resolve(yk)
        if kx is None or ky is None:
            return None
        out = []
        for cid in sorted(set(self.by[kx]) & set(self.by[ky])):
            a, b = self.by[kx][cid], self.by[ky][cid]
            thr = b[0] - max(a[1], b[1])
            out.append({"case": cid, "x": a[0], "y": b[0], "u": max(a[1], b[1]), "threshold": thr, "ok": bool(a[0] < thr),
                        "note": "; ".join(n for n in (nx, ny) if n)})
        return out or None


def cost_rows(D):
    return Cost(D)


def evaluate(D, e, cases, costget):
    """All PLAN section-10 checks for one entry (m, closure, n_h, hidden) on the given case set."""
    m = e["m"]
    G = defaultdict(list)
    cells = defaultdict(list)       # for the inconclusive logic
    protos = ("P0", "P1")
    for cid in cases:
        for P in protos:
            xk, rk, k33 = entry_key(e, P), rrr_key(m, P), rrr_key(33, P)
            X, R, R33 = D.row(cid, xk), D.row(cid, rk), D.row(cid, k33)
            if X is None or R is None:
                for grp in ("stable", "edyn_beyond", "price_beyond", "noninf"):
                    G[grp].append({"case": cid, "protocol": P, "quantity": grp, "x_model": xk, "y_model": rk, "ok": False,
                                   "missing": True, "note": "model or baseline row missing", "rule": grp})
                continue
            # stability
            unst = bool(X.get("unstable") or X.get("failed"))
            if unst:
                X = {"metrics": None, "unstable": True}     # unstable rows carry NaN-based metrics: every value check fails
            G["stable"].append({"case": cid, "protocol": P, "quantity": "unstable_flag", "x_model": xk, "y_model": rk,
                                "x": unst, "y": None, "ok": not unst, "missing": False, "rule": "stable"})
            ud = D.u_dyn(cid, m)
            c = mkcheck("beyond", Q_EDYN, cid, P, xk, rk, vf(X, Q_EDYN), vf(R, Q_EDYN), ud, ud, "edyn_beyond")
            G["edyn_beyond"].append(c)
            cells["edyn"].append((c.get("gain"), c["threshold"] if "threshold" in c else None))
            u = D.u(cid, Q_PRICE)
            c = mkcheck("beyond", Q_PRICE, cid, P, xk, rk, vf(X, Q_PRICE), vf(R, Q_PRICE), u, u, "price_beyond")
            G["price_beyond"].append(c)
            cells["price"].append((c.get("gain"), c.get("threshold")))
            gs = None
            if vsame(X) is not None and vsame(R) is not None:
                gs = vsame(R) - vsame(X)
            cells["sign"].append((c.get("gain"), gs))
            G["noninf"] += noninf_checks(D, cid, P, xk, rk, X, R, "noninf")
            # compression vs RRR33
            if R33 is None:
                for g in ("c33_price", "c33_other"):
                    G[g].append({"case": cid, "protocol": P, "quantity": g, "x_model": xk, "y_model": k33, "ok": False,
                                 "missing": True, "note": "RRR33 row missing", "rule": g})
            else:
                u = D.u(cid, Q_PRICE)
                G["c33_price"].append(mkcheck("noninf", Q_PRICE, cid, P, xk, k33, vf(X, Q_PRICE), vf(R33, Q_PRICE), u, u or 0.0, "c33_price"))
                for q, kind in ((Q_DENS, "u"), (Q_NEG, "gamma"), (Q_INV, "frac")):
                    uu, tol = tol_for(D, cid, q, kind)
                    G["c33_other"].append(mkcheck("noninf", q, cid, P, xk, k33, vf(X, q), vf(R33, q), uu, tol, "c33_other"))
            # stateful advantage
            if e["clos"] in HIDDEN_CLOSURES:
                lmk = f"LM_m{m}__{P}"
                LMr = D.row(cid, lmk)
                eqk = rrr_key(m + e["h"], P) if e["h"] else None
                EQ = D.row(cid, eqk) if eqk else None
                for (Y, yk, g1, g2) in ((LMr, lmk, "lm_edyn", "lm_noninf"), (EQ, eqk, "eq_edyn", "eq_noninf")):
                    if Y is None:
                        for g in (g1, g2):
                            G[g].append({"case": cid, "protocol": P, "quantity": g, "x_model": xk, "y_model": yk, "ok": False,
                                         "missing": True, "note": "comparison model row missing", "rule": g})
                        continue
                    G[g1].append(mkcheck("beyond", Q_EDYN, cid, P, xk, yk, vf(X, Q_EDYN), vf(Y, Q_EDYN), ud, ud, g1))
                    G[g2] += noninf_checks(D, cid, P, xk, yk, X, Y, g2, include_price=True)
                if EQ is not None:
                    # equal-state unclosed model at least as good as X (e_dyn and price)
                    G["_eq_as_good"].append(bool(vf(EQ, Q_EDYN) is not None and vf(X, Q_EDYN) is not None and vf(EQ, Q_EDYN) <= vf(X, Q_EDYN)
                                                 and vf(EQ, Q_PRICE) is not None and vf(X, Q_PRICE) is not None and vf(EQ, Q_PRICE) <= vf(X, Q_PRICE)))
    # cost
    cost_checks, cost_status = [], "evaluated"
    for P in protos:
        xk, k33 = entry_key(e, P), rrr_key(33, P)
        cmp_ = costget.compare(xk, k33)
        if cmp_ is None:
            cost_status = "unavailable"
            cost_checks.append({"case": "all", "protocol": P, "quantity": "online_median_seconds", "x_model": xk, "y_model": k33, "ok": False,
                                "missing": True, "note": "cost.json lacks a timing row", "rule": "cost"})
            continue
        for c_ in cmp_:
            cost_checks.append({"case": c_["case"], "protocol": P, "quantity": "online_median_seconds", "x_model": xk, "y_model": k33,
                                "x": c_["x"], "y": c_["y"], "u": c_["u"], "threshold": c_["threshold"], "ok": c_["ok"], "missing": False,
                                "rule": "cost", "note": c_["note"]})
    G["cost"] = cost_checks

    ok = lambda g: bool(G[g]) and all(c["ok"] for c in G[g] if not isinstance(c, bool))  # noqa: E731
    flags = {g: ok(g) for g in ("stable", "edyn_beyond", "price_beyond", "noninf", "c33_price", "c33_other")}
    rdi = flags["stable"] and flags["edyn_beyond"]
    useful = rdi and flags["price_beyond"] and flags["noninf"]
    cost_ok = None if cost_status == "unavailable" else ok("cost")
    success = bool(useful and flags["c33_price"] and flags["c33_other"] and cost_ok)
    price_only = bool((not success) and rdi and flags["price_beyond"] and flags["c33_price"])
    stateful = None
    if e["clos"] in HIDDEN_CLOSURES:
        stateful = bool(ok("lm_edyn") and ok("lm_noninf") and ok("eq_edyn") and ok("eq_noninf"))
    # inconclusive / benefit diagnostics
    incon = {"mixed_edyn": False, "mixed_price": False, "sign_disagree": False}
    for q in ("edyn", "price"):
        beyond = sum(1 for g, t in cells[q] if g is not None and t is not None and g > t)
        below = sum(1 for g, t in cells[q] if g is not None and t is not None and 0 < g <= t)
        incon[f"mixed_{q}"] = bool(beyond and below)
        incon[f"n_beyond_{q}"], incon[f"n_pos_below_{q}"], incon[f"n_cells_{q}"] = beyond, below, len(cells[q])
    incon["sign_disagree"] = bool(any(a is not None and b is not None and a * b < 0 for a, b in cells["sign"]))
    inconclusive = incon["mixed_edyn"] or incon["mixed_price"] or incon["sign_disagree"]
    eqg = G.get("_eq_as_good")
    reasons = []
    if incon["n_beyond_edyn"] == 0 and incon["n_beyond_price"] == 0:
        reasons.append("gains within uncertainty (no cell with a gain beyond 2u in e_dyn or price)")
    if eqg and all(eqg):
        reasons.append("unclosed model of equal total state is at least as good (e_dyn and price, every cell)")
    if not flags["noninf"]:
        reasons.append("validity deteriorated (a noninferiority condition fails in some cell)")
    classes = []
    if rdi:
        classes.append("resolved_dynamics_improvement")
    if useful:
        classes.append("useful_closure")
    if success:
        classes.append("successful_compression")
    if price_only:
        classes.append("price_only_compression")
    if stateful:
        classes.append("stateful_advantage")
    # exceptions per class = failed checks of the groups the class requires
    need = {"resolved_dynamics_improvement": ["stable", "edyn_beyond"],
            "useful_closure": ["stable", "edyn_beyond", "price_beyond", "noninf"],
            "successful_compression": ["stable", "edyn_beyond", "price_beyond", "noninf", "c33_price", "c33_other", "cost"],
            "price_only_compression": ["stable", "edyn_beyond", "price_beyond", "c33_price"],
            "stateful_advantage": ["lm_edyn", "lm_noninf", "eq_edyn", "eq_noninf"]}
    exceptions = {}
    for cls, groups in need.items():
        if cls == "stateful_advantage" and e["clos"] not in HIDDEN_CLOSURES:
            continue
        exc = []
        for g in groups:
            for c in G.get(g, []):
                if isinstance(c, dict) and not c["ok"]:
                    exc.append({k: v for k, v in c.items() if k in ("case", "protocol", "quantity", "x_model", "y_model", "x", "y", "u", "tol",
                                                                    "gain", "threshold", "missing", "note", "group")})
                    exc[-1]["condition"] = g
        exceptions[cls] = exc
    return {"classes": classes, "flags": {**flags, "rdi": rdi, "useful": useful, "cost_ok": cost_ok, "cost_status": cost_status,
                                          "success": success, "price_only": price_only, "stateful": stateful},
            "inconclusive": inconclusive, "inconclusive_detail": incon, "no_benefit_reasons": reasons, "exceptions": exceptions}


def primary_class(res, rep_limited):
    cl = res["classes"]
    for c in ("successful_compression", "price_only_compression", "useful_closure", "resolved_dynamics_improvement"):
        if c in cl:
            return c
    if rep_limited:
        return "representation_limited"
    if res["inconclusive"]:
        return "inconclusive"
    return "no_supported_benefit"


def rep_limited(D, m):
    s = D.diag_summary
    if not isinstance(s, dict):
        return None
    for path in (("rule_per_m", f"m{m}", "representation_limited"), ("rule_per_m", str(m), "representation_limited"), ("representation_limited", str(m)), ("by_m", str(m), "representation_limited"), ("dims", str(m), "representation_limited"),
                 (str(m), "representation_limited")):
        cur = s
        for p in path:
            cur = cur.get(p) if isinstance(cur, dict) else None
        if isinstance(cur, bool):
            return cur
    return None


def discover_entries(D, cases):
    ents = set()
    for cid in cases:
        for k in D.models(cid):
            p = parse_key(k)
            if p and p["clos"] in DECISION_CLOSURES and p["m"] in DIMS_DECISION:
                ents.add((p["m"], p["clos"], p["h"], p["hid"] if p["proto"] == "P1" else None))
    # P0 keys carry no hidden suffix: expand to the hidden variants seen under P1
    merged = set()
    byb = defaultdict(set)
    for m, c, h, hid in ents:
        byb[(m, c, h)].add(hid)
    for (m, c, h), hs in byb.items():
        hs2 = {x for x in hs if x} or {None}
        for hid in hs2:
            merged.add((m, c, h, hid))
    out = [{"m": m, "clos": c, "h": h, "hid": hid} for m, c, h, hid in merged]
    out.sort(key=lambda e: (e["m"], CLOSURE_ORDER.index(e["clos"]), e["h"] or 0, e["hid"] or ""))
    return out


def entry_label(e):
    return f"{e['clos']}_m{e['m']}" + (f"_h{e['h']}" if e["h"] else "") + (f"/{e['hid']}" if e["hid"] else "")


def compute_decision(D):
    costget = cost_rows(D)
    ti, ood = D.cases("test_interp"), D.cases("test_ood")
    if not ti:
        warn("decision: no test_interp cases; classification not possible")
    ents = discover_entries(D, ti + ood)
    out = {"rules": RULES, "gamma": D.gamma, "test_interp_cases": ti, "test_ood_cases": ood, "entries": [],
           "representation_limited": {str(m): rep_limited(D, m) for m in DIMS_DECISION}, "per_m": {}, "S1": {}}
    for e in ents:
        rl = rep_limited(D, e["m"])
        rec = {"entry": entry_label(e), "m": e["m"], "closure": e["clos"], "n_h": e["h"], "hidden": e["hid"],
               "representation_limited_m": rl}
        r = evaluate(D, e, ti, costget) if ti else None
        if r:
            r["primary"] = primary_class(r, rl)
            rec["test_interp"] = r
        ro = evaluate(D, e, ood, costget) if ood else None
        if ro:
            ro["primary"] = primary_class(ro, rl)
            rec["test_ood"] = ro
        regimes = {}
        for cid in ti + ood:
            rc = evaluate(D, e, [cid], costget)
            regimes[cid] = {"split": D.split_of(cid), "family": D.scores[cid].get("family"),
                            "useful_closure": "useful_closure" in rc["classes"],
                            "resolved_dynamics_improvement": "resolved_dynamics_improvement" in rc["classes"]}
        rec["regimes"] = regimes
        out["entries"].append(rec)
    cav = []
    if D.gamma is None:
        cav.append("roundoff guard gamma missing: the negative-mass rule used gamma = 0")
    if D.missing_u:
        cav.append(f"uncertainty u missing for {len(D.missing_u)} (case, quantity) pairs (treated as 0, fractions floored at 1/N_interior): "
                   + ", ".join(sorted(D.missing_u)[:12]) + (" ..." if len(D.missing_u) > 12 else ""))
    if any(r.get("test_interp", {}).get("flags", {}).get("cost_status") == "unavailable" for r in out["entries"]):
        cav.append("cost.json timing rows missing for some entries: the cost condition of successful_compression could not be evaluated (treated as not met)")
    for c in cav:
        warn("decision caveat: " + c)
    out["caveats"] = cav
    # per-m verdict
    for m in DIMS_DECISION:
        es = [r for r in out["entries"] if r["m"] == m and "test_interp" in r]
        best_rank = ["successful_compression", "price_only_compression", "useful_closure", "resolved_dynamics_improvement"]
        helpers = {c: [r["entry"] for r in es if r["test_interp"]["primary"] == c] for c in best_rank}
        stateful = [r["entry"] for r in es if "stateful_advantage" in r["test_interp"]["classes"]]
        sel = {r["entry"]: [cid for cid, g in r["regimes"].items() if g["useful_closure"]] for r in es}
        sel = {k: v for k, v in sel.items() if v}
        top = next((c for c in best_rank if helpers[c]), None)
        if top in ("successful_compression", "price_only_compression", "useful_closure"):
            verdict = "closure_helps"
        elif sel:
            verdict = "helps_only_in_selected_regimes"
        elif top == "resolved_dynamics_improvement":
            verdict = "resolved_dynamics_only_no_deployment_benefit"
        else:
            verdict = "closure_does_not_help"
        out["per_m"][str(m)] = {"verdict": verdict, "top_class": top, "entries_by_class": helpers, "stateful_advantage": stateful,
                                "selected_regimes": sel, "representation_limited": rep_limited(D, m),
                                "unclosed_33_remains_recommendation": not bool(helpers["successful_compression"]),
                                "n_entries": len(es)}
    # S1: m=33 LM/QL density vs RRR33 beyond u_dens
    for clos in ("QLp", "QLpm", "QLs", "LM"):
        chk = []
        for cid in ti:
            for P in ("P0", "P1"):
                xk, yk = f"{clos}_m33__{P}", rrr_key(33, P)
                X, Y = D.row(cid, xk), D.row(cid, yk)
                u = D.u(cid, Q_DENS)
                chk.append(mkcheck("beyond", Q_DENS, cid, P, xk, yk, vf(X, Q_DENS), vf(Y, Q_DENS), u, u))
        if any(not c["missing"] for c in chk):
            out["S1"][clos] = {"density_beyond_uncertainty_every_case": all(c["ok"] for c in chk), "checks": chk}
    return out


def verdict_sentence(m, v):
    vv = v["verdict"]
    if vv == "closure_helps":
        names = ", ".join(v["entries_by_class"][v["top_class"]])
        s = f"m = {m}: closure helps on the held-out interpolation cases (highest class `{v['top_class']}`: {names})."
    elif vv == "helps_only_in_selected_regimes":
        named = "; ".join(f"{k} on {', '.join(c)}" for k, c in v["selected_regimes"].items())
        s = f"m = {m}: no closure passes on every held-out interpolation case; closure helps only in selected regimes: {named}."
    elif vv == "resolved_dynamics_only_no_deployment_benefit":
        s = (f"m = {m}: the best result is resolved-dynamics improvement ({', '.join(v['entries_by_class']['resolved_dynamics_improvement'])}), "
             "which is not a deployment benefit by itself (the useful-closure conditions fail).")
    else:
        s = f"m = {m}: closure does not help (no entry reaches resolved-dynamics improvement on every held-out interpolation case)."
    if v.get("representation_limited"):
        s += " The dimension is representation-limited by the PLAN stopping rule."
    s += (" The unclosed 33-mode model remains the recommendation." if v["unclosed_33_remains_recommendation"]
          else " A closed model meets the successful-compression criteria against the unclosed 33-mode model.")
    if v["stateful_advantage"]:
        s += f" Stateful advantage: {', '.join(v['stateful_advantage'])}."
    return s


def fmt_exc(x):
    q = x.get("quantity")
    base = f"{x.get('case')} / {x.get('protocol')} / {q} [{x.get('condition')}]"
    if x.get("missing"):
        return f"{base}: {x.get('note', 'missing')} ({x.get('x_model')} vs {x.get('y_model')})"
    return (f"{base}: X={fe(x.get('x')) if not isinstance(x.get('x'), bool) else x.get('x')}, Y={fe(x.get('y'))}, "
            f"u={fe(x.get('u'))}, tol={fe(x.get('tol'))}, threshold={fe(x.get('threshold'))}"
            + (f", gain={fe(x.get('gain'))}" if x.get("gain") is not None else ""))


def report_decision(D, dec):
    md = ["# Decision classification (PLAN section 10)", "",
          "Classification per (m, closure, hidden protocol) on the `test_interp` cases under both start protocols. `test_ood` is reported "
          "separately and enters only the selected-regimes wording. Rules as implemented:", ""]
    md += [f"* **{k}**: {v}" for k, v in RULES.items()]
    if dec.get("caveats"):
        md += ["", "**Caveats (inputs missing, classification is provisional):**"] + [f"* {c}" for c in dec["caveats"]]
    md += ["", f"gamma = {fe(D.gamma)}; test_interp cases: {', '.join(dec['test_interp_cases']) or '–'}; test_ood cases: {', '.join(dec['test_ood_cases']) or '–'}.", ""]
    md += ["## Verdict per dimension", ""]
    for m in DIMS_DECISION:
        v = dec["per_m"].get(str(m))
        if v and v["n_entries"]:
            md += [f"* {verdict_sentence(m, v)}"]
        else:
            md += [f"* m = {m}: no closed-model entries available."]
    md += ["", "## Classification table (test_interp)", ""]
    rows = []
    for r in dec["entries"]:
        t = r.get("test_interp")
        if not t:
            continue
        ex = t["exceptions"]
        rows.append([r["entry"], t["primary"], ", ".join(t["classes"]) or "–", fb(t["flags"]["cost_ok"]) if t["flags"]["cost_status"] == "evaluated" else "n/a",
                     str(len(ex.get("resolved_dynamics_improvement", []))), str(len(ex.get("useful_closure", []))),
                     str(len(ex.get("successful_compression", []))), "; ".join(t["no_benefit_reasons"]) if t["primary"] == "no_supported_benefit" else ""])
    md += table(["entry", "primary class", "all classes holding", "cost beats RRR33", "#exc RDI", "#exc useful", "#exc compression", "reasons (no supported benefit)"], rows) + [""]
    md += ["## Exceptions (first 30 per entry and class; complete lists in decision.json)", ""]
    for r in dec["entries"]:
        t = r.get("test_interp")
        if not t:
            continue
        for cls, exc in t["exceptions"].items():
            if cls in t["classes"] or not exc:
                continue
            md += [f"### {r['entry']} — not `{cls}` ({len(exc)} exceptions)", ""]
            md += [f"* {fmt_exc(x)}" for x in exc[:30]]
            if len(exc) > 30:
                md += [f"* ... {len(exc) - 30} more"]
            md += [""]
    md += ["## test_ood outcome (reported separately; selected-regimes wording)", ""]
    rows = []
    for r in dec["entries"]:
        o = r.get("test_ood")
        useful_cases = [cid for cid, g in r["regimes"].items() if g["useful_closure"]]
        rows.append([r["entry"], o["primary"] if o else "–", ", ".join(o["classes"]) if o and o["classes"] else "–",
                     ", ".join(f"{c}" for c in useful_cases) or "–"])
    md += table(["entry", "test_ood primary class", "classes holding", "cases where useful_closure holds (interp+ood, both protocols)"], rows) + [""]
    md += ["## Secondary S1 (m = 33 closed models, density vs RRR33 beyond u_dens on test_interp)", ""]
    if dec["S1"]:
        for clos, s in dec["S1"].items():
            bad = [c for c in s["checks"] if not c["ok"]]
            md += [f"* {clos}_m33: density improvement beyond 2u on every case and protocol: {fb(s['density_beyond_uncertainty_every_case'])}"
                   + ("" if not bad else f" (exceptions: {', '.join(c['case'] + '/' + c['protocol'] for c in bad[:8])}{' ...' if len(bad) > 8 else ''})")]
    else:
        md += ["* no m = 33 closed rows available."]
    write_text(os.path.join(RESULTS, "tables", "decision.md"), "\n".join(md) + "\n")
    write_json(os.path.join(RESULTS, "decision.json"), dec)


# ----------------------------------------------------------------------------- 4. decomposition
def diag_block(d, m, proto):
    """s3 layout: d['models'][<closure>_m<m>__<proto>] rows (PROJ/RRR/TF/BLOCK) + d['decomposition'][RRR key]."""
    mods = d.get("models")
    if isinstance(mods, dict) and any(k.startswith("RRR_m") for k in mods):
        g = lambda c: (mods.get(f"{c}_m{m}__{proto}") or {}).get("metrics") or {}  # noqa: E731
        if not mods.get(f"RRR_m{m}__{proto}"):
            return None
        return {"PROJ": g("PROJ"), "RRR": g("RRR"), "TF": g("TF"), "decomposition": (d.get("decomposition") or {}).get(f"RRR_m{m}__{proto}", {})}
    for root in (d.get("by_m"), d.get("dims"), d.get("m"), d):
        if isinstance(root, dict) and str(m) in root:
            b = root[str(m)]
            if isinstance(b, dict) and proto in b:
                return b[proto]
    return None


def report_decomposition(D):
    out = {"note": "Projection and dynamics parts are orthogonal only in the full-grid Euclidean norm (not in the weighted interior-RMS "
                   "norm used for the scores); the orthogonality defect is reported where available.", "cases": {}, "aggregate": {}}
    md = ["# Projection / dynamics / density decomposition", "",
          "Per dimension m and start protocol, from `results/diagnose`: error of the exact projected reconstruction (PROJ) = projection part; "
          "error from the coefficient dynamics (RRR vs reference coefficients) = dynamics part; interior RMS of the price. "
          "**Note:** the two parts are orthogonal only in the full-grid Euclidean norm; interior-band RMS values do not add in quadrature.", ""]
    if not D.diag:
        warn("decomposition: no diagnose files")
        md += ["No diagnose files available.", ""]
    agg = defaultdict(lambda: defaultdict(list))
    for cid, d in D.diag.items():
        rows = []
        for m in (9, 13, 17, 25, 33, 65):
            for P in ("P0", "P1"):
                b = diag_block(d, m, P)
                if not b:
                    continue
                dec_ = b.get("decomposition", {})
                rec = {"m": m, "protocol": P,
                       "proj_part_interior_rms": dec_.get("proj_interior_rms", dec_.get("proj_rms")),
                       "dyn_part_interior_rms": dec_.get("dyn_interior_rms", dec_.get("dyn_rms")),
                       "orthogonality_defect": dec_.get("orthogonality_defect"),
                       "PROJ_density_l2": vf({"metrics": b.get("PROJ", {})}, Q_DENS), "RRR_density_l2": vf({"metrics": b.get("RRR", {})}, Q_DENS),
                       "TF_density_l2": vf({"metrics": b.get("TF", {})}, Q_DENS),
                       "PROJ_price": vf({"metrics": b.get("PROJ", {})}, Q_PRICE), "RRR_price": vf({"metrics": b.get("RRR", {})}, Q_PRICE),
                       "TF_price": vf({"metrics": b.get("TF", {})}, Q_PRICE)}
                rows.append(rec)
                for k, v in rec.items():
                    if k not in ("m", "protocol"):
                        agg[(m, P)][k].append(v)
        out["cases"][cid] = rows
        if rows:
            md += [f"## {cid} (split {d.get('split', '?')})", ""]
            md += table(["m", "protocol", "projection part int. RMS", "dynamics part int. RMS", "orthogonality defect", "price PROJ", "price RRR", "price TF",
                         "density L2 PROJ", "density L2 RRR", "density L2 TF"],
                        [[r["m"], r["protocol"], fe(r["proj_part_interior_rms"]), fe(r["dyn_part_interior_rms"]), fe(r["orthogonality_defect"]),
                          fe(r["PROJ_price"]), fe(r["RRR_price"]), fe(r["TF_price"]), fe(r["PROJ_density_l2"]), fe(r["RRR_density_l2"]), fe(r["TF_density_l2"])]
                         for r in rows]) + [""]
    if agg:
        md += ["## Geometric mean over the diagnosed cases", ""]
        rows = []
        for (m, P) in sorted(agg):
            a = agg[(m, P)]
            out["aggregate"][f"{m}|{P}"] = {k: gmean(v) for k, v in a.items()}
            g = out["aggregate"][f"{m}|{P}"]
            rows.append([m, P, fe(g.get("proj_part_interior_rms")), fe(g.get("dyn_part_interior_rms")), fe(g.get("PROJ_density_l2")),
                         fe(g.get("RRR_density_l2")), fe(g.get("TF_density_l2")), str(len(a["m"]) if "m" in a else len(next(iter(a.values()))))])
        md += table(["m", "protocol", "projection part", "dynamics part", "density L2 PROJ", "density L2 RRR", "density L2 TF", "n cases"], rows) + [""]
    write_text(os.path.join(RESULTS, "decomposition", "decomposition.md"), "\n".join(md) + "\n")
    write_json(os.path.join(RESULTS, "decomposition", "decomposition.json"), out)


# ----------------------------------------------------------------------------- 5. panels
EARLY_T = ("0.05", "0.10", "0.15")


def early_vals(row):
    ep = val(row, "early_panel")
    out = {}
    if not isinstance(ep, dict):
        return out
    for t, d in ep.items():
        try:
            tk = f"{float(t):.2f}"
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        neg = d.get("negative_mass", d.get(Q_NEG, d.get(Q_NEGF)))
        if isinstance(neg, list):
            neg = max(neg) if neg else None
        out[tk] = {"price": d.get(Q_PRICE), "density": d.get("density_l2_interior", d.get(Q_DENS)), "negmass": neg}
    return out


def report_panels(D):
    md = ["# Early-time panel (protocol P0)", "",
          "Scored at t~ = 0.05, 0.10, 0.15 (payoff start only; the early rows are not part of the 24 output maturities). "
          "Per model: price interior RMS (refined), density L2 (interior mean), negative mass.", ""]
    ej = {}
    for cid in D.scores:
        keys = [k for k in sorted(D.models(cid), key=key_sort) if proto_group(k) == "P0"]
        rows = []
        for k in keys:
            ev = early_vals(D.row(cid, k))
            if not ev:
                continue
            ej.setdefault(cid, {})[k] = ev
            cells = []
            for t in EARLY_T:
                e = ev.get(t, {})
                cells += [fe(e.get("price")), fe(e.get("density")), fe(e.get("negmass"))]
            rows.append([k] + cells)
        if rows:
            md += [f"## {cid} ({D.split_of(cid)})", ""]
            hdr = ["model"] + [f"{q} @ {t}" for t in EARLY_T for q in ("price", "density", "negmass")]
            md += table(hdr, rows) + [""]
    if not ej:
        warn("early-time panel: no early_panel data in score rows")
    write_text(os.path.join(RESULTS, "panels", "early_time.md"), "\n".join(md) + "\n")
    write_json(os.path.join(RESULTS, "panels", "early_time.json"), ej)
    # maturities
    T = np.linspace(0.3, 1.5, 24)
    cols = [0, 3, 7, 11, 15, 19, 23]
    md = ["# Per-maturity price RMS (refined truth, interior band)", "",
          f"Protocol P0. Columns are a subset of the 24 maturities, T = {', '.join(f'{T[i]:.3f}' for i in cols)}; the complete curves are in `maturity.json`.", ""]
    mj = {}
    for cid in D.scores:
        keys = [k for k in sorted(D.models(cid), key=key_sort) if proto_group(k) == "P0"]
        rows = []
        for k in keys:
            v = val(D.row(cid, k), "price_per_maturity_rms")
            if isinstance(v, list) and len(v) == 24:
                mj.setdefault(cid, {})[k] = v
                rows.append([k] + [fe(v[i]) for i in cols])
        if rows:
            md += [f"## {cid} ({D.split_of(cid)})", ""] + table(["model"] + [f"T={T[i]:.2f}" for i in cols], rows) + [""]
    if not mj:
        warn("maturity panel: no price_per_maturity_rms data")
    write_text(os.path.join(RESULTS, "panels", "maturity.md"), "\n".join(md) + "\n")
    write_json(os.path.join(RESULTS, "panels", "maturity.json"), {"T": T, "curves": mj})
    return ej, mj


# ----------------------------------------------------------------------------- 6. figures
FAM_COL = {"FULL": "k", "RRR": "tab:blue", "QL": "tab:orange", "LM": "tab:green", "LR": "tab:red", "PROJ": "tab:gray", "TF": "tab:purple", "BLOCK": "tab:brown"}


def colour(key):
    p = parse_key(key)
    c = p["clos"] if p else key
    for pre in ("FULL", "RRR", "QL", "LMb", "LM", "LR", "PROJ", "TF", "BLOCK"):
        if c.startswith(pre):
            return FAM_COL.get("LM" if pre == "LMb" else pre, "tab:gray")
    return "tab:gray"


def fig_models(D, cases, proto="P0"):
    return [k for k in D.all_keys(cases) if proto_group(k) == proto and (parse_key(k) or {"clos": ""})["clos"] not in ORACLES]


def savefig(fig, name):
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    fig.savefig(os.path.join(RESULTS, "figures", name), dpi=130, bbox_inches="tight")
    plt.close(fig)


def bar_with_worst(ax, labels, means, worst, logy=False, floor=1e-12):
    x = np.arange(len(labels))
    mv = [max(m_, floor) if (fin(m_) and logy) else (m_ if fin(m_) else 0.0) for m_ in means]
    ax.bar(x, mv, color=[colour(k) for k in labels], alpha=0.75)
    wv = [(max(w, floor) if logy else w) if fin(w) else np.nan for w in worst]
    ax.plot(x, wv, "kv", ms=4, label="worst case")
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([short(k) for k in labels], rotation=75, ha="right", fontsize=7)


def online_cost(D, costget, key, cases):
    r = costget(key)
    if r:
        return r[0], "cost.json"
    vals = [online_total(D.row(c, key)) for c in cases]
    vals = [v for v in vals if fin(v)]
    return (float(np.mean(vals)), "score online_seconds (mean over cases)") if vals else (None, None)


def make_figures(D, ej, mj):
    ti = D.cases("test_interp")
    if not ti:
        warn("figures: no test_interp cases")
        return
    keys = fig_models(D, ti, "P0")
    costget = cost_rows(D)

    def stat(k, q, fn=np.mean):
        v = [vf(D.row(c, k), q) for c in ti if D.row(c, k) is not None and not (D.row(c, k).get("unstable") or D.row(c, k).get("failed"))]
        v = [x for x in v if x is not None]
        return (float(fn(v)) if v else None)
    try:
        fig, axs = plt.subplots(2, 1, figsize=(max(9, 0.35 * len(keys) + 3), 8), sharex=True)
        bar_with_worst(axs[0], keys, [stat(k, Q_INV) for k in keys], [stat(k, Q_INV, np.max) for k in keys])
        axs[0].set_ylabel("invvol invalid fraction (full domain)")
        axs[0].legend(fontsize=7)
        bar_with_worst(axs[1], keys, [stat(k, Q_COV) for k in keys], [stat(k, Q_COV, np.min) for k in keys])
        axs[1].set_ylabel("coverage (identifiable points)")
        axs[1].set_ylim(0, 1.02)
        fig.suptitle("Validity and coverage, test_interp mean (bars) and worst case (markers), protocol P0", fontsize=10)
        savefig(fig, "validity_coverage.png")
    except Exception as e:  # noqa: BLE001
        warn(f"figure validity_coverage failed: {e}")
    try:
        fig, ax = plt.subplots(figsize=(max(9, 0.35 * len(keys) + 3), 5))
        bar_with_worst(ax, keys, [stat(k, Q_NEG) for k in keys], [stat(k, Q_NEG, np.max) for k in keys], logy=True)
        ax.set_ylabel("negative mass, max over post-initial rows (floor 1e-12)")
        ax.legend(fontsize=7)
        ax.set_title("Negative mass, test_interp mean and worst case, protocol P0", fontsize=10)
        savefig(fig, "negative_mass.png")
    except Exception as e:  # noqa: BLE001
        warn(f"figure negative_mass failed: {e}")

    def scatter(xfun, xlabel, ylab, qy, name, logx=True, title=""):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        pts = []
        for k in keys:
            y = gmean([vf(D.row(c, k), qy) for c in ti])
            x = xfun(k)
            if y is None or x is None:
                continue
            pts.append((k, x, y))
            ax.plot(x, y, "o", color=colour(k), ms=5)
            ax.annotate(short(k), (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        rr = sorted([(parse_key(k)["m"], x, y) for k, x, y in pts if parse_key(k)["clos"] == "RRR"])
        if rr:
            ax.plot([p[1] for p in rr], [p[2] for p in rr], "-", color=FAM_COL["RRR"], lw=1, alpha=0.6, label="unclosed RRR")
            ax.legend(fontsize=7)
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        savefig(fig, name)
        return len(pts)
    srcs = set()
    try:
        def x_rt(k):
            v, s = online_cost(D, costget, k, ti)
            if s:
                srcs.add(s)
            return v
        n = scatter(x_rt, "online total time per query [s]", "price interior RMS (refined), geometric mean over test_interp", Q_PRICE,
                    "pareto_runtime.png", True, "Error versus online runtime, P0 (" + "; ".join(sorted(srcs)) + ")" if srcs else "Error versus online runtime, P0")
        if not n:
            warn("pareto_runtime: no timing data")
    except Exception as e:  # noqa: BLE001
        warn(f"figure pareto_runtime failed: {e}")

    def x_state(k):
        r = next((D.row(c, k) for c in ti if D.row(c, k) is not None), None)
        s = r.get("total_state") if r else None
        if not fin(s):
            p = parse_key(k)
            s = (p["m"] or 0) + (p["h"] or 0) if p and p["m"] else None
        return s
    try:
        scatter(x_state, "total state (modes + hidden)", "price interior RMS (refined), GM over test_interp", Q_PRICE, "pareto_state.png", True,
                "Error versus total state, P0")
        scatter(x_state, "total state (modes + hidden)", "density L2 (interior mean), GM over test_interp", Q_DENS, "density_vs_state.png", True,
                "Density error versus total state, P0")
    except Exception as e:  # noqa: BLE001
        warn(f"figure state plots failed: {e}")
    # early panel + per maturity
    try:
        sel = [k for k in keys if (parse_key(k)["clos"] in ("RRR", "FULL256") and parse_key(k)["m"] in (None, 9, 17, 33))
               or (parse_key(k)["clos"] in DECISION_CLOSURES and parse_key(k)["m"] == 9)]
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
        tt = [float(t) for t in EARLY_T]
        any_ = False
        for k in sel:
            for ax, qn in zip(axs, ("price", "density", "negmass")):
                ys = []
                for t in EARLY_T:
                    v = [early_vals(D.row(c, k)).get(t, {}).get(qn) for c in ti]
                    ys.append(gmean(v))
                if all(y is not None for y in ys):
                    ax.plot(tt, ys, "o-", ms=3, lw=1, color=colour(k), label=short(k))
                    any_ = True
        for ax, qn in zip(axs, ("price interior RMS", "density L2 (interior mean)", "negative mass")):
            ax.set_yscale("log")
            ax.set_xlabel("t~")
            ax.set_title(qn, fontsize=9)
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=6)
        fig.suptitle("Early-time panel, P0, geometric mean over test_interp", fontsize=10)
        if any_:
            savefig(fig, "early_panel.png")
        else:
            plt.close(fig)
            warn("early_panel.png not written (no early-panel data)")
    except Exception as e:  # noqa: BLE001
        warn(f"figure early_panel failed: {e}")
    try:
        T = np.linspace(0.3, 1.5, 24)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        any_ = False
        for ax, mm in zip(axs, DIMS_DECISION):
            for k in keys:
                p = parse_key(k)
                if not ((p["clos"] in ("RRR",) and p["m"] in (mm, 33)) or p["clos"] == "FULL256" or (p["clos"] in DECISION_CLOSURES and p["m"] == mm)):
                    continue
                curves = [val(D.row(c, k), "price_per_maturity_rms") for c in ti if D.row(c, k) is not None]
                curves = [np.array(c, float) for c in curves if isinstance(c, list) and len(c) == 24]
                if not curves:
                    continue
                g = np.exp(np.mean(np.log(np.maximum(np.array(curves), 1e-16)), axis=0))
                ax.plot(T, g, lw=1.2, color=colour(k), ls="--" if p["clos"] == "RRR" and p["m"] == 33 else "-", label=short(k))
                any_ = True
            ax.set_yscale("log")
            ax.set_xlabel("maturity T")
            ax.set_title(f"m = {mm}", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)
        axs[0].set_ylabel("price interior RMS (refined), GM over test_interp")
        if any_:
            savefig(fig, "per_maturity.png")
        else:
            plt.close(fig)
            warn("per_maturity.png not written (no per-maturity data)")
    except Exception as e:  # noqa: BLE001
        warn(f"figure per_maturity failed: {e}")


# ----------------------------------------------------------------------------- 7. memo
BEGIN, END = "<!-- AUTOGEN:BEGIN {} -->", "<!-- AUTOGEN:END {} -->"
NARR = "<!-- coordinator: narrative -->"


def memo_blocks(D, dec):
    B = OrderedDict()
    ti = D.cases("test_interp")
    # decision
    L = ["## Decision (computed from results/decision.json)", ""]
    for m in DIMS_DECISION:
        v = dec["per_m"].get(str(m))
        L.append("* " + (verdict_sentence(m, v) if v and v["n_entries"] else f"m = {m}: no closed-model entries available."))
    L += ["", "Classification per (m, closure, hidden protocol), test_interp, both start protocols:", ""]
    rows = []
    for r in dec["entries"]:
        t = r.get("test_interp")
        if t:
            rows.append([r["entry"], t["primary"], ", ".join(t["classes"]) or "–", str(sum(len(v) for v in t["exceptions"].values()))])
    L += table(["entry", "primary class", "classes holding", "listed exceptions (all classes)"], rows) if rows else ["(no entries)"]
    s1 = dec.get("S1", {})
    if s1:
        L += ["", "Secondary S1 (m = 33 density gain beyond 2 u_dens, every test_interp case and protocol): "
              + "; ".join(f"{c}: {fb(s['density_beyond_uncertainty_every_case'])}" for c, s in s1.items()) + "."]
    L += ["", f"Full rules, exceptions and the separate test_ood outcome: `results/tables/decision.md`, `results/decision.json`.", "", NARR]
    B["decision"] = "\n".join(L)
    # per-case excerpt
    L = ["## Per-case table excerpt (test_interp, protocol P0)", "", "Complete tables: `results/tables/per_case_models.md`.", ""]
    for cid in ti:
        L += case_header(D, cid)
        keys = [k for k in sorted(D.models(cid), key=key_sort) if proto_group(k) == "P0" and (parse_key(k) or {"clos": ""})["clos"] not in ORACLES]
        head = ["model", "price int. RMS (refined)", "density L2 int.", "neg. mass max post-init.", "invvol invalid full", "coverage", "stable", "online total s"]
        rows = []
        for k in keys:
            r = D.row(cid, k)
            rows.append([k, fe(vf(r, Q_PRICE)), fe(vf(r, Q_DENS)), fe(vf(r, Q_NEG)), ff(vf(r, Q_INV)), ff(vf(r, Q_COV)),
                         "failed" if r.get("failed") else ("no" if r.get("unstable") else "yes"), fe(online_total(r), 3)])
        L += table(head, rows) + [""]
    if not ti:
        L.append("No test_interp cases available.")
    B["per_case"] = "\n".join(L)
    # uncertainty table
    L = ["## Numerical uncertainty u per held-out case", "",
         "u_q = max(u_grid, u_step) (PLAN section 3); u_dyn for the resolved-coordinate error.", ""]
    rows = []
    for cid in D.cases("test_interp") + D.cases("test_ood"):
        rows.append([cid, D.split_of(cid), fe(D.u(cid, Q_PRICE)), fe(D.u(cid, Q_DENS)), fe(D.u(cid, "density_l1_interior_mean")), fe(D.u(cid, Q_NEG)),
                     fe(D.u(cid, Q_INV)), fe(D.u_dyn(cid, 9)), fe(D.u_dyn(cid, 17))])
    L += table(["case", "split", "u price", "u density L2", "u density L1", "u negmass", "u invvol invalid", "u_dyn m=9", "u_dyn m=17"], rows)
    B["uncertainty"] = "\n".join(L)
    # gate 5
    L = ["## Gate 5 regression", ""]
    g = D.gate5
    if isinstance(g, dict) and "n_compared" in g:
        mx, mk = None, None
        for cid, c in (g.get("cases") or {}).items():
            for key, r in (c.get("metrics") or {}).items():
                a, b = r.get("committed"), r.get("reproduced")
                if fin(a) and fin(b) and abs(a) > 1e-8 and "frac" not in key:
                    dv = abs(a - b) / max(abs(a), abs(b))
                    if mx is None or dv > mx:
                        mx, mk = dv, f"{cid}:{key}"
        if mx is None:
            mx, mk = g.get("max_rel_dev"), g.get("max_rel_dev_key")
        L.append(f"Gate 5 reproduction (unchanged original script): pass = {fb(g.get('pass'))}; metrics compared = {g.get('n_compared')}; "
                 f"failed = {g.get('n_failed')}; maximum relative deviation among committed values with |v| > 1e-8 (fractions excluded) = {fe(mx)}"
                 + (f" at `{mk}`" if mk else "") + ".")
    else:
        L.append("Gate 5 reproduction comparison not available (results/gate5_reproduction/results/gate5/comparison.json).")
    B["gate5"] = "\n".join(L)
    # cost
    L = ["## Cost summary", ""]
    c = D.cost
    if isinstance(c, dict):
        cg = Cost(D)
        be_all = defaultdict(list)
        for cid, per in (c.get("break_even") or {}).items():
            if isinstance(per, dict):
                for k, r in per.items():
                    if isinstance(r, dict) and fin(r.get("N_star")):
                        be_all[k].append(r["N_star"])
        rows = []
        for k in sorted(cg.by, key=key_sort):
            r = cg(k)
            cold = [x[2] for x in cg.by[k].values() if x[2] is not None]
            rows.append([k, str(len(cg.by[k])), fe(r[0], 3), fe(r[1], 3), fe(float(np.median(cold)) if cold else None, 3),
                         fe(float(np.median(be_all[k])), 3) if be_all.get(k) else "–"])
        L += table(["model", "timed cases", "online warm median [s] (median over cases)", "max IQR [s]", "cold [s]", "break-even N* vs RRR33 (median)"], rows) \
            if rows else ["No per-model timing rows in cost.json."]
        off = c.get("offline")
        if isinstance(off, dict) and off.get("per_model"):
            L += ["", "Offline cost per closed model [s]: " + "; ".join(
                f"{k}: {fe(v.get('offline_total_seconds'), 3)}" for k, v in off["per_model"].items() if isinstance(v, dict)) + "."]
        L += ["", "A cost difference counts only when the medians differ by more than the larger IQR (PLAN section 9); machine and threads in `results/manifests/TIMING_ENV.json`."]
    else:
        L.append("cost.json not available.")
    B["cost"] = "\n".join(L)
    L = ["## Warnings from the report script", ""]
    L += [f"* {w}" for w in WARN] if WARN else ["None."]
    B["warnings"] = "\n".join(L)
    return B


def write_memo(blocks):
    path = os.path.join(DOCS, "MEMO.md")
    os.makedirs(DOCS, exist_ok=True)
    wrapped = {k: f"{BEGIN.format(k)}\n{v}\n{END.format(k)}" for k, v in blocks.items()}
    if os.path.isfile(path):
        txt = open(path, encoding="utf-8").read()
        for k, w in wrapped.items():
            pat = re.compile(re.escape(BEGIN.format(k)) + r".*?" + re.escape(END.format(k)), re.S)
            if pat.search(txt):
                txt = pat.sub(lambda _m, w=w: w, txt)
            else:
                txt += "\n\n" + w + "\n"
        write_text(path, txt)
        return path
    head = ["# Decision memo: closure test for small boundary-adapted sine models", "",
            f"_Skeleton generated by `scripts/s7_report.py` on {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
            "Blocks between AUTOGEN markers are recomputed on every run (edit outside them). Every statement inside them is computed from "
            "the result files; narrative goes at the marked placeholders._", "", NARR, ""]
    for k in ("decision", "uncertainty", "per_case", "gate5", "cost", "warnings"):
        head += [wrapped[k], "", NARR if k in ("uncertainty", "per_case", "cost") else "", ""]
    write_text(path, "\n".join(head) + "\n")
    return path


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resume", action="store_true", help="accepted for run_all compatibility (the report is always recomputed)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    D = Data()
    steps = [("per_case_models", lambda: report_per_case(D)), ("summary_by_split", lambda: report_summary_by_split(D))]
    for name, fn in steps:
        try:
            fn()
            print(f"wrote {name}")
        except Exception as e:  # noqa: BLE001
            import traceback
            warn(f"step {name} failed: {e!r}")
            traceback.print_exc()
    dec = None
    try:
        dec = compute_decision(D)
        report_decision(D, dec)
        print("wrote decision.json / decision.md")
    except Exception as e:  # noqa: BLE001
        import traceback
        warn(f"decision failed: {e!r}")
        traceback.print_exc()
    ej = mj = None
    for name, fn in (("decomposition", lambda: report_decomposition(D)), ("panels", lambda: report_panels(D))):
        try:
            r = fn()
            if name == "panels":
                ej, mj = r
            print(f"wrote {name}")
        except Exception as e:  # noqa: BLE001
            import traceback
            warn(f"step {name} failed: {e!r}")
            traceback.print_exc()
    if not args.no_figures:
        try:
            make_figures(D, ej, mj)
            print("wrote figures")
        except Exception as e:  # noqa: BLE001
            warn(f"figures failed: {e!r}")
    if dec is not None:
        try:
            path = write_memo(memo_blocks(D, dec))
            print(f"wrote {os.path.relpath(path, scc.OUTPUT_ROOT)}")
        except Exception as e:  # noqa: BLE001
            import traceback
            warn(f"memo failed: {e!r}")
            traceback.print_exc()
    wmd = ["# Report warnings", "", f"{len(WARN)} warning(s)."] + [f"* {w}" for w in WARN]
    write_text(os.path.join(RESULTS, "tables", "WARNINGS.md"), "\n".join(wmd) + "\n")
    print(f"{len(WARN)} warning(s) -> results/tables/WARNINGS.md")
    if dec:
        for m in DIMS_DECISION:
            v = dec["per_m"].get(str(m))
            if v and v["n_entries"]:
                print(verdict_sentence(m, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
