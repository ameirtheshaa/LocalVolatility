#!/usr/bin/env python3
r"""Realised DAX data: the P-measure side of the forward test.

Everything here reads data/dax_gdaxi_close.csv, which is a one-off yfinance snapshot of
^GDAXI committed to the repo.  Nothing downstream touches the network: the offload runner
has no yfinance, and a paper result should not depend on a live API.

Provenance check that matters: Yahoo's close on 2001-08-07 is 5752.509765625, which is
EXACTLY the S0 in dax_expk/data_multi/7_8_2001_underlying.csv.  Same index, same scale, so
realised levels and quoted strikes can be put on one axis without a rebasing fudge.
assert_same_index() enforces this.

The five 7-August expiries are real third Fridays -- 21 Sep 2001, 19 Oct 2001, 21 Dec 2001,
15 Mar 2002, 21 Jun 2002 -- so "the realised outcome at expiry" is a well-defined number
and not an interpolation.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "dax_gdaxi_close.csv")

# Quoted spot on each calibration date, from <tag>_underlying.csv.
SPOT = {"7aug": 5752.51, "8aug": 5614.51, "9aug": 5512.28,
        "4jul": 6015.72, "8jun": 6187.21}
DATE = {"7aug": "2001-08-07", "8aug": "2001-08-08", "9aug": "2001-08-09",
        "4jul": "2001-07-04", "8jun": "2001-06-08"}

TRADING_DAYS = 252.0
DAYS_PER_YEAR = 365.25


def load_dax():
    """Full ^GDAXI close series as a float Series indexed by Timestamp."""
    df = pd.read_csv(CSV, parse_dates=["Date"])
    return pd.Series(df["Close"].to_numpy(float), index=df["Date"], name="Close")


def assert_same_index(c=None, tol=0.01):
    """Fail loudly if the downloaded series is not the index the quotes are written on."""
    c = load_dax() if c is None else c
    for tag, s in SPOT.items():
        got = float(c.loc[pd.Timestamp(DATE[tag])])
        assert abs(got - s) < tol, (
            f"{tag}: yfinance close {got} != quoted spot {s}. The CSV is not the "
            f"series the option file is written against; do not overlay them.")
    return True


def window(start, n_trading, c=None):
    """The first n_trading closes on/after `start` -- exactly Nicolas's slicing."""
    c = load_dax() if c is None else c
    return c.loc[pd.Timestamp(start):].iloc[:n_trading]


def expiry_date(base_date, tau, c=None):
    """First trading day on/after base_date + tau years, with the realised close.

    tau is measured in calendar time (the pipeline's convention: T*365.25 = days), so the
    lookup is calendar-based.  Returns (Timestamp, level).
    """
    c = load_dax() if c is None else c
    want = pd.Timestamp(base_date) + pd.Timedelta(days=round(tau * DAYS_PER_YEAR))
    i = min(c.index.searchsorted(want), len(c) - 1)
    return c.index[i], float(c.iloc[i])


def realised_at_expiries(base_date, taus, c=None):
    """DataFrame of the realised outcome at each expiry: date, level, return."""
    c = load_dax() if c is None else c
    S0 = float(c.loc[pd.Timestamp(base_date)])
    rows = []
    for t in taus:
        d, S = expiry_date(base_date, t, c)
        rows.append(dict(tau=t, cal_days=round(t * DAYS_PER_YEAR), expiry=d,
                         weekday=d.day_name(), S=S, ret_pct=100.0 * (S / S0 - 1.0)))
    return pd.DataFrame(rows)


def hist_returns(before, tau, c=None):
    """Overlapping tau-horizon log returns from history strictly BEFORE `before`.

    Overlapping is deliberate: it uses every observation, and the estimator is unbiased.
    What it costs is independence -- the returned eff_n is len//n_step, the number of
    genuinely non-overlapping blocks, and it is small (12-96 here).  Report it.
    """
    c = load_dax() if c is None else c
    s = np.log(c.loc[:pd.Timestamp(before)].to_numpy(float))
    n = int(round(tau * TRADING_DAYS))
    r = s[n:] - s[:-n]
    return r, max(len(r) // n, 1), n


def realised_vol(start, end, c=None):
    """Annualised close-to-close vol actually realised over [start, end]."""
    c = load_dax() if c is None else c
    w = c.loc[pd.Timestamp(start):pd.Timestamp(end)]
    return float(np.diff(np.log(w.to_numpy(float))).std(ddof=1) * np.sqrt(TRADING_DAYS))


if __name__ == "__main__":
    c = load_dax()
    assert_same_index(c)
    print(f"{CSV}\n  {len(c)} closes, {c.index[0].date()} -> {c.index[-1].date()}")
    print("  spot check vs quoted underlying: OK for", ", ".join(SPOT))
    print()
    taus = [0.123, 0.200, 0.373, 0.603, 0.871]
    df = realised_at_expiries("2001-08-07", taus, c)
    df["real_vol_%"] = [100 * realised_vol("2001-08-07", d, c) for d in df["expiry"]]
    print("Realised outcomes, calibration date 7 Aug 2001 (S0 = 5752.51):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    for n in (219, 316):
        w = window("2001-08-07", n, c)
        x = np.log(w.to_numpy(float))
        print(f"  window {n:3d} trading days -> ends {w.index[-1].date()}  "
              f"mean(lnS) {x.mean():.3f}  sd {x.std(ddof=1):.3f}")
