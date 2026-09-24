#!/usr/bin/env python3
r"""Figure 1 (deck numbering; the hero): the forward density at every horizon.

This is the claim the validation figures exist to support, so it is drawn before them
and with as little furniture as possible.

  (a)  the density itself at 33 maturities spanning the quoted range, coloured by tau.
       One calibration; the maturities in between were never quoted and are not
       interpolated between fitted densities -- they are read straight off the
       calibrated surface at whatever tau is asked for.
  (b)  the same object as a quantile fan, which is what makes the widening legible as
       a single shape rather than 33 overlapping curves.  The forward S_0 e^{r tau} is
       drawn over the median so the drift and the spread can be separated by eye.

The maturity range is the QUOTED range and nothing beyond it: the surface is not
extrapolated in tau to make the picture look better.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import pubstyle as PS

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "presentation",
                      "figures", "forward_density")


def main():
    PS.use()
    d = np.load(os.path.join(HERE, "pub_data.npz"))
    st = json.load(open(os.path.join(HERE, "pub_stats.json")))
    K = d["K"]
    tg, dens, quant = d["f4_tau"], d["f4_dens"], d["f4_quant"]
    fwd, mean = d["f4_fwd"], d["f4_mean"]
    taus_q = list(st["taus"])
    S0 = st["S0"]
    qs = st["f4"]["fan_q"]

    fig = plt.figure(figsize=(PS.SLIDE_W, 2.34))
    # wspace is generous because the colorbar and its label live BETWEEN the panels;
    # at the default spacing the colorbar label lands on top of panel (b)'s ylabel.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.20, 1.0], wspace=0.62)

    # ------------------------------------------------------ (a) the densities --
    ax = fig.add_subplot(gs[0, 0])
    sc = 1e4
    for i, T in enumerate(tg):
        ax.plot(K, sc * dens[i], color=PS.tau_color(T, tg), lw=0.75,
                alpha=0.95, zorder=2 + i / len(tg))
    ax.axvline(S0, color=PS.ACCENT, lw=0.8, ls=(0, (2.6, 1.8)), zorder=20)
    # left of the line: the tallest (shortest-tau) peak sits just to its right
    ax.annotate(rf"$S_0={S0:,.0f}$", xy=(S0, ax.get_ylim()[1]), xytext=(-3.0, -1.5),
                textcoords="offset points", fontsize=6.2, color=PS.ACCENT,
                ha="right", va="top")
    ax.set_xlim(2600, 10200)
    ax.set_ylim(bottom=0)
    PS.thousands(ax)
    ax.set_xlabel(r"index level $K$")
    ax.set_ylabel(r"risk-neutral density $f_\tau(K)$   $\times 10^{-4}$")
    ax.set_title("one calibration, a density at every horizon",
                 fontsize=6.8, pad=2.5)
    PS.tau_colorbar(fig, ax, tg, fraction=0.055, pad=0.02, aspect=26)
    PS.panel_tag(ax, "(a)")

    # ------------------------------------------------------- (b) quantile fan --
    axf = fig.add_subplot(gs[0, 1])
    lo, q1, med, q3, hi = (quant[:, j] for j in range(5))
    axf.fill_between(tg, lo, hi, color=PS.TAU_CMAP(0.72), alpha=0.22, lw=0)
    axf.fill_between(tg, q1, q3, color=PS.TAU_CMAP(0.45), alpha=0.42, lw=0)
    axf.plot(tg, med, color=PS.TAU_CMAP(0.12), lw=1.4, zorder=4)
    axf.plot(tg, fwd, color=PS.ACCENT, lw=0.9, ls=(0, (2.8, 1.8)), zorder=5)
    axf.set_xlim(tg[0], tg[-1])
    axf.set_ylim(lo.min() - 120, hi.max() + 120)
    # the five quoted maturities as a rug on the axis, where they cannot be mistaken
    # for part of the fan
    for T in taus_q:
        axf.plot([T, T], [0.0, 0.028], transform=axf.get_xaxis_transform(),
                 color=PS.REF, lw=0.9, zorder=8, clip_on=False)
    axf.plot([], [], color=PS.REF, lw=0.9, label="quoted maturities")

    # direct labels at the terminal maturity, where the bands are farthest apart --
    # one tag per series, sparingly, colour-matched to its curve so the reader is
    # not sent hunting through a legend for what each band and line means.  The
    # 50%/90% tags use REF (near-black) rather than the band's own fill colour:
    # the fill at panel opacity is too light to read as text.
    tail = tg[-1]
    axf.annotate(f"{100*(qs[4]-qs[0]):.0f}%", xy=(tail, hi[-1]),
                 xytext=(-2.5, -2.0), textcoords="offset points",
                 fontsize=6.0, color=PS.REF, ha="right", va="top")
    axf.annotate(f"{100*(qs[3]-qs[1]):.0f}%", xy=(tail, q3[-1]),
                 xytext=(-2.5, 2.0), textcoords="offset points",
                 fontsize=6.0, color=PS.REF, ha="right", va="bottom")
    axf.annotate("median", xy=(tail, med[-1]), xytext=(-2.5, 2.0),
                 textcoords="offset points", fontsize=6.2,
                 color=PS.TAU_CMAP(0.12), ha="right", va="bottom")
    axf.annotate(r"forward $S_0e^{r\tau}$", xy=(tail, fwd[-1]),
                 xytext=(-2.5, -2.0), textcoords="offset points",
                 fontsize=6.2, color=PS.ACCENT, ha="right", va="top")

    axf.set_xlabel(r"maturity $\tau$ (years)")
    axf.set_ylabel(r"index level")
    axf.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axf.set_title("checked against the forward", fontsize=6.8, pad=2.5)
    axf.legend(loc="upper left", fontsize=6.0, ncol=1)
    PS.panel_tag(axf, "(b)")

    PS.finish(fig, left=0.088, right=0.985, top=0.86, bottom=0.165, wspace=0.62)
    PS.check_fit(fig, "fig1_evolution")
    PS.save(fig, OUTDIR, "fig1_evolution")


if __name__ == "__main__":
    main()
