#!/usr/bin/env python3
r"""Figure 5 (deck numbering): the risk-neutral density vs the realised DAX path.

TWO panels, not four.  The working version carried a LOG-space row beside the level
row, which invited the reader to compare them as equals.  They are not equals: the
left-edge sensitivity test in pub_compute shows the log skew moves by 2.6 when the
grid edge moves between two windows that BOTH retain essentially all the mass, while
the level skew moves by 0.05.  The log statistic is therefore a property of where the
grid was cut, not of the density, so it is stated in the caption as a caveat and kept
off the canvas rather than drawn as though it were a second finding.

Panel (b) is the one to read.  z = Phi^{-1}(Fhat(x)) from the empirical CDF has no
bandwidth at all, so a Gaussian is exactly the 45-degree line and curvature is skew,
with no smoothing choice for a reader to argue with.  Panel (a) is for orientation
and is drawn at a single common bandwidth so the widths are comparable.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

import pubstyle as PS

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "presentation",
                      "figures", "forward_density")


def kde_fixed(x, grid, h):
    """KDE at a prescribed bandwidth, pre-shrunk so the drawn curve has s.d. exactly 1."""
    xs = np.asarray(x, float) / np.sqrt(1.0 + h ** 2)
    k = gaussian_kde(xs)
    k.set_bandwidth(bw_method=h / xs.std(ddof=1))
    return k(grid)


def nq(x):
    """Normal-quantile transform of the empirical CDF at Blom positions.  No bandwidth."""
    xs = np.sort(np.asarray(x, float))
    n = xs.size
    return xs, norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))


def main():
    PS.use()
    d = np.load(os.path.join(HERE, "pub_data.npz"))
    st = json.load(open(os.path.join(HERE, "pub_stats.json")))
    taus = list(st["taus"])
    cols = PS.tau_colors(len(taus))
    his = d["f2_his_level"]
    ours = d["f2_ours_level"]
    H = st["f2_bandwidth"]

    # Skew numbers for the on-canvas annotation -- same level-space statistic
    # pub_captions.py cites (FDhisSkew / FDoursSkewLo / FDoursSkewHi), not a
    # new one, so the canvas and the caption can never disagree.
    mom = st["f2_moments"]
    his_sk = mom["his"]["skew_level"]
    ours_sk = [o["skew_level"] for o in mom["ours"]]
    sk_lo, sk_hi = min(ours_sk), max(ours_sk)

    TITLE_FS = plt.rcParams["axes.titlesize"]
    NOTE_FS = plt.rcParams["legend.fontsize"]
    LEADER = dict(arrowstyle="-", color=PS.GREY, lw=0.6, shrinkA=1.0, shrinkB=1.5)

    L = np.ceil(max(abs(his.min()), abs(his.max())) * 10) / 10
    g = np.linspace(-L, L, 700)

    fig, (ax, axq) = plt.subplots(1, 2, figsize=(PS.SLIDE_W, 2.14),
                                  gridspec_kw=dict(wspace=0.26))

    # ------------------------------------------------------------ (a) density --
    ax.hist(his, bins=42, density=True, color=PS.BAND, edgecolor=PS.BAND_EDGE,
            lw=0.3, zorder=1)
    ax.plot(g, norm.pdf(g), color=PS.GREY, lw=0.9, ls=(0, (3, 2)), zorder=3,
            label="standard Gaussian")
    for i, T in enumerate(taus):
        ax.plot(g, kde_fixed(ours[i], g, H), color=cols[i], lw=1.15, zorder=4)
    ax.plot(g, kde_fixed(his, g, H), color=PS.ACCENT, lw=1.9, zorder=6)
    ax.set_xlim(-L, L)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("standardised value")
    ax.set_ylabel("density")
    # "realised DAX (P)" now carries a direct on-curve label below, so the
    # legend keeps only the one entry it still needs.
    ax.legend(loc="upper left", fontsize=6.2)
    PS.panel_tag(ax, "(a)")

    # Headline + skew numbers, ABOVE the axes -- the message and the numbers
    # a reader needs are on the canvas itself, not only in the LaTeX caption.
    ax.text(0.5, 1.28, "Both left-skewed, in the same direction",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=TITLE_FS, clip_on=False)
    # Shifted right of centre (not x=0.5) so its left edge clears the (a) tag
    # instead of sitting on top of it.
    ax.text(0.58, 1.14,
            f"skew (level): realised (P) {his_sk:+.2f}   vs   "
            f"model (Q) {sk_lo:+.2f} to {sk_hi:+.2f}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=NOTE_FS, clip_on=False)

    # Direct labels beside the curves themselves.  Which curve is the one
    # P-measure path and which five are Q-measure model densities is the
    # figure's most important caveat, so it must not depend on the caption.
    # relpos pins the leader line to the text's own top-right corner (which
    # is where xytext+ha="right",va="top" actually anchors it) instead of
    # matplotlib's default box-centre, which otherwise sends the two leaders
    # swinging across each other.
    ax.annotate("realised DAX path (P)", xy=(1.17, 0.28), xycoords="data",
                xytext=(0.97, 0.97), textcoords="axes fraction",
                ha="right", va="top", fontsize=NOTE_FS,
                arrowprops=dict(LEADER, relpos=(1, 1)))
    ax.annotate("model risk-neutral densities (Q)", xy=(1.5, 0.12),
                xycoords="data", xytext=(0.97, 0.80),
                textcoords="axes fraction", ha="right", va="top",
                fontsize=NOTE_FS, arrowprops=dict(LEADER, relpos=(1, 1)))

    # ----------------------------------------------------- (b) normal-quantile --
    axq.plot([-3.3, 3.3], [-3.3, 3.3], color=PS.GREY, lw=0.9, ls=(0, (3, 2)),
             zorder=2)
    for i, T in enumerate(taus):
        xs, zz = nq(ours[i])
        axq.plot(xs, zz, color=cols[i], lw=1.15, zorder=3)
    xs, zz = nq(his)
    axq.plot(xs, zz, color=PS.ACCENT, lw=1.9, zorder=5,
              label="realised DAX path (P)")
    axq.set_xlim(-3.3, 3.3)
    axq.set_ylim(-3.3, 3.3)
    axq.set_xlabel("standardised value")
    axq.set_ylabel(r"$z=\Phi^{-1}(\hat F)$")
    # "Gaussian (45 deg)" now carries a direct on-line label below, so the
    # legend keeps only the one entry it still needs.
    axq.legend(loc="upper left", fontsize=6.2)
    PS.panel_tag(axq, "(b)")

    axq.text(0.5, 1.28, "Same left-skew, no bandwidth needed",
              transform=axq.transAxes, ha="center", va="bottom",
              fontsize=TITLE_FS, clip_on=False)

    # Direct-label the 45 deg reference line itself: under the empirical-CDF
    # transform it is EXACTLY Gaussian, no bandwidth choice involved, so this
    # line is the one honest yardstick in the panel.  Anchored well below and
    # right of the line, in the wedge every curve stays clear of, then right-
    # aligned so the (rather long) label cannot run off the right edge.
    axq.annotate(r"Gaussian (exact $45^\circ$)", xy=(1.0, 1.0),
                 xycoords="data", xytext=(2.55, -1.0), textcoords="data",
                 ha="right", va="center", fontsize=NOTE_FS,
                 arrowprops=dict(LEADER, relpos=(1, 0.5)))

    # one shared tau key, since neither panel has room for five labels
    hl = [plt.Line2D([], [], color=c, lw=1.15) for c in cols]
    # No legend title: it collided with the (b) panel tag, and "the model's
    # risk-neutral density at the quoted maturities" is caption material anyway.
    fig.legend(hl, [rf"$\tau={t:.3f}$" for t in taus], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=6.3, frameon=False,
               columnspacing=1.5, handlelength=1.7)
    PS.finish(fig, left=0.078, right=0.995, top=0.72, bottom=0.135, wspace=0.26)
    PS.check_fit(fig, "fig5_realised")
    PS.save(fig, OUTDIR, "fig5_realised")


if __name__ == "__main__":
    main()
