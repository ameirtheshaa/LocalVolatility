#!/usr/bin/env python3
r"""Figure 2 (deck numbering): why the quotes cannot be differenced directly.

The negative control for the whole method.  Breeden-Litzenberger says the density is
the second strike-derivative of the call price, so the obvious thing to try is to apply
it straight to the quoted prices.  This figure shows what that gives.

It is also the answer to a reasonable question about Figure 3: there, finite differences
are applied to the NETWORK's price output, never to the quotes.  This figure is the
reason -- the quoted ladder is too coarse and too heavily rounded to be differenced
twice, so the finite-difference route only becomes meaningful once a smooth
arbitrage-free surface has been fitted to the quotes.

Both maturities shown are the ones with the most quotes (58 and 62), i.e. the best case
for the naive approach, not the worst.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import pubstyle as PS

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "presentation",
                      "figures", "forward_density")

SHOW = (0, 2)          # tau indices to draw: the two with the densest strike ladders


def main():
    PS.use()
    d = np.load(os.path.join(HERE, "pub_data.npz"))
    st = json.load(open(os.path.join(HERE, "pub_stats.json")))
    taus = list(st["taus"])
    cols = PS.tau_colors(len(taus))
    sc = 1e4

    fig, axes = plt.subplots(1, 2, figsize=(PS.SLIDE_W, 2.16),
                             gridspec_kw=dict(wspace=0.24))
    K = d["K"]
    xlo, xhi = 3400, 8600
    for ax, i in zip(axes, SHOW):
        T = taus[i]
        Kq, d2q = d[f"f2q_K_{i}"], d[f"f2q_d2_{i}"]
        fnn = np.exp(st["r"] * T) * d["f1_an"][i]        # the model density, same units
        row = st["f2_quotes"][i]
        neg = d2q < 0

        ax.axhline(0.0, color=PS.GREY, lw=0.5, zorder=2)
        # the model density: what the fitted surface says
        ax.plot(K, sc * fnn, color=cols[i], lw=1.5, zorder=5)
        # the quote-differenced values: stems, negatives flagged
        ax.vlines(Kq[~neg], 0, sc * d2q[~neg], color=PS.REF, lw=0.55, zorder=3)
        ax.plot(Kq[~neg], sc * d2q[~neg], ls="none", marker="o", ms=1.7, mew=0,
                color=PS.REF, zorder=4)
        ax.vlines(Kq[neg], 0, sc * d2q[neg], color=PS.ACCENT, lw=0.8, zorder=3)
        ax.plot(Kq[neg], sc * d2q[neg], ls="none", marker="v", ms=2.6, mew=0,
                color=PS.ACCENT, zorder=6)

        ax.set_xlim(xlo, xhi)
        top = sc * max(d2q.max(), fnn.max()) * 1.10
        ax.set_ylim(sc * min(d2q.min() * 1.25, -0.02e-3), top)
        PS.thousands(ax)
        ax.set_xlabel(r"strike $K$")
        # the title carries the finding, not the parameters: this is the negative
        # control for the whole method, and tau alone (unique per panel) is enough
        # to tell the two apart -- quote count and dK already live in the caption.
        ax.set_title(rf"$\tau={T:.3f}$: differencing goes negative",
                     fontsize=6.8, pad=3)
        if i == SHOW[0]:
            ax.set_ylabel(r"density  $\times 10^{-4}$")

        # ---- direct labels: say what each mark is, right on the canvas --------
        # "fitted surface": hug the curve on its quiet left flank, clear of the
        # jagged ATM cluster where the huge spikes live.
        i_pk = int(np.argmax(fnn))
        i_lo = int(np.argmin(np.abs(K - (K[i_pk] - 0.19 * (xhi - xlo)))))
        ax.annotate("fitted surface", xy=(K[i_lo], sc * fnn[i_lo]), xytext=(-4, 6),
                    textcoords="offset points", fontsize=6.2, ha="right", va="bottom")

        # "differenced quotes (raw)": the defining feature is the huge spike, so
        # point straight at its tip -- the one point a reader's eye goes to anyway.
        j_sp = int(np.argmax(d2q))
        ax.annotate("differenced\nquotes (raw)",
                    xy=(Kq[j_sp], sc * d2q[j_sp]),
                    xytext=(Kq[j_sp] + 0.16 * (xhi - xlo), 0.72 * top),
                    fontsize=6.2, ha="left", va="center", linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", color=PS.REF, lw=0.6,
                                     shrinkA=2, shrinkB=3))

        # the negative count, once per panel, not a number on every marker: point
        # at the right-most negative strike -- already clear of the ATM cluster --
        # from a fixed corner of the axes, so the label never has to dodge whatever
        # mid-height quote spikes happen to sit at the tau-dependent strikes between.
        Kneg = Kq[neg]
        k_ref = Kneg.max()
        y_ref = sc * d2q[neg][np.argmax(Kneg)]
        ax.annotate(f"{row['n_neg']} of {row['n_interior']} interior\nstrikes negative",
                    xy=(k_ref, y_ref), xycoords="data",
                    xytext=(0.97, 0.14), textcoords="axes fraction",
                    fontsize=6.2, ha="right", va="bottom", linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", color=PS.ACCENT, lw=0.6,
                                     shrinkA=2, shrinkB=3))

    PS.finish(fig, left=0.088, right=0.992, top=0.905, bottom=0.145, wspace=0.24)
    PS.check_fit(fig, "fig2_quotes")
    PS.save(fig, OUTDIR, "fig2_quotes")


if __name__ == "__main__":
    main()
