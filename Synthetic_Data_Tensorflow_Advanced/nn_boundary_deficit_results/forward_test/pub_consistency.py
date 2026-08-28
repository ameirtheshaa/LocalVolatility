#!/usr/bin/env python3
r"""Figures 6 and A1 (deck numbering): self-consistency -- density vs reprice under its own sigma.

The headline claim is that two independently derived densities agree, so the headline
figure shows only that: one row, five maturities, the analytic density over the Monte
Carlo reprice, with the MC standard-error band.  Nothing else.

The residual and the implied-vol comparison are real and are kept, but as a COMPANION
figure (figA1_residual) for the appendix.  Putting them under the headline row was what
made the working version hard to read: three rows of five panels each, on one slide,
competing for the eye when only the top row carries the claim.

  fig6_selfconsistency   analytic density vs MC reprice, +/- 2 SE band
  figA1_residual         density residual against the MC noise band, and the
                         implied-vol difference on the quoted strikes
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
    K, taus = d["K"], list(st["taus"])
    cols = PS.tau_colors(len(taus))
    S0 = st["S0"]

    # ================================================ (a) headline: densities ===
    ks_max = max(row["ks"] for row in st["f3_rows"])   # worst-of-5 KS -> the headline bound
    fig, axes = plt.subplots(1, 5, figsize=(PS.SLIDE_W, 2.04),
                             gridspec_kw=dict(wspace=0.16))
    for i, (T, ax) in enumerate(zip(taus, axes)):
        gA, gB, gse = d["f3_gA"][i], d["f3_gB"][i], d["f3_gse"][i]
        sc = 1e4                                    # densities are ~1e-4 per index point
        ax.fill_between(K, sc * (gB - 2 * gse), sc * (gB + 2 * gse),
                        color=PS.BAND, lw=0, zorder=1, rasterized=True)
        ax.plot(K, sc * gB, color=PS.REF, lw=1.5, zorder=2, rasterized=True)
        ax.plot(K, sc * gA, color=cols[i], lw=0.95, ls=(0, (3.0, 1.8)), zorder=3,
                rasterized=True)
        ax.axvline(S0, color=PS.ACCENT, lw=0.55, ls=(0, (1, 1.6)), zorder=4)
        ax.set_xlim(2500, 10500)
        ax.set_ylim(bottom=0)
        ax.set_xticks([4000, 7000, 10000])
        PS.thousands(ax)
        ax.tick_params(labelsize=5.8)
        ax.set_title(rf"$\tau={T:.3f}$: agree" "\n"
                     rf"$\mathrm{{KS}}={st['f3_rows'][i]['ks']:.4f}$",
                     fontsize=6.5, pad=2.5)
        ax.set_xlabel(r"$K$", fontsize=6.4, labelpad=1.0)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"density  $\times 10^{-4}$", fontsize=6.5)
            # Direct-label the two routes on the one panel where they overlap --
            # so the identity of each curve doesn't depend on a trip to the legend.
            j = int(np.argmin(np.abs(K - 5500)))
            k = int(np.argmin(np.abs(K - 6550)))
            ax.annotate("Monte-\nCarlo\nreprice", xy=(K[j], sc * gB[j]),
                        xytext=(4270, 8.85), color=PS.REF, fontsize=5.3,
                        ha="center", va="center", linespacing=1.15,
                        arrowprops=dict(arrowstyle="-", color=PS.REF, lw=0.5,
                                        shrinkA=3, shrinkB=2))
            ax.annotate("analytic", xy=(K[k], sc * gA[k]),
                        xytext=(8450, 8.7), color=cols[0], fontsize=5.3,
                        ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=cols[0], lw=0.5,
                                        shrinkA=3, shrinkB=2))
    hl = [plt.Line2D([], [], color=PS.REF, lw=1.5),
          plt.Line2D([], [], color=cols[2], lw=0.95, ls=(0, (3.0, 1.8))),
          plt.Rectangle((0, 0), 1, 1, fc=PS.BAND, ec="none")]
    fig.legend(hl, [r"Monte Carlo reprice under $\sigma_{NN}$",
                    r"analytic $e^{r\tau}\partial^2C/\partial K^2$",
                    r"$\pm 2\,$SE (MC)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=3, fontsize=6.4,
               frameon=False, columnspacing=2.0, handlelength=1.8)
    fig.text(0.5, 1.0,
             rf"Two independent routes to the density agree: "
             rf"$\mathrm{{KS}} \leq {ks_max:.4f}$ at every maturity",
             ha="center", va="top", fontsize=7.2, fontweight="bold", color=PS.REF)
    PS.finish(fig, left=0.083, right=0.988, top=0.71, bottom=0.155, wspace=0.16)
    PS.check_fit(fig, "fig6_selfconsistency")
    PS.save(fig, OUTDIR, "fig6_selfconsistency")

    # ================================ (b) companion: residual and implied vol ===
    # Worst-of-5 IV disagreement (RMSE across quoted strikes) -> annotate on its panel.
    worst_i = int(np.argmax([row["iv_rmse"] for row in st["f3_rows"]]))
    fig = plt.figure(figsize=(PS.SLIDE_W, 2.32))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.0], hspace=0.52, wspace=0.16)
    for i, T in enumerate(taus):
        gA, gB, gse = d["f3_gA"][i], d["f3_gB"][i], d["f3_gse"][i]
        sc = 1e4
        ax = fig.add_subplot(gs[0, i])
        ax.fill_between(K, -2 * sc * gse, 2 * sc * gse, color=PS.BAND, lw=0, zorder=1,
                        rasterized=True)
        ax.axhline(0.0, color=PS.GREY, lw=0.45, zorder=2)
        ax.plot(K, sc * (gA - gB), color=cols[i], lw=0.8, zorder=3, rasterized=True)
        ax.set_xlim(2500, 10500)
        ax.set_xticks([4000, 7000, 10000])
        PS.thousands(ax)
        ax.tick_params(labelsize=5.6, labelbottom=False)
        ax.set_title(rf"$\tau={T:.3f}$", fontsize=6.5, pad=2.5)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("analytic $-$ MC" "\n" r"($\times 10^{-4}$)", fontsize=6.2)
            # Direct-label the noise band where it is widest -- the legend names
            # it, the on-canvas label shows what it actually bounds here.  Anchor
            # the text as a fraction of the *actual* autoscaled range (never an
            # absolute offset) so it always lands well clear of the panel title,
            # and give the leader line a real vertical drop so it clears the
            # text instead of grazing through it.
            j = int(np.argmax(2 * sc * gse))
            yb_lo, yb_hi = ax.get_ylim()
            ax.annotate(r"$\pm 2\,$SE (MC)", xy=(K[j], 2 * sc * gse[j]),
                        xytext=(K[j] - 1650, yb_lo + 0.62 * (yb_hi - yb_lo)),
                        color=PS.REF, fontsize=5.3, ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=PS.REF, lw=0.5,
                                        shrinkA=3, shrinkB=2))

        axv = fig.add_subplot(gs[1, i])
        Kq, ivn, ivm = d[f"f3_ivK_{i}"], d[f"f3_ivnn_{i}"], d[f"f3_ivmc_{i}"]
        axv.plot(Kq, 100 * ivn, color=cols[i], lw=1.0, marker="o", ms=1.8, mew=0,
                 zorder=3)
        axv.plot(Kq, 100 * ivm, color=PS.REF, lw=0.7, ls=(0, (2.6, 1.6)), marker="s",
                 ms=1.6, mew=0, zorder=4)
        axv.set_xlim(2500, 10500)
        axv.set_xticks([4000, 7000, 10000])
        PS.thousands(axv)
        axv.tick_params(labelsize=5.6)
        axv.set_xlabel(r"$K$", fontsize=6.4, labelpad=1.0)
        if i:
            axv.tick_params(labelleft=False)
        else:
            axv.set_ylabel("implied vol (%)", fontsize=6.2)
        if i == worst_i:
            # Annotate the worst-case IV disagreement (RMSE over quoted strikes)
            # directly on the maturity panel it belongs to.  Anchor y as a
            # fraction of the actual autoscaled range (the quoted-strike curve
            # already runs close to its own ylim top), well below the target so
            # the leader line drops at a real angle instead of grazing the text.
            # Panels are ~0.9in wide -- a one-line label this long clips against
            # the next panel, so it is wrapped to fit inside its own column.
            rmse_pct = 100 * st["f3_rows"][i]["iv_rmse"]
            yv_lo, yv_hi = axv.get_ylim()
            axv.annotate("worst case:" "\n" rf"RMSE {rmse_pct:.2f} vol pts",
                         xy=(Kq[0], 100 * ivm[0]),
                         xytext=(Kq[0] - 950, yv_lo + 0.50 * (yv_hi - yv_lo)),
                         color=PS.ACCENT, fontsize=5.3, fontweight="bold",
                         linespacing=1.2, ha="left", va="center",
                         arrowprops=dict(arrowstyle="-", color=PS.ACCENT, lw=0.5,
                                         shrinkA=3, shrinkB=2))
    hl = [plt.Rectangle((0, 0), 1, 1, fc=PS.BAND, ec="none"),
          plt.Line2D([], [], color=cols[2], lw=1.0, marker="o", ms=1.8, mew=0),
          plt.Line2D([], [], color=PS.REF, lw=0.7, ls=(0, (2.6, 1.6)), marker="s",
                     ms=1.6, mew=0)]
    fig.legend(hl, [r"$\pm 2\,$SE (MC)", r"from $C_{NN}$", r"from the reprice"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.4,
               frameon=False, columnspacing=2.0, handlelength=1.9)
    PS.finish(fig, left=0.098, right=0.988, top=0.875, bottom=0.135, wspace=0.16, hspace=0.52)
    PS.check_fit(fig, "figA1_residual")
    PS.save(fig, OUTDIR, "figA1_residual")


if __name__ == "__main__":
    main()
