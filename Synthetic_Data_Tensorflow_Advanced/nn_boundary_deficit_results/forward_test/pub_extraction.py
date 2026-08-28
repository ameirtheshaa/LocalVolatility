#!/usr/bin/env python3
r"""Figures 3 and 4 (deck numbering): the density extraction is correct.

Two files, because the two claims need different amounts of room and cramming them
together is what made the working version unreadable:

  fig3_extraction  the five quoted maturities, autodiff against central differences,
                   each with a residual strip carrying the float32 stencil envelope.
  fig4_stepsize    the step-size sweep -- the U-curve whose two branches have fitted
                   slopes -2 and +2 -- and the retained-mass check for the window.

Everything that was an in-plot annotation in the working version (the caveat block,
the per-panel error statistics, the "grey dotted line is the float32 floor" note) is
gone from the canvas and lives in the LaTeX caption, which is where a reader of a
paper expects to find it.
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
    klo, khi = st["K_train_lo"], st["K_train_hi"]

    # worst-case scale-relative residual (max|autodiff-FD|/peak, any tau, any K) --
    # drives both the headline annotation below and the deck's own \FDworstScale
    # macro (2.2e-3), so it is computed once here rather than asserted.
    scale_max = np.array([e["scale_max"] for e in st["f1_err"]])
    i_worst = int(np.argmax(scale_max))
    worst_val = scale_max[i_worst]
    worst_exp = int(np.floor(np.log10(worst_val)))
    worst_mant = worst_val / 10 ** worst_exp

    # ============================================ (a) extraction comparison ====
    fig = plt.figure(figsize=(PS.SLIDE_W, 2.34))   # 2.42 overflowed the slide by ~4pt
    gs = fig.add_gridspec(2, 5, height_ratios=[2.45, 1.0], hspace=0.10, wspace=0.14)
    # a common y-scale makes the five maturities comparable at a glance; the peak
    # falls by ~4x across them, so each panel is normalised by its own peak instead
    # and the absolute peak is reported on the panel.
    for i, T in enumerate(taus):
        an, fd, env = d["f1_an"][i], d["f1_fd"][i], d["f1_env"][i]
        pk = st["f1_err"][i]["peak"]

        ax = fig.add_subplot(gs[0, i])
        ax.axvspan(klo, khi, color=PS.BAND, lw=0, zorder=0)
        ax.axhline(0.0, color=PS.GREY, lw=0.45)
        ax.axvline(S0, color=PS.ACCENT, lw=0.6, ls=(0, (1, 1.6)))
        ax.plot(K, an / pk, color=cols[i], lw=1.5, zorder=3)
        ax.plot(K, fd / pk, color=PS.REF, lw=0.7, ls=(0, (3.2, 2.0)), zorder=4)
        ax.set_xlim(K[0], 13000)
        ax.set_ylim(-0.12, 1.08)
        ax.set_title(rf"$\tau={T:.3f}$", fontsize=7.0, pad=3)
        ax.set_xticks([2500, 7500, 12500])
        PS.thousands(ax)
        ax.tick_params(labelbottom=False)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"$\partial^2C/\partial K^2$  (peak-normalised)", fontsize=6.6)
            # Direct-label the two routes flanking the peak they both trace out.
            # In axes-fraction coordinates, not data K: each panel is only ~0.9in
            # wide, so a data-anchored label's rendered width (fixed in points)
            # covers a wildly different K-span per panel than intuition suggests,
            # and a shoulder anchor close to the spike (tried first) drove the
            # label straight through the peak. Axes-fraction sidesteps that --
            # 0.34-0.51 is where the peak rises above the flat flanks, so the two
            # labels are anchored just outside it and grow further away from it.
            ax.text(0.32, 0.15, "autodiff", color=cols[0], fontsize=5.2,
                    transform=ax.transAxes, ha="right", va="bottom", zorder=6)
            ax.text(0.55, 0.15, "central diff.", color=PS.REF, fontsize=5.2,
                    transform=ax.transAxes, ha="left", va="bottom", zorder=6)
            ax.text(S0 + 150, -0.078, r"$S_0$", color=PS.ACCENT, fontsize=6.0,
                    ha="left", va="center", zorder=6)

        axr = fig.add_subplot(gs[1, i])
        axr.axvspan(klo, khi, color=PS.BAND, lw=0, zorder=0)
        axr.axhline(0.0, color=PS.GREY, lw=0.45)
        axr.plot(K, (an - fd) / pk, color=PS.ACCENT, lw=0.7, zorder=3)
        axr.plot(K, env / pk, color=PS.GREY, lw=0.55, ls=(0, (1, 1.4)), zorder=2)
        axr.plot(K, -env / pk, color=PS.GREY, lw=0.55, ls=(0, (1, 1.4)), zorder=2)
        axr.set_xlim(K[0], 13000)
        # scaled to the data (worst scale-relative residual is ~2e-3), not to a round
        # number: at +/-0.02 the whole strip was a flat line and showed nothing.
        axr.set_ylim(-0.0042, 0.0042)
        axr.set_yticks([-0.004, 0, 0.004])
        axr.set_yticklabels(["-0.004", "0", "0.004"])
        axr.set_xticks([2500, 7500, 12500])
        PS.thousands(axr)
        axr.set_xlabel(r"strike $K$", fontsize=6.6, labelpad=1.5)
        if i:
            axr.tick_params(labelleft=False)
        else:
            axr.set_ylabel("residual", fontsize=6.6)
            # label the float32 floor where it is cleanly separated from both
            # zero and the residual curve: left of where the at-the-money
            # excursion begins (axes-fraction, for the same reason as above).
            axr.text(0.02, 0.64, "float32 floor", fontsize=4.8,
                      color="#111111", transform=axr.transAxes,
                      ha="left", va="bottom", zorder=6)

        if i == i_worst:
            # headline number: worst scale-relative residual, any tau, any K.
            # "worst" itself is said once, in the suptitle; this is the number,
            # in the one panel it occurs in.  No leader line to the exact K: at
            # this maturity the residual is a dense oscillation over a wide
            # band (visible below), not one isolated spike, so a pointer in the
            # curve's own colour would either vanish into it or misrepresent a
            # single noisy sample as *the* worst point.
            axr.text(0.03, 0.90, rf"${worst_mant:.1f}\times10^{{{worst_exp}}}$",
                      transform=axr.transAxes, fontsize=5.4, color=PS.ACCENT,
                      ha="left", va="top", zorder=6)

    fig.suptitle(rf"Two independent routes to the same second derivative agree "
                 rf"to ${worst_mant:.1f}\times10^{{{worst_exp}}}$ of peak at worst",
                 x=0.538, y=0.995, fontsize=7.4, fontweight="bold",
                 color="#111111")
    PS.finish(fig, left=0.093, right=0.982, top=0.845, bottom=0.135, wspace=0.14, hspace=0.10)
    PS.check_fit(fig, "fig3_extraction")
    PS.save(fig, OUTDIR, "fig3_extraction")

    # ================================================= (b) step-size sweep ====
    sw = st["f1_sweep"]
    hs = np.array(sw["hs"])
    fig = plt.figure(figsize=(PS.SLIDE_W, 2.12))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    for i, T in enumerate(taus):
        ax.loglog(hs, sw["per_tau"][i], "-", lw=0.7, color=cols[i], alpha=0.85,
                  marker="o", ms=1.8, mew=0, zorder=3)
    ax.loglog(hs, sw["overall_mean"], "-", lw=1.8, color=PS.REF, zorder=5)
    ax.axvline(sw["h_star"], color=PS.ACCENT, lw=0.9, ls=(0, (3, 2)), zorder=4)
    # guide lines at the two theoretical slopes, anchored at the minimum
    hm, em = sw["h_star"], sw["best"]
    hl_, hr_ = hs[hs <= 2.0], hs[hs >= 150.0]
    ax.loglog(hl_, em * (hl_ / hm) ** -2.0, color=PS.GREY, lw=0.55, ls=(0, (1, 1.5)),
              zorder=2)
    ax.loglog(hr_, em * (hr_ / hm) ** 2.0, color=PS.GREY, lw=0.55, ls=(0, (1, 1.5)),
              zorder=2)
    PS.log_ticks(ax)
    ax.set_ylim(6e-5, 2e2)
    # branch labels, in axes coordinates, kept clear of the curves -- named by
    # mechanism rather than left as a bare fitted number
    ax.text(0.045, 0.30, "float32 roundoff\n"
            rf"slope ${sw['slope_round']:+.2f}$ (theory $-2$)",
            transform=ax.transAxes, fontsize=6.0, color="#111111", ha="left", va="top")
    ax.text(0.985, 0.60, r"$h^2$ truncation" "\n"
            rf"slope ${sw['slope_trunc']:+.2f}$ (theory $+2$)",
            transform=ax.transAxes, fontsize=6.0, color="#111111", ha="right", va="top")
    # the minimum itself, direct-labelled on the canvas rather than left to the legend
    ax.text(sw["h_star"], 5e-3, rf"$h={sw['h_star']:g}$", color=PS.ACCENT,
            fontsize=6.0, ha="center", va="bottom", zorder=6)
    ax.set_xlabel(r"central-difference step $h$  (strike units)")
    ax.set_ylabel(r"mean $|\mathrm{FD}-\mathrm{autodiff}|\,/\,\max|\partial^2C/\partial K^2|$",
                  fontsize=6.2)
    ax.grid(alpha=0.45, which="major")
    hl2 = [plt.Line2D([], [], color=PS.REF, lw=1.8),
           plt.Line2D([], [], color=cols[2], lw=0.7, marker="o", ms=1.8, mew=0)]
    ax.legend(hl2, ["pooled mean", "each maturity"],
              loc="upper center", fontsize=6.0, ncol=1)
    PS.panel_tag(ax, "(a)")

    # (b) retained mass.  Drawn as points, NOT bars: every value sits within 5e-4 of
    # 1, so bars would need a truncated baseline and would misrepresent the ratios.
    axm = fig.add_subplot(gs[0, 1])
    mass = [m["mass"] for m in st["f1_mass"]]
    xx = np.arange(len(taus))
    axm.axhline(1.0, color=PS.GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=2)
    axm.axhline(0.99, color=PS.ACCENT, lw=0.9, ls=(0, (3, 2)), zorder=3)
    axm.text(-0.275, 0.99, "0.99", color=PS.ACCENT, fontsize=5.2,
             ha="center", va="bottom", zorder=6)
    axm.vlines(xx, 0.99, mass, color=PS.GREY_LIGHT, lw=0.8, zorder=3)
    axm.scatter(xx, mass, s=22, c=cols, edgecolors=PS.REF, linewidths=0.4, zorder=5)
    # one annotation, not a number at every point: the worst (smallest) case only
    i_worst_mass = int(np.argmin(mass))
    axm.text(xx[i_worst_mass] + 0.12, mass[i_worst_mass] + 0.0013,
              f"worst {mass[i_worst_mass]:.4f}", color=PS.ACCENT, fontsize=6.0,
              ha="left", va="bottom", zorder=6)
    axm.set_xticks(xx)
    axm.set_xticklabels([f"{t:.3f}" for t in taus], fontsize=6.0)
    axm.set_xlim(-0.55, len(taus) - 0.45)
    axm.set_ylim(0.9875, 1.0022)
    axm.set_xlabel(r"maturity $\tau$ (years)")
    axm.set_ylabel("retained mass")
    axm.grid(alpha=0.45, axis="y")
    PS.panel_tag(axm, "(b)")

    PS.finish(fig, left=0.098, right=0.990, top=0.905, bottom=0.175, wspace=0.30)
    PS.check_fit(fig, "fig4_stepsize")
    PS.save(fig, OUTDIR, "fig4_stepsize")


if __name__ == "__main__":
    main()
