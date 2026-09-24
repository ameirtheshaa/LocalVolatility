#!/usr/bin/env python3
r"""Shared publication style for the forward-density deck figures.

ONE design system for all four figures, so they read as a set rather than as four
working sessions.  Everything a figure is allowed to decide for itself is a datum;
everything about how it looks lives here.

FONT.  The deck is beamer's `default` theme at 11pt, whose body face is Computer
Modern Sans.  matplotlib bundles that exact face as `cmss10`, so figures set in it
match the surrounding slide text rather than sitting in DejaVu next to it.  Math is
set in the `cm` fontset for the same reason.  cmss10 has no U+2212, so
axes.unicode_minus is off and matplotlib falls back to the hyphen it does have.

SIZE.  Figures are drawn at their FINAL size on the slide (\textwidth = 398.34pt =
5.512in, \textheight = 252.07pt = 3.487in, measured from the beamer class, not
guessed) and included at width=\linewidth.  At 1:1 a 7pt label in the figure is a
7pt label on the slide -- roughly \scriptsize in an 11pt deck.  Designing large and
letting LaTeX shrink would make every figure a different effective font size, which
is the usual reason decks look inconsistent.

COLOUR.  tau is always the same sequential map (viridis, clipped away from its dark
and light ends so every curve stays legible on white), in the same direction, in
every figure that shows more than one maturity -- that is the one convention the
reader has to learn.  Everything else is drawn from a small fixed set: one near-black
for "the independent second route", one red for reference lines and warnings (the
deck's own ntured, so figures and slides share an accent), and greys for context that
must not compete.  No figure invents a colour.

OUTPUT.  Vector PDF is what the deck includes; a 400-dpi PNG is written alongside for
quick viewing and for anything that cannot take a PDF.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

# ---------------------------------------------------------------- geometry ----
# Measured from the beamer class itself:
#   \the\textwidth  = 398.33860pt   \the\textheight = 252.07480pt   (1in = 72.27pt)
SLIDE_W = 398.3386 / 72.27      # 5.512 in
SLIDE_H = 252.0748 / 72.27      # 3.487 in

# ------------------------------------------------------------------ colour ----
TAU_CMAP = cm.viridis
TAU_LO, TAU_HI = 0.12, 0.88     # clip both ends: pure viridis endpoints read badly on white

REF = "#1a1a1a"                 # the independent second route (FD, reprice)
ACCENT = "#C02026"              # deck ntured -- reference lines, S0, warnings
ACCENT_SOFT = "#E08C90"
GREY = "#8C8C8C"                # context that must not compete
GREY_LIGHT = "#D9D9D9"
BAND = "#EAEEF2"                # shaded regions (trained range, uncertainty)
BAND_EDGE = "#C3CCD6"


def tau_colors(n):
    """n colours along the tau map, in increasing-tau order.  The only tau palette."""
    return TAU_CMAP(np.linspace(TAU_LO, TAU_HI, n))


def tau_norm(taus):
    return colors.Normalize(vmin=float(np.min(taus)), vmax=float(np.max(taus)))


def tau_color(tau, taus):
    """Colour for an arbitrary tau on the same scale the discrete curves use."""
    n = tau_norm(taus)
    return TAU_CMAP(TAU_LO + (TAU_HI - TAU_LO) * n(tau))


def tau_colorbar(fig, ax, taus, label=r"maturity $\tau$ (years)", **kw):
    sm = cm.ScalarMappable(norm=tau_norm(taus), cmap=TAU_CMAP)
    sm.set_clim(float(np.min(taus)), float(np.max(taus)))
    cb = fig.colorbar(sm, ax=ax, **kw)
    cb.set_label(label, fontsize=7.0)
    cb.ax.tick_params(labelsize=6.4, length=2.2, width=0.5)
    cb.outline.set_linewidth(0.5)
    return cb


# --------------------------------------------------------------- rcParams ----
def use():
    """Install the style.  Call once at the top of every figure script."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 120,

        "font.family": "sans-serif",
        "font.sans-serif": ["cmss10", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,     # cmss10 has no U+2212

        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.4,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.2,

        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#111111",
        "text.color": "#111111",
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "grid.color": "#D0D0D0",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.7,
        "axes.axisbelow": True,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,

        "lines.linewidth": 1.25,
        "lines.solid_capstyle": "round",

        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#C8C8C8",
        "legend.borderpad": 0.32,
        "legend.labelspacing": 0.28,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,

        "pdf.fonttype": 42,              # embed as TrueType, not Type3
        "ps.fonttype": 42,
    })


def panel_tag(ax, s, dx=-0.012, dy=1.045):
    """(a), (b), ... in the journal position: above-left of the axes, bold, small."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=7.6,
            fontweight="bold", va="bottom", ha="left")


def thousands(ax, axis="x"):
    """Strike axes carry 4-5 digit numbers; comma separators stop them reading as noise."""
    import matplotlib.ticker as mt
    f = mt.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def log_ticks(ax, axis="both"):
    r"""Decade labels as $10^{n}$, replacing matplotlib's LogFormatterSciNotation.

    Necessary, not cosmetic.  The stock log formatter emits `$\mathdefault{10^{-2}}$`,
    and \mathdefault routes every character -- including the minus -- through
    mathtext's ITALIC font.  Under the `cm` fontset that is cmmi10, which has no
    U+2212, so every negative decade on a log axis renders as a dummy box.  Plain
    `$10^{-2}$` takes the minus from cmsy10 and is fine, so this emits that instead.
    Minor tick labels are suppressed: on a 4-decade axis they are pure clutter.
    """
    import matplotlib.ticker as mt

    def fmt(v, _):
        if v <= 0:
            return ""
        n = int(round(np.log10(v)))
        return rf"$10^{{{n}}}$"

    for a in ((ax.xaxis, ax.yaxis) if axis == "both" else
              (ax.xaxis,) if axis == "x" else (ax.yaxis,)):
        a.set_major_formatter(mt.FuncFormatter(fmt))
        a.set_minor_formatter(mt.NullFormatter())


def finish(fig, **margins):
    """Set the axes rectangle EXPLICITLY.  Deliberately not tight_layout.

    tight_layout refuses to run on a figure carrying a `fig.legend` -- it emits
    "This figure includes Axes that are not compatible with tight_layout" as a
    warning and then leaves the default subplot params in place.  The warning is easy
    to miss, and the failure is silent in the output: the `rect` is ignored, the axes
    stay where matplotlib put them, and a two-line axes title runs straight through
    the figure legend.  Three of the figures here carry a figure legend, so the
    margins are set directly instead and `check_fit` verifies the result.

    Pass any of left/right/top/bottom/wspace/hspace as figure fractions.
    """
    fig.subplots_adjust(**margins)


def check_fit(fig, name=""):
    """Assert nothing is clipped: the drawn extent must lie inside the canvas.

    Replaces eyeballing the PNG.  Returns the overflow on each side in points; all
    zeros means every label, title and legend fits.
    """
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    W, H = fig.get_size_inches()
    over = dict(left=max(0.0, -tb.x0) * 72, bottom=max(0.0, -tb.y0) * 72,
                right=max(0.0, tb.x1 - W) * 72, top=max(0.0, tb.y1 - H) * 72)
    bad = {k: round(v, 2) for k, v in over.items() if v > 0.5}
    print(f"  fit {name or fig.get_label()}: " +
          ("OK" if not bad else f"CLIPPED by {bad} pt"))
    return bad


def save(fig, outdir, stem):
    """Write <stem>.pdf (vector, what the deck includes) and <stem>.png (400 dpi).

    Deliberately NOT bbox_inches="tight".  Cropping to the drawn content makes the
    saved width depend on how wide that figure's y-label happens to be (4.60in to
    4.81in across this set), so \\includegraphics[width=\\linewidth] would scale each
    figure by a different factor and the 7pt labels would land on the slide at
    anything from 8.0 to 8.4pt.  Keeping the full canvas means every figure is
    included at exactly 1:1 and every label is the same size on every slide.
    """
    os.makedirs(outdir, exist_ok=True)
    paths = []
    # dpi=400 on BOTH: the PDF stays vector everywhere except artists marked
    # rasterized=True (dense per-maturity curve/fill stacks), which otherwise
    # silently embed at matplotlib's ~120 dpi default -- fine on screen, not
    # print-safe. Explicit dpi here is what makes rasterization safe to use.
    for ext, dpi in (("pdf", 400), ("png", 400)):
        p = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    for p in paths:
        print(f"  -> {p}  ({os.path.getsize(p)/1024:.0f} KB)")
    return paths
