#!/usr/bin/env python3
r"""Second-derivative machinery for the 7 August 2001 exp_k calibration (nbexpk7).

Shared by fig_d2_nn_vs_fd.py, fig_discounted_nicolas_vs_nn.py and fig_nn_d2_vs_reprice.py so
the three figures cannot drift apart in how they read the surface.

WHICH ROUTE "THE PIPELINE" ACTUALLY USES -- checked, not assumed.
dupire_pipeline.PDFAnalyzer.extract_model_implied_density (and its unclipped sibling
_raw_model_density) uses NESTED tf.GradientTape AUTOMATIC DIFFERENTIATION -- there is no finite
difference anywhere in that path.  It differentiates the normalised price phi~ twice with respect
to the scaled strike k~ and then chain-rules to K:

    k~ = e^{-rT} K / K_max                      (prepare_data_for_nn)
    d2C/dK2 = d2phi~/dk~2 * (e^{-rT}/K_max)^2 * S0
    f(K)    = e^{rT} * d2C/dK2

BOTH SIDES NOW COME FROM THE PIPELINE -- no local re-implementation of the ansatz.
This module used to source the exp_k call map from forward_test.price_expk, because
PDFAnalyzer.phi_tilde_from_nn offered only 'transformed' (1 - exp(-NN_phi)) and 'legacy'
(raw), and its constructor rejected anything else -- so pointing it at nbexpk7, whose
metadata records ansatz = "exp_k", would have silently applied the wrong ansatz.  The
pipeline-side exp_k support has since landed: PDFAnalyzer.__init__ resolves the price
mapping from metadata['ansatz'] and phi_tilde_from_nn has an 'exp_k' branch

    phi~ = exp(-k~ * softplus(a + k~ * NN_phi)),   a = y + log1p(-e^{-y}),  y = K_max/S0

so the whole read -- ansatz, derivative and chain rule -- is now the pipeline's own code:

    call_nn()     -> S0 * PDFAnalyzer.phi_tilde_from_nn
    d2_autodiff() -> PDFAnalyzer._raw_model_density, de-discounted

_analyzer() asserts that S0, r, K_max and T_max agree between the pipeline analyzer and
forward_test, since a silent constant mismatch there is the one way this swap could move the
numbers.  Cross-checked against the previous route on the five quoted maturities over
K in [100, 15000]: feeding both formulas the same (t~, k~) makes them bit-identical, and end
to end they agree to 1.5e-5 relative worst case (1.4 float32 ulps of the price scale), the
1-ulp difference in how each side rounds k~ into float32.

d2_fd() is the independent check: central differences on the network's OWN call output, no tape.

WHAT THIS MODULE DOES NOT DO.  It says nothing about whether the surface is right -- only about
whether two ways of reading the SAME surface agree, and whether a strike window keeps the mass.
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ADV = os.path.dirname(os.path.dirname(HERE))        # Synthetic_Data_Tensorflow_Advanced
sys.path.insert(0, HERE)
sys.path.insert(0, ADV)

import forward_test as FT      # noqa: E402
from config import DupirePipelineConfig   # noqa: E402
from dupire_pipeline import PDFAnalyzer   # noqa: E402

# np.trapz was removed in numpy 2.0 in favour of the identical np.trapezoid.  dupire_pipeline
# still calls np.trapz, so it only runs under numpy < 2, while this module was written against
# numpy >= 2 -- importing the pipeline here forces the two to coexist.  The two functions are
# the same routine under two names, so aliasing is numerically a no-op.
_trapz = getattr(np, "trapezoid", None) or np.trapz

DAY = "7aug"

# The comparison grid.  K_LO > 2*h_max keeps the central-difference stencil at strictly
# positive strikes: the exp_k ansatz is exp(-k~ * softplus(...)), which for k~ < 0 returns
# more than S0 and is meaningless, so FD must never be asked for a negative strike.
K_LO, K_HI, N_K = 100.0, 15000.0, 5961

# Chosen by h_sweep(); see fig_d2_nn_vs_fd.py, which plots the sweep that justifies it.
H_FD = 50.0


def load():
    """(phi, eta, S0, Tmax, taus) for the 7 Aug calibration.

    Tmax is derived through FT.tmax_of (= max quoted call maturity, as the training pipeline
    sets it), never hardcoded.  taus are the quoted maturities in 7-Aug time.
    """
    phi, eta = FT.load_models(DAY)
    S0 = float(FT.DAYS[DAY]["S0"])
    Tmax = FT.tmax_of(DAY)
    taus = [float(t) for t in sorted(FT.load_quotes(DAY)["T"].unique())]
    return phi, eta, S0, Tmax, taus


_ANALYZER = {}


def _analyzer(phi, S0, Tmax):
    """A PDFAnalyzer wrapped around THESE weights, with its exp_k mapping auto-resolved.

    The mapping is never passed in: phi_mapping=None makes PDFAnalyzer read it off
    metadata['ansatz'], which is the behaviour being validated.  It is then asserted to be
    'exp_k', so a metadata edit or a regression in the resolution table fails loudly here
    instead of quietly pricing nbexpk7 under the wrong ansatz.

    Every scaling constant is asserted against forward_test as well.  S0, r, K_max and T_max
    enter both the ansatz offset a = K_max/S0 + log1p(-e^{-K_max/S0}) and the chain rule
    (e^{-rT}/K_max)^2, so a silent disagreement between the two sides -- e.g. T_max hardcoded
    rather than derived from the quotes -- is the one way this route could differ from the
    forward_test one by more than float32 round-off.

    nn_eta is None on purpose: this module only ever reads the phi side (price and its second
    strike derivative).  The eta network is loaded separately by load() and driven through
    forward_test.mc_paths, so anything eta-dependent on the analyzer would fail loudly here
    rather than quietly using a stand-in surface.
    """
    key = (id(phi), float(S0), float(Tmax))
    if key not in _ANALYZER:
        meta = json.load(open(os.path.join(FT.MODELS, FT.DAYS[DAY]["model"], "metadata.json")))
        mc = meta["config"]
        cfg = DupirePipelineConfig(S0=mc["S0"], r=mc["r"], K_min=mc["K_min"],
                                   K_max=mc["K_max"], T_min=mc["T_min"], T_max=mc["T_max"])
        an = PDFAnalyzer(nn_phi=phi, nn_eta=None, config=cfg, metadata=meta, phi_mapping=None)
        assert an.phi_mapping == "exp_k", (
            f"metadata ansatz={meta.get('ansatz')!r} resolved to phi_mapping="
            f"{an.phi_mapping!r}, expected 'exp_k'")
        for name, got, want in (("S0", an.config.S0, float(S0)), ("r", an.config.r, FT.R),
                                ("K_max", an.k_max, FT.KMAX), ("T_max", an.t_max, float(Tmax))):
            assert got == want, f"{name} mismatch: pipeline {got!r} vs forward_test {want!r}"
        _ANALYZER[key] = an
    return _ANALYZER[key]


def call_nn(phi, S0, Tmax, T, K):
    """C_NN(T, K) in currency units, from PDFAnalyzer.phi_tilde_from_nn (float32 network).

    C = S0 * phi~, the pipeline's own normalisation (phi_norm='s0' in prepare_data).
    """
    K = np.asarray(K, dtype=float)
    an = _analyzer(phi, S0, Tmax)
    t_tilde, k_tilde = an.prepare_data_for_nn(float(T), K)
    return (an.config.S0 * an.phi_tilde_from_nn(t_tilde, k_tilde)).numpy().flatten().astype(float)


def d2_autodiff(phi, S0, Tmax, T, K):
    """d2C_NN/dK2 by nested tf.GradientTape -- the pipeline's own extraction, called directly.

    dupire_pipeline.PDFAnalyzer._raw_model_density returns e^{rT} d2C/dK2 (it differentiates
    phi~ twice in k~ and applies (dk~/dK)^2 * S0 with dk~/dK = e^{-rT}/K_max), so the discount
    is stripped back off here to keep this function's meaning -- the bare second derivative --
    unchanged for its callers.  _raw_model_density is the unclipped, unnormalised sibling of
    extract_model_implied_density: the negative excursions are the diagnostic, so the clipping
    and renormalisation in the public method must not be in this path.
    """
    K = np.asarray(K, dtype=float)
    an = _analyzer(phi, S0, Tmax)
    return an._raw_model_density(float(T), K) * np.exp(-FT.R * float(T))


def d2_fd(phi, S0, Tmax, T, K, h=H_FD):
    """(C(K+h) - 2C(K) + C(K-h)) / h^2 on the network's own call output.  No tape."""
    Cp = call_nn(phi, S0, Tmax, T, np.asarray(K, float) + h)
    C0 = call_nn(phi, S0, Tmax, T, K)
    Cm = call_nn(phi, S0, Tmax, T, np.asarray(K, float) - h)
    return (Cp - 2.0 * C0 + Cm) / h ** 2


def grid():
    """The common comparison grid, wide enough that every maturity keeps >= 99% of its mass."""
    return np.linspace(K_LO, K_HI, N_K)


def density(phi, S0, Tmax, T, K, clip=False):
    """f(K) = e^{rT} d2C_NN/dK2.  Unclipped by default: the negative excursions are the
    diagnostic, and clipping them silently inflates the mass."""
    f = np.exp(FT.R * float(T)) * d2_autodiff(phi, S0, Tmax, T, K)
    return np.maximum(f, 0.0) if clip else f


def mass_table(phi, S0, Tmax, taus, K=None):
    """Retained mass per maturity on K -- the check the truncation lesson demands.

    Reports the raw trapezoid integral (which is what "retained mass" means), the mass after
    clipping negatives, the negative mass itself, and the mean against the forward S0 e^{rT}:
    a wide grid that keeps the mass should also reproduce the forward, since both follow from
    the same integration by parts.
    """
    K = grid() if K is None else K
    rows = []
    for T in taus:
        f = density(phi, S0, Tmax, T, K)
        m_raw = float(_trapz(f, K))
        fc = np.maximum(f, 0.0)
        m_clip = float(_trapz(fc, K))
        fn = fc / m_clip
        mean = float(_trapz(K * fn, K))
        sd = float(np.sqrt(_trapz((K - mean) ** 2 * fn, K)))
        fwd = S0 * np.exp(FT.R * float(T))
        rows.append(dict(tau=float(T), mass=m_raw, mass_clip=m_clip,
                         neg_mass=m_raw - m_clip,
                         neg_frac=float(np.mean(f < 0)), min_f=float(f.min()),
                         mean=mean, fwd=fwd, fwd_err_pct=100.0 * (mean - fwd) / fwd, sd=sd))
    return rows


def print_mass_table(rows, label="retained mass on the comparison grid"):
    print(f"{label}   (K in [{K_LO:.0f}, {K_HI:.0f}], {N_K} points)")
    print(f"{'tau':>7}{'mass':>9}{'mass>=0':>9}{'neg mass':>10}{'neg pts':>9}"
          f"{'min f':>11}{'mean':>10}{'forward':>10}{'mean-fwd%':>11}{'sd':>9}")
    for r in rows:
        print(f"{r['tau']:7.3f}{r['mass']:9.4f}{r['mass_clip']:9.4f}{r['neg_mass']:+10.4f}"
              f"{100*r['neg_frac']:8.2f}%{r['min_f']:+11.2e}{r['mean']:10.2f}{r['fwd']:10.2f}"
              f"{r['fwd_err_pct']:+11.3f}{r['sd']:9.2f}")
    worst = min(r["mass"] for r in rows)
    print(f"  -> worst retained mass across maturities: {worst:.4f} "
          f"({'OK, >= 0.99' if worst >= 0.99 else 'TOO NARROW -- widen the grid'})")


def quantile_sample(phi, S0, Tmax, T, n=20000, K=None):
    """A DETERMINISTIC sample of the analytic density, for the KDE / normal-quantile panels.

    The analytic density is a curve, not a sample, but kde_fixed() and nq() need samples.
    Inverting the analytic CDF at the Blom positions (i-0.375)/(n+0.25) -- the same positions
    nq() uses -- gives the exact quantile function rather than a Monte Carlo draw, so the
    normal-quantile panel carries NO sampling noise and the KDE panel carries only the smoothing
    that is applied identically to every curve.

    Cross-checked against exact_moments(), which integrates the density directly: in LEVEL space
    the two agree to 3 decimals at the longer maturities and to ~0.02 of skew at the shortest,
    where the density is sharply peaked and the CDF inversion is interpolating a steeper curve.
    In LOG space they do NOT agree, and that disagreement is itself the diagnostic -- see the
    note in exact_moments() and in fig_discounted_nicolas_vs_nn.py.
    """
    K = grid() if K is None else K
    f = np.maximum(density(phi, S0, Tmax, T, K), 0.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(K))])
    cdf /= cdf[-1]
    u = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    return np.interp(u, cdf, K)


def exact_moments(phi, S0, Tmax, T, K=None, space="level"):
    """(mean, sd, skew, excess kurtosis) of the analytic density by direct integration.

    space="log" integrates in x = log K under the change of variables f_X(x) = K f_K(K), which is
    the cross-check that matters for the log panels: if the integrated and sampled log-moments
    disagree, the log-space statistic is being set by a handful of extreme far-left quantiles
    rather than by the body of the density.  In LEVEL space the far-left tail is harmless (it sits
    a few s.d. out); in LOG space K -> 0 is at -infinity, so the same grid that is comfortably wide
    for level moments can be tail-dominated for log ones.  This is the mirror image of the
    truncation problem: too NARROW loses mass, too WIDE imports log outliers.
    """
    K = grid() if K is None else K
    f = np.maximum(density(phi, S0, Tmax, T, K), 0.0)
    if space == "log":
        x = np.log(K)
        w = f * K                       # f_X(x) dx = f_K(K) dK
    elif space == "level":
        x, w = K, f
    else:
        raise ValueError(f"space must be 'level' or 'log', got {space!r}")
    wn = w / _trapz(w, x)
    m = float(_trapz(x * wn, x))
    sd = float(np.sqrt(_trapz((x - m) ** 2 * wn, x)))
    z = (x - m) / sd
    return m, sd, float(_trapz(z ** 3 * wn, x)), float(_trapz(z ** 4 * wn, x) - 3.0)


def h_sweep(phi, S0, Tmax, taus, hs=None, K=None):
    """Central-difference step-size sweep: the U-curve that picks H_FD.

    The error is normalised by max|d2C/dK2| at that maturity, NOT pointwise: the second
    derivative crosses zero in the tail, so a pointwise relative error is unbounded there and
    would select h by where the zero-crossing happens to sit.

    The two branches are the whole justification for the chosen h.  Too small and the float32
    call output dominates: the network's last layer is dtype float32, so C carries an absolute
    noise of order eps*|C| ~ 1e-7 * S0, and dividing it by h^2 gives an error growing as h^-2.
    Too large and the truncation term h^2 C''''/12 dominates, growing as h^+2.  The minimum sits
    where the two cross.  Fitted slopes are printed so the branches can be checked, not asserted.
    """
    hs = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0] if hs is None else hs
    K = grid() if K is None else K
    an = {T: d2_autodiff(phi, S0, Tmax, T, K) for T in taus}
    scale = {T: float(np.max(np.abs(an[T]))) for T in taus}
    per_tau = {T: [] for T in taus}
    overall = []
    for h in hs:
        allr = []
        for T in taus:
            r = np.abs(d2_fd(phi, S0, Tmax, T, K, h) - an[T]) / scale[T]
            per_tau[T].append(float(r.mean()))
            allr.append(r)
        allr = np.concatenate(allr)
        overall.append((float(allr.mean()), float(allr.max())))
    return hs, per_tau, overall


def wrap_math(text, width=185):
    """Wrap a suptitle at spaces that lie OUTSIDE $...$ spans.

    Needed because these captions are assembled from derived numbers, so their length is not known
    when the figure is written, and a caption line longer than the axes makes bbox_inches="tight"
    blow the canvas out sideways.  A naive textwrap would break inside a mathtext span such as
    $dS_t = rS_t dt$ and matplotlib would then fail to parse it, so tokens are grouped into atoms
    that each close their own dollar signs before any break is considered.
    """
    lines = []
    for para in text.split("\n"):
        atoms, cur, in_math = [], "", False
        for tok in para.split(" "):
            in_math ^= (tok.count("$") % 2 == 1)
            cur = tok if not cur else cur + " " + tok
            if not in_math:
                atoms.append(cur)
                cur = ""
        if cur:
            atoms.append(cur)
        line = ""
        for at in atoms:
            if line and len(line) + 1 + len(at) > width:
                lines.append(line)
                line = at
            else:
                line = at if not line else line + " " + at
        lines.append(line)
    return "\n".join(lines)


def suptitle_fit(fig, text, width=250, fontsize=10, pad_in=0.38, line_factor=1.55):
    """Wrap `text`, attach it as the suptitle, and return the tight_layout rect top to use.

    These captions are built from derived numbers, so neither their length nor the number of lines
    they wrap to is known when the figure is written.  Guessing a rect top then means the caption
    either floats in white space or lands on top of the panel titles.  Measuring the wrapped line
    count and converting it to a figure fraction removes the guess.
    """
    wrapped = wrap_math(text, width=width)
    n_lines = wrapped.count("\n") + 1
    line_in = line_factor * fontsize / 72.0
    need_in = n_lines * line_in + 2 * pad_in
    top = 1.0 - need_in / fig.get_figheight()
    fig.suptitle(wrapped, fontsize=fontsize, y=1.0 - pad_in / fig.get_figheight(), va="top")
    return (0.0, 0.0, 1.0, max(top, 0.55))


def show_layout(rect, label=""):
    """Print the derived suptitle rect, so a collision is visible in the log, not only the PNG."""
    print(f"  layout{' ' + label if label else ''}: tight_layout rect top = {rect[3]:.3f}")


def fit_slope(hs, errs, lo, hi):
    """log-log slope of the error over the h-window [lo, hi] -- expect ~-2 then ~+2."""
    hs, errs = np.asarray(hs, float), np.asarray(errs, float)
    m = (hs >= lo) & (hs <= hi)
    return float(np.polyfit(np.log(hs[m]), np.log(errs[m]), 1)[0])


if __name__ == "__main__":
    phi, eta, S0, Tmax, taus = load()
    print(f"nbexpk7  S0={S0}  Tmax={Tmax} (derived)  r={FT.R}  K_max={FT.KMAX}")
    print(f"quoted maturities: {taus}\n")
    print_mass_table(mass_table(phi, S0, Tmax, taus))
