# From self-consistent calibration to a forward-looking density

*A pedagogical derivation, starting from Wang, Shaa, Privault & Guet, "Deep
self-consistent learning of local volatility," Journal of Computational
Finance 29(2), 1–25 (2025) — hereafter WSPG25 — and continuing to the exp_k
boundary ansatz and the forward-looking-density result in
`presentation/journal_forward_density.pdf`.*

## 0. Roadmap

WSPG25 calibrates a European call price $\pi(K,T)$ and a squared local
volatility $\eta(K,T)$ jointly, using two neural networks trained so that the
pair is *self-consistent*: the price solves Dupire's PDE with that local
volatility. The paper's own call-price ansatz, equation (3.9), turns out to
satisfy one of the two boundary conditions of the Dupire problem exactly and
the other only asymptotically, through training. That asymptotic boundary is
the deep-in-the-money one, $K=0$ — and it happens to be the one boundary at
which the exact option price, its exact slope, and the exact mass of the
recovered risk-neutral density can all be written down in closed form,
without any calibration at all. Sections 1–2 recall the paper's setup and
make this gap precise. Section 3 derives a replacement ansatz (`exp_k`) that
hits the closed-form value *and* slope by construction. Sections 4–5 show
that this is not a cosmetic fix: the closed-form slope is exactly what makes
Breeden–Litzenberger applied to the calibrated surface integrate to a unit
mass. Section 6 explains why this whole exercise — differentiating a
*calibrated, smooth* surface rather than the raw quotes — is the same idea
the paper already used for the volatility, one derivative order higher.
Section 7 derives what each of the three validation checks in
`journal_forward_density.pdf` actually establishes, and what it does not.

Throughout, equation numbers in parentheses like (3.9) refer to WSPG25
itself; equations introduced here are numbered (D.1), (D.2), … .

## 1. WSPG25's framework, recalled

**The model.** The underlying asset follows, under the pricing measure,

$$\mathrm dS_t = rS_t\,\mathrm dt + \sigma(S_t,t)S_t\,\mathrm dB_t,\qquad S_{t=0}=S_0, \tag{1.1}$$

with $\sigma(\cdot,\cdot)\ge0$ the (unknown) local volatility. Call and put
prices are

$$\pi^{\mathrm c}(K,T)=\mathrm e^{-rT}\mathbb E\big[(S_T-K)^+\big],\qquad
\pi^{\mathrm p}(K,T)=\mathrm e^{-rT}\mathbb E\big[(K-S_T)^+\big]. \tag{2.1}$$

**Dupire's PDE.** Differentiating (2.1) under the expectation gives an
initial–boundary value problem for $\pi^{\mathrm c}$ alone — Dupire's
equation,

$$\frac{\partial\pi}{\partial T}(K,T)+rK\frac{\partial\pi}{\partial K}(K,T)-\tfrac12K^2\sigma^2(K,T)\frac{\partial^2\pi}{\partial K^2}(K,T)=0, \tag{2.2}$$

subject to $\pi^{\mathrm c}(K,0)=(S_0-K)^+$, $\pi^{\mathrm c}(\infty,T)=0$ (and
the mirror conditions for puts), and the model-free bound $0\le
\pi^{\mathrm c}(K,T)\le S_0$ from (2.1) alone. WSPG25's whole point is that
*inverting* (2.2) for $\sigma$ by finite-differencing noisy, unevenly-spaced
quotes is numerically unstable — so instead of ever forming (1.2)'s ratio of
finite differences, they **parameterize both unknowns as neural networks and
train them together.**

**Rescaling.** So that no term dominates during optimization, WSPG25 changes
variables to

$$k=\frac{\mathrm e^{-rT}K}{K_{\max}},\qquad t=\frac{T}{T_{\max}},\qquad
\eta(k,t):=\tfrac12T_{\max}\sigma^2(K,T), \tag{3.1}$$

with $K_{\max}=\max(K)$, $T_{\max}=\max(T)$ over the quoted data. The rescaled
Dupire equation is

$$f_{\mathrm{dup}}(k,t):=\frac{\partial\pi}{\partial t}(k,t)-\eta(k,t)k^2\frac{\partial^2\pi}{\partial k^2}(k,t)=0, \tag{3.2}$$

with $\pi^{\mathrm c}(k,0)=(S_0-K_{\max}k)^+$ and $\pi^{\mathrm c}(1,t)=0$ — the
point $k=1$ (the largest quoted strike, discounted) standing in for the true
$K\to\infty$ boundary. **This substitution is worth flagging early, because
it will matter in Section 5:** $k=0$ is the *exact* image of $K=0$ under
(3.1), for every maturity, with no approximation involved; $k=1$ is a
*practical* stand-in for a limit that is mathematically at $K=\infty$. The
two boundaries are not on the same footing, and the derivation below only
ever needs the one that is exact.

**No-arbitrage, combined.** WSPG25 shows (Lemma 3.1) that calendar-spread
arbitrage ($\partial\pi/\partial T\ge0$) and butterfly arbitrage
($\partial^2\pi/\partial K^2\ge0$) are *simultaneously* excluded by the single
inequality

$$f_{\mathrm{arb}}(k,t):=\frac{\partial\pi}{\partial t}(k,t)-rT_{\max}k\Big(\frac{\partial\pi}{\partial k}(k,t)\Big)^{\!+}\ge0, \tag{3.7}$$

by a two-case argument (if $\partial\pi/\partial k\ge0$, (3.7) *is* the
calendar condition and (3.2) upgrades it to the butterfly condition for free;
if $\partial\pi/\partial k<0$, (3.7) reduces to the calendar condition
directly). This is a genuinely economical trick — one soft constraint doing
the work of two — and it is one of the two mechanisms this document leaves
completely untouched.

**The ansatz.** WSPG25 does not fit $\pi$ directly; it fits an exponential
form whose activation structure builds in *some* constraints for free:

$$\pi^{\mathrm c}_\theta(k,t)=S_0\Big(1-\mathrm e^{-(1-k)\mathcal N_{\mathrm c}(k,t;\theta_{\mathrm c})}\Big), \tag{3.9}$$

$$\pi^{\mathrm p}_\theta(k,t)=K_{\max}\mathrm e^{rT_{\max}t}\,k\Big(1-\mathrm e^{-k\mathcal N_{\mathrm p}(k,t;\theta_{\mathrm p})}\Big), \tag{3.10}$$

with $\mathcal N_{\mathrm c},\mathcal N_{\mathrm p}\ge0$ enforced by a
softplus output layer. The squared local volatility is a second network,
$\eta_\theta(k,t)=\mathcal N_\eta(k,t;\theta_\eta)\ge0$, also softplus-output.

**Training.** Four loss terms — data fit $L_{\mathrm{fit}}$, the $T=0$
initial condition $L_{\mathrm{ini}}$, the arbitrage penalty $L_{\mathrm{arb}}$
built from (3.7), and the PDE residual $L_{\mathrm{dup}}$ built from (3.2) —
are combined as $L_\pi=L_{\mathrm{fit}}+\lambda_{\mathrm{ini}}L_{\mathrm{ini}}+\lambda_{\mathrm{arb}}L_{\mathrm{arb}}+\lambda_{\mathrm{dup}}L_{\mathrm{dup}}$
(3.13) and minimized alternately over $\theta_{\mathrm c}$ (or
$\theta_{\mathrm p}$) and $\theta_\eta$. *Self-consistency* is exactly the
statement that $L_{\mathrm{dup}}\to0$: the calibrated price and the
calibrated local volatility end up describing the same PDE. Section 4.2.1 of
WSPG25 applies this to daily DAX call quotes for 7, 8 and 9 August 2001 — the
same three days this derivation, and the deck it supports, uses throughout.

Everything above is WSPG25. Nothing in it mentions the risk-neutral density,
and nothing in it treats the two halves of the $k=0$/$k=1$ boundary pair
differently. Both of those are where this document starts.

## 2. Two exact facts about $K=0$ that the ansatz doesn't use

**Fact 1 — the exact value.** Since $(S_T-0)^+=S_T$ always,

$$\pi^{\mathrm c}(0,T)=\mathrm e^{-rT}\mathbb E[S_T]. \tag{D.1}$$

Under (1.1), $\mathrm e^{-rt}S_t$ is a martingale, so
$\mathbb E[S_T]=S_0\mathrm e^{rT}$ exactly, and

$$\boxed{\pi^{\mathrm c}(0,T)=S_0\quad\text{for every }T.} \tag{D.2}$$

This is not a modeling choice or a boundary condition imposed for
convenience — it is a restatement of the martingale property already built
into (1.1). It holds for *any* local volatility $\sigma$ whatsoever.

**Fact 2 — the exact slope.** Differentiating the payoff pathwise,
$\partial_K(S_T-K)^+=-\mathbf 1_{\{S_T>K\}}$ for a.e. $K$, so

$$-\frac{\partial\pi^{\mathrm c}}{\partial K}(K,T)=\mathrm e^{-rT}\,\mathbb P(S_T>K). \tag{D.3}$$

This is the first-derivative half of Breeden–Litzenberger (Section 4 derives
the second-derivative half properly). At $K=0$, and using that a local-vol
diffusion started at $S_0>0$ stays strictly positive,
$\mathbb P(S_T>0)=1$, so

$$\boxed{\frac{\partial\pi^{\mathrm c}}{\partial K}(0,T)=-\mathrm e^{-rT}\quad\text{for every }T.} \tag{D.4}$$

Converting to the rescaled coordinate via (3.1), $K=K_{\max}\mathrm e^{rT}k$,
so $\partial\pi/\partial k=\partial\pi/\partial K\cdot K_{\max}\mathrm
e^{rT}$. Writing $\tilde\varphi(k,t):=\pi_\theta^{\mathrm c}(k,t)/S_0$ for the
*normalized* price surface,

$$\boxed{\frac{\partial\tilde\varphi}{\partial k}(0,t)=-\frac{K_{\max}}{S_0}\quad\text{for every }t.} \tag{D.5}$$

Both (D.2) and (D.5) are exact identities, independent of $t$, independent of
$\sigma$, and known *before any training happens* — they follow from the SDE
alone. A calibration that has to *learn* them is spending capacity on
something that was never in doubt.

**What the published ansatz (3.9) actually does at $k=0$.** Evaluate (3.9) at
$k=0$:

$$\pi^{\mathrm c}_\theta(0,t)=S_0\big(1-\mathrm e^{-\mathcal N_{\mathrm c}(0,t)}\big). \tag{D.6}$$

This equals $S_0$ *only in the limit* $\mathcal N_{\mathrm c}(0,t)\to\infty$.
Nothing in the loss $L_\pi$ forces that limit — $L_{\mathrm{fit}}$ pulls
$\mathcal N_{\mathrm c}$ toward whatever value reprices the nearby quotes, and
$L_{\mathrm{ini}}$/$L_{\mathrm{dup}}$ act through collocation points sampled
uniformly on $[0,1]\times[0,1]$, not concentrated at the single measure-zero
point $k=0$. Training therefore converges to some large but *finite*
$\mathcal N_{\mathrm c}(0,t)$, leaving a residual multiplicative deficit

$$S_0-\pi^{\mathrm c}_\theta(0,t)=S_0\,\mathrm e^{-\mathcal N_{\mathrm c}(0,t)}>0, \tag{D.7}$$

observed empirically as a 7–13% shortfall on high-volatility synthetic data
(higher $\sigma$ makes the price surface steeper near $k=0$, so a larger
$\mathcal N_{\mathrm c}$ is needed to close the same relative gap, and
training converges to it less completely). And (3.9) says nothing at all
about the *slope* at $k=0$ — (D.5) is simply not a target the ansatz can be
pushed toward, exactly or otherwise, because differentiating (D.6) only
constrains $\partial_k\mathcal N_{\mathrm c}(0,t)$, not the value $S_0e^{-\mathcal N_{\mathrm
c}(0,t)}$ that (D.5) actually depends on being small. Compare this with the
*other* boundary: at $k=1$, (3.9) gives $\pi_\theta^{\mathrm c}(1,t)=S_0(1-\mathrm
e^0)=0$ **exactly**, for any $\mathcal N_{\mathrm c}$ whatsoever — the
$(1-k)$ factor was placed exactly so that this boundary needs no training.
The asymmetry is the whole story: one boundary was built in, the other was
left to be learned, and it happens to be the one with a closed form.

## 3. The `exp_k` ansatz, derived from (D.2) and (D.5)

We want a replacement $\tilde\varphi(k,t)$ that satisfies (D.2) and (D.5)
*for every* value the network can take — i.e. structurally, not
asymptotically — while keeping the same two properties (3.9) already gave
us for free: an output that is automatically in $(0,1]$ (so
$\pi_\theta^{\mathrm c}\in(0,S_0]$, matching (2.3)), built from a single
unconstrained network output so it stays as expressive as before.

**Step 1 — pick a form with a free constant to absorb the slope.** Try

$$\tilde\varphi(k,t)=\exp\!\big(-k\cdot g(k,t)\big),\qquad g(k,t):=\mathrm{softplus}\big(a+kH(k,t;\theta)\big), \tag{D.8}$$

where $H$ is now the network's **raw, linearly-activated** output (no
softplus on $H$ itself — the softplus is applied explicitly by the ansatz,
once, to the combination $a+kH$), and $a$ is a constant to be fixed in Step
3. Since $\mathrm{softplus}>0$ always and $k\ge0$, the exponent $-kg(k,t)\le
0$, so $\tilde\varphi\in(0,1]$ automatically, for *any* $H$ — the same
boundedness (3.9) had, now from a single multiplicative structure instead of
a difference from 1.

**Step 2 — check the value at $k=0$.** $\tilde\varphi(0,t)=\exp(0)=1$,
*for every $H$ and every $a$*. (D.2) is satisfied structurally: there is
nothing left to learn here.

**Step 3 — match the slope at $k=0$.** Differentiate (D.8) using the product
rule on $k\,g(k,t)$:

$$\frac{\partial\tilde\varphi}{\partial k}(k,t)=-\tilde\varphi(k,t)\Big[g(k,t)+k\frac{\partial g}{\partial k}(k,t)\Big].$$

At $k=0$: $\tilde\varphi(0,t)=1$ and the bracket reduces to $g(0,t)=\mathrm{softplus}(a)$
(the $k\,\partial_kg$ term vanishes because of the explicit factor $k$), so

$$\frac{\partial\tilde\varphi}{\partial k}(0,t)=-\mathrm{softplus}(a). \tag{D.9}$$

Setting (D.9) equal to the required value (D.5),
$\mathrm{softplus}(a)=K_{\max}/S_0$, and inverting
$\mathrm{softplus}(x)=\log(1+\mathrm e^x)$ gives $\mathrm e^x=\mathrm e^{\mathrm{softplus}(x)}-1$, i.e.

$$\boxed{a=\log\!\Big(\mathrm e^{K_{\max}/S_0}-1\Big).} \tag{D.10}$$

This is a closed-form constant, fixed once from $(K_{\max},S_0)$ — not a
parameter that is trained, and not something the network ever has to
discover. With $a$ fixed this way, **both** (D.2) and (D.5) hold at $k=0$ for
*every* value the free network $H$ can take, at initialization and
throughout training. The network's only remaining job is to shape
$\tilde\varphi$ away from $k=0$ — where nothing is known in closed form, and
where fitting the data is genuinely necessary.

**What this does not fix.** The $k=1$ practical-infinity boundary is left
exactly as it was — a soft target enforced through $L_{\mathrm{ini}}$/$L_{\mathrm{dup}}$
collocation, not a structural one — because that boundary was never exact in
the first place (Section 1's remark on (3.1)); there is no closed form at
$k=1$ to build in. Nor does (D.8) guarantee $\tilde\varphi$ is monotone
decreasing in $k$ — differentiating again shows the sign of
$\partial^2\tilde\varphi/\partial k^2$ still depends on $H$'s own
$k$-dependence, exactly as it did for (3.9). Monotonicity (a "vertical
spread" no-arbitrage condition, distinct from the calendar and butterfly
conditions WSPG25 names explicitly) is not structurally enforced by either
ansatz; it is left to emerge from fitting genuinely monotone market data.
This derivation only ever claimed to fix the one boundary that has a closed
form — and, as Section 5 shows, that turns out to be exactly the boundary
that matters for the next step.

## 4. From a calibrated price surface to a density

WSPG25 never needs the risk-neutral density — Dupire's formula (1.2) only
calls for $\sigma$. But once $\pi_\theta$ is a *smooth, differentiable*
surface (the entire point of using a neural ansatz rather than raw quotes),
recovering the density is one more differentiation away, by
**Breeden–Litzenberger.**

**Derivation.** Let $f_T$ be the risk-neutral density of $S_T$. Writing (2.1)
as an integral against $f_T$,

$$\pi^{\mathrm c}(K,T)=\mathrm e^{-rT}\int_K^\infty(x-K)f_T(x)\,\mathrm dx,$$

and differentiating twice under the integral sign (Leibniz's rule),

$$\frac{\partial\pi^{\mathrm c}}{\partial K}(K,T)=-\mathrm e^{-rT}\int_K^\infty f_T(x)\,\mathrm dx=-\mathrm e^{-rT}\big(1-F_T(K)\big) \tag{D.11}$$

— matching (D.3) exactly, as it must — and differentiating once more,

$$\boxed{f_T(K)=\mathrm e^{rT}\frac{\partial^2\pi^{\mathrm c}}{\partial K^2}(K,T).} \tag{D.12}$$

**In rescaled coordinates.** With $\tilde\varphi=\pi_\theta^{\mathrm
c}/S_0$ and $K=K_{\max}\mathrm e^{rT}k$, the chain rule applied twice gives

$$\frac{\partial^2\pi_\theta^{\mathrm c}}{\partial K^2}=S_0\Big(\frac{\mathrm e^{-rT}}{K_{\max}}\Big)^{\!2}\frac{\partial^2\tilde\varphi}{\partial k^2},$$

so

$$\boxed{f_T(K)=\frac{S_0\,\mathrm e^{-rT}}{K_{\max}^2}\,\frac{\partial^2\tilde\varphi}{\partial k^2}(k,t),\qquad K=K_{\max}\mathrm e^{rT}k,\ \ T=T_{\max}t.} \tag{D.13}$$

**Why this needs the network, not the quotes.** $\tilde\varphi(k,t)$ is
differentiable *everywhere* on $[0,1]\times[0,1]$, so (D.13) can be evaluated
at any $(K,T)$ at all — including maturities and strikes that were never
quoted. This is the precise mathematical content of "one calibration $\to$ a
density at every horizon": the calibration step already had to produce a
globally differentiable $\tilde\varphi$ for $L_{\mathrm{dup}}$ to make sense
(Dupire's equation needs $\partial^2\pi/\partial k^2$ pointwise); (D.13) says
that same object, differentiated once more, *is* the forward-looking
density, off the shelf.

## 5. Unit mass is a corollary of the slope fix, not a separate check

A density must integrate to 1. Using (D.12),

$$\int_0^\infty f_T(K)\,\mathrm dK=\mathrm e^{rT}\Big[\frac{\partial\pi^{\mathrm c}}{\partial K}(K,T)\Big]_{K=0}^{K=\infty}
=\mathrm e^{rT}\Big(\lim_{K\to\infty}\frac{\partial\pi^{\mathrm c}}{\partial K}(K,T)-\frac{\partial\pi^{\mathrm c}}{\partial K}(0,T)\Big). \tag{D.14}$$

The far-tail limit vanishes because (D.3) shows
$\partial\pi^{\mathrm c}/\partial K=-\mathrm e^{-rT}\mathbb P(S_T>K)\to0$ as
$K\to\infty$ (assuming $\mathbb E[S_T]<\infty$, which the martingale property
already gives). Substituting (D.4) for the remaining term,

$$\int_0^\infty f_T(K)\,\mathrm dK=\mathrm e^{rT}\big(0-(-\mathrm e^{-rT})\big)=1. \tag{D.15}$$

**The point of this computation is what it depends on.** Unit mass follows
from exactly two ingredients: the far-tail slope vanishing, and the $K=0$
slope equaling $-\mathrm e^{-rT}$. The second of these is precisely (D.5) —
the quantity the `exp_k` ansatz now matches by construction (Section 3). A
network trained under the *old* ansatz can only approach the right value at
$K=0$ (D.7); it has no mechanism at all for approaching the right *slope*
there, so the mass computed from (D.13) under the old ansatz has no reason
to be close to 1 — the deficit in (D.7) shows up not as "slightly less
density everywhere" but specifically as missing slope, hence missing mass,
concentrated at the deep-in-the-money end.

The far-tail term in (D.14), by contrast, is genuinely only *approximate* in
this framework: $k=1$ is a finite cutoff standing in for $K\to\infty$
(Section 1), so in practice $\int_0^{K_{\max}}f_T\,\mathrm dK$ falls slightly
short of 1 by whatever tail mass lies beyond the largest quoted strike. This
is exactly what the retained-mass check in `journal_forward_density.pdf`
(Figure 4b) is measuring — worst case $0.9995$ over $K\in[100,15000]$ — and
this derivation shows precisely what that $0.05\%$ residual *is*: not a
symptom of the boundary fix being incomplete, but the honest, separately
-bounded cost of truncating an infinite integral at a finite $K_{\max}$.

## 6. Why not just difference the quotes — the same lesson, one order higher

WSPG25 opens by rejecting exactly this move for $\sigma$: applying (1.2)'s
finite differences directly to sparse, unevenly-spaced market quotes is
unstable, because dividing a noisy numerator by a small, noisy $\partial^2\pi/\partial K^2$
amplifies error. The identical failure mode reappears one derivative order
higher if (D.12) is applied to the *raw quoted prices* instead of a
calibrated surface: with only $N\sim200$ quotes per maturity, the three-point
second difference on the strike ladder is dominated by rounding in the
quoted prices (to the nearest 0.01 index point) and by the ladder's uneven
spacing, and it goes **negative** at a meaningful fraction of interior
strikes — 6–25% per maturity, empirically, in
`journal_forward_density.pdf`'s Figure 2 — which is not merely inaccurate,
it is not a density at all. This is why a smooth, self-consistently
calibrated $\tilde\varphi$ has to come *first*: (D.12)–(D.13) are only
usable once differentiability is bought by calibration rather than assumed
of the raw data, which is the same reason WSPG25 built $\pi_\theta$ and
$\eta_\theta$ as networks in the first place. The density-recovery step adds
nothing conceptually new to the paper's own argument — it just applies it to
$\partial^2/\partial K^2$ instead of stopping at $\partial/\partial K$ and
$\partial/\partial T$.

## 7. Validating a recovered density that no quoted price can check directly

Nothing in the market gives a ground truth for $f_T(K)$ directly — there is
no quoted "density." Three independent checks are used instead, each
targeting a different possible point of failure, and it is worth deriving
precisely what each one can and cannot rule out.

### 7.1 Is the second derivative extracted correctly?

This check compares two ways of computing $\partial^2\tilde\varphi/\partial k^2$
**from the same trained surface** — automatic differentiation (exact, up to
floating-point error, for the function the network actually represents) against a
three-point central difference $D_h^2\tilde\varphi(k):=\big(\tilde\varphi(k+h)-2\tilde\varphi(k)+\tilde\varphi(k-h)\big)/h^2$
on the network's own call output. Since both routes read the *same trained
weights*, agreement or disagreement here says nothing about calibration
quality — it only tests whether the second derivative was extracted
correctly from a surface that is already fixed.

Taylor-expanding $\tilde\varphi(k\pm h)$ to fourth order and combining gives
the standard result

$$D_h^2\tilde\varphi(k)=\frac{\partial^2\tilde\varphi}{\partial k^2}(k)+\frac{h^2}{12}\frac{\partial^4\tilde\varphi}{\partial k^4}(k)+O(h^4), \tag{D.16}$$

a **truncation** error growing as $O(h^2)$. Separately, each evaluation of
$\tilde\varphi$ in float32 carries an absolute error of order
$\varepsilon_{32}|\tilde\varphi|$ (machine epsilon times the function scale);
propagated through the three evaluations in $D_h^2$ and divided by $h^2$,
this gives a **roundoff** error of order

$$\frac{4\varepsilon_{32}|\tilde\varphi|}{h^2}, \tag{D.17}$$

*shrinking* as $O(h^{-2})$. On a log–log plot of error against $h$, (D.16)
is a line of slope $+2$ and (D.17) a line of slope $-2$; the two must cross,
and the crossing is the best achievable step size,

$$h^\star=\Big(\frac{48\,\varepsilon_{32}|\tilde\varphi|}{|\partial^4\tilde\varphi/\partial k^4|}\Big)^{1/4}, \tag{D.18}$$

obtained by minimizing the sum of (D.16) and (D.17). This is exactly the
U-shaped curve and the fitted exponents ($-1.97$ against a predicted $-2$;
$+1.90$ against a predicted $+2$) reported for the step-size sweep in
`journal_forward_density.pdf` — the empirical fit is a confirmation of
(D.16)–(D.18), not an independent phenomenon. Away from the money the
residual sits at the roundoff floor (D.17) itself, with no headroom left to
improve; only at the money does the true fourth derivative
$\partial^4\tilde\varphi/\partial k^4$ (sharply peaked, since $\tilde\varphi$
is close to a smoothed step there) push the disagreement visibly above that
floor — which is a statement about $\tilde\varphi$'s curvature, not a defect
in either differentiation route.

### 7.2 Does the calibration actually solve its own PDE?

Self-consistency training minimizes $L_{\mathrm{dup}}$, but only
approximately — it is a soft penalty, not an exact constraint, so
$f_{\mathrm{dup}}(k,t)$ in (3.2) is small but not identically zero after
training. There are two ways to compute "the density at maturity $T$" from a
converged $(\pi_\theta,\eta_\theta)$ pair, and if $f_{\mathrm{dup}}\equiv0$
held exactly, they would coincide identically:

- **Direct:** apply (D.12)–(D.13) to $\pi_\theta$ itself.
- **Simulated:** integrate (1.1) forward from $S_0$ using $\sigma_\theta(x,t)=\sqrt{2\eta_\theta(x,t)/T_{\max}}$
  — the paper's own reprice recipe, equation (4.2) — and read off the
  law of the simulated $S_T$ (equivalently, Monte-Carlo reprice the same
  calls and apply (D.12) to *that* reprice).

Both routes start from the calibrated pair and never touch the market again.
Their disagreement is a direct, quantitative readout of how close
$f_{\mathrm{dup}}$ actually got to zero — which is precisely what the
Kolmogorov–Smirnov distance between the two densities in
`journal_forward_density.pdf` (worst case $0.0047$ across five maturities)
is measuring. A large gap here would mean the training's own regularizer,
$L_{\mathrm{dup}}$, had not converged well; it says nothing about whether
the calibration matches the *market*.

### 7.3 Comparing against what the index actually did

This is the most interpretively delicate of the three, and it is worth
being precise about exactly what object is being compared to what.
$f_T(\cdot)$ from (D.12) is the $\mathbb Q$-measure (risk-neutral) marginal
law of $S_T$ at a **single, fixed** maturity $T$. A realized index path
observed at $n$ discrete dates, $\{S_{u_1},\dots,S_{u_n}\}_{u_i\le T}$, has
an empirical distribution that is instead informative about a **time
average of $\mathbb P$-measure marginals**,

$$\frac1n\sum_{i=1}^n\delta_{S_{u_i}}\ \approx\ \frac1T\int_0^T \mathrm{Law}_{\mathbb P}(S_u)\,\mathrm du, \tag{D.19}$$

which coincides with the single-time law $\mathrm{Law}(S_T)$ only if the
process is stationary — false here, since (1.1) has a drift $r$ and a
generally time-inhomogeneous $\sigma(S,t)$ — **and** coincides with the
$\mathbb Q$-law only if there is no risk premium, i.e. $\mathbb P=\mathbb Q$,
which is also generally false for an equity index. Two distinct assumptions
would be needed for this comparison to be a formal goodness-of-fit test, and
neither holds. What survives without either assumption is a much weaker,
purely *qualitative* statement: if the calibrated $\mathbb Q$-density and the
realized path's occupation measure are both, say, left-skewed by a similar
order of magnitude, that is evidence the calibration is not producing an
obviously unreasonable shape — a sanity check, not a hypothesis test. This is
exactly the hedge `journal_forward_density.pdf`'s own caption places on
Figure 5, and (D.19) is the reason it has to be there.

## 8. Summary

Starting only from WSPG25's own model (1.1) and price definition (2.1), two
facts about the $K=0$ boundary — the exact value $\pi^{\mathrm c}(0,T)=S_0$
(D.2) and the exact slope $\partial\pi^{\mathrm c}/\partial K(0,T)=-\mathrm
e^{-rT}$ (D.4) — follow from the martingale property and pathwise
differentiation alone, with no calibration involved. The published ansatz
(3.9) reaches the first only asymptotically and does not target the second
at all. Replacing it with (D.8), and fixing its one free constant to
(D.10), reproduces both exactly, for every value the underlying network can
take. Differentiating the resulting smooth surface twice more,
Breeden–Litzenberger (D.12) turns a single day's calibration into a density
at every horizon (D.13); the slope fix from Section 3 is exactly what makes
that density integrate to 1 (D.15), up to a separately-bounded, honestly
small far-tail truncation. What the resulting object *is* — the market's own
risk-neutral view on that one calibration day, checked for internal
consistency three independent ways — and what it is *not* — a real-world
($\mathbb P$-measure) forecast — follows directly from which of the
identities above needed no data to be true, and which did.
