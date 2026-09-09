# Soft APLS: methodology

Formal specification of a differentiable route-level loss whose kernel is the
one APLS scores. The staging, cost model, measurement protocol and risks live in
[`soft_apls_loss.md`](soft_apls_loss.md); this document is the mathematics.

## 1. Notation

| Symbol | Meaning |
|---|---|
| $I$ | image crop, $H \times W$, resolution $r$ metres/pixel |
| $f_\theta(I) = (\ell, \hat d)$ | network: road logits and a predicted distance field |
| $p = \sigma(\ell)$ | road probability, used by BCE / Dice / clDice / thresholding |
| $\hat d \ge 0$ | predicted distance to the nearest road, in metres |
| $G$ | truth road graph restricted to the crop, edge lengths in metres |
| $M$ | truth mask, $G$ rasterized at width $w$ |
| $d^\star = \min(\mathrm{DT}(M),\, d_{\max})$ | truncated ground-truth distance transform |

Two heads on one encoder. The probability head keeps every existing consumer —
`predict_mask`, `skeleton`, IoU, clDice — unchanged; the distance head exists
only to define the routing field of §3.

## 2. Grid graph

$\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with $\mathcal{V}$ the pixels of the
crop and $\mathcal{E}$ the 8-neighbour edges. Each edge carries its geometric
length

$$\delta_e = \begin{cases} r & \text{axial} \\ r\sqrt{2} & \text{diagonal}\end{cases}$$

## 3. Cost field

$$c_e(\hat d) \;=\; \delta_e\left(1 + \kappa\,\bar d_e^{\,\gamma}\right),
\qquad \bar d_e = \tfrac{1}{2}\!\left(\hat d_i + \hat d_j\right)$$

Three properties fix this form.

**Calibration.** On road $\hat d = 0$, so $c_e = \delta_e$ and route cost is
measured in metres. Without this the cost has no length interpretation and the
APLS ratio of §6 is dimensionless nonsense.

**Multiplicative.** An additive penalty $c_e = \delta_e + \kappa \bar d_e$
charges an axial and a diagonal step alike despite the diagonal covering
$\sqrt2$ times the ground, so the router prefers diagonals through open terrain.

**Distance, not probability.** With a probability field the natural penalty
$(1-p)^\gamma$ saturates: once $p \approx 0$ it is flat, so it carries no
information about *how far* the nearest road is and a 3 m gap costs the same per
metre as a 50 m one — CAPE's stated reason for regressing distance instead. A
distance field rises with separation from the road, so crossing a gap of width
$g$ costs

$$\int_0^{g} \kappa \min(x,\, g-x)^{\gamma}\,dx \;=\;
\begin{cases} \kappa g^2 / 4 & \gamma = 1 \\ \kappa g^3 / 12 & \gamma = 2\end{cases}$$

superlinear either way, and the field has nonzero gradient far from any
foreground — it points back toward road, where $1-p$ is flat.

$\gamma$ is a real sweep axis, not a settled choice. CAPE uses $\gamma = 2$
explicitly, to make severe disconnections dominate mild ones. The argument for
$\gamma = 1$ is that the distance field is already superlinear in gap width at
$\gamma = 1$, so squaring may over-weight the middle of wide gaps relative to
the many narrow ones that actually cost APLS. Measure it.

## 4. Path measure and expected occupancy

For a path $\pi$ from $a$ to $b$ let $y^\pi \in \{0,1\}^{\mathcal{E}}$ be its
edge indicator, so $c(\pi) = \langle c, y^\pi\rangle$. Put a Gibbs measure on
paths at inverse temperature $\beta$,

$$P_\beta(\pi) \;\propto\; \exp\!\left(-\beta\, c(\pi)\right),$$

and define the **expected occupancy**

$$\bar y(a,b) \;=\; \mathbb{E}_{\pi \sim P_\beta}\!\left[y^\pi\right] \;\in\; [0,1]^{\mathcal{E}}.$$

As $\beta \to \infty$, $\bar y$ concentrates on the Dijkstra path; as
$\beta \to 0$ it approaches the random walk, whose associated distance is the
commute cost.

## 5. The penalty decomposition

$c$ is affine in $\kappa$, so route cost splits exactly:

$$D(a,b) \;=\; \langle c, \bar y\rangle
\;=\; \underbrace{\langle \delta, \bar y\rangle}_{\hat L(a,b)}
\;+\; \kappa\,\underbrace{\langle \delta \odot \bar d^{\,\gamma}, \bar y\rangle}_{V(a,b)}$$

- $\hat L(a,b)$ — the **objective**: geometric length of the route, in metres.
- $V(a,b)$ — the **constraint violation**: how far off-road the route strays,
  integrated along it.

$\kappa$ is the penalty parameter of an exterior penalty relaxation of

$$\hat L(a,b) \;=\; \min_{\pi} \sum_{e \in \pi} \delta_e
\quad \text{subject to} \quad \pi \text{ lies on road.}$$

Exterior rather than barrier ($\hat d^{-1}$, $-\log$) because an untrained
network has no feasible path between most pairs, so infeasibility has to be
tolerated rather than excluded.

**$D$ alone is CAPE.** Dropping the $+1$ in §3 sets $\hat L \equiv 0$ and leaves
$D = \kappa V$: a pure feasibility measure, minimized by any on-road route
regardless of length. That is the structural reason accumulated traversal cost
cannot express length agreement, and the one-character fix that recovers it.

**$\hat L$ needs the smoothing.** $\hat L$ depends on $\hat d$ only through
*which* routes are likely. Under a single hard path $\bar y$ is piecewise
constant in $\hat d$, so $\partial \hat L / \partial \hat d = 0$ identically. The
Gibbs average is what makes the length term differentiable at all — it is
load-bearing, not variance reduction.

## 6. Control points and the scored pairs

$$N \;=\; \big\{\text{pixel coordinates of } \texttt{densify}(G,\, s,\, \text{"reference"}).\text{nodes}\big\}$$

with $s = 50$ m, matching `metrics._directional`. Reference lengths
$L(a,b)$ are shortest-path distances in that densified truth graph — constants,
computed once per crop and cached with it.

$$P \;=\; \left\{(a,b) \in N^2 \;:\; a < b,\; L(a,b) \ge \ell_{\min}\right\},
\qquad \ell_{\min} = 10\text{ m}$$

## 7. Soft clip

APLS clips the length ratio at 1. That clip is flat above $x = 1$, which is
where every disconnected pair sits early in training. Writing
$\min(1,x) = 1 - \mathrm{relu}(1-x)$ and softening the ReLU:

$$\psi_\tau(x) \;=\; 1 - \tau \,\mathrm{softplus}\!\left(\frac{1-x}{\tau}\right),
\qquad \lim_{\tau \to 0} \psi_\tau(x) = \min(1, x)$$

## 8. Loss terms

**Length agreement** — APLS's kernel verbatim:

$$\mathcal{L}_{\text{len}} \;=\; \frac{1}{|P|} \sum_{(a,b) \in P}
\psi_\tau\!\left(\frac{\big|\,L(a,b) - \hat L(a,b)\,\big|}{L(a,b)}\right)$$

**Feasibility** — the constraint violation, normalized to a dimensionless
fraction of route length:

$$\mathcal{L}_{\text{feas}} \;=\; \frac{1}{|P|} \sum_{(a,b) \in P} \frac{V(a,b)}{L(a,b)}$$

**Reverse** — invented roads. A literal mirror of APLS's $\text{prop} \to
\text{gt}$ direction makes the reference length itself differentiable and the
objective degenerates, since the model can shrink the loss by shrinking $\hat L$.
Take MALIS's negative pass instead: over a set $Q$ of pairs the truth graph
places far apart relative to their Euclidean separation,

$$\mathcal{L}_{\text{rev}} \;=\; \frac{1}{|Q|} \sum_{(a,b) \in Q}
\frac{\mathrm{relu}\!\big(L(a,b) - D(a,b)\big)}{L(a,b)}$$

**Distance regression** — supervision for the routing field:

$$\mathcal{L}_{\text{dt}} \;=\; \frac{1}{|\Omega|} \sum_{u \in \Omega}
\mathrm{huber}\!\left(\hat d_u - d^\star_u\right)$$

CAPE uses a plain MSE on an untruncated transform. Truncation at $d_{\max}$ is
an addition here, and the reason is the domain: their datasets are retinal
vessels, neurons and brain volumes, where foreground is dense and the transform
never grows large. Overhead imagery over desert has pixels a hundred-plus metres
from any road, and an untruncated MSE is dominated by far-field values whose
exact magnitude is irrelevant — no scored route should be there. Huber over MSE
for the same reason. Set $d_{\max}$ to the widest gap worth bridging, and treat
it as a sweep axis, since it also caps how far the cost field can see.

**Total.**

$$\mathcal{L} \;=\;
\underbrace{\mathcal{L}_{\text{BCE}} + w_{\text{dice}}\mathcal{L}_{\text{dice}}
+ w_{\text{cl}}\mathcal{L}_{\text{clDice}}}_{\text{existing}}
\;+\; w_{\text{dt}}\mathcal{L}_{\text{dt}}
\;+\; w_{\text{len}}\mathcal{L}_{\text{len}}
\;+\; w_{\text{feas}}\mathcal{L}_{\text{feas}}
\;+\; w_{\text{rev}}\mathcal{L}_{\text{rev}}$$

A sum, not the harmonic mean the metric uses: $\mathrm{hmean}$ of two near-zero
terms is numerically vicious early in training. Note the divergence where the
weights are defined.

Neither $\mathcal{L}_{\text{len}}$ nor $\mathcal{L}_{\text{feas}}$ carries
$\kappa$. Once $D$ is decomposed, $\kappa$'s only remaining job is deciding where
routes may go, which is what makes it schedulable rather than a weight to tune.

## 9. Gradients

$$\frac{\partial V}{\partial \hat d_u} \;=\;
\underbrace{\frac{\gamma}{2} \sum_{e \ni u} \delta_e\, \bar d_e^{\,\gamma-1}\, \bar y_e}_{\text{direct — pixels on the route}}
\;+\;
\underbrace{\left\langle \delta \odot \bar d^{\,\gamma},\; \frac{\partial \bar y}{\partial \hat d_u}\right\rangle}_{\text{through the routing}}$$

The first term is CAPE's gradient: Danskin's theorem applied to
$\min_\pi c(\pi;\hat d)$, valid when $\bar y$ is a single hard path. The second
term exists only under smoothing, and it is the only source of gradient for
$\hat L$:

$$\frac{\partial \hat L}{\partial \hat d_u} \;=\;
\left\langle \delta,\; \frac{\partial \bar y}{\partial \hat d_u}\right\rangle$$

## 10. Estimators for $\bar y$

**Hard (Tier 1).** $\bar y = y^{\pi^\star}$, $\pi^\star$ from
`scipy.sparse.csgraph.dijkstra`. Gives $\mathcal{L}_{\text{feas}}$ only;
$\mathcal{L}_{\text{len}}$ is identically flat.

**Perturb-and-MAP (Tier 2).**

$$\bar y \;\approx\; \frac{1}{M} \sum_{m=1}^{M} y^{\pi^\star\!\left(c + \sigma \epsilon^{(m)}\right)},
\qquad \epsilon^{(m)} \overset{\text{iid}}{\sim} \text{Gumbel}$$

$M$ independent Dijkstra runs on perturbed cost maps. Embarrassingly parallel,
no linear algebra, unbiased for the smoothed objective.

**Randomized shortest paths (Tier 3).** With $P^{\text{ref}}$ a reference walk
and $W_{ij} = P^{\text{ref}}_{ij} e^{-\beta c_{ij}}$, let
$Z = (I - W)^{-1}$, which exists iff $\rho(W) < 1$ — a lower bound on $\beta$
that must be checked, not assumed. Then

$$\bar y_{ij}(a,b) \;=\; \frac{Z_{ai}\, W_{ij}\, Z_{jb}}{Z_{ab}}$$

so the exact Gibbs average costs two sweeps per source rather than $M$ samples.

## 11. Consistency with the discrete metric

Set $\hat d = d^\star$, and take $\kappa \to \infty$, $\beta \to \infty$,
$\tau \to 0$. Routes are then confined to the truth mask, $\bar y$ concentrates
on the mask geodesic, $\hat L(a,b)$ becomes the on-mask shortest-path length,
and

$$1 - \mathcal{L}_{\text{len}} \;\longrightarrow\; \mathrm{APLS}_{\text{gt}\to\text{prop}}$$

evaluated on the rasterized truth — that is, up to the rasterization ceiling
already measured at ~0.90, not to 1. This limit is the reference-implementation
check: `metrics.apls` agrees with `CosmiQ/apls` to 3e-5, so it can serve as the
fixed point. Disagreement is a finding for `EXPERIMENT_LOG.md` before either
side is changed.

## 12. Relation to CAPE

CAPE's loss, stated exactly, for a pair of graph vertices $v_1, v_2$:

$$\mathrm{cost}(\pi_{\hat y}) = \sum_{n \in \pi_{\hat y}} \hat y(n)^2,
\qquad
\pi_{\hat y} = \mathrm{Dijkstra}\big(\hat y \cdot M_{\text{dilated}},\, v_1', v_2'\big)$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(y, \hat y) + \alpha\,\mathcal{L}_{\text{CAPE}}$$

with three mechanisms this specification treats differently.

**Vertex projection.** $v_i'$ is the local minimum of $\hat y$ in a $7\times7$
neighbourhood of $v_i$, absorbing small deviations between annotation and
predicted centreline. This is the analogue of APLS's `max_snap` injection, done
on the distance field rather than by nearest-edge search. Adopt it; the harness
already needs the same tolerance for the same reason.

**Corridor masking.** The ground-truth path is rendered, dilated by 10 px, and
the search confined to that corridor, so the router cannot loop around a
disconnection via a parallel structure and hide it. (The paper writes this as
$\hat y \cdot M_{\text{dilated}}$; multiplication by a binary mask would make
outside-corridor pixels free, so read it as restricting the search domain, which
is what the surrounding text says.)

**The masking is a workaround for the missing length term, and $+1$ removes the
need for it.** Because CAPE's cost is zero on any on-road route, a loop-around
through a parallel street costs nothing and the disconnection becomes invisible
— hence the corridor. Under $c_e = \delta_e(1 + \kappa \hat d^{\gamma})$ a
loop-around is *longer*, so it registers as $\hat L > L$, which is precisely the
signal APLS scores. Free routing is therefore the default here.

The interaction runs the other way too, and matters: **corridor masking makes
$\mathcal{L}_{\text{len}}$ nearly vacuous**, since a path pinned to within 10 px
of the truth has $\hat L \approx L$ by construction. Masking and the length term
are alternatives, not complements. If free routing proves unstable, the fallback
is a wide corridor — wide enough to admit the detours worth penalizing — not
CAPE's 10 px.

**Coverage, and why it cannot be copied verbatim.** CAPE randomly picks a
connected pair, routes it on $G$, then deletes $\pi_G$'s edges *from $G$*
("to ensure unique processing"), repeating until $E = \emptyset$. Each edge is
supervised exactly once and the loop terminates in at most $|E|$ steps.

The deletion is destructive, and the paper does not discuss what it costs. After
the first iteration $G$ is fragmented along the path just removed, so later pairs
are drawn from progressively smaller components and the routes degrade from long
natural corridors to short stubs. Supervision quality is therefore strongly
non-uniform across the sweep.

CAPE tolerates this because **it never uses the ground-truth path's length**.
$\pi_G$ serves only to place the corridor mask and to supply the endpoint pair;
the number that enters the loss is computed entirely on $\hat y$. A mutilated
$\pi_G$ still yields a valid corridor and a valid pair.

For $\mathcal{L}_{\text{len}}$ it would be fatal. $L(a,b)$ is the reference the
kernel measures against, and a residual-graph distance is not the road distance —
training against it teaches the network to reproduce routes the network does not
have. Two rules follow:

1. **Separate the graphs.** $L(a,b)$ is always computed on the intact densified
   truth graph; edge removal touches only a bookkeeping copy that decides which
   pairs still need visiting.
2. **Termination must account for $\ell_{\min}$.** Pairs shorter than
   $\ell_{\min}$ are skipped by §6, so $E$ never empties and the loop does not
   halt. Stop when no connected pair in the residual copy exceeds
   $\ell_{\min}$, and record the uncovered edge fraction as a diagnostic.

Because the truth graph for a crop is fixed, the cleanest form is to compute the
path cover **once, offline, and cache it with the crop**. That also removes the
order-dependence: line 3's "pick a pair" is unspecified and the resulting cover
is greedy and order-sensitive, which is not reproducible across epochs unless
seeded. Caching makes it a property of the data rather than of the training run,
which is what the frozen-run-config rule wants.

The alternative worth measuring against it is coverage-weighted sampling — keep
$G$ intact and sample pairs with probability falling in how often each edge has
already been covered. Softer, no fragmentation, no exactly-once guarantee. The
backlog already notes that per-tile APLS rests on pair counts spanning 91 to
6328, so how $P$ is sampled is a live question in its own right.

## 13. Parameters

| Symbol | Code name | Role | Start |
|---|---|---|---|
| $\kappa$ | `kappa` | penalty parameter; routing only, annealed | $1 \to 30$ |
| $\gamma$ | `gam` | penalty curvature | sweep {1, 2}; CAPE uses 2 |
| $d_{\max}$ | `d_max` | distance-transform truncation (not in CAPE) | $3w$ |
| $\beta$ | `beta` | path smoothing; $\infty$ is hard Dijkstra | tier-dependent |
| $\sigma$ | `sigma` | perturbation scale (Tier 2) | sweep |
| $M$ | `n_perturbations` | samples per pair (Tier 2) | 8 |
| $\tau$ | `tau` | clip softening; $0$ is the exact metric | 0.1 |
| $w_\bullet$ | weights | term weights | sweep |

$\kappa$ is annealed rather than tuned. Large $\kappa$ recovers the hard
constraint but the gradient into off-road pixels vanishes, and those are the
gaps — the entire point; small $\kappa$ keeps gradient everywhere but geodesics
cut across open ground and $\hat L$ stops meaning road distance. This is the
standard exterior-penalty conditioning tradeoff, and the standard answer is
continuation: start small, raise on a schedule as the distance field sharpens.
Method of multipliers reaches the constrained solution at finite $\kappa$ but
wants a dual variable per grid edge; try the schedule first.
