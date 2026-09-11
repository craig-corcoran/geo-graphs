# Learned gap closing: link prediction over the proposal graph

## Motivating question

Can a *selective* connectivity operator move `gt_to_prop` where a *global* one
cannot?

The mask threshold is the only connectivity knob the pipeline has, and it is
blunt: lowering it adds road pixels everywhere at once, buying `gt_to_prop`
against `prop_to_gt`. The sweep in `outputs/threshold_sweep.json` already found
where those cancel, and past that point the trade collapses.

| threshold | `gt_to_prop` | `prop_to_gt` | APLS |
|---|---|---|---|
| 0.01 | 0.8273 | 0.7705 | 0.7885 |
| **0.02** | **0.8232** | **0.8258** | **0.8071** |
| 0.05 | 0.7907 | 0.8401 | 0.7967 |
| 0.12 | 0.7505 | 0.8669 | 0.7875 |
| 0.50 | 0.6600 | 0.8739 | 0.7247 |

Moving 0.02 to 0.01 buys **+0.0041** `gt_to_prop` and costs **−0.0553**
`prop_to_gt`. The mask is saturated; additional pixels are noise. So the
remaining `gt_to_prop` loss of 0.177 is not reachable by thresholding, and any
mechanism that reaches it has to add specific edges rather than lower a global
bar.

A link predictor over the proposal graph is that mechanism. Whether it beats the
trade curve above is the experiment; the curve is the baseline it must clear.

## Answer

It does not clear it by enough to be worth building as a learned stage, and the
ceiling was measured before any model existed. Three findings close it
independently, each recorded in `EXPERIMENT_LOG.md` 2026-09-10.

**The ceiling is +0.0224 APLS** (t +4.53, CI [+0.0127, +0.0322]), 12.9% of the
reachable headroom, for a *perfect* oracle. A real predictor gets a fraction of
that.

**The ablation ladder cannot resolve its own arms at that scale.** Plausible
differences between a distance gate, an MLP and a GNN are ~0.002 against a
paired sem of 0.005, so the experiment returns "no detectable difference"
whatever is true. Since separating those arms was the point, the experiment
cannot answer its own question.

**Under attachment weighting the technique is harmful.** It adds 2.67 junctions
per chip of which only 0.49 match a truth junction, so 82% of the junctions it
creates are not on the ground, and junction F1 falls 0.0240. An endpoint-to-edge
connector splits the target edge and makes a degree-3 node where the truth road
simply continues. APLS charges nothing for that; a map someone looks at does.

A fourth measurement removed the most likely objection. The OSM crosscheck found
that roughly seven in eight zero-purity stubs lie on no way OSM maps either, so
the ceiling was not measured against materially incomplete truth and
`prop_to_gt` has been penalising real invention.

### What remains worth doing

One thing, and it needs no model. `both_high_purity` keeps 84% of the APLS gain
at 28% of the junction damage, using 188 edges instead of 443, and sits on the
Pareto frontier across all three metric families. A purity-filtered heuristic gap
closer is an afternoon's work and a modest, honest positive.

Everything below stands as the design that was validated on the way to that
answer. The Tier 0 and Tier 1 sections carry the measurements; the design
decisions above them are reusable by any later link-prediction work, since most
were established by measurement rather than chosen.

## Where the code starts

- `cleanup.clean` is `simplify_edges` → `prune_spurs(20)` → `snap_junctions(8)` →
  `prune_spurs(20)`, a fixed sequence with three constants and no registry.
- `skeleton.graph_from_mask` traces thinned pixels into an `nx.MultiGraph`.
  Node positions live in `pos` as `(x, y)`; 1 px is 1 m, so lengths need no
  conversion.
- `geograph.inject_points(G, points, max_dist)` splits the nearest edge at a
  projected location and returns the node id each point became, or `None` when
  nothing lay within `max_dist`. This is the primitive both the labeller and the
  endpoint-to-edge candidate kind need.
- `metrics.apls` reports both directions separately and combines them with a
  harmonic mean. `gt_to_prop` falls when real roads are missing; `prop_to_gt`
  falls when roads are invented.
- `outputs/vegas_best.pt` is the frozen segmentation stage. Its split is recorded
  in `outputs/vegas_best.json`: 785 train ids, 196 val ids. The first 40 val
  chips were spent selecting the mask threshold; the remaining 155 are the
  reporting holdout, minus `img1612`, whose ceiling is 0.0.
- `data/AOI_2_Vegas` holds 989 chips, 981 with labels. Chips are roughly
  396x324 px. Paris, Shanghai and Khartoum exist only as 10-chip samples.
- `model.py` carries the Protocol-plus-registry pattern this work should follow:
  `Fusion`, `EarlyFusion`, and the definition-site pin `_: type[Fusion] =
  EarlyFusion`.
- Nothing in the pipeline adds an edge. `prune_spurs` removes and
  `snap_junctions` merges; there is no repair operation anywhere.

## Design decisions

### The predictor is a registry stage, and its default is identity

`geo_graphs/refine.py` defines a `@runtime_checkable Refiner` Protocol and a
`REFINER_REGISTRY: dict[str, Callable[[RefineConfig], Refiner]]`, pinned at the
definition site. The registered default is identity, so committed numbers do not
move until the default is deliberately changed. `BACKLOG.md` already asks for
`cleanup` to become registry stages; this pays that item down rather than adding
to it.

### The predictor runs before the first prune

`clean()`'s first `prune_spurs(min_length=20)` deletes short dead-end edges, and
a road severed by a shadow leaves exactly such a stub. Running the predictor
after `clean()` searches for candidates that `clean()` already destroyed.

The decoder becomes `simplify` → `snap` → `link` → `prune`. Stubs that the
predictor declines to connect are still pruned, so nothing survives that the
current pipeline would have kept.

### The label is the APLS pair score

For a candidate joining `a` and `b`, inject both into the truth graph at
`max_dist = 25.0`, which is APLS's own `max_snap`, then:

```
undeterminable  either endpoint fails to land: no truth evidence either way
negative        both land, but no short truth route exists between them
positive        a short truth route exists, and the proposal has no
                comparable route (or none at all)
```

Concretely, positive requires `truth_dist <= tau * euclidean(a, b)` with
`tau = 1.5`, and `prop_dist > (1 + eps) * truth_dist` with `eps = 0.2`, or
`prop_dist` infinite.

This makes the training target the same quantity the metric scores: APLS
compares route lengths between injected control points, and so does the label.
A predictor that fits the label is fitting the metric rather than a proxy for
it, which is the failure the clDice arm ran into from the other direction.

### The labeller's snap radius is its own parameter, set at 10

`max_snap = 25.0` is APLS's default and is correct there, where it applies
symmetrically to both graphs. As a labelling radius on a 396x324 px chip it is
loose: roughly an eighth of a Vegas block. The labeller therefore takes
`label_snap` independently of `metrics.apls`'s `max_snap`.

Shrinking `label_snap` can only ever move a label to undeterminable, never to
negative. `geograph.inject_points` calls `query_nearest(p, max_distance=...)`
and then takes the globally nearest hit; the selected edge and the projected arc
position are both independent of the radius, which gates only whether the point
lands at all. So `truth_dist` is bit-identical across radii whenever both ends
land, and the label is a deterministic function of quantities that do not move.

The consequence for measurement: a snap sweep reports exclusion, not
correctness, and cannot on its own validate the label set. Positive sets nest,
so `label_snap = 10` positives are a strict subset of `label_snap = 25`
positives.

`10` is the committed value. The evidence is stub purity below, which is the
only measurement here that discriminates rather than excludes.

### Stub purity is the label-quality diagnostic

For a candidate endpoint, `stub_purity` is the fraction of its incident edge
polyline lying on the label mask, sampled at 1 px spacing. A genuinely severed
road should score high, since the stub is real road the model found. An invented
stub should score low.

The census measurement, at `R = 60`:

| population | high purity (>= 0.5) |
|---|---|
| all candidate endpoints | 23.0% |
| positives at `label_snap = 25` | 17.7% |
| positives at `label_snap = 15` | 26.3% |
| positives at `label_snap = 10` | **39.1%** |

At the loose radius the positive class is *less* likely to lie on labelled road
than the pool it was drawn from. Snap 10 is the only setting that inverts this,
and it is why the committed value is 10 rather than 25.

Two controls establish the metric reads the graph correctly rather than being
broken. Non-stub edges average 0.749 purity against 0.394 for stub edges, so
dead-end stubs specifically are the off-road part. And dilating the truth mask
to a 17 px band still leaves roughly 41% of candidate endpoints missing the road
entirely, so the mass at zero is not an artifact of mask width. The *middle* of
the distribution is dilation-sensitive, so 0.5 is a reporting threshold rather
than a calibrated one.

Purity is carried as a feature and as an optional candidate filter, with the
filter's value measured in Tier 1 rather than assumed.

### Low purity has two explanations, and they are not separated

Either the model invents dead ends, or SpaceNet Vegas does not label every paved
way and some stubs are real roads absent from the truth. Both cost `prop_to_gt`
identically when an edge is added, so no decision in this plan turns on which is
true, but the project's account of its own results does.

`geo_graphs/osm.py` already pulls OSM ground truth, and OSM carries alleys and
service roads SpaceNet frequently omits. Checking whether low-purity stubs land
on OSM ways separates the two. It needs the `network` marker and it is a
follow-up rather than a gate.

### Positives are split by whether a route already exists

`prop_dist` infinite means the two endpoints sit in different components and the
candidate is genuine connectivity repair. `prop_dist` finite but worse than
`(1 + eps) * truth_dist` means the proposal already has a route and the
candidate is a shortcut across an existing detour.

These are different modelling problems, and they carry different risk: a
shortcut that is wrong costs `prop_to_gt` directly, while a missed component
join only fails to earn `gt_to_prop`. The two are counted and reported
separately from the outset rather than pooled into one positive class.

### The label is three-valued, not boolean

"No truth evidence at this location" and "the truth says these are not
connected" are different, and collapsing them onto `False` teaches the predictor
that unmappable regions are negative evidence. Represented as `bool | None` or
an explicit enum, with `None` excluded from the loss. This is the
overloaded-boolean rule.

### Candidates are gated on distance from the chip border

Every road crossing a chip edge produces a degree-1 node there. At roughly
396x324 px the perimeter is a large share of the area, so border endpoints are
numerous, are artifacts of chipping rather than gaps, and would label
overwhelmingly negative or undeterminable. Left in, they set the class balance.

Candidates come only from endpoints more than `border_margin` pixels from the
edge, default 10.

### Endpoint-edge is the primary candidate generator

- **endpoint-edge**: an interior degree-1 node whose nearest point on a
  non-incident edge lies within `R`, one candidate per endpoint. A severed road
  frequently dangles near the middle of another rather than near its end, and
  `geograph.inject_points` already performs the split this requires.
- **endpoint-endpoint**: two interior degree-1 nodes within `R`, not already
  joined by a single edge.

The second adds no endpoint coverage over the first. In the census the count of
interior endpoints holding any candidate equals the endpoint-edge candidate count
exactly, in all eight split-and-radius cells, and the reason is geometric rather
than incidental: if another endpoint lies within `R`, that endpoint's incident
edge is non-incident to this one and also lies within `R`.

So endpoint-endpoint supplies alternative *targets* for the predictor to score,
never additional reach. Endpoint-edge is the generator that must exist; the other
is an arm.

### Whole chips, not crops

The segmentation model trains on 3200 crops of 256 px, but a crop boundary
severs roads artificially and manufactures endpoints that are not gaps. The
predictor operates on whole-chip proposals: 785 graphs for training, not 3200.

### Message passing runs over existing edges only

Node states are built by propagating along the road network the proposal already
asserts. The head then reads the two endpoint embeddings plus the candidate's own
geometric and image features. Letting candidate edges carry messages leaks the
hypothesis into the node states that are supposed to evaluate it.

Whether candidate edges *should* participate as a second relation type is an arm
in Tier 3, not the starting point.

### Added edges need real geometry

A `geograph` edge carries `pts` and `length`. The connector is drawn either as a
straight line or as a minimum-cost path through `-log p` in the predicted
probability raster. The second is cheap and follows the evidence the model
already produced; both are arms.

## Features

Per candidate `(a, b)`:

**Geometric.** `norm(a - b)`; the angle between each terminal tangent and the
connector; whether the two tangents are anti-parallel, which is the signature of
a severed straight road.

**Image.** Mean, minimum and 10th percentile of predicted probability sampled
along the connector. A canopy gap still carries weak probability while genuine
non-road carries almost none, so the low-order statistics are the discriminative
ones.

**Topological.** `prop_dist(a, b)`, the detour the proposal currently forces.
This feature is the control in the ablation below: it is the hand-computed
version of what message passing is supposed to learn.

## Staging

### Tier 0 — endpoint census

`scripts/endpoint_census.py` runs against the frozen checkpoint at threshold 0.02
over all 981 chips and writes `outputs/endpoint_census.json`. It reports, split
by train and val: interior versus border endpoints per chip; candidate counts at
radii 15, 25, 40 and 60 for both candidate kinds; positive, negative and
undeterminable counts; and how many interior endpoints `clean()`'s first prune
destroys.

Two decision rules were fixed before the numbers existed. Both pass, one of them
narrowly.

| gate | rule | measured at `label_snap = 10` |
|---|---|---|
| pool size | above ~2,000 comfortable, 500 to 2,000 workable with k-fold | 1,529 at `R = 25`, **2,040** at `R = 60` |
| distribution shift | val positive rate within ~1.5x of train | 0.684 against 0.691 at `R = 25` |

The pool clears 2,000 only at the widest radius, and there 9,784 of 14,924
candidates are undeterminable. This sits in the band the rule calls workable
rather than comfortable, so the predictor is trained with k-fold rather than a
single split. Train positives across the radius grid at the committed snap:

| `R` | positives | negatives | undeterminable |
|---|---|---|---|
| 15 | 1,154 | 481 | 1,814 |
| 25 | 1,529 | 1,208 | 4,246 |
| 40 | 1,783 | 1,927 | 6,601 |
| 60 | 2,040 | 3,100 | 9,784 |

Supporting counts: 7,948 interior endpoints on train and 2,337 on val, against
roughly 12 border endpoints per chip on both splits, so the border margin
excludes about half of all raw endpoints. `clean()` destroys 47.0% of train
interior endpoints and 44.8% of val.

The candidate frontier over radius, on train:

| `R` | endpoint coverage | positive rate |
|---|---|---|
| 15 | 0.382 | 0.777 |
| 25 | 0.687 | 0.704 |
| 40 | 0.837 | 0.646 |
| 60 | 0.949 | 0.585 |

Coverage climbs steeply to `R = 25` and flattens after; the positive rate
declines throughout. About 5% of interior endpoints hold no candidate at any
radius tested, which caps recall independently of the predictor.

The positive rate above is quoted at `label_snap = 25`, where both splits carry
the same contamination and the comparison between them stays fair. The pool
counts are quoted at the committed `label_snap = 10`.

### Tier 0 conclusion on memorisation

The predictor trains on in-fold proposals from the 785, and out-of-fold
generation is not required.

The evidence is the second gate above: at `R >= 25` the train and val positive
rates are indistinguishable, and the per-chip fraction of interior endpoints
holding at least one positive matches as well (0.352 against 0.330 at `R = 25`,
p = 0.32). The shift that does exist is in candidate *count*, not label mix: val
chips carry 1.80 more interior endpoints each (p = 0.015), which is the
segmentation model fragmenting unseen imagery more.

The limit on that conclusion, stated because it is easy to overread: this
compares the marginal label rate, not the feature distribution the predictor
consumes. Matching positive rates do not establish that the geometry and image
features match. Claiming that would need a metric that measures it.

The escalation path, if a later tier produces evidence of feature-level shift,
is k-fold out-of-fold proposal generation over the 785 at a cost of k
segmentation training runs.

### Tier 1 — oracle ceiling

Add exactly the positive edges and score APLS on the 155-chip holdout. This is
the maximum the entire line of work can buy, computed before any model exists.
The ordering is the one the harness itself was built on: measure the ceiling
first, then decide whether to chase it. An oracle worth +0.005 ends this line of
work at a cost of one inference pass.

`scripts/link_oracle.py` runs six arms against the frozen checkpoint at threshold
0.02, with `preprune` meaning `simplify_edges` then `snap_junctions`:

| arm | decoder | purpose |
|---|---|---|
| 0 `baseline` | `cleanup.clean` as shipped | what every other arm is compared against |
| 1 `reorder` | `simplify -> snap -> prune` | the reorder alone, no edges added |
| 2 `join` | arm 1 + `component_join` edges | conservative ceiling |
| 3 `shortcut` | arm 1 + `detour_shortcut` edges | the riskier class alone |
| 4 `both` | arm 1 + both classes | full ceiling |
| 5 `both_high_purity` | arm 4 filtered on stub purity | does the purity filter recover ceiling? |

Arm 1 is what makes the rest interpretable. `cleanup.clean` is
`simplify -> prune -> snap -> prune`, so dropping the first prune changes what
`snap_junctions` sees and changes the emitted graph on its own; without arm 1 the
oracle's gain is confounded with the reorder's. The census already showed the
reorder is not neutral, since `clean()` yields 534 more interior endpoints on
train than survive from `preprune`.

Arms 0 and 1 are independent of radius and snap. Arms 2 to 4 run at `R` in
{25, 60} and `label_snap` in {10, 25}; arm 5 at `label_snap = 10` only.

Both APLS directions are reported separately for every arm. The shortcut class
is expected to move them in opposite directions, and pooling into the harmonic
mean would hide exactly that. **The `prop_to_gt` gap between the snap-10 and
snap-25 oracles is the label-noise measurement** that the snap sweep could not
provide, and it works precisely because the positive sets nest.

Oracle connectors are straight lines. That understates the ceiling slightly
against a connector following the probability ridge, since a straight chord
across a curve adds length error the metric charges for; the bias has a known
direction and is not corrected here.

### The ladder Tier 1 ruled out

The learned stage would have been separated by this ladder:

| arm | model | isolates |
|---|---|---|
| A | distance and heading gate | the null model, no learning |
| B | MLP on geometric and image features | is image evidence sufficient? |
| C | B plus the explicit `prop_dist` feature | does topology matter at all? |
| D | GNN over existing proposal edges | does learned propagation beat one hand-computed topological feature? |

C against D was the question worth asking. It is not askable here: with a total
ceiling of +0.0224 APLS, plausible differences between these arms sit around
0.002 against a paired sem of 0.005 over 155 chips. Resolving 0.002 at t = 2
would need roughly 25x the chips, about 3,900; all of Vegas is 981.

This is the transferable lesson rather than a fact about gap closing. **Check
that an ablation can resolve its own arms before building the arms**, by pairing
the ceiling measurement with the sem of the metric the arms will be compared on.
Both numbers were available here before any model was written.

An attribute-inference task escapes this by sample size: nodes placed along
roads give tens of thousands of labelled items per split rather than a few
hundred edges read through a chip-level aggregate, so node-level macro-F1 has
error bars tight enough to separate the same arms.

## Measurement

**Link level.** Average precision on held-out candidates. Cheap, and the signal
to iterate on.

**End to end.** APLS delta on the 155-chip holdout with the mask threshold fixed
at 0.02, so exactly one thing changes. Both directions reported separately.

**Against the baseline trade curve.** The predictor has its own operating
threshold. Sweep it and plot each operating point as `(gt_to_prop, prop_to_gt)`
on the same axes as the 15 mask thresholds. Points landing on the existing curve
mean the predictor is an expensive reimplementation of lowering the threshold.
Points outside it mean it is doing something thresholding cannot.
`threshold_sweep.json` already carries a `pareto_frontier` key, so the
convention exists.

**Stage separation.** Candidate generation recall, predictor precision and
recall, and end-to-end APLS are reported separately. A predictor that looks
excellent on a candidate set that never contained the real gaps is the failure
this separation exists to catch.

## Cost knobs

Committed defaults are the expensive, correct settings; each is a parameter, not
an edit to remember reverting.

| knob | cheap | committed |
|---|---|---|
| `--limit` chips per split | 8 | all |
| candidate radii | one value | 15, 25, 40, 60 |
| `label_snap` | one value | 10 committed; 25 also scored in Tier 1 |
| message passing hops | 1 | swept |
| connector geometry | straight | minimum-cost path |
| out-of-fold proposals | not run | not run; Tier 0 ruled it unnecessary |

## Make targets

- `census` — run the endpoint census against the frozen checkpoint.
- `link-oracle` — score the oracle gap-closer on the holdout.
- `link-sweep` — train and sweep the predictor against the cached candidate set.

Each runs against a frozen prior stage, so no target in this plan retrains the
segmentation model.

## Which risks bit

Recorded because the pattern is more useful than the list: the risk that killed
this was not the one flagged as most likely.

**The top-flagged risk did not bite.** "The positive class rests on stubs that
are largely off the labelled road" was called the largest known threat. It was
real — 47% of candidate endpoints have a stub that never touches the label mask —
but the OSM crosscheck showed those stubs are genuinely invented rather than
unlabelled roads, so the ceiling was measured against adequate truth and the
number stood.

**The risk that killed it was filed as "not a project risk."** The entry said
"the GNN loses to arm C. Not a project risk. It is a result." That was wrong in
a specific way: the problem was never that the GNN might lose, it was that the
comparison could not be *run* at this effect size. A result requires a
measurement that can distinguish the arms, and this one could not.

**One risk was mis-stated rather than wrong.** Class imbalance was expected to
need subsampling or a focal loss; at `label_snap` 10 the labelled classes are
close to balanced (2,040 positive against 3,100 negative). The real
distributional problem is the two thirds of candidates that are undeterminable:
excluded from the loss, present at inference.

**And one hazard was not on the list at all.** The metric's choice is upstream of
the verdict. Gap closing looks positive under route weighting and negative under
attachment weighting, and nothing in the original plan considered that the answer
might depend on which metric refereed. `metrics.buffer_length_prf` and
`metrics.junction_prf` exist because of this.

## References

- Sat2Graph, He et al., ECCV 2020. Graph-tensor encoding, verified against the
  paper: a `W x H x (1 + 3 * D_max)` tensor with `D_max = 6`, one vertexness
  channel plus six sector-restricted `(edgeness, dx, dy)` slots. Relevant here
  as the alternative decoder, not as a GNN.
- The pairwise-connectivity formulations that predict keypoints and then classify
  edges between them (SAM-Road, RNGDet) are the published relatives of Tier 2
  arm D. Recalled, not checked against the papers.
