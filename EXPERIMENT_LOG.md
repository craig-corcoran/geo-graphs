# Experiment log

Append-only. Newest entries at the bottom.

---

## 2026-08-23 — Evaluation harness before the model

**Change under test.** Built the Stage 1 evaluation path (OSM ground truth →
1 m/px mask → skeleton → graph → APLS) ahead of any segmentation model, on the
argument that a metric validated against a known-correct answer is a
precondition for trusting any model number later. Ground truth comes from OSM
via osmnx rather than SpaceNet, so the whole loop runs with no imagery download.

**What was measured.** Round trip on a 1024 m tile of downtown Las Vegas
(36.1699, -115.1398 — inside SpaceNet AOI 2). Ground truth is rendered to a
*perfect* mask, traced back out, and scored against itself. Any loss here is
the pipeline's own ceiling.

### Three bugs the invariants caught

1. **Polyline orientation.** An undirected `nx.MultiGraph` reports each edge in
   whichever orientation its adjacency dict stores, so `data["pts"]` is not
   guaranteed to start at `u`. Three call sites assumed it did and silently
   built mirrored geometry. Symptom: control points displaced up to 318 px, and
   `apls(G, G)` scoring 0.935 instead of 1.0. Fixed with `geograph.oriented_pts`.

2. **Control points snapped to nodes, not edges.** Snapping a ground-truth
   control point to the nearest *node* of the proposal moves it by up to the
   node spacing, and that displacement becomes fake path-length error. Short
   routes were worst hit: mean score 0.57 for routes under 100 m against 0.92
   above 1 km. Replaced with `geograph.inject_points`, which splits the nearest
   edge at the projected location. Displacement is now exactly 0.

3. **Node id collision in the reference adapter.** Our truth and proposal graphs
   both numbered nodes from 0. The reference inserts each control point into the
   *other* graph under its own id, so those inserts landed on unrelated existing
   nodes. This is invisible in SpaceNet's own workflow, where the two graphs
   carry disjoint namespaces (OSM ids vs generated). Fixed by offsetting
   proposal ids; agreement on the round trip went from 0.39 apart to 0.0004.

### Reference implementation

`CosmiQ/apls` 0.1.0 cannot be installed on a current interpreter: it pins
`GDAL==2.4.0` and uses Python-2 style implicit relative imports. The scoring
math needs none of that — GDAL appears only in the geojson/GeoTIFF ingestion and
plotting layers. Vendored the GDAL-free scoring subset (Apache-2.0) into
`tests/reference/` as a test oracle.

**It uses a harmonic mean, not an arithmetic one.** The code calls
`scipy.stats.hmean` while printing the result labelled `Total APLS Metric =
Mean(...)`. The label is indistinguishable from correct whenever the two
directional scores are close, which is exactly the regime of a healthy
proposal — the divergence only shows up on lopsided ones. Trusting the label
cost 0.38 APLS on a proposal missing half its edges.

### Numbers

Agreement between our implementation and the reference, downtown Las Vegas tile:

| Case | Ours | Reference | Δ |
|---|---|---|---|
| identity | 1.0000 | 1.0000 | 0.0000 |
| drop 5% edges | 0.9448 | 0.9445 | +0.0003 |
| drop 15% edges | 0.7724 | 0.7559 | +0.0165 |
| drop 30% edges | 0.4457 | 0.4202 | +0.0255 |
| drop 50% edges | 0.1019 | 0.0968 | +0.0051 |
| skeleton round trip | 0.8753 | 0.8757 | −0.0004 |

Mean |Δ| 0.008, max 0.026. Residual disagreement is control-point *sampling*,
not scoring: we densify every edge uniformly, the reference skips midpoints on
straight edges and spaces them with `linspace`.

**Current harness ceiling** (after the orientation fix also landed in
`snap_junctions`): APLS **0.8794**, gt→prop 0.9532, prop→gt 0.8161, mask IoU
0.7651.

**What it motivates.** The ceiling is asymmetric — we recover nearly every real
road (0.95) but invent extra ones (0.82). The suspect is spurious skeleton
structure where dilated intersections blob together, plus parallel streets
closer than the road width merging into one centerline. Tightening that is
worth more than any model change, because every model number inherits it.

---

## 2026-08-23 (later) — The ceiling is non-planarity, not road width

**Correction to the entry above.** That entry attributed the 0.879 ceiling to
"spurious skeleton branches where dilated intersections blob together, and
parallel streets closer than the road width merging into a single centerline."
Measurement refutes that. The real cause is that a 2D binary mask cannot
represent a road network that is not planar.

**Change under test.** Swept rasterization road width and grid resolution, on
the theory that if merging drove the ceiling, thinner roads at finer resolution
should approach 1.0.

| res m/px | half_w px | road m | truth len | rec len | APLS | gt→p | p→gt |
|---|---|---|---|---|---|---|---|
| 0.5 | 1 | 1.5 | 19435 | 19288 | 0.8981 | 0.9642 | 0.8405 |
| 0.5 | 2 | 2.5 | 19435 | 19276 | 0.8972 | 0.9640 | 0.8390 |
| 0.5 | 4 | 4.5 | 19435 | 19229 | 0.8927 | 0.9623 | 0.8325 |
| 1.0 | 1 | 3.0 | 19435 | 19182 | 0.8949 | 0.9545 | 0.8424 |
| 1.0 | 2 | 5.0 | 19435 | 19191 | 0.8794 | 0.9532 | 0.8161 |
| 1.0 | 4 | 9.0 | 19435 | 18796 | 0.8868 | 0.9522 | 0.8298 |
| 2.0 | 1 | 6.0 | 19435 | 18838 | 0.9013 | 0.9489 | 0.8583 |
| 2.0 | 2 | 10.0 | 19435 | 18019 | 0.8965 | 0.9483 | 0.8501 |
| 2.0 | 4 | 18.0 | 19435 | 16364 | 0.8608 | 0.9427 | 0.7920 |

**It does not converge.** Twelve-fold variation in road width (1.5 m to 18 m)
and four-fold in resolution move APLS only between 0.861 and 0.901. There is a
floor near 0.90 that no amount of thinning crosses. Merging is real but minor:
it shows up as the length loss in the bottom rows, not as the ceiling.

**What actually costs the score.** Breaking down all 84,227 scored route pairs
in the proposal→truth direction:

| Outcome | Pairs | Share |
|---|---|---|
| within 5% of truth | 62,072 | 73.7% |
| proposal **shorter** than truth | 10,858 | 12.9% |
| proposal **longer** than truth | 10 | 0.0% |
| no path in truth | 11,287 | 13.4% |

The asymmetry is the whole story. We essentially never miss a connection (10
pairs). We invent them, 12.9% of the time. The recovered graph is strictly
*over*-connected.

The worst cases are extreme — the proposal routes between two points in 14 m
where the truth needs 958 m. Those are not geometric error; they are junctions
that do not exist.

**Confirmed cause.** The ground-truth graph contains **20 pairs of edges that
cross with no shared node**, and raw OSM tags **13 bridges and 4 tunnels** in
this tile. The pixel locations of those crossings coincide with the worst
invented shortcuts — (439, 61) and (980, 251) appear in both lists. Where a
road passes over another, OSM correctly records no junction, because you cannot
turn from one to the other. Rasterized to a plane, the two roads touch, and
skeletonization has no choice but to emit a junction.

**What it motivates.** This is not a knob to tune. Segmentation-plus-
skeletonization cannot express an overpass, so a ~0.90 ceiling is intrinsic to
Stage 1 on any city with grade separation. Two consequences:

1. Stop trying to close the gap by tuning cleanup. The remaining headroom below
   ~0.90 is small; the rest requires a different representation.
2. This is the concrete, measured argument for Stage 2. Sat2Graph's per-cell
   directional edge slots can encode two roads meeting at one pixel without a
   junction. The plan asserts that as motivation; this tile now demonstrates it
   with a number.

**Also measured: is our APLS worth keeping for speed?** Ours 0.489 s per call,
reference 0.615 s — only 1.26x. Speed is not an argument for maintaining a
second implementation.

---

## 2026-08-23 (later still) — Matched the reference's control point sampling

**Change under test.** Our APLS densified every edge uniformly; the reference
skips interior control points on straight edges entirely, skips edges under
0.75×delta, and spaces the rest with `linspace`. Previously dismissed as
mimicking a quirk. The stronger argument for matching is that agreement was
only good to 0.026, forcing a test tolerance of 0.08 — loose enough that a bug
shifting APLS by 0.05 would pass unnoticed.

**What was measured.** Reimplemented the reference rule and compared, with
subsampling disabled, on the Las Vegas tile.

| Case | Reference | Uniform | Δ | Matched | Δ |
|---|---|---|---|---|---|
| identity | 1.000000000 | 1.000000 | +0.0000 | 1.000000000 | 0 |
| drop 5% | 0.944518624 | 0.940894 | −0.0036 | 0.944518624 | 0 |
| drop 15% | 0.755870790 | 0.788723 | +0.0329 | 0.755900997 | 3.0e−05 |
| drop 30% | 0.420172728 | 0.477363 | +0.0572 | 0.420172728 | 1.7e−16 |
| drop 50% | 0.096768947 | 0.101949 | +0.0052 | 0.096782004 | 1.3e−05 |
| round trip | 0.877559293 | 0.881382 | +0.0038 | 0.877588593 | 2.9e−05 |

Max residual **3.0e−05**, and three of six cases are bit-exact.

The sampling difference was larger than expected: on the truth graph, uniform
gives 435 control points against the reference's 176 (144 junctions plus 32
midpoints, all on curved edges). Note also that the earlier reported agreement
of "mean 0.008" was partly luck — random subsampling to 300 control points was
masking some of the divergence. With subsampling off, uniform drifts by up to
0.057.

**What changed.** `sampling="reference"` and `max_control=None` are now the
defaults, so `metrics.apls(truth, proposal)` is directly comparable to the
published numbers. The dense `"uniform"` rule is retained as an opt-in, because
it is the better estimator for *localizing* loss — the non-planarity finding in
the previous entry came from having 84k route pairs to classify rather than 15k.
`test_against_reference.py` now asserts agreement to **1e-4** instead of 0.08,
and separately pins that uniform sampling is a different estimator, so the two
cannot be confused.

**Effect on the headline number.** The Stage 1 ceiling moves 0.8794 → **0.8776**
(gt→prop 0.9372, prop→gt 0.8251). The conclusion of the previous entry is
unchanged: the ceiling is non-planarity, not sampling.

---

## 2026-08-24 — Offline test coverage found two silent geometry bugs

**Change under test.** An audit against `CLAUDE.md` showed the fast loop tested
almost nothing: `skeleton`, `cleanup`, `raster` and `tiles` were imported only
by a `network`-marked test, so `make check` exercised `geograph` and `metrics`
and nothing else. Wrote the missing per-module offline suites. The offline suite
went from 28 tests to 163.

Writing them turned up two defects that every number in this log had been
quietly carrying.

### 1. Parallel edges silently discarded during tracing

`graph_from_mask` traces each edge from both ends and drops the second visit.
The dedup key was `(min(start, end), max(start, end), len(path))` — endpoint
*pixels* plus path length. Two genuinely distinct routes sharing endpoints and
size collide, and the second is thrown away. Because `_trace` has already marked
its pixels visited, `_add_isolated_loops` cannot recover them either: the
geometry is gone, not merely un-deduplicated.

Minimal reproduction — two diamond outlines meeting at one shared pixel:

| | Expected | Got |
|---|---|---|
| nodes | 1 | 1 |
| degree | 4 | 2 |
| self-loops | 2 | 1 |
| total length | 90.51 | 45.25 |
| interior pixels on some edge | 62 / 62 | 31 / 62 |

Exactly half the network vanished. The failure is latent rather than universal:
it needs single-pixel junctions, so a junction that thins to a multi-pixel blob
gives the two traces different start pixels and different keys. That is why it
never showed on typical thick-road masks. Fixed by keying on the pixel path
itself, canonicalized against its own reverse, which cannot collide.

### 2. Rasterization was not a function of the geometry

`skimage.draw.line` is not symmetric — drawing A→B and B→A differ by 4 pixels on
a test diagonal. Edge orientation in an undirected MultiGraph is arbitrary, so
the same road network built in a different order rasterized to a different mask.
Fixed by drawing each segment from a canonical endpoint.

### Effect on the tile

| | Before | After |
|---|---|---|
| recovered nodes / edges | 142 / 215 | 141 / 214 |
| recovered length (m) | 19 191 | 19 302 |
| mask IoU | 0.7651 | 0.7703 |
| APLS | 0.8776 | 0.8764 |
| gt→prop | 0.9372 | 0.9372 |
| prop→gt | 0.8251 | 0.8231 |

Recovered length moved toward truth (19 435 m) and IoU improved, both consistent
with geometry that was being dropped now being kept. APLS moved down 0.0012 —
worth stating plainly: recovering the missing arcs *adds* connections, and the
proposal's problem is that it is already over-connected. The bug was mildly
flattering the score.

**What it motivates.** The ~0.90 non-planarity ceiling from the previous entry
stands; these fixes do not touch it. The lesson is about where coverage was
missing rather than about the metric: both defects sat in code the offline loop
never ran, and both were invisible in aggregate numbers — length was off by 1%,
APLS by 0.001. Neither would have been caught by watching the headline score,
which is the trap the "don't let a coarse-grained metric carry a fine-grained
claim" rule warns about.

---

## 2026-09-01 — Crossing *density*, not count, tracks the ceiling

**Change under test.** Wrote a walkthrough notebook (`examples/walkthrough.ipynb`)
and tried to have it demonstrate the non-planarity result rather than assert it,
by comparing tiles of different sizes around the same centre.

**What was measured.** Perfect-mask round trip at three tile sizes.

| tile | km road | crossings | per km | APLS | prop→gt |
|---|---|---|---|---|---|
| 512 m | 4.5 | 0 | 0.00 | 0.9624 | 0.9612 |
| 1024 m | 19.4 | 20 | 1.03 | 0.8764 | 0.8231 |
| 2048 m | 71.3 | 40 | 0.56 | 0.9487 | 0.9278 |

**The naive reading is wrong.** The 2048 m tile has *twice* the crossings of the
1024 m tile and scores *higher* — 0.9487 against 0.8764. Raw crossing count does
not predict the ceiling, and taken alone it would have looked like evidence
against the non-planarity explanation from the 2026-08-23 entry.

**Normalizing fixes it.** Crossings per km orders the three tiles exactly:
0.00 → 0.9624, 0.56 → 0.9487, 1.03 → 0.8764. The mechanism is straightforward
once stated — APLS averages over route pairs, so twenty crossings spread across
71 km of network corrupt a much smaller share of routes than twenty concentrated
in 19 km. The loss also lands in `prop→gt` in every case, which is the specific
signature an invented junction should leave.

**Strength of the claim.** Three tiles is three points, and tile size varies more
than crossing density alone; this is corroboration, not proof. The independent
evidence remains the width/resolution sweep in the 2026-08-23 (later) entry,
which no amount of thinning moved. Also worth noting the ceiling sits below 1.0
even at zero crossings — rasterizing and thinning lose a little geometry
regardless, through junction-blob collapse and coordinate quantization.

**What it motivates.** Two things. First, when Stage 1 model numbers arrive,
compare them against the ceiling *for that tile*, not a global constant — a
model evaluated on flat downtown tiles and one evaluated across an interchange
are not being held to the same standard. Second, this is the failure mode the
"don't let a coarse-grained metric carry a fine-grained claim" rule describes:
the count was the coarse measure, and it pointed the wrong way.

---

## 2026-09-01 — Training loop built before the imagery

**Change under test.** Built the segmentation half — `data`, `model`, `train` —
against fabricated imagery, rather than waiting on a SpaceNet download. The bet
is the same one that paid off for the harness: when the real data lands, only
one thing is new, instead of data and PyTorch and evaluation all at once.

`TileSource` is the seam. `SyntheticTileSource` pairs genuine OSM labels with an
invented image; a SpaceNet source drops in behind the same Protocol and changes
one config key. Crop sampling is pure numpy and returns *windows* rather than
pixels, so a dataset is described by reproducible specs and materialized on
demand.

**The imagery is fabricated, so no number here is a result.** The image is
derived from the label. It is built to be non-trivially separable — blurred road
edges, textured background, road-coloured rectangular distractors, per-channel
tint, additive noise — which is enough to catch a channel-order slip or a broken
normalization, and nothing more. A test asserts no single global threshold
recovers the mask, so the fake cannot silently degrade into the identity task.

### Sanity checks, all passing

| Check | Expected | Got |
|---|---|---|
| BCE at zero logits | ln 2 = 0.6931 | 0.6931 |
| Dice, perfect prediction | 0 | 0.0000 |
| Dice, predicting nothing | ~1 | 0.9984 |
| Parameters with no gradient | none | none |
| Overfit 4 crops, 600 steps @ lr 1e-2 | → 0 | 0.0003 |

The overfit-one-batch diagnostic is a first-class function and a test, not a
scratch script. It initially looked like a failure — loss stalled at 0.31 — which
turned out to be 120 steps being too few rather than anything wrong. Worth
recording, because "the loop is broken" and "the schedule is too short" look
identical for the first hundred steps.

### End to end

The whole chain runs: train → full-tile inference → skeletonize → clean → APLS.
On a held-out 512 m tile after 6 epochs:

| Stage | Score |
|---|---|
| pixel IoU | 0.9522 |
| APLS, raw traced graph | 0.9608 |
| APLS, after cleanup | 0.9715 |
| APLS ceiling, perfect mask | 0.9715 |
| fraction of ceiling | 1.000 |

Again: **these are plumbing numbers, not quality numbers.** The CLI logs a
warning to that effect whenever the source is synthetic.

**Per-stage metrics now exist**, which closes a backlog item. `EvalReport`
separates pixel agreement, the raw traced graph and the cleaned graph, so a
cleanup regression can no longer masquerade as a skeleton regression. The row
that matters is `fraction_of_ceiling`: the ceiling is tile-dependent (0.96
downtown, 0.88 across the interchange), so a raw model APLS compared against a
global constant would rate a good model on hard tiles as a bad one.

**What it motivates.** The remaining Stage 1 work is genuinely just data: fetch
SpaceNet Roads, add a `SpaceNetTileSource`, register it. After that the first
real question is threshold selection — `predict_mask` takes one, and section 6 of
the walkthrough is the argument for tuning it on APLS rather than IoU.

---

## 2026-09-01 (later) — Interior control points buy geometry, not topology

**Change under test.** Not a pipeline change — a probe of the control point
sampling rule, prompted by asking what edge sampling is actually for when the
two graphs already share a node set. Junction-only sampling is reachable
without new code: `spacing=1e9` puts every edge under the reference rule's
`0.75 * spacing` floor, so no interior points are cut.

**What was measured.** Two proposals built on the `curvy_graph` fixture (4x4
grid, every third edge bowed 20 m), each constructed to have a node set
*identical* to the truth graph. Synthetic geometry, so these are metric
characterization numbers, not pipeline results.

| proposal | junction-only (C=16) | reference (C=32) | uniform (C=48) |
|---|---|---|---|
| mirrored bows, arc length preserved | **1.0000** | 0.2948 | 0.6777 |
| bows flattened to their chords | 0.9687 | 0.9632 | 0.9676 |

The mirrored proposal has the same nodes and the same total length as the truth
(2473.8 m both), with every curved road displaced by up to 40 m. Junction-only
sampling calls it perfect. The flattened proposal is the control: its error
lands *in path length*, the junctions already carry it, and interior points add
nothing.

**Snapping is the only channel through which physical location enters the
score, and snapping happens only at control points.** Junction-only sampling
therefore collapses APLS to a comparison of the shortest-path metric on the
shared node set — a pure graph comparison, blind to where the roads run.
Topology is carried by the junctions for free; the interior points are what buy
geometric sensitivity.

**That sensitivity is gated entirely by `max_snap`.** Same mirrored proposal,
reference sampling:

| max_snap | 10 | 25 | 40 | 60 |
|---|---|---|---|---|
| APLS | 0.2419 | 0.2948 | **0.8995** | 0.8995 |

Peak displacement is 40 m; at or above it the same proposal jumps from 0.29 to
0.90. So "geometric sensitivity" means precisely "sensitivity to displacement
exceeding `max_snap`" — below that threshold APLS is designed not to care. Any
claim that an APLS number reflects geometric accuracy has to quote `max_snap`
alongside it.

**This is a better justification for the straight-edge skip than the one in the
2026-08-23 entry.** That reading was about path-length redundancy: an interior
point on a straight edge has distances determined by its two endpoints. The
sharper statement is about degrees of freedom in the *geometry*. A straight
edge's interior is fixed by its endpoints, so a control point there can discover
nothing; a curved edge's interior is unconstrained by them, which is exactly
where a proposal can be wrong while the junctions look fine. That is why
reference sampling detects the mirrored displacement far better than uniform
(0.2948 against 0.6777) despite carrying a third fewer points — it concentrates
the sample where geometry is free, while uniform dilutes it with points on
straight edges that were never going to disagree.

**Only the source graph is densified per direction.** `_directional(A, B)`
samples `A`; `B` contributes geometry for snapping and edge lengths for
Dijkstra. Densifying `B` would be a no-op for shortest paths — inserting a
degree-2 node into an edge changes no distance. The reference densifies both
graphs (`apls_reference.py:1052`, `:1105`) because each is the source of its own
direction, not to match densities.

**What changed.** `APLSResult` now carries `n_control_gt` and `n_control_prop`;
the reverse direction's count was previously computed and discarded. On the Las
Vegas round trip:

| | nodes | edges | length | C (reference) | C (uniform) |
|---|---|---|---|---|---|
| truth | 144 | 188 | 19435.3 | 176 | 435 |
| recovered | 141 | 214 | 19301.8 | 179 | 414 |

**What it motivates.** The two counts should not be normalized to match — their
ratio is a diagnostic. Here they are within 2% while the recovered graph carries
26 more edges for 0.7% *less* road, which is fragmentation at the junctions
rather than curved-edge noise. A proposal whose `n_control_prop` runs well above
`n_control_gt` is failing the straightness test on edges the truth considers
straight, i.e. wobbly traced geometry. Worth watching once model proposals
replace perfect-mask round trips, since it separates that failure from
fragmentation, which the edge count catches instead.

---

## 2026-09-01 (later) — Real SpaceNet imagery, and two things the data does differently

**Change under test.** Downloaded the SpaceNet Roads sample (0.71 GB, all four
AOIs, ten chips each) rather than the 24 GB Vegas tarball, on the grounds that
every unknown worth resolving — georeferencing, label format, alignment — is
answerable from ten chips. Built `spacenet.SpaceNetTileSource` behind the
existing `TileSource` Protocol.

Two assumptions broke, and both would have quietly corrupted every model number.

### 1. The imagery is geographic, not projected

RGB-PanSharpen ships as EPSG:4326. A Vegas pixel is **0.243 m east-west and
0.300 m north-south** — square in degrees, and 19% out of square on the ground.
Our entire pipeline measures distance in pixels and calls it metres.

Each chip is now reprojected to its local UTM zone at a fixed metric resolution
on load, defaulting to 1 m/px to match both the published numbers and our own
ceiling measurements. `tiles.tile_from_transform` refuses a geographic CRS
outright: its earlier square-pixel check compared the transform's own units, so
it would have accepted this raster and been wrong.

Alignment is verified rather than assumed. Vegas asphalt is dark against bright
desert, so a correctly aligned mask has measurably darker pixels underneath it —
0.222 on road against 0.345 off it. A reprojection slip would erase that gap.

### 2. The labels are not noded

SpaceNet ships each road as a single LineString running **straight through its
intersections**. Two crossing streets share no vertex, so a graph built from the
features as given is a heap of disconnected stubs. One chip: 33 edges across
**30 components**, largest holding 5% of the nodes.

The damage lands entirely in `prop→gt`, because the rasterized mask joins what
the labels leave apart and the metric then calls every junction invented:

| | Unnoded | Noded |
|---|---|---|
| components (img794) | 30 | 3 |
| `gt→prop` | 0.8788 | — |
| `prop→gt` | 0.0549 | 0.9916 |
| **mean ceiling, 10 chips** | **0.245** | **0.9822** |

Snapping nearby endpoints does not fix it — even an 8 px tolerance left 26
components — because the endpoints are not near each other. The side street
meets the main road's *interior*. The fix is noding: split every line at every
intersection, via `shapely.ops.unary_union`.

**Without this, every Stage 1 model number would have been measured against a
ceiling of 0.25.** OSM does this noding for us, which is exactly why the OSM
path never needed it and why the gap was invisible until real labels arrived.

A related sharp edge: `unary_union` emits a degenerate zero-length piece at some
intersections, which becomes a self-loop adding two to a junction's degree and
no geometry at all. Those are now dropped.

`unary_union` only splits at *exact* intersections, and real SpaceNet labels do
intersect exactly. A test built around a T-junction whose endpoint reprojection
had nudged a fraction of a pixel off the line did not split at all — worth
knowing as a fragility, and the reason the regression test uses an unambiguous
crossing.

### End to end on real imagery

Eight chips for training, two held out, 15 epochs:

| | img767 | img794 |
|---|---|---|
| pixel IoU | 0.4658 | 0.4431 |
| APLS | 0.5147 | 0.3311 |
| APLS ceiling | 0.9832 | 0.9826 |
| fraction of ceiling | 0.523 | 0.337 |

**Eight chips is roughly 1 km² of training data.** Published SpaceNet numbers of
63–74 APLS come from thousands of chips, so these are not comparable to anything
and the epoch-to-epoch validation swing (IoU bouncing between 0.002 and 0.48)
is what a dataset this small looks like. What the run establishes is that the
path is complete and the ceiling is real.

**What it motivates.** The loader is proven, so the 24 GB AOI 2 Vegas tarball is
now worth pulling. After that the first genuine experiment is threshold
selection, tuned on APLS rather than IoU.

## 2026-09-08 — The threshold was worth 0.08 APLS, and IoU pointed the wrong way

`predict_mask`'s threshold had never been tuned; every number so far came from
the 0.5 default. Swept it against APLS on the frozen `vegas_best.pt` checkpoint,
over the same 40 validation tiles the run itself scored. No retraining —
`make threshold-sweep`, about 25 seconds.

Logits and the ceiling graph are threshold-independent, so both are computed
once per tile and only mask → skeleton → cleanup → APLS repeats. That is what
makes a 15-point sweep cost roughly one evaluation pass rather than fifteen.

### The trade, measured

| thr | IoU | APLS | median | gt→prop | prop→gt | frac of ceiling |
|---|---|---|---|---|---|---|
| 0.01 | 0.4702 | 0.7885 | 0.8246 | 0.8273 | 0.7705 | 0.8119 |
| **0.02** | 0.5162 | **0.8071** | 0.8374 | 0.8232 | 0.8258 | **0.8314** |
| 0.03 | 0.5392 | 0.8030 | 0.8340 | 0.8148 | 0.8261 | 0.8271 |
| 0.05 | 0.5626 | 0.7967 | **0.8427** | 0.7907 | 0.8401 | 0.8205 |
| 0.10 | 0.5842 | 0.7933 | 0.8289 | 0.7614 | 0.8605 | 0.8168 |
| 0.15 | 0.5907 | 0.7699 | 0.8064 | 0.7313 | 0.8473 | 0.7924 |
| 0.30 | **0.5947** | 0.7558 | 0.7914 | 0.7018 | 0.8728 | 0.7779 |
| 0.50 | 0.5923 | 0.7247 | 0.7595 | 0.6600 | 0.8739 | 0.7459 |
| 0.80 | 0.5702 | 0.6688 | 0.7030 | 0.5915 | 0.8559 | 0.6878 |

`gt→prop` falls monotonically with threshold across the whole range, from 0.827
to 0.592: thinning the mask severs roads, exactly as `predict_mask`'s docstring
claims. `prop→gt` is far flatter, 0.77 to 0.87, so the trade is not symmetric —
most of what the threshold buys or loses is on the missed-road side.

### The headline, and it is the project's own thesis

**IoU peaks at 0.30. APLS peaks at 0.02.** Tuning this threshold on pixel
overlap would have chosen 0.30 and left 0.05 APLS on the table; the APLS-optimal
0.02 is 0.078 *worse* on IoU than the IoU-optimal point. The two metrics do not
merely differ in scale, they disagree about the direction of improvement. This
is the clearest instance so far of the gap the project exists to demonstrate,
and it cost no training to produce.

### How much of it is real

Paired per-tile differences across the same 40 tiles:

| comparison | mean Δ APLS | sem | t |
|---|---|---|---|
| 0.02 vs 0.50 | +0.0824 | 0.0265 | +3.11 |
| 0.05 vs 0.50 | +0.0720 | 0.0252 | +2.86 |
| 0.10 vs 0.50 | +0.0685 | 0.0202 | +3.39 |
| 0.02 vs 0.05 | +0.0104 | 0.0090 | +1.14 |
| 0.02 vs 0.10 | +0.0138 | 0.0135 | +1.02 |
| 0.05 vs 0.10 | +0.0035 | 0.0105 | +0.33 |

Two separate conclusions, and they should not be run together. **"Move well
below 0.5" is solid** — around +0.07 to +0.08 APLS at t ≈ +3, improving 22-24
tiles out of 40. **"Which value in 0.02–0.12" is not resolvable at n=40** — every
within-plateau comparison sits under |t| = 1.6. Reporting 0.02 as *the* optimum
would be reading noise; the finding is the plateau, not its argmax.

Per-tile spread stays high throughout (sd ≈ 0.16 against a mean of 0.80), which
is the ~300 m chip size doing what the `AggregateReport` docstring warns about.

### Choosing within the plateau

0.02 has the best mean and the best fraction of ceiling. It also sits one step
from a cliff: at 0.01 `prop→gt` collapses from 0.826 to 0.771 and mean APLS
drops 0.019, because a near-zero threshold starts admitting noise as road. 0.05
holds the best *median* and has margin on both sides; 0.10 gives up 0.014 mean
APLS for the most margin of all and near-peak IoU.

Since the plateau is flat within noise, the house rule says decide on what
survives scale rather than on the argmax. **0.05** is the defensible default: a
cliff at 0.01 that a retrained or differently-calibrated model could shift is a
real operational risk, and 0.05 sits far enough from it to absorb that without
giving up measurable score. Left unset in code pending that call.

### What it motivates

The sweep's first range started at 0.2 and missed the optimum entirely — the
committed default now spans 0.01 to 0.80, dense at the low end. Worth
remembering as a general failure mode: a sweep that peaks at its own boundary
has not found an optimum, it has found the edge of the grid.

Two follow-ons. The threshold interacts with `cleanup`'s spur-pruning length —
a lower threshold produces more short spurs, which is exactly what pruning
removes, so the two should be swept jointly rather than one at a time. And this
sweep re-ranks nothing about epoch selection: the checkpoint was chosen by
`val_loss`, a pixel measure, and the same disagreement that shows up here
between IoU and APLS applies to that choice too.

## 2026-09-08 (later) — clDice: a weak metric with a differently-shaped gradient

Implemented `soft_cldice_loss` and dihedral augmentation, then measured clDice
statically before spending a training run on it. The static result is a clean
negative; the dynamic question it raises is not, and the two get reported
separately because the first does not imply the second.

### What clDice does on our geometry

Cost of a severing gap, relative to the same damage measured by soft Dice:

| geometry | clDice cost ÷ Dice cost |
|---|---|
| 3×3 grid, roads spanning the tile (**matches a SpaceNet chip**) | 0.87 |
| single bar, edge-to-edge | 0.99–1.00 |
| single bar, inset margin 2, 9 px wide | 1.14 |
| single bar, inset margin 8, 13 px wide | 1.32 |

On the geometry we actually have, clDice punishes a severing gap **less** than
plain Dice does. Two mechanisms, both structural rather than tunable.

**The sensitivity term counts centreline pixels, not connectivity.**
`sum(S_true · V_pred) / sum(S_true)` asks how much of the true centreline the
prediction still covers. A 9 px cut removes 9 px of true centreline whether it
severs a block mid-run or merely trims a dead end — measured identical to five
decimals. Whether the cut disconnects the network never enters the numerator.
The part that can distinguish them is end-retraction in the *precision* term,
where a severing cut creates two new ends that erode back by the road
half-width: skeleton sums 550 (severed) against 554 (trimmed) out of 567, a
0.7% effect that the sensitivity term swamps.

**Roads leaving the tile never retract.** `max_pool2d` pads with `-inf`, so
erosion (`-maxpool(-x)`) treats out-of-frame as foreground. A road running off
the chip keeps its skeleton to the border and loses no length at that end. Our
chips are ~390 m squares whose roads leave on all four sides, so this is the
common case, not the exception.

### The knobs do not move it

| eps | iterations 5 | 10 | 20 |
|---|---|---|---|
| 1.0 | 0.871 | 0.871 | 0.871 |
| 1e-2 | 0.872 | 0.872 | 0.872 |
| 1e-4 | 0.872 | 0.872 | 0.872 |

Four orders of magnitude on the smoothing term and a 4× range on the peel count
move the ratio by 0.001. The soft skeleton converges by iteration 5 for a 9 px
road, so `SKELETON_ITERATIONS = 10` is comfortably past sufficiency and raising
it buys nothing. This axis is closed: no tuning of `eps` or `iterations`
rescues clDice's behaviour *as a measurement*.

### Where it stops being a negative result

Loss value and training signal are different objects, and the above measures
only the first. Gradient mass landing inside the severing gap, which is 0.9% of
the tile's pixels:

| loss | share of \|grad\| in the gap | total \|grad\| |
|---|---|---|
| soft Dice | 0.90% | 0.0184 |
| clDice | **1.59%** | 0.0091 |

Dice is *exactly* uniform — 0.90% of gradient mass on 0.9% of pixels means it is
indifferent to where the error sits, which is the whole complaint against pixel
losses stated numerically. clDice concentrates 1.8× on the break. So it pushes
where we want even while scoring the finished state lower, and its total
magnitude is about half Dice's, which makes `cldice_weight` the real lever —
the one knob the static sweep above does not cover.

A loss that discriminates poorly between two finished states can still steer
well. Nothing measured so far settles that, so the A/B is run rather than
skipped, and no default is changed on the strength of the static result alone.

### What it motivates

Two runs at `epochs=40, patience=10` (raised from 20/5 because the dihedral
group gives 8× effective data and the baseline was still improving when
patience cut it at epoch 16): augmentation alone, then augmentation plus
`cldice_weight=0.5`. Baseline for both is the 2026-09-08 Vegas run — unchanged,
since `augment` and `cldice_weight` default off and the seam commit is a
verified no-op.

Independent of the outcome: this is a local, soft proxy that rewards
overlapping skeletons rather than connected routes, so two fragments 20 px apart
still get no gradient pulling them together, and it cannot touch the 0.9703
non-planarity ceiling, which is a representation limit rather than a loss one.
Putting topology in the objective properly is Stage 2 (Sat2Graph).

---

## 2026-09-09 — The published-baseline range on the showcase page was wrong

### What was checked

The showcase page claimed "Published SpaceNet APLS figures run roughly
0.63–0.74" and that our 0.7247 "sits inside that range." Neither the repo nor
the page cited a source for it. Checked against two primary sources:

| Source | Dataset | APLS |
|---|---|---|
| Van Etten et al., arXiv:1807.01232, Table 4 | SN3 challenge, 4-city total, top 5 | 0.628–0.6663 |
| Van Etten et al., arXiv:1807.01232, Table 4 | SN3 challenge, **Las Vegas column**, top 5 | 0.771–0.801 |
| Sat2Graph, ECCV 2020, Table 1 | SpaceNet Roads (Seg-UNet … Sat2Graph-DLA) | 0.5377–0.6443 |

### What the numbers say

The 0.63 lower bound roughly matches the four-city totals; nothing found
supports 0.74 as an upper bound. More importantly, the four-city total was never
the comparable figure: we score one AOI. Against the Las Vegas column — the only
like-for-like comparison available — our 0.7247 sits **0.046 to 0.076 below**
the top five, not inside their range.

The direction of the error matters. The old text implied a result comparable to
published work; the corrected comparison says we are below the challenge
leaders on the easiest city, and still not strictly comparable (private 20%
holdout, not the official test split).

Unchanged and still verified: agreement with the vendored reference
implementation, worst case ~3e-5 (`test_against_reference.py`, tolerance pinned
at 1e-4).

### What it motivates

Every external number quoted on a deliverable page carries its citation in the
page source from here on. The SN3 Las Vegas column is now a named constant in
`showcase.template.html` with the arXiv id and table number beside it, so the
comparison recomputes from `S.apls_mean` rather than being retyped.

`scripts/build_site_data.py` gained `--page-only`, matching
`build_apls_explainer.py`: a prose change to the showcase template no longer
costs a 40-chip inference pass.

## 2026-09-09 — Augmentation lands, clDice does not; IoU disagrees with both

Two runs against the 2026-09-08 Vegas baseline, holding everything constant but
the variable under test: 785 train tiles, 3140 crops, 392 val crops, 256 px,
batch 8, lr 1e-3, `dice_weight` 0.5, seed 0. Budget raised to `epochs=40,
patience=10` from 20/5, because the baseline's best epoch was 10 of 16 and the
curve was still descending when patience cut it.

| run | APLS mean | median | fraction of ceiling | mask IoU | best epoch |
|---|---|---|---|---|---|
| baseline | 0.7247 | 0.7595 | 0.7459 | 0.5923 | 10 of 16 |
| + augmentation | **0.7925** | **0.8620** | **0.8152** | 0.5869 | 21 of 32 |
| + augmentation + clDice 0.5 | 0.7683 | 0.8216 | 0.7915 | 0.5832 | 21 of 22 † |

Paired over the 40 common held-out tiles:

| comparison | mean Δ APLS | sem | t | 95% CI | better/worse |
|---|---|---|---|---|---|
| augmentation vs baseline | **+0.0678** | 0.0199 | **+3.40** | [+0.029, +0.107] | 26 / 14 |
| clDice on top of augmentation | −0.0242 | 0.0207 | −1.17 | [−0.065, +0.016] | 21 / 19 |
| both vs baseline | +0.0435 | 0.0217 | +2.01 | [+0.001, +0.086] | 23 / 17 |

### Augmentation

A real effect, and the largest single improvement Stage 1 has produced. The
overfitting gap it targets closed as predicted: final train/val separation went
from +0.0803 to +0.0320. The raised budget was necessary rather than incidental
— the best epoch moved from 10 to 21, so the old `epochs=20, patience=5` would
have stopped this run before it reached its own optimum. Median gains more than
mean (+0.103 against +0.068), so it is lifting the middle of the distribution
rather than rescuing a few disasters.

### The pixel metric points the other way

**Augmentation *lowered* mask IoU, 0.5923 to 0.5869, while raising APLS by
0.068.** Every previous demonstration of the IoU/APLS divergence in this project
has been constructed — the gap-punching sweep damages a perfect mask on purpose.
This one is not constructed: an ordinary training intervention moved the two
metrics in opposite directions on real held-out imagery. A team tuning on pixel
overlap would have measured augmentation as a small regression and dropped it.
That is the project's thesis observed rather than argued, and it is the more
persuasive form of the claim.

### clDice

No detectable effect at weight 0.5. The confidence interval spans zero, the
point estimate is negative, and the per-tile split is 21/19 — indistinguishable
from a coin flip. This is consistent with the 2026-09-08 static analysis, which
found the term a *worse* discriminator of a severing gap than plain Dice on
tile-spanning geometry, and it does not vindicate the gradient-concentration
result that motivated running it anyway: 1.8x gradient mass on the break did
not convert into topology the metric can see.

Two limits on how far to push this. **Only one blend weight was tested.**
clDice's total gradient magnitude is about half Dice's, so weight 0.5 replaces a
substantial share of the pixel signal to buy a weak topological one; a lower
weight might add the nudge without the dilution, and that is untested.
† **The clDice arm was truncated** at epoch 22 by an OOM kill against the
augmentation arm's 32. Both peaked at epoch 21, and the augmentation arm
improved on none of its epochs 22-32, so the truncation probably cost little —
but "probably" is doing work in that sentence, and the arms are not equal in
opportunity.

### What it motivates

Augmentation should become the default; it is measured, it is large, and the
symmetry it exploits is exact for overhead imagery. clDice stays at weight 0 and
stays in the tree: it costs nothing off, it is tested, and the weight sweep is
cheap to run if wanted.

The IoU result sharpens an existing backlog item. `TrainConfig.select_on`
defaults to `val_loss` — a pixel measure — and this run is direct evidence that
pixel measures and APLS disagree about which model is better *on real training
decisions*, not merely on synthetic damage. Selecting the epoch on a cheap APLS
proxy is now better motivated than it was when it was filed.

---

## 2026-09-09 (later) — The tuned threshold survives leave-one-out, and it crosses the Vegas leaderboard column

### Change under test

The showcase reported the checkpoint at `predict_mask`'s 0.5 default, which the
2026-09-08 sweep had already shown to be the wrong operating point. Moving the
page to the APLS-optimal 0.02 raises the headline from 0.7247 to 0.8071, so the
question is whether that 0.08 is real or an artifact of choosing the threshold
on the same 40 chips the number is reported on.

### What was measured

Threshold selection re-run as a cross-validation over the existing sweep
artifact. No inference: `outputs/threshold_sweep.json` already carries all
40 x 15 per-tile scores.

| Estimator | Threshold picked | Mean APLS |
|---|---|---|
| Untuned default | 0.50 | 0.7247 |
| Tuned on 40, scored on 40 (in-sample) | 0.02 | 0.8071 |
| **Leave-one-out: tune on 39, score the 40th** | **0.02 on all 40 folds** | **0.8071** |
| Split-half, A tunes / B scores | 0.10 | 0.7769 |
| Split-half, B tunes / A scores | 0.02 | 0.8048 |

**The selection buys itself nothing measurable.** Every LOO fold picks 0.02 and
the held-out mean matches the in-sample figure to four decimals. One scalar
chosen over 40 samples is not enough freedom to overfit here. Split-half is
noisier and one half picks 0.10, which is the expected behavior with 20 chips
of evidence rather than 39, not a contradiction.

**This corrects the earlier caveat.** The first write-up of this sweep said the
direction of the finding survived but the magnitude did not. The LOO estimate
says the magnitude survives too.

### The consequence nobody asked for

At 0.8071 the result crosses the SpaceNet 3 Las Vegas column (0.771-0.801,
Van Etten et al. 2018, Table 4) rather than sitting below it. That makes the
comparability caveat more load-bearing, not less: the split is a private 20%
holdout rather than the official test split, and challenge entrants could not
tune a threshold against the test set at all. 0.7247 at the untuned default is
the like-for-like figure, and the page now says so explicitly.

### What it motivates

`predict_mask`'s own default is still 0.5, and `train.py` still evaluates there,
so the run artifact and the showcase now disagree by construction. Either the
training-time eval threshold becomes a config field, or the default moves. That
is a change to the pipeline rather than to a page, so it is filed rather than
made here.

`scripts/build_site_data.py` gained `--threshold`, defaulting to 0.02, and
re-scores every held-out chip rather than reading the run artifact's numbers.
Ceilings still come from the artifact untouched: a perfect mask does not pass
through `predict_mask`, so no threshold can move them.

---

## 2026-09-09 (later still) — A disjoint holdout removes the leak, and exposes a degenerate chip

Supersedes the leave-one-out entry above as the basis for the reported number.
LOO was the right check given one scored set; it turned out a second set was
sitting unused.

### The data that was already there

`vegas_best.json` carries 196 validation chips, and the run's own eval scored
only the first 40 (`apls_eval_tiles`). The other 156 had never been touched by
anything. Selecting the threshold on the 40 and reporting on the 156 removes the
leak outright, with no new data and about four minutes of inference.

`threshold_sweep.py` gained `--skip-tiles` so the two slices can be addressed
separately; `build_site_data.py` gained `--skip-chips` / `--n-scored` and now
re-scores chips end to end rather than reading the run artifact's numbers.

### What was measured, on 155 held-out chips

| Threshold | Where it came from | APLS |
|---|---|---|
| 0.50 | `predict_mask` default, untuned | 0.7550 |
| **0.02** | **swept on the disjoint 40** | **0.7976** |
| 0.03 | the holdout's own optimum | 0.8016 |

**The transfer costs 0.0040.** Importing a threshold chosen on a different 40
chips gives up four thousandths against tuning on the holdout directly. The
in-sample figure on the 40 was 0.8071; the 0.01 above the holdout is the two
chip sets differing in difficulty, not the threshold overfitting.

**The core finding replicates on 3.9x the data.** IoU still peaks at 0.30 and
APLS at 0.03. `gt→prop` still falls monotonically across the whole sweep, 0.840
to 0.620, while `prop→gt` stays between 0.788 and 0.886.

### The degenerate chip

`img1612` has a **ceiling of 0.0**: a perfect mask, traced back, scores zero
against its own truth graph. It has edges, so the existing `number_of_edges()`
guard did not catch it. Averaged in, it drags the mean down while measuring
nothing about the model, and its `fraction_of_ceiling` is 0/0. Both scorers now
skip zero-ceiling chips and say so. Excluding it moves the reported mean from
0.7925 to 0.7976.

`img1119` is the opposite case and stays in: ceiling 0.9865, model IoU 0.0000,
APLS 0.0000. The model found no road pixels at all on that chip. That is a real
failure and belongs in the distribution.

The 40-chip eval never hit either case, which is the argument for scoring the
whole split rather than a prefix of it.

### What it motivates

The reported figure of 0.7976 now sits inside the SpaceNet 3 Las Vegas column
(0.771-0.801, Van Etten et al. 2018, Table 4) rather than above it, which is a
more defensible place for a baseline with one hyperparameter tuned. The
comparison is still not a leaderboard placing: private holdout, not the official
test split.

Root cause of the zero ceiling is unexamined. Likely every edge is shorter than
`min_path_length`, so no control-point pair survives, but that is a guess and
not a measurement.

---

## 2026-09-10 — Gap closing measured out, three ways

### Change under test

Whether a learned link predictor reconnecting severed roads is worth building.
The pipeline has no repair operation at all: `prune_spurs` removes and
`snap_junctions` merges, and nothing ever adds an edge. The motivating fact was
that at the tuned threshold `gt_to_prop` is pinned near 0.823 and the mask
threshold cannot move it — dropping 0.02 to 0.01 buys +0.0041 `gt_to_prop` and
costs −0.0553 `prop_to_gt`, so the mask is saturated and further pixels are
noise.

No model was built. Four measurements were made first, and they say not to.

### The candidate pool exists

`scripts/endpoint_census.py`, 981 chips against the frozen `vegas_best.pt` at
threshold 0.02. Two gates were fixed before the numbers existed.

| gate | rule | measured at `label_snap` 10 |
|---|---|---|
| pool size | >2,000 comfortable, 500–2,000 workable | 1,529 at R=25, 2,040 at R=60 |
| distribution shift | val rate within 1.5x of train | 0.684 against 0.691 |

Both pass, the first narrowly. Train and val positive rates are
indistinguishable at R>=25, so the predictor could train on in-fold proposals
without out-of-fold generation; the shift that exists is in candidate *count*
(val chips carry 1.80 more interior endpoints, p=0.015), not in label mix.

Three structural findings. `clean()` destroys **47.0%** of train interior
endpoints before the shipped decoder emits its graph, because its first
`prune_spurs(20)` removes exactly the stubs a severed road leaves. Endpoint-edge
candidates subsume endpoint-endpoint ones for coverage in all eight
split-and-radius cells, which is geometric rather than incidental: if another
endpoint lies within R, its incident edge does too. And **47% of candidate
endpoints have a stub that never touches the label mask**.

### A snap sweep cannot validate labels, and the reason is structural

The labeller projects both candidate ends onto the truth graph with
`inject_points(truth, [a, b], max_dist=label_snap)`. Sweeping `label_snap` was
intended to test whether positives were snapping artifacts. It cannot.
`query_nearest` returns the globally nearest edge whenever it falls inside the
radius, and `LineString.project` does not consult the radius at all, so
`truth_dist` is bit-identical across radii whenever both ends land. A positive
can become undeterminable but never negative.

Confirmed empirically: **0 of 41,368 positive-instances flipped to negative** in
any split, radius or target cell. That column is a property of the code, not
evidence about label quality. Positive sets nest, so `label_snap=10` positives
are a strict subset of `label_snap=25` positives.

### Stub purity does discriminate, and the answer is bad

Fraction of a candidate endpoint's incident polyline lying on the label mask,
sampled at 1 px. At R=60, train:

| population | high purity (>=0.5) |
|---|---|
| all candidate endpoints | 23.0% |
| positives at `label_snap` 25 | 17.7% |
| positives at `label_snap` 15 | 26.3% |
| positives at `label_snap` 10 | 39.1% |

At the loose radius **the positive class is less likely to lie on labelled road
than the pool it was drawn from**. Snap 10 is the only setting that inverts
this, which is why it is the committed value; the cost is that two thirds of
candidates become undeterminable there.

Two controls say the metric reads the graph correctly rather than being broken.
Non-stub edges average 0.749 purity against 0.394 for stub edges. And dilating
the truth mask to a 17 px band still leaves ~41% of candidate endpoints missing
the road entirely, so the mass at zero is not a mask-width artifact — though the
middle of the distribution is dilation-sensitive, so 0.5 is a reporting
threshold rather than a calibrated one.

### The ceiling: +0.0224 APLS

`scripts/link_oracle.py`, six arms on the 155-chip reporting holdout at
threshold 0.02. Arm 0 reproduces `outputs/threshold_holdout.json` to **exact
float equality** (0.7976468051731842 and both directional scores), so the deltas
are trustworthy.

| arm | edges | APLS | delta vs baseline |
|---|---|---|---|
| baseline `clean()` | 0 | 0.7976 | — |
| reorder only | 0 | 0.7984 | +0.0008 (t +0.5) |
| `join@R60_snap10` | 262 | 0.8147 | +0.0170 (t +3.6) |
| **`both@R60_snap10`** | 443 | **0.8201** | **+0.0224 (t +4.5)** |
| `both@R60_snap25` | 1140 | 0.8025 | +0.0048 (t +0.7) |

95% CI [+0.0127, +0.0322], 103 chips better / 38 worse. `gt_to_prop` +0.0450
(t +5.78), `prop_to_gt` −0.0021 (t −0.63, CI spanning zero), so a perfect oracle
costs essentially nothing in the reverse direction. It recovers **12.9%** of the
0.1743 gap between the shipped decoder and a perfect mask.

**The decoder reorder is not worth making.** +0.0008, CI [−0.0025, +0.0040]. It
decomposes into `gt_to_prop` +0.0055 (t +2.31) and `prop_to_gt` −0.0042
(t −3.00): both halves individually resolvable, and they cancel. Dropping the
first prune walks the same trade curve the mask threshold already walks. That
closes the question the census's 47% destruction rate opened.

**The `label_snap` 25 positive class is affirmatively bad, not merely noisy.**
Three of its arms score at or below doing nothing (`join@R25_snap25` 0.7933,
`both@R25_snap25` 0.7958, against a 0.7976 baseline). Paired at R60, snap 25
minus snap 10 is −0.0176 APLS (t −3.48), spending 0.0333 of `prop_to_gt` to buy
0.0075 of `gt_to_prop`. An oracle scoring worse than inaction is the cleanest
available verdict on a label set.

One prediction failed: the shortcut class was expected to trade the two
directions against each other. At snap 10 it improves both. The class that
trades directions is snap-25 labelling, whichever positive kind it is applied to.

### The stubs are invented, not unlabelled roads

`scripts/osm_crosscheck.py` against a pinned Geofabrik extract
(`nevada-latest.osm.pbf`, MD5 `5c750d8e270510e12dce81711c201491`), 155 chips.

Step 0 gates on registration before anything else runs: OSM `drive` length near
a SpaceNet way is mean 0.897 / median 0.995 at 8 px, against a diagonally
shifted null control at mean 0.225. A 4x separation, so the frames genuinely
register rather than both merely being dense with roads.

SpaceNet is not badly under-labelled. Median per-chip ratio of OSM drivable to
SpaceNet is **1.064**, and 97.5% of SpaceNet road length lies within 8 px of
some OSM way. The surplus of `all` over `drive` is 60% `service` and 30%
`footway`, which is why a third OSM network (`drivable` = `all` minus pedestrian
classes) is carried: comparing against `all` would substantially be comparing
against sidewalks.

**The deciding number.** Of the 903 endpoints with zero purity against SpaceNet:

| measured against | reach >=0.5 | any purity |
|---|---|---|
| OSM `drive` | 11 (1.2%) | 27 (3.0%) |
| OSM `drivable` | 101 (11.2%) | 197 (21.8%) |
| OSM `all` | 118 (13.1%) | 226 (25.0%) |

Roughly **seven in eight zero-purity stubs are not on a way OSM maps either** —
not a service road, not an alley, not a footpath. The corroborating fact: OSM
adds 170 km of service roads across these chips and the endpoint zero-purity
fraction falls only from 0.491 to 0.465. The extra mapped length is not where
the stubs are.

So `prop_to_gt` has been penalising the model mostly for real invention, and the
oracle ceiling was not measured against materially incomplete truth. A genuine
11–13% correction is owed, higher (19.6%) in the positive class specifically.

The temporal caveat points the same way: OSM read in 2026 against 2015–2017
imagery inflates OSM's length, so the real gap of that era is smaller than 1.064.

### The stop is a property of gap closing, not of APLS

APLS scores routes, so an edge counts in proportion to how many shortest paths
cross it. The project's objective has moved toward a recognisable map, which
weights presence and attachment instead. `buffer_length_prf` and `junction_prf`
were added alongside `apls`, which is untouched — all 2,480 chip-arm APLS values
reproduce bit-for-bit.

As a share of each metric's own reachable headroom, `both@R60_snap10`:

| metric | baseline | ceiling | delta | share of headroom |
|---|---|---|---|---|
| APLS | 0.7976 | 0.9720 | +0.0224 | **+12.8%** |
| buffer F1 @10 | 0.9192 | 0.9947 | +0.0054 | +7.2% |
| buffer F1 @5 | 0.8910 | 0.9926 | −0.0002 | −0.2% |
| junction F1 @10 | 0.7091 | 0.8814 | −0.0240 | **−13.9%** |

Coverage weighting shrinks the gain; attachment weighting reverses it. The one
framing under which it looks better than APLS says is buffer *recall* alone
(+0.0100, t +4.24), which refuses to charge for invented road.

**The mechanism, measured.** Per chip the oracle adds 2.67 junctions (t +7.44)
of which only 0.49 match a truth junction (t +5.57), so **82% of the junctions
it creates are not on the ground**. Proposal junction count goes from 105% of
truth's to 121%. Junction precision falls 0.0655 and is flat across matching
radius (−0.062 / −0.066 / −0.064 at 5/10/20 px), so these are extra junctions
rather than near-misses. The cause is structural: an endpoint-to-edge connector
splits the target edge and creates a degree-3 node where the truth road simply
continues. APLS charges nothing for it.

The geometry agrees: 82.7 m of road drawn per chip buys ~22.5 m of newly covered
truth length, and buffer precision at 5 m falls 0.0095 (t −6.41) while at 10 m
it is flat. The straight chords are roughly in the right place, off the
centreline.

**The heavy tail does not flatten.** Zero-gain chips stay at 43–48% under every
metric, and concentration gets worse: top-5 share of gain 0.34 → 0.42 → 0.54,
Gini 0.83 → 0.89 as the buffer widens.

### What it motivates

Gap closing is not worth building as a learned stage. The ceiling is real
(t +4.53) but small; the A/B/C/D ablation ladder cannot resolve its arms against
a paired sem of 0.005, so the experiment would return "no detectable difference"
whatever the truth; and on the attachment axis the technique is actively
harmful.

One decision the suite does change: filtering candidates on stub purity looked
like a pure loss of 0.0035 under APLS, and across the suite it keeps 84% of the
APLS gain at 28% of the junction damage using 188 edges instead of 443. It joins
the Pareto frontier. A purity-filtered heuristic gap closer is a modest positive
that costs no model.

Per-edge marginals say the metrics disagree about *which* edge to add: Pearson
+0.22 between APLS and buffer F1@10, and 11 of APLS's top 50 edges are in
coverage's top 50. Component joins help junction F1 (+0.0035/edge) while
shortcuts hurt it (−0.0077/edge, negative on 113 of 224) — a distinction APLS
cannot express, since it scores both positive.

**The number this opens.** Junction F1's *decoder* ceiling is 0.8814, so
`skeletonize → simplify → snap(8 px)` destroys 18% of truth junctions from a
perfect mask. The comparable APLS figure is 0.0280. `cleanup.clean`'s three
constants have only ever been evaluated against APLS, which is nearly blind to
what they do. Sweeping them against `junction_prf` is the cheapest unaddressed
measurement on the board.

### Methodological notes

`apls_uniform` contributed nothing: +0.976 per-edge correlation with `apls` and
within 0.001 on every arm delta. Same metric sampled denser, not an independent
check. `buffer_f1_20` rewards indiscriminate edge addition — at 20 m the
dirtiest arm (1,140 edges) has the best delta while doing the worst junction
damage — so report 5 and 10.

The metric suite was added immediately after APLS returned an unwelcome answer
on the same holdout, by the same process, with the same arms. `BACKLOG.md` has
asked for a local metric since 2026-09-01, 21 commits before the oracle ran, and
the objective genuinely moved from routing to a recognisable map. Neither
defence is decisive alone. The risk did not bite here only because the new
metrics returned a *worse* answer than the one they followed.

Overpass rate-limited the crosscheck out of service for about two hours, which
is why OSM now comes from a pinned extract. The deeper reason to keep it there:
an Overpass query cannot satisfy the content-hashed run-identity rule, because
the database changes under it. Overpass-versus-pbf agreement was never actually
validated and remains open; `ox.graph_from_bbox` defaults to `retain_all=False`,
so the Overpass path returns only the largest weakly connected component, and
that divergence was deliberately not replicated.

---

## 2026-09-10 (later) — Two sweeps into the decoder, and what APLS cannot see

Both follow from the previous entry's finding that the decoder's own loss is
0.0280 of APLS but 0.1186 of junction F1. Neither retrains anything.

### `cleanup.clean`'s constants: half the junction loss is one badly chosen number

`scripts/cleanup_sweep.py`, 4x4x4 factorial over `simplify_tolerance`,
`spur_length` and `snap_tolerance`, both arenas (model proposal and perfect-mask
ceiling) at every grid point, 155-chip holdout, 22.5 min. The shipped row
reproduces `link_oracle.json`'s baseline to the digit, and the pipeline-prefix
sharing that makes the grid affordable was proven against `cleanup.clean` on 384
(chip, arena, grid point) triples with exact geometry fingerprints, 0 mismatches.

**`spur_length` is the entire effect.** 1-D spans through the shipped setting:

| axis | model APLS | model jF1@10 | ceiling jF1@10 |
|---|---|---|---|
| `simplify_tolerance` | 0.0070 | 0.0018 | 0.0004 |
| `spur_length` | **0.0624** | **0.0809** | **0.1656** |
| `snap_tolerance` | 0.0102 | 0.0112 | 0.0312 |

`simplify_tolerance` is inert. The 8 px snap radius, which looked like the
obvious culprit, costs the ceiling 0.0312 against the 20 m spur rule's 0.1656.

**The precision/recall split is the mechanism.** `spur_length` 40 down to 5:

| arena | precision | recall |
|---|---|---|
| ceiling | 0.9782 → 0.9756 (flat) | 0.6528 → 0.8831 |
| model | 0.7939 → 0.7102 | 0.5858 → 0.7962 |

On a perfect mask pruning buys essentially no precision and costs enormous
recall: it is pure destruction, since every deleted stub takes its junction with
it as the attachment point drops from degree 3 to degree 2. On real model output
precision genuinely climbs with pruning. **`prune_spurs` is a noise filter and
its correct setting is a function of mask noise, nothing else.**

**How much of the 0.1186 is recoverable.** On the ceiling, `s0.5_p5_n4` gives
junction F1@10 +0.0618 (t +7.07, CI [+0.0445, +0.0790], 101 chips better / 10
worse), so **52% is a badly chosen constant and 48% is a genuine decoder floor**.
The share is stable across matching radius: 47.7% at 5 px, 52.1% at 10, 56.2% at
20.

**On the actual model that gain does not materialise.** Best model-arena junction
F1 is +0.0075 at t +0.61, CI spanning zero. Tightening the constants lets
skeleton noise through and the false junctions cancel the real ones recovered.
The extra stubs the model keeps are largely invented, which the OSM crosscheck
already established.

**Retuning widens the model-to-ceiling gap rather than closing it**: +0.1723 at
the shipped setting, +0.1988 at `s2_p10_n8`, +0.2073 at `s0.5_p10_n4`, because
the ceiling rises faster than the model does. After retuning, junction loss is
*more* a model problem and *less* a decoder problem. That points at Stage 2 or a
better segmentation model, not at further decoder tuning.

**A free APLS gain is sitting there.** `s0.5_p10_n4` is +0.0120 APLS (t +2.96,
CI [+0.0040, +0.0201]) with junction F1 flat. Moving `spur_length` 20 to 10 and
nothing else is +0.0054 APLS and +0.0070 junction F1 together. For scale, the
entire gap-closing oracle ceiling was +0.0224, so **over half of it is available
by changing one number**, with no model.

**Two caveats that keep this from being a simple win.** The domination is
matching-radius-specific: at junction radius 5 the shipped setting is the
model-arena *maximum*, rank 1 of 64, and none of the eight configs that dominate
it at radius 10 beats it at radius 5. And the shipped setting sits on the
junction precision/recall frontier at its high-precision end (jP 0.7693 /
jR 0.7148), so if precision is the goal, F1 is the wrong referee. Also: spur 5 is
the grid's low end *and* the ceiling optimum, so the true optimum may lie below
it and is not bracketed. Settle that with a short run at spur 2 and 3 before
committing a default.

Do not read the simplify axis on APLS: control-point counts move with it (the
ceiling carries 53.0 proposal control points at 0.5 against 41.4 at 1.0), so the
estimator is not held fixed along that axis. The effect is 0.004 and no single
mechanism covers both arenas.

### `max_snap`: 19% of the reported score is snapping slack

`scripts/snap_sweep.py`, `max_snap` in {5, 10, 15, 25} on the same holdout, 57 s.
`metrics.apls`'s default is unchanged at 25.0 and every score at 25 reproduces
`link_oracle.json` bit-for-bit; `APLSResult` gained landing and pair counts as
additive fields with defaults, and the reference test still passes at 1e-4.

The module docstring already said displacement below `max_snap` is deliberately
invisible. This prices it.

| arena | snap 25 | snap 5 | survives |
|---|---|---|---|
| proposal | 0.7976 | 0.6453 | **80.9%** |
| ceiling | 0.9720 | 0.9315 | 95.8% |

**The unmatched-versus-displacement split this was designed around is not a real
distinction.** A pair whose endpoints both land keeps exactly the `l_b` it had,
because splitting an edge preserves path length and the nearest edge does not
change with the radius. A smaller radius can only refuse to land points. The
unlanded term therefore exceeds 100% of the drop (proposal gt→prop: −0.1539 =
unlanded +0.2088, nopath −0.0297, length −0.0252), the other terms falling as
fewer pairs survive to be compared. Displacement is the cause; unmatched points
are the whole mechanism.

**The number that answers the question is the landing-rate difference**, which
isolates control points that are present but off the line:

| | present, 5–25 m off | never within 25 m |
|---|---|---|
| proposal gt→prop | **13.4%** | 2.9% |
| proposal prop→gt | 12.9% | 7.3% |
| ceiling gt→prop | 5.7% | 1.2% |
| ceiling prop→gt | **0.0%** | 0.0% |

Radius is not a tradeoff axis: the Pareto frontier is the single point 25.0 in
both arenas with zero monotonicity violations, which follows from
`inject_points` taking the nearest edge inside the radius.

**Containment is one-way and exact.** The ceiling's `prop_to_gt` is flat at every
radius with 6320/6320 landing at 5 m, because every ceiling edge is traced from a
mask rasterized from the truth graph. The reverse fails at 5.7%, because
`prune_spurs(20)` deletes truth stubs and `snap_junctions(8)` moves truth
junctions. The proposal arena is nearly symmetric instead, since the model both
misses and invents.

**The buffer cross-check disagrees, and that is the finding.** At 5 m the two
measures differ by 5.6–8.8 points on the same chips, buffer always the more
forgiving, gap widening as the radius shrinks. The cause is measurable: **78.2%
of truth control points are the graph's own nodes**, because reference sampling
places no interior control point on a straight edge. APLS's control points
cluster at junctions and dead ends; buffer coverage is length-weighted and
dominated by accurate straight middles. The ceiling pins it — 1.3% of truth
*length* lies beyond 5 m of ceiling road against 6.9% of truth *control points*,
a factor of 5.

So under reference sampling on a grid city, **APLS is closer to a
junction-and-endpoint metric than a length metric**. That is a structural
property of the estimator, and it connects to the existing note that straight
edges receive no interior control points.

Not measured, and the direct test of that explanation: landing rate split by node
versus interior control point. One field away, `densify` would have to report
which nodes it added. The alternative it cannot rule out is that the gap is truth
roads wholly absent rather than displaced.

### The ceiling is not reachable even in principle

Ceiling `prop_to_gt` carries a `nopath` loss of **0.0069 at every radius,
including 25**: both endpoints land, on the truth graph, and the truth graph
cannot route between them. That is truth-graph disconnection rather than decoder
error, and every "fraction of ceiling" figure in this project inherits it as a
floor.

Six proposal chips zero out at `max_snap` 5 (img1638 has 6 truth control points
and 3 scored pairs, so losing 2 landings takes a direction to 0 and the harmonic
mean with it). The median, 0.6978, is the more robust read at that radius.

---

## 2026-09-10 — What a road-class ablation could resolve, before training one

**Change under test.** Nothing was trained. The attribute-inference project's
first question is the one gap closing failed: can a two-arm comparison on this
data separate its arms at all? `scripts/attr_resolvability.py` answers it from
counting and simulation, at a cost of 44 seconds.

**What was measured.** Nodes placed along the OSM `drive` network in all 981
chips (785 train, 196 val), labelled with the way's `highway` class folded into
six: motorway, primary, secondary, tertiary, residential, unclassified. Links
join their parent. A model at a stated per-unit accuracy is simulated — correct
with probability `a`, otherwise predicting a class drawn from the label prior
excluding the true one — its macro-F1 is bootstrapped over chips, and the
smallest difference a comparison could call at 80% power and alpha 0.05 follows
from the standard error. Node spacing (5, 10, 20, 40 m), accuracy (0.5, 0.7,
0.9), the error-correlation regime and the between-arm correlation are swept.

The statistical core was checked against independent implementations before the
numbers were read: `_z` against `scipy.stats.norm.ppf` (max error 2.7e-15),
`macro_f1` against a per-class loop (5.6e-17), the chip bootstrap against a
jackknife (ratio 0.985 at 196 chips, 0.992 at 785), and identical chips against
the exact answer of zero. The headline reproduces to within 3% at a second seed.

### Node spacing does not buy statistical power

On the val split at accuracy 0.7, node regime:

| spacing | nodes | sem(macro-F1) |
|---|---|---|
| 5 m | 56,901 | 0.0120 |
| 10 m | 28,552 | 0.0121 |
| 20 m | 14,487 | 0.0126 |
| 40 m | 7,632 | 0.0132 |

A 7.5-fold change in node count moves the noise floor by 10%. The reason is that
the bootstrap resamples **chips**, and the chip count is fixed at 196: the
variance is dominated by which chips were drawn and how their class mix differs,
not by how densely each was sampled. This holds even in the regime where every
node's error is drawn independently, which is the most favourable case density
could have had.

So the spacing decision is free on statistical grounds and should be made on
compute: **20 m stands**, and 10 m would quadruple the node count for a 4%
narrower interval.

### The correlation regime matters 2.3x more than the spacing

Errors drawn per node treat a 400 m street as 20 observations; drawn per OSM way
they treat it as one. At val, 20 m, accuracy 0.7, sem is 0.0126 per node, 0.0185
per way, 0.0289 per same-class component. The way and component unit counts are
constant in spacing by construction — 2,453 ways and 791 components on val, at
every density — which is the mechanism behind the table above.

### What the comparison would need to show

Minimum detectable macro-F1 difference at 20 m spacing, accuracy 0.7:

| scored on | regime | unpaired | rho=0.8 | rho=0.9 |
|---|---|---|---|---|
| val, 196 chips | node | 0.0498 | 0.0223 | 0.0158 |
| val, 196 chips | way | 0.0731 | 0.0327 | 0.0231 |
| val, 196 chips | component | 0.1144 | 0.0512 | 0.0362 |
| train-sized, 785 chips | node | 0.0239 | 0.0107 | 0.0076 |
| train-sized, 785 chips | way | 0.0358 | 0.0160 | 0.0113 |
| train-sized, 785 chips | component | 0.0572 | 0.0256 | 0.0181 |

`rho` is the between-arm correlation, which is a property of two models that do
not exist yet, so it is tabulated rather than measured. Two seeds of one
architecture sit near 0.9; two unrelated models near 0.

Read on the val split as it stands, an ablation needs **2 to 5 macro-F1 points**
under strong pairing, and 5 to 11 unpaired. Read on 785 chips it needs 1 to 3.
That is the whole design constraint, and it is about the size of the evaluation
set, not about the node sampling.

### Three preconditions cleared, one problem found

**The metric is not trivially satisfiable.** Residential is 54.8% of val nodes,
so always predicting it scores 0.548 accuracy and **0.118 macro-F1**. Macro-F1
is doing the work accuracy would not.

**The vocabulary covers the data.** Zero nodes fell outside the six classes: the
`drive` filter over Las Vegas admits nothing else. No trunk roads exist here, so
that class is defined in the map and absent from the vocabulary; another AOI
would need it back.

**The rare class is thin but not empty.** Unclassified is 2.7–3.5% of nodes: 505
nodes on 109 ways and 46 components in val, 1,669 nodes on 361 ways and 193
components in train. Under the component regime the rarest class carries about
42 independent units on val, and one sixth of the macro average rests on them.

**876 OSM ways reach both splits** — 35.7% of val's 2,453 ways also appear in a
train chip. The existing split was drawn over chips at random, and SpaceNet
chips are adjacent tiles, so a street cut by the split is seen from both sides.
For segmentation this leaks texture. For attribute inference the label is a
*property of the way and constant along it*, so it leaks the answer: a model
that memorises "way 12345 is tertiary" collects it again at evaluation. This is
the one finding that changes the project's design rather than confirming a
default.

**Next.** The remaining Tier 0 checks are untouched and neither is answered by
counting: whether the class is predictable from imagery at all, and whether a
topology-only baseline already gets it. This run says only that if a difference
exists and exceeds a few macro-F1 points, the data can show it.

---

## 2026-09-10 — Splitting by ground position, and the bill for it

**Change under test.** `geo_graphs/split.py` and `scripts/split_sweep.py`.
Thirteen candidate train/val splits over the 981 Las Vegas chips, each drawn at
five seeds, scored on leakage, adjacency, class mix and the validation side's
macro-F1 noise floor. Nothing trained.

**Cross-check first.** The sweep's `random` draw at seed 0 reproduces the ids
and order of the shipped split exactly, and reports 876 shared ways of 2,453 —
the same figure `attr_resolvability` reached by different code earlier today.

### The random split is worse than the way count said

45.4% of validation nodes lie on an OSM way that also touches a training chip
(41.6–48.8% across five draws), against 35.7% of *ways*. Shared ways are the
long ones, so counting ways understates the share of the evaluation set whose
answer is already visible. By length the figure is the same 45%.

### Blocking helps and does not fix it

Mean over five draws, at 20 m node spacing:

| candidate | train | leaked val nodes | sem(macro-F1) | max class drift | rare-class ways |
|---|---|---|---|---|---|
| random | 785 | 0.4542 | 0.0187 | 0.0139 | 97 |
| blocked-640 | 784 | 0.3321 | 0.0200 | 0.0321 | 79 |
| blocked-1280 | 779 | 0.1635 | 0.0196 | 0.0333 | 87 |
| blocked-1920 | 779 | 0.1442 | 0.0221 | 0.0544 | 57 |
| blocked-2560 | 767 | 0.0865 | 0.0211 | 0.0391 | 73 |
| buffered-2560+500 | 723 | 0.0377 | 0.0217 | 0.0391 | 73 |
| buffered-2560+1000 | 589 | 0.0125 | 0.0212 | 0.0391 | 73 |
| buffered-640+1000 | 185 | 0.0109 | 0.0201 | 0.0321 | 79 |

Those eight are the Pareto frontier on leakage against training chips; the other
five candidates are dominated. Blocking alone bottoms out around 9% leakage
because roads cross block boundaries, and 10.9–20.7% of validation chips still
share a border with a training chip at 640–1920 m. Only a buffer takes the
touching fraction to zero.

**Larger blocks buffer more cheaply.** A validation set of many small squares has
far more perimeter than one of a few large squares, so the same 1000 m margin
costs 599 training chips at 640 m blocks and 178 at 2560 m. That is the whole
reason `buffered-2560+1000` sits on the frontier and `buffered-1280+1000` does
not.

**Block size is not monotonic.** 1920 m is worse than 2560 m on noise (0.0221 vs
0.0211), class drift (0.0544 vs 0.0391) and rare-class ways (57 vs 73), despite
cutting the AOI into 62 blocks rather than 35. The AOI is 14.1 by 15.3 km, so
how a block grid lands on it matters as much as how fine it is.

### What blocking costs: motorway

Validation class share, mean over five draws:

| | motorway | primary | secondary | tertiary | residential | unclassified |
|---|---|---|---|---|---|---|
| whole AOI | 0.0604 | 0.1444 | 0.1094 | 0.1116 | 0.5458 | 0.0284 |
| random | 0.0548 | 0.1440 | 0.1174 | 0.1059 | 0.5474 | 0.0304 |
| blocked-2560 | 0.0311 | 0.1494 | 0.1223 | 0.1083 | 0.5521 | 0.0368 |

Almost all the drift is motorway: 6.0% of the AOI, 3.1% of a blocked validation
set. Motorways are few, long and linear, so a block either contains one or does
not, and the class that already had the fewest independent units loses a third
of them. Rare-class ways fall from 97 to 73.

### Evaluation noise barely moves

sem(macro-F1) under the way regime goes from 0.0187 (random) to 0.0212
(`buffered-2560+1000`), a 13% rise; the minimum detectable difference at rho 0.9
goes from 0.0234 to 0.0265. Blocking costs almost nothing in power, because the
validation *chip count* barely changes and that is what the bootstrap resamples.
The cost is training chips and motorway coverage, not error bars.

### The frontier, and the two contenders

`buffered-2560+500` cuts leakage 12-fold, from 0.454 to 0.038, for 62 training
chips (8%). `buffered-2560+1000` cuts it 36-fold, to 0.013, for 196 chips (25%).
Beyond that the trade collapses: `buffered-640+1000` buys another 0.0016 of
leakage for a further 404 chips.

Which of the two is right depends on whether 4% memorised validation is
tolerable, and that is a judgement about the claim being made, not about the
data. Both are on the frontier. The seed matters too and has to be recorded: at
2560 m the AOI is 35 blocks and 7 are held out, so a single draw's class mix
ranges over 0.026–0.062 in max drift.

**Not done.** `train.assign_split` defaults to `random`, so `vegas_best.pt` and
every number measured on it keep their provenance. Retraining segmentation on a
spatial split is a separate decision with its own cost, and it would invalidate
the existing APLS numbers' comparability.
