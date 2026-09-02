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
