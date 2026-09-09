# Multimodal road network extraction: directions

A proposal for continuing the road-graph work beyond single-modality satellite
imagery. Covers the data that exists, the methods worth trying, and an ordering
for the next few pieces of work.

## Where this stands

The current pipeline is imagery-only: satellite RGB → segmentation mask →
skeleton → cleanup → graph, scored by APLS against a truth graph. Three
properties of that pipeline shape everything below.

- **APLS is a connectivity metric, and the pixel losses are not.** The measured
  threshold sweep puts the APLS optimum at 0.02 and the IoU optimum at 0.30 —
  the two objectives disagree about the operating point by more than an order of
  magnitude. Any modality that closes gaps will show up far more strongly in
  APLS than in IoU.
- **The harness has an intrinsic ~0.90 ceiling from non-planarity.** The truth
  graph has 20 edge pairs that cross with no shared node; a 2D mask cannot
  represent a road passing over another without inventing a junction. This is a
  representation limit, not a tuning one.
- **Tier 1 of the availability-conditioning plan is implemented against
  synthetic lidar.** `TileSample.aux`/`aux_valid`, `synthesize_height`,
  `CoverageSampler`, `aux_stack`, and a `FUSION_REGISTRY` with an `early`
  implementation exist and are tested; `train.py` takes `--lidar`, `--fusion`,
  and `--coverage`. No training run has been made, so the plumbing is proven and
  nothing about quality is known. See `plans/lidar_availability_conditioning.md`.

## Three problems, not one

Worth keeping separate because they have different solutions and different
failure modes.

1. **Missing modality.** Lidar in the training areas, absent or partial
   everywhere else.
2. **Domain shift.** New geography, new appearance statistics, new road
   morphology. Likely the largest term in the error budget, and independent of
   the lidar question.
3. **Active sensing.** Where a limited lidar budget goes.

The trap is treating (1) as the whole problem.

## What lidar contributes that imagery cannot

Not "another band" — a different physical measurement. The useful products are
derived rasters rather than the raw cloud.

| Product | Derivation | Road signal |
|---|---|---|
| DTM | ground-classified returns (CSF, progressive TIN densification) | locally planar, bounded slope, constant ~2% crown |
| nDSM | DSM − DTM | roads at ≈0 above ground; overpasses are not |
| Intensity | return strength at 1064 nm | asphalt dark in NIR, vegetation bright, paint retroreflects |
| Echo ratio | multi-return statistics | single return = bare pavement, multi = canopy |
| Roughness / slope / aspect | local DTM statistics | pavement smooth at a scale nothing natural matches |

Four imagery failure modes it attacks:

- **Canopy occlusion.** The dominant source of broken graphs and the one APLS
  punishes hardest — a gap under a tree line disconnects two subgraphs and costs
  every source-target pair spanning it. Last-return lidar penetrates canopy.
- **Grade separation.** From a projection, an overpass and an at-grade crossing
  are genuinely ambiguous. nDSM resolves it outright — gated on the z-aware
  graph representation, without which better height data still collapses into
  the same false junctions.
- **Illumination.** Active sensing: no sun angle, no building shadow, no
  seasonal albedo, no cloud.
- **Attributes.** Longitudinal grade and vertical curvature come free, turning a
  topological graph into a routable one.

Honest costs: coverage is patchy outside a few countries and refreshes on a
years-to-decades cycle against days for satellite; parking lots and aprons are
geometrically identical to roads under every cue above; unpaved roads have
intensity indistinguishable from bare soil. The last two are where imagery is
strong, which is the argument for fusion rather than substitution.

### One published ranking signal

A study fusing six lidar-derived channels with RGB across 23 urban and rural
areas (864 km²) reports that **slope and roughness performed best** — ahead of
DTM, hillshade, intensity, and aspect. The paper gives the ranking but not the
fusion-vs-imagery-only margin, so treat it as a channel-selection prior rather
than an effect size.

Consequence for the current code: `AUX_CHANNEL_NAMES` is `("ndsm", "intensity")`
and roughness was deferred as "an append." If that ranking generalizes, the most
useful channel is the one not yet implemented.

## Datasets

### Closest match: US3D / DFC19

Satellite imagery with matched airborne lidar and semantic labels, ~100 km² over
Jacksonville FL, Omaha NE, and Atlanta GA. WorldView-3 at 0.3–0.4 m GSD, view
angles 5–30°, distributed with nDSM alongside RGB and multispectral. Track 1 is
2,783 tiles at 1024×1024.

Three properties make it unusually well-suited:

- **Satellite, not aerial.** Most lidar-fusion datasets pair lidar with 5–10 cm
  aerial photography. US3D pairs it with 0.3 m WorldView-3 — the same sensor
  family and resolution as SpaceNet, so the current pipeline's assumptions carry
  over without rescaling.
- **The Atlanta portion overlaps SpaceNet Challenge 4**, giving an existing
  SpaceNet AOI with matched lidar.
- **It carries an elevated-road / bridge class**, which is directly the grade
  separation case the non-planarity ceiling is about.

Limitation: labels are semantic rasters, not road graphs. Either derive graphs
from the raster, or use US3D to train the fusion encoder and keep SpaceNet for
graph supervision.

### The structural gap

**No canonical road-graph extraction benchmark includes lidar.** SpaceNet 3,
DeepGlobe, CityScale, and RoadTracer are imagery-only. The recent graph-labeled
benchmarks — RoadGIE, WorldRoadSeg-360K, Global-Scale, WildRoad — are larger and
more geographically diverse but remain imagery-only.

This is the gap, and it is also the opportunity: assembling lidar + imagery +
graph labels with a topological metric is itself a contribution, and it is the
reason the APLS harness is worth more here than a segmentation benchmark.

### Assemble-it-yourself from national open data

Probably the strongest option, and it produces exactly the "few areas with
high-quality labels" this line of work assumes, at zero licensing cost.

| Region | Lidar | Imagery | Road labels |
|---|---|---|---|
| US | USGS 3DEP, QL1/QL2, COPC on S3 | NAIP 0.6 m RGB+NIR | OSM / TIGER |
| Netherlands | AHN4/AHN5, 8–10+ pts/m² | Luchtfoto 8 cm | BGT / NWB |
| England | EA national lidar, 1–2 m DTM/DSM | OS aerial | OS Open Roads |

The Netherlands stack is the highest-quality free combination available — very
dense lidar, 8 cm imagery, authoritative road vectors. The US stack has the most
area and composes with existing SpaceNet work.

These entries are from background knowledge rather than verified access checks;
confirm current availability and licensing before planning around specifics.

### Near misses, and why

- **Point-cloud semantic benchmarks** — DALES (40 km² ALS), DublinCity, LASDU,
  Hessigheim 3D. Labeled lidar, no paired overhead imagery.
- **Mobile lidar** — Toronto-3D, Paris-Lille-3D, SemanticKITTI. Ground-level
  viewpoint, wrong for overhead extraction; right for lane-level HD mapping.
- **ISPRS Vaihingen / Potsdam** — co-registered DSM and IR-R-G with an
  impervious-surface class, the classic height-fusion benchmark. Small (33 and
  38 patches) and does not separate roads from other paved surfaces.

## Methodology options

Presented as a space to measure across, not a ranking. The axes that separate
them are in **Measurement** below.

### Fusion architecture

- **A. Privileged information only.** Lidar supervises the imagery encoder
  during training — an auxiliary head regressing nDSM from RGB, or feature
  distillation from a fused teacher into an imagery-only student. Nothing enters
  the inference graph. Bounded by a sharp limitation: the cases where lidar
  helps most are exactly the cases where RGB→height prediction fails, since
  under closed canopy RGB carries no information about the surface below. What
  it actually buys is narrower and still real — the encoder learns to exploit
  *weak proxies* for geometry (the linear canopy gap over a corridor, texture
  discontinuity, visible entry and exit either side of the occlusion).
- **B. Availability-conditioned fusion.** One network; imagery always present,
  lidar as an optional side input with an explicit validity channel, trained
  under structured coverage dropout so one set of weights spans 0–100% coverage.
  Subsumes A at the zero-coverage limit and is the only option that can *use*
  sparse collected lidar. This is what Tier 1 implements.
- **C. Graph-level conflation.** Independent imagery and lidar graphs merged by
  node/edge matching. Weakest on accuracy — it can add or remove whole segments
  but never disambiguate marginal evidence. One real virtue: the lidar
  contribution is auditable edge by edge, which makes it the presentable
  ablation even if it is not the main line.

### Fusion depth, and why it is a registry stage

`early` (concatenate at the stem) and `dual` (separate encoders fused at each
decoder scale) are not equivalent under domain shift. nDSM, roughness, and slope
are physical measurements in metres — they do not shift with atmosphere, sun
angle, sensor calibration, or season. RGB shifts with all of them. Early fusion
entangles the two in the first convolution and they cannot then be regularized
differently; `dual` allows heavy photometric augmentation on the imagery branch
and none on the geometry branch.

That argument only pays off *across* areas, so it cannot be settled in-domain —
which is why both are registered and compared rather than chosen here. `dual`
will not fit the current `Fusion` Protocol, which assumes stem-time fusion; the
Protocol needs widening first.

### Domain generalization

Independent of the lidar question and probably larger. A known tension worth
resolving empirically: the alignment check measured Vegas asphalt at 0.222
against desert at 0.345, and the backlog warns that colour jitter erodes exactly
that contrast. Correct *in domain*. Across domains, that same contrast is a cue
with no reason to survive a move to a wet temperate city or anywhere with snow,
so a model leaning on it is the failure mode. Jitter plausibly costs in-area
APLS and buys transfer APLS — a tradeoff to measure on the two-axis frontier
below, not to settle by argument.

Candidates: photometric augmentation strength, instance normalization or
histogram matching in early layers, and the geometry branch as the
transfer-stable path in `dual`.

### Sparse target-area lidar has three uses

Ranked by APLS per km² collected, which inverts the obvious ordering:

1. **Pseudo-label verification.** Run the imagery model over the new area,
   generate pseudo-labels everywhere, use the lidar strips to verify or correct
   them where they overlap, then retrain on corrected pseudo-labels across the
   *whole* area. The lidar's influence escapes its own footprint — that is the
   leverage.
2. **Domain calibration.** Lidar-confirmed road pixels are a free labeled sample
   of what road looks like in the target domain's appearance statistics — enough
   to fit a normalization or run a few adaptation steps, with no human labeling.
   Much cheaper than (1) and may capture most of the gain; measure separately.
3. **Input channel at inference.** Helps only where flown. Bounded by coverage
   by construction.

(1) and (2) require the target lidar before inference rather than with it.

The risk in (1) is confirmation bias: self-training amplifies what the model
already believes, and lidar verification breaks that loop only where it covers.
Guard by holding out some strips from verification and using them purely to
audit pseudo-labels the model was confident about.

### Active collection

"Where the imagery is weakest" is the natural phrasing, but pixel-level
uncertainty is the wrong unit for a graph problem. An uncertain pixel mid-road
is worth nothing; an uncertain pixel at a gap whose closure would merge two
large components is worth a great deal, and uncertainty sampling cannot tell
them apart.

The criterion should be **expected change in the graph**. For a candidate gap
bridging components of size `a` and `b`, the number of source-target pairs whose
shortest path changes is proportional to `a·b` — cheap to compute, directly
proportional to APLS impact, and expressible with the existing control-point
machinery. Rank sites by uncertainty × topological leverage.

Two structural priors on top, marking places lidar is *irreplaceable* rather
than merely helpful: predicted corridors intersecting canopy, and candidate
grade separations (crossings where centerline width or marking continuity is
inconsistent across the junction).

Then the practical constraint: collection is corridor-shaped, not pixel-shaped.
The acquisition function must emit flight lines under a path-length budget —
submodular maximization under a routing constraint, where greedy selection gets
the standard (1−1/e) guarantee and is adequate.

If pseudo-label repair (use 1 above) is the dominant use, the acquisition
function should select for *repair value*, which ranks sites differently: a
strip through a region where the model is confidently wrong is worthless as an
inference-time input but extremely valuable as a label corrector, because the
correction propagates to every similar region.

## Measurement

**Leave-one-area-out, always.** In-area held-out tiles share sensor, season, sun
angle, and road morphology and will overstate transfer badly. With few areas the
LOO sample is small, so report per-area numbers and do not lean on the mean.

Per config, per held-out area: APLS after cleanup, `fraction_of_ceiling`, pixel
IoU, and APLS as a function of held-out coverage fraction — a **curve, not a
scalar**.

Two Pareto frontiers, reported rather than collapsed into a ranking:

- `(coverage fraction, APLS)` — how much geometry must be bought for how much
  connectivity.
- `(in-area APLS, held-out-area APLS)` — where fusion depth and augmentation
  strength trade against each other, and where a config winning in-domain can
  lose overall.

Expect the lidar gain in APLS to exceed the gain in IoU by a wide margin, since
the mechanism is canopy gaps and grade separations — small in pixel count, large
in topology cost. The divergence is itself the result.

### The experiment that makes active collection cheap

Simulate sparsity by **ablating lidar from fully-covered areas**. Complete
coverage held in a few places gives ground truth for "what would the model have
done with only this collection plan," letting acquisition strategies be compared
offline before anything is flown.

Compute the bounds explicitly: full coverage is the ceiling, random strips at
equal budget the floor. The gap between them is the entire value the acquisition
function competes for — and if it is small, the active-collection work is not
worth doing, learned for the cost of one sweep.

## Next steps

Two tracks that can run in parallel; the method track does not block on data.

### Method track

1. **Close Tier 1.** Train the same fused config at full and zero coverage and
   confirm the endpoints separate. `make coverage-endpoints` is the invocation.
   It passes a bare `TRAIN_ARGS`, so it inherits the `n_train_crops` silent
   floor — read the crop count from the startup log or the comparison is quietly
   invalid rather than failing.
2. **Promote roughness** to `AUX_CHANNEL_NAMES` ahead of `dual`. Small change,
   and the published ranking signal puts it above both currently implemented
   channels.
3. **Coverage curriculum and the first curve.** Sweep held-out coverage over
   `{0, 0.1, 0.25, 0.5, 1.0}`, emitting APLS-vs-coverage per area as JSON.
4. **Widen the `Fusion` Protocol and add `dual`.** First early-vs-dual numbers.
5. **Availability self-distillation.** Two forward passes, one with lidar and
   one masked, with a loss pulling the zero-lidar features toward the full-lidar
   features. Attacks the specific failure mode of step 3 — a model trained with
   lidar mostly present learns to lean on it and the zero-coverage path silently
   undertrains. The number it must move is the 0% end of the curve, without
   dropping the 100% end. The baseline that matters is an imagery-only model
   trained alone: conditioning is only worth its complexity if it costs nothing
   at zero coverage.
6. **Self-training with lidar-verified pseudo-labels**, once a real target area
   exists.
7. **Topological acquisition**, against the random floor and full-coverage
   ceiling from the ablation harness.

### Data track

1. **Confirm access** to US3D and to one national stack (3DEP + NAIP, or AHN).
   Verify licensing and current download paths rather than assuming.
2. **Build the pairing for one area.** Reproject, resample lidar products to the
   imagery grid, and — the step that dominates quality — **co-register**.
   Sub-pixel alignment matters more than which fusion architecture is chosen.
   Ortho satellite imagery carries residual building lean and off-nadir
   parallax; the lidar DSM is true-nadir. The clean fix is to true-orthorectify
   the imagery *using* the lidar DTM, solving alignment and parallax together.
3. **Derive graph labels** from US3D semantic rasters, or pair the national
   stack against OSM.
4. **Quantify temporal mismatch.** Lidar and imagery years will differ. Harmless
   when lidar is an input channel; actively damaging when lidar corrects labels.

### Prerequisite, orthogonal to both

**The z-aware graph representation.** `geograph` stores `pos` as `(x, y)` and
`cleanup.merge_close_nodes` fuses on planar distance, so a grade separation
cannot survive the pipeline even if something upstream produced one. Carrying z
is a prerequisite rather than a fix — it buys nothing without a height source —
but neither does a height source pay off while the container discards the
distinction. Testable on hand-built graphs with no model in the loop, and it
gates the entire overpass share of any lidar gain.

## Open questions

- **Does synthetic transfer at all?** Tier 1 results are on generated height and
  prove plumbing and relative ordering only. Whether an architecture chosen on
  synthetic data survives real lidar is untested and worth checking early on a
  small real pairing.
- **Is the active-collection gap large enough to chase?** Answered by the
  ablation harness bounds before any acquisition function is written.
- **Does distilled geometry transfer?** Option A stores lidar-derived knowledge
  in appearance-dependent features, which may transfer worse across geography
  than the geometry branch it came from. A clean ablation.
- **Do parking lots break it?** Every lidar cue treats aprons as roads. Whether
  the imagery branch suppresses them adequately, or a dedicated negative class
  is needed, is unmeasured.

## What would be new

Treat as hypotheses pending a systematic search rather than settled novelty
claims.

- Lidar fusion evaluated with a **topological** metric. Published fusion work
  reports pixel accuracy and F1; the divergence between IoU and APLS is exactly
  what fusion should exaggerate, and nobody appears to have measured it.
- **Availability-conditioned** road extraction — one model spanning the coverage
  spectrum, reported as a coverage-APLS curve rather than at endpoints.
- **Topologically-driven active lidar collection.** Acquisition on expected
  graph change rather than pixel uncertainty, evaluated by ablation against
  full-coverage ground truth.
- A road-**graph** benchmark with paired lidar, which does not currently exist.
