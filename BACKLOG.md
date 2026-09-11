# Backlog

Deferred work, known residual issues, and scope exclusions. Check here before
proposing new architectural work.

## Evaluation harness

- **19% of the reported APLS is snapping slack, and the metric says so itself.**
  `metrics.py`'s docstring already states that displacement below `max_snap` is
  deliberately invisible. Priced 2026-09-10 (`scripts/snap_sweep.py`): the
  proposal scores 0.7976 at `max_snap` 25 and **0.6453 at 5**, the ceiling 0.9720
  and 0.9315. **13.4% of truth control points have proposal road within 25 m but
  not within 5 m**, and APLS charges nothing for them. Do not change the default:
  25.0 is what agrees with the reference to 3e-5 and what makes 0.7976 comparable
  to the published Vegas column. This is a diagnostic, on the same footing as
  `sampling="uniform"`. Note the mechanism is *only* unmatched points — a pair
  whose ends both land keeps exactly its `l_b`, since splitting preserves path
  length and the nearest edge does not change with the radius.

- **Under `"reference"` sampling, APLS is closer to a junction-and-endpoint
  metric than a length metric.** **78.2% of truth control points are the graph's
  own nodes**, because straight edges receive no interior control points. Buffer
  coverage, being length-weighted, disagrees with APLS landing rates by 5.6–8.8
  points at 5 m, always more forgiving. The ceiling pins the cause: 1.3% of truth
  *length* lies beyond 5 m of ceiling road against 6.9% of truth *control
  points*. This sharpens the sampling-density entry below from a weighting
  concern into a statement about what the metric is measuring. Untested and one
  field away: landing rate split by node versus interior control point, which
  needs `densify` to report which nodes it added.

- **The ceiling is not reachable even in principle.** Ceiling `prop_to_gt`
  carries a `nopath` loss of 0.0069 at every `max_snap` including 25 — both
  endpoints land, on the truth graph, and the truth graph cannot route between
  them. That is truth-graph disconnection, not decoder error, and every
  "fraction of ceiling" figure in this project inherits it as a floor.

- **Harness ceiling is ~0.90 APLS on a perfect mask, and it is intrinsic.**
  Asymmetric: gt→prop 0.95, prop→gt 0.82. The cause is non-planarity, not
  tuning — the truth graph has 20 edge pairs that cross with no shared node
  (13 bridges, 4 tunnels), and a 2D mask cannot represent a road passing over
  another without inventing a junction. Sweeping road width 1.5–18 m and
  resolution 0.5–2 m/px moves the score only between 0.861 and 0.901, so
  thinning does not close it. Every Stage 1 model number inherits this. Do not
  spend time tuning cleanup against it; see `EXPERIMENT_LOG.md` 2026-08-23
  (later).
- **The graph container is planar and cleanup enforces it.** `geograph` stores
  `pos` as `(x, y)` pixel coordinates and `cleanup.merge_close_nodes` fuses any
  two nodes within a planar tolerance, so a grade separation cannot survive the
  pipeline even if something upstream produced one. Note where the limit
  actually sits: the *truth* graph already represents the 20 crossing pairs
  correctly, by simply not sharing a node. It is the proposal path that
  flattens — a raster skeleton makes a 4-way junction pixel wherever two
  centerlines cross, and no 2D cue distinguishes that from a real intersection.
  So carrying z is a prerequisite, not a fix: it buys nothing until there is a
  height source (lidar nDSM gives the 4-6 m of vertical separation at an
  overpass directly) or a predictor that emits non-planar structure (Stage 2).
  Conversely, neither of those pays off while the container and cleanup discard
  the distinction, which is why this is worth doing first and separately — it is
  a representation change, testable on hand-built graphs, with no model in the
  loop. Three places move: a `z` node attribute, `merge_close_nodes` gating on
  vertical distance as well as horizontal, and the nearest-edge injection in
  `geograph.inject_points` becoming layer-aware, which is the ambiguity the
  entry below already flags. The vendored `CosmiQ/apls` scoring subset is 2D and
  must stay byte-for-byte, so a z-aware metric would be first-party code and the
  reference-agreement check would keep applying only to the 2D path. The number
  this is aimed at is the ~0.90 ceiling above, which is the one thing tuning
  cannot move.

- **Per-tile APLS rests on wildly different numbers of pairs, all weighted
  equally.** Control point counts across four sampled Vegas tiles run 14 to 113,
  so the pair counts a tile's score averages over run 91 to 6328 — a 70x spread
  that `AggregateReport` then means with equal weight. The cause is the
  `"reference"` sampling rule, which gives straight edges *no* interior control
  points at all: in a grid city most edges are straight (img540: 14 of 14,
  img63: 100 of 114), so the nominal 50 m spacing rarely fires and the count
  collapses to roughly the node count. Effective sampling density therefore
  varies 2.5x, from 42 to 105 m per control point, while nominally fixed at 50.
  Under-weighting long straight roads is deliberate per edge and matches the
  reference; equal-weighting tiles that were sampled at very different densities
  is probably not intended, and is a plausible contributor to the sd ~= 0.16
  per-tile spread seen in the 2026-09-08 sweep. Worth checking whether a
  pair-count-weighted mean, or reporting the `"uniform"` rule alongside, changes
  the ranking of any comparison already made. Do not change the default sampling
  to fix it — `"reference"` is what agrees with published numbers to 3e-5.

- **TOPO metric not implemented, and its reference provenance is unresolved.**
  The plan calls for APLS *and* TOPO; only APLS exists. Two first-party local
  metrics now sit beside it — `metrics.buffer_length_prf` weights by road length
  and `metrics.junction_prf` by junction count — but neither is TOPO and neither
  has a published reference. The house rule wants validation against one.
  Reconnaissance found that `tests/reference/` vendors only APLS scoring
  (`VENDORED` maps exactly two upstream paths) and nothing TOPO-shaped is in the
  venv or the local caches. Recalled but unverified: upstream `CosmiQ/apls`
  carries `apls/topo_metric.py` beside `apls/apls.py`. Whether the pinned
  `apls-0.1.0` sdist ships it is one network call using the URL the faithfulness
  test already pins — fetch the tarball and list its members. If it does, TOPO
  gets the APLS treatment: vendor byte-for-byte, extend `VENDORED`, adapt beside
  it. If not, the reference has to come from GitHub or from Biagioni & Eriksson
  directly, which is weaker provenance. The primitives TOPO needs already exist
  as `geograph.densify` plus `nx.single_source_dijkstra_path_length`.
- ~~**Control-point sampling differs from the reference.**~~ Done 2026-08-23.
  `sampling="reference"` is now the default and agrees to 3e-5; the dense
  `"uniform"` rule is kept as an opt-in for diagnosis. The old 0.08 test
  tolerance is now 1e-4.
- **Only the scoring subset of `CosmiQ/apls` is vendored.** Enough to validate
  scores; the geojson/GeoTIFF ingestion and plotting layers are omitted because
  they need GDAL 2.4. If we ever want to score directly against SpaceNet's own
  geojson files rather than our graphs, that gap reopens. The functions that
  *are* carried over stay byte-for-byte upstream, checked by
  `tests/reference/test_vendored_is_faithful.py`; matplotlib is a dev
  dependency solely because that faithfulness forbids editing out a `plt` call
  in an unreachable branch.
- **Nearest-edge injection is ambiguous where geometry overlaps.** A control
  point equidistant from two edges picks arbitrarily. Harmless on planar road
  networks, but it is exactly the overpass case Stage 2 exists to handle, so
  revisit when non-planar structure arrives.

## Data

- ~~**No imagery yet.**~~ Done 2026-09-01. The 0.71 GB sample is loaded through
  `spacenet.SpaceNetTileSource`, reprojected to UTM and noded. Ten chips per AOI.
- ~~**Only the sample is downloaded.**~~ Done. The full 24 GB Vegas AOI is
  extracted at `data/AOI_2_Vegas`: 989 chips, 981 with labels, split 785 train
  / 196 val. Paris, Shanghai and Khartoum exist only as 10-chip samples and are
  separate downloads.

- **The shipped split leaks whole roads; the replacement exists and is not the
  default.** 45.4% of validation *nodes* lie on an OSM way that also touches a
  training chip (876 of 2,453 ways, 35.7% — the way count understates it
  because shared ways are the long ones). `split.SPLIT_REGISTRY` now offers
  `blocked` and `buffered`, and `scripts/split_sweep.py` reports the frontier:
  `buffered-2560+500` takes leakage to 3.8% for 8% of the training chips,
  `buffered-2560+1000` to 1.3% for 25%. ~~Which of those two to freeze, and at
  which seed.~~ Settled: `buffered-2560+1000` at seed 0, committed as
  `splits/buffered_2560_1000_seed0.json`, leaking 0.67% of validation nodes.
  Still open: **whether to retrain segmentation on it**. `train.assign_split`
  defaults to `random` so `vegas_best.pt` keeps its provenance, and switching
  would make the existing APLS numbers incomparable; `train.main --split-file`
  is the path when that call is made. Blocking costs motorway coverage
  specifically — 6.0% of the AOI, 3.2% of the frozen validation set, 74 ways
  carrying one sixth of the macro average. See `EXPERIMENT_LOG.md` 2026-09-10.

- **Way-level holdout is unexplored and would dominate on power.** Every
  candidate measured partitions *chips*, which costs either training data (the
  buffer) or honesty (the leak). Supervising on training ways and scoring on
  held-out ways across all 981 chips would take identity leakage to zero while
  keeping every chip, at the cost of masked supervision in the training loop
  and a residual neighbourhood correlation that blocking does address. Worth
  pricing before the attribute model is built.

- **The attribute-inference model is designed and not built.** Both Tier 0
  checks passed: the comparison is resolvable (3.6 macro-F1 points on the frozen
  split under the way regime) and the signal exists (linear probe 0.3820 against
  a 0.1165 majority baseline). `plans/gnn_attribute_inference.md` carries the
  design, including the three decisions that change what the project can claim —
  line graph over way pieces, inductive with two disjoint graphs, and a per-piece
  MLP arm without which a GNN win conflates pooling with propagation. Arms A and
  B exist in `scripts/attr_probe.py`; C and D do not.

- **OSM now comes from a pinned local extract, and Overpass agreement is
  unvalidated.** `data/nevada-latest.osm.pbf` (MD5
  `5c750d8e270510e12dce81711c201491`) is read through `WAY_SOURCE_REGISTRY`.
  The reason is reproducibility rather than the rate limiting that forced it:
  an Overpass query cannot satisfy the content-hashed run-identity rule because
  the database moves under it. The two sources have never been compared on the
  same tile — the comparison test exists and is `network`-marked but has never
  run — and they are known to differ in three ways: `ox.graph_from_bbox`
  defaults to `retain_all=False` so the Overpass path keeps only the largest
  weakly connected component, osmnx simplifies degree-2 chains, and the pbf path
  nodes explicitly while osmnx arrives pre-noded.

- **`osm.ground_truth_graph` has always dropped disconnected components.**
  `retain_all=False` is osmnx's default and nothing here ever overrode it, so
  every OSM truth graph this repo has pulled — including the one behind the
  0.876 roundtrip ceiling — silently kept only the largest weakly connected
  component. Nobody chose that. Whether it is the right behaviour for a *truth*
  graph is a live question; it is definitely wrong for the crosscheck, where a
  disconnected alley is exactly the road being looked for, and the pbf path
  deliberately does not replicate it.

- **Two chips yield an empty OSM `drive` network without explanation.**
  `img1268` (732 m `residential`) and `img1279` (323 m `unclassified`) should
  have produced non-empty drive networks and did not; suspected `access` tags
  the filter excludes, unverified. Affects 2 of 155 chips and only the step 0
  gate, not the length or purity numbers.
- **Noding depends on exact intersections.** `shapely.ops.unary_union` splits
  only where geometries truly meet. Real SpaceNet labels do, but a line ending a
  fraction of a pixel short of another is left disconnected and silently costs
  connectivity. A snap-then-node pass would be more robust.
- **Datasets load every chip eagerly.** Measured at 27 ms and 1.7 MB per chip,
  so one AOI (989 chips) costs 27 s and 1.6 GB resident — fine. Four AOIs would
  be 6.6 GB, which is where a memory-mapped preprocessed cache starts to earn
  its keep. Not before.
- **Chips are not mosaicked.** Each is scored independently at roughly 300 m
  square, which is small enough that a single break moves APLS a lot. Stitching
  adjacent chips into larger tiles would give steadier per-tile numbers and
  exercise the tiling problem the plan mentions.
- ~~**Threshold is untuned.**~~ Swept 2026-09-08. Moving off the 0.5 default is
  worth about +0.08 APLS (paired t = +3.1 over 40 val tiles); the optimum is a
  flat plateau across 0.02-0.12 whose argmax is not resolvable at that n.
  IoU peaks at 0.30 and APLS at 0.02, so tuning on pixel overlap would have
  picked the wrong end. `make threshold-sweep` reruns it against a frozen
  checkpoint. **No default has been changed in code yet** — the value is still
  0.5 everywhere, so every committed number still reflects the untuned setting.
  Remaining work: pick the default, and sweep it jointly with `cleanup`'s
  spur-prune length, which it interacts with. See `EXPERIMENT_LOG.md`
  2026-09-08.

## Model and training

- **`spur_length` is mistuned, and the fix is not collectable yet.** Swept
  2026-09-10 (`scripts/cleanup_sweep.py`, 4x4x4, both arenas). Correcting an
  earlier claim here: the three constants had never been swept against *any*
  metric, not merely against APLS. `spur_length` is the whole effect —
  `simplify_tolerance` is inert (0.0004 of ceiling junction F1 across an 8x
  range) and `snap_tolerance` matters 5x less. On a perfect mask, relaxing the
  prune recovers **52% of the 0.1186 junction-F1 decoder loss** (+0.0618,
  t +7.07, 101 chips better / 10 worse); the residual 48% is a genuine decoder
  floor. On the *model's* output the gain is +0.0075 at t +0.61, not resolvable,
  because tightening lets skeleton noise through and the false junctions cancel
  the real ones. **The constants are coupled to mask quality and should be
  re-swept whenever the model improves** — nothing in the repo knows that today.

- **Retuning the decoder widens the model-to-ceiling gap.** Junction F1@10 gap is
  +0.1723 at the shipped setting and +0.2073 at `s0.5_p10_n4`, because the
  ceiling rises faster than the model does. After retuning, junction loss is more
  a model problem and less a decoder problem, which argues for Stage 2 or a better
  segmentation model over further decoder work.

- **A free +0.0120 APLS is on the table, and the optimum is not bracketed.**
  `s0.5_p10_n4` scores +0.0120 (t +2.96, CI [+0.0040, +0.0201]) with junction F1
  flat; `spur_length` 20 to 10 alone is +0.0054 APLS and +0.0070 junction F1.
  That is over half the entire gap-closing oracle ceiling for no model work.
  Two things to settle before flipping a default. **Spur 5 is the grid's low end
  *and* the ceiling optimum**, so run spur 2 and 3 first. And the domination is
  matching-radius-specific: at junction radius 5 the shipped setting is the
  model-arena maximum, rank 1 of 64, and it sits at the high-precision end of the
  junction precision/recall frontier (jP 0.7693 / jR 0.7148). If junction
  precision is the goal, F1 is the wrong referee and shipped is defensible.
  Changing the default invalidates every committed artifact, so it wants a
  clean-break rebuild rather than a quiet edit.

- **Do not read the simplify axis on APLS.** Control-point counts move with
  `simplify_tolerance` (the ceiling carries 53.0 proposal control points at 0.5
  against 41.4 at 1.0), so the estimator is not held fixed along it. The effect
  is 0.004 and no single mechanism covers both arenas.

- ~~**Reorder the decoder to `simplify -> snap -> link -> prune`.**~~ Closed
  2026-09-10. The census found `clean()`'s first `prune_spurs` destroys 47% of
  interior endpoints, which looked like an argument for reordering on its own
  merits. Scored: **+0.0008 APLS**, CI [-0.0025, +0.0040]. It decomposes into
  `gt_to_prop` +0.0055 (t +2.31) and `prop_to_gt` -0.0042 (t -3.00) — both
  halves resolvable, and they cancel. Dropping the first prune walks the same
  trade curve the mask threshold already walks. Not worth making.

- ~~**clDice loss not implemented.**~~ Implemented and measured 2026-09-09.
  `soft_cldice_loss` ships in `model.py`; `cldice_weight` defaults to 0.0, a
  bit-exact no-op that never builds the skeleton. **It does not help.** At
  weight 0.5 on top of augmentation it scored −0.024 APLS against augmentation
  alone (t = −1.17, CI [−0.065, +0.016], 21 tiles better and 19 worse). The
  static analysis explains why: on tile-spanning geometry the term is a *worse*
  discriminator of a severing gap than plain Dice (ratio 0.87), because its
  sensitivity term counts centreline pixels rather than connectivity. Residual
  work if anyone wants to reopen it: only weight 0.5 was tried, and clDice's
  gradient magnitude is about half Dice's, so a lower weight might add the
  topological nudge without diluting the pixel signal. The arm was also
  truncated at epoch 22 by an OOM kill against the other arm's 32. See
  `EXPERIMENT_LOG.md` 2026-09-08 and 2026-09-09.

- ~~**No augmentation.**~~ Done 2026-09-09, and it is the largest single
  improvement Stage 1 has produced: **+0.068 APLS** (0.7247 to 0.7925, t = +3.40,
  CI [+0.029, +0.107], 26 tiles better and 14 worse), with the train/val gap
  closing from +0.080 to +0.032. The full dihedral group only; no colour jitter,
  for the contrast reason recorded when this was filed. Note the budget
  interaction — the best epoch moved from 10 to 21, so `epochs=40, patience=10`
  was needed to let the run reach its own optimum. **`augment` still defaults to
  `False` in code** pending the call to flip it.

- **Epoch selection uses a pixel metric to choose a topological result.**
  `TrainConfig.select_on` defaults to `"val_loss"`, and `val_apls` is `None` for
  every epoch of the 2026-09-08 run, so the reported checkpoint is best-by-BCE-
  plus-Dice while the number reported from it is APLS. `SelectOn` already admits
  `"val_apls"`, so the wiring exists; what is missing is a per-epoch APLS cheap
  enough to select on (scoring is the expensive stage, hence the current
  default). Given how weakly pixel overlap tracks connectivity, there is no
  reason to think epoch 10 is the APLS-best epoch. A cheap proxy scored on a
  fixed small subset of val tiles would be enough to test whether the choice
  moves the final number.

- **`n_train_crops` degrades silently to one crop per tile.** `build_dataset`
  computes `per_sample = max(n_crops // len(usable), 1)`, so any total below the
  tile count collapses to 1 per tile with no warning. With 785 Vegas tiles the
  `TrainConfig` default of 256 yields 785 crops rather than 256, a 4x shortfall
  against the 3200 the real runs use — and the run proceeds normally, so the
  only symptom is a quietly worse number. Nearly invalidated the augmentation
  A/B on 2026-09-08; caught by reading the crop count in the startup log against
  the baseline's config. Either warn when the floor binds, or rename the
  parameter to `crops_per_tile` so the units are what the code actually honours.

- **The canonical training invocation is not codified.** `make train` takes a
  bare `TRAIN_ARGS` with no defaults, so reproducing the Vegas run means
  remembering seven flags, and getting one wrong produces a plausible-looking
  number rather than a failure (see the entry above). The house rule wants one
  canonical invocation per routine operation; this should be a `train-vegas`
  target, or `TRAIN_ARGS ?=` should carry the real defaults.

- **Lidar fusion: what tier one deliberately left out.** The plumbing is in —
  `TileSample.aux`/`aux_valid`, `synthesize_height`, `CoverageSampler`, and
  `early` in `FUSION_REGISTRY` — and the two endpoint runs are launchable. The
  deferred pieces, in the order the plan wants them:
  - **`dual` fusion.** Separate encoders fused at each decoder scale, so the
    imagery branch can take heavy photometric augmentation and the geometry
    branch none. It does not fit the current `Fusion` Protocol, which assumes
    fusion happens at the stem; registering it means widening the Protocol,
    which is fine — it has one implementor and is checked at its definition
    site. Only settleable across areas, never in-domain.
  - **The roughness channel.** The plan's stack is
    `[nDSM, intensity, roughness, valid]`; `synthesize_height` emits the first
    two. Roughness appends to `AUX_CHANNEL_NAMES` and `AUX_CHANNEL_SCALES` and
    nothing else changes — the stack width is read off the data.
  - **Coverage resampled per epoch for the training split.** Coverage is
    currently fixed per crop index, so validation stays comparable across
    epochs. That also caps the training set at one pattern per crop; the fix is
    a per-visit stream for the training split only, mirroring how augmentation
    already splits.
  - **A real `TileSource` with lidar.** Everything above is validated against
    synthetic channels. No number from a synthetic run is a result.

## Project stages (from the plan)

- **Stage 2: direct graph-tensor prediction (Sat2Graph).** Predict vertex
  presence plus directional edge slots per cell instead of a mask. Fixes
  connectivity in the loss rather than in post-hoc heuristics, and admits
  non-planar structure. Optional per the plan's own scoping. Its target is now
  better quantified than "the ~0.90 ceiling": on the SpaceNet holdout the
  decoder's own loss is 0.0280 of APLS but **0.1186 of junction F1**, so what
  skeletonisation destroys is junction structure, and a decoder that emits a
  graph directly is aimed squarely at that.

- ~~**Learned gap closing / link prediction.**~~ Measured out 2026-09-10 without
  building a model. The candidate pool exists (2,040 train positives at R=60,
  `label_snap` 10) and a perfect oracle is worth **+0.0224 APLS** (t +4.53, CI
  [+0.0127, +0.0322]), 12.9% of the reachable headroom. Three things close it:
  the A/B/C/D ablation ladder cannot resolve its arms against a paired sem of
  0.005, so the experiment returns "no detectable difference" whatever is true;
  the OSM crosscheck says ~7 in 8 zero-purity stubs are invented rather than
  unlabelled roads, so the ceiling was not measured against incomplete truth;
  and under attachment weighting the technique is *harmful*, adding 2.67
  junctions per chip of which only 0.49 are real. See `EXPERIMENT_LOG.md`
  2026-09-10. **Residual worth keeping:** a purity-filtered heuristic gap closer
  (`both_high_purity`) keeps 84% of the APLS gain at 28% of the junction damage
  with 188 edges instead of 443, and sits on the Pareto frontier. That is an
  afternoon and no model.
- **Stage 3: parametric curve fitting.** B-splines or arc-and-clothoid fits to
  extracted centerlines, optimized with chamfer plus a curvature regularizer.
  The most on-thesis part for inverse procedural modeling, and the smallest.

## Known deviations from CLAUDE.md

Found in the 2026-08-24 audit and deliberately deferred. The vendored-code,
offline-coverage and mechanical items from that audit are done — writing the
missing offline suites turned up two real geometry bugs, both since fixed; see
`EXPERIMENT_LOG.md` 2026-08-24.

- **Algorithm choices dispatch on strings, not Protocol + Registry.**
  `geograph.control_cuts` branches on `sampling`, and `cleanup.clean` hardcodes
  a fixed simplify → prune → snap → prune sequence. The pluggable-pipelines
  rule wants both as registry stages. Deferred because there are only two
  sampling rules and one cleanup pipeline today; the abstraction earns its keep
  once a third arrives, which Stage 2 will likely force.
- **No frozen-checkpoint rerun entry point** (partly addressed). `train()` and
  `overfit_one_batch()` accept pre-built datasets, so a sweep over model settings
  pays tile ingestion once. `roundtrip.run()` still refetches and re-rasterizes
  every call, and there is still no graph save/load for scoring a stored
  truth/proposal pair without recomputing.
- **`make lint` does not cover `scripts/`.** `SRC` is `$(PKG) $(TESTS)`, so the
  builder scripts are unlinted; ruff run directly on
  `scripts/build_apls_explainer.py` flags line-wrapping and an E501. Widening
  `SRC` is the right fix but breaks the fast loop until those are cleaned, so it
  wants to be one change: fix the findings, then widen. Scripts are the least
  reviewed code in the repo and the most likely to rot.

- **No content-hashed run identity.** Nothing implements
  `timestamp + SHA-256(config + inputs)`. Matters once sweeps start producing
  numbers that need to be told apart.
- ~~**Quality metrics are not separated by stage.**~~ Done 2026-09-01.
  `train.EvalReport` reports pixel IoU, APLS on the raw traced graph, APLS after
  cleanup, and the tile's own ceiling, plus the fraction of it captured.
- **`metrics._directional` accumulates in a Python double loop.** Rectangular
  numeric work over path-length arrays; vectorizes cleanly. ~15k iterations on
  default sampling, ~94k on uniform. Style rule prefers vectorized; also a
  speed win if scoring ever lands in a training loop.

## Toolchain exceptions

- **pandas is present, transitively.** The house rule is polars, never pandas,
  but `osmnx` → `geopandas` → `pandas` is not separable without giving up the
  geospatial stack. No first-party code imports pandas, and none should. Revisit
  only if a pure-shapely/pyproj path to OSM ingestion becomes worthwhile.

- **Reconcile the eval threshold between `train.py` and the site build.** The
  run artifact scores at `predict_mask`'s 0.5 default while the showcase now
  reports at 0.02, so the two disagree by construction. Options: make the
  training-time eval threshold a `TrainConfig` field, or move `predict_mask`'s
  default. Moving the default invalidates the existing run artifacts, so it
  needs a clean-break rebuild rather than a shim. Evidence for 0.02 is the
  2026-09-09 leave-one-out entry in `EXPERIMENT_LOG.md`.

- **Diagnose `img1612`'s zero ceiling.** A perfect mask traced back scores 0.0
  against its own truth graph, and the chip has edges so the empty-graph guard
  misses it. Both scorers now skip zero-ceiling chips, which is a sanity guard
  rather than an explanation. Suspected cause is every edge falling under
  `min_path_length` so no control-point pair survives; unverified. See the
  2026-09-09 disjoint-holdout entry in `EXPERIMENT_LOG.md`.
- **`make lint` does not cover `scripts/`.** `SRC` in the Makefile is
  `$(PKG) $(TESTS)`, so script files are never formatted or linted by the repo's
  own target and have accumulated at least one E501. Widening `SRC` will surface
  pre-existing violations, so it is a small cleanup rather than a one-line edit.
