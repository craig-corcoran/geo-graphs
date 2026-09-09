# Backlog

Deferred work, known residual issues, and scope exclusions. Check here before
proposing new architectural work.

## Evaluation harness

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

- **TOPO metric not implemented.** The plan calls for APLS *and* TOPO; only APLS
  exists. TOPO is the more local measure and would help localize where the
  ceiling above is lost.
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
- **Only the sample is downloaded.** Ten Vegas chips is about 1 km², far too
  little to train on; validation swings wildly between epochs. The 24 GB
  `SN3_roads_train_AOI_2_Vegas.tar.gz` is the next pull now that the loader is
  proven. Disk has room.
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
  non-planar structure. Optional per the plan's own scoping — but now the only
  route past the ~0.90 ceiling above, which is measured rather than asserted.
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
