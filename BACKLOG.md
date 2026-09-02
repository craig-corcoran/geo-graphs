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

- **No imagery yet — the last thing blocking Stage 1.** Ground truth comes from
  OSM directly, and `data.SyntheticTileSource` fabricates imagery so the model
  and training loop could be built and tested. Remaining work is a
  `SpaceNetTileSource` (AWS S3) registered under `TILE_SOURCE_REGISTRY`; nothing
  downstream changes. The Las Vegas dev tile is inside SpaceNet AOI 2 so the
  tuning transfers.
- **Threshold is untuned.** `model.predict_mask` takes one and it is a real
  hyperparameter trading the two APLS directions against each other. Tune it on
  APLS, not IoU, once a model is trained on real imagery.

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
