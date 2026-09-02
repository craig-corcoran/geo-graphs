# geo-graphs

Road graph extraction from overhead imagery: pixels in, a typed graph with
geometry out.

## Where this is

The evaluation harness is built and validated; no model exists yet. That
ordering is deliberate — the interesting failure in this problem is that *a mask
that looks great produces a graph that is badly wrong*, and that gap only shows
up in a topology-aware metric you can trust. Ground truth comes from
OpenStreetMap directly, so the whole loop runs without downloading imagery.

```
OSM ground truth ──► 1 m/px mask ──► skeleton ──► graph ──► APLS
```

Rendering ground truth to a *perfect* mask and tracing it back measures the
pipeline's own ceiling. It currently scores **0.876 APLS**, and that is not a
tuning failure: the tile contains 20 road crossings with no junction (bridges
and tunnels), and a 2D mask cannot represent a road passing over another
without inventing one. No Stage 1 model scored through this pipeline can beat
it. See `EXPERIMENT_LOG.md`.

## Getting started

```bash
make sync
```

Then read [`examples/walkthrough.ipynb`](examples/walkthrough.ipynb) — it walks
the whole pipeline end to end, frames each module by the job it does during
training and inference, and shows where the model slots in.

```bash
make examples
```

```bash
make check
```

`make help` lists the rest. `make roundtrip` scores one tile end to end and
takes `TILE_LAT` / `TILE_LON`.

## Layout

| Module | Role |
|---|---|
| `tiles` | Ground patches and the 1 m/px raster grid. One pixel is one metre, so lengths need no conversion. |
| `osm` | Ground-truth road graphs from OpenStreetMap, clipped to a tile. |
| `geograph` | The graph type and its geometry operations, in pixel coordinates. |
| `raster` | Graph to binary road mask. |
| `skeleton` | Mask to graph, by morphological thinning and tracing. |
| `cleanup` | Simplify, prune spurs, snap junction clusters. |
| `metrics` | IoU and APLS. |
| `roundtrip` | Measures the pipeline's own ceiling on a tile. |
| `data` | Tile sources, synthetic imagery, crop sampling. |
| `model` | U-Net, and the BCE + soft Dice loss. |
| `train` | Training loop, checkpoints, per-stage evaluation. |

`raster` builds the segmentation model's training targets; `skeleton` + `cleanup`
are the inference-time decoder that runs on its predictions.

The model trains today against **fabricated** imagery (`data.SyntheticTileSource`),
which exists so the loop and the evaluation path could be built before any
download. Those runs prove plumbing, never quality. Real imagery arrives as a
second `TileSource` registered in `TILE_SOURCE_REGISTRY`; nothing downstream
changes.

```bash
make train
```

## On the metric

APLS is implemented here rather than imported: the reference implementation
(`CosmiQ/apls`) pins `GDAL==2.4.0` and cannot be installed on a current
interpreter. Its scoring subset is vendored into `tests/reference/` as a test
oracle, and `tests/test_against_reference.py` pins the agreement to **1e-4**.

The defaults (`sampling="reference"`, `max_control=None`) reproduce the
reference to within 3e-5, so `metrics.apls(truth, proposal)` is comparable to
published leaderboard numbers as-is. `sampling="uniform"` samples control points
far more densely; it is the better tool for localizing *where* a proposal loses
score, but it is a different estimator and drifts by up to 0.06. Report on the
defaults.
