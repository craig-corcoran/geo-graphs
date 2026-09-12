# geo-graphs

Road graph extraction from overhead imagery: pixels in, a typed graph with
geometry out.

## Where this is

A U-Net baseline scores **0.7976 mean APLS** on 155 held-out SpaceNet 3 chips
over Las Vegas, which is inside the 0.771–0.801 range the challenge's top five
reported for that city. The splits differ, so it is not a leaderboard claim,
only evidence that the replication sits in the right regime.

```
imagery ──► U-Net ──► mask ──► skeleton ──► cleanup ──► graph ──► APLS
```

The harness was built and validated before the model, which is the ordering the
problem rewards: a mask that looks great can produce a graph that is badly
wrong, and that gap is invisible to anything but a topology-aware metric you
trust. The metric came first for the same reason four bugs in it produced
scores in the expected range and were caught by invariants rather than by
reading output.

## What the remaining 0.20 is made of

Rendering ground truth to a *perfect* mask and tracing it back measures what
the pipeline can reach with a flawless segmenter. On the held-out set that
ceiling is **0.9720**, which splits the shortfall in two:

| | APLS | needs |
|---|---|---|
| ~0.174 | model error | a better objective or more training |
| ~0.028 | structural | a different output representation |

The structural part is grade-separated crossings. A 2D mask cannot record
"these roads cross in the image but do not connect", so tracing invents a
junction. In the worst observed case the graph traced from a *perfect* mask
offers a 14 m route where the true network requires 958 m. It survives perfect
segmentation, so no amount of training touches it.

`EXPERIMENT_LOG.md` is the chronological record; `BACKLOG.md` holds deferred
work; `plans/` holds the designs that have not been built.

## Write-ups

`site/` holds two self-contained pages, each a single HTML file:

| file | what it covers |
|---|---|
| `transport_networks.html` | The case study: baseline, the residual split, whether APLS is the right objective, and two extensions measured before either was built. |
| `where_road_graphs_break.html` | The replication and the ceiling argument in full, at more depth. |

## Getting started

```bash
make sync
```

Then read [`examples/walkthrough.ipynb`](examples/walkthrough.ipynb) — it walks
the whole pipeline end to end and frames each module by the job it does during
training and inference.

```bash
make check
```

`make help` lists the rest. Every routine operation is a target: `make
roundtrip` scores the OSM round trip on one tile, `make train` trains and scores
per stage, and the analysis targets (`make link-oracle`, `make attr-probe`,
`make split-sweep`, `make cleanup-sweep`) each write a JSON artifact under
`outputs/` that the notebooks and pages read rather than recompute.

Imagery is a registry choice. `data.SyntheticTileSource` is the default and
exists so the loop could be built before any download; those runs prove
plumbing and never quality. Real runs register the SpaceNet source
(`geo_graphs/spacenet.py`) into `TILE_SOURCE_REGISTRY`, and nothing downstream
changes. `make fetch-spacenet` and `make extract-spacenet` pull the tarballs.

## Layout

| Module | Role |
|---|---|
| `tiles` | Ground patches and the 1 m/px raster grid. One pixel is one metre, so lengths need no conversion. |
| `osm` | Ground-truth road graphs from OpenStreetMap, clipped to a tile. |
| `spacenet` | SpaceNet Roads as a tile source, including the non-square-pixel correction. |
| `geograph` | The graph type and its geometry operations, in pixel coordinates. |
| `raster` | Graph to binary road mask. |
| `skeleton` | Mask to graph, by morphological thinning and tracing. |
| `cleanup` | Simplify, prune spurs, snap junction clusters. |
| `metrics` | IoU, APLS, and the two presence metrics below. |
| `split` | Train/validation assignment, at random or by ground position. |
| `roundtrip` | Measures the pipeline's own ceiling on a tile. |
| `data` | Tile sources, synthetic imagery, crop sampling. |
| `model` | U-Net, and the BCE + soft Dice loss. |
| `train` | Training loop, checkpoints, per-stage evaluation. |

`raster` builds the segmentation model's training targets; `skeleton` +
`cleanup` are the inference-time decoder that runs on its predictions.

## On the metric

APLS is implemented here rather than imported: the reference implementation
(`CosmiQ/apls`) pins `GDAL==2.4.0` and cannot be installed on a current
interpreter. Its scoring subset is vendored byte-for-byte into
`tests/reference/` as a test oracle, unmodified so its independence survives,
and `tests/test_against_reference.py` pins agreement at **1e-4**. The worst
case observed on the defaults is 3e-5.

The defaults (`sampling="reference"`, `max_control=None`) reproduce the
reference, so `metrics.apls(truth, proposal)` is comparable to published
leaderboard numbers as-is. `sampling="uniform"` samples control points far more
densely; it is the better tool for localizing *where* a proposal loses score,
but it is a different estimator and the test allows it 0.08 of drift. Report on
the defaults.

Where the control points land is a choice the metric makes quietly. The
reference rule places interior points only on edges passing a curvedness test,
and a grid city is mostly straight edges, so 78.2% of the truth's control
points end up being the graph's own nodes. Under that rule APLS behaves less
like a length metric and more like a junction-and-endpoint one.

APLS also scores routes, so an edge counts in proportion to how many shortest
paths cross it. That is the right weighting for routing and the wrong one for a
map someone reads: a missing cul-de-sac costs APLS almost nothing and is a
visible defect to the people who live on it. Two metrics beside it weight
presence instead.

| function | question | weights by |
|---|---|---|
| `metrics.buffer_length_prf` | is this street here at all? | road length |
| `metrics.junction_prf` | is it attached to the right cross-streets? | junction count |

`buffer_length_prf` is the share of truth road length lying within `X` metres of
proposal road and the reverse, which makes it blind to topology by
construction: a road cut in two one pixel apart is still fully covered.
`junction_prf` matches nodes of degree 3 or more one-to-one within a radius and
also reports how often matched junctions agree on degree. Neither is comparable
to a leaderboard number, and neither replaces APLS. Sweep the buffer:
displacement below it is invisible, so the buffer is the geometric tolerance the
number is quoted at.

The three disagree. Scoring one fixed set of added edges under all three on the
same chips moves routing +12.8% of its headroom, coverage +7.2%, and junction
accuracy −13.9%. Which metric to optimise is a product decision, not a
technical one.

## Scope

Every measured number here comes from Las Vegas at one resolution. SpaceNet 3
publishes full training sets for three further cities, so cross-city
replication is the obvious next check, and one city is a scoped choice rather
than a limit of the data.
