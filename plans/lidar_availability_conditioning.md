# Availability-conditioned lidar fusion

## Motivating question

Can one set of weights span the whole lidar-coverage spectrum — full coverage in
the training areas, zero coverage in a new city, arbitrary partial coverage in
between — without the zero-coverage path degrading below an imagery-only model
trained alone?

That is the load-bearing claim behind every later step. Active collection is
pointless if the model cannot use partial coverage; imagery-only deployment is
pointless if conditioning on lidar during training makes the no-lidar forward
pass worse. Both are measurable before any real lidar exists.

## Where the code starts

- `TileSample` is frozen with `image (H, W, C)`, `mask (H, W)`, `truth` graph.
  `C` is already free; nothing downstream assumes 3.
- `UNet.__init__(in_channels=3, ...)` is parameterized. `model.py:66` heads to a
  single logit channel.
- `data.augment_crop` applies one dihedral element to image and mask jointly.
  `dihedral` is per-array, so extending to a channel stack is mechanical.
- `data.crop_specs` samples crop placement with a road-share acceptance rule;
  there is no notion of a per-crop coverage mask.
- `SyntheticTileSource` builds imagery from a mask via `synthesize_image`, so a
  synthetic height field can be generated from the same graph the mask came from.
- `train.EvalReport` already separates pixel IoU, raw APLS, cleaned APLS, tile
  ceiling, and `fraction_of_ceiling`.
- No real lidar exists yet. Everything below is built and validated against
  synthetic channels; the real-data path is a new `TileSource`.

## Design decisions

### Lidar enters as a channel stack plus an explicit validity channel

`[nDSM, intensity, roughness, valid]`. The validity channel is not optional.
Without it `nDSM = 0` means both "measured, at grade" and "never measured", and
zero-filling teaches the model that absent lidar is positive evidence of flat
ground — worse than supplying no lidar at all. This is the overloaded-boolean
rule applied to tensors: two semantically distinct cases must not collapse onto
one value.

Unmeasured positions carry `valid = 0` and an arbitrary fixed fill in the data
channels. Tests assert the fill value cannot leak: two crops differing only in
the fill under `valid = 0` must produce identical logits.

### Coverage masks are spatially structured, not per-pixel

The deployment condition is contiguous corridors shaped like flight lines.
Bernoulli-per-sample dropout trains for all-or-nothing; per-pixel dropout trains
an inpainter. Neither is the target regime, and a model trained on either will
look correct at the endpoints of the coverage sweep and sag through the middle,
which is the only part that matters.

`CoverageSampler` emits binary masks at crop resolution:

- `full` — all ones. The upper endpoint.
- `none` — all zeros. The lower endpoint, and the imagery-only deployment case.
- `strips` — parallel bands at a sampled orientation, width and spacing set to
  hit a target coverage fraction.
- `blocks` — axis-aligned rectangles, for tiled rather than swathed collection.

Coverage fraction is drawn per crop from a curriculum spanning `[0, 1]`, with
mass deliberately placed at both endpoints so neither degenerate case is rare.

### Fusion is a registry stage, not a fixed choice

`FUSION_REGISTRY: dict[str, Callable[[FusionConfig], Fusion]]` against a
`@runtime_checkable Fusion` Protocol, pinned at the definition site with
`_: type[Fusion] = EarlyFusion`.

- `early` — concatenate onto the input stem, `in_channels = 3 + 4`. New stem
  weights initialize to the mean of the RGB weights so pretraining survives.
- `dual` — separate encoders, fused at each decoder scale.

The two are not equivalent under domain shift, and that is the point of making
it a stage rather than picking one. nDSM and roughness are physical measurements
in metres: they do not shift with atmosphere, sun angle, sensor calibration, or
season. RGB shifts with all of them. Early fusion entangles the two in the first
convolution, so they cannot be regularized differently; `dual` allows heavy
photometric augmentation on the imagery branch and none on the geometry branch.
That is a real argument for the extra parameters, and it is separable from the
modality-dropout argument. It is also an argument that only pays off across
areas, so it cannot be settled in-domain — which is exactly why it goes to the
registry and gets measured, rather than being decided here.

### Synthetic lidar is generated from the same graph as the mask

`synthesize_height` takes the truth graph and emits `nDSM` and `intensity`
consistent with it: roads at grade with low intensity, off-road terrain given
correlated noise at a chosen relief scale, canopy patches raised with high
intensity, and — where the graph has crossing edges with no shared node — one
edge lifted to overpass height.

Two properties matter and both are things real lidar has:

- Height carries information the imagery does not. Canopy patches occlude the
  road in the synthetic *image* while leaving the synthetic *ground return*
  intact, so the fusion model has something to gain that an imagery-only model
  cannot recover. Without this the whole experiment measures nothing.
- Height is not a noiseless copy of the label. If `nDSM` were a clean function
  of `mask` the model would learn to read the answer off it, every coverage
  sweep would be trivially monotone, and nothing would transfer.

Synthetic results prove plumbing and relative ordering, never quality. No number
from a synthetic run gets reported as a result.

## Staging

### Tier 1 — plumbing, synthetic, endpoints only

`TileSample` gains `aux: np.ndarray | None` and `aux_valid: np.ndarray | None`.
`None` means the source has no lidar at all, distinct from an all-zero
`aux_valid` meaning a source that has lidar but not here. `augment_crop`
transforms the stack jointly. `CoverageSampler` in the crop path. `early` fusion
registered. Train at `full` and at `none`, confirm the endpoints differ in the
direction expected under synthetic canopy.

Exit: the coverage sweep runs end to end and the two endpoints are separated.

### Tier 2 — the curve

Coverage curriculum on. Sweep held-out coverage in `{0, 0.1, 0.25, 0.5, 1.0}`
and emit APLS against coverage as a curve per area, not a scalar. Register
`dual` and run the same sweep, giving the early-vs-dual comparison its first
numbers.

Exit: a coverage-APLS curve per fusion mode, written as JSON by the pipeline.

### Tier 3 — availability self-distillation

Two forward passes per batch, one with lidar and one masked out, plus a loss
pulling the zero-lidar features toward the full-lidar features. This attacks the
specific failure mode of Tier 2: a model trained with lidar mostly present
learns to lean on it and the zero-coverage path silently undertrains. The number
it must move is the `coverage = 0` end of the curve, and it must move it without
dropping the `coverage = 1` end.

Exit: the 0% endpoint under distillation, against the 0% endpoint without it and
against an imagery-only model trained alone. That third baseline is the one that
matters — conditioning is only worth its complexity if it does not cost anything
at zero coverage.

## Measurement

Leave-one-area-out, always. In-area held-out tiles share sensor, season, sun
angle, and road morphology, and will overstate transfer badly. With few areas
the LOO sample is small, so report per-area numbers and do not lean on the mean.

Per config, per held-out area:

- APLS after cleanup, and `fraction_of_ceiling`
- pixel IoU
- APLS as a function of held-out coverage fraction

Two Pareto frontiers, reported rather than collapsed into a ranking:

- `(coverage fraction, APLS)` — how much geometry must be bought for how much
  connectivity.
- `(in-area APLS, held-out-area APLS)` — where fusion depth and augmentation
  strength trade against each other, and where a config that wins in-domain can
  lose overall.

Expect the lidar gain in APLS to exceed the gain in IoU by a wide margin, since
the mechanism is canopy gaps and grade separations — small in pixel count, large
in topology cost. Report both; the divergence is itself the result. The overpass
share is gated on the z-aware graph representation and will not appear until
that lands, so it is not evidence against fusion if it is missing here.

## Cost knobs

Committed defaults are the expensive, correct setting; a smoke run turns them
down at the call site.

- `--coverage-samples` — points on the coverage sweep. Default 5, smoke 2.
- `--fusion` — registry key. Default `early` until Tier 2 says otherwise.
- `--distill-weight` — 0 disables the second forward pass entirely, halving
  step cost.
- Existing `--n-train-crops`, `--epochs`, `--resolution`.

## Risks

- **Synthetic lidar is too easy.** If `nDSM` is nearly a function of the label,
  every result is vacuous. Guard: an imagery-blind model given only the lidar
  stack must score well below the fused model, and the occlusion generator must
  actually break the imagery. Check both before reading any curve.
- **The curriculum makes the endpoints rare.** A uniform coverage draw puts
  almost no mass at exactly 0 or 1, and those are the two numbers being
  reported. Guard: explicit mass at both endpoints.
- **`n_train_crops` floors silently.** Known: `build_dataset` collapses to one
  crop per tile below the tile count, with no warning. It nearly invalidated the
  augmentation A/B. Read the crop count out of the startup log for every run in
  this plan.
- **Early fusion wins in-domain and loses across areas.** This is the expected
  shape and the reason both are registered. Do not settle it on in-area numbers.
