# Soft APLS: a differentiable connectivity loss

## Motivating question

Does supervising *routes* rather than pixels close any of the measured gap
between what the segmentation loss optimizes and what APLS scores?

The gap is measured, not assumed. The 2026-09-08 threshold sweep puts the APLS
optimum at 0.02 and the IoU optimum at 0.30 — the two objectives disagree about
the operating point by more than an order of magnitude, and tuning on pixel
overlap picks the wrong end. `fraction_of_ceiling` sits at 0.746 against an
intrinsic ceiling of ~0.90, so there is roughly 0.15 of loss-side headroom
before the non-planarity limit binds.

The work here builds a loss whose unit of supervision is a path between two
control points, scored with APLS's own length-ratio kernel, and measures whether
it moves `fraction_of_ceiling`.

## Where the code starts

- `model.segmentation_loss` is BCE + soft Dice + optional soft-clDice
  (`soft_skeleton`, iterated min/max pooling). clDice is a local skeleton
  overlap: a break wider than the soft skeleton's reach produces no gradient
  pulling the fragments together.
- `metrics.apls` reproduces the reference to 3e-5. `metrics._directional` gives
  the control-point machinery: `geograph.densify(A, spacing, sampling)`,
  `geograph.inject_points`, `spacing=50.0`, `min_path_length=10.0`,
  `max_snap=25.0`, `sampling="reference"`, harmonic mean of the two directions.
- `TrainConfig.crop_size = 256` at `--resolution 1.0`, so a 256 m chip and a
  65536-node grid.
- The truth graph is available per crop, so reference path lengths are constants
  that can be computed once and cached with the crop.

## What is new here

Not new: path-based differentiable connectivity losses (CAPE, MALIS, CP-loss);
smoothed shortest-path solvers (DataSP, randomized shortest paths,
Mensch-Blondel smoothed DP). CAPE is explicitly APLS-inspired and is the nearest
prior work.

Apparently unpublished, on a literature search of moderate depth:

- CAPE-style path supervision applied to **roads** at all — it is validated on
  neuron and vessel tracing, 2D and 3D.
- APLS's **length-ratio kernel** `|L - Lhat| / L` as the loss, contracting the
  expected path occupancy against edge lengths rather than against edge costs.
  CAPE's cost is zero on any on-road route regardless of length, so it has no
  notion of a correct length — only of a cheaper one. The ratio kernel has an
  interior optimum at the truth, which is also what penalizes shortcuts.
- The randomized-shortest-path free energy used as a **segmentation** loss. RSP
  has been used for node clustering and semi-supervised classification, not for
  dense prediction.

Treat the novelty claim as a hypothesis. Before writing the paper-shaped version
of any result, do a systematic search; this one was ~8 targeted queries.

## Specification

The mathematics is in [`soft_apls_methodology.md`](soft_apls_methodology.md).
The shape of it, for orientation:

- The network grows a **second head** predicting a truncated distance-to-road
  field `d_hat`, regressed against the ground-truth distance transform. The
  probability head is untouched, so `predict_mask`, `skeleton`, IoU and clDice
  keep working and every existing number stays comparable.
- Routes are shortest paths on the pixel grid under
  `c_e = delta_e * (1 + kappa * d_hat_e ** gam)`. A distance field rather than
  `1 - p` because `1 - p` saturates: it says a pixel is not road but not how far
  from road it is, so a 3 m gap and a 50 m gap cost the same per metre. The
  `+ 1` is the addition to CAPE's construction — it is what keeps route cost
  measured in metres, and it is also what removes CAPE's need to fence the
  router into a corridor around the truth path.
- That cost decomposes exactly into `Lhat + kappa * V`: the geometric length of
  the route, and how far off-road it strays. Both are contractions of one
  expected-occupancy vector `y_bar`. `Lhat` is what APLS scores; `V` is what
  CAPE minimizes.
- The loss weights the two separately, with APLS's own ratio kernel and a
  softened clip on the length half. `kappa` leaves the loss entirely and becomes
  an annealed routing parameter.
- `Lhat` has zero gradient under a single hard path, so the smoothing of `y_bar`
  is what makes the length term exist at all.

## Staging

Each tier is a separate implementation of one `Occupancy` Protocol, selected by
config string through an `OCCUPANCY_REGISTRY` of lambda factories, per the
pluggable-pipelines rule. Adding a tier is one class plus one `register()` call
with no change to the training loop. The Protocol returns `y_bar`, not a
distance, because every loss term is a contraction of it; `domain` carries the
optional corridor Tier 1 needs.

    @runtime_checkable
    class Occupancy(Protocol):
        def occupancy(
            self,
            cost: Tensor,
            pairs: Tensor,
            domain: Tensor | None = None,
        ) -> Tensor: ...

### Tier 0 — distance head

A second output channel on the existing U-Net regressing the truncated
ground-truth distance transform, with a Huber loss. No routing, no path terms.
Roughly 60 lines across `model.py` and `data.py` plus the transform in the crop
pipeline.

Worth landing and measuring alone: distance-transform regression is a known
auxiliary task and may move `fraction_of_ceiling` by itself, which would need to
be subtracted from any later attribution to the path terms.

### Tier 1 — CAPE as published

Vertex projection to the local minimum of `d_hat` in a 7x7 window; the
ground-truth path rendered and dilated to a 10 px corridor; Dijkstra confined to
it; loss `sum d_hat ** 2` along the returned route. The path is held fixed and
gradient flows to the pixels on it — Danskin's theorem applied to
`min over paths of c(path; d_hat)`.

In this specification's terms that is `V` alone, at `gam = 2`, unnormalized, and
computed inside a corridor. `y_bar` is a single path indicator, so
`<delta, y_bar>` is piecewise constant and **the length term cannot be expressed
at this tier** — not as a simplification but by construction.

Roughly 150 lines. Its job is to answer whether path supervision moves the number
*at all* before the novel part gets built, and to give the published method a
fair run on road data, which nobody has reported. If hard-path supervision does
nothing here, smoothing will not rescue it.

### Tier 2 — perturbed paths, APLS kernel

Run Dijkstra on M noise-perturbed cost maps and average the resulting path
indicators. That average *is* `y_bar`, so both contractions come from one
computation and the length term becomes available. Embarrassingly parallel, no
linear algebra.

**The corridor is dropped here.** It exists in Tier 1 only because a cost that
is zero on-road makes a loop around a break free, so the router has to be fenced
in to see the break at all. With the `+ 1` a loop-around is longer, which is the
signal `L_len` reads. Keeping the corridor would make `L_len` vacuous, since a
route pinned within 10 px of the truth has `Lhat` pinned to `L`. If free routing
proves unstable the fallback is a corridor wide enough to admit the detours worth
penalizing, not 10 px.

Roughly 350 lines. **This is the novel object and the one to build.**

### Tier 3 — randomized shortest paths

Full Gibbs average over all paths. The expected edge-visitation vector comes from
the RSP identity

    n_bar_e  proportional to  z_bwd_i * w_ij * z_fwd_j

so the entire average costs **two sweeps per source**, not M samples. Two real
obstacles:

- On a 256 px crop the diameter is ~600 px, so soft Bellman-Ford needs T ~ 700
  iterations. Each is a 3x3 log-sum-exp reduce over a `(B, K, 256, 256)` tensor;
  unrolling for autograd is ~30 GB of activations at `B=8, K=20`. Implicit
  differentiation is mandatory — a custom `autograd.Function` whose backward
  solves the adjoint with the same iteration, `O(1)` memory in T.
- On a cyclic grid the partition function diverges unless the spectral radius of
  `W` is below 1, which puts a lower bound on `beta` that has to be detected and
  enforced rather than assumed.

Estimate ~1400 sequential memory-bound kernel launches per step, so 0.5-2 s per
batch against a U-Net step — 10 to 50x. Not a default; an every-N-steps term or a
fine-tuning-only loss. Build only if Tier 2 shows the smoothing is what carries
the gain.

## Cost knobs

Committed defaults are the expensive, correct settings. Smoke runs turn them
down at the call site, never by editing the committed artifact.

| Knob | Default | Smoke |
|---|---|---|
| `n_sources` per crop | all control points | 4 |
| `n_perturbations` (Tier 2) | 8 | 1 |
| `rsp_iterations` (Tier 3) | 700 | 50 |
| `d_max` truncation | `3 * road_width` | same |
| `occupancy` registry key | tier under test | `"dijkstra"` |

A turned-down run proves plumbing, never quality. Its numbers do not get
reported.

## Measurement

- **Reference agreement.** As `beta -> inf`, `tau -> 0`, `kappa -> inf` with
  `d_hat` set to the true transform, `1 - L_len` must converge to
  `metrics.apls(...).gt_to_prop` computed on the rasterized truth. This is the reference-implementation check the house
  rules require, and it is cheap: `metrics.apls` is already validated to 3e-5.
  Disagreement is a finding for `EXPERIMENT_LOG.md` before either side changes.
- **Gate metric is `fraction_of_ceiling`**, from `train.EvalReport`, on
  APLS-after-cleanup. Not IoU, and not raw APLS — the ceiling varies per tile.
- **Report both directions separately.** `gt_to_prop` and `prop_to_gt` fail for
  different reasons, and a reverse term that works should move the second one.
  Aggregate APLS can hide a trade between them.
- **Artifacts before notebooks.** The sweep writes JSON per config; any
  comparison reads that file.
- **Report every candidate with per-metric numbers and identify the Pareto
  frontier** across `fraction_of_ceiling`, IoU, and seconds per step. No
  threshold gates that disqualify a config on a value judgement.
- Run against a frozen checkpoint where possible, the way `make threshold-sweep`
  does, so tuning the loss does not pay for retraining.

## Make targets

- `soft-apls-smoke` — one crop, `n_sources=4`, asserts a finite gradient reaches
  the gap pixels. Fast loop.
- `soft-apls-reference` — the `beta -> inf` agreement check against
  `metrics.apls`.
- `distance-head-ab` — Tier 0 alone against the current baseline, so its
  contribution can be separated from the path terms'.
- `soft-apls-sweep` — the real sweep over the `kappa` schedule, `gam`, `w_len`,
  `w_feas`, `w_rev`; writes JSON.

## Risks

- **The ceiling is untouched.** Nothing here helps with 13 bridges and 4 tunnels;
  a 2D mask cannot represent an overpass. The plausible outcome is that Tier 2
  recovers most of the available 0.15 and Tier 3 adds little, in which case the
  honest conclusion is that the remaining gap is representation and the next work
  is Stage 2 (Sat2Graph graph-tensor). That is a useful result and should be
  written up as one rather than treated as a failure.
- **Tier 0 may carry the gain.** If the distance head alone moves the number and
  the path terms add nothing, the honest report is that the auxiliary task did
  the work. This is why Tier 0 is measured separately.
- **The `kappa` schedule may have no good endpoint.** The conditioning-versus-
  gradient tension is structural. If no schedule both trains and keeps geodesics
  on-road, that is the finding, and the next thing to try is an augmented
  Lagrangian rather than more sweeping.
- **Two heads can disagree.** Nothing ties `p` and `d_hat` together, so the
  probability head can call a pixel road where the distance head puts it 5 m
  from one. Routes follow `d_hat` while thresholding follows `p`, so a
  divergence shows up as a loss that improves while APLS does not. Worth a
  consistency diagnostic before a consistency term.
- **`Lhat` approximates something intractable.** What an uncertain prediction
  really implies is the expected shortest path over the distribution of road
  networks the prediction admits. That expectation is not computable at this
  scale; the penalty relaxation is a surrogate, and the two disagree most where
  the prediction is diffuse rather than confident either way.
- **Cost may dominate.** If Tier 2 at default settings triples epoch time, the
  right shape may be a fine-tuning term applied to a checkpoint trained on
  BCE + Dice + clDice rather than a from-scratch objective.
- **Edge-deletion coverage is a trap if copied verbatim.** CAPE deletes each
  routed path's edges from the truth graph itself, which fragments it and
  degrades later routes. It is harmless there because the truth path is only a
  mask and an endpoint pair. Here `L(a, b)` is the reference the kernel measures
  against, so deletion must touch only a bookkeeping copy, and the loop needs a
  termination condition that accounts for `min_path_length`. See the methodology
  doc; precomputing and caching the cover per crop is the form to prefer.
- **Prior art may exist.** Search depth here was moderate. Check before claiming
  novelty anywhere public.

## References

- CAPE: Connectivity-Aware Path Enforcement Loss — arXiv 2504.00753
- Maximin affinity learning of image segmentation (MALIS) — arXiv 0911.5372
- Developments in the theory of randomized shortest paths — arXiv 1212.1666
- Sparse randomized shortest paths with Tsallis divergence — arXiv 2007.00419
- DataSP: differentiable all-to-all shortest paths — arXiv 2405.04923
- clDice — arXiv 2003.07311
- Pitfalls of topology-aware image segmentation — arXiv 2412.14619
