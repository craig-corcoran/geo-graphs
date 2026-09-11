# Road class inference on the extracted graph

A node-classification GNN over the road network: given overhead imagery and the
graph traced from it, label each road by its OSM `highway` class. The claim the
work exists to test is narrow and stated up front, because the measurement
apparatus below was built to make it falsifiable.

**The claim.** Message passing across junctions predicts road class better than
the same features read one road at a time.

Nothing else is claimed. Not that a GNN is the best model for road class, not
that road class is the most useful attribute, and not that the result
generalizes past Las Vegas.

## Why this question and not gap closing

The gap-closing line was measured out and closed; `plans/gnn_link_prediction.md`
records it. Its failure was not the idea but the evidence: the effect was real
at the link level and smaller than the noise where the metric reported it. This
project therefore answers the resolvability question before building anything,
and the answer is in hand.

## What is already measured

**The comparison is resolvable.** On the frozen split at 20 m spacing, the
minimum detectable macro-F1 difference at 80% power and alpha 0.05:

| unit errors are drawn on | sem | unpaired | rho=0.8 | rho=0.9 |
|---|---|---|---|---|
| node | 0.0145 | 0.0576 | 0.0258 | 0.0182 |
| way | 0.0202 | 0.0801 | 0.0358 | 0.0253 |
| component | 0.0285 | 0.1130 | 0.0505 | 0.0357 |

A line-graph model makes errors way-correlated by construction, so **0.0358 at
rho=0.8 is the number this project is held to**. An arm-to-arm difference under
about 3 macro-F1 points will not be reportable.

**Node spacing does not buy power.** Between 5 m and 40 m the noise floor moves
10%, because the bootstrap resamples chips and there are 226 of them. 20 m
stands, and finer sampling would quadruple the compute for a 4% narrower
interval.

**The signal exists and is linearly decodable.** A linear probe on the frozen
segmentation decoder's features reaches 0.3820 macro-F1 against a 0.1165
majority baseline, with a shuffled-label control at 0.1242. The decoder was
trained only to say road or not-road, so this is a floor.

**The split does not leak.** `splits/buffered_2560_1000_seed0.json`, digest
`66b989f5…`: 615 train chips, 226 val, 140 dropped. Four OSM ways of the
validation side's 2,371 also touch a training chip, so 0.67% of validation nodes
carry a memorisable label, against 45.4% under the shipped random split. This
matters more here than for segmentation: the class is constant along a way, so a
leak does not merely inflate a score, it hands the higher-capacity arm a win for
the wrong reason.

## What is built

| piece | where |
|---|---|
| Six-class vocabulary, node placement, class and cluster counts | `scripts/attr_resolvability.py` |
| Macro-F1, chip bootstrap, power calculation | same, verified against `scipy.stats.norm.ppf`, a per-class loop and a jackknife |
| Splitter Protocol and registry, frozen-split read/write with content hashes | `geo_graphs/split.py` |
| Candidate-split scoring: leakage, adjacency, class mix, power | `scripts/split_sweep.py` |
| Frozen decoder features at four scales via `grid_sample`, and the per-node baselines | `scripts/attr_probe.py` |
| OSM way identity through the noding pass | `osm.TaggedWay.osm_id`, `osm.node_sources` |

Environment: torch 2.13.0 with MPS available; `torch_geometric` is **not**
installed.

## The dataset the model runs on

Per side of the frozen split, OSM `drive` ways clipped to the chips and noded:

| | chips | line-graph nodes | line-graph edges | sampled points |
|---|---|---|---|---|
| train | 615 | 13,224 | 20,076 | 49,339 |
| val | 226 | 4,193 | 6,194 | 15,944 |

3.7 sampled points per piece, so a piece is about 75 m. Class counts on the
training side: residential 27,100, primary 6,510, tertiary 5,518, secondary
5,355, motorway 3,593, unclassified 1,263. One third of the macro average rests
on motorway and unclassified.

The graph is small enough for full-batch training in seconds, which is what
makes five seeds per arm affordable.

## Design decisions

Three of these change what the project can claim; the rest have defaults that do
not need revisiting.

### A node is a way piece, not a sampled point

The line graph: each noded piece is a node, and two pieces are adjacent when
they share an endpoint. Predictions are broadcast back to the 20 m points so the
metric stays node-level macro-F1 and stays comparable to the 0.4191 per-node
baseline.

The alternative, sampling points as nodes, makes the model's easiest win
*averaging along the chain*, because class is constant along a way. That is real
smoothing and a trivial pooling baseline already gets it. Pooling within a piece
before the model forces message passing to earn its keep on junction structure
instead.

### Inductive, with two disjoint graphs

One graph from the training chips' ways and one from the validation chips', with
message passing never crossing between them. The 1 km buffer guarantees the two
are genuinely disjoint.

The transductive alternative — one graph over every chip with the loss masked to
training nodes — is the standard node-classification setup and would undo the
buffer. The 140 dropped chips were paid so that no road bridges the two sides;
letting messages bridge it restores the leak in another form.

### The arms, and why C is not optional

| arm | isolates | status |
|---|---|---|
| A majority | the floor | 0.1165 |
| B per-node MLP | imagery, no graph | 0.4191 |
| C per-piece MLP | imagery plus within-way pooling, no message passing | to build |
| D GNN | the above plus propagation across junctions | to build |
| E GNN, topology only | whether imagery is needed at all | optional |
| F GNN, imagery and topology | the best model | optional |

**C against D is the only clean claim available.** Without C, comparing D to B
conflates pooling along the way with propagation across junctions, and pooling
is the trivial half.

### Defaults that do not need revisiting

**GraphSAGE-mean, depth swept over one, two and three layers.** Inductive by
construction, tolerant of varying degree, and its concat-with-self keeps the
node's own features intact — which matters when the per-node signal is already
0.42. Each layer reaches one junction further, about 75 m. GCN's symmetric
normalization is a poor fit for a road network; GAT is a reasonable second arm
if attention over junction neighbours is worth a look.

**Message passing hand-rolled with `index_add_`, not `torch_geometric`.** About
fifteen lines, no heavy dependency pinned to a torch version, and at 20,076
edges the library's optimizations buy nothing. The mechanism being visible is
the point. Swapping in PyG later is a contained change.

**Features frozen and precomputed.** Featurize once, then each arm trains in
seconds, which is what makes five seeds per arm affordable. Per piece: mean and
max pooling over its sampled points, concatenated, plus piece length, sinuosity
and endpoint degrees. Fine-tuning the decoder end to end is a follow-up.

**Inverse-frequency class weighting**, as the probe used. Macro-F1 is the metric
and an unweighted fit on a 54% majority class optimises a different average.

**Five seeds per arm.** The paired C-to-D difference is reported with its
measured standard error. This is the payoff of the earlier work: `rho`, the
between-arm correlation the detectable difference had to assume, becomes
something measured rather than posited.

## What a null looks like, and why it is publishable

Arm B already reaches 0.4191. For C against D to clear 0.0358, propagation
across junctions has to add something beyond what the imagery at a single piece
already says. Even odds at best.

A null here is a result, not a failure, and it is a result precisely because the
detectable difference was established before the arms were built. "Message
passing added 1.2 macro-F1 points against a 3.6 point detection threshold, on a
split that leaks 0.67%" is a finding. The same sentence without the threshold is
a shrug.

## Risks

**The frozen decoder caps every arm equally.** It was trained on a binary
objective, so all of A through F inherit whatever it failed to encode. This
biases against no arm in particular, and it does mean the absolute numbers
understate what the task admits.

**The rare classes carry a third of the macro average on thin evidence.**
Motorway has 506 validation nodes on 74 ways, unclassified 615 nodes. A model
that gets motorway right by luck on a handful of ways moves macro-F1 visibly.
Per-class F1 is reported alongside the macro figure for this reason.

**Blocked splits under-represent motorway systematically.** 6.0% of the AOI,
3.2% of this validation set. Any motorway number is measured on less of it than
the city contains.

**One AOI.** Las Vegas is a grid city with unusually regular block structure,
which is exactly the structure a junction-propagating model exploits. A result
here says little about Paris or Khartoum, both of which exist as 10-chip samples
only.
