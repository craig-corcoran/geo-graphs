# Claude Code Instructions

## Collaboration style

- **Ask clarifying questions liberally, before acting.** Don't assume intent —
  confirm it. When the request is ambiguous, or when there are aspects of the
  problem the user may not have considered, ask first.
- **Walk through design decisions rather than making them silently.** Surface
  tradeoffs, explain the reasoning, let the user choose.
- **The user picks the winner on substantive choices.** For sweeps, parameter
  tuning, and algorithm comparisons: report *all* candidates with their per-metric
  numbers, identify the **Pareto-frontier** configs across the relevant axes,
  and write a short tradeoff narrative for the top 2–3 contenders. Do not encode
  hard gates that auto-disqualify candidates on a value judgement (a threshold
  like "precision ≥ 0.80" bakes in which tradeoff is "correct" before the data
  is in hand). Sanity guards that filter degenerate results are fine; ranking
  that pre-empts the user's decision is not.
- **Clean slate.** Assume no backwards-compatibility constraints. Rename,
  restructure, or remove freely — no deprecation shims, no compatibility layers.
  When a refactor invalidates existing config or data files, do a clean-break
  migration (a one-shot script), not a shim.
- **When two options tie today, pick the one that survives scale.** If current
  measurements are within noise, decide on future-proofing grounds (asymptotic
  cost, operational headroom) and say that's why.

## Response style

- No preambles, no hedging, no summaries of what was asked.
- Direct answer → code. Explain only what's non-obvious.
- Technical terms precise. Prose tight.
- In planning and reasoning steps: compressed notation, fragments, and
  abbreviations are fine.
- User-facing output: concise but readable; normal code comments.

## Agent behavior

- **Use subagents to preserve context.** Delegate research, exploration, and
  multi-step subtasks so the parent context window stays focused on the main
  task.
- **Plan in the parent, implement in a subagent.** For non-trivial
  implementation work, build the plan in the parent conversation, then delegate
  each step (or group of independent steps). The parent orchestrates and
  reviews; it does not implement.
- **Worktree isolation when file changes need review** before they reach the
  working tree — but verify the worktree's base first. An isolated worktree
  branched from a stale base silently produces an unusable diff against the
  wrong codebase; when in doubt, implement in the active working tree or spawn a
  non-isolated agent.
- **Validate at the cheap setting; reserve the expensive one for the number that
  gets reported.** A run whose only purpose is "does this path execute and return
  a well-formed result?" doesn't need full resolution, the dense sampling the
  real metric requires, or the exact solver where an approximate one proves the
  wiring. Expose whatever dominates cost — input size, resolution, sample count,
  iteration or refinement budget, algorithm choice, external calls — as a
  parameter or env var whose *committed default is the expensive, correct
  setting*, so a smoke run turns it down at the call site without editing the
  committed artifact. Two consequences worth stating separately: a turned-down
  run proves plumbing, never quality, so its numbers never get reported as
  results; and the knob has to be a real parameter rather than an edit you
  remember to revert.

---

## Design philosophy: functional and stateless

### Pure functions over class methods

Logic and shared behavior belong in standalone functions, not methods on
classes. Dependencies (config, connections, collaborators) are passed explicitly
as parameters.

```python
# Do: standalone function, dependencies as parameters
def score(index: Index, query: Query) -> float:
    candidate = index.nearest(query)
    return similarity(candidate, query)


# Don't: method that couples logic to an interface
class Index(Protocol):
    def score(self, query: Query) -> float:
        candidate = self.nearest(query)  # calls self — OO coupling
        return similarity(candidate, query)
```

Default behavior that combines Protocol methods belongs in a standalone helper
function, not a method on the Protocol.

### Immutable data

All data containers use `@dataclass(frozen=True, slots=True)` or frozen Pydantic
models. Never mutate after construction.

- If a factory must return both a value and a resource handle, return a tuple:
  `(value, handle)`.
- In tests, use `model_copy(update=...)` rather than attribute assignment.

### Protocols over ABCs

Interfaces are `@runtime_checkable` Protocols (structural typing). Do not use
ABCs or `@abstractmethod`.

```python
# Do
@runtime_checkable
class Clusterer(Protocol):
    def cluster(self, points: np.ndarray) -> np.ndarray: ...


# Don't
class Clusterer(ABC):
    @abstractmethod
    def cluster(self, points: np.ndarray) -> np.ndarray:
        pass
```

Implementations satisfy Protocols structurally — never by inheriting from them:

```python
class DBSCANClusterer:  # NOT: class DBSCANClusterer(Clusterer)
    def cluster(self, points: np.ndarray) -> np.ndarray: ...
```

**Protocol design gotchas:**

- **Assert conformance at the definition site.** Structural typing checks at the
  *widening point* — wherever a concrete class enters a Protocol-typed slot
  (registry value, function parameter, annotated field). An implementation not
  yet wired anywhere is therefore checked nowhere. Pin it with one line beside
  the class: `_: type[Clusterer] = DBSCANClusterer`. Annotate `type[...]` rather
  than an instance, so it needs no constructor arguments and builds no runtime
  object. Don't reach for `isinstance` instead — `@runtime_checkable` compares
  method *names* only, so a `cluster()` with the wrong arity passes it, while the
  annotation reports the full signature diff.
- Protocol method *bodies* are not inherited by adapter-style implementors.
  Never rely on a Protocol default as a runtime fallback.
- Don't add introspection methods (`capabilities()`, `supports()`) to Protocols.
  Structural typing already answers "does this implement the interface"; a
  capability method turns a static contract into a runtime branch.
- Registries hold **lambda factories**, not pre-instantiated singletons, so each
  lookup yields a fresh instance:
  `CLUSTERER_REGISTRY.register("dbscan", lambda: DBSCANClusterer())`.

### Explicit dependency injection

No hidden state, no singletons, no module-level mutable state. Connections,
config, and collaborators are always passed as function or constructor
arguments.

### Concurrency follows the bottleneck

Where the work is CPU-bound, **sync is the default** and async is not a style
goal. Speed comes from vectorizing first (see below); reach for a process pool
only once a measurement says the array work is already tight and the remaining
cost is fan-out across independent units.

Async is for the genuinely I/O-bound edges — network calls, remote fetches,
anything where concurrency buys real wall-clock. Those are the places where
`async def` earns its complexity; wrap blocking clients in `asyncio.to_thread`
rather than letting async leak inward into the compute core. The cost of that
leak is mostly maintenance — `async` is viral, so one async leaf recolors every
caller above it — but there is a performance trap too: async buys concurrency for
*waiting*, never parallelism for *computing*, and a CPU-bound coroutine that
never awaits blocks the event loop outright, stalling the I/O it was supposed to
overlap.

### Style follows the package

The functional/stateless standard above applies to the main package. **Outside
it, match the patterns already in the file** unless told otherwise — introducing
Protocols and frozen dataclasses into code built on another set of conventions
just creates stylistic mismatch. When a seam crosses the boundary, the seam
matches the *outer* code's style while the helper it wraps keeps the functional
discipline.

Vendored third-party code is the hard case: it stays byte-for-byte in its
original style. When it serves as a reference implementation, being *unmodified*
is the whole point — restyling it, however small, destroys its standing as an
independent check. Adaptation belongs in a thin adapter beside it, which matches
our style while the vendored module keeps its own.

Existing stateful code inside the main package that isn't the focus of the
current change stays untouched — file a follow-up rather than burying an
unrelated refactor in the PR.

---

## Code style

### Declarative over imperative

Prefer expressions that describe *what* you want over statement sequences that
accumulate mutable state. In rough order of preference:

1. **Vectorized operations** (numpy/scipy, or polars for tabular results) —
   best when the data is a rectangular array of numbers.
2. **Comprehensions / generator expressions** — best when building a collection
   from an iterable.
3. **`for` loops** — when the logic has side effects, early exits, complex
   branching, or would force more than two nested levels in a comprehension.

Each step down is fine when the next level up sacrifices clarity; don't contort a
comprehension into unreadability just to avoid a loop.

Avoid (for loop with accumulator — walks the array one element at a time):

```python
totals = []
for seq in sequences:
    total = 0.0
    for i in range(len(seq) - 1):
        total += abs(seq[i + 1] - seq[i])
    totals.append(total)
```

Best (vectorized — the inner accumulation becomes two array ops):

```python
def total_variation(seq: np.ndarray) -> float:
    return float(np.abs(np.diff(np.asarray(seq, dtype=float))).sum())


totals = [total_variation(seq) for seq in sequences]
```

Note where the levels land. The *inner* loop was over a rectangular array, so it
vectorizes away completely. The *outer* iteration stays a comprehension: the
sequences are ragged, so forcing them into one padded matrix costs more clarity
than it buys. Vectorize the numeric core; iterate over the structure around it.

### No overloaded booleans

When a boolean field has two semantically distinct cases that both produce
`False` — "genuinely not X" vs "couldn't determine X" — use `Optional[bool]` and
reserve `None` for the not-determinable case. This applies even to
module-private fields with a single current consumer: threading `None` through
internal callers is a small cost; a later reader treating `False` as a positive
determination is a subtle bug. Doesn't apply when the boolean is genuinely
bimodal — `False` that honestly means "definitely not" is fine.

---

## Pluggable algorithmic pipelines

Subsystems that compose multiple algorithmic choices (preprocessing, candidate
generation, clustering, matching, scoring) expose those choices as
**Protocol + Registry** stages, not inline `if/elif`. Each stage:

1. Defines a `@runtime_checkable` Protocol with a single method taking explicit
   dependencies (no hidden state).
2. Registers implementations in a module-level
   `dict[str, Callable[[Config], Impl]]` (e.g. `CLUSTERER_REGISTRY`,
   `MATCHER_REGISTRY`).
3. Selects implementations by a config string key dispatched through the
   registry.
4. Ships a default implementation that reproduces pre-refactor behavior *within
   the new framework*. Config keys predating the refactor get a clean-break
   migration, not a deprecation shim.

Adding a new algorithm is one implementation class plus one `register()` call —
no edits to orchestration code. This buys: domain-specific overrides via config;
A/B comparison via config swap; agent-driven autoresearch loops that enumerate
configs, measure, rank, and iterate without code changes.

Note: "registry" is reserved for this software sense — the lookup pattern. Don't
use it as a synonym for structured or tabular data.

## Reproducibility and experiment practices

The practices that make the pluggable design actually pay off:

- **Cheap rerun entry points.** Every subsystem that depends on an expensive
  prior stage (network calls, large I/O, heavy computation) exposes a public
  function that runs against a *frozen checkpoint* of that prior stage — a cached
  response, a serialized intermediate. Agents and sweeps depend on this entry
  point; without it, every config change costs a full pipeline run, and tuning a
  late stage means paying for every earlier one.
- **Quality metrics separated by stage.** A bug in stage N often looks like a bug
  in stage N−1 unless the metrics separate them. Split the metrics so each
  stage's quality can regress independently — and so an error introduced upstream
  doesn't get credited to a downstream improvement.
- **Machine-readable before/after artifacts are the gate; notebooks are the
  presentation layer.** Acceptance decisions run off JSON/CSV written by the
  pipeline itself; notebooks read those artifacts rather than recomputing. Keeps
  humans and agents looking at the same numbers, and makes rerunning a comparison
  a one-line call instead of a notebook re-execution.
- **Frozen, content-hashed run configs.** Run identity is
  `timestamp + SHA-256(config + inputs)`. Prevents silent non-reproducibility
  across sweeps. When the run depends on a pinned external dependency, fold the
  resolved SHA into the run identity too.
- **Don't let a coarse-grained metric carry a fine-grained claim.** State what
  the metric actually measures; if the stronger claim matters, verify it directly
  rather than inferring it. Aggregate overlap scores are the classic trap: a
  single structural break can leave the aggregate almost unchanged while
  invalidating a whole class of downstream results. A high average is not
  evidence that the structure is intact — that claim needs a metric that measures
  structure.

---

## Documentation hygiene

- **Keep docstrings in sync with code.** When you add, rename, or remove a
  parameter, or change what a function or class does or returns, update the
  docstring in the same change. No stale signatures, no stale behavior
  descriptions, no undocumented new parameters. Public and internal APIs alike.
- **Google style.** Sections: `Args:`, `Returns:`, `Raises:`, `Yields:`,
  `Example:`. Omit a section when it adds nothing (no `Returns:` on `-> None`).
- **Coverage tiers:**

  | Tier | What gets it | Format |
  |------|-------------|--------|
  | **Full** | Public functions/methods with non-trivial behavior, multiple params, or complex return values; public classes (data models, Protocols, registries); modules with non-obvious structure | Summary line + `Args` / `Returns` / `Raises` as needed |
  | **Minimal** | Simple public functions where name + types are nearly self-documenting but a one-liner adds value; private helpers called from multiple sites; Protocol method stubs (the class gets the full doc) | One-line summary only |
  | **None** | Trivially self-documenting private helpers; simple property accessors; test functions (the test name is the doc); `__init__` when all params are on the class docstring; overrides that add no new behavior | — |

- **No tracker IDs in docs or deliverables.** Describe the work in its own terms
  ("the repeatable-demo recipe"), not by ticket number. Docs outlive tickets and
  reach readers who don't have the tracker. Ticket IDs are fine in commit
  messages, PR descriptions, and plans.

---

## Project memory: experiment log, backlog, plans

Three durable files carry decision history forward. Keeping them current is part
of the work, not an afterthought.

### `EXPERIMENT_LOG.md`

Append-only chronological record of quality and behavior investigations. Add a
new dated entry when running an experiment that measures pipeline behavior,
debugging a systematic quality issue, or landing a fix whose effect needs to be
tracked against numbers. Each entry records: the change under test, what was
measured, what the numbers say, and the next refinement it motivates. **Never
rewrite past entries — append new ones.**

### `BACKLOG.md`

Future ideas, deferred work, and scope exclusions. Add an entry when surfacing
work that's out of scope for the current task but worth preserving (follow-ups,
deferred redesigns, known residual issues). Strike through (`~~...~~`) or remove
entries as they're completed. **Check the backlog before proposing new
architectural work** — the idea may already be tracked, with context.

### Plan docs

- **A plan doc is owned by a specific task.** Edit in place only when revising
  *that* task's plan. If the plan is for a different task — even a closely
  related one — write a new file with a new name. Do not overwrite, replace, or
  delete the existing plan. The test is the task (goal, scope, motivating
  question), not the topic or the filename. When in doubt, ask before writing.
- **Same-task revisions: rewrite in place, don't append amendments.** For a plan
  still being actively revised, update the affected sections so the document
  reads as one coherent, current plan. Stale text plus a supersession pointer is
  harder to read than an up-to-date document. Reserve append-style amendments for
  plans that have already shipped or have other readers depending on the
  historical text.
- **Plans describe the work going forward.** No "was X, now Y" framing, no
  crossed-out sections showing what changed, no "source: prior plan" annotations,
  no status markers narrating the revision. Git log handles how we got here.
  Describing the *current code state* the work begins from is not history — that
  stays.
- If an overwrite is genuinely unavoidable (it should be rare), read the existing
  plan first and verify its actionable items are captured in the backlog
  item-by-item — not "looks fine." If items would be lost, ask first.

---

## Testing

- **Test files mirror source modules** — `test_<module>.py` per module.
- **Shared helpers in `conftest.py`** for patterns that recur across many test
  files (building result objects, creating mock clients). This avoids silent
  drift when the underlying models change. Shared helpers are plain functions,
  not fixtures, and require explicit import.
- **Local helpers stay local** when the helper encodes test-specific data that
  matters for assertions, or wraps a shared helper with file-specific defaults.
- **Scripts follow the same locality preference.** Duplicating a small utility
  across two scripts is preferable to a premature shared module; extract only
  when three or more scripts share identical, stable logic.
- **Tests that reach external services** carry the `network` marker. The default
  run excludes them (`addopts = "-m 'not network'"` in `pyproject.toml`); opt in
  with `-m network`. Keeping the default offline is what makes the fast loop fast
  and deterministic — never let an unmarked test reach the network.
- **Validate against a reference implementation, not against intuition.** Where
  the algorithm has a published reference, the test pins our output to it.
  Disagreements are findings: record which one is right and why in the experiment
  log before changing either side.
- New code keeps the suite green — no growing skip list.

---

## Version control

- **Atomic commits** when working across multiple modules.
- **Commit the recipe, not the artifacts.** Stage source: code, scripts, `.md`
  docs, configs. Leave generated outputs (`.csv`, `.pdf`, `.zip`, dumps, build
  products) untracked unless explicitly asked. Artifacts churn; the code that
  produced them is what belongs in history. When the line between hand-authored
  and emitted is unclear (a YAML that could be either), ask before staging.
- **Pre-commit hooks reformat files.** If a commit fails because a hook
  rewrote something, re-stage and commit again. **Never use `--no-verify`.**
- **Protected integration branches: offer the PR path first.** When credentials
  can bypass a branch-protection rule, the push succeeds and the bypass is
  recorded — the rule is invisible until after the fact. Name the protection rule
  and offer a feature branch plus PR before pushing anything beyond a trivial
  docs edit.
- Commit or push only when asked.

## Secrets

Never `cat`, `head`, `tail`, or otherwise read the contents of an `.env` file or
anything resembling a secrets file (`.env.local`, `*.env`, `credentials.*`). To
check what's available, surface **keys only** —
`grep -E '^[A-Z_]+=' .env | cut -d= -f1` — or test a specific variable with
`test -n "$VAR" && echo present`. If a value is needed at runtime, let the script
load it; don't handle it. A value read into the conversation lands in history and
any downstream telemetry, and has to be rotated.

---

## Repo operations are codified, not remembered

Every routine operation on the repo gets **one canonical invocation**, checked in
as a task-runner target. Nobody — human or agent — should have to reconstruct the
flag soup for a common task, and there should be exactly one right way to run it.

- **What earns a target.** Installing dependencies, formatting, linting, type
  checking, the fast validation loop, the full test suite, starting and stopping
  local dependencies, loading or wiping local data, and any cross-repo or
  environment-linking step. Rule of thumb: if it's run more than twice, or if
  getting it wrong is costly, it's a target — not a line in someone's shell
  history.
- **Invoke the target, not the underlying command.** Docs, CI, and agents all
  call the target. That makes the target the single place a change lands, and
  keeps local runs and CI from drifting apart. If you find yourself typing the
  raw command, either the target is missing (add it) or it's wrong (fix it).
- **Targets are self-documenting.** Each carries a one-line description surfaced
  by a `help` target, so `make help` is the discovery path. A new contributor
  should learn the repo's operations from the task runner, not from prose.
- **Composable, not monolithic.** Small targets that do one thing, plus a couple
  of aggregate targets that chain them — a fast pre-commit check distinct from
  the full suite. Keep the fast loop fast; that's what makes it get run.
- **Externally overridable.** Tool paths, service names, and file arguments come
  from variables with defaults (`VAR ?= default`) so a target can be pointed at a
  different input without editing the file.
- **One target set at the root.** This is a single-package repo, so the root
  `Makefile` is the whole surface. If it ever splits, each package owns targets
  with the same names, so `check` means the same thing everywhere.
- **Scoped side effects.** A target that touches shared or generated state says
  so in its description, and destructive variants are separate targets rather
  than flags on a safe one.

---
## Preferred stack

Concrete tool choices. Swap this section wholesale for a project on a different
stack; everything above is tool-agnostic.

| Concern | Choice |
|---|---|
| Language | Python 3.13+ (pinned in `.python-version`) |
| Environments / packaging | `uv` (venv at `.venv/`) |
| Task runner | `make` — one root `Makefile`; `## `-comment target descriptions feed `make help` |
| Numerics | `numpy` + `scipy`; domain libraries added as the problem requires |
| Dataframes | **polars over pandas** — if tabular results ever need a frame. Don't introduce a pandas dependency if it's easily avoidable. |
| Format / lint | `ruff` |
| Type checking | `pyright` |
| Tests | `pytest`, with a `network` marker for tests that reach external services |
| Logging | `loguru` (`from loguru import logger`), not stdlib `logging` |
| Plotting | `plotly` |
| Config | YAML, validated with Pydantic |
| Data models | Frozen Pydantic models / `@dataclass(frozen=True, slots=True)` |

Conventional target names: `make sync` (install dependency groups),
`make check` (lint + typecheck + offline tests — the fast loop),
`make test` (full suite, network tests included), and `make help` (discover the
rest).
