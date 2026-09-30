# Graph layers: how stages after the graph add to it

**Status:** ADOPTED (2026-09-30). First layer: `gaussians`, written by the `handles` stage.
**Date:** 2026-09-30
**Serves:** G2 of `docs/design/2026-09-20-conceptgraphs-verdict-and-graph-plan.md` (the graph
honours an edit), and every later stage that learns something about an entity (captioning in G3)

---

## 1. The problem

The graph is the scene's record, and the pipeline keeps learning about its entities after the
graph stage has built it. The first case is the render: which Gaussians an entity *is* can only be
known once the scene is trained, and training depends on the graph (optimization reads it; the id
maps that assignment uses are its entities projected into each frame). So the graph cannot be
built after render, and it cannot know its Gaussians when it is built.

Two ways out were weighed:

- **a separate artifact** per later stage, joined with the graph at read time;
- **the graph grows**: later stages add to it.

The second matches what the graph is for: one record that answers questions about an entity,
with the evidence for each answer attached. It is safe for resume because an artifact's digest
(`reifyscene/api/contracts.py`) covers its contract record -- stage, settings, inputs, upstream
digests -- not its content; adding attributes to the graph does not make the stages built on it
stale. What needs a rule is not resume but *authorship*: many stages writing one document.

Birthing the graph at the start of the pipeline and letting every stage evolve it was also
considered and deferred (§5).

## 2. The rule

1. **The graph stage builds the graph.** Its nodes, edges, and its own attributes and metadata
   are its facts. It is the only writer of them.
2. **A later stage adds a layer.** A layer has one name, and writes only:
   - `node.attrs[<layer>]` on the nodes it describes;
   - `metadata["layers"][<layer>]`: which stage wrote it, when, what it was computed from
     (enough to tell when that source has changed), its method and parameters, and which
     capability grades it `provides`;
   - files under `<graph>/layers/`, named for the layer, for anything too large for the
     document a person reads.
3. **A layer never changes another stage's facts** -- not a node's position, not an edge, not
   another layer. When it disagrees with them, it says so in its own attributes.
4. **A layer is replaced whole.** Re-running the stage rewrites its key on every node, its
   metadata entry and its files; nothing is merged with the previous run of it.
5. **A layer records its source, and readers check it.** A layer is only as current as what it
   was computed from. Readers that act on it (an edit) compare the recorded source with what they
   are about to act on and refuse a mismatch; readers that only report it (a query) say which
   source it describes.
6. **Capability grades a layer provides are declared, not overwritten.** The graph stage marks a
   grade it cannot give as `pending` with the layer that will provide it (`renderable:
   "pending: gaussians layer"`); the layer's grade lives in its own `provides`. A reader takes
   the layer's grade when the layer is present and current, the graph's otherwise.
7. **A re-built graph has no layers.** When the graph stage re-runs, it writes a fresh graph;
   every layer stage downstream re-runs, since the graph is among its upstream. Entities may be
   renumbered or merged, so old layers would describe nodes that no longer exist.

## 3. The `gaussians` layer

Written by the `handles` stage, after render (`reifyscene/graph/handles.py`).

| where | what |
|---|---|
| `node.attrs["gaussians"]` | `status` (`grounded` / `unassignable`), `fg`, `bg` counts, `assigned_fraction`, `filter_dropped`, and `reason` when the entity is not in the id maps |
| `layers/gaussians.npz` | each entity's fg and bg index arrays |
| `metadata["layers"]["gaussians"]` | checkpoint folder + content digest, method (FlashSplat evidence majority + floater filter), slackness, `provides.renderable` |

The source that matters is the checkpoint: indices refer to one trained scene's Gaussians in one
order. `load_handles` refuses a checkpoint whose digest differs (`StaleHandles`), and the editor
then computes the selection itself, as it did before the layer existed.

The layer holds the default selection (slackness 0, filter on). An edit asked for with other
parameters computes its own and does not touch the layer.

## 4. Adding a layer

- Name it for what it describes (`gaussians`, `captions`), not for the stage.
- Give it a stage with a contract whose upstream includes `graph` and whatever it was
  computed from.
- Document its attributes in a table like §3's, in the design doc that introduces it.
- Test that it leaves every other attribute and edge of the graph unchanged.

## 5. Open

1. **Birth the graph earlier.** The graph could exist from the moment entities are stable (after
   `reid`), with `objects`, `humans`, `semantics` and `tracking` each adding a layer instead of
   feeding one assembling stage. Deferred until `gaussians` and captioning have tested the rule:
   it touches every stage and resume, and identity is still settling before `reid` (tracklets
   merge and split), which is the churn a layer should not have to follow.
2. **DynamicGSG** (`2026-09-20-...` §7.1) updates a Gaussian scene graph in place when the scene
   changes between visits; worth reading for how it keeps such updates consistent before §5.1 is
   decided.
