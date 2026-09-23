# Roadmap — in-house components, a second modality, and the contract that holds them

**Status:** PROPOSED — direction, not a schedule
**Date:** 2026-09-19
**Holds:** the direction agreed 2026-09-18/19 — three in-house implementations, audio as a second
modality, and the modality-general contract every one of them depends on
**Extends:** §3 (the contract) of `docs/design/2026-09-12-back-half-ladders-and-graph-contract.md`,
which the stage audit confirms stands, and §6 (work items) of
`docs/design/2026-09-13-stage-audit-and-end-to-end-plan.md`
**Detail lives elsewhere:** the first in-house component has its own proposal at
`docs/design/2026-09-18-learned-covisible-merge-proposal.md`
**The graph's own near-term plan:** `docs/design/2026-09-20-conceptgraphs-verdict-and-graph-plan.md` §6 — the ConceptGraphs verdict and the
vertical slice (answer a question, honour an edit) that the components below sit after

---

## 0. The one idea underneath all of it

The contract already states this for pixels. §3.4 gives every entity a **renderable handle** —
entity id → the Gaussians that *are* that entity — because the op-support matrix promises
`remove(entity)` is "honored — leaves a real hole". You cannot leave a hole in something you
cannot decompose.

Audio needs the same thing: removing a person has to remove what they said, which means the audio
must be separable by entity. A 4D NeRF has to honour the same op, which means it must be
compositional rather than one monolithic field. These are not three requirements. They are one:

> **Grounding law.** Every modality attached to the graph must be decomposable along entity
> boundaries. A modality that cannot answer *which part of me is this entity* cannot support the
> edit operations, and so cannot be attached — it can only be carried alongside, unedited.

Stating it once, now, is the cheap moment. Retrofitting it after a second modality is already
attached is the expensive one.

---

## 1. The contract change — grounding handles

`renderable` stops being a one-off field and becomes the first member of a family.

**A grounding handle** is, for one modality: entity id → the subset of that modality's
representation attributable to the entity, plus the three numbers §3.4 already reports — assigned
count, assigned fraction, and an explicit **`unassignable`** flag for entities the modality cannot
resolve. The flag is the important part: it is how a modality says "this entity is here but I
cannot isolate it", which is a different claim from "this entity is absent", and the two must
never collapse.

| handle | representation | status |
|---|---|---|
| `renderable` | Gaussian index set over the trained cloud | assignment landed, storage open (§4 of the contract) |
| `audible` | speech/sound intervals + a separation handle | new, §4 below |

**Two additions to the node/edge vocabulary:**

- **`utterance` node** — `text`, `[t_start, t_end]`, confidence — joined to its speaker by a
  **`spoken_by`** edge. A node and not an attribute, because the text is queryable content in its
  own right: *"what did she say about the chair?"* is a question about the utterance, not about
  the person.
- **`emitted_by`** for non-speech sound events, same shape.

**Deletion then comes for free**, from a law the contract already has: *a relation's validity
interval is intersected with both endpoints' presence, so an edge is never asserted outside it.*
Remove the person and their `spoken_by` edges die with them. No new deletion rule is needed —
which is the sign the abstraction is in the right place.

**The op-support matrix gains a column**, carried from the start for the same reason the render
column was: so the IR does not fracture silently.

| Op | 3DGS (ours) | 4D NeRF (§3) | audio (§4) |
|---|---|---|---|
| `remove(entity)` | honored — leaves a real hole | honored by construction (compositional) | mute (floor) / separation (top) |
| `move(entity, delta)` | static transform; dynamic bases open | canonical-space, propagates through time | not meaningful |
| `recolor` | not implemented | canonical-space | not meaningful |

---

## 2. In-house component 1 — the co-visible merge transformer

Full proposal: `docs/design/2026-09-18-learned-covisible-merge-proposal.md`.

The current co-visible merge rule recovers **31 of 247** same-object tracklet pairs (12.6%) with
**0 false merges of 3,351** — it catches exact duplicate masks and is structurally blind to
over-segmented *parts*, whose centroids sit further apart than any distance threshold allows. An
in-house transformer over the per-frame evidence sequence is the proposed rung above it.

**What it establishes beyond itself:** `reifyscene/learned/nn/` — attention and encoder blocks
written here, verified against PyTorch's as an oracle in the tests. Component 3 (§4) is the first
real test of whether those blocks were written generally enough.

---

## 3. In-house component 2 — a compositional 4D NeRF

`reifyscene/render/registry.py` holds one real backend behind four aliases. This is a genuine
second entry, implemented rather than ported.

**Compositional by requirement, not by taste.** Per-entity radiance fields plus a background
field, composited — because `remove(entity)` must leave a hole (§1). A single 4D field cannot do
it, so the grounding law decides the architecture before any quality argument does.

**A deformation field, not an explicit 4D decomposition.** Canonical space plus a
time-conditioned deformation MLP, over a HexPlane/K-Planes-style factorisation, for two reasons
specific to this pipeline rather than general ones:

- we already have per-entity trajectories to initialise and supervise the deformation;
- canonical-space editing propagates through time for free, which is exactly the edit semantics
  the project wants — edit the canonical entity, not each frame.

**Reuses:** VGGT poses and intrinsics, per-entity masks as compositional supervision, depth for
ray sampling bounds, and the existing dynamic masks.

**Written from scratch:** the volume rendering integral, positional encoding, hierarchical
sampling, ray batching, the deformation MLP.

**Evaluated against the existing instrument.** The covisibility-masked novel-view-synthesis
benchmark (`reifyscene/eval/benchmarks/covisibility_masked_novel_view_synthesis/`) already scores
held-out views; the NeRF rung goes through it, so it is compared to the 3DGS rung on the same
views by the same code rather than on its own terms.

**What is not claimed.** It will be slower than the 3DGS rung and will probably look worse. It is
also **not** proposed as the ladder floor — a compositional deformation NeRF is not simple, and a
floor rung should be robust above all. Its honest advantages are two: no gsplat CUDA-extension
dependency, and canonical-space edit semantics.

---

## 4. A second modality — audio

**Current state: hooks, no implementation.** `ingest/audio.py`'s `AudioExtractor.extract()`
returns `None`. `sync/audio_sync.py`'s `estimate_offset()` returns `(0.0, 0.0)`. `has_audio` sits
on `VideoStreamRef` and is probed — and the frame-sequence decoder hardcodes it `False`.

**The stage band**, each a ladder, split by what is worth building versus importing:

| Stage | Top rung | In-house |
|---|---|---|
| `audio_ingest` | ffmpeg waveform, resample, channels — and **finally implements the sync stub** by envelope cross-correlation | no |
| `speech` | Whisper (MIT), word-level timestamps | no — nothing is gained by reimplementing it |
| `diarization` | pyannote / NeMo → anonymous speaker clusters | no |
| `sound_events` | PANNs / BEATs — *"remove the dog"* must remove the barking | no |
| **`av_association`** | speech or sound → **visual entity** | **yes — component 3** |
| `audio_edit` | mute dominant intervals (floor) → target-speaker extraction (top) | partly |

**Component 3, `av_association`, is the project-specific part.** Diarization says *someone* spoke
from 4.2 s to 7.8 s; nothing in an audio model says *which visible person that was*. The evidence
is per-entity mouth motion over time against audio activity over time — two variable-length
sequences to align, the same shape as the co-visible merge, and the reason `learned/nn/` should be
written to be reused.

For non-speech the floor rung is cruder and worth stating plainly: sound-event class matched
against entity category, intersected with presence intervals. A bark during the dog's presence
interval is attributed to the dog. That is weak evidence and it should be marked as such in the
handle's confidence rather than dressed up.

**On removal, temper expectations.** Subtracting one voice from a mono mix with overlapping
speech is hard, and target-speaker extraction leaves artifacts. The floor — mute the intervals
where that speaker is dominant — is honest and usually enough to demonstrate the edit. Both rungs
exist and the manifest says which ran.

**The real cost is data.** DAVIS is image sequences with no audio, so the entire benchmark
instrument this project has built is silent. Audio needs its own evaluation from scratch:

- **AVA-ActiveSpeaker** — speaking/not-speaking labels, for `av_association`;
- **VoxConverse** or **AMI** — for diarization;
- **our own footage** — the only way to test the full *remove a person, their voice goes too*
  loop end to end. The capture scripts record video only today, so they need audio added.

**What audio does not change:** cross-view association remains an explicit non-claim of the
contract. Audio does not give it to us and must not be read as if it does.

---

## 5. Sequencing, and the risk of widening

**The risk, stated plainly.** The visual graph still cannot answer much: only `observed_by` edges
exist, the relation computer is unbuilt, the query layer is stubs, and `entity_gaussians` has zero
callers (`renderable: absent`). Adding a modality before the graph can answer a question about the
modality it already has is how a project gets wide instead of deep.

**The counterargument that survives it.** Only §1 needs to happen early, and §1 is a contract
change, not an implementation. Designing `audible` while `renderable` is still being wired is what
makes the abstraction general once instead of retrofitted twice. The rest of audio can wait.

**Proposed order:**

| | work | why here | depends on |
|---|---|---|---|
| 0 | **§1, the grounding law + handles** | cheap now, expensive to retrofit | nothing |
| 1 | visual close-out already in flight — ConceptGraphs baseline, relations, query | the graph must answer *something* before it answers across modalities | — |
| 2 | component 1, the co-visible transformer | attacks a measured defect; establishes `learned/nn/` | the cached pair table |
| 3 | audio integration stages (`audio_ingest`, `speech`, `diarization`, `sound_events`) | mostly wiring; gets real audio into the graph | §1 |
| 4 | component 3, `av_association` | the part nothing off the shelf does | 3, `learned/nn/` |
| 5 | component 2, the 4D NeRF | independent of everything above | poses, masks, the NVS benchmark |

Item 5 is deliberately unpinned: it touches only the render registry and the grounding law, so it
can slot wherever there is appetite for it.

---

## 6. What this adds up to

Three in-house implementations of three different kinds — a **learned associator** (sequence
modelling), a **differentiable renderer** (graphics and optimisation), and a **cross-modal
grounder** (multimodal) — around integrations with VGGT, SAM 2, CoTracker, DINOv2, SigLIP 2,
Sapiens, Whisper and ConceptGraphs, inside one system that does something real end to end.

That spread is deliberate and worth protecting when scope pressure arrives: three components of
three kinds demonstrate more than three of the same kind, and the integrations demonstrate
something the in-house work does not — that the system can absorb state-of-the-art tools rather
than compete with them.

---

## 7. Open decisions

1. **4D from the start, or static-first?** A static compositional NeRF is a smaller first target
   and the deformation field bolts on; but static-first means the first version cannot render our
   actual scenes, which are all dynamic. Leaning 4D from the start, accepting the slower first
   result.
2. **Audio corpora** — is AVA-ActiveSpeaker + VoxConverse the right pair, and do we shoot our own
   clip early (to have the end-to-end loop) or late (when the stages exist)?
3. **Does `audible` ship with §1 or with the first audio stage?** Shipping the field early with
   every entity `unassignable` is honest and makes the retrofit impossible; shipping it late keeps
   the contract from carrying a field nothing fills.
4. **Utterance granularity** — one node per utterance, or per speaker turn? Whisper segments,
   diarization turns and sentence boundaries do not agree, and the query surface will care.
