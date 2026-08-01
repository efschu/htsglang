# DESIGN #140 — active weightless workers, and phase-multiplexed VRAM slots

Design and task cuts only, no implementation. Discharges the
feature-analysis-file duty for #140.

Two ideas that share one question — *what can a card do when it holds no
weights, and can the same VRAM hold different things at different times* —
composed over machinery that already exists.

---

## 1. What a weightless worker is today

The Variant-C lane (#115/#131/#136/#143) splits a TP/DCP group into two
structurally different roles. The head rank holds all weights and runs
Q/O-proj + FFN + GDN as collective-free TP=1. Every weightless worker holds
**only a KV token-shard** and computes attention over it, contributing zero
heads (`distributed/utils.py`, `weightless_head_counts`: the head-count vector
is `[total, 0, 0, ...]`).

**What crosses the wire per token, today** (`distributed/utils.py:309-311`):
a 0-count rank contributes an empty slice, so *"the Q all-gather becomes a
broadcast from the head rank and the O merge slices the merged output back to
the head rank only"*. So per decode step: **Q out, partial-O + LSE back**.
That is the budget any new work has to fit inside or beat.

**The constraint that shapes every candidate below**: this rig has **no
GPUDirect P2P** — *"All rig-1 GPU pairs are PHB, GPU0 sits on a x4 link, and
CUDA reports no GPUDirect P2P. NCCL stages through the host here"*
(`rig-runbook.md:1929-1930`). #114 already concluded that draft
disaggregation wants NVLink. Everything crossing cards here pays a
host-staging round trip.

---

## 2. Active-worker candidates

Judged on one axis first: **what moves per token**, because that is what the
absent P2P taxes.

### 2a. Ensemble / committee drafting — the strongest candidate

Weightless workers each hold a *cheap drafter* and propose candidate
continuations; the head rank verifies. The #156 cross-algorithm finding is the
reason to prefer a committee over one drafter: **specialization beats
one-size** — the DFLASH-multiturn arm was the *worst* arm, so a single drafter
tuned for one regime is wrong for another, while a committee can route by
content class.

*Wire cost per token*: draft token IDs and their logprobs — **tens of bytes**,
not tensors. This is the one candidate whose payload is small enough that
host-staging latency is amortized against a whole verify step.

*Why it fits the lane specifically*: a weightless rank already has no weights
competing for its VRAM, so a small drafter is close to free there, and the
head rank's verify is the thing that was always going to be the bottleneck.

**Not rig-gated.** Small payloads survive PHB. This is the candidate to build.

### 2b. Verification assistance — rig-gated

Split the verify forward so a weightless worker checks some candidates.

*Wire cost*: the verify needs the **weights**, which the worker does not have.
Either the weights move (absurd) or the hidden states do, per candidate per
layer. That is tensor traffic on every step.

**Rig-gated, and probably gated everywhere**: it inverts the lane's whole
premise (the head rank holds the weights *because* moving them is the
expensive thing). Excluded with a named reason.

### 2c. Prefetch / speculative KV work — partially free

A weightless worker already owns a KV shard. Give it work that operates on
KV *in place*: prefix-cache warming, eviction scoring, spill staging for
#236, compaction ahead of a reshard.

*Wire cost*: **zero for the work itself** — it acts on data the worker already
holds. Only the decision crosses, and decisions are scalars.

**Not rig-gated**, and the cheapest thing on this list, but the yield is
housekeeping rather than tokens/s. Worth doing when it removes work from the
head rank's critical path, not for its own sake.

### 2d. Idle-tenant work — already exists

A weightless rank with slack is an idle-workbench (#347) tenant like any
other. No new mechanism; it is a scheduling question against one ledger.

---

## 3. The correctness problem that gates 2a — cited, not resolved

Ensemble drafting is speculative decoding, so it inherits the lane's
**registered no-oracle problem**. The runbook states it
(`rig-runbook.md:2103`):

> **Do not use `topk 1` as a losslessness oracle on this configuration.** A
> no-speculation greedy run (self-identical 3/3) diverges from *both*
> speculative arms at temp 0 [...] The verify forward's batch shape (M=4/M=8
> vs the M=1 no-spec decode) changes GEMM tiling and flips near-tie argmax
> [...] A temp-0 difference between `topk 1` and `topk 2` here is therefore
> evidence of nothing on its own.

**This design does not resolve that and must not pretend to.** What it can do
is name the instrument that replaced identity: the #328 chain-quality gate —
graded content against a same-boot A-vs-A band, never text identity, never a
pre-registered constant — plus accept length from `meta_info` (#326). A
committee changes *which* tokens are proposed, so the acceptance rate is the
measurable, and the gate is what says the output did not get worse. Any cut
below that claims correctness without both instruments is unfunded.

---

## 4. Phase-multiplexed VRAM slots

The same bytes serving different roles at different times: prefill scratch →
decode drafter → KV headroom.

### The slot contract

A slot is **not** a new allocator. It is a post in the existing offload
register, whose classes already are `graph_rungs`, `drafter_heads`,
`lane_workspaces`, `cold_lane`, `experts`, `gdn_state_sets` — note that
`drafter_heads` and `lane_workspaces` are *exactly* the two roles a
phase multiplex would swap between. The contract:

1. **One ledger.** A slot's bytes are register posts and #330 dial capacity,
   never a second accounting — the same rule #305 states for model rungs.
2. **A slot has one owner per phase**, declared, not inferred. Two roles
   believing they own the same bytes is the failure the register exists to
   prevent.
3. **VA stability decides the route.** Items marked `va_stable_required`
   (captured graphs address them) must come back at the same address — the
   #93 tag route — and `RealMovementBackend` refuses `TensorPayload` for them.
   A slot that swaps a graph-addressed buffer is not a tensor copy.
4. **Transitions happen at the #309 quiesce boundary**, between ticks, never
   during a forward. Same instrument as #305's within-geometry rung changes
   and #364's between-tick executor: the world geometry does not change, only
   what occupies the bytes.

### What a flip costs, against what it buys

| flip | cost, from the measured record | buys |
| --- | --- | --- |
| KV headroom ↔ drafter | the #330 dial re-raise, **< 1 s** class (#297 delta) | a drafter's residency without a permanent KV tax |
| anything graph-addressed | **3-6 s** recapture (`ANALYSE_363:112`) | nothing, at decode cadence — this is the flip to avoid |
| prefill scratch ↔ decode role | dial-class if no graph is involved | the prefill peak stops being a decode-time reservation |

**The honest read: only the sub-second, non-graph flips are usable at phase
cadence.** A flip that forces recapture costs 3-6 s and a phase can be shorter
than that, so the multiplex must be restricted to slots whose occupants are
*not* graph-addressed — which is a design constraint, not a tuning parameter.
#363's regime signals are the eventual trigger, but only once the classifier
has been shown to discriminate (the same gate I put on #305 cut 4).

---

## 5. Task cuts, falsifier-first, effort/yield pairs

No thresholds — effort against yield as a ratio.

| # | cut | effort | yield |
| --- | --- | --- | --- |
| 1 | **Measure the head-staging round trip** for a token-ID-sized payload between two ranks on this rig. Pure transport, no drafter. | S | Decides 2a's whole premise. If tens of bytes cost more than a verify step saves, the committee is rig-gated too and the line stops here. Falsifier-first by construction. |
| 2 | **One weightless drafter, single proposer**, verified by the head rank, gated by #328 + accept length. | M | The minimum that proves an active weightless worker produces tokens at all. Also the first honest measurement of the no-oracle situation under a committee. |
| 3 | **Committee of two with content-class routing** (#156's specialization finding). | M-L | The actual idea. Only worth it if cut 2's accept length beats the single-drafter baseline. |
| 4 | **Slot contract in the register**: declare a slot, one owner per phase, refuse graph-addressed occupants. No movement. | S | The #111 seam: makes the multiplex expressible and refusable before anything moves. |
| 5 | **KV-headroom ↔ drafter flip** at the #309 boundary, dial-class only. | M | The one flip the measured record says is affordable. |
| 6 | Regime-triggered flips (#363). | M | Only after #363 discriminates. |

Cuts 1 and 4 are cheap and independently useful; cut 1 can kill the expensive
half of the doc in one window.

---

## 6. Rig-gated vs general (rig-is-lower-bound)

* **Rig-gated here, fine elsewhere**: verification assistance (2b) and any
  candidate moving hidden states per step. On NVLink these are ordinary; on
  PHB with host staging they are not. Recorded as *this rig cannot*, **not**
  as *the feature cannot*.
* **General**: ensemble drafting (2a) — small payloads survive any
  interconnect; KV-local work (2c) — no wire at all; and the slot contract
  (§4), which is arithmetic over a ledger.
* **General constraint, not rig-specific**: the recapture cost of
  graph-addressed flips. That is a property of CUDA graphs, and it bounds the
  multiplex on every rig.

---

## 7. Recommendation

**Ensemble/committee drafting on the weightless workers is the strongest
candidate**, because it is the only active role whose per-token payload
(draft token IDs and logprobs, tens of bytes) survives a rig with no P2P,
and because #156 already measured that specialization beats one drafter.

Do **cut 1** first — one window, pure transport, and it can falsify the
premise before any drafter exists. Then cut 4, which is desk-shaped and makes
the slot idea expressible. Everything else waits on those two, and nothing in
here claims correctness without the #328 gate and `meta_info` accept length,
because the no-oracle problem (§3) is registered and unresolved.
