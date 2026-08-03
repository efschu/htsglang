# #462 — the breakable MoE expert-offload graph route

Status: **DESK-WRITTEN, NEVER EXECUTED ON A CARD.** Gated OFF by default.
Nothing in this note is a measurement. The discharge path is
`docs/dev/TICKET_462_f2_and_replay.md`.

Branch `feat/breakable-offload-graph-462`, two slices:
`e55aaaf25a` (seam extraction, no behaviour change) and `f2de995671` (the
route).

---

## 1. Why this route and no other

`NOTE_452_desync_boot_refutation.md` §3 priced three ways to put a MoE expert
offload under CUDA graphs and rejected two:

| option | verdict |
|---|---|
| 1 — staged async H2D into a graph-stream double buffer | **do not build.** Yield ceiling ~1.25x, still 5.3x slower than eager, at 4.25 GiB/rank |
| 2 — CUDA conditional graph nodes | **not now.** The only mechanism that truly reconciles the two, but torch exposes no API |
| 3 — capture the compute, keep the fetch eager | **"the only option with a plausibly positive yield"** |

This note builds Option 3. `ANALYSE_456` §2.2 cell **#302b** states the fact it
rests on:

> A decode graph is captured against **slot addresses**, not against which
> expert occupies a slot at replay time. Any expert whose bytes are
> materialised into the scratch slots **eagerly, before replay runs**, is
> compatible with the graph.

The tension #452 measured is between "a graph must move the worst case" and
"an offload exists because moving only the miss is cheaper". This route does
not resolve that tension — it **sidesteps** it by taking the fetch out of the
graph entirely. The fetch keeps its measured eager volume (0.366–0.535
GiB/token); the graph keeps only the launch-overhead saving on everything
around it.

**The in-graph fetch stays refuted.** `refuse_capturable_offload_decode` is
untouched, and both spellings of it (`SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` and the
new `SGLANG_MOE_OFFLOAD_GRAPH_MODE=capturable`) hit the same refusal — pinned
by `test_both_spellings_of_the_refuted_path_still_refuse`.

## 2. What already existed, and what is new

The single most important finding of the analysis phase: **the slot arena is
not new, and that is why this works at all.**

`MoEExpertOffloadCache.install()` already builds a `[R+C]`-slot device buffer
per expert tensor and binds it into the layer's parameters
(`expert_offload.py`, `setattr(self.layer, attr, ...)`), then advertises
`layer.num_local_experts = R+C` so the grouped-GEMM sizes to the buffer. A
graph captured over that layer therefore **already** addresses slots. Nothing
had to be built for the "graphs address SLOTS" half of the ticket.

What was missing was the other half of the contract — publishing *which expert
is in which slot* through a fixed address, and getting the host-dependent
planning out of the captured region:

| piece | status |
|---|---|
| `[R+C]` slot arena at fixed device addresses | **existed** (`install()`) |
| eager fetch of only the missed rows | **existed** (`_fetch`) |
| residency planning | **existed** (`ExpertResidencyPlanner.resolve`) |
| a static per-shape buffer holding the slot vector | **new** (`BreakableOffloadArena`) |
| a host-side remap that avoids shipping a LUT | **new** (`remap_ids_host`) |
| the graph break itself | **new** (`breakable_moe_offload_fetch`) |
| boot preconditions | **new** (`validate_breakable_boot`) |

## 3. The mechanism, in order

`eager_on_graph` (already in the tree, already used by attention, mamba and the
DSA indexer) is the whole attachment. Called during a breakable capture it ends
the open segment, runs the decorated callable eagerly, appends it to
`cuda_graph._break_fns` so it re-runs before every replay, and opens a fresh
segment.

Per MoE layer, per step:

1. **break** — `breakable_moe_offload_fetch(cache, arena, topk_ids)` ends the
   segment captured so far.
2. **read** — `topk_ids.tolist()`: the one D2H rendezvous. Legal here; illegal
   two lines earlier.
3. **observe** — `_observe_routing()`: #390 router stats, Stage-1 hot freeze,
   #302a heat migration. Same code and same order as the eager path, because
   slice 1 extracted it instead of copying it.
4. **check** — the fixed-shape invariant. A captured segment cannot
   wave-split, so a scratch overflow is a wrong answer, not a slow one:
   `BreakableScratchOverflow` names the four numbers the remedy needs.
5. **resolve + publish** — `planner.resolve()`, then the slot vector is
   computed on the host (`remap_ids_host`) and copied into the bridge.
6. **fetch** — `_fetch(fetch_plan)`, unchanged, async on the copy stream.
7. **capture** — a fresh segment opens; `apply_fn` is captured, reading only
   the arena and the bridge, both at fixed addresses.

On replay only steps 2–6 re-run (as the registered break function) and the
captured segment replays around them.

## 4. Sync points — the number F2 will price

**1 host/device rendezvous + 1 pinned blocking copy, per MoE layer per step.**

The rendezvous is **irreducible**, and this is the honest framing of the
briefing's "43 baseline": the route does not beat 43, it *pays* 43. Deciding
which rows to fetch is host knowledge by construction — that is exactly what
makes the fetch cheap — and MoE routing is **sequential across layers**, since
layer L+1's router consumes layer L's output. There is no point in a step where
several layers' routing decisions are simultaneously available to batch into
one sync. Any scheme that removes the rendezvous is the in-graph fetch again,
which is refuted.

What the route *does* remove is the eager path's additional host blocking:

| | eager `run_waves` | breakable |
|---|---|---|
| D2H rendezvous (`topk_ids.tolist()`) | 1 | 1 |
| `_build_lut` → `idx.to(device, non_blocking=True)` | 1 | — |
| `_build_lut` → `val.to(device, non_blocking=True)` | 1 | — |
| pinned publish (blocking, `tokens x top_k` ints) | — | 1 |
| **host-blocking crossings per layer** | **3** | **2** |

The two `_build_lut` copies come out of `torch.from_numpy` memory, which is
**pageable**; `non_blocking=True` is honoured only for pinned memory, so both
block the host despite asking not to. Removing them is the mechanical win of
computing the remap host-side — the ids are already on the host, so the LUT
round trip buys nothing.

On DeepSeek-V4-Flash (43 MoE layers/rank): **129 host-blocking crossings/step →
86**, of which 43 are the irreducible rendezvous.

Constants live in code, not only here
(`breakable_offload.HOST_SYNCS_PER_LAYER_PER_STEP`,
`HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP`,
`EAGER_HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP`) and are pinned by tests
including a positive control that goes red if `_build_lut` stops issuing its
two transfers.

**This is a count, not a cost.** Whether 43 rendezvous plus 43 segment breaks
are cheaper than the launch overhead the graph removes is F2, unmeasured. F1's
5.3–8.4x ms/verify figure is a ceiling from Qwen3.6-35B-A3B and is **not** a
DSV4F number.

## 5. The combine seam for #302c

`ANALYSE_456` §2.2 names this so it is not rediscovered during implementation:
*the combine step must accept partial contributions from outside the graph.* A
CPU- or remote-computed expert's output was never a graph tensor, so it has to
be added after the replay that needed it.

Nothing here forecloses it, and the reason is structural rather than lucky:

* the dispatch **decision** is host knowledge already available at step 2 — it
  decides routing, not arithmetic, and #302c's own framing puts it strictly
  before replay;
* the **bridge is the one channel** between that eager decision and the
  captured compute. A foreign-dispatched expert is expressed by writing it a
  slot the captured apply contributes zero for — the mechanism already exists
  in the shape of the `-1` pad marker, which `remap_ids_host` passes through
  untouched and the pad-contract tests pin;
* the combine stays a plain sum over slot outputs, so a foreign contribution
  enters at a **second break** placed after the captured apply, added into the
  layer output like any other term.

What a #302c implementation still owes: the quality labelling. CPU and GPU
expert compute are **not** bit-identical (`ANALYSE_393`), and the #120 pattern
applies — which format's dispatch is bit-identical is a fact to state, not to
discover from a regression.

## 6. #286 consumption, and an open finding

The arena consumes `short_term_offload_register` and does not modify it
(territory rule):

* `describe_class("experts")` — the asset class is looked up, not restated. A
  class without a descriptor fails at import there, so this doubles as an
  assertion that the class still exists.
* `refuse_if_capture_active` — `arena.park()` routes through the register's
  gate rather than inventing a local rule, because #452 settled that page
  movement belongs BETWEEN replays.

**OPEN FINDING — the `experts` descriptor's `va_stable_required=False` is not
true under this route.** It is correct for the eager offload: the host pool is
the source of truth and the VRAM copy is a droppable cache, so the arena may
move. Under the breakable route a captured graph holds the arena's device
addresses, and moving it invalidates every graph that baked them in. The arena
therefore refuses a park unconditionally — not only under an active capture —
and both halves are pinned as tests
(`test_arena_refuses_to_park_while_a_capture_is_active`,
`test_arena_park_refuses_by_name_even_between_replays`). Whether the descriptor
should grow a per-route VA-stability qualifier is left to the #286 owner; this
note records the contradiction rather than silently working around it, in the
same style as #286's own unresolved `gdn_state_sets` finding.

**RESOLVED by #468 (2026-08-03), register side.** The descriptor did grow the
qualifier: `va_stable_when_graph_addressed=True` on `experts`, combined with
the permanent flag in the single predicate
`AssetClassDescriptor.va_stability_required(graph_addressed=...)`. A graph
family declares `addresses_classes` at registration and the register's own gate
(`refuse_if_move_illegal`, same `OffloadUnderCaptureRefused`, new
`ground=GROUND_GRAPH_ADDRESSED`) refuses the park — so the rule the arena
enforces locally is now the register's, one rule with two grounds rather than
two overlapping ones. `plan_spill` refuses to plan it as well. Nothing in this
module changed; `refuse_if_capture_active` still spells exactly the old
class-agnostic check, so `arena.park()`'s behaviour is unchanged. One
correction this surfaced: the arena's remedy text says to drop the decode
graphs via "the #286 rung-1 family eviction", but a rung-1 PARK preserves the
family's VAs by construction and therefore keeps the addresses live — the
family has to be unregistered. Details: `DESIGN_286_short_term_register.md` §8b.

## 7. Bug families designed against

| family | how it is addressed | falsifier |
|---|---|---|
| **shared buffer** (htccl `_get_out_buf`, `GraphSharedOutput`, `_DEQUANT_WS`) | each captured shape owns its `(bridge, stage)` pair — no process-wide staging buffer exists to alias; and the publish is deliberately **blocking**, so no ordering rule survives only as a comment | `test_two_capture_shapes_get_disjoint_bridges`, `test_preparing_one_bucket_does_not_disturb_another`, `test_every_bridge_owns_its_stage_exclusively`, `test_the_staged_publish_is_blocking` |
| **pad-slot** (#444a/#444e) | `-1` passes through the host remap untouched; padded rows carrying REAL ids are planned and fetched like any other row, and count against the scratch bound | `test_pad_rows_survive_as_pad`, `test_padded_rows_carrying_real_ids_are_still_fetched`, `test_pad_only_batch_fetches_nothing_and_stays_pad` |
| **rank-local before collective** | **not applicable, by construction** — the route adds no synchronization point and no collective. The break is rank-local throughout: every rank runs its own `tolist`, its own plan, its own fetch. Stated rather than omitted, because the audit is mandatory even when the answer is "none" |
| **desk-written / never executed** | labelled here, in the module docstring, in the catalog and in the ticket |

## 8. Deviations from `PLAN.md`

`PLAN.md` is a **probe** plan (Ticket 5b), not a construction plan — its §7.1
says explicitly "do not implement from this plan" and lists four code items as
sizing input. Treated as such. Its §7.1 (a)–(d):

| plan item | what was built |
|---|---|
| (a) breakable-aware boot guard distinguishing capture from graph break | built as `validate_breakable_boot`, plus two preconditions the plan did not name (decode backend must be `breakable`; prefill must be eager) |
| (b) `eager_on_graph`-wrapped MoE fetch | built, as specified |
| (c) device-side compact fetch descriptor replacing `topk_ids.tolist()` | **deliberately not built** — see below |
| (d) revisit the DSV4 capture-pool OOM rule | **not touched** — it rewrites `prefill.backend` only, and this route requires prefill eager anyway, so the rule and the route agree. Decode-breakable on DSV4 is unaffected by it |

**On (c) — the one substantive deviation.** The plan proposed shrinking the
sync payload from `O(T x top_k)` Python ints to an `O(C)` device-built
descriptor copied back once per layer. That does not remove the rendezvous
(the plan says so itself: "it can only be shrunk and moved"), and it *adds* a
device round trip to build the descriptor. Since the ids must reach the host
regardless, computing the remap there as well is strictly cheaper: it deletes
two pageable H2D copies rather than adding a device kernel. The payload at a
decode operating point is 6–96 ints, so the `O(T x k)` vs `O(C)` distinction is
not the term that matters — the *rendezvous* is, and neither design removes it.
Recorded as a decision, not an oversight.

Also from the plan, and confirmed still true: `PLAN.md` Appendix A's claim that
the catalog has "16 sections, there is no §17" is **stale** — §17 exists at
`FEATURE_CATALOG.md:555`. It was read against `/spinning/wt-video-probe`, an
older tree.

## 9. BOOT-PENDING

Everything. Specifically:

- capture has never been attempted — the `eager_on_graph` interaction is
  argued from its source, not observed;
- the bridge is allocated in the eager break region on the assumption that
  allocations there are not captured into a graph mempool. This follows from
  `eager_on_graph`'s own handling of its break output ("allocated between graph
  captures and is the static input address consumed by the next captured
  segment") but is **unverified on hardware**;
- no replay has run, so replay-without-recapture is unproven;
- the pinned-stage branch of the publish never executes without CUDA, so its
  hermetic coverage asserts the blocking *contract*, not the DMA behaviour;
- no ms/verify number exists, and none may be quoted until F2.
