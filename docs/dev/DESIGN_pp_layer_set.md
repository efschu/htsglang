# DESIGN — non-contiguous PP layer ownership (the addressing blocker)

Built on `feat/pp-layer-set` off the train tip. Hermetic; no boot, no GPU.

`DESIGN_family_fullplan.md` (branch `docs/family-fullplan`) named two
independent blockers for the family full plan: the WIRE (barlink send/recv,
Slot-3) and the ADDRESSING — a pipeline stage has always been an interval, so
"all 48 linear-attention layers here, the 16 interleaved full-attention layers
there" could not be said at all. This is the addressing half.
**§4 below belongs in `DESIGN_family_fullplan.md` §2.1 when the branches
meet** — it answers the capacity question that spec left open.

## 1. The index contract

`SGLANG_PP_LAYER_SET`, per-stage sets separated by `;`, each a comma list of
ranges and singletons:

```
SGLANG_PP_LAYER_SET="0,1,2,4,5,6,…;3,7,11,…;35,39,…"
```

`get_pp_layer_set()` (`distributed/utils.py`) returns this stage's owned ids,
or **`None` when the variable is unset** — which is what keeps the contiguous
path byte-identical. `get_pp_indices` is untouched.

**The validation is the deliverable, not the parser.** Both ways of getting a
partition wrong are silent: a duplicated layer is computed twice and merely
costs time, and a missing layer becomes a `PPMissingLayer` pass-through
(`torch.nn.Identity`), so the model answers with that layer quietly skipped.
Refusals therefore name the exact layers: duplicates with the stages that claim
them, missing layers as a list, out-of-range ids, wrong stage count, backwards
ranges, non-integers.

## 2. Construction

`make_layers` already built a FULL-LENGTH `ModuleList` with placeholders
outside the owned range, which is why this was cheap: interleaved placeholders
are the same object as trailing ones. Under a set it emits a real layer at each
owned index and a placeholder elsewhere, and publishes `owned_layers` for
membership questions. `start_layer`/`end_layer` become FIRST and one-past-LAST
owned — which is what their consumers actually use them for ("am I the first
layer on this stage").

## 3. The consumer audit — one real hazard, one filed question

**FIXED: `num_effective_layers` (`model_executor/model_runner.py`).** It was
`end_layer - start_layer`, i.e. the SPAN, which equals the count only while a
stage is an interval. It feeds `layer_num=` for KV pool allocation at 8+ sites
in `model_runner_kv_cache_mixin.py`. Under the full plan's FA stage — 8 layers
spanning 29 — the pool would be sized for **29 layers instead of 8**, and the
boot would fail on memory it never needed. It now takes `len(owned)` when a set
is configured, reading the SAME parser rather than an attribute the models
would each have to propagate (two derivations is how they disagree).

**FILED, not forced: KV pool INDEXING under non-contiguous ownership.** Sizing
is fixed; addressing is not. The pool is indexed by layer, and a stage owning
`{3,7,11,…}` needs either a compacted index or a sparse map — that is a design
question with a correctness surface, not a mechanical rename. Adjacent prior
art is the #718/#719 pool-rebind work. **Not attempted here**; a wrong choice
would be silently wrong per layer, which is the worst available outcome.

### 3.1 The audit completed — one concept, ~80 inlined copies

Counting the shapes that can be wrong under a set, across `python/sglang/srt`:

| shape | sites | verdict |
| --- | --- | --- |
| `layer_id - self.start_layer` | **100** | **THE hazard** — see below |
| `range(self.start_layer, …)` | 56 | iterate the owned SET instead; 19 are vendor NPU |
| `end_layer - self.start_layer` | 5 | span-vs-count; the one on this rig's path is FIXED |
| `== self.start_layer` | 6 | SAFE — "am I the first owned layer", which `min(owned)` preserves |
| `== self.end_layer` | 2 | same shape, safe |

**The 100 `layer_id - start_layer` sites are one concept, not a hundred
problems**: translating a global layer id into a stage-LOCAL slot index. Under
a contiguous stage that translation is a subtraction. Under a set it is a RANK
LOOKUP — the position of `layer_id` within the sorted owned ids — and for a
stage owning `{3, 7, 11, …}` the subtraction gives layer 7 index 4 where its
its local index is 1.

Where they live: **63 in `mem_cache/memory_pool.py`**, 19 in the vendor
`hardware_backend/npu/memory_pool_npu.py`, 11 in
`mem_cache/dsa_cache_layer_split.py`, 5 in
`mem_cache/deepseek_v4_memory_pool.py`, 1 each in `model_runner.py` and
`swa_memory_pool.py`. **That is the KV pool**, which is exactly the design
question filed above — now with its concrete shape and count.

**The accessor already exists as a concept.**
`dsa_cache_layer_split.py:81-82` names it:

```python
def _local_layer_idx(self, layer_id: int) -> int:
    return layer_id - self.start_layer
```

So the fix is not inventing an abstraction — it is routing ~80 inlined
subtractions through an accessor that does a rank lookup instead. Mechanical in
form, but with a correctness surface: each site must genuinely mean "local slot
index" and not something else that happens to be spelled the same way, and a
site that means something else and is converted anyway is silently wrong for
one layer. **That is why this is filed rather than done in this cut** — 80
call-site conversions verified individually is its own slice, and doing it
carelessly is worse than not doing it.

Also noted, not fixed: `hardware_backend/npu/memory_pool_npu.py` iterates
`range(self.start_layer, …)` in six places; vendor-owned and not on this rig's
path. `deepseek_v2.py`, `glm4_moe_lite.py`, `kimi_linear.py`,
`bailing_moe_linear.py` each compute `total_num_layers = end - start`; the same
span-vs-count shape, per model, and each needs the same one-line treatment if
that model is ever placed non-contiguously. **INFERRED** — I did not read all
59 `start_layer` consumers; I audited the two grep shapes that can be wrong
(range loops and span arithmetic) and fixed the one on this rig's path.

## 4. GDN state capacity — ANALYTIC, and it PINCHES

The full-plan spec left this open. Derived from the checkpoint config and the
tree's own shape code (`configs/mamba_utils.py`: `temporal = (num_heads,
head_dim, state_size)`, `conv = (intermediate + 2·n_groups·state_size,
conv_kernel−1)`):

| quantity | value | source |
| --- | --- | --- |
| temporal state | 48 × 128 × 128 = 786 432 el | `linear_num_value_heads/​value_head_dim/​key_head_dim` |
| conv state | (6144 + 2·16·128) × 3 = 30 720 el | `linear_conv_kernel_dim 4` |
| per layer per slot | 817 152 elements | sum |
| **state dtype** | **float32** | `text_config.mamba_ssm_dtype: float32`, explicit in the checkpoint, and the tree's default (`mamba_utils.py:133`) |
| **per layer per slot** | **3.117 MiB** | 817 152 × 4 B |
| **48 layers per slot** | **149.6 MiB** | the full plan concentrates ALL GDN state on the 5090 |

Against the 5090 headroom the spec computed (free after GDN weights, #727 int8
vocab, arming floor 1728 and corridor 1024). The spec's **4738 MiB** was taken
against the NOMINAL 32768 MiB total; the card's live NVML total is **32607
MiB**, so the real figure is **4577 MiB**. The width-canon lesson applies to
totals as well as to widths — use what NVML reports, not the sticker:

| graph pool | slots (fp32, 48 layers) | on the nominal total |
| --- | --- | --- |
| 0 MiB | 30 | (31) |
| 1024 MiB | **23** | (24) |
| 1536 MiB | **20** | (21) |

**And without #727 the plan pinches hard.** On bf16 vocab the free figure is
2313 MiB nominal / **2152 MiB** on the NVML total, giving **~7 slots** at a
1 GiB graph pool. That moves #727 from "load-bearing" to **required**, now with
a number behind the word.

**Headline, stated loudly as asked: at fp32 state — which this checkpoint
explicitly selects — the full plan supports roughly 20–23 concurrent mamba
slots on the 5090.** That is the same order as the rig's existing
`--max-mamba-cache-size` settings, so it does not KILL the plan, but it leaves
little margin, and it is **2× the concentration of today's contiguous layout**,
where the 5090 holds 24 GDN layers rather than 48. Anyone planning higher
concurrency on the full plan should treat this as the binding constraint —
not the weights, which fit comfortably.

**ANALYTIC, pending metal confirmation.** It is arithmetic over the config and
the tree's shape code; it does not include allocator granularity, fragmentation
or any per-slot overhead beyond the state tensors.

## 5. The crossing schedule — BUILT against the shape, not a transport

`distributed/pp_crossing_schedule.py`. Given a layer map it emits every
ownership change in layer order, with a **per-pair slot** so a transport can
match a send to its recv without a global sequence number.

**31 is now executable rather than asserted.** The suite computes it from the
actual map: 16 crossings leave the GDN card, 15 return. A contiguous map through
the same function still yields `pp_size - 1`, which is the falsifier — the 31
has to come from non-contiguity and not from the function counting something
else.

**Reconciled with DESIGN_family_fullplan.md: the two counts are 31 and 32, and
both are right about different things.** 31 is the INTER-LAYER crossing count,
which is what this schedule emits and what a transport must match sends to. 32
is the full-forward MOVEMENT count, because `DESIGN_family_fullplan.md` §2.1
places `lm_head` on the 5090, so the terminal layer's output crosses back to the
head as well. Wherever a cost figure in either document is derived from 31, the
transport term is understated by one movement, i.e. by **~3 %**; that is below
the resolution of every verdict those figures support, so the numbers are left
as printed rather than re-derived, and this is the note saying so.

**Slot-3's corrections are baked in.** Cost is priced PER LINK
(`schedule_cost`), because the rig's edges differ and a schedule's cost depends
on WHICH pairs it uses, not only on how many crossings it makes. An unpriced
pair falls back to a default rather than costing zero: a modelling gap must
show up as a number instead of vanishing.

**The terminal-layer lever is measured, not claimed.** Whichever stage owns the
final layer is owed one fewer crossing (15 rather than 16 touching it), so
giving that half to the SLOWEST link is free — the split is 8/8 either way. On
this rig that means the x4-linked card should hold the half containing layer
63. The suite measures the difference rather than asserting it.

It also pins why 8/8 is free in COUNT but not in TIME: `len(s88) == len(s124)`
while the per-link cost of 8/8 is higher, which is exactly the trade Slot-3's
survey priced (8/8 stays, and buys KV at a cost in ms/pass).

**The double is deliberately strict.** `LoopbackLink` refuses a recv with no
sender (the bounded-wait requirement made visible), refuses a second recv on a
consumed slot, and refuses a reused slot within one pass. A double more
permissive than the transport it stands for proves nothing about a schedule
driven through it.

**Still not built**: the transport itself. This module moves no bytes and knows
no wire format — by design, so it could be finished before the wire exists.

## 6. The routing conversion — done, with two honest exclusions

§3.1's ~80 inlined subtractions are converted. `KVCache.local_slot` is the one
rule: the plain subtraction under contiguous ownership, the rank of the layer
within the owned set otherwise, and a **refusal** for a layer the stage does not
own. The old expression's real defect was not that it was wrong but that it
ANSWERED — an off-by-N index into a live KV buffer returns another layer's keys
without crashing or warning, for exactly one layer.

Converted (family **a**, local-slot semantics, all indexing a per-layer buffer):
`memory_pool.py` (64 sites), `dsa_cache_layer_split.py` (11, through its own
`_local_layer_idx`, which keeps its name and delegates),
`deepseek_v4_memory_pool.py` (5).

**Not** converted, deliberately (family **b**): `swa_memory_pool.py`. `SWAKVPool`
pins `start_layer = 0` and never calls `super().__init__`, so its subtraction is
an identity on a GLOBAL id and the accessor's map is never built there.
Converting it would have been the wrong conversion. Making that pool PP-aware is
a separate question; the reason is pinned at the site and in the suite, so a
later `super().__init__` re-opens it. The 19 vendor NPU sites are untouched.

### 6.1 The one thing the accessor does NOT fix: the PD wire format

`local_slot` maps global -> local. `layer_shard_start` needs the INVERSE, and
that direction is where non-contiguous ownership stops being an indexing
problem. `disaggregation/prefill.py:170` builds the transfer descriptor as
`prefill_start_layer + len(kv_data_ptrs)` — a start plus a **count**, read on
the wire as a contiguous global range.

For the family plan's second FA stage (layers 35..63 step 4: start 35, count 8)
that pair claims layers 36..42 the stage does not own and omits 47..63 that it
does. No index translation repairs this; the descriptor cannot represent a set.
So `shard_start_global` **refuses** it rather than emitting a plausible range
that would corrupt a KV transfer silently.

This is scoped OUT of the family plan rather than blocking it: the plan is
single-node PP+TP with no prefill/decode disaggregation, so it never builds this
descriptor. Carrying a layer SET across the PD wire is a real design question
(descriptor format + both endpoints), and it is open.

## 7. Iteration — the other half, and why a guard beats 34 edits

Translation (§6) fixed global -> local INDEXING. Iteration is separate:
`for i in range(self.start_layer, self.end_layer)` is correct only while
ownership is an interval.

`owned_layer_ids(layers, start, end)` reads the `owned_layers` frozenset
`make_layers` already publishes. Contiguous ownership returns the identical
`range` object; a set returns the owned ids in ascending order (order is
semantic, and a set has none, so the helper guarantees it rather than the
caller).

**A correction to my own note in `make_layers`.** It said interleaved
placeholders "are the same object as placeholders at the ends". True as
objects, false in the way that matters. `PPMissingLayer.forward` returns its
FIRST argument, and that was harmless only because it was never invoked --
loop bounds and ownership were the same thing, so placeholders sat outside the
iterated interval. A gapped set is the first case where a loop can reach one:

* `hidden_states, residual = layer(positions, hidden_states, ...)` raises a
  confusing unpack error, and
* `hidden_states = layer(positions, hidden_states, forward_batch)` --
  `orion.py:270`, `persimmon.py:253`, `phi3_small.py:356` -- SILENTLY
  substitutes the positions tensor for the hidden states.

So the span is not merely wasteful to iterate; it is wrong.

**Converted:** `qwen3_5.py`, the family plan's own model
(`Qwen3_5ForConditionalGeneration` is what the Qwen3.6-27B checkpoints
declare). **Not converted:** the other ~33 model forward loops.

That is a deliberate choice, not an omission. Those models cannot be executed
on this rig, so a conversion in them would be unverifiable here -- the same
argument that keeps the 19 vendor NPU sites untouched (`hardware_backend/npu/`:
vendor code, unreachable on this hardware, so a conversion could not be
tested). Instead the failure is made loud at ONE point: placeholders built
under non-contiguous ownership are armed at construction, and invoking one
raises a `RuntimeError` naming the layer and pointing at `owned_layer_ids`.
Contiguous ownership arms nothing, so the default path keeps the pass-through
byte-for-byte. A missed loop is then a named error at first forward instead of
a corrupted hidden state -- which is the property that actually matters, and it
holds for all ~33 without editing any of them.

## 8. The audit in §3.1 was NOT complete

§3.1 counted ~100 translation sites and located them in `mem_cache/`,
`model_runner.py` and `swa_memory_pool.py`. A second sweep found raw
`- start_layer` translations OUTSIDE that scope, which the tally therefore
never covered:

| site | note |
| --- | --- |
| `layers/attention/aiter_utils.py:147-148` | `sub_pool.k_buffer[id - sub_pool.start_layer]`; ROCm path, not this rig |
| `layers/attention/flashinfer_backend.py:2845-2851` | `_wl_full_layer_idx`, after a second mapping through `_transfer_full_attention_id` |
| `layers/attention/flashinfer_backend.py:3321-3329` | `_sess_full_layer_idx`, same shape |

The flashinfer pair is on this rig's backend AND on the hybrid linear/full
attention path the family plan uses, so it matters. It is filed rather than
converted: both compose a raw subtraction with `_transfer_full_attention_id`,
a SECOND translation whose interaction with set ownership I have not
established, and the rules of engagement say file an ambiguous site rather
than force it. These remain a silent-wrongness risk under a layer set and
should be the next slice.

Also found, all LOUD under a set rather than silent, so lower priority:
`routed_experts_weights_of_layer` dicts in 7 model files (indexing a
placeholder's `.mlp` raises `AttributeError`), `deepseek_v2.py:2549`
(`next_full_attention_layer_id` built from consecutive SPAN ids),
`deepseek_v4.py:2611`, `bailing_moe_linear.py:1119`,
`deepseek_weight_loader.py:484`, `quest_algorithm.py:36` (over-allocates, does
not corrupt), and `model_runner.py:1629` (`adjust_hybrid_swa_layers_for_pp`,
which also carries a pre-existing `end_layer + 1` worth a separate look).

### 8.1 A defect the accessor itself introduced: sub-pool layer frames

Following the flashinfer sites down produced a concrete instance of the filed
"KV pool layout" question, and it is one I created.

`KVCache.__init__` attaches an ownership map keyed by GLOBAL layer ids to every
pool. That is right for a pool the model addresses with global ids. It is wrong
for a SUB-pool, because its wrapper re-indexes first: `HybridLinearKVPool`
(`memory_pool.py:3796`) maps a global id through
`full_attention_layer_id_mapping` into a DENSE 0..N-1 full-attention index and
then calls `full_kv_pool.get_key_buffer(mapped)`. Under a layer set the map
holds `{35: 0, 39: 1, ...}` while the arriving id is `0..7`, so every lookup
misses.

It fails LOUDLY -- the accessor refuses an unowned layer rather than answering
-- which is the property that made this findable at all. But it would break the
family plan's own pool, since the plan is exactly GDN + full attention.

Fixed by `mark_as_sub_pool(pool)`: a sub-pool carries no global ownership map,
so its accessor degenerates to the plain subtraction inside its own dense
frame, which is what its buffers are laid out for. Applied in
`HybridLinearKVPool` and to both `SWAKVPool` sub-pools. On the contiguous path
there is no map to drop and it is a no-op. Verified at unit level against the
frame mismatch; NOT verified by a boot.

### 8.2 The flashinfer pair: the index's full journey

Written down before touching the code, because §8.1 showed the hazard is the
FRAME an id is in, not the arithmetic.

**Hop 1 -- what produces the incoming id.** `layer.layer_id` on a
`RadixAttention` is a GLOBAL model layer id. It is never stage-local: the layer
module is constructed with its global index in `make_layers`.

**Hop 2 -- what the pool does with it.** Two shapes, and they differ:

| helper | pool it indexes | chain |
| --- | --- | --- |
| `_wl_full_layer_idx` (`:2845`) | `_wl_full_pool = token_to_kv_pool.full_kv_pool` -- always the SUB-pool | global id -> `_transfer_full_attention_id` -> DENSE full-attention index -> `- start_layer` |
| `_sess_full_layer_idx` (`:3321`) branch A | `_sess_full_pool = getattr(pool, "full_kv_pool", pool)` -- the SUB-pool when a wrapper exists | same as above |
| `_sess_full_layer_idx` (`:3321`) branch B | `_sess_full_pool` falls back to `pool` ITSELF -- the MAIN, globally indexed pool | global id -> `- start_layer`, with no first hop at all |

So the two helpers are not the same site twice. The verdict differs per branch:

* **The sub-pool branches are already correct**, and for a reason worth stating
  rather than trusting: `_transfer_full_attention_id` returns the DENSE
  `full_attention_layer_id_mapping` index, and `HybridLinearKVPool` builds its
  sub-pool with `layer_num=` but no `start_layer=`, so `KVCache.__init__`'s
  `start_layer or 0` leaves it at 0. The subtraction subtracts ZERO. It is a
  no-op that merely looks like a translation.
* **Branch B is a genuine global -> local translation** on a globally indexed
  pool -- the same shape as the ~100 sites already converted, and it was missed
  because it hides behind a `getattr` fallback rather than reading as a pool
  access.

Both are routed through `local_slot` on the pool they actually index, so the
frame is chosen by the pool rather than by the call site. On the sub-pool
branches that degenerates to `mapped - 0` (unchanged); on branch B it becomes
the rank lookup, which is the fix.

**aiter_utils.py:147 has the identical two-branch shape** (`layers_mapping`
present -> sub-pool ids; absent -> `sub_pool = pool` and a global id). It is
left untouched, on the same grounds as the vendor NPU sites: it is the AMD/ROCm
path, unreachable on this NVIDIA rig, so a conversion there could not be
executed or tested here. It is not covered by the `PPMissingLayer` refusal --
that guards layer MODULE invocation, not KV index arithmetic -- so if the ROCm
path is ever brought under a layer set, branch B there needs the same one-line
routing applied here.

## 9. Integration readiness

**Everything below is DESK-PROVEN and BOOT-GATED.** Every claim rests on
hermetic unit evidence with can-fail proofs. Nothing here has run a model. No
number in this document was measured on hardware.

Desk-proven:

* the layer-set contract, its parser and its named refusals (§1, §2);
* `num_effective_layers` as a COUNT, not the span (the KV pool would otherwise
  be sized 3.6x too large for the FA stage);
* global -> local indexing through `KVCache.local_slot`, ~100 sites, including
  the derived pools and both flashinfer helpers (§6, §8.2);
* iteration through `owned_layer_ids`, plus a refusal at the single point where
  a missed loop would otherwise corrupt a forward pass (§7);
* sub-pool layer frames (§8.1);
* both PD descriptor paths refusing rather than mislabelling (§6.1);
* the crossing schedule against a loopback double (§5).

### 9.1 The first boot should NOT be the family layout

The family layout needs the per-layer wire, which does not exist yet. But
addressing can be separated from transport, and should be, because a failure in
a combined first boot would be ambiguous:

**Step 1 -- contiguous-as-set (bootable today, no new transport).** Express an
ordinary contiguous PP split through the SET mechanism:

    SGLANG_PP_LAYER_SET="0-31;32-63"   # with --pipeline-parallel-size 2

Ownership is still contiguous, so the existing stage-boundary transport carries
it unchanged, while the entire new path is exercised: parser, `make_layers` set
branch, `local_slot`, `owned_layer_ids`, pool sizing, the sub-pool frames.

*Acceptance:* byte-identical output against the same run with the env var
unset, same seed and prompt; identical KV pool sizing in the startup log; and
NO `PPMissingLayer` RuntimeError, since with contiguous ownership no
placeholder sits inside the iterated interval. That last one is the point: the
guard must stay silent here, and a firing guard means the loop conversion is
wrong, not the layout.

**Step 2 -- a gapped set (needs Slot-3's wire).** Only after step 1 is clean:

    SGLANG_PP_LAYER_SET="0-2,4-6,...;3,7,11,..."   # GDN vs FA, per §4

*Acceptance:* the 31 inter-layer crossings of §5 observed on the
wire (32 movements including the terminal output to `lm_head`), and generated text
matching the contiguous baseline for the same seed.

### 9.2 Refusals that SHOULD fire, and are part of acceptance

A guard nobody has seen fire is not evidence. Each of these should be triggered
once, deliberately:

| trigger | expected |
| --- | --- |
| any layer set + prefill/decode disaggregation | `NotImplementedError` naming the set and the descriptor |
| a layer set + a model whose loop was not converted (any but qwen3_5) | `RuntimeError` naming the layer and `owned_layer_ids` |
| a set that omits a layer, duplicates one, or has the wrong stage count | `PPLayerSetError` naming the layer and stages |

### 9.3 Known-unverifiable

The 19 vendor NPU sites and `aiter_utils.py:147` (AMD/ROCm) are untouched:
unreachable on this NVIDIA rig, so a conversion there could not be executed.
~33 model forward loops are unconverted for the same reason and are covered by
the refusal instead of by edits.

## 10. What remains

- The transport (§5): the schedule drives a `Link`; no wire exists yet.
- The two flashinfer translations in §8 — the remaining SILENT risk.
- KV-pool LAYOUT for non-contiguous FA ownership, beyond index translation.
  Adjacent prior art: #718/#719 pool-rebind.
- Carrying a layer SET across the PD wire (§6.1). Both descriptor paths now
  refuse rather than mislabel, so this is a missing feature, not a live bug.
