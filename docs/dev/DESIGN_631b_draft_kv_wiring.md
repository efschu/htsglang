# DESIGN_631b — wiring the canonical draft-KV handover

Companion to the contract landed in `214433ec2b`
(`disaggregation/draft_kv_canonical.py`) and to DESIGN_625 §7a. Written
BEFORE the wiring so the GPU window is spent executing rather than deciding.

Status: specification only. Nothing here is built. The #631a refusal stays in
force until §1 is satisfied in full.

## 0. What is already true, and one premise that is not

Established by reading the tree, not assumed:

- **The draft KV pool is ALREADY registered for transfer, on both arms.** Its
  contiguous buffers are appended to `kv_data_ptrs` / `kv_data_lens` /
  `kv_item_lens` as extra layers — `prefill.py:186-195` and
  `decode.py:439-448` — with the comment "The indices are always shared with
  a target model." So the draft pool already rides the main KV transfer, keyed
  by the same token indices.
- **That registration is gated only by layer-sharding**, not by anything to
  do with speculation: `transfer_draft_cache = not layer_shard_enabled or
  layer_shard_rank == layer_shard_size - 1` (`prefill.py:164-165`).
- Therefore the wiring slice is NOT "build a draft-KV transfer". It is
  "make the existing draft-KV transfer correct under this fork's head
  sharding, and stop refusing once it is".

**The premise still to verify, and step W0 of the build.** #631a's refusal
states the reason as: the MTP/EAGLE draft KV pool is *uneven-head-sharded*
(not DCP token-sharded), so its transfer would need general uneven head
reslicing. That is the fork's own statement of why it refused; it is quoted in
`pd_disaggregation_hook.py` and it is the whole basis for this slice. I have
NOT confirmed it in the pool construction code — the draft pool's head count
derivation did not fall out of the searches I ran, and I am not going to write
a design that treats a comment as a measurement. The constant-vs-code error in
R6 was exactly this shape.

So W0 is: read the draft KV pool's construction, establish its per-rank
`head_num` under `--rank-tp-ratio`, and record whether it is (a) uneven per
rank, (b) replicated full-head, or (c) evenly sharded. **The rest of this
document assumes (a).** If W0 finds (b), the draft pool already carries every
head on every rank, the canonical layout is a no-op on the wire, and the slice
collapses to lifting the refusal plus the §1 checks. If W0 finds (c), the
stock path may already work and #631a's refusal is over-broad. Either outcome
is cheaper than (a) and must be reported before building, not discovered
during.

## 0a. W0 EXECUTED — and it changes the recommended path

W0 needed no cards, so I did it rather than leaving it as a step. The answer
is definitive and it partly supersedes the contract I just landed. Recording
that plainly is worth more than defending the commit.

**The premise holds — for the DEFAULT draft layout only.** The draft pool's
head count is decided by one predicate,
`layers/dcp/owner.py:102 draft_pool_is_replicated(is_draft_worker, server_args)`
`= is_draft_worker and not draft_kv_layout_is_dcp(server_args)`, read by both
the pool sizing and the attention backend so the two cannot disagree:

- `--draft-kv-layout replicated` (**the default, and production**) -> True ->
  "the draft runner keeps its tiny full-context, HEAD-SHARDED 1-layer pool and
  every DCP branch stays off for it". Under `--rank-tp-ratio` that head shard
  is uneven. This is outcome (a), and #631a's stated reason is correct here.
- `--draft-kv-layout dcp` -> False -> "the draft pool takes the **identical
  target-model treatment**": full replicated kv-heads, token-sharded by the
  same weighted owner rule (`model_runner_kv_cache_mixin.py:2783-2791`).
  Outcome (b).

**Outcome (b) dissolves the problem instead of solving it.** If the draft pool
is shaped exactly like the target pool, it crosses the PD boundary by exactly
the mechanism that already carries the target pool: the extra-layers
registration in `prefill.py:186` / `decode.py:439` (shared token indices) plus
the `owned_ordinals` owner-rule scatter the R6 verdict established as already
wired. No canonical shipment, no head reslicing, nothing new on the wire.

**And production is inside `--draft-kv-layout dcp`'s covered shape.** #108's
gate admits exactly "the uneven-weighted-DCP path carrying a LINEAR draft
chain (MTP/NEXTN, topk == 1, one draft KV layer)"
(`server_args.py:7784-7799`). Production is NEXTN, `--speculative-eagle-topk
1`, one draft layer, `--rank-tp-ratio` installed, `dcp_size=3`. That is the
covered shape, item for item.

### Revised recommendation

Take **path B first**: require `--draft-kv-layout dcp` for a PD pair that
wants speculation, and let the draft KV ride the existing target-pool
mechanism. It reuses two things already built and validated (#108's owner-rule
kernels, the `owned_ordinals` scatter) instead of adding a third transfer
shape, and it removes the head axis from the problem rather than managing it.

**What that means for the contract in `214433ec2b`.** It is not wasted and it
is not wrong, but it is not the first tool to reach for:

- It remains the answer for the DEFAULT `replicated` draft layout, i.e. the
  head-sharded case #631a's text describes, if we ever need PD+spec without
  requiring #108.
- `check_full_head_shipment_is_justified` and the versioning discipline stay
  useful under path B too: path B still has to refuse a checkpoint or layout
  it cannot carry, and the version-refusal-not-reinterpretation rule is
  layout-agnostic.
- `local_head_window` stays independently valuable as the recorded workaround
  for #641, which is why that filing already points at it.

I am NOT deleting the contract on this finding. Path B has to be demonstrated
on-card before anything is retired, and retiring a proven desk artifact on an
unproven expectation is how the fallback disappears exactly when it is needed.

### What path B still has to prove (it is not free)

1. That the draft pool's DCP treatment is genuinely identical to the target's
   at the transfer boundary, not merely similar — the extra-layers
   registration assumes matching per-rank item lengths, and #345 is the
   standing example of a stride mismatch that corrupts silently rather than
   crashing.
2. That `--draft-kv-layout dcp` composes with `disaggregation_mode` at all;
   #108's gate was written for a monolithic server and no boot has combined
   them.
3. The measured cost: #108 recorded -67 % draft KV, but with a note that
   REPLICATED degrades above `tp > kv`. Here `kv=4 > tp=3`, so that note does
   not bite, but the acceptance-rate check (L6) still has to run.

## 0b. Owed item (2) SETTLED at the desk: the combination is UNCONSIDERED

**Composition verdict: neither accepted nor refused — unconsidered.** Every
parse-time interaction, exhaustively:

- `draft_kv_layout` appears in `server_args.py` exactly twice: the argument
  definition (`:1517`) and the #108 gate (`:7799`). The #108 gate contains no
  reference to disaggregation, prefill or decode.
- `pd_disaggregation_hook.py` contains no reference to `draft_kv_layout`.
- Only two files in `srt/` mention both concepts at all, and in both the
  mention is incidental (`server_args.py`, `model_runner.py`, the latter only
  for `reject_multi_layer_draft_kv_dcp`).

Nothing refuses the combination and nothing supports it. That is the dangerous
answer, so here is what breaks first, established from code rather than
predicted.

### The assumption that breaks first: a shared index space that is not shared

Three facts, each read directly:

1. **The transport is draft-blind.** `mooncake/conn.py` contains ZERO
   occurrences of "draft". The draft pool's buffers are appended to
   `kv_data_ptrs` / `kv_item_lens` as ordinary extra layers
   (`prefill.py:186-195`, `decode.py:439-448`) under the comment "The indices
   are always shared with a target model."
2. **The decode arm translates indices once, for everything.**
   `decode.py:1119-1125` rewrites `kv_indices` into compact owner-rule rows
   `(L // S) * (hi - lo) + (L % S - lo)` when `uneven_dcp_owner_bounds()` is
   set. That single translated array addresses every registered buffer,
   target layers and draft layers alike.
3. **The draft pool's row space depends on `--draft-kv-layout`**, and the
   default is NOT compact. `model_runner_kv_cache_mixin.py:2821-2823`:
   `if draft_pool_is_replicated(...) or not uneven_dcp_active(...): return
   int(global_rows)`, commented "Default 'replicated' keeps the early return,
   i.e. the full global context per rank, byte-identical." Under
   `--draft-kv-layout dcp` the same function falls through to
   `dcp_compact_pool_rows(...)` — the target pool's own row count.

Put together: **on a DCP decode arm with the DEFAULT draft layout, compact row
indices are applied to a pool sized to the full global context.** The two are
different coordinate systems, so every draft row lands at the wrong address,
with an offset that grows with the slot id — #345's right-token/wrong-slot
drift exactly, which is silent rather than a crash. It is latent today only
because #631a nulls `speculative_algorithm` on a PD arm, so no draft worker
exists and the extra-layers registration never runs.

Under `--draft-kv-layout dcp` the draft pool takes `dcp_compact_pool_rows`,
i.e. the same coordinate system as the target, so the shared indices are
correct BY CONSTRUCTION and no new translation code is needed. That is the
strongest argument for path B yet, and it is not an efficiency argument: path
B is the only one of the two layouts on which the existing shared-index
transfer is correct at all.

**This settles owed item (1).** The per-rank item-length question resolves the
same way: under `dcp` the draft pool carries `get_total_num_kv_heads()`
(`:2788-2791`), matching the target's treatment under
`uneven_dcp_kv_replicated`, so per-rank item lengths agree by the same
construction. The window can therefore CONFIRM rather than investigate.

### New refusal this uncovered: the arms must agree on `draft_kv_layout`

If the two arms run different `--draft-kv-layout` values, the draft pool's row
count AND its per-row byte length both differ between them, while the
transport indexes both with one shared array. Nothing checks this today, and
**guard 1 will not catch it**: `draft_kv_layout` is not part of
`compute_model_identity_hash`, correctly, since it is a parallelism decision
rather than a weights one.

So the wiring slice owes one more refusal: the arms must agree on
`draft_kv_layout`, refused at the handshake by name. Natural home is beside
the layout version in `DraftKvCanonicalLayout` — which is a second reason not
to retire that contract, since its version-negotiation channel is exactly
where this belongs.

## 1. The #631a refusal-lift criterion

The checkable list. The refusal comes off when **every** item is satisfied,
each by the named instrument. "Looks fine on a boot" satisfies nothing here.

| # | Condition | Instrument |
|---|---|---|
| L1 | Both arms exchange a `DraftKvCanonicalLayout` and `assert_compatible` passes before any draft byte moves. A version or geometry mismatch REFUSES. | Hermetic test (exists) + boot proof that the layout crosses the handshake; can-fail by skewing the version on one arm. |
| L2 | `check_full_head_shipment_is_justified` runs at boot against the LOADED geometry and passes. | Boot-time call site + can-fail with a synthetic wide-KV config. |
| L3 | Every canonical head has exactly one owner across the decode arm's ranks, for the deployed `(num_kv_heads, tp_size)`. | `local_head_window` partition test (exists) + a boot assertion on the deployed pair. |
| L4 | Draft-layer KV for **every prompt position** is present on the decode arm before its first speculative round. | On-card counter: draft KV rows written == prompt tokens, asserted per request, not sampled. |
| L5 | The received draft KV equals what the decode arm would have computed locally: a PD run and a monolithic run on the same prompt, greedy, produce the SAME accepted-token sequence. | Same-prompt A/B, monolithic vs PD pair, `meta_info` compared — not `spec_ema_accept_len`, which is not the accept length. |
| L6 | Acceptance rate on the PD pair is within the same-boot noise floor of the monolithic arm. | ms/round canon, A-vs-A floor first, warm-up draw discarded. A draft whose KV is subtly wrong still decodes correctly — it just accepts less — so L5 alone cannot catch it. |
| L7 | Any of L1-L4 unsatisfiable at runtime causes a REFUSAL, never a fallback to no-spec. | Can-fail per refusal path. |
| L8 | The #636 contract still holds unchanged (page_size 1, dcp == tp, mooncake, hisparse off). | Existing gate; no new exemption. |
| L10 | Both arms run the SAME `--draft-kv-layout`, refused by name at the handshake if not. Guard 1 cannot cover it: `draft_kv_layout` is deliberately not part of `compute_model_identity_hash`. | Hermetic test + can-fail by skewing the layout on one arm. |
| L11 | On a DCP decode arm, `--draft-kv-layout` is `dcp`. Under `replicated` the shared transfer indices address a full-context pool with compact rows (§0b) — silent wrong-slot writes. | Boot refusal; can-fail with `replicated` + `dcp_size > 1` + `disaggregation_mode decode`. |
| L9 | Monolithic servers are byte-identical. The standing production boot is a monolithic NEXTN server. | Base-vs-branch by name, not by count. |

L6 is the one most likely to be skipped and the one that matters most: a draft
KV that is wrong in a way that does not crash shows up ONLY as a lower
acceptance rate, which reads as "this rig is slow" rather than as a defect.
That is the same silent-degradation class as #631a's original auto-disable,
and lifting the refusal without L6 would reintroduce it in a new place.

## 2. Wiring specification

### W1 — Who computes the draft KV: the PREFILL arm

Non-negotiable, and the reason is the one that decided variant (iii): the
decode arm CANNOT compute it. It receives the main model's K and V, and K/V
are projections that do not invert back into hidden states. The prefill arm
already holds the hidden states in flight, so the marginal cost is one draft
layer on top of the model's full depth.

Consequence: the prefill arm must LOAD the draft layer's weights even though
it never drafts. That is one layer, and it must be an explicit, logged
decision at boot — not an incidental side effect of the spec flags — because
a PP prefill group's stage weights are already budgeted (B3).

### W2 — Where it enters the handover: the existing draft-pool registration

Reuse `prefill.py:186-195` / `decode.py:439-448` rather than adding a second
path. The draft pool already rides the main transfer as extra layers on shared
token indices, which is exactly the shape needed. What changes is only what
the buffers MEAN:

- The prefill arm registers its draft pool in the CANONICAL full-head layout.
- The decode arm registers a receive region in the same canonical layout,
  sized `bytes_per_token() * tokens`, NOT its own sharded pool.
- After the transfer lands, the decode arm slices `local_head_window(...)` out
  of the canonical region into its own draft pool.

The slice is a receive-side operation, which is the same shape as the
`owned_ordinals` scatter that already makes the token axis work (R6 verdict).
Head axis and token axis are then handled the same way: the receiver adapts,
the sender ships a canonical whole.

### W3 — What the decode arm does on receipt

1. Verify layout compatibility (L1) — before reading a byte.
2. Slice its `local_head_window` from the canonical region into its draft
   pool, at the token rows the main transfer already established.
3. Assert row count == prompt tokens (L4) and refuse the request on mismatch.
4. Only then admit the request to its first speculative round.

Step 4 is ordering, and ordering is where this family fails: admitting the
request before its draft KV is complete gives a first round that drafts from
uninitialised rows. That is #633's lesson — forward before you block — applied
to a different resource.

### W4 — Refusals that must exist

At boot: layout version unsupported; geometry not justified (L2); head
partition not exact (L3); #636 contract violated (L8); prefill arm asked to
ship draft KV without the draft layer's weights loaded (W1).

At handshake: peer layout mismatch (L1), naming both sides.

Per request, at runtime: draft KV row count != prompt tokens (L4).

Every one refuses. None falls back to no-spec — that is L7, and it is the
whole point of #631a.

## 3. Sequencing for the GPU window

The window is gated on #640 plus a quiet period, so this is what it should
execute, in order, with the desk work already done:

1. W0 (read-only, can be done NOW without cards — and should be).
2. Boot the PD pair with the #636 contract satisfied, spec still refused,
   and confirm the main KV handover is healthy on its own. Baseline first.
3. Wire W1-W3 behind the existing refusal.
4. Lift the refusal only after L1-L9 are each demonstrated.

Step 2 exists because a PD pair on this rig has never been booted in anger.
Debugging a draft-KV handover on top of an unproven main handover would
confound two unknowns, and the main one is the cheaper to establish.
