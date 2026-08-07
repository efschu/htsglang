# #616g — the wedge is a rank-divergent PREFIX, not a shifted collective

Specimen: 2026-08-06 21:52:25, production boot `/spinning/wt-530-serving`
@ 3752da399d (the guard fix from #616f), pids 60382/60384/60385.
Stacks and locals: `/spinning/wedge-catch-603b/wedge_20260806_215225_*`.

## 1. What the specimen says

`Bar1CollectiveStalled` fired on all three ranks at 21:53:39, abort word
CLEAN on every one — a stall, not a trip: a peer contribution never arrived.

    rank 0: op=all_reduce nbytes=17305600   = 1690 tokens
    rank 1: op=all_reduce nbytes=18616320   = 1818 tokens
    rank 2: op=all_reduce nbytes=18616320   = 1818 tokens

`nbytes / (5120 * 2)` divides exactly on every rank, so all three are real
`(num_tokens, hidden)` bf16 payloads and the token counts are 1690 / 1818 /
1818.

The py-spy stacks settle the question the numbers alone cannot. All three
ranks are parked at the SAME call site:

    all_reduce (barlink.py:1008)
    attention_tensor_model_parallel_all_reduce (communication_op.py:67)
    _gather_hidden_states_and_residual (communicator.py:1287)
    prepare_mlp (communicator.py:874)
    forward (qwen3_5.py:1241)

and the locals give the same `layer_idx: 7` on all three. The second dump
round, seconds later, has them at layers 9 / 9 / 8 — still moving — with each
rank's `nbytes` UNCHANGED from its first-round value.

That is the whole verdict:

* the per-rank size is constant across layers, so it is a property of the
  rank's whole forward, not of one collective;
* the ranks agree on the layer index, so the collective SEQUENCE is aligned.

**Sequence-count verdict: no pairing shift.** This is not a rank running one
collective ahead. It is one logical collective whose SHAPE differs per rank —
the ranks are reducing differently sized forward batches.

## 2. Where a rank-dependent token count comes from

`prepare_for_extend` computes the forward's token axis as

    input_ids = [r.get_fill_ids()[len(r.prefix_indices):] for r in reqs]
    extend_num_tokens = sum(len(ids) for ids in input_ids)

so the count is `sum(seq_len - matched_prefix_len)`. Everything on the right
is replicated EXCEPT `len(r.prefix_indices)`, which comes from
`tree_cache.match_prefix` (schedule_batch.py:1277) — i.e. from the CONTENT of
this rank's radix cache.

Under uneven TP/DCP the ranks' pools are different sizes by construction. The
same boot's allocation lines:

    TP0  #tokens: 179825      TP1  #tokens: 143860      TP2  #tokens: 136667

Two triggers then mutate those caches from rank-LOCAL state:

1. **Eviction** — `evict_from_tree_cache` (mem_cache/common.py) fires on
   `allocator.available_size() < num_tokens`. The DEMAND is replicated
   (`batch.extend_num_tokens`); the AVAILABILITY is this rank's own shard.
   The roomy rank declines an eviction the tight ranks take.

2. **hicache load-back** — `HiRadixCache.init_load_back` EXTENDS the device
   prefix, and only on the ranks whose own free device space accepts the
   load. This one needs no eviction at all to diverge the trees.

Both push in the same direction, and it is the direction the specimen shows:
rank 0 owns the largest pool, so it evicts least and loads back most, matches
the longest prefix, and computes the FEWEST new tokens — 1690 against 1818.
The earlier 21:08 specimen has the same asymmetry (rank 0 at 334 tokens
against 1229), which is why "rank 0 differs, peers agree" recurred.

Once the trees stop being replicas the divergence is self-sustaining, and
every per-layer TP all_reduce of the next extend forward is entered with
mismatched shapes. BAR1 spins on a contribution that cannot come.

## 3. Why the existing pins did not catch it

#603 (decode-mem) and #610 (prefill admission) made the DECISIONS
rank-uniform via a MIN-reduce, and both explicitly kept eviction as a local
side effect — "Still evict locally for the space". That side effect is itself
a divergence source one level below the decisions they pinned. The decisions
were uniform; the cache CONTENT they decided against was not.

## 4. The fix

Publish the group MIN of `available_size()` on the tree cache once per
iteration (`Scheduler._publish_uniform_evict_floor`) and let both mutation
triggers decide from it.

* **No new collective.** The value rides the reduce
  `_update_uniform_pool_budget` already performs, unconditionally and
  pre-branch, once per scheduler iteration. One extra element (`-local_avail`)
  gives the group MAX from the same MIN all_reduce.
* **Direction is safe by construction.** `min <= local`, so
  `floor < num_tokens` is true whenever the local test was, and sometimes when
  it was not. Every rank evicts at least as often as before; under-eviction
  (which would surface as an allocator OOM) is arithmetically impossible. The
  price is that slack ranks drop cache they did not personally need — that is
  what keeping the replicas identical costs.
* **Activation is collective-derived**, not config-derived: the floor is
  published only when the group's min and max availability differ, i.e. when
  the pools are actually uneven. On even TP the two are equal, the floor stays
  None, every trigger reads its live local value, and the path is
  byte-identical.

  That predicate is not a heuristic, it is exact. The floor is withheld
  precisely when every rank reports the SAME availability — and then the local
  test `avail < num_tokens` compares the same left side against a replicated
  right side on every rank, so it is already uniform and there is nothing for
  the floor to fix. Floor published <=> the local test could have diverged.

Engagement on the validated boot, checked rather than assumed: the serving
process runs with `PYTHONPATH=/spinning/wt-616c-hunter5/python` (the fixed
tree), its three pools are 179825 / 143860 / 136667 tokens — the wedging
geometry — and `_update_uniform_pool_budget` demonstrably runs every
iteration, because `uniform_min_avail()` RAISES on a multi-rank boot that
reaches a decode-mem decision without it (#603) and the server serves.

Touched: `managers/scheduler.py`, `mem_cache/common.py`,
`mem_cache/unified_radix_cache.py`, `mem_cache/hiradix_cache.py`,
`mem_cache/base_prefix_cache.py`.

### The load-back pin had to be written twice, and why that is not redundancy

The first cut pinned `HiRadixCache.load_back` only — and this rig does not use
that class. Checked rather than assumed, from the boot log: the deployed tree
cache is `UnifiedRadixCache` (with an `MHATokenToKVPool` allocator, no SWA),
which carries its OWN `load_back` with its own
`available_size() < kv_tokens` gate. The eviction pin in
`mem_cache/common.py` is class-agnostic and did bind, but half the fix was
sitting in a class the validated boot never instantiates.

Both classes now consult the floor, and
`test_both_load_back_sites_consult_the_floor` pins both by name — a deletion
falsifier, since neither site can be driven hermetically without standing up a
cache controller, host pool, lock refs and transfer plans. That test earned its
place immediately: it caught the pin being written into
`HiRadixCache.init_load_back`'s neighbour rather than `load_back` itself.

### What was deliberately NOT changed

The HOST pool is rank-sized too (359652 / 287722 / 273336 tokens), so
`write_backup`'s host admission is also a per-rank verdict. The argument for
pinning it is that backup state gates device eviction — but under this boot's
`write_through` policy the eviction path gates on `write_policy ==
"write_back"`, not on backup state, so the chain to the device tree is NOT
established. A pin was drafted and dropped: the defect class is suggestive,
the mechanism is not demonstrated, and pinning it would trade real cache
capacity for a hypothesis.

## 5. On-card validation

### 5.0 The first A/B proved nothing, and measuring it is what showed that

Two 45-minute arms under the four-way `verify-stress.sh` harness -- fixed
(A2) and unfixed (B) -- both came back completely clean: 0 Stalled, 0
Aborted, no restarts, ~1400 prefill batches each. Read naively, arm A2 "met
the acceptance bar". It did not, and arm B is the proof: an unfixed tree that
survives the same window says the window does not test the defect.

The reason is measurable in the logs of both arms:

    full token usage   A2: n=1564 median 0.20 peak 0.66
                        B: n=1559 median 0.21 peak 0.58

This defect lives at the pool boundary. `evict_from_tree_cache` only fires
when availability drops under the demand, and load-back only fails when the
device has no room. At a median occupancy of 0.20 neither trigger runs at
all, so neither arm exercised the mechanism under test. `verify-stress.sh`
reuses its prompts, so most of its traffic is served from the prefix cache
and never fills the pool -- exactly the wrong workload for this defect.

### 5.1 A reproducer that actually engages the mechanism

`/spinning/616f-harness/616g/pressure.py`: six workers, UNIQUE 20-45K-token
contexts (nothing hits the prefix cache, so every request must allocate),
prefill-dominated. It drives the pool to the eviction boundary in about three
minutes and holds it there.

Against the UNFIXED tree (3752da399d), that turned a ~17-minute-mean random
wedge into an 80-second deterministic one, with the specimen's signature:

    00:17:14  rank 1 last_op=all_reduce last_nbytes=20971520  (2048 tokens)
              rank 0 last_op=all_reduce last_nbytes=10485760  (1024 tokens)
    00:18:36  Bar1CollectiveStalled, rank 0, abort word CLEAN

Rank 0 -- the largest pool -- again carries the SMALLER count, which is the
direction the root cause predicts and the third independent specimen to show
it. The `dcp:0` all_gather diverged in the same window (rank 0 4194304 against
peers' 4046848), so the divergence is not specific to one collective family.
Archived: `/spinning/616f-harness/616g/UNFIXED_pressure_wedge.log`.

### 5.1b The reproducer needed REUSE as well as pressure

The pure-pressure recipe was not repeatable: a second run on the unfixed tree
went 8 minutes to peak 0.76 with nothing at all. The reason is in the recipe
itself -- UNIQUE contexts fill the pool but never populate the host tier, so
`load_back` is never even attempted, and load-back is the trigger the A-to-A2
step implicated as dominant. Pressure alone therefore exercises only half the
defect.

The recipe that fires reliably is BOTH at once: four `verify-stress.sh` loops
(prompt REUSE -> populated radix + host tiers -> load-back actually runs)
plus the pressure generator at concurrency 3 (unique 15-35K contexts -> pool
driven to the eviction boundary). That is also, in hindsight, exactly the
state the production server was in during all of the evening's wedges.

### 5.2 The matched-condition comparison

Same box, same hour, same four stress loops
(`verify-stress.sh`, `URL=http://127.0.0.1:30030 MODEL=Qwen3.6-27B`), same
flagset. The only variable is the tree.

Every run of the evening, in order, with the load that was actually applied:

| # | tree | load | duration | peak usage | divergence |
|---|------|------|----------|-----------|------------|
| 1 | unfixed (production) | agent traffic + 4x stress | evening | (warm, saturated) | 3 wedges, incl. the 21:52:25 specimen |
| 2 | partial fix (load-back pin in the unused class) | 4x stress | 6m15s | - | 1 Aborted |
| 3 | **complete fix** | 4x stress | 46m39s | 0.66 | none |
| 4 | unfixed | 4x stress | 45m | 0.58 | none |
| 5 | unfixed | pressure only | ~1m | 0.63 | **Stalled**, rank 0 1024 vs peers 2048 |
| 6 | unfixed | pressure only, cold boot | 8m | 0.76 | none |
| 7 | **complete fix** | pressure only | 25m | **0.91** | none |
| 8 | unfixed | **reuse + pressure** | **91s** | 0.76 | **Stalled**, rank 0 265 vs peers 2048 |
| 9 | **complete fix** | **reuse + pressure** | **25m** | 0.55 | **none** |

Runs 3 and 4 are the pair that must not be read as a result: both clean, but
both at a median occupancy of 0.20, so neither engaged the defect (5.0).
Runs 8 and 9 are the comparison that counts -- the same load, with a
demonstrated positive control that fails in 91 seconds, survived for 25
minutes (16x) by the fixed tree, with 459940 cached tokens and 1079 prefill
batches proving the reuse and load-back paths were live throughout.

One asymmetry in run 9 is worth naming rather than hiding: peak occupancy was
0.55 against the unfixed tree's 0.76, so the fixed tree did not reach quite
the same fill level. That is a PREDICTED consequence of the fix, not a gap in
the test. `full token usage` is logged by rank 0, the LARGEST pool, and the
floor makes rank 0 evict at the BINDING rank's threshold instead of its own --
so the roomiest rank is expected to sit lower once the pin is in. The same
mechanism that removes the divergence necessarily lowers that number.

The A-to-A2 step (rows 2 and 3) is worth recording on its own: arm A carried
the eviction pin and still died in six minutes, because its load-back pin had
been written into the class this deployment does not use. Adding the pin to
the class it actually runs is what turned six minutes into a clean 45 -- the
cleanest available evidence that load-back, not eviction alone, is the
dominant divergence source on this rig.

Raw evidence: `UNFIXED_combo_wedge.log`, `FIXED_combo_clean.log`,
`UNFIXED_pressure_wedge.log`, `FIXED_pressure_clean.log`,
`A2_FULL_WINDOW.log`, `events_A*.log`, `events_B.log`, all under
`/spinning/616f-harness/616g/`. Reproducer: `pressure.py` in the same
directory, run alongside four `verify-stress.sh` loops.

The A-to-A2 step is worth recording on its own: arm A carried the eviction
pin and still died in six minutes, because its load-back pin had been written
into the class this deployment does not use. Adding the pin to the class it
actually runs is what turned six minutes into a clean 45. That is also the
cleanest available evidence that load-back -- not eviction alone -- is the
dominant divergence source on this rig.


## 6. Falsifier

`test/registered/unit/distributed/test_uniform_evict_floor_616g.py`, hermetic
(no CUDA, no process group). The load-bearing case builds the three real pool
sizes and a replicated demand of 150000 tokens, which sits between them:

    local predicate:   [False, True, True]     <- the ranks split
    with the floor:    [True,  True, True]

Proven able to fail: reverting `evict_from_tree_cache` to
`allocator.available_size()` turns the second list back into the first and
fails the test (checked by doing exactly that, then restoring).

The suite also pins the no-op direction (even pools publish no floor, single
rank publishes no floor), that the #610 quantities and the single-collective
count are unchanged, and that the reduce payload width is rank-uniform — a
width that varied per rank would hang the reduce it rides on.
