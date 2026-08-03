## Axis B — comment invariants as testable claims

Desk audit plus two executed hermetic falsifiers (`CUDA_VISIBLE_DEVICES=99`,
`PYTHONPATH=/spinning/wt-505-silent/python`, interpreter
`/spinning/htsglang-gpu/.venv/bin/python`). No GPU held, no server booted, no
`.py` file changed. Base commit `d653405223` on `docs/silent-wrongness-505`.

The axis is CLAUDE.md's third SUCCESS-CLAIMS law, verbatim:

> (3) a code comment asserting an invariant ("leaves no partial state",
> "never reached") is a TESTABLE CLAIM — pin it with a test or treat it as
> a bug report against the comment.

The occasion is #501: a comment asserting *"a decline leaves no partial
state"* while the code above it had already committed an irreversible
reclaim. Fix `1f2b24ef11` exists on a sibling branch and is **not** an
ancestor of this audit base (`git merge-base --is-ancestor 1f2b24ef11 HEAD`
→ false), so #501 is still live in the tree this sweep read. That makes it
the calibration row, not a new finding — see §"Calibration".

Method template: `docs/dev/AUDIT_500_mechanism_reach.md` §§1-6 — verbatim
quoting, `file:line` on every row, four honest verdict classes, coverage
gaps stated rather than papered over. Catalog sections read:
`FEATURE_CATALOG.md` §1, §3, §4, §6, §7, §12, §17, plus §0's registry rule.
`CLAUDE.md` read in full.

### Coverage + triage rule

**Grep total (this base, the briefing's extraction verbatim):**

```
cd python/sglang/srt && grep -rniE "(#|\"\"\"|').*\b(never|always|must |guaranteed|\
leaves no|cannot happen|can not happen|safe because|invariant|by construction|\
impossible|is not possible|no partial|atomic|exactly once|only ever)\b" \
  --include=*.py mem_cache managers model_executor speculative distributed \
  layers/dcp layers/moe memtier model_loader
```

**1345 hits**, not the ~1111 the briefing estimated (the briefing's per-dir
figures are all ~12 % lower; the base differs). Per directory, measured:

| dir | hits | briefing said |
|---|---|---|
| model_executor | 310 | 259 |
| managers | 286 | 251 |
| speculative | 192 | 174 |
| distributed | 175 | 141 |
| mem_cache | 171 | 135 |
| layers/moe | 135 | 97 |
| model_loader | 41 | 33 |
| memtier | 21 | 17 |
| layers/dcp | 14 | 4 |

**Triage rule, applied mechanically, then by hand.** A claim is KEPT only if
it asserts something about **STATE or ORDERING that other code relies on**.
Discarded without individual inspection: English-usage "must"/"always" in
docstrings ("the caller must pass a tensor"), restatements of the line below
them, prose in module headers with no referent, and every `must` that is an
argument-validation message rather than an invariant.

The 1345 were not read one by one. They were reduced by intersecting the
invariant vocabulary with the four priority vocabularies the briefing ranks,
each grep run over the same nine directories:

| priority | pattern axis | candidate lines |
|---|---|---|
| 1 | rollback / revert / restore / cleanup / on-failure / decline / abort / partial | 27 |
| 2 | all ranks / every rank / only rank 0 / lockstep / rank-uniform | 31 |
| 3 | alias / caller owns / safe to reuse / scratch / in-place / view of | 7 |
| 4 | lock held / reentrant / runs exactly once / idempotent / single-threaded | 33 |
| 4b | never None / always set / never empty / is never / always returns | 43 |
| 4c | "must run/be called/happen/precede … before\|after\|first\|last" | 13 |
| | **candidate union** | **154** |

Plus one AST pass (below) over `managers`, `mem_cache`, `model_executor`,
`layers/moe`, `speculative`, `memtier`, `distributed` that pairs an
invariant comment inside a function with an irreversible-looking call
followed by a decline (`return None/False`) or a `raise` — the mechanical
form of the #501 shape. Narrow vocabulary: 3 hits. Broad vocabulary: 43
hits, mostly heuristic noise.

- grep total: **1345**
- candidate union after the priority intersections: **154**
- sites **opened and read in full context**: **41**
- claims **KEPT** (load-bearing, verdict assigned): **22**
- **triaged out: 1323** (1345 − 22), of which 1191 never entered a candidate
  list and 132 were candidates that on reading restated the adjacent line or
  carried no state/ordering assertion.

**Not reached** (stated, not implied): `model_loader` (41 hits — only
`weight_utils.py:571` opened), `memtier` (21 hits — the registry's
provenance claims scanned but none opened), `layers/dcp` (14 hits, none
opened), `mem_cache/storage/**` (flexkv / lmcache / mooncake / umbp
sub-backends), and all of `managers/scheduler.py`'s 100+ hits beyond the
four opened. `speculative/` was sampled through the priority-2 list only.
A shallow all-clear over those is exactly what this task exists to prevent,
so they are named as unswept rather than as clean.

### Calibration — the instrument re-finds #501 without being told where

The AST pass, run with no file list and no knowledge of #501:

```
python/sglang/srt/managers/kv_session_offload.py try_spill invcomment@3397
   first muts: [(3358,'free'), (3437,'pop'), ...]  declines after: [3382, 3394, 3411, 3434]
```

Exactly one hit in the narrow vocabulary, and it is #501: the allocator
`free` at `:3358` precedes four `return False` declines, under a comment at
`:3397-3399` that claims the opposite. The sweep discriminates. Fix
`1f2b24ef11` lands this on another branch; nothing here re-opens it.

### Table CONTRADICTED (priority bug candidates)

| file:line | comment (verbatim) | contradicting code (file:line + verbatim) | failure mode |
|---|---|---|---|
| `python/sglang/srt/model_executor/offload_movement.py:916-918` | `# Error path: retrieval failed physically. The item stays`<br>`# PARKED (its bytes are still at the park target); the`<br>`# failure is reported, never swallowed.` | The `try` whose failure this handles does **not** end at the retrieval. `offload_movement.py:910-914`:<br>`elif mv.handle is not None:`<br>`    self._ops.wait(mv.handle)  # park copy must have landed`<br>`    back = self._ops.copy_in_tensors(mv.handle)`<br>`    self._ops.wait(back)`<br>`    self._ops.free_destination(mv.handle)`<br>and the handler at `:919` unconditionally writes `mv.state = STATE_PARKED`, while the booking release `self.ledger.release(mv.target, mv.peer_device, mv.booked_bytes)` sits **outside** the try at `:924`. `free_destination`'s own contract (`:333-335`) is *"Release the park destination (pinned rows / peer allocation) after a completed wave-in."* | If the exception originates at `:913` or `:914` — i.e. after `copy_in_tensors` has already returned — the bytes are **not** still at the park target: they are back on the device and the destination release is half-done. The item is nevertheless marked PARKED and its park booking is never released, so (a) the memtier ledger over-books the park target forever while the item is resident, and (b) a later `wave_in` (state PARKED) re-runs `wait(handle)` → `copy_in_tensors(handle)` on a handle whose destination `free_destination` was already invoked on. **EXECUTED**, hermetic, both injection points, see below. |

**Executed falsifier for the row above** (no file written; the registered
test module's own harness driven inline):

```
fail_at=copy_in:          MovementError state=parked booked=1000
                          ops=['copy_out','wait','wait','copy_in']
fail_at=free_destination: MovementError state=parked booked=1000
                          ops=['copy_out','wait','wait','copy_in','wait','free_destination']
```

The second line is the contradiction, in the code's own op trace: `copy_in`
**and its `wait` completed**, `free_destination` was entered, and the item is
still reported `parked` with 1000 bytes still booked at the park target.

The existing test injects only at the first op —
`test/registered/unit/model_executor/test_offload_movement.py:280-289`,
`test_failed_wave_in_stays_parked_and_keeps_booking`, with
`ops = FakeDeviceOps(fail_ops={"copy_in": RuntimeError("boom")})` — which is
the one injection point for which the comment is true. Nothing injects at
`wait(back)` or `free_destination`, and `FakeDeviceOps._op`
(`offload_movement.py:489-493`) already supports both names, so the falsifier
is one dict key away.

Honest scope: the contradiction is conditional on **which** statement inside
the try raises. The comment states its claim unconditionally, and the two
statements for which it is false are the last two in the block.

### Table UNPINNED (testable-claim backlog, ranked)

Ranked by what breaks if the claim is false.

| file:line | claim (restated as falsifiable) | what breaks if false | proposed falsifier test |
|---|---|---|---|
| `layers/moe/expert_offload.py:3529-3531` — `# Host snapshot of the current resident experts [0,R) so overwriting`<br>`# buf[0:R] in place can never corrupt a not-yet-moved source.` | For every hot-set permutation, every `_src(e)` read in `_apply_hotset_freeze` resolves to host memory (`resident_host` snapshot or `old_spill`), never to a `buf` row already overwritten by an earlier iteration of the `for i, e in enumerate(hot)` loop. | Silently wrong expert weights after a heat freeze — the #302a/#443 class: fluent output, no crash, no hang. The whole zero-extra-VRAM rearrange rests on this one line. | The method carries `def _apply_hotset_freeze(self, hot):  # pragma: no cover - requires CUDA` (`:3498`) and **NO TEST** references `_apply_hotset_freeze` / `_freeze_hotset` / `freeze_from_source` anywhere under `test/`. A CPU-tensor stand-in with a permutation that maps a hot expert onto a slot whose old occupant is itself hot (e.g. `hot=[1,0]` at `R=2`) falsifies the negation directly: drop the `.to("cpu")` snapshot and the second `copy_` reads a clobbered row. |
| `model_executor/runner/decode_cuda_graph_runner.py:436-444` — `# ORDERING IS LOAD-BEARING: this must come AFTER every mutation of`<br>`# self.capture_bs (the MoE-offload cap and the weightless-KV cap above).` … `# Registering last makes`<br>`# the published set exactly the captured set` | (a) No write to `self.capture_bs` follows `_register_gguf_decode_buckets(self.capture_bs, …)` at `:445`. (b) The stated rule — *published set == captured set* — holds for every publication of `capture_bs`, not only this one. | (a) holds today (writes at `:371, :387, :422` only; `:506, :648, :1275, :1396, :1975` are reads). **(b) does not.** `KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)` at `:392` publishes **before** the weightless-KV cap at `:422` shrinks the list, so under weightless-KV × ktransformers the KT wrapper holds a strict superset. A captured graph replaying a kernel other than the one it was captured with is the exact hazard the comment names. | Two tests. (1) An AST/source ratchet asserting no `self.capture_bs` assignment appears after the `_register_gguf_decode_buckets` call in `__init__` — the same shape as `test_barlink_port.py:264-271`. (2) A constructor-level test with `SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS` and `SGLANG_WL_GRAPH_MAX_BS` both binding, asserting the value handed to `set_capture_batch_sizes` equals the final `self.capture_bs`. `test_gguf_mmq_decode_threshold.py:201` pins union-merging of the registry but nothing pins either ordering. |
| `mem_cache/hiradix_cache.py:957-960` (identically `hi_mamba_radix_cache.py:440`, `:471`; `unified_radix_cache.py:2652`, `:2682`) — `# Every rank must enter the all_reduce below; ongoing_write_through can`<br>`# diverge across ranks (e.g. write_backup returning 0 on a subset under`<br>`# host memory pressure), so a conditional skip desyncs the NCCL op`<br>`# sequence and deadlocks under TP > 1.` | No path through `writing_check` / `loading_check` reaches a `return` before `self._all_reduce(...)` on a rank-divergent condition. | A TP-wide hang — the four-incident rank-local-condition-before-a-group-collective family (#94/#194/#312/#431). | The unconditional body is correct, but `writing_check(write_back=True)` **does** return at `:945` before the collective (`# blocking till all write back complete … return`). Nothing pins that `write_back` is itself rank-uniform at every call site. Falsifier: a two-rank harness where rank 1 enters with `write_back=True` and rank 0 with `False`, asserting the decision recorder (`barlink_uniformity.first_divergence`, already the standing instrument) reports a split. Existing tests only call `writing_check(write_back=True)` single-rank (`test_hiradix_cache_unit.py:112`, `test_unified_radix_cache_unittest.py:495`), which cannot fail on the negation. |
| `managers/vram_dial.py:497-498` — `# Compute all new budgets first, validate all, then commit all --`<br>`# a multi-rank dial must not half-apply.` | For `device="all"` with one rank below its floor, **no** rank's `budget_bytes` changes and `_op_seq` does not advance. | A partially applied group dial desynchronises the replicated budget vector; the consensus round at the next boundary then reduces over ranks that disagree — the failure the `_consensus_check` MIN-reduce exists to make loud, arrived at from inside instead of outside. | `test_vram_dial.py:365-377` (`test_below_floor_rejected_with_exact_numbers`) pins the claim only for `device="rank:0"`, and `_dial_setup()` builds a **one-rank** runtime (verified by execution: `n ranks: 1`). Falsifier: a ≥2-rank runtime where rank 0's request is valid and rank 1's is below `min_viable_budget_bytes(1)`, asserting `rt._ranks[0].budget_bytes` is unchanged. `test_group_grow_commits_on_every_rank:555` covers the commit path, not the rejection path. |
| `managers/tp_worker.py:587-596` — `# ORDERING (#143): the is_verify early return must come FIRST. On a`<br>`# verify step the head skips sampling entirely … so`<br>`# it never reaches the matching send below -- a weightless worker`<br>`# that took the recv branch here would block in gloo forever.` | `if is_verify: return batch_result` (`:597-599`) precedes the weightless `broadcast_pyobj` recv branch (`:601-613`) on every reachable path. | A permanent gloo block on the weightless workers — the reported #143 symptom, and structurally the same family as the hiradix row above. | Holds at this base (`:597` is above `:601`). No test asserts the ORDER; `test_weightless_chain_spec.py` and `test_draft_solo_placement.py` exercise the paths but not the precedence. Falsifier: a source/AST ratchet asserting the `is_verify` return statement's line number is lower than the `is_weightless_worker` branch's inside `forward_batch_generation` — cheap, and it is the only thing that survives a future edit that reorders the branches. |
| `managers/kv_reshard.py:449` — `# Bounds check BEFORE any write: the fitted ceiling must hold.` and `:481-484` — `# EXCHANGE (pool still untouched): a failure up to and including this`<br>`# point aborts the attempt with the pool byte-identical -- a later`<br>`# boundary may retry. Only the WRITE phase below is the`<br>`# no-return-on-error region.` | Every `raise KvReshardError` at `:452`, `:494`, `:504` fires with the KV pool byte-identical to its pre-call state. | A failed reshard that has already scribbled rows leaves the pool in a state no retry can recover — the #501 shape at pool scale. | The claim holds by reading (`max_new_row` bounds at `:450-457`; PACK is `read_rows` only; EXCHANGE writes nothing to the pool). `test/registered/scheduler/test_kv_reshard.py` exists but nothing asserts pool byte-identity across a raising `_execute`. Falsifier: checksum the pool before, force `_exchange` to return a short payload (the `:494` arm) or a bad checksum (`:504`), assert the checksum is unchanged. |
| `distributed/parallel_state.py:1077-1080` — `# Time this collective for the per-rank compute/wait split, then`<br>`# re-enter with the clock disarmed: the body runs exactly once,`<br>`# and a collective built out of other collectives is counted once`<br>`# rather than once per level.` | The `_COLLECTIVE_CLOCK.span()` re-entry executes `all_reduce`'s body exactly once and cannot recurse a second time. | Double-counted ms/round — and the ms/round figure is this project's declared measuring stick, so a silently doubled wait term corrupts every phase verdict built on it. | No test found for `_COLLECTIVE_CLOCK` re-entry. Falsifier: arm the clock, call `all_reduce` on a 1-GPU group, assert exactly one span was opened and the body's `world_size == 1` short-circuit was hit once (a counter on the fake group). |
| `distributed/device_communicators/barlink_ucx.py:1183-1186` — `` # `_get_out_buf` always returns an ``<br>`` # `empty_like` of an already-contiguous input, so this never fires; ``<br>`# if it ever does, it fails loudly instead of returning garbage.` | `out` reaching `out.view(-1)` at `:1187` is always contiguous. | `view` on a non-contiguous tensor raises — loud, per the comment's own escape clause — so the risk here is a *false* sense of coverage, not silent wrongness. Ranked low deliberately. | True at this base: `barlink.py:834` does `inp = input_.contiguous()` before `t.barlink_all_reduce(self, inp)` (`:838`), and `_get_out_buf` is `return torch.empty_like(ref)` (`barlink.py:775`). The *freshness* half is pinned (below); the *contiguity* half is not. Falsifier: call the ucx all-reduce path with a non-contiguous input through the public seam and assert it either succeeds or raises by name, never returns an untouched tensor. |
| `layers/moe/expert_offload.py:1462-1464` — `if R != len(plan.resident_ids):  # defensive: plan invariant` | The "defensive" check can prevent the damage it names. | It cannot: the loop at `:1459-1462` has already run `out[slot].copy_(source(expert_id))` **and** `release(expert_id)` for every resident id before the check executes. A guard placed after the irreversible work is decoration — the generic form of #501, in a file that stages weights. | Falsifier: construct an `ExpertStagingPlan` with `resident_count != len(resident_ids)` and a `release` callback that records calls; assert `release` was never invoked when the `RuntimeError` fires. Fails today. (Whether such a plan is constructible is exactly what the check claims not to know — that is the argument for hoisting it above the loop, not for deleting it.) |
| `model_executor/model_runner_kv_cache_mixin.py:2007-2012` — `# #364 resident-slot cap. Applied AFTER every profiling branch and`<br>`# after the uneven-TP min-sync, so it is the last word on the pool`<br>`# geometry and cannot be undone by a branch that recomputes a size.`<br>`# Rank-uniform without a collective: it is a server arg, identical on`<br>`# every rank by construction, and it only ever lowers` | (a) nothing after `:2050` recomputes `max_mamba_cache_size`; (b) `effective_state_slots` only ever lowers. | A cap that is undone downstream re-inflates the state pool after the KV budget was already sized against the capped figure — an OOM at capture, or KV tokens silently handed back. | (b) is pinned by construction and documented at `gdn_slot_ladder.py:86-91` (`cap_is_binding` is `resident_cap < profiled_slots`). (a) is not pinned by anything; the only later touch is the `<= 0` validation at `:2063`. Falsifier: a source ratchet asserting no assignment to `server_args.max_mamba_cache_size` follows the `#364` block inside `handle_max_mamba_cache`. |

### Table PINNED (evidence that the sweep discriminates)

Ten of the twenty-two kept claims are genuinely pinned. Per the SUCCESS
CLAIMS law an instrument counts only after it shows it can discriminate;
these rows are that demonstration, and they are why the UNPINNED table above
is a backlog rather than a verdict on the project's testing culture.

| file:line | claim | pinning test file:line |
|---|---|---|
| `speculative/eagle_utils.py:117-119` — `# So the choice is made ONCE, collectively: every rank contributes its own`<br>`# capability, the MINIMUM rules, and if one rank lacks the native kernels then`<br>`# ALL ranks take Triton. Never a mixture.` | one rank without native spec kernels moves the whole group to Triton | `test/registered/unit/spec/test_spec_kernel_backend.py:106` `test_one_rank_without_kernels_moves_EVERY_rank_to_triton`; `:121` `test_decision_is_uniform_across_ranks`; `:138` `test_refusal_leaves_no_half_decision_behind` |
| `speculative/eagle_utils.py:135-141` (docstring) — *"A second, DIFFERENT value is refused rather than silently winning"* | `set_spec_kernel_backend` refuses a conflicting second decision | `test_spec_kernel_backend.py:74` `test_conflicting_decision_is_refused` |
| `distributed/parallel_state.py:759-764` — `# RANK-UNIFORM by construction: the condition reads only`<br>`# envs.SGLANG_BARLINK and self.world_size` | the pynccl-suppression decision cannot split the group | `test/registered/unit/distributed/test_barlink_suppresses_pynccl.py:73` `test_decision_is_rank_uniform`; `:111` `test_matches_parallel_state_source` |
| `distributed/device_communicators/barlink.py:752-753` — `"""One FRESH output tensor per call — never a shape-keyed cache.` | `_get_out_buf` allocates per call; two same-shape results are never the same tensor | `test/registered/unit/distributed/test_barlink_port.py:264-271` — AST ratchet, *"_get_out_buf must allocate a fresh tensor per call"* |
| `distributed/device_communicators/barlink_bar1.py:3533-3537` — `# THE ONE CONDITION THAT IS STRICTER HERE THAN FOR ALL_GATHER: the`<br>`# round count must be the same on ALL ranks, even though only one`<br>`# sends.` | `bc_plan`'s round count is a function of `nbytes` and slot only | `test/registered/unit/distributed/test_barlink_bar1_broadcast.py:202` `test_round_count_is_rank_uniform`; `:352` `test_the_round_count_is_the_same_on_every_rank`; `:358` `test_the_kernel_variant_is_decided_group_uniformly` |
| `distributed/device_communicators/barlink_bar1.py:3541-3545` — *"the extension rejects `in is out`"* | broadcast never runs with input aliasing output | `test_barlink_bar1_broadcast.py:369` `test_input_and_output_are_never_the_same_buffer` |
| `model_executor/kv_pressure_ladder.py:1491` — `# Invariant 2: never jump over a relief rung into a geometry flip.` | ascent exhausts relief before geometry | `test/registered/unit/model_executor/test_kv_pressure_ladder.py:536` `test_full_climb_exhausts_relief_before_geometry`; `:552` `test_forced_target_skipping_relief_is_a_hard_error` |
| `model_executor/kv_pressure_ladder.py:1438` — `# Invariant 3: only at round boundaries, never inside a capture.` | no flip inside a capture | `test_kv_pressure_ladder.py:579` `test_capture_guard_blocks_the_flip`; `:590` `test_flip_only_at_round_boundaries` |
| `model_executor/kv_pressure_ladder.py:1505` / `:1553` / `:1564` — `# Invariant 4: no flip (and no pre-stage) onto uncaptured graphs.` / `# Invariant 6: external rungs need the long hysteresis.` / `# Invariant 5: protected sessions stay on the fast rung.` | each stated as a hard error / block | `:560` `test_flip_onto_uncaptured_graphs_is_a_hard_error`, `:570` `test_pre_stage_onto_uncaptured_graphs_is_the_same_hard_error`; `:616` `test_external_rung_needs_the_long_hysteresis`; `:598` `test_protected_sessions_stay_on_the_fast_rung`, `:607` `test_all_protected_blocks_the_ascent` |
| `model_executor/short_term_offload_register.py:190-192` — `#: Ranks :func:\`plan_spill\` will never plan. §8.5 is not a soft preference:`<br>`#: … FCFS, never by a memory-pressure planner reaching past four other rungs.` | active work is never a spill victim | `test/registered/unit/model_executor/test_short_term_offload_register.py:350` `test_active_work_is_never_planned`; `:245` `test_the_ladder_is_ordered_as_the_doctrine_writes_it`; `:550` `test_the_origin_card_is_never_its_own_park_target` |

### Top findings

**#505-B-01 — `wave_in`'s error path claims the bytes are still parked; for
the last two ops in its own try block they are not.**
`offload_movement.py:916-918` vs `:910-914` and `:924`. Executed both
injection points hermetically: with `fail_ops={"free_destination": …}` the
op trace is `['copy_out','wait','wait','copy_in','wait','free_destination']`
— the copy-in **and** its wait completed — and the item is still reported
`parked` with 1000 bytes still booked. Ledger over-booking is permanent
(the release at `:924` is outside the try), and a retry re-copies from a
destination that was already told to free. The registered test injects only
at `copy_in`, the one point where the comment is true.
*Task title:* `#505-B-01 offload_movement.wave_in: the error path must distinguish "never retrieved" from "retrieved, release failed"`

**#505-B-02 — the "published set == captured set" rule the decode graph
runner states for itself is broken one publication earlier.**
`decode_cuda_graph_runner.py:436-444` argues, correctly and at length, that
publishing `capture_bs` before a later shrink is only accidentally safe —
and then `KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)` at `:392`
does exactly that, ahead of the weightless-KV cap at `:422`. A rule stated
in a comment and honoured at one of two sites is the registry-disagreement
shape of #500, moved from the catalog into the source file.
*Task title:* `#505-B-02 publish capture_bs to every consumer after the last cap, not just to the GGUF bucket registry`

**#505-B-03 — the hot-set freeze's aliasing argument is carried entirely by
a comment, on a method marked `# pragma: no cover - requires CUDA`.**
`expert_offload.py:3529-3531`. The whole zero-extra-VRAM in-place rearrange
rests on `resident_host = buf[:R].to("cpu")` making every `_src(e)` a host
read. No test anywhere in `test/` names `_apply_hotset_freeze`,
`_freeze_hotset` or `freeze_from_source`. This is the highest-consequence
UNPINNED claim in the sweep: its negation is wrong expert weights with
fluent output, which is precisely the failure class #452's B2 arm could not
localise.
*Task title:* `#505-B-03 pin the hot-set freeze aliasing invariant with a CPU-tensor permutation test`

**#505-B-04 — a "defensive: plan invariant" check that runs after the work
it guards.** `expert_offload.py:1462-1464`: the residency loop has already
copied and `release`d every resident expert before `R != len(resident_ids)`
is tested. The generic form of #501, and cheap to fix by hoisting.
*Task title:* `#505-B-04 hoist the staging-plan consistency check above the loop it is meant to guard`

**#505-B-05 — the radix caches' "Every rank must enter the all_reduce" claim
has an exempt door nobody pins.** `hiradix_cache.py:957-960` (and three
siblings) argue the collective correctly, but `writing_check(write_back=True)`
returns at `:945` before it, and nothing establishes that `write_back` is
rank-uniform at every call site. Given this project's four-incident history
in exactly this family, an unpinned rank-uniformity precondition on a
collective's entry gate is worth a test even though the reading is clean.
*Task title:* `#505-B-05 pin write_back rank-uniformity on the writing_check collective gate`

**#505-B-06 — the multi-rank no-half-apply promise is tested only on a
one-rank runtime.** `vram_dial.py:497-498`; `_dial_setup()` builds `n=1`
(verified by execution), so `test_below_floor_rejected_with_exact_numbers`
cannot fail on the negation of the *multi-rank* claim.
*Task title:* `#505-B-06 vram dial: a group dial with one out-of-floor rank must leave every rank untouched`

**#505-B-07 — ordering claims that are true today and unratcheted.**
`tp_worker.py:587-596` (the `is_verify` return must precede the weightless
recv branch), `model_runner_kv_cache_mixin.py:2007-2012` (the #364 cap must
be the last word), `decode_cuda_graph_runner.py:436` (no `capture_bs` write
after publication). All three hold at this base and all three are one
reordering edit away from a silent hang / silent re-inflation, with nothing
to catch it. Source/AST ratchets in the shape of
`test_barlink_port.py:264-271` are the cheap instrument this tree already
owns.
*Task title:* `#505-B-07 source ratchets for the three load-bearing statement orderings`

**Finding about the sweep itself.** Ten of twenty-two kept claims are
pinned, and the pinning tests are unusually good ones — `test_barlink_port`'s
AST ratchet, `test_spec_kernel_backend`'s mocked MIN-collective,
`test_kv_pressure_ladder`'s six numbered invariants. The failure is not a
missing testing culture; it is that the culture stops at module boundaries.
Every UNPINNED row above sits in a module whose *neighbours* are pinned. The
#501 shape survives where an invariant spans two functions (guard here,
mutation there) or two consumers (publish here, cap there) — never inside a
single well-tested predicate.
