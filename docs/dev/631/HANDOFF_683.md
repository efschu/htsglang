# HANDOFF 683 — #656 / #631 Route A, successor 39 (queue item #657)

The shift that was asked to FUND the REBALANCE tier. It built the actuator
five handoffs have called missing, showed its decision correct on metal —
and then watched it take the instance down, for a reason its own instrument
named in the same second. The tier is still empty, and for the first time the
reason is structural rather than mysterious. Errors first; the falsifications
are the deliverable.

---

## 0. THE ONE-LINE STATE

**Item 16's REBALANCE actuator now exists and its DECISION is group-uniform —
and the mechanism ships OFF, hard, because it cannot move a committed byte
and it killed its own confirmation window at t+18.** What a card commits is
decided by the KV pool's backing WATERMARK, not by which rank owns a row; and
re-applying the steer on a rank-local clock desynchronised the three ranks'
free lists until a cutover refused to scatter. Both failures were caught by
checks written before the boot — a hermetic test for the first, the
mechanism's own replication checksum for the second.

**This shift also falsified the lever it was about to recommend.** The
"evictable" headroom under the watermark, which I priced mid-shift at ~592
MiB per card from s38's log, is not evictable at all: measured at one
instant by this shift's new instrument, a resident request holds the same top
row the cache does, and an id-targeted eviction would free **0 rows**.

---

## 1. ERRORS FIRST

### 1a. THE BRIEFED MECHANISM IS BUILDABLE, ITS EFFECT IS NIL WHERE IT WAS WANTED, AND NEGATIVE WHERE IT WAS NOT

The brief was to "weight NEW KV allocations toward the card with the most
free headroom via the uneven-DCP token-vector machinery". The lever exists,
and it is not the token vector — it is the free list:

* a token's rank is a pure function of its global slot id and the vector
  (`layers/dcp/owner.py:431-437`), and the vector is boot-constant, so
  ownership cannot be re-decided at runtime;
* but **every allocation path takes the HEAD of `free_pages`** — `alloc`
  (`allocator/paged.py:162`, `allocator/token.py:63`), and the two Triton
  kernels index it from 0 — so which id is handed out next IS a runtime
  choice among already-legal placements.

A stable residue-class partition of the free list therefore steers all
allocation paths at once. It places bytes, moves none, frees none, cannot
starve an allocation (the tail is the rest of the list), and it degrades to a
no-op when the class runs out. Built, armed, and correct on metal.

**And it changes nothing a card can feel.** Three facts, each with its own
site, compose into that:

1. **The scheduler keeps ONE allocator for process life — the PP stack's**
   (`phase_flip_boot.py:750-756`), and the PP layout has `dcp_size == 1`, so
   in that pool the row of slot `L` **is** `L`, identically on every rank
   (`phase_flip_plan.py:9-11`). Ownership does not enter.
2. **The TP pools are pre-sized at boot**, "no growth, no address change"
   (`phase_flip_runtime.py:24-26`). A rank owning more tokens does not commit
   more VRAM; it fills a pool that was already there.
3. **The only lever that returns bytes to the driver is the PP pool's
   backing watermark**, floored by the MAXIMUM LIVE SLOT ID
   (`kv_backing_relief.py:406-441`) and released only at the pp->tp gate
   (`phase_flip_runtime.py:3964`), re-committed at the tp->pp hook (`:1203`).

So the quantity a steer can move (which rank owns a row) and the quantity the
corridor law is written in (what each card commits) are not connected. Worse,
they are connected the wrong way: a class bias makes the allocator hand out
HIGHER ids, which RAISES the ceiling the watermark is floored by. Pinned
hermetically in `test_a_class_bias_raises_the_maximum_live_id`.

> Registered as **C22** and as law 11: *placement is not residency.* Item 16
> says "redistribute onto the card with the most headroom"; five shifts read
> that as an ownership problem. Before building relief, name the quantity the
> DRIVER sees and check the mechanism moves THAT one.

**And it is not merely inert — it is a cost, measured.** The hermetic test
said a class bias raises the maximum live slot id; the window says by how
much, against s38's window at matched load (both drive the same
`s33_occupancy_leg.py --sessions 3 --tokens 130000`):

| max_live at the KV rung's proposals | s38, no steer | s39, steer ON |
|---|---|---|
| p50 over all proposals | 132507 | **512543** |
| at usage >= 0.20 | 343949 | **512543** |
| share of the 512552-row id space | 67% | **99%** |

The rung's floor IS `max_live + 1 + margin + admission reserve`, so a steer
that scatters allocations across a residue class drives that floor to the top
of the pool and the rung -- the funder of the fatal pp->tp leg -- can return
almost nothing. **STEERING SHIPS OFF.** The prediction came from a test
written before the boot, which is the only reason the price is legible now
rather than after a wedge.

### 1b. #287 CANNOT BE WIRED AS THE REBALANCE PROVIDER, ON THREE INDEPENDENT AXES

The brief was "wire, don't rebuild". It cannot be wired, and the gap is not
adapter glue:

* **Shape.** `KvPressureLadder.on_pressure_boundary`
  (`model_executor/kv_pressure_ladder.py:1413`) returns a `LadderPlan` — a
  plan, in which nothing moves (`:1268`) — and `apply()` returns `None`. The
  provider contract wants driver BYTES, synchronously
  (`corridor_guard.py:389-405`). There is no byte count anywhere in the plan.
* **Cadence.** #287 commits only at a rank-uniform consensus boundary every
  8 rounds behind a gloo MIN reduction (`managers/kv_pressure_runtime.py:263`).
  The guard's providers run behind its **rank-local arm condition**, where the
  tree states plainly that a collective is "a deadlock waiting for the one
  rank that took a different branch" (`corridor_guard.py:357-361`). This is
  exactly why `KvBackingRelief` is built but deliberately NOT registered
  (`phase_flip_spill.py:1468-1477`) — a precedent directly on point.
* **Payload, the fatal one.** Of its five relief rungs
  (`kv_pressure_ladder.py:151-159`): `admission_cap` moves no KV by its own
  docstring; `dcp_ratio` is the token vector, i.e. the thing that does not
  apply in PP and needs an idle instance anywhere else; `kv_spill` and
  `weightless_rank` are planned-only; `session_offload` moves real bytes but
  its destination is host RAM, which is `RELIEF_HOST` by definition — routing
  it through REBALANCE would launder a host spill past item 16's fleet gate.
  Every `KvHandover` except `NoHandover` raises `NotImplementedError`
  (`:356-500`), so the "speculative shadow pre-staging" the brief names is a
  stub.

**Worth salvaging, and only this:** #287's `KvPressureSensor` (`:1068`) gives
projected exhaustion with hysteresis, where the lender triggers on an
instantaneous watermark difference (`corridor_rebalance.py:283`). That is a
real, small wiring job — but it changes WHEN the tier is spent, not WHAT it
has to spend, and the measured failure was always the payload.

### 1c. TWO BOOTS WERE SPENT ON IDENTITY BUGS, AND BOTH SHOWED UP AS INERTNESS

Recorded as law 12, because the shape repeats: an identity bug does not
produce a wrong answer, it produces a mechanism that quietly does nothing.

* **Boot 1: the bias sat on the wrong allocator class.** It was written on
  `PagedTokenToKVPoolAllocator`. This rig's scheduler holds the PP stack's
  allocator, and at `page_size == 1` with `dcp_size == 1` the chooser builds a
  plain `TokenToKVPoolAllocator` (`model_runner_kv_cache_mixin.py:4083`). The
  boot logged `the active allocator has no owner bias` nine times. Moved to
  `BaseTokenToKVPoolAllocator`, which both classes inherit.
* **Boot 2: `ps.tp_rank` is 0 on every rank in the PP layout.** All three
  ranks wrote their NVML column into slot 0 of the reduction, the permutation
  came back `(0, 1048576, 1048576)`, and the steer **disarmed itself rather
  than guess a column** (register law 9). Now indexed on
  `get_world_group().rank_in_group`, the identity the cutover itself uses.

Both are regression tests carrying the observed values. Note what saved both
boots: the mechanism reports what it RESOLVED and refuses to act on an
unresolved identity. A version that logged only its intent would have shipped
twice, doing nothing, and looked healthy.

### 1d. TWO THINGS THAT ARE NOT #657, CHECKED SO THE NEXT SHIFT DOES NOT RE-CHECK THEM

* **`test/registered/unit/mem_cache/` is red and segfaults, on the parent
  commit too.** 4 failures in `test_mamba_checkpoint_interval.py` and a
  segfault at 22% of the directory — reproduced identically on `70865d7069`
  in a detached worktree. Pre-existing, and outside the family suite.
* **A graceful serving stop takes about ten minutes to drain** on this
  configuration, with the soak and leg shepherd still feeding it. Stop the
  traffic generators by PID first, then the launcher, and budget the wait.

---

## 2. THE MECHANISM AS SHIPPED (OFF), AND THE ONE PART WORTH KEEPING

`managers/corridor_steering.py`, armed by `SGLANG_CORRIDOR_STEERING=1`, OFF
by default — with it unset, **no extra collective is entered at all**, no
free list is ever reordered, and the restored ship boot logs zero
`CORRIDOR-STEER` lines. Carried rather than reverted because its
instrumentation is what produced this shift's two findings; **do not arm it**
without fixing §3b first.

| | |
|---|---|
| what it does | promotes the free slots of ONE DCP owner class to the head of `free_pages`, so the next allocations land on that rank |
| where the choice lives | `BaseTokenToKVPoolAllocator.set_owner_bias` / `_apply_owner_bias` — the base, so both allocator classes inherit it |
| the decision | taken at the flip SEAM, the one point every rank reaches unconditionally with a bounded collective in hand (`phase_flip_runtime._corridor_gate`) |
| the input | the NVML free column -> the absorbing card -> its RANK |
| group uniformity | the rank index, the rank->NVML permutation, and a checksum of the free list all ride ONE MIN reduction. A steer decided per rank would order three free lists differently and split one token's KV across two rows |
| the fail-closed half | an unresolved permutation, a divergent checksum, a paged layout, or a malformed class DISARM the mechanism for the life of the process |

**The checksum is the part worth keeping regardless of the verdict.** The
whole flip design rests on "the free list is replicated scheduler state"
(asserted in `alloc_owner_matched_classes`' docstring, never verified on
metal). This window verified it, continuously, for the first time: every
promoted-slot count appeared in triples across the three ranks.

---

## 3. THE EVIDENCE

### 3a. THE WINDOW DID NOT SURVIVE, AND THE MECHANISM IS WHY

`/spinning/evidence-631/s39/window3`, ship config + `SGLANG_CORRIDOR_STEERING=1`,
booted 22:32:37Z on `e6649eaa26`. **At 22:50:30Z, t+18, all three ranks went
down inside a pp->tp cutover:**

    sglang.srt.layers.dcp.reshard_plan.KvReshardError: PHASE-FLIP payload
    checksum mismatch from peer 1: sender 2756967953890194568,
    receiver 10156871172 -- refusing to scatter

**In the same second, on all three ranks, the steer's OWN replication check
fired and disarmed it:**

    CORRIDOR-STEER DISARMED: the free list is NOT replicated across ranks
    (checksums 455859173976 vs 455936170468); steering cannot be
    group-uniform on a list the ranks disagree about

Two independent checksums, one instrument's and the flip's, naming the same
failure in the same second. **The free lists had diverged, and the steer
diverged them.**

### 3b. THE ROOT CAUSE: A GROUP-UNIFORM DECISION IS NOT A GROUP-UNIFORM MUTATION

The decision was uniform — that half was designed carefully and it held (0
disagreements in the reduction, the permutation agreed, the promoted counts
in triples). What is NOT uniform is WHEN the partition is applied.

`steer_on_round` re-applies the bias on a **rank-local monotonic clock**
(`corridor_steering.py`, `_reapply_s`, 1.0 s), because frees return pages to
the head of the list and wash the order out. Three ranks therefore re-sort
their free lists at three different instants, and between those instants the
three lists are in different ORDERS while holding the same members. The
partition is a pure function; applying it on a private clock is not.

The reshard plan is built from the allocator's state at the seam, so two
ranks that had last re-partitioned at different moments built different
payloads, and the flip's checksum refused to scatter. **That is a fail-safe,
not a corruption** — the design's own no-return-region guard did exactly its
job, and every request was lost with the instance rather than answered
wrongly.

> The fix, for the record, is not a smaller interval: it is to apply the
> partition ONLY inside a group-synchronised region (the seam, where the
> decision is already reduced) and never on the round clock. It is not worth
> doing — see §1a's price — but the next person to reach for a free-list
> mutation should know which half was the trap.

### 3c. WHAT THE 18 MINUTES DID SHOW

The placement axis, which is what the mechanism was to be judged on, works:

| axis, up to the crash | value |
|---|---|
| ranks armed / "not applicable" | 3 / 0 |
| rank -> NVML permutation (by UUID, group-agreed) | **[1, 0, 2]** |
| decisions by absorbing rank | rank 0: 18, rank 2: 15, **binding card: 0** |
| decisions naming the actually-fullest card | 24/30 (80%) |
| promoted-slot counts in TRIPLES | **11/11** |
| DISARMS before the fatal seam | 0 |

And the corridor, over the 7786 samples before the crash, shows the levelling
it was supposed to deliver **did not happen** — which §1a predicts, because
ownership does not decide residency:

| | s38 (30 min, no steer) | s39 (18 min, steer ON) |
|---|---|---|
| corridor breaches | 0 | **0** |
| free-headroom spread p50 | 2741 MiB | **2625 MiB** |
| spread mean | 2474 MiB | **2435 MiB** |
| per-card MIN | 1083 / 1580 / 1581 | 1121 / 1558 / 1415 |
| per-card p50 | 2307 / 5010 / 2717 | 2225 / 4848 / 2621 |
| survived its window | **yes** | **NO** |

A ~100 MiB move in the spread median, at a shorter window and a lower peak
occupancy (16.4% of pool against s38's 63.9%), against an instance that died
at t+18. There is no reading of this table in which the mechanism ships on.

**Serving was restored on the ship config immediately** (22:59:08Z, health
200, `SGLANG_CORRIDOR_STEERING` unset, 0 steer lines in the log — the default
path is untouched by this shift's code) and left running.

---

## 4. WHAT TO DO NEXT, IN ORDER

**0. DO NOT BUILD THE RADIX-EVICTION LEVER. This shift priced it, then
falsified it with its own instrument — read this before rediscovering it.**

Mid-shift I derived a lever that looked well-priced: the rung's floor is
`max_live_slot_id + 1 + margin + admission_reserve`, and on s38's shipped log
that ceiling sat **55k-210k rows above the live token count** (median ~71k
rows = ~592 MiB on the binding 3080), while the traffic returned
`#cached-token: 0` on **41952** batches. Read that way, the floor was pinned
by a prefix cache the workload never hits, and evicting its tail would let
the rung return hundreds of MiB per card.

**It is wrong, and the error is register law 7's shape: two instruments, two
clocks.** `full token usage` was joined to a proposal line emitted at the
seam, and the two are not sampled together. Measured at ONE instant by this
shift's own instrument, which reads both sources inside the same live-set
enumeration:

    ceiling pinned by the radix tree (tree_max=300001 over 131270 rows,
      req_max=300001 over 259509 rows); an id-targeted eviction could lower
      max_live=300001 by at most 0 rows

`tree_max == req_max` in every sample: **a RESIDENT REQUEST holds the same
top row the cache does**, so evicting the cache moves the ceiling by nothing.
Every proposal in the window reported `at most 0 rows`. The resident set is
also far larger than `full token usage` suggests (259509 rows against the
133264 the usage figure implies), which is the other half of why the
subtraction looked profitable.

What survives is the instrument, and it is worth keeping: it answers "would
an eviction help" continuously, in the rung's own line, and it answered NO
before anyone spent a shift building the actuator.

0b. **If anyone revives the steer, fix its reduction first.** The absorbing
   rank is agreed by a MIN over each rank's proposed rank INDEX, which is
   uniform (all that correctness needs) but biased toward LOW indices rather
   than toward the fullest card — measured at 24/30 (80%) of decisions naming
   the card that was actually fullest. Packing `(free_bytes, rank)` into the
   reduction would make the agreed answer the true maximum instead of the
   smallest index among candidates. Left as-is deliberately: the mechanism
   ships off, and a fix to a lever that costs more than it pays is polish on
   the wrong object.
1. **Instrument the re-application path.** The steer's DECISIONS are logged;
   its per-round re-partitions are silent, so the window can show what was
   decided but not how often the order was restored. `AllocationSteering.
   report()` exists and nothing calls it.
2. **Per-class free-slot occupancy at each decision.** Today the log carries
   the promoted count for the biased class only. Logging all three classes
   would make the placement EFFECT visible directly, not by inference.
3. C21: price the seam against what it HOLDS (HANDOFF_682 §1c) — unchanged.
4. The margin's own sizing is still measured on PRE-margin data (681 §4.2).
5. A one-boot A/B of the margin's price (681 §4.1).
6. An abandoned pp->tp leaves the KV pool capped (681 §4.3).
7. The abandon path does not drain the abort deferral window (681 §4.4).
8. `SGLANG_FORWARD_PEAK_PATH` on the next acceptance boot.
9. C18: give `vram_dial` the corridor guard's floor before the dial is on.
10. The corridor counters are still write-only (680 §4.5).
11. Still carried from 682 §4a: loosen the drafter's phase precondition; the
    host half / dynamic-chunking A/B; the draft-weights provider reports an
    ARENA COUNT, not an NVML delta.

---

## 5. PROCESS NOTES

* **Name the quantity the driver sees before building relief for it** (§1a).
  Ownership, occupancy and residency are three different things on this rig,
  and only the third one is what the corridor law is written in.
* **A mechanism should report what it RESOLVED, not what it intended**
  (§1c). Two identity bugs in two boots were caught by exactly that, and
  neither would have been visible in a result.
* **Check an inherited red on the parent commit before spending a shift on
  it** (§1d) — a detached worktree costs a minute and `git stash` is
  forbidden here for good reason.
