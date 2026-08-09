# #656 HANDOFF v19 — successor 16

Written 2026-08-09, tree `/spinning/wt-631-routea`, branch `feat/route-a-631`.

Read HANDOFF_658 first (defects M, N, and §4e the fairness-window
regression). This handoff adds the MEASURED seam peak, which changes what
"fix the seam" means, and closes defect M's guard and one contributor to N.

---

## 1. THE HEADLINE NUMBER: the cutover costs 1.4–3.0 GiB per card

Measured 2026-08-09 22:09Z on the live server (boot 22:04:14Z, yarn1.5,
ctx 393216, pool 253528, purity off, windows off), NVML free sampled at
100 ms across one real `pp_to_tp` cutover driven by
`POST /phase_flip {"direction":"pp_to_tp"}`:

| card | baseline free | trough free | SEAM PEAK | trough above 1024 floor |
|---|---|---|---|---|
| gpu0 5090  | 5705 MiB | 2725 MiB | **2980 MiB** | 1701 MiB |
| gpu1 3080a | 6248 MiB | 4872 MiB | **1376 MiB** | 3848 MiB |
| gpu2 3080b | 4507 MiB | 2679 MiB | **1828 MiB** | 1655 MiB |

Raw samples: `/tmp/seam.csv` (110 samples, 11.0 s). ONE cutover, one
direction, one config — labelled as such. It is not a distribution.

**Why this reframes everything.** Every previous conversation about
headroom was denominated in the 1024 MiB corridor floor, and §4e cites
ranks sitting "only ~530–610 MiB above the floor" at runtime. Against a
seam that transiently wants **1.4–3.0 GiB**, 530–610 MiB is not a thin
margin, it is a guaranteed death the next time a flip lands. That is
consistent with all three of the boots that died: the seam does not fail
because a flip is unlucky, it fails because the transient does not fit.

**Consequences, in the order they bite:**

1. The binding constraint on POOL SIZE is not the corridor floor. It is
   `card_total − corridor_floor(1024) − seam_peak(~1.4–3.0 GiB)`. Any
   pool sized against the floor alone is sized to kill the next flip.
2. Therefore the **full-KV >600k goal and the flip feature are in direct
   tension**, and that tension is now numeric rather than suspected. A
   bigger pool shrinks baseline free by exactly what it takes, and the
   seam peak does not shrink with it.
3. The fairness windows are not wrong so much as unaffordable: they raise
   the flip rate against a per-flip cost nobody had priced. §4e's verdict
   stands, and now with a number attached.

**Do not** read this as "the seam peak is a constant 3 GiB". It was
measured at pool 253528. The re-commit stages KV backing, so the peak
plausibly SCALES with the pool — which, if true, makes the >600k pool
worse than linearly. **Measuring the peak at two different pool sizes is
the single highest-value next measurement in this whole task**, because
it decides whether full-KV and auto-flip can coexist at all, or whether
the flip must become memory-flat first. It is cheap: two boots, one
scripted flip each, the sampler below.

Reusable instrument (written this shift, no new machinery needed):
sample `nvidia-smi --query-gpu=memory.free` at 100 ms, POST one flip,
take baseline = median of the pre-flip samples and trough = min.

## 1a. ANSWER (measured after the above was written): it SCALES

Point 2, pool 126000, produced by replaying the point-1 launch with ONE
substitution (`--max-total-tokens 500000 -> 126000`; per-card budget, ctx,
weights and corridor all unchanged):

| card | seam peak @ pool 253528 | seam peak @ pool 126000 |
|---|---|---|
| gpu0 5090  | 2980 MiB | **368 MiB** |
| gpu1 3080a | 1376 MiB | **0 MiB** |
| gpu2 3080b | 1828 MiB | **48 MiB** |

Pool x0.497 -> peak x0.12 / x0.0 / x0.026. Far more than proportional.

Direction confound EXCLUDED by log, not by assumption: both windows
straddle the same pair (epoch 3 `pp_to_tp`, epoch 4 `tp_to_pp`).

**Mechanism, which matters more than the ratio.** The instrument measures
the DRIVER-visible transient, and that is the correct observable because
it is exactly what refuses a `cuMemCreate`. The driver is asked for
(staging size) - (slack torch already holds). A bigger pool grows the
staging and shrinks the slack simultaneously, so the driver-visible peak
grows much faster than the pool. This is precisely how defect N died:
`empty_cache()` handed back the slack, and the next 128 MiB ask went to
the driver and was refused.

**So the branch is decided.** Full-KV >600k cannot be reached by sizing
around the seam peak. It needs pre-reserved, zero-allocation staging so
the cutover never asks the driver for anything. The affordability
pre-flight (§7.2) is still needed regardless, because until the staging
is pre-reserved an unaffordable flip must abandon rather than die.

---

## 2. Defect M — guard SHIPPED, source still open (as expected)

Committed inside `b0ae546715` (see §5 on why the commit boundary is odd).

The guard, and why each half exists:

* `phase_flip_resident_carry._reqs_of` now **type-checks before
  materialising**. The old body was `list(getattr(batch, "reqs", []) or [])`
  and *the hang was inside that `list()`* — ten million allocations are
  the damage, not a slow road to it. Any check written after the `list()`
  is a check that never runs. It names the offending object's TYPE, and
  reports a byte-count reading when the length is a MiB multiple (that
  reading is what identified the object class).
* `harvest_resident_batches` adds the scheduler-aware ceiling
  (`max_running_requests`) and names the offending SLOT. A list of the
  right type but an impossible length is still corrupt and still reaches
  `committed_slots`.
* `arm_draft_bootstrap` refuses an implausible resident set itself. The
  consumer that ALLOCATES guards its own input rather than trusting every
  present and future caller — this is the lethal half.
* `Scheduler.maybe_arm_phase_policy` catches `ResidentCarryError` and
  declines to arm, keeping the instance serving. Caught **only** there:
  that call site is a policy OBSERVATION where "don't flip" is always a
  safe answer. The CUTOVER path deliberately does not catch, because
  proceeding there is what killed the 21:47Z run.
* No clamping anywhere.

**The source is still unknown, and the guard is the instrument that will
name it.** A full read-only sweep of `python/sglang` found **no static
write of a tensor, ndarray, bytes, bytearray, memoryview or mmap to any
attribute named `reqs` in production code**, and no `setattr` loop that
could write one from foreign data. So M is runtime aliasing, not a
source-level bug, which is why reading the code has not found it and why
the next occurrence must be made to speak. Ranked leads:

* `distributed/device_communicators/barlink_ucx.py:347` — `_UcxAsyncHandle`
  genuinely carries `"reqs"` in its `__slots__`, meaning UCX request
  handles. Same name, entirely different meaning. Too small to be 10 MiB
  by itself, but it proves the name collision is real in this tree.
* `model_executor/input_buffers.py:181` — `share_input_buffers_in(obj)`
  walks `vars(obj)` and replaces **every** field with a pooled tensor
  keyed by attribute NAME. It asserts the buffer is a Tensor so it cannot
  convert a list, but it is a name-keyed global tensor pool that writes
  tensors onto arbitrary objects. Audit its callers.
* `managers/scheduler.py:5589/:5609` — a dataclass-field snapshot/restore
  round-trips `reqs` through a dict. Confirm both halves act on the SAME
  object.
* Arithmetic worth keeping in view: 10485760 = 10×2²⁰ = 4096×2560 and
  12582912 = 12×2²⁰ = 4096×3072. Both read as MiB byte counts AND as
  plausible `req_to_token` numels. The `mem_cache`/DCP/sparsity layer uses
  `reqs` as the name of an int32 tensor of request rows
  (`sparsity/algorithms/quest_algorithm.py:58`), while the scheduler layer
  uses it for a list of `Req`. That collision is the structural
  precondition.

## 3. Defect N — one contributor fixed, the dominant term NOT fixed

The reported cause ("cuMemCreate re-committing TP KV backing") was only
the line above the fatal one. The exception that actually killed the
22:00:12Z boot:

```
gdn_flip_mover.py:346 _pack_pp_side -> kv_reshard.py:198 _checksum
  -> weights_arena.py:66 uint8_checksum -> chunk.sum(dtype=torch.int64)
torch.OutOfMemoryError: tried to allocate 128.00 MiB, 106.38 MiB free
```

`sum(dtype=int64)` casts its input first, so the transient is 8× the
chunk, and the chunk was a CONSTANT 16 MiB set on 2026-08-08 against a
*host-RAM* blowup. 8×16 = exactly the 128 MiB that did not fit. It was
primed by the line above: the backing path had just been refused
161480704 bytes and called `empty_cache()`, handing back the very cache
this allocation would have used.

**Fixed** (`weights_arena._checksum_chunk_bytes`): the chunk is sized
against `torch.cuda.mem_get_info` on the payload's device — 25 % of free,
divided by 8 for the cast — floor 1 MiB, CPU path unchanged at 16 MiB.
The checksum VALUE is chunk-invariant (integer sum is associative), which
is what makes it safe when checksums are compared across ranks that now
pick different chunk sizes; that invariance is pinned by a test rather
than asserted here.

**HONEST LIMIT — the important sentence in this section.** That fix
removes a **128 MiB** contributor from a seam whose measured peak is
**1400–3000 MiB**. It is roughly 4–9 % of the problem. It converts one
specific reproducible crash into a non-crash; it does **not** make the
seam affordable, and it must not be quoted as having fixed defect N or
§4e. The dominant term is the KV-backing re-commit staging, untouched.

Coordinator's two options for the real fix, unchanged and both still
open: pre-reserve the staging so the flip is zero-allocation (the
VA-reservation machinery from exclusive backing is the natural home), or
extend the affordability pre-flight to cover the TP re-backing path so an
unaffordable flip **abandons cleanly instead of dying**. Given §1, the
pre-flight is the one that should exist regardless: with a peak of 1.4–3.0
GiB, "can this flip afford to commit right now" is a question the runtime
must be able to answer, and today it cannot.

## 4. Corpse M-host: the death mechanism was a HOST-RAM OOM

Recorded because it was mis-stated as a pure spin and the distinction
changes what to watch. Rank 0's scheduler took **exit code -9** — a
kernel SIGKILL, not a CUDA fault and not an exception
(`serving-30030.boot.20260809T214835Z.log:1266`). The container cgroup
shows `memory.events oom_kill` incremented and `memory.peak` at
**120348282880 B = 112 GiB of 120 GB**. `committed_slots` allocating one
tensor per claimed request against 10.5 M claimed requests is a host-RAM
exhaustion, and the kernel killed the largest process.

Practical consequences: `dmesg` is NOT readable in this container
(`read kernel buffer failed: Operation not permitted`), so
`/sys/fs/cgroup/memory.events` `oom_kill` is the only monotonic witness —
read it BEFORE and AFTER a run. A sampler for exactly this is committed
at `scripts/host_ram_sample.sh`; it separates `anon` from `file` because
only anon can OOM a swapless box, and page cache from reading a 27 GB
checkpoint three times inflates the total harmlessly.

Also note: an exit code of -9 will never appear as a Python traceback, so
a log search for `Traceback` finds nothing and the tail shows only NCCL
broken-pipe noise from the surviving ranks. Search for
`crashed with exit code` instead.

## 5. Process hazard that bit this shift: TWO agents in ONE worktree

Successor 15 and successor 16 were live in `/spinning/wt-631-routea`
simultaneously. Concrete damage:

* My defect-M and defect-N code and tests were swept into successor 15's
  commits (`9bdae43656`, `b0ae546715`) whose messages describe different
  work. The code is in and the author is right; the commit boundaries are
  a lie. Do not trust `git log -S` archaeology on this range.
* A full-suite run reported `1 failed`
  (`test_phase_policy::test_a_chunked_prefill_is_visible_to_the_policy`)
  that was a **race against a concurrent mid-edit save**, not a
  regression — the same file passed 82/82 in isolation seconds later.
  A red suite here needs an isolated re-run before it is believed.

If two strands must share a tree again, they need separate worktrees or
an explicit file-level split. Neither happened here.

## 6. Test state

* Family suite `bash scripts/run_631_flip_family.sh`: **616 passed** with
  the M fix (609 baseline + 7 new M pins), then **621 passed + 1 raced
  failure** with M+N (622 total = 616 + 6 new N pins); the raced file
  passes 82/82 isolated. Successor 15 reports 622 green at `1d6d8c5c09`.
* New pins, defect M (`test_phase_flip_resident_carry.py`): a non-list
  `reqs` is refused by TYPE; **the guard refuses WITHOUT iterating** (the
  pin that guards the fix's actual mechanism — a version that checked
  after materialising would pass every other test and still kill the
  instance); the byte-count reading is reported; plain lists pass through;
  the ceiling names the slot; no-ceiling schedulers still harvest.
* New pins, defect N (`test_weights_arena.py`): CPU keeps the measured
  16 MiB constant; a card with 106.38 MiB free (the crash's exact
  condition) shrinks the chunk so the 8× transient fits; a card with room
  keeps the big chunk; an exhausted card returns a usable floor rather
  than `split(0)`; an unqueryable device falls back; and **the checksum
  value does not depend on the chunk size**.

## 7. Exact next steps

1. ~~Measure the seam peak at a second pool size~~ **DONE, see §1a. The
   peak SCALES with the pool, steeply, so full-KV needs a zero-allocation
   seam rather than a bigger headroom allowance.**
2. **Seam affordability pre-flight** covering the TP re-backing path, so
   an unaffordable flip abandons cleanly. With a 1.4–3.0 GiB peak this is
   needed whether or not the staging is later pre-reserved.
3. **Then** pre-reserved (zero-allocation) staging, if the peak scales
   with the pool.
4. **Only then** re-enable the fairness windows and purity strict, and
   prove them on a ≥60-min run with a minutes-scale settle window.
   Purity flips often by design; it cannot precede a flat seam.
5. Full-KV boot-time route, gated on step 1's answer.
6. Graph A/Bs (NEXTN draft graphs, DFLASH×graphs, prefill-graphs-in-PP)
   and the 5090 stage-imbalance measurement — all untouched this shift.
7. Final green run with real agent traffic + judged extract, then ship.
   Note in the holder when a stable ≥60-min window exists; the operator
   re-arms the six Qwen doc agents as traffic at that point.

## 8. Standing state

Serving UP on 30030 since 22:04:14Z, yarn1.5 / ctx 393216 / pool 253528,
`phase_flip_policy=auto`, `phase_flip_purity=off`, fairness windows 0.
Health 200. One manual `pp_to_tp` flip issued this shift (the §1
measurement) — it armed and returned success; note that for any evidence
extract asserting "no manual flips".
