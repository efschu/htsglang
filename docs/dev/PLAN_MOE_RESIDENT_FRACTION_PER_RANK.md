# Per-rank MoE resident-expert fraction

Branch `feat/moe-resident-fraction-per-rank`, off `feat/dsv4-sm86-sm120-417`
at `6e3b850afb`. Desk phase: `CUDA_VISIBLE_DEVICES=99` on every python call.

## Why

Window 2 (#417 follow-up) closed a feasibility box on this rig. Six
configurations, all carrying the #417 attention fixes, all loading cleanly,
none failing on anything #417 touches:

| cfg | f | scratch | reserve(3080) | outcome |
|---|---|---|---|---|
| 417b | 0.45 | 4 | 1400 | served warmup, died on first generate (5 spill > 4 slots) |
| 417c | 0.45 | 8 | 1400 | VRAM OOM in load |
| 417d | 0.45 | 6 | 1400 | KV budget fail by **32 MiB** |
| 417e | 0.40 | 6 | 1400 | host guard, `memory.current` 101.5 > 98 GiB, peak 103.6 |
| 417f | 0.45 | 6 | 1000 | KV pool created, warmup OOM, **152 MiB free** |

The routed top-k is 6 (measured: activations/tokens = 8460/1410 = 6.0), so
`SGLANG_MOE_SCRATCH_SLOTS` must be at least 6 or a token that spills 6 experts
kills the server. At scratch=6 the 20 GiB cards are 32 MiB short; the obvious
lever (lower `f`) moves the pressure onto the host, because the pinned pool
lands in FILE (#391 rider B) — f=0.40 peaked 103.6 GiB against a 104 GiB
container.

`SGLANG_MOE_RESIDENT_EXPERT_FRACTION` is declared `EnvFloat(1.0)`: **one
scalar for every rank**. Measured at f=0.45 in 417f the 5090 sat at 28569 of
32607 MiB (~4.0 GiB spare) while both 3080s were short. There is spare VRAM on
this rig, on the wrong card, and no way to express using it.

## What the vector actually does — correction to the framing

The brief describes the fix as "shifting enough experts to the 5090". It does
not do that, and the difference matters for the arithmetic.

`f` is **per-rank and applies to that rank's own expert shard**: it sets the
GPU-resident / host-pinned split *within* a rank. It does not move experts
between ranks — that is `--rank-moe-ratio` (shard ownership), a different
axis, and it is not what is being changed here.

So a vector helps through **two independent mechanisms**, not one:

1. **Lower `f` on the 3080s** — frees 3080 VRAM (the binding constraint),
   at the cost of a larger host pinned pool.
2. **Raise `f` on the 5090** — spends the 5090's idle ~4.0 GiB and *shrinks*
   the host pinned pool, because those experts move from host RAM onto the
   5090.

Mechanism 2 pays for mechanism 1 on the host side. A scalar can do neither
without doing the other harm everywhere; that is the whole point.

## Fixposten (feasibility before measurement), from measured anchors

Anchors, all measured on this rig (boot 11b notes + windows 1-2):

| term | 3080 rank | 5090 rank |
|---|---:|---:|
| non-expert VRAM | 6.62 GiB | 8.31 GiB |
| expert bytes at f=1 | 24.47 GiB | 42.21 GiB |
| usable VRAM (observed) | 19.58 GiB | 31.34 GiB |

Cost of the two extra scratch slots, derived from two measured points:
417b (f=0.45, scratch=4) allocated `max_total_num_tokens=90624`, i.e. a
0.70 GiB KV pool at `bytes_per_full_token=7705.45`, so weights+runtime there
was <= 18.63 - 0.70 = **17.93 GiB**. 417d (f=0.45, scratch=6) measured
weights+runtime = **18.66 GiB**. Two slots therefore cost 0.73 GiB, i.e.
**~0.365 GiB per scratch slot** on a 3080 rank.

3080 requirement as a function of its own fraction, at scratch=6:

    W(f) = 18.66 - 24.47 * (0.45 - f)      [GiB]

Budget the rest from measurement, not from taste: KV pool 0.35 GiB (417f's
44544 tokens), warmup transients 0.40 GiB (417f's failing allocation was
256 MiB, so 400 gives margin), corridor floor 0.40 GiB.

    W(f) <= 19.58 - 0.35 - 0.40 - 0.40 = 18.43 GiB
    24.47 * (0.45 - f) >= 0.23  ->  f <= 0.4406

Take **f_3080 = 0.42** (not the bare minimum — 417f showed the transient peak
is not fully characterised, so buy margin):

    frees 24.47 * 0.03 = 0.734 GiB  ->  W = 17.93 GiB
    headroom = 19.58 - 17.93 = 1.65 GiB for KV + transients + corridor
    corridor after a 0.35 GiB KV pool and 0.40 GiB transients: ~0.90 GiB
    -> comfortably above the 400 MiB floor

Host side, kept neutral on purpose. Two 3080 ranks each push
24.47 * 0.03 = 0.734 GiB to the host, so +1.468 GiB. To give that back:

    42.21 * delta = 1.468  ->  delta = 0.0348  ->  f_5090 = 0.485

5090 VRAM cost of that: 42.21 * 0.0348 = **1.47 GiB** against ~4.0 GiB spare,
leaving ~2.5 GiB. Fits with room.

**Proposed vector: `0.485,0.42,0.42`** (rank 0 = 5090, ranks 1,2 = 3080).
Host pinned pool stays at the f=0.45-uniform footprint that 417b already
proved fits (cgroup peak ~87 GiB against a 98 GiB guard); no new host risk is
taken, which is the point of balancing rather than just lowering.

## Design

`SGLANG_MOE_RESIDENT_EXPERT_FRACTION` has ~15 call sites of three different
kinds. Rather than change the meaning of `.get()` under all of them, the
value is resolved through one accessor and the raw env is made to **fail
loudly** if a vector is set and something still reads it as a scalar. A
missed call site must be a clear error, never a silently wrong number — the
same discipline as #192's pairing invariant and #343's per-device gates.

* `EnvFloatVector` in `environ.py` (alongside the existing `EnvTuple`
  comma-list precedent): parses `"0.45"` and `"0.485,0.42,0.42"` alike.
* One accessor module, used by every SIZING and ACCOUNTING site:
  `resident_fraction_for_rank(rank)`, plus `offload_active()` for the
  ACTIVATION sites (`< 1.0`) — those ask "is offload on at all", which is
  true if *any* rank offloads.
* `--rank-moe-resident-fraction f,f,...` in the rank-vector flag family,
  validated **eagerly** at CLI-parse time like its siblings (length == tp_size
  or 1, each entry in (0, 1]). Deliberately at the top of `_handle_uneven_tp`,
  before that method's early return: the fraction is meaningful whether or not
  an uneven-TP plan is active, and a wrong length must be a CLI error rather
  than a worker crash six minutes into a 98 GiB checkpoint stream.
* Env list and flag must agree or hard-error, per the brief. Note this
  deliberately differs from the `--rank-*-ratio` family, where the env var
  silently wins; these two set the same thing, so disagreement is a launch
  that means two things, and it is refused.
* Scalar path stays byte-identical and is pinned by a test.

Three things the consumer map turned up that the first cut had wrong:

* **The rank identity is the MoE-TP rank, not the plain TP rank.** The
  fraction splits a rank's own *expert* shard, so `moe_tp_rank` is the
  identity that owns it. On a pure-TP model the two are the same number, but
  they are separate groups in this codebase. Where they differ a vector is
  ambiguous -- its length is specified per TP rank but it would be indexed by
  the MoE rank -- so that combination is **refused by name** rather than
  guessed. A scalar stays legal there, because a scalar is unambiguous.
* **`offload_active()` must stay group-wide.** `_expert_offload_lane_active`
  gates an `all_reduce(MIN)` and a barrier. A rank that answered "my own
  fraction is 1.0, so no offload" would skip a collective its peers enter and
  hang the group. The helper is therefore `any(f < 1.0 for f in vector)`,
  which every rank computes from the same launch string and so cannot
  disagree on. The weight-placement sites in fp8/awq/gptq use the same
  group-wide answer, for the weaker but still real reason that ranks should
  construct structurally identical modules.
* **The offline planner was silently dropping the pool.**
  `planner/placement.py::_compute_offload` did `float(env)` and, on the
  `ValueError` a comma-list raises, fell back to `frac = None` -- which omits
  the entire offload section from the capacity report. That is an unbooked
  pool in the one place whose job is to book it (#400 class). It now parses
  the vector and sizes each simulated rank from its own entry; a length that
  matches neither 1 nor `tp` raises instead of silently reporting nothing.
  The single `resident_fraction` headline field in the returned rule still
  carries rank 0's value, with the truth in the per-rank list beside it.

## Not done here, with reasons

**The rammon guard needed no change.** The brief called it "a hardcoded 98".
It is not: `scripts/dsv4/rammon.sh` already derives the guard from the cgroup
limit, and its own first log line proves it —
`rammon: limit=104.0 GiB margin=6 GiB guard=98.0 GiB (on memory.current)`.
98 was 104 - 6, i.e. the ceiling the user raised it to, minus the margin I
passed. Changing the margin to 4 GiB would have set the guard at 100 GiB and
would **not** have saved boot 417e, which peaked at 103.6 GiB — it would have
traded a clean guard stop for a kernel OOM kill 2 GiB later, which is worse
because it can take unrelated processes with it. No change made.

One real latent hazard was found while checking, and is left alone as
out-of-scope: `resolve_limit_bytes()` falls back to `MemTotal` when
`memory.max` reads `max`. On this container lxcfs makes that the honest
ceiling, but the script's own comment notes it is the wrong number on a bare
host. Worth a ticket, not a same-pass edit.

---

# Window-4 plan, and the two probes that precede it

## Item 1a — the prefill peak is now measured, not inferred

`model_executor/forward_peak.py`: `reset_peak_memory_stats` before every
forward and `max_memory_allocated` after, bucketed by phase and token count,
driven from `_forward_raw` (the same place the #343 layer tap lives, for the
same reason: the model is entered as `model.forward(...)`, so a module hook
would never fire). A `peak_scope` context manager closes the bracket on the
exception path too -- an OOM is exactly when the peak matters.

Off unless `SGLANG_FORWARD_PEAK_PATH` is set.

**It writes a file per rank, not a log line, and that is evidence-driven.**
The explorer confirmed `/server_info` already aggregates a per-rank
`memory_usage` dict through `get_internal_state`, which is the natural home
for a live number. It is the wrong home for THIS one: the measurement exists
to characterise a peak that precedes an OOM death, and `/server_info` needs a
live server to answer. The per-rank JSON files survive the death -- that is
how the window-3 expert_stats arrived from a boot whose rank had already
OOMed. Worker `logger.warning` records do not reach the log at all on this rig
(#417), which rules out the third option. Adding the same number to
`get_internal_state` afterwards is a good follow-up, not a substitute.

## Item 1b — a per-rank chunked-prefill-size is NOT possible. Named limit.

Checked, and the answer is a hard no, not an omission. `chunked_prefill_size`
**is the batch-shape knob**: it sets how many tokens each rank contributes to
prefill's own collectives. Every rank parses the same `server_args` and reads
one scalar; the only place it is ever divided (`dp_size`) divides it
identically everywhere.

Making it per-rank would reproduce precisely the failure class this fork
already built machinery to prevent. From `kv_session_offload.py:2559`:

> "A divergent decision makes rank 0 take the prefill branch while ranks 1/2
> take the decode branch of get_next_batch_to_run; those branches carry
> DIFFERENT collectives, so the ranks desync (NCCL/gloo hang, observed in
> recv_requests one iteration later)."

and `scheduler.py:3643` documents the admission budget as deliberately
"RANK-UNIFORM". `--rank-moe-resident-fraction` is safe as a vector *because*
it only moves bytes within a rank's own weight/host split and never changes a
batch shape; chunk size is the opposite knob. This is a genuine
physical/logical limit, so it is recorded as an exclusion with its reason
rather than left as a gap.

**Consequence for window 4:** the prefill transient on the 3080s can only be
reduced by lowering the GLOBAL `--chunked-prefill-size` (which costs prefill
throughput on all three ranks, including the 5090), or by freeing that rank's
memory some other way. It cannot be shaved per card.

## Item 2 — the load-time page-cache race, made managed

Chosen: **trim coupled to consumer progress** (`ProgressCoupledTrim` in
`model_loader/gguf_shards.py`, called from `ConsumedPageDropper._flush_shard`
after each advice batch). Off unless
`SGLANG_GGUF_STREAM_TRIM_SOFT_GIB` is set.

Why this one of the three:

* The failure is a **rate** mismatch, not a threshold error. `cachetrim.sh`
  samples on a 5 s wall clock; window 3 moved `memory.current` 88.2 -> 102.5
  GiB inside one 15 s window. Whether a boot lived came down to where a noisy
  peak fell between two samples -- w3a survived the same recipe w3b died on.
* Coupling the trim to the stream makes a faster load trim more often,
  automatically, with no interval to tune. That is the difference between
  managed and lucky.
* *Loader waits on a trim watermark* (option a) is this plus a stall: strictly
  more invasive, and it can deadlock if the watermark is unreachable because
  the resident pool -- which reclaim cannot touch -- is already above it.
* *A tighter load-phase threshold* (option c) only moves a number that a 5 s
  sampler can still overshoot; it does not address the rate.

`cachetrim.sh` stays as-is and is still useful (it also trims what the loader
does not cause). The rammon guard is untouched, as instructed.

Safety is the same argument as `cachetrim.sh`'s and is **checked, not
assumed**: with no swap, cgroup reclaim cannot evict anonymous memory, so the
pinned pool, CUDA host allocations and the Python heap are structurally out of
reach and only page cache can be taken. With swap present it refuses to act.
It also disables itself if `memory.reclaim` is unavailable -- a probe must
never be why a load fails.

## Item 3 — window 4

**One boot, ~30 min.** w3a's vector unchanged: `--rank-moe-resident-fraction
0.485,0.42,0.42`, `SGLANG_MOE_SCRATCH_SLOTS=6`, reserve `2200,1400,1400`,
`SGLANG_OPT_USE_TOPK_V2=0`. That configuration reached HEALTHY with the full
corridor (639/1830/639 MiB free) and KV 90624; it failed only on the prefill
transient, which is the thing now being measured.

New for this boot:
* `SGLANG_FORWARD_PEAK_PATH=<run>/peak` -- the measurement that makes the next
  fixposten arithmetic real.
* `SGLANG_GGUF_STREAM_TRIM_SOFT_GIB=88`, `..._TARGET_GIB=78` -- below the
  98 GiB rammon guard with room, so the load peak is managed rather than raced.

Measurement order: A-vs-A floor (byte-compare; window 3 measured the floor at
1.02%) -> prefill point -> windowed decode bs=1 (~15 s) -> coherence probe ->
expert_stats -> **collect the peak JSONs from all three ranks** -> cards FREI.

Prefill approach, deliberately staged so the transient is characterised even
if it still OOMs: start at a ~256-token prompt and step up (256, 512, 1024,
2048). Each step that completes writes a peak row; the first step that fails
brackets the ceiling between two measured numbers instead of producing one
failed allocation size. That is the difference between this window and the
last one.

### The graphs question: named reason, and a real but non-free escape hatch

`--disable-cuda-graph` is **not** a conservative choice and **not** about DSV4
or GGUF. It is forced by MoE expert offload, with a fail-fast at layer
construction (`fused_moe_triton/layer.py:612`):

> "MoE expert-offload ... requires --disable-cuda-graph: the per-forward
> expert residency plan is data-dependent and cannot be captured into a CUDA
> graph."

The mechanism is a device->host sync (`topk_ids.tolist()`) plus Python-side
planning inside every forward, which is illegal during capture
(`expert_offload.py:68`). Since window 4 runs at fraction < 1.0 on every rank,
graphs cannot be enabled for prefill. **Stated, not forced**, as instructed.

There is one genuine escape hatch worth naming rather than burying:
`SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` opens a capturable **decode-only** path
(frozen residency, on-device index math, captured UVA gather). It would give
the graphs decode point the brief asked about. It is not free: it *freezes*
the resident set, so it is a different configuration, not a toggle on the same
one, and its decode number would not be comparable to the eager number from
the same boot. Recommendation: take the eager numbers first in window 4, and
treat a frozen-residency graphs arm as its own later boot with its own A-vs-A
floor -- comparing an eager and a captured number across configurations would
be exactly the kind of cross-arm claim the benchmark rules forbid.
