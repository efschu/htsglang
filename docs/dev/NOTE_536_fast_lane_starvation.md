# #536 — fast-lane starvation: priority cannot buy memory

Desk only, 2026-08-17. No boot, no model load, no serving contact. **No fix
shipped**, deliberately: the mechanism does not name a clear scheduler cut, it
names a capacity decision. §6 states the stop condition.

## 0 — Retraction of my own first root

I first concluded that the translator "never asked for the fast lane" — that
`MtConfig.extra_body` defaults to `{}` and `mt.py:_body` adds no `lane`, so
`is_fast_lane` was always False. I wrote and landed a client-side default on
that basis.

**That was wrong, and I reverted it.** The deployed translator *does* ask:

- `translator/launch.py:134` — `--mt-lane` defaults to `"fast"`
- `translator/launch.py:483-484` — `if args.mt_lane: extra_body["lane"] = args.mt_lane`

The library-level `MtConfig` carries no lane because `launch.py` is the single
authority for it, which is correct; adding a second default would have created
two authorities for one value. My change also duplicated an existing fix rather
than adding one.

The error was reading `mt.py` in isolation and concluding absence from the file
I happened to open. `launch.py`'s own comment even records the earlier
measurement that motivated the lane: *"Measured without a lane: MT
time-to-first-token 19.7 s behind a 46k prefill, against a ~0.24 s median on an
idle backend."*

## 1 — The mechanism, verified at the code

The fast lane is a **queue-ordering and preemption** mechanism. Admission is
additionally gated on **KV capacity**. Those are different resources, and the
second is what binds here.

**A fast-lane request cannot preempt an in-flight chunked prefill.** Three
links, each verified directly:

1. `scheduler.py:6346-6361` — when `self.chunked_req is not None` the chunk is
   advanced **unconditionally** (`adder.add_chunked_req`), *before* the
   waiting-queue admission loop at `:6380-6397` where preemption can fire.
2. `schedule_policy.py:1626-1634` — `preempt_to_schedule` builds its victim set
   from `self.running_batch.reqs` only.
3. `scheduler.py:3005` — **"a chunked request is resident but sits in no
   batch (#631 defect O)"**, which is why the scheduler can read
   `#running-req: 0` while that request's prefix is locked.

So the chunked prefill is in neither the preemptible pool nor any batch, while
holding KV behind a locked prefix. Preemption can evict running *decode*
heavies, but it cannot free what the chunked request holds, and
`scheduler.py:3005` records that eviction on exactly this specimen "would have
delivered 0 and said so".

**The consequence, and it is the finding:** priority orders the queue, but a
fast request still needs KV to be admitted. When the memory it needs is held by
a co-tenant's in-flight chunked prefix, no priority value releases it. The fast
request waits not one chunk but until that prefill completes. A worst case of
tens of seconds is structurally available regardless of `fast_lane_priority`
being 1,000,000.

## 2 — What the fast lane *does* fix, so the scope stays honest

It works when the binding constraint is queue order rather than capacity:

- `DESIGN_466_live_translator.md:4058` — `mt_first_token` measured at
  **0.13–0.37 s** over the fast lane, i.e. inside the design budget.
- `launch.py:477-482` — the without-lane baseline was 19.7 s behind a 46k
  prefill against ~0.24 s idle.

So this is not a broken feature. It is a feature whose guarantee stops at the
resource it does not control.

## 3 — The budget the specimen violates

From `DESIGN_466_live_translator.md:335-347`:

| stage | budget |
|---|---|
| **MT first token** | **150–400 ms** |
| first translated audio after the pause | ~1.1–2.4 s, inside a 2–3 s target |

34.5 s is 86×–230× the stage budget.

## 4 — The premise I could NOT establish, and it decides the next step

**Was `--enable-fast-lane` actually on, and did the request carry
`lane="fast"`, during the 34.5 s run?** Not determinable from code:
`enable_fast_lane` defaults to False (`server_args.py:1235`) and is a launch
argument; `--mt-lane` defaults to `"fast"` but can be emptied.

The two possibilities need different work and must not be conflated:

- **Lane was OFF** — then 34.5 s is a pre-fix/misconfigured baseline, the
  remedy is operational (launch with `--enable-fast-lane`), and §1 is a latent
  finding rather than this specimen's cause.
- **Lane was ON** — then §1 *is* the cause, and the remedy is a capacity
  decision, not a priority one.

Guessing between them would be exactly the error I made in §0. The live window
resolves it in one observation (§5).

## 5 — What the next live window must capture

Boot with `--enable-fast-lane --enable-metrics` and record, for a fast-lane
request issued while a large chunked co-tenant prefill runs:

1. **Was it fast-lane at all** — TTFT is already labelled by priority when
   priority scheduling is on (`tokenizer_manager.py:2517-2529`), so a
   `priority=1000000` series existing at all answers §4 directly.
2. **Queue versus forward split** — `observability/req_time_stats.py:1027-1044`
   (`get_queueing_time`, `convert_to_duration`) already produces
   `queue_duration` / `forward_duration` per request. If §1 is the mechanism,
   the fast request's time is almost entirely `queue_duration` and it tracks
   the co-tenant's remaining prefill, not its own forward.
3. **`mt_first_token_ms` against 150–400 ms**, A/B: the same turn set alone
   versus under co-tenant load.

**Harness gap.** `scripts/translator/contention_probe.py` is already the right
A/B stopwatch — it drives its own session at a fixed cadence and reads the same
per-stage numbers the server publishes on `turn.done`, deliberately not scoring
audio so it stays off the critical path. But its load is a **second translator
conversation**, not a heavy co-tenant chunked prefill, which is the #536 shape.
Replacing the load generator is the one piece of setup the window owes.

## 6 — Why no fix here (stop condition)

Per the brief: fix only if the mechanism names a clear cut. It does not.

The cut that would follow from §1 is either (a) making an in-flight chunked
request preemptible, or (b) reserving KV headroom the fast lane can always be
admitted into. Both are capacity decisions with a stated cost, not scheduler
hygiene:

- (a) touches the chunked-prefill path, whose protected prefix exists to keep a
  partially-prefilled request from losing work; preempting it re-does that
  prefill, so the co-tenant pays twice and the throughput cost is unbounded in
  the pathological case.
- (b) is the #274 Slice-D pairing question in a different dress: standing
  headroom for a latency tenant is capacity permanently withheld from the
  bulk tenant, and how much depends on the fast tenant's arrival rate and
  working set — neither measured here.

**Failure direction, stated for whichever is chosen:** both trade co-tenant
prefill throughput for translator latency. The existing bounds are
`fast_lane_reserved_heavy_slots = 1` (heavy forward progress is guaranteed) and
`fast_lane_heavy_aging_ms = 10_000` (a heavy request waiting >10 s is promoted
ahead of the fast tier for one admission). Note the asymmetry that leaves: a
starved heavy request waits up to 10 s by design, while the fast tenant's own
budget is 150–400 ms. Neither bound helps with §1, because both act on queue
order and §1 is about memory.

## 7 — What this note does not claim

It does not touch `dual_group_lane.py`. That subsystem (#274) has its own tick
(`scheduler.py:2363-2368`, `:8087`), its own thread and CUDA-stream priority,
and is gated on `--dual-group-lane`, which nothing in `translator/` references.
It is not required to explain this specimen and is neither cleared nor
convicted here.

It measures nothing. The 34.5 s remains unvalidated by anything in this note,
and §4 is unresolved by design rather than by omission.

**Naming collision worth flagging:** `translator/idle_park.py:30` cites "#536"
for a VRAM idle-park threshold controller reusing the `SpillTickController`
pattern (`kv_session_offload.py:2215`). That is unrelated to fast-lane
scheduling. Two different pieces of code carry the same ticket number, and only
one is this bug.
