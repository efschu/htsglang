# DESIGN #347 — the idle workbench

Status: **M1 implemented** (work-queue abstraction, one scheduler, three
registered tenants, read-only + control HTTP surface). This file is the
persistent record of the #347-M1 analysis and decisions; the chat history is
not authoritative.

## Goal

ANALYSE #347 item 1: training is one idle tenant among many. The machinery
#341 built for training — idle detection, VRAM lease, preemption by
checkpoint-and-release, an event channel — carries a *queue* of useful idle
work. This slice generalizes it: one interface, one scheduler, one priority
order, everything preemptible by serving demand inside the existing grace
window. Training becomes tenant #1 of N without changing how it behaves.

## Decisions

### W1 — Generalize the tenant, do not duplicate it

`python/sglang/srt/workbench/` is the generalization; `python/sglang/srt/
training/` is untouched except for one optional keyword argument. The three
mechanisms #341 named stay exactly where they were:

| #341 mechanism | Where it lives now | Who uses it |
|---|---|---|
| `IdleMonitor`, `DemandSample`, `IdleVerdict` | `training/tenant.py` (unchanged) | imported by the workbench scheduler |
| ledger lease (`MultiCardReservation`) | `registry/ledger.py` (unchanged) | the scheduler leases for tenants that do not lease themselves |
| checkpoint-and-release preemption | the `BackendRun.preempt` contract | mirrored by `WorkSegment.preempt` |
| event channel with cursor pagination + subscribers | `training/store.py` `JobStore` | pattern re-applied in `workbench/log.py` for a rig-wide, job-less log |

Idle detection is *not* re-implemented. The workbench builds an
`IdleMonitor` from the same two sources the training service uses
(`local_activity_source`, `registry_activity_source`), so a rig that looks
idle to training looks idle to the workbench by construction rather than by
coincidence.

### W2 — The interface: estimate, feasibility, bounded segment, events

`workbench/tenant.py`:

```
IdleWorkTenant            name, priority, available(), pending(), estimate(),
                          start_segment(grant, sink), snapshot(),
                          enqueue(item), pause/resume
WorkEstimate              per-card bytes with NAMED POSTS, cards wanted,
                          disk/RAM, expected seconds, self_leased
Feasibility               fits / reason / chosen cards / shortfall, rendered
                          with the arithmetic (D2 rule)
WorkSegment               wait() / preempt(timeout_s) / cancel(timeout_s)
SegmentOutcome            status + artifact_path + data
WorkEvent, EventSink      the same shape as BackendEvent / EventSink
```

`WorkSegment` deliberately has the same three methods as `BackendRun`. A
tenant that can only be killed is interruptible, not preemptible; `preempt`
is the difference and every tenant must answer it inside
`--workbench-preempt-timeout-s`.

**Work is segmented, not open-ended.** A segment is a unit the tenant can
abandon cleanly: one tuner shape, one card probe, one training attempt. The
scheduler grants one segment at a time and re-evaluates demand between
segments as well as inside them, so the worst-case latency between a request
arriving and the rig being free is `poll_seconds + the tenant's own preempt
time`, never "until the queue is empty".

### W3 — Feasibility is the D2 formula, applied to any work

`price_segment()` in `workbench/tenant.py` is the D2 rule generalized off
training: every post named, no safety factor, no implicit ceiling, priced
against the *actual* machine (NVML totals) and the *current* ledger
(`plan_reservation`, so the corridor and other tenants' holdings count).
Card choice picks the emptiest cards that each satisfy the per-card demand;
a tenant may pin cards instead. A rejection renders the arithmetic —
requested, held, corridor, total, shortfall — exactly as a training
rejection does.

`self_leased=True` is the one escape hatch, used by the training adapter:
training prices and leases per *job*, inside its own gate, because the
amount depends on the job. The scheduler then arbitrates ordering only and
does not lease on that tenant's behalf. Said out loud rather than assumed.

### W4 — Priority order, one scheduler

Lower number runs first. Registered in M1:

| Priority | Tenant | Work item | Preemption | Artifact |
|---|---|---|---|---|
| 10 | `training` (#341 adapter) | one training attempt | `force_preempt` -> the trainer checkpoints, then the loop is stopped and the per-job lease released | the job's own `output_dir` |
| 50 | `fp8_tuner` (#255 remnant) | one block-FP8 GEMM shape | SIGTERM the tuner subprocess; the shape stays queued and nothing partial is written | `<artifact-root>/fp8_tuner/configs/` |
| 70 | `card_probe` (dashboard self-benchmark) | one short-probe run over every card | SIGTERM the probe subprocess; the factor stays absent | `~/.cache/sglang/card_probe-<digest>.json` |

The order is a policy statement: a user's submitted training job outranks
work the rig invented for itself, and measurement that only refreshes a
dashboard tile outranks nothing. It is a per-tenant integer, so a deployment
that disagrees changes one number.

The scheduler never runs two tenants at once. Idle work is opportunistic by
definition and two opportunistic tenants sharing a card would measure and
tune each other.

### W5 — The workbench claims cards through the cross-session arbiter

`workbench/arb.py` implements the `/spinning/gpu-arb/` protocol as code:
`holder` with a heartbeat, orphan detection, `free-until` respected as a
promise to the other session, and — before every claim, regardless of what
the files say — an NVML occupancy check, because the hardware is right and
the files are intentions. Windows can exceed 20 minutes (a tuner queue, a
long training attempt), so the heartbeat is touched from the supervision
loop, not once at claim time.

The directory is a flag (`--workbench-arb-dir`, or `$HTSGLANG_GPU_ARB_DIR`)
and defaults to off. Hardcoding this rig's path would be exactly the
rig-only assumption ANALYSE #347 rules out; the path is a deployment fact,
the protocol is the mechanism.

### W6 — The tuner commits nothing

The tuner tenant writes tuned configs to its artifact directory and reports
the repo path an operator would copy them to. It does not write into
`python/sglang/srt/layers/quantization/configs/` and does not commit. A
kernel config that lands in the tree without a human reading the A/B is a
silent performance change with no provenance; #255 round 2 is on record as
having been decided by a measured comparison, and that gate stays.

Shapes come from a queue file or the enqueue endpoint, never from a
hardcoded list: a shape that matters on this rig is a fact about this rig's
model and TP split. A shape whose config file already exists for the running
device name is skipped, which makes the queue idempotent and makes "run it
again after a driver change" a delete-and-requeue rather than a code edit.

### W7 — Self-benchmarking wraps existing measurement, adds none

The `card_probe` tenant runs `python -m sglang.srt.rigmon.card_probe --run`
as a subprocess — the same entry point `ProbeJobStore` already uses for
`POST /api/card_probe`. It owns no measurement code. `pending()` is 1 when
the cached profile is absent or older than `--workbench-probe-max-age-s`,
which is the ANALYSE #347 sentence "the dashboard's absent factor tiles
re-measure themselves when the rig is idle", implemented as a queue entry
rather than as dashboard behavior. One factor pair in M1 (`card_rates` +
`pair_link`, filled by one run); the other tiles are M2.

### W8 — The API is read-only plus two controls

Under the `x-htsglang` namespace on the serving surface, #305-M1 style
(named 503 when the feature is off, never a 404):

```
GET  /x-htsglang/workbench              snapshot: config, tenants, current
                                        segment, idle verdict, arb state
GET  /x-htsglang/workbench/events       cursor-paginated event log
POST /x-htsglang/workbench/pause        {"paused": bool, "tenant": name?}
POST /x-htsglang/workbench/enqueue      {"tenant": name, "item": {...}}
```

No frontend in M1. The Training and Rig tabs consume this later; the surface
is the contract, the page is one client of it.

## M1 as built (2026-07-31)

| Module | Carries |
|---|---|
| `workbench/tenant.py` | W2 interface, W3 pricing, `SubprocessSegment` |
| `workbench/log.py` | the rig-wide event log: ring buffer, seq cursor, subscribers |
| `workbench/arb.py` | W5 cross-session claim, heartbeat, orphan reap |
| `workbench/scheduler.py` | the loop: idle -> price -> claim -> lease -> segment -> release |
| `workbench/service.py` | assembly from server args, tenant registration |
| `workbench/http_api.py` | W8 payload functions, framework-free |
| `workbench/tenants/training.py` | the #341 adapter (W4 priority 10) |
| `workbench/tenants/fp8_tuner.py` | W6 (priority 50) |
| `workbench/tenants/card_probe.py` | W7 (priority 70) |

None of them imports torch, so the whole surface is testable on a card-less
host — the same property #341-M1 holds.

Server flags: `--enable-idle-workbench`, `--workbench-artifact-root`,
`--workbench-tenants`, `--workbench-idle-grace-seconds`,
`--workbench-poll-seconds`, `--workbench-preempt-timeout-s`,
`--workbench-segment-timeout-s`, `--workbench-arb-dir`,
`--workbench-arb-heartbeat-s`, `--workbench-tuner-queue`,
`--workbench-probe-max-age-s`. Documented in `docs/rig-runbook.md` §11.

### Resolved: who owns the training loop

Two schedulers deciding when training runs is one too many. The adapter owns
`TrainingTenant.start()` / `stop()`: with `--enable-idle-workbench`, the
training service is started surface-only (`start(start_tenant=False)`) and
the loop is started when the workbench grants training a segment. The
training tenant's own code is unchanged — the same loop, the same idle
check, the same lease — only the caller of `start` moved. That is why the
#341 test suite passes unmodified: it constructs `TrainingTenant` and
`TrainingService` directly and never went through `http_server`.

`TrainingService.start()` gained a keyword-only `start_tenant: bool = True`.
Default preserved, so every existing caller behaves as before.

### Resolved: a segment is granted, not a card

`WorkGrant` carries card UUIDs *and* their NVML indices, and the tenants set
`CUDA_VISIBLE_DEVICES` from the indices on the subprocess they launch. This
is the fork's standing isolation rule: inside the child, `cuda:0` is
unambiguous, and no in-process logical-to-physical mapping table exists.

### Resolved: an unavailable tenant is skipped by name, not retried blindly

`available()` returns `(bool, reason)` — the tuner script missing from an
installed wheel, `pynvml` absent, a probe module that cannot import. An
unavailable tenant contributes no work and says why in the snapshot, which
is the `BackendProbe` pattern from #341-D1 applied one level up.

## Open items / deliberately M2

- **More tenants.** ANALYSE #347 names four more, all of which fit this
  interface without changing it: TRT staircase rung compilation (#337),
  CUDA-graph prewarm for registered-but-cold engines, cold-tier compression
  of hibernate images (#306), and the integration boot matrix (#349) as a
  standing bug net. None is built here.
- **The other seven factor tiles.** Only `card_probe` is wrapped. `power`
  (`POST /api/measure_power`) and `prefix_cache` (`POST /api/hicache_saved`)
  are the obvious next two; `mlp_split` is the crossover sweep and is the
  expensive one.
- **Segment cost is estimated, not measured.** `expected_seconds` comes from
  the tenant's own guess. A measured per-segment cost should feed the
  decision of whether a segment is worth starting when the rig has been idle
  for only slightly longer than the grace window.
- **The event log is in memory.** A restart loses it, which is honest
  because a restart also kills every segment. Persisting it needs the
  segments to outlive the server, the same M3 shape #341 has.
- **No SSE endpoint.** `WorkLog` carries subscribers so the frontend can
  stream later; only the cursor-paginated GET is wired in M1.
- **Priority is static.** Aging (a starved low-priority tenant eventually
  outranking a chatty high-priority one) is not implemented; with three
  tenants and a queue that drains, it has nothing to fix yet.
