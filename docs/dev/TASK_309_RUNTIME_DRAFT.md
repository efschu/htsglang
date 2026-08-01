# Task #309 — runtime drafter add/remove, manual selection, task routing

The "alles auswaehlbar" program applied to speculation: a drafter should be
something an operator adds to and removes from a RUNNING server, the active arm
should be selectable rather than only controller-chosen, and a request should be
able to say what kind of work it is.

Status: **the decision layer is built and hermetically tested; the executor
that loads and frees weights is the GPU ticket.** Split that way on purpose —
everything below is the part whose failure modes are silent or corrupting, and
all of it is decidable without a card.

## The state-machine contract

`speculative/runtime_draft.py`. Four states, one transition point.

```
        request_attach                 step(quiesced)
DETACHED ──────────────► ATTACH_PENDING ──────────────► ATTACHED
   ▲                          │                            │
   │      cancel_pending      │                            │ request_detach
   └──────────────────────────┘                            ▼
   ▲                                                 DETACH_PENDING
   │              step(quiesced)                           │
   └────────────────────────────────────────────────────────┘
```

| state | `serves_drafts` | meaning |
|---|---|---|
| `DETACHED` | no | no drafter; server behaves as a spec-less boot |
| `ATTACH_PENDING` | **no** | accepted, weights not in yet |
| `ATTACHED` | yes | loaded and serving |
| `DETACH_PENDING` | **yes** | still loaded; the in-flight verify may finish against it |

The two `serves_drafts` answers in bold are the load-bearing ones. Serving from
`ATTACH_PENDING` is a forward against a drafter that does not exist; *not*
serving from `DETACH_PENDING` frees the draft pool under a verify that still has
to read it. Both are pinned by tests.

### The boundary

`step(report)` is the only transition point and takes a `QuiesceReport` —
running requests, waiting requests, spec-verify-in-flight, graph-active. It
is called from the scheduler's **#364 between-tick window**, where the previous
batch is retired and the next is not yet selected, so no captured graph can
replay while drafter-owned state is rewritten (#52/#53). Reused, not
re-derived: the GDN resident-slot executor already established that window and
already refuses to run outside it. `step` raises `TypeError` naming #364 if
called without a report, so the placement is enforced rather than documented.

**Quiesce rule.** Running requests block (a request admitted under one drafter
configuration must not have it changed mid-generation); an in-flight verify
blocks; an active graph blocks. Waiting requests do **not** block — they have
not been scheduled, so they pick up whatever is live when they are. Each of the
three blocking conditions is pinned by an isolating test, because a report that
trips two conditions still fails the predicate when one is deleted, and would
therefore let a safety condition rot unnoticed.

### The three hard cases

| case | behaviour |
|---|---|
| attach while requests in flight | held in `ATTACH_PENDING`, deferred at each boundary with a reason, lands on the first quiesced one |
| detach mid-verify | held in `DETACH_PENDING` and **keeps serving** until the verify completes |
| attach on a model that refuses spec | `DrafterUnsupported` — a distinct type from `DrafterBusy`, because retrying will not help and the operator's next action differs |

`cancel_pending()` exists so a boundary that never arrives under sustained load
is escapable without a reboot — the thing this task removes.

## Selection and routing

`speculative/draft_selection.py`. Rungs are the existing `cross_algo_utils`
`(family, value)` pairs — `("nextn", k)` / `("dflash", block)` — validated
against the same loaded-arm set `resolve_drafter_policy_table` validates
against. A second spelling of "which drafter" is how two tables drift.

**Precedence**, highest first:

1. `MANUAL` — an operator pin. A pin a controller could override is not a pin.
2. `ROUTED` — a request tag with an entry. Per-request beats per-server
   automatic because the tag describes *this* request's workload, which the
   controller cannot see.
3. `CONTROLLER` — #156's adaptive choice, now **one source among several**.
4. `BOOT` — the configured default.

Every resolution reports its `SelectionSource`, because "who chose this arm" is
the first question anyone asks of a run.

**Routing table**: `code=nextn:3,multiturn=nextn:5`. No built-in tag names —
which tags a deployment uses is a fact about its traffic. The canonical entry is
keeping multiturn off DFLASH (#156 measured it the worst arm there), and that
belongs in a config line, not an `if tag == "multiturn"`.

**No silent fallback.** An unloaded arm, an unknown family, a malformed entry, a
duplicate tag — each is a named `SelectionError` listing what *is* available.
Routing rungs are validated at parse time *and* re-validated at resolve time,
because a runtime detach can remove an arm a boot-validated table still names.
An unknown tag falls through to the controller by default (tagging some traffic
and not the rest is normal) and errors under `strict_tags` for deployments that
want to know their tagger and table have drifted apart.

## API surface (proposed; not yet wired)

Following the fork's `/x-htsglang/...` control-endpoint pattern:

| endpoint | body | returns |
|---|---|---|
| `POST /x-htsglang/drafter/attach` | `{"algorithm","k"|"block"}` | accepted + pending id, or a named refusal |
| `POST /x-htsglang/drafter/detach` | `{}` | accepted + pending id |
| `POST /x-htsglang/drafter/cancel` | `{}` | the withdrawn action, or null |
| `GET /x-htsglang/drafter/status` | — | `lifecycle.snapshot()` + active selection |
| `POST /x-htsglang/drafter/select` | `{"rung":"nextn:5"}` or `{"auto":true}` | the new `Selection`, or a named refusal |

Per-request routing rides an existing request field (`task_tag`) rather than a
new transport. Deliberately **not** wired in this commit: an endpoint that
returns "accepted" and never executes, because the executor does not exist yet,
is worse than no endpoint.

## Tests

`test/registered/unit/runtime_draft/`, 65 hermetic CPU tests. Lifecycle:
happy path, exactly-once execution, all three hard cases, each blocking
condition in isolation, boundary enforcement, cancel, snapshot, and the
**detached-is-inert regression pin** (stepping a detached machine through every
report shape forever changes nothing — so a server that never attaches behaves
as a spec-less boot). Selection: rung parsing, arm validation, routing parse and
resolve, the full precedence order, and every refusal.

Falsifier-checked: making `DETACH_PENDING` stop serving drafts reds the
mid-verify test; deleting the verify condition from the quiesce predicate reds
two tests. That second one **initially stayed green** — the combined report also
set `running_requests=1`, so the test passed for the wrong reason. The isolating
reports were added in response; noted here because the same shape will recur
whenever a predicate gains a term.

## What the GPU ticket must validate

One TP=3 uneven-DCP boot per arm, all with full CUDA graphs.

1. **Attach on an idle server, then generate.** Boot spec-less, attach a NEXTN
   drafter, confirm `spec_accept_length` becomes non-trivial. Until this passes
   nothing else is meaningful.
2. **Attach under sustained load.** Drive continuous requests, request an
   attach, confirm it defers (the status endpoint shows the reason), then lands
   when the load drains — and that no request in flight across the boundary
   produces a malformed answer.
3. **Detach returns VRAM to the pool.** Measure free VRAM before attach, after
   attach, after detach. The #119 post-install re-profiling and #330 VRAM dial
   are the levers; the point to prove is that the freed bytes are *usable* (KV
   capacity goes back up), not merely unreferenced.
4. **Detach mid-verify does not corrupt.** Request a detach while a spec batch
   is verifying; the in-flight request must complete coherently and the detach
   must land after it. This is where a wrong `serves_drafts` answer shows up.
5. **Refusal on an unsupported model.** Attach a drafter to a model whose spec
   support is refused at parse time; the runtime refusal must name the same
   reason and leave the server serving.
6. **Manual selection switches the live arm**, and `spec_accept_length` moves
   with it. Then confirm the controller does not override the pin.
7. **Routing takes.** Tag two request streams differently and confirm each ran
   its mapped arm — read it from the per-request selection source, not inferred
   from throughput.
8. **The detached-is-inert claim, on hardware.** A boot that never attaches must
   be byte-identical to a spec-less boot on the same prompt at temperature 0.
   This is the regression the hermetic pin can only argue for.

## Honest remainder

1. **The executor.** `step()` returns `"attach"` / `"detach"`; nothing performs
   them. The weight load must go through the memory-saver/suspend machinery
   (`enable_memory_saver`, the torch-memory-saver adapter) rather than a second
   loader path, and the freed state must be tagged in the #286 offload register.
2. **Not wired into the scheduler.** The `step()` call at the #364 boundary is
   not placed yet; it lands with the executor.
3. **No endpoints.** Surface proposed above; wiring one before the executor
   exists would ship a knob that reports success and does nothing.
4. **No `task_tag` on the request object** and no server flags
   (`--drafter-routing-table`, `--drafter-strict-tags`) yet — same reason.
5. The rung vocabulary covers `nextn` and `dflash`. EAGLE3/STANDALONE/NGRAM are
   spellable as families but have no arm-set representation; adding them means
   extending `ArmSet`, and the refusal is currently "unknown drafter family".
