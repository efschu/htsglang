# #305 determination — multi-model serving, per promise

**2026-08-17, Slot-3. Determination + one small composition.** Every claim below
was verified at code; the two mechanical sweeps were delegated, the load-bearing
facts re-read by hand.

**Headline: #305 is far more BUILT than its `in_progress` status suggests, and
what is missing is not movers — it is BINDING.** A real control plane exists
with real actuators. Nothing in the serving path drives it, and one ladder edge
does not exist for any tenant class.

## The four promises

| promise | verdict | evidence |
|---|---|---|
| **N registriert** | **BUILT** (in-memory) | `EngineSpec` (`registry/spec.py:118-212`), `EngineRegistry.register` (`arbiter.py:367-428`), `deregister` (`:430-438`), HTTP `POST/DELETE /registry/engines` (`http_api.py:109-123`). Server-side, locked. **Runtime additions are not persisted** — the `--engines` file (`launch.py:56-64`) is read at start and never written back; `ledger.py`'s disk store holds VRAM *leases*, not specs. |
| **M hot** | **mechanism BUILT, live trigger ABSENT** | `hot_capacity()` (`arbiter.py:553-614`) derives capacity from the byte ledger, applies `--registry-max-hot` as a cap (`:581-586`), and `_promote`/`_demote` (`:773`, `:883`) bind to real `adapter.promote/demote`. But `acquire_for_request`/`ensure_state` are reached **only** from the admin route `POST /registry/engines/{id}/state` (`http_api.py:125-147`). The OpenAI request path never calls them. |
| **Idle-Default-Set** | **set is OPERATOR-DECLARED; automatic demotion ABSENT** | `default_hot` comes from the config file (`launch.py:60-61`) or `POST /registry/default_hot` (`http_api.py:184-189`); nothing derives it. The actuator `return_to_idle()` (`arbiter.py:929-958`) is real and has **zero automatic callers** — `arbiter.py:932-935` names its own absence: *"Idle handling is explicit rather than a background thread: the control plane calls it on its own tick"*. There is no tick. |
| **Residenz-Leiter** | **all four rungs EXIST; one edge exists nowhere** | see below |

## The ladder, and the vocabulary gap that hides it

The rungs exist under **different names**, which is exactly what makes them
re-discoverable:

| #305 promise | `rungs.py:56-61` | stored `TenantState` (`ledger.py:80-84`) |
|---|---|---|
| HOT | `HOT` | `HOT` |
| **TEIL-HOT** | `TEIL_HOT` | `WARM_GPU` |
| **WARM** | `WARM` | `WARM_HOST` |
| COLD | `COLD` | `COLD` |

`rungs.py:24` states its own scope: *"NOTHING HERE MOVES A MODEL. Rungs are
declared and reported."*

### Per-transition verdicts

| transition | verdict |
|---|---|
| HOT ↔ TEIL-HOT | actuator **real** (`class1_srt.py:238-243`, reusing the live `/release_memory_occupation` / `/resume_memory_occupation` endpoints); **orchestrator unwired** |
| HOT ↔ WARM | actuator **real** for Class 2 (`class2_diffusion.py:290-318`); **live and automatic** for the translator (#546: `translator/idle_park.py` → `translator/ledger.py:394,525`) |
| **TEIL-HOT ↔ WARM** | **ABSENT FOR EVERY CLASS** — structurally, not by oversight |
| any → COLD | **working**, HTTP-triggered (`http_server.py:1874` → `weight_updater.py:326` → `model_loader/hibernate.py`); round trip flagged BOOT-PENDING in the catalog |
| generic per-item HOT↔WARM (#286) | **UNWIRED** — `RealMovementBackend` (`offload_movement.py:639`) has zero production callers; `model_runner.py:622` passes no backend, so the default fake one is used even with the flag on |

**Why TEIL-HOT ↔ WARM is absent, in the code's own words:** Class 1 refuses
`WARM_HOST` — *"sharded, quantised, post-processed weights are not a"*
reloadable image (`class1_srt.py:220-224`); Class 2 refuses `WARM_GPU` — *"the
upstream server has no route to drop just the BCG pool"*
(`class2_diffusion.py:276-279`); Class 3 refuses it too (`class3_utility.py:88-93`).
**Each class refuses the rung the other implements.** A ladder walk that assumes
the four rungs form a chain is wrong on this rig.

## #306 cold-tier compression: MEASURED AND REFUTED, not absent

Compression was built, measured, and found to give **zero** allocation win on
this box's ZFS (ZFS already captures the same zero-page fraction) and a real win
only on non-folding filesystems, with no write-time win on either — inside the
noise floor. What shipped instead is `sparse_write.py` (lseek-over-zero-pages,
default-on, sha256-gated), which is *not* compression. Recorded as a negative
result so it is not re-attempted.

## What is actually running live today

Narrower than either framework: two hand-built tenant-specific movers — **#546
translator idle-park** (fully automatic) and **#364 GDN session ladder** (called
from `scheduler.py`) — plus whole-server **#89 hibernate**. None is the general
ladder #305 promises.

## Built here: the reachable-edge declaration

`registry/ladder.py` + 16 tests. It **moves nothing** (pinned) and adds no
mover. It declares, per engine class, which rungs exist and which transitions
are therefore reachable, so a caller is refused **before** driving an actuator
rather than discovering an `AdapterError` mid-promotion — and the refusal quotes
the architecture. `universally_absent()` **computes** the TEIL-HOT↔WARM gap
rather than asserting it, so the day any adapter implements both middle rungs,
this determination's central claim retires by arithmetic.

The declaration is pinned against the adapters' own refusal text, so it cannot
drift from the code it describes (a mutation adding a rung the adapter refuses
turns 7 tests red).

## Built since: the binding and the tick (2026-08-18, Slot-2)

Remainders 1 and 2 below are now code. Desk + hermetic; the GPU leg is still
open (see "Still open" further down).

| piece | where | what it does |
|---|---|---|
| **request-path binding** | `entrypoints/openai/request_binding.py`; `serving_base.py` `handle_request` / `_serve_bound` / `_serve` | model name -> registry lookup -> acquire -> hold for the request lifetime -> release. Two binders behind one contract: `InProcessBinder` (an `EngineRegistry` in this process) and `HttpBinder` (the control plane's new `POST /registry/engines/{id}/acquire` and `/release`). |
| **control tick** | `registry/tick.py`; `--tick-interval-s` in `launch.py`; `POST` and `GET /registry/tick` | the periodic idle-set evaluation `return_to_idle`'s docstring names as missing. Steps idle tenants ONE rung down, only along edges `ladder.py` declares reachable. |

**Refusal, never a hang.** Four named errors: 404 `model_not_found`, 409
`ladder_edge_unbuilt` (waiting cannot help — the rung does not exist for that
class), 503 `engine_not_wakeable` carrying the arbiter's own projected wait and
eviction list, 503 `registry_unreachable`. The anti-hang property is
structural, not a timeout bolted on: a FINITE `max_promotion_wait_ms` (30 s
default) makes the arbiter refuse an expensive promotion up front rather than
start it, and the HTTP binder's socket timeout bounds the conversation.

**The single-model fast path pays nothing.** `binding_enabled()` is one
module-global boolean, false unless `SGLANG_REQUEST_BINDING` names a control
plane. Pinned two ways: the fast path returns the object `_serve` produced
(identity, not equality), and a counting binder must record zero calls.

**The tick is not `return_to_idle` on a timer.** That actuator takes everything
outside `default_hot` straight to COLD, discarding the middle rung on the only
class that implements it. The tick asks `ladder.step_down_target` which rung
the class can actually reach next — adjacency is not reachability — reports any
rung it stepped over, refuses an undeclared class loudly and leaves it where it
is, and NEVER promotes (that would be cut 4, gate UNFULFILLED per #375). It
does not touch #286's `RealMovementBackend`; that is pinned by source
inspection, so a later edit that reached for the mover turns red.

**Defect found on the way, and fixed.** `ensure_state` routed on the TARGET's
name (`if target in (HOT, WARM_GPU): _promote`), so HOT -> WARM_GPU went into
`_promote`, where Class 1 refuses it outright ("no promotion path HOT ->
WARM_GPU", `adapters/class1_srt.py`). **The TEIL-HOT rung was unreachable
downward for the only class that has it** — the rung this determination calls
the valuable one could be entered from COLD but never from HOT. It now routes
on direction (`_RESIDENCY_RANK`, pinned against `ladder.RUNG_ORDER`).

**In-flight accounting** is new on the arbiter (`acquire_for_request` /
`release_after_request` / `inflight`, and in `snapshot()`). It is what stops
the tick demoting mid-generation: `last_used_ts` records when a request
STARTED, which is not enough for a generation longer than the idle threshold.

Tests: `test/registered/unit/entrypoints/openai/test_request_binding_305.py`
(31) and `test/registered/unit/registry/test_control_tick_305.py` (23). 54 new,
zero new failures against a 660-passed baseline at the harvest tip.

## Honest remainder — what is genuinely open

1. ~~**The binding.**~~ BUILT (above). What remains is the GPU leg: no boot has
   yet driven a real promotion through the request path, so promotion LATENCY
   under a live request is unmeasured.
2. ~~**The tick.**~~ BUILT (above), default off. Its steps have never been
   executed against a real Class-1 adapter, so the TEIL-HOT park's actual
   reclaim is still the card window's question.
3. **Registry persistence** for runtime-added specs (design cut 5).
4. **Cut 4 (autonomous promotion policy) should NOT be built**: its gate is
   recorded UNFULFILLED in #375 — the objective divergence that would justify it
   never materialized. Building it now would be building a policy for a
   divergence nobody has observed.
5. **TEIL-HOT ↔ WARM** stays absent unless an adapter class gains both middle
   rungs; on the stated architecture (Class 1's weights are not a reloadable
   image) that is unlikely to change for the sglang class.
