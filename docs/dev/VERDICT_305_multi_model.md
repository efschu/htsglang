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

## Honest remainder — what is genuinely open

1. **The binding.** Every state-changing verb is admin-HTTP or startup-file
   only. Wiring the request path to `acquire_for_request` is the single change
   that would make #305 autonomous — and it is a serving-path change needing a
   boot, not desk work.
2. **The tick.** `return_to_idle()` needs a caller. Also boot-gated.
3. **Registry persistence** for runtime-added specs (design cut 5).
4. **Cut 4 (autonomous promotion policy) should NOT be built**: its gate is
   recorded UNFULFILLED in #375 — the objective divergence that would justify it
   never materialized. Building it now would be building a policy for a
   divergence nobody has observed.
5. **TEIL-HOT ↔ WARM** stays absent unless an adapter class gains both middle
   rungs; on the stated architecture (Class 1's weights are not a reloadable
   image) that is unlikely to change for the sglang class.
