# DESIGN #286 — the short-term offload register, asset-class layer

Status: **desk slice, no page has ever moved.** The register plans, refuses and
prices; the one production mover is desk-written and never executed. §7 is the
BOOT-PENDING list and the GPU ticket that would retire it.

Scope of this document: what already existed before this slice (§1), what the
slice adds and why each piece is where it is (§2–§5), the falsifier results
(§6), what needs a card (§7), and the one contradiction the work surfaced and
did not resolve (§8).

---

## 1. Survey — what #286 already was

Task #286 is long-open and had two prior merges. Neither is re-implemented.

| Merge | Module | What it owns |
|---|---|---|
| `1981846dc5` / `5846cf8ed5` (CPU phase) | `model_executor/offload_register.py` (1328 L) | Item registration, per-class knobs `resident\|ram\|auto` + depth fraction, hysteresis, hot refusal, priority protection, `on_phase_boundary` / `on_admission_boundary` planners, the process-global holder, `SGLANG_OFFLOAD_REGISTER` gate |
| `c226fe498e` / `53db42152a` (GPU prep) | `model_executor/offload_movement.py` (930 L) | Movement state machine `resident → park_in_flight → parked → wave_in_flight`, `TensorPayload` / `TagPayload` (#93 pause/resume) / `SuspendPayload` routes, park-target ladder, directed P2P capability model, `CapacityLedger` |
| same | `offload_sizes.py` (107 L), `offload_bus_budget.py` (258 L) | Live size resolution (already understands the #102 `_StateRecord.footprint_bytes`); the PCIe arbiter |
| `2a568f04c7` (Erg. 8) | `offload_gdn_states.py` (375 L) | `gdn_state_sets` as a class with its own session ladder |
| #428 | `server_args.py` + `configure_global_register_from_server_args` | `--lane-offload-profile / -class-policy / -park-targets` reach the register at runner init |

`graph_rungs` also already has **two** producers: the per-lane speculative
K-ladder (`dual_group_lane.py:1861+`, ids `lane{n}/graph_rung/k{k}`) and the KV
pressure ladder's steps (`kv_pressure_ladder.py:691`, ids
`kv_ladder/graph_rung/{step}`).

**What was therefore genuinely missing**, and is what this slice builds:

1. No asset-class *descriptors*. The register knew class NAMES and nothing
   about what a class is — no ladder position, no payload class, no grain.
2. No executable importance ladder. `DESIGN_407` §8 and catalog §17 make one
   global order mandatory; it existed only as prose in two documents.
3. No partial spill. `park()` is per item; who goes first was the caller's.
4. No graph-state **family** asset. Both existing producers key by *rung
   within one running configuration*. `DESIGN_363` §20.3 needs a whole
   layout/algo family, pre-captured at boot and parked while inactive.
5. No refusal under active capture anywhere in the register.
6. No consumer of the #407 registry — `PARK_TARGETS` was three hand-written
   strings, and #421 F6 pinned the absence.

---

## 2. The one deliverable that is a decision, not a mechanism

The ladder. `DESIGN_407` §8's five rungs are now `LadderRank`, an `IntEnum`, so
"least important first" is a plain sort and the doctrine's order cannot drift
from a re-spelling. `plan_spill` walks:

1. ascending rank;
2. ascending `last_access_s` within a rank — §8's "coldest-first within a
   class";
3. item id, purely for reproducibility.

Two rules are enforced rather than documented:

* **Rank 5 is unreachable.** §8.5's "active work, last — and never out of FCFS
  order" is a #273 guarantee owned by the scheduler's admission path. A memory
  planner that could reach it by widening a budget would spend a fairness
  guarantee it does not own. `_UNPLANNABLE_RANKS` makes that a refusal with a
  named reason, not a preference.
* **Spill is partial.** The walk stops at the first item whose cumulative bytes
  cover the shortfall. §8: "only the overflowing part spills". A class is never
  emptied to satisfy a request a single item already covers.

`plan_spill` is **pure**. Nothing moves, so a controller can price a rung before
committing to it and a hermetic test can assert the ORDER with no mover at all.
`SpillPlan.render()` itemises rank, class and bytes per step.

---

## 3. Graph-state families

`GraphFamilyRegister` is the `graph_rungs` class at layout/algo-family grain.

* **Id namespace** `graph_family/{key}`, deliberately disjoint from the two
  existing producers. Three producers share one class and one id space; a
  collision would re-bind an item rather than raise. Pinned by test.
* **Hot criterion** is the family's own `active` flag, and `set_active` clears
  the others — exactly one family serves. Two hot families would make both
  unparkable and the whole rung a no-op.
* **Sizing** comes from #102's tags. There is no driver query-by-tag API: the
  real figure is the measured free-memory delta around `pause`
  (`_StateRecord.paused_bytes`, exposed as `footprint_bytes`), which only
  exists after a first park. Registering with `size_bytes=0` and refreshing
  later is the normal lifecycle, and `offload_sizes` already understands the
  attribute. A 0 is reported as *unknown*, never as free: `plan_spill` skips a
  zero-sized item with a reason naming `refresh_sizes()`.
* **Movement** goes through `GraphStateMover`. The production implementation
  `AdaptiveGraphStateMover` delegates to `AdaptiveGraphMemoryManager`
  (`pause_after_build` / `ensure_active`) — the module that already owns the
  per-tag `torch.cuda.MemPool`s, the per-tag `graph_pool_handle()`s and the
  torch_memory_saver tag interception. **No second VMM layer**, for the same
  reason #330 did not build one: #330's dial reaches the driver through
  `KvVmmArena.commit_range` / `decommit_range`, and this register reaches it
  through the tag pause/resume. Both hand pages back; they are sequenced by the
  caller and never nested, and this module never calls the dial.
* **`restore_cost_ms` defaults to 0.0 and should stay there** until a card
  measures it. 0.0 makes an item look free to retrieve, which biases a planner
  *against* parking — the safe direction. Substituting §20.3's ~25 ms would be
  substituting a projection for a measurement (see §7).

---

## 4. The capture gate

`OffloadUnderCaptureRefused`, raised by `refuse_if_capture_active`, probing
`runner_utils.capture_mode.get_is_capture_mode()` (which ORs the model-capture
contextvar with the breakable-graph one — both context-local, #274 slice C, so
the answer is about the capture this thread's park would corrupt).

Why a park is refused and not merely deferred: a park unmaps physical pages.
That is eager work, and #452 settled where eager work belongs — BETWEEN
replays, with the compute captured. Unmapping pages a recording capture is
writing into corrupts the recording rather than failing it, which is why the
capture check runs *first* among the runtime gates, before the base register's
policy checks.

Both directions are gated. An onload remaps, which is as illegal mid-capture as
an unmap; retrieval being unrefusable on *policy* grounds does not make it legal
at any moment. **Planning is not gated** — it touches no page, and refusing it
would blind the controller exactly while a capture runs, for no physical reason.

Precedent followed: a named `RuntimeError` subclass carrying structured fields
(`operation`, `subject`, `where`), as `offload_capture_gate.OffloadCaptureBreach`
does, so a caller meaning to retry at the next replay boundary can catch this
and nothing else.

---

## 5. Tier targets from the #407 registry

`price_park_target` is the fork's first production consumer of `memtier`.

`origin` is a **required** argument with no default. The registry orders
candidates by bandwidth, and the fastest tier admitting a class is almost always
the card the bytes are already on; parking there is not parking.
`offload_register` already spells the rule out for its own ladder — `own_vram`
is tier 0, "stay resident", not a park destination — so the origin is dropped
here rather than left to every caller to notice. Note this is a rule about the
ORIGIN, not about device tiers: a peer card stays a legitimate target.

Refusals are the registry's own, and all of them bite:

* volatility law — a graph family's evacuated capture state may not rest on a
  `RECONSTRUCTABLE_OK` tmpfs, while a lane workspace may (pinned by test);
* headroom;
* `require_measured_bandwidth=True` by default. This register acts on the
  number, so an ESTIMATE is not good enough without an explicit caller opt-in,
  and an ABSENT one is refused under every setting — the #348b D4 defect one
  layer up. There is a second, defensive absent-check inside
  `price_park_target` itself; falsifier F3 showed it is load-bearing rather than
  decorative (see §6).

**The derived `move_ms` is an ESTIMATE even off a MEASURED bandwidth, and is
labelled a FLOOR.** It is a link time. The move is dominated by remap and
zeroing, which the link model does not contain — see §7.

### Payload classes, and why each

| Class | Payload | Reason |
|---|---|---|
| `graph_rungs` | `EXPENSIVE_RECONSTRUCTABLE` | Regenerating means a cold recapture, priced by §20.3 RUNG 2 at 3-6 s of visible stall — not the "known, bounded cost" a droppable payload means. Declaring it droppable would let the table admit a resting place that may lose the bytes; the whole point of the #93 route is that content is evacuated, not discarded. |
| `drafter_heads` | `EXPENSIVE_RECONSTRUCTABLE` | Same rung — the weight half of the same "this family is not running" decision — but no capture points at it, so `va_stable_required=False`. |
| `lane_workspaces` | `RECONSTRUCTABLE` | Genuine scratch: content after a resume is undefined anyway. The one class here that may rest on a `RECONSTRUCTABLE_OK` tier. |
| `cold_lane` | `EXPENSIVE_RECONSTRUCTABLE` | #89 suspend route. Indivisible: the suspend path takes a lane's whole tag set, so partial spill of this class is a no-op rather than a fraction, and its only dimension preset is the lane. |
| `experts` | `RECONSTRUCTABLE` | The host pool is the source of truth; the VRAM copy is a cache. |
| `gdn_state_sets` | `DEVICE_BOUND` | DESIGN_407 X2. See §8. |
| `kv_shadow` | `RECONSTRUCTABLE`, rank 1 | The old layout stays the source of truth while a shadow exists, so park/discard is free. Cheaper to give up than a cold second model, hence most disposable of all. |

---

## 6. Falsifier results (executed, 2026-08-03)

`CUDA_VISIBLE_DEVICES=99`, `test/registered/unit/model_executor/test_short_term_offload_register.py`.
Baseline **55 passed**. Eight can-fail arms, each run and reverted:

| # | Neuter | Result |
|---|---|---|
| F1 | `_ladder_sort_key` → `(0, 0.0, item_id)` | **3 red**: global ladder order, coldest-first, partial spill |
| F2 | `refuse_if_capture_active` removed from `offload_family` / `onload_family` | **4 red**: named error, nothing-moved, onload refusal, post-capture pass |
| F3 | `require_measured_bandwidth=False` + `allow_unmeasured_bandwidth=True` | **1 red** (estimate). Absent stayed green — caught by the module's own defensive check |
| F3b | F3 **and** the defensive absent-check removed | **2 red**: absent and estimate |
| F4 | `break` dropped from `plan_spill` | **1 red**: partial spill reports 4 steps for a 1-item request |
| F5 | origin exclusion dropped from `price_park_target` | **8 red** across the pricing group |
| F6 | `_UNPLANNABLE_RANKS` emptied | **1 red**: rank 5 becomes plannable and `satisfied` flips True |
| F7 | `rung1_evict` class list widened to the whole ladder | **1 red**: experts and sessions appear in a RUNG-1 plan |
| F8 | `move_ms` relabelled `Provenance.MEASURED` | **1 red** |

Baseline restored and re-run green after each arm.

---

## 7. BOOT-PENDING — what needs a card

Nothing in this slice has run on a GPU. Claims that require the window, stated
so none of them reads as established:

1. **`AdaptiveGraphStateMover` has never executed.** Verified at the desk: the
   call surface exists at the names used (`pause_after_build`, `ensure_active`,
   `_states[tag].footprint_bytes`). NOT verified: that a pause/resume pair
   driven from *this* caller, at a replay boundary rather than at the manager's
   own ladder boundary, leaves a captured graph replayable.
2. **No reload-latency figure is claimed.** `DESIGN_363` §20.3 carries ~25 ms
   per state class as an explicitly provisional projection. **The tree's own
   #102 GPU validation contradicts it**: `adaptive_graph_memory.py:211-213`
   records swap latency "organic avg 40-51 ms, max 85 ms (vs 14 ms Stage-1 —
   the price of remapping+zeroing ~1 GB per swap)" on the 5090 + 2×3080 rig.
   That is the nearest measured analogue and it is ~2-3.4× the projection. The
   ~25 ms target should be treated as refuted-pending-remeasurement, not as a
   budget.
3. **`price_park_target`'s `move_ms` is a floor.** Link time only. The gap
   between it and the observed swap time is the remap+zero cost, which is
   exactly what item 2 says the projection omits.
4. **The park→mover ordering is not transactional.** `OffloadRegister.park()`
   sets the parked flag before the mover runs, so a mover failure leaves the
   item marked parked with its pages still mapped. Deliberate for now: making
   it transactional needs a real mover to fail against. Open risk.
5. **Per-family capture-pool residue is assumed, not measured, for a LAYOUT
   family.** #102's ~0.3 GB/state is the spec-ladder rung case; §20.3 itself
   carries it "by analogy" to the layout-family case.

### GPU ticket for the next window

Vehicle: the standard TP=3 uneven boot on 5090 + 2×3080. Hold `/spinning/gpu-arb/`.

1. **Pre-capture N families.** Boot with N ≥ 2 declared graph/layout families;
   confirm all N capture at boot and record per-family `footprint_bytes` from
   the pause delta. Compare against #102's ~0.3 GB/state (item 5).
2. **Offload N-1.** Register each family, `set_active` one, `rung1_evict` the
   rest with `execute=True`. Confirm: free VRAM rises by the recorded
   footprints; `nvidia-smi` and `mem_get_info` agree.
3. **Flip + reload.** `set_active` a parked family, `onload_family`, serve a
   token. Gate: **the graph replays without recapture** — this is the load-
   bearing claim of the whole #363 §20.3 route, and it is item 1's unknown.
4. **Measure ms/reload,** A-vs-A on the target family per the #360 standard,
   decomposed as §20.3's "measurement duty when built" requires: diff move /
   repack / KV move / graph-state reload separately. Report against the 40-85 ms
   #102 band, **not** against ~25 ms.
5. **Capture-gate can-fail on hardware:** call `offload_family` from inside a
   real capture and confirm `OffloadUnderCaptureRefused` rather than a
   corrupted graph. The hermetic arm fakes the probe; this one does not.
6. **Compose with #330:** run a dial shrink and a family park in the same boot,
   sequenced, and confirm neither strands pages the other expected.

---

## 8. Unresolved: `gdn_state_sets` has no park target

Pinned as a test
(`TierPricingTest::test_a_device_bound_class_has_no_park_target_at_all`) rather
than papered over.

`memtier.tiers.admission_refusal` states the `DEVICE_BOUND` law in two parts:
the volatility table admits it only on `DEVICE_BOUND_ONLY` tiers, **and** it
"never travels, not even one hop over P2P". Once the origin card is excluded —
and it must be, because parking to where the bytes already are is not parking —
a `DEVICE_BOUND` class has nowhere to go at all.

That is the #407 law working exactly as written. It contradicts
`offload_register.py`'s Erg.-8 docstring, which says a surplus GDN session set
is "parkable (host RAM or peer VRAM)", and the `offload_gdn_states` ladder built
on that claim.

One of the two is wrong. Either:

* GDN state is genuinely device-bound and the Erg.-8 ladder can only ever
  *free* sets, never park them — in which case its docstring and its target
  list need correcting; or
* a lossless, order-preserving round trip of GDN state to host RAM is in fact
  safe, and `DESIGN_407` X2's assignment is too strict — in which case the
  payload class, not the ladder, is what changes.

This slice does not wire `gdn_state_sets` and does not settle it. The test
exists so the next author meets the contradiction as a red assertion rather than
as a silent divergence between two documents.

---

## 9. Interfaces this slice deliberately did NOT edit

Concurrency discipline — two other agents were active in adjacent territory.

* **The regime controller / autocheck / `DESIGN_363` files** (#363-S1's
  strand). The §20.3 consumption contract is implemented on THIS side as
  `GraphFamilyRegister.rung1_evict(bytes_needed, execute=False)`. Integration
  point for the controller: call it plan-only to price RUNG 1, call it with
  `execute=True` to enter the rung, and call `onload_family(key)` +
  `set_active(key)` on a flip. Nothing in `DESIGN_363` was edited.
* **`layers/moe/expert_offload.py` and `offload_capture_gate.py`** (#452-prep's
  strand) — read-only reference for the refusal precedent.
* **`offload_register.py` itself** was not modified. The ladder rank lives in
  this module's class descriptors, which is also how `DESIGN_407` §8 frames it
  ("importance is a registry attribute per asset"). A future per-ITEM rank
  override would want a field on `OffloadItem`; it is not needed yet and would
  have widened the merge surface for no current consumer.
