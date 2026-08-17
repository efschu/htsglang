# ANALYSE #553 — elastic co-residence: what exists, what is missing, where to cut

Desk analysis, 2026-08-04, branch `fix/spill-composability`. **No build.** The
deliverable is an inventory of the existing building blocks against the events
#553 needs, and a cut list ordered so each cut is provable on its own.

Scope of #553 as taken here: a tenant (translator, video, a second model, a
training job) must be able to go HOT and COLD on demand, in **both**
directions, while a serving tenant keeps running — not merely "park when idle
and wake on request", which #546 already does for one tenant, but a general
event both a planner and an operator can raise against any registered
consumer.

A standing constraint runs through every verdict below: **CUDA-graph
compatibility is the target state, not an optional extra.** A path that only
works in eager mode is a step backwards from where the fork already is, so
every cut states what it does to graph state and how it gets back to full
coverage. Where a cut cannot yet be graph-compatible, the route to becoming so
is part of the verdict rather than a footnote.

---

## 1. The four blocks, and what each ACTUALLY does

### #330 — the VRAM dial (`managers/vram_dial.py`)

**Is:** a real mechanism. CUDA-VMM-backed KV pool whose physical backing can be
committed and released at runtime; `_commit_on_scheduler`,
`_refresh_capacity_snapshots`, `verify_pool_reached_capacity`, and a
`DialParticipant` registry so several pools can move together.

**Constraint that matters most to #553:** `store_bound_rows` is a LIFETIME
bound baked into captured graphs — "CUDA graphs captured over these buffers
bake this bound into their store_kvcache launches, so growing past it would
fail as a device assert on legal slot ids (#352) rather than here". So the dial
can shrink and re-grow freely **below** the captured bound, and cannot cross it
without re-capture. That is the single most important fact for #553: it means
elastic co-residence has a **free band** (down to any floor, back up to the
boot bound) and a **hard wall** above it.

**Verdict for #553:** usable as-is, inside the band. The band is also exactly
where graph compatibility is preserved, so the first cut should live entirely
inside it and never touch capture.

**Also relevant:** `validate_vram_dial_compat` currently REFUSES the dial
together with kv-session-offload and with hicache storage ("host rows sized
from boot C"). #553 will have to revisit that, and it is the same shape of
refusal as #547 — a sizing assumption, not an impossibility.

### #364 — the GDN resident-slot ladder (`managers/gdn_slot_runtime.py`, `mem_cache/gdn_slot_*.py`)

**Is:** a real, executing mechanism — a planner (`vacate_plan`) plus an
executor that exports a session's GDN/Mamba state to a host blob and restores
it into any free slot. Idle-only vacate; active sessions are never victims.

**Verdict for #553:** the closest existing thing to a per-consumer hot/cold
event, and the only one that already moves state OUT and back IN under a cap.
Its inventory seam (`live_offload_reqs`, fixed under #551) is the pattern the
other consumers need: **the owner enumerates its live set; consumers never read
a container directly.**

**Graph note:** the ladder moves state between slots, not geometry, so it is
graph-neutral by construction. It is the existence proof that a hot/cold event
need not touch capture.

### #287 — the KV pressure ladder (`model_executor/kv_pressure_ladder.py`)

**Is:** a PLANNER ONLY. The module docstring says so plainly: "The table, the
sensor, the flip contract and the handover INTERFACE — nothing moves. Every
real handover strategy is a `NotImplementedError` stub". Seventeen
`NotImplementedError` sites confirm it.

**Verdict for #553:** this is the biggest gap, and also the best-specified one.
The ordering invariants are already enforced in code (`base` < `relief` <
`geometry_flip` < `external`; one rung at a time), and the `relief` rungs
reference EXISTING features by name rather than reimplementing them — so the
relief half of the ladder can be made to execute by delegating to mechanisms
that already work (admission cap, uneven-DCP ratio, KV spill, session offload).
The `geometry_flip` half is the one that needs capture work, and the docstring
already states the intended route: per-step graphs captured in advance, with
cold step graphs parked as register class `graph_rungs`. **That is the
graph-compatible design, already written down; nothing about it is executed.**

### #286 — the short-term offload register (`model_executor/short_term_offload_register.py`)

**Is:** the vocabulary. One descriptor per offload class, completeness checked
at import, `park_requires_suspend` marking the classes whose park is a
vacate-then-move (#461). Plus the translator's `AudioAssetLedger`, which is a
working park/restore for one tenant's modules (and whose non-persistent-buffer
defect was fixed under #568 in this same branch).

**Verdict for #553:** the register is the right place for the class table, and
the ledger is the right shape for a per-tenant mover — but there is exactly ONE
ledger, wired to the translator. There is no generic "tenant" the register can
raise an event against.

---

## 2. The events #553 needs, against what exists

| event | direction | exists? | what is missing |
|---|---|---|---|
| tenant COLD (release VRAM, keep restorable state) | down | partial | works for the translator (`ledger.park_all`) and for GDN slots (#364). No generic per-tenant mover; no single caller that can address "tenant X" |
| tenant HOT (reacquire VRAM, restore state) | up | partial | same; and nothing arbitrates WHO gets the freed bytes — the dial's participants and the ledger's assets are two registries that do not know about each other |
| serving tenant SHRINKS to make room | down | yes, in-band | `#330` dial, below the captured bound. Above it: re-capture, not built |
| serving tenant GROWS back | up | yes, in-band | same wall |
| pressure-driven relief, ordered | down | planner only | #287's relief rungs need an executor that delegates to the existing features it already names |
| geometry flip | both | planner only | needs the pre-captured per-rung graphs the docstring describes |

The pattern in that table: **the down-direction is largely built and the
up-direction is largely built, but nothing owns the ARBITRATION between
tenants.** Two registries exist (`vram_dial` participants, `#286` asset
classes) and neither can see the other, so "free 4 GiB for the video tenant"
has no addressee.

---

## 3. Cut list

Ordered so that each cut is provable alone, and so that no cut before the last
touches CUDA-graph capture.

**Cut 1 — one registry, or an explicit bridge between the two.**
`vram_dial`'s `DialParticipant` list and `#286`'s asset classes are the same
question asked twice ("what can give bytes back, and how much"). Either fold
them, or give the register a query that returns dial participants as classes.
Hermetically testable; no GPU; no graph impact. This is the cut everything else
addresses.

**Cut 2 — a generic tenant mover, factored out of the translator ledger.**
`AudioAssetLedger` already is one: register modules, park, restore, measure the
restore cost, report parked bytes per device. What makes it translator-specific
is only its wake-rank vocabulary. Factor the mover, keep the vocabulary as
configuration. Provable with the ledger's existing hermetic tests plus a second
fake tenant; no graph impact.

**Cut 3 — make #287's RELIEF rungs execute, and only those.**
Every relief rung names an existing feature. An executor that delegates —
lower the admission cap, arm the spill, shift the DCP ratio — needs no new
mechanism and no capture change, because relief rungs are declared "No KV
layout change, hence handover `none`". This converts the largest block of
`NotImplementedError` into working code without approaching the graph wall.
The `geometry_flip` and `external` stubs stay stubs, and the ladder's own
ordering invariant guarantees relief is tried first anyway.

**Cut 4 — the arbitration policy, on top of cuts 1-3.**
Who yields, how much, in what order, with what hysteresis. This is a decision
layer over the now-addressable registries, and it should reuse the ordering
discipline #287 already enforces rather than invent a second one.

**Cut 5 (LAST, and only with the rig) — cross the graph wall.**
Everything above lives below the captured `store_bound_rows`. Growing past it
needs the per-rung pre-captured graphs #287's docstring describes, with cold
rungs parked as `graph_rungs`. The route to full graph coverage is therefore:
capture every rung at boot (paying the capture cost once), park the cold ones
to RAM through the existing #286 register, and make a rung change a plan flip
at a round boundary rather than a re-capture. Until cut 5 lands, #553 must
state its band explicitly — "elastic within the boot capture bound" — rather
than implying unlimited elasticity.

---

## 4. Instrumentation for whoever executes this

Use `SGLANG_KVSO_TICK_TRACE=1`. It emits one throttled line per spilled session
while any spill is in flight: effective interval, measured `tick_cost`, the
binding headroom ratio and the current host-tail size in tokens. It is pure
logging, changes no control decision, and is a no-op unless the adaptive
regulator is on — so it can be armed on a real boot without altering what is
being measured.

It is the right instrument here because every cut above ultimately shows up as
a change in the spill tick's cost or cadence, which is the latency the user
feels. Note what it is NOT: it is a regulator trace, not a per-phase breakdown
of device time. Any claim about where time goes inside a round needs a
different instrument, and none is being proposed here — a cut list should not
invent measurements it has not checked exist.

Per the benchmark-harness rule, establish the noise floor with an A-vs-A repeat
before comparing any cut against the base.

---

## 5. What this analysis deliberately does not decide

Whether #553 should own the arbitration policy at all, or whether it belongs to
the planner (#363). That is a real fork in the design and it needs the #363
stage feed to be bound first (see `TICKET_363_S8_stage_feed.md`) — an
arbitration policy fed by a table that permanently holds one stage would be a
policy with nothing to choose between, which is the same failure #363/S8 just
documented one level up.

---

## 6. Cut 1 DELIVERED (2026-08-17)

`managers/coresidency_registry.py` + 16 pins in
`test/registered/unit/managers/test_coresidency_registry_553.py`.

**The gap was real and unbuilt.** Confirmed before building: nothing in the
tree imports both `vram_dial` and `short_term_offload_register` — the two
registries genuinely could not see each other, so "free 4 GiB for the video
tenant" had no addressee. §2's diagnosis holds exactly.

**What landed.** One query — `enumerate_reclaim_sources(...)` — returning a
`ReclaimView` of ranked `ReclaimSource`s (dial participants first: returning
VMM pages inside the band is cheaper than parking a class) plus, and this is
the design point, a list of `Unavailable`s **carrying their reason**.

Four properties, each pinned and each with a failure it prevents:

- **Refusal is carried, not filtered.** "Nothing can give bytes" and "three
  things could but none may" must not look alike to a caller.
- **VA stability is ASKED, never re-derived.** The module calls
  `AssetClassDescriptor.va_stability_required(graph_addressed=…)` and passes
  the route flag through, because #468's route-acquired pin flips the same
  class's answer. Re-deriving that rule here would have created a second
  authority for it.
- **No silent partial** (#268): `plan_for` returns `None`, never a short list.
  Unavailable bytes never count toward `can_fund`.
- **No invented numbers.** Neither registry publishes a byte figure, so with
  no probe injected a source is refused BY NAME rather than assumed empty or
  assumed plentiful — both guesses hide.

Mutation: making refusals fall through and `plan_for` return what it has fails
5 of 16.

**Deliberately not done here.** It enumerates and ranks; it does not move,
choose a victim, or fire an actuator. The actuators have very different prices
— a #704a rung change costs a full ~1575 ms arena refill, a GDN slot vacate
does not — and a module that both priced and pulled would hide that. Cuts 2-4
remain as listed in §3.

**Honest limit.** Both byte probes are injected and no caller supplies them
yet, so on a live boot this returns all-unavailable with the "no probe" reason.
That is the intended first state: the bridge exists and says truthfully that
nobody has taught it to measure. Wiring the dial's rows-above-floor and the
register's live extent is the first half of Cut 2, and it needs the live
proof named in §4 rather than a desk number.

## 7. Cut 2 FIRST HALF DELIVERED (2026-08-17) — the two byte probes

The bridge no longer answers all-unavailable when a caller supplies probes.

**Dial side** — `vram_dial.reclaimable_bytes_for(participant, floor_rows)`. A
LIVE read: `full_pool_backed_rows` (the bound eager launches actually pass,
the same quantity `verify_pool_reached_capacity` checks a commit against) times
`_pool_row_nbytes` (the pool's real per-row K+V bytes), minus the floor.

`floor_rows` is **required and not derived there**. There is no per-pool floor
authority in that module — the dial's floor is a card-level NVML measurement
(`_measure_local_floor_bytes`) taken at boot — and inventing a per-pool one
would create a second authority for a number #584 says has exactly one.

**Register side** — `OffloadRegister.reclaimable_bytes(offload_class)`,
mirroring `latency_term_ms`'s lock-and-filter shape so the register answers
about itself rather than the bridge re-deriving its accounting. Resident AND
not hot: parked bytes are already gone (counting them promises the same bytes
twice) and `park()` refuses hot items unconditionally, so hot bytes are
resident but not reclaimable. An unanswerable hotness predicate counts as HOT
— the safe direction is refusing to reclaim, never assuming free to move.

**The distinction the whole cut turns on:** `ProbeUnavailable` vs zero. Zero is
a measurement ("at its floor" / "nothing resident"); a failed probe is the
absence of one. Both probes return `None`/raise rather than 0, and the bridge
turns that into a NAMED refusal. Collapsing them would remove a real source
from an elastic plan while looking like it was considered — the #606
defaulted-measurement defect. Mutation: making a failed probe collapse to 0
fails 3 of 30 pins.

**HONEST LIMIT, unchanged.** The hermetic proof exercises the PLUMBING with
faked dial/register state. No live number's correctness is claimed here; that
remains the window item §4 already names. In particular the dial probe is only
as right as the floor its caller supplies, and no caller supplies one yet.

## 8. Cold-direction policy — SKETCH ONLY, not built

What would consume the bridge on a tenant-idle event. Recorded so the shape is
on record before anyone writes it; every number below is a placeholder.

1. **Event.** Tenant idle timer fires (the translator already has one, #546
   `ledger.park_all`). The event carries a tenant id and nothing else — it does
   not name an actuator.
2. **Query.** `enumerate_reclaim_sources(graph_addressed=<route>)`. If
   `plan_for(want)` returns None, the event ends there: refuse, do not take a
   partial. There is no "free what you can" path, by design (#268).
3. **Order: cheap first.** GDN slot vacate (#364) before the dial, because the
   dial's grow/shrink is in-band VMM work while a slot vacate is a pure
   release. The bridge's `cost_rank` already encodes this ordering; the policy
   consumes it rather than re-deciding.
4. **Then the dial** (#330) grows the KV pool into the freed bytes, staying
   below the captured bound — above it, re-capture is not built and the event
   must stop rather than attempt one.
5. **Debounce, and #704a is the reason.** A rung change costs a full ~1575 ms
   arena refill. An idle/hot flap that crossed a rung boundary twice would pay
   that twice for no net capacity. So: hysteresis on the EVENT (a tenant must
   stay idle for N ticks) and a separate refusal if the plan would cross a rung
   boundary within a cooldown. Without that the policy is a thrash generator
   with good intentions — the same reasoning the KVSO spill/restore pendulum
   already carries as `SpillCooldownRegistry`.

**Stays refused / design-only:** anything requiring a geometry flip. Those
share the #677 arming-floor budget, and the flip's price is a separate
decision with its own gate — see NOTE_677 §5 for why crediting on-demand
capacity against a standing floor is not a small build.

## 9. Cut 3 DELIVERED (2026-08-17) — the events actuate

**What was wired vs priced-only, answered by grep before building:** nothing
imported `coresidency_registry` outside its own module. So a tenant COLD event
**did not actuate** — the bridge priced and no one called it. Same for HOT.

**`managers/coresidency_policy.py`** is the caller, and it introduces **no new
actuator**. Every move is an existing authority's own API:
`vram_dial.apply_budget_request` (replicated grow/shrink, already enforces the
floor, "rejections carry the exact floor arithmetic and change nothing"),
`vram_dial.verify_pool_reached_capacity` (the read-back),
`GdnSlotRuntime.unbind` (#364), and the Cut 1/2 bridge.

### The two directions fail differently, and the code is shaped by that

**COLD — shrink must not strand bytes.** Every source drawn on is recorded
with what it was asked for and what it *reported* giving. A source that was
asked and reported nothing is carried as **stranded** — never counted as zero,
never dropped. Bytes that left one ledger and entered none are the shape that
goes unnoticed for weeks. A delivered zero is an accounting; silence is not,
and the two are distinguishable in the result.

Per #694, totals come only from actuator reports — nothing increments a
"reclaimed" figure from the plan.

An unfundable ask **refuses and actuates nothing**, rather than drawing what
is there. The refusal states that unavailable bytes did not count toward it.

**HOT — grow must not exceed the floor**, and that is *not* enforced here on
purpose: the dial enforces it. This module's duty is to not paper over the
refusal, so a floor refusal is returned unchanged and is **never retried
smaller** — a floor refusal is a statement about the rig, not a negotiation.

**#217 shapes the hot path.** A restore that "came back" was measured at 23%
of target. So a grow is followed by a read-back and the result reports what
was **measured**, never what was requested. A caller asking "is the tenant
warm again" reads `reached_bytes`. An absent read-back leaves it `None` —
"not measured", distinct from "reached nothing" (#606). A read-back that
raises **refuses**, because a tenant must not be reported warm on an
unverified grow.

Mutations: hiding stranding, and reporting the request instead of the
measurement, fail 4 of 16.

### Still priced-only

The policy takes injected `release_fn` / `grow_fn` / `measure_fn`. Binding
them to the live dial and slot runtime is the seam this stops at — it needs a
scheduler and belongs with the live proof, not a desk fake. Live acceptance is
filed in `/spinning/GPU_WINDOWS.md` (a NEW #553 ticket; there was none before
— grepped).

---

## 10. Remainder determination (2026-08-17)

### 10.1 What the delivered "Cut 3" actually is

The commit titled `[#553] Cut 3` delivered the tenant hot/cold EVENT actuation
(`coresidency_policy`). §3's Cut 3 is a different thing: "make #287's RELIEF
rungs execute, and only those". The event layer touches no rung.

Both are real work and neither is wrong; the numbering is. Recorded so the next
reader does not tick off a cut that was never done. The planned Cut 3 is
delivered by §10.3 below.

### 10.2 Which actuation targets the event layer reaches, and in which direction

Verified at code, not from the cut message.

| target | COLD (shrink) | HOT (grow) |
|---|---|---|
| **KV pool** (#330 dial) | reached — `apply_budget_request` shrink | reached — `apply_budget_request` grow + `verify_pool_reached_capacity` read-back |
| **GDN slots** (#364) | reached — `GdnSlotRuntime.unbind` | **NOT reached** — no rebind |
| **Schnitte** (#287/#704 rungs) | **NOT reached** | **NOT reached** |

Two named gaps:

* **G1 — GDN slots are one-directional.** A cold event vacates a slot; a hot
  event grows the KV pool and nothing rebinds the slot. A tenant that went cold
  and came back gets its bytes and not its slot. `hot_event` takes `grow_fn`
  and `measure_fn` and has no third hook. This is a real asymmetry, not an
  oversight of mine to fix silently: rebinding needs the slot runtime's own
  admission rules, and inventing them here would be the second-authority
  defect. **Filed for the cut that owns slot policy.**
* **G2 — the rungs were not reachable at all**, which is what §10.3 fixes.

### 10.3 Planned Cut 3 DELIVERED — the relief rungs can now execute

`managers/relief_rung_executor.py`. Grepping `RELIEF_FEATURES` across the tree
had found exactly two consumers: the ladder that defines it and
`planner/kv_ladder_table.py`, which checks membership. **Nothing mapped a
chosen rung to the actuator of the feature it names**, so the ladder could rank
and ascend and nothing would change -- a counter with no actuator.

`apply_relief_rung(rung, actuators)` delegates to one injected actuator per
feature and reports what that actuator returned, never what the plan intended
(#694). The vocabulary comes from `RELIEF_FEATURES` rather than being restated,
so the executor cannot grow a sixth feature the ladder cannot rank, nor leave a
ranked one unreachable.

Refusals are the design, because the failure being replaced is silence:

* an unknown feature RAISES, naming the known ones;
* a known feature whose actuator is not wired RAISES -- "nothing happened" must
  not be indistinguishable from "it worked", or the ladder ascends believing
  pressure was relieved and picks the next rung against an unchanged state;
* a `geometry_flip` or `external` rung is refused. The plan said "relief rungs,
  **and only those**", and this executor must not become the place a geometry
  change quietly starts.

Mutation-proven: making a missing actuator a no-op, and shrinking the
vocabulary to one feature, each red exactly the test that asserts against it.

### 10.4 Interactions, named

* **#698 BOTH-BLOCKED** (`phase_policy.py:120`, `:1437`). The `admission_cap`
  relief feature throttles inflow. Driven repeatedly by cold events it could
  reach a cap where nothing can run -- exactly that wedge. This executor does
  not clamp, and that is deliberate: the floor belongs to the actuator, the same
  split `hot_event` already uses for the dial's floor ("a floor refusal is a
  statement about the rig, not a negotiation"). **The obligation is on whoever
  wires the admission actuator**, and it is stated here so it is not discovered
  by hitting it.
* **#684 recover() clamp** (`kv_backing_relief.py:1809`, "CLAMP TO WHAT THE
  ACTUATOR CAN ACCEPT, NOT TO WHAT WE [want]"). The precedent this module's
  reporting follows: the result carries the actuator's answer, so a caller
  cannot read back its own request.
* **#305 multi-model binding gap** (`registry/rungs.py:8`, the registry "does
  not yet" carry what `DESIGN_305` asks). Co-residency is multi-tenant by
  definition, so **Cut 4 (arbitration) depends on that binding**: deciding who
  yields presupposes knowing which tenant a rung belongs to. Cut 4 is therefore
  gated on #305, not merely on cuts 1-3.

### 10.5 Remainder after this cut

| item | state |
|---|---|
| Cut 2 second half — generic tenant mover | **DELIVERED 2026-08-17**, see §11 |
| Cut 4 — arbitration policy | gated on **#305 binding** (§10.4) as well as cuts 1-3 |
| Cut 5 — cross the graph wall | **rig-gated by its own text** ("LAST, and only with the rig") |
| G1 — GDN slot rebind on hot | filed, needs the slot runtime's admission rules |
| Binding `release_fn`/`grow_fn`/`measure_fn` to the live dial | **window item**, already filed by Cut 3 |

**Window items, named rather than attempted:** the live binding of the event
layer's injected callables, and Cut 5. Anything that resizes a live pool needs a
scheduler and a boot; §9 already filed the live acceptance.


---

## 11. Cut 2 SECOND HALF DELIVERED (2026-08-17) — `managers/tenant_mover.py`

### The gating column was right here, and I checked rather than trusted it

§10 caught this table lying about NUMBERING, so its desk/window column earned
the same skepticism. Checked: `test/registered/translator/test_ledger.py` runs
**22 passed** under `CUDA_VISIBLE_DEVICES=""`. The ledger's park/restore is
already exercised hermetically with CPU tensors, so a mover over the same
protocol is desk-provable and the marking holds. Had those tests needed a
device, this would have been a window item and the table would have been wrong
twice.

### Not a second ledger

`AudioAssetLedger` already parks and restores, and its `ParkRoute` protocol
(`device` / `park` / `restore` / `size_bytes`) is already generic. What was
missing was never the parking — §2's tenant-COLD row says it: "**no generic
per-tenant mover; no single caller that can address 'tenant X'**". So
`TenantMover` registers TENANTS over that same protocol and reimplements
nothing. A parallel asset ledger would have been the two-authorities defect the
#553 bridge exists to reconcile.

### The vocabulary is per tenant, which is the whole factoring

The translator's ASR -> talker -> codec need order is one tenant's physics, not
a property of moving tenants. `register(tenant, routes, ranks=...)` takes it as
configuration, and a test asserts a second tenant restores in `y, x` — an order
that is nonsense for the translator and correct for that tenant. An unranked
route sorts last (`UNRANKED`), because an unlisted asset is unknown, not urgent.

### Stranding, again, and deliberately the same shape

Released bytes come from what a route RETURNED (#694). A route asked to park
that reports nothing is STRANDED, never zero: `park()` returning `None`, and a
route that raises, both land there. A route reporting `0` is an accounting
("nothing to give"); silence is the absence of one. `release_fn()` hands
`cold_event` a `None` in exactly that case, which is how that layer already
reads "did not report" — so a stranded tenant surfaces at the policy layer
instead of being counted as a delivered zero.

`parked_bytes_by_device()` omits a device with nothing parked rather than
reporting it as 0 (#606): an all-zeros map reads as "measured, empty".

### Refusal

An unknown tenant RAISES and names who is registered. "No such tenant" and
"that tenant had nothing to give" are different answers, and a cold event that
cannot tell them apart keeps asking the wrong one.

Mutation-proven: collapsing stranding to zero reds the two stranding pins;
ignoring the supplied ranks reds all three vocabulary pins.

### A defect in my own test, recorded

The first `_Route` stub used `parked_bytes=None` for "not specified", so
"reports nothing" was inexpressible and two stranding tests failed against
correct code. A distinct `REPORTS_NOTHING` sentinel fixes it. A stub whose
default and whose absence-case share a value cannot test an absence rule.

### #553's desk surface is now exhausted

| item | state |
|---|---|
| Cuts 1, 2 (both halves), planned cut 3 | delivered |
| Cut 4 — arbitration | gated on **#305** multi-model binding (§10.4) |
| Cut 5 — cross the graph wall | **rig-gated by its own text** |
| G1 — GDN slot rebind on hot | filed; needs the slot runtime's admission rules |
| live binding of `release_fn`/`grow_fn`/`measure_fn` | **window item** (§9) |
| correctness of any live number | **window item** (§7 honest limit) |

Nothing desk-fundable remains.
