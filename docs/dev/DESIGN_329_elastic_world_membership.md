# DESIGN #329 — elastic world membership

Hotplug of cards and hosts by world RE-FORMATION with preserved state.
Design and task placement only; no implementation. This file discharges the
feature-analysis-file duty for #329.

The question: a serving world (a TP/DCP group, possibly spanning rigs) gains
or loses a member — a card comes online, a host joins over RDMA, a card is
withdrawn for another tenant — **without losing sessions**.

Nearly every piece of machinery this needs already exists. The design's job is
to compose them, name the boundary conditions, and be honest about what is
missing.

---

## 1. The constraint that shapes everything

**NCCL communicators cannot shrink or grow in place.** A world with a
different member set is a different communicator, full stop. So "elastic
membership" cannot mean live membership change; it means

> **checkpoint → flip → restore at a quiesce boundary.**

That is the same shape as #364's between-tick executor and #309's runtime
draft flip, one tier up: instead of mutating a pool between ticks, we tear
down and rebuild the process group between ticks. Everything below follows
from accepting that, and any proposal that tries to add a rank to a live
communicator should be rejected on sight.

Second constraint, from the standing rules: **admitting a member is a PLANNER
decision, not an availability reflex.** The slowest rank sets the pace
(langsamster-Rang-Taktgeber); a weak card joining a strong group can make the
world slower than it was. The trigger may come from availability, but the
verdict comes from the cost library (#348b/#359) — and it may be "no".

---

## 2. State machine

```
  STABLE ──(membership event)──► PROPOSED ──(planner says yes)──► QUIESCE
     ▲                               │                              │
     │                          (planner says no)                   ▼
     └──────────────────────────────┘                          SNAPSHOT
     ▲                                                              │
     │                                                              ▼
  RESUME ◄──── RESTORE ◄──── RE-FORM (new communicators) ◄──────────┘
     │                          │
     │                    (member lost here)
     └──────────────────────────┴──► ROLLBACK ──► STABLE (old membership)
```

Five lines, one per phase:

1. **QUIESCE** — stop admitting, drain in-flight forwards to a tick boundary,
   park the drafter; no collective may be in flight when the group dies.
2. **SNAPSHOT** — serialize what must survive: KV (per-session), GDN state
   (resident, never evicted to the radix route), scheduler session table.
3. **RE-FORM** — destroy the old communicators, build the new ones over the
   new member set; this is the only irreversible step and it is bounded.
4. **RESTORE** — reshard and rehydrate into the new geometry (#297 for KV,
   #261's handover translation for the weight/geometry change).
5. **RESUME** — recapture graphs for the new shapes, unpark the drafter,
   re-open admission.

**ROLLBACK** exists only between QUIESCE and RE-FORM. Once the old
communicators are destroyed there is nothing to roll back TO, which is why
§5 treats a member lost during RE-FORM as its own failure class.

---

## 3. What moves at which tier

| tier | what happens | machinery | why not something cheaper |
| --- | --- | --- | --- |
| **weights** | re-shard to the new rank count; on a joining host, load from local disk | #261 handover (round-trip byte-identical), #89 hibernate for the local load | weights are the big object; moving them over the wire when a local copy exists is the mistake to avoid |
| **KV** | reshard to the new token/owner geometry | #297 (`kv_reshard.py`, reshard vectors already a first-class flag) | the only tier with a built delta-move path |
| **GDN / mamba state** | **stays resident, moves as an opaque blob** | #364 `export_state_blob` / `import_state_blob` | HARD RULE: recurrent state is positional and not prefix-shareable; the radix route would mean a silent full re-prefill (#212) |
| **CUDA graphs** | discarded and recaptured | — | shapes change with the rank count; a captured graph addresses the old geometry |
| **drafter** | parked, then rebuilt with the target | #309 runtime draft flip | draft KV has its own geometry and must follow the target's |
| **scheduler session table** | preserved in-process (no card involved) | — | the sessions are the thing we are protecting |

The asymmetry worth naming: **weights are re-derivable, KV and GDN are not.**
A joining host can load weights from its own disk in parallel with everything
else; the session state is the only payload that must actually survive the
flip, and it is the smaller one.

---

## 4. Timing budget from the measured record

Every number below is a MEASURED figure already on record in this repo, not
an estimate. The point of the table is to decide whether the whole thing is
worth building, before building it.

| phase | budget | source |
| --- | --- | --- |
| QUIESCE (drain to tick boundary) | ms-class at bs=1..16 | one decode tick; the #364 between-tick boundary is the same seam |
| SNAPSHOT KV | dominated by capacity, not rate | #297: the design deliberately chose an idle boundary so the delta is structurally EMPTY (`DESIGN_297_kv_resharding.md:83-87`) |
| RE-FORM (communicator rebuild) | **unmeasured — the gap** | see §7 |
| RESTORE weights (local load) | **8-14 s** for uneven TP=3, down from 50 s | `DESIGN_201:1635` (#89 hibernate resume) |
| RESTORE KV delta move | **target < 1 s** | `DESIGN_297_kv_resharding.md:83` |
| RESUME graph recapture + repack | **order 3-6 s** incl. quantized repack | `ANALYSE_363:112` |

**Read the total honestly: 12-20 s of silence for a membership change**, of
which the weight restore and the graph recapture dominate and the state
preservation is nearly free. That is a maintenance-window number, not a
transparent one. It is acceptable for "a tenant releases a card" and for "a
host joins the world"; it is NOT acceptable as a reflex to a transient blip,
which is why §5's liveness distinction is load-bearing rather than decorative.

---

## 5. Failure semantics

**Detecting a lost member vs a slow one** is #312's bounded-peer-liveness
question, and it must be answered BEFORE the state machine starts: a
membership event triggered by a slow peer costs 12-20 s of silence and gains
nothing. The bound belongs on the collective, not on a heartbeat above it —
a peer that is slow inside an all-reduce is not visible to an application-level
ping. (`liveness/` in this tree is the SESSION-liveness layer and says so
itself: "Not this module's layer: #312's bounded peer liveness inside
collectives", `liveness/__init__.py:16`.)

**Member lost DURING re-formation** is the one genuinely new failure class,
and it needs an explicit answer rather than a retry loop:

* Lost between QUIESCE and RE-FORM → **ROLLBACK** to the old membership. The
  old communicators still exist; nothing has been destroyed. This is why the
  design destroys late rather than early.
* Lost during RE-FORM (old group gone, new group not up) → **the world is
  down**. There is no state to roll back to and no group to serve on. The
  honest behaviour is to fail the world loudly and let the supervisor restart
  it from the SNAPSHOT, which is exactly why the snapshot must be durable
  (disk-parked, #89's disk park with the NVML-UUID lock) and not merely
  in-memory on the ranks that are re-forming.
* Lost after RESTORE, before RESUME → re-enter the state machine with the
  smaller member set; the snapshot is still valid.

**Rank-local before collective** (#94) applies to every membership decision:
each rank must decide locally whether it can proceed, and only then enter the
agreement collective. A rank that discovers its own problem after the others
have entered a barrier produces the hang this rule exists to prevent — and a
membership change is a barrier-dense operation, so the rule binds harder here
than anywhere else.

**Card identity** across the flip is #331's UUID/PCI-BDF map, already done:
a card that leaves and returns must be recognised as the same card, and index
order is not identity (NVML and torch enumeration diverge, and both can shift
across a driver state change).

---

## 6. NORDSTERN fit (a host joins)

The cross-rig case is the same state machine with a slower RE-FORM and a
different transport. Two things it changes:

* **Transport ladder**: L0 gloo-TCP first, then UCX, then hierarchical. A
  host join should be proven on L0 before anyone tries it on RDMA — the state
  machine is the risk, not the wire, and debugging both at once is how a
  window gets lost.
* **Weight locality**: the joining host loads its own shard from its own disk
  (#89), so the wire carries only the KV/GDN payload for the sessions being
  preserved. This is the design's main argument for why a host join is even
  plausible: the big object never crosses the link.

#201 (TPxPPxTP cross-rig worlds) already establishes that such a world can be
COMPOSED. #329 is the claim that its membership can CHANGE — strictly the
harder statement, and it should not be attempted until a static cross-rig
world runs.

---

## 7. The gap: RE-FORM is unmeasured

Everything in the timing table has a measured source except the one step that
is new. Nobody in this tree has torn down and rebuilt NCCL communicators in a
live process and measured it. It could be 200 ms; it could be seconds; on a
cross-rig world it could fail in ways a single-node test never shows.

**Therefore the first cut is a measurement, not a feature** — see #8, cut 1.
If communicator rebuild in-process turns out to be unreliable (a known-hostile
area: NCCL cleanup, lingering handles, CUDA context state), the honest
fallback is process-level: snapshot to disk, exit, restart with the new
membership, restore. That is slower and it is still elastic; it just is not
in-process. Naming that fallback now keeps cut 1 from becoming a blocker.

---

## 8. Task cuts, falsifier-first

| # | cut | effort | falsifier / gate |
| --- | --- | --- | --- |
| 1 | **Measure in-process communicator teardown+rebuild.** No sessions, no state, two ranks: destroy the group, build a new one with a different member set, do a collective. Repeat 20x for stability. | S, one window | If it is unreliable or > ~2 s, take the process-level fallback (§7) and re-plan. Everything else waits on this. |
| 2 | **Quiesce + snapshot + restore with NO membership change.** Same member set, full round trip through the state machine. | M | Round-trip byte-identity of KV and GDN state (#261's standard; #364's blob round-trip test is the unit-level precedent). If state does not survive an identity flip, membership is irrelevant. |
| 3 | **Shrink by one** (a card is withdrawn), single node, sessions live across it. | M | Sessions complete with coherent output; the #328 chain-quality gate on the output across the flip; measured silence within the §4 budget. |
| 4 | **Grow by one**, single node. | M | Same gates, plus: the planner must be able to say NO (a weak member that would slow the world). |
| 5 | **Planner admission rule** — cost-library verdict on whether a candidate member improves the world. | M | Must reject a member that lowers aggregate throughput; a test with a deliberately weak candidate. |
| 6 | **Host join over L0 gloo**, cross-rig. | L | Only after a static cross-rig world runs (#201). |
| 7 | RDMA transport for the join payload. | L | Only after 6. |

Cuts 1 and 2 are cheap and answer the two questions that decide the rest.
Cut 2 is worth building even if #329 never ships: a snapshot/restore round
trip at the world level is the missing half of #89's disk park.

---

## 9. Not before X

* **Not before a static cross-rig world runs** (#201) for cuts 6-7. Changing
  the membership of a world that cannot yet be composed is out of order.
* **Not before #312's bounded peer liveness exists.** Without it the trigger
  cannot distinguish a dead member from a slow one, and a 12-20 s
  re-formation fired by a slow peer is a self-inflicted outage.
* **Not before the regime controller (#363) is observe-only-proven.** #363 is
  the eventual trigger source; wiring an automatic trigger to an unproven
  classifier would let a misclassification take the world down. Manual
  trigger (an operator or an API call) first, automatic much later.
* **Not as a response to transient failure.** This is a planned-maintenance
  and capacity mechanism. Fault tolerance is a different feature with a
  different budget, and conflating them produces a system that reforms the
  world every time a link hiccups.

---

## 10. Honest summary

The state preservation is nearly free — the fork already has every tier's
machinery and the measured numbers say the payload that must survive is the
small one. The cost is dominated by weight restore (8-14 s) and graph
recapture (3-6 s), both irreducible with today's parts, giving a 12-20 s
maintenance-window silence rather than a transparent change.

The single unknown is in-process communicator rebuild, and it is a one-window
measurement. Do cut 1, then cut 2, then re-decide. If cut 1 says NCCL will not
rebuild reliably in-process, the feature still exists via the process-level
fallback, and that is worth knowing before anyone designs around the
in-process assumption.
