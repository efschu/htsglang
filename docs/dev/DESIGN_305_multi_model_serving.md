# DESIGN #305 — multi-model serving and the residency ladder

N models registered, M hot (configurable), an idle default set, and a
four-rung residency ladder. **Design and task cuts only, no implementation**
— the GPU-proof backlog drains first. Discharges the feature-analysis-file
duty for #305.

This is the #274 Aufsatz end state, so the design's job is **composition over
machinery that already exists**, with the gaps named rather than papered over.

---

## 1. The ladder

Costs are the measured record where one exists, and marked as a gap where
none does. "Entry" = getting to this rung from the rung below (colder);
"exit" = leaving it upward toward HOT.

| rung | resident | idle VRAM | idle POWER | promote to HOT | first-request latency class |
| --- | --- | --- | --- | --- | --- |
| **HOT** | weights + KV pools + graphs + drafter | full | full (see §2) | — | serving latency; nothing to pay |
| **TEIL-HOT** | weights + graphs; KV pools reduced via the #330 dial, offloadable posts parked (#102/#286) | weights only | full board idle | **< 1 s** — KV delta move (#297) | one dial re-raise; a long-context request may wait for pool growth |
| **WARM** | suspend-to-RAM: memory-saver tags paused, VA reserved, content in host RAM (#89) | ~0 (VA only) | idle floor | **3-6 s** graph recapture, plus tag resume | seconds; a request arrives to a rebuild |
| **COLD** | disk image only (#89 disk park, NVML-UUID lock) | 0 | 0 | **8-14 s** weight load (uneven TP=3, from 50 s) + 3-6 s recapture | ~12-20 s; effectively a boot |
| **(registered)** | nothing; a name and a config in the registry | 0 | 0 | full cold start | model is *known*, not *ready* |

Measured sources: #89 hibernate resume 8-14 s at uneven TP=3
(`DESIGN_201:1635`); #297 KV delta target < 1 s
(`DESIGN_297_kv_resharding.md:83`); graph recapture / weight flip order 3-6 s
(`ANALYSE_363:112`).

**The fifth row is deliberate.** "Registered but not staged" must be a
distinct state or the API cannot answer "do you know this model" separately
from "can you serve it now" — and those are different questions for a client.

---

## 2. Idle power is a per-model cost, not a rounding error

Freshly measured on this rig during the #350 window, holding contexts and
serving nothing:

    5090   45.9 W idle
    3080   57.2 W idle
    3080  105.4 W idle
    ------------------
    total ~208 W just to keep a TP=3 world resident

That is the recurring cost of one HOT model on this rig, and it is why the
demotion policy in §3 must be allowed to consult the **energy** objective
(#350) and not only throughput: a model that is HOT and unused is burning
~208 W to save a 12-20 s cold start. Whether that trade is worth it depends
on the request arrival rate, which is exactly the sort of judgement the
planner exists for.

(The two nominally identical 3080s differ ~2x in idle draw — 57.2 vs 105.4 W.
That anomaly is its own ticket; it is noted here because a per-model idle
budget computed from a nameplate figure would be wrong by that much.)

---

## 3. Admission and eviction policy

**The rule: promotion and demotion are PLANNER decisions on the #348b cost
library under the selected #350 objective, never an availability reflex.**

A request for a COLD model does not automatically demote something. The
planner answers a bounded question:

* what does promoting this model cost (the ladder entry cost above),
* what does demoting each candidate cost (its exit cost, plus the sessions it
  strands, §4),
* and under the active objective — throughput or energy — is the swap worth
  it at the observed arrival rate?

The answer may be **no**: serve the request from a colder rung and pay the
latency, or refuse with a named reason. "A request arrived" is not a
justification for evicting a model that is actively serving.

**Fairness during demotion is #273's rule, unchanged.** In-flight sessions of
the demoted model are FCFS-protected: the oldest running session is not
sacrificed to make room, retraction happens before abortion, and a sole
surviving session is retracted rather than 500'd. Multi-model demotion must
not become a back door around a rule the single-model path already enforces.

**Pinning overrides the planner in one direction only.** `pin` prevents
demotion; it cannot force promotion past the VRAM ledger. A pin that cannot
be honoured is an error at pin time, not a silent downgrade later.

---

## 4. Session semantics under demotion

A demoted model's open sessions are the hard part, and the answer differs by
rung transition:

| transition | what happens to sessions |
| --- | --- |
| HOT -> TEIL-HOT | sessions survive. KV shrinks via the dial; sessions beyond the reduced pool spill to host through **kv-session-offload** (FCFS victim order, #236/#242) |
| TEIL-HOT -> WARM | sessions must be **spilled or drained** first. A suspended model has no device pool to decode into |
| WARM -> COLD | sessions are already off-device; their blobs either persist with the disk image or the sessions end |
| any -> HOT | restore in FCFS order; a session's GDN state comes back before its next decode tick |

**The GDN hard rule holds at every rung**: recurrent state travels as an
opaque session blob (#364 `export_state_blob` / `import_state_blob`), never
through the HiCache radix route (#212) — a positional state is not
prefix-shareable, and evicting it as cache is a silent full re-prefill.

### Which transition instrument, per rung

This is the question the brief asks to settle explicitly, and the answer is
that the two instruments are for different things:

* **#309's quiesce/state machine — for transitions WITHIN one world
  geometry.** HOT <-> TEIL-HOT and TEIL-HOT <-> WARM change *how much* is
  resident; the member set, the rank count and the communicators do not move.
  Drain to a tick boundary, mutate, resume. No group is torn down.
* **#329's five-phase machine — for transitions that CHANGE THE WORLD.**
  Only needed when promoting a model requires a different rank count or a
  different card set (a 2-card model displacing a 3-card one). That is a
  membership change, communicators must be rebuilt, and the full
  QUIESCE/SNAPSHOT/RE-FORM/RESTORE/RESUME machine with its rollback boundary
  applies.

**Design consequence worth stating: keep every registered model on the same
world geometry if you possibly can.** Then #305 never needs #329, and the
whole feature is the cheaper instrument. A model that needs a different
geometry is not merely another entry in the ladder — it is a world
re-formation with a 12-20 s floor, and it should be labelled as such in the
registry rather than discovered at request time.

---

## 5. API surface

Client-compat rule: standard protocols, so an unmodified OpenAI client works.

* **Routing**: the existing `model` field in the request routes to the
  registered model. No new parameter, no custom header — this is the
  behaviour clients already expect from a multi-model endpoint.
* **`/v1/models`** lists every REGISTERED model (all five rows of §1), each
  with its rung and a coarse readiness hint. A client that ignores the extra
  field sees a normal model list; one that reads it can avoid a 20 s first
  token. Reporting only HOT models would be the wrong choice: it makes a
  registered model invisible and gives the client no way to warm it.
* **Admin surface** (not OpenAI-shaped, so it lives under the fork's own
  namespace): `register`, `unregister`, `pin`/`unpin`, `promote`/`demote` as
  *requests* the planner may refuse with a reason, and a status read.
* **Refusals are named.** A promote that the ledger cannot fund returns the
  ledger arithmetic, not a generic error — the same discipline as the #350
  energy refusal and the #364 cap refusal.

---

## 6. Multi-tenant interaction

The idle workbench (#347) and the video lanes (#333/#341) are **tenants
competing for the same VRAM**, and they already have a ledger:
`CapacityLedger` (`model_executor/offload_movement.py:514`) plus the #330
dial's boot capacity plan.

**One ledger. No second accounting.** The residency ladder must express a
model's rung as posts in that ledger, exactly as the offload register
expresses parked items — otherwise two allocators will each believe they own
the same bytes, which is the failure the ledger was introduced to prevent.
Concretely: a model at TEIL-HOT has released KV bytes into the ledger, and an
idle-workbench tenant may legitimately take them; promoting that model back to
HOT is then a ledger negotiation, not an assumption.

This also gives the eviction policy its natural extension: an idle tenant is a
demotion candidate like any other, and the planner compares them on one axis
because there is one ledger.

---

## 7. Task cuts, effort/yield pairs

No thresholds — effort against yield, judged as a ratio.

| # | cut | effort | yield |
| --- | --- | --- | --- |
| 1 | **Registry + `/v1/models` with rungs, single HOT model.** No ladder movement; every registered model is either the one HOT model or "registered". | S | Establishes the API and the vocabulary with no runtime risk. A client can already discover models. |
| 2 | **HOT <-> TEIL-HOT via the #330 dial**, #309 quiesce as the instrument, sessions surviving through kv-session-offload. | M | The highest-value rung pair: sub-second promotion, and it is where the VRAM actually comes from. Needs no new mechanism, only composition. |
| 3 | **WARM via #89 suspend**, with the ledger posts released. | M | Turns "second model costs 12-20 s" into "3-6 s". The measured jump from 50 s to 8-14 s already happened; this spends it. |
| 4 | **Planner promotion/demotion policy** on the cost library under both objectives, with #273 fairness during demotion. | M-L | This is where the feature becomes autonomous rather than manual. Also where it can do damage, hence the not-before gates. |
| 5 | **COLD/disk rung + registry persistence.** | M | Completes the ladder; mostly bookkeeping once 3 exists. |
| 6 | Cross-geometry models via #329. | L | Only if a registered model genuinely needs a different world. Avoidable by design (§4). |

Cuts 1-3 are composition and are worth doing on their own merits. Cut 4 is the
one that needs judgement and evidence.

---

## 8. Not before X

* **Not before the GPU-proof backlog drains.** Stated in the brief and
  correct: this design composes machinery whose own proofs are still queued
  (#329 cut 1 has not measured communicator rebuild; the #350 objective has
  just been shown to be plan-neutral at one operating point). Building the
  composition on unproven parts multiplies the unknowns.
* **Not before #309's quiesce boundary is exercised** for a within-geometry
  mutation with live sessions. Cut 2 depends on it entirely.
* **Cut 4 not before the planner's objective has been shown to discriminate**
  on a real operating point. The #350 window found the objective wired but
  plan-neutral there; an eviction policy driven by an objective that cannot
  yet be shown to prefer anything would be a coin flip with a cost model
  attached.
* **Not as a fault-tolerance mechanism.** Same separation as #329: this is a
  capacity and latency feature. A model that crashes is a different problem
  with a different budget.

---

## 9. Summary

The ladder maps cleanly onto machinery that exists — the #330 dial for
TEIL-HOT, #89 for WARM and COLD, #309 for within-geometry transitions, #329
only when the world itself changes, kv-session-offload plus the GDN blob rule
for sessions, and one shared `CapacityLedger` for tenants. The measured entry
costs (< 1 s / 3-6 s / 8-14 s) make the rungs genuinely distinct rather than
cosmetic.

Two things are genuinely new and should be treated as such: the **policy**
(cut 4), which needs an objective that has been shown to discriminate, and the
**idle-power accounting** (§2), which on this rig is ~208 W per resident world
and is the strongest argument for the ladder existing at all.
