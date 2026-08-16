# MERGE NOTES — branch `fix/602-fill-side`

Base `7936bc4850`. Nine commits, `890e0b1f35` .. `5301b94019`.
Everything below is **desk-only**: hermetic tests (`CUDA_VISIBLE_DEVICES=""`),
no boot, no live flag changed, nothing deployed.

---

## 1. The chain, in order

| # | commit | what it is |
|---|---|---|
| 1 | `890e0b1f35` | **#602** KV-floor PP cut solver. `world_kv_floor`, `stage_kv_capacities`, `solve_pp_cut_for_kv_floor` — a maximin DP over `min_r(capacity_r)`, added because `solve_pp_cut`'s existing objective is makespan and headroom is not capacity. Names the draft-head gap it could not yet price. |
| 2 | `d9ed47b895` | **#602** Draft runner calibrated from the flight recorder. Net residency = `weights_draft − inter_runner_gap` (2906/2250/2250 MiB); refuse-on-absence; **and** the seam turned out to flip the cut, so it is solved as a fixed point rather than left documented. |
| 3 | `7f2f8c6aa1` | **#602** Fixed overhead + transient read from the recorder. The transient is reported as *observed*, never as a worst case (law 31); the refuse-on-absence guard fired on a real in-flight boot while this was written. |
| 4 | `f93be643cc` | **#602** Weights priced from the checkpoint safetensors headers. Found a **vision tower (879 MiB) and MTP head (405 MiB) resident on every stage**. Identity closes to −53/+8/+39 MiB and the model now reproduces the live floor within 1.8 %. **Reverses the recommendation** (see §3). |
| 5 | `9f5877cb62` | **#685** Rank-0 seam slope decomposed: it is **one received attention layer**, not a pathology. Mechanism at `phase_flip_runtime.py:4487-4497`; frozen constant at `phase_flip_seam_reserve.py:217`. |
| 6 | `c41645c8c9` | **#685** `managers/seam_slope.py` — the derivation as a dependency-free module. **Cherry-pickable standalone.** |
| 7 | `5a1e087c37` | **#685** De-duplication: the planner's copy delegates to that module. Kept separate so #6 cherry-picks alone. |
| 8 | `ce6035884d` | **#691** Per-rank prefill timer admitted under PP (per-stage local pairing) with a divergence guard that refuses rather than mispairs. **Cherry-pickable standalone.** |
| 9 | `5301b94019` | **#685** Cold-boot seam slope derived and announced. **Stops short of the sizing fix** — see §4. |

## 2. Cherry-pickable standalone

* **`c41645c8c9`** — 2 new files, modifies nothing. For the `fundable_width`
  correction: ranks that receive **zero** layers are being taxed for bytes that
  never move (ranks 1 and 2 on the shipping cut).
* **`ce6035884d`** — touches `metrics_reporter.py` + its new test only. Gives
  per-stage prefill compute-vs-wait ms/round in a PP=3 boot with no flag.

Neither depends on the #602 solver files. `5a1e087c37` is the only commit that
couples the two areas and a cherry-pick does not need it.

## 3. Retracted, on the record — TWICE

**Revision 1** recommended `31,16,17` at **+227095 tokens (+36.3 %)**.
Withdrawn: it came from #485 reference-bench weights, and the real linear layer
is 476 MiB, so moving layers costs far more than the model believed.

**Revision 2** recommended `29,19,16` at **+17235 tokens (+3.6 %)**.
**Also withdrawn**, after F4-r4's census boots. Two separate defects:

* The ±5 % gate compared a **corridor-safe floor** (worst load transient
  funded) against a **measured pool** (sizer, no transient funded). In his
  regime the worst load state is a SEAM on every rank, so the two are ~29 %
  apart and the comparison read -23.3 %. He refused the arm; he was right.
  Fixed in `90e37f0510` by separating `world_predicted_pool` from
  `world_corridor_safe_floor` — validated at **-0.5 %** against his measured
  471303.
* Re-solved on his terms, the **incumbent `28,20,16` is the global optimum**
  over all 1953 contiguous cuts; `29,19,16` is 6.3 % *worse*. There is no cut
  to arm on that regime. The +3.6 % was a property of my boot's terms, not a
  portable result.

**Standing rule now in the ticket: re-solve on the regime being booted, and
check `world_predicted_pool` against its measured pool, before any arm.**

## 4. Open decision points — two, both outside my lane

### 4a. R′ semantics for the cold seam (blocks the cold-overshoot fix)

`5301b94019` derives and announces the cold per-token seam slope but the
reserve stays **inactive**, so cold boots still size floor-only.
`SeamReserve.active` requires `id_space > 0` — the token count a record was
*measured* at — and the downstream solve anchors on it with `have_bytes`.
A derived slope has no measurement point; setting those would fabricate the
anchor.

Consuming it needs an anchor-free branch. `solve_pool_tokens` is exactly that
form **but has no live caller anywhere in the tree**, so which budget it is
solved against is a boot-path design decision, not wiring.
**Owner: F4-r4.** One decision; the slope is already there and tested.

### 4b. #602 metal arm — NO ARM on the censused regime

Not merely deferred: its premise is absent there (§3). The incumbent cut is
already optimal, so there is nothing to measure.
`TICKET_602_METAL.md` stays on file with its verdict; it carries a hard ±5 %
floor gate (earned from the 1.8 % demonstrated error) and a **non-concurrency
constraint against F4-r4's #689 window** — re-cutting moves per-stage arena and
seam geometry and invalidates the cached seam records.

## 5. Test state at the tip

* `test/registered/unit/planner` — 2561 passed, 2 failed.
* `test/registered/unit/managers` — 2052 passed, 4 failed.

All six failures are **pre-existing on the `7936bc4850` base**, verified by
reverting. They are not from this branch:

* `test_rejected_evidence_pins.py` ×2 (planner)
* `test_first_chunk_dynamic_chunking.py` ×2 (managers)
* `test_collective_family_siblings_610.py::PrefillAdmissionBudgetTest` ×2

A seventh (`test_load_snapshot_backends.py::TestZmqRoundTrip`) appears only in
batch runs and passes 22/22 in isolation — flaky.

### 5a. Merge-queue evidence: does #681 fix the `610` pair? **NO.**

`0274bed857` (`fix/680-spec-candidates-dtype`) was cherry-picked into a scratch
copy of this tree and the pair re-run hermetically. Both still fail, unchanged:

```
FAILED ...PrefillAdmissionBudgetTest::test_admission_budget_is_rank_uniform
FAILED ...PrefillAdmissionBudgetTest::test_budget_state_agrees_across_ranks
AttributeError("'BudgetHarness' object has no attribute '_local_mamba_avail'")
```

**Diagnosis: a stale test harness, not a product defect.**
`Scheduler._local_mamba_avail` is real (`scheduler.py:4257`, called from
`_update_uniform_pool_budget` at `:4067`); the `BudgetHarness` stand-in
(`test_collective_family_siblings_610.py:463`) never binds it, so the harness
has drifted behind the production budget-reduce path. Nothing to do with
#681's `fundable_extend_floor`. Scratch dropped.

## 6. Related work not on this branch

`0274bed857` (#681, admission ceiling) is on `fix/680-spec-candidates-dtype`
and still awaits its metal window. Per the metal ticket it can share a window
with the #602 arm: it is an admission-path arm and moves no layers.
