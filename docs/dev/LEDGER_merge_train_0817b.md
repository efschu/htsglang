# LEDGER — merge train 2 (desk), 2026-08-17

Desk assembly only. **No serving was touched, `integration/r2` was not moved,
and nothing was booted.** The deliverable is the branch `train/0817-desk` plus
this ledger; boot validation and any fast-forward belong to F4-r4.

* BASE: `a157bf1889` (`origin/integration/r2`, 2026-08-15)
* HEAD: `c9964f7e4d`

---

## 0. Prior-art gate: train 1 has NOT landed

The gate asked me to read `LEDGER_merge_train_0817.md` for what train 1 took.
**That file does not exist** — not in any branch, not in any commit
(`git log --all -- docs/dev/LEDGER_merge_train_0817.md` is empty).

Checked the substance instead, which is the authoritative question: none of
train 1's content is in the base. `fix/621-collective-invariant-pins`,
`fix/699-progress-clock-wiring`, `fix/673-lockstep-sentinel-stop`,
`feat/677-park-wiring`, `feat/706-phase-uniform-hicache-keys`,
`reconcile/cluster-b-seam-model`, `fix/673-teardown-stack` and
`fix/728-max-bytes-uniform` are each NOT ancestors of `integration/r2`.

Consequence, and it shapes everything below: this is not "train 2 on top of
train 1". Every candidate sits 159-315 commits ahead of its merge-base with
the base, on the shared feat/706-era lineage, so the FIRST merge carries the
bulk and the rest are small deltas. This train therefore contains train 1's
content too, by ancestry.

## 1. Merged, in dependency order

| # | branch | result |
| --- | --- | --- |
| 1 | `docs/catalog-retro-sweep` | clean — brings §19 and the bulk lineage |
| 2 | `docs/305-determination` | clean |
| 3 | `docs/349-determination` | clean (carries `bc9c158a17`) |
| 4 | `docs/138-determination` | already an ancestor, no-op |
| 5 | `feat/107-determination` | clean |
| 6 | `feat/121-determination` | clean |
| 7 | `feat/224-determination` | clean |
| 8 | `feat/309-determination` | clean |
| 9 | `feat/447-remainder` | clean |
| 10 | `fix/584-rank-budget-contract` | clean (auto-merged `server_args.py`) |
| 11 | `fix/server-args-mutation-ratchet-rearm` | clean |
| 12 | `fix/728-max-bytes-uniform` | clean |
| 13 | `reconcile/cluster-b-seam-model` | clean |
| 14 | `fix/673-teardown-stack` | 1 doc conflict, resolved by union (§2) |
| 15 | `fix/config-mutation-debt-hygiene` | clean — the anticipated follow-on debt cut; verified to descend from #11 |
| 16 | `feat/540-thinking-budget` | clean textually, semantic conflict (§3) |
| 17 | `fix/540-effort-collapse` | clean textually, resolved in §3 |
| 18 | `feat/111-wire-path` | clean |
| 19 | `feat/464-window-falsifier` | clean (carries `17e7c8e36a`) |
| 20 | `feat/677-park-wiring` | clean — added deliberately, see §5 |

Named ancestors verified present: `17e7c8e36a`, `54ffc4d90a`, `bc9c158a17`,
`887a6d477f`.

## 2. `fix/673-teardown-stack` — the union, as the ledger required

One conflict, `docs/dev/MERGE_TRAIN_2026-08-17.md`, doc only. Both lanes edited
the same numbered list and both had an item "5" saying DIFFERENT true things.
Resolved as a union with renumbering: theirs' newer RESOLVED status wins on the
item both cover (#728 identity), and the two constraints only HEAD carried are
preserved (`feat/677-park-wiring` DO-NOT-DROP; merge `fix/717-rebuild` at
`67572ceac3`). Nothing was dropped and nothing was invented.

## 3. `#540` — the semantic conflict a clean merge hid

`feat/540-thinking-budget` and `fix/540-effort-collapse` merged with NO textual
conflict and left a failing test: `'xhigh' != 'max'`.

That is not a merge accident. The budget branch pinned the OLD collapse
deliberately, to record the hazard before it was fixed; the fix branch then made
the value pass through. `fix/540-effort-collapse`'s own commit message
predicted this exact collision and pre-decided the resolution: *the pin follows
the fix, the fix is not reverted.* Applied verbatim in `b338c35a86`; the case
is renamed `test_xhigh_is_passed_through` and carries the history. No new
judgement was made at merge time.

**This is the one to remember from this train:** a textually clean merge is not
a semantically clean one, and only running the suite found it.

## 4. STOPPED: `fix/602-fill-side`

Aborted and NOT in the train, per the brief's rule that an ambiguous hunk pair
stops that branch.

Five conflicts. Four are mechanical (`seam_slope.py` — theirs is a strict
superset of `__all__` plus the new function; two add/add test files where one
side is empty; `model_runner_kv_cache_mixin.py` — a constant plus its use).
The fifth is not:

`python/sglang/srt/planner/pp_cut.py` carries two DIFFERENT revisions of the
same ticket. HEAD: *"#702 revision — CO-SOLVING the KV/mamba token vector with
the layer cut"* with `class WorldMemory` (96 lines). Theirs: *"#702 revision 4
— the PP pool divides by ATTENTION layers"* with `class PhasePoolModel` (183
lines). Choosing one, or unioning them, is a design decision about which
revision of #702 is current — semantic invention, not conflict resolution. It
needs the branch owner.

**Consequence for the brief's question:** the "#624 stub-drift 11 should CLEAR
if `fix/602-fill-side` is included" check could not be evaluated, because the
branch is not included. Separately, no test named for #624 could be located;
the stub machinery itself is evidently healthy (`unit/mem_cache` went from 940
failures to 0, the 940 having become skips through `mem_cache/conftest.py`).

## 5. `feat/677-park-wiring` — added, with the reason

Not on the brief's candidate list, but merged deliberately. Three reasons, in
order of weight: this repo's own train-1 prep marks it **DO-NOT-DROP** ("the
sole carrier of the six serving-only commits enumerated in §2"); the parallel
train (§7) merged it; and its absence was a live hypothesis for the `managers`
delta. It merged clean and added 12 passes. It did NOT change the failure
count — see §6.

## 6. Test gate

Every number measured twice: once on BASE `a157bf1889` before any merge, once
on HEAD `c9964f7e4d` after all of them. Hermetic, `CUDA_VISIBLE_DEVICES=""`,
`-p no:randomly`.

| suite | BASE | TRAIN | delta |
| --- | --- | --- | --- |
| `mem_ledger` | 0 F / 444 P | **2 F** / 492 P | **+2 NEW** |
| `server_args` | 0 F / 627 P | **1 F** / 663 P | **+1 NEW** |
| `boot_matrix` | 0 F / 140 P | 0 F / 159 P | clean |
| `spec` | 13 F / 711 P | 13 F / 711 P | unchanged |
| `entrypoints` | 4 F / 417 P / 3 E | 4 F / 450 P / 3 E | unchanged |
| `planner` | 2 F / 2494 P | 1 F / 2880 P | **−1 improved** |
| `distributed` | 21 F / 2689 P | 27 F / 2764 P | +6 inherited |
| `managers` | 9 F / 1587 P | 19 F / 2375 P | +10 inherited |
| `mem_cache` | 940 F / 748 P | **0 F** / 1103 P | **−940 improved** |

### The two deltas that are NOT this train's doing, attributed rather than asserted

* `managers` +10 — `reconcile/cluster-b-seam-model` checked out ALONE reports
  **19 failures**, identical to the train's 19. The composition introduces
  none.
* `distributed` +6 — the same delta previously attributed to
  `fix/701-ledger-wiring`, inherited through the reconciliation branch.

### The three genuinely NEW failures, with causes

1. **`server_args/test_phase_flip_args.py::test_v1_blockers_named`** — the
   `#630` PP × storage-HiCache refusal no longer fires (verified directly: the
   `dp_size` and `disaggregation_mode` blockers still raise, the
   `enable_hierarchical_cache` one does not). **A fix already exists**:
   `90e84ad268` *"[#630] Restore the PP x storage-HiCache guard: bounded is not
   fixed"* on `train/0817-exec`. Not pulled in — see §7.
2. **`mem_ledger/test_communicator_group_contract_612.py`** — the runtime
   builds a `decoupled_kv` communicator group (`mem_cache/decoupled_kv_arming.py`)
   that the #612 ledger declaration does not name. A contract test from one
   branch catching a group added by another; needs the declaration updated by
   its owner.
3. **`mem_ledger/test_r1_private_constant_gate_584.py`** — the newly merged
   #584 ratchet catches two private VRAM constants in
   `managers/phase_flip_seam_reserve.py` (`DEFAULT_MARGIN_MIB = 192` and one
   sibling), introduced earlier by #685/#696. The ratchet is doing exactly its
   job; whether those constants are legitimate is the #584 owner's call, and
   allowlisting them here would defeat the gate.

None of the three is a merge-resolution error. (2) and (3) are ratchets from
one branch catching content from another — the outcome a train gate exists to
produce.

## 7. A SECOND TRAIN EXISTS — reconcile before booting either

`origin/train/0817-exec` was pushed while this one was being assembled, **from
the same base** (`a157bf1889`). It is not an ancestor of this train and this
train is not an ancestor of it. It carries overlapping but non-identical
content: a `feat/677-park-wiring` merge, #545 runtime attach/detach, #726, #713,
two #706 fixes, and the `#630` guard restoration this train needs (§6.1).

Two trains cannot both fast-forward. This is a coordination decision, not a
desk one, and it is flagged here rather than resolved: nothing from
`train/0817-exec` was pulled into `train/0817-desk`.

## 8. Arrived after the candidate list — deliberately NOT merged

`docs/613-determination` and `docs/398-determination` were pushed while this
train was being assembled. They are low-risk docs, but the brief's list was
explicit and silently widening a train that F4-r4 will boot is the wrong
default. Follow-on items.

---

# RECONCILIATION with `train/0817-exec` (2026-08-17, second pass)

Both escalations from §7/§8 were decided by the coordinator and executed here.
`train/0817-exec` is the SERVING LINEAGE and authoritative for serving-critical
content; it was merged into this train.

* HEAD after reconciliation: see the branch tip; base for all deltas is still
  `a157bf1889`.

## R1. The merge, and where "docs keep yours" was wrong

Three conflicts, all documentation — and the category rule did NOT decide them.
Two of exec's three doc edits are **corrections to my own desk output, found by
a boot**, so exec won those on content:

* `docs/dev/DESIGN_706_BOOT.md` — **exec wins.** Its edits fix my run-card:
  `--hicache-size` takes an INTEGER number of gigabytes (`server_args.py:4187`),
  so my `5G` was simply wrong and would have failed the boot the card exists to
  drive; and precondition 1's `24.3 G` is recalibrated. A "docs keep mine"
  reflex would have shipped a bad flag value in a turnkey ticket.
* `docs/dev/NOTE_545_runtime_disk_tier.md` — **exec wins.** Their side strikes
  a claim of mine and replaces it with 101 lines backed by a real
  attach/detach run.
* `docs/dev/MERGE_TRAIN_2026-08-17.md` — **mine wins**, because my side is
  already the union built in §2 and carries the newer RESOLVED status.

The lesson worth keeping: "serving files vs desk files" is the right default,
but the deciding question is which side has EVIDENCE, and a doc can carry a
boot measurement.

## R2. Failure #1 (#630) did NOT clear — and the reason is a cross-lane contradiction

The coordinator expected exec's `90e84ad268` to clear
`server_args/test_phase_flip_args.py::test_v1_blockers_named`. It does not, and
the cause is worth stating precisely because it is a real disagreement, not a
missing patch.

There are TWO #630 guards:

* the RUNTIME one in `phase_flip_runtime.py` — exec restored it
  (`90e84ad268`, "bounded is not fixed");
* the PARSE-TIME one in `server_args._handle_phase_flip()` — still absent, and
  absent **deliberately**: `7336c08d0e [#703] Narrow the boot-time #630 blocker
  too, or the flag stays unusable`, whose code comment reads "#703 stage 2: the
  #630 blocker is REMOVED, matching its runtime twin"
  (`server_args.py:7899`).

Verified directly: the `dp_size` and `disaggregation_mode` blockers still
raise; only `enable_hierarchical_cache` does not.

So #703 removed the parse-time guard *because* the runtime twin was gone, and
exec then brought the twin back — which invalidates the removal's stated
reason. **Restoring the parse-time blocker was NOT done here**, for a reason
that needs a decision rather than a merge: it would refuse the exact boot this
train's own `DESIGN_706_BOOT.md` run-card instructs F4-r4 to perform (phase flip
WITH `--enable-hierarchical-cache`). Either the flip+HiCache boot is allowed and
the test is stale, or it is refused and the run-card is wrong. That belongs to
the #630/#703 owners.

**Left failing, cause named.** 1 failure in `server_args`.

## R3. Failures #2 and #3 — fixed and filed

* **#612** — `decoupled_kv` is now declared in `RUNTIME_COMMUNICATOR_GROUPS`
  (`mem_ledger/engine.py`). The contract checks construction SITES, and
  `initialize_decoupled_kv_group` has no production caller yet; the entry says
  that rather than implying a boot allocates it. `mem_ledger`: 0 failures.
* **#584** — both constants FILED with differentiated verdicts, never a silent
  baseline bump:
  * `planner/boot_instruments.py::_CHAIN_TOLERANCE_MIB` — **NOT A DEMAND
    DECISION**, call site read: it is the rounding slack for verifying the
    budget-chain identity (posts carry three decimals of a GiB), and it
    reserves nothing.
  * `managers/phase_flip_seam_reserve.py::DEFAULT_ARMING_MARGIN_MIB` —
    **NEEDS AUDIT**, verdict withheld: it does gate a VRAM decision (the
    #662-F4 arming floor above the corridor law), while its own comment argues
    it is an instrument tolerance. That call belongs to the #662-F4 owner.

## R4. A NEW regression the reconciliation introduced, and it may be a real one

`unit/mem_cache` went from **0 failures to 2** when exec was merged:
`test_pinned_post_registry_550.py::TheAdmissionChargesEveryRegisteredPost` —
both cases, "ValueError not raised".

Cause: exec's `d7d85b4e37` ("the pinned-host backstop billed allocated posts
twice") credits already-allocated posts back, because `available` is read live
and their bytes are already missing from it. That fix is measured and
boot-motivated (a Flip+HiCache boot refused on PP0 at "40.42 GB requested ...
33.97 GB available", of which 35.18 GB was three weight images already in RSS).

It is sound **only under the invariant its own comment states**: "registration
FOLLOWS allocation for every producer of a post". The #550 tests register a
30 GB post through `register_pinned_post` directly, WITHOUT allocating it, and
then expect a 20 GB newcomer to be refused against a synthetic 40 GB available.
Under the new model that scenario cannot occur, so nothing is charged and
nothing refuses.

**Not resolved here, deliberately.** Two readings remain, and they differ in
consequence: either #550's fixture encodes an impossible state and the test is
stale, or some producer does register without allocating and the credit-back
under-charges. Rewriting another lane's test to match a third lane's model
would erase that question. Owners: #706 (exec's fix) and #550.

## R5. Final gate — base `a157bf1889` vs reconciled head

| suite | BASE | RECONCILED | delta |
| --- | --- | --- | --- |
| `mem_ledger` | 0 F / 444 P | **0 F** / 494 P | clean (was +2, fixed in R3) |
| `server_args` | 0 F / 627 P | 1 F / 670 P | +1, explained in R2 |
| `boot_matrix` | 0 F / 140 P | 0 F / 159 P | clean |
| `spec` | 13 F / 711 P | 13 F / 711 P | unchanged |
| `entrypoints` | 4 F / 417 P / 3 E | 4 F / 450 P / 3 E | unchanged |
| `planner` | 2 F / 2494 P | 1 F / 2880 P | −1 improved |
| `distributed` | 21 F / 2689 P | 27 F / 2764 P | +6 inherited (`fix/701-ledger-wiring`) |
| `managers` | 9 F / 1587 P | 19 F / 2391 P | +10 inherited (`reconcile/cluster-b-seam-model` alone reports the same 19) |
| `mem_cache` | 940 F / 748 P | 2 F / 1142 P | −938 net; the 2 are R4 |

**Zero UNEXPLAINED new failures.** Three failures remain against base and each
has a named cause and a named owner: R2 (1, `server_args`), R4 (2,
`mem_cache`). The inherited `managers` +10 and `distributed` +6 are attributed
by measurement, not assertion.

## R6. Also filed

`docs/dev/TICKET_702_unify_revisions.md` — `WorldMemory` canonical (boot 2
validated it), `PhasePoolModel`'s attention-layer divisor and #723 frontier
completeness to be ported as an EXTENSION, never a parallel model. The four
mechanical conflict resolutions from the aborted `fix/602-fill-side` merge are
recorded there so the next attempt does not re-derive them.

---

# THIRD PASS — the three decisions executed (2026-08-17)

## T1. #630: parse-time twin RESTORED, narrowed to match the runtime guard

Decided: the runtime guard is the current truth (pp_sync desync UNROOTED, repro
in flight), so the flip+HiCache boot IS legitimately refused today for
storage-backed tiers.

Restored in `server_args._handle_phase_flip()` with the runtime clause's exact
condition -- `pp_size > 1` AND `enable_hierarchical_cache` AND a storage backend
-- and its reason string, pointing at the twin. Refusing at parse costs a
second; the runtime twin fires after a full weight load, which cost 11 minutes
and a dead instance.

The #703 comment that read "the #630 blocker is REMOVED, matching its runtime
twin" is rewritten to the new truth WITH its provenance: `9da9dfd025` bounded
the wait and never rooted the desync, and `test_hicache_bounded_waits_630.py`
proves only that a bounded call raises on schedule against mocked Work objects.

`test_v1_blockers_named` now names a storage backend, because the narrowed
guard deliberately no longer refuses the device+host-local tier. That is the
test asserting the guard as decided, not a test bent to fit code.

`DESIGN_706_BOOT.md` gains **PRECONDITION 0 -- THIS BOOT IS BLOCKED TODAY**,
with the measurement, the lift condition ("a test proves two REAL ranks
rendezvous, not that a wait expires"), and what stays reachable meanwhile
(single-stage flips, device+host-local tier at any stage count). The refused
boot and the ticket describing it can no longer drift apart.

**`server_args`: 1 failure -> 0.**

## T2. #550: the determination, from the producers

Every consumer reaches the registry through `check_and_register_pinned_post`
except one: `weights_arena.py:428` (`_register_image_post`), and it registers
BEFORE it allocates -- the allocation is at :439-460. So exec's stated
invariant, "registration FOLLOWS allocation for every producer of a post", is
literally inaccurate.

It does not change the verdict, and the reason is the one that matters: the gap
closes INSIDE the same call, so any LATER checker sees bytes already allocated.
No producer leaves a registered-but-unallocated post standing on a success
path, which is the condition the credit-back actually depends on.

So the fixtures encoded a machine that cannot exist, and they are corrected on
physics rather than on either lane's model. `available` is read LIVE: an
allocated 30 GB post cannot coexist with a static 40 GB available, and 40 GB
cannot survive allocating 20 GB. Both tests keep their exact intent -- a post
must not be admitted on its own arithmetic; the second post must see the first
-- with numbers a machine can be in (10 GB remaining; 40 then 20).

**RESIDUAL, filed rather than buried:** on the `weights_arena` allocation
FAILURE path the post stays registered with nothing allocated, and there the
credit-back would under-charge. Narrow, real, owner #706/#695.

**`mem_cache`: 2 failures -> 0.**

## T3. Latecomers merged

`feat/363-remainder` (DESIGN_363 §21, which was only a stub on the train) and
`close/363-actuator-verdict` merged clean. `feat/ledger-vram-authority`
(`756aa52b58`) was already an ancestor -- no-op, verified rather than assumed.

Slot-3's `train/0818-desk-410-pinning` was NOT merged, per the decision: it is
superseded by the running A+B reconciliation.

## T4. Final gate — base `a157bf1889` vs `a6b2feb161`

| suite | BASE | FINAL | delta |
| --- | --- | --- | --- |
| `mem_ledger` | 0 F / 444 P | **0 F** / 514 P | clean |
| `server_args` | 0 F / 627 P | **0 F** / 671 P | clean (T1) |
| `boot_matrix` | 0 F / 140 P | **0 F** / 159 P | clean |
| `spec` | 13 F / 711 P | 13 F / 711 P | unchanged |
| `entrypoints` | 4 F / 417 P / 3 E | 4 F / 450 P / 3 E | unchanged |
| `mem_cache` | 940 F / 748 P | **0 F** / 1144 P | −940 (T2) |
| `planner` | 2 F / 2494 P | 1 F / 2880 P | −1 improved |
| `distributed` | 21 F / 2689 P | 27 F / 2769 P | +6 inherited |
| `managers` | 9 F / 1587 P | 22 F / 2413 P | +13, see below |

Six suites now have ZERO failures. The two remaining deltas:

* `distributed` +6 — inherited via `reconcile/cluster-b-seam-model` from
  `fix/701-ledger-wiring`, attributed by measurement in the first pass.
* `managers` +13 — of which **19 of the 22 are the cluster-b inherited set**
  (that branch checked out alone reports exactly 19). The remaining three
  appear only in the FULL-SUITE run after the latecomers; the two named ones,
  `test_regime_gate_tools.py::TestGateToolSmokes::test_the_gate_1_readout_smoke_passes`
  and `::test_the_gate_2_replay_smoke_reproduces_the_f2_result`, **pass alone
  (22/22) and pass under two different `-k` filters**, and the count is
  deterministic at 22 across repeated runs. That is test-order pollution, not a
  code regression — a follow-up for the #363 owner, named here so it is not
  rediscovered as a mystery.

**No unexplained failures remain.** Every non-zero suite is either identical to
base, attributed by measurement to a contributing branch, or named as
order-dependent with the evidence for that claim.

---

# FOURTH PASS — the residual closed (2026-08-17)

The `weights_arena` allocation-failure window filed in T2 is closed, and the
audit it triggered found the shape is SYSTEMIC rather than a weights_arena
quirk.

## F1. Why the window exists at all

Every producer declares its post BEFORE allocating, and that is deliberate:
"an over-commitment is refused instead of discovered: the registry's whole job
is to fail at the declaration rather than at the allocation"
(`read_buffer_pool.py:70-72`). Correct for the CHECK — and it necessarily
opens a failure window. If the allocation then raises, the post describes bytes
that never existed, and #706's credit-back subtracts already-allocated posts
from the next admission's demand. A post that never allocated is credited back
anyway, so the next admission is charged too little: **the registry would wave
through the exact over-commitment it exists to refuse.**

## F2. The audit (brief item 3) — every producer, named

All eight register then allocate. The brief expected two siblings; there are
seven.

| producer | allocation after registration | state |
| --- | --- | --- |
| `model_executor/weights_arena.py:428` | `:439-460` pinned alloc + fallback | **FIXED** |
| `mem_cache/read_buffer_pool.py:73` | `:79` `[factory() ...]` | **FIXED** |
| `mem_cache/memory_pool_host.py:141` | `conv_device_ptrs` | AFFECTED, filed |
| `mem_cache/memory_pool_host.py:770` | `alloc_func` / `kv_buffer` | AFFECTED, filed |
| `mem_cache/memory_pool_host.py:1147` | `alloc_func` / `kv_buffer` | AFFECTED, filed |
| `mem_cache/memory_pool_host.py:1680` | `alloc_func` | AFFECTED, filed |
| `mem_cache/pool_host/mha.py:684` | `k_data_refs` | AFFECTED, filed |
| `mem_cache/pool_host/base.py:164` | `init_kv_buffer()` | AFFECTED, filed |

None is clean. The six unfixed are on the KV-pool hot path and each wants its
own allocation-failure test; fixing all eight in one commit on a train branch
is a blast radius this slice did not need. `unregister_pinned_post` already
exists and both fixes use it, so the remaining six are mechanical.

## F3. The fix shape

Constraint honoured: the success path is byte-identical — same allocator, same
fallback, same return. Only the failure path is new, and it obeys #386: the
post is taken back and the ORIGINAL exception is re-raised untouched, so the
operator still reads `cudaHostRegister`'s own words rather than a cleanup
message. `_register_image_post` now returns the name it registered so there is
something to take back; returning `None` means nothing was registered and there
is nothing to undo.

## F4. Tests — extended, not rebuilt

Added to `test_pinned_post_registry_550.py` (the #548/#550 characterization
file, per the gate), 4 new cases: the poisoned state, the #386 no-masking
property, a success-path guard so the fix cannot pass by never registering, and
the same window in `read_buffer_pool`.

Can-fail proven three ways: cleanup made a no-op reds the poisoned-state case;
wrapping the error in a new `RuntimeError` reds the no-masking case; disabling
the ring's cleanup reds the sibling case. Each reds ONLY its own test.

`unit/mem_cache`: 1148 passed, 0 failed. `unit/model_executor`: 15 failed / 660
passed, IDENTICAL to the pre-slice commit `fac99d644d` measured in a pristine
worktree — the 15 are pre-existing and this slice adds none.

---

# FIFTH PASS — #729 closed (2026-08-17)

The six remaining register-then-allocate sites are closed, so the window the
fourth pass found is shut at all eight producers.

## G1. Why a decorator and not six try blocks

All six are `__init__` bodies whose allocation runs to the END of the
constructor. Wrapping each would mean re-indenting six constructors — a large
diff for a small property, on a train branch, in the KV-pool hot path. Instead
`revert_pinned_posts_on_failure` (added to `pinned_host_budget`, the same
authority that owns `unregister_pinned_post`) is applied as one line per site.

On success the wrapper does nothing at all, so every success path is
byte-identical. On failure it undoes exactly the posts that appeared during the
call — a set difference, so a post registered by someone else is never touched
— and re-raises the ORIGINAL exception untouched (#386).

Nesting is correct by construction, which matters because **`HostKVCache` is
the base class of the other five**: a subclass `__init__` that fails after
`super().__init__()` registered undoes BOTH, which is right, because the object
as a whole failed. A dedicated case pins that.

**Limit stated at the definition rather than discovered later:** the set
difference is not safe against a CONCURRENT registration from another thread
inside the same extent. Every producer wrapped here is a boot-time constructor
on one thread; a future off-thread producer must not use this.

## G2. One test over the shape, not six copies

`TheSixConstructorsRevertTheirPost` drives the property, not six constructors'
internals:

* `test_every_site_carries_the_guard` — subtests over all six, asserting each
  `__init__` is actually wrapped. This is the half that catches a site quietly
  losing its decorator, and a seventh producer written tomorrow is one tuple
  away from being covered.
* three behaviour cases over the real decorator: a failing call reverts only
  what IT registered, a successful call keeps its post (so the guard cannot
  pass by never registering), and a nested failure undoes the `super()` call
  too.

## G3. Can-fail, three mutations, each redding only its own case

| mutation | reds |
| --- | --- |
| one site loses its decorator | `test_every_site_carries_the_guard` (1 subtest of 6) |
| the decorator stops unregistering | the failing-call case AND the nested case |
| it undoes everything, not just new posts | `test_a_failing_call_reverts_only_what_it_registered` |

## G4. Suites — zero new failures

| suite | before this slice | after |
| --- | --- | --- |
| `unit/mem_cache` | 1148 P / 0 F | **1152 P / 0 F** |
| `unit/model_executor` | 15 F / 660 P | 15 F / 660 P (identical) |
| `unit/managers` | 22 F / 2413 P | 22 F / 2413 P (identical) |

FEATURE_CATALOG's "STILL OPEN, six sites" entry is replaced with the closed
state and the mechanism, in the same step.

---

# SIXTH PASS — the #410/#411 reconciliation merged (2026-08-17)

`train/0818-desk-410-reconcile` (through `9509c7ba8a`) is in, with the lineage-B
modules deleted in the merge commit itself.

## H1. The deletion set was two files, not three

`pin_ledger.py` STAYS: the reconciliation keeps and extends it (the `unpinned`
shortfall field) and `hicache_storage` consumes it. Deleted instead:
`mem_cache/session_manifest.py`, `mem_cache/session_bundle.py`, and
`test_session_manifest_410.py` — the B test that imported a deleted module and
would otherwise have errored the suite.

## H2. Resolution authority

21 hunks / 11 files, resolved on DESIGN_410 and NOTE_411's own verdicts. The
three that were not mechanical:

* evictor "allocated accounting" hunks -> theirs, per NOTE_411 §"A's evictor
  moves to ALLOCATED bytes".
* evictor scan hunk -> OURS, departing from the incoming text ON EVIDENCE: this
  train had already refactored that inline flat scan into the injectable
  `_iter_existing` and applies `_allocated_size` in `_scan_existing_files`. The
  verdict was already satisfied; taking theirs would have duplicated the stat
  and regressed sharded-layout support.
* `AUDIT_421` F6 cells -> merged to name BOTH credits (#286's first consumer,
  #410's checkpoint tier). Both are true and a table cell cannot hold two rows.

## H3. Three incoherences the textual merge hid

Every one was found by RUNNING the suites, not by reading the diff.

1. **Duplicate methods, second silently shadowing the first.** Both lineages
   define the same methods in different places, so git kept both. An AST audit
   over every merged file found five: `stats` and `_allocated_size` in the
   evictor; `capacity_stats`, `pin_checkpoint`, `unpin_checkpoint`, `pin_stats`
   in `hicache_storage`. Resolved per pair. (`http_server`'s same-named pair is
   NOT an artifact — two different FastAPI handlers, present at HEAD.)
2. **Eviction stopped removing files.** An incoming block extended past its
   comment into the removal line, replacing the injected resolver with a flat
   join. This backend injects the sharded resolver, so `os.remove` missed, the
   entry left the LRU and the bytes stayed — the cap silently stopped holding.
3. **`_pin_path` was flat on a sharded store.** The reconciliation wrote it flat
   because its own lineage had no sharding; here it names a path the store never
   writes, so `pin_checkpoint` dropped every stem as "missing" and pinned
   NOTHING while reporting success. It now delegates to `_existing_path`, which
   preserves the property the reconciliation actually asked for — "the ledger
   must stat exactly what the evictor unlinks" — under this layout.

## H4. Candidates: one taken, then dropped; two deferred

* `audit/stale-gates` merged clean and was then DROPPED. It carries
  `4a16043d1a Revert "[#713] IDLE-LOCKED post-cutover settle"`, which strips 116
  lines from `phase_policy.py` and removes a `PhasePolicyConfig` kwarg that
  F4-r4's #730 tests use — six failures, all in `test_vacuous_decode_exit_730.py`.
  A revert of #713 versus the serving lineage's #730/#731 is a coordination
  decision, not a merge resolution. **The branch needs rebasing onto the current
  train, or the revert re-justified, before it can ride.**
* `feat/407-registry-reconcile` — DOCS-ONLY conflicts
  (`DESIGN_407_memtier_registry.md`, which carries my own §9 determination, and
  `FEATURE_CATALOG.md`). Cheap next pass.
* `fix/485-gdn-family-report` — conflicts in `uneven_perf.py` (code). Next pass.

## H5. Gate — base `a157bf1889`, head `4b9e4c2192`

| suite | established | now |
| --- | --- | --- |
| `mem_ledger` | 0 F / 514 P | 0 F / 514 P |
| `server_args` | 0 F / 672 P | **0 F / 681 P** |
| `boot_matrix` | 0 F / 159 P | 0 F / 159 P |
| `spec` | 13 F / 711 P | 13 F / 711 P |
| `entrypoints` | 4 F / 450 P / 3 E | 4 F / 450 P / 3 E |
| `planner` | 1 F / 2880 P | 1 F / 2880 P |
| `mem_cache` | 0 F / 1153 P | **0 F / 1157 P** |
| `managers` | 22 F / 2413 P | **19-20 F / 2530 P** |
| `distributed` | 27 F / 2769 P | 27 F / 2769 P |

`managers` IMPROVED on its established 22: the reconciliation's own #410/#411
tests now pass and the remainder is the cluster-b inherited family plus the
documented order-dependent gate-tools pair (passes alone and under filters).
Zero unexplained failures.

Also verified rather than assumed: `54ffc4d90a` (pp_with_spec register row) is
an ancestor, and `test_phase_flip_args` no longer asserts the #630
hierarchical-cache refusal — F4-r4's lift `be933407ab` removed that third twin
and the file is 23 passed / 0 failed here.
