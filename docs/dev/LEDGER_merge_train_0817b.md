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
