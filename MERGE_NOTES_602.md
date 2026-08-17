# MERGE NOTES — branch `fix/602-fill-side`

Base `7936bc4850` (hotfix/677 line). Tip `5939d0e4a4`. **22 commits.**
All work is **desk-only**: hermetic tests (`CUDA_VISIBLE_DEVICES=""`), no boot
run from this branch, nothing deployed *by* it.

**Merge target: `integration/r2`** — that is the live line; the serving tree
descends from its tip, now `a157bf1889` (`a73a0d8da8` at revision 1;
`feat/qwen38-switchover` has landed there since).
(`integration/r3-probe-next2` is stale, last touched 2026-08-08.)

**Revision 2, 2026-08-16.** Extends the table by four commits, re-runs the dry
run against the new target tip, and folds in F4-r4's vouch (§5). Two commits
named in the operator's update are **not on this branch** — `5af1531c70`
(#696) and `c738ef57dc` (#689) are F4-r4's line, landing on serving with the
bundle. They are in §6 because they lift holds here, not because they are mine
to merge.

---

## 1. The chain

`RT` = touches runtime code (anything under `srt/` that is not
`planner/pp_cut.py`). `DEP` = an equivalent patch is **already on the serving
line**, cherry-picked by F4-r4 (verified by `git cherry`, not by message).

| # | commit | what it fixes | tests | RT | DEP |
|---|---|---|---|---|---|
| 1 | `890e0b1f35` | #602 KV-floor PP cut solver (`world_kv_floor`, `solve_pp_cut_for_kv_floor`) — makespan ≠ capacity | 18, 12 red-first | — | — |
| 2 | `d9ed47b895` | #602 draft runner calibrated from recorder (net = weights − overlap credit); seam solved as a fixed point | 20 | — | — |
| 3 | `7f2f8c6aa1` | #602 fixed overhead + transient from recorder; refuse-on-absence | 14 | — | — |
| 4 | `f93be643cc` | #602 weights from the checkpoint; found vision tower + MTP head replicated per stage | 14 | — | — |
| 5 | `9f5877cb62` | #685 rank-0 seam slope decomposed = one received attention layer | 5 | — | — |
| 6 | `c41645c8c9` | #685 `managers/seam_slope.py` — derivation as a dependency-free module | 10 | **yes** | **yes** |
| 7 | `5a1e087c37` | #685 planner delegates to that module (de-dup) | — | — | — |
| 8 | `ce6035884d` | #691 per-rank prefill timer admitted under PP + divergence guard | 10 | **yes** | **yes** |
| 9 | `5301b94019` | #685 cold-boot seam slope derived and announced | 14 | **yes** | no |
| 10 | `f6525562c6` | docs — merge notes v1 | — | — | — |
| 11 | `9aae89ddee` | #624 `BudgetHarness` drift guard | 4 | — | — |
| 12 | `f57ac09273` | #624 `_Sched` drift guard; one guard idiom | 5 | — | — |
| 13 | `90e37f0510` | #602 `predicted_pool` vs `corridor_safe_floor` separated; cut withdrawn | 13, 10 red-first | — | — |
| 14 | `d7e34e84d1` | docs — second retraction | — | — | — |
| 15 | `658ea3ac11` | #681 new-request admission gated on the group-uniform floor | 18, 12 red-first | **yes** | **yes** |
| 16 | `dd4e647d16` | #624 adder stub carries `fundable_extend_floor` + guard | 2 | — | — |
| 17 | `84b0171fa6` | #681 **third root** — eviction counted staged frees; non-closing `flush_free_group` | 8, 5 red-first | **yes** | **yes** |
| 18 | `e21e87f204` | #690 flip tail timed (movers/cutover) + DONE-line arity guard | 9 | **yes** | bundle |
| 19 | `3c984add3a` | docs — merge notes revision 1 | — | — | — |
| 20 | `8fb86efec9` | #697 the planner's budget sweep was deleting the seam records; shape-rule `is_budget_file` | 9 | **yes** | bundle |
| 21 | `5e0fa1ec00` | #441 wheel pinned, segfault attribution corrected, both guard flips refused | guard retrofit in an 8-test file | — | — |
| 22 | `5939d0e4a4` | #524 draft-pick sync fused to ONE host-path collective | 7, red-first | **yes** | no |

**Runtime-touching: 8 of 22.** Four are already on the serving line
(`c41645c8c9`, `ce6035884d`, `658ea3ac11`, `84b0171fa6` — see §5 for the
patch-id evidence). Two more, `e21e87f204` and `8fb86efec9`, are landing there
**now** with F4-r4's bundle; `DEP: bundle` becomes `DEP: yes` when he reports
the boot commit (§6). Two remain branch-only: `5301b94019` and `5939d0e4a4` —
see §4.

`5e0fa1ec00` (#441) is **not** runtime despite its ticket: it adds
`scripts/handover/live_handover_gate_shortrun.sh` and retrofits one test
guard. The wheel pin it records documents an existing artefact; it changes no
code that runs in the server.

Everything else is `planner/pp_cut.py` (a desk tool, imported by no serving
path), test files, or docs.

## 2. Dry-run merge — **zero conflicts**

Re-run for revision 2 against the **current** target tip `a157bf1889`
(2026-08-16), not revision 1's `a73a0d8da8`. Performed in a throwaway worktree
off `integration/r2`, `--no-commit --no-ff`, then aborted and the worktree
removed. Nothing was resolved, because nothing needed resolving:

```
Automatic merge went well; stopped before committing as requested
unmerged paths: 0
```

**Tests on the merged tree** (not just on the branch):

* `managers` — **2093 passed, 0 failed, 18 skipped** (identical to revision 1
  and to the branch)
* `planner` + the three new commits' test files — **2594 passed, 2 failed,
  127 skipped**
* Those 2 failures are `PpWithSpecEvidenceTest` in
  `test_rejected_evidence_pins.py`, and they are **pre-existing on the
  target**. Because the target tip moved since revision 1, this was
  re-verified rather than carried over: the merge was aborted and that file
  run on clean `a157bf1889` — same 2 failed, 2 passed. The chain introduces
  no new failure.

So the merge is clean textually **and** semantically on the suites this chain
touches, against the current target.

### 2a. But the merge is not 22 commits — it is 109

`7936bc4850` is **not** an ancestor of `integration/r2`. Merging this branch
therefore drags in its whole base lineage:

```
integration/r2..5939d0e4a4   109 commits
  of which mine               22
  the other 87                #662 x20, [PhasePolicy] x18, #677 x8, #678 x7,
                              #662-F4 x7, #679 x6, [PhaseFlip] x3, ...
```

124 files, +22 196 / −729.

Down from revision 1's 115 / 129 files / +22 731 because the target tip has
advanced: part of the shared lineage has landed on `integration/r2` in the
meantime. The ratio is unchanged in kind — four fifths of this merge is still
not mine.

**This is the single fact the operator needs.** Approving this merge is
approving the hotfix/677 lineage, most of which is not mine and not mine to
vouch for. If only my work is wanted, it has to be cherry-picked, not merged —
and §3 says which pieces do that cleanly.

## 3. Order — not atomic, and it splits cleanly

The chain is **not** atomic. It separates into four groups that can land
independently, in this order:

1. **Already on serving** (`c41645c8c9`, `ce6035884d`, `658ea3ac11`,
   `84b0171fa6`). Merging these into integration only reconciles the
   integration line with what is already running. Lowest risk; do first.
   **Joining this group with the bundle:** `e21e87f204` (#690) and
   `8fb86efec9` (#697) move here out of group 4 once F4-r4 reports the
   bundle's boot commit, because they deploy with it. The trigger is his boot
   commit, not this document — until it is reported they stay held.
2. **Test-only and script-only** (`9aae89ddee`, `f57ac09273`, `dd4e647d16`,
   `5e0fa1ec00`) — three #624 drift guards, plus the #441 handover script and
   its one guard retrofit. No runtime. They take `managers` from 4 failures to
   0 and should land early so the integration line's suite is green while the
   rest is reviewed.
3. **Desk tool + docs** (`890e0b1f35`, `d9ed47b895`, `7f2f8c6aa1`,
   `f93be643cc`, `9f5877cb62`, `5a1e087c37`, `90e37f0510`, `f6525562c6`,
   `d7e34e84d1`, `3c984add3a`) — all `planner/pp_cut.py` and markdown.
   `pp_cut` is imported by no serving path, so this group cannot change
   runtime behaviour. Note `5a1e087c37` depends on `c41645c8c9` (group 1) and
   `90e37f0510` depends on the four #602 commits before it, so keep this group
   internally ordered.
4. **Hold** — `5301b94019`, `5939d0e4a4`, and — until the bundle boot commit
   is reported — `e21e87f204` and `8fb86efec9`. See §4.

## 4. Hold list, and what lifts each hold

**Lifted by the bundle** (F4-r4's directed deploy, in flight 2026-08-16):

* **`e21e87f204` (#690, flip tail timing)** and **`8fb86efec9` (#697, the
  seam-record deleter)**. Revision 1 held #690 behind the #694 soak on the
  grounds that merging it into integration would put it in front of the soak
  meant to measure it. The operator has since routed both into the bundle, so
  that objection is spent: the hold lifts when F4-r4 reports the boot commit,
  and they become group 1. Record the boot commit against them; no other
  action here.

**Still held:**

* **`5301b94019` (#685, cold-boot seam slope).** Touches
  `model_executor/model_runner_kv_cache_mixin.py` — the boot sizing path. It is
  announce-only today (the derived reserve stays inactive, so no pool size
  changes) and it is guarded by abstentions, but it is still an unsoaked
  boot-path change, and the R′ decision it waits on (§7) is not made.
* **`5939d0e4a4` (#524, fused draft-pick sync).** Touches
  `speculative/spec_utils.py` and `eagle_worker_v2.py` — the per-round
  speculative decode path. Desk-verified only: 7 red-first tests, and `fuse`
  is opt-in at exactly one call site, so every captured path and every other
  caller stays byte-identical. It has still never run on metal, and its A/B
  is window-gated (§7). Do not merge ahead of that window.

Nothing else in the chain can change serving behaviour: groups 2 and 3 are
tests, a handover script, docs, and a desk-only module.

## 5. F4-r4's vouch — prefer the originals, exclude the scaffolding

F4-r4 cherry-picked four of these commits onto the serving line. **Take the
originals, not the duplicates.** That is not a preference: `git patch-id
--stable` makes each pair byte-identical, so the choice costs nothing and the
originals carry the authored history.

| duplicate (F4-r4) | original (this branch) | shared patch-id | ticket |
|---|---|---|---|
| `f630947ef3` | `c41645c8c9` | `2b77e28eba24` | #685 seam-slope module |
| `7c58aba7f2` | `ce6035884d` | `4bfb67fa2fa0` | #691 prefill timer |
| `f1f31d2991` | `658ea3ac11` | `fe32b521f97a` | #681 chunked gate |
| `de92bb6529` | `84b0171fa6` | `1924f14c939a` | #681 third root |

Because the patch-ids match exactly, `git cherry` already reports these four as
upstream-equivalent, and whichever side lands second is a no-op. The risk here
is not a conflict — it is double attribution in the log.

**Excluded from the merge: `1073702664`** ("#685 One receipt at the point of no
return"). Patch-id `8595e66885ac`, matching nothing on this branch: it is
F4-r4's diagnostic scaffolding, not part of this chain. Not mine to vouch for,
and it should not ride in on this merge.

## 6. The two bundle commits that are not on this branch

`5af1531c70` (#696, the arming floor may not be excused by funding that
evaporates) and `c738ef57dc` (#689, guard-ask receipt wording) are on
**F4-r4's line**, not on `fix/602-fill-side`. They appear in these notes only
because they are in the bundle now deploying, and because #696 is what was
DoSing the lanes — the serving outage observed during #524 was that directed
deploy, not a fault.

They need no action from this document: their merge path is his, and this
branch neither contains nor depends on them.

## 7. Open decisions carried forward

* **R′ semantics for the cold seam** — cold boots size floor-only; the derived
  slope is announced but the reserve stays inactive because `SeamReserve.active`
  needs an `id_space` anchor a derivation does not have, and the anchor-free
  `solve_pool_tokens` has no live caller. Boot-path design, F4-r4's call.
* **#602 metal arm — no arm on the censused regime.** Re-solved on F4-r4's
  census, the incumbent `28,20,16` is the global optimum over all 1953 cuts;
  `29,19,16` is 6.3 % worse. `TICKET_602_METAL.md` revision 3 is canonical, and
  carries the standing precondition: re-solve on the regime being booted and
  check `world_predicted_pool` against its measured pool before any arm.
* **#690 wave-count probe** — `SGLANG_FLIP_SEAM_WAVES` already overrides W, so
  the W=8 vs W=4 A/B needs no code. Queued behind the #694 soak.
* **#524 A/B — window-gated, and the old price does not transfer.** Arms:
  `5e0fa1ec00` (control) vs `5939d0e4a4` (fused); the diff is one keyword
  argument, so no runtime knob is needed. Recipe: TP=3, NEXTN 3/1/4, topk=1,
  no rejection sampling, barlink active, graphs on. The **primary** check is
  the collective count, not the perf delta — `_note_launch` already records
  launches, so 3 host-path broadcasts per round dropping to 2 is decisive and
  cheap. #476's 6.64 pp Seam A figure must **not** be reused as the expected
  gain: it was measured with the pre-#517 guard that synchronised once per
  collective, and #517 removed that sync. Run the A-vs-A noise floor first; if
  its spread exceeds the effect, report it as unresolvable rather than as a
  win.

## 8. Retracted, on the record — twice

Revision 1 recommended `31,16,17` at **+36.3 %**; revision 2 `29,19,16` at
**+3.6 %**. Both withdrawn — the first for bench-priced weights, the second
because the ±5 % gate was comparing a corridor-safe floor against a measured
pool (~29 % apart in a seam-worst regime, read −23.3 %), fixed in `90e37f0510`
and validated at **−0.5 %** against F4-r4's measured 471 303.
