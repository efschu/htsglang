# MERGE NOTES — branch `fix/602-fill-side`

Base `7936bc4850` (hotfix/677 line). Tip `e21e87f204`. **18 commits.**
All work is **desk-only**: hermetic tests (`CUDA_VISIBLE_DEVICES=""`), no boot
run from this branch, nothing deployed *by* it.

**Merge target: `integration/r2`** — that is the live line; the serving tree
descends from its tip `a73a0d8da8`. (`integration/r3-probe-next2` is stale,
last touched 2026-08-08.)

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
| 18 | `e21e87f204` | #690 flip tail timed (movers/cutover) + DONE-line arity guard | 9 | **yes** | no |

**Runtime-touching: 6 of 18.** Four are already on the serving line
(`c41645c8c9`, `ce6035884d`, `658ea3ac11`, `84b0171fa6` — landed there as
`f1f31d2991` and `de92bb6529` among others). Two are branch-only:
`5301b94019` and `e21e87f204` — see §4.

Everything else is `planner/pp_cut.py` (a desk tool, imported by no serving
path), test files, or docs.

## 2. Dry-run merge — **zero conflicts**

Performed in a throwaway worktree off `integration/r2`, `--no-commit --no-ff`,
then aborted and the worktree dropped. Nothing was resolved, because nothing
needed resolving:

```
Automatic merge went well; stopped before committing as requested
unmerged paths: 0
```

**Tests on the merged tree** (not just on the branch):

* `managers` — **2093 passed, 0 failed** (identical to the branch)
* `planner` — **2574 passed, 2 failed** — the same two pre-existing
  `test_rejected_evidence_pins` failures, present on the base and unrelated

So the merge is clean textually **and** semantically on the two suites this
chain touches.

### 2a. But the merge is not 18 commits — it is 115

`7936bc4850` is **not** an ancestor of `integration/r2`. Merging this branch
therefore drags in its whole base lineage:

```
integration/r2..e21e87f204   115 commits
  of which mine               18
  the other 97                #662 x20, [PhasePolicy] x18, #677 x8, #678 x7,
                              #679 x6, #684 x3, #631 x3, #681 x2, ...
```

129 files, +22 731 / −762.

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
2. **Test-only** (`9aae89ddee`, `f57ac09273`, `dd4e647d16`) — three #624 drift
   guards. No runtime. They take `managers` from 4 failures to 0 and should
   land early so the integration line's suite is green while the rest is
   reviewed.
3. **Desk tool + docs** (`890e0b1f35`, `d9ed47b895`, `7f2f8c6aa1`,
   `f93be643cc`, `9f5877cb62`, `5a1e087c37`, `90e37f0510`, `f6525562c6`,
   `d7e34e84d1`) — all `planner/pp_cut.py` and markdown. `pp_cut` is imported
   by no serving path, so this group cannot change runtime behaviour. Note
   `5a1e087c37` depends on `c41645c8c9` (group 1) and `90e37f0510` depends on
   the four #602 commits before it, so keep this group internally ordered.
4. **Hold** — `5301b94019`, `e21e87f204`. See §4.

## 4. Do NOT merge yet

Two commits change runtime behaviour that the **current soak is validating**,
and neither is on the serving line:

* **`e21e87f204` (#690, flip tail timing).** Touches
  `managers/phase_flip_runtime.py` — the seam hot path — and changes the
  `PHASE-FLIP DONE` line format. The operator has already queued it to land on
  the deploy line **together with the W=8/W=4 probe, after the #694 soak
  verdict**. Merging it into integration ahead of that puts it in front of the
  soak it is meant to be measured by. Hold until that window.
* **`5301b94019` (#685, cold-boot seam slope).** Touches
  `model_executor/model_runner_kv_cache_mixin.py` — the boot sizing path. It is
  announce-only today (the derived reserve stays inactive, so no pool size
  changes) and it is guarded by abstentions, but it is still an unsoaked boot-path
  change, and the R′ decision it waits on (§5) is not made. Hold with it.

Nothing else in the chain can change serving behaviour: groups 2 and 3 are
tests, docs, and a desk-only module.

## 5. Open decisions carried forward

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

## 6. Retracted, on the record — twice

Revision 1 recommended `31,16,17` at **+36.3 %**; revision 2 `29,19,16` at
**+3.6 %**. Both withdrawn — the first for bench-priced weights, the second
because the ±5 % gate was comparing a corridor-safe floor against a measured
pool (~29 % apart in a seam-worst regime, read −23.3 %), fixed in `90e37f0510`
and validated at **−0.5 %** against F4-r4's measured 471 303.
