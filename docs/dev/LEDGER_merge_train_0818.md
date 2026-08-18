# LEDGER: merge-train pass 3 lineage map (2026-08-18)

Written BEFORE the train, because today ran on TWO divergent lineages and
the 55-commit gap was found only when a fix branch missed the review tip.
Every authority claim below is backed by the quoted command output; nothing
here executes a merge — comp4's boot proof gates pass 3.

The two lineages:

* **review lineage** — tip `c3fd6f6ab8` ("[merge-train pass 2] Ledger: five
  of eight already in"); carries the layer-set memory_pool refactor
  (`local_slot`) and 55 commits the desk lineage lacks.
* **desk lineage** — forked at `aa2abc525b` (pre-review desk train);
  today's #747/#750/#752/#754/#755 originally landed here.
* **the composite** — `c546eed923`, F4-r5's cherry-pick of the desk fixes
  ONTO the review lineage. It is the authority the next train tip grows
  from; comp4 = composite + fix/756 (`921d63defc`).

## (a) Merge-base of every branch vs the review tip

```
$ for b in ...; do git merge-base $b c3fd6f6ab8; git merge-base --is-ancestor c3fd6f6ab8 $b; done
2c936ff82b  merge-base=c3fd6f6ab8f7  CONTAINS_REVIEW_TIP    fix/749-thread-patch-leak
e33dff1bc0  merge-base=aa2abc525b8d  diverged               fix/750-interval-divisibility
a983811f16  merge-base=aa2abc525b8d  diverged               fix/752 (desk form)
f384591531  merge-base=aa2abc525b8d  diverged               fix/754-layerset-flip-scope
d60b11a258  merge-base=aa2abc525b8d  diverged               feat/755-mamba-slot-demand
921d63defc  merge-base=c3fd6f6ab8f7  CONTAINS_REVIEW_TIP    fix/756-localslot-family
ada1995e3c  merge-base=c3fd6f6ab8f7  CONTAINS_REVIEW_TIP    fix/flip-image-host-post
d7984ab810  merge-base=bc9c158a175a  diverged               train/0817-desk-727
d11b29d2dd  merge-base=422a682eff35  diverged               docs/740-scaffold-residual
2cf6a6e4b8  merge-base=c3fd6f6ab8f7  CONTAINS_REVIEW_TIP    fix/751 boundary pin
c546eed923  merge-base=c3fd6f6ab8f7  CONTAINS_REVIEW_TIP    composite
```

feat/753-gapped-pp-wire (Slot-3, in flight): local tip at write time
`082293f20b` ("[#753] Item 3: the wire over a REAL 3-process gloo group");
its compose base `a91a8d2927` merged fix/752 — Slot-3 re-bases per their
own plan once comp4 proves.

## (b) Absorption: what the composite already carries

Patch identity by `git cherry <composite> <branch> <base>` — `-` means the
composite carries a patch-equivalent commit, `+` means it does not:

```
$ git cherry c546eed923 e33dff1bc0 aa2abc525b
- da555ba0e7 #742 fastsafetensors refusal
- ff049c77bd #747 step 1 determination
- 50e548803c #747 seams 1-2
- 0b94eb6160 #747 seams 3-5
- 6cc1891660 #747 lift
- e33dff1bc0 #750 divisibility
$ git cherry c546eed923 ada1995e3c c3fd6f6ab8
- ada1995e3c flip-images
$ git cherry c546eed923 a983811f16 e33dff1bc0
+ a983811f16 #752 (desk form)          <- NOT patch-identical, see (d)
$ git cherry c546eed923 2cf6a6e4b8 c3fd6f6ab8
+ 2cf6a6e4b8 #751 boundary pin
$ git cherry c546eed923 2c936ff82b c3fd6f6ab8
+ ae739399d7 #733 refutation
+ 2c936ff82b #749 order-dependence root
$ git cherry c546eed923 f384591531 a983811f16
+ 3facc4b80c #735 arithmetic-check doc adoption (pp-layer-set tip)
+ f384591531 #754 scope guard
$ git cherry c546eed923 d60b11a258 f384591531
+ d60b11a258 #755 determination (docs only)
```

The layer-set FEATURE content below `3facc4b80c` is patch-absorbed in the
review lineage (everything else in the `f384591531` range showed `-`);
`git merge-base --is-ancestor 3facc4b80c c3fd6f6ab8` is false only because
the branch TIP (a two-file docs commit: `DESIGN_pp_layer_set.md`,
`NOTE_735_arithmetic_check.md`) was never taken.

## (c) Proposed order onto the next train tip (comp4 = 921d63defc), with conflict predictions

Rule of the day: divergent-lineage BRANCHES are never merged as branches —
their unique commits are cherry-picked. A branch merge drags the whole
desk history back in and re-fights every conflict the composite already
resolved.

1. **fix/756 (`921d63defc`)** — IS the comp4 tip; nothing to do. Note: the
   composite's copy of `test_hicache_gdn_layer_counter_752.py` has NO
   `_local_slot_of` fixture line (0 grep hits at `c546eed923`); #756's
   class attribute is what makes that suite green — keep #756 first.
2. **fix/749 (`ae739399d7` + `2c936ff82b`, Slot-3)** — review-based, merge
   as a branch. Files: `test/conftest.py`, NOTE_749, two tests. Predicted
   conflicts: none (test infra the composite did not touch). Early because
   it de-flakes every suite run after it.
3. **fix/751 (`2cf6a6e4b8`)** — review-based, one test-only commit, merge
   as a branch. Predicted conflicts: none
   (`test_preflight_and_config.py` untouched by the composite).
4. **#754: cherry-pick `f384591531` ONLY.** Files: `distributed/utils.py`
   (the scope guard inside `get_pp_layer_set` — the composite's copy of
   that function is the absorbed layer-set version, same body; expect
   clean or context-trivial), new test file (clean), and a one-line
   fixture edit to `test_hicache_gdn_layer_counter_752.py` — **predicted
   conflict: that hunk will not apply** (the composite's test copy differs
   and #756 made the fixture unnecessary). Resolution: DROP the fixture
   hunk; keep the other two files.
5. **#735 docs: cherry-pick `3facc4b80c`.** Both files exist at the review
   tip — **predicted conflict in `DESIGN_pp_layer_set.md`** (both
   lineages amended it). Resolution: editorial union; the load-bearing
   deltas are the NVML-total slot ceiling (31/24/21 -> 30/23/20) and the
   31-vs-32 crossing-count reconciliation — they must survive.
6. **#727: cherry-pick the trio `3f8996a1a6`, `5745a545d6`, `d7984ab810`**
   — never the branch (its old base `bc9c158a175a` drags bench/588,
   bench/613, family-placement docs and FEATURE_CATALOG history along:
   see the `git diff --name-only bc9c158a17 d7984ab810` sweep). Trio
   files: `ct_embedding.py` (new file, clean), `tools/*` (new, clean),
   tests (new, clean), TICKET_727 (new, clean); **predicted small
   conflicts** in `compressed_tensors.py` (`get_quant_method` hunk) and
   `qwen3_5.py` (the embedding-gate hunk) if the review lineage moved
   those regions — both are two-line gates, resolution is re-application
   by hand.
7. **docs, cherry-picks:** `d60b11a258` (#755 determination + A/B ticket:
   two NEW files, clean) and #740 as a PAIR: `0480f4be27` (the original
   note) THEN `d11b29d2dd` (the §5a residual). CORRECTED BY THE EXECUTOR'S
   DRY-RUN SMOKE: this ledger's first draft said d11b29d2dd "lands as a
   new file" — wrong, it is only the §5a diff and conflicts without the
   note commit underneath it (the smoke aborted on exactly that
   unpredicted conflict, which is the abort discipline working).
8. **feat/753 (Slot-3, in flight)** — lands last, per Slot-3's own
   sequencing after comp4's proof; its compose base already contains
   fix/752, so the only fresh content is the wire itself.

## (d) SUPERSEDED — do NOT merge again (double-application hazard)

* **fix/750-interval-divisibility (`e33dff1bc0`)** and with it the whole
  #742/#747 branch (`fix/742-fastsafetensors-drop-cache`,
  `6cc1891660`): all six commits patch-absorbed (`-` above). Re-merging
  re-fights the #747 seam hunks against their own cherry-picks.
* **fix/flip-image-host-post (`ada1995e3c`)**: patch-absorbed.
* **fix/752 desk form (`a983811f16`)**: semantically superseded, NOT
  patch-identical — the composite's `c546eed923` is the SAME skip
  resolved against the review lineage's `local_slot` form of
  `mamba2_layer_cache`. Merging the desk form guarantees a conflict in
  exactly that hunk with zero content gain. The composite's version is
  the authority.
* **feat/755, train/0817-desk-727, fix/754 as BRANCHES**: not superseded
  in content, but superseded as merge units — their unique commits are
  the cherry-picks of (c) 4-7; the branches themselves drag divergent
  history.

## Executor

`scripts/merge_train_pass3.py` executes this section (c) mechanically:
comp4-marker gate, live (a)-table re-check, mapped resolutions only
("ours" drop / "theirs"+marker verify / "manual" resumable stop),
LOUD ABORT on any unpredicted conflict, per-step suites, resumable JSON
state, `--dry-run` scratch-clone smoke. The smoke already ran end to end
against the real refs: steps 1-8 clean (the #727 gate-hunk "manual"
predictions did NOT fire — the picks applied), and it caught the #740
pairing error now fixed in (c)7. Nothing pushes; pass 3 still waits for
F4-r5's COMP4_ACCEPTED marker.

## Standing rule extracted from today

A fix branch must state its base hash in the commit message (all of
today's do). Before pass 3 starts, re-run the (a) table against the ACTUAL
train tip of that day — this file's outputs are frozen at 2026-08-18 and
go stale the moment the train moves.

---

# REFRESH — second wave (2026-08-18, post-harvest-composite)

The first map froze before today's second wave. New authority: **F4-r5's
harvest composite, tip `59ce2d8a30`** ("[#758] The batch line names the
rank..."), declared COMPLETE by its owner, review tip still an ancestor.
**Pass 3 is redefined: the harvest tip is the new train base; the remaining
unabsorbed branches cherry-pick on top.**

## (a') Ancestry vs the harvest tip (`git merge-base --is-ancestor R 59ce2d8a30`)

```
ABSORBED  921d63defc  comp4 (and its whole ancestry: #756/#752/#747/#750/
                      flip-images/#540-fix 57b04b2434)
ABSORBED  915ce1b868  #757 (F4-r5's own, regression-verified in-composite)
NOT       everything else listed below
```

## (b') Patch identity vs the harvest tip (`git cherry`)

Absorbed as DIFFERENT commits — **do NOT merge, the desk-#752 hazard
class**:

```
- 1cc0d24ae7  fix/748-armed-gate-scope   (absorbed=1, unabsorbed=0)
- d3bef88e66  fix/759-arming-economy     (absorbed=1, unabsorbed=0)
- d25b7a08c3  feat/755-slot-reorder      (absorbed=1, unabsorbed=0)
```

The #758 emitters need no branch: the harvest TIP ITSELF is a #758 commit.

Genuinely pending (unabsorbed commits > 0): `9e5647706f` (#757
independent — F4-r5's in-composite note: REVIEW against 915ce1b868, never
merge on top), `eab1926ea8` (#745 reachability suite, 1), `2cf6a6e4b8`
(#751, 1), `2c936ff82b` (#749, 2 — but CONTAINED in feat/753's lineage,
self-skips once 753 lands), `082293f20b` (feat/753, 10 — Slot-3, in
flight), `3facc4b80c` (#735 docs), the #727 lineage picks (+ head-chain
`3682331d33`), `f199828d11` (#738), `c92e65c547` (#535), `d60b11a258`
(#755 docs), the #740 pair, this ledger branch. `fix/706-remainder`: not
yet visible; slot reserved.

## (c'/d') New order and supersessions

1. **feat/753 lands FIRST, by its owner** — it is a 10-commit in-flight
   branch whose compose base already resolved the double-lineage, it
   carries #749, AND it folds #754 at the same seam
   (`dd08f8fe63`, `distributed/utils.py:1709` = its own pp_size=1
   phase-flip handling). The executor does not automate an owner's live
   branch; it runs AFTER 753 is on the tip.
2. **fix/754 (`f384591531`) is SUPERSEDED by the 753 fold** — semantic,
   not patch-identical (`git cherry 082293f20b f384591531` shows `+`):
   merging it after 753 guarantees a conflict in `get_pp_layer_set` with
   zero content gain. Removed from the executor plan. Same verdict class
   as desk-#752 in the first map. My own branch; retired without regret —
   its test file rides 753's fold or gets re-picked by the owner if their
   fold lacks one.
3. Executor cherry order on the post-753 tip: #745 suite, #751, #749
   (self-skips if 753 carried it), #735 docs, the #727 quartet
   (3f8996a1a6, 5745a545d6, d7984ab810, 3682331d33), #738, #535, #755
   docs, the #740 pair. Conflict predictions carry over from the first
   map where still applicable; the #754 fixture-hunk prediction is
   retired with its step.
4. **9e5647706f** goes to REVIEW, never merge (its owner's own
   in-composite note names the procedure).

Standing rule re-applied: these outputs freeze at the second wave;
re-derive (a') against the actual tip on train day.

## ADDENDUM: the double-#757 adjudication, and the tip moved again

**Harvest tip correction:** the composite advanced while the refresh was
being written — authority is now `da818719fe` (hicache PP-election fix),
which contains `59ce2d8a30` (#758 phase-tag) AND `caca35264d` (F4-r5
absorbed WINDOW_LADDER_0818 + its runner into the harvest). Both fresh
commits are absorbed by construction; the executor's HARVEST constant is
bumped.

**Double-#757 verdict: SEMANTICALLY DIFFERENT — LOUD REVIEW ITEM, not a
supersession.** The double-#754 template does NOT apply. Both fixes attack
the same root (corpse-S drain disabled; rank-local disarm routes; in-flight
dict-wire messages from before the arm) but at DIFFERENT intervention
points:

* `915ce1b868` (F4-r5, IN the harvest, regression-verified under load):
  the pre-arm leftover is **drained at DISARM** — the rank-local abandon
  routes clear the channel before control returns to the loop.
* `9e5647706f` (Slot-3, independent): the **ARMED drain re-enabled,
  demultiplexed** — a pure 4-way classifier
  (`output -> STASH / never-ran proxy -> DISCARD / ran proxy -> STASH /
  unstamped -> STASH`) handles messages AT ARRIVAL while armed, and its
  commit states the liveness property the disarm-time form does not have:
  "the message leaves the WIRE either way — the upstream's blocking commit
  waits on exactly that."

The review question, stated sharply: **during a long armed window, does an
upstream's blocking send on the dict wire stall the group under
`915ce1b868` (messages stay on the wire until the downstream disarms),
which `9e5647706f`'s arrival-time consumption would prevent?** If yes, the
two are COMPLEMENTARY halves (arrival-time demultiplex + disarm-time
sweep), not duplicates, and the merge form is Slot-3's classifier reviewed
against — and possibly layered onto — the harvest's fix rather than either
discarded. If no, `915ce1b868` stands and `9e5647706f` retires with its
test ideas re-hosted.

Test-coverage union (either way worth keeping): Slot-3's suite carries the
hermetic 3-process gloo repro (a stub cannot prove a message left on a real
blocking wire), 4 killed mutants, specimen numbers verbatim, and the
#631-pin correction. Its tests bind to `classify_armed_drain_message`
(their impl), so the tests ride the review verdict — they cannot be
cherry-picked bare.

Until the review lands: `9e5647706f` stays OUT of the executor plan
(unchanged), and the review is a NAMED PRECONDITION of pass 3 alongside
feat/753.

## REVIEW ITEM RESOLVED: double-#757 — verdict (b), COMPLEMENTARY, fold shipped

The named question was measured (`tools/probe_757_gloo_liveness.py`, 2-rank
gloo, CVD=99): gloo posts an isend instantly but `wait()` blocks until the
peer posts its recv — size-independent (proxy-4KiB: 3000.3 ms stall for a
3 s armed window vs 0.2 ms with immediate recv; output-8MiB: 3003.4 vs
3.9). The in-tree sender is exactly that shape
(`_pp_send_dict_to_next_stage` posts async, `_pp_commit_comm_work`
waits). **So the disarm-time form alone stalls an abandoned upstream at
its commit for the remainder of the downstream's armed window** (bounded
by the park/presence deadlines at ~30 s — not a deadlock, but a real
per-occurrence pipeline stall). Verdict (b): complementary halves.

Fold shipped as `fix/757-armed-liveness` (`194c3ea284` cherry of
`9e5647706f` onto the harvest tip, one predicted-shape conflict resolved
to BOTH mechanisms with the measurement quoted at the seam, plus the
probe commit `5e2c121595`). Both suites + the corrected #631 pin green
together (28); managers selection 633 passed vs the pristine tip's 625
with the IDENTICAL 2 pre-existing order-dependent failures — 0 new.
`9e5647706f` itself is now SUPERSEDED by the fold (never merged raw).
The executor gains the fold step; the pass-3 precondition list drops the
review and keeps only feat/753.

## feat/753 PREPARED: fix/753-on-harvest@8f2094f62b — hand-over to the router

The pass-3 sole remaining precondition is now a branch, same discipline as
the #757-liveness fold (scratch from the composite tip, owner's semantics
preserved, composite untouched): all three #753 commits
(`0882c7ae02`/`dd08f8fe63`/`082293f20b`) cherry-picked CLEAN onto
`b7e6a4110b` — zero conflicts; the #754 seam had no competing resolution on
the tip, exactly as this ledger adjudicated.

The zero-new-fails gate EARNED ITS KEEP again: the picked gloo suite failed
2 under the broad selection (tip baseline 75/0). Root, measured via an
instrumented child: `test_prefill_graph_barlink.py` — a file that arrived
from the wt-prefill-graph-qwen worktree — prepended that FOREIGN tree to
sys.path at COLLECTION-time import, and every later multiprocess test's
spawn children inherited it and resolved sglang from the wrong worktree
(ModuleNotFoundError on modules that exist only here). Fixed at both ends:
the foreign prepend removed (its own suite stays green against this tree),
and the picked gloo test hardened with a module-top `__file__`-derived pin
so its children are immune to any future collection-time prepend. After:
distributed selection 105/0 vs the 75/0 baseline; byte-gates (wire OFF =
NoCrossingWire = identity, both directions), the 31-crossing pin, the
can-fail (a dropped crossing changes the answer), and the 3-process gloo
progress test all green on the fold.

Routing: F4-r5 merges `fix/753-on-harvest` (one branch). The executor's
precondition list is then EMPTY; WINDOW_TICKET_735_STEP2's blocker row
reads PREPARED(branch) in both copies.
