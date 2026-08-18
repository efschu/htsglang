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
   two NEW files, clean) and `d11b29d2dd` (#740 §5a:
   `NOTE_740_prefix_cache_divergence.md` is ABSENT at the review tip —
   the cherry-pick brings the whole note including the §4a correction;
   clean as a new file).
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

## Standing rule extracted from today

A fix branch must state its base hash in the commit message (all of
today's do). Before pass 3 starts, re-run the (a) table against the ACTUAL
train tip of that day — this file's outputs are frozen at 2026-08-18 and
go stale the moment the train moves.
