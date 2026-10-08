# F0-E: the mechanical run on the NF line (08.10.2026)

Branch `desk/flliper-nf-1007` = NF freeze head `a452294dd2` + 15 kit-data commits (11 cherry-picked from F0-D, two of them pre-rename source commits; 4 F0-E kit data) + 3 kit commits
(mechanical, identifiers, translation) + 3 Nacharbeit commits (this file is in the third).  Kit-data branch: `desk/flliper-nf-kitdata-1008`.  Run directory of the kit:
`/root/.claude/jobs/a7d09d42/tmp/planer-f0e/runs/run-1008145951`.

## 1. The run

```
LINE=nf COLLISION_OK_FILE=tools/release/data/collision_ok_1007_nf.json FIXMAP_EXTRA=tools/release/data/ident_map_1007.json \
IDENT_FIX_COLLISION_OK_FILE=tools/release/data/ident_fix_collision_ok_1007_nf.json IDENT_FIX_WEB=1 \
RIG_TEST_WRAP=tools/release/capped_run.sh SKIP_DRY=0 bash tools/release/release_rename.sh <kit-data sha> --branch desk/flliper-nf-1007
```

`IDENT_FIX_WEB=1` and `IDENT_FIX_COLLISION_OK_FILE` are part of the line's run environment (the first run without `IDENT_FIX_WEB` left the dashboard
`.js`/`.html` on the German identifiers: dashboard suite 76 failed; it was discarded).  Result: collision_auto refused the seven F0-B/F0-C double-read
files (all byte-identical to the 27B tree; now in `collision_ok_1007_nf.json`, the two shell tools stay content-locked); verify PASS; imports PASS;
test set old vs renamed 10 failed / 1162 passed / 9 skipped / 3 errors in both, 0 new failures; step times 3 / 53 / 263 / 64 / 351 / 27 / 1833 (lock
wait) s.

## 2. What the pass leaves wrong, closed by rule

* Code and tests (15 files, identical on both lines): taken from the F0-D result (RENAME_PLAN 8.16: persisted documents `features.json` /
  `hardware.json` keep their spelling and are read in both, CLI dest of `features_update.py`, re-pinned digests and sort orders, pre-rename scenario
  tests skipped).
* Planner goldens: `planer_fixture_sync.py` (name rule; now also the NF reference golden `nf_n3_unchanged_1005`, whose `vector_lengths` keys are env
  names) and `planer_golden_regen.py --line nf` (`golden/nf/plan_nf_abl_n3.txt` from the NF launcher's own dump at the empty-HOME reference state S0:
  350 stable lines, 30 live-box lines; method graded 333/350 on the stable lines; unaligned 0).  The 27B-line dump goldens are not exercised on the NF
  line; they carry the bytes of the F0-D result.

## 3. Gates (pytest_gedeckelt.sh; old tree = the unchanged base, `$HOME` as named)

| Set | renamed tree | old tree |
|---|---|---|
| planner / catalog / compat (26 files, HOME = empty S0, hw-generic profile dir = the live profiles converted by the kit function) | 6 failed, 429 passed, 21 skipped | 5 failed, 436 passed, 14 skipped |
| `rigdash/tests` | 3 failed (playwright executable missing), 964 passed, 15 skipped | 3 failed, 957 passed, 15 skipped |
| `rigmon/test_hardware_profile_persist_1006` + `_950` | 3 failed, 52 passed | 3 failed, 47 passed |

The 5 common reds are the live-box state of the planner dump goldens (plan diff 53 lines on both trees); the sixth on the renamed tree is
`test_build_and_shipped_catalog_agree` (shipped `catalog.json` is rebuilt in F0-F).  Kantenkatalog anchors for tree `nf`: 131 edges, 77 unique / 10
near / 44 other line on both trees.  Dry-runs `nf-int4`, `nf-int4-h6-abl` (profconv vs live profile): 0 diff modulo names; both refuse with W128
because the draft files are not visible on this box.  L3 store identity: `l3_persist_identity` / `l3_persist_dir_name` give the same digest on both
trees for the same inputs.
