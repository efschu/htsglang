# fLLiper 0.1.0: release notes (state of the integration branches, F0-I, 09.10.2026)

Scope: the rename of htsglang to fLLiper and what the release contains. Every number below comes from a work-package report
(F0-D .. F0-M, F0-I) or from a test run named in it; "unproven" means nobody ran it. Nothing here has been published: the push to
`github.com/efschu/fLLiper`, the image build on the build host, the push to `ghcr.io` and the acceptance boots (F0.5) are user gates.

## 1. What 0.1.0 is

| | |
|---|---|
| Name | fLLiper (derived from SGLang, Apache-2.0). Python package `flliper`, environment `FLLIPER_*`, P/D-flip subsystem `pdflip` (`FLLIPER_PDFLIP_*`, `--pdflip-*`, markers `PDFLIP-*`) |
| Purpose | LLM inference on mismatched GPUs: heterogeneous tensor parallelism, prefill/decode flipping on one box (`pdflip`), profile planner, dashboard |
| Lines | two code states in one image: **27B** (`desk/flliper-27b-int-1008`) and **Flash-Next / NF** (`desk/flliper-nf-int-1008`) |
| Image | Duo image `ghcr.io/efschu/flliper:0.1.0-cu130` (and `flliper:cu130-<sha10>`), labels `io.github.efschu.flliper.*`, source `https://github.com/efschu/fLLiper`; built by `docker/flliper/make_flat_ctx.sh` + `Dockerfile.flliper`, gated by `host_publish_flliper.sh` |
| Predecessor | htsglang; the published tag `htsglang:cu130-nccl2307` is untouched |

## 2. What the rename did (RENAME_PLAN sections 1, 2, 8)

* `sglang` -> `flliper` (package, imports, process titles `flliper::scheduler_TP0`, torch op namespace, cache directory), `SGLANG_*` ->
  `FLLIPER_*`, `weg2` -> `pdflip` (module directory `srt/pdflip`, flags, env, markers, endpoints), German identifiers and prose -> English.
* Compatibility (RENAME_PLAN 4, F0-B/F0-C): the environment mirror reads both spellings (the new one wins, a difference warns, one
  deprecation line); the launcher parser accepts `--weg2-x` and `--pdflip-x`; flip-front routes answer under both prefixes; the rig state
  directory reads across `~/.cache/sglang` and `~/.cache/flliper`; our tools (host acceptance, monitors, log parsers, the dashboard) read old and new
  markers and log stems. **No Python import alias** `sglang` -> `flliper` (plan row F0-C, not built).
* Must-keep (RENAME_PLAN 2): licence and attribution lines, URLs and upstream ids, foreign packages, the kernel wheel `sgl_kernel`, C/C++/CUDA
  sources, host paths `/spinning/htsglang*`, evidence names (boot tags `weg2xsn246`, `boot_weg2_*` logs, `/spinning/gpu-arb/weg2`),
  persisted format ids (`weg2-footprint/1`, `weg2-x-curves/1`, `weg2.form_measures/N`), the product environment `HTSGLANG_*`.
* **Metric names are must-keep** (user decision 08.10.2026 "pdflip as the name, the metrics stay", F0-M, RENAME_PLAN 8.17): the engine's
  `sglang:*` / `sglang_*` and the flip subsystem's `weg2_*` (front, rank, boot, GPU, Influx `weg2_flip`, label `weg2_group`, scrape job
  `weg2-front`) keep their spelling, so VictoriaMetrics history and Grafana panels continue; the dashboard readers select both stems
  (`{__name__=~"(weg2|pdflip)_x"}`, RENAME_PLAN 8.18).
* Profiles: the release profiles are converted to the new spelling in the tree (`docker/flliper/profiles_release/*.env`,
  `profiles/nf-int4-h6-abl.env`); the old files stay next to them as `*.env.alt` for comparison and fall-back (F0-G). The live profile
  directories of the rig are **not** converted yet (switch-over, operator).

## 3. Components and verification numbers

Numbers of the work-package reports (before the integration):

| Package | Result |
|---|---|
| F0-D 27B (`8aaa67e72c`) | verify PASS, imports PASS; kit test set old = new (26 failed / 1193 passed); dashboard 966 passed / 3 skipped / 13 (the 3 skipped are Playwright, as on the old tree); planner goldens regenerated from the renamed dump 47 / 70 / 53 = the old ones; persisted `features.json` / `hardware.json` keep their spelling, the code reads both (280 JSON files audited) |
| F0-E NF (`ffea1c00ff`) | R1 by AST comparison: 759 files, 0 structure differences; kit test set 10 failed / 1162 passed on both trees; dashboard 964 passed / 3 skipped; edge anchors 77 / 10 / 44 on both |
| F0-F 27B (`33006acfb2`) | catalog union (+10 entries, edge anchors 119 / 12 / 0), rank-record double reading (`records/pdflip` + old), metric double reading in PromQL, state dir `/var/lib/flliper` with fall-back `/var/lib/rigdash`, endpoint aliases `/weg2` and `/pdflip`; dashboard 1006 passed / 1 failed / 3 skipped (the one: live NF profile with old env names) |
| F0-F NF (`021e9ebcbf`) | catalog built once from both renamed trees; dashboard 1010 / 0 / 5; group `pdflip` 247 / 0 / 10 |
| F0-G (`585c557cf4`, `3c4c119501`) | profiles converted (`*.env.alt` self-contained); dry-run comparison old vs new over 16 profiles: 16 OK, 0 FAIL (13 compared up to the launcher's first refusal because model / draft files are invisible on the test box, 1 up to W64, 2 complete); container files (`Dockerfile.flliper`, `make_flat_ctx.sh --check`, publish gate with the label check `io.github.efschu.flliper.revision` == OCI revision) |
| F0-H / F0-H2 (NF `10198b0310`, `88c368f44d`) | 19 post-freeze fix commits ported by the rename rule (0 conflicts, 43 delta files byte-equal to the pipeline output), then 3 more ports (D-anchor `9a3cb0c415`, PR-ARENA `f6e38f086a`, W3-HOST-LEAF `554388cb3a`) and the catalog layer (5 explained entries, edge K132); 8 delta files byte-equal |
| F0-M (`e6fc27c6d2`, `af0e4e17f3`) | metric names restored; 159 `sglang:*` tokens of the old tree present; in scratch merges rigdash 989 / 0 (27B) and 992 / 0 (NF) |

Integration (F0-I): one integration branch per line, merged in the order F0-M, F0-F (recipe 8.18), F0-G, NF: F0-H2; the final catalog is one build from both heads (sha256 `6b8a4fb9e00c4d6a633284e2cc327c06034a8d6f229535c9786e3a4613e804f4` on both lines, 2704 entries (862 flags, 1827 envs): 119 curated, 309 explained, 1674 harvested, 602 unexplained; 132 edges); gates, reds and their reading: `docs/dev/RENAME_PLAN.md` 8.19. Rest inventory of old names: 0 unexplained hits on both lines (`deskq/done/f0i-rest-inventar-1008.md`).

## 4. Fixes after the freeze (`deskq/FIXES-NACH-FREEZE-1007.md`), NF line

The 27B line has no post-freeze fixes. On the NF line the tree contains, ported by the rename rule (F0-H, F0-H2):

* **H98e**: root cause behind the D death of 07.10. 21:37Z (form A admission: room first).
* **H110**: the outage of 08.10. 05:53Z was a prefill OOM on all three D ranks (eviction under-delivered), not a flip stall; the admission now counts all rows of the pass.
* **#580 / #791b**: grammar requests entered the D queue only after the verdict drain and stopped all D ranks (marker `#580 GRAMMAR-INTAKE PENDING`).
* **W98** host-rate-latch measurement fix.
* **CAPPARK-FLIP-HOLD** (`FLLIPER_PDFLIP_ENABLE_CAPPARK_FLIP_HOLD`): the capacity re-queue rests while a flip park is open. **user_flipzeit** instrument per the user's definition (start = the later of the last D token and the waiter's arrival; "start unproven" instead of a substitute number).
* **D queue head** (int18 decode dip): one free base for SEAT-AGE and form A, an unservable D head goes to P, re-evaluation only on a state change (markers `D-HEAD-HOLD`, `D-WALL-HEAD`, `W50-REROUTE`).
* **UD-H**: P rank death "prefill out of memory / eviction under-delivered": host-only children are evicted before dropping.
* **PP-ROOM-VOTE** (int21): one agreed room count per pass across the P stages (`PR PP-ROOM-CAP`, `PR AGREED-ROOM SHORT`).
* **W3 spill anchor pool** (`FLLIPER_PDFLIP_ENABLE_W3_SPILL_ANCHOR_POOL`, default off) and the three int23 parts: Mamba last-resort (`FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT`, default off), PR-ARENA (`FLLIPER_PDFLIP_ENABLE_PP_ROOM_ARENA_FREE_ROOM`, default on), W3 host leaves (`FLLIPER_PDFLIP_ENABLE_W3_SPILL_HOST_LEAVES`, default off).

Metal status of these fixes is the NF seat's (list in `FIXES-NACH-FREEZE-1007.md`); this document claims none of them as proven on metal.
Not in these heads: fixes after int23, among them the H88 W4A8 work with its scale fix `dec66fb0b4` (the NF tree has no `marlin_a8` files) and the room-short repro; a second port round is open.

## 5. Known limits and open points

1. Live profiles under `/spinning/gpu-arb` are not converted (`profconv.py --convert-live --apply`, after the gpu-arb patch `f0b-gpu-arb-1007-v2`); tests that read the live profiles stay red until then.
2. Dashboard deploy (`install.sh`, `install_510.sh`), image build from the integration heads on the build host (`make_flat_ctx.sh`, `Dockerfile.flliper`), F0.5 acceptance boots, the push to `efschu/fLLiper` and `ghcr.io`: seats and user gates.
3. Product layer ("phase 2b" of section 5, the lower-case `htsglang` product name: units `htsglang-*.service`, `/etc|/var/lib|/opt/htsglang`, `x-htsglang` API namespace, `docker/htsglang*` compose / Dockerfiles, volume names): not renamed in F0. Counted as findings in the inventory; the release image carries `/opt/htsglang/src-{27b,nf}` and the `SGLANG_WEG2_*` variables the double-reading entrypoint expects.
4. Prose pass (Variant B): comments and docstrings, mixed language left by `ident_fix`, root notes (`FEATURES_VS_UPSTREAM.md`, `HANDOVER_760.md`, ...), the draft release README `docker/pdflip-release/README_RELEASE_DRAFT.md`.
5. Default decision (user): `FLLIPER_PDFLIP_ENABLE_W3_SPILL_HOST_LEAVES` and `FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT` are off; the metal evidence is the NF seat's (`FIXES-NACH-FREEZE-1007.md`).
6. Second NF port round for fixes after int23 (not in the NF head: the H88 W4A8 work with its scale fix `dec66fb0b4`, the room-short repro).
7. No `sglang` -> `flliper` import alias (plan row F0-C).
8. Dry-runs end at the launcher's first refusal on this box (W61 / W128 / W163, W64 on one profile): what lies behind it is compared on the host in F0-J.
9. Known red tests, not touched here: `test_rename_collision_free_0928` (209 files carry old and new names, also on the base), 7 `test_webui` cases (404), P1b reference tests with the live profiles, `test_plan_parser`, `flipzeit_dp_user_0930` (a source finding, red on int22 too), Playwright tests (module missing).
10. Small items of the work packages: `make_flat_ctx.sh` takes `entrypoint.sh` from `/spinning/gpu-arb/docker` (live copy of 03.10.), not the tree master; `profconv.py --convert-live --apply` does not check siblings; the dashboard's `StateDirectory` must stay a relative path; the card-plan screen shows GATE-FEHLT on the desktop screenshot; a stray empty tracked file `0` at the repository root; two vendored files (`_vendor/rife`) name the old module path in a comment.
