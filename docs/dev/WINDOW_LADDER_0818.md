# WINDOW LADDER 0818 — every pending boot-gated acceptance, one harvest plan

Consolidated 2026-08-18 from: the comp4 gate run (progress.662-F4-r5
06:27-06:50Z), WINDOW_TICKET_745, WINDOW_TICKET_755, NOTE_747 §8/§9,
NOTE_738, NOTE_755, TICKET_727, and the operator ledger. Runner:
`scripts/run_window_ladder.sh` (PASS/FAIL table against a live server;
dry-run proven, never boots anything itself).

REFRESHED 2026-08-18 (post-09:03Z evidence). Provenance of the base, because
two copies of this file exist: the harvest-descended branches all carry blob
`b3c095c39b`, while `docs/merge-train-0818-ledger` carries `746d6a6b34`, which
is the same file plus three refinements the harvest copy lacks (the #540
effort-arms row, the retirement of the SILENT-INERT suspicion on #758-1, and
the #727 arm spec with the ARM-C-is-the-#735-funder note). The LEDGER copy is
therefore the TRUE OPERATIVE VERSION and this refresh branches from it; the
runner script is byte-identical on both (`27dfb2ad5f`), so it has no such split.
Deltas carried in by this refresh: the GATE-A rate bar, the across-a-flip cache
acceptance, the #760 kernel verdict and its rewrite trap, the ARM II flag
precondition, the #743 instrument prerequisite, the host-ledger dirty-page term
as a reported risk note, and the #735 Step-2 precondition status. Every row
cites its source; rows whose figures could not be found in the evidence tree
say so rather than carrying the number.

## Structure: why not one boot

Two hard incompatibilities force arms:

1. **WINDOW_TICKET_745 Arm 1 excludes `--mamba-checkpoint-interval`** (clean
   attribution of the first hicache boot) while the #758 anchor-cadence
   observable REQUIRES it — so hicache-baseline and interval acceptances are
   two arms of one flag apart.
2. **#713 TTFT<3s needs a QUIET router** (ledger 11883: measuring behind
   145k-token agent requests measures the queue, not the seam) while every
   other loaded gate needs the soak backlog — so #713 is a sub-phase BEFORE
   the backlog starts, not a separate boot.

## ARM I — hicache harvest boot (the big one)

Precondition flags: composite tip carrying #756 AND the #757 fix
(F4-r5's armed-downstream send gate — Gate C died at 63s without it),
`--enable-hierarchical-cache` + the WINDOW_TICKET_745 host-tier posts,
`--weight-loader-drop-cache-after-load` (+ the two direct-io flags of the
proven rows basis), file-backed flip images
(`SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED=1` + dir), NO checkpoint interval.

**#760 RESOLVED at the kernel level, and it CHANGES this precondition list:
every hicache boot from here must gate `--hicache-mem-layout
page_first_direct`.** The verdict is PROVEN rather than inferred from silence.
On the `disc` boot (`9ba46eb31a`, the commit that made the guard say it is
armed) the transfer shape guard ARMED 3/3, once per PP seam, and all three
reported MATCHING shapes:
`SPECIMEN-2026-08-18T0903Z-disc-kernel-verdict.log:1245` (`src layer-vectors=[7,
7] dst layer-vectors=[7, 7]`), `:1249` (`[5, 5]`), `:1255` (`[4, 4]`), with zero
`KvTransferShapeMismatch` refusals anywhere in the log. The process segfaulted
anyway: `:21745` `Fatal Python error: Segmentation fault`, traceback inside
`cudaMemcpyBatchAsync` (`:22140-22160`), `:22216` and `:22218` schedulers 1 and
2 killed by SIGSEGV. Matched shapes plus a crash inside the copy is a
kernel-level defect, not a binding or shape defect — the layout is BROKEN ON
THIS RIG and stays out of hicache boots until it is fixed. Do not read
`ARM_disc.log:13` ("guard markers 0") as a contradiction: that is a
monitoring-script artifact, its grep in `arm_boot_735.sh:112` does not match the
literal `KV-TRANSFER-GUARD ARMED` string, and the specimen carries the three
lines.

This collides with the "two direct-io flags of the proven rows basis" above, and
the collision is not hypothetical: `route_a_631_prod_boot.sh` passes BOTH
`--hicache-io-backend direct` and `--hicache-mem-layout page_first_direct`
explicitly, and every arm run so far read back `hicache_mem_layout='page_first_direct'`
with `hicache_io_backend='direct'` — arm1 (`SPECIMEN-2026-08-18T0737Z-arm1.log`),
composite (`SPECIMEN-2026-08-18T0614Z-composite.log`) and disc alike. The harvest
boot therefore cannot keep its current flag set unchanged.

**The server_args rewrite trap — only combinations that NO rule rewrites are
valid escapes.** Dropping the layout flag while keeping `--hicache-io-backend
direct` does NOT escape: `page_first` (the default) plus `direct` is silently
rewritten back to `page_first_direct`
(`python/sglang/srt/server_args.py:16464-16471`, a `logger.warning`, not a
refusal). The reverse rule fires too: `page_first_direct` plus
`--hicache-io-backend kernel` rewrites the IO BACKEND to `direct`
(`:16455-16462`). So the escape must move the IO backend as well as the layout.
The combinations that survive `_resolve_layout_io_compatibility` unchanged are
`layer_first` with either backend, and `page_first` with `kernel`. Read the
effective value back off the boot's own `server_args=ServerArgs(...)` line
before trusting any arm — the runner's `760-layout-gated` row does exactly that.

### Phase 0 — during load (no traffic)

| acceptance | readout | pass |
| --- | --- | --- |
| #738 no-99G-plateau | `free -g` sampled 5s during weights load (runner does this if attached in time; else HOSTMEM csv) | buff/cache peak << 99G; no plateau that survives load end |
| flip-image reclaimability | boot summary line `images ... host (file-backed reclaimable)` | present; disk returns after unlink (already seen once on comp4: 197G->157G->197G) |
| host ledger admits the boot | `[HOST-LEDGER <tag>/pre]` and `/post` lines | verdict `OK`, on the ledger's EARNED terms. The dirty-page transient (`~68.7G`) is REPORTED, NOT CHARGED: the term exists (`evidence-665-f1/host_ledger.sh:81-114`) but `FLIP_DIRTY_GB=0` is hardcoded at `:113`, because summing it failed its own discrimination test on 2026-08-18 -- ARM I (reached health twice) and the layer_first arm (OOM-killed in init) both scored 113.7G (`host_ledger.sh:99-111`, the script's own comment). Charging it refuses a boot that demonstrably runs: the retro replay is `HOST_LEDGER.log:81` (`RETRO-lf-that-died/pre posts=113.7 headroom=-26.7`), and the 08:31-08:33 boots that did charge it read `posts=68.7`. So this row reads the ledger as a RISK NOTE on the dirty window and a GATE on the earned terms -- it is NOT "the transient is part of the gate". |

### Phase 1 — idle, QUIET router (before the soak driver)

| acceptance | readout | pass |
| --- | --- | --- |
| #713 TTFT<3s idle box | timed single small completion against 30030 | TTFT < 3s |
| #540 effort arms (3 requests via 30099) | explicit `xhigh` / omitted / explicit `max` | xhigh: 200 with thinking (post-57b04b2434 pass-through); omit: 200; max on Qwen3.8: honest refusal, never a silent rewrite (deferred here from the desk pass — serving was down) |
| boot health | `/health` 200, one real generation correct | 200 + sane text |
| corridor idle | NVML free per card | >= ~1024 MiB/card target (WARNING below, not fail) |

### Phase 2 — under the soak backlog (the 06:44Z driver is the load source, per the Lastprobe rule)

| acceptance | readout (log grep unless noted) | pass |
| --- | --- | --- |
| GATE A anti-thrash — RATE bar (the 135-flip baseline is RETIRED) | `PHASE-POLICY arming` lines over the policy-live window, expressed as flips/min | **< 1/min = pass; at or above 2.67/min = hard FAIL regardless of backlog** (`NOTE_690_refill_commit_split.md:140-141`). The old "135 flips/15 min" figure is retired as a bar and kept only as the pre-fix reference: at 9.0/min it is 3.4x above the physical ceiling, which means arms were aborting or overlapping rather than completing (`NOTE_690:130-133`). REGIME, and the bar must not bake one in: 2.67/min is `60 / 22.5s`, and 22.5s is the FILE-BACKED leg measured on `boot_735_arm1` (`NOTE_690 §1`, 5 flips x 3 ranks, 22.179-23.559s on rank 0). The marks' own comparison line puts the pinned regime at ~3.1s (`NOTE_690:86-89`, 7.4x cheaper) and `NOTE_690 §5` states the pinned counterpart was NOT measured on that boot. So: recompute the ceiling as `60 / leg_seconds` from the leg the boot under test actually reports, and read the < 1/min bar against that ceiling. A warm-page-cache leg of 2.42s (n=158) was reported at the desk but is NOT in the evidence tree (searched; not found) — recorded UNSOURCED and deliberately not used as the ceiling's basis. |
| GATE A arm mix — WHICH path churns | split the `PHASE-POLICY arming` lines into IDLE-LOCKED vs economic | Churn is expected to be LOCALIZED to the IDLE-LOCK path, and an economic-arm count at or near ZERO is EXPECTED on this workload, not a defect: the soak driver caps each session at 48 000 chars (~12k tok) over 4 sessions, so total backlog sits at or below the ~49 250-token break-even for most of the run (`NOTE_690:134-138`). Structurally, IDLE-LOCK is the one arm that bypasses both dampers — it skips the #688 layout-hold verdict and the #689 formation gate by construction (`phase_policy.py:1320-1326` and `:1347-1354`) — so it is where any remaining churn must surface. #748's idle-lock shape is OPEN AGAIN in effect: the fix shipped on the 08-17 train (`docs/dev/MERGE_TRAIN_PASS2_2026-08-17.md:32`) and the shape still appeared x5 on comp4 (see the #748 shape-2 row below). No register line using the word "reopened" was found; this row cites the recurrence, not a tracker state. |
| GATE B reachability | tp_to_pp arm + `reached pp` + pool usage climb | arms >=1, reached, usage climbs |
| GATE C sustained 300s | scheduler alive the whole window, no traceback | no crash (comp4: CRASH @63s -> #757) |
| #757 race-holds-under-backlog | the #631 PROXY guard sentence | 0 occurrences across >=300s with flips firing both directions |
| #748 shape 1 (no-provider) | `no KV provider` | 0 occurrences (comp4: x9) |
| #748 shape 2 (idle-lock) | `IDLE-LOCKED: no batch of either work class` | 0 occurrences (comp4: x5) |
| #748 vacuous-relief | `relief returned NOTHING before the gate` | 0 on a direction that then commits |
| #744 / #717 rung-funded flip | `KV backing relief returned .* MiB before the gate` | >=1 with nonzero MiB on every committed tp_to_pp |
| #690 refill split | seam census `refill_highwater` mark + per-rank flip timing block | census line present per flip; commit-vs-copy split readable per rank |
| #758-2 mamba host resume | host load_back line for a mamba node (WINDOW_TICKET_745 line 3) | >=1 resume from a host-backed node |
| WT_745 cache reuse — ACROSS-A-FLIP, not absolute hit rate | take the FIRST flip in the harvest log, find the next request from a session that already ran a turn BEFORE that flip, read `#cached-token` on it | `#cached-token > 0` on that request. The absolute hit-rate acceptance ("materially above the 4/121 = 3.3% baseline") is RETIRED AS INVALID: the soak driver truncates each turn to a sliding window, literally `hist[-48000:]` (`evidence-665-f1/soak_driver.sh:15`), so a low absolute hit rate is what that driver produces by construction and says nothing about the cache. The replacement is the #706 determination's own wording — "one number, decidable from the existing cache-report line, not confounded by the 48k window" (commit `9875bafa95`, §4; wired as row E of `WINDOW_TICKET_735_STEP2.md:44`). |
| #743 agent-soak prefix reuse | PREREQUISITE NOT PRESENT — do not schedule this row yet | The 12-slot mamba pool bounds reuse of the 436k-token KV pool (the match walk advances only at nodes owning a recurrent state and cuts the returned KV indices back to that node), so the question is real; but it is unanswerable from logs today because no discrete slot-eviction event is logged at all — `NOTE_743_mamba_slot_hitrate.md §2.2` ("Instrumentation gap (ABSENT, not zero)") and §4.1, which proposes the counter and log line at `mamba_radix_cache.evict_mamba:1161`. The instrument is PROPOSED ONLY: commit `cc4ac02321` is docs-only (one file, `FEATURES_VS_UPSTREAM.md`, +17 lines) and its own message says "Determination only, no code change". `NOTE_743:213-216` states the ordering explicitly — the soak needs the §4.1 instrument in place FIRST, "since without the instrument the soak would produce the same unanswerable logs". Prerequisite, not a row that can pass or fail. |
| WT_745 ledger post | hicache host post vs the sized 3591 MiB | post present and within the ledger |
| corridor minima under load | NVML free per card, 2s sampling, window minima | all cards above OOM floor (comp4 PASS: 2359/2476/1763) |

### Phase 3 — teardown

| acceptance | readout | pass |
| --- | --- | --- |
| image files reclaimed | `/spinning` df + leftover `flip-image-*.img` count | disk back, 0 files |

## ARM II — interval harvest (ARM I flags + `--mamba-checkpoint-interval 8192`)

Only after ARM I accepts (one-flag delta keeps attribution clean; #750
makes it a pure config flip — chunk budget stays 512).

**PRECONDITION, checked before any ARM II row is read: the flag must actually
be on the boot.** ARM I ran with `mamba_checkpoint_interval=None`
(`SPECIMEN-2026-08-18T0737Z-arm1.log:18`), which is correct for ARM I and is
exactly why this precondition exists — `_handle_mamba_checkpoint_interval`
returns immediately when the value is `None`
(`python/sglang/srt/server_args.py:14322-14324`), so an ARM II boot that forgot
`--mamba-checkpoint-interval 8192` is byte-identical to ARM I and every row
below would read UNOBS forever while looking like a measurement. This is the
#742 silently-inert-flag class. Read `mamba_checkpoint_interval=8192` off the
boot's own `server_args=ServerArgs(...)` line first; if it is `None`, the arm is
NOT ARM II and its rows are void, not failed.

| acceptance | readout | pass |
| --- | --- | --- |
| #758-1 anchor cadence | anchor-cadence lines (comp4: 0 lines — F4-r5 is building the emitters) | anchors at exact 8192-multiples, every 16th chunk end. SILENT-INERT suspicion RETIRED at the desk: the write path beneath is proven alive link by link (fix/745-anchor-reachability — grid decision at exactly the 16th/32nd chunk ends, donation → retained node → BACKUP_HOST transfer, dead-grid mutant reds the drive); the emitter is the ONLY outstanding piece |
| NOTE_747 §8.1 composed boot serves | health + generation with interval x hierarchical | serves |
| NOTE_747 §8.2 anchor survives to host + resumes | interval-position load_back line | >=1 at an 8192-multiple |
| NOTE_747 §8.3 churn determinism | identical request resumes at same anchor after device eviction | same resume point |

## SEPARATE windows (cannot ride the harvest boots)

| item | why separate | form |
| --- | --- | --- |
| #727 three-arm quality A/B | 4 boots (A1, A2, B, C), only --model-path differs | `tools/ab_vocab_int8_727.py`, own window. Arms (models-cache root): A1/A2 `Qwen3.8-27B-INT8-yarn1.5`, B `Qwen3.8-27B-INT8-vocabint8-embed`, C `Qwen3.8-27B-INT8-vocabint8-both`; all other flags = the composite recipe unchanged (spec ON so the accept_len readout fires). GATE 0 = `INT8-VOCAB ENGAGED` count 0/0/1/2; GATE A = PP0 −1212 MiB (B, C), PP2 −1212 MiB (C only); GATE B incl. optional accept_len vs the A-vs-A floor; decision rule pre-written in TICKET_727. ARM C IS THE #735 FUNDER (lm_head sits on the 5090; slots 21-24 need its 1212 MiB). |
| WINDOW_TICKET_755 slots A/B | 12/4 vs 6/2 changes pool geometry (confound for everything above) | 2 boots, two flags |
| #755 metal retraction proof | needs the NOTE_755 §2 lock reorder — NOT BUILT; nothing to measure yet | future |
| #709 uneven-TP A/B | own boot pair per its arm spec (opportunistic per ledger 11866) | owner: F4 lineage |
| #735 Step-2 gapped layout | new PP cut (48 GDN -> 5090) = new boot | after Step-1 on the harvest boot if cheap. PRECONDITION STATUS: feat/753 (the crossing wire + gapped-refusal lift + the #754 fold) is **PREPARED(branch): `fix/753-on-harvest@8f2094f62b`** — all three #753 commits cherry-picked clean onto tip `b7e6a4110b`, zero conflicts, distributed selection 105/0 against the tip's 75/0 baseline, byte-gates + the 31-crossing pin + the 3-process gloo progress test green (`WINDOW_TICKET_735_STEP2.md:54`, ledger commit `1e5a4ba9ff`). PREPARED means a branch exists and is green, NOT merged: the executor's precondition list empties only once F4-r5 merges that one branch. The gapped layer set is legal only with #753 on the tip, and #753 is also what provides the wire — one precondition, two reasons (`WINDOW_TICKET_735_STEP2.md:32-34`). |
| #602 instrument lines | detail held by its owner (not recoverable from the register at desk today) — slot reserved, runner has a placeholder check | owner fills the grep |
| #536/#537 | not found in the swept sources with acceptance shape — explicitly NOT consolidated rather than invented | owners to file |
| #470 Boot A/B(/C) | own residency-cut pricing + spec config; three comparable boots in one window | TICKET_470 §3/§6 (#535: Boot B unblocked, pure flags; Boot C carries a NAMED EAGLE-publish refusal hazard) |
| #462-F2 breakable route | breakable-vs-eager needs its own capture boots | TICKET_462_RESULT §4 (#535: LPO branch landed, pure flags; first hardware verdict on the 43-crossings figure) |

## Runner

`scripts/run_window_ladder.sh --log <bootlog> [--url http://127.0.0.1:30030]
[--phase idle|loaded|all] [--arm I|II] [--leg-seconds 22.5]
[--dry-run <fixture-dir>]` — evaluates every grep/curl row above and prints a
PASS/FAIL/UNOBS table. It never boots, never kills, never touches the soak
driver; #713 runs only with `--phase idle` (the runner cannot verify router
quiet — the operator asserts it by choosing the phase).

`--leg-seconds` is how the regime stays out of the bar: GATE A's hard ceiling
is computed as `60 / leg_seconds`, not stored as a constant. The default 22.5
is the file-backed arm's measured leg; pass the boot's own number when it
differs.

GATE A is evaluated on two measures. SUSTAINED = arms over the policy-live
window (first `PHASE-POLICY armed:` line to the last timestamped line), against
the < 1/min bar. PEAK = the most arms in any 60 s window, against the ceiling —
a burst at or above the ceiling means more arms than the seam can physically
complete, i.e. arms aborting or overlapping, which is the 135-flip diagnosis
rather than the 135-flip bar. Only PP0 emits the arming line, so one line is one
arm. The arm mix (idle-lock vs economic) is reported and never fails.

Dry-run fixture: `/spinning/evidence-665-f1/ladder-fixture-0818/log.txt`, built
from the 0903Z disc specimen. Smoke of the refreshed runner against it, `--arm
II --leg-seconds 22.5`: PASS=7 FAIL=7 UNOBS/DRY=7. The seven FAILs are real
verdicts on that boot, not fixture artifacts —

* `760-layout-gated` FAIL: the disc boot's effective layout really was
  `page_first_direct` with `direct` io, i.e. the gate would have refused the
  boot that then segfaulted;
* `760-guard-armed` PASS with 3 ARMED / 0 mismatches — the row that makes the
  segfault a kernel verdict instead of an unread guard;
* `armII-precondition-interval` FAIL: `mamba_checkpoint_interval=None`, so the
  runner correctly refuses to treat that boot as ARM II;
* `gate-a-flip-rate` FAIL on the burst measure (3 arms in 57 s = peak 3 against
  a 2.67/min ceiling) while sustained is only 0.39/min — the two measures
  disagreeing is the intended resolution, not a bug;
* `706-cache-across-flip` FAIL: 42 prefill lines on that boot, zero with
  `#cached-token > 0`, so across-a-flip reuse is genuinely UNMET there;
* the three #748 shape FAILs, unchanged from the previous runner.

Regime can-flip check (the proof the bar is not baked in): the same fixture with
`--leg-seconds 3.1` (the pinned-regime figure) reprices the ceiling to 19.35/min
and `gate-a-flip-rate` turns PASS. The knob changes the verdict, so it is load-
bearing rather than decorative.

Can-pass check on the three new binary rows, so none of them is a row that can
only ever fail: a mutant of the same fixture with the layout string changed to
`layer_first`, the interval to `8192`, and the post-flip `#cached-token` to
nonzero turns `760-layout-gated`, `armII-precondition-interval` and
`706-cache-across-flip` all PASS (10/4/5). Each row therefore discriminates.
