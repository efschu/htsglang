# WINDOW LADDER 0818 — every pending boot-gated acceptance, one harvest plan

Consolidated 2026-08-18 from: the comp4 gate run (progress.662-F4-r5
06:27-06:50Z), WINDOW_TICKET_745, WINDOW_TICKET_755, NOTE_747 §8/§9,
NOTE_738, NOTE_755, TICKET_727, and the operator ledger. Runner:
`scripts/run_window_ladder.sh` (PASS/FAIL table against a live server;
dry-run proven, never boots anything itself).

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

### Phase 0 — during load (no traffic)

| acceptance | readout | pass |
| --- | --- | --- |
| #738 no-99G-plateau | `free -g` sampled 5s during weights load (runner does this if attached in time; else HOSTMEM csv) | buff/cache peak << 99G; no plateau that survives load end |
| flip-image reclaimability | boot summary line `images ... host (file-backed reclaimable)` | present; disk returns after unlink (already seen once on comp4: 197G->157G->197G) |

### Phase 1 — idle, QUIET router (before the soak driver)

| acceptance | readout | pass |
| --- | --- | --- |
| #713 TTFT<3s idle box | timed single small completion against 30030 | TTFT < 3s |
| boot health | `/health` 200, one real generation correct | 200 + sane text |
| corridor idle | NVML free per card | >= ~1024 MiB/card target (WARNING below, not fail) |

### Phase 2 — under the soak backlog (the 06:44Z driver is the load source, per the Lastprobe rule)

| acceptance | readout (log grep unless noted) | pass |
| --- | --- | --- |
| GATE A anti-thrash | `PHASE-POLICY arming` lines during 8 small requests | 0 lines (comp4: FAIL with 2) |
| GATE B reachability | tp_to_pp arm + `reached pp` + pool usage climb | arms >=1, reached, usage climbs |
| GATE C sustained 300s | scheduler alive the whole window, no traceback | no crash (comp4: CRASH @63s -> #757) |
| #757 race-holds-under-backlog | the #631 PROXY guard sentence | 0 occurrences across >=300s with flips firing both directions |
| #748 shape 1 (no-provider) | `no KV provider` | 0 occurrences (comp4: x9) |
| #748 shape 2 (idle-lock) | `IDLE-LOCKED: no batch of either work class` | 0 occurrences (comp4: x5) |
| #748 vacuous-relief | `relief returned NOTHING before the gate` | 0 on a direction that then commits |
| #744 / #717 rung-funded flip | `KV backing relief returned .* MiB before the gate` | >=1 with nonzero MiB on every committed tp_to_pp |
| #690 refill split | seam census `refill_highwater` mark + per-rank flip timing block | census line present per flip; commit-vs-copy split readable per rank |
| #758-2 mamba host resume | host load_back line for a mamba node (WINDOW_TICKET_745 line 3) | >=1 resume from a host-backed node |
| WT_745 hit-rate lift | cache-report hit rate vs the 4/121 baseline | materially above 3.3% on re-sent scaffold prefixes |
| WT_745 ledger post | hicache host post vs the sized 3591 MiB | post present and within the ledger |
| corridor minima under load | NVML free per card, 2s sampling, window minima | all cards above OOM floor (comp4 PASS: 2359/2476/1763) |

### Phase 3 — teardown

| acceptance | readout | pass |
| --- | --- | --- |
| image files reclaimed | `/spinning` df + leftover `flip-image-*.img` count | disk back, 0 files |

## ARM II — interval harvest (ARM I flags + `--mamba-checkpoint-interval 8192`)

Only after ARM I accepts (one-flag delta keeps attribution clean; #750
makes it a pure config flip — chunk budget stays 512).

| acceptance | readout | pass |
| --- | --- | --- |
| #758-1 anchor cadence | anchor-cadence lines (comp4: 0 lines — F4-r5 is building the emitters; SILENT-INERT suspicion stands until they exist) | anchors at exact 8192-multiples, every 16th chunk end |
| NOTE_747 §8.1 composed boot serves | health + generation with interval x hierarchical | serves |
| NOTE_747 §8.2 anchor survives to host + resumes | interval-position load_back line | >=1 at an 8192-multiple |
| NOTE_747 §8.3 churn determinism | identical request resumes at same anchor after device eviction | same resume point |

## SEPARATE windows (cannot ride the harvest boots)

| item | why separate | form |
| --- | --- | --- |
| #727 three-arm quality A/B | 4 boots (A1, A2, B, C), only --model-path differs | `tools/ab_vocab_int8_727.py`, own window |
| WINDOW_TICKET_755 slots A/B | 12/4 vs 6/2 changes pool geometry (confound for everything above) | 2 boots, two flags |
| #755 metal retraction proof | needs the NOTE_755 §2 lock reorder — NOT BUILT; nothing to measure yet | future |
| #709 uneven-TP A/B | own boot pair per its arm spec (opportunistic per ledger 11866) | owner: F4 lineage |
| #735 Step-2 gapped layout | new PP cut (48 GDN -> 5090) = new boot | after Step-1 on the harvest boot if cheap |
| #602 instrument lines | detail held by its owner (not recoverable from the register at desk today) — slot reserved, runner has a placeholder check | owner fills the grep |
| #536/#537 | not found in the swept sources with acceptance shape — explicitly NOT consolidated rather than invented | owners to file |

## Runner

`scripts/run_window_ladder.sh --log <bootlog> [--url http://127.0.0.1:30030]
[--phase idle|loaded|all] [--dry-run <fixture-dir>]` — evaluates every
grep/curl row above and prints a PASS/FAIL/UNOBS table. It never boots,
never kills, never touches the soak driver; #713 runs only with
`--phase idle` (the runner cannot verify router quiet — the operator
asserts it by choosing the phase). Dry-run proven against a fixture built
from the comp4 specimen lines (reproduces the comp4 verdicts: GATE A FAIL,
#748 FAILs, #744 PASS, cadence UNOBS).
