# Spill-night combination matrix — 2026-08-04

Owner: GPU validation lead (agent-spill-night).
Tree: `/spinning/wt-spill-matrix`, branch `feat/spill-matrix-night`,
base `origin/integration/r3-probe-next2` @ `e08946d49e`.

This file is written BEFORE the GPU window. Every cell carries a verdict from
the vocabulary below at all times, so that "nobody looked" is never
indistinguishable from "looked and found nothing" (ANALYSE_456 discipline).

## FINAL TALLY — window closed 2026-08-05 05:56Z

Boots executed: K0 (control), K1 x3, K2 (spill) + K2 (control), A-vs-A, C1 x3.
Serving restored, locks released, holder handed back to SERVING.

**Exercised on hardware and PASSED (8):** H1, H2, H3, H6 · L1 · C1, C3, plus
H5's multi-session-spill half.

**Decided at the desk without GPU (7):** H15-H18 REFUSED-BY-DESIGN, L3 STUB,
L9 BLOCKED, L10 REFUSED-BY-DESIGN, C5 no-production-caller, C6 PLAN-ONLY.

**Honestly open (the point of the vocabulary):** H7 signal PASS but correctness
UNDECIDED-by-instrument; H4, H5-order INCONCLUSIVE; H8 signal N/A for the
config; H9 STILL UNOBSERVED — the one round the spec gate exists for;
H10-H14, L2, L4-L8 NOT-EXAMINED; C2 NOT VERIFIED; C4 NOT EXERCISED;
H19 DEFERRED(TICKET_551).

**The three findings that change what "it works" means:**
1. **S6** — the KV pressure ladder has NO actuator on the production flag set;
   its flips are logged and move nothing. Confirmed in both directions (L1
   PASS shows it actuating once kvso or an admission ceiling is wired).
2. **S14** — byte-equality of generated text is not a valid instrument on this
   rig: two identical loads with zero spills already diverge 6/6. This retired
   a spill-vs-control divergence that looked like corruption.
3. **S8** — #363 `act` mode has zero flip targets by construction, so it can
   never actuate regardless of flags.

**Consequence for #552: the default stays OFF.** The named signal is in the
log; the correctness leg is undecided, and an undecided carries no flip.

## Verdict vocabulary

| Verdict | Meaning |
| --- | --- |
| `PASS` | Exercised on hardware in this window, named proof signal observed. |
| `FAIL` | Exercised on hardware, the named proof signal did not appear or the correctness probe diverged. A defect. |
| `BLOCKED(#n)` | Could not be exercised because another defect stopped it. Ticket named. |
| `REFUSED-BY-DESIGN(cite)` | The combination is refused by a deliberate in-tree gate. Not a defect — the code says no on purpose. Citation is `file:line`. |
| `DEFERRED(reason)` | Deliberately out of scope for THIS window. Named reason. |
| `NOT-EXAMINED` | Nobody looked. Present so the gap is visible. |

Evidence rule: a verdict of `PASS` requires a named observable — a literal log
string, a metric, or an HTTP response field — recorded in the cell's
`RESULTS_<cell>.md`. A boot that merely does not crash is not a PASS.

## Structural findings established at desk (code-verified, this tip)

These are not opinions from the catalog; each was read out of the source in
this worktree.

- **S1 — kvso and HiCache are mutually exclusive.** `server_args.py:6773`
  (`if self.enable_hierarchical_cache or self.enable_unified_memory: raise`).
  The production serving recipe runs `--enable-hierarchical-cache
  --hicache-storage-backend file`, so **kvso cannot run on the production
  recipe at all**. Every HOT cell therefore needs its own boot with the
  hicache flags removed. Composability is task **#547** per
  `docs/rig-runbook.md:877` (the briefing called this #550 — the in-tree name
  is #547).
- **S2 — kvso x speculation is opt-in, not refused.** `server_args.py:6698-6744`
  raises unless `KVSO_ALLOW_SPEC=1`. The refusal text itself states the
  mechanism is built and names the one unobserved round (a spill landing in
  the same round as a drafter-in-tick step). `KVSO_RESUME=1` is a SECOND,
  independent gate for resume-under-spec
  (`managers/kv_session_offload.py:504-530`).
- **S3 — the #330 VRAM dial cannot run on the production recipe either.**
  `managers/vram_dial.py:1040-1086` refuses when `hicache_storage_backend` is
  set, and separately when `enable_kv_session_offload` is set. So #330 is
  disjoint from BOTH the production recipe and the whole HOT arm. It also
  requires WEIGHTED uneven DCP (`vram_dial.py:1110`).
- **S4 — the LEITER arm is almost entirely dark in production.** Read out of
  the running server's own `server_args` dump (boot 22:23):
  `regime_controller='off'`, `enable_vram_dial=False`,
  `kv_reshard_vectors=None`. Only `kv_pressure_ladder='auto'` is live.
  So "the dynamic ladder runs in production today" is **false** for three of
  the four features; only #287 is armed.
- **S5 — the runbook has no kvso boot recipe.** `docs/rig-runbook.md` mentions
  kvso only in refusal lists (`:748`, `:806`, `:877`). Any kvso recipe this
  window validates is new runbook material and must land in the same commit.
- **S6 — the KV pressure ladder has NO ACTUATOR on the production recipe.**
  This is the sharpest finding of the desk phase. `--kv-pressure-ladder auto`
  is live in production, but `wired_relief_features()`
  (`managers/kv_ladder_auto.py:85-109`) returns an EMPTY tuple unless one of
  `--kv-reshard-vectors`, `--enable-kv-session-offload` or
  `--max-running-requests-ceiling` is set — and production sets none of them.
  The table still builds and occupancy is still sampled every round, so rung
  flips are still LOGGED; they just move nothing. The flip line's trailing
  field (`kv_pressure_runtime.py:375-393`) carries `"no actuator declared"`
  in exactly that case. **A validator that greps for `FLIP` alone would report
  working pressure relief where only a counter is incrementing.** So the K1/K2
  recipes deliberately switch kvso on WITH the ladder — that is the one
  configuration in which the ladder genuinely actuates.
- **S7 — the ladder's two rung classes have different truth values.**
  `model_executor/kv_pressure_ladder.py` is the CPU phase by declaration
  (docstring `:2`, `:10-16`: "The table, the sensor, the flip contract and the
  handover INTERFACE -- nothing moves"), with 15 `NotImplementedError` raise
  sites (`:301-500`). But that applies to the `geometry_flip` handover
  strategies. `relief` rungs are explicitly different: they carry handover
  `none` and reference EXISTING features by name (`:23-30`). So:
  **relief rungs can actuate when wired (S6); geometry_flip rungs are STUB and
  move nothing regardless of flags.** Neither "the ladder works" nor "the
  ladder is a stub" is true on its own.
- **S8 — #363 `act` mode has nothing to actuate, by construction.**
  `build_regime_stage_table()` (`managers/regime_runtime.py:885-926`) is the
  only production call site of `planner_candidates()` and calls it at `:911`
  with NO `solve_fn`; `regime_stages.py:350-389` returns `[]` when `solve_fn`
  is None. So the production stage table holds the booted stage and ZERO flip
  targets even with a fully open entry gate. A code-level gap, not a recipe
  one. Also: the `REGIME-OBSERVE` summary line ends `"NOT ACTUATED
  (observe-only)."` **unconditionally, in both modes**
  (`regime_runtime.py:559`) — it must not be used as an actuation signal. The
  `--regime-trace` JSONL `"actuated"` field is the only trustworthy one.
- **S9 — #297 and #330 need DIFFERENT amounts of hicache removed.** #330's
  refusal reads `hicache_storage_backend` (`vram_dial.py:1051`), so
  dropping only `--hicache-storage-backend file` suffices. #297's Stage-A
  guard reads `enable_hierarchical_cache` itself
  (`managers/kv_reshard.py:652-653`), so the whole flag must go. The guards are
  computed ONCE at construction (`:644-672`) and stored, so no runtime toggle
  clears them.
- **S10 — the GDN state-set ladder is PLAN-ONLY, and contains no
  `NotImplementedError` at all.** A Qwen-sourced claim that
  `offload_gdn_states.py:67-70` holds unwired `park()`/`wave_in()` stubs
  raising `NotImplementedError` was checked directly and is **wrong on its
  mechanism**: that file has ZERO such raises. Lines `:67-72` are prose stating
  the GPU phase is unbuilt and "Nothing in this module touches torch.cuda."
  The correct classification is PLAN-ONLY, not STUB-RAISES. (Direction of the
  claim was right; the named mechanism was not. Recorded because a wrong
  mechanism in a matrix is worse than a missing one.)
- **S11 — #286's page mover has no production caller.**
  `AdaptiveGraphStateMover` (`short_term_offload_register.py:1167-1230`) is
  instantiated only in `test/registered/unit/model_executor/test_short_term_offload_register.py:974,982,989`,
  each time with a mock manager. There is nothing to exercise on hardware.
  Separately, the asset-class layer IS reached in production with
  `SGLANG_OFFLOAD_REGISTER` unset, from `layers/moe/breakable_offload.py:216`
  and `:268-272` — already recorded as `AUDIT_500` finding S3-37.
- **S12 — hibernate refuses `load_format='auto'`.** Found at the desk by
  `smoke.sh` step 6, which pushes every recipe through the real validator:
  pointing `--model-path` at a GGUF checkpoint is NOT enough, because the gate
  reads `server_args.load_format` before auto-resolution
  (`server_args.py:13651-13655`). `--load-format gguf` is mandatory. This cost
  zero GPU time to find, which is the entire argument for that smoke step.
- **S13 — the spill-tick decomposition instrument does not exist.** A
  collective/launch/transfer split of the host-decode tick was requested as a
  price input. Searched tree-wide: the kvso env surface is exactly
  `KVSO_ALLOW_SPEC`, `KVSO_RESUME`, `KVSO_S`, `KVSO_S1_VERIFY`,
  `KVSO_ATTN_SELFTEST`, `KVSO_GRAPH_SELFTEST`, `SGLANG_KVSO_DECOUPLE`,
  `SGLANG_KVSO_SPILL_GRAPH`, `SGLANG_KVSO_TICK_TRACE`. There is no
  `KVSO_BLOCK` (the knob is `--kv-session-offload-block-size`, default 8192)
  and `kv_session_offload.py` contains no `cuda.Event` / `elapsed_time` at all.
  The `SGLANG_BREAK_COST_*` probe family is #494's MoE break-cost instrument,
  a different seam. What CAN be delivered from K2 without new code:
  `SGLANG_KVSO_TICK_TRACE=1` emits, per spilled session every 16th iteration
  (`:4465`), the effective interval, measured `tick_cost` and host-tail size
  (`:520-528`) — i.e. **ms/step, but not the three-way split.**

- **S14 — byte-equality of generated text is NOT a valid instrument on this
  rig tonight.** Measured, not assumed: one boot, two IDENTICAL loads against
  the same process (same prompts, concurrency, context, `temperature=0`,
  fixed seed), zero spills — `identical=0 diverged=6 missing=0`, several
  streams differing from character 0. Consequences: (a) the spill-vs-control
  6/6 divergence proves nothing about the spill and would have been a false
  alarm; (b) that comparison was independently confounded, because the ladder
  flipped `admission limit -> 1` in the spill arm only, so batch composition
  differed too; (c) every correctness claim about spill tonight must use
  decode-class bands, not byte identity. Full record:
  `RESULTS_AVA_determinism_floor.md`.

## Boot plan (a boot is the expensive unit; cells are grouped per boot)

| Boot | Recipe sketch | Cells covered |
| --- | --- | --- |
| `K0` | A-vs-A floor on `K1` recipe, no spill forced | floor for every HOT delta |
| `K1` | kvso base, uneven TP3 + uneven DCP, no spec, small KV pool | H1-H5 |
| `K2` | `K1` + NEXTN + `KVSO_ALLOW_SPEC=1` (+ `KVSO_RESUME=1`) | H6-H9 |
| `K3` | `K2` + `SGLANG_KVSO_SPILL_GRAPH=1` | H10 |
| `K4` | `K1` + budgets + adaptive tick + fast lane | H11-H14 |
| `L1` | production-like minus hicache, + dial + ladder + regime + reshard | L1-L6 |
| `C1` | GGUF model + hibernate to disk | C1-C3 |

Pressure is forced by capping the KV pool (`--max-total-tokens`) rather than
by sending 262k-token prompts — this makes a spill reachable in seconds and
keeps every measurement inside the 10-20 s time box.

---

## HOT — KV session offload (kvso)

Boot base for all H cells (hicache removed per S1):

```
--model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8
--tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance
--rank-auto-reserve-mib 5500,3800,3800
--kv-cache-dtype fp8_e4m3 --max-running-requests 4
--enable-kv-session-offload --kv-session-offload-host-ram-gib <N>
--max-total-tokens <small>   # forces pressure fast
--enable-metrics --trust-remote-code
```
env: `SGLANG_UNEVEN_DCP=1 SGLANG_UNEVEN_DCP_WEIGHTED=1 SGLANG_MAMBA_SSM_DTYPE=bfloat16`

| Cell | Combination | Proof signal (literal) | Verdict |
| --- | --- | --- | --- |
| H1 | kvso arms at boot under uneven TP3 x uneven DCP | `kv-session-offload (S4) armed: mode=` (`:2435`) | **PASS** — armed on all 3 ranks, `mode=weighted S=64 prefix=[0, 30, 47, 64]` (non-uniform, so genuinely under uneven DCP). K1. |
| H2 | Spill fires under KV pressure | `kv-session-offload SPILL(partial): rid=` (`:3747`) | **PASS** — K1 run3: 12 lines = 4 spill events x 3 ranks, 2 distinct victims. Partial spill confirmed (`L=1557 boundary=1026 host_tail=531`). |
| H3 | Host-side decode continues while spilled (host-ticking, not parking) | **CORRECTED SIGNAL**: `kv-session-offload restore-gate: iter=N L=N boundary=N` (`:4465`), requires `SGLANG_KVSO_TICK_TRACE=1`. The original choice `tick build: rid=` (`:3895`) was WRONG — it is an error path ("has no output token yet") that a healthy tick never emits, so it would have reported a working feature as broken. | **PASS** — K1 run3, trace present on all 3 ranks. |
| H4 | Wave-back + restore to device | **CORRECTED SIGNAL**: `restored to device` (`:5276`) only. The original second signal `wave-back THRESHOLD armed` (`:2454`) was WRONG — it fires only when `--kv-session-offload-wave-back-min-free-tokens` is NON-default, so at the default 0 its absence carries no information. | in progress — needs a load where a spilled session OUTLIVES the pressure (uniform-length streams all host-tick to completion, so no restore is ever required) |
| H5 | Multi-session FCFS victim order (youngest first, no idleness term) | victim key `(spill_class_rank, fast_lane, -kv_arrival_seq)` (`:867`). **The original regex was too weak** — it matched a single spill and would have reported PASS on one victim, which cannot show an order. | **SPLIT**: multi-session spill = **PASS** (2 distinct victims, `rid=4076a152 seq=5`, `rid=bf447dd1 seq=7`). Victim ORDER = **INCONCLUSIVE**: seq=5 was spilled before seq=7, which is consistent with youngest-first only if seq=7 had not yet arrived at the first decision — and the eligible set at each decision was not recorded. Deciding it needs all candidates admitted before any spill. |
| H6 | kvso x NEXTN spec, `KVSO_ALLOW_SPEC=1` | `kv-session-offload: draft-KV bundle armed` (`:2175`) | **PASS** (K2) — `draft pool size=4096 layer_num=1 head_num=1` on one rank and `head_num=2` on another. The differing head counts are the uneven TP shard, so the draft-KV bundle armed under genuine uneven TP. |
| H7 | Spilled session resumes under spec — now the #552 flag `--kv-session-offload-resume-under-spec` (legacy `KVSO_RESUME` deliberately NOT exported, so the new surface stands alone) | `MTP RESUME seed published: rid=... L=...` | **SIGNAL PASS / CORRECTNESS UNDECIDED** (K2). 12 hits, 2 victims, all 3 ranks: `MTP RESUME seed published: rid=53af3878... L=1143`. `L` grew 1141 -> 1143 across the event, and `finished on host` = 0, so not the host-finish route. NOT corroborated: `restored to device` = 0. Correctness leg undecided — see `RESULTS_AVA_determinism_floor.md`. **#552 default stays OFF.** |
| H8 | Drafter runs inside the spill tick (`--kv-session-offload-spec-in-tick`) | `kv-session-offload spec-in-tick: reserved %d draft-read` (`:2520`) | **SIGNAL N/A FOR THIS CONFIG** — the boot logs `spec-in-tick: --kv-session-offload-mtp-resident-slices is 0 (uncapped)`; the reservation message belongs to the CAPPED path. Signal choice, not a feature failure. Needs a capped-slices boot. |
| H9 | **The named unobserved round**: spill landing in the same round as a drafter-in-tick step (the reason the S2 gate exists) | `kv-session-offload spec-in-tick: rid=%s spill batch armed with` (`:3867`) co-occurring with a SPILL line | **NOT OBSERVED** (K2, count 0) — the one round the S2 gate exists for did not occur tonight either. Honest status: still unobserved, which is exactly why the gate stays. |
| H10 | Spill graph path (`SGLANG_KVSO_SPILL_GRAPH=1`) | `:467/:488`; byte-identity of spilled-vs-device output | NOT-EXAMINED |
| H11 | Spill budget arming + demote-to-HiCache | `kv-session-offload SPILL BUDGET (#236) armed:` (`:2409`), `BUDGET: DEMOTING rid=` (`:3054`) | NOT-EXAMINED |
| H12 | Adaptive/self-calibrating tick cadence | `kv-session-offload: SELF-CALIBRATING spill-tick cadence armed` (`:2464`) | NOT-EXAMINED |
| H13 | Fast-lane interaction with eviction (fast_lane is term 2 of the victim key) | a fast-lane request is NOT chosen while a non-fast-lane candidate exists | NOT-EXAMINED |
| H14 | Prefill-spill / born-spilled (`--kv-session-offload-prefill`) | `kv-session-offload prefill-spill (born-spilled) ENABLED` (`:2201`) | NOT-EXAMINED |
| H15 | kvso x HiCache | — | REFUSED-BY-DESIGN(`server_args.py:6773`) — composability is #547. Measured, not built. |
| H16 | kvso x #330 VRAM dial | — | REFUSED-BY-DESIGN(`managers/vram_dial.py:1052`) |
| H17 | kvso x weightless-KV fastlane | — | REFUSED-BY-DESIGN(`server_args.py:6673`) |
| H18 | kvso x PD disagg / dp>1 / pp>1 / page_size>1 / non-flashinfer | — | REFUSED-BY-DESIGN(`server_args.py:6596-6641`) |
| H19 | GDN vacate x kvso (#551) | park + vacate must BOTH be observed in one run | **DEFERRED(TICKET_551)** — desk fix pushed (`730e09d0ac`); the GPU proof needs a hybrid model, `--kv-session-offload-max-spills 1` to force a park, and a binding `--gdn-resident-state-slots` to force a vacate. **The ticket's rule is recorded here and is binding: a run without an observed park AND vacate proves nothing and must be reported as "path not exercised", never as passed.** Not attempted tonight; the ticket carries everything for the next window. |

## LEITER — dynamic steps between phase optima

Boot `L1` arms all four at once. It is NOT the production recipe and cannot
be: `--hicache-storage-backend file` must go for #330 (S9) and hicache
entirely for #297 (S9). kvso is absent because the dial refuses it (S3).
`--regime-controller observe` only — `act` is unreachable (S8).

| Cell | Combination | Proof signal | Verdict |
| --- | --- | --- | --- |
| L1 | #287 ladder flips a RELIEF rung with a real actuator (needs kvso — so measured in K1/K2, not here) | `KV-PRESSURE-LADDER FLIP rung` with a named relief in the trailing field, NOT `no actuator declared` (`kv_pressure_runtime.py:375-393`) | **PASS** (K1, all 3 ranks): `FLIP rung 0 -> 1 (admission_cap, epoch 1, occupancy 0.531): ... ('admission_cap', relief); handover 'none'. actuator WIRED: admission limit -> 1 (changed=True)`. Confirms the positive half of S6 and, via `handover 'none'`, S7's split between relief and geometry rungs. |
| L2 | #287 ladder on the PRODUCTION flag set (ladder auto, no relief flags) | expected: `FLIP` lines whose trailing field is `no actuator declared` | NOT-EXAMINED — expected to confirm S6 |
| L3 | #287 `geometry_flip` rung actually moves geometry | — | STUB(`model_executor/kv_pressure_ladder.py:301-500`, 15 `NotImplementedError` sites) — CPU phase by declaration, see S7. Not a FAIL. |
| L4 | #330 dial actuates via `POST /vram_budget` | `VRAM-DIAL DONE ... max_total_num_tokens %d -> %d ... released %.1f MiB` (`vram_dial.py:818-831`) | NOT-EXAMINED |
| L5 | #330 `POST /vram_budget {"query": true}` returns live state | response field `state` from `KvCapacityRuntime.status()` (`vram_dial.py:407-434`) | NOT-EXAMINED |
| L6 | #297 reshard at a phase boundary via `POST /kv_reshard` | `KV-RESHARD DONE %s -> %s (epoch %d) in %.1f ms` (`kv_reshard.py:558-576`) | NOT-EXAMINED |
| L7 | #297 reshard is lossless (determinism probe across the flip) | identical greedy output either side of the reshard | NOT-EXAMINED |
| L8 | #363 observe mode emits a per-round verdict | `--regime-trace` JSONL rows; field `actuated` (`regime_runtime.py:409/423`). NOT the log line — S8 | NOT-EXAMINED |
| L9 | #363 `act` mode actuates a stage flip | — | BLOCKED(S8: `regime_runtime.py:911` passes no `solve_fn`, so the stage table has zero flip targets; `regime_stages.py:350-389`). Code gap, ticket owed. |
| L10 | LEITER stack x production recipe (hicache present) | — | REFUSED-BY-DESIGN(`vram_dial.py:1051` for #330; `kv_reshard.py:652-653` for #297) |

## COLD — park / hibernate / restore

| Cell | Combination | Proof signal | Verdict |
| --- | --- | --- | --- |
| C1 | #89 hibernate round trip: park -> process exit -> restore | park / restore lines per rank | **PASS** — 1202 params parked and restored per rank on all 3 ranks, 15 GiB image, `POST /hibernate` -> `{"message":"hibernate: weights parked to disk."}`. Cold 249 s -> restore 193 s = **1.29x**. Post-restore greedy smoke coherent (`Red`). Details: `RESULTS_C1_hibernate.md`. |
| C2 | #456 sparse write, and dense-vs-sparse byte identity | the `sparse write skipped %d of %d 4 KiB pages` clause; sha256 dense-vs-sparse (DESIGN_456 §7) | **NOT VERIFIED** — clause not observed in the park lines and no sha256 arm run. Note the image landed on `/spinning` (ZFS), where DESIGN_456 already measured the byte win as exactly ZERO because compression had taken it; this is the arm where the mechanism is expected to show nothing on this filesystem. |
| C3 | #499/#520 restore identity — the fast restore is REACHED, not silently cold-loaded | `matching manifest found in %s -> load_format='hibernate' (fast restore)` | **PASS** — line observed by name. Every per-rank artifact is keyed by the card's NVML UUID (`rank0_GPU-31d7ef41-...`), i.e. the identity mechanism itself. Exercises the MATCH branch only; the moved-card mismatch branch stays unexamined. |
| C4 | #89 x NEXTN speculation (C1 model carries MTP) | as C3, plus the #520 `context_length` re-derivation | **NOT EXERCISED** — the C1 recipe did not enable speculation. One boot in a later window; the recipe is now correct and reusable. |
| C5 | #286 asset-class park/restore moving real pages | — | DEFERRED(no production caller: `AdaptiveGraphStateMover` is instantiated only in unit tests with a mock manager — S11. Nothing to exercise; not a defect of this window.) |
| C6 | GDN state-set ladder park/wave_in | — | PLAN-ONLY(`model_executor/offload_gdn_states.py:67-72`) — GPU phase unbuilt, module touches no CUDA. See S10; NOT a `NotImplementedError` stub. |
| C7 | #546 translator idle park/wake on a live tenant | `RESIDENCY_EVENT` marker + `park_complete`/`wake_complete` (`translator/residency.py:77-81`) | DEFERRED(no translator tonight — user decision 2026-08-04; GPUs reserved for Qwen/DSV4F) |
| C8 | #568 inv_freq rotary buffers survive a park/restore | — | DEFERRED(needs tenant window). Defect CONFIRMED at desk: `refresh_rotary_buffers` (`translator/qwen3_tts_compat.py:370-411`) has exactly ONE caller, `translator/inprocess_tts.py:217`, on the LOAD path; `AudioAssetLedger.restore()` (`translator/ledger.py:453-521`) does `to_empty()` + `load_state_dict()` and never calls it, so a non-persistent buffer comes back uninitialised. **The "19 buffers" figure is NOT corroborated** — static inspection finds 2 in the loaded talker/code-predictor tree; the count is runtime-logged, not static. Fix site: `ledger.py` ~`:502`. |
