# ROADMAP #456 — execution plan for the DSV4-Flash matrix sweep

Operational companion to `ANALYSE_456_dsv4f_matrix_sweep.md`. Waves, gates,
and the evidence each item is expected to produce. This is a plan, not a
result: every entry not marked `DONE` is `BOOT-PENDING` or `UNMEASURED` by
default, stated explicitly rather than implied by tense.

## Cross-cutting rules, every wave

* **Eviction doctrine lands in the `#407` registry as a build slice**, not
  ad hoc per feature — `DESIGN_407_memtier_registry.md` §8 is the doctrine;
  any wave item that evicts something (heat migration, KV spill, layout
  switching) consumes it rather than writing a local victim policy.
* **barlink is the default transport everywhere the combination supports
  it** (2026-08-03 user order, `docs/rig-runbook.md` §2); NCCL appears only
  as an explicit control arm or a named fallback with the reason stated.
  Published numbers lead with the barlink row.
* **Every wave ends with merge + `FEATURE_CATALOG.md` update + a SITREP
  entry.** A wave that lands code without updating the catalog is not
  considered closed.
* **Every measured claim carries its own A-vs-A floor** (#360 standard). A
  number with no floor next to it is not a result in this roadmap's sense,
  whatever wave it came from.

---

## WAVE 0 — live now (GPU night window 2)

| item | task # | prerequisite / gate | evidence expected |
|---|---|---|---|
| Video frontier probes: fp16/bf16-TRT SR + RIFE 4.26 + P1 decode/encode rows | #333-M2 | TRT 11.2.1.2 installed (done, `ANALYSE_363...` §"TensorRT rungs") | fills the one unmeasured number in `TASK_333_M2_VIDEO_ENHANCE.md` §16.3's budget table |
| #439 confirmation band | #439 | arm boots without the three defects already fixed at desk (`FEATURE_CATALOG.md`) | confirms or refutes 1.392x/1.450x; `2026-08-03_439_confirm/` is the target directory, empty as of this writing |
| #444a + bs=12/16/24 sweep + spec-off point | #444a | none named beyond a free card window | fills the bs sweep gap noted in prior batteries |
| #447 DSpark spec arm | #447 | DSpark head assembled (`ANALYSE_447_llamacpp_dsv4_harvest.md`) | accept-rate and multiplier measurement on the actual head, replacing the 0.6-0.77/1.5-1.8x estimate in `ANALYSE_456` §6 |
| Optional: #410 gate / #452 arms (B2 control + tolist sizing probe) | #410, #452 | slot availability in the same window | prices the breakable-graph route (`ANALYSE_456` §2.2 #302b) for DSV4-Flash specifically |

BOOT-PENDING in full; nothing in this wave has run as of this document.

---

## WAVE 1 — desk, launches as slots free, in order

| order | item | task # | prerequisite / gate | evidence expected |
|---|---|---|---|---|
| 1 | #302a heat migration — build the falsifier | #302a | none — runs against existing `expert_stats_*.json` (`ANALYSE_456` §2.2 cell 4 path) | desk-computed layer-to-layer top-k correlation and a projected hit-rate lift; GPU arm follows in WAVE 2 |
| 2 | #450 dual-lane collision family | #450 | already running per user note; no new gate | whatever that family's own falsifiers report |
| 3 | #306 lossless cold-tier compression — desk falsification | #306 | none — sample-based, no card (`ANALYSE_456` §2.2 cell 2) | achievable zstd-after-byte-plane-split ratio, per asset type (expert tensors, fp8-KV blocks, GDN blobs, hibernate images), honest verdict per type including "not worth it" |
| 4 | #363 slice 1: worth-it autocheck + rung-0 dual residency | #363 | **this docs merge** (the present task) | `DESIGN_363_regime_controller.md` §20.1/§20.3 move from decided-on-paper to built |
| 5 | #286 graph-rungs offload class | #286 | none beyond design (`DESIGN_363` §20.3 names it as a dependency) | pre-capture prerequisite for #363 slice 3 (weight mover, WAVE 4) |

---

## WAVE 2 — GPU window 3

| item | task # | prerequisite / gate | evidence expected |
|---|---|---|---|
| Heat-migration A/B (hit-rate + decode) | #302a | WAVE 1 item 1's desk falsifier passing | hit-rate delta (0.81 -> target) and decode tok/s delta, both against an A-vs-A floor on the SAME workload |
| #452 sizing probe, if not done in WAVE 0 | #452 | WAVE 0's optional slot, deferred if skipped | VRAM/launch-overhead price of the breakable route for DSV4-Flash |
| FP8xbar1 fresh speed run | #431/#438 unlock | barlink-default order (cross-cutting rule above) | headline barlink row for FP8, current state per `docs/rig-runbook.md` note: "unlocked since #431/#438, fresh speed run pending" |
| 128K/256K long-context arm on the proven HiCache host tier | #441c | HiCache shim-free tier proven (tonight, per `ANALYSE_456` §3) | `max_total_num_tokens` at 128K/256K without additional VRAM, against the current 42240-class cap (flagged as an unverified figure in `ANALYSE_456` §3 — re-confirm on this boot's own plan log) |
| #445 PP=3-vs-TP=3 A/B | #445 | none beyond a free card window; reframed as a disjointness experiment (`ANALYSE_456` §5) | link-contention evidence for/against PP=3 clearing the expert-fetch path, not merely a raw throughput number |

---

## WAVE 3

| item | task # | prerequisite / gate | evidence expected |
|---|---|---|---|
| #302c per-expert runtime dispatch | #302c | the #452 sizing probe (WAVE 0/2) and the #453 CPU-lane pricing (below), both landed | a working load-aware dispatcher; the combine-step seam (`ANALYSE_456` §2.2 #302c) is a build item here, not before |
| #453 remote falsification afternoon + #423 striping probe | #453, #423 | one rig-2 setup, run in the **same window** (`NOTE_453_remote_expert_lane.md` §5) | #453: round-trip-vs-streaming verdict on the compute lane; #423: win-under-load / neutral-without-load verdict per its own can-fail criterion (`DESIGN_423_striped_offload.md` §4) |
| #448 live topology build | #448 | the TRT numbers from WAVE 0 (needs a finished frontier to build a live topology against) | — |
| #451 tipping points, re-read from the new frontier | #451 | WAVE 0's SR/RIFE numbers landed | `TASK_333_M2_VIDEO_ENHANCE.md` §14.5/§16 tipping-point table re-computed on real TRT figures instead of the fp32 ONNX placeholder |

---

## WAVE 4

| item | task # | prerequisite / gate | evidence expected |
|---|---|---|---|
| #363 weight mover + full pre-capture | #363 | #286 (WAVE 1 item 5) landed | `DESIGN_363_regime_controller.md` §20.2/§20.3 built and card-gated per that document's own §11.7-style evidence discipline |
| Remote integration into `expert_compute_placement` | #439 + #453 | WAVE 3's #302c and #453 both landed | the #439 compute-placement mechanism extended to route to the remote CPU lane, not only across local ranks |
| #411 portable sessions | #411 | none named here beyond the never-silent-conversion contract already cited (`DESIGN_261_live_session_handover.md`) | — |
| #412 determinism mode | #412 | — | — |

---

## LOSSY BUCKET — gated at the end, not scheduled in a wave

| item | task # | prerequisite / gate | evidence expected |
|---|---|---|---|
| #126 cold-expert Q2-class quantisation | #126 | **(a)** every lossless gain above landed (WAVE 0-4, in particular #306); **(b)** a quality gate on cold-only-expert accuracy loss, comparable in rigor to the #77/#120 bar already applied to Marlin expert-offload | bytes-per-miss reduction figure AND a quality delta, reported together — a bytes number with no quality number next to it does not clear this gate |

Per the fork's standing quality-last rule, this bucket is deliberately **not**
assigned a wave number: it is picked up only once WAVE 0-4 are otherwise
exhausted, and only with its own quality gate cleared, never on the strength
of its bytes-per-miss estimate alone. See `ANALYSE_456_dsv4f_matrix_sweep.md`
§7 for the reasoning.

---

## Reading this roadmap

Waves are ordered by dependency and by the effort/yield ranking in
`ANALYSE_456_dsv4f_matrix_sweep.md` §7, not by calendar date — "WAVE 0" is
whatever is live in the current GPU window, and later waves shift if an
earlier one's gate is not cleared. An item listed as a prerequisite for a
later wave that has not landed blocks that later item outright; this
document is not to be read as a schedule that proceeds regardless of gates.
