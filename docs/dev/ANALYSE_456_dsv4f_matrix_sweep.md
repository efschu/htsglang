# ANALYSE #456 — the DSV4-Flash Asset x Tier x Primitive x Control matrix

Desk analysis, no cards. Goal: decode 7.15 -> >18.4 tok/s on this rig for
DeepSeek-V4-Flash (factor >= 2.6x), the reference baseline being
`BENCH_394_v4flash_club3090.md`'s measured 7.10-7.15 tok/s narrative/code
decode figure. This is a systematic sweep of what moves that number and what
does not, cross-referenced against the fork's existing analyses rather than
re-derived from nothing. **Every number below not attributed to a specific
boot log is an estimate or a prediction, named as such.**

## 1. Why a matrix, not a list

The fork's own DSV4/V4-Flash work is spread across `ANALYSE_393_ik_llama.md`,
`ANALYSE_389_nvme_expert_tier.md`, `FEATURE_CATALOG.md` §"MoE" and
`BENCH_394_v4flash_club3090.md`, each addressing one lever. Read together they
name four assets, each with several occupied and several unoccupied
(tier, primitive, control) cells. Naming the unoccupied cells explicitly is
the point of this document — a cell nobody has looked at reads identically to
a cell somebody checked and rejected until it is written down which one it
is.

## 2. Asset EXPERTS (~100 of 119 GiB, the dominant asset)

The expert weights are the large majority of the checkpoint's footprint and
the dominant term in decode latency (host-tier misses dominate the per-token
cost per `ANALYSE_393_ik_llama.md`'s own arithmetic). Occupied and newly named
cells:

### 2.1 Occupied cells (built or measured elsewhere, cited not restated)

* **RAM tier + residency fraction** (#77/#123 expert offload) — the base
  mechanism: prefetch, presplit, resident-fraction placement.
* **Prefetch / double-buffer** (#125) — overlaps the host-tier fetch with
  compute so a miss is hidden rather than serialised.
* **Link-proportional BYTES** (#394) — `SGLANG_MOE_HOST_SHARD_RATIO`, sizes
  each rank's cold-expert share by measured H2D bandwidth rather than an
  equal split (the ANALYSE_393 §7.3/§7.4 145 -> 86 ms/token arithmetic this
  document's §5 cut-2 entry in `DESIGN_407_memtier_registry.md` also cites).
* **Link-proportional COMPUTE** (#439) — moves the *compute assignment*, not
  only the bytes, onto the #82 expert-range mechanism. **Status: CONFIRMED
  2026-08-03, corridor-red.** The window
  (`/spinning/gpu-battery-results/2026-08-03_439_confirm/RESULTS.md`) measured
  the clock rank at **1.4253x** on the transfer term, read work-matched off
  both arms' final dump revisions, against the 1.411x its own model predicted
  for the resolved base plan `30407,19080,19080`, and **-7.67 %** end-to-end
  against a same-window floor of 4.09 %. (The 192.7 s -> 128.8 s = 1.496x this
  paragraph carried is the PRE-TEARDOWN revision, ~5 % high because the two
  arms' dumps land at different fractions of their runs; withdrawn per
  #482/#523.) The calibrated sub-arm was FALSIFIED in the same window on its
  END-TO-END leg (-0.94 %, inside the floor) and on its mechanism — NOT on the
  transfer term, where work-matched it reads 1.4573x and slightly wins: a
  per-rank cold-traffic coefficient treats the
  hit rate as a rank property and it tracks the owned range SIZE. Two things
  keep the cell from being closed: every arm ran outside the 400 MiB corridor
  (3080s at 211-251 MiB during load), so one green-corridor re-proof is owed
  at the repaired reserve `2200,1800,1800`; and `--rank-auto-reserve-mib auto`
  is infeasible on this recipe. Spec: `ARM3_COMPUTE.md`, "Green-corridor
  window".
* **Expert-major prefill waves** (#254) — orders prefill's expert dispatch by
  expert rather than by token batch, a prefill-side lever distinct from the
  decode-side levers above.
* **Remote-CPU lane** (#453) — see `NOTE_453_remote_expert_lane.md`: a
  compute lane, not a bandwidth lane, and hard-gated on sm75 for the 2080 Ti
  arm.
* **DDR-contention striping** (#423) — see `DESIGN_423_striped_offload.md`:
  applies here when a local-RAM expert fetch competes with other local-DDR
  traffic; the disjointness has to be checked against the *contended*
  resource, not assumed from "two different links."

### 2.2 Newly named cells (not built, not previously written down as a cell)

**(1) Dynamic heat migration (#302a).** #302 (dynamic expert placement) exists
as a registered extension (`ANALYSE_363_dynamic_regime_controller.md`: "for
MoE the mover largely EXISTS ... #302 dynamic placement is the registered
extension") but is pending, not built. The distinction this cell names: the
measured **0.812 activation-grain hit rate** (`ANALYSE_389_nvme_expert_tier.md`
§"0.812 activation grain (0.764 / 0.836 / 0.837), 0.620 unique grain") is a
**static, load-time** placement decision — the resident set is chosen once,
at load, and never re-ranked against what the router actually sends at
runtime. #390's live router-stats collection (already wired for
diagnostics — see `INCIDENT_394_sigusr2.md`, `FEATURE_CATALOG.md`,
`NOTE_443_v4_fetch_desync.md`) is the missing ingredient to re-rank the
resident set against live traffic. Moving the hit rate from 0.81 toward
0.9+ roughly halves the miss-byte volume (the miss fraction goes from 0.19 to
0.1, a 1.9x reduction in the term that dominates decode latency). This is the
**largest unused decode lever after #439's compute placement**, and it
composes with every other cell in this section rather than competing with
any of them — it changes which experts are resident, not how a miss is
served. Named **cut 1** of the #302 extension: the falsifier runs against the
existing `expert_stats_*.json` artifacts (§2.2 cell 4's path) at desk cost
before any GPU arm, and the GPU confirmation follows in the next window
(`ROADMAP_456_matrix_execution.md` WAVE 1/WAVE 2).

> **CORRECTION / UPDATE, 2026-08-03 (#302a desk falsifier ran).** The cell above
> was written as a prediction. It has now been falsified against data, and the
> prediction was conservative. Evidence:
> `scripts/dev/302a_heat_desk/RESULTS.md`, computed over the recorded
> `expert_stats_*.json` of four independent boots.
>
> * The static layout is reconstructed from `plan_load_time_staging`'s own rule
>   and reproduces the recorded per-rank hit rates 0.7623 / 0.8427 / 0.8463 with
>   **delta 0.0000** — the gate that says the simulation measures the real
>   placement rather than a model of it.
> * The **oracle ceiling at the SAME resident-set size** is
>   **0.9836 / 0.9844 / 0.9850**, i.e. **+22.12 / +14.18 / +13.87 pp**. The
>   "0.81 toward 0.9+" this section estimated understates the headroom: the
>   ceiling is 0.98, and with the #82 pad expert's structural always-hit
>   excluded the static set catches 0.42-0.48 of the routed mass against an
>   oracle 0.94-0.96.
> * **Achievable, not just ceiling**: a ranking trained on one boot and scored on
>   another — a different day, a different workload, i.e. the limit case of
>   staleness — captures **40-83 %** of that ceiling (+1.24 to +18.26 pp across
>   the 4x4 x 3-rank matrix, 57-83 % within the same-day family). Every
>   off-diagonal cell is positive.
> * Verdict **MATERIAL**, far above the 2-3 pp weak threshold. Built and merged
>   off by default as `SGLANG_MOE_HEAT_MIGRATION`
>   (`layers/moe/expert_heat_migration.py`); the GPU arm is
>   `scripts/dev/302a_heat_desk/AB_SPEC.md` and is **BOOT-PENDING**. Hit rate is
>   a necessary condition for the H2D reduction this section prices, not a
>   decode measurement — no tok/s figure exists for this cell yet.
>
> **Honest scope of what the artifacts could answer.** They hold whole-run
> per-expert totals, not a per-token routing trace, so an intra-run WINDOWED
> simulation (re-rank every N steps from the preceding N steps) is not computable
> from them; the cross-run transfer test above is the substitute and is harsher
> than the thing it substitutes for.
>
> **Sub-cell #302-lookahead ("does layer N's top-k predict layer N+1's"):
> REFUTED at the aggregate grain.** Adjacent-layer heat correlation is
> indistinguishable from zero — mean Spearman between -0.03 and +0.01 with no
> consistent sign across ranks, top-R set overlap on the chance line `R/E` to
> within 2.5 pp, and top-8 overlap at or below chance once the pad expert (which
> is the top expert in every layer) is removed. Two consequences worth carrying:
> a cross-layer prefetch hint has nothing to work with at this grain, and one
> shared heat ranking cannot serve several layers, so #302a being per-layer is a
> measured requirement rather than a conservative default. This does NOT refute a
> PER-TOKEN cross-layer correlation, which these artifacts cannot see;
> `SGLANG_MOE_OFFLOAD_TRACE` already logs what that would need and it is a
> separate cheap desk item.

**#302b — cold experts under CUDA graphs.** A decode graph is captured
against **slot addresses**, not against which expert occupies a slot at
replay time — this is the structural fact `NOTE_452_desync_boot_refutation.md`
established while pricing the miss-fetch-inside-the-graph options. Any expert
whose bytes are materialised into the scratch slots **eagerly, before
replay runs**, is compatible with the graph: this is exactly Option 3 of that
note (`breakable_cuda_graph_backend.py`, `is_in_breakable_cuda_graph`) — "capture
the compute, keep the fetch eager" — the breakable route that survived the
option comparison. Fetching **inside** the graph, by contrast, stays
register-rejected: Option 1 of the same note measured a **6.60x** wall clock
against a 5.30x volume ratio for the zero-copy-gather-in-graph approach, "5.3x
slower than the eager path that already works," and was rejected on that
basis (yield ceiling ~1.25x at multi-GiB VRAM cost, "do not build"). The
`#452` sizing probe is what prices the breakable route's actual VRAM and
launch-overhead cost for DSV4-Flash specifically — the note's own numbers are
from its own workload, not yet re-run on this checkpoint.

**#302c — per-expert runtime dispatch.** When a routed expert is not
GPU-resident, a per-step dispatcher decides, from **current load**, whether to
compute it on CPU (local or the rig-2 lane of `NOTE_453_remote_expert_lane.md`
§2 — an activation-payload round trip either way) or to load it onto one or
several GPUs (via the `#423` striping principle,
`DESIGN_423_striped_offload.md`) and run it under the graph. The decision is
priced from the `#407` registry's measured tier figures plus the live load
signals that actually determine which path is cheap *right now* — H2D link
saturation, SM occupancy, host-DDR contention — not from a static preference
ordering. The precedent for a load-aware dispatch decision of this shape is
`#279`'s load-aware barlink dispatcher, and the decision itself is a
**micro-instance of the `#363` WORTH-IT AUTOCHECK**
(`DESIGN_363_regime_controller.md` §20.1): a computed "does this path beat
its cost right now" verdict rather than a flag.

Named design seam, stated so it is not rediscovered as a surprise during
implementation: **the combine step must accept partial contributions from
outside the graph.** A CPU-computed expert's output has to be added into the
layer's weighted sum *after* replay, since it was never a graph tensor to
begin with — the dispatch decision itself sits strictly *before* replay
(it decides the routing, not the arithmetic), but the result of a
CPU-dispatched expert arrives after the graph that needed it has already run.
This is a real seam in the combine kernel, not a detail to paper over.

Quality gate: the CPU and GPU expert-compute paths are **not bit-identical**
— `ANALYSE_393_ik_llama.md` already states this plainly for the CPU case ("a
CPU-computed expert is not bit-identical to the same expert computed on"
[GPU]) — and format-dependence follows the same shape the fork already labels
under the **#120 pattern** (GPTQ/AWQ-Marlin expert-offload output is
intrinsically ~1e-2 from tiling, accepted at the #77/#120 quality bar, while
an FP8 offload path is byte-identical because it is format-independent). A
per-expert dispatch that silently mixes CPU and GPU compute for the same
layer needs the same explicit labelling discipline: which format's dispatch
is bit-identical and which is not is a fact to state, not to discover from a
quality regression.

**(2) Cold-tier compression — lossless first (#306).** #306 (cold-tier
compression, currently unbuilt — `DESIGN_407_memory_tier_registry.md` §2.8:
"GREENFIELD. Its only claim on cut 1 is a capability flag") is the compression
lever this sweep evaluates first, and it is **lossless**: sparse dump + a
byte-plane split (separate the mantissa-like bulk from the small
scales/zero-points, which compress very differently) + zstd, as the default
compression path for every cold asset in the #407 registry, not an
expert-specific mechanism. Compression **in general** is the lever that
multiplies with heat migration (§2.2 cell 1) — every miss that still occurs
after heat migration moves fewer bytes if the cold tier is compressed at all
— and the lossless variant is the one allowed to multiply first, per the
fork's standing quality-last rule (lossy features only after every lossless
gain is taken, and never without a quality gate — see the #120 pattern cited
above for #302c). #126 (a lossy, quantisation-based cold-expert tier) is
therefore **not** ordered here; it is named and deferred in §7's ordered plan
and `ROADMAP_456_matrix_execution.md`'s end-of-roadmap bucket.

The first step is a **desk falsification**, not a build: measure the
achievable zstd-after-byte-plane-split ratio on real asset samples — expert
tensors pulled from the UD-Q3_K_XL GGUF, fp8-KV blocks, GDN state blobs, and
a hibernate image — before assuming the lever is worth building out.
Quantised weight bytes are already high-entropy by construction, so the
achievable ratio on the expert tensors specifically may be small and that
cell may turn out dead; the byte-plane split is expected to help mainly the
scales/mins sub-arrays, which are lower-entropy than the quantised bulk. This
is exactly the kind of claim this document's discipline requires be measured
rather than assumed — report the ratio **per asset type**, with an honest
verdict for each (including "not worth it" if that is what the sample
shows), rather than one blended number.

**Placement guidance: lossless compression pays where the link is slow
relative to compute, not on a fast local path.** The three cold-asset classes
where it is expected to matter are the 40G remote tier (~2 GB/s per #201,
the transfer class #453/#224 already use), the NVMe tier (#389), and disk
images (hibernate #89, the #305 WARM/COLD rungs, and checkpoint loading in
general). On the local PCIe H2D path specifically, compression only pays if
decompression happens **on the GPU side** (the nvcomp class of mechanism) —
CPU-side decompression followed by a raw H2D transfer saves host RAM
*capacity* only, it does not reduce the bytes crossing the PCIe link, since
the link still moves the decompressed size. This distinction is stated
explicitly here because it is easy to conflate "compression saves bytes" with
"compression saves link bytes," and only the GPU-side-decompress case does
the latter.

**(3) Peer-VRAM expert tier via BAR1.** Not previously named as a cell: use
the 5090's own VRAM as a *tier* the two 3080 ranks can miss into, over
barlink BAR1's peer-VRAM path, rather than only host RAM. This is honestly a
**capacity/hit-rate lever, not a bandwidth lever** — the requesting rank's
own PCIe hop into its local VRAM is shared by every source that ends there
(the exact §2 principle of `DESIGN_423_striped_offload.md`: the destination
link is the resource, and a peer-VRAM read still crosses it), so this cell
does not add bandwidth on top of a host-RAM miss of the same size. What it
adds is a bigger, faster-warming resident pool than host RAM alone offers on
the two smaller cards, at whatever aperture BAR1 makes available
(`docs/rig-runbook.md` §"BAR1 window, settable per communicator group. 96 MiB
maps contiguously out of 256 gross" — the per-group window is the binding
constraint on how large a peer-VRAM expert tier can be addressed at once).

**(4) Speculative router lookahead.** If #390's live stats show meaningful
layer-to-layer top-k correlation (this layer's expert selection predicts a
useful amount about the next layer's), a speculative miss-prefetch becomes
possible: start the host-tier fetch for the likely next-layer experts before
the router has actually committed to them, overlapping the fetch further than
#125's double-buffer already does. **This is falsifiable at the desk, at zero
GPU cost**, directly from the existing expert-stats artifacts:
`/spinning/gpu-battery-results/2026-08-02_439_arm3/expert_stats_*.json`
(confirmed present: `expert_stats_equal.tp{0,1,2}ep0.json`). Computing the
layer-to-layer top-k overlap from those files is the next action on this
cell, before any card time is spent on it — a desk-computable correlation
number should exist before a speculative-prefetch mechanism is designed
against a lift nobody has measured.

## 3. Asset KV

CSA-compressed and small relative to the expert asset, but the pool is the
binding constraint on context length: `max_total_num_tokens` caps the
128K/256K context arms at **42240** (this run's own admitted pool — an
estimate pending the specific boot's plan log, not yet cross-checked against
a cited artifact in this pass). The shim-free HiCache host tier proved
tonight (#441c) applies directly here: it is a host-RAM KV tier without the
compatibility shim's overhead, and long-context arms become drivable without
buying more VRAM for the KV pool specifically. This is a capacity lever
independent of the expert-asset levers in §2 — it does not compete for the
same bytes or the same links, so it composes freely with any of them.

## 4. Asset GRAPH STATES

The "capturable fetch" idea was checked and refuted at the register-entry
level (i.e. graph-capture pools do not benefit from a fetch-style lookahead
the way expert weights do — capture is a build-time cost, not a per-token
miss). What B1 proved instead: **capture itself is free** once done, so the
pre-capture doctrine of `DESIGN_363_regime_controller.md` §20.3 applies here
directly — the decode family should be pre-captured, with an eager fetch path
in front of it for the (rarer) case a fetch has not landed yet, i.e. a
breakable graph rather than a hard requirement that every input already be
resident. #452's sizing probe is what prices the actual VRAM cost of holding
this pre-captured state, and that pricing is a prerequisite for deciding how
much of it this rig can afford alongside the expert asset's own VRAM demand.

## 5. Asset TRUNK LAYOUTS

The `#363` regime/stage ladder (`DESIGN_363_regime_controller.md`) applies to
V4-Flash the same way it applies to the 27B checkpoints in that document's
own §4 stage table — the phase-optimal (prefill-concentrated vs
decode-VRAM-auto) split is a per-checkpoint solve, and **the V4-Flash phase
optimum is unmeasured**: nothing in this analysis or the cited battery
results runs the #354/#357 phase-prefill solve against a V4-Flash operating
point. It is a named gap, not a claim that V4-Flash behaves like the 27B
checkpoints.

`#445` (PP=3 vs TP=3 for this checkpoint) is reframed here as a
**disjointness experiment** in the `#423` sense (§2 of
`DESIGN_423_striped_offload.md`): under TP=3, per-layer collectives (the
attention/dense-layer all-reduces) and the expert-fetch streams contend for
the same PCIe/interconnect links on every layer. Under PP=3, each stage owns
its layers outright and the cross-stage handoff is a much smaller, much less
frequent transfer — so PP=3 is not simply "a different parallelism strategy
to benchmark", it is a way to **clear the links for the expert-fetch traffic**
that dominates decode latency per §2. Whether that trade is favourable
depends on PP's own per-microbatch bubble cost against the collective
contention it removes — an open question this reframing poses but does not
answer.

## 6. Asset DRAFTER

The DSpark speculative head is a **multiplier on everything above**: at a
measured/estimated accept rate of 0.6-0.77 it is worth an estimated
**1.5-1.8x** on top of whatever the base decode rate becomes once the expert
and KV levers land — it does not compete with them for bytes or links, it
multiplies the token rate the rest of the stack achieves. It is itself a
spillable asset: DSpark runs solo on the 5090 (the #160 canon
configuration — small enough to fit the concentrated card alongside the
target model's own resident set without contending for the smaller cards'
budget).

## 7. Ordered plan

Effort/yield order, **not naive multiplication** of the individual factors
above — the actual composed gain depends on which resource is the bottleneck
after each step, and the slowest rank clocks the whole TP group regardless of
how much headroom the other ranks gained:

1. **#439 band** (link-proportional compute) — already built, pending
   confirmation only; highest yield-per-effort of anything on this list
   because the mechanism exists and the remaining work is a card window.
2. **Heat migration** (§2.2 cell 1, #302 raised from "registered extension"
   to active work) — the largest unused lever, and it composes with #439
   rather than competing with it (one changes compute placement, the other
   changes which bytes need fetching at all).
3. **DSpark** — a multiplier, so it is worth applying once the base rate from
   1-2 is real; applying it earlier just multiplies a smaller number.
4. **#306 lossless cold-tier compression** — the desk falsification (§2.2
   cell 2) comes first, cheap and card-free; if it clears, it multiplies with
   heat migration's reduced miss rate on whichever asset types show a real
   ratio.
5. **HiCache-128K** (#441c on the KV asset) — orthogonal capacity lever,
   unlocks long-context arms rather than raising short-context decode rate;
   ordered here because it is proven tonight and cheap to wire, not because
   it competes with the expert-asset levers above.
6. **#445** PP=3-vs-TP=3 disjointness experiment — higher effort (a real
   architecture change to benchmark), ordered after the cheaper wins because
   its yield is conditional on how much link contention the earlier steps
   leave behind: if heat migration and #439 already reduce expert-fetch
   traffic enough, the collective-contention #445 addresses may be smaller
   than it looks today.
7. **Remote-CPU-lane decision** (#453), with **DDR-contention striping**
   (#423) as a companion lever on the same asset — ordered last because both
   depend on the one-afternoon falsification probes named in their own
   documents, and because #453's own gating (§4 of
   `NOTE_453_remote_expert_lane.md`) makes it a capacity, not a bandwidth,
   decision that only matters once the cheaper local-tier levers are
   exhausted.

**Lossy bucket, deliberately last and separately gated: #126.** Cold-expert
quantisation (Q2-class, below the checkpoint's native cold-tier precision) is
**not** in the ordered plan above. Per the fork's standing quality-last rule
(lossy features ship only after every lossless gain is taken, and never
without a quality gate — the same #120 pattern cited in §2.2's #302c cell),
#126 is gated on: (a) items 1-7 above landing, so the lossless gains are
banked first, and (b) a quality gate on cold-only-expert accuracy loss,
comparable in rigor to the #77/#120 bar already applied to Marlin
expert-offload. It is not scheduled here; `ROADMAP_456_matrix_execution.md`
carries it in its own end-of-roadmap bucket rather than in a numbered wave.

## 8. Cross-references

* `DESIGN_407_memtier_registry.md` §8 ("Global eviction doctrine") governs
  which experts get evicted when the resident set changes under heat
  migration (§2.2 cell 1) or cold-tier compression (§2.2 cell 2, lossless
  #306 or, later and separately gated, lossy #126) — victim selection for
  the expert tier is an instance of that doctrine, not a separate
  MoE-specific policy.
* `DESIGN_363_regime_controller.md` §20 for the pre-capture doctrine (§4
  above) and the phase-ladder applicability question (§5 above).
* `DESIGN_423_striped_offload.md` for the disjointness principle applied
  twice in this document (§2.1 DDR-contention striping, §5 the PP-vs-TP
  reframing).
* `NOTE_453_remote_expert_lane.md` for the remote-CPU-lane assessment this
  document's §2.1 and §7 point 7 summarise.

## 9. What this document is not

Not a measurement, not an implementation plan with line numbers, not a
commitment to build any of the four newly named cells (§2.2). It is the
matrix itself, so a later agent picking up any one cell starts from "this
was checked and here is what is known" rather than re-deriving the landscape
from the individual ANALYSE documents each cell was pulled from.
