# TICKET 460 — RIFE frontier sweep and raw-frame transport validation

GPU window spec. Everything here is BOOT-PENDING: it is the set of numbers the
#460 ladder and the #457 stage pricer were built to consume and could not
measure at a desk. Nothing in this file has been run.

Desk half merged on `feat/video-rife-ladder-460`; read
`docs/dev/TASK_333_M2_VIDEO_ENHANCE.md` §17 first — it states what is already
measured, what the ladder does with it, and which cells are absent.

## 0. Preconditions

* Hold `/spinning/gpu-arb/` (holder + heartbeat; stop the heartbeat BEFORE
  releasing). Do not start without it — other agents share this box.
* Resolve cards by NVML UUID, never by torch order. Ticket V's mapping is a
  record, not a constant: `5090` was NVML idx 1, the x8 3080 idx 2, the **x4**
  3080 idx 0. Re-resolve. Set `CUDA_VISIBLE_DEVICES` to the UUID string so each
  worker sees exactly one device as `cuda:0`.
* venv `/spinning/htsglang-gpu/.venv/bin/python`.
* Weights: all eight rungs are already on disk at
  `/spinning/llm_stuff/k3-models/rife/` and verified against
  `rife.KNOWN_WEIGHT_SHA256`. Set `SGLANG_RIFE_WEIGHT_DIR` to that path. Do not
  spend window time downloading; if a file is missing use
  `scripts/video_enhance/fetch_rife_weights.py <version>` (it refuses an
  unpinned artifact).
* VRAM corridor: free >= 400 MiB absolute at peak, and report peak, not idle.
  Mark a point INFEASIBLE-AT-CORRIDOR rather than shrinking it silently.

## 1. Per-boot A-vs-A floor first (rig convention)

Before any delta is reported, measure the floor **in the same process** as the
rows it governs. Do not reuse ticket V's 0.682 % / 0.108 %. Discard one
JIT/clock-ramp block, then time two blocks of the same work and report their
spread as the floor. Size each block to fill 10-20 s rather than to a fixed
iteration count.

Anything below the floor is not reported as a difference.

## 2. The frontier sweep — the ticket's main product

Six rungs are ABSENT in `rife_ladder.seeded_frontier()` and are therefore
**never auto-picked** by the ladder. They are the work list, and the policy
report prints it: `Selection.measure_first`.

| version | arch group | encode head | modulo | status |
|---|---|---|---|---|
| 4.15 | `IFNet_HDv3_v4_18` | 8 ch | 32 | **absent** |
| 4.17 | `IFNet_HDv3_v4_18` | 8 ch | 32 | **absent** |
| 4.18 | `IFNet_HDv3_v4_18` | 8 ch | 32 | **absent** |
| 4.15.lite | `IFNet_HDv3_v4_17_lite` | 4 ch | 32 | **absent** |
| 4.16.lite | `IFNet_HDv3_v4_17_lite` | 4 ch | 32 | **absent** |
| 4.17.lite | `IFNet_HDv3_v4_17_lite` | 4 ch | 32 | **absent** |
| 4.6 | `IFNet_HDv3_v4_6` | none | 32 | measured (ticket V) |
| 4.26 | `IFNet_HDv3_v4_26` | 4 ch, 5 blocks | 64 | measured (ticket V) |

Grid: **6 versions x 2 arches x {1920x1080, 3840x2160} x scale {1.0, 0.5}** =
48 points. Re-run 4.6 and 4.26 at one point per arch as a cross-check against
ticket V; a disagreement beyond the floor is a finding, not a rounding error.

Unit: **milliseconds per frame pair at multiplier 2**, which is what
`RifeFrontier` cells are in and what `chain_policy` prices against.

### 2.1 The encode-cache correction is mandatory

`RifeStage._encode_cache` is keyed by `Frame.order_key`. 4.6 has no encode head
and is unaffected; every other rung has one. A timing loop over a single fixed
pair pays the encode once and never again, which **flatters every rung except
4.6** — and the ladder is precisely a comparison between 4.6 and those rungs,
so an uncorrected number would bias the very decision this ticket exists to
inform.

Report all three columns, as ticket V did:

```
cached      timing loop over one fixed pair
cleared     cache cleared between iterations (pays the encode twice per pair)
amortised   cached + (cleared - cached) / 2      <- the planning figure
```

The **amortised** column is what goes into the frontier: a sequential stream
pays the encode once per source frame, because each frame is the right member
of one pair and the left member of the next.

### 2.2 Peak device bytes per pair (P4)

Take `vram_peak_mib` at each 4K scale-1.0 point at minimum, both arches, all
six versions. Ticket V has only 4.6 (4742 MiB) and 4.26 (7855 MiB) on the 5090.
The lite family is expected to be the cheapest class and that expectation is
exactly what wants checking — `VramClass` is currently derived from the
architecture, not from a measurement.

### 2.3 The `scale` lever

Ticket V's sharpest finding was that `scale=0.5` is worth far less on 4.26
(4K: 32.97 -> 25.37, -23 %) than on 4.6 (20.54 -> 11.36, -45 %), because 4.26's
fifth pyramid level and its encode head do not shrink with `scale`. The 4.1x
group has four blocks and an 8-channel head; the lite group has four blocks and
a 4-channel head. Report the s1.0 -> s0.5 discount per version explicitly —
the planner uses `scale` to buy headroom and needs to know where that lever
still works.

### 2.4 Harness

`scripts/gpu_battery/` **inside the worktree** is the battery harness; the
video probes live in `scripts/video_enhance/`. Reuse
`scripts/video_enhance/stage_rate_sweep.py` and the `RifeStage` from the tree
rather than reimplementing the stage — ticket V's drivers had to exist only
because `trt_engine_bench.py` benches one resolution per invocation.

Emit into the existing schema (`started_at`, `finished_at`, `host{card_index,
card_name, nvml_uuid, total_mib}`, `noise_floor_pct`, `samples[...]`, plus an
additive `extra` block carrying the floor method, the discarded block, and
`min_free_mib_observed`), written to `docs/dev/measurements/333-m2/`.

### 2.5 Feeding it back

`rife_ladder._SEED_MS` and `_SEED_VRAM_MIB` are the tables to extend. Add the
rows, keep the `_TICKET_V` provenance string accurate (it names the source
document and the amortisation convention), and extend
`test_rife_ladder.TheSeededFrontierTest` — `test_only_46_and_426_are_measured`
will fail on purpose, which is the intended signal that the frontier moved.

## 3. What this ticket does NOT settle: quality

The ladder's `DEFAULT_QUALITY_RANK` is an **ASSUMPTION**, labelled as one in
every report. This ticket measures speed and memory. It does not grade output.

A quality gate is a separate ticket and would be: held-out frame triples
(frame *n-1*, *n*, *n+1*) from real content, interpolate *n* from its
neighbours, PSNR/SSIM against the true *n*, several content classes (high
motion, slow pan, fades, text overlay). Until then, do not report a version as
"better" — only as faster or cheaper. If that gate ever runs, its output goes
in through `RifeLadder.with_quality_ranks` and touches no selection code.

## 4. Transport validation points (#457)

The stage pricer models three transport facts. Two are modelled from a single
measured number and one is entirely absent. All three are needed before a
stage-pipeline plan is run rather than priced.

### 4.1 Host bounce, observed rather than modelled

`stage_pipeline.MEASURED_X8_GIB_S = 13.70` comes from `p1_5090.json` and is a
**round-trip** figure that RESULTS.md then used one-way. Measure D2H and H2D
separately, per card, at the payload sizes the chain actually moves:

| boundary | payload | MiB |
|---|---|---:|
| decode -> sr | 1080p NV12 | 2.97 |
| resize -> rife | 4K fp16 | 47.46 |
| rife -> encode | 4K fp16 x2 | 94.92 |
| sr -> resize | **8K fp16** | 189.84 |

Include the 8K row even though co-residency forbids that crossing today: it is
the number the taboo is stated in and it is currently an inference.

### 4.2 The x4 card

`host_bounce_links` gives the x4 card **half** the measured x8 rate and labels
it `estimate`. Nobody benchmarked it. Measure it. Read the **negotiated** link
width, not the NVML nameplate.

This matters more than it looks: at `prefetch_depth=1` the x4 crossing in the
best placement is fully hidden and contributes zero, so the verdict is reported
as `measured`. If the real x4 rate is much worse than half, that hiding may not
hold and the same placement becomes an `estimate` verdict with a lower number.

### 4.3 Prefetch hiding — the proof, not the model

Directive 3 is priced as `max(0, transfer − window)` with the window being the
receiving card's compute times `prefetch_depth`. That is an assertion about a
dedicated copy stream overlapping a compute stream, and it has not been shown
on this rig for video frames.

Falsifier: run the same placement twice, once with the copy issued on the
compute stream and once on a dedicated copy stream with depth 1, and show the
period differs by the transfer time. If it does not, the overlap is not
happening and the pricer's optimistic column is wrong — which is the outcome
worth knowing.

### 4.4 barlink BAR1

`stage_pipeline.barlink_link()` carries an **absence**, and a placement needing
it is refused by name. barlink is the house default transport wherever a
combination supports it, so this is the gap that matters most: a measured BAR1
peer rate for raw video frames would remove the host bounce from every stage
split. Establish whether a 4K fp16 frame can cross a 256-MiB BAR1 window at all
on this rig, and at what rate. If it can, `CardProfile.link_gib_s` takes the
number and nothing else in the pricer changes.

## 5. The fused SR tail (#457 build half)

Added 2026-08-03. `TASK_333_M2_VIDEO_ENHANCE.md` §17.7 built the fusion and
graded the artifact on the CPU provider; the parity of the *filter* is settled
(145 dB against the two-stage reference, and the deliberately-wrong `nearest`
arm is rejected at 17 dB). What a desk cannot answer is what the fused engine
costs on silicon, and every re-priced chain figure in §17.7 is ESTIMATE until
these rows exist. They join this window rather than opening a competing one —
the same cards, the same corridor rule, the same floor discipline as §1.

### 5.1 Build the fused engine on both arches

```
PYTHONPATH=python:/tmp/onnxtools python scripts/video_enhance/export_sr_fused_tail.py \
    --arm lanczos3 --grade --report <out>/fused_tail_cpu.json
```

then build the TensorRT engine from the derived artifact exactly as ticket V
built the unfused one: `NetworkDefinitionCreationFlag.STRONGLY_TYPED`, dynamic
profile min 960x540 / opt 1280x720 / max 1920x1080 **on the input**, which is
unchanged — the tail is shape-agnostic and the profile does not move. Record
build seconds, engine bytes and the consumer matrix per arch, the same three
columns ticket V's §1 table has.

The #339 engine-build discipline applies unchanged and there is one addition
worth checking explicitly: the engine's **output** signature is now
`1x3x(2*h)x(2*w)`, not `1x3x(4*h)x(4*w)`. A consumer that derives the output
shape from `SrModel.scale` gets it right (the derived model's scale is the
**net** scale, x2); one that assumes x4 does not. Verify all three profile
resolutions produce half-size output before timing anything.

### 5.2 Parity on the card, both arches

The desk gate compared fp32-graph against fp32-graph and measured float
rounding. The window gate is the one that matters: **fused fp16 TRT engine vs
the fp32 unfused ONNX followed by `resize.lanczos3_resize`**, which is
`fused_tail.grade_fused_tail`, thresholds 40 dB / 0.995, 3 samples at 960x540
as in ticket V.

Expect it to land near ticket V's 48.1 dB rather than near the desk's 145 dB:
the fp16 conversion dominates, and the tail adds a 12-tap sum whose fp16
accumulation is the one genuinely new numerical risk. **If it fails on one
arch, that is the finding** — the fallback is `--io-cast-only` on the fused
artifact (the fp16 export composes with the fused one; verified at the desk),
not abandoning the fusion.

### 5.3 The row the whole re-pricing hangs on

`sr_fused` ms/frame at 1920x1080 input, TRT fp16, both arches, per-boot A-vs-A
floor first. Against ticket V's 25.424 (5090) / 90.343 (3080) for `sr` alone
and 24.367 (5090) for `resize`:

* the fused stage cannot be cheaper than `sr` alone;
* §17.7.4's verdict of 21.24 src-fps holds for any fused cost up to
  **21.66 ms** above the unfused `sr` figure on the 5090, i.e. the fusion only
  has to beat the separate resize's 24.367 ms by 2.7 ms;
* if the fused cost lands *above* `sr + resize`, the fusion is a regression
  and that is a publishable negative result. Report it as one.

Take the 4K-output payload check alongside it: peak device bytes for the fused
stage should drop by roughly the 8K intermediate (189.84 MiB per in-flight
frame) against the unfused pair, which is the other half of what the fusion
buys and is a `forward_peak`-style corridor observation, not a timing.

### 5.4 The three rows that were already missing

These are §17.5's named absences. They gate the pipeline-vs-replication
comparison and are cheap next to the sweep:

1.  **`resize` on a 3080** at 7680x4320 -> 3840x2160. Still wanted even though
    the fusion removes the stage: it is the before-number the fused row is
    judged against on that arch, and it is what makes the *unfused* Regime-A
    figure a value instead of the 8.67/18.299 bound.
2.  **`color_to_rgb` at 1920x1080** and **`color_to_yuv` at 3840x2160**. Both
    are carried as `unpriced_stages` on every report, so every absolute fps
    figure in §17.5 and §17.7 is optimistic by an unknown amount.

### 5.5 Re-price and re-pin

With the real `sr_fused` row, re-run §17.7.4 and update both the table and
`test_stage_pipeline.FusedVerdictTest`, which pins it. Replace `tail_ms=0.0`
with the measured difference and drop the ESTIMATE label from the cells that
earned it. The test exists so the document cannot drift from the pricer.

## 6. Re-derivation to repeat once the numbers land

`TASK_333_M2_VIDEO_ENHANCE.md` §17.5 is pinned by
`test_stage_pipeline.TicketVVerdictTest`. With real frontier rows for the six
new rungs and real transport rates:

1.  Re-run `best_placement` over the full ladder, not just 4.6, and report the
    binding card and stage per version. The s0.5 placement binds on the 5090's
    `sr + resize` pair at 49.791 ms (20.08 src-fps); a cheaper RIFE cannot move
    that, but a *more expensive* one can move the bind onto a 3080.
2.  Take the **3080 resize row**. Without it the Regime-A comparison is a bound
    (8.67 src-fps strict lower, 18.299 upper) rather than a value, and the
    pipeline-vs-replication verdict cannot be settled.
3.  Take `color_to_rgb` / `color_to_yuv` at 1080p and 2160p. They are absent at
    these resolutions, carried as `unpriced_stages`, and they make every
    absolute fps figure in §17.5 slightly optimistic.
4.  Update §17.5 and the test in the same commit. The test exists so the
    document cannot drift from the pricer.
