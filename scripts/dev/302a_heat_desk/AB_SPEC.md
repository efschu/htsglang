# #302a heat migration — GPU A/B spec (WAVE 2, NOT YET RUN)

Written at the desk alongside the falsifier in this directory. **Nothing in
this file has been observed on hardware.** Every figure the "expected" column
carries is a projection from `RESULTS.md`, which is a simulation over recorded
router statistics, not a measurement of this feature.

Target directory for the window: `/spinning/gpu-battery-results/2026-08-0X_302a_heat/`.
Conventions follow `2026-08-02_439_arm3/CONFIRMATION_WINDOW.md`, which this
spec deliberately mirrors rather than reinventing.

## 0. Gate

`ROADMAP_456_matrix_execution.md` WAVE 2 lists this arm behind "WAVE 1 item 1's
desk falsifier passing". It has passed (`RESULTS.md` §4, verdict MATERIAL), and
the build has landed with 33 hermetic tests and two executed can-fail arms. The
gate is cleared; what is missing is a card window.

## 1. Arms, in boot order

Three boots. The recipe is the one `2026-08-02_439_arm3` used — DSV4-Flash GGUF,
uneven TP=3, eager offload path (`--disable-cuda-graph`, the shipped path per
#452), `SGLANG_EXPERT_STATS=1`, `RESERVE_MIB=auto`, barlink transport per the
standing default.

| # | arm | added env | what it establishes |
|---|---|---|---|
| 0 | `floor` | none (baseline boot) | **A-vs-A floor, taken first.** 3 x 450-token generations inside this one boot, same prompt, before any delta is quoted. |
| 1 | `static` | none | control: the shipped static residency, hit rate + decode + H2D. This is arm 0's boot continued, not a separate one. |
| 2 | `heat` | `SGLANG_MOE_HEAT_MIGRATION=1` | the treatment. Same recipe, same prompts, same order. |

Arms 0 and 1 share a boot on purpose: the floor must come from the same process
whose numbers it is used to judge. Arm 2 is a separate boot because the feature
is resolved once per layer at cache construction.

Optional fourth boot if the window has room: `SGLANG_MOE_HOT_RESIDENCY=1`
(Stage-1, one-shot freeze). It separates "re-ranking helps" from "any non-static
choice helps", which the desk data cannot separate and which is the most
plausible alternative explanation for a positive arm 2.

## 2. Knobs for arm 2, and why these values

Defaults ship as `period=512`, `decay=0.5`, `hysteresis=0.25`, `min_gain=8.0`,
`max_swaps=4`. For a first window use the defaults unchanged — a window that
tunes and measures at the same time cannot attribute its own result. If arm 2
records `heat_rounds > 0` but `heat_swaps == 0`, the margins are too tight for
this workload and a second boot at `SGLANG_MOE_HEAT_MIN_GAIN=2` is the follow-up,
reported as a separate arm rather than as a correction to the first.

## 3. What is read, and from where

Per rank, from `expert_stats_heat.tp{0,1,2}ep0.json`:

| quantity | field | expected direction |
|---|---|---|
| activation-grain hit rate | `totals.hit_rate` | **up.** Desk projection: 0.7623 / 0.8427 / 0.8463 -> the transfer test's in-family band, i.e. roughly 0.89-0.95 / 0.92-0.96 / 0.92-0.96. |
| H2D bytes | `totals.h2d_bytes` | **down**, roughly in proportion to the miss fraction. tp0's 888.6 GiB at miss 0.238 would fall toward ~0.11 miss. |
| migration cost | `totals.heat_h2d_bytes` + `heat_d2h_bytes` | **small next to the above.** This is the honest cost line; a win that the migration traffic eats is not a win. |
| rounds / swaps | `totals.heat_rounds`, `heat_swaps`, `heat_rounds_migrating` | non-zero, and front-loaded. `rounds_migrating / rounds` falling over the run is the anti-thrash claim showing up on hardware. |
| decode | `decode_heat.json` | ms/verify and ms/prefill per the standing measure, tok/s alongside. |

Per-rank, always — the slowest rank is the clock (`langsamster-rang-taktgeber`),
and a hit-rate lift that lands only on the fast card moves nothing.

## 4. Falsifiers, named before the run

* **Null hit-rate delta on all three ranks.** The desk falsifier says the static
  set is 40-50 pp off the achievable one; if a live re-rank cannot close any of
  that, the transfer result does not carry to in-session traffic and the cell is
  refuted for this workload.
* **Hit rate up, decode flat or worse.** Falsifies the *economic* claim, not the
  mechanism. Most likely cause would be the migration's own PCIe traffic or the
  two `cuda.synchronize()` calls in the round; both are measurable separately
  (`heat_h2d_bytes`, and a boot at `SGLANG_MOE_HEAT_PERIOD=100000` which keeps
  the accounting and never migrates).
* **Residency size moved.** Would falsify the VRAM-neutrality invariant that the
  hermetic tests pin. Read `resident_count` per layer in both dumps; they must
  be equal field for field. A difference here is a bug report, not a result.
* **Output divergence.** Same greedy prompt, arm 1 vs arm 2, must produce the
  same text. The desk tests pin bit-identity of the MoE output across a
  migration; hardware is where that claim is actually tested. A divergence is
  the #452-B2 shape of finding and outranks every number in §3.

## 5. Cost of the window

Three boots at ~7 min load. No re-boots to re-read something. VRAM corridor
sampled as usual; the feature allocates nothing on the device, so the corridor
is a check that this is true rather than a budget item.
