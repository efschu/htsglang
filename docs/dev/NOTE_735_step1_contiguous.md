# #735 Step-1 — the contiguous full plan boots, serves, and prefills ~10% faster

GPU window 2026-08-18 23:13-23:30Z, rig CT999, tip `1d1363cbc6`. Booted through
the ticket's own runner (`arm_boot_735.sh`), which pins the commit, gates on
BAR1 and the host ledger, and collects the acceptance itself. Two independent
boots, `step1ctg` and `step1ctgB`. No host tier, per the ticket.

Configuration: `argv_735_step1.txt` (PP=3, `--pp-stage-ratio 31,17,16`,
`--pp-attn-stage-ratio 7,5,4`, flip and NEXTN spec on, context 327680,
`--max-total-tokens 436275`) plus `SGLANG_PP_LAYER_SET="0-30;31-47;48-63"`.

## Acceptance

| item | step1ctg | step1ctgB |
|---|---|---|
| health 200 | after 181 s | after 190 s |
| guard markers (`PPLayerSetError` / `PPMissingLayer`) | 0 | 0 |
| `max_total_num_tokens` | 436275 | 436275 |
| probe "Say the word banana." | `'\n\nbanana'` | `'\n\nbanana'` |
| probe "17 plus 25, number only" | `'\n\n42'` | `'\n\n42'` |
| probe "largest ocean" | `'\n\nThe Pacific Ocean.'` | `'\n\nThe Pacific Ocean.'` |

All probes at temperature 0, seed 735000001, `finish_reason: stop`. Corridor at
steady state: 2785 / 3614 / 3571 MiB free, all above the ~1024 MiB target.

## Prefill throughput vs the incumbent contiguous baseline

Same 5.8k-token prompt on both arms, `max_tokens=1` to isolate prefill.

| arm | steady-state runs (tok/s) | median | within-arm spread |
|---|---|---|---|
| incumbent (`--pp-stage-ratio 14,10,8`, knowngood) | 5104 / 5120 / 5126 / 5107 | **5106.8** | 0.4 % |
| #735 step-1 (31,17,16 + FA 7,5,4) | 5641 / 5625 / 5620 / 5600 | **5620.2** | 0.7 % |

**+10.1 %**, against a within-arm spread under 1 % — roughly fourteen times the
noise. The cold first-touch pair agrees in direction and size: 1572.8 vs 1415.1,
+11.1 %.

Two measurement traps had to be cleared before this number meant anything, and
both are worth keeping:

- **Prefix caching.** Repeating the identical prompt returns in 0.13 s
  (~45 000 tok/s) off the radix cache. Every run now carries a UUID salt at the
  head of the prompt so no prefix can be reused.
- **First-touch warmup.** Even with unique salts, run 0 is ~4x slower than runs
  1-4 (JIT, autotune, allocator growth). A single-shot measurement therefore
  compares warmup against steady state; the first comparison this window
  produced (+9.9 % from one run per arm) sat inside a 1449-1940 tok/s band on
  the incumbent alone and was worthless. `prefill_probe.py` now runs N salted
  requests and reports the median with the spread beside it.

## Status

Step-1 serves correctly and is ~10 % faster on prefill than the incumbent. That
makes it a perf carrier CANDIDATE. The ship config was NOT switched, and this
note does not recommend switching it: the arms measured prefill only, on one
prompt shape, with no decode-throughput or long-run stability evidence.
