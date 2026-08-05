# GPU-window ticket: prefill graphs, interleaved timing + determinism arms

Prepared at desk 2026-08-05, no cards touched. Everything below is executable
as written; the window should be pure measurement.

**Power caps (measured, must be re-verified in the report header):**
RTX 3080 **200 W**, RTX 5090 **400 W**, RTX 3080 **200 W**
(`power.default_limit` 320 / 575 / 320). All arms share these caps, which is
what makes ms-per-fixed-work comparable across arms. Comparisons against
pre-change archive numbers are confounded.

**Standing rule for this ticket (user, 2026-08-05).** At a fixed power limit a
lower clock usually means HIGHER load, not a disadvantage: a power-limited card
drops clocks when it does more work per cycle, especially with high power draw
at the same time; low clock with LOW power is the light-load case. Therefore:
score on **ms per fixed unit of work**, sample **clock and power together**,
read them **only jointly and only as diagnostic annotation**, and **never
normalise by clock**. Interleaving is in this design to remove slow drift --
that is its only job.

---

## Part A -- the timing question (window 4)

**Command**

    cd /spinning/wt-prefill-graphs
    STAGE unused; run directly:
    bash tests/prefill_graphs/window4_interleaved.sh

Defaults: `REPS=3`, `TRANSPORT=nccl`, `RESERVE=5500,4200,4200`, port 30043,
artifacts to `/spinning/gpu-battery-results/2026-08-05_prefill_graphs_w4/`.

### Transport -- decided explicitly, not inherited

Production now exports `SGLANG_BARLINK=1` (user decision 2026-08-05:
production runs barlink, usage is the soak). That changes this ticket in two
ways, and the first one is a live hazard:

1. **The NCCL arms MUST unset it per-arm.** The earlier revision of
   `window4_interleaved.sh` never mentioned `SGLANG_BARLINK`, so it would have
   inherited the transport from whatever shell launched it. An arm that
   silently ran barlink would not be comparable with window 3, and a run
   launched from a production-flavoured shell would have compared
   prefill-graph-on-barlink against window 3's prefill-graph-on-NCCL while
   reporting it as a prefill-graph result. Both failure modes are silent.
   The script now sets `TRANSPORT` explicitly (`nccl` default -> `unset
   SGLANG_BARLINK`; `barlink` -> `export SGLANG_BARLINK=1`), refuses any other
   value, stamps the transport into **every** artifact JSON, writes
   `transport.txt`, prints it in the report header, and the reporter **aborts**
   if the arms disagree. Verified by fixture: a mixed-transport artifact set
   exits 1 with "arms used DIFFERENT transports -- not comparable".

2. **Window 4's answer is therefore about NCCL, and production no longer runs
   NCCL.** This is deliberate -- holding the transport at window 3's value is
   what makes the prefill-graph variable isolated and the two windows
   comparable. But it must be said plainly in the writeup: an NCCL-only result
   is **not by itself a production rollout argument** any more. The
   decision-relevant cell for today's production is barlink x prefill-graphs,
   which is the `TRANSPORT=barlink` half of the 2x2 and still needs its own
   eager floor. Same command, `TRANSPORT=barlink`.

### Reserve -- rebased on the current production value

Production raised the 3080 term `3800 -> 4200` on 2026-08-05: its own demand
model derives 4160 MiB (activation + graph capture + GDN prefill scratch) and
warned that 3800 was 360 MiB short with the 96-slot mamba pool. Booting the
arms at the stale 3800 would have sized a different KV pool than production
runs, so the default here follows production at `5500,4200,4200`.

**Arms** -- interleaved, alternating, so neither treatment owns the cool end:

    E1  eager                              G1  --cuda-graph-backend-prefill breakable
    E2  eager                              G2  breakable
    E3  eager                              G3  breakable

**Workload points -- two only.** The spread is saturated after 5-20 s per
point, so more points or longer points buy nothing and turn a few-minute window
into a battery.

| point | shape | sizing | expected wall |
|---|---|---|---|
| `1900` | long single-stream prefill, GEMM-bound | `--prompts 8 --passes 1` | ~9 s |
| `256c4` | 256-token prompts, 4 in flight, launch-train bound | `--prompts 24 --passes 2 --concurrency 4` | ~8 s |

Both arms run a byte-identical prompt set (fixed seed, fixed count, fixed
order), verified in the smoke: 8 prefills / 15200 tokens in each arm.

**Content gate** runs on pair 1 only -- window 3 already answered the content
question with a passing boot-to-boot floor, and re-confirming it on every rep
costs a minute per arm.

**Time plan (minutes, not tens of minutes)**


| item | count | each | total |
|---|---|---|---|
| boots | 6 | ~40 s eager / ~70 s graphs | ~5.5 min |
| teardown | 6 | ~10 s | ~1.0 min |
| measurement (2 points) | 6 | ~20 s | ~2.0 min |
| content gate | 2 | ~25 s | ~0.8 min |
| **total** | | | **~9-10 min** |

Boots are ~60 % of the window and are irreducible: the prefill backend is fixed
at boot, so an A/B needs separate processes. The measurement itself is ~2 min.
`REPS=2` trims to ~6-7 min at the cost of a thinner floor.

**Acceptances**

* Report header prints the measured power caps AND the transport of every arm.
* All arms report the same `transport`; the reporter aborts otherwise.
* Each point reports `ms_per_prefill` per arm plus paired deltas `G_i` vs `E_i`.
* `cached_tokens_total == 0` everywhere, else the point is INVALID.
* Verdict per point is one of REPORTABLE / INSIDE FLOOR / INVALID / NO FLOOR.
  A paired delta inside the A-vs-A floor is **not a result**.
* Clock+power printed as annotation; no point is rejected or normalised on them.

---

## Part B -- the determinism arms (ED/GD), with a derived reserve

Window 3's determinism arms never ran: both OOM'd during capture
(`boot_ED.log`: `avail mem=0.05 GB` on TP1, then
`multimem all-gather disabled (CUDA driver error: out of memory)`).

### Root cause, derived from configuration

`--enable-deterministic-inference` rewrites the flashinfer workspace size at
backend construction:

* `python/sglang/srt/environ.py:1270` -- default
  `SGLANG_FLASHINFER_WORKSPACE_SIZE = 384 MiB`
* `python/sglang/srt/layers/attention/flashinfer_backend.py:987` -- certain
  Qwen2/Qwen3 architectures raise it to 512 MiB. The served arch is
  `Qwen3_5ForConditionalGeneration`, which is **not** in that list, so the
  baseline here stays 384 MiB.
* `python/sglang/srt/layers/attention/flashinfer_backend.py:1007` -- under
  deterministic inference it is set to **2048 MiB**.

So the per-rank delta is a pure config term, no literal of my own:

    delta_ws_per_rank = effective_ws(deterministic) - effective_ws(baseline)
                      = 2048 MiB - 384 MiB
                      = 1664 MiB

`TERM_ATTN_WORKSPACE` is not in `BUDGET_FUNDED_TERMS` (`engine.py:105`, which
holds only weights and the mamba pool), so the workspace is allocated *after*
the KV pool is sized and has to come out of reserve headroom. One rank process
per card here, so the delta lands once per card.

**Confirmation from the failed boot** (derivation is primary, this is the
check): between `Memory pool end` and `Capture target verify begin`, arm E1
consumed 2.49 -> 1.91 GB while arm ED consumed 2.48 -> 0.30 GB. Extra demand
= 1.61 GB = **1649 MiB**, against 1664 MiB derived. Agreement within allocator
rounding, and the KV pools were identical (same token counts, same K/V sizes),
so nothing else moved.

### The reserve the follow-up window must boot with

    RESERVE="7164,5864,5864"     # = 5500+1664, 4200+1664, 4200+1664

    TRANSPORT=nccl RESERVE="7164,5864,5864" REPS=1 \
      bash tests/prefill_graphs/window4_interleaved.sh   # ED/GD variant

Rebased on the current production baseline: the 3080 term is 4200, not the
3800 this ticket first assumed, so the determinism vector is 4200+1664 = 5864
rather than 5464. The two corrections are independent -- 4200 fixes a
previously-diagnosed activation/capture/scratch shortfall, and +1664 is the
deterministic workspace delta on top of it.

with `--enable-deterministic-inference` added to both arms. Acceptance: both
arms reach health 200, and the content gate answers whether deterministic
inference closes the 4/8 divergence (byte-strict reachable) or not (the gate is
distribution-level at best).

### What the ledger cannot express yet

The ledger *should* carry this as a `MODELED` term -- it is pure configuration,
reproducible without a card. It currently cannot, and the gap is one line:

    python/sglang/srt/mem_ledger/engine.py:293
        flashinfer_mib = int(envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()) // (1 << 20)

The ledger samples the environment variable at ledger-build time, but that
variable is **rewritten later**, inside `FlashInferAttnBackend.__init__`, by
both the arch rule (`:987`) and the determinism rule (`:1007`). So the ledger
reads 384 MiB for a boot that will allocate 2048 MiB, and
`TERM_ATTN_WORKSPACE` does not move when `enable_deterministic_inference`
moves -- which is exactly the property the ledger's own hermetic test asserts
for a MODELED term.

**Missing term, stated rather than padded:** a resolver that replicates the
backend's own rule, plus the inputs that drive it.

    effective_flashinfer_workspace_mib(server_args) =
        2048                              if enable_deterministic_inference
        512                               if arch in {Qwen2ForCausalLM,
                                                      Qwen3ForCausalLM,
                                                      MiMoForCausalLM,
                                                      Qwen3VLForConditionalGeneration,
                                                      Qwen3VLMoeForConditionalGeneration}
        SGLANG_FLASHINFER_WORKSPACE_SIZE  otherwise (default 384 MiB)

and `TERM_ATTN_WORKSPACE.inputs` gains `enable_deterministic_inference` and
`model_config.hf_config.architectures`. Until that lands, the 7164/5864/5864
vector above is derived by hand from the same rule and should be passed
explicitly rather than trusted to `auto`.

---

## Harness files

| file | role |
|---|---|
| `tests/prefill_graphs/window4_interleaved.sh` | interleaved E,G,E,G,E,G driver |
| `tests/prefill_graphs/prefill_perf.py` | fixed-work probe, ms/prefill, clock+power |
| `tests/prefill_graphs/report_interleaved.py` | paired deltas, A-vs-A floor, annotation |
| `tests/prefill_graphs/content_gate.py` | byte-identity oracle |
| `tests/prefill_graphs/mock_server.py` | cardless smoke vehicle |

All were smoke-tested end-to-end against the mock server with no GPU:
fixed-work identical-work check, concurrent point, content gate PASS path and
FAIL path, and every report verdict branch (REPORTABLE / INSIDE FLOOR /
INVALID / NO FLOOR).
