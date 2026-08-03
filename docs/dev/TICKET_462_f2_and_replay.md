# GPU ticket #462 — F2 first, then the replay gate, then the A/B

Discharges the DESK-WRITTEN-NEVER-EXECUTED label on
`docs/dev/DESIGN_462_breakable_route.md`. Branch
`feat/breakable-offload-graph-462`.

**Order is mandatory.** F2 is the first measurement of the window and it gates
default-on and every performance claim. If F2 says the breaks cost more than
the graph saves, steps 2 and 3 are not worth their slot and the correct action
is to record the number and leave the route gated OFF — that is a result, not a
failure.

Harness lives at `scripts/gpu_battery/` **inside this worktree**
(`/spinning/wt-462-breakable/scripts/gpu_battery/`). The `memory/scripts` path
in some session directory listings does not exist.

---

## 0. Preconditions

- Hold `/spinning/gpu-arb/` for the whole window; stop the heartbeat BEFORE
  releasing.
- Corridor sampler started **before** the server, sampling **during load** —
  the 2026-08-02 window's post-boot readings overstated free memory by ~1.4 GB.
  Floor: **≥400 MiB free per card**, judged at peak, not idle.
- `--rank-gpu-id` takes **CUDA ordinals**, not NVML indices. Resolve via the
  IdentityMap (`registry/nvml.py`) at run time; never assume.
- `--enable-metrics` on **every** boot line.
- Own PIDs via pidfiles. Never `pkill -f`. `py-spy dump` before killing
  anything wedged. Bounded waits only.

## 1. Boot lines

Base recipe is the 2026-08-02 offload reference (DeepSeek-V4-Flash-0731
UD-IQ3_XXS, TP=3), which reached HEALTHY and served at 149.1 ms/token eager.

```bash
setsid "$VENV/bin/python" -u -m sglang.launch_server \
  --model-path "$GGUF_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf" \
  --tp-size 3 --rank-gpu-id <NVML-resolved> --rank-tp-ratio auto \
  --rank-auto-reserve-mib 2200,1400,1400 \
  --rank-moe-resident-fraction 0.485,0.42,0.42 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 8192 --max-running-requests 1 \
  --chunked-prefill-size 512 \
  --trust-remote-code --enable-metrics \
  --host 127.0.0.1 --port 30462 \
  <ARM FLAGS>
```

with `SGLANG_MOE_HOST_SHARD_RATIO=1,1,1`, `SGLANG_EXPERT_STATS=1`,
`SGLANG_EXPERT_STATS_INTERVAL_SEC=45`,
`SGLANG_GGUF_STREAM_TRIM_SOFT_GIB=88` / `_TARGET_GIB=78`,
`SGLANG_DSV4_FP4_EXPERTS=0`, and **unset** `SGLANG_MOE_COLD_TIER_SHM`,
`SGLANG_MOE_OFFLOAD_CUDA_GRAPH`, `SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE`,
`SGLANG_MOE_HOT_RESIDENCY`.

| arm | ARM FLAGS | env |
|---|---|---|
| `eager` (control) | `--cuda-graph-backend-decode=disabled --cuda-graph-backend-prefill=disabled` | `SGLANG_MOE_OFFLOAD_GRAPH_MODE` unset |
| `breakable` | `--cuda-graph-backend-decode=breakable --cuda-graph-backend-prefill=disabled` | `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable` |

**Scratch sizing, do this before booting the breakable arm.** The bound is
`min(max_captured_bs x top_k, E_local - R)` and it counts graph-PADDED rows,
which carry real routed ids. `top_k = 6`; `E_local` 114/71/71, `R` 56/30/30 at
this geometry, so the cold set is 58/41/41. At `--cuda-graph-bs-decode 1`,
`C = 6`. Raising the captured bs raises `C` and therefore resident VRAM — treat
it as a corridor decision, not a free knob. Undersizing surfaces as a named
`BreakableScratchOverflow`, not a wrong answer.

## 2. Arm self-identification — run BEFORE quoting any number

An arm that fails its own check is reported failed, not repaired.

| check | requirement |
|---|---|
| breakable arm actually replayed | `cuda graph: True` ≥ 1 in the decode-batch lines; report the True/False split |
| eager arm really eager | `cuda graph: True` == 0 |
| no capture error | 0 hits of `cudaErrorStreamCaptureUnsupported` / `cudaErrorStreamCaptureInvalidated` |
| resolved config is the one asked for | grep the printed `cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(backend=...` out of the startup `server_args=ServerArgs(...)` line and paste it in — this is the only post-cascade view |
| offload is ON | `[moe-staging-trace] ... experts resident` present; `residency.fetches > 0` and `residency.h2d_bytes > 0` |
| the refuted path is NOT involved | 0 hits of `REFUTED` / `SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE` |
| the breakable route is the one running | the boot must NOT carry `requires --disable-cuda-graph` (that string means the boot died) |
| **the break actually fired** | the route is only real if the MoE fetch ran as a break. Confirm at DEBUG: `Break graph due to function: _moe_offload_fetch_step`, and count them — expect **43 per captured decode step** on DSV4F. A capture that shows 0 is the pass-through case and invalidates the arm |

Start the expert-stats snapshotter before any arm with
`SGLANG_EXPERT_STATS=1` (`expert_stats.py` overwrites one path per rank every
45 s). Headline numbers come from `read_<arm>.txt` captured **before**
teardown, never the SIGTERM revision left on disk.

## 3. F2 — the gating measurement (FIRST)

**Question: what does one MoE layer's break cost, per layer per step, on
DSV4F?** Decomposed, because the remedy differs per term:

| term | what to measure |
|---|---|
| D2H rendezvous | the `topk_ids.tolist()` stall — the irreducible one |
| segment break overhead | ending a segment + reopening one, per `eager_on_graph` call |
| host planning | `_observe_routing` + `split_needed` + `resolve` + `remap_ids_host` |
| pinned publish | the blocking stage → bridge copy |

Method: instrument the break with CUDA events **read one step late** (never
read an event in the step that recorded it — that reintroduces the stall being
measured), plus a host-side wall clock around the four terms. Report per-layer
mean and the 43-layer sum, per rank. Per-rank matters: the capacity split loads
the weakest card most, and overlap only moves the bottleneck.

**The verdict F2 produces:** `43 x (break + rendezvous + planning + publish)`
against the launch-overhead saving the graph buys. Report both numbers and the
ratio. No kill threshold — Aufwand/Ertrag decides, and a small win that is
cheap to keep is still a win.

## 4. Replay-without-recapture gate (SECOND)

Only if F2 leaves the route worth running.

1. Serve N decode steps at a captured bucket. Assert the graph is replayed, not
   recaptured: capture count must stay flat while step count rises.
2. Assert the break function re-ran each replay — fetch counters
   (`residency.fetches`) must advance monotonically across replays. A flat
   counter under rising steps means the eager phase is NOT re-running and the
   graph is replaying stale slots, which is the single most dangerous failure
   mode of this design.
3. Two captured buckets, alternated. Each must keep its own bridge: the
   hermetic `test_preparing_one_bucket_does_not_disturb_another` covers the
   aliasing, but the graph-level version is only observable here.
4. **Correctness, same boot:** greedy prompt, 3 runs, breakable vs eager. Each
   arm must first be internally deterministic (3 identical hashes), then the
   two arms compared. This is the B2 question re-asked for THIS route — #452's
   B2 divergence was never localised, so a divergence here is not
   automatically this route's fault, but it is a stop-and-report.

## 5. ms/verify A/B (THIRD)

Same boot per arm, `--folge 0` and `--folge 1` at every quoted point, so each
point carries its **own** A-vs-A floor. Report nothing whose magnitude does not
exceed its own point's floor; if a delta lands inside the floor, say so — that
is the result.

```
python3 scripts/gpu_battery/s14_decode_punkt.py --port 30462 --out-dir "$RUN" \
  --arm <arm> --bs <N> --folge <0|1> --context-tokens 2048 \
  --ramp-seconds 6 --window-seconds 15 --drain-seconds 5 \
  --server-log "$RUN/boot_<arm>.log"
```

Report **both** `ms/Verify` (per request) and `ms_pro_schritt = bs x ms/Verify`
(wall step time) — at bs>1 they differ by `bs` and quoting the wrong one is a
factor-N error. Accept length comes from the `Decode batch` tick line, not the
EMA on the same line.

Prefill: `s12_prefill_kurve.py --mode messen` then `--mode zusammenfassen`.
Prefill is eager in **both** arms by construction, so a prefill delta is a
red flag about the boot, not a finding.

## 6. Reporting rules

- ms/verify and ms/prefill. Never tok/s.
- F1's 5.3–8.4x prize is a **ceiling measured on Qwen3.6-35B-A3B** and may
  never be quoted as a DSV4F figure.
- The 43 rendezvous/step are irreducible on this route (see DESIGN_462 §4).
  Do not report their removal as an achievable optimisation.
- Corridor readings sampled during load, per card, with the minimum stated.
- Everything still unmeasured stays labelled unmeasured.

## 7. What would retire the route

If F2 shows `43 x break-cost` ≥ the graph's saving on DSV4F, the honest
outcome is: record the number in `NOTE_452` and `FEATURE_CATALOG` §3, leave the
route gated OFF, and close the line. Option 2 (CUDA conditional graph nodes)
remains the only known mechanism that reconciles capturable with
move-only-the-miss, and torch still exposes no API for it — so "closed" here
means closed until that changes, not closed forever.
