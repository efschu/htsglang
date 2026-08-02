# #443 — the V4-Flash decode-graph proof window

The DeepSeek-V4-Flash GGUF recipe has run with `--disable-cuda-graph` since the
expert offload landed, and BENCH_394 records why: the eager fetch path reads
`topk_ids.tolist()` once per layer per forward, and a capture cannot contain a
host read. That sync is the ranked-#2 cause of the 2.6x decode gap against the
club-3090 reference.

#443 ports the V4 path onto the capturable path the catalog already lists
(§3, "CUDA-graph-compatible path EXISTS"): frozen residency, on-device index
math, and a captured `index_select` over a UVA view of the pinned pool. It
invents nothing — it removes the reasons the existing path could not be used
here, and pins the properties that make it safe.

## Already proven, no card involved

`tests/moe_offload/test_capture_desync_port.py` — 63 tests, ~16 s,
`CUDA_VISIBLE_DEVICES=99`.

| pins | how |
|---|---|
| no host read survives in the ported step | `torch.Tensor.{item,tolist,cpu,numpy,nonzero,__bool__,__int__,__float__,__index__}` intercepted around a full `prepare_capturable`; call list must be empty, armed seam included |
| the port moves data, not math | the captured gather lands byte-identical scratch rows vs the eager `_fetch`, and every remapped id addresses the row of the expert it routed to |
| the capture-admission bound is tight | `worst_case_unique_spill` = min(routed slots, cold set); reachable and not exceedable over 200 adversarial draws |
| host-only branches are decided by capture state | the trace and `run_waves` are both structurally off under capture; without the opt-in, capture never reaches the ported step |
| the #394 seam cannot read a row it does not own | delegated expert counted on device, index clamped, raised by name at the replay boundary (#431 pattern) |

Executed can-fail arms (each reverted after):

| revert | result |
|---|---|
| `worst_case_unique_spill` → `return slots` | 5 tests fail |
| drop the `src_row.clamp(min=0)` | 4 tests fail, and the index really is out of bounds (`IndexError`) |
| test the breach count on the host inside the step | 1 test fails, catching a `__bool__` |
| `cumsum(...) - 1` → `cumsum(...)` (off-by-one slot) | 48 tests fail across this file and `test_capturable_planner.py` |

## Only a boot can show these

| id | claim | readout |
|---|---|---|
| B1 | capture SUCCEEDS where it previously refused | `ARM=graphs` boot log: `DecodeCudaGraphRunner` captures its bucket list, no `cudaErrorStreamCaptureUnsupported` / `...Invalidated`, no `MoE expert-offload ... requires --disable-cuda-graph` |
| B2 | replay is CORRECT | greedy decode, fixed prompt + seed, `ARM=graphs` vs `ARM=eager`, same boot day and same residency → identical text expected (data movement, not math) |
| B3 | the UVA view holds under capture | B2 covers it observationally; direct check is to poison one pinned row between capture and replay and confirm the output changes |
| B4 | perf | ms/round for decode, per rank, both arms. #2 of the ranked causes is not #1: a partial recovery is the honest expectation, and the number to publish is the measured one |

The A-vs-A floor comes first: two `ARM=eager` boots before any arm comparison,
per the benchmark-harness rule. BENCH_394's 7.15 / 7.10 decode TPS is the prior
day's figure and is not the floor for this window.

## Run order

```
python3 scripts/dev/443_graph_proof/scratch_preflight.py \
    --experts-per-rank <from the boot log's per-rank #82 range> \
    --resident-fraction 0.485,0.42,0.42 \
    --top-k <model top_k, read from the checkpoint> \
    --max-graph-bs 1 --scratch-slots 6
bash scripts/dev/394_s2_proof/preflight.sh     # NVML table -> RANK_GPU_ID
ARM=eager  bash scripts/dev/443_graph_proof/boot_graphs.sh
ARM=graphs bash scripts/dev/443_graph_proof/boot_graphs.sh
```

`scratch_preflight.py` calls the same `worst_case_unique_spill` the runtime
calls, so the preflight and the capture-admission check cannot drift. Without
it the first thing a wrong `SGLANG_MOE_SCRATCH_SLOTS` produces is a refusal
several minutes into warmup, after a full GGUF load.

## Sizing: capture is not free, and the recipe's own operating point fits

For decode buckets the cold set is never binding — `bs * top_k` is — so the
requirement is simply `bs * top_k` scratch slots, and each slot is one more
expert's worth of resident VRAM per layer.

`DeepSeekV4Config` defaults to `num_experts_per_tok = 6`
(`python/sglang/srt/configs/deepseek_v4.py:74`). CONFIRM it against the
checkpoint before sizing — a class default is not a checkpoint. If it holds for
V4-Flash-0731, then at `bs=1` the requirement is exactly 6, which is what the
2026-08-02 battery already ran (`SGLANG_MOE_SCRATCH_SLOTS=6`), and the recipe's
own operating point (`--max-running-requests 1`) is capture-eligible at no
extra VRAM. Every larger bucket costs `6` more slots per unit of `bs`, which is
why `boot_graphs.sh` caps the captured list instead of leaving it at the
default — and why raising `MAX_GRAPH_BS` is a corridor decision, not a tuning
knob.

## Not in this window

The shared cold tier (`SGLANG_MOE_COLD_TIER_SHM`) stays OFF. Capture over
peer-owned cold rows is refused by name: the captured gather has no peer
source, and the peer's `cudaHostRegister`'d UVA pointer is unverified. Opening
both at once would make any failure un-attributable.
`SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE=1` is a later window's switch; past it, a
routed delegated expert is counted on device and raised by name at the replay
boundary rather than silently reading local row 0.
