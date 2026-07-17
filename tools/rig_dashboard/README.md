# Rig-Dashboard

A lightweight, self-contained live view of the heterogeneous 3-GPU rig
(2× RTX 3080 20 GB + 1× RTX 5090 32 GB) and the uneven-TP shard plan that
sglang materializes at boot.

One small Python process + one HTML page. **No CDN, no npm, no build step.**
Only stdlib + optional `pynvml` (falls back to parsing `nvidia-smi`).

## Start (one command)

```bash
python3 tools/rig_dashboard/server.py \
    --sglang   http://127.0.0.1:30010 \
    --boot-log /path/to/your/sglang_server.log
```

Then open <http://127.0.0.1:8770/> . The page auto-refreshes ~1 Hz.

Everything is optional and degrades gracefully:

* no `--sglang` reachable  → dashboard shows NVML + boot-log plan only.
* no `--boot-log`          → dashboard shows live NVML (+ live server) only.
* neither                 → dashboard shows live NVML alone (idle cards render
  a near-empty bar, which is correct).

### Flags

| flag | default | meaning |
|------|---------|---------|
| `--host` / `--port` | `127.0.0.1` / `8770` | where to serve the page |
| `--sglang` | `http://127.0.0.1:30010` | sglang base URL (blank disables scraping) |
| `--boot-log` | `$RIG_BOOT_LOG` | sglang boot log to parse for the uneven-TP plan |
| `--q-heads` / `--kv-heads` / `--gdn-v-heads` / `--gdn-k-heads` / `--vocab-units` | Qwen3.6-27B defaults | model geometry used to *draw* the head/unit split |

## Data sources

1. **NVML live** (`pynvml`, ~1 s poll): per card VRAM used/total, temperature
   (80 °C mark drawn), utilization, power, PCIe gen/width, name, UUID.
2. **sglang live** (`/get_server_info` + `/metrics`, robust to server-down):
   model name, TP/DCP, `max_total_num_tokens`, tok/s (`sglang:gen_throughput`),
   running/queued requests, KV used tokens + token usage, spec accept length,
   cache-hit rate.
3. **Uneven-TP plan** — parsed from the **boot log** (`plan_parser.py`),
   **streamed, never slurped**: parsing stops at the "server is fired up"
   line or after 128 MB, whichever comes first (live crash-loop logs of
   17 GB have been observed). The server caches the parsed plan and reparses
   at most every 15 s while a boot is in progress; a completed boot's plan is
   final and only reparsed if the file shrinks (new boot). The
   materialized splits (MLP units, vocab ratio, DCP token vector, per-rank
   weight/KV/mamba GB, per-rank profiled capacity) are computed inside the TP
   worker processes at boot and printed to the log; they are **not** in
   `ServerArgs`, so `/server_info` cannot expose them. Parsing the log is what
   lets the dashboard work against an **unmodified** server. See
   "Why a boot-log parser instead of a `/uneven_plan` endpoint" below.

## What you see

* **VRAM per physical GPU** — a stacked bar per card, scaled to the card's NVML
  total. Segments are **measured GB from the boot log**: weights, draft/MTP,
  KV pool (with a live-filled overlay from `kv_used_tokens`), mamba pool, and
  free/unaccounted. A red line marks current NVML `used`. Co-located ranks
  (duplicate `--rank-gpu-id`) are summed onto their shared card. Tooltips give
  GB and %.
* **Head / unit distribution** — Q heads as boxes colored by owning rank.
  Under replicated KV (TP > kv-heads) the split is taken **materialized** from
  the boot log's `REPLICATED-KV geometry active ... q heads split [8, 4, 4]`
  line; under sharded KV it is derived (whole-GQA-group rule, e.g. 24 →
  12/6/6) — the source is labelled either way. Plus KV heads (sharded or
  replicated), GDN value units, MoE expert ownership (materialized from
  `rank N owns experts [a, b) of E` lines), MLP units (materialized), and
  vocab split.
* **Token owner** — one 64-token DCP block colored by owning rank
  (e.g. 30/17/17), with the "repeats every 64 tokens" legend and per-rank
  profiled capacity.
* **Live numbers** — tok/s, running/queued reqs, accept length, max total
  tokens, KV usage, cache hit, TP/DCP, plus per-card temp/util/power/PCIe.

Dark and light themes (follows OS; toggle button in the header).

## Note on physical-index correlation

The server's boot-time GPU enumeration and the live NVML order can diverge
(observed live: sglang's GPU 0 was the RTX 5090, while system NVML had it at
index 1). The dashboard therefore maps boot-log GPU indices onto live NVML
cards (`map_plan_gpus_to_nvml`): first by unique card name from the
auto-performance block, then by per-rank memory-budget best-fit (a 29607 MiB
budget can only live on the 32 GB card), falling back to identity. The GPU
name is shown on each card so any residual mismatch stays visible.

## Why a boot-log parser instead of a `/uneven_plan` endpoint

The task allowed adding a server-side `/uneven_plan` endpoint if the plan were
not already exposed. It is **not** in `/server_info` — but the authoritative
materialized numbers (token vector, per-rank KV/weight GB, profiled capacity)
are produced inside the **TP worker processes**, not the tokenizer-manager
process that serves HTTP. Exposing them via an endpoint would require new IPC
plumbing through `get_internal_state()` into a compute-adjacent path, which is
exactly the kind of change the project rules mark as risky and which cannot be
validated without a live multi-GPU boot. The boot-log parser recovers the same
numbers losslessly, is fully unit-tested against real logs with no GPU, and has
the extra benefit of working against **stock, unmodified** servers. So no
server code was changed. `/server_info` still supplies the live ServerArgs-level
plan (rank_gpu_id, rank_tp_ratio, rank_mlp_ratio, rank_vocab_ratio, TP/DCP).

## Tests

```bash
python3 tools/rig_dashboard/test_plan_parser.py     # 10 tests vs real boot logs
# or: python3 -m pytest tools/rig_dashboard/test_plan_parser.py -q
```

The parser is validated against real M38 boot logs (AWQ / FP8 full plans,
GGUF pinned-partial, and an early minimal boot) including the subtle case where
the DCP log line contains both a restart *recommendation*
(`SGLANG_UNEVEN_TOKEN_VECTOR=…`) and the *active* materialized vector — the
parser must take the active one.
