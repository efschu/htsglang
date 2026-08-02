# club-3090 bench suite — DeepSeek-V4-Flash, 2026-08-02

`scripts/bench.sh` run **unmodified**, no copy made, defaults RUNS=5 / WARMUPS=3.
Exact CLI in `bench_cli.txt`. Output scrubbed of the model path in
`bench_club3090_scrubbed.log`.

## Results

| prompt | wall_TPS | decode_TPS | CV | TTFT |
|---|---|---|---|---|
| narrative (1000 tok) | 7.10 | 7.15 | 1.4 % | 913 ms |
| code (800 tok) | 7.15 | 7.21 | 1.0 % | 929 ms |

GPU state at finish: 82 / 94 / 87 % util, 19841 / 30829 / 19841 MiB,
212 / 142 / 205 W, 77 / 58 / 83 °C.

## Deviations from the standard setup

1. **Engine is sglang (htsglang fork), not vLLM.** Endpoint-first mode.
   Model: DeepSeek-V4-Flash-0731, GGUF unsloth dynamic **UD-IQ3_XXS** —
   verified against the load log (IQ3_XXS, not Q3_K_XL). `--served-model-name`
   was not set, so the served id is the checkpoint path itself.

2. **Geometry: TP=3 across mismatched cards** — one RTX 5090 (x8) and two
   RTX 3080 (x4 / x8), `--rank-tp-ratio auto` (uneven shards sized per card),
   KV fp8_e4m3, context 8192, `--max-running-requests 1`,
   `--chunked-prefill-size 512`.

3. **Eager, not CUDA graphs.** `--disable-cuda-graph` is required: the MoE
   expert offload's default fetch path takes a device->host sync per forward,
   which a captured graph cannot contain. The published baseline for this
   configuration is likewise eager, so this is comparable, not degraded.

4. **RAM-tier expert offload active.** Resident expert fraction
   0.485 / 0.42 / 0.42 per rank; the rest live in a pinned host pool and stream
   over PCIe on demand. Decode is PCIe-bound here, not VRAM-bandwidth-bound, so
   the absolute tok/s is NOT comparable with a fully resident model of the same
   size. Over this bench the three ranks moved **17.8 TiB** of expert weights
   host->device.

5. **Link-proportional cold-expert sharding (#394) is NOT ACTIVE.** This is the
   equal-shard baseline. The proportional arm was built and does resolve its
   ratio, but it cannot serve on this model: under the #82 expert-dim shard the
   ranks hold disjoint expert ranges, so a delegated cold expert is not
   relocated to a peer — it is absent, and the first token routed to it fails.
   The arm is refused at boot rather than crashing at the first forward. No A/B
   delta is reported because there is no second arm to compare against.

6. **Speculative decoding: OFF** — `speculative_algorithm=None` at launch (read
   from the boot log, not assumed): no draft model, no NEXTN, no steps or topk.

7. **Docker log scrape skipped, by the script's own design.** The closing
   "SpecDecoding metrics" section reads `docker logs $CONTAINER`; there is no
   container, so `CONTAINER=none` was passed, which the script already handles
   with a guarded skip. It reports `PP tok/s n/a` for the same reason (it
   scrapes vLLM logs); use `PP=1` for the long-prompt fallback if that number
   is wanted.

8. **Model path redacted** to `<models-cache>` throughout the scrubbed log.

## The number worth keeping

CV is **1.0–1.4 %** at 800–1000 generated tokens, against ~5 % on a 96-token
probe in the same boot. Measurement length, not the rig, set the earlier noise
floor. **A future #394 A/B should use bench-length generations**, where the
predicted effect is far above the floor.

Per-rank H2D over this bench, the direct measure of what #394 targets:

| rank | H2D | share | link | implied transfer |
|---|---|---|---|---|
| tp0 (5090, x8) | 7502 GiB | 42.1 % | 14.42 GB/s | 559 s |
| tp1 (3080, **x4**) | 5155 GiB | 28.9 % | 6.45 GB/s | **858 s** |
| tp2 (3080, x8) | 5181 GiB | 29.0 % | 13.41 GB/s | 415 s |

The x4 rank is the clock at **1.40x the mean**; ideal proportional placement is
559 s on every rank, i.e. **1.54x recoverable on the transfer term** for this
long-generation workload (the short-probe mix gave 1.36x).
