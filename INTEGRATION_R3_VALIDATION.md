# integration/r3-probe — validation record

Hardware: 1x RTX 5090 (sm_120) + 2x RTX 3080 (sm_86), uneven TP=3, uneven DCP.
Model: Qwen3.6-27B-FP8. Every arm run with **CUDA graphs + speculative
decoding** (not eager), `--rank-tp-ratio auto-performance`,
`--rank-kv-ratio capacity`.

## Boot matrix — after merging feat/htccl-port

| arm | axis added | result | key figure |
|---|---|---|---|
| A | default regression (no new flags, HTCCL unset) | GREEN | budgets `[26107,18280,18280]`, ownership `[18,23,23]`, `max_total_num_tokens=98328`, accept 3.69/2.82/3.28 |
| B | `--enable-kv-session-offload` + P2 host-RAM budget | GREEN | P2 pool 1.00 GB/rank of 24 GiB; accept level with A |
| C | cross-algo + lazy single capture | GREEN | lazy capture ACTIVE, adaptive graph memory released 542.0 MiB; accept 3.24/3.79/4.31 |
| D | `--kv-session-offload-spec-in-tick` x cross-algo | GREEN 9/9, 0 INVALID | tps p50 82.95/77.88/91.91 |
| E | `SGLANG_HTCCL_TRANSPORT=device` + spec | GREEN | calibration `[2,1,2]` on all ranks; accept 3.45/2.75/3.33 |
| G | all three axes together | GREEN | HTCCL + cross-algo + offload + spec-in-tick under graphs; accept 3.65/3.10/4.23 |
| H | PS2 deep prefill-spill without spec | GREEN | PS2 fired x3; `released device head=961 (boundary=961 protected=0)` |
| I | DFLASH per-rank shards x spill | GREEN | MLP vector `2,1,1`, units `[68,34,34]`, capacity `[64758,198244,198244]` |
| J | P1 wave-back threshold x PS2 | GREEN | wave-back armed x3; `released device head=0 (boundary=0 protected=0)` |

`cuda graph: True` observed in the decode lines of every arm.

## Defects found and fixed on this branch

1. **HTCCL shared output buffer** — `_get_out_buf` returned a persistent
   per-(shape, dtype) buffer, so two same-shape `all_reduce` results were the
   SAME tensor. Corrupted the model forward outright (garbage tokens, HTTP
   200, no crash, no hang) on every non-device transport. Not shm-specific:
   `gloo` reproduced it; `device` escaped only because it allocates fresh.
2. **HTCCL device extension built for one GPU arch** — `load_inline` with no
   arch flags and a name-only cache key; under per-rank
   `CUDA_VISIBLE_DEVICES` each worker sees one card, so the first compiler
   fixed the arch for everyone and the sm_86 ranks died on their first kernel
   launch. Invisible on a uniform-GPU rig.
3. **spec-in-tick pool starvation** — `--kv-session-offload-mtp-resident-slices`
   removed its slots from the allocator permanently, unbounded against the
   pool. 2048 of 3600 left 1552 against a 2048-token prefill chunk, so no full
   prefill could ever be assembled: requests accepted, queued, never run. Not
   a collective desync (577k-entry collective sequences were rank-identical
   and all ranks kept advancing during the wedge).
4. **PS2 born-spilled off-by-one** — `output_ids` is empty at a born-spilled
   session's first tick under the overlap scheduler; the tick now defers one
   iteration. Decode-spill arithmetic unchanged.
5. **Spilled request donated sentinel rows to the radix tree** — the
   unfinished-cache seam lacked the guard its finish-path sibling already had,
   so host sentinels entered the device radix tree (`evictable=4351` against
   `total=3600`). Also repaired a second leak: the inflated protected length
   made the device-head free a no-op.
6. **HTCCL capture-time broadcast (cross-feature, merge-only)** — the
   speculative draft-pick sync fell back to HTCCL's host-staged broadcast
   inside a CUDA-graph capture once PyNccl was suppressed under HTCCL.
   Green on either branch alone. The device transport now implements a
   capturable broadcast built on all_gather + src slice (byte-exact for the
   int64 `topk_index`, which the reduce kernel does not dispatch).

## Not validated — carried forward as open, not as passing

* **R1** (spec-in-tick spill scenario) — never triggered; no spill ever
  coincided with the tick. A misconfiguration now fails fast; the scenario
  itself remains untested.
* **R2** (rung switching under offload) — 0 rung switches with and without
  offload, 0 `swap_reject` events. Untriggered, neither confirmed nor refuted.
* **3-session co-residency** — unreachable on this scheduler: admission
  follows `3 x prompt_len + ~0.73 x sum(max_new) <= max_total_tokens`, and
  `--schedule-conservativeness` does not scale that reservation. Proven NOT to
  be the offload feature by a flag-OFF control showing the identical ceiling.
  The spill lifecycle is therefore demonstrated at two co-resident sessions
  (host floor ratio 0.238, against an independent 0.284 from arm B).

## Pre-existing test failures (unrelated, environment-bound)

`test/registered/unit/distributed`: 4x `test_vmm_utils` `KeyError: 'LOCAL_RANK'`
and 1x `test_uneven_tp_nccl_env` MPS-pipe. Verified identical before and after
the merge by materialising the pre-merge sources and re-running.
