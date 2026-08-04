# RUNSHEET 544 — disk-HiCache serving restart

Boot recipe and validation record for the user-ordered restart of the 30030
serving instance. Branch `boot/hicache-preserve-544`.

Scope: the **current** serving layout (context 262144, NEXTN MTP, existing
parser flags) plus

* a disk HiCache tier at `/spinning/hicache`,
* `preserve_thinking` as a server default (from `feat/hicache-runtime-544`),
* the #540 per-request thinking budget, merged into this branch.

Explicitly **not** in this boot: the #543 YaRN long-context layout, and
kv-session-offload. `--enable-kv-session-offload` and
`--enable-hierarchical-cache` are mutually exclusive
(`server_args.py:6664-6667`), so the #543 spill lane and this disk tier cannot
share a boot. That exclusivity is its own composability task.

## Flag decisions and why

| decision | reason |
|---|---|
| `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/spinning/hicache` | the file backend takes its directory from the environment, not from a flag, and defaults to `/tmp/hicache` (`mem_cache/hicache_storage.py:409-412`). `/spinning` has 293 GB free. |
| `--hicache-size 24` | `--hicache-size` is the **host L2 pool in GB** and overrides `--hicache-ratio`. Left on the 2.0 ratio it tracks the device pool to roughly 20 GiB of host memory as a derived quantity; pinned explicitly instead. Host RAM is 98 GB with about 23 GB in use, so 24 GB is comfortable. |
| `--hicache-write-policy write_through`, `--hicache-mem-layout page_first`, `--hicache-io-backend kernel` | the live defaults, carried over unchanged. `page_first_direct` is deliberately avoided: it was the layout blocked on this rig by the #436 `cudaMemcpyBatchAsync` ABI segfault. |
| context 262144, NEXTN MTP, parser flags unchanged | the restart is additive; nothing about the serving shape changes. |

`prepare_server_args` accepts the full command hermetically on this tree:
`hicache True / file / 24`, `spec EAGLE`, `ctx 262144`, `kvso False`. hicache ×
NEXTN is not gated here.

## Observability gap (follow-up candidate)

**The hierarchical cache emits no Prometheus series at all.** Grepping
`observability/metrics_collector.py` for `hicache`, `storage`, `prefetch` or
`storage_hit` returns nothing; the only runtime signals are two log lines
(`hicache_storage.py:459`, `:836`). There is therefore no way to answer "is the
L3 disk tier being hit, and at what rate" from `/metrics`.

Consequences:

* Activation can only be proven indirectly — by the boot marker plus files
  actually appearing under the storage directory. `validate_544.sh` does both.
* Hit rate against the disk tier is not separable from the existing
  `sglang:cache_hit_rate` / `sglang:cached_tokens_total`, which are
  radix-level and do not distinguish L1 from L2 from L3.
* Capacity and eviction on the disk tier are unmeasured; nothing reports the
  directory's size or the eviction rate back to the operator.

Worth a task: L2/L3 hit, miss, write and eviction counters plus a
`storage_tier_bytes` gauge, so a disk tier can be operated rather than merely
enabled. Related precedent: the kvso spill tier does expose
`sglang:spill_tier_used_bytes` / `sglang:spill_tier_total_bytes`
(`metrics_collector.py:733-750`), so the pattern to copy already exists in tree.

## Boot

```bash
scripts/dev/543_yarn/boot_hicache_preserve_544.sh
```

Reads `PRESERVE_THINKING_FLAG` from the environment so the #544 flag can be
dropped in without editing the recipe. Kills nothing by itself — the operator
stops pgid 1236 first, and **must not touch translator PID 30439**.

## Validation

```bash
scripts/dev/543_yarn/validate_544.sh
```

Covers: health and identity; boot markers by named grep only; short-context
sanity; disk-tier activation by file landing; the `preserve_thinking` two-turn
prefix-reuse probe
(`/spinning/gpu-battery-results/2026-08-04_541_thinking_ab/probe_preserve_thinking.py`);
the #540 budget check (`reasoning_tokens`, overshoot against draft_token_num 4);
a 100 ms-sampled VRAM corridor check against the 400 MiB floor; host RAM read
from the cgroup rather than `free`; and translator-tenant liveness.

**Read RAM from the cgroup, not from `free` or `/proc/meminfo`.** Both are
lxcfs-distorted in this container — they report 120 GB against a real 98 GB.
Any host-pool plausibility guard that reads `psutil.available` is therefore
blind here, which is why the 24 GB L2 pin is explicit rather than derived.

## Results

Pending — not yet booted.
