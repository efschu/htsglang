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
| `--hicache-ratio 2` | NOT `--hicache-size`: that flag is per pool per rank and OOM-killed the first boot. See the section below. |
| `--hicache-mem-layout page_first_direct`, `--hicache-io-backend direct` | mandatory for a GDN hybrid and measured working — see the layout section below. The earlier `page_first` + `kernel` choice was wrong on both counts. |
| `--hicache-write-policy write_through`, `--hicache-storage-prefetch-policy timeout` | live defaults, carried over. |
| `--hicache-storage-backend-extra-config '{"max_size": "100Gi", "min_free_space": "20Gi"}'` | without `max_size` the disk tier grows unbounded. `100Gi` (93.1 GiB), not `100G`. |
| context 262144, NEXTN MTP, parser flags unchanged | the restart is additive; nothing about the serving shape changes. |

`prepare_server_args` accepts the full command hermetically on this tree:
hicache `True / file`, layout `page_first_direct`, io `direct`, spec `EAGLE`,
ctx 262144, kvso `False`. hicache × NEXTN is not gated here.

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

Carries `--chat-template-default-kwargs '{"preserve_thinking": true}'` from
`feat/hicache-runtime-544` (commit `470b88e0a2`, merged into this branch). Kills
nothing by itself — the operator stops the serving pgid first, and **must not
touch translator PID 30439**.

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
blind here — nothing will stop an oversized host pool before the OOM killer
does, as the first boot attempt demonstrated.

## Layout: page_first_direct + direct, and why the first choice was wrong

An earlier revision of this runsheet specified `page_first` + `kernel`, citing
#436. That was wrong in both halves and is corrected here.

* `MambaPoolHost` **raises** unless `layout == "page_first_direct"`
  (`memory_pool_host.py:96-100`), and the hybrid assembler constructs it
  **unconditionally** for a GDN model (`hybrid_pool_assembler.py:551`). With
  `page_first` this boot would have failed hard — it would not have quietly
  degraded to a KV-only tier.
* `page_first` + `kernel` is the route with an **open, unfixed segfault on both
  the cu12 and cu13 wheels** (`transfer_kv_all_layer_lf_ph`, runbook §2). That
  was the combination originally chosen.
* #436 blocked the *direct* route on **cu12**; the cu13 rebuild fixed it.

Measured on this rig before booting (cu13 wheel, sgl_kernel 0.4.4, RTX 3080),
`transfer_kv_all_layer_direct_lf_pf` through
`MHATokenToKVPoolHost(layout="page_first_direct")`:

| host memory | stream | result |
|---|---|---|
| pageable | default | `cudaMemcpyBatchAsync failIdx=SIZE_MAX invalid argument` |
| pageable | side | `CUDA error: invalid argument` |
| pinned | default | `cudaMemcpyBatchAsync failIdx=SIZE_MAX invalid argument` |
| **pinned** | **side** | **OK — the production shape** |

Production satisfies both conditions: host pools default to `pin_memory=True`
(`pool_host/mha.py:97`), and the copy runs inside
`with device_module.stream(self.write_stream)`
(`cache_controller.py:276`, `:742-749`).

So **#441(b)'s guard is not merely stale**. The test fails for real, because it
allocates pageable host memory (`pin_memory = io_backend == "kernel"`) and
copies on the default stream. Unskipping it alone leaves it red; making it
production-shaped is the actual follow-up. The guard was therefore restored with
the matrix above as its documented reason.

## `--hicache-size` is per pool, per rank — it OOM-killed the first boot

The first boot attempt died with `Rank 0 scheduler died during initialization
(exit code: -9)` — the OS OOM killer.

`--hicache-size` is **not** a node-wide budget. It is applied per host pool per
rank (`memory_pool_host.py:121-126`), and a GDN hybrid builds **two** host pools
(KV and Mamba) plus a third for the NEXTN draft KV. On TP=3 the requested 24 GB
therefore asked for roughly 6 × 24 GB of pinned host memory on a 98 GB box.

Fixed by using `--hicache-ratio 2` instead, which sizes each host pool relative
to its own device pool. Resulting pinned total, measured: **shmem 36.7 GB**.

**Read `anon + shmem`, not `memory.current`.** After boot `memory.current` reads
88.0 GB of 98, which looks alarming and is not: 31.9 GB of that is reclaimable
page cache from reading the 29 GB checkpoint. Non-reclaimable is
`anon 18.8 + shmem 36.7 = 55.5 GB`, leaving about 42 GB of real headroom.

## Results — boot 2026-08-04T11:25Z, green

Boot command: `scripts/dev/543_yarn/boot_hicache_preserve_544.sh`. Server up 55 s
after launch. Serving pgid 1236 stopped, no orphans; translator PID 30439
untouched throughout.

Configuration confirmed live on `/get_server_info`:

```
enable_hierarchical_cache            True
hicache_storage_backend              file
hicache_ratio                        2.0          (hicache_size 0)
hicache_mem_layout                   page_first_direct
hicache_io_backend                   direct
hicache_storage_backend_extra_config {"max_size": "100Gi", "min_free_space": "20Gi"}
chat_template_default_kwargs         {"preserve_thinking": true}
context_length 262144   max_total_num_tokens 333254   speculative_algorithm EAGLE
enable_kv_session_offload            False
```

| check | result |
|---|---|
| health | ok |
| storage backend created | `Creating storage backend 'file' (HiCacheFile)` on TP0, TP1, TP2 |
| pools | KV 333254 tokens (2.54/1.27/1.27 GB), Mamba 37 slots — unchanged from the previous boot |
| **disk tier active** | files under `/spinning/hicache` **381 → 12270**, **271 MB** written |
| cache hit rate | 0.978, `cached_tokens_total` 1983 |
| short-context sanity | "Lisbon, 391" — correct (17 × 23 = 391) |
| **preserve_thinking** | kwarg **has effect**: turn-2 prefix reuse 77.2 % vs 6.4 % between the two variants |
| #540 budget | reasoning_tokens 70 / 137 / 262 for budgets 64 / 128 / 256 |
| VRAM corridor (100 ms sampling) | min free 3009 / 3577 / 3013 MiB — all far above the 400 MiB floor |
| host RAM | anon 18.8 GB, shmem 36.7 GB, 55.5 GB non-reclaimable of 98 GB |
| translator tenant | PID 30439 alive |

### Two findings for the owning tickets

**#540 — the budget overshoot bound is understated.** Overshoot is bounded and
near-constant (6, 9, 6 tokens at budgets 64, 128, 256), but the stated criterion
is "≤ draft_token_num", i.e. ≤ 4. The gap is explained: the closing marker is
itself 1-3 tokens (`</think>` = 1, `</think>\n\n` = 2, `\n</think>\n\n` = 3) and
is counted into `reasoning_tokens`, on top of the 4-token NEXTN verify
granularity. 4 + 3 = 7 brackets the observed range. The bound should read
`draft_token_num + marker_tokens`, not `draft_token_num`. Not a serving blocker.

**#544 — the preserve_thinking spread is inverted against the probe's own
prediction.** `probe_preserve_thinking.py` predicts that preserving thinking
keeps turn-2's prefix intact and therefore yields HIGH reuse, while stripping it
collapses reuse. Measured is the opposite: `preserve_thinking: false` gave
77.2 % reuse and `true` gave 6.4 %. The probe's discriminator fires — the kwarg
demonstrably reaches the template, which is what it was built to detect — but the
direction contradicts the documented mechanism, so either the prediction or the
render is wrong. Worth resolving before the reuse figure is quoted as a benefit.
Note the server now defaults to `preserve_thinking: true`, so the per-request
`false` in variant A is an override of that default, not of a bare server.

### Validation-script artifacts fixed after this run

Section 3 first reported an empty answer. That was the probe's own
`max_tokens: 60` being consumed entirely by the thinking block, not a serving
fault — raised to 400 and the answer is correct. Section 1's inline Python had a
quoting bug and printed a SyntaxError instead of the config table; fixed.
