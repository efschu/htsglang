# NOTE #544 / #545 — preserve_thinking as a serving default, and a runtime-resizable disk HiCache tier

Status: **DESK COMPLETE, hermetically validated.** Everything below ran on CPU
only (`CUDA_VISIBLE_DEVICES=99`, `PYTHONPATH=<worktree>/python`). No GPU window
was taken, no model was loaded, the live server (PID 1236, port 30030) was not
touched, and nothing here is a live-boot claim. The statements that still need
a boot are named in §7.

Branch `feat/hicache-runtime-544`, cut from `integration/r3-probe-next2`
(tip `09db875dbb`).

---

## 1 — Why

Our Claude-Code Qwen subagents are multi-turn: every tool roundtrip appends a
turn. The Qwen3.6 chat template drops prior-turn `<think>` blocks unless
`preserve_thinking` is passed, so the turn-N rendered prompt stops being a
prefix of "turn-(N-1) prompt + what the model actually generated". The radix /
KV prefix cache then misses and the server re-prefills nearly the whole
context on every single turn. `docs/dev/NOTE_542_chat_template_audit.md` §5
measured the divergence; §3 below re-proves it at render level and adds the
token-level number.

`preserve_thinking` was only reachable as a per-request `chat_template_kwargs`
field. Nothing let an operator make it the serving default. That is work item
B, and it is what gates the pending restart.

Separately (#545), the disk tier of the hierarchical cache had a capacity cap
that could only be set at process start. That is work item A.

---

## 2 — What already existed (do not rebuild it)

Mapped before writing anything; a good deal of the ask was already in the tree.

| capability | state on `integration/r3-probe-next2` | origin |
|---|---|---|
| runtime **attach** of a storage backend | present — `PUT /hicache/storage-backend` | upstream #15892 |
| runtime **detach** | present — `DELETE /hicache/storage-backend` | upstream #15892 |
| storage backend **clear** | present — `POST /hicache/storage-backend/clear` | fork |
| storage backend **status** (config view) | present — `GET /hicache/storage-backend` | fork |
| disk **LRU eviction + size cap** for the `file` backend | present but **boot-time only**, and **off by default** | upstream #26670, #29716 |
| runtime **resize** of the cap | **absent** — this is the #545 gap | — |
| server default for `chat_template_kwargs` | **absent** — this is the #544 gap | — |

So #545 reduced to: make the existing `LRUFileEvictor` re-cappable while the
server runs, and expose that over the same admin-RPC pattern the attach and
detach routes already use.

The cap itself is configured through `--hicache-storage-backend-extra-config`
(`max_size`, `min_free_space`, `eviction_ratio`) or the
`SGLANG_HICACHE_FILE_BACKEND_*` env vars. There is no dedicated `--hicache-*`
CLI flag for it, and when neither `max_size` nor `min_free_space` is set the
evictor is inert and the file backend is unbounded
(`lru_file_evictor.py:93`, `_eviction_configured`).

---

## 3 — Work item B: `--chat-template-default-kwargs`

### Flag

```
--chat-template-default-kwargs '{"preserve_thinking": true}'
```

A JSON object of chat-template kwargs applied to every chat completion before
rendering. Validated at parse time
(`ServerArgs._handle_chat_template_default_kwargs`): malformed JSON, a
non-object payload, or non-string keys all abort the boot rather than
silently degrading.

### Precedence

`serving_chat.merge_chat_template_kwargs(defaults, reasoning_effort, request_kwargs)`
resolves one request, lowest priority first:

1. server defaults from the flag,
2. the request's `reasoning_effort`,
3. the request's own `chat_template_kwargs`.

Per-request values win key by key. That ordering is deliberate and load
bearing: the Anthropic front injects the reasoning toggle through
`apply_reasoning_enabled()`, which writes into
`request.chat_template_kwargs`, so it keeps overriding a server default. With
the flag unset the merge returns exactly what the old code returned, so the
default path is unchanged.

### Reach

Both fronts. The Anthropic front does not render its own template — it
translates to a `ChatCompletionRequest` and funnels through
`OpenAIServingChat`, i.e. the same `apply_chat_template` call site
(`serving_chat.py:869/890`). One change covers OpenAI and Anthropic.

Not covered: `serving_embedding.py`, `entrypoints/ollama/serving.py`, and the
bespoke DeepSeek `dsv4`/`dsv32` encoders in `chat_encoding.py`, none of which
accept arbitrary template kwargs. Irrelevant for this model and this boot.

### Proof

Rendered through the real Qwen3.6-27B-INT8-W8A8 tokenizer, simulating a
two-turn agent exchange. `preserve_thinking` appears twice in that chat
template, so the flag is genuinely wired to model behaviour and not a no-op.

The generation prompt already ends in `<think>\n`, so the model's stream
starts with the reasoning body, not with a `<think>` tag — the test encodes
that exactly rather than assuming a tag.

| render | `turn2.startswith(turn1 + generated)` | token-level common prefix |
|---|---|---|
| without `preserve_thinking` | **False** | 17 tokens |
| with `preserve_thinking` | **True** | 35 tokens = the entire turn-1 stream |

The 17-token figure is the whole point: without the flag the reusable prefix
collapses to the leading system/user framing and everything after it is
re-prefilled.

### Trade-off (explicitly unmeasured)

Preserved think blocks stay in the context permanently, so the conversation
grows faster in tokens than it would with them stripped — the flag trades
context length for prefix reuse. Whether keeping prior reasoning in-context
helps or hurts answer quality is **not measured here**; #541 is the ticket
that quantifies it. The KV-reuse win above is a rendering fact, the quality
effect is an open question.

---

## 4 — Work item A: runtime resize of the disk tier

### Endpoint

```
POST /hicache/storage-backend/resize
```

```bash
# grow the tier to 200 GiB
curl -s -X POST http://127.0.0.1:30030/hicache/storage-backend/resize \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -d '{"max_size_gb": 200}'

# shrink to 50 GiB and keep 20 GiB of filesystem headroom
curl -s -X POST http://127.0.0.1:30030/hicache/storage-backend/resize \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -d '{"max_size_gb": 50, "min_free_gb": 20}'
```

Response:

```json
{
  "success": true,
  "message": "Resized HiCache storage backend successfully.",
  "stats": {
    "configured": true, "enabled": true, "is_storage_owner": true,
    "max_size_bytes": 53687091200, "min_free_bytes": 21474836480,
    "eviction_ratio": 0.9, "used_bytes": 48318382080,
    "num_entries": 12043, "freed_bytes": 53687091200,
    "file_path": "/spinning/hicache"
  }
}
```

Both fields are optional; `None` leaves that limit unchanged. Rejected with
400: both fields omitted, `max_size_gb <= 0`, `min_free_gb < 0`, hierarchical
cache not enabled, no backend attached, or a backend with no capacity
accounting of its own.

### Semantics

**Grow** — raises the ceiling and returns. Nothing is evicted; subsequent
writes simply have more room.

**Shrink** — evicts LRU victims inline through the same `_evict_locked` path
a write-time overflow uses, so the post-shrink target is
`max_size * eviction_ratio` (0.9 by default), i.e. usage lands *under* the new
cap rather than exactly on it. This reuses the existing policy rather than
inventing a second one. The call blocks for the duration of the unlinks and
returns only once usage is back under the cap; `freed_bytes` reports what went.

**In-flight writes are never evicted.** A backup that has `reserve()`d but not
yet `commit()`ed is pinned, so a shrink racing a backup can legitimately land
slightly above target until those writes commit.

**Turning eviction on at runtime works.** An evictor that booted with no cap
is inert and keeps no LRU index (`reserve`/`touch` are no-ops). Setting a cap
therefore seeds the index from disk first, exactly as `__init__` does, before
evicting. This is what makes "boot unbounded, cap later" a usable workflow.

**Removing the cap is not offered.** `max_size_gb: 0` is a 400 rather than a
synonym for unbounded, because "0" reads equally well as "evict everything"
and guessing wrong destroys a cache. Detach and re-attach to go unbounded.

**MLA non-owner ranks** record the new limits but stay inert; rank 0 owns the
shared files and does all the evicting. Under pure TP every rank caps its own
shard directory with the same limits, so ranks differ only in usage — the
endpoint reports rank 0's snapshot.

### Concurrency: why this does not require an idle scheduler

Attach and detach both refuse unless `is_fully_idle()`, because they start and
stop background threads and rebuild backend state. Resize does neither: it
only mutates the evictor's own counters and unlinks files, all under the
evictor's existing `threading.Lock`, which is the same lock the backup and
prefetch threads already contend for via `reserve`/`commit`/`touch`. So the
handler deliberately does **not** demand idleness — an operator can re-cap a
busy server. The cost is that a large shrink delays the next batch by however
long its unlinks take, which is documented on the endpoint.

### Layers touched

`LRUFileEvictor.set_limits()` / `.stats()` → `HiCacheFile.resize()` /
`.capacity_stats()` (with no-op defaults on the `HiCacheStorage` base, so
other backends degrade to a clean "not supported") → `HiRadixCache.resize_storage_backend()`
→ `Scheduler.resize_hicache_storage_wrapped()` → `TokenizerManager.resize_hicache_storage()`
→ the HTTP route. This is the same chain, in the same order, that attach and
detach already use (`ResizeHiCacheStorageReqInput/Output` in `io_struct.py`, a
`_COMMUNICATOR_SPECS` entry, a `TypeBasedDispatcher` entry).

---

## 5 — The GDN hybrid path: what it does and does not support

Qwen3.6-27B is a hybrid GDN model, so `registry.py:106-113` routes it to
`_create_unified_radix_cache` — it gets a **`UnifiedRadixCache`, not a
`HiRadixCache`**. `UnifiedRadixCache` does not subclass `HiRadixCache`; the two
are disjoint hierarchies. This drives everything below and was the single most
consequential thing found while mapping the tree.

### 5.1 The disk tier does work at boot (initial concern retracted)

`docs/rig-runbook.md` §4.8 warns that the HiCache L3 store "does not work
here" for a hybrid GDN model, because a store round trip carries KV pages
while the GDN recurrent state lives in a separate pool. **That paragraph is
about the PD/satellite cross-rig handover and does not generalise to the local
disk tier.** On the unified path the GDN state is carried too:

`UnifiedRadixCache.init_hicache()` → `attach_hybrid_pool_to_unified_cache()` →
`_MambaStrategy.build()` (`hybrid_pool_assembler.py:520-600`) creates a
`MambaPoolHost` with `allocator_type=server_args.hicache_storage_backend` and
registers it against the same `HybridCacheController`, which calls
`register_mem_host_pool_v2(...)` for **every** component — `FULL` and `MAMBA`
alike (`hybrid_cache_controller.py:214-230`). So both the full-attention paged
KV and the GDN recurrent state are backed by the file store.

Corroborated by two independent sources in-tree:

* `test/registered/hicache/test_qwen35_hicache.py` — a CUDA CI test that boots
  Qwen3.5-27B (same GDN architecture) with `--enable-hierarchical-cache
  --hicache-storage-backend file` and runs gsm8k twice with a `/flush_cache`
  in between, i.e. it exercises a genuine store round trip and checks accuracy
  holds.
* `docs/benchmarks/htsglang_tp4.json` — documented healthy live boots of
  Qwen3.6-27B itself with these flags. Weaker evidence: those runs were
  explicitly `max_cached_tokens=0`, so they prove the boot serves cleanly, not
  that the cache round-trips.

Note the CI test sets `--hicache-mem-layout page_first_direct
--hicache-io-backend direct` explicitly. That is not cosmetic: the runbook's
cu13 entry records that `MambaPoolHost` accepts **only**
`layout="page_first_direct"`. The §6 boot command sets both.

### 5.2 Runtime attach/detach do NOT work on this path; resize now does

`UnifiedRadixCache.attach_storage_backend` and `.detach_storage_backend`
(`unified_radix_cache.py:2601-2621`) are hard-coded stubs that always return
`(False, "...does not support runtime HiCache storage attach/detach yet.
Configure hicache_storage_backend at startup instead.")`. The scheduler's
`hasattr` guards pass, so there is no crash — the calls simply always fail.
**For this model, runtime attach and detach of the storage backend are not
available, at boot-time configuration only.**

Resize is different, and that difference is the reason it could be
implemented here. Attach and detach have to build or tear down the hybrid
controller's storage threads and per-component host-pool registrations against
a live tree, which is why they were stubbed. Resize touches none of that: it
moves the backend evictor's own byte counters and unlinks files, under the
evictor's own lock. So `resize_storage_backend` / `storage_capacity_stats`
were added to `UnifiedRadixCache` as well, with the same contract as on
`HiRadixCache`, and runtime resize **is** available for the GDN boot.

Net effect on #545 as it applies to the imminent boot: the tier must be
configured at boot (§6), and from then on its cap is live-adjustable.

Related exclusivity, noted for the record and **out of scope** (task #547):
`--enable-hierarchical-cache` cannot be combined with
`--enable-kv-session-offload` (`server_args.py:6678`) or with
`--weightless-kv-host-spill-tokens` (`server_args.py:6278`) — each is its own
host tier. The current boot uses neither, so there is no conflict today.

---

## 6 — Prepared next boot

Not executed here. `--enable-fast-lane`, `--kv-pressure-ladder`,
`--retraction-policy` and the uneven-TP flags were each checked against every
`enable_hierarchical_cache` validation site in `server_args.py`; none
conflict.

**The file backend's directory has no CLI flag.** `HiCacheFile.__init__` reads
`envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` and otherwise defaults to
`/tmp/hicache`; the backend factory passes only `storage_config` for `file`.
Setting the path via environment is therefore mandatory, not stylistic — miss
it and 100 GB of KV pages land on the container's root filesystem.

**Size suffixes are decimal unless you write `i`.** `human_readable_int`
parses `100G` as 100_000_000_000 bytes (93.1 GiB) and `100Gi` as 107_374_182_400
(100 GiB). The runtime resize endpoint takes GiB. `100Gi` is used below so the
boot cap and a later `{"max_size_gb": 100}` mean the same number.

```bash
mkdir -p /spinning/hicache

export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/spinning/hicache

python -m sglang.launch_server \
  --model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8 \
  --tp-size 3 \
  --rank-gpu-id 0,1,2 \
  --rank-tp-ratio auto-performance \
  --rank-perf-tune phase-decode \
  --rank-auto-reserve-mib 13000,4200,4200 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 262144 \
  --max-running-requests 4 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --kv-pressure-ladder auto \
  --enable-fast-lane \
  --retraction-policy priority \
  --enable-metrics \
  --chat-template-default-kwargs '{"preserve_thinking": true}' \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-storage-backend file \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend-extra-config '{"max_size": "100Gi", "min_free_space": "20Gi"}' \
  --port 30030
```

`--hicache-storage-backend-extra-config` is the place for the cap; it takes
precedence over the `SGLANG_HICACHE_FILE_BACKEND_*` env vars.
`min_free_space` protects the rest of `/spinning` against the tier filling the
volume (293 GB free at time of writing).

`--hicache-mem-layout page_first_direct --hicache-io-backend direct` are
required rather than optional here — see §5.1.

Once up, the cap is live-adjustable without a restart:

```bash
curl -s -X POST http://127.0.0.1:30030/hicache/storage-backend/resize \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $ADMIN_API_KEY" \
  -d '{"max_size_gb": 150}'
```

Also merge `feat/thinking-budget-540` into the boot tree — #540's live
validation is bound to the same restart window.

### What still needs a restart, after this change

| item | restart-bound? |
|---|---|
| the disk HiCache tier, **first** enablement | **yes** — two independent reasons. `--enable-hierarchical-cache` builds the device/host tiers at startup, and the live server has no hicache flags at all, so its scheduler answers every attach call with "Hierarchical cache is not enabled." On top of that, this model's `UnifiedRadixCache` refuses runtime attach outright (§5.2). **This is the thing we assumed was live-attachable and is not.** |
| disk-tier **resize** after boot | no — live, on both cache classes (§4) |
| disk-tier attach / detach after boot | **yes for this model** — `UnifiedRadixCache` stubs both (§5.2). Live only for non-hybrid models on `HiRadixCache`. |
| `preserve_thinking` default | **yes** — see below |
| #540 / #543 | yes |

**`preserve_thinking` did not become runtime-updatable, and it should not be
forced through the existing RPC.** The template render happens in the
tokenizer/HTTP process (`self.tokenizer_manager.tokenizer.apply_chat_template`),
whereas `FanOutCommunicator` fans out to *schedulers*. A `set_internal_state`
-style call would update scheduler state that the renderer never reads. Making
it live means mutating per-HTTP-worker state across N worker processes, which
is a different mechanism, not the same pattern — so it was left restart-bound
rather than shipped half-working. Cheap follow-up if wanted; not cheap enough
to claim as done.

---

## 7 — Validation

Hermetic, CPU only.

Two new test files, 35 tests, all green:

* `test/registered/unit/entrypoints/openai/test_chat_template_default_kwargs_544.py`
  — 14 tests: merge precedence (5), server-arg validation (4), render-level
  prefix identity against the real tokenizer (4), boot-config parse (1).
* `test/registered/unit/mem_cache/test_hicache_runtime_resize_545.py`
  — 21 tests: evictor state machine (9), backend delegation (2), scheduler
  request validation (7), both cache classes expose resize (3).

Whole-suite comparison against the base commit, same command, same box:

```
                                              failed  passed  skipped
base (09db875dbb, clean worktree)                937     741      695
feat/hicache-runtime-544                         937     776      695
```

Identical failure count; `+35` passed is exactly the new tests. The 937 are
`RuntimeError: No accelerator (CUDA, ...)` — GPU-bound tests that cannot run
desk-side, pre-existing and untouched. `test_hicache_nixl_storage.py` is
excluded from both runs: it fails to *collect* because `nixl` is not installed
in this venv, also pre-existing.

Can-fail proof, per gate — mutate, watch the intended test go red, revert:

| mutation | result |
|---|---|
| `merged = dict(defaults)` → `merged = {}` | 2 failed, 12 passed (both defaults-applied assertions) |
| `if not isinstance(parsed, dict)` → `if False` | 1 failed, 13 passed (`test_non_object_json_rejected`) |
| shrink path: drop the `_evict_locked` / `_enforce_free_space_locked` calls | 3 failed, 18 passed (both shrink tests + the runtime-enable test) |
| `if not was_enabled and not self._lru:` → `if False` (no disk rescan) | 1 failed, 20 passed (`test_enabling_eviction_at_runtime_adopts_existing_files`) |
| `if evict_stem in self._pending_writes:` → `if False` | 1 failed, 20 passed (`test_in_flight_write_is_not_evicted_by_a_shrink`) |

All reverted; the suite is green as recorded above.

Lint on every touched file: `ruff check --select F401,F821,UP037` clean,
`codespell` clean. `black` reports `scheduler.py` and `server_args.py` as
needing reformatting — **pre-existing on the base commit** (verified against
`git show HEAD:<file>`); no black-relevant diff falls inside the hunks added
here.

Not validated (needs a GPU window): that a real boot with the §6 flag set
comes up; that the disk tier actually serves a restored prefix across a
restart on this hybrid GDN model (§5.1 argues it should from code and CI
evidence, but that is not the same as having measured it here); that the
resize endpoint round-trips against a live server.
