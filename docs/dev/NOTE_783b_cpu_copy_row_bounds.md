# #783b — a Silently-Wrong defect whose crash was the only reason we found it

**`test_a_negative_row_does_NOT_fault_and_hits_the_last_row` PASSES TODAY.**
`-1` is simultaneously the classic stale-row sentinel and valid torch indexing,
so a stale id silently writes the LAST row of the pool — wrong KV under a prefix
the tree reports as restored, no crash, no log line. **The IMA we chased is the
LOUD half. The silent half is refused nowhere, and it is on a shipped path.**
The same control passes on the mamba axis
(`test_a_negative_slot_does_NOT_fault_and_hits_the_last_slot`).

This is not "a crash to fix". It is Silently-Wrong class, and the crash was the
instrument that exposed it.

Prepared read-only while R6 held the tree. Land order: R6 first, then re-run the
A/B against R6's new tip — the reds below were measured against a tree carrying
R6's in-flight #858 edits.

## The mechanism

`restore_seam_state` (schedule_batch.py:2043) compares `kv_cache_cpu_extent`
against `_seam_extent_of(req)` — one scalar, a row COUNT — and then forwards
`req_to_token[req_pool_idx, : seqlen - 1]` to the pool as ids that nothing
validates. **Equal length is not equal validity.** Two outcomes:

* `id >= rows` — on CUDA an **asynchronous** illegal memory access, surfaced by
  whatever call synchronizes next. W40b (19:52:52, three PP ranks, instance down
  5 s later), traceback naming `synchronize()` and not the store that caused it.
* `id < 0` — **not a fault at all.** See the top of this note.

## Decision: beside `:3298`, not subsuming it

`assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)` compares the
saved host chunk's SHAPE against the chunk LENGTH; the guard compares index
VALUES against the pool's bound. Orthogonal — neither implies the other, both
stay. 25e7849844 already called `:3298` "per-chunk, on the wrong axis entirely.
There is no backstop below." That citation settles it.

## Decision: a module function, and the bound is an argument

`check_cpu_copy_rows(indices, rows, direction, axis)`. **A module function, not
a method**, because the pools that need it do not share a base: `MHATokenToKVPool`
and `MLATokenToKVPool` are `KVCache` subclasses and `MambaPool` is not. A guard
installed on one ancestor would be silently inert on the other.

The bound and the axis are passed in because they genuinely differ: MHA/MLA are
bounded by KV buffer rows on **dim 0**; `MambaPool` by its slot count on **dim
1** (`conv[:, indices]`, layout `[num_layers, num_slots, ...]`). A helper that
read `k_buffer[0].shape[0]` would be perfectly inert on Mamba — a guard that
cannot fire, which is the shape this sweep exists to remove.

## Prior evidence that this belongs at the pool boundary, not per-caller

Two forwarders already filter their indices — `swa_memory_pool.py:295`
(`swa_indices[swa_mask]`) and `unified_memory_pool.py:1227` (`swa_phys[valid]`)
— **the SWA half only, with the unfiltered `full_phys` sitting directly beside
them**. The need was seen once and not generalised. That is the same class one
more time, and it is the argument for putting the check where every caller must
pass through it rather than trusting each caller to remember.

## The sweep — 14 definitions in `memory_pool.py` alone

My first pass assumed one site.

| site | status |
|---|---|
| module `check_cpu_copy_rows` | the guard, once |
| `MHATokenToKVPool` :3262/:3284 | **WIRED** — the W40b site, and what the rig's `HybridLinearKVPool` delegates its KV half to (:4390, :4401) |
| `MambaPool` :1180/:1192 | **WIRED** — dim 1, bound `mamba_cache.conv[0].shape[1]`. Reached on this rig via `HybridLinearKVPool` → `mamba_pool.load_cpu_copy(_mamba_translate(...))`, and that translation is a second staleness source |
| `HybridLinearKVPool` :4389/:4399 | pure delegator — covered by the two above |
| `KVCache` ABC :2319/:2322 | `NotImplementedError` stubs, nothing to guard |
| `PageMajorMHATokenToKVPool` :4006/:4012 | enumerated, not wired |
| `MLATokenToKVPool` :4674/:4689 | enumerated, not wired |
| `DSATokenToKVPool` :5023/:5048 | enumerated, not wired |
| `dsa_cache_layer_split.py` | enumerated, not wired |
| `allocator/{base,token,paged,swa}.py`, `swa_memory_pool.py:285-330`, `unified_memory_pool.py:1223-1237` | forwarders that add no checks — the shape 25e7849844 named: "every pool-level `load_cpu_copy` merely forwards indices" |

The 12 unwired sites are not on this rig's path. They stay enumerated here with
their file:line rather than wired on faith.

## Emitter move

`restore_seam_state`'s log line sat AFTER `req.load_kv_cache`, so the one event
it exists to explain — a restore that does not return — is the one it
structurally could not witness. W40a logged 18 seam lines; W40b crashed inside
the call and logged **zero**. An ATTEMPT line goes before, rate-limited on the
same cadence as the success line.

## Measured

* **9 red / 8 green** against the tree today; **17 green** against the patched
  shadow. The A/B is the mutation proof: the reds are exactly the tests the
  guard exists for; the greens are controls that must not move, including the
  two that pin the silent half as a fact about today's code.
* Regression against the patched shadow: `test_seam_extent_contract_783.py` plus
  the whole `mem_cache` unit suite — **1768 passed, 1658 skipped, 361 subtests,
  2 failed**, and those 2 are the known `No CUDA GPUs are available` pair in
  `test_acceptance_emitters_758.py`, baselined pre-existing earlier tonight.
* Read-only: cwd `/tmp`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`.
  Verified no `__pycache__` in the tree and `git status` unchanged.

## STILL OPEN — do not close this by guessing

**Why did W40a's nine restores survive and W40b's first one not?** Unknown, and
no trigger has been written into anything. The guard lands regardless because it
is the **instrument that will answer it**: it converts an async IMA attributed to
an unrelated `synchronize()` into a deterministic host-side refusal naming the
id, the bound and the axis. `CUDA_LAUNCH_BLOCKING=1` belongs in the acceptance
arm so an IMA is attributed to its own kernel.
