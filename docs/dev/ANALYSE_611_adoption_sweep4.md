# ANALYSE 611 — upstream adoption sweep 4 (2026-08-06)

Five upstream sgl-project/sglang artifacts flagged as fork-relevant, assessed
against our tree and adopted where they apply. Bugs first.

Base: `origin/integration/r3-probe-next2` @ `c671a21d39`.
Branch: `fix/adoption-sweep-611`.

Upstream paths do not exist here — the fork renamed and moved these modules — so
every item was located by symbol, never by path.

## Verdict table

| # | Upstream | Kind | Our verdict | Evidence |
|---|---|---|---|---|
| 1 | [#33713](https://github.com/sgl-project/sglang/issues/33713) unified_cache MAMBA nodes pruned instead of downgraded | issue, **no fix PR** | **NOT-EXPOSED** | `unified_radix_cache.py:1797-1815`, `:1456-1471`, `mamba_component.py:640-644` |
| 2 | [#33810](https://github.com/sgl-project/sglang/pull/33810) GDN `-1` sentinel in chunked extend kernel | PR, open | **EXPOSED → PORTED** | `chunk_delta_h.py:115` vs `fused_recurrent.py:908,997`; reachable via `gdn_triton.py:183` ← `hybrid_linear_attn_backend.py:96-99` |
| 3 | [#33795](https://github.com/sgl-project/sglang/pull/33795) DSpark graph capture JIT race | PR, open | **EXPOSED → PORTED** | `jit_cold_build.py:371-382` — sync/barrier lead each forward, none trails the last |
| 4 | [#33777](https://github.com/sgl-project/sglang/pull/33777) HiCache write_back duplicate reclaim | PR, open, **optimization not bug** | **NOT-EXPOSED** | no `full_host_duplicates` / `load_back_pending_id` in our tree; nothing to race |
| 5 | [#33656](https://github.com/sgl-project/sglang/issues/33656) DSv4 SWA position corruption | issue, **no fix PR** | **NOT-EXPOSED (guarded)** — GPU falsifier owed | `swa_component.py:734-752` rebuilds the mapping at load-back |

Confirmed via `gh api .../timeline`: items 1 and 5 have **no** cross-referenced
fix PR upstream. There is no upstream patch to port for either; they are
diagnosis-only here.

---

## 1. #33713 — MAMBA component pruned instead of downgraded — NOT-EXPOSED

**Upstream claim.** On device eviction the unified tree decides host-leaf
eligibility from the FULL component alone, so a hybrid-KDA node whose MAMBA host
copy is live gets detached from the radix tree wholesale, and the later request
never attempts H→D load-back.

**Our tree carries the same predicate.** `UnifiedTreeNode.backuped` is FULL-only:

- `python/sglang/srt/mem_cache/unified_radix_cache.py:175-177` — `backuped` is
  `component_data[FULL].host_value is not None`
- `:1663` — `_is_host_leaf` gates on exactly that
- `:1713` — `_evict_device_leaf` deletes the node outright when `not backuped`
  under write-through

The path is live for us: hybrid SSM + hierarchical cache routes to
`UnifiedRadixCache` (`registry.py:108-111`), with `ComponentType.MAMBA` in
`tree_components` (`registry.py:179-180`).

**Why it still cannot fire here.** The premise of the upstream bug — a node with
a live MAMBA host copy but no FULL host copy — is unreachable in our tree. Both
places that can set MAMBA `host_value` are structurally coupled to FULL's:

1. **Backup.** `write_backup` (`unified_radix_cache.py:1797-1815`) obtains
   `host_indices` once and commits FULL and every aux component in the same
   commit block. MAMBA's `host_value` write
   (`mamba_component.py:640-644`) only executes inside that block, so it cannot
   happen without FULL's.
2. **L3 storage prefetch.** `mamba_component.py:686` writes MAMBA `host_value`
   onto `insert_result.inserted_host_node`, and that field is only ever set to a
   node whose FULL `host_value` is already non-None — `unified_radix_cache.py:1456-1460`
   tests it explicitly before assigning, and `:1467` sets FULL's `host_value`
   before `:1471` assigns the new node.

So for every node that has a MAMBA host copy, `backuped` is True, `_is_host_leaf`
admits it, and `_evict_device_leaf` demotes rather than deletes.

**Not adopted**, and no patch exists upstream to adopt. Should our backup path
ever decouple FULL from aux commits, this predicate becomes wrong immediately —
that coupling is the invariant doing the work, and it is not asserted anywhere.
Worth a pin test if the backup path is refactored.

---

## 2. #33810 — GDN `-1` padding sentinel in the chunked extend kernel — EXPOSED, PORTED

**The defect.** `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` loads a slot id
and derives the state pointer without checking the padding sentinel:

`python/sglang/srt/layers/attention/fla/chunk_delta_h.py:115` (pre-fix)

```python
index = tl.load(initial_state_indices + i_n).to(tl.int32)
h0 = initial_state + index * stride_h
ht = initial_state + index * stride_h
```

With `index == -1` both pointers land one full `stride_h` **below** the state
pool base. The lane then reads its initial state from there
(`USE_INITIAL_STATE` block) and writes its final state back there
(`INPLACE_UPDATE` epilogue).

**The asymmetry is real and local.** The decode kernel already guards the same
sentinel — `fla/fused_recurrent.py:908` (`if idx >= 0:  # Assuming negative
indices are invalid`) and `:997`. Only the chunked extend path was unguarded.

**Reachability in our tree** (this is what makes it EXPOSED rather than
theoretical):

- `hybrid_linear_attn_backend.py:96-99` stamps `-1` over the tail of
  `mamba_cache_indices` whenever `forward_batch._original_batch_size` is smaller
  than the buffer — set by `prepare_mlp_sync_batch`
  (`forward_batch_info.py:1178`), i.e. the DP-attention padded-batch path.
- `gdn_backend.py:713-721` passes that same tensor as `cache_indices` into
  `kernel_dispatcher.extend` for every non-`target_verify` forward.
- `kernels/gdn_triton.py:183` forwards it verbatim as
  `initial_state_indices` into `chunk_gated_delta_rule`.

No mask sits anywhere on that route. (The `target_verify` arm is separately
masked — `gdn_backend.py:435-456` — which is precisely why the extend arm's gap
went unnoticed.)

**Ported**, adapted to our kernel (ours is `.to(tl.int32)` / `stride_h`; upstream
has since moved to `.to(tl.int64)` / `stride_init_state`, an unrelated envelope-pitch
change we do not carry). One `valid_state = index >= 0` predicate, conjoined into
the initial-state load and the in-place epilogue.

**Falsifier:** `test/registered/unit/layers/attention/test_gdn_chunk_h_pad_sentinel_611.py`,
5 tests. The state pool is carved out of a larger buffer whose slot 0 is a
poisoned guard, so slot `-1` lands inside the guard and both the OOB write and
the OOB read become observable on CPU. Verified in both directions: **5/5 green
with the fix, and reverting either `valid_state` conjunct turns exactly
`test_padded_lane_does_not_write_below_the_pool_base` and
`test_padded_lane_does_not_read_below_the_pool_base` red** while the control and
the interpreter self-check stay green.

Upstream's own regression signal is an 8-GPU `--dp=8 --enable-dp-attention`
model test, unavailable at the desk; this runs the real kernel through Triton's
interpreter instead. It executes in a child process — `@triton.jit` reads
`TRITON_INTERPRET` at decoration time, so an in-process env write both loses to
whatever suite imported the kernel first and poisons every suite collected
after. A partial `importlib.reload` was tried and rejected: the interpreted
kernel then calls still-compiled `@triton.jit` helpers from `fla.op` and dies
with *"Cannot call @triton.jit'd outside of the scope of a kernel"*.

### Not ported: the XPU sibling

`python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_delta_h.py:108-113`
carries the byte-identical unguarded pattern. Deliberately left alone: it is a
vendor backend we do not run, there is no XPU device here to falsify against,
and upstream's PR does not touch it either. Follow-up, not a silent omission.

---

## 3. #33795 — JIT still in flight when graph capture opens — EXPOSED, PORTED

**Upstream's fix** adds `synchronize()` + `tp_group.barrier()` between the last
warmup forward and `torch.cuda.CUDAGraph()` in `full_cuda_graph_backend.capture_one`.
Without it, async JIT compilation triggered by that forward (DeepGEMM/TVM; in
the DSpark compact ragged-verify arm every new non-uniform `verify_lens` shape
triggers a fresh compile) issues driver calls — `cuModuleLoadData` and friends —
from *inside* the stream-capture region, aborting capture with
`CUDA_ERROR_ILLEGAL_ADDRESS` / `cudaErrorStreamCaptureUnsupported`.

**Our tree has the same gap, one level down.** The fork hoisted the warmup loop
into a shared helper, `run_capture_warmups`
(`python/sglang/srt/utils/jit_cold_build.py:371-382`). Its sync and barrier
**lead** each iteration:

```python
for _ in range(repeats):
    device_module.synchronize()
    tp_group.barrier()
    out = forward_fn()
    if post_warmup_hook is not None:
        post_warmup_hook()
return out          # <- last forward's async work still in flight
```

`full_cuda_graph_backend.py:119-129` then constructs the CUDAGraph on the very
next statement. Same defect as upstream, exactly.

**Ported at the helper**, not per backend — which covers all three graph
backends at once (`full_cuda_graph_backend.py:119`,
`tc_piecewise_cuda_graph_backend.py:236`, `breakable_cuda_graph_backend.py:246`),
where upstream fixed only one. Note `breakable` is the PD-prefill path, which is
graph-covered by default here, so our exposure is wider than upstream's.

**One deliberate deviation from upstream:** the trailing join goes *inside*
`cold_build_window`. A rank reaching it has finished its own builds, but a peer
may still be in nvcc inside its own warmup loop; under the steady-state deadline
that wait reads as a wedge. Closing the window first would convert a slow peer's
cold build into a false wedge — the exact failure the window exists to prevent.
The recorded pass still runs outside the window, unchanged.

`skip_barrier` suppresses the trailing barrier as well as the per-iteration
ones; the trailing `synchronize()` always runs.

**Falsifier:** `test/registered/unit/distributed/test_jit_cold_build_window.py`,
25 tests (2 new, 2 existing order assertions updated). Verified in both
directions: **25/25 green with the fix; stripping the trailing join turns 4 red**,
including the new `test_no_forward_is_left_in_flight_when_the_warmups_return`
(no sync/barrier after the last forward) and
`test_the_post_warmup_join_stays_inside_the_cold_build_window` (join ran under
the relaxed deadline).

---

## 4. #33777 — HiCache write_back duplicate reclaim — NOT-EXPOSED

**This is not a bug.** The PR's stated motivation is *"to maximize the cache
capacity benefit of the write-back policy"* — a capacity optimization. It adds a
`full_host_duplicates` registry, a two-pass reclaim of host copies that are
redundant because the same Full KV is resident on both tiers, and a TP-consistency
digest over the victim ids.

The briefing's hypothesis was that upstream might have found a *different*
write-back race than the one we fixed in #60. It did not. The
`load_back_pending_id` field the PR introduces is not a fix for a pre-existing
race — it is new bookkeeping *required by* the new reclaim, guarding host slots
that an in-flight H→D load-back is reading from being reclaimed underneath it.
That hazard only exists once you start reclaiming host copies.

Our tree has none of these structures — no `full_host_duplicates`, no
`load_back_pending_id`, no reclaim digest — and `drive_host_eviction`
(`unified_radix_cache.py`) does not reclaim duplicated host copies at all. There
is nothing to race and nothing to fix.

**Not adopted.** Optional capacity follow-up if write-back host pressure ever
shows up as a measured problem; it would land as a feature with its own
TP-determinism argument, not as a bug fix, and it is explicitly out of scope for
a bugs-first sweep.

---

## 5. #33656 — DSv4 SWA wrong-position writes under HiCache — NOT-EXPOSED (guarded)

**Reported symptom.** Deterministic `TAIL_K_SWA` KV corruption on
DeepSeek-V4-Flash with `--enable-hierarchical-cache`: token id and chain hash
match expectations exactly, only the *position* is wrong (written at 512,
expected 8448) — a slot/translation bookkeeping fault, not data damage. All 8 TP
ranks report identical violation counts. The burst coincides with a prefill
carrying a large prefix-cache hit. Reporter suspects a sibling of #25889 (stale
full→SWA translation indices after a HiCache rebuild) in the newer
UnifiedRadixTree path.

Worth noting: the reporter diagnosed this with `--kv-canary`, which is **our**
instrument (`python/sglang/srt/kv_canary/`, `python/sglang/jit_kernel/kv_canary/`).
The report is therefore unusually well-shaped for us to act on — but it is still
an open issue with no upstream fix.

**Our exposure.** The suspected mechanism is a full→SWA mapping left stale after
an H→D load-back re-homes the SWA data into different device slots. Our
load-back rebuilds it explicitly, per node, at commit time:

`python/sglang/srt/mem_cache/unified_cache_components/swa_component.py:734-752`

```python
for n in xfer.nodes_to_load or []:
    ...
    swa_chunk = device_indices[offset : offset + n_tokens].clone()
    self._restore_device_value(n, swa_chunk)
    assert cd_full_n.value is not None and len(cd_full_n.value) == n_tokens
    # rebuild the mapping for the loaded SWA chunk
    allocator.set_full_to_swa_mapping(cd_full_n.value, swa_chunk)
```

The FULL device value is asserted present and length-matched before it is used
as the mapping's key space, so the "restored SWA slots still addressed by the
pre-eviction mapping" shape does not survive here.

**Verdict: NOT-EXPOSED, with an honest caveat.** The assert catches a *missing*
FULL device value, not a *stale* one — it would not fire if `cd_full_n.value`
were populated but no longer the value the mapping should key on (e.g. an
ordering change that let SWA commit before FULL's own load-back). That residual
is a real, if narrow, gap, and a desk read cannot close it.

**Follow-up owed (GPU, not desk):** a DSv4 run with `--kv-canary=log` +
`--enable-hierarchical-cache` at high prefix-reuse, watching
`per_forward_tail_k_swa` for `write_position` violations. This task is desk-only
and the production server is off-limits, so it is recorded here rather than
attempted. Nothing was changed for this item.

---

## Test results

Hermetic, CPU-only, no GPU touched:

```
CUDA_VISIBLE_DEVICES=99 PYTHONPATH=/spinning/wt-611-adoption/python \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest -q \
  test/registered/unit/distributed/test_jit_cold_build_window.py \
  test/registered/unit/model_executor/test_bcg_logits_output_buffer_462.py \
  test/registered/unit/layers/attention/test_gdn_chunk_h_pad_sentinel_611.py
=> 48 passed, 3 subtests passed
```

Falsifier runs (fix reverted, then restored):

| Fix reverted | Red | Green |
|---|---|---|
| `valid_state` conjuncts in `chunk_delta_h.py` | 2 (both OOB guards) | 3 (control + interpreter self-check + valid-lane) |
| trailing join in `jit_cold_build.py` | 4 | 21 |

Both suites also pass in the reverse collection order, which matters for the
Triton test specifically (subprocess isolation, see item 2).

`black`, `isort`, `codespell` clean on all touched files. `ruff check` clean on
all touched files except one **pre-existing** `E741` at
`test_jit_cold_build_window.py:553` in code this change did not author — left
alone deliberately.

## Confounds

- Items 1, 4, 5 are verdicts from code reading; only items 2 and 3 have
  executable evidence.
- Item 2's falsifier runs the real kernel under Triton's **interpreter**, not on
  a GPU. It proves the index arithmetic and the guard; it does not prove
  numerics on device.
- Item 3's falsifier is a call-order pin over mocks. It proves the ordering
  invariant, not that any real JIT race was observed here — the race itself is
  upstream's, reproduced on their DSpark arm.
- Item 5 is the weakest verdict: guarded by construction, not by a test, and
  the residual stale-vs-missing gap named above is not covered.
- Items 2 and 3 are ports of **open, unmerged** upstream PRs. If upstream
  revises either before merge, these adaptations should be re-checked.
