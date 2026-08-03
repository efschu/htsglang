# NOTE 472 — graph-padded `positions` × DCP KV ownership

Verdict: **RED on the even-modulo lane, GREEN on our weighted lane.** Fixed.

Upstream cross-reference: sgl-project/sglang#33253, "Fix padded positions in
breakable CUDA Graph attention" (open at the time of writing). Their diff
narrows `forward_batch.positions` alongside `out_cache_loc` in
`radix_attention._unified_attention_with_output_impl` and restores it after the
backend call — four lines plus a test. Their evidence: Qwen3.5-397B-A17B FP8,
TP16/DCP4, breakable-graph prefill + full decode graphs, GSM8K-128 8-shot
0.875 → 0.984375.

## 1. Trace in our tree

Every hop verified by reading, on `origin/integration/r3-probe-next2`
(`031ee8becf`).

| # | Hop | Where |
| - | --- | ----- |
| 1 | Breakable/piecewise prefill replay takes `positions` and `out_cache_loc` from the captured static slots, both padded to the bucket | `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:958-959`, consumed at `:997` / `positions=` in the `ForwardBatch` replace |
| 2 | Both slots pad with **ZERO**, not with a sentinel | `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py:606-620` (prefill registry) and `:876-896` (breakable registry) — `PaddingPolicy.ZERO` on `positions` and `out_cache_loc` alike |
| 3 | `num_token_non_padded_cpu` stays the REAL count through the replace | `prefill_cuda_graph_runner.py:1031` |
| 4 | The piecewise attention op narrows Q/K/V and `out_cache_loc` — and, before this change, **not** `positions` | `python/sglang/srt/layers/radix_attention.py:223-227` (Q/K/V) and `:252-255` (`out_cache_loc`) as they stood pre-fix; second unnarrowed site in `unified_sparse_attention_with_output` at `:317-318` |
| 5a | **Weighted owner rule (ours, #173)** derives ownership AND the compact row from `out_cache_loc` only | `python/sglang/srt/layers/dcp/owner.py:362-392` `dcp_weighted_write_slots`; called at `triton_backend.py:2186` and `flashinfer_backend.py:2485` |
| 5b | **Even modulo rule (upstream)** derives ownership from `positions` | `triton_backend.py:2194-2203` and `flashinfer_backend.py:2491-2496`, pre-fix |
| 6 | Length guard `positions.numel() == loc.numel()` (from upstream #25090) fails, because step 4 narrowed `loc` but not `positions` | same two sites |
| 7 | Fallback `forward_batch.dcp_kv_mask` is populated **only on HIP** | `python/sglang/srt/model_executor/forward_batch_info.py:901-908` (`and is_hip()`) — so it is `None` on CUDA |
| 8 | `dcp_kv_mask=None` skips the masked kernel entirely and takes the plain unmasked scatter | `python/sglang/srt/mem_cache/memory_pool.py:2412` (`if dcp_kv_mask is not None:`) |
| 9 | #355's bound does **not** eat it: `masked_set_kv_buffer_kernel`'s `tl.device_assert((loc >= 0) & (loc < bound))` bounds the write to the BUFFER, not to owned rows — and it is only reached at all when a mask exists | `memory_pool.py:4095-4160`, bound from `kv_store_bound` at `:2420-2427` |

### What that actually does

Reproduced hermetically on the pre-fix code (`dcp_size=3`, `dcp_rank=1`, 6 real
tokens padded to 10, `out_cache_loc = positions = 100..105`):

```
mask                : None
owned rows          : [0, 3]
written rows        : [0, 1, 2, 3, 4, 5]
compact rows written: [33, 33, 34, 34, 34, 35]
FOREIGN rows written: [1, 2, 4, 5]
```

Every rank writes every token; compact rows 33/34 are written repeatedly and
the survivor is whichever `set_kv_buffer` row happened to be last. Silent
cross-rank KV corruption, no crash, no assert — the same failure signature
upstream measured as an 11-point GSM8K drop.

Note this is a *different* mechanism from the one upstream's prose describes
("virtual padded tokens compete with real tokens for the same physical KV
slots"). The padded rows themselves are harmless here: their ZERO
`out_cache_loc` is the reserved slot the pool over-allocates (`self.size +
self.page_size`), and in any case `out_cache_loc` was already narrowed. The
damage comes from the length DISAGREEMENT the unnarrowed `positions` creates,
which drops the owner mask for the REAL rows.

### Why the weighted lane is immune

`dcp_weighted_write_slots` reads `cache_loc` and nothing else. That is not an
accident of implementation: the docstring at `owner.py:369-383` states the
reason (the compact row must be an injective function of the slot id so it
stays collision-free across concurrent requests), and `out_cache_loc` is
exactly the tensor the piecewise wrapper already narrows. The lane cannot see
`positions` at all, padded or otherwise. The falsifier pins this positively —
it re-drives the weighted write with `positions` overwritten by `-7` and
asserts the emitted `loc`/mask are bit-identical.

## 2. Fix (our semantics)

Three parts, all in this branch:

1. **`radix_attention.narrow_pcg_token_views` / `restore_pcg_token_views`** —
   the narrow/restore pair, stated once, covering `out_cache_loc` **and**
   `positions`. Both piecewise attention ops
   (`unified_attention_with_output`, `unified_sparse_attention_with_output`)
   and `radix_linear_attention.unified_linear_attention_with_output` delegate
   to it. Upstream fixed one call site; we had three, one of which
   (`unified_sparse_attention_with_output`) upstream's diff does not touch at
   all.
2. **`layers/dcp/owner.dcp_even_write_mask`** — the even-lane mask decision,
   also stated once, shared by the Triton and flashinfer twins. It **raises**
   when neither `positions` nor the precomputed `forward_batch.dcp_kv_mask`
   agrees with the write's row count, instead of returning `None` and letting
   the write go out unmasked. This is the §12 "fail loudly instead of silently
   downgrading" line applied to the owner rule: for a token-sharded DCP write
   there is no such thing as a correct maskless write, so the absence of a
   mask is a bug, not a mode.
3. Both backends call the helper; the HIP precomputed-mask path is preserved
   as the second source.

Divergence from upstream worth recording: upstream's fix restores the length
agreement and stops there, so a future caller that breaks it again lands back
in the silent-unmasked-write state. Ours closes the same hole at the wrapper
AND makes the owner rule refuse the degenerate input, so the class cannot
recur silently through a third call site.

## 3. Falsifier

`test/registered/unit/distributed/test_dcp_pad_positions_472.py`, 14 tests,
hermetic (CPU, `CUDA_VISIBLE_DEVICES=99`, no collectives, no model). It drives
the production write paths directly —
`TritonAttnBackend._set_kv_buffer` and `FlashInferAttnBackend._dcp_write_scatter`
called unbound against a recording pool — rather than re-implementing the rule.

Inventory:

* `TestDcpPadPositionsEvenLane` — padded positions refused on both lanes; the
  post-narrowing batch produces exactly the owned rows; the non-graph
  (unpadded) path is unchanged; the HIP precomputed-mask fallback survives.
* `TestDcpEvenWriteMaskHelper` — the four helper outcomes.
* `TestDcpPadPositionsWeightedLane` — the immunity pin: arbitrary/adversarial
  `positions` do not move a single write, and across the group every real row
  is written exactly once (plan `[2,1,1]`).
* `TestPiecewiseWrapperNarrowsPositions` — narrow/restore round trip, `None`
  tolerance, and an AST ratchet asserting no function in `radix_attention` /
  `radix_linear_attention` other than the two helpers assigns
  `forward_batch.out_cache_loc` directly (i.e. no op may hand-roll a
  narrowing that forgets `positions` again).

### Can-fail proof (both arms executed, both reverted)

* Arm 1 — drop the `positions` line from `narrow_pcg_token_views`:
  `FAILED (errors=3)` — `test_even_lane_after_wrapper_narrowing` (both
  subtests) and `test_narrow_covers_positions_and_out_cache_loc`.
* Arm 2 — restore the pre-#472 `return precomputed` silent fallback in
  `dcp_even_write_mask`: `FAILED (errors=5)` — the three helper-refusal tests
  and both backend refusal tests. The corruption probe quoted in §1 was taken
  under this arm.

## 4. Reachability / DESK labels

* The bug needs the piecewise or breakable graph AND a DCP write on the even
  modulo lane. In this tree the even lane is reached whenever `dcp_size > 1`
  and no non-uniform token vector is installed (`uneven_dcp_active` false) —
  i.e. stock even DCP, and also `--rank-tp-ratio` + a uniform token vector
  (`uneven_dcp_kv_replicated` true, `uneven_dcp_active` false). There is no
  guard anywhere refusing DCP under the breakable graph, so the combination is
  live, not hypothetical.
* **BOOT-PENDING (1):** our own default is the WEIGHTED lane, which was already
  immune, so no boot of this rig can exhibit the original defect. The
  end-to-end value of the fix is on the even lane; proving it on hardware needs
  an even-token-vector DCP boot with `--enable-breakable-cuda-graph`. The unit
  falsifier is the substitute, and it drives the real backend methods.
* **BOOT-PENDING (2), ROCm only:** `deepseek_v4_backend_hip_radix.py:1241-1251`
  reads `forward_batch.positions` inside the backend forward with a round-robin
  `slice(cp_rank, None, cp_size)` and its comment asserts the tensor is "the
  full (padded) global layout". With the narrowing that tensor is now the
  real-token view. The backend is HIP-gated
  (`attention_registry.py:137-145`), unreachable on CUDA rigs, and upstream's
  #33253 makes the identical narrowing — but the interaction is unproven and
  belongs to a ROCm arm.
* `mrope_positions` is a separate padded token-axis slot and is **not**
  narrowed (neither here nor upstream). No DCP ownership reads it; recorded so
  the next reader does not assume it was covered.

## 5. Test results (verbatim)

```
$ cd /spinning/wt-472-pad-dcp/test/registered/unit/distributed
$ CUDA_VISIBLE_DEVICES=99 PYTHONPATH=/spinning/wt-472-pad-dcp/python \
    /spinning/htsglang-gpu/.venv/bin/python -m unittest test_dcp_pad_positions_472
Ran 14 tests in 1.370s
OK

$ ... -m unittest test_dcp_weighted_owner_rule
Ran 5 tests in 0.004s
OK

$ ... -m unittest test_dcp_weighted_index_math
Ran 8 tests in 0.014s
OK

$ ... -m unittest test_triton_weighted_dcp_wiring
Ran 14 tests in 0.025s
OK

$ cd ../model_executor && ... -m unittest test_cuda_graph_buffer_registry
Ran 33 tests in 0.043s
OK
```

Pre-existing and unrelated: `test_dcp_token_vector_collective` fails 11 tests
on this base with `AttributeError: 'types.SimpleNamespace' object has no
attribute 'uneven_kv_derived_mode'` — a stale server-args test double in a file
this branch does not touch.

`ruff check` and `ruff format --check` clean on all six touched files plus the
new test (flashinfer_backend.py carries 31 pre-existing findings, unchanged in
count by this branch); `codespell` clean.
