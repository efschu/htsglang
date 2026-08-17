# #441 — page-head transfer root, and why neither guard may be flipped yet

Desk only, 2026-08-17. No GPU launch, no boot, no serving contact.

## (a) The structural difference between `lf_ph` and its `lf_pf` sibling

**The copy helper requires 8-byte alignment and only one thing checks it.**

`transfer_item_warp` (`sgl-kernel/csrc/kvcacheio/transfer.cu:20-37`) moves
bytes with 64-bit PTX — `ld.global.nc.b64` / `st.global.cg.b64` (`:29-30`).
`.b64` accesses fault on a misaligned address. The only alignment guard in the
whole path is `TORCH_CHECK(item_size % 8 == 0)` in the launcher (`:285`).

**Why that guard is sufficient for `lf_pf` and not for `lf_ph`:**

| | `transfer_kv_all_layer_lf_pf` (`:481`) | `transfer_kv_all_layer_lf_ph` (`:516`) |
|---|---|---|
| offset pair | `get_global_offset_lf_tbl` → `get_global_offset_pf` | `get_global_offset_per_head_lf_tbl` → `get_global_offset_ph` |
| kernel | `transfer_kernel_impl` | `transfer_page_head_kernel_impl` (`:122`) |
| copy unit | `item_size_bytes` | `head_size_bytes = item_size / head_num` (`:145`) |
| offset terms | multiples of `item_size` and `layout_dim` only | **three terms divided by `head_num`** (`:106-119`) |

The argument wiring of the two is otherwise identical — I compared them
line by line (`:496-513` against `:532-551`): same `empty` source tensors, same
tables in the same slots, same `0` / `dst_layout_dim` layout dims. The
difference is entirely in the offset arithmetic.

`get_global_offset_ph` subdivides:

```
page_dim / head_num * head_id * page_size
page_id % page_size * page_dim / head_num
layer_id * item_size_bytes / head_num
```

Nothing anywhere requires `item_size / head_num` to be a multiple of 8. So a
shape can pass the only guard and still generate misaligned `.b64` accesses.

**ROOT, at file:line:** the missing invariant is
`item_size % (8 * head_num) == 0` (equivalently `head_size_bytes % 8 == 0`),
absent from the launcher guard at `transfer.cu:285`, and required by the PTX
at `transfer.cu:29-30` for every offset produced by
`get_global_offset_ph` at `transfer.cu:106-119`.

Proven hermetically, no CUDA:
`test/registered/unit/mem_cache/test_page_head_offset_alignment_441.py`
(6 pins + 6 subtests, 0.07 s). The smallest faulting shape is `head_num=2,
head_dim=2, fp16`: `item_size=8` passes the guard, `head_size=4` does not, and
24 of the enumerated offsets are misaligned. The pf sibling is pinned as safe
by construction over the same shapes.

### What this does NOT explain, stated plainly

**The reported crash shapes are aligned.** `test_minimax_sparse_pool_host_unit`
uses `page_size=4, float32, head_num=1, head_dim=2` → `item_size=8`,
`head_size=8`, zero misaligned offsets. So the defect above is real and latent
but is **not** the cause of that segfault. So are this rig's own serving shapes
(`head_num=4, head_dim=256, fp8` → `head_size=256`): no production shape here
is exposed.

Both facts are pinned in the test so the attribution cannot quietly widen.

Two hypotheses were checked and closed on the way:

- *Layer pointer table on the wrong device.* No: `pool_host/mha.py:135-144`
  builds it as a device `uint64` tensor, correctly.
- *`layout_dim` meant for page_first reused for page_head.* No: `layout_dim =
  token_stride_size * layer_num` (`mha.py:182`) is exactly the per-token
  all-layer stride the `ph` formula needs; the four terms are self-consistent
  with it.
- *K-only pool reaching `lf_ph` with a 12-byte item.* No: it raises for any
  layout other than `page_first` under the kernel backend
  (`mha.py:1113-1118`).

Attribution of the actual crash therefore needs metal. The falsifier is
written and filed, not run: `tools/441/falsify_lf_ph_441.py`, three arms
(`repro`, `alignment`, `bisect`), one arm per invocation because each is
expected to kill the process. Run under a gpu-arb claim.

## (b) Both guards: DO NOT FLIP. Neither condition is resolved.

The brief's premise was that the #436 rebuild landed and the guards are now
obsolete. Checking them at the code says otherwise, and for two different
reasons — neither of which is the ABI issue.

### `pytestmark` module skip, `sgl-kernel/tests/test_kvcacheio.py:20-24`

Its stated reason is *"test_kvcacheio segfaults on CUDA 13.x (sgl-kernel
bug)"*. But the `lf_ph` segfault this ticket is about reproduces on **both
wheels** — that is the ticket's own premise and the reason it is not the #436
ABI issue. A CUDA-13-only reason is therefore wrong, or at best incomplete.

Flipping it would expose a live crash on cu12 as well. It must stay until (a)
is fixed. What should change now is the REASON STRING, because a wrong reason
sends the next reader to the wrong wheel and costs them the same day it cost
this analysis.

### `_DIRECT_PF_BATCHCOPY_BROKEN_CUDA13`, `test_minimax_sparse_pool_host_unit.py:71`

This guard never guarded an ABI or wheel issue at all. Per the measured matrix
recorded above it (`:36-69`), it guards a **test-shape violation** of the
`cudaMemcpyBatchAsync` contract:

```
pageable + default stream -> failIdx=SIZE_MAX invalid argument
pinned   + side stream    -> OK          <- the production shape
```

Production satisfies both requirements (`pool_host/mha.py:97` pins;
`managers/cache_controller.py:276,:742-749` uses a side stream); the test
allocates pageable host memory and copies on the default stream.

The file's own comment already states the conclusion: *"Closing #441(b)'s flip
half needs this test made production-shaped (pin the host pool and copy on a
side stream), not merely unskipped — unskipping alone leaves it red."* That
remains true on the current line. The rebuild landing does not change a
contract the test violates by construction.

**So the correct next action is not a flip but a test-shape fix** — pin the
host pool and move the copy to a side stream — and verifying that fix is GPU
work, because the failure mode is a CUDA runtime error.

### Pin tests

Not written, deliberately. A pin that "would go red if the underlying breakage
returned" presupposes the breakage is currently GONE; for both guards it is
not. Writing pins now would encode a green state that does not exist. They
belong with the fixes, and each fix is named above.

## Not delivered

The #261-Gate short-run without the harness shim is **not** written. Preparing
it turnkey needs the gate's shim details verified at the code, which I have not
done, and shipping a script I cannot stand behind would be worse than saying
so. It is the one item of this brief left open.
