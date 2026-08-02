# #436 — standalone repro for the HiCache host-tier KV-transfer segfault

`kv_transfer_repro.py` is the falsifier for #436: it fails while the bug is
present and passes once the `sgl_kernel` wheel is built against the same CUDA
major as torch.

## The bug

`MHATokenToKVPoolHost(layout="page_first_direct", io_backend="direct")` calls
`sgl_kernel.transfer_kv_all_layer_direct_lf_pf`, which reaches
`cudaMemcpyBatchAsync` in `sgl-kernel/csrc/kvcacheio/transfer.cu`. Hybrid-GDN
has no way around that path: `MambaPoolHost` only accepts `page_first_direct`.

CUDA 13 dropped the trailing `size_t* failIdx` parameter from
`cudaMemcpyBatchAsync`. `transfer.cu` handles both shapes at runtime (upstream
`Fix segfault in cudaMemcpyBatchAsync on CUDA 13.0`): it selects the signature
from `cudaRuntimeGetVersion()` and calls the pointer that
`dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync")` hands back.

Those two lookups can disagree, and on a cu12-built wheel inside a cu13 torch
process they always do:

| lookup | resolves against | answer |
|---|---|---|
| `cudaRuntimeGetVersion` — a linked, *version-tagged* import (`objdump -T` shows `(libcudart.so.12)`) | the cudart the extension was built against | `12090` → "use the 9-argument form" |
| `dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync")` — unversioned, whole-process, load order | torch's `libcudart.so.13` | the 8-argument cu13 function |

So the 9-argument call convention is applied to the 8-argument function: the
stack address meant for `failIdx` is read as the stream. Observed as
`cudaErrorInvalidValue` in one place (see the skip marker on
`test_minimax_sparse_pool_host_unit.py::test_device_to_host_direct_page_first_direct`)
and as a SIGSEGV in the server.

The fix is not a source change — the shim is already correct for a consistent
toolchain. It is to build the wheel against CUDA 13, so both halves resolve to
`libcudart.so.13`.

## Running it

```bash
V=/spinning/htsglang-gpu/.venv
export LD_LIBRARY_PATH="$V/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
$V/bin/python scripts/dev/436_kv_transfer_repro/kv_transfer_repro.py            # both stages
$V/bin/python scripts/dev/436_kv_transfer_repro/kv_transfer_repro.py --mode abi # desk only
```

`--mode abi` needs no GPU and creates no CUDA context — run it under
`CUDA_VISIBLE_DEVICES=99` while the cards are busy. It prints `ABI_SPLIT` or
`ABI_CONSISTENT`.

`--mode call` needs exactly one card for a few seconds. The default `drive`
mode runs the probe in-process and then re-executes the call in a child, so a
segfault is reported as `SIGSEGV` instead of taking the caller down.

Exit codes: `0` pass, `1` fail, `2` skipped (no GPU / no `sgl_kernel`).

## Can-fail proof

Same script, same card (RTX 3080), 2026-08-02, only the installed wheel
swapped:

| wheel | build | ABI probe | the call |
|---|---|---|---|
| `e7b16e1d…` (`/spinning/wt-327a-wheel/`) | CUDA 12.9 | `ABI_SPLIT` | `FAIL -- child died from SIGSEGV inside the KV transfer (#436 reproduced)` |
| `cc98be5d…` (`/spinning/wt-436-wheel/`) | CUDA 13.0.88 | `ABI_CONSISTENT` | `PASS (no fault, bytes match the per-page reference)` |

## The stream, and why `--stream default` exists

CUDA's contract for `cudaMemcpyBatchAsync` is that `hStream` **must not be the
legacy NULL stream**. Torch's default stream is exactly that, so issuing the
transfer on it is refused with `cudaErrorInvalidValue` on either wheel. That is
the API's rule, not #436.

The default (`--stream dedicated`) matches production, which runs the transfer
inside `with device_module.stream(self.write_stream)`
(`cache_controller.py:742`). `--stream default` reproduces the refusal on
purpose, so the two failure modes are never confused: on the old wheel the call
dies before any argument is validated, on a correctly built one it comes back
with a clean error.
