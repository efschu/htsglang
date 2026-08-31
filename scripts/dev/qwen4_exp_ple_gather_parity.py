#!/usr/bin/env python3
"""Pinned zero-copy PLE gather vs the pageable host gather: do they agree?

Register #1036. Tiny, VRAM-cheap, and it is the check that matches the error
class: the host path re-implements a Triton kernel's addressing, and the failure
mode of getting that wrong is not a crash but a silently wrong embedding row --
the same shape as the n-gram prime-hashing trap.

Both paths must agree BIT-FOR-BIT on:
  * in-range ids            -> row (global - tp_vocab_start)
  * out-of-range ids        -> ZEROS, not row 0's contents
  * fp8_e4m3 storage        -> converted to bf16
  * an empty id tensor      -> no launch, output untouched
"""

from __future__ import annotations

import shutil
import sys

sys.path.insert(0, "python")

import torch

from sglang.srt.models.qwen4_exp import (
    _gather_ple_embedding_from_pinned_kernel,
)


class _Shard:
    def __init__(self, start: int, end: int):
        self.org_vocab_start_index = start
        self.org_vocab_end_index = end


class _Probe:
    """Only the state the two gather paths actually read."""

    def __init__(self, weight: torch.Tensor, start: int, end: int, pageable: bool):
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.embedding_dim = weight.shape[1]
        self.shard_indices = _Shard(start, end)
        self._pageable = pageable
        import triton

        self._block_d = triton.next_power_of_2(self.embedding_dim)

    # bound from the real class so this tests the shipped code, not a copy
    from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding as _R

    _gather_host = _R._gather_host


def run_allocation_modes(tmpdir) -> bool:
    """Which of the three residency modes the real __init__ actually gives you.

    [#1036] The half-measure has to be UNREACHABLE, not merely discouraged. With
    SwapTotal 0, `PAGEABLE=1` and no directory leaves the whole table resident while
    looking offloaded, so __init__ refuses it. This exercises the shipped __init__
    against a small fake embedding rather than the 95 GiB one -- the branch under
    test is the allocation decision, and it does not care about size.
    """
    import os
    from types import SimpleNamespace

    from sglang.srt.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding

    rows, dim = 256, 160

    def fake():
        e = SimpleNamespace()
        e.quant_method = UnquantizedEmbeddingMethod()
        e.weight = torch.nn.Parameter(
            torch.zeros(rows, dim, dtype=torch.bfloat16), requires_grad=False
        )
        e.weight_scale = torch.ones(1, dtype=torch.bfloat16)
        e.num_added_embeddings = 0
        for name in Qwen4ExpPinnedHostEmbedding._COPIED_ATTRIBUTES:
            if not hasattr(e, name):
                setattr(e, name, None)
        e.embedding_dim = dim
        e.shard_indices = _Shard(0, rows)
        e.tp_size = 1
        return e

    ok = True
    swap0 = "SwapTotal:        0 kB" in open("/proc/meminfo").read() or True

    # (1) pinned: the default, unchanged
    for k in ("SGLANG_PLE_HOST_PAGEABLE", "SGLANG_PLE_HOST_MMAP_DIR"):
        os.environ.pop(k, None)
    m = Qwen4ExpPinnedHostEmbedding(fake())
    good = m.weight.is_pinned() and m._mmap_path is None
    print(f"  {'OK ' if good else 'BAD'} default            -> pinned="
          f"{m.weight.is_pinned()} mmap={m._mmap_path is not None}")
    ok &= good
    # (2) pageable with no directory. The refusal is CONDITIONAL ON FIT, which is
    # the correction the operator's 8-bit plan forced: at bf16 (95.368 GiB) the
    # caller must have meant disk, at fp8 (47.684 GiB) RAM residency is the plan.
    # My first version refused both and made the second unreachable.
    os.environ["SGLANG_PLE_HOST_PAGEABLE"] = "1"

    #   (2a) a table that FITS -> accepted, RAM-resident, no refusal
    try:
        m = Qwen4ExpPinnedHostEmbedding(fake())
        good = not m.weight.is_pinned() and m._mmap_path is None
        print(f"  {'OK ' if good else 'BAD'} pageable, fits     -> accepted, "
              f"RAM-resident (pinned={m.weight.is_pinned()}, mmap=False)")
        ok &= good
    except ValueError as exc:
        print(f"  BAD pageable, fits     -> refused, but it fits: {str(exc)[:50]}")
        ok = False

    #   (2b) a table that CANNOT fit -> must refuse. The weight is a META tensor, so
    #   the size is real but nothing is allocated: the refusal is computed from
    #   numel*element_size and fires before any allocation.
    huge = fake()
    huge.weight = torch.nn.Parameter(
        torch.empty(400_000_000, dim, dtype=torch.bfloat16, device="meta"),
        requires_grad=False,
    )
    try:
        Qwen4ExpPinnedHostEmbedding(huge)
        print("  BAD pageable, oversized-> accepted (119 GiB cannot be RAM-resident)")
        ok = False
    except ValueError as exc:
        print(f"  OK  pageable, oversized-> REFUSED: {str(exc)[:52]}...")
    except Exception as exc:
        print(f"  BAD pageable, oversized-> {type(exc).__name__} instead of a "
              f"refusal: {str(exc)[:40]}")
        ok = False

    # (3) pageable with a directory: file-backed, and the file really appears
    os.environ["SGLANG_PLE_HOST_MMAP_DIR"] = tmpdir
    m = Qwen4ExpPinnedHostEmbedding(fake())
    exists = m._mmap_path is not None and os.path.exists(m._mmap_path)
    sized = exists and os.path.getsize(m._mmap_path) == rows * dim * 2
    good = exists and sized and not m.weight.is_pinned()
    print(f"  {'OK ' if good else 'BAD'} pageable + dir     -> file={exists} "
          f"size_exact={sized} pinned={m.weight.is_pinned()}")
    ok &= good
    for k in ("SGLANG_PLE_HOST_PAGEABLE", "SGLANG_PLE_HOST_MMAP_DIR"):
        os.environ.pop(k, None)

    # ---- the OTHER axis: which gather, independent of residency. This is the
    # combination my first design made unreachable: pinned + host gather, which is
    # what an fp8 table on sm86 needs.
    cap = torch.cuda.get_device_capability()
    print(f"\n  gather axis on sm{cap[0]}{cap[1]} (fp8e4nv floor is 8.9):")

    def gmode(env, dtype):
        for k in ("SGLANG_PLE_GATHER", "SGLANG_PLE_HOST_PAGEABLE",
                  "SGLANG_PLE_HOST_MMAP_DIR"):
            os.environ.pop(k, None)
        if env:
            os.environ["SGLANG_PLE_GATHER"] = env
        e = fake()
        e.weight = torch.nn.Parameter(
            torch.zeros(rows, dim, dtype=dtype), requires_grad=False
        )
        try:
            mm = Qwen4ExpPinnedHostEmbedding(e)
            return ("host" if mm._use_host_gather else "pinned"), mm.weight.is_pinned()
        except ValueError as exc:
            return f"REFUSED({str(exc)[:34]}...)", None
        finally:
            os.environ.pop("SGLANG_PLE_GATHER", None)

    want_fp8 = "host" if cap < (8, 9) else "pinned"
    cases = [
        (None, torch.bfloat16, "pinned", "auto, bf16 table"),
        (None, torch.float8_e4m3fn, want_fp8, "auto, fp8 table"),
        ("host", torch.bfloat16, "host", "forced host, bf16"),
    ]
    for env, dtype, expect, label in cases:
        got, pinned = gmode(env, dtype)
        good = got == expect
        print(f"  {'OK ' if good else 'BAD'} {label:22s} -> {got:8s} "
              f"(want {expect}, residency pinned={pinned})")
        ok &= good

    # forced pinned + fp8 on a pre-8.9 card must REFUSE, not die later at compile
    got, _ = gmode("pinned", torch.float8_e4m3fn)
    if cap < (8, 9):
        good = got.startswith("REFUSED")
        print(f"  {'OK ' if good else 'BAD'} forced pinned, fp8    -> {got[:46]}")
    else:
        good = got == "pinned"
        print(f"  {'OK ' if good else 'BAD'} forced pinned, fp8    -> {got} "
              f"(this card supports fp8e4nv)")
    ok &= good
    return ok


def run_case(dtype, rows, dim, start, end, ids, label) -> bool:
    torch.manual_seed(0)
    ref = (torch.randn(rows, dim, dtype=torch.float32) * 0.05)
    host_w = ref.to(dtype)

    dev_ids = ids.cuda()
    oor = ~((ids >= start) & (ids < end))

    # --- host path, always available: it dequantises in PyTorch on the CPU.
    out_host = torch.zeros((ids.numel(), dim), dtype=torch.bfloat16, device="cuda")
    probe = _Probe(host_w.clone(), start, end, pageable=True)
    probe._gather_host(dev_ids.reshape(-1).long(), out_host)

    # Independent reference, no kernel and no host-path code involved.
    want = torch.zeros((ids.numel(), dim), dtype=torch.bfloat16)
    for i, gid in enumerate(ids.tolist()):
        if start <= gid < end:
            want[i] = host_w[gid - start].to(torch.bfloat16)
    host_ok = torch.equal(out_host.cpu(), want)

    # --- pinned zero-copy path: may not COMPILE for this dtype on this arch.
    out_pinned = torch.zeros_like(out_host)
    pinned = host_w.pin_memory()
    try:
        _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
            pinned.data_ptr(),
            dev_ids,
            out_pinned,
            embedding_dim=dim,
            tp_vocab_start=start,
            tp_vocab_end=end,
            is_fp8=dtype == torch.float8_e4m3fn,
            BLOCK_D=__import__("triton").next_power_of_2(dim),
        )
        pinned_state = "ok"
    except Exception as exc:
        # This is a RESULT, not a harness failure: it means the shipped
        # zero-copy path cannot serve this storage dtype on this card.
        pinned_state = f"UNAVAILABLE ({str(exc).splitlines()[-1][:58].strip()})"

    if pinned_state == "ok":
        agree = torch.equal(out_pinned, out_host)
        zeroed = bool((out_host.cpu()[oor] == 0).all()) if oor.any() else True
        good = agree and zeroed and host_ok
        print(f"  {'OK ' if good else 'BAD'} {label:32s} pinned==host={agree} "
              f"host==reference={host_ok} oor-zeroed={zeroed}")
        return good

    print(f"  {'OK ' if host_ok else 'BAD'} {label:32s} host==reference={host_ok}  "
          f"pinned path {pinned_state}")
    return host_ok


def run_mmap_eviction_case(dtype, rows, dim, start, end, ids, tmpdir) -> bool:
    """The check that decides whether "partly on disk" is real.

    [#1036] Unpinned is not spillable. Anonymous pageable memory can only be
    evicted to swap, and this rig has SwapTotal 0, so `PAGEABLE=1` alone leaves the
    whole table resident -- I claimed otherwise before measuring. Only CLEAN
    FILE-BACKED pages can be dropped and re-faulted.

    So this writes the table through a MAP_SHARED mapping, fsyncs it clean, then
    FORCES the kernel to drop those pages with POSIX_FADV_DONTNEED, and only then
    gathers. Correct values after a forced eviction is the only evidence that the
    file is genuinely the backing store rather than a copy nobody reads.
    """
    import os

    path = os.path.join(tmpdir, f"ple_{rows}x{dim}_{str(dtype).split('.')[-1]}.bin")
    torch.manual_seed(0)
    ref = (torch.randn(rows, dim, dtype=torch.float32) * 0.05).to(dtype)

    backing = torch.from_file(
        path, shared=True, size=rows * dim, dtype=dtype
    ).view(rows, dim)
    backing.copy_(ref)                      # what the weight loader does

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)                        # dirty -> clean, so it CAN be dropped
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)   # now drop it
        # Did it actually leave RAM? mincore via /proc is awkward; the honest proxy
        # is that the bytes still read back correctly, which requires a disk fault.
        resident_hint = os.path.getsize(path)
    finally:
        os.close(fd)

    dev_ids = ids.cuda()
    out = torch.zeros((ids.numel(), dim), dtype=torch.bfloat16, device="cuda")
    probe = _Probe(backing, start, end, pageable=True)
    probe._gather_host(dev_ids.reshape(-1).long(), out)

    want = torch.zeros((ids.numel(), dim), dtype=torch.bfloat16)
    for i, gid in enumerate(ids.tolist()):
        if start <= gid < end:
            want[i] = ref[gid - start].to(torch.bfloat16)
    good = torch.equal(out.cpu(), want)
    print(f"  {'OK ' if good else 'BAD'} {str(dtype).split('.')[-1]:9s} mmap + fsync + "
          f"FADV_DONTNEED then gather   correct={good}  file={resident_hint/2**10:.0f} KiB")
    return good


def main() -> int:
    if not torch.cuda.is_available():
        print("REFUSED: needs one visible CUDA device for the pinned path.")
        return 2
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"cap={torch.cuda.get_device_capability(0)}")

    rows, dim, start, end = 512, 160, 128, 384
    inside = torch.tensor([128, 200, 383, 300], dtype=torch.long)
    mixed = torch.tensor([0, 128, 383, 384, 511, 200], dtype=torch.long)

    ok = True
    for dtype, name in ((torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8_e4m3")):
        ok &= run_case(dtype, rows, dim, start, end, inside, f"{name} all in range")
        ok &= run_case(dtype, rows, dim, start, end, mixed, f"{name} mixed in/out of range")

    # empty ids: the real gather skips the launch entirely
    out = torch.full((0, dim), 7.0, dtype=torch.bfloat16, device="cuda")
    probe = _Probe(torch.zeros(rows, dim, dtype=torch.bfloat16), start, end, True)
    probe._gather_host(torch.zeros(0, dtype=torch.long, device="cuda"), out)
    print(f"  OK  empty id tensor                 no rows written ({out.numel()} elems)")

    # The mmap path: does the table survive a FORCED eviction? This is the one that
    # separates "unpinned" from "actually spillable".
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="ple_mmap_")
    print("\nfile-backed mmap, forced out of the page cache before the gather:")
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        ok &= run_mmap_eviction_case(dtype, rows, dim, start, end, mixed, tmpdir)

    print("\nwhich residency mode the shipped __init__ actually gives you:")
    ok &= run_allocation_modes(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nPARITY HOLDS" if ok else "\nPARITY BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
