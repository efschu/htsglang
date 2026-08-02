#!/usr/bin/env python3
"""Standalone falsifier for #436: the HiCache host-tier KV transfer segfault.

The failing production path is

    MHATokenToKVPoolHost(layout="page_first_direct", io_backend="direct")
        -> sgl_kernel.transfer_kv_all_layer_direct_lf_pf
            -> transfer_kv_page_first_direct_impl<true>   (transfer.cu)
                -> cudaMemcpyBatchAsync

Hybrid-GDN is forced onto that path because MambaPoolHost only accepts
``page_first_direct``, so this call is not optional for that model family.

Why the call can die
--------------------
``cudaMemcpyBatchAsync`` changed signature between CUDA 12 and CUDA 13: the
12.x form takes a trailing ``size_t* failIdx`` before the stream, the 13.x form
dropped it.  ``transfer.cu`` handles that with a runtime shim (upstream
``Fix segfault in cudaMemcpyBatchAsync on CUDA 13.0``): it picks the signature
from ``cudaRuntimeGetVersion()`` and calls the pointer returned by
``dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync")``.

That shim is only correct while both halves see the *same* libcudart:

* ``cudaRuntimeGetVersion`` is a normal linked symbol -- it answers for the
  cudart the extension was built and linked against;
* ``dlsym(RTLD_DEFAULT, ...)`` searches the whole process in load order, so it
  answers for whichever libcudart was loaded first, which in a torch process is
  torch's own.

A cu12-built ``sgl_kernel`` inside a torch ``+cu130`` process can therefore read
"runtime 12.x, use the 9-argument form" and then jump into the 8-argument cu13
function.  The 9th argument (the stream) lands nowhere and the 8th (a stack
address that was meant to receive ``failIdx``) is read as the stream:
``cudaErrorInvalidValue`` at best, a segfault at worst.  Both have been observed
on this rig.

Modes
-----
``--mode abi``   Desk diagnostic, no GPU and no CUDA context needed.  Reports
                 the cudarts loaded in the process, which library owns the
                 ``cudaMemcpyBatchAsync`` that ``dlsym`` would return, what
                 ``cudaRuntimeGetVersion`` answers per loaded cudart, and the
                 ``DT_NEEDED`` cudart of the installed ``sgl_kernel`` object.
                 Prints ``ABI_CONSISTENT`` or ``ABI_SPLIT``.

``--mode call``  The actual call.  Needs one GPU.  Builds the smallest tensor
                 set that reaches ``cudaMemcpyBatchAsync`` (2 layers, 2 pages,
                 non-MLA, so 8 batched copies) and checks the copied bytes
                 against a reference.

``--mode drive`` Default.  Runs ``abi`` in-process, then re-executes ``call`` in
                 a child so that a segfault is *reported* (``signal 11``)
                 instead of taking the harness down with it.

Exit codes: 0 pass, 1 fail (wrong bytes, CUDA error, or child killed by a
signal), 2 skipped (no GPU / no sgl_kernel).
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import os
import subprocess
import sys

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 2

# Small on purpose: the point is to reach cudaMemcpyBatchAsync, not to move
# bytes.  num_copies = num_pages * num_layers * 2 = 8.
NUM_LAYERS = 2
PAGE_SIZE = 4
ITEM_SIZE = 8
TOTAL_PAGES = 4
TOTAL_ITEMS = TOTAL_PAGES * PAGE_SIZE
PAGES_TO_MOVE = 2


# --------------------------------------------------------------------------
# mode: abi
# --------------------------------------------------------------------------


def _loaded_maps() -> list[str]:
    try:
        with open("/proc/self/maps", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    seen: list[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        if path.startswith("/") and path not in seen:
            seen.append(path)
    return seen


def _loaded_cudarts() -> list[str]:
    return [p for p in _loaded_maps() if "libcudart.so" in os.path.basename(p)]


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _dladdr(addr: int) -> str | None:
    """Which shared object provides the code at ``addr``."""
    libdl = ctypes.CDLL(None)
    info = _DlInfo()
    rc = libdl.dladdr(ctypes.c_void_p(addr), ctypes.byref(info))
    if rc == 0 or not info.dli_fname:
        return None
    return info.dli_fname.decode()


def _dt_needed(path: str) -> list[str]:
    try:
        out = subprocess.run(
            ["objdump", "-p", path],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    needed = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("NEEDED"):
            needed.append(line.split(None, 1)[1].strip())
    return needed


def _sgl_kernel_objects() -> list[str]:
    try:
        import sgl_kernel  # noqa: PLC0415
    except Exception:  # pragma: no cover - reported by the caller
        return []
    root = os.path.dirname(sgl_kernel.__file__)
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".so"):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def mode_abi(verbose: bool = True) -> int:
    """Report the cu12/cu13 split without creating a CUDA context.

    Nothing here initialises a device: ``cudaRuntimeGetVersion`` is documented
    to work with no device present, and it is the only CUDA call made.
    """
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    emit("== #436 cudaMemcpyBatchAsync ABI probe ==")

    try:
        import torch  # noqa: PLC0415

        emit(f"torch                     : {torch.__version__}")
        emit(f"torch.version.cuda        : {torch.version.cuda}")
    except Exception as exc:  # pragma: no cover
        emit(f"torch                     : NOT IMPORTABLE ({exc})")

    have_kernel = True
    try:
        import sgl_kernel  # noqa: PLC0415

        emit(
            f"sgl_kernel                : {sgl_kernel.__version__} @ {os.path.dirname(sgl_kernel.__file__)}"
        )
        emit(
            f"transfer op registered    : {hasattr(torch.ops.sgl_kernel, 'transfer_kv_all_layer_direct_lf_pf')}"
        )
    except Exception as exc:
        have_kernel = False
        emit(f"sgl_kernel                : NOT IMPORTABLE ({exc})")

    emit()
    emit("-- DT_NEEDED cudart of the installed sgl_kernel objects --")
    kernel_cudarts: set[str] = set()
    for obj in _sgl_kernel_objects():
        needed = [n for n in _dt_needed(obj) if n.startswith("libcudart.so")]
        kernel_cudarts.update(needed)
        emit(f"  {os.path.basename(obj):<45} {','.join(needed) or '(none)'}")

    emit()
    emit("-- libcudart mapped into this process --")
    loaded = _loaded_cudarts()
    for path in loaded:
        emit(f"  {path}")
    if not loaded:
        emit("  (none yet -- import torch/sgl_kernel first)")

    emit()
    emit("-- who owns the dlsym'd cudaMemcpyBatchAsync --")
    libdl = ctypes.CDLL(None)
    libdl.dlsym.restype = ctypes.c_void_p
    libdl.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    RTLD_DEFAULT = ctypes.c_void_p(0)
    sym = libdl.dlsym(RTLD_DEFAULT, b"cudaMemcpyBatchAsync")
    dlsym_owner = None
    if not sym:
        emit(
            "  cudaMemcpyBatchAsync     : NOT FOUND (kernel falls back to per-page copy)"
        )
    else:
        dlsym_owner = _dladdr(sym)
        emit(f"  cudaMemcpyBatchAsync     : {hex(sym)} in {dlsym_owner}")

    emit()
    emit("-- cudaRuntimeGetVersion, per loaded cudart --")
    runtime_versions: dict[str, int] = {}
    for path in loaded:
        try:
            lib = ctypes.CDLL(path)
            ver = ctypes.c_int(0)
            rc = lib.cudaRuntimeGetVersion(ctypes.byref(ver))
            runtime_versions[path] = ver.value if rc == 0 else -1
            emit(f"  {os.path.basename(path):<24} rc={rc} version={ver.value}")
        except Exception as exc:  # pragma: no cover
            emit(f"  {os.path.basename(path):<24} query failed: {exc}")

    emit()
    # The verdict.  A split is: sgl_kernel links one cudart major, and the
    # dlsym'd batch-copy symbol comes from a different one.
    verdict = "UNDETERMINED"
    if dlsym_owner and kernel_cudarts:
        owner_base = os.path.basename(dlsym_owner)
        owner_major = (
            owner_base.split("libcudart.so.")[-1].split(".")[0]
            if "libcudart.so." in owner_base
            else "?"
        )
        kernel_majors = {
            n.split("libcudart.so.")[-1].split(".")[0] for n in kernel_cudarts
        }
        emit(f"sgl_kernel links cudart major(s): {sorted(kernel_majors)}")
        emit(f"dlsym cudaMemcpyBatchAsync from : cudart major {owner_major}")
        if owner_major != "?" and kernel_majors == {owner_major}:
            verdict = "ABI_CONSISTENT"
        else:
            verdict = "ABI_SPLIT"
    emit(f"VERDICT: {verdict}")
    emit(
        "  ABI_SPLIT means transfer.cu picks its signature from one cudart and "
        "calls a function from another -- the #436 hypothesis."
    )

    if verbose:
        print("\n".join(lines))
    if not have_kernel:
        return EXIT_SKIP
    return EXIT_PASS


# --------------------------------------------------------------------------
# mode: call
# --------------------------------------------------------------------------


def mode_call(use_default_stream: bool = False) -> int:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:
        print(f"SKIP: torch not importable: {exc}")
        return EXIT_SKIP

    try:
        from sgl_kernel.kvcacheio import transfer_kv_all_layer_direct_lf_pf  # noqa: PLC0415
    except Exception as exc:
        print(f"SKIP: sgl_kernel.kvcacheio not importable: {exc}")
        return EXIT_SKIP

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device visible (CUDA_VISIBLE_DEVICES?)")
        return EXIT_SKIP

    device = torch.device("cuda:0")
    print(
        f"device: {torch.cuda.get_device_name(device)} cc={torch.cuda.get_device_capability(device)}"
    )
    print(f"torch {torch.__version__}  runtime cuda {torch.version.cuda}")

    dtype = torch.bfloat16

    # Device side, layer-first: one [total_items, item_size] buffer per layer,
    # K layers then V layers -- exactly how MHATokenToKVPoolHost passes
    # ``device_pool.k_buffer + device_pool.v_buffer``.
    #
    # Sample on the CPU and move: on-GPU RNG is not bit-identical across
    # architectures, and this comparison must not depend on which card runs it.
    gen = torch.Generator().manual_seed(436)
    src_k = [
        torch.randn(TOTAL_ITEMS, ITEM_SIZE, generator=gen, dtype=torch.float32).to(
            device=device, dtype=dtype
        )
        for _ in range(NUM_LAYERS)
    ]
    src_v = [
        torch.randn(TOTAL_ITEMS, ITEM_SIZE, generator=gen, dtype=torch.float32).to(
            device=device, dtype=dtype
        )
        for _ in range(NUM_LAYERS)
    ]

    # Host side, page-first: [total_pages, num_layers, page_size, item_size],
    # pinned, one buffer for K and one for V.
    dst_k = torch.zeros(
        TOTAL_PAGES, NUM_LAYERS, PAGE_SIZE, ITEM_SIZE, dtype=dtype
    ).pin_memory()
    dst_v = torch.zeros_like(dst_k)

    # Two whole pages, moved to different page slots so a stride bug cannot be
    # mistaken for success.
    src_pages = [0, 2]
    dst_pages = [3, 1]
    src_indices = torch.tensor(
        [p * PAGE_SIZE + t for p in src_pages for t in range(PAGE_SIZE)],
        dtype=torch.int64,
    )
    dst_indices = torch.tensor(
        [p * PAGE_SIZE + t for p in dst_pages for t in range(PAGE_SIZE)],
        dtype=torch.int64,
    )

    num_copies = PAGES_TO_MOVE * NUM_LAYERS * 2
    print(
        f"calling transfer_kv_all_layer_direct_lf_pf: "
        f"{NUM_LAYERS} layers, {PAGES_TO_MOVE} pages, page_size={PAGE_SIZE} "
        f"-> {num_copies} batched copies"
    )
    # Production runs this on a dedicated stream
    # (``cache_controller.py``: ``with device_module.stream(self.write_stream)``),
    # and it has to: CUDA's own contract for cudaMemcpyBatchAsync is "hStream
    # must not be legacy NULL stream", which is exactly what torch's default
    # stream is. On the default stream the call is refused with
    # cudaErrorInvalidValue -- a property of the API, not of the wheel, and not
    # what #436 is about. ``--stream default`` exists to show that refusal.
    if use_default_stream:
        print(
            "stream: torch default (legacy NULL) -- expected to be REFUSED by the API"
        )
        stream_ctx: object = contextlib.nullcontext()
    else:
        print("stream: dedicated non-default stream, as production uses")
        stream_ctx = torch.cuda.stream(torch.cuda.Stream())

    sys.stdout.flush()  # so the last line survives a segfault

    torch.cuda.synchronize()
    with stream_ctx:
        transfer_kv_all_layer_direct_lf_pf(
            src_ptrs=src_k + src_v,
            dst_ptrs=[dst_k, dst_v],
            src_indices=src_indices,
            dst_indices=dst_indices,
            page_size=PAGE_SIZE,
        )
    torch.cuda.synchronize()
    print("call returned without a fault")

    # Reference: the same movement, done with plain per-page copies.
    ref_k = torch.zeros_like(dst_k)
    ref_v = torch.zeros_like(dst_v)
    for sp, dp in zip(src_pages, dst_pages):
        rows = slice(sp * PAGE_SIZE, (sp + 1) * PAGE_SIZE)
        for layer in range(NUM_LAYERS):
            ref_k[dp, layer] = src_k[layer][rows].cpu()
            ref_v[dp, layer] = src_v[layer][rows].cpu()

    ok = True
    for name, got, want in (("K", dst_k, ref_k), ("V", dst_v, ref_v)):
        if not torch.equal(got, want):
            ok = False
            bad = (got != want).nonzero()
            print(
                f"MISMATCH in {name}: {bad.shape[0]} differing elements, first at {bad[0].tolist()}"
            )
    if not ok:
        print("RESULT: FAIL (call completed but moved the wrong bytes)")
        return EXIT_FAIL

    print("RESULT: PASS (no fault, bytes match the per-page reference)")
    return EXIT_PASS


# --------------------------------------------------------------------------
# mode: drive
# --------------------------------------------------------------------------


def mode_drive(argv: list[str]) -> int:
    print("### stage 1: ABI probe (no GPU needed)")
    mode_abi()
    print()
    print("### stage 2: the call, in a child process so a fault is reported")
    cmd = [sys.executable, os.path.abspath(__file__), "--mode", "call", *argv]
    proc = subprocess.run(cmd, check=False)
    rc = proc.returncode
    if rc < 0:
        signo = -rc
        name = {11: "SIGSEGV", 6: "SIGABRT", 4: "SIGILL", 7: "SIGBUS"}.get(
            signo, f"signal {signo}"
        )
        print()
        print(
            f"RESULT: FAIL -- child died from {name} inside the KV transfer (#436 reproduced)"
        )
        return EXIT_FAIL
    if rc == EXIT_SKIP:
        print()
        print("RESULT: SKIPPED -- see the child's message above")
        return EXIT_SKIP
    if rc != 0:
        print()
        print(f"RESULT: FAIL -- child exited {rc}")
        return EXIT_FAIL
    print()
    print("RESULT: PASS")
    return EXIT_PASS


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=("abi", "call", "drive"), default="drive")
    ap.add_argument(
        "--stream",
        choices=("dedicated", "default"),
        default="dedicated",
        help="which stream to issue the batch copy on; 'default' is the legacy "
        "NULL stream, which the CUDA API refuses by contract",
    )
    args = ap.parse_args()

    if args.mode == "abi":
        return mode_abi()
    if args.mode == "call":
        return mode_call(use_default_stream=args.stream == "default")
    return mode_drive(["--stream", args.stream])


if __name__ == "__main__":
    raise SystemExit(main())
