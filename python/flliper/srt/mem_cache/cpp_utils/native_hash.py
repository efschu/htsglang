import importlib.util
import logging
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
from array import array
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MODULE_LOCK = threading.Lock()
_MODULE: Any = None


def _cpu_supports_avx2() -> bool:
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            return "avx2" in f.read().lower()
    except OSError:
        return False


def _load_native_hash_module() -> Any:
    """The native hash module, loaded ONCE per process and VERIFIED.

    #1409 (boot xsn156, 2026-09-16): on two of three D ranks the first call
    returned a module object WITHOUT ``get_hash`` (the .so on disk was
    complete and carried it; the third rank and every P rank loaded it fine
    in the same minute). The prefetch thread died on the AttributeError, the
    request behind it waited 174 s into a 503, and the group could not stop
    its storage threads at the next flip. Two things changed here: the load
    is serialised (an ``lru_cache`` let two threads -- the wake-time store
    rescan and the prefetch thread -- run torch's ``load()`` concurrently on
    the same extension name), and the result is checked; a module without the
    binding is imported again from a private copy of the artifact, which
    bypasses CPython's per-process (name, path) extension cache, and a second
    miss raises instead of handing back a dead module.
    """
    global _MODULE
    m = _MODULE
    if m is not None:
        return m
    with _MODULE_LOCK:
        if _MODULE is not None:
            return _MODULE
        m = _load_via_torch()
        if not hasattr(m, "get_hash"):
            m = _reload_from_copy(m)
        _MODULE = m
        return m


def _reload_from_copy(module: Any) -> Any:
    path = getattr(module, "__file__", None)
    if not path or not os.path.exists(path):
        raise RuntimeError(
            f"#1409 native hash module {module!r} has no get_hash and no "
            f"artifact path to reload from"
        )
    name = os.path.splitext(os.path.basename(path))[0]
    private = tempfile.mkdtemp(prefix=f"flliper_{name}_{os.getpid()}_")
    copy = os.path.join(private, os.path.basename(path))
    shutil.copy2(path, copy)
    spec = importlib.util.spec_from_file_location(name, copy)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"#1409 cannot build an import spec for {copy}")
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    if not hasattr(fresh, "get_hash"):
        raise RuntimeError(
            f"#1409 native hash artifact {path} has no get_hash even after a "
            f"fresh import from {copy}; the build is not this tree's"
        )
    logger.warning(
        "#1409 NATIVE-HASH first import of %s returned a module without "
        "get_hash; reloaded from a private copy %s",
        path,
        copy,
    )
    return fresh


def _stage_sources(*, sources: list, build_dir: str) -> tuple[list, list]:
    """Copy each source INTO the build directory, writing only on a content change.

    ``load()`` hands ninja the tree's own path, and ninja decides "dirty" by
    mtime alone. A delta image replaces the whole tree from a tar whose mtimes
    are the context's creation time, while the build directory lives on a
    cache volume that outlives the image (keyed by source CONTENT). So every
    new image looked newer than the object built by the previous one, and the
    first process to hash a page rebuilt the extension -- 4-7 s, on every image
    switch of 2026-10-01 (y6h 15:20, y6m 19:52, y6o 21:36: the first request
    after boot y6o sat in queue_ms=6338). The same image booted a second time
    was warm, which is why it looked random.

    A copy that is rewritten only when its BYTES change keeps its mtime across
    images with an identical source, so ninja finds nothing to do; a real
    change rewrites it and ninja rebuilds as before. ``load_inline`` (barlink)
    has had this property all along through torch's write-if-changed.
    Returns ``(staged_paths, written_paths)``.
    """
    staged: list = []
    written: list = []
    for src in sources:
        dst = os.path.join(build_dir, os.path.basename(src))
        with open(src, "rb") as fh:
            data = fh.read()
        try:
            with open(dst, "rb") as fh:
                same = fh.read() == data
        except OSError:
            same = False
        if not same:
            # write-then-rename: ranks sharing the volume stage concurrently,
            # and the compiler must never read a half-written file
            tmp = f"{dst}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dst)
            written.append(dst)
        staged.append(dst)
    return staged, written


def _load_via_torch() -> Any:
    if sys.byteorder != "little" or not sys.platform.startswith("linux"):
        raise RuntimeError(
            "HiCache native hash is only supported on little-endian Linux"
        )

    try:
        from torch.utils.cpp_extension import load

        from flliper.jit_kernel.baton_health import (
            jit_build_guard,
            torch_build_directory,
        )

        abs_path = os.path.dirname(os.path.abspath(__file__))
        tree_sources = [f"{abs_path}/hash_binding.cpp"]
        extra_cflags = ["-O3", "-std=c++17", "-DNDEBUG"]
        if _cpu_supports_avx2():
            extra_cflags.append("-mavx2")
            isa = "avx2"
        else:
            isa = "baseline"
        # The instruction set belongs in the extension NAME, not only in the
        # flags: torch keys its build directory on the name, and that directory
        # is host-global. An -mavx2 build reached from a cache volume shared
        # with a host without AVX2 is an illegal instruction, and the reverse is
        # a silent slowdown -- the same class as the arch tag missing from the
        # JIT kernel cache key in jit_kernel/utils.py.
        libname = f"hicache_hash_cpp_{isa}"
        build_dir = str(torch_build_directory(libname))
        try:
            sources, written = _stage_sources(
                sources=tree_sources, build_dir=build_dir
            )
            staged = "written" if written else "unchanged"
        except OSError as exc:
            # never fail the load on staging: the tree path is the old
            # behaviour and costs at most the mtime rebuild
            sources, staged = tree_sources, f"off ({exc!r})"
        t0 = time.monotonic()
        # This callsite is where an abandoned torch build lock was first seen
        # stalling a run for 18 minutes with the .so already built. The guard
        # tells baton_health which sources the artifact must be newer than, so
        # such a lock is recognised as orphaned on the first poll instead of
        # waited on forever.
        with jit_build_guard(libname, sources=sources):
            module = load(
                name=libname,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_ldflags=["-lcrypto"],
                build_directory=build_dir,
                with_cuda=False,
                verbose=False,
            )
        logger.info(
            "HICACHE-HASH-JIT %s ready in %.2f s (source staged=%s in %s)",
            libname,
            time.monotonic() - t0,
            staged,
            build_dir,
        )
        return module
    except Exception as exc:
        raise RuntimeError("Failed to load HiCache native hash extension") from exc


def _native_hash_input(token_ids: Any) -> tuple[array, int, int, bool]:
    raw_token_ids = getattr(token_ids, "raw_token_ids", None)
    raw = (
        raw_token_ids()
        if raw_token_ids is not None
        else getattr(token_ids, "token_ids", token_ids)
    )

    logical_len = len(token_ids)
    is_bigram = getattr(token_ids, "is_bigram", False)

    if isinstance(raw, array) and raw.typecode in ("I", "q", "Q", "L"):
        if is_bigram and logical_len > 0 and len(raw) < logical_len + 1:
            raise ValueError("bigram token buffer is shorter than logical length")
        return raw, logical_len, 2 if is_bigram else 1, is_bigram

    if is_bigram:
        return array("I", raw[: logical_len + 1]), logical_len, 2, is_bigram

    if logical_len == 0:
        return array("I"), logical_len, 1, is_bigram

    first_token = raw[0]
    if isinstance(first_token, tuple):
        unit_width = len(first_token)
        return (
            array("I", (elem for token in raw[:logical_len] for elem in token)),
            logical_len,
            unit_width,
            is_bigram,
        )

    return array("I", raw[:logical_len]), logical_len, 1, is_bigram


def get_native_hash(
    token_ids: Any, prior_digest: Optional[bytes], page_size: Optional[int] = None
) -> str | list[str]:
    raw, logical_len, unit_width, is_bigram = _native_hash_input(token_ids)
    return _load_native_hash_module().get_hash(
        raw, logical_len, unit_width, is_bigram, prior_digest, page_size
    )
