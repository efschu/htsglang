import os
import platform
import sys
from array import array
from functools import lru_cache
from typing import Any, Optional


def _cpu_supports_avx2() -> bool:
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            return "avx2" in f.read().lower()
    except OSError:
        return False


@lru_cache(maxsize=1)
def _load_native_hash_module() -> Any:
    if sys.byteorder != "little" or not sys.platform.startswith("linux"):
        raise RuntimeError(
            "HiCache native hash is only supported on little-endian Linux"
        )

    try:
        from torch.utils.cpp_extension import load

        from sglang.jit_kernel.baton_health import jit_build_guard

        abs_path = os.path.dirname(os.path.abspath(__file__))
        sources = [f"{abs_path}/hash_binding.cpp"]
        extra_cflags = ["-O3", "-std=c++17", "-DNDEBUG"]
        if _cpu_supports_avx2():
            extra_cflags.append("-mavx2")
        # This callsite is where an abandoned torch build lock was first seen
        # stalling a run for 18 minutes with the .so already built. The guard
        # tells baton_health which sources the artifact must be newer than, so
        # such a lock is recognised as orphaned on the first poll instead of
        # waited on forever.
        with jit_build_guard("hicache_hash_cpp", sources=sources):
            return load(
                name="hicache_hash_cpp",
                sources=sources,
                extra_cflags=extra_cflags,
                extra_ldflags=["-lcrypto"],
                with_cuda=False,
                verbose=False,
            )
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
