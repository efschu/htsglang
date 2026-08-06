"""Unit tests for `barlink_bar1.BarlinkBar1Transport._abort_flag_snapshot`.

Exercises the post-abort diagnostic read of the device flag buffer.
The production method reads one 256-byte line per (topology, step, sender)
and reports only the first dword (flag/generation word) of each line.
"""

from __future__ import annotations

import ctypes
import struct
from typing import Optional

# ---------------------------------------------------------------------------
# Minimal fakes -- no CUDA, no barlink dependency
# ---------------------------------------------------------------------------


class FakeCuda:
    """Stands in for the real CUDA runtime wrapper used by the class."""

    def __init__(self, data: bytes):
        """data is the synthetic device memory the memcpy will expose."""
        self._data = data
        self.copy_calls: list[tuple[int, int, int]] = []

    def memcpy(self, dst_addr: int, src: int, count: int):
        """Copy *count* bytes from synthetic device memory to dst."""
        self.copy_calls.append((dst_addr, src, count))
        src_bytes = self._data[:count]
        # Write src_bytes into the destination ctypes buffer
        for idx, b in enumerate(src_bytes):
            ptr = ctypes.cast(dst_addr + idx, ctypes.POINTER(ctypes.c_ubyte))
            ptr.contents.value = b


class FakeBar1Transport:
    """Stripped-down shell carrying only the attrs `_abort_flag_snapshot` reads."""

    def __init__(
        self,
        rank: int = 0,
        world: int = 1,
        group: Optional[str] = None,
        own_flag: tuple = (0x1000, None, 256 * 4),
        cuda: Optional[FakeCuda] = None,
    ):
        self.rank = rank
        self.world = world
        self.group = group
        self._own_flag = own_flag
        self._cuda = cuda


def _build_lines(dwords: list[int]) -> bytes:
    """Pack *dwords* into contiguous 256-byte lines (first dword set, rest zero)."""
    line = 256
    parts = []
    for val in dwords:
        first4 = struct.pack("<I", val)
        parts.append(first4 + b"\x00" * (line - 4))
    return b"".join(parts)


def _call_snapshot(obj: FakeBar1Transport, max_lines: int = 64) -> Optional[str]:
    """Import and call the real `_abort_flag_snapshot` on our fake object."""
    from sglang.srt.distributed.device_communicators.barlink_bar1 import (
        BarlinkBar1Transport,
    )

    return BarlinkBar1Transport._abort_flag_snapshot(obj, max_lines=max_lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_none_when_fptr_is_zero():
    """_own_flag[0] == 0 means no flag buffer; snapshot must be None."""
    obj = FakeBar1Transport(
        own_flag=(0, None, 256 * 4),
        cuda=FakeCuda(b""),
    )
    result = _call_snapshot(obj)
    assert result is None


def test_returns_none_when_fsize_is_zero():
    """Zero flag buffer size means nothing to read; snapshot must be None."""
    obj = FakeBar1Transport(
        own_flag=(0x1000, None, 0),
        cuda=FakeCuda(b""),
    )
    result = _call_snapshot(obj)
    assert result is None


def test_returns_none_when_cuda_is_none():
    """Missing CUDA runtime means we cannot do device reads; snapshot must be None."""
    obj = FakeBar1Transport(
        own_flag=(0x1000, None, 256 * 4),
        cuda=None,
    )
    result = _call_snapshot(obj)
    assert result is None


def test_snapshot_contains_expected_line_entries():
    """Four lines with distinct dwords: all must appear in the output string."""
    dwords = [0xDEAD, 0xBEEF, 0xCAFE, 0xBAAD]
    data = _build_lines(dwords)
    fsize = len(data)
    obj = FakeBar1Transport(
        rank=1,
        world=3,
        group="test-tp",
        own_flag=(0x1000, None, fsize),
        cuda=FakeCuda(data),
    )
    result = _call_snapshot(obj)
    assert result is not None
    assert "1/3" in result
    assert "test-tp" in result
    for i, w in enumerate(dwords):
        assert f"{i}:{w}" in result, f"Expected line entry {i}:{w} in snapshot"


def test_max_lines_caps_output():
    """max_lines=3 must cap to exactly 3 line entries regardless of actual fsize."""
    dwords = list(range(10))  # ten lines available
    data = _build_lines(dwords)
    fsize = len(data)
    obj = FakeBar1Transport(
        own_flag=(0x1000, None, fsize),
        cuda=FakeCuda(data),
    )
    result = _call_snapshot(obj, max_lines=3)
    assert result is not None
    # Entries 0, 1, 2 must appear
    for i in range(3):
        assert f"{i}:{i}" in result, f"Expected line entry {i}:{i} in snapshot"
    # Entry 3 must NOT appear (capped)
    assert "3:3" not in result
    # The reported count must be 3
    assert "3 lines" in result


def test_little_endian_first_dword():
    """Verify little-endian byte order for the first dword of each line.

    513 = 0x0201. In LE the bytes are [0x01, 0x02, 0x00, 0x00].
    If mistakenly read as big-endian the value would be 0x01020000 = 16908288.
    """
    le_513 = struct.pack("<I", 513)
    assert le_513 == bytes([0x01, 0x02, 0x00, 0x00])

    line = 256
    data = le_513 + b"\x00" * (line - 4)
    obj = FakeBar1Transport(
        own_flag=(0x1000, None, line),
        cuda=FakeCuda(data),
    )
    result = _call_snapshot(obj)
    assert result is not None
    assert "0:513" in result, "Little-endian read must report 513"
    # Must NOT report the big-endian misinterpretation
    assert "0:16908288" not in result
