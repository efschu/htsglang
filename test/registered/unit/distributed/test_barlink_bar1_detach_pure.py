"""VA-stable BAR1 detach/re-attach: the pure parts (no GPU).

The metal half is benchmark/bar1_reattach_probe.py. Pinned here: the BAR1
slice rule (same as _bind_region), the re-attach plan that decides between
in-place, fixed remap and refusal, and that FixedMap really replaces a
mapping at the SAME address (the property the whole remap path rests on).
"""
import ctypes
import mmap
import os
import tempfile

from sglang.srt.distributed.device_communicators.barlink_bar1_detach import (
    MAP_ANONYMOUS,
    MAP_PRIVATE,
    MAP_SHARED,
    PROT_NONE,
    PROT_READ,
    PROT_WRITE,
    FixedMap,
    RegionState,
    contiguous_bar1_slice,
    reattach_plan,
)

PAGE = mmap.PAGESIZE
BASE, END = 0x1000_0000, 0x1000_0000 + (256 << 20)


def test_slice_takes_the_contiguous_beginning_inside_the_aperture():
    sg = [(BASE + 0x200000, 0x100000), (BASE + 0x300000, 0x100000),
          (BASE + 0x500000, 0x100000),          # gap: not part of the slice
          (0x10, 0x1000)]                       # outside the aperture
    assert contiguous_bar1_slice(sg, BASE, END) == (0x200000, 0x200000)


def test_slice_outside_the_aperture_is_empty():
    assert contiguous_bar1_slice([(0x10, 0x1000)], BASE, END) == (0, 0)


def _old(off=0x200000, length=0x200000, lead=0):
    return RegionState(peer=1, kind="payload", bar1_offset=off, length=length,
                       lead_in=lead, reg_address=0x7f0000000000, dev_ptr=0x7f0000000000)


def test_plan_in_place_when_the_offset_is_unchanged():
    assert reattach_plan(_old(), 0x200000, 0x200000, PAGE)[0] == "in_place"


def test_plan_remap_when_the_offset_moved_page_aligned():
    how, why = reattach_plan(_old(), 0x600000, 0x400000, PAGE)
    assert how == "remap" and "0x200000" in why and "0x600000" in why


def test_plan_refuses_a_shorter_slice_even_at_the_same_offset():
    how, why = reattach_plan(_old(), 0x200000, 0x1ff000, PAGE)
    assert how == "refuse" and "shrank" in why


def test_plan_refuses_a_different_lead_in():
    how, why = reattach_plan(_old(lead=0), 0x600100, 0x400000, PAGE)
    assert how == "refuse" and "lead-in" in why


def test_fixedmap_replaces_a_mapping_at_the_same_address():
    first = mmap.mmap(-1, 4 * PAGE)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(first))
    ph = FixedMap(addr, 4 * PAGE, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "f")
        with open(path, "wb") as f:
            f.write(b"\xab" * (4 * PAGE))
        fd = os.open(path, os.O_RDWR)
        try:
            fm = FixedMap(addr, 4 * PAGE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        finally:
            os.close(fd)
        ph._open = False
        assert fm.addr == addr
        assert ctypes.string_at(addr, 4) == b"\xab" * 4
        ctypes.memset(addr, 0xcd, 1)
        fm.close()
        with open(path, "rb") as f:
            assert f.read(1) == b"\xcd"
    # the original Python object's later munmap of this now-unmapped range is harmless
    first.close()
