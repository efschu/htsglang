"""Kriech-Sitz 29.09. (z30w-park): SGLANG_OPT_WEG2_TAIL_READ_MMAP.

Measured on z30w-park (08:31-08:57Z): every tail stage on a D rank put
+250-390 MiB of glibc heap on the rank (HOST-ANON-DELTA, thread
weg2-tail-stage, malloc_inuse), which only the sleep's malloc_trim returned --
the 1.5-2 GiB anon sawtooth of the six ranks under the container's
memory.max. Source: ``read_part`` loaded the WHOLE bundle as an anonymous copy
(torch.load) and ``digest`` copied every tensor once more (tobytes).

What these cases pin:
* switch on: the bundle's tensors are views of the part file (file-backed
  storage), not a heap copy; the anon growth of a read stays far below the
  bundle's size while the copy path grows by at least the bundle;
* the same bytes and the same digest on both paths, a tampered part refused on
  both, and an unlink of the part after the read leaves the tensors readable;
* switch off: the torch.load copy, byte for byte (no file-backed storage).
"""

import os

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_handoff as th

N, PAGE, RATIO = 241, 64, 4  # c = 240, page prefix 192, 48 rows
RID, PART = "weg2-0-9", "pp0-1"


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True):
        yield tmp_path


def _write(rows: int = 48, width: int = 4096):
    """A part of 2 x 3 x rows x width float32 (48 x 4096 x 4 B = 0.75 MiB a
    tensor, 4.5 MiB the bundle) -- big enough that a heap copy shows."""
    g = torch.Generator().manual_seed(29)
    spec = th.spec_for(RID, list(range(N)), None, PAGE, RATIO)
    fa = {gid: tuple(torch.randn(rows, width, generator=g) for _ in range(3)) for gid in (3, 7)}
    gdn = {5: (torch.randn(1, 3, generator=g),)}
    h = th.write_part(spec, PART, fa, gdn)
    return h, fa, gdn


def _is_file_backed(t: torch.Tensor, path: str) -> bool:
    """The tensor's bytes lie inside a mapping of ``path`` (/proc/self/maps).
    torch 2.11 slices each tensor out of the one mmapped file storage, so the
    slice's ``storage.filename`` is None -- the address is the instrument."""
    ptr = t.data_ptr()
    real = os.path.realpath(path)
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            if lo <= ptr < hi and parts[5].strip() in (real, path):
                return True
    return False


def _rss_anon() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024
    return -1


def test_switch_on_reads_views_of_the_part_file(arena):
    h, fa, gdn = _write()
    _j, ppath = th.part_paths(RID, PART)
    with envs.SGLANG_OPT_WEG2_TAIL_READ_MMAP.override(True):
        bundle, why = th.read_part(h, check_digest=True)
    assert why == "" and bundle is not None
    for gid, ts in fa.items():
        for got, want in zip(bundle["fa"][gid], ts):
            assert _is_file_backed(got, ppath)
            assert torch.equal(got, want)
    assert torch.equal(bundle["gdn"][5][0], gdn[5][0])


def test_switch_off_is_the_heap_copy(arena):
    h, _fa, _gdn = _write()
    _j, ppath = th.part_paths(RID, PART)
    with envs.SGLANG_OPT_WEG2_TAIL_READ_MMAP.override(False):
        bundle, why = th.read_part(h, check_digest=True)
    assert why == ""
    assert not any(_is_file_backed(t, ppath) for ts in bundle["fa"].values() for t in ts)


def test_same_digest_on_both_paths(arena):
    h, fa, _gdn = _write()
    order = [t for gid in sorted(fa) for t in fa[gid]]
    with envs.SGLANG_OPT_WEG2_TAIL_READ_MMAP.override(False):
        off = th.digest(order)
    with envs.SGLANG_OPT_WEG2_TAIL_READ_MMAP.override(True):
        on = th.digest(order)
        # a non-contiguous view hashes its logical bytes, as before
        nc = fa[3][0][:, ::2]
        assert th.digest([nc]) == th.digest([nc.contiguous()])
    assert on == off == h.fa_digest


@pytest.mark.parametrize("mmap_on", [False, True])
def test_tampered_part_is_refused_on_both_paths(arena, mmap_on):
    h, _fa, gdn = _write()
    _j, ppath = th.part_paths(RID, PART)
    torch.save({"fa": {3: tuple(torch.ones(48, 4096) for _ in range(3)),
                       7: tuple(torch.ones(48, 4096) for _ in range(3))},
                "gdn": gdn}, ppath)
    with envs.SGLANG_OPT_WEG2_TAIL_READ_MMAP.override(mmap_on):
        bundle, why = th.read_part(h, check_digest=True)
    assert bundle is None and why == "digest_MISMATCH"


def test_unlink_after_read_keeps_the_views(arena):
    h, fa, _gdn = _write()
    _j, ppath = th.part_paths(RID, PART)
    with envs.SGLANG_OPT_WEG2_TAIL_READ_MMAP.override(True):
        bundle, _ = th.read_part(h, check_digest=True)
    os.remove(ppath)  # D's consumption receipt / P's prune
    assert torch.equal(bundle["fa"][7][2], fa[7][2])


def test_the_read_puts_no_bundle_on_the_heap(arena):
    # 2 x 3 x 480 x 8192 x 4 B = 90 MiB of bundle: well above the dynamic
    # mmap threshold's reach and the page-cache noise of a test process
    h, _fa, _gdn = _write(rows=480, width=8192)
    bundle_bytes = 2 * 3 * 480 * 8192 * 4

    def grown(on: bool) -> int:
        with envs.SGLANG_OPT_WEG2_TAIL_READ_MMAP.override(on):
            a0 = _rss_anon()
            bundle, why = th.read_part(h, check_digest=True)
            a1 = _rss_anon()
        assert why == ""
        del bundle
        return a1 - a0

    copy = grown(False)
    view = grown(True)
    assert copy >= 0.9 * bundle_bytes, copy  # the old path: the whole bundle, anonymous
    assert view <= 0.1 * bundle_bytes, view  # the switch: file pages, no anon copy
