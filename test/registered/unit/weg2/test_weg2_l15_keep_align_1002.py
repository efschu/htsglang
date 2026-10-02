# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-KEEPALIGN (N3k 02.10. 01:07Z): the keep byte ranges are
granule-aligned before tms_set_keep_spans.

N3k logged the first real hold (L15-RETAIN epoch=2 n=2 on all three D
ranks, one fingerprint), then "keep-arm FAILED rank=0/1/2 rc=-3": the
native set_keep_spans (tms_csrc/core.cpp:457) refuses any range that is
not granule-aligned, empty, past the allocation, unsorted or overlapping,
and the hook collects ROW-exact ranges (row * stride * elsize). The fake
adapter below applies EXACTLY that C check, so a mutant without the
alignment is red here.
"""

from __future__ import annotations

from types import SimpleNamespace

from sglang.srt.weg2.l15_keep_arm import align_keep_ranges, arm_keep_spans

G = 2 * 1024 * 1024  # 2 MiB, the rig's VMM granularity


class _CCheckAdapter:
    """tms_set_keep_spans' validation, byte for byte (core.cpp:440-458)."""

    def __init__(self, g, sizes):
        self.g = g
        self.sizes = sizes  # base id -> md.size (granule multiple)
        self.calls = []

    def set_keep_byte_spans(self, base, spans):
        ranges = sorted((int(lo), int(hi)) for lo, hi in spans)
        self.calls.append((base.id, tuple(ranges)))
        size = self.sizes[base.id]
        prev_hi = 0
        for i, (a, b) in enumerate(ranges):
            if (b <= a or a % self.g or b % self.g or b > size
                    or (i > 0 and a < prev_hi)):
                return -3
            prev_hi = b
        return 0


class _Storage:
    def __init__(self, n):
        self._n = n

    def nbytes(self):
        return self._n


def _base(i, nbytes):
    return SimpleNamespace(id=i, device="cuda:0",
                           untyped_storage=lambda: _Storage(nbytes))


def test_row_exact_ranges_arm_after_alignment(tmp_path):
    # two layer views in one base: K rows [0, 53337) and V rows starting at
    # an unaligned view offset; unit = 32768 bytes per row (27B KV cell)
    unit = 32768
    nbytes = 2 * 60000 * unit
    base = _base(1, nbytes)
    v_off = 60000 * unit + 4096  # deliberately unaligned
    raw = [(0, 53337 * unit), (v_off, v_off + 53337 * unit)]
    ad = _CCheckAdapter(G, {1: -(-nbytes // G) * G})
    mpath = tmp_path / "m.json"
    mpath.write_text("{}")
    logs = []
    ok = arm_keep_spans(ad, {1: (base, raw)}, str(mpath), rank=0,
                        log=logs.append, granule=G)
    assert ok is True, logs
    assert mpath.exists()
    assert len(ad.calls) == 1
    # every row of the raw ranges is still covered
    (_bid, spans), = ad.calls
    for lo, hi in raw:
        assert any(a <= lo and hi <= b for a, b in spans)
    assert logs[0].startswith("L15-KEEP-ALIGN rank=0 bases=1 ")


def test_raw_ranges_are_refused_by_the_c_check(tmp_path):
    # the N3k shape without alignment: rc=-3 (pins the fake to the C rule)
    ad = _CCheckAdapter(G, {1: 4 * G})
    assert ad.set_keep_byte_spans(_base(1, 4 * G), [(0, 32768 * 3)]) == -3


def test_align_widens_caps_and_merges():
    assert align_keep_ranges([(1, 5)], 4) == [(0, 8)]
    # cap at the allocation limit
    assert align_keep_ranges([(9, 11)], 4, limit=10) == [(8, 10)]
    # overlap after widening merges; unsorted input is sorted
    assert align_keep_ranges([(9, 12), (1, 6)], 4) == [(0, 12)]
    # empty / inverted ranges drop
    assert align_keep_ranges([(5, 5), (7, 3)], 4) == []


def test_extra_bytes_are_logged(tmp_path):
    base = _base(1, 8 * G)
    ad = _CCheckAdapter(G, {1: 8 * G})
    mpath = tmp_path / "m.json"
    mpath.write_text("{}")
    logs = []
    arm_keep_spans(ad, {1: (base, [(0, G + 1)])}, str(mpath), rank=1,
                   log=logs.append, granule=G)
    # kept 2G for G+1 row bytes -> extra just under 2 MiB
    assert "row_bytes=%d kept_bytes=%d" % (G + 1, 2 * G) in logs[0]
    assert "extra_mib=2.0" in logs[0]
