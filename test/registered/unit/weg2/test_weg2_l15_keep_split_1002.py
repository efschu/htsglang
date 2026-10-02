# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-KEEP-SPLIT (N3t 07:02:35Z): the hold survives the kv pause only on
span-mapped allocations.

N3t: L15-CHECK-DIAG live_absmax=0 / l2_absmax=218 on every sampled held row,
and the hold sleep's kv pause freed the whole pool (nvml 8714 -> 8856 MiB with
2.46 GB "kept"): tms_csrc/core.cpp pass 3 keeps only extents WHOLLY inside a
keep range, and a STOCK allocation (no extents) is unmapped whole. The fake
saver below applies exactly that rule.
"""

from __future__ import annotations

from types import SimpleNamespace

from sglang.srt.weg2 import l15_keep_split as ks
from sglang.srt.weg2.l15_keep_arm import arm_keep_spans

G = 2 * 1024 * 1024


class FakeSaver:
    """core.cpp semantics: set_spans(now) makes extents; pause keeps an extent
    only if it lies wholly inside a keep range; stock = whole allocation."""

    def __init__(self, size):
        self.size = size
        self.extents = []          # [] = stock
        self.keep = []
        self.available = True

    def info(self, ptr):
        return SimpleNamespace(size=self.size, active=True)

    def set_spans(self, ptr, plan, now):
        self.extents = [tuple(p) for p in plan]
        return 0

    def set_keep_byte_spans(self, base, ranges):
        self.keep = [tuple(r) for r in ranges]
        return 0

    def pause_kept_bytes(self):
        if not self.extents:
            return 0               # stock: unmapped whole, keep ignored
        return sum(b - a for a, b in self.extents
                   if any(k0 <= a and b <= k1 for k0, k1 in self.keep))


class Base:
    def __init__(self, ptr):
        self._ptr = ptr
        self._base = None
        self.device = "cpu"

    def data_ptr(self):
        return self._ptr


def setup_function(_f):
    ks.forget_all()


def test_stock_allocation_keeps_nothing_and_arm_refuses(tmp_path):
    sv = FakeSaver(64 * G)
    base = Base(0x1000)
    m = tmp_path / "m.json"
    m.write_text("{}")
    logs = []
    ok = arm_keep_spans(sv, {1: (base, [(0, 3 * G)])}, str(m), rank=1,
                        log=logs.append, granule=G,
                        split_lookup=lambda b, r: ks.keep_extents(b.data_ptr(), r))
    assert ok is False, "a stock base must refuse, not pretend to keep"
    assert "not-split-or-outside-hold" in logs[-1]
    assert sv.pause_kept_bytes() == 0


def test_split_base_keeps_the_whole_hold_region(tmp_path):
    sv = FakeSaver(64 * G)
    base = Base(0x2000)
    n = ks.ensure_split({0x2000: (base, [(0, G // 4)])},  # unit G/4 per row
                        lambda p: 40, lambda b: G, sv, lambda t: None,
                        lambda m: None)
    assert n == 1 and sv.extents[0] == (0, 10 * G)
    m = tmp_path / "m.json"
    m.write_text("{}")
    ok = arm_keep_spans(sv, {1: (base, [(0, 3 * G + 5)])}, str(m), rank=1,
                        log=lambda s: None, granule=G,
                        split_lookup=lambda b, r: ks.keep_extents(b.data_ptr(), r))
    assert ok is True
    assert sv.keep == [(0, 10 * G)]
    assert sv.pause_kept_bytes() == 10 * G


def test_range_beyond_the_hold_region_refuses(tmp_path):
    sv = FakeSaver(64 * G)
    base = Base(0x3000)
    ks.ensure_split({0x3000: (base, [(0, G)])}, lambda p: 4, lambda b: G, sv,
                    lambda t: None, lambda m: None)
    m = tmp_path / "m.json"
    m.write_text("{}")
    logs = []
    ok = arm_keep_spans(sv, {1: (base, [(0, 9 * G)])}, str(m), rank=1,
                        log=logs.append, granule=G,
                        split_lookup=lambda b, r: ks.keep_extents(b.data_ptr(), r))
    assert ok is False


def test_layer_major_views_get_one_hold_extent_per_layer():
    holds = ks.hold_regions([(0, 1024), (8 * G, 1024), (16 * G, 1024)],
                            hold_rows=9, granule=G, size=24 * G)
    assert holds == [(0, G), (8 * G, 9 * G), (16 * G, 17 * G)]
    plan = ks.split_plan(holds, 24 * G, G)
    assert plan[0] == (0, G) and plan[1] == (G, 8 * G) and plan[-1] == (17 * G, 24 * G)
