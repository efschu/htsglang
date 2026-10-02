"""RB: a live KV-stage grow paid with expert rows releases the rows FIRST.

Metal (NF y3w e033a931db, D log, 01:45:51, weg2-34-56 incoming=77427):
D-MEM-SCHED grew S7 -> S8 (262144 -> 393216 tokens) live under load; the cell
table paid the KV with expert rows (``LIVE-SPANS n=3 stage=S8 rows_on 15->6``
TP0, ``12->6`` TP1, TP2 17 -> fewer). ``apply_stage`` mapped the KV prefix
FIRST and released the bank AFTER it, so each rank needed the new KV pages
while the rows' pages were still mapped. TP0 had the slack (card_free 2535
MiB), TP2 (3080, ``WEG2-VRAM-PEAK ... card_free_mib=758``) did not:
``WEG2-TMS-RESUME cuMemCreate FAILED rc=2 (out of memory) size=37748736`` ->
``Weg2DSeatVramRefused ... tms_set_spans(v8, now=True) rc=2`` -> SIGQUIT, gloo
aborts on TP0/TP1, W17 group dead.

The fake saver below is the #251c desk rank's core.cpp model with ONE extra
rule: the card holds exactly the pages mapped before the move (no slack, no
reserve), and a live ``set_spans`` that would map beyond it answers rc=2 like
cuMemCreate. RED on a332187f28 (KV first -> rc=2), GREEN with RB.
"""
from __future__ import annotations

import importlib.util
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_live_wipe_rb", os.path.join(os.path.dirname(__file__), "test_weg2_d_seat_live_wipe_s1_0928.py"))
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


class CardTms(h.CoreTms):
    """core.cpp's span logic on a card with a fixed number of bytes."""

    budget = None

    def total(self):
        return sum(self.mapped(p) for p in self.allocs)

    def set_spans(self, ptr, spans, *, now):
        a = self.allocs[int(ptr)]
        if now and a["active"] and self.budget is not None:
            want = [tuple(r) for r in spans]
            if want == [(0, a["size"])]:
                want = []
            want = want or [(0, a["size"])]
            kept = sum(e[1] - e[0] for e in a["ext"]
                       if any(lo <= e[0] and e[1] <= hi for lo, hi in want))
            after = self.total() - self.mapped(ptr) + kept + (
                sum(hi - lo for lo, hi in want) - kept)
            if after > self.budget:
                return 2  # CUDA_ERROR_OUT_OF_MEMORY, as cuMemCreate answers
        return super().set_spans(ptr, spans, now=now)


def _live_s0(tms):
    r = h._rank(tms)
    h._pause_all(tms)
    dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)  # S0, 32 rows ON
    for p in list(tms.allocs):
        tms.resume(p)  # the phase runs: bank, KV prefix and GDN pool all mapped
    return r, r.sched._weg2_d_seat_vram


def test_y3w_tp2_the_stage_grow_paid_by_expert_rows_fits_a_card_without_slack(caplog):
    """RED on a332187f28: rc=2 -> Weg2DSeatVramRefused (the metal's TP2).
    GREEN: the rows' cells go first, the KV maps into them; the LIVE-SPANS
    line names the order (the metal marker of the non-death)."""
    import logging

    caplog.set_level(logging.INFO)
    tms = CardTms()
    with h._armed(tms):
        r, ctl = _live_s0(tms)
        tms.budget = tms.total()  # the card is full: exactly the boot form
        applied = ctl.apply_stage(2, 1)  # S0 -> S1, paid by 32 -> 16 rows
        assert (applied.stage, applied.extra_rows) == (1, 16)
        assert tms.total() <= tms.budget
        for t in r.kv:
            assert tms.mapped(t.data_ptr()) == (128 + h.PAGE) * 1024
        line = [m for m in caplog.messages if dsv.LIVE_MARK in m][-1]
        assert "rows_on 32->16" in line and "order=bank-first" in line


def test_the_way_back_releases_the_kv_before_the_rows_map():
    """S1 -> S0 (the pending shrink): the KV prefix goes before the rows
    come back ON -- the old loop order, kept."""
    tms = CardTms()
    with h._armed(tms):
        r, ctl = _live_s0(tms)
        tms.budget = tms.total()
        ctl.apply_stage(2, 1)
        applied = ctl.apply_stage(2, 0)
        assert (applied.stage, applied.extra_rows) == (0, 32)
        assert tms.total() <= tms.budget


def test_a_card_really_too_small_still_stops_by_name():
    """No reserve is invented: a grow the card cannot hold even after the
    release is still the named refusal, never a silent partial form."""
    tms = CardTms()
    with h._armed(tms):
        r, ctl = _live_s0(tms)
        tms.budget = tms.total() - 64 * 1024
        with pytest.raises(dsv.Weg2DSeatVramRefused):
            ctl.apply_stage(2, 1)
