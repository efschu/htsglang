"""L15-CHECK-SNAP + L15-CHECK-WHO (N6e 17:34:38, 0831cdc4d3: bad=1 foreign at
gen=1 on a fully L2-backed prompt row). The sleep snapshots the wake's sample
(device + L2); at the wake each bad row is named (rid, token, slot, pre-move
slot, layers that differ) and attributed: the card moved during the pause, the
L2 page moved in place, or they already disagreed at the sleep.

Hermetic: CPU tensors; the arena load is faked through l15_sample.
"""
import inspect
from types import SimpleNamespace

import torch

from sglang.srt.weg2 import l15_check, l15_check_snap as CS, l15_sample
from sglang.srt.weg2.l15_manifest import HoldSpan, Manifest

PREFIX = [0, 1]  # one rank owns everything: compact row == slot


class _Pool:
    def __init__(self, rows, layers=2, width=3):
        self.k_buffer = [torch.zeros(rows, width) for _ in range(layers)]
        self.v_buffer = [torch.zeros(rows, width) for _ in range(layers)]


def _manifest():
    span = HoldSpan(rid="weg2-8-8", depth=4, slots=(1, 2, 3, 4), anchor_slot=9,
                    l2_slots=(11, 12, 13, 14), l2_gens=(1, 1, 1, 1))
    return Manifest(epoch=8, pid=1, spans=(span,), rows_by_rank=(8,), anchor_slots=2)


def _fill(pool, row, val):
    for b in pool.k_buffer + pool.v_buffer:
        b[row] = val


def test_snapshot_then_who_names_the_row_and_what_moved(monkeypatch):
    dev = _Pool(8)
    for r in range(8):
        _fill(dev, r, float(r))
    l2 = {11: 1.0, 12: 2.0, 13: 3.0, 14: 4.0}   # L2 equals the card at the sleep

    def fake_load(sampled, host_pool, scratch, page_tokens):
        for j, t in enumerate(sampled):
            for b in scratch.k_buffer + scratch.v_buffer:
                b[j] = l2[int(t[2])]
        return list(range(len(sampled)))

    monkeypatch.setattr(l15_sample, "load_into_scratch", fake_load)
    monkeypatch.setattr(CS, "LAST", None)
    CS.note_moves([(7, 3)])                     # token at slot 3 came from slot 7
    m = _manifest()
    assert CS.snap_at_sleep(m, 0, PREFIX, dev, SimpleNamespace(_arena_page_tokens=1)) == 4
    # during the pause: the card row 3 changes in ONE layer (a partial write)
    dev.k_buffer[1][3] = 99.0
    now_dev = l15_sample.read_rows(dev, [3])[0]
    now_l2 = l15_sample.read_rows(dev, [3])[0].clone()
    now_l2[:] = 3.0
    lines = CS.explain(0, m, PREFIX, dev, [(3, now_dev, now_l2)])
    assert len(lines) == 1
    ln = lines[0]
    assert "rid=weg2-8-8 token=2 slot=3 pre_slot=7" in ln
    assert "layers_diff=1/4" in ln
    assert "dev_moved=1 l2_moved=0 at_sleep=0" in ln


def test_who_without_snapshot_still_names_the_row(monkeypatch):
    monkeypatch.setattr(CS, "LAST", None)
    CS.note_moves([])
    dev = _Pool(8)
    m = _manifest()
    a = l15_sample.read_rows(dev, [4])[0]
    b = a.clone() + 1.0
    ln = CS.explain(0, m, PREFIX, dev, [(4, a, b)])[0]
    assert "token=3 slot=4 pre_slot=4" in ln and "layers_diff=4/4" in ln and "snap=none" in ln


def test_wiring():
    from sglang.srt.managers import scheduler
    from sglang.srt.managers.scheduler_components import weight_updater
    from sglang.srt.weg2 import l15_retain

    assert "explain_current(bad_rows)" in inspect.getsource(l15_check.sample_check)
    assert "_l15_cs.snap_at_sleep(" in inspect.getsource(scheduler)
    assert "_l15_cs.set_wake_context(rank, m, prefix, device_pool)" in inspect.getsource(weight_updater)
    CS.set_wake_context(None, None, None, None)
    assert CS.explain_current([(0, torch.zeros(1), torch.ones(1))]) == []
    assert "_l15_cs.note_moves(plan.moves)" in inspect.getsource(l15_retain)


def test_switch(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L15_CHECK_SNAP", "0")
    assert CS.enabled() is False
    assert CS.snap_at_sleep(_manifest(), 0, PREFIX, _Pool(8), None) == 0
