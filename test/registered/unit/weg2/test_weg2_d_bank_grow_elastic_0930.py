"""A growing expert bank takes the rows the card can back -- never a dead rank.

Metal (NF y4g b8e559c3e3, D log 07:48:04, TP1): D-MEM-SCHED moved S8 -> S6 after
a request ended; RB released the KV prefix first, then the bank grew. The card
no longer had the freed bytes: ``WEG2-TMS-RESUME cuMemCreate FAILED rc=2 (out of
memory) size=4194304`` -> ``Weg2DSeatVramRefused ... tms_set_spans(46.w2_weight_packed,
now=True) rc=2`` -> SIGQUIT, W17 group dead (the same death: e033a931db, 09300130).

Expert rows are elastic by law (free VRAM belongs to the experts, missing VRAM
sends them to the host store): the grow is halved until the map fits. The KV
side stays strict -- ``test_weg2_d_stage_grow_release_first_rb`` keeps a KV the
card cannot hold a named refusal.
"""
from __future__ import annotations

import importlib.util
import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_rb_elastic", os.path.join(os.path.dirname(__file__), "test_weg2_d_stage_grow_release_first_rb.py"))
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
h = rb.h


def _at_s1_with_a_foreign_bite(bite: int):
    """S0 -> S1 (32 -> 16 rows ON), then someone else takes ``bite`` bytes of the
    card: the way back to S0 cannot map all 16 rows it would give back."""
    tms = rb.CardTms()
    ctx = h._armed(tms)
    ctx.__enter__()
    r, ctl = rb._live_s0(tms)
    tms.budget = tms.total()
    ctl.apply_stage(2, 1)
    tms.budget -= bite
    return tms, ctx, r, ctl


def test_y4g_tp1_the_bank_grow_the_card_cannot_back_keeps_the_rank_alive(caplog):
    """RED on b8e559c3e3: rc=2 on the bank -> Weg2DSeatVramRefused (the metal's
    TP1). GREEN: fewer rows ON, nothing above the card, the named line."""
    caplog.set_level(logging.WARNING)
    tms, ctx, r, ctl = _at_s1_with_a_foreign_bite(bite=1)
    try:
        applied = ctl.apply_stage(2, 0)
        assert applied.stage == 0
        assert 16 <= applied.extra_rows < 32
        assert ctl.rows_on == applied.extra_rows
        assert tms.total() <= tms.budget
        assert any("BANK-GROW-ELASTIC" in m for m in caplog.messages)
    finally:
        ctx.__exit__(None, None, None)


def test_every_bank_tensor_ends_on_the_same_row_count():
    """The retry rewrites tensors that were already mapped for the larger count:
    all row tensors of all layers must end at the SAME rows (one table)."""
    tms, ctx, r, ctl = _at_s1_with_a_foreign_bite(bite=1)
    try:
        applied = ctl.apply_stage(2, 0)
        k = applied.extra_rows
        for m in ctl.row_tensors:
            assert tms.mapped(m.ptr) == sum(b - a for a, b in ctl.bank_spans(m, k)), m.geom.name
    finally:
        ctx.__exit__(None, None, None)
