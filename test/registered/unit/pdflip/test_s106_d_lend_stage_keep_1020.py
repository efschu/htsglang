"""S106 STAGE-KEEP: a live phase move keeps the open D-TRANSIENT-LEND the card holds.

THE METAL (NF rc22-rc26, 20.10., D logs ``boot_..._1020_094747`` .. ``boot_..._1020_130752``).
Every D-MEM-SCHED stage move ended the lend in the same apply (``lent_from = None``: rows OFF
coldest first, sync, lent cells released), and the lend verdict mapped the SAME cells again a few
rounds later. rc26 TP0: 135 lend-ending stage moves, 134 re-lends right after, 133 of them back at
or above the rows the move had dropped (the card held the lend beside the new stage all along);
group decode pause of those re-lends 82.4 s in 0.44 h. Example rc26 TP0 13:27:15-13:27:18:
``LIVE-SPANS S8 rows_on 24->10`` (grow S5->S8) -> ``lend 10->28`` (407 ms gap) ->
``LIVE-SPANS S5 rows_on 28->12`` (shrink at the request's end) -> ``lend 12->28`` (697 ms gap).

WHAT MUST HOLD.
(1) A stage grow with a lend open keeps the lent rows the card holds beside the fresh KV, priced
    with the lend's own price (floor + a pending short extend's KEEP-SHORT price): no row moves
    when it all fits, the lend stays open from the NEW phase rows (``lent_from`` = the cell's k).
(2) Short card: only the lattice cells the KV + price need go (whole cells, no straddle), never
    below the card budget; nothing fits -> exactly the old move (lend ends at the cell's rows).
(3) No price (``keep_lend_need`` None: nothing priced, a big extend pending) -> byte-identical old move.
(4) A stage shrink keeps the whole lend (it maps no row), ``lent_from`` = the new cell's k.
(5) The return after a kept move gives back exactly to the new phase rows (no wipe).
(6) A live re-seat keeps the lend the same way: a SHRINK maps nothing, a GROW prices the new seats'
    GDN cells (RB: the rows' cells go first); without a price both end it as before.
(7) The price: floor with nothing pending, + the KEEP-SHORT price with a short extend pending,
    None for a big/unpriced/chunked one, no floor, or nothing lent.
(8) The runtime tick hands the price to ``apply_stage`` and the gate tops up only after a partial keep.
(9) Rank unity: three ranks with different cards run the same group-verdict moves, keep rank-local
    amounts, never enter a collective, and none maps beyond its card.
(10) D-DECODE-GAP: a gap whose opener's requests all finished is ``idle`` (rc26 13:50:44, 7.7 s
    named lend_return, running=0); a timed return names a gap only when its own wall dominates.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_decode_gap as ddg  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "_t_s106_rb", os.path.join(_HERE, "test_pdflip_d_stage_grow_release_first_rb.py"))
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
h = rb.h

ROW = 4096
STEP = 4 * ROW * 2          # one lattice step (4 rows) over the desk rank's two banks
KV_GROW = 2 * 64 * 1024     # S0 -> S1: two KV tensors x 64 tokens x 1 KiB


class FakeCard:
    """cudaMemGetInfo of the desk card: what the saver's budget leaves (no allocator cache)."""

    def __init__(self, tms):
        self.tms, self.trims = tms, 0

    def free(self):
        return int(self.tms.budget - self.tms.total())

    def trim(self):
        self.trims += 1
        return 0


def _floor(v="0"):
    return envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(v)


def _lent_s0(tms):
    """Live S0 (32 rows ON), then a lend to 40 (lent_from 32)."""
    r, ctl = rb._live_s0(tms)
    assert ctl.lend(10 * STEP) == 8 and ctl.rows_on == 40 and ctl.lent_from == 32
    return r, ctl


def _bank_mapped(tms, r):
    return sum(tms.mapped(b.data_ptr()) for b in r.banks)


# ---- (1)-(3) the stage grow -------------------------------------------------------------------

def test_rc26_stage_grow_keeps_the_whole_lend_when_the_card_holds_it(caplog):
    """RED on 1a9b2d9d03 (``apply_stage`` has no price: the lend ends, rows 40 -> 16).
    GREEN: the card holds lend + fresh KV + price -> no row moves, the lend rides on from S1's 16."""
    caplog.set_level(logging.INFO)
    tms = rb.CardTms()
    with h._armed(tms), _floor():
        r, ctl = _lent_s0(tms)
        need = 3 * STEP
        tms.budget = tms.total() + KV_GROW + need
        bank_before = _bank_mapped(tms, r)
        with mock.patch.object(dsv, "_CARD", FakeCard(tms)):
            applied = ctl.apply_stage(2, 1, keep_lend_need=need)
        assert applied.stage == 1 and ctl.rows_on == 40 and ctl.lent_from == 16
        assert _bank_mapped(tms, r) == bank_before          # no lent cell went and came back
        assert tms.total() <= tms.budget
        for t in r.kv:
            assert tms.mapped(t.data_ptr()) == (128 + h.PAGE) * 1024
        for c in r.caches:
            assert int(c._pool_tables.seat_on) == 40
        keep = [m for m in caplog.messages if "stage-keep" in m][-1]
        assert "rows_on 40->40 phase_rows=16" in keep and "rides on" in keep


def test_a_short_card_gives_back_only_the_cells_the_kv_and_the_price_need():
    tms = rb.CardTms()
    with h._armed(tms), _floor():
        r, ctl = _lent_s0(tms)
        need = 3 * STEP
        tms.budget = tms.total() + KV_GROW + need - STEP     # one lattice step short
        with mock.patch.object(dsv, "_CARD", FakeCard(tms)):
            ctl.apply_stage(2, 1, keep_lend_need=need)
        assert ctl.rows_on == 36 and ctl.lent_from == 16      # the largest cut that pays
        assert tms.total() <= tms.budget
        assert tms.budget - tms.total() >= need               # the price stays free


def test_a_half_step_short_card_keeps_the_lattice_cut_below_never_a_row_between():
    """S1-Wisch: a kept lend ends on the lattice (whole cells) -- 38 rows would pay too, but 38 is no
    cut; the keep takes 36."""
    tms = rb.CardTms()
    with h._armed(tms), _floor():
        r, ctl = _lent_s0(tms)
        tms.budget = tms.total() + KV_GROW - STEP // 2
        with mock.patch.object(dsv, "_CARD", FakeCard(tms)):
            ctl.apply_stage(2, 1, keep_lend_need=0)
        assert 38 not in ctl.row_cut_ks
        assert ctl.rows_on == 36 and ctl.lent_from == 16 and tms.total() <= tms.budget


def test_a_card_that_holds_no_lent_row_moves_exactly_as_before():
    tms = rb.CardTms()
    with h._armed(tms), _floor():
        _r, ctl = _lent_s0(tms)
        tms.budget = tms.total()                              # free 0
        need = 24 * 2 * ROW                                   # more than any kept cut can pay
        with mock.patch.object(dsv, "_CARD", FakeCard(tms)):
            applied = ctl.apply_stage(2, 1, keep_lend_need=need)
        assert (applied.extra_rows, ctl.rows_on, ctl.lent_from) == (16, 16, None)
        assert tms.total() <= tms.budget


def test_without_a_price_the_move_is_the_old_one():
    """(3) the default call (no price) ends the lend like test_pdflip_d_transient_lend_1001."""
    tms = rb.CardTms()
    with h._armed(tms), _floor():
        _r, ctl = _lent_s0(tms)
        tms.budget = tms.total() + 10 * KV_GROW
        with mock.patch.object(dsv, "_CARD", FakeCard(tms)):
            applied = ctl.apply_stage(2, 1)
        assert (applied.stage, applied.extra_rows, ctl.rows_on, ctl.lent_from) == (1, 16, 16, None)


def test_no_card_reading_keeps_nothing():
    """CPU / no context: nothing measured, nothing kept (the old move)."""
    tms = rb.CardTms()
    with h._armed(tms), _floor():
        _r, ctl = _lent_s0(tms)
        none_card = types.SimpleNamespace(free=lambda: None, trim=lambda: 0)
        with mock.patch.object(dsv, "_CARD", none_card):
            ctl.apply_stage(2, 1, keep_lend_need=0)
        assert ctl.rows_on == 16 and ctl.lent_from is None


# ---- (4)-(5) the shrink and the return ------------------------------------------------------------

def test_a_stage_shrink_keeps_the_whole_lend_and_rebases_it():
    tms = rb.CardTms()
    with h._armed(tms), _floor():
        r, ctl = rb._live_s0(tms)
        ctl.apply_stage(2, 1)                                  # S1: 16 rows
        assert ctl.lend(10 * STEP) == 24 and ctl.lent_from == 16  # 16 -> 40
        tms.budget = tms.total()
        bank_before = _bank_mapped(tms, r)
        with mock.patch.object(dsv, "_CARD", FakeCard(tms)):
            applied = ctl.apply_stage(2, 0, keep_lend_need=0)  # S1 -> S0: the KV goes, no row moves
        assert applied.stage == 0 and ctl.rows_on == 40 and ctl.lent_from == 32
        assert _bank_mapped(tms, r) == bank_before
        for t in r.kv:
            assert tms.mapped(t.data_ptr()) == (64 + h.PAGE) * 1024
        assert tms.total() <= tms.budget


def test_the_return_after_a_kept_move_ends_on_the_new_phase_rows():
    tms = rb.CardTms()
    with h._armed(tms), _floor():
        r, ctl = _lent_s0(tms)
        tms.budget = tms.total() + KV_GROW
        with mock.patch.object(dsv, "_CARD", FakeCard(tms)):
            ctl.apply_stage(2, 1, keep_lend_need=0)
        assert ctl.rows_on == 40 and ctl.lent_from == 16
        assert ctl.unlend() == 24 and ctl.rows_on == 16 and ctl.lent_from is None
        for b in r.banks:
            m = next(m for m in ctl.row_tensors if m.ptr == b.data_ptr())
            assert tms.mapped(b.data_ptr()) == dsv.span_bytes(ctl.bank_spans(m, 16))
        assert tms.total() <= tms.budget


# ---- (6) the live re-seat ---------------------------------------------------------------------------

def test_a_reseat_shrink_keeps_the_lend_a_grow_keeps_what_the_card_holds_beside_the_gdn():
    tms = rb.CardTms()
    with h._armed(tms), _floor(), envs.FLLIPER_PDFLIP_D_SEAT_REWAKE.override(True):
        r, ctl = rb._live_s0(tms)
        ctl.apply_stage(2, 1)                                  # S1, n=2: 16 rows
        ctl.lend(10 * STEP)                                    # 16 -> 40
        tms.budget = tms.total()
        with mock.patch.object(dsv, "_CARD", FakeCard(tms)):
            ctl.reseat_live(1, 1, keep_lend_need=0)            # n 2 -> 1: GDN cells go, cell k = 23
        assert ctl.rows_on == 40 and ctl.lent_from == 23
        assert tms.total() <= tms.budget
        gdn_before = sum(tms.mapped(p) for p in [r.temporal.data_ptr()])
        tms.budget = tms.total()                               # a full card: the GDN is paid by lent rows
        with mock.patch.object(dsv, "_CARD", FakeCard(tms)):
            ctl.reseat_live(2, 1, keep_lend_need=0)            # n 1 -> 2 maps the new seats' GDN cells
        gdn_fresh = sum(tms.mapped(p) for p in [r.temporal.data_ptr()]) - gdn_before
        assert gdn_fresh > 0 and tms.total() <= tms.budget     # RB: the rows' cells went first
        assert 16 < ctl.rows_on < 40 and ctl.lent_from == 16   # only what the GDN needed went
        rows_kept = ctl.rows_on
        ctl.reseat_live(1, 1, keep_lend_need=0)                # no card reading (CPU): the old move
        ctl.reseat_live(2, 1)                                  # no price: the grow ends the lend
        assert ctl.rows_on == 16 and ctl.lent_from is None and rows_kept > 16


# ---- (7) the price ------------------------------------------------------------------------------------

def _price_sched(queue=(), chunked=None):
    return types.SimpleNamespace(waiting_queue=list(queue), chunked_req=chunked,
                                 ps=types.SimpleNamespace(tp_rank=1))


def test_the_price_is_the_lends_own():
    ctl = types.SimpleNamespace(lent_from=16)
    MIB = 1 << 20
    with _floor("767,700,701"):
        assert dtl.stage_keep_need_bytes(_price_sched(), ctl) == 700 * MIB       # floor only
        with mock.patch.object(dtl, "short_pending_rows", lambda s: 200), \
                mock.patch("flliper.srt.pdflip.extend_trim.chunk_need_mib", lambda rank, rows: 300 + rows):
            assert dtl.stage_keep_need_bytes(_price_sched([object()]), ctl) == (700 + 500) * MIB
        with mock.patch.object(dtl, "short_pending_rows", lambda s: None):  # big / unpriced pending
            assert dtl.stage_keep_need_bytes(_price_sched([object()]), ctl) is None
        with mock.patch.object(dtl, "short_pending_rows", lambda s: 200), \
                mock.patch("flliper.srt.pdflip.extend_trim.chunk_need_mib", lambda rank, rows: None):
            assert dtl.stage_keep_need_bytes(_price_sched([object()]), ctl) is None  # no rate
        assert dtl.stage_keep_need_bytes(_price_sched(chunked=object()), ctl) is None
        assert dtl.stage_keep_need_bytes(_price_sched(), types.SimpleNamespace(lent_from=None)) is None
    assert dtl.stage_keep_need_bytes(_price_sched(), ctl) is None              # no floor


def test_a_held_queue_prices_like_an_empty_one():
    """S93: the held requests build no extend this pass -- the floor alone."""
    ctl = types.SimpleNamespace(lent_from=16)
    with _floor("700"), mock.patch.object(dtl, "takeable_waiting", lambda s: []):
        assert dtl.stage_keep_need_bytes(_price_sched([object()]), ctl) == 700 << 20


# ---- (8) the tick and the gate --------------------------------------------------------------------------

from test_pdflip_d_mem_sched_0929 import _req, tick_env  # noqa: E402,F401


class _LentCtl:
    def __init__(self, rows_after):
        self.cells = {(6, 0): None}
        self.applied, self.kw = [], []
        self.evicted_rows = self.refilled_rows = 0
        self.sync_ms = self.copy_ms = 0.0
        self.bytes_copied = 0
        self.rows_on, self.lent_from, self._after = 40, 32, rows_after

    def grow_room(self, n, stage, **_kw):
        return dsv.GrowRoom(ok=True, need=1, free_before=1 << 30, free_after=1 << 30, trimmed=0,
                            rows_from=40, rows_cell=16, rows_to=None, release_all=0)

    def apply_stage(self, n, stage, keep_lend_need=None):
        self.applied.append((n, stage))
        self.kw.append(keep_lend_need)
        self.rows_on, self.lent_from = self._after
        return types.SimpleNamespace(extra_rows=self.rows_on)

    def warm_rows(self):
        return 0


@pytest.mark.parametrize("after,topup", [((40, 16), True), ((36, 16), True), ((16, None), False)])
def test_the_tick_hands_the_price_and_the_gate_tops_up_after_a_kept_lend(tick_env, after, topup):
    """TOPUP (20.10.): every move that keeps the lend arms a top-up (the card changed); one that ends it does
    not (the full lend follows the settle)."""
    dsv_, sched, _caps, _floor_, _votes, _grid = tick_env
    ctl = _LentCtl(after)
    setattr(sched, dsv_.CTL_ATTR, ctl)
    sched.running_batch.reqs = [_req("a", 300_000)]
    with mock.patch.object(dtl, "stage_keep_need_bytes", lambda s, c=None: 4242):
        st = dsv_.runtime_tick(sched)
    assert st.changed and ctl.kw == [4242]
    assert dtl._gate(sched).topup is topup


def test_the_tick_without_a_price_calls_the_old_signature(tick_env):
    """A controller double without the keyword (and no lend) still moves: the call shape is the old one."""
    dsv_, sched, _caps, _floor_, _votes, _grid = tick_env

    class _Old(_LentCtl):
        def apply_stage(self, n, stage):
            self.applied.append((n, stage))
            return types.SimpleNamespace(extra_rows=6)

    ctl = _Old((16, None))
    ctl.lent_from = None
    setattr(sched, dsv_.CTL_ATTR, ctl)
    sched.running_batch.reqs = [_req("a", 300_000)]
    st = dsv_.runtime_tick(sched)
    assert st.changed and ctl.applied


# ---- (9) rank unity ---------------------------------------------------------------------------------------

def test_three_ranks_same_verdict_rank_local_amounts_no_collective():
    """The rig's form: three cards with different free bytes (5090 + two 3080). The group verdict
    (the stage sequence) is the same on every rank; each keeps what ITS card holds; no rank enters a
    collective in the keep (every collective entry point raises); no rank maps beyond its card."""
    import torch.distributed as tdist

    def _boom(*a, **k):
        raise AssertionError("S106: a collective in the stage-keep path")

    slack = {0: 6 * STEP, 1: 0, 2: -2 * STEP}             # beyond KV + price, per card (2: short)
    out = {}
    with mock.patch.object(tdist, "all_reduce", _boom), mock.patch.object(tdist, "all_gather", _boom), \
            mock.patch.object(tdist, "broadcast", _boom), mock.patch.object(tdist, "barrier", _boom):
        for rank in (0, 1, 2):
            tms = rb.CardTms()
            with h._armed(tms), _floor():
                r, ctl = _lent_s0(tms)
                tms.budget = tms.total() + KV_GROW + slack[rank]
                with mock.patch.object(dsv, "_CARD", FakeCard(tms)):
                    a1 = ctl.apply_stage(2, 1, keep_lend_need=0)     # grow
                    assert tms.total() <= tms.budget
                    a2 = ctl.apply_stage(2, 0, keep_lend_need=0)     # shrink
                    assert tms.total() <= tms.budget
                out[rank] = (a1.stage, a2.stage, ctl.rows_on, ctl.lent_from)
    assert {v[:2] for v in out.values()} == {(1, 0)}                 # one verdict
    assert out[0][2] == 40 and out[0][3] == 32                       # the 5090 keeps it all
    assert out[2][2] < 40                                            # a full 3080 pays the KV with rows
    assert len({v[2] for v in out.values()}) > 1                     # the amount is rank-local


# ---- (10) the instrument ------------------------------------------------------------------------------------

def _req_ns(rid):
    return types.SimpleNamespace(rid=rid)


def _decode(rids):
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    return types.SimpleNamespace(forward_mode=ForwardMode.DECODE, reqs=[_req_ns(x) for x in rids],
                                 batch_size=lambda: len(rids))


def _gap_sched():
    return types.SimpleNamespace(tp_rank=0)


def test_rc26_a_gap_after_the_running_requests_finished_is_idle(caplog, monkeypatch):
    """rc26 13:50:44 TP0: gap 7728 ms 'cause=lend_return' with running=0 (DQH held the new head for its
    KV device load, harvest_ms=2560) -- nothing was decoding. RED on 1a9b2d9d03: lend_return."""
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    s = _gap_sched()
    ddg.on_batch(s, _decode(["a"]), now=100.0)
    ddg.note_ms(s, "lend_return", 300.0)
    ddg.on_batch(s, _decode(["b"]), now=107.728)
    line = [m for m in caplog.messages if m.startswith("D-DECODE-GAP gap_ms")][-1]
    assert "cause=idle" in line and "return_ms=300" in line


def test_a_timed_return_names_the_gap_only_when_it_dominates(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    s = _gap_sched()
    ddg.on_batch(s, _decode(["a"]), now=0.0)
    ddg.note_ms(s, "lend_return", 300.0)
    ddg.on_batch(s, _decode(["a"]), now=2.2)               # 2200 ms gap, 300 ms of it the return
    line = [m for m in caplog.messages if m.startswith("D-DECODE-GAP gap_ms")][-1]
    assert "cause=other" in line and "return_ms=300" in line and "notes=lend_return:1" in line
    ddg.note_ms(s, "lend_return", 400.0)
    ddg.on_batch(s, _decode(["a"]), now=2.7)               # 500 ms gap, 400 ms the return
    line = [m for m in caplog.messages if m.startswith("D-DECODE-GAP gap_ms")][-1]
    assert "cause=lend_return" in line and "return_ms=400" in line


def test_the_lend_moves_are_timed_into_the_gap(monkeypatch):
    tms = h.CoreTms()
    got = []
    with h._armed(tms), _floor("0"), envs.FLLIPER_PDFLIP_D_LEND_SETTLE_ROUNDS.override(1), \
            mock.patch.object(dtl, "_card_free_bytes", lambda: 10 * STEP), \
            mock.patch.object(ddg, "note_ms", lambda s, c, ms: got.append((c, ms))):
        from flliper.srt.model_executor.forward_batch_info import ForwardMode

        r, ctl = rb._live_s0(tms)
        r.sched.waiting_queue, r.sched.chunked_req = [], None
        assert dtl.on_batch(r.sched, types.SimpleNamespace(forward_mode=ForwardMode.DECODE)) == "lend"
        assert dtl.on_batch(r.sched, types.SimpleNamespace(forward_mode=ForwardMode.EXTEND)) == "return"
    assert [c for c, _ in got] == ["lend", "lend_return"] and all(ms >= 0.0 for _, ms in got)
