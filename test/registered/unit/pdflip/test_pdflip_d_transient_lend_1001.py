"""D-TRANSIENT-LEND: D's booked transient is expert rows between two extends.

Metal (NF fqnsdm, 01.10. 14:00-14:39Z, 1-s NVML sampler): D awake ~95 % of the
time with 1.7-2.0 GiB NVML-free per card -- corridor floor, awake overshoot and
the D-extend activation, booked statically and spent only at the extend peak.
User law 10:20Z: free VRAM is experts until it is needed.

What must hold (desk rank of test_pdflip_d_seat_live_wipe_s1_0928: two MoE layers
of 4 KiB rows, R3 + C7 + X40, S0 = 32 stage rows ON):
(1) a lend maps whole lattice cells ABOVE the phase's rows and turns them ON;
    the rows the phase holds keep their extents (bytes);
(2) the return releases exactly those cells (S1-Wisch: no straddle) and turns
    the rows OFF before a page goes;
(3) a stage move after a lend ends it in the same apply, no W-SEAT-WIPE;
(4) the verdict follows the batch kind: return before an extend, lend after
    SETTLE decode rounds;
(5) without the launcher's floor nothing changes (byte-identical lattice);
(6) the lend head is never a capture floor row.
"""
from __future__ import annotations

import importlib.util
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_live_wipe_lend", os.path.join(os.path.dirname(__file__), "test_pdflip_d_seat_live_wipe_s1_0928.py"))
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

ROW = 4096          # one bank row of the desk rank = one granule
STEP_BYTES = 4 * ROW * 2  # one lattice step (4 rows) over the two layers' banks


def _live_s0(tms):
    r = h._rank(tms)
    h._pause_all(tms)
    dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
    for p in list(tms.allocs):
        tms.resume(p)
    return r, r.sched._pdflip_d_seat_vram


def _floor(value="700"):
    return envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(value)


def _ids_of_rows(tms, bank, rows):
    return tms.ids(bank.data_ptr(), 0, (10 + rows) * ROW)


def test_the_gate_returns_before_an_extend_and_lends_after_settle():
    g = dtl.LendGate(3)
    assert [g.step("decode"), g.step("idle"), g.step("decode")] == ["hold", "hold", "lend"]
    assert g.step("extend") == "return"
    assert g.step("decode") == "hold"


def test_batch_kind_follows_the_forward_mode():
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    def b(mode):
        return types.SimpleNamespace(forward_mode=mode)

    assert dtl.batch_kind(None) == "idle"
    assert dtl.batch_kind(b(ForwardMode.EXTEND)) == "extend"
    assert dtl.batch_kind(b(ForwardMode.MIXED)) == "extend"
    assert dtl.batch_kind(b(ForwardMode.DECODE)) == "decode"
    assert dtl.batch_kind(b(ForwardMode.TARGET_VERIFY)) == "decode"


def test_a_lend_maps_cells_above_the_phase_and_keeps_its_rows():
    tms = h.CoreTms()
    with h._armed(tms), _floor():
        r, ctl = _live_s0(tms)
        assert ctl.rows_on == 32 and ctl.x_max == 40
        kept = {b.data_ptr(): _ids_of_rows(tms, b, 32) for b in r.banks}
        added = ctl.lend(10 * STEP_BYTES)
        assert added == 8 and ctl.rows_on == 40 and ctl.lent_from == 32
        for b in r.banks:
            assert _ids_of_rows(tms, b, 32) == kept[b.data_ptr()]  # (1) the phase's bytes stay
            assert tms.mapped(b.data_ptr()) == dsv.span_bytes(ctl.bank_spans(
                next(m for m in ctl.row_tensors if m.ptr == b.data_ptr()), 40))
        for c in r.caches:
            assert int(c._pool_tables.seat_on) == 40


def test_the_budget_buys_whole_lattice_steps_only():
    tms = h.CoreTms()
    with h._armed(tms), _floor():
        _r, ctl = _live_s0(tms)
        assert ctl.lend(STEP_BYTES - 1) == 0 and ctl.lent_from is None
        assert ctl.lend(STEP_BYTES) == 4 and ctl.rows_on == 36


def test_the_return_releases_exactly_the_lent_cells():
    tms = h.CoreTms()
    with h._armed(tms), _floor():
        r, ctl = _live_s0(tms)
        kept = {b.data_ptr(): _ids_of_rows(tms, b, 32) for b in r.banks}
        ctl.lend(10 * STEP_BYTES)
        assert ctl.unlend() == 8
        assert ctl.rows_on == 32 and ctl.lent_from is None and ctl.returned_rows == 8
        for b in r.banks:
            assert _ids_of_rows(tms, b, 32) == kept[b.data_ptr()]  # (2) no wipe
            m = next(m for m in ctl.row_tensors if m.ptr == b.data_ptr())
            assert tms.mapped(b.data_ptr()) == dsv.span_bytes(ctl.bank_spans(m, 32))
        assert ctl.unlend() == 0


def test_a_stage_move_after_a_lend_ends_it_without_a_wipe():
    tms = h.CoreTms()
    with h._armed(tms), _floor():
        _r, ctl = _live_s0(tms)
        ctl.lend(10 * STEP_BYTES)
        applied = ctl.apply_stage(2, 1)  # (3) S0 -> S1: 40 -> 16 in one apply
        assert (applied.stage, applied.extra_rows) == (1, 16)
        assert ctl.rows_on == 16 and ctl.lent_from is None


def test_on_batch_lends_after_settle_and_returns_before_the_extend():
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    decode = types.SimpleNamespace(forward_mode=ForwardMode.DECODE)
    extend = types.SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    tms = h.CoreTms()
    with h._armed(tms), _floor("0"), envs.FLLIPER_PDFLIP_D_LEND_SETTLE_ROUNDS.override(2), \
            mock.patch.object(dtl, "_card_free_bytes", lambda: 10 * STEP_BYTES):
        r, ctl = _live_s0(tms)
        assert dtl.on_batch(r.sched, decode) is None      # hold
        assert dtl.on_batch(r.sched, decode) == "lend"
        assert ctl.rows_on == 40
        assert dtl.on_batch(r.sched, decode) is None      # one lend per settle
        assert dtl.on_batch(r.sched, extend) == "return"  # (4) before the extend runs
        assert ctl.rows_on == 32


def test_pending_extend_work_returns_before_the_width_vote_and_lends_nothing():
    """rc12g's width vote reads the card while the batch is built (before
    on_batch): a lend still held there would narrow the extend chunk. With a
    waiting request the round's first hook gives the rows back, and a waiting
    queue never lends."""
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    decode = types.SimpleNamespace(forward_mode=ForwardMode.DECODE)
    tms = h.CoreTms()
    with h._armed(tms), _floor("0"), envs.FLLIPER_PDFLIP_D_LEND_SETTLE_ROUNDS.override(1), \
            mock.patch.object(dtl, "_card_free_bytes", lambda: 10 * STEP_BYTES):
        r, ctl = _live_s0(tms)
        r.sched.waiting_queue, r.sched.chunked_req = [], None
        assert dtl.on_batch(r.sched, decode) == "lend" and ctl.rows_on == 40
        r.sched.waiting_queue = [object()]
        assert dtl.round_start(r.sched) == "return" and ctl.rows_on == 32
        for _ in range(3):  # decode rounds while a request waits: no lend
            assert dtl.on_batch(r.sched, decode) is None
        assert ctl.rows_on == 32 and ctl.lent_from is None
        r.sched.waiting_queue = []
        assert dtl.on_batch(r.sched, decode) == "lend"


def test_without_the_floor_nothing_changes():
    tms = h.CoreTms()
    with h._armed(tms):
        _r, ctl = _live_s0(tms)
        base_cuts = ctl.row_cut_ks
        assert not dsv.lend_armed()
    tms2 = h.CoreTms()
    with h._armed(tms2), envs.FLLIPER_PDFLIP_D_TRANSIENT_LEND.override(False):
        _r2, ctl2 = _live_s0(tms2)
        assert ctl2.row_cut_ks == base_cuts  # (5) byte-identical lattice
    tms3 = h.CoreTms()
    with h._armed(tms3), _floor():
        _r3, ctl3 = _live_s0(tms3)
        assert set(dsv.lend_lattice(40, 4)) <= set(ctl3.row_cut_ks)


def test_the_lend_head_is_never_a_capture_floor_row():
    from flliper.srt.layers.moe.expert_offload import MoEExpertOffloadCache

    cache = types.SimpleNamespace(seat_rows=16, seat_lend_head=16,
                                  layer=types.SimpleNamespace(top_k=8))
    with mock.patch.object(dsv, "capture_floor_rows", lambda n, per: 7):
        assert MoEExpertOffloadCache._stage_capture_rows(cache, 4) == 0  # (6)
        cache.seat_lend_head = 0
        assert MoEExpertOffloadCache._stage_capture_rows(cache, 4) == 7


def test_presplit_adds_the_head_only_with_the_floor():
    layer = types.SimpleNamespace(moe_tp_rank=1)
    env = {dsv.GROUP_ENV: "D", dsv.POOL_GRAPH_MODE_ENV: "pool"}
    tms = h.CoreTms()
    with mock.patch.dict(os.environ, env), mock.patch.object(dsv, "_TMS", tms), \
            envs.FLLIPER_OPT_PDFLIP_D_SEAT_VRAM.override(True), \
            envs.FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS.override("0,15,15"):
        assert dsv.presplit_seat_rows(layer) == 15
        with _floor("767,700,701"), envs.FLLIPER_PDFLIP_D_LEND_HEAD_ROWS.override("16"):
            assert dsv.presplit_seat_rows(layer) == 31
            assert layer._pdflip_seat_lend_head == 16
        assert dsv.presplit_seat_rows(types.SimpleNamespace(moe_tp_rank=0)) == 0


def test_the_floor_is_read_per_rank():
    assert dsv.lend_floor_mib_for_rank(1, "767,700,701") == 700.0
    assert dsv.lend_floor_mib_for_rank(0, "650") == 650.0
    assert dsv.lend_floor_mib_for_rank(0, "") is None
    assert dtl.lend_budget_bytes(1850 << 20, 700) == 1150 << 20
    assert dtl.lend_budget_bytes(100 << 20, 700) == 0


def test_the_launcher_writes_the_ledger_floor():
    from flliper.srt.pdflip import launcher

    ledger = types.SimpleNamespace(floor_mib=(766.2, 700.0, 701.0))
    assert launcher.d_lend_floor_env(ledger) == "767,700,701"
