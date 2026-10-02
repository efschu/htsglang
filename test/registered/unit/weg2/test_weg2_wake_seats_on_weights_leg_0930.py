"""WT2: D learns the phase's seat count on the WEIGHTS leg of the wake.

Metal (NF y4g b8e559c3e3, D TP0, 22 wakes): the weights leg carried no count,
so D's first request of the wake set the cap form (S0, 10 bank rows) on the
paused bank and the tags mapped it; the kv leg that carried n then moved the
LIVE bank (10 -> 20/16/15 rows mapped, or -> 6/7 with a device sync behind the
draft unpark's 1.5 GB H2D): ``WEG2-WAKE-TAIL-SUB seat_vram`` median ~140 ms,
611/1482 ms, all inside the kv call's cg_resume.

The front now sends ``handoff_n``/``parked_n``/``phase_kv_tokens`` on the
weights leg too; D plans the bank while it is paused, the tag maps it once,
and the kv leg of the same epoch changes nothing (``d_seat_vram.on_wake``:
has_n) -- no live span call on a bank tensor.
"""
from __future__ import annotations

import importlib.util
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402
from sglang.srt.weg2 import front as fr  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_251c_wt2", os.path.join(os.path.dirname(__file__), "test_weg2_d_kv_stage_runtime_251c.py"))
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


def test_the_front_sends_the_seat_counts_on_the_weights_leg():
    """RED on 351aa9c20f: the weights leg's body was {"tags", "epoch"} only."""
    src = open(fr.__file__).read()
    i = src.index('self.timed_rpc(D, "/resume_memory_occupation",\n'
                  '                               dict({"tags": pause_order, "epoch": flip_epoch}')
    assert "**_wake_extra" in src[i:i + 200]


def test_n_on_the_weights_leg_leaves_the_kv_leg_no_live_bank_move():
    """The weights leg with n plans the stage's rows on the paused bank; after
    the tags map it the kv leg (same epoch, same n) moves nothing live."""
    tms = rt.FakeTms()
    with rt._armed(tms):
        r = rt._rank(tms)
        for p in list(tms.allocs):
            tms.pause(p)
        wake = types.SimpleNamespace(epoch="e1", handoff_n=2, parked_n=0, phase_kv_tokens=100)
        st = dsv.on_wake(r.sched, wake, rt._seats(2))
        ctl = r.sched._weg2_d_seat_vram
        assert (st.stage, ctl.applied.extra_rows) == (1, 32 - 16)
        bank = {b.data_ptr() for c in r.caches for b in c._resident.values()}
        for p in bank:
            tms.resume(p)
        n_calls = len(tms.calls)
        assert dsv.on_wake(r.sched, wake, rt._seats(2)) is None
        assert not [c for c in tms.calls[n_calls:] if c[0] in bank]
        assert ctl.applied.extra_rows == 32 - 16
