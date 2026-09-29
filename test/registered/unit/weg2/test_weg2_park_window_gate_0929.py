"""PARK-WINDOW-GATE (27B decision 29.09. ~13:55Z): D admits no extend whose
forward would end after the open collect window's deadline.

MEASURED (F22 marker audit): the park RPC in front of a D->P flip waits for the
D pass running when the window fires -- NF z30w-park median 0.40 s, p90 2.32 s
(09:01:38-43: a resume extend of 2564 new tokens at prefix 43712, eager expert
pass 5.1 s, park 6.07 s); 27B z30j median 0.66 s, p90 1.15 s, max 3.31 s. A
running forward cannot be parked; the gate keeps the long one from starting.
The window (98b596db36 / 741bdfcef4) stays the one decision site: the front
sends its deadline and D's X-COST-LINE, D only compares.

Hermetic, CPU. RED on 895559fed2: no gate module, no front sender, no handler.
"""

import asyncio
import collections
import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers import scheduler as sched_mod  # noqa: E402
from sglang.srt.managers.io_struct import Weg2ParkWindowReqInput  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.weg2 import park_window_gate as G  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402

# the measured lines (phase_policy X-COST-LINE header; 27B z30y 09291331 record)
LINE_27B = {"a_ms": 190.0, "b_ms": 0.575, "c_ms": 0.0010}
LINE_NF = {"a_ms": 1900.0, "b_ms": 1.352, "c_ms": 0.00088}


def _front(line):
    f = object.__new__(Front)
    f.epoch = 7
    f.session = object()
    f.counters = collections.Counter()
    f._park_window_sent = None
    f._d_cost_rows = f._d_cost_all = None
    f._x_cost_seed = None
    f._x_cost_line_of = lambda rows, every, seed: (line, "test")
    f.posted = []

    async def _rpc(g, path, body, timeout):
        f.posted.append((path, dict(body)))
        return 200, b""

    f.rpc = _rpc
    return f


def _drive(f, *lefts):
    async def _go():
        for left in lefts:
            f._park_window_send("D", left)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_go())
    return f.posted


def _sched():
    return types.SimpleNamespace()


def _win(line, left_ms):
    return Weg2ParkWindowReqInput(epoch=7, left_ms=left_ms, **line)


# ------------------------------------------------------------------ 27B (pflicht)
def test_27b_switch_off_sends_nothing_and_d_admits_byte_for_byte():
    """Off (the default until the first series): the front sends no window even
    while one is open, and D's gate is inert -- no verdict, no state written."""
    assert envs.SGLANG_WEG2_ENABLE_PARK_WINDOW_GATE.default is False
    with envs.SGLANG_WEG2_ENABLE_PARK_WINDOW_GATE.override(False):
        f = _front(LINE_27B)
        assert _drive(f, 1800, 900, -1) == [] and f._park_window_sent is None
    s = _sched()
    assert G.defers(s, object(), uncached=12000, prefix_tokens=0, batch_empty=True, running_n=3) is False
    assert vars(s) == {}, "no window: the admission touches nothing"


def test_27b_switch_on_the_gate_holds_back_what_ends_after_the_deadline():
    with envs.SGLANG_WEG2_ENABLE_PARK_WINDOW_GATE.override(True):
        f = _front(LINE_27B)
        (path, body), = _drive(f, 1000)
    assert path == G.PATH and body["left_ms"] == 1000 and body["a_ms"] == 190.0
    s = _sched()
    G.note(s, _win(LINE_27B, 1000))
    # 190 + (0.575 + 0.020) * 3000 = 1975 ms > 1000: waits
    assert G.defers(s, object(), uncached=3000, prefix_tokens=20000, batch_empty=True, running_n=2)
    # 190 + 0.595 * 1000 = 785 ms: admitted; the next one in the SAME forward
    # adds its marginal cost (a counted once): 785 + 0.595 * 500 = 1082 > 1000
    assert not G.defers(s, object(), uncached=1000, prefix_tokens=20000, batch_empty=True, running_n=2)
    assert G.defers(s, object(), uncached=500, prefix_tokens=20000, batch_empty=False, running_n=2)
    # a new pass starts over
    assert not G.defers(s, object(), uncached=500, prefix_tokens=20000, batch_empty=True, running_n=2)


# ------------------------------------------------------------------------- NF
def test_nf_the_metal_resume_extend_would_have_waited():
    """NF 09:01:38: 2564 new tokens at prefix 43712 -- 1900 + (1.352 + 0.0385)
    * 2564 = 5465 ms against a 5 s window: it waits, the park does not."""
    s = _sched()
    G.note(s, _win(LINE_NF, 5000))
    assert G.defers(s, object(), uncached=2564, prefix_tokens=43712, batch_empty=True, running_n=3)
    # NF's fixed forward cost alone (1.9 s, the eager expert pass) exceeds a short window
    G.note(s, _win(LINE_NF, 1500))
    assert G.defers(s, object(), uncached=1, prefix_tokens=0, batch_empty=True, running_n=1)


def test_an_idle_d_is_never_held():
    """With nothing running the window fires at once ('d-idle'); holding the
    extend would only idle D."""
    s = _sched()
    G.note(s, _win(LINE_NF, 100))
    assert not G.defers(s, object(), uncached=20000, prefix_tokens=0, batch_empty=True, running_n=0)


def test_the_window_clears_at_the_park_at_the_wake_and_on_the_fronts_clear(monkeypatch):
    s = _sched()
    G.note(s, _win(LINE_27B, 800))
    G.note(s, Weg2ParkWindowReqInput(epoch=7, left_ms=-1))
    assert getattr(s, G.STATE_ATTR) is None
    # the park: the window's deadline is spent
    G.note(s, _win(LINE_27B, 800))
    import sglang.srt.weg2.d_park_runtime as dpr
    monkeypatch.setattr(dpr, "park_running", lambda sched, req, late_hold_armed: "parked")
    assert Scheduler.handle_weg2_park_running(s, object()) == "parked"
    assert getattr(s, G.STATE_ATTR) is None
    # the wake clears a window of the last D phase before anything is admitted
    src = inspect.getsource(Scheduler._weg2_release_dormant_hold)
    assert '_pwg.clear(self, "wake")' in src


def test_the_front_resends_only_on_a_step_a_new_epoch_or_a_new_line():
    body, sent = G.front_message(None, epoch=3, left_ms=2000, line=LINE_27B)
    assert body is not None
    assert G.front_message(sent, epoch=3, left_ms=2000 - G.RESEND_MS + 1, line=LINE_27B)[0] is None
    assert G.front_message(sent, epoch=3, left_ms=2000 - G.RESEND_MS, line=LINE_27B)[0] is not None
    assert G.front_message(sent, epoch=4, left_ms=1990, line=LINE_27B)[0] is not None
    assert G.front_message(sent, epoch=3, left_ms=1990, line=LINE_NF)[0] is not None
    # no line yet: nothing to send; a clear goes only after a window was sent
    assert G.front_message(None, epoch=3, left_ms=500, line=None) == (None, None)
    assert G.front_message(None, epoch=3, left_ms=-1, line=LINE_27B) == (None, None)
    assert G.front_message(sent, epoch=3, left_ms=-1, line=None)[0] == {"epoch": 3, "left_ms": -1}


def test_the_window_in_force_is_in_the_rankstats_ipc():
    """IPC (RANKSTATS-S3 `sched`): the window D applies and the extends it held
    back are readable without a log line; null = no window (never 0)."""
    from sglang.srt.weg2 import rankstats as rs

    s = types.SimpleNamespace(waiting_queue=[], running_batch=types.SimpleNamespace(reqs=[1, 2]))
    assert rs._park_window_left_ms(s) is None
    G.note(s, _win(LINE_27B, 1000))
    G.defers(s, object(), uncached=3000, prefix_tokens=20000, batch_empty=True, running_n=2)
    assert rs._park_window_left_ms(s) == 1000 and s._weg2_park_window_defer_n == 1
    src = inspect.getsource(rs)
    assert '"park_window_left_ms": _park_window_left_ms(scheduler)' in src
    assert '"park_window_defers"' in src


def test_the_admission_asks_the_gate_after_the_x_gate_and_the_rpc_is_wired():
    src = inspect.getsource(sched_mod)
    i_ref = src.index('_note_skip("weg2_x_refused", req.rid)')
    i_pw = src.index('_note_skip("weg2_park_window", req.rid)')
    assert i_ref < i_pw < src.index("#791 PP ADMISSION UNIFORMITY", i_ref)
    assert "(Weg2ParkWindowReqInput, self.handle_weg2_park_window)" in src
    from sglang.srt.entrypoints import http_server

    assert any(getattr(r, "path", None) == G.PATH for r in http_server.app.routes)
