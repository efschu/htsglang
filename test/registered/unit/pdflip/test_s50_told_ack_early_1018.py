"""S50 (18.10.) TOLD-ACK-EARLY: a P follower's read ack leaves from its forward window when the read ends.

S48 class 1 (rc13 P 1018_200305, pdflip-8-123): reads done 20:19:07.1 on every rank, followers harvested and acked only at
20:19:10 (``harvest_ms=2603/2621``: their forward), PP0 took the acks after its commit block and admitted 20:19:11.

Drives the real ``pdflip_told_ack_early.EarlyAck`` (window lock, worker thread, read-thread wake) and the real, unchanged
``pdflip_told_fallback.follower_pump`` on a stand-in tree / channel. Danger directions: an ack BEFORE the read terminated,
an early pump OUTSIDE the forward window (concurrent with the scheduler's tree work), the scheduler leaving the window while
a pump still runs, a lost wake, misframed / reordered acks (seq), arming anywhere but a PF follower.
"""
from __future__ import annotations

import ast
import os
import pathlib
import threading
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import pdflip_told_ack_early as tae  # noqa: E402
from flliper.srt.managers import pdflip_told_fallback as fb  # noqa: E402

SRT = pathlib.Path(__import__("flliper.srt", fromlist=["x"]).__path__[0])


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_TOLD_ANCHOR_ACKS", "0")
    monkeypatch.setenv("FLLIPER_PDFLIP_TOLD_ROOM_DEFER", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_TOLD_ACK_EARLY", raising=False)
    yield
    for ea in list(tae._LISTENERS):
        ea.stop()


class Tree:
    """Stand-in tree: a read terminates when the test says so; the harvest leaves ``completed`` behind."""

    def __init__(self):
        self.done = {}
        self.completed = {}
        self.harvest_threads = []

    def prefetch_progress_is_collective_free(self):
        return True

    def check_prefetch_progress(self, rid):
        self.harvest_threads.append(threading.current_thread().name)
        return bool(self.done.get(rid))

    def completed_prefetch_tokens(self, rid):
        return self.completed.get(rid)


class Chan:
    """Stand-in ack channel: records every ack; asserts no two pump/send calls overlap (framing)."""

    def __init__(self):
        self.sent = []
        self._busy = False
        self.overlap = 0

    def _enter(self):
        if self._busy:
            self.overlap += 1
        self._busy = True

    def send_nowait(self, ack):
        self._enter()
        time.sleep(0.005)
        self.sent.append(ack)
        self._busy = False
        return True

    def pump(self, beat=True):
        self._enter()
        self._busy = False
        return True


def _sched(rank=1, pp_size=3, pf=True, tp_size=1):
    s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=rank, pp_size=pp_size, tp_size=tp_size), tree_cache=Tree())
    s._pdflip_fb_channel = Chan()
    if pf:
        s._pdflip_fb_follower = fb._FState()
    return s


def _register(s, rid, told):
    fb.follower_expect(s, rid, told)
    fb.follower_note_registered(s, types.SimpleNamespace(rid=rid))


@pytest.fixture
def own(monkeypatch):
    """own_prefix = the harvested prefix (its anchor / room terms are covered by the PF tests)."""
    monkeypatch.setattr(fb, "own_prefix", lambda sched, req, rid, told: int(sched.tree_cache.completed.get(rid, 0)))


def _wait(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


# ------------------------------------------------------------------------------------------ the metal shape (8-123)
def test_metal_ack_leaves_inside_the_forward_when_the_read_ends(own):
    """pdflip-8-123: told 64320; the follower's read ends ~50 ms into its 2.6 s forward -> the ack is out before the
    forward returns (old: at the next absorb, 2.6 s later)."""
    s = _sched()
    _register(s, "pdflip-8-123", 64320)
    win = tae.forward_window(s)
    ea = tae.instance(s)
    assert ea is not None
    with win:
        time.sleep(0.05)
        s.tree_cache.done["pdflip-8-123"] = True
        s.tree_cache.completed["pdflip-8-123"] = 64320
        tae.note_read_end()                                   # the HiCache read thread
        assert _wait(lambda: s._pdflip_fb_channel.sent)        # still inside the forward
        sent_in_forward = list(s._pdflip_fb_channel.sent)
    assert [a.reads for a in sent_in_forward] == [[("pdflip-8-123", 64320)]]
    assert sent_in_forward[0].seq == 0 and sent_in_forward[0].rank == 1
    assert "pdflip-told-ack-early" in s.tree_cache.harvest_threads
    assert ea.early_sends == 1


# ------------------------------------------------------------------------------------------ ack only after the read
def test_no_ack_before_the_read_terminated(own):
    """'Ack vor Abschluss': a wake inside the window for a read that has not terminated reports nothing; the ack goes
    out once the harvest says terminated."""
    s = _sched()
    _register(s, "r", 1000)
    win = tae.forward_window(s)
    ea = tae.instance(s)
    with win:
        tae.note_read_end()                                   # another read ended; r's still runs
        assert _wait(lambda: ea.runs >= 1)
        time.sleep(0.03)
        assert s._pdflip_fb_channel.sent == []
        s.tree_cache.done["r"] = True
        s.tree_cache.completed["r"] = 1000
        tae.note_read_end()
        assert _wait(lambda: s._pdflip_fb_channel.sent)
    assert s._pdflip_fb_channel.sent[0].reads == [("r", 1000)]


def test_short_read_acks_its_real_reach_never_told(own):
    s = _sched()
    _register(s, "r", 64320)
    with tae.forward_window(s):
        s.tree_cache.done["r"] = True
        s.tree_cache.completed["r"] = 52480                   # terminated short (P-HELD shape)
        tae.note_read_end()
        assert _wait(lambda: s._pdflip_fb_channel.sent)
    assert s._pdflip_fb_channel.sent[0].reads == [("r", 52480)]


# ------------------------------------------------------------------------------------------ the window
def test_outside_the_forward_the_worker_never_pumps(own):
    s = _sched()
    _register(s, "r", 10)
    tae.forward_window(s)                                     # create (the scheduler thread owns the window)
    ea = tae.instance(s)
    s.tree_cache.done["r"] = True
    s.tree_cache.completed["r"] = 10
    tae.note_read_end()
    time.sleep(0.1)
    assert ea.run_once() is False
    assert s._pdflip_fb_channel.sent == [] and s.tree_cache.harvest_threads == []


def test_a_wake_outside_the_window_is_served_when_the_window_opens(own):
    s = _sched()
    _register(s, "r", 10)
    tae.forward_window(s)                                     # create
    s.tree_cache.done["r"] = True
    s.tree_cache.completed["r"] = 10
    tae.note_read_end()                                       # read ended while the scheduler held the window
    time.sleep(0.05)
    assert s._pdflip_fb_channel.sent == []
    with tae.forward_window(s):
        assert _wait(lambda: s._pdflip_fb_channel.sent, timeout=0.5)   # well below the worker's 1 s idle wait


def test_the_scheduler_leaves_the_window_only_after_a_running_pump(monkeypatch):
    s = _sched()
    marks = {}

    def slow_pump(sched):
        marks["start"] = time.monotonic()
        time.sleep(0.2)
        marks["end"] = time.monotonic()

    ea = tae.EarlyAck(s, pump=slow_pump)
    setattr(s, tae.ATTR, ea)
    with ea.forward():
        ea.poke()
        assert _wait(lambda: "start" in marks)
    marks["exit"] = time.monotonic()
    assert marks["exit"] >= marks["end"]


def test_boundary_and_early_pumps_share_one_ordered_stream(own):
    s = _sched()
    for i in range(6):
        _register(s, f"r{i}", 100 + i)
    tae.forward_window(s)
    ea = tae.instance(s)
    stop = threading.Event()

    def reads():
        i = 0
        while not stop.is_set() and i < 6:
            s.tree_cache.done[f"r{i}"] = True
            s.tree_cache.completed[f"r{i}"] = 100 + i
            tae.note_read_end()
            i += 1
            time.sleep(0.01)

    t = threading.Thread(target=reads)
    t.start()
    for _ in range(6):
        with tae.forward_window(s):
            time.sleep(0.02)
        fb.follower_pump(s)                                   # the pass-boundary pump (scheduler thread, window held)
    stop.set()
    t.join()
    fb.follower_pump(s)
    sent = s._pdflip_fb_channel.sent
    assert [a.seq for a in sent] == list(range(len(sent)))
    got = [r for a in sent for r in a.reads]
    assert sorted(got) == [(f"r{i}", 100 + i) for i in range(6)] and len(got) == 6   # each read reported exactly once
    assert s._pdflip_fb_channel.overlap == 0
    assert ea.runs >= 1


# ------------------------------------------------------------------------------------------ arming
@pytest.mark.parametrize("kw", [dict(rank=0), dict(pp_size=1, rank=0), dict(pf=False), dict(tp_size=2)])
def test_never_armed_on_pp0_d_or_without_the_pf_stream(kw):
    s = _sched(**kw)
    import contextlib

    assert isinstance(tae.forward_window(s), contextlib.nullcontext)
    assert tae.instance(s) is None


def test_a_nested_forward_keeps_the_window(own):
    s = _sched()
    tae.forward_window(s)
    ea = tae.instance(s)
    with tae.forward_window(s):
        with tae.forward_window(s):
            assert ea.in_window
        assert ea.in_window and ea.window.acquire(blocking=False)   # still released: the worker could take it
        ea.window.release()
    assert not ea.in_window and not ea.window.acquire(blocking=False)


def test_switch_off(monkeypatch):
    import contextlib

    from flliper.srt.environ import envs

    assert envs.FLLIPER_PDFLIP_TOLD_ACK_EARLY.default is True
    monkeypatch.setenv("FLLIPER_PDFLIP_TOLD_ACK_EARLY", "0")
    assert isinstance(tae.forward_window(_sched()), contextlib.nullcontext)


# ------------------------------------------------------------------------------------------ call edges
def test_call_edge_the_window_is_exactly_the_plain_pp_model_forward():
    t = ast.parse((SRT / "managers/scheduler.py").read_text())
    fn = next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == "_run_batch_forward")
    withs = [n for n in ast.walk(fn) if isinstance(n, ast.With)
             and any("_pdflip_told_ack_early.forward_window(self)" in ast.unparse(i.context_expr) for i in n.items)]
    assert len(withs) == 1
    body = withs[0].body
    assert len(body) == 1 and "self.model_worker.forward_batch_generation(batch, **kwargs)" in ast.unparse(body[0])


def test_call_edge_the_read_thread_wakes_once_it_is_done_with_the_operation():
    """The wake comes from the read thread after the read-end stamp AND after its own tail work on the operation
    (probe give-back, release of the unread host slots), inside the same try (a failed read wakes nobody)."""
    src = (SRT / "managers/cache_controller.py").read_text()
    t = ast.parse(src)
    fn = next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == "prefetch_io_aux_func")
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
    body = next("\n".join(ast.unparse(b) for b in tr.body) for tr in tries
                if any("_page_transfer" in ast.unparse(b) for b in tr.body))
    k = body.index("self._page_transfer(operation)")
    i = body.index("operation.read_end_time = time.monotonic()")
    r = body.index("self.append_host_mem_release(")
    j = body.index("_told_ack_early.note_read_end(operation)")
    assert k < i < r < j
    assert body.count("_told_ack_early.note_read_end(operation)") == 1
