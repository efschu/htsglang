"""weg2xsn272 (18.09.2026): P prefilled six ~98k prompts and kept them for D;
the seventh (span 95,476 tokens) never fit PP0's pool (max_total_num_tokens
121,190) -- the adder answered NO_TOKEN silently with an EMPTY running
batch, the scheduler spun (#969M ARM every ms) and the front waited for a
drain that could not end: IDLE-WEDGE after 150 s, boot killed.

Now the scheduler names the stall (WEG2-INTAKE-STALL, 503 in W88's exit
form) once the same request has been refused with nothing running for
hold_s; the front requeues it at the head, stops dispatching, aborts it on
every P rank and flips.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import intake_stall as st  # noqa: E402


def _obs(w, rid, now, **kw):
    kw.setdefault("need_tokens", 95476)
    kw.setdefault("rem_total_tokens", 23000)
    kw.setdefault("cur_rem_tokens", 23000)
    kw.setdefault("running_empty", True)
    return w.observe(rid=rid, now=now, **kw)


def test_the_stall_is_named_once_after_the_hold_with_nothing_running():
    w = st.IntakeStallWatch(hold_s=1.0)
    assert _obs(w, "weg2-5-4", 100.0) is None          # first sight: clock starts
    assert _obs(w, "weg2-5-4", 100.5) is None          # inside the hold
    msg = _obs(w, "weg2-5-4", 101.2)
    assert msg and msg.startswith(st.STALL_MARK) and "rid=weg2-5-4" in msg
    assert "need_tokens=95476" in msg and "rem_total_tokens=23000" in msg
    assert w.stalls == 1
    assert _obs(w, "weg2-5-4", 105.0) is None          # reported once, never twice
    assert st.is_intake_stall(f"leg1 on P returned 503: b'{msg[:80]}'")
    assert not st.is_intake_stall("leg1 on P returned 503: b'W88 ...'")


def test_a_running_batch_or_progress_resets_the_clock():
    w = st.IntakeStallWatch(hold_s=1.0)
    assert _obs(w, "r", 0.0) is None
    assert _obs(w, "r", 0.6, running_empty=False) is None   # something runs: not a stall
    assert _obs(w, "r", 1.5) is None                         # clock restarted at 1.5
    assert _obs(w, "r", 2.4) is None
    assert _obs(w, "r", 2.6) is not None
    w2 = st.IntakeStallWatch(hold_s=1.0)
    assert _obs(w2, "a", 0.0) is None
    w2.progress()
    assert _obs(w2, "a", 1.5) is None                        # restarted by progress
    assert _obs(w2, "b", 1.6) is None                        # a different rid restarts too
    assert _obs(w2, "b", 2.7) is not None


def test_the_scheduler_names_the_stall_only_with_an_empty_batch_and_on_group_p():
    from sglang.srt.managers import scheduler as sch
    src = open(sch.__file__).read()
    i = src.index('self._pp_batch_full_setter = "add_one_req_NO_TOKEN"')
    blk = src[i:i + 900]
    assert "running_batch.is_empty()" in blk and "not adder.can_run_list" in blk
    assert "self.chunked_req is None" in blk and "_weg2_intake_stall_observe(req, adder)" in blk
    j = src.index("def _weg2_intake_stall_observe")
    body = src[j:j + 3000]
    assert 'GROUP_ENV' in body and '!= "P"' in body                 # group P only
    assert "HTTPStatus.SERVICE_UNAVAILABLE" in body                  # W88's exit form
    assert "self.waiting_queue = [q for q in self.waiting_queue if id(q) != refused_id]" in body


def test_the_front_requeues_at_the_head_stops_dispatching_and_never_hands_off():
    from sglang.srt.weg2 import front as fr
    src = open(fr.__file__).read()
    assert "intake_stalled: bool = False" in src                      # Pending field
    i = src.index("async def one(p: Pending) -> Pending:")
    assert "if is_intake_stall(e):" in src[i:i + 900]
    assert "await self._requeue_intake_stalled(p, e)" in src[i:i + 900]
    j = src.index("def _on_leg1_done(p: Pending)")
    assert "if p.intake_stalled:" in src[j:j + 300]
    k = src.index("passes = await _p_drain_pool(")
    assert 'not self._p_intake_stalled' in src[k:k + 300]
    m = src.index("async def _requeue_intake_stalled")
    body = src[m:m + 2000]
    assert "self.queue.appendleft(p)" in body and '"/abort_request"' in body
    assert "self._p_intake_stalled = True" in body
    # the drain resets the flag at its start, so the next P phase dispatches again
    assert "self._p_intake_stalled = False" in src[src.index("t_drain0 = time.time()"):][:600]


def test_requeue_helper_puts_the_request_first_and_aborts_on_p():
    import asyncio
    asyncio.run(_requeue_case())


async def _requeue_case():
    import asyncio
    import collections
    from types import SimpleNamespace
    from sglang.srt.weg2 import front as fr
    calls = []

    class _F:
        queue = collections.deque()
        counters = collections.Counter()
        groups = {"P": SimpleNamespace(url="http://p")}
        _p_intake_stalled = False

        async def rpc(self, g, path, payload, timeout):
            calls.append((path, payload))
            return 200, b"{}"

    f = _F()
    older = SimpleNamespace(rid="old")
    f.queue.append(older)
    loop = asyncio.get_running_loop()
    p = fr.Pending(rid="weg2-5-4", path="/generate", payload={}, text="", t_arrive=0.0,
                   fut=loop.create_future(), est_prompt=134757)
    await fr.Front._requeue_intake_stalled(f, p, RuntimeError("leg1 on P returned 503: WEG2-INTAKE-STALL rid=weg2-5-4"))
    assert list(f.queue)[0] is p and list(f.queue)[1] is older
    assert p.intake_stalled and p.x_requeues == 1 and not p.leg1_done
    assert f._p_intake_stalled is True and f.counters["p_intake_stalls"] == 1
    assert calls == [("/abort_request", {"rid": "weg2-5-4"})]
    assert not p.fut.done()                                           # the client keeps waiting


def test_xsn273_the_seat_gate_in_front_of_the_adder_feeds_the_watch_too():
    """xsn273 (a612f98346): the NO_TOKEN hook never ran -- P declined at the
    SEAT gate (`get_num_allocatable_reqs(0) <= 0`: the six parked, prefilled
    requests held every request slot) before any request reached the adder;
    the same 150-s idle, the same kill. The decline branch now observes the
    head of the waiting queue with the gate's own terms."""
    from sglang.srt.managers import scheduler as sch
    src = open(sch.__file__).read()
    i = src.index("weg2xsn273: the seat gate declined")
    blk = src[i:i + 1200]
    assert "if running_batch.is_empty() and self.waiting_queue:" in blk
    assert "self._weg2_intake_stall_observe(" in blk and "self.waiting_queue[0], None," in blk
    assert "allocatable_reqs=" in blk and "req_slots_free=" in blk
    w = st.IntakeStallWatch(hold_s=1.0)
    assert _obs(w, "weg2-5-4", 0.0) is None
    msg = _obs(w, "weg2-5-4", 1.1, extra="gate=seats allocatable_reqs=0 req_slots_free=0 waiting=1")
    assert msg and "gate=seats allocatable_reqs=0" in msg and "need_tokens=95476" in msg


def test_xsn275_the_admission_wedge_drain_hands_the_head_to_the_refusal():
    """xsn275 (812e04a7f4+e1bcd01374): neither the NO_TOKEN hook nor the seat
    gate ran in the wedged state; the watchdog's ADMISSION-WEDGE detector
    did (posted, drained NOT_APPLICABLE). The drain now hands the head of
    the waiting queue to the refusal, immediately (the detector already
    waited >= 20 s with nothing running)."""
    from types import SimpleNamespace
    from sglang.srt.managers import wedge_recovery as wr
    src = open(wr.__file__).read()
    i = src.index("def drain(self, scheduler")
    assert "_weg2_intake_stall_from_wedge(scheduler, reason)" in src[i:i + 2500]
    calls = []

    class _Batch:
        def __init__(self, empty): self._e = empty
        def is_empty(self): return self._e

    head = SimpleNamespace(rid="weg2-5-4")
    sch = SimpleNamespace(waiting_queue=[head], running_batch=_Batch(True), chunked_req=None,
                          _weg2_intake_stall_observe=lambda req, adder, note="", immediate=False: calls.append((req.rid, note, immediate)))
    wr._weg2_intake_stall_from_wedge(sch, "ADMISSION-WEDGE: 1 queued, 0 running")
    assert calls == [("weg2-5-4", "gate=admission-wedge waiting=1 reason=ADMISSION-WEDGE: 1 queued, 0 running", True)]
    # something running, or a chunked request in flight: not a stall
    calls.clear()
    wr._weg2_intake_stall_from_wedge(SimpleNamespace(waiting_queue=[head], running_batch=_Batch(False), chunked_req=None,
                                                     _weg2_intake_stall_observe=lambda *a, **k: calls.append(1)), "r")
    wr._weg2_intake_stall_from_wedge(SimpleNamespace(waiting_queue=[head], running_batch=_Batch(True), chunked_req=object(),
                                                     _weg2_intake_stall_observe=lambda *a, **k: calls.append(1)), "r")
    wr._weg2_intake_stall_from_wedge(SimpleNamespace(waiting_queue=[], running_batch=_Batch(True), chunked_req=None,
                                                     _weg2_intake_stall_observe=lambda *a, **k: calls.append(1)), "r")
    assert calls == []
    # immediate: the watch names the stall on the first sight, once
    w = st.IntakeStallWatch(hold_s=1.0)
    msg = _obs(w, "weg2-5-4", 0.0, immediate=True, extra="gate=admission-wedge")
    assert msg and "gate=admission-wedge" in msg
    assert _obs(w, "weg2-5-4", 0.1, immediate=True) is None


def test_xsn276_the_tokenizer_abort_reaches_every_rank_on_a_weg2_group(monkeypatch):
    """xsn276 (96e5283fd1): the refusal fired (P-INTAKE-STALL, /abort_request
    -> 200) and P died in the flip: W29 on PP2, 'release_memory_occupation
    should be called only when server is idle'. The tokenizer's abort_request
    returned early because the rid was no longer in rid_to_state (the 503 had
    finalised it), so PP1/PP2 kept the request. On a Weg 2 group the abort is
    dispatched anyway."""
    from sglang.srt.managers import tokenizer_manager as tm
    src = open(tm.__file__).read()
    i = src.index("def abort_request(self, rid")
    blk = src[i:i + 1600]
    assert "abort_must_reach_every_rank(rid_known=False, abort_all=abort_all)" in blk
    assert "dispatched to every" in blk
    monkeypatch.delenv("SGLANG_WEG2_GROUP", raising=False)
    assert st.abort_must_reach_every_rank(rid_known=False, abort_all=False) is False
    assert st.abort_must_reach_every_rank(rid_known=True, abort_all=False) is True
    assert st.abort_must_reach_every_rank(rid_known=False, abort_all=True) is True
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert st.abort_must_reach_every_rank(rid_known=False, abort_all=False) is True
