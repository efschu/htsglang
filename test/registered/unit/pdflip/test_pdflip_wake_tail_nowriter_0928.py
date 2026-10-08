"""WT (28.09.): a request the group settled at THIS wake does not re-read a
store-short tail -- D is awake, P sleeps, nobody writes it.

THE SPECIMEN. NF rc12z22 D (dkrnfh91dprsavisnoadoptstbar1dauer09281447,
7d507357b9) TP0: pdflip-4-11 left the #1471 settle at 14:56:26 (``#x38
SETTLE-TAIL ... released``); its admission read terminated short (delivered
73600, tail 2593) and went round the rc12y fresh-mark cycle five times --
``#1068 PREFETCH DEFERRED reason=store_prefix_short`` / ``PDFLIP-LOAD-DEVICE
tokens=128`` -- until ``PREFETCH-DEFER-FALLBACK ... cycles=5 bound=4
reason=no_writer_progress`` at 14:56:29: 3 s of re-reads nobody could grow.

Pinned (hermetic, the rc12y harness): the settled request falls back on its
FIRST short read (``reason=settled_no_writer``, no DEFERRED line); the same
request from an earlier wake, a dormant D, the switch off and an unsettled
request keep the rc12y bound; the stamp is set on every release path and the
wake counter moves once per wake.
"""

import logging
import types

import pytest

from flliper.srt.managers import scheduler as sched_mod

import importlib.util as _ilu  # noqa: E402
import os as _os  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_rc12y_harness", _os.path.join(_os.path.dirname(__file__), "test_pdflip_store_short_fallback_rc12y.py"))
_rc12y = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rc12y)
BOUND, _cycle, _deferred_lines, _sched = _rc12y.BOUND, _rc12y._cycle, _rc12y._deferred_lines, _rc12y._sched

#: pdflip-4-11: 76193 tokens, delivered 73600, X 12288 (tail 2593 within X)
WAKE_4_11 = (76193, 73600, 76192, 12288)


@pytest.fixture(autouse=True)
def _nf_profile(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_STORE_SHORT_TAIL", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_STORE_SHORT_MAX_CYCLES", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_WAKE_SHORT_DECIDE", raising=False)


def _run(s, r, caplog, limit=20):
    outs = []
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        for _ in range(limit):
            outs.append(_cycle(s, r))
            if outs[-1] == "expired":
                break
    fb = [x.getMessage() for x in caplog.records if x.getMessage().startswith("PREFETCH-DEFER-FALLBACK")]
    return outs, fb


def _settled(wake_seq=3, stamp=3):
    s, r = _sched(*WAKE_4_11)
    s._pdflip_wake_seq = wake_seq
    r._pdflip_settled_wake = stamp
    return s, r


def test_settled_at_this_wake_falls_back_on_the_first_short_read(caplog):
    s, r = _settled()
    outs, fb = _run(s, r, caplog)
    assert outs == ["expired"]
    assert _deferred_lines(caplog) == 0
    assert len(fb) == 1 and "reason=settled_no_writer" in fb[0] and "tail=2593" in fb[0]
    assert r in s.waiting_queue and r.prefetch_deferred is None


@pytest.mark.parametrize("case", ["earlier_wake", "dormant", "switch_off", "unsettled"])
def test_everything_else_keeps_the_rc12y_bound(case, caplog, monkeypatch):
    s, r = _settled()
    if case == "earlier_wake":
        s._pdflip_wake_seq = 4
    elif case == "dormant":
        s.pdflip_dormant = True
    elif case == "switch_off":
        monkeypatch.setenv("FLLIPER_PDFLIP_WAKE_SHORT_DECIDE", "0")
    else:
        del r._pdflip_settled_wake
    outs, fb = _run(s, r, caplog)
    assert outs[0] == "deferred"
    if case == "dormant":
        assert "expired" not in outs     # a sleeping D does not count cycles (xsn344)
        return
    assert outs[-1] == "expired" and 1 < len(outs) and _deferred_lines(caplog) == BOUND
    assert "reason=no_writer_progress" in fb[0]


def test_every_release_path_stamps_this_wake():
    import inspect

    S = sched_mod.Scheduler
    hold_src = inspect.getsource(S._pdflip_release_dormant_hold)
    assert "self._pdflip_wake_seq = int(getattr(self, \"_pdflip_wake_seq\", 0) or 0) + 1" in hold_src
    assert hold_src.index("_pdflip_wake_seq = int") < hold_src.index("if not hold:")
    assert "_r._pdflip_settled_wake = self._pdflip_wake_seq" in hold_src
    tick_src = inspect.getsource(S._pdflip_post_wake_settle_tick)
    assert "req._pdflip_settled_wake = getattr(self, \"_pdflip_wake_seq\", None)" in tick_src


def test_the_settle_tick_stamps_what_the_group_releases():
    import time

    S = sched_mod.Scheduler
    h = types.SimpleNamespace(waiting_queue=[], pdflip_dormant=False, _pdflip_wake_seq=7,
                              PDFLIP_POST_WAKE_SETTLE_S=S.PDFLIP_POST_WAKE_SETTLE_S,
                              ps=types.SimpleNamespace(tp_size=1))
    h.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace(pdflip_hold_rids={"a", "b"}))
    h._pdflip_post_wake_settle_tick = types.MethodType(S._pdflip_post_wake_settle_tick, h)
    h._pdflip_group_min_flags = types.MethodType(S._pdflip_group_min_flags, h)
    st = {"a": "complete", "b": "reading"}
    h._pdflip_refetch_one = lambda req, now, allow_reissue=True: st[req.rid]
    now = time.monotonic()
    a, b = (types.SimpleNamespace(rid=x, _1471_since=now) for x in "ab")
    h.pdflip_post_wake_settle = [a, b]
    assert h._pdflip_post_wake_settle_tick() == 1
    assert a._pdflip_settled_wake == 7 and not hasattr(b, "_pdflip_settled_wake")


# --- BA (28.09.): why nothing was admitted beside the decode 14:56:27-31 ------


def test_post_wake_pass_line_names_the_admission_census():
    import inspect

    src = inspect.getsource(sched_mod.Scheduler._pdflip_post_wake_pass_log)
    assert "waiting=%d admit=%s" in src
    assert '"_admission_decline_note", None) or "-")[:240]' in src
    nb = inspect.getsource(sched_mod.Scheduler.get_next_batch_to_run)
    assert 'self._admission_decline_note = "gate=phase_prefill_blocked"' in nb
    assert 'self._admission_decline_note = "gate=congruent_lane_cadence"' in nb


def test_x_defer_speaks_the_first_defer_of_every_rid(caplog, monkeypatch):
    from flliper.srt.managers import tp_head_congruence as thc

    h = types.SimpleNamespace(_pdflip_x_defer_passes=500)   # past the global n <= 5
    h._pdflip_local_store_read_pending_ms = lambda req: 2000
    h._pdflip_x_store_read_bound_s = lambda req: 60.0
    monkeypatch.setattr(thc, "group_store_read_pending_ms", lambda hi, rid: 1500)
    monkeypatch.setattr(thc, "x_completion_verdict", lambda ms, b: thc.X_DEFER)
    xd = types.MethodType(sched_mod.Scheduler._pdflip_x_defers, h)
    reqs = [types.SimpleNamespace(rid=r, prefetch_deferred=None) for r in ("pdflip-10-16", "pdflip-11-18")]
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        for _ in range(3):
            for r in reqs:
                assert xd(r) is True
    lines = [x.getMessage() for x in caplog.records if "PDFLIP X-DEFER" in x.getMessage()]
    assert len(lines) == 2          # one per rid, the repeats stay under the global cap
    assert "pdflip-10-16" in lines[0] and "pdflip-11-18" in lines[1]
