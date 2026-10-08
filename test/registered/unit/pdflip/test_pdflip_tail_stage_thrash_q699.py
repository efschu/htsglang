"""Q-699 TAIL-STAGE THRASH (NF y9n abl 76163d3aef, boot ...10032307, D).

23:13:44Z P>D wake: ``PDFLIP-TAIL-STAGE-EARLY held=9 started=9`` -- nine hand-offs, KEEP_AGREED 8.
The FIFO prune at job creation dropped the job of a rid whose prefetch was still being checked;
its next check made a new job (evicting the next rid) with a FRESH ``t_first`` and a fresh
worker submission. Every pass all nine were re-staged: the H45 hold (``vote_hold``, bounded from
``t_first``) never lapsed, no read terminated -- ``ADMISSION-WEDGE: 9 queued, 0 running`` for
~190 s until the clients aborted -- and the worker queue grew to ``jobs=35397 ...
queue_wait_ms_max=342081.4``; after the aborts it read the deleted park parts until 23:22:26
(90927 ``PDFLIP-TAIL part unreadable`` tracebacks, FileNotFoundError).

Pinned: a live rid's job is never pruned under the cap (only one no check touched for
JOB_STALE_S, or beyond JOB_HARD_CAP); a rid's first-check time survives its job (the hold can
never renew); the worker skips an item whose job is gone or re-made; a missing part is a
terminal 'lost' verdict, one throttled line, no traceback; an abort drops the rid's staging.
"""

import logging
import time
import types

import pytest
import torch

from flliper.srt.environ import envs
from flliper.srt.pdflip import tail_adopt as ta
from flliper.srt.pdflip import tail_handoff as th

LOGGER = "flliper.srt.pdflip.tail_adopt"
RIDS = [f"pdflip-4-{i}" for i in range(13, 22)]  # the nine of the 23:13:44 wake


def _held():
    bf = str(torch.bfloat16)
    return ta.HeldShapes(fa={3: [ta.RowSpec([2, 8], bf)]}, gdn={0: [ta.RowSpec([4, 16], bf)]}, qsa_ratio=0)


class _FakeWorker:
    """The worker queue as the scheduler thread sees it: submissions, none run yet."""

    def __init__(self):
        self.subs = []

    def submit(self, rid, box, headers, held, verify, device, page_size=0):
        self.subs.append(rid)
        return types.SimpleNamespace(rid=rid)


@pytest.fixture
def rig(monkeypatch):
    for d in (ta._JOBS, ta._AGREED, getattr(ta, "_T_FIRST", {})):
        d.clear()
    w = _FakeWorker()
    monkeypatch.setattr(ta, "_worker", lambda: w)
    monkeypatch.setattr(ta, "_candidate", lambda rid: rid.startswith("pdflip-"))
    monkeypatch.setattr(th, "headers_for", lambda rid: [types.SimpleNamespace(rid=rid)])
    monkeypatch.setattr(th, "manifest_state", lambda headers: ("complete", 3, 3))
    monkeypatch.setattr(ta, "held_shapes", lambda kv, r2t: _held())
    tree = types.SimpleNamespace(token_to_kv_pool_allocator=types.SimpleNamespace(get_kvcache=lambda: None),
                                 req_to_token_pool=None, page_size=64)
    yield w, tree
    for d in (ta._JOBS, ta._AGREED, getattr(ta, "_T_FIRST", {})):
        d.clear()


def test_nine_handoffs_are_staged_once_and_the_hold_lapses(rig):
    w, tree = rig
    with envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.override(True), envs.FLLIPER_PDFLIP_TAIL_WAIT_MS.override(50):
        for _pass in range(5):  # check_prefetch_progress of every held rid, every pass
            for rid in RIDS:
                ta.stage(rid, tree)
        assert sorted(w.subs) == sorted(RIDS), "each rid staged once (metal: every pass, 35397 jobs)"
        assert set(ta._JOBS) == set(RIDS), "no live job pruned"
        time.sleep(0.07)
        for rid in RIDS:
            ta.stage(rid, tree)
        assert all(ta.vote_hold(rid) == 0 for rid in RIDS), "the H45 bound lapses: the reads terminate"


def test_a_stale_job_is_pruned_and_its_first_check_survives(rig):
    w, tree = rig
    with envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.override(True):
        for rid in RIDS:
            ta.stage(rid, tree)
        t0 = ta._JOBS[RIDS[0]].t_first
        ta._JOBS[RIDS[0]].t_last -= ta.JOB_STALE_S + 1  # a prefetch that never completed
        ta.stage("pdflip-5-30", tree)
        assert RIDS[0] not in ta._JOBS
        ta.stage(RIDS[0], tree)  # it is checked again after all
        assert ta._JOBS[RIDS[0]].t_first == t0, "the re-made job keeps the first check: no renewed hold"


def test_hard_cap_bounds_the_jobs(rig):
    w, tree = rig
    with envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.override(True):
        for i in range(ta.JOB_HARD_CAP + 5):
            ta.stage(f"pdflip-9-{i}", tree)
    assert len(ta._JOBS) == ta.JOB_HARD_CAP


def test_agree_and_abort_end_the_staging(rig):
    w, tree = rig
    with envs.FLLIPER_PDFLIP_TAIL_STAGE_WORKER.override(True):
        for rid in RIDS:
            ta.stage(rid, tree)
    assert ta.drop_aborted("pdflip-4-2") == 2  # pdflip-4-20, pdflip-4-21 (the scheduler's prefix match)
    assert "pdflip-4-20" not in ta._JOBS and "pdflip-4-20" not in ta._T_FIRST
    assert ta.drop_aborted("", abort_all=True) == len(RIDS) - 2
    assert not ta._JOBS


def test_worker_skips_an_item_whose_job_is_gone(monkeypatch, caplog):
    ta._JOBS.clear()
    staged = []
    monkeypatch.setattr(ta, "_stage_into", lambda box, headers, held, cd, dev: staged.append(headers[0].rid)
                        or box.append(types.SimpleNamespace(ok=True)))
    w = ta.StageWorker(seats=1, pin=False)
    hdr = [types.SimpleNamespace(rid="pdflip-4-20", spec=types.SimpleNamespace(rid="pdflip-4-20"))]
    live_box = []
    ta._JOBS["pdflip-4-21"] = ta._Job(box=live_box)
    hdr21 = [types.SimpleNamespace(rid="pdflip-4-21", spec=types.SimpleNamespace(rid="pdflip-4-21"))]
    with caplog.at_level(logging.INFO, logger=LOGGER):
        h1 = w.submit("pdflip-4-20", [], hdr, _held(), False, None, page_size=64)   # job dropped (abort)
        h2 = w.submit("pdflip-4-21", [], hdr21, _held(), False, None, page_size=64)  # job re-made: other box
        h3 = w.submit("pdflip-4-21", live_box, hdr21, _held(), False, None, page_size=64)
        for h in (h1, h2, h3):
            assert h._done.wait(5)
        time.sleep(0.3)  # the batch line follows the last done-flag
    assert staged == ["pdflip-4-21"], "only the live job's box is staged"
    assert len(live_box) == 1
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(ta.WORKER_LINE)]
    assert sum(int(m.split("skipped_stale=")[1].split()[0]) for m in lines) == 2, lines
    ta._JOBS.clear()


def test_a_missing_part_is_lost_once_named_no_traceback(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(th, "_dir", lambda: str(tmp_path))
    th._LOST["n"] = 0
    hdr = types.SimpleNamespace(spec=types.SimpleNamespace(rid="pdflip-4-20"), part="dpark2-1958")
    with caplog.at_level(logging.WARNING, logger="flliper.srt.pdflip.tail_handoff"):
        for _ in range(40):
            assert th.read_part(hdr, check_digest=False) == (None, "lost")
    msgs = [r for r in caplog.records if "PDFLIP-TAIL part lost" in r.getMessage()]
    assert 8 <= len(msgs) <= 12, len(msgs)        # throttled: 1..8, 16, 32
    assert all(r.exc_info is None for r in msgs)  # no traceback (metal: 90927 of them)
    assert not any("part unreadable" in r.getMessage() for r in caplog.records)


def test_a_lost_part_is_a_named_zero_vote(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "_dir", lambda: str(tmp_path))
    monkeypatch.setattr(th, "local_readiness", lambda spec, headers, fa, gdn: "")
    monkeypatch.setattr(ta, "cut_gate", lambda held: None)
    spec = types.SimpleNamespace(rid="pdflip-4-20")
    hdr = types.SimpleNamespace(spec=spec, part="dpark2-1958", e1=True)
    st = ta._stage_e1([hdr], _held(), False, [])
    assert st.verdict == "lost:dpark2-1958"
    assert not st.ok
