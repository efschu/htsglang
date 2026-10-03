"""TAIL-STAGE-EARLY (30.09., NF P->D flip): the E2 tail staging of the
dormant hold starts at the START of D's weight legs.

Metal (y4l 11e5db4370 / y4k dff1a7fed4, IPC flip_first_work P>D, D TP0):
``WEG2-TAIL-READY ... waited_ms`` -- the H45 hold that kept a hand-off's
FINISHED store read from terminating until its tail staging was done -- is
median 291 ms (y4l, n=12) / 286 ms (y4k, n=14), 549-757 ms at a wake cohort
of 4-6, between the kv resume and the first pass (flip_first_work median
2536 / 2334 ms). The KV read got its early start in F22 (WAKE-READ-EARLY,
issued at the legs' start); the staging still began at the first prefetch
check after the wake.

Hermetic: the staging entry is the real ``tail_adopt.stage`` with its file
and pool reads stubbed; what is pinned is WHEN a job starts and that a
partial manifest leaves nothing behind.
"""

import ast
import logging
import pathlib
import threading
import types

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_adopt as ta
from sglang.srt.weg2 import tail_handoff as th

REPO = pathlib.Path(__file__).resolve().parents[4]
LOGGER = "sglang.srt.weg2.tail_adopt"


def _switch(on):
    return envs.SGLANG_WEG2_ENABLE_TAIL_STAGE_EARLY.override(on)


@pytest.fixture
def rig(monkeypatch):
    """Every hand-off rid is a candidate, its manifest state is set per test,
    the staging read appends a marker when released."""
    ta._JOBS.clear()
    ta._AGREED.clear()
    state = {"manifest": {}, "staged": [], "gate": threading.Event()}
    state["gate"].set()
    monkeypatch.setattr(ta, "_candidate", lambda rid: rid.startswith("weg2-"))
    monkeypatch.setattr(ta, "adopt_enabled", lambda: True)
    monkeypatch.setattr(th, "headers_for", lambda rid: [types.SimpleNamespace(rid=rid)])
    monkeypatch.setattr(th, "manifest_state",
                        lambda headers: state["manifest"].get(headers[0].rid, ("complete", 3, 3)))
    monkeypatch.setattr(ta, "held_shapes", lambda kv, r2t: "held")

    def fake_stage_into(box, headers, held, check_digest, device):
        state["gate"].wait(5)
        state["staged"].append(headers[0].rid)
        box.append(types.SimpleNamespace(ok=True, end_ok=True, e1=True))

    monkeypatch.setattr(ta, "_stage_into", fake_stage_into)
    tree = types.SimpleNamespace(token_to_kv_pool_allocator=types.SimpleNamespace(get_kvcache=lambda: None),
                                 req_to_token_pool=None)
    sched = types.SimpleNamespace(tree_cache=tree, weg2_dormant_hold=[])
    yield state, sched
    for job in list(ta._JOBS.values()):
        if job.thread is not None:
            job.thread.join(5)
    ta._JOBS.clear()
    ta._AGREED.clear()


def _hold(sched, *rids):
    sched.weg2_dormant_hold = [types.SimpleNamespace(rid=r) for r in rids]


def test_switch_default_off_and_off_is_the_old_path(rig):
    state, sched = rig
    assert envs.SGLANG_WEG2_ENABLE_TAIL_STAGE_EARLY.get() is False
    _hold(sched, "weg2-2-5")
    assert ta.stage_at_wake_begin(sched) is None
    assert ta._JOBS == {}  # the first post-wake check creates the job, as before


def test_wake_begin_starts_the_staging_of_every_held_handoff(rig, caplog):
    state, sched = rig
    _hold(sched, "weg2-22-37", "weg2-22-38")
    with _switch(True), caplog.at_level(logging.INFO, logger=LOGGER):
        out = ta.stage_at_wake_begin(sched)
    assert out == {"weg2-22-37": "started", "weg2-22-38": "started"}
    for rid in out:
        ta._JOBS[rid].thread.join(5)
    assert sorted(state["staged"]) == ["weg2-22-37", "weg2-22-38"]
    # after the legs the hold does not wait: the staging is done
    assert all(ta.vote_hold(rid) == 0 for rid in out)
    line = [r.getMessage() for r in caplog.records if "WEG2-TAIL-STAGE-EARLY" in r.getMessage()]
    assert line and "held=2 started=2" in line[0]


def test_the_post_wake_check_does_not_stage_twice(rig):
    state, sched = rig
    state["gate"].clear()  # the read is still running when the check comes
    _hold(sched, "weg2-2-5")
    with _switch(True):
        ta.stage_at_wake_begin(sched)
        job = ta._JOBS["weg2-2-5"]
        ta.stage("weg2-2-5", sched.tree_cache)  # the first prefetch check after the wake
        assert ta._JOBS["weg2-2-5"] is job and ta.vote_hold("weg2-2-5") == 1
        assert ta.stage_early(["weg2-2-5"], sched.tree_cache) == {"weg2-2-5": "exists"}
    state["gate"].set()
    job.thread.join(5)
    assert state["staged"] == ["weg2-2-5"]


def test_a_partial_manifest_leaves_no_job_behind(rig):
    """P's publish threads may still write: no job, so the first post-wake
    check creates it with its own t_first (the H45 bound unchanged)."""
    state, sched = rig
    state["manifest"]["weg2-2-5"] = ("partial", 1, 3)
    _hold(sched, "weg2-2-5")
    with _switch(True):
        assert ta.stage_at_wake_begin(sched) == {"weg2-2-5": "manifest_partial"}
    assert "weg2-2-5" not in ta._JOBS


def test_group_p_says_nothing(rig, caplog):
    state, sched = rig
    _hold(sched, "p-internal-1")  # not a hand-off candidate (group P publishes)
    with _switch(True), caplog.at_level(logging.INFO, logger=LOGGER):
        assert ta.stage_at_wake_begin(sched) == {"p-internal-1": "not_candidate"}
    assert not [r for r in caplog.records if "WEG2-TAIL-STAGE-EARLY" in r.getMessage()]
    assert ta._JOBS == {}


def test_it_never_raises(rig, monkeypatch):
    state, sched = rig

    def boom(rid):
        raise RuntimeError("store gone")

    monkeypatch.setattr(th, "headers_for", boom)
    _hold(sched, "weg2-2-5")
    with _switch(True):
        assert ta.stage_at_wake_begin(sched) == {"weg2-2-5": "refused:RuntimeError"}
        sched.tree_cache = None
        monkeypatch.setattr(th, "headers_for", lambda rid: [types.SimpleNamespace(rid=rid)])
        assert ta.stage_at_wake_begin(sched)["weg2-2-5"].startswith(("refused", "started"))


def test_the_wake_leg_calls_it_beside_wake_read_early():
    src = (REPO / "python/sglang/srt/managers/scheduler_components/weight_updater.py").read_text()
    i_read = src.index("_pl3_early.issue_reads_at_wake_begin(self.scheduler)")
    i_tail = src.index("_ta_early.stage_at_wake_begin(self.scheduler)")
    i_phase = src.index('_weg2_ph("read_early")', i_tail)
    i_legs = src.index("tag_bytes = {tag: self._weg2_tag_bytes(tag) for tag in weights_tags}", i_tail)
    assert i_read < i_tail < i_phase < i_legs
