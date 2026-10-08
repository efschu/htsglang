"""TAIL-STAGE-AFTER-LEGS (30.09., NF P->D flip): TAIL-STAGE-EARLY's staging
starts behind the LAST weight collect instead of at the legs' start.

Metal (IPC flip_first_work P>D + PDFLIP-BAR1 lane-time, y4s-tse 83f7fbfb2c
n=17 against y4l 11e5db4370 n=12, same dmatrix probe): legs_wall median
1544 -> 1803 ms, seat-matched at 5-6 seats ~1522 -> ~2030 ms. The copies did
not slow (P PP0 p0 deposit copy_sync_ms 674 -> 573); the D collectors issued
slowly (TP0 issue_ms 202-286 -> 359-1127 at 5-6 seats) while up to six
per-rid staging threads (part read, digest, pin_memory per layer) ran beside
them, so P's deposit lanes waited on credit (451-584 -> 951-2094 ms).

Hermetic: the real ``tail_adopt.stage_at_wake_begin`` with the staging read
stubbed; what is pinned is WHICH of the two call sites stages, that exactly
one does per wake, and where the weight leg calls the second one.
"""

import logging
import pathlib
import threading
import types

import pytest

from flliper.srt.environ import envs
from flliper.srt.pdflip import tail_adopt as ta
from flliper.srt.pdflip import tail_handoff as th

REPO = pathlib.Path(__file__).resolve().parents[4]
LOGGER = "flliper.srt.pdflip.tail_adopt"


@pytest.fixture
def rig(monkeypatch):
    ta._JOBS.clear()
    ta._AGREED.clear()
    staged = []
    monkeypatch.setattr(ta, "_candidate", lambda rid: rid.startswith("pdflip-"))
    monkeypatch.setattr(ta, "adopt_enabled", lambda: True)
    monkeypatch.setattr(th, "headers_for", lambda rid: [types.SimpleNamespace(rid=rid)])
    monkeypatch.setattr(th, "manifest_state", lambda headers: ("complete", 3, 3))
    monkeypatch.setattr(ta, "held_shapes", lambda kv, r2t: "held")

    def fake_stage_into(box, headers, held, check_digest, device):
        staged.append(headers[0].rid)
        box.append(types.SimpleNamespace(ok=True, end_ok=True, e1=True))

    monkeypatch.setattr(ta, "_stage_into", fake_stage_into)
    tree = types.SimpleNamespace(token_to_kv_pool_allocator=types.SimpleNamespace(get_kvcache=lambda: None),
                                 req_to_token_pool=None)
    sched = types.SimpleNamespace(tree_cache=tree,
                                  pdflip_dormant_hold=[types.SimpleNamespace(rid="pdflip-32-48"),
                                                     types.SimpleNamespace(rid="pdflip-32-49")])
    yield staged, sched
    for job in list(ta._JOBS.values()):
        if job.thread is not None:
            job.thread.join(5)
    ta._JOBS.clear()
    ta._AGREED.clear()


def _wake(sched):
    """The two calls of one wake leg, in the weight_updater's order."""
    a = ta.stage_at_wake_begin(sched)  # beside WAKE-READ-EARLY
    b = ta.stage_at_wake_begin(sched, site=ta.SITE_LEGS_END)  # behind the last collect
    return a, b


def test_switch_default_off():
    assert envs.FLLIPER_PDFLIP_TAIL_STAGE_AFTER_LEGS.get() is False
    assert ta.stage_site() == ta.SITE_LEG_BEGIN


def test_off_stages_at_the_legs_start_only(rig):
    staged, sched = rig
    with envs.FLLIPER_PDFLIP_ENABLE_TAIL_STAGE_EARLY.override(True):
        a, b = _wake(sched)
    assert a == {"pdflip-32-48": "started", "pdflip-32-49": "started"}
    assert b is None
    for job in ta._JOBS.values():
        job.thread.join(5)
    assert sorted(staged) == ["pdflip-32-48", "pdflip-32-49"]


def test_on_stages_behind_the_last_collect_only(rig, caplog):
    staged, sched = rig
    with envs.FLLIPER_PDFLIP_ENABLE_TAIL_STAGE_EARLY.override(True), \
            envs.FLLIPER_PDFLIP_TAIL_STAGE_AFTER_LEGS.override(True), \
            caplog.at_level(logging.INFO, logger=LOGGER):
        a, b = _wake(sched)
    assert a is None  # nothing started beside the collectors
    assert b == {"pdflip-32-48": "started", "pdflip-32-49": "started"}
    for job in ta._JOBS.values():
        job.thread.join(5)
    assert sorted(staged) == ["pdflip-32-48", "pdflip-32-49"]  # each rid once
    lines = [r.getMessage() for r in caplog.records if "PDFLIP-TAIL-STAGE-EARLY" in r.getMessage()]
    assert len(lines) == 1 and "held=2 started=2" in lines[0] and "site=legs_end" in lines[0]


def test_the_after_legs_switch_alone_stages_nothing(rig):
    """AFTER_LEGS only moves TAIL-STAGE-EARLY; without it the first post-wake
    check stages, as before."""
    staged, sched = rig
    with envs.FLLIPER_PDFLIP_TAIL_STAGE_AFTER_LEGS.override(True):
        assert _wake(sched) == (None, None)
    assert ta._JOBS == {} and staged == []


def test_the_wake_leg_calls_the_second_site_behind_the_collects():
    src = (REPO / "python/flliper/srt/managers/scheduler_components/weight_updater.py").read_text()
    i_begin = src.index("_ta_early.stage_at_wake_begin(self.scheduler)")
    i_collects = src.index('_pdflip_ph("leg_collects")', i_begin)
    i_late = src.index("_ta_late.stage_at_wake_begin(self.scheduler, site=_ta_late.SITE_LEGS_END)", i_collects)
    i_rearm = src.index("_wake_models = self._pdflip_wake_models()", i_collects)
    assert i_begin < i_collects < i_late < i_rearm
