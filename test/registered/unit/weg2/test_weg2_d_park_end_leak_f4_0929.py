"""F4 park parts leave /dev/shm with the request that owns them (29.09.).

F4 (``SGLANG_WEG2_ENABLE_D_PARK_END``) writes one END part per rank and
parked request into ``$SGLANG_HICACHE_ARENA_DIR/handoff`` -- tmpfs, ~64 MB per
request on NF-D (TP0 GDN state 57 MB, measured from P's END part in z30u).
Before this fix only the adopt verdict took them (``tail_adopt.agree`` ->
``remove_park``); P's census/prune (H63b) skips them by design. A parked rid
that never resumes on D kept its parts until the boot ended:

* aborted while parked, queued, held or running -- no resume, no verdict;
* finished or re-routed to P without an adopt verdict (staging voted 0).

What these cases pin:

* the abort reaches the park parts as it reaches the D park (``park_abort``,
  the scheduler's prefix match, ``abort_all`` too) -- with or without a
  parked list;
* at every D park each rank reaps ITS OWN park parts whose rid D no longer
  holds (running, parked, queued, settle, dormant hold, chunked); a live
  rid's parts, a peer rank's parts, P's parts and writes in flight stay;
* the switch off touches nothing.
"""

import contextlib
import os
from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import d_park_runtime as dpr
from sglang.srt.weg2 import tail_handoff as th

GONE, LIVE, QUEUED, HELD = "weg2-3-11", "weg2-4-12", "weg2-5-13", "weg2-6-14"


def _park_switch(on: bool):
    env = getattr(envs, "SGLANG_WEG2_ENABLE_D_PARK_END", None)
    return env.override(on) if env is not None else contextlib.nullcontext()


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_ADOPT.override(True), \
            envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.override(True), _park_switch(True):
        d = th._dir()
        os.makedirs(d, exist_ok=True)
        yield d


def _touch(d, name):
    with open(os.path.join(d, name), "wb") as f:
        f.write(b"x" * 16)


def _parts(d, rid, ranks=(0, 1, 2)):
    for r in ranks:
        for ext in ("json", "pt"):
            _touch(d, f"{rid}.tail.dpark{r}-7{r}.{ext}")


def _sched(tp_rank=1, **kw):
    base = dict(ps=SimpleNamespace(tp_rank=tp_rank, tp_size=3), page_size=64,
                server_args=SimpleNamespace(mamba_track_interval=64),
                req_to_token_pool=None, token_to_kv_pool_allocator=None,
                weg2_d_parked=[], waiting_queue=[], weg2_post_wake_settle=[],
                weg2_dormant_hold=[], chunked_req=None,
                enable_hicache_storage=False,
                ipc_channels=SimpleNamespace(send_to_tokenizer=SimpleNamespace(send_output=lambda *a: None)))
    base.update(kw)
    return SimpleNamespace(**base)


def _left(d):
    return sorted(os.listdir(d))


# ---------------------------------------------------------------- the abort
def test_an_abort_takes_the_park_parts_of_the_aborted_rid(arena):
    _parts(arena, GONE)
    _parts(arena, LIVE)
    _touch(arena, f"{GONE}.tail.pp0-1.json")  # P's part: not the park's business
    # the rid is not in the D park any more (resumed into the queue, held, ...)
    dpr.park_abort(_sched(), SimpleNamespace(rid=GONE, abort_all=False))
    left = _left(arena)
    assert not [f for f in left if f.startswith(GONE) and ".dpark" in f]
    assert f"{GONE}.tail.pp0-1.json" in left
    assert len([f for f in left if f.startswith(LIVE)]) == 6


def test_abort_all_takes_every_park_part(arena):
    _parts(arena, GONE)
    _parts(arena, LIVE)
    dpr.park_abort(_sched(), SimpleNamespace(rid="", abort_all=True))
    assert _left(arena) == []


def test_an_abort_of_a_parked_request_still_drops_it_and_its_parts(arena, monkeypatch):
    from sglang.srt.weg2 import d_park_draft

    monkeypatch.setattr(d_park_draft, "drop_all", lambda *a, **k: 0)
    _parts(arena, GONE)
    sched = _sched(weg2_d_parked=[SimpleNamespace(rid=GONE)])
    assert dpr.park_abort(sched, SimpleNamespace(rid=GONE, abort_all=False)) == 1
    assert sched.weg2_d_parked == [] and _left(arena) == []


# ---------------------------------------------------------- the reap at a park
def test_a_park_reaps_this_ranks_parts_of_rids_d_no_longer_holds(arena, monkeypatch):
    monkeypatch.setattr(th, "publish_park_end", lambda *a, **k: ("", None))
    for rid in (GONE, LIVE, QUEUED, HELD):
        _parts(arena, rid)
    _touch(arena, f"{GONE}.tail.dpark1-71.json.tmp")  # a write in flight is its writer's
    sched = _sched(tp_rank=1, waiting_queue=[SimpleNamespace(rid=QUEUED)],
                   weg2_dormant_hold=[SimpleNamespace(rid=HELD)])
    dpr._park_end(sched, [SimpleNamespace(rid=LIVE)])
    left = _left(arena)
    # GONE: this rank's (dpark1) parts reaped, the peers' stay for their own park
    assert f"{GONE}.tail.dpark1-71.json" not in left and f"{GONE}.tail.dpark1-71.pt" not in left
    assert f"{GONE}.tail.dpark0-70.json" in left and f"{GONE}.tail.dpark2-72.pt" in left
    assert f"{GONE}.tail.dpark1-71.json.tmp" in left
    for rid in (LIVE, QUEUED, HELD):
        assert len([f for f in left if f.startswith(rid)]) == 6, rid


def test_every_rank_reaps_its_own_so_the_group_leaves_nothing(arena, monkeypatch):
    monkeypatch.setattr(th, "publish_park_end", lambda *a, **k: ("", None))
    _parts(arena, GONE)
    for tp in (0, 1, 2):
        dpr._park_end(_sched(tp_rank=tp), [])
    assert _left(arena) == []


def test_switch_off_touches_nothing(arena):
    _parts(arena, GONE)
    with _park_switch(False):
        dpr._park_end(_sched(), [])
        dpr.park_abort(_sched(), SimpleNamespace(rid=GONE, abort_all=False))
    assert len(_left(arena)) == 6
