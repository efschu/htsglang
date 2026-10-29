"""Adversarial review of WAKE-OUTLIERS-1024 (desk/27b-wake-outliers-1024 @ 2ab3bed847).

Every test here is hard-bounded: an autouse SIGALRM fixture fails a test after 20 s, and every helper thread is a
daemon joined with a timeout -- a hung case can never keep a remote desk container alive (the author's M5 left one
waiting up to 3600 s).

EXPECTED ON 2ab3bed847 (the reviewed head):
  * RED  test_tp_peer_keeps_rank_suffixed_component_line[*]  -- MUSS 1: ``_COMPONENT_STEM_TAGS`` lists only
         .qsa_indexer/.mamba/.draft; .swa/.indexer/.deepseek_v4_* keys take ``config_suffix`` (tp-rank-suffixed,
         hicache_storage._suffix_for_key fall-through), i.e. they are PER-RANK files. F1 drops the TP peer's T line for
         ITS OWN file, which no other rank journals -- the owner's LRU keeps it at write time and evicts it first.
  * RED  test_park_demote_pause_is_self_bounded               -- SOLL: the demoter's pause has no bound of its own;
         it ends only through wake_cpu_defer's release (whose bound fires only inside a waiter's wait()).
  * RED  test_bound_waits_for_the_running_launch_not_the_next  -- SOLL: a launch that starts while the waiter has not
         yet seen the previous launch_end extends the hold (in_launch is a flag, not a generation).
  * GREEN the rest (controls and the exception path of run_batch).
"""

import inspect
import os
import signal
import tempfile
import threading
import time
import types
from unittest import mock

import pytest

from flliper.srt.mem_cache.storage.file import store_journal as SJ
from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

SHARED = "_K"
ENV_PEERS = "FLLIPER_HICACHE_L3_JOURNAL_TOUCH_TP_PEERS"
HARD_LIMIT_S = 20


@pytest.fixture(autouse=True)
def _hard_limit():
    def _boom(signum, frame):
        raise TimeoutError(f"hard test limit {HARD_LIMIT_S} s")

    old = signal.signal(signal.SIGALRM, _boom)
    signal.alarm(HARD_LIMIT_S)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.fixture
def store(monkeypatch):
    for k in (SJ.ENV, "FLLIPER_PDFLIP_GROUP", "FLLIPER_HICACHE_L3_JOURNAL_TOUCH",
              "FLLIPER_HICACHE_L3_JOURNAL_TOUCH_S", ENV_PEERS):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_EPOCH", "100")
    with tempfile.TemporaryDirectory() as d:
        yield d


def _ev(d, tp=0, pp=0, cp=0, group="D"):
    def _iter_existing():
        for name in sorted(os.listdir(d)):
            if name.endswith(".bin"):
                yield name[:-4], os.stat(os.path.join(d, name))

    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_GROUP": group}):
        return LRUFileEvictor(d, SHARED, tp_rank=tp, pp_rank=pp, attn_cp_rank=cp,
                              writes_shared_keys=True, scan_suffixes=(SHARED,),
                              extra_config={"max_size": str(10 ** 9), "max_size_scope": "shared"},
                              writer_count=1, iter_existing=_iter_existing)


def _t_lines(ev):
    with open(ev._journal.path, "rb") as f:
        recs, _ = SJ.parse_lines(f.read())
    return [r[3] for r in recs if r[1] == "T"]


# ------------------------------------------------------------------------------------------- F1: per-rank component keys
# A per-rank component key carries config_suffix = _{model}[_{hash}]_{tp_rank}_{tp_size}[...]: rank 1 of 3 names its
# OWN file. Nobody else reads or journals that file.
@pytest.mark.parametrize("pool", ["swa", "indexer", "deepseek_v4_c4", "deepseek_v4_c128_state"])
def test_tp_peer_keeps_rank_suffixed_component_line(store, pool):
    peer = _ev(store, tp=1)
    own_file = f"a000.{pool}{SHARED}_1_3"
    peer.touch(own_file, os.path.join(store, own_file + ".bin"), mtime=False)
    assert _t_lines(peer) == [own_file], (
        f"TP1's own per-rank .{pool} file lost its read-recency line: the owner never reads or journals that file")


def test_control_plain_kv_page_of_a_peer_is_still_deduplicated(store):
    peer = _ev(store, tp=1)
    kv = f"b000{SHARED}"
    peer.touch(kv, os.path.join(store, kv + ".bin"), mtime=False)
    assert _t_lines(peer) == []


# ------------------------------------------------------------------------------------------------ F2: launch hooks
@pytest.fixture
def wcd(monkeypatch):
    from flliper.srt.pdflip import wake_cpu_defer as W

    for k in (W.ENV, W.ENV_MAX_MS, W.ENV_EXTENDS, W.ENV_LAUNCH_MAX_MS):
        monkeypatch.delenv(k, raising=False)
    W._reset_for_tests()
    yield W
    W._reset_for_tests()


def _waiter(W, out):
    th = threading.Thread(target=lambda: out.setdefault("h", W.wait("store_rescan")), daemon=True)
    th.start()
    return th


def test_run_batch_raising_still_ends_the_launch(wcd, monkeypatch):
    """The finally path for real: _run_batch_forward raises, in_launch must be cleared, and a later bound releases
    as plain ``bound`` (no 8 s hold left behind)."""
    from flliper.srt.managers import scheduler as S

    raw = inspect.unwrap(S.Scheduler.run_batch)
    monkeypatch.setattr(S._prefix_lens_check, "has_deferred", lambda: False)

    def _boom(batch, pp_proxy_tensors=None):
        raise RuntimeError("forward failed")

    fake = types.SimpleNamespace(_run_batch_forward=_boom)
    monkeypatch.setenv(wcd.ENV_MAX_MS, "50")
    assert wcd.arm(has_work=True)
    with pytest.raises(RuntimeError, match="forward failed"):
        raw(fake, types.SimpleNamespace(mixed_verify_decode=None))
    assert wcd.state()["in_launch"] is False
    t0 = time.monotonic()
    wcd.wait("x")
    assert time.monotonic() - t0 < 1.0 and wcd.state()["why"] == "bound"


def test_bound_waits_for_the_running_launch_not_the_next(wcd, monkeypatch):
    monkeypatch.setenv(wcd.ENV_MAX_MS, "30")
    monkeypatch.setenv(wcd.ENV_LAUNCH_MAX_MS, "3000")
    assert wcd.arm(has_work=True)
    wcd.launch_begin()
    out = {}
    th = _waiter(wcd, out)
    time.sleep(0.15)                       # bound expired, the waiter now waits for THIS launch
    assert th.is_alive()
    wcd.launch_end()                       # the launch returned ...
    wcd.launch_begin()                     # ... and the next one starts before the waiter got the GIL
    th.join(0.5)
    released = not th.is_alive()
    wcd.launch_end()
    th.join(4.0)
    assert released, "the running launch ended 0.5 s ago; the deferral still held for the NEXT launch"


def test_bound_releases_at_a_realistic_gap_between_launches(wcd, monkeypatch):
    monkeypatch.setenv(wcd.ENV_MAX_MS, "30")
    monkeypatch.setenv(wcd.ENV_LAUNCH_MAX_MS, "3000")
    assert wcd.arm(has_work=True)
    wcd.launch_begin()
    out = {}
    th = _waiter(wcd, out)
    time.sleep(0.15)
    wcd.launch_end()
    time.sleep(0.02)                       # result processing / next schedule between two run_batch calls
    wcd.launch_begin()
    th.join(0.5)
    released = not th.is_alive()
    wcd.launch_end()
    th.join(4.0)
    assert released and wcd.state()["why"] == "bound+launch_end"


# ------------------------------------------------------------------------------------------------ F3: park demote
def test_park_demote_pause_is_self_bounded(wcd, monkeypatch):
    """armed() without a waiter (the rescan thread died before wait(), or never reached it): the demote pause must
    end on its own within the deferral's own bound (MAX_MS + LAUNCH_MAX_MS), not at the next release leg."""
    from flliper.srt.pdflip import park_demote as PD

    calls = []
    monkeypatch.setattr(PD, "demote_once", lambda backend, pools, **kw: calls.append(time.monotonic()))
    monkeypatch.setenv(wcd.ENV_MAX_MS, "100")
    monkeypatch.setenv(wcd.ENV_LAUNCH_MAX_MS, "100")
    assert wcd.arm(has_work=True)
    d = PD._Demoter(object(), lambda: [], 0.01)
    d.thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls, "demote still paused 2 s after arm with a 0.2 s deferral bound: the pause has no bound"
    finally:
        d.stop.set()
        d.thread.join(2.0)


def test_control_park_demote_resumes_after_release(wcd, monkeypatch):
    from flliper.srt.pdflip import park_demote as PD

    calls = []
    monkeypatch.setattr(PD, "demote_once", lambda backend, pools, **kw: calls.append(1))
    monkeypatch.setenv(wcd.ENV_MAX_MS, "100")
    assert wcd.arm(has_work=True)
    out = {}
    th = _waiter(wcd, out)                 # the real path: the rescan thread waits, the bound releases
    d = PD._Demoter(object(), lambda: [], 0.01)
    d.thread.start()
    try:
        th.join(2.0)
        deadline = time.monotonic() + 2.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls and wcd.state()["why"] == "bound"
    finally:
        d.stop.set()
        d.thread.join(2.0)
