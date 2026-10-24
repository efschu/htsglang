"""WAKE-OUTLIERS-1024 (Auftrag #209): tok/s outliers right after a P/D wake on the 27B line.

Boot ``boot_weg2_dkr27bbar1drei10240803_b9cddabda1_1024_080355`` (int8, P = PP3, D = TP3):

(1) P PP0, first forwards after the wake: PP0's ``#PGAP launch`` 45-133 ms instead of 3 ms (fi_plan 7-101 ms), the
    rank gpu-ms 85-124 ms vs PP1/PP2 54-60 ms -- exactly in the window between ``PDFLIP-WAKE-DEFER ... released_by=
    extends=2`` and ``PDFLIP-STORE-RESCAN at wake: ... in N ms (mode journal)`` (08:11:08-12, 08:17:06-07, 08:18:31-33,
    08:18:52-53). The journal apply parsed 0.1-3.4 M lines per wake, 0.4-6.8 s, 95 % of them ``T`` lines
    ('read-recency: 3195071 page(s)'), and every D TP rank journals the SAME page: PDFLIP-READ-STAGES summed
    8776008 pages on TP0, TP1 and TP2 alike. Fix: a pure TP peer of the owner leaves the canonical KV page's T line
    to the owner (component pages and PP stages keep theirs).

(2) D, first extend after the wake: 08:48:22 TP0 compute 2879 ms for 244 tokens, TP1/TP2 waiting 3.3-3.6 s;
    'PDFLIP-WAKE-DEFER who=store_rescan waited_ms=4026 ... released_by=bound extends=0' at 08:48:21 -- the bound ran
    out while the first extend was still launching and the journal apply (690 ms, TP0 = the owner) ran inside it.
    Fix: a bound that expires inside a launch waits for that launch to return (capped).
"""

import inspect
import os
import signal
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import pytest

from flliper.srt.mem_cache.storage.file import store_journal as SJ
from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

SHARED = "_K"
PAGE = 4096
ENV_PEERS = "FLLIPER_HICACHE_L3_JOURNAL_TOUCH_TP_PEERS"
HARD_LIMIT_S = 20


@pytest.fixture(autouse=True)
def _hard_limit():
    """Review 1024 MUSS 3: every test of this file is hard-bounded (no pytest-timeout in the desk image) -- a hung
    case (a mutant's never-ending launch) fails here instead of keeping a remote desk container alive."""
    def _boom(signum, frame):
        raise TimeoutError(f"hard test limit {HARD_LIMIT_S} s")

    old = signal.signal(signal.SIGALRM, _boom)
    signal.alarm(HARD_LIMIT_S)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _ev(d, tp=0, pp=0, cp=0, group="D", shared=True, **kw):
    def _iter_existing():
        for name in sorted(os.listdir(d)):
            if name.endswith(".bin"):
                yield name[:-4], os.stat(os.path.join(d, name))

    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_GROUP": group}):
        return LRUFileEvictor(d, SHARED, tp_rank=tp, pp_rank=pp, attn_cp_rank=cp,
                              writes_shared_keys=shared, scan_suffixes=(SHARED,),
                              extra_config={"max_size": str(10 ** 9), "max_size_scope": "shared"},
                              writer_count=1, iter_existing=_iter_existing, **kw)


def _t_lines(ev):
    path = ev._journal.path
    with open(path, "rb") as f:
        recs, _ = SJ.parse_lines(f.read())
    return [r[3] for r in recs if r[1] == "T"]


class _Env(unittest.TestCase):
    KEYS = (SJ.ENV, "FLLIPER_PDFLIP_L3_EPOCH", "FLLIPER_PDFLIP_GROUP", "FLLIPER_HICACHE_L3_JOURNAL_TOUCH",
            "FLLIPER_HICACHE_L3_JOURNAL_TOUCH_S", ENV_PEERS)

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in self.KEYS}
        self._td = tempfile.TemporaryDirectory()
        self.d = self._td.name
        os.environ["FLLIPER_PDFLIP_L3_EPOCH"] = "100"

    def tearDown(self):
        self._td.cleanup()
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def _touch(self, ev, stems):
        for s in stems:
            ev.touch(s, os.path.join(self.d, s + ".bin"), mtime=False)


KV = [f"k{i:03d}{SHARED}" for i in range(6)]
COMP = ["m000.mamba" + SHARED, "q000.qsa_indexer" + SHARED, "d000.draft" + SHARED]


class TestTouchLinesOfTpPeers(_Env):
    def test_only_the_group_wide_kv_page_qualifies(self):
        peer = _ev(self.d, tp=1)
        assert all(peer._is_group_wide_kv_key(s) for s in KV)
        assert not any(peer._is_group_wide_kv_key(s) for s in COMP)
        assert not peer._is_group_wide_kv_key("k000" + SHARED + "_1_3"), "a per-rank suffix is never group-wide"
        assert not peer._is_group_wide_kv_key("a000.swa" + SHARED + "_1_3")

    def test_review_muss2_structural_gates_keep_every_line(self):
        """dcp owner form (#239: the host may abstain from a read its peers do), DP attention (disjoint requests) and
        a phase-changing rank identity (Weg 1) -- the peer keeps its KV line."""
        for kw in ({"dcp_owner_mode": True}, {"dp_attention": True}, {"rank_identity_stable": False}):
            peer = _ev(self.d, tp=1, **kw)
            self._touch(peer, KV)
            assert _t_lines(peer) == KV, kw

    def test_live_suffix_from_the_backend_decides(self):
        """The backend re-derives its KV suffix when the canonical page is installed after construction: the evictor
        asks it at every touch. None = no key is group-wide (rank term in the suffix / dcp owner form)."""
        cur = {"s": None}
        peer = _ev(self.d, tp=1, touch_dedupe_suffix=lambda: cur["s"])
        self._touch(peer, KV[:2])
        cur["s"] = SHARED
        self._touch(peer, KV[2:4])
        cur["s"] = "_model_1_3"
        self._touch(peer, KV[4:])
        assert _t_lines(peer) == KV[:2] + KV[4:]

    def test_tp_peer_leaves_the_kv_page_line_to_the_owner(self):
        owner, peer1, peer2 = _ev(self.d, tp=0), _ev(self.d, tp=1), _ev(self.d, tp=2)
        for ev in (owner, peer1, peer2):
            self._touch(ev, KV)
        assert _t_lines(owner) == KV, "the owner journals every read hit"
        assert _t_lines(peer1) == [] and _t_lines(peer2) == [], "a TP peer's KV line is the owner's duplicate"

    def test_tp_peer_keeps_component_page_lines(self):
        peer = _ev(self.d, tp=1)
        self._touch(peer, COMP + KV)
        assert _t_lines(peer) == COMP, "component pages may be rank-specific: their line stays"

    def test_pp_stages_and_cp_ranks_keep_their_lines(self):
        # PP stages read DIFFERENT page sets (P READ-STAGES: PP0 2008189, PP1 2351133, PP2 2286285 pages)
        pp1, pp2_tp1, cp1 = _ev(self.d, pp=1, group="P"), _ev(self.d, tp=1, pp=1, group="P"), _ev(self.d, tp=1, cp=1)
        for ev in (pp1, pp2_tp1, cp1):
            self._touch(ev, KV)
            assert _t_lines(ev) == KV, (ev._journal.path, "only a pure TP peer of the owner is deduplicated")

    def test_switch_restores_every_rank(self):
        os.environ[ENV_PEERS] = "1"
        peer = _ev(self.d, tp=1)
        self._touch(peer, KV)
        assert _t_lines(peer) == KV

    def test_owner_wake_apply_sees_the_same_recency_without_the_peer_lines(self):
        """The other group's owner (P PP0) applies D's journals at its wake: with the TP peer's duplicate lines
        gone the touched pages still reach MRU -- D TP0's own line carries each page."""
        reader = _ev(self.d, group="P")             # P PP0: the other group's owner, reads D's journals
        d0, d1 = _ev(self.d, tp=0), _ev(self.d, tp=1)
        for s in KV:
            assert reader.reserve(s, PAGE, key=s, owner_writes_whole_file=False)
            with open(os.path.join(self.d, s + ".bin"), "wb") as f:
                f.write(b"\x07" * PAGE)
            reader.commit(s)
        reader.rescan()
        order_before = list(reader._lru)
        self._touch(d0, list(reversed(KV[:3])))
        self._touch(d1, list(reversed(KV[:3])))
        reader.rescan()
        assert list(reader._lru)[-3:] == list(reversed(KV[:3])), (order_before, list(reader._lru))
        assert len(_t_lines(d1)) == 0


# ------------------------------------------------------------------------------------------------ (2) wake deferral
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


def test_bound_inside_a_launch_waits_for_the_launch_to_return(wcd, monkeypatch):
    monkeypatch.setenv(wcd.ENV_MAX_MS, "50")
    assert wcd.arm(has_work=True)
    wcd.launch_begin()                      # the wake's first extend is launching (host-paced, 3 s on metal)
    out = {}
    th = _waiter(wcd, out)
    time.sleep(0.25)
    assert th.is_alive() and wcd.armed(), "the bound ran out INSIDE the launch: the journal apply must not start"
    wcd.launch_end()
    th.join(2.0)
    assert not th.is_alive() and wcd.state()["why"] == "bound+launch_end"
    assert 0.2 <= out["h"] < 2.0


def test_bound_outside_a_launch_releases_as_before(wcd, monkeypatch):
    monkeypatch.setenv(wcd.ENV_MAX_MS, "50")
    wcd.arm(has_work=True)
    t0 = time.monotonic()
    held = wcd.wait("x")
    assert 0.04 <= held < 1.0 and time.monotonic() - t0 < 1.0 and wcd.state()["why"] == "bound"


def test_a_launch_that_never_returns_is_capped(wcd, monkeypatch):
    monkeypatch.setenv(wcd.ENV_MAX_MS, "50")
    monkeypatch.setenv(wcd.ENV_LAUNCH_MAX_MS, "150")
    wcd.arm(has_work=True)
    wcd.launch_begin()
    t0 = time.monotonic()
    wcd.wait("x")
    dt = time.monotonic() - t0
    assert 0.15 <= dt < 1.5 and wcd.state()["why"] == "bound+launch_cap", "never a gate"


def test_launch_hooks_are_inert_while_not_armed(wcd):
    wcd.launch_begin()
    assert wcd.state()["in_launch"] is False
    wcd.launch_end()
    assert wcd.wait("x") == 0.0


def test_extends_and_the_release_leg_still_release_during_a_launch(wcd, monkeypatch):
    monkeypatch.setenv(wcd.ENV_MAX_MS, "5000")
    wcd.arm(has_work=True)
    wcd.launch_begin()
    out = {}
    th = _waiter(wcd, out)
    time.sleep(0.05)
    wcd.release("join:release")            # the release leg's join never waits for the deferral
    th.join(2.0)
    assert not th.is_alive() and wcd.state()["why"] == "join:release"
    wcd.launch_end()


def test_a_release_while_the_bound_waits_for_the_launch_wins(wcd, monkeypatch):
    monkeypatch.setenv(wcd.ENV_MAX_MS, "30")
    wcd.arm(has_work=True)
    wcd.launch_begin()
    out = {}
    th = _waiter(wcd, out)
    time.sleep(0.15)                        # past the bound, now waiting for the launch
    assert th.is_alive()
    wcd.release("join:release")
    th.join(2.0)
    assert not th.is_alive() and wcd.state()["why"] == "join:release"
    wcd.launch_end()


def test_run_batch_brackets_the_forward_with_the_launch_hooks():
    from flliper.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler.run_batch)
    i_begin = src.index("_pdflip_wake_cpu_defer.launch_begin()")
    i_fwd = src.index("self._run_batch_forward(batch, pp_proxy_tensors)")
    i_fin = src.index("finally:", i_fwd)
    i_end = src.index("_pdflip_wake_cpu_defer.launch_end()", i_fin)
    assert i_begin < i_fwd < i_fin < i_end, "begin before the forward, end in its finally (also when it raises)"


# ------------------------------------------------------------------------------------------------ (3) park demote
def test_park_demote_waits_while_the_wake_deferral_is_armed(wcd, monkeypatch):
    from flliper.srt.pdflip import park_demote as PD

    calls = []
    monkeypatch.setattr(PD, "demote_once", lambda backend, pools, **kw: calls.append(time.monotonic()))
    monkeypatch.setenv(wcd.ENV_MAX_MS, "60000")
    assert wcd.arm(has_work=True)
    d = PD._Demoter(object(), lambda: [], 0.01)
    d.thread.start()
    try:
        time.sleep(0.2)
        assert calls == [], "no copy beside the wake's first launches"
        t_rel = time.monotonic()
        wcd.release("extends=2")
        deadline = time.monotonic() + 2.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls and calls[0] >= t_rel
    finally:
        d.stop.set()
        d.thread.join(2.0)


def test_park_demote_runs_when_no_wake_is_deferred(wcd, monkeypatch):
    from flliper.srt.pdflip import park_demote as PD

    calls = []
    monkeypatch.setattr(PD, "demote_once", lambda backend, pools, **kw: calls.append(1))
    d = PD._Demoter(object(), lambda: [], 0.01)
    d.thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while len(calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(calls) >= 3
    finally:
        d.stop.set()
        d.thread.join(2.0)


def test_hicache_file_hands_the_evictor_the_structural_facts_and_the_live_suffix():
    from flliper.srt.mem_cache import hicache_storage as H

    src = inspect.getsource(H.HiCacheFile.__init__)
    for frag in ("dcp_owner_mode=self.dcp_owner_mode", "dp_attention=_dp_attention_on()",
                 'rank_identity_stable=self._l3p_group() in ("P", "D")', "touch_dedupe_suffix=self._touch_dedupe_suffix"):
        assert frag in src, frag
    fake = types.SimpleNamespace(dcp_owner_mode=False, _kv_config_suffix_is_group_wide=True, kv_config_suffix="_m_h")
    assert H.HiCacheFile._touch_dedupe_suffix(fake) == "_m_h"
    fake.dcp_owner_mode = True
    assert H.HiCacheFile._touch_dedupe_suffix(fake) is None
    fake.dcp_owner_mode, fake._kv_config_suffix_is_group_wide = False, False
    assert H.HiCacheFile._touch_dedupe_suffix(fake) is None


def test_park_demote_own_bound_is_the_deferral_bound_in_ticks(wcd, monkeypatch):
    from flliper.srt.pdflip import park_demote as PD

    monkeypatch.setenv(wcd.ENV_MAX_MS, "4000")
    monkeypatch.setenv(wcd.ENV_LAUNCH_MAX_MS, "8000")
    assert PD._max_skipped_ticks(1.0) == 13, "ceil((4000 + 8000) ms / 1 s) + 1"
