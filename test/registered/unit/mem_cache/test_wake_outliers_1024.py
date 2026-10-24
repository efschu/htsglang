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
import tempfile
import threading
import time
import unittest
from unittest import mock

import pytest

from flliper.srt.mem_cache.storage.file import store_journal as SJ
from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor, _is_component_stem

SHARED = "_K"
PAGE = 4096
ENV_PEERS = "FLLIPER_HICACHE_L3_JOURNAL_TOUCH_TP_PEERS"


def _ev(d, tp=0, pp=0, cp=0, group="D", shared=True):
    def _iter_existing():
        for name in sorted(os.listdir(d)):
            if name.endswith(".bin"):
                yield name[:-4], os.stat(os.path.join(d, name))

    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_GROUP": group}):
        return LRUFileEvictor(d, SHARED, tp_rank=tp, pp_rank=pp, attn_cp_rank=cp,
                              writes_shared_keys=shared, scan_suffixes=(SHARED,),
                              extra_config={"max_size": str(10 ** 9), "max_size_scope": "shared"},
                              writer_count=1, iter_existing=_iter_existing)


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
    def test_component_tags_are_the_eviction_instruments_tags(self):
        assert all(_is_component_stem(s) for s in COMP)
        assert not any(_is_component_stem(s) for s in KV)

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
    th = threading.Thread(target=lambda: out.setdefault("h", W.wait("store_rescan")))
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
