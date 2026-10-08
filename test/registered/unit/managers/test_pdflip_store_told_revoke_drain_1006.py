"""#1400 (NF cand4c 06.10. 11:08:10Z, PP2, rid pdflip-74-260): the told admission
gate waited ON the scheduler thread for a read that only the scheduler thread
could finish.

The boot: three ranks read ``ongoing_prefetch=1`` in HICACHE-ROUND 104381; PP0
and PP1 ran rounds 104383 / 104384 and drained to 0, PP2 never ran another
round. Its follower read of a cold 257k prompt (told 0) had been REVOKED by the
probe (hits below the threshold): a revoked operation is not terminated, its
record leaves ``ongoing_prefetch`` only when the scheduler thread drains the
revoke queue (once per round), and ``check_prefetch_progress`` answers False
until then. ``admission`` spun on that call for WAIT_CAP_S, twice (the early
settle loop broke out and the main loop waited the budget again): 11:06:10 ->
11:08:10, then ``PdFlipStoreToldMismatch #1400 STORE-TOLD WAIT EXCEEDED ... the
storage thread is stuck`` and the group died. The storage thread had long
answered.

Fix pinned here: the waits drain the revoke queue of the tree while they spin
(``drain_prefetch_revokes_uncoordinated``, only where the round drain is local
anyway), and a read that spends the budget in the early settle is named at
once instead of waiting a second budget.
"""

import time as _real_time
import types
from types import SimpleNamespace

import pytest

from flliper.srt.managers import pdflip_store_told as m


class _RevokedTree:
    """Tree double of a REVOKED follower read: ``check_prefetch_progress`` is
    False until the revoke queue was drained -- by the scheduler round, which
    the held scheduler thread never reaches, or by the wait itself."""

    def __init__(self, can_drain: bool = True):
        self.revoked = {}
        self.drain_calls = 0
        self.progress_calls = 0
        self.prefetch_loaded_tokens_by_reqid = {}
        self._completed = {}
        self._can_drain = can_drain
        if can_drain:
            self.drain_prefetch_revokes_uncoordinated = self._drain

    def revoke(self, rid):
        self.revoked[rid] = True

    def _drain(self):
        self.drain_calls += 1
        self.revoked.clear()
        return True

    def check_prefetch_progress(self, rid):
        self.progress_calls += 1
        return rid not in self.revoked

    def completed_prefetch_tokens(self, rid):
        return self._completed.get(rid)

    def pop_prefetch_loaded_tokens(self, rid):
        return self.prefetch_loaded_tokens_by_reqid.pop(rid, 0)


class _Sched:
    def __init__(self, tree, pp_rank=2):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=3, tp_size=1)
        self.enable_hicache_storage = True
        self.tree_cache = tree
        self.waiting_queue = []
        self.pp_flip_counters = None
        self.registered = []

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        self.registered.append((req.rid, limit_tokens))
        return "issued"


class _FakeClock:
    """``time`` of the module under test, advanced by its own sleeps: the
    budget arithmetic is exact and no test depends on machine load."""

    def __init__(self):
        self.now = 1000.0
        self.slept = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += s
        self.slept += s

    def __getattr__(self, name):
        return getattr(_real_time, name)


def _req(rid):
    return SimpleNamespace(rid=rid, prefetch_deferred=None)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(m.ENV_ARMED, raising=False)
    yield


@pytest.fixture
def clock(monkeypatch):
    c = _FakeClock()
    monkeypatch.setattr(m, "time", c)
    monkeypatch.setattr(m, "WAIT_CAP_S", 5.0)
    return c


def test_main_wait_drains_the_revoke_the_held_scheduler_never_reaches(clock):
    """The measured death, on the follower's own wait: told 0, the read was
    revoked, nothing but the wait itself can drain it."""
    tree = _RevokedTree()
    s = _Sched(tree)
    m.armed(s)
    r = _req("rev00001")
    m.intake(s, r, lambda k: None)
    m.follower_absorb(s, [m.PdFlipStoreTold("rev00001", 0)])
    tree.revoke("rev00001")
    assert m.admission(s, r, lambda k, rid: None) == 0
    assert tree.drain_calls >= 1
    assert clock.slept < 0.1  # one drain, not the budget


def test_early_settle_wait_drains_the_revoke_too(clock):
    """The first of the two waits of the measured boot (the follower's early
    read, ``_pdflip_early_told`` set when the told arrived)."""
    tree = _RevokedTree()
    s = _Sched(tree)
    m.armed(s)
    r = _req("rev00002")
    m.intake(s, r, lambda k: None)
    m.follower_absorb(s, [m.PdFlipStoreTold("rev00002", 0)])
    r._pdflip_early_told = 0
    tree.revoke("rev00002")
    assert m.admission(s, r, lambda k, rid: None) == 0
    assert tree.drain_calls >= 1
    assert clock.slept < 0.1


def test_a_tree_without_the_drain_keeps_the_unchanged_wait_and_cap(clock):
    """27B hi-mamba / doubles: no method, no drain, the wait ends at the cap
    with the named stop, as before."""
    tree = _RevokedTree(can_drain=False)
    s = _Sched(tree)
    m.armed(s)
    r = _req("rev00003")
    m.intake(s, r, lambda k: None)
    m.follower_absorb(s, [m.PdFlipStoreTold("rev00003", 0)])
    tree.revoke("rev00003")
    with pytest.raises(m.PdFlipStoreToldMismatch, match="WAIT EXCEEDED"):
        m.admission(s, r, lambda k, rid: None)
    assert tree.drain_calls == 0
    assert 5.0 <= clock.slept < 5.1


def test_a_read_that_never_terminates_costs_one_budget_not_two(clock):
    """11:06:10 -> 11:08:10 was two budgets for ONE read: the early settle loop
    `break`-ed on the cap and the main loop waited the cap again. The early
    wait names the stuck read itself now."""

    class _NeverDone(_RevokedTree):
        def check_prefetch_progress(self, rid):
            self.progress_calls += 1
            return False

    tree = _NeverDone()
    s = _Sched(tree)
    m.armed(s)
    r = _req("rev00004")
    m.intake(s, r, lambda k: None)
    m.follower_absorb(s, [m.PdFlipStoreTold("rev00004", 0)])
    r._pdflip_early_told = 0
    with pytest.raises(m.PdFlipStoreToldMismatch, match="WAIT EXCEEDED"):
        m.admission(s, r, lambda k, rid: None)
    assert 5.0 <= clock.slept < 5.1, clock.slept  # base: ~10.0 (two budgets)


def _fake_cache(collective: bool):
    calls = []
    fake = types.SimpleNamespace(
        _drain_agreement_is_collective=lambda: collective,
        _drain_storage_control_queues_impl=lambda **kw: calls.append(kw),
    )
    return fake, calls


def test_tree_drains_only_the_revoke_queue_where_the_round_drain_is_local():
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    fake, calls = _fake_cache(collective=False)
    assert UnifiedRadixCache.drain_prefetch_revokes_uncoordinated(fake) is True
    assert calls == [
        dict(n_revoke=None, n_backup=0, n_release=0,
             extra_release_counts=None, log_metrics=False)
    ]


def test_tree_does_nothing_where_the_drain_needs_an_agreement():
    """An attention group > 1 drains by the group's MIN of the queue sizes; a
    rank-local drain inside a wait would break that order."""
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    fake, calls = _fake_cache(collective=True)
    assert UnifiedRadixCache.drain_prefetch_revokes_uncoordinated(fake) is False
    assert calls == []
