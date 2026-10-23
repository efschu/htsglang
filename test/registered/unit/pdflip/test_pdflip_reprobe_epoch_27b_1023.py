"""REPROBE-1023: D's store arm re-asks a short answer on the GROUP's schedule, not in every round.

Metal 27B AutoRound, boot ``..._a45112428e_1023_144422.D.log`` (D = TP2), 14:49:15-35Z: the lifetime-refused
hand-backs pdflip-1-3 (N=118937) and pdflip-1-4 (N=107847) were asked of the store in every round by
``Scheduler._pdflip_local_store_matches`` (#1317 X-STORE-VOTE, S64 fresh arm): spans 118709 / 107619 (anchor
depth minus the 227-token host match), ``readable=0``, ``S64 STORE-ARM COST probes=64 ms_mean=210.9``,
``D-ADMIT-AFTER-REFUSALS refusals=37 gates={lifetime:37}``, STALE 0 in the whole boot.

The tests drive the REAL vote (``Scheduler._pdflip_local_store_matches``) on two rank stand-ins, MIN-reduce
the store votes and the store-epoch slot the way the packed head-vote reduce does, and hand the reduced values to
``store_reprobe_epoch.after_reduce`` on both ranks -- one scheduler pass per loop step.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import scheduler as sch  # noqa: E402
from flliper.srt.pdflip import store_reprobe_epoch as rpe  # noqa: E402

HOST_MATCH = 227
RIDS = {"pdflip-1-3": 118937, "pdflip-1-4": 107847}
ROUNDS = 37  # D-ADMIT-AFTER-REFUSALS refusals=37


class _Probe:
    def __init__(self, pages=0):
        self.pages = pages
        self.calls = []  # (pass, span) per probe

    def __call__(self, tokens, last_hash, prefix_keys=None, **_kw):
        self.calls.append((self.pass_no, len(tokens)))
        return self.pages


def _rank():
    probe = _Probe(0)
    probe.pass_no = 0
    reqs = []
    for rid, n in RIDS.items():
        reqs.append(SimpleNamespace(
            rid=rid, full_untruncated_fill_ids=list(range(n)), prefix_indices=[], host_hit_length=HOST_MATCH,
            last_host_node=None, extra_key=None, _compute_max_prefix_len=lambda k: k - 1,
        ))
    tree = SimpleNamespace(cache_controller=SimpleNamespace(store_presence_pages=probe), page_size=1,
                           is_eagle=False, hicache_storage_pass_prefix_keys=False)
    return SimpleNamespace(tree_cache=tree, waiting_queue=reqs), probe


def _canonical():
    return sorted(RIDS)


def _pass(ranks, pass_no, epoch_of=None):
    """One scheduler pass on every rank: vote, MIN-reduce (store arm + epoch slot), read back on every rank.
    ``epoch_of`` (mutant hook): what each rank hands to ``after_reduce`` as the epoch -- default the group MIN."""
    canonical = _canonical()
    votes, epochs = [], []
    for s, probe in ranks:
        probe.pass_no = pass_no
        v = sch.Scheduler._pdflip_local_store_matches(s, canonical)
        votes.append([int(v.get(r, -1)) for r in canonical])
        epochs.append(rpe.build_epoch_payload(s)[0])
    group = [min(col) for col in zip(*votes)]
    g_epoch = min(epochs)
    for i, (s, _p) in enumerate(ranks):
        rpe.after_reduce(s, canonical, group, g_epoch if epoch_of is None else epoch_of(i, epochs))
    return group


def _asked_passes(probe):
    return sorted({p for p, _n in probe.calls})


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(rpe.ENV, raising=False)
    monkeypatch.delenv(rpe.MAX_ENV, raising=False)
    monkeypatch.delenv(sch.S64_STORE_ARM_FRESH_ENV, raising=False)
    rpe.reset_stats()
    yield


def test_repro_switch_off_asks_every_round_like_boot_144422(monkeypatch):
    """RED shape (the boot): every round asks both spans on both ranks -- N probes per rid and rank."""
    monkeypatch.setenv(rpe.ENV, "0")
    ranks = [_rank(), _rank()]
    for k in range(ROUNDS):
        _pass(ranks, k)
    for _s, probe in ranks:
        assert len(probe.calls) == 2 * ROUNDS
        assert sorted({n for _p, n in probe.calls}) == [107846 - HOST_MATCH, 118936 - HOST_MATCH]  # 107619/118709


def test_repro_on_refused_handbacks_are_asked_on_the_backoff_only_and_identically():
    ranks = [_rank(), _rank()]
    for k in range(ROUNDS):
        group = _pass(ranks, k)
        assert group == [HOST_MATCH, HOST_MATCH]  # the vote is unchanged: tree match, the store holds nothing
    asked = [_asked_passes(p) for _s, p in ranks]
    # first vote, the first scheduled re-ask, then 2, 4, 8, 16 passes apart
    assert asked[0] == [0, 1, 3, 7, 15, 31]
    assert asked[0] == asked[1]  # R11: both ranks ask on the same passes
    for _s, probe in ranks:
        assert len(probe.calls) == 2 * 6  # was 2 * 37
    st = rpe.stats()
    assert st["held"] > 0 and st["asked"] > 0


def test_epoch_moved_on_all_ranks_asks_next_pass_on_both():
    ranks = [_rank(), _rank()]
    for k in range(10):
        _pass(ranks, k)
    for s, _p in ranks:
        rpe.note_store_write(s.tree_cache, 2)  # e.g. pdflip-1-2's turn-end L2 write acked on both ranks
    _pass(ranks, 10)  # the reduce that carries the moved epoch
    _pass(ranks, 11)
    for _s, p in ranks:
        assert 11 in _asked_passes(p)
        assert 10 not in _asked_passes(p)


def test_epoch_moved_on_one_rank_only_asks_on_no_rank():
    """MIN semantics: the group epoch moves when it moved on ALL ranks -- a lone rank never asks alone."""
    ranks = [_rank(), _rank()]
    for k in range(10):
        _pass(ranks, k)
    rpe.note_store_write(ranks[0][0].tree_cache, 5)
    for k in range(10, 15):
        _pass(ranks, k)
    assert _asked_passes(ranks[0][1]) == _asked_passes(ranks[1][1])
    assert not [p for p in _asked_passes(ranks[0][1]) if 10 <= p < 15]


def test_store_filling_is_followed_every_pass_s64_kept():
    """S64's purpose stays: a group vote that changes re-asks on the next pass."""
    ranks = [_rank(), _rank()]
    for k in range(8):
        _pass(ranks, k)
    for _s, p in ranks:
        p.pages = 50000  # the store grew on both ranks without an epoch D saw (group P wrote)
    seen = []
    for k in range(8, 40):
        seen.append(_pass(ranks, k)[0])
    # held until the back-off bound asks (pass 15), then the changed group vote re-asks on the next pass
    assert seen[15 - 8] == HOST_MATCH + 50000
    assert all(v == HOST_MATCH for v in seen[: 15 - 8])
    assert 16 in _asked_passes(ranks[0][1])


def test_backoff_is_bounded_by_max_passes(monkeypatch):
    monkeypatch.setenv(rpe.MAX_ENV, "4")
    ranks = [_rank(), _rank()]
    for k in range(30):
        _pass(ranks, k)
    assert _asked_passes(ranks[0][1]) == [0, 1, 3, 7, 11, 15, 19, 23, 27]


def test_mutant_rank_local_epoch_splits_the_vote():
    """DANGER DIRECTION, shown on the harness: a rank that re-schedules on its OWN epoch (not the group MIN) asks
    while its peer holds -- the votes of one pass are then one fresh, one stale. The production form of this
    mutant (``after_reduce`` taking ``local_epoch``) turns ``test_epoch_moved_on_one_rank_only_asks_on_no_rank``
    RED (mutant run M1 in the report)."""
    ranks = [_rank(), _rank()]
    for k in range(10):
        _pass(ranks, k)
    rpe.note_store_write(ranks[0][0].tree_cache, 1)  # only rank 0 saw a write land
    for k in range(10, 14):
        _pass(ranks, k, epoch_of=lambda i, epochs: epochs[i])  # the mutant: rank-local epoch
    a0, a1 = _asked_passes(ranks[0][1]), _asked_passes(ranks[1][1])
    assert a0 != a1  # the split the real (MIN) path cannot produce -- see test_epoch_moved_on_one_rank_only


def test_payload_is_one_slot_on_every_rank():
    s, _p = _rank()
    assert rpe.build_epoch_payload(s) == [0]
    rpe.note_store_write(s.tree_cache, 3)
    assert rpe.build_epoch_payload(s) == [3]
    assert rpe.build_epoch_payload(SimpleNamespace()) == [0]  # no tree: still one slot


def test_default_on_and_switch():
    assert rpe.enabled({}) and not rpe.enabled({rpe.ENV: "0"})
    assert rpe.max_passes({}) == rpe.DEFAULT_MAX_PASSES and rpe.max_passes({rpe.MAX_ENV: "x"}) == 32


def test_no_reduce_no_schedule_asks_every_vote():
    """A vote that never went through a reduce (no schedule) asks as S64 did -- the S64 desk tests keep."""
    s, probe = _rank()
    for k in range(3):
        probe.pass_no = k
        sch.Scheduler._pdflip_local_store_matches(s, _canonical())
    assert len(probe.calls) == 6


def test_scheduler_reads_the_reduced_epoch_slot_not_the_local_one():
    """The read-back hands ``after_reduce`` the REDUCED slot of the packed MIN reduce (``t[...]``), never this
    rank's own epoch -- the production form of the mutant above. Layout pinned the way #1317 pins W86."""
    import inspect

    src = inspect.getsource(sch.Scheduler._update_uniform_pool_budget)
    assert "_xstore_epoch_at = len(vals)" in src
    assert "_store_reprobe_epoch.build_epoch_payload(self)" in src
    assert "int(t[_xstore_epoch_at].item())" in src
    assert "STORE-EPOCH LAYOUT STOP" in src
    # the epoch slot is appended right after the store-match slice, before the usable arm and the tail ballot
    assert src.index("_xstore_at = len(vals)") < src.index("_xstore_epoch_at = len(vals)") < src.index(
        "_usable_at = len(vals)")


def test_drain_backup_feeds_the_store_epoch():
    import inspect

    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache._drain_storage_control_queues_impl)
    assert "_store_reprobe_epoch.note_store_write(self, drained)" in src


def test_epoch_move_while_the_rid_is_not_voted_still_asks():
    """Review 1023 finding 1: the group epoch moves on a pass where the rid is not in the canonical head (here: an
    S42-held pass, canonical empty). The rid's schedule carries the epoch it was taken under, so it asks on the next
    pass it is voted -- on both ranks."""
    ranks = [_rank(), _rank()]
    for k in range(10):
        _pass(ranks, k)
    for s, _p in ranks:
        rpe.note_store_write(s.tree_cache, 1)
    for s, _p in ranks:  # the reduce of pass 10 carried no head (held), only the epoch slot
        rpe.after_reduce(s, [], [], min(rpe.build_epoch_payload(r)[0] for r, _ in ranks))
    for k in range(11, 21):
        _pass(ranks, k)
    for _s, p in ranks:
        # asks on the next voted pass, then the back-off restarts (the schedule took the new epoch) -- not every pass
        assert [x for x in _asked_passes(p) if x >= 11] == [11, 12, 14, 18]
    assert _asked_passes(ranks[0][1]) == _asked_passes(ranks[1][1])


def test_absent_group_vote_is_no_event_on_any_rank():
    """Review 1023 finding 4: on an S42 split the non-holding rank still walks its canonical head but reads the
    group value -1 (the holding rank voted ABSENT). That is no event: the schedule stands on every rank."""
    ranks = [_rank(), _rank()]
    for k in range(10):
        _pass(ranks, k)
    before = [dict(getattr(r, rpe.SCHED_ATTR)) for r in ranks[0][0].waiting_queue]
    ge = getattr(ranks[0][0], rpe.GROUP_EPOCH_ATTR)
    rpe.after_reduce(ranks[0][0], _canonical(), [-1, -1], ge)  # non-holding rank: head walked, group ABSENT
    rpe.after_reduce(ranks[1][0], [], [], ge)                  # holding rank: no head
    after = [dict(getattr(r, rpe.SCHED_ATTR)) for r in ranks[0][0].waiting_queue]
    assert before == after
    for k in range(11, 20):
        _pass(ranks, k)
    assert _asked_passes(ranks[0][1]) == _asked_passes(ranks[1][1])


def test_a_raising_after_reduce_never_raises_and_is_counted(monkeypatch):
    s, _p = _rank()
    monkeypatch.setattr(rpe, "max_passes", lambda env=None: (_ for _ in ()).throw(RuntimeError("boom")))
    rpe.after_reduce(s, _canonical(), [HOST_MATCH, HOST_MATCH], 0)
    assert rpe.stats()["raised"] == 1
    assert rpe.tick(s) == 1  # the tick advanced before the failure: still one per reduce on every rank
