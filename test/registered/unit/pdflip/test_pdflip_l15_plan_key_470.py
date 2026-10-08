"""Item 470 (27B y8r flip regression): the L15 wake plan.

y8r 09:14:13 and 09:17:44, both D sleeps that went through the SLEEP1X reuse
(second flush, ``restamp`` writes the release's flip epoch into the published
manifest):

    L15-REFILL rank=0 done: 54409 KV row(s) ... steps=setup:0,plan:972,...
    L15-REFILL rank=0 done: 52687 KV row(s) ... steps=setup:0,plan:844,...

The holds without a restamp read ``plan:21 / 35 / 52 / 58``. The plan cache was
keyed with ``fingerprint(m)``, which hashes ``m.epoch``; the warm at the first
flush carried the old epoch, the wake read the restamped one and rebuilt the
plan on the resume RPC (every peer's fence waits for TP0). The plan does not
depend on the epoch.

- the cache key ignores the epoch, still follows every content field;
- the numpy plan returns exactly the reference answer (order, rids, lanes,
  short l2 lists, shared prefixes, duplicate rids) and leaves conflicts to
  the reference, which raises.
"""
import dataclasses
import random
import time

import pytest

from flliper.srt.pdflip import l15_restore
from flliper.srt.pdflip.l15_manifest import HoldSpan, Manifest

PREFIX = [0, 62, 83, 104]


def _tips(n_tips=7, tokens=28000, shared=20000, lanes=True, seed=3):
    rnd = random.Random(seed)
    base = list(range(1000, 1000 + shared))
    spans, nxt = [], 1000 + shared + 7
    for k in range(n_tips):
        own = list(range(nxt, nxt + tokens - shared))
        nxt += len(own) + rnd.randint(0, 5)
        slots = tuple(base + own)
        spans.append(HoldSpan(
            rid="tree:%d" % k, depth=len(slots), slots=slots, anchor_slot=k,
            l2_slots=tuple(s + 5 for s in slots),
            l2_gens=tuple([3] * len(slots)),
            l2_lanes=tuple(s % 2 for s in slots) if lanes else ()))
    return Manifest(epoch=10, pid=1, spans=tuple(spans),
                    rows_by_rank=(50000, 50000, 50000), anchor_slots=n_tips + 1)


def test_cache_key_ignores_the_epoch_stamp():
    m = _tips(3, 3000, 2000)
    restamped = dataclasses.replace(m, epoch=12)
    assert l15_restore._plan_key(m, 0, PREFIX) == l15_restore._plan_key(restamped, 0, PREFIX)
    # ... and still follows every content field and the rank / prefix
    sp0 = m.spans[0]
    bad_gen = dataclasses.replace(sp0, l2_gens=(9,) + sp0.l2_gens[1:])
    other = dataclasses.replace(m, spans=(bad_gen,) + m.spans[1:])
    assert l15_restore._plan_key(m, 0, PREFIX) != l15_restore._plan_key(other, 0, PREFIX)
    assert l15_restore._plan_key(m, 0, PREFIX) != l15_restore._plan_key(m, 1, PREFIX)
    assert l15_restore._plan_key(m, 0, PREFIX) != l15_restore._plan_key(m, 0, [0, 1, 2, 3])


def test_wake_after_a_restamp_hits_the_plan_warmed_at_the_first_flush(monkeypatch):
    """The SLEEP1X reuse path: warm at the first flush (epoch 10), restamp to
    the flip epoch (12), the wake must find the plan without rebuilding."""
    m = _tips(3, 3000, 2000)
    l15_restore._PLAN_CACHE.clear()
    warmed = l15_restore.owned_l2_rows(m, 0, PREFIX)

    calls = []
    real = l15_restore.owned_l2_rows_uncached

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(l15_restore, "owned_l2_rows_uncached", counting)
    wake_m = dataclasses.replace(m, epoch=12)  # what l15_sleep_once.restamp writes
    got = l15_restore.owned_l2_rows(wake_m, 0, PREFIX)
    assert got == warmed
    assert not calls, "the restamped manifest rebuilt the plan on the resume RPC"
    # the sample plan of the same wake shares it too
    assert l15_restore.rid_tagged_plan(wake_m, 0, PREFIX)
    assert not calls


def _same(a, b):
    assert a == b


@pytest.mark.parametrize("seed", range(6))
def test_numpy_plan_equals_the_reference(seed):
    rnd = random.Random(seed)
    spans = []
    pool = list(range(0, 4000))
    for k in range(rnd.randint(1, 6)):
        n = rnd.randint(0, 600)
        # shared prefixes: spans draw from one slot pool in ascending runs
        start = rnd.choice([0, 0, 100, 700])
        slots = tuple(pool[start:start + n]) + tuple(
            rnd.sample(range(5000, 9000), rnd.randint(0, 50)))
        if rnd.random() < 0.3:
            l2s = tuple(s + 7 for s in slots[: max(0, len(slots) - rnd.randint(0, 40))])
        else:
            l2s = tuple(s + 7 if rnd.random() > 0.1 else -1 for s in slots)
        gens = tuple([4] * len(l2s))
        if rnd.random() < 0.3:
            gens = gens[: max(0, len(gens) - rnd.randint(0, 5))]
        lanes = () if rnd.random() < 0.4 else tuple(
            (s % 3) for s in slots)[: rnd.randint(0, len(slots))]
        spans.append(HoldSpan(
            rid="r%d" % (k if rnd.random() > 0.2 else 0), depth=len(slots),
            slots=slots, anchor_slot=k, l2_slots=l2s, l2_gens=gens, l2_lanes=lanes))
    m = Manifest(epoch=5, pid=1, spans=tuple(spans), rows_by_rank=(1, 1, 1), anchor_slots=2)
    for rank in range(3):
        try:
            ref = l15_restore.owned_l2_rows_reference(m, rank, PREFIX)
        except l15_restore.L15RefillError:
            # a real identity conflict: the numpy plan steps aside, the
            # reference raises through owned_l2_rows_uncached
            assert l15_restore._owned_l2_rows_np(m, rank, PREFIX) is None
            with pytest.raises(l15_restore.L15RefillError):
                l15_restore.owned_l2_rows_uncached(m, rank, PREFIX)
            continue
        np_rows = l15_restore._owned_l2_rows_np(m, rank, PREFIX)
        if np_rows is not None:
            assert np_rows == ref
        assert l15_restore.owned_l2_rows_uncached(m, rank, PREFIX) == ref


def test_numpy_plan_leaves_a_conflict_to_the_reference():
    a = HoldSpan(rid="a", depth=2, slots=(10, 11), anchor_slot=0,
                 l2_slots=(100, 101), l2_gens=(1, 1))
    b = HoldSpan(rid="b", depth=2, slots=(10, 12), anchor_slot=1,
                 l2_slots=(100, 102), l2_gens=(2, 1))  # slot 10: same l2 slot, other gen
    m = Manifest(epoch=1, pid=1, spans=(a, b), rows_by_rank=(1, 1, 1), anchor_slots=2)
    assert l15_restore._owned_l2_rows_np(m, 0, PREFIX) is None
    with pytest.raises(l15_restore.L15RefillError):
        l15_restore.owned_l2_rows_uncached(m, 0, PREFIX)
    with pytest.raises(l15_restore.L15RefillError):
        l15_restore.owned_l2_rows_reference(m, 0, PREFIX)


def test_numpy_plan_is_much_cheaper_on_the_y8r_hold_size():
    """7 tips, 196k held tokens that share a 20k prefix (y8r 09:14)."""
    m = _tips()
    t = time.perf_counter()
    ref = l15_restore.owned_l2_rows_reference(m, 0, PREFIX)
    t_ref = time.perf_counter() - t
    t = time.perf_counter()
    got = l15_restore.owned_l2_rows_uncached(m, 0, PREFIX)
    t_np = time.perf_counter() - t
    assert got == ref and len(ref) > 40000
    assert t_np < t_ref / 2, (t_np, t_ref)
    # shared prefix rows carry every sharing rid
    assert any(len(r[4]) == 7 for r in got)
