"""h18port 1014 (desk/nf-h18port-1014), JOIN-IDLE / REF-AT-LAND / JOIN-LIVENESS against raenge-nie-uneins.

THE QUESTION (coordinator): JOIN-IDLE breaks a FOREIGN claim rank-locally after 1 s without a move; the holder's
late completion is refused (status 6) and its read ends at that page. The read lengths of the three P stages then
differ -- the cache cases show exactly that shape (PP0 1475 pages, PP1/PP2 2016). Does a rank-local read length
ever become a rank-local ACTION (a different batch on one stage)?

THE ANSWER this file pins, driven through the line's told machinery (PF-NF + W27-HELD-STOP, the rig of
test_pdflip_w27_held_stop_1014 with the cache-case numbers): the read length only enters PP0's told decision. Every
asymmetry ends in ONE answer for all three stages -- the told where every follower reproduces it, told=0 for every
rank (PF TOLD-FALLBACK: a uniform recompute, the cost of the asymmetry) where one does not. RANK-EQUALITY MUTANT: a
follower that admits on its OWN read instead of the told plans a different batch than PP0 -- the #1233 split the
told machinery exists to prevent.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

import test_pdflip_w27_held_stop_1014 as W  # noqa: E402

PAGE = 64
SHORT = W.HEAD + 1475 * PAGE    # the stage whose claims JOIN-IDLE broke: its read ended at page 1475
LONG = W.HEAD + 2016 * PAGE     # the stages that read on


def _set(monkeypatch, *, pp0, fol, fol_anchors):
    """PP0 told = ``pp0`` (its own reach, state there); the followers read to ``fol`` with their recurrent states
    at ``fol_anchors``."""
    monkeypatch.setattr(W, "TOLD", pp0)
    monkeypatch.setattr(W, "OWN", fol)
    monkeypatch.setattr(W, "PROMPT", max(pp0, fol) + 20000)
    monkeypatch.setattr(W, "_pp0_tree", lambda: W._Tree(dev=W.HEAD, host=pp0 - W.HEAD, anchors=(W.HEAD, pp0),
                                                         record=pp0 - W.HEAD))
    monkeypatch.setattr(W, "_follower_tree", lambda: W._Tree(dev=W.HEAD, host=fol - W.HEAD,
                                                             anchors=(W.HEAD,) + tuple(fol_anchors),
                                                             record=fol - W.HEAD))


def _uniform(plans):
    """One answer for the group: the same batch on every stage, or the same named stop on every follower."""
    vals = list(plans.values())
    if all(isinstance(p, list) for p in vals):
        return vals[0] == vals[1] == vals[2]
    return False


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("FLLIPER_PDFLIP_FOLLOWER_REACH_TOLD", "FLLIPER_PDFLIP_DUAL_SHARE", "FLLIPER_PDFLIP_DUAL_LAYOUT",
              W.tf.ENV, W.HELD_STOP_ENV, W.fb.ENV_FALLBACK):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(W.m.ENV_FOLLOWER_EARLY_READ, "1")
    monkeypatch.setattr(W.m, "_absolute_armed", lambda: True)


def test_pp0_short_followers_long_resumable_at_told_one_batch(monkeypatch):
    """PP0 1475 pages, PP1/PP2 2016 with a state at PP0's depth too: the followers reproduce the told."""
    _set(monkeypatch, pp0=SHORT, fol=LONG, fol_anchors=(SHORT, LONG))
    pf, told_final, plans = W._three_stage_plans(monkeypatch)
    assert pf is True
    assert _uniform(plans), {r: (W._rows(p) if isinstance(p, list) else repr(p)) for r, p in plans.items()}
    assert told_final in (0, SHORT)


def test_pp0_short_followers_long_unresumable_at_told_told_zero_for_all(monkeypatch):
    """PP0 1475 pages, PP1/PP2 2016 with their state at 2016 only (the metal shape): PF answers told=0 for EVERY
    rank -- one batch, the asymmetry costs a recompute, never a split."""
    _set(monkeypatch, pp0=SHORT, fol=LONG, fol_anchors=(LONG,))
    pf, told_final, plans = W._three_stage_plans(monkeypatch)
    assert pf is True and told_final == 0
    assert _uniform(plans)


def test_rank_equality_mutant_follower_admits_on_its_own_read(monkeypatch):
    """MUTANT: a follower that plans from its OWN read length (its tree reaches 2016 pages, nobody asks PP0) runs a
    different batch than PP0, which stands at its told of 1475 pages -- the #1233 split. The real path above gives
    one batch."""
    _set(monkeypatch, pp0=SHORT, fol=LONG, fol_anchors=(SHORT, LONG))
    monkeypatch.setattr(W, "EXTENDS", {"pdflip-0-9": 5698, "pdflip-1-11": 25, W.RID: 10560})
    pp0 = W._stage(0, W._pp0_tree())
    pp0._pdflip_store_told[W.RID] = SHORT
    pp0_batch = W._plan(pp0, W._queue(), set())
    # the follower's extend for the rid is its own remainder: prompt - its own reach (2016 pages), not PP0's
    own_extends = dict(W.EXTENDS)
    own_extends[W.RID] = max(0, W.EXTENDS[W.RID] - (LONG - SHORT))
    monkeypatch.setattr(W, "EXTENDS", own_extends)
    fol_batch = W._plan(W._stage(1, W._follower_tree()), W._queue(), set())
    assert W._rows(pp0_batch) != W._rows(fol_batch), (pp0_batch, fol_batch)
