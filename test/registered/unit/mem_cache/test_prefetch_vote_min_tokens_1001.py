"""HANDBACK VOTE MIN: the #580 group vote decides against the same smallest read
as the local gate.

Metal gmps7 (dkr27bnvfp4dual1mbar1fs10011748, group D TP=3, 17:52:27-17:52:56):
every P hand-back whose span beyond D's match was below the prefetch threshold
(256) was refused ``#915 PREFETCH REFUSED reason=vote_negative`` although all
three ranks probed it present (H108 PRESENCE-PROBE present=True): weg2-0-6
need=24, weg2-0-10 need=40. The scheduler passes ``min_tokens=1`` for a
hand-back (handback_claim.handback_min_tokens) and the LOCAL gate honoured it
(``_min_len``), but the group vote compared ``group_len`` with the bare
``prefetch_threshold`` -> W31 -> W50 -> P ran leg 1 twice -> W53/503. Same code on
the flip (TP=3 D reads its hand-backs through the same vote).

DANGER DIRECTION: a short hand-back span with every rank present must register on
a group-decided (symmetric) tree; the default (no min_tokens) keeps refusing below
the threshold. MUTANT: the vote's threshold turned back -> the test is red
(asserted in-suite on the real method's source).
"""

import importlib.util
import inspect
import os
import textwrap
import types

import pytest

from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_census915", os.path.join(_HERE, "test_prefetch_gate_census_915.py")
)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

THRESHOLD = 256


def _group(prefetch=None):
    """One rank of a TP group whose ranks all vote the same span (every rank
    present): the MIN-reduce leaves the vote unchanged. A real
    UnifiedRadixCache (the census harness), symmetric = group-decided."""
    reduces = []

    def _all_reduce(tensor, op, label=""):
        reduces.append(label)

    cache = H._serving_tree(available=100000, symmetric=True, vote=_all_reduce)
    assert cache.prefetch_threshold == THRESHOLD
    if prefetch is not None:
        cache.prefetch_from_storage = types.MethodType(prefetch, cache)
    return cache, reduces


def _read(cache, rid, need, min_tokens):
    cache.prefetch_from_storage(
        rid, H._node(), list(range(need)), locally_eligible=True, min_tokens=min_tokens,
    )
    return rid in cache.ongoing_prefetch


@pytest.mark.parametrize("rid,need", [("weg2-0-6", 24), ("weg2-0-10", 40)])
def test_a_short_handback_span_registers_through_the_group_vote(rid, need, prefetch=None):
    cache, reduces = _group(prefetch)
    assert _read(cache, rid, need, min_tokens=1), (
        "every rank present, min_tokens=1: the group vote must not refuse a "
        "hand-back below the 256 threshold (metal: vote_negative -> W31/W50/W53)")
    assert "prefetch_participation_vote" in reduces, "the group decided (symmetric tree)"


def test_without_min_tokens_the_group_still_refuses_below_the_threshold():
    cache, reduces = _group()
    assert not _read(cache, "weg2-0-6", 24, min_tokens=None)
    assert "prefetch_participation_vote" in reduces


def test_at_the_threshold_the_default_read_registers():
    cache, _ = _group()
    assert _read(cache, "weg2-0-1", THRESHOLD, min_tokens=None)


def _mutant_threshold_vote():
    """The real prefetch_from_storage with the vote's bound turned back to the
    bare threshold (the pre-fix line)."""
    fn = inspect.unwrap(urc.UnifiedRadixCache.prefetch_from_storage)
    src = textwrap.dedent(inspect.getsource(fn))
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("@"))
    fixed = "if group_len < _min_len or _end_decline:"
    assert src.count(fixed) == 1, "the vote's bound moved -- re-aim the mutant"
    src = src.replace(fixed, "if group_len < self.prefetch_threshold or _end_decline:")
    ns = dict(vars(urc))
    exec(compile(src, urc.__file__, "exec"), ns)
    return ns["prefetch_from_storage"]


def test_the_threshold_mutant_turns_the_short_handback_test_red():
    mutant = _mutant_threshold_vote()
    with pytest.raises(AssertionError):
        test_a_short_handback_span_registers_through_the_group_vote("weg2-0-10", 40, prefetch=mutant)
