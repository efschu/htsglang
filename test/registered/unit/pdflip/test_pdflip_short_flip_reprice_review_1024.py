"""REVIEW SHORT-FLIP-REPRICE-1024: the span start is read from the REQUEST (``_prefetch_registered_prefix_len``), not
from the RECORD it belongs to.

``Scheduler._prefetch_kvcache`` rewrites that stamp at EVERY call, before any verdict (scheduler.py, the #1176 stamp
right after ``_prefetch_span_tokens``): the deferral retry, the #1456 HOLD-REFETCH, a declined re-issue (metal 27B
rc41 ..._1024_110644 D 11:41:02 TP0, pdflip-21-215: ``#915 PREFETCH REFUSED reason=vote_negative`` +
``#1456 HOLD-REFETCH ... verdict=declined:vote_negative``). The record of the PREVIOUS read stays in
``prefetch_loaded_tokens_by_reqid`` until the next read terminates (only overwritten at termination, popped at
admission, cleared at abort), and ``_pdflip_store_shortfall_pass`` re-runs the arm over the whole waiting queue every
drain, in flight or not. Record of read k + stamp of call k+1 = a base that is not this record's start.

The plausibility bound (base + deliverable > len(ids) -> 0) catches the stale stamp only when the deliverable reaches
the prompt end; it then falls back to the OLD reading (the 215 defect returns). Where the deliverable is shorter
(P4b no-writer cap, park/fork-capped span end) the stale base is ACCEPTED and the delivered prefix is counted twice:
the remainder is under-read by what the read delivered (SOLL R7 direction; only the admission X gate on the group
match stands behind it).

The two xfail(strict) tests stated the required behaviour; MUSS-1 bound the span start to the record
(``PrefetchOutcome.span_start``, copied from the read's operation) -- the marks are gone and the former "today" pins
are inverted (author, 24.10.).
"""
from __future__ import annotations

import os
import sys
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_pdflip_short_flip_reprice_1024 as T  # noqa: E402

from flliper.srt.mem_cache.hicache_storage import PrefetchOutcome  # noqa: E402
from flliper.srt.pdflip import d_norecompute as dn  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

_env = T._env  # the author's autouse fixture (STORE_SHORT_TAIL=0, no cycle override)


def _rearm(s, r, new_stamp):
    """A later ``_prefetch_kvcache`` call re-stamped the request; the previous record still stands."""
    r._prefetch_registered_prefix_len = new_stamp
    return s._pdflip_note_store_shortfall(r)


def test_first_observation_is_right():
    s, r, _ = T._arm(T._rec215())
    assert r._pdflip_store_delivered == 70841 and dn.d_would_compute(r) == 158


def test_215_restamped_by_a_reissue_keeps_the_absolute_depth():
    """215 after one HOLD-REFETCH that did not replace the record (declined / still in flight): the stamp is now the
    matched depth 70841; 70841 + 17244 > 70999 -> base 0 -> witness 17091 -> would_compute 53908 (the metal defect)."""
    s, r, _ = T._arm(T._rec215())
    _rearm(s, r, T.BASE + T.GOT)
    assert r._pdflip_store_delivered == 70841
    assert dn.d_would_compute(r) == 158


def test_215_restamped_no_longer_reverts_to_the_old_reading():
    """Inverted pin (was: witness 17091 / would_compute 53908 after the re-stamp)."""
    s, r, _ = T._arm(T._rec215())
    _rearm(s, r, T.BASE + T.GOT)
    assert r._pdflip_store_delivered != T.GOT and dn.d_would_compute(r) != 53908


#: a capped deliverable (P4b no-writer cap: the store's own hit) -- read [40000, 70994), store holds 12000 of it,
#: the read delivered 10000. Real remainder 70999 - 50000 = 20999 > X.
_N, _B1, _MAT, _HELD = 70999, 40000, 10000, 12000


def _rec_capped():
    return PrefetchOutcome(_MAT, matched=0, deliverable=_HELD, synced=_MAT)


def test_capped_deliverable_restamped_is_not_counted_twice():
    s, r, _ = T._arm(_rec_capped(), base=_B1, n=_N)
    assert dn.d_would_compute(r) == 20999
    _rearm(s, r, _B1 + _MAT)                 # the re-issue matched the 10000 delivered tokens
    assert r._pdflip_store_delivered == _B1 + _MAT
    assert dn.d_would_compute(r) == 20999


def test_capped_deliverable_restamped_no_longer_under_reads_over_x():
    """Inverted pin (was: witness 60000, would_compute 10999 < X, ``d_direct_extends`` True for a real 20999 > X)."""
    s, r, _ = T._arm(_rec_capped(), base=_B1, n=_N)
    _rearm(s, r, _B1 + _MAT)
    assert r._pdflip_store_delivered == _B1 + _MAT
    rem = dn.d_would_compute(r)
    assert rem == 20999 > T.X
    r.weg2_front_x, r.weg2_front_priced, r.output_ids = T.X, 865, []
    assert dn.d_direct_extends(T._dsched(), r, rem) is False      # R7: over X it is not D's own extend


def test_refresh_with_a_restamped_request_and_a_whole_record():
    """refresh_delivered: a whole record of the read from 40000, the request stamp moved to 50000. Inverted pin (was:
    62000): the record's own start counts -> 40000 + 12000 = 52000; without a start on the record nothing is raised."""
    req = types.SimpleNamespace(rid="x", full_untruncated_fill_ids=list(range(_N)),
                                _prefetch_registered_prefix_len=_B1 + _MAT, _pdflip_store_delivered=_B1 + _MAT)
    whole = PrefetchOutcome(_HELD, matched=0, deliverable=_HELD, synced=_HELD)
    assert not whole.is_incomplete
    assert dn.refresh_delivered(req, whole) is False and req._pdflip_store_delivered == _B1 + _MAT
    whole.span_start = _B1
    assert dn.refresh_delivered(req, whole) is True
    assert req._pdflip_store_delivered == _B1 + _HELD
