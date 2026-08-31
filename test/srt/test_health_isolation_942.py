"""#942 HEALTH ISOLATION -- the never-again anchor.

WHY THIS TEST EXISTS AS A TEST, under a speed mode that keeps suites out of
the critical path. It is the standing user order's own acceptance clause:
health checks must never influence measurements, and the anchor demanded for
it is "a test that drives a health rid through admission and asserts zero
arming". It is ONE targeted cold test, not a battery, and it can be run
without a GPU.

THE RULE IT PINS (state it here so a future reader cannot re-derive it wrong):

    A `/health_generate` probe is a LIVENESS INSTRUMENT, NOT WORK.
    It must never appear in a backlog or economy term -- `_pending_prefill_
    tokens(include_health=False)`, `_admissible_prefill_tokens(include_health=
    False)`, and everything derived from them (`starved`, `work_exists`, the
    tp-ward arm). It IS served, in place, by the #887 one-chunk grant, and the
    grant's own reading therefore keeps health INCLUDED. Two questions, split
    at the call site -- never one number doing both jobs.

THE TRAP THIS PINS SHUT, and it is the reason the split is at the call site:
a blanket exclusion is the obvious form and it is WRONG in the silent
direction. `phase_purity.tp_compute_fits_in_one_chunk` grants only while
`0 < pending < chunk`. Subtract the probe there too and a lone health probe
measures 0, `0 < 0` is False, the grant collapses, and the probe the isolation
exists to serve in place is served by nothing at all. Hence `test_grant_
reading_still_sees_health`, which fails if someone "simplifies" the split away.

Measured basis (#942): 1-token health probes armed full ~4 s cutovers because
`tp_threshold` is 0 under purity, so ANY pending token fires the tp-ward arm
-- 12 arms for a one-token probe on one boot, on the same boot whose gate
logged `LAYOUT-ALLOWED tp_compute_one_chunk (#887)` 132 times.
"""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "python")
)

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.managers.scheduler import (
    Scheduler,
    _arriving_prefill_tokens,
)
from sglang.srt.managers.utils import is_health_check_generate_req


class _Req:
    """Minimal stand-in carrying only what the backlog terms read."""

    def __init__(self, rid, n_tokens):
        self.rid = rid
        self.origin_input_ids = list(range(n_tokens))
        self.is_retracted = False
        self.extend_range = None
        self.cache_protected_len = 0
        self.needs_prefill_pass = False


class _Sched:
    """Bind the real, unbound Scheduler methods onto a bare namespace.

    Deliberately NOT a mock of the methods under test: the functions executed
    here are the production ones, so a change to them is what this test sees.
    """

    def __init__(self, queued):
        self.waiting_queue = list(queued)
        self.chunked_req = None
        self.running_batch = None

    _pending_prefill_tokens = Scheduler._pending_prefill_tokens
    _admissible_prefill_tokens = Scheduler._admissible_prefill_tokens


def _health_req(n_tokens=1):
    return _Req(f"{HEALTH_CHECK_RID_PREFIX}_deadbeef", n_tokens)


def _real_req(n_tokens=4618):
    return _Req("bce57cf8e3ad4dd2", n_tokens)


class TestHealthIsolation(unittest.TestCase):
    def test_predicate_recognises_the_source_tag(self):
        """The tag is upstream's rid prefix, set at the /health_generate
        source. No second tag: if this ever fails, someone introduced
        parallel bookkeeping."""
        self.assertTrue(is_health_check_generate_req(_health_req()))
        self.assertFalse(is_health_check_generate_req(_real_req()))

    def test_health_contributes_zero_to_the_economy_term(self):
        """THE CORE ASSERTION: a health probe is not backlog."""
        s = _Sched([_health_req(1)])
        self.assertEqual(s._pending_prefill_tokens(include_health=False), 0)
        self.assertEqual(s._admissible_prefill_tokens(include_health=False), 0)

    def test_zero_arming_at_idle_the_942_shape(self):
        """The exact #942 shape: idle box, one 1-token probe, purity in force.

        Under purity `tp_threshold` is 0, so the tp-ward arm fires on
        `pending > 0`. With the probe excluded, pending is 0 and the
        comparison is False -- no arm. This is the assertion the order calls
        'Null-Arming'."""
        s = _Sched([_health_req(1)])
        pending = s._pending_prefill_tokens(include_health=False)
        admissible = s._admissible_prefill_tokens(include_health=False)
        tp_threshold = 0  # purity
        self.assertFalse(
            pending > tp_threshold, "a health probe armed the tp-ward flip"
        )
        # `starved` bypasses the min-dwell thrash bound; it must not fire
        # either. Its form is: no decode work AND max(pending, admissible) > 0.
        self.assertFalse(
            max(pending, admissible) > 0,
            "a health probe would bypass the min-dwell thrash bound",
        )

    def test_real_work_is_untouched(self):
        """The exclusion must be surgical: real prefill still counts in full,
        with or beside a probe."""
        s = _Sched([_real_req(4618)])
        self.assertEqual(s._pending_prefill_tokens(include_health=False), 4618)
        mixed = _Sched([_real_req(4618), _health_req(1)])
        self.assertEqual(
            mixed._pending_prefill_tokens(include_health=False),
            4618,
            "the probe leaked into the economy term beside real work",
        )
        self.assertEqual(
            mixed._pending_prefill_tokens(),
            4619,
            "the service reading lost a token it must still serve",
        )

    def test_grant_reading_still_sees_health(self):
        """THE ANTI-SIMPLIFICATION CLAUSE.

        If a future reader "cleans up" the split by excluding health
        everywhere, this fails. The #887 grant is what serves the probe in
        place; its reading must keep health INCLUDED or the probe is served
        by nothing and the isolation has made things worse, silently."""
        s = _Sched([_health_req(1)])
        self.assertEqual(
            s._pending_prefill_tokens(),
            1,
            "the grant reading lost the probe it exists to serve in place",
        )

    def test_default_is_byte_identical_for_every_legacy_caller(self):
        """Every pre-existing caller passes no keyword and must be
        unchanged."""
        s = _Sched([_real_req(100), _health_req(1)])
        self.assertEqual(s._pending_prefill_tokens(), 101)
        self.assertEqual(s._admissible_prefill_tokens(), 101)

    def test_arriving_term_excludes_health_only_when_asked(self):
        """The #713 inflight term is the one that fires FIRST on an idle box
        -- the probe is counted there before it is ever queued."""
        inflight = [_health_req(1), _real_req(50)]
        self.assertEqual(_arriving_prefill_tokens(inflight), 51)
        self.assertEqual(
            _arriving_prefill_tokens(
                inflight, None, exclude=is_health_check_generate_req
            ),
            50,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
