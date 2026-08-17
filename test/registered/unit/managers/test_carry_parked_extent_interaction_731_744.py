"""#731 x #744: consuming the queue entry must not change the parked extent.

Two fixes landed on neighbouring machinery within hours of each other, and
nobody had checked that they compose.

* **#731** (`fdcf837206`) makes the flip carry CONSUME the waiting-queue entry
  of a request it re-homes, so the request stops existing in two places and the
  backlog stops being billed twice.
* **#744** (`5085766fa9`) makes the phase flip's PARKED EXTENT visible to the
  KV backing rung, so the rung cannot evict rows the flip is about to pack.

The question this file answers: does removing the request from
``waiting_queue`` change what ``_flip_pending`` enumerates? If it did, #731
would have silently narrowed #744's protection and re-opened the illegal-access
window from the other side -- and neither suite would have noticed, because
each pins its own half.

**It does not, and the reason is structural rather than lucky.**
``_live_reqs`` enumerates ``running_mbs``, ``running_batch``, ``last_batch``
and ``chunked_req``. It has never read ``waiting_queue``. The carry ADDS the
request to ``running_mbs[0]``, which IS enumerated, so the carried request's
rows reach the parked extent through the resident side both before and after
the consume. Before #731 the request sat in both sets and the enumeration
deduplicated by identity; after #731 it sits in one. Either way it contributes
its rows exactly once.

That is a property worth pinning rather than re-deriving: the next person to
change either enumeration or the carry needs this to fail if they break it.
"""

import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SEQLEN = 51_311  # the specimen's request length, from ANALYSE_741
POOL_IDX = 3


class _Req:
    def __init__(self, rid, seqlen, req_pool_idx):
        self.rid = rid
        self.seqlen = seqlen
        self.req_pool_idx = req_pool_idx


class _Batch:
    def __init__(self, reqs):
        self.reqs = list(reqs)


def _scheduler(*, carried, queued):
    """A scheduler stub carrying exactly what the enumeration reads."""
    import torch

    n_rows, n_cols = 8, SEQLEN
    req_to_token = torch.arange(n_rows * n_cols, dtype=torch.int64).reshape(
        n_rows, n_cols
    )
    return types.SimpleNamespace(
        running_mbs=[_Batch([carried])] if carried is not None else [],
        running_batch=None,
        last_batch=None,
        chunked_req=None,
        waiting_queue=list(queued),
        req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
        tree_cache=types.SimpleNamespace(all_values_flatten=lambda: None),
        flip_live_split=None,
    )


def _enumerate(sched):
    """Run the flip's live-slot function; return (rows, max, extent)."""
    from sglang.srt.managers.phase_flip_runtime import build_flip_live_slots_fn

    fn = build_flip_live_slots_fn(sched)
    fn()
    split = fn.last_split
    return (
        int(split["req_rows"]),
        int(split["req_max"]),
        getattr(fn, "last_req_extent", None),
    )


class TestConsumeDoesNotNarrowTheParkedExtent(CustomTestCase):
    def setUp(self):
        self.req = _Req("carried-rid", SEQLEN, POOL_IDX)

    def test_the_carried_request_is_enumerated_at_all(self):
        """CAN-FAIL GUARD, and it has to come first.

        If the enumeration returned nothing in both states the comparison
        below would pass while proving nothing at all. Establish that the
        carried request really does contribute rows before comparing.
        """
        rows, top, extent = _enumerate(_scheduler(carried=self.req, queued=[self.req]))
        self.assertEqual(rows, SEQLEN, "the carried request must contribute its rows")
        self.assertGreater(top, 0)
        self.assertEqual(extent, (SEQLEN, top), "#744's sticky extent must be set")

    def test_the_extent_is_identical_before_and_after_the_consume(self):
        """THE INTERACTION. #731 removes the queue entry; #744 must not care."""
        before = _enumerate(_scheduler(carried=self.req, queued=[self.req]))
        after = _enumerate(_scheduler(carried=self.req, queued=[]))
        self.assertEqual(
            before,
            after,
            "consuming the waiting-queue entry changed what the flip enumerates "
            "-- #731 would have narrowed #744's parked extent",
        )

    def test_the_real_consume_helper_produces_that_state(self):
        """Use #731's own helper rather than hand-building the after state."""
        from sglang.srt.managers.phase_flip_resident_carry import (
            _consume_carried_from_waiting_queue,
        )

        sched = _scheduler(carried=self.req, queued=[self.req])
        before = _enumerate(sched)
        removed = _consume_carried_from_waiting_queue(
            sched, _Batch([self.req])
        )
        self.assertEqual(removed, 1, "#731's consume did not fire")
        self.assertEqual(sched.waiting_queue, [])
        after = _enumerate(sched)
        self.assertEqual(before, after)

    def test_a_queue_only_request_never_entered_the_extent(self):
        """The converse, which is why the consume is safe.

        A request that is ONLY queued contributes nothing to the parked
        extent, because the enumeration does not read the queue. So the queue
        was never a source the consume could remove from.
        """
        queued_only = _Req("queued-only", SEQLEN, POOL_IDX + 1)
        rows, _, _ = _enumerate(_scheduler(carried=None, queued=[queued_only]))
        self.assertEqual(rows, 0, "the queue must contribute nothing")

    def test_the_enumeration_reads_no_queue(self):
        """Structural pin, parsed not grepped.

        The property above holds because ``_live_reqs`` sources are the
        resident ones. If someone adds the queue as a source, the two fixes
        start interacting and this file's premise dies -- so fail here, where
        the reason is written down, rather than somewhere far away.
        """
        import ast
        import inspect

        from sglang.srt.managers import phase_flip_runtime as m

        src = inspect.getsource(m._live_reqs)
        names = {
            n.attr
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Attribute)
        } | {
            c.value
            for c in ast.walk(ast.parse(src))
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        }
        self.assertIn("running_mbs", names, "parser must actually see the sources")
        self.assertNotIn(
            "waiting_queue",
            names,
            "_live_reqs now reads the queue: #731 and #744 interact, re-derive "
            "this file's premise before changing either",
        )


if __name__ == "__main__":
    unittest.main()
