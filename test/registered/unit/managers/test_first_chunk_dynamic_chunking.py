"""#656: dynamic chunking must size the FIRST chunk too.

THE DEFECT. ``get_new_batch_prefill`` gated the dynamic size on
``self.chunked_req is not None`` -- an in-flight partially-prefilled
request. So a prefill's first chunk always took the STATIC
``chunked_prefill_size`` and only chunks 2..N were sized dynamically.

That is not a documented quirk, it is a bug, and it matters most exactly
where the feature is supposed to pay. A request whose prompt fits inside
one chunk never becomes a ``chunked_req`` at all, so it is never sized
dynamically -- the predictor is skipped for the entire class of requests
it could serve in full. For longer prompts the first chunk is the one
that sets the pipeline's initial bubble, and it is the one chunk taken
blind.

Nothing about the sizing inputs requires an in-flight request: the
predictor takes ``history_len``, and at admission time the history of a
fresh prefill is 0. That is a real, known value, not a placeholder.

These tests pin the decision itself rather than the whole batch path, so
they cannot pass by accident when some unrelated branch above returns
early.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

STATIC = 8192
DYNAMIC = 3000


class _Sched:
    """Just enough scheduler to answer the sizing question."""

    def __init__(self, chunked_req, enabled=True, predicted=DYNAMIC):
        self.chunked_prefill_size = STATIC
        self.chunked_req = chunked_req
        self.enable_dynamic_chunking = enabled
        self._predicted = predicted
        self.seen_history = []

    def predict_next_chunk_size(self, history_len):
        self.seen_history.append(history_len)
        return self._predicted

    # The method under test, bound off the real class so the test cannot
    # drift from the implementation it is pinning.
    size_for = Scheduler.dynamic_chunked_prefill_size


def _chunked_req(prefix_len):
    return SimpleNamespace(prefix_indices=list(range(prefix_len)))


class FirstChunkIsSizedDynamically(unittest.TestCase):
    def test_a_fresh_prefill_uses_the_dynamic_size(self):
        """The bug, stated as the thing that must be true.

        No in-flight chunked request means this batch is starting a
        prefill from scratch. history_len is 0 -- known, not guessed.
        """
        s = _Sched(chunked_req=None)
        self.assertEqual(
            s.size_for(),
            DYNAMIC,
            "the first chunk fell back to the static size, so a prompt "
            "that fits in one chunk is never sized dynamically at all",
        )
        self.assertEqual(s.seen_history, [0])

    def test_a_continuing_prefill_still_uses_its_real_history(self):
        """The behaviour that already worked must not move."""
        s = _Sched(chunked_req=_chunked_req(1234))
        self.assertEqual(s.size_for(), DYNAMIC)
        self.assertEqual(s.seen_history, [1234])


class TheStaticSizeIsStillReachable(unittest.TestCase):
    """Can-fail proofs: the dynamic path must be refusable.

    Without these the fix could be "always return DYNAMIC", which would
    pass the tests above and break every boot that has the feature off.
    """

    def test_the_feature_off_takes_the_static_size(self):
        s = _Sched(chunked_req=None, enabled=False)
        self.assertEqual(s.size_for(), STATIC)
        self.assertEqual(s.seen_history, [], "predictor consulted while off")

    def test_a_predictor_that_declines_takes_the_static_size(self):
        """``predict_next_chunk_size`` returns None until it has profiled.

        Treating that None as a size would hand the scheduler a chunk of
        ``None`` tokens on every boot before the predictor is ready.
        """
        s = _Sched(chunked_req=None, predicted=None)
        self.assertEqual(s.size_for(), STATIC)

    def test_a_continuing_prefill_with_the_feature_off_is_static(self):
        s = _Sched(chunked_req=_chunked_req(99), enabled=False)
        self.assertEqual(s.size_for(), STATIC)


if __name__ == "__main__":
    unittest.main()
