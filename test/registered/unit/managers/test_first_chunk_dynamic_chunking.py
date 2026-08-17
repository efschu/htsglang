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
    # #624 stub drift, fourth instance of the same shape: #656 added an
    # engagement-proof call to the method under test and nothing bound it
    # here, so both cases failed with
    # ``AttributeError: '_Sched' object has no attribute
    # '_log_dynamic_chunk_engagement'``. Bound off the real class for the same
    # reason ``size_for`` is: a stand-in that reimplements it would stop
    # pinning it. It writes ``_dyn_chunk_last_logged`` on this instance and
    # reads ``chunked_prefill_size``, both of which this harness already has.
    _log_dynamic_chunk_engagement = Scheduler._log_dynamic_chunk_engagement


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


# ---------------------------------------------------------------------------
# The harness must track the production surface (#624 stub drift)
# ---------------------------------------------------------------------------


def _self_attributes_read_by(func) -> set:
    """Names this function reads off ``self`` (assignment targets excluded).

    Source-level on purpose: the drift is "someone added a ``self.X`` to the
    method and did not bind X on the stand-in", which is a property of the
    method's TEXT. Executing it to find out would need the very stand-in whose
    completeness is in question.
    """
    import inspect
    import re
    import textwrap

    src = textwrap.dedent(inspect.getsource(func))
    assigned = set(re.findall(r"self\.([A-Za-z_]\w*)\s*=", src))
    return set(re.findall(r"self\.([A-Za-z_]\w*)", src)) - assigned


def _required_surface(entry, harness_cls) -> set:
    """Every ``self`` member ``entry`` needs from ``harness_cls``.

    ONE LEVEL, AND ONLY THROUGH METHODS THE HARNESS ACTUALLY INHERITS. If the
    stand-in supplies its OWN implementation of a callee -- as ``_Sched`` does
    for ``predict_next_chunk_size``, which is the whole point of the harness --
    then the production version's internals (``length_predictor``,
    ``model_config``, ``ps`` ...) are never reached and demanding them would
    make this guard fail on bindings the harness legitimately does not need.
    Descending only into inherited callees is what keeps the requirement equal
    to what the code path actually touches.
    """
    needed = _self_attributes_read_by(entry)
    for name in sorted(needed):
        production = getattr(Scheduler, name, None)
        if not callable(production):
            continue
        if getattr(harness_cls, name, None) is production:
            needed = needed | _self_attributes_read_by(production)
    return needed


class TheHarnessTracksTheProductionSurface(unittest.TestCase):
    """#624 stub-drift class, closed for this harness.

    ``_Sched`` binds ``Scheduler.dynamic_chunked_prefill_size`` and hand-writes
    the state it reads. When #656 added the engagement-proof call, nothing
    bound it here and both cases went red with an ``AttributeError`` naming ONE
    attribute -- the same worst-possible signal that let the sibling
    ``BudgetHarness`` drift three times running: it reports the first missing
    member, not the set, so fixing the named one can leave the next queued.

    This checks the harness against the real method instead of against the last
    incident. A member added to ``dynamic_chunked_prefill_size`` and not bound
    here fails THIS case, with the full list, before it can fail the sizing
    cases with a single name.
    """

    def test_every_member_the_sizer_touches_is_on_the_harness(self):
        harness = _Sched(chunked_req=None)
        required = _required_surface(Scheduler.dynamic_chunked_prefill_size, _Sched)
        missing = sorted(n for n in required if not hasattr(harness, n))
        self.assertEqual(
            missing,
            [],
            "_Sched has drifted behind Scheduler.dynamic_chunked_prefill_size: "
            f"{missing}. Bind these from Scheduler (or give the harness a "
            "stand-in field) so the sizing cases keep exercising the real "
            "contract instead of failing with an AttributeError.",
        )

    def test_the_guard_can_fail(self):
        """A drift detector that cannot report a drift is decoration."""
        harness = _Sched(chunked_req=None)
        required = _required_surface(Scheduler.dynamic_chunked_prefill_size, _Sched) | {
            "_a_member_added_next_quarter"
        }
        missing = sorted(n for n in required if not hasattr(harness, n))
        self.assertEqual(missing, ["_a_member_added_next_quarter"])

    def test_an_overridden_callee_is_not_descended_into(self):
        """The refinement that keeps this guard honest: ``_Sched`` supplies its
        own ``predict_next_chunk_size``, so the production one's state is never
        reached and must not be demanded."""
        required = _required_surface(Scheduler.dynamic_chunked_prefill_size, _Sched)
        self.assertIn("predict_next_chunk_size", required)
        for only_reachable_through_the_real_predictor in (
            "length_predictor",
            "model_config",
            "ps",
        ):
            self.assertNotIn(only_reachable_through_the_real_predictor, required)

    def test_an_inherited_callee_is_descended_into(self):
        """The other half: ``_log_dynamic_chunk_engagement`` IS inherited, so
        what it reads is genuinely required."""
        required = _required_surface(Scheduler.dynamic_chunked_prefill_size, _Sched)
        self.assertIn("_log_dynamic_chunk_engagement", required)
        self.assertIn("chunked_prefill_size", required)


if __name__ == "__main__":
    unittest.main()
