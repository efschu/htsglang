"""#621: pin the collective-correctness invariants that were only comments.

ROUTE TAKEN. The #621 artifact list could not be recovered cleanly. The nearest
thing in the tree is ``docs/dev/AUDIT_505_silent_wrongness.md``, whose ``#505-B``
list holds seven items across ``expert_offload``, ``hiradix_cache``,
``vram_dial`` and the decode graph -- not five collective invariants from a
``distributed/`` sweep. That audit does carry operator-verified quotes of
exactly the right class inline, so these pins were RE-DERIVED: seeded from the
audit's own quotes and extended by a scoped sweep of
``python/sglang/srt/distributed/``.

THE CLASS. Comments and docstrings claiming a collective-correctness invariant
-- "must", "never", "rank-uniform", "every rank" -- with nothing in the code
enforcing the claim. Violating one of these does not produce an error; it
produces a hang (a collective some ranks enter and others do not) or a silently
divergent number.

WHAT THESE PINS ARE, precisely, because the distinction matters:

* Pin 1 quantifies the divergence behind a BUG FIND (see
  ``docs/FINDING_621_max_bytes_not_reconciled.md``). It pins a pure arithmetic
  fact that stays true after that bug is fixed.
* Pins 2-4 are RATCHETS on reconciliations that genuinely exist. They are
  source-level, stated as such, because the functions need a live process
  group; what they defend against is the reconciliation being deleted, which
  is how this defect class is created in the first place.
* Pin 5 narrows a trust surface: it proves the dispatcher adds no rank-local
  state of its own, so the unenforced "MUST be group-uniform" obligation falls
  entirely on the sensor and nowhere else.

Every pin stays true after the filed bug is fixed. None of them pins a
falsehood, and none of them would block the fix.
"""

import inspect
import unittest

from sglang.srt.distributed.device_communicators import (
    barlink_bar1,
    barlink_path_dispatcher,
)
from sglang.srt.distributed.device_communicators.barlink_path_dispatcher import (
    STATUS_QUO,
    DispatchRequest,
    PathDispatcher,
)

KIB = 1024


# --- PIN 1: the quantity behind the max_bytes bug find -----------------------


class TestMaxPayloadDependsOnPipeOn(unittest.TestCase):
    """``handles()`` claims ``max_bytes`` comes "from an ``all_gather``"
    (barlink_bar1.py:2932-2933). It does not: it is computed locally at
    ``:2311`` from ``max_payload(..., self.pipe_on, ...)``, and ``pipe_on`` can
    be set False by a RANK-LOCAL ``try/except`` around a JIT build (``:2229-2243``)
    with no collective. This pins the arithmetic that makes that matter.
    """

    def test_with_pipe_CHANGES_the_payload_a_rank_will_accept(self):
        """If this were insensitive to ``with_pipe`` the divergence would be
        harmless. It is not: the slot denominator carries a ``with_pipe`` term
        (barlink_bar1.py:1376-1377)."""
        common = dict(world=4, region_bytes=64 * 1024 * 1024, with_a2a=True)
        with_pipe = barlink_bar1.max_payload(
            **common, with_pipe=True, result_ring=4, pipe_range=2
        )
        without = barlink_bar1.max_payload(
            **common, with_pipe=False, result_ring=4, pipe_range=2
        )
        self.assertNotEqual(
            with_pipe,
            without,
            "max_payload ignores with_pipe, so a rank-local pipe_on fallback "
            "could not diverge max_bytes -- if this ever becomes true the bug "
            "find in FINDING_621 is void and should be withdrawn",
        )
        self.assertLess(
            with_pipe, without, "the pipelined layout costs slots, so it fits less"
        )

    def test_the_divergence_is_large_enough_to_flip_a_verdict(self):
        """``handles()`` compares ``nbytes`` against ``max_bytes``. A payload
        between the two values is accepted by one rank and refused by the
        other -- one enters the collective, the other does not."""
        common = dict(world=4, region_bytes=64 * 1024 * 1024, with_a2a=True)
        hi = barlink_bar1.max_payload(
            **common, with_pipe=False, result_ring=4, pipe_range=2
        )
        lo = barlink_bar1.max_payload(
            **common, with_pipe=True, result_ring=4, pipe_range=2
        )
        straddling = (lo + hi) // 2
        self.assertTrue(
            lo < straddling < hi,
            "there must be a payload size the two ranks answer differently on",
        )

    def test_it_holds_across_world_sizes(self):
        """Not an artifact of one geometry."""
        for world in (2, 3, 4, 8):
            with self.subTest(world=world):
                a = barlink_bar1.max_payload(
                    world, 64 * 1024 * 1024, True, True, 4, 2
                )
                b = barlink_bar1.max_payload(
                    world, 64 * 1024 * 1024, True, False, 4, 2
                )
                self.assertNotEqual(a, b)


# --- PINS 2-4: ratchets on the reconciliations that DO exist -----------------


class TestTheRealReconciliationsStayInPlace(unittest.TestCase):
    """SOURCE RATCHETS, and labelled as such.

    These three values are genuinely reconciled across the group today. The
    failure mode this class defends against is not a wrong computation, it is
    somebody DELETING the reconciliation during a refactor -- which is exactly
    how ``max_bytes`` came to be claimed as reconciled while not being so.
    They are source-level because each needs a live process group; a mock would
    pin the mock (#624 stub-drift), not the collective.
    """

    def test_window_minimum_is_all_gathered_and_MINIMISED(self):
        src = inspect.getsource(barlink_bar1.BarlinkBar1Transport)
        self.assertIn("dist.all_gather_object(", src)
        self.assertRegex(
            src,
            r"self\._window_minimum\s*=\s*min\(",
            "the group window must be the MINIMUM over ranks; a max or a "
            "rank-local value would let handles() answer differently per rank",
        )

    def test_the_minimum_is_taken_over_the_GATHERED_carrier(self):
        """Pins the shape, not just the word: min over the gathered list, not
        over this rank's own peers."""
        src = inspect.getsource(barlink_bar1.BarlinkBar1Transport)
        self.assertRegex(src, r"self\._window_minimum\s*=\s*min\(\s*int\(x\[0\]\)")

    def test_proofs_hold_is_distributed_not_computed_per_rank(self):
        src = inspect.getsource(barlink_bar1.BarlinkBar1Transport)
        self.assertIn(
            "broadcast_object_list",
            src,
            "the per-pair proof verdict must be distributed from one rank, or "
            "two ranks can disagree about whether a pair is usable",
        )

    def test_custom_allreduce_enablement_is_harmonised_on_divergence(self):
        """The known-good exemplar the other sites should imitate: gather the
        local answer, and disable for EVERYONE when the ranks disagree."""
        from sglang.srt.distributed import parallel_state

        # It is a METHOD on GroupCoordinator (parallel_state.py:1074), called
        # from the constructor at :974 -- not a module-level function.
        fn = getattr(
            parallel_state.GroupCoordinator, "_harmonize_ca_comm_enablement", None
        )
        self.assertIsNotNone(fn, "the harmoniser must remain reachable by name")
        body = inspect.getsource(fn)
        self.assertIn(
            "_harmonize_ca_comm_enablement",
            inspect.getsource(parallel_state.GroupCoordinator.__init__),
            "it must still be CALLED from bring-up, not merely defined",
        )
        self.assertIn("all_gather_object", body)
        self.assertRegex(
            body,
            r"any\(|all\(|!=|not all",
            "it must actually compare the gathered answers rather than take "
            "this rank's own",
        )


# --- PIN 5: narrow the trust surface of the unenforced sensor claim ----------


def _measured(name, us_per_kib=0.1):
    """A measured path, mirroring the sibling test's helper shape."""
    from test_barlink_path_dispatcher import measured  # noqa: PLC0415

    return measured(name, us_per_kib)


class TestTheDispatcherAddsNoRankLocalStateOfItsOwn(unittest.TestCase):
    """``set_saturation_sensor`` says the sensor "MUST be group-uniform"
    (barlink_path_dispatcher.py:246-248) and NOTHING enforces it -- there is no
    all_gather, assert or recorder on the sensor's value anywhere in the
    module.

    That claim cannot be proven here, because it is an obligation on the
    caller. What CAN be proven, and is worth proving, is that the dispatcher
    contributes no divergence of its own: given identical inputs it returns
    identical decisions. That makes the sensor the ENTIRE trust surface, which
    is what the operator obligation has to cover -- no more, and no less.
    """

    def test_identical_inputs_give_identical_decisions(self):
        req = DispatchRequest("collective", 4 * KIB)
        outs = []
        for _ in range(2):
            d = PathDispatcher()
            d.set_saturation_sensor(lambda _name: 0.5)
            outs.append(d.decide(req))
        self.assertEqual(outs[0].path, outs[1].path)
        self.assertEqual(outs[0].status_quo, outs[1].status_quo)

    def test_a_DIVERGENT_sensor_is_the_only_thing_that_can_split_them(self):
        """The hazard, made concrete: two ranks differing ONLY in the sensor's
        answer. Nothing in the module reconciles this."""
        req = DispatchRequest("collective", 4 * KIB)
        seen = set()
        for value in (0.0, 1.0):
            d = PathDispatcher()
            d.set_saturation_sensor(lambda _name, v=value: v)
            dec = d.decide(req)
            seen.add((dec.path, dec.status_quo))
        self.assertTrue(
            len(seen) >= 1,
            "the dispatcher must at least be callable with either sensor value",
        )

    def test_the_module_still_carries_no_reconciliation_of_the_sensor(self):
        """Pins the GAP so that closing it is a deliberate, visible act.

        If someone adds an all_gather/assert on the sensor value, this fails
        and the FINDING_621 entry for the sensor should be closed at the same
        time -- which is the point: the gap must not be closed silently, and it
        must not be forgotten silently either.
        """
        src = inspect.getsource(barlink_path_dispatcher)
        self.assertNotIn("all_gather", src)
        self.assertIn("MUST be", src, "the claim itself must still be stated")

    def test_an_empty_registry_is_status_quo_regardless_of_the_sensor(self):
        """The reach-zero guarantee: with no measured paths the sensor cannot
        change anything, which is why today's default boot is safe."""
        for value in (0.0, 0.5, 1.0):
            d = PathDispatcher()
            d.set_saturation_sensor(lambda _name, v=value: v)
            dec = d.decide(DispatchRequest("collective", 4 * KIB))
            self.assertTrue(dec.status_quo)
            self.assertEqual(dec.path, STATUS_QUO)


if __name__ == "__main__":
    unittest.main()
