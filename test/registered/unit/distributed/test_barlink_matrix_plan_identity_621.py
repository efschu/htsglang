# SPDX-License-Identifier: Apache-2.0
"""Bit-identity pin for the planner's plan decision (#621, Test A).

INVARIANT BEING PINNED
----------------------
barlink_matrix.py ~line 785 (``_quant`` docstring):
  "The plan must come out bit-identical on every rank."

The raw measurement values differ between ranks by floating-point noise;
``_quant`` rounds them to 3 decimals so that ``plan_collective`` receives
identical inputs on every rank.  A decision threshold that flips on the
12th decimal digit is a random-number generator, not a planner.

HOW _QUANT IS ACTUALLY CALLED
-----------------------------
``_quant`` is called during the measurement phase in
``BarlinkMatrixPlanner._run()`` (lines 1916-2035 of barlink_matrix.py),
NOT inside ``plan_collective``.  The Measurement values passed to
``plan_collective`` are already quantized.  ``plan_collective`` itself is
purely computational with no I/O and does not call ``_quant``.

This test therefore uses synthetic Measurement objects to verify:
  1. The planner is deterministic (identical inputs -> identical plans).
  2. Quantized values that round to the same number produce the same plan.
  3. Unquantized values that differ (even at the 12th decimal) CAN produce
     different plans, proving the test is not vacuous and that ``_quant``
     is what prevents the divergence.

FINDING: the planner's ring_order tie-breaking is EXTREMELY sensitive:
when all edges have equal capacity, a 1e-12 decrease on one edge flips
the greedy nearest-neighbour ordering because it breaks the symmetric
tie.  An equal increase does NOT flip the order (the bidirectional
min is still capped by the peer's inbound rate).  This asymmetry means
that floating-point noise on tied edges is a real risk, and ``_quant``
eliminates it by collapsing symmetric edges to the exact same value.

CPU-only, no GPU, no torch.distributed.
"""

from __future__ import annotations

import unittest

import sglang.srt.distributed.device_communicators.barlink_matrix as matrix_mod
from sglang.srt.distributed.device_communicators.barlink_matrix import (
    BarlinkConfig,
    Measurement,
    plan_collective,
)
from sglang.test.test_utils import CustomTestCase


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _synthetic_measurement(
    world: int,
    sizes: tuple[int, ...],
    rates: float = 6.67,
    edge_override: dict[tuple[int, int], float] | None = None,
) -> Measurement:
    """Build a deterministic Measurement for testing.

    All edges default to ``rates`` GB/s.  Specific edges can be overridden
    via ``edge_override`` (keys are ``(src, dst)`` tuples).

    BDFs and names are synthetic placeholders; only the numerical values
    matter for the planner decision.
    """
    m = Measurement(
        world=world,
        sizes=sizes,
        bdfs=tuple(f"0000:0{r}:00.0" for r in range(world)),
        names=tuple(f"gpu{r}" for r in range(world)),
        sensor="synthetic",
    )
    for r in range(world):
        m.outbound[r] = [rates] * len(sizes)
        m.inbound[r] = [rates] * len(sizes)
        m.latency_s[r] = 0.001

    for (src, dst), val in (edge_override or {}).items():
        m.edge[(src, dst)] = [val] * len(sizes)

    return m


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_WORLD = 3
_SIZES = (1_000_000,)  # 1 MiB
_DEFAULT_RATE = 6.67  # GB/s -- symmetric capacity creates tie-breaks


def _base_measurement() -> Measurement:
    """All edges at identical capacity (symmetric tie-break scenario)."""
    return _synthetic_measurement(_WORLD, _SIZES, rates=_DEFAULT_RATE)


def _base_config() -> BarlinkConfig:
    return BarlinkConfig()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestPlanBitIdentity(CustomTestCase):
    """The planner must emit the same plan for identical inputs."""

    def test_identical_inputs_identical_plan(self) -> None:
        """``plan_collective`` is purely computational; same input -> same plan.

        Call it twice with the exact same Measurement and BarlinkConfig and
        assert the returned ``Plan.checksum()`` values are equal.
        """
        m = _base_measurement()
        k = _base_config()

        plan1 = plan_collective(m, k)
        plan2 = plan_collective(m, k)

        self.assertEqual(
            plan1.checksum(),
            plan2.checksum(),
            (
                "plan_collective is non-deterministic with identical inputs -- "
                "the plan must come out bit-identical on every rank"
            ),
        )

    def test_quantized_inputs_agree(self) -> None:
        """Two measurements whose perturbation rounds away still agree.

        After ``_quant`` rounds to 3 decimals, two ranks whose raw
        measurements differ by floating-point noise converge to the same
        quantized values.  We simulate this by providing both measurements
        with the same quantized edge value (6.670, which is what any value
        in [6.6695, 6.6705) rounds to).
        """
        m_base = _base_measurement()
        # A rank that measured 6.670000000001 would round to 6.670.
        # In the measurement pipeline, _quant(6.670000000001) -> 6.670,
        # which equals _quant(6.67) -> 6.670.
        m_perturbed_quantized = _synthetic_measurement(
            _WORLD, _SIZES, rates=_DEFAULT_RATE,
            edge_override={(0, 1): 6.670},  # already quantized
        )
        k = _base_config()

        plan_base = plan_collective(m_base, k)
        plan_perturbed = plan_collective(m_perturbed_quantized, k)

        self.assertEqual(
            plan_base.checksum(),
            plan_perturbed.checksum(),
            (
                "Plans diverged despite identical quantized inputs -- "
                "the _quant rounding invariant is broken. Two ranks that "
                "measured slightly different values should agree after "
                "_quant collapses them to the same rounded value."
            ),
        )

    def test_unquantized_tie_break_diverges(self) -> None:
        """THE CAN-FAIL PROOF: unquantized inputs can flip the plan.

        With symmetric edges all at 6.67 GB/s, the ring_order tie-breaking
        is deterministic (by PCI proximity, then rank index).  A decrease
        on just one edge (0->1) breaks the symmetry and flips the greedy
        nearest-neighbour ordering: (0,1,2) -> (0,2,1).

        This happens even at the 12th decimal: a rate of 6.669999999999
        (decreased by 1e-12) is enough because the ring_order uses
        min(capacity(src,dst), capacity(dst,src)) for tie-breaking, and
        the decrease on one direction makes min(src->dst, dst->src) <
        the symmetric edges.

        The monkeypatch of ``_quant`` to identity documents the invariant
        (in the real measurement pipeline, _quant is what rounds these
        values before they reach plan_collective).  Here we construct the
        Measurement directly with un-quantized values to simulate what
        would happen if _quant were bypassed.
        """
        m_base = _base_measurement()
        # Decrease edge 0->1 by 1e-12 to break the symmetric tie.
        # In the real pipeline, _quant would round this to 6.670,
        # restoring the tie.  Without _quant, the tie breaks.
        m_perturbed = _synthetic_measurement(
            _WORLD, _SIZES, rates=_DEFAULT_RATE,
            edge_override={(0, 1): _DEFAULT_RATE - 1e-12},
        )
        k = _base_config()

        # Monkeypatch _quant to identity -- in the real measurement pipeline
        # this would mean raw values reach plan_collective without rounding.
        # Note: plan_collective itself does not call _quant (it is called
        # during measurement collection), so the monkeypatch is a
        # documentation of the invariant rather than a mechanical dependency.
        original_quant = matrix_mod._quant
        try:
            matrix_mod._quant = lambda x: float(x)  # type: ignore[assignment]

            plan_base = plan_collective(m_base, k)
            plan_perturbed = plan_collective(m_perturbed, k)

            self.assertNotEqual(
                plan_base.checksum(),
                plan_perturbed.checksum(),
                (
                    "Plans did NOT diverge when _quant is disabled and inputs "
                    "differ by 1e-12 on one edge (symmetric tie-break scenario). "
                    "This means the test is vacuous -- the perturbation does not "
                    "reach the planner decision boundary. The ring_order should "
                    "flip when a symmetric tie is broken."
                ),
            )
            # Additional assertion: the specific element that changed.
            self.assertNotEqual(
                plan_base.ring_order,
                plan_perturbed.ring_order,
                (
                    "The ring_order did not change despite a perturbed edge "
                    "in a symmetric scenario. Expected the greedy tie-breaking "
                    "to be affected by the asymmetry."
                ),
            )
        finally:
            matrix_mod._quant = original_quant


if __name__ == "__main__":
    unittest.main()
