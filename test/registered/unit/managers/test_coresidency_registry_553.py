"""#553 Cut 1: the bridge between the two "who can give bytes back" registries.

ANALYSE_553 §3 calls this "the cut everything else addresses": `vram_dial`'s
`DialParticipant` list and #286's asset classes are the same question asked
twice, and today neither can see the other — nothing in the tree imports both
— so a hot/cold event has no addressee.

WHAT THESE PINS PROTECT, in the order that getting them wrong would hurt:

  1. **Refusal is carried, not filtered.** A source that exists but may not be
     drawn on comes back as an `Unavailable` WITH its reason. "Nothing can
     give bytes" and "three things could but none may" are different states,
     and a caller that cannot tell them apart will misreport the rig.
  2. **VA stability is asked, never re-derived.** #93/#468: a class whose
     addresses a captured graph holds cannot be parked, and `graph_addressed`
     flips that answer for classes that acquire the requirement from the
     ROUTE. The register owns that rule; this module must consult it.
  3. **No silent partial** (#268). `plan_for` returns None rather than a
     short list. A caller that receives fewer bytes than it asked for and
     proceeds is the failure the None exists to prevent.
  4. **No invented numbers.** Neither registry publishes a byte figure, so
     with no probe supplied a source is refused by name rather than assumed
     empty or assumed plentiful — both guesses hide.

Hermetic: pure data in, ranked view out. No CUDA, no registries touched.
"""

import unittest

from sglang.srt.managers.coresidency_registry import (
    ORIGIN_ASSET,
    ORIGIN_DIAL,
    ReclaimSource,
    enumerate_reclaim_sources,
)


class _Participant:
    def __init__(self, is_target=True):
        self.is_target = is_target


class _Descriptor:
    """Stands in for AssetClassDescriptor's one relevant question."""

    def __init__(self, pinned=False, pinned_when_graph=False, raises=False):
        self._pinned = pinned
        self._pinned_when_graph = pinned_when_graph
        self._raises = raises

    def va_stability_required(self, *, graph_addressed=False):
        if self._raises:
            raise RuntimeError("descriptor is confused")
        return self._pinned or (graph_addressed and self._pinned_when_graph)


def _view(**kw):
    kw.setdefault("dial_participants", [])
    kw.setdefault("asset_classes", {})
    return enumerate_reclaim_sources(**kw)


class TestBothRegistriesReachOneAnswer(unittest.TestCase):
    def test_a_dial_participant_and_an_asset_class_land_in_one_list(self):
        view = _view(
            dial_participants=[_Participant()],
            asset_classes={"experts": _Descriptor()},
            dial_reclaimable_bytes=lambda p: 4096,
            asset_reclaimable_bytes=lambda n, d: 1024,
        )
        origins = {s.origin for s in view.available}
        self.assertEqual(origins, {ORIGIN_DIAL, ORIGIN_ASSET})
        self.assertEqual(view.total_reclaimable_bytes, 5120)

    def test_the_dial_is_ranked_ahead_of_asset_parks(self):
        """Ordering is a claim about COST: returning VMM pages inside the band
        is cheaper than parking a class, so it goes first."""
        view = _view(
            dial_participants=[_Participant()],
            asset_classes={"experts": _Descriptor()},
            dial_reclaimable_bytes=lambda p: 1,
            asset_reclaimable_bytes=lambda n, d: 1_000_000,
        )
        self.assertEqual(view.available[0].origin, ORIGIN_DIAL)

    def test_the_order_is_total_and_not_dict_iteration_order(self):
        view = _view(
            asset_classes={
                "b": _Descriptor(),
                "a": _Descriptor(),
                "c": _Descriptor(),
            },
            asset_reclaimable_bytes=lambda n, d: 100,
        )
        self.assertEqual([s.name for s in view.available], ["a", "b", "c"])


class TestRefusalIsCarriedNotFiltered(unittest.TestCase):
    def test_a_va_pinned_class_is_reported_with_its_reason(self):
        view = _view(
            asset_classes={"experts": _Descriptor(pinned=True)},
            asset_reclaimable_bytes=lambda n, d: 4096,
        )
        self.assertEqual(view.available, ())
        self.assertEqual(len(view.unavailable), 1)
        self.assertIn("VA-stable", view.unavailable[0].reason)
        self.assertIn("capture", view.unavailable[0].reason)

    def test_empty_registries_differ_from_all_refused(self):
        """THE PIN behind the whole design: these two states must not look
        the same to a caller."""
        empty = _view()
        refused = _view(
            asset_classes={"experts": _Descriptor(pinned=True)},
            asset_reclaimable_bytes=lambda n, d: 4096,
        )
        self.assertEqual(empty.available, ())
        self.assertEqual(empty.unavailable, ())
        self.assertEqual(refused.available, ())
        self.assertTrue(refused.unavailable, "the refusal was silently dropped")

    def test_a_source_at_its_floor_says_so(self):
        view = _view(
            dial_participants=[_Participant()],
            dial_reclaimable_bytes=lambda p: 0,
        )
        self.assertEqual(view.available, ())
        self.assertIn("floor", view.unavailable[0].reason)

    def test_a_descriptor_that_raises_is_not_assumed_movable(self):
        view = _view(
            asset_classes={"broken": _Descriptor(raises=True)},
            asset_reclaimable_bytes=lambda n, d: 4096,
        )
        self.assertEqual(view.available, ())
        self.assertIn("not assumed movable", view.unavailable[0].reason)


class TestVaStabilityIsAskedNotDerived(unittest.TestCase):
    """#468: the same class flips answer depending on the ROUTE."""

    def test_a_route_acquired_pin_only_binds_when_graph_addressed(self):
        cls = {"experts": _Descriptor(pinned_when_graph=True)}
        eager = _view(
            asset_classes=cls,
            asset_reclaimable_bytes=lambda n, d: 4096,
            graph_addressed=False,
        )
        captured = _view(
            asset_classes=cls,
            asset_reclaimable_bytes=lambda n, d: 4096,
            graph_addressed=True,
        )
        self.assertEqual(len(eager.available), 1, "movable under the eager route")
        self.assertEqual(captured.available, (), "pinned once a graph holds it")


class TestNoSilentPartial(unittest.TestCase):
    def test_plan_for_returns_none_when_the_ask_cannot_be_funded(self):
        view = _view(
            dial_participants=[_Participant()],
            dial_reclaimable_bytes=lambda p: 1000,
        )
        self.assertIsNone(view.plan_for(5000))
        self.assertFalse(view.can_fund(5000))

    def test_plan_for_covers_the_ask_cheapest_first(self):
        view = _view(
            dial_participants=[_Participant()],
            asset_classes={"experts": _Descriptor()},
            dial_reclaimable_bytes=lambda p: 3000,
            asset_reclaimable_bytes=lambda n, d: 3000,
        )
        plan = view.plan_for(2000)
        self.assertEqual(len(plan), 1, "took more sources than the ask needed")
        self.assertEqual(plan[0].origin, ORIGIN_DIAL)

    def test_unavailable_bytes_never_count_toward_funding(self):
        """The silent-partial shape: a VA-pinned class must not make an ask
        look fundable."""
        view = _view(
            asset_classes={"experts": _Descriptor(pinned=True)},
            asset_reclaimable_bytes=lambda n, d: 1_000_000,
        )
        self.assertFalse(view.can_fund(1))
        self.assertIsNone(view.plan_for(1))

    def test_a_zero_ask_is_trivially_funded(self):
        self.assertEqual(_view().plan_for(0), ())


class TestNoInventedNumbers(unittest.TestCase):
    """Neither registry publishes a byte figure. With no probe, refuse."""

    def test_a_dial_participant_without_a_probe_is_refused_by_name(self):
        view = _view(dial_participants=[_Participant()])
        self.assertEqual(view.available, ())
        self.assertIn("will not invent", view.unavailable[0].reason)

    def test_an_asset_class_without_a_probe_is_refused_by_name(self):
        view = _view(asset_classes={"experts": _Descriptor()})
        self.assertEqual(view.available, ())
        self.assertIn("no reclaimable-bytes probe", view.unavailable[0].reason)

    def test_a_negative_reclaimable_is_a_registry_bug_not_a_small_budget(self):
        with self.assertRaises(ValueError):
            ReclaimSource(
                name="x", origin=ORIGIN_DIAL, reclaimable_bytes=-1, restorable=True
            )


class TestItReadsTheRealRegistriesByDefault(unittest.TestCase):
    """The defaults must point at the real thing, or this bridge is a toy."""

    def test_the_default_accessors_resolve(self):
        from sglang.srt.managers import coresidency_registry as m

        self.assertTrue(callable(m._default_dial_participants))
        self.assertTrue(callable(m._default_asset_classes))
        classes = m._default_asset_classes()
        self.assertTrue(classes, "the real ASSET_CLASSES table is empty")
        for descriptor in classes.values():
            self.assertTrue(hasattr(descriptor, "va_stability_required"))


if __name__ == "__main__":
    unittest.main()
