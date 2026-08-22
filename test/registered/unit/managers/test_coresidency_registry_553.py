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


class TestCut2TheProbesAreWired(unittest.TestCase):
    """#553 Cut 2 first half: the bridge stops answering all-unavailable.

    Before this, every source came back refused with "no probe supplied".
    These pins are the transition -- a wired probe turns a specific source
    from refused-no-probe into ranked-with-bytes -- and, more importantly,
    that a probe which CANNOT MEASURE still refuses by name rather than
    ranking the source at zero (#606: zero is a measurement, a failed probe
    is the absence of one, and collapsing them removes a real source from an
    elastic plan while looking like it was considered).
    """

    def test_a_wired_dial_probe_turns_refusal_into_a_ranked_source(self):
        before = _view(dial_participants=[_Participant()])
        self.assertEqual(before.available, ())
        self.assertIn("will not invent", before.unavailable[0].reason)

        after = _view(
            dial_participants=[_Participant()],
            dial_reclaimable_bytes=lambda p: 8192,
        )
        self.assertEqual(len(after.available), 1)
        self.assertEqual(after.available[0].reclaimable_bytes, 8192)

    def test_an_unmeasurable_dial_probe_refuses_by_name(self):
        """THE #606 PIN. Not 0 bytes; a named refusal."""
        from sglang.srt.managers.coresidency_registry import ProbeUnavailable

        def _blind(participant):
            raise ProbeUnavailable("unbooted pool")

        view = _view(dial_participants=[_Participant()], dial_reclaimable_bytes=_blind)
        self.assertEqual(view.available, ())
        self.assertEqual(len(view.unavailable), 1)
        self.assertIn("unbooted pool", view.unavailable[0].reason)

    def test_an_unmeasurable_asset_probe_refuses_by_name(self):
        from sglang.srt.managers.coresidency_registry import ProbeUnavailable

        def _blind(name, descriptor):
            raise ProbeUnavailable("no offload register on this process")

        view = _view(
            asset_classes={"experts": _Descriptor()},
            asset_reclaimable_bytes=_blind,
        )
        self.assertEqual(view.available, ())
        self.assertIn("no offload register", view.unavailable[0].reason)

    def test_a_measured_zero_is_distinguishable_from_an_unmeasured_one(self):
        """The distinction the whole exception exists for."""
        measured = _view(
            dial_participants=[_Participant()], dial_reclaimable_bytes=lambda p: 0
        )
        self.assertIn("floor", measured.unavailable[0].reason)

        from sglang.srt.managers.coresidency_registry import ProbeUnavailable

        def _blind(p):
            raise ProbeUnavailable("unmeasurable row width")

        unmeasured = _view(
            dial_participants=[_Participant()], dial_reclaimable_bytes=_blind
        )
        self.assertIn("unmeasurable", unmeasured.unavailable[0].reason)
        self.assertNotEqual(
            measured.unavailable[0].reason, unmeasured.unavailable[0].reason
        )


class TestCut2TheDialProbeReadsLive(unittest.TestCase):
    """``vram_dial.reclaimable_bytes_for``: a live read, None when unknown."""

    class _Pool:
        def __init__(self, backed=None, rows_raise=False):
            self.full_pool_backed_rows = backed
            self._raise = rows_raise
            self.full_kv_pool = self

        @property
        def k_buffer(self):
            if self._raise:
                raise RuntimeError("no buffers")
            import torch

            return [torch.zeros(4, 8)]

        @property
        def v_buffer(self):
            import torch

            return [torch.zeros(4, 8)]

    def _probe(self, pool, floor):
        from sglang.srt.managers.vram_dial import reclaimable_bytes_for

        return reclaimable_bytes_for(type("P", (), {"pool": pool})(), floor)

    def test_no_floor_authority_is_none_not_zero(self):
        self.assertIsNone(self._probe(self._Pool(backed=100), None))

    def test_unreadable_backed_rows_is_none_not_zero(self):
        self.assertIsNone(self._probe(self._Pool(backed=None), 10))

    def test_unmeasurable_row_width_is_none_not_zero(self):
        self.assertIsNone(self._probe(self._Pool(backed=100, rows_raise=True), 10))

    def test_at_or_below_the_floor_is_a_measured_zero(self):
        self.assertEqual(self._probe(self._Pool(backed=10), 10), 0)
        self.assertEqual(self._probe(self._Pool(backed=5), 10), 0)

    def test_rows_above_the_floor_become_bytes(self):
        # one row = k(8 floats*4B) + v(8 floats*4B) = 64 B
        self.assertEqual(self._probe(self._Pool(backed=12), 10), 2 * 64)


class TestCut2TheRegisterProbeExcludesWhatItMustNot(unittest.TestCase):
    """``OffloadRegister.reclaimable_bytes``: resident AND not hot."""

    def _register(self, items):
        from sglang.srt.model_executor.offload_register import OffloadRegister

        reg = OffloadRegister.__new__(OffloadRegister)
        import threading

        reg._lock = threading.RLock()
        reg._items = {str(i): it for i, it in enumerate(items)}
        return reg

    def _item(self, cls="experts", size=100, parked=False, hot=False, hot_raises=False):
        def _hot():
            if hot_raises:
                raise RuntimeError("cannot answer")
            return hot

        return type(
            "I",
            (),
            {
                "offload_class": cls,
                "size_bytes": size,
                "parked": parked,
                "hot": staticmethod(_hot),
            },
        )()

    def test_resident_and_cold_counts(self):
        reg = self._register([self._item(size=100), self._item(size=50)])
        self.assertEqual(reg.reclaimable_bytes("experts"), 150)

    def test_parked_bytes_are_not_counted_twice(self):
        reg = self._register([self._item(size=100, parked=True), self._item(size=50)])
        self.assertEqual(reg.reclaimable_bytes("experts"), 50)

    def test_hot_items_are_excluded_because_park_refuses_them(self):
        reg = self._register([self._item(size=100, hot=True), self._item(size=50)])
        self.assertEqual(
            reg.reclaimable_bytes("experts"),
            50,
            "hot bytes are resident but NOT reclaimable; counting them hands "
            "the caller a figure it cannot spend",
        )

    def test_an_unanswerable_hotness_predicate_is_treated_as_hot(self):
        """Safe direction: refuse to reclaim, never assume free to move."""
        reg = self._register([self._item(size=100, hot_raises=True)])
        self.assertEqual(reg.reclaimable_bytes("experts"), 0)

    def test_other_classes_do_not_leak_in(self):
        reg = self._register([self._item(cls="other", size=999), self._item(size=50)])
        self.assertEqual(reg.reclaimable_bytes("experts"), 50)
