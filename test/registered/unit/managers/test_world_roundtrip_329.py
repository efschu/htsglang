"""#329 cut 2 -- the in-process round trip, hermetically falsified.

DESIGN_329 §8 cut 2 is "quiesce + snapshot + restore with NO membership
change", and its stated falsifier is round-trip identity of the state. These
tests cover the phase machine, the completeness gate and the rollback
guarantee; the byte-identity half needs a card and belongs to the window
falsifier named in the module docstring.

WHY THE SEAMS ARE FAKED HERE. Every card-touching step arrives as an injected
callable, so a planted asset omission MUST fail the gate and the gate must also
be able to pass -- the pattern ``session_handover.py`` established one tier
down. A test that needed a scheduler could not run either arm.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.world_roundtrip import (
    ASSET_CLASSES,
    Phase,
    Trigger,
    WorldRoundTrip,
    WorldRoundTripError,
    _Seams,
    validate_roundtrip_completeness,
)
from sglang.test.test_utils import CustomTestCase

FULL = {a.name: f"blob:{a.name}" for a in ASSET_CLASSES}


class _Rig:
    """A fake world. Records what the phase machine did to it."""

    def __init__(self, *, members=(0, 1, 2), in_flight=0, drains=True, manifest=None):
        self.members = list(members)
        self.in_flight = in_flight
        self.drains = drains
        self._manifest = dict(FULL if manifest is None else manifest)
        self.paused_reason = None
        self.admission_open = True
        self.drafter_parked = False
        self.graphs_restored = 0
        self.read_with = None

    def seams(self):
        return _Seams(
            live_membership=lambda: self.members,
            in_flight_count=lambda: self.in_flight,
            pause_admission=self._pause,
            resume_admission=self._resume,
            drain_to_boundary=lambda _d: self.drains,
            write_snapshot=lambda: dict(self._manifest),
            read_snapshot=self._read,
            restore_graphs=self._restore_graphs,
            park_drafter=self._park,
            unpark_drafter=self._unpark,
        )

    def _pause(self, reason):
        self.paused_reason = reason
        self.admission_open = False

    def _resume(self):
        self.admission_open = True

    def _read(self, manifest):
        self.read_with = manifest

    def _restore_graphs(self):
        self.graphs_restored += 1

    def _park(self):
        self.drafter_parked = True

    def _unpark(self):
        self.drafter_parked = False


def _trip(rig, trigger=Trigger.OPERATOR):
    return WorldRoundTrip(rig.seams(), trigger=trigger)


class TestTheHappyRoundTrip(CustomTestCase):
    def test_all_four_phases_run_in_order(self):
        rig = _Rig()
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        t.snapshot()
        t.restore()
        t.resume()
        self.assertEqual(
            t.history,
            (Phase.QUIESCE, Phase.SNAPSHOT, Phase.RESTORE, Phase.RESUME),
        )

    def test_admission_closes_then_reopens(self):
        rig = _Rig()
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        self.assertFalse(rig.admission_open)
        t.snapshot()
        t.restore()
        t.resume()
        self.assertTrue(rig.admission_open)

    def test_the_pause_reason_names_the_trigger(self):
        """An operator reading 'why is this refusing traffic' must find the
        answer in the refusal, not in a changelog."""
        rig = _Rig()
        _trip(rig, Trigger.PLANNER).quiesce(deadline_s=1.0)
        self.assertIn("#329", rig.paused_reason)
        self.assertIn("planner", rig.paused_reason)

    def test_the_drafter_is_parked_and_unparked(self):
        rig = _Rig()
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        self.assertTrue(rig.drafter_parked)
        t.snapshot()
        t.restore()
        t.resume()
        self.assertFalse(rig.drafter_parked)

    def test_restore_is_handed_the_snapshot_that_was_taken(self):
        rig = _Rig()
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        manifest = t.snapshot()
        t.restore()
        self.assertEqual(rig.read_with, manifest)

    def test_graphs_come_back_exactly_once(self):
        rig = _Rig()
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        t.snapshot()
        t.restore()
        t.resume()
        self.assertEqual(rig.graphs_restored, 1)


class TestMembershipIsHeldFixed(CustomTestCase):
    """The line between cut 2 and cut 1. Cut 1 (live communicator
    teardown/rebuild) is UNMEASURED, so this cut must refuse rather than
    perform four fifths of a membership change."""

    def test_a_different_target_membership_is_refused_at_quiesce(self):
        rig = _Rig(members=(0, 1, 2))
        with self.assertRaises(WorldRoundTripError) as cm:
            _trip(rig).quiesce(target_membership=(0, 1), deadline_s=1.0)
        self.assertIn("cut 2", str(cm.exception))

    def test_the_refusal_names_both_member_sets(self):
        rig = _Rig(members=(0, 1, 2))
        with self.assertRaises(WorldRoundTripError) as cm:
            _trip(rig).quiesce(target_membership=(0, 1), deadline_s=1.0)
        self.assertIn("[0, 1, 2]", str(cm.exception))
        self.assertIn("[0, 1]", str(cm.exception))

    def test_an_identical_target_membership_is_fine(self):
        """Passing the same set is not an error -- it is a caller stating an
        expectation, which is the useful thing to allow."""
        rig = _Rig(members=(0, 1, 2))
        _trip(rig).quiesce(target_membership=[0, 1, 2], deadline_s=1.0)

    def test_membership_moving_underneath_is_caught_at_restore(self):
        """A member that vanishes mid-trip means the geometry the snapshot
        describes is no longer the one it would go back into."""
        rig = _Rig(members=(0, 1, 2))
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        t.snapshot()
        rig.members = [0, 1]
        with self.assertRaises(WorldRoundTripError) as cm:
            t.restore()
        self.assertIn("changed underneath", str(cm.exception))


class TestTheCompletenessGate(CustomTestCase):
    """THE FALSIFIER. A planted omission must fail, and the gate must also be
    able to pass -- a gate that only ever fails proves nothing either."""

    def test_a_full_manifest_passes(self):
        validate_roundtrip_completeness(FULL)

    def test_every_single_class_omission_is_caught(self):
        for cls in ASSET_CLASSES:
            if not cls.required:
                continue
            with self.subTest(omitted=cls.name):
                planted = {k: v for k, v in FULL.items() if k != cls.name}
                with self.assertRaises(WorldRoundTripError) as cm:
                    validate_roundtrip_completeness(planted)
                self.assertIn(cls.name, str(cm.exception))

    def test_an_explicitly_empty_class_is_accepted(self):
        """ "This world has no GDN state" is a fact. Forcing a fake blob to
        satisfy a gate teaches everyone to fake blobs."""
        validate_roundtrip_completeness({**FULL, "gdn_state": None})

    def test_snapshot_refuses_an_incomplete_manifest(self):
        rig = _Rig(manifest={k: v for k, v in FULL.items() if k != "gdn_state"})
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        with self.assertRaises(WorldRoundTripError):
            t.snapshot()

    def test_the_recurrent_state_class_carries_the_212_reason(self):
        """#212: GDN state is positional and not prefix-shareable, so losing it
        yields WRONG output rather than slow output. The reason belongs next to
        the entry, or the next person drops it as redundant with KV."""
        gdn = next(a for a in ASSET_CLASSES if a.name == "gdn_state")
        self.assertIn("#212", gdn.why)

    def test_the_non_persistent_buffer_class_names_the_general_rule(self):
        """#568's fix is a RULE over the persistence property (iterate
        named_buffers, carry what state_dict omits), not a list of names. The
        entry says so, so a third implementation cannot regress to state_dict
        and look correct."""
        buf = next(a for a in ASSET_CLASSES if a.name == "non_persistent_buffers")
        self.assertIn("named_buffers", buf.why)
        self.assertIn("state_dict", buf.why)


class TestQuiesceRefusesRatherThanWedges(CustomTestCase):
    def test_a_failed_drain_reopens_admission(self):
        """The worst outcome of a failed quiesce is a world that refuses
        traffic forever because the rollback was forgotten."""
        rig = _Rig(in_flight=3, drains=False)
        t = _trip(rig)
        with self.assertRaises(WorldRoundTripError):
            t.quiesce(deadline_s=0.01)
        self.assertTrue(rig.admission_open)

    def test_a_failed_drain_leaves_the_machine_stable(self):
        rig = _Rig(drains=False)
        t = _trip(rig)
        with self.assertRaises(WorldRoundTripError):
            t.quiesce(deadline_s=0.01)
        self.assertIs(t.phase, Phase.STABLE)

    def test_the_drain_refusal_names_the_in_flight_count(self):
        rig = _Rig(in_flight=7, drains=False)
        with self.assertRaises(WorldRoundTripError) as cm:
            _trip(rig).quiesce(deadline_s=0.01)
        self.assertIn("7", str(cm.exception))


class TestPhaseOrderIsEnforced(CustomTestCase):
    def test_snapshot_before_quiesce_is_refused(self):
        with self.assertRaises(WorldRoundTripError):
            _trip(_Rig()).snapshot()

    def test_restore_before_snapshot_is_refused(self):
        rig = _Rig()
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        with self.assertRaises(WorldRoundTripError):
            t.restore()

    def test_resume_before_restore_is_refused(self):
        rig = _Rig()
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        t.snapshot()
        with self.assertRaises(WorldRoundTripError):
            t.resume()


class TestRollbackIsFreeInThisCut(CustomTestCase):
    """In the full machine rollback dies at RE-FORM, because destroying the old
    communicators leaves nothing to return to. This cut has no RE-FORM, so the
    window is the WHOLE round trip. Asserted rather than enjoyed quietly: an
    edit that makes a phase destructive must break this."""

    def test_abort_is_legal_from_every_phase_before_resume(self):
        for stop_after in ("quiesce", "snapshot", "restore"):
            with self.subTest(phase=stop_after):
                rig = _Rig()
                t = _trip(rig)
                t.quiesce(deadline_s=1.0)
                if stop_after in ("snapshot", "restore"):
                    t.snapshot()
                if stop_after == "restore":
                    t.restore()
                t.abort("test")
                self.assertIs(t.phase, Phase.STABLE)
                self.assertTrue(rig.admission_open)

    def test_abort_after_resume_is_refused(self):
        rig = _Rig()
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        t.snapshot()
        t.restore()
        t.resume()
        with self.assertRaises(WorldRoundTripError):
            t.abort("too late")

    def test_abort_unparks_the_drafter(self):
        rig = _Rig()
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        t.abort("test")
        self.assertFalse(rig.drafter_parked)

    def test_an_aborted_trip_is_single_use(self):
        rig = _Rig()
        t = _trip(rig)
        t.quiesce(deadline_s=1.0)
        t.abort("test")
        with self.assertRaises(WorldRoundTripError):
            t.quiesce(deadline_s=1.0)


class TestNeverATransientFailureReflex(CustomTestCase):
    """DESIGN_329 §9. A round trip costs a maintenance window; firing one at a
    link hiccup is a self-inflicted outage. The type system carries the rule."""

    def test_the_trigger_vocabulary_has_no_failure_member(self):
        self.assertEqual(
            {t.value for t in Trigger},
            {"operator", "planner"},
            "a TRANSIENT_FAILURE trigger would make the reflex expressible, "
            "which DESIGN_329 section 9 forbids",
        )

    def test_a_string_trigger_is_refused(self):
        """So no caller can pass 'nccl_timeout' and have it read as a
        deliberate decision."""
        with self.assertRaises(WorldRoundTripError) as cm:
            WorldRoundTrip(_Rig().seams(), trigger="nccl_timeout")
        self.assertIn("section 9", str(cm.exception))

    def test_the_module_holds_no_self_trigger(self):
        """Nothing here may construct its own trigger from an error path."""
        import inspect

        from sglang.srt.managers import world_roundtrip

        src = inspect.getsource(world_roundtrip)
        self.assertNotIn("except Exception", src)


if __name__ == "__main__":
    unittest.main()
