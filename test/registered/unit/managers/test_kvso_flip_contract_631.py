"""#656: kv-session-offload and the phase flip are a STATE question.

RED-FIRST NOTE. Before ``kvso_flip_contract`` existed,
``flip_blocking_guards`` appended "kv-session-offload" whenever the manager
was merely present, so :func:`test_guard_admits_an_idle_manager` and
:func:`test_guard_admits_parked_images_of_the_outgoing_phase` both failed on
the old tree -- a configured-but-idle kvso refused every flip. The three
refusal tests below failed the other way: there was no state to distinguish,
so a busy manager and an idle one produced the same verdict, and a test that
cannot tell them apart cannot see the hazard.
"""

import types
import unittest

from sglang.srt.managers.kvso_flip_contract import (
    STATE_ABSENT,
    STATE_BUSY,
    STATE_IDLE,
    STATE_PARKED,
    flip_safety_state,
    pin_spills_to_phase,
    restore_permitted,
    stamp_spill,
)


class _Slot:
    """A spill slot with the two fields the contract touches."""

    def __init__(self, stamp=None, suppress=False):
        self.flip_layout = stamp
        self.suppress_tick = suppress


class _Event:
    def __init__(self, done=True, raises=False):
        self._done = done
        self._raises = raises

    def query(self):
        if self._raises:
            raise RuntimeError("event unreadable")
        return self._done


def _manager(spills=None, copy_done=True, event_raises=False):
    return types.SimpleNamespace(
        spills=dict(spills or {}),
        backend=types.SimpleNamespace(
            _sess_wave_done=_Event(copy_done, event_raises)
        ),
    )


class TestFlipSafetyState(unittest.TestCase):
    def test_absent_manager_says_nothing(self):
        state, _ = flip_safety_state(None, current_phase="pp")
        self.assertEqual(state, STATE_ABSENT)

    def test_configured_but_empty_manager_is_idle(self):
        # THE REGRESSION THIS MODULE EXISTS FOR: kvso on, nothing spilled,
        # and the old guard still refused the flip.
        state, _ = flip_safety_state(_manager(), current_phase="pp")
        self.assertEqual(state, STATE_IDLE)

    def test_parked_images_of_the_live_phase_are_safe_to_carry(self):
        m = _manager({1: _Slot("pp"), 2: _Slot("pp")})
        state, detail = flip_safety_state(
            m, current_phase="pp", incoming_phase="tp"
        )
        self.assertEqual(state, STATE_PARKED)
        self.assertIn("2 spilled session", detail)

    def test_copy_in_flight_refuses(self):
        m = _manager({1: _Slot("pp")}, copy_done=False)
        state, detail = flip_safety_state(m, current_phase="pp")
        self.assertEqual(state, STATE_BUSY)
        self.assertIn("in flight", detail)

    def test_unreadable_copy_event_refuses(self):
        # An unanswerable question is not a yes.
        m = _manager({1: _Slot("pp")}, event_raises=True)
        state, _ = flip_safety_state(m, current_phase="pp")
        self.assertEqual(state, STATE_BUSY)

    def test_unstamped_image_refuses(self):
        m = _manager({7: _Slot(None)})
        state, detail = flip_safety_state(m, current_phase="pp")
        self.assertEqual(state, STATE_BUSY)
        self.assertIn("no layout stamp", detail)

    def test_image_stamped_with_the_incoming_phase_refuses(self):
        # The one case parked images are NOT safe: entering "tp" would make a
        # tp-stamped image restore-eligible across a layout change it did not
        # survive.
        m = _manager({3: _Slot("tp")})
        state, detail = flip_safety_state(
            m, current_phase="pp", incoming_phase="tp"
        )
        self.assertEqual(state, STATE_BUSY)
        self.assertIn("INCOMING", detail)


class TestTickPin(unittest.TestCase):
    def test_pin_suppresses_only_foreign_layouts(self):
        own, foreign = _Slot("tp"), _Slot("pp")
        m = _manager({1: own, 2: foreign})
        self.assertEqual(pin_spills_to_phase(m, "tp"), 1)
        self.assertFalse(own.suppress_tick)
        self.assertTrue(foreign.suppress_tick)

    def test_pin_is_re_applied_after_the_picker_clears_it(self):
        # suppress_tick is a ONE-SHOT the tick picker resets. A pin that were
        # set once would release itself on the first tick it prevented.
        foreign = _Slot("pp")
        m = _manager({2: foreign})
        pin_spills_to_phase(m, "tp")
        foreign.suppress_tick = False  # the picker consumed it
        self.assertEqual(pin_spills_to_phase(m, "tp"), 1)
        self.assertTrue(foreign.suppress_tick)

    def test_unstamped_slots_are_pinned(self):
        unstamped = _Slot(None)
        self.assertEqual(pin_spills_to_phase(_manager({1: unstamped}), "tp"), 1)
        self.assertTrue(unstamped.suppress_tick)


class TestRestorePermission(unittest.TestCase):
    def test_matching_phase_restores(self):
        self.assertTrue(restore_permitted(_Slot("tp"), "tp"))

    def test_foreign_phase_does_not_restore(self):
        self.assertFalse(restore_permitted(_Slot("pp"), "tp"))

    def test_unstamped_does_not_restore_under_a_flip(self):
        self.assertFalse(restore_permitted(_Slot(None), "tp"))

    def test_no_flip_means_always_permitted(self):
        # A process without the flip has ONE layout for its whole life. If a
        # missing phase read as "refuse", this guard would switch kvso's
        # restore path off for every user who never enabled the flip.
        self.assertTrue(restore_permitted(_Slot(None), None))
        self.assertTrue(restore_permitted(_Slot("pp"), ""))


class TestStamping(unittest.TestCase):
    def test_stamp_records_the_phase(self):
        s = _Slot()
        stamp_spill(s, "pp")
        self.assertEqual(s.flip_layout, "pp")

    def test_stamp_of_none_stays_unprovable(self):
        s = _Slot("pp")
        stamp_spill(s, None)
        self.assertIsNone(s.flip_layout)

    def test_a_slot_that_cannot_carry_a_stamp_does_not_raise(self):
        class Sealed:
            __slots__ = ()

        stamp_spill(Sealed(), "pp")  # warns, does not raise


class TestGuardWiring(unittest.TestCase):
    """The guard must consult the state, not the presence."""

    def _guards(self, kvso, phase="pp"):
        from sglang.srt.managers.phase_flip_runtime import flip_blocking_guards

        sched = types.SimpleNamespace(
            server_args=types.SimpleNamespace(
                enable_hierarchical_cache=False, dual_group_lane=None
            ),
            kv_session_offload=kvso,
            phase_flip_active_stack=phase,
            is_dual_group_lane=False,
            tree_cache=types.SimpleNamespace(all_values_flatten=lambda: []),
        )
        from sglang.srt.disaggregation.utils import DisaggregationMode

        sched.disaggregation_mode = DisaggregationMode.NULL
        return flip_blocking_guards(sched)

    def test_guard_admits_an_idle_manager(self):
        self.assertEqual(self._guards(_manager()), [])

    def test_guard_admits_parked_images_of_the_outgoing_phase(self):
        self.assertEqual(self._guards(_manager({1: _Slot("pp")}), "pp"), [])

    def test_guard_refuses_a_busy_manager(self):
        guards = self._guards(_manager({1: _Slot("pp")}, copy_done=False))
        self.assertEqual(len(guards), 1)
        self.assertIn("kv-session-offload busy", guards[0])

    def test_guard_still_admits_when_kvso_is_off(self):
        self.assertEqual(self._guards(None), [])


if __name__ == "__main__":
    unittest.main()
