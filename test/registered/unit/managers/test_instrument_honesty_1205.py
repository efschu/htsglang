"""#1205: three instruments that cannot measure what they claim.

None of these kills a boot. Each costs a DIAGNOSIS, which on a nine-boot
campaign is the same currency: the first line a post-mortem reads is one of
them, and all three currently answer a different question than their label.

E1 -- THE STALE-GATE BUSY PROBE IS A CONSTANT.
``_release_residents_for_cutover`` gated its "#719 STALE-GATE BLIND" alarm on

    _busy = bool(cc.write_queue or cc.load_queue or cc.ack_backup_queue)

``write_queue`` and ``load_queue`` are plain lists (``cache_controller.py``
:716-717) and are honestly falsy when empty. ``ack_backup_queue`` is a
``queue.Queue`` (``cache_controller.py:801``), and ``Queue`` defines NEITHER
``__bool__`` NOR ``__len__`` -- so the object is truthy at every depth,
including zero. Whenever storage is enabled the ``or`` chain therefore ends on
a constant ``True``.

That matters because the #861e fix stated its own arithmetic in the comment
directly above it -- *"count a zero-streak only while the controller reports
device-tier work in flight"* -- and that arithmetic was never implemented. The
only thing that shipped was the threshold 2 -> 4. The alarm that fired 24 times
on a healthy W37-D boot still fires on every zero; it just needs four in a row.

AND THE NAIVE REPAIR IS WORSE, which is why the fix is not "call qsize()".
The probe runs inside ``_release_residents_for_cutover``, which the seam calls
one statement after ``self._seam_drain_ms = self._quiesce_hicache(direction)``
(``phase_flip_runtime.py`` :11938 / :11949). New device-tier I/O is refused by
``hicache_seam_active`` and the old I/O has just been drained, so an honest
depth reading at that point is ~always 0 -- and a depth-only gate would make
the alarm CONSTANTLY SILENT. Constant-true is noise; constant-false is a false
all-clear, and the false all-clear is the direction that cost W37-C. So the
traffic term is taken from the heartbeat's OWN history instead: if this process
has ever reported ``checked>0``, the gate is reachable on this workload, and a
subsequent run of zeroes is evidence of blindness rather than of idleness.

E2 -- THE POOL CENSUS LABELS NAME THE WRONG POPULATIONS.
``cur_slot_reqs`` carries ``len(_live_reqs(scheduler))``, which is the WIDE
count across every microbatch slot -- the exact opposite of "current slot".
``resident_reqs`` sums ``len(mb.reqs)`` over ``running_mbs`` with no dedup,
while ``_live_reqs`` dedups by ``id()``, so it is a count of LIST ENTRIES and
not a count of requests at all. In boot 9 this one line printed the whole
divergence (``resident_reqs=1/0/0``, ``resident_slots=[1]/[]/[]``) under labels
that make it unreadable.

E3 -- SEAM-REFUSED WRITES HAVE NO SENTINEL.
``write_backup`` has seven paths that return a bare 0, and the flip writeback
fence counts all seven as ``refused_silently``. One of them is the seam guard
(``cache_controller.py:1596`` -> ``hicache_phase_guard.device_tier_disarmed``),
which is the ONLY one whose count answers the seam-arm-ordering question. Its
own log line is once-per-direction-per-process by design, so after the first
flip it is silent. Without a distinct counter no instrument in the tree can
confirm or close that question.

No GPU, no flip, no scheduler: the shipped helpers and the shipped census
method are driven directly.
"""

import logging
import queue
import unittest
import unittest.mock
from types import SimpleNamespace

from sglang.srt.managers.phase_flip_runtime import (
    PhaseFlipRuntime,
    controller_device_queue_depth,
    parse_gate_heartbeat,
    stale_gate_zero_streak,
)
from sglang.srt.mem_cache import hicache_phase_guard
from sglang.srt.mem_cache.hicache_flip_writeback import FlipWritebackReport
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5)

STREAK_THRESHOLD = 4


def _controller(*, write=0, load=0, acks=0):
    """A controller stub shaped exactly like the real one's storage path."""
    ack_backup_queue: queue.Queue = queue.Queue()
    for i in range(acks):
        ack_backup_queue.put(i)
    return SimpleNamespace(
        # cache_controller.py:716-717 -- plain lists.
        write_queue=[object() for _ in range(write)],
        load_queue=[object() for _ in range(load)],
        # cache_controller.py:801 -- a queue.Queue.
        ack_backup_queue=ack_backup_queue,
    )


class TestTheOldBusyProbeCouldNotFail(CustomTestCase):
    """E1, direction 1: pin the defect, then pin that the new form escapes it."""

    def test_an_empty_Queue_is_truthy_which_is_the_whole_defect(self):
        empty: queue.Queue = queue.Queue()
        self.assertTrue(bool(empty))
        self.assertFalse(hasattr(queue.Queue, "__bool__"))
        self.assertFalse(hasattr(queue.Queue, "__len__"))
        self.assertEqual(empty.qsize(), 0)

    def test_the_shipped_or_chain_is_a_constant_on_a_fully_idle_controller(self):
        """The exact shipped expression, on a controller holding nothing."""
        cc = _controller(write=0, load=0, acks=0)
        shipped = bool(
            getattr(cc, "write_queue", None)
            or getattr(cc, "load_queue", None)
            or getattr(cc, "ack_backup_queue", None)
        )
        self.assertTrue(shipped, "the shipped probe cannot return False")

    def test_the_replacement_reports_zero_on_that_same_controller(self):
        cc = _controller(write=0, load=0, acks=0)
        self.assertEqual(controller_device_queue_depth(cc), 0)


class TestTheDepthProbeCountsInsteadOfTesting(CustomTestCase):
    """E1, direction 2: it must be a DEPTH, and unmeasured must not read 0."""

    def test_it_sums_all_three_queues(self):
        cc = _controller(write=2, load=3, acks=5)
        self.assertEqual(controller_device_queue_depth(cc), 10)

    def test_a_queue_only_controller_is_read_through_qsize(self):
        cc = SimpleNamespace(ack_backup_queue=queue.Queue())
        cc.ack_backup_queue.put(1)
        self.assertEqual(controller_device_queue_depth(cc), 1)

    def test_a_controller_with_no_readable_queue_reports_UNMEASURED(self):
        """None, never 0: #872's probe failure is not repeated here."""
        self.assertIsNone(controller_device_queue_depth(SimpleNamespace()))
        self.assertIsNone(controller_device_queue_depth(None))


class TestGateHeartbeatParsing(CustomTestCase):
    def test_it_reads_the_shipped_heartbeat_form(self):
        self.assertEqual(parse_gate_heartbeat("checked=7 refused=2"), (7, 2))

    def test_a_zero_heartbeat_is_a_measured_zero(self):
        self.assertEqual(parse_gate_heartbeat("checked=0 refused=0"), (0, 0))

    def test_an_unreadable_heartbeat_is_not_a_zero(self):
        self.assertEqual(parse_gate_heartbeat("controller went away"), (None, None))
        self.assertEqual(parse_gate_heartbeat(None), (None, None))


class TestTheStreakCountsBlindnessAndNotIdleness(CustomTestCase):
    """E1, direction 3: both failure directions of the alarm, explicitly."""

    def test_a_genuinely_idle_instance_never_accumulates_a_streak(self):
        """The #861e false positive: 24 fires on a healthy boot.

        Nothing has ever been checked and no work is queued, so every zero is
        the CORRECT reading and none of them is evidence of blindness.
        """
        streak = 0
        for _ in range(10):
            streak = stale_gate_zero_streak(
                streak, checked=0, depth=0, ever_checked=False
            )
        self.assertEqual(streak, 0)

    def test_a_workload_that_HAS_reached_the_gate_makes_zeroes_evidence(self):
        """THE DANGEROUS DIRECTION: the alarm must still be able to fire.

        Depth is 0 at every one of these cutovers, because the probe runs one
        statement after the seam drain. A gate keyed on depth alone would be
        silent here forever -- a false all-clear, which is exactly the W37-C
        failure the alarm exists to catch.
        """
        streak = 0
        streak = stale_gate_zero_streak(streak, checked=3, depth=0, ever_checked=False)
        self.assertEqual(streak, 0, "a reached gate resets the streak")
        for expected in (1, 2, 3, 4):
            streak = stale_gate_zero_streak(
                streak, checked=0, depth=0, ever_checked=True
            )
            self.assertEqual(streak, expected)
        self.assertGreaterEqual(streak, STREAK_THRESHOLD)

    def test_queued_work_alone_also_makes_a_zero_evidence(self):
        streak = stale_gate_zero_streak(0, checked=0, depth=4, ever_checked=False)
        self.assertEqual(streak, 1)

    def test_a_reached_gate_resets_a_running_streak(self):
        self.assertEqual(
            stale_gate_zero_streak(3, checked=1, depth=0, ever_checked=True), 0
        )

    def test_an_unreadable_heartbeat_neither_counts_nor_clears(self):
        self.assertEqual(
            stale_gate_zero_streak(3, checked=None, depth=None, ever_checked=True), 3
        )

    def test_an_unmeasured_depth_does_not_read_as_queued_work(self):
        self.assertEqual(
            stale_gate_zero_streak(0, checked=0, depth=None, ever_checked=False), 0
        )


class _Pages:
    """Stands in for the allocator's id tensors; only `.tolist()` is used."""

    def __init__(self, ids):
        self._ids = list(ids)

    def tolist(self):
        return list(self._ids)


def _census_line(running_mbs):
    """Drive the real `_pool_census` and return the line it logged."""
    alloc = SimpleNamespace(
        size=100,
        free_pages=_Pages(range(1, 51)),
        release_pages=_Pages([]),
        page_size=1,
        residency_withheld_slots=0,
        available_size=lambda: 50,
    )
    tree = SimpleNamespace(all_values_flatten=lambda: _Pages(range(51, 101)))
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=alloc,
        tree_cache=tree,
        running_mbs=running_mbs,
        running_batch=None,
        last_batch=None,
        phase_flip_stacks=None,
        tp_worker=None,
    )
    stub = SimpleNamespace(_census_scheduler=scheduler)
    stub._owner_ident = PhaseFlipRuntime._owner_ident
    stub._owner_pool_of = PhaseFlipRuntime._owner_pool_of
    stub._census_owner_probe = lambda *a, **k: None
    stub._pool_census = PhaseFlipRuntime._pool_census.__get__(stub, SimpleNamespace)

    logger = logging.getLogger("sglang.srt.managers.phase_flip_runtime")
    with unittest.mock.patch.object(logger, "warning") as warn:
        stub._pool_census("at-arm", "pp_to_tp")
    assert warn.called, "the census must always emit"
    args = warn.call_args[0]
    return args[0] % tuple(args[1:])


class TestCensusLabelsNameTheirPopulations(CustomTestCase):
    """E2: one request, two slot entries, and the line must say which is which."""

    def test_one_request_in_two_slots_prints_one_request_and_two_entries(self):
        """The discriminator. `_live_reqs` dedups by id(); the slot sum does not.

        Under the old labels this line read `cur_slot_reqs=1 resident_reqs=2`:
        a "current slot" count that is the wide count, and a "requests" count
        that is a number of list entries.
        """
        req = SimpleNamespace(rid="r0")
        line = _census_line([SimpleNamespace(reqs=[req]), SimpleNamespace(reqs=[req])])
        self.assertIn("live_reqs=1", line)
        self.assertIn("resident_slot_entries=2", line)
        self.assertIn("resident_slots=[0, 1]", line)

    def test_the_misleading_labels_are_gone_from_the_line(self):
        line = _census_line([SimpleNamespace(reqs=[SimpleNamespace(rid="r0")])])
        self.assertNotIn("cur_slot_reqs=", line)
        self.assertNotIn("resident_reqs=", line)

    def test_an_empty_rank_still_prints_both_terms(self):
        line = _census_line([])
        self.assertIn("live_reqs=0", line)
        self.assertIn("resident_slot_entries=0", line)


class _Runtime:
    """A weakref-able stand-in for PhaseFlipRuntime (SimpleNamespace is not)."""

    def __init__(self, *, hicache_seam_active, phase):
        self.hicache_seam_active = hicache_seam_active
        self.phase = phase


class TestSeamRefusalSentinel(CustomTestCase):
    """E3: the seam guard's refusals must be countable after the first one."""

    def setUp(self):
        hicache_phase_guard.clear_flip_phase_authority()
        hicache_phase_guard.reset_seam_refusals()
        hicache_phase_guard._WARNED.clear()

    def tearDown(self):
        hicache_phase_guard.clear_flip_phase_authority()
        hicache_phase_guard.reset_seam_refusals()
        hicache_phase_guard._WARNED.clear()

    def test_every_seam_refusal_is_counted_not_just_the_first_logged_one(self):
        runtime = _Runtime(hicache_seam_active=True, phase="pp")
        hicache_phase_guard.register_flip_phase_authority(runtime)
        for _ in range(3):
            self.assertTrue(hicache_phase_guard.device_tier_disarmed("write"))
        hicache_phase_guard.device_tier_disarmed("load")
        self.assertEqual(hicache_phase_guard.seam_refusals("write"), 3)
        self.assertEqual(hicache_phase_guard.seam_refusals("load"), 1)
        self.assertEqual(hicache_phase_guard.seam_refusals(), 4)

    def test_a_refusal_outside_the_seam_is_not_counted_as_one(self):
        runtime = _Runtime(hicache_seam_active=False, phase="tp")
        hicache_phase_guard.register_flip_phase_authority(runtime)
        hicache_phase_guard.device_tier_disarmed("write")
        self.assertEqual(hicache_phase_guard.seam_refusals("write"), 0)

    def test_the_report_prints_the_sentinel_and_spells_unmeasured_as_such(self):
        common = dict(
            eligible=1,
            staged=0,
            already_staged=0,
            acknowledged=0,
            outstanding=0,
            elapsed_s=0.0,
            deadline_s=1.0,
            refused_silently=1,
        )
        self.assertIn(
            "seam_refused=1", FlipWritebackReport(**common, seam_refused=1).as_log()
        )
        self.assertIn(
            "seam_refused=0", FlipWritebackReport(**common, seam_refused=0).as_log()
        )
        self.assertIn("seam_refused=?", FlipWritebackReport(**common).as_log())


if __name__ == "__main__":
    unittest.main()
