"""#800 x #799: the published wedge verdict must carry the recovery outcome.

THE SEAM, AND WHY IT IS A DEFECT AND NOT A MISSING FEATURE
----------------------------------------------------------
#799 carried the admission-wedge verdict out of the scheduler process, which
was the right edge to build: a log line has no consumer, and a supervisor
needs a machine-readable fact it can poll without asking the suspect server
anything.

#800 gave the recovery attempt a named outcome instead of a ``None`` that six
different exits produced.

The two did not meet, and the gap had a direction. In #799's poller the
publish ran BEFORE the recovery attempt:

    _publish(alarm, detail)          # <- the file is written here
    ...
    _attempt_recovery(age, threshold)  # <- the attempt happens here

so the published record could never contain that attempt's outcome. Not
"stale by one poll" -- structurally absent, because the payload had no field
for it either. A supervisor reading that file learns a rank is wedged and
cannot learn whether anything was tried or what came back.

THAT BLINDNESS IS NOT HYPOTHETICAL: it is how the 2026-08-22 diagnosis went
wrong. The conclusion drawn was "the gate is off or inert on this boot", about
a gate the same boot had logged ``ARMED on device 0`` for on all three ranks
and which had cleared four prefill admissions reclaiming 232 / 1112 / 1126 /
1216 MiB. The correcting fact existed inside the process and never left it.

WHAT THIS FILE GUARDS, and the mutant each test kills:

  * ``test_the_published_file_carries_the_settled_recovery_outcome`` -- the
    load-bearing one. Drives the REAL ``_poll_once`` and reads the REAL file.
    Red the moment the publish moves back in front of the step.
  * ``test_a_rank_that_never_tried_publishes_no_recovery_and_says_so`` -- the
    other direction. "Nothing tried" must be visibly different from "tried and
    fine". A can-fail that only checks the populated case would pass against a
    publisher that hardcodes a cheerful record.
  * the payload/reader tests -- additive-by-omission, and a malformed record
    dropped rather than forwarded.

Hermetic: no CUDA, no scheduler boot, no threads, temp dirs only.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers import wedge_status
from sglang.srt.managers.corridor_admission import (
    REASON_HEADROOM_SUFFICIENT,
    REASON_NO_GUARD,
    PrefillAdmissionGate,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    AdmissionWedgeRecovery,
    _rank_label,
    make_admission_wedge_poller,
)
from sglang.srt.managers.wedge_recovery import (
    STATE_INERT,
    STATE_NOT_APPLICABLE,
    get_recovery_channel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1024 * 1024


class _FakeVerdict:
    def __init__(self, ok=True, law_breached=False, reclaimed=0, detail="d"):
        self.ok = ok
        self.law_breached = law_breached
        self.reclaimed = reclaimed
        self.detail = detail


class _FakeGuard:
    def __init__(self, free_mib=4096, floor_mib=1331):
        self._free = free_mib * MIB
        self.floor_bytes = floor_mib * MIB
        self._verdict = _FakeVerdict()
        self.providers = ["allocator-cache"]
        self.delta_bytes = 256 * MIB
        self.device_index = 0

    def free_bytes(self):
        return self._free

    def ensure_headroom(self, want, reason="", refusal_is_fatal=False):
        return self._verdict


def _gate(guard):
    scheduler = SimpleNamespace(server_args=SimpleNamespace(enable_phase_flip=True))
    gate = PrefillAdmissionGate(scheduler, cooldown_s=0.0, clock=lambda: 0.0)
    gate._guard = lambda: guard
    gate._maybe_lend = lambda *a, **k: None
    gate._announce_once = lambda *a, **k: None
    return gate


class _FakeSessionController:
    def maybe_reap(self, now):
        return None


class _FakeFlushWrapper:
    def check_pending(self):
        return None


class _FakeScheduler:
    def __init__(self, *, queued=2, running=0, age=400.0):
        self.is_initializing = False
        self.waiting_queue = [object()] * queued
        self.running_batch = SimpleNamespace(reqs=[object()] * running)
        self.last_first_token_progress_time = -age
        self.server_args = SimpleNamespace(enable_phase_flip=True)
        self.session_controller = _FakeSessionController()
        self.flush_wrapper = _FakeFlushWrapper()
        self.external_corpus_manager = None
        self.return_health_check_ipcs = []


class _PrivateDir:
    """A temp directory that is NOT exported through the env knob.

    WHY THE DISTINCTION MATTERS, learned the hard way in this ticket. Exporting
    ``ENV_DIR`` does not merely tell THIS test where to write -- it redirects
    every publisher in the process, including any admission-wedge watchdog
    thread some earlier test left running. ``Scheduler.__init__`` starts one
    at scheduler.py:1889 and passes no ``stop``, so a test that builds a real
    Scheduler leaks a publisher for the life of the pytest process. That
    stray then writes ``wedge.pid<N>.json`` into the exported directory, and
    ``wedge.pid...`` sorts BEFORE ``wedge.pp1-tp0.json`` -- so a reader test
    asserting on "the wedged rank" got the stray's record instead of its own.
    Green alone, red in the full suite, and the assertion looked like a bug in
    the code under test rather than in the test.

    Tests that pass ``directory=`` explicitly therefore use THIS, which
    exports nothing and cannot be found by a stray.
    """

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="wedgeseam-")
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


class _StatusDir(_PrivateDir):
    """A temp status directory, installed via the module's own env knob.

    Only for tests that drive a publisher which resolves its own directory --
    i.e. the poller. Those must select their OWN rank's record out of the
    directory (see :func:`_record_for`) rather than trusting it to be alone in
    there, for the reason spelled out above.
    """

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="wedgeseam-")
        self._patch = mock.patch.dict(os.environ, {wedge_status.ENV_DIR: self.path})
        self._patch.start()
        return self.path

    def __exit__(self, *exc):
        self._patch.stop()
        return super().__exit__(*exc)


def _record_for(path, rank):
    """This rank's published record, or None. Never "whatever sorted first"."""
    name = os.path.join(path, f"wedge.{rank}.json")
    if not os.path.exists(name):
        return None
    with open(name) as fh:
        return json.load(fh)


def _listing(path):
    """Every wedge file present, for an assertion message that names strays."""
    return sorted(n for n in os.listdir(path) if n.startswith("wedge."))


# --- part 1: the payload is additive by omission --------------------------


class TestPayloadAndReader(CustomTestCase):
    def test_payload_omits_recovery_when_there_is_none(self):
        """Omitted, not null: an old reader and a new file must agree."""
        rec = json.loads(wedge_status._payload("pp0-tp0", True, "d", 1.0))
        self.assertNotIn("recovery", rec)

    def test_payload_carries_recovery_when_given(self):
        payload = {"state": STATE_NOT_APPLICABLE, "reason": REASON_HEADROOM_SUFFICIENT}
        rec = json.loads(wedge_status._payload("pp0-tp0", True, "d", 1.0, payload))
        self.assertEqual(rec["recovery"], payload)

    def test_reader_surfaces_the_wedged_ranks_recovery(self):
        with _PrivateDir() as d:
            wedge_status.publish_verdict(
                "pp0-tp0", False, "healthy", directory=d, recovery=None
            )
            wedge_status.publish_verdict(
                "pp1-tp0",
                True,
                "wedged",
                directory=d,
                recovery={
                    "state": STATE_INERT,
                    "reason": REASON_NO_GUARD,
                    "consecutive_non_actuating": 3,
                    "escalated": True,
                },
            )
            signal = wedge_status.read_wedge_signal(directory=d)
        self.assertIs(signal.verdict, True)
        self.assertIsNotNone(signal.recovery)
        self.assertEqual(signal.recovery["state"], STATE_INERT)
        self.assertEqual(signal.recovery["reason"], REASON_NO_GUARD)
        self.assertIn("ESCALATED", signal.detail)

    def test_reader_on_an_old_file_reports_no_recovery_not_a_healthy_one(self):
        """MUTANT KILLED: default the missing field to a benign record."""
        with _PrivateDir() as d:
            wedge_status.publish_verdict("pp1-tp0", True, "wedged", directory=d)
            signal = wedge_status.read_wedge_signal(directory=d)
        self.assertIs(signal.verdict, True)
        self.assertIsNone(signal.recovery)
        self.assertIn("no recovery attempt has been settled", signal.detail)

    def test_a_malformed_recovery_record_is_dropped_not_forwarded(self):
        with _PrivateDir() as d:
            path = os.path.join(d, "wedge.pp1-tp0.json")
            with open(path, "w") as fh:
                fh.write(
                    json.dumps(
                        {
                            "wedged": True,
                            "detail": "d",
                            "wall": 1.0e12,
                            "rank": "pp1-tp0",
                            "recovery": "not-a-dict",
                        }
                    )
                )
            signal = wedge_status.read_wedge_signal(
                directory=d, now=lambda: 1.0e12 + 1.0
            )
        self.assertIs(signal.verdict, True)
        self.assertIsNone(signal.recovery)

    def test_the_recovery_state_published_is_the_wedged_ranks_own(self):
        """Not a reduction across ranks -- the ranks DISAGREEING is the finding.

        On 2026-08-22 the stuck rank never reached the recovery threshold while
        a healthy one did. Averaging or merging the recovery state across ranks
        would have destroyed exactly that.

        MUTANT KILLED: merge/reduce recovery across ranks.
        """
        with _PrivateDir() as d:
            wedge_status.publish_verdict(
                "pp0-tp0",
                False,
                "healthy",
                directory=d,
                recovery={"state": STATE_NOT_APPLICABLE, "reason": "x"},
            )
            wedge_status.publish_verdict(
                "pp1-tp0",
                True,
                "wedged",
                directory=d,
                recovery={"state": STATE_INERT, "reason": REASON_NO_GUARD},
            )
            signal = wedge_status.read_wedge_signal(directory=d)
        self.assertEqual(signal.recovery["state"], STATE_INERT)


# --- part 2: the ordering, through the production callable ----------------


class TestTheOrderingIsTheFix(CustomTestCase):
    def test_the_published_file_carries_the_settled_recovery_outcome(self):
        """THE LOAD-BEARING TEST. Real poller, real publish, real file.

        Poll 1 posts the request. The scheduler thread drains it through the
        REAL ``Scheduler.process_input_requests``. Poll 2 settles it and must
        publish that settled outcome in the SAME poll.

        MUTANT KILLED: move ``_publish`` back in front of ``driver.step`` --
        #799's original order. The second poll then writes the state as it was
        before the settle, the file has no recovery record, and this is red.
        A test that only called ``publish_verdict`` directly would stay green
        through that mutation, which is why this one drives ``_poll_once``.
        """
        with _StatusDir() as d:
            sched = _FakeScheduler(queued=2, running=0, age=400.0)
            setattr(
                sched,
                "phase_flip_corridor_admission",
                _gate(_FakeGuard(free_mib=4096, floor_mib=1331)),
            )
            poll = make_admission_wedge_poller(sched)
            rank = _rank_label(sched)

            self.assertIs(poll(), True)  # posts request #1
            channel = get_recovery_channel(sched)
            self.assertEqual(channel.requested_seq, 1)
            first = _record_for(d, rank)
            self.assertIsNotNone(first, f"nothing published; dir={_listing(d)}")
            self.assertIsNone(
                first.get("recovery"),
                "nothing is settled yet, so nothing may be published",
            )

            Scheduler.process_input_requests(sched, [])  # scheduler thread acks

            self.assertIs(poll(), True)  # settles AND publishes in one poll
            record = _record_for(d, rank)

        self.assertIsNotNone(record)
        self.assertIn("recovery", record)
        self.assertEqual(record["recovery"]["state"], STATE_NOT_APPLICABLE)
        self.assertEqual(record["recovery"]["reason"], REASON_HEADROOM_SUFFICIENT)
        self.assertEqual(record["recovery"]["seq"], 1)

    def test_a_stray_publishers_file_cannot_be_mistaken_for_ours(self):
        """The test-hygiene defect this file hit, turned into a test.

        A watchdog thread left running by an earlier test publishes into
        whatever directory ``ENV_DIR`` currently names -- and ``wedge.pid<N>``
        sorts BEFORE ``wedge.pp1-tp0``. Taking "the first record in the
        directory" therefore reads a stranger's state and calls it ours: green
        alone, red in the full suite, and the failure points at the code under
        test rather than at the test.

        MUTANT KILLED: go back to ``records(d)[0]``.
        """
        with _StatusDir() as d:
            # A stray that sorts first and claims a completely different state.
            with open(os.path.join(d, "wedge.pid000001.json"), "w") as fh:
                fh.write(
                    json.dumps(
                        {
                            "wedged": True,
                            "detail": "not ours",
                            "wall": 1.0,
                            "rank": "pid000001",
                            "recovery": {"state": STATE_INERT, "reason": "stray"},
                        }
                    )
                )
            sched = _FakeScheduler(queued=2, running=0, age=400.0)
            setattr(
                sched,
                "phase_flip_corridor_admission",
                _gate(_FakeGuard(free_mib=4096, floor_mib=1331)),
            )
            poll = make_admission_wedge_poller(sched)
            rank = _rank_label(sched)
            poll()
            Scheduler.process_input_requests(sched, [])
            poll()
            record = _record_for(d, rank)
            self.assertIn("wedge.pid000001.json", _listing(d), "stray must be present")

        self.assertIsNotNone(record)
        self.assertEqual(record["rank"], rank)
        self.assertEqual(record["recovery"]["state"], STATE_NOT_APPLICABLE)
        self.assertNotEqual(record["recovery"].get("reason"), "stray")

    def test_a_rank_that_never_tried_publishes_no_recovery_and_says_so(self):
        """The other direction, and it must not be silent.

        A healthy rank publishes on every poll (that is #799's own rule, so
        staleness has something to measure against). It must publish NO
        recovery record, and the reader must say so in words rather than
        leaving the reader to infer health from an absence.

        MUTANT KILLED: publish a placeholder record when nothing was tried.
        """
        with _StatusDir() as d:
            sched = _FakeScheduler(queued=0, running=4, age=400.0)
            poll = make_admission_wedge_poller(sched)
            self.assertIs(poll(), False)
            record = _record_for(d, _rank_label(sched))
            signal = wedge_status.read_wedge_signal(directory=d)

        self.assertIsNotNone(record)
        self.assertIs(record["wedged"], False)
        self.assertNotIn("recovery", record)
        self.assertIs(signal.verdict, False)
        self.assertIsNone(signal.recovery)

    def test_recovery_status_is_none_until_something_settles(self):
        """``None`` means no measurement, and must not be faked into a value."""
        sched = _FakeScheduler(queued=2, running=0, age=400.0)
        driver = AdmissionWedgeRecovery(sched)
        self.assertIsNone(driver.recovery_status())
        driver.step(alarm=True)  # posts, nothing settled yet
        self.assertIsNone(driver.recovery_status())

    def test_a_publish_failure_cannot_take_the_poll_down(self):
        """Telemetry must never kill the thing it observes."""
        with _StatusDir() as d:
            sched = _FakeScheduler(queued=2, running=0, age=400.0)
            poll = make_admission_wedge_poller(sched)
            with mock.patch.object(
                wedge_status, "publish_verdict", side_effect=RuntimeError("disk gone")
            ):
                self.assertIs(poll(), True)
            self.assertIsNone(_record_for(d, _rank_label(sched)))
            # and the recovery machinery still ran despite the sink failing
            self.assertEqual(get_recovery_channel(sched).requested_seq, 1)


if __name__ == "__main__":
    unittest.main()
