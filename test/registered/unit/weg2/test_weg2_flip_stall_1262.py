# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#1262 tier 3: the dead-man switch is blind to a livelock behind a healthy
front, and the front is the process that can see it.

THE COVERAGE GAP, measured (BOOT_weg2t2a_0908.md, 2026-09-08). The front logged
``WEG2-FLIP begin epoch=0 sleep=D wake=P outstanding=0 queue=1`` at 11:57:07Z
and was still at ``state=flipping epoch=0 flips=0`` seven minutes later,
because all six scheduler ranks were spinning inside an idle-time invariant
check. All three deadmen were ARMED (pgrep-proven) and NONE fired -- correctly:

* tier 1 asks whether a serving process exists. Six did.
* tier 2 probes ``/health_generate``. All three ports answered **200**
  throughout, because the FRONT was never wedged; only the groups behind it.

Neither tier is at fault and no threshold of theirs would have helped. The
question they do not ask is "did a flip that started ever finish", and the
front is the only process that holds both halves of it (``state``, ``epoch``,
and the measured cost of every completed flip).

THE BOUND IS DERIVED, and that is the load-bearing property. A literal "if a
flip takes more than N seconds" would be wrong on both arms of this campaign's
own spread -- weg2rg3 flipped in 3.1-4.7 s, weg2dk5 in 13.4/18.4 s. So:

* after the first measured flip -- ``FLIP_STALL_SLACK`` (4x) times that flip's
  own ``flip_ms``, read from the same ``flip_log`` records
  ``_derived_min_dwell_ms`` already prices a round trip with;
* before it -- ``drain_deadline_s``, the front's OWN published bound on a
  single flip phase. **Epoch 0 is the specimen**, so this branch is the one
  that had to work, and it deliberately reuses an existing declared front
  bound rather than inventing a number.

The line the deadman keys on is anchored ``WEG2-FLIP STALL epoch=<digits>``
(#995: the bare token also appears in prose, in this file and in
boot_deadman.sh's own comments). The deadman's half is pinned by case 7c of
``boot_deadman.sh --selftest``, not here.

Hermetic: the ``Front`` object is built directly, no sockets, no aiohttp, no
GPU; the clock is passed in.
"""

import inspect
import re
import unittest

from sglang.srt.weg2.front import (
    DRAIN_DEADLINE_DEFAULT_S,
    FLIP_STALL_SLACK,
    Front,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _front(**kw):
    return Front(
        "http://127.0.0.1:31000",
        "http://127.0.0.1:31001",
        "D",
        "t1262",
        "",
        0,
        0,
        {},
        45.0,
        **kw,
    )


def _open_flip(front, t0=1000.0, stage="drain", epoch=0):
    front.state = "flipping"
    front._flip_t0 = t0
    front._flip_stage = stage
    front.epoch = epoch
    return front


class TestTheBoundIsDerivedNotALiteral1262(CustomTestCase):
    def test_before_the_first_flip_the_front_uses_its_own_drain_deadline(self):
        f = _front()
        bound, why = f._flip_stall_bound_s()
        self.assertEqual(bound, DRAIN_DEADLINE_DEFAULT_S)
        self.assertIn("no flip measured", why)
        self.assertIn("drain-deadline-s", why)

    def test_after_a_flip_the_bound_carries_that_flips_own_measured_cost(self):
        """#1317h CHANGED THE FORMULA, NOT THE PRINCIPLE.

        #1262's principle -- the bound is DERIVED from this boot's own
        measurements and is never a literal -- survives and is still asserted
        here. What changed is that `FLIP_STALL_SLACK * flip_ms` was not a
        bound a healthy flip could satisfy: `flip_ms` is measured AFTER the
        drain succeeded, so the drain's own cost was never in it, and boot
        weg2sn6g fired twice IN THE DRAIN STAGE on a healthy boot
        (elapsed 17.5 / bound 14.1, and 16.1 / 13.5). The bound is now the
        drain distribution plus the flip -- both halves still this boot's own
        measurements.
        """
        f = _front()
        f.flip_log.append({"sleep": "D", "wake": "P", "flip_ms": 3200})
        bound, why = f._flip_stall_bound_s()
        # the flip half is still that flip's own cost, named in the provenance
        self.assertIn("3200 ms", why)
        self.assertIn("D->P", why)
        # and the drain half is ADDED, not replaced: with a young distribution
        # it is the front's own published deadline.
        self.assertAlmostEqual(bound, f.drain_deadline_s + 3.2, places=3)
        self.assertGreater(bound, FLIP_STALL_SLACK * 3.2)

    def test_the_bound_tracks_a_slow_form_without_a_second_number(self):
        """weg2rg3 (3.1 s) and weg2dk5 (18.4 s) are the same code path.

        #1317h: THE INTENT IS UNTOUCHED -- one code path, no second number,
        and the bound still MOVES with this boot's own flip cost. Only the
        formula changed, from `SLACK x flip_ms` to `drain p99 + flip_ms`,
        because flip_ms is measured after the drain succeeded and so the
        drain's own cost was never in the old bound (the weg2sn6g false
        positive). Asserted as the identity AND as monotonicity, so a future
        formula that stopped tracking the measurement would still fail here.
        """
        bounds = {}
        for ms in (3100, 18400):
            with self.subTest(flip_ms=ms):
                f = _front()
                f.flip_log.append({"sleep": "P", "wake": "D", "flip_ms": ms})
                bound, _ = f._flip_stall_bound_s()
                self.assertAlmostEqual(
                    bound, f.drain_deadline_s + ms / 1000.0, places=3
                )
                bounds[ms] = bound
        self.assertGreater(bounds[18400], bounds[3100],
                           "the bound must still track the measured flip cost")

    def test_a_front_configured_with_a_different_deadline_derives_from_it(self):
        f = _front(drain_deadline_s=45.0)
        bound, _ = f._flip_stall_bound_s()
        self.assertEqual(bound, 45.0, "the bound follows the front's own flag")

    def test_a_zero_flip_ms_record_is_skipped_not_believed(self):
        """A record with no measured cost cannot set a bound of 0 s."""
        f = _front()
        f.flip_log.append({"sleep": "D", "wake": "P", "flip_ms": 0})
        bound, why = f._flip_stall_bound_s()
        self.assertEqual(bound, DRAIN_DEADLINE_DEFAULT_S)
        self.assertIn("no flip measured", why)


class TestItFiresAtTheBoundAndNotBefore1262(CustomTestCase):
    def test_nothing_fires_before_the_bound(self):
        f = _open_flip(_front())
        for elapsed in (0.0, 1.0, DRAIN_DEADLINE_DEFAULT_S - 0.01):
            with self.subTest(elapsed=elapsed):
                self.assertIsNone(f.flip_stall_check(now=1000.0 + elapsed))
        self.assertEqual(f.counters["flip_stall"], 0)

    def test_it_fires_at_the_bound(self):
        f = _open_flip(_front())
        line = f.flip_stall_check(now=1000.0 + DRAIN_DEADLINE_DEFAULT_S)
        self.assertIsNotNone(line, "the detector must fire AT its own bound")
        self.assertEqual(f.counters["flip_stall"], 1)

    def test_the_line_carries_epoch_elapsed_stage_and_provenance(self):
        f = _open_flip(_front(), stage="gathered-legs", epoch=3)
        line = f.flip_stall_check(now=1000.0 + 418.0)
        self.assertIsNotNone(line)
        self.assertIn("WEG2-FLIP STALL epoch=3", line)
        self.assertIn("elapsed=418.0 s", line)
        self.assertIn("bound=120.0 s", line)
        # #1264 fix 2b (1): this assertion used to read `stage=gathered-legs`,
        # and that spelling was the defect, not a detail of it. `_flip_stage`
        # names the last stage ENTERED, and on boot weg2t2b the line reported
        # `stage=sleep-kv` about a stage that had COMPLETED 123 s earlier under
        # a dead controller -- which is what sent the triage to the HiCache
        # drain instead of to the front. The field now says what it knows and
        # carries the age that makes it checkable.
        self.assertIn("stage_last_known=gathered-legs", line)
        self.assertRegex(line, r"age_s=(\d+\.\d|unknown)")
        self.assertNotRegex(line, r"(^|\s)stage=")
        self.assertIn("bound provenance:", line)

    def test_the_deadmans_anchor_matches_the_line_the_front_writes(self):
        """#995: the tier-3 grep is 'WEG2-FLIP STALL epoch=<digit>'.

        The bare token appears in prose (boot_deadman.sh's own comments, this
        module's docstring); only the genuine emitter follows it with epoch=
        and a digit. If the front's format ever loses that, the deadman goes
        silent with no error anywhere -- so the anchor is asserted against the
        real line here, on the producing side.
        """
        f = _open_flip(_front())
        line = f.flip_stall_check(now=1000.0 + 200.0)
        self.assertRegex(line, r"WEG2-FLIP STALL epoch=[0-9]")
        self.assertIsNotNone(
            re.search(r"WEG2-FLIP STALL epoch=[0-9]", line),
            "the deadman's tier-3 pattern must match the front's own line",
        )

    def test_it_fires_once_per_flip_not_once_per_poll(self):
        """A repeating alarm is a persistent monitor, which this rig forbids."""
        f = _open_flip(_front())
        self.assertIsNotNone(f.flip_stall_check(now=1000.0 + 200.0))
        for extra in (210.0, 400.0, 4000.0):
            self.assertIsNone(f.flip_stall_check(now=1000.0 + extra))
        self.assertEqual(f.counters["flip_stall"], 1)

    def test_a_later_flip_gets_its_own_line(self):
        f = _open_flip(_front(), epoch=0)
        self.assertIsNotNone(f.flip_stall_check(now=1000.0 + 200.0))
        # flip 0 completed; flip 1 opens and also stalls
        f.epoch = 1
        f._flip_t0 = 2000.0
        self.assertIsNotNone(f.flip_stall_check(now=2000.0 + 200.0))
        self.assertEqual(f.counters["flip_stall"], 2)

    def test_a_serving_front_never_fires(self):
        f = _front()
        f.state = "serving"
        f._flip_t0 = 1000.0
        self.assertIsNone(f.flip_stall_check(now=1000.0 + 99999.0))

    def test_a_stopped_front_never_fires(self):
        f = _open_flip(_front())
        f.state = "STOP"
        self.assertIsNone(f.flip_stall_check(now=1000.0 + 99999.0))

    def test_no_open_flip_never_fires(self):
        f = _front()
        f.state = "flipping"
        f._flip_t0 = None
        self.assertIsNone(f.flip_stall_check(now=1e12))


class TestTheWeg2t2aSpecimen1262(CustomTestCase):
    """The measured case, replayed on its own numbers."""

    def test_the_specimen_would_have_been_named_at_120_s_not_at_489_s(self):
        """weg2t2a: WEG2-FLIP begin 11:57:07Z, still flipping at 12:04:05Z.

        489 s of uptime, epoch 0, flips=0, three ports answering 200. With no
        completed flip on that boot the bound is the front's own 120 s drain
        deadline, so the line lands ~6 min before the operator's py-spy dumps
        did -- and unlike the dumps it lands without anyone watching.
        """
        f = _open_flip(_front(), t0=0.0, stage="drain", epoch=0)
        f.queue.append(object())  # queue=1, as the begin line recorded
        self.assertIsNone(f.flip_stall_check(now=119.0))
        line = f.flip_stall_check(now=418.0)  # 11:57:07 -> 12:04:05
        self.assertIsNotNone(line)
        self.assertIn("epoch=0", line)
        self.assertIn("queue=1", line)
        self.assertIn("flips=0", line)

    def test_the_bound_stays_derived_and_firing_early_is_no_longer_a_virtue(self):
        """THIS TEST'S DIRECTION WAS REFUTED ON METAL, and the correction is
        the point of #1317h.

        It used to argue that calling a stall at 12.8 s was BETTER than a
        fixed 120 s, because "a boot whose flips measure 3.2 s would let a
        flip run 37x its own cost before saying anything". Boot weg2sn6g
        showed the cost of that argument: the detector fired TWICE on a
        healthy serving boot (epoch 4, elapsed 17.5 s against bound 14.1 s;
        epoch 6, 16.1 against 13.5), BOTH in the drain stage, because a drain
        may legitimately wait up to `drain_deadline_s` for a decode that
        #1011 forbids cutting. Firing early is not sensitivity here -- it is a
        false wedge verdict on work in progress.

        WHAT SURVIVES is #1262's actual invariant: the bound is DERIVED from
        this boot's own measurements and is never a literal. That is asserted
        below. What replaces "fire sooner" is the PROGRESS GATE (#1317h): at
        the bound, a group that is still emitting is WAITING, not stalled --
        so sensitivity now comes from asking whether work is happening, not
        from a shorter deadline.
        """
        fast = _front()
        fast.flip_log.append({"sleep": "D", "wake": "P", "flip_ms": 3200})
        bound, why = fast._flip_stall_bound_s()
        # DERIVED, not a literal: it moves with this boot's own flip_ms.
        slow = _front()
        slow.flip_log.append({"sleep": "D", "wake": "P", "flip_ms": 18400})
        assert slow._flip_stall_bound_s()[0] > bound
        self.assertIn("3200 ms", why)
        # and the two recorded sn6g fires no longer convict a healthy flip
        for elapsed in (17.5, 16.1):
            _open_flip(fast, t0=0.0, epoch=1)
            fast._flip_stall_reported_epoch = None
            self.assertIsNone(
                fast.flip_stall_check(now=elapsed),
                f"elapsed={elapsed} s fired on a healthy boot -- the weg2sn6g "
                f"false positive is back",
            )


class TestTheDetectorIsWired1262(CustomTestCase):
    """Delivery, not presence (#859): the sampler must be started."""

    def test_startup_creates_the_sampler_and_cleanup_cancels_it(self):
        startup = inspect.getsource(Front.startup)
        cleanup = inspect.getsource(Front.cleanup)
        self.assertIn("flip_stall_sampler()", startup)
        self.assertIn('app["flip_stall"]', startup)
        self.assertIn('"flip_stall"', cleanup)

    def test_flip_stamps_its_t0_and_its_stages(self):
        src = inspect.getsource(Front.flip)
        self.assertIn("self._flip_t0 = t_flip0", src)
        for stage in ("drain", "quiesce", "sleep-kv", "gathered-legs", "wake-kv"):
            self.assertIn(f'self._flip_stage = "{stage}"', src)
        self.assertIn(
            "self._flip_t0 = None",
            src,
            "a completed flip must clear the open-flip stamp",
        )


if __name__ == "__main__":
    unittest.main()
