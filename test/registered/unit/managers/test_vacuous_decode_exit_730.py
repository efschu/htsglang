# Copyright 2026 SGLang Team
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
"""#730: zero work must not read as all work.

THE SPECIMEN, 2026-08-17 14:48:22, from a boot whose warmup generation never
reached a first token:

    arming tp_to_pp: decode bundle complete: 0 reqs decoded in 183.6 s
    (13777 tok prefill waiting) -- exit condition: decode drained

The tp phase was entered with NOTHING decoding, sat 183.6 s, and left claiming
completion. Six such cycles in one boot. ``bundle_at_phase_entry`` is
``running_bs`` AT ENTRY (phase_policy.py:2093), so 0 means the bundle was never
resident -- there was no work to drain.

The sentence is the defect. A drained bundle and an empty visit produced the
SAME words, differing only by a number that reads as a quantity rather than as
a verdict, so an operator scanning the log saw "complete" either way.

LEAVING IS STILL CORRECT -- a decode layout with nothing to decode should yield
to prefill -- so these tests pin the DIAGNOSIS, not a control-flow change. The
direction stays TP_TO_PP in both cases; only the reason differs.

WHY IT MATTERS BEYOND WORDING: the #699 admission-wedge detector fired 17 times
on this same boot and recorded "no phase-policy corroboration seen". An empty
decode phase standing beside a non-empty backlog IS that corroboration -- it is
the layout's own evidence that work is queued which nothing is making runnable
(#731). The vacuous message was withholding it.
"""

import unittest

from sglang.srt.managers.phase_policy import (
    PHASE_TP,
    TP_TO_PP,
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    decide,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2)

#: The specimen's numbers.
PENDING = 13777
ELAPSED_S = 183.6


def _cfg(**kw):
    # #815: `idle_locked_settle_s=0.0` USED TO BE HERE and is not an omission.
    # It was #713's post-cutover settle, and this file set it to 0.0 to hold the
    # knob OUT of the way while the reason strings below are read. That field no
    # longer exists: 4a16043d1a reverted 97cb40bba4 whole, with no successor
    # knob (`grep settle python/sglang/srt/managers/phase_policy.py` is empty).
    #
    # THE HISTORY IS A MERGE CONFLUENCE, NOT A MISSED UPDATE, and the difference
    # matters to whoever merges next. This file was green when it was written:
    # at its own commit 7592f3243e the field was still present in phase_policy
    # (three occurrences), and the revert is NOT an ancestor of that commit. The
    # two lines met later, and the meeting left a kwarg with nothing behind it.
    # So the state this line modelled -- "the settle is not interfering" -- is
    # now the only state there is, and expressing it is exactly the empty set.
    base = dict(
        enabled=True,
        flip_tokens=7004,
        min_dwell_s=3.0,
        idle_dwell_s=3.0,
        rest_state="decode",
        drain_mode=True,
        pp_exit_tokens=0,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


def _decide(bundle_at_entry: int, running_bs: int = 0):
    state = PhasePolicyState(phase_since=100.0 - ELAPSED_S, last_flip_at=1.0)
    state.bundle_at_phase_entry = bundle_at_entry
    inp = PhasePolicyInputs(
        phase=PHASE_TP,
        pending_prefill_tokens=PENDING,
        running_bs=running_bs,
        now=100.0,
    )
    return decide(_cfg(), state, inp)


class AnEmptyDecodePhaseIsNamedAsSuch(unittest.TestCase):
    def test_the_specimen_no_longer_claims_completion(self):
        """RED before the fix: this said 'decode bundle complete'."""
        d = _decide(bundle_at_entry=0)
        self.assertEqual(TP_TO_PP, d.direction, "leaving is still correct")
        self.assertNotIn("bundle complete", d.reason)
        self.assertNotIn("drained", d.reason.split("NOTHING WAS DRAINED")[0])

    def test_it_says_plainly_that_nothing_ran(self):
        d = _decide(bundle_at_entry=0)
        self.assertIn("ran EMPTY", d.reason)
        self.assertIn("NOTHING WAS DRAINED", d.reason)
        self.assertIn("no bundle was ever resident", d.reason)

    def test_it_points_at_admission_and_carries_the_backlog(self):
        """The corroboration #699 said was missing."""
        d = _decide(bundle_at_entry=0)
        self.assertIn(str(PENDING), d.reason)
        self.assertIn("ADMISSION", d.reason)
        self.assertIn("#731", d.reason)
        self.assertIn("#699", d.reason)


class ARealDrainStillReadsAsCompletion(unittest.TestCase):
    """CAN-FAIL: the fix must not rename the healthy path.

    If an actually-drained bundle stopped saying 'complete', every green window
    would look like a defect and the new line would be noise instead of signal.
    """

    def test_a_drained_bundle_says_complete(self):
        d = _decide(bundle_at_entry=3)
        self.assertEqual(TP_TO_PP, d.direction)
        self.assertIn("decode bundle complete", d.reason)
        self.assertIn("3 req bundle drained", d.reason)
        self.assertNotIn("ran EMPTY", d.reason)

    def test_the_two_paths_do_not_share_a_verdict_word(self):
        """The whole defect was one sentence serving both states."""
        empty = _decide(bundle_at_entry=0).reason
        drained = _decide(bundle_at_entry=3).reason
        self.assertNotEqual(empty, drained)
        self.assertTrue(
            ("complete" in drained) and ("complete" not in empty.split("--")[0]),
            "an empty visit must not read as completion",
        )

    def test_a_running_bundle_is_untouched(self):
        """running_bs > 0 still refuses to leave mid-bundle."""
        d = _decide(bundle_at_entry=3, running_bs=2)
        self.assertIsNone(d.direction)
        self.assertIn("decode bundle running", d.reason)


if __name__ == "__main__":
    unittest.main()
