# SPDX-License-Identifier: Apache-2.0
"""#1361 [22-fix5] -- a dying boot wrote its own agony into the ledger.

Boot weg2xsn25 died on 2026-09-13 between 07:01:14 (first rank lost) and
07:04:16 (no live schedulers). At **07:03:59Z**, three minutes into that
cascade, its sampler wrote a run-moment residual of **34.61 GiB** with
``load_class=idle`` and ``pids=[]``.

The load class was TRUE and meaningless: a box with no live ranks has no queue,
so it measures idle. What the sample actually measured is whatever else was on
the host. The ledger took it as this line's run-moment floor and every arm
gained ~28 GiB:

    run origin 6.56 GiB  ->  34.61 GiB
    run_peak   95.18     ->  132.49 / 127.55 / 125.08  vs hard bound 94.43

No arm funded; the next boot could not be priced. Confirmed on three
independent trees, and on ONE tree across the record's own change with no code
difference between the runs: 95.22 GiB at 06:51:24Z, 123.23 GiB after.

A RECORD IS NOT A COMMIT. It is not picked, not gated and not rolled back, and
it prices every boot that follows. That is a class our merge discipline does
not catch structurally -- which is why the guard has to live in the reader.

THE EVIDENCE WAS ALREADY IN EVERY SAMPLE EVER WRITTEN: ``pids``. It had no
consumer. Same shape as #1350b (Sigma H raise refused) and #1350e (mid-flip
samples refused), for the third number.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

#: The two real weg2xsn25 samples, verbatim from
#: /spinning/evidence-665-f1/weg2_measured_record.json.
HEALTHY = {
    "boot_tag": "weg2xsn25", "commit": "b568f9afd5", "at": "2026-09-12T21:50:16Z",
    "group": "D", "run_residual_gib": -1.8551340340905682, "load_class": "loaded",
    "sampled_at_flip_epoch": 0, "interleaved": True,
    "pids": [3547090, 3548889, 3548890, 3548891, 3548892, 3548893,
             3556038, 3556039, 3556040],
    "pids_asked": [3547090, 3548889, 3548890, 3548891, 3548892, 3548893,
                   3556038, 3556039, 3556040],
    "rss_shmem_gib": 48.0, "residual_charges_gib": 12.0, "arm": {"s_gb": 1, "m_mib": 150},
}
DEATH = {
    "boot_tag": "weg2xsn25", "commit": "8f11b2e4c6", "at": "2026-09-13T07:03:59Z",
    "group": "D", "run_residual_gib": 34.607529713279675, "load_class": "idle",
    "sampled_at_flip_epoch": 0, "interleaved": True,
    "pids": [], "pids_asked": [],
    "rss_shmem_gib": 48.0, "residual_charges_gib": 12.0, "arm": {"s_gb": 1, "m_mib": 150},
}


def _origin(record, launch_gib=6.62):
    return hl.run_origin_gib(cg_nonreclaim_gib=launch_gib, record=record)


class TheDeathSampleIsRefusedByName(CustomTestCase):
    def test_the_3461_sample_does_not_become_the_floor(self):
        val, why = _origin({"D": dict(DEATH)})
        self.assertLess(
            val, 10.0,
            f"the 07:03:59 death sample became the origin ({val:.2f} GiB); "
            f"that is the +28 GiB that refused every arm",
        )
        self.assertIn("DEATH-SAMPLE", why)
        self.assertIn("pids=0", why)

    def test_the_refusal_names_the_record_not_just_the_number(self):
        """A rejected number that cannot be traced to its sample is unactionable."""
        _, why = _origin({"D": dict(DEATH)})
        for token in ("weg2xsn25", "2026-09-13T07:03:59Z", "34.61"):
            self.assertIn(token, why, f"the refusal does not name {token}")

    def test_load_class_idle_is_not_evidence_of_health(self):
        """The whole trap in one assertion.

        The sample says `idle`, and it is not lying -- it had 0 queued and 0
        outstanding. It had 0 queued because it had 0 processes. A guard keyed
        on load_class would have passed this sample through.
        """
        self.assertEqual(DEATH["load_class"], "idle")
        self.assertEqual(DEATH["pids"], [])
        _, why = _origin({"D": dict(DEATH)})
        self.assertIn("true and empty", why)

    def test_a_healthy_sample_is_still_read(self):
        """The control: the filter must not simply refuse everything.

        weg2xsn25's OWN earlier sample, 9 live pids, is untouched by this guard
        (it is still rejected as interleaved by #1350e, which is a different
        rule and prints a different reason).
        """
        _, why = _origin({"D": dict(HEALTHY)})
        self.assertNotIn("DEATH-SAMPLE", why)

    def test_a_kill_between_two_samples_of_one_boot_is_visible(self):
        """The second witness, from the counter the writer now stamps.

        A sample taken after this boot's kill counter has risen is refused even
        if some ranks were still alive when it was taken.
        """
        early = dict(HEALTHY, cg_oom_kill=4)
        late = dict(HEALTHY, cg_oom_kill=6, run_residual_gib=30.0,
                    at="2026-09-13T07:03:59Z")
        _, why = _origin({"P": early, "D": late})
        self.assertIn("DEATH-SAMPLE", why)
        self.assertIn("kill counter had risen to 6", why)

    def test_an_unreadable_counter_is_not_a_healthy_one(self):
        """`None` must not be scored as zero.

        A sample that could not read the counter carries no evidence either
        way; it must not be admitted BECAUSE the evidence is missing.
        """
        self.assertIsNone(dict(HEALTHY).get("cg_oom_kill"))
        _, why = _origin({"D": dict(HEALTHY)})
        self.assertNotIn("kill counter", why)

    def test_the_writer_stamps_the_counter_into_new_samples(self):
        """Reachability: a filter reading a field nobody writes is decoration."""
        rec = hl.dormant_image_sample(
            group="P", shmem_before_bytes=0, shmem_after_bytes=0, pids=[],
            weight_tags_gib=1.0, interleaved=False, boot_tag="t", commit="c",
        )
        self.assertIn("cg_oom_kill", rec)


if __name__ == "__main__":
    unittest.main()


class AbsenceOfEvidenceIsNotEvidenceOfDeath(CustomTestCase):
    """#1361 [22-fix5b] -- the regression the train seat's red caught.

    [22-fix5] wrote ``len(e.get("pids") or [])``, which reads an ABSENT field as
    an empty one. Every record written before ``pids`` existed was then refused
    as a death sample: weg2sn6s's origin went from 7.63 GiB to 0.0 and the
    #1325 contract broke in two named tests.

    That is absence of evidence scored as evidence of death -- the same error
    [22-fix] made with an empty model digest and [22-fix2] had to undo, made
    again by the same seat four commits later. A sample that never recorded its
    process set says nothing about it; only a sampler that LOOKED and found
    none is a witness.

    The real weg2xsn25 death sample carries ``pids: []`` AND ``pids_asked: []``
    -- the keys are PRESENT and empty, which is the sampler reporting that it
    looked. That is what makes it distinguishable from a legacy record, and it
    is why this guard can be both strict and safe.
    """

    #: A pre-`pids` record, shaped like the #1325 weg2sn6s fixture: no process
    #: set was ever recorded.
    LEGACY_NO_PIDS = {
        "group": "P", "at": "2026-09-10T15:21:55Z", "boot_tag": "weg2sn6s",
        "commit": "23dd8ab2c1", "rss_shmem_gib": 38.63, "weight_tags_gib": 28.83,
        "arm": {"m_mib": 600, "s_gb": 1, "s_gb_d": 4}, "run_residual_gib": 16.34,
    }

    def test_a_record_without_a_pids_key_is_not_a_death_sample(self):
        self.assertNotIn("pids", self.LEGACY_NO_PIDS)
        _, why = hl.run_origin_gib(8.0, {"P": dict(self.LEGACY_NO_PIDS)})
        self.assertNotIn("DEATH-SAMPLE", why)

    def test_the_1325_origin_contract_survives_the_new_filter(self):
        """The exact two numbers the #1325 tests pin, asserted here too.

        Pinned in THIS file as well as in test_weg2_run_residual_currency_1325
        so the coupling is visible from the guard's own side: whoever changes
        this filter sees what it may not move.
        """
        origin, src = hl.run_origin_gib(8.0, {"P": dict(self.LEGACY_NO_PIDS)})
        self.assertAlmostEqual(origin, 8.0, places=6)
        self.assertIn("AT OR ABOVE", src)

    def test_an_empty_but_present_pids_set_is_still_a_death_sample(self):
        """The complement: the guard must not be softened into uselessness."""
        entry = dict(self.LEGACY_NO_PIDS, pids=[], pids_asked=[],
                     run_residual_gib=34.607529713279675)
        _, why = hl.run_origin_gib(8.0, {"P": entry})
        self.assertIn("DEATH-SAMPLE", why)
