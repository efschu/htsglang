"""#605: the ledger's dominant post was a silent zero, and the fix was on disk.

WHAT IT COST, TWICE IN ONE NIGHT. The mem_ledger dumped on the running boot
reported `model weights (shards) = 0 MiB` on all three cards while those cards
held 10194 / 16196 / 10832 MiB of them, marked three posts UNBOUNDED, and said
`fits=False` everywhere. In that state it could arbitrate neither the #683
attribution ("who took the card") nor the #678 sizing question, and both were
answered by hand instead.

WHY THE ZERO. Not an unwired term and not a fingerprint that never matches:
`reconcile.completeness_failures` already names it -- the shipped config pins
`--rank-gpu-memory-mib`, which is the PIN PATH, and the pin path skips the
planner that computes the shard vector. The term is built, formatted and dumped
from an all-zero vector, indistinguishable in the JSON from a model that needs
no weights.

THE NUMBER WAS ALREADY BEING RECORDED. The flight recorder marks every boot at
`pre_weight_load` and `weights_loaded`; the reserved-bytes delta between them IS
the shard footprint, measured rather than modeled, per card, on every boot. So
this is not a new measurement path -- it is reading the instrument that was
already running.

WHERE THE DETECTION LIVES, AND WHY NOT IN THE ENGINE. `completeness_failures`
already calls a zero here "the PIN PATH signature" and `require_complete` raises
on it, so this slice substitutes the measurement and does NOT add a second
refusal inside the engine. A refusal there would change the verdict of every
card whose shard vector is legitimately absent at that point in the boot --
including the pre-planner ledger whose residual is the RANK BUDGET rather than
the KV pool. The detector is therefore the acceptance, pinned in both
directions: it must stop firing once the post is measured and keep firing on a
fingerprint with no marks to stand in.

THE TRANSIENT SPLIT. `fits` was `not unbounded and committed <= total`, so ONE
inherently-unbounded transient made every card unfittable forever -- the load
transient refuses on EVIDENCE (563 boots, 0-18486 MiB spread, above the 50%
refusal rule), and that refusal is correct and must stay. Resident posts that
cannot be priced must still block the verdict; a transient that cannot be
summarised is a RISK BAND beside it, with its evidence attached. Two claims,
two outputs, no silent estimate either way.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.mem_ledger import flight_recorder as fr
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

MIB = 1 << 20
CARD = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"


def _mark(phase, resv_mib, *, card=CARD, alloc_mib=None, peak_mib=None, wall=0.0):
    return {
        "phase": phase,
        "boot_id": "boot-1",
        "pid": 4242,
        "wall": wall,
        "card_uuid": card,
        "reserved_bytes": resv_mib * MIB,
        "allocated_bytes": (alloc_mib if alloc_mib is not None else resv_mib) * MIB,
        "reserved_peak_bytes": (peak_mib if peak_mib is not None else resv_mib) * MIB,
        "allocated_peak_bytes": (peak_mib if peak_mib is not None else resv_mib) * MIB,
    }


class _OnDisk(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = self._dir.name

    def tearDown(self):
        self._dir.cleanup()

    def _write(self, marks, stem=fr.MARKS_FILE_STEM, rank=0):
        path = os.path.join(self.dir, f"{stem}{rank}.jsonl")
        with open(path, "a") as fh:
            for m in marks:
                fh.write(json.dumps(m) + "\n")


class TheWeightsPostIsReadFromTheMarks(_OnDisk):
    def test_the_reserved_delta_across_the_load_is_the_shard_footprint(self):
        """The running boot's own numbers for the 5090."""
        self._write(
            [
                _mark("process_start", 0, wall=1.0),
                _mark("pre_weight_load", 2, wall=2.0),
                _mark("weights_loaded", 16198, wall=3.0),
                _mark("kv_pool_sized", 23070, wall=4.0),
            ]
        )
        posts = fr.measured_card_posts(self.dir)
        self.assertIn(CARD, posts)
        self.assertEqual(16196, posts[CARD]["weights_mib"])

    def test_the_FIRST_load_is_the_shard_post_not_a_later_layout_or_drafter(self):
        """A phase-flip boot loads three times: two layouts and the drafter.

        The term names the model's shards, so it is the first pair. Taking a
        later one would price the drafter as the model.
        """
        self._write(
            [
                _mark("pre_weight_load", 2, wall=2.0),
                _mark("weights_loaded", 16198, wall=3.0),
                _mark("pre_weight_load", 7792, wall=5.0),
                _mark("weights_loaded", 24182, wall=6.0),
                _mark("pre_weight_load", 24130, wall=7.0),
                _mark("weights_loaded", 25966, wall=8.0),
            ]
        )
        self.assertEqual(16196, fr.measured_card_posts(self.dir)[CARD]["weights_mib"])

    def test_cards_are_kept_apart_by_uuid(self):
        other = "GPU-5c648f96-be1d-42"
        self._write(
            [
                _mark("pre_weight_load", 2, wall=2.0),
                _mark("weights_loaded", 16198, wall=3.0),
                _mark("pre_weight_load", 2, card=other, wall=2.0),
                _mark("weights_loaded", 10196, card=other, wall=3.0),
            ]
        )
        posts = fr.measured_card_posts(self.dir)
        self.assertEqual(16196, posts[CARD]["weights_mib"])
        self.assertEqual(10194, posts[other]["weights_mib"])

    def test_an_unpaired_load_yields_nothing_rather_than_a_guess(self):
        """A boot that died between the two marks must not price the post."""
        self._write([_mark("pre_weight_load", 2, wall=2.0)])
        posts = fr.measured_card_posts(self.dir)
        self.assertNotIn("weights_mib", posts.get(CARD, {}))

    def test_no_marks_at_all_is_an_empty_answer_not_a_zero(self):
        self.assertEqual({}, fr.measured_card_posts(self.dir))

    def test_a_missing_directory_is_survivable(self):
        self.assertEqual({}, fr.measured_card_posts(os.path.join(self.dir, "nope")))


class TheCaptureAndActivationPosts(_OnDisk):
    def test_the_capture_delta_is_reported_for_the_GATED_investigation(self):
        """Recorded, but #605(c) keeps it OUT of the calibration store until
        the 182-vs-'3.3-3.8x low' discrepancy is explained."""
        self._write(
            [
                _mark("capture_begin", 32006, wall=1.0),
                _mark("capture_end", 32190, wall=2.0),
            ]
        )
        self.assertEqual(184, fr.measured_card_posts(self.dir)[CARD]["capture_mib"])

    def test_the_activation_envelope_needs_serving_marks(self):
        """Boot marks stop at first_forward, so they cannot bound activation
        under load. Absent serving samples the post has no envelope at all."""
        self._write(
            [
                _mark("boot_complete", 32548, alloc_mib=32000, wall=1.0),
                _mark("first_forward", 32598, alloc_mib=32100, wall=2.0),
            ]
        )
        self.assertNotIn(
            "activation_upper_mib", fr.measured_card_posts(self.dir).get(CARD, {})
        )

    def test_a_serving_sample_supplies_an_UPPER_BOUND_only(self):
        """The marks never reset the peak counters, so what they carry is a
        monotone envelope since process start -- an upper bound, never a
        per-phase point estimate. Safe in the OOM direction, and it must be
        labelled as such rather than published as a measurement."""
        self._write(
            [_mark("boot_complete", 32548, alloc_mib=32000, peak_mib=32000, wall=1.0)]
        )
        self._write(
            [
                _mark(
                    fr.SERVING_PHASE, 32600, alloc_mib=32050, peak_mib=32900, wall=9.0
                )
            ],
            stem=fr.SERVING_FILE_STEM,
        )
        posts = fr.measured_card_posts(self.dir)
        self.assertEqual(900, posts[CARD]["activation_upper_mib"])


if __name__ == "__main__":
    unittest.main()


class TheCompletenessDetectorIsTheAcceptance(_OnDisk):
    """#605(a): `completeness_failures` is the pin, in BOTH directions.

    It already names the defect -- "the PIN PATH signature" -- so the engine
    does not duplicate the detection; what changed is that a boot whose marks
    can supply the post no longer reaches it.
    """

    def _ledger_payload(self, weights_mib):
        return {
            "cards": [
                {
                    "gpu_id": 0,
                    "card": "GPU 0 (NVIDIA GeForce RTX 5090, NVML total 32607 MiB)",
                    "terms": [{"name": "model weights (shards)", "mib": weights_mib}],
                    "unbounded": [],
                }
            ]
        }

    def test_it_keeps_firing_on_an_uncalibrated_fingerprint(self):
        """No marks to stand in, so the term is still a priced zero and the
        detector must say so LOUDLY rather than the ledger guessing."""
        from sglang.srt.mem_ledger import reconcile

        failures = reconcile.completeness_failures(self._ledger_payload(0))
        self.assertEqual(1, len(failures))
        self.assertIn("PIN PATH", failures[0])
        self.assertIn("model weights (shards)", failures[0])

    def test_it_stops_firing_once_the_post_is_measured(self):
        """The measured substitution is what clears it -- 16196 MiB is this
        rig's own pre_weight_load -> weights_loaded delta on the 5090."""
        from sglang.srt.mem_ledger import reconcile

        self.assertEqual(
            [], reconcile.completeness_failures(self._ledger_payload(16196))
        )

    def test_a_refusal_naming_the_term_is_also_accepted(self):
        """A term that says out loud it could not be priced is the honest
        outcome; only a silent zero is a failure. Unchanged by this slice, and
        asserted so the split cannot quietly remove it."""
        from sglang.srt.mem_ledger import reconcile

        payload = self._ledger_payload(0)
        payload["cards"][0]["unbounded"] = [
            "model weights (shards) on GPU 0: refused, no marks"
        ]
        self.assertEqual([], reconcile.completeness_failures(payload))


class TheMeasuredPostsMatchThisRig(unittest.TestCase):
    """A guard on the reader's arithmetic, using the numbers the running boot
    actually wrote. If the mark schema or the phase names move, this fails
    with the rig's own figures rather than an abstract shape."""

    def test_the_reader_reproduces_the_running_boots_weights(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"{fr.MARKS_FILE_STEM}0.jsonl")
            with open(path, "w") as fh:
                for m in (
                    _mark("pre_weight_load", 2, wall=1.0),
                    _mark("weights_loaded", 16198, wall=2.0),
                    _mark("boot_complete", 32548, alloc_mib=32000, wall=3.0),
                ):
                    fh.write(json.dumps(m) + "\n")
            posts = fr.measured_card_posts(d)
        self.assertEqual(16196, posts[CARD]["weights_mib"])
