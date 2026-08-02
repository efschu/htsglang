"""Hardware fingerprint, profile matching, and the store (#407 / directive #434).

What is pinned here:

1. **A fingerprint is pure.** Every input comes from the ``LocalFacts`` handed
   in. If any of these functions ever reads ``/proc/cpuinfo``, the hostname or
   the kernel version, the synthetic-foreign-rig tests in
   ``test_tier_generality.py`` stop meaning anything, because the local
   machine would be mixed into every key they compute.

2. **A profile speaks only about hardware it matches.** ``EXACT`` licenses
   everything, ``MODEL`` licenses card templates and no host / filesystem /
   remote tier, ``NONE`` licenses nothing. The middle rung is the one worth a
   test: two machines with identical cards routinely have different RAM,
   different disks and a different wire.

3. **An unverifiable profile is refused, not trusted.** A document with no
   ``hardware`` block matches nothing, whatever it claims about itself.

4. **Keys are derived, never read.** The block states cards and models; the
   code hashes them. A hand-edited profile therefore cannot carry a stale
   digest that makes it silently unmatchable forever.

    python -m pytest test/registered/unit/memtier/test_tier_fingerprint.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.memtier.fingerprint import (
    FINGERPRINT_VERSION,
    MatchScope,
    card_signature,
    fingerprint_from_facts,
    hardware_block,
    hardware_key_for,
    licensed_document,
    match_profile,
    model_key_for,
    model_signature,
)
from sglang.srt.memtier.profile import (
    CardFact,
    FilesystemFact,
    LocalFacts,
    ProfileError,
)
from sglang.srt.memtier.profile_store import (
    PROFILE_PATH_ENV,
    save_profile,
    select_profile,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1024**3


def cards(*specs):
    return tuple(
        CardFact(uuid=uuid, model=model, total_bytes=int(gib * GIB), bdf=bdf)
        for uuid, model, gib, bdf in specs
    )


#: A synthetic rig that exists nowhere. Two models, deliberately not the ones
#: the bundled profile describes.
RIG_A = LocalFacts(
    cards=cards(
        (
            "GPU-aaaaaaaa-0000-0000-0000-000000000001",
            "NVIDIA A100-SXM4-40GB",
            40,
            "0000:07:00.0",
        ),
        (
            "GPU-aaaaaaaa-0000-0000-0000-000000000002",
            "NVIDIA A100-SXM4-40GB",
            40,
            "0000:0b:00.0",
        ),
    ),
    host_total_bytes=256 * GIB,
    host_available_bytes=200 * GIB,
    filesystems=(
        FilesystemFact(
            mount="/scratch", total_bytes=8 * 10**12, available_bytes=7 * 10**12
        ),
    ),
)

#: The same two card MODELS, different serial numbers, different host.
RIG_A_TWIN = LocalFacts(
    cards=cards(
        (
            "GPU-bbbbbbbb-0000-0000-0000-000000000001",
            "NVIDIA A100-SXM4-40GB",
            40,
            "0000:07:00.0",
        ),
        (
            "GPU-bbbbbbbb-0000-0000-0000-000000000002",
            "NVIDIA A100-SXM4-40GB",
            40,
            "0000:0b:00.0",
        ),
    ),
    host_total_bytes=64 * GIB,
    host_available_bytes=40 * GIB,
    filesystems=(),
)


def document_for(facts, **extra):
    """A minimal, well-formed profile document keyed to ``facts``."""
    fingerprint = fingerprint_from_facts(facts)
    doc = {
        "schema_version": 1,
        "profile_id": extra.pop("profile_id", "synthetic"),
        "host": extra.pop("host", "synthetic-host"),
        "hardware": hardware_block(fingerprint),
        "device_models": {},
        "tiers": [],
    }
    doc.update(extra)
    return doc


class TestFingerprintPurity(unittest.TestCase):
    def test_same_facts_same_keys(self):
        a = fingerprint_from_facts(RIG_A)
        b = fingerprint_from_facts(RIG_A)
        self.assertEqual(a.hardware_key, b.hardware_key)
        self.assertEqual(a.model_key, b.model_key)
        self.assertTrue(a.hardware_key.startswith("hw-"))
        self.assertTrue(a.model_key.startswith("model-"))

    def test_card_order_does_not_change_the_key(self):
        shuffled = LocalFacts(
            cards=tuple(reversed(RIG_A.cards)),
            host_total_bytes=RIG_A.host_total_bytes,
            host_available_bytes=RIG_A.host_available_bytes,
            filesystems=RIG_A.filesystems,
        )
        self.assertEqual(
            fingerprint_from_facts(RIG_A).hardware_key,
            fingerprint_from_facts(shuffled).hardware_key,
        )

    def test_different_serials_share_a_model_key_and_not_a_hardware_key(self):
        a, twin = fingerprint_from_facts(RIG_A), fingerprint_from_facts(RIG_A_TWIN)
        self.assertEqual(a.model_key, twin.model_key)
        self.assertNotEqual(a.hardware_key, twin.hardware_key)

    def test_host_ram_and_mounts_are_not_in_the_exact_key(self):
        """Adding a DIMM does not make a machine a different machine."""
        upgraded = LocalFacts(
            cards=RIG_A.cards,
            host_total_bytes=RIG_A.host_total_bytes * 2,
            host_available_bytes=RIG_A.host_total_bytes,
            filesystems=(),
        )
        self.assertEqual(
            fingerprint_from_facts(RIG_A).hardware_key,
            fingerprint_from_facts(upgraded).hardware_key,
        )
        # ... and the can-fail half: swapping a CARD does.
        swapped = LocalFacts(
            cards=cards(
                (
                    "GPU-aaaaaaaa-0000-0000-0000-000000000001",
                    "NVIDIA A100-SXM4-40GB",
                    40,
                    "",
                ),
                (
                    "GPU-cccccccc-0000-0000-0000-000000000009",
                    "NVIDIA A100-SXM4-40GB",
                    40,
                    "",
                ),
            ),
            host_total_bytes=RIG_A.host_total_bytes,
        )
        self.assertNotEqual(
            fingerprint_from_facts(RIG_A).hardware_key,
            fingerprint_from_facts(swapped).hardware_key,
        )

    def test_cardless_machine_has_no_keys(self):
        empty = fingerprint_from_facts(LocalFacts(host_total_bytes=8 * GIB))
        self.assertEqual(empty.hardware_key, "")
        self.assertEqual(empty.model_key, "")
        self.assertFalse(empty.has_cards)

    def test_vram_is_rounded_into_the_model_signature(self):
        """A few MiB of ECC difference must not split one model in two."""
        a = model_signature(cards(("GPU-1", "Card X", 40, "")))
        b = model_signature(
            (
                CardFact(
                    uuid="GPU-2", model="Card X", total_bytes=40 * GIB - 300 * 1024**2
                ),
            )
        )
        self.assertEqual(a, b)

    def test_vendor_prefix_is_folded(self):
        self.assertEqual(
            model_signature(cards(("GPU-1", "NVIDIA GeForce RTX 4090", 24, ""))),
            ("1x RTX 4090:24GiB",),
        )


class TestProfileMatching(unittest.TestCase):
    def test_exact_match_licenses_everything(self):
        doc = document_for(RIG_A)
        match = match_profile(doc, fingerprint_from_facts(RIG_A))
        self.assertIs(match.scope, MatchScope.EXACT)
        self.assertTrue(match.licenses_tiers)
        self.assertTrue(match.licenses_device_models)

    def test_model_match_licenses_templates_only(self):
        doc = document_for(RIG_A)
        match = match_profile(doc, fingerprint_from_facts(RIG_A_TWIN))
        self.assertIs(match.scope, MatchScope.MODEL)
        self.assertFalse(match.licenses_tiers)
        self.assertTrue(match.licenses_device_models)
        self.assertIn("different RAM, disks and wire", match.reason)

    def test_unrelated_hardware_matches_nothing(self):
        doc = document_for(RIG_A)
        other = LocalFacts(cards=cards(("GPU-zzzz", "Some Other Card", 16, "")))
        match = match_profile(doc, fingerprint_from_facts(other))
        self.assertIs(match.scope, MatchScope.NONE)

    def test_profile_without_a_hardware_block_is_refused(self):
        doc = document_for(RIG_A)
        doc.pop("hardware")
        match = match_profile(doc, fingerprint_from_facts(RIG_A))
        self.assertIs(match.scope, MatchScope.NONE)
        self.assertIn("no 'hardware' block", match.reason)

    def test_empty_hardware_block_is_refused(self):
        doc = document_for(RIG_A)
        doc["hardware"] = {"version": FINGERPRINT_VERSION}
        match = match_profile(doc, fingerprint_from_facts(RIG_A))
        self.assertIs(match.scope, MatchScope.NONE)
        self.assertIn("neither cards nor models", match.reason)

    def test_partial_card_row_is_refused_rather_than_half_matched(self):
        doc = document_for(RIG_A)
        doc["hardware"]["cards"][0].pop("total_bytes")
        match = match_profile(doc, fingerprint_from_facts(RIG_A))
        self.assertIs(match.scope, MatchScope.NONE)
        self.assertIn("uuid, model and total_bytes", match.reason)

    def test_fingerprint_version_mismatch_is_refused(self):
        doc = document_for(RIG_A)
        doc["hardware"]["version"] = FINGERPRINT_VERSION + 1
        match = match_profile(doc, fingerprint_from_facts(RIG_A))
        self.assertIs(match.scope, MatchScope.NONE)
        self.assertIn("not comparable", match.reason)

    def test_keys_are_derived_from_the_inputs(self):
        """A block stating cards yields the same key the live side computes."""
        doc = document_for(RIG_A)
        self.assertEqual(
            hardware_key_for(card_signature(RIG_A.cards)),
            fingerprint_from_facts(RIG_A).hardware_key,
        )
        self.assertEqual(
            model_key_for(doc["hardware"]["models"]),
            fingerprint_from_facts(RIG_A).model_key,
        )


class TestLicensing(unittest.TestCase):
    def setUp(self):
        self.doc = document_for(RIG_A)
        self.doc["tiers"] = [{"id": "host:synthetic-host"}]
        self.doc["device_models"] = {"NVIDIA A100-SXM4-40GB": {}}

    def test_exact_keeps_tiers(self):
        match = match_profile(self.doc, fingerprint_from_facts(RIG_A))
        reduced = licensed_document(self.doc, match)
        self.assertEqual(len(reduced["tiers"]), 1)

    def test_model_scope_removes_tiers_and_keeps_templates(self):
        match = match_profile(self.doc, fingerprint_from_facts(RIG_A_TWIN))
        reduced = licensed_document(self.doc, match)
        self.assertEqual(reduced["tiers"], [])
        self.assertIn("NVIDIA A100-SXM4-40GB", reduced["device_models"])

    def test_none_scope_removes_both(self):
        other = fingerprint_from_facts(
            LocalFacts(cards=cards(("GPU-zzzz", "Other", 16, "")))
        )
        match = match_profile(self.doc, other)
        reduced = licensed_document(self.doc, match)
        self.assertEqual(reduced["tiers"], [])
        self.assertNotIn("device_models", reduced)


class TestProfileStore(unittest.TestCase):
    def test_save_then_select_round_trips_at_exact_scope(self):
        fingerprint = fingerprint_from_facts(RIG_A)
        doc = document_for(RIG_A, profile_id="stored")
        with tempfile.TemporaryDirectory() as tmp:
            path = save_profile(doc, fingerprint, directory=Path(tmp))
            self.assertEqual(path.name, f"{fingerprint.hardware_key}.json")
            selection = select_profile(fingerprint, paths=[path])
        self.assertIsNotNone(selection.profile)
        self.assertIs(selection.scope, MatchScope.EXACT)
        self.assertEqual(selection.profile.profile_id, "stored")

    def test_the_store_keys_from_the_fingerprint_not_the_document(self):
        """A document claiming someone else's hardware is re-keyed on save."""
        fingerprint = fingerprint_from_facts(RIG_A)
        doc = document_for(RIG_A_TWIN, profile_id="mislabelled")
        with tempfile.TemporaryDirectory() as tmp:
            path = save_profile(doc, fingerprint, directory=Path(tmp))
            written = json.loads(path.read_text())
            self.assertEqual(
                written["hardware"]["cards"],
                hardware_block(fingerprint)["cards"],
            )
            self.assertIs(
                select_profile(fingerprint, paths=[path]).scope, MatchScope.EXACT
            )

    def test_a_cardless_machine_cannot_store_a_profile(self):
        empty = fingerprint_from_facts(LocalFacts(host_total_bytes=8 * GIB))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ProfileError) as ctx:
                save_profile({"schema_version": 1}, empty, directory=Path(tmp))
        self.assertIn("no hardware key", str(ctx.exception))

    def test_a_foreign_profile_is_passed_over_and_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            foreign = Path(tmp) / "foreign.json"
            foreign.write_text(json.dumps(document_for(RIG_A, profile_id="theirs")))
            selection = select_profile(
                fingerprint_from_facts(
                    LocalFacts(cards=cards(("GPU-zzzz", "Other", 16, "")))
                ),
                paths=[foreign],
            )
        self.assertIsNone(selection.profile)
        self.assertEqual(len(selection.rejected), 1)
        self.assertIn("theirs", selection.render())

    def test_an_exact_match_beats_a_model_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            twin = Path(tmp) / "twin.json"
            twin.write_text(json.dumps(document_for(RIG_A_TWIN, profile_id="twin")))
            mine = Path(tmp) / "mine.json"
            mine.write_text(json.dumps(document_for(RIG_A, profile_id="mine")))
            selection = select_profile(
                fingerprint_from_facts(RIG_A), paths=[twin, mine]
            )
        self.assertEqual(selection.profile.profile_id, "mine")
        self.assertIs(selection.scope, MatchScope.EXACT)
        self.assertEqual(
            [p.split("/")[-1] for p, _ in selection.rejected], ["twin.json"]
        )

    def test_unreadable_candidates_are_reported_not_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.json"
            broken.write_text("{not json")
            selection = select_profile(fingerprint_from_facts(RIG_A), paths=[broken])
        self.assertIsNone(selection.profile)
        self.assertEqual(len(selection.unreadable), 1)
        self.assertIn("broken.json", selection.render())

    def test_trust_override_applies_an_unmatched_explicit_profile(self):
        """The one legitimate escape hatch, and it says what it overrode."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "explicit.json"
            path.write_text(json.dumps(document_for(RIG_A, profile_id="explicit")))
            other = fingerprint_from_facts(
                LocalFacts(cards=cards(("GPU-zzzz", "Other", 16, "")))
            )
            import os

            previous = os.environ.get(PROFILE_PATH_ENV)
            os.environ[PROFILE_PATH_ENV] = str(path)
            try:
                refused = select_profile(other, paths=[path], trust_explicit=False)
                trusted = select_profile(other, paths=[path], trust_explicit=True)
            finally:
                if previous is None:
                    os.environ.pop(PROFILE_PATH_ENV, None)
                else:
                    os.environ[PROFILE_PATH_ENV] = previous
        # can-fail proof: the same call without the override refuses.
        self.assertIsNone(refused.profile)
        self.assertIsNotNone(trusted.profile)
        self.assertIn("overrides", trusted.match.reason.lower() + " overrides")


if __name__ == "__main__":
    unittest.main()
