# SPDX-License-Identifier: Apache-2.0
"""#1362 -- three 27B fossils, found by running the 4B-FP8 transition vehicle.

The desk run of RedHatAI/Qwen3.5-4B-FP8-dynamic (7.48 GiB, 32 layers, LLLA x8)
exposed constants that were measured on the 27B and say so nowhere:

  (3) Sigma H is solved from the PREVIOUS BOOT's census. The 4B would have been
      handed `Sigma H 48240 MiB = per card the LARGER of [tag census 47512,
      measured dormant image 48672]` -- digit for digit weg2xsn25's -- i.e. a
      47 GiB ring for a 7.48 GiB model, dying in W48 + 3x W51 without ever
      naming the cause. Hen-and-egg: the record can only come from a boot, and
      the boot cannot start because it is priced from the old model.
  (1) P_PP_STAGE_RATIO_SCORES=(32,18,14) and MEASURED_MS_PER_LAYER are a
      64-layer measurement with no model reference. On 32 layers the chain ends
      in a BARE ValueError from the cut solver -- the only unnamed refusal in
      the whole launch path.
  (4) `--served-model-name "Qwen3.8-27B"` was a literal on EVERY boot: the name
      the front routes on and every probe asserts against, so a 4B boot answers
      under the 27B's name. A silent identity swap in the one field whose job
      is identity.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher as lc
from sglang.test.test_utils import CustomTestCase

#: The two models this ticket has to keep apart, as the launcher sees them.
M27 = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8"
M4B = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.5-4B-FP8-dynamic"


def _calib(d, digest, **over):
    rec = {
        "schema": hl.CALIB_SCHEMA, "model_digest": digest,
        "measured_ms_per_layer": [8.10, 35.16, 33.59],
        "measured_counts": [11, 11, 10], "measured_attn_counts": [2, 3, 3],
        "calibration_prefix_tokens": 4096, "provenance": "boot weg2q4b @ test",
    }
    rec.update(over)
    with open(os.path.join(d, f"{digest}.json"), "w") as f:
        json.dump(rec, f)
    return rec


class TheDigestSeparatesTheModels1362(CustomTestCase):
    def test_two_models_never_share_a_digest(self):
        self.assertNotEqual(hl.model_digest(M27), hl.model_digest(M4B))

    def test_the_digest_is_stable_and_names_the_model(self):
        self.assertEqual(hl.model_digest(M4B), hl.model_digest(M4B + "/"))
        self.assertTrue(hl.model_digest(M4B).startswith("Qwen3.5-4B-FP8-dynamic@"))


class TheCalibrationIsNeverBorrowed1362(CustomTestCase):
    """(1): a 64-layer measurement may not be spent on a 32-layer model."""

    def test_a_missing_calibration_refuses_by_name_not_by_ValueError(self):
        with tempfile.TemporaryDirectory() as d:
            rec, why = hl.read_pp_calibration(hl.model_digest(M4B), d)
            self.assertIsNone(rec)
            self.assertIn("no PP calibration", why)
            exc = hl.refuse_foreign_calibration(hl.model_digest(M4B), why)
            self.assertIsInstance(exc, hl.Weg2ModelIdentityMismatch)
            msg = str(exc)
            self.assertIn("W99 Weg2ModelIdentityMismatch", msg)
            self.assertIn("MEASURED_MS_PER_LAYER", msg)
            self.assertIn("bare ValueError", msg)
            self.assertIn("measuring run", msg)

    def test_a_file_holding_another_models_numbers_is_refused(self):
        """The exact transfer this check exists to stop."""
        with tempfile.TemporaryDirectory() as d:
            _calib(d, hl.model_digest(M4B), model_digest=hl.model_digest(M27))
            rec, why = hl.read_pp_calibration(hl.model_digest(M4B), d)
            self.assertIsNone(rec)
            self.assertIn("carries model_digest", why)

    def test_ms_and_counts_are_validated_as_a_PAIR(self):
        with tempfile.TemporaryDirectory() as d:
            dg = hl.model_digest(M4B)
            _calib(d, dg, measured_counts=None, stage_layer_counts=None)
            rec, why = hl.read_pp_calibration(dg, d)
            self.assertIsNone(rec)
            self.assertIn("as a PAIR", why)
            _calib(d, dg, measured_attn_counts=[2, 3])      # one short
            rec, why = hl.read_pp_calibration(dg, d)
            self.assertIsNone(rec)
            self.assertIn("disagrees with itself", why)

    def test_a_refused_measurement_is_an_absence_WITH_a_reason(self):
        """'refused at measuring time' and 'nobody measured' are different
        states, and only one is fixed by booking another window."""
        with tempfile.TemporaryDirectory() as d:
            dg = hl.model_digest(M4B)
            _calib(d, dg, refused="P died in load, no ms figures")
            rec, why = hl.read_pp_calibration(dg, d)
            self.assertIsNone(rec)
            self.assertIn("REFUSED at measuring time", why)
            self.assertIn("P died in load", why)

    def test_the_matching_record_is_returned_with_its_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            dg = hl.model_digest(M4B)
            _calib(d, dg)
            rec, why = hl.read_pp_calibration(dg, d)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["measured_counts"], [11, 11, 10])
            self.assertIn("MEASURED for model", why)
            self.assertIn("weg2q4b", why)


class TheServedNameFollowsTheModel1362(CustomTestCase):
    """(4): the literal is gone; the name is derived."""

    def test_the_27B_keeps_the_name_it_always_had(self):
        self.assertEqual(lc.served_model_name(M27), "Qwen3.8-27B-INT8")

    def test_the_4B_does_not_answer_under_the_27Bs_name(self):
        self.assertEqual(lc.served_model_name(M4B), "Qwen3.5-4B-FP8-dynamic")
        self.assertNotIn("27B", lc.served_model_name(M4B))

    def test_a_snapshot_path_resolves_to_the_repo_not_the_commit(self):
        p = ("/cache/models--RedHatAI--Qwen3.5-4B-FP8-dynamic/snapshots/"
             "397b7ba47a99b3221ebfc0cfc5a279118cb733ad")
        self.assertEqual(lc.served_model_name(p), "RedHatAI/Qwen3.5-4B-FP8-dynamic")

    def test_an_override_wins(self):
        self.assertEqual(lc.served_model_name(M4B, "my-name"), "my-name")

    def test_the_literal_is_gone_from_the_argv_builder(self):
        """RATCHET: red before this commit, where the string was hard-coded."""
        import inspect
        src = inspect.getsource(lc)
        i = src.index('"--served-model-name"')
        self.assertIn("served_model_name(model)", src[i:i + 120])


class TheSurchargeIsPrintedNotAdopted1362(CustomTestCase):
    """(3): the manifest anchor's 40 % shortfall, named and borrowed openly."""

    def test_the_ratio_is_the_measured_one(self):
        self.assertAlmostEqual(hl.MANIFEST_CENSUS_SURCHARGE, 47512 / 32783, places=3)

    def test_it_says_it_is_borrowed_and_conservative(self):
        s = hl.MANIFEST_CENSUS_SURCHARGE_SOURCE
        self.assertIn("BORROWED", s)
        self.assertIn("conservative", s)
        self.assertIn("32783", s)
        self.assertIn("47512", s)

    def test_it_makes_the_ring_bigger_never_smaller(self):
        """DANGER DIRECTION: a foreign number may only over-size here."""
        self.assertGreater(hl.MANIFEST_CENSUS_SURCHARGE, 1.0)


if __name__ == "__main__":
    unittest.main()


class TheDormantImageIsNeverSpentOnAnotherModel1362(CustomTestCase):
    """(3) WIRED, not merely built -- the [21b] lesson applied in advance.

    A gate that exists and is never called is the state this line has paid for
    four times. So: the launcher STAMPS the digest into the record, and the
    consumer REFUSES a foreign one.
    """

    def test_the_launcher_stamps_the_digest_into_the_record(self):
        """RATCHET on reachability: red before this commit (0 occurrences)."""
        import inspect
        src = inspect.getsource(lc)
        i = src.index("host_ledger.dormant_image_sample(")
        call = src[i:src.index("log(host_ledger.format_dormant_image", i)]
        # #1362 [22-fix2] TIGHTENED: [22] pinned the PATH digest here, which no
        # reader ever compares against -- so the stamp was reachable but inert
        # and every record this boot wrote would have stayed legacy forever.
        # The pin is now the CONTENT digest, the identity the readers key on.
        self.assertIn("model_digest_=(host_ledger.checkpoint_digest(ns.model)[0]", call)

    def test_the_record_carries_it(self):
        rec = hl.dormant_image_sample(
            group="P", shmem_before_bytes=0, shmem_after_bytes=0, pids=[],
            weight_tags_gib=1.0, interleaved=False, boot_tag="t", commit="c",
            model_digest_=hl.model_digest(M4B))
        self.assertEqual(rec["model_digest"], hl.model_digest(M4B))

    def test_a_foreign_image_refuses_by_name(self):
        entry = {"boot_tag": "weg2xsn25", "commit": "b568f9afd5",
                 "model_digest": hl.model_digest(M27)}
        with self.assertRaises(hl.Weg2ModelIdentityMismatch) as cm:
            hl.refuse_foreign_image(entry, hl.model_digest(M4B))
        msg = str(cm.exception)
        self.assertIn("W99 Weg2ModelIdentityMismatch", msg)
        self.assertIn("47 GiB ring for a 7.48 GiB model", msg)
        self.assertIn("W48", msg)
        self.assertIn("1.449", msg)          # the named way forward

    def test_an_empty_digest_is_UNKNOWN_and_not_matching(self):
        """DANGER DIRECTION: a pre-#1362 record must not pass as 'same model'."""
        with self.assertRaises(hl.Weg2ModelIdentityMismatch) as cm:
            hl.refuse_foreign_image({"boot_tag": "old"}, hl.model_digest(M4B))
        self.assertIn("NO model digest", str(cm.exception))

    def test_the_matching_model_passes(self):
        dg = hl.model_digest(M27)
        hl.refuse_foreign_image({"model_digest": dg}, dg)   # must not raise


M4B_REAL = ("/root/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-4B-FP8-dynamic/"
            "snapshots/397b7ba47a99b3221ebfc0cfc5a279118cb733ad")
#: The boot seat's own record, keyed by the CONTENT digest.
REAL_4B_DIGEST = "89dcd9c419533886144102a0df37d292375d5ea678f80d2f8fd19c68413b9675"


class TheDigestIsTheCheckpointsNotThePaths1362(CustomTestCase):
    """[22-fix]: the path digest could never have found the record.

    Measured: the path form yields `397b7ba47a99...@138b30a8` -- the SNAPSHOT
    COMMIT -- while the record is keyed `89dcd9c4195338...`. The same checkpoint
    re-downloaded under a new snapshot hash would orphan its own calibration.
    """

    def test_the_content_digest_reproduces_the_boot_seats_record_bit_for_bit(self):
        if not os.path.isdir(M4B_REAL):
            self.skipTest("4B checkpoint not on this box")
        got, why = hl.checkpoint_digest(M4B_REAL)
        self.assertEqual(got, REAL_4B_DIGEST)
        self.assertIn("snapshot-independent", why)

    def test_the_path_digest_would_have_missed_it(self):
        self.assertNotEqual(hl.model_digest(M4B_REAL), REAL_4B_DIGEST)

    def test_an_unreadable_checkpoint_is_an_absence_with_a_reason(self):
        got, why = hl.checkpoint_digest("/no/such/model")
        self.assertIsNone(got)
        self.assertIn("no stable model identity", why)


class TheReadersAreREACHEDFromProduction1362(CustomTestCase):
    """REACHABILITY RATCHETS -- red on 8509b9ea68, where both counts were 0.

    8509b9ea68 built `read_pp_calibration` and `refuse_foreign_image` and called
    NEITHER from the production path: counted 0 and 0 outside host_ledger and
    tests. That is the fourth present-but-unwired instance in this family, and
    it was produced in the same commit whose message cited the lesson. Counting
    call sites is the only form of this check that does not depend on my
    remembering.
    """

    def _prod_src(self):
        import inspect
        from sglang.srt.weg2 import launcher as lc2
        return inspect.getsource(lc2)

    def test_the_calibration_reader_is_called_on_the_p_cut_path(self):
        src = self._prod_src()
        self.assertGreaterEqual(src.count("host_ledger.read_pp_calibration("), 1)
        self.assertGreaterEqual(src.count("refuse_foreign_calibration("), 1)
        # ...at the site that otherwise hands the incumbent vector over unasked
        i = src.index("stage_ratio = stage_ratio or _csv(P_PP_STAGE_RATIO_SCORES)")
        self.assertIn("read_pp_calibration", src[max(0, i - 2000):i])

    def test_the_image_check_is_reached_through_price(self):
        import inspect
        self.assertIn("want_digest=model_digest_want",
                      inspect.getsource(hl.price))
        self.assertIn("refuse_foreign_image",
                      inspect.getsource(hl.resolve_image_terms))
        self.assertIn("model_digest_want=", self._prod_src())

    def test_the_launcher_computes_the_CONTENT_digest_not_the_path_one(self):
        src = self._prod_src()
        self.assertIn("host_ledger.checkpoint_digest(ns.model)", src)


class TheRealRecordDrivesTheRefusal1362(CustomTestCase):
    """ACCEPTANCE against the boot seat's REAL file, not a double."""

    def test_the_refusal_carries_the_measured_reason_verbatim(self):
        rec, why = hl.read_pp_calibration(REAL_4B_DIGEST)
        if rec is None and "does not exist" in why:
            self.skipTest("4B calibration record not on this box")
        self.assertIsNone(rec)
        self.assertIn("REFUSED at measuring time", why)
        self.assertIn("KERNEL-ARCH-ABBRUCH", why)
        msg = str(hl.refuse_foreign_calibration(REAL_4B_DIGEST, why))
        self.assertIn("W99 Weg2ModelIdentityMismatch", msg)
        self.assertIn("KERNEL-ARCH-ABBRUCH", msg)
        self.assertIn("measuring run", msg)

    def test_a_64_layer_model_keeps_the_incumbent_vector(self):
        """The 27B measured the constants; it may go on using them."""
        self.assertEqual(lc.CALIBRATION_LAYERS, 64)
        self.assertEqual(hl.checkpoint_layers(M27) or 64, 64)


class LegacyRecordTransition(CustomTestCase):
    """#1362 [22-fix2] -- the hen-and-egg [22-fix] re-created in its own fix.

    [22-fix] wired `refuse_foreign_image` into `price()` and every dormant-image
    record on this rig instantly became unusable: they were all written BEFORE
    #1362, so they carry no `model_digest`, and the rule "empty means unknown"
    refused the very boots that wrote them. Measured on the gdncov serving form:
    rc=2, zero ARM lines, W99 naming weg2xsn27 @ 3468c2b535.

    The transition path admits a record with NO digest when this boot's group-P
    FORM KEY matched the source boot's -- the form key carries `--model-path`
    and excludes labels (`FORM_KEY_EXCLUDED_FLAGS`), so a match is positive
    evidence of the same checkpoint path, not an absence of evidence. It prints
    one line per admitted record, and the line is RETURNED so a caller can read
    it rather than only logged.

    Everything else is unchanged: a FOREIGN digest is still W99 even under a
    form-key match (a digest that disagrees is evidence, not silence), and a
    record with no digest AND no form-key match is still refused.
    """

    REC = {"boot_tag": "weg2xsn27", "commit": "3468c2b535"}
    WANT = "5a324e4044bf915181537d1481662a3050e984acd2eb3977174a54aba0ab3143"

    def test_1_legacy_record_with_form_key_match_is_admitted_and_names_itself(self):
        line = hl.refuse_foreign_image(dict(self.REC), self.WANT, form_key_match=True)
        # READ, not merely emitted: an instrument without a reader is the
        # defect the train seat named for exactly this line.
        self.assertIsNotNone(line, "legacy record under a form-key match must be admitted")
        self.assertIn("WEG2-MODEL-IDENTITY LEGACY", line)
        self.assertIn("record=weg2xsn27", line)
        self.assertIn("digest=absent", line)
        self.assertIn("form_key=match", line)
        self.assertIn("accepted", line)

    def test_2_foreign_digest_stays_w99_even_under_a_form_key_match(self):
        rec = dict(self.REC, model_digest="f" * 64)
        with self.assertRaises(hl.Weg2ModelIdentityMismatch) as cm:
            hl.refuse_foreign_image(rec, self.WANT, form_key_match=True)
        self.assertIn("W99", str(cm.exception))

    def test_3_legacy_record_without_form_key_match_stays_refused(self):
        with self.assertRaises(hl.Weg2ModelIdentityMismatch) as cm:
            hl.refuse_foreign_image(dict(self.REC), self.WANT, form_key_match=False)
        self.assertIn("W99", str(cm.exception))

    def test_4_the_legacy_line_reaches_the_printed_ledger_lines(self):
        """The reachability half: a returned line nobody appends is also unread."""
        rec = {"P": dict(self.REC), "D": dict(self.REC)}
        hl.resolve_image_terms(rec, want_digest=self.WANT, form_key_match=True)
        self.assertTrue(
            any("WEG2-MODEL-IDENTITY LEGACY" in x for x in hl.LEGACY_IDENTITY_LINES),
            "resolve_image_terms must hand the legacy line onward to the ledger",
        )

    def test_5_the_running_boot_stamps_the_content_digest_not_the_path_one(self):
        """Item (2): records written from here on carry the digest readers key on.

        [22] stamped `model_digest(path)` -- a PATH hash no reader compares
        against -- so every record this boot wrote would have stayed legacy
        forever and the transition path would never retire.
        """
        import inspect

        src = inspect.getsource(lc)
        self.assertIn("model_digest_=(host_ledger.checkpoint_digest(ns.model)[0]", src)
        self.assertNotIn("model_digest_=host_ledger.model_digest(ns.model)", src)

    def test_6_the_form_verdict_is_read_off_the_solved_table_not_the_plan(self):
        """The getattr trap, pinned.

        `HostRingPlan` has no `form_same`; the verdict lives on the RingTable it
        wraps (`plan.table.form_same`). A `getattr(ring_plan, "form_same",
        False)` therefore reads False on EVERY boot, and the transition path
        becomes dead code that no dry run can tell from a working one --
        measured: run gdncov2 refused W99 with `form key 24eb56724d73 MATCHES`
        printed four lines above the refusal. A default that stands in for an
        absent attribute is the same defect class as a silent 0 in the ledger.
        """
        import inspect

        src = inspect.getsource(lc)
        self.assertNotIn('getattr(ring_plan, "form_same"', src)
        self.assertIn('_rt = getattr(ring_plan, "table", None)', src)
        self.assertIn("form_key_matches=_form_key_matches", src)
        # HostRingPlan really does not carry it -- the assertion above is about
        # a real absence, not a style preference.
        self.assertFalse(hasattr(lc.HostRingPlan(), "form_same"))


class TheMarkerIsNeverInThePropaganda(CustomTestCase):
    """#1362 [22-fix3] -- an instrument must not write its own name into its prose.

    [22-fix2]'s W99 explained the transition path with the sentence "a matching
    key would admit it with a printed WEG2-MODEL-IDENTITY LEGACY line". That
    quoted the emitter's own marker inside a REFUSAL, so a census of "how many
    legacy admissions happened" scored 1 on a log carrying ZERO of them -- the
    train seat read exactly that off the default-form dry run, and reported
    `LEGACY 1` for a run whose genuine count was 0.

    This is #995's prose trap, re-created by the very file that was supposed to
    know better. The invariant it costs us is the one pinned here:

        count(lines carrying the LOG-PREFIXED marker) == count(legacy events)

    and the bare count must not exceed it, in EVERY form. Two forms are
    fixtured, because the trap only showed up in the one that refuses -- a test
    written against the happy form alone would have stayed green through it.

    FORM KEYS, measured on this box 2026-09-13 and named in the assertions
    because an rc or a count without its form key is not a number ([16], and
    the train seat handed it back to this seat for exactly this commit):

        24eb56724d73  serving form (xsn27 launch.sh argv)  rc=0, 2 events
        d0de83d152f0  default form (scripts/weg2/boot_weg2.sh) rc=2, 0 events
    """

    MARKER = "WEG2-MODEL-IDENTITY LEGACY"
    PREFIXED = "WEG2-LAUNCH " + MARKER
    REC = {"boot_tag": "weg2xsn27", "commit": "3468c2b535"}
    WANT = "5a324e4044bf915181537d1481662a3050e984acd2eb3977174a54aba0ab3143"

    #: (label, form key measured on the box, does the key match the source,
    #:  how many legacy events that form produces)
    FORMS = (
        ("serving (xsn27 launch.sh argv)", "24eb56724d73", True, 2),
        ("default (scripts/weg2/boot_weg2.sh)", "d0de83d152f0", False, 0),
    )

    def _log_for(self, form_key, form_key_match):
        """Synthesise that form's launcher log from the CODE, not from a paste.

        A pasted log is a photograph of a tree that may since have moved; these
        lines come out of the same functions the launcher calls, so the fixture
        cannot drift away from the emitter.
        """
        lines = []
        hl.LEGACY_IDENTITY_LINES.clear()
        try:
            hl.resolve_image_terms(
                {"P": dict(self.REC), "D": dict(self.REC, boot_tag="weg2xsn25")},
                want_digest=self.WANT,
                form_key_match=form_key_match,
            )
        except hl.Weg2ModelIdentityMismatch as e:
            lines.append(f"[2026-09-13T00:00:00Z] WEG2-LAUNCH REFUSED: {e}")
        for ll in dict.fromkeys(hl.LEGACY_IDENTITY_LINES):
            lines.append(f"[2026-09-13T00:00:00Z] WEG2-LAUNCH {ll}")
        return "\n".join(lines)

    def test_the_prefixed_count_equals_the_event_count_in_both_forms(self):
        for label, form_key, match, want_events in self.FORMS:
            with self.subTest(form=label, form_key=form_key):
                log = self._log_for(form_key, match)
                prefixed = sum(1 for ln in log.splitlines() if self.PREFIXED in ln)
                bare = sum(1 for ln in log.splitlines() if self.MARKER in ln)
                self.assertEqual(
                    prefixed, want_events,
                    f"form {form_key} ({label}): {prefixed} prefixed marker line(s), "
                    f"{want_events} legacy event(s) expected",
                )
                # THE TRAP ITSELF: a bare hit that is not a prefixed hit is a
                # mention, and a mention counted as an event is how a refusal
                # was read as an admission.
                self.assertEqual(
                    bare, prefixed,
                    f"form {form_key} ({label}): {bare} bare marker hit(s) vs "
                    f"{prefixed} real line(s) -- the difference is prose, and "
                    f"prose carrying the marker makes every census of it ambiguous",
                )

    def test_no_w99_line_carries_the_marker_in_either_form(self):
        for label, form_key, match, _ in self.FORMS:
            with self.subTest(form=label, form_key=form_key):
                log = self._log_for(form_key, match)
                offenders = [
                    ln for ln in log.splitlines() if "W99" in ln and self.MARKER in ln
                ]
                self.assertEqual(
                    offenders, [],
                    f"form {form_key} ({label}): a W99 refusal quotes the legacy "
                    f"marker. Describe the line, never quote its marker (#995).",
                )

    def test_the_refusal_still_explains_the_transition_path(self):
        """Removing the marker must not remove the EXPLANATION with it.

        The sentence exists because an operator reading a bare W99 cannot tell
        a record with a transition path from one without; dropping it to dodge
        the census trap would trade one blindness for another.
        """
        log = self._log_for("d0de83d152f0", False)
        self.assertIn("legacy-identity line", log)
        self.assertIn("form key does NOT match", log)

    def test_the_emitter_still_carries_the_marker(self):
        """The complement: the trap is fixed by moving the marker, not deleting it."""
        line = hl.refuse_foreign_image(dict(self.REC), self.WANT, form_key_match=True)
        self.assertTrue(line.startswith(self.MARKER), line[:60])
