"""#1261 train fix 5: the ring's FORM GATE and its independent L6 source.

Boot weg2tr2 died because the host ring was sized from boot weg2rg6, whose
group P carried no MTP head, while this train's group P carries one on its LAST
stage.  Two of three stages matched the predecessor TO THE MEGABYTE (13860 and
7548), which is what made the third one's 2426-with-204-free look like a
runtime accident rather than a stale table -- and the runtime's own W31 line
said ``the launch check (L6) was violated``, because L6 was comparing the ring
against the same census that had sized it.

Four properties are pinned here, each red before the fix:

1. FORM KEY -- a source boot whose group-P form differs is not charged as a
   measurement of this boot, and one whose form matches still is.
2. SIZING FROM THE CHECKPOINT -- the per-stage weight terms come from the
   safetensors headers and the shipped cut, with the drafter on the last stage.
3. L6 AGAINST AN INDEPENDENT SOURCE -- a span below the checkpoint's own weight
   bytes is refused by name (W49) at launch.
4. THE W31 COLLISION -- the front's ``Weg2TpPrefillExceeded`` no longer shares a
   number with ``Weg2HostRingExhausted``.

The evidence-tree cases skip where the two real boot logs are absent (they live
only on the rig); the arithmetic cases are synthetic and run anywhere.
"""

import json
import os
import struct
import unittest

from sglang.srt.weg2 import front, host_ledger, ring_table

#: #1264 fix 2c: THE BOOT LOGS THIS SUITE READS ARE NOW IN-TREE.
#:
#: They used to be `/spinning/evidence-665-f1`, the LIVE evidence tree, and that
#: made the suite non-hermetic in a way that bit twice. `ring_table.solve` reads
#: TWO things out of that directory: the stem's own logs (immutable once a boot
#: ends) and `weg2_measured_record.json`, an APPEND-ONLY sidecar every later
#: boot writes to. So boots t2a and t2b appended dormant samples and moved the
#: priced spans of a solve pinned to weg2rg6 -- the #1264 (B) ratchet -- and
#: these guards went red for a reason that had nothing to do with the form gate
#: they exist to protect. A guard that a later boot can flip is not a guard.
#:
#: The fixture is COPIED VERBATIM, never synthesized: every line is a real line
#: from the real boot, kept when it matches a predicate one of `ring_table`'s
#: own parsers applies, with the source sha256 and the kept/total counts in
#: PROVENANCE.json beside it. Proven faithful at build time -- solve() on the
#: fixture returns spans, form_same and residuals byte-equal to solve() on the
#: live tree, for all three p_argv arms.
#:
#: The sidecar is copied too, WITH t2a's and t2b's foreign samples still in it.
#: That is deliberate: those two entries are exactly what used to poison this
#: solve, so keeping them turns the fixture into a standing proof of the (B)
#: rule instead of a way of dodging it. See `test_foreign_boot_samples_cannot_
#: reach_this_solve`.
EVIDENCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "ring_form_gate_1261"
)
#: The tree this suite must never touch again.
_FORBIDDEN_ROOTS = ("/spinning/evidence-665-f1", "/spinning/gpu-arb")
RG6 = "boot_weg2_weg2rg6_7f88b1c75d_0908_070324"
TR2 = "boot_weg2_weg2tr2_d047f8e80b_0908_110126"
MIB = 1024 * 1024


class Card:
    def __init__(self, uuid, nvml_index, name):
        self.uuid, self.nvml_index, self.name = uuid, nvml_index, name


CARDS = [
    Card("GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d", 1, "NVIDIA GeForce RTX 5090"),
    Card("GPU-5c648f96-be1d-42d5-0221-34d11ab137f7", 0, "NVIDIA GeForce RTX 3080"),
    Card("GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4", 2, "NVIDIA GeForce RTX 3080"),
]


def have_evidence():
    """True on every box now, and that is the point of fix 2c.

    This used to gate the suite on the rig's own evidence tree, so the guards
    that matter most SKIPPED on the remote desk -- silently, which is the worst
    way for a guard to be absent. The fixture is in the repository, so they run
    wherever the tests run.
    """
    return all(
        os.path.exists(os.path.join(EVIDENCE, f"{stem}.{half}.log"))
        for stem in (RG6, TR2)
        for half in ("front",)
    ) and os.path.exists(os.path.join(EVIDENCE, f"{RG6}.P.log"))


class _NoLiveEvidence:
    """Context manager: any read under the live evidence/gpu-arb trees RAISES.

    The hermeticity claim has to be ENFORCED, not documented -- a later edit
    that reaches for `/spinning/evidence-665-f1` again would otherwise re-open
    the exact hole fix 2c closes, and it would pass on the rig and fail only on
    a machine that does not have the tree.

    Scoped to the two mutable data trees rather than to all of `/spinning`: the
    sglang package and this worktree live under `/spinning` too, and pytest
    reads source files lazily when it renders a traceback, so a blanket ban
    would fire on the test harness itself rather than on the defect.
    """

    def __enter__(self):
        import builtins

        self._open, self._listdir = builtins.open, os.listdir

        def guard(path, *a, **kw):
            s = str(path)
            for root in _FORBIDDEN_ROOTS:
                if s.startswith(root):
                    raise AssertionError(
                        f"NON-HERMETIC: this suite opened {s!r}. The boot logs and "
                        f"the measured-record sidecar are fixtures under "
                        f"fixtures/ring_form_gate_1261 precisely so that no boot "
                        f"can move these expectations again (#1264 fix 2c)."
                    )
            return None

        def _open_guarded(path, *a, **kw):
            guard(path)
            return self._open(path, *a, **kw)

        def _listdir_guarded(path=".", *a, **kw):
            guard(path)
            return self._listdir(path, *a, **kw)

        builtins.open = _open_guarded
        os.listdir = _listdir_guarded
        return self

    def __exit__(self, *exc):
        import builtins

        builtins.open, os.listdir = self._open, self._listdir
        return False


def write_synthetic_checkpoint(directory, *, n_layers, attn_every, with_mtp=True):
    """A safetensors dir whose HEADERS carry the shapes, and no payload.

    ``checkpoint_weight_terms`` reads headers only, so a zero-byte body is a
    complete input -- which is what makes this test hermetic and instant.
    """
    tensors = {}

    def add(name, elems):
        tensors[name] = {"dtype": "I8", "shape": [elems], "data_offsets": [0, 0]}

    for i in range(n_layers):
        if i % attn_every == 0:
            add(f"model.layers.{i}.self_attn.q_proj.weight", 100)
        else:
            add(f"model.layers.{i}.mlp.down_proj.weight", 200)
    add("model.embed_tokens.weight", 1000)
    add("lm_head.weight", 2000)
    add("model.visual.patch_embed.weight", 500)
    if with_mtp:
        add("mtp.fc.weight", 300)
    os.makedirs(directory, exist_ok=True)
    header = json.dumps(tensors).encode()
    with open(os.path.join(directory, "model-00001-of-00001.safetensors"), "wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
    return directory


BASE_ARGV = [
    "/py", "-m", "sglang.launch_server",
    "--model-path", "/ckpt",
    "--tp-size", "1", "--pp-size", "3",
    "--pp-stage-ratio", "4,3,3", "--pp-attn-stage-ratio", "2,1,1",
    "--enable-memory-saver", "--enable-weights-cpu-backup",
]


class FormKey(unittest.TestCase):
    def test_drafter_flags_change_the_form(self):
        """The exact difference weg2tr2 had over weg2rg6, in one assertion."""
        plain, _ = ring_table.p_form_key(BASE_ARGV)
        spec, _ = ring_table.p_form_key(
            BASE_ARGV + ["--speculative-algorithm", "NEXTN",
                         "--speculative-draft-kv-only"]
        )
        self.assertNotEqual(plain, spec)

    def test_cut_change_changes_the_form(self):
        a, _ = ring_table.p_form_key(BASE_ARGV)
        moved = list(BASE_ARGV)
        moved[moved.index("4,3,3")] = "5,3,2"
        b, _ = ring_table.p_form_key(moved)
        self.assertNotEqual(a, b)

    def test_checkpoint_change_changes_the_form(self):
        a, _ = ring_table.p_form_key(BASE_ARGV)
        other = list(BASE_ARGV)
        other[other.index("/ckpt")] = "/other-ckpt"
        b, _ = ring_table.p_form_key(other)
        self.assertNotEqual(a, b)

    def test_flag_order_is_not_a_form_change(self):
        """MEASURED on the rig: weg2rg6 emits --disable-overlap-schedule and
        --max-running-requests early, weg2tr2 emits them late, same values.  An
        order-sensitive key would refuse a table for a difference that is not
        one, and hide the drafter difference in the noise."""
        a, _ = ring_table.p_form_key(BASE_ARGV + ["--x", "1", "--y", "2"])
        b, _ = ring_table.p_form_key(BASE_ARGV + ["--y", "2", "--x", "1"])
        self.assertEqual(a, b)

    def test_excluded_flags_cannot_reach_the_key(self):
        """The three host-ledger flags are priced FROM the ring, so a key that
        moved with them could never match any predecessor at all."""
        a, _ = ring_table.p_form_key(BASE_ARGV + ["--hicache-size", "1", "--port", "30031"])
        b, _ = ring_table.p_form_key(BASE_ARGV + ["--hicache-size", "48", "--port", "30099"])
        self.assertEqual(a, b)

    def test_the_exclusion_list_is_a_blacklist(self):
        """A whitelist stops discriminating the day a new flag is added, which
        is exactly how this defect class arrives.  Pinned as policy, not style."""
        self.assertEqual(ring_table.FORM_KEY_POLICY, "blacklist")
        unknown, _ = ring_table.p_form_key(BASE_ARGV + ["--some-future-form-flag"])
        known, _ = ring_table.p_form_key(BASE_ARGV)
        self.assertNotEqual(unknown, known)


class CheckpointStageWeights(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="ring-form-gate-")
        write_synthetic_checkpoint(self.dir, n_layers=10, attn_every=5)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def test_terms_are_per_stage_and_named(self):
        got = ring_table.checkpoint_stage_weights(self.dir, [4, 3, 3], [2, 1, 1], False)
        self.assertEqual([s.attn_layers for s in got], [2, 1, 1])
        self.assertEqual([s.linear_layers for s in got], [2, 2, 2])
        # embedding on stage 0 only, lm_head on the last stage only
        self.assertGreater(got[0].embedding_mib, 0)
        self.assertEqual(got[1].embedding_mib, 0)
        self.assertEqual(got[0].lm_head_mib, 0)
        self.assertGreater(got[2].lm_head_mib, 0)
        # the vision tower is carried by EVERY stage (calibrated on weg2rg6)
        self.assertEqual(
            {round(s.replicated_mib, 6) for s in got}, {round(got[0].replicated_mib, 6)}
        )
        self.assertGreater(got[0].replicated_mib, 0)

    def test_the_drafter_lands_on_the_last_stage_only(self):
        without = ring_table.checkpoint_stage_weights(self.dir, [4, 3, 3], [2, 1, 1], False)
        with_ = ring_table.checkpoint_stage_weights(self.dir, [4, 3, 3], [2, 1, 1], True)
        self.assertEqual([s.drafter_mib for s in with_][:2], [0.0, 0.0])
        self.assertGreater(with_[2].drafter_mib, 0)
        self.assertEqual(without[0].total_mib, with_[0].total_mib)
        self.assertGreater(with_[2].total_mib, without[2].total_mib)

    def test_the_drafter_carries_its_own_embedding_and_head(self):
        """MEASURED, not assumed: weg2tr2's PP2 log carries a SECOND
        ``Load weight end ... type=Qwen3_5ForCausalLMMTP`` whose avail mem falls
        by ~4.00 units, against an mtp tensor sum of only 405 MiB.  A NEXTN head
        is its own model runner and materialises both again."""
        got = ring_table.checkpoint_stage_weights(self.dir, [4, 3, 3], [2, 1, 1], True)
        last, first = got[2], got[0]
        mtp_only = last.drafter_mib - first.embedding_mib - last.lm_head_mib
        self.assertGreater(mtp_only, 0)
        self.assertGreater(last.drafter_mib, mtp_only * 2)
        self.assertIn("second model runner", last.drafter_terms)

    def test_mismatched_cut_lengths_refuse_by_name(self):
        with self.assertRaises(ring_table.Weg2RingFormMismatch):
            ring_table.checkpoint_stage_weights(self.dir, [4, 3, 3], [2, 1], False)

    def test_argv_route_reads_cut_and_drafter_off_the_argv(self):
        argv = [
            "--model-path", self.dir,
            "--pp-stage-ratio", "4,3,3", "--pp-attn-stage-ratio", "2,1,1",
            "--speculative-algorithm", "NEXTN",
        ]
        got, why = ring_table.stage_weights_from_argv(argv)
        self.assertEqual(why, "")
        self.assertGreater(got[2].drafter_mib, 0)
        plain, _ = ring_table.stage_weights_from_argv(argv[:6])
        self.assertEqual(plain[2].drafter_mib, 0.0)

    def test_a_cutless_argv_is_a_named_absence_not_a_zero(self):
        got, why = ring_table.stage_weights_from_argv(["--model-path", self.dir])
        self.assertIsNone(got)
        self.assertIn("--pp-stage-ratio", why)


class IndependentL6(unittest.TestCase):
    """W49: span1 checked against the checkpoint, which no census touches."""

    def _card(self, span1, ckpt):
        c = ring_table.CardRing(uuid="GPU-x", nvml_index=2, name="RTX 3080")
        c.image_p_mib = span1
        c.image_d_mib = 1
        c.stage_index = 2
        c.ckpt_weights_mib = ckpt
        c.ckpt_terms = "synthetic"
        return c

    def test_a_short_span_is_refused_by_name(self):
        t = ring_table.RingTable(boot="b", instrument="i", lines_read=1)
        t.cards.append(self._card(span1=8504, ckpt=12429.0))
        bad = t.ckpt_refusals()
        self.assertEqual(len(bad), 1)
        self.assertIn("RING UNDER-SIZED", bad[0])
        self.assertIn("3925", bad[0].replace(".0", ""))  # 12429 - 8504

    def test_a_sufficient_span_passes(self):
        t = ring_table.RingTable(boot="b", instrument="i", lines_read=1)
        t.cards.append(self._card(span1=12548, ckpt=12429.0))
        self.assertEqual(t.ckpt_refusals(), [])

    def test_mutant_self_referential_l6_cannot_see_it(self):
        """MUTANT 2: L6 back to comparing the ring against the census that sized
        it.  Modelled by removing the independent term -- the row the census
        cannot produce -- and the check goes silent on the very table that
        killed weg2tr2.  That silence is the defect, and it is why the check may
        never be keyed on a term the census also supplies."""
        t = ring_table.RingTable(boot="b", instrument="i", lines_read=1)
        card = self._card(span1=8504, ckpt=12429.0)
        self.assertNotEqual(t.__class__(boot="b", instrument="i", lines_read=1), None)
        t.cards.append(card)
        self.assertEqual(len(t.ckpt_refusals()), 1)
        card.ckpt_weights_mib = 0.0  # the mutant: no independent source
        self.assertEqual(t.ckpt_refusals(), [])

    def test_the_row_states_its_terms(self):
        t = ring_table.RingTable(boot="b", instrument="i", lines_read=1)
        c = self._card(span1=12548, ckpt=12429.0)
        c.residual_mib = 118.2
        c.residual_source = "boot rg6 stage 2"
        t.cards.append(c)
        rows = [ln for ln in t.format_l6() if "RING-CKPT" in ln]
        self.assertEqual(len(rows), 1)
        self.assertIn("INDEPENDENT source", rows[0])
        self.assertIn("stage=2", rows[0])
        self.assertIn("residual 118.2", rows[0])


class RenumberedCode(unittest.TestCase):
    """W31 named TWO refusals on this rig; a census keyed on the number merged a
    fatal host-ring exhaustion with a serving-path re-route."""

    def test_the_front_refusal_is_no_longer_w31(self):
        self.assertTrue(front.X_REFUSAL_NAME.startswith("W50 "))
        self.assertNotIn("W31", front.X_REFUSAL_NAME)

    def test_the_detector_is_keyed_on_the_name_not_the_number(self):
        """The durable half: a wire protocol keyed on a renumberable label
        breaks silently at exactly the renumber that fixes the collision."""
        self.assertEqual(front.X_REFUSAL_MARKER, "Weg2TpPrefillExceeded")
        self.assertTrue(front.x_refusal_marker_in("... W50 Weg2TpPrefillExceeded ..."))
        self.assertTrue(front.x_refusal_marker_in("... W31 Weg2TpPrefillExceeded ..."))
        self.assertFalse(front.x_refusal_marker_in("W31 Weg2HostRingExhausted"))

    def test_the_host_ring_exhaustion_is_not_matched_by_the_front(self):
        self.assertFalse(
            front.is_x_refusal(503, "[host_ring.cpp] W31 Weg2HostRingExhausted ...")
        )


@unittest.skipUnless(have_evidence(), "the weg2rg6/weg2tr2 boot logs live only on the rig")
class AgainstTheRealBoots(unittest.TestCase):
    """The killer, reproduced and then refused, on the two real boot logs."""

    def setUp(self):
        # #1264 fix 2c: every test in this class runs with the live evidence and
        # gpu-arb trees SEALED. The fixture is the only data source, and an edit
        # that reaches past it fails here instead of on someone else's box.
        self._sealed = _NoLiveEvidence()
        self._sealed.__enter__()
        self.addCleanup(self._sealed.__exit__, None, None, None)
        self.tr2, _ = ring_table.parse_p_form(os.path.join(EVIDENCE, f"{TR2}.front.log"))
        self.rg6, _ = ring_table.parse_p_form(os.path.join(EVIDENCE, f"{RG6}.front.log"))
        self.assertIsNotNone(self.tr2)
        self.assertIsNotNone(self.rg6)

    def test_the_seal_can_actually_fail(self):
        """A guard nobody has seen refuse is not a guard (desk-written-never-
        executed). This is the can-fail proof for the seal itself."""
        for probe in (
            "/spinning/evidence-665-f1/weg2_measured_record.json",
            "/spinning/gpu-arb/weg2/WEG2_BUILD_DECISIONS_0906.md",
        ):
            with self.assertRaises(AssertionError) as ctx:
                open(probe)
            self.assertIn("NON-HERMETIC", str(ctx.exception))
        # ... and it does NOT seal the package or the fixture it must read.
        with open(os.path.join(EVIDENCE, f"{RG6}.front.log")) as f:
            self.assertTrue(f.readline())

    def test_the_suite_reads_no_live_evidence_at_all(self):
        """The whole solve path, under the seal: if `ring_table` reached for the
        live tree anywhere -- stems, logs, or the sidecar -- this raises."""
        t, _ = ring_table.solve(CARDS, EVIDENCE, RG6, p_argv=self.rg6)
        self.assertIsNotNone(t)
        self.assertEqual([c.span1_mib for c in t.cards], [13860, 7548, 8504])

    def test_the_two_forms_differ_by_the_drafter(self):
        a, an = ring_table.p_form_key(self.tr2)
        b, bn = ring_table.p_form_key(self.rg6)
        self.assertNotEqual(a, b)
        only_tr2 = set(an.split(" ")) - set(bn.split(" "))
        self.assertTrue(any(t.startswith("--speculative-") for t in only_tr2), only_tr2)

    def test_same_form_reproduces_the_measurement_exactly(self):
        """NO REGRESSION: a source boot of this boot's own form still yields its
        own measured spans, to the megabyte."""
        t, _ = ring_table.solve(CARDS, EVIDENCE, RG6, p_argv=self.rg6)
        self.assertIs(t.form_same, True)
        self.assertEqual([c.span1_mib for c in t.cards], [13860, 7548, 8504])

    def test_a_foreign_form_is_repriced_and_the_killer_disappears(self):
        """weg2tr2's own arithmetic: the two stages that fitted stay put, and the
        one whose contents changed is repriced until L6 is satisfied.

        #1264 fix 2c: THE ASSERTION IS THE RELATION, NOT A LITERAL. This read
        `assertGreater(spans[2], 12000)`, a threshold calibrated on a PP2
        checkpoint of 12429 MiB. That number stopped existing at `d7e7df4ed1`
        ("the ring is sized for the head that exists"): #1259 made the MTP head
        share the target's `lm_head`, so the drafter's own 2425 MiB head is
        never allocated and PP2's checkpoint is 10004 MiB. 12429 - 10004 =
        2425.0 exactly -- the test was red because the fork got BETTER, and a
        literal could not tell that from a regression.

        So it now asserts what the gate is actually for: the repriced span must
        COVER this form's own checkpoint, whatever that checkpoint becomes. The
        next change to the drafter moves both sides together and this stays
        green for the right reason.
        """
        t, _ = ring_table.solve(CARDS, EVIDENCE, RG6, p_argv=self.tr2)
        self.assertIs(t.form_same, False)
        spans = [c.span1_mib for c in t.cards]
        self.assertEqual(spans[:2], [13860, 7548], "the fitting stages stay put")
        # The stage was REPRICED UPWARD off the source's measurement ...
        same, _ = ring_table.solve(CARDS, EVIDENCE, RG6, p_argv=self.rg6)
        self.assertGreater(spans[2], same.cards[2].span1_mib)
        # ... and far enough that the independent L6 gate is satisfied.
        self.assertGreaterEqual(spans[2], t.cards[2].ckpt_weights_mib)
        self.assertEqual(t.ckpt_refusals(), [])
        self.assertIn("W48", t.form_verdict)

    def test_mutant_form_key_ignored_reproduces_the_killer(self):
        """MUTANT 1: the gate disarmed.  The predecessor's behaviour exactly --
        PP2 sized off a foreign boot's measurement while this form's stage will
        load more than that -- and the independent L6 catches it where the
        census could not.

        #1264 fix 2c: `assertGreater(armed - disarmed, 3000)` was the same stale
        literal in a second spelling (3925 = 12429 - 8504, the pre-#1259 gap).
        The property is not a gap size; it is that the disarmed table UNDER-SIZES
        the ring and L6 REFUSES it by name while the armed one passes.
        """
        t, _ = ring_table.solve(CARDS, EVIDENCE, RG6, p_argv=None)
        self.assertIsNone(t.form_same)
        self.assertEqual([c.span1_mib for c in t.cards], [13860, 7548, 8504])

        # THE KILLER, stated as the inequality it is: this form loads more than
        # the disarmed table reserves.
        armed, _ = ring_table.solve(CARDS, EVIDENCE, RG6, p_argv=self.tr2)
        need = armed.cards[2].ckpt_weights_mib
        self.assertLess(t.cards[2].span1_mib, need, "the killer must still be a killer")
        self.assertGreater(armed.cards[2].span1_mib, t.cards[2].span1_mib)
        self.assertGreaterEqual(armed.cards[2].span1_mib, need)

        # And L6 is the instrument that says so, on the DISARMED table -- the
        # census that sized it cannot, which is the whole point of W49.
        t.cards[2].ckpt_weights_mib = need
        t.cards[2].ckpt_terms = "weg2tr2 form, stage 2"
        refusals = t.ckpt_refusals()
        self.assertEqual(len(refusals), 1)
        self.assertIn("RING UNDER-SIZED", refusals[0])

    def test_foreign_boot_samples_cannot_reach_this_solve(self):
        """#1264 (B), pinned where it broke: the sidecar in this fixture STILL
        carries weg2t2a's and weg2t2b's dormant samples, and a solve pinned to
        weg2rg6 must be untouched by them.

        That is the whole ratchet, in one assertion. Before the stem-binding,
        `read_measured_record` returned the newest entry per group -- whatever
        booted last -- while the correction subtracted from it came from the
        STEM's own arm, so every new boot moved these numbers. Two boots, one
        subtraction.
        """

        sidecar = os.path.join(EVIDENCE, "weg2_measured_record.json")
        with open(sidecar) as f:
            tags = {s.get("boot_tag") for s in json.load(f)["samples"]}
        self.assertTrue(
            {"weg2t2a", "weg2t2b"} <= tags,
            "the fixture must KEEP the poisoning samples, or it proves nothing",
        )
        self.assertNotIn("weg2rg6", tags, "the stem itself has no sample here")

        # Bound to the stem -> the foreign samples are an ABSENCE, not a value.
        self.assertEqual(
            host_ledger.read_measured_record(sidecar, boot_tag="weg2rg6"), {}
        )
        t, _ = ring_table.solve(CARDS, EVIDENCE, RG6, p_argv=self.rg6)
        self.assertEqual([c.span1_mib for c in t.cards], [13860, 7548, 8504])

    def test_the_fixture_still_matches_its_provenance_record(self):
        """TAMPER-EVIDENCE, on every box, not only the one that built it.

        The source sha256 in PROVENANCE.json cannot be re-checked away from the
        rig (the 33/67 MB originals live only there), so the property that IS
        checkable everywhere is checked instead: each fixture still holds
        exactly the number of extracted lines the record claims, under the
        three-line header the extractor wrote. A hand-edit to make a red test
        green -- the failure mode a frozen fixture invites -- moves that count.
        """
        with open(os.path.join(EVIDENCE, "PROVENANCE.json")) as f:
            prov = json.load(f)
        self.assertGreaterEqual(len(prov), 5, "every fixture file must be recorded")
        for name, rec in prov.items():
            path = os.path.join(EVIDENCE, name)
            self.assertTrue(os.path.exists(path), f"{name} is recorded but missing")
            self.assertIn("source_sha256", rec)
            if not name.endswith(".log"):
                continue
            with open(path) as f:
                lines = f.read().splitlines()
            header = [ln for ln in lines[:3] if ln.startswith("#")]
            self.assertEqual(len(header), 3, f"{name} lost its provenance header")
            self.assertEqual(
                len(lines) - 3, rec["kept"],
                f"{name} holds {len(lines) - 3} extracted lines, record says "
                f"{rec['kept']} -- the fixture was edited after extraction",
            )
            self.assertLess(rec["kept"], rec["total"], "extraction must be a subset")

    def test_the_same_stem_prices_the_same_table_every_time(self):
        """DETERMINISM is the invariant fix 2c exists to hold: one stem, one
        table, however often it is solved and whatever else has booted."""
        first, _ = ring_table.solve(CARDS, EVIDENCE, RG6, p_argv=self.rg6)
        second, _ = ring_table.solve(CARDS, EVIDENCE, RG6, p_argv=self.rg6)
        self.assertEqual(
            [(c.span1_mib, round(c.residual_mib, 3)) for c in first.cards],
            [(c.span1_mib, round(c.residual_mib, 3)) for c in second.cards],
        )

    def test_the_residual_is_small_positive_and_uniform(self):
        """The calibration the whole derivation rests on: with the vision tower
        charged to every stage and the mtp tensors to none, weg2rg6's three
        measured spans exceed the checkpoint lower bound by ~118-140 MiB.  All
        POSITIVE, so the bound is genuinely one-sided."""
        t, _ = ring_table.solve(CARDS, EVIDENCE, RG6, p_argv=self.rg6)
        residuals = [c.residual_mib for c in t.cards]
        for r in residuals:
            self.assertGreater(r, 0)
            self.assertLess(r, 400)
        self.assertLess(max(residuals) - min(residuals), 100)

    def test_a_source_boot_without_a_p_argv_line_is_refused_by_name(self):
        import shutil
        import tempfile

        d = tempfile.mkdtemp(prefix="ring-form-noargv-")
        try:
            for half in ("P", "D"):
                shutil.copy(os.path.join(EVIDENCE, f"{RG6}.{half}.log"),
                            os.path.join(d, f"b.{half}.log"))
            src = os.path.join(EVIDENCE, f"{RG6}.front.log")
            with open(src, errors="replace") as fh, open(os.path.join(d, "b.front.log"), "w") as out:
                for line in fh:
                    if "group P argv:" not in line:
                        out.write(line)
            t, why = ring_table.solve(CARDS, d, None, p_argv=self.tr2)
            self.assertIsNone(t)
            self.assertIn("W48", why)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
