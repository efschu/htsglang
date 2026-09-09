"""#1264: draft KV on group P is a SWITCH, default on; ``off`` is the rg6 form.

``--draft-kv-on-p {on,off}`` decides whether the Weg-2 prefill group carries
the checkpoint's ``mtp.*`` head as a draft-KV-only producer (today's behaviour,
added by ae04c43ed1) or boots the rg6-proven form with no speculative flag at
all.  Group D is unchanged by either value: ``off`` removes the PRODUCER, not
speculative decode.

What is pinned here, each property red before the switch existed:

1. ARGV -- ``off`` puts ZERO ``--speculative*`` tokens on group P's argv and
   ``on`` puts all five (plus the ``--max-total-tokens`` term whose premise is
   the head), and group D's argv is byte-identical under both.
2. THE FORM KEY -- the two forms hash APART.  This is the mutant's target: the
   whole point of ``FORM_KEY_POLICY = "blacklist"`` is that a new form-changing
   flag discriminates with no code, and a key that collides would let a ring
   solved from an ``on`` boot be charged as a MEASUREMENT of an ``off`` one --
   the weg2tr2 defect with the switch as its new vector.
3. THE RING'S DRAFTER TERM -- under ``off`` the last stage is priced with NO
   drafter bytes, and the resulting PP2 requirement reproduces boot weg2rg6's
   OWN measured dormant image, because that boot's residual was defined as
   that image minus a no-drafter weight statement.  This is the arithmetic
   identity that makes "``off`` == the rg6-form derivation" checkable rather
   than asserted.
4. ONE PREDICATE -- the fact is read back off the argv by
   ``ring_table.p_carries_drafter`` and nowhere else, so the ring's drafter
   term and the launcher's W10/W11 gates cannot disagree with what shipped.
5. THE FRONT -- ``front_argv_for`` is byte-identical under both values (its
   draft handling is observational, not flagged).

The checkpoint cases skip where this rig's model tree is absent; the argv,
form-key and predicate cases are synthetic and run anywhere.
"""

import json
import unittest

import pytest

#: #1236: argv_p / argv_d take the RENDERED backend extra-config, not a
#: store size in GiB -- the store is a directory on disk sized from the P
#: KV pool, so a GiB number no longer means anything at this seam.
#: Spelled out here rather than imported from the launcher: this module
#: is read before its guarded launcher import, and the SHAPE is what the
#: argv seam is being tested on.
STORE_CFG = json.dumps(
    {"max_size": str(30 * 1024 ** 3), "min_free_space": str(32 * 1024 ** 3),
     "max_size_scope": "shared"},
    separators=(",", ":"),
)

try:
    from sglang.srt.weg2 import ring_table
    from sglang.srt.weg2.launcher import (
        P_DRAFT_KV_FLAGS,
        argv_d,
        argv_p,
        draft_kv_off_line,
    )
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(
        f"#249 default-device collection leak broke the import chain: {_import_err}",
        allow_module_level=True,
    )

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PY = "/nonexistent/python"
MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
BUDGETS = [27960, 17064, 16552]

#: This rig's shipped cut, and the stage the drafter lands on.
LAYER_SPLIT = [32, 18, 14]
ATTN_SPLIT = [8, 4, 4]

#: Boot weg2rg6's OWN numbers for PP2, off its RING-CKPT line: the measured
#: dormant image of stage 2, and the checkpoint weight statement that was
#: subtracted from it to leave the residual this tree still carries.  rg6's
#: group P carried NO MTP head, so its weight statement is exactly what an
#: ``off`` boot re-derives -- which is why the two must add back up.
RG6_PP2_MEASURED_IMAGE_MIB = 9730.0
RG6_PP2_RESIDUAL_MIB = 1344.2


#: #1305 item 2: the cap is the SHIPPED cut's priced pool, handed to argv_p by
#: the caller that holds PCutFacts (weg2sn5pre priced 42,11,11 at 463,763); it
#: is no longer a launcher constant, so this fixture supplies one.
P_CAP = 463763


def p(**kw):
    args = dict(
        py=PY, model=MODEL, budgets=BUDGETS, s_gb=48, m_mib=2400, store_cfg=STORE_CFG,
        extra=[], p_max_total_tokens=P_CAP,
    )
    args.update(kw)
    return argv_p(**args)


class TestArgvCarriesTheSwitch(unittest.TestCase):
    """Property 1: the argv is the only place the two forms differ."""

    def test_off_carries_zero_speculative_flags(self):
        argv = p(draft_kv_on_p=False)
        spec = [t for t in argv if t.startswith("--speculative")]
        self.assertEqual(
            spec, [],
            "group P must carry NO speculative flag under --draft-kv-on-p off; "
            f"found {spec}",
        )
        # The --max-total-tokens term goes with the head: its derivation
        # subtracts the head's own resident budget from the last stage's pool,
        # so keeping it under `off` would hand back 1618 MiB less pool than
        # the form can use, and the rg6 baseline carried no such flag.
        self.assertNotIn("--max-total-tokens", argv)

    def test_on_carries_all_five_speculative_flags(self):
        argv = p(draft_kv_on_p=True)
        for flag in (
            "--speculative-algorithm",
            "--speculative-num-steps",
            "--speculative-eagle-topk",
            "--speculative-num-draft-tokens",
            "--speculative-draft-kv-only",
        ):
            self.assertIn(flag, argv, f"{flag} missing from the on form")
        self.assertEqual(len([t for t in argv if t.startswith("--speculative")]), 5)
        self.assertEqual(
            argv[argv.index("--max-total-tokens") + 1], str(P_CAP)
        )
        # The values are D's, byte-for-byte (they hash into the drafter
        # identity, W5) -- a value drift here is a W10 refusal on metal.
        self.assertEqual(argv[argv.index("--speculative-algorithm") + 1], "NEXTN")
        self.assertEqual(argv[argv.index("--speculative-num-steps") + 1], "2")
        self.assertEqual(argv[argv.index("--speculative-eagle-topk") + 1], "1")
        self.assertEqual(argv[argv.index("--speculative-num-draft-tokens") + 1], "3")

    def test_default_is_on(self):
        """The standing user order of 2026-09-07: draft KV across the flip."""
        self.assertEqual(p(), p(draft_kv_on_p=True))

    def test_the_two_forms_differ_by_exactly_the_flag_block(self):
        on, off = p(draft_kv_on_p=True), p(draft_kv_on_p=False)
        # Not a set difference: the block is CONTIGUOUS and in one place, so a
        # future edit that scatters it across argv_p fails here.
        # #1305 item 2: the cap rides directly behind the head's flags, with a
        # value the CALLER hands in (the shipped cut's priced pool), so the
        # contiguous block is the five flags plus that pair.
        block = list(P_DRAFT_KV_FLAGS) + ["--max-total-tokens", str(P_CAP)]
        i = on.index("--speculative-algorithm")
        self.assertEqual(on[i:i + len(block)], block)
        self.assertEqual(on[:i] + on[i + len(block):], off)

    def test_group_d_is_untouched_by_the_switch(self):
        """D keeps its own NEXTN head in BOTH forms."""
        d = argv_d(py=PY, model=MODEL, budgets=BUDGETS, s_gb=48, m_mib=2400,
                   store_cfg=STORE_CFG, extra=[])
        for flag in ("--speculative-algorithm", "--speculative-num-steps",
                     "--speculative-eagle-topk", "--speculative-num-draft-tokens"):
            self.assertIn(flag, d)
        # ... and NOT the producer flag: D verifies, it does not produce.
        self.assertNotIn("--speculative-draft-kv-only", d)
        # argv_d takes no draft_kv_on_p parameter at all -- the switch cannot
        # reach it even by accident.
        import inspect

        self.assertNotIn("draft_kv_on_p", inspect.signature(argv_d).parameters)

    def test_the_off_line_is_the_ordered_sentence(self):
        line = draft_kv_off_line()
        self.assertEqual(
            line,
            "WEG2 DRAFT-KV-ON-P: off -- group P boots without the MTP head; "
            "D reads KV+Mamba from the store, draft state is cold after every "
            "flip (user order 2026-09-07 stands; this form is the rg6 baseline)",
        )


class TestFormKeyDiscriminates(unittest.TestCase):
    """Property 2 -- and the mutant's target.

    A form key that ignored the switch would let a ring table solved from an
    ``on`` boot be charged as a MEASUREMENT of an ``off`` one.  That is boot
    weg2tr2's death (a ring sized from a no-drafter predecessor for a boot with
    a drafter) with the sign flipped and the switch as its new vector.
    """

    def test_on_and_off_hash_to_different_forms(self):
        on_key, on_norm = ring_table.p_form_key(p(draft_kv_on_p=True))
        off_key, off_norm = ring_table.p_form_key(p(draft_kv_on_p=False))
        self.assertNotEqual(
            on_key, off_key,
            "the on and off forms of group P collide in the form key: a ring "
            "solved from one would be charged as MEASURED for the other",
        )
        self.assertNotEqual(on_norm, off_norm)

    def test_the_speculative_family_is_not_excluded_by_name(self):
        """Why property 2 needs no code: the policy is a blacklist."""
        self.assertEqual(ring_table.FORM_KEY_POLICY, "blacklist")
        for flag in ring_table.FORM_KEY_EXCLUDED_FLAGS:
            self.assertFalse(flag.startswith("--speculative"))
            self.assertNotEqual(flag, "--max-total-tokens")

    def test_the_difference_is_exactly_the_switched_flags(self):
        _, on_norm = ring_table.p_form_key(p(draft_kv_on_p=True))
        _, off_norm = ring_table.p_form_key(p(draft_kv_on_p=False))
        only_on = set(on_norm.split(" ")) - set(off_norm.split(" "))
        self.assertEqual(
            only_on,
            {
                "--speculative-algorithm=NEXTN",
                "--speculative-num-steps=2",
                "--speculative-eagle-topk=1",
                "--speculative-num-draft-tokens=3",
                "--speculative-draft-kv-only",
                f"--max-total-tokens={P_CAP}",
            },
        )
        self.assertEqual(set(off_norm.split(" ")) - set(on_norm.split(" ")), set())


class TestOnePredicate(unittest.TestCase):
    """Property 4: one reader, asked of the argv, not of the CLI value."""

    def test_predicate_reads_the_switch_off_the_argv(self):
        self.assertTrue(ring_table.p_carries_drafter(p(draft_kv_on_p=True)))
        self.assertFalse(ring_table.p_carries_drafter(p(draft_kv_on_p=False)))

    def test_predicate_is_true_for_group_d(self):
        """D carries the family WITHOUT the producer flag and is still a
        drafter -- which is why the predicate tests the family, not the
        silencer."""
        d = argv_d(py=PY, model=MODEL, budgets=BUDGETS, s_gb=48, m_mib=2400,
                   store_cfg=STORE_CFG, extra=[])
        self.assertTrue(ring_table.p_carries_drafter(d))

    def test_ring_module_has_no_second_inline_copy(self):
        """The predicate must be CALLED, not re-typed, in ring_table."""
        import inspect

        src = inspect.getsource(ring_table.stage_weights_from_argv)
        self.assertIn("p_carries_drafter(argv)", src)
        self.assertNotIn('any(str(t).startswith("--speculative-")', src)


@unittest.skipUnless(
    __import__("os").path.isdir(MODEL), f"checkpoint {MODEL} is not on this box"
)
class TestRingDrafterTerm(unittest.TestCase):
    """Property 3: ``off`` reproduces the rg6-form derivation, arithmetically."""

    def _stages(self, on: bool):
        stages, why = ring_table.stage_weights_from_argv(p(draft_kv_on_p=on))
        self.assertIsNotNone(stages, why)
        return stages

    def test_off_prices_no_drafter_on_any_stage(self):
        for st in self._stages(on=False):
            self.assertEqual(st.drafter_mib, 0.0, f"stage {st.stage}")
            self.assertEqual(st.drafter_terms, "")

    def test_on_prices_the_drafter_on_the_last_stage_only(self):
        stages = self._stages(on=True)
        for st in stages[:-1]:
            self.assertEqual(st.drafter_mib, 0.0, f"stage {st.stage}")
        self.assertGreater(stages[-1].drafter_mib, 0.0)

    def test_pp2_off_equals_the_rg6_measured_image_within_the_residual(self):
        """THE IDENTITY that makes "off == rg6 form" checkable.

        The residual this tree carries for PP2 was defined on the RING-CKPT
        line as boot weg2rg6's measured dormant image MINUS its own checkpoint
        weight statement -- and rg6's group P carried no MTP head.  So an
        ``off`` boot's PP2 requirement, weights + that residual, must land back
        on rg6's measured image.  If the drafter term leaked into the ``off``
        form this sum overshoots by ~1618 MiB and the assertion fails.
        """
        off_pp2 = self._stages(on=False)[-1].total_mib
        self.assertAlmostEqual(
            off_pp2 + RG6_PP2_RESIDUAL_MIB, RG6_PP2_MEASURED_IMAGE_MIB, delta=1.0,
            msg=(
                f"the off-form PP2 weight statement is {off_pp2:.1f} MiB; with "
                f"the carried residual {RG6_PP2_RESIDUAL_MIB} it must reproduce "
                f"boot weg2rg6's own measured image {RG6_PP2_MEASURED_IMAGE_MIB} "
                "MiB, because that residual IS that image minus a no-drafter "
                "weight statement"
            ),
        )

    def test_on_pp2_exceeds_off_pp2_by_exactly_the_drafter_term(self):
        on_last = self._stages(on=True)[-1]
        off_last = self._stages(on=False)[-1]
        self.assertAlmostEqual(
            on_last.total_mib - off_last.total_mib, on_last.drafter_mib, delta=0.05
        )
        # And the first two stages are identical under both forms -- the switch
        # moves ONE stage, which is the rg6-vs-tr2 shape ("two of three stages
        # matched the predecessor TO THE MEGABYTE").
        for a, b in zip(self._stages(on=True)[:-1], self._stages(on=False)[:-1]):
            self.assertAlmostEqual(a.total_mib, b.total_mib, delta=0.01)


class TestFrontArgvDoesNotFollowTheSwitch(unittest.TestCase):
    """Property 5: the front is not told twice.

    Its draft handling is observational (``Front._draft_terms`` reads D's own
    counters; the W9 census counts ``.draft`` files by name), so no term of
    ``front_argv_for`` depends on the producer.  Pinned so that a future draft
    flag on the front is a deliberate decision and not a silent divergence.
    """

    def test_front_argv_is_byte_identical_under_both_values(self):
        import argparse

        from sglang.srt.weg2.launcher import front_argv_for

        class C:
            uuid, nvml_index = "GPU-aaa", 0

        ns = argparse.Namespace(
            tag="t", fairness_w_s=1.0, drain_deadline_s=30.0,
            min_dwell_ms=None, d_admit_max_tokens=None,
        )
        kw = dict(
            py=PY, store_dir="/tmp/s", p_pid=1, d_pid=2, dc_expect_d={"GPU-aaa": 0},
            cards=[C()], ns=ns, chunk_count=8, carrier_max_tokens=27466, p_bs=8,
            d_bs=4, x_tokens=10649, flip_min_work_tokens=0, idle_layout_front="tp",
        )
        # front_argv_for takes no draft parameter at all; the two forms of the
        # launcher therefore hand it the same argv by construction.
        import inspect

        params = set(inspect.signature(front_argv_for).parameters)
        self.assertNotIn("draft_kv_on_p", params)
        self.assertEqual(front_argv_for(**kw), front_argv_for(**kw))
        self.assertEqual(
            [t for t in front_argv_for(**kw) if "draft" in t.lower()], []
        )


if __name__ == "__main__":
    unittest.main()
