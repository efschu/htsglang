"""Manual selection, task routing, and the precedence between them (#309).

The #156 controller becomes ONE source among several, so "who chose this arm"
has to be answerable. Every refusal is named -- a routing miss that quietly
runs the default arm is indistinguishable from routing that works.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.speculative.draft_selection import (
    ArmSet,
    Selection,
    SelectionError,
    SelectionSource,
    arms_from_server_args,
    parse_routing_table,
    parse_rung,
    resolve_selection,
    selection_is_noop,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ARMS = ArmSet(nextn_ks=(3, 5), dflash_block=16)
BOOT = ("nextn", 3)


class TestRungParsing(CustomTestCase):
    def test_well_formed_rungs(self):
        self.assertEqual(parse_rung("nextn:3"), ("nextn", 3))
        self.assertEqual(parse_rung(" DFLASH : 16 "), ("dflash", 16))

    def test_missing_colon_is_refused_with_an_example(self):
        with self.assertRaises(SelectionError) as cm:
            parse_rung("nextn3")
        self.assertIn("family:value", str(cm.exception))

    def test_unknown_family_is_refused(self):
        with self.assertRaises(SelectionError) as cm:
            parse_rung("eagle9:3")
        self.assertIn("unknown drafter family", str(cm.exception))

    def test_non_integer_value_is_refused(self):
        with self.assertRaises(SelectionError):
            parse_rung("nextn:auto")


class TestArmSetValidation(CustomTestCase):
    """A selection is checked against what the server LOADED, not against what
    is spellable."""

    def test_a_loaded_arm_validates(self):
        ARMS.validate(("nextn", 3), what="t")
        ARMS.validate(("dflash", 16), what="t")

    def test_an_unloaded_k_is_refused_and_lists_what_exists(self):
        with self.assertRaises(SelectionError) as cm:
            ARMS.validate(("nextn", 7), what="manual selection")
        msg = str(cm.exception)
        self.assertIn("not a loaded arm", msg)
        self.assertIn("nextn:3", msg)
        self.assertIn("nextn:5", msg)

    def test_dflash_with_no_dflash_arm_is_refused(self):
        with self.assertRaises(SelectionError) as cm:
            ArmSet(nextn_ks=(3,)).validate(("dflash", 16), what="t")
        self.assertIn("no DFLASH arm", str(cm.exception))

    def test_a_wrong_dflash_block_is_refused(self):
        with self.assertRaises(SelectionError) as cm:
            ARMS.validate(("dflash", 32), what="t")
        self.assertIn("block size 16", str(cm.exception))

    def test_describe_reads_as_a_list_of_arms(self):
        self.assertEqual(ARMS.describe(), "nextn:3, nextn:5, dflash:16")
        self.assertEqual(ArmSet().describe(), "(no arms loaded)")


class TestRoutingTable(CustomTestCase):
    def test_a_table_parses_and_validates(self):
        t = parse_routing_table("code=nextn:3,multiturn=nextn:5", arms=ARMS)
        self.assertEqual(t, {"code": ("nextn", 3), "multiturn": ("nextn", 5)})

    def test_empty_or_none_is_an_empty_table(self):
        for raw in (None, "", "   "):
            with self.subTest(raw=raw):
                self.assertEqual(parse_routing_table(raw, arms=ARMS), {})

    def test_the_canonical_156_entry(self):
        """DFLASH is the worst arm on multiturn -- a fact about a workload, so
        it lives in config, not in an if-statement."""
        t = parse_routing_table("prose=dflash:16,multiturn=nextn:5", arms=ARMS)
        self.assertEqual(t["multiturn"], ("nextn", 5))
        self.assertEqual(t["prose"], ("dflash", 16))

    def test_a_routing_entry_naming_an_unloaded_arm_is_refused_at_parse_time(self):
        """So a routing miss can never first appear as a request that quietly
        ran the wrong arm."""
        with self.assertRaises(SelectionError) as cm:
            parse_routing_table("code=nextn:7", arms=ARMS)
        self.assertIn("not a loaded arm", str(cm.exception))

    def test_a_malformed_entry_is_refused_with_an_example(self):
        with self.assertRaises(SelectionError) as cm:
            parse_routing_table("code nextn:3", arms=ARMS)
        self.assertIn("tag=family:value", str(cm.exception))

    def test_a_duplicate_tag_is_refused(self):
        with self.assertRaises(SelectionError) as cm:
            parse_routing_table("code=nextn:3,code=nextn:5", arms=ARMS)
        self.assertIn("twice", str(cm.exception))

    def test_an_empty_tag_is_refused(self):
        with self.assertRaises(SelectionError):
            parse_routing_table("=nextn:3", arms=ARMS)

    def test_tags_are_case_insensitive(self):
        t = parse_routing_table("CODE=nextn:3", arms=ARMS)
        self.assertIn("code", t)


class TestPrecedence(CustomTestCase):
    """THE order, in one place, because 'who won' is the first question."""

    def _resolve(self, **kw):
        base = dict(arms=ARMS, boot_rung=BOOT)
        base.update(kw)
        return resolve_selection(**base)

    def test_boot_when_nothing_else_is_set(self):
        s = self._resolve()
        self.assertEqual(s.rung, BOOT)
        self.assertIs(s.source, SelectionSource.BOOT)

    def test_controller_beats_boot(self):
        s = self._resolve(controller_rung=("nextn", 5))
        self.assertIs(s.source, SelectionSource.CONTROLLER)

    def test_routing_beats_the_controller(self):
        """The tag is a statement about THIS request's workload, which the
        controller cannot see."""
        s = self._resolve(
            request_tag="multiturn",
            routing={"multiturn": ("nextn", 5)},
            controller_rung=("dflash", 16),
        )
        self.assertIs(s.source, SelectionSource.ROUTED)
        self.assertEqual(s.rung, ("nextn", 5))

    def test_manual_beats_everything(self):
        """A pin a controller could override is not a pin."""
        s = self._resolve(
            manual_rung=("nextn", 3),
            request_tag="multiturn",
            routing={"multiturn": ("nextn", 5)},
            controller_rung=("dflash", 16),
        )
        self.assertIs(s.source, SelectionSource.MANUAL)
        self.assertEqual(s.rung, ("nextn", 3))

    def test_an_untagged_request_falls_through_to_the_controller(self):
        s = self._resolve(
            request_tag=None,
            routing={"code": ("nextn", 5)},
            controller_rung=("dflash", 16),
        )
        self.assertIs(s.source, SelectionSource.CONTROLLER)

    def test_an_unknown_tag_falls_through_by_default(self):
        """A deployment that tags some traffic and not the rest is normal."""
        s = self._resolve(
            request_tag="unmapped",
            routing={"code": ("nextn", 5)},
            controller_rung=("dflash", 16),
        )
        self.assertIs(s.source, SelectionSource.CONTROLLER)

    def test_strict_tags_turns_an_unknown_tag_into_a_named_error(self):
        with self.assertRaises(SelectionError) as cm:
            self._resolve(
                request_tag="unmapped",
                routing={"code": ("nextn", 5)},
                strict_tags=True,
            )
        msg = str(cm.exception)
        self.assertIn("unmapped", msg)
        self.assertIn("code", msg)


class TestNoSilentFallback(CustomTestCase):
    def test_a_manual_pin_to_an_unloaded_arm_is_refused(self):
        with self.assertRaises(SelectionError):
            resolve_selection(arms=ARMS, boot_rung=BOOT, manual_rung=("nextn", 7))

    def test_a_controller_choice_outside_the_arm_set_is_refused(self):
        """The controller is not privileged: it is validated like anyone else."""
        with self.assertRaises(SelectionError):
            resolve_selection(arms=ARMS, boot_rung=BOOT, controller_rung=("nextn", 9))

    def test_routing_is_revalidated_at_resolve_time(self):
        """An arm can go away under a runtime detach, so a table validated at
        boot would be routing to something no longer loaded."""
        table = parse_routing_table("code=nextn:5", arms=ARMS)
        shrunk = ArmSet(nextn_ks=(3,), dflash_block=None)
        with self.assertRaises(SelectionError) as cm:
            resolve_selection(
                arms=shrunk, boot_rung=BOOT, request_tag="code", routing=table
            )
        self.assertIn("not a loaded arm", str(cm.exception))


class TestArmsFromServerArgs(CustomTestCase):
    def test_boot_k_and_adaptive_candidates_are_collected(self):
        sa = SimpleNamespace(
            speculative_num_steps=3,
            speculative_adaptive_config="2,3,5",
            speculative_dflash_block_size=16,
        )
        arms = arms_from_server_args(sa)
        self.assertEqual(arms.nextn_ks, (2, 3, 5))
        self.assertEqual(arms.dflash_block, 16)

    def test_a_spec_less_server_has_no_arms(self):
        sa = SimpleNamespace(
            speculative_num_steps=None,
            speculative_adaptive_config=None,
            speculative_dflash_block_size=None,
        )
        arms = arms_from_server_args(sa)
        self.assertEqual(arms.nextn_ks, ())
        self.assertIsNone(arms.dflash_block)
        # and every selection against it is refused, by name
        with self.assertRaises(SelectionError) as cm:
            arms.validate(("nextn", 3), what="t")
        self.assertIn("(no arms loaded)", str(cm.exception))


class TestNoopDetection(CustomTestCase):
    def test_reselecting_the_active_rung_is_a_noop(self):
        cur = Selection(("nextn", 3), SelectionSource.MANUAL)
        self.assertTrue(
            selection_is_noop(cur, Selection(("nextn", 3), SelectionSource.ROUTED))
        )

    def test_a_different_rung_is_not_a_noop(self):
        cur = Selection(("nextn", 3), SelectionSource.MANUAL)
        self.assertFalse(
            selection_is_noop(cur, Selection(("nextn", 5), SelectionSource.MANUAL))
        )

    def test_no_current_selection_is_not_a_noop(self):
        self.assertFalse(selection_is_noop(None, Selection(BOOT, SelectionSource.BOOT)))


class TestSelectionJson(CustomTestCase):
    def test_json_names_the_source(self):
        j = Selection(("nextn", 5), SelectionSource.ROUTED, "tag 'multiturn'").to_json()
        self.assertEqual(j["family"], "nextn")
        self.assertEqual(j["value"], 5)
        self.assertEqual(j["source"], "routed")
        self.assertIn("multiturn", j["detail"])


if __name__ == "__main__":
    unittest.main()
