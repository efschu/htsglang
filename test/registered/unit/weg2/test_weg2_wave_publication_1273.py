# SPDX-License-Identifier: Apache-2.0
"""B4d: the PRICED wave partition reaches the ranks, and a mismatch is
group-uniform.

``xchg_residency.XchgCensus.waves``' own TODO asked for this in its own words:
*"S1/S6 must publish this list to the ranks and have them refuse a mismatch,
exactly as the front already refuses a pause order that is not its own weights
tags."*  Until now the two objects were unrelated -- the launcher PRICED the PP
form's per-card ``chunk_tag_cards`` partition (THREE waves on this rig, which is
what spec section 5's ARMED line expects) while every rank derived its own from
the EMPTY map and got ONE (``wave_map=uniform-assumed``, its own stated
deviation, because a rank holds only its own stage's layer count).

MEASURED at the desk (B4c): the uniform partition puts both groups' whole
images on the 5090 at once -- ``18608 + 17510 + 2x1668 = 39454 MiB`` against a
``32607 MiB`` board, W71 in both directions, free_at_peak **-6847 MiB**.  So
the two partitions are not a preference: one of them cannot boot, and pricing
one while executing the other is round-2 refuter F11's hazard one layer up.

THE THREE THINGS ASSERTED HERE

* the launcher PUBLISHES the census's partition by the same channel the inject
  mode already travels, and publishes NOTHING when there is no census;
* ``waves_for_plan`` is a DROP-IN for ``derive_waves`` -- same signature -- so
  the rank-side change is one line, and with nothing published it returns the
  derived partition BYTE-IDENTICALLY (acceptance (iii): a shadow boot that
  publishes no list is unaffected);
* a disagreement is GROUP-UNIFORM, not rank-local.  The carrier is the wake
  path's own wired fence (``_weg2_group_fence_impl``): the reason travels in
  the per-rank dict, any ``ok=False`` makes EVERY rank raise
  ``Weg2FlipRankDisagree`` (W29) with the same peer list, so the line names
  WHICH rank derived WHAT.  W29 stays the raise -- the fence's contract and
  tests carry it -- and the new code is the REASON TEXT only.

Hermetic: env monkeypatched, no NVML, no torch.distributed.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher, weight_exchange as wx
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

#: The rig's own three-wave partition, as B4c's census file carries it.
PRICED = (("weights_0", "weights_4", "weights_6"),
          ("weights_1", "weights_5", "weights_7"),
          ("weights_2", "weights_3", "weights"))
FAMILY = [f"weights_{k}" for k in range(8)] + ["weights"]
#: The PP form's per-card map, ordinal-keyed, from the realized split [39,13,12].
PP_MAP = {"weights_0": (0,), "weights_1": (0,), "weights_2": (0,), "weights_3": (0,),
          "weights_4": (0, 1), "weights_5": (1,), "weights_6": (1, 2), "weights_7": (2,)}


class _Env:
    """Set/clear the published-waves variable without leaking into other tests."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = os.environ.get(wx.WAVES_ENV)
        if self.value is None:
            os.environ.pop(wx.WAVES_ENV, None)
        else:
            os.environ[wx.WAVES_ENV] = self.value
        wx.reset_wave_disagreement()
        return self

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop(wx.WAVES_ENV, None)
        else:
            os.environ[wx.WAVES_ENV] = self.old
        wx.reset_wave_disagreement()


class TheLauncherPublishesThePricedPartition(CustomTestCase):
    def test_the_wire_form_round_trips(self):
        wire = wx.publish_waves(PRICED)
        with _Env(wire):
            self.assertEqual(wx.published_waves(), PRICED)

    def test_nothing_published_reads_as_None_not_as_an_empty_partition(self):
        """ABSENT must never mean "no waves" -- that would be a schedule."""
        with _Env(None):
            self.assertIsNone(wx.published_waves())

    def test_a_malformed_publication_refuses_rather_than_guessing(self):
        for bad in ("", "|", "||", "   "):
            with _Env(bad):
                self.assertIsNone(wx.published_waves(), bad)

    def test_the_env_publisher_carries_it_on_every_armed_arm(self):
        for arm in launcher.WEIGHT_SOURCE_ARMED:
            env = launcher.prepare_xchg_env(
                lambda *a, **k: None, "1789", arm, dry=True, waves=PRICED)
            self.assertEqual(env.get(wx.WAVES_ENV), wx.publish_waves(PRICED), arm)

    def test_no_census_publishes_no_waves_key_at_all(self):
        env = launcher.prepare_xchg_env(
            lambda *a, **k: None, "1789", "exchange", dry=True, waves=None)
        self.assertNotIn(wx.WAVES_ENV, env)

    def test_the_ring_arm_publishes_nothing_even_with_a_partition_in_hand(self):
        """Pin: `ring` stays byte-identical (step 7's own invariant)."""
        self.assertEqual(
            launcher.prepare_xchg_env(
                lambda *a, **k: None, "1789", "ring", dry=True, waves=PRICED), {})


class WavesForPlanIsADropInForDeriveWaves(CustomTestCase):
    def test_the_signature_matches_so_the_rank_change_is_one_line(self):
        import inspect

        self.assertEqual(
            list(inspect.signature(wx.waves_for_plan).parameters),
            list(inspect.signature(wx.derive_waves).parameters))

    def test_with_nothing_published_it_is_byte_identical_to_the_derivation(self):
        """Acceptance (iii): a boot that publishes no list is unaffected."""
        with _Env(None):
            for tag_cards in ({}, PP_MAP):
                self.assertEqual(wx.waves_for_plan(FAMILY, tag_cards, (0, 1, 2)),
                                 wx.derive_waves(FAMILY, tag_cards, (0, 1, 2)))
            self.assertIsNone(wx.wave_disagreement())

    def test_a_rank_takes_the_PUBLISHED_partition_and_prints_three_waves(self):
        """Acceptance (i): today the rank derives ONE wave from the empty map.

        This is the whole point of B4d: the rank is handed the map it cannot
        compute (it holds only its own stage's layer count) and ends up with
        the partition the ARM line priced.
        """
        with _Env(wx.publish_waves(PRICED)):
            self.assertEqual(len(wx.derive_waves(FAMILY, {}, (0, 1, 2))), 1)
            got = wx.waves_for_plan(FAMILY, {}, (0, 1, 2))
            self.assertEqual(len(got), 3)
            self.assertEqual(tuple(tuple(w) for w in got), PRICED)

    def test_agreement_is_not_a_disagreement(self):
        with _Env(wx.publish_waves(PRICED)):
            wx.waves_for_plan(FAMILY, PP_MAP, (0, 1, 2))
            self.assertIsNone(wx.wave_disagreement())

    def test_the_published_partition_must_still_be_the_family(self):
        """A publication naming a tag this rank does not have is a refusal, not
        a partition -- the same check ``derive_leg_plan`` makes on its own."""
        with _Env(wx.publish_waves((("weights_0",), ("weights_99",)))):
            with self.assertRaises(wx.Weg2XchgWavePartitionDisagree):
                wx.waves_for_plan(FAMILY, {}, (0, 1, 2))


class ADisagreementIsGroupUniformNotRankLocal(CustomTestCase):
    def test_a_different_published_list_records_the_reason_with_both_digests(self):
        """Acceptance (ii).  The rank does NOT raise here: it records, and the
        wake fence turns it into one W29 on every rank."""
        other = (tuple(FAMILY[:5]), tuple(FAMILY[5:]))
        with _Env(wx.publish_waves(other)):
            got = wx.waves_for_plan(FAMILY, PP_MAP, (0, 1, 2))
            self.assertEqual(tuple(tuple(w) for w in got), other)
            reason = wx.wave_disagreement()
            self.assertIsNotNone(reason)
            self.assertIn(wx.WCODE_WAVE_PARTITION, reason)
            self.assertIn(wx.waves_digest(other), reason)
            self.assertIn(wx.waves_digest(wx.derive_waves(FAMILY, PP_MAP, (0, 1, 2))),
                          reason)

    def test_the_reason_names_the_two_counts_so_the_line_is_readable(self):
        other = (tuple(FAMILY),)
        with _Env(wx.publish_waves(other)):
            wx.waves_for_plan(FAMILY, PP_MAP, (0, 1, 2))
            self.assertIn("published=1", wx.wave_disagreement())
            self.assertIn("derived=3", wx.wave_disagreement())

    def test_the_digest_is_order_sensitive_within_a_wave_but_stable(self):
        a = wx.waves_digest((("x", "y"),))
        self.assertEqual(a, wx.waves_digest((("x", "y"),)))
        self.assertNotEqual(a, wx.waves_digest((("y", "x"),)))
        self.assertNotEqual(a, wx.waves_digest((("x",), ("y",))))

    def test_the_code_is_minted_from_the_free_three_and_is_not_W19(self):
        self.assertIn(wx.WCODE_WAVE_PARTITION.split()[0], ("W5", "W39", "W90"))
        self.assertNotIn("W19", wx.WCODE_WAVE_PARTITION)

    def test_the_group_uniform_raise_is_still_W29(self):
        """The ruling: the fence's existing contract carries it, so B4d mints
        no second group-uniform error."""
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu.SchedulerWeightUpdaterManager._weg2_group_fence_impl)
        self.assertIn("W29 Weg2FlipRankDisagree", src)
        self.assertNotIn(wx.WCODE_WAVE_PARTITION.split()[0] + " ", src)

    def test_reset_clears_it_so_one_flip_cannot_poison_the_next(self):
        other = (tuple(FAMILY),)
        with _Env(wx.publish_waves(other)):
            wx.waves_for_plan(FAMILY, PP_MAP, (0, 1, 2))
            self.assertIsNotNone(wx.wave_disagreement())
            wx.reset_wave_disagreement()
            self.assertIsNone(wx.wave_disagreement())


if __name__ == "__main__":
    unittest.main()


class TheFenceCarriesItAndTheRaiseStaysW29(CustomTestCase):
    """The per-rank dict is the carrier (operator ruling): no new bus.

    Structural, via AST, so it cannot pass on a coincidental token elsewhere in
    the method -- the trap a mutant charged this campaign for one slice ago.
    """

    def _fence_dict_keys(self):
        import ast
        import inspect
        import textwrap

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = textwrap.dedent(inspect.getsource(
            wu.SchedulerWeightUpdaterManager._weg2_group_fence_impl))
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "mine" for t in node.targets):
                return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        self.fail("the fence no longer builds a per-rank dict called `mine`")

    def test_both_digests_ride_the_per_rank_dict(self):
        keys = self._fence_dict_keys()
        self.assertIn("waves_published", keys)
        self.assertIn("waves_planned", keys)
        # and the fields the fence's own contract already names are still there
        for k in ("rank", "ok", "failure", "card", "leg_ms", "per_tag"):
            self.assertIn(k, keys)

    def test_a_recorded_mismatch_makes_this_ranks_vote_not_ok(self):
        """The one line that turns a rank-local finding into a group stop."""
        import ast
        import inspect
        import textwrap

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = textwrap.dedent(inspect.getsource(
            wu.SchedulerWeightUpdaterManager._weg2_group_fence_impl))
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "mine" for t in node.targets):
                pairs = dict(zip([k.value for k in node.value.keys],
                                 [ast.unparse(v) for v in node.value.values]))
                self.assertEqual(pairs["ok"], "bool(ok) and (not wave_reason)")
                self.assertIn("wave_reason", pairs["failure"])
                return
        self.fail("no `mine` dict")

    def test_the_planned_partition_is_recorded_even_on_agreement(self):
        with _Env(wx.publish_waves(PRICED)):
            wx.waves_for_plan(FAMILY, PP_MAP, (0, 1, 2))
            self.assertEqual(wx.planned_waves(), PRICED)
            self.assertIsNone(wx.wave_disagreement())

    def test_with_nothing_published_the_planned_partition_is_the_derived_one(self):
        with _Env(None):
            got = wx.waves_for_plan(FAMILY, {}, (0, 1, 2))
            self.assertEqual(wx.planned_waves(), tuple(tuple(w) for w in got))

    def test_reset_clears_the_planned_partition_too(self):
        with _Env(wx.publish_waves(PRICED)):
            wx.waves_for_plan(FAMILY, PP_MAP, (0, 1, 2))
            wx.reset_wave_disagreement()
            self.assertIsNone(wx.planned_waves())
