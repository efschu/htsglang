"""CPU unit tests for the join-time compatibility gate."""

import unittest

from sglang.srt.rigmon.aggregator import (
    Aggregator,
    CompatibilityRefused,
)
from sglang.srt.rigmon.compat import (
    BLOCK,
    OK,
    WARN,
    NodeIdentity,
    check_compatibility,
    local_identity,
)
from sglang.srt.rigmon.config import AggregatorConfig
from sglang.srt.rigmon.series import TierSpec
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

TIERS = (TierSpec("live", 1.0, 60.0),)

MODEL = {"exists": True, "fingerprint": "abc123", "files": 3}
MODEL_OTHER = {"exists": True, "fingerprint": "zzz999", "files": 3}


def ident(node_id, **kw):
    base = dict(
        commit="a" * 40,
        branch="feat/x",
        dirty=False,
        torch_version="2.9.0",
        cuda_version="13.2",
        driver_version="595.58.03",
        gpu_archs=["sm86", "sm120"],
        built_archs=["sm86", "sm120"],
        models={"/models/q": dict(MODEL)},
    )
    base.update(kw)
    return NodeIdentity(node_id=node_id, **base)


def verdicts(report):
    return {c.key: c.verdict for c in report.checks}


class TestGate(CustomTestCase):
    def test_identical_nodes_pass(self):
        r = check_compatibility(ident("a"), ident("b"))
        self.assertFalse(r.blocked)
        self.assertEqual(verdicts(r)["commit"], OK)
        self.assertEqual(verdicts(r)["model:/models/q"], OK)

    def test_different_commit_blocks(self):
        r = check_compatibility(ident("a"), ident("b", commit="b" * 40))
        self.assertTrue(r.blocked)
        c = [x for x in r.checks if x.key == "commit"][0]
        self.assertEqual(c.verdict, BLOCK)
        self.assertIn("same commit", c.remedy)

    def test_dirty_tree_warns_without_blocking(self):
        r = check_compatibility(ident("a"), ident("b", dirty=True))
        self.assertFalse(r.blocked)
        self.assertEqual(verdicts(r)["dirty_tree"], WARN)

    def test_arch_coverage_gap_blocks_and_names_the_missing_arch(self):
        """The recurring failure: a build for one card reused on the other,
        surfacing as cudaErrorNoKernelImageForDevice deep into a run."""
        remote = ident("b", gpu_archs=["sm75"], built_archs=["sm86", "sm120"])
        r = check_compatibility(ident("a"), remote)
        self.assertTrue(r.blocked)
        blockers = [c for c in r.checks if c.verdict == BLOCK]
        self.assertTrue(any("sm75" in c.detail for c in blockers))
        self.assertTrue(any("TORCH_CUDA_ARCH_LIST" in (c.remedy or "") for c in blockers))
        self.assertTrue(any("cache key" in (c.remedy or "") for c in blockers))

    def test_arch_coverage_ok_when_both_builds_cover_the_union(self):
        remote = ident("b", gpu_archs=["sm75"], built_archs=["sm75", "sm86", "sm120"])
        local = ident("a", built_archs=["sm75", "sm86", "sm120"])
        r = check_compatibility(local, remote)
        self.assertEqual(verdicts(r)["arch_coverage"], OK)
        self.assertFalse(r.blocked)

    def test_torch_mismatch_blocks_but_driver_only_warns(self):
        r = check_compatibility(
            ident("a"), ident("b", torch_version="2.8.0", driver_version="580.1")
        )
        self.assertEqual(verdicts(r)["torch"], BLOCK)
        self.assertEqual(verdicts(r)["driver"], WARN)

    def test_differing_model_fingerprints_block(self):
        r = check_compatibility(
            ident("a"), ident("b", models={"/models/q": dict(MODEL_OTHER)})
        )
        self.assertTrue(r.blocked)
        c = [x for x in r.checks if x.key == "model:/models/q"][0]
        self.assertIn("quality seam", c.remedy)

    def test_missing_model_on_one_side_blocks(self):
        r = check_compatibility(
            ident("a"), ident("b", models={"/models/q": {"exists": False}})
        )
        self.assertTrue(r.blocked)

    def test_disjoint_model_sets_warn(self):
        r = check_compatibility(
            ident("a"), ident("b", models={"/models/other": dict(MODEL)})
        )
        self.assertEqual(verdicts(r)["model_sets"], WARN)

    def test_no_models_declared_warns(self):
        r = check_compatibility(ident("a", models={}), ident("b", models={}))
        self.assertEqual(verdicts(r)["model_sets"], WARN)

    def test_json_roundtrip(self):
        i = ident("a")
        again = NodeIdentity.from_json(i.to_json())
        self.assertEqual(again.commit, i.commit)
        self.assertEqual(again.built_archs, i.built_archs)

    def test_from_json_ignores_unknown_fields(self):
        d = ident("a").to_json()
        d["future_field"] = 1
        self.assertEqual(NodeIdentity.from_json(d).node_id, "a")

    def test_local_identity_is_readonly_and_does_not_raise(self):
        i = local_identity("self", model_paths=["/nonexistent/model"])
        self.assertEqual(i.node_id, "self")
        self.assertFalse(i.models["/nonexistent/model"]["exists"])


class TestGateInJoin(CustomTestCase):
    def _agg(self, local):
        agg = Aggregator(AggregatorConfig(tiers=TIERS))
        agg.local_identity = local
        return agg

    def test_incompatible_node_is_refused_with_reasons(self):
        agg = self._agg(ident("a"))
        tok = agg.mint_join_token()["join_token"]
        with self.assertRaises(CompatibilityRefused) as cm:
            agg.redeem_join_token(tok, "b", identity=ident("b", commit="b" * 40).to_json())
        self.assertIn("fork commit", str(cm.exception))
        self.assertTrue(cm.exception.report.blocked)

    def test_a_refused_join_does_not_burn_the_token(self):
        """The user fixes the mismatch and retries; making them mint a new
        token for a check that is meant to be diagnostic would be hostile."""
        agg = self._agg(ident("a"))
        tok = agg.mint_join_token()["join_token"]
        with self.assertRaises(CompatibilityRefused):
            agg.redeem_join_token(tok, "b", identity=ident("b", commit="b" * 40).to_json())
        out = agg.redeem_join_token(tok, "b", identity=ident("b").to_json())
        self.assertTrue(out["push_token"])

    def test_force_overrides_the_gate_and_records_that_it_did(self):
        agg = self._agg(ident("a"))
        tok = agg.mint_join_token()["join_token"]
        out = agg.redeem_join_token(
            tok, "b", identity=ident("b", commit="b" * 40).to_json(), force=True
        )
        self.assertTrue(out["push_token"])
        self.assertTrue(out["forced"])
        self.assertTrue(out["compatibility"]["blocked"])

    def test_compatible_join_carries_the_report(self):
        agg = self._agg(ident("a"))
        tok = agg.mint_join_token()["join_token"]
        out = agg.redeem_join_token(tok, "b", identity=ident("b").to_json())
        self.assertFalse(out["compatibility"]["blocked"])
        self.assertFalse(out["forced"])

    def test_report_is_visible_on_the_node_listing(self):
        agg = self._agg(ident("a"))
        tok = agg.mint_join_token()["join_token"]
        agg.redeem_join_token(tok, "b", identity=ident("b").to_json())
        agg.ingest({"node_id": "b", "points": {}})
        node = [n for n in agg.nodes() if n["node_id"] == "b"][0]
        self.assertIsNotNone(node["compatibility"])
        self.assertFalse(node["compatibility"]["blocked"])
        self.assertEqual(agg.compatibility("b")["remote"], "b")

    def test_join_without_identity_still_works(self):
        """A node that predates the gate must not be locked out; it simply has
        no report."""
        agg = self._agg(ident("a"))
        tok = agg.mint_join_token()["join_token"]
        out = agg.redeem_join_token(tok, "b")
        self.assertTrue(out["push_token"])
        self.assertNotIn("compatibility", out)

    def test_expired_token_is_still_refused_before_the_gate(self):
        clock = [1000.0]
        agg = Aggregator(
            AggregatorConfig(tiers=TIERS, join_token_ttl_s=10.0), clock=lambda: clock[0]
        )
        agg.local_identity = ident("a")
        tok = agg.mint_join_token()["join_token"]
        clock[0] += 20.0
        with self.assertRaises(PermissionError):
            agg.redeem_join_token(tok, "b", identity=ident("b").to_json())


if __name__ == "__main__":
    unittest.main()
