# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for rig_artifact.py -- the #271 shared artifact and share path.

Hermetic: no network (the GitHub REST API is a fake), no GPU, no measurement.
Covers the four properties the feature stands on:

  * the SCHEMA is one schema for both sources, and a source contributes rows
    rather than a format;
  * the ANONYMIZATION gate refuses an artifact carrying an IP, a path, a
    hostname, a GPU UUID, a username or a rig-env value -- and the gate runs
    inside build_digest, so a preview cannot skip it;
  * the CURATION is automatic: dedupe, error folding, delta on re-share, and
    a size ceiling met by aggregating harder rather than truncating;
  * the SHARE path cannot post without a preview and an explicit confirm, is
    routed per rig FINGERPRINT (index in the body, one comment per rig), and
    never leaks the token.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.planner import github_share, rig_artifact
from sglang.srt.planner.rig_artifact import (
    ARTIFACT_SCHEMA,
    AGGREGATION_LADDER,
    Capability,
    ErrorSignature,
    Measurement,
    SourceSections,
    assert_anonymized,
    build_digest,
    build_report,
    build_index_body,
    comment_marker,
    compound_fingerprint,
    curate,
    digest_from_comment,
    error_signature,
    merge_digests,
    parse_index_body,
    rig_fingerprint,
    scrub_tree,
)

TOKEN = "ghp_SUPERSECRETPATVALUE1234567890"

RIG = {
    "cards": [
        {"index": 0, "name": "NVIDIA GeForce RTX 3080", "vram_mib": 20480},
        {"index": 1, "name": "NVIDIA GeForce RTX 5090", "vram_mib": 32607},
        {"index": 2, "name": "NVIDIA GeForce RTX 3080", "vram_mib": 20470},
    ],
    "card_summary": "2x RTX 3080 + 1x RTX 5090",
    "driver": "595.58.03",
    "cuda": "13.0",
    "torch": "2.11.0+cu130",
    "nccl": "2.28.9",
}


def _m(mid, value=1.0, **kw):
    kw.setdefault("label", mid)
    kw.setdefault("source", "comm_suite")
    kw.setdefault("unit", "us")
    kw.setdefault("taken_at", "2026-07-28")
    return Measurement(id=mid, value=value, **kw)


def _sections(measurements=(), **kw):
    return SourceSections(
        source=kw.pop("source", "comm_suite"),
        rig=kw.pop("rig", dict(RIG)),
        measurements=list(measurements),
        capabilities=list(kw.pop("capabilities", [])),
        errors=list(kw.pop("errors", [])),
        notes=list(kw.pop("notes", [])),
    )


class SchemaTest(unittest.TestCase):
    def test_one_schema_serves_several_sources(self):
        a = _sections([_m("comm/gloo/all_reduce/20KiB", 103.7)],
                      source="comm_suite")
        b = _sections([_m("card/RTX 5090/membw", 1660.4,
                          source="hardware_profile")],
                      source="hardware_profile")
        d = build_digest([a, b])
        self.assertEqual(d["schema"], ARTIFACT_SCHEMA)
        self.assertEqual(d["sources"], ["comm_suite", "hardware_profile"])
        ids = {r["id"] for r in d["measurements"]}
        self.assertIn("comm/gloo/all_reduce/20KiB", ids)
        self.assertIn("card/RTX 5090/membw", ids)

    def test_row_carries_aggregate_spread_date_and_context(self):
        d = build_digest([_sections([
            _m("comm/gloo/all_reduce/20KiB", 103.7, spread_pct=61.5, n=60,
               p5=90.1, p95=154.0,
               context={"op": "all_reduce", "size_kib": 20, "world": 2})])])
        row = d["measurements"][0]
        for key in ("value", "spread_pct", "n", "taken_at", "context",
                    "status", "unit"):
            self.assertIn(key, row, key)
        self.assertEqual(row["context"]["size_kib"], 20)

    def test_no_raw_logs_survive(self):
        # No source emits a log; the scrub strips log-shaped keys anyway,
        # because "a future source forgets" is the failure mode.
        d = build_digest([_sections(
            [_m("x/y", 1.0, context={"stdout": "boom\ntrace", "op": "ar"})],
            notes=["fine"])])
        self.assertNotIn("stdout", json.dumps(d))
        self.assertIn("op", d["measurements"][0]["context"])


class AnonymizationTest(unittest.TestCase):
    def test_gate_rejects_ip_uuid_path_and_host(self):
        for bad in (
            {"a": "peer at 192.168.0.89"},
            {"a": "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"},
            {"a": "/spinning/llm_stuff/models-cache/Qwen3.6"},
        ):
            with self.assertRaises(ValueError):
                assert_anonymized(bad)

    def test_gate_accepts_models_versions_and_numbers(self):
        assert_anonymized({
            "cards": ["2x RTX 3080", "1x RTX 5090"],
            "driver": "595.58.03", "nccl": "2.28.9",
            "value": 103.7, "unit": "us",
        })

    def test_scrub_removes_identity_from_a_hostile_payload(self):
        hostile = {
            "note": "ran on 10.10.10.2 from /spinning/wt-commsuite/python",
            "uuid": "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d",
            "host": "some-box.local",
            "cards": ["RTX 5090"],
        }
        out = scrub_tree(hostile, literals=[])
        self.assertNotIn("uuid", out)
        self.assertNotIn("host", out)
        blob = json.dumps(out)
        self.assertNotIn("10.10.10.2", blob)
        self.assertNotIn("/spinning", blob)
        self.assertIn("RTX 5090", blob)

    def test_rig_env_values_are_scrubbed_by_literal_match(self):
        # The rig-env rule: a value that came out of the rig env file leaves
        # by literal match even when it reads as ordinary text.
        with mock.patch.dict(os.environ, {"RIG2_HOST": "workbench-two",
                                          "RIG1_REPO_ROOT": "/opt/checkout"},
                             clear=False):
            d = build_digest([_sections(
                [_m("x/y", 1.0)],
                notes=["measured against workbench-two in /opt/checkout"])])
            blob = json.dumps(d)
            self.assertNotIn("workbench-two", blob)
            self.assertNotIn("/opt/checkout", blob)

    def test_build_digest_runs_the_gate(self):
        # The gate is INSIDE build_digest: a preview cannot be produced from
        # an artifact that would leak.
        with mock.patch.object(rig_artifact, "scrub_tree",
                               side_effect=lambda o, *a, **k: o):
            with self.assertRaises(ValueError):
                build_digest([_sections(
                    [_m("x/y", 1.0, note="peer 192.168.0.89")])])


class FingerprintTest(unittest.TestCase):
    def test_stable_and_independent_of_card_order(self):
        a = rig_fingerprint(RIG)
        shuffled = dict(RIG, cards=list(reversed(RIG["cards"])))
        self.assertEqual(a["id"], rig_fingerprint(shuffled)["id"])

    def test_vram_jitter_does_not_split_a_profile(self):
        # 20480 vs 20470 MiB is the same card model; a fingerprint that split
        # on it would report one rig as two.
        jittered = dict(RIG, cards=[dict(c, vram_mib=(c["vram_mib"] or 0) - 7)
                                    for c in RIG["cards"]])
        self.assertEqual(rig_fingerprint(RIG)["id"],
                         rig_fingerprint(jittered)["id"])

    def test_different_hardware_gets_a_different_id(self):
        other = dict(RIG, cards=[{"name": "NVIDIA GeForce RTX 4090",
                                  "vram_mib": 24564}])
        self.assertNotEqual(rig_fingerprint(RIG)["id"],
                            rig_fingerprint(other)["id"])

    def test_signature_carries_no_identity(self):
        sig = rig_fingerprint(RIG)["signature"]
        assert_anonymized({"sig": sig})
        for key in ("uuid", "serial", "host", "hostname", "mac"):
            self.assertNotIn(key, sig)

    def test_id_still_matches_its_signature_after_the_scrub(self):
        # If the scrub ever edited a signature string, the published id would
        # no longer hash to the published signature.
        d = build_digest([_sections([_m("x/y", 1.0)])])
        fp = d["fingerprint"]
        self.assertEqual(rig_artifact._stable_hash(fp["signature"], "rig-"),
                         fp["id"])

    def test_compound_fingerprint_is_its_own_thing(self):
        a, b = rig_fingerprint(RIG)["id"], "rig-deadbeef0000"
        comp = compound_fingerprint([a, b], "RoCE 40G")
        self.assertTrue(comp["id"].startswith("link-"))
        self.assertEqual(comp["kind"], "compound")
        self.assertNotEqual(comp["id"], a)
        # Member order must not matter: the pair is the identity.
        self.assertEqual(comp["id"],
                         compound_fingerprint([b, a], "RoCE 40G")["id"])
        self.assertNotEqual(comp["id"],
                            compound_fingerprint([a, b], "1 GbE")["id"])


class CurationTest(unittest.TestCase):
    def test_duplicates_are_dropped_newest_wins(self):
        d = curate([_sections([
            _m("x/y", 1.0, taken_at="2026-07-01"),
            _m("x/y", 2.0, taken_at="2026-07-28"),
        ])])
        self.assertEqual(len(d["measurements"]), 1)
        self.assertEqual(d["measurements"][0]["value"], 2.0)
        self.assertEqual(d["curation"]["duplicates_dropped"], 1)

    def test_errors_fold_into_signatures_with_counts(self):
        errs = [error_signature("CUDA out of memory: tried to allocate 512 MiB",
                                where="arm/a"),
                error_signature("CUDA out of memory: tried to allocate 977 MiB",
                                where="arm/b")]
        d = curate([_sections(errors=errs)])
        self.assertEqual(len(d["errors"]), 1)
        self.assertEqual(d["errors"][0]["count"], 2)
        self.assertIn("<n>", d["errors"][0]["signature"])

    def test_delta_share_drops_unchanged_rows_only(self):
        first = curate([_sections([_m("a", 1.0), _m("b", 2.0)])])
        previous = {"generated_at": first["generated_at"],
                    "measurements": first["measurements"]}
        second = curate([_sections([_m("a", 1.0), _m("b", 99.0)])],
                        previous=previous)
        ids = {r["id"] for r in second["measurements"]}
        self.assertEqual(ids, {"b"})
        self.assertEqual(second["curation"]["carried_over_unchanged"], 1)

    def test_a_value_inside_its_own_tolerance_is_not_a_change(self):
        first = curate([_sections([_m("a", 100.0)])])
        previous = {"generated_at": "x", "measurements": first["measurements"]}
        again = curate([_sections([_m("a", 100.4)])], previous=previous)
        self.assertEqual(again["measurements"], [])

    def test_ceiling_aggregates_harder_and_never_truncates(self):
        many = [_m(f"comm/gloo/cell{i}", float(i), spread_pct=1.0, n=60,
                   p5=float(i), p95=float(i),
                   context={"op": "all_reduce", "size_kib": i, "junk": "x" * 40},
                   note="a note that costs bytes " * 3)
                for i in range(400)]
        d = curate([_sections(many)], max_bytes=20_000)
        self.assertLessEqual(len(json.dumps(d)), 20_000)
        self.assertNotEqual(d["curation"]["aggregation_level"],
                            AGGREGATION_LADDER[0])
        # Aggregated, not cut: every input row is still accounted for.
        if d["curation"]["aggregation_level"] == "group_measurements":
            self.assertEqual(
                sum(r["aggregated_rows"] for r in d["measurements"]), 400)
        else:
            self.assertEqual(len(d["measurements"]), 400)

    def test_small_digest_stays_at_full_detail(self):
        d = curate([_sections([_m("a", 1.0, p5=0.9, p95=1.1)])])
        self.assertEqual(d["curation"]["aggregation_level"], "full")
        self.assertIn("p5", d["measurements"][0])


class MergeTest(unittest.TestCase):
    def test_second_machine_of_a_profile_becomes_a_sample(self):
        a = build_digest([_sections([_m("x/y", 100.0)])])
        b = build_digest([_sections([_m("x/y", 130.0)])])
        b["machines"] = ["m-second"]
        b["measurements"][0]["machines"] = ["m-second"]
        merged = merge_digests(a, b)
        self.assertEqual(merged["sample_count"], 2)
        row = merged["measurements"][0]
        self.assertEqual(row["samples"], 2)
        self.assertEqual(row["min"], 100.0)
        self.assertEqual(row["max"], 130.0)
        self.assertEqual(row["across_machines_pct"], 23.1)

    def test_rows_the_delta_omitted_are_kept_in_the_merge(self):
        published = build_digest([_sections([_m("a", 1.0), _m("b", 2.0)])])
        delta_only = build_digest([_sections([_m("b", 3.0)])])
        merged = merge_digests(published, delta_only)
        ids = {r["id"] for r in merged["measurements"]}
        self.assertEqual(ids, {"a", "b"})
        carried = [r for r in merged["measurements"] if r["id"] == "a"][0]
        self.assertTrue(carried["carried_over"])

    def test_digest_round_trips_through_a_published_comment(self):
        d = build_digest([_sections([_m("x/y", 1.0)])])
        body = build_report(d)
        back = digest_from_comment(body)
        self.assertIsNotNone(back)
        self.assertEqual(back["fingerprint"]["id"], d["fingerprint"]["id"])
        self.assertEqual(len(back["measurements"]), len(d["measurements"]))


class IndexTest(unittest.TestCase):
    def test_index_body_round_trips(self):
        entries = [
            {"id": "rig-aaa", "label": "2x RTX 3080", "sample_count": 3,
             "sources": ["comm_suite"], "updated": "2026-07-28T09:00:00Z"},
            {"id": "link-bbb", "label": "2 rigs over RoCE", "sample_count": 1,
             "sources": ["comm_suite"], "updated": "2026-07-28T09:00:00Z"},
        ]
        parsed = parse_index_body(build_index_body(entries))
        self.assertEqual({e["id"] for e in parsed}, {"rig-aaa", "link-bbb"})
        self.assertEqual(
            [e["sample_count"] for e in parsed if e["id"] == "rig-aaa"], [3])

    def test_comment_marker_separates_fingerprints_and_labels(self):
        self.assertNotEqual(comment_marker("rig-a"), comment_marker("rig-b"))
        self.assertNotEqual(comment_marker("rig-a"),
                            comment_marker("rig-a", "bench2"))
        # The suffix is sanitized: a marker is matched as a literal string.
        self.assertEqual(comment_marker("rig-a", "a b/c<!->"),
                         comment_marker("rig-a", "abc-"))


class FakeGitHub:
    """A GitHub that records every call and never touches a socket."""

    def __init__(self, issues=None, comments=None):
        self.calls = []
        self.issues = issues if issues is not None else []
        self.comments = comments if comments is not None else []

    def __call__(self, method, url, token, body=None, timeout=30.0):
        self.calls.append((method, url, body))
        if url.endswith("/user"):
            return 200, {"login": "someone"}
        if "/issues/comments/" in url and method == "PATCH":
            return 200, {"id": 7, "html_url": "https://x/c7"}
        if url.endswith("/comments") and method == "GET":
            return 200, self.comments
        if url.endswith("/comments") and method == "POST":
            return 201, {"id": 8, "html_url": "https://x/c8"}
        if method == "GET":
            return 200, self.issues
        if method == "PATCH":
            return 200, {"number": 5, "html_url": "https://x/5"}
        return 201, {"number": 5, "html_url": "https://x/5"}


class SharePathTest(unittest.TestCase):
    def setUp(self):
        self.digest = build_digest([_sections([_m("x/y", 1.0)])])
        self.report = build_report(self.digest)

    def test_preview_is_pure(self):
        # build_report must be usable without a token and without a network:
        # that is what makes "look before you post" possible at all.
        self.assertIn(self.digest["fingerprint"]["id"], self.report)
        self.assertIn(comment_marker(self.digest["fingerprint"]["id"]),
                      self.report)

    def test_no_post_without_confirmation_and_no_call_is_made(self):
        api = FakeGitHub()
        with self.assertRaises(github_share.GitHubShareError):
            rig_artifact.submit(self.report, TOKEN, self.digest, api=api)
        self.assertEqual(api.calls, [],
                         "an unconfirmed submit must make NO network call")

    def test_confirmed_submit_writes_index_body_and_a_comment(self):
        api = FakeGitHub()
        out = rig_artifact.submit(self.report, TOKEN, self.digest,
                                  confirmed=True, api=api)
        methods = [c[0] for c in api.calls]
        self.assertIn("POST", methods)
        bodies = [c[2] for c in api.calls if c[2]]
        issue_body = next(b["body"] for b in bodies if "body" in b
                          and rig_artifact.SHARE_MARKER in b["body"])
        self.assertIn(self.digest["fingerprint"]["id"], issue_body)
        comment_bodies = [b["body"] for b in bodies
                          if comment_marker(self.digest["fingerprint"]["id"])
                          in b["body"]]
        self.assertEqual(len(comment_bodies), 1)
        self.assertEqual(comment_bodies[0], self.report,
                         "the posted comment must be the previewed text")
        self.assertEqual(out["fingerprint"], self.digest["fingerprint"]["id"])

    def test_sharing_rig_b_does_not_rewrite_rig_a(self):
        existing_body = build_index_body([
            {"id": "rig-other", "label": "1x RTX 4090", "sample_count": 2,
             "sources": ["comm_suite"], "updated": "2026-07-01T00:00:00Z"}])
        api = FakeGitHub(issues=[{"number": 5, "body": existing_body}])
        rig_artifact.submit(self.report, TOKEN, self.digest, confirmed=True,
                            api=api)
        new_body = next(c[2]["body"] for c in api.calls
                        if c[2] and "body" in c[2]
                        and rig_artifact.SHARE_MARKER in c[2]["body"])
        self.assertIn("rig-other", new_body)
        self.assertIn(self.digest["fingerprint"]["id"], new_body)

    def test_token_never_appears_in_the_result_or_an_error(self):
        api = FakeGitHub()
        out = rig_artifact.submit(self.report, TOKEN, self.digest,
                                  confirmed=True, api=api)
        self.assertNotIn(TOKEN, json.dumps(out))
        try:
            rig_artifact.submit(self.report, "", self.digest, confirmed=True,
                                api=api)
        except github_share.GitHubShareError as e:
            self.assertNotIn(TOKEN, str(e))

    def test_results_marker_and_rig_marker_are_different_issues(self):
        self.assertNotEqual(github_share.MARKER, rig_artifact.SHARE_MARKER)


class TokenStoreTest(unittest.TestCase):
    def test_store_is_opt_in_0600_and_never_read_back_by_the_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp},
                                 clear=False):
                self.assertFalse(rig_artifact.have_token())
                path = rig_artifact.save_token(TOKEN)
                self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
                self.assertTrue(rig_artifact.have_token())
                self.assertEqual(rig_artifact.load_token(), TOKEN)
                self.assertTrue(rig_artifact.forget_token())
                self.assertFalse(rig_artifact.have_token())

    def test_machine_tag_is_stable_and_carries_no_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp},
                                 clear=False):
                a = rig_artifact.machine_tag()
                self.assertEqual(a, rig_artifact.machine_tag())
                assert_anonymized({"tag": a})


class ReportTest(unittest.TestCase):
    def test_report_names_absent_and_failed_without_inventing_values(self):
        d = build_digest([_sections(
            [_m("x/ok", 1.0, status="ok")],
            capabilities=[Capability("comm/nccl", None, "absent",
                                     "cards were held by another job")],
            errors=[ErrorSignature("gate: <n> mismatches", 1, "byte_gate")])])
        r = build_report(d)
        self.assertIn("absent", r)
        self.assertIn("Error signatures", r)
        self.assertIn("cards were held by another job", r)

    def test_multi_machine_report_says_how_many(self):
        a = build_digest([_sections([_m("x/y", 100.0)])])
        b = build_digest([_sections([_m("x/y", 130.0)])])
        b["machines"] = ["m-second"]
        b["measurements"][0]["machines"] = ["m-second"]
        r = build_report(merge_digests(a, b))
        self.assertIn("2 machines of this profile", r)


if __name__ == "__main__":
    unittest.main()
