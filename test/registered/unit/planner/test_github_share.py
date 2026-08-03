# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for github_share.py -- the #152 GitHub results-sharing core.

Hermetic: the GitHub REST API is a fake; no network. Covers:

  * the report contains the start command (argv + env, SCRUBBED -- see
    test_github_share_scrub_505d3.py for the anonymity falsifiers) and the
    full metrics + quality shot (SVG, verdict, token counts) + the marker,
  * credential-looking env values are redacted from the report,
  * the PAT NEVER appears in results or exception text (redaction),
  * create vs update-in-place: marker+creator lookup chooses PATCH for an
    existing issue and POST otherwise; explicit issue number skips lookup,
  * NO submission without the confirmed flag (and zero API calls made).
"""

import unittest

from sglang.srt.planner.github_share import (
    API_ROOT,
    DEFAULT_REPO,
    GitHubShareError,
    MARKER,
    build_report,
    find_existing_issue,
    redact,
    submit,
)

TOKEN = "ghp_SUPERSECRETPATVALUE1234567890"

SVG = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'

PAYLOAD = {
    "model": "Qwen3.6-27B",
    "hardware": "1x RTX 5090 + 2x RTX 3080",
    "command": {
        "argv": [
            "python", "-m", "sglang.launch_server",
            "--model-path", "/models/Qwen3.6-27B-FP8",
            "--tp", "3", "--rank-gpu-id", "0,1,2",
        ],
        "env": {
            "SGLANG_UNEVEN_DCP": "1",
            "SGLANG_UNEVEN_TOKEN_VECTOR": "33,13,18",
            "HF_TOKEN": "hf_secretsecretsecret",
        },
    },
    "metrics": {
        "decode_tok_s": 42.5,
        "prefill_tok_s": 1810.0,
        "ttft_ms": 950,
        "j_per_decode_token": 1.9,
        "per_card": {
            "RTX 5090": {"power_w": 420, "tok_s": 21.0},
            "RTX 3080 (a)": {"power_w": 260, "tok_s": 11.0},
        },
    },
    "quality": {
        "svg": SVG,
        "verdict": "correct",
        "tokens": {"prompt": 220, "completion": 3900, "total": 4120},
        "report": "32/32 pieces on the correct squares",
    },
}


class FakeApi:
    """Records calls; scripted responses keyed by (method, path suffix)."""

    def __init__(self, issues=None, login="alice", fail_with=None):
        self.calls = []
        self.issues = issues if issues is not None else []
        self.login = login
        self.fail_with = fail_with

    def __call__(self, method, url, token, body=None, timeout=30.0):
        self.calls.append({"method": method, "url": url, "body": body})
        if self.fail_with is not None:
            raise self.fail_with
        if method == "GET" and url.endswith("/user"):
            return 200, {"login": self.login}
        if method == "GET" and "/issues?" in url:
            return 200, self.issues
        if method == "POST" and url.endswith("/issues"):
            return 201, {"number": 7,
                         "html_url": f"https://github.com/{DEFAULT_REPO}/issues/7"}
        if method == "PATCH" and "/issues/" in url:
            n = int(url.rsplit("/", 1)[1])
            return 200, {"number": n,
                         "html_url": f"https://github.com/{DEFAULT_REPO}/issues/{n}"}
        return 404, {}


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------
class TestBuildReport(unittest.TestCase):
    def test_command_is_exact_except_for_the_anonymity_scrub(self):
        md = build_report(PAYLOAD)
        # argv on one line, every flag and value kept -- but the model path is
        # a basename, not the local filesystem path (#505-D3). Sharing a
        # reproducible result needs the FLAGS, never the directory layout.
        self.assertIn(
            "python -m sglang.launch_server --model-path "
            "Qwen3.6-27B-FP8 --tp 3 --rank-gpu-id 0,1,2",
            md,
        )
        self.assertNotIn("/models/Qwen3.6-27B-FP8", md)
        self.assertIn("SGLANG_UNEVEN_DCP=1", md)
        self.assertIn("SGLANG_UNEVEN_TOKEN_VECTOR=33,13,18", md)

    def test_credential_env_values_redacted(self):
        md = build_report(PAYLOAD)
        self.assertNotIn("hf_secretsecretsecret", md)
        self.assertIn("HF_TOKEN=<redacted>", md)

    def test_metrics_present(self):
        md = build_report(PAYLOAD)
        self.assertIn("Decode: 42.5 tok/s", md)
        self.assertIn("Prefill: 1810 tok/s", md)
        self.assertIn("TTFT: 950 ms", md)
        self.assertIn("Energy decode: 1.9 J/token", md)
        self.assertIn("RTX 5090", md)
        self.assertIn("420", md)  # per-card power in the table

    def test_quality_shot(self):
        md = build_report(PAYLOAD)
        self.assertIn(SVG, md)
        self.assertIn("Verdict: **correct**", md)
        self.assertIn("prompt: 220", md)
        self.assertIn("completion: 3900", md)
        self.assertIn("32/32 pieces", md)

    def test_quality_optional(self):
        p = dict(PAYLOAD)
        p.pop("quality")
        md = build_report(p)
        self.assertNotIn("Quality shot", md)
        self.assertNotIn("<svg", md)

    def test_marker_always_present(self):
        self.assertIn(MARKER, build_report(PAYLOAD))
        self.assertIn(MARKER, build_report({}))

    def test_bench_results_table(self):
        p = dict(PAYLOAD)
        p["bench_results"] = [
            {"test_id": 1, "label": "Basic completion (Paris)",
             "status": "pass",
             "metric": {"name": "contains", "value": "Paris",
                        "numeric": None, "unit": None}},
            {"test_id": 7, "label": "MTP acceptance length",
             "status": "pass",
             "metric": {"name": "acceptance_length", "value": 2.85,
                        "numeric": 2.85, "unit": "tokens"}},
        ]
        md = build_report(p)
        self.assertIn("| 7 | MTP acceptance length | pass "
                      "| acceptance_length=2.85 tokens |", md)


# ---------------------------------------------------------------------------
# Token redaction.
# ---------------------------------------------------------------------------
class TestRedaction(unittest.TestCase):
    def test_redact(self):
        self.assertNotIn(TOKEN, redact(f"error with {TOKEN} inside", TOKEN))
        self.assertEqual(redact("clean", TOKEN), "clean")
        self.assertEqual(redact("", TOKEN), "")
        self.assertEqual(redact("x", None), "x")

    def test_token_never_in_exception_text(self):
        api = FakeApi(fail_with=RuntimeError(
            f"connection reset while sending Bearer {TOKEN}"))
        with self.assertRaises(GitHubShareError) as cm:
            submit(build_report(PAYLOAD), TOKEN, confirmed=True,
                   existing_issue=3, api=api)
        self.assertNotIn(TOKEN, str(cm.exception))
        self.assertIn("<redacted-token>", str(cm.exception))

    def test_token_never_in_result(self):
        api = FakeApi()
        out = submit(build_report(PAYLOAD), TOKEN, confirmed=True, api=api)
        self.assertNotIn(TOKEN, str(out))


# ---------------------------------------------------------------------------
# Consent gate.
# ---------------------------------------------------------------------------
class TestConsent(unittest.TestCase):
    def test_no_submit_without_confirmed(self):
        api = FakeApi()
        with self.assertRaises(GitHubShareError) as cm:
            submit("report", TOKEN, api=api)
        self.assertIn("did not confirm", str(cm.exception))
        self.assertEqual(api.calls, [], "no API call may happen unconfirmed")

    def test_missing_token_rejected(self):
        api = FakeApi()
        with self.assertRaises(GitHubShareError):
            submit("report", "", confirmed=True, api=api)
        self.assertEqual(api.calls, [])


# ---------------------------------------------------------------------------
# Create vs update-in-place.
# ---------------------------------------------------------------------------
class TestCreateOrUpdate(unittest.TestCase):
    def test_create_when_no_existing_issue(self):
        api = FakeApi(issues=[])
        out = submit(build_report(PAYLOAD), TOKEN, confirmed=True, api=api)
        self.assertEqual(out["action"], "created")
        self.assertEqual(out["number"], 7)
        methods = [c["method"] for c in api.calls]
        self.assertIn("POST", methods)
        self.assertNotIn("PATCH", methods)
        # the creator filter was used in the lookup
        lookup = [c for c in api.calls if "/issues?" in c["url"]][0]
        self.assertIn("creator=alice", lookup["url"])
        self.assertIn("state=all", lookup["url"])

    def test_update_in_place_via_marker(self):
        api = FakeApi(issues=[
            {"number": 3, "body": "some other issue"},
            {"number": 5, "body": f"old report\n{MARKER}"},
        ])
        out = submit(build_report(PAYLOAD), TOKEN, confirmed=True, api=api)
        self.assertEqual(out["action"], "updated")
        self.assertEqual(out["number"], 5)
        patch = [c for c in api.calls if c["method"] == "PATCH"][0]
        self.assertTrue(patch["url"].endswith("/issues/5"))
        self.assertIn(MARKER, patch["body"]["body"])
        self.assertEqual([c for c in api.calls if c["method"] == "POST"], [])

    def test_issue_without_marker_not_matched(self):
        api = FakeApi(issues=[{"number": 3, "body": "unrelated issue"}])
        out = submit(build_report(PAYLOAD), TOKEN, confirmed=True, api=api)
        self.assertEqual(out["action"], "created")

    def test_explicit_issue_number_skips_lookup(self):
        api = FakeApi()
        out = submit(build_report(PAYLOAD), TOKEN, confirmed=True,
                     existing_issue=11, api=api)
        self.assertEqual(out["action"], "updated")
        self.assertEqual(out["number"], 11)
        self.assertEqual(
            [c for c in api.calls if c["url"].endswith("/user")], [],
            "explicit issue number must not trigger the lookup")

    def test_marker_appended_when_report_lacks_it(self):
        # A body from ANOTHER route (own marker, own anonymity gate -- the
        # #271 rig artifact) still gets its marker appended if it lacks one.
        # The #152 marker cannot reach this path: a body that build_report
        # did not render is refused outright (see the scrub falsifiers).
        other = "<!-- some-other-route v1 -->"
        api = FakeApi(issues=[])
        submit("a report without a marker", TOKEN, confirmed=True,
               marker=other, api=api)
        post = [c for c in api.calls if c["method"] == "POST"][0]
        self.assertIn(other, post["body"]["body"])

    def test_default_repo_target(self):
        api = FakeApi(issues=[])
        submit(build_report(PAYLOAD), TOKEN, confirmed=True, api=api)
        post = [c for c in api.calls if c["method"] == "POST"][0]
        self.assertEqual(post["url"],
                         f"{API_ROOT}/repos/{DEFAULT_REPO}/issues")
        self.assertEqual(DEFAULT_REPO, "noonghunna/club-3090")

    def test_find_existing_issue_none(self):
        api = FakeApi(issues=[])
        self.assertIsNone(find_existing_issue(TOKEN, api=api))


if __name__ == "__main__":
    unittest.main()
