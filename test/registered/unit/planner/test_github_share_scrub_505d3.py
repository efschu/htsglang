# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#505-D3 falsifiers: the #152 result-share route must be anonymity-gated.

The sibling rig-artifact route runs ``scrub_tree`` + ``assert_anonymized``
inside ``build_digest`` (``rig_artifact.py:784-795``) so the three steps cannot
be separated. The #152 route rendered the start command verbatim, so a real
launch command -- ``/spinning/llm_stuff/club-3090/models-cache/...``, the
hostname, ``$USER``, a credential in a variable whose NAME does not end in one
of the five redacted suffixes -- went into a PUBLIC issue body unchanged.

Every test here fails on the unfixed module and passes on the fixed one.
Hermetic: no network, the GitHub REST API is a fake.
"""

import os
import socket
import unittest
from unittest import mock

from sglang.srt.planner.github_share import (
    DEFAULT_REPO,
    MARKER,
    GitHubShareError,
    build_report,
    submit,
)

TOKEN = "ghp_TESTTOKENFORTHISTESTONLY123456"

#: A REAL start command from this rig (CLAUDE.md reference command).
MODEL_PATH = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-AWQ-BF16-INT4"


def _payload(**over):
    p = {
        "model": "Qwen3.6-27B",
        "hardware": "1x RTX 5090 + 2x RTX 3080",
        "command": {
            "argv": [
                "python3", "-m", "sglang.launch_server",
                "--model-path", MODEL_PATH,
                "--tokenizer", MODEL_PATH,
                "--tp", "3",
            ],
            "env": {"SGLANG_UNEVEN_TOKEN_VECTOR": "33,13,18"},
        },
        "metrics": {"decode_tok_s": 42.5},
    }
    p.update(over)
    return p


class FakeApi:
    """Records calls; scripted responses. No network."""

    def __init__(self, issues=None, login="alice"):
        self.calls = []
        self.issues = issues if issues is not None else []
        self.login = login

    def __call__(self, method, url, token, body=None, timeout=30.0):
        self.calls.append({"method": method, "url": url, "body": body})
        if method == "GET" and url.endswith("/user"):
            return 200, {"login": self.login}
        if method == "GET" and "/issues?" in url:
            return 200, self.issues
        if method == "POST" and url.endswith("/issues"):
            return 201, {"number": 7,
                         "html_url": f"https://github.com/{DEFAULT_REPO}/issues/7"}
        if method == "PATCH" and "/issues/" in url:
            return 200, {"number": 5, "html_url": "https://x/5"}
        return 404, {}


class AbsolutePathTest(unittest.TestCase):
    """The named falsifier: a realistic model path in ``command.argv``."""

    def test_absolute_model_path_is_not_in_the_posted_markdown(self):
        md = build_report(_payload())
        self.assertNotIn(MODEL_PATH, md)
        self.assertNotIn("/spinning", md)
        self.assertNotIn("club-3090", md)
        # The information that is actually shareable survives: the basename.
        self.assertIn("Qwen3.6-27B-AWQ-BF16-INT4", md)

    def test_path_valued_env_is_not_in_the_posted_markdown(self):
        md = build_report(_payload(command={
            "argv": ["python3"],
            "env": {"SGLANG_MOE_HOTSET_FILE": "/spinning/htsglang/hotset.json"},
        }))
        self.assertNotIn("/spinning/htsglang/hotset.json", md)
        self.assertIn("hotset.json", md)

    def test_path_in_notes_is_not_in_the_posted_markdown(self):
        md = build_report(_payload(notes=f"booted from {MODEL_PATH}"))
        self.assertNotIn(MODEL_PATH, md)


class MachineIdentityTest(unittest.TestCase):
    def test_hostname_is_not_in_the_posted_markdown(self):
        host = socket.gethostname()
        self.assertTrue(host, "no hostname to test with")
        md = build_report(_payload(notes=f"run on {host}"))
        self.assertNotIn(host, md)

    def test_username_is_not_in_the_posted_markdown(self):
        with mock.patch.dict(os.environ, {"USER": "efschu"}, clear=False):
            md = build_report(_payload(notes="started by efschu"))
        self.assertNotIn("efschu", md)

    def test_ip_and_gpu_uuid_are_not_in_the_posted_markdown(self):
        md = build_report(_payload(
            notes="peer 192.168.0.89 card "
                  "GPU-12345678-1234-1234-1234-123456789abc"))
        self.assertNotIn("192.168.0.89", md)
        self.assertNotIn("GPU-12345678-1234-1234-1234-123456789abc", md)


class SecretInAnyVariableTest(unittest.TestCase):
    """The five NAME suffixes were the ONLY redaction; a credential in a
    differently-named variable was posted verbatim."""

    def test_credential_value_in_a_non_suffixed_variable_is_redacted(self):
        md = build_report(_payload(command={
            "argv": ["python3"],
            "env": {
                "MY_CREDENTIAL": "hf_abcdefghijklmnopqrstuvwxyz012345",
                "CI_HELPER": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            },
        }))
        self.assertNotIn("hf_abcdefghijklmnopqrstuvwxyz012345", md)
        self.assertNotIn("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", md)

    def test_suffix_redaction_is_still_exactly_as_strict(self):
        md = build_report(_payload(command={
            "argv": ["python3"],
            "env": {"HF_TOKEN": "hf_secretsecretsecret",
                    "SGLANG_UNEVEN_TOKEN_VECTOR": "33,13,18"},
        }))
        self.assertIn("HF_TOKEN=<redacted>", md)
        self.assertNotIn("hf_secretsecretsecret", md)
        # a tuning knob whose name merely ENDS in a suffix-lookalike stays exact
        self.assertIn("SGLANG_UNEVEN_TOKEN_VECTOR=33,13,18", md)


class QualityShotUntouchedTest(unittest.TestCase):
    """The chess SVG is markup, not collected environment: basenaming it
    would corrupt it (``</defs>`` -> ``<defs>``, the xmlns URI -> ``svg``).
    It is the ONE documented exception and must stay byte-exact."""

    SVG = ('<svg xmlns="http://www.w3.org/2000/svg">'
           '<defs><rect id="a"/></defs><g><use href="#a"/></g></svg>')

    def test_svg_survives_the_scrub_unchanged(self):
        md = build_report(_payload(quality={
            "svg": self.SVG, "verdict": "correct",
            "tokens": {"prompt": 220, "completion": 3900, "total": 4120},
        }))
        self.assertIn(self.SVG, md)
        self.assertIn("Verdict: **correct**", md)


class GateFailsClosedTest(unittest.TestCase):
    """``scrub_tree`` rewrites string LEAVES; it does not rewrite dict KEYS.
    That is precisely why the gate checks the SERIALIZED form. An identifying
    key must abort the render, not produce a half-scrubbed report."""

    def test_identifying_dict_key_aborts_the_render(self):
        payload = _payload(metrics={
            "decode_tok_s": 42.5,
            "per_card": {"/spinning/rig/gpu0": {"power_w": 420}},
        })
        with self.assertRaises(GitHubShareError) as cm:
            build_report(payload)
        self.assertIn("refusing", str(cm.exception))

    def test_an_aborted_render_cannot_then_be_submitted(self):
        api = FakeApi(issues=[])
        payload = _payload(metrics={
            "per_card": {"/spinning/rig/gpu0": {"power_w": 420}}})
        try:
            body = build_report(payload)
        except GitHubShareError:
            body = None
        self.assertIsNone(body)
        with self.assertRaises(GitHubShareError):
            submit(f"per_card /spinning/rig/gpu0\n{MARKER}", TOKEN,
                   confirmed=True, api=api)
        self.assertEqual(api.calls, [])


class PreviewIsTheSubmittedBodyTest(unittest.TestCase):
    """The preview only works as a consent mechanism if the bytes the user
    approved are the bytes that get posted."""

    def test_posted_body_is_byte_identical_to_the_preview(self):
        preview = build_report(_payload())
        api = FakeApi(issues=[])
        submit(preview, TOKEN, confirmed=True, api=api)
        post = [c for c in api.calls if c["method"] == "POST"][0]
        self.assertEqual(post["body"]["body"], preview)
        self.assertNotIn(MODEL_PATH, post["body"]["body"])

    def test_submit_refuses_a_body_this_process_did_not_render(self):
        api = FakeApi(issues=[])
        hand_made = f"## hand-made\n\n```\npython3 --model-path {MODEL_PATH}\n```\n{MARKER}"
        with self.assertRaises(GitHubShareError) as cm:
            submit(hand_made, TOKEN, confirmed=True, api=api)
        self.assertIn("build_report", str(cm.exception))
        self.assertEqual(api.calls, [],
                         "a body that skipped the scrub must reach no API call")


if __name__ == "__main__":
    unittest.main()
