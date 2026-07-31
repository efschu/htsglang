# SPDX-License-Identifier: Apache-2.0
"""#326: the s12 accept probe read /v1/chat/completions, which never attaches
`meta_info` unless the request opts in (`protocol.py` `return_meta_info: bool
= False`, never set by the probe). `spec_accept_length` came back None on all
eight arms of the 2026-07-30 run, silently, and every `ms_pro_verify` derived
from those arms was void -- see #320 and the s12 sections of
docs/dev/INTEGRATION_R3_VALIDATION.md dated 2026-07-30.

The fix moves the probe to the native `/generate` endpoint (the pattern the
s14 scripts in this same directory already use, see s14_decode_punkt.py's
`Stream` and its own accept probe), where `meta_info` rides on every response
with no opt-in flag. A None `spec_accept_length` must from now on carry a
named reason (`accept_probe_note`) and, when the probe itself could not be
answered at all -- the exact shape of the original bug, a response carrying
no `meta_info` whatsoever -- `accept_probe_fatal` must be True so
`mode_measure` fails loudly (non-zero exit) instead of letting the run pass
with a void measurement.

Every string this file asserts against is IMPORTED from
scripts/gpu_battery/s12_prefill_kurve.py, never retyped -- the #315 lesson:
a hand-typed copy of an emitted literal is exactly what let the s11 BAR1
regexes drift onto dead German wording without any test noticing.

Hermetic and CPU-only: no card, no host, no ssh, no server. Every HTTP call
is mocked at ``urllib.request.urlopen``.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
BATTERY = os.path.join(REPO_ROOT, "scripts", "gpu_battery")
sys.path.insert(0, BATTERY)

import s12_prefill_kurve as s12  # noqa: E402


class _Resp:
    """Minimal stand-in for the object ``urllib.request.urlopen`` returns:
    a context manager whose ``.read()`` gives the response bytes."""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def _mock_urlopen(payload: dict):
    return mock.patch("urllib.request.urlopen", return_value=_Resp(payload))


# ---------------------------------------------------------------------------
# marker coupling (#315 lesson): the probe must hit the path this module
# names, never a hand-typed literal re-guessed here.
# ---------------------------------------------------------------------------


class TestAcceptProbeHitsTheNamedPathNotChatCompletions:
    def test_probe_path_constant_is_generate(self):
        # Regression guard for the exact bug: the constant this module's HTTP
        # call actually uses must be /generate, never the chat endpoint that
        # caused #326.
        assert s12.ACCEPT_PROBE_PATH == "/generate"

    def test_probe_actually_requests_the_named_path(self):
        """Couples the constant to the REAL call site: patch urlopen, read
        back the Request object it was given, and check its full_url against
        s12.ACCEPT_PROBE_PATH -- not against a retyped "/generate" string."""
        captured = {}

        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _Resp({"text": "ok", "meta_info": {"spec_accept_length": 2.5}})

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            s12._probe_accept_length(port=30030)

        assert captured["url"].endswith(s12.ACCEPT_PROBE_PATH)
        assert "chat/completions" not in captured["url"]


# ---------------------------------------------------------------------------
# the bug's exact shape: a response with no meta_info at all
# ---------------------------------------------------------------------------


class TestNoMetaInfoIsFatalAndNamed:
    def test_response_without_meta_info_is_fatal(self):
        # Shaped like what /v1/chat/completions returned on 2026-07-30: a
        # normal 200 answer that simply carries no meta_info block at all.
        # Hitting this shape at the probe's OWN endpoint must fail loudly.
        with _mock_urlopen({"text": "hello"}):
            out = s12._probe_accept_length(port=30030)
        assert out["spec_accept_length"] is None
        assert out["accept_probe_fatal"] is True
        assert out["accept_probe_note"] == s12.ACCEPT_PROBE_NOTE_NO_META_INFO

    def test_response_with_empty_meta_info_is_fatal(self):
        with _mock_urlopen({"text": "hello", "meta_info": {}}):
            out = s12._probe_accept_length(port=30030)
        assert out["accept_probe_fatal"] is True
        assert out["accept_probe_note"] == s12.ACCEPT_PROBE_NOTE_NO_META_INFO

    def test_error_object_is_fatal(self):
        with _mock_urlopen({"error": {"message": "boom"}}):
            out = s12._probe_accept_length(port=30030)
        assert out["spec_accept_length"] is None
        assert out["accept_probe_fatal"] is True
        assert out["accept_probe_note"] == s12.ACCEPT_PROBE_NOTE_GENERATE_ERROR.format(
            message="boom"
        )

    def test_network_error_is_fatal(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            out = s12._probe_accept_length(port=30030)
        assert out["spec_accept_length"] is None
        assert out["accept_probe_fatal"] is True
        assert "OSError" in out["accept_probe_note"]


# ---------------------------------------------------------------------------
# the legitimate case: meta_info present, no spec block, not fatal
# ---------------------------------------------------------------------------


class TestNoSpecAcceptLengthIsNotedButNotFatal:
    def test_meta_info_without_spec_block_is_noted_not_fatal(self):
        # A boot without speculative decoding: meta_info is real, but there is
        # no spec_accept_length key at all. This is a TRUE absence, not a
        # broken probe, so it must be named but must NOT fail the run.
        with _mock_urlopen({"text": "hello", "meta_info": {"completion_tokens": 5}}):
            out = s12._probe_accept_length(port=30030)
        assert out["spec_accept_length"] is None
        assert out["accept_probe_fatal"] is False
        assert out["accept_probe_note"] == s12.ACCEPT_PROBE_NOTE_NO_SPEC_ACCEPT_LENGTH


class TestHappyPathReadsCleanly:
    def test_meta_info_with_spec_accept_length_reads_through(self):
        with _mock_urlopen(
            {"text": "hello", "meta_info": {"spec_accept_length": 3.25}}
        ):
            out = s12._probe_accept_length(port=30030)
        assert out["spec_accept_length"] == 3.25
        assert out["accept_probe_fatal"] is False
        assert out["accept_probe_note"] is None
        assert out["output_sample"] == "hello"


# ---------------------------------------------------------------------------
# end to end: mode_measure must fail loudly (non-zero exit) when a decode
# point's accept probe came back structurally fatal
# ---------------------------------------------------------------------------


class TestModeMeasureFailsLoudlyOnFatalAcceptProbe:
    def _args(self, out_dir: str, **overrides) -> SimpleNamespace:
        base = dict(
            out_dir=out_dir,
            arm="bar1",
            sessions=1,
            folge=0,
            point_seconds=1.0,
            warmup_seconds=0.0,
            prompt_tokens=64,
            with_decode=1,
            decode_batches="1",
            server_log="",
            port=30030,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_fatal_accept_probe_makes_mode_measure_return_nonzero(self, tmp_path):
        args = self._args(str(tmp_path))
        fake_prefill = {"prefill_tok_s": 123.4, "requests": 4, "roh": []}
        fake_decode = {
            "batch": 1,
            "spec_accept_length": None,
            "accept_probe_note": s12.ACCEPT_PROBE_NOTE_NO_META_INFO,
            "accept_probe_fatal": True,
        }
        with mock.patch.object(
            s12, "measure_prefill", return_value=dict(fake_prefill)
        ), mock.patch.object(s12, "measure_decode", return_value=dict(fake_decode)):
            rc = s12.mode_measure(args)
        assert rc == 1

    def test_non_fatal_accept_probe_absence_does_not_fail_the_point(self, tmp_path):
        args = self._args(str(tmp_path))
        fake_prefill = {"prefill_tok_s": 123.4, "requests": 4, "roh": []}
        fake_decode = {
            "batch": 1,
            "spec_accept_length": None,
            "accept_probe_note": s12.ACCEPT_PROBE_NOTE_NO_SPEC_ACCEPT_LENGTH,
            "accept_probe_fatal": False,
        }
        with mock.patch.object(
            s12, "measure_prefill", return_value=dict(fake_prefill)
        ), mock.patch.object(s12, "measure_decode", return_value=dict(fake_decode)):
            rc = s12.mode_measure(args)
        assert rc == 0
