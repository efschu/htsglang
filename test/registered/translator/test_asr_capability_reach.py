# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The advertised ASR set must not follow the current selection (§19.14).

The field symptom was that only German and Spanish were ever selectable, on a
recognizer that hears 102 languages. The cause was one attribute doing two
jobs: `_restrict` held BOTH the deployment's configured bound and the live
decode whitelist that `set_restrict_languages` pushes from the session's
participant set before every recognition (§17.8.6). `supported_languages()`
read that attribute, so `GET /api/translator/languages` answered with
whichever conversation had spoken last -- a capability report narrowed by the
selection it exists to inform.

Measured on the live service before the fix: `stages.asr = ["de", "es"]`,
`sources = ["de", "es"]`, and every other language marked
`"reason": "ASR cannot hear it"`.

These drive the SHIPPED adapters, constructed without their models (the model
is irrelevant to a capability answer, and requiring one would make this a GPU
test for a pure bookkeeping defect).
"""

import unittest

from sglang.srt.translator.asr_backends import FasterWhisperAsr, NemoStreamingAsr


def _whisper(deployment=()):
    """A FasterWhisperAsr with its language bookkeeping and no model."""
    asr = FasterWhisperAsr.__new__(FasterWhisperAsr)
    asr._deployment_languages = [str(c).lower() for c in deployment]
    asr._restrict = list(asr._deployment_languages)
    return asr


def _nemo(deployment=(), declared=("de", "es", "fr", "it")):
    asr = NemoStreamingAsr.__new__(NemoStreamingAsr)
    asr._deployment_languages = [str(c).lower() for c in deployment]
    asr._restrict = list(asr._deployment_languages)
    asr._languages = list(declared)
    return asr


class TestAsrCapabilityReach(unittest.TestCase):
    def test_a_session_whitelist_does_not_narrow_the_capability(self):
        """THE falsifier: this is the exact sequence the live service runs."""
        asr = _whisper()
        before = set(asr.supported_languages())
        self.assertGreater(
            len(before), 50,
            "an unrestricted whisper must advertise its full set; got "
            f"{sorted(before)}",
        )
        # What session.py:1510 does before EVERY recognition.
        asr.set_restrict_languages(["de", "es"])
        after = set(asr.supported_languages())
        self.assertEqual(
            before, after,
            "the participant whitelist narrowed the ADVERTISED set: "
            f"{sorted(after)}",
        )

    def test_the_whitelist_still_reaches_the_decode(self):
        """The fix must not disarm §17.8.6 -- the constraint still applies."""
        asr = _whisper()
        asr.set_restrict_languages(["de", "es"])
        self.assertEqual(asr._restrict, ["de", "es"])

    def test_a_deployment_bound_is_still_a_capability(self):
        """An operator's configured restriction is a real, honest narrowing."""
        asr = _whisper(deployment=["de", "es", "fr"])
        self.assertEqual(set(asr.supported_languages()), {"de", "es", "fr"})
        # ... and it survives a session pushing something narrower.
        asr.set_restrict_languages(["de"])
        self.assertEqual(set(asr.supported_languages()), {"de", "es", "fr"})

    def test_the_nemo_adapter_has_the_same_split(self):
        asr = _nemo()
        before = set(asr.supported_languages())
        self.assertEqual(before, {"de", "es", "fr", "it"})
        asr.set_restrict_languages(["de", "es"])
        self.assertEqual(set(asr.supported_languages()), before)

    def test_nemo_honours_a_deployment_bound(self):
        asr = _nemo(deployment=["de", "es"])
        self.assertEqual(set(asr.supported_languages()), {"de", "es"})


if __name__ == "__main__":
    unittest.main()
