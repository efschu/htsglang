"""H90 (cu130 acceptance dkrnfbar1agent09252021, 25.09.2026): under agent load
(Claude Code over ``/v1/messages``) the Weg 2 front stopped with W3
``Weg2DrainWitnessDisagreement`` -- PP1 and PP2 still held a request in their
waiting queues that PP0 had refused 503 (WEG2-INTAKE-STALL).

The front names every request ``weg2-<epoch>-<n>`` in the payload's ``rid`` and
aborts it by that name on every rank. ``AnthropicMessagesRequest`` did not
declare ``rid`` (pydantic ``extra="ignore"``), so the scheduler ran it under a
fresh uuid, the abort ``rid=weg2-42-33`` matched nothing on PP1/PP2, and the
next flip's flush failed. Hermetic: request model in, converted request out."""

import unittest

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing


class _NoReasoning:
    def apply_reasoning_enabled(self, request, enabled):
        pass


def _convert(**kwargs):
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
    base.update(kwargs)
    serving = AnthropicServing.__new__(AnthropicServing)
    serving._merge_inline_system = False
    serving.openai_serving_chat = _NoReasoning()
    return AnthropicServing._convert_to_chat_completion_request(
        serving, AnthropicMessagesRequest(**base))


class TestTheCallersRidReachesTheScheduler(unittest.TestCase):
    def test_the_front_rid_survives_the_conversion(self):
        """THE DEFECT: the rid the front aborts by must be the rid the
        scheduler queues under; dropped, the abort misses every follower."""
        self.assertEqual(_convert(rid="weg2-42-33").rid, "weg2-42-33")

    def test_absent_stays_absent(self):
        """No invented id: without a caller rid the server mints its own."""
        self.assertIsNone(_convert().rid)

    def test_streaming_keeps_it_too(self):
        """Claude Code streams; the stream branch builds the same request."""
        self.assertEqual(_convert(rid="weg2-7-1", stream=True).rid, "weg2-7-1")


if __name__ == "__main__":
    unittest.main()
