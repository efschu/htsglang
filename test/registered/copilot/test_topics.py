"""Topic warmth: the residency probe, and its can-fail proof.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        python -m pytest test/registered/copilot/test_topics.py -v

Why this file exists: "we primed the topic" is a success message about state,
and this tree has an explicit law that such a message is not evidence. There is
no pin API (``mem_cache/radix_cache.py:598-625`` -- ``lock_ref`` is
in-flight-only), so a primed prefix can be evicted between requests without
anything saying so. The only honest report of warmth is the measured
``cached_tokens`` of a subsequent request, and the runtime reports a ZERO hit
by OMITTING the details object (``usage_processor.py:15``). A probe that reads
the absent object as "unknown" would never see a miss.
"""

import asyncio

import pytest

from sglang.srt.copilot.briefing import parse_briefing
from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.hints import DeskFakeHints, prime_chat_request
from sglang.srt.copilot.topics import (
    TopicRegistry,
    Warmth,
    read_cached_tokens,
)

BRIEFING = """# Briefing

## Contract renewal
The current contract ends in March.

## Migration timeline
Two clusters move in Q3.
"""


def registry() -> TopicRegistry:
    reg = TopicRegistry(config=CopilotConfig())
    reg.sync_from_briefing(parse_briefing(BRIEFING))
    return reg


class TestCachedTokenReading:
    def test_absent_details_means_zero_not_unknown(self):
        """Pins ``usage_processor.py:15``'s ``if count > 0 else None`` shape."""
        assert read_cached_tokens({"prompt_tokens": 100}) == 0
        assert (
            read_cached_tokens({"prompt_tokens": 100, "prompt_tokens_details": None})
            == 0
        )
        assert read_cached_tokens(None) == 0

    def test_present_details_are_read(self):
        usage = {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 90}}
        assert read_cached_tokens(usage) == 90

    def test_null_cached_tokens_field_means_zero(self):
        usage = {"prompt_tokens_details": {"cached_tokens": None}}
        assert read_cached_tokens(usage) == 0


class TestWarmthVerdicts:
    def test_prime_alone_does_not_make_a_topic_warm(self):
        reg = registry()
        reg.record_prime("contract-renewal", prompt_tokens=200)
        assert reg.get("contract-renewal").warmth is Warmth.UNKNOWN

    def test_full_hit_is_warm(self):
        reg = registry()
        reg.record_prime("contract-renewal", 200)
        assert reg.observe("contract-renewal", 200) is Warmth.WARM

    def test_partial_hit_is_partial(self):
        reg = registry()
        reg.record_prime("contract-renewal", 200)
        assert reg.observe("contract-renewal", 100) is Warmth.PARTIAL

    def test_no_hit_is_cold_and_counted_as_a_miss(self):
        reg = registry()
        reg.record_prime("contract-renewal", 200)
        assert reg.observe("contract-renewal", 0) is Warmth.COLD
        assert reg.get("contract-renewal").misses == 1

    def test_observation_without_a_prime_is_cold(self):
        reg = registry()
        assert reg.observe("contract-renewal", 0) is Warmth.COLD

    def test_changed_prefix_invalidates_measured_warmth(self):
        """A rewritten section is a different token path.

        Carrying the old verdict over would report warmth for a prefix that no
        longer exists.
        """
        reg = registry()
        reg.record_prime("contract-renewal", 200)
        reg.observe("contract-renewal", 200)
        assert reg.get("contract-renewal").warmth is Warmth.WARM
        changed = reg.sync_from_briefing(
            parse_briefing(BRIEFING.replace("ends in March", "ends in April"))
        )
        assert "contract-renewal" in changed
        assert reg.get("contract-renewal").warmth is Warmth.UNKNOWN
        assert reg.get("contract-renewal").primed_tokens == 0

    def test_miss_report_aggregates(self):
        reg = registry()
        reg.record_prime("contract-renewal", 100)
        reg.record_prime("migration-timeline", 100)
        reg.observe("contract-renewal", 100)
        reg.observe("migration-timeline", 0)
        report = reg.miss_report()
        assert report["observations"] == 2
        assert report["misses"] == 1
        assert report["miss_rate"] == pytest.approx(0.5)


class TestPriming:
    def test_prime_request_is_prefill_only_and_high_priority(self):
        config = CopilotConfig()
        reg = registry()
        req = reg.prime_request("contract-renewal")
        assert req.max_tokens == 0
        assert req.priority == config.hint_priority
        chat = prime_chat_request(config, "contract-renewal", req.prompt)
        assert chat.max_tokens == 0
        # Priming must never occupy the fast lane -- that lane is for the live
        # hint the user is waiting to read.
        assert chat.lane is None

    def test_prefix_contains_the_briefing_header(self):
        reg = registry()
        prefix = reg.get("contract-renewal").prefix_text
        assert prefix.startswith("# Briefing")
        assert "Contract renewal" in prefix

    def test_touch_cadence(self):
        config = CopilotConfig(topic_touch_interval_s=30.0)
        reg = TopicRegistry(config=config)
        reg.sync_from_briefing(parse_briefing(BRIEFING))
        assert len(reg.due_for_prime(now=0.0)) == 2
        reg.record_prime("contract-renewal", 10, now=0.0)
        reg.record_prime("migration-timeline", 10, now=0.0)
        assert reg.due_for_prime(now=10.0) == []
        assert len(reg.due_for_prime(now=31.0)) == 2


class TestDeskFakeDifference:
    """The desk fake's named difference, executed.

    ``DeskFakeHints`` claims a perfect cache hit on every call. A probe tested
    only against it never sees a miss and is therefore untested. These two
    tests are the can-fail proof: the same probe code produces WARM against the
    optimistic fake and COLD against the pessimistic one.
    """

    def test_optimistic_fake_always_reports_warm(self):
        config = CopilotConfig()
        reg = registry()
        backend = DeskFakeHints(config=config, always_warm=True)
        req = reg.prime_request("contract-renewal")
        result = asyncio.run(
            backend.complete(prime_chat_request(config, "contract-renewal", req.prompt))
        )
        reg.record_prime("contract-renewal", result.prompt_tokens)
        assert reg.observe_usage("contract-renewal", result.usage) is Warmth.WARM

    def test_pessimistic_fake_reports_cold_through_the_same_probe(self):
        config = CopilotConfig()
        reg = registry()
        backend = DeskFakeHints(config=config, always_warm=False, cached_fraction=0.0)
        req = reg.prime_request("contract-renewal")
        result = asyncio.run(
            backend.complete(prime_chat_request(config, "contract-renewal", req.prompt))
        )
        reg.record_prime("contract-renewal", result.prompt_tokens)
        # The miss arrives as an ABSENT details object, exactly as the runtime
        # would report it.
        assert "prompt_tokens_details" not in result.usage
        assert reg.observe_usage("contract-renewal", result.usage) is Warmth.COLD
        assert reg.get("contract-renewal").misses == 1
