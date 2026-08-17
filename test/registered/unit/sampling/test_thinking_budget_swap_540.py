"""#540: the derived marker ids must survive a CHECKPOINT SWAP.

The existing suite proves the ids come from the tokenizer rather than from
class constants, on ONE checkpoint. That is the defect that was reported. It
is not quite the property the feature promises: "derived, not hardcoded" only
pays off if the SAME running code answers differently for a DIFFERENT
checkpoint, and keeps answering correctly when checkpoints alternate.

The failure this closes is the one hardcoding always turns into: a value
resolved once and then reused. A cache keyed on nothing, a module-level global,
or an lru_cache on a tokenizer object that compares equal would all pass the
single-checkpoint tests and silently serve checkpoint A's ids to checkpoint B —
which is exactly the original bug with an extra step, and it would cap thinking
at a token id that means something else entirely in the new vocabulary.

Hermetic: no server, no model load, no GPU. The fixture tokenizers carry the
real Qwen3.6 ids and a deliberately different second vocabulary.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest  # noqa: E402

from sglang.srt.sampling.thinking_budget import (  # noqa: E402
    ThinkingBudgetUnsupportedError,
    resolve_thinking_budget_token_ids,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402

# Checkpoint A: the real Qwen3.6-27B ids (same values the sibling suite uses).
A_START, A_END, A_NEWLINE = 248068, 248069, 198

# Checkpoint B: a deliberately different vocabulary. The values are the LEGACY
# hardcoded Qwen3 ids, chosen on purpose -- if anything ever reintroduces the
# constants, a test that used arbitrary numbers here would still pass while
# this one pins that B is served B's ids because B's TOKENIZER said so.
B_START, B_END, B_NEWLINE = 151667, 151668, 271


class _Tokenizer:
    """Encodes only the markers; everything else is multi-token by design."""

    def __init__(self, name, start, end, newline):
        self.name_or_path = name
        self.table = {
            "<think>": [start],
            "</think>": [end],
            "\n": [newline],
        }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text in self.table:
            return list(self.table[text])
        return [ord(c) for c in text] + [0]


def _tok_a():
    return _Tokenizer("checkpoint-A", A_START, A_END, A_NEWLINE)


def _tok_b():
    return _Tokenizer("checkpoint-B", B_START, B_END, B_NEWLINE)


class TestTheIdsFollowTheCheckpoint(CustomTestCase):
    def test_two_checkpoints_get_their_own_ids(self):
        a = resolve_thinking_budget_token_ids(_tok_a(), "qwen3", "A")
        b = resolve_thinking_budget_token_ids(_tok_b(), "qwen3", "B")

        self.assertEqual((a.start, a.end, a.newline), (A_START, A_END, A_NEWLINE))
        self.assertEqual((b.start, b.end, b.newline), (B_START, B_END, B_NEWLINE))
        self.assertNotEqual(a.start, b.start)
        self.assertNotEqual(a.end, b.end)

    def test_resolving_b_after_a_does_not_serve_a_s_ids(self):
        """The swap direction that a cache would break."""
        resolve_thinking_budget_token_ids(_tok_a(), "qwen3", "A")
        b = resolve_thinking_budget_token_ids(_tok_b(), "qwen3", "B")
        self.assertEqual(b.start, B_START, "checkpoint A's ids leaked into B")

    def test_swapping_back_restores_the_first_checkpoints_ids(self):
        """A -> B -> A. A one-shot memo passes the first two and fails here."""
        resolve_thinking_budget_token_ids(_tok_a(), "qwen3", "A")
        resolve_thinking_budget_token_ids(_tok_b(), "qwen3", "B")
        again = resolve_thinking_budget_token_ids(_tok_a(), "qwen3", "A")
        self.assertEqual((again.start, again.end), (A_START, A_END))

    def test_alternating_many_times_is_stable(self):
        """Order independence, not just a single round trip."""
        for _ in range(4):
            a = resolve_thinking_budget_token_ids(_tok_a(), "qwen3", "A")
            b = resolve_thinking_budget_token_ids(_tok_b(), "qwen3", "B")
            self.assertEqual(a.start, A_START)
            self.assertEqual(b.start, B_START)

    def test_two_tokenizer_objects_of_the_same_checkpoint_agree(self):
        """Distinct objects, same vocabulary: the answer must not depend on
        object identity either."""
        first = resolve_thinking_budget_token_ids(_tok_a(), "qwen3", "A")
        second = resolve_thinking_budget_token_ids(_tok_a(), "qwen3", "A")
        self.assertEqual((first.start, first.end), (second.start, second.end))

    def test_a_checkpoint_without_atomic_markers_is_refused_not_defaulted(self):
        """The swap that must FAIL rather than fall back.

        A checkpoint whose markers are not single tokens has no id to cap on.
        Falling back to the previous checkpoint's ids would be the corrupting
        outcome; the contract is a named refusal.
        """

        class _NoMarkers(_Tokenizer):
            def __init__(self):
                super().__init__("checkpoint-C", 0, 0, 0)
                self.table = {"\n": [198]}

        resolve_thinking_budget_token_ids(_tok_a(), "qwen3", "A")
        with self.assertRaises(ThinkingBudgetUnsupportedError) as ctx:
            resolve_thinking_budget_token_ids(_NoMarkers(), "qwen3", "C")
        self.assertIn("C", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
