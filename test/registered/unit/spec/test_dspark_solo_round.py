"""DSpark under draft-solo placement: the round contract — CPU only.

Falsifier-first. The property under test is the one the solo refusal exists to
protect: given ONE per-round payload from the host, every rank must name the
SAME committed tokens. DSpark's confidence head truncates the block per
request, so the payload has to carry that length; a payload that carries only
the ids is the bug this file is built to catch.

The collective is faked (``FakeTpGroup``): the rank-local condition is checked
before any group collective is involved, which is the audit pattern the
hanger-family lesson prescribes. No CUDA, no scheduler, no model.
"""

import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_solo import (
    DsparkSoloRoundCodec,
    committed_prefix,
    refuse_solo_nongreedy_round,
    validate_verify_lens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

GAMMA = 5


class FakeTpGroup:
    """A TP group whose broadcast is a memcpy from the host's buffer into each
    shadow's buffer. Records every call so a test can assert the round really
    cost ONE broadcast."""

    def __init__(self, world_size: int = 3):
        self.world_size = world_size
        self.pynccl_comm = None
        self.calls = []
        self._published = None

    def publish(self, buf: torch.Tensor) -> None:
        self._published = buf.detach().clone()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> None:
        self.calls.append((tuple(tensor.shape), src))
        if self._published is None:
            raise AssertionError("shadow received a broadcast before the host sent one")
        if tensor.numel() != self._published.numel():
            raise AssertionError(
                f"shadow buffer {tensor.numel()} != host payload "
                f"{self._published.numel()}"
            )
        tensor.copy_(self._published)


def make_codec(gamma: int = GAMMA, max_bs: int = 4) -> DsparkSoloRoundCodec:
    return DsparkSoloRoundCodec(gamma=gamma, max_bs=max_bs, device=torch.device("cpu"))


def run_round(*, draft_tokens, verify_lens, graph_num_tokens, run_compact=True):
    """Host encodes -> fake broadcast -> two shadows decode. Returns the host's
    round and both shadows' rounds."""
    bs = int(draft_tokens.shape[0])
    group = FakeTpGroup(world_size=3)

    host_codec = make_codec()
    buf = host_codec.encode(
        bs=bs,
        draft_tokens=draft_tokens,
        verify_lens=verify_lens,
        graph_num_tokens=graph_num_tokens,
        run_compact=run_compact,
    )
    group.publish(buf)
    host_round = host_codec.decode(buf, bs=bs)

    shadow_rounds = []
    for _ in range(2):
        shadow_codec = make_codec()
        # The shadow sizes its buffer from rank-uniform state only.
        recv = shadow_codec.buffer(bs)
        group.broadcast(recv, src=0)
        shadow_rounds.append(shadow_codec.decode(recv, bs=bs))
    return host_round, shadow_rounds, group


class TestRoundContract(CustomTestCase):
    def test_truncated_block_commits_identically_on_every_rank(self):
        # THE falsifier: request 0 keeps its whole block, request 1 is
        # truncated by the confidence head after two positions, request 2 down
        # to the bare anchor.
        anchor = torch.tensor([11, 22, 33], dtype=torch.int64)
        draft_tokens = torch.tensor(
            [
                [101, 102, 103, 104, 105],
                [201, 202, 203, 204, 205],
                [301, 302, 303, 304, 305],
            ],
            dtype=torch.int64,
        )
        verify_lens = torch.tensor([6, 3, 1], dtype=torch.int32)
        host_round, shadow_rounds, group = run_round(
            draft_tokens=draft_tokens,
            verify_lens=verify_lens,
            graph_num_tokens=16,
        )
        expected = committed_prefix(
            anchor_tokens=anchor,
            draft_tokens=draft_tokens,
            verify_lens=verify_lens,
        )
        self.assertEqual(
            expected, [[11, 101, 102, 103, 104, 105], [22, 201, 202], [33]]
        )
        for r in [host_round] + shadow_rounds:
            got = committed_prefix(
                anchor_tokens=anchor,
                draft_tokens=r.draft_tokens,
                verify_lens=r.verify_lens,
            )
            self.assertEqual(got, expected)
        # One broadcast per round, per shadow.
        self.assertEqual(len(group.calls), 2)

    def test_dropping_the_block_length_is_detected(self):
        """CAN-FAIL ARM. Without the length in the payload a shadow can only
        assume the full block, and the test above must then fail. Executed
        here so the falsifier is known to be able to fail."""
        anchor = torch.tensor([11, 22, 33], dtype=torch.int64)
        draft_tokens = torch.tensor(
            [
                [101, 102, 103, 104, 105],
                [201, 202, 203, 204, 205],
                [301, 302, 303, 304, 305],
            ],
            dtype=torch.int64,
        )
        truncated = torch.tensor([6, 3, 1], dtype=torch.int32)
        host = committed_prefix(
            anchor_tokens=anchor, draft_tokens=draft_tokens, verify_lens=truncated
        )
        # A length-blind shadow: no verify_lens on the wire -> whole block.
        blind = committed_prefix(
            anchor_tokens=anchor,
            draft_tokens=draft_tokens,
            verify_lens=torch.full((3,), GAMMA + 1, dtype=torch.int32),
        )
        self.assertNotEqual(host, blind)

    def test_payload_is_one_tensor(self):
        draft_tokens = torch.arange(2 * GAMMA, dtype=torch.int64).view(2, GAMMA)
        _, _, group = run_round(
            draft_tokens=draft_tokens,
            verify_lens=torch.tensor([2, 6], dtype=torch.int32),
            graph_num_tokens=8,
        )
        # Two shadows, one broadcast each -- never two per rank.
        self.assertEqual([c[1] for c in group.calls], [0, 0])
        self.assertEqual(len({c[0] for c in group.calls}), 1)

    def test_shadow_buffer_size_needs_no_host_message(self):
        codec = make_codec()
        for bs in (1, 3, 4):
            self.assertEqual(codec.buffer(bs).numel(), 3 + bs * (GAMMA + 1))

    def test_run_compact_and_graph_tier_survive_the_wire(self):
        draft_tokens = torch.zeros((2, GAMMA), dtype=torch.int64)
        for compact in (True, False):
            host_round, shadows, _ = run_round(
                draft_tokens=draft_tokens,
                verify_lens=torch.tensor([4, 4], dtype=torch.int32),
                graph_num_tokens=32,
                run_compact=compact,
            )
            for r in [host_round] + shadows:
                self.assertEqual(r.run_compact, compact)
                self.assertEqual(r.graph_num_tokens, 32)

    def test_no_layout_round_declares_the_full_block(self):
        draft_tokens = torch.zeros((2, GAMMA), dtype=torch.int64)
        host_round, shadows, _ = run_round(
            draft_tokens=draft_tokens,
            verify_lens=None,
            graph_num_tokens=None,
            run_compact=False,
        )
        for r in [host_round] + shadows:
            self.assertFalse(r.has_layout)
            self.assertIsNone(r.verify_lens)


class TestPayloadRefusals(CustomTestCase):
    def test_bs_echo_mismatch_refused(self):
        codec = make_codec()
        buf = codec.encode(
            bs=2,
            draft_tokens=torch.zeros((2, GAMMA), dtype=torch.int64),
            verify_lens=torch.tensor([1, 1], dtype=torch.int32),
            graph_num_tokens=8,
            run_compact=False,
        )
        # A rank that prepared a different batch must not silently verify.
        wrong = torch.zeros(3 + 1 * (GAMMA + 1), dtype=torch.int64)
        wrong[: min(wrong.numel(), buf.numel())] = buf[: wrong.numel()]
        with self.assertRaisesRegex(ValueError, "batch-size echo mismatch"):
            codec.decode(wrong, bs=1)

    def test_wrong_block_shape_refused(self):
        codec = make_codec()
        with self.assertRaisesRegex(ValueError, "bs x gamma"):
            codec.encode(
                bs=2,
                draft_tokens=torch.zeros((2, GAMMA - 1), dtype=torch.int64),
                verify_lens=None,
                graph_num_tokens=None,
                run_compact=False,
            )

    def test_payload_length_mismatch_refused(self):
        codec = make_codec()
        with self.assertRaisesRegex(ValueError, "payload length mismatch"):
            codec.decode(torch.zeros(7, dtype=torch.int64), bs=2)

    def test_zero_block_length_refused(self):
        with self.assertRaisesRegex(ValueError, "below 1"):
            validate_verify_lens([2, 0], gamma=GAMMA, graph_num_tokens=None)

    def test_overlong_block_length_refused(self):
        with self.assertRaisesRegex(ValueError, "above gamma"):
            validate_verify_lens([GAMMA + 2], gamma=GAMMA, graph_num_tokens=None)

    def test_block_lengths_over_the_graph_tier_refused(self):
        with self.assertRaisesRegex(ValueError, "graph_num_tokens"):
            validate_verify_lens([6, 6], gamma=GAMMA, graph_num_tokens=8)

    def test_valid_lengths_pass(self):
        validate_verify_lens([1, 3, GAMMA + 1], gamma=GAMMA, graph_num_tokens=64)


class _FakeMarkovHead:
    def __init__(self, tp_shard_on: bool):
        self._opt_markov_w2_tp_shard = tp_shard_on
        self._tp_shard = "untouched"


class _FakeDraftModel:
    def __init__(self, tp_shard_on: bool):
        self._opt_markov_w2_tp_shard = tp_shard_on
        self.markov_head = _FakeMarkovHead(tp_shard_on)


class _Sampling:
    def __init__(self, all_greedy: bool):
        self.is_all_greedy = all_greedy


class TestSoloLimits(CustomTestCase):
    def test_greedy_round_allowed(self):
        refuse_solo_nongreedy_round(None)
        refuse_solo_nongreedy_round(_Sampling(True))

    def test_nongreedy_round_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "greedy acceptance only"):
            refuse_solo_nongreedy_round(_Sampling(False))

    def test_markov_w2_tp_shard_disabled_under_solo(self):
        # Default-ON optimization: solo switches it OFF (it would skip the very
        # all_gather the shadows sit in) rather than refusing the config.
        from sglang.srt.speculative.dspark_components.dspark_solo import (
            apply_solo_dspark_overrides,
        )

        model = _FakeDraftModel(tp_shard_on=True)
        apply_solo_dspark_overrides(model, tp_rank=0)
        self.assertFalse(model._opt_markov_w2_tp_shard)
        self.assertFalse(model.markov_head._opt_markov_w2_tp_shard)
        self.assertIsNone(model.markov_head._tp_shard)

    def test_override_is_a_noop_when_already_off(self):
        from sglang.srt.speculative.dspark_components.dspark_solo import (
            apply_solo_dspark_overrides,
        )

        model = _FakeDraftModel(tp_shard_on=False)
        apply_solo_dspark_overrides(model, tp_rank=0)
        self.assertFalse(model._opt_markov_w2_tp_shard)
        # Untouched: the split path must not see a cleared shard geometry.
        self.assertEqual(model.markov_head._tp_shard, "untouched")


if __name__ == "__main__":
    unittest.main()
