# SPDX-License-Identifier: Apache-2.0
"""#488 graphs cut slice 2: what can be proven about the capture without a card.

Capture itself needs a GPU. Everything that DECIDES whether a replay is correct
does not, and it is all here, because every failure mode this slice introduces
is silent:

1. **The cache geometry.** Predictor step ``g`` writes slot ``1 + g``, and the
   scratch cache is sized by the last of them. An off-by-one makes the final
   residual group attend to an unwritten slot -- stale keys from the previous
   frame, a plausible token, degraded timbre, no error.

2. **The mask.** ``StaticLayer.update`` returns the WHOLE padded cache and
   ``get_mask_sizes`` reports the full length (``cache_utils.py:457-461``), so
   attention sees unwritten slots on every single step and the mask is the only
   thing keeping them out. Its exact shape is asserted, including the causal
   diagonal on the two-token prefill.

3. **The refusals.** Four conditions that corrupt audio without raising:
   a cache whose write position is a Python int, a sliding-window layer whose
   mask sizes come from a host int, a shared-pool replay out of capture order,
   and a replay before any capture. Each is exercised until it fires -- a gate
   that has never been seen to fire is not known to be a gate.

The GPU arm -- capture, token identity against the eager path, and the ladder
measurement -- is ``scripts/dev/488_talker_profile/validate_graphs.py``.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")

import torch

from sglang.srt.models.qwen3_tts_fast_predictor import step_schedule
from sglang.srt.models.qwen3_tts_graph_driver import (
    GraphCaptureRefusal,
    _OrderedReplay,
    decode_mask,
    predictor_cache_lengths,
    refuse_unless_graph_safe,
    reset_cache_positions,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeLayer:
    def __init__(self, position=None):
        self.cumulative_length = (
            torch.tensor([0]) if position is None else position
        )


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers


class _FakeStep:
    def __init__(self, name):
        self.name = name
        self.replays = 0

    def replay(self):
        self.replays += 1


class TestPredictorCacheGeometry(CustomTestCase):
    """The scratch cache is 0.3 MiB, so the only way to get it wrong is arithmetic."""

    def test_prefill_occupies_two_slots(self):
        # The reference prefills with cat((past_hidden, last_id_hidden)):
        # modeling_qwen3_tts.py:1671. Two, not one.
        self.assertEqual(predictor_cache_lengths(16)[0], 2)

    def test_one_slot_per_step_after_prefill(self):
        lengths = predictor_cache_lengths(16)
        self.assertEqual(lengths, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    def test_length_matches_the_step_schedule(self):
        # The two must agree or a step writes a slot another step owns.
        self.assertEqual(len(predictor_cache_lengths(16)), len(step_schedule(16)))

    def test_scratch_size_is_the_last_occupancy(self):
        # ANALYSE_488 §7.3 specifies a 17-slot scratch; 16 is what the frame
        # actually reaches, and the design's spare slot is headroom, not a
        # requirement. Pinned so a change to either is deliberate.
        self.assertEqual(max(predictor_cache_lengths(16)), 16)

    def test_degenerate_geometry_refuses(self):
        with self.assertRaises(ValueError):
            predictor_cache_lengths(2)


class TestDecodeMask(CustomTestCase):
    def test_decode_step_admits_exactly_the_written_slots(self):
        mask = decode_mask(valid_len=5, cache_len=17, query_len=1)
        self.assertEqual(tuple(mask.shape), (1, 1, 1, 17))
        self.assertEqual(int(mask.sum()), 5)
        self.assertTrue(bool(mask[0, 0, 0, 4]))
        self.assertFalse(bool(mask[0, 0, 0, 5]))

    def test_prefill_is_causal_within_its_two_tokens(self):
        # The second prefill token may see the first; the first may not see
        # the second. Getting this backwards leaks a future token into the
        # frame's first residual group.
        mask = decode_mask(valid_len=2, cache_len=17, query_len=2)
        self.assertEqual(tuple(mask.shape), (1, 1, 2, 17))
        self.assertEqual(int(mask[0, 0, 0].sum()), 1)
        self.assertEqual(int(mask[0, 0, 1].sum()), 2)

    def test_stale_slots_are_never_admitted(self):
        # The whole point: slots beyond valid_len hold the previous frame's
        # keys, and admitting them produces plausible wrong audio.
        for valid in range(1, 17):
            mask = decode_mask(valid_len=valid, cache_len=17, query_len=1)
            self.assertEqual(int(mask[0, 0, 0, valid:].sum()), 0)

    def test_overflowing_the_cache_refuses(self):
        with self.assertRaises(ValueError):
            decode_mask(valid_len=18, cache_len=17, query_len=1)

    def test_query_longer_than_the_written_span_refuses(self):
        with self.assertRaises(ValueError):
            decode_mask(valid_len=1, cache_len=17, query_len=2)


class TestGraphSafetyRefusals(CustomTestCase):
    def test_a_tensor_write_position_is_accepted(self):
        refuse_unless_graph_safe(_FakeCache([_FakeLayer(), _FakeLayer()]))

    def test_an_int_write_position_refuses(self):
        # THE CAN-FAIL PROOF for the assumption the whole slice rests on: if
        # cumulative_length were a Python int, capture would bake slot 0 into
        # every replay and the cache would silently hold one entry.
        cache = _FakeCache([_FakeLayer(), _FakeLayer(position=0)])
        with self.assertRaises(GraphCaptureRefusal) as caught:
            refuse_unless_graph_safe(cache)
        self.assertIn("layer 1", str(caught.exception))

    def test_a_sliding_window_layer_refuses(self):
        layer = _FakeLayer()
        layer.cumulative_length_int = 0
        with self.assertRaises(GraphCaptureRefusal) as caught:
            refuse_unless_graph_safe(_FakeCache([layer]))
        self.assertIn("sliding window", str(caught.exception).lower())

    def test_a_cache_without_layers_refuses(self):
        with self.assertRaises(GraphCaptureRefusal):
            refuse_unless_graph_safe(_FakeCache([]))

    def test_reset_rewinds_every_layer_in_place(self):
        layers = [_FakeLayer(), _FakeLayer()]
        for layer in layers:
            layer.cumulative_length.fill_(7)
        originals = [layer.cumulative_length for layer in layers]
        reset_cache_positions(_FakeCache(layers))
        for layer, original in zip(layers, originals):
            self.assertEqual(int(layer.cumulative_length), 0)
            # In place, not rebound: the graph captured this exact address.
            self.assertIs(layer.cumulative_length, original)


class TestOrderedReplay(CustomTestCase):
    """Fifteen graphs share one memory pool, and the price is replay order."""

    def test_in_order_replay_passes_and_wraps(self):
        steps = [_FakeStep(f"s{i}") for i in range(3)]
        order = _OrderedReplay(steps)
        for _ in range(2):  # two frames
            for index in range(3):
                order.replay(index)
        self.assertEqual([s.replays for s in steps], [2, 2, 2])

    def test_out_of_order_replay_refuses(self):
        # THE CAN-FAIL PROOF: replaying a shared-pool graph out of order reads
        # another graph's live intermediates. It does not fault -- it returns
        # numbers -- so this must be an assertion, not a comment.
        steps = [_FakeStep(f"s{i}") for i in range(3)]
        order = _OrderedReplay(steps)
        order.replay(0)
        with self.assertRaises(GraphCaptureRefusal) as caught:
            order.replay(2)
        self.assertIn("capture order", str(caught.exception))

    def test_rewind_restarts_the_frame(self):
        steps = [_FakeStep(f"s{i}") for i in range(3)]
        order = _OrderedReplay(steps)
        order.replay(0)
        order.rewind()
        order.replay(0)
        self.assertEqual(steps[0].replays, 2)


class TestReplayBeforeCapture(CustomTestCase):
    def test_generate_without_capture_refuses(self):
        """A driver with no graphs must not fall through to the eager path.

        It would work, and it would report the eager path's timings as the
        graphed arm's -- the one failure that corrupts a measurement instead of
        the audio.
        """
        from sglang.srt.models.qwen3_tts_graph_driver import GraphedPredictorFrame

        driver = GraphedPredictorFrame.__new__(GraphedPredictorFrame)
        driver.steps = []
        with self.assertRaises(GraphCaptureRefusal) as caught:
            GraphedPredictorFrame.generate(driver, torch.zeros(1, 2, 4))
        self.assertIn("before capture", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
