"""Unit tests for T156 stage 3: per-batch cross-algorithm switching.

CPU-only tests of the pure switching mechanics: force-string parsing, the
atomic runtime shape swap, the round-counter schedule decision, the draft-input
conversion at switch boundaries, and the FutureMap's None-tolerant stash under
mixed-algorithm payloads. GPU behavior (graph swap, warm-keeping forwards) is
covered by the live validation protocol, not here.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.cross_algo_utils import (
    RUNTIME_SHAPE_FIELDS,
    apply_runtime_shape,
    parse_cross_force,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestParseCrossForce(CustomTestCase):
    def test_static_modes(self):
        self.assertEqual(parse_cross_force("nextn"), ("nextn", None))
        self.assertEqual(parse_cross_force("dflash"), ("dflash", None))

    def test_schedule_mode(self):
        self.assertEqual(parse_cross_force("schedule:64"), ("schedule", 64))
        self.assertEqual(parse_cross_force("schedule:1"), ("schedule", 1))

    def test_invalid_values_raise(self):
        for bad in [None, "", "foo", "schedule", "schedule:", "schedule:0",
                    "schedule:-3", "schedule:x", "SCHEDULE:4"]:
            with self.subTest(force=bad):
                with self.assertRaises(ValueError):
                    parse_cross_force(bad)


class _FakeArgs:
    """Records override() calls like the audited ServerArgs mutation point."""

    def __init__(self):
        self.calls = []
        for f in RUNTIME_SHAPE_FIELDS:
            setattr(self, f, None)

    def override(self, reason, **kwargs):
        self.calls.append((reason, dict(kwargs)))
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestRuntimeShapeSwap(CustomTestCase):
    def _shape(self, algo):
        if algo == "DFLASH":
            return {
                "speculative_num_steps": 1,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 16,
                "speculative_dflash_block_size": 16,
            }
        return {
            "speculative_num_steps": 3,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
            "speculative_dflash_block_size": None,
        }

    def test_swap_is_one_override_call_covering_all_fields(self):
        args = _FakeArgs()
        apply_runtime_shape(args, self._shape("DFLASH"), "test.swap")
        # Atomicity: a single override call carries EVERY runtime shape field
        # (no reader can observe a half-swapped shape between two calls).
        self.assertEqual(len(args.calls), 1)
        reason, kwargs = args.calls[0]
        self.assertEqual(reason, "test.swap")
        self.assertEqual(set(kwargs), set(RUNTIME_SHAPE_FIELDS))
        self.assertEqual(args.speculative_num_draft_tokens, 16)
        self.assertEqual(args.speculative_dflash_block_size, 16)

    def test_swap_round_trip_restores_original_values(self):
        args = _FakeArgs()
        apply_runtime_shape(args, self._shape("EAGLE"), "boot")
        before = {f: getattr(args, f) for f in RUNTIME_SHAPE_FIELDS}
        apply_runtime_shape(args, self._shape("DFLASH"), "to_dflash")
        apply_runtime_shape(args, self._shape("EAGLE"), "back")
        after = {f: getattr(args, f) for f in RUNTIME_SHAPE_FIELDS}
        self.assertEqual(before, after)


def _bare_worker():
    """A CrossAlgoWorker shell with only the schedule-decision state set."""
    from sglang.srt.speculative.cross_algo_worker import CrossAlgoWorker

    w = object.__new__(CrossAlgoWorker)
    w._schedule_interval = 4
    w._active_name = "nextn"
    w._rounds_total = 0
    w._rounds_at_switch = 0
    w._switch_count = 0
    w._forced_algo = SpeculativeAlgorithm.EAGLE
    w._secondary_algo = SpeculativeAlgorithm.DFLASH
    return w


class TestScheduleDecision(CustomTestCase):
    def test_switches_exactly_every_interval(self):
        w = _bare_worker()
        switches = []

        def fake_switch(batch):
            switches.append(w._rounds_total)
            w._active_name = "dflash" if w._active_name == "nextn" else "nextn"
            w._rounds_at_switch = w._rounds_total

        w._switch_rung = fake_switch
        batch = SimpleNamespace(spec_algorithm=None)
        for _ in range(20):
            w.maybe_switch_rung(batch)  # scheduler hook (pre-round)
            w._rounds_total += 1  # worker forward (the round runs)
        self.assertEqual(switches, [4, 8, 12, 16])

    def test_restamps_active_algorithm_every_call(self):
        w = _bare_worker()
        w._switch_rung = lambda batch: None
        batch = SimpleNamespace(spec_algorithm=None)
        w.maybe_switch_rung(batch)
        self.assertEqual(batch.spec_algorithm, SpeculativeAlgorithm.EAGLE)
        w._active_name = "dflash"
        w.maybe_switch_rung(batch)
        self.assertEqual(batch.spec_algorithm, SpeculativeAlgorithm.DFLASH)

    def test_static_force_mode_is_inert(self):
        w = _bare_worker()
        w._schedule_interval = None
        w._rounds_total = 100

        def boom(batch):
            raise AssertionError("must not switch in static force mode")

        w._switch_rung = boom
        batch = SimpleNamespace(spec_algorithm="sentinel")
        w.maybe_switch_rung(batch)
        # Static mode: no switch AND no restamp (stage-2 path untouched).
        self.assertEqual(batch.spec_algorithm, "sentinel")


class TestConvertSpecInfo(CustomTestCase):
    def _batch(self, bs=2):
        return SimpleNamespace(
            spec_info=None,
            seq_lens=torch.tensor([7, 9], dtype=torch.int64),
            batch_size=lambda: bs,
        )

    def test_eagle_to_dflash_and_back_carries_relay_identity(self):
        from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        w = _bare_worker()
        batch = self._batch()
        fi = torch.tensor([3, 5], dtype=torch.int64)
        eagle = EagleDraftInput(
            topk_p=torch.zeros((2, 1)),
            topk_index=torch.zeros((2, 1), dtype=torch.int64),
            hidden_states=torch.zeros((2, 8)),
            bonus_tokens=torch.tensor([11, 12]),
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        eagle.future_indices = fi
        batch.spec_info = eagle

        w._convert_spec_info(batch, "dflash")
        self.assertIsInstance(batch.spec_info, DFlashDraftInputV2)
        self.assertIs(batch.spec_info.future_indices, fi)
        self.assertIs(batch.spec_info.bonus_tokens, eagle.bonus_tokens)

        w._convert_spec_info(batch, "nextn")
        self.assertIsInstance(batch.spec_info, EagleDraftInput)
        self.assertIs(batch.spec_info.future_indices, fi)
        self.assertIs(batch.spec_info.topk_p, eagle.topk_p)

    def test_wrong_boundary_type_asserts(self):
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        w = _bare_worker()
        batch = self._batch()
        batch.spec_info = EagleDraftInput(bonus_tokens=torch.tensor([1, 2]))
        batch.spec_info.future_indices = torch.tensor([0, 1])
        with self.assertRaises(AssertionError):
            w._convert_spec_info(batch, "nextn")  # already eagle-typed

    def test_none_spec_info_is_a_no_op(self):
        w = _bare_worker()
        batch = self._batch()
        batch.spec_info = None
        w._convert_spec_info(batch, "dflash")
        self.assertIsNone(batch.spec_info)

    def test_batch_size_mismatch_asserts(self):
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        w = _bare_worker()
        batch = self._batch(bs=3)
        si = EagleDraftInput(bonus_tokens=torch.tensor([1, 2]))
        si.future_indices = torch.tensor([0, 1])  # 2 != bs 3
        batch.spec_info = si
        with self.assertRaises(AssertionError):
            w._convert_spec_info(batch, "dflash")


class TestFutureMapMixedPayloadStash(CustomTestCase):
    """The relay must tolerate bonus-only payloads (cross-algo rounds) under a
    need_topk-typed map, allocating extras bufs on the first payload that has
    them."""

    def _future_map(self):
        from sglang.srt.managers.overlap_utils import FutureMap
        from sglang.srt.runtime_context import get_context
        from sglang.srt.server_args import ServerArgs

        # spec_need_hidden_states (lazy buf init) reads the global server
        # args; provide an EAGLE-shaped one (model_path="dummy"
        # short-circuits __post_init__, same pattern as the overlap tests).
        args = ServerArgs(model_path="dummy")
        args.speculative_algorithm = "EAGLE"
        get_context().set_server_args(args)
        self.addCleanup(lambda: get_context().set_server_args(None))

        pool = SimpleNamespace(req_to_token=torch.zeros((8, 4), dtype=torch.int64))
        return FutureMap(
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.EAGLE,
            req_to_token_pool=pool,
            needs_cpu_seq_lens=True,
        )

    def test_bonus_only_then_full_payload(self):
        from sglang.srt.managers.overlap_utils import RelayPayload

        fm = self._future_map()
        idx = torch.tensor([1, 2], dtype=torch.int64)
        # Round 1: bonus-only (e.g. a DFLASH round without attached seeds).
        fm.stash(idx, RelayPayload(bonus_tokens=torch.tensor([5, 6])))
        self.assertIsNone(getattr(fm, "topk_p_buf", None))
        # Round 2: full eagle payload -> extras bufs come up lazily.
        fm.stash(
            idx,
            RelayPayload(
                bonus_tokens=torch.tensor([7, 8]),
                topk_p=torch.ones((2, 1)),
                topk_index=torch.zeros((2, 1), dtype=torch.int64),
                hidden_states=torch.zeros((2, 8)),
            ),
        )
        self.assertIsNotNone(fm.topk_p_buf)
        self.assertEqual(int(fm.output_tokens_buf[1]), 7)
        # Round 3: bonus-only again -> extras keep their previous values.
        fm.stash(idx, RelayPayload(bonus_tokens=torch.tensor([9, 10])))
        self.assertEqual(int(fm.output_tokens_buf[2]), 10)
        self.assertEqual(float(fm.topk_p_buf[1, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
