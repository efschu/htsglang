"""CudaGraphBufferRegistry fill contract + SGLANG_POISON_GRAPH_PAD falsifier.

#50 campaign round 8: the eager runner's registry buffers showed changed
hashes between requests. Code audit: eager never pads (load_batch passes
padded == raw) and the forward consumes [:raw] slices, so the changed bytes
are dead tails — UNLESS some kernel reads beyond the active region. The
poison lever tests that claim empirically on the GPU; these tests pin the
lever's semantics and the fill contract itself on CPU.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    GraphSlot,
    PaddingPolicy,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

MAX_BS = 4
MAX_TOKENS = 16


class _FB:
    """Minimal ForwardBatch stand-in: fill_from only getattr()s slot names."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _build_registry() -> CudaGraphBufferRegistry:
    reg = CudaGraphBufferRegistry(
        device=torch.device("cpu"), max_bs=MAX_BS, max_num_tokens=MAX_TOKENS
    )
    reg.register_slot(
        GraphSlot("input_ids", lambda _b, mt: (mt,), torch.int64, axis="tokens")
    )  # FOREACH_COPY (default)
    reg.register_slot(
        GraphSlot(
            "positions",
            lambda _b, mt: (mt,),
            torch.int64,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
        )
    )
    reg.register_slot(
        GraphSlot(
            "seq_lens",
            lambda b, _mt: (b,),
            torch.int64,
            axis="bs",
            padding_policy=PaddingPolicy.FILL_SENTINEL,
            pad_value=7,
        )
    )
    reg.register_slot(
        GraphSlot(
            "hidden_probe",
            lambda _b, mt: (mt,),
            torch.float32,
            axis="tokens",
        )
    )
    return reg


def _fill(reg, raw_bs, padded_bs, raw_tokens, padded_tokens):
    fb = _FB(
        input_ids=torch.arange(raw_tokens, dtype=torch.int64),
        positions=torch.arange(raw_tokens, dtype=torch.int64) + 50,
        seq_lens=torch.full((raw_bs,), 3, dtype=torch.int64),
        hidden_probe=torch.ones(raw_tokens, dtype=torch.float32),
    )
    reg.fill_from(
        fb,
        raw_bs=raw_bs,
        padded_bs=padded_bs,
        raw_num_tokens=raw_tokens,
        padded_num_tokens=padded_tokens,
    )
    return fb


class TestFillContract(CustomTestCase):
    def test_eager_unpadded_foreach_copy_keeps_stale_tail(self):
        # Pin the pre-existing behavior the round-8 dump surfaced: with
        # padded == raw (eager), FOREACH_COPY tails keep previous content —
        # dead bytes by contract, exercised by the poison falsifier below.
        reg = _build_registry()
        reg.get_slot("input_ids").buffer.fill_(99)
        _fill(reg, raw_bs=2, padded_bs=2, raw_tokens=4, padded_tokens=4)
        buf = reg.get_slot("input_ids").buffer
        self.assertEqual(buf[:4].tolist(), [0, 1, 2, 3])  # head copied
        self.assertTrue(torch.all(buf[4:] == 99))  # stale tail kept

    def test_graph_padded_semantic_resets(self):
        reg = _build_registry()
        reg.get_slot("positions").buffer.fill_(99)
        reg.get_slot("seq_lens").buffer.fill_(99)
        _fill(reg, raw_bs=2, padded_bs=4, raw_tokens=4, padded_tokens=8)
        pos = reg.get_slot("positions").buffer
        self.assertEqual(pos[:4].tolist(), [50, 51, 52, 53])
        self.assertTrue(torch.all(pos[4:8] == 0))  # ZERO pad window
        sl = reg.get_slot("seq_lens").buffer
        self.assertEqual(sl[:2].tolist(), [3, 3])
        self.assertEqual(sl[2:4].tolist(), [7, 7])  # sentinel pad window


class TestPoisonGraphPad(CustomTestCase):
    def test_poison_covers_inactive_regions_and_respects_semantics(self):
        reg = _build_registry()
        with envs.SGLANG_POISON_GRAPH_PAD.override(True):
            _fill(reg, raw_bs=2, padded_bs=4, raw_tokens=4, padded_tokens=8)
        ids = reg.get_slot("input_ids").buffer
        self.assertEqual(ids[:4].tolist(), [0, 1, 2, 3])  # head intact
        self.assertTrue(torch.all(ids[4:] == 100))  # int poison
        pos = reg.get_slot("positions").buffer
        self.assertTrue(torch.all(pos[4:8] == 0))  # semantic ZERO wins
        self.assertTrue(torch.all(pos[8:] == 100))  # beyond padded: poison
        sl = reg.get_slot("seq_lens").buffer
        self.assertEqual(sl[2:4].tolist(), [7, 7])  # sentinel wins
        self.assertTrue(torch.all(sl[4:] == 100))
        probe = reg.get_slot("hidden_probe").buffer
        self.assertTrue(torch.all(probe[:4] == 1.0))
        self.assertTrue(torch.isnan(probe[4:]).all())  # float poison

    def test_poison_is_deterministic_wrt_history(self):
        # Two registries with different previous content must be bitwise
        # identical after a poisoned fill (the falsifier itself must not
        # carry history).
        bufs = []
        for stale in (11, 77):
            reg = _build_registry()
            for name in ("input_ids", "positions", "seq_lens"):
                reg.get_slot(name).buffer.fill_(stale)
            reg.get_slot("hidden_probe").buffer.fill_(float(stale))
            with envs.SGLANG_POISON_GRAPH_PAD.override(True):
                _fill(reg, raw_bs=2, padded_bs=2, raw_tokens=4, padded_tokens=4)
            bufs.append(
                [reg.get_slot(n).buffer.clone() for n in reg.slot_names()]
            )
        for a, b in zip(*bufs):
            self.assertTrue(torch.equal(a.nan_to_num(-1), b.nan_to_num(-1)))

    def test_off_by_default_no_behavior_change(self):
        reg = _build_registry()
        reg.get_slot("hidden_probe").buffer.fill_(5.0)
        _fill(reg, raw_bs=2, padded_bs=2, raw_tokens=4, padded_tokens=4)
        probe = reg.get_slot("hidden_probe").buffer
        self.assertTrue(torch.all(probe[4:] == 5.0))  # untouched without env


if __name__ == "__main__":
    unittest.main()
