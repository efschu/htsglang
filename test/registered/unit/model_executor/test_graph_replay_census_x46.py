"""23.09. (fnFL2x46): the first verify graph replay of a new request dies and
one after an eager round lives, so the replay's INPUTS are censused right
before the replay: static buffers, attention forward metadata, expert pool
tables. What must hold: off by default, only verify replays of real contexts
spend the budget, the census never walks into pools (a KV cache reduced to
min/max would stall the replay for seconds), and it ends with its denominator.
"""
import logging
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.runner import graph_replay_census as grc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-weg2-unit")


class _Mode:
    def __init__(self, verify):
        self._verify = verify

    def is_target_verify(self):
        return self._verify


class FakeKVPool:
    def __init__(self):
        self.k_buffer = torch.zeros(8)


class _Tables:
    def __init__(self, error):
        self.hot_phys = torch.tensor([0, -1, 3], dtype=torch.int32)
        self.host_row = torch.tensor([-1, 0, 1], dtype=torch.int32)
        self.row_key = torch.tensor([0, -1, 2, -1], dtype=torch.int32)
        self.row_use = torch.zeros(4, dtype=torch.int64)
        self.error = torch.tensor([error], dtype=torch.int32)
        self.staging_rows = torch.tensor([3], dtype=torch.int32)
        # resident row 0, LRU rows 1..2, staging row 3; row 2 names expert 2,
        # whose hot_phys points at the staging row: one #104 bijection break
        self.lru_start = 1
        self.pool_rows = 3


class _MoE(torch.nn.Module):
    def __init__(self, layer_id, error):
        super().__init__()
        self._expert_offload = SimpleNamespace(
            _pool_tables=_Tables(error), layer=SimpleNamespace(layer_id=layer_id)
        )


def _runner():
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([659], dtype=torch.int32),
        indexer=SimpleNamespace(page_table=torch.tensor([[3, 7]], dtype=torch.int32)),
        token_to_kv_pool=FakeKVPool(),
    )
    backend = SimpleNamespace(
        forward_metadata=None,
        full_attn_backend=SimpleNamespace(forward_metadata=metadata),
        linear_attn_backend=None,
    )
    model = torch.nn.Sequential(_MoE(0, 0), _MoE(1, 1))
    return SimpleNamespace(
        buffers=SimpleNamespace(seq_lens=torch.tensor([655, 0], dtype=torch.int32)),
        attn_backend=backend,
        model_runner=SimpleNamespace(model=model),
    )


def _batch(verify=True, ctx=655):
    return SimpleNamespace(
        forward_mode=_Mode(verify), seq_lens=torch.tensor([ctx]), batch_size=1
    )


class GraphReplayCensus(unittest.TestCase):
    def setUp(self):
        self._env = grc.os.environ.pop(grc.ENV, None)
        grc._STATE.update(budget=None, n=0)

    def tearDown(self):
        grc.os.environ.pop(grc.ENV, None)
        if self._env is not None:
            grc.os.environ[grc.ENV] = self._env
        grc._STATE.update(budget=None, n=0)

    def test_off_by_default(self):
        with self.assertNoLogs(grc.logger, level=logging.INFO):
            grc.maybe_census(_runner(), _batch())

    def test_only_verify_replays_of_real_contexts_spend_the_budget(self):
        grc.os.environ[grc.ENV] = "1"
        with self.assertNoLogs(grc.logger, level=logging.INFO):
            grc.maybe_census(_runner(), _batch(verify=False))
            grc.maybe_census(_runner(), _batch(ctx=1))
        with self.assertLogs(grc.logger, level=logging.INFO):
            grc.maybe_census(_runner(), _batch())
        self.assertEqual(grc._STATE["n"], 1)

    def test_slotted_and_struct_metadata_is_walked_not_skipped(self):
        # x47: the QSA forward metadata has no __dict__; the census printed
        # no attn.full line at all and "tensors=15" was the only trace of it.
        import msgspec

        class QsaMeta(msgspec.Struct):
            sequence_lengths: object = None

        runner = _runner()
        runner.attn_backend.full_attn_backend.forward_metadata = QsaMeta(
            sequence_lengths=torch.tensor([659], dtype=torch.int32)
        )
        grc.os.environ[grc.ENV] = "1"
        with self.assertLogs(grc.logger, level=logging.INFO) as cm:
            grc.maybe_census(runner, _batch())
        self.assertIn(
            "attn.full.forward_metadata.sequence_lengths shape=(1,) dtype=int32 range=659..659",
            "\n".join(cm.output),
        )

    def test_a_census_names_every_input_skips_pools_and_ends_with_its_denominator(self):
        grc.os.environ[grc.ENV] = "2"
        with self.assertLogs(grc.logger, level=logging.INFO) as cm:
            grc.maybe_census(_runner(), _batch())
        text = "\n".join(cm.output)
        self.assertIn("n=1 ctx=655 bs=1 buffers.seq_lens shape=(2,) dtype=int32 range=0..655", text)
        self.assertIn("attn.full.forward_metadata.indexer.page_table", text)
        self.assertIn("range=3..7", text)
        self.assertNotIn("k_buffer", text)
        self.assertIn("pool layers=2 error_layers=[1]", text)
        self.assertIn("pool.hot_phys min=-1 max=3", text)
        self.assertIn("pool.bijection_breaks total=2 layers=2", text)
        self.assertIn("done tensors=3", cm.output[-1])


if __name__ == "__main__":
    unittest.main()
