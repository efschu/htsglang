"""P prefill graph with several buckets (27B line, xsn434, --p-prefill-graph-tiny
16): every captured bucket owns its GDN graph metadata.

xsn434 captured [16, 512] largest-first and died in the 16-token bucket's
capture warmup (PP1 and PP2, first GDN layer). The static GDN metadata of the
full prefill graph -- ``query_start_loc`` / state indices, and FLA's chunk
tables PINNED to that query_start_loc object -- was keyed by the request-slot
count only, so both buckets shared ONE pair: the tables stayed the 512
bucket's (8 chunks) for the 16-token graph. Now the pair is keyed by
(slots, bucket): ``capture_one_shape`` and ``_prepare_forward_metadata_for_
replay`` name the bucket (``prefill_graph_bucket``), and each bucket's capture
pins its own tensor, so its tables are computed from its own capture content.

The kernel-level two-bucket proof (Triton interpreter, [16, 512] in sequence)
is test_p_prefill_graph_two_buckets_0924.py.
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.fla import index as fidx
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import prefill_cuda_graph_runner as pcgr
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CPU = torch.device("cpu")


class _Pool:
    mapping = torch.tensor([7, 3, 5, 9], dtype=torch.int32)

    def get_mamba_indices(self, rpi):
        return self.mapping[rpi.long()]

    def translate_mamba_indices(self, x):
        return x


def _gdn_backend():
    be = object.__new__(MambaAttnBackendBase)
    be.device = CPU
    be.pad_slot_id = -1
    be.replayssm_write_pos_list = None
    be._extend_graph_static = {}
    be.req_to_token_pool = _Pool()
    return be


def _view(lens, rpi, bucket):
    lens_t = torch.tensor(lens, dtype=torch.int64)
    real = sum(lens)
    starts, acc = [], 0
    for n in lens:
        starts.append(acc if n > 0 else real)
        acc += n
    return types.SimpleNamespace(
        batch_size=len(lens),
        forward_mode=ForwardMode.EXTEND,
        extend_start_loc=torch.tensor(starts, dtype=torch.int64),
        extend_seq_lens=lens_t,
        req_pool_indices=torch.tensor(rpi, dtype=torch.int64),
        seq_lens_cpu=lens_t.clone(),
        mamba_track_mask=None,
        spec_info=None,
        prefill_graph_bucket=bucket,
    )


def _capture(be, lens, rpi, bucket):
    be.init_forward_metadata_out_graph(_view(lens, rpi, bucket), in_capture=True)
    return be.forward_metadata.query_start_loc, be.forward_metadata.mamba_cache_indices


class TestEveryBucketOwnsItsGdnMetadata(CustomTestCase):
    def test_largest_first_capture_gives_each_bucket_its_own_pair(self):
        be = _gdn_backend()
        qsl_big, idx_big = _capture(be, [512], [2], 512)
        # the big bucket's first use computes ITS tables (8 chunks of 64)
        big_ci = fidx.prepare_chunk_indices(qsl_big, 64)
        qsl_tiny, idx_tiny = _capture(be, [16], [1], 16)
        # red on the per-slot key: the tiny bucket got the SAME objects
        self.assertIsNot(qsl_tiny, qsl_big)
        self.assertIsNot(idx_tiny, idx_big)
        tiny_ci = fidx.prepare_chunk_indices(qsl_tiny, 64)
        self.assertEqual(int(big_ci.shape[0]), 8)
        self.assertEqual(int(tiny_ci.shape[0]), 1)
        self.assertEqual(tiny_ci.tolist(), [[0, 0]])
        self.assertEqual(fidx.prepare_chunk_offsets(qsl_tiny, 64).tolist(), [0, 1])
        self.assertEqual(fidx.prepare_chunk_offsets(qsl_big, 64).tolist(), [0, 8])
        # both pinned, each to its own object
        self.assertIn(("chunk_indices", 64), fidx.graph_static_tables(qsl_big))
        self.assertIn(("chunk_indices", 64), fidx.graph_static_tables(qsl_tiny))
        # the tiny capture left the big bucket's buffers alone
        self.assertEqual(qsl_big.tolist(), [0, 512])
        self.assertEqual(idx_big.tolist(), [5])

    def test_replays_refresh_the_replayed_buckets_objects_only(self):
        be = _gdn_backend()
        qsl_big, idx_big = _capture(be, [512], [2], 512)
        big_ci = fidx.prepare_chunk_indices(qsl_big, 64)
        qsl_tiny, idx_tiny = _capture(be, [16], [1], 16)
        tiny_ci = fidx.prepare_chunk_indices(qsl_tiny, 64)

        be.init_forward_metadata_out_graph(_view([1], [3], 16))
        self.assertIs(be.forward_metadata.query_start_loc, qsl_tiny)
        self.assertIs(be.forward_metadata.mamba_cache_indices, idx_tiny)
        self.assertEqual(qsl_tiny.tolist(), [0, 1])
        self.assertEqual(idx_tiny.tolist(), [9])
        self.assertEqual(qsl_big.tolist(), [0, 512])  # untouched

        be.init_forward_metadata_out_graph(_view([300], [2], 512))
        self.assertIs(be.forward_metadata.query_start_loc, qsl_big)
        self.assertEqual(qsl_big.tolist(), [0, 300])
        self.assertEqual(qsl_tiny.tolist(), [0, 1])  # untouched
        # the pinned tables are each bucket's capture tables, forever
        self.assertIs(fidx.prepare_chunk_indices(qsl_big, 64), big_ci)
        self.assertIs(fidx.prepare_chunk_indices(qsl_tiny, 64), tiny_ci)

    def test_a_caller_without_a_bucket_keeps_the_per_slot_key(self):
        be = _gdn_backend()
        qa, _ = _capture(be, [64], [1], None)
        qb, _ = _capture(be, [32], [1], None)
        self.assertIs(qa, qb)
        self.assertEqual(sorted(be._extend_graph_static), [(1, None)])


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def init_forward_metadata_out_graph(self, fb, in_capture=False):
        self.calls.append((getattr(fb, "prefill_graph_bucket", "UNSET"), in_capture))


class TestRunnerNamesTheBucket(CustomTestCase):
    def test_capture_names_the_bucket_before_the_metadata(self):
        be = _RecordingBackend()
        seen = {}

        class _Backend:
            def capture_one(self, key, run_once, dummies=None, post_warmup_hook=None):
                seen["key"] = key

        fake = types.SimpleNamespace(
            capture_prepare=lambda n: (types.SimpleNamespace(), be),
            _is_full_backend=True,
            backend=_Backend(),
            _run_forward=lambda fb, n: None,
        )
        pcgr.PrefillCudaGraphRunner.capture_one_shape(fake, 16)
        pcgr.PrefillCudaGraphRunner.capture_one_shape(fake, 512)
        self.assertEqual(be.calls, [(16, True), (512, True)])

    def test_replay_names_the_padded_bucket_not_the_live_count(self):
        be = _RecordingBackend()
        r = 1
        s = {
            k: torch.zeros(4, dtype=torch.int64)
            for k in ("seq_lens", "req_pool_indices", "extend_seq_lens",
                      "extend_prefix_lens", "extend_start_loc")
        }
        fake = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(attn_backend=be),
            _is_full_backend=True,
            _capture_req_slots=r,
            _prefill_static_buffers=s,
            _full_cg_seq_lens_cpu=torch.zeros(r, dtype=torch.int64),
        )
        live = types.SimpleNamespace(
            batch_size=1, seq_lens_cpu=torch.tensor([1]), positions=torch.zeros(1)
        )
        pcgr.PrefillCudaGraphRunner._prepare_forward_metadata_for_replay(
            fake, live, None, 16
        )
        self.assertEqual(be.calls, [(16, False)])
        # the live batch itself is not marked (the view is a copy)
        self.assertFalse(hasattr(live, "prefill_graph_bucket"))


if __name__ == "__main__":
    unittest.main()
