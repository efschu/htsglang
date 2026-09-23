# SPDX-License-Identifier: Apache-2.0
"""#1243 SLICE 2e-II -- ``--kv-tail-headroom``: the ring takes the KV pool the
admission bound can never use.

Basis 2.2 / 2.8 / 7.7: above the guaranteed minimum the 16-bit tail grows into
FREE KV headroom and costs no guaranteed pool. On the Next Flash D form the
pool is far larger than any admitted request can fill (fnFL2x25, TP0:
596864 tokens at --max-running-requests 1 and a 262144 context), so about
335k tokens of fp8 KV sit unusable by construction. ``--kv-tail-headroom``
sizes the ring from exactly that remainder: the body pool keeps
``max_running_requests x per-request cap`` (page-rounded), the rest becomes
bf16 ring rows, and the elastic window (slice 2e) lets the running requests
grow into it. Pinned here:

* the sizing arithmetic on the Next Flash cell, the pool keeping at least the
  admission bound after every post (ring + mapping);
* --max-kv-per-request lowers the bound and raises the ring;
* the refusals by name: the paged (FlashInfer) reader, whose window is fixed
  at the minimum, and weighted DCP, whose token-share sizing is not built;
* a rank without KV (Form A expert worker) posts nothing;
* the knob turns the tail on by itself, and the mixin installs the ring with
  the SIZED rows, not a second auto derivation.
"""

import types
import unittest

from sglang.srt.mem_cache.kv_tail import (
    KvTailKnobs,
    Weg2KvTailFormRefused,
    Weg2KvTailUnfundable,
)
from sglang.test.test_utils import CustomTestCase

import sglang.srt.model_executor.pool_configurator as pc

#: Next Flash on TP0: 12 full-attention layers x 2 kv heads x 256 x (K+V) at
#: fp8, plus the QSA index keys (4 x 128 x 2 B / ratio 4 x 12 layers).
NF_KV_CELL = 12288
NF_QSA_CELL = 768
NF_CELL = NF_KV_CELL + NF_QSA_CELL


def _qsa_hf_config():
    return types.SimpleNamespace(
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )


def _cfg(qsa=True, context=262144, mrr=1, max_kv=None, **sa_kw):
    cfg = pc.DefaultPoolConfigurator.__new__(pc.DefaultPoolConfigurator)
    cfg._cell_size = NF_CELL if qsa else NF_KV_CELL
    cfg._kv_tail_target_cell_size = cfg._cell_size
    cfg._kv_tail_qsa_cell = NF_QSA_CELL if qsa else 0
    cfg._kv_tail_body_itemsize = 1
    sa = types.SimpleNamespace(
        kv_tail_min_tokens=16384,
        kv_tail_max_tokens=-1,
        kv_tail_ring_rows=None,
        kv_tail_host_max_tokens=None,
        kv_tail_headroom=True,
        max_running_requests=mrr,
        max_kv_per_request=max_kv,
        kv_cache_dtype="fp8_e4m3",
    )
    sa.__dict__.update(sa_kw)
    cfg._kv_tail_mr = types.SimpleNamespace(
        server_args=sa,
        model_config=types.SimpleNamespace(
            context_len=context,
            hf_config=_qsa_hf_config() if qsa else types.SimpleNamespace(),
        ),
    )
    return cfg


class _Parallel:
    def __init__(self, dcp_size=1, dcp_rank=0):
        self.p = types.SimpleNamespace(attn_dcp_size=dcp_size, attn_dcp_rank=dcp_rank)

    def __enter__(self):
        self.real = pc.get_parallel
        pc.get_parallel = lambda: self.p

    def __exit__(self, *a):
        pc.get_parallel = self.real


FNFL2X25_TP0_BYTES = 8441872384  # "KV pool sizing: available_bytes=8441872384" (boot fnFL2x25)


class TestTheHeadroomRingSizing(CustomTestCase):
    def test_the_ring_takes_what_the_admission_bound_cannot_use(self):
        cfg = _cfg()
        with _Parallel():
            post, terms = cfg._kv_tail_ring_post(page_size=64, available_bytes=FNFL2X25_TP0_BYTES)
        total = FNFL2X25_TP0_BYTES // NF_CELL
        self.assertTrue(terms["headroom"])
        self.assertEqual(terms["need_tokens"], 262144)
        self.assertGreater(terms["rows"], 150000)  # ~167k bf16 rows on this form
        # the pool keeps at least the admission bound after the ring post AND
        # the mapping (the second correction pass of calculate_pool_sizes)
        left = FNFL2X25_TP0_BYTES - post - cfg._kv_tail_map_bytes(total, 64)
        self.assertGreaterEqual(left // NF_CELL // 64 * 64, 262144)

    def test_a_lower_per_request_cap_leaves_more_for_the_ring(self):
        with _Parallel():
            _p1, t1 = _cfg()._kv_tail_ring_post(page_size=64, available_bytes=FNFL2X25_TP0_BYTES)
            _p2, t2 = _cfg(max_kv=131072)._kv_tail_ring_post(
                page_size=64, available_bytes=FNFL2X25_TP0_BYTES
            )
        self.assertEqual(t2["need_tokens"], 131072)
        self.assertGreater(t2["rows"], t1["rows"])

    def test_no_headroom_means_the_guaranteed_minimum_only(self):
        # a pool that the admission bound fills exactly: the ring is the auto minimum
        avail = 262144 * NF_CELL
        with _Parallel():
            _post, terms = _cfg()._kv_tail_ring_post(page_size=64, available_bytes=avail)
        self.assertEqual(terms["rows"], 16384)
        self.assertEqual(terms["spare_tokens"], 0)

    def test_the_minimum_ring_is_kept_when_the_headroom_is_smaller(self):
        avail = (262144 + 1000) * NF_CELL
        with _Parallel():
            _post, terms = _cfg()._kv_tail_ring_post(page_size=64, available_bytes=avail)
        self.assertEqual(terms["rows"], 16384)

    def test_the_sized_rows_are_handed_to_the_installer(self):
        cfg = _cfg()
        with _Parallel():
            _post, terms = cfg._kv_tail_ring_post(page_size=64, available_bytes=FNFL2X25_TP0_BYTES)
        self.assertEqual(cfg._kv_tail_mr._kv_tail_ring_rows_sized, terms["rows"])


class TestTheHeadroomRefusals(CustomTestCase):
    def test_the_paged_reader_is_refused_by_name(self):
        with _Parallel(), self.assertRaises(Weg2KvTailFormRefused) as cm:
            _cfg(qsa=False)._kv_tail_ring_post(page_size=1, available_bytes=FNFL2X25_TP0_BYTES)
        self.assertIn("--kv-tail-headroom", str(cm.exception))

    def test_weighted_dcp_is_refused_by_name(self):
        with _Parallel(dcp_size=3, dcp_rank=0), self.assertRaises(Weg2KvTailUnfundable) as cm:
            _cfg()._kv_tail_ring_post(page_size=64, available_bytes=FNFL2X25_TP0_BYTES)
        self.assertIn("--kv-tail-headroom", str(cm.exception))

    def test_a_rank_without_kv_posts_nothing(self):
        cfg = _cfg()
        cfg._cell_size = NF_QSA_CELL  # Form A expert worker: the QSA index only
        cfg._kv_tail_target_cell_size = NF_QSA_CELL
        with _Parallel():
            post, _terms = cfg._kv_tail_ring_post(page_size=64, available_bytes=1_500_000_000)
        self.assertEqual(post, 0)


class TestTheInstallerUsesTheSizedRows(CustomTestCase):
    def test_the_ring_is_built_with_the_rows_the_post_charged(self):
        import sglang.srt.model_executor.model_runner_kv_cache_mixin as mixin
        from sglang.srt.mem_cache import memory_pool

        pool = memory_pool.MHATokenToKVPool(
            size=256, page_size=4, dtype=__import__("torch").float8_e4m3fn, head_num=2,
            head_dim=8, layer_num=2, device="cpu", enable_memory_saver=False,
            enable_alt_stream=False,
        )
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(
                kv_tail_min_tokens=16, kv_tail_max_tokens=-1, kv_tail_ring_rows=None,
                kv_tail_host_max_tokens=None, kv_tail_shrink_hysteresis_rounds=None,
                kv_tail_virtual_fp8=False, kv_tail_sidecar=False, kv_tail_draft=False,
                kv_tail_headroom=True, max_running_requests=1, enable_memory_saver=False,
            ),
            token_to_kv_pool=pool,
            is_draft_worker=False,
            is_draft_pool_worker=False,
            model_config=types.SimpleNamespace(hf_config=_qsa_hf_config()),
            _kv_tail_ring_rows_sized=96,
        )
        real = mixin.get_parallel
        mixin.get_parallel = lambda: types.SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0)
        try:
            ring = mixin.ModelRunnerKVCacheMixin._install_kv_tail_ring(runner)
        finally:
            mixin.get_parallel = real
        self.assertIsNotNone(ring)
        self.assertEqual(ring.ring_rows, 96)
        self.assertEqual(ring.reader, "rows")


class TestTheHeadroomParseGate(CustomTestCase):
    def _sa(self, **kw):
        from sglang.srt.server_args import ServerArgs

        sa = ServerArgs.__new__(ServerArgs)
        for k, v in dict(
            kv_tail_min_tokens=16384, kv_tail_max_tokens=-1, kv_tail_ring_rows=None,
            kv_tail_host_max_tokens=None, kv_tail_shrink_hysteresis_rounds=None,
            kv_tail_virtual_fp8=False, kv_tail_sidecar=False, kv_tail_draft=False,
            kv_tail_headroom=True, page_size=1, disable_cuda_graph=False,
        ).items():
            setattr(sa, k, v)
        sa.__dict__.update(kw)
        return sa

    def test_headroom_on_a_paged_reader_is_refused_at_parse_time(self):
        sa = self._sa(page_size=1)  # the 27B form: page 1, FlashInfer
        sa._handle_kv_tail()
        with self.assertRaises(ValueError) as cm:
            sa._handle_kv_tail_page_form(hf_config=types.SimpleNamespace())
        self.assertIn("W142", str(cm.exception))
        self.assertIn("--kv-tail-headroom", str(cm.exception))

    def test_headroom_on_the_qsa_reader_parses(self):
        sa = self._sa(page_size=64)
        sa._handle_kv_tail()
        sa._handle_kv_tail_page_form(hf_config=_qsa_hf_config())


class TestTheKnob(CustomTestCase):
    def test_headroom_alone_turns_the_tail_on(self):
        self.assertTrue(KvTailKnobs(min_tokens=0, headroom=True).enabled)
        self.assertFalse(KvTailKnobs(min_tokens=0).enabled)

    def test_the_server_args_field_exists_and_defaults_off(self):
        from sglang.srt.server_args import ServerArgs

        self.assertIn("kv_tail_headroom", ServerArgs.__dataclass_fields__)
        self.assertFalse(ServerArgs.__dataclass_fields__["kv_tail_headroom"].default)


if __name__ == "__main__":
    unittest.main()
