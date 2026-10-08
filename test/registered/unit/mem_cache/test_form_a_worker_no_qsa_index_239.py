# SPDX-License-Identifier: Apache-2.0
"""#239 S0 -- a Form A expert worker keeps no compressed QSA index.

rc12r (D-Log docker_dkrnfh91dprbar1dauer09271632, Z.3698-3701 / 7840): TP1
and TP2 sized a KV cell of 768 B/token -- 12 layers x 64 B of compressed
QSA index keys -- for 262144 tokens (192 MiB per 3080) and allocated a
0.27 GB 'qsa_indexer' HiCache host sidecar, although a Form A worker runs
no QSA indexer: its attention modules are host-only placeholders
(form_a_construction), the H101 prewarm already skips it ("attends
nothing"), and the HiCache assembler/cache controller were written for a
worker with an EMPTY compressed list ("no sidecar on that rank"). The
index was pure dead weight on the two cards whose expert rows are scarce.
"""

import inspect
import types

import pytest

from flliper.srt import rank_role
from flliper.srt.mem_cache import qsa_kv_pool
from flliper.srt.model_executor import pool_configurator
from flliper.srt.model_executor import model_runner_kv_cache_mixin as mixin

NF_TEXT = types.SimpleNamespace(
    indexer_n_heads=4,
    indexer_kv_heads=1,
    indexer_head_dim=128,
    indexer_compress_ratio=4,
    indexer_budget=2048,
)
NF_HF = types.SimpleNamespace(text_config=NF_TEXT)


@pytest.fixture
def plan():
    yield
    rank_role.set_form_a_role_plan(None, 0)


def _cell(num_layers=12):
    return pool_configurator.DefaultPoolConfigurator._compute_qsa_cell_size(
        hf_config=NF_HF, num_layers=num_layers
    )


def test_classic_and_host_ranks_keep_the_768_byte_index(plan):
    assert _cell() == 768  # the rc12r cell: 12 x 128 x bf16 / 4
    rank_role.set_form_a_role_plan(rank_role.RankRolePlan.parse("host,worker,worker"), 0)
    assert _cell() == 768


@pytest.mark.parametrize("rank", [1, 2])
def test_a_form_a_worker_prices_no_qsa_index(plan, rank):
    rank_role.set_form_a_role_plan(rank_role.RankRolePlan.parse("host,worker,worker"), rank)
    assert _cell() == 0


def test_the_pool_builds_no_compressed_index_when_told_so():
    sig = inspect.signature(qsa_kv_pool.QSATokenToKVPool.__init__)
    assert sig.parameters["qsa_index_on_rank"].default is True
    src = inspect.getsource(qsa_kv_pool.QSATokenToKVPool.__init__)
    skip = src.index("if not qsa_index_on_rank:")
    build = src.index("One contiguous allocation behind per-layer views")
    assert skip < build
    assert "self.qsa_compressed_k_buffer_pool = []" in src[skip:build]


def test_the_mixin_hands_the_role_to_the_pool():
    src = inspect.getsource(mixin)
    i = src.index("_kv_pool_class = QSATokenToKVPool")
    assert "qsa_index_on_rank=not this_rank_is_form_a_worker()" in src[i : i + 800]


def test_the_empty_list_is_the_shape_the_hicache_side_reads_as_no_sidecar():
    from flliper.srt.managers import cache_controller
    from flliper.srt.mem_cache.hybrid_cache import hybrid_pool_assembler

    assert "if not device_pool.qsa_compressed_k_buffer_pool:" in inspect.getsource(
        cache_controller
    )
    assert (
        "isinstance(kvcache, QSATokenToKVPool) and kvcache.qsa_compressed_k_buffer_pool"
        in inspect.getsource(hybrid_pool_assembler)
    )
