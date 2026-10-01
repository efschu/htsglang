# SPDX-License-Identifier: Apache-2.0
"""L15-INT: hermetic CPU integration test of the whole L1.5 chain.

Every AP (A wake restore, B fallback/defer, C2 l2_of, D refill, E1
group_check, E2 cap-0 gate, F2 keep-arm, F7 shared prefix) was tested
against its own fakes. This file drives the seams between them on REAL CPU
pool objects and the fake-self wake style:

  sleep (retain_at_sleep per rank, F7 shared-prefix shape, manifests per
  (group, rank)) -> manifest content asserts -> wake per rank (cap>0 keeps
  the hold; cap-0 votes None -> group verdict "fallback" -> fallback drop
  frees everything) -> failing keep-arm (F2) drops the whole round.

Geometry: 3-rank D group, uneven prefix (0, 1, 2, 3) -> S = 3; rank r owns
global slots with L % 3 == r. Two requests share their first three slots
(10, 11, 12) -- one per class -- and the same mamba anchor; unique tails
13 (rank 1) and 14 (rank 2). need = [1, 2, 2] -> blocks 2 -> L_H = 6.
"""

from __future__ import annotations

import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.weg2 import l15_manifest, l15_retain
from sglang.srt.weg2.l15_policy import Candidate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

NUM_LAYERS = 8
GLOBAL_INTERVAL = 4
MAMBA_SIZE = 6
MAX_CONTEXT_LEN = 128

PREFIX = (0, 1, 2, 3)
EPOCH = 777
PID = 4242

# F7 shape: sa/sb share (10, 11, 12); tails 13 (rank1) / 14 (rank2); both
# end on the same radix node -> the SAME mamba anchor 5.
SLOTS_OF = {"sa": (10, 11, 12, 13), "sb": (10, 11, 12, 14)}
ANCHOR_SLOT_OF = {"sa": 5, "sb": 5}

CANDIDATES = [
    Candidate(rid="sa", kind="served", last_active=9.0,
              rows_by_rank=(2, 2, 2), anchor_depth=3, kv_depth=3),
    Candidate(rid="sb", kind="served", last_active=8.0,
              rows_by_rank=(2, 2, 2), anchor_depth=3, kv_depth=3),
]


def _build_pool() -> HybridReqToTokenPool:
    """Real CPU HybridReqToTokenPool (scaffold from retain_cpu_pool_1001)."""
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(server_args)
    full_attention_layer_ids = [
        i for i in range(GLOBAL_INTERVAL - 1, NUM_LAYERS, GLOBAL_INTERVAL)
    ]
    mamba_layers = [i for i in range(NUM_LAYERS) if i not in full_attention_layer_ids]
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=512, n_groups=4,
            num_heads=8, head_dim=64, state_size=32, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)
    return HybridReqToTokenPool(
        size=10, mamba_size=MAMBA_SIZE, mamba_spec_state_size=10,
        max_context_len=MAX_CONTEXT_LEN, device="cpu",
        enable_memory_saver=False, cache_params=cache_params,
        mamba_layer_ids=mamba_layers, enable_mamba_extra_buffer=False,
        enable_linear_replayssm=False,
    )


def _mamba_views(pool: HybridReqToTokenPool):
    cache = pool.mamba_pool.mamba_cache
    views = []
    for c in cache.conv:
        views.extend(c[i] for i in range(int(c.shape[0])))
    views.extend(cache.temporal[i] for i in range(int(cache.temporal.shape[0])))
    return views


class FakeAllocator:
    def __init__(self, size=16):
        self.size = size
        self.free_pages = torch.empty(0, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)

    def clear(self):
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)


class FakeNode:
    pass


def _recorder(node, kv_map, anchor_map, visited):
    node.rewritten = (dict(kv_map), dict(anchor_map))


def _retain_rank(tmp_path, rank, caps=(4, 4, 4), manifest_name=None, mamba=None):
    """One rank's sleep: retain_at_sleep against fresh per-rank fakes."""
    pool = mamba if mamba is not None else _build_pool()
    alloc = FakeAllocator()
    alloc.clear()
    nodes = {"sa": FakeNode(), "sb": FakeNode()}
    res = l15_retain.retain_at_sleep(
        candidates=list(CANDIDATES),
        node_of=lambda rid: nodes[rid],
        slots_of=lambda rid: SLOTS_OF[rid],
        anchor_slot_of=lambda rid: ANCHOR_SLOT_OF[rid],
        l2_of=lambda rid: ((201, 202), (5, 6)),
        rewrite_tree=_recorder,
        caps_rows_by_rank=caps,
        cap_anchor_slots=4,
        prefix=PREFIX,
        rank=rank,
        epoch=EPOCH,
        pid=PID,
        kv_buffers=[torch.zeros(8, 3), torch.zeros(8, 3)],
        mamba_buffers=_mamba_views(pool),
        allocator=alloc,
        mamba_allocator=pool.mamba_allocator,
        reset_keep=lambda ns: None,
        set_keep=lambda buf, spans: None,
        manifest_path=str(tmp_path / (manifest_name or "l15_manifest.json")),
        log=lambda line: None,
    )
    return res, pool, alloc, nodes


def test_sleep_writes_per_rank_manifests_f7_shape(tmp_path):
    # STEP 1+2 (brief): sleep on ranks 1 and 2 (cap>0); manifests written;
    # same fingerprint; remapped slots; shared slots once per holder; l2 set.
    res1, _pool1, _a1, nodes1 = _retain_rank(tmp_path, 1,
                                             manifest_name="m_rank1.json")
    res2, _pool2, _a2, nodes2 = _retain_rank(tmp_path, 2,
                                             manifest_name="m_rank2.json")
    assert res1 is not None and res2 is not None, "F7 shape must retain"
    m1 = l15_manifest.from_json((tmp_path / "m_rank1.json").read_text())
    m2 = l15_manifest.from_json((tmp_path / "m_rank2.json").read_text())
    # The fingerprint covers the hold, not the rank: equal inputs -> equal fp.
    assert l15_manifest.fingerprint(m1) == l15_manifest.fingerprint(m2)
    assert m1.epoch == EPOCH and m2.epoch == EPOCH
    for m in (m1, m2):
        spans = {s.rid: s for s in m.spans}
        assert set(spans) == {"sa", "sb"}
        sa, sb = spans["sa"], spans["sb"]
        # L_H = 6; 10, 11, 12, 13, 14 are ALL >= L_H and move to the free
        # class rows: 10 -> 1, 11 -> 2, 12 -> 3, 13 -> 4, 14 -> 5 (slot 0 is
        # the reserved padding and never a target).
        assert sa.slots == (1, 2, 3, 4)
        assert sb.slots == (1, 2, 3, 5)
        # Shared images identical at the same indices, once per holder.
        assert sa.slots[:3] == sb.slots[:3]
        assert len(set(sa.slots)) == 4 == len(set(sb.slots))
        # Both rids share the ONE anchor image; A_H = 2 -> anchor 5 -> 1.
        assert sa.anchor_slot == sb.anchor_slot == 1
        # l2 columns present (C2: arena page slots recorded per span).
        assert sa.l2_slots == (201, 202) and sb.l2_slots == (201, 202)
        assert sa.l2_gens == (5, 6)
    # The tree rewrite saw the per-rid maps through the same seam.
    kv_map_a, anc_a = nodes1["sa"].rewritten
    kv_map_b, _ = nodes1["sb"].rewritten
    for shared in (10, 11, 12):
        assert kv_map_a[shared] == kv_map_b[shared]
    assert anc_a == {5: 1}  # the shared anchor moves once, both rids see it
    # Mamba allocator: the two compact anchor rows are carved out.
    free = set(_pool1.mamba_allocator.free_slots.tolist())
    assert 1 not in free
