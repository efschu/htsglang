"""DFlash2 across the flip, D side (parts C/D of PLAN_DFLASH2_P_0917).

Hermetic. Pins: the canonical draft head window follows the installed
uneven-TP plan (8 kv heads on 3991/1000/1000 are 5/2/1, the cut the draft's
attention and pool use), the full-head shipment check weighs a DFlash
draft's five context hiddens, the cutover seeds a DFlash draft input, and
the cache controller translates row-addressed draft transfers through a
mapped pool's slot mapper.
"""

from types import SimpleNamespace

import pytest
import torch

import sglang.srt.distributed.utils as du
from sglang.srt.disaggregation.draft_kv_canonical import (
    DraftKvCanonicalLayout,
    DraftKvLayoutMismatch,
    check_full_head_shipment_is_justified,
    local_head_window,
)


@pytest.fixture
def uneven_plan():
    du.set_tp_partition_ratios([3991, 1000, 1000])
    try:
        yield
    finally:
        du.set_tp_partition_ratios(None)


def test_head_window_follows_the_plan(uneven_plan):
    assert [local_head_window(8, 3, r) for r in range(3)] == [(0, 5), (5, 7), (7, 8)]
    # the window is the pool's own cut (tp_partition_size, units = every head)
    assert [du.tp_partition_size(8, 3, r, 8) for r in range(3)] == [5, 2, 1]
    # NEXTN's 4 heads keep 2/1/1 under the plan too
    assert [local_head_window(4, 3, r) for r in range(3)] == [(0, 2), (2, 3), (3, 4)]
    # group P (tp 1) still writes the whole page
    assert local_head_window(8, 1, 0) == (0, 8)


def test_head_window_without_plan_is_largest_remainder():
    assert [local_head_window(8, 3, r) for r in range(3)] == [(0, 3), (3, 6), (6, 8)]


def test_full_head_shipment_weighs_every_context_layer():
    dflash2 = DraftKvCanonicalLayout(
        version=1, num_kv_heads=8, head_dim=128, element_size=2, num_draft_layers=5
    )
    assert dflash2.bytes_per_token() == 20480
    # against ONE hidden the shipment would be refused ...
    with pytest.raises(DraftKvLayoutMismatch):
        check_full_head_shipment_is_justified(dflash2, 5120, 2)
    # ... against the five hiddens the draft actually needs it is the cheaper way
    check_full_head_shipment_is_justified(dflash2, 5120, 2, num_context_layers=5)
    nextn = DraftKvCanonicalLayout(
        version=1, num_kv_heads=4, head_dim=256, element_size=1, num_draft_layers=1
    )
    check_full_head_shipment_is_justified(nextn, 5120, 2, num_context_layers=1)


def test_cutover_seeds_a_dflash_draft_input():
    from sglang.srt.managers.phase_flip_draft_bootstrap import (
        build_bootstrap_dflash_input,
    )
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

    reqs = [
        SimpleNamespace(rid="a", origin_input_ids=[1, 2, 3], output_ids=[7, 8]),
        SimpleNamespace(rid="b", origin_input_ids=[4, 5], output_ids=[]),
    ]
    batch = SimpleNamespace(reqs=reqs, seq_lens=torch.tensor([5, 2]))
    sch = SimpleNamespace(device="cpu")
    inp = build_bootstrap_dflash_input(sch, batch)
    assert isinstance(inp, DFlashDraftInputV2)
    assert inp.bonus_tokens.tolist() == [8, 5]
    assert inp.new_seq_lens.tolist() == [5, 2]


def test_cache_controller_translates_draft_rows_through_the_mapper():
    from sglang.srt.managers.cache_controller import HiCacheController

    calls = []
    mapper = SimpleNamespace(
        translate_read=lambda idx: calls.append(("read", idx.tolist())) or idx + 100,
        translate_write=lambda idx: calls.append(("write", idx.tolist())) or idx + 200,
    )
    cc = HiCacheController.__new__(HiCacheController)
    cc.mem_pool_device_draft = SimpleNamespace(weg2_slot_mapper=mapper)
    idx = torch.tensor([3, 4])
    assert cc._draft_device_indices(idx, "write").tolist() == [103, 104]
    assert cc._draft_device_indices(idx, "load").tolist() == [203, 204]
    assert [c[0] for c in calls] == ["read", "write"]
    cc.mem_pool_device_draft = SimpleNamespace()  # a mirror pool
    assert cc._draft_device_indices(idx, "load") is idx
