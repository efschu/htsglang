# SPDX-License-Identifier: Apache-2.0
"""AP L15-D1 cross-module contract audit fixes (2026-10-01).

Fixes pinned here:
* l15_shadow.rows_split: the proportional remainder skips zero-share ranks
  (the NF form's rank 0 owns no slot under the owner rule);
* l15_bind.build_retain_kwargs: candidate rows_by_rank follow the
  owner-weighted prefix widths, not an even split (mirrors the shadow hook);
* l15_bind.anchor_slot_of_req / node_of_req: a missing (None) or padding (0)
  anchor / missing node RAISES (benign pre-step-3 skip), it does not return
  a sentinel -- the old docstrings claimed "0/None mean nothing to hold";
* hot_handover.decide: a first candidate failing a condition is skipped and
  the next one is tried (the old docstring claimed the first candidate wins);
* l15_row_plan.plan_p_to_d: the P-side source row is the GLOBAL token index
  (``i`` itself), 0-based within the chunk only when ``a == 0``.
"""

from types import SimpleNamespace

import pytest

from sglang.srt.weg2 import l15_bind, l15_shadow
from sglang.srt.weg2.hot_handover import decide
from sglang.srt.weg2.l15_row_plan import plan_p_to_d


def test_rows_split_remainder_skips_zero_share_rank():
    # NF form [0, 9, 7] over 10 tokens: rank 0 must stay at 0 rows.
    rows = l15_shadow.rows_split(10, 3, [0, 9, 7])
    assert sum(rows) == 10
    assert rows[0] == 0


def test_rows_split_weighted_parts_sum_and_order():
    # 27B form [7, 4, 5] over 7 tokens: floor 3/1/2, remainder -> rank 0.
    assert l15_shadow.rows_split(7, 3, [7, 4, 5]) == (4, 1, 2)
    assert sum(l15_shadow.rows_split(10, 3, [7, 4, 5])) == 10


def _retain_kwargs(reqs, prefix):
    return l15_bind.build_retain_kwargs(
        reqs,
        None,
        caps_rows_by_rank=(0, 0, 0),
        cap_anchor_slots=8,
        prefix=prefix,
        rank=0,
        epoch=1,
        pid=1,
        kv_buffers=[],
        mamba_buffers=[],
        allocator=None,
        reset_keep=None,
        set_keep=None,
        manifest_path="/nonexistent/l15.manifest",
        log=lambda line: None,
    )


def test_bind_candidate_rows_follow_prefix_widths_not_even_split():
    req = SimpleNamespace(rid="r1", origin_input_ids=list(range(8)), output_ids=[])
    kwargs = _retain_kwargs([req], [0, 7, 11, 16])
    rows = kwargs["candidates"][0].rows_by_rank
    # span = seqlen - 1 = 7 -> weighted (4, 1, 2); an even split would be
    # (3, 2, 2), which under-prices rank 0 (owns 7/16) against its cap.
    assert rows == (4, 1, 2)


def test_bind_candidate_rows_zero_share_rank_stays_empty():
    req = SimpleNamespace(rid="r2", origin_input_ids=list(range(8)), output_ids=[])
    kwargs = _retain_kwargs([req], [0, 0, 9, 16])
    rows = kwargs["candidates"][0].rows_by_rank
    assert rows[0] == 0
    assert sum(rows) == 7


def test_anchor_slot_of_req_raises_on_missing_and_padding():
    with pytest.raises(ValueError):
        l15_bind.anchor_slot_of_req(SimpleNamespace(rid="x", mamba_pool_idx=None))
    with pytest.raises(ValueError):
        l15_bind.anchor_slot_of_req(SimpleNamespace(rid="x", mamba_pool_idx=0))


def test_node_of_req_raises_on_missing_node():
    with pytest.raises(ValueError):
        l15_bind.node_of_req(SimpleNamespace(rid="x", last_node=None))


def test_decide_skips_first_failing_candidate_and_tries_next():
    bad = {"rid": "a", "hot_in_d": True, "prefix_tokens": 5, "anchor_depth": 3}
    good = {"rid": "b", "hot_in_d": True, "prefix_tokens": 4, "anchor_depth": 4}
    plan = decide([bad, good], p_free_rows=10)
    assert plan is not None
    assert plan.rid == "b"
    assert plan.n_tokens == 4
    assert plan.p_row0 == 0


def test_plan_p_to_d_src_rows_are_global_token_indices():
    # Chunk tokens [2, 5) at e0=32: residues 2,3,4 of S=16 all sit in rank 0's
    # [0,7); with tp0_skip off the P-side rows must be the token indices
    # 2,3,4 themselves -- not 0-based chunk positions 0,1,2.
    pieces = plan_p_to_d(
        a=2,
        b=5,
        e0=32,
        prefix=[0, 7, 11, 16],
        stage=0,
        stage_layers=[(0,)],
        stage_card=[0],
        rank_card=[0, 1, 2],
        row_bytes_per_layer=1024,
        slot_bytes=1 << 20,
        tp0_skip=False,
    )
    assert pieces, "expected at least one piece"
    src_rows = sorted({r for p in pieces for r in p.src_rows})
    assert src_rows == [2, 3, 4]
