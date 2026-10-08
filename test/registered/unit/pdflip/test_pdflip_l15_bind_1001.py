"""AP L15-11c: unit tests for the pure L1.5 retain bindings (l15_bind).

Hermetic: CPU tensors only, SimpleNamespace fakes, no scheduler import.
The end-to-end test reuses the module-level fakes of
test_pdflip_l15_retain_0930.py loaded by path.
"""

import importlib.util
import os
from types import SimpleNamespace

import torch

from flliper.srt.pdflip import l15_retain
from flliper.srt.pdflip.l15_bind import (
    anchor_slot_of_req,
    build_retain_kwargs,
    node_of_req,
    slots_of_req,
)

_RETAIN_TEST = os.path.join(
    os.path.dirname(__file__), "test_pdflip_l15_retain_0930.py"
)


def _load_retain_test_module():
    spec = importlib.util.spec_from_file_location(
        "test_pdflip_l15_retain_0930", _RETAIN_TEST
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _req(rid, pool_idx, in_len, out_len, anchor, node, kind, active):
    return SimpleNamespace(
        rid=rid,
        req_pool_idx=pool_idx,
        origin_input_ids=list(range(in_len)),
        output_ids=list(range(out_len)),
        mamba_pool_idx=anchor,
        last_node=node,
        l15_kind=kind,
        l15_last_active=active,
    )


def _req_to_token():
    rtt = torch.zeros(4, 8, dtype=torch.int64)
    rtt[0, :6] = torch.tensor([1, 2, 4, 6, 3, 9], dtype=torch.int64)
    rtt[1, 0] = 7
    return rtt


def _rewrite_recorder(node, kv_map, anchor_map, visited):
    # L15-11d recorder standing in for l15_bind.rewrite_tree_chain: the
    # fake node only captures the maps (the real rewrite is pinned in
    # test_pdflip_l15_tree_rewrite_1001.py). The scheduler hook passes the
    # real callable next to **kwargs; build_retain_kwargs does not return
    # it, so the caller supplies it here too.
    node.rewritten = (dict(kv_map), dict(anchor_map))


def test_slots_of_req_returns_token_order_ints():
    req = _req("r_seat", 0, 4, 2, 4, object(), "seat", 2.0)
    slots = slots_of_req(req, _req_to_token())
    # seqlen is 6 but KV exists for seqlen - 1 slots only; the 9 at
    # position 5 (the last, not-yet-forwarded token) is not held.
    assert slots == (1, 2, 4, 6, 3)
    assert all(type(s) is int for s in slots)


def test_slots_of_req_excludes_stale_slot_at_unwritten_last_position():
    # Position seqlen - 1 has no KV yet (schedule_batch.py:2821). A stale
    # nonzero value there would hold someone else's row: never held.
    rtt = _req_to_token()
    rtt[0, 5] = 4242  # stale value at position seqlen - 1
    req = _req("r_seat", 0, 4, 2, 4, object(), "seat", 2.0)
    slots = slots_of_req(req, rtt)
    assert 4242 not in slots
    assert slots == (1, 2, 4, 6, 3)


def test_slots_of_req_raises_on_padding_slot_zero_in_span():
    rtt = _req_to_token()
    rtt[0, 2] = 0  # padding slot inside the token span
    req = _req("r_seat", 0, 4, 2, 4, object(), "seat", 2.0)
    try:
        slots_of_req(req, rtt)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for slot 0 inside span")


def test_anchor_slot_rejects_none_and_zero():
    req = _req("r_a", 0, 1, 0, None, object(), "seat", 1.0)
    try:
        anchor_slot_of_req(req)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for anchor None")
    req0 = _req("r_b", 0, 1, 0, 0, object(), "seat", 1.0)
    try:
        anchor_slot_of_req(req0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for anchor 0")
    assert anchor_slot_of_req(_req("r_c", 0, 1, 0, 5, object(), "seat", 1.0)) == 5


def test_node_of_req_raises_without_last_node():
    req = _req("r_n", 0, 1, 0, 3, None, "seat", 1.0)
    try:
        node_of_req(req)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for last_node None")


def test_build_retain_kwargs_drives_retain_at_sleep_end_to_end(tmp_path):
    rt = _load_retain_test_module()
    sc = rt.make_scenario(tmp_path, [])
    kwargs = dict(sc["kwargs"])
    # build_retain_kwargs owns the req-derived keys and takes a fixed
    # signature; rewrite_tree belongs to the caller (the scheduler passes
    # it next to **kwargs), so drop it from the builder kwargs here.
    for key in ("candidates", "node_of", "slots_of", "anchor_slot_of",
                "l2_of", "rewrite_tree"):
        kwargs.pop(key)
    node_a = rt.FakeNode([])
    node_b = rt.FakeNode([])
    reqs = [
        _req("r_seat", 0, 4, 2, 4, node_a, "seat", 2.0),
        _req("r_parked", 1, 1, 0, 5, node_b, "parked", 1.0),
    ]
    bound = build_retain_kwargs(reqs, _req_to_token(), **kwargs)
    assert bound["l2_of"]("r_seat") == ((), ())
    result = l15_retain.retain_at_sleep(rewrite_tree=_rewrite_recorder, **bound)
    assert isinstance(result, l15_retain.RetainResult)
    assert len(result.keep_nodes) == 2
    assert any(n is node_a for n in result.keep_nodes)
    assert any(n is node_b for n in result.keep_nodes)


def test_rows_by_rank_is_the_exact_owned_count(tmp_path):
    # Prefix [0, 1, 7]: rank 0 owns residue 0 only, rank 1 owns 1..6. The
    # req's 6 slots [1, 2, 4, 6, 3, 9] are all owned by rank 1: exact rows
    # (0, 6). The old proportional estimate rows_split(6, 2, ratios=[1, 6])
    # gave (1, 5) -- over-owning rank 0 by one row against its cap (audit
    # item 11: compact_plan reserves by the exact count).
    req = _req("r1", 0, 3, 4, 1, object(), "served", 0.0)
    kw = build_retain_kwargs(
        [req],
        _req_to_token(),
        caps_rows_by_rank=(100, 100),
        cap_anchor_slots=10,
        prefix=[0, 1, 7],
        rank=0,
        epoch=1,
        pid=1,
        kv_buffers=[],
        mamba_buffers=[],
        allocator=None,
        reset_keep=lambda _ns: None,
        set_keep=lambda _b, _s: None,
        manifest_path=str(tmp_path / "m.json"),
        log=lambda _msg: None,
    )
    assert kw["candidates"][0].rows_by_rank == (0, 6), (
        "rows_by_rank must be the exact owned count, not a proportional split"
    )
