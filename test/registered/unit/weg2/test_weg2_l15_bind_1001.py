"""AP L15-11c: unit tests for the pure L1.5 retain bindings (l15_bind).

Hermetic: CPU tensors only, SimpleNamespace fakes, no scheduler import.
The end-to-end test reuses the module-level fakes of
test_weg2_l15_retain_0930.py loaded by path.
"""

import importlib.util
import os
from types import SimpleNamespace

import torch

from sglang.srt.weg2 import l15_retain
from sglang.srt.weg2.l15_bind import (
    anchor_slot_of_req,
    build_retain_kwargs,
    node_of_req,
    slots_of_req,
)

_RETAIN_TEST = os.path.join(
    os.path.dirname(__file__), "test_weg2_l15_retain_0930.py"
)


def _load_retain_test_module():
    spec = importlib.util.spec_from_file_location(
        "test_weg2_l15_retain_0930", _RETAIN_TEST
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
    for key in ("candidates", "node_of", "slots_of", "anchor_slot_of", "l2_of"):
        kwargs.pop(key)
    node_a = rt.FakeNode([])
    node_b = rt.FakeNode([])
    reqs = [
        _req("r_seat", 0, 4, 2, 4, node_a, "seat", 2.0),
        _req("r_parked", 1, 1, 0, 5, node_b, "parked", 1.0),
    ]
    bound = build_retain_kwargs(reqs, _req_to_token(), **kwargs)
    assert bound["l2_of"]("r_seat") == ((), ())
    result = l15_retain.retain_at_sleep(**bound)
    assert isinstance(result, l15_retain.RetainResult)
    assert len(result.keep_nodes) == 2
    assert any(n is node_a for n in result.keep_nodes)
    assert any(n is node_b for n in result.keep_nodes)
