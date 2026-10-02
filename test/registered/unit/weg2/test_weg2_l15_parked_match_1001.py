# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-PARKED / L15-FIX-EPOCH: parked reqs are held from the radix tree,
and the retain round carries the front's flip index.

N3f (10012208, 01.10. 22:21Z): the D park retracts every running req with
retain=True (release_kv_cache(is_insert=True) + reset_for_retract): its span
is inserted into the tree, but req_pool_idx/last_node/mamba_pool_idx become
None. The shadow planned "n=2 rows_by_rank=55961,51095,48660" from token
counts, retain logged "skipped 2 req(s) without a holdable span ... req has
no req_pool_idx" -> "n=0 nothing_to_hold". Every sleep also logged epoch=0
(_weg2_vote_epoch is PP0's idle-vote counter, never moved on D).
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
)
from sglang.srt.weg2 import l15_bind, l15_retain  # noqa: E402
from test_weg2_l15_bind_1001 import (  # noqa: E402
    _load_retain_test_module,
    _req,
    _req_to_token,
    _rewrite_recorder,
)


def _parked_node(rt, anchor):
    node = rt.FakeNode([])
    cds = [SimpleNamespace(value=None, host_value=None) for _ in range(3)]
    cds[ComponentType.MAMBA] = SimpleNamespace(
        value=torch.tensor([anchor], dtype=torch.int64), host_value=None)
    node.component_data = cds
    return node


class _FakeTree:
    """match_prefix returns the span the park inserted (device indices +
    the node carrying the mamba checkpoint); records the params."""

    def __init__(self, by_len):
        self.by_len = by_len
        self.calls = []

    def match_prefix(self, params):
        # the real tree compares keys with RadixKey.match, which asserts the
        # SAME container type as the stored keys: array('q') (N3h 22:52Z:
        # AssertionError (array.array, list) on a list key)
        from array import array as _array

        tid = params.key.token_ids
        assert isinstance(tid, _array) and tid.typecode == "q", (type(tid), getattr(tid, "typecode", None))
        self.calls.append(params)
        n = len(params.key.token_ids)
        slots, node = self.by_len[n]
        return SimpleNamespace(
            device_indices=torch.tensor(slots, dtype=torch.int64),
            last_device_node=node,
        )


def _parked_req(rid, in_len, out_len, kind, active):
    # exactly what reset_for_retract leaves: no row, no node, no anchor
    return _req(rid, None, in_len, out_len, None, None, kind, active)


def _kwargs(rt, tmp_path, logged):
    sc = rt.make_scenario(tmp_path, [])
    kwargs = dict(sc["kwargs"])
    for key in ("candidates", "node_of", "slots_of", "anchor_slot_of",
                "l2_of", "rewrite_tree"):
        kwargs.pop(key)
    kwargs["log"] = logged.append
    return kwargs


def test_match_parked_reads_span_node_and_anchor():
    rt = _load_retain_test_module()
    node = _parked_node(rt, 6)
    tree = _FakeTree({5: ([11, 12, 13, 14, 15], node)})
    req = _parked_req("p1", 4, 2, "parked", 1.0)
    slots, got_node, anchor = l15_bind.match_parked(req, tree)
    assert slots == (11, 12, 13, 14, 15)
    assert got_node is node
    assert anchor == 6
    p = tree.calls[0]
    assert list(p.key.token_ids) == [0, 1, 2, 3, 0]  # seqlen - 1 tokens
    assert p.cow_mamba is False  # never allocates a mamba slot


def test_match_parked_without_device_span_or_anchor_raises():
    rt = _load_retain_test_module()
    no_anchor = rt.FakeNode([])
    no_anchor.component_data = [SimpleNamespace(value=None, host_value=None)] * 3
    req = _parked_req("p1", 4, 2, "parked", 1.0)
    with pytest.raises(ValueError):
        l15_bind.match_parked(req, _FakeTree({5: ([], no_anchor)}))
    with pytest.raises(ValueError):
        l15_bind.match_parked(req, _FakeTree({5: ([3, 4, 5, 6, 7], no_anchor)}))


def test_parked_req_is_held_through_the_hook_shape(tmp_path):
    rt = _load_retain_test_module()
    logged = []
    kwargs = _kwargs(rt, tmp_path, logged)
    kwargs["caps_rows_by_rank"] = (10, 10)  # room for both (the policy is not under test)
    node_a = rt.FakeNode([])
    node_p = _parked_node(rt, 3)
    reqs = [
        _req("r_seat", 0, 4, 2, 4, node_a, "seat", 2.0),
        _parked_req("r_park", 3, 1, "parked", 3.0),
    ]
    tree = _FakeTree({3: ([5, 7, 8], node_p)})
    bound = l15_bind.build_retain_kwargs(reqs, _req_to_token(), tree_cache=tree,
                                         **kwargs)
    rids = {c.rid for c in bound["candidates"]}
    assert rids == {"r_seat", "r_park"}, (rids, logged)
    assert bound["slots_of"]("r_park") == (5, 7, 8)
    assert bound["node_of"]("r_park") is node_p
    assert bound["anchor_slot_of"]("r_park") == 3
    result = l15_retain.retain_at_sleep(rewrite_tree=_rewrite_recorder, **bound)
    assert isinstance(result, l15_retain.RetainResult)
    assert any(n is node_p for n in result.keep_nodes)


def test_without_tree_cache_a_parked_req_is_still_skipped(tmp_path):
    rt = _load_retain_test_module()
    logged = []
    kwargs = _kwargs(rt, tmp_path, logged)
    reqs = [_req("r_seat", 0, 4, 2, 4, rt.FakeNode([]), "seat", 2.0),
            _parked_req("r_park", 3, 1, "parked", 3.0)]
    bound = l15_bind.build_retain_kwargs(reqs, _req_to_token(), **kwargs)
    assert {c.rid for c in bound["candidates"]} == {"r_seat"}
    assert any("r_park" in m for m in logged)


def test_sleep_epoch_prefers_the_front_flip_index():
    assert l15_bind.sleep_epoch(SimpleNamespace(_weg2_vote_epoch=0, _l15_sleep_flip=17)) == 17
    assert l15_bind.sleep_epoch(SimpleNamespace(_weg2_vote_epoch=3, _l15_sleep_flip=-1)) == 3
    assert l15_bind.sleep_epoch(SimpleNamespace(_weg2_vote_epoch=0)) == 0


def test_hook_passes_tree_cache_and_sleep_epoch():
    import ast

    src = (pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang"
           / "srt" / "managers" / "scheduler.py").read_text()
    tree = ast.parse(src)
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", None) == "build_retain_kwargs"]
    assert calls, "no build_retain_kwargs call in scheduler.py"
    kws = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kws.get("tree_cache") == "self.tree_cache"
    assert "sleep_epoch" in kws.get("epoch", "")
