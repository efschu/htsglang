# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-NOIDX: a candidate req without a req_to_token row must be skipped
per rid, not kill the whole retain round.

N3c (01.10. 20:18-20:26Z): every D sleep logged "L15-RETAIN failed before
the move (flushing as today): TypeError: int() argument must be ... not
'NoneType'" -- slots_of_req did int(req.req_pool_idx) on a req whose row
was already released (req_pool_idx None). The TypeError escaped
build_retain_kwargs, so no round ever retained anything.

Driven through the hook's real call shape: build_retain_kwargs, then
retain_at_sleep(rewrite_tree=..., **kwargs) exactly as scheduler.py calls it.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pytest  # noqa: E402

from sglang.srt.weg2 import l15_retain  # noqa: E402
from sglang.srt.weg2.l15_bind import build_retain_kwargs, slots_of_req  # noqa: E402
from test_weg2_l15_bind_1001 import (  # noqa: E402
    _load_retain_test_module,
    _req,
    _req_to_token,
    _rewrite_recorder,
)


def test_slots_of_req_without_row_raises_value_error():
    req = _req("r_gone", None, 4, 2, 4, object(), "seat", 1.0)
    with pytest.raises(ValueError):
        slots_of_req(req, _req_to_token())


def test_req_without_row_is_skipped_per_rid_and_round_still_retains(tmp_path):
    rt = _load_retain_test_module()
    sc = rt.make_scenario(tmp_path, [])
    kwargs = dict(sc["kwargs"])
    for key in ("candidates", "node_of", "slots_of", "anchor_slot_of",
                "l2_of", "rewrite_tree"):
        kwargs.pop(key)
    logged = []
    kwargs["log"] = logged.append
    node_a = rt.FakeNode([])
    node_b = rt.FakeNode([])
    node_gone = rt.FakeNode([])
    reqs = [
        _req("r_seat", 0, 4, 2, 4, node_a, "seat", 2.0),
        _req("r_gone", None, 4, 2, 6, node_gone, "seat", 3.0),
        _req("r_parked", 1, 1, 0, 5, node_b, "parked", 1.0),
    ]
    bound = build_retain_kwargs(reqs, _req_to_token(), **kwargs)
    rids = {c.rid for c in bound["candidates"]}
    assert rids == {"r_seat", "r_parked"}, rids
    assert any("req_pool_idx" in m and "r_gone" in m for m in logged), logged
    result = l15_retain.retain_at_sleep(rewrite_tree=_rewrite_recorder, **bound)
    assert isinstance(result, l15_retain.RetainResult)
    assert len(result.keep_nodes) == 2
    assert not any(n is node_gone for n in result.keep_nodes)
