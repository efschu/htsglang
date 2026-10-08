# SPDX-License-Identifier: Apache-2.0
"""L15-12c-E2a: the GDN anchor's L2 identity is recorded at sleep.

HoldSpan gains anchor_l2_slot / anchor_l2_gen (the mamba anchor state's
arena slot and generation), so the cap-0 wake (TP0) can later refill its
anchors from L2 instead of voting "no hold". Pinned here:

(a) manifest round-trip carries the two fields; an OLD record without them
    loads with -1 (the wake gate reads them as "absent");
(b) the fingerprint covers the anchor columns (group agreement includes
    the anchor L2 identity);
(c) l15_bind.anchor_l2_of maps the anchor node's MAMBA host_value row
    through the mamba host pool (arena slot = row - staging_rows, one
    state per slot; a staging row is no L2 copy -> (-1, -1));
(d) retain_at_sleep writes the columns into the published HoldSpan.
"""

from __future__ import annotations

import importlib.util
import json
import os
from types import SimpleNamespace

import torch

from flliper.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from flliper.srt.pdflip import l15_manifest, l15_retain
from flliper.srt.pdflip.l15_bind import build_retain_kwargs
from flliper.srt.pdflip.l15_policy import Candidate

_HERE = os.path.dirname(__file__)
_RETAIN_TEST = os.path.join(_HERE, "test_pdflip_l15_retain_0930.py")


def _load_retain_test_module():
    spec = importlib.util.spec_from_file_location(
        "test_pdflip_l15_retain_0930", _RETAIN_TEST
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- manifest


def _span(anchor_l2=(3, 9)):
    return l15_manifest.HoldSpan(
        rid="r_a",
        depth=4,
        slots=(1, 2),
        anchor_slot=4,
        l2_slots=(201, 202),
        l2_gens=(5, 6),
        anchor_l2_slot=anchor_l2[0],
        anchor_l2_gen=anchor_l2[1],
    )


def _manifest(span):
    return l15_manifest.Manifest(
        epoch=7, pid=101, spans=(span,), rows_by_rank=(4, 4), anchor_slots=3
    )


def test_manifest_roundtrip_carries_anchor_l2():
    m = _manifest(_span((3, 9)))
    back = l15_manifest.from_json(l15_manifest.to_json(m))
    sp = back.spans[0]
    assert sp.anchor_l2_slot == 3
    assert sp.anchor_l2_gen == 9
    assert l15_manifest.fingerprint(back) == l15_manifest.fingerprint(m)


def test_old_record_loads_with_minus_one():
    # A record written before E2a: no anchor_l2_* keys anywhere.
    old = {
        "epoch": 7,
        "pid": 101,
        "spans": [
            {
                "rid": "r_a",
                "depth": 4,
                "slots": [1, 2],
                "anchor_slot": 4,
                "l2_slots": [201, 202],
                "l2_gens": [5, 6],
            }
        ],
        "rows_by_rank": [4, 4],
        "anchor_slots": 3,
    }
    back = l15_manifest.from_json(json.dumps(old))
    sp = back.spans[0]
    assert sp.anchor_l2_slot == -1
    assert sp.anchor_l2_gen == -1


def test_fingerprint_covers_anchor_l2_gen():
    fp_a = l15_manifest.fingerprint(_manifest(_span((3, 9))))
    fp_b = l15_manifest.fingerprint(_manifest(_span((3, 10))))
    assert fp_a != fp_b
    fp_c = l15_manifest.fingerprint(_manifest(_span((4, 9))))
    assert fp_a != fp_c


# -------------------------------------------------------------------- bind


class _FakeMambaPool:
    """staging_rows + slot_gens, the two attributes anchor_l2_of reads."""

    def __init__(self, staging_rows, gens):
        self.staging_rows = staging_rows
        self._gens = gens

    def slot_gens(self, slots):
        return [self._gens.get(int(s), -1) for s in slots]


def _cd(value, host_value):
    def _t(v):
        return None if v is None else torch.tensor([v], dtype=torch.int64)

    return SimpleNamespace(value=_t(value), host_value=_t(host_value),
                           host_lock_ref=0)


def _node(anchor, host_row):
    return SimpleNamespace(
        parent=None,
        key=(0,),
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        component_data={
            ComponentType.FULL: _cd(None, None),
            # the anchor node carries the anchor's device slot as value and
            # its L2 host row as host_value
            ComponentType.MAMBA: _cd(anchor, host_row),
        },
    )


def _req(rid, pool_idx, in_len, anchor, node):
    return SimpleNamespace(
        rid=rid,
        req_pool_idx=pool_idx,
        origin_input_ids=list(range(in_len)),
        output_ids=[],
        mamba_pool_idx=anchor,
        last_node=node,
        l15_kind="served",
        l15_last_active=1.0,
    )


def _req_to_token():
    rtt = torch.zeros(4, 8, dtype=torch.int64)
    rtt[0, :6] = torch.tensor([1, 2, 4, 6, 3, 9], dtype=torch.int64)
    rtt[1, 0] = 7
    return rtt


def _build(tmp_path, pool):
    reqs = [
        # anchor 4, L2 host row 9 -> arena slot 9-2=7, gen 42
        _req("r_seat", 0, 6, 4, _node(4, 9)),
        # anchor 5, host row 1 < staging_rows=2 -> staging, no L2 copy
        _req("r_parked", 1, 1, 5, _node(5, 1)),
    ]
    return build_retain_kwargs(
        reqs,
        _req_to_token(),
        caps_rows_by_rank=(4, 10),
        cap_anchor_slots=5,
        prefix=(0, 1, 2),
        rank=1,
        epoch=77,
        pid=4242,
        kv_buffers=[],
        mamba_buffers=[],
        allocator=None,
        reset_keep=lambda ns: None,
        set_keep=lambda buf, spans: None,
        manifest_path=str(tmp_path / "m.json"),
        log=lambda s: None,
        mamba_host_pool=pool,
    )


def test_anchor_l2_of_maps_host_row_to_slot_gen(tmp_path):
    pool = _FakeMambaPool(2, {7: 42})
    bound = _build(tmp_path, pool)
    assert bound["anchor_l2_of"]("r_seat") == (7, 42)


def test_anchor_l2_of_staging_row_is_minus_one(tmp_path):
    pool = _FakeMambaPool(2, {7: 42})
    bound = _build(tmp_path, pool)
    assert bound["anchor_l2_of"]("r_parked") == (-1, -1)
    assert bound["anchor_l2_of"]("r_unknown") == (-1, -1)


# ------------------------------------------------------------------ retain


def test_retain_writes_anchor_columns_into_holdspan(tmp_path):
    mod = _load_retain_test_module()
    events: list = []
    sc = mod.make_scenario(tmp_path, events)
    sc["kwargs"]["anchor_l2_of"] = lambda rid: {
        "r_seat": (7, 42),
        "r_parked": (11, 7),
    }.get(rid, (-1, -1))
    res = l15_retain.retain_at_sleep(**sc["kwargs"])
    assert res is not None, f"retain skipped: {sc['log_lines']}"
    # PID 4242 is a fake: read must not reap the record as "owner dead".
    m = l15_manifest.read(sc["manifest_path"], pid_alive=lambda _pid: True)
    by_rid = {s.rid: s for s in m.spans}
    assert by_rid["r_seat"].anchor_l2_slot == 7
    assert by_rid["r_seat"].anchor_l2_gen == 42
    assert by_rid["r_parked"].anchor_l2_slot == 11
    assert by_rid["r_parked"].anchor_l2_gen == 7


def test_retain_without_anchor_l2_of_keeps_minus_one(tmp_path):
    # Existing callers (no anchor_l2_of kwarg) keep working; the columns
    # land as -1/-1, which the cap-0 wake gate reads as "no hold".
    mod = _load_retain_test_module()
    events: list = []
    sc = mod.make_scenario(tmp_path, events)
    res = l15_retain.retain_at_sleep(**sc["kwargs"])
    assert res is not None, f"retain skipped: {sc['log_lines']}"
    m = l15_manifest.read(sc["manifest_path"], pid_alive=lambda _pid: True)
    for s in m.spans:
        assert s.anchor_l2_slot == -1
        assert s.anchor_l2_gen == -1
