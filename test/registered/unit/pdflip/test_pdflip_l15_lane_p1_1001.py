# SPDX-License-Identifier: Apache-2.0
"""L15-12c-P1: the lane per token for P>1 (the NF form).

C2 recorded only the arena PAGE slot per token, so refill could never
serve page_tokens > 1 without copying foreign lanes. Pinned here:

(a) HoldSpan.l2_lanes round-trips (one lane per held token; () on old
    records); the fingerprint covers it;
(b) l15_bind records lane = (row - S) % P next to slot = (row - S) // P
    (staging rows: lane -1; P == 1: all lanes 0) and exposes it through
    the NEW l2_lanes_of -- l2_of's 2-tuple contract is unchanged;
(c) l15_refill.refill with P>1 loads ONE page-grouped call with the
    common lane list (device_indices in (page, lane) order) when every
    page owns the same lanes, and names the loader-contract refusal
    when the pages differ; P == 1 behaviour byte-identical.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from flliper.srt.pdflip import l15_manifest, l15_refill  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
)
from flliper.srt.pdflip.l15_bind import build_retain_kwargs  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
_RETAIN_TEST = str(_HERE / "test_pdflip_l15_retain_0930.py")


def _load_retain_test_module():
    spec = importlib.util.spec_from_file_location(
        "test_pdflip_l15_retain_0930", _RETAIN_TEST
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- manifest


def _span(l2_lanes):
    kw = {} if l2_lanes is None else {"l2_lanes": l2_lanes}
    return l15_manifest.HoldSpan(
        rid="r", depth=3, slots=(1, 2, 3), anchor_slot=4,
        l2_slots=(10, 11, 12), l2_gens=(5, 5, 6), **kw,
    )


def _manifest(span):
    return l15_manifest.Manifest(
        epoch=3, pid=11, spans=(span,), rows_by_rank=(4, 4), anchor_slots=2
    )


def test_l2_lanes_roundtrip_and_old_record():
    m = _manifest(_span((0, 1, 2)))
    back = l15_manifest.from_json(l15_manifest.to_json(m))
    assert back.spans[0].l2_lanes == (0, 1, 2)
    old = json.loads(l15_manifest.to_json(m))
    for sp in old["spans"]:
        sp.pop("l2_lanes")
        sp.pop("anchor_l2_slot", None)
        sp.pop("anchor_l2_gen", None)
    back2 = l15_manifest.from_json(json.dumps(old))
    assert back2.spans[0].l2_lanes == ()


def test_fingerprint_covers_l2_lanes():
    fp_a = l15_manifest.fingerprint(_manifest(_span((0, 1, 2))))
    fp_b = l15_manifest.fingerprint(_manifest(_span((0, 1, 3))))
    assert fp_a != fp_b


# ----------------------------------------------------------------- bind


class _FakeKVPool:
    def __init__(self, staging_rows, pages, gens=None):
        self.staging_rows = staging_rows
        self._arena_page_tokens = pages
        self.row_slot = None
        self._gens = gens or {}

    def slot_gens(self, slots):
        return [self._gens.get(int(s), -1) for s in slots]


def _node(host_rows):
    def _t(v):
        return None if v is None else torch.tensor(v, dtype=torch.int64)

    return SimpleNamespace(
        parent=None,
        key=(0,),
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        component_data={
            ComponentType.FULL: SimpleNamespace(
                value=None, host_value=_t(host_rows), host_lock_ref=0
            ),
            ComponentType.MAMBA: SimpleNamespace(
                value=_t([4]), host_value=_t([3]), host_lock_ref=0
            ),
        },
    )


def _bind(tmp_path, pool):
    req = SimpleNamespace(
        rid="r_seat",
        req_pool_idx=0,
        origin_input_ids=list(range(9)),  # seq 9 -> span 8
        output_ids=[],
        mamba_pool_idx=4,
        last_node=_node([0, 1, 2, 3, 4, 5, 6, 7]),
        l15_kind="served",
        l15_last_active=1.0,
    )
    rtt = torch.zeros(2, 12, dtype=torch.int64)
    rtt[0, :9] = torch.arange(1, 10, dtype=torch.int64)
    return build_retain_kwargs(
        [req], rtt,
        caps_rows_by_rank=(4, 10), cap_anchor_slots=5,
        prefix=(0, 1, 2), rank=1, epoch=77, pid=4242,
        kv_buffers=[], mamba_buffers=[], allocator=None,
        reset_keep=lambda ns: None, set_keep=lambda buf, spans: None,
        manifest_path=str(tmp_path / "m.json"), log=lambda s: None,
        host_pool=pool,
    )


def test_l2_lanes_of_p4_divmod(tmp_path):
    # rows 0,1 staging -> lane -1; rows 2..7: slot (r-2)//4, lane (r-2)%4
    bound = _bind(tmp_path, _FakeKVPool(2, 4, {0: 5, 1: 6}))
    assert bound["l2_lanes_of"]("r_seat") == (-1, -1, 0, 1, 2, 3, 0, 1)
    # row S+5 = 7 -> slot 1 lane 1; the l2_of 2-tuple contract unchanged
    slots, gens = bound["l2_of"]("r_seat")
    assert slots == (-1, -1, 0, 0, 0, 0, 1, 1)
    assert gens == (-1, -1, 5, 5, 5, 5, 6, 6)


def test_l2_lanes_of_p1_all_zero(tmp_path):
    bound = _bind(tmp_path, _FakeKVPool(2, 1, {0: 5, 1: 6}))
    assert bound["l2_lanes_of"]("r_seat") == (-1, -1, 0, 0, 0, 0, 0, 0)
    assert bound["l2_lanes_of"]("r_unknown") == ()


# -------------------------------------------------------------- refill


class _RecHostPool:
    def __init__(self, raise_exc=False):
        self.calls = []
        self.raise_exc = raise_exc

    def _load_pages_all_layers(self, device_pool, slots, device_indices,
                                lanes=None, mode=None):
        if self.raise_exc:
            raise RuntimeError("boom")
        self.calls.append(
            ([int(x) for x in slots], [int(x) for x in device_indices],
             None if lanes is None else [int(x) for x in lanes])
        )


def test_refill_p4_one_page_grouped_call():
    # pages 5 and 9, each owning lanes 2 and 3 -> ONE call, lanes [2, 3],
    # device_indices in (page, lane) order
    plan = [
        ("a", 0, 5, 7, 2), ("b", 1, 5, 7, 3),
        ("c", 2, 9, 4, 2), ("d", 3, 9, 4, 3),
    ]
    host = _RecHostPool()
    n = l15_refill.refill(plan, host, SimpleNamespace(), page_tokens=4)
    assert n == 4
    assert host.calls == [([5, 9], [0, 1, 2, 3], [2, 3])]


def test_refill_p4_differing_lane_sets_refuse_named():
    plan = [
        ("a", 0, 5, 7, 2), ("b", 1, 5, 7, 3),
        ("c", 2, 9, 4, 0),  # page 9 owns lane 0 instead -> not expressible
    ]
    host = _RecHostPool()
    try:
        l15_refill.refill(plan, host, SimpleNamespace(), page_tokens=4)
        raise AssertionError("expected L15RefillError")
    except l15_refill.L15RefillError as exc:
        assert "lane" in str(exc).lower()
    assert host.calls == []


def test_refill_p4_needs_lane_tagged_rows():
    host = _RecHostPool()
    try:
        l15_refill.refill(
            [("a", 0, 5, 7)], host, SimpleNamespace(), page_tokens=4
        )
        raise AssertionError("expected L15RefillError")
    except l15_refill.L15RefillError as exc:
        assert "lane" in str(exc).lower()
    assert host.calls == []


def test_refill_p4_loader_failure_named_raise():
    host = _RecHostPool(raise_exc=True)
    try:
        l15_refill.refill(
            [("a", 0, 5, 7, 0)], host, SimpleNamespace(), page_tokens=4
        )
        raise AssertionError("expected L15RefillError")
    except l15_refill.L15RefillError:
        pass


# --------------------------------------------------- retain l2_lanes kwarg


def test_retain_writes_l2_lanes(tmp_path):
    mod = _load_retain_test_module()
    sc = mod.make_scenario(tmp_path, [])
    sc["kwargs"]["l2_lanes_of"] = lambda rid: (0, 1, 2, 3)
    from flliper.srt.pdflip import l15_retain

    res = l15_retain.retain_at_sleep(**sc["kwargs"])
    assert res is not None, f"retain skipped: {sc['log_lines']}"
    m = l15_manifest.read(sc["manifest_path"], pid_alive=lambda _pid: True)
    for s in m.spans:
        assert s.l2_lanes == (0, 1, 2, 3)


def test_retain_without_l2_lanes_of_keeps_empty(tmp_path):
    mod = _load_retain_test_module()
    sc = mod.make_scenario(tmp_path, [])
    from flliper.srt.pdflip import l15_retain

    res = l15_retain.retain_at_sleep(**sc["kwargs"])
    assert res is not None, f"retain skipped: {sc['log_lines']}"
    m = l15_manifest.read(sc["manifest_path"], pid_alive=lambda _pid: True)
    for s in m.spans:
        assert s.l2_lanes == ()
