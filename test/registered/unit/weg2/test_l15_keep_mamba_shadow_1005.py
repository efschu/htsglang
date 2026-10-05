# SPDX-License-Identifier: Apache-2.0
"""L15-KEEP-MAMBA-SHADOW (desk 2025, kurz3 21:34-21:38Z, image l15e = SPLIT_SHADOW + KEEP_SHADOW on).

Specimen (boot ..._6361b47420_1005_213043, D log, TP0, TREE-CAND-LOSS lines at every vote):
no_kv_host 0 and anc_miss 0 / tip_miss 0 (fixes A and B work), but ``no_mamba_host`` 28 of 55 tips, and
``WEG2-ANCHOR-LOST at=flush`` lists exactly the held tips ([768,768,768,19712], [..5..], ...): after
``reset_keep`` the kept nodes have ``Mamba.host_value = None``; the anchor-only backup that follows copies
the anchor into an arena slot "not in the tree" (Full has no host copy -- the tree invariant 'aux host
value requires Full host value' forbids a node-visible mamba host row there). So at the next sleep the
held tip has no mamba host row (``l15_tree_cand.l2_backed`` -> False) and no END-anchor L2 identity
(``l15_bind.anchor_host_row`` -1 -> ``manifest_vote`` refuses the round): no tip survives a second sleep.

Fix (default off, L15 master + group D + not dual): ``SGLANG_WEG2_L15_KEEP_MAMBA_SHADOW`` -- ``reset_keep``
records the about-to-be-nulled mamba host row + arena generation as ``_weg2_l2_mamba_shadow``;
``l2_backed`` and the census accept it; the bind adopts it only where the slot still carries that
generation (a re-claimed slot is a foreign anchor: (-1, -1)).

Hermetic, no CUDA. Run:
  cd /spinning/wt-27b-l15-econsume2-1005 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/weg2/test_l15_keep_mamba_shadow_1005.py
"""
from __future__ import annotations

import inspect
import os
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.weg2 import l15_plan, l15_tree_cand, park_l3  # noqa: E402
from sglang.srt.weg2.l15_bind import anchor_shadow, build_retain_kwargs  # noqa: E402

ROOT = pathlib.Path(l15_plan.__file__).resolve().parents[1]
ENVIRON = (ROOT / "environ.py").read_text()
URC = (ROOT / "mem_cache" / "unified_radix_cache.py").read_text()

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
MS = park_l3.MAMBA_SHADOW_ATTR
ON = {"SGLANG_WEG2_L15": "1", "SGLANG_WEG2_L15_KEEP_MAMBA_SHADOW": "1", "SGLANG_WEG2_GROUP": "D"}
KEY = "SGLANG_WEG2_L15_KEEP_MAMBA_SHADOW"


# ------------------------------------------------------------------ predicate

def _matrix(fn):
    cases = [(dict(ON), True)]
    for drop in ("SGLANG_WEG2_L15", KEY, "SGLANG_WEG2_GROUP"):
        e = dict(ON)
        e.pop(drop)
        cases.append((e, False))
    for bad in ("0", "", "no", "off", "false"):
        cases.append(({**ON, KEY: bad}, False))
    for grp in ("P", "", "x"):
        cases.append(({**ON, "SGLANG_WEG2_GROUP": grp}, False))
    cases.append(({**ON, "SGLANG_WEG2_DUAL_LAYOUT": "1"}, False))
    cases.append(({**ON, "SGLANG_WEG2_GROUP": " d "}, True))
    cases.append(({}, False))
    return [(c, want) for c, want in cases if fn(c) != want]


def test_predicate_matrix_and_default_off():
    assert _matrix(l15_plan.keep_mamba_shadow_active) == []
    assert "%s = EnvBool(False)" % KEY in ENVIRON


@pytest.mark.parametrize("old,new", [
    ("master_on(env)\n        and ", "True\n        and "),
    ('group == "D"', "True"),
    ('dual != "1"', "True"),
    ("_switch(env, KEEP_MAMBA_SHADOW_ENV)", "True"),
])
def test_predicate_mutants_die(old, new):
    src = inspect.getsource(l15_plan.keep_mamba_shadow_active)
    assert old in src, old
    ns = {"Mapping": __import__("typing").Mapping, "_switch": l15_plan._switch,
          "master_on": l15_plan.master_on, "KEEP_MAMBA_SHADOW_ENV": KEY}
    exec(src.replace(old, new), ns)
    assert _matrix(ns["keep_mamba_shadow_active"]), (old, new)


# ------------------------------------------------------------------ recording

class _KVPool:
    staging_rows = 100
    _arena_page_tokens = 1

    def slot_gens(self, want):
        return [7000 + int(s) for s in want]


class _MPool:
    staging_rows = 2

    def __init__(self, gens=None, boom=False):
        self._g = gens if gens is not None else {}
        self._boom = boom

    def slot_gens(self, slots):
        if self._boom:
            raise RuntimeError("boom")
        return [self._g.get(int(s), 9000 + int(s)) for s in slots]


def _node(mamba_row=9, kv_rows=(2000, 2001), pending=None, anchor=4):
    """A kept tip as the cache hands it to reset_keep: device anchor (value), host rows (host_value)."""
    def t(v):
        return None if v is None else torch.tensor(v if isinstance(v, (list, tuple)) else [v], dtype=torch.int64)

    return SimpleNamespace(
        parent=None, key=(0,), write_through_pending_id=pending, id=1,
        component_data={FULL: SimpleNamespace(value=None, host_value=t(list(kv_rows) if kv_rows else None)),
                        MAMBA: SimpleNamespace(value=t(anchor), host_value=t(mamba_row), host_lock_ref=0)})


class _Tree:
    def __init__(self, pools):
        self._pools = pools

    def _weg2_arena_pools(self):
        return self._pools


def _rec(nodes, env=ON, pools=None, enabled=True):
    pools = {FULL: _KVPool(), MAMBA: _MPool()} if pools is None else pools
    with patch.dict(os.environ, env, clear=True), patch.object(park_l3, "enabled", lambda: enabled):
        return park_l3.record_keep_shadow(_Tree(pools), nodes)


def test_records_row_and_generation():
    n = _node(mamba_row=9)
    assert _rec([n]) == 1
    assert getattr(n, MS) == (9, 9000 + 7)  # slot = row - staging_rows(2) = 7
    assert getattr(n, "_weg2_l2_shadow", None) is None  # the KV switch is a separate one


def test_both_switches_record_both_shadows():
    n = _node()
    env = {**ON, "SGLANG_WEG2_L15_KEEP_SHADOW": "1"}
    assert _rec([n], env=env) == 2 + 1
    assert getattr(n, "_weg2_l2_shadow") == ((2000, 2001), (7000 + 1900, 7000 + 1901))
    assert getattr(n, MS) == (9, 9007)


def test_staging_row_records_generation_minus_one():
    n = _node(mamba_row=1)  # below staging_rows=2: staging only, no L2 copy
    assert _rec([n]) == 1
    assert getattr(n, MS) == (1, -1)
    assert l15_tree_cand._mamba_shadow_ok(n) is False


@pytest.mark.parametrize("drop", ["SGLANG_WEG2_L15", KEY, "SGLANG_WEG2_GROUP"])
def test_gate_off_records_nothing(drop):
    env = dict(ON)
    env.pop(drop)
    n = _node()
    assert _rec([n], env=env) == 0 and getattr(n, MS, None) is None


def test_group_p_dual_park_off_no_mamba_pool_record_nothing():
    for env, kw in (({**ON, "SGLANG_WEG2_GROUP": "P"}, {}), ({**ON, "SGLANG_WEG2_DUAL_LAYOUT": "1"}, {}),
                    (ON, {"enabled": False}), (ON, {"pools": {FULL: _KVPool()}})):
        n = _node()
        assert _rec([n], env=env, **kw) == 0 and getattr(n, MS, None) is None, (env, kw)


def test_pending_and_hostless_nodes_are_skipped_and_errors_never_escape():
    pend, bare = _node(pending=5), _node(mamba_row=None)
    assert _rec([pend, bare]) == 0
    assert getattr(pend, MS, None) is None and getattr(bare, MS, None) is None
    n = _node()
    assert _rec([n], pools={FULL: _KVPool(), MAMBA: _MPool(boom=True)}) == 0
    assert getattr(n, MS, None) is None


def test_old_shadow_survives_a_node_without_host_row():
    n = _node(mamba_row=None)
    setattr(n, MS, (9, 9007))
    assert _rec([n]) == 0
    assert getattr(n, MS) == (9, 9007)


def _mut_record(old, new):
    src = inspect.getsource(park_l3.record_keep_shadow)
    assert old in src, old
    ns = dict(vars(park_l3))
    exec(src.replace(old, new), ns)
    return ns["record_keep_shadow"]


@pytest.mark.parametrize("old,new", [
    ("if mb_on:\n", "if True:\n"),
    ("if kv_on and pool is not None:", "if pool is not None:"),
    ("if not enabled() or not _group_d():\n            return 0", "pass"),
])
def test_record_mutants_die(old, new):
    mut = _mut_record(old, new)
    wrong = []
    kv_only = {"SGLANG_WEG2_L15": "1", "SGLANG_WEG2_L15_KEEP_SHADOW": "1", "SGLANG_WEG2_GROUP": "D"}
    for env, kw in ((kv_only, {}), (ON, {"enabled": False}), ({**ON, "SGLANG_WEG2_GROUP": "P"}, {}), (ON, {})):
        n = _node()
        pools = {FULL: _KVPool(), MAMBA: _MPool()}
        en = kw.get("enabled", True)
        with patch.dict(os.environ, env, clear=True), patch.object(park_l3, "enabled", lambda e=en: e), \
                patch.dict(mut.__globals__, {"enabled": lambda e=en: e}):
            mut(_Tree(pools), [n])
        # the unmutated function records: kv_only -> KV shadow only; ON -> mamba shadow only; gated -> nothing
        want_m = env is ON and en
        want_k = env is kv_only
        wrong.append((getattr(n, MS, None) is not None) != want_m or (getattr(n, "_weg2_l2_shadow", None) is not None) != want_k)
    assert any(wrong), (old, new)


def test_record_unmutated_matches_the_expectations_the_mutants_are_judged_by():
    kv_only = {"SGLANG_WEG2_L15": "1", "SGLANG_WEG2_L15_KEEP_SHADOW": "1", "SGLANG_WEG2_GROUP": "D"}
    n = _node()
    _rec([n], env=kv_only)
    assert getattr(n, MS, None) is None and getattr(n, "_weg2_l2_shadow", None) is not None
    n = _node()
    _rec([n], env=ON)
    assert getattr(n, MS, None) is not None and getattr(n, "_weg2_l2_shadow", None) is None


# ------------------------------------------------------------------ l2_backed / census

class _Root:
    pass


def _chain(shadow, mamba_host=None, kv_ok=True):
    """root -> A(KV host) -> TIP(KV host); the tip's mamba host row as given, plus the shadow."""
    root = _Root()

    def cd(kv, mh):
        def t(v):
            return None if v is None else torch.tensor(v, dtype=torch.int64)

        return {FULL: SimpleNamespace(host_value=t([1, 2] if kv else None)),
                MAMBA: SimpleNamespace(host_value=t(mh))}

    a = SimpleNamespace(parent=root, component_data=cd(kv_ok, None), key=SimpleNamespace(token_ids=[1, 2], is_bigram=False))
    tip = SimpleNamespace(parent=a, component_data=cd(kv_ok, mamba_host), key=SimpleNamespace(token_ids=[3, 4], is_bigram=False))
    if shadow is not None:
        setattr(tip, MS, shadow)
    return root, tip


@pytest.mark.parametrize("shadow,host,want", [
    (None, None, False),          # the specimen: held tip, mamba host nulled, nothing else
    ((9, 9007), None, True),      # shadow: row + gen known
    ((9, -1), None, False),       # staging-only / unknown generation: no identity
    ((-1, 5), None, False),
    ("junk", None, False),
    ((9,), None, False),
    (None, [9], True),            # a live host row as before
    ((9, 9007), [9], True),
])
def test_l2_backed_mamba_half(shadow, host, want):
    root, tip = _chain(shadow, host)
    assert l15_tree_cand.l2_backed(tip, root) is want


def test_census_counts_no_mamba_host_only_without_shadow():
    for shadow, bucket in ((None, "no_mamba_host"), ((9, 9007), "l2_ok")):
        root, tip = _chain(shadow, None)
        tree = SimpleNamespace(root_node=root, ongoing_write_through={})
        with patch.object(l15_tree_cand, "tips_of", lambda tc, t=tip: [t]):
            c = l15_tree_cand.loss_census(tree, True)
        assert c[bucket] == 1 and c["tips"] == 1, (shadow, c)


def _exec_variant(fn, old, new):
    src = inspect.getsource(fn)
    assert old in src, old
    ns = dict(vars(l15_tree_cand))
    exec(src.replace(old, new), ns)
    return ns[fn.__name__], ns


def test_l2_backed_mutant_without_shadow_acceptance_dies():
    mut, ns = _exec_variant(l15_tree_cand.l2_backed, "and not _mamba_shadow_ok(node)", "")
    root, tip = _chain((9, 9007), None)
    assert mut(tip, root) is False  # mutant misses the shadow -> the green case above would fail


def test_shadow_ok_mutants_die():
    gen_mut, _ = _exec_variant(l15_tree_cand._mamba_shadow_ok, "int(sh[0]) >= 0 and int(sh[1]) >= 0", "int(sh[0]) >= 0")
    assert gen_mut(SimpleNamespace(**{MS: (9, -1)})) is True  # mutant accepts an unknown generation
    assert l15_tree_cand._mamba_shadow_ok(SimpleNamespace(**{MS: (9, -1)})) is False
    len_mut, _ = _exec_variant(l15_tree_cand._mamba_shadow_ok, "len(sh) == 2 and ", "")
    with pytest.raises(IndexError):
        len_mut(SimpleNamespace(**{MS: (9,)}))
    assert l15_tree_cand._mamba_shadow_ok(SimpleNamespace(**{MS: (9,)})) is False


# ------------------------------------------------------------------ bind (anchor identity)

def _bnode(anchor, host_row, shadow=None):
    def t(v):
        return None if v is None else torch.tensor([v], dtype=torch.int64)

    n = SimpleNamespace(
        parent=None, key=(0,), tree_components=(FULL, MAMBA),
        component_data={FULL: SimpleNamespace(value=None, host_value=None, host_lock_ref=0),
                        MAMBA: SimpleNamespace(value=t(anchor), host_value=t(host_row), host_lock_ref=0)})
    if shadow is not None:
        setattr(n, MS, shadow)
    return n


def _breq(rid, pool_idx, in_len, anchor, node):
    return SimpleNamespace(rid=rid, req_pool_idx=pool_idx, origin_input_ids=list(range(in_len)), output_ids=[],
                           mamba_pool_idx=anchor, last_node=node, l15_kind="served", l15_last_active=1.0)


class _BindPool:
    def __init__(self, staging_rows, gens):
        self.staging_rows = staging_rows
        self._gens = gens

    def slot_gens(self, slots):
        return [self._gens.get(int(s), -1) for s in slots]


def _anchor_of(tmp_path, node, gens):
    rtt = torch.zeros(4, 8, dtype=torch.int64)
    rtt[0, :6] = torch.tensor([1, 2, 4, 6, 3, 9], dtype=torch.int64)
    bound = build_retain_kwargs(
        [_breq("r_seat", 0, 6, 4, node)], rtt, caps_rows_by_rank=(4, 10), cap_anchor_slots=5, prefix=(0, 1, 2),
        rank=1, epoch=77, pid=4242, kv_buffers=[], mamba_buffers=[], allocator=None,
        reset_keep=lambda ns: None, set_keep=lambda buf, spans: None, manifest_path=str(tmp_path / "m.json"),
        log=lambda s: None, mamba_host_pool=_BindPool(2, gens))
    return bound["anchor_l2_of"]("r_seat")


def test_bind_adopts_the_shadow_where_the_generation_matches(tmp_path):
    assert _anchor_of(tmp_path, _bnode(4, None, shadow=(9, 42)), {7: 42}) == (7, 42)


def test_bind_red_without_a_shadow_the_held_anchor_has_no_identity(tmp_path):
    """The specimen: reset_keep nulled the mamba host row, nothing recorded -> (-1, -1) ->
    manifest_vote 'END anchor without L2 identity' -> the round would be refused."""
    assert _anchor_of(tmp_path, _bnode(4, None), {7: 42}) == (-1, -1)


def test_bind_refuses_a_reclaimed_slot(tmp_path):
    assert _anchor_of(tmp_path, _bnode(4, None, shadow=(9, 42)), {7: 43}) == (-1, -1)  # gen bumped: foreign
    assert _anchor_of(tmp_path, _bnode(4, None, shadow=(9, 42)), {}) == (-1, -1)       # unknown gen


def test_bind_staging_shadow_and_live_row_precedence(tmp_path):
    assert _anchor_of(tmp_path, _bnode(4, None, shadow=(1, 5)), {7: 42}) == (-1, -1)  # row < staging_rows
    # a live host row wins over an (older) shadow, with its CURRENT generation as before
    assert _anchor_of(tmp_path, _bnode(4, 9, shadow=(8, 1)), {7: 42}) == (7, 42)


def test_anchor_shadow_unit():
    assert anchor_shadow(_bnode(4, None, shadow=(9, 42)), 4) == (9, 42)
    assert anchor_shadow(_bnode(4, None), 4) is None
    assert anchor_shadow(_bnode(4, 9, shadow=(9, 42)), 4) is None      # live row: anchor_host_row's job
    assert anchor_shadow(_bnode(4, None, shadow=(9, -1)), 4) is None
    assert anchor_shadow(_bnode(5, None, shadow=(9, 42)), 4) is None   # another slot's node


@pytest.mark.parametrize("old,new", [
    ("if _eg is not None and anchor_l2_by_rid[rid][1] != _eg:", "if False:"),
    ("_ash = anchor_shadow(_anode, _aslot)\n                if _ash is not None:", "_ash = None\n                if _ash is not None:"),
])
def test_bind_mutants_die(tmp_path, old, new):
    from sglang.srt.weg2 import l15_bind

    src = inspect.getsource(l15_bind.build_retain_kwargs)
    assert old in src, old
    ns = dict(vars(l15_bind))
    exec(src.replace(old, new), ns)
    mut = ns["build_retain_kwargs"]
    rtt = torch.zeros(4, 8, dtype=torch.int64)
    rtt[0, :6] = torch.tensor([1, 2, 4, 6, 3, 9], dtype=torch.int64)

    def run(node, gens):
        b = mut([_breq("r_seat", 0, 6, 4, node)], rtt, caps_rows_by_rank=(4, 10), cap_anchor_slots=5,
                prefix=(0, 1, 2), rank=1, epoch=77, pid=4242, kv_buffers=[], mamba_buffers=[], allocator=None,
                reset_keep=lambda ns_: None, set_keep=lambda buf, spans: None, manifest_path=str(tmp_path / "m2.json"),
                log=lambda s: None, mamba_host_pool=_BindPool(2, gens))
        return b["anchor_l2_of"]("r_seat")

    good = run(_bnode(4, None, shadow=(9, 42)), {7: 42}) == (7, 42)
    stale_ok = run(_bnode(4, None, shadow=(9, 42)), {7: 43}) == (-1, -1)
    assert not (good and stale_ok), (old, new)


def test_hook_sits_in_reset_keep_before_the_nulling():
    keep = URC[URC.index("    def reset_keep("):URC.index("    def weg2_node_depth(")]
    assert keep.index("keep_set.add(cur)") < keep.index("record_keep_shadow(self, keep_set)") \
        < keep.index("self._reset_full()\n        _t2") < keep.index("cd.host_value = None")
