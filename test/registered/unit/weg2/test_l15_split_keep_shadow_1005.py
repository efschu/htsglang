# SPDX-License-Identifier: Apache-2.0
"""L15-SPLIT-SHADOW + L15-KEEP-SHADOW + TREE-CAND-DIAG-ALL (desk 2025, kurz2 20:01-20:05Z).

Specimen (boot ..._53798f4686_1005_195833, D log): the cap-0 rank TP0 offered 1-2 of the 4-7
device tips TP1/TP2 offered at every sleep after the first session; no session tip was ever
agreed twice, ``matched`` of every k=1 turn stayed 44. Two code-derived causes, both rank-local:

(i)  SPLIT: a later session's store load (``matched=57`` against the held chain of the earlier
     one) splits the store-loaded node. ``_split_node`` hands ``host_value`` to both halves
     (``FullComponent.redistribute_on_node_split``) and ``l3_present`` (fnFL2x105) but not the
     L2 shadow attribute: the new parent has no KV host row and no shadow (``l2_backed`` False
     for every chain through it; the sweep skips it, ``l3_present``) and the child's shadow is
     longer than its tokens (``l15_bind`` refuses it). Log: ``first_miss_tok=46/48/58`` (the
     root-most missing node ends right behind the 44-token head), ``head_miss=4 of 4``,
     ``shadow_len_mismatch=4``.
(ii) KEEP: ``reset_keep`` nulls ``Full.host_value`` of every kept node; the publish sweep skips
     ``l3_present`` nodes: at the next sleep the held tip has no KV host row on the cap-0 rank.

Fixes (default off, behind the L15 master, group D, not dual): ``SGLANG_WEG2_L15_SPLIT_SHADOW``
splits the shadow at ``split_len`` like ``host_value``; ``SGLANG_WEG2_L15_KEEP_SHADOW`` records the
about-to-be-nulled host rows as the shadow in ``reset_keep``. Log-only:
``SGLANG_WEG2_L15_TREE_CAND_DIAG_ALL`` (census at every vote) with ``tip_miss`` / ``anc_miss``.

The tests drive the REAL ``UnifiedRadixCache`` (insert -> split, reset_keep) and the real
``l15_tree_cand.l2_backed`` / ``l15_bind.chain_host_rows_ex``. Hermetic, no CUDA. Run:
  cd /spinning/wt-27b-l15-econsume-1005 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/weg2/test_l15_split_keep_shadow_1005.py
"""
from __future__ import annotations

import inspect
import os
import pathlib
import sys
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "mem_cache"))

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.weg2 import l15_bind, l15_plan, l15_tree_cand, park_l3  # noqa: E402

ROOT = pathlib.Path(l15_plan.__file__).resolve().parents[1]  # python/sglang/srt
URC = (ROOT / "mem_cache" / "unified_radix_cache.py").read_text()
ENVIRON = (ROOT / "environ.py").read_text()

ON = {
    "SGLANG_WEG2_L15": "1",
    "SGLANG_WEG2_L15_SPLIT_SHADOW": "1",
    "SGLANG_WEG2_L15_KEEP_SHADOW": "1",
    "SGLANG_WEG2_GROUP": "D",
}
ATTR = park_l3.L2_SHADOW_ATTR


# ------------------------------------------------------------------ predicates

def _matrix(fn, key):
    base = dict(ON)
    cases = [(dict(base), True)]
    for drop in ("SGLANG_WEG2_L15", key, "SGLANG_WEG2_GROUP"):
        e = dict(base)
        e.pop(drop)
        cases.append((e, False))
    for bad in ("0", "", "no", "off", "false"):
        e = dict(base)
        e[key] = bad
        cases.append((e, False))
    for grp in ("P", "", "x"):
        e = dict(base)
        e["SGLANG_WEG2_GROUP"] = grp
        cases.append((e, False))
    e = dict(base)
    e["SGLANG_WEG2_DUAL_LAYOUT"] = "1"
    cases.append((e, False))
    e = dict(base)
    e["SGLANG_WEG2_GROUP"] = " d "
    cases.append((e, True))
    cases.append(({}, False))
    return [(c, want, fn(c)) for c, want in cases if fn(c) != want]


def test_split_predicate_matrix():
    assert _matrix(l15_plan.split_shadow_active, "SGLANG_WEG2_L15_SPLIT_SHADOW") == []


def test_keep_predicate_matrix():
    assert _matrix(l15_plan.keep_shadow_active, "SGLANG_WEG2_L15_KEEP_SHADOW") == []


@pytest.mark.parametrize("name", ["SGLANG_WEG2_L15_SPLIT_SHADOW", "SGLANG_WEG2_L15_KEEP_SHADOW",
                                  "SGLANG_WEG2_L15_TREE_CAND_DIAG_ALL"])
def test_env_declared_default_off(name):
    assert "%s = EnvBool(False)" % name in ENVIRON


def _mutant(fn, old, new):
    src = inspect.getsource(fn)
    assert old in src, old
    ns = {"Mapping": __import__("typing").Mapping, "_switch": l15_plan._switch,
          "master_on": l15_plan.master_on, "SPLIT_SHADOW_ENV": l15_plan.SPLIT_SHADOW_ENV,
          "KEEP_SHADOW_ENV": l15_plan.KEEP_SHADOW_ENV}
    exec(src.replace(old, new), ns)
    return ns[fn.__name__]


@pytest.mark.parametrize("fn,key", [(l15_plan.split_shadow_active, "SGLANG_WEG2_L15_SPLIT_SHADOW"),
                                    (l15_plan.keep_shadow_active, "SGLANG_WEG2_L15_KEEP_SHADOW")])
@pytest.mark.parametrize("old,new", [
    ("master_on(env)\n        and ", "True\n        and "),
    ('group == "D"', "True"),
    ('dual != "1"', "True"),
])
def test_predicate_mutants_die(fn, key, old, new):
    assert _matrix(_mutant(fn, old, new), key), (old, new)


def test_predicate_mutant_switch_ignored_dies():
    for fn, key, sw in ((l15_plan.split_shadow_active, "SGLANG_WEG2_L15_SPLIT_SHADOW", "SPLIT_SHADOW_ENV"),
                        (l15_plan.keep_shadow_active, "SGLANG_WEG2_L15_KEEP_SHADOW", "KEEP_SHADOW_ENV")):
        m = _mutant(fn, "_switch(env, %s)" % sw, "True")
        assert _matrix(m, key)


# ------------------------------------------------------------------ real cache fixtures

def _cache():
    from unittest.mock import patch as _p

    from test_unified_radix_cache_unittest import CacheConfig, build_fixture

    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    with _p("test_unified_radix_cache_unittest.get_device", return_value="cpu"):
        return build_fixture(CacheConfig(page_size=1, components=(ComponentType.FULL,)))


def _ins(cache, allocator, toks):
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache.insert(InsertParams(key=RadixKey(list(toks), None),
                              value=allocator.alloc(len(toks)).to(dtype=torch.int64)))


def _walk(cache, toks):
    """The node chain root -> the node whose key ends the token list (exact node boundaries)."""
    node, pos, out = cache.root_node, 0, []
    while pos < len(toks):
        nxt = None
        for ch in node.children.values():
            k = list(ch.key.token_ids)
            if k and k[0] == toks[pos]:
                nxt = ch
                break
        assert nxt is not None, (pos, toks[:pos])
        out.append(nxt)
        pos += len(nxt.key)
        node = nxt
    return out


class _View:
    """A real node seen through the cap-0 rank's eyes: Full as is, plus a published mamba host row.

    The FULL-only fixture has no Mamba component; ``l2_backed`` / ``loss_census`` read
    ``component_data[MAMBA].host_value`` of the TIP. Everything else (Full.host_value, the
    ``_weg2_l2_shadow`` attribute, key, parent) is the real node's, read live."""

    def __init__(self, node, root):
        from types import SimpleNamespace

        from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        self._n = node
        self._root = root
        self.component_data = {
            ComponentType.FULL: node.component_data[ComponentType.FULL],
            ComponentType.MAMBA: SimpleNamespace(host_value=torch.arange(1, 2, dtype=torch.int64)),
        }

    @property
    def parent(self):
        p = self._n.parent
        return p if p is None or p is self._root else _View(p, self._root)

    def __getattr__(self, name):
        return getattr(self._n, name)


def _view(cache, node):
    return _View(node, cache.root_node)


def _backed(cache, node):
    return l15_tree_cand.l2_backed(_view(cache, node), cache.root_node)


def _census(cache, node, require_l2=True):
    with patch.object(l15_tree_cand, "tips_of", lambda tc: [_view(cache, node)]):
        return l15_tree_cand.loss_census(cache, require_l2)


def _scenario(cache, allocator):
    """The kurz2 shape: head(40) + loaded session node (shadow rows, gens) + published ext node
    (host rows); then ANOTHER session whose prompt shares 56 tokens splits the loaded node."""
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    s0 = list(range(1, 41))
    _ins(cache, allocator, s0)
    (loaded,) = _walk(cache, s0)
    loaded.l3_present = True  # its pages came out of the store (transit rows released)
    setattr(loaded, ATTR, (tuple(range(1000, 1040)), tuple(range(5000, 5040))))
    ext_toks = s0 + list(range(41, 51))
    _ins(cache, allocator, ext_toks)
    ext = _walk(cache, ext_toks)[-1]
    ext.component_data[ComponentType.FULL].host_value = torch.arange(2000, 2010, dtype=torch.int64)
    ext.l3_present = True
    return s0, loaded, ext_toks, ext


def _second_session(cache, allocator):
    _ins(cache, allocator, list(range(1, 17)) + list(range(101, 125)))  # shares 16 tokens


def test_split_hands_the_shadow_to_both_halves():
    cache, allocator, _ = _cache()
    s0, loaded, ext_toks, ext = _scenario(cache, allocator)
    with patch.dict(os.environ, ON, clear=False):
        _second_session(cache, allocator)
    chain = _walk(cache, ext_toks)
    parent, child, e = chain
    assert len(parent.key) == 16 and len(child.key) == 24 and e is ext
    assert getattr(parent, ATTR) == (tuple(range(1000, 1016)), tuple(range(5000, 5016)))
    assert getattr(child, ATTR) == (tuple(range(1016, 1040)), tuple(range(5016, 5040)))
    assert child is loaded and parent.l3_present  # l3_present as before (fnFL2x105)


def test_split_makes_the_chain_backed_for_tree_cand_and_bind():
    cache, allocator, _ = _cache()
    s0, loaded, ext_toks, ext = _scenario(cache, allocator)
    assert _backed(cache, ext)  # before the split: backed
    with patch.dict(os.environ, ON, clear=False):
        _second_session(cache, allocator)
    assert _backed(cache, ext)
    rows, rec = l15_bind.chain_host_rows_ex(ext)
    assert rows == tuple(range(1000, 1040)) + tuple(range(2000, 2010))
    assert -1 not in rows
    assert rec[:40] == tuple(range(5000, 5040))


def test_red_without_the_switch_the_split_parent_is_unbacked():
    """The specimen: switch off = today's _split_node. The chain is lost for the cap-0 rank."""
    cache, allocator, _ = _cache()
    s0, loaded, ext_toks, ext = _scenario(cache, allocator)
    _second_session(cache, allocator)  # no env at all
    parent, child, e = _walk(cache, ext_toks)
    assert getattr(parent, ATTR, None) is None
    assert len(getattr(child, ATTR)[0]) == 40 and len(child.key) == 24  # longer than its tokens
    assert not _backed(cache, ext)
    rows, _rec = l15_bind.chain_host_rows_ex(ext)
    assert -1 in rows


@pytest.mark.parametrize("drop", ["SGLANG_WEG2_L15", "SGLANG_WEG2_L15_SPLIT_SHADOW"])
def test_split_gate_off_leaves_split_unchanged(drop):
    cache, allocator, _ = _cache()
    s0, loaded, ext_toks, ext = _scenario(cache, allocator)
    env = dict(ON)
    env.pop(drop)
    with patch.dict(os.environ, env, clear=True):
        _second_session(cache, allocator)
    parent, child, _e = _walk(cache, ext_toks)
    assert getattr(parent, ATTR, None) is None and len(getattr(child, ATTR)[0]) == 40


def test_split_group_p_and_dual_leave_split_unchanged():
    for extra in ({"SGLANG_WEG2_GROUP": "P"}, {"SGLANG_WEG2_DUAL_LAYOUT": "1"}):
        cache, allocator, _ = _cache()
        s0, loaded, ext_toks, ext = _scenario(cache, allocator)
        with patch.dict(os.environ, {**ON, **extra}, clear=True):
            _second_session(cache, allocator)
        parent, child, _e = _walk(cache, ext_toks)
        assert getattr(parent, ATTR, None) is None


def test_split_of_a_node_without_shadow_is_unchanged_and_host_value_still_splits():
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    cache, allocator, _ = _cache()
    toks = list(range(1, 41))
    _ins(cache, allocator, toks)
    (n,) = _walk(cache, toks)
    n.component_data[ComponentType.FULL].host_value = torch.arange(300, 340, dtype=torch.int64)
    with patch.dict(os.environ, ON, clear=False):
        _second_session(cache, allocator)
    parent, child = _walk(cache, toks)
    assert getattr(parent, ATTR, None) is None and getattr(child, ATTR, None) is None
    assert parent.component_data[ComponentType.FULL].host_value.tolist() == list(range(300, 316))
    assert child.component_data[ComponentType.FULL].host_value.tolist() == list(range(316, 340))


# ---- split_l2_shadow on its own (inconsistent shadows stay as they were) ----------------

class _N:
    def __init__(self, n, sh=None):
        self.key = list(range(n))
        if sh is not None:
            setattr(self, ATTR, sh)


def test_split_l2_shadow_unit():
    parent, child = _N(0), _N(7, ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), tuple(range(10))))
    assert park_l3.split_l2_shadow(parent, child, 3) is True
    assert getattr(parent, ATTR) == ((1, 2, 3), (0, 1, 2))
    assert getattr(child, ATTR) == ((4, 5, 6, 7, 8, 9, 10), tuple(range(3, 10)))


@pytest.mark.parametrize("sh,n_child", [
    (None, 7),
    (((1, 2, 3), (0, 1, 2)), 7),                         # not the node's whole span
    (((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), (0, 1)), 7),      # rows/gens differ
    ((1, 2, 3), 7),                                      # not a pair
])
def test_split_l2_shadow_inconsistent_untouched(sh, n_child):
    parent, child = _N(0), _N(n_child, sh)
    before = getattr(child, ATTR, None)
    assert park_l3.split_l2_shadow(parent, child, 3) is False
    assert getattr(parent, ATTR, None) is None
    assert getattr(child, ATTR, None) is before


def test_split_l2_shadow_never_raises():
    assert park_l3.split_l2_shadow(object(), object(), 3) is False

    class Bad:
        key = None

    setattr(Bad, ATTR, ((1,), (0,)))
    assert park_l3.split_l2_shadow(_N(0), Bad(), 1) is False


@pytest.mark.parametrize("old,new", [
    ("rows[:k]), tuple(gens[:k])", "rows[k:]), tuple(gens[:k])"),                  # parent gets the wrong half
    ("tuple(rows[k:]), tuple(gens[k:])", "tuple(rows), tuple(gens)"),             # child not truncated
    ("or len(rows) != k + len(child.key)", ""),                                    # length guard removed
    ("len(rows) != len(gens) or ", ""),                                            # pair guard removed
])
def test_split_l2_shadow_mutants_die(old, new):
    src = inspect.getsource(park_l3.split_l2_shadow)
    assert old in src, old
    ns = {"L2_SHADOW_ATTR": ATTR}
    exec(src.replace(old, new), ns)
    mut = ns["split_l2_shadow"]
    ok = True
    try:
        p, c = _N(0), _N(7, ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), tuple(range(10))))
        mut(p, c, 3)
        ok = (getattr(p, ATTR) == ((1, 2, 3), (0, 1, 2))
              and getattr(c, ATTR) == ((4, 5, 6, 7, 8, 9, 10), tuple(range(3, 10))))
        p2, c2 = _N(0), _N(7, ((1, 2, 3), (0, 1, 2)))
        mut(p2, c2, 3)
        ok = ok and getattr(p2, ATTR, None) is None
        p3, c3 = _N(0), _N(7, ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), (0, 1)))
        mut(p3, c3, 3)
        ok = ok and getattr(p3, ATTR, None) is None
    except Exception:
        ok = False
    assert not ok, (old, new)


# ------------------------------------------------------------------ KEEP: reset_keep

class _Pool:
    """Arena KV host pool stand-in: the three members ``record_l2_shadow`` reads."""

    staging_rows = 100
    _arena_page_tokens = 1

    def slot_gens(self, want):
        return [7000 + int(s) for s in want]


def _keep_scenario():
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    cache, allocator, _ = _cache()
    s0, loaded, ext_toks, ext = _scenario(cache, allocator)
    # the head chain nodes of the tip carry host rows on the arena (rows >= staging_rows)
    loaded.component_data[ComponentType.FULL].host_value = None
    ext.component_data[ComponentType.FULL].host_value = torch.arange(2000, 2010, dtype=torch.int64)
    cache._weg2_arena_pools = lambda: {ComponentType.FULL: _Pool()}
    return cache, ext, loaded


def test_keep_shadow_gives_the_kept_tip_an_l2_identity_red_without_it():
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    cache, ext, loaded = _keep_scenario()
    # hold the tip's pending 'mamba host' (== Full.host_value in this FULL-only fixture) check out of
    # the question: after reset_keep it is None, so judge the KV chain half via chain_host_rows_ex
    env_off = {k: v for k, v in ON.items() if k != "SGLANG_WEG2_L15_KEEP_SHADOW"}
    with patch.dict(os.environ, env_off, clear=True), patch.object(park_l3, "enabled", lambda: True):
        cache.reset_keep([ext])
    assert ext.component_data[ComponentType.FULL].host_value is None  # reset_keep contract
    rows_off, _ = l15_bind.chain_host_rows_ex(ext)
    assert -1 in rows_off[40:], "red: the ext node has no L2 identity after reset_keep"
    assert getattr(ext, ATTR, None) is None

    cache, ext, loaded = _keep_scenario()
    with patch.dict(os.environ, ON, clear=True), patch.object(park_l3, "enabled", lambda: True):
        cache.reset_keep([ext])
    assert ext.component_data[ComponentType.FULL].host_value is None
    assert getattr(ext, ATTR) == (tuple(range(2000, 2010)), tuple(7000 + (r - 100) for r in range(2000, 2010)))
    rows_on, rec_on = l15_bind.chain_host_rows_ex(ext)
    assert rows_on == tuple(range(1000, 1040)) + tuple(range(2000, 2010)) and -1 not in rows_on
    assert rec_on[40:] == tuple(7000 + (r - 100) for r in range(2000, 2010))


def test_keep_shadow_chain_passes_l2_backed_only_with_the_switch():
    """The specimen of cause (ii): the held tip after reset_keep, judged by the real l2_backed."""
    env_off = {k: v for k, v in ON.items() if k != "SGLANG_WEG2_L15_KEEP_SHADOW"}
    for env, want in ((env_off, False), (ON, True)):
        cache, ext, _loaded = _keep_scenario()
        assert _backed(cache, ext)  # before the reset_keep: ext still has its published host rows
        with patch.dict(os.environ, env, clear=True), patch.object(park_l3, "enabled", lambda: True):
            cache.reset_keep([ext])
        assert _backed(cache, ext) is want, env


@pytest.mark.parametrize("drop", ["SGLANG_WEG2_L15", "SGLANG_WEG2_L15_KEEP_SHADOW", "SGLANG_WEG2_GROUP"])
def test_keep_gate_off_records_nothing(drop):
    cache, ext, loaded = _keep_scenario()
    env = dict(ON)
    env.pop(drop)
    with patch.dict(os.environ, env, clear=True), patch.object(park_l3, "enabled", lambda: True):
        cache.reset_keep([ext])
    assert getattr(ext, ATTR, None) is None


def test_keep_requires_park_l3_enabled_and_an_arena_pool():
    cache, ext, loaded = _keep_scenario()
    with patch.dict(os.environ, ON, clear=True), patch.object(park_l3, "enabled", lambda: False):
        cache.reset_keep([ext])
    assert getattr(ext, ATTR, None) is None
    cache, ext, loaded = _keep_scenario()
    cache._weg2_arena_pools = lambda: {}
    with patch.dict(os.environ, ON, clear=True), patch.object(park_l3, "enabled", lambda: True):
        cache.reset_keep([ext])
    assert getattr(ext, ATTR, None) is None


def test_keep_skips_write_pending_nodes_and_never_raises():
    cache, ext, loaded = _keep_scenario()
    ext.write_through_pending_id = 5
    with patch.dict(os.environ, ON, clear=True), patch.object(park_l3, "enabled", lambda: True):
        cache.reset_keep([ext])
    assert getattr(ext, ATTR, None) is None

    class Boom:
        def _weg2_arena_pools(self):
            raise RuntimeError("boom")

    with patch.dict(os.environ, ON, clear=True), patch.object(park_l3, "enabled", lambda: True):
        assert park_l3.record_keep_shadow(Boom(), [object()]) == 0


@pytest.mark.parametrize("old,new", [
    ("if not enabled() or not _group_d():\n            return 0\n        from sglang.srt.weg2 import l15_plan\n",
     "from sglang.srt.weg2 import l15_plan\n"),
    ("if not l15_plan.keep_shadow_active(os.environ):\n            return 0", "pass"),
    ('if getattr(node, "write_through_pending_id", None) is not None:\n                continue',
     "pass"),
    ("pool = tree._weg2_arena_pools().get(BASE_COMPONENT_TYPE)\n        if pool is None:\n            return 0",
     "pool = _Pool()"),
])
def test_keep_mutants_die(old, new):
    """Each guard is load-bearing: the mutated copy of record_keep_shadow misbehaves in at least
    one of the gate scenarios the tests above pin."""
    src = inspect.getsource(park_l3.record_keep_shadow)
    assert old in src, old
    ns = dict(vars(park_l3))
    ns["_Pool"] = _Pool
    exec(src.replace(old, new), ns)
    mut = ns["record_keep_shadow"]

    def run(env, enabled=True, arena=True, pending=False):
        cache, ext, loaded = _keep_scenario()
        if not arena:
            cache._weg2_arena_pools = lambda: {}
        if pending:
            ext.write_through_pending_id = 5
        with patch.dict(os.environ, env, clear=True), \
                patch.object(park_l3, "enabled", lambda: enabled), patch.dict(ns, {"enabled": lambda: enabled}):
            keep = {ext, loaded}
            mut(cache, keep)
        return getattr(ext, ATTR, None) is not None

    assert run(ON) is True  # the mutant still works in the good case ...
    wrong = (
        run({k: v for k, v in ON.items() if k != "SGLANG_WEG2_L15_KEEP_SHADOW"}) is not False
        or run(ON, enabled=False) is not False
        or run({k: v for k, v in ON.items() if k != "SGLANG_WEG2_GROUP"}) is not False
        or run(ON, arena=False) is not False
        or run(ON, pending=True) is not False
    )
    assert wrong, (old, new)  # ... and misbehaves in a gated case


# ------------------------------------------------------------------ DIAG-ALL / tip_miss / anc_miss

def test_diag_all_default_off_and_only_with_its_own_switch():
    assert l15_tree_cand.diag_all({}) is False
    assert l15_tree_cand.diag_all({"SGLANG_WEG2_L15_TREE_CAND_DIAG": "1"}) is False
    for v in ("1", "true", "on"):
        assert l15_tree_cand.diag_all({"SGLANG_WEG2_L15_TREE_CAND_DIAG_ALL": v}) is True
    assert l15_tree_cand.diag_all({"SGLANG_WEG2_L15_TREE_CAND_DIAG_ALL": "0"}) is False


def _build_logs(env):
    cache, allocator, _ = _cache()
    s0, loaded, ext_toks, ext = _scenario(cache, allocator)
    lines = []
    with patch.object(l15_tree_cand, "tips_of", lambda tc: [_view(cache, ext)]):
        l15_tree_cand.build(cache, lambda local: [list(local)], 0, env=env, log=lines.append,
                            require_l2=True)
    return lines


def test_build_logs_the_census_at_every_vote_only_with_diag_all():
    base = {"SGLANG_WEG2_L15_TREE_CAND_DIAG": "1"}
    quiet = _build_logs(base)
    assert any("L15-TREE-CAND local=1 agreed=1" in ln for ln in quiet)
    assert not any("TREE-CAND-LOSS" in ln for ln in quiet)  # agreed>=1: silent, as in kurz2
    loud = _build_logs({**base, "SGLANG_WEG2_L15_TREE_CAND_DIAG_ALL": "1"})
    assert any("TREE-CAND-LOSS" in ln and "tip_miss=0 anc_miss=0" in ln for ln in loud)
    off = _build_logs({"SGLANG_WEG2_L15_TREE_CAND_DIAG_ALL": "1"})
    assert not any("TREE-CAND-LOSS" in ln for ln in off)  # needs the DIAG switch too


def test_census_names_anc_miss_for_the_split_specimen():
    cache, allocator, _ = _cache()
    s0, loaded, ext_toks, ext = _scenario(cache, allocator)
    _second_session(cache, allocator)  # switch off: the split parent has no backing
    c = _census(cache, ext)
    assert (c["no_kv_host"], c["anc_miss"], c["tip_miss"]) == (1, 1, 0)
    assert c["shadow_len_mismatch"] == 1 and c["head_miss"] == 1 and c["l2_ok"] == 0
    assert c["first_miss_tok"] == [16]
    line = l15_tree_cand.loss_line(0, c, 0, 0, True)
    assert "tip_miss=0 anc_miss=1" in line and "first_miss_tok=16" in line


def test_census_names_tip_miss_for_the_held_tip_specimen():
    cache, ext, _loaded = _keep_scenario()
    env_off = {k: v for k, v in ON.items() if k != "SGLANG_WEG2_L15_KEEP_SHADOW"}
    with patch.dict(os.environ, env_off, clear=True), patch.object(park_l3, "enabled", lambda: True):
        cache.reset_keep([ext])
    c = _census(cache, ext)
    assert (c["no_kv_host"], c["tip_miss"], c["anc_miss"]) == (1, 1, 0)
    assert c["first_miss_tok"] == [50]  # the ext node ends at 50 (40 loaded + 10)


def test_census_is_clean_after_both_fixes():
    cache, allocator, _ = _cache()
    s0, loaded, ext_toks, ext = _scenario(cache, allocator)
    with patch.dict(os.environ, ON, clear=True), patch.object(park_l3, "enabled", lambda: True):
        _second_session(cache, allocator)
        from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        cache._weg2_arena_pools = lambda: {ComponentType.FULL: _Pool()}
        cache.reset_keep([ext])
    c = _census(cache, ext)
    assert (c["no_kv_host"], c["l2_ok"], c["kept"]) == (0, 1, 1)
    assert _backed(cache, ext)


def test_hooks_sit_where_the_fix_says():
    split = URC[URC.index("    def _split_node("):URC.index("    def _touch_node(")]
    assert split.index("redistribute_on_node_split") < split.index("split_shadow_active(os.environ)") \
        < split.index("split_l2_shadow(new_node, child, split_len)") \
        < split.index("new_node.parent.children[key.child_key(self.page_size)] = new_node")
    keep = URC[URC.index("    def reset_keep("):URC.index("    def weg2_node_depth(")]
    assert keep.index("keep_set.add(cur)") < keep.index("record_keep_shadow(self, keep_set)") \
        < keep.index("self._reset_full()\n        _t2") < keep.index("cd.host_value = None")
