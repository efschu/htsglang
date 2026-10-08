# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-DUPKW: the scheduler retain hook's CALL SHAPE.

The first L15=1 boot died on "retain_at_sleep() got multiple values for
keyword argument 'mamba_allocator'": the hook passed mamba_allocator
explicitly AND through **build_retain_kwargs(), whose dict always
carries the key. This test pins the call site (not l15_retain alone):
AST over the real scheduler.py, the REAL build_retain_kwargs with
fakes, and a signature bind of the real retain_at_sleep.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flliper.srt.pdflip import l15_bind, l15_retain  # noqa: E402
from test_pdflip_l15_bind_1001 import _req, _req_to_token  # noqa: E402

SCHED = REPO / "python" / "flliper" / "srt" / "managers" / "scheduler.py"

# fake values for every keyword the hook hands to build_retain_kwargs
# (shapes reused from test_pdflip_l15_bind_1001)
_FAKE_KW = {
    "caps_rows_by_rank": (100, 100),
    "cap_anchor_slots": 10,
    "prefix": [0, 1, 7],
    "rank": 0,
    "epoch": 1,
    "pid": 1,
    "kv_buffers": [],
    "mamba_buffers": [],
    "allocator": None,
    "reset_keep": lambda _ns: None,
    "set_keep": lambda _b, _s: None,
    "manifest_path": tempfile.mkdtemp() + "/m.json",
    "log": lambda _msg: None,
    "mamba_allocator": None,
    "host_pool": None,
    "mamba_host_pool": None,
    # L15-HOSTLOCK: sink recording the sleep's arena refs (master on only)
    "hold_sink": lambda _rec: None,
    # L15-FIX-PARKED: the hook passes the radix tree for parked reqs
    "tree_cache": None,
    # L15-TREE-CAND: cap on the finished-request tips
    "tree_cand_max": 8,
    # L15-TREE-DISAGREE: the tips are agreed + probed before the bind
    "tree_agreed": True,
}


def _reqs():
    return [
        _req("r_seat", 0, 4, 2, 4, object(), "seat", 2.0),
        _req("r_parked", 1, 1, 0, 5, object(), "parked", 1.0),
    ]


def _flush_cache(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "flush_cache":
            return node
    raise AssertionError("no flush_cache method in scheduler.py")


def _call_in(func_node, dotted):
    name, attr = dotted
    hits = [
        c
        for c in ast.walk(func_node)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == attr
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == name
    ]
    assert len(hits) == 1, "expected exactly one %s call, got %d" % (
        dotted, len(hits),
    )
    return hits[0]


def test_retain_at_sleep_call_shape_has_no_duplicate_keywords():
    tree = ast.parse(SCHED.read_text())
    flush = _flush_cache(tree)
    retain = _call_in(flush, ("l15_retain", "retain_at_sleep"))
    retain_kw = [kw.arg for kw in retain.keywords if kw.arg is not None]
    assert any(kw.arg is None for kw in retain.keywords), (
        "the hook must pass **kwargs (that is where the builder dict goes)"
    )
    build = _call_in(flush, ("l15_bind", "build_retain_kwargs"))
    build_kw = {kw.arg for kw in build.keywords if kw.arg is not None}
    kwargs = {name: _FAKE_KW[name] for name in sorted(build_kw)}
    bound = l15_bind.build_retain_kwargs(_reqs(), _req_to_token(), **kwargs)
    # the real builder's dict keys may not collide with the explicit ones
    dup = set(retain_kw) & set(bound)
    assert not dup, "duplicate keyword(s) %s: the hook would die with a " \
        "TypeError exactly like the first L15=1 boot" % (sorted(dup),)
    # every required parameter supplied exactly once
    inspect.signature(l15_retain.retain_at_sleep).bind(
        **{k: object() for k in retain_kw}, **bound
    )
