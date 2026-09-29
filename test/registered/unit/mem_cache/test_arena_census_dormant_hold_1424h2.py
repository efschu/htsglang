"""#1424h2 (rc12t D-TP0 18:17:50, ``ARENA-REF-HOLDERS ... tree=0 ... sum=0
own_held=2771 gap=2771``): no leak. SLEEP 18:17:41 (``HOLD-OWNED
prefetch=4``); at 18:17:44 the four reads of the #1443 dormant hold closed
with ``prefetch success`` (loaded 52480+8896+4544+111424 tokens = 2771 pages
= own_held). The resolve took the pages' reader references and no holder
class named them until the rid is admitted. The census now names them
``dormant_hold`` (in ``sum``), and counts a page only once: when a tree node
names it, it is the tree's.

Hermetic: the #1424g tree shell (real ``pdflip_arena_holder_census``, real
``pop_prefetch_loaded_tokens`` / ``release_aborted_request`` / ``_reset_full``)
on a real C arena."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import shutil
import textwrap

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t1424g", os.path.join(os.path.dirname(__file__), "test_arena_reset_orphans_1424g.py"))
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

arena = g.arena
pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")

HELD = "pdflip-0-2"
READ = ["d0", "d1", "d2"]


def _held_read_completed(arena):
    """The metal shape: a dormant hold's read of 3 pages completed -- the
    resolve holds one reference per page -- and no tree node names them."""
    slot_of = g._publish(arena, READ)
    pool = g._pool(arena)
    t, FULL = g._tree(arena, pool)
    t.root_node = type("R", (), {"children": {}})()
    t.cache_controller.pdflip_hold_rids = {HELD}
    t.prefetch_loaded_tokens_by_reqid = {HELD: 12}
    t._prefetch_completed_tokens = {}
    t._prefetch_span_pins = {}
    rows = g._resolve(arena, slot_of, READ)
    t._pdflip_note_dormant_done(HELD, rows)
    return t, pool, FULL, rows


def test_a_completed_dormant_hold_read_not_yet_in_the_tree_is_named(arena):
    """RED on 27e1d90738: gap=3 (rc12t: gap=2771), no class names the pages."""
    t, _, _, _ = _held_read_completed(arena)
    assert g._own(arena) == 3
    line = t.pdflip_arena_holder_census(arena)
    assert "tree=0 " in line and "dormant_hold=3 " in line and "sum=3 " in line, line
    assert "own_held=3 gap=0" in line, line


def test_a_page_the_tree_names_is_the_trees_never_counted_twice(arena):
    t, _, FULL, rows = _held_read_completed(arena)
    t.root_node.children = {1: g._node(FULL, 1, rows, lock=1)}   # the #1417 pin
    line = t.pdflip_arena_holder_census(arena)
    assert "tree_in_use=3 " in line and "dormant_hold=0 " in line and "gap=0" in line, line


def test_the_admission_ends_the_class(arena):
    t, _, _, _ = _held_read_completed(arena)
    t.pop_prefetch_loaded_tokens(HELD)
    assert "dormant_hold=0 " in t.pdflip_arena_holder_census(arena)


def test_the_abort_ends_the_class(arena):
    t, _, _, _ = _held_read_completed(arena)
    t.release_aborted_request(HELD)
    assert "dormant_hold=0 " in t.pdflip_arena_holder_census(arena)


def test_the_reset_ends_the_class_and_gives_the_references_back(arena):
    """The rows die with the tree they were read for: the reset drops the
    record, and the #1424g orphan pass gives the references back."""
    t, _, _, _ = _held_read_completed(arena)
    t._reset_full()
    assert not t._pdflip_dormant_done
    assert g._own(arena) == 0
    assert "own_held=0 gap=0" in t.pdflip_arena_holder_census(arena)


def test_a_read_outside_the_hold_is_not_this_class(arena):
    t, _, _, rows = _held_read_completed(arena)
    t._pdflip_dormant_done.clear()
    t._pdflip_note_dormant_done("pdflip-9-9", rows)     # not a held rid: the tree's business
    assert not t._pdflip_dormant_done


def test_the_completion_notes_exactly_the_rows_it_adopted():
    """Wiring: the one site that terminates a prefetch notes the rows its
    insert adopted (past the unclaimed head, up to the group's completion),
    after the record left ``ongoing_prefetch``."""
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = textwrap.dedent(inspect.getsource(UnifiedRadixCache.check_prefetch_progress))
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "_pdflip_note_dormant_done"]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[1]) == "host_indices[unclaimed_to:min_completed_tokens]"
    assert src.index("del self.ongoing_prefetch[req_id]") < src.index("_pdflip_note_dormant_done")
