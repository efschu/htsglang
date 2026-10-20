"""S85 (rc19 boot ..._10e9b2b492_1019_230333, pdflip-106-2655): the park-end chain's store hole.

THE METAL (D log): 05:29:26 ``ARENA-REF-CENSUS ... complete=6485 pinned=6485 refs=22046`` on every D rank --
every KV-arena slot pinned, all references D's own (TP0 tree 5555; TP1/TP2 ``ARENA-REF-HOLDERS ... queue=2829``:
references held only by the workers' host-release queue, TP0 queue=0 -- the S83 class: the release drain counts
the group MIN). 05:29:54 ``#1421 BACKUP-REFUSED why=arena_claim node=6298 tokens=640 depth=199232`` (the node
starts at 198592, the hole's first token) and node 6305; 05:29:58 ``#1427 ARENA-DROP need=8 freed=0
stages=i:0,ii:0,iii:0``; 05:29:59 ``#1470 FLUSH-PUBLISH issued=1 unbacked_left=27`` and ``#1470 FLUSH-UNBACKED
site=flush nodes=27 tokens=17408 ids=[6298, 6305, 6311, 6312, 6333, ...]`` -- 6333 is the
``PARK-END-ANCHOR-FIRST MARK ... node=6333 depth=206080`` node. The sleep reset dropped the chain; D's wake read
(05:30:18) and P's read (05:30:30) ended at 198592 (``PDFLIP-CLAIM-SHORT ... tokens=198592``), 7488 tokens were
recomputed. No line named it: S80 asks only at P's wake probe of a read that ran during the flip, S81 watches
only CLAIMED share-0 slots, and these pages were never claimed.

(1) the root, composed: an arena pinned in full by queue-only references refuses the park-end chain's claim;
    the sleep flush round (scheduler #1470: ``tree_cache._s83_release_queue_surplus_refs(site="flush")`` before
    each round, S83) gives those references back, the clock finds victims, the claim lands. RED on rc19
    (no such hook: the references wait for the MIN, the claim stays refused).
(2) the instrument: a park-end chain that still goes into the reset un-backed is named with its store end and
    the tokens the resume must recompute (``S85 PARK-END-UNBACKED``). RED on rc19 (no line).
"""
from __future__ import annotations

import logging
import os
import pathlib
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from test_s83_release_queue_local_refs_1019 import (  # noqa: E402,F401
    SLOT_BYTES, SLOTS, _pool, _publish, _refs, _resolve, _tree, arena,
)


# ---- (1) the root, composed --------------------------------------------------------------------

def _sleep_flush_round(tree):
    """What the scheduler's #1470 loop does before each flush round (S83): give back the arena references
    still queued on this rank. Absent on rc19 -- then nothing is given back."""
    hook = getattr(tree, "_s83_release_queue_surplus_refs", None)
    if callable(hook):
        hook(site="flush")


def _queue_pinned_full_arena(arena):
    """Every slot COMPLETE and referenced ONLY by this worker's host-release queue (the rc19 05:29:26 shape:
    pinned=complete=all slots, the workers' references queued, the tree holding none of these pages)."""
    stems = ["p%d" % i for i in range(SLOTS)]
    slot_of = _publish(arena, stems)
    pool = _pool(arena)
    t, _FULL = _tree(arena, pool)
    q = t.cache_controller.host_mem_release_queue
    for k in stems:
        q.put(_resolve(arena, slot_of, [k]))          # +1 per page, the reference only the queue names
    assert all(r == 1 for r in _refs(arena)), _refs(arena)
    return t, slot_of


def _claim_park_end_page(arena):
    (s, st, g), = arena.claim_slots(["park-end-206080"], [SLOT_BYTES])
    return s, st


def test_rc19_a_queue_pinned_full_arena_refuses_the_park_end_chain_until_the_sleep_flush_round(arena):
    t, slot_of = _queue_pinned_full_arena(arena)
    s, st = _claim_park_end_page(arena)
    assert st != 0, "the claim found a slot in an arena pinned in full"          # the metal: arena_claim refused
    assert arena.drop_unreferenced(list(range(SLOTS))) == 0                     # ARENA-DROP freed=0
    _sleep_flush_round(t)                                                       # the #1470 round's S83 step
    assert t.cache_controller.host_mem_release_queue.qsize() == 0, \
        "the queued references still pin the arena at the sleep (rc19: they wait for the group MIN)"
    assert arena.drop_unreferenced(list(range(SLOTS))) > 0, "the clock still finds no victim"
    s, st = _claim_park_end_page(arena)
    assert st == 0, "the park-end chain's page is still refused after the sleep flush round"


def test_the_scheduler_flush_loop_runs_the_round_before_it_claims():
    """The composition the test above relies on: the #1470 loop calls the S83 step before each sweep."""
    here = pathlib.Path(__file__).resolve()
    src = next(p for p in here.parents if (p / "python" / "flliper").is_dir())
    text = (src / "python/flliper/srt/managers/scheduler.py").read_text()
    i = text.index("for _round in (range(64) if not _memo_hit else ()):")
    window = text[i:i + 1200]
    assert "_s83_release_queue_surplus_refs" in window and 'site="flush"' in window, \
        "the #1470 flush round does not give the queued arena references back before it claims"
    assert window.index("_s83_release_queue_surplus_refs") < window.index("_sweep(max_issue=256)")


# ---- (2) the instrument --------------------------------------------------------------------------

def _chain_tree(spec):
    """A tree of one chain root -> n1 -> ... ; spec = [(tokens, backed)], the last node the park end."""
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    root = types.SimpleNamespace(id=0, parent=None, key=[], children={})
    nodes, parent = [], root
    for i, (n, backed) in enumerate(spec):
        nd = types.SimpleNamespace(id=6290 + i, parent=parent, key=list(range(n)), children={}, evicted=False,
                                   backuped=backed, l3_present=False,
                                   component_data={BASE_COMPONENT_TYPE: types.SimpleNamespace(value=torch.zeros(1))})
        parent.children[i] = nd
        nodes.append(nd)
        parent = nd
    end = nodes[-1]
    end.pdflip_park_end = True
    end.pdflip_park_rid = "pdflip-106-2655"

    def depth_of(node):
        d = 0
        while node is not None and node is not root:
            d += len(node.key)
            node = node.parent
        return d

    tree = types.SimpleNamespace(root_node=root, _pdflip_park_end_nodes=[end], pdflip_node_depth=depth_of,
                                 _collect_all_nodes=lambda: [root] + nodes)
    return tree, nodes


def test_the_metal_chain_dropped_by_the_reset_is_named_with_its_store_end(caplog):
    """198592 backed, then the refused nodes (6298: 640 tokens from 198592, 6305: 192, ..., 6333 to 206080)."""
    from flliper.srt.managers import pdflip_flush_nonblock as F

    caplog.set_level(logging.INFO)
    tree, nodes = _chain_tree([(198592, True), (640, False), (192, False), (6656, False)])
    c = F.note_unbacked_at_reset(tree, site="flush")
    assert c["nodes"] == 3
    pe = c.get("park_end")
    assert pe == [{"rid": "pdflip-106-2655", "node": nodes[-1].id, "depth": 206080, "store_end": 198592,
                   "lost_tokens": 7488, "unbacked_nodes": 3}], pe
    assert any("S85 PARK-END-UNBACKED rid=pdflip-106-2655" in m and "store_end=198592" in m
               and "lost_tokens=7488" in m for m in caplog.messages), caplog.messages


def test_a_backed_park_end_with_an_unbacked_ancestor_is_named_from_the_ancestor(caplog):
    """The park-end node itself got its write, an ancestor did not: the store still ends at the ancestor."""
    from flliper.srt.managers import pdflip_flush_nonblock as F

    caplog.set_level(logging.INFO)
    tree, nodes = _chain_tree([(1000, True), (500, False), (300, True), (200, True)])
    c = F.note_unbacked_at_reset(tree, site="flush")
    assert c["park_end"] and c["park_end"][0]["store_end"] == 1000 and c["park_end"][0]["lost_tokens"] == 1000


def test_a_backed_chain_says_nothing(caplog):
    from flliper.srt.managers import pdflip_flush_nonblock as F

    caplog.set_level(logging.INFO)
    tree, _nodes = _chain_tree([(1000, True), (500, True)])
    c = F.note_unbacked_at_reset(tree, site="flush")
    assert c["nodes"] == 0 and c.get("park_end") == []
    assert not any("S85 PARK-END-UNBACKED" in m for m in caplog.messages)
