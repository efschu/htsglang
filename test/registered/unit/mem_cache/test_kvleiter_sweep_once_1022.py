"""KVLEITER-1022 sweep cost: every node of a publish sweep is visited ONCE, in the old first-visit order.

THE METAL (NF rc32 ded00bd5a7, boot ..._1022_030836 D, 473 S108 turn ends to 04:25): ``S108 TURN-END-L2 issue_ms``
summed 71.6 s on the scheduler thread; with <= 3 un-backed nodes a chain of 80-200 nodes still cost a median 232 ms,
up to 704 ms (03:46:06 chain=137 unbacked=1 issued=1: 285.76 ms; 04:19:34 chain=171: 961.65 ms), and ``unbacked``
read 684 for a 171-node chain. THE CAUSE: the turn end sweeps ``first`` = its chain (root first) + deferred nodes with
``chain_only``; a chain node is in ``first`` AND the child of the chain node before it, and every visit of a node
queued all its children again -- the node at depth k was visited k+1 times (n^2/2 visits, each un-backed one counted
and asked again), on a list with pop(0) and list membership.

THE FIX keeps the queue (now a deque) and skips a node's later entries: the first-visit order is unchanged (a FIFO
with later duplicate entries removed keeps the order of the rest). CPU only.
"""
from __future__ import annotations

import inspect
import os
import random
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.mem_cache import unified_radix_cache as urc  # noqa: E402
from flliper.srt.pdflip import d_turn_end_l2 as te  # noqa: E402


class _Real:
    def __getattr__(self, name):
        f = getattr(urc.UnifiedRadixCache, name)
        return f.__get__(self) if callable(f) and not isinstance(f, type) else f


class _N:
    _id = 0

    def __init__(self, parent, backed):
        _N._id += 1
        self.id = _N._id
        self.parent, self.children = parent, {}
        self.evicted, self.backuped, self.l3_present = False, backed, False
        self.write_through_pending_id = None
        self.key = [0] * 4
        self.component_data = {urc.BASE_COMPONENT_TYPE: types.SimpleNamespace(value=torch.arange(4), host_value=None)}
        if parent is not None:
            parent.children[self.id] = self


def _tree():
    t = _Real()
    root = _N(None, True)
    t.__dict__.update(cache_controller=types.SimpleNamespace(), disable=False, root_node=root,
                      ongoing_write_through={}, ongoing_backup={}, component_protected_size_={},
                      _mamba_pin_budget=8, _pdflip_park_track=None, visits=[])
    t._pdflip_full_arena_skip_armed = lambda: False
    t._pdflip_publish_window = lambda: 0
    t._mamba_pins_held = lambda: 0
    t._mamba_write_through_pin_admissible = lambda node, write_back=False: True

    def anchor_only(node, kv_on_device=False):
        t.visits.append(node)                    # a backed node's visit
        return False

    def write_backup(node):
        t.visits.append(node)                    # an un-backed node's visit
        if node.id % 3:
            node.write_through_pending_id = node.id
            return 1
        return 0
    t._pdflip_anchor_only_candidate = anchor_only
    t.write_backup = write_backup
    return t, root


def _random_world(rng, L):
    t, root = _tree()
    chain, side, x = [], [], root
    for _ in range(L):
        x = _N(x, rng.random() < 0.6)
        chain.append(x)
    for c in chain:
        for _ in range(rng.randrange(0, 4)):
            s = _N(c, rng.random() < 0.5)
            side.append(s)
            for _ in range(rng.randrange(0, 3)):
                side.append(_N(s, rng.random() < 0.5))
    return t, root, chain, side


def _old_visits(first, chain_only, root):
    """The pre-fix loop's visit sequence (ded00bd5a7): list FIFO, pop(0), list membership, children queued at
    EVERY visit."""
    queue = list(first or []) + ([] if chain_only else [root])
    out = []
    while queue:
        node = queue.pop(0)
        if chain_only and node is not root and node not in first:
            continue
        for child in list(node.children.values()):
            queue.append(child)
        out.append(node)
    return out


def _first_occurrences(seq, root):
    seen, out = set(), []
    for n in seq:
        if n is not root and id(n) not in seen:
            seen.add(id(n))
            out.append(n)
    return out


def test_random_chains_old_and_new_visit_the_same_nodes_in_the_same_order():
    rng = random.Random(1022)
    dupes = 0
    for case in range(60):
        t, root, chain, side = _random_world(rng, rng.randrange(3, 50))
        mode = case % 3
        if mode == 0:       # the S108 turn end: own chain + deferred nodes, chain only
            first, chain_only = chain + rng.sample(side, min(len(side), rng.randrange(0, 8))), True
        elif mode == 1:     # a retain publish with the whole tree behind its chain
            first, chain_only = list(chain), False
        else:               # the flush sweep: the whole tree from the root
            first, chain_only = None, False
        old = _old_visits(first, chain_only, root)
        dupes += len(old) - len({id(n) for n in old})
        stats = urc.UnifiedRadixCache.publish_unbacked_sweep(t, max_issue=1 << 20, first=first,
                                                            chain_only=chain_only)
        want = _first_occurrences(old, root)
        assert [n.id for n in t.visits] == [n.id for n in want], case
        unb = [n for n in want if not n.backuped]
        assert stats["issued"] == sum(1 for n in unb if n.id % 3) and stats["unbacked"] == len(unb)
    assert dupes > 1000, "the old loop must have revisited nodes (the defect this fixes)"


def test_a_long_chain_is_visited_linearly():
    """The quadratic form: a 400-node chain visited 400 times (the old loop: 400*401/2 = 80200), and the wall
    time grows like the chain, not like its square."""
    def run(L):
        t, root = _tree()
        chain, x = [], root
        for _ in range(L):
            x = _N(x, True)
            chain.append(x)
        t0 = time.perf_counter()
        urc.UnifiedRadixCache.publish_unbacked_sweep(t, max_issue=64, first=list(chain), chain_only=True)
        return len(t.visits), time.perf_counter() - t0

    n400, t400 = run(400)
    assert n400 == 400, n400
    _n100, t100 = min((run(100) for _ in range(3)), key=lambda r: r[1])
    t400 = min([t400] + [run(400)[1] for _ in range(2)])
    assert t400 < 10 * t100 + 0.005, (t100, t400)


# SWEEPCENSUS-1023 port to the 27B line: test_in_tree_memo_gives_the_old_answer (_s108_in_tree memo) and
# test_the_turn_line_carries_the_split_after_n (turn line match/sweep/claim split) stay on the NF line -- those
# KVLEITER-1022 parts are not ported here.
