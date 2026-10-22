"""Review KVLEITER-1022 SWEEP (60e1181e27): a node the sweep RE-QUEUES on purpose after its visit.

``publish_unbacked_sweep`` splits a node longer than the publish window (#1407; on a TP group the window is the
chunk, 4096 tokens: ``FLLIPER_PDFLIP_ENABLE_TP_UNIFORM_PUBLISH_WINDOW`` default on). The ORIGINAL node object becomes
the tail (it keeps the Mamba anchor and the children), the head pieces are inserted above it, the visit backs the
last head piece and ``queue.append(_tail)`` re-queues the tail for its own backup. The visit-once set marks the node
at its first visit -- before the split -- so the re-queued tail is skipped: the anchored tail of the turn's chain is
not backed at this turn end (rc32 1022_030836 D 03:23:52 TP0: ``PUBLISH-SPLIT node=210 pieces=5`` then ``#924D
station=backup ... node=210`` in the same S108 turn end -- the tail's own backup, which 60e1181e27 drops).

xfail strict (Auflage S-A): green on 3ea132ca14 (the pre-fix loop visits the tail again), red on 60e1181e27.
CPU only, the author's harness (test_kvleiter_sweep_once_1022).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache import unified_radix_cache as urc  # noqa: E402

from test_kvleiter_sweep_once_1022 import _N, _tree  # noqa: E402


def _world():
    t, root = _tree()
    a = _N(root, True)                    # backed prefix
    leaf = _N(a, False)                   # the turn's un-backed leaf, longer than the window
    leaf.key = [0] * 8
    t._pdflip_publish_window = lambda: 4
    backed = []

    def write_backup(node):
        backed.append(node)
        node.write_through_pending_id = node.id
        return 1

    def split(node, window):
        head = _N(None, False)            # a new KV-only head piece between the parent and the node
        par = node.parent
        par.children = {k: v for k, v in par.children.items() if v is not node}
        head.parent = par
        par.children[head.id] = head
        node.parent = head
        head.children[node.id] = node
        node.key = [0] * window
        return node

    t.write_backup = write_backup
    t._pdflip_split_for_publish = split
    return t, root, a, leaf, backed


@pytest.mark.parametrize("chain_only", [True, False])
@pytest.mark.xfail(strict=True, reason="Auflage S-A of review KVLEITER-1022: the visit-once set drops the split "
                                       "tail's intentional re-queue")
def test_the_split_tail_is_backed_in_the_same_sweep(chain_only):
    t, root, a, leaf, backed = _world()
    first = [a, leaf] if chain_only else None
    urc.UnifiedRadixCache.publish_unbacked_sweep(t, max_issue=64, first=first, chain_only=chain_only)
    assert leaf.parent is not a, "the split happened"
    assert leaf in backed, "the anchored tail (the original node) got its own backup in this sweep"
