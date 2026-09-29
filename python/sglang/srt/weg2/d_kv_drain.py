"""D-KV-DRAIN (29.09., NF1d/NF1e): a pending KV shrink that only the TREE holds.

The metal (NF1d z30y3f 09291811, carried over into NF1e): after the 245k cell
the end event left ``pending S0`` (cap 32768) with the floor at 248832 -- no
request held those pages any more, the RETAINED radix tree did. The
D-MEM-SCHED machine waits for the floor to drain, but nothing drains a tree
that nobody evicts: the peel runs only under allocation pressure, and under
the pending cap the pressure lands on the ids BELOW it (c158a43406 lifts the
cap for that). So the stage stayed up until the next sleep and the expert rows
of the stage's cell stayed OFF -- against the Grundgesetz "free VRAM is
experts, always".

This module closes that gap. While a shrink is pending and its floor is
blocked, the tick demotes the tree nodes that hold pages ABOVE the pending cap
to the host: ``_evict_to_host`` on nodes whose KV is already backed (the L2
copy is acked -- nothing is dropped, a later match loads it back); a hot node
without its L2 copy yet gets its write-through issued and waits for the ack.
Only unlocked nodes move -- a page a running request (decode) reads is never
touched, so nothing is taken from the decode path.

GROUP UNIFORMITY (the whole risk of a tree edit on D): the page lists differ
per rank (uneven DCP, #603), the tree does not. Every rank lists its device
nodes in ONE canonical order (children by key), the group first agrees on the
shape (count + crc of the keys, MIN/MAX), then on per-node flags in one MIN
collective: ``hot`` (any rank holds a page above the cap there, MAX), ``dev``
(device-on on any rank, MAX), ``host`` (unlocked + host copy on EVERY rank,
MIN) and ``l3`` (unlocked, no host copy but store-acked ``l3_present`` on
EVERY rank, MIN; write-through only). A hot node is demoted with its whole
device subtree only if every device-on node in it is in ONE of the two classes
on every rank -- ``host`` nodes stay in the tree host-only (``_evict_to_host``),
``l3`` nodes leave the device the way the write-through peel lets them
(``_evict_device_leaf``; the store holds the bytes). So every rank evicts the
same nodes the same way in the same post-order. A mixed class waits, a shape
mismatch abstains (counted); nothing guesses.

Cost: entered only under a replicated condition (pending shrink, floor above
the pending cap, and either no running request or every ``DRAIN_EVERY``-th
round), two small CPU collectives and a host-side tree walk; no CUDA sync, no
copy (the L2 copy already exists). The counters live on the MemSched and go to
the rank's RankState stats file (``rankstats`` block ``mem_sched``); the log
line ``WEG2 D-MEM-SCHED DRAIN`` is for humans only.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

MARKER = "WEG2 D-MEM-SCHED DRAIN"
#: replicated cadence while requests run (every round when the group is idle)
DRAIN_EVERY = 16
_FULL = None


def enabled() -> bool:
    """Grundgesetz VRAM = experts: default ON; the switch is a diagnostic stop."""
    from sglang.srt.environ import envs

    return not bool(envs.SGLANG_WEG2_DISABLE_D_KV_DRAIN.get())


def _full_type():
    global _FULL
    if _FULL is None:
        from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        _FULL = ComponentType.FULL
    return _FULL


@dataclass
class DrainResult:
    ran: bool = False
    nodes: int = 0
    tokens: int = 0
    hot: int = 0
    waiting_backup: int = 0
    backup_issued: int = 0
    locked: int = 0
    l3_dropped: int = 0
    mismatch: bool = False
    reason: str = ""
    counters: dict = field(default_factory=dict)


def _key_repr(k) -> bytes:
    return repr(k).encode("utf-8", "backslashreplace")


def canonical_nodes(tree) -> List[tuple]:
    """Every non-root node in ONE order all ranks share (children by key):
    ``(node, parent_index, key_bytes)``. The index of a node's parent comes
    before the node (pre-order), so a reversed walk is a valid post-order."""
    out: List[tuple] = []
    stack = [(tree.root_node, -1, b"")]
    while stack:
        node, pidx, kb = stack.pop()
        idx = -1
        if node is not tree.root_node:
            idx = len(out)
            out.append((node, pidx, kb))
        kids = sorted(((_key_repr(k), c) for k, c in node.children.items()),
                      key=lambda kv: kv[0], reverse=True)
        for ckb, child in kids:
            stack.append((child, idx, ckb))
    return out


def _shape(nodes: Sequence[tuple]) -> int:
    crc = 0
    for _node, pidx, kb in nodes:
        crc = zlib.crc32(kb, crc)
        crc = zlib.crc32(str(pidx).encode(), crc)
    return crc & 0x7FFFFFFF


def _device_value(node):
    cd = node.component_data[_full_type()]
    return getattr(cd, "value", None)


def _locked(node) -> bool:
    return any(int(getattr(cd, "lock_ref", 0) or 0) > 0 for cd in node.component_data)


def _write_through(tree) -> bool:
    ctl = getattr(tree, "cache_controller", None)
    return ctl is not None and getattr(ctl, "write_policy", None) != "write_back"


def _in_flight(tree, node) -> bool:
    return getattr(node, "id", None) in (getattr(tree, "ongoing_write_through", None) or {})


def _max_page(value, page: int) -> int:
    if value is None or not hasattr(value, "numel") or not value.numel():
        return -1
    return int(value.max()) // max(1, int(page))


def drain(tree, cap_tokens: int, page_size: int,
          gmin: Optional[Callable[[list], list]]) -> DrainResult:
    """Demote the unlocked, backed tree nodes above ``cap_tokens`` to the host,
    group-uniformly. Returns what moved and why the rest stayed."""
    res = DrainResult(ran=True)
    if tree is None or not hasattr(tree, "_evict_to_host") or getattr(tree, "disable", False):
        res.ran, res.reason = False, "no hierarchical tree"
        return res
    page = max(1, int(page_size))
    lim = int(cap_tokens) // page
    nodes = canonical_nodes(tree)
    n, fp = len(nodes), _shape(nodes)
    agree = gmin if gmin is not None else (lambda v: v)
    shape = agree([n, -n, fp, -fp])
    if not (int(shape[0]) == -int(shape[1]) == n and int(shape[2]) == -int(shape[3]) == fp):
        res.mismatch, res.reason = True, "tree shape differs across ranks (n=%d fp=%d)" % (n, fp)
        return res
    if n == 0:
        res.reason = "empty tree"
        return res
    wt = _write_through(tree)
    hot, dev, on_host, on_l3 = [0] * n, [0] * n, [0] * n, [0] * n
    for i, (node, _p, _k) in enumerate(nodes):
        val = _device_value(node)
        if val is None:
            continue
        dev[i] = 1
        if _max_page(val, page) > lim:
            hot[i] = 1
        if _locked(node):
            continue
        if bool(getattr(node, "backuped", False)):
            on_host[i] = 1
        elif wt and bool(getattr(node, "l3_present", False)):
            on_l3[i] = 1
    votes = agree([-h for h in hot] + [-d for d in dev] + on_host + on_l3)
    hot_g = [-int(v) for v in votes[:n]]
    dev_g = [-int(v) for v in votes[n:2 * n]]
    host_g = [int(v) for v in votes[2 * n:3 * n]]
    l3_g = [int(v) for v in votes[3 * n:]]
    ok_g = [1 if (host_g[i] or l3_g[i]) else 0 for i in range(n)]
    children: List[List[int]] = [[] for _ in range(n)]
    for i, (_node, pidx, _k) in enumerate(nodes):
        if pidx >= 0:
            children[pidx].append(i)

    def subtree(i: int) -> List[int]:
        out, st = [], [i]
        while st:
            j = st.pop()
            out.append(j)
            st.extend(children[j])
        return out

    chosen = set()
    for i in range(n):
        if not hot_g[i] or i in chosen:
            continue
        res.hot += 1
        sub = [j for j in subtree(i) if dev_g[j]]
        bad = [j for j in sub if not ok_g[j]]
        if bad:
            for j in bad:
                node = nodes[j][0]
                if _locked(node):
                    res.locked += 1
                elif (_device_value(node) is not None and not getattr(node, "backuped", False)
                      and not getattr(node, "l3_present", False)):
                    res.waiting_backup += 1
                    if _in_flight(tree, node):
                        continue  # its write-through is out: the ack decides
                    try:
                        if tree.write_backup(node) > 0:
                            res.backup_issued += 1
                    except Exception as exc:  # noqa: BLE001 -- the ack path owns failures
                        logger.info("%s write_backup refused node=%s (%s: %s)", MARKER,
                                    getattr(node, "id", "?"), type(exc).__name__, exc)
            continue
        chosen.update(sub)
    if not chosen:
        res.reason = "nothing movable (locked=%d waiting_backup=%d)" % (res.locked, res.waiting_backup)
        return res
    tracker = {ct: 0 for ct in tree.tree_components}
    for j in sorted(chosen, reverse=True):  # pre-order reversed = children first
        node = nodes[j][0]
        if _device_value(node) is None:
            continue
        if not tree._is_device_leaf(node):
            res.reason = "node %s no device leaf at its turn -- stopped" % getattr(node, "id", "?")
            break
        if host_g[j] and node.backuped:
            tree._evict_to_host(node, tracker)
        elif l3_g[j] and getattr(node, "l3_present", False):
            tree._evict_device_leaf(node, tracker)
            res.l3_dropped += 1
        else:
            res.reason = "node %s changed class at its turn -- stopped" % getattr(node, "id", "?")
            break
        res.nodes += 1
    res.tokens = int(tracker.get(_full_type(), 0))
    res.reason = res.reason or "demoted to host"
    return res
