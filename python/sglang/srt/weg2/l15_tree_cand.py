"""L15-TREE-CAND: finished requests stay L1.5 hold candidates.

27B boot ...e8dd3fd36a_1002_195124: the 15-s DECODE-COLLECT window lets the
decodes FINISH before the sleep (FLIP begin outstanding=0), so the hold
candidate list (running_batch.reqs + weg2_d_parked) is empty at every sleep:
held=0, SHADOW n=0. The finished requests are not gone, though -- their
spans were inserted into the radix tree (release_kv_cache(is_insert=True)) and
sit on the device with their END anchor (the mamba checkpoint of the tip node).
They are exactly what the next turn of the same session prefix-hits, so they
are what the hold must keep.

This module adds them as candidates, in two steps:

1. ``local_candidates``: this rank walks its own tree and lists the TIPS --
   device-resident nodes that carry a device mamba value and have no such
   node below them -- most recently used first, each with the digest of its
   full token chain (token identity, not a per-process counter).
2. ``agree``: ONE host gather over D's group. Every rank sees the same
   gathered lists and derives the same answer: the digests present on EVERY
   rank, ordered by the summed recency position (ties by digest). Nothing
   rank-local (``last_access_time`` is a per-process clock) decides before the
   collective.

The agreed tips become Req-shaped pseudo requests (``pseudo_req``) that
``l15_bind.build_retain_kwargs`` resolves through the same tree match as a
PARKED req (``match_parked``): kind "served", rid ``tree:<digest>``.

Pure Python + duck typing (no torch.cuda, no scheduler import) so the unit
tests run with SimpleNamespace trees.
"""

from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

TREE_CAND_ENV = "SGLANG_WEG2_L15_TREE_CAND"
TREE_CAND_N_ENV = "SGLANG_WEG2_L15_TREE_CAND_N"
DEFAULT_MAX_N = 8
RID_PREFIX = "tree:"


def env_on(env: Optional[Mapping[str, str]] = None) -> bool:
    """Default ON (L15 ships switched on); ``=0`` restores the old list."""
    import os

    env = os.environ if env is None else env
    return str(env.get(TREE_CAND_ENV, "1")).strip().lower() not in (
        "0", "off", "false", "no")


def max_n(env: Optional[Mapping[str, str]] = None) -> int:
    import os

    env = os.environ if env is None else env
    try:
        return max(0, int(str(env.get(TREE_CAND_N_ENV, DEFAULT_MAX_N)).strip()))
    except ValueError:
        return DEFAULT_MAX_N


@dataclass(frozen=True)
class TreeCand:
    digest: str
    n_tokens: int
    last_access: float
    tokens: Tuple[int, ...] = field(default=(), compare=False, repr=False)
    extra_key: Optional[str] = field(default=None, compare=False, repr=False)


def _mamba_type():
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        ComponentType,
    )

    return ComponentType.FULL, ComponentType.MAMBA


def _has_value(node, ctype) -> bool:
    try:
        v = node.component_data[ctype].value
    except (AttributeError, KeyError, IndexError, TypeError):
        return False
    if v is None:
        return False
    try:
        return len(v) > 0
    except TypeError:
        return True


def _key_tokens(node) -> Tuple[List[int], bool]:
    """(raw token ids of the node's key, is_bigram). A bigram key holds
    ``units + 1`` raw tokens (RadixKey: "slices share one boundary token");
    ``limit`` is honoured through ``raw_token_ids`` where the key has it."""
    key = getattr(node, "key", None)
    raw = getattr(key, "raw_token_ids", None)
    ids = raw() if callable(raw) else getattr(key, "token_ids", None)
    if ids is None:
        return [], False
    toks = [int(x) for x in (ids.tolist() if hasattr(ids, "tolist") else ids)]
    return toks, bool(getattr(key, "is_bigram", False))


class ChainError(ValueError):
    """The node keys do not compose to one token chain (bigram boundary
    token of a child differs from its parent's last token)."""


def chain_tokens(node, root) -> List[int]:
    """Token ids of the chain root -> node (the root excluded), exactly the
    raw ids the tree was inserted with.

    L15-UNHOLDABLE (27B y8j 06:31:38, BIGRAM_KEYS=1 / DFLASH): under bigram
    keys every node's key carries ``units + 1`` raw tokens and a child's FIRST
    raw token IS its parent's LAST one (RadixKey.__getitem__ slices bigrams
    ``[start, stop)`` as raw ``[start, stop + 1)``). Concatenating the keys
    as-is doubled that boundary token at every node edge: the match walked
    the first node, asked for the child keyed ``(t, t)``, found none and
    stopped on a node without a mamba value -- "match-census ...
    MambaComponent:absent", all 4 agreed tips "unholdable" at the bind. The
    boundary token is kept once."""
    parts: List[Tuple[List[int], bool]] = []
    cur = node
    while cur is not None and cur is not root:
        parts.append(_key_tokens(cur))
        cur = getattr(cur, "parent", None)
    out: List[int] = []
    for toks, bigram in reversed(parts):
        if bigram and out and toks:
            if toks[0] != out[-1]:
                raise ChainError(
                    "bigram boundary token %d != parent's last token %d"
                    % (toks[0], out[-1]))
            out.extend(toks[1:])
        else:
            out.extend(toks)
    return out


def digest_of(tokens: Sequence[int], extra_key=None) -> str:
    h = hashlib.blake2b(array("q", tokens).tobytes(), digest_size=8)
    if extra_key is not None:
        h.update(b"\x00" + str(extra_key).encode("utf-8", "replace"))
    return h.hexdigest()


def tips_of(tree_cache) -> List[object]:
    """Device-resident nodes with a device mamba value (an END anchor) and no
    such node below them. Iterative walk -- one pass, no recursion."""
    full_t, mamba_t = _mamba_type()
    root = getattr(tree_cache, "root_node", None)
    if root is None:
        return []
    order: List[object] = []
    stack = [root]
    while stack:
        n = stack.pop()
        order.append(n)
        ch = getattr(n, "children", None) or {}
        stack.extend(ch.values())
    qual = {}
    below = set()  # id(node) having a qualifying node strictly below it
    tips: List[object] = []
    for n in reversed(order):  # children before parents
        if n is root:
            continue
        q = _has_value(n, full_t) and _has_value(n, mamba_t)
        qual[id(n)] = q
        p = getattr(n, "parent", None)
        if (q or id(n) in below) and p is not None:
            below.add(id(p))
        if q and id(n) not in below:
            tips.append(n)
    return tips


def l2_backed(node, root) -> bool:
    """Every chain node has its KV host rows (or the recorded L2 shadow) and
    the tip has its mamba host row: the cap-0 rank refills the hold from L2,
    and one unbacked token makes its POST vote refuse the WHOLE round
    (l15_sleep_agree.post_vote) -- a tip it cannot refill must not be offered."""
    full_t, mamba_t = _mamba_type()
    try:
        mh = node.component_data[mamba_t].host_value
    except (AttributeError, KeyError, IndexError, TypeError):
        return False
    if mh is None or len(mh) == 0:
        return False
    cur = node
    while cur is not None and cur is not root:
        try:
            hv = cur.component_data[full_t].host_value
        except (AttributeError, KeyError, IndexError, TypeError):
            return False
        if hv is None or len(hv) == 0:
            sh = getattr(cur, "_weg2_l2_shadow", None)
            if sh is None or len(sh) != 2 or not len(sh[0]):
                return False
        cur = getattr(cur, "parent", None)
    return True


def local_candidates(tree_cache, limit: int,
                     require_l2: bool = False) -> List[TreeCand]:
    """This rank's tips, most recently used first, at most ``limit``.
    ``require_l2`` (the cap-0 rank): only tips whose chain is L2-backed."""
    if limit <= 0:
        return []
    root = getattr(tree_cache, "root_node", None)
    tips = tips_of(tree_cache)
    tips.sort(key=lambda n: -float(getattr(n, "last_access_time", 0) or 0))
    if require_l2:
        tips = [n for n in tips if l2_backed(n, root)]
    out: List[TreeCand] = []
    for n in tips:
        if len(out) >= limit:
            break
        try:
            toks = chain_tokens(n, root)
        except ChainError:
            continue  # this tip cannot be matched; the next one may
        if not toks:
            continue
        ek = getattr(getattr(n, "key", None), "extra_key", None)
        out.append(TreeCand(
            digest=digest_of(toks, ek),
            n_tokens=len(toks),
            last_access=float(getattr(n, "last_access_time", 0) or 0),
            tokens=tuple(toks),
            extra_key=ek,
        ))
    return out


def agree(local: Sequence[TreeCand], gather: Callable[[object], List[object]],
          limit: int) -> List[TreeCand]:
    """The rank-agreed candidates, most recent first.

    ``gather`` is ONE all_gather_object over D's group: every rank sends
    ``[(digest, n_tokens), ...]`` in its own recency order (an empty list when
    its walk failed or the switch is off -- the rank still takes part, so the
    collective is entered by all). The answer is a pure function of the
    gathered vectors, so every rank derives the same list: digests present on
    EVERY rank, ordered by the summed recency position, ties by digest.
    Returns this rank's own TreeCand objects (tokens attached).
    """
    votes = gather([(c.digest, int(c.n_tokens)) for c in local])
    if not votes:
        return []
    pos: Dict[str, List[int]] = {}
    sizes: Dict[str, set] = {}
    for vec in votes:
        for i, (d, n) in enumerate(vec or ()):
            pos.setdefault(d, []).append(i)
            sizes.setdefault(d, set()).add(int(n))
    everywhere = [
        d for d, ps in pos.items()
        if len(ps) == len(votes) and len(sizes[d]) == 1
    ]
    everywhere.sort(key=lambda d: (sum(pos[d]), d))
    mine = {c.digest: c for c in local}
    out = []
    for d in everywhere[: max(0, int(limit))]:
        if d in mine:
            out.append(mine[d])
    return out


def pseudo_req(c: TreeCand, rank_in_order: int, total: int):
    """A Req-shaped stand-in l15_bind resolves through the tree match.

    ``req_pool_idx`` None sends it down the PARKED path; ``l15_tree_tokens``
    makes ``match_parked`` match the WHOLE chain (a real req's last token has
    no KV yet, a tree tip's does). ``l15_last_active`` is rank-uniform (the
    agreed position), larger = younger.
    """
    return SimpleNamespace(
        rid=RID_PREFIX + c.digest,
        req_pool_idx=None,
        origin_input_ids=c.tokens,
        output_ids=[],
        l15_tree_tokens=tuple(c.tokens),
        extra_key=c.extra_key,
        mamba_pool_idx=None,
        last_node=None,
        l15_kind="served",
        l15_last_active=float(total - rank_in_order),
    )


def tip_spans(spans: Sequence[Mapping]) -> List[Tuple[str, int]]:
    """``(rid, depth)`` of the published hold spans that are tree tips."""
    out: List[Tuple[str, int]] = []
    for s in spans or ():
        rid = str(s.get("rid", ""))
        depth = int(s.get("depth", 0) or 0)
        if rid.startswith(RID_PREFIX) and depth > 0:
            out.append((rid, depth))
    return out


def match_tip(spans: Sequence[Mapping], token_ids: Sequence[int],
              extra_key=None
              ) -> Tuple[Optional[Tuple[str, int, int]], List[Tuple[str, int]]]:
    """L15-TREE-FRONT: the held tree tip a prompt extends.

    ``spans``: the published hold descriptor's spans (every rank publishes the
    same agreed list). A tip is hot for ``token_ids`` when the prompt's first
    ``raw`` tokens hash to the tip's digest (the rid is ``tree:<digest>``,
    digest = :func:`digest_of` of the tip's whole RAW token chain, which is the
    request's own token sequence -- :func:`chain_tokens` keeps a bigram
    boundary token once). The span's ``depth`` counts KV slots: ``raw`` ==
    ``depth`` on a plain tree, ``depth + 1`` on a bigram tree (units + 1 raw
    tokens); both are tried. Pure function of data every rank sees -- the
    answer is the same on every P stage. The LONGEST matching tip wins.

    Returns ``((rid, depth, raw) or None, tips)``; ``tips`` is every
    ``(rid, depth)`` for the miss marker."""
    tips = tip_spans(spans)
    if not tips:
        return None, tips
    arr = array("q", [int(t) for t in token_ids])
    for rid, depth in sorted(tips, key=lambda t: (-t[1], t[0])):
        for raw in (depth, depth + 1):
            if raw > len(arr):
                continue
            if digest_of(arr[:raw], extra_key) == rid[len(RID_PREFIX):]:
                return (rid, depth, raw), tips
    return None, tips


def is_tree_req(req) -> bool:
    return getattr(req, "l15_tree_tokens", None) is not None


def unholdable_line(at: str, rids: Sequence[str], why: Mapping[str, str]) -> str:
    """L15-UNHOLDABLE counter line: a count per reason and the reason per
    candidate (first 8), e.g. ``L15-TREE-CAND unholdable at=bind n=4
    why=partial_match:4 rids=tree:ab..(partial_match),...``."""
    counts: Dict[str, int] = {}
    for r in rids:
        k = why.get(r, "?")
        counts[k] = counts.get(k, 0) + 1
    return ("L15-TREE-CAND unholdable at=%s n=%d why=%s rids=%s"
            % (at, len(rids),
               ",".join("%s:%d" % kv for kv in sorted(counts.items())),
               ",".join("%s(%s)" % (r, why.get(r, "?")) for r in list(rids)[:8])))


def agree_holdable(reqs: Sequence[object], probe: Callable[[object], Optional[str]],
                   gather: Callable[[object], List[object]]
                   ) -> Tuple[List[object], Dict[str, str]]:
    """L15-UNHOLDABLE: ONE more gather over D's group -- each rank's hold
    test of every agreed candidate. Kept: the candidates EVERY rank can hold
    (a rank-uniform answer: the agreed list is identical everywhere, the
    votes are the same gathered vectors). Returns (kept reqs, rid -> reason
    of each dropped one, ``<code>@r<rank>`` of the first refusing rank)."""
    mine = []
    for r in reqs:
        try:
            why = probe(r)
        except Exception as exc:  # noqa: BLE001 -- a refusal, not an escape
            why = "probe_error:%s" % type(exc).__name__
        mine.append((str(r.rid), why))
    votes = gather(mine) or []
    maps = [dict(v or ()) for v in votes]
    kept, dropped = [], {}
    for r in reqs:
        rid = str(r.rid)
        reason = None
        for rk, m in enumerate(maps):
            if rid not in m:
                reason = "peer_missing@r%d" % rk
                break
            if m[rid] is not None:
                reason = "%s@r%d" % (m[rid], rk)
                break
        if reason is None and maps:
            kept.append(r)
        else:
            dropped[rid] = reason or "no_votes"
    return kept, dropped


def build(tree_cache, gather: Callable[[object], List[object]], n_live: int,
          env: Optional[Mapping[str, str]] = None, log=None,
          require_l2: bool = False,
          probe: Optional[Callable[[object], Optional[str]]] = None) -> List[object]:
    """Local walk + the single collective + pseudo reqs; never raises.

    The gather is entered by every rank no matter what its local walk did
    (a failed walk votes the empty list), so one rank's exception cannot
    leave the others blocked in the collective.

    ``probe`` (L15-UNHOLDABLE): when given, every rank tests every agreed
    candidate (``l15_bind.tree_probe``) and a SECOND gather keeps only the
    ones every rank can hold; the dropped ones are named per candidate in an
    ``L15-TREE-CAND unholdable at=probe`` line. The second gather is entered
    by every rank too (an empty vote when the first agreement failed).
    """
    lim = max_n(env)
    local: List[TreeCand] = []
    err = None
    try:
        # live reqs may cover some tips (dropped at the bind), so look
        # n_live deeper than the cap
        local = local_candidates(tree_cache, lim + max(0, int(n_live)),
                                 require_l2=require_l2)
    except Exception as exc:  # noqa: BLE001 -- vote empty, stay in the collective
        err = exc
        local = []
    try:
        agreed = agree(local, gather, lim + max(0, int(n_live)))
    except Exception as exc:  # noqa: BLE001 -- no agreed list, no tree candidates
        agreed = []
        err = err or exc
    if log is not None:
        log("L15-TREE-CAND local=%d agreed=%d%s rids=%s"
            % (len(local), len(agreed),
               "" if err is None else " walk_failed=%s: %s" % (type(err).__name__, err),
               ",".join("%s(%d)" % (RID_PREFIX + c.digest, c.n_tokens)
                        for c in agreed[:6])))
    reqs = [pseudo_req(c, i, len(agreed)) for i, c in enumerate(agreed)]
    if probe is None:
        return reqs
    try:
        kept, dropped = agree_holdable(reqs, probe, gather)
    except Exception as exc:  # noqa: BLE001 -- no candidates, never an escape
        if log is not None:
            log("L15-TREE-CAND probe failed: %s: %s" % (type(exc).__name__, exc))
        return []
    if log is not None:
        log("L15-TREE-CAND probe agreed=%d holdable=%d unholdable=%d"
            % (len(reqs), len(kept), len(dropped)))
        if dropped:
            log(unholdable_line("probe", list(dropped), dropped))
    return kept
