"""#59 (D side): the depth a finished leg 2 can be RESUMED from.

The front credits a D-served leg by its response's ``cached_tokens`` -- the KV
D found on ARRIVAL. What D keeps for the NEXT turn is a different number. The
finish inserts the request's whole key, but a hybrid GDN/Mamba model resumes
only from a recurrent anchor (#928), and a finish anchors on the track grid
(``mamba_last_track_seqlen``, #747): an off-grid finish keeps KV above its last
anchor that no admission can resume from (RETAIN off-grid). The front then
prices the next turn's D prefill below what D computes.

``pdflip_resumable_depth`` is the number the next admission would take. The
finish's insert runs before the output is streamed (``release_kv_cache`` in
the result processor), so the streamer probes the tree as the next turn will
find it: the side-effect-free admission probe the RU/H98 votes already use
(:func:`tp_match_floor.admission_probe` -- host admission length with the
#1040 state-aligned extent, the #928 anchor test, the #1424d proof cut) on the
request's whole token sequence, without this turn's ``len - 1`` limit (the
next turn is longer).

GROUP-UNIFORM, never one rank's guess:

* Form A (H98): the attention host decides the depth and the workers follow,
  so the host's probe IS the group's; a worker computes nothing;
* Form A x the token cut (#239 S3d, ``kv=qsa_forma_dcp``): a worker owns real
  full-attention rows, so losing one of its pages lowers the depth -- MIN over
  the TP cpu group (the classic TP collective), the host voting its admission
  probe, each worker its KV reach (the H98 follow walk: it holds no anchor);
* one rank (tp 1, no DP attention, no PP): trivially;
* a classic TP group: MIN over the TP cpu group -- a rank that cannot realize
  a depth takes the group there, as in the RU usable vote;
* anything else (DP attention without Form A, PP): no field, and the front
  keeps its old price rather than a number one rank could not realize.

Stamped on the finishing output only (``time_stats.pdflip_resumable_depth``,
``UNSET`` = absent; 0 is a MEASURED "nothing resumable" and is sent). An
aborted request finishes through the same path and carries it too. No switch:
an absent field is the front's old behaviour.

#59c PARK-SECURED (rc27 boot ..._893d09fa3f_1021_023225, D 02:44:15): the park
answer is what the front prices a parked request's presence with, and the flip's
sleep follows the park at once -- its reset drops every node whose KV lives on
the device only. The admission probe counts those nodes: ``#59b PARK-RESUMABLE
... pdflip-2-30=139008`` was answered one line after ``PARK-BACKUP REFUSED ...
verdict=stands`` (node 155, [133632, 139008), had no host copy and no L3 copy);
the reset lost it (``S85 PARK-END-UNBACKED ... store_end=133632
lost_tokens=5376``), P resumed at 133632. The answer named 5376 tokens no tier
held. Now each rank caps its park depth at its SECURED end -- the deepest point
up to which every node of the request's path holds its KV off the device (a host
copy / arena claim, written or in flight -- the #1470 loop joins those before
the reset -- or ``l3_present``) -- and re-probes there (the anchor rule picks the
deepest resumable point at or below the cap). The cap rides the SAME single MIN
reduce as the depth (one collective, twice the slots), so the group answers one
number. The read clamp of the parked requests keeps the uncapped depth: a tail
the sleep's flush still secures (S86e) is read, not recomputed; a lower answer
only makes the front price conservatively. A probe that cannot walk the path
keeps the uncapped depth (the old answer), counted in ``SECURED_STATS``.
"""

from __future__ import annotations

import logging
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: ``time_stats.pdflip_resumable_depth`` before (or without) a stamp.
UNSET = -1
#: meta_info / usage key the front reads.
FIELD = "pdflip_resumable_depth"

MODE_HOST = "form-a-host"
MODE_FOLLOW = "form-a-worker"
MODE_DCP_MIN = "form-a-dcp-min"
MODE_SOLO = "solo"
MODE_MIN = "tp-min"
MODE_NONE = "none"


def group_mode(ps: Any) -> str:
    """How this rank's group makes the depth uniform (see the module doc).
    Group-uniform by construction: every input is the group's (the Form A
    role plan, the parallel sizes)."""
    from flliper.srt.managers import tp_match_floor

    if int(getattr(ps, "pp_size", 1) or 1) > 1:
        return MODE_NONE
    if tp_match_floor.form_a_follow_active():
        from flliper.srt.rank_role import form_a_token_cut_active

        if form_a_token_cut_active():
            return MODE_DCP_MIN
        return MODE_FOLLOW if tp_match_floor.this_rank_follows() else MODE_HOST
    dp = int(getattr(ps, "attn_dp_size", 1) or 1)
    if dp > 1:
        return MODE_NONE
    if int(getattr(ps, "tp_size", 1) or 1) <= 1:
        return MODE_SOLO
    return MODE_MIN


def _next_turn_view(req: Any) -> Any:
    """``req`` as the next turn's admission sees it: the whole sequence, the
    limit at its full length, none of THIS turn's read caps."""
    output = getattr(req, "output_ids", None)
    return types.SimpleNamespace(
        rid=getattr(req, "rid", None),
        origin_input_ids=req.origin_input_ids,
        output_ids=[] if output is None else output,
        extra_key=getattr(req, "extra_key", None),
        positional_embed_overrides=getattr(req, "positional_embed_overrides", None),
        _compute_max_prefix_len=lambda n: n,
    )


def local_depth(tree_cache: Any, req: Any, *, follow: bool = False) -> int:
    """This rank's realizable resume depth for ``req``'s sequence (0 when the
    probe cannot price it -- the safe direction, as in every vote). ``follow``
    (a Form A worker under the token cut): its KV reach, walked with the
    mamba rule suspended -- it holds no anchor to test."""
    from flliper.srt.managers import tp_match_floor

    if tree_cache is None:
        return 0
    return max(
        0,
        int(tp_match_floor.admission_probe(tree_cache, _next_turn_view(req), follow=follow)),
    )


def _track_field(req: Any, name: str) -> str:
    """Observation only: a request's mamba track position (``-`` = None / absent),
    read where the stamp runs. ``mamba_branching_seqlen`` is the match's branching
    point, ``mamba_last_track_seqlen`` the extend's tracked position the finish cuts
    the key at (nf-next-1006-29: a branching track displaces the end track). It is
    None after a finish whose insert already consumed it; then the field reads ``-``."""
    v = getattr(req, name, None)
    if v is None:
        return "-"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "?"


#: #59c instrument: parks answered, rids capped, tokens taken off, walks that could not tell (uncapped)
SECURED_STATS = {"rids": 0, "capped": 0, "tokens": 0, "unknown": 0}


def _node_secured(node: Any) -> bool:
    """#59c: this node's KV survives the sleep's reset -- a host copy (an arena claim, written or in flight) or its
    pages in L3 (``l3_present``). A device-only node does not."""
    if bool(getattr(node, "l3_present", False)):
        return True
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    cd = node.component_data[BASE_COMPONENT_TYPE]
    hv = getattr(cd, "host_value", None)
    return hv is not None and int(hv.numel()) > 0


def secured_end(tree_cache: Any, req: Any, depth: int) -> Optional[int]:
    """#59c: the depth (<= ``depth``) up to which every node of ``req``'s path holds its KV off the device; None
    when the path cannot be walked (no cap -- the old answer). Side-effect free on the request (the probe's match)."""
    depth = int(depth)
    if depth <= 0 or tree_cache is None:
        return depth
    from array import array

    from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from flliper.srt.mem_cache.radix_cache import RadixKey

    ids = array("q", req.origin_input_ids)
    ids.extend(getattr(req, "output_ids", None) or ())
    res = tree_cache.match_prefix(MatchPrefixParams(
        key=RadixKey(token_ids=ids, extra_key=getattr(req, "extra_key", None), limit=depth),
        cow_mamba=False, req=None))
    node = getattr(res, "last_host_node", None) or getattr(res, "best_match_node", None)
    root = getattr(tree_cache, "root_node", None)
    if node is None or root is None:
        return None
    chain, cur, hops = [], node, 0
    while cur is not None and cur is not root and hops < 1_000_000:
        chain.append(cur)
        cur, hops = getattr(cur, "parent", None), hops + 1
    if cur is not root:
        return None                       # a hand-built / detached node: cannot tell
    chain.reverse()
    start = 0
    for n in chain:
        if start >= depth:
            break
        key = getattr(n, "key", None)
        if key is None:
            return None
        if not _node_secured(n):
            return start
        start += len(key)
    return min(depth, start)


def _secured_local(tree_cache: Any, req: Any, depth: int, follow: bool) -> int:
    """#59c: this rank's park depth capped at its secured end, re-probed there (anchor rule). Any failure: the
    uncapped depth (counted as ``unknown``)."""
    try:
        end = secured_end(tree_cache, req, depth)
    except Exception:  # noqa: BLE001 - an instrument of the answer never breaks the park
        logger.warning("#59c PARK-SECURED walk raised for rid=%s: uncapped", getattr(req, "rid", "?"),
                       exc_info=True)
        end = None
    if end is None:
        SECURED_STATS["unknown"] += 1
        return int(depth)
    if end >= int(depth):
        return int(depth)
    from flliper.srt.managers import tp_match_floor

    view = _next_turn_view(req)
    view._compute_max_prefix_len = lambda n, _c=int(end): min(int(n), _c)
    try:
        return max(0, min(int(end), int(tp_match_floor.admission_probe(tree_cache, view, follow=follow))))
    except Exception:  # noqa: BLE001
        return max(0, int(end))


def _tp_min(values: Sequence[int]) -> List[int]:
    import torch
    import torch.distributed as dist

    from flliper.srt.distributed.parallel_state import get_tp_group

    t = torch.tensor(list(values), dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    return [int(v) for v in t.tolist()]


def _group_depths_observed(
    tree_cache: Any,
    reqs: Sequence[Any],
    ps: Any,
    *,
    reduce_min: Optional[Callable[[Sequence[int]], List[int]]] = None,
) -> Tuple[Optional[str], Optional[List[int]], Optional[List[int]]]:
    """:func:`group_depths` plus this rank's OWN pre-reduce depths (the third
    element, None where ``depths`` is None). Observation only: the same probe,
    the same single MIN reduce, no collective added (nf-next-1006-30)."""
    if not reqs:
        return None, None, None
    mode = group_mode(ps)
    if mode in (MODE_NONE, MODE_FOLLOW):
        return mode, None, None
    follow = False
    if mode == MODE_DCP_MIN:
        from flliper.srt.managers import tp_match_floor

        follow = tp_match_floor.this_rank_follows()
    depths = [local_depth(tree_cache, r, follow=follow) for r in reqs]
    own = [max(0, int(d)) for d in depths]
    if mode in (MODE_MIN, MODE_DCP_MIN):
        depths = (reduce_min or _tp_min)(depths)
    return mode, [max(0, int(d)) for d in depths], own


def group_depths(
    tree_cache: Any,
    reqs: Sequence[Any],
    ps: Any,
    *,
    reduce_min: Optional[Callable[[Sequence[int]], List[int]]] = None,
) -> Tuple[Optional[str], Optional[List[int]]]:
    """``(mode, depths)`` for ``reqs`` -- the same list on every rank of the
    group. ``depths`` is None where this rank names nothing (a Form A worker,
    a group that cannot make the depth uniform). The MIN reduce runs only for
    a non-empty list, so every rank enters it together or not at all."""
    mode, depths, _own = _group_depths_observed(tree_cache, reqs, ps, reduce_min=reduce_min)
    return mode, depths


def park_depths_split(
    tree_cache: Any,
    reqs: Sequence[Any],
    ps: Any,
    *,
    reduce_min: Optional[Callable[[Sequence[int]], List[int]]] = None,
    secured: bool = True,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """#59b + #59c: ``(raw, answer)`` -- ``{rid: depth}`` for the requests a flip park holds, after its retraction
    retained their spans. ``raw`` is the admission depth (the parked requests' read clamp); ``answer`` is it capped
    at the group's SECURED end (what the park answer -- ``/pdflip/park_running``, the front's WAIT-BOUND park --
    names, so the front never credits KV the sleep drops). One MIN reduce over both (twice the slots), the same
    on every rank. Both empty where the group names nothing (the front keeps its old price). ``secured=False`` (a
    park no sleep follows -- the priority-lane park, D stays awake and its device KV stays): no cap, ``answer`` is
    ``raw``, one reduce of the depths as before (the flag is the caller's, the same on every rank)."""
    if tree_cache is None or ps is None or not reqs:
        return {}, {}
    mode = group_mode(ps)
    if mode in (MODE_NONE, MODE_FOLLOW):
        return {}, {}
    follow = False
    if mode == MODE_DCP_MIN:
        from flliper.srt.managers import tp_match_floor

        follow = tp_match_floor.this_rank_follows()
    depths = [max(0, int(local_depth(tree_cache, r, follow=follow))) for r in reqs]
    if secured:
        caps = [_secured_local(tree_cache, r, d, follow) for r, d in zip(reqs, depths)]
        if mode in (MODE_MIN, MODE_DCP_MIN):
            both = list((reduce_min or _tp_min)(depths + caps))
            depths, caps = both[:len(reqs)], both[len(reqs):]
    else:
        if mode in (MODE_MIN, MODE_DCP_MIN):
            depths = list((reduce_min or _tp_min)(depths))
        caps = list(depths)
    rids = [str(getattr(r, "rid", "")) for r in reqs]
    raw = {rid: max(0, int(d)) for rid, d in zip(rids, depths)}
    ans = {rid: max(0, min(int(d), int(c))) for rid, d, c in zip(rids, depths, caps)}
    capped = {rid: (raw[rid], ans[rid]) for rid in rids if ans[rid] < raw[rid]}
    if secured:
        SECURED_STATS["rids"] += len(rids)
    SECURED_STATS["capped"] += len(capped)
    SECURED_STATS["tokens"] += sum(a - b for a, b in capped.values())
    if int(getattr(ps, "attn_tp_rank", 0) or 0) == 0:
        logger.info(
            "#59b PARK-RESUMABLE mode=%s %s%s", mode,
            " ".join("%s=%d" % (rid, d) for rid, d in ans.items()),
            (" | #59c PARK-SECURED capped=%s (KV the sleep's reset drops is not credited: device-only nodes on "
             "the path; rids=%d capped=%d tokens=%d unknown=%d)" % (
                 ",".join("%s:%d>%d" % (rid, a, b) for rid, (a, b) in capped.items()),
                 SECURED_STATS["rids"], SECURED_STATS["capped"], SECURED_STATS["tokens"],
                 SECURED_STATS["unknown"])) if capped else "",
        )
    return raw, ans


def park_depths(
    tree_cache: Any,
    reqs: Sequence[Any],
    ps: Any,
    *,
    reduce_min: Optional[Callable[[Sequence[int]], List[int]]] = None,
) -> Dict[str, int]:
    """#59b/#59c: the park ANSWER -- ``{rid: depth}`` capped at the secured end (see :func:`park_depths_split`).
    Empty where the group names nothing (the front keeps its old price)."""
    return park_depths_split(tree_cache, reqs, ps, reduce_min=reduce_min)[1]


def stamp_finished(
    tree_cache: Any,
    reqs: Sequence[Any],
    ps: Any,
    *,
    reduce_min: Optional[Callable[[Sequence[int]], List[int]]] = None,
) -> Optional[str]:
    """Stamp ``time_stats.pdflip_resumable_depth`` on every request of
    ``reqs`` -- the requests whose finishing output this pass streams, the
    same list on every rank (replicated scheduling; the caller's filter is
    ``finished() and not finished_output``). Returns the mode, None when there
    was nothing to stamp (see :func:`group_depths`)."""
    mode, depths, own = _group_depths_observed(tree_cache, reqs, ps, reduce_min=reduce_min)
    if depths is None:
        return mode
    announce = int(getattr(ps, "attn_tp_rank", 0) or 0) == 0
    # nf-next-1006-30 (observation only): a MIN-reduced mode hides which rank
    # took the group down (nf-next-1006-27, class B). TP0's line carries its own
    # pre-reduce depth as ``local_depth=`` (appended after ``mode=``); every other
    # rank logs its own under ``#59 RESUMABLE-LOCAL`` (a different marker, so the
    # ``#59 RESUMABLE rid=`` readers never see it). No collective, no new value.
    local_line = mode in (MODE_MIN, MODE_DCP_MIN) and not announce
    for i, (req, depth) in enumerate(zip(reqs, depths)):
        ts = getattr(req, "time_stats", None)
        if ts is None:
            continue
        ts.pdflip_resumable_depth = max(0, int(depth))
        if announce or local_line:
            seq = len(req.origin_input_ids) + len(getattr(req, "output_ids", None) or ())
            local = int(own[i]) if own is not None else ts.pdflip_resumable_depth
        if announce:
            logger.info(
                "#59 RESUMABLE rid=%s depth=%d seq=%d cached_tokens=%d mode=%s local_depth=%d "
                "branching=%s last_track=%s",
                str(getattr(req, "rid", "")), ts.pdflip_resumable_depth, seq,
                int(getattr(req, "cached_tokens", 0) or 0), mode, local,
                _track_field(req, "mamba_branching_seqlen"),
                _track_field(req, "mamba_last_track_seqlen"),
            )
        elif local_line:
            logger.info(
                "#59 RESUMABLE-LOCAL rid=%s local_depth=%d min=%d seq=%d mode=%s "
                "branching=%s last_track=%s",
                str(getattr(req, "rid", "")), local, ts.pdflip_resumable_depth, seq, mode,
                _track_field(req, "mamba_branching_seqlen"),
                _track_field(req, "mamba_last_track_seqlen"),
            )
    return mode


def stamp_stream(tree_cache: Any, reqs: Sequence[Any], skip_req: Any, ps: Any) -> Optional[str]:
    """The output streamer's hook: stamp exactly the requests whose FINISHING
    output this pass streams -- finished, not streamed as finished before
    (the overlap schedule outputs a finished request twice), not the pass's
    ``skip_req``."""
    finishing = [
        r
        for r in reqs
        if r is not skip_req and r.finished() and not getattr(r, "finished_output", False)
    ]
    return stamp_finished(tree_cache, finishing, ps)


def meta_value(time_stats: Any) -> Optional[int]:
    """The stamped depth as the tokenizer reads it off an output's
    ``time_stats`` (absent attribute or ``UNSET`` -> None)."""
    if time_stats is None:
        return None
    v = getattr(time_stats, FIELD, UNSET)
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


def from_meta_infos(meta_infos: Sequence[Any]) -> Optional[int]:
    """One response's depth from its choices' meta_info: the MIN over the
    choices that carry it (a multi-choice response resumes where all can),
    None when none does."""
    vals = []
    for mi in meta_infos:
        if isinstance(mi, dict) and mi.get(FIELD) is not None:
            try:
                vals.append(int(mi[FIELD]))
            except (TypeError, ValueError):
                continue
    return min(vals) if vals else None
