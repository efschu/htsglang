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


def _tp_min(values: Sequence[int]) -> List[int]:
    import torch
    import torch.distributed as dist

    from flliper.srt.distributed.parallel_state import get_tp_group

    t = torch.tensor(list(values), dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    return [int(v) for v in t.tolist()]


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
    if not reqs:
        return None, None
    mode = group_mode(ps)
    if mode in (MODE_NONE, MODE_FOLLOW):
        return mode, None
    follow = False
    if mode == MODE_DCP_MIN:
        from flliper.srt.managers import tp_match_floor

        follow = tp_match_floor.this_rank_follows()
    depths = [local_depth(tree_cache, r, follow=follow) for r in reqs]
    if mode in (MODE_MIN, MODE_DCP_MIN):
        depths = (reduce_min or _tp_min)(depths)
    return mode, [max(0, int(d)) for d in depths]


def park_depths(
    tree_cache: Any,
    reqs: Sequence[Any],
    ps: Any,
    *,
    reduce_min: Optional[Callable[[Sequence[int]], List[int]]] = None,
) -> Dict[str, int]:
    """#59b: ``{rid: depth}`` for the requests a flip park holds, after its
    retraction retained their spans -- what the park answer
    (``/pdflip/park_running``, also the front's WAIT-BOUND park) names so the
    front caps a parked request's presence credit as it does a finished one.
    Empty where the group names nothing (the front keeps its old price)."""
    if tree_cache is None or ps is None:
        return {}
    mode, depths = group_depths(tree_cache, reqs, ps, reduce_min=reduce_min)
    if depths is None:
        return {}
    out = {str(getattr(r, "rid", "")): d for r, d in zip(reqs, depths)}
    if int(getattr(ps, "attn_tp_rank", 0) or 0) == 0:
        logger.info(
            "#59b PARK-RESUMABLE mode=%s %s", mode,
            " ".join("%s=%d" % (rid[:24], d) for rid, d in out.items()),
        )
    return out


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
    mode, depths = group_depths(tree_cache, reqs, ps, reduce_min=reduce_min)
    if depths is None:
        return mode
    announce = int(getattr(ps, "attn_tp_rank", 0) or 0) == 0
    for req, depth in zip(reqs, depths):
        ts = getattr(req, "time_stats", None)
        if ts is None:
            continue
        ts.pdflip_resumable_depth = max(0, int(depth))
        if announce:
            seq = len(req.origin_input_ids) + len(getattr(req, "output_ids", None) or ())
            logger.info(
                "#59 RESUMABLE rid=%s depth=%d seq=%d cached_tokens=%d mode=%s",
                str(getattr(req, "rid", ""))[:24], ts.pdflip_resumable_depth, seq,
                int(getattr(req, "cached_tokens", 0) or 0), mode,
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
