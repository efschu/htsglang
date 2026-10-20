"""KNOTENTEILUNG (NF rc28, 20.10.): every rank of a TP group cuts its radix tree at the same depths.

Boot ``..._297198745d_1020_174205`` (D, Form A token cut, TP0 host + TP1/TP2 workers):
``PDFLIP PUBLISH-SPLIT`` TP0 7x, TP1/TP2 8x each; at 18:10:02 TP0 held 447 [83072, 87168) +
437 [87168, 87616), the workers 456 [83072, 85248) + 457 [85248, 87616). The R12 keys
(last page hash, depth) of TP0's PUBLISH/L3/STATE then name no worker node (review
nf-review-anker-s74b-1020 A2: the S81 revoke loop; residual risk ``l3_present`` on a larger
worker node). Two causes, both rank-local inputs to a tree edit:

1. THE CHUNK INSERT LENGTH. ``cache_unfinished_req`` inserts up to the mamba track
   (``effective_cache_len`` = min over the components). The track sits at the match's
   branching point (``mamba_branching_seqlen``) when it lies inside the step
   (``schedule_batch`` "track branching point in this forward"). TP0's own full-key walk
   found it (``MATCH-CENSUS-DEEP reached=83072 accepted=82880 branching=83072``); a worker
   adopts the group depth through ``follow_rematch`` on a key CUT at that depth, and the
   re-match carries no branching point (``#59 FINISH-TRACK ... branching=83072`` on TP0,
   ``branching=-`` on TP1/TP2). So pdflip-4-201's first chunk went into TP0's tree up to
   83072 and into the workers' up to 85248, the second chunk made one 4544-token leaf on
   TP0 and a 2368-token leaf on the workers.
   Fix: on a Form A group every rank sets the branching point of its admission match by the
   SAME rule at the end of the OUTER walk (``UnifiedRadixCache._match_post_processor``, after
   the components -- so also after the worker's follow re-match and the host's H96 re-match):
   ``floor(reach, grid)`` when the walk's reach lies beyond the depth this rank admits, else
   none. ``reach`` is the walk over the tree (identical trees -> identical value), the
   admitted depth is the group's by H98/H96 construction (every rank admits it, or stops by
   name). No rank-local anchor verdict enters (the host's and a worker's validators answer
   differently). The extend then filters it by the group-uniform geometry
   (``branching > len(prefix_indices)``, inside the step, on the chunk grid). No collective.

2. THE PUBLISH SPLIT. ``publish_unbacked_sweep`` split a node longer than the window
   (H74: the chunk on every TP rank) into pieces counted from the NODE's start, and only
   when the sweep reached it. The sweep's reach is rank-local: a ``mamba_claim`` refusal
   stops it on TP0 only (``retain_publish.mamba_full_stops_sweep``; 18:09:39, 19:22:36,
   19:26:08: TP0 ``PUBLISH-SWEEP unbacked=1 refused=1`` after ``BACKUP-REFUSED
   why=mamba_claim``, the workers split 452 / 1448 / 1483), and ``backuped`` /
   ``l3_present`` / the issue cap differ per rank as well. A split counted from the node
   start is not confluent: once one rank has split and the other has not, a later insert
   split leaves pieces that never line up again.
   Fix: (a) the split points are ABSOLUTE depths, multiples of the window (a pure function
   of the node's span); (b) a device leaf longer than the window is cut at those points
   where it is CREATED (``_insert_helper``) -- the insert runs on every rank at the same step
   with the same key, so the sweep never meets such a node; the sweep's own split stays as
   the fallback and cuts at the same absolute points.

Switches (default ON, Nutzerorder neue Features an):
``FLLIPER_PDFLIP_FORM_A_UNIFORM_BRANCHING`` (1.) and ``FLLIPER_PDFLIP_PUBLISH_SPLIT_GRID`` (2.);
0 = the rc28 form.

PURE helpers here; the call sites are ``UnifiedRadixCache._match_post_processor`` (1.) and
``UnifiedRadixCache._insert_helper`` / ``_pdflip_split_for_publish`` (2.).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


def uniform_branching_on() -> bool:
    """1. armed: the switch AND a Form A follow group (group-uniform by construction:
    the same role plan and the same environment on every rank)."""
    from flliper.srt.environ import envs

    if not envs.FLLIPER_PDFLIP_FORM_A_UNIFORM_BRANCHING.get():
        return False
    from flliper.srt.managers.tp_match_floor import form_a_follow_active

    return bool(form_a_follow_active())


def uniform_branching(chunk_lens: Sequence[int], grid: int) -> Optional[int]:
    """The branching point every rank of the group derives: the walk's reach floored to
    the grid (``branch_grid`` of the caller: the checkpoint interval, else the FLA chunk).
    None when the floored reach is 0. No anchor verdict enters: the host's and a worker's
    validators answer differently (byteless worker anchors), the reach does not."""
    reach = int(sum(int(n) for n in chunk_lens))
    g = max(1, int(grid or 1))
    aligned = reach // g * g
    return aligned if aligned > 0 else None


def apply_uniform_branching(result, chunk_lens: Sequence[int], interval, grid_fn):
    """1. at the end of an outer admission walk: ``result`` with the group's branching
    point. ``chunk_lens`` = the walk's value chunks, ``interval`` the checkpoint grid
    (None = off), ``grid_fn()`` the FLA chunk grid (called only when needed). A reach not
    beyond the admitted depth has no branching point inside any step the group runs
    (the extend starts at the admitted depth), so it is cleared -- on every rank alike."""
    reach = int(sum(int(n) for n in chunk_lens))
    admitted = len(result.device_indices) + int(getattr(result, "host_hit_length", 0) or 0)
    if reach <= admitted:
        return result._replace(mamba_branching_seqlen=None)
    return result._replace(
        mamba_branching_seqlen=uniform_branching(chunk_lens, interval or grid_fn())
    )


def split_grid_on(tree) -> bool:
    """2. armed on THIS tree: the switch, a TP group (``_attn_reduce_world() > 1``) and
    the H74 group-uniform window -- without H74 the window is a quarter of the rank's own
    host pool and no absolute grid could be the same on every rank."""
    from flliper.srt.environ import envs

    if not envs.FLLIPER_PDFLIP_PUBLISH_SPLIT_GRID.get():
        return False
    if not envs.FLLIPER_PDFLIP_ENABLE_TP_UNIFORM_PUBLISH_WINDOW.get():
        return False
    try:
        return int(tree._attn_reduce_world()) > 1
    except Exception:  # noqa: BLE001 - group-uniform answer: the groups exist on every rank or on none
        return False


def grid_cuts(start: int, length: int, window: int) -> List[int]:
    """Cut offsets (relative to the node's start) for a node spanning
    ``[start, start + length)``: every multiple of ``window`` strictly inside the span.
    Empty when the node is not longer than the window or the window is 0.

    Pure function of the span: two ranks holding the same span cut it at the same
    depths whatever their history, and cutting a piece again yields no new point
    (confluent)."""
    start, length, window = int(start), int(length), int(window)
    if window <= 0 or length <= window:
        return []
    first = (start // window + 1) * window
    return [d - start for d in range(first, start + length, window)]


def node_depth(node) -> int:
    """Absolute start depth of ``node`` in key units: the summed key lengths of its
    ancestors (the root's key is empty)."""
    depth = 0
    p = getattr(node, "parent", None)
    while p is not None:
        depth += len(getattr(p, "key", None) or ())
        p = getattr(p, "parent", None)
    return depth


_EAGER_N = [0]


def note_eager_split(node_id, start: int, length: int, pieces: int, window: int) -> None:
    """Instrument: ``KNOTEN GRID-SPLIT`` (first 16, then every 256th)."""
    _EAGER_N[0] += 1
    n = _EAGER_N[0]
    if n <= 16 or n % 256 == 0:
        logger.info(
            "KNOTEN GRID-SPLIT n=%d node=%s start=%d tokens=%d pieces=%d window=%d (a new device leaf "
            "longer than the window is cut at absolute multiples of it where it is inserted -- the "
            "same step and key on every rank, so every rank holds the same nodes)",
            n, node_id, int(start), int(length), int(pieces), int(window),
        )
