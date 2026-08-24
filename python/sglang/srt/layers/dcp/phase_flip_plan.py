# SPDX-License-Identifier: Apache-2.0
"""Pure planning arithmetic for the #631 PP<->TP phase flip.

The #297 reshard moves KV between two WEIGHTED TOKEN vectors on a fixed
layer set. The phase flip moves KV between two LAYOUTS of the same three
ranks:

* PP layout (prefill): full-attention layer ordinal ``f`` belongs to
  exactly one stage ``s`` (the replicated ``layer_map``); that stage holds
  ALL tokens of its layers, and the stage pool row of global slot ``L`` is
  ``L`` itself (dcp_size=1 -- the compact rule degenerates to identity).
* TP layout (decode): every rank holds ALL layer ordinals, token-sharded
  by the weighted DCP owner rule of ``owner.py``; the pool row is the
  compact row of ``reshard_plan.rows_of``.

Both layouts store rows in the SAME byte format (the fork's weighted DCP
replicates KV heads), so a flip is a pure row redistribution.

The moved cell set of the (stage ``s``, dcp rank ``r``) pair is
``layers(s) x slots_owned_by_r`` -- INDEPENDENT of the flip direction.
Direction only decides which end sends. Enumeration order, the shared
payload-layout convention on both ends (a checksum keeps it falsifiable
at runtime, as in #297): layer ordinals ascending, slots ascending within
a layer, K bytes then V.

Everything here is a pure function of replicated inputs (the live slot
set, the layer map, the token vector): every rank computes the SAME
transition, so pairwise transfer lists agree by construction and the
consensus round never carries plan metadata. Token ownership is
layer-independent, which is why one row list per pair, reused for every
layer of the pair, is enough.
"""

import functools
import itertools
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch

from sglang.srt.layers.dcp.reshard_plan import KvReshardError, owner_of, rows_of

PP_TO_TP = "pp_to_tp"
TP_TO_PP = "tp_to_pp"
_DIRECTIONS = (PP_TO_TP, TP_TO_PP)


def validate_layer_map(
    layer_map: Sequence[Sequence[int]], n_layers: int
) -> Tuple[Tuple[int, ...], ...]:
    """Check that ``layer_map`` partitions ``range(n_layers)`` exactly.

    ``layer_map[s]`` lists the full-attention pool ordinals stage ``s``
    owns. Every ordinal must appear exactly once across all stages (a
    stage MAY be empty -- a degenerate split is legal, a hole or a
    duplicate is not). Returns the normalized tuple form.
    """
    # Ascending order per stage is part of the payload-layout convention
    # (module docstring); normalize here so callers cannot diverge on it.
    norm = tuple(tuple(sorted(int(x) for x in stage)) for stage in layer_map)
    seen: Dict[int, int] = {}
    for s, layers in enumerate(norm):
        for f in layers:
            if not (0 <= f < n_layers):
                raise KvReshardError(
                    f"layer map stage {s} names ordinal {f}, outside [0, {n_layers})"
                )
            if f in seen:
                raise KvReshardError(
                    f"layer map assigns ordinal {f} to both stage "
                    f"{seen[f]} and stage {s}; the PP layout owns every "
                    f"layer exactly once"
                )
            seen[f] = s
    missing = [f for f in range(n_layers) if f not in seen]
    if missing:
        raise KvReshardError(
            f"layer map covers no stage for ordinal(s) {missing}; a hole "
            f"in the partition would silently drop those layers' KV at "
            f"the flip"
        )
    return norm


def default_wave_count(layer_map: Sequence[Sequence[int]]) -> int:
    """The most waves the seam can be split into with every rank paying in.

    A wave is the unit at which the flip releases the source layout's
    backing and restores the destination's, so the two must NET OUT inside
    each wave -- see :func:`layer_waves`. That holds only while every rank
    contributes at least one of its OWN layers per wave, so the ceiling is
    the SMALLEST stage's layer count: 4 on this rig, whose flip layer map
    is [7, 5, 4] over the 16 FULL-ATTENTION ordinals. (Not [28, 20, 16] --
    that is the raw model-layer split from ``--pp-layer-ratio``, a
    different unit, and an earlier version of this docstring conflated the
    two.)

    SUPERSEDED AS THE DEFAULT by :func:`restore_first_wave_count` once the
    seam restores before it releases (#631 2.1b). Still correct, and still
    what the aliased single-wave path reasons about.
    """
    sizes = [len(stage) for stage in layer_map if len(stage)]
    return max(1, min(sizes)) if sizes else 1


def layer_waves(
    layer_map: Sequence[Sequence[int]], n_waves: int
) -> Tuple[Tuple[int, ...], ...]:
    """Split the global layer ordinals into ``n_waves`` balanced waves.

    THE BALANCE IS THE WHOLE POINT, and it is a memory argument rather
    than a load-balancing one. The flip's seam swaps physical backing
    between the two layouts: the source layout's pages for a layer may be
    released once that layer has been read, and the destination layout's
    pages for a layer must be committed before it is written. Doing that
    ONCE for the whole pool forces every byte that crosses the seam to be
    staged at the same instant, which is the unbounded term behind the
    one-request livelock (#631, HANDOFF_666): staging grew with the
    resident live set until no request past a certain length could ever
    flip, and under strict purity that wedges rather than degrades.

    Splitting the seam into waves bounds the staged bytes to ONE wave's
    share -- but only if a wave's releases pay for its commits. Rank ``r``
    releases the wave's layers that IT owns in the PP layout (``L_r/W`` of
    them, each spanning the full PP pool) and commits the wave's layers in
    the TP layout (``64/W`` of them, each spanning its ``w_r`` share of the
    rows). Those are equal exactly when each rank contributes a
    PROPORTIONAL slice of its own block to every wave, which is what this
    split produces: each stage's ascending ordinals are cut into ``n_waves``
    contiguous groups and group ``j`` of every stage forms wave ``j``.

    Integer floors leave at most one layer of drift between the two sides
    at a wave boundary, so the peak carries ONE layer-span of slack. That
    term is priced by ``PhaseFlipRuntime._backing_slack_bytes``, which
    walks the plan for the worst boundary rather than charging a flat
    layer (the flat version over-reserved by 3x on this rig). It is a
    constant of the pool geometry and does not grow with the request.

    THIS SPLIT IS NO LONGER THE DEFAULT. Under restore-first the wave
    order matters as much as the balance, so ``_flip_waves`` asks
    :func:`ordered_layer_waves` instead; this function remains for the
    release-first A/B baseline and for the tests that pin it.
    """
    n_waves = max(1, int(n_waves))
    waves: list[list[int]] = [[] for _ in range(n_waves)]
    for stage_layers in layer_map:
        stage = list(stage_layers)
        total = len(stage)
        for j in range(n_waves):
            lo = (j * total) // n_waves
            hi = ((j + 1) * total) // n_waves
            waves[j].extend(stage[lo:hi])
    return tuple(tuple(sorted(w)) for w in waves)


def restore_first_wave_count(layer_map: Sequence[Sequence[int]]) -> int:
    """The wave ceiling once the seam restores BEFORE it releases (#631 2.1b).

    :func:`default_wave_count` stops at the smallest stage because under
    release-first a wave's own releases must pay for its own commits, and a
    rank with no layer of its own in a wave has nothing to pay with.
    Restore-first drops that per-wave coupling -- the imbalance is carried
    across the whole prefix instead, and :func:`ordered_layer_waves` is what
    keeps that prefix imbalance off the binding cards. So the only remaining
    bound is one layer per wave.
    """
    n_layers = sum(len(stage) for stage in layer_map)
    return max(1, n_layers)


def _seam_shares(tp_vector: Sequence[int]) -> Tuple[float, ...]:
    total = float(sum(tp_vector)) or 1.0
    return tuple(float(w) / total for w in tp_vector)


def seam_transient_peaks(
    order: Sequence[int],
    layer_map: Sequence[Sequence[int]],
    tp_vector: Sequence[int],
    direction: str,
) -> Tuple[float, ...]:
    """Per-rank peak backing transient of a layer ORDER, in layer-spans.

    The unit is one FULL-pool layer span, so a peak of 1.0 means that rank
    holds one extra layer's worth of physical pages at the worst instant
    (~1.953 MiB per 1000 pool tokens on the current rig geometry).

    Restore-first means the commit of position ``j`` lands BEFORE the
    release of position ``j``, so the worst instant of position ``j`` holds
    everything committed through ``j`` against everything released through
    ``j - 1``. The two directions are NOT symmetric:

    * ``tp_to_pp`` -- the destination is PP, where rank ``r`` owns only its
      own stage's layers but each spans the FULL pool; the source is TP,
      where every rank holds every layer at ``share_r`` of the rows. So
      ``r`` commits only on ITS layers and releases a little on every one:
      ``M_r(j) - share_r * j``. Its owned layers therefore want to be LATE.
    * ``pp_to_tp`` -- exactly mirrored, so its owned layers want to be
      EARLY: ``share_r * (j + 1) - M_r(j - 1)``.

    That mirroring is why the order is chosen per direction rather than once
    at boot: a single order cannot be good for both.
    """
    shares = _seam_shares(tp_vector)
    owner = {}
    for r, stage in enumerate(layer_map):
        for f in stage:
            owner[int(f)] = r
    n_ranks = len(shares)
    counts = [0] * n_ranks
    peaks = [0.0] * n_ranks
    for j, layer in enumerate(order):
        r = owner[int(layer)]
        prev = list(counts)
        counts[r] += 1
        for b in range(n_ranks):
            if direction == TP_TO_PP:
                t = counts[b] - shares[b] * j
            else:
                t = shares[b] * (j + 1) - prev[b]
            if t > peaks[b]:
                peaks[b] = t
    return tuple(peaks)


def _transient_at(
    prev: Sequence[int],
    cur: Sequence[int],
    j: int,
    shares: Sequence[float],
    direction: str,
) -> Tuple[float, ...]:
    """Every rank's transient at position ``j``, given the counts around it.

    Kept as one helper so :func:`seam_transient_peaks` (the checker the tests
    assert against) and :func:`_optimal_layer_order` (the chooser) cannot
    drift apart -- a chooser optimising a formula the checker does not use
    would look optimal and behave otherwise.
    """
    out = []
    for b in range(len(shares)):
        if direction == TP_TO_PP:
            t = cur[b] - shares[b] * j
        else:
            t = shares[b] * (j + 1) - prev[b]
        out.append(t if t > 0.0 else 0.0)
    return tuple(out)


@functools.lru_cache(maxsize=64)
def _optimal_layer_order(
    layer_map: Tuple[Tuple[int, ...], ...],
    tp_vector: Tuple[int, ...],
    direction: str,
) -> Tuple[int, ...]:
    """The layer order minimising the binding cards' peak backing transient.

    TWO scalar dynamic programs rather than one vector one, because a
    lexicographic minimum over componentwise maxima is not a valid DP -- a
    path that is worse on the first objective can be the only route to the
    best second one. Run separately, each pass IS a correct min-max DP:

    1. minimise the worst BINDING-rank peak (every rank except the one with
       the largest share -- the card sized to absorb this);
    2. over the transitions that keep pass 1 optimal, minimise the big
       card's own peak, so absorbing the transient does not become an
       excuse to pile the whole flip onto it.

    State is the per-rank count of layers already placed; position ``j`` is
    implied by the count sum, which is what keeps the state small.
    """
    shares = _seam_shares(tp_vector)
    n_ranks = len(layer_map)
    sizes = [len(stage) for stage in layer_map]
    n_layers = sum(sizes)
    if n_layers == 0:
        return ()
    big = max(range(n_ranks), key=lambda r: (shares[r], -r)) if n_ranks else 0
    binding = [r for r in range(n_ranks) if r != big]

    states = sorted(itertools.product(*[range(s + 1) for s in sizes]), key=sum)
    start = tuple([0] * n_ranks)

    def _passes(score_of):
        best: Dict[Tuple[int, ...], float] = {start: 0.0}
        parent: Dict[Tuple[int, ...], int] = {}
        for st in states:
            if st == start:
                continue
            j = sum(st) - 1
            top = None
            for r in range(n_ranks):
                if st[r] == 0:
                    continue
                prev = list(st)
                prev[r] -= 1
                prev_t = tuple(prev)
                if prev_t not in best:
                    continue
                t = _transient_at(prev_t, st, j, shares, direction)
                cand = max(best[prev_t], score_of(t))
                if top is None or cand < top[0] - 1e-12:
                    top = (cand, r)
            if top is not None:
                best[st] = top[0]
                parent[st] = top[1]
        return best, parent

    best_b, parent_b = _passes(lambda t: max((t[b] for b in binding), default=0.0))
    full = tuple(sizes)
    optimum = best_b.get(full, 0.0)

    # Pass 2 walks only the transitions that keep pass 1 at its optimum, by
    # scoring any other transition as unreachable.
    def _score_big(t):
        return t[big]

    best2: Dict[Tuple[int, ...], float] = {start: 0.0}
    parent2: Dict[Tuple[int, ...], int] = {}
    for st in states:
        if st == start:
            continue
        j = sum(st) - 1
        top = None
        for r in range(n_ranks):
            if st[r] == 0:
                continue
            prev = list(st)
            prev[r] -= 1
            prev_t = tuple(prev)
            if prev_t not in best2 or prev_t not in best_b:
                continue
            t = _transient_at(prev_t, st, j, shares, direction)
            bind_here = max(best_b[prev_t], max((t[b] for b in binding), default=0.0))
            if bind_here > optimum + 1e-12:
                continue
            cand = max(best2[prev_t], _score_big(t))
            if top is None or cand < top[0] - 1e-12:
                top = (cand, r)
        if top is not None:
            best2[st] = top[0]
            parent2[st] = top[1]

    # Pass 2 reaches the full state whenever pass 1 did -- keeping every
    # pass-1-optimal transition is enough to keep at least one whole path.
    # The fallback is defensive only, and it degrades to a still-optimal
    # binding peak rather than to nothing.
    use_parent = parent2 if full in best2 else parent_b
    st = full
    picks: list[int] = []
    while st != start:
        r = use_parent[st]
        picks.append(r)
        nxt = list(st)
        nxt[r] -= 1
        st = tuple(nxt)
    picks.reverse()

    # Within a rank, its own layers go in ascending ordinal order: they all
    # span the same bytes, so only the RANK sequence carries the memory
    # argument, and ascending keeps the pack/exchange legs' layer indexing
    # unsurprising.
    cursor = [0] * n_ranks
    order: list[int] = []
    for r in picks:
        order.append(int(layer_map[r][cursor[r]]))
        cursor[r] += 1
    return tuple(order)


def ordered_layer_waves(
    layer_map: Sequence[Sequence[int]],
    tp_vector: Sequence[int],
    n_waves: int,
    direction: str,
) -> Tuple[Tuple[int, ...], ...]:
    """Layer waves ORDERED so the backing transient lands on the big card.

    #631 section 2.1b, move 3, and it is the non-obvious one. Restore-first
    lifts the wave cap, but it also means a wave commits before it releases,
    so SOMEBODY holds both for an instant. That instant is unavoidable --
    what it is not is unassignable.

    Whoever owns the FIRST layer processed pays a full layer span with
    nothing yet released. Give that layer to the rank with the largest TP
    share -- which is the card the operator sized largest, since the TP
    vector is what expresses relative card capacity -- and delay every other
    rank's first owned layer until its proportional releases have paid for
    it. On the rig's ``[7, 5, 4]`` that means both 3080s pay approximately
    zero and the 5090 absorbs about one layer span.

    Derived from the replicated layer map and TP vector ONLY, like
    :func:`layer_waves`: both ends of every pair must cut the wire the same
    way, and the direction is already agreed by consensus before the seam
    runs.

    HOW MUCH THIS ACTUALLY BUYS, measured rather than assumed. On the rig
    (``[7, 5, 4]`` layers, share vector ``(14, 10, 8)``) the exhaustive
    optimum is a binding-card peak of 0.5 layer spans against 0.688 for the
    proportional W=4 split -- a 27% cut, NOT a cut to zero. Zero is
    arithmetically unreachable and it is worth recording why, because the
    design note this implements claimed otherwise: rank 1 holds 5 of 16
    layers at a 0.3125 share, so even with its last layer placed dead last
    it ends at ``5 - 0.3125 * 15 = 0.3125``; rank 2 likewise floors at
    ``4 - 0.25 * 15 = 0.25``; and both floors need the SAME final position,
    so no order attains both. Delaying each card's FIRST owned layer past
    ``1 / share_r`` -- the rule the note gives -- fixes only the first term
    and says nothing about the catch-up the remaining layers must do.

    An exact search is affordable because the layer map is fixed at boot:
    the state is the per-rank placed count, so there are
    ``prod(L_r + 1)`` states (240 on this rig, ~10k on a 64-layer map), and
    the result is memoised per (map, vector, direction).
    """
    n_waves = max(1, int(n_waves))
    n_layers = sum(len(s) for s in layer_map)
    if n_layers == 0:
        return tuple(() for _ in range(n_waves))

    order = _optimal_layer_order(
        tuple(tuple(int(f) for f in stage) for stage in layer_map),
        tuple(int(w) for w in tp_vector),
        str(direction),
    )

    waves: list[list[int]] = [[] for _ in range(n_waves)]
    for idx, layer in enumerate(order):
        lo = (idx * n_waves) // n_layers
        waves[min(lo, n_waves - 1)].append(layer)
    # Waves keep the CHOSEN order between them; within a wave the ordinals
    # are sorted because the pack/exchange legs index layers ascending and
    # both ends must agree on that cut.
    return tuple(tuple(sorted(w)) for w in waves)


@dataclass
class PhaseFlipTransition:
    """One rank's share of a PP<->TP layout flip.

    Row tensors are int64 on CPU and refer to rows in THIS rank's pool of
    the respective side: PP-side rows are global slot ids, TP-side rows
    are weighted compact rows. ``send_rows[p]`` / ``recv_rows[p]`` are
    reused for every layer in ``send_layers[p]`` / ``recv_layers[p]``
    (token ownership is layer-independent). Layer tuples are ascending;
    slot enumeration within a pair is ascending on both ends.
    """

    rank: int
    direction: str
    n_layers: int
    layer_map: Tuple[Tuple[int, ...], ...]
    tp_vector: Tuple[int, ...]
    #: my stage's layer ordinals (both directions: the local move covers
    #: my layers x my dcp-owned slots).
    local_layers: Tuple[int, ...]
    #: PP-pool rows (= slot ids) of my dcp-owned slots.
    local_pp_rows: torch.Tensor
    #: TP-pool compact rows of the same slots, same order.
    local_tp_rows: torch.Tensor
    #: peer -> layer ordinals covered by the payload I send to that peer.
    send_layers: Dict[int, Tuple[int, ...]]
    #: peer -> rows in MY (sending-side) pool, one list per pair.
    send_rows: Dict[int, torch.Tensor]
    #: peer -> layer ordinals covered by the payload that peer sends me.
    recv_layers: Dict[int, Tuple[int, ...]]
    #: peer -> rows in MY (receiving-side) pool the payload fills.
    recv_rows: Dict[int, torch.Tensor]
    total_slots: int

    @property
    def outgoing_cells(self) -> int:
        return sum(
            len(self.send_layers[p]) * int(r.numel()) for p, r in self.send_rows.items()
        )

    @property
    def incoming_cells(self) -> int:
        return sum(
            len(self.recv_layers[p]) * int(r.numel()) for p, r in self.recv_rows.items()
        )

    @property
    def moves_nothing(self) -> bool:
        """Does this plan carry any KV at all?

        Under #856 the seam retracts every resident and drops the tree before
        the wave loop, so the plan is rebuilt EMPTY by construction and the
        loop walks waves with nothing in them (W27-retry measured 16 of them
        for ~314 ms). This is the predicate that lets the seam say so.

        ALL THREE LEGS, not just the peer exchange. A plan with no send and no
        recv can still have a LOCAL move -- my layers crossed with my own
        dcp-owned slots -- and skipping the loop on a plan that still has one
        would drop KV silently, which is the one failure mode worse than the
        314 ms.
        """
        return (
            self.outgoing_cells == 0
            and self.incoming_cells == 0
            and int(self.local_pp_rows.numel()) == 0
            and int(self.local_tp_rows.numel()) == 0
        )

    def max_pp_row(self) -> int:
        """Largest PP-pool row this rank touches (bounds-check input)."""
        mx = -1
        if self.local_pp_rows.numel():
            mx = max(mx, int(self.local_pp_rows.max().item()))
        rows = self.send_rows if self.direction == PP_TO_TP else self.recv_rows
        for r in rows.values():
            if r.numel():
                mx = max(mx, int(r.max().item()))
        return mx

    def max_tp_row(self) -> int:
        """Largest TP-pool row this rank touches (bounds-check input)."""
        mx = -1
        if self.local_tp_rows.numel():
            mx = max(mx, int(self.local_tp_rows.max().item()))
        rows = self.recv_rows if self.direction == PP_TO_TP else self.send_rows
        for r in rows.values():
            if r.numel():
                mx = max(mx, int(r.max().item()))
        return mx


def build_phase_flip_transition(
    slots: torch.Tensor,
    layer_map: Sequence[Sequence[int]],
    n_layers: int,
    tp_vector: Sequence[int],
    rank: int,
    direction: str,
) -> PhaseFlipTransition:
    """Derive this rank's flip transition from the replicated inputs.

    ``slots``: int64, sorted ascending, unique (checked -- the pairwise
    payload layout is derived from this order on both ends, as in #297).
    ``layer_map``: stage -> full-attention pool ordinals, a partition of
    ``range(n_layers)``. ``tp_vector``: the weighted DCP token vector of
    the TP layout. ``direction``: :data:`PP_TO_TP` or :data:`TP_TO_PP`.
    """
    if direction not in _DIRECTIONS:
        raise KvReshardError(
            f"unknown flip direction {direction!r}; expected one of {_DIRECTIONS}"
        )
    vec = tuple(int(x) for x in tp_vector)
    norm_map = validate_layer_map(layer_map, n_layers)
    if len(norm_map) != len(vec):
        raise KvReshardError(
            f"layer map has {len(norm_map)} stages but the TP vector "
            f"{vec} has {len(vec)} ranks; the flip reuses the SAME "
            f"three ranks, so the counts must match"
        )
    n_ranks = len(vec)
    if not (0 <= rank < n_ranks):
        raise KvReshardError(f"rank {rank} out of range for {n_ranks} ranks")
    slots = slots.detach().to("cpu", torch.int64)
    if slots.numel():
        if not bool((slots[1:] > slots[:-1]).all()):
            raise KvReshardError(
                "live slot ids must be sorted ascending and unique -- the "
                "pairwise payload layout is derived from this order on "
                "both ends"
            )
        if int(slots[0].item()) < 0:
            raise KvReshardError("negative KV slot id in live set")

    owner = owner_of(slots, vec)
    my_slots = slots[owner == rank]
    my_layers = norm_map[rank]
    local_pp = my_slots.clone()  # PP row of slot L is L itself (dcp_size=1)
    local_tp = rows_of(my_slots, vec, rank)

    send_layers: Dict[int, Tuple[int, ...]] = {}
    send_rows: Dict[int, torch.Tensor] = {}
    recv_layers: Dict[int, Tuple[int, ...]] = {}
    recv_rows: Dict[int, torch.Tensor] = {}
    for peer in range(n_ranks):
        if peer == rank:
            continue
        peer_slots = slots[owner == peer]
        peer_layers = norm_map[peer]
        if direction == PP_TO_TP:
            # I am stage `rank` (sender side) and dcp rank `rank`
            # (receiver side): send my layers x peer's slots; receive
            # peer's layers x my slots into my compact rows.
            if my_layers and peer_slots.numel():
                send_layers[peer] = my_layers
                send_rows[peer] = peer_slots.clone()
            if peer_layers and my_slots.numel():
                recv_layers[peer] = peer_layers
                recv_rows[peer] = local_tp.clone()
        else:
            # TP_TO_PP: send peer's layers x my slots out of my compact
            # rows; receive my layers x peer's slots at their slot ids.
            if peer_layers and my_slots.numel():
                send_layers[peer] = peer_layers
                send_rows[peer] = local_tp.clone()
            if my_layers and peer_slots.numel():
                recv_layers[peer] = my_layers
                recv_rows[peer] = peer_slots.clone()

    return PhaseFlipTransition(
        rank=rank,
        direction=direction,
        n_layers=n_layers,
        layer_map=norm_map,
        tp_vector=vec,
        local_layers=my_layers,
        local_pp_rows=local_pp,
        local_tp_rows=local_tp,
        send_layers=send_layers,
        send_rows=send_rows,
        recv_layers=recv_layers,
        recv_rows=recv_rows,
        total_slots=int(slots.numel()),
    )
