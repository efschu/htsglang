# SPDX-License-Identifier: Apache-2.0
"""PRECISION TAIL (#1243), SLICE 1 -- the bf16 ring, its mapping, and the
index partition that splits one paged read into a body half and a tail half.

Design authority: ``/spinning/gpu-arb/weg2/WEG2_KVTAIL_DESIGN_BASIS_0909.md``
(section 7 overrides sections 2 and 4), implemented per
``/spinning/gpu-arb/weg2/WEG2_KVTAIL_SPEC_0909.md`` Part 2.

WHAT THIS SLICE IS, AND WHAT IT DELIBERATELY IS NOT.

Slice 1 is the D side only: a SECOND ``MHATokenToKVPool`` at bf16 (the ring),
a plain ``TokenToKVPoolAllocator`` over it, a dense body-slot -> ring-row
mapping, and the per-request split of the existing weighted-DCP index vector
into a body prefix and a tail suffix.  The write is a DOUBLE write -- the body
row is still written exactly as today.  ``min == max``, so the ring is not yet
elastic.  No sidecar (slice 4), no draft tail (slice 5), no pressure rungs
(slice 2), no P-side ring (slice 6), no extend-side tail (slice 2).

THE ONE NEW BOOKKEEPING, AND WHY IT IS NOT A SECOND ONE.  ``full_to_tail``
maps a COMPACTED PHYSICAL SLOT -- the value the weighted owner rule already
produced (``layers/dcp/owner.py`` ``dcp_weighted_write_slots`` on the write
side, its documented exact inverse ``build_dcp_weighted_kv_indices`` on the
read side) -- to a ring row.  It never recomputes ownership, never holds a
position, and never appears in ``req_to_token``.  Everything that decides
WHICH tokens this rank stores stays where it is.

TWO INDEX SPACES, NOT ONE.  The mapping is keyed by the COMPACTED PHYSICAL
SLOT.  The BODY ALLOCATOR under weighted DCP is NOT in that space: it is sized
over the GLOBAL logical context C, and the pool holds only
``dcp_compact_pool_rows(C, cp_S, cp_ratio)`` rows.  Every index arriving from
its free path is therefore translated by ``to_compact_slots`` through the SAME
owner primitive the write used.  Indexing the mapping raw was the first cut's
worst defect: for a high slot it runs off the end of the mapping, and for a low
one it silently frees a DIFFERENT live token's ring row while the true owner
keeps a pointer to a row that is back in the free list.

THE ARM KEEPS THE RING OFF THE EXTEND PATH.  ``_dcp_write_scatter`` is the
extend write as well as the decode write, and the age-out that bounds the
window runs only at DECODE plan time.  An always-claiming ring therefore filled
front-to-back with the OLDEST prompt tokens of a long prefill and clamped away
exactly the newest ones the tail exists for.  So the ring is ARMED by the
decode plan and DISARMED at the top of every forward; that also collapses the
per-LAYER allocator mutation and its host sync into ONE per step, on the decode
plan path #616c deliberately de-synced.

THE NULL IS -1, NOT 0.  The SWA translation table this is modelled on
(``mem_cache/allocator/swa.py:82``) can use 0 as its null because SWA slot 0
is reserved.  Ours is written as -1 and read as ``mapping >= 0`` so that the
null does not depend on an allocator's reservation convention at all: a
0-valued null under a ``>= 0`` predicate reads EVERY unmapped body slot as
ring row 0, which is a silent whole-pool aliasing rather than a crash.

THE COUNTER IS THE POINT.  Boot ``weg2kvtail1`` printed a banner and no
counter, and its four arms came back bit-identical at all 53,047 scored
positions -- a probe that never engaged is indistinguishable from a probe
that engaged and changed nothing, unless something counts.  So every plan
prints ``attended_rows``, ``body_rows`` and ``untrimmed_owned``, and their
identity is asserted; and a ring that HOLDS rows while attending none is a
refusal by name (W56), not a quiet no-op.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "KV_TAIL_NULL",
    "KV_TAIL_OPEN",
    "KvTailKnobs",
    "KvTailRing",
    "Weg2KvTailUnfundable",
    "Weg2KvTailNoOp",
    "Weg2KvTailFormRefused",
    "auto_ring_rows",
    "install_kv_tail_ring",
    "position_tail_lengths",
    "split_owned_indices",
    "tail_window_owned_lengths",
]

#: The mapping's null.  Read as ``mapping >= 0``; see the module docstring for
#: why this is not 0.
KV_TAIL_NULL: int = -1

#: ``--kv-tail-max-tokens`` sentinel for "open upwards" (basis 7.7).  It is -1
#: and NOT 0, because 7.4 makes ``0`` a real, meaningful value (a hard zero
#: tail) and because ``0`` as a sentinel collides with the ``max < min``
#: validator at the shipped defaults (``0 < 16384``).
KV_TAIL_OPEN: int = -1


# ---------------------------------------------------------------------------
# Refusals.  W-codes taken from >= 54 by enumeration (spec section 1.6), so the
# free-list assertion of test_weg2_wcode_uniqueness_1263.py (which pins
# 15/18/23/39 as free) is left untouched.
# ---------------------------------------------------------------------------


class Weg2KvTailUnfundable(RuntimeError):
    """W54 Weg2KvTailUnfundable -- the ring or the guaranteed minimum cannot be
    funded, or a knob combination cannot be honoured.

    Refused, never clamped: silently lowering the ring gives back precision the
    operator asked for, and silently raising it takes KV tokens the operator
    did not offer.  Both choices belong to the operator and both are named in
    the message (the ``server_args.py`` mamba-floor validators are the shape).
    """


class Weg2KvTailNoOp(RuntimeError):
    """W56 Weg2KvTailNoOp -- the tail rule was DUE on this form and attended
    nothing.

    This is boot ``weg2kvtail1``'s failure made loud.  There the ring was wiped
    once per LAYER (a deleter sitting between the writer and its only reader,
    separated by the layer event), so no ring ever held a row, the eviction
    loop found nothing to do, and four arms came back bit-identical with the
    banner printed.  Two independent contradictions raise here: rows are held
    and none is attended; and rows were claimed, none are held, and none was
    ever released.
    """


class Weg2KvTailFormRefused(RuntimeError):
    """W58 Weg2KvTailFormRefused -- the tail cannot be honoured on this pool
    form, so it is refused by name instead of installed as a no-op.

    The ring is a second pool of the SAME geometry as the body pool and the
    mapping is indexed by a row of that pool.  A paged (``page_size > 1``), HND
    or vectorized-5d body pool, an MLA pool, or a VA-backed / post-capture pool
    each break one of those two assumptions in a way that is not a crash but a
    wrong row.
    """


# ---------------------------------------------------------------------------
# Knobs (spec section 1.4).  Named defaults only; every sentinel is checked
# BEFORE any comparison.
# ---------------------------------------------------------------------------


@dataclass
class KvTailKnobs:
    """The tail's operator surface, with the basis item each knob serves.

    ``min_tokens``   -- 7.6, default 16384.  0 is legal (7.4).
    ``max_tokens``   -- 7.7, default OPEN (``KV_TAIL_OPEN`` = -1).
    ``ring_rows``    -- the real elasticity bound under the boot-fixed-ring
                        deviation (spec decision D2).  ``None`` = auto.
    ``host_max_tokens``       -- 7.5 needs a value while 7.7's max is open.
    ``shrink_hysteresis_rounds`` -- 2.2; implemented in slice 2, validated here.
    ``virtual_fp8`` / ``sidecar`` / ``draft`` -- slices 3 / 4 / 5, off here.
    """

    #: Basis 7.6 sets the OPERATOR default of ``--kv-tail-min-tokens`` to
    #: 16384. Slice 1 ships the dataclass (and the flag) at 0 -- i.e. the tail
    #: OFF -- and 16384 is the value the slice-1 boot arms pass explicitly.
    #: NAMED DEVIATION, not an oversight: at the shipped ``--d-bs 8`` a 16384
    #: guarantee is 8 requests x 16384 tokens x 64 KiB = 8 GiB of the group's
    #: VRAM, and the spec puts that number in front of the user BEFORE it
    #: becomes a default. Until then the default path must be byte-identical,
    #: which a non-zero default here would silently end.
    min_tokens: int = 0
    max_tokens: int = KV_TAIL_OPEN
    ring_rows: Optional[int] = None
    host_max_tokens: Optional[int] = None
    shrink_hysteresis_rounds: Optional[int] = None
    virtual_fp8: bool = False
    sidecar: bool = False
    draft: bool = False

    @property
    def enabled(self) -> bool:
        """The tail is OFF unless a positive minimum or an explicit ring was
        asked for.  ``--kv-tail-min-tokens 0`` with no ring is the identity
        arm: nothing is constructed, nothing is planned, and the default path
        is byte-identical."""
        if self.ring_rows is not None and self.ring_rows > 0:
            return True
        return self.min_tokens > 0

    def resolved_host_max(self) -> int:
        if self.host_max_tokens is None:
            return self.min_tokens
        return self.host_max_tokens

    def validate(self, page_size: int = 1) -> None:
        """Spec section 1.4, IN THIS ORDER.  The sentinel is resolved first, so
        the open default cannot be caught by the ``max < min`` rule it has
        nothing to do with."""
        if self.min_tokens < 0:
            raise Weg2KvTailUnfundable(
                "W54 Weg2KvTailUnfundable: --kv-tail-min-tokens "
                f"{self.min_tokens} is negative. 0 is legal (basis 7.4, no "
                "guaranteed 16-bit portion); a negative minimum is not a "
                "smaller tail, it is an unreadable one."
            )
        # (1) sentinel BEFORE any comparison against max.
        if self.max_tokens != KV_TAIL_OPEN:
            if self.max_tokens < 0:
                raise Weg2KvTailUnfundable(
                    "W54 Weg2KvTailUnfundable: --kv-tail-max-tokens "
                    f"{self.max_tokens} is negative and is not the open "
                    f"sentinel ({KV_TAIL_OPEN}). 0 is a REAL value (a hard "
                    "zero tail), which is exactly why it may not double as "
                    "the sentinel."
                )
            # (2)
            if self.max_tokens < self.min_tokens:
                raise Weg2KvTailUnfundable(
                    "W54 Weg2KvTailUnfundable: --kv-tail-max-tokens "
                    f"{self.max_tokens} is below --kv-tail-min-tokens "
                    f"{self.min_tokens}. The minimum is the GUARANTEED, "
                    "ledger-priced portion (basis 7.6) and the maximum is the "
                    "elastic ceiling above it (basis 7.7); a ceiling under the "
                    "floor has no reading. Pass "
                    f"--kv-tail-max-tokens {KV_TAIL_OPEN} for open."
                )
        # (4)
        if self.ring_rows is not None and self.ring_rows < self.min_tokens:
            raise Weg2KvTailUnfundable(
                "W54 Weg2KvTailUnfundable: --kv-tail-ring-rows "
                f"{self.ring_rows} is below the guaranteed minimum "
                f"--kv-tail-min-tokens {self.min_tokens}. The minimum is a "
                "per-rank budget post that must always fit for an admitted "
                "request (basis 7.2); a ring smaller than one request's floor "
                "cannot honour it. Raise --kv-tail-ring-rows to at least "
                f"{self.min_tokens}, or lower --kv-tail-min-tokens."
            )
        # (5)
        host_max = self.resolved_host_max()
        if host_max < self.min_tokens:
            raise Weg2KvTailUnfundable(
                "W54 Weg2KvTailUnfundable: --kv-tail-host-max-tokens "
                f"{host_max} is below --kv-tail-min-tokens "
                f"{self.min_tokens}. The guaranteed tail must always have a "
                "host-tier landing place at the flip (basis 7.5), so the host "
                "post may not be smaller than the device floor it carries."
            )
        # (6)
        if self.sidecar and page_size != 1:
            raise Weg2KvTailUnfundable(
                "W54 Weg2KvTailUnfundable: --kv-tail-sidecar needs "
                f"--page-size 1, got {page_size}. The sidecar's tail length is "
                "not stored -- a page header is forbidden in the canonical "
                "store (canonical_page_store.py:60-63) -- it is recovered as "
                "the trailing-run count of keys carrying a tail16 component "
                "(PoolHitPolicy.TRAILING_PAGES). Above page-size 1 that count "
                "is PAGES, not tokens, and must be re-derived rather than "
                "scaled."
            )


def auto_ring_rows(
    max_running_requests: int,
    min_tokens: int,
    owned_share_num: int,
    owned_share_den: int,
) -> int:
    """``--kv-tail-ring-rows auto`` (spec R6).

    ``max_running_requests`` IS READ, NEVER TYPED.  The launcher has a function
    whose whole reason for existing is that a previous consumer restated this
    number instead of reading it and became wrong when the flag moved
    (``weg2/launcher.py:3790``); the group's D value is emitted from ``--d-bs``,
    whose default is 8.  A typed 4 or 8 here would silently stop tracking it,
    which is why the caller passes the value it read and the test moves it.

    ``owned_share_num / owned_share_den`` is this rank's live DCP share
    (``cp_ratio / cp_S`` under the weighted owner rule), read from the backend,
    never a hardcoded vector.
    """
    if max_running_requests <= 0:
        raise Weg2KvTailUnfundable(
            "W54 Weg2KvTailUnfundable: --kv-tail-ring-rows auto needs a "
            f"positive max_running_requests, got {max_running_requests}. The "
            "ring is sized from the group's own concurrency; a zero means the "
            "value was not read off the argv it lives on."
        )
    if owned_share_den <= 0 or owned_share_num <= 0:
        raise Weg2KvTailUnfundable(
            "W54 Weg2KvTailUnfundable: --kv-tail-ring-rows auto needs a "
            f"positive DCP share, got {owned_share_num}/{owned_share_den}."
        )
    group_tokens = max_running_requests * min_tokens
    # Ceil, so the three ranks of an uneven split together cover the group's
    # guaranteed tail rather than falling one row short of it.
    return -(-group_tokens * owned_share_num // owned_share_den)


# ---------------------------------------------------------------------------
# The index partition (spec R7).  Pure tensor functions: no device, no
# collective, no host sync -- so the arithmetic that decides whether the split
# is CORRECT can be pinned on CPU against an independent reference.
# ---------------------------------------------------------------------------


def tail_window_owned_lengths(
    owned: torch.Tensor,
    lens: torch.Tensor,
    tail_lens: torch.Tensor,
) -> torch.Tensor:
    """How many of a request's OWNED slots fall inside its tail window.

    ``owned`` is the flat per-request ownership mask in request order (exactly
    the vector ``build_dcp_weighted_kv_indices`` computes), ``lens`` the global
    per-request lengths, ``tail_lens`` the per-request tail length in GLOBAL
    POSITIONS.  The window is the last ``tail_lens[i]`` positions of request
    ``i``, so this is a segmented sum over the segment suffix.

    Global positions in, owned ROW counts out.  That asymmetry is the whole
    rank-uniformity argument (basis 2.6): the boundary is a position, identical
    on every rank without any reduce; only how many rows it covers differs.
    """
    lens64 = lens.to(torch.int64)
    tail64 = torch.clamp(tail_lens.to(torch.int64), min=0)
    tail64 = torch.minimum(tail64, lens64)
    bs = lens64.numel()
    out = torch.zeros(bs, dtype=torch.int64, device=lens64.device)
    total = int(owned.numel())
    if total == 0 or bs == 0:
        return out
    seg = torch.repeat_interleave(
        torch.arange(bs, device=lens64.device, dtype=torch.int64),
        lens64,
        output_size=total,
    )
    starts = torch.zeros(bs + 1, dtype=torch.int64, device=lens64.device)
    starts[1:] = torch.cumsum(lens64, dim=0)
    pos = torch.arange(total, device=lens64.device, dtype=torch.int64) - starts[seg]
    in_window = pos >= (lens64 - tail64)[seg]
    out.scatter_add_(0, seg, (owned.to(torch.bool) & in_window).to(torch.int64))
    return out


def split_owned_indices(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    owned_tail_len: torch.Tensor,
    mapping: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ONE weighted-DCP paged plan into ``(body_indptr, body_indices,
    tail_indptr, tail_ring_indices, age_out_slots)``.

    ``build_dcp_weighted_kv_indices`` already emits each request's owned slots
    in INCREASING POSITION ORDER, compacted, with per-request counts in
    ``kv_indptr``.  So the window is a per-request SUFFIX of that vector and the
    body is its prefix; no new kernel and no second ownership pass.

    A slot is in the TAIL half iff it is inside the window AND it currently
    holds a ring row.  The second condition is not a safety net, it is the
    definition: a token has a bf16 row or it does not, and every token without
    one must be attended from the body.  The two halves are therefore a
    PARTITION of the untrimmed vector, which is what makes
    ``attended_rows + body_rows == untrimmed_owned`` an identity able to
    falsify a double count or a gap rather than a hopeful log line.

    ``age_out_slots`` is the converse: slots that have LEFT the window and still
    hold a ring row.  Ageing out is decided HERE, once per step, out of graph,
    because this is the only place that holds a request's whole owned slot list
    in position order; deciding it at the write site would mean re-deriving the
    request's history from ``req_to_token`` a second time.

    THE BODY TRIM IS REQUIRED FROM SLICE 1.  Under a double write an untrimmed
    body plan plus a tail pass attends the same token twice; that is not a
    performance detail, it is a wrong answer.

    HAZARD, NAMED RATHER THAN HIDDEN: the boolean masking below is an implicit
    device->host sync on CUDA, and this function runs inside the decode plan
    window that #616c deliberately de-synced (``total_tokens`` exists precisely
    to remove the one blocking D2H that wedged three ranks against each other's
    BAR1 spin).  It is out of graph and once per step, not per layer, so it is
    the same order of cost as the plan itself -- but it is an ADDED sync on
    that path and it is UNMEASURED on metal.  If a wedge appears at the decode
    plan under load, this is the first line to suspect, and the fix is a
    fixed-shape gather against a fixed capacity rather than a mask.  It is one
    sync per STEP, not per layer: the per-layer syncs the write side once had
    are gone (see ``begin_decode_step``).
    """
    bs = int(kv_indptr.numel()) - 1
    device = kv_indices.device
    total = int(kv_indices.numel())
    empty_i32 = torch.zeros(0, dtype=kv_indices.dtype, device=device)
    if total == 0 or bs <= 0:
        z = torch.zeros(max(bs + 1, 1), dtype=torch.int32, device=device)
        return z, empty_i32, z.clone(), empty_i32.clone(), empty_i32.clone()
    lens = kv_indptr[1:].to(torch.int64) - kv_indptr[:bs].to(torch.int64)
    tail_len = torch.clamp(owned_tail_len.to(torch.int64), min=0)
    tail_len = torch.minimum(tail_len, lens)
    seg = torch.repeat_interleave(
        torch.arange(bs, device=device, dtype=torch.int64), lens, output_size=total
    )
    starts = kv_indptr[:bs].to(torch.int64)
    pos = torch.arange(total, device=device, dtype=torch.int64) - starts[seg]
    in_window = pos >= (lens - tail_len)[seg]
    if mapping is None:
        mapped = torch.zeros(total, dtype=torch.bool, device=device)
        ring_rows = torch.full((total,), KV_TAIL_NULL, dtype=torch.int64, device=device)
    else:
        ring_rows = mapping[kv_indices.to(torch.int64)].to(torch.int64)
        mapped = ring_rows >= 0
    is_tail = in_window & mapped
    age_out = mapped & ~in_window
    tail_ring_indices = ring_rows[is_tail].to(kv_indices.dtype)
    body_indices = kv_indices[~is_tail]
    age_out_slots = kv_indices[age_out]
    tail_per_req = torch.zeros(bs, dtype=torch.int64, device=device)
    tail_per_req.scatter_add_(0, seg, is_tail.to(torch.int64))
    body_per_req = lens - tail_per_req
    tail_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    body_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    tail_indptr[1:] = torch.cumsum(tail_per_req, dim=0).to(torch.int32)
    body_indptr[1:] = torch.cumsum(body_per_req, dim=0).to(torch.int32)
    return body_indptr, body_indices, tail_indptr, tail_ring_indices, age_out_slots


def position_tail_lengths(
    paged_kernel_lens: torch.Tensor, min_tokens: int
) -> torch.Tensor:
    """The tail boundary, in GLOBAL POSITIONS: ``min(seq_len, min_tokens)``.

    Rank-uniform BY CONSTRUCTION and with no collective: ``seq_lens`` is
    rank-uniform (the property the weightless block-decode path already relies
    on), so every rank computes the identical position boundary from it and
    only the number of ROWS that boundary covers differs.  That is basis 2.6,
    and it is why no PP0 verdict and no reduce is minted here.

    Slice 1 is non-elastic (``min == max``), so the boundary is a pure function
    of the length vector.  Storing it per request would be a second bookkeeping
    of a derivable number; the elastic form of slice 2 is where a per-request
    field earns its place.
    """
    if min_tokens <= 0:
        return torch.zeros_like(paged_kernel_lens, dtype=torch.int64)
    return torch.clamp(paged_kernel_lens.to(torch.int64), max=int(min_tokens))


# ---------------------------------------------------------------------------
# The ring.
# ---------------------------------------------------------------------------


@dataclass
class KvTailCounters:
    """Every counter names its population; nothing here is a rate."""

    claimed_total: int = 0
    #: Rows returned by the ONE cast primitive (age-out / free / pressure).
    #: This is the population the word "materialised" names, and from slice 3
    #: it is the count of completed bf16 -> fp8 casts.
    materialised_total: int = 0
    #: Rows the CUTOVER wiped. A wipe is data DISCARDED, not data cast, so it
    #: is counted apart: from slice 3 (virtual fp8) folding it into
    #: materialised_total would report a loss as a completed cast.
    wiped_rows: int = 0
    demoted_total: int = 0
    pressure_demotions: int = 0
    #: One population only: rows the allocator could not hand out at claim
    #: time (basis 7.2's write-side fallback). There is no second producer.
    clamped_alloc: int = 0
    resets: int = 0
    #: RESIDENCY, added after boot weg2kvtail5 (#1243). That boot showed
    #: `rows_held=0` on all 168 lines with `materialised == demoted`, and the
    #: line could not say WHICH of two very different worlds it was in:
    #: "rows were held and then demoted" or "rows were never held at all".
    #: Both print the same. These terms separate them, and every demotion now
    #: names the path that caused it -- there are exactly THREE producers and
    #: each has its own total, so a sum can never hide which one moved.
    demoted_by_age: int = 0
    demoted_by_free: int = 0
    demoted_by_pressure: int = 0
    #: Sampled at the TOP of `plan`, before the age-out that plan may perform.
    #: `rows_held` alone is sampled after, which is why it read 0 forever.
    rows_held_pre: int = 0
    demoted_this_pass: int = 0
    last_trigger: str = "none"
    #: THE DENOMINATOR for claimed_total. The ring is armed only on a DECODE
    #: step (`begin_decode_step`, the F8 fix that keeps it off the extend
    #: path), so on a prefill-only workload it CANNOT fill -- and that is not
    #: a policy defect, it is the absence of the population. weg2kvtail5 ran
    #: 463 prefill batches and 1 decode batch; without this term the log
    #: cannot distinguish "the demoter is too eager" from "the ring was never
    #: armed", and the first reading costs a boot.
    decode_steps: int = 0
    attended_rows: int = 0
    body_rows: int = 0
    untrimmed_owned: int = 0
    reqs: int = 0

    @property
    def released_total(self) -> int:
        """Every row that left the ring, by either path. The DUE gate's
        clause (b) asks exactly this question and must not be able to miss a
        wipe."""
        return self.materialised_total + self.wiped_rows


class KvTailRing:
    """The bf16 ring: a second ``MHATokenToKVPool`` + its allocator + the dense
    body-slot -> ring-row mapping.

    NO NEW POOL CLASS AND NO HAND-WRITTEN FREE LIST.  The ring is the existing
    pool class at ``dtype=torch.bfloat16`` and the existing token allocator over
    it; composition, the shape ``HybridLinearKVPool`` and ``MiniMaxSparseKVPool``
    already use in the same module.  Geometry is READ off the body pool rather
    than recomputed, so the two cannot disagree about heads, head_dim or the
    layer set.
    """

    def __init__(
        self,
        body_pool,
        knobs: KvTailKnobs,
        ring_rows: int,
        device: Optional[str] = None,
        enable_memory_saver: bool = False,
        owner_bounds: Optional[Tuple[int, int, int, int]] = None,
        layer_id_transfer=None,
        _pool_factory=None,
        _allocator_factory=None,
    ):
        self.knobs = knobs
        self.counters = KvTailCounters()
        self._refuse_unsupported_form(body_pool, knobs)
        # THE THIRD INDEX SPACE (boot weg2kvtail4's killer).  The two named
        # below are TOKEN spaces; this one is the LAYER space, and slice 1
        # crossed it without translating.
        #
        #   GLOBAL           `layer.layer_id`, what the attention backend
        #                    carries.
        #   DENSE FULL-ATTN  what `HybridLinearKVPool.full_kv_pool` -- our
        #                    `body_pool` -- is addressed in.  The wrapper
        #                    converts with `_transfer_full_attention_id` and
        #                    hands the sub-pool the result as
        #                    `layer_id_override`; the sub-pool is explicitly
        #                    `mark_as_sub_pool`ed so its `local_slot`
        #                    degenerates to the subtraction inside that dense
        #                    frame.
        #
        # The ring's pool is a SECOND ALLOCATION OF THE BODY POOL'S FRAME, so
        # it must be addressed exactly as the body pool is.  `layer_id_transfer`
        # is the wrapper's OWN translation, passed in by `install_kv_tail_ring`
        # -- never a table rebuilt here.  A second layer table beside the
        # wrapper's is the same defect class as a second owner rule beside
        # `dcp_weighted_write_slots`, and this file exists to not have one.
        self._layer_id_transfer = layer_id_transfer
        # TWO INDEX SPACES, NAMED (the F1 defect of the first cut).  The
        # mapping is keyed by the COMPACTED PHYSICAL SLOT
        # (``dcp_weighted_write_slots``), because that is what the write side
        # produces and what ``build_dcp_weighted_kv_indices`` emits.  The BODY
        # ALLOCATOR under weighted DCP does not live in that space: its size is
        # ``max_total_num_tokens`` = the GLOBAL logical context C
        # (``model_runner_kv_cache_mixin.py``: "the allocator index space is C
        # itself"), while the pool holds only ``dcp_compact_pool_rows(C, cp_S,
        # cp_ratio)`` rows.  So every index arriving from the allocator's free
        # path must be translated through the SAME owner primitive the write
        # used -- never indexed raw, which for a low slot silently frees a
        # DIFFERENT live token's ring row and for a high slot runs off the end
        # of the mapping.
        #
        # ``owner_bounds`` None means the two spaces coincide (no DCP): the
        # allocator is sized over the same rows the pool holds.
        self.owner_bounds = owner_bounds

        if ring_rows <= 0:
            raise Weg2KvTailUnfundable(
                "W54 Weg2KvTailUnfundable: the tail ring was asked for with "
                f"{ring_rows} rows. A tail that is enabled and has no rows is "
                "a banner without a counter -- exactly the shape boot "
                "weg2kvtail1 shipped."
            )
        self.ring_rows = int(ring_rows)
        self.device = device if device is not None else body_pool.device
        self.body_rows = int(body_pool.size) + int(body_pool.page_size)
        self.page_size = int(body_pool.page_size)

        if _pool_factory is None:
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

            _pool_factory = MHATokenToKVPool
        if _allocator_factory is None:
            from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

            _allocator_factory = TokenToKVPoolAllocator

        # bf16 ring of the SAME geometry as the body pool.  Every argument is
        # read off the body pool; nothing about heads or layers is restated.
        self.pool = _pool_factory(
            size=self.ring_rows,
            page_size=self.page_size,
            dtype=torch.bfloat16,
            head_num=int(body_pool.head_num),
            head_dim=int(body_pool.head_dim),
            layer_num=int(body_pool.layer_num),
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=int(getattr(body_pool, "v_head_dim", body_pool.head_dim)),
            start_layer=int(getattr(body_pool, "start_layer", 0) or 0),
            end_layer=int(
                getattr(body_pool, "end_layer", body_pool.layer_num - 1)
                if getattr(body_pool, "end_layer", None) is not None
                else body_pool.layer_num - 1
            ),
            enable_alt_stream=False,
        )
        # The ring's pool is a second allocation of the BODY POOL'S frame, so
        # its slot map is the body pool's, inherited -- not one re-derived from
        # the process-global layer-set parser that `KVCache.__init__` consults.
        # Under a PP layer set that parser would attach a GLOBAL-keyed map to a
        # pool addressed with DENSE ids, and every lookup would miss (the exact
        # reason `mark_as_sub_pool` exists for `full_kv_pool`). Inheriting makes
        # `ring.pool.local_slot` agree with `body_pool.local_slot` by
        # construction rather than by coincidence.
        self.pool._local_slot_of = getattr(body_pool, "_local_slot_of", None)
        self.allocator = _allocator_factory(
            size=self.ring_rows,
            dtype=torch.bfloat16,
            device=self.device,
            kvcache=self.pool,
            need_sort=False,
        )
        # R3: one dense int32 row, the SWA translation-table shape, null = -1.
        self.mapping = torch.full(
            (self.body_rows + 1,),
            KV_TAIL_NULL,
            dtype=torch.int32,
            device=self.device,
        )
        self.map_bytes = self.mapping.numel() * self.mapping.element_size()
        self.rows_held = 0
        # Armed only for the duration of one DECODE step; see begin_decode_step.
        self._armed = False
        self._claim_cache = None

    # -- form gate ---------------------------------------------------------

    @staticmethod
    def _refuse_unsupported_form(body_pool, knobs: KvTailKnobs) -> None:
        """Refuse by name where the tail cannot be honoured on this form.

        Each clause is a broken ASSUMPTION, not a missing feature: the mapping
        is indexed by a row of the body pool (so a paged or HND body pool maps
        the wrong thing), the ring is a second allocation of the same shape (so
        an MLA pool has no such shape), and a VA-backed pool's rows can be
        unmapped underneath a ring that knows nothing about backing.
        """
        page_size = int(getattr(body_pool, "page_size", 1))
        if page_size != 1:
            raise Weg2KvTailFormRefused(
                "W58 Weg2KvTailFormRefused: the precision tail needs "
                f"--page-size 1, got {page_size}. The body-slot -> ring-row "
                "mapping is indexed per TOKEN by the compacted physical slot "
                "the weighted DCP owner rule produces; at page-size > 1 that "
                "index addresses a page and the tail boundary would have to "
                "be re-derived, not scaled."
            )
        if getattr(body_pool, "use_mla", False):
            raise Weg2KvTailFormRefused(
                "W58 Weg2KvTailFormRefused: the precision tail is an MHA-pool "
                "feature; this pool is MLA. The ring is a second pool of the "
                "SAME (head_num, head_dim) geometry as the body pool and an "
                "MLA pool does not carry that shape."
            )
        layout = getattr(body_pool, "kv_cache_layout", "nhd")
        if getattr(body_pool, "use_hnd", False) or layout != "nhd":
            raise Weg2KvTailFormRefused(
                "W58 Weg2KvTailFormRefused: the precision tail needs the NHD "
                f"row-major KV layout, got {layout!r} (use_hnd="
                f"{bool(getattr(body_pool, 'use_hnd', False))}). HND and "
                "vectorized_5d fold (page, head) into the first index, so a "
                "row of the mapping no longer names one token."
            )
        if getattr(body_pool, "post_capture_active", False) or getattr(
            body_pool, "swappable_backing", False
        ):
            raise Weg2KvTailFormRefused(
                "W58 Weg2KvTailFormRefused: the precision tail refuses a "
                "VA-backed / post-capture body pool. Those pools may have "
                "layer backing unmapped underneath them (memory_pool.py "
                "_released_layers); the ring holds no backing state and would "
                "read a row that is not mapped."
            )

    # -- lifecycle ---------------------------------------------------------

    def attach_to_allocator(self, body_allocator) -> None:
        """Bind the ring's lifetime to the BODY SLOT's.

        THE LIFECYCLE TABLE, made structural instead of remembered. A ring row
        exists only while the body slot it shadows is allocated, so the one
        correct deleter is the body allocator's own free path -- which retract,
        abort, request finish and the cutover full reset all already go
        through. Registering here means no second site has to remember, and it
        is why R7 ("retract does not know the ring") cannot be true on this
        tree: retract frees body slots, and freeing a body slot frees its ring
        row.

        The CUTOVER is a real deleter of this mapping (Cutover-Full-Reset law)
        and in slice 1 there is no re-establisher -- the sidecar read that
        becomes one arrives in slice 4 and must then be proven to run AFTER
        this reset. Until then the tail simply rebuilds from decode, and the
        reset is COUNTED.
        """
        body_allocator.register_free_listener(self._on_body_free, on_clear=self.reset)
        self._body_allocator = body_allocator

    def to_compact_slots(self, allocator_index: torch.Tensor) -> torch.Tensor:
        """Allocator index space -> the mapping's compacted-slot space.

        The ONE translation, through the owner primitive the write side used,
        so read, write and free cannot drift into three different opinions of
        which row a token lives in.  Unowned slots drop out here: a free of a
        token another rank owns must touch nothing on this rank.

        The result is in range BY CONSTRUCTION rather than by a clamp:
        ``dcp_compact_pool_rows`` ceils to a whole owner block precisely so
        that the top slot ``C`` compacts inside the pool, and the mapping is
        sized ``body_pool.size + page_size``.
        """
        if self.owner_bounds is None:
            return allocator_index.to(torch.int64)
        from sglang.srt.layers.dcp.owner import dcp_weighted_write_slots

        cp_S, cp_lo, cp_hi, cp_ratio = self.owner_bounds
        loc, mask = dcp_weighted_write_slots(
            allocator_index.to(torch.int64), cp_S, cp_lo, cp_hi, cp_ratio
        )
        return loc[mask]

    def _on_body_free(self, free_index) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        slots = self.to_compact_slots(free_index)
        if slots.numel() == 0:
            return
        self.materialise_body_rows(slots, trigger="free")

    def available_size(self) -> int:
        """Free RING rows -- the admission bound of basis 7.10.

        This is the one place a rank-LOCAL number is correct, because it bounds
        a rank-local resource.  Rank uniformity is carried separately, by the
        position-derived boundary; it is NOT ``uniform_min_avail()``, which
        bounds the body pool the 16-bit rows do not come from.
        """
        return int(self.allocator.available_size())

    def begin_decode_step(self) -> None:
        """ARM the ring for exactly one decode step, and drop the per-step
        claim cache.

        THE ARM IS WHAT KEEPS THE RING OFF THE EXTEND PATH (the F8 defect of
        the first cut).  ``_dcp_write_scatter`` is the extend write as well as
        the decode write, and the age-out that bounds the window runs only at
        DECODE plan time.  An unarmed ring therefore filled front-to-back with
        the OLDEST prompt tokens of a long prefill and clamped away exactly the
        newest ones the tail exists for -- the inverse of the design, and
        basis 2.5's "extend untouched" was false while it could happen.

        It is also what removes the per-LAYER host syncs (F4): the allocation
        and its one ``int(...)`` happen on the first write of an ARMED step and
        are reused by the remaining layers, so a step costs ONE sync on the
        decode plan path #616c de-synced, not one per attention layer.
        """
        self._armed = True
        self._claim_cache = None
        self.counters.decode_steps += 1

    def disarm(self) -> None:
        """Every non-decode step: extend, draft, spec, idle.  An unarmed ring
        claims nothing and touches no device memory at the write site."""
        self._armed = False
        self._claim_cache = None

    def claim(self, loc: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Claim ring rows for the owned body slots in ``loc``.

        ``loc`` and ``mask`` are the SAME tensors the owner rule just produced
        for the body write -- no second ownership arithmetic, ever.  Slots that
        already hold a ring row keep it (a re-write of the same slot must not
        leak a row).  Returns ``(ring_loc, ring_mask)`` shaped like ``loc``, or
        ``(None, None)`` when the ring is not armed for this step.

        Called once per attention LAYER with the same ``loc``; the result is
        memoised for the step, keyed on the tensor's identity so a caller that
        ever passes a different write does not silently reuse the wrong rows.
        """
        if not getattr(self, "_armed", False):
            return None, None
        key = (loc.data_ptr(), int(loc.numel()))
        cached = getattr(self, "_claim_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        if mask is None:
            mask = torch.ones_like(loc, dtype=torch.bool)
        owned = loc[mask].to(torch.int64)
        if owned.numel() == 0:
            self._claim_cache = (key, None, None)
            return None, None
        existing = self.mapping[owned].to(torch.int64)
        fresh = existing < 0
        # The ONE host sync of the step (see begin_decode_step): the allocator
        # takes a count, not a mask.
        n_fresh = int(fresh.sum())
        if n_fresh:
            rows = self.allocator.alloc(n_fresh)
            if rows is None:
                # No room: the tokens simply stay body-only.  A short ring is a
                # smaller tail, never a wrong answer -- basis 7.2's fallback,
                # here at the write side.  NOTE (basis 2.6 / F14): WHICH tokens
                # lose their 16-bit row is a rank-LOCAL outcome, so a request's
                # realised precision can differ per rank even though the tail
                # BOUNDARY is rank-uniform.  That is benign for the LSE math --
                # each token is still attended exactly once, by exactly one
                # rank -- but it is narrower than "the tail is rank-uniform"
                # and is counted rather than assumed.
                self.counters.clamped_alloc += n_fresh
            else:
                self.mapping[owned[fresh]] = rows.to(torch.int32)
                self.rows_held += n_fresh
                self.counters.claimed_total += n_fresh
        ring_loc = torch.zeros_like(loc, dtype=torch.int64)
        full = torch.full_like(loc, KV_TAIL_NULL, dtype=torch.int64)
        full[mask] = self.mapping[owned].to(torch.int64)
        ring_mask = full >= 0
        ring_loc[ring_mask] = full[ring_mask]
        # NO ``.any()`` HERE.  An all-false mask is a legal no-op for the
        # masked write kernel, and asking the device whether the mask has any
        # true entry is a second blocking D2H per layer on the decode path.
        ring_loc = ring_loc.to(loc.dtype)
        self._claim_cache = (key, ring_loc, ring_mask)
        return ring_loc, ring_mask

    def local_layer_id(self, layer_id: int) -> int:
        """GLOBAL layer id -> the frame the ring's pool is addressed in.

        THE ONE LAYER-ID AUTHORITY for every ring <-> pool crossing, read and
        write alike. It delegates to the wrapper's own
        ``_transfer_full_attention_id``; it does not reimplement it and holds
        no table of its own.

        A layer the tail does not hold (a linear/GDN layer, which has no
        full-attention slot at all) is REFUSED by that same rule rather than
        translated -- for the reason ``KVCache.local_slot`` states about the
        subtraction it replaced: the failure mode is to return a plausible
        index into ANOTHER layer's KV.

        Identity when there is no wrapper: then the global id already IS the
        pool's frame, which is the pre-#1243 behaviour byte for byte.
        """
        if self._layer_id_transfer is None:
            return layer_id
        return self._layer_id_transfer(layer_id)

    def write(self, layer, ring_loc, ring_mask, cache_k, cache_v) -> None:
        """The SECOND ``set_kv_buffer``, against the ring, at the SAME masked
        write the body took.

        The ring's dtype is bf16 and ``cache_k`` arrives bf16, so the cast at
        ``MHATokenToKVPool.set_kv_buffer`` is a structural no-op on this call
        and the masked-write kernel is reused unchanged.

        ``layer_id_override`` is the SAME mechanism and the SAME value
        ``HybridLinearKVPool.set_kv_buffer`` uses to reach this pool's twin --
        boot weg2kvtail4 died here because the raw GLOBAL id was passed to a
        pool sized for the DENSE full-attention frame.
        """
        self.pool.set_kv_buffer(
            layer,
            ring_loc,
            cache_k,
            cache_v,
            None,
            None,
            layer_id_override=self.local_layer_id(layer.layer_id),
            dcp_kv_mask=ring_mask,
        )

    def materialise_body_rows(
        self, slots: torch.Tensor, pressure: bool = False, trigger: str = "free"
    ) -> int:
        """THE ONE PRIMITIVE for all four cast sites (basis 7.1).

        In slice 1 it degenerates to "free the ring rows and clear the mapping",
        because the double write already put the fp8 row in place.  From slice 3
        (virtual fp8) the same call gathers the ring rows, casts them through
        the body pool's own ``set_kv_buffer`` into the SAME slots, and only then
        frees -- which is why the body row stays ALLOCATED while tail-resident
        (spec decision D3): a cast can then never fail for lack of a
        destination.
        """
        if slots is None or slots.numel() == 0:
            return 0
        slots64 = slots.to(torch.int64)
        rows = self.mapping[slots64].to(torch.int64)
        live = rows >= 0
        n = int(live.sum())
        if n == 0:
            return 0
        self.allocator.free(rows[live])
        self.mapping[slots64[live]] = KV_TAIL_NULL
        self.rows_held -= n
        self.counters.materialised_total += n
        self.counters.demoted_total += n
        # ATTRIBUTED, not summed. `pressure` stays the flag the cast primitive
        # already took; `trigger` names the CALLER, and the two are reconciled
        # here so a caller cannot claim one and be counted as the other.
        if pressure:
            self.counters.pressure_demotions += n
            self.counters.demoted_by_pressure += n
            self.counters.last_trigger = "pressure"
        elif trigger == "age":
            self.counters.demoted_by_age += n
            self.counters.last_trigger = "age"
        else:
            self.counters.demoted_by_free += n
            self.counters.last_trigger = "free"
        self.counters.demoted_this_pass += n
        return n

    def reset(self) -> None:
        """CUTOVER: zero the mapping and the allocator.

        The cutover is a REAL deleter of this mapping (Cutover-Full-Reset law),
        and in slice 1 there is no re-establisher -- the sidecar read that will
        become one arrives in slice 4 and must then be proven to run AFTER this
        reset.  So here the tail simply rebuilds from decode, and the reset is
        COUNTED, because a silent wipe between the writer and the reader is
        precisely what made boot weg2kvtail1 a no-op.
        """
        self.mapping.fill_(KV_TAIL_NULL)
        self.allocator.clear()
        # A WIPE, NOT A CAST.  Counted in its own field: from slice 3 the 8-bit
        # row of a tail-resident token does not exist yet, so reporting a
        # cutover wipe as a materialisation would report DATA LOSS as a
        # completed conversion.
        self.counters.wiped_rows += self.rows_held
        self.counters.resets += 1
        self.rows_held = 0
        self._armed = False
        self._claim_cache = None

    # -- planning ----------------------------------------------------------

    def plan(self, kv_indptr, kv_indices, owned_tail_len, site: str = "decode"):
        """Split a plan, age out what left the window, and account for it.

        Returns ``(body_indptr, body_indices, tail_indptr, tail_ring_indices)``.
        The age-out runs BEFORE the split is accounted, so the counters describe
        the plan that is actually issued.
        """
        # RESIDENCY BEFORE THE DEMOTION STEP. `rows_held` is read again at
        # emission time, i.e. AFTER any age-out below; reporting only that is
        # what made weg2kvtail5's line unable to say whether a row was ever
        # resident. Reset per pass so the term is a PASS quantity, not a total.
        self.counters.rows_held_pre = self.rows_held
        self.counters.demoted_this_pass = 0
        pre = split_owned_indices(kv_indptr, kv_indices, owned_tail_len, self.mapping)
        age_out = pre[4]
        if age_out.numel():
            self.materialise_body_rows(age_out, trigger="age")
            body_indptr, body_indices, tail_indptr, tail_ring, _ = split_owned_indices(
                kv_indptr, kv_indices, owned_tail_len, self.mapping
            )
        else:
            body_indptr, body_indices, tail_indptr, tail_ring, _ = pre
        attended = int(tail_ring.numel())
        body = int(body_indices.numel())
        untrimmed = int(kv_indices.numel())
        c = self.counters
        c.attended_rows = attended
        c.body_rows = body
        c.untrimmed_owned = untrimmed
        c.reqs = max(int(kv_indptr.numel()) - 1, 0)
        if attended + body != untrimmed:
            raise Weg2KvTailNoOp(
                "W56 Weg2KvTailNoOp: the tail/body split is not a partition -- "
                f"attended_rows={attended} + body_rows={body} != "
                f"untrimmed_owned={untrimmed} at site={site}. Under the "
                "slice-1 double write a token attended twice is a wrong answer "
                "and a token attended zero times is a hole; the identity is "
                "the only falsifier either has."
            )
        self._due_gate(attended, untrimmed, site)
        return body_indptr, body_indices, tail_indptr, tail_ring

    def _due_gate(self, attended: int, untrimmed: int, site: str) -> None:
        """W56: the rule was DUE and nothing happened.

        Two INDEPENDENT contradictions, because one of them alone is what boot
        weg2kvtail1 slipped through:
          (a) rows are physically held and the plan attends none of them;
          (b) rows were claimed at some point, none are held, and none was ever
              released -- a wipe that took the rows without going through the
              one release path.
        """
        if untrimmed > 0 and self.rows_held > 0 and attended == 0:
            raise Weg2KvTailNoOp(
                "W56 Weg2KvTailNoOp: the tail ring holds "
                f"{self.rows_held} rows and the {site} plan attended 0 of "
                f"them over {untrimmed} owned slots. A held row that is never "
                "read is the shape boot weg2kvtail1 shipped: banner printed, "
                "nothing rounded, four arms bit-identical."
            )
        c = self.counters
        if c.claimed_total > 0 and self.rows_held == 0 and c.released_total == 0:
            raise Weg2KvTailNoOp(
                "W56 Weg2KvTailNoOp: "
                f"{c.claimed_total} ring rows were claimed, 0 are held and 0 "
                "were ever released. Rows left the ring without passing the "
                "one release path -- a deleter between the writer and its "
                "only reader (the per-LAYER reset that rooted weg2kvtail1)."
            )

    # -- the counter line (L2) ---------------------------------------------

    def counter_line(self, site: str = "decode") -> str:
        """One line per decode plan.  NOT rate-limited, and therefore carrying
        no ``suppressed=`` field: a field that can only ever print 0 cannot do
        the job the denominator law added it for, and printing one is a claim
        that a limiter was checked.  If a limiter is ever added here, it prints
        its own suppressed count and this docstring is the contract."""
        c = self.counters
        mx = "open" if self.knobs.max_tokens == KV_TAIL_OPEN else self.knobs.max_tokens
        return (
            f"KV-TAIL min={self.knobs.min_tokens} max={mx} "
            f"in_tail_tokens={self.rows_held} "
            f"demoted_total={c.demoted_total} "
            f"pressure_demotions={c.pressure_demotions} "
            f"headroom_16bit_tokens={self.available_size()} "
            f"site={site} "
            f"rows_held={self.rows_held}/{self.ring_rows} reqs={c.reqs} "
            f"attended_rows={c.attended_rows} body_rows={c.body_rows} "
            f"untrimmed_owned={c.untrimmed_owned} "
            f"claimed_total={c.claimed_total} "
            f"materialised_rows={c.materialised_total} "
            f"wiped_rows={c.wiped_rows} "
            f"ring_free={self.available_size()} "
            f"clamped_alloc={c.clamped_alloc} "
            f"resets={c.resets} "
            f"rows_held_pre={c.rows_held_pre} "
            f"demoted_this_pass={c.demoted_this_pass} "
            f"trigger={c.last_trigger} "
            f"demoted_by_age={c.demoted_by_age} "
            f"demoted_by_free={c.demoted_by_free} "
            f"demoted_by_pressure={c.demoted_by_pressure} "
            f"decode_steps={c.decode_steps} "
            f"instrument=plan-counts"
        )

    def log_counters(self, site: str = "decode") -> None:
        logger.info("%s", self.counter_line(site))


def install_kv_tail_ring(
    token_to_kv_pool,
    knobs: KvTailKnobs,
    max_running_requests: int,
    owned_share_num: int,
    owned_share_den: int,
    enable_memory_saver: bool = False,
    owner_bounds: Optional[Tuple[int, int, int, int]] = None,
    allocator_index_space: str = "compact",
) -> Optional[KvTailRing]:
    """Attach a ring to the FULL-ATTENTION pool, or return ``None``.

    The ring is instantiated only for the pool reached through
    ``HybridLinearKVPool.full_kv_pool`` (basis 2.10): the linear / GDN side
    never sees it, structurally rather than by a flag.

    ``allocator_index_space`` names which space the BODY ALLOCATOR frees in --
    ``"compact"`` (no DCP: allocator rows are pool rows) or ``"global"``
    (weighted DCP: the allocator is sized over the global context C and every
    freed index must be translated by ``owner_bounds``).  Anything else is
    refused: an untranslatable free path frees a DIFFERENT live token's ring
    row, which is a wrong 16-bit read rather than a crash.

    Returns ``None`` -- byte-identically the pre-slice path -- whenever the tail
    is off.  A tail that is ON and cannot be honoured RAISES; it never installs
    itself as a quiet no-op.
    """
    if not knobs.enabled:
        return None
    knobs.validate(page_size=int(getattr(token_to_kv_pool, "page_size", 1)))
    if allocator_index_space not in ("compact", "global"):
        raise Weg2KvTailFormRefused(
            "W58 Weg2KvTailFormRefused: the precision tail refuses an "
            f"allocator index space of {allocator_index_space!r}. Slice 1 can "
            "translate a freed allocator index into the mapping's compacted "
            "slot only where the weighted owner rule defines that translation "
            "(uneven DCP) or where the two spaces coincide (no DCP). Under "
            "EVEN modulo DCP the allocator index space is inflated by the "
            "split factor and the write loc is not the weighted compact slot, "
            "so a free would index the mapping with a number that names a "
            "different token. Run the tail on the uneven-DCP form."
        )
    if allocator_index_space == "global" and owner_bounds is None:
        raise Weg2KvTailFormRefused(
            "W58 Weg2KvTailFormRefused: the precision tail was asked for a "
            "global allocator index space with no owner bounds to translate "
            "it. The bounds come from dcp_weighted_owner_bounds and are the "
            "SAME derivation the write side used; without them there is no "
            "honest mapping from a freed slot to a ring row."
        )
    body_pool = getattr(token_to_kv_pool, "full_kv_pool", token_to_kv_pool)
    # The unwrap above changes the LAYER FRAME as well as the object: a
    # `full_kv_pool` is addressed with the DENSE full-attention id its wrapper
    # produces, never with the global id the attention backend carries. Take
    # the wrapper's OWN translation with us, so the ring resolves layers
    # through the same authority the body write does. `None` where there is no
    # wrapper -- then the global id already is the pool's frame.
    layer_id_transfer = (
        getattr(token_to_kv_pool, "_transfer_full_attention_id", None)
        if body_pool is not token_to_kv_pool
        else None
    )
    rows = knobs.ring_rows
    if rows is None:
        rows = auto_ring_rows(
            max_running_requests, knobs.min_tokens, owned_share_num, owned_share_den
        )
    ring = KvTailRing(
        body_pool,
        knobs,
        ring_rows=rows,
        enable_memory_saver=enable_memory_saver,
        owner_bounds=owner_bounds if allocator_index_space == "global" else None,
        layer_id_transfer=layer_id_transfer,
    )
    body_pool.kv_tail = ring
    token_to_kv_pool.kv_tail = ring
    return ring
