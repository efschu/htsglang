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
refusal by name (W141), not a quiet no-op.

SLICE 2q (23.09.): THE ROWS READER.  Next Flash reads its full-attention KV
through QSA, whose kernel takes one explicit pool ROW per selected token
(``qsa/sparse_attn.py`` ``_sparse_attn_rows_fwd``) -- not a paged plan.  For
that reader the ring needs no plan split at all: the kernel looks every row up
in the mapping and reads the bf16 ring row where one exists, the fp8 body row
where none does.  Each selected token is therefore read exactly once, from
exactly one source, BY CONSTRUCTION -- the partition identity the paged
reader has to assert is structural here.  Three things follow:

* ``page_size > 1`` is legal for this reader (QSA requires it: the compressed
  index is ``full_slot // ratio``).  The mapping stays per TOKEN because every
  read and write of this reader addresses a token row; only the paged
  FlashInfer reader, whose plan indexes PAGES, keeps the page-size-1 gate.
* The window slides at EVERY written step -- extend chunks included (slice 2d
  for this reader): each token written at request index ``p`` pushes the
  token at ``p - N`` out of the window.  Age-out first, precommit second, both
  at plan time and out of graph, so a long prefill ends with exactly its
  newest ``N`` prompt tokens in bf16.
* The READ fact is counted by the kernel itself (tail lanes read, a device
  scalar), and the WIRING fact by a device counter per tail-enabled launch;
  both are folded at the next plan (#1429's rule: under CUDA graphs a Python
  counter in the forward is blind).
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
    "KV_TAIL_READERS",
    "KvTailKnobs",
    "KvTailRing",
    "release_request_ring_rows",
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

#: Who reads the ring (slice 2q).  ``"paged"`` is the FlashInfer reader: its
#: plan indexes pages, so the ring there needs ``--page-size 1``.  ``"rows"``
#: is the QSA reader: one pool row per selected token, any page size.
KV_TAIL_READERS = ("paged", "rows")

#: Rows-reader counter line: every plan is printed up to this many, then every
#: ``_ROWS_LOG_EVERY``-th, and the line says how many it left out.
_ROWS_LOG_FIRST = 64
_ROWS_LOG_EVERY = 128


# ---------------------------------------------------------------------------
# Refusals.  RENUMBERED 23.09. (the merge onto the Next Flash line): the tail
# took W54/W56/W58 on 09.09., enumerated free then; the line has since given
# W54/W56 to the corridor (Weg2CorridorBudgetUnpriced / FloorUnmeasured) and
# W58 to the store (Weg2StoreArcRefused), and test_weg2_wcode_uniqueness_1263
# names each collision. That census reads TWO-digit codes only, while W100-W119
# are live on this line (the NF flip chain numbers them in sequence, W119 on
# 23.09.), so "first free above the census maximum" would have collided again:
# the tail took W140-W142, a gap above the running sequence, checked free over
# python/, test/, scripts/ and the gpu-arb docs.
# ---------------------------------------------------------------------------


class Weg2KvTailUnfundable(RuntimeError):
    """W140 Weg2KvTailUnfundable -- the ring or the guaranteed minimum cannot be
    funded, or a knob combination cannot be honoured.

    Refused, never clamped: silently lowering the ring gives back precision the
    operator asked for, and silently raising it takes KV tokens the operator
    did not offer.  Both choices belong to the operator and both are named in
    the message (the ``server_args.py`` mamba-floor validators are the shape).
    """


class Weg2KvTailNoOp(RuntimeError):
    """W141 Weg2KvTailNoOp -- the tail rule was DUE on this form and attended
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
    """W142 Weg2KvTailFormRefused -- the tail cannot be honoured on this pool
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
    #: Slice 2e-II (basis 2.2/2.8/7.7): size the ring from the KV pool the
    #: admission bound can never use (pool minus max_running_requests x the
    #: per-request cap) -- 16 bit as long as KV VRAM is free. Rows reader,
    #: dcp 1; refused by name elsewhere (pool_configurator._kv_tail_ring_post).
    headroom: bool = False

    @property
    def enabled(self) -> bool:
        """The tail is OFF unless a positive minimum, an explicit ring or the
        headroom ring was asked for.  ``--kv-tail-min-tokens 0`` with no ring
        is the identity arm: nothing is constructed, nothing is planned, and
        the default path is byte-identical."""
        if self.ring_rows is not None and self.ring_rows > 0:
            return True
        if self.headroom:
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
                "W140 Weg2KvTailUnfundable: --kv-tail-min-tokens "
                f"{self.min_tokens} is negative. 0 is legal (basis 7.4, no "
                "guaranteed 16-bit portion); a negative minimum is not a "
                "smaller tail, it is an unreadable one."
            )
        # (1) sentinel BEFORE any comparison against max.
        if self.max_tokens != KV_TAIL_OPEN:
            if self.max_tokens < 0:
                raise Weg2KvTailUnfundable(
                    "W140 Weg2KvTailUnfundable: --kv-tail-max-tokens "
                    f"{self.max_tokens} is negative and is not the open "
                    f"sentinel ({KV_TAIL_OPEN}). 0 is a REAL value (a hard "
                    "zero tail), which is exactly why it may not double as "
                    "the sentinel."
                )
            # (2)
            if self.max_tokens < self.min_tokens:
                raise Weg2KvTailUnfundable(
                    "W140 Weg2KvTailUnfundable: --kv-tail-max-tokens "
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
                "W140 Weg2KvTailUnfundable: --kv-tail-ring-rows "
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
                "W140 Weg2KvTailUnfundable: --kv-tail-host-max-tokens "
                f"{host_max} is below --kv-tail-min-tokens "
                f"{self.min_tokens}. The guaranteed tail must always have a "
                "host-tier landing place at the flip (basis 7.5), so the host "
                "post may not be smaller than the device floor it carries."
            )
        # (6)
        if self.sidecar and page_size != 1:
            raise Weg2KvTailUnfundable(
                "W140 Weg2KvTailUnfundable: --kv-tail-sidecar needs "
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
            "W140 Weg2KvTailUnfundable: --kv-tail-ring-rows auto needs a "
            f"positive max_running_requests, got {max_running_requests}. The "
            "ring is sized from the group's own concurrency; a zero means the "
            "value was not read off the argv it lives on."
        )
    if owned_share_den <= 0 or owned_share_num <= 0:
        raise Weg2KvTailUnfundable(
            "W140 Weg2KvTailUnfundable: --kv-tail-ring-rows auto needs a "
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
    #: READ-PATH PROOF (#1243 fact 3). `attended_rows` is a PLAN fact -- it
    #: says a plan NAMED rows. Only this says a kernel actually read them.
    #: The gap between the two is the kvtail1 "probe did not bite" class:
    #: banner printed, counter moving, nothing rounded.
    tail_merges: int = 0
    tail_merges_this_step: int = 0
    attended_rows: int = 0
    body_rows: int = 0
    untrimmed_owned: int = 0
    reqs: int = 0
    #: Slice 2q (rows reader).  ``tail_rows_read`` is the READ fact the rows
    #: kernel counts on the device: selected lanes it served from the ring,
    #: summed over layers and query rows.  ``rows_launches`` is the WIRING
    #: fact: tail-enabled kernel launches.  Both are folded out of graph.
    tail_rows_read: int = 0
    tail_rows_read_step: int = 0
    rows_launches: int = 0
    #: Rows released because the request that wrote them FINISHED (basis 7.1:
    #: the radix tree holds fp8 only, the tail is private per request).
    demoted_by_finish: int = 0
    extend_steps: int = 0
    verify_steps: int = 0
    #: Plans the rows-reader limiter did not print (see _ROWS_LOG_EVERY).
    log_suppressed: int = 0

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
        reader: str = "paged",
        _pool_factory=None,
        _allocator_factory=None,
    ):
        self.knobs = knobs
        self.counters = KvTailCounters()
        if reader not in KV_TAIL_READERS:
            raise Weg2KvTailFormRefused(
                f"W142 Weg2KvTailFormRefused: unknown tail reader {reader!r}; "
                f"the ring knows {KV_TAIL_READERS}. A reader this file does "
                "not know is a read path nobody wired."
            )
        self.reader = reader
        self._refuse_unsupported_form(
            body_pool, knobs, under_memory_saver=bool(enable_memory_saver), reader=reader
        )
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
                "W140 Weg2KvTailUnfundable: the tail ring was asked for with "
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
        #: #1427: rows for this step handed out at PLAN time (precommit); the
        #: in-graph claim is then a pure gather.
        self._precommitted = False
        #: #1429: the READ fact as a DEVICE counter. ``note_merge`` bumps it
        #: with a captured kernel, so a CUDA-graph REPLAY -- which runs no
        #: Python -- still moves it; ``plan`` reads it out of graph. Boot kvt7d
        #: (16.09.): merge captured and replayed, Python counter frozen at 288,
        #: W56 (now W141) fired on a tail that WAS read.
        self._merge_dev = torch.zeros((), dtype=torch.int64, device=self.device)
        self._merge_dev_seen = 0
        #: attended_rows of the PREVIOUS plan, for the fact-4 gate below.
        self._last_plan_attended = 0
        #: Slice 2q: tail lanes the rows kernel served, a DEVICE scalar the
        #: kernel adds to (graph replays included); folded by plan_rows_step.
        self._read_dev = torch.zeros((), dtype=torch.int64, device=self.device)
        self._read_dev_seen = 0
        #: What the previous rows plan left behind, for its W141 wiring gate.
        self._rows_expect_read = False
        self._rows_held_after = 0
        self._rows_plans = 0
        #: Slice 2e (rows reader): each live request's window, keyed by its
        #: req_to_token row: ``[tail_start, l_last]`` -- the oldest request
        #: index still in 16 bit and the request's length at its last plan.
        #: Host-side ints: the policy runs out of graph and needs no sync.
        self._win = {}

    # -- form gate ---------------------------------------------------------

    @staticmethod
    def _refuse_unsupported_form(
        body_pool, knobs: KvTailKnobs, under_memory_saver: bool = False, reader: str = "paged"
    ) -> None:
        """Refuse by name where the tail cannot be honoured on this form.

        Each clause is a broken ASSUMPTION, not a missing feature: the mapping
        is indexed by a row of the body pool (so a paged or HND body pool maps
        the wrong thing), the ring is a second allocation of the same shape (so
        an MLA pool has no such shape), and a VA-backed pool's rows can be
        unmapped underneath a ring that knows nothing about backing.

        Slice 2q: the page-size clause belongs to the PAGED reader only. Its
        plan indexes pages, so a tail boundary inside a page has no reading.
        The rows reader addresses one token row per selected token, so the
        per-token mapping means the same thing at every page size.
        """
        page_size = int(getattr(body_pool, "page_size", 1))
        if page_size != 1 and reader == "paged":
            raise Weg2KvTailFormRefused(
                "W142 Weg2KvTailFormRefused: the precision tail needs "
                f"--page-size 1, got {page_size}. The body-slot -> ring-row "
                "mapping is indexed per TOKEN by the compacted physical slot "
                "the weighted DCP owner rule produces; at page-size > 1 that "
                "index addresses a page and the tail boundary would have to "
                "be re-derived, not scaled."
            )
        if getattr(body_pool, "use_mla", False):
            raise Weg2KvTailFormRefused(
                "W142 Weg2KvTailFormRefused: the precision tail is an MHA-pool "
                "feature; this pool is MLA. The ring is a second pool of the "
                "SAME (head_num, head_dim) geometry as the body pool and an "
                "MLA pool does not carry that shape."
            )
        layout = getattr(body_pool, "kv_cache_layout", "nhd")
        if getattr(body_pool, "use_hnd", False) or layout != "nhd":
            raise Weg2KvTailFormRefused(
                "W142 Weg2KvTailFormRefused: the precision tail needs the NHD "
                f"row-major KV layout, got {layout!r} (use_hnd="
                f"{bool(getattr(body_pool, 'use_hnd', False))}). HND and "
                "vectorized_5d fold (page, head) into the first index, so a "
                "row of the mapping no longer names one token."
            )
        # #1425 (serving form, 16.09.): the weg2 KV pool sits on a VA
        # reservation for the phase flip (`swappable_backing=True`). The ring
        # reads NO body row -- the mapping is keyed by body slot but holds ring
        # rows -- so the body's unmapped layers cannot reach it. What the ring
        # must follow is the flip itself: under the SAME memory-saver tag it
        # pauses and resumes with the group. Only a ring that cannot follow the
        # tag is refused on this form.
        if (getattr(body_pool, "post_capture_active", False) or getattr(
            body_pool, "swappable_backing", False
        )) and not under_memory_saver:
            raise Weg2KvTailFormRefused(
                "W142 Weg2KvTailFormRefused: the precision tail refuses a "
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
        slots = self.to_compact_slots(self._whole_pages(free_index))
        if slots.numel() == 0:
            return
        self.materialise_body_rows(slots, trigger="free")

    def _whole_pages(self, free_index: torch.Tensor) -> torch.Tensor:
        """Slice 2q: a paged allocator frees PAGES. ``PagedTokenToKVPoolAllocator
        .free`` returns ``unique(free_index // page_size)`` to the free list,
        so every token of such a page is free afterwards whether it was listed
        or not. The listener therefore releases the whole page: a ring row left
        mapped under a token the next owner of the page writes would be read as
        that new token's 16-bit K/V. Identity at page size 1."""
        if self.page_size <= 1:
            return free_index
        p = int(self.page_size)
        pages = torch.unique(free_index.to(torch.int64) // p)
        offs = torch.arange(p, dtype=torch.int64, device=pages.device)
        return (pages[:, None] * p + offs[None, :]).reshape(-1)

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
        self.begin_step("decode")

    def begin_step(self, site: str = "decode") -> None:
        """#1426 slice 2: ARM the ring for one step of `site` ("decode" or
        "verify" -- the MTP target-verify extend writes the spec tokens' K/V
        through the same `_dcp_write_scatter` and must double-write them)."""
        self._armed = True
        self._claim_cache = None
        self._precommitted = False
        if site == "verify":
            self.counters.verify_steps += 1
        elif site == "extend":
            self.counters.extend_steps += 1
        else:
            self.counters.decode_steps += 1

    def disarm(self) -> None:
        """Every non-decode step: extend, draft, spec, idle.  An unarmed ring
        claims nothing and touches no device memory at the write site."""
        self._armed = False
        self._claim_cache = None
        self._precommitted = False

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
        if mask is None:
            mask = torch.ones_like(loc, dtype=torch.bool)
        capturing = bool(
            loc.is_cuda and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if capturing or getattr(self, "_precommitted", False):
            # #1427 CAPTURE-SAFE FORM. The rows were handed out by
            # ``precommit`` at PLAN time, out of graph; here only fixed-shape,
            # sync-free gathers run, so the captured graph reads the CURRENT
            # mapping at every replay. A slot nobody precommitted gathers the
            # null and is masked out -- never row 0. Boot kvt4d (16.09.) died
            # at ``loc[mask]`` here: "operation not permitted when stream is
            # capturing".
            rows = self.mapping[loc.to(torch.int64)].to(torch.int64)
            ring_mask = mask & (rows >= 0)
            ring_loc = torch.where(ring_mask, rows, torch.zeros_like(rows)).to(loc.dtype)
            return ring_loc, ring_mask
        key = (loc.data_ptr(), int(loc.numel()))
        cached = getattr(self, "_claim_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
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

    def precommit(self, loc: torch.Tensor, mask: Optional[torch.Tensor] = None) -> int:
        """#1427: hand out ring rows for the slots THIS STEP will write, at PLAN
        time and OUT OF GRAPH.

        ``loc``/``mask`` are the owner-rule tensors for the step's
        ``out_cache_loc`` -- the same expression the write site evaluates
        (``dcp_weighted_write_slots``), computed once here. This is the only
        place that allocates: the data-dependent ``loc[mask]``, the ``int(...)``
        of the fresh count and the allocator call all happen here, where a host
        sync is legal, so the in-graph ``claim`` can be a pure gather. Slots
        that already hold a row keep it. Returns the number of fresh rows.
        """
        if not getattr(self, "_armed", False):
            return 0
        if mask is None:
            mask = torch.ones_like(loc, dtype=torch.bool)
        owned = loc[mask].to(torch.int64)
        n_fresh = 0
        if owned.numel():
            existing = self.mapping[owned].to(torch.int64)
            fresh = existing < 0
            n_fresh = int(fresh.sum())
            if n_fresh:
                targets = owned[fresh]
                rows = self.allocator.alloc(n_fresh)
                if rows is None:
                    # PARTIAL, not all-or-nothing (slice 2q): the allocator
                    # refuses a count larger than its free list, and one
                    # extend chunk asking for more rows than are free must not
                    # leave the WHOLE chunk body-only. The free rows go to the
                    # NEWEST tokens -- the ones that stay in the window
                    # longest; ``loc`` arrives in request order, positions
                    # ascending. The rest stay body-only and are counted.
                    avail = int(self.allocator.available_size())
                    got = min(avail, n_fresh)
                    self.counters.clamped_alloc += n_fresh - got
                    rows = self.allocator.alloc(got) if got > 0 else None
                    if rows is None:
                        if got > 0:
                            self.counters.clamped_alloc += got
                        got = 0
                    else:
                        targets = targets[n_fresh - got :]
                    n_fresh = got
                if n_fresh:
                    self.mapping[targets] = rows.to(torch.int32)
                    self.rows_held += n_fresh
                    self.counters.claimed_total += n_fresh
        self._precommitted = True
        self._claim_cache = None
        return n_fresh

    def precommit_skip(self) -> None:
        """#1427: a step whose write must NOT allocate (the graph CAPTURE pass
        writes the runner's dummy slots). Switches the claim to the gather
        form without handing out a row."""
        self._precommitted = True
        self._claim_cache = None

    def note_merge(self) -> None:
        """The READ half reports itself. Called once per LAYER from
        ``_kv_tail_merge_decode``, after the tail wrapper has actually run.

        This is the only fact in the instrument that a PLAN cannot fake."""
        self.counters.tail_merges += 1
        self.counters.tail_merges_this_step += 1
        self._merge_dev += 1  # captured into the graph; replay moves it, Python does not run

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
        elif trigger == "finish":
            self.counters.demoted_by_finish += n
            self.counters.last_trigger = "finish"
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
        self._merge_dev.zero_()
        self._merge_dev_seen = 0
        self._read_dev.zero_()
        self._read_dev_seen = 0
        self._win = {}
        # The wiring gate judges launches SINCE the previous plan; a reset
        # zeroes the device counters, so the previous plan's expectation must
        # go with them (a flush between a step and its next plan is not a
        # ring nobody read).
        self._rows_expect_read = False
        self._rows_held_after = 0
        self.rows_held = 0
        self._armed = False
        self._claim_cache = None
        self._precommitted = False

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
        # FACT 4, the kvtail1 gate: the PREVIOUS step planned a tail and no
        # kernel ever read it. Checked HERE because a plan runs once per step
        # and the merges of the step before it are complete by now.
        # #1429: fold the device-side merge count in BEFORE judging. Eager
        # steps counted in Python already (delta == this_step); a graph replay
        # counted only on the device (this_step == 0, delta == layers merged).
        dev_total = int(self._merge_dev.item())
        dev_delta = dev_total - self._merge_dev_seen
        self._merge_dev_seen = dev_total
        if dev_delta > self.counters.tail_merges_this_step:
            self.counters.tail_merges += dev_delta - self.counters.tail_merges_this_step
            self.counters.tail_merges_this_step = dev_delta
        if self._last_plan_attended > 0 and self.counters.tail_merges_this_step == 0:
            raise Weg2KvTailNoOp(
                "W141 Weg2KvTailNoOp: the previous decode step planned "
                f"{self._last_plan_attended} attended tail rows and the tail "
                "merge ran 0 times, so no kernel read a single 16-bit row. "
                "attended_rows is a PLAN fact; tail_merges is the READ fact. "
                "A tail that is planned and never read is the shape boot "
                "weg2kvtail1 shipped: banner printed, counters moving, four "
                "arms bit-identical."
            )
        self.counters.tail_merges_this_step = 0
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
                "W141 Weg2KvTailNoOp: the tail/body split is not a partition -- "
                f"attended_rows={attended} + body_rows={body} != "
                f"untrimmed_owned={untrimmed} at site={site}. Under the "
                "slice-1 double write a token attended twice is a wrong answer "
                "and a token attended zero times is a hole; the identity is "
                "the only falsifier either has."
            )
        self._due_gate(attended, untrimmed, site)
        self._last_plan_attended = attended
        return body_indptr, body_indices, tail_indptr, tail_ring

    def _due_gate(self, attended: int, untrimmed: int, site: str) -> None:
        """W141: the rule was DUE and nothing happened.

        Two INDEPENDENT contradictions, because one of them alone is what boot
        weg2kvtail1 slipped through:
          (a) rows are physically held and the plan attends none of them;
          (b) rows were claimed at some point, none are held, and none was ever
              released -- a wipe that took the rows without going through the
              one release path.
        """
        if untrimmed > 0 and self.rows_held > 0 and attended == 0:
            raise Weg2KvTailNoOp(
                "W141 Weg2KvTailNoOp: the tail ring holds "
                f"{self.rows_held} rows and the {site} plan attended 0 of "
                f"them over {untrimmed} owned slots. A held row that is never "
                "read is the shape boot weg2kvtail1 shipped: banner printed, "
                "nothing rounded, four arms bit-identical."
            )
        c = self.counters
        if c.claimed_total > 0 and self.rows_held == 0 and c.released_total == 0:
            raise Weg2KvTailNoOp(
                "W141 Weg2KvTailNoOp: "
                f"{c.claimed_total} ring rows were claimed, 0 are held and 0 "
                "were ever released. Rows left the ring without passing the "
                "one release path -- a deleter between the writer and its "
                "only reader (the per-LAYER reset that rooted weg2kvtail1)."
            )

    # -- the rows reader (slice 2q: QSA) -----------------------------------

    def rows_operands(self, layer_id: int):
        """``(k, v, mapping, read_counter)`` for ONE layer, in the frame the rows
        kernel reads: the ring pool's buffers through the one layer-id
        authority (the frame ``write`` uses), the mapping keyed by the body
        pool row the kernel is handed, and the device scalar the kernel adds its
        tail lanes to."""
        lid = self.local_layer_id(layer_id)
        return (
            self.pool.get_key_buffer(lid),
            self.pool.get_value_buffer(lid),
            self.mapping,
            self._read_dev,
        )

    def note_rows_launch(self) -> None:
        """One tail-enabled rows-kernel launch. DEVICE-side only: the Python
        side learns it at the next ``plan_rows_step``, so an eager step and a
        graph replay are counted by the same instrument (#1429)."""
        self._merge_dev += 1

    def _compact_and_owned(self, allocator_index: torch.Tensor):
        """Allocator index space -> ``(compact_slot, owned)`` in the SAME shape,
        through the owner primitive the write used (see ``to_compact_slots``,
        which is the masked-out form of this). Shape-static, sync-free."""
        idx = allocator_index.to(torch.int64)
        if self.owner_bounds is None:
            return idx, torch.ones_like(idx, dtype=torch.bool)
        from sglang.srt.layers.dcp.owner import dcp_weighted_write_slots

        cp_S, cp_lo, cp_hi, cp_ratio = self.owner_bounds
        loc, mask = dcp_weighted_write_slots(idx, cp_S, cp_lo, cp_hi, cp_ratio)
        return loc.to(torch.int64), mask

    def window_tokens(self) -> int:
        """The GUARANTEED window in request-token positions (basis 7.6): the
        minimum. Above it the window is elastic (``plan_rows_step``)."""
        return int(self.knobs.min_tokens)

    def _max_window(self) -> Optional[int]:
        """The elastic ceiling (basis 7.7): ``None`` = open, the ring bounds it."""
        mx = self.knobs.max_tokens
        return None if mx == KV_TAIL_OPEN else int(mx)

    def _age_positions(self, req_to_token, idx: int, lo: int, hi: int, *, pressure: bool = False,
                       trigger: str = "age") -> int:
        """Release the ring rows of request row ``idx`` at request indices
        ``[lo, hi)`` (the window's back end). Returns the rows released."""
        if hi <= lo:
            return 0
        slots = req_to_token[int(idx), int(lo):int(hi)]
        compact, owned = self._compact_and_owned(slots.reshape(-1))
        return self.materialise_body_rows(compact[owned], pressure=pressure, trigger=trigger)

    def _relieve(self, req_to_token, deficit: int, batch_need: dict) -> int:
        """Basis 7.1/7.8: free ``deficit`` ring rows. (1) Water-levelling: the
        LARGEST holdings shrink first, from the back, down to a common level,
        never below the guaranteed minimum. (2) Still short: every request of
        THIS step slides its own window (ages as many of its own oldest rows
        as it claims new ones), so a request at the minimum keeps the minimum
        while it advances. Whatever is left is clamped by the precommit, the
        newest tokens first. Returns the rows freed."""
        n_min = self.window_tokens()
        freed = 0
        held = {idx: max(0, w[1] - w[0]) for idx, w in self._win.items()}
        order = sorted(held.items(), key=lambda kv: -kv[1])
        # (1) the water level: the highest level L >= n_min such that
        # cutting every holding above L down to L frees >= deficit.
        tops = [h for _, h in order if h > n_min]
        level = None
        if tops:
            cut = 0
            for k in range(len(tops)):
                nxt = tops[k + 1] if k + 1 < len(tops) else n_min
                nxt = max(nxt, n_min)
                width = k + 1
                if cut + width * (tops[k] - nxt) >= deficit:
                    level = tops[k] - -(-(deficit - cut) // width)  # ceil
                    level = max(level, nxt)
                    break
                cut += width * (tops[k] - nxt)
            if level is None:
                level = n_min
        if level is not None:
            for idx, h in order:
                if freed >= deficit or h <= level:
                    continue
                w = self._win[idx]
                k = min(h - level, deficit - freed)
                freed += self._age_positions(
                    req_to_token, idx, w[0], w[0] + k, pressure=True, trigger="pressure"
                )
                w[0] += k
        # (2) self-slide of this step's requests.
        if freed < deficit:
            for idx, need in batch_need.items():
                if freed >= deficit:
                    break
                w = self._win[idx]
                k = min(need, max(0, w[1] - w[0]), deficit - freed)
                freed += self._age_positions(req_to_token, idx, w[0], w[0] + k)
                w[0] += k
        return freed

    def release_request(self, idx: int, slots: torch.Tensor, trigger: str = "finish") -> int:
        """A request leaves: its rows go, and so does its window."""
        self._win.pop(int(idx), None)
        return self.release_slots(slots, trigger=trigger)

    def _fold_rows_facts(self, site: str) -> None:
        """Fold the previous step's device facts and judge its wiring.

        W141 (wiring): the previous step was one whose attention MUST run the
        rows kernel (decode, verify, an extend over a prefix), the ring held
        rows when it started, and not one tail-enabled launch happened. That
        is a ring nobody reads -- e.g. a route that bypassed the rows kernel.
        It is NOT raised when the kernel ran and simply selected no tail lane:
        a sparse selection may legitimately skip the newest tokens, and that
        is what ``tail_rows_read`` reports instead of hiding."""
        launches_total, read_total = (
            int(v) for v in torch.stack((self._merge_dev, self._read_dev)).tolist()
        )
        launches = launches_total - self._merge_dev_seen
        read = read_total - self._read_dev_seen
        self._merge_dev_seen = launches_total
        self._read_dev_seen = read_total
        c = self.counters
        c.rows_launches += launches
        c.tail_merges += launches
        c.tail_merges_this_step = launches
        c.tail_rows_read += read
        c.tail_rows_read_step = read
        if self._rows_expect_read and self._rows_held_after > 0 and launches == 0:
            raise Weg2KvTailNoOp(
                "W141 Weg2KvTailNoOp: the previous step's attention had to run "
                "the rows kernel (decode / verify / extend over a prefix) while "
                f"the ring held {self._rows_held_after} rows, and 0 tail-enabled "
                "launches happened. A route bypassed the reader -- the 16-bit "
                "rows were written and nobody could read them (the shape boot "
                "weg2kvtail1 shipped, and kvt6d on the DCP extend path)."
            )

    def plan_rows_step(
        self,
        site: str,
        loc: torch.Tensor,
        tok_req: torch.Tensor,
        tok_index: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        expect_read: bool,
    ) -> int:
        """ONE step of the sliding window for the rows reader. Out of graph.

        ``loc``        the allocator slots this step WRITES, one per token;
        ``tok_req``    each written token's batch index;
        ``tok_index``  its index in its request (the ``req_to_token`` column),
                       never a rope position -- Qwen4-Exp is an M-RoPE model,
                       and an image moves the rope position, not the column;
        ``expect_read`` whether this step's attention must run the rows kernel.

        The step's writes are contiguous per request (decode: one token;
        extend: the chunk; verify: the chain of draft tokens, topk 1).

        SLICE 2e, THE ELASTIC WINDOW (basis 2.2, 7.1, 7.7, 7.8). Each request
        owns a window ``[tail_start, L)`` in its own request indices. It grows
        with every written token while the ring has free rows, up to
        ``--kv-tail-max-tokens`` (open by default: the ring bounds it) -- the
        16-bit portion uses whatever headroom there is. When the step needs
        more rows than are free, holdings shrink FROM THE BACK: the largest
        first, water-levelled down to a common level, never below the
        guaranteed minimum; then every request of this step slides its own
        window (a request AT the minimum keeps it while it advances); only
        then does the precommit clamp, newest tokens first. ``max == min`` is
        the fixed sliding window. Returns the rows precommitted.
        """
        self._fold_rows_facts(site)
        self.begin_step(site)
        c = self.counters
        c.rows_held_pre = self.rows_held
        c.demoted_this_pass = 0
        n_max = self._max_window()
        total = int(loc.numel())
        bs = int(req_pool_indices.numel())
        c.reqs = bs
        fresh = 0
        if (n_max is None or n_max > 0) and total > 0 and bs > 0:
            dev = loc.device
            pos = tok_index.to(device=dev, dtype=torch.int64).reshape(-1)
            req = tok_req.to(device=dev, dtype=torch.int64).reshape(-1)
            big = torch.iinfo(torch.int64).max
            first = torch.full((bs,), big, dtype=torch.int64, device=dev)
            first.scatter_reduce_(0, req, pos, reduce="amin", include_self=True)
            l_new = torch.zeros(bs, dtype=torch.int64, device=dev)
            l_new.scatter_reduce_(0, req, pos + 1, reduce="amax", include_self=True)
            first_l, l_l = first.tolist(), l_new.tolist()
            rpi_l = [int(x) for x in req_pool_indices.tolist()]
            need = {}
            for i in range(bs):
                f, ln, idx = int(first_l[i]), int(l_l[i]), rpi_l[i]
                if f == big:
                    continue  # a padded request with no written token
                w = self._win.get(idx)
                if w is None or f < w[0]:
                    # A new request on this row (or one restarted behind its
                    # own window): the tail starts at its first written
                    # token -- a prefix it found in the tree stays fp8.
                    w = self._win[idx] = [f, f]
                if n_max is not None and ln - w[0] > n_max:
                    # The ceiling: age the back end past L - max. Tokens of
                    # THIS step below the new start are simply not claimed.
                    new_start = ln - n_max
                    self._age_positions(req_to_token, idx, w[0], min(new_start, f))
                    w[0] = new_start
                need[idx] = need.get(idx, 0) + max(0, ln - max(w[0], f))
            deficit = sum(need.values()) - self.available_size()
            if deficit > 0:
                self._relieve(req_to_token, deficit, need)
            start = torch.tensor(
                [self._win[idx][0] if idx in self._win else big for idx in rpi_l],
                dtype=torch.int64,
                device=dev,
            )
            in_window = pos >= start[req]
            loc_c, owned = self._compact_and_owned(loc.reshape(-1))
            # Slot 0 is the padding slot every padded token writes; it is
            # never a real token's row and never gets a ring row.
            real = loc.reshape(-1).to(torch.int64) > 0
            fresh = self.precommit(loc_c, owned & in_window & real)
            for i, idx in enumerate(rpi_l):
                if idx in self._win and int(first_l[i]) != big:
                    self._win[idx][1] = int(l_l[i])
        else:
            self.precommit_skip()
        self._rows_expect_read = bool(expect_read)
        self._rows_held_after = self.rows_held
        self._rows_plans += 1
        c.attended_rows = c.tail_rows_read_step
        self.log_rows_counters(site)
        return fresh

    def release_slots(self, slots: torch.Tensor, trigger: str = "finish") -> int:
        """Release the ring rows of these ALLOCATOR slots (any subset; unmapped
        and foreign slots are no-ops). Used when a request finishes: the tree
        takes its slots and holds fp8 only (basis 7.1, tail private per
        request), so the rows must not linger until the tree evicts."""
        if slots is None or slots.numel() == 0:
            return 0
        compact, owned = self._compact_and_owned(slots.reshape(-1))
        return self.materialise_body_rows(compact[owned], trigger=trigger)

    def log_rows_counters(self, site: str) -> None:
        """The rows reader's counter line, rate-limited WITH its own count of
        what it left out: the first ``_ROWS_LOG_FIRST`` plans, then every
        ``_ROWS_LOG_EVERY``-th."""
        k = self._rows_plans
        if k <= _ROWS_LOG_FIRST or k % _ROWS_LOG_EVERY == 0:
            logger.info("%s", self.counter_line(site))
        else:
            self.counters.log_suppressed += 1

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
            f"tail_merges={c.tail_merges} "
            + (
                # Slice 2q. attended_rows above is the PREVIOUS step's READ
                # fact on this reader (lanes the kernel served from the ring),
                # not a plan count -- the rows reader has no plan to count.
                f"reader=rows window={self.window_tokens()} "
                f"windows={len(getattr(self, '_win', {}))} "
                f"window_max_held={max((w[1] - w[0] for w in getattr(self, '_win', {}).values()), default=0)} "
                f"tail_rows_read={c.tail_rows_read} "
                f"tail_rows_read_step={c.tail_rows_read_step} "
                f"rows_launches={c.rows_launches} "
                f"extend_steps={c.extend_steps} verify_steps={c.verify_steps} "
                f"demoted_by_finish={c.demoted_by_finish} "
                f"suppressed={c.log_suppressed} "
                f"instrument=kernel-read-counts"
                if getattr(self, "reader", "paged") == "rows"
                else "instrument=plan-counts"
            )
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
    reader: str = "paged",
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
            "W142 Weg2KvTailFormRefused: the precision tail refuses an "
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
            "W142 Weg2KvTailFormRefused: the precision tail was asked for a "
            "global allocator index space with no owner bounds to translate "
            "it. The bounds come from dcp_weighted_owner_bounds and are the "
            "SAME derivation the write side used; without them there is no "
            "honest mapping from a freed slot to a ring row."
        )
    body_pool = getattr(token_to_kv_pool, "full_kv_pool", token_to_kv_pool)
    if int(getattr(body_pool, "head_num", 0) or 0) <= 0:
        # Form A (Next Flash D): the expert workers hold routed experts and
        # the router only -- no dense weights, no KV. The tail lives where the
        # KV lives (the host rank); here there is no body row to shadow, so
        # nothing is installed, and the log says so rather than staying quiet.
        logger.info(
            "KV-TAIL not installed on this rank: its full-attention pool holds "
            "no KV heads (head_num=%s; Form A expert worker). The tail is "
            "installed on the rank(s) that hold the KV.",
            getattr(body_pool, "head_num", None),
        )
        return None
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
        reader=reader,
    )
    body_pool.kv_tail = ring
    token_to_kv_pool.kv_tail = ring
    return ring


def release_request_ring_rows(req, tree_cache) -> int:
    """Slice 2q: a request is leaving (finish, abort, retract) -- release the
    ring rows of every slot it wrote, BEFORE the tree takes the slots.

    Basis 7.1: the radix tree holds fp8 only, the tail is private per request.
    Without this a finished request's rows stay mapped under the tree's slots
    until the tree evicts them, and a serving ring fills with the tails of
    requests that are gone (every later claim clamps: a tail that silently
    stops existing). Returns the rows released; 0 and no device work when no
    ring is installed, so the tail-off path is untouched."""
    alloc = getattr(tree_cache, "token_to_kv_pool_allocator", None)
    get_kv = getattr(alloc, "get_kvcache", None)
    pool = get_kv() if callable(get_kv) else None
    ring = getattr(pool, "kv_tail", None)
    if ring is None:
        return 0
    idx = getattr(req, "req_pool_idx", None)
    n = int(getattr(req, "kv_allocated_len", 0) or 0)
    r2t = getattr(getattr(tree_cache, "req_to_token_pool", None), "req_to_token", None)
    if idx is None or n <= 0 or r2t is None:
        if idx is not None:
            ring._win.pop(int(idx), None)
        return 0
    return ring.release_request(int(idx), r2t[int(idx), :n], trigger="finish")
