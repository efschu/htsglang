# SPDX-License-Identifier: Apache-2.0
"""Canonical full-head layout for the draft KV crossing a PD boundary (#631b).

Speculative decoding is refused on a PD arm (#631a) because the MTP/EAGLE
draft KV pool is uneven-head-sharded and "its transfer would need general
uneven head reslicing". This module is the alternative that avoids needing it.

**The shape of the idea.** The PREFILL arm computes the draft layer's KV
during its own prefill -- where the hidden states already exist in flight, so
the marginal cost is one layer on top of the model's full depth -- and ships
it in a CANONICAL layout holding every KV head. The decode arm then slices out
the heads its own sharding owns. Neither side ever has to understand the
other's sharding, which is precisely the "general uneven head reslicing" the
refusal balks at.

**Why not the two obvious alternatives.**

*Recompute locally on the decode arm.* The decode arm cannot: it receives the
main model's K and V, and K/V are projections that do not invert back into
hidden states. So it is not "one layer of recompute", it is "ship the last
layer's hidden states for every prompt position, then recompute one layer".
On the reference checkpoint that payload is ``hidden_size`` 5120 x 2 B =
10240 B/token, five times this module's 2048 B/token.

*General uneven head reslicing.* It dies on arithmetic, not on effort. The
existing pairwise helper, ``staging_buffer.compute_head_slice_params``,
computes ``total_kv_heads // attn_tp_size``; with 4 KV heads over 3 decode
ranks that is 1, and a head is silently dropped. A canonical intermediate
needs no pairwise mapping at all -- each side maps only between its OWN local
heads and global head indices.

**Two invariants this module exists to hold** (both are the whole point, per
the #631b review):

1. The layout is VERSIONED, and a mismatch is a named refusal. The moment one
   side can guess wrong about the other's layout, the silent-wrongness problem
   that variant (iii) was chosen to avoid has been rebuilt inside it.
2. The full-head shipment is justified from RUNTIME geometry, never from the
   reference checkpoint's numbers. A wider-KV checkpoint makes the premise
   false, and :func:`check_full_head_shipment_is_justified` refuses there
   rather than quietly shipping a payload larger than the alternative it was
   chosen over.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

__all__ = [
    "CANONICAL_LAYOUT_VERSION",
    "DraftKvCanonicalLayout",
    "DraftKvLayoutMismatch",
    "check_full_head_shipment_is_justified",
    "local_head_window",
]

#: Version of the canonical wire layout. BUMP whenever the meaning of the
#: bytes changes -- head ordering, K/V interleaving, element type handling,
#: or the per-token stride. Never reuse a version for a changed meaning: the
#: refusal below is only as good as this number's honesty.
CANONICAL_LAYOUT_VERSION = 1


class DraftKvLayoutMismatch(RuntimeError):
    """The two arms do not agree on the canonical draft-KV layout."""


@dataclasses.dataclass(frozen=True)
class DraftKvCanonicalLayout:
    """What both arms must agree on before a single draft-KV byte moves.

    Built from LOCAL facts on each side and compared at the handshake, in the
    same spirit as ``TransportIdentity`` -- no collective decides it, so a
    disagreement is found by comparison rather than by a group that
    half-joined.

    Every field is part of the meaning of the bytes. ``num_kv_heads`` is the
    checkpoint's TOTAL head count, not any rank's share: that is what makes
    the layout canonical, and shipping the total is what lets a 4-head model
    cross onto 3 ranks at all.
    """

    version: int
    #: TOTAL kv heads in the checkpoint, not this rank's share.
    num_kv_heads: int
    head_dim: int
    #: Bytes per element of the draft KV cache (e.g. 1 for fp8, 2 for bf16).
    element_size: int
    #: Draft layers carried. NEXTN/MTP is 1; kept explicit so a multi-layer
    #: draft cannot be mistaken for a single-layer one by size alone.
    num_draft_layers: int

    def bytes_per_token(self) -> int:
        """K and V, every head, all draft layers, for one token."""
        return (
            2
            * self.num_kv_heads
            * self.head_dim
            * self.element_size
            * self.num_draft_layers
        )

    def assert_compatible(self, other: "DraftKvCanonicalLayout", *, peer: str) -> None:
        """Raise unless ``other`` means the same bytes. Loud and specific."""
        if self.version != other.version:
            raise DraftKvLayoutMismatch(
                f"draft-KV canonical layout version mismatch with {peer}: "
                f"local v{self.version}, peer v{other.version}. The layout "
                "version is refused rather than reinterpreted, because a "
                "guess about the peer's byte order would reproduce exactly "
                "the silent wrong-output failure this canonical layout was "
                "introduced to remove. Run both arms on builds that share a "
                "layout version."
            )
        problems = [
            f"{name}: local {getattr(self, name)} != peer {getattr(other, name)}"
            for name in ("num_kv_heads", "head_dim", "element_size", "num_draft_layers")
            if getattr(self, name) != getattr(other, name)
        ]
        if problems:
            raise DraftKvLayoutMismatch(
                f"draft-KV canonical layout mismatch with {peer}: "
                + "; ".join(problems)
                + ". Same version, different geometry -- the arms are not "
                "serving the same draft, so a transfer would place bytes the "
                "receiver reads as different heads."
            )


def check_full_head_shipment_is_justified(
    layout: DraftKvCanonicalLayout,
    hidden_size: int,
    hidden_element_size: int,
) -> None:
    """Refuse when shipping every head costs more than the option it replaced.

    Variant (iii) was chosen over local recompute on ONE quantitative ground:
    the whole draft KV is smaller than the hidden states local recompute would
    have to be fed. That is a property of the CHECKPOINT, not of the design,
    and it is false for a wide-KV model -- an MHA checkpoint with many KV
    heads inverts the comparison. Rather than let the reference model's
    numbers sit baked in as an assumption, the comparison is re-derived here
    from the geometry actually loaded, every time.

    The bound is deliberately the alternative's own cost rather than a round
    number in MB: a megabyte threshold would be arbitrary and would rot, while
    "cheaper than the thing we rejected it for" stays meaningful on any
    checkpoint and explains itself to the next reader.
    """
    canonical = layout.bytes_per_token()
    recompute = hidden_size * hidden_element_size
    if canonical > recompute:
        raise DraftKvLayoutMismatch(
            "canonical full-head draft-KV shipment is not justified on this "
            f"checkpoint: it costs {canonical} B/token "
            f"({layout.num_kv_heads} kv heads x {layout.head_dim} head_dim x "
            f"2 (K+V) x {layout.element_size} B x "
            f"{layout.num_draft_layers} layer(s)), against {recompute} B/token "
            f"for shipping hidden states instead ({hidden_size} x "
            f"{hidden_element_size} B). Shipping every head was chosen "
            "BECAUSE it was the cheaper of the two on narrow-KV (GQA) "
            "checkpoints; on this geometry that reason is gone, so the "
            "configuration is refused instead of silently moving more bytes "
            "than the alternative it replaced."
        )


def local_head_window(
    num_kv_heads: int,
    tp_size: int,
    tp_rank: int,
) -> Tuple[int, int]:
    """The ``[start, end)`` canonical head range this rank owns.

    Each side calls this for ITSELF against the canonical total, which is what
    keeps the two arms decoupled -- there is no src/dst pair here, and so no
    pairwise divisibility assumption to violate.

    Heads are dealt out largest-remainder-first, so an indivisible count is
    spread rather than truncated: 4 heads over 3 ranks is (2, 1, 1) and every
    head has exactly one owner. ``staging_buffer.compute_head_slice_params``
    takes ``num_kv_heads // tp_size`` and would make that (1, 1, 1), dropping
    head 3 -- silently, since nothing downstream counts heads. That is the
    arithmetic wall general reslicing runs into, and the reason this function
    does not reuse it.

    A rank beyond the head count owns an EMPTY window rather than raising:
    replicated-KV layouts legitimately place more ranks than heads, and those
    ranks read a replica instead of a slice.
    """
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank {tp_rank} out of range for tp_size {tp_size}")
    if num_kv_heads < 0:
        raise ValueError(f"num_kv_heads must be non-negative, got {num_kv_heads}")

    base, remainder = divmod(num_kv_heads, tp_size)
    # Ranks below `remainder` take one extra head; the offset is therefore
    # base*rank plus however many extra heads were already handed out.
    start = base * tp_rank + min(tp_rank, remainder)
    count = base + (1 if tp_rank < remainder else 0)
    return start, start + count
