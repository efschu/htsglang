"""#704b: the canonical KV page (Option A), device side.

#706 settled the page shape: a page carries ALL attention layers for its token
range, layer-major within the page. ``page_size == 1`` is mandatory (required
by ``dcp_owner_mode``, since a multi-token page would span owner ranks), so a
page is ONE token -- on this checkpoint 16 x 2048 = 32,768 B.

The contract both strands carry says the stored form is CANONICAL and
layout-neutral: it depends on the model geometry ALONE, never on the PP cut,
the token-share vector, or which phase wrote the bytes. That is what lets one
key name one set of bytes across both phases, and it is asserted directly
rather than assumed.

Two ways it can be violated silently, both guarded here:

**1. Rank-local layer indices.** Option A globalises ``start_layer`` precisely
so a page slot is addressed by the GLOBAL attention-layer index. A stage that
writes its layers at rank-local offsets deposits the right NUMBER of bytes in
the WRONG slots. That is the same failure shape as cutting a mamba blob as one
flat range -- correct length, wrong channels -- and it is equally silent.

**2. Untracked partial writes.** Token-sharding the STORAGE does not collapse a
page to a single writer: production is layer-sharded, so on this rig the 16
slots arrive from THREE PP stages (7 + 5 + 4). An unwritten slot is
indistinguishable from a legitimately zero one unless completeness is tracked,
which is why #706 records that the marker must be built.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch


class CanonicalPageError(ValueError):
    """A page layout violation. Never downgraded to a warning."""


@dataclasses.dataclass(frozen=True)
class CanonicalPageSpec:
    """Geometry of one canonical page. Equality IS the layout contract.

    Deliberately carries nothing about the cut, the shares, or the phase: two
    ranks with different layouts must compare equal here, or the same key would
    name different bytes.
    """

    num_attn_layers: int
    kv_bytes_per_token_per_attn_layer: int

    @property
    def page_bytes(self) -> int:
        return int(self.num_attn_layers) * int(self.kv_bytes_per_token_per_attn_layer)

    def layer_span(self, slot: int) -> tuple[int, int]:
        """Byte range of one attention slot, layer-major."""
        if not 0 <= int(slot) < int(self.num_attn_layers):
            raise CanonicalPageError(
                f"slot {slot} is outside the {self.num_attn_layers} attention "
                "slots this page carries."
            )
        cell = int(self.kv_bytes_per_token_per_attn_layer)
        lo = int(slot) * cell
        return lo, lo + cell


def attn_layer_index(global_layer_id: int, attn_layer_ids: Sequence[int]) -> int:
    """GLOBAL attention-layer index of a global layer id.

    The only admissible way to address a page slot. Takes the model's own
    attention-layer id list, so a rank cannot substitute its local ordering:
    the argument is global by construction, not by convention.
    """
    ids = list(attn_layer_ids)
    try:
        return ids.index(int(global_layer_id))
    except ValueError:
        raise CanonicalPageError(
            f"layer {global_layer_id} is not a full-attention layer, so it has "
            "no page slot. Only attention layers carry token-scaling KV; linear "
            "(GDN) layers are served by MambaPool and have their own canonical "
            "form (see the #706 mamba spec)."
        ) from None


def scatter_page(page: torch.Tensor, spec: CanonicalPageSpec) -> list[torch.Tensor]:
    """Per-slot VIEWS into a canonical page. Copies nothing."""
    if page.dtype != torch.uint8:
        raise CanonicalPageError(
            f"a canonical page is raw bytes (uint8), got {page.dtype}."
        )
    if int(page.numel()) != spec.page_bytes:
        raise CanonicalPageError(
            f"page holds {int(page.numel())} bytes but the spec describes "
            f"{spec.page_bytes} bytes; refusing to pad or truncate."
        )
    out = []
    for slot in range(int(spec.num_attn_layers)):
        lo, hi = spec.layer_span(slot)
        out.append(page[lo:hi])
    return out


def gather_page(
    per_layer: Sequence[torch.Tensor], spec: CanonicalPageSpec
) -> torch.Tensor:
    """Assemble a canonical page from per-slot buffers, layer-major."""
    if len(per_layer) != int(spec.num_attn_layers):
        raise CanonicalPageError(
            f"{len(per_layer)} slots supplied for a page of "
            f"{spec.num_attn_layers}; a page missing a slot is not a shorter "
            "page, it is an incomplete one -- see PageCompleteness."
        )
    cell = int(spec.kv_bytes_per_token_per_attn_layer)
    for i, buf in enumerate(per_layer):
        if int(buf.numel()) != cell:
            raise CanonicalPageError(
                f"slot {i} holds {int(buf.numel())} bytes, expected {cell}."
            )
    return torch.cat([b.reshape(-1) for b in per_layer], dim=0)


class PageCompleteness:
    """Tracks which attention slots of a page have been written.

    Required because production is layer-sharded while storage is
    token-sharded: the slots of one page arrive from several PP stages, and a
    zero slot that was never written reads exactly like a zero slot that was.
    """

    def __init__(self, spec: CanonicalPageSpec) -> None:
        self._spec = spec
        self._written: set[int] = set()

    def mark(self, slot: int) -> None:
        if not 0 <= int(slot) < int(self._spec.num_attn_layers):
            raise CanonicalPageError(
                f"slot {slot} is outside the {self._spec.num_attn_layers} "
                "attention slots this page carries."
            )
        if int(slot) in self._written:
            raise CanonicalPageError(
                f"slot {slot} was already written. Two writers for one slot is "
                "a layout bug -- the stage ranges must partition the attention "
                "layers -- not an idempotent retry."
            )
        self._written.add(int(slot))

    def missing(self) -> tuple[int, ...]:
        return tuple(
            i for i in range(int(self._spec.num_attn_layers)) if i not in self._written
        )

    def is_complete(self) -> bool:
        return len(self._written) == int(self._spec.num_attn_layers)
