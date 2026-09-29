"""#704b: the canonical KV page (Option A), device side.

#706 settled the page shape: a page carries ALL attention layers for its
token range. ``page_size == 1`` is mandatory (required by ``dcp_owner_mode``,
since a multi-token page would span owner ranks), so a page is ONE token -- on
this checkpoint 16 x 2048 = 32,768 B.

BYTE ORDER (#1233, boot weg2ls3b4 root): the page is K/V-MAJOR, not
layer-major. It is the flat host page as ``MHATokenToKVPoolHost.get_data_page``
produces it and ``set_from_flat_data_page`` consumes it (``pool_host/mha.py``:
``layer_first`` dims ``(2, layer_num, size, H, D)``, sliced per token and
flattened): ``[K slot0 .. K slotN-1][V slot0 .. V slotN-1]``, one HALF-CELL of
``head_num * head_dim * itemsize`` bytes per attention slot per K/V half. A
slot's bytes are therefore TWO ranges, one in each half, never one contiguous
cell. A rank holding every attention layer is self-consistent under either
reading, which is why a same-geometry round trip cannot detect the difference;
three PP stages depositing their local ``[K][V]`` blocks as one extent each
scrambled 20 of 32 half-cells, and every answer decoded from the other group's
pages was wrong while the GDN blob (composed per region) arrived intact.

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

    def __post_init__(self) -> None:
        if int(self.kv_bytes_per_token_per_attn_layer) % 2:
            raise CanonicalPageError(
                f"a cell of {self.kv_bytes_per_token_per_attn_layer} bytes "
                "cannot be split into a K half and a V half; the page is "
                "K/V-major and every slot has two equal halves."
            )

    @property
    def page_bytes(self) -> int:
        return int(self.num_attn_layers) * int(self.kv_bytes_per_token_per_attn_layer)

    @property
    def half_cell_bytes(self) -> int:
        """Bytes of one attention slot in ONE K/V half (``head_num * head_dim *
        itemsize``, ``MHATokenToKVPoolHost.token_stride_size``)."""
        return int(self.kv_bytes_per_token_per_attn_layer) // 2

    @property
    def half_page_bytes(self) -> int:
        """Where the V region starts: all K half-cells come first."""
        return int(self.num_attn_layers) * self.half_cell_bytes

    def slot_spans(self, slot: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """The TWO byte ranges of one attention slot: its K half-cell in the
        K region, its V half-cell in the V region. Never one range -- a page is
        K/V-major (see the module docstring)."""
        if not 0 <= int(slot) < int(self.num_attn_layers):
            raise CanonicalPageError(
                f"slot {slot} is outside the {self.num_attn_layers} attention "
                "slots this page carries."
            )
        half = self.half_cell_bytes
        k_lo = int(slot) * half
        v_lo = self.half_page_bytes + k_lo
        return (k_lo, k_lo + half), (v_lo, v_lo + half)


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
