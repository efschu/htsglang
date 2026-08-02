"""Draft-KV geometry migration: the second blob-spec type (#261, second half).

The offline handover (``hicache_migrate``) skips ``{hash}.draft`` pages by
default, and that verdict stands: draft KV is the exact MIRROR of owner-mode
target KV (head-SHARDED, token-COMPLETE), so a suffix rename can never carry
it. This module is the real umsharder that verdict called for -- "a scoped
second spec type" -- built exactly along the sketch recorded in
``hicache_migrate``'s module docstring:

* ``DraftBlobSpec`` mirrors ``MambaBlobSpec``: the declared byte layout of one
  ``.draft`` host page (``MHATokenToKVPoolHost.get_data_page``: a flat slice
  of ``kv_buffer`` sized ``2 * layer_num * page_size * head_num * head_dim *
  itemsize``). With ``page_size == 1`` (which ``dcp_owner_mode`` already
  forces) the ``layer_first`` / ``page_first`` / ``page_first_direct`` host
  layouts all flatten to ``[2][layer][head][dim]``, so a head shard is ONE
  contiguous range per (kv-half, layer) -- the same extent family as
  ``temporal_extents``. ``page_head`` puts heads outermost, needs its own
  case, and is REFUSED until that case is built.
* Head-shard widths come from ``partition_sizes(...)``, imported from the
  runtime -- the same anti-drift rule as the GDN split.
* ``attn_kv_replicated`` (TP exceeds the draft's kv-head count) is the one
  configuration where the bytes are rank-independent and a pure key rewrite
  is enough. It is a DECLARED mode (``--draft-kv-replicated``), never
  inferred from the store: the store cannot show it.

Which speculative algorithms this may be applied to is a DECLARED capability,
registered in ONE source keyed by the canonical ``SpeculativeAlgorithm``
names (#379: one name source, parse-time refusal for unknown names -- the
resolver goes through ``SpeculativeAlgorithm.from_string`` and the alias set,
so this module cannot grow a second name list that drifts). Where re-sharding
is not declared compatible, the verdict is a refusal that names the algorithm
and the reason -- never a silent conversion and never a silent skip (the #411
contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

# Host memory layouts of the draft pool whose flattened byte order is
# [2][layer][head][dim] at page_size == 1. ``page_head`` is NOT among them.
FLAT_HEAD_CONTIGUOUS_LAYOUTS = ("layer_first", "page_first", "page_first_direct")
REFUSED_LAYOUTS = ("page_head",)


class DraftReshardError(ValueError):
    """A draft migration input that cannot be honored. Always names why."""


class DraftReshardRefusal(DraftReshardError):
    """A DECLARED refusal: the configuration is recognized but re-sharding is
    not declared compatible with it. Never a silent conversion (#411)."""


class DraftReshardCapability(Enum):
    """What the registry declares for one speculative algorithm."""

    RESHARD = "reshard"  # DraftBlobSpec migration (or declared key rewrite)
    NO_DRAFT_KV = "no_draft_kv"  # the algorithm has no draft KV state at all
    REFUSE = "refuse"  # recognized, but the store layout is not modeled


@dataclass(frozen=True)
class DraftReshardVerdict:
    algorithm: str  # canonical (enum-member) name
    capability: DraftReshardCapability
    reason: str


# ONE source for "which algorithm's draft state may be re-sharded", keyed by
# canonical SpeculativeAlgorithm member names. Every row is a declaration we
# can back with read code, nothing more; widening a REFUSE row to RESHARD
# requires modeling that algorithm's store behavior first. Keys are audited
# against the enum by ``audit_capability_names`` (called by the resolver), so
# a renamed algorithm breaks loudly here instead of silently orphaning a row.
DRAFT_RESHARD_CAPABILITIES = {
    "EAGLE": DraftReshardVerdict(
        "EAGLE",
        DraftReshardCapability.RESHARD,
        "plain-MHA linear-append draft pool; the .draft host page is "
        "MHATokenToKVPoolHost.get_data_page, modeled by DraftBlobSpec",
    ),
    "NGRAM": DraftReshardVerdict(
        "NGRAM",
        DraftReshardCapability.NO_DRAFT_KV,
        "ngram speculation keeps no draft model KV; there is nothing to "
        "re-shard and nothing to lose",
    ),
    "NONE": DraftReshardVerdict(
        "NONE",
        DraftReshardCapability.NO_DRAFT_KV,
        "speculative decoding is off; there is no draft KV state",
    ),
    "EAGLE3": DraftReshardVerdict(
        "EAGLE3",
        DraftReshardCapability.REFUSE,
        "EAGLE3's draft store layout (multi-layer draft with auxiliary "
        "hidden capture) is not modeled; refusing rather than guessing",
    ),
    "STANDALONE": DraftReshardVerdict(
        "STANDALONE",
        DraftReshardCapability.REFUSE,
        "standalone-draft store layout is not modeled; refusing rather "
        "than guessing",
    ),
    "DFLASH": DraftReshardVerdict(
        "DFLASH",
        DraftReshardCapability.REFUSE,
        "DFLASH's draft KV access pattern is not modeled for the store; "
        "refusing rather than guessing",
    ),
    "DSPARK": DraftReshardVerdict(
        "DSPARK",
        DraftReshardCapability.REFUSE,
        "DSPARK's draft KV access pattern is not modeled for the store; "
        "refusing rather than guessing",
    ),
    "FROZEN_KV_MTP": DraftReshardVerdict(
        "FROZEN_KV_MTP",
        DraftReshardCapability.REFUSE,
        "frozen-KV MTP's draft/target pool relationship is not modeled for "
        "the store; refusing rather than guessing",
    ),
}

# Aliases the arg hook resolves before the enum sees them. The capability of
# an alias is the capability of what it resolves to.
_ALIAS_CAPABILITY = {"NEXTN": "EAGLE"}


def audit_capability_names() -> None:
    """Every registry key must be a canonical enum member name and every
    enum member must have a row: the two sources cannot drift silently."""
    from sglang.srt.speculative.spec_info import (
        SPECULATIVE_ALGORITHM_ALIASES,
        SpeculativeAlgorithm,
    )

    enum_names = {member.name for member in SpeculativeAlgorithm}
    rows = set(DRAFT_RESHARD_CAPABILITIES)
    if rows - enum_names:
        raise DraftReshardError(
            f"draft re-shard capability rows {sorted(rows - enum_names)} name "
            "no SpeculativeAlgorithm member -- the registry drifted"
        )
    if enum_names - rows:
        raise DraftReshardError(
            f"SpeculativeAlgorithm member(s) {sorted(enum_names - rows)} have "
            "no draft re-shard capability row -- declare one (RESHARD, "
            "NO_DRAFT_KV or REFUSE with a reason)"
        )
    if set(_ALIAS_CAPABILITY) != set(SPECULATIVE_ALGORITHM_ALIASES):
        raise DraftReshardError(
            f"alias capability map {sorted(_ALIAS_CAPABILITY)} does not match "
            f"SPECULATIVE_ALGORITHM_ALIASES {sorted(SPECULATIVE_ALGORITHM_ALIASES)}"
        )


def resolve_draft_reshard(name: str) -> DraftReshardVerdict:
    """Capability verdict for one algorithm name, at parse time.

    Unknown names are refused HERE, with the full ``known_names()`` list --
    the same refusal ``--speculative-algorithm`` gives (#379). Registered
    plugin algorithms without a declared capability row are refused too: a
    valid serving algorithm is not automatically a re-shardable one.
    """
    from sglang.srt.speculative.spec_info import (
        SPECULATIVE_ALGORITHM_ALIASES,
        SpeculativeAlgorithm,
    )

    audit_capability_names()
    upper = name.upper()
    if upper in SPECULATIVE_ALGORITHM_ALIASES:
        canonical = _ALIAS_CAPABILITY[upper]
    else:
        try:
            resolved = SpeculativeAlgorithm.from_string(upper)
        except ValueError:
            raise DraftReshardError(
                f"unknown speculative algorithm '{name}' for "
                "--draft-spec-algorithm; known names: "
                f"{', '.join(SpeculativeAlgorithm.known_names())}"
            ) from None
        canonical = (
            resolved.name if isinstance(resolved, SpeculativeAlgorithm) else upper
        )
    verdict = DRAFT_RESHARD_CAPABILITIES.get(canonical)
    if verdict is None:
        # Only reachable for plugin-registered algorithms; builtin members
        # are guaranteed a row by audit_capability_names().
        return DraftReshardVerdict(
            canonical,
            DraftReshardCapability.REFUSE,
            f"'{canonical}' is a registered plugin algorithm with no declared "
            "draft re-shard capability; declare one before migrating its "
            "draft state",
        )
    return verdict


# ---------------------------------------------------------------------------
# Draft blob geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftBlobSpec:
    """Byte layout of ONE ``{hash}.draft`` blob (page_size == 1).

    Flattened order ``[2][layer][head][dim]`` -- K halves of every layer
    first, then V halves, layer-major, heads contiguous inside a layer. Sizes
    are the FULL (unsharded) ones; a rank's own blob is the same layout with
    the sharded head count.

    ``num_layers`` is the DRAFT model's layer count (independent of the
    target model's -- one for the MTP/NEXTN chain). ``units`` is the number
    of indivisible units handed to ``partition_sizes``; ``None`` means one
    unit per kv-head (head granularity).
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    itemsize: int
    mem_layout: str = "layer_first"
    units: Optional[int] = None

    def __post_init__(self):
        if self.mem_layout in REFUSED_LAYOUTS:
            raise DraftReshardRefusal(
                f"draft host layout '{self.mem_layout}' puts heads outermost; "
                "its extent family is not built -- refusing rather than "
                "cutting the wrong axis"
            )
        if self.mem_layout not in FLAT_HEAD_CONTIGUOUS_LAYOUTS:
            raise DraftReshardError(
                f"unknown draft host layout '{self.mem_layout}'; known: "
                f"{', '.join(FLAT_HEAD_CONTIGUOUS_LAYOUTS + REFUSED_LAYOUTS)}"
            )
        for field_name in ("num_layers", "num_kv_heads", "head_dim", "itemsize"):
            if getattr(self, field_name) <= 0:
                raise DraftReshardError(f"{field_name} must be positive")
        if self.units is not None and self.units <= 0:
            raise DraftReshardError("units must be positive (or None for head granularity)")

    @property
    def per_head_bytes(self) -> int:
        return self.head_dim * self.itemsize

    @property
    def half_layer_bytes(self) -> int:
        """One (kv-half, layer) block: all heads of that half of that layer."""
        return self.num_kv_heads * self.per_head_bytes

    @property
    def total_bytes(self) -> int:
        return 2 * self.num_layers * self.half_layer_bytes

    @property
    def effective_units(self) -> int:
        """One unit per kv-head unless a coarser unit count is declared."""
        return self.num_kv_heads if self.units is None else self.units

    def shard_for_rank(self, ratios: Sequence[int], rank: int) -> "DraftBlobSpec":
        heads = _partition(self.num_kv_heads, ratios, self.effective_units)[rank]
        return DraftBlobSpec(
            num_layers=self.num_layers,
            num_kv_heads=heads,
            head_dim=self.head_dim,
            itemsize=self.itemsize,
            mem_layout=self.mem_layout,
            units=None,
        )


def _partition(total: int, ratios: Sequence[int], units: int) -> List[int]:
    """Per-rank head counts under the runtime's own partition rule --
    imported so the migration cannot drift from what the ranks expect."""
    from sglang.srt.distributed.utils import partition_sizes

    return partition_sizes(total, list(ratios), units)


def _prefix_offsets(sizes: Sequence[int]) -> List[int]:
    out, acc = [], 0
    for s in sizes:
        out.append(acc)
        acc += s
    return out


def draft_extents(
    spec: DraftBlobSpec, ratios: Sequence[int], rank: int
) -> List[Tuple[int, int]]:
    """(offset, length) byte ranges of ``rank``'s shard of one FULL blob, in
    target order: one contiguous head slice per (kv-half, layer)."""
    sizes = _partition(spec.num_kv_heads, ratios, spec.effective_units)
    offs = _prefix_offsets(sizes)
    out = []
    for half in range(2):
        for layer in range(spec.num_layers):
            base = (half * spec.num_layers + layer) * spec.half_layer_bytes
            out.append(
                (
                    base + offs[rank] * spec.per_head_bytes,
                    sizes[rank] * spec.per_head_bytes,
                )
            )
    return out


def draft_reverse_extents(
    spec: DraftBlobSpec, ratios: Sequence[int]
) -> List[Tuple[int, int, int]]:
    """``(rank, offset in THAT rank's blob, length)`` in FULL-blob order.

    Per (kv-half, layer) the full blob runs rank 0's heads, rank 1's heads,
    ... -- so the outer loops are (half, layer) and the inner loop is the
    rank. Each rank's own blob is the same [2][layer][head][dim] order with
    its sharded head count, so its (half, layer) block sits at the
    shard-local offset, not the full-blob one.
    """
    shards = [spec.shard_for_rank(ratios, r) for r in range(len(ratios))]
    out: List[Tuple[int, int, int]] = []
    for half in range(2):
        for layer in range(spec.num_layers):
            for rank, s in enumerate(shards):
                off = (half * s.num_layers + layer) * s.half_layer_bytes
                out.append((rank, off, s.half_layer_bytes))
    return out


def draft_shard_sizes(spec: DraftBlobSpec, ratios: Sequence[int]) -> List[int]:
    """Per-rank head counts -- the target blobs' own geometry."""
    return _partition(spec.num_kv_heads, ratios, spec.effective_units)
