"""Geometry migration of a persisted HiCache ``file`` store (#121 handover).

Moves a session's persisted context from the geometry that WROTE it to a
different one -- concretely: a fast single-card run (TP=1) hands its work to
the big group (TP=3, uneven ratio, weighted uneven DCP). The store survives
the process; this module rewrites it so the next boot's ranks find their own
shards.

Why a separate, explicit operation
----------------------------------
HiCache storage keys carry the parallel geometry on purpose (task #241):
``config_suffix = _{model}_{identity_hash}_{tp_rank}_{tp_size}``. That is what
keeps a run from reading pages another geometry wrote. A handover must cross
that line, so it crosses it VISIBLY -- an offline, auditable rewrite of the
store, never a softened key.

What actually has to move, and what does not
--------------------------------------------
* **KV pages are geometry-neutral in their BYTES.** Under weighted uneven DCP
  the store runs in ``dcp_owner_mode``: pages hold the FULL replicated
  kv-heads and are TOKEN-sharded, so a page is complete on whichever rank
  owned it (``cache_controller._dcp_kv_transfer_pairs``,
  ``HiCacheStorageConfig.dcp_owner_mode``). A TP=1 run writes exactly those
  bytes too -- one rank owning every token. Migrating KV is therefore a pure
  key rewrite: drop the ``_0_1`` rank suffix. Which rank later reads which
  page is decided at read time by the consuming boot's own owner rule; no
  byte is touched, reordered, or recomputed.
* **The GDN/Mamba recurrent state is HEAD-sharded and must be cut.** A
  ``{hash}.mamba`` blob is one radix node's complete state. Splitting it for N
  ranks means slicing the temporal state by heads and the conv state by its
  ``[query_key | query_key | value]`` sub-blocks -- each sub-block sharded
  INDEPENDENTLY by heads, so a rank's conv shard is three concatenated ranges,
  not one contiguous slice (``MambaPool.get_conv_subblock_spec``). Cutting it
  as one flat range delivers the wrong channels.
* **Nothing is recomputed and nothing is quantized.** Every byte of every
  target file is a byte of a source file at a named offset: ``plan_migration``
  returns exactly that provenance, which is what makes the migration checkable
  without a GPU (see ``test/registered/unit/mem_cache/test_hicache_migrate.py``).

Deliberate limits (named, not hidden)
-------------------------------------
* ``file`` backend only -- the same backend restriction ``dcp_owner_mode``
  itself carries, and the only one whose on-disk layout is a plain byte blob
  per key.
* ``page_size == 1`` -- required by ``dcp_owner_mode`` (a multi-token page
  would span owner ranks); the KV rewrite inherits that.
* Draft (``{hash}.draft``) pages are NOT migrated, and a suffix rule could not
  do it. Draft KV is the exact MIRROR of target KV under ``dcp_owner_mode``:
  target KV is head-REPLICATED and token-sharded (hence one complete, rank-
  independent file per page), while draft KV is head-SHARDED and token-
  COMPLETE. ``ModelRunner._pool_kv_head_num`` gives the draft pool this rank's
  ``get_num_kv_heads(attn_tp_size)`` shard, and the draft worker is explicitly
  held out of the DCP token split (``_draft_non_dcp``;
  ``cache_controller.start_writing`` hands the draft path the RAW index pairs,
  not ``_dcp_kv_transfer_pairs``' compacted ones). So every rank writes every
  draft page, each under its own rank suffix, each holding a different head
  subset -- and a TP=1 ``.draft`` blob is a different LENGTH from any TP=3
  rank's. Renaming would deliver a wrong-sized file of other ranks' heads.
  Carrying them needs a real byte split, sketched under "What a draft
  migration would take" below. Run the handover without speculative decoding,
  or accept that the draft pool starts cold.
* Only 1->N and N->1 are built. Both directions go through the SAME full,
  unsharded state as their pivot, so a general N->M reshard is the two chained
  -- there is no direct shard-to-shard path here.

The reverse direction (N -> 1, reassembly)
------------------------------------------
``plan_reverse_migration`` hands the big group's context back to the single
card. It is not the forward plan run backwards by symmetry -- the source is now
N rank blobs that must be REASSEMBLED into one full state before anything can
be named:

* KV pages need no reassembly at all. In ``dcp_owner_mode`` they are already
  one shared, rank-less file per page holding full replicated kv-heads, so the
  reverse is again a pure key rewrite -- this time ADDING the ``_0_1`` suffix a
  TP=1 boot looks for. Outside ``dcp_owner_mode`` the reverse is refused rather
  than guessed: per-rank KV pages there are genuine head shards, and stitching
  them is a different operation from renaming.
* The GDN state is interleaved back. For each layer the temporal state is
  rank 0's heads, then rank 1's, ...; for each layer the conv state is rank 0's
  ``q`` shard, rank 1's ``q`` shard, ... then the same for ``k`` and for ``v``.
  The ``[query_key | query_key | value]`` rule bites here too, in mirror image:
  a rank's conv shard is three separate ranges of the full blob, so
  concatenating whole rank shards per layer would interleave the sub-blocks
  wrongly.

``verify_plan`` is direction-agnostic -- it only ever asks whether the bytes on
disk match the named source extents and whether every source byte is consumed
exactly once -- so the same gate proves both directions, and chaining them
gives the round-trip byte gate (``TestRoundTrip`` in the unit proof).

What a draft migration would take (not built)
---------------------------------------------
Recorded so the next slice does not have to re-derive it. A ``.draft`` page is
a plain MHA host page, ``MHATokenToKVPoolHost.get_data_page``: a flat slice of
``kv_buffer``, sized ``2 * layer_num * page_size * head_num * head_dim *
itemsize``. Migrating it needs, mirroring ``MambaBlobSpec``:

1. a ``DraftBlobSpec``: the DRAFT model's ``layer_num`` (independent of the
   target model's), ``head_dim``, dtype itemsize, total kv-head count, and the
   ``hicache_mem_layout`` -- four layouts exist and they do not agree on where
   ``head_num`` sits. With ``page_size == 1`` (which ``dcp_owner_mode`` already
   forces) ``layer_first``/``page_first``/``page_first_direct`` all flatten to
   ``[2][layer][head][dim]``, so a head shard is one contiguous range per
   (kv-half, layer) -- the same extent shape as ``temporal_extents``.
   ``page_head`` puts heads outermost and needs its own case.
2. head-shard widths from ``partition_sizes(total_kv_heads, ratios, ...)``,
   imported from the runtime for the same anti-drift reason as the GDN split.
3. the two configurations where the bytes ARE rank-independent and a pure key
   rewrite is enough -- ``attn_kv_replicated`` (TP exceeds the draft's kv-head
   count, so every rank holds all heads) and an MLA draft pool. Neither is
   visible in the store; both must come from the caller's config.

That is a second spec type, a second extent family with a layout switch, and a
CLI surface for all of it -- comfortably past the point where bolting it onto
this module is the cheap option, and none of it is checkable without a boot
that runs speculative decoding on both sides. Hence: documented, not guessed.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Byte ranges of one target file: (source path, source offset, length), in
# target order. Concatenating them IS the target file.
Extent = Tuple[str, int, int]
MigrationPlan = List[Tuple[str, List[Extent]]]

# `{hash}.{pool}` component suffix used by HiCacheStorage._get_component_key.
MAMBA_POOL = "mamba"
DRAFT_POOL = "draft"

_STEM_RE = re.compile(
    r"^(?P<key>[0-9a-fA-F]+)(?:\.(?P<pool>[a-z0-9_]+))?_(?P<rest>.+)$"
)


@dataclass(frozen=True)
class StoreEntry:
    """One file of a HiCache ``file`` store, decomposed."""

    path: str
    key: str  # page hash
    pool: Optional[str]  # None for plain KV pages
    suffix_rest: str  # everything after the key, without the leading '_'
    size: int

    @property
    def is_kv(self) -> bool:
        return self.pool is None


def parse_store_filename(path: str, size: int = 0) -> Optional[StoreEntry]:
    """Decompose ``<hash>[.<pool>]_<model>_<idhash>_<tp_rank>_<tp_size>.bin``.

    Returns None for anything that is not a store blob (the store directory
    also holds the backend's own bookkeeping files)."""
    name = os.path.basename(path)
    if not name.endswith(".bin"):
        return None
    m = _STEM_RE.match(name[:-4])
    if m is None:
        return None
    return StoreEntry(
        path=path,
        key=m.group("key"),
        pool=m.group("pool"),
        suffix_rest=m.group("rest"),
        size=size,
    )


def scan_store(directory: str) -> List[StoreEntry]:
    entries = []
    for name in sorted(os.listdir(directory)):
        p = os.path.join(directory, name)
        if not os.path.isfile(p):
            continue
        e = parse_store_filename(p, os.path.getsize(p))
        if e is not None:
            entries.append(e)
    return entries


def strip_rank_suffix(suffix_rest: str, tp_rank: int, tp_size: int) -> str:
    """Remove the trailing ``_{tp_rank}_{tp_size}`` a non-MLA store appends.

    Raises when it is not there: the source geometry was misdeclared, and
    guessing would silently migrate the wrong files."""
    tail = f"_{tp_rank}_{tp_size}"
    if not suffix_rest.endswith(tail):
        raise ValueError(
            f"store entry suffix '{suffix_rest}' does not end in '{tail}' -- "
            f"source geometry (tp_rank={tp_rank}, tp_size={tp_size}) does not "
            "match this store"
        )
    return suffix_rest[: -len(tail)]


def target_kv_name(
    entry: StoreEntry,
    base_suffix: str,
    dcp_owner_mode: bool,
    tp_rank: int,
    tp_size: int,
) -> str:
    """Target filename of a KV page. In ``dcp_owner_mode`` KV pages are
    rank-shared and carry no rank suffix; otherwise they stay per-rank."""
    if dcp_owner_mode:
        return f"{entry.key}_{base_suffix}.bin"
    return f"{entry.key}_{base_suffix}_{tp_rank}_{tp_size}.bin"


def target_mamba_name(
    entry: StoreEntry, base_suffix: str, tp_rank: int, tp_size: int
) -> str:
    """Component pools stay per-rank in every mode (genuine head shards)."""
    return f"{entry.key}.{MAMBA_POOL}_{base_suffix}_{tp_rank}_{tp_size}.bin"


def target_draft_name(
    entry: StoreEntry, base_suffix: str, tp_rank: int, tp_size: int
) -> str:
    """Draft pages are a component pool too: per-rank in every mode."""
    return f"{entry.key}.{DRAFT_POOL}_{base_suffix}_{tp_rank}_{tp_size}.bin"


# ---------------------------------------------------------------------------
# Manifest scoping (#261 live handover)
# ---------------------------------------------------------------------------


def load_manifest(path: str) -> Dict:
    """Load a session-handover manifest (see ``managers/session_handover``).

    Only the key inventory is consumed here; identity/geometry checks belong
    to the destination's verify-import step."""
    import json

    with open(path) as f:
        manifest = json.load(f)
    for field in ("kv_keys",):
        if field not in manifest:
            raise ValueError(f"manifest {path} has no '{field}' field")
    return manifest


def manifest_key_pairs(manifest: Dict) -> List[Tuple[str, Optional[str]]]:
    """(page hash, pool-or-None) of every store blob the manifest names.

    The leaf hash may legally appear twice -- once as a KV page and once as
    the ``.mamba`` component -- so this is a pair list, not a hash map."""
    pairs: List[Tuple[str, Optional[str]]] = [(k, None) for k in manifest["kv_keys"]]
    mamba_key = manifest.get("mamba_key")
    if mamba_key:
        pairs.append((mamba_key.split(".")[0], MAMBA_POOL))
    for draft_key in manifest.get("draft_keys") or []:
        pairs.append((draft_key.split(".")[0], DRAFT_POOL))
    return pairs


def filter_entries_by_manifest(
    entries: Sequence[StoreEntry], manifest: Dict
) -> List[StoreEntry]:
    """Restrict a store scan to exactly the manifest's blobs.

    This is what makes the migration safe against a LIVE source store: the
    manifest-listed files are complete and immutable (the session is parked;
    store files are content-addressed and written atomically), while
    everything else in the directory -- including files other sessions are
    writing right now -- is ignored. A manifest key with no file behind it is
    a hard error naming the key: a partial session state would be a
    plausible-looking but wrong session, exactly like a partial GDN state.
    """
    wanted = set(manifest_key_pairs(manifest))
    selected = [e for e in entries if (e.key, e.pool) in wanted]
    present = {(e.key, e.pool) for e in selected}
    missing = [
        key if pool is None else f"{key}.{pool}"
        for key, pool in manifest_key_pairs(manifest)
        if (key, pool) not in present
    ]
    if missing:
        raise ValueError(
            f"manifest names {len(missing)} blob(s) absent from the source "
            f"store: {missing[:8]}{' ...' if len(missing) > 8 else ''} -- "
            "refusing to migrate a partial session"
        )
    return selected


# ---------------------------------------------------------------------------
# Mamba blob geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MambaBlobSpec:
    """Byte layout of ONE ``{hash}.mamba`` blob.

    Mirrors ``MambaPoolHost.get_data_page``: the temporal state of every layer
    first, then the conv state of every layer -- both layer-major and
    C-contiguous, in both host layouts (page-first indexes a page, layer-first
    slices one page out of each layer; the flattened order is the same).

    Sizes are the FULL (unsharded) ones; the per-rank blob of a sharded run is
    the same layout with the sharded head/channel counts.
    """

    num_layers: int
    num_heads: int  # value heads (temporal state)
    head_dim: int
    state_size: int
    conv_dim: int
    conv_width: int  # conv_kernel - 1
    key_dim: int  # per query_key sub-block of conv_dim
    value_dim: int  # value sub-block of conv_dim
    units: int  # uneven-TP unit count (gdn_tp_units)
    temporal_itemsize: int
    conv_itemsize: int

    def __post_init__(self):
        if self.key_dim * 2 + self.value_dim != self.conv_dim:
            raise ValueError(
                f"conv sub-blocks {self.key_dim}+{self.key_dim}+{self.value_dim} "
                f"do not add up to conv_dim {self.conv_dim}"
            )

    @property
    def temporal_layer_bytes(self) -> int:
        return self.num_heads * self.head_dim * self.state_size * self.temporal_itemsize

    @property
    def temporal_bytes(self) -> int:
        return self.num_layers * self.temporal_layer_bytes

    @property
    def conv_layer_bytes(self) -> int:
        return self.conv_dim * self.conv_width * self.conv_itemsize

    @property
    def conv_bytes(self) -> int:
        return self.num_layers * self.conv_layer_bytes

    @property
    def total_bytes(self) -> int:
        return self.temporal_bytes + self.conv_bytes

    def shard_for_rank(self, ratios: Sequence[int], rank: int) -> "MambaBlobSpec":
        """The layout of ``rank``'s OWN blob: every sharded dimension --
        heads and each conv sub-block -- cut with the runtime's rule, so the
        result is a valid spec (sub-blocks still add up to its conv_dim)."""
        heads = _partition(self.num_heads, ratios, self.units)[rank]
        key = _partition(self.key_dim, ratios, self.units)[rank]
        value = _partition(self.value_dim, ratios, self.units)[rank]
        return MambaBlobSpec(
            num_layers=self.num_layers,
            num_heads=heads,
            head_dim=self.head_dim,
            state_size=self.state_size,
            conv_dim=key * 2 + value,
            conv_width=self.conv_width,
            key_dim=key,
            value_dim=value,
            units=self.units,
            temporal_itemsize=self.temporal_itemsize,
            conv_itemsize=self.conv_itemsize,
        )


def _partition(total: int, ratios: Sequence[int], units: int) -> List[int]:
    """Per-rank sizes under the runtime's own partition rule. Imported from
    the runtime so the migration cannot drift from what the ranks expect."""
    from sglang.srt.distributed.utils import partition_sizes

    return partition_sizes(total, list(ratios), units)


def _prefix_offsets(sizes: Sequence[int]) -> List[int]:
    out, acc = [], 0
    for s in sizes:
        out.append(acc)
        acc += s
    return out


def temporal_extents(
    spec: MambaBlobSpec, ratios: Sequence[int], rank: int
) -> List[Tuple[int, int]]:
    """(offset, length) byte ranges of ``rank``'s temporal state, in target
    order: one range per layer (this rank's contiguous head slice)."""
    sizes = _partition(spec.num_heads, ratios, spec.units)
    offs = _prefix_offsets(sizes)
    per_head = spec.head_dim * spec.state_size * spec.temporal_itemsize
    out = []
    for layer in range(spec.num_layers):
        base = layer * spec.temporal_layer_bytes
        out.append((base + offs[rank] * per_head, sizes[rank] * per_head))
    return out


def conv_extents(
    spec: MambaBlobSpec, ratios: Sequence[int], rank: int
) -> List[Tuple[int, int]]:
    """(offset, length) byte ranges of ``rank``'s conv state, in target order.

    Three ranges per layer, one per ``[query_key | query_key | value]``
    sub-block: each is sharded independently by heads, so the rank's conv
    shard is those three ranges concatenated -- not one flat slice of
    ``conv_dim``. Cutting it flat is the documented way to get the wrong
    channels (``MambaPool.get_conv_subblock_spec``)."""
    sub_fulls = [spec.key_dim, spec.key_dim, spec.value_dim]
    per_channel = spec.conv_width * spec.conv_itemsize
    pairs = []
    src_base = 0
    for sub_full in sub_fulls:
        sizes = _partition(sub_full, ratios, spec.units)
        offs = _prefix_offsets(sizes)
        pairs.append((src_base + offs[rank], sizes[rank]))
        src_base += sub_full
    out = []
    for layer in range(spec.num_layers):
        base = spec.temporal_bytes + layer * spec.conv_layer_bytes
        for off, length in pairs:
            out.append((base + off * per_channel, length * per_channel))
    return out


def reverse_extents(
    spec: MambaBlobSpec, ratios: Sequence[int]
) -> List[Tuple[int, int, int]]:
    """``(rank, offset in THAT rank's blob, length)`` in FULL-blob order.

    The inverse of ``temporal_extents`` + ``conv_extents``: concatenating these
    ranges reproduces the unsharded blob byte for byte. Two orderings have to be
    right at once, and neither is the obvious one:

    * per layer, the temporal state runs rank 0's heads, rank 1's heads, ... --
      so the outer loop is the layer and the inner loop is the rank, not the
      other way round (each rank's own blob is layer-major, so consecutive
      full-blob layers sit far apart inside one source file);
    * per layer, the conv state runs ALL ranks' ``q`` shards, then all ranks'
      ``k`` shards, then all ranks' ``v`` shards. A rank's conv shard is the
      three concatenated in ITS file, so every sub-block is a separate seek.
      Appending whole rank shards per layer -- the obvious reassembly -- would
      produce ``[q0 k0 v0 q1 k1 v1 ...]`` and hand the recurrent path the wrong
      channels, the same failure mode the forward direction guards against.
    """
    n = len(ratios)
    shards = [spec.shard_for_rank(ratios, r) for r in range(n)]
    out: List[Tuple[int, int, int]] = []
    for layer in range(spec.num_layers):
        for rank, s in enumerate(shards):
            out.append((rank, layer * s.temporal_layer_bytes, s.temporal_layer_bytes))
    per_channel = spec.conv_width * spec.conv_itemsize
    # Per rank: the byte offset of its q / k / v sub-block inside one conv layer
    # of its own blob, plus that sub-block's length.
    sub_spans = [
        list(
            zip(
                _prefix_offsets([s.key_dim, s.key_dim, s.value_dim]),
                [s.key_dim, s.key_dim, s.value_dim],
            )
        )
        for s in shards
    ]
    for layer in range(spec.num_layers):
        for sub in range(3):
            for rank, s in enumerate(shards):
                off, length = sub_spans[rank][sub]
                base = s.temporal_bytes + layer * s.conv_layer_bytes
                out.append((rank, base + off * per_channel, length * per_channel))
    return out


def shard_sizes(spec: MambaBlobSpec, ratios: Sequence[int]) -> List[Tuple[int, int]]:
    """(heads, conv_dim) per rank -- the target blob's own geometry."""
    heads = _partition(spec.num_heads, ratios, spec.units)
    sub_fulls = [spec.key_dim, spec.key_dim, spec.value_dim]
    conv = [0] * len(ratios)
    for sub_full in sub_fulls:
        for r, s in enumerate(_partition(sub_full, ratios, spec.units)):
            conv[r] += s
    return list(zip(heads, conv))


# ---------------------------------------------------------------------------
# Planning and execution
# ---------------------------------------------------------------------------


def plan_migration(
    entries: Sequence[StoreEntry],
    target_dir: str,
    *,
    source_tp_rank: int = 0,
    source_tp_size: int = 1,
    target_tp_size: int,
    target_ratios: Sequence[int],
    mamba_spec: Optional[MambaBlobSpec],
    dcp_owner_mode: bool = True,
    skip_pools: Sequence[str] = (DRAFT_POOL,),
    draft_spec=None,
    draft_key_rewrite: bool = False,
) -> MigrationPlan:
    """Full byte-provenance plan of the migration.

    Every entry is ``(target path, [(source path, offset, length), ...])`` in
    target order -- the concatenation of the extents IS the target file. No
    step computes a value; the plan is a permutation of source bytes into
    target files, which is exactly what the unit proof checks.

    Draft (``.draft``) pages are skipped by default (see the module
    docstring: the skip-instead-of-rename verdict). To carry them, drop
    ``DRAFT_POOL`` from ``skip_pools`` and pass EITHER a
    ``draft_migrate.DraftBlobSpec`` (real head split) OR
    ``draft_key_rewrite=True`` (the declared ``attn_kv_replicated``
    configuration: bytes are rank-independent, every target rank gets a full
    copy under its own suffix).
    """
    if len(target_ratios) != target_tp_size:
        raise ValueError(
            f"target ratio vector {list(target_ratios)} has "
            f"{len(target_ratios)} entries but target_tp_size={target_tp_size}"
        )
    plan: MigrationPlan = []
    for e in entries:
        if e.pool in skip_pools:
            continue
        base = strip_rank_suffix(e.suffix_rest, source_tp_rank, source_tp_size)
        if e.is_kv:
            for rank in range(target_tp_size):
                name = target_kv_name(e, base, dcp_owner_mode, rank, target_tp_size)
                plan.append((os.path.join(target_dir, name), [(e.path, 0, e.size)]))
                if dcp_owner_mode:
                    # One shared file for every rank; emitting it once is the
                    # whole point of the rank-less key.
                    break
            continue
        if e.pool == MAMBA_POOL:
            if mamba_spec is None:
                raise ValueError(
                    "the store holds mamba component blobs but no MambaBlobSpec "
                    "was given -- refusing to guess the GDN state layout"
                )
            if e.size != mamba_spec.total_bytes:
                raise ValueError(
                    f"mamba blob {os.path.basename(e.path)} is {e.size} B but the "
                    f"declared geometry implies {mamba_spec.total_bytes} B "
                    f"(layers={mamba_spec.num_layers}, heads={mamba_spec.num_heads}, "
                    f"head_dim={mamba_spec.head_dim}, state={mamba_spec.state_size}, "
                    f"conv_dim={mamba_spec.conv_dim}, width={mamba_spec.conv_width})"
                )
            for rank in range(target_tp_size):
                extents = [
                    (e.path, off, length)
                    for off, length in (
                        temporal_extents(mamba_spec, target_ratios, rank)
                        + conv_extents(mamba_spec, target_ratios, rank)
                    )
                ]
                name = target_mamba_name(e, base, rank, target_tp_size)
                plan.append((os.path.join(target_dir, name), extents))
            continue
        if e.pool == DRAFT_POOL:
            if draft_spec is None and not draft_key_rewrite:
                raise ValueError(
                    f"draft page {os.path.basename(e.path)} is in scope but "
                    "neither a DraftBlobSpec nor draft_key_rewrite was "
                    "declared -- refusing to guess the draft layout"
                )
            if draft_key_rewrite:
                # Declared attn_kv_replicated configuration: every rank holds
                # all draft kv-heads, so the bytes are rank-independent and
                # each target rank gets a full copy under its own suffix
                # (whole-file fan-out; verify_plan checks each copy).
                for rank in range(target_tp_size):
                    name = target_draft_name(e, base, rank, target_tp_size)
                    plan.append(
                        (os.path.join(target_dir, name), [(e.path, 0, e.size)])
                    )
                continue
            if e.size != draft_spec.total_bytes:
                raise ValueError(
                    f"draft blob {os.path.basename(e.path)} is {e.size} B but "
                    f"the declared draft geometry implies "
                    f"{draft_spec.total_bytes} B (layers={draft_spec.num_layers}, "
                    f"kv_heads={draft_spec.num_kv_heads}, "
                    f"head_dim={draft_spec.head_dim}, "
                    f"itemsize={draft_spec.itemsize})"
                )
            from sglang.srt.mem_cache.draft_migrate import draft_extents

            for rank in range(target_tp_size):
                extents = [
                    (e.path, off, length)
                    for off, length in draft_extents(draft_spec, target_ratios, rank)
                ]
                name = target_draft_name(e, base, rank, target_tp_size)
                plan.append((os.path.join(target_dir, name), extents))
            continue
        raise ValueError(
            f"unhandled component pool '{e.pool}' in {os.path.basename(e.path)}"
        )
    return plan


def plan_reverse_migration(
    entries: Sequence[StoreEntry],
    target_dir: str,
    *,
    source_tp_size: int,
    source_ratios: Sequence[int],
    target_tp_rank: int = 0,
    target_tp_size: int = 1,
    mamba_spec: Optional[MambaBlobSpec],
    dcp_owner_mode: bool = True,
    target_dcp_owner_mode: bool = False,
    skip_pools: Sequence[str] = (DRAFT_POOL,),
    draft_spec=None,
    draft_key_rewrite: bool = False,
) -> MigrationPlan:
    """Full byte-provenance plan of the N -> 1 handover (reassembly).

    ``mamba_spec`` is the FULL, unsharded geometry -- the same object the
    forward direction consumes. The per-rank source layouts are derived from it
    with ``shard_for_rank``, so the two directions cannot drift apart: a wrong
    rank blob size is a hard error naming both numbers, not a silent misparse.

    The returned plan has the same shape as ``plan_migration``'s and therefore
    passes through the same ``execute_plan`` / ``verify_plan``.
    """
    if len(source_ratios) != source_tp_size:
        raise ValueError(
            f"source ratio vector {list(source_ratios)} has "
            f"{len(source_ratios)} entries but source_tp_size={source_tp_size}"
        )
    if not dcp_owner_mode:
        raise ValueError(
            "reverse migration requires the source store to be in "
            "dcp_owner_mode: without it every rank wrote its OWN kv-head shard "
            "of a page, and rebuilding one full page means interleaving heads, "
            "not renaming files. That is a different operation and is not built."
        )
    shard_specs = (
        None
        if mamba_spec is None
        else [
            mamba_spec.shard_for_rank(source_ratios, r) for r in range(source_tp_size)
        ]
    )
    plan: MigrationPlan = []
    # {(key, base): {rank: entry}} -- component pools arrive as one file per
    # rank and only become a target once the set is complete.
    mamba_groups: Dict[Tuple[str, str], Dict[int, StoreEntry]] = {}
    draft_groups: Dict[Tuple[str, str], Dict[int, StoreEntry]] = {}
    for e in entries:
        if e.pool in skip_pools:
            continue
        if e.is_kv:
            # dcp_owner_mode KV keys carry no rank suffix at all, so the whole
            # remainder IS the base suffix.
            name = target_kv_name(
                e, e.suffix_rest, target_dcp_owner_mode, target_tp_rank, target_tp_size
            )
            plan.append((os.path.join(target_dir, name), [(e.path, 0, e.size)]))
            continue
        if e.pool == MAMBA_POOL:
            if mamba_spec is None:
                raise ValueError(
                    "the store holds mamba component blobs but no MambaBlobSpec "
                    "was given -- refusing to guess the GDN state layout"
                )
            rank = _rank_of(e, source_tp_size)
            base = strip_rank_suffix(e.suffix_rest, rank, source_tp_size)
            want = shard_specs[rank].total_bytes
            if e.size != want:
                raise ValueError(
                    f"mamba shard {os.path.basename(e.path)} is {e.size} B but "
                    f"rank {rank} of ratios {list(source_ratios)} implies {want} B "
                    f"(heads={shard_specs[rank].num_heads}, "
                    f"conv_dim={shard_specs[rank].conv_dim}) -- declared source "
                    "geometry does not match this store"
                )
            group = mamba_groups.setdefault((e.key, base), {})
            if rank in group:
                raise ValueError(
                    f"two files claim rank {rank} of {e.key}.mamba: "
                    f"{os.path.basename(group[rank].path)} and "
                    f"{os.path.basename(e.path)}"
                )
            group[rank] = e
            continue
        if e.pool == DRAFT_POOL:
            if draft_spec is None and not draft_key_rewrite:
                raise ValueError(
                    f"draft page {os.path.basename(e.path)} is in scope but "
                    "neither a DraftBlobSpec nor draft_key_rewrite was "
                    "declared -- refusing to guess the draft layout"
                )
            rank = _rank_of(e, source_tp_size)
            base = strip_rank_suffix(e.suffix_rest, rank, source_tp_size)
            group = draft_groups.setdefault((e.key, base), {})
            if rank in group:
                raise ValueError(
                    f"two files claim rank {rank} of {e.key}.draft: "
                    f"{os.path.basename(group[rank].path)} and "
                    f"{os.path.basename(e.path)}"
                )
            group[rank] = e
            continue
        raise ValueError(
            f"unhandled component pool '{e.pool}' in {os.path.basename(e.path)}"
        )

    layout = None if mamba_spec is None else reverse_extents(mamba_spec, source_ratios)
    for (key, base), group in mamba_groups.items():
        missing = [r for r in range(source_tp_size) if r not in group]
        if missing:
            raise ValueError(
                f"cannot reassemble {key}.mamba: rank blob(s) {missing} of "
                f"{source_tp_size} are absent. A partial state would be a "
                "plausible-looking but wrong GDN state, so this is fatal."
            )
        extents = [(group[rank].path, off, length) for rank, off, length in layout]
        name = target_mamba_name(group[0], base, target_tp_rank, target_tp_size)
        plan.append((os.path.join(target_dir, name), extents))

    if draft_groups:
        from sglang.srt.mem_cache.draft_migrate import draft_reverse_extents

        draft_shards = (
            None
            if draft_spec is None
            else [
                draft_spec.shard_for_rank(source_ratios, r)
                for r in range(source_tp_size)
            ]
        )
        draft_layout = (
            None
            if draft_spec is None
            else draft_reverse_extents(draft_spec, source_ratios)
        )
        for (key, base), group in draft_groups.items():
            missing = [r for r in range(source_tp_size) if r not in group]
            if missing:
                raise ValueError(
                    f"cannot reassemble {key}.draft: rank blob(s) {missing} of "
                    f"{source_tp_size} are absent. A partial draft state would "
                    "be a plausible-looking but wrong draft KV, so this is "
                    "fatal."
                )
            name = target_draft_name(group[0], base, target_tp_rank, target_tp_size)
            if draft_key_rewrite:
                # Replicated draft: every rank wrote the full blob; take rank
                # 0's bytes for the single target. Size equality across ranks
                # is the cheap declared-geometry check available here.
                sizes = {group[r].size for r in range(source_tp_size)}
                if len(sizes) != 1:
                    raise ValueError(
                        f"draft blobs of {key} differ in size across ranks "
                        f"({sorted(sizes)} B) -- they cannot be the declared "
                        "replicated configuration"
                    )
                plan.append(
                    (
                        os.path.join(target_dir, name),
                        [(group[0].path, 0, group[0].size)],
                    )
                )
                continue
            for rank in range(source_tp_size):
                want = draft_shards[rank].total_bytes
                if group[rank].size != want:
                    raise ValueError(
                        f"draft shard {os.path.basename(group[rank].path)} is "
                        f"{group[rank].size} B but rank {rank} of ratios "
                        f"{list(source_ratios)} implies {want} B "
                        f"(kv_heads={draft_shards[rank].num_kv_heads}) -- "
                        "declared draft geometry does not match this store"
                    )
            extents = [
                (group[rank].path, off, length)
                for rank, off, length in draft_layout
            ]
            plan.append((os.path.join(target_dir, name), extents))
    return plan


def _rank_of(entry: StoreEntry, tp_size: int) -> int:
    """Read the ``_{rank}_{tp_size}`` tail off a component key."""
    m = re.search(rf"_(\d+)_{tp_size}$", entry.suffix_rest)
    if m is None:
        raise ValueError(
            f"component entry '{os.path.basename(entry.path)}' has no "
            f"'_<rank>_{tp_size}' suffix -- declared source_tp_size={tp_size} "
            "does not match this store"
        )
    rank = int(m.group(1))
    if rank >= tp_size:
        raise ValueError(
            f"'{os.path.basename(entry.path)}' names rank {rank} of "
            f"tp_size {tp_size}"
        )
    return rank


def execute_plan(plan: MigrationPlan, *, chunk: int = 1 << 22) -> Dict[str, int]:
    """Materialize the plan. Pure byte copies; a single-extent whole-file entry
    is copied as a file so KV migration stays a cheap rename-equivalent."""
    os.makedirs(os.path.dirname(plan[0][0]), exist_ok=True) if plan else None
    stats = {"files": 0, "bytes": 0}
    for target, extents in plan:
        if len(extents) == 1 and extents[0][1] == 0:
            src, _, length = extents[0]
            if length == os.path.getsize(src):
                shutil.copyfile(src, target)
                stats["files"] += 1
                stats["bytes"] += length
                continue
        with open(target, "wb") as out:
            for src, off, length in extents:
                with open(src, "rb", buffering=0) as f:
                    f.seek(off)
                    left = length
                    while left:
                        buf = f.read(min(chunk, left))
                        if not buf:
                            raise IOError(
                                f"short read on {src} at {off} (+{length - left})"
                            )
                        out.write(buf)
                        left -= len(buf)
        stats["files"] += 1
        stats["bytes"] += sum(n for _, _, n in extents)
    return stats


def verify_plan(plan: MigrationPlan) -> Dict[str, int]:
    """Byte-permutation gate on the REAL files.

    Two properties, both checked against the bytes on disk, not against the
    plan that produced them:
    1. every target file equals the concatenation of its source extents --
       byte identity, no tolerance;
    2. per source file, the extents cover each byte AT MOST once, and a source
       that is consumed by several targets (the mamba split) is covered
       EXACTLY once in total -- nothing duplicated, nothing dropped.
    Raises on the first violation, naming file and byte offset.

    Direction-agnostic on purpose: it never looks at which geometry produced
    the plan, only at extents and bytes. One source cut across many targets
    (1 -> N) and many sources gathered into one target (N -> 1) are the same
    statement to it, which is what lets the round-trip gate reuse it twice.

    One named allowance: a source whose EVERY use in the plan is a whole-file
    extent used by several targets is a REPLICATION (rank-independent bytes
    fanned out under per-rank keys -- the declared ``attn_kv_replicated``
    draft configuration, or per-rank KV outside ``dcp_owner_mode``). Each
    copy is still byte-checked against the source in full; only the
    consumed-exactly-once rule is waived, because fan-out is not a
    permutation. Partial-extent double consumption stays fatal.
    """
    # Pre-classify replication fan-out: every use is (0, full file size) and
    # there is more than one use.
    uses: Dict[str, List[Tuple[int, int]]] = {}
    for _target, extents in plan:
        for src, off, length in extents:
            uses.setdefault(src, []).append((off, length))
    fanout_sources = {
        src
        for src, ranges in uses.items()
        if len(ranges) > 1
        and all(off == 0 and length == os.path.getsize(src) for off, length in ranges)
    }
    per_source: Dict[str, bytearray] = {}
    checked = {"targets": 0, "bytes": 0, "sources": 0}
    for target, extents in plan:
        with open(target, "rb") as f:
            got = f.read()
        if len(got) != sum(n for _, _, n in extents):
            raise ValueError(
                f"{os.path.basename(target)}: {len(got)} B on disk, plan says "
                f"{sum(n for _, _, n in extents)} B"
            )
        pos = 0
        for src, off, length in extents:
            with open(src, "rb", buffering=0) as f:
                f.seek(off)
                want = f.read(length)
            if got[pos : pos + length] != want:
                raise ValueError(
                    f"{os.path.basename(target)} differs from "
                    f"{os.path.basename(src)}[{off}:{off + length}]"
                )
            if src in fanout_sources:
                # Replication: byte identity per copy is checked above; the
                # exactly-once rule does not apply to fan-out.
                pos += length
                continue
            seen = per_source.setdefault(src, bytearray(os.path.getsize(src)))
            # Slice-wise rather than byte-wise: the reverse direction plans
            # hundreds of extents over blobs of tens of MiB, and a Python-level
            # per-byte loop turns the gate into the slowest part of the run.
            window = seen[off : off + length]
            if len(window) != length:
                raise ValueError(
                    f"extent {os.path.basename(src)}[{off}:{off + length}] runs "
                    f"past the end of the file ({os.path.getsize(src)} B)"
                )
            hit = window.find(1)
            if hit >= 0:
                raise ValueError(
                    f"source {os.path.basename(src)} byte {off + hit} used twice"
                )
            seen[off : off + length] = b"\x01" * length
            pos += length
        checked["targets"] += 1
        checked["bytes"] += len(got)
    for src, seen in per_source.items():
        # KV pages are copied once as a whole; mamba blobs are cut across
        # ranks. Either way the source must end up fully covered.
        covered = seen.count(1)
        if covered != len(seen):
            raise ValueError(
                f"source {os.path.basename(src)}: {len(seen) - covered} B "
                "never migrated"
            )
    checked["sources"] = len(per_source) + len(fanout_sources)
    return checked


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(
        prog="python -m sglang.srt.mem_cache.hicache_migrate",
        description="Migrate a HiCache 'file' store between TP geometries: "
        "TP=1 -> TP=N (uneven, weighted-DCP) by default, TP=N -> TP=1 with "
        "--reverse.",
    )
    p.add_argument("--source-dir", required=True)
    p.add_argument("--target-dir", required=True)
    p.add_argument(
        "--reverse",
        action="store_true",
        help="hand the group's context BACK to a single card: reassemble the "
        "N per-rank GDN shards into one full state and re-suffix the KV pages",
    )
    p.add_argument("--source-tp-size", type=int, default=1)
    p.add_argument("--source-tp-rank", type=int, default=0)
    p.add_argument(
        "--source-ratios",
        help="--reverse only: the resolved --rank-tp-ratio vector of the boot "
        "that WROTE the store, e.g. 6,1,1",
    )
    p.add_argument("--target-tp-size", type=int, default=1)
    p.add_argument("--target-tp-rank", type=int, default=0)
    p.add_argument(
        "--target-ratios",
        help="forward only: the consuming boot's resolved --rank-tp-ratio "
        "vector, e.g. 6,1,1",
    )
    p.add_argument(
        "--model-config",
        help="path to the model's config.json (for the GDN state layout)",
    )
    p.add_argument("--num-linear-layers", type=int)
    p.add_argument(
        "--gdn-units",
        type=int,
        help="the model's gdn_tp_units (COARSENED for GGUF K-quant)",
    )
    p.add_argument("--temporal-itemsize", type=int, default=2)
    p.add_argument("--conv-itemsize", type=int, default=2)
    p.add_argument(
        "--no-dcp-owner-mode",
        action="store_true",
        help="target boot does NOT run weighted uneven DCP (KV keys stay per-rank)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--verify",
        action="store_true",
        help="after writing, re-read every target file and prove byte identity "
        "with its source extents (permutation gate)",
    )
    p.add_argument(
        "--manifest",
        help="session-handover manifest (managers/session_handover): restrict "
        "the migration to exactly the manifest's blobs. REQUIRED for a live "
        "source store -- everything outside the manifest, including files "
        "other sessions are writing right now, is ignored",
    )
    p.add_argument(
        "--draft-spec-algorithm",
        help="canonical speculative algorithm name whose draft state the "
        "store carries (same names as --speculative-algorithm). Unknown "
        "names are refused with the known list; algorithms without a "
        "declared re-shard capability are refused with the reason",
    )
    p.add_argument(
        "--draft-num-layers",
        type=int,
        help="the DRAFT model's layer count (independent of the target's)",
    )
    p.add_argument("--draft-kv-heads", type=int, help="total draft kv-head count")
    p.add_argument("--draft-head-dim", type=int)
    p.add_argument("--draft-itemsize", type=int, default=2)
    p.add_argument(
        "--draft-mem-layout",
        default="layer_first",
        help="the draft pool's hicache_mem_layout (page_head is refused: its "
        "extent family is not built)",
    )
    p.add_argument(
        "--draft-units",
        type=int,
        default=None,
        help="indivisible unit count for the draft head split; default is "
        "one unit per kv-head (head granularity)",
    )
    p.add_argument(
        "--draft-kv-replicated",
        action="store_true",
        help="declared attn_kv_replicated draft configuration (TP exceeds the "
        "draft's kv-head count): bytes are rank-independent, migration is a "
        "per-rank key rewrite, no split. Not visible in the store, so it "
        "must be declared",
    )
    p.add_argument(
        "--draft-cold-start",
        action="store_true",
        help="explicitly accept that the draft pool starts cold on the "
        "destination. In --manifest mode this (or --draft-spec-algorithm) is "
        "MANDATORY when the manifest names draft blobs: silent skipping is "
        "not available on the handover path",
    )
    a = p.parse_args(argv)

    if a.reverse:
        if not a.source_ratios:
            p.error("--reverse needs --source-ratios (the writing boot's vector)")
        if a.source_tp_size <= 1:
            p.error("--reverse needs --source-tp-size > 1")
        ratios = [int(x) for x in a.source_ratios.split(",") if x.strip()]
    else:
        if not a.target_ratios:
            p.error("--target-ratios is required (the consuming boot's vector)")
        ratios = [int(x) for x in a.target_ratios.split(",") if x.strip()]
    spec = None
    if a.model_config:
        cfg = json.load(open(a.model_config))
        text_cfg = cfg.get("text_config", cfg)
        if a.num_linear_layers is None or a.gdn_units is None:
            p.error("--model-config needs --num-linear-layers and --gdn-units")
        spec = qwen3_5_mamba_spec(
            text_cfg,
            num_linear_layers=a.num_linear_layers,
            units=a.gdn_units,
            temporal_itemsize=a.temporal_itemsize,
            conv_itemsize=a.conv_itemsize,
        )

    entries = scan_store(a.source_dir)
    manifest = None
    if a.manifest:
        manifest = load_manifest(a.manifest)
        entries = filter_entries_by_manifest(entries, manifest)

    # Draft disposition (#261 second half). Default outside --manifest mode
    # stays the documented skip; in --manifest mode a draft blob REQUIRES an
    # explicit disposition -- never a silent skip on the handover path.
    from sglang.srt.mem_cache.draft_migrate import (
        DraftBlobSpec,
        DraftReshardCapability,
        DraftReshardError,
        resolve_draft_reshard,
    )

    draft_entries = [e for e in entries if e.pool == DRAFT_POOL]
    draft_spec = None
    draft_key_rewrite = False
    skip_pools: Tuple[str, ...] = (DRAFT_POOL,)
    draft_disposition = "skipped (legacy default)"
    if a.draft_spec_algorithm and a.draft_cold_start:
        p.error("--draft-spec-algorithm and --draft-cold-start are mutually exclusive")
    if a.draft_spec_algorithm:
        try:
            verdict = resolve_draft_reshard(a.draft_spec_algorithm)
        except DraftReshardError as err:
            p.error(str(err))
        if verdict.capability is DraftReshardCapability.REFUSE:
            p.error(
                f"draft re-shard refused for {verdict.algorithm}: {verdict.reason}"
            )
        if verdict.capability is DraftReshardCapability.NO_DRAFT_KV:
            if draft_entries:
                p.error(
                    f"{verdict.algorithm} declares no draft KV state "
                    f"({verdict.reason}), but the store holds "
                    f"{len(draft_entries)} .draft blob(s) -- the declaration "
                    "contradicts the store; refusing to guess which is wrong"
                )
            draft_disposition = f"none ({verdict.algorithm} has no draft KV)"
        elif a.draft_kv_replicated:
            skip_pools = ()
            draft_key_rewrite = True
            draft_disposition = (
                f"key rewrite ({verdict.algorithm}, declared attn_kv_replicated)"
            )
        else:
            if (
                a.draft_num_layers is None
                or a.draft_kv_heads is None
                or a.draft_head_dim is None
            ):
                p.error(
                    "--draft-spec-algorithm needs --draft-num-layers, "
                    "--draft-kv-heads and --draft-head-dim (or "
                    "--draft-kv-replicated for the replicated configuration)"
                )
            try:
                draft_spec = DraftBlobSpec(
                    num_layers=a.draft_num_layers,
                    num_kv_heads=a.draft_kv_heads,
                    head_dim=a.draft_head_dim,
                    itemsize=a.draft_itemsize,
                    mem_layout=a.draft_mem_layout,
                    units=a.draft_units,
                )
            except DraftReshardError as err:
                p.error(str(err))
            skip_pools = ()
            draft_disposition = f"re-shard ({verdict.algorithm})"
    elif a.draft_cold_start:
        draft_disposition = "cold start (declared)"
    elif manifest is not None and draft_entries:
        p.error(
            f"the manifest names {len(draft_entries)} draft blob(s); declare "
            "a disposition: --draft-spec-algorithm NAME (re-shard, or refusal "
            "with the algorithm's reason) or --draft-cold-start (draft pool "
            "starts cold on the destination). Silent skipping is not "
            "available on the handover path."
        )

    os.makedirs(a.target_dir, exist_ok=True)
    if a.reverse:
        plan = plan_reverse_migration(
            entries,
            a.target_dir,
            source_tp_size=a.source_tp_size,
            source_ratios=ratios,
            target_tp_rank=a.target_tp_rank,
            target_tp_size=a.target_tp_size,
            mamba_spec=spec,
            dcp_owner_mode=not a.no_dcp_owner_mode,
            skip_pools=skip_pools,
            draft_spec=draft_spec,
            draft_key_rewrite=draft_key_rewrite,
        )
    else:
        plan = plan_migration(
            entries,
            a.target_dir,
            source_tp_rank=a.source_tp_rank,
            source_tp_size=a.source_tp_size,
            target_tp_size=a.target_tp_size,
            target_ratios=ratios,
            mamba_spec=spec,
            dcp_owner_mode=not a.no_dcp_owner_mode,
            skip_pools=skip_pools,
            draft_spec=draft_spec,
            draft_key_rewrite=draft_key_rewrite,
        )
    kv = sum(1 for e in entries if e.is_kv)
    mamba = sum(1 for e in entries if e.pool == MAMBA_POOL)
    skipped = len(entries) - kv - mamba
    direction = (
        f"TP={a.source_tp_size} -> TP={a.target_tp_size} (reassembly)"
        if a.reverse
        else f"TP={a.source_tp_size} -> TP={a.target_tp_size} (split)"
    )
    print(
        f"source {a.source_dir} [{direction}]: {kv} KV pages, "
        f"{mamba} mamba blobs, {len(draft_entries)} draft blobs "
        f"[{draft_disposition}], {skipped - len(draft_entries)} other "
        f"skipped -> {len(plan)} target files"
        + (f" (manifest-scoped: {a.manifest})" if manifest is not None else "")
    )
    if spec is not None:
        side = "source" if a.reverse else "target"
        for rank, (heads, conv) in enumerate(shard_sizes(spec, ratios)):
            print(
                f"  {side} rank {rank}: temporal heads {heads}/{spec.num_heads}, "
                f"conv channels {conv}/{spec.conv_dim}"
            )
    if a.dry_run:
        return 0
    stats = execute_plan(plan)
    print(f"wrote {stats['files']} files, {stats['bytes'] / (1 << 20):.1f} MiB")
    if a.verify:
        v = verify_plan(plan)
        print(
            f"permutation gate PASSED: {v['targets']} target files "
            f"({v['bytes'] / (1 << 20):.1f} MiB) are byte-identical to "
            f"{v['sources']} fully covered source files"
        )
    return 0


def qwen3_5_mamba_spec(
    text_config: Dict,
    *,
    num_linear_layers: int,
    units: int,
    temporal_itemsize: int,
    conv_itemsize: int,
) -> MambaBlobSpec:
    """MambaBlobSpec for a Qwen3.5/3.6 GDN text config.

    ``units`` is the model's ``gdn_tp_units`` -- for a GGUF K-quant checkpoint
    that is the COARSENED unit count (``_quant_block_aligned_units``), not the
    key-head count. Passing the wrong one produces a plausible but wrong split,
    so it is an explicit argument rather than a re-derivation.
    """
    num_heads = int(text_config["linear_num_value_heads"])
    head_dim = int(text_config["linear_value_head_dim"])
    state_size = int(text_config["linear_key_head_dim"])
    n_groups = int(text_config["linear_num_key_heads"])
    value_dim = num_heads * head_dim
    key_dim = n_groups * state_size
    return MambaBlobSpec(
        num_layers=num_linear_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        state_size=state_size,
        conv_dim=key_dim * 2 + value_dim,
        conv_width=int(text_config["linear_conv_kernel_dim"]) - 1,
        key_dim=key_dim,
        value_dim=value_dim,
        units=units,
        temporal_itemsize=temporal_itemsize,
        conv_itemsize=conv_itemsize,
    )


if __name__ == "__main__":
    raise SystemExit(_cli())
