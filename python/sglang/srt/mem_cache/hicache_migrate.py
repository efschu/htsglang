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
* Draft (``{hash}.draft``) pages are NOT migrated. They stay rank-suffixed
  even in ``dcp_owner_mode``, so re-owning them needs the consuming boot's DCP
  token vector. Run the handover without speculative decoding, or accept that
  the draft pool starts cold.
* Source geometry is a single rank (TP=1). A general N->M reshard would have
  to reassemble the full state from several source shards first; that is a
  strictly larger operation and is not built here.
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
) -> MigrationPlan:
    """Full byte-provenance plan of the migration.

    Every entry is ``(target path, [(source path, offset, length), ...])`` in
    target order -- the concatenation of the extents IS the target file. No
    step computes a value; the plan is a permutation of source bytes into
    target files, which is exactly what the unit proof checks.
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
        raise ValueError(
            f"unhandled component pool '{e.pool}' in {os.path.basename(e.path)}"
        )
    return plan


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
    """
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
            seen = per_source.setdefault(src, bytearray(os.path.getsize(src)))
            for b in range(off, off + length):
                if seen[b]:
                    raise ValueError(
                        f"source {os.path.basename(src)} byte {b} used twice"
                    )
                seen[b] = 1
            pos += length
        checked["targets"] += 1
        checked["bytes"] += len(got)
    for src, seen in per_source.items():
        # KV pages are copied once as a whole; mamba blobs are cut across
        # ranks. Either way the source must end up fully covered.
        if sum(seen) != len(seen):
            raise ValueError(
                f"source {os.path.basename(src)}: {len(seen) - sum(seen)} B "
                "never migrated"
            )
    checked["sources"] = len(per_source)
    return checked


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(
        prog="python -m sglang.srt.mem_cache.hicache_migrate",
        description="Migrate a HiCache 'file' store from a TP=1 geometry to a "
        "TP=N (uneven, weighted-DCP) geometry.",
    )
    p.add_argument("--source-dir", required=True)
    p.add_argument("--target-dir", required=True)
    p.add_argument("--source-tp-size", type=int, default=1)
    p.add_argument("--source-tp-rank", type=int, default=0)
    p.add_argument("--target-tp-size", type=int, required=True)
    p.add_argument(
        "--target-ratios",
        required=True,
        help="the consuming boot's resolved --rank-tp-ratio vector, e.g. 6,1,1",
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
    a = p.parse_args(argv)

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
    os.makedirs(a.target_dir, exist_ok=True)
    plan = plan_migration(
        entries,
        a.target_dir,
        source_tp_rank=a.source_tp_rank,
        source_tp_size=a.source_tp_size,
        target_tp_size=a.target_tp_size,
        target_ratios=ratios,
        mamba_spec=spec,
        dcp_owner_mode=not a.no_dcp_owner_mode,
    )
    kv = sum(1 for e in entries if e.is_kv)
    mamba = sum(1 for e in entries if e.pool == MAMBA_POOL)
    skipped = len(entries) - kv - mamba
    print(
        f"source {a.source_dir}: {kv} KV pages, {mamba} mamba blobs, "
        f"{skipped} skipped (draft/other) -> {len(plan)} target files"
    )
    if spec is not None:
        for rank, (heads, conv) in enumerate(shard_sizes(spec, ratios)):
            print(
                f"  rank {rank}: temporal heads {heads}/{spec.num_heads}, "
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
