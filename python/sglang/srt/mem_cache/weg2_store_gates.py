"""Weg 2 S5: the named refusals that guard the store as the SOLE carrier.

Weg 2 runs two independent process groups on the same cards -- group P
(``pp_size=3, tp_size=1``, prefill) and group D (``tp_size=3, pp_size=1``,
decode) -- and NOTHING crosses at a flip except what P already wrote into the
L3 canonical page store. That promotes a set of conditions from "degrades the
hit rate" to "delivers nothing at all, silently":

* the two groups' KV keys must name one page (W9, W10),
* a hybrid's recurrent anchor must travel with it (W7): ``batch_exists_v2``
  takes the MINIMUM across pools and the mamba pool is registered
  TRAILING_PAGES, so a missing GDN blob truncates the whole KV prefix to zero,
* the store must be bounded by a number the filesystem can actually fund (W8),
* and the bound must be enforced against the store that exists, not the
  fraction of it the eviction index happens to see (W8b).

Every one of these is invisible at runtime: the reader misses, the miss looks
like a cold cache, and the campaign reads as a capacity problem. So each is a
LAUNCH-TIME refusal with the arithmetic in its message, not a warning.

ONE DEFINITION, TWO CALLERS BY DESIGN. The rank-side attach calls these at the
moment the number exists; the S3 launcher calls the same functions over the two
groups' reported values. A second copy of any threshold here would be exactly
the fork-owned twin bookkeeping the upstream-minimal law refuses.
"""

from __future__ import annotations

import os
from typing import Mapping

# Spec 3.3 (4): the eviction owner must index at least this fraction of the
# directory's bytes, or the cap it enforces is a fiction. Measured on the store
# of record: 104,267 of 124,610 files (83.7 %) ended with no rank's
# ``config_suffix`` and were invisible to the LRU index.
INDEX_COVERAGE_FLOOR = 0.5


class Weg2StoreGateError(RuntimeError):
    """Base of the S5 launch refusals. Never caught to continue."""


class Weg2MambaBlobAbsent(Weg2StoreGateError):
    """W7: a hybrid rank has a canonical KV page but no canonical GDN blob."""


class Weg2StoreCapUnfundable(Weg2StoreGateError):
    """W8: the store filesystem cannot hold max_size + min_free_space."""


class Weg2StoreIndexBlind(Weg2StoreGateError):
    """W8b: the eviction owner indexes less than half the directory."""


class Weg2StoreIdentityMismatch(Weg2StoreGateError):
    """W9: the two groups computed different store identity hashes."""


class Weg2CanonicalPageMissing(Weg2StoreGateError):
    """W10: a group launched without the canonical page format."""


def writes_shared_keys(
    *,
    is_mla_model: bool,
    dcp_owner_mode: bool = False,
    canonical_kv_page: object = None,
) -> bool:
    """Do the ranks of this group write the SAME physical files? (F7)

    ``is_mla_model`` was the tree's proxy for this question, and it was a
    correct proxy exactly while MLA was the only way ranks could share a file.
    Two later formats share files too:

    * weighted uneven-DCP owner mode -- a KV page's bytes are complete on its
      owner rank, so one file is written once and read by everyone;
    * the #706 canonical page -- the stored bytes stop depending on the cut, so
      a GQA model's ranks deposit into ONE page under ONE key.

    Under Weg 2 the canonical page is always on, so keeping the proxy would
    make all six ranks eviction owners over one directory (six private LRU
    indices racing on one budget) and would divide the operator's single cap by
    ``tp_size``, which is 1 in group P and 3 in group D -- two different caps
    over the same files.

    THE QUESTION THIS ANSWERS IS PHYSICAL, not architectural: it is about file
    identity, never about attention shape. Anything that makes two ranks write
    one path belongs in this predicate and nowhere else.
    """
    return bool(is_mla_model) or bool(dcp_owner_mode) or canonical_kv_page is not None


def owner_write_covers_whole_file(
    *,
    is_mla_model: bool,
    canonical_extent_write: bool = False,
    path_is_group_wide: bool = True,
) -> bool:
    """Would the storage owner's write alone put EVERY byte of this file on disk?

    THE SECOND QUESTION, kept apart from ``writes_shared_keys`` on purpose.
    That predicate answers file IDENTITY -- do two ranks name one path. This
    one answers file COMPLETENESS -- is one rank's write the whole content.
    They coincide only under MLA, and conflating them is what made a
    correct-looking eviction election refuse writes:

    * ``writes_shared_keys`` became true for a GQA model under the #706
      canonical page and under dcp owner mode, so ranks 1..n-1 stopped being
      storage owners -- correct, one LRU index per directory;
    * but ``reserve()`` read "not the owner" as "has nothing to contribute",
      and those ranks hold bytes NOBODY else writes: their extent of the
      canonical blob (a PP stage holds its layers, a TP rank its head
      channels), their dcp-owned pages, their own suffixed draft files. On
      Weg 2's store, where the cap is always configured, refusing them is a
      carrier that moves nothing and looks exactly like a cold cache.

    Under MLA the suffix drops the TP terms and ONLY those:
    ``_derive_key_suffixes`` appends ``_{tp_rank}_{tp_size}`` under
    ``if not is_mla_model``, but appends ``_{pp_size}_{pp_rank}`` and
    ``_cp{attn_cp_rank}_{attn_cp_size}`` two branches below with no such
    guard. "MLA" therefore does NOT mean "every rank names one path", and
    reading it that way was a measured regression: at 8a71eb87, hermetically,
    an MLA run with pp_size=3 gave stage 1 the suffix ``_M_H_3_1`` -- a path
    the elected owner (tp0/pp0/cp0) never writes -- and stages 1 and 2 were
    refused every write, as was attn_cp rank 1. The parent aef3ae76 admitted
    all of them, so this was a NEW refusal on a shipping non-Weg-2 path, under
    a log line that asserted the false premise ("the owner writes this whole
    file").

    The caller therefore supplies the FACT rather than a list of shapes:
    ``path_is_group_wide`` is True only when the suffix this key carries is
    the string every rank of the group produces. ``HiCacheStorage`` answers it
    by re-deriving its suffixes with the rank terms zeroed and comparing --
    one derivation, two inputs -- so a rank axis added later is covered the
    day it is added rather than the day someone remembers to extend a list
    here. Where the path is this rank's alone, nobody else writes those bytes
    and refusing is a carrier that moves nothing.

    It defaults to True so a caller that does not know keeps the historical
    refusal, which is the safe direction: a refused duplicate is lost cache,
    while an admitted write the owner cannot account for is unbounded disk.

    A canonical EXTENT write is never whole by construction -- the format
    exists so that several ranks each deposit a part and the blob becomes
    readable only when the last byte lands -- so it answers False whatever the
    attention shape.
    """
    if canonical_extent_write:
        return False
    if not path_is_group_wide:
        return False
    return bool(is_mla_model)


def check_mamba_blob_present(
    *,
    canonical_page_on: bool,
    has_mamba_pool: bool,
    mamba_blob: object,
) -> None:
    """W7. Refuse a canonical KV page on a hybrid whose GDN blob is absent.

    ``resolve_linear_layer_ids`` returns an empty list ONLY on a positive dense
    proof, so ``_canonical_mamba_window`` returning None is meant to mean "this
    model has no linear layers". The second witness is the runtime itself: a
    bound mamba pool. When the two disagree -- no blob, but a mamba pool
    exists -- the resolver was wrong about the model, and running on is the
    measured #931 failure: "#706 canonical KV page active" x3 and "#706
    canonical GDN blob active" x0, with zero refusals and every cross-group
    lookup missing.
    """
    if not canonical_page_on or not has_mamba_pool or mamba_blob is not None:
        return
    raise Weg2MambaBlobAbsent(
        "W7 Weg2MambaBlobAbsent: this rank installed a canonical KV page but "
        "no canonical GDN blob, while a mamba/GDN pool IS bound -- so the "
        "model is a hybrid and the linear-layer resolver claimed it was "
        "dense. A KV-only canonical page delivers ZERO usable prefix: "
        "batch_exists_v2 takes the MINIMUM across pools and the mamba pool is "
        "registered TRAILING_PAGES, so the missing blob truncates the whole KV "
        "prefix to zero. Under Weg 2 the store is the SOLE carrier between the "
        "prefill and the decode group, so this is not a degraded hit rate -- "
        "it is a carrier that moves nothing, and it would look exactly like a "
        "cold cache. Refusing the launch instead."
    )


def check_canonical_page_enabled(group_flags: Mapping[str, bool]) -> None:
    """W10, the launcher's half: BOTH groups carry the format, or neither runs.

    One group with the canonical page and one without is the 100 % miss with
    extra steps: the group without it re-appends its own geometry to the KV
    key, so the two key spaces are disjoint and nothing ever raises.
    """
    missing = sorted(name for name, on in group_flags.items() if not on)
    if not missing:
        return
    raise Weg2CanonicalPageMissing(
        "W10 Weg2CanonicalPageMissing: group(s) "
        f"{', '.join(missing)} launched WITHOUT --hicache-canonical-kv-page "
        f"(of {len(group_flags)} groups: "
        f"{ {name: bool(on) for name, on in group_flags.items()} }). "
        "The store is Weg 2's only carrier across a flip, and a group without "
        "the geometry-neutral format re-appends _{tp_rank}_{tp_size} and "
        "_{pp_size}_{pp_rank} to every KV key. The two key spaces would be "
        "disjoint and every cross-group read would miss without raising."
    )


def check_identity_hashes(group_hashes: Mapping[str, str]) -> str:
    """W9. The groups' store identity hashes must be byte-identical.

    The hash covers model_path, revision, dtype, quantization and
    kv_cache_dtype -- the terms that describe the stored BYTES. Under the
    canonical page the parallel tail is dropped (``canonical_identity_hash_for``
    -> ``include_parallel_vectors=False``), so two groups that agree on the
    byte format agree on the hash whatever their cut. A difference therefore
    means the two launches disagree about the model or the KV byte format, and
    the consequence is a permanent, silent 100 % miss on the sole carrier.
    """
    distinct = sorted(set(group_hashes.values()))
    if len(distinct) <= 1:
        return distinct[0] if distinct else ""
    raise Weg2StoreIdentityMismatch(
        "W9 Weg2StoreIdentityMismatch: the groups computed "
        f"{len(distinct)} different store identity hashes over "
        f"{len(group_hashes)} groups: "
        f"{ {name: h for name, h in sorted(group_hashes.items())} }. "
        "The hash covers model_path, revision, dtype, quantization and "
        "kv_cache_dtype -- a difference means the two launches do not describe "
        "the same bytes, and every key one group writes is a key the other "
        "never asks for. Refusing to serve: the alternative is a permanent "
        "silent 100 % miss on the only carrier between the groups."
    )


def check_store_cap_fundable(
    store_path: str,
    max_size_bytes: int,
    min_free_bytes: int,
) -> None:
    """W8. The filesystem must be able to hold cap + watermark (Q5).

    The tree's own answer to an over-large cap is ``_clamp_max_size_to_fs``, a
    SILENT clamp. That was right while this store was a cache in front of a
    recomputable prefix: a smaller cache is slower, not wrong. Under Weg 2 the
    store is the retention tier AND the carrier, so a silently smaller cap is a
    capacity regression that shows up as a hit-rate decay weeks later, with the
    operator's own number still in the launch line. Q5 supplies the numbers
    (200 GiB cap, 100 GiB min free on /spinning/hicache-weg2); this refuses
    rather than quietly serving a different budget.

    Total capacity is the denominator, not free space: free space moves, and a
    cap that does not fit the DEVICE can never be honoured.
    """
    if max_size_bytes <= 0 and min_free_bytes <= 0:
        return
    try:
        st = os.statvfs(store_path)
    except (OSError, AttributeError):
        # No filesystem answer -- this gate has no denominator, so it refuses
        # nothing. The evictor's own watermark still applies.
        return
    total = st.f_blocks * (st.f_frsize or st.f_bsize or 4096)
    need = max(0, int(max_size_bytes)) + max(0, int(min_free_bytes))
    if need <= total:
        return
    raise Weg2StoreCapUnfundable(
        f"W8 Weg2StoreCapUnfundable: {store_path!r} has {total} B of total "
        f"capacity, but the store was configured for max_size="
        f"{max_size_bytes} B plus min_free_space={min_free_bytes} B = "
        f"{need} B ({need - total} B short). The two together are the whole "
        "budget: the cap is what the store may hold and the watermark is what "
        "it must leave behind, so a filesystem that cannot fund both can never "
        "honour the configured retention. Refusing the launch rather than "
        "clamping the cap, because under Weg 2 this store is the only carrier "
        "between the prefill and the decode group and a silently smaller cap "
        "reads as a cold cache, never as a misconfiguration."
    )


def check_index_coverage(
    *,
    store_path: str,
    indexed_bytes: int,
    seen_bytes: int,
    indexed_entries: int,
    seen_entries: int,
    floor: float = INDEX_COVERAGE_FLOOR,
) -> float:
    """W8b. The eviction index must see most of the directory it bounds.

    MEASURED DEFECT this exists for (spec 3.3 (4)): ``_scan_existing_files``
    filtered on ``config_suffix``, the suffix that still carries
    ``_{tp_rank}_{tp_size}_{pp_size}_{pp_rank}``, while the canonical KV pages
    and GDN blobs end with ``kv_config_suffix`` instead. On the store of record
    that made 104,267 of 124,610 files (83.7 %) invisible to the LRU, so
    ``max_size`` could never bound the KV pages -- only ``min_free_space``, a
    statvfs fact, could. Latent under Weg 1; load-bearing under Weg 2, where
    this store is the sole carrier.

    Returns the coverage fraction and always reports BOTH denominators, so a
    number from this gate can never be read without the population it came
    from.
    """
    fraction = (indexed_bytes / seen_bytes) if seen_bytes > 0 else 1.0
    if fraction >= floor:
        return fraction
    raise Weg2StoreIndexBlind(
        f"W8b Weg2StoreIndexBlind: the eviction owner of {store_path!r} "
        f"indexed {indexed_bytes} of {seen_bytes} B "
        f"({fraction:.1%}, floor {floor:.0%}) across "
        f"{indexed_entries} of {seen_entries} files. The byte cap is enforced "
        "against the indexed set alone, so the rest of the store is outside "
        "every bound this process can apply; only the free-space watermark "
        "would still act. The known cause is a scan filter that matches the "
        "per-rank config_suffix while the canonical pages carry the "
        "kv_config_suffix. Refusing the launch: a cap over a fraction of the "
        "store is not a cap."
    )
