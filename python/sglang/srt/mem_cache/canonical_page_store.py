"""#706: the PERSISTED canonical page -- partial writes, completeness, read cut.

``canonical_kv_page`` defines the page (Option A: every attention layer of one
token, layer-major, ``page_size == 1``). This module is the storage protocol
built on it: how three PP stages, each holding only its own layers, deposit
their slots into ONE suffix-free page file, and how any later geometry reads
its own slice back out of it.

Why this exists at all. Today a KV page key carries ``_{pp_size}_{pp_rank}``
(``hicache_storage.py``) and it carries it HONESTLY: under PP a stage's page
holds only that stage's layers, so the bytes really are stage-specific. The key
rule is not "keys always carry geometry" but "**the key carries exactly the
geometry the bytes still depend on**". Make the page complete across stages and
the suffix has nothing left to name -- which is the same argument
``dcp_owner_mode`` already used to drop the tp suffix on the token axis.

The protocol, in one paragraph. A blob file is full width from the moment it is
created (sparse until filled). A writer takes an exclusive lock on the partial
file, ``pwrite``s its own extents at their GLOBAL byte offsets, and records
those ranges in a small sidecar marker. The writer that covers the last byte
fsyncs the data and ``os.replace``s the blob to its final ``.bin`` name,
deleting the marker. Readers only ever open ``.bin``: an incomplete blob is not
"readable with holes", it is INVISIBLE, so a half-written one can never be
served. That is the "completeness marker + rename-on-complete" protocol
``DESIGN_706_mechanism.md`` says must be built, and both halves are load
bearing -- the marker is how a writer knows the blob is done, the rename is how
a reader is prevented from finding out too early.

It carries two pools. The KV page (all 16 attention layers of one token) is the
one-extent case. The GDN/mamba blob is the reason the protocol is expressed in
extents at all: its layer cut is two disjoint ranges and its head cut is three
sub-block ranges per layer. Both matter, because a KV-only prefix is worth
nothing on a hybrid checkpoint -- ``batch_exists_v2`` takes the MINIMUM across
pools, so a missing GDN blob truncates the whole KV prefix to zero
(``test_mamba_gates_the_hit_706.py``).

Completeness is tracked in BYTES rather than slots for the same reason. A KV
layer is written by exactly one writer (kv-heads are replicated, tokens are
sharded), but under TP one GDN layer's bytes arrive from several ranks, so a
layer-granular marker would call a layer complete when a third of its channels
were present.

Two traps this file is shaped around:

**The rank-local index trap.** A slot is addressed by the GLOBAL attention-layer
index. A stage that addresses slots with its rank-LOCAL layer ordering writes
the right NUMBER of bytes into the WRONG slots, silently -- the same failure
shape as cutting a mamba blob as one flat range. ``window_for_layers`` takes the
model's own attention-layer id list and the stage's GLOBAL layer ids, so the
index is global by construction. And when the trap is planted anyway, the
marker converts it from corruption into a permanent MISS: rank-local windows
from different stages overlap, the high slots are never written, the page never
completes, and nothing ever reads it.

**Silent partial pages.** An unwritten slot reads exactly like a legitimately
zero one. Nothing in the byte stream distinguishes them, which is why
completeness is tracked outside the bytes rather than inferred from them.

The marker deliberately lives in a SIDECAR rather than in a page header. A page
file must stay pure bytes: ``HiCacheFile.get`` reads a whole file into an
exactly-sized tensor, the LRU evictor accounts raw file bytes, and
``hicache_migrate`` parses the store as bare pages. A header would corrupt all
three at once, and quietly.
PINNED HOST MEMORY: this store allocates NONE, so it registers nothing with
the process-wide pinned budget (``pinned_host_budget.check_and_register_pinned_post``,
whose consumers are the kvso host pool, the HiCache host pool and the
phase-flip weight images). Both directions take CALLER-OWNED buffers --
``write_extents`` reads the payload it is handed, ``read_extents`` fills the
target it is handed -- and the file I/O is ``os.pwrite`` / ``os.pread`` against
those. There is no steady-state buffer of the store's own to account for, which
is why "register at attach/resize" does not apply here rather than having been
overlooked.

One transient allocation on this path is worth naming for whoever owns the
budget question, because it is NOT the store's and is NOT registered: the
backend obtains its read target from ``host_pool.get_dummy_flat_data_page()``,
which allocates a fresh PINNED tensor per read on both pool families
(``pin_memory=self.pin_memory``). It predates this work and scales with
concurrent reads rather than with tier size, so it is a per-operation spike the
joint budget cannot see, not a pool the budget is missing.
"""

from __future__ import annotations

import dataclasses
import errno
import fcntl
import logging
import os
import struct
import time
from collections.abc import Sequence
from typing import Callable, Optional

import torch

from sglang.srt.mem_cache.canonical_kv_page import (
    CanonicalPageError,
    CanonicalPageSpec,
    PageCompleteness,
    attn_layer_index,
)

logger = logging.getLogger(__name__)

# Sidecar suffixes. ``.bin`` (the complete page) is the only name a reader looks
# for, so neither of these can ever be mistaken for a servable page.
PART_SUFFIX = ".part706"
MARKER_SUFFIX = ".slots706"

_MARKER_MAGIC = b"SGL706\x02"
# magic | total bytes (u64) | interval count (u16) | that many (start, end) pairs
_MARKER_HEADER = struct.Struct("<QH")
_MARKER_INTERVAL = struct.Struct("<QQ")


@dataclasses.dataclass(frozen=True)
class CanonicalExtentWindow:
    """The byte ranges one rank owns inside a canonical blob.

    The KV page is the one-extent case (a PP stage's attention layers are a
    contiguous run of slots). The mamba blob is the reason this generalisation
    exists: its layer cut is TWO disjoint ranges -- the blob is temporal state
    for every layer followed by conv state for every layer, so a stage's layers
    are contiguous within each region but the regions are far apart -- and under
    TP the head cut adds three sub-block ranges per layer. Both are just extent
    lists, so one protocol carries both.

    ``extents`` are in PAYLOAD order: the caller's flat buffer is consumed
    front to back across them, which is the order ``temporal_extents`` /
    ``conv_extents`` / ``layer_extents`` already return.
    """

    total_bytes: int
    extents: tuple[tuple[int, int], ...]
    label: str = "page"

    def __post_init__(self) -> None:
        if int(self.total_bytes) <= 0:
            raise CanonicalPageError(
                f"a canonical {self.label} cannot be {self.total_bytes} bytes."
            )
        if not self.extents:
            raise CanonicalPageError(
                f"a rank with no bytes in the canonical {self.label} has no "
                "window; it must not take part in the protocol at all."
            )
        seen: list[tuple[int, int]] = []
        for off, length in self.extents:
            if int(length) <= 0:
                raise CanonicalPageError(f"extent at offset {off} has length {length}.")
            if int(off) < 0 or int(off) + int(length) > int(self.total_bytes):
                raise CanonicalPageError(
                    f"extent [{off}, {int(off) + int(length)}) runs outside the "
                    f"{self.total_bytes}-byte canonical {self.label}."
                )
            seen.append((int(off), int(off) + int(length)))
        seen.sort()
        for (a_lo, a_hi), (b_lo, _b_hi) in zip(seen, seen[1:]):
            if b_lo < a_hi:
                raise CanonicalPageError(
                    f"extents [{a_lo}, {a_hi}) and [{b_lo}, ...) of this "
                    f"{self.label} window overlap. A rank writes its OWN bytes "
                    "once; an overlap means the cut is wrong, and the payload "
                    "would land twice at one offset."
                )

    @property
    def payload_bytes(self) -> int:
        return sum(int(length) for _off, length in self.extents)

    @property
    def is_whole(self) -> bool:
        return self.payload_bytes == int(self.total_bytes)


@dataclasses.dataclass(frozen=True)
class CanonicalPageWindow:
    """The contiguous run of GLOBAL page slots one rank writes and reads.

    A PP stage owns a contiguous run of model layers, so its attention layers
    are a contiguous run of page slots. The TP decode phase owns all of them
    (``first_slot=0, num_slots=spec.num_attn_layers``) and therefore uses the
    very same code path -- there is no second reader.
    """

    spec: CanonicalPageSpec
    first_slot: int
    num_slots: int

    def __post_init__(self) -> None:
        total = int(self.spec.num_attn_layers)
        if int(self.num_slots) <= 0:
            raise CanonicalPageError(
                f"a window must cover at least one slot, got {self.num_slots}. "
                "A rank with no attention layers has nothing to persist and "
                "must not be given a page window at all."
            )
        if not 0 <= int(self.first_slot) < total:
            raise CanonicalPageError(
                f"window starts at slot {self.first_slot}, outside the {total} "
                "slots this page carries."
            )
        if int(self.first_slot) + int(self.num_slots) > total:
            raise CanonicalPageError(
                f"window [{self.first_slot}, "
                f"{int(self.first_slot) + int(self.num_slots)}) runs past the "
                f"{total} slots this page carries."
            )

    @property
    def cell_bytes(self) -> int:
        return int(self.spec.kv_bytes_per_token_per_attn_layer)

    @property
    def byte_offset(self) -> int:
        return int(self.first_slot) * self.cell_bytes

    @property
    def byte_length(self) -> int:
        return int(self.num_slots) * self.cell_bytes

    @property
    def slots(self) -> tuple[int, ...]:
        return tuple(range(int(self.first_slot), int(self.first_slot + self.num_slots)))

    @property
    def is_whole_page(self) -> bool:
        return int(self.num_slots) == int(self.spec.num_attn_layers)

    def as_extents(self) -> CanonicalExtentWindow:
        """The KV window as the generic one-extent case."""
        return CanonicalExtentWindow(
            total_bytes=self.spec.page_bytes,
            extents=((self.byte_offset, self.byte_length),),
            label="KV page",
        )


def window_for_layers(
    spec: CanonicalPageSpec,
    attn_layer_ids: Sequence[int],
    local_layer_ids: Sequence[int],
) -> CanonicalPageWindow:
    """Page window of a rank that holds ``local_layer_ids``.

    ``attn_layer_ids`` is the MODEL's full attention-layer id list (global, in
    model order); ``local_layer_ids`` are this rank's attention layers, given by
    their GLOBAL ids -- e.g. the keys of
    ``HybridLinearKVPool.full_attention_layer_id_mapping``, which are global
    because the list was filtered by the runner's global stage bounds, not
    renumbered.

    Passing rank-local ordinals here is THE trap of this design: on a hybrid
    checkpoint some local ordinals are themselves valid attention-layer ids, so
    the lookup succeeds and returns a window of the right size at the wrong
    offset. Nothing downstream can tell the difference -- see the module
    docstring for why the completeness marker still turns that into a miss
    rather than a wrong answer.
    """
    ids = [int(i) for i in local_layer_ids]
    if not ids:
        raise CanonicalPageError(
            "a rank with no attention layers has no page window; it must not "
            "take part in the canonical page protocol at all."
        )
    if ids != sorted(ids):
        raise CanonicalPageError(
            f"local attention layers {ids} are not in model order; a page "
            "window is a run of slots and the caller's order decides which "
            "bytes land in which slot."
        )
    slots = [attn_layer_index(i, attn_layer_ids) for i in ids]
    expected = list(range(slots[0], slots[0] + len(slots)))
    if slots != expected:
        raise CanonicalPageError(
            f"attention layers {ids} map to page slots {slots}, which is not a "
            "contiguous run. A PP stage owns a contiguous layer range, so a "
            "gap here means the layer list is not this stage's range -- "
            "refusing rather than writing a page nobody can cut."
        )
    if len(attn_layer_ids) != int(spec.num_attn_layers):
        raise CanonicalPageError(
            f"the model has {len(attn_layer_ids)} attention layers but the "
            f"page spec describes {spec.num_attn_layers} slots. The spec IS "
            "the layout contract: one key must name one set of bytes for every "
            "geometry, so a mismatch here is a different page format."
        )
    return CanonicalPageWindow(spec=spec, first_slot=slots[0], num_slots=len(slots))


def local_attention_layer_ids(device_pool, attn_layer_ids: Sequence[int]) -> list[int]:
    """This rank's attention layers, by their GLOBAL model ids.

    The one question the whole protocol turns on, so it is answered from the
    pool that KNOWS the answer rather than reconstructed from a layer count:

    1. ``full_attention_layer_id_mapping`` (``HybridLinearKVPool``) is a map
       from GLOBAL attention-layer id to the rank's local index. Its keys are
       global because the runner filtered the model's list with its own stage
       bounds -- ``model_runner_kv_cache_mixin`` -- and never renumbered them.
    2. ``start_layer``/``end_layer`` bounds, but ONLY when they cannot be a
       zero-based local renumbering: either they start above zero, or they span
       the whole model.
    3. A pool that holds ALL of the model's attention layers (no PP split, or
       the TP decode phase) owns the whole list.

    Anything else RAISES, and rule 2's restriction is the reason. A KV pool
    counts ATTENTION layers, so a middle PP stage's local pool reports
    ``start_layer=0, end_layer=5`` while its global range is ``[28, 48)``.
    Filtering the model's attention layers by ``[0, 5)`` yields layer 3 -- one
    plausible layer, at the wrong offset, silently. The two cases are
    indistinguishable from the pool alone, so a zero-based range that does not
    span the model is REFUSED rather than resolved. That refusal also catches
    the honest-but-unanswerable case (a dense model's PP stage 0, whose global
    range genuinely does start at zero): loud, with the reason, instead of a
    page whose slots are off by a stage.
    """
    ids = [int(i) for i in attn_layer_ids]
    mapping = getattr(device_pool, "full_attention_layer_id_mapping", None)
    if mapping is None:
        inner = getattr(device_pool, "full_kv_pool", None)
        mapping = getattr(inner, "full_attention_layer_id_mapping", None)
    if mapping:
        return sorted(int(i) for i in mapping.keys())

    start = getattr(device_pool, "start_layer", None)
    end = getattr(device_pool, "end_layer", None)
    if start is not None and end is not None and int(end) > int(start):
        spans_model = int(end) >= ids[-1] + 1
        if int(start) > 0 or spans_model:
            within = [i for i in ids if int(start) <= i < int(end)]
            if within:
                return within

    layer_num = getattr(device_pool, "layer_num", None)
    if layer_num is not None and int(layer_num) == len(ids):
        return ids

    raise CanonicalPageError(
        f"cannot establish which of the model's {len(ids)} attention layers "
        f"{type(device_pool).__name__} holds (start_layer={start}, "
        f"end_layer={end}, layer_num={layer_num}). The canonical page addresses "
        "slots by GLOBAL attention-layer index, and a zero-based range that "
        "does not span the model is indistinguishable from a rank-local "
        "renumbering -- which would deposit the right number of bytes in the "
        "wrong slots. Refusing to guess."
    )


def host_page_bytes(host_pool) -> int:
    """Bytes of the flat page this host pool reads and writes.

    Taken from ``get_dummy_flat_data_page`` -- the very object the storage path
    hands to the backend -- rather than from ``get_size_per_token``, whose
    meaning differs between pool classes (per token across layers for the MHA
    host pool, per layer for the V4 paged one). The page is the contract; a
    derived number that disagrees with it would size the window against
    something nobody writes.
    """
    page = host_pool.get_dummy_flat_data_page()
    return int(page.numel()) * int(page.element_size())


def resolve_attn_layer_ids(model_config) -> list[int]:
    """The model's full-attention layer ids, or a NAMED refusal.

    THE SPECIMEN, 2026-08-17. The Flip+HiCache boot died on

        CanonicalPageError: attention layers [51, 55, 59, 63] map to page slots
        [51, 55, 59, 63], which is not a contiguous run.

    Slots should have been [12, 13, 14, 15]. The caller had resolved the id
    list as ``full_attention_layer_ids or range(num_hidden_layers)``, under the
    comment "a model with no hybrid split" -- a guard that cannot tell NOT
    HYBRID from NOT POPULATED. This checkpoint is a GDN hybrid whose 16
    full-attention layers sit at 3, 7, ... 63, so it took the dense branch and
    the page was cut against 64 slots instead of 16.

    A LADDER, because there are two different populated sources and one of them
    is empty for this model:

    (a) ``ModelConfig.full_attention_layer_ids`` -- the SWA-hybrid path. It is
        filled by ``get_hybrid_layer_ids``, which ModelConfig calls ONLY under
        ``if self.is_hybrid_swa``. A GDN hybrid is correctly not
        ``is_hybrid_swa``, so this is empty here and must not be read as "dense".
    (b) the checkpoint config's OWN property. ``Qwen3NextConfig`` derives
        ``full_attention_layer_ids`` from ``layers_block_type``, and
        ``Qwen3_5TextConfig`` inherits it. Measured on the deployed checkpoint:
        returns [3, 7, ... 63], count 16 -- exactly the question being asked.
        Sibling GDN configs differ, so this is a per-class lookup by attribute
        rather than an assumption about all of them.
    (c) ``range(n)`` ONLY on a POSITIVE DENSE PROOF: the checkpoint declares no
        layer kinds at all and claims neither hybrid flag. The old comment
        becomes an actual predicate.
    (d) anything else -- kinds declared but no id list -- is a REFUSAL naming
        the architecture and the kinds seen. Guessing here writes a page whose
        slots nobody can cut, and the only reason that was survivable last time
        is that the contiguity check happened to catch it.

    Scope is deliberately this mapping. ``get_hybrid_layer_ids`` is SWA-scoped
    and shared by every model; widening it to classify GDN hybrids would either
    misroute them into SWA machinery or never run, since the ``is_hybrid_swa``
    gate stays False for them.
    """
    # (a) the SWA-hybrid path, when it actually has an answer.
    ids = getattr(model_config, "full_attention_layer_ids", None)
    if ids:
        return [int(i) for i in ids]

    text_config = getattr(model_config, "hf_text_config", None)

    # (b) the checkpoint's own property (GDN hybrids: Qwen3-Next family, ...).
    own = getattr(text_config, "full_attention_layer_ids", None) if text_config else None
    if own:
        return [int(i) for i in own]

    # (c) dense, PROVEN rather than assumed.
    kinds = None
    for attr in ("layer_types", "layers_block_type", "hybrid_layer_pattern"):
        kinds = getattr(text_config, attr, None) if text_config else None
        if kinds:
            break
    declares_hybrid = bool(getattr(model_config, "is_hybrid", None)) or bool(
        getattr(model_config, "is_hybrid_swa", None)
    )
    if not kinds and not declares_hybrid:
        return list(range(int(model_config.num_hidden_layers)))

    # (d) kinds are declared but nothing resolved them -- refuse, and say what
    # was seen so the next reader does not have to re-derive it from a crash.
    arch = getattr(getattr(model_config, "hf_config", None), "architectures", None)
    seen = sorted({str(k) for k in (kinds or [])})
    raise CanonicalPageError(
        f"cannot resolve the full-attention layer ids for {arch}: "
        f"ModelConfig.full_attention_layer_ids is empty, the checkpoint config "
        f"({type(text_config).__name__}) exposes no full_attention_layer_ids "
        f"property, and it declares layer kinds {seen}. This is a hybrid whose "
        f"attention layers are not where a dense model's are, so falling back "
        f"to range(num_hidden_layers) would cut the canonical page against the "
        f"wrong slots -- refusing instead of guessing. Give that config class a "
        f"full_attention_layer_ids property, as Qwen3NextConfig has."
    )


def resolve_linear_layer_ids(model_config) -> list[int]:
    """The model's GDN/linear layer ids, or a NAMED refusal. #931.

    THE SIBLING ``resolve_attn_layer_ids`` WAS GIVEN AND THIS HALF WAS NOT.
    The attention half above got this ladder on 2026-08-17 because
    ``full_attention_layer_ids or range(n)`` could not tell NOT HYBRID from NOT
    POPULATED. The mamba half kept a bare lookup -- and it was not merely
    fragile, it was aimed at the wrong object:

        cache_params = getattr(model_config, "mamba2_cache_params", None)

    ``mamba2_cache_params`` is a property of the CHECKPOINT config
    (``Qwen3NextConfig.mamba2_cache_params``, configs/qwen3_next.py:288, whose
    ``layers=self.linear_layer_ids``), never of sglang's ``ModelConfig``:
    ``grep -c mamba2_cache_params configs/model_config.py`` is 0. So the getattr
    missed on EVERY model, the id list was always empty, and
    ``CacheController._canonical_mamba_window`` returned ``None`` through its
    one silent branch on every boot since a38f39f1ee landed (2026-08-17).

    WHAT THAT COST, measured on boot 2g (2026-08-27,
    boot_2f_698cd396ce_0827_0704.log): with the format explicitly armed
    (``phase_flip_canonical_kv_page=True``, writeback on, 'file' backend,
    page_size 1) the log carries "#706 canonical KV page active" three times,
    once per rank -- and "#706 canonical GDN blob active" ZERO times, with zero
    refusals. Exactly the split ``_canonical_mamba_window``'s own docstring
    names as fatal: "a canonical KV page beside a phase-local GDN blob delivers
    ZERO usable prefix ... Silently running KV-only would look like the feature
    was on while every cross-phase lookup missed." The KV crossed the flip
    geometry-neutrally; the recurrent anchor stayed phase-local, and #928 is
    what that reads like at the API.

    THE SAME LADDER, deliberately, so the two halves cannot drift again:

    (a) ``ModelConfig``'s own list, when a future ModelConfig populates one.
        Empty today for every model, which is why it may never be read as
        "this model has no linear layers".
    (b) the checkpoint config's OWN properties -- ``linear_layer_ids`` first
        (the primitive; Qwen3-Next derives it from ``layers_block_type`` and
        Qwen3_5TextConfig inherits it), then ``mamba2_cache_params.layers``,
        which is that same list wrapped. Read through ``hf_text_config``, the
        hop ``gdn_flip_mover.py:929`` already uses for GDN geometry.
    (c) ``[]`` ONLY on a POSITIVE DENSE PROOF: no layer kinds declared and no
        hybrid flag. An empty list from here is the ONE case where "no blob" is
        the right answer, and the caller may then skip the blob safely.
    (d) anything else -- kinds declared but no id list -- is a REFUSAL. A
        hybrid whose blob cannot be made canonical must not run with canonical
        KV pages alone; that combination hits nothing at all and, worse, serves
        the other phase's recurrent bytes rather than missing cleanly.
    """
    # (a) ModelConfig's own, when it has an answer.
    ids = getattr(model_config, "linear_layer_ids", None)
    if ids:
        return [int(i) for i in ids]

    text_config = getattr(model_config, "hf_text_config", None)

    # (b) the checkpoint's own properties, primitive first.
    own = getattr(text_config, "linear_layer_ids", None) if text_config else None
    if own:
        return [int(i) for i in own]
    cache_params = (
        getattr(text_config, "mamba2_cache_params", None) if text_config else None
    )
    wrapped = getattr(cache_params, "layers", None)
    if wrapped:
        return [int(i) for i in wrapped]

    # (c) no linear layers, PROVEN rather than assumed.
    kinds = None
    for attr in ("layer_types", "layers_block_type", "hybrid_layer_pattern"):
        kinds = getattr(text_config, attr, None) if text_config else None
        if kinds:
            break
    declares_hybrid = bool(getattr(model_config, "is_hybrid", None)) or bool(
        getattr(model_config, "is_hybrid_swa", None)
    )
    if not kinds and not declares_hybrid:
        return []

    # (d) kinds declared, nothing resolved them: refuse and say what was seen.
    arch = getattr(getattr(model_config, "hf_config", None), "architectures", None)
    seen = sorted({str(k) for k in (kinds or [])})
    raise CanonicalPageError(
        f"cannot resolve the GDN/linear layer ids for {arch}: the checkpoint "
        f"config ({type(text_config).__name__}) exposes neither a "
        f"linear_layer_ids property nor mamba2_cache_params.layers, and it "
        f"declares layer kinds {seen}. This is a hybrid, so treating it as "
        f"having no recurrent layers would attach a canonical KV page with a "
        f"PHASE-LOCAL GDN blob -- a prefix that hits nothing, and whose "
        f"recurrent state comes from whichever phase wrote it last (#928). "
        f"Give that config class a linear_layer_ids property, as "
        f"Qwen3NextConfig has."
    )


def build_page_window(
    attn_layer_ids: Sequence[int], device_pool, host_pool
) -> CanonicalPageWindow:
    """This rank's window in the canonical page, from the live pools.

    Also the place where a rank whose per-layer KV size is not the canonical
    one is caught: the flat page it writes must be exactly its slot count times
    the page's cell size, or its bytes are not the canonical form and the
    suffix-free key would name something else.
    """
    ids = [int(i) for i in attn_layer_ids]
    local = local_attention_layer_ids(device_pool, ids)
    page_bytes = host_page_bytes(host_pool)
    if page_bytes % len(local):
        raise CanonicalPageError(
            f"this rank's flat KV page is {page_bytes} bytes over {len(local)} "
            "attention layers, which does not divide evenly. The canonical page "
            "is layer-major with one cell per layer, so an indivisible page is "
            "a different format."
        )
    cell = page_bytes // len(local)
    spec = CanonicalPageSpec(
        num_attn_layers=len(ids), kv_bytes_per_token_per_attn_layer=cell
    )
    window = window_for_layers(spec, ids, local)
    if window.byte_length != page_bytes:
        raise CanonicalPageError(
            f"window [{window.first_slot}, {window.first_slot + window.num_slots}) "
            f"is {window.byte_length} bytes but this rank's page is {page_bytes}."
        )
    return window


def local_mamba_layer_range(
    mamba_pool, mamba_layer_ids: Sequence[int]
) -> tuple[int, int]:
    """This rank's GDN layers as an index range into the model's layer list.

    ``MambaPool.mamba_map`` maps GLOBAL layer id to local index, filtered by the
    runner's own stage bounds -- the same construction that makes the KV side's
    ``full_attention_layer_id_mapping`` global. The blob is laid out in the
    model's mamba-layer ORDER, so what the protocol needs is the position of
    this rank's first and last layer in that list, not the layer ids.

    Refuses rather than guesses, for the reason the KV resolver refuses: a pool
    that cannot name its global layers would deposit the right number of bytes
    at the wrong offsets, and both regions of the blob would be wrong at once.
    """
    ids = [int(i) for i in mamba_layer_ids]
    mapping = getattr(mamba_pool, "mamba_map", None)
    if not mapping:
        raise CanonicalPageError(
            f"{type(mamba_pool).__name__} does not expose mamba_map, so which "
            "GDN layers this rank holds cannot be established. The canonical "
            "blob addresses layers globally; refusing to guess from a count."
        )
    local = sorted(int(i) for i in mapping.keys())
    try:
        positions = [ids.index(i) for i in local]
    except ValueError as e:
        raise CanonicalPageError(
            f"this rank holds GDN layers {local}, which are not all in the "
            f"model's mamba layer list {ids}."
        ) from e
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise CanonicalPageError(
            f"this rank's GDN layers map to positions {positions}, which is not "
            "a contiguous run. A PP stage owns a contiguous layer range, so a "
            "gap means the list is not this stage's range."
        )
    return positions[0], positions[-1] + 1


def derive_mamba_blob_spec(model_config, mamba_pool, *, num_linear_layers: int):
    """FULL (unsharded) GDN blob geometry for this checkpoint.

    Reuses ``qwen3_5_mamba_spec`` -- the recipe the offline migration already
    trusts -- rather than re-deriving the field arithmetic here. The itemsizes
    come from the live pool, because the blob's bytes are the pool's bytes; the
    head/channel counts come from the model config, because the canonical blob
    is the UNSHARDED one and the pool only knows its own shard.

    Raises ``CanonicalPageError`` when the config does not expose the GDN
    fields. That refusal is deliberate and load-bearing: a hybrid model whose
    blob cannot be made canonical must not run with canonical KV pages alone,
    since a KV-only prefix hits nothing at all.
    """
    from sglang.srt.mem_cache.hicache_migrate import qwen3_5_mamba_spec

    text = getattr(model_config, "hf_text_config", None) or model_config
    fields = (
        "linear_num_value_heads",
        "linear_value_head_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_conv_kernel_dim",
    )
    missing = [f for f in fields if getattr(text, f, None) is None]
    if missing:
        raise CanonicalPageError(
            f"the model config does not expose {', '.join(missing)}, so the "
            "canonical GDN blob geometry cannot be derived. Only the Qwen3.5/3.6 "
            "GDN family is described today (hicache_migrate.qwen3_5_mamba_spec); "
            "another linear-attention family needs its own spec before its blob "
            "can be made phase-uniform."
        )
    # Itemsizes from whichever mamba pool the caller has: the host pool exposes
    # them directly, the device pool through the cache it mirrors. Both are the
    # same numbers -- MambaPoolHost reads exactly these two attributes.
    temporal_itemsize = getattr(mamba_pool, "temporal_dtype", None)
    conv_itemsize = getattr(mamba_pool, "conv_dtype", None)
    if temporal_itemsize is None or conv_itemsize is None:
        cache = getattr(mamba_pool, "mamba_cache", None)
        temporal = getattr(cache, "temporal", None)
        conv = getattr(cache, "conv", None)
        temporal_itemsize = getattr(temporal, "dtype", None)
        conv_itemsize = (
            getattr(conv[0], "dtype", None)
            if isinstance(conv, (list, tuple)) and conv
            else getattr(conv, "dtype", None)
        )
    if temporal_itemsize is None or conv_itemsize is None:
        raise CanonicalPageError(
            f"{type(mamba_pool).__name__} exposes neither temporal_dtype / "
            "conv_dtype nor a mamba_cache to read them from, so the blob's byte "
            "sizes cannot be established."
        )
    # MambaBlobSpec models exactly ONE conv region per layer, while
    # MambaPoolHost.get_data_page emits one all-layers region per tensor in
    # ``mamba_cache.conv`` (a list). Every shape builder in this tree returns a
    # single-element list, so the two agree today -- but a config with two conv
    # tensors would lay the page out as [temporal][conv0][conv1] and the spec
    # would describe [temporal][conv0], silently mis-cutting every extent past
    # the first conv region. Checked, not assumed.
    conv_tensors = getattr(getattr(mamba_pool, "mamba_cache", None), "conv", None)
    if isinstance(conv_tensors, (list, tuple)) and len(conv_tensors) != 1:
        raise CanonicalPageError(
            f"this pool carries {len(conv_tensors)} conv state tensors per "
            "layer; the canonical blob layout describes exactly one "
            "(MambaBlobSpec.conv_dim = key_dim*2 + value_dim). Refusing rather "
            "than cutting extents against the wrong region boundaries."
        )
    # gdn_tp_units, with the runtime's own fallback: the key-head count when the
    # model does not coarsen it (mamba_utils.Mamba2StateShape.create).
    units = getattr(text, "gdn_tp_units", None) or int(text.linear_num_key_heads)
    return qwen3_5_mamba_spec(
        {f: getattr(text, f) for f in fields},
        num_linear_layers=int(num_linear_layers),
        units=int(units),
        temporal_itemsize=int(temporal_itemsize.itemsize),
        conv_itemsize=int(conv_itemsize.itemsize),
    )


def _merge_sequential(
    extents,
) -> tuple[tuple[int, int], ...]:
    """Merge ranges that are adjacent in BOTH payload order and target offset.

    Order-preserving on purpose: a global sort-merge could reorder the payload,
    and the payload is consumed front to back across the extents. Merging only
    what is already consecutive keeps the write count down (three PP stages
    write two ranges each instead of dozens) without touching which byte goes
    where.
    """
    out: list[list[int]] = []
    for off, length in extents:
        if out and out[-1][0] + out[-1][1] == int(off):
            out[-1][1] += int(length)
        else:
            out.append([int(off), int(length)])
    return tuple((off, length) for off, length in out)


def build_mamba_window(
    spec, *, ratios: Sequence[int], rank: int, layer_lo: int, layer_hi: int
) -> CanonicalExtentWindow:
    """This rank's window in the canonical ``{hash}.mamba`` blob.

    Composed from the cut definitions that already exist (``hicache_migrate``),
    never re-derived:

    * ``temporal_extents`` -- one range per layer, this rank's head slice;
    * ``conv_extents`` -- THREE ranges per layer, one per ``[q | k | v]``
      sub-block, each sharded independently by heads. A rank's conv shard is
      those three concatenated, not a flat slice of ``conv_dim``; flattening it
      is the documented way to get the wrong channels.

    The LAYER axis (what PP shards) is then a selection over those per-layer
    entries, which is why the two axes compose without either knowing about the
    other: the head cut decides the SIZE of a layer's ranges, the layer cut
    decides WHICH layers' ranges are taken.

    With full heads (``ratios=[1]``) the result merges back to exactly
    ``layer_extents(spec, layer_lo, layer_hi)`` -- two disjoint ranges, temporal
    and conv -- and a test pins that equality, so this composition cannot drift
    from the definition it is supposed to reuse.
    """
    from sglang.srt.mem_cache.hicache_migrate import conv_extents, temporal_extents

    lo, hi = int(layer_lo), int(layer_hi)
    if not 0 <= lo < hi <= int(spec.num_layers):
        raise CanonicalPageError(
            f"mamba layer range [{lo}, {hi}) is not within "
            f"[0, {spec.num_layers}) -- a stage cannot own layers the blob does "
            "not describe."
        )
    temporal = temporal_extents(spec, list(ratios), int(rank))[lo:hi]
    conv = conv_extents(spec, list(ratios), int(rank))[3 * lo : 3 * hi]
    window = CanonicalExtentWindow(
        total_bytes=spec.total_bytes,
        extents=_merge_sequential(list(temporal) + list(conv)),
        label="mamba blob",
    )
    # The size this rank's own blob has, according to the spec API itself. If
    # the composition above and the spec disagree, the geometry-free key would
    # name bytes of a different shape -- so it is checked, not assumed.
    own = spec.shard_for_rank(list(ratios), int(rank)).for_layers(lo, hi)
    if window.payload_bytes != own.total_bytes:
        raise CanonicalPageError(
            f"composed mamba window is {window.payload_bytes} bytes but this "
            f"rank's own blob is {own.total_bytes} bytes "
            f"(ratios={list(ratios)}, rank={rank}, layers=[{lo}, {hi}))."
        )
    return window


@dataclasses.dataclass(frozen=True)
class ExtentWriteResult:
    """Outcome of depositing one window into a canonical blob."""

    completed: bool  # this write covered the last missing byte
    already_complete: bool  # the blob was whole before this call
    coverage: tuple[tuple[int, int], ...]  # byte ranges present afterwards

    @property
    def missing_bytes(self) -> int:
        total = self.coverage[-1][1] if self.coverage else 0
        return max(0, total - sum(hi - lo for lo, hi in self.coverage))


@dataclasses.dataclass(frozen=True)
class SliceWriteResult:
    """Outcome of depositing one window into a KV page."""

    completed: bool  # this write set the last missing slot
    already_complete: bool  # the page was whole before this call
    missing: tuple[int, ...]  # slots still absent afterwards


def _missing_slots_from_coverage(
    coverage: Sequence[tuple[int, int]], spec: CanonicalPageSpec
) -> tuple[int, ...]:
    """KV slots not FULLY covered, expressed through Slot-3's contract object.

    Fully is the operative word: a slot whose bytes are partly present is
    missing, exactly as it would be under a per-slot bitmap.
    """
    completeness = PageCompleteness(spec)
    for slot in range(int(spec.num_attn_layers)):
        lo, hi = spec.layer_span(slot)
        if any(c_lo <= lo and hi <= c_hi for c_lo, c_hi in coverage):
            completeness.mark(slot)
    return completeness.missing()


def part_path(final_path: str) -> str:
    return final_path + PART_SUFFIX


def marker_path(final_path: str) -> str:
    return final_path + MARKER_SUFFIX


def sweep_partials(
    root_dir: str,
    *,
    older_than_s: float,
    is_pinned: Optional[Callable[[str], bool]] = None,
) -> int:
    """Reap orphaned ``.part706`` / ``.slots706`` files. Returns the count.

    A partial blob is invisible to every reader and untracked by the LRU
    evictor, which walks ``.bin`` only -- so a page whose other writers never
    arrive (a stage that crashed, a prefix evicted from the device tier before
    the last stage got to it) leaves a full-width sparse file with no owner.
    That is the same lifecycle gap today's ``.tmp.`` staging files have, and the
    same answer: reap by age, never by guessing intent.

    ``older_than_s`` must be comfortably longer than the time all writers of one
    page need. Anything younger is presumed live and left alone -- reaping a
    partial another stage is still filling would silently undo its work, and the
    page would simply never complete.
    """
    cutoff = time.time() - float(older_than_s)
    reaped = 0
    for path in _iter_partial_files(root_dir):
        try:
            if os.stat(path).st_mtime >= cutoff:
                continue
        except FileNotFoundError:
            continue
        if is_pinned is not None and is_pinned(_stem_of_partial(path)):
            # #410: a conversation checkpoint references this page. Its partial
            # may be mid-assembly by another stage, and reaping it would reset
            # work the checkpoint is waiting on -- age alone cannot tell those
            # apart, but a pin can.
            continue
        _unlink_quiet(path)
        reaped += 1
    if reaped:
        logger.info(
            "Reaped %d orphaned canonical partial file(s) older than %.0fs in %s.",
            reaped,
            older_than_s,
            root_dir,
        )
    return reaped


def _stem_of_partial(path: str) -> str:
    """The store stem a ``.part706`` / ``.slots706`` file belongs to."""
    name = os.path.basename(path)
    for suffix in (PART_SUFFIX, MARKER_SUFFIX):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.endswith(".bin"):
        name = name[:-4]
    return name


def _iter_partial_files(root_dir: str):
    """Every partial/marker file under the store, shard directories included."""
    try:
        with os.scandir(root_dir) as it:
            top = list(it)
    except FileNotFoundError:
        return
    for entry in top:
        if entry.is_dir():
            try:
                with os.scandir(entry.path) as shard_it:
                    children = list(shard_it)
            except OSError:
                continue
        else:
            children = [entry]
        for child in children:
            if child.name.endswith((PART_SUFFIX, MARKER_SUFFIX)):
                yield child.path


def _merge(intervals: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Sorted, non-overlapping, adjacency-merged coverage."""
    out: list[list[int]] = []
    for lo, hi in sorted((int(a), int(b)) for a, b in intervals):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return tuple((lo, hi) for lo, hi in out)


def _covers(coverage: Sequence[tuple[int, int]], total_bytes: int) -> bool:
    return len(coverage) == 1 and coverage[0] == (0, int(total_bytes))


def _encode_marker(total_bytes: int, coverage: Sequence[tuple[int, int]]) -> bytes:
    """Completeness as BYTE coverage, not a slot bitmap.

    The KV page could be tracked by layer slot, because its kv-heads are
    replicated and token-sharded -- a layer is written by exactly one writer, so
    the layer is the atom. The mamba blob cannot: under TP its state is
    HEAD-sharded, so one layer's bytes arrive from several ranks, and a
    layer-granular marker would call a layer complete when a third of its
    channels were present. Bytes are the only unit both axes agree on, so both
    pools use it and there is one marker format rather than one per pool.
    """
    merged = _merge(coverage)
    body = b"".join(_MARKER_INTERVAL.pack(lo, hi) for lo, hi in merged)
    return _MARKER_MAGIC + _MARKER_HEADER.pack(int(total_bytes), len(merged)) + body


def _decode_marker(
    blob: bytes, total_bytes: int
) -> Optional[tuple[tuple[int, int], ...]]:
    """Recorded coverage, or ``None`` when the marker is unusable.

    ``None`` covers a truncated marker and one written for a blob of a
    DIFFERENT size, i.e. under a different geometry. Both mean the partial file
    on disk cannot be reasoned about, and since it is only ever a cache entry
    the answer is to discard it and start again -- loudly, never by
    reinterpreting foreign bytes.
    """
    head = len(_MARKER_MAGIC) + _MARKER_HEADER.size
    if len(blob) < head or blob[: len(_MARKER_MAGIC)] != _MARKER_MAGIC:
        return None
    recorded_total, count = _MARKER_HEADER.unpack(blob[len(_MARKER_MAGIC) : head])
    if recorded_total != int(total_bytes):
        return None
    body = blob[head:]
    if len(body) != count * _MARKER_INTERVAL.size:
        return None
    intervals = []
    for i in range(count):
        lo, hi = _MARKER_INTERVAL.unpack_from(body, i * _MARKER_INTERVAL.size)
        if not 0 <= lo < hi <= int(total_bytes):
            return None
        intervals.append((lo, hi))
    return _merge(intervals)


def _as_bytes(payload: torch.Tensor) -> memoryview:
    if payload.dtype != torch.uint8:
        payload = payload.view(torch.uint8)
    return memoryview(payload.contiguous().numpy()).cast("B")


class PageComplete(Exception):
    """Raised inside the lock when another writer finished the blob first."""


def _open_part_locked(part: str, final_path: str) -> int:
    """Open the partial blob and hold an exclusive lock on it.

    The lock IS the ``.part706`` file rather than a separate ``.lock`` file, and
    that is a correctness-shaped choice, not a tidiness one: a dedicated lock
    file has no lifecycle -- nothing may delete it while another process might
    be about to lock it -- so it accumulates one inode per page, forever. The
    partial file already has a lifecycle: the rename that publishes the page
    takes its name away.

    The cost is the classic lock-the-file-you-opened race: a writer can be
    granted a lock on an inode that has since been renamed to the final name, or
    replaced. Both are detected -- the final path is re-checked, and the fd's
    inode is compared against the path's -- and the answer is to retry, never to
    write into a file that is no longer the one this path names.

    Raises ``PageComplete`` when the blob was finished while waiting.
    """
    for _attempt in range(8):
        fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        if os.path.exists(final_path):
            _close_unlocked(fd)
            raise PageComplete()
        try:
            on_disk = os.stat(part).st_ino
        except FileNotFoundError:
            _close_unlocked(fd)
            continue
        if on_disk == os.fstat(fd).st_ino:
            return fd
        # The file at this path is no longer the one we locked: another writer
        # renamed or replaced it. Drop the stale inode and take the new one.
        _close_unlocked(fd)
    raise CanonicalPageError(
        f"could not obtain a stable lock on {os.path.basename(part)} after 8 "
        "attempts; the partial page is being replaced faster than it can be "
        "written."
    )


def _close_unlocked(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            logger.warning("Could not remove %s: %s", path, e)


def write_extents(
    final_path: str,
    window: CanonicalExtentWindow,
    payload: torch.Tensor,
    *,
    fsync: bool = True,
    space_check: Optional[Callable[[int], None]] = None,
) -> ExtentWriteResult:
    """Deposit one window's bytes into the canonical blob at ``final_path``.

    The single write path for every pool. Returns without touching anything when
    the blob is already complete: blobs are content-addressed, so a complete one
    already holds exactly these bytes and rewriting them would only risk tearing
    a file a reader is using.
    """
    data = _as_bytes(payload)
    if data.nbytes != window.payload_bytes:
        raise CanonicalPageError(
            f"this {window.label} window expects {window.payload_bytes} bytes "
            f"across {len(window.extents)} extent(s), got {data.nbytes}. A rank "
            "whose shard size differs from the canonical form is not writing "
            "that form -- refusing rather than padding."
        )

    if os.path.exists(final_path):
        return ExtentWriteResult(
            completed=False, already_complete=True, coverage=((0, window.total_bytes),)
        )

    if space_check is not None:
        # Before the part file is created, so a refusal leaves nothing behind.
        space_check(window.total_bytes)

    part = part_path(final_path)
    marker = marker_path(final_path)
    try:
        fd = _open_part_locked(part, final_path)
    except PageComplete:
        return ExtentWriteResult(
            completed=False,
            already_complete=True,
            coverage=((0, window.total_bytes),),
        )
    try:
        covered: tuple[tuple[int, int], ...] = ()
        fresh = os.fstat(fd).st_size == 0
        if not fresh:
            try:
                with open(marker, "rb") as f:
                    decoded = _decode_marker(f.read(), window.total_bytes)
            except FileNotFoundError:
                decoded = None
            if decoded is None:
                logger.warning(
                    "Resetting partial canonical %s %s: its completeness marker "
                    "is missing or describes a different geometry.",
                    window.label,
                    os.path.basename(part),
                )
                # Reset in place rather than unlink: the lock lives on this
                # inode, so replacing the file would hand a waiter a lock on a
                # file nobody else can see.
                os.ftruncate(fd, 0)
            else:
                covered = decoded

        if os.fstat(fd).st_size != window.total_bytes:
            # Full width from creation: every writer addresses absolute offsets,
            # so the file cannot grow into its own layout.
            os.ftruncate(fd, window.total_bytes)
        taken = 0
        for off, length in window.extents:
            os.pwrite(fd, data[taken : taken + length], off)
            taken += length
        # Re-writing bytes this rank already wrote is legitimate (a crashed run
        # resumes; the bytes are content-addressed and identical), which is why
        # coverage is a union rather than a strict claim.
        coverage = _merge(
            list(covered) + [(off, off + length) for off, length in window.extents]
        )

        if not _covers(coverage, window.total_bytes):
            with open(marker, "wb") as f:
                f.write(_encode_marker(window.total_bytes, coverage))
            return ExtentWriteResult(
                completed=False, already_complete=False, coverage=coverage
            )

        if fsync:
            # The blob becomes visible by rename, so the DATA must reach the
            # medium before the name does; otherwise a crash can publish a
            # complete-looking file with unwritten holes. The directory entry
            # itself is deliberately not synced: losing the rename costs a cache
            # miss, which is free.
            os.fsync(fd)
        # Publish while still holding the lock: a waiter that wakes after this
        # finds the final path present and returns "already complete" without
        # touching the inode it locked.
        os.replace(part, final_path)
        _unlink_quiet(marker)
        return ExtentWriteResult(
            completed=True, already_complete=False, coverage=coverage
        )
    finally:
        _close_unlocked(fd)


def write_slice(
    final_path: str,
    window: CanonicalPageWindow,
    payload: torch.Tensor,
    *,
    fsync: bool = True,
    space_check: Optional[Callable[[int], None]] = None,
) -> SliceWriteResult:
    """The KV page's view of ``write_extents``: slots instead of byte ranges."""
    outcome = write_extents(
        final_path,
        window.as_extents(),
        payload,
        fsync=fsync,
        space_check=space_check,
    )
    return SliceWriteResult(
        completed=outcome.completed,
        already_complete=outcome.already_complete,
        missing=_missing_slots_from_coverage(outcome.coverage, window.spec),
    )


def read_extents(
    final_path: str, window: CanonicalExtentWindow, out: torch.Tensor
) -> bool:
    """Read this window's extents out of a COMPLETE canonical blob.

    False means "not served": the file does not exist, or exists only as an
    incomplete ``.part706``. There is no third answer -- a partial blob is never
    handed back, whatever the caller happens to need, because a page missing
    another stage's layers (or a GDN blob missing another rank's channels) is a
    prefix nobody can continue from.
    """
    expected = window.payload_bytes
    buf = _as_bytes(out)
    if buf.nbytes != expected:
        raise CanonicalPageError(
            f"read target holds {buf.nbytes} bytes but this {window.label} "
            f"window is {expected} bytes."
        )
    try:
        fd = os.open(final_path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        size = os.fstat(fd).st_size
        if size != window.total_bytes:
            logger.warning(
                "Canonical %s %s is %d bytes, expected %d; refusing to cut a "
                "slice out of a file of the wrong width.",
                window.label,
                os.path.basename(final_path),
                size,
                window.total_bytes,
            )
            return False
        taken = 0
        for off, length in window.extents:
            got = 0
            while got < length:
                chunk = os.pread(fd, length - got, off + got)
                if not chunk:
                    break
                buf[taken + got : taken + got + len(chunk)] = chunk
                got += len(chunk)
            if got != length:
                logger.warning(
                    "Short read of canonical %s %s: %d of %d bytes at offset %d.",
                    window.label,
                    os.path.basename(final_path),
                    got,
                    length,
                    off,
                )
                return False
            taken += length
    finally:
        os.close(fd)
    return True


def read_slice(final_path: str, window: CanonicalPageWindow, out: torch.Tensor) -> bool:
    """The KV page's view of ``read_extents``."""
    return read_extents(final_path, window.as_extents(), out)


def page_is_complete(final_path: str) -> bool:
    """True when a servable page exists. Incomplete pages are invisible."""
    return os.path.exists(final_path)


def missing_slots(final_path: str, spec: CanonicalPageSpec) -> tuple[int, ...]:
    """Slots still absent from a page, for diagnostics and tests.

    ``()`` for a complete page; every slot for one that does not exist at all.
    """
    if os.path.exists(final_path):
        return ()
    coverage = recorded_coverage(final_path, spec.page_bytes)
    if coverage is None:
        return tuple(range(int(spec.num_attn_layers)))
    return _missing_slots_from_coverage(coverage, spec)


def recorded_coverage(
    final_path: str, total_bytes: int
) -> Optional[tuple[tuple[int, int], ...]]:
    """Byte ranges an INCOMPLETE blob already holds, or ``None`` if unknown.

    ``None`` means there is nothing usable to reason about -- no marker, a
    truncated one, or one written for a different geometry.
    """
    try:
        with open(marker_path(final_path), "rb") as f:
            return _decode_marker(f.read(), int(total_bytes))
    except FileNotFoundError:
        return None


class OutOfSpace(OSError):
    """Refused before writing: the filesystem is too close to full."""


# statvfs is a syscall per call; the answer moves slowly compared to page
# writes, so it is cached per directory for this long.
_SPACE_PROBE_INTERVAL_S = 2.0
_SPACE_CACHE: dict = {}


def free_bytes(root_dir: str, *, now: Callable[[], float] = time.monotonic) -> int:
    """Free bytes on the filesystem holding ``root_dir``, TTL-cached."""
    cached = _SPACE_CACHE.get(root_dir)
    stamp = now()
    if cached is not None and stamp - cached[0] < _SPACE_PROBE_INTERVAL_S:
        return cached[1]
    st = os.statvfs(root_dir)
    value = int(st.f_bavail) * int(st.f_frsize)
    _SPACE_CACHE[root_dir] = (stamp, value)
    return value


def reset_space_cache() -> None:
    """Test hook: forget the cached statvfs answers."""
    _SPACE_CACHE.clear()


def ensure_space(root_dir: str, need_bytes: int, floor_bytes: int) -> None:
    """Refuse a canonical write that would take the filesystem below the floor.

    THE CASE THIS EXISTS FOR is specific to this protocol. A page is published
    by rename after its last byte lands, so a disk that fills mid-write leaves a
    ``.part706`` -- invisible, reaped by TTL, harmless. But the protocol also
    ftruncates a page to FULL WIDTH on creation while only writing this rank's
    slice, so a store can be far closer to full than its file sizes suggest,
    and the failure lands in the middle of a multi-writer assembly where the
    other stages have already spent their IO. Refusing at the door costs one
    cached statvfs; discovering it at pwrite costs the page and everyone's
    work on it.

    Deliberately NOT the LRU evictor's watermark: that one is disabled
    entirely unless a cap or a min-free is configured (``_eviction_configured``),
    which is the default, so on a default store nothing stands between this
    protocol and ENOSPC.
    """
    if floor_bytes <= 0:
        return
    available = free_bytes(root_dir)
    if available - int(need_bytes) < int(floor_bytes):
        raise OutOfSpace(
            f"refusing a {need_bytes}-byte canonical write: {available} bytes "
            f"free on {root_dir!r} would fall below the {floor_bytes}-byte "
            "floor. The store is a cache -- a refused write costs a later miss, "
            "while filling the filesystem costs whatever else lives on it."
        )
