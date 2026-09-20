# SPDX-License-Identifier: Apache-2.0
"""Task #58 slices 2 and 3 -- READING the transient tower, and HANDING ON its
embeddings.

Slice 1 (``vision_stage.py``) answers *which card*.  This module answers the
two questions on either side of the encoder forward:

* **Slice 2, the load.**  Where in the checkpoint the tower is, how many bytes
  it is, and how to get exactly those bytes and nothing else -- without
  reading the 93 GB of language-model tensors that sit in the same directory.
* **Slice 3, the hand-off.**  How the encoder's output reaches a P prefill
  whose model has ``self.visual is None``.

Neither needs CUDA.  :func:`tower_extent` needs no torch at all (it parses the
safetensors header, which is JSON behind an 8-byte length).
:func:`tower_state_dict` and :func:`attach_precomputed_embeddings` use torch
for tensors only, never a device.

THE LOAD, AND WHY IT IS ONE READ
--------------------------------
Measured on the serving checkpoint
(``.../models-cache/Qwen3.8-27B-INT8-gdncov``, 2026-09-20): the 333 ``visual``
tensors all live in ``model-00001-of-00018.safetensors``, CONTIGUOUSLY, from
data offset 4_889_688 to 926_349_880 -- 921_460_192 bytes, BF16 throughout.
So the tower is a single sequential extent of 0.858 GiB in one shard, and
:func:`tower_extent` proves that per checkpoint rather than assuming it.

The reader already exists and is the right one:
``model_loader/weight_utils.py:1305 pread_safetensors_file(st_file,
should_load)`` reads tensor-by-tensor with ``os.preadv`` in FILE ORDER, no
mmap and no whole-file buffer, and its ``should_load`` predicate answers
``True`` / ``False`` / ``"meta"`` per name.  Its own docstring records why not
mmap: on this rig's ZFS a page-faulted mmap read runs ~0.5 GB/s per rank
against ~3 GB/s for ``read()``.  This module adds the predicate, not a second
reader.

WHAT THIS MODULE DOES *NOT* CLAIM
---------------------------------
``pread_safetensors_file`` is BUFFERED.  Measured here on exactly this extent:
buffered per-piece ``pread`` 1.08-1.15 GB/s (~820 ms) against O_DIRECT
3.85 GB/s (~239 ms).  The 0.58 s difference is TTFT on every image request,
and closing it means an O_DIRECT path this module does NOT build -- it only
refuses to let a planner quote the direct rate for a load that will take the
buffered one.  :data:`LOADER_IS_BUFFERED` is that statement in a constant, so
a caller of ``plan_vision_stage(read_gbps=...)`` has one place to look.

A FINDING THAT FELL OUT, filed and not fixed here
-------------------------------------------------
``Qwen3_5ForConditionalGeneration`` and ``Qwen3VLForConditionalGeneration``
have **no** ``weight_name_needed`` (grep over ``srt/models/``: only
``qwen4_exp.py:2180`` and ``qwen4_exp_mtp.py:246`` define one).  The loader
looks the method up by name (``loader.py:763``) and passes it to the readers
as ``should_load``.  Its absence means the tower's 0.858 GiB is READ OFF DISK
on every text-only rank and then dropped by NAME afterwards in
``load_weights`` (``skip_vision_weight``, ``qwen3_vl.py:1243``) -- 0.858 GiB
per rank, six ranks, ~5.1 GiB of reads per boot that are thrown away.
:func:`text_only_weight_veto` is the predicate that would end it; wiring it
onto the model is a one-line change in the model class and is NOT made here,
because it changes the boot's load path and belongs in its own slice with its
own boot.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

#: ``pread_safetensors_file`` goes through the page cache.  Measured on this
#: box against the tower's own extent: 1.08-1.15 GB/s buffered against
#: 3.85 GB/s O_DIRECT.  A planner must be handed the BUFFERED number for a
#: load that uses this path.
LOADER_IS_BUFFERED = True
MEASURED_LOADER_GBPS = 1.08

#: The checkpoint-name test the model file already uses
#: (``qwen3_vl.py:1238 is_vision_weight``).  Kept as one constant so the
#: reader's predicate and the model's skip cannot drift apart.
VISION_NAME_TOKEN = "visual"


class VisionStageLoadRefused(RuntimeError):
    """The tower cannot be read, and the message says which shard and why."""


class VisionEmbeddingRefused(RuntimeError):
    """The encoder's output cannot be handed to the prefill as it stands.

    Raised in preference to letting a wrong-width or wrong-length tensor reach
    ``_get_precomputed_embedding`` (``mm_utils.py:413``), where it would be
    concatenated and scattered and come out as plausible WRONG text -- the
    same failure the Weg-2 front's W101 refusal exists to prevent
    (``weg2/front.py:2620``).
    """


def is_vision_weight(name: str) -> bool:
    """Same test as ``qwen3_vl.py:1238``, restated here so the reader-side
    predicate and the model-side skip are one definition."""
    return VISION_NAME_TOKEN in name


def text_only_weight_veto(name: str) -> bool:
    """``should_load`` for a rank that will NOT build a tower.

    ``False`` means the reader skips the tensor entirely -- it is never read
    off disk.  This is the predicate the finding in the module docstring is
    about; it is exported so the fix, when it is made, uses the same test the
    model's ``skip_vision_weight`` uses and cannot drift from it.
    """
    return not is_vision_weight(name)


def vision_only_selector(name: str) -> bool:
    """``should_load`` for the transient stage: the tower and nothing else."""
    return is_vision_weight(name)


def vision_meta_selector(name: str):
    """``should_load`` that yields the tower's SHAPES without reading a byte.

    ``pread_safetensors_file`` answers a ``"meta"`` verdict with a
    ``device="meta"`` tensor of the right shape and dtype
    (``weight_utils.py:1330-1332``).  That is how the stage can be sized, and
    its module built, at a desk with no GPU and no 0.9 GiB of reads.
    """
    return "meta" if is_vision_weight(name) else False


@dataclass(frozen=True)
class TowerExtent:
    """Where the tower is in a checkpoint, as its headers say.

    ``contiguous`` is the load-bearing field: when it is True the whole tower
    is one sequential read, and only then may a planner quote a sequential
    rate.  :meth:`as_span_args` hands exactly the four numbers
    ``vision_stage.tower_from_span`` checks.

    THE OFFSET TRAP, named because it cost a test: safetensors'
    ``data_offsets`` are relative to the START OF THE DATA SECTION, not to the
    start of the file.  ``first_offset`` / ``last_end`` here are the
    safetensors numbers; ``data_base`` (8 + header length) is what turns them
    into the ``pread`` positions, and :attr:`file_first_offset` /
    :attr:`file_last_end` do it.  The two differ by ``data_base`` and the SPAN
    is the same either way -- which is exactly why a mismatch between the two
    conventions reads as "close enough" and is not.
    """

    shard: str
    pieces: int
    first_offset: int
    last_end: int
    byte_sum: int
    dtypes: Tuple[str, ...]
    #: 8 + header length: the file offset the data section starts at.
    data_base: int = 0

    @property
    def contiguous(self) -> bool:
        return (self.last_end - self.first_offset) == self.byte_sum

    @property
    def file_first_offset(self) -> int:
        """Absolute ``pread`` position of the tower's first byte."""
        return int(self.data_base) + int(self.first_offset)

    @property
    def file_last_end(self) -> int:
        return int(self.data_base) + int(self.last_end)

    @property
    def gap_bytes(self) -> int:
        """Bytes inside the extent that are NOT tower. Zero when contiguous."""
        return (self.last_end - self.first_offset) - self.byte_sum

    def as_span_args(self) -> Dict[str, Any]:
        return dict(
            pieces=self.pieces,
            first_offset=self.first_offset,
            last_end=self.last_end,
            byte_sum=self.byte_sum,
            shard=os.path.basename(self.shard),
        )


def read_safetensors_header(path: str) -> Tuple[Dict[str, Any], int]:
    """Header dict and the offset its data section starts at.

    Deliberately a local 12-line parser rather than an import of
    ``weight_utils``: the whole point of this function is that a planner can
    price a checkpoint without pulling torch, CUDA and the loader's import
    graph into the process.
    """
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise VisionStageLoadRefused(
                f"{path}: not a safetensors file (shorter than its 8-byte "
                "header length)"
            )
        (hlen,) = struct.unpack("<Q", raw)
        blob = fh.read(hlen)
        if len(blob) != hlen:
            raise VisionStageLoadRefused(
                f"{path}: header claims {hlen} bytes, file has {len(blob)}"
            )
    return json.loads(blob), 8 + hlen


def tower_extent(
    shard: str, *, selector: Callable[[str], bool] = is_vision_weight
) -> TowerExtent:
    """Measure the tower's extent in ONE shard, from its header only.

    Refuses when the shard holds no matching tensor -- an empty extent priced
    as zero bytes would place the stage anywhere at all.
    """
    header, base = read_safetensors_header(shard)
    picked = [
        (k, v)
        for k, v in header.items()
        if k != "__metadata__" and selector(k) and "data_offsets" in v
    ]
    if not picked:
        raise VisionStageLoadRefused(
            f"{shard}: no tensor matches the tower selector. A tower of zero "
            "bytes would be placeable on any card, so this refuses rather "
            "than returning an empty extent."
        )
    offsets = [(int(v["data_offsets"][0]), int(v["data_offsets"][1])) for _, v in picked]
    return TowerExtent(
        shard=shard,
        pieces=len(picked),
        first_offset=min(s for s, _ in offsets),
        last_end=max(e for _, e in offsets),
        byte_sum=sum(e - s for s, e in offsets),
        dtypes=tuple(sorted({str(v["dtype"]) for _, v in picked})),
        data_base=int(base),
    )


def find_tower_shard(
    model_dir: str, *, selector: Callable[[str], bool] = is_vision_weight
) -> str:
    """The ONE shard holding the tower, from the index -- or a refusal.

    A tower split across shards is not an error in the checkpoint, but it is a
    different load (N opens, N extents) and the single-extent cost model does
    not hold for it.  Saying so here is cheaper than discovering it on metal.
    """
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index):
        raise VisionStageLoadRefused(
            f"{model_dir}: no model.safetensors.index.json; this loader prices "
            "a sharded checkpoint and will not guess a single-file layout"
        )
    with open(index, "rb") as fh:
        weight_map = json.load(fh)["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if selector(k)})
    if not shards:
        raise VisionStageLoadRefused(
            f"{model_dir}: the index lists no tower tensor. This checkpoint "
            "cannot serve images at all -- refuse the request, do not start a "
            "stage that would load nothing."
        )
    if len(shards) > 1:
        raise VisionStageLoadRefused(
            f"{model_dir}: the tower spans {len(shards)} shards ({', '.join(shards)}). "
            "The single-extent cost model in vision_stage.py does not hold; "
            "price it as one read per shard before planning a placement."
        )
    return os.path.join(model_dir, shards[0])


def tower_state_dict(shard: str, *, meta: bool = False) -> Dict[str, Any]:
    """Read the tower's tensors -- and only those -- out of one shard.

    ``meta=True`` returns ``device="meta"`` tensors: the exact shapes and
    dtypes, zero bytes read.  That is the desk form, and it is also the form a
    caller wants when it only needs to build the module before deciding
    whether to fill it.

    torch is imported HERE, not at module scope, so a planner that only calls
    :func:`tower_extent` never pays for it.
    """
    from sglang.srt.model_loader.weight_utils import pread_safetensors_file

    selector = vision_meta_selector if meta else vision_only_selector
    sd = pread_safetensors_file(shard, should_load=selector)
    if not sd:
        raise VisionStageLoadRefused(
            f"{shard}: the reader returned no tower tensor. Either the shard "
            "is not the tower's shard, or the selector and the checkpoint's "
            "naming have drifted apart."
        )
    return sd


#: Substring renames the tower module needs on top of the prefix strip, taken
#: 1:1 from ``Qwen3VLForConditionalGeneration.hf_to_sglang_mapper``
#: (``models/qwen3_vl.py:1252-1254``) and from that class's own loader
#: (``:1649``: ``name.replace(r"attn.qkv.", r"attn.qkv_proj.")``).
#:
#: THE DEFECT THIS EXISTS TO END (metal boot xsn406, 20.09. 16:49Z): the stage
#: stripped the prefix and stopped there, so ``blocks.N.attn.qkv.weight`` was
#: handed to a module whose parameter is ``blocks.N.attn.qkv_proj.weight``
#: (``VisionAttention`` with ``use_qkv_parallel=True``, ``qwen3_vl.py:205``).
#: 27 blocks x (weight + bias) = **54 unfilled parameters**, and the stage
#: refused with ``W106 Weg2VisionLoadFailed``. The checkpoint name is right and
#: the module name is right; only the MAP between them was missing.
#:
#: The DOTTED form is deliberate. Upstream's ``orig_to_new_substr`` entry is
#: the undotted ``"attn.qkv" -> "attn.qkv_proj"``, which is safe there because
#: ``WeightsMapper`` runs once over raw checkpoint names. Applied as a bare
#: substring a second time it would turn ``attn.qkv_proj.weight`` into
#: ``attn.qkv_proj_proj.weight``. The dotted form is IDEMPOTENT, so this
#: mapper can be applied to an already-mapped dict without corrupting it --
#: and it is exactly the spelling qwen3_vl's own loader uses.
#: ``test_vision_tower_name_mapping_0920`` pins it against BOTH upstream forms,
#: so a rename upstream fails at the desk instead of on metal.
TOWER_SUBSTR_RENAMES: Tuple[Tuple[str, str], ...] = (
    ("attn.qkv.", "attn.qkv_proj."),
)


def map_tower_param_name(name: str) -> str:
    """One checkpoint tensor name -> the standalone tower's parameter name.

    Pure, idempotent, and the single place the rename table is applied, so a
    caller cannot map half the names.
    """
    for old, new in TOWER_SUBSTR_RENAMES:
        if new in name:
            continue  # already mapped; the rename is idempotent by design
        name = name.replace(old, new)
    return name


def strip_checkpoint_prefix(
    state_dict: Dict[str, Any], prefix: str = "model.visual."
) -> Dict[str, Any]:
    """Map checkpoint names onto the tower module's own parameter names.

    Two steps, and the second one is the xsn406 fix:

    1. The PREFIX goes. The checkpoint calls it
       ``model.visual.blocks.0.attn.qkv.weight``;
       ``Qwen3VLForConditionalGeneration.hf_to_sglang_mapper`` rewrites that to
       ``visual....`` inside the full model (``qwen3_vl.py:1258``), but a stage
       that builds the tower ALONE needs the prefix gone entirely.
    2. The SUBSTRING RENAMES of :data:`TOWER_SUBSTR_RENAMES` apply --
       ``attn.qkv.`` -> ``attn.qkv_proj.``. Skipping this was what left 54
       parameters unfilled on xsn406.

    Refuses on a name that does not carry the prefix rather than passing it
    through, because a silently unmapped key lands in ``missing_keys`` and a
    tower with an unfilled block returns plausible wrong embeddings.
    """
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            bare = k[len(prefix) :]
        elif k.startswith("visual."):
            bare = k[len("visual.") :]
        else:
            raise VisionStageLoadRefused(
                f"tower tensor {k!r} carries neither {prefix!r} nor 'visual.'; "
                "refusing to pass an unmapped name to the module, where it "
                "would end up in missing_keys and the block would stay empty"
            )
        out[map_tower_param_name(bare)] = v
    return out


# --------------------------------------------------------------- slice 3 --


def attach_precomputed_embeddings(
    items: Sequence[Any],
    embeddings: Sequence[Any],
    *,
    expected_width: int,
) -> int:
    """Hand the stage's output to the prefill, item by item.

    THE PATH THIS USES ALREADY EXISTS UPSTREAM, which is the point:
    ``MultimodalDataItem.precomputed_embeddings``
    (``managers/schedule_batch.py:800``) is read by
    ``_get_precomputed_embedding`` (``managers/mm_utils.py:413-472``) inside
    ``get_embedding_and_mask``, and only when that returns ``None`` does
    ``embed_mm_inputs`` reach for the model's ``get_image_feature``
    (``mm_utils.py:833-843``).  So a P rank whose ``self.visual is None``
    (``qwen3_vl.py:1286``) never dereferences the tower it does not have, and
    ``_require_visual`` (``qwen3_vl.py:1421``) stays what it is: the named
    refusal for the case where the stage did NOT run.

    The precedent for setting it in a NON-disaggregated boot is
    ``multimodal/processors/moss_vl.py:587``, which fills the same field in
    the tokenizer process.

    ``expected_width`` is ``out_hidden_size * (1 + len(deepstack_visual_indexes))``
    -- the width ``encode_server.py:441-450`` ships.  On the serving
    checkpoint ``deepstack_visual_indexes`` is EMPTY, so it is plain
    ``out_hidden_size`` = 5120 = the LLM's hidden size and nothing has to be
    split by ``separate_deepstack_embeds`` on the far side.

    Returns the number of rows attached.  Every refusal is by name.
    """
    if len(items) != len(embeddings):
        raise VisionEmbeddingRefused(
            f"the stage produced {len(embeddings)} embedding tensors for "
            f"{len(items)} multimodal items. A positional mismatch would "
            "attach one image's embedding to another image's placeholders and "
            "return fluent wrong text; nothing is guessed here."
        )
    if not items:
        raise VisionEmbeddingRefused(
            "attach called with no items: the stage ran for nothing, which "
            "means the trigger and the request disagree about this request "
            "carrying an image"
        )
    rows = 0
    for idx, (item, emb) in enumerate(zip(items, embeddings)):
        shape = tuple(getattr(emb, "shape", ()))
        if len(shape) != 2:
            raise VisionEmbeddingRefused(
                f"item {idx}: the stage returned shape {shape}; the prefill "
                "consumes a 2-D (rows, width) tensor "
                "(mm_utils.py:468-470 reshapes to 2-D and would silently "
                "accept a wrong fold)"
            )
        if int(shape[1]) != int(expected_width):
            raise VisionEmbeddingRefused(
                f"item {idx}: embedding width {shape[1]} != expected "
                f"{expected_width}. A width mismatch is either the wrong tower "
                "or a deepstack checkpoint whose rows must be split "
                "(mm_utils.py:873-877) -- both are wrong text, not an error, "
                "if they reach the scatter."
            )
        if getattr(emb, "is_cuda", False):
            raise VisionEmbeddingRefused(
                f"item {idx}: the embedding is still on the stage's card. The "
                "stage tears that card down; hand on a host tensor (or one "
                "already moved to the P stage-0 device) so the embedding "
                "outlives the tower."
            )
        item.precomputed_embeddings = emb
        item.feature = None
        rows += int(shape[0])
    return rows


def expected_embedding_rows(
    height: int, width: int, *, patch_size: int, spatial_merge_size: int, frames: int = 1
) -> int:
    """Rows the prefill will expect for one image -- the count
    ``pad_input_ids`` reserves placeholders for.  Kept next to the attach so a
    caller can check the stage's output against the geometry BEFORE the
    scatter, where a count mismatch is a silent misalignment."""
    import math

    gh = height // int(patch_size)
    gw = width // int(patch_size)
    gt = max(1, math.ceil(frames / 2))
    return int(gt * gh * gw) // (int(spatial_merge_size) ** 2)
