# SPDX-License-Identifier: Apache-2.0
"""Task #58 -- the tower loader's name map, against the REAL checkpoint index.

Metal boot xsn406 (20.09. 16:49Z, tree e964a25983) got past the census fix --
placement ran, no ``W105``, the image request reached the tower -- and then::

    W106 Weg2VisionLoadFailed rid=weg2-8-27 -- VisionStageLoadRefused: the
    tower module has 54 unfilled parameter(s) after loading
    (first: ['blocks.0.attn.qkv_proj.weight', 'blocks.0.attn.qkv_proj.bias',
             'blocks.1.attn.qkv_proj.weight'])

54 = 27 blocks x (weight + bias). The checkpoint calls the fused projection
``attn.qkv``; the module (``VisionAttention`` with ``use_qkv_parallel=True``,
``qwen3_vl.py:205``) calls the same parameter ``attn.qkv_proj``.
``strip_checkpoint_prefix`` stripped the prefix and stopped -- the rename was
never applied, although ``Qwen3VLForConditionalGeneration`` carries it twice
(``hf_to_sglang_mapper``, ``qwen3_vl.py:1253``, and its own loader, ``:1649``).

WHY THESE TESTS READ THE CHECKPOINT INDEX AND NOT THE MODULE
------------------------------------------------------------
The strongest test would build the real tower and assert ``missing == []``.
It cannot run at a desk: ``ServerArgs`` for this model asserts CUDA long
before the module is reached (``server_args.py:15179``,
``"extra_buffer needs CUDA/MUSA/NPU (FLA)"``), and a module built from a
hand-written config would be a re-implementation testing itself.

So these tests take the two ends that ARE desk-readable and pin the map
between them:

* the checkpoint's real tensor NAMES, out of ``model.safetensors.index.json``
  -- names only, not one byte of tensor data;
* upstream's own rename table, read off the class that owns it, so a rename
  in ``qwen3_vl.py`` fails HERE instead of on metal.

The one number that ties them to the boot is **54**: the same 54 the refusal
counted.
"""

from __future__ import annotations

import json
import os
import re
import struct

import pytest

from sglang.srt.planner.vision_stage_load import (
    TOWER_SUBSTR_RENAMES,
    VisionStageLoadRefused,
    map_tower_param_name,
    strip_checkpoint_prefix,
)

MODEL_DIR = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
)
INDEX = os.path.join(MODEL_DIR, "model.safetensors.index.json")

#: The boot's own arithmetic, spelled out so a changed checkpoint fails loudly
#: rather than quietly proving a weaker statement.
BLOCKS = 27
FUSED_QKV_TENSORS = BLOCKS * 2  # weight + bias
TOWER_TENSORS = 333  # the xsn63 manifest number, re-checked here

needs_checkpoint = pytest.mark.skipif(
    not os.path.exists(INDEX), reason=f"checkpoint index not present: {INDEX}"
)


def _checkpoint_tower_names():
    """Every ``model.visual.*`` tensor name. NAMES ONLY -- no bytes read."""
    with open(INDEX) as fh:
        weight_map = json.load(fh)["weight_map"]
    return sorted(k for k in weight_map if ".visual." in k or k.startswith("visual."))


# ===========================================================================
# 1. The checkpoint really is shaped the way the refusal implied.
# ===========================================================================
@needs_checkpoint
class TestTheCheckpointsRealNames:
    def test_the_tower_has_the_manifest_tensor_count(self):
        assert len(_checkpoint_tower_names()) == TOWER_TENSORS

    def test_the_attention_is_stored_FUSED_as_qkv_not_split_and_not_qkv_proj(self):
        """Rules out the other two shapes the coordinator named as possible."""
        names = _checkpoint_tower_names()
        fused = [n for n in names if re.search(r"\.attn\.qkv\.(weight|bias)$", n)]
        assert len(fused) == FUSED_QKV_TENSORS, fused[:5]
        # NOT split q/k/v ...
        assert not [n for n in names if re.search(r"\.attn\.[qkv]\.(weight|bias)$", n)]
        # ... and NOT already carrying the module's spelling.
        assert not [n for n in names if ".attn.qkv_proj." in n]

    def test_the_tower_is_bf16_throughout_and_NOT_quantised(self):
        """The INT8 in the model name does not reach the tower.

        Read from the safetensors HEADER -- dtypes and shapes, zero tensor
        bytes. It matters for the stage: a quantised tower would need a
        quant_config on the module and a different byte budget than the
        0.858 GiB the arming line books.
        """
        with open(INDEX) as fh:
            weight_map = json.load(fh)["weight_map"]
        shards = {v for k, v in weight_map.items() if ".visual." in k}
        assert len(shards) == 1, f"the tower must be ONE extent, got {shards}"
        path = os.path.join(MODEL_DIR, next(iter(shards)))
        with open(path, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        dtypes = {
            h["dtype"]
            for k, h in header.items()
            if k != "__metadata__" and ".visual." in k
        }
        assert dtypes == {"BF16"}, dtypes
        qkv = header["model.visual.blocks.0.attn.qkv.weight"]
        # 3 x hidden(1152) -- the fused projection, so ONE tensor maps to ONE
        # module parameter and no shard_id juggling is needed.
        assert qkv["shape"] == [3456, 1152], qkv


# ===========================================================================
# 2. The map, applied to those real names.
# ===========================================================================
@needs_checkpoint
class TestTheMapOnRealNames:
    def test_every_real_name_maps_and_none_keeps_the_old_spelling(self):
        mapped = strip_checkpoint_prefix({n: object() for n in _checkpoint_tower_names()})
        assert len(mapped) == TOWER_TENSORS, "a collision would silently drop a tensor"
        assert not [k for k in mapped if re.search(r"\.attn\.qkv\.(weight|bias)$", k)]
        assert not [k for k in mapped if k.startswith("model.")]
        assert not [k for k in mapped if k.startswith("visual.")]

    def test_the_54_that_xsn406_counted_are_exactly_the_ones_now_renamed(self):
        """THE REGRESSION, in the boot's own number."""
        mapped = strip_checkpoint_prefix({n: object() for n in _checkpoint_tower_names()})
        renamed = [k for k in mapped if ".attn.qkv_proj." in k]
        assert len(renamed) == FUSED_QKV_TENSORS == 54
        assert "blocks.0.attn.qkv_proj.weight" in mapped
        assert "blocks.0.attn.qkv_proj.bias" in mapped
        assert "blocks.1.attn.qkv_proj.weight" in mapped

    def test_the_names_that_already_matched_are_untouched(self):
        """The other 279 filled on metal; the fix must not disturb them."""
        mapped = strip_checkpoint_prefix({n: object() for n in _checkpoint_tower_names()})
        for name in (
            "blocks.0.attn.proj.weight",
            "blocks.0.mlp.linear_fc1.weight",
            "blocks.0.mlp.linear_fc2.bias",
            "blocks.0.norm1.weight",
            "merger.linear_fc1.weight",
            "merger.norm.bias",
            "patch_embed.proj.weight",
            "pos_embed.weight",
        ):
            assert name in mapped, name
        assert len(mapped) - len([k for k in mapped if ".attn.qkv_proj." in k]) == 279


# ===========================================================================
# 3. The map itself: pure, idempotent, and NOT drifted from upstream.
# ===========================================================================
class TestTheMapIsSoundAndMatchesUpstream:
    def test_it_renames_the_fused_projection(self):
        assert (
            map_tower_param_name("blocks.0.attn.qkv.weight")
            == "blocks.0.attn.qkv_proj.weight"
        )

    def test_it_is_idempotent(self):
        """The reason the table is spelled with dots.

        Upstream's undotted ``"attn.qkv" -> "attn.qkv_proj"`` applied twice
        would produce ``attn.qkv_proj_proj``. This one cannot.
        """
        once = map_tower_param_name("blocks.0.attn.qkv.weight")
        assert map_tower_param_name(once) == once
        assert "qkv_proj_proj" not in map_tower_param_name(once)

    def test_it_leaves_every_other_name_alone(self):
        for name in ("blocks.0.attn.proj.weight", "merger.norm.bias", "pos_embed.weight"):
            assert map_tower_param_name(name) == name

    def test_the_table_has_not_drifted_from_qwen3_vl(self):
        """Read off the class that OWNS the rename, not retyped here.

        If upstream renames the parameter again, this fails at the desk -- the
        alternative is another 54-unfilled refusal on a booked GPU window.
        """
        from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration

        upstream = dict(
            Qwen3VLForConditionalGeneration.hf_to_sglang_mapper.orig_to_new_substr
        )
        assert upstream == {"attn.qkv": "attn.qkv_proj"}, upstream
        # Ours is the same rename in the idempotent (dotted) spelling.
        assert dict(TOWER_SUBSTR_RENAMES) == {
            f"{k}.": f"{v}." for k, v in upstream.items()
        }

    def test_the_second_upstream_spelling_agrees_too(self):
        """``qwen3_vl.py:1649`` does the replace literally; pin that source."""
        import inspect

        from sglang.srt.models import qwen3_vl

        src = inspect.getsource(qwen3_vl)
        assert 'name.replace(r"attn.qkv.", r"attn.qkv_proj.")' in src


# ===========================================================================
# 4. The refusal that guards the map is still total.
# ===========================================================================
class TestUnmappableNamesStillRefuse:
    def test_a_name_without_a_known_prefix_refuses(self):
        with pytest.raises(VisionStageLoadRefused) as ei:
            strip_checkpoint_prefix({"language_model.layers.0.mlp.up_proj.weight": 1})
        assert "missing_keys" in str(ei.value)

    def test_the_visual_prefix_form_is_mapped_too(self):
        out = strip_checkpoint_prefix({"visual.blocks.3.attn.qkv.weight": 1})
        assert list(out) == ["blocks.3.attn.qkv_proj.weight"]
