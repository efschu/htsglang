# SPDX-License-Identifier: Apache-2.0
"""Task #58 slices 2 and 3 -- reading the transient tower, handing on its rows.

HERMETIC: no CUDA, no GPU, no network.  The core tests build a synthetic
safetensors file in ``tmp_path``; the four marked tests additionally read the
HEADER (never the data) of the real serving checkpoint and skip when it is not
mounted.

WHAT THIS PINS:

1. A checkpoint's tower extent is MEASURED from the header, and the answer for
   the serving checkpoint is the xsn63 manifest to the byte: 333 pieces,
   921_460_192 bytes, one shard, contiguous.
2. A tower split across shards is REFUSED, because the single-extent cost
   model in ``vision_stage.py`` does not hold for it.
3. The meta selector yields every shape with ZERO bytes read -- which is what
   makes the load path desk-testable at all.
4. The vision selector reads the tower and NOTHING else; the text-only veto
   is its exact complement, and it is the predicate the models are currently
   missing (see the module docstring's filed finding).
5. Checkpoint names are mapped onto module names, and an UNMAPPABLE name
   refuses instead of passing through into ``missing_keys``.
6. The hand-off attaches ``precomputed_embeddings`` -- the field
   ``mm_utils.py:413`` consumes BEFORE it would call ``get_image_feature`` --
   and refuses every shape that would otherwise scatter into fluent wrong
   text: count mismatch, wrong rank, wrong width, still-on-the-stage's-card.
"""

import json
import os
import struct

import pytest

from sglang.srt.planner import vision_stage as vs
from sglang.srt.planner import vision_stage_load as vsl

MODEL_DIR = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov"
HAVE_CHECKPOINT = os.path.exists(
    os.path.join(MODEL_DIR, "model.safetensors.index.json")
)
needs_checkpoint = pytest.mark.skipif(
    not HAVE_CHECKPOINT, reason=f"serving checkpoint not mounted at {MODEL_DIR}"
)

# the measured truth this slice is built on
TOWER_PIECES = 333
TOWER_BYTES = 921_460_192
#: safetensors ``data_offsets``, i.e. relative to the DATA SECTION.
TOWER_FIRST = 4_841_984
TOWER_LAST = 926_302_176
#: the same two as ``pread`` positions in the file (data_base = 47_704).
TOWER_FILE_FIRST = 4_889_688
TOWER_FILE_LAST = 926_349_880
TOWER_SHARD = "model-00001-of-00018.safetensors"


# ------------------------------------------------------------- synthetic --


def _write_safetensors(path, tensors):
    """tensors: {name: (dtype, shape, payload_len)} laid out in order."""
    header = {}
    off = 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * off)
    return path


@pytest.fixture
def shard(tmp_path):
    return _write_safetensors(
        tmp_path / "model-00001-of-00002.safetensors",
        {
            "model.visual.blocks.0.attn.qkv.weight": ("BF16", (3456, 1152), 7_962_624),
            "model.visual.blocks.0.attn.qkv.bias": ("BF16", (3456,), 6_912),
            "model.visual.merger.linear_fc2.weight": ("BF16", (5120, 4608), 47_185_920),
            "model.layers.0.self_attn.q_proj.weight": ("BF16", (5120, 5120), 52_428_800),
        },
    )


# --------------------------------------------------------- slice 2: extent --


def test_the_extent_is_measured_from_the_header_not_assumed(shard):
    ext = vsl.tower_extent(str(shard))
    assert ext.pieces == 3
    assert ext.byte_sum == 7_962_624 + 6_912 + 47_185_920
    assert ext.contiguous
    assert ext.gap_bytes == 0
    assert ext.dtypes == ("BF16",)


def test_a_gap_inside_the_extent_is_reported_not_smoothed(tmp_path):
    """A language tensor BETWEEN two tower tensors: the extent is no longer
    one sequential read, and the planner must not be told it is."""
    p = _write_safetensors(
        tmp_path / "s.safetensors",
        {
            "model.visual.a": ("BF16", (2,), 4),
            "model.layers.0.w": ("BF16", (512,), 1024),
            "model.visual.b": ("BF16", (2,), 4),
        },
    )
    ext = vsl.tower_extent(str(p))
    assert not ext.contiguous
    assert ext.gap_bytes == 1024
    with pytest.raises(vs.VisionStageTowerUnreadable):
        vs.tower_from_span(**ext.as_span_args())


def test_a_shard_with_no_tower_refuses_rather_than_pricing_zero(tmp_path):
    p = _write_safetensors(
        tmp_path / "s.safetensors", {"model.layers.0.w": ("BF16", (2,), 4)}
    )
    with pytest.raises(vsl.VisionStageLoadRefused) as e:
        vsl.tower_extent(str(p))
    assert "placeable on any card" in str(e.value)


def test_a_truncated_file_refuses_by_name(tmp_path):
    p = tmp_path / "short.safetensors"
    p.write_bytes(b"\x01\x02")
    with pytest.raises(vsl.VisionStageLoadRefused):
        vsl.read_safetensors_header(str(p))


def test_a_tower_split_across_shards_is_refused(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.visual.a": "model-00001-of-00002.safetensors",
                    "model.visual.b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    with pytest.raises(vsl.VisionStageLoadRefused) as e:
        vsl.find_tower_shard(str(tmp_path))
    assert "spans 2 shards" in str(e.value)
    assert "single-extent cost model" in str(e.value)


def test_a_checkpoint_with_no_tower_at_all_refuses_the_request(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.w": "a.safetensors"}})
    )
    with pytest.raises(vsl.VisionStageLoadRefused) as e:
        vsl.find_tower_shard(str(tmp_path))
    assert "cannot serve images at all" in str(e.value)


def test_a_directory_without_an_index_refuses_instead_of_guessing(tmp_path):
    with pytest.raises(vsl.VisionStageLoadRefused) as e:
        vsl.find_tower_shard(str(tmp_path))
    assert "will not guess" in str(e.value)


# ------------------------------------------------------- slice 2: selectors --


def test_the_selectors_are_exact_complements():
    names = [
        "model.visual.blocks.0.attn.qkv.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "visual.merger.linear_fc1.bias",
        "lm_head.weight",
    ]
    for n in names:
        assert vsl.vision_only_selector(n) is not vsl.text_only_weight_veto(n)
    assert [vsl.vision_only_selector(n) for n in names] == [True, False, True, False]
    # the meta selector answers the reader's third verdict, not a bool
    assert vsl.vision_meta_selector(names[0]) == "meta"
    assert vsl.vision_meta_selector(names[1]) is False


def test_the_meta_read_yields_every_shape_and_reads_no_bytes(shard):
    sd = vsl.tower_state_dict(str(shard), meta=True)
    assert set(sd) == {
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.blocks.0.attn.qkv.bias",
        "model.visual.merger.linear_fc2.weight",
    }
    for t in sd.values():
        assert t.device.type == "meta"
        assert str(t.dtype) == "torch.bfloat16"
    assert tuple(sd["model.visual.merger.linear_fc2.weight"].shape) == (5120, 4608)


def test_the_real_read_returns_the_tower_and_nothing_else(shard):
    sd = vsl.tower_state_dict(str(shard))
    assert all("visual" in k for k in sd)
    assert len(sd) == 3
    assert sd["model.visual.blocks.0.attn.qkv.bias"].numel() == 3456
    assert all(t.device.type == "cpu" for t in sd.values())


def test_the_loader_is_declared_buffered_so_a_planner_quotes_the_right_rate():
    """The danger this pins: planning the stage at the O_DIRECT rate
    (3.85 GB/s, measured) while the code path runs buffered (1.08 GB/s,
    measured) understates TTFT by 0.58 s on every image request."""
    assert vsl.LOADER_IS_BUFFERED is True
    assert vsl.MEASURED_LOADER_GBPS == pytest.approx(1.08)
    assert vsl.MEASURED_LOADER_GBPS < vs.MEASURED_ODIRECT_GBPS


def test_checkpoint_names_are_mapped_onto_module_names(shard):
    sd = vsl.tower_state_dict(str(shard), meta=True)
    mapped = vsl.strip_checkpoint_prefix(sd)
    # CORRECTED after metal boot xsn406: this line used to assert
    # `blocks.0.attn.qkv.weight`, i.e. it pinned the prefix strip alone as
    # sufficient -- and that is precisely the defect. The module's parameter is
    # `attn.qkv_proj` (VisionAttention, use_qkv_parallel=True), so the old
    # expectation was the 54-unfilled-parameter refusal written down as a
    # passing test. See test_vision_tower_name_mapping_0920.py.
    assert "blocks.0.attn.qkv_proj.weight" in mapped
    assert "blocks.0.attn.qkv.weight" not in mapped
    assert "merger.linear_fc2.weight" in mapped
    assert not any(k.startswith("model.") for k in mapped)
    assert len(mapped) == len(sd)


def test_an_unmappable_name_refuses_instead_of_landing_in_missing_keys():
    with pytest.raises(vsl.VisionStageLoadRefused) as e:
        vsl.strip_checkpoint_prefix({"thinker.visual.blocks.0.w": object()})
    assert "missing_keys" in str(e.value)


# ------------------------------------------------ slice 2 against the metal --


@needs_checkpoint
def test_the_serving_checkpoint_has_its_tower_in_one_shard():
    shard = vsl.find_tower_shard(MODEL_DIR)
    assert os.path.basename(shard) == TOWER_SHARD


@needs_checkpoint
def test_the_serving_checkpoints_extent_is_the_xsn63_manifest_to_the_byte():
    ext = vsl.tower_extent(vsl.find_tower_shard(MODEL_DIR))
    assert ext.pieces == TOWER_PIECES
    assert ext.byte_sum == TOWER_BYTES
    assert ext.first_offset == TOWER_FIRST
    assert ext.last_end == TOWER_LAST
    assert ext.contiguous
    assert ext.dtypes == ("BF16",)
    # THE OFFSET TRAP, pinned: the safetensors offsets are data-section
    # relative; the pread positions are 47_704 bytes further in.  Both spans
    # are 921_460_192, which is why mixing them reads as "close enough".
    assert ext.data_base == TOWER_FILE_FIRST - TOWER_FIRST == 47_704
    assert ext.file_first_offset == TOWER_FILE_FIRST
    assert ext.file_last_end == TOWER_FILE_LAST
    assert ext.file_last_end - ext.file_first_offset == ext.byte_sum


@needs_checkpoint
def test_the_extent_feeds_the_planners_tower_spec_directly():
    ext = vsl.tower_extent(vsl.find_tower_shard(MODEL_DIR))
    spec = vs.tower_from_span(**ext.as_span_args())
    assert spec.pieces == TOWER_PIECES
    assert spec.weight_bytes == TOWER_BYTES
    assert spec.weight_bytes / vs.GIB == pytest.approx(0.858, abs=0.001)


@needs_checkpoint
def test_the_meta_read_of_the_real_tower_costs_no_data_bytes():
    """333 shapes, zero payload -- the desk form of 'build the tower'."""
    sd = vsl.tower_state_dict(vsl.find_tower_shard(MODEL_DIR), meta=True)
    assert len(sd) == TOWER_PIECES
    assert all(t.device.type == "meta" for t in sd.values())
    # and the shapes agree with the config's arithmetic
    total = sum(t.numel() * t.element_size() for t in sd.values())
    assert total == TOWER_BYTES


# --------------------------------------------------------------- slice 3 --


class _Item:
    """The two fields of ``MultimodalDataItem`` this path touches
    (``schedule_batch.py:800`` for ``precomputed_embeddings``)."""

    def __init__(self):
        self.precomputed_embeddings = None
        self.feature = object()


def _emb(rows, width, cuda=False):
    import torch

    t = torch.zeros(rows, width, dtype=torch.bfloat16)
    if cuda:  # never actually allocated -- only the attribute is inspected
        class _Fake:
            shape = (rows, width)
            is_cuda = True

        return _Fake()
    return t


def test_the_handoff_sets_the_field_the_prefill_reads_before_the_tower():
    items = [_Item(), _Item()]
    rows = vsl.attach_precomputed_embeddings(
        items, [_emb(1024, 5120), _emb(256, 5120)], expected_width=5120
    )
    assert rows == 1280
    assert all(i.precomputed_embeddings is not None for i in items)
    # the raw pixels are dropped: leaving them would let a later pass re-embed
    assert all(i.feature is None for i in items)


def test_a_count_mismatch_refuses_by_name():
    with pytest.raises(vsl.VisionEmbeddingRefused) as e:
        vsl.attach_precomputed_embeddings(
            [_Item(), _Item()], [_emb(8, 5120)], expected_width=5120
        )
    assert "fluent wrong text" in str(e.value)


def test_an_empty_call_refuses_because_the_trigger_disagrees():
    with pytest.raises(vsl.VisionEmbeddingRefused) as e:
        vsl.attach_precomputed_embeddings([], [], expected_width=5120)
    assert "trigger and the request disagree" in str(e.value)


def test_a_wrong_width_refuses_and_names_the_deepstack_case():
    """4x5120 is exactly what a deepstack checkpoint's tower emits
    (encode_server.py:441-450); the serving checkpoint's list is EMPTY, so
    that width here means the wrong tower."""
    with pytest.raises(vsl.VisionEmbeddingRefused) as e:
        vsl.attach_precomputed_embeddings(
            [_Item()], [_emb(1024, 4 * 5120)], expected_width=5120
        )
    assert "deepstack" in str(e.value)


def test_a_non_2d_embedding_refuses_before_the_silent_reshape():
    import torch

    with pytest.raises(vsl.VisionEmbeddingRefused) as e:
        vsl.attach_precomputed_embeddings(
            [_Item()], [torch.zeros(4, 256, 5120)], expected_width=5120
        )
    assert "2-D" in str(e.value)


def test_an_embedding_still_on_the_stages_card_refuses():
    """The stage tears its card down.  An embedding left there is a dangling
    pointer with a shape."""
    with pytest.raises(vsl.VisionEmbeddingRefused) as e:
        vsl.attach_precomputed_embeddings(
            [_Item()], [_emb(8, 5120, cuda=True)], expected_width=5120
        )
    assert "outlives the tower" in str(e.value)


def test_the_expected_row_count_matches_the_planners_geometry():
    cfg = vs.VisionEncoderConfig()
    assert vsl.expected_embedding_rows(
        1024, 1024, patch_size=cfg.patch_size,
        spatial_merge_size=cfg.spatial_merge_size,
    ) == cfg.vision_tokens(1024, 1024) == 1024
