"""#1418: the after-part digest of `weights_draft` never reads a piece that
aliases the target's `weights` region (NO-WRITE pieces AND name-shared
measured shares); `weights` grades them all."""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager as WeightUpdater  # noqa: E402


def _idn(tag, name):
    return types.SimpleNamespace(tag=tag, name=name)


INV = [
    (_idn("weights", "lm_head.weight"), 1),
    (_idn("weights", "model.layers.0.q.weight"), 2),
    (_idn("weights_draft", "lm_head.weight"), 3),            # name-shared measured share
    (_idn("weights_draft", "model.embed_tokens.weight"), 4),  # NO-WRITE alias
    (_idn("weights_draft", "fc.weight"), 5),                  # the draft's own bytes
    (_idn("weights_3", "model.layers.30.q.weight"), 6),
]
NW = {("weights_draft", "model.embed_tokens.weight")}
W_NAMES = {"lm_head.weight", "model.layers.0.q.weight"}


def _names(items):
    return sorted((str(i.tag), str(i.name)) for i, _ in items)


def test_draft_part_reads_only_its_own_bytes():
    items = WeightUpdater._weg2_seam_after_part_items("weights_draft", INV, NW, W_NAMES)
    assert _names(items) == [("weights_draft", "fc.weight")]


def test_weights_part_grades_its_own_and_every_alias():
    items = WeightUpdater._weg2_seam_after_part_items("weights", INV, NW, W_NAMES)
    assert _names(items) == [
        ("weights", "lm_head.weight"),
        ("weights", "model.layers.0.q.weight"),
        ("weights_draft", "lm_head.weight"),
        ("weights_draft", "model.embed_tokens.weight"),
    ]


def test_a_layer_tag_is_untouched():
    items = WeightUpdater._weg2_seam_after_part_items("weights_3", INV, NW, W_NAMES)
    assert _names(items) == [("weights_3", "model.layers.30.q.weight")]
