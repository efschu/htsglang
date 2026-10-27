"""Tensor truth-value traps (qwen load review 27.10.): ``x or ()`` on a ``torch.Tensor`` raises from two elements on.

``Req.prefix_indices`` (schedule_batch.py) and ``MatchResult.device_indices`` (base_prefix_cache.py) are int64 tensors.
Two readers still took their length through ``... or ()``: ``d_match_probe.match_numbers`` (no try/except: a hard
crash on the probe path) and ``mixed_d_short.stamp_chunked`` (inside a try/except: the stamp went silently
"unreadable", the request big for life). Both read the length with ``is None`` and ``len`` now.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import d_match_probe, mixed_d_short  # noqa: E402


def _req(prefix_len, seqlen, extend_range=None):
    return types.SimpleNamespace(
        prefix_indices=torch.arange(prefix_len, dtype=torch.int64),
        seqlen=seqlen,
        extend_range=extend_range,
    )


def test_match_numbers_reads_a_multi_element_tensor():
    result = types.SimpleNamespace(
        device_indices=torch.arange(7, dtype=torch.int64), host_hit_length=3, key_match_depth=9
    )
    assert d_match_probe.match_numbers(result) == (7, 3, 9)


def test_match_numbers_empty_and_missing_device_indices_are_zero():
    empty = types.SimpleNamespace(device_indices=torch.empty((0,), dtype=torch.int64), host_hit_length=0)
    assert d_match_probe.match_numbers(empty) == (0, 0, 0)
    assert d_match_probe.match_numbers(types.SimpleNamespace()) == (0, 0, 0)


def test_stamp_chunked_without_extend_range_uses_the_tensor_length():
    req = _req(prefix_len=5, seqlen=100)
    mixed_d_short.stamp_chunked(req)
    assert not getattr(req, mixed_d_short.ATTR_UNREADABLE, False)
    assert getattr(req, mixed_d_short.ATTR_TOTAL) == 95
