# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""#3287: the ``.update()`` callers must supply the prefix mirror they promise.

#616h removed an unbounded blocking device-to-host read from the uneven-DCP
extend index build by adding an ``extend_prefix_lens_cpu`` channel. #623 wired
the updater ARMS. This file pins the layer above them: the
``FlashInferAttnBackend.init_forward_metadata_out_graph`` callsites that invoke
``indices_updater_prefill.update(...)``.

WHY A NEW PIN. ``test_dcp_index_host_sum_623.py`` already carries a census, but
it is a source-text scan that asserts each builder callsite NAMES
``total_tokens=``. It says nothing about whether the argument can ever be
non-None at runtime -- which is the #616h failure mode itself, "a wired channel
with no caller supplying it", recurring one layer out. A pin of that shape
cannot catch a caller that passes nothing, so this one is behavioural: it drives
the real method and inspects the kwargs that actually arrive.

THE DEFECT IT CAUGHT. Two of the six callsites reached the consuming branch
(``call_begin_forward``'s ``spec_info is None and self.attn_backend.uneven_dcp``
arm) without the mirror, so both fell back to the device read they were supposed
to remove -- ``owner.py`` ``int(full_indptr[bs].item())`` on the weighted rule,
``int(dcp_lens.sum().item())`` on the even one:

  * the dLLM extend graph replay
  * plain EXTEND under the full prefill CUDA graph

MIRROR THE EXPRESSION, NOT THE NAME. The consuming branch indexes over
``prefix_lens``, so the mirror must be the host twin of whatever ``prefix_lens``
IS at that callsite -- the parameter name ``extend_prefix_lens_cpu`` describes
its usual source, not a contract. At the full-prefill-CG site ``prefix_lens`` is
``forward_batch.extend_prefix_lens``, so the mirror is
``forward_batch.extend_prefix_lens_cpu``. At the dLLM site it is NOT: the
callsite computes ``seq_lens - block_size`` on the device, so forwarding
``extend_prefix_lens_cpu`` there would hand the builder a mirror of a DIFFERENT
vector -- a silent mis-size, strictly worse than the stall. The mirror is
derived by the same subtraction on the host instead, which is also why the dLLM
site needs no new field on the decode replay view.

These tests assert the VALUE, not merely presence: each mirror must equal the
host twin of the device ``prefix_lens`` handed to the same call. A fix that
forwarded the wrong-but-non-None vector would pass a presence check and fail
here.

Hermetic: no CUDA, no process group, no model.
"""

import types

import pytest
import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

_DLLM_BLOCK = 8

# The device vector and the host mirror are given DIFFERENT numbers so that a
# mirror sourced from the wrong vector cannot pass by coincidence.
_SEQ_LENS = [41214, 39166]


class _Sentinel(Exception):
    """Raised from the recorder to stop before any device work."""


def _forward_mode(mode: str):
    return types.SimpleNamespace(
        is_decode_or_idle=lambda: mode == "decode",
        is_target_verify=lambda: mode == "target_verify",
        is_dllm_extend=lambda: mode == "dllm_extend",
        is_draft_extend_v2=lambda: mode == "draft_extend_v2",
        # Every one of the three extend-family modes also answers is_extend();
        # the dispatch relies on branch ORDER, so the fake must reproduce that
        # rather than pretend the modes are disjoint.
        is_extend=lambda: mode in ("dllm_extend", "draft_extend_v2", "extend"),
    )


def _backend(captured):
    """A FlashInferAttnBackend stand-in carrying only what the dispatch reads."""

    def _recorder(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise _Sentinel()

    fake = types.SimpleNamespace(
        uneven_dcp=True,
        use_paged=True,
        disable_cuda_graph_kv_split=False,
        dllm_config=types.SimpleNamespace(block_size=_DLLM_BLOCK),
        prefill_cuda_graph_metadata={"k": [None]},
        draft_extend_cuda_graph_metadata={len(_SEQ_LENS): [None]},
        full_cg_prefill_wrappers=[None],
        indices_updater_prefill=types.SimpleNamespace(update=_recorder),
        indices_updater_decode=types.SimpleNamespace(update=_recorder),
        _ragged_wrapper_override=None,
    )
    fake._verify_cg_key = lambda bs, forward_mode, spec_info: "k"
    fake.init_forward_metadata_out_graph = types.MethodType(
        FlashInferAttnBackend.init_forward_metadata_out_graph, fake
    )
    return fake


def _forward_batch(mode: str):
    bs = len(_SEQ_LENS)
    seq_lens = torch.tensor(_SEQ_LENS, dtype=torch.int32)
    # A prefix genuinely different from both seq_lens and seq_lens - block, so
    # the three candidate mirrors are three distinct vectors.
    extend_prefix_lens = torch.tensor([31000, 29000], dtype=torch.int32)
    return types.SimpleNamespace(
        batch_size=bs,
        req_pool_indices=torch.arange(bs, dtype=torch.int32),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        seq_lens_sum=int(seq_lens.sum()),
        encoder_lens=None,
        forward_mode=_forward_mode(mode),
        spec_info=None,
        extend_prefix_lens=extend_prefix_lens,
        extend_prefix_lens_cpu=extend_prefix_lens.tolist(),
        kv_session_spill_tick=False,
        positions=torch.zeros(bs, dtype=torch.int32),
    )


def _drive(mode: str):
    captured = {}
    fake = _backend(captured)
    fb = _forward_batch(mode)
    with pytest.raises(_Sentinel):
        fake.init_forward_metadata_out_graph(fb, in_capture=False)
    return captured, fb


def _as_list(x):
    return x.tolist() if isinstance(x, torch.Tensor) else list(x)


@pytest.mark.parametrize("mode", ["dllm_extend", "extend"])
def test_the_prefix_mirror_reaches_the_consuming_extend_branch(mode):
    """Both spec_info=None graph callsites must supply the mirror.

    Without it the uneven-DCP extend branch falls back to
    ``int(full_indptr[bs].item())`` (weighted) or ``int(dcp_lens.sum().item())``
    (even) -- the unbounded blocking D2H inside the collective window that the
    channel exists to remove.
    """
    captured, _ = _drive(mode)
    mirror = captured["kwargs"].get("extend_prefix_lens_cpu")
    assert mirror is not None, (
        f"{mode}: init_forward_metadata_out_graph called "
        "indices_updater_prefill.update() without extend_prefix_lens_cpu, so "
        "the uneven-DCP extend branch has no host total and takes the "
        "blocking device read"
    )


@pytest.mark.parametrize("mode", ["dllm_extend", "extend"])
def test_the_prefix_mirror_describes_the_prefix_lens_actually_passed(mode):
    """The mirror must be the host twin of THIS callsite's prefix_lens.

    The consuming branch indexes over ``prefix_lens``. A mirror of some other
    vector is a silent mis-size of the index buffer -- worse than the stall it
    replaces, and invisible to any check that only tests for non-None.
    """
    captured, _ = _drive(mode)
    prefix_lens = captured["kwargs"].get("prefix_lens")
    if prefix_lens is None:
        prefix_lens = captured["args"][5]
    mirror = captured["kwargs"].get("extend_prefix_lens_cpu")
    assert mirror is not None, f"{mode}: no mirror forwarded"
    assert _as_list(mirror) == _as_list(prefix_lens), (
        f"{mode}: mirror {_as_list(mirror)} does not describe the device "
        f"prefix_lens {_as_list(prefix_lens)} handed to the same call"
    )


def test_the_dllm_mirror_is_not_the_extend_prefix_vector():
    """The trap this file exists to keep shut.

    At the dLLM callsite ``prefix_lens`` is ``seq_lens - block_size``, not
    ``forward_batch.extend_prefix_lens``. Forwarding the latter would satisfy
    a presence check while describing a different vector entirely.
    """
    captured, fb = _drive("dllm_extend")
    mirror = captured["kwargs"].get("extend_prefix_lens_cpu")
    assert mirror is not None, "dllm_extend: no mirror forwarded"
    assert _as_list(mirror) != fb.extend_prefix_lens_cpu, (
        "dllm_extend: the mirror is forward_batch.extend_prefix_lens_cpu, "
        "which is NOT what this callsite passes as prefix_lens"
    )
    assert _as_list(mirror) == [s - _DLLM_BLOCK for s in _SEQ_LENS]


def test_the_wired_sibling_still_forwards_its_mirror():
    """Control: the eager extend site (#616h) is unchanged by this fix."""
    captured, fb = _drive("extend")
    assert captured["kwargs"].get("extend_prefix_lens_cpu") is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
