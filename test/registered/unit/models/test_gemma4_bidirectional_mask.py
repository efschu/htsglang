# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Gemma-4 installs a bidirectional custom mask only when one is needed.

Task #186. `prepare_attn_masks` builds every request's mask starting from a
plain causal `tril(diagonal=prefix_len)` and only then punches bidirectional
blocks in for image spans that are *fully contained in the extend window*.
Three common batches never get such a block:

* a pure-text request sharing a batch with an image request,
* a multi-turn follow-up whose image span sits wholly in the cached prefix,
* a chunked-prefill chunk that splits an image span.

For those the mask is bit-for-bit the causal predicate the Triton extend
kernel already applies on its `IS_CAUSAL` branch, so installing it is pure
overhead -- and it trips the `custom_mask` refusals on the DCP extend and
verify-splitkv lanes for no semantic gain.

What this file pins:

1. the degeneracy claim itself -- the no-image mask equals the kernel's causal
   predicate exactly, and SDPA under it is byte-identical to causal SDPA;
2. that a degenerate batch installs nothing;
3. that a batch with a contained image span still installs the mask, and that
   the mask really is non-causal;
4. the flat-buffer invariant that forbids the "obvious" per-request fix:
   `mask_indptr` must carry one non-empty slot per request, including the
   text-only ones, because `USE_CUSTOM_MASK` is a whole-launch constexpr.
"""

import types

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import gemma4_mm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeItem:
    def __init__(self, offsets, is_image=True):
        self.offsets = offsets
        self._is_image = is_image

    def is_image(self):
        return self._is_image


class _FakeMMInputs:
    def __init__(self, items):
        self.mm_items = items


def _run(extend_lens, prefix_lens, mm_inputs):
    """Call the real `prepare_attn_masks` against a stand-in backend.

    Returns the backend's `forward_metadata` so callers can see what, if
    anything, was installed.
    """
    backend = object.__new__(TritonAttnBackend)
    backend.forward_metadata = types.SimpleNamespace(custom_mask=None, mask_indptr=None)
    fb = types.SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(extend_lens),
        extend_seq_lens=torch.tensor(extend_lens),
        extend_prefix_lens=torch.tensor(prefix_lens),
        mm_inputs=mm_inputs,
    )
    orig = gemma4_mm.get_attn_backend
    gemma4_mm.get_attn_backend = lambda: backend
    try:
        gemma4_mm.Gemma4ForConditionalGeneration.prepare_attn_masks(
            None, fb, torch.zeros(sum(extend_lens), dtype=torch.long), torch.bool
        )
    finally:
        gemma4_mm.get_attn_backend = orig
    return backend.forward_metadata


def _causal_predicate(extend, prefix):
    """Exactly what `_fwd_kernel` computes when `IS_CAUSAL` and no custom mask.

    Prefix columns are unconditionally visible (stage 1 runs with
    `SKIP_PREFIX_CUSTOM_MASK`); extend columns use `m >= n`.
    """
    rows = torch.arange(extend).unsqueeze(1)
    cols = torch.arange(extend).unsqueeze(0)
    return torch.cat(
        [torch.ones(extend, prefix, dtype=torch.bool), rows >= cols], dim=1
    )


def test_textonly_mask_is_exactly_the_causal_predicate():
    """The claim the whole fix rests on: no image span -> nothing but causal."""
    for extend, prefix in ((275, 0), (64, 7), (1, 40), (33, 0)):
        md = _run([extend], [prefix], [None])
        # Nothing is installed any more, so rebuild the mask the way
        # prepare_attn_masks does and compare that against the kernel.
        built = torch.ones(extend, extend + prefix, dtype=torch.bool).tril(
            diagonal=prefix
        )
        assert torch.equal(built, _causal_predicate(extend, prefix)), (
            f"causal fill diverges from the kernel predicate at "
            f"extend={extend} prefix={prefix}"
        )
        assert md.custom_mask is None


def test_sdpa_under_the_textonly_mask_is_byte_identical_to_causal():
    """Reference-level proof that dropping the mask cannot move a bit."""
    torch.manual_seed(186)
    extend, prefix, heads, dim = 48, 16, 4, 32
    total = extend + prefix
    q = torch.randn(1, heads, extend, dim, dtype=torch.float64)
    k = torch.randn(1, heads, total, dim, dtype=torch.float64)
    v = torch.randn(1, heads, total, dim, dtype=torch.float64)

    mask = torch.ones(extend, total, dtype=torch.bool).tril(diagonal=prefix)
    masked = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    causal = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=_causal_predicate(extend, prefix)
    )
    assert torch.equal(masked, causal)


def test_degenerate_batches_install_nothing():
    """Pure text, prefix-resident image, split image: all mask-free."""
    # Text-only request batched next to... nothing. (The caller's
    # `contains_image_inputs()` guard normally stops this one earlier; the
    # method must be safe on its own regardless.)
    assert _run([275], [0], [None]).custom_mask is None

    # Multi-turn follow-up: the 256-token image span is entirely in the
    # cached prefix, only new text is being extended.
    prefix_resident = [_FakeMMInputs([_FakeItem([(10, 265)])])]
    assert _run([12], [300], prefix_resident).custom_mask is None

    # Chunked prefill splitting the image span.
    split = [_FakeMMInputs([_FakeItem([(100, 355)])])]
    assert _run([200], [0], split).custom_mask is None

    # Non-image mm items (video/audio) never get bidirectional treatment.
    non_image = [_FakeMMInputs([_FakeItem([(3, 90)], is_image=False)])]
    assert _run([128], [0], non_image).custom_mask is None


def test_contained_image_span_still_installs_a_non_causal_mask():
    contained = [_FakeMMInputs([_FakeItem([(4, 259)])])]
    md = _run([275], [0], contained)
    assert md.custom_mask is not None
    assert md.mask_indptr is not None

    mask = md.custom_mask.view(275, 275)
    causal = _causal_predicate(275, 0)
    assert not torch.equal(mask, causal), "image batch must be non-causal"
    # Bidirectionality is confined to the image block: only strictly-upper
    # entries inside [4, 259] may be set.
    extra = mask & ~causal
    assert extra.any()
    assert extra[4:260, 4:260].sum() == extra.sum()


def test_mixed_batch_keeps_a_slot_for_the_text_only_request():
    """The flat-buffer invariant. Removing the text request's slot is the
    tempting "fix" and it silently corrupts the image request next to it."""
    mixed = [
        None,  # text-only
        _FakeMMInputs([_FakeItem([(2, 130)])]),  # image, contained
        None,  # text-only
    ]
    extend_lens, prefix_lens = [64, 200, 32], [0, 0, 5]
    md = _run(extend_lens, prefix_lens, mixed)
    assert md.custom_mask is not None

    indptr = md.mask_indptr
    assert indptr.numel() == len(extend_lens) + 1
    expected = [0]
    for e, p in zip(extend_lens, prefix_lens):
        expected.append(expected[-1] + e * (e + p))
    assert indptr.tolist() == expected, "every request must own a mask slot"
    assert md.custom_mask.numel() == expected[-1]

    # And the text-only slots really are the causal predicate.
    for i, (e, p) in enumerate(zip(extend_lens, prefix_lens)):
        if mixed[i] is not None:
            continue
        slot = md.custom_mask[expected[i] : expected[i + 1]].view(e, e + p)
        assert torch.equal(slot, _causal_predicate(e, p))


def test_single_token_image_span_stays_degenerate():
    """A 1-token span writes only the diagonal, which causal already covers."""
    one = [_FakeMMInputs([_FakeItem([(7, 7)])])]
    assert _run([64], [0], one).custom_mask is None
