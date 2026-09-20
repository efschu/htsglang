# SPDX-License-Identifier: Apache-2.0
"""Task #58 -- the transient tower's attention backend, chosen from the card.

Metal boot xsn407 (20.09. 17:16Z, tree 16d39cbb1c) got further than any before
it: census, placement (card0), tower load all ran. The ENCODER FORWARD then
refused::

    W107 Weg2VisionEncodeFailed rid=weg2-8-27 -- VisionStageEncodeFailed:
    card0: the encoder forward over 1 item(s) raised: out of resource: shared
    memory, Required: 131072, Hardware limit: 101376. Reducing block sizes or
    num_stages may help.

WHY IT COULD NOT HAVE WORKED ON ANY CARD OF THIS RIG
-----------------------------------------------------
``VisionAttention._determine_attention_backend``
(``layers/attention/vision.py:1085``) special-cases exactly two capabilities --
major 9 -> ``fa3``, major 10 -> ``fa4`` -- and lets every other CUDA capability
fall through to ``triton_attn``. Both card families here fall through: the
3080 is sm_86, the 5090 is sm_120. And both admit 101376 B of opt-in shared
memory against the kernel's 131072 B. So the default backend fits NO CARD of
this rig, and the way that surfaced was a launch failure mid-encode rather than
a decision anyone could read.

The fix decides the backend BEFORE the module is built, out of
``shared_memory_per_block_optin`` -- torch's spelling of
``cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)``.

THE OTHER LEVER, AND WHY IT IS NOT TAKEN
-----------------------------------------
The driver's own hint says "Reducing block sizes or num_stages may help". It is
not available here: ``VisionTritonAttention.forward`` calls the SHARED prefill
kernel ``context_attention_fwd`` (``vision.py:378``) and exposes no block-size
knob. Halving those blocks would change the kernel the whole engine prefills
with -- not a transient tower's business. ``sdpa`` has no opt-in shared-memory
demand at all; its cost is a boolean mask ``[1, s, s]``
(``vision.py:209 _generate_mask_cache``), 16 MiB at the design's 4096 patch
rows, beside the 79 MiB of activation the arming line already books.
"""

from __future__ import annotations

import pytest

from sglang.srt.weg2.vision_stage_boot import (
    TRITON_BACKED_BACKENDS,
    TRITON_VISION_ATTN_SMEM_BYTES,
    VISION_BACKEND_FALLBACK,
    VisionStageBackendRefused,
    choose_vision_attention_backend,
)

#: The two numbers the driver printed on xsn407. Written down, not derived, so
#: a changed constant has to face them.
XSN407_REQUIRED = 131072
XSN407_HARDWARE_LIMIT = 101376

#: Every card of this rig reports this opt-in ceiling: the 3080s (sm_86) and --
#: per the `[nan-49c] marlin switches: smem_optin=101376(ok)` line -- the 5090
#: (sm_120) too.
RIG_SMEM_OPTIN = 101376


class TestTheConstantsAreTheMeasuredOnes:
    def test_the_triton_demand_is_the_number_the_driver_printed(self):
        assert TRITON_VISION_ATTN_SMEM_BYTES == XSN407_REQUIRED == 131072

    def test_the_fallback_is_not_triton_backed(self):
        assert VISION_BACKEND_FALLBACK not in TRITON_BACKED_BACKENDS

    def test_the_fallback_is_a_real_registered_backend(self):
        """A fallback that is not in the registry is a KeyError at load time."""
        from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL

        assert VISION_BACKEND_FALLBACK in QKV_BACKEND_IMPL
        for name in TRITON_BACKED_BACKENDS:
            assert name in QKV_BACKEND_IMPL


class TestTheChoiceAtEachSmemLimit:
    def test_this_rigs_cards_get_sdpa(self):
        """THE xsn407 REGRESSION, at the rig's real number."""
        backend, why = choose_vision_attention_backend(
            smem_optin_bytes=RIG_SMEM_OPTIN, card=0
        )
        assert backend == VISION_BACKEND_FALLBACK == "sdpa"
        # Both numbers in the reason, and the shortfall spelled out.
        assert str(RIG_SMEM_OPTIN) in why
        assert str(TRITON_VISION_ATTN_SMEM_BYTES) in why
        assert str(TRITON_VISION_ATTN_SMEM_BYTES - RIG_SMEM_OPTIN) in why

    def test_a_card_with_exactly_enough_gets_triton(self):
        backend, why = choose_vision_attention_backend(
            smem_optin_bytes=TRITON_VISION_ATTN_SMEM_BYTES, card=1
        )
        assert backend == "triton_attn"
        assert str(TRITON_VISION_ATTN_SMEM_BYTES) in why

    def test_one_byte_short_is_short(self):
        """The boundary is >=, and it is tested on both sides of one byte."""
        backend, _ = choose_vision_attention_backend(
            smem_optin_bytes=TRITON_VISION_ATTN_SMEM_BYTES - 1, card=2
        )
        assert backend == VISION_BACKEND_FALLBACK

    def test_a_roomier_card_gets_triton(self):
        backend, _ = choose_vision_attention_backend(
            smem_optin_bytes=227 * 1024, card=1
        )
        assert backend == "triton_attn"

    @pytest.mark.parametrize("bad", [None, 0, -1])
    def test_an_unreadable_ceiling_REFUSES_instead_of_defaulting(self, bad):
        """A kernel picked against an unknown ceiling is how xsn407 failed."""
        with pytest.raises(VisionStageBackendRefused) as ei:
            choose_vision_attention_backend(smem_optin_bytes=bad, card=0)
        assert "could not be read" in str(ei.value)
        assert "card0" in str(ei.value)

    def test_the_reason_never_leaves_a_number_out(self):
        for smem in (1024, RIG_SMEM_OPTIN, TRITON_VISION_ATTN_SMEM_BYTES, 1 << 20):
            _, why = choose_vision_attention_backend(smem_optin_bytes=smem, card=0)
            assert str(smem) in why
            assert str(TRITON_VISION_ATTN_SMEM_BYTES) in why
            assert "nan" not in why.lower()


class TestTheOperatorOverride:
    def test_a_non_triton_override_is_honoured_as_given(self):
        backend, why = choose_vision_attention_backend(
            smem_optin_bytes=RIG_SMEM_OPTIN, card=0, operator_override="fa3"
        )
        assert backend == "fa3"
        assert "honoured as given" in why

    def test_a_non_triton_override_is_honoured_even_with_an_unreadable_ceiling(self):
        """The ceiling is a Triton-kernel property; it must not gate others."""
        backend, _ = choose_vision_attention_backend(
            smem_optin_bytes=None, card=0, operator_override="sdpa"
        )
        assert backend == "sdpa"

    def test_a_triton_override_on_this_rigs_cards_REFUSES(self):
        """Honouring it blindly would reproduce xsn407 on purpose."""
        with pytest.raises(VisionStageBackendRefused) as ei:
            choose_vision_attention_backend(
                smem_optin_bytes=RIG_SMEM_OPTIN,
                card=0,
                operator_override="triton_attn",
            )
        text = str(ei.value)
        assert str(RIG_SMEM_OPTIN) in text
        assert str(TRITON_VISION_ATTN_SMEM_BYTES) in text
        assert "xsn407" in text
        assert VISION_BACKEND_FALLBACK in text  # tells the operator the way out

    def test_a_triton_override_on_a_card_that_fits_is_honoured(self):
        backend, _ = choose_vision_attention_backend(
            smem_optin_bytes=TRITON_VISION_ATTN_SMEM_BYTES,
            card=1,
            operator_override="triton_attn",
        )
        assert backend == "triton_attn"

    def test_a_triton_override_with_an_unreadable_ceiling_REFUSES(self):
        with pytest.raises(VisionStageBackendRefused):
            choose_vision_attention_backend(
                smem_optin_bytes=None, card=0, operator_override="triton_attn"
            )

    def test_an_empty_override_is_not_an_override(self):
        """`mm_attention_backend` is None when unset; "" must not slip past."""
        backend, _ = choose_vision_attention_backend(
            smem_optin_bytes=RIG_SMEM_OPTIN, card=0, operator_override=""
        )
        assert backend == VISION_BACKEND_FALLBACK


class TestTheRefusalIsMappedToAWCode:
    def test_it_is_a_load_refusal_so_code_for_gives_W106(self):
        from sglang.srt.planner.vision_stage_load import VisionStageLoadRefused
        from sglang.srt.weg2.vision_stage_service import W_LOAD, code_for

        assert issubclass(VisionStageBackendRefused, VisionStageLoadRefused)
        code, fatal = code_for(VisionStageBackendRefused("x"))
        assert code == W_LOAD
        assert fatal is False


class TestTheUpstreamDefaultIsWhatWeThinkItIs:
    """Pin the premise. If upstream starts handling sm_86/sm_120, this fails
    here rather than leaving a fallback nobody revisits."""

    def test_the_default_table_only_special_cases_sm90_and_sm100(self):
        import inspect

        from sglang.srt.layers.attention import vision

        src = inspect.getsource(vision.VisionAttention._determine_attention_backend)
        assert 'backend = "fa3"' in src and "major == 9" in src
        assert 'backend = "fa4"' in src and "major == 10" in src
        # everything else on CUDA falls through to triton
        assert 'backend = "triton_attn"' in src

    def test_the_triton_backend_delegates_to_the_shared_prefill_kernel(self):
        """Why block sizes are not a knob this stage can turn."""
        import inspect

        from sglang.srt.layers.attention import vision

        src = inspect.getsource(vision.VisionTritonAttention)
        assert "context_attention_fwd" in src
        assert "BLOCK" not in src, (
            "if a block-size knob appears on this class, the fallback decision "
            "should be revisited -- halving blocks may beat switching backends"
        )


class TestTheWiring:
    def test_the_builder_decides_before_it_builds(self):
        import inspect

        from sglang.srt.weg2 import vision_stage_boot as vsb

        src = inspect.getsource(vsb._build_qwen3vl_tower)
        assert "choose_vision_attention_backend(" in src
        assert "smem_optin_for_torch_index(torch_index)" in src
        assert "[vision-attn]" in src
        # the choice is applied while the module is constructed, not after
        assert src.index("choose_vision_attention_backend(") < src.index(
            "Qwen3VLMoeVisionModel("
        )
        assert "with _mm_attention_backend(backend):" in src

    def test_the_override_context_restores_the_previous_value(self):
        import types

        from sglang.srt.weg2 import vision_stage_boot as vsb

        fake = types.SimpleNamespace(mm_attention_backend="original")
        import sglang.srt.runtime_context as rc

        real = rc.get_server_args
        rc.get_server_args = lambda: fake
        try:
            with vsb._mm_attention_backend("sdpa"):
                assert fake.mm_attention_backend == "sdpa"
            assert fake.mm_attention_backend == "original"
            # and it restores on the way out of an exception, too
            with pytest.raises(RuntimeError):
                with vsb._mm_attention_backend("sdpa"):
                    raise RuntimeError("boom")
            assert fake.mm_attention_backend == "original"
        finally:
            rc.get_server_args = real

    def test_the_probe_returns_None_rather_than_a_fabricated_ceiling(self):
        """With no CUDA present it must be None -- the refusing value."""
        from sglang.srt.weg2.vision_stage_boot import smem_optin_for_torch_index

        assert smem_optin_for_torch_index(0) is None
