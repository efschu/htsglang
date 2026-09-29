# SPDX-License-Identifier: Apache-2.0
"""Config classes for the Qwen3-TTS talker lane (#488).

Three nested configs, keyed strictly on ``model_type`` (#497 canon): nothing
here reads a version string out of the model name and nothing hardcodes a
geometry constant, so a differently-sized Qwen3-TTS checkpoint that reuses
``model_type: qwen3_tts`` loads with no code change.

**The M-RoPE key trap, gated here and nowhere else.** The checkpoint writes

    "rope_scaling": {"interleaved": true, "mrope_section": [24, 20, 20], ...}

while the runtime's rotary factory reads a DIFFERENT key
(``layers/rotary_embedding/factory.py``)::

    mrope_interleaved=rope_scaling.get("mrope_interleaved", False)

Passed through unchanged the factory silently builds NON-interleaved M-RoPE.
Nothing raises: the model loads, runs at full speed, emits plausible codec
tokens, and the audio comes out with subtly wrong prosody and timbre. It is the
cheapest bug in this bring-up and the most expensive to diagnose backwards from
a waveform. So the mapping is not a helper callers may remember to use -- it is
an assert that fires at config construction, before a weight is touched.

The predicate is deliberately duplicated from
``srt/translator/talker_config.py`` rather than imported: the translator's
in-process rung (the #286-ledgered module path) and this lane must not be able
to drift into the trap independently, and a shared import would make the lane
depend on the translator package. ``test/registered/unit/models/
test_qwen3_tts_talker_lane_488.py`` drives BOTH against the same fixture.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from transformers import PretrainedConfig

__all__ = [
    "CHECKPOINT_INTERLEAVED_KEY",
    "FACTORY_INTERLEAVED_KEY",
    "MRopeMappingError",
    "Qwen3TTSCodePredictorConfig",
    "Qwen3TTSConfig",
    "Qwen3TTSTalkerConfig",
    "normalize_rope_scaling",
    "assert_mrope_mapped",
]

#: What the checkpoint writes.
CHECKPOINT_INTERLEAVED_KEY = "interleaved"
#: What ``layers/rotary_embedding/factory.py`` actually reads.
FACTORY_INTERLEAVED_KEY = "mrope_interleaved"


class MRopeMappingError(ValueError):
    """A rope_scaling dict that would build the wrong rotary embedding."""


def normalize_rope_scaling(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Rewrite a checkpoint ``rope_scaling`` into the form the factory reads.

    Copies ``interleaved`` onto ``mrope_interleaved``; everything else passes
    through untouched. Raises when both keys are present and disagree -- that
    is a hand-edited config, and picking a winner silently is exactly the
    failure this module exists to prevent.
    """
    if raw is None:
        raise MRopeMappingError("rope_scaling is None; the talker requires M-RoPE")
    scaling = dict(raw)

    checkpoint_value = scaling.get(CHECKPOINT_INTERLEAVED_KEY)
    factory_value = scaling.get(FACTORY_INTERLEAVED_KEY)
    if checkpoint_value is not None and factory_value is not None:
        if bool(checkpoint_value) != bool(factory_value):
            raise MRopeMappingError(
                f"rope_scaling has {CHECKPOINT_INTERLEAVED_KEY}="
                f"{checkpoint_value!r} but {FACTORY_INTERLEAVED_KEY}="
                f"{factory_value!r}; refusing to guess which one is meant"
            )
    elif checkpoint_value is not None:
        scaling[FACTORY_INTERLEAVED_KEY] = bool(checkpoint_value)
    return scaling


def assert_mrope_mapped(
    scaling: Mapping[str, Any],
    head_dim: int,
    source: Optional[Mapping[str, Any]] = None,
) -> None:
    """Hard gate. Raises unless ``scaling`` builds the intended rotary.

    Three checks, one silent-failure mode behind each:

    1. ``mrope_section`` present -- without it the factory falls through to
       plain rotary and the three positional axes collapse into one;
    2. the section sums to ``head_dim / 2``, or the factory's auto-correction
       rewrites it and the positions stop meaning what the checkpoint trained;
    3. if the ORIGINAL config asked for interleaving, the normalised dict
       carries it under the key the factory reads. This is the trap.
    """
    if "mrope_section" not in scaling:
        raise MRopeMappingError(
            "rope_scaling has no mrope_section; the factory would build plain "
            "rotary and silently collapse the three M-RoPE position axes"
        )
    section = list(scaling["mrope_section"])
    expected = head_dim // 2
    if sum(section) != expected:
        raise MRopeMappingError(
            f"mrope_section {section} sums to {sum(section)}, expected "
            f"{expected} (= head_dim {head_dim} / 2); the factory would "
            "auto-correct it and the positions would stop matching training"
        )
    origin = source if source is not None else scaling
    wanted = origin.get(CHECKPOINT_INTERLEAVED_KEY)
    if wanted is None:
        wanted = origin.get(FACTORY_INTERLEAVED_KEY)
    if bool(wanted) and not bool(scaling.get(FACTORY_INTERLEAVED_KEY, False)):
        raise MRopeMappingError(
            f"the checkpoint asks for interleaved M-RoPE "
            f"({CHECKPOINT_INTERLEAVED_KEY}=True) but the normalised "
            f"rope_scaling does not carry {FACTORY_INTERLEAVED_KEY}. Passed to "
            "the rotary factory as-is this builds NON-interleaved M-RoPE: it "
            "loads, runs, emits plausible codec tokens, and sounds wrong. "
            "Run the dict through normalize_rope_scaling() first."
        )


class Qwen3TTSCodePredictorConfig(PretrainedConfig):
    """The depth transformer that expands one audio frame into its residual
    codebook entries.

    NOT a sequence model in the runtime's sense: its ``num_code_groups - 1``
    steps all sit at ONE talker sequence position. Its cache never touches the
    paged KV pool -- see the model file.
    """

    model_type = "qwen3_tts_talker_code_predictor"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        num_code_groups: int = 16,
        vocab_size: int = 2048,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 65536,
        attention_bias: bool = False,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_code_groups = num_code_groups
        self.vocab_size = vocab_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.attention_bias = attention_bias
        super().__init__(**kwargs)


class Qwen3TTSTalkerConfig(PretrainedConfig):
    """The autoregressive trunk: one decode step per audio frame.

    ``rope_scaling`` is normalised and gated in ``__init__``, so there is no
    path that produces a talker config carrying the M-RoPE trap.
    """

    model_type = "qwen3_tts_talker"
    sub_configs = {"code_predictor_config": Qwen3TTSCodePredictorConfig}

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        vocab_size: int = 3072,
        num_code_groups: int = 16,
        text_hidden_size: int = 2048,
        text_vocab_size: int = 151936,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        position_id_per_seconds: int = 13,
        attention_bias: bool = False,
        codec_bos_id: int = 2149,
        codec_eos_token_id: int = 2150,
        codec_pad_id: int = 2148,
        code_predictor_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.num_code_groups = num_code_groups
        self.text_hidden_size = text_hidden_size
        self.text_vocab_size = text_vocab_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.position_id_per_seconds = position_id_per_seconds
        self.attention_bias = attention_bias
        self.codec_bos_id = codec_bos_id
        self.codec_eos_token_id = codec_eos_token_id
        self.codec_pad_id = codec_pad_id

        # The gate. Runs before any weight is touched, on every construction
        # path (from_pretrained, from_dict, and a plain kwargs build).
        normalised = normalize_rope_scaling(rope_scaling)
        assert_mrope_mapped(normalised, self.head_dim, source=rope_scaling)
        self.rope_scaling = normalised

        if isinstance(code_predictor_config, Qwen3TTSCodePredictorConfig):
            self.code_predictor_config = code_predictor_config
        else:
            self.code_predictor_config = Qwen3TTSCodePredictorConfig(
                **(code_predictor_config or {})
            )
        super().__init__(**kwargs)


class Qwen3TTSConfig(PretrainedConfig):
    """Top-level Qwen3-TTS config.

    Only the ``talker_config`` half is lane-owned. The speaker encoder
    (ECAPA-TDNN, ~9 MiB) and the 12 Hz codec/vocoder under ``speech_tokenizer/``
    are NOT sharded and NOT part of the model lane: both run once per turn, not
    once per decode step, so they carry no share of the real-time factor. They
    stay in-process modules with their own #286 ledger asset classes, exactly as
    ``srt/translator/inprocess_tts.py`` registers them today.
    """

    model_type = "qwen3_tts"
    sub_configs = {"talker_config": Qwen3TTSTalkerConfig}

    def __init__(
        self,
        talker_config: Optional[Dict[str, Any]] = None,
        speaker_encoder_config: Optional[Dict[str, Any]] = None,
        tokenizer_type: str = "qwen3_tts_tokenizer_12hz",
        tts_bos_token_id: int = 151672,
        tts_eos_token_id: int = 151673,
        tts_pad_token_id: int = 151671,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        assistant_token_id: int = 77091,
        **kwargs,
    ) -> None:
        if isinstance(talker_config, Qwen3TTSTalkerConfig):
            self.talker_config = talker_config
        else:
            self.talker_config = Qwen3TTSTalkerConfig(**(talker_config or {}))
        self.speaker_encoder_config = dict(speaker_encoder_config or {})
        self.tokenizer_type = tokenizer_type
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.assistant_token_id = assistant_token_id

        # Mirror the trunk geometry onto the top level. The runtime's
        # ModelConfig probes hidden_size / num_hidden_layers / head counts on
        # the object it is handed, and a talker-only checkpoint has no separate
        # text_config for it to descend into.
        talker = self.talker_config
        self.hidden_size = talker.hidden_size
        self.num_hidden_layers = talker.num_hidden_layers
        self.num_attention_heads = talker.num_attention_heads
        self.num_key_value_heads = talker.num_key_value_heads
        self.head_dim = talker.head_dim
        self.intermediate_size = talker.intermediate_size
        self.vocab_size = talker.vocab_size
        self.max_position_embeddings = talker.max_position_embeddings
        self.rms_norm_eps = talker.rms_norm_eps
        self.rope_theta = talker.rope_theta
        self.rope_scaling = talker.rope_scaling
        super().__init__(**kwargs)
