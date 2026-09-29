# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Talker geometry, and the M-RoPE key trap that must never ship silently.

**The trap, stated once so it is never rediscovered.** The Qwen3-TTS checkpoint
writes its interleaving flag as::

    "rope_scaling": {"interleaved": true, "mrope_section": [24, 20, 20], ...}

The runtime's rotary factory reads a *different* key
(``layers/rotary_embedding/factory.py``)::

    mrope_interleaved=rope_scaling.get("mrope_interleaved", False)

Pass the checkpoint dict through unchanged and the factory silently builds
NON-interleaved M-RoPE. Nothing raises. The model loads, runs at full speed,
emits perfectly plausible codec tokens, and the audio comes out wrong -- and
"wrong" here means subtly wrong prosody and timbre, which is the hardest
possible signal to debug backwards from a waveform, in a language the operator
may not speak, on a phone, in Spain.

It is the cheapest bug on the project and the most expensive to diagnose. So
the mapping is not a helper that callers may remember to use: it is an assert
that fires at config construction, before a single weight is touched.

The same normalisation serves both rungs of the audio-out ladder (see
DESIGN_466 §12): the in-process rung and the future native-lane rung consume
identical geometry, so neither can drift into the trap independently.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    "MRopeMappingError",
    "TalkerGeometry",
    "FACTORY_INTERLEAVED_KEY",
    "CHECKPOINT_INTERLEAVED_KEY",
    "normalize_rope_scaling",
    "assert_mrope_mapped",
    "read_talker_geometry",
    "ShapeContractError",
    "assert_prompt_block",
    "assert_position_contract",
    "assert_rotary_contract",
]

#: What the checkpoint writes.
CHECKPOINT_INTERLEAVED_KEY = "interleaved"
#: What the rotary factory actually reads.
FACTORY_INTERLEAVED_KEY = "mrope_interleaved"


class MRopeMappingError(ValueError):
    """A rope_scaling dict that would build the wrong rotary embedding."""


def normalize_rope_scaling(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Rewrite a checkpoint ``rope_scaling`` into the form the factory reads.

    Concretely: copy ``interleaved`` onto ``mrope_interleaved``. Everything
    else is passed through untouched, because the factory understands the rest
    (``mrope_section``, ``rope_type``, ``type``) as written.

    Raises when the two keys are both present and disagree -- that is a
    corrupted or hand-edited config, and picking a winner silently is exactly
    the failure this module exists to prevent.
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
    """Hard gate. Raises unless ``scaling`` will build the intended rotary.

    Three checks, each of which has a silent-failure mode behind it:

    1. ``mrope_section`` must be present -- without it the factory falls through
       to plain :class:`RotaryEmbedding` and the three positional axes collapse
       into one;
    2. the section must sum to ``head_dim / 2``, or the factory's
       auto-correction path rewrites it and the positions no longer mean what
       the checkpoint trained;
    3. if the ORIGINAL config asked for interleaving, the normalised dict must
       actually carry it under the key the factory reads. This is the trap.
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


@dataclasses.dataclass(frozen=True)
class TalkerGeometry:
    """Everything the talker's shape depends on, read from the checkpoint.

    Deliberately a plain dataclass rather than a transformers config: both
    rungs consume it, and neither should have to agree on a config class to
    agree on a geometry.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    #: Already normalised: safe to hand to the rotary factory.
    rope_scaling: Dict[str, Any]
    #: Residual codebooks per audio frame. One decode step emits this many
    #: codes at ONE sequence position.
    num_code_groups: int
    #: Codec vocabulary of the first codebook (the talker's own head).
    codec_vocab_size: int
    #: Text side, for the conditioning projection.
    text_hidden_size: int
    text_vocab_size: int
    #: Frame rate scaffolding.
    position_id_per_seconds: int
    #: Control ids. None of these are derivable; they come from the config.
    codec_bos_id: int
    codec_eos_id: int
    codec_pad_id: int
    #: The depth transformer that expands one frame into its residual codes.
    code_predictor_layers: int
    code_predictor_hidden_size: int
    code_predictor_vocab_size: int

    @property
    def frame_rate_hz(self) -> float:
        return float(self.position_id_per_seconds)

    def codes_per_second(self) -> float:
        """Total codes emitted per second of audio, across all codebooks."""
        return self.frame_rate_hz * self.num_code_groups

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def read_talker_geometry(model_dir: Path) -> TalkerGeometry:
    """Load and VALIDATE the talker geometry from a checkpoint directory.

    The normalisation and the assert both run here, so there is no path that
    produces a geometry object carrying the trap.
    """
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"no config.json under {model_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        talker = config["talker_config"]
    except KeyError as exc:
        raise MRopeMappingError(
            f"{config_path} has no talker_config; this is not a Qwen3-TTS "
            "checkpoint"
        ) from exc

    head_dim = int(
        talker.get("head_dim")
        or talker["hidden_size"] // talker["num_attention_heads"]
    )
    raw_scaling = talker.get("rope_scaling")
    scaling = normalize_rope_scaling(raw_scaling)
    assert_mrope_mapped(scaling, head_dim, source=raw_scaling)

    predictor = talker.get("code_predictor_config", {})
    return TalkerGeometry(
        hidden_size=int(talker["hidden_size"]),
        num_hidden_layers=int(talker["num_hidden_layers"]),
        num_attention_heads=int(talker["num_attention_heads"]),
        num_key_value_heads=int(talker["num_key_value_heads"]),
        head_dim=head_dim,
        vocab_size=int(talker["vocab_size"]),
        intermediate_size=int(talker["intermediate_size"]),
        rms_norm_eps=float(talker["rms_norm_eps"]),
        rope_theta=float(talker["rope_theta"]),
        max_position_embeddings=int(talker["max_position_embeddings"]),
        rope_scaling=scaling,
        num_code_groups=int(talker["num_code_groups"]),
        codec_vocab_size=int(talker["vocab_size"]),
        text_hidden_size=int(talker["text_hidden_size"]),
        text_vocab_size=int(talker["text_vocab_size"]),
        position_id_per_seconds=int(talker["position_id_per_seconds"]),
        codec_bos_id=int(talker["codec_bos_id"]),
        codec_eos_id=int(talker["codec_eos_token_id"]),
        codec_pad_id=int(talker["codec_pad_id"]),
        code_predictor_layers=int(predictor.get("num_hidden_layers", 0)),
        code_predictor_hidden_size=int(predictor.get("hidden_size", 0)),
        code_predictor_vocab_size=int(predictor.get("vocab_size", 0)),
    )


def factory_would_interleave(rope_scaling: Mapping[str, Any]) -> bool:
    """Exactly what the rotary factory decides, reproduced for tests.

    Mirrors ``factory.py``'s read verbatim rather than paraphrasing it, so the
    falsifier below is testing the real predicate and not a restatement of the
    fix. If the factory's key ever changes, this is the one place to update and
    the falsifier will catch the drift.
    """
    return bool(rope_scaling.get(FACTORY_INTERLEAVED_KEY, False))


def describe_trap() -> Tuple[str, str]:
    """The two key names, for error messages and docs. Single source of truth."""
    return CHECKPOINT_INTERLEAVED_KEY, FACTORY_INTERLEAVED_KEY


# ---------------------------------------------------------------------------
# Shape contracts
# ---------------------------------------------------------------------------
#
# The class of bug these exist for: a tensor that is the right SIZE but the
# wrong SHAPE travels a long way before anything complains. The decode-step
# rotary bug is the worked example -- a (1, 1, 1024) query rotated against
# full-length cos/sin silently became eleven positions, and the first symptom
# was a matmul mismatch in o_proj, thirty layers and one module away from the
# cause. Prefill was unaffected throughout, which is why every earlier check
# passed.
#
# Each contract below is cheap enough to assert on every call and names the
# invariant it protects, so the next member of this family fails at the seam
# instead of downstream.


class ShapeContractError(ValueError):
    """A prompt or position tensor whose shape breaks a stated invariant."""


def _shape_of(tensor: object) -> Tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise ShapeContractError(f"expected a tensor, got {type(tensor).__name__}")
    return tuple(int(d) for d in shape)


def assert_prompt_block(
    name: str,
    tensor: object,
    hidden_size: int,
    batch: int = 1,
    min_length: int = 1,
) -> Tuple[int, ...]:
    """A prompt building block must be ``(batch, length, hidden_size)``.

    Every piece the talker's input is assembled from -- the role embeddings,
    the projected text, the speaker prompt, the codec prompt, the BOS/EOS
    markers -- has this shape. Flattening any of them to ``(batch, length *
    hidden)`` keeps the element count identical, which is exactly why a
    concatenation of the wrong-shaped block still succeeds and the failure
    appears much later.
    """
    shape = _shape_of(tensor)
    if len(shape) != 3:
        raise ShapeContractError(
            f"prompt block {name!r} has shape {shape}; expected 3 dimensions "
            f"(batch, length, hidden={hidden_size}). A 2-D block here is the "
            "flattening failure mode: the element count still matches, so a "
            "later concatenation succeeds and the error surfaces in an "
            "unrelated matmul."
        )
    if shape[0] != batch:
        raise ShapeContractError(
            f"prompt block {name!r} has batch {shape[0]}, expected {batch}"
        )
    if shape[2] != hidden_size:
        raise ShapeContractError(
            f"prompt block {name!r} has hidden size {shape[2]}, expected "
            f"{hidden_size}"
        )
    if shape[1] < min_length:
        raise ShapeContractError(
            f"prompt block {name!r} has length {shape[1]}, expected at least "
            f"{min_length}"
        )
    return shape


def assert_position_contract(
    position_ids: object,
    query_length: int,
    cache_length: Optional[int] = None,
    axes: int = 3,
) -> None:
    """Positions must describe the QUERY, never the whole cache.

    The invariant, and the reason it is worth an assert on every step:
    M-RoPE turns ``position_ids`` into ``cos``/``sin`` of that length, and the
    rotation is element-wise against the query. Hand it full-length positions
    on a one-token decode step and the query BROADCASTS to the cache length
    instead of failing -- so attention silently returns one row per cached
    position, and the mismatch only becomes visible when ``o_proj`` receives
    ``cache_length * head_dim * heads`` features.

    Prefill hides this completely, because there query length and cache length
    are equal. That asymmetry is what makes the bug survive every
    prefill-shaped test.

    ``axes`` is M-RoPE's section count (3 for this checkpoint: the position
    tensor is ``(axes, batch, length)``).
    """
    shape = _shape_of(position_ids)
    length = shape[-1]
    if length != query_length:
        detail = (
            f"position_ids has length {length} but the query has "
            f"{query_length}"
        )
        if cache_length is not None and length == cache_length:
            detail += (
                f" -- that is the CACHE length ({cache_length}). On a decode "
                "step the rotary must be sliced to the current position; full "
                "positions broadcast the query across the whole cache instead "
                "of raising"
            )
        raise ShapeContractError(detail)
    if len(shape) == 3 and shape[0] != axes:
        raise ShapeContractError(
            f"position_ids has {shape[0]} M-RoPE axes, expected {axes}"
        )


def assert_rotary_contract(
    cos: object, sin: object, query_length: int, cache_length: Optional[int] = None
) -> None:
    """``cos``/``sin`` must match the query length, not the cache length."""
    for name, tensor in (("cos", cos), ("sin", sin)):
        shape = _shape_of(tensor)
        length = shape[-2] if len(shape) >= 2 else shape[-1]
        if length != query_length:
            detail = (
                f"rotary {name} has length {length} but the query has "
                f"{query_length}"
            )
            if cache_length is not None and length == cache_length:
                detail += (
                    f" -- that is the CACHE length ({cache_length}); the "
                    "query would broadcast across it instead of raising"
                )
            raise ShapeContractError(detail)
