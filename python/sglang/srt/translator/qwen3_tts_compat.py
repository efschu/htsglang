# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Running the Qwen3-TTS reference modeling code under our `transformers`.

The package pins ``transformers==4.57.3``; this venv carries 5.12.1 for sglang,
and per the 2026-08-03 user order a version conflict is an engineering item,
never a reason to stand up a second environment. Measured, the pin turns out to
be conservative: all twenty imported symbols resolve under 5.12.1 and exactly
three APIs changed shape. This module is those three fixes, applied once,
idempotently, before the modeling modules are imported.

The drifts, each with what it would cost unfixed:

1. ``transformers.utils.generic.check_model_inputs`` became a plain decorator
   (it was a decorator factory). One call site. Unfixed: ``TypeError`` at
   import -- loud, harmless.
2. ``ROPE_INIT_FUNCTIONS`` lost its ``"default"`` key (now ``dynamic``,
   ``linear``, ``llama3``, ``longrope``, ``proportional``, ``yarn``). Unfixed:
   ``KeyError`` at construction -- loud, harmless.
3. The mask helpers renamed ``input_embeds`` to ``inputs_embeds`` and stopped
   accepting ``cache_position``. Unfixed: ``TypeError`` on first forward --
   loud, harmless.

All three fail loudly, which is the good case: none of them can silently
produce wrong audio. That is worth stating explicitly, because the M-RoPE key
trap in :mod:`talker_config` is the opposite kind of bug and the two must not
be filed under the same heading.

**The librosa stub.** The package's ``__init__`` imports a 25 Hz tokenizer we
never use, which imports ``librosa``, which would drag numba and llvmlite into
a venv that serves the LLM. So it is stubbed -- but the stub RAISES on any
attribute access rather than returning zeros. A silent stub would let a code
path we did not audit run on fake mel filters and produce plausible garbage;
this one stops. If it ever fires, the fix is to install librosa, not to soften
the stub.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import inspect
import logging
import sys
import threading
import types
from typing import Any, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_qwen3_tts_importable",
    "load_talker_modeling",
    "load_codec_modeling",
    "CompatError",
    "refresh_rotary_buffers",
    "restore_cache_position",
    "applied_shims",
]

_LOCK = threading.Lock()
_APPLIED: Tuple[str, ...] = ()
_DONE = False


class CompatError(RuntimeError):
    """The reference modeling code cannot be made importable here."""


class _RaisingCallable:
    """Binds fine, raises the moment anyone actually calls it.

    The distinction matters and was found by running this: the package's
    ``__init__`` chain reaches a 25 Hz tokenizer we never use, which does a
    MODULE-LEVEL ``from librosa.filters import mel``. That is a binding, not a
    use. Raising on attribute access would therefore block the import of code
    paths that never touch librosa at runtime.

    Raising on CALL keeps the guarantee that matters -- nothing ever computes
    on fabricated mel filters -- while letting the unrelated import complete.
    Weakening this further (returning zeros, say) is what the guarantee
    forbids: it would let an unaudited path run on fake data and produce
    plausible garbage.
    """

    def __init__(self, qualname: str) -> None:
        self._qualname = qualname

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise CompatError(
            f"{self._qualname} was CALLED, but it is stubbed out in this "
            "deployment. The translator's audio path is not supposed to reach "
            "it. Install the real package rather than weakening this stub -- "
            "returning a placeholder would let the code compute on fabricated "
            "inputs and produce plausible garbage."
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<stubbed {self._qualname}>"


class _RaisingStub(types.ModuleType):
    """A module whose members bind but cannot be used. See :class:`_RaisingCallable`."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _RaisingCallable(f"{self.__name__}.{name}")


def _stub_package(name: str) -> None:
    if name in sys.modules:
        return
    module = _RaisingStub(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module


def _default_rope_init(config, device=None, seq_len=None, **_kwargs):
    """4.57's ``"default"`` rope: plain inverse-frequency, no scaling.

    Reimplemented rather than aliased to ``"linear"``: linear applies a scaling
    factor, and defaulting to it would change positions subtly instead of
    loudly.
    """
    import torch

    base = getattr(config, "rope_theta", 10000.0)
    dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device).float() / dim)
    )
    return inv_freq, 1.0


def _adapt_mask_helper(fn):
    """Adapt a 4.57 mask-helper call to the 5.12 signature WITHOUT losing meaning.

    Two renames and one genuine hazard.

    The renames are trivial: ``input_embeds`` became ``inputs_embeds``.

    The hazard is ``cache_position``, which 5.12 no longer accepts. Filtering
    it out as an unknown kwarg -- the obvious thing, and what this wrapper did
    first -- silently changes what the mask MEANS. ``cache_position`` is what
    tells the builder that a one-token decode step sits at absolute position
    N: without it the mask is built as though the single query token were the
    whole sequence, attention returns one row per CACHED position instead of
    one per query token, and the error finally surfaces as a shape mismatch in
    ``o_proj`` several frames away
    (``mat1 and mat2 shapes cannot be multiplied (1x22528 and 2048x1024)``,
    where 22528 = 11 cached positions x 2048). Prefill is unaffected, so the
    model loads, prefills correctly, and dies on the first decode step.

    5.12 carries the same information in ``position_ids``, so the fix is to
    translate rather than drop: promote ``cache_position`` to ``position_ids``
    when the caller did not supply its own.
    """
    accepted = set(inspect.signature(fn).parameters)

    def wrapper(**kwargs):
        if "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        cache_position = kwargs.pop("cache_position", None)
        if (
            cache_position is not None
            and kwargs.get("position_ids") is None
            and "position_ids" in accepted
        ):
            positions = cache_position
            if hasattr(positions, "dim") and positions.dim() == 1:
                positions = positions.unsqueeze(0)
            kwargs["position_ids"] = positions
        return fn(**{k: v for k, v in kwargs.items() if k in accepted})

    wrapper.__name__ = getattr(fn, "__name__", "mask_helper")
    return wrapper


#: Attributes 4.57's ``PretrainedConfig`` provided on every config and 5.x
#: does not. Restored as CLASS defaults, with 4.57's own value (None), so no
#: consumer can observe a different number than it would have on the pinned
#: version.
_BASE_CONFIG_DEFAULTS = ("pad_token_id", "bos_token_id", "eos_token_id",
                         "sep_token_id", "decoder_start_token_id")


def librosa_resample(y, orig_sr, target_sr, **_kwargs):
    """``librosa.resample``, delegated to ``scipy.signal.resample_poly``.

    The raising stub found this one the same way it found the mel filterbank:
    the assumption that the audio path never reaches librosa was wrong twice.
    The reference wrapper resamples every reference clip from the codec's rate
    to the speaker encoder's before extracting the x-vector
    (``qwen3_tts_model.py:441-447``), so this is on the voice-cloning path, not
    a corner of it.

    Delegated rather than reimplemented. ``resample_poly`` is a polyphase
    rational resampler with a Kaiser-windowed anti-alias filter -- the same
    algorithm librosa itself uses under ``res_type="polyphase"`` -- and scipy
    is already in this venv. librosa's *default* is soxr, so this is a
    different high-quality resampler rather than a bit-exact twin: the
    difference is far below what a speaker embedding can resolve, and
    :mod:`test_resample` pins the properties that would actually move if the
    conversion were wrong (frequency preserved, no aliasing, correct length).

    Not routed through :func:`translator.audio.resample`: that one deliberately
    REFUSES rate pairs without a small rational ratio, which is right for the
    transport (a wrong pair there means a pitch-shifted conversation) and wrong
    here, where the reference clip's rate is whatever the phone produced.
    """
    import numpy as np
    from scipy.signal import resample_poly

    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    y = np.asarray(y, dtype=np.float32)
    if orig_sr == target_sr:
        return y
    if orig_sr <= 0 or target_sr <= 0:
        raise CompatError(
            f"cannot resample {orig_sr} Hz -> {target_sr} Hz: rates must be positive"
        )
    from math import gcd

    divisor = gcd(orig_sr, target_sr)
    up, down = target_sr // divisor, orig_sr // divisor
    return resample_poly(y, up, down, axis=-1).astype(np.float32)


def _config_token_defaults() -> None:
    """Give the Qwen3-TTS config classes back the base attributes 5.x dropped."""
    try:
        from qwen_tts.core.models import configuration_qwen3_tts as cfg
    except ImportError:  # pragma: no cover - package not installed
        return
    for name in dir(cfg):
        candidate = getattr(cfg, name)
        if not isinstance(candidate, type):
            continue
        if not name.endswith("Config"):
            continue
        for attribute in _BASE_CONFIG_DEFAULTS:
            if not hasattr(candidate, attribute):
                setattr(candidate, attribute, None)


def applied_shims() -> Tuple[str, ...]:
    """Which compatibility fixes were installed. For the health endpoint."""
    return _APPLIED


def ensure_qwen3_tts_importable() -> Tuple[str, ...]:
    """Install the compatibility shims. Idempotent; safe to call repeatedly."""
    global _APPLIED, _DONE
    with _LOCK:
        if _DONE:
            return _APPLIED

        applied = []

        import transformers.utils.generic as generic

        original_check = generic.check_model_inputs
        if original_check is not None:

            def check_model_inputs_compat(func=None, **kwargs):
                if func is None or not callable(func):
                    return lambda f: original_check(f)
                return original_check(func)

            generic.check_model_inputs = check_model_inputs_compat
            applied.append("check_model_inputs:decorator-form")

        from transformers import modeling_rope_utils

        if "default" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
            modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
            applied.append("ROPE_INIT_FUNCTIONS:default")

        from transformers import masking_utils

        for helper in (
            "create_causal_mask",
            "create_sliding_window_causal_mask",
            "create_chunked_causal_mask",
        ):
            fn = getattr(masking_utils, helper, None)
            if fn is None:
                continue
            params = set(inspect.signature(fn).parameters)
            if "input_embeds" not in params:
                setattr(masking_utils, helper, _adapt_mask_helper(fn))
                applied.append(f"{helper}:kwargs")

        for name in ("librosa", "librosa.filters", "librosa.util", "sox"):
            _stub_package(name)
        # The audio path DOES reach exactly one librosa function: the speaker
        # encoder builds a mel filterbank to turn reference audio into the
        # x-vector that voice cloning conditions on. The raising stub found
        # that (the assumption that nothing needed librosa was wrong), so the
        # honest fix is a real implementation rather than a softer stub:
        # `mel_filters.mel_filterbank` is validated element-wise against real
        # librosa 0.11.0 in test_mel_filters.py. Everything else in librosa
        # stays stubbed and still raises on use.
        from sglang.srt.translator.mel_filters import mel_filterbank

        sys.modules["librosa.filters"].mel = mel_filterbank
        applied.append("librosa.filters.mel:validated-reimplementation")
        # ... and exactly one more: the reference clip is resampled to the
        # speaker encoder's rate before the x-vector is extracted. Same
        # discovery mechanism, same treatment -- a real implementation, not a
        # softer stub. See librosa_resample.
        sys.modules["librosa"].resample = librosa_resample
        applied.append("librosa.resample:scipy-polyphase")
        applied.append("librosa,sox:raising-stub")

        # Drift 4, found by going past import into CONSTRUCTION: 4.57's
        # PretrainedConfig carried pad/bos/eos_token_id as base attributes
        # defaulting to None, and 5.x stopped doing so. The talker config does
        # not declare them, so `config.pad_token_id` raises AttributeError the
        # moment the embedding layer is built. Supplying the 4.57 defaults on
        # the config CLASSES is the smallest correct fix: None is exactly what
        # 4.57 handed back, so nothing downstream sees a different value.
        _config_token_defaults()
        applied.append("config:pad/bos/eos_token_id-defaults")

        _APPLIED = tuple(applied)
        _DONE = True
        logger.info("qwen3-tts compatibility shims applied: %s", ", ".join(applied))
        return _APPLIED


def _import_with_shims(module_path: str):
    ensure_qwen3_tts_importable()
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise CompatError(
            f"cannot import {module_path}: {exc}. Install the modeling code "
            "with `pip install --no-deps qwen-tts` -- --no-deps matters, the "
            "package's own pins would downgrade transformers underneath sglang."
        ) from exc

    # The shims must also be visible under the names the module bound at
    # import time; patching only the source modules leaves the already-bound
    # references pointing at the old callables.
    from transformers import masking_utils, modeling_rope_utils

    if hasattr(module, "ROPE_INIT_FUNCTIONS"):
        module.ROPE_INIT_FUNCTIONS.setdefault(
            "default", modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"]
        )
    for helper in (
        "create_causal_mask",
        "create_sliding_window_causal_mask",
        "create_chunked_causal_mask",
    ):
        if hasattr(module, helper):
            setattr(module, helper, getattr(masking_utils, helper))
    return module


def refresh_rotary_buffers(model) -> int:
    """Recompute every rotary ``inv_freq`` after loading. Returns how many.

    Drift 5, and the only one of the set that does NOT fail loudly.
    ``inv_freq`` is registered ``persistent=False``, so it is absent from the
    checkpoint and must be computed at construction. transformers 5.x
    constructs on the meta device and materialises afterwards, and a
    non-persistent buffer that nothing re-initialises comes back as
    uninitialised memory -- in practice NaN.

    The consequence is the worst kind: NaN inv_freq gives NaN cos/sin, which
    gives NaN attention scores, which gives NaN logits, and the first thing
    anyone sees is ``probability tensor contains inf, nan or element < 0``
    from ``torch.multinomial`` -- thirty layers away from the cause, and only
    when sampling actually runs. Nothing before that point misbehaves, which
    is why it survived load, speaker-embedding extraction and prompt building.

    Recomputing from each module's own ``rope_init_fn`` and config is exactly
    what its ``__init__`` would have done on 4.57.
    """
    import torch

    refreshed = 0
    for module in model.modules():
        init_fn = getattr(module, "rope_init_fn", None)
        config = getattr(module, "config", None)
        if init_fn is None or config is None:
            continue
        existing = getattr(module, "inv_freq", None)
        if existing is None:
            continue
        device = existing.device if torch.is_tensor(existing) else None
        if device is not None and device.type == "meta":
            device = None
        inv_freq, attention_scaling = init_fn(config, device)
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module.original_inv_freq = module.inv_freq
        module.attention_scaling = attention_scaling
        refreshed += 1
    if refreshed:
        logger.info("recomputed %d rotary inv_freq buffers after load", refreshed)
    return refreshed


def restore_cache_position(model_class) -> bool:
    """Drift 6: give a model back the ``cache_position`` 5.x stopped creating.

    This is the decode-seam bug, and it is worth stating precisely because the
    obvious diagnosis is wrong.

    The talker picks its position bookkeeping by branching on
    ``cache_position`` (``modeling_qwen3_tts.py:1693-1711``):

    * ``cache_position is None`` **or** ``cache_position[0] == 0`` -- prefill.
      Rebuild the whole sequence's M-RoPE positions from the attention mask via
      ``get_rope_index``, which returns one position per MASK entry.
    * otherwise -- decode. ``arange(query_len) + cache_position[0] +
      rope_deltas``, i.e. one position per QUERY token.

    transformers 4.57 created ``cache_position`` in
    ``prepare_inputs_for_generation`` for every model. 5.x removed it, and now
    creates it only for remote-code models (``generation/utils.py:596-604``,
    behind ``self.is_remote_code()``). ``qwen-tts`` is an ordinary installed
    package, so it never qualifies: the talker sees ``None`` on every step and
    takes the PREFILL branch forever.

    The consequence is silent for exactly one step and then fatal. On a decode
    step the attention mask has already grown to the cache length, so
    ``get_rope_index`` hands M-RoPE one position per CACHED token while the
    query is one token. ``cos``/``sin`` come back at cache length, the
    single-token query broadcasts across them instead of raising, and the first
    complaint is a matmul several frames away::

        mat1 and mat2 shapes cannot be multiplied (1x22528 and 2048x1024)

    where 22528 = 11 cached positions x 2048. Prefill is unaffected -- there
    query length IS the cache length -- which is why the model loads, prefills
    correctly, and dies on the first decode step.

    **Not the earlier suspicion.** The talker's ``arange(seq_length)`` reads
    ``input_ids.shape``, and 5.x does still slice ``input_ids`` to the query
    width for cached generation (``next_sequence_length = 1``,
    ``generation/utils.py:2793``), so that line was never the defect. It is
    simply unreachable. That is also why slicing ``input_ids`` in a wrapper
    changed nothing.

    The fix restores the input the talker's own position math was written
    against rather than reimplementing that math: ``arange(query_len) +
    past_seen_tokens``, which is what 4.57 supplied and what 5.x still supplies
    to remote-code models. The talker's decode branch then runs verbatim,
    including its ``rope_deltas`` correction for left padding -- which a
    reimplementation would have had to rediscover.

    Idempotent; returns whether it installed anything.
    """
    if getattr(model_class, "_htsglang_cache_position_restored", False):
        return False

    import functools

    import torch

    base = model_class.prepare_inputs_for_generation

    @functools.wraps(base)
    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = base(self, *args, **kwargs)
        if model_inputs.get("cache_position") is not None:
            return model_inputs
        probe = model_inputs.get("input_ids")
        if probe is None:
            probe = model_inputs.get("inputs_embeds")
        if probe is None or probe.ndim < 2:
            return model_inputs
        past = model_inputs.get("past_key_values")
        past_seen = past.get_seq_length() if past is not None else 0
        model_inputs["cache_position"] = (
            torch.arange(probe.shape[1], device=probe.device) + past_seen
        )
        return model_inputs

    model_class.prepare_inputs_for_generation = prepare_inputs_for_generation
    model_class._htsglang_cache_position_restored = True
    return True


def load_talker_modeling():
    """The Qwen3-TTS talker modeling module, importable under 5.12.1."""
    return _import_with_shims("qwen_tts.core.models.modeling_qwen3_tts")


def load_codec_modeling():
    """The 12 Hz codec (tokenizer v2) modeling module."""
    return _import_with_shims(
        "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2"
    )
