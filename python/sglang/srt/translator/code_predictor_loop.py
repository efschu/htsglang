# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Run the talker's code predictor as one loop instead of 15 ``generate()`` calls.

WHY THIS EXISTS. `MEASURE_TTS_LATENCY.md` timed a talker decode step on the
RTX 5090 at 92.6 ms, split 22.6 ms trunk / 59.2 ms code predictor / ~10.9 ms
host. The predictor's 59.2 ms is 15 sequential invocations of a FIVE-LAYER,
hidden-1024 model -- 3.95 ms each, against a memory roofline nearer 0.1 ms.
A batch sweep at fixed shape settled what that 3.95 ms is: 32x the arithmetic
costs 1.03x the time, so at most ~3 % of a step is arithmetic and ~97 % is
per-invocation overhead. There is nothing to make faster inside the kernels;
there are calls to remove.

The step rate decides whether the live translator can speak without gaps. Audio
is consumed at 12.5 frames per second and produced at 10.5 steps per second, so
RTF = 1.20: playback drains faster than generation fills. Streaming emission
(171c3f2932) hid that for short turns by buying a pre-roll, and the field
promptly showed the residue -- two mid-turn underruns, both immediately before
the FINAL chunk of a long turn, the exact signature of a pre-roll draining at a
constant rate. **1.26x on the step rate puts RTF below 1.0 and removes the
turn-length ceiling entirely.** That is the whole target of this module.

WHAT IS REMOVED, and nothing else. The reference calls
``code_predictor.generate()`` once per talker step
(``modeling_qwen3_tts.py:1671``). Each call re-enters the whole of
``transformers``' generation machinery for fifteen one-token steps:

* the ``generate()`` prologue -- generation-config merge and deep copy, model
  input preparation, special-token preparation, and a freshly CONSTRUCTED
  logits-processor list and stopping-criteria list, all rebuilt for parameters
  that never change within a process;
* per sub-step, ``prepare_inputs_for_generation`` (which clones inputs for
  stride reasons), ``_update_model_kwargs_for_generation``, and a growing
  all-ones attention mask that is rebuilt into a causal mask by
  ``create_causal_mask`` on every one of the fifteen forwards;
* per sub-step, ``unfinished_sequences.max() == 0`` -- a device-to-host
  synchronization, fifteen times per talker step, on a loop whose trip count is
  statically ``num_code_groups - 1`` and whose stopping rule is a constant.

This module replaces that with the loop the machinery was executing anyway.
It does NOT touch the talker's own ``generate``, the KV cache layout, the
sampling, the emitter, or the runaway guard.

WHY THE SEMANTICS SURVIVE, and how that is enforced rather than asserted. The
licence granted for this work was about BITS, not about meaning: the audio need
not be byte-identical, but an unrolled loop that changed the sampling order,
the RNG draw sequence, the stopping rule or the residual-codebook order would
change what is SAID. So this module does not re-derive any of those. On the
first call for a given set of sampling parameters it runs the REFERENCE
``generate`` with two spies attached, and keeps the objects transformers itself
built:

* the ``LogitsProcessorList``, used afterwards verbatim and in order, so the
  temperature/top-k/top-p chain is not a reimplementation of the rules but the
  very list the reference would have applied;
* the ``StoppingCriteriaList``, which is then CHECKED to be exactly one
  ``MaxLengthCriteria`` at ``max_new_tokens`` with no EOS criterion and no
  ``MaxTimeCriteria``. If it is anything else -- a checkpoint that gives the
  predictor an EOS token, a future caller that arms a deadline on the predictor
  the way `inprocess_tts.arm_generation_deadline` arms one on the talker -- the
  loop refuses to install and every call falls back to the reference. A loop
  with a constant trip count is only equivalent while the stopping rule is a
  constant, and that is a fact about the configuration, not about this code.

Everything else is copied line for line from ``GenerationMixin._sample``: the
float32 cast of the last-position logits, the processor call with the running
``input_ids``, ``softmax`` then ``torch.multinomial(probs, num_samples=1)`` in
that order. Same tensors in the same order into the same RNG means the same
draws, so on this venv the sequences are bit-identical, which is a stronger
statement than the one required and is checked directly by
``scripts/translator/probe_unrolled_predictor.py``.

WHERE THE ATTENTION MASK GOES. The two mask decisions are the only places where
this loop passes something the reference did not, so both are argued.

* Prefill (2 positions) passes ``attention_mask=None`` where the reference
  passed an all-ones ``(B, 2)`` mask. Causality still comes from
  ``create_causal_mask``, which builds the same causal structure either way; an
  all-ones padding mask is a no-op AND-ed into it. This matters and is not
  cosmetic: position 0's key/value states feed layers 1..4, so a prefill that
  let position 0 see position 1 would poison the cache. The mask is kept.
* The fourteen decode steps pass ``{"full_attention": None}``, the dict form
  the model already accepts to mean "masks are prepared, do not build them"
  (``modeling_qwen3_tts.py:1094``). A one-token query against a cache holding
  only earlier positions of the SAME call has no position it must not see and
  no padding -- the cache is built entirely inside this call from a full
  2-position prefill, for every row of the batch -- so the correct mask is no
  mask, for eager and sdpa alike. This is what removes fifteen
  ``create_causal_mask`` calls per talker step.

``cache_position`` and ``position_ids`` are sliced from a single ``arange``
allocated once per device, instead of a fresh small tensor per sub-step; the
values are identical to what the model computes from ``get_seq_length()``.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["unroll_code_predictor", "loop_stats", "set_enabled", "is_enabled"]


#: Call counters, for a one-line answer to "is the fast path actually live".
#: Reported by `InProcessQwen3Tts.stats()`; not used for any control decision.
_STATS: Dict[str, int] = {"calibrated": 0, "unrolled": 0, "fell_back": 0}

#: Runtime switch, consulted per call rather than only at install time.
#:
#: The patch lands on the predictor CLASS, so a per-instance flag could not turn
#: it off again -- and the measurement this module exists to justify has to
#: INTERLEAVE the unrolled and reference arms inside one process. Blocked arms
#: cannot tell a speedup from card drift, and two processes cannot share a warm
#: CUDA context. So the switch is here, and the reference loop stays reachable
#: for the life of the process rather than only across a restart.
_ENABLED: bool = True


def set_enabled(flag: bool) -> bool:
    """Turn the unrolled loop on or off for later calls. Returns the old state."""
    global _ENABLED
    previous = _ENABLED
    _ENABLED = bool(flag)
    return previous


def is_enabled() -> bool:
    """Whether later predictor calls take the unrolled loop."""
    return _ENABLED


def loop_stats() -> Dict[str, int]:
    """How many predictor calls took each path since process start."""
    return dict(_STATS)


class _Plan:
    """What the reference built once, reused for every later call.

    Holds transformers' own logits processors -- not a reconstruction of them --
    plus the two scalars the loop needs. ``lm_head_base`` is the code group the
    prefill's logits come from; the reference derives it as
    ``inputs_embeds.shape[1] - 2`` and every later sub-step walks up from there.
    """

    __slots__ = ("processors", "do_sample", "max_new_tokens", "lm_head_base")

    def __init__(self, processors, do_sample: bool, max_new_tokens: int,
                 lm_head_base: int) -> None:
        self.processors = processors
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.lm_head_base = lm_head_base


def unroll_code_predictor(predictor_class) -> bool:
    """Install the unrolled loop on the code predictor CLASS. Idempotent.

    Returns whether it installed anything. Wrapping ``generate`` rather than
    editing the talker's forward keeps this to one seam: the talker's call site
    (``modeling_qwen3_tts.py:1671``) is unchanged, still receives an object with
    ``.sequences``, and any call shape this loop does not recognise reaches the
    reference untouched.
    """
    if getattr(predictor_class, "_htsglang_unrolled_predictor", False):
        return False

    base_generate = predictor_class.generate

    @functools.wraps(base_generate)
    def generate(self, *args, **kwargs):
        if not _ENABLED or args or not _shape_supported(kwargs):
            _STATS["fell_back"] += 1
            return base_generate(self, *args, **kwargs)
        plans = self.__dict__.setdefault("_htsglang_predictor_plans", {})
        key = _plan_key(kwargs)
        if key not in plans:
            result, plan = _calibrate(self, base_generate, kwargs)
            plans[key] = plan
            _STATS["calibrated"] += 1
            return result
        plan = plans[key]
        if plan is None:
            _STATS["fell_back"] += 1
            return base_generate(self, *args, **kwargs)
        _STATS["unrolled"] += 1
        return _run_unrolled(self, plan, kwargs)

    predictor_class.generate = generate
    predictor_class._htsglang_unrolled_predictor = True
    return True


#: Arguments whose presence means the call is not the talker's per-step call.
#: Each of them would route ``generate`` somewhere this loop does not model --
#: beam search, an external streamer, a caller-supplied stopping rule -- so the
#: honest response is to hand the call back rather than to approximate it.
_UNSUPPORTED_KWARGS = (
    "input_ids",
    "attention_mask",
    "generation_config",
    "logits_processor",
    "stopping_criteria",
    "prefix_allowed_tokens_fn",
    "streamer",
    "num_beams",
    "assistant_model",
    "past_key_values",
)


def _shape_supported(kwargs: Dict[str, Any]) -> bool:
    """Is this the talker's own per-step call, in the shape the loop models?"""
    if any(kwargs.get(name) is not None for name in _UNSUPPORTED_KWARGS):
        return False
    embeds = kwargs.get("inputs_embeds")
    if embeds is None or embeds.dim() != 3 or embeds.shape[1] < 2:
        return False
    max_new_tokens = kwargs.get("max_new_tokens")
    return isinstance(max_new_tokens, int) and max_new_tokens >= 1


def _plan_key(kwargs: Dict[str, Any]) -> Tuple:
    """Everything that could change what the reference would have built."""
    return (
        kwargs.get("max_new_tokens"),
        kwargs.get("do_sample"),
        kwargs.get("temperature"),
        kwargs.get("top_k"),
        kwargs.get("top_p"),
        kwargs.get("inputs_embeds").shape[1],
    )


def _calibrate(self, base_generate, kwargs: Dict[str, Any]):
    """Run the reference once and keep what it built. Returns (result, plan).

    ``plan is None`` means: this configuration is not one a constant-trip loop
    can reproduce, so every later call with the same key goes to the reference.
    The reason is logged once, at warning level, naming what was found -- a
    silent fallback would look exactly like a fast path that is quietly not
    running, which is the failure mode this whole round exists to avoid.
    """
    captured: Dict[str, Any] = {}

    def spy_processors(*args, **spy_kwargs):
        result = type(self)._get_logits_processor(self, *args, **spy_kwargs)
        captured["processors"] = result
        captured["generation_config"] = spy_kwargs.get("generation_config")
        return result

    def spy_criteria(*args, **spy_kwargs):
        result = type(self)._get_stopping_criteria(self, *args, **spy_kwargs)
        captured["criteria"] = result
        return result

    self.__dict__["_get_logits_processor"] = spy_processors
    self.__dict__["_get_stopping_criteria"] = spy_criteria
    try:
        result = base_generate(self, **kwargs)
    finally:
        self.__dict__.pop("_get_logits_processor", None)
        self.__dict__.pop("_get_stopping_criteria", None)

    refusal = _refuse_reason(self, captured, kwargs, result)
    if refusal is not None:
        logger.warning(
            "code predictor stays on the reference generate(): %s. The talker "
            "still produces correct audio; it keeps the measured 59 ms/step "
            "cost of 15 nested generate() calls.",
            refusal,
        )
        return result, None

    plan = _Plan(
        processors=captured["processors"],
        do_sample=bool(captured["generation_config"].do_sample),
        max_new_tokens=int(kwargs["max_new_tokens"]),
        lm_head_base=int(kwargs["inputs_embeds"].shape[1]) - 2,
    )
    logger.info(
        "code predictor unrolled: %d sub-steps per talker step, processors=%s, "
        "do_sample=%s",
        plan.max_new_tokens,
        [type(p).__name__ for p in plan.processors],
        plan.do_sample,
    )
    return result, plan


def _refuse_reason(self, captured, kwargs, result) -> Optional[str]:
    """Why this configuration must keep the reference loop, or None.

    The checks are the equivalence argument in executable form. A constant-trip
    unrolled loop reproduces ``_sample`` only while (1) sampling is the decoding
    mode, (2) nothing but the length limit can stop it, and (3) that limit is
    exactly the trip count. Each is verified against what transformers actually
    built for THIS call, not against what the config file says.
    """
    if "processors" not in captured or "criteria" not in captured:
        return "transformers did not go through the expected preparation path"

    config = captured.get("generation_config")
    if config is None:
        return "the generation config could not be observed"
    if getattr(config, "num_beams", 1) != 1:
        return f"num_beams={config.num_beams}, which is not a sampling loop"
    if getattr(config, "num_return_sequences", 1) != 1:
        return f"num_return_sequences={config.num_return_sequences}"
    if not getattr(config, "use_cache", True):
        return "use_cache is off, so the loop would recompute the prefix"
    if getattr(config, "output_attentions", False):
        return "output_attentions was requested and the loop does not collect it"

    criteria = list(captured["criteria"])
    names = [type(c).__name__ for c in criteria]
    if names != ["MaxLengthCriteria"]:
        # An EOS criterion or a MaxTimeCriteria makes the trip count a runtime
        # value; unrolling it would silently generate past a stop.
        return f"stopping criteria are {names}, not a bare length limit"
    max_length = getattr(criteria[0], "max_length", None)
    if max_length != kwargs["max_new_tokens"]:
        return (
            f"the length limit is {max_length} but the caller asked for "
            f"{kwargs['max_new_tokens']} new tokens, so the loop's trip count "
            "would not match the reference's"
        )

    sequences = getattr(result, "sequences", None)
    if sequences is None:
        return "the reference returned no `sequences` to match the shape of"
    if sequences.shape[1] != kwargs["max_new_tokens"]:
        return (
            f"the reference produced {sequences.shape[1]} tokens for "
            f"max_new_tokens={kwargs['max_new_tokens']}; the prompt is not "
            "empty and the loop's indexing would be off"
        )

    if getattr(getattr(self, "model", None), "has_sliding_layers", False):
        # The decode steps hand the model a mask mapping with only a
        # "full_attention" entry; a sliding layer would ask for a key that is
        # not there, and a window that truncates the cache is a different
        # computation besides.
        return "the predictor has sliding-attention layers"

    heads = getattr(self, "lm_head", None)
    last_head = int(kwargs["inputs_embeds"].shape[1]) - 2 + kwargs["max_new_tokens"] - 1
    if heads is None or last_head >= len(heads):
        return "the code groups do not line up with the model's lm_head list"
    return None


def _positions(self, device, length: int):
    """A single cached ``arange`` on this device, sliced by the loop.

    The reference lets the model derive ``cache_position`` from
    ``get_seq_length()`` on every sub-step, which allocates a small tensor per
    call. The values here are identical -- position i is i -- and slicing a
    cached tensor is a view, so the loop neither allocates nor copies to the
    device inside a talker step.
    """
    import torch

    cache = self.__dict__.setdefault("_htsglang_predictor_positions", {})
    entry = cache.get(device)
    if entry is None or entry.shape[0] < length:
        entry = torch.arange(max(length, 32), device=device, dtype=torch.long)
        cache[device] = entry
    return entry


def _run_unrolled(self, plan: _Plan, kwargs: Dict[str, Any]):
    """``GenerationMixin._sample`` for this one model, with the loop laid open.

    Every arithmetic line has a counterpart in ``generation/utils.py`` and is
    kept in the same order for the same reason: the RNG is a shared, sequential
    resource, so equal draws require equal calls in equal order.
    """
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.generation.utils import GenerateDecoderOnlyOutput

    inputs_embeds = kwargs["inputs_embeds"]
    device = inputs_embeds.device
    batch = inputs_embeds.shape[0]
    prefill_len = inputs_embeds.shape[1]
    total = prefill_len + plan.max_new_tokens - 1
    positions = _positions(self, device, total)

    collect_hidden = bool(kwargs.get("output_hidden_states"))
    hidden_states: Tuple = ()

    past_key_values = DynamicCache(config=self.config)
    # `prepare_inputs_for_generation` clones the prompt embeds for a consistent
    # stride (transformers #32227). Kept: a different stride can select a
    # different kernel, and the point of this loop is to change the call count,
    # not the numerics.
    prompt = inputs_embeds.clone(memory_format=torch.contiguous_format)
    outputs = self(
        inputs_embeds=prompt,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=positions[:prefill_len],
        output_hidden_states=collect_hidden,
        return_dict=True,
    )

    input_ids = torch.zeros((batch, 0), dtype=torch.long, device=device)
    generation_steps = plan.lm_head_base
    for step in range(plan.max_new_tokens):
        if collect_hidden:
            hidden_states += (outputs.hidden_states,)
        # `_sample`: copy so the (large) logits of the first iteration are not
        # kept alive, and process in float32 whatever the model's dtype is.
        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=device
        )
        next_token_scores = plan.processors(input_ids, next_token_logits)
        if plan.do_sample:
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if step == plan.max_new_tokens - 1:
            # The stopping rule is the length limit and nothing else; this is
            # the check `_refuse_reason` proved, so the loop stops here rather
            # than paying a device-to-host sync to be told the same thing.
            break
        generation_steps += 1
        position = positions[prefill_len + step: prefill_len + step + 1]
        outputs = self(
            input_ids=next_tokens[:, None],
            generation_steps=generation_steps,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=position,
            # "Masks are prepared, do not build them." A one-token query
            # against this call's own unpadded cache has nothing to hide from.
            attention_mask={"full_attention": None},
            output_hidden_states=collect_hidden,
            return_dict=True,
        )

    return GenerateDecoderOnlyOutput(
        sequences=input_ids,
        scores=None,
        logits=None,
        attentions=None,
        hidden_states=hidden_states if collect_hidden else None,
        past_key_values=past_key_values,
    )
