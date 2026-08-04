# SPDX-License-Identifier: Apache-2.0
"""#488 graphs cut, slice 1: the code predictor without ``generate()``.

WHY THIS MODULE EXISTS, in one measurement
------------------------------------------
`/spinning/gpu-battery-results/2026-08-04_488_precursor/RESULTS.md`: of a
142.78 ms talker frame, **104.44 ms is one call to
``code_predictor.generate()``** -- 16.21 ms of it kernel work and **602 host
syncs**. The HF generation loop re-enters its whole machinery (logits
processors, stopping criteria, cache bookkeeping, `.item()` checks) fifteen
times per audio frame, for a 5-layer model whose actual work is under a
millisecond per step.

This module replaces that ONE call with a raw loop. It is a drop-in: the
reference talker calls
``self.code_predictor.generate(inputs_embeds=..., max_new_tokens=..., ...)``
(`modeling_qwen3_tts.py:1671`) and reads ``.sequences``; :class:`FastCodePredictor`
answers the same shape. Nothing else in the reference changes, so the 1596-line
prompt builder, the trunk loop and the codec are untouched.

THE STEP SCHEDULE, because getting it wrong is silent
-----------------------------------------------------
The reference's own bookkeeping (`modeling_qwen3_tts.py:1276-1301`) is:

* **prefill** with 2 embeddings ``(past_hidden, last_id_hidden)``, which sets
  ``generation_steps = inputs_embeds.shape[1] - 2 = 0``, so the first logits
  come from ``lm_head[0]`` and produce code group **1**;
* **decode step g** (g = 1..14) embeds the PREVIOUS code with
  ``codec_embedding[g-1]`` and takes its logits from ``lm_head[g]``.

Off-by-one here does not crash: it picks a neighbouring codebook head and the
audio degrades in timbre. :func:`step_schedule` states the mapping as data so a
test can pin it.

WHAT IS AND IS NOT VALIDATED
----------------------------
Hermetic (this tree): the sampler is checked against transformers' OWN warper
classes on fixed logits, and the step schedule is pinned. End-to-end **greedy
code-token identity** against the reference ``generate`` needs the real
checkpoint and the tenant's import shims, so it runs in the GPU validation
script and is reported with the measurement -- see
``scripts/dev/488_talker_profile/validate_fast_predictor.py``.

Graph capture is slice 2 and is deliberately NOT in this file. The loop had to
come first: it removes the 602 syncs, and a region that syncs cannot be
captured at all.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "FastCodePredictor",
    "PredictorOutput",
    "step_schedule",
    "apply_warpers",
]


@dataclasses.dataclass
class PredictorOutput:
    """The only field the reference reads off the generate result."""

    sequences: torch.Tensor


def step_schedule(num_code_groups: int) -> List[Tuple[int, Optional[int], int]]:
    """``(generation_step, embedding_index, head_index)`` for one frame.

    ``embedding_index`` is ``None`` on the prefill step, which consumes the two
    incoming hidden states instead of a codebook embedding. Returned as data so
    the off-by-one that would silently pick the wrong codebook head is a test
    assertion rather than a listening test.
    """
    schedule: List[Tuple[int, Optional[int], int]] = [(0, None, 0)]
    for step in range(1, num_code_groups - 1):
        schedule.append((step, step - 1, step))
    return schedule


def apply_warpers(
    logits: torch.Tensor,
    temperature: Optional[float],
    top_k: Optional[int],
    top_p: Optional[float],
) -> torch.Tensor:
    """Temperature, then top-k, then top-p -- transformers' own order.

    Reproduced rather than imported because the warper classes are constructed
    inside ``generate`` and calling them per step would reintroduce part of the
    machinery this module exists to remove. The equivalence is a TEST against
    the real ``TemperatureLogitsWarper`` / ``TopKLogitsWarper`` /
    ``TopPLogitsWarper``, not a comment claiming it.

    The skips match transformers': a temperature of 1.0, a ``top_k`` of 0, and
    a ``top_p`` of 1.0 are no-ops there and must be no-ops here, or a default
    config would take a different path in the two implementations.
    """
    if temperature is not None and temperature != 1.0:
        logits = logits / temperature
    if top_k:
        k = min(int(top_k), logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None and top_p < 1.0:
        ordered, order = torch.sort(logits, descending=False, dim=-1)
        probabilities = ordered.softmax(dim=-1).cumsum(dim=-1)
        remove = probabilities <= (1.0 - top_p)
        # transformers keeps at least one token; the highest-probability entry
        # is the last one after an ascending sort.
        remove[..., -1:] = False
        logits = ordered.masked_fill(remove, float("-inf")).gather(
            -1, order.argsort(-1)
        )
    return logits


class FastCodePredictor:
    """Drop-in replacement for ``code_predictor.generate``.

    Holds no weights: it drives the predictor modules the process already has,
    so installing it costs no VRAM.
    """

    def __init__(self, predictor, num_code_groups: int) -> None:
        self.predictor = predictor
        self.num_code_groups = num_code_groups
        self.schedule = step_schedule(num_code_groups)
        #: Set by the caller to make a run reproducible against the reference.
        self.generator: Optional[torch.Generator] = None

    # -- installation -------------------------------------------------------

    @classmethod
    def install(cls, talker) -> "FastCodePredictor":
        """Swap ``talker.code_predictor.generate`` for this loop.

        Returns the driver, which keeps ``original_generate`` so a caller can
        put the reference back -- the identity gate needs both paths live in
        one process, and an operator needs a way out that is not a restart.
        """
        predictor = talker.code_predictor
        driver = cls(predictor, talker.config.num_code_groups)
        driver.original_generate = predictor.generate
        predictor.generate = driver.generate  # type: ignore[method-assign]
        logger.info(
            "#488: code_predictor.generate replaced by the raw %d-step loop "
            "(reference kept on the driver as original_generate)",
            len(driver.schedule),
        )
        return driver

    def uninstall(self) -> None:
        if getattr(self, "original_generate", None) is not None:
            self.predictor.generate = self.original_generate
            logger.info("#488: reference code_predictor.generate restored")

    # -- the loop -----------------------------------------------------------

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        **ignored,
    ) -> PredictorOutput:
        from transformers import DynamicCache  # noqa: PLC0415

        predictor = self.predictor
        model = predictor.model
        wanted = max_new_tokens or (self.num_code_groups - 1)
        if wanted != len(self.schedule):
            raise ValueError(
                f"max_new_tokens={wanted} does not match this checkpoint's "
                f"{len(self.schedule)} residual groups; refusing rather than "
                f"emitting a partial frame, which would desync the codebooks."
            )

        projection = getattr(predictor, "small_to_mtp_projection", None)
        cache = DynamicCache()
        codes: List[torch.Tensor] = []
        embeddings = model.get_input_embeddings()
        hidden = inputs_embeds

        for generation_step, embedding_index, head_index in self.schedule:
            if embedding_index is not None:
                hidden = embeddings[embedding_index](codes[-1])
            if projection is not None:
                hidden = projection(hidden)
            outputs = model(
                inputs_embeds=hidden,
                past_key_values=cache,
                use_cache=True,
            )
            last = outputs.last_hidden_state[:, -1:, :]
            logits = predictor.lm_head[head_index](last)[:, -1, :]
            if do_sample:
                logits = apply_warpers(logits, temperature, top_k, top_p)
                probabilities = logits.softmax(dim=-1)
                token = torch.multinomial(
                    probabilities, num_samples=1, generator=self.generator
                )
            else:
                token = logits.argmax(dim=-1, keepdim=True)
            # Stays on the device on purpose: every `.item()` here is one of
            # the 602 syncs per frame this module exists to remove, and a
            # host round trip would also make the loop uncapturable in slice 2.
            codes.append(token)
            hidden = None  # replaced at the top of the next iteration

        return PredictorOutput(sequences=torch.cat(codes, dim=-1))
