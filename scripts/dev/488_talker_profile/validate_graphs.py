# SPDX-License-Identifier: Apache-2.0
"""#488 slice 2 GPU arm: the identity gate, then the ladder.

TWO JOBS, IN THIS ORDER, AND THE ORDER IS THE POINT
---------------------------------------------------
1. **Prove the graphed path computes what the eager path computes.** A graph
   that is wrong is *faster*, so a measurement taken before the gate would look
   like a win. ``ANALYSE_488 §7.5`` gate 1: same prompt, same pre-drawn
   uniforms, the graphed driver must emit the **identical** codec token
   sequence as the eager reference -- all groups, every frame. That one
   assertion covers the static-cache swap, the mask, the sampling rewrite and
   the capture in a single check, before any audio exists.
2. **Then** measure the ladder, in the same window, against the same
   calibration arms the precursor used, so the rungs are comparable to
   ``2026-08-04_488_precursor/RESULTS.md`` and to each other.

WHAT COMPARABILITY REQUIRES HERE, AND WHAT IT CANNOT DELIVER
------------------------------------------------------------
All arms run back to back in one process on one card, so they share a power
state, a clock, a driver and a contention environment. The report records the
power state, because these cards run with a lowered power target and a number
taken under a different one is not comparable to these.

What it cannot deliver: comparability with the precursor's ABSOLUTE 142.78 ms.
That figure came from a second copy of the talker beside a live tenant on a
different boot. The **decomposition** transfers; the absolute does not. So this
script re-measures the reference arm itself rather than quoting it, and every
rung is reported against that same-window baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pathlib
import subprocess
import sys
import time
from typing import Dict, List

logger = logging.getLogger("488-validate-graphs")

_HERE = pathlib.Path(__file__).resolve().parent
_CORRIDOR_MIB = 400.0
#: MEASURED, not estimated. The first full run on 2026-08-04 was budgeted at
#: 3000 MiB from a line-item estimate and actually peaked at ~5090 MiB, which
#: left the 5090 at **312 MiB free** and breached the standing 400 MiB
#: corridor. The line items were right and the total was wrong: what the
#: estimate omitted is that each phase's static caches, graph pools and cuBLAS
#: workspaces were still resident when the next phase allocated its own.
#:
#: Two things changed in response, and the constant is only the smaller one:
#: phases now release before the next allocates (:func:`release`), and the
#: corridor is re-checked BETWEEN phases (:func:`enforce_corridor`) instead of
#: only at the ends, where a transient breach is invisible.
_NEED_MIB = 3600.0


def release() -> None:
    """Drop a phase's allocations before the next phase claims its own.

    ``empty_cache`` alone is not enough: the caching allocator only returns
    blocks with no live tensors, and a captured graph's pool stays alive as
    long as any Python reference to the graph does. So the collection has to
    happen first, and it has to be explicit -- ``gc.collect()`` is doing real
    work here, not cargo cult, because the driver objects participate in
    reference cycles through their captured closures.
    """
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


#: The two slice modules live on THIS branch; the model loader lives on agent
#: 8's (`wt-466-translator`), and it is not optional -- it applies nine
#: transformers-5.12 compat shims and repairs rotary buffers that are
#: otherwise NaN after meta-device construction (`inprocess_tts.py:205-214`).
#: Reimplementing that here would be a second, diverging copy of somebody
#: else's hard-won load path. So the run takes the 466 tree as `sglang` and
#: injects these two by path instead.
_LOCAL_MODULES = {
    "sglang.srt.models.qwen3_tts_fast_predictor": (
        _HERE.parents[2] / "python/sglang/srt/models/qwen3_tts_fast_predictor.py"
    ),
    "sglang.srt.models.qwen3_tts_graph_driver": (
        _HERE.parents[2] / "python/sglang/srt/models/qwen3_tts_graph_driver.py"
    ),
}


def install_local_modules() -> List[str]:
    """Register this branch's slice modules under their real dotted names.

    Registered in ``sys.modules`` before anything imports them, so that
    ``qwen3_tts_graph_driver``'s own ``from ... import`` of the fast predictor
    resolves to the file next to it rather than to whatever the ``sglang`` on
    ``PYTHONPATH`` happens to carry. Order matters: the dependency first.
    """
    installed = []
    for name, path in _LOCAL_MODULES.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path} is missing")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        installed.append(f"{name} <- {path}")
    return installed


def _load_profiler():
    """Reuse the precursor's instrument rather than writing a second one.

    Two instruments measuring the same thing is how two numbers that disagree
    get published. `_profile_region`, the calibration arms and the
    discrimination gate are imported, not reimplemented.
    """
    spec = importlib.util.spec_from_file_location(
        "talker_profile_488", _HERE / "profile_talker_steps.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def power_state() -> List[dict]:
    """Record the clock/power regime. Measurements across regimes do not compare."""
    fields = (
        "index,name,uuid,power.limit,power.default_limit,clocks.max.sm,"
        "clocks.sm,temperature.gpu,persistence_mode,memory.free"
    )
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        return [{"error": str(exc)}]
    keys = fields.split(",")
    return [
        dict(zip(keys, [cell.strip() for cell in line.split(",")]))
        for line in output.strip().splitlines()
    ]


def corridor_report() -> dict:
    """Free VRAM on every card against the standing 400 MiB corridor.

    Read through NVML and not through torch, deliberately. This process runs
    with ``CUDA_VISIBLE_DEVICES`` pinned to one card, so ``torch`` can only see
    that one -- and a corridor check that cannot see the cards it is supposed
    to protect is worse than none, because it reports "all_ok" about a rig it
    never looked at.
    """
    cards = []
    for card in power_state():
        if "error" in card or "memory.free" not in card:
            return {"floor_mib": _CORRIDOR_MIB, "cards": [card], "all_ok": None}
        free = float(card["memory.free"])
        cards.append({
            "index": card["index"],
            "name": card["name"],
            "free_mib": free,
            "corridor_ok": free >= _CORRIDOR_MIB,
        })
    return {"floor_mib": _CORRIDOR_MIB, "cards": cards,
            "all_ok": all(card["corridor_ok"] for card in cards)}


class CorridorBreach(RuntimeError):
    """The rig-wide 400 MiB floor was crossed. Stop, do not continue measuring."""


def enforce_corridor(phase: str, log: List[dict]) -> None:
    """Re-read the corridor between phases and abort on a breach.

    Checking only before and after is what let the first run breach silently:
    the guest's own peak sat in the middle, and by the time the process exited
    the memory was back. A measurement is not worth taking if taking it puts a
    serving rank's card under the floor, so this raises rather than warns.
    """
    state = corridor_report()
    state["phase"] = phase
    log.append(state)
    if state["all_ok"] is False:
        offenders = [
            f"card {card['index']} ({card['name']}): {card['free_mib']:.0f} MiB"
            for card in state["cards"]
            if not card["corridor_ok"]
        ]
        raise CorridorBreach(
            f"after phase '{phase}' the {_CORRIDOR_MIB:.0f} MiB corridor is "
            f"breached on: {'; '.join(offenders)}. Aborting before the next "
            f"phase allocates."
        )


# ---------------------------------------------------------------------------
# gate 1 -- token identity
# ---------------------------------------------------------------------------


def _eager_inverse_cdf(logits, uniform, temperature, top_k, top_p, vocab_size):
    """The eager twin of the graphed sampler, spelled out separately on purpose.

    If both arms called the same function the gate would only prove that a
    graph replays a function -- not that the function survived capture with its
    thresholds and its cache intact. This is written against the same
    definition, and the gate is that they agree.
    """
    import torch

    from sglang.srt.models.qwen3_tts_fast_predictor import apply_warpers

    warped = apply_warpers(logits, temperature, top_k, top_p)
    cumulative = warped.float().softmax(dim=-1).cumsum(dim=-1)
    return torch.searchsorted(cumulative, uniform.clamp(max=1.0)).clamp(
        max=vocab_size - 1
    )


def _eager_static_frame(talker, prompt, uniforms, sample_kwargs, vocab):
    """One predictor frame, eager, over the SAME static cache and masks.

    This is the arm the capture gate compares against, and the choice is the
    correction to ``ANALYSE_488 §7.5``. The design asked for token identity
    against the *reference* (DynamicCache, no mask); measurement on 2026-08-04
    showed that is unachievable, because the static cache is a prerequisite of
    graph capture and perturbs bf16 rounding by ~2e-2 relative all by itself --
    enough to flip near-tied argmaxes -- while being the *same computation*
    (proven in fp32 by ``check_static_cache_semantics.py``).

    Comparing against this arm instead isolates exactly what capture can break
    and demands **bit-exactness** for it, which is a stronger gate than the
    original, not a weaker one.
    """
    import torch
    from transformers import StaticCache

    from sglang.srt.models.qwen3_tts_fast_predictor import step_schedule
    from sglang.srt.models.qwen3_tts_graph_driver import (
        decode_mask,
        predictor_cache_lengths,
    )

    predictor = talker.code_predictor
    model = predictor.model
    groups = talker.config.num_code_groups
    device = next(predictor.parameters()).device
    embeddings = model.get_input_embeddings()
    projection = getattr(predictor, "small_to_mtp_projection", None)
    lengths = predictor_cache_lengths(groups)
    slots = lengths[-1]

    cache = StaticCache(config=model.config, max_cache_len=slots)
    codes = []
    hidden = prompt
    for position, (_, embedding_index, head_index) in enumerate(step_schedule(groups)):
        if embedding_index is not None:
            hidden = embeddings[embedding_index](codes[-1])
        if projection is not None:
            hidden = projection(hidden)
        query_len = hidden.shape[1]
        valid = lengths[position]
        cache_position = torch.arange(valid - query_len, valid, device=device)
        outputs = model(
            inputs_embeds=hidden,
            attention_mask=decode_mask(valid, slots, query_len, str(device)),
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
        logits = predictor.lm_head[head_index](
            outputs.last_hidden_state[:, -1:, :]
        )[:, -1, :]
        if sample_kwargs["do_sample"]:
            codes.append(
                _eager_inverse_cdf(
                    logits, uniforms[:, position : position + 1],
                    sample_kwargs.get("temperature"), sample_kwargs.get("top_k"),
                    sample_kwargs.get("top_p"), vocab,
                )
            )
        else:
            codes.append(logits.argmax(dim=-1, keepdim=True))
    return torch.cat(codes, dim=-1)


def identity_gate(talker, frames: int = 4, seed: int = 488) -> dict:
    """Graphed vs eager-over-the-same-static-cache. Must be BIT-EXACT.

    Several frames and not one: a frame must not inherit cache state from the
    frame before it, and that bug shows up on frame 2, never on frame 1.
    """
    import torch

    from sglang.srt.models.qwen3_tts_graph_driver import GraphedPredictorFrame

    predictor = talker.code_predictor
    groups = talker.config.num_code_groups
    device = next(predictor.parameters()).device
    dtype = next(predictor.parameters()).dtype
    hidden = predictor.model.config.hidden_size
    vocab = predictor.lm_head[0].out_features

    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Sampled on the CPU and moved: torch.randn on the device is not
    # arch-identical, and an input that differs between arms is not a gate.
    prompts = [
        torch.randn(1, 2, hidden, generator=generator).to(device=device, dtype=dtype)
        for _ in range(frames)
    ]
    uniforms = [
        torch.rand(1, groups - 1, generator=generator).to(device=device)
        for _ in range(frames)
    ]

    results: Dict[str, object] = {"frames": frames, "seed": seed}

    for arm, sample_kwargs in (
        ("greedy", {"do_sample": False}),
        ("sampled", {"do_sample": True, "temperature": 0.9, "top_k": 50, "top_p": 1.0}),
    ):
        graphed = GraphedPredictorFrame(predictor, groups).capture(**sample_kwargs)

        mismatches = 0
        total = 0
        first_bad = None
        for frame in range(frames):
            with torch.inference_mode():
                eager_tokens = _eager_static_frame(
                    talker, prompts[frame], uniforms[frame], sample_kwargs, vocab
                )
                graph_tokens = graphed.generate(
                    prompts[frame], max_new_tokens=groups - 1,
                    uniforms=uniforms[frame], **sample_kwargs,
                ).sequences
            equal = torch.equal(eager_tokens, graph_tokens)
            total += eager_tokens.numel()
            if not equal:
                mismatches += int((eager_tokens != graph_tokens).sum())
                if first_bad is None:
                    first_bad = {
                        "frame": frame,
                        "eager": eager_tokens.flatten().tolist(),
                        "graphed": graph_tokens.flatten().tolist(),
                    }
        results[arm] = {
            "tokens_compared": total,
            "mismatches": mismatches,
            "identical": mismatches == 0,
            "first_mismatch": first_bad,
        }
        del graphed
        torch.cuda.empty_cache()

    results["passed"] = all(
        results[arm]["identical"] for arm in ("greedy", "sampled")
    )
    return results


def _eager_frame_sampled(driver, prompt, uniform, sample_kwargs, vocab, sampler):
    """Run slice 1's loop but with the inverse-CDF draw, so the arms are comparable.

    Slice 1 samples with ``torch.multinomial``, which consumes entropy the
    graphed arm cannot be given. Driving both from the same uniforms is what
    makes the sampled arm a gate rather than a coincidence test.
    """
    import torch
    from transformers import DynamicCache

    predictor = driver.predictor
    model = predictor.model
    projection = getattr(predictor, "small_to_mtp_projection", None)
    embeddings = model.get_input_embeddings()
    cache = DynamicCache()
    codes = []
    hidden = prompt
    for position, (_, embedding_index, head_index) in enumerate(driver.schedule):
        if embedding_index is not None:
            hidden = embeddings[embedding_index](codes[-1])
        if projection is not None:
            hidden = projection(hidden)
        outputs = model(inputs_embeds=hidden, past_key_values=cache, use_cache=True)
        logits = predictor.lm_head[head_index](
            outputs.last_hidden_state[:, -1:, :]
        )[:, -1, :]
        codes.append(
            sampler(
                logits, uniform[:, position : position + 1],
                sample_kwargs.get("temperature"), sample_kwargs.get("top_k"),
                sample_kwargs.get("top_p"), vocab,
            )
        )
    return torch.cat(codes, dim=-1)


def reference_divergence(talker, frames: int = 4, seed: int = 488) -> dict:
    """How far the static-cache path's TOKENS drift from today's reference.

    Reported, never gated. The two paths are the same computation
    (``check_static_cache_semantics.py``, fp32), but the code predictor's
    argmax is near-tied often enough that bf16 reduction-order noise changes
    which codebook entry wins. The resulting audio is a different valid sample,
    not a degraded one -- but "different" is a fact the listening arm needs to
    be told in advance, so it is measured rather than discovered.
    """
    import torch

    from sglang.srt.models.qwen3_tts_fast_predictor import FastCodePredictor

    predictor = talker.code_predictor
    groups = talker.config.num_code_groups
    device = next(predictor.parameters()).device
    dtype = next(predictor.parameters()).dtype
    hidden = predictor.model.config.hidden_size
    vocab = predictor.lm_head[0].out_features

    generator = torch.Generator(device="cpu").manual_seed(seed)
    driver = FastCodePredictor(predictor, groups)
    greedy = {"do_sample": False}
    differing = 0
    total = 0
    for _ in range(frames):
        prompt = torch.randn(1, 2, hidden, generator=generator).to(
            device=device, dtype=dtype
        )
        with torch.inference_mode():
            reference = driver.generate(
                prompt, max_new_tokens=groups - 1, do_sample=False
            ).sequences
            static = _eager_static_frame(talker, prompt, None, greedy, vocab)
        differing += int((reference != static).sum())
        total += reference.numel()
    return {
        "frames": frames,
        "tokens_compared": total,
        "tokens_differing": differing,
        "divergence_rate": round(differing / total, 4),
        "note": (
            "Same computation, different bf16 reduction order (proven in fp32 "
            "by check_static_cache_semantics.py). Observability, not a gate."
        ),
    }


def gate_loaded_model(model, frames: int = 3, keep_installed: bool = False) -> dict:
    """THE HANDOFF ENTRY POINT: gate the install seam inside a LIVE tenant.

    Takes an already-loaded ``Qwen3TTSModel`` wrapper -- the thing
    ``InProcessQwen3Tts._model`` holds -- and proves, in that process, on those
    weights, that installing the graph driver changes nothing about what the
    code predictor computes. No second copy of the model, no server restart,
    no conversation state touched, no audio synthesised.

    Safe to call in the live tenant, and each of those is a property, not a
    hope: it allocates ~60 MiB (fifteen graphs sharing one pool, a 16-slot
    scratch cache and a 4096-frame uniform pool), it runs a handful of frames,
    and unless ``keep_installed=True`` it puts the reference ``generate`` back
    before returning.

    The sampling is not a parameter. It is read from the reference's own
    ``subtalker_*`` defaults (:func:`reference_subtalker_defaults`), because
    the one way to get this wrong is to pass the tenant's ``temperature/top_p``
    -- which belong to the trunk, not the predictor.
    """
    import torch

    from sglang.srt.models.qwen3_tts_graph_driver import (
        GraphedPredictorFrame,
        reference_subtalker_defaults,
    )

    inner = getattr(model, "model", model)
    talker = getattr(inner, "talker", None)
    if talker is None:
        raise ValueError(
            "no .talker on the given model; expected the reference "
            "Qwen3TTSModel wrapper (see translator/inprocess_tts.py)"
        )
    predictor = talker.code_predictor
    groups = talker.config.num_code_groups
    device = next(predictor.parameters()).device
    dtype = next(predictor.parameters()).dtype
    hidden = predictor.model.config.hidden_size
    vocab = predictor.lm_head[0].out_features

    sampling = reference_subtalker_defaults(model)
    report: Dict[str, object] = {
        "sampling_read_from_reference": dict(sampling),
        "corridor_before": corridor_report(),
    }

    free_before = torch.cuda.mem_get_info(device.index or 0)[0] / (1024 * 1024)
    driver = GraphedPredictorFrame.install(talker, model=model, **sampling)
    free_after = torch.cuda.mem_get_info(device.index or 0)[0] / (1024 * 1024)
    report["vram"] = {
        "allocator_delta_mib": driver.vram_cost_mib(),
        "card_free_before_mib": round(free_before, 1),
        "card_free_after_mib": round(free_after, 1),
        "card_delta_mib": round(free_before - free_after, 1),
    }

    try:
        generator = torch.Generator(device="cpu").manual_seed(488)
        mismatches = 0
        total = 0
        for _ in range(frames):
            prompt = torch.randn(1, 2, hidden, generator=generator).to(
                device=device, dtype=dtype
            )
            uniforms = torch.rand(1, groups - 1, generator=generator).to(device=device)
            with torch.inference_mode():
                eager = _eager_static_frame(
                    talker, prompt, uniforms,
                    {"do_sample": sampling["do_sample"], **{
                        k: sampling[k] for k in ("temperature", "top_k", "top_p")
                    }}, vocab,
                )
                # Through the INSTALLED seam, i.e. exactly the call the
                # reference makes -- not the driver object directly.
                graphed = predictor.generate(
                    inputs_embeds=prompt, max_new_tokens=groups - 1,
                    uniforms=uniforms, output_hidden_states=True,
                    return_dict_in_generate=True, **sampling,
                ).sequences
            total += eager.numel()
            mismatches += int((eager != graphed).sum())
        report["capture_gate"] = {
            "frames": frames,
            "tokens_compared": total,
            "mismatches": mismatches,
            "identical": mismatches == 0,
        }
        report["sampling_fallbacks"] = driver.sampling_fallbacks
        report["passed"] = mismatches == 0 and driver.sampling_fallbacks == 0
    finally:
        if not keep_installed:
            driver.uninstall()
            release()
    report["installed_on_return"] = keep_installed
    report["corridor_after"] = corridor_report()
    report["verdict"] = (
        "INSTALL SEAM GREEN -- the graphed predictor is bit-identical to the "
        "eager path on this instance's own weights."
        if report["passed"]
        else "REFUSED -- the installed seam diverged; do not wire this in."
    )
    return report


def trunk_gate(talker, steps: int = 8) -> dict:
    """Three trunk arms, and each pair answers a different question.

    ``dynamic``  -- DynamicCache, no mask: what the tenant runs today.
    ``static``   -- StaticCache + our 4-D mask, eager: the prerequisite of capture.
    ``graphed``  -- the same, captured.

    **static vs graphed is the gate, and it must be bit-exact.** That is the
    only pair in which a capture bug can hide.

    **dynamic vs static is reported, not gated.** It is nonzero in bf16 by
    reduction order alone; ``check_static_cache_semantics.py`` settles in fp32
    that the two are the same computation. What is still worth watching here is
    whether the difference GROWS with the step index -- growth would mean the
    mask is admitting a stale slot, which rounding cannot cause.
    """
    import torch
    from transformers import DynamicCache, StaticCache

    from sglang.srt.models.qwen3_tts_graph_driver import (
        GraphedTrunkStep,
        decode_mask,
        reset_cache_positions,
    )

    trunk = talker.model
    device = next(trunk.parameters()).device
    dtype = next(trunk.parameters()).dtype
    hidden_size = trunk.config.hidden_size
    prefill_len = 6
    slots = 1024

    generator = torch.Generator(device="cpu").manual_seed(488)
    prefill = torch.randn(1, prefill_len, hidden_size, generator=generator).to(
        device=device, dtype=dtype
    )
    inputs = [
        torch.randn(1, 1, hidden_size, generator=generator).to(device=device, dtype=dtype)
        for _ in range(steps)
    ]

    with torch.inference_mode():
        dynamic_cache = DynamicCache()
        trunk(inputs_embeds=prefill, past_key_values=dynamic_cache, use_cache=True)
        dynamic_out = [
            trunk(
                inputs_embeds=tensor, past_key_values=dynamic_cache, use_cache=True
            ).last_hidden_state.float().clone()
            for tensor in inputs
        ]

        # Each arm's cache is released before the next allocates. Three
        # 1024-slot trunk caches are 117 MiB each, and holding all three at
        # once is a third of what put the 5090 under the corridor on the first
        # run. The outputs are already detached copies, so nothing is lost.
        del dynamic_cache
        release()

        static_cache = StaticCache(config=trunk.config, max_cache_len=slots)
        trunk(
            inputs_embeds=prefill, past_key_values=static_cache, use_cache=True,
            cache_position=torch.arange(prefill_len, device=device),
        )
        static_out = []
        for index in range(steps):
            position = prefill_len + index
            cache_position = torch.tensor([position], device=device)
            static_out.append(
                trunk(
                    inputs_embeds=inputs[index],
                    attention_mask=decode_mask(position + 1, slots, 1, str(device)),
                    position_ids=cache_position.view(1, 1, 1).expand(3, 1, 1),
                    cache_position=cache_position,
                    past_key_values=static_cache, use_cache=True,
                ).last_hidden_state.float().clone()
            )

        del static_cache
        release()

        graphed = GraphedTrunkStep(trunk, max_positions=slots)
        graphed.capture(prefill_len=prefill_len)
        # Prefill stays eager by design (ANALYSE_488 §7.3): it is dynamic-length
        # and runs once per clause, off the per-frame path.
        reset_cache_positions(graphed.cache)
        trunk(
            inputs_embeds=prefill, past_key_values=graphed.cache, use_cache=True,
            cache_position=torch.arange(prefill_len, device=device),
        )
        graph_out = []
        for index, tensor in enumerate(inputs):
            if index:
                graphed.advance()
            graph_out.append(graphed.forward(tensor).float().clone())

    capture_deltas = [
        float((a - b).abs().max()) for a, b in zip(static_out, graph_out)
    ]
    padding_deltas = [
        float((a - b).abs().max()) for a, b in zip(dynamic_out, static_out)
    ]
    scale = max(float(a.abs().max()) for a in dynamic_out)
    return {
        "steps": steps,
        "hidden_state_scale": round(scale, 4),
        "capture_exact": max(capture_deltas) == 0.0,
        "capture_max_abs_delta": max(capture_deltas),
        "padding_max_abs_delta": round(max(padding_deltas), 6),
        "padding_max_relative": round(max(padding_deltas) / scale, 6),
        "padding_grows_with_step": (
            padding_deltas[-1] > 4 * max(padding_deltas[0], 1e-9)
        ),
        "passed": max(capture_deltas) == 0.0,
    }


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------


def ladder(profiler, talker) -> dict:
    """Every rung, back to back, in one window, on one card."""
    import torch

    from sglang.srt.models.qwen3_tts_fast_predictor import FastCodePredictor
    from sglang.srt.models.qwen3_tts_graph_driver import (
        GraphedPredictorFrame,
        GraphedTrunkStep,
    )

    trunk = talker.model
    predictor = talker.code_predictor
    groups = talker.config.num_code_groups
    device = next(trunk.parameters()).device
    dtype = next(trunk.parameters()).dtype
    hidden_size = trunk.config.hidden_size

    generator = torch.Generator(device="cpu").manual_seed(488)
    embeds = torch.randn(1, 1, hidden_size, generator=generator).to(
        device=device, dtype=dtype
    )
    prompt = torch.randn(1, 2, hidden_size, generator=generator).to(
        device=device, dtype=dtype
    )
    uniforms = torch.rand(1, groups - 1, generator=generator).to(device=device)

    sampling = {"do_sample": True, "temperature": 0.9, "top_k": 50, "top_p": 1.0}
    eager_driver = FastCodePredictor(predictor, groups)
    graphed_predictor = GraphedPredictorFrame(predictor, groups).capture(**sampling)
    graphed_trunk = GraphedTrunkStep(trunk, max_positions=1024).capture(prefill_len=1)

    def reference_generate() -> None:
        with torch.inference_mode():
            predictor.generate(
                inputs_embeds=prompt, max_new_tokens=groups - 1, do_sample=False,
                output_hidden_states=True, return_dict_in_generate=True,
            )

    def eager_frame() -> None:
        with torch.inference_mode():
            eager_driver.generate(prompt, max_new_tokens=groups - 1, do_sample=False)

    def graphed_frame() -> None:
        with torch.inference_mode():
            graphed_predictor.generate(
                prompt, max_new_tokens=groups - 1, uniforms=uniforms, **sampling
            )

    def trunk_eager() -> None:
        with torch.inference_mode():
            trunk(inputs_embeds=embeds, use_cache=False)

    def trunk_graphed() -> None:
        with torch.inference_mode():
            graphed_trunk.forward(embeds)

    arms = {}
    plan = (
        ("predictor_generate_reference", reference_generate, 10,
         "the HF generate() envelope the reference re-enters per frame"),
        ("predictor_frame_eager", eager_frame, 20,
         "slice 1: the raw 15-step loop, no graphs"),
        ("predictor_frame_graphed", graphed_frame, 50,
         "slice 2: 15 graph replays"),
        ("trunk_step_eager", trunk_eager, 30,
         "one 28-layer trunk forward, batch 1, one position"),
        ("trunk_step_graphed", trunk_graphed, 50,
         "slice 2: the 28-layer trunk as one graph replay"),
    )
    for name, fn, iterations, note in plan:
        arms[name] = profiler._profile_region(name, fn, iterations=iterations, note=note)
        logger.info(
            "%-32s %8.3f ms/iter  kernel %7.3f ms  gap %5.1f %%",
            name, arms[name].wall_per_iter_ms, arms[name].kernel_per_iter_ms,
            100 * arms[name].gap_fraction,
        )

    def frame(trunk_arm: str, predictor_arm: str) -> dict:
        wall = arms[trunk_arm].wall_per_iter_ms + arms[predictor_arm].wall_per_iter_ms
        kernel = (
            arms[trunk_arm].kernel_per_iter_ms + arms[predictor_arm].kernel_per_iter_ms
        )
        return {
            "frame_ms": round(wall, 3),
            "frame_kernel_ms": round(kernel, 3),
            # 12 Hz codec: one audio-second is twelve frames.
            "rtf": round(wall * 12.0 / 1000.0, 4),
            "kernel_only_rtf": round(kernel * 12.0 / 1000.0, 4),
        }

    rungs = {
        "reference": frame("trunk_step_eager", "predictor_generate_reference"),
        "slice1_raw_loop": frame("trunk_step_eager", "predictor_frame_eager"),
        "slice2_predictor_graphs": frame("trunk_step_eager", "predictor_frame_graphed"),
        "slice2_both_graphs": frame("trunk_step_graphed", "predictor_frame_graphed"),
    }
    baseline = rungs["reference"]["frame_ms"]
    for name, rung in rungs.items():
        rung["speedup_vs_reference"] = round(baseline / rung["frame_ms"], 3)

    return {
        "arms": {name: arm.to_json() for name, arm in arms.items()},
        "rungs": rungs,
        "note": (
            "Frames are DERIVED from the step arms (trunk + predictor envelope), "
            "the same derivation the precursor used, so the rungs compare to each "
            "other and to that report's method -- not to its absolute ms, which "
            "came from a different boot."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base",
    )
    parser.add_argument("--json", default=None)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument(
        "--skip-ladder", action="store_true",
        help="run only the identity gate (cheap, for iterating on correctness)",
    )
    parser.add_argument(
        "--install-seam-only", action="store_true",
        help=(
            "run only gate_loaded_model against a standalone copy -- the same "
            "check agent 8 runs in-process against the live tenant"
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    profiler = _load_profiler()
    profiler._tag_process()
    for line in install_local_modules():
        logger.info("local module: %s", line)

    import torch

    free_mib = torch.cuda.mem_get_info(0)[0] / (1024 * 1024)
    if free_mib - _NEED_MIB < _CORRIDOR_MIB:
        logger.error(
            "REFUSED: %.0f MiB free, this run needs ~%.0f MiB and the corridor "
            "requires %.0f MiB to remain.",
            free_mib, _NEED_MIB, _CORRIDOR_MIB,
        )
        return 2

    report: Dict[str, object] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "power_state_before": power_state(),
        "corridor_before": corridor_report(),
    }

    try:
        return _run(args, profiler, report)
    except CorridorBreach as breach:
        # A breach is a RESULT, reported with the phase log that shows where
        # the peak was, not a traceback that loses it.
        report["verdict"] = f"ABORTED -- {breach}"
        report["corridor_after"] = corridor_report()
        text = json.dumps(report, indent=2)
        print(text)
        if args.json:
            pathlib.Path(args.json).write_text(text + "\n", encoding="utf-8")
        logger.error("%s", report["verdict"])
        return 3


def _run(args, profiler, report) -> int:
    from sglang.srt.translator.inprocess_tts import (
        InProcessQwen3Tts,
        InProcessTtsConfig,
    )

    tts = InProcessQwen3Tts(InProcessTtsConfig(model_dir=pathlib.Path(args.model_dir)))
    tts.load()
    inner = getattr(tts._model, "model", tts._model)
    talker = inner.talker

    # The instrument must clear its own discrimination gate before any arm of
    # it testifies -- and the gpu_bound arm doubles as this window's contention
    # reading (ANALYSE_488 §7.7).
    device = str(next(talker.model.parameters()).device)
    calibration = profiler._calibration_arms(device)
    discrimination = profiler.check_discrimination(
        calibration["calib_gpu_bound"], calibration["calib_launch_bound"]
    )
    report["discrimination"] = discrimination.to_json()
    report["calibration"] = {k: v.to_json() for k, v in calibration.items()}

    phases: List[dict] = []
    enforce_corridor("weights_loaded", phases)

    if args.install_seam_only:
        report["install_seam"] = gate_loaded_model(tts._model)
        report["verdict"] = report["install_seam"]["verdict"]
        report["power_state_after"] = power_state()
        report["corridor_after"] = corridor_report()
        text = json.dumps(report, indent=2)
        print(text)
        if args.json:
            pathlib.Path(args.json).write_text(text + "\n", encoding="utf-8")
        logger.info("%s", report["verdict"])
        return 0 if report["install_seam"]["passed"] else 1

    logger.info("gate 1: predictor capture, graphed vs eager over the same cache")
    report["identity_gate"] = identity_gate(talker, frames=args.frames)
    release()
    enforce_corridor("identity_gate", phases)

    logger.info("gate 2: trunk capture, graphed vs eager over the same cache")
    report["trunk_gate"] = trunk_gate(talker)
    release()
    enforce_corridor("trunk_gate", phases)

    logger.info("observability: token drift of the static-cache path vs today")
    report["reference_divergence"] = reference_divergence(talker, frames=args.frames)
    release()
    enforce_corridor("reference_divergence", phases)
    report["phase_headroom"] = phases

    gates_ok = report["identity_gate"]["passed"] and report["trunk_gate"]["passed"]
    report["gates_passed"] = gates_ok

    if not gates_ok:
        # Deliberate: a wrong graph is FASTER, so publishing its timings next to
        # a failed gate is how a wrong number gets quoted later.
        report["verdict"] = (
            "REFUSED -- the correctness gates did not pass, so no timing was "
            "taken. A graph that computes the wrong thing is faster than one "
            "that computes the right thing."
        )
    elif args.skip_ladder:
        report["verdict"] = "GATES PASSED -- ladder skipped by request."
    elif not discrimination.ok:
        report["verdict"] = (
            "GATES PASSED, ladder REFUSED -- " + discrimination.reason
        )
    else:
        logger.info("ladder")
        report["ladder"] = ladder(profiler, talker)
        release()
        enforce_corridor("ladder", phases)
        rungs = report["ladder"]["rungs"]
        report["verdict"] = (
            f"GATES PASSED. RTF {rungs['reference']['rtf']:.3f} (reference) -> "
            f"{rungs['slice2_both_graphs']['rtf']:.3f} (both graphs), "
            f"{rungs['slice2_both_graphs']['speedup_vs_reference']:.2f}x."
        )

    report["power_state_after"] = power_state()
    report["corridor_after"] = corridor_report()
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        pathlib.Path(args.json).write_text(text + "\n", encoding="utf-8")
    logger.info("%s", report["verdict"])
    return 0 if gates_ok else 1


if __name__ == "__main__":
    sys.exit(main())
