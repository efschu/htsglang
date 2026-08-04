# SPDX-License-Identifier: Apache-2.0
"""#488 Step A: execution smoke for the install seam, on a TINY model.

WHY A TINY MODEL AND NOT THE CHECKPOINT
---------------------------------------
What Step A adds over slice 2 is *mechanics*, not weights: a device-resident
uniform pool whose cursor advances inside a captured graph, an ``install()``
that validates its sampling against the reference's own signature, and the
seam that swaps ``code_predictor.generate``. None of that depends on having
0.6 B real parameters, and all of it is exactly the kind of code that must not
ship desk-written.

A randomly-initialised 2-layer predictor exercises every one of those paths for
**~50 MiB**, which means this smoke runs on a contended card without competing
with the live tenant for 2.7 GiB. The full-weights gate is a different check
with a different home: :func:`validate_graphs.gate_loaded_model`, run
in-process against the tenant that already holds the weights.

WHAT IT PROVES
--------------
1. capture works against a real ``Qwen3TTSTalkerCodePredictorModel``;
2. the uniform pool advances per frame **inside** the graph -- consecutive
   frames with identical inputs must produce DIFFERENT tokens, which is the
   whole reason the pool exists and the thing a baked draw would break;
3. supplying uniforms explicitly reproduces the eager path **bit-exactly**;
4. the seam is live: calling ``predictor.generate`` goes through the graphs;
5. ``uninstall()`` puts the reference back;
6. the sampling-trap refusal fires on the tenant's trunk values.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import pathlib
import sys
import types

logger = logging.getLogger("488-smoke-install-seam")
_HERE = pathlib.Path(__file__).resolve().parent


def _install_local_modules():
    for name, rel in (
        ("sglang.srt.models.qwen3_tts_fast_predictor",
         "python/sglang/srt/models/qwen3_tts_fast_predictor.py"),
        ("sglang.srt.models.qwen3_tts_graph_driver",
         "python/sglang/srt/models/qwen3_tts_graph_driver.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, _HERE.parents[2] / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


class _ReferenceStub:
    """Carries the signature ``install()`` reads the predictor sampling from."""

    def generate_voice_clone(
        self, text=None, subtalker_dosample: bool = True,
        subtalker_top_k: int = 50, subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
    ):
        raise NotImplementedError


def build_tiny_talker(device, dtype, groups: int = 4):
    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
    )
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    )

    config = Qwen3TTSTalkerCodePredictorConfig(
        num_hidden_layers=2, hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=32, num_code_groups=groups, max_position_embeddings=128,
    )
    # Matching hidden_size on purpose: it makes small_to_mtp_projection an
    # Identity, exactly as on the real checkpoint (both are 1024 there), so the
    # smoke exercises the same branch production takes.
    talker_config = Qwen3TTSTalkerConfig(
        hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_code_groups=groups,
    )
    predictor = (
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
            config, talker_config
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    talker = types.SimpleNamespace(
        code_predictor=predictor,
        config=types.SimpleNamespace(num_code_groups=groups),
    )
    return talker


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from sglang.srt.translator.qwen3_tts_compat import ensure_qwen3_tts_importable

    ensure_qwen3_tts_importable()
    _install_local_modules()

    import torch

    from sglang.srt.models.qwen3_tts_graph_driver import (
        GraphCaptureRefusal,
        GraphedPredictorFrame,
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    groups = args.groups
    talker = build_tiny_talker(device, dtype, groups)
    predictor = talker.code_predictor
    reference = _ReferenceStub()
    hidden = predictor.model.config.hidden_size

    free_before = torch.cuda.mem_get_info(0)[0] / (1024 * 1024)
    failures = []

    # -- 6. the trap refusal, first: it must fire before anything is installed
    try:
        GraphedPredictorFrame.install(
            talker=talker, model=reference, do_sample=True,
            temperature=0.9, top_k=50, top_p=0.9,  # 0.9 is the TRUNK's top_p
        )
        failures.append("the tenant's trunk top_p=0.9 was NOT refused")
    except GraphCaptureRefusal as refusal:
        logger.info("trap refusal fired as designed: %.90s...", str(refusal))

    # Compare the underlying FUNCTION, not the bound method: attribute access
    # builds a fresh bound-method object every time, so `is` on
    # `predictor.generate` can never match and would report a false failure.
    original_func = predictor.generate.__func__
    driver = GraphedPredictorFrame.install(
        talker=talker, model=reference, do_sample=True,
        temperature=0.9, top_k=50, top_p=1.0,
    )

    # -- 4. the seam is live
    if getattr(predictor.generate, "__func__", None) is original_func:
        failures.append("install() did not replace code_predictor.generate")

    generator = torch.Generator(device="cpu").manual_seed(488)
    prompt = torch.randn(1, 2, hidden, generator=generator).to(
        device=device, dtype=dtype
    )

    # -- 2. the pool advances INSIDE the graph: same input, different frames
    with torch.inference_mode():
        first = predictor.generate(
            inputs_embeds=prompt, max_new_tokens=groups - 1, do_sample=True,
            temperature=0.9, top_k=50, top_p=1.0,
        ).sequences.clone()
        second = predictor.generate(
            inputs_embeds=prompt, max_new_tokens=groups - 1, do_sample=True,
            temperature=0.9, top_k=50, top_p=1.0,
        ).sequences.clone()
    advanced = not torch.equal(first, second)
    logger.info(
        "pool advance: frame1=%s frame2=%s -> %s",
        first.flatten().tolist(), second.flatten().tolist(),
        "DIFFERENT (correct)" if advanced else "IDENTICAL (pool is baked!)",
    )
    if not advanced:
        failures.append(
            "consecutive frames with identical input produced identical "
            "tokens -- the uniform draw is baked into the graph"
        )

    # -- 3. explicit uniforms reproduce the eager path bit-exactly
    sys.path.insert(0, str(_HERE))
    from validate_graphs import _eager_static_frame  # noqa: PLC0415

    sampling = {"do_sample": True, "temperature": 0.9, "top_k": 50, "top_p": 1.0}
    vocab = predictor.lm_head[0].out_features
    mismatches = 0
    compared = 0
    for _ in range(3):
        uniforms = torch.rand(1, groups - 1, generator=generator).to(device=device)
        with torch.inference_mode():
            eager = _eager_static_frame(talker, prompt, uniforms, sampling, vocab)
            graphed = predictor.generate(
                inputs_embeds=prompt, max_new_tokens=groups - 1,
                uniforms=uniforms, **sampling,
            ).sequences
        compared += eager.numel()
        mismatches += int((eager != graphed).sum())
    logger.info("bit-exactness: %d mismatches of %d tokens", mismatches, compared)
    if mismatches:
        failures.append(f"{mismatches}/{compared} tokens diverged from eager")

    # -- 5. uninstall restores the reference
    driver.uninstall()
    if predictor.generate.__func__ is not original_func:
        failures.append("uninstall() did not restore the reference generate")

    free_after = torch.cuda.mem_get_info(0)[0] / (1024 * 1024)
    logger.info(
        "VRAM: allocator delta %.2f MiB, card free %.0f -> %.0f MiB, "
        "fallbacks %d",
        driver.vram_cost_mib(), free_before, free_after,
        driver.sampling_fallbacks,
    )

    if failures:
        for failure in failures:
            logger.error("FAILED: %s", failure)
        return 1
    logger.info("SMOKE GREEN: all six install-seam properties hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
