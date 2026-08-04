# SPDX-License-Identifier: Apache-2.0
"""#488: what a GRAPHED talker instance costs in VRAM, measured per term.

WHY THIS EXISTS
---------------
The pool decision rests on one number the project does not have: the resident
footprint of a talker instance **with its graphs captured**. The 2678 MiB on
record (`2026-08-04_488_precursor`) is the *eager* instance at rest. Graphs do
not shrink that — they ADD static caches and memory pools — so using 2678 to
decide whether an instance fits on a 3080 understates it.

WHAT IS MEASURED HERE AND WHAT IS NOT
-------------------------------------
Measuring the whole graphed instance needs ~2.9 GiB, and no card on this rig
currently has that free above the 400 MiB corridor. But the term that is
*missing* — the graph overhead — does not depend on weight VALUES, only on
shapes. So this probe instantiates the **real geometry** with random weights,
captures against it, and measures each increment exactly:

    graphed_instance = 2678 (measured eager, precursor) + graph overhead (here)

Both terms measured, the composition stated rather than hidden. The overhead is
deliberately measured on an **sm86 card**, because that is where a pooled
instance would live and graph pools are not architecture-independent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pathlib
import sys
import types

logger = logging.getLogger("488-footprint")
_HERE = pathlib.Path(__file__).resolve().parent
_EAGER_INSTANCE_MIB = 2678.0  # measured, 2026-08-04_488_precursor
_CORRIDOR_MIB = 400.0


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
    def generate_voice_clone(
        self, text=None, subtalker_dosample: bool = True,
        subtalker_top_k: int = 50, subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
    ):
        raise NotImplementedError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base/config.json",
    )
    parser.add_argument("--trunk-slots", type=int, default=1024)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from sglang.srt.translator.qwen3_tts_compat import ensure_qwen3_tts_importable

    ensure_qwen3_tts_importable()
    _install_local_modules()

    import torch
    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
    )
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
        Qwen3TTSTalkerModel,
    )

    from sglang.srt.models.qwen3_tts_graph_driver import (
        GraphedPredictorFrame,
        GraphedTrunkStep,
    )

    raw = json.loads(pathlib.Path(args.config).read_text())["talker_config"]
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    free_start = torch.cuda.mem_get_info(0)[0] / (1024 * 1024)
    name = torch.cuda.get_device_name(0)
    capability = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    logger.info("card: %s (sm%s), %.0f MiB free", name, capability, free_start)

    def mib() -> float:
        return torch.cuda.memory_allocated(0) / (1024 * 1024)

    # Real geometry, random weights: graph pools depend on shapes, not values.
    talker_keys = (
        "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "vocab_size", "num_code_groups", "max_position_embeddings",
        "text_hidden_size",
        # M-RoPE is not optional geometry: the trunk's attention indexes
        # rope_scaling["mrope_section"] and ["interleaved"] directly
        # (modeling_qwen3_tts.py:779), so a config without it does not run at
        # all -- and the section split changes the captured kernels.
        "rope_scaling", "rope_theta", "rms_norm_eps", "hidden_act",
        "attention_bias", "sliding_window", "use_sliding_window",
    )
    talker_config = Qwen3TTSTalkerConfig(**{
        k: raw[k] for k in talker_keys if k in raw
    })
    # The real text_vocab_size (151936) would allocate a 593 MiB embedding that
    # NO captured region touches: the trunk decode step takes inputs_embeds
    # directly, and prompt build stays eager by design (ANALYSE_488 §7.3).
    # Shrinking it keeps this probe ~600 MiB smaller without changing a single
    # byte of the graph overhead being measured -- and the composed total takes
    # the full-size embedding from the measured 2678 MiB base anyway.
    talker_config.text_vocab_size = 256
    predictor_raw = raw["code_predictor_config"]
    predictor_config = Qwen3TTSTalkerCodePredictorConfig(**{
        k: predictor_raw[k] for k in (
            "num_hidden_layers", "hidden_size", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "num_code_groups", "max_position_embeddings",
        ) if k in predictor_raw
    })

    base = mib()
    trunk = Qwen3TTSTalkerModel(talker_config).to(device=device, dtype=dtype).eval()
    trunk_weights = mib() - base
    predictor = (
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
            predictor_config, talker_config
        ).to(device=device, dtype=dtype).eval()
    )
    predictor_weights = mib() - base - trunk_weights
    logger.info(
        "weights (this probe, random init): trunk %.1f MiB, predictor %.1f MiB",
        trunk_weights, predictor_weights,
    )

    talker = types.SimpleNamespace(
        code_predictor=predictor,
        config=types.SimpleNamespace(
            num_code_groups=talker_config.num_code_groups
        ),
    )

    before_predictor = mib()
    driver = GraphedPredictorFrame.install(
        talker=talker, model=_ReferenceStub(), do_sample=True,
        temperature=0.9, top_k=50, top_p=1.0,
    )
    predictor_graphs = mib() - before_predictor

    before_trunk = mib()
    graphed_trunk = GraphedTrunkStep(trunk, max_positions=args.trunk_slots)
    graphed_trunk.capture(prefill_len=1)
    trunk_graphs = mib() - before_trunk

    overhead = predictor_graphs + trunk_graphs
    free_end = torch.cuda.mem_get_info(0)[0] / (1024 * 1024)

    report = {
        "card": name,
        "compute_capability": f"sm{capability}",
        "trunk_static_slots": args.trunk_slots,
        "measured": {
            "predictor_graphs_and_pool_mib": round(predictor_graphs, 2),
            "trunk_graph_and_static_cache_mib": round(trunk_graphs, 2),
            "graph_overhead_total_mib": round(overhead, 2),
            "probe_trunk_weights_mib": round(trunk_weights, 1),
            "probe_predictor_weights_mib": round(predictor_weights, 1),
        },
        "composed": {
            "eager_instance_mib": _EAGER_INSTANCE_MIB,
            "eager_instance_source": "2026-08-04_488_precursor, measured at rest",
            "graphed_instance_mib": round(_EAGER_INSTANCE_MIB + overhead, 1),
        },
        "card_free_start_mib": round(free_start, 1),
        "card_free_end_mib": round(free_end, 1),
    }
    logger.info(
        "GRAPH OVERHEAD: predictor %.1f + trunk %.1f = %.1f MiB",
        predictor_graphs, trunk_graphs, overhead,
    )
    logger.info(
        "GRAPHED INSTANCE = %.0f (eager, measured) + %.0f (overhead, measured) "
        "= %.0f MiB",
        _EAGER_INSTANCE_MIB, overhead, _EAGER_INSTANCE_MIB + overhead,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        pathlib.Path(args.json).write_text(text + "\n", encoding="utf-8")
    del driver, graphed_trunk
    return 0


if __name__ == "__main__":
    sys.exit(main())
