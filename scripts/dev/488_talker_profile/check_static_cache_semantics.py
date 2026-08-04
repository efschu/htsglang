# SPDX-License-Identifier: Apache-2.0
"""#488: is the dynamic-vs-static difference bf16 rounding, or semantics?

WHY THIS SCRIPT EXISTS
----------------------
Graph capture needs a ``StaticCache``, and a static cache is padded, and a
padded cache needs an explicit mask to keep unwritten slots out. On the GPU in
bf16 that path disagrees with the reference (``DynamicCache``, no mask) by
**~2e-2 relative**, which is enough to flip a near-tied argmax in the code
predictor and change roughly a quarter of a frame's codec tokens.

Two explanations fit that observation and they have opposite consequences:

* **rounding** -- the explicit mask makes sdpa pick a different backend, whose
  reduction order differs; at bf16 (eps 7.8e-3) a few ulp of accumulated
  difference is exactly this size. Harmless: the same function, computed as
  accurately as before.
* **semantics** -- the mask admits or excludes a slot the reference does not,
  in which case the graph path computes something else and must not ship.

Run in float32 on the CPU, the two explanations separate cleanly: rounding
collapses to fp32 scale, a semantic difference does not.

RESULT, 2026-08-04 (this script, on the real checkpoint)
--------------------------------------------------------
``|A-B| relative = 1.4e-6 .. 2.4e-6`` over three decode steps, against
``fp32 eps = 1.2e-7`` -- i.e. 12-20 eps accumulated over 28 layers. **Rounding.**
The static+mask path is the same computation; only its reduction order differs.

CONSEQUENCE FOR THE PLAN: ``ANALYSE_488 §7.5`` gate 1 asked for identical codec
tokens against the eager reference. That is **not achievable**, and not because
the cut is wrong -- the static cache is a *prerequisite* of graphs and it
perturbs bf16 rounding on its own, before any graph exists. The gate is
therefore split in two (see ``validate_graphs.py``): capture is checked against
the same static-cache path and must be **bit-exact**, which is strictly
stronger than the original gate for what capture can break; and the padding
change is checked here, for semantics, in a precision where the question is
answerable.
"""
import importlib.util
import pathlib
import sys

import torch

HERE = pathlib.Path("/spinning/wt-488-talker-lane")
for name, rel in (
    ("sglang.srt.models.qwen3_tts_fast_predictor",
     "python/sglang/srt/models/qwen3_tts_fast_predictor.py"),
    ("sglang.srt.models.qwen3_tts_graph_driver",
     "python/sglang/srt/models/qwen3_tts_graph_driver.py"),
):
    spec = importlib.util.spec_from_file_location(name, HERE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

from sglang.srt.models.qwen3_tts_graph_driver import decode_mask  # noqa: E402
from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    InProcessQwen3Tts, InProcessTtsConfig,
)
from transformers import DynamicCache, StaticCache  # noqa: E402

cfg = InProcessTtsConfig(
    model_dir=pathlib.Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base"),
    device="cpu", dtype="float32",
)
tts = InProcessQwen3Tts(cfg)
tts.load()
talker = getattr(tts._model, "model", tts._model).talker
trunk = talker.model
H = trunk.config.hidden_size
CACHE = 64  # smaller pad on CPU; the question is semantics, not pad width

g = torch.Generator().manual_seed(488)
prefill = torch.randn(1, 6, H, generator=g)
steps = [torch.randn(1, 1, H, generator=g) for _ in range(3)]

with torch.inference_mode():
    dyn = DynamicCache()
    trunk(inputs_embeds=prefill, past_key_values=dyn, use_cache=True)
    A = [trunk(inputs_embeds=t, past_key_values=dyn,
               use_cache=True).last_hidden_state.clone() for t in steps]

    st = StaticCache(config=trunk.config, max_cache_len=CACHE)
    trunk(inputs_embeds=prefill, past_key_values=st, use_cache=True,
          cache_position=torch.arange(6))
    B = []
    for i, t in enumerate(steps):
        pos = 6 + i
        cp = torch.tensor([pos])
        B.append(trunk(inputs_embeds=t,
                       attention_mask=decode_mask(pos + 1, CACHE, 1),
                       position_ids=cp.view(1, 1, 1).expand(3, 1, 1),
                       cache_position=cp, past_key_values=st,
                       use_cache=True).last_hidden_state.clone())

scale = max(float(a.abs().max()) for a in A)
print(f"\nFP32/CPU  scale = {scale:.4f}")
for i in range(len(A)):
    d = float((A[i] - B[i]).abs().max())
    print(f"  step {i}: |A-B| = {d:.3e}   relative = {d / scale:.3e}")
print("\nbf16 eps = 7.81e-03 ; fp32 eps = 1.19e-07")
