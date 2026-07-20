# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Offline config planner (design #97, stage S1): capacity / feasibility /
split — never throughput.

A pure library over the fork's EXISTING parse-time sizing math
(``PerfCostModel`` / ``resolve_cp_token_ratios`` / the ServerArgs reserve
formulas): given a model checkpoint (HF ``config.json`` dir or a single
``.gguf`` file) and a hardware description (live NVML, nvidia-smi, or a
hand-declared inventory), it answers

  1. does it fit?                       (feasibility, fail-loud with the card)
  2. with which split?                  (rank->GPU map, budgets, tp/DCP ratio)
  3. what is the honest advantage over
     stock even-TP?                     (feasibility + capacity-%, NOT tok/s)

HONESTY DISCIPLINE (structural, design §3.4): no type in this package has an
estimated-throughput / estimated-tok/s field. Measured per-card scores appear
only when a cached hardware profile already exists
(``get_cached_hardware_profile``, cache-only — a probe is never triggered);
otherwise those fields are absent, not guessed. CPU-only: nothing here loads
weights, boots a server, or touches a GPU.

Single entry point::

    from sglang.srt.planner import plan
    result = plan(model_path, hardware=..., rank_gpu_id=[0, 1, 2], ...)

or the CLI: ``python -m sglang.planner --model <path> ...``.
"""

from sglang.srt.uneven_perf import PlanInputs  # noqa: F401  (re-export)


def plan(*args, **kwargs):
    """Lazy wrapper around :func:`sglang.srt.planner.feasibility.plan` (the
    submodule import pulls in the sglang import chain, which is heavy)."""
    from sglang.srt.planner.feasibility import plan as _plan

    return _plan(*args, **kwargs)
