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
"""The prefill lane of ``--disaggregation-topology=colocated-congruent`` (#107).

One process group carries two roles: every card keeps its decode rank AND
serves prefill, computing with the SAME resident weight shard. The lane is
the in-process replacement for a separate prefill server; its scheduling
contract is DECODE PRIORITY — the property PD exists for — enforced as a
cadence gate instead of the stock prefill-first policy.

INVARIANT — only the WEIGHTS are shared between the lanes:

* weights: the prefill forward reads the decode ranks' resident shards.
  The sharing is by construction (same process, same model object) and is
  VERIFIED, not assumed: ``bind_model`` records the storage identity of
  every parameter and ``assert_congruent`` re-checks it against the model
  that is actually about to run, naming the first diverging parameter and
  both storage pointers. A later change that gives the lane its own runner
  (and thereby a second weight copy) fails this check instead of silently
  doubling the VRAM footprint.
* KV: a prefill request writes into its OWN slots of the shared pool; slot
  ownership is per-request disjoint. The in-process "transfer" is that the
  decode lane later reads those slots — there is nothing to copy.
* activations / forward state: strictly per-forward, never shared between
  the lanes. Prefill and decode batches are NEVER mixed
  (``enable_mixed_chunk`` is rejected at validation).

Execution model (stage 2, serial): a lane tick runs INSTEAD of a device
decode iteration, gated by ``--disaggregation-prefill-lane-interval`` (the
same S1 pattern as the kv-session-offload spill tick). CONCURRENT dispatch
on a second stream is deliberately NOT done here: the model's TP/MoE
all-reduces share their communicators between the lanes, and concurrent
collectives on one communicator are the documented hang risk of the S4b
decouple work (see ``Scheduler._dispatch_concurrent_spill``). Overlapping
the lanes needs the comm-B split for the model collectives first; that is
the named follow-up slice, not a silent property of this one.

Rank uniformity: the gate's inputs are replicated scheduler state (batch
emptiness, the deterministic tick counter), so every rank takes the same
prefill/decode branch each iteration — the same argument the spill tick
makes. A rank-local input here would split the ranks across branches with
mismatched collective counts (the known hang family).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CongruentPrefillLane:
    """Decode-priority pacing + weight-sharing verification for the lane."""

    def __init__(self, tick_interval: int):
        if tick_interval < 1:
            raise ValueError(
                f"--disaggregation-prefill-lane-interval must be >= 1, got "
                f"{tick_interval} (1 alternates prefill and decode; larger "
                "values protect the decode cadence more strongly)."
            )
        self.tick_interval = tick_interval
        self._device_iters_since_prefill = tick_interval  # first prefill is free
        self._weight_ptrs: Optional[Dict[str, int]] = None
        self._verified = False

    # ------------------------------------------------------------------
    # Weight-sharing invariant
    # ------------------------------------------------------------------

    def bind_model(self, model: Any) -> None:
        """Record the storage identity of the decode ranks' weights."""
        self._weight_ptrs = {
            name: p.data_ptr() for name, p in model.named_parameters()
        }
        self._verified = False

    def assert_congruent(self, model: Any) -> None:
        """Verify the model about to prefill IS the decode weight set.

        Cheap after the first call (identity is stable within a run); the
        check exists so a future lane-owned runner cannot silently
        reintroduce a second weight copy.
        """
        if self._verified:
            return
        if self._weight_ptrs is None:
            raise AssertionError(
                "congruent lane: assert_congruent called before bind_model."
            )
        for name, p in model.named_parameters():
            recorded = self._weight_ptrs.get(name)
            if recorded is None:
                raise AssertionError(
                    f"congruent lane: parameter {name!r} of the prefill "
                    "model has no counterpart in the bound decode weights — "
                    "the lane is not computing with the decode shards."
                )
            ptr = p.data_ptr()
            if ptr != recorded:
                raise AssertionError(
                    "congruent lane: weight sharing broken — parameter "
                    f"{name!r} has storage 0x{ptr:x} in the prefill model "
                    f"but 0x{recorded:x} in the bound decode weights. The "
                    "lane MUST read the decode ranks' resident shards (one "
                    "weight copy per card); a second copy voids the VRAM "
                    "plan this topology was admitted under."
                )
        self._verified = True

    # ------------------------------------------------------------------
    # Decode-priority pacing (rank-uniform: replicated inputs only)
    # ------------------------------------------------------------------

    def allow_prefill(self, device_has_decode_work: bool) -> bool:
        """Gate for building a prefill batch this iteration.

        Without decode work there is nothing to protect — prefill runs
        freely (and the counter does not advance: idle iterations are not
        decode progress). With decode work, prefill is admitted only every
        ``tick_interval``-th device iteration.
        """
        if not device_has_decode_work:
            return True
        self._device_iters_since_prefill += 1
        return self._device_iters_since_prefill > self.tick_interval

    def note_prefill_ran(self, model: Any = None) -> None:
        """Called when a prefill batch was actually selected."""
        self._device_iters_since_prefill = 0
        if model is not None:
            self.assert_congruent(model)
