"""fnFL2 H35: ``ple.wait`` -- the PLE layer's wait on its n-gram gather, as a
family of the per-rank decode clock.

WHY. On Qwen3.8-Flash-Next the PLE n-gram table (95.4 GiB bf16) stays on disk:
the ``checkpoint`` backend maps the safetensors shards read-only and the
gather kernel reads the rows through HMM, inside the captured verify graph,
on the rank that owns the dense side (Form A: TP0 only). A row whose page is
not yet mapped in THIS process faults inside the kernel, served one at a
time from ZFS (21.09. probes: ~1.6k cold pages/s). The layer overlaps the
gather with the preceding decoder layer on a side stream
(``Qwen4ExpPLELayer.start_prefetch``) and joins it with ``wait_stream`` --
so every cold page lands in TP0's ``compute`` column of ``Decode rank
batch``, indistinguishable from GEMMs.

WHAT THE LOGS ALREADY SAY (H35, x144 against fnFA23/fnFA28, TP0 medians):
the verify compute has a floor of 9.9-10.0 ms on both forms, and a bump of
+3..+10 ms that follows the NOVELTY of the generated text in this process,
not the form: x144 mid (first) 17.2 -> mid2 (same answer again) 10.0,
needle1 (new answer) 19.65 -> needle2 (mid's answer again, after a flip)
10.05; fnFA28 thinking@128 (first) 12-14 -> thinking@10k (same reasoning)
10.0; code@10k on fnFA23 10.0 because code@128 had generated the module just
before, x144 code@10k 16.4 because it is the first code answer of the boot.
This family makes that attribution a measurement instead of an inference.

THE SPAN. ``span`` inside a capture lays an event-record pair into the graph
(#1241b); here it brackets the MAIN stream's ``wait_stream`` on the gather's
side stream, so ``ple.wait`` is exactly the time the verify forward stalled
on the PLE rows (0 when the gather finished under layer 1's compute). On the
non-prefetch path it brackets the whole lookup (hash + gather). Under the
decode round's phase it prints as ``spec_verify:ple.wait``; the round's
``compute`` then EXCLUDES it and ``wait`` includes it -- the switch changes
the split, never ``gpu-ms``.

COST. Switch off (default): one env read per PLE layer call -- in a replayed
graph none at all. On: two event-record nodes per captured graph. Switch
``SGLANG_DEBUG_DECODE_PLE_WAIT`` (debug instrumentation, per rank env --
set it in the D group's env, the launcher env is not the rank env).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

from sglang.srt.environ import envs

__all__ = ["PLE_WAIT_FAMILY", "ple_wait_scope", "wait_for_ple_prefetch"]

#: The clock family, before the phase prefix (``spec_verify:``) is applied.
PLE_WAIT_FAMILY = "ple.wait"


def ple_wait_scope(clock: Optional[Any] = None):
    """A ``ple.wait`` span on the decode/prefill clock, or a no-op.

    No-op unless the switch is on AND the clock is armed (a round or a
    prefill forward is bracketed, or a graph is being captured under a
    scope) -- the same guard every dispatch site of the clock uses, so an
    unarmed call lays nothing and records nothing."""
    if not envs.SGLANG_DEBUG_DECODE_PLE_WAIT.get():
        return nullcontext()
    if clock is None:
        from sglang.srt.utils.collective_clock import collective_clock

        clock = collective_clock()
    if not clock.armed:
        return nullcontext()
    return clock.span(PLE_WAIT_FAMILY)


def wait_for_ple_prefetch(
    prefetch_stream: Any,
    *,
    clock: Optional[Any] = None,
    current_stream: Optional[Any] = None,
) -> None:
    """The PLE layer's join on its prefetch gather, timed as ``ple.wait``.

    ``current_stream`` is the stream that waits (default: torch's current
    stream); injectable so the join is testable without a device."""
    if current_stream is None:
        import torch

        current_stream = torch.cuda.current_stream()
    with ple_wait_scope(clock):
        current_stream.wait_stream(prefetch_stream)
