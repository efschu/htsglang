"""What a control buys and what it costs, in two half-sentences.

Every setting, slider, template and mode switch in the dashboard carries a
hover explanation of the same shape: what turning it up gets you, and what it
takes away. That is the only shape a tuning knob has -- a knob with no cost
would already be the default.

WHY THIS IS ONE FILE
--------------------
The text belongs to the *control*, not to the place it happens to be drawn.
The same flag appears in the flag surface, in a preset summary, in the
placement panel and in a launch command; four copies of a sentence is four
chances to describe a flag the way it behaved two releases ago. One entry
here feeds all of them, and it sits next to nothing else, so changing a
flag's behaviour and changing its stated cost is one edit in one place.

This registry deliberately does NOT live in ``flags.py``. That module is the
machine-checked catalog -- types, allowed values, mutual exclusions, hard
rejects -- and it is edited whenever the server's argument surface moves.
This is prose for a reader. Keeping them apart means a wording fix never
touches validation logic, and vice versa.

NUMBERS
-------
A number is quoted only when this rig actually measured it. The registry
stores a *pointer* to where a figure would come from (``measured_from``), not
a figure; :func:`describe` fills it in when a finding is on disk and says
"not measured on this rig" when there is none. Nothing here is estimated,
rounded up from a different rig, or carried over from a benchmark blog. A
missing number is stated as missing.

ADDING A CONTROL
----------------
Add one :class:`Tradeoff` to :data:`TRADEOFFS`, keyed by:

* a flag id (``"rank_kv_ratio"``) -- applies to the whole control;
* a flag id and value (``"rank_kv_ratio=speed"``) -- overrides the above when
  that value is selected, for enums whose options pull in opposite
  directions;
* a UI control key (``"view_mode.expert"``, ``"tune.maxkv"``) -- for controls
  that are not flags.

Nothing else needs touching: the web UI merges this into the flag catalog it
already serves, and renders whatever it finds. A control with no entry simply
falls back to the catalog's own ``hover`` text, so an unlisted flag is never
worse off than before -- it just does not yet state its cost.

A worked example, for the MoE offload wave ordering that is on its way. The
switch chooses between ordering the offload waves by token and by expert, so
it is one control with two values that pull opposite ways -- the same shape
as ``rank_kv_ratio``. Three entries, no other file::

    "moe_offload_wave_order": Tradeoff(
        gain="...what choosing the ordering at all buys...",
        cost="...what it costs...",
    ),
    "moe_offload_wave_order=tokenmajor": Tradeoff(gain=..., cost=...),
    "moe_offload_wave_order=expertmajor": Tradeoff(gain=..., cost=...),

If the delivering change measured the difference, point ``measured_from`` at
the study that produced the figure and add that study to
:data:`MEASUREMENT_SOURCES`; do not paste the number into the sentence, or it
will still be quoted long after the hardware has changed.

There is one guard rail: ``test_fork_flags_are_covered`` fails when a
non-env flag this fork invented has no entry here. A knob nobody else
documents is exactly the knob that has to say what it costs, so adding the
flag and adding its trade-off is deliberately one commit.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional

__all__ = [
    "Tradeoff",
    "TRADEOFFS",
    "describe",
    "tooltip_map",
    "missing_coverage",
]


@dataclasses.dataclass(frozen=True)
class Tradeoff:
    """What a control buys, and what it costs.

    ``gain`` and ``cost`` are one short sentence each, written for someone
    deciding whether to touch the control -- not a restatement of what the
    flag is called.

    ``measured_from`` names the study whose result would put a number on
    this, when one has been run on this rig. It is a pointer, never a value:
    the number is looked up at render time or reported as absent.
    """

    gain: str
    cost: str
    measured_from: str = ""


#: Studies that can supply a figure, and the human name used when one is
#: missing. Extend alongside a new measurement harness, not before it.
MEASUREMENT_SOURCES = {
    "mlp_crossover": "the MLP-split crossover study",
    "power_profile": "the per-card power calibration",
    "bench_suite": "the benchmark suite",
    "moe_offload_wave_order": (
        "the #254 offload wave-order study (Qwen3.6-35B-A3B-AWQ, single 5090, "
        "resident expert fraction 0.25)"
    ),
}


TRADEOFFS: Dict[str, Tradeoff] = {
    # ---- view modes -------------------------------------------------------
    "view_mode.simple": Tradeoff(
        gain="Shows each card's VRAM as one number and one budget slider.",
        cost="Hides the per-segment breakdown and the split ratios; a "
        "configuration built here uses the defaults for everything not shown.",
    ),
    "view_mode.expert": Tradeoff(
        gain="Exposes every dimension this fork has, each recomputing the "
        "whole plan live.",
        cost="Many of them interact; a bad combination is caught by the "
        "planner rather than by the control itself.",
    ),
    # ---- tuning objective templates ---------------------------------------
    "tune.both": Tradeoff(
        gain="Targets prefill and concurrent throughput together.",
        cost="Neither one is pushed as far as its dedicated setting would.",
    ),
    "tune.maxkv": Tradeoff(
        gain="Gives the KV cache everything left over, for the longest "
        "context these cards can hold.",
        cost="Concentrates KV on the largest card, so deep-context attention "
        "waits on it.",
        measured_from="mlp_crossover",
    ),
    "tune.dec": Tradeoff(
        gain="Shifts KV ownership toward the faster cards, which is where "
        "decode time is actually spent.",
        cost="Cuts usable context, because the small cards stop contributing "
        "their share of the KV pool.",
        measured_from="mlp_crossover",
    ),
    "tune.enc": Tradeoff(
        gain="Concentrates the MLP families on the compute-strong ranks, "
        "which is the measured prefill lever.",
        cost="Costs decode throughput and some context; only worth it when "
        "prompts dominate output.",
        measured_from="mlp_crossover",
    ),
    # ---- simple-view controls ---------------------------------------------
    "card_budget": Tradeoff(
        gain="Caps how much of this card the server may take, leaving the "
        "rest for anything else on the machine.",
        cost="Lowering it shrinks the KV cache first, so maximum context "
        "falls before anything else does.",
    ),
    "context_length": Tradeoff(
        gain="Raises the longest prompt plus output a single request may use.",
        cost="Every concurrent slot reserves that much, so raising it lowers "
        "how many requests can run at once.",
    ),
    "max_running_requests": Tradeoff(
        gain="More requests served in parallel.",
        cost="Each slot takes its own KV and recurrent state, so maximum "
        "context falls as this rises.",
    ),
    # ---- heterogeneous rank mapping ---------------------------------------
    "rank_gpu_id": Tradeoff(
        gain="Puts each rank on a chosen physical card, including several "
        "ranks on one card.",
        cost="Ranks sharing a card share its bandwidth and its power budget; "
        "the slowest rank still sets the pace of every collective.",
    ),
    "rank_gpu_memory_mib": Tradeoff(
        gain="Gives each rank an absolute VRAM budget, so mixed card sizes "
        "stop being described by one fraction that fits none of them.",
        cost="The budget is taken at face value: too tight fails at boot, too "
        "loose fails later under fragmentation.",
    ),
    "rank_auto_reserve_mib": Tradeoff(
        gain="Holds headroom back from the derived budgets for runtime "
        "allocations and graph capture.",
        cost="Every MiB reserved is a MiB the KV cache does not get.",
    ),
    "rank_tp_ratio": Tradeoff(
        gain="Splits every sharded dimension by card capability instead of "
        "evenly, so a big card carries more than a small one.",
        cost="The split is in whole KV-head units, so it lands on the grid "
        "rather than exactly on the ratio asked for.",
    ),
    "rank_tp_ratio=auto": Tradeoff(
        gain="Derives the weights from the cards' memory budgets: the "
        "largest total context these cards can hold.",
        cost="Loads the weakest card in proportion to its memory rather than "
        "its speed, so it becomes the pace-setter.",
    ),
    "rank_tp_ratio=auto-performance": Tradeoff(
        gain="Starts from the memory split, then moves the MLP families "
        "toward the compute-strong ranks for throughput.",
        cost="Trades away some context to do it; the context floor is set by "
        "--rank-perf-loose-ctx-percent.",
        measured_from="mlp_crossover",
    ),
    "rank_perf_tune": Tradeoff(
        gain="Says which axis auto-performance should optimise.",
        cost="Each target gives up the others; there is no setting that wins "
        "on all three.",
    ),
    "rank_perf_loose_ctx_percent": Tradeoff(
        gain="Lets the optimiser trade this much predicted context away for "
        "throughput.",
        cost="Exactly that much context, and it is spent whether or not the "
        "throughput gain turns out to be real.",
    ),
    "rank_mlp_ratio": Tradeoff(
        gain="Moves dense-FFN weight off the small cards, which frees the "
        "memory that sets the shared KV pool.",
        cost="Adds prefill work to whichever rank takes it on.",
        measured_from="mlp_crossover",
    ),
    "rank_moe_ratio": Tradeoff(
        gain="Distributes expert weight by card capability rather than "
        "evenly.",
        cost="Uneven expert placement makes per-token routing cost depend on "
        "which experts a token picks.",
    ),
    "rank_vocab_ratio": Tradeoff(
        gain="Balances the lm_head read time across cards of different "
        "bandwidth, so no rank waits on the vocab matvec.",
        cost="Rounds to 64-row units and moves embedding memory onto the "
        "faster cards.",
    ),
    "rank_kv_ratio": Tradeoff(
        gain="Decides which rank owns which context tokens, separately from "
        "the weight split.",
        cost="Ownership decides where deep-context attention runs, so it "
        "trades capacity against latency on the same axis.",
    ),
    "rank_kv_ratio=capacity": Tradeoff(
        gain="Weights KV ownership by memory, for the largest total context "
        "these cards can hold.",
        cost="Sends deep-context attention work to whichever card holds the "
        "most, which is not always the fastest one.",
    ),
    "rank_kv_ratio=speed": Tradeoff(
        gain="Takes KV ownership off the slowest card, so it stops setting "
        "the pace on long contexts.",
        cost="Costs KV capacity on the small cards, and with it maximum "
        "context.",
        measured_from="mlp_crossover",
    ),
    "rank_kv_ratio=coupled": Tradeoff(
        gain="Keeps KV ownership tied to the weight split, which is the "
        "long-standing behaviour.",
        cost="Gives up the capacity and latency this axis could buy "
        "separately.",
    ),
    # ---- memory / cache ----------------------------------------------------
    "kv_cache_dtype": Tradeoff(
        gain="fp8 KV roughly halves what a context token costs, so the same "
        "cards hold far more of it.",
        cost="Lossy: outputs are no longer bit-identical to the bf16 run.",
    ),
    "enable_kv_session_offload": Tradeoff(
        gain="Moves an idle session's KV to host RAM so an active one can "
        "have the device memory.",
        cost="Bringing it back costs a transfer over PCIe before that "
        "session's next token.",
    ),
    "kv_session_offload_budget_episode_seconds": Tradeoff(
        gain="Bounds how long one uninterrupted spill episode may run.",
        cost="Once it elapses the session is handed to the cache, so "
        "continuing is a prefix hit rather than an immediate resume.",
    ),
    "kv_session_offload_budget_session_tokens": Tradeoff(
        gain="Caps how much of a single session may be spilled at once.",
        cost="A session larger than the cap is only partly protected; the "
        "remainder is evicted normally.",
    ),
    "kv_session_offload_host_ram_gib": Tradeoff(
        gain="More host RAM for spilled sessions means more of them survive "
        "device pressure.",
        cost="That RAM is unavailable to everything else on the machine, and "
        "the host has no swap to fall back on.",
    ),
    "cpu_offload_gb": Tradeoff(
        gain="Lets a model that does not fit in VRAM run at all, by keeping "
        "part of the weights in host RAM.",
        cost="Every fetch of an offloaded weight crosses PCIe, which on this "
        "rig has no peer-to-peer path.",
    ),
    "hicache_ratio": Tradeoff(
        gain="A larger host-side prefix cache turns more repeat prompts into "
        "cache hits instead of prefill.",
        cost="Takes host RAM, and a miss still pays the lookup.",
    ),
    # ---- speculative decoding ---------------------------------------------
    "speculative_adaptive": Tradeoff(
        gain="Follows the accepted draft length instead of holding one fixed "
        "number of speculative tokens.",
        cost="The controller needs a few rounds to settle after the workload "
        "changes shape.",
    ),
    "speculative_adaptive_config": Tradeoff(
        gain="Overrides how eagerly the adaptive controller raises or drops "
        "the draft length.",
        cost="Hand-set thresholds stop following the workload; the defaults "
        "are what the measurements were taken against.",
    ),
    "speculative_adaptive_graph_memory": Tradeoff(
        gain="Decides where the draft model's CUDA graphs live; offloading "
        "them returns VRAM to the KV cache.",
        cost="An offloaded graph is fetched over PCIe before the draft pass, "
        "which shows up in every verify round.",
    ),
    "speculative_num_steps": Tradeoff(
        gain="More drafted tokens per round means fewer verify rounds when "
        "acceptance is high.",
        cost="Every rejected token is drafting work thrown away, so a low "
        "acceptance rate makes this a loss.",
    ),
    # ---- offload -----------------------------------------------------------
    "SGLANG_MOE_RESIDENT_EXPERT_FRACTION": Tradeoff(
        gain="Keeping more experts resident on the device avoids fetching "
        "them per token.",
        cost="Resident experts occupy VRAM that would otherwise be KV cache.",
    ),
    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": Tradeoff(
        gain="Releases captured CUDA-graph memory when the server idles.",
        cost="The first request after an idle period pays the capture again.",
    ),
    "SGLANG_MOE_OFFLOAD_WAVE_ORDER": Tradeoff(
        gain="Chooses how a chunk's offload waves are ordered, which decides "
        "how often a spilled expert has to cross PCIe.",
        cost="The two orderings trade PCIe traffic against a transient "
        "per-layer buffer; they produce byte-identical output either way.",
        measured_from="moe_offload_wave_order",
    ),
    "SGLANG_MOE_OFFLOAD_WAVE_ORDER=token": Tradeoff(
        gain="Today's behaviour, and it needs no extra memory.",
        cost="Every spilled expert is streamed again for each wave, and with "
        "a wave covering only a token or two the PCIe traffic grows with the "
        "chunk instead of with the number of experts.",
        measured_from="moe_offload_wave_order",
    ),
    "SGLANG_MOE_OFFLOAD_WAVE_ORDER=expert": Tradeoff(
        gain="Each spilled expert crosses PCIe once per chunk, so offloaded "
        "prefill stops scaling with the prompt length -- the longer the "
        "prompt, the larger the margin.",
        cost="One transient tokens x top_k x hidden buffer per layer, on the "
        "order of tens of MiB at a 2048-token chunk.",
        measured_from="moe_offload_wave_order",
    ),
    # ---- weightless KV lane ------------------------------------------------
    "weightless_kv_fastlane": Tradeoff(
        gain="Lets the weight-holding rank answer without waiting on the "
        "KV-only ranks each step.",
        cost="Only helps while the lane is in lock-step; it is a narrower "
        "configuration than plain TP.",
    ),
    "weightless_kv_host_spill_tokens": Tradeoff(
        gain="More context held per card by pushing the tail of the KV to "
        "host RAM.",
        cost="Tokens past the device cap are read back over PCIe when "
        "attention reaches them.",
    ),
    "weightless_kv_head_rank": Tradeoff(
        gain="Names the rank that holds the weights, so the fast card can be "
        "chosen deliberately rather than by rank order.",
        cost="That rank carries the whole model; the others contribute only "
        "KV and attention.",
    ),
    "weightless_kv_chunked_block_size": Tradeoff(
        gain="Larger chunks mean fewer, bigger transfers between the head "
        "rank and the KV-only ranks.",
        cost="Each chunk must be buffered before it moves, so the buffers "
        "grow with it and latency comes in coarser steps.",
    ),
    "weightless_kv_spill_device_cap": Tradeoff(
        gain="Sets how much KV stays resident per card before the rest goes "
        "to host RAM.",
        cost="A lower cap frees VRAM but sends more of the context across "
        "PCIe when attention reaches it.",
    ),
    # ---- persistence -------------------------------------------------------
    "hibernate_dir": Tradeoff(
        gain="Writes the loaded state to disk, so a restart skips weight "
        "loading and comes back in seconds.",
        cost="Needs disk room for the whole resident state, and the image is "
        "only valid for the exact configuration that wrote it.",
    ),
}


def describe(
    key: str,
    *,
    measurements: Optional[Dict[str, str]] = None,
    fallback: str = "",
) -> str:
    """One hover string for ``key``: what it gives, then what it costs.

    ``measurements`` maps a :data:`MEASUREMENT_SOURCES` key to a short figure
    measured **on this rig** (for example ``{"mlp_crossover": "+6% prefill"}``).
    A trade-off that points at a source with no measurement says so rather
    than going quiet about it -- the reader needs to know the difference
    between "no effect" and "nobody measured".

    ``fallback`` is returned when the key has no entry, so a control that has
    not been written up yet keeps whatever text it already had.
    """
    t = TRADEOFFS.get(key)
    if t is None:
        return fallback
    out = f"{t.gain} Costs: {t.cost}"
    if t.measured_from:
        figure = (measurements or {}).get(t.measured_from)
        if figure:
            out += f" Measured here: {figure}."
        else:
            source = MEASUREMENT_SOURCES.get(t.measured_from, t.measured_from)
            out += f" Not measured on this rig yet (run {source})."
    return out


def tooltip_map(measurements: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Every trade-off, rendered, keyed exactly as :data:`TRADEOFFS` is.

    The web UI fetches this once and looks controls up by key, so a control
    never carries its own copy of the sentence.
    """
    return {k: describe(k, measurements=measurements) for k in TRADEOFFS}


def missing_coverage(flag_ids) -> list:
    """Flag ids with no trade-off written yet.

    Used by the test suite to keep the fork's own flags covered: a knob this
    fork invented should say what it costs, because nobody else's
    documentation will.
    """
    return sorted(f for f in flag_ids if f not in TRADEOFFS)
