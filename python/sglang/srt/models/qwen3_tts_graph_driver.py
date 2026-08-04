# SPDX-License-Identifier: Apache-2.0
"""#488 graphs cut, slice 2: CUDA graphs over the talker's token loop.

WHAT SLICE 1 LEFT ON THE TABLE
------------------------------
`/spinning/gpu-battery-results/2026-08-04_488_precursor/RESULTS.md` decomposed a
142.78 ms talker frame into **21.85 ms of kernel and 120.93 ms of nothing** --
84.7 % of the frame is covered by no kernel at all. Slice 1
(`qwen3_tts_fast_predictor.py`) removed the 602 host syncs that made the region
uncapturable. This module removes the launch gap itself, which is the whole of
the remaining 84.7 %.

The ladder under test, from that same measurement:
``RTF 1.713 today -> ~0.26 with graphs -> 0.05-0.10 with fused/TRT engines``.
This module is the middle rung and the control arm for the TRT lane.

THE SEAM, AND WHY NO TRANSFORMERS MONKEYPATCH WAS NEEDED
--------------------------------------------------------
A captured region may not sync to the host and may not allocate. The reference
decode path does both, but only through arguments it lets the caller supply --
read at source rather than assumed (transformers 5.12.1, qwen-tts 0.1.1):

* ``modeling_qwen3_tts.py:1492`` (trunk) and ``:1085`` (predictor) build
  ``cache_position`` with a host-side ``torch.arange`` **only when the caller
  passes None**, and its start comes from ``past_key_values.get_seq_length()``,
  which a ``StaticLayer`` returns as a *tensor* (``cache_utils.py:463``) -- so
  leaving it to the reference would either sync or break. We pass it.
* ``masking_utils.py:840`` -- ``_preprocess_mask_arguments`` returns an
  ``attention_mask`` that is already 4-D **as-is**. So a caller-owned static
  mask buffer reaches sdpa untouched, through both the trunk's single-mask path
  (``modeling_qwen3_tts.py:1510``) and the predictor's dict path (``:1095``).
  No patching of transformers, and no reimplementation of the layer stack.
* the one genuine sync on the decode path, ``cache_position[0] == 0`` at
  ``modeling_qwen3_tts.py:1696``, lives in the OUTER
  ``Qwen3TTSTalkerForConditionalGeneration.forward``. We capture ``talker.model``
  and ``predictor.model`` directly, which is the same thing the precursor timed,
  so it is never on the captured path.

WHY ``StaticCache`` AND NOT A HAND-ROLLED ONE
---------------------------------------------
``StaticLayer.update`` (``cache_utils.py:441-456``) is already written for this:
it keeps ``cumulative_length`` as a **device tensor**, advances it with an
in-place ``add_``, and writes with ``index_copy_``. Every one of those is
capturable, and the in-place advance means a replayed graph **moves to the next
slot by itself** -- the write position is state inside the graph, not an
argument to it. :func:`refuse_unless_graph_safe` pins that property instead of
trusting the version, because a future transformers that spells
``cumulative_length`` as a Python int would silently bake slot 0 into every
replay and overwrite the same cache entry forever.

The consequence to keep in mind: ``update`` returns the **whole padded cache**
and ``get_mask_sizes`` reports ``kv_length = max_cache_len``
(``cache_utils.py:457-461``). Attention therefore sees unwritten slots on every
step, and the mask is the only thing keeping them out. That is why the mask is
built here rather than delegated.

WHAT IS CONSTANT AND WHAT MOVES
-------------------------------
The two graphs differ in a way worth stating, because it decides how much
per-frame host work survives:

* **predictor** -- step ``g`` always runs at the same cache length, every frame,
  forever (prefill writes 2 slots, decode step ``g`` writes slot ``1 + g``). So
  its masks are *constant per step* and are filled once at capture. A predictor
  frame costs 15 replays and **no mask maintenance at all**.
* **trunk** -- one step per frame, at a cache length that grows with the
  utterance. Its mask and position buffers are updated in place before each
  replay: three tiny device ops (:meth:`GraphedTrunkStep.advance`), against 28
  layers of work.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, List, Optional, Sequence

import torch

from sglang.srt.models.qwen3_tts_fast_predictor import (
    FastCodePredictor,
    PredictorOutput,
    apply_warpers,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GraphCaptureRefusal",
    "CapturedStep",
    "GraphedPredictorFrame",
    "GraphedTrunkStep",
    "decode_mask",
    "predictor_cache_lengths",
    "refuse_unless_graph_safe",
]


class GraphCaptureRefusal(RuntimeError):
    """Raised instead of capturing something that would replay wrongly.

    Every refusal in this module is for a condition that is **silent** at
    runtime: a stale cache slot, a baked write position, a replay order that
    does not match the capture order. None of them crash on their own, and all
    of them corrupt audio. Refusing by name is the only way they get noticed.
    """


# ---------------------------------------------------------------------------
# pure geometry -- no CUDA, so it is all hermetically testable
# ---------------------------------------------------------------------------


def predictor_cache_lengths(num_code_groups: int) -> List[int]:
    """Cache occupancy **after** each step of one predictor frame.

    The reference prefills the predictor with two hidden states
    (``modeling_qwen3_tts.py:1671`` passes ``cat((past_hidden, last_id_hidden))``)
    and then takes ``num_code_groups - 2`` single-token decode steps. So the
    occupancy sequence is ``2, 3, 4, ...`` and the last entry is the scratch
    size the cache must be allocated for.

    Returned as data because the alternative -- an ``arange`` written inline at
    the capture site -- is exactly the off-by-one that produces a frame whose
    last residual group attends to an unwritten slot. That does not crash; it
    degrades timbre.
    """
    if num_code_groups < 3:
        raise ValueError(
            f"num_code_groups={num_code_groups} leaves no residual groups to "
            f"predict; this checkpoint geometry is not a talker frame."
        )
    steps = num_code_groups - 1
    return [2] + [2 + step for step in range(1, steps)]


def decode_mask(
    valid_len: int,
    cache_len: int,
    query_len: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """A 4-D boolean sdpa mask: ``True`` where a kv slot may be attended.

    ``valid_len`` counts slots **already written including this step's own
    query**, so the final query position attends to itself -- the causal
    diagonal. Slots at or beyond ``valid_len`` hold whatever the previous
    utterance left there, and admitting them is the failure this function
    exists to prevent: softmax over stale keys produces plausible, wrong audio
    rather than an error.

    Shape ``(1, 1, query_len, cache_len)``, which
    ``_preprocess_mask_arguments`` (``masking_utils.py:840``) passes through
    untouched.
    """
    if valid_len > cache_len:
        raise ValueError(
            f"valid_len={valid_len} exceeds the static cache ({cache_len} "
            f"slots); the utterance outgrew the buffer it was captured for."
        )
    if query_len > valid_len:
        raise ValueError(
            f"query_len={query_len} exceeds valid_len={valid_len}; a query "
            f"cannot precede the slots it writes."
        )
    slots = torch.arange(cache_len, device=device)
    # Query q of this step occupies absolute slot (valid_len - query_len + q),
    # and attends to every slot up to and including its own.
    last_slot = valid_len - query_len + torch.arange(query_len, device=device)
    mask = slots.view(1, -1) <= last_slot.view(-1, 1)
    return mask.view(1, 1, query_len, cache_len)


def refuse_unless_graph_safe(cache) -> None:
    """Refuse a cache whose write position is not device-resident state.

    The whole design rests on ``StaticLayer.cumulative_length`` being a tensor
    advanced in place (``cache_utils.py:445``): that is what makes a *single*
    captured graph walk forward through the cache on each replay. If it were a
    Python int, capture would bake slot 0 into the graph and every replay would
    overwrite the same entry -- a silent single-slot cache, not a crash.
    """
    layers = getattr(cache, "layers", None)
    if not layers:
        raise GraphCaptureRefusal(
            f"{type(cache).__name__} exposes no .layers; this module captures "
            f"only transformers StaticCache-family caches."
        )
    for index, layer in enumerate(layers):
        position = getattr(layer, "cumulative_length", None)
        if not isinstance(position, torch.Tensor):
            raise GraphCaptureRefusal(
                f"layer {index} of {type(cache).__name__} keeps its write "
                f"position as {type(position).__name__}, not a device tensor. "
                f"Capturing against it would bake one slot into every replay "
                f"and silently overwrite it. Refusing to capture."
            )
        # A sliding-window static layer ALSO carries a host-side
        # `cumulative_length_int` (cache_utils.py:492) and derives its mask
        # sizes from it (`:576-578`). That int is advanced by eager Python, so
        # a captured graph freezes the window where it stood at capture and
        # the attention span silently stops moving. Both checkpoints in play
        # have `sliding_window: None`, so this is a guard against a future
        # config, not a live condition -- which is exactly when a silent
        # failure would be hardest to attribute.
        if hasattr(layer, "cumulative_length_int"):
            raise GraphCaptureRefusal(
                f"layer {index} is a {type(layer).__name__}, whose mask sizes "
                f"come from a host-side cumulative_length_int. A captured "
                f"graph would freeze its sliding window at the capture "
                f"position. Refusing to capture."
            )


def reset_cache_positions(cache) -> None:
    """Rewind a static cache to slot 0 without touching its contents.

    The stale keys stay where they are on purpose -- masking them out is
    :func:`decode_mask`'s job, and zeroing 17 slots per frame would be device
    work that buys nothing. Used between predictor frames, which is the only
    place a rewind is correct: the trunk's cache spans the utterance.
    """
    for layer in getattr(cache, "layers", []) or []:
        position = getattr(layer, "cumulative_length", None)
        if isinstance(position, torch.Tensor):
            position.zero_()


# ---------------------------------------------------------------------------
# capture plumbing
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CapturedStep:
    """One captured graph plus the static buffers it reads and writes.

    ``inputs`` and ``outputs`` are the *only* tensors that cross the graph
    boundary. A caller copies into the former, replays, and reads the latter;
    anything else it holds a reference to is an alias into the graph's private
    pool and is invalid between replays.
    """

    name: str
    graph: "torch.cuda.CUDAGraph"
    inputs: dict
    outputs: dict

    def replay(self) -> None:
        self.graph.replay()


def _capture(
    name: str,
    body: Callable[[], dict],
    inputs: dict,
    pool=None,
    warmup: int = 3,
    setup: Optional[Callable[[], None]] = None,
) -> CapturedStep:
    """Warm up on a side stream, then capture ``body`` into a graph.

    The side-stream warmup is not optional and not cosmetic: cuBLAS and sdpa
    allocate their workspaces on first call for a given shape, and an
    allocation during capture either fails outright or -- worse -- captures a
    pointer into memory the caching allocator will hand to somebody else.

    ``setup`` runs before every warmup call and once before the capture, and
    crucially runs **outside** the captured region. That distinction is the
    difference between a graph that walks forward through its cache and one
    that rewrites the same slot forever: every warmup call advances the static
    cache's write pointer, so the pointer has to be put back before capture --
    but if that reset were captured too, each replay would reset as well.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            if setup is not None:
                setup()
            body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    if setup is not None:
        setup()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        outputs = body()
    return CapturedStep(name=name, graph=graph, inputs=inputs, outputs=outputs)


class _OrderedReplay:
    """Enforces that a shared-pool graph set replays in its capture order.

    PyTorch permits several graphs to share one memory pool, which is what
    keeps fifteen predictor graphs at ~60 MiB instead of fifteen private pools.
    The documented price is that they must be **replayed in the order they were
    captured** -- their private allocations overlap, so a reordered replay reads
    another graph's live intermediates. It does not fault; it returns numbers.

    A frame is a fixed fifteen-step sequence today, so the invariant holds by
    construction. It is asserted anyway, because "by construction" is a
    property of this file at this moment and the cost of checking is one
    integer comparison per replay.
    """

    def __init__(self, steps: Sequence[CapturedStep]) -> None:
        self._steps = list(steps)
        self._expected = 0

    def rewind(self) -> None:
        self._expected = 0

    def replay(self, index: int) -> None:
        if index != self._expected:
            raise GraphCaptureRefusal(
                f"shared-pool graphs must replay in capture order: expected "
                f"step {self._expected} ({self._steps[self._expected].name}), "
                f"got step {index} ({self._steps[index].name}). Replaying out "
                f"of order reads another graph's live intermediates and "
                f"returns plausible wrong values rather than failing."
            )
        self._steps[index].replay()
        self._expected = (self._expected + 1) % len(self._steps)


# ---------------------------------------------------------------------------
# the predictor frame -- 15 graphs, replacing the whole generate() envelope
# ---------------------------------------------------------------------------


class GraphedPredictorFrame(FastCodePredictor):
    """The 15-step residual loop as 15 captured graphs.

    Subclasses slice 1 rather than replacing it: the step schedule, the sampler
    and the ``install()`` seam are the same, and the eager loop stays reachable
    as the reference arm of the identity gate.

    Fifteen graphs and not one, per ``ANALYSE_488 §7.3``: the steps are
    shape-identical but each uses its own ``lm_head[g]`` and
    ``codec_embedding[g-1]``, and a graph cannot switch which ``nn.Linear``
    runs. Stacking the heads into one indexed table would need a 120 MiB
    re-layout of tensors that are already resident, against ~60 MiB of shared
    graph pool for capturing fifteen.

    Sampling stays inside the graph. ``torch.multinomial`` with a device
    generator is capturable, but its *seeding* is not something a replay
    re-draws, so a pre-drawn uniform buffer is used instead -- which is also
    what makes the identity gate possible: the same uniforms driven through the
    eager and the graphed path must produce the identical token sequence.
    """

    def __init__(
        self,
        predictor,
        num_code_groups: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(predictor, num_code_groups)
        parameter = next(predictor.parameters())
        self.device = device or parameter.device
        self.dtype = dtype or parameter.dtype
        self.cache_lengths = predictor_cache_lengths(num_code_groups)
        self.scratch_slots = self.cache_lengths[-1]
        self.hidden_size = predictor.model.config.hidden_size
        self.vocab_size = predictor.lm_head[0].out_features
        self.cache = None
        self.steps: List[CapturedStep] = []
        self._order: Optional[_OrderedReplay] = None
        #: Sampling knobs are baked at capture time, not per call. A graph
        #: cannot branch, so `do_sample` and the warper thresholds are part of
        #: what was captured; changing them requires a re-capture, and
        #: :meth:`generate` refuses a mismatch rather than ignoring it.
        self.captured_sampling: Optional[tuple] = None

    # -- capture ------------------------------------------------------------

    def capture(
        self,
        do_sample: bool = True,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> "GraphedPredictorFrame":
        from transformers import StaticCache  # noqa: PLC0415

        predictor = self.predictor
        model = predictor.model
        config = model.config

        self.cache = StaticCache(config=config, max_cache_len=self.scratch_slots)
        refuse_unless_graph_safe(
            _lazy_init(self.cache, model, self.device, self.dtype)
        )
        embeddings = model.get_input_embeddings()
        projection = getattr(predictor, "small_to_mtp_projection", None)

        pool = None
        self.steps = []
        for position, (_, embedding_index, head_index) in enumerate(self.schedule):
            query_len = 2 if embedding_index is None else 1
            valid_len = self.cache_lengths[position]
            cache_start = valid_len - query_len

            # Per-step static buffers. The mask is CONSTANT for this step --
            # step g runs at the same cache length in every frame of every
            # utterance -- so it is filled once, here, and never maintained.
            buffers = {
                "hidden": torch.zeros(
                    1, query_len, self.hidden_size,
                    device=self.device, dtype=self.dtype,
                ),
                "token_in": torch.zeros(1, 1, device=self.device, dtype=torch.long),
                "uniform": torch.zeros(1, 1, device=self.device, dtype=torch.float32),
                "cache_position": torch.arange(
                    cache_start, valid_len, device=self.device, dtype=torch.long
                ),
                "mask": decode_mask(
                    valid_len, self.scratch_slots, query_len, device=str(self.device)
                ),
            }
            buffers["position_ids"] = buffers["cache_position"].unsqueeze(0)

            def body(
                buffers=buffers,
                embedding_index=embedding_index,
                head_index=head_index,
                cache_start=cache_start,
            ) -> dict:
                # PINNED INSIDE THE GRAPH, deliberately, and the opposite of
                # what GraphedTrunkStep does. A predictor step owns exactly one
                # slot for the life of the process -- step g always writes slot
                # 1+g, in every frame of every utterance -- so capturing the
                # pin makes each step idempotent in the write position and
                # removes the between-frame rewind from the host path
                # entirely. The trunk cannot do this: its slot advances.
                self._pin_write_slot(cache_start)
                if embedding_index is None:
                    hidden = buffers["hidden"]
                else:
                    hidden = embeddings[embedding_index](buffers["token_in"])
                if projection is not None:
                    hidden = projection(hidden)
                outputs = model(
                    inputs_embeds=hidden,
                    attention_mask=buffers["mask"],
                    position_ids=buffers["position_ids"],
                    cache_position=buffers["cache_position"],
                    past_key_values=self.cache,
                    use_cache=True,
                )
                last = outputs.last_hidden_state[:, -1:, :]
                logits = predictor.lm_head[head_index](last)[:, -1, :]
                token = self._sample(
                    logits, buffers["uniform"], do_sample, temperature, top_k, top_p
                )
                return {"token": token, "logits": logits}

            step = _capture(f"predictor_step_{position}", body, buffers, pool=pool)
            pool = step.graph.pool()
            self.steps.append(step)

        self._order = _OrderedReplay(self.steps)
        self.captured_sampling = (do_sample, temperature, top_k, top_p)
        logger.info(
            "#488: captured %d predictor step graphs over a %d-slot static "
            "scratch cache (shared pool)",
            len(self.steps), self.scratch_slots,
        )
        return self

    def _pin_write_slot(self, slot: int) -> None:
        """Set every layer's write position to ``slot``, in place.

        In place and not rebound: the captured graph holds these exact
        addresses. The *contents* of the earlier slots do not matter to which
        slot this step writes, and the mask decides what is readable, so
        nudging the counter is the whole of what capture needs.
        """
        for layer in self.cache.layers:
            position = getattr(layer, "cumulative_length", None)
            if isinstance(position, torch.Tensor):
                position.fill_(slot)

    def _sample(
        self, logits, uniform, do_sample, temperature, top_k, top_p
    ) -> torch.Tensor:
        """Sample without leaving the device and without drawing entropy.

        ``torch.multinomial`` is replaced by an inverse-CDF draw against a
        caller-supplied uniform, for two reasons that both matter here: a
        replayed graph would otherwise consume the same captured random state
        every frame, and the identity gate needs the eager and graphed arms to
        be driven by the *same* draws.
        """
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)
        warped = apply_warpers(logits, temperature, top_k, top_p)
        # float32 before the cumsum, for two independent reasons: a bf16 cumsum
        # over 2048 entries loses the tail of the distribution to rounding, and
        # torch.searchsorted wants a well-defined ordered dtype. The eager arm
        # of the identity gate does the same, so the gate compares the same
        # arithmetic driven two ways.
        probabilities = warped.float().softmax(dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        # searchsorted is the device-side equivalent of multinomial's own
        # inverse-CDF step, and unlike multinomial it takes the draw as data.
        return torch.searchsorted(cumulative, uniform.clamp(max=1.0)).clamp(
            max=self.vocab_size - 1
        )

    # -- replay -------------------------------------------------------------

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        uniforms: Optional[torch.Tensor] = None,
        **ignored,
    ) -> PredictorOutput:
        """One frame: 15 graph replays, no host sync, no allocation.

        Same signature and same return shape as slice 1 and as the reference
        ``code_predictor.generate``, so this stays a drop-in.
        """
        if not self.steps:
            raise GraphCaptureRefusal(
                "generate() called before capture(); a graph driver with no "
                "graphs would silently fall through to the eager path and "
                "report the eager path's timings as the graphed arm's."
            )
        requested = (
            bool(do_sample) if do_sample is not None else self.captured_sampling[0],
            temperature, top_k, top_p,
        )
        if do_sample is not None and requested != self.captured_sampling:
            raise GraphCaptureRefusal(
                f"sampling parameters {requested} differ from the captured "
                f"{self.captured_sampling}. A graph cannot re-branch: the "
                f"warpers are baked in. Re-capture instead of silently "
                f"sampling with the wrong thresholds."
            )
        wanted = max_new_tokens or (self.num_code_groups - 1)
        if wanted != len(self.steps):
            raise ValueError(
                f"max_new_tokens={wanted} does not match the {len(self.steps)} "
                f"captured steps; refusing rather than emitting a partial "
                f"frame, which would desync the codebooks."
            )

        # No cache rewind here on purpose: each step graph pins its own write
        # slot inside the captured region (see capture()), so a frame cannot
        # inherit a position from the frame before it. The five fill_ launches
        # a rewind would cost per frame are exactly the kind of host-side
        # residue this slice exists to delete.
        self._order.rewind()
        codes: List[torch.Tensor] = []
        for position, step in enumerate(self.steps):
            if position == 0:
                step.inputs["hidden"].copy_(inputs_embeds)
            else:
                step.inputs["token_in"].copy_(codes[-1])
            if uniforms is not None:
                step.inputs["uniform"].copy_(uniforms[:, position : position + 1])
            self._order.replay(position)
            # The output buffer is overwritten by the NEXT replay of this same
            # step, i.e. on the next frame -- not by any replay within this
            # frame. Cloning keeps the frame's codes valid past the loop.
            codes.append(step.outputs["token"].clone())
        return PredictorOutput(sequences=torch.cat(codes, dim=-1))


# ---------------------------------------------------------------------------
# the trunk step -- one graph per frame, over a cache that spans the utterance
# ---------------------------------------------------------------------------


class GraphedTrunkStep:
    """The 28-layer trunk decode step as one captured graph.

    Unlike the predictor, this one graph serves every frame of an utterance, at
    a cache length that grows by one per frame. Three things therefore move
    between replays, and all three are in-place device writes on static
    buffers, so no host sync and no re-capture:

    * ``cache_position`` and ``position_ids`` -- incremented by one;
    * the mask -- one more slot admitted.

    The static cache's own write pointer needs no maintenance: ``StaticLayer``
    advances it inside the captured region (``cache_utils.py:445``).
    """

    def __init__(self, trunk, max_positions: int = 1024) -> None:
        parameter = next(trunk.parameters())
        self.trunk = trunk
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.hidden_size = trunk.config.hidden_size
        self.max_positions = max_positions
        self.cache = None
        self.step: Optional[CapturedStep] = None
        self._valid = 0

    def capture(self, prefill_len: int = 1) -> "GraphedTrunkStep":
        """Capture the decode step as it will run at ``prefill_len + 1``.

        The capture-time cache length is irrelevant to correctness -- the mask
        and positions are read from buffers at replay -- but it must be a
        length the graph will actually see, so the shapes match.
        """
        from transformers import StaticCache  # noqa: PLC0415

        self.cache = StaticCache(
            config=self.trunk.config, max_cache_len=self.max_positions
        )
        self._valid = prefill_len + 1

        buffers = {
            "hidden": torch.zeros(
                1, 1, self.hidden_size, device=self.device, dtype=self.dtype
            ),
            "cache_position": torch.tensor(
                [prefill_len], device=self.device, dtype=torch.long
            ),
            "mask": torch.zeros(
                1, 1, 1, self.max_positions, device=self.device, dtype=torch.bool
            ),
        }
        buffers["mask"][..., : self._valid] = True
        # M-RoPE: the trunk expects (3, batch, seq) and splits off a text axis
        # (modeling_qwen3_tts.py:1500). One buffer, three views of the same
        # position, advanced together.
        buffers["position_ids"] = buffers["cache_position"].view(1, 1, 1).expand(3, 1, 1)

        def body() -> dict:
            outputs = self.trunk(
                inputs_embeds=buffers["hidden"],
                attention_mask=buffers["mask"],
                position_ids=buffers["position_ids"],
                cache_position=buffers["cache_position"],
                past_key_values=self.cache,
                use_cache=True,
            )
            return {"hidden": outputs.last_hidden_state}

        def setup() -> None:
            # OUTSIDE the captured region, and that is the whole point. Each
            # warmup call advances the cache, so the write slot has to be put
            # back before capture -- but a captured reset would fire on every
            # replay and the trunk would rewrite one slot for the entire
            # utterance. The predictor does the opposite, for the opposite
            # reason: see GraphedPredictorFrame.capture.
            for layer in self.cache.layers:
                position = getattr(layer, "cumulative_length", None)
                if isinstance(position, torch.Tensor):
                    position.fill_(prefill_len)

        refuse_unless_graph_safe(
            _lazy_init(self.cache, self.trunk, self.device, self.dtype)
        )
        self.step = _capture("trunk_step", body, buffers, setup=setup)
        logger.info(
            "#488: captured the 28-layer trunk decode step over a %d-position "
            "static cache",
            self.max_positions,
        )
        return self

    def advance(self) -> None:
        """Move to the next frame: three tiny device ops, no host sync."""
        if self._valid >= self.max_positions:
            raise GraphCaptureRefusal(
                f"the utterance reached {self._valid} frames, the whole of the "
                f"{self.max_positions}-position static cache. Continuing would "
                f"wrap onto slot 0 and attend to this utterance's own opening "
                f"frames as if they were the newest."
            )
        self.step.inputs["cache_position"].add_(1)
        self.step.inputs["mask"][..., self._valid] = True
        self._valid += 1

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.step is None:
            raise GraphCaptureRefusal("forward() called before capture()")
        self.step.inputs["hidden"].copy_(hidden)
        self.step.replay()
        return self.step.outputs["hidden"]


def _lazy_init(cache, module, device, dtype):
    """Force a StaticCache to materialise its layers so they can be checked.

    ``StaticLayer`` allocates on its first ``update`` (``cache_utils.py:437``),
    so ``cumulative_length`` is only a tensor after that point. Checking the
    cache before it has ever been written would pass vacuously -- the refusal
    has to be able to see the real thing.
    """
    layers = getattr(cache, "layers", None)
    if layers:
        for layer in layers:
            if not getattr(layer, "is_initialized", False):
                heads = getattr(module.config, "num_key_value_heads", 1)
                head_dim = getattr(
                    module.config,
                    "head_dim",
                    module.config.hidden_size // module.config.num_attention_heads,
                )
                probe = torch.zeros(1, heads, 0, head_dim, device=device, dtype=dtype)
                layer.lazy_initialization(probe, probe)
    return cache
