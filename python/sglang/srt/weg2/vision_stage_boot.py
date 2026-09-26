# SPDX-License-Identifier: Apache-2.0
"""Task #58 slice 10 -- ARMING the transient vision stage, as part of the boot.

Slice 9 built :class:`~sglang.srt.weg2.vision_stage_service.VisionStageService`
and the seam in the processor that calls it.  Everything about it was correct
and none of it ran, for one reason: nothing ever called ``install``.  A boot
with ``--weg2-vision transient`` therefore did exactly what a boot without it
did -- the tokenizer built ``mm_items``, the seam was a no-op, the items left
with ``feature`` and no rows, and the failure surfaced three hops later inside
a rank as ``_require_visual``.  This module is the missing call, and it is a
FIXED PART OF THE BOOT, not a hand step.

WHAT ARMING MEANS, exactly
--------------------------
``arm_transient_vision`` runs once, in the group's tokenizer/multimodal
processor process (``TokenizerManager.init_tokenizer_and_processor``), and
either

* returns a service that was installed -- every hook in the design's table
  §6 present and named -- or
* refuses **by name** with a W111 reason, records it via
  ``vision_stage_service.install_refusal``, and lets the boot continue serving
  TEXT.  The first image request then refuses by name too (W112), quoting the
  arming reason.

The one thing it must never do is neither: a ``transient`` boot that is
silently unarmed is the defect this file closes.

WHY THE CUDA CONTEXT IS MEASURED IN A CHILD PROCESS
---------------------------------------------------
``TowerSpec.ctx_bytes`` is the CUDA context of the process that runs the
stage.  Its docstring already says it must be MEASURED, "because a context
size guessed from a driver version is exactly the kind of number that later
eats a prefill".  Measuring it in THIS process would strand a context: CUDA
never gives one back, so the probe would permanently hold ~0.3-0.5 GiB on
whichever card it touched, and that card is not necessarily the one the
placement later picks -- the process would then pay the post twice, once
where it is invisible to the planner and once where it is not.

So the probe is a short-lived CHILD: NVML free before, ``torch.cuda.init()``,
NVML free after, print the delta, exit.  The context dies with it.  The number
is a true post for whichever card the plan picks, and the real context appears
only when ``load_tower`` binds ``cuda:<plan.card>``.

The child is pinned to the card by **UUID**, not by index
(``CUDA_VISIBLE_DEVICES=GPU-...``), so ``cuda:0`` inside it is that NVML card
and nothing else.  Torch's enumeration order and NVML's can diverge; every
number in this file that names a card names an NVML index, and
:func:`torch_index_for_nvml_card` is the ONE place the two are joined, by UUID,
refusing rather than assuming when it cannot.

WHAT IS BOOKED, AND WHAT IS NOT
-------------------------------
The law is "Transienten explizit buchen, keine Reserven".  Four posts go into
the :class:`TowerSpec`:

======================  =========================================
``weight_bytes``        read from the checkpoint header (exact)
``ctx_bytes``           measured by the child probe (exact, 1 MiB grain)
``activation_bytes``    MODELLED, for ONE named geometry (below)
``embedding_bytes``     arithmetic from that same geometry (exact)
======================  =========================================

The activation is the only modelled one and it depends on the IMAGE, which
nobody knows at arming.  It is therefore booked for a NAMED geometry --
:data:`BOOKED_IMAGE_HW`, the design's acceptance image, 1024x1024 -- and the
arm log line prints that geometry beside the number.  A larger image is not
repriced: that limit is named here, in :func:`arm_transient_vision`'s log, and
in the design's risk 3 (an image-area ceiling belongs in the front).  Nothing
here rounds up "to be safe": a floor above the booked posts would be a
reserve, and reserves are forbidden.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from sglang.srt.name_compat import canonical_env
from sglang.srt.planner.vision_stage import (
    GIB,
    MIB,
    TowerSpec,
    VisionEncoderConfig,
    VisionStageTowerUnreadable,
    tower_from_span,
)
from sglang.srt.planner.vision_stage_load import (
    MEASURED_LOADER_GBPS,
    VisionStageLoadRefused,
    find_tower_shard,
    strip_checkpoint_prefix,
    tower_extent,
    tower_state_dict,
)
from sglang.srt.weg2 import vision_stage_service as _vss

logger = logging.getLogger(__name__)

#: LAUNCHER OUTPUT, same discipline as ``SGLANG_WEG2_GROUP`` and the host-ring
#: family (launcher R19): published only for the group that runs the stage and
#: POPPED otherwise, so a value inherited from an operator's shell can never
#: arm a stage nobody asked for.  Defined in ``vision_stage_service`` -- the
#: light module -- and re-exported here; see its comment for why.
VISION_ENV = _vss.VISION_ENV
#: Which group's processor process runs the stage.  P: it is the P layout that
#: has no tower and needs the rows before its prefill.
VISION_GROUP_ENV = _vss.VISION_GROUP_ENV
VISION_GROUP = _vss.VISION_GROUP
VISION_TRANSIENT = _vss.VISION_TRANSIENT

#: MEASURED host-to-device rates per NVML card, Memory ``RANG-LINK-ZUORDNUNG``:
#: rank 0 and 2 sit on x8 (14.4 / 13.3 GB/s), rank 1 on x4 (6.5 GB/s).  A card
#: missing here is DROPPED from the placement by ``census_from_nvml`` rather
#: than given an assumed rate -- see its docstring.
MEASURED_H2D_GBPS: Dict[int, float] = {0: 14.4, 1: 6.5, 2: 13.3}

#: The geometry the transient posts are booked for.  The design's acceptance
#: image (§6): 1024x1024, one still, one part.  NOT a cap -- the front has no
#: image-area ceiling (design risk 3) -- but the number the log prints, so an
#: operator reading a placement knows what it was priced against.
BOOKED_IMAGE_HW: Tuple[int, int] = (1024, 1024)

#: bytes per element of the tower's activations.  BF16 throughout: 333/333
#: tower tensors in this checkpoint are bf16 (design §1.4).
ACTIVATION_DTYPE_BYTES = 2

#: The read rate the planner is given.  **1.08, not 3.85.**  The loader in
#: this tree is ``pread_safetensors_file`` and it is BUFFERED
#: (``vision_stage_load.LOADER_IS_BUFFERED``); the O_DIRECT path that measured
#: 3.85 GB/s is slice 7 and is NOT built.  Passing the direct rate here would
#: plan 239 ms and get 853 ms.
ARM_READ_GBPS = MEASURED_LOADER_GBPS

#: Timeout for the CUDA-context probe child, seconds.  Generous: the child
#: imports torch and initialises a context, which is seconds on a cold page
#: cache.  Exceeding it is an arming REFUSAL, never a default value.
CONTEXT_PROBE_TIMEOUT_S = 180.0


class VisionStageArmRefused(RuntimeError):
    """A named precondition for arming the transient stage was not met.

    Every path out of :func:`arm_transient_vision` that is not a working
    service raises (and records) one of these.  The message always names the
    precondition, because the W111 line in the boot log is the only thing an
    operator has between "I passed --weg2-vision transient" and "the first
    image request failed".
    """


# ---------------------------------------------------------------------------
# what mode this process is in
# ---------------------------------------------------------------------------


def vision_mode(env: Optional[Dict[str, str]] = None) -> str:
    """``"transient"`` when this process should arm, else ``""``.

    Reads the launcher's two variables and nothing else.  A process that is
    not the P group's, or a boot that is not transient, gets ``""`` -- and
    :func:`arm_transient_vision` then returns without touching any state, so
    the text path is what it was before this module existed.
    """
    e = os.environ if env is None else env
    if (e.get(VISION_ENV, "") or "").strip() != VISION_TRANSIENT:
        return ""
    group = (e.get(VISION_GROUP_ENV, "") or "").strip()
    if group and group != VISION_GROUP:
        return ""
    return VISION_TRANSIENT


# ---------------------------------------------------------------------------
# the encoder config, from the checkpoint rather than from the defaults
# ---------------------------------------------------------------------------


def encoder_config_from_hf(vision_config: Any) -> VisionEncoderConfig:
    """Build the planner's encoder config out of the model's ``vision_config``.

    :class:`VisionEncoderConfig`'s defaults ARE this checkpoint's numbers, and
    that is exactly why they must not be relied on here: a default that happens
    to be right is indistinguishable from one that silently went stale when the
    checkpoint changed.  Every field is read; a missing one refuses.
    """
    if vision_config is None:
        raise VisionStageArmRefused(
            "the model config carries no `vision_config`. A transient stage "
            "sized from VisionEncoderConfig's defaults would be sized from "
            "ANOTHER checkpoint's numbers -- refusing rather than guessing."
        )

    def _need(name: str) -> Any:
        val = getattr(vision_config, name, None)
        if val is None and isinstance(vision_config, dict):
            val = vision_config.get(name)
        if val is None:
            raise VisionStageArmRefused(
                f"`vision_config.{name}` is absent from this checkpoint; the "
                "stage cannot be sized without it"
            )
        return val

    ds = getattr(vision_config, "deepstack_visual_indexes", None)
    if ds is None and isinstance(vision_config, dict):
        ds = vision_config.get("deepstack_visual_indexes")
    return VisionEncoderConfig(
        depth=int(_need("depth")),
        hidden_size=int(_need("hidden_size")),
        intermediate_size=int(_need("intermediate_size")),
        num_heads=int(_need("num_heads")),
        out_hidden_size=int(_need("out_hidden_size")),
        patch_size=int(_need("patch_size")),
        temporal_patch_size=int(_need("temporal_patch_size")),
        spatial_merge_size=int(_need("spatial_merge_size")),
        deepstack_visual_indexes=tuple(int(x) for x in (ds or ())),
    )


def encoder_activation_bytes(
    cfg: VisionEncoderConfig, patch_rows: int, *, dtype_bytes: int = ACTIVATION_DTYPE_BYTES
) -> int:
    """Peak live activation of ONE encoder block, in bytes.

    ONE block, not the sum over ``depth``: the blocks are sequential and
    autograd is off, so block ``k``'s intermediates are freed before block
    ``k+1`` allocates.  Summing them would book 27x the truth, and booking
    27x is how a placement refuses a card that would have fit.

    The terms, per row: the qkv projection (``3h``), the residual carried
    across the block (``h``), the attention output (``h``) and the MLP's
    intermediate (``i``).  Flash attention is ASSUMED -- a backend that
    materialises the ``rows x rows`` score matrix would add
    ``depth``-independent ``rows^2 * heads * dtype_bytes`` on top, which at
    4096 rows is another 512 MiB.  That assumption is the one soft spot in
    this number and it is named rather than buried; the design says the same
    in §3.
    """
    if patch_rows <= 0:
        raise ValueError(f"patch_rows={patch_rows} must be > 0")
    h = int(cfg.hidden_size)
    i = int(cfg.intermediate_size)
    per_row = (3 * h) + h + h + i
    return int(patch_rows) * per_row * int(dtype_bytes)


def embedding_bytes_for(
    cfg: VisionEncoderConfig, patch_rows: int, *, dtype_bytes: int = ACTIVATION_DTYPE_BYTES
) -> int:
    """Bytes of the rows the stage hands on: exact, not modelled."""
    out_rows = int(patch_rows) // cfg.merge_factor
    return int(out_rows) * int(cfg.embed_width) * int(dtype_bytes)


# ---------------------------------------------------------------------------
# the CUDA context, measured
# ---------------------------------------------------------------------------

#: The child probe.  Printed as JSON on ONE line so the parent parses a
#: document rather than scraping a number out of torch's own chatter on
#: stdout/stderr.
CONTEXT_PROBE_SOURCE = textwrap.dedent(
    """
    import json, os, sys

    uuid = sys.argv[1]
    from sglang.srt.registry import nvml as _nvml

    def free_bytes():
        for dev, mem in _nvml.memory_snapshot():
            if str(dev.uuid) == uuid:
                return int(mem.free_mib) * (1 << 20)
        raise SystemExit("probe: no NVML card with uuid " + uuid)

    before = free_bytes()
    import torch
    # CUDA_VISIBLE_DEVICES is this one card by UUID, so cuda:0 IS that card.
    torch.cuda.init()
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    after = free_bytes()
    sys.stdout.write("VISION_CTX_PROBE " + json.dumps({
        "uuid": uuid,
        "free_before_bytes": before,
        "free_after_bytes": after,
        "ctx_bytes": before - after,
    }) + "\\n")
    """
)

_PROBE_MARKER = "VISION_CTX_PROBE "


@dataclass(frozen=True)
class ContextMeasurement:
    """What the probe measured, with both readings kept.

    Both readings, not just the delta, for the same reason every number in
    this tree carries its denominator: a delta of 400 MiB out of 2 GiB free
    and one out of 30 GiB free are different facts about the same box.
    """

    card: int
    uuid: str
    free_before_bytes: int
    free_after_bytes: int
    ctx_bytes: int

    def log_line(self) -> str:
        return (
            f"[vision-ctx] card={self.card} uuid={self.uuid} "
            f"free_before={self.free_before_bytes / GIB:.3f}GiB "
            f"free_after={self.free_after_bytes / GIB:.3f}GiB "
            f"ctx_bytes={self.ctx_bytes / MIB:.0f}MiB "
            "(measured in a child process, NVML before/after cuda init; the "
            "child exits so no context is stranded on this card)"
        )


def context_probe_command(
    uuid: str, *, python: str = "", tree: str = ""
) -> Tuple[Sequence[str], Dict[str, str]]:
    """argv and environment for the child probe.  PURE -- pinned by a test.

    ``CUDA_VISIBLE_DEVICES`` is the UUID, never an index: the whole point of
    the probe is a number attached to a specific physical card, and an index
    that torch and NVML order differently would attach it to the wrong one.
    """
    if not uuid:
        raise VisionStageArmRefused(
            "cannot probe a CUDA context without the card's NVML UUID; an "
            "index would be torch's order, not NVML's"
        )
    env = canonical_env(dict(os.environ))  # rename 1b: the pop below sees one spelling
    env["CUDA_VISIBLE_DEVICES"] = uuid
    # The probe must not inherit a rank's memory-saver preload or a partial
    # distributed environment; it initialises one context and exits.
    for key in ("LD_PRELOAD", "SGLANG_WEG2_TMS_PRELOAD_SO"):
        env.pop(key, None)
    if tree:
        env["PYTHONPATH"] = (
            os.path.join(tree, "python") + os.pathsep + env.get("PYTHONPATH", "")
        ).rstrip(os.pathsep)
    return [python or sys.executable, "-c", CONTEXT_PROBE_SOURCE, uuid], env


def parse_context_probe(stdout: str, *, card: int) -> ContextMeasurement:
    """Read the probe's one JSON line, or refuse with what it actually said."""
    line = ""
    for raw in (stdout or "").splitlines():
        if raw.startswith(_PROBE_MARKER):
            line = raw[len(_PROBE_MARKER) :]
    if not line:
        tail = (stdout or "").strip()[-600:]
        raise VisionStageArmRefused(
            "the CUDA-context probe printed no measurement line "
            f"({_PROBE_MARKER!r}); its output ended with: {tail!r}"
        )
    doc = json.loads(line)
    ctx = int(doc["ctx_bytes"])
    if ctx <= 0:
        raise VisionStageArmRefused(
            f"the CUDA-context probe measured ctx_bytes={ctx} on card{card} "
            f"(free {doc['free_before_bytes']} -> {doc['free_after_bytes']}). "
            "A context that does not show up in NVML means the probe measured "
            "something other than a context -- another tenant freed memory "
            "under it, or CUDA never initialised. Refusing rather than "
            "planning against a zero-cost context."
        )
    return ContextMeasurement(
        card=int(card),
        uuid=str(doc["uuid"]),
        free_before_bytes=int(doc["free_before_bytes"]),
        free_after_bytes=int(doc["free_after_bytes"]),
        ctx_bytes=ctx,
    )


def measure_context_bytes(
    card: int,
    uuid: str,
    *,
    runner: Optional[Callable[[Sequence[str], Dict[str, str]], str]] = None,
    tree: str = "",
) -> ContextMeasurement:
    """Measure this process's CUDA context ONCE, in a child that then dies.

    ``runner`` is injected so the whole arming path is drivable at a desk with
    no GPU; the default spawns the real child.
    """
    argv, env = context_probe_command(uuid, tree=tree)
    if runner is not None:
        return parse_context_probe(runner(argv, env), card=card)
    try:
        proc = subprocess.run(
            list(argv),
            env=env,
            capture_output=True,
            text=True,
            timeout=CONTEXT_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise VisionStageArmRefused(
            f"the CUDA-context probe for card{card} did not finish within "
            f"{CONTEXT_PROBE_TIMEOUT_S:.0f} s"
        ) from exc
    except OSError as exc:
        raise VisionStageArmRefused(
            f"the CUDA-context probe for card{card} could not be started: {exc}"
        ) from exc
    if proc.returncode != 0:
        raise VisionStageArmRefused(
            f"the CUDA-context probe for card{card} exited {proc.returncode}: "
            f"{(proc.stderr or '').strip()[-600:]!r}"
        )
    return parse_context_probe(proc.stdout, card=card)


# ---------------------------------------------------------------------------
# NVML identity and the torch index that goes with it
# ---------------------------------------------------------------------------


def probe_card(snapshot: Sequence[Tuple[Any, Any]], h2d_gbps: Dict[int, float]) -> Tuple[int, str]:
    """(NVML index, UUID) of the card the context probe runs on.

    Any card with a measured link rate will do -- a CUDA context is the same
    size on each -- so this takes the LOWEST such index, deterministically,
    rather than the emptiest: picking by free memory would make the arming
    result depend on what else happened to be running.
    """
    for dev, _mem in snapshot:
        idx = int(getattr(dev, "index", -1))
        if idx in h2d_gbps:
            return idx, str(getattr(dev, "uuid", ""))
    raise VisionStageArmRefused(
        "NVML reports no card with a measured h2d rate "
        f"(have rates for {sorted(h2d_gbps)}); the context cannot be probed "
        "and no placement would be valid anyway"
    )


def torch_index_for_nvml_card(card: int, snapshot: Sequence[Tuple[Any, Any]]) -> int:
    """Torch's device ordinal for an NVML index, joined BY UUID.

    The one place in this path where the two enumerations meet.  Torch's order
    and NVML's can diverge (``CUDA_VISIBLE_DEVICES`` alone guarantees it), and
    a stage that loads a 0.858 GiB tower onto the wrong card puts it exactly
    where the planner proved there was no room.  Refuses rather than falling
    back to ``card`` as an ordinal.
    """
    import torch

    want = ""
    for dev, _mem in snapshot:
        if int(getattr(dev, "index", -1)) == int(card):
            want = str(getattr(dev, "uuid", ""))
            break
    if not want:
        raise VisionStageLoadRefused(
            f"NVML card{card} is not in the snapshot this placement came from"
        )
    for i in range(torch.cuda.device_count()):
        uuid = str(getattr(torch.cuda.get_device_properties(i), "uuid", ""))
        if uuid and (uuid in want or want in uuid):
            return i
    raise VisionStageLoadRefused(
        f"no visible torch device carries NVML card{card}'s UUID {want!r} "
        f"(torch sees {torch.cuda.device_count()} device(s)). The stage will "
        "not load a tower onto a card chosen by ordinal coincidence."
    )


# ---------------------------------------------------------------------------
# the three CUDA hooks
# ---------------------------------------------------------------------------


@dataclass
class TowerHooks:
    """``load_tower`` / ``encode`` / ``release_tower``, sharing one state.

    Built as a small object rather than three closures because the three ARE
    one state machine -- the handle ``load_tower`` returns is what ``encode``
    runs and ``release_tower`` frees -- and a closure trio would hide that
    the module is kept alive only between them.
    """

    model_dir: str
    hf_config: Any
    snapshot: Callable[[], Sequence[Tuple[Any, Any]]]
    #: injected so the desk can drive the whole thing; the default is the real
    #: Qwen3-VL tower module.
    build_module: Optional[Callable[[Any, int], Any]] = None
    _shard: str = ""

    def shard(self) -> str:
        if not self._shard:
            self._shard = find_tower_shard(self.model_dir)
        return self._shard

    # -- load ---------------------------------------------------------------
    def load_tower(self, card: int) -> Any:
        """Build the tower on ``cuda:<torch index of NVML card>`` and fill it.

        The state dict is read with ``vision_only_selector`` -- the tower and
        nothing else -- and its names are stripped of ``model.visual.`` so they
        match a STANDALONE module's parameters.  ``strip_checkpoint_prefix``
        refuses an unmapped name instead of passing it through, because an
        unmapped name lands in ``missing_keys`` and a tower with an empty block
        returns plausible wrong embeddings rather than failing.
        """
        import torch

        idx = torch_index_for_nvml_card(card, list(self.snapshot()))
        module = (self.build_module or _build_qwen3vl_tower)(self.hf_config, idx)
        sd = strip_checkpoint_prefix(tower_state_dict(self.shard()))
        dev = torch.device("cuda", idx)
        sd = {
            k: (v.to(dev) if hasattr(v, "to") else v) for k, v in sd.items()
        }
        missing, unexpected = module.load_state_dict(sd, strict=False)
        if missing:
            raise VisionStageLoadRefused(
                f"the tower module has {len(missing)} unfilled parameter(s) "
                f"after loading (first: {list(missing)[:3]}). An unfilled block "
                "encodes garbage and returns plausible wrong text; refusing."
            )
        if unexpected:
            logger.warning(
                "vision stage: %d checkpoint tensor(s) the tower module does "
                "not have (first: %s)",
                len(unexpected),
                list(unexpected)[:3],
            )
        module.eval()
        return module

    # -- encode -------------------------------------------------------------
    def encode(self, handle: Any, items: Sequence[Any]) -> Sequence[Any]:
        """Run the tower and bring the rows BACK TO THE HOST.

        The same two lines ``qwen3_vl.get_image_feature`` runs (:1461-1476),
        with one addition that is not decoration: ``.to("cpu")``.
        ``attach_precomputed_embeddings`` REFUSES a tensor still on the stage's
        card, and that refusal is what makes delivery free -- a host tensor
        travels with the request through the normal path and
        ``_get_precomputed_embedding`` moves it to whatever device asks.
        Leaving the rows on the card would strand 10 MiB there past the
        teardown and break the invariant this whole stage is built around.
        """
        import torch

        with torch.inference_mode():
            pixel_values = torch.cat([it.feature for it in items], dim=0).type(
                handle.dtype
            )
            grid_thw = torch.concat([it.image_grid_thw for it in items], dim=0)
            dev = next(handle.parameters()).device
            out = handle(pixel_values.to(dev), grid_thw=grid_thw.to(dev))
        rows = out.to("cpu")
        # One tensor out, one per item in: split it back the way the items came
        # in, so `attach_precomputed_embeddings` can check each item's rank and
        # row count instead of a single blob's.
        if len(items) == 1:
            return [rows]
        counts = [int(it.image_grid_thw.prod().item()) for it in items]
        merge = getattr(handle, "spatial_merge_unit", 4) or 4
        splits = [max(1, c // merge) for c in counts]
        return list(torch.split(rows, splits, dim=0))

    # -- release ------------------------------------------------------------
    def release_tower(self, handle: Any) -> None:
        import torch

        del handle
        torch.cuda.empty_cache()


class VisionStageBackendRefused(VisionStageLoadRefused):
    """No multimodal attention backend fits this card, by its own numbers.

    A subclass of :class:`VisionStageLoadRefused` on purpose: it happens on the
    LOAD leg, before a single pixel is encoded, and ``code_for`` already maps
    that class to ``W106``.  A new W-code for "the kernel would not have fit"
    would split one seam across two codes for no gain.
    """


#: Shared memory ONE BLOCK of the Triton vision-attention kernel demands, bytes.
#:
#: MEASURED, not modelled -- this is the number the driver itself reported on
#: metal boot xsn407 (20.09. 17:16Z), card0, refusing the launch::
#:
#:   W107 Weg2VisionEncodeFailed rid=weg2-8-27 -- VisionStageEncodeFailed:
#:   card0: the encoder forward over 1 item(s) raised: out of resource: shared
#:   memory, Required: 131072, Hardware limit: 101376.
#:
#: It is a FLOOR for the decision, not a formula: the block sizes live inside
#: the shared prefill kernel ``context_attention_fwd``
#: (``VisionTritonAttention.forward`` just calls it), so they are not a knob
#: this stage can turn -- halving them would change the kernel the whole engine
#: prefills with, which is not a transient tower's business.
TRITON_VISION_ATTN_SMEM_BYTES = 131072

#: The backend chosen when the Triton kernel does not fit.  ``sdpa`` has no
#: opt-in shared-memory demand at all: it builds a boolean mask ``[1, s, s]``
#: (``VisionSdpaAttention._generate_mask_cache``, ``layers/attention/vision.py:209``)
#: and hands the rest to torch.  At the design's named geometry -- 1024x1024,
#: 4096 patch rows -- that mask is 4096^2 bytes = 16 MiB, which is small beside
#: the 79 MiB of encoder activation the arming line already books.
VISION_BACKEND_FALLBACK = "sdpa"

#: Backends that carry the Triton kernel's demand.  Named as a set rather than
#: an ``== "triton_attn"`` so a second Triton-backed entry cannot slip past.
TRITON_BACKED_BACKENDS = frozenset({"triton_attn"})


def choose_vision_attention_backend(
    *,
    smem_optin_bytes: Optional[int],
    card: int = -1,
    operator_override: Optional[str] = None,
) -> Tuple[str, str]:
    """(backend, reason) for the transient tower, from the card's OWN numbers.

    THE DEFECT THIS ENDS (metal boot xsn407): upstream's default table
    (``vision.py:_determine_attention_backend``) special-cases exactly two
    capabilities -- major 9 -> ``fa3``, major 10 -> ``fa4`` -- and everything
    else on CUDA falls through to ``triton_attn``.  BOTH card families of this
    rig fall through: the 3080 is sm_86 and the 5090 is sm_120.  And both report
    the same opt-in shared memory, 101376 bytes, against the kernel's 131072 --
    so the Triton vision kernel fits on NO CARD OF THIS RIG, and the way that
    surfaced was a kernel launch failure in the middle of the encode leg.

    The decision is made HERE, before the module is built, out of
    ``shared_memory_per_block_optin`` -- the torch spelling of
    ``cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)``.

    Four outcomes, and none of them is a guess:

    * an operator override that is not Triton-backed -- honoured as given;
    * an operator override that IS, on a card too small -- REFUSED by name,
      because running it reproduces xsn407 on purpose;
    * a capability that cannot be read -- REFUSED by name.  Choosing a kernel
      against an unknown limit is how a modelled fit becomes a launch failure;
    * otherwise: Triton when it fits, ``sdpa`` when it does not, with BOTH
      numbers in the reason either way.
    """
    if operator_override:
        if operator_override not in TRITON_BACKED_BACKENDS:
            return operator_override, (
                f"operator override --mm-attention-backend={operator_override} "
                "honoured as given (not a Triton-backed backend, so the "
                "shared-memory ceiling does not apply)"
            )
        if smem_optin_bytes is None or int(smem_optin_bytes) <= 0:
            raise VisionStageBackendRefused(
                f"card{card}: --mm-attention-backend={operator_override} was "
                "requested and this card's opt-in shared memory could not be "
                "read, so whether the kernel fits is unknown. Refusing rather "
                "than launching into the xsn407 failure."
            )
        if int(smem_optin_bytes) < TRITON_VISION_ATTN_SMEM_BYTES:
            raise VisionStageBackendRefused(
                f"card{card}: --mm-attention-backend={operator_override} was "
                f"requested, and that kernel needs "
                f"{TRITON_VISION_ATTN_SMEM_BYTES} B of shared memory per block "
                f"while this card admits {int(smem_optin_bytes)} B opt-in. The "
                "launch would fail inside the encode leg (metal boot xsn407). "
                f"Drop the override to get {VISION_BACKEND_FALLBACK}, which has "
                "no opt-in shared-memory demand."
            )
        return operator_override, (
            f"operator override --mm-attention-backend={operator_override} "
            f"honoured: card admits {int(smem_optin_bytes)} B opt-in >= the "
            f"kernel's {TRITON_VISION_ATTN_SMEM_BYTES} B"
        )

    if smem_optin_bytes is None or int(smem_optin_bytes) <= 0:
        raise VisionStageBackendRefused(
            f"card{card}: the opt-in shared memory per block could not be read "
            f"(got {smem_optin_bytes!r}), so no backend can be chosen against a "
            "known limit. Refusing rather than defaulting -- a kernel picked "
            "against an unknown ceiling is how xsn407 failed mid-encode."
        )

    smem = int(smem_optin_bytes)
    if smem >= TRITON_VISION_ATTN_SMEM_BYTES:
        return "triton_attn", (
            f"card{card} admits {smem} B opt-in shared memory >= the Triton "
            f"vision kernel's {TRITON_VISION_ATTN_SMEM_BYTES} B"
        )
    return VISION_BACKEND_FALLBACK, (
        f"card{card} admits only {smem} B opt-in shared memory and the Triton "
        f"vision kernel needs {TRITON_VISION_ATTN_SMEM_BYTES} B (short by "
        f"{TRITON_VISION_ATTN_SMEM_BYTES - smem} B), so the transient tower "
        f"uses {VISION_BACKEND_FALLBACK} instead. Upstream's default table "
        "would have picked triton_attn here: it special-cases only sm_90 and "
        "sm_100 and lets every other CUDA capability fall through"
    )


def smem_optin_for_torch_index(torch_index: int) -> Optional[int]:
    """This card's ``cudaDevAttrMaxSharedMemoryPerBlockOptin``, or ``None``.

    ``None`` on any failure rather than a number: the caller REFUSES on
    ``None``, and a fabricated ceiling is the one answer that turns a readable
    refusal back into a kernel launch failure.
    """
    try:
        import torch

        return int(
            torch.cuda.get_device_properties(
                torch_index
            ).shared_memory_per_block_optin
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "vision stage: could not read shared_memory_per_block_optin for "
            "torch index %s: %s",
            torch_index,
            exc,
        )
        return None


@contextlib.contextmanager
def _mm_attention_backend(backend: str):
    """Hold ``mm_attention_backend`` at ``backend`` while the tower is built.

    Construction-scoped on purpose, and that is enough: ``VisionAttention``
    resolves the backend ONCE in ``__init__`` and stores the instance as
    ``self.qkv_backend`` (``layers/attention/vision.py:993``), so the choice is
    baked into the module and the global goes back to what it was. The process
    this runs in is the tokenizer's -- it serves no attention of its own -- but
    the restore is not optional for that reason: a global left changed is a
    defect waiting for the next reader, not a saved line.
    """
    from sglang.srt.runtime_context import get_server_args

    server_args = get_server_args()
    previous = server_args.mm_attention_backend
    server_args.mm_attention_backend = backend
    try:
        yield
    finally:
        server_args.mm_attention_backend = previous


def _build_qwen3vl_tower(hf_config: Any, torch_index: int) -> Any:
    """The real tower module, alone, in a process that is not a rank.

    A world-size-1 distributed environment has to exist first: the module
    reaches for ``get_pp_group()`` and builds a ``VocabParallelEmbedding``
    (``qwen3_vl.py:325,344``), neither of which exists in a tokenizer process.
    The precedent is upstream's own standalone encoder
    (``disaggregation/encode_server.py:305-312``), which does exactly this
    before ``get_model``.
    """
    import torch

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
    from sglang.srt.utils.network import get_open_port

    torch.cuda.set_device(torch_index)
    if not model_parallel_is_initialized():
        init_distributed_environment(
            backend="gloo",
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
    # THE BACKEND IS DECIDED BEFORE THE MODULE EXISTS, from this card's own
    # shared-memory ceiling -- see `choose_vision_attention_backend`. Deciding
    # it afterwards is what xsn407 did by omission, and the answer arrived as a
    # kernel launch failure in the middle of the encode leg.
    from sglang.srt.runtime_context import get_server_args

    backend, why = choose_vision_attention_backend(
        smem_optin_bytes=smem_optin_for_torch_index(torch_index),
        card=torch_index,
        operator_override=get_server_args().mm_attention_backend,
    )
    logger.info(
        "[vision-attn] torch_index=%d backend=%s -- %s", torch_index, backend, why
    )

    vision_config = getattr(hf_config, "vision_config", None)
    with _mm_attention_backend(backend):
        module = Qwen3VLMoeVisionModel(
            vision_config,
            norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            quant_config=None,
            prefix="model.visual",
            use_data_parallel=False,
        )
    return module.to(device=torch.device("cuda", torch_index))


# ---------------------------------------------------------------------------
# arming
# ---------------------------------------------------------------------------

#: The hook names the design's §6 table requires.  Checked as a SET against the
#: built service, so a hook that is added to the design and forgotten here is a
#: failing test and not a silent no-op at 3 a.m.
REQUIRED_HOOKS = (
    "nvml_snapshot",
    "h2d_gbps",
    "tower",
    "load_tower",
    "encode",
    "release_tower",
    "read_gbps",
    "pause_tag",
    "resume_tag",
)


def missing_hooks(service: Any) -> Tuple[str, ...]:
    """Hook names from :data:`REQUIRED_HOOKS` the service does not carry.

    ``pause_tag`` and ``resume_tag`` count as PRESENT when they are ``None``:
    ``None`` is their correct value in this process (the design's §6 table
    says so, and ``VisionStageService.eviction_available`` reads it), so the
    check is "the attribute exists", not "the attribute is truthy" -- the
    naive-truthy shape would have called the honest default a missing hook.
    """
    return tuple(h for h in REQUIRED_HOOKS if not hasattr(service, h))


def build_service(
    *,
    model_dir: str,
    hf_config: Any,
    snapshot: Callable[[], Sequence[Tuple[Any, Any]]],
    h2d_gbps: Optional[Dict[int, float]] = None,
    context_runner: Optional[Callable[[Sequence[str], Dict[str, str]], str]] = None,
    build_module: Optional[Callable[[Any, int], Any]] = None,
    booked_hw: Tuple[int, int] = BOOKED_IMAGE_HW,
    tree: str = "",
    prefer_cards: Tuple[int, ...] = (),
) -> Tuple[Any, ContextMeasurement, VisionEncoderConfig]:
    """Build the service the design's §6 table describes.  Raises, never half-builds.

    Order matters and is the cheap-first order: the checkpoint is read from
    headers (microseconds, no GPU) BEFORE a child process is spawned to touch
    CUDA, so a checkpoint that cannot serve images refuses without ever having
    cost a context.
    """
    rates = dict(MEASURED_H2D_GBPS if h2d_gbps is None else h2d_gbps)
    cfg = encoder_config_from_hf(getattr(hf_config, "vision_config", None))

    # -- the checkpoint, from headers only ---------------------------------
    shard = find_tower_shard(model_dir)
    extent = tower_extent(shard)
    if not extent.contiguous:
        raise VisionStageTowerUnreadable(
            f"the tower in {shard} is not one contiguous extent "
            f"({extent.gap_bytes} bytes of non-tower inside it); the "
            "single-extent cost model does not hold"
        )

    # -- the posts that depend on the booked geometry ----------------------
    rows = cfg.patch_rows(booked_hw[0], booked_hw[1])
    activation = encoder_activation_bytes(cfg, rows)
    embeddings = embedding_bytes_for(cfg, rows)

    # -- the context, measured in a child that then dies -------------------
    snap = list(snapshot())
    card, uuid = probe_card(snap, rates)
    ctx = measure_context_bytes(card, uuid, runner=context_runner, tree=tree)
    logger.info("%s", ctx.log_line())

    tower: TowerSpec = tower_from_span(
        **extent.as_span_args(),
        ctx_bytes=ctx.ctx_bytes,
        activation_bytes=activation,
        embedding_bytes=embeddings,
    )
    hooks = TowerHooks(
        model_dir=model_dir,
        hf_config=hf_config,
        snapshot=snapshot,
        build_module=build_module,
    )
    service = _vss.VisionStageService(
        nvml_snapshot=snapshot,
        h2d_gbps=rates,
        tower=tower,
        load_tower=hooks.load_tower,
        encode=hooks.encode,
        release_tower=hooks.release_tower,
        read_gbps=ARM_READ_GBPS,
        # `None` on purpose and NOT an omission: band displacement lives in the
        # rank processes (`offload_movement.py:338-341`) and this one cannot
        # reach it. With both None the service FORBIDS eviction and a placement
        # that would need a band refuses by name (W105) with its arithmetic,
        # instead of being silently attempted or silently skipped.
        pause_tag=None,
        resume_tag=None,
        encoder_config=cfg,
        prefer_cards=tuple(prefer_cards),
    )
    absent = missing_hooks(service)
    if absent:
        raise VisionStageArmRefused(
            f"the service was built without {list(absent)}; the design's "
            "§6 hook table is the contract and a partial service is a stage "
            "that fails at the seam nobody filled"
        )
    return service, ctx, cfg


def arm_transient_vision(
    server_args: Any = None,
    model_config: Any = None,
    *,
    multimodal: bool = True,
    env: Optional[Dict[str, str]] = None,
    snapshot: Optional[Callable[[], Sequence[Tuple[Any, Any]]]] = None,
    context_runner: Optional[Callable[[Sequence[str], Dict[str, str]], str]] = None,
    build_module: Optional[Callable[[Any, int], Any]] = None,
    h2d_gbps: Optional[Dict[int, float]] = None,
    booked_hw: Tuple[int, int] = BOOKED_IMAGE_HW,
) -> Optional[Any]:
    """Arm the transient stage for this process, or refuse BY NAME.

    Returns the installed service, or ``None``.  ``None`` has two meanings and
    they are NOT the same, which is why the refusal is recorded rather than
    returned:

    * this is not a transient boot -- nothing happened, no state was touched,
      and the text path is byte-for-byte what it was;
    * this IS a transient boot and arming refused -- W111 is in the log, the
      reason is recorded via ``vision_stage_service.install_refusal``, and the
      first image request refuses by name with that reason quoted (W112).

    It never raises.  An arming failure must not take down a boot that can
    still serve text perfectly well; it must make the IMAGE path loud, which
    is exactly what the recorded refusal does.
    """
    if not vision_mode(env):
        return None

    model_dir = ""
    for src, name in ((server_args, "model_path"), (model_config, "model_path")):
        model_dir = model_dir or str(getattr(src, name, "") or "")
    hf_config = getattr(model_config, "hf_config", None)

    try:
        if not multimodal:
            raise VisionStageArmRefused(
                "this boot asked for --weg2-vision transient and its tokenizer "
                "built NO multimodal processor, so no mm_items will ever exist "
                "and the stage could never run. The transient form needs "
                "`--json-model-override-args '{\"language_model_only\": true}'` "
                "and NOT `--no-enable-multimodal`: the latter also switches off "
                "the tokenizer's image path (model_config.py:573 "
                "is_multimodal). Check the argv this boot was launched with."
            )
        if not model_dir:
            raise VisionStageArmRefused(
                "no model path on the server args; the tower's shard cannot be "
                "found and the stage cannot be sized"
            )
        if hf_config is None:
            raise VisionStageArmRefused(
                "no hf_config on the model config; the encoder geometry would "
                "have to come from VisionEncoderConfig's defaults, which are "
                "ANOTHER checkpoint's numbers"
            )
        snap = snapshot
        if snap is None:
            from sglang.srt.registry.nvml import memory_snapshot as snap  # noqa: N813
        service, ctx, cfg = build_service(
            model_dir=model_dir,
            hf_config=hf_config,
            snapshot=snap,
            h2d_gbps=h2d_gbps,
            context_runner=context_runner,
            build_module=build_module,
            booked_hw=booked_hw,
        )
    except Exception as exc:  # noqa: BLE001 -- every path out is a named refusal
        _vss.install_refusal(f"{type(exc).__name__}: {exc}")
        return None

    _vss.install(service)
    tower = service.tower
    logger.info(
        "%s ARMED pid=%d model=%s tower=%d pieces %.3fGiB ctx=%.0fMiB "
        "activation=%.0fMiB(booked for %dx%d = %d patch rows) "
        "embeddings=%.0fMiB total=%.3fGiB read_gbps=%.2f(BUFFERED) "
        "h2d_gbps=%s eviction=%s",
        _vss.W_STAGE_OK,
        os.getpid(),
        model_dir,
        tower.pieces,
        tower.weight_bytes / GIB,
        tower.ctx_bytes / MIB,
        tower.activation_bytes / MIB,
        booked_hw[0],
        booked_hw[1],
        cfg.patch_rows(booked_hw[0], booked_hw[1]),
        tower.embedding_bytes / MIB,
        tower.total_bytes / GIB,
        service.read_gbps,
        dict(sorted(service.h2d_gbps.items())),
        "UNAVAILABLE in this process (rank-side only)",
    )
    return service
