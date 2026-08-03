# SPDX-License-Identifier: Apache-2.0
"""Boot refusal and replay-boundary check for the capturable MoE offload fetch.

THREE GATES LIVE HERE
---------------------
1. :func:`refuse_capturable_offload_decode` -- a BOOT refusal for the whole
   capturable decode path (``SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1``). #443's port
   was measured on a card and REFUTED: see the module constant
   :data:`REFUTATION` and #452.
2. :func:`check_after_graph_replay` -- the #394 cold-tier seam's breach counter,
   read at the replay boundary. Reachable only past gate 1's development
   override, and documented below.
3. :func:`validate_breakable_boot` -- the BOOT preconditions of #462's
   breakable route, the mode gate 1's refusal left as the only survivor.
   Selected through :func:`resolve_offload_graph_mode`, which is now THE one
   place the offload's graph mode is decided; the mechanism lives in
   ``layers/moe/breakable_offload.py``.

WHY THE MODE DECISION IS HERE AND NOT WITH THE MECHANISM
--------------------------------------------------------
Same invariant this module was split out for (see WHY A SEPARATE MODULE): the
decision must be reachable without importing torch, the breakable-graph runner
or the 3.8k-line offload machinery, so that a boot can refuse before any of it
is constructed and so the refusals stay testable hermetically. The mode
constants therefore live here and ``breakable_offload`` imports them, not the
other way round.

WHAT GATE 1 GUARDS
------------------
``scripts/dev/443_graph_proof/`` specified four boot claims for the capturable
decode path. The window ran on 2026-08-02 (evidence:
``/spinning/gpu-battery-results/2026-08-02_desync_graph_proof/RESULTS.md``,
DeepSeek-V4-Flash-0731 UD-IQ3_XXS, TP=3, both arms same boot day, same frozen
residency, same reserve):

* **B1 PASS** -- capture succeeds where the eager path refused. 13 of 18 decode
  batches report ``cuda graph: True``; no capture error in the boot log.
* **B2 FAIL** -- the graph arm and the eager arm decode DIFFERENT text from the
  same greedy prompt. Each arm is internally deterministic over three runs (one
  output hash each), so the difference is systematic, not sampling noise. The
  texts diverge at character 5 of 533.
* **B4 REGRESSION 6.60x** -- 984.4 ms/token captured vs 149.1 ms/token eager.
  Localised to the captured decode step: the per-rank prefill rounds move only
  +1.5 % between the arms.

B4 is STRUCTURAL, not a tuning miss, and that is why this is a refusal rather
than a warning. A CUDA graph cannot vary its work with the data, so the
captured gather must move the WORST-CASE scratch set -- all ``C`` slots -- every
layer, every step. The eager fetch moves only the experts the step actually
missed. On the measured recipe that is 6 rows vs a measured mean of 1.03-1.50
rows per (layer, forward): 2.128 GiB/token against 0.366-0.535 GiB/token, a
5.35x PCIe multiplier on a decode step that is PCIe-bound (eager: 0.398 GiB in
149.1 ms = 2.67 GiB/s, near this rig's H2D DMA ceiling). No amount of copy
engine or stream work removes bytes that the graph is obliged to move.
See ``docs/dev/NOTE_452_desync_boot_refutation.md``.

WHAT GATE 2 GUARDS
------------------
The capturable offload path (``expert_offload.prepare_capturable``) resolves a
routed expert to a row of THIS rank's pinned cold pool with pure on-device
index math, and gathers that row with a captured ``index_select``. Under the
#394 shared cold tier a routed expert can be DELEGATED: its bytes live in a
peer's segment and this rank's pool has no row for it, so the frozen
``spill_pool_row_lut`` holds ``-1`` for it. ``-1`` is not a diagnosis; it is an
out-of-bounds index.

The eager path never reaches that state, because ``_fetch`` branches on
``expert_id in self._remote_ids`` and copies from the peer view instead. That
branch is a host read of a Python set and cannot exist inside a capture, which
is exactly why the capturable installer refuses a cold tier by default.

``SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE=1`` is the development window's switch past
that refusal. With this gate in place it is no longer a switch into undefined
behaviour: the capturable remap clamps the index to a valid row and increments
a DEVICE-RESIDENT breach counter instead, and the counter is read here -- on
the host, after the graph has finished replaying -- and turned into a named
exception. A window that trips it learns which layer and how many routed slots,
rather than reading a plausible wrong expert's weights.

WHY A SEPARATE MODULE
---------------------
Same reason as ``barlink_abort_gate`` (#431), whose shape this follows: the
check fires at the CUDA-graph replay boundary in ``model_executor``, which must
not grow an import of ``expert_offload`` -- 3.6k lines pulling in the whole
staging, planner and NVML stack -- in a process that never installed an
offload. This module's own import cost is ``logging``, ``os`` and
``threading``. Its PARENT package is not free (``layers/moe/__init__`` builds
the runner config), but the graph backends already import it transitively, so
the marginal cost of the line added there is the gate module alone; verified
by importing ``full_cuda_graph_backend`` and reading ``sys.modules``:
``layers.moe`` present, ``layers.moe.expert_offload`` absent.

CAPTURE SAFETY
--------------
Reading the counter is a device read and therefore synchronizes; inside a
stream capture that is illegal, not merely slow. Nothing in this module is
called from inside a capture: the counter is incremented by a captured kernel
and read only here, at the boundary, which is the next host point after the
replay.

DEFAULT PATH
------------
With no cold-tier offload cache in the process the registry is empty and
``check_after_graph_replay`` returns after one truth test on a list. A launch
that does not set ``SGLANG_MOE_COLD_TIER_SHM`` never registers anything.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, List

logger = logging.getLogger(__name__)

#: Kill switch for the whole family. "0" restores the pre-port behaviour: the
#: breach counter is still incremented by the captured kernel, but nothing
#: reads it, so a delegated expert routed into a captured step silently reads
#: row 0 of the local pool. It exists so an operator can bisect around the
#: check, not because running without it is a reasonable operating point.
ENV_ENABLE = "SGLANG_MOE_OFFLOAD_CAPTURE_GATE"

#: Development override for gate 1. Set it to run the REFUTED capturable decode
#: path in a card window (to localise B2, or to measure a candidate fix against
#: the numbers in :data:`REFUTATION`). It is not a performance option: the
#: measured operating point is 6.60x SLOWER than eager and decodes different
#: text. Read per call, like every other flag here.
ENV_GRAPH_REFUTED_OVERRIDE = "SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE"

#: The measured facts the refusal cites, kept as data so the boot message, the
#: docs and the tests quote ONE source instead of three paraphrases.
REFUTATION = {
    "evidence": (
        "/spinning/gpu-battery-results/2026-08-02_desync_graph_proof/RESULTS.md"
    ),
    "b1": "capture succeeds (13/18 decode batches captured, no capture error)",
    "b2": (
        "systematic content divergence -- the graph arm and the eager arm decode "
        "different text from the same greedy prompt, each arm internally "
        "deterministic over 3 runs, diverging at character 5 of 533"
    ),
    "b4": (
        "6.60x decode regression -- 984.4 ms/token captured vs 149.1 ms/token "
        "eager, localised to the captured decode step (prefill +1.5 %)"
    ),
}

#: #462 mode selector. ``""``/``eager`` keep the shipped eager offload path;
#: ``breakable`` selects the fetch-outside-the-graph route. ``capturable`` names
#: the refuted path in the same vocabulary -- selecting it is equivalent to
#: ``SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1`` and hits the same #452 refusal.
ENV_GRAPH_MODE = "SGLANG_MOE_OFFLOAD_GRAPH_MODE"

MODE_EAGER = "eager"
MODE_CAPTURABLE = "capturable"
MODE_BREAKABLE = "breakable"
MODES = (MODE_EAGER, MODE_CAPTURABLE, MODE_BREAKABLE)

_FALSE = ("0", "false", "no", "off", "")

_lock = threading.Lock()
_caches: List[Any] = []


class CapturableOffloadRefuted(RuntimeError):
    """The capturable MoE offload decode path was refuted at boot (#452)."""


class BreakableModeRefused(RuntimeError):
    """The breakable offload route was asked for in a shape that cannot work.

    Named, rather than a bare ``RuntimeError``, for the reason the sibling
    refusals give: an operator who catches this wants to know WHICH
    precondition failed, and a test that pins the refusal must not be satisfied
    by an unrelated error raised from the same constructor.
    """

    def __init__(self, reason: str, remedy: str) -> None:
        self.reason = reason
        self.remedy = remedy
        super().__init__(
            f"MoE expert offload: the breakable graph route "
            f"({ENV_GRAPH_MODE}={MODE_BREAKABLE}) refuses -- {reason}. {remedy}"
        )


def graph_override_enabled() -> bool:
    """True when the development override past gate 1 is set."""
    return _env_flag(ENV_GRAPH_REFUTED_OVERRIDE, False)


def refuse_capturable_offload_decode(layer_id: Any = None) -> None:
    """Gate 1: refuse ``SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1`` by name.

    Called from ``FusedMoE.__init__`` at the ONE point where the capturable
    decode mode is selected, so a launch that asks for it aborts before any
    weight is staged rather than serving 6.60x slower and wrong. Nothing on the
    default path reaches this function: it runs only when the offload fraction
    is < 1.0 AND the opt-in env is set.

    Raises :class:`CapturableOffloadRefuted` unless
    ``SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE=1``, in which case it logs the same
    facts once and returns.
    """
    if graph_override_enabled():
        logger.warning(
            "MoE capturable offload decode (SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1) is "
            "REFUTED at boot and is running only because "
            "%s=1. Measured 2026-08-02: B2 %s; B4 %s. Evidence: %s.",
            ENV_GRAPH_REFUTED_OVERRIDE,
            REFUTATION["b2"],
            REFUTATION["b4"],
            REFUTATION["evidence"],
        )
        return
    where = "" if layer_id is None else f" (layer {layer_id})"
    raise CapturableOffloadRefuted(
        f"SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1 selects the capturable MoE offload "
        f"decode path{where}, which was REFUTED on hardware (#452). Measured "
        f"2026-08-02 on DeepSeek-V4-Flash-0731 UD-IQ3_XXS, TP=3, both arms same "
        f"boot day and same frozen residency: B1 {REFUTATION['b1']}, but B2 "
        f"{REFUTATION['b2']}, and B4 {REFUTATION['b4']}. B4 is structural: a "
        f"CUDA graph cannot vary its work with the data, so the captured gather "
        f"moves the worst-case scratch set every layer every step (2.128 "
        f"GiB/token) where the eager fetch moves only the missed experts "
        f"(0.366-0.535 GiB/token measured). Run the eager offload path "
        f"(--disable-cuda-graph, unset SGLANG_MOE_OFFLOAD_CUDA_GRAPH), which is "
        f"the 149.1 ms/token arm. Evidence: {REFUTATION['evidence']}; verdict "
        f"and repricing: docs/dev/NOTE_452_desync_boot_refutation.md. "
        f"{ENV_GRAPH_REFUTED_OVERRIDE}=1 re-opens it for a card window."
    )


def env_graph_mode() -> str:
    """The requested offload graph mode. Read per call, like every flag here."""
    raw = (os.environ.get(ENV_GRAPH_MODE) or "").strip().lower()
    if not raw:
        return MODE_EAGER
    if raw not in MODES:
        raise BreakableModeRefused(
            reason=f"{ENV_GRAPH_MODE}={raw!r} is not a known mode",
            remedy=f"Pick one of {', '.join(MODES)}.",
        )
    return raw


def resolve_offload_graph_mode(
    offload_fraction: float, opt_in: bool, layer_id: Any = None
) -> str:
    """THE selection point for the MoE offload's graph mode.

    Returns one of :data:`MODES`:

    * ``eager`` -- the shipped ``run_waves`` path, byte-untouched. Every launch
      that does not offload, and every launch that asks for nothing, lands here.
    * ``capturable`` -- the in-graph fetch. REFUTED (#452); reaching this return
      at all requires the development override, because
      :func:`refuse_capturable_offload_decode` raises first otherwise.
    * ``breakable`` -- #462: eager fetch into fixed slots before replay, compute
      captured. Its own preconditions are checked by
      :func:`validate_breakable_boot`, which the caller invokes next.

    Two ways to ask for the refuted path (the legacy
    ``SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1`` and the new
    ``SGLANG_MOE_OFFLOAD_GRAPH_MODE=capturable``) resolve to the same outcome
    and the same refusal, so the refusal cannot be walked around by spelling it
    the new way.
    """
    offloading = float(offload_fraction) < 1.0
    mode = env_graph_mode()

    if mode == MODE_BREAKABLE:
        if not offloading:
            # Nothing to offload -> nothing for a slot arena to hold. Let
            # validate_breakable_boot say so with its own reason rather than
            # silently downgrading to eager.
            return MODE_BREAKABLE
        return MODE_BREAKABLE

    if mode == MODE_CAPTURABLE or bool(opt_in):
        if not offloading:
            return MODE_EAGER
        refuse_capturable_offload_decode(layer_id)
        return MODE_CAPTURABLE

    return MODE_EAGER


def resolve_graph_mode(
    offload_fraction: float, opt_in: bool, layer_id: Any = None
) -> bool:
    """Is the CAPTURABLE decode mode selected? (compatibility shim.)

    ``FusedMoE.__init__`` stores this in ``_moe_offload_graph_mode``. Kept as a
    boolean over :func:`resolve_offload_graph_mode` so the one decision lives in
    one place while the capturable path's existing call sites and its #452
    regate tests keep their exact contract.

    Returns False -- the eager ``run_waves`` path, byte-untouched -- for every
    launch that does not both offload (``fraction < 1.0``) and select the
    capturable mode. For the launch that does, refuses by name (#452) unless the
    development override is set, in which case it returns True.
    """
    return (
        resolve_offload_graph_mode(offload_fraction, opt_in, layer_id)
        == MODE_CAPTURABLE
    )


def validate_breakable_boot(offload_fraction: float, layer_id: Any = None) -> None:
    """Gate 3: refuse, at boot, every shape of the breakable route that cannot
    work.

    Called from ``FusedMoE.__init__`` right after the mode is selected, so a
    launch that asks for something impossible dies before a weight is staged
    rather than at first capture -- the fail-fast rule the rest of the offload
    surface already follows.

    Three preconditions, each with its own reason:

    1. there is something to offload (``fraction < 1.0``);
    2. the decode phase is actually captured by the BREAKABLE backend, because
       ``eager_on_graph`` is a PASS-THROUGH under any other backend -- there is
       no segment to split -- so the route's host reads would land inside a real
       stream capture, where a D2H sync is illegal rather than merely slow;
    3. the refuted capturable opt-in is not also set, so a launch cannot ask for
       two mutually exclusive graph modes and silently get one of them.
    """
    if not float(offload_fraction) < 1.0:
        raise BreakableModeRefused(
            reason=(
                f"the MoE expert offload is not active on layer {layer_id} "
                f"(resident fraction {float(offload_fraction):.3f} >= 1.0), so "
                f"there are no slots for a graph to address"
            ),
            remedy=(
                "Set --rank-moe-resident-fraction below 1.0, or unset "
                f"{ENV_GRAPH_MODE}."
            ),
        )

    if _env_flag("SGLANG_MOE_OFFLOAD_CUDA_GRAPH", False):
        raise BreakableModeRefused(
            reason=(
                "SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1 (the capturable IN-GRAPH "
                "fetch) is set at the same time. The two are mutually "
                "exclusive: one keeps the fetch eager and captures only the "
                "compute, the other captures the fetch itself and was REFUTED "
                f"on hardware -- B2 {REFUTATION['b2']}; B4 {REFUTATION['b4']}"
            ),
            remedy=(
                "Unset SGLANG_MOE_OFFLOAD_CUDA_GRAPH. The breakable route "
                "neither needs it nor is compatible with it."
            ),
        )

    backend = resolved_backend("decode")
    if backend is NO_SERVER_ARGS:
        # There are no server args in this process AT ALL (unit / test
        # context): nothing to validate against, and no boot to protect. The
        # runtime scratch guard still applies.
        return
    if backend is None:
        # Server args ARE present and the decode backend could not be resolved
        # from them (#500-B8). This used to share the arm above, which made the
        # gate's own None branch a total bypass of BOTH preconditions -- and
        # `None` is what `resolved_backend` answers for a missing
        # `cuda_graph_config`, a missing phase config and a phase whose
        # `backend` is unset, all of which are real boots. Unresolvable is not
        # "nothing to check": the route is a no-op off the breakable backend,
        # so admitting an unknown backend admits exactly the illegal in-capture
        # host read this gate exists to prevent.
        raise BreakableModeRefused(
            reason=(
                f"the decode CUDA-graph backend could not be resolved on layer "
                f"{layer_id}. Server args are present, but "
                f"`cuda_graph_config.decode.backend` is unset, so the gate "
                f"cannot establish that the decode graph is the BREAKABLE one. "
                f"The route splits the decode graph at the MoE fetch through "
                f"eager_on_graph, which is a NO-OP under any other backend -- "
                f"admitting an unknown backend would let the fetch's host reads "
                f"execute inside a real stream capture, where they are illegal"
            ),
            remedy=(
                "Launch with an explicit --cuda-graph-backend-decode=breakable, "
                f"or unset {ENV_GRAPH_MODE} to keep the shipped eager offload "
                "path."
            ),
        )
    if backend != "breakable":
        raise BreakableModeRefused(
            reason=(
                f"the resolved decode CUDA-graph backend is {backend!r}, not "
                f"'breakable'. The route splits the decode graph at the MoE "
                f"fetch through eager_on_graph, which is a NO-OP under any "
                f"other backend, so the fetch's host reads would execute inside "
                f"a real stream capture where they are illegal"
            ),
            remedy=(
                "Launch with --cuda-graph-backend-decode=breakable, or unset "
                f"{ENV_GRAPH_MODE} to keep the shipped eager offload path."
            ),
        )

    # Reached only with the decode backend RESOLVED to 'breakable', so the
    # args and their cuda_graph_config both exist and `prefill` is never the
    # NO_SERVER_ARGS sentinel here. `None` at this point means the launch
    # configures no prefill graph at all -- which IS the state this
    # precondition requires, so it passes rather than refusing.
    prefill = resolved_backend("prefill")
    if prefill is not None and prefill != "disabled":
        raise BreakableModeRefused(
            reason=(
                f"the resolved prefill CUDA-graph backend is {prefill!r}, not "
                f"'disabled'. A prefill chunk routes far more DISTINCT experts "
                f"than the slot arena has scratch slots, and the eager path "
                f"handles that by wave-splitting the forward -- which a "
                f"captured segment cannot do, because its work is fixed at "
                f"capture time. Prefill therefore has to stay eager; this is "
                f"the same decode-graph + eager-prefill hybrid the capturable "
                f"path uses, and it is a hard shape limit, not a default"
            ),
            remedy=(
                "Launch with --cuda-graph-backend-prefill=disabled. With "
                "prefill eager, prefill forwards fall through to run_waves and "
                "keep their wave splitting; only decode is captured."
            ),
        )


class _NoServerArgs:
    """Sentinel: this process has no ``ServerArgs`` to read at all."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "NO_SERVER_ARGS"


#: Distinct from ``None`` on purpose (#500-B8). "There are no server args"
#: (a unit/test context, nothing to protect) and "the args are there and the
#: backend could not be resolved from them" (a real boot with an unknown graph
#: backend) are opposite situations, and collapsing both onto ``None`` gave
#: :func:`validate_breakable_boot` a silent total-bypass arm. Callers that
#: treat an unresolved backend as permissive MUST distinguish the two.
NO_SERVER_ARGS = _NoServerArgs()


def resolved_backend(phase: str) -> Any:
    """The POST-cascade backend for ``phase``.

    Three distinct answers, and the distinction is load-bearing:

    * a backend name -- resolved;
    * ``None`` -- server args ARE present, but this phase's backend is not
      resolvable from them (no ``cuda_graph_config``, no config for the phase,
      or ``backend`` unset);
    * :data:`NO_SERVER_ARGS` -- there are no server args in this process.

    Read off ``cuda_graph_config`` rather than off the legacy
    ``disable_cuda_graph`` boolean, because several rules rewrite a phase's
    backend after parsing (``_disable_breakable_cudagraph_if_incompatible`` is
    one of them) and the boolean does not see those. The boolean is still
    honoured where it IS authoritative: it forces both phases off.
    """
    try:
        from sglang.srt.runtime_context import get_server_args

        args = get_server_args()
    except Exception:
        return NO_SERVER_ARGS
    if args is None:
        return NO_SERVER_ARGS
    config = getattr(args, "cuda_graph_config", None)
    phase_config = getattr(config, phase, None)
    backend = getattr(phase_config, "backend", None)
    if backend is None:
        return None
    if getattr(args, "disable_cuda_graph", False):
        return "disabled"
    return backend


class OffloadCaptureBreach(RuntimeError):
    """A captured offload gather referenced a row this rank does not own.

    Carries the layer and the counts rather than a bare message: the operating
    question after one of these is always "which layer, and was it one slot or
    every slot", and a summary that drops that costs the reader the only two
    facts that distinguish a mis-built plan from a routing excursion.
    """

    def __init__(self, layer_id: Any, breaches: int, where: str):
        self.layer_id = layer_id
        self.breaches = int(breaches)
        self.where = where
        super().__init__(
            f"MoE offload capture gate ({where}): layer {layer_id} gathered "
            f"{self.breaches} scratch slot(s) whose expert has no row in this "
            f"rank's cold pool. Those experts are DELEGATED to a peer's #394 "
            f"segment, and the capturable gather has no source for them -- the "
            f"clamped index read row 0 of the local pool, so this step's MoE "
            f"output is wrong, not approximate. Run eager "
            f"(--disable-cuda-graph) or without SGLANG_MOE_COLD_TIER_SHM until "
            f"the capturable path sources peer rows."
        )


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE


def gate_enabled() -> bool:
    """Read per call, not at import: tests and operators change it in place."""
    return _env_flag(ENV_ENABLE, True)


def register(cache: Any) -> None:
    """Publish an offload cache that carries a device breach counter."""
    with _lock:
        if cache not in _caches:
            _caches.append(cache)


def unregister(cache: Any) -> None:
    with _lock:
        if cache in _caches:
            _caches.remove(cache)


def registered() -> List[Any]:
    with _lock:
        return list(_caches)


def reset_for_test() -> None:
    """Drop the registry. Tests only."""
    with _lock:
        _caches.clear()


def check_after_graph_replay(where: str = "cuda-graph replay") -> None:
    """Raise if any registered cache's captured gather clamped an index.

    The FIRST statement is the empty-registry test, and that is the whole
    default path: a process without a cold-tier offload pays one truth test on
    a list per replay.

    Raises :class:`OffloadCaptureBreach` for the first layer that reports.
    Deliberately not aggregated across layers -- the first one to breach is the
    one whose plan is wrong, and every later layer would report the same cause.
    """
    if not _caches:
        return
    if not gate_enabled():
        return
    for cache in registered():
        check = getattr(cache, "check_capture_breach", None)
        if check is not None:
            check(where)


__all__ = [
    "ENV_ENABLE",
    "ENV_GRAPH_MODE",
    "ENV_GRAPH_REFUTED_OVERRIDE",
    "MODES",
    "MODE_BREAKABLE",
    "MODE_CAPTURABLE",
    "MODE_EAGER",
    "NO_SERVER_ARGS",
    "REFUTATION",
    "BreakableModeRefused",
    "CapturableOffloadRefuted",
    "OffloadCaptureBreach",
    "check_after_graph_replay",
    "env_graph_mode",
    "gate_enabled",
    "graph_override_enabled",
    "refuse_capturable_offload_decode",
    "register",
    "resolve_graph_mode",
    "resolve_offload_graph_mode",
    "resolved_backend",
    "registered",
    "reset_for_test",
    "unregister",
    "validate_breakable_boot",
]
