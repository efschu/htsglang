# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the shvllm fork (branch feature/barlink),
# vllm/distributed/device_communicators/barlink.py
"""barlink — heterogeneous collective communication layer.

Vendor-neutral collectives for TP groups that span GPUs which share no
common device-native collective library (e.g. NVIDIA/NCCL + AMD/RCCL).
Every collective is executed over the host-staging path that NCCL itself
falls back to when P2P is unavailable:

    GPU (D2H, async) -> pinned host buffer -> gloo collective on the
    group's CPU process group -> (H2D, async) -> GPU

gloo runs entirely CPU-side, so the two endpoints of the collective may
be CUDA and ROCm processes — the device only ever performs plain
``memcpy`` to/from its own pinned staging buffer, which both vendors
implement identically.

Large tensors are processed in chunks and pipelined: while gloo reduces
chunk *i* on the CPU, the D2H copy of chunk *i+1* is already in flight
on the device's copy stream. On systems without P2P between the GPUs
this is functionally the same transport NCCL would use, so forcing
barlink on an all-NVIDIA group (``SGLANG_BARLINK=1``) is a faithful test bed
for the mixed-vendor case.

Limitations (v1):
- Collectives synchronize with the CPU, so they cannot be captured in
  CUDA graphs — run with ``--enforce-eager``.
- Reduction happens on the CPU inside gloo (fp32 accumulation for
  half/bfloat16 inputs via upcast).
"""

import logging
import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators import (
    barlink_env_guard,  # noqa: F401  (rejects retired SGLANG_HTCCL* vars)
    barlink_liveness,
)

logger = logging.getLogger(__name__)

# Chunk size for the D2H -> gloo -> H2D pipeline. Small enough to
# overlap copy and CPU reduction, large enough to amortize per-op
# latency. Tunable via SGLANG_BARLINK_CHUNK_MIB.
_CHUNK_BYTES = int(os.environ.get("SGLANG_BARLINK_CHUNK_MIB", "8")) * 1024 * 1024

# gloo reduces half/bfloat16 with fp32 accumulation only when the
# tensor is upcast explicitly; reducing bf16 directly through gloo
# accumulates in bf16 and loses precision vs NCCL. Upcast by default,
# disable with SGLANG_BARLINK_FP32_REDUCE=0 to trade accuracy for speed.
_FP32_REDUCE = bool(int(os.environ.get("SGLANG_BARLINK_FP32_REDUCE", "1")))


# Preferred data plane: "shm" (pinned shared-memory slots + GPU-side
# reduction, single-node — matches NCCL's no-P2P SHM path) or "gloo"
# (TCP data plane, also works multi-node; slower).
# "device" = GPU-driven kernels over the mapped segment (fastest,
# CUDA-graph-capturable), "shm" = CPU-orchestrated pinned staging,
# "gloo" = TCP data plane (also multi-node).
_TRANSPORT = os.environ.get("SGLANG_BARLINK_TRANSPORT", "device")
# Per-rank shm slot size; all_reduce payloads above this fall back to
# the gloo path. 64 MiB covers a 4096-token x 5120-hidden bf16 chunk.
_SLOT_BYTES = int(os.environ.get("SGLANG_BARLINK_SLOT_MIB", "64")) * 1024 * 1024


class Bar1Failed(RuntimeError):
    """The direct path failed to come up -- WITH a reason.

    The difference from a silent ``None`` is the whole point: a ``None``
    selects the gloo plane and afterwards looks like success. This
    exception carries the reason through to ``_build_transport``, which
    writes it into a warning and into ``_STATE``.
    """

    def __init__(self, reason: str, stage: str = "setup"):
        super().__init__(reason)
        self.reason = reason
        self.stage = stage


#: What ACTUALLY runs per group. Keyed by group name
#: (``GroupCoordinator.unique_name``: "tp", "dcp", ...).
#:
#: This table exists because the log line "barlink enabled for group 'X'
#: (transport=bar1)" used to name the REQUESTED transport, not the one
#: actually achieved. On the real model that meant: tp ran over BAR1, dcp
#: fell back to gloo with ENOMEM, both log lines looked identical, and the
#: throughput number derived from that run (22.83 tok/s) was in part not a
#: BAR1 number at all. A measurement whose arm can't be read off the log
#: is not a measurement.
_STATE: dict[str, dict] = {}


def report_state(group: str, requested: str, achieved: str,
                reason: str = "", stage: str = "") -> dict:
    """Record what this group will actually run on.

    One entry per group name. Nameless groups (which don't actually occur
    in sglang -- ``GroupCoordinator.unique_name`` is always set) get a
    numbered placeholder name, so that two of them don't overwrite each
    other and one silently disappears.
    """
    key = group
    if not key:
        i = 0
        while f"<unnamed #{i}>" in _STATE:
            i += 1
        key = f"<unnamed #{i}>"
    entry = {
        # NOTE: these dict keys are a cross-file data contract.
        # python/sglang/srt/distributed/parallel_state.py reads them back
        # via ``state.get("achieved", ...)`` to build its own
        # "requested=%s, ACHIEVED=%s" log line, and
        # scripts/gpu_battery/s11_bar1_e2e.py rebuilds a lookalike dict with
        # the same spellings by regex-scanning that log line. Change a key
        # here only together with those consumers -- and with the
        # ``SCHEMA_VERSION`` of every gpu_battery artifact that persists it
        # (bar1_e2e.json, prefill_kurve.json), so a stale artifact is
        # rejected loudly instead of silently read as empty.
        "group": key,
        "requested": requested,
        "achieved": achieved,
        "reason": reason,
        "stage": stage,
        "direct": achieved == requested and achieved not in ("gloo", ""),
    }
    _STATE[key] = entry
    return entry


def group_states() -> dict[str, dict]:
    """Which groups actually run over the requested transport.

    Made queryable, not just logged: a measurement program should be able
    to CHECK this, instead of parsing log lines.
    """
    return dict(_STATE)


def state_summary() -> str:
    """One line per group, for the log and for the measurement report."""
    if not _STATE:
        return "barlink: no group reported."
    lines = []
    for name, e in sorted(_STATE.items()):
        if e["direct"]:
            lines.append(f"  {name}: {e['achieved']}")
        else:
            lines.append(
                f"  {name}: {e['achieved']} (REQUESTED WAS "
                f"{e['requested']}; {e['stage']}: {e['reason']})"
            )
    all_direct = all(e["direct"] for e in _STATE.values())
    header = ("barlink: all groups are running the requested transport."
              if all_direct else
              "barlink: NOT all groups are running the requested transport -- "
              "a measurement over this configuration is mixed.")
    return header + "\n" + "\n".join(lines)


# ----------------------------------------------------------------------
# Transport seam
#
# A transport is anything exposing:
#     handles(op: str, nbytes: int) -> bool
#     barlink_<op>(comm, ...)            for each op it declares
# The communicator ASKS `handles` rather than knowing a transport's limits, so
# no transport-specific condition (e.g. the shm slot-size ceiling) lives at a
# call site. Adding a transport -- `ucx` for RDMA is the expected next one --
# is one registry entry plus its module; no dispatch site changes.
#
# `None` means "no transport object": the inline gloo data plane below, which
# is always available and is the universal fallback.
# ----------------------------------------------------------------------


def _make_device_transport(cpu_group, device):
    from sglang.srt.distributed.device_communicators.barlink_device import (
        BarlinkDeviceTransport,
    )

    return BarlinkDeviceTransport(
        cpu_group=cpu_group, device=device, slot_bytes=_SLOT_BYTES
    )


def _make_shm_transport(cpu_group, device):
    from sglang.srt.distributed.device_communicators.barlink_shm import (
        BarlinkShmTransport,
    )

    return BarlinkShmTransport(
        cpu_group=cpu_group, device=device, slot_bytes=_SLOT_BYTES
    )


def _make_host_transport(cpu_group, device):
    from sglang.srt.distributed.device_communicators.barlink_host import (
        BarlinkHostTransport,
    )

    return BarlinkHostTransport(
        cpu_group=cpu_group, device=device, slot_bytes=_SLOT_BYTES
    )


def _make_ucx_transport(cpu_group, device):
    from sglang.srt.distributed.device_communicators.barlink_ucx import (
        BarlinkUcxTransport,
    )

    return BarlinkUcxTransport(cpu_group=cpu_group, device=device)


def _make_bar1_transport(cpu_group, device, group: str = ""):
    """The BAR1 direct path on its own -- no planner, no measurement.

    The source card DMAs straight into the target card's BAR1 aperture: no
    host memory, no NIC, no NCCL. Which of the two ported kernels runs at
    which size follows the `SGLANG_BARLINK_BAR1_RING_THRESHOLD` threshold here,
    because there is no plan to ask. That threshold is a default, not a
    finding -- between 1 and 16 MiB mesh and ring measured within 1..7 % of
    each other. Use "matrix" to have that decided by measurement instead.

    `build_bar1` returns None with a logged reason on any machine that
    cannot do this (holder module absent, driver regkey unset, byte proof
    failed). None is not an error here: it selects the inline gloo plane,
    exactly as an unknown transport name would.
    """
    from sglang.srt.distributed.device_communicators.barlink_bar1 import build_bar1
    from sglang.srt.distributed.device_communicators.barlink_matrix_transport import (
        window_for,
    )

    report: dict = {}
    t = build_bar1(cpu_group, device, window_for(group, device), report,
                  group=group)
    # "holds_space" is written by barlink_bar1.build_bar1 and also read by
    # barlink_matrix_transport; the key spelling is a cross-module contract,
    # so all three sides moved to the English name in one step (#358).
    if t is None or report.get("holds_space"):
        raise Bar1Failed(
            report.get("reason", "no reason reported"),
            stage=report.get("stage", "unknown"),
        )
    return t


def _make_matrix_transport(cpu_group, device, group: str = ""):
    """Planner + BAR1 direct path.

    Builds the direct path, hands its ACTUALLY mapped window to the planner
    as a capability, runs `plan()`, logs `plan.explanation()` on rank 0, and
    feeds the plan back so the kernel choice per size comes from the
    measurement rather than from a built-in number. barlink_matrix_transport
    explains why that is the only order that works.
    """
    from sglang.srt.distributed.device_communicators.barlink_matrix_transport import (
        BarlinkMatrixTransport,
    )

    t = BarlinkMatrixTransport(cpu_group=cpu_group, device=device, group=group)
    if t.bar1 is None:
        # The planner alone is not a transport: `handles` would return
        # False for everything, and every collective would run over the
        # gloo layer -- while the log says "transport=matrix". Exactly the
        # mix-up this is meant to fix.
        reason = getattr(t, "bar1_reason", "") or "no reason reported"
        t.close()
        raise Bar1Failed(reason, stage=getattr(t, "bar1_stage", "setup"))
    return t


# name -> factory. "gloo" is intentionally absent: it is the inline plane.
TRANSPORT_REGISTRY = {
    "device": _make_device_transport,
    "shm": _make_shm_transport,
    # Pinned, portable host memory, driven entirely by two kernels. The name
    # says where the BYTES live, not who drives: unlike shm/gloo/ucx this one
    # never synchronizes with the host, so it is capturable like "device".
    # Its slot geometry follows SGLANG_BARLINK_SLOT_MIB unless
    # SGLANG_BARLINK_HOST_SLOT_MIB overrides it; the rest of its knobs live in
    # its own module, because they describe its kernels, not the communicator.
    "host": _make_host_transport,
    # RDMA data plane for groups that span hosts. Same host-staged semantics
    # as gloo, UCX instead of TCP. Sizing/threshold knobs live in its own
    # module (SGLANG_BARLINK_UCX_*), not here, because they describe the wire,
    # not the communicator.
    "ucx": _make_ucx_transport,
    # GPU-to-GPU straight through the target's BAR1 aperture. Neither the
    # host nor a NIC touches the payload. Needs the relaxed driver guard
    # (BarlinkPeerBar1), the dmabuf_holder module and a passing byte
    # proof; without any of them it opts out cleanly and the gloo plane
    # runs. Its knobs (SGLANG_BARLINK_BAR1_*) live in its own module because
    # they describe its kernels and its BAR1 geometry, not the communicator.
    "bar1": _make_bar1_transport,
    # The same direct path, but with the path-matrix planner deciding role,
    # algorithm and kernel per size from a start-up measurement instead of
    # from a threshold. Strictly more than "bar1"; "bar1" exists so the
    # transport can be measured WITHOUT the planner in the loop.
    "matrix": _make_matrix_transport,
}

# Transports that must NOT silently fall back to the gloo plane on failure.
# The rule is exactly CAPTURABLE_BARLINK_TRANSPORTS: the compilation config
# allowed CUDA graphs on the strength of these, so a CPU-orchestrated
# replacement would be captured and crash later -- and it would crash far from
# the transport that actually failed to come up. "host" is here for the same
# reason "device" is, not for a new one.
#
# "bar1" and "matrix" are deliberately NOT here, and correspondingly not in
# CAPTURABLE_BARLINK_TRANSPORTS. Their data path never touches the host and the
# round number lives in device memory precisely so a replayed graph would not
# reuse a stale one -- so they are capturable BY CONSTRUCTION. What is missing
# is a measurement: nobody has captured a cooperative launch
# (cudaLaunchCooperativeKernel, used above the SGLANG_BARLINK_BAR1_GRID_THRESHOLD
# threshold) into a CUDA graph on this rig and replayed it. Claiming
# capturability on a construction argument is exactly the kind of plausible
# assumption that has been failing against this hardware all day.
_NO_FALLBACK = frozenset({"device", "host"})


def _no_fallback(name: str) -> bool:
    """Whether ``name`` must raise on a build failure instead of falling back.

    The rule is unchanged -- "exactly the capturable set" -- only it is now
    looked up from ``parallel_state`` instead of being written out a second
    time here. Otherwise the enable switch could apply in one place and not
    the other, and bar1 could end up BOTH enabled AND silently falling back
    to gloo: the worst possible combination of the two.
    """
    if name in _NO_FALLBACK:
        return True
    try:
        from sglang.srt.distributed.parallel_state import capturable_transports

        return name in capturable_transports()
    except Exception:
        return False


def _invoke_factory(factory, cpu_group, device, group: str):
    """Call the factory, passing the group name only if it accepts one.

    Two factories (bar1, matrix) need it -- for the per-group window size.
    The others, and any externally registered factory, still use the
    two-argument form. The signature is inspected rather than catching a
    ``TypeError``: a TypeError raised FROM the factory would look
    identical and would silently be misread as "old form".
    """
    import inspect

    try:
        takes_group = "group" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        takes_group = False
    if takes_group:
        return factory(cpu_group, device, group=group)
    return factory(cpu_group, device)


def _build_transport(name: str, cpu_group, device, disabled: bool,
                     group: str = ""):
    if disabled:
        report_state(group, name, "none (world_size 1)")
        return None
    factory = TRANSPORT_REGISTRY.get(name)
    if factory is None:
        report_state(group, name, "gloo",
                    reason="no such name in TRANSPORT_REGISTRY",
                    stage="selection")
        return None  # "gloo" or an unknown name -> inline data plane
    if _no_fallback(name):
        t = _invoke_factory(factory, cpu_group, device, group)
        report_state(group, name, name)
        return t
    try:
        t = _invoke_factory(factory, cpu_group, device, group)
    except Exception as e:
        stage = getattr(e, "stage", "setup")
        reason = getattr(e, "reason", f"{type(e).__name__}: {e}")
        report_state(group, name, "gloo", reason=reason, stage=stage)
        # WARNING, not INFO, and with the group name. One group failing is
        # not an edge case: it turns any measurement over this run into a
        # mixed one, and that is exactly what happened here unnoticed.
        logger.warning(
            "barlink: group %r does NOT get the requested transport %r "
            "(%s: %s). This group runs over the host-staged gloo layer. "
            "Any measurement over this run is therefore mixed and must "
            "NOT be reported as a %r number.",
            group or "<unnamed>", name, stage, reason, name,
        )
        return None
    if t is None:
        report_state(group, name, "gloo",
                    reason="factory returned None without a reason", stage="setup")
        logger.warning(
            "barlink: group %r does not get the requested transport %r "
            "(the factory returned None). gloo layer.",
            group or "<unnamed>", name,
        )
        return None
    report_state(group, name, name)
    return t


def graph_capture_running() -> bool:
    """``True`` while the CURRENT stream is being captured into a CUDA graph.

    **One** definition, here rather than reimplemented per module: it
    decides at three places (the fallback gate below, kernel variant, and
    direct mode in ``barlink_bar1``), and two versions of the same question
    would be exactly where they drift apart.

    UNIFORM ACROSS RANKS only insofar as the capture itself is -- and in
    sglang it is: the graph runner records the same shapes in the same
    order on every rank. Anyone using this function to make a COLLECTIVE
    decision is relying on exactly that; where it doesn't hold, the
    decision is misplaced.

    Never raises: without an initialized CUDA context,
    ``is_current_stream_capturing`` throws (``cudaErrorNoDevice``), and this
    question must never take down a caller who is only asking it as a
    precaution.
    """
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return False
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _transport_name(t) -> str:
    """The name of a transport for error messages, without ever raising itself.

    ``name()`` is a promise of the transport interface, but a transport
    that is currently being named as the CAUSE of an error message is the
    last one that should be trusted with a call.
    """
    try:
        name_fn = getattr(t, "name", None)
        if callable(name_fn):
            return str(name_fn())
    except Exception:  # noqa: BLE001 - a name must never itself be the cause
        pass
    return type(t).__name__


def _covered_ops(t) -> str:
    """The operations ``t`` actually offers -- straight from THE source.

    Reads ``BARLINK_OPS`` off the transport itself, never a list carried
    along in the error text. A list in the text would be exactly the kind
    of promise that goes stale the next time a collective is added, and
    then the message says "X is missing" while X has long been present and
    something else is actually stuck.
    """
    ops = getattr(t, "BARLINK_OPS", None)
    if not ops:
        return "unknown (the transport declares no BARLINK_OPS)"
    return ", ".join(sorted(str(o) for o in ops))


def _row_bytes(t: torch.Tensor) -> int:
    """Bytes of one row along axis 0. For 1-D that's a single element."""
    n = 1
    for d in t.shape[1:]:
        n *= int(d)
    return n * t.element_size()


def _group_max(value: int, cpu_group, table=None) -> int:
    """Maximum across the group, on the CPU.

    Only meant for the case where the caller brings BOTH split-size lists
    itself: then no rank knows the other pairs' blocks, and the slot
    decision must still come out uniform across ranks. An int64 over gloo
    -- measurably expensive relative to a MoE dispatch, but cheaper than a
    hang, and the evenly-split case doesn't need it at all.
    """
    t = torch.tensor([int(value)], dtype=torch.int64)
    barlink_liveness.bounded_collective(
        lambda: dist.all_reduce(
            t, op=dist.ReduceOp.MAX, group=cpu_group, async_op=True
        ),
        "barlink gloo group_max (all_reduce MAX)",
        table=table,
    )
    return int(t.item())


class BarlinkCommunicator:
    """Host-staged collectives over the group's gloo CPU process group."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        group: str = "",
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(cpu_group)
        self.rank = dist.get_rank(cpu_group)
        self.disabled = self.world_size == 1
        #: Who the peer PROCESSES of this group are. Published once, here,
        #: before any transport exists -- every gloo call below is bounded
        #: against it, so a SIGKILLed rank ends the wait with a named error
        #: instead of the 7200 s gloo process-group timeout. ``None`` when
        #: the feature is switched off; the helpers then degrade to the
        #: exact blocking calls they replaced.
        self._peer_table = barlink_liveness.install(cpu_group)
        #: (Operation, size class) pairs for which the loud fallback
        #: notice has already gone out. Kept per group, because coverage
        #: can differ per group: tp and dcp get differently sized windows,
        #: and what fits in one doesn't have to fit in the other.
        self._fallback_reported: set = set()
        self.transport = _build_transport(
            _TRANSPORT, cpu_group, device, disabled=self.disabled, group=group,
        )
        #: What this group ACTUALLY runs on -- not what was requested.
        #: Nameless means: the entry `_build_transport` just created, i.e.
        #: the most recently inserted one.
        self.state = (
            _STATE.get(group, {}) if group
            else (list(_STATE.values())[-1] if _STATE else {})
        )
        # #279 path dispatcher (skeleton, flag-gated, default None). With an
        # empty registry every decision is status quo, so building it does
        # not change any selection -- see barlink_path_dispatcher.
        from sglang.srt.distributed.device_communicators.barlink_path_dispatcher import (
            maybe_build_dispatcher,
        )

        self._path_dispatcher = None if self.disabled else maybe_build_dispatcher()
        # Dedicated copy stream: D2H of the next chunk overlaps with the
        # CPU-side gloo reduction of the current one.
        self._stream = torch.cuda.Stream(device=device)
        # Pinned staging buffers, grown on demand and reused. Two
        # buffers per direction so chunk i+1 can stage while chunk i is
        # still being reduced/written back.
        self._host_bufs: list[torch.Tensor] = []
        self._host_buf_bytes = 0

    def _select(self, op: str, nbytes: int):
        """The transport for ``op`` at this size, or None for the gloo plane.

        One attribute test plus the transport's own `handles` -- the same shape
        at every dispatch site, so op coverage can no longer differ silently
        between ops the way it did when each site hard-coded its own condition.

        **And this is exactly where the fallback gate comes in.** A
        ``None`` means "gloo layer", and the gloo layer is host-staged:
        pinned allocation, ``dist.*`` on the CPU, ``Event.synchronize()``.
        Inside a CUDA graph capture that is either an abort with an error
        that looks like something else entirely -- or, worse, a CPU
        reduction that runs ONCE at capture time and is missing on every
        replay: wrong numbers with no crash.

        This is not a hypothetical case. A transport like ``bar1`` does not
        cover every operation (``BARLINK_OPS`` in ``barlink_bar1.py``) and
        itself answers ``False`` for a covered one below ``min_bytes``, at
        ``nbytes % 16 != 0``, or above the mapped window. Under capture,
        every one of these cases used to land silently in the loop further
        below.

        Hence: under capture there is no falling back, only an
        announcement with a reason.

        **Two mechanisms, in this order.** The #279 path dispatcher may
        still refine the class choice; the gate then guards the FINAL
        choice:

        1. ``handles`` supplies the class choice (#240).
        2. ``refine_transport_choice`` refines it (#279). Without measured
           rates every decision is status quo, i.e. it returns unchanged --
           the placeholder-neutrality that ``test_barlink_path_dispatcher.py``
           pins down remains exactly intact, because outside a capture
           nothing further happens after step 2.
        3. Only then the gate. The order is not arbitrary: a ``HINT_GLOO``
           can still set the choice back to ``None`` even after ``handles``
           had said yes. Before the dispatcher existed, the gate would have
           let through exactly this one case -- the only one where a
           measured decision leads into the gloo layer under capture.
        """
        t = self.transport
        chosen = t if (t is not None and t.handles(op, nbytes)) else None
        dispatcher = getattr(self, "_path_dispatcher", None)
        if dispatcher is not None:
            # Thin #279 hook onto the existing #240 class choice: status-quo
            # decisions (today: all of them) return `chosen` unchanged.
            from sglang.srt.distributed.device_communicators.barlink_path_dispatcher import (  # noqa: E501
                refine_transport_choice,
            )

            chosen = refine_transport_choice(dispatcher, op, nbytes, chosen)
        if chosen is None and t is not None and not graph_capture_running():
            # THE HONESTY RULE: falling back is fine, doing it silently is not.
            #
            # Outside a capture, the gloo layer is a viable, just slower,
            # path -- so nothing aborts here. But a transport that opts out
            # for a given size while the log says "transport=bar1"
            # invalidates any measurement taken afterwards. That is exactly
            # what happened during prefill: from 2457 tokens per batch
            # onward, bar1 answers False for all_reduce, and with a
            # `chunked_prefill_size` of 4096 or 8192 the direct path would
            # have quietly dropped out there without a single log line.
            #
            # Once per (operation, size class) and group, not per call: in
            # the hot path the same sizes recur thousands of times, and a
            # warning per collective would be a log storm nobody reads. The
            # size class is the base-2 logarithm -- fine enough that a new
            # operating size stands out, coarse enough that noise doesn't
            # produce a new line every time.
            size_class = int(nbytes).bit_length()
            key = (op, size_class)
            # Created lazily, not assumed to exist: the communicator also
            # exists as a `__new__` stand-in (tests, and the path
            # dispatcher builds itself one), and a warning that dies on the
            # missing marker would be a new failure right at the point
            # where one is currently being closed off.
            reported = getattr(self, "_fallback_reported", None)
            if reported is None:
                reported = set()
                self._fallback_reported = reported
            if key not in reported:
                reported.add(key)
                reason = ""
                why_not = getattr(t, "why_not", None)
                if callable(why_not):
                    try:
                        reason = why_not(op, nbytes) or ""
                    except Exception as e:      # noqa: BLE001
                        reason = f"(reason could not be determined: {e!r})"
                logger.warning(
                    "barlink[%s]: %s does NOT cover %r at %d bytes -- falling "
                    "back to the host-staged layer. Covered there: %s.%s "
                    "This message appears once per operation and size "
                    "class; this run's numbers for this size are NOT %s "
                    "numbers.",
                    getattr(self, "group", "?"), _transport_name(t), op,
                    nbytes, _covered_ops(t),
                    f" Reason: {reason}." if reason else "",
                    _transport_name(t),
                )
        if chosen is None and graph_capture_running():
            # The gate sits behind the dispatcher, not in front of it: it
            # guards the FINAL choice. A ``HINT_GLOO`` can still set
            # `chosen` back to None even after `handles` had said yes --
            # and this is exactly the one case that would otherwise fall
            # through ungated into the gloo layer during capture.
            if t is None:
                reason = "no transport is built at all"
            elif t.handles(op, nbytes):
                reason = (
                    f"{_transport_name(t)} can do it, but the path dispatcher "
                    f"decided on the gloo layer"
                )
            else:
                reason = (
                    f"{_transport_name(t)} reports handles({op!r}, {nbytes}) "
                    f"-> False; covered there: {_covered_ops(t)}"
                )
            raise RuntimeError(
                f"barlink: {op!r} with {nbytes} bytes during a CUDA graph "
                f"capture, but {reason}. The fallback path is the "
                f"host-staged gloo layer (pinned allocation, dist.* on the "
                f"CPU, Event.synchronize()) -- which runs ONCE at capture "
                f"time and not at all on replay. That would produce wrong "
                f"numbers without a crash, so this aborts instead. Fix: "
                f"--disable-cuda-graph, or choose a transport that can "
                f"genuinely run this operation at this size "
                f"(SGLANG_BARLINK_TRANSPORT=device covers "
                f"all_reduce/all_gather/reduce_scatter/broadcast "
                f"completely)."
            )
        return chosen

    def _get_out_buf(self, ref: torch.Tensor) -> torch.Tensor:
        """One FRESH output tensor per call — never a shape-keyed cache.

        This used to hand out a persistent per-(shape, dtype) buffer, on the
        theory that a piecewise CUDA graph captured downstream of the
        collective needs the result at a stable address. That reasoning does
        not survive contact with either path that exists:

        * the CPU transports (shm/gloo) synchronize with the host inside the
          collective, so they can never be inside a captured region at all;
        * the one graph-capturable transport (`barlink_device`) does NOT use
          this helper — it allocates with `torch.empty_like` per call, and is
          correct precisely because capture-time allocations come from the
          graph's private pool and therefore already replay at a stable
          address.

        Meanwhile the cache actively BREAKS the documented contract of
        `all_reduce` ("returns a new tensor, out-of-place"): two results of
        the same shape and dtype were the SAME tensor, so the second call
        silently overwrote the first while the model still held it. That is
        not a hypothetical — it corrupted the forward outright (garbage
        tokens, no crash, no hang) on every non-device transport.
        """
        return torch.empty_like(ref)

    def _get_host_bufs(self, nbytes: int, count: int = 2) -> list[torch.Tensor]:
        if self._host_buf_bytes < nbytes or len(self._host_bufs) < count:
            self._host_bufs = [
                torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                for _ in range(count)
            ]
            self._host_buf_bytes = nbytes
        return self._host_bufs

    # ------------------------------------------------------------------
    # all_reduce
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # async all_reduce (issue/wait split; ucx transport only today)
    #
    # supports_async() is deliberately shaped like handles(): its answer
    # depends only on group-uniform state (env-selected transport, class of
    # that transport), never on the payload -- so no rank can decide to go
    # async while a peer goes sync. Callers must treat a None from
    # all_reduce_async as "issue unavailable" and fall back to the sync
    # all_reduce; wait_async must be called exactly once per handle.
    # ------------------------------------------------------------------

    def supports_async(self) -> bool:
        t = self.transport
        return (
            not self.disabled
            and t is not None
            and hasattr(t, "all_reduce_async")
            and t.handles("all_reduce", 0)
        )

    def all_reduce_async(self, input_: torch.Tensor):
        """Issue a sum-all-reduce; returns a handle for wait_async, or None.

        None means the async path is unavailable here (no transport, or the
        transport has no async support) -- the caller runs the sync
        all_reduce instead. That decision is group-uniform by construction
        (see supports_async).
        """
        if not self.supports_async():
            return None
        return self.transport.all_reduce_async(self, input_.contiguous())

    def wait_async(self, handle) -> torch.Tensor:
        """Complete an all_reduce_async handle; returns the fresh result."""
        return self.transport.wait_async(handle)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """Sum-all-reduce ``input_`` across the group, host-staged.

        Returns a new tensor (out-of-place), matching the contract of
        the other vLLM all-reduce backends.
        """
        if self.disabled:
            return input_.clone()
        inp = input_.contiguous()
        nbytes = inp.numel() * inp.element_size()
        t = self._select("all_reduce", nbytes)
        if t is not None:
            return t.barlink_all_reduce(self, inp)
        out = self._get_out_buf(inp)

        reduce_dtype = (
            torch.float32
            if _FP32_REDUCE and inp.dtype in (torch.float16, torch.bfloat16)
            else inp.dtype
        )
        elem_bytes = torch.tensor([], dtype=reduce_dtype).element_size()
        chunk_elems = max(_CHUNK_BYTES // elem_bytes, 1)

        flat_in = inp.view(-1)
        flat_out = out.view(-1)
        n = flat_in.numel()
        n_chunks = (n + chunk_elems - 1) // chunk_elems

        bufs = self._get_host_bufs(min(n, chunk_elems) * elem_bytes)
        staged: list[tuple[int, int, torch.Tensor, torch.cuda.Event]] = []

        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)

        def _stage(ci: int) -> None:
            start = ci * chunk_elems
            end = min(start + chunk_elems, n)
            host = (
                bufs[ci % len(bufs)][: (end - start) * elem_bytes]
                .view(reduce_dtype)[: end - start]
            )
            with torch.cuda.stream(self._stream):
                src = flat_in[start:end]
                if src.dtype != reduce_dtype:
                    src = src.to(reduce_dtype)
                host.copy_(src, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self._stream)
            staged.append((start, end, host, ev))

        _stage(0)
        for ci in range(n_chunks):
            if ci + 1 < n_chunks:
                _stage(ci + 1)  # D2H of next chunk overlaps gloo below
            start, end, host, ev = staged[ci]
            ev.synchronize()
            barlink_liveness.bounded_collective(
                lambda: dist.all_reduce(
                    host, group=self.cpu_group, async_op=True
                ),
                f"barlink gloo all_reduce chunk {ci}/{n_chunks}",
                table=self._peer_table,
            )
            with torch.cuda.stream(self._stream):
                dst = flat_out[start:end]
                if host.dtype != dst.dtype:
                    dst.copy_(host.to(dst.dtype), non_blocking=False)
                else:
                    dst.copy_(host, non_blocking=True)

        current.wait_stream(self._stream)
        return out

    # ------------------------------------------------------------------
    # all_gather / reduce_scatter / broadcast
    # ------------------------------------------------------------------

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.disabled:
            return input_
        t = self._select(
            "all_gather", input_.numel() * input_.element_size()
        )
        if t is not None:
            return t.barlink_all_gather(self, input_, dim)
        if dim < 0:
            dim += input_.dim()
        inp = input_.contiguous()
        input_size = inp.size()

        host_in = torch.empty(
            inp.shape, dtype=inp.dtype, pin_memory=True
        )
        host_in.copy_(inp, non_blocking=False)
        host_out = [torch.empty_like(host_in) for _ in range(self.world_size)]
        barlink_liveness.bounded_collective(
            lambda: dist.all_gather(
                host_out, host_in, group=self.cpu_group, async_op=True
            ),
            "barlink gloo all_gather",
            table=self._peer_table,
        )

        output = torch.empty(
            (self.world_size,) + tuple(input_size),
            dtype=inp.dtype,
            device=inp.device,
        )
        for i, h in enumerate(host_out):
            output[i].copy_(h, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()

        output = output.movedim(0, dim)
        return output.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.disabled:
            return input_
        t = self._select(
            "reduce_scatter", input_.numel() * input_.element_size()
        )
        if t is not None:
            return t.barlink_reduce_scatter(self, input_, dim)
        if dim < 0:
            dim += input_.dim()
        # Host-staged: full all-reduce, then slice this rank's shard.
        # For the small TP world sizes barlink targets (2-4 ranks) the
        # extra traffic vs a true reduce-scatter is bounded and the
        # code stays trivially correct. Axis handling mirrors the base
        # communicator's reduce_scatter exactly.
        reduced = self.all_reduce(input_)
        # movedim(dim, 0) -- NOT movedim(0, dim). The two agree only for dim
        # in {0, 1}; from dim >= 2 they differ, and the old form left the
        # ORIGINAL axis 1 in front, so the scatter sliced the wrong axis while
        # every shape check still passed. Measured: shape (4,6,2) dim=2 sliced
        # 6 instead of 2; (2,4,6,2) dim=3 sliced 4 instead of 2. The signature
        # defaults to dim=-1, so a bare reduce_scatter(x) on ndim >= 3
        # scattered the wrong axis SILENTLY. 2-D happened to be correct, which
        # is why it survived.
        moved = reduced.movedim(dim, 0).contiguous()
        assert moved.shape[0] % self.world_size == 0
        chunk = moved.shape[0] // self.world_size
        shard = moved[self.rank * chunk : (self.rank + 1) * chunk]
        return shard.movedim(0, dim).contiguous()

    # ------------------------------------------------------------------
    # all_to_all_single
    #
    # WHO ACTUALLY CALLS THIS -- checked, not assumed:
    #
    # * `GroupCoordinator.all_to_all_single(output, input)`
    #   (parallel_state.py:1199 -> :1196) is the ONLY torch.distributed a2a
    #   call site in the whole srt tree. It is out-of-place, evenly split,
    #   with no split sizes and no scatter/gather axis -- and today it has
    #   no caller (upstream surface from #27492).
    # * The MoE token dispatchers do NOT call in here. deepep.py:578
    #   `buffer.dispatch(...)`, mooncake.py:236, nixl.py:293, moriep.py:724
    #   and flashinfer.py:259 `moe_a2a.dispatch(...)` bypass
    #   torch.distributed and go straight to their own libraries. Their
    #   semantics are, however, precisely the unevenly split kind: they
    #   pass in `num_tokens_per_rank`/`num_tokens_per_expert`, because the
    #   token count per expert varies.
    #
    # From this follows the signature here: that of `torch.distributed.
    # all_to_all_single`, i.e. the evenly-split form used by the one real
    # caller, PLUS the split sizes, without which MoE could not be
    # represented. Nothing here is guessed -- the evenly-split form is a
    # special case (both lists None) and takes the same path.
    # ------------------------------------------------------------------

    def all_to_all_counts(self, input_split_sizes) -> list[list[int]]:
        """The full R x R count matrix, from each rank's own send counts.

        ``matrix[i][j]`` = what rank i sends to rank j. An
        ``all_gather_object`` over the CPU group -- exactly the step DeepEP
        takes before dispatch (``get_dispatch_layout`` ->
        ``num_tokens_per_rank``), and for the same reason: the receiver
        cannot know its buffer size before the sender has counted.

        This is a **host collective**. It sits in front of the data path,
        not inside it, and it is the reason the unevenly split case is not
        CUDA-graph-capable. The evenly-split case doesn't need it, and is
        capturable precisely because of that.
        """
        matrix: list = [None] * self.world_size
        # ``all_gather_object`` has no ``async_op`` form -- torch runs it
        # inline -- so it cannot be polled. The next best bound is to refuse
        # to enter it when a peer is already provably gone; this sits on the
        # MoE dispatch path, where entering it blind costs 7200 s.
        barlink_liveness.check_peers(
            "barlink gloo all_to_all counts (all_gather_object)",
            table=self._peer_table,
        )
        dist.all_gather_object(
            matrix, [int(x) for x in input_split_sizes], group=self.cpu_group
        )
        return [[int(v) for v in row] for row in matrix]  # type: ignore[union-attr]

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        output_split_sizes=None,
        input_split_sizes=None,
    ) -> torch.Tensor:
        """``torch.distributed.all_to_all_single`` over barlink.

        Splits ``input_`` along axis 0 into ``world_size`` blocks, sends
        block j to rank j, and places the received blocks into ``output``
        in the same order. ``output`` is written in place and returned.

        ``*_split_sizes`` are **row counts**, not bytes -- same as in
        torch. ``None`` means evenly split. If only ``input_split_sizes``
        is given, the receive counts are obtained via
        :meth:`all_to_all_counts`; ``output`` must already be large
        enough in that case.
        """
        if self.disabled:
            output.copy_(input_)
            return output
        inp = input_.contiguous()
        if inp.dim() == 0 or output.dim() == 0:
            raise ValueError("all_to_all_single requires at least one axis")

        row_elems = 1
        for d in inp.shape[1:]:
            row_elems *= int(d)
        row_bytes = row_elems * inp.element_size()
        if row_bytes != _row_bytes(output):
            raise ValueError(
                f"all_to_all_single: row width mismatch -- input "
                f"{row_bytes} bytes, output {_row_bytes(output)} bytes. "
                f"Only axis 0 is split."
            )

        w = self.world_size
        # The counts. Order: fetch what the FALLBACK also needs first --
        # otherwise whether a collective runs at all would hinge on the
        # transport choice, and that is exactly the kind of rank dependency
        # that hangs.
        matrix = None
        if input_split_sizes is None:
            if inp.shape[0] % w:
                raise ValueError(
                    f"all_to_all_single without input_split_sizes requires "
                    f"an axis 0 divisible by {w}, but got {inp.shape[0]}."
                )
            in_rows = [inp.shape[0] // w] * w
        else:
            in_rows = [int(x) for x in input_split_sizes]
            if len(in_rows) != w or sum(in_rows) != inp.shape[0]:
                raise ValueError(
                    f"input_split_sizes {in_rows} does not match axis 0 = "
                    f"{inp.shape[0]} for {w} ranks."
                )
        if output_split_sizes is None:
            if input_split_sizes is None:
                if output.shape[0] % w:
                    raise ValueError(
                        f"all_to_all_single without output_split_sizes "
                        f"requires an axis 0 of the output divisible by "
                        f"{w}, but got {output.shape[0]}."
                    )
                out_rows = [output.shape[0] // w] * w
            else:
                matrix = self.all_to_all_counts(in_rows)
                out_rows = [matrix[i][self.rank] for i in range(w)]
        else:
            out_rows = [int(x) for x in output_split_sizes]
            if len(out_rows) != w:
                raise ValueError(
                    f"output_split_sizes has length {len(out_rows)}"
                )
        if sum(out_rows) > output.shape[0]:
            raise ValueError(
                f"output holds {output.shape[0]} rows, but {sum(out_rows)} "
                f"are being received."
            )

        send_bytes = [n * row_bytes for n in in_rows]
        recv_bytes = [n * row_bytes for n in out_rows]
        nbytes = sum(send_bytes)

        t = self._select("all_to_all", nbytes)
        if t is not None and not (
            hasattr(t, "barlink_all_to_all_single") and hasattr(t, "supports_a2a")
        ):
            # handles() said yes, but the methods are missing. That is a
            # bug IN THE TRANSPORT, not a runtime condition: it fails the
            # same way on every rank, because the class is uniform across
            # ranks. So the fallback is safe -- and the warning names the
            # culprit instead of letting it disappear into the fallback.
            logger.warning(
                "barlink: transport %s reports handles('all_to_all') as yes, "
                "but has no barlink_all_to_all_single/supports_a2a. Running "
                "the CPU layer.", type(t).__name__,
            )
            t = None
        if t is not None:
            # The exact slot check needs the largest block across ALL
            # pairs, not just this rank's own row -- otherwise one rank
            # could say yes and another no, turning this into a hang
            # instead of an error. When evenly split, the maximum is the
            # same number on every rank and needs no collective;
            # otherwise it comes from the count matrix, or, if the caller
            # brought both lists itself, from a maximum across the group.
            if input_split_sizes is None and output_split_sizes is None:
                largest_block = max(send_bytes + recv_bytes)
            elif matrix is not None:
                largest_block = max(max(row) for row in matrix) * row_bytes
            else:
                largest_block = _group_max(
                    max(send_bytes + recv_bytes), self.cpu_group,
                    table=self._peer_table,
                )
            if t.supports_a2a(largest_block):
                # The round count follows from the GROUP-WIDE largest
                # block we just computed -- not from this rank's own row,
                # which differs per rank. A transport without this
                # information runs a single round, as before.
                rounds_for = getattr(t, "a2a_rounds_for", None)
                rounds = (
                    rounds_for(largest_block) if callable(rounds_for) else None
                )
                return t.barlink_all_to_all_single(
                    self, output, inp, send_bytes, recv_bytes, rounds=rounds
                )

        # Fallback: the same decomposition over the CPU group. Pinned, so
        # the two copies don't go through a pageable intermediate buffer.
        host_in = torch.empty(inp.shape, dtype=inp.dtype, pin_memory=True)
        host_in.copy_(inp, non_blocking=False)
        host_out = torch.empty(
            (sum(out_rows),) + tuple(output.shape[1:]),
            dtype=output.dtype, pin_memory=True,
        )
        try:
            barlink_liveness.bounded_collective(
                lambda: dist.all_to_all_single(
                    host_out, host_in,
                    output_split_sizes=out_rows, input_split_sizes=in_rows,
                    group=self.cpu_group, async_op=True,
                ),
                "barlink gloo all_to_all_single",
                table=self._peer_table,
            )
        except barlink_liveness.PeerLivenessError:
            # A dead peer or an expired deadline is not "this layer cannot
            # run the call" -- it already names the cause. Rewrapping it as
            # NotImplementedError below would bury that.
            raise
        except (RuntimeError, NotImplementedError) as e:
            raise NotImplementedError(
                f"all_to_all_single: neither the BAR1 direct path nor the "
                f"group's CPU layer can run this call ({e}). No silent "
                f"fallback to NCCL: on a group spanning two vendors, that "
                f"is not a slower path, it is a hang."
            ) from e
        output[: sum(out_rows)].copy_(host_out, non_blocking=False)
        return output

    # ------------------------------------------------------------------
    # out-parameter forms
    #
    # sglang calls these directly (they are NOT reachable from the dim-based
    # variants above), so leaving them out would silently route part of the
    # traffic back to NCCL -- which on a mixed-vendor group is not a slow
    # path but a hang. Both are pure compositions of the collectives above:
    # they introduce NO new collective, which keeps the rank-uniformity
    # argument unchanged.
    # ------------------------------------------------------------------

    def all_gather_into_tensor(
        self, output: torch.Tensor, input_: torch.Tensor
    ) -> None:
        """`output[i*n:(i+1)*n] = input_` of rank i, matching
        torch.distributed.all_gather_into_tensor."""
        if self.disabled:
            output.copy_(input_)
            return
        # dim=0 concatenation IS the [world, n]-flattened layout this API
        # specifies, so no extra transposition is needed.
        gathered = self.all_gather(input_, dim=0)
        output.copy_(gathered.reshape(output.shape))

    def reduce_scatter_tensor(
        self, output: torch.Tensor, input_: torch.Tensor
    ) -> None:
        """Sum-reduce `input_` and scatter along dim 0 into `output`,
        matching torch.distributed.reduce_scatter_tensor."""
        if self.disabled:
            output.copy_(input_)
            return
        shard = self.reduce_scatter(input_, dim=0)
        output.copy_(shard.reshape(output.shape))

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.disabled:
            return tensor
        # A transport that can broadcast on-device does so: the host-staged
        # path below synchronizes with the host and is therefore ILLEGAL
        # inside a CUDA-graph capture, which the speculative draft-pick sync
        # performs. See BarlinkDeviceTransport.barlink_broadcast.
        t = self._select("broadcast", tensor.numel() * tensor.element_size())
        if t is not None:
            return t.barlink_broadcast(self, tensor, src)
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        if self.rank == src:
            host.copy_(tensor, non_blocking=False)
        barlink_liveness.bounded_collective(
            lambda: dist.broadcast(
                host, src=dist.get_global_rank(self.cpu_group, src),
                group=self.cpu_group, async_op=True,
            ),
            f"barlink gloo broadcast (src rank {src})",
            table=self._peer_table,
        )
        if self.rank != src:
            tensor.copy_(host, non_blocking=False)
        return tensor

    # ------------------------------------------------------------------
    # teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the POSIX shm segment backing the shm/device transports.

        Without this the segment survives the process and leaks a /dev/shm
        entry per run (rank 0 owns the unlink). Called from
        GroupCoordinator.destroy().
        """
        if self.transport is None:
            return
        try:
            self.transport.close()
        except Exception as e:  # teardown must never mask the real error
            logger.warning("barlink: transport close failed (%s).", e)
