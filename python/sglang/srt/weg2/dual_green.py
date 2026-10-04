"""DUAL-SHARE STAGE 3: the green-context ladder of P, dynamic and stepwise (item 1330, user order 04.10.).

User order 04.10.: P's SM share must go DYNAMICALLY and STEPWISE, up AND down -- P = 100 / 75 / 50 / 25 /
0 % -- and may bite only while D really runs (D empty = P at 100 % at once). Today the throttle is
static (``--dual-p-sm-pct``: ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` of P's process, read when the client
starts; P chunk 1024 at 25 % = 536 ms against 194 ms at 100 %, also with D empty -- boots 092421 vs 071053).

WHAT THIS MODULE IS (desk/1330-p-drossel-dynamisch.md, section 8 = the specification):

* the CUDA side behind a thin interface (:class:`GreenBackend`): :class:`CtypesBackend` binds
  ``cuDeviceGetDevResource`` / ``cuDevSmResourceSplitByCount`` / ``cuDevResourceGenerateDesc`` /
  ``cuGreenCtxCreate`` / ``cuGreenCtxStreamCreate`` through ctypes (the binding of the metal probe
  ``greenctx_mps_probe``, which ran on all three cards on 03.10. 19:53Z under MPS: the mask bites with
  0.79-0.93 of the SM ratio, a stage change costs microseconds). Every SM number is read from the driver
  (granularity, rounding), nothing here knows a card. The tests drive the same code with a fake backend.
* :class:`GreenLadder`: one green context + stream per rung below 100 %, built once at init, probed
  (a timed GEMM per rung: the mask must bite), never destroyed (a graph after ``cuGreenCtxDestroy`` fails
  with CONTEXT_IS_DESTROYED). Rung 0 is the primary context's ``forward_stream`` (no green context).
* the FRONT side (:class:`GreenController`): the 5-stage automaton of section 8.2 -- stages 0..3 = 100/75/50/
  25 %, H = hold (0 %) which is OBSERVER-ONLY by default. Entry stage from the open loop (section 8.3:
  the more D bs, the DEEPER the stage -- the budget per request is fixed and D solo grows with bs),
  then the closed loop on the measured D round against the target (descend at once, ascend after calm).
* the P side (:class:`GreenActuator`): PP0 reads the front's ctl file, stamps ONE
  :class:`Weg2DualGreenRung` on the request wire when the stage changes (and as a heartbeat), every follower
  relays and absorbs it; every stage then launches each forward on the rung's green stream.

RANK UNIFORMITY (RAENGE-NIE-UNEINS) -- why PP0 stamps the wire and followers never read the ctl file:
the stage must be the same on all three P stages in the same logical pass. A mask changes no shape, no
extent, no collective, so a rank that read the file 20 ms later would not break the pipeline -- but it
would run one pass at another speed than its neighbours, and "which stage ran this chunk on PP1" would be
unanswerable from the logs. The wire gives the stronger property for the price of one tuple: PP0's decision
rides list m exactly like PP0's pass clock (``anchor_tails.Weg2BurstClock``) and the V2 arena-trim order
(``dual_arena_spill.Weg2DualArenaTrim``); a follower's pass m consumes PP0's list m, so every stage applies the
SAME stage at the SAME logical pass, from ONE reader (PP0's :class:`CtlReader`). Followers read no ctl file,
no clock, no env threshold of their own. The sequence number is logged with every switch and every
``Prefill rank batch`` line carries ``rung=``, so uniformity is checkable from the three logs afterwards.
Each stage maps the stage's FRACTION to its OWN card's ladder (5090: 8-SM granularity, 3080: 2-SM): the
fraction is the shared quantity, the SM count is local.

STAGE CHANGES ONLY AT FORWARD BOUNDARIES: ``apply`` only records the wanted stage; :meth:`GreenActuator.pick`
(called in ``_pp_launch_batch``, once per forward) is the only place the active stage changes, and a forward
already launched keeps its stream.

GRAPHS (decision, with the reading of the capture paths): CUDA graphs take the SM resources of the
execution context their nodes were CAPTURED in (Programming Guide 4.6), not those of the stream they are
launched on. P's prefill graph (bucket 512) is captured in the primary context and would run unbounded on a
narrow stream. Capturing one graph set per rung would cost VRAM P does not have (activation reserve 0.41 GiB)
and the graph advantage is gone in narrow stages anyway: the host floor of an eager 512 chunk (63/42/74 ms per
stage, 760 4.3) is below its compute time at 75 % already. So below 100 % the prefill graph runner answers
``can_run_graph`` False (:func:`force_eager`) and the chunk runs eager on the rung's green stream.

0 % = HOLD. A green context needs at least ``minSmPartitionSize`` SM (8 on sm120, 2 on sm86): 0 SM is no
stage. The hold is PP0 not launching a chunk (:class:`HoldGate`, PP0-local: followers idle through the
pipeline's own backpressure and are never held alone). It is OBSERVER-ONLY by default: the log carries
``would_hold`` and nothing holds until ``--dual-green-ladder hold``. Its inputs are the arena fill PP0 reads
(``arena_pinned / slots``) -- NOT D's ``full token usage`` (p50 0.96 at running-req 1 in boot 092421, the
gate would stop P continuously). A hold is bounded (``HOLD_MAX_S``), cooled down, released at once when D is
empty, and exempt from the starvation clamp (an old waiting request ends it).

Everything is behind the dual gate (``dual_p_kv_stage.armed``) AND its own switch
(``SGLANG_WEG2_DUAL_GREEN_LADDER=1``, set by the launcher from ``--dual-green-ladder``, default off): flip,
NF and 27B-INT8 never import a symbol of this module's runtime path, with the switch off the launcher's
argv/env, the wire and the launch path are those of the base. Fallback = the existing duty/chunk actuators
(named ``W-DUAL-SHARE-FALLBACK mech=green ...``), never a crash.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from sglang.srt.weg2 import dual_share as _ds

logger = logging.getLogger(__name__)

MARK = "DUAL-GREEN"
LADDER_ENV = _ds.GREEN_LADDER_ENV
#: "1" = a hold really holds PP0 (default: observer, the log carries ``would_hold``)
HOLD_ENV = "SGLANG_WEG2_DUAL_GREEN_HOLD"
HOLD_HI_ENV = "SGLANG_WEG2_DUAL_GREEN_HOLD_ARENA_HI"
HOLD_LO_ENV = "SGLANG_WEG2_DUAL_GREEN_HOLD_ARENA_LO"
HOLD_MAX_S_ENV = "SGLANG_WEG2_DUAL_GREEN_HOLD_MAX_S"
HOLD_COOLDOWN_S_ENV = "SGLANG_WEG2_DUAL_GREEN_HOLD_COOLDOWN_S"
#: "0" = no boot probe of the masks (default 1)
PROBE_ENV = "SGLANG_WEG2_DUAL_GREEN_PROBE"
#: a mask must produce at least this fraction of the SLOWDOWN its SM ratio predicts. A mask that does not bite
#: gives ~0 (noise <= 0.1); the metal (03.10.) showed 0.65-0.9 at effect 0.79-0.93 -- the criterion is steepest
#: on the 75 % rung (SM ratio 1.31-1.33: an effect of 0.8 would read only 0.15 there), so 0.3 sits above the
#: noise and below every measured rung. The 0.8-effect reading is logged next to it, never gated on.
PROBE_MIN_SLOWDOWN_FRAC_ENV = "SGLANG_WEG2_DUAL_GREEN_PROBE_MIN_FRAC"

HOLD_ARENA_HI_DEFAULT = 0.97
HOLD_ARENA_LO_DEFAULT = 0.90
HOLD_MAX_S_DEFAULT = 2.0
HOLD_COOLDOWN_S_DEFAULT = 1.0
#: PP0 re-stamps the current stage at least this often (an idempotent re-assert, new seq)
HEARTBEAT_S = 1.0
#: PP0 reads the arena header at most this often for the hold gate / the observer
ARENA_READ_S = 1.0

#: front-side knobs, ``SGLANG_WEG2_DUAL_SHARE_GREEN_<NAME>`` (``dual_share.ENV_PREFIX``)
DEFAULT_FACTORS = (4.5, 2.8, 1.5, 1.24)      # s_k: D round / D solo at stage k (rung 0: real y9d3 4.5-9x; 1-3: 5090 proxy)
DEFAULT_TSOLO_MS = ((1, 28.0), (3, 40.0), (6, 57.0))   # INT8-y9e D solo round by bs (section 8.3)
DEFAULT_ACCEPT_LEN = 2.5


# ----------------------------------------------------------------------------- module state (P process)

_STATE: Dict[str, Any] = {"eager": False, "serving": False, "actuator": None}


def force_eager() -> bool:
    """The prefill graph runner's question: must this forward run eager? True while the active stage is a
    green stream (graphs would ignore its mask). False in every process that never armed the ladder."""
    return bool(_STATE["eager"])


def _reset_state_for_tests() -> None:
    _STATE.update(eager=False, serving=False, actuator=None)
    _ds.set_green_serves(None)


# ----------------------------------------------------------------------------- gate / env

def ladder_switch(env: Optional[Mapping[str, str]] = None) -> bool:
    e = os.environ if env is None else env
    return str(e.get(LADDER_ENV, "") or "").strip() == "1"


def armed(env: Optional[Mapping[str, str]] = None) -> bool:
    """P side: the dual P gate (``dual_p_kv_stage.armed``) AND the switch AND the ``green`` actuator."""
    e = os.environ if env is None else env
    if not ladder_switch(e):
        return False
    from sglang.srt.weg2 import dual_p_kv_stage as _pk

    if not _pk.armed(e):
        return False
    try:
        acts = _ds.parse_actuators(str(e.get(_ds.ACT_ENV, "") or "chunk"))
    except ValueError:
        return False
    return "green" in acts and bool(str(e.get(_ds.CTL_ENV, "") or "").strip())


def hold_armed(env: Optional[Mapping[str, str]] = None) -> bool:
    e = os.environ if env is None else env
    return str(e.get(HOLD_ENV, "") or "").strip() == "1"


def _envf(env: Mapping[str, str], key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(str(env.get(key, "") or "").strip() or default)
    except ValueError:
        v = default
    return min(max(v, lo), hi)


# ----------------------------------------------------------------------------- the CUDA interface

class GreenError(RuntimeError):
    """A driver / probe failure with its NAME (never a bare traceback in the fallback line)."""


@dataclass(frozen=True)
class DevInfo:
    ordinal: int
    name: str
    arch: str
    uuid: str
    sm_total: int
    granularity: int          # max(minSmPartitionSize, smCoscheduledAlignment); smallest split group as fallback
    driver: str = ""


class GreenBackend:
    """What :class:`GreenLadder` needs from CUDA. Real: :class:`CtypesBackend`. Tests: a fake."""

    def info(self) -> DevInfo:
        raise NotImplementedError

    def split_sizes(self, want: int) -> List[int]:
        """Group sizes of ``cuDevSmResourceSplitByCount(minCount=want)``; [] when the driver gives none."""
        raise NotImplementedError

    def create(self, want: int) -> Tuple[int, Any]:
        """One green context of >= ``want`` SM with its stream: (sm_real, stream). Raises :class:`GreenError`."""
        raise NotImplementedError

    def time_stream(self, stream: Any) -> float:
        """Median ms of a fixed compute load on ``stream`` (None = the primary context's current stream)."""
        raise NotImplementedError


class _SmRes(ctypes.Structure):
    # CUDA 12.x: smCount only; 13.x adds minSmPartitionSize, smCoscheduledAlignment, flags
    _fields_ = [("smCount", ctypes.c_uint), ("minSmPartitionSize", ctypes.c_uint),
                ("smCoscheduledAlignment", ctypes.c_uint), ("flags", ctypes.c_uint)]


class _ResUnion(ctypes.Union):
    _fields_ = [("sm", _SmRes), ("_oversize", ctypes.c_ubyte * 48)]


class _DevResource(ctypes.Structure):
    """cuda.h: type, 92 bytes of internal padding, 48-byte union (the shape of the metal probes)."""

    _fields_ = [("type", ctypes.c_int), ("_internal_padding", ctypes.c_ubyte * 92), ("u", _ResUnion)]


_CU_DEV_RESOURCE_TYPE_SM = 1
_CU_GREEN_CTX_DEFAULT_STREAM = 0x1
_CU_STREAM_NON_BLOCKING = 0x1
_CU_ATTR_SM_COUNT = 16
_CU_ATTR_CC_MAJOR = 75
_CU_ATTR_CC_MINOR = 76


class CtypesBackend(GreenBackend):
    """libcuda + torch. Needs the process's primary context (torch has made it by the time a scheduler exists)."""

    def __init__(self, ordinal: Optional[int] = None):
        import torch

        self._torch = torch
        self.ordinal = int(torch.cuda.current_device() if ordinal is None else ordinal)
        self.lib = ctypes.CDLL("libcuda.so.1")
        v = ctypes.c_void_p
        need = {
            "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            "cuDeviceGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
            "cuDriverGetVersion": [ctypes.POINTER(ctypes.c_int)],
            "cuDeviceGetDevResource": [ctypes.c_int, ctypes.POINTER(_DevResource), ctypes.c_int],
            "cuDevSmResourceSplitByCount": [ctypes.POINTER(_DevResource), ctypes.POINTER(ctypes.c_uint),
                                            ctypes.POINTER(_DevResource), ctypes.POINTER(_DevResource),
                                            ctypes.c_uint, ctypes.c_uint],
            "cuDevResourceGenerateDesc": [ctypes.POINTER(v), ctypes.POINTER(_DevResource), ctypes.c_uint],
            "cuGreenCtxCreate": [ctypes.POINTER(v), v, ctypes.c_int, ctypes.c_uint],
            "cuGreenCtxStreamCreate": [ctypes.POINTER(v), v, ctypes.c_uint, ctypes.c_int],
        }
        missing = []
        for name, argtypes in need.items():
            try:
                fn = getattr(self.lib, name)
            except AttributeError:
                missing.append(name)
                continue
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
        if missing:
            raise GreenError(f"driver lacks the green-context API: {missing}")
        self.lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.cuGetErrorName.restype = ctypes.c_int
        dev = ctypes.c_int()
        self._chk(self.lib.cuDeviceGet(ctypes.byref(dev), self.ordinal), "cuDeviceGet")
        self._dev = dev.value
        self._total = _DevResource()
        self._chk(self.lib.cuDeviceGetDevResource(self._dev, ctypes.byref(self._total), _CU_DEV_RESOURCE_TYPE_SM),
                  "cuDeviceGetDevResource")
        self._groups: Dict[int, Any] = {}
        self._keep: List[Any] = []       # green contexts live to process end (never destroyed)

    def _err(self, code: int) -> str:
        name = ctypes.c_char_p()
        if self.lib.cuGetErrorName(code, ctypes.byref(name)) == 0 and name.value:
            return name.value.decode()
        return f"CUDA_ERROR_{code}"

    def _chk(self, code: int, what: str) -> None:
        if code != 0:
            raise GreenError(f"{what} -> {self._err(code)}")

    def _attr(self, a: int) -> int:
        v = ctypes.c_int()
        self.lib.cuDeviceGetAttribute(ctypes.byref(v), a, self._dev)
        return int(v.value)

    def info(self) -> DevInfo:
        torch = self._torch
        props = torch.cuda.get_device_properties(self.ordinal)
        sm = self._total.u.sm
        gran = max(int(sm.minSmPartitionSize), int(sm.smCoscheduledAlignment))
        if gran <= 0:
            sizes = self.split_sizes(1)
            gran = sizes[0] if sizes else 1
        drv = ctypes.c_int()
        self.lib.cuDriverGetVersion(ctypes.byref(drv))
        return DevInfo(ordinal=self.ordinal, name=str(props.name),
                       arch=f"sm{self._attr(_CU_ATTR_CC_MAJOR)}{self._attr(_CU_ATTR_CC_MINOR)}",
                       uuid=str(getattr(props, "uuid", self.ordinal)), sm_total=int(sm.smCount),
                       granularity=gran, driver=f"cuda{int(drv.value)}")

    def _split(self, want: int) -> Tuple[Any, List[int]]:
        nb = ctypes.c_uint(0)
        rem = _DevResource()
        rc = self.lib.cuDevSmResourceSplitByCount(None, ctypes.byref(nb), ctypes.byref(self._total),
                                                  ctypes.byref(rem), 0, ctypes.c_uint(int(want)))
        if rc != 0 or nb.value == 0:
            return None, []
        n = int(nb.value)
        groups = (_DevResource * n)()
        nb2 = ctypes.c_uint(n)
        rc = self.lib.cuDevSmResourceSplitByCount(groups, ctypes.byref(nb2), ctypes.byref(self._total),
                                                  ctypes.byref(rem), 0, ctypes.c_uint(int(want)))
        if rc != 0:
            return None, []
        return groups, [int(groups[i].u.sm.smCount) for i in range(int(nb2.value))]

    def split_sizes(self, want: int) -> List[int]:
        return self._split(want)[1]

    def create(self, want: int) -> Tuple[int, Any]:
        groups, sizes = self._split(want)
        if not sizes:
            raise GreenError(f"cuDevSmResourceSplitByCount(min={want}) gave no group")
        desc = ctypes.c_void_p()
        self._chk(self.lib.cuDevResourceGenerateDesc(ctypes.byref(desc), ctypes.byref(groups[0]), 1),
                  "cuDevResourceGenerateDesc")
        gctx = ctypes.c_void_p()
        self._chk(self.lib.cuGreenCtxCreate(ctypes.byref(gctx), desc, self._dev, _CU_GREEN_CTX_DEFAULT_STREAM),
                  "cuGreenCtxCreate")
        raw = ctypes.c_void_p()
        self._chk(self.lib.cuGreenCtxStreamCreate(ctypes.byref(raw), gctx, _CU_STREAM_NON_BLOCKING, 0),
                  "cuGreenCtxStreamCreate")
        self._keep.append((gctx, raw))
        stream = self._torch.cuda.ExternalStream(int(raw.value), device=self._torch.device(f"cuda:{self.ordinal}"))
        return int(sizes[0]), stream

    def time_stream(self, stream: Any, reps: int = 16, m: int = 2048, k: int = 4096, n: int = 4096) -> float:
        import contextlib
        import statistics

        torch = self._torch
        dev = torch.device(f"cuda:{self.ordinal}")
        a = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
        b = torch.randn(k, n, dtype=torch.bfloat16, device=dev)
        c = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
        ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        out = []
        with ctx:
            for r in range(4):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(reps):
                    torch.matmul(a, b, out=c)
                e1.record()
                e1.synchronize()
                if r > 0:                        # round 0 is the warm-up
                    out.append(e0.elapsed_time(e1))
        del a, b, c
        return float(statistics.median(out))


# ----------------------------------------------------------------------------- the ladder

@dataclass
class Rung:
    index: int
    fraction: float
    want: int
    sm: int
    stream: Any = None
    timed_ratio: Optional[float] = None     # t_rung / t_full from the boot probe
    effect: Optional[float] = None          # time ratio / SM ratio (the 0.8 reading of the metal probe)


def plan_rungs(info: DevInfo, fractions: Sequence[float], split_sizes: Callable[[int], List[int]]
               ) -> List[Tuple[int, float, int, int, Optional[str]]]:
    """Per rung index: (index, fraction, want, sm_real, error). Rung 0 (f >= 1) is the primary context and
    absent here. The driver's own group size is the truth (it rounds UP to the granularity); when it gives
    nothing for ``want`` the search steps down one granule at a time (the metal probe's ``plan_ladder``)."""
    gran = max(1, int(info.granularity or 1))
    out = []
    for i, f in enumerate(fractions):
        if f >= 1.0:
            continue
        want0 = max(gran, int(round(f * info.sm_total)))
        want, sizes = want0, []
        for _ in range(64):
            sizes = split_sizes(want)
            if sizes:
                break
            want -= gran
            if want < 1:
                break
        if not sizes:
            out.append((i, f, want0, 0, "driver gives no group"))
        else:
            out.append((i, f, want0, int(sizes[0]), None))
    return out


class GreenLadder:
    """One green context + stream per rung below 100 %. ``build`` never raises: a rung that cannot be built
    or whose mask does not bite is named (``W-DUAL-SHARE-FALLBACK mech=green``) and served by the fallback
    actuator; the counters go into the marker line."""

    def __init__(self, backend: GreenBackend, fractions: Sequence[float], *, probe: bool = True,
                 min_slowdown_frac: float = 0.3, log: Callable[[str], None] = logger.info,
                 warn: Callable[[str], None] = logger.warning, rank: int = 0):
        self.backend = backend
        self.fractions = tuple(float(f) for f in fractions)
        self.probe = bool(probe)
        self.min_frac = float(min_slowdown_frac)
        self._log, self._warn = log, warn
        self.rank = int(rank)
        self.rungs: Dict[int, Rung] = {}
        self.failed: Dict[int, str] = {}
        self.info: Optional[DevInfo] = None
        self.counters = {"created": 0, "create_failed": 0, "probe_rejected": 0, "probe_run": 0}

    @property
    def healthy_count(self) -> int:
        return len(self.rungs)

    def build(self) -> "GreenLadder":
        card = "?"
        try:
            info = self.info = self.backend.info()
            card = info.uuid
            plan = plan_rungs(info, self.fractions, self.backend.split_sizes)
        except Exception as e:  # noqa: BLE001 - the driver may lack the API; the rank must still boot
            self._warn(_ds.fallback_line("green", card, f"{type(e).__name__}: {e} -> duty/chunk (ladder not built)"))
            for i, f in enumerate(self.fractions):
                if f < 1.0:
                    self.failed[i] = f"{type(e).__name__}"
            self.counters["create_failed"] += len(self.failed)
            return self
        made: Dict[int, Rung] = {}                 # sm_real -> rung (two fractions rounding to one group share it)
        for i, f, want, sm, err in plan:
            if err:
                self.failed[i] = err
                self.counters["create_failed"] += 1
                self._warn(_ds.fallback_line("green", info.uuid, f"rung {i} f={f:.2f}: {err} -> served by duty/chunk"))
                continue
            if sm in made:
                self.rungs[i] = replace(made[sm], index=i, fraction=f)
                continue
            try:
                sm_real, stream = self.backend.create(want)
            except Exception as e:  # noqa: BLE001 - cuGreenCtxCreate under MPS may fail on a driver
                self.failed[i] = f"{type(e).__name__}: {e}"
                self.counters["create_failed"] += 1
                self._warn(_ds.fallback_line("green", info.uuid, f"rung {i} f={f:.2f} not created ({e}) -> "
                                             "served by duty/chunk"))
                continue
            r = Rung(index=i, fraction=f, want=want, sm=int(sm_real), stream=stream)
            self.counters["created"] += 1
            made[int(sm_real)] = r
            self.rungs[i] = r
        if self.probe and self.rungs:
            self._probe_all()
        return self

    def _probe_all(self) -> None:
        info = self.info
        try:
            t_full = float(self.backend.time_stream(None))
        except Exception as e:  # noqa: BLE001
            self._warn(_ds.fallback_line("green", info.uuid, f"probe baseline failed ({e}) -> rungs kept unprobed"))
            return
        done: Dict[int, Tuple[float, float, bool]] = {}
        for i in sorted(self.rungs):
            r = self.rungs[i]
            if r.sm in done:
                ratio, eff, ok = done[r.sm]
            else:
                try:
                    t = float(self.backend.time_stream(r.stream))
                except Exception as e:  # noqa: BLE001
                    self._reject(i, f"probe failed ({e})")
                    continue
                self.counters["probe_run"] += 1
                ratio = t / t_full if t_full > 0 else 1.0
                sm_ratio = info.sm_total / max(1, r.sm)
                eff = ratio / sm_ratio if sm_ratio > 0 else 1.0
                predicted = sm_ratio - 1.0
                # a rung that rounded up to the whole card has no mask to bite
                ok = True if predicted <= 1e-9 else (ratio - 1.0) >= self.min_frac * predicted
                done[r.sm] = (ratio, eff, ok)
            r.timed_ratio, r.effect = ratio, eff
            if not ok:
                self._reject(i, f"mask does not bite: time x{ratio:.2f} for {r.sm}/{info.sm_total} SM "
                                f"(effect {eff:.2f}; need >= {self.min_frac:.2f} of the predicted slowdown)")

    def _reject(self, i: int, why: str) -> None:
        r = self.rungs.pop(i, None)
        self.failed[i] = why
        self.counters["probe_rejected"] += 1
        self._warn(_ds.fallback_line("green", self.info.uuid if self.info else "?",
                                     f"rung {i} f={(r.fraction if r else 0):.2f} {why} -> served by duty/chunk"))

    def index_for(self, fraction: float) -> int:
        """The rung index whose fraction is nearest to the wanted one (the fraction is the shared quantity)."""
        return min(range(len(self.fractions)), key=lambda k: abs(self.fractions[k] - float(fraction)))

    def entry(self, fraction: float) -> Optional[Rung]:
        """The healthy rung for ``fraction``; None = served by the primary stream (rung 0) or by the fallback."""
        return self.rungs.get(self.index_for(fraction))

    def serves(self, fraction: float) -> bool:
        """True when this fraction is served here: rung 0 (nothing to throttle) or a healthy green rung."""
        if float(fraction) >= 1.0 - 1e-9:
            return True
        return self.entry(fraction) is not None

    def marker(self) -> str:
        info = self.info
        sm = ",".join(f"{self.fractions[i]:.2f}:{r.sm}" for i, r in sorted(self.rungs.items()))
        bad = ",".join(f"{i}" for i in sorted(self.failed))
        return (f"{MARK} P ladder pp={self.rank} card={info.uuid if info else '?'} "
                f"{(info.name + ' ' + info.arch) if info else '?'} sm_total={info.sm_total if info else '?'} "
                f"granularity={info.granularity if info else '?'} driver={info.driver if info else '?'} "
                f"rungs=[{sm}] failed_rungs=[{bad}] created={self.counters['created']} "
                f"create_failed={self.counters['create_failed']} probe_rejected={self.counters['probe_rejected']} "
                f"probe_run={self.counters['probe_run']} (rung 0 = primary stream, hold = PP0 not launching)")


# ----------------------------------------------------------------------------- the wire

class Weg2DualGreenRung(NamedTuple):
    """PP0's stage for the logical pass, riding the request wire (rank uniformity: see the module docstring).
    ``f_ppm`` = the fraction in millionths; each stage maps it onto its own card's ladder."""

    seq: int
    rung: int
    f_ppm: int


def without_green_rung(recv_reqs: Sequence[Any]) -> List[Any]:
    """The list as the request trace should see it (a stage order is not a request)."""
    return [r for r in (recv_reqs or ()) if not isinstance(r, Weg2DualGreenRung)]


# ----------------------------------------------------------------------------- hold gate (PP0-local)

class HoldVerdict(NamedTuple):
    would_hold: bool
    hold: bool
    reason: str


class HoldGate:
    """The 0 % stage. ``would_hold`` is the condition (arena fill with hysteresis, only while D runs);
    ``hold`` is the actuation: armed, not exempt (starving), at most ``max_s`` per episode, ``cooldown_s``
    between episodes, released at once when D is empty."""

    def __init__(self, *, hi: float = HOLD_ARENA_HI_DEFAULT, lo: float = HOLD_ARENA_LO_DEFAULT,
                 max_s: float = HOLD_MAX_S_DEFAULT, cooldown_s: float = HOLD_COOLDOWN_S_DEFAULT,
                 actuate: bool = False, clock: Callable[[], float] = time.monotonic):
        lo = min(lo, hi - 0.01)
        self.hi, self.lo, self.max_s, self.cooldown_s, self.actuate = float(hi), float(lo), float(max_s), \
            float(cooldown_s), bool(actuate)
        self._clock = clock
        self.would = False
        self.holding = False
        self._hold_t0 = 0.0
        self._cool_until = 0.0
        self.episodes = 0
        self.observed_episodes = 0
        self.hold_s = 0.0
        self.capped = 0

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, clock=time.monotonic) -> "HoldGate":
        e = os.environ if env is None else env
        return cls(hi=_envf(e, HOLD_HI_ENV, HOLD_ARENA_HI_DEFAULT, 0.10, 1.0),
                   lo=_envf(e, HOLD_LO_ENV, HOLD_ARENA_LO_DEFAULT, 0.05, 1.0),
                   max_s=_envf(e, HOLD_MAX_S_ENV, HOLD_MAX_S_DEFAULT, 0.0, 60.0),
                   cooldown_s=_envf(e, HOLD_COOLDOWN_S_ENV, HOLD_COOLDOWN_S_DEFAULT, 0.0, 600.0),
                   actuate=hold_armed(e), clock=clock)

    def update(self, arena: Optional[float], d_busy: bool, starve: bool) -> HoldVerdict:
        now = self._clock()
        if not d_busy:                                     # D empty: P at 100 % at once, hold or not
            if self.holding:
                self.hold_s += now - self._hold_t0
            self.would = self.holding = False
            return HoldVerdict(False, False, "d_idle")
        if arena is None:
            would = self.would
        elif self.would:
            would = arena > self.lo                        # hysteresis: ends at LO
        else:
            would = arena >= self.hi
        if would and not self.would:
            self.observed_episodes += 1
        self.would = would
        if not would:
            if self.holding:
                self.hold_s += now - self._hold_t0
            self.holding = False
            return HoldVerdict(False, False, "below")
        if not self.actuate:
            return HoldVerdict(True, False, "observer")
        if starve:                                         # the starvation clamp wins over a hold
            if self.holding:
                self.hold_s += now - self._hold_t0
            self.holding = False
            return HoldVerdict(True, False, "starve_exception")
        if self.holding:
            if now - self._hold_t0 >= self.max_s:
                self.hold_s += now - self._hold_t0
                self.holding = False
                self.capped += 1
                self._cool_until = now + self.cooldown_s
                return HoldVerdict(True, False, "hold_max")
            return HoldVerdict(True, True, "hold")
        if now < self._cool_until:
            return HoldVerdict(True, False, "cooldown")
        self.holding = True
        self._hold_t0 = now
        self.episodes += 1
        return HoldVerdict(True, True, "hold")


# ----------------------------------------------------------------------------- P side: observation files

def pobs_path(ctl: str, rank: int) -> str:
    return f"{ctl}.pobs{int(rank)}"


class PObsWriter:
    """Every P stage tells the front what only it knows (its card's real SM count of the active stage; PP0 also
    the arena fill and the hold verdict) -- one tiny file per stage, atomic rename, on change or once a second."""

    def __init__(self, ctl: str, rank: int, clock: Callable[[], float] = time.monotonic):
        self.path = pobs_path(ctl, rank)
        self.rank = int(rank)
        self._clock = clock
        self._last_key = None
        self._last_t = -1e9
        self.errors = 0

    def update(self, **kv: Any) -> bool:
        key = tuple(sorted((k, v) for k, v in kv.items() if k not in ("arena_ppm", "seq")))
        now = self._clock()
        if key == self._last_key and now - self._last_t < 1.0:
            return False
        line = "v1 rank=%d %s\n" % (self.rank, " ".join(f"{k}={v}" for k, v in kv.items()))
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                f.write(line)
            os.replace(tmp, self.path)
        except OSError:
            self.errors += 1
            return False
        self._last_key, self._last_t = key, now
        return True


def read_pobs(ctl: str, ranks: Sequence[int] = (0, 1, 2), stale_s: float = 5.0,
              wall: Callable[[], float] = time.time) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for r in ranks:
        p = pobs_path(ctl, r)
        try:
            st = os.stat(p)
            if wall() - st.st_mtime > stale_s:
                continue
            with open(p) as f:
                txt = f.read(512)
        except OSError:
            continue
        parts = txt.split()
        if not parts or parts[0] != "v1":
            continue
        out[r] = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
    return out


# ----------------------------------------------------------------------------- P side: the actuator

def default_arena_reader(sched) -> Optional[Tuple[Optional[float], str]]:
    """(fill, kind) of the shared arena as PP0 sees it, or None when there is no arena pool. ``kind`` is
    ``exact`` (the pinned census) or ``bound`` (the O(1) header bound, below the hold threshold)."""
    try:
        from sglang.srt.weg2 import dual_arena_spill as _das

        pool = _das._tree_pool(sched)
        arena = getattr(pool, "arena", None)
        if arena is None:
            return None
        hi = _envf(os.environ, HOLD_HI_ENV, HOLD_ARENA_HI_DEFAULT, 0.10, 1.0)
        pinned, slots = _das.read_fill(arena, False, hi)
        if pinned is None:
            st = arena.stats()
            return float(int(st["complete"])) / max(1, int(slots)), "bound"
        return float(pinned) / max(1, int(slots)), "exact"
    except Exception:  # noqa: BLE001 - an instrument never breaks the pass
        return None


class GreenActuator:
    """The P-side object of one scheduler: wire (PP0 stamps, followers absorb), stream pick per forward, the
    hold gate (PP0) and the observation files. ``ladder`` may serve only some rungs; the rest fall back."""

    def __init__(self, ladder: GreenLadder, reader: _ds.CtlReader, *, pp_rank: int, pp_size: int,
                 hold: Optional[HoldGate] = None, pobs: Optional[PObsWriter] = None,
                 arena_reader: Callable[[Any], Optional[Tuple[Optional[float], str]]] = default_arena_reader,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = logger.info):
        self.ladder = ladder
        self.reader = reader
        self.pp_rank = int(pp_rank)
        self.pp_size = int(pp_size)
        self.first = self.pp_rank == 0
        self.hold = hold
        self.pobs = pobs
        self._arena_reader = arena_reader
        self._clock, self._sleep, self._log = clock, sleep, log
        # wire state
        self.seq = 0                    # PP0: last stamped; followers: last absorbed
        self.rung = 0                   # wanted stage (set by apply, consumed by pick)
        self.f = 1.0
        self.active = 0                 # the stage of the forward being launched / last launched
        self.active_sm = ladder.info.sm_total if ladder.info else 0
        self._stamped_key = None
        self._stamp_t = -1e9
        self._arena_t = -1e9
        self.arena: Optional[float] = None
        self.arena_kind = "n/a"
        self.verdict = HoldVerdict(False, False, "init")
        self._last_verdict_key = None
        self._held_once_named = False
        # counters (marker)
        self.switches = 0
        self.forwards = 0
        self.by_rung: Dict[int, int] = {}
        self.fallback_forwards = 0      # a stage the ladder could not serve ran on the primary stream
        self.switch_us_total = 0.0
        self.stamps = 0
        self.holds = 0
        self.hold_slept_s = 0.0
        self.last_forward_rung: Optional[int] = None

    # -- serving state ------------------------------------------------------------
    def serves_rung(self, rung: int) -> bool:
        """ShareDuty / ChunkCap ask: is this ctl rung served by the ladder (so they must not throttle too)?"""
        f = self._fraction_of(rung)
        return self.ladder.serves(f)

    def _fraction_of(self, rung: int) -> float:
        fr = self.ladder.fractions
        return fr[rung] if 0 <= int(rung) < len(fr) else 1.0

    # -- PP0: decide + stamp -------------------------------------------------------
    def pp0_stamp(self, wire_reqs: Sequence[Any], now: Optional[float] = None) -> Tuple[List[Any], Optional[Weg2DualGreenRung]]:
        """PP0, before the chain send: ``(list to SEND, stamp or None)``. Off PP0: the list itself."""
        if not self.first or self.pp_size < 2:
            return wire_reqs, None
        t = self._clock() if now is None else float(now)
        try:
            st = self.reader.read()
            key = (int(st.rung), round(float(st.fraction), 4))
            if key == self._stamped_key and t - self._stamp_t < HEARTBEAT_S:
                return wire_reqs, None
            self.seq += 1
            cmd = Weg2DualGreenRung(self.seq, int(st.rung), int(round(float(st.fraction) * 1_000_000)))
        except Exception:  # noqa: BLE001 - an optional throttle never takes the pass (and the rank) down
            logger.exception("%s STOP stamp_failed: no stage this pass (the previous one stays)", MARK)
            return wire_reqs, None
        self._stamped_key, self._stamp_t = key, t
        self.stamps += 1
        self.apply(cmd)
        return list(wire_reqs or ()) + [cmd], cmd

    def follower_absorb(self, recv_reqs: List[Any]) -> List[Any]:
        """A follower, after relaying the list onward and before dispatch: take PP0's stage off it."""
        if not recv_reqs:
            return recv_reqs
        cmds = [r for r in recv_reqs if isinstance(r, Weg2DualGreenRung)]
        if not cmds:
            return recv_reqs
        rest = [r for r in recv_reqs if not isinstance(r, Weg2DualGreenRung)]
        for c in sorted(cmds, key=lambda c: c.seq):
            self.apply(c)
        return rest

    def apply(self, cmd: Weg2DualGreenRung) -> None:
        """Record the wanted stage. The ACTIVE stage changes only in :meth:`pick` (a forward boundary)."""
        self.seq = int(cmd.seq)
        self.rung = int(cmd.rung)
        self.f = max(0.0, min(1.0, cmd.f_ppm / 1_000_000.0))

    # -- the forward ----------------------------------------------------------------
    def pick(self, sched) -> Tuple[Any, Optional[Any]]:
        """``(stream context, stream or None)`` for the forward about to launch. None stream = the primary
        ``forward_stream`` (rung 0, or a stage the ladder cannot serve: named, counted)."""
        t0 = time.perf_counter()
        entry = self.ladder.entry(self.f) if self.f < 1.0 - 1e-9 else None
        want = self.rung
        served = self.f >= 1.0 - 1e-9 or entry is not None
        if not served:
            self.fallback_forwards += 1
        eff = want if served else 0
        if eff != self.active or self.forwards == 0:
            prev = self.active
            self.active = eff
            self.active_sm = entry.sm if entry is not None else (self.ladder.info.sm_total if self.ladder.info else 0)
            _STATE["eager"] = entry is not None
            self.switches += 1 if self.forwards else 0
            self._log(f"{MARK} P rung {prev}->{eff} f={self.f:.2f} sm_real={self.active_sm} "
                      f"eager={1 if entry is not None else 0} seq={self.seq} pp={self.pp_rank} "
                      f"served={1 if served else 0} switches={self.switches}")
        else:
            _STATE["eager"] = entry is not None
        self.forwards += 1
        self.by_rung[eff] = self.by_rung.get(eff, 0) + 1
        self.last_forward_rung = eff
        self.switch_us_total += (time.perf_counter() - t0) * 1e6
        self._observe_files()
        if entry is None:
            return sched.forward_stream_ctx, None
        return sched.device_module.stream(entry.stream), entry.stream

    # -- PP0: hold (observer by default) ------------------------------------------------
    def _read_arena(self, sched) -> None:
        now = self._clock()
        if now - self._arena_t < ARENA_READ_S:
            return
        self._arena_t = now
        got = self._arena_reader(sched)
        if got is None:
            self.arena, self.arena_kind = None, "n/a"
        else:
            self.arena, self.arena_kind = got

    def before_forward(self, sched) -> float:
        """PP0 only (False elsewhere): evaluate the hold gate; with ``--dual-green-ladder hold`` block here
        (bounded) while it holds. Returns the seconds slept. Always writes the observation files."""
        slept = 0.0
        if self.hold is None or not self.first:
            return 0.0
        deadline = self._clock() + self.hold.max_s + 0.5
        while True:
            st = self.reader.read()
            if self.first:
                self._read_arena(sched)
            v = self.hold.update(self.arena, bool(st.d_busy), bool(getattr(st, "starve", False)))
            self.verdict = v
            self._note_verdict(v)
            if not v.hold or self._clock() >= deadline:
                break
            self._sleep(_ds.READ_EVERY_S)
            slept += _ds.READ_EVERY_S
            self.holds += 1
        self.hold_slept_s += slept
        return slept

    def _note_verdict(self, v: HoldVerdict) -> None:
        key = (v.would_hold, v.hold, v.reason)
        if key == self._last_verdict_key:
            return
        self._last_verdict_key = key
        if v.would_hold or self._last_verdict_key is not None and (v.reason in ("below", "d_idle")):
            self._log(f"{MARK} P hold {'OBSERVER ' if not self.hold or not self.hold.actuate else ''}"
                      f"would_hold={int(v.would_hold)} hold={int(v.hold)} reason={v.reason} "
                      f"arena={'n/a' if self.arena is None else format(self.arena, '.3f')}({self.arena_kind}) "
                      f"hi={self.hold.hi if self.hold else 0:.2f} lo={self.hold.lo if self.hold else 0:.2f} "
                      f"episodes={self.hold.episodes if self.hold else 0} "
                      f"observed={self.hold.observed_episodes if self.hold else 0} "
                      f"slept_s={self.hold_slept_s:.2f} pp={self.pp_rank}")

    def _observe_files(self) -> None:
        if self.pobs is None:
            return
        v = self.verdict
        self.pobs.update(sm=self.active_sm, rung=self.active, seq=self.seq, eager=int(_STATE["eager"]),
                         served=int(self.ladder.serves(self.f)),
                         arena_ppm=-1 if self.arena is None else int(self.arena * 1_000_000),
                         would_hold=int(v.would_hold), hold=int(v.hold))

    def status_line(self) -> str:
        return (f"{MARK} P status pp={self.pp_rank} forwards={self.forwards} switches={self.switches} "
                f"by_rung={dict(sorted(self.by_rung.items()))} fallback_forwards={self.fallback_forwards} "
                f"stamps={self.stamps} seq={self.seq} switch_us_avg="
                f"{(self.switch_us_total / max(1, self.forwards)):.1f} holds_polls={self.holds} "
                f"hold_slept_s={self.hold_slept_s:.2f}")


class _DeadBackend(GreenBackend):
    """Stands in when the real backend cannot even be made (no libcuda symbols, no CUDA): the ladder then has
    no rung, every stage is served by the duty/chunk fallback, and the wire stays symmetric on all P stages."""

    def __init__(self, why: str):
        self.why = why

    def info(self) -> DevInfo:
        raise GreenError(self.why)


def maybe_arm(sched, env: Optional[Mapping[str, str]] = None, *, backend: Optional[GreenBackend] = None,
              log: Callable[[str], None] = logger.info,
              warn: Callable[[str], None] = logger.warning) -> Optional[GreenActuator]:
    """The scheduler's construct step. None off the gate (nothing imported or created: flip/NF/INT8 and a default
    boot). ON the gate it never returns None and never raises: whatever fails (no green-context API under MPS,
    a rung, the probe) is named and the ladder serves fewer rungs -- but the actuator exists on EVERY P stage,
    so PP0's stage order on the wire always finds a follower that takes it off the list."""
    e = os.environ if env is None else env
    if not armed(e):
        if ladder_switch(e):
            warn(_ds.fallback_line("green", "all", "SGLANG_WEG2_DUAL_GREEN_LADDER=1 but the dual P gate is off (needs "
                                   "SGLANG_WEG2_DUAL_LAYOUT=1, SGLANG_WEG2_GROUP=P, SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS>0, "
                                   "the green actuator and a ctl file) -> duty/chunk, no ladder"))
        return None
    ps = sched.ps
    pp_rank, pp_size = int(ps.pp_rank), int(ps.pp_size)
    cfg = _ds.config_from_env(e)
    try:
        be = backend if backend is not None else CtypesBackend()
    except Exception as ex:  # noqa: BLE001 - never a crash: the base actuators stay
        be = _DeadBackend(f"{type(ex).__name__}: {ex}")
    probe = str(e.get(PROBE_ENV, "1")).strip() != "0"
    ladder = GreenLadder(be, cfg.rungs, probe=probe,
                         min_slowdown_frac=_envf(e, PROBE_MIN_SLOWDOWN_FRAC_ENV, 0.3, 0.0, 1.0),
                         log=log, warn=warn, rank=pp_rank).build()
    log(ladder.marker())
    if ladder.healthy_count == 0:
        warn(_ds.fallback_line("green", ladder.info.uuid if ladder.info else "all",
                               "no rung served -> the duty/chunk actuators take over (rung 0 = primary stream)"))
    ctl = str(e.get(_ds.CTL_ENV, "") or "").strip()
    act = GreenActuator(ladder, _ds.CtlReader(ctl, log=warn, card=ladder.info.uuid if ladder.info else "all"),
                        pp_rank=pp_rank, pp_size=pp_size,
                        hold=HoldGate.from_env(e) if pp_rank == 0 else None,
                        pobs=PObsWriter(ctl, pp_rank), log=log)
    _STATE["actuator"] = act
    _STATE["serving"] = ladder.healthy_count > 0
    if _STATE["serving"]:
        _ds.set_green_serves(act.serves_rung)
    log(f"{MARK} P actuator armed pp={pp_rank}/{pp_size} ctl={ctl} hold="
        f"{'ARMED' if hold_armed(e) else 'observer'} serving={int(_STATE['serving'])} (PP0 stamps the wire, "
        f"followers absorb; below 100 % the prefill graph runner answers eager)")
    return act


# ----------------------------------------------------------------------------- FRONT: the 5-stage automaton

def parse_factors(raw: str) -> Tuple[float, ...]:
    vals = tuple(float(x) for x in raw.split(",") if x.strip())
    if not vals or any(v <= 0 for v in vals):
        raise ValueError(f"{_ds.LOG_TAG}: GREEN_FACTORS need positive values, got {raw!r}")
    return vals


def parse_tsolo(raw: str) -> Tuple[Tuple[int, float], ...]:
    pts = []
    for kv in raw.split(","):
        if ":" in kv:
            b, ms = kv.split(":", 1)
            pts.append((int(b), float(ms)))
    pts.sort()
    if not pts or any(ms <= 0 for _, ms in pts):
        raise ValueError(f"{_ds.LOG_TAG}: GREEN_TSOLO_MS needs bs:ms points, got {raw!r}")
    return tuple(pts)


@dataclass(frozen=True)
class GreenConfig:
    factors: Tuple[float, ...] = DEFAULT_FACTORS
    tsolo_ms: Tuple[Tuple[int, float], ...] = DEFAULT_TSOLO_MS
    accept_len: float = DEFAULT_ACCEPT_LEN
    desc_min_s: float = 0.5           # descend (more D) at once, at most this often. Spec 0.25 s, 0.5 here ON PURPOSE: the
                                      # front's D-rate window is 0.5 s, a shorter dwell would re-read the old stage's rounds
    desc_fast_ratio: float = 2.0      # round > this x target: two stages at once
    asc_ratio: float = 0.7            # ascend only when round <= this x target ...
    asc_calm_s: float = 1.5           # ... for this long ...
    asc_dwell_s: float = 2.0          # ... and this long since the last change
    table: Tuple[Tuple[int, int, int], ...] = ((2, 1, 0), (4, 2, 1), (10 ** 9, 3, 2))   # (bs <=, tau low, tau high)
    tick_s: float = 0.05

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "GreenConfig":
        def g(name: str) -> str:
            return str(env.get(_ds.ENV_PREFIX + "GREEN_" + name, "") or "").strip()

        kw: Dict[str, Any] = {}
        if g("FACTORS"):
            kw["factors"] = parse_factors(g("FACTORS"))
        if g("TSOLO_MS"):
            kw["tsolo_ms"] = parse_tsolo(g("TSOLO_MS"))
        for name, key in (("ACCEPT_LEN", "accept_len"), ("DESC_MIN_S", "desc_min_s"),
                          ("DESC_FAST_RATIO", "desc_fast_ratio"), ("ASC_RATIO", "asc_ratio"),
                          ("ASC_CALM_S", "asc_calm_s"), ("ASC_DWELL_S", "asc_dwell_s")):
            if g(name):
                kw[key] = float(g(name))
        if g("TABLE"):          # "2:1:0;4:2:1;99:3:2" = bs<=, tau-low stage, tau-high stage
            kw["table"] = tuple(tuple(int(x) for x in row.split(":")) for row in g("TABLE").split(";"))
        cfg = cls(**kw)
        if cfg.accept_len <= 0 or not (0 < cfg.asc_ratio < 1):
            raise ValueError(f"{_ds.LOG_TAG}: GREEN accept_len > 0 and 0 < asc_ratio < 1 needed")
        return cfg


class TSolo:
    """D's solo round (ms) by bs: the default points interpolated; where the front SAW D with P idle,
    the EWMA of that measurement (n >= 3)."""

    def __init__(self, points: Sequence[Tuple[int, float]], alpha: float = 0.2):
        self.points = tuple(points)
        self.alpha = alpha
        self.learned: Dict[int, List[float]] = {}     # bs -> [ewma_ms, n]

    def note(self, bs: int, round_ms: float) -> None:
        if bs <= 0 or not (5.0 <= round_ms <= 2000.0):
            return
        ent = self.learned.get(bs)
        if ent is None:
            self.learned[bs] = [float(round_ms), 1]
        else:
            ent[0] += self.alpha * (round_ms - ent[0])
            ent[1] += 1

    def get(self, bs: int) -> Tuple[float, str]:
        bs = max(1, int(bs))
        ent = self.learned.get(bs)
        if ent is not None and ent[1] >= 3:
            return ent[0], "learned"
        pts = self.points
        if bs <= pts[0][0]:
            if len(pts) == 1:
                return pts[0][1], "default"
            (b0, t0), (b1, t1) = pts[0], pts[1]
        elif bs >= pts[-1][0]:
            if len(pts) == 1:
                return pts[-1][1], "default"
            (b0, t0), (b1, t1) = pts[-2], pts[-1]
        else:
            b0, t0, b1, t1 = pts[0][0], pts[0][1], pts[1][0], pts[1][1]
            for a, b in zip(pts, pts[1:]):
                if a[0] <= bs <= b[0]:
                    (b0, t0), (b1, t1) = a, b
                    break
        slope = (t1 - t0) / max(1, b1 - b0)
        return max(5.0, t0 + slope * (bs - b0)), "default"


@dataclass
class GreenDecision(_ds.Decision):
    green: bool = True
    start: int = 0
    adj: int = 0
    d_round_ms: Optional[float] = None
    target_ms: Optional[float] = None
    tsolo_ms: Optional[float] = None
    tsolo_src: str = "default"
    starve: bool = False
    bs_class: str = "-"


class GreenController(_ds.ShareController):
    """Front side, ``--dual-priority dynamic`` with the ladder armed (``balanced`` / ``d`` keep their static
    stage, ``p`` stays 100 %). Stages 0..3 = ``cfg.rungs`` (100/75/50/25 %); the hold is not decided here (PP0
    owns it, section 8.4 -- the arena is PP0's to read).

    ENTRY (open loop, section 8.3): D busy after idle -> the stage at once. With a target (R_min = the
    controller's ``d_min_rate_tps``): the smallest stage k with ``s_k * T_solo(bs) <= accept_len / R_min``
    (the more bs, the deeper); without: the table (bs class x tau class). THEN the closed loop on the
    measured D round (``accept_len / d_rate``): round over target -> descend at once (two stages when over
    ``desc_fast_ratio`` x), at most every ``desc_min_s``; round <= ``asc_ratio`` x target for ``asc_calm_s``
    and ``asc_dwell_s`` since the last change, and the next stage's PREVIEW (the model factor scaled by what
    the current stage measured) under target -> ascend. D empty -> stage 0 at once, state reset. The dead band
    is the 0.7 / 1.0 gap of the ratio plus ``cfg.deadband`` on the tau class."""

    def __init__(self, cfg: _ds.ShareConfig, mode: str, gcfg: Optional[GreenConfig] = None,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__(cfg, mode, clock=clock)
        self.gcfg = gcfg or GreenConfig()
        if len(self.gcfg.factors) < len(cfg.rungs):
            raise ValueError(f"{_ds.LOG_TAG}: GREEN_FACTORS needs {len(cfg.rungs)} values (one per rung)")
        self.tsolo = TSolo(self.gcfg.tsolo_ms)
        self.adj = 0
        self._was_busy = False
        self._calm_since: Optional[float] = None
        self._pobs: Dict[int, Dict[str, str]] = {}
        self._last_start = 0

    # -- front feeds ---------------------------------------------------------------
    def feed_pobs(self, pobs: Mapping[int, Mapping[str, str]]) -> None:
        self._pobs = {int(k): dict(v) for k, v in pobs.items()}

    def pobs_view(self) -> Dict[str, str]:
        """arena (PP0), would_hold (PP0), sm per stage, as log text."""
        p0 = self._pobs.get(0, {})
        ap = p0.get("arena_ppm")
        arena = "n/a" if ap is None or int(ap) < 0 else f"{int(ap) / 1e6:.3f}"
        sm = ",".join(f"{r}:{v['sm']}" for r, v in sorted(self._pobs.items()) if "sm" in v) or "n/a"
        return {"arena": arena, "would_hold": p0.get("would_hold", "n/a"), "hold": p0.get("hold", "n/a"), "sm": sm}

    # -- the open loop --------------------------------------------------------------
    def start_stage(self, bs: int, tau_s: Optional[float]) -> Tuple[int, str]:
        cfg, g = self.cfg, self.gcfg
        top = cfg.max_rung()
        rmin = float(cfg.d_min_rate_tps)
        if rmin > 0.0:
            target = 1000.0 * g.accept_len / rmin
            ts, src = self.tsolo.get(bs)
            for k in range(0, top + 1):
                if g.factors[k] * ts <= target:
                    return k, f"model(rmin={rmin:g},tsolo={ts:.0f}{src[0]})"
            return top, f"model(rmin={rmin:g},tsolo={ts:.0f}{src[0]},none_fits)"
        self._tau_cls = (None if tau_s is None else
                         _ds._classify(tau_s, cfg.tau_edges_s, self._tau_cls, cfg.deadband))
        high = 1 if (self._tau_cls is not None and self._tau_cls >= 2) else 0
        for edge, lo_k, hi_k in g.table:
            if bs <= edge:
                return min(top, hi_k if high else lo_k), f"table(bs<={edge if edge < 10 ** 8 else 'inf'},tau={'high' if high else 'low'})"
        return top, "table(overflow)"

    def _bs_class(self, bs: int) -> str:
        for edge, _lo, _hi in self.gcfg.table:
            if bs <= edge:
                return f"<={edge if edge < 10 ** 8 else 'inf'}"
        return "?"

    def observe(self, *, q_tokens: float, b: int, seats: int, p_rate_tps: Optional[float],
                d_rate_tps: Optional[float], oldest_age_s: float = 0.0, p_idle: bool = False,
                **_ignored) -> GreenDecision:
        cfg, g = self.cfg, self.gcfg
        now = self._clock()
        dt = 0.0 if self._t is None else now - self._t
        self._t = now
        self._q = self._ewma(self._q, max(0.0, float(q_tokens)), dt)
        self._b = self._ewma(self._b, max(0, int(b)), dt)
        tau = (self._q / p_rate_tps) if (p_rate_tps and p_rate_tps > 0) else None
        rmin = float(cfg.d_min_rate_tps)
        d_round = (1000.0 * g.accept_len / d_rate_tps) if (d_rate_tps and d_rate_tps > 0) else None
        target_ms = (1000.0 * g.accept_len / rmin) if rmin > 0 else None
        ts_ms, ts_src = self.tsolo.get(max(1, int(b)))
        if p_idle and b > 0 and d_round is not None:
            self.tsolo.note(int(b), d_round)            # D alone: that IS the solo round
        top = cfg.max_rung()
        prev = self.rung
        immediate = False
        start = 0
        reason = ""
        since = now - self.last_change_t
        if self.mode == "p":
            target, reason = 0, "mode_p"
            immediate = True
        elif b <= 0:
            target, reason, immediate = 0, "d_idle", True
            self.adj, self._calm_since, self._was_busy = 0, None, False
        else:
            if self.mode in ("balanced", "d"):
                st = cfg.static_rung(self.mode)
                target, reason = (st if st is not None else 0), f"static_{self.mode}"
                start = target
                if not self._was_busy:                  # D busy after idle: the static stage at once
                    immediate = True
                    reason += "+entry"
            else:
                start, why = self.start_stage(int(b), tau)
                reason = why
                if not self._was_busy:                  # entry after idle: the open loop's stage at once
                    self.adj, self._calm_since = 0, None
                    immediate = True
                    reason += "+entry"
                target = max(0, min(top, start + self.adj))
                # ---- closed loop (needs a target and a measured round) ----
                if target_ms is not None and d_round is not None and not immediate:
                    ratio = d_round / target_ms
                    if ratio > 1.0:
                        self._calm_since = None
                        if since >= g.desc_min_s and self.rung < top:
                            step = 2 if ratio > g.desc_fast_ratio else 1
                            target = min(top, self.rung + step)
                            self.adj = target - start
                            immediate = True
                            reason += f"+d_round_over(x{ratio:.2f})"
                    elif ratio <= g.asc_ratio:
                        if self._calm_since is None:
                            self._calm_since = now
                        elif (now - self._calm_since >= g.asc_calm_s and since >= g.asc_dwell_s
                              and self.rung > 0):
                            nxt = self.rung - 1
                            model_cur = g.factors[self.rung]
                            meas = d_round / max(1.0, ts_ms)
                            scale = min(4.0, max(0.25, meas / model_cur))
                            if g.factors[nxt] * scale * ts_ms <= target_ms:
                                target = nxt
                                self.adj = target - start
                                self._calm_since = now
                                immediate = True
                                reason += f"+d_round_calm(x{ratio:.2f},preview_ok)"
                            else:
                                reason += f"+calm_preview_blocks(x{ratio:.2f})"
                    else:
                        self._calm_since = None
                self._last_start = start
            self._was_busy = True
        starve = bool(self.mode != "p" and oldest_age_s > cfg.starve_age_s)
        if starve and target > cfg.starve_max_rung:
            target = cfg.starve_max_rung
            reason += "+starve"
            immediate = True
        if target > top:
            target = top
            reason += "+p_min_share"
        held = False
        if target != self.rung:
            if immediate:
                self.rung = target
            elif target > self.rung:
                if since >= cfg.dwell_to_d_s:
                    self.rung += 1
                else:
                    held = True
            else:
                if since >= cfg.dwell_to_p_s:
                    self.rung -= 1
                else:
                    held = True
        self._mode_jump = False
        changed = self.rung != prev
        dwell = now - self.last_change_t if self.last_change_t > -1e8 else 0.0
        if changed:
            d = 1 if self.rung > prev else -1
            if self._last_dir and d != self._last_dir and dwell < cfg.flap_window_s:
                self.flaps += 1
            self._last_dir = d
            self.last_change_t = now
            self.changes += 1
        if held:
            reason += "+dwell_hold"
        return GreenDecision(mode=self.mode, rung=self.rung, prev_rung=prev, target=target,
                             fraction=cfg.rungs[self.rung], tau_s=tau, b=int(b), seats=int(seats),
                             q_tokens=float(self._q or 0.0), p_rate_tps=p_rate_tps, d_rate_tps=d_rate_tps,
                             reason=reason, changed=changed, held=held, flaps=self.flaps, dwell_s=dwell,
                             start=start, adj=self.adj, d_round_ms=d_round, target_ms=target_ms,
                             tsolo_ms=ts_ms, tsolo_src=ts_src, starve=starve, bs_class=self._bs_class(int(b)))

    def set_mode(self, mode: str) -> bool:
        ch = super().set_mode(mode)
        if ch:
            self.adj, self._calm_since = 0, None
        return ch


def pstufe_line(d: GreenDecision, obs: Mapping[str, str], kv: str = "n/a") -> str:
    """The ONE measurement line per stage decision (grep ``P-STUFE``), section 8.5. ``stufe`` is the rung
    index 0..3 = 100/75/50/25 %; the hold is PP0's and appears as ``would_hold`` (observer) -- ``kv`` is n/a
    until a D-side feed exists (D's ``full token usage`` is the wrong quantity as a gate)."""
    tau = "n/a" if d.tau_s is None else f"{d.tau_s:.2f}s"
    dr = "n/a" if d.d_round_ms is None else f"{d.d_round_ms:.0f}"
    tg = "n/a" if d.target_ms is None else f"{d.target_ms:.0f}"
    return (f"P-STUFE bs={d.b}/{d.seats} kv={kv} arena={obs.get('arena', 'n/a')} pending={d.q_tokens:.0f} "
            f"tau={tau} d_round_ms={dr} target_ms={tg} tsolo_ms={d.tsolo_ms:.0f}({d.tsolo_src}) "
            f"stufe={d.prev_rung}->{d.rung} f={d.fraction:.2f} start={d.start} adj={d.adj} "
            f"sm={obs.get('sm', 'n/a')} would_hold={obs.get('would_hold', 'n/a')} hold={obs.get('hold', 'n/a')} "
            f"reason={d.reason} flaps={d.flaps} dwell_s={d.dwell_s:.2f}")
