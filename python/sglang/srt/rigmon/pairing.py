"""Couple a second rig: reach it, gate it, choose a transport, emit a config.

Task #214. The whole flow runs **on the host**, exactly like telemetry
collection and plan computation already do. The dashboard is a steering
surface: it posts a command and renders the state that comes back. No step of
this flow is implemented in the browser, and the browser is never told
anything the host has not first computed and stored.

Three things follow, all of them wanted:

* The flow works **without a dashboard**. Every step is one endpoint taking
  one small JSON body, so a shell script with ``curl`` is a complete client.
* The state — which step, which result, which error with which remedy — lives
  here, so a browser reload resumes a pairing instead of restarting it.
* Slow steps do not block. :meth:`PairingStore.advance` returns as soon as the
  step is *running*; the caller polls. Nothing waits on a network round trip
  while holding an HTTP request open.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not boot anything across rigs. The flow ends at a validated start
configuration -- an argv and an environment block -- which a human or a
separate call then uses. Task #214 is the coupling, not the launch.

It also invents nothing that rigmon already has. The capability table
(:mod:`.capabilities`), the join gate (:mod:`.compat`), the transport ranking
(:mod:`.transport`) and the pair matrix (:mod:`.probe`) are all existing
machinery; this module is the sequencing around them plus the one piece that
genuinely was missing -- a client that READS a remote rigmon. Until now rigmon
only ever received pushes, so no code here could ask "what does rig B say
about itself".
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Dict, List, Optional

from sglang.srt.rigmon import capabilities as capsmod
from sglang.srt.rigmon import compat as compatmod
from sglang.srt.rigmon import facilities as facilitiesmod
from sglang.srt.rigmon import probe as probemod
from sglang.srt.rigmon import transport as transportmod

__all__ = [
    "STEPS",
    "PENDING",
    "RUNNING",
    "OK",
    "BLOCKED",
    "ERROR",
    "RemoteView",
    "StepResult",
    "PairingSession",
    "PairingStore",
    "GateRow",
    "fetch_remote",
    "gate_rows",
    "transport_rows",
    "launch_config",
    "STORE",
]

#: The flow, in order. Each is one endpoint call; each records its own result.
STEPS = ("reach", "gate", "transport", "config")

PENDING = "pending"
RUNNING = "running"
OK = "ok"
BLOCKED = "blocked"
ERROR = "error"

DEFAULT_TIMEOUT_S = 6.0
#: A session with no activity for this long is dropped, so a long-lived
#: dashboard process does not accumulate abandoned pairings.
SESSION_TTL_S = 24 * 3600


# ---------------------------------------------------------------------------
# Reading a remote rigmon
# ---------------------------------------------------------------------------


def _default_opener(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "htsglang-pairing"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


@dataclasses.dataclass
class RemoteView:
    """What the far rig says about itself.

    ``reachable`` false is a normal, expected outcome with a reason attached,
    not an exception: "the other rig is off" is the most common answer this
    step gives and the UI has to render it like any other state.
    """

    url: str
    reachable: bool
    rtt_ms: Optional[float] = None
    error: Optional[str] = None
    remedy: Optional[str] = None
    node_ids: List[str] = dataclasses.field(default_factory=list)
    identity: Optional[dict] = None
    capabilities: Optional[dict] = None
    facilities: List[dict] = dataclasses.field(default_factory=list)
    hw_profile: Optional[dict] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def fetch_remote(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    opener: Optional[Callable[[str, float], bytes]] = None,
) -> RemoteView:
    """GET a remote rigmon aggregator and read back what it knows.

    ``opener`` is the injection seam: tests hand in a callable returning
    canned bytes, which is why none of this needs a second machine to be
    exercised. It is the same shape ``EngineScraper`` and ``ProbeEnv`` use
    elsewhere in rigmon.

    Only the aggregator's read-only endpoints are touched (``/api/nodes``,
    ``/api/snapshot``). Nothing is pushed, joined or mutated: reaching a rig
    must be safe to do repeatedly against a rig that is busy serving.
    """
    get = opener or _default_opener
    base = url.rstrip("/")
    if not base.startswith("http"):
        base = "http://" + base
    started = time.time()
    try:
        raw = get(base + "/api/nodes", timeout)
        nodes = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        return RemoteView(
            url=base,
            reachable=False,
            error=f"HTTP {e.code} from {base}/api/nodes",
            remedy="Check that the port belongs to a rigmon aggregator "
            "(rigmon serve), not to an sglang server or another service.",
        )
    except Exception as e:
        return RemoteView(
            url=base,
            reachable=False,
            error=f"{type(e).__name__}: {e}",
            remedy="Start the aggregator on the far rig "
            "(python -m sglang.srt.rigmon serve) and make sure its port is "
            "reachable from here.",
        )
    rtt_ms = (time.time() - started) * 1000.0

    snapshot: dict = {}
    try:
        snapshot = json.loads(get(base + "/api/snapshot", timeout).decode("utf-8"))
    except Exception:
        # Reachable but not yet reporting: a collector that has not ticked.
        # Not an error -- the gate below simply has less to work with.
        snapshot = {}

    node_list = nodes.get("nodes") or []
    node_ids = [n.get("node_id") or n.get("id") or "" for n in node_list]
    node_ids = [n for n in node_ids if n]

    identity = None
    caps = None
    facs: List[dict] = []
    hw_profile = None
    for _nid, entry in (snapshot.get("nodes") or {}).items():
        state = (entry or {}).get("state") or {}
        identity = identity or state.get("identity")
        caps = caps or state.get("capabilities")
        facs = facs or (state.get("facilities") or [])
        hw_profile = hw_profile or state.get("hw_profile")
    if identity is None:
        for n in node_list:
            if n.get("identity"):
                identity = n["identity"]
                break

    return RemoteView(
        url=base,
        reachable=True,
        rtt_ms=round(rtt_ms, 2),
        node_ids=node_ids,
        identity=identity,
        capabilities=caps,
        facilities=facs,
        hw_profile=hw_profile,
    )


# ---------------------------------------------------------------------------
# The compatibility gate
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GateRow:
    """One precondition held against both rigs.

    Every unmet row carries ``reason`` AND ``remedy``. Nothing is greyed out
    silently: a control the reader cannot use is only informative if it says
    why and what would change that. This is the existing capability-table
    rule, applied across two rigs instead of one.
    """

    key: str
    label: str
    verdict: str  # ok | warn | block
    local: str = ""
    remote: str = ""
    reason: str = ""
    remedy: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


#: Capabilities that must line up on BOTH sides before a cross-rig run is
#: worth attempting, with what to do when they do not.
_PAIRED_CAPABILITIES = {
    "nccl_colocation": (
        "Both rigs need an NCCL new enough for multiple ranks per GPU.",
        "Install NCCL 2.30 or newer on the rig reported below, or set "
        "SGLANG_NCCL_SO_PATH to one that is.",
    ),
    "rdma": (
        "RDMA is the low-latency inter-rig transport; without it the pairing "
        "falls back to TCP.",
        "Bring up an InfiniBand or RoCE port on the rig reported below, or "
        "accept the TCP transport in the next step.",
    ),
    "mps": (
        "MPS lets several ranks share one card without serialising them.",
        "Start the MPS control daemon on the rig reported below "
        "(nvidia-cuda-mps-control -d) if that rig is meant to host "
        "co-located ranks.",
    ),
}


def _cap_state(report: Optional[dict], key: str) -> tuple:
    for c in ((report or {}).get("capabilities") or []):
        if c.get("key") == key:
            return c.get("state") or capsmod.UNKNOWN, c.get("reason") or ""
    return capsmod.UNKNOWN, "not reported"


def gate_rows(
    local_identity: compatmod.NodeIdentity,
    remote_identity: Optional[compatmod.NodeIdentity],
    local_caps: Optional[dict],
    remote_caps: Optional[dict],
) -> List[GateRow]:
    """Hold both rigs against each other and report every mismatch.

    Two sources, deliberately kept distinct: the existing join gate
    (:func:`compat.check_compatibility`, which already knows commit, torch,
    CUDA, driver, arch coverage and model fingerprints) and the capability
    tables, which the join gate does not look at.
    """
    rows: List[GateRow] = []

    if remote_identity is None:
        rows.append(
            GateRow(
                key="identity",
                label="Remote identity",
                verdict=compatmod.BLOCK,
                reason="The far rig is reachable but reports no identity, so "
                "nothing can be compared.",
                remedy="Run the collector on the far rig so it publishes its "
                "identity (python -m sglang.srt.rigmon collect).",
            )
        )
        return rows

    report = compatmod.check_compatibility(local_identity, remote_identity)
    for c in report.checks:
        rows.append(
            GateRow(
                key=c.key,
                label=c.label,
                verdict=c.verdict,
                local=local_identity.node_id,
                remote=remote_identity.node_id,
                reason=c.detail,
                # compat already attaches the fix for everything it blocks;
                # only the OK rows legitimately have none.
                remedy=c.remedy or "",
            )
        )

    for key, (why, fix) in _PAIRED_CAPABILITIES.items():
        lstate, lreason = _cap_state(local_caps, key)
        rstate, rreason = _cap_state(remote_caps, key)
        good = {capsmod.ACTIVE, capsmod.AVAILABLE}
        if lstate in good and rstate in good:
            verdict = compatmod.OK
            reason = f"available on both ({lstate} / {rstate})"
            remedy = ""
        else:
            # Missing on either side is a warning, not a block: the transport
            # step below degrades to what IS available and says so. Refusing
            # the pairing outright here would hide a usable slower path.
            verdict = compatmod.WARN
            weak = "this rig" if lstate not in good else "the far rig"
            detail = lreason if lstate not in good else rreason
            reason = f"{why} Not available on {weak}"
            reason += f" ({detail})." if detail else "."
            remedy = fix
        rows.append(
            GateRow(
                key=f"capability:{key}",
                label=key,
                verdict=verdict,
                local=lstate,
                remote=rstate,
                reason=reason,
                remedy=remedy,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Transport choice
# ---------------------------------------------------------------------------


def transport_rows(
    local_profile: Optional[dict],
    remote: RemoteView,
    *,
    local_node_id: str = "local",
    facility_keys: Optional[List[str]] = None,
) -> dict:
    """Rank the transports for this pair, from measurements where they exist.

    When the #213 short probe has run, :func:`transport.choose_all_pairs`
    ranks from real latency and bandwidth. When it has not, the answer is
    "not yet measured" plus an offer to run it -- never a guess dressed up as
    a recommendation. ``choose_all_pairs`` already emits an ``unknown`` entry
    for every unmeasured pair, so an incomplete matrix cannot look complete.
    """
    facs = facility_keys
    if facs is None:
        facs = [f.key for f in facilitiesmod.facilities() if f.available]

    local_probe = None
    if local_profile:
        try:
            local_probe = probemod.from_hardware_profile(
                local_profile, local_node_id
            )
        except Exception:
            local_probe = None

    remote_probe = None
    if remote.hw_profile:
        try:
            remote_probe = probemod.from_hardware_profile(
                remote.hw_profile, (remote.node_ids or ["remote"])[0]
            )
        except Exception:
            remote_probe = None

    merged = None
    if local_probe is not None and remote_probe is not None:
        merged = local_probe.merge(remote_probe)
    else:
        merged = local_probe or remote_probe

    if merged is None:
        return {
            "measured": False,
            "pairs": [],
            "note": "No pair measurements on either rig yet, so no transport "
            "can be recommended from data.",
            "offer": {
                "study": "short probe (#213)",
                "what": "Measures per-card rates and per-pair latency and "
                "bandwidth; budgeted at "
                f"{int(probemod.BUDGET_SECONDS)} s.",
                "cost": "Occupies the cards for the duration; it does not "
                "boot anything across rigs.",
            },
        }

    choices = transportmod.choose_all_pairs(merged, available_facilities=facs)
    pairs = [c.to_json() for c in choices]
    cross = [p for p in pairs if not p.get("same_node")]
    unmeasured = [p for p in cross if p.get("chosen") is None]

    probe_offer = {
        "study": "short probe (#213)",
        "what": "Measures per-card rates and per-pair latency and bandwidth; "
        f"budgeted at {int(probemod.BUDGET_SECONDS)} s.",
        "cost": "Occupies the cards for the duration; it does not boot "
        "anything across rigs.",
    }

    # Having measured THIS rig says nothing about the link to the other one.
    # Reporting "measured" here because a local matrix exists would be the
    # exact dishonesty the transport module's unknown rows are there to
    # prevent, one level up.
    if not cross:
        return {
            "measured": False,
            "pairs": pairs,
            "cross_rig_pairs": 0,
            "note": "Only this rig has been measured. Nothing has been "
            "measured across the rig boundary, so no transport can be "
            "recommended from data.",
            "offer": probe_offer,
        }

    return {
        "measured": not unmeasured,
        "pairs": pairs,
        "cross_rig_pairs": len(cross),
        "note": (
            "Ranked from the measured cross-rig pair matrix."
            if not unmeasured
            else f"{len(unmeasured)} of {len(cross)} cross-rig pair(s) are "
            "still unmeasured and are reported as unknown rather than guessed."
        ),
        "offer": None if not unmeasured else probe_offer,
    }


# ---------------------------------------------------------------------------
# The resulting start configuration
# ---------------------------------------------------------------------------


def launch_config(
    remote: RemoteView,
    transport: dict,
    *,
    local_node_id: str = "local",
    master_port: int = 29500,
) -> dict:
    """The end of the flow: a validated configuration, ready to copy or hand on.

    Deliberately *not* a launch. #214 is the coupling; starting a run across
    rigs is a separate act with separate consequences, and conflating them
    would mean a reachability check could end in two servers booting.

    No value from this host is baked in as a literal. Host names and ports
    come out as environment placeholders in the ``${VAR:-default}`` form, so
    the block is safe to paste into a repository, an issue or a chat.
    """
    chosen = None
    for p in transport.get("pairs") or []:
        if not p.get("same_node") and p.get("chosen"):
            chosen = p["chosen"]
            break

    env = {
        "HTSGLANG_MASTER_ADDR": "${HTSGLANG_MASTER_ADDR:-<this-rig-address>}",
        "HTSGLANG_MASTER_PORT": "${HTSGLANG_MASTER_PORT:-%d}" % master_port,
        "HTSGLANG_PEER_RIGMON": "${HTSGLANG_PEER_RIGMON:-<far-rig-host:port>}",
    }
    if chosen == "rdma":
        env["NCCL_IB_DISABLE"] = "0"
    elif chosen in ("gloo-tcp", "barlink-ucx"):
        env["NCCL_IB_DISABLE"] = "1"

    notes = []
    if chosen:
        notes.append(f"Transport chosen from measurements: {chosen}.")
    else:
        notes.append(
            "No cross-rig transport could be recommended from measurements; "
            "run the short probe or pick one explicitly before launching."
        )
    notes.append(
        "This is a configuration, not a launch. Nothing has been started on "
        "either rig."
    )

    return {
        "ready": bool(chosen),
        "transport": chosen,
        "env": env,
        "argv": [],
        "notes": notes,
        "peer": {"url_placeholder": "${HTSGLANG_PEER_RIGMON:-<far-rig-host:port>}"},
    }


# ---------------------------------------------------------------------------
# Session state, held on the host
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StepResult:
    step: str
    state: str = PENDING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    detail: Dict[str, Any] = dataclasses.field(default_factory=dict)
    error: Optional[str] = None
    remedy: Optional[str] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PairingSession:
    session_id: str
    target: str
    created: float
    updated: float
    steps: Dict[str, StepResult]

    @property
    def next_step(self) -> Optional[str]:
        """The first step not yet finished successfully.

        A blocked or failed step stays current: the flow does not walk past a
        gate that said no. Re-running it is how the reader retries after
        fixing whatever the remedy named.
        """
        for s in STEPS:
            if self.steps[s].state != OK:
                return s
        return None

    @property
    def blocked(self) -> bool:
        return any(s.state == BLOCKED for s in self.steps.values())

    def to_json(self) -> dict:
        return {
            "session_id": self.session_id,
            "target": self.target,
            "created": self.created,
            "updated": self.updated,
            "next_step": self.next_step,
            "blocked": self.blocked,
            "complete": self.next_step is None,
            "steps": [self.steps[s].to_json() for s in STEPS],
        }


class PairingStore:
    """Sessions, held on the host, advanced in the background.

    Advancing returns immediately with the step marked ``running``; the work
    happens on a worker thread and the caller polls :meth:`get`. That is what
    keeps a slow or unreachable far rig from holding an HTTP request open,
    and it is why a browser reload mid-probe loses nothing.
    """

    def __init__(self, opener: Optional[Callable[[str, float], bytes]] = None):
        self._lock = threading.Lock()
        self._sessions: Dict[str, PairingSession] = {}
        self._opener = opener
        #: Overridable for tests: run the step inline instead of threading.
        self.synchronous = False

    # -- lifecycle ---------------------------------------------------------

    def create(self, target: str) -> PairingSession:
        if not (target or "").strip():
            raise ValueError("no target given")
        now = time.time()
        s = PairingSession(
            session_id=uuid.uuid4().hex[:12],
            target=target.strip(),
            created=now,
            updated=now,
            steps={k: StepResult(step=k) for k in STEPS},
        )
        with self._lock:
            self._expire_locked(now)
            self._sessions[s.session_id] = s
        return s

    def get(self, session_id: str) -> Optional[PairingSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def list(self) -> List[PairingSession]:
        with self._lock:
            return sorted(self._sessions.values(), key=lambda s: -s.updated)

    def reset(self, session_id: str) -> Optional[PairingSession]:
        s = self.get(session_id)
        if s is None:
            return None
        with self._lock:
            s.steps = {k: StepResult(step=k) for k in STEPS}
            s.updated = time.time()
        return s

    def _expire_locked(self, now: float) -> None:
        for sid, s in list(self._sessions.items()):
            if now - s.updated > SESSION_TTL_S:
                del self._sessions[sid]

    # -- advancing ---------------------------------------------------------

    def advance(self, session_id: str, step: Optional[str] = None) -> PairingSession:
        """Start one step. Returns at once; poll :meth:`get` for the result."""
        s = self.get(session_id)
        if s is None:
            raise KeyError(session_id)
        step = step or s.next_step
        if step is None:
            return s
        if step not in STEPS:
            raise ValueError(f"unknown step {step!r}")
        res = s.steps[step]
        if res.state == RUNNING:
            return s
        with self._lock:
            s.steps[step] = StepResult(
                step=step, state=RUNNING, started_at=time.time()
            )
            s.updated = time.time()
        if self.synchronous:
            self._run(s, step)
        else:
            threading.Thread(
                target=self._run, args=(s, step), daemon=True
            ).start()
        return s

    def _finish(
        self,
        s: PairingSession,
        step: str,
        state: str,
        detail: Optional[dict] = None,
        error: Optional[str] = None,
        remedy: Optional[str] = None,
    ) -> None:
        with self._lock:
            r = s.steps[step]
            s.steps[step] = StepResult(
                step=step,
                state=state,
                started_at=r.started_at,
                finished_at=time.time(),
                detail=detail or {},
                error=error,
                remedy=remedy,
            )
            s.updated = time.time()

    def _run(self, s: PairingSession, step: str) -> None:
        try:
            if step == "reach":
                self._run_reach(s)
            elif step == "gate":
                self._run_gate(s)
            elif step == "transport":
                self._run_transport(s)
            elif step == "config":
                self._run_config(s)
        except Exception as e:  # pragma: no cover - defensive
            self._finish(
                s,
                step,
                ERROR,
                error=f"{type(e).__name__}: {e}",
                remedy="This is a bug in the pairing flow rather than a "
                "configuration problem; the detail above is the whole of it.",
            )

    def _run_reach(self, s: PairingSession) -> None:
        view = fetch_remote(s.target, opener=self._opener)
        if not view.reachable:
            self._finish(
                s,
                "reach",
                BLOCKED,
                detail=view.to_json(),
                error=view.error,
                remedy=view.remedy,
            )
            return
        self._finish(s, "reach", OK, detail=view.to_json())

    def _run_gate(self, s: PairingSession) -> None:
        reach = s.steps["reach"]
        if reach.state != OK:
            self._finish(
                s,
                "gate",
                BLOCKED,
                error="The far rig has not been reached yet.",
                remedy="Run the reach step first.",
            )
            return
        view = RemoteView(**reach.detail)
        remote_ident = (
            compatmod.NodeIdentity.from_json(view.identity)
            if view.identity
            else None
        )
        local_ident = compatmod.local_identity(_local_node_id())
        local_caps = capsmod.probe_all().to_json()
        rows = gate_rows(local_ident, remote_ident, local_caps, view.capabilities)
        blocked = [r for r in rows if r.verdict == compatmod.BLOCK]
        detail = {
            "rows": [r.to_json() for r in rows],
            "local_capabilities": local_caps,
            "remote_capabilities": view.capabilities,
            "blocked": bool(blocked),
        }
        if blocked:
            self._finish(
                s,
                "gate",
                BLOCKED,
                detail=detail,
                error=f"{len(blocked)} precondition(s) block this pairing.",
                remedy="Each blocked row carries its own remedy; fix those "
                "and run this step again.",
            )
            return
        self._finish(s, "gate", OK, detail=detail)

    def _run_transport(self, s: PairingSession) -> None:
        reach = s.steps["reach"]
        if reach.state != OK:
            self._finish(
                s,
                "transport",
                BLOCKED,
                error="The far rig has not been reached yet.",
                remedy="Run the reach step first.",
            )
            return
        view = RemoteView(**reach.detail)
        local_profile = None
        try:
            from sglang.srt.rigmon.collector import load_cached_profiles

            # (hardware profile, power profile); cache-only, never triggers a
            # probe -- a probe would allocate a CUDA context on every card.
            local_profile, _power = load_cached_profiles()
        except Exception:
            local_profile = None
        rows = transport_rows(local_profile, view, local_node_id=_local_node_id())
        # Not measured is a perfectly good answer, so this step succeeds and
        # carries the offer; only a hard failure would block here.
        self._finish(s, "transport", OK, detail=rows)

    def _run_config(self, s: PairingSession) -> None:
        reach, tr = s.steps["reach"], s.steps["transport"]
        if reach.state != OK or tr.state != OK:
            self._finish(
                s,
                "config",
                BLOCKED,
                error="Reach and transport must both have run.",
                remedy="Run the earlier steps first.",
            )
            return
        view = RemoteView(**reach.detail)
        cfg = launch_config(view, tr.detail, local_node_id=_local_node_id())
        self._finish(s, "config", OK, detail=cfg)


def _local_node_id() -> str:
    try:
        from sglang.srt.rigmon.config import default_node_id

        return default_node_id()
    except Exception:  # pragma: no cover - defensive
        return "local"


#: Process-wide store. The dashboard and a CLI share it, which is the point:
#: a pairing started with curl is visible in the browser and vice versa.
STORE = PairingStore()
