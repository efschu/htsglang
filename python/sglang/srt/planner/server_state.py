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
"""What state is the monitored server actually in -- four states, one probe
each, no guessing.

The defect this module exists to remove: the landing page used to render
"Server started without --enable-metrics" whenever the /metrics scrape failed
for ANY reason. A scrape that came back "connection refused" and a scrape that
came back 404 are TWO DIFFERENT STATES of the world -- a dead server and a
live server without the flag -- and the failure of a metrics scrape alone
cannot tell them apart. The old message picked one of the two causes and
asserted it. That is a guess printed as a diagnosis.

The discriminator is a SECOND probe against an endpoint that does not depend
on ``--enable-metrics``: the ordinary API surface (``/get_model_info``, then
``/health``). Only when THAT answers is "the server is up but serves no
metrics" a supported claim.

The four states, and what each one is allowed to say:

``NOT_RUNNING``
    Neither the API nor /metrics answers, and nothing is known to be booting.
    The honest sentence is "no server is running". Nothing may be said about
    which flags it was started with -- a dead server carries no evidence of
    its own launch command.

``STARTING``
    A boot is known to be in progress. Shown from the moment the dashboard
    knows about the boot, not after the first successful scrape. Evidence is
    NAMED (:class:`BootEvidence`) and is one of:
      * ``managed`` -- the dashboard launched this process itself and the
        supervisor reports ``booting``; the strongest form, known at launch;
      * ``port-open`` -- something accepts TCP on the target port while the
        API does not answer yet, i.e. a process is bound but not serving.
    Where neither holds, the state is NOT_RUNNING until the port answers.
    There is no heuristic "it is probably coming up".

``RUNNING_NO_METRICS``
    The API probe SUCCEEDED and the metrics scrape did not. This -- and only
    this -- is where the "started without --enable-metrics, rates unavailable,
    NVML keeps working" diagnosis is evidence-backed.

``RUNNING_WITH_METRICS``
    The scrape succeeded. /metrics is served by the same HTTP server as the
    API, so a successful scrape already proves the API is up; the API probe is
    skipped in this state rather than issued twice (recorded as
    ``attempted=False`` with its reason, not as a silent gap).

Structural guarantee: :func:`classify` reaches ``RUNNING_NO_METRICS`` only
inside the ``api.ok`` branch, so the old message cannot come back by accident.
``test_server_state.py`` pins that over the exhaustive input product.

All probes are BOUNDED (explicit ``timeout``); nothing in this module waits on
a condition, retries in a loop, or blocks a poll tick.
"""

from __future__ import annotations

import dataclasses
import socket
import urllib.error
import urllib.request
from typing import Callable, Optional, Sequence

__all__ = [
    "NOT_RUNNING",
    "STARTING",
    "RUNNING_NO_METRICS",
    "RUNNING_WITH_METRICS",
    "STATES",
    "API_PROBE_PATHS",
    "Probe",
    "BootEvidence",
    "ServerState",
    "classify",
    "build",
    "probe_http",
    "probe_api",
    "probe_metrics",
    "port_accepts_tcp",
    "boot_evidence",
    "resolve",
]

NOT_RUNNING = "not_running"
STARTING = "starting"
RUNNING_NO_METRICS = "running_no_metrics"
RUNNING_WITH_METRICS = "running_with_metrics"

#: Every state the landing page can be in, in ascending order of liveness.
STATES = (NOT_RUNNING, STARTING, RUNNING_NO_METRICS, RUNNING_WITH_METRICS)

#: The API probe paths, in order. Both are served regardless of
#: ``--enable-metrics``, which is the whole point: they answer the question
#: "is a server there at all" that /metrics cannot answer.
API_PROBE_PATHS = ("/get_model_info", "/health")

#: Supervisor states that mean "a boot this dashboard started is in progress".
#: Kept as data next to the state machine so the coupling to
#: ``server_manager._STATE_BOOTING`` is visible at one place.
MANAGED_BOOTING_STATES = ("booting",)


@dataclasses.dataclass(frozen=True)
class Probe:
    """One bounded HTTP probe, or an explicitly NOT-attempted one.

    ``attempted=False`` is a first-class result, not a missing value: it says
    the probe was skipped and ``reason`` says why. A skipped probe never reads
    as a failed one.
    """

    ok: bool
    path: str
    status: Optional[int] = None
    error: Optional[str] = None
    attempted: bool = True
    reason: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "path": self.path,
            "status": self.status,
            "error": self.error,
            "attempted": self.attempted,
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class BootEvidence:
    """Why we believe a boot is in progress -- or that we have no evidence.

    ``source`` is one of ``managed`` / ``port-open`` / ``none``. It is carried
    into the display so the reader sees WHAT the "starting" claim rests on;
    an unsourced "starting" is exactly the kind of guess this module removes.
    """

    starting: bool = False
    source: str = "none"
    detail: str = ""

    def to_json(self) -> dict:
        return {
            "starting": self.starting,
            "source": self.source,
            "detail": self.detail,
        }


@dataclasses.dataclass(frozen=True)
class ServerState:
    """The classified state plus the evidence that produced it."""

    state: str
    headline: str
    detail: str
    api: Probe
    metrics: Probe
    boot: BootEvidence

    @property
    def running(self) -> bool:
        """True in the two states that have a live server behind them."""
        return self.state in (RUNNING_NO_METRICS, RUNNING_WITH_METRICS)

    def to_json(self) -> dict:
        return {
            "state": self.state,
            "running": self.running,
            "headline": self.headline,
            "detail": self.detail,
            "api": self.api.to_json(),
            "metrics": self.metrics.to_json(),
            "boot": self.boot.to_json(),
        }


# ---------------------------------------------------------------------------
# The state machine itself (pure).
# ---------------------------------------------------------------------------
def classify(api: Probe, metrics: Probe, boot: BootEvidence) -> str:
    """Map the two probes plus the boot evidence onto exactly one state.

    Order matters and is the fix: the metrics probe decides only between the
    two RUNNING states, never between running and not running. The
    ``RUNNING_NO_METRICS`` return sits INSIDE the ``api.ok`` branch and
    nowhere else, so that state cannot be produced without a successful API
    probe -- which is exactly the claim the old message made without one.
    """
    if metrics.ok:
        return RUNNING_WITH_METRICS
    if api.ok:
        return RUNNING_NO_METRICS
    if boot.starting:
        return STARTING
    return NOT_RUNNING


def _texts(state: str, api: Probe, metrics: Probe, boot: BootEvidence):
    """Headline + detail for one state. The wording is part of the contract:
    the "--enable-metrics" sentence exists in exactly one branch.
    """
    if state == RUNNING_WITH_METRICS:
        return ("Server running", "API and /metrics both answer.")
    if state == RUNNING_NO_METRICS:
        return (
            "Server running without --enable-metrics",
            "The API answered on {} but /metrics did not ({}). Live rates "
            "(decode / prefill tok/s, per-request throughput, MTP acceptance, "
            "cache hit) are unavailable from this server; per-card VRAM, power "
            "and utilisation come from NVML and keep working. Restart with "
            "--enable-metrics.".format(api.path, metrics.error or "no answer"),
        )
    if state == STARTING:
        return (
            "Server starting",
            "Boot in progress ({}). {} Readings appear once the API "
            "answers.".format(boot.source, boot.detail).strip(),
        )
    return (
        "No server running",
        "Neither the API ({}) nor /metrics answers, and no boot is known to be "
        "in progress. Nothing can be said about how a server that is not there "
        "was started.".format(api.path),
    )


def build(api: Probe, metrics: Probe, boot: Optional[BootEvidence] = None) -> ServerState:
    """Classify + attach the wording. Pure; the tests drive this directly."""
    boot = boot or BootEvidence()
    state = classify(api, metrics, boot)
    headline, detail = _texts(state, api, metrics, boot)
    return ServerState(
        state=state,
        headline=headline,
        detail=detail,
        api=api,
        metrics=metrics,
        boot=boot,
    )


# ---------------------------------------------------------------------------
# Bounded probes.
# ---------------------------------------------------------------------------
def probe_http(base_url: str, path: str, timeout: float = 1.0,
               opener: Optional[Callable] = None) -> Probe:
    """One bounded GET. Never raises; an HTTP error code is a RESULT (with its
    status), a transport failure is a result with its error string.
    """
    url = base_url.rstrip("/") + path
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(url, timeout=timeout) as r:
            code = r.getcode()
            return Probe(ok=code == 200, path=path, status=code)
    except urllib.error.HTTPError as e:  # answered, but not 200
        return Probe(ok=False, path=path, status=e.code, error=f"HTTP {e.code}")
    except Exception as e:
        return Probe(ok=False, path=path, error=f"{type(e).__name__}: {e}")


def probe_api(base_url: str, timeout: float = 1.0,
              opener: Optional[Callable] = None,
              paths: Sequence[str] = API_PROBE_PATHS) -> Probe:
    """Is a server ANSWERING at all -- independent of ``--enable-metrics``.

    Tries the paths in order and returns the first 200. When none answers, the
    LAST result is returned so its error names a real attempt rather than a
    synthesised one.
    """
    last = Probe(ok=False, path=paths[0] if paths else "", error="no probe path")
    for p in paths:
        last = probe_http(base_url, p, timeout=timeout, opener=opener)
        if last.ok:
            return last
    return last


def probe_metrics(base_url: str, timeout: float = 1.0,
                  opener: Optional[Callable] = None) -> Probe:
    """The Prometheus scrape, as a probe result rather than an exception."""
    return probe_http(base_url, "/metrics", timeout=timeout, opener=opener)


def port_accepts_tcp(host: str, port: int, timeout: float = 0.15) -> bool:
    """Something is bound and accepting on host:port. Bounded by ``timeout``;
    a closed port refuses immediately."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def boot_evidence(
    *,
    managed_state: Optional[str] = None,
    managed_detail: str = "",
    host: Optional[str] = None,
    port: Optional[int] = None,
    tcp_probe: Optional[Callable[[str, int], bool]] = None,
    tcp_timeout: float = 0.15,
) -> BootEvidence:
    """Evidence that a boot is in progress -- managed first, port second.

    A dashboard-started boot is known the moment it is launched, so the
    supervisor's own state wins and no probing is needed. For a foreign boot
    the only evidence available without process scanning is that the port
    ACCEPTS TCP while the API is not answering yet: a process is bound but not
    serving. Neither present -> no evidence, and the caller must fall back to
    NOT_RUNNING rather than guess.
    """
    if managed_state in MANAGED_BOOTING_STATES:
        return BootEvidence(
            starting=True,
            source="managed",
            detail=managed_detail or "this dashboard launched the process.",
        )
    if host and port:
        probe = tcp_probe or (lambda h, p: port_accepts_tcp(h, p, timeout=tcp_timeout))
        if probe(host, int(port)):
            return BootEvidence(
                starting=True,
                source="port-open",
                detail=f"{host}:{port} accepts TCP but the API does not answer yet.",
            )
    return BootEvidence()


def resolve(
    base_url: Optional[str],
    *,
    metrics: Optional[Probe] = None,
    timeout: float = 1.0,
    opener: Optional[Callable] = None,
    managed_state: Optional[str] = None,
    managed_detail: str = "",
    host: Optional[str] = None,
    port: Optional[int] = None,
    tcp_probe: Optional[Callable[[str, int], bool]] = None,
) -> ServerState:
    """The whole decision for one poll tick.

    ``metrics`` may be passed in by a caller that already scraped (the landing
    poll does), so no second scrape is issued. The API probe is only spent
    when the scrape FAILED: /metrics and the API share one HTTP server, so a
    successful scrape already proves the API is up. That keeps the healthy
    path at exactly the one request it has always made.
    """
    if not base_url:
        api = Probe(ok=False, path="", attempted=False,
                    reason="no endpoint resolved")
        m = metrics or Probe(ok=False, path="/metrics", attempted=False,
                             reason="no endpoint resolved")
        return build(api, m, BootEvidence())

    m = metrics if metrics is not None else probe_metrics(
        base_url, timeout=timeout, opener=opener)
    if m.ok:
        api = Probe(
            ok=True,
            path="/metrics",
            attempted=False,
            reason="not needed: /metrics answered on the same HTTP server",
        )
        return build(api, m, BootEvidence())

    api = probe_api(base_url, timeout=timeout, opener=opener)
    boot = (
        BootEvidence()
        if api.ok
        else boot_evidence(
            managed_state=managed_state,
            managed_detail=managed_detail,
            host=host,
            port=port,
            tcp_probe=tcp_probe,
        )
    )
    return build(api, m, boot)
