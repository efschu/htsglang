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
"""The aggregator: multi-node store plus the read-only API the UI consumes.

The web UI is a **reader of a stored series**. It never touches hardware, never
opens NVML, never scrapes an engine. Every number it shows was sampled by a
collector on the host that owns the device — which is the only place tok/s and
card state can be joined, and the only place that can see a second rig's GPUs
at all.

Two node paths meet here:

* the **local** collector (same process, in-memory);
* **remote** collectors, which PUSH. Outbound-only, so the remote rig needs no
  inbound firewall rule.

Joining is a two-step handshake: the aggregator mints a short-lived pairing
token, the joining node redeems it once for a long-lived node token. The
pairing token is what a user copies across; it expires, so a leaked one is not
a standing grant.

Secrets stay server-side. Node tokens are held here and are never included in
any response the browser can read — the UI is given node *identity*, never node
*credentials*.
"""

from __future__ import annotations

import dataclasses
import hmac
import json
import logging
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from sglang.srt.rigmon.config import AggregatorConfig
from sglang.srt.rigmon.series import Point, TimeSeries

logger = logging.getLogger(__name__)

__all__ = [
    "NodeState",
    "Aggregator",
    "CompatibilityRefused",
    "make_handler",
    "serve",
]


class CompatibilityRefused(Exception):
    """The join gate blocked the pairing. Carries the full report so the
    joining side gets the reasons, not just a refusal."""

    def __init__(self, report):
        self.report = report
        blockers = [c for c in report.checks if c.verdict == "block"]
        super().__init__(
            "compatibility gate refused the join: "
            + "; ".join(f"{c.label}: {c.detail}" for c in blockers)
        )

#: Refuse absurd pushes outright rather than letting one node exhaust memory.
MAX_PUSH_BYTES = 8 * 1024 * 1024


@dataclasses.dataclass
class NodeState:
    node_id: str
    series: TimeSeries
    last_push: float = 0.0
    last_meta: Dict[str, Any] = dataclasses.field(default_factory=dict)
    remote: bool = True
    address: Optional[str] = None
    points_received: int = 0

    def age_s(self, now: float) -> float:
        return now - self.last_push if self.last_push else float("inf")


class Aggregator:
    """The store. HTTP is a thin shell over this; everything here is testable
    without a socket."""

    def __init__(
        self,
        config: Optional[AggregatorConfig] = None,
        clock=time.time,
    ):
        self.config = config or AggregatorConfig()
        self.clock = clock
        self._lock = threading.RLock()
        self._nodes: Dict[str, NodeState] = {}
        self._local = None  # Optional[Collector]
        #: node_id -> token. Server-side only; never serialised to a client.
        self._node_tokens: Dict[str, str] = {}
        #: pairing token -> expiry
        self._join_tokens: Dict[str, float] = {}
        #: node_id -> CompatReport from its join.
        self._compat: Dict[str, Any] = {}
        #: This node's identity, set by the CLI when a model is declared.
        self.local_identity = None

    # -- local node ---------------------------------------------------------

    def attach_local(self, collector) -> None:
        """Register the in-process collector. Its series is used directly — a
        local node needs no push."""
        with self._lock:
            self._local = collector
            self._nodes[collector.config.node_id] = NodeState(
                node_id=collector.config.node_id,
                series=collector.series,
                last_push=self.clock(),
                remote=False,
                address="local",
            )

    def touch_local(self) -> None:
        with self._lock:
            if self._local is not None:
                st = self._nodes.get(self._local.config.node_id)
                if st is not None:
                    st.last_push = self.clock()
                    st.last_meta = self._local.snapshot()

    # -- pairing ------------------------------------------------------------

    def mint_join_token(self) -> Dict[str, Any]:
        """A short-lived pairing token. This is the string a user carries to
        the second machine; it is single-use and expires."""
        tok = secrets.token_urlsafe(18)
        with self._lock:
            self._join_tokens[tok] = self.clock() + self.config.join_token_ttl_s
        return {
            "join_token": tok,
            "expires_in_s": self.config.join_token_ttl_s,
            "expires_at": self.clock() + self.config.join_token_ttl_s,
        }

    def redeem_join_token(
        self,
        token: str,
        node_id: str,
        identity: Optional[dict] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Exchange a pairing token for this node's long-lived push token.

        The compatibility gate runs HERE, before the two sides are joined,
        because every check it performs otherwise surfaces much later — as a
        worker crash, or as a silent quality difference. A blocking verdict
        refuses the join and returns the reasons; ``force`` overrides it, which
        exists because a user who understands the mismatch may still want
        telemetry from the other node.
        """
        now = self.clock()
        with self._lock:
            for t, exp in list(self._join_tokens.items()):
                if exp < now:
                    del self._join_tokens[t]
            exp = self._join_tokens.get(token)
            if exp is None:
                raise PermissionError("unknown or already-used pairing token")

        report = None
        if identity is not None and self.local_identity is not None:
            from sglang.srt.rigmon.compat import NodeIdentity, check_compatibility

            report = check_compatibility(
                self.local_identity, NodeIdentity.from_json(identity)
            )
            if report.blocked and not force:
                raise CompatibilityRefused(report)

        with self._lock:
            # Consume the token only once the gate has passed, so a refused
            # join can be retried after the mismatch is fixed.
            if token not in self._join_tokens:
                raise PermissionError("pairing token was consumed concurrently")
            del self._join_tokens[token]
            node_token = secrets.token_urlsafe(24)
            self._node_tokens[node_id] = node_token
            self._compat[node_id] = report
            self._persist_tokens()
        out: Dict[str, Any] = {"node_id": node_id, "push_token": node_token}
        if report is not None:
            out["compatibility"] = report.to_json()
            out["forced"] = bool(force and report.blocked)
        return out

    def compatibility(self, node_id: str) -> Optional[dict]:
        with self._lock:
            r = self._compat.get(node_id)
        return r.to_json() if r is not None else None

    def _token_path(self) -> str:
        return os.path.join(self.config.state_dir, "node_tokens.json")

    def _persist_tokens(self) -> None:
        try:
            os.makedirs(self.config.state_dir, exist_ok=True)
            path = self._token_path()
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._node_tokens, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("rigmon: could not persist node tokens: %s", e)

    def load_tokens(self) -> None:
        try:
            with open(self._token_path()) as f:
                data = json.load(f)
            if isinstance(data, dict):
                with self._lock:
                    self._node_tokens.update({str(k): str(v) for k, v in data.items()})
        except (OSError, ValueError):
            pass

    def check_token(self, node_id: str, token: str) -> bool:
        with self._lock:
            expected = self._node_tokens.get(node_id) or self.config.token
        if not expected:
            # No token configured at all: only tolerable on loopback, which
            # AggregatorConfig.validate() enforces at startup.
            return True
        return hmac.compare_digest(str(expected), str(token or ""))

    # -- ingest -------------------------------------------------------------

    def ingest(self, payload: Dict[str, Any], address: Optional[str] = None) -> dict:
        node_id = str(payload.get("node_id") or "").strip()
        if not node_id:
            raise ValueError("push payload has no node_id")
        now = self.clock()
        with self._lock:
            st = self._nodes.get(node_id)
            if st is None:
                st = NodeState(
                    node_id=node_id,
                    series=TimeSeries(self.config.tiers),
                    remote=True,
                    address=address,
                )
                self._nodes[node_id] = st
            if not st.remote:
                raise ValueError(
                    f"node_id {node_id!r} collides with the local node; give the "
                    "remote collector a distinct --node-id"
                )
        accepted = 0
        skipped: List[str] = []
        for tier_name, points in (payload.get("points") or {}).items():
            try:
                accepted += st.series.ingest_points(
                    tier_name, [Point.from_json(p) for p in points]
                )
            except KeyError:
                # The remote runs a different resolution cascade. Its buckets
                # cannot be placed without lying about their width, so they are
                # refused with a name, not silently dropped.
                skipped.append(tier_name)
        with self._lock:
            st.last_push = now
            st.last_meta = payload.get("meta") or {}
            st.address = address or st.address
            st.points_received += accepted
        return {
            "accepted": accepted,
            "skipped_tiers": skipped,
            "warning": (
                f"resolutions {skipped} are not configured on this aggregator; "
                "run both sides with the same --resolution set"
                if skipped
                else None
            ),
        }

    # -- read ---------------------------------------------------------------

    def nodes(self) -> List[dict]:
        now = self.clock()
        self.touch_local()
        with self._lock:
            out = []
            for st in self._nodes.values():
                age = st.age_s(now)
                out.append(
                    {
                        "node_id": st.node_id,
                        "remote": st.remote,
                        "address": st.address,
                        "last_push": st.last_push,
                        "age_s": None if age == float("inf") else round(age, 2),
                        "stale": age > self.config.node_stale_s,
                        # Only meaningful for a pushing node; the local one
                        # writes straight into its series.
                        "points_received": (
                            st.points_received if st.remote else None
                        ),
                        "resolutions": st.series.resolutions(),
                        "compatibility": (
                            self._compat[st.node_id].to_json()
                            if self._compat.get(st.node_id) is not None
                            else None
                        ),
                    }
                )
            return sorted(out, key=lambda d: (d["remote"], d["node_id"]))

    def snapshot(self) -> dict:
        """Current state of every node. Node tokens are deliberately absent."""
        now = self.clock()
        self.touch_local()
        with self._lock:
            nodes = {}
            for st in self._nodes.values():
                meta = dict(st.last_meta)
                meta.pop("push_token", None)
                age = st.age_s(now)
                nodes[st.node_id] = {
                    "remote": st.remote,
                    "stale": age > self.config.node_stale_s,
                    "age_s": None if age == float("inf") else round(age, 2),
                    "state": meta,
                }
        return {"ts": now, "nodes": nodes, "stale_after_s": self.config.node_stale_s}

    def query(
        self,
        node_id: Optional[str] = None,
        resolution: Optional[str] = None,
        window_s: Optional[float] = None,
        keys: Optional[List[str]] = None,
        max_points: int = 600,
    ) -> dict:
        with self._lock:
            targets = (
                [self._nodes[node_id]]
                if node_id
                else list(self._nodes.values())
            ) if not node_id or node_id in self._nodes else []
        if node_id and not targets:
            raise KeyError(f"unknown node {node_id!r}")
        out = {}
        for st in targets:
            out[st.node_id] = st.series.query(
                resolution=resolution,
                window_s=window_s,
                keys=keys,
                max_points=max_points,
                now=self.clock(),
            )
        return {"nodes": out}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def make_handler(agg: Aggregator, static_dir: Optional[str] = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rigmon"

        def _send(self, code: int, body, ctype: str = "application/json"):
            data = body if isinstance(body, bytes) else str(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except BrokenPipeError:
                pass

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj).encode())

        # -- GET ------------------------------------------------------------

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)

            def one(name, cast=str, default=None):
                v = q.get(name)
                if not v:
                    return default
                try:
                    return cast(v[0])
                except (TypeError, ValueError):
                    return default

            try:
                if u.path == "/api/nodes":
                    return self._json(200, {"nodes": agg.nodes()})
                if u.path == "/api/snapshot":
                    return self._json(200, agg.snapshot())
                if u.path == "/api/series":
                    return self._json(
                        200,
                        agg.query(
                            node_id=one("node"),
                            resolution=one("resolution"),
                            window_s=one("window", float),
                            keys=(q.get("keys")[0].split(",") if q.get("keys") else None),
                            max_points=one("max_points", int, 600) or 600,
                        ),
                    )
                if u.path == "/api/join_token":
                    # Minting a pairing token is an administrative action, so it
                    # is loopback-only regardless of bind address.
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        return self._json(
                            403,
                            {"error": "join tokens can only be minted from localhost"},
                        )
                    return self._json(200, agg.mint_join_token())
                if u.path in ("/", "/index.html") and static_dir:
                    path = os.path.join(static_dir, "index.html")
                    if os.path.isfile(path):
                        with open(path, "rb") as f:
                            return self._send(200, f.read(), "text/html; charset=utf-8")
                    return self._send(404, b"no UI installed", "text/plain")
                return self._json(404, {"error": f"no such path {u.path}"})
            except KeyError as e:
                return self._json(404, {"error": str(e)})
            except Exception as e:
                logger.exception("rigmon: GET %s failed", u.path)
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        # -- POST -----------------------------------------------------------

        def do_POST(self):  # noqa: N802
            u = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._json(400, {"error": "bad Content-Length"})
            if length > MAX_PUSH_BYTES:
                return self._json(
                    413, {"error": f"payload exceeds {MAX_PUSH_BYTES} bytes"}
                )
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8", "replace"))
            except ValueError as e:
                return self._json(400, {"error": f"malformed JSON: {e}"})
            try:
                if u.path == "/api/join":
                    try:
                        return self._json(
                            200,
                            agg.redeem_join_token(
                                str(payload.get("join_token") or ""),
                                str(payload.get("node_id") or ""),
                                identity=payload.get("identity"),
                                force=bool(payload.get("force")),
                            ),
                        )
                    except CompatibilityRefused as e:
                        # 409, not 403: the credential was fine, the two sides
                        # are not. The report is the actionable part.
                        return self._json(
                            409,
                            {
                                "error": str(e),
                                "compatibility": e.report.to_json(),
                                "hint": "fix the blocking checks, or retry with "
                                "force=true to join anyway",
                            },
                        )
                if u.path == "/api/push":
                    node_id = str(payload.get("node_id") or "")
                    token = self.headers.get("X-Rigmon-Token") or ""
                    if not agg.check_token(node_id, token):
                        return self._json(
                            401,
                            {
                                "error": "invalid push token for node "
                                f"{node_id!r}; re-run the join handshake"
                            },
                        )
                    return self._json(
                        200, agg.ingest(payload, address=self.client_address[0])
                    )
                return self._json(404, {"error": f"no such path {u.path}"})
            except PermissionError as e:
                return self._json(403, {"error": str(e)})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            except Exception as e:
                logger.exception("rigmon: POST %s failed", u.path)
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):  # quiet
            pass

    return Handler


def serve(
    agg: Aggregator, static_dir: Optional[str] = None
) -> ThreadingHTTPServer:
    errs = agg.config.validate()
    if errs:
        raise ValueError("; ".join(errs))
    agg.load_tokens()
    srv = ThreadingHTTPServer(
        (agg.config.host, agg.config.port), make_handler(agg, static_dir)
    )
    return srv
