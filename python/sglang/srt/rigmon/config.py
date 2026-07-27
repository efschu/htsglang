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
"""Configuration for the collector and the aggregator.

Three settings DESIGN_216 names as missing are first-class here: the sample
**resolution** cascade, the **port**, and the address of the aggregator a
second node pushes to.
"""

from __future__ import annotations

import dataclasses
import os
import socket
import uuid
from typing import List, Optional, Sequence, Tuple

from sglang.srt.rigmon.series import DEFAULT_TIERS, TierSpec, parse_tier_spec

__all__ = ["CollectorConfig", "AggregatorConfig", "default_node_id"]


def default_node_id() -> str:
    """Stable per-host identity: the hostname plus a short machine id, so two
    hosts that happen to share a hostname still separate."""
    host = socket.gethostname()
    mid = ""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            with open(path) as f:
                mid = f.read().strip()[:8]
                break
        except OSError:
            continue
    if not mid:
        mid = uuid.uuid4().hex[:8]
    return f"{host}-{mid}"


@dataclasses.dataclass
class CollectorConfig:
    """One node's sampler."""

    node_id: str = dataclasses.field(default_factory=default_node_id)
    #: Base cadence in seconds. Guaranteed server-side, which is the whole
    #: reason collection moved off the browser.
    interval_s: float = 1.0
    #: Read the expensive profiling counters every Nth tick (0 = never).
    profile_every: int = 10
    tiers: Tuple[TierSpec, ...] = DEFAULT_TIERS
    #: Local engine to join with (blank disables the engine side).
    engine_url: str = "http://127.0.0.1:30000"
    #: Where to PUSH to. Empty = this node only serves locally.
    #: Push is outbound by design: the second node then needs no inbound
    #: firewall rule, which is what makes joining a rig a one-click step.
    aggregator_url: str = ""
    push_token: str = ""
    push_every_s: float = 5.0
    #: Local read-only API (0 disables; a pure push node needs no listener).
    listen_host: str = "127.0.0.1"
    listen_port: int = 0
    #: Optional boot log to classify.
    boot_log: str = ""
    #: Refresh the capability table every N seconds (probes are not free).
    capability_refresh_s: float = 60.0

    @staticmethod
    def tiers_from_strings(items: Sequence[str]) -> Tuple[TierSpec, ...]:
        return tuple(parse_tier_spec(i) for i in items)


@dataclasses.dataclass
class AggregatorConfig:
    """The read side. Serves the UI and ingests pushes from other nodes."""

    host: str = "127.0.0.1"
    port: int = 8770
    #: Token every pushing node must present. Empty means "accept unauthenticated
    #: pushes", which is only sane on a loopback bind and is refused otherwise.
    token: str = ""
    tiers: Tuple[TierSpec, ...] = DEFAULT_TIERS
    #: Drop a node from the display after this long without a push.
    node_stale_s: float = 30.0
    #: Lifetime of a pairing token handed out for a join.
    join_token_ttl_s: float = 600.0
    state_dir: str = os.path.expanduser("~/.cache/sglang/rigmon")

    def validate(self) -> List[str]:
        """Configuration errors, as a list rather than an exception, so a CLI
        can print all of them at once."""
        errs = []
        if not self.token and self.host not in ("127.0.0.1", "localhost", "::1"):
            errs.append(
                f"--host {self.host} exposes the aggregator beyond loopback but "
                "no --token is set: pushes would be unauthenticated. Set a token "
                "or bind to 127.0.0.1."
            )
        if not (0 < self.port < 65536):
            errs.append(f"--port {self.port} is out of range")
        return errs
