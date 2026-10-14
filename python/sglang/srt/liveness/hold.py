# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Requests the server holds ON PURPOSE, so their silence is not the consumer's.

LIVEPARK (14.10.): the LLM-stream watchdog measures time since the transport
last accepted a byte. That is the consumer's silence only while the producer
has something to send. A Weg-2 group D parks running requests on purpose --
the flip park (PARK-IMMEDIATE / wait bound), the lane park (``hold="lane"``),
the RESUME-VIA-P hold (D keeps the stream while P re-prefills), the #1443
dormant hold over D's sleep and the #1471 post-wake settle (the wake read) --
and a parked stream writes nothing for as long as the park lasts. The watchdog
read that as a dead client: NF boot ``..._1014_073255`` D.log Z.403857
``attachment stream-weg2-134-476 (llm_stream): grace -> dead after 90.8s``,
then ``abort_request`` and the front's 503 (front.log Z.27137), for a request
D had parked at 08:59:50, resumed at 09:00:45 and parked again at 09:01:02.

The same rule Kubernetes applies to a paused rollout (no progress-deadline
check while paused): a held request is not checked; its clock starts again at
the resumption, and a request that then makes no progress is reaped exactly
as before.

The scheduler is the only process that knows which requests it holds, and the
watchdog lives in the tokenizer/HTTP process. The scheduler publishes the WHOLE
held set whenever it changes (``Weg2LivenessHoldOutput``, replace semantics --
one lost or reordered notice heals at the next change); this registry is the
receiving side. No server ever sends a notice unless it holds something, so a
server without Weg-2 parks keeps the watchdog's behaviour byte for byte.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["HoldRegistry", "global_hold_registry"]


class HoldRegistry:
    """The set of request ids the server currently holds on purpose."""

    def __init__(self) -> None:
        self._held: frozenset[str] = frozenset()
        #: notices taken, for tests and the status line
        self.updates = 0

    def replace(self, rids: Iterable[str]) -> None:
        """Install the scheduler's whole held set (replace, never merge)."""
        self._held = frozenset(str(r) for r in rids if r)
        self.updates += 1

    def is_held(self, rid: str) -> bool:
        return str(rid) in self._held

    def any_held(self, rids: Iterable[str]) -> bool:
        held = self._held
        if not held:
            return False
        return any(str(r) in held for r in rids)

    def snapshot(self) -> frozenset[str]:
        return self._held


_GLOBAL_HOLDS = HoldRegistry()


def global_hold_registry() -> HoldRegistry:
    """The process-wide registry the scheduler's notices land in."""
    return _GLOBAL_HOLDS
