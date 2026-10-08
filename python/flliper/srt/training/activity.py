# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The serving process's own answer to "is anything being served right now".

The registry knows when an engine was last *acquired*, which is the rig-wide
signal and the one DESIGN #341 D4 names. It is not the only one that matters:
a server that has held a hot engine for an hour and is answering a request
this second looks identical to an idle one from the outside, because
acquisition already happened.

So the process stamps a float on every inbound generation request. One store
to a module global, no lock -- a torn read of a float that is being
monotonically advanced cannot produce a value that is *older* than the true
one, and older is the safe direction: it makes the rig look busier, never
idler.
"""

from __future__ import annotations

import time

_last_activity_ts: float = 0.0
_request_count: int = 0


def note_serving_activity() -> None:
    """Called on every inbound generation request. Must stay this cheap."""
    global _last_activity_ts, _request_count
    _last_activity_ts = time.time()
    _request_count += 1


def last_activity_ts() -> float:
    """Wall-clock of the most recent request, or ``0.0`` if there was none."""
    return _last_activity_ts


def request_count() -> int:
    return _request_count


def reset() -> None:
    """Tests only."""
    global _last_activity_ts, _request_count
    _last_activity_ts = 0.0
    _request_count = 0
