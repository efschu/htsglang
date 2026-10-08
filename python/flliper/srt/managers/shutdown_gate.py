"""#840: make SIGTERM converge -- refuse the refill, and bound the drain.

THE DEFECT. ``TokenizerManager.sigterm_watchdog`` drains in-flight requests
before it asks the schedulers to exit. That drain had no deadline and no
admission gate, so it could be kept alive forever by traffic that arrived
AFTER the shutdown began. Specimen
``/spinning/evidence-665-f1/window4A_teardown_hang_2114/``, 2026-08-23 21:15:

    21:15:21  Remaining number of requests 4   [fa8d, ed7c, fcc9, fab8]
    21:15:26  Remaining number of requests 3   [fa8d,       fcc9, fab8]
    21:15:31  Remaining number of requests 4   [fa8d,       fcc9, fab8, 873e]
    21:15:36  Remaining number of requests 3   [fa8d,       fcc9, fab8]

``873e...`` is admitted ten seconds into the shutdown. The count is not
converging; it is being refilled.

WHY THAT HOLDS THE GPUs. ``ShutdownReq`` is dispatched to the schedulers only
after the drain loop breaks. A scheduler checks ``gracefully_exit`` once per
event-loop iteration and sets it only when that request is dequeued, so a drain
that never breaks means a scheduler that never leaves its loop -- and the
``finally`` that runs ``release_distributed`` (#673, the thing
``--scheduler-distributed-teardown`` arms) is downstream of exactly that. The
observed consequence is the one the window protocol pays for: SIGTERM to the
parent frees nothing, and the cards come back only after an explicit TERM to
each rank PID.

TWO HALVES, and neither is sufficient alone. Bounding the drain without the
gate turns every shutdown under load into a timeout, discarding requests that
would have finished. Gating admissions without the bound leaves a request that
genuinely never completes able to hold the instance forever. Together they give
the property the shutdown was always supposed to have: it converges, and it
converges quickly, and the deadline is a backstop rather than the mechanism.

This module is deliberately dependency-free so both halves can be tested
without constructing a server.
"""

from __future__ import annotations

from typing import Any

# The drain loop's poll cadence, in seconds. Named here rather than written as
# a literal at the call site so the shutdown suite can drive many drain ticks
# without sleeping minutes of wall clock.
DRAIN_POLL_INTERVAL_S = 5.0

# How long the drain may run before it stops waiting, in seconds.
#
# Chosen to be longer than any request this server answers in practice and far
# shorter than "forever". The gate above is what makes an ordinary shutdown
# finish in seconds; this is only reached when something in flight will not
# complete, which before #840 was an unbounded hang.
DEFAULT_DRAIN_TIMEOUT_S = 120.0


class ServerShuttingDown(ValueError):
    """A request arrived after SIGTERM started the drain (#840).

    ``ValueError`` by inheritance ON PURPOSE. Every request entrypoint in this
    tree already has an ``except ValueError`` arm, so this refusal reaches the
    client as a proper response on every route the day it lands -- including
    routes added later that have never heard of this type. Entrypoints that DO
    know about it upgrade the status to 503 (Service Unavailable, which is what
    a shutting-down server owes a load balancer); the rest still refuse, with
    400. Refusing with the wrong status is a cosmetic defect. NOT refusing is
    the defect this class exists for, and it is the one that costs the cards.
    """


def drain_timeout_s(server_args: Any) -> float:
    """The drain budget in seconds; ``0.0`` means the pre-#840 unbounded loop.

    ``getattr`` with a default rather than attribute access: this is read on
    the shutdown path, where an older or partial ``ServerArgs`` must degrade to
    the shipped default instead of raising and leaving the instance up.

    An explicit ``0`` survives as ``0`` -- it is the documented bisecting mode,
    and a falsy-default here would silently turn it into the bounded one, which
    is precisely the kind of substitution that makes a bisect lie. A negative
    value means the same thing as zero; there is no wait shorter than none.
    """
    value = getattr(server_args, "shutdown_drain_timeout_s", None)
    if value is None:
        return float(DEFAULT_DRAIN_TIMEOUT_S)
    value = float(value)
    return value if value > 0.0 else 0.0
