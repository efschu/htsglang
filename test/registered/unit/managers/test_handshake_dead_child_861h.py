# SPDX-License-Identifier: Apache-2.0
"""#861h: waiting for a handshake from a dead peer is a hang, not a wait.

THE SPECIMEN, py-spy'd 2026-08-25. `test_dirty_session_refuses_to_run` sat in
`ScriptedHttpServer._await_handshake` at 0 % CPU while its spawned server child
(pid 2502190) was already <defunct> -- dead at startup, never reaped. The call
had a DEADLINE and no LIVENESS check, so it waited out the full timeout for a
handshake that could never arrive. "Aktiv aber tot", in the test harness.

CLASS: a bounded wait whose bound is the only thing it checks. A deadline
answers "how long", never "is the thing I am waiting for still there". The
sibling rule: any wait on a peer must end the moment the peer does, and must
say WHY.

MEASURED AFTER THE FIX: the same test now raises in ~15 s carrying the child's
exit code, instead of holding the run.
"""

import time

import pytest

from sglang.test.scripted_runtime.http_server import ScriptedHttpServer


class DeadChild:
    """A child that has already exited. `multiprocessing.Process` API."""

    pid = 4242

    def __init__(self, exitcode=1):
        self.exitcode = exitcode

    def is_alive(self):
        return False


class LiveSilentChild:
    """Alive, but never speaks -- the case the DEADLINE must still cover."""

    pid = 4243
    exitcode = None

    def is_alive(self):
        return True


class NeverReadySocket:
    def poll(self, timeout_ms):
        time.sleep(min(timeout_ms, 50) / 1000.0)
        return False


def _session(child, stderr="boom: No accelerator available"):
    s = object.__new__(ScriptedHttpServer)
    s._socket = NeverReadySocket()
    s._server_process = child
    s._drain_child_stderr = lambda: stderr
    return s


def test_a_dead_child_raises_immediately_not_after_the_deadline():
    """THE FIX. Seconds, not the full timeout -- and the exit code is named."""
    s = _session(DeadChild(exitcode=1))
    t0 = time.monotonic()
    with pytest.raises(RuntimeError) as ctx:
        ScriptedHttpServer._await_handshake(s)
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f"took {elapsed:.1f}s -- it waited out the deadline"
    msg = str(ctx.value)
    assert "EXITED with code 1" in msg
    assert "4242" in msg, "the child's pid must be named"


def test_the_raise_carries_the_childs_own_output():
    """A dead child without WHY moves the hang one question later."""
    s = _session(DeadChild(), stderr="No accelerator (CUDA, XPU, ...) available")
    with pytest.raises(RuntimeError, match="No accelerator"):
        ScriptedHttpServer._await_handshake(s)


def test_a_live_but_silent_child_still_hits_the_deadline():
    """BOTH DIRECTIONS. The liveness check must not swallow the timeout: a peer
    that is alive and mute is a different failure and keeps its own message."""
    import sglang.test.scripted_runtime.http_server as mod

    original = mod.LISTENER_ACCEPT_TIMEOUT_S
    mod.LISTENER_ACCEPT_TIMEOUT_S = 0.5
    try:
        s = _session(LiveSilentChild())
        with pytest.raises(TimeoutError, match="still alive"):
            ScriptedHttpServer._await_handshake(s)
    finally:
        mod.LISTENER_ACCEPT_TIMEOUT_S = original


def test_the_liveness_accessor_matches_the_rest_of_the_file():
    """SELF-CAUGHT BUG, pinned. The first cut used `.poll()`; the object is a
    `multiprocessing.Process`, so it would have AttributeError'd on the very
    failure path it was written for. `execute_script` already used
    `is_alive()`; one idiom per file."""
    import inspect

    src = inspect.getsource(ScriptedHttpServer._await_handshake)
    assert "is_alive()" in src
    assert "_server_process.poll()" not in src
