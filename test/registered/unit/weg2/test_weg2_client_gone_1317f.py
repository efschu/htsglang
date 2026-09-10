"""#1317f -- a request whose client left is aborted on D and on P.

SPECIMEN, boot weg2sn6e: the probe timed out client-side at 240 s and D kept
decoding for NINE more minutes -- 686 `Decode batch` lines, `#full token`
8,814 -> 84,384 -- for a receiver that no longer existed, while the drain
refused three windows on it. Nothing aborted the rid, because nothing was
watching the connection.

RULING 2026-09-10: aborted on both groups. Not a #1011 question -- that law
protects a decode someone is waiting for; a decode with no receiver is
wasted D time that blocks the flip and the P backlog. Upstream sglang
aborts on disconnect too.

MUTANTS, and the danger direction here is a FALSE abort (killing a request
someone IS waiting for), so they push that way:
  M1  a completed request must never trigger it
      -> test_a_completed_request_is_never_gone
  M2  a SERVER fault must never be blamed on the client
      -> test_a_server_fault_is_not_a_client_disconnect
  M3  an unreadable transport must never read as gone
      -> test_an_unreadable_transport_is_never_gone
  M4  W36 keeps its own case and must not double-fire
      -> test_w36_still_owns_the_never_posted_case
"""

import asyncio


def _gone(raised=None, closing=None, completed=False):
    from sglang.srt.weg2.front import weg2_client_is_gone

    return weg2_client_is_gone(raised, closing, completed)


# --------------------------------------------------------------------------
# the predicate: every shape the four endpoints can produce
# --------------------------------------------------------------------------

def test_the_streaming_shape_a_write_to_a_dropped_socket():
    assert _gone(raised=ConnectionResetError()) is True
    assert _gone(raised=BrokenPipeError()) is True


def test_the_streaming_shape_aiohttp_cancels_the_handler():
    """aiohttp cancels the handler task when the client disconnects, so the
    disconnect surfaces as CancelledError rather than as a write error."""
    assert _gone(raised=asyncio.CancelledError()) is True


def test_the_aiohttp_client_side_reset_names():
    """The reset arrives under several class names depending on where in the
    stack it is raised; matching on the name is deliberate, because importing
    every aiohttp exception here would couple this test to its layout."""
    for name in ("ClientConnectionResetError", "ClientOSError",
                 "ClientConnectionError"):
        exc = type(name, (Exception,), {})()
        assert _gone(raised=exc) is True, name


def test_the_non_stream_shape_nothing_raises_so_the_transport_decides():
    """Non-stream writes nothing until the end, so no exception is raised and
    the only evidence is the transport."""
    assert _gone(closing=True) is True
    assert _gone(closing=False) is False


# --------------------------------------------------------------------------
# the mutants: a FALSE abort is the danger direction
# --------------------------------------------------------------------------

def test_a_completed_request_is_never_gone():
    """M1, THE FALSE POSITIVE THAT MATTERS MOST. A well-behaved non-streaming
    client closes the socket the instant it has its answer, so `closing` is
    True on almost every SUCCESSFUL request. Without the `completed` guard the
    front would abort a rid it had just served."""
    assert _gone(closing=True, completed=True) is False
    assert _gone(raised=ConnectionResetError(), completed=True) is False


def test_a_server_fault_is_not_a_client_disconnect():
    """M2. Blaming the client for a server bug would abort the request AND
    hide the bug behind a CLIENT-GONE line."""
    for exc in (ValueError("bad payload"), KeyError("k"), RuntimeError("boom")):
        assert _gone(raised=exc) is False, type(exc).__name__


def test_an_unreadable_transport_is_never_gone():
    """M3. None means "cannot say", and an abort may never rest on that."""
    assert _gone(closing=None) is False


def test_w36_still_owns_the_never_posted_case():
    """M4. W36 fires from the admitter on a Pending whose client never reached
    its POST; this decides a request that DID reach the handler and minted a
    rid. Disjoint by construction -- and the middleware can only fire when
    `weg2_rid` was stashed, which happens inside the handler."""
    import inspect

    from sglang.srt.weg2 import front

    w36 = inspect.getsource(front.Front.d_admitter)
    assert "W36_Weg2AdmitterBarrierExpired" in w36
    assert "weg2_client_gone" not in w36, (
        "W36's site must not also raise a CLIENT-GONE -- one event, one owner"
    )
    mw = inspect.getsource(front.main)
    assert 'request.get("weg2_rid")' in mw, (
        "the middleware must fire only for a request that minted a rid"
    )


# --------------------------------------------------------------------------
# the abort itself: both groups, the line, the seat
# --------------------------------------------------------------------------

def test_the_abort_targets_both_groups_and_names_which():
    import inspect

    from sglang.srt.weg2.front import Front

    src = inspect.getsource(Front.weg2_client_gone)
    assert 'for name in ("D", "P")' in src, (
        "a request can be mid-leg-1 on P when its client leaves; aborting only "
        "D would leave P prefilling for nobody"
    )
    assert "/abort_request" in src
    assert "WEG2 CLIENT-GONE" in src
    for term in ("rid=%s", "aborted_on=%s", "elapsed=%.1fs"):
        assert term in src, f"the line lost {term}"
    assert 'release("client_gone")' in src, "the seat must be released"


def test_the_abort_never_raises():
    """It runs in the `finally` of a request that is already failing; an
    exception there would replace a client disconnect with a 500 on a
    connection nobody is reading."""
    import inspect

    from sglang.srt.weg2.front import Front

    src = inspect.getsource(Front.weg2_client_gone)
    assert src.count("except Exception") >= 1
    assert "NEVER RAISES" in src


def test_the_middleware_shields_the_abort():
    """The abort is awaited inside the `finally` of a request that is usually
    being CANCELLED. Un-shielded, the await is cancelled immediately: the
    abort would be issued and never sent -- a fix that logs and does nothing."""
    import inspect

    from sglang.srt.weg2 import front

    src = inspect.getsource(front.main)
    assert "asyncio.shield(" in src
    i_sh = src.index("asyncio.shield(")
    i_cg = src.index("weg2_client_gone(")
    assert i_sh < i_cg


def test_the_middleware_is_installed_on_the_app():
    import inspect

    from sglang.srt.weg2 import front

    src = inspect.getsource(front.main)
    assert "middlewares=[weg2_client_gone_middleware]" in src, (
        "an uninstalled middleware is the desk-written-never-executed class"
    )


def test_the_middleware_reraises_so_the_response_is_unchanged():
    """Detection may not swallow the original outcome: a cancelled request
    must still cancel, and a server error must still surface."""
    import inspect

    from sglang.srt.weg2 import front

    src = inspect.getsource(front.main)
    i_ex = src.index("except BaseException as e:")
    tail = src[i_ex:i_ex + 400]
    assert "raise" in tail, "the middleware must re-raise, always"
