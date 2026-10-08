"""HANDOFF-LOST front seam (#243, NF rc12r pdflip-12-39 / pdflip-13-44; 27.09.).

THE CLASS. P prefills a request completely and publishes its hand-off (END-ANCHOR
ok, TAIL-PUBLISH); the rid then waits for a D seat -- 255 s on NF rc12r with the
6 seats taken. In that time a flip's P reset gives the arena refs back (#1427
RESET-RELEASE) and H81 lets the Mamba end anchors go ``at=wake``; the pages are
overwritten, and at the D admission ``#1028B FETCH CAP lost=1196`` / ``#928 no
recurrent state`` / ``X-GATE uncached=76602`` refuse it mid-stream. The hand-off's
life was tied to phases, not to its consumption.

THE SPLIT (agreed 27B <-> NF, 27.09.). NF's side (``flliper.srt.pdflip.handoff_pending``):
a hand-off is a pending mark from P's publish until D consumes it; every claim
evicts pending pages LAST; an overflow is NAMED -- ``HANDOFF-LOST`` plus
``<arena>/handoff/lost/<rid>.json``. Its two calls never raise:

* ``status(rid)`` -> ``{state: pending|lost|none, first_lost_page, pages, page_size}``
* ``drop(rid, reason)``

THIS SIDE (the front): a rid whose leg 1 ran and that waits for a D seat is
checked each controller pass and at the admitter before the seat. ``lost``: the
credit is ``first_lost_page * page_size`` (what is still whole in front of the
first lost page); ``uncached = prompt - credit``. Over X the rid goes back to P
at once -- ``PDFLIP HANDOFF-LOST-REROUTE ... path=fresh-P`` -- instead of spending
the rest of its wait on a hand-off that can no longer serve it. At or under X D
prefills the lost tail itself (law 4 allows it) and nothing moves. The front
drops the rid's marks at every rid end (served, abort, disconnect, reroute,
W35), so no orphaned mark reads as a later false HANDOFF-LOST.

GUARDED: without NF's module (an image built before it) or when it raises, the
state is ``none`` -- the front behaves exactly as before. Switch
``FLLIPER_PDFLIP_HANDOFF_LOST_REROUTE`` (default on; ``0`` = no status reads, no
reroute; the drop at rid end stays, it is bookkeeping only).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: The one NF module this seam reads; a module attribute so a test can point it
#: at a fake without touching sys.modules.
MODULE = "flliper.srt.pdflip.handoff_pending"

_NONE: Dict[str, Any] = {"state": "none", "first_lost_page": 0, "pages": 0, "page_size": 0}
#: import cache: unset -> not tried yet; None -> the import failed (not retried:
#: the image does not grow a module at runtime, and a failed import re-scans
#: sys.path on every call -- every 0.2 s per waiting rid).
_UNSET = object()
_mod: Any = _UNSET


class PdFlipHandoffLost(Exception):
    """A rid whose hand-off was lost a second time over X: its future fails with
    this, and handle_generate answers it with a named 503 (no stream opened)."""


def enabled(env=None) -> bool:
    """``FLLIPER_PDFLIP_HANDOFF_LOST_REROUTE`` (default on; 0 = off)."""
    e = os.environ if env is None else env
    raw = (e.get("FLLIPER_PDFLIP_HANDOFF_LOST_REROUTE", "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _module() -> Any:
    global _mod
    if _mod is _UNSET:
        try:
            import importlib

            _mod = importlib.import_module(MODULE)
        except Exception as e:  # noqa: BLE001 -- absent module = state none
            logger.info("HANDOFF-SEAM: %s not importable (%s) -- every status reads none",
                        MODULE, type(e).__name__)
            _mod = None
    return _mod


def reset_module_cache() -> None:
    """Tests: forget the cached import."""
    global _mod
    _mod = _UNSET


def status(rid: str) -> Dict[str, Any]:
    """NF's ``status(rid)``, normalised; ``none`` on a missing module, a raise
    or an answer of the wrong shape (never a guess: an unreadable answer is not
    a loss)."""
    m = _module()
    fn = getattr(m, "status", None) if m is not None else None
    if fn is None:
        return dict(_NONE)
    try:
        raw = fn(rid)
    except Exception as e:  # noqa: BLE001
        logger.warning("HANDOFF-SEAM status(%s) raised %s -- read as none", rid, type(e).__name__)
        return dict(_NONE)
    if not isinstance(raw, dict):
        return dict(_NONE)
    state = str(raw.get("state", "none") or "none")
    if state not in ("pending", "lost", "none"):
        return dict(_NONE)
    out = {"state": state}
    for k in ("first_lost_page", "pages", "page_size"):
        try:
            out[k] = max(0, int(raw.get(k, 0) or 0))
        except (TypeError, ValueError):
            out[k] = 0
    return out


def drop(rid: str, reason: str) -> None:
    """NF's ``drop(rid, reason)``; silent without the module, never raises."""
    m = _module()
    fn = getattr(m, "drop", None) if m is not None else None
    if fn is None:
        return
    try:
        fn(rid, reason)
    except Exception as e:  # noqa: BLE001
        logger.warning("HANDOFF-SEAM drop(%s, %s) raised %s", rid, reason, type(e).__name__)


def lost_terms(st: Dict[str, Any], prompt_tokens: int, x: int) -> Optional[Tuple[int, int, bool]]:
    """``(credit, uncached, over_x)`` for a ``lost`` status, else None.

    ``credit = first_lost_page * page_size``: the pages before the first lost
    one are still whole. A status without ``page_size`` credits NOTHING (the
    front does not assume a page size; NF's status carries it) -- the whole
    prompt is then uncached, which is the conservative reading of "lost"."""
    if st.get("state") != "lost":
        return None
    credit = int(st.get("first_lost_page", 0)) * int(st.get("page_size", 0))
    prompt = max(0, int(prompt_tokens))
    credit = min(credit, prompt)
    uncached = prompt - credit
    return credit, uncached, uncached > int(x)


def end_reason(status_code: Optional[int], exc: Optional[BaseException]) -> str:
    """The ``drop`` reason of a front handler's end."""
    if exc is not None:
        import asyncio

        if isinstance(exc, (asyncio.CancelledError, ConnectionResetError)):
            return "disconnect"
        return f"abort:{type(exc).__name__}"
    if status_code is None:
        return "end"
    return "served" if int(status_code) < 400 else f"status_{int(status_code)}"


#: aiohttp Request key under which handle_generate leaves the rid it assigned.
RID_KEY = "pdflip_rid"


def note_request_rid(request: Any, rid: str) -> None:
    """Remember the rid on the aiohttp request (a MutableMapping) so the
    rid-end wrapper can drop its marks; a test double without item assignment
    is left alone."""
    try:
        request[RID_KEY] = rid
    except Exception:  # noqa: BLE001
        pass


def wrap_handler(handler):
    """The ONE rid-end site: every return, exception and cancel of the wrapped
    front handler drops the rid's hand-off marks (served, 4xx/5xx incl. W35,
    abort, client disconnect). A request that never got a rid (refused before
    the routing) drops nothing."""
    import functools

    @functools.wraps(handler)
    async def _wrapped(request):
        resp = None
        exc: Optional[BaseException] = None
        try:
            resp = await handler(request)
            return resp
        except BaseException as e:  # noqa: BLE001 -- re-raised below
            exc = e
            raise
        finally:
            rid = None
            try:
                rid = request.get(RID_KEY)
            except Exception:  # noqa: BLE001
                rid = None
            if rid:
                drop(str(rid), end_reason(getattr(resp, "status", None), exc))
    return _wrapped
