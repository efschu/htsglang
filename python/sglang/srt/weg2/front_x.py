"""FRONT-X-MIT (15.10.): the front's decision X travels with the D leg, and D's
X gate checks the REALISED uncached extent against THAT X -- by name, never silently.

THE GAP (NF boot 124245, h16p 226aad7105, report nf-core-dx-1015)
------------------------------------------------------------------
The front prices a D-direct request with a DYNAMIC X (X_k / X_busy / X_1, 1010..6733
in that boot).  D's own gate (``scheduler._weg2_x_refuses``) only knew the STATIC
``--tp-prefill-max-tokens`` (12288).  A price that was too LOW (the front credited a
presence span deeper than D's resume anchor: ``rid weg2-40-127`` priced 237, D computed
1517) was therefore computed on D without a word, as long as it stayed under 12288 --
no violation in that boot, but the edge of the safeguard: the law "no D-direct prefill
over X, an overrun never silent" was only enforced at the static ceiling.

THE RULE
--------
1. The front hands the X of its dispatch decision (and the uncached amount it priced
   at that decision) to D in two request headers, on the D-direct leg 2 only (no leg 1
   ran: a request D reads back from P's hand-off prefills nothing and carries no X).
2. D's X gate refuses (W31, the same named 503 / the same return to P as above the
   static X) when the realised uncached extent exceeds ``front_x + tolerance``.  The
   static X stays the hard ceiling: ``limit = min(static_x, front_x + tolerance)``.
3. Applies to the FIRST admission of a request only (no output yet).  A D-park resume
   re-admits a request that already decoded: its extent is the resume's, not the price
   the front put on the prompt, and the static X stays its bound (resume-via-P owns it).

TOLERANCE = ONE PAGE (``server_args.page_size``, 64 on NF)
---------------------------------------------------------
The only SYSTEMATIC difference between the front's token-exact price and D's realised
extent is the page rounding of the matched prefix: D reuses whole pages, the front
credits a token span.  The rounding is always in the direction "D computes a little
more" and is strictly below one page.  So a request the front priced exactly at its X
must not be refused for rounding (that would bounce it D -> P -> D around the
threshold: W31 re-routes it, the flip costs seconds, the next pricing lands on the
same side again).  One page is the smallest tolerance that makes the check exact
for the systematic part and loud for everything else.  One CHUNK (``chunked_prefill_size``,
thousands of tokens) would swallow the very mispricings this exists to name
(weg2-40-127: +1280, weg2-24-60: +512, weg2-38-105: +320 over the price).

GROUP UNIFORMITY (MUST NOT 7)
-----------------------------
Both numbers are fields of the request: they cross the HTTP hop once, on TP0's
tokenizer manager, and reach every rank inside the ``TokenizedGenerateReqInput`` that
rank 0 broadcasts (``request_receiver.recv_requests``).  No rank-local quantity enters
the limit (``page_size`` is a launch argument, equal on every rank), so the verdict is
the group's like the static one it refines.

SWITCH
------
``SGLANG_WEG2_FRONT_X_CARRY`` (default ON) on the front AND on D.  Off on the front =
no header, D = the static gate byte for byte; off on D = headers ignored.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

#: The X of the front's dispatch decision for this D leg (tokens).
HEADER_X = "x-weg2-front-x"
#: What the front priced at that decision (uncached tokens D must prefill).
HEADER_PRICED = "x-weg2-front-priced"

#: Request attribute names (GenerateReqInput / TokenizedGenerateReqInput / Req).
ATTR_X = "weg2_front_x"
ATTR_PRICED = "weg2_front_priced"

#: Upper bound accepted from a header: a defensive sanity cap, far above any X.
_MAX_TOKENS = 1 << 31


def enabled() -> bool:
    """Both ends read the same switch; default ON (user order 14.10.)."""
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_FRONT_X_CARRY.get())
    except Exception:  # noqa: BLE001 - a missing registry means "default": on
        return True


def tolerance_tokens(page_size: Optional[int]) -> int:
    """The rounding tolerance: ONE page (see the module docstring)."""
    try:
        return max(1, int(page_size or 1))
    except (TypeError, ValueError):
        return 1


def request_headers(x: Optional[int], priced: Optional[int]) -> Dict[str, str]:
    """The headers the front adds to a D-direct leg 2; ``{}`` = nothing to carry.

    No header for ``x <= 0`` (no decision X: the static gate alone) and none with the
    switch off."""
    try:
        xi = int(x) if x is not None else 0
    except (TypeError, ValueError):
        return {}
    if xi <= 0 or not enabled():
        return {}
    out = {HEADER_X: str(xi)}
    if priced is not None:
        try:
            out[HEADER_PRICED] = str(max(0, int(priced)))
        except (TypeError, ValueError):
            pass
    return out


def _parse_one(headers: Mapping[str, Any], name: str) -> Optional[int]:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        v = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("WEG2 FRONT-X header %s=%r is not an integer -- ignored (the static X gate stands)", name, raw)
        return None
    if v < 0 or v > _MAX_TOKENS:
        logger.warning("WEG2 FRONT-X header %s=%r out of range -- ignored (the static X gate stands)", name, raw)
        return None
    return v


def parse_headers(headers: Optional[Mapping[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
    """``(front_x, front_priced)`` of a request, ``(None, None)`` when absent/invalid/off.

    A zero or malformed X is treated as ABSENT -- never as "refuse everything"."""
    if headers is None or not enabled():
        return None, None
    try:
        x = _parse_one(headers, HEADER_X)
        priced = _parse_one(headers, HEADER_PRICED)
    except Exception:  # noqa: BLE001 - a header parser may never break admission
        return None, None
    if not x:
        return None, None
    return x, priced


def stamp_request(adapted: Any, raw_request: Any) -> None:
    """Copy the carrier headers of ``raw_request`` onto the internal request object.

    Called from the OpenAI serving layer's ``_convert_to_internal_request`` (chat and
    completions; the Anthropic ``/v1/messages`` handler converts through chat, so it is
    covered).  A request without the headers is untouched: its attributes stay ``None``
    and the static gate is the only gate."""
    headers = getattr(raw_request, "headers", None)
    x, priced = parse_headers(headers)
    if x is None:
        return
    try:
        setattr(adapted, ATTR_X, x)
        setattr(adapted, ATTR_PRICED, priced)
    except Exception:  # noqa: BLE001
        logger.warning("WEG2 FRONT-X could not stamp %s on %s", ATTR_X, type(adapted).__name__)


def limit_for(static_x: int, front_x: Optional[int], page_size: Optional[int]) -> int:
    """The refusal limit of one request: ``min(static_x, front_x + one page)``.

    ``static_x <= 0`` means the gate is off and stays off (returned unchanged)."""
    sx = int(static_x or 0)
    if sx <= 0 or front_x is None:
        return sx
    return min(sx, int(front_x) + tolerance_tokens(page_size))


def request_front_x(req: Any) -> Tuple[Optional[int], Optional[int]]:
    """``(front_x, front_priced)`` of a scheduler ``Req``, or ``(None, None)``.

    ONLY for a request that has produced no output yet (the first admission): a
    resumed request is bounded by the static X (module docstring, rule 3)."""
    try:
        if getattr(req, "output_ids", None):
            return None, None
        x = getattr(req, ATTR_X, None)
        if x is None or int(x) <= 0:
            return None, None
        p = getattr(req, ATTR_PRICED, None)
        return int(x), (None if p is None else int(p))
    except Exception:  # noqa: BLE001
        return None, None


def resolve_limit(req: Any, static_x: int, page_size: Optional[int]) -> Tuple[int, Optional[int], Optional[int]]:
    """``(limit, front_x, front_priced)`` of one request at D's gate: ``limit = min(static_x,
    front_x + one page)`` when the request carries the front's decision X (first admission
    only), else the static X. A module function (not a Scheduler method) so a partial
    scheduler double reaches it too. Never raises: any fault is "nothing carried"."""
    try:
        if not enabled():
            return int(static_x), None, None
        fx, priced = request_front_x(req)
        if fx is None:
            return int(static_x), None, None
        return limit_for(static_x, fx, page_size), fx, priced
    except Exception:  # noqa: BLE001 - a limit helper may never break admission
        return int(static_x), None, None
