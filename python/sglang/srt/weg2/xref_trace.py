"""XREF-TRACE (#1970, 05.10.2026): the x_refusal tail's two sides on ONE line form.

INSTRUMENT ONLY -- no decision reads anything in here. Default OFF
(``SGLANG_WEG2_XREF_TRACE``), dual layout only (front: ``dual_layout``; ranks:
``SGLANG_WEG2_DUAL_LAYOUT=1`` + group D). Off = no line, no probe, no lookup
beyond one ``os.environ.get``; a rank-local log write needs no vote.

THE QUESTION (deskq/done/1965-xrefusal-tail.md): the front prices a small
request ``short`` on an l3_index credit (an anchored page run in the shared
store), D refuses it with ``#1028B FETCH CAP kv=70 claimed=0`` /
``#1035c ZERO-ANSWER cause=CAPPED by=mamba`` (rid weg2-0-47), while the
siblings weg2-0-48/-49 (same length, same depth) read fine. Three candidates,
none proven: (1) the probe's and D's notion of "anchor present" differ,
(2) a concurrent sibling claimed/overlapped the same pages, (3) the anchor aged
out between the probe and the read. This line lets ONE boot separate them:

    WEG2 XREF-TRACE side=front|d stage=verdict|fetch|gate rid=<16> key0=<h12> keys=<n>
        kv=<pages> anchor_page=<idx|-1> anchor_key=<h12> claimed=<pages> lost=<pages>
        sib=<rid:claimed@age_ms,..|-> st=<s0/s1/s2|-> [extra k=v ...]

Both sides print the SAME fields in the SAME order (a missing value is ``-``):

* ``key0`` / ``anchor_key`` -- first 12 hex chars of the first page hash and of
  the hash of the anchor page. front and D equal = they asked the same stems;
  different = the probe and the read disagree on the KEYS (candidate 1).
* ``anchor_page`` -- index of the DEEPEST page carrying the trailing (mamba)
  anchor; front: ``pages-1`` of the L3-INDEX-PRESENCE depth, D: the deepest
  anchor the D read found among the KV pages it holds (-1 = none).
* ``claimed`` -- pages the side counts: front = anchored pages credited,
  D = pages the claim survived the component caps with.
* ``lost`` -- D: ``kv - claimed`` (what the caps took; the existing ``lost=`` of
  ``#1028B``). ``sib`` -- D: recent probes of the SAME first key (siblings) with
  their claimed pages and age; ``lost`` plus a sibling in ``sib`` that held the
  pages at that moment is candidate 2.
* ``st`` -- D: arena states of the mamba stems over the last 64 KV pages,
  ``absent/claimed/complete`` (0/1/2: ``claimed`` = another writer still holds
  the slot); ``-`` where the arena is not asked (file path).
"""
from __future__ import annotations

import collections
import os
import threading
import time
from typing import Any, Dict, Optional

ENV = "SGLANG_WEG2_XREF_TRACE"
LINE = "XREF-TRACE"
#: the one field order of both sides
FIELDS = ("side", "stage", "rid", "key0", "keys", "kv", "anchor_page", "anchor_key",
          "claimed", "lost", "sib", "st")
#: deepest-anchor scan bound on D (pages walked back from the end of the KV run)
SCAN_CAP = 4096
#: arena-state window (pages from the end of the KV run)
ST_WINDOW = 64
#: lines printed in full, then every 64th (a boot-bounded instrument)
LINE_CAP = 4000
SIB_WINDOW_S = 3.0

_RECENT: "collections.deque" = collections.deque(maxlen=256)
_FRONT: "collections.OrderedDict" = collections.OrderedDict()
_TL = threading.local()
_LOCK = threading.Lock()
_N = [0]


def switch_on(env=None) -> bool:
    """The env switch alone (the front has its own dual flag)."""
    e = os.environ if env is None else env
    return str(e.get(ENV, "") or "").strip().lower() in ("1", "true", "yes", "on")


def d_on(env=None) -> bool:
    """Switch on AND group D of the dual layout (the ranks' side)."""
    e = os.environ if env is None else env
    return (switch_on(e)
            and (e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() == "1"
            and (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "D")


def h12(key: Any) -> str:
    return "-" if not key else str(key)[:12]


def name_of(x: Any) -> str:
    """A pool name as plain text (``PoolName.MAMBA`` -> ``mamba``)."""
    return str(getattr(x, "value", x))


def _v(x: Any) -> str:
    return "-" if x is None or x == "" else name_of(x)


def fmt(side: str, stage: str, rid: Any, extra: Optional[Dict[str, Any]] = None, **kw: Any) -> str:
    """One XREF-TRACE line body. ``rid`` is cut to 16 chars on both sides (the
    X-GATE lines cut there), so a join on it needs no normalisation."""
    vals = dict(kw)
    vals.update(side=side, stage=stage, rid=str(rid)[:16] if rid is not None else None)
    parts = ["%s=%s" % (f, _v(vals.get(f))) for f in FIELDS]
    for k in sorted(extra or {}):
        parts.append("%s=%s" % (k, _v(extra[k])))
    return "WEG2 %s %s" % (LINE, " ".join(parts))


def emit_ok() -> bool:
    """Bounded output: the first LINE_CAP lines, then every 64th. Counts only
    lines that would be printed."""
    with _LOCK:
        _N[0] += 1
        n = _N[0]
    return n <= LINE_CAP or n % 64 == 0


# -- D side: the probe scope (rid for the storage layer) and the sibling ring ------------

def probe_begin(rid: Any, keys) -> Optional[tuple]:
    """Before ``batch_exists_v2`` on a D rank: the rid and first key become
    visible to the storage layer of this thread. None when off."""
    if not d_on():
        return None
    key0 = h12(keys[0]) if keys else "-"
    tok = (str(rid)[:16], key0, time.monotonic())
    _TL.cur = tok
    return tok


def probe_clear() -> None:
    _TL.cur = None


def current() -> Optional[tuple]:
    """(rid, key0, t0) of the probe running in this thread, or None."""
    return getattr(_TL, "cur", None)


def probe_done(tok: Optional[tuple], claimed: Any) -> None:
    """After the probe: the sibling ring learns this rid's outcome."""
    if tok is None:
        return
    with _LOCK:
        _RECENT.append((tok[0], tok[1], time.monotonic(), int(claimed or 0)))


def siblings(rid: str, key0: str, now: Optional[float] = None) -> str:
    """Other probes of the same first key within SIB_WINDOW_S: ``rid:claimed@age_ms``."""
    t = time.monotonic() if now is None else now
    with _LOCK:
        recent = list(_RECENT)
    out = ["%s:%d@%d" % (r, c, int((t - te) * 1000.0))
           for (r, k0, te, c) in recent
           if r != rid and k0 == key0 and t - te <= SIB_WINDOW_S]
    return ",".join(out[-6:]) or "-"


def state_hist(states, n_keys: int) -> str:
    """``absent/claimed/complete`` counts of the last ST_WINDOW arena states."""
    if states is None:
        return "-"
    try:
        win = list(states[max(0, int(n_keys) - ST_WINDOW):int(n_keys)])
    except Exception:  # noqa: BLE001 -- an instrument never raises into the read path
        return "-"
    return "%d/%d/%d" % (sum(1 for s in win if int(s) == 0), sum(1 for s in win if int(s) == 1),
                         sum(1 for s in win if int(s) == 2))


def deepest_anchor(has_component, name: str, kv_pages: int) -> int:
    """Deepest page index < kv_pages for which ``has_component(i, name)``,
    walked back from the end at most SCAN_CAP pages (-1 = none within the scan)."""
    lo = max(0, int(kv_pages) - SCAN_CAP)
    for i in range(int(kv_pages) - 1, lo - 1, -1):
        if has_component(i, name):
            return i
    return -1


# -- front side: the probe's answer, kept per rid until the ROUTE-VERDICT --------------

def note_front(rid: Any, depth: Any) -> None:
    """The L3 probe's Depth for ``rid`` (front, dual, switch on)."""
    with _LOCK:
        _FRONT[str(rid)] = depth
        while len(_FRONT) > 512:
            _FRONT.popitem(last=False)


def take_front(rid: Any) -> Any:
    with _LOCK:
        return _FRONT.pop(str(rid), None)


def front_verdict_line(rid: Any, depth: Any, verdict: Any, uncached: Any, credit_tok: Any,
                       src: Any) -> str:
    """The front's ROUTE-VERDICT-time line: what the probe credited, in D's fields."""
    pages = int(getattr(depth, "pages", 0) or 0) if depth is not None else None
    return fmt(
        "front", "verdict", rid,
        extra={"verdict": verdict, "uncached": uncached, "credit_tok": credit_tok, "src": src,
               "tier": getattr(depth, "tier", None) if depth is not None else None,
               "l3_pages": getattr(depth, "l3_pages", None) if depth is not None else None,
               "probe": "none" if depth is None else getattr(depth, "form", "-")},
        key0=(getattr(depth, "key0", "") or None) if depth is not None else None,
        keys=getattr(depth, "n_keys", None) if depth is not None else None,
        kv=getattr(depth, "kv_pages", None) if depth is not None else None,
        anchor_page=(pages - 1 if pages else -1) if pages is not None else None,
        anchor_key=(getattr(depth, "anchor_key", "") or None) if depth is not None else None,
        claimed=pages,
    )

