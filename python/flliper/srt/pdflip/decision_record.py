"""EZ-S1 (plan nf-entscheidungszentrale-plan-1019, step S1, SOLL R11): the ONE decision record.

Every place of the front that decides about route, flip, park, hold, abort, admission or class writes ONE line of
the same shape, so a reader (the S71 monitors, logindex, S4's checker) reads one format instead of fifty log
sentences::

    WEG2 DECISION seq=<n> epoch=<e> kind=<kind> action=<ACTION> rule=<Rn|NS:name|Gate-x|ALT:marker> set=<id|->
         rid=<rid|-> members=<rid:u/s_p@src/age_ms,...|-> S=<int> S_P=<int> X=<int> halfX=<int> classes=<..|->
         d_decoding=<0|1> seats_free=<int>@<src> kv_free=<int|->@<src>/<age_ms|-> r6_rest=<int> unstarted=<int>
         t_half_s=<f> waited_s=<f> waited_on=<..|-> numbers_age_ms_max=<int> t_arrive_to_decide_ms=<int> why=<text>
    WEG2 DECISION-DONE seq=<n> rid=<rid> executor=<front|D|P|PP0> outcome=<ok|deferred|refused|mismatch>
         measured_u=<int|-> src=<..> detail=<text>
    WEG2 DECISION-SUM window_s=<int> sites=<k> n=<total> lines=<L> by=<kind|rule|action>=<n>/<lines>;...

(One physical line each; the lines above are wrapped for reading only.)

SHADOW MODE (S1): a decision that is still taken by the OLD code is recorded with ``rule=ALT:<marker>`` (the marker
names the old site: ``dwell_ok``, ``flip_economics_ok`` ...). The next boot then counts, by ``ALT:*``, how many
decisions are not central yet -- the yardstick for S2a..S3b. The shadow changes NOTHING: the record is written after
the old site returned its value, an exception of the record never reaches the caller.

LOG VOLUME (plan A15): sites that the controller passes every 0.2 s are EDGE-TRIGGERED -- a line only when
``(action, members)`` of that site (and rid) changes against its last line; every repeat is counted and reported
once per window in ``DECISION-SUM`` (1000 identical ticks = 1 DECISION line + ``=1000/1`` in the next SUM).

``SET-VERDICT`` (SOLL section 4) is derived from the same data by :func:`set_verdict_line`: byte-identical to the line
the front has written since H91 (golden test), so every monitor and the SOLL tests read it unchanged.
"""

from __future__ import annotations

import collections
import functools
import inspect
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: the line prefixes (a reader greps these)
DECISION_PREFIX = "WEG2 DECISION"
DONE_PREFIX = "WEG2 DECISION-DONE"
SUM_PREFIX = "WEG2 DECISION-SUM"

#: window of one DECISION-SUM line (seconds)
SUM_WINDOW_S = 60.0

#: kinds / outcomes / executors (the checker of S4 reads these closed sets)
KINDS = ("admit", "arrival", "set", "seat", "park", "reroute", "measure", "phase_end", "idle", "boot", "manual",
         "abort", "alt")
OUTCOMES = ("ok", "deferred", "refused", "mismatch")
EXECUTORS = ("front", "D", "P", "PP0")

RULE_ALT_PREFIX = "ALT:"
RULE_NS_PREFIX = "NS:"


def _tok(s: Any) -> str:
    """One whitespace-free token (``-`` for nothing)."""
    t = re.sub(r"\s+", "_", str(s if s is not None else "").strip())
    return t or "-"


def _text(s: Any) -> str:
    """The free-text tail: one line."""
    return re.sub(r"[\r\n]+", " ", str(s if s is not None else "")).strip()


@dataclass(frozen=True)
class Member:
    """One member of a decision: its OWN u (decides class and S), its S_P share (report only), the source and age of
    the number (plan A4/A16)."""
    rid: str
    u_own: int
    s_p: int = 0
    src: str = "-"
    age_ms: int = 0

    def fmt(self) -> str:
        return "%s:%d/%d@%s/%d" % (_tok(self.rid), int(self.u_own), int(self.s_p), _tok(self.src), int(self.age_ms))


@dataclass(frozen=True)
class Decision:
    kind: str
    action: str
    rule: str
    why: str = ""
    epoch: int = 0
    seq: int = 0
    set_id: Optional[int] = None
    rid: Optional[str] = None
    members: Tuple[Member, ...] = ()
    s_tokens: int = 0
    s_p_tokens: int = 0
    x: int = 0
    half_x: int = 0
    classes: Tuple[str, ...] = ()
    d_decoding: bool = False
    seats_free: int = 0
    seats_src: str = "-"
    kv_free: Optional[int] = None
    kv_src: str = "absent"
    kv_age_ms: Optional[int] = None
    r6_rest: int = 0
    unstarted: int = 0
    t_half_s: float = 0.0
    waited_s: float = 0.0
    waited_on: str = "-"
    numbers_age_ms_max: int = 0
    t_arrive_to_decide_ms: int = 0

    def line(self) -> str:
        return ("%s seq=%d epoch=%d kind=%s action=%s rule=%s set=%s rid=%s members=%s S=%d S_P=%d X=%d halfX=%d "
                "classes=%s d_decoding=%d seats_free=%d@%s kv_free=%s@%s/%s r6_rest=%d unstarted=%d t_half_s=%.1f "
                "waited_s=%.1f waited_on=%s numbers_age_ms_max=%d t_arrive_to_decide_ms=%d why=%s" % (
                    DECISION_PREFIX, int(self.seq), int(self.epoch), _tok(self.kind), _tok(self.action),
                    _tok(self.rule), "-" if self.set_id is None else int(self.set_id), _tok(self.rid or "-"),
                    ",".join(m.fmt() for m in self.members) or "-", int(self.s_tokens), int(self.s_p_tokens),
                    int(self.x), int(self.half_x), ",".join(_tok(c) for c in self.classes) or "-",
                    int(bool(self.d_decoding)), int(self.seats_free), _tok(self.seats_src),
                    "-" if self.kv_free is None else int(self.kv_free), _tok(self.kv_src),
                    "-" if self.kv_age_ms is None else int(self.kv_age_ms), int(self.r6_rest), int(self.unstarted),
                    float(self.t_half_s), float(self.waited_s), _tok(self.waited_on),
                    int(self.numbers_age_ms_max), int(self.t_arrive_to_decide_ms), _text(self.why)))


def done_line(seq: int, rid: Optional[str], executor: str, outcome: str, measured_u: Optional[int] = None,
              src: str = "-", detail: str = "") -> str:
    return "%s seq=%d rid=%s executor=%s outcome=%s measured_u=%s src=%s detail=%s" % (
        DONE_PREFIX, int(seq), _tok(rid or "-"), _tok(executor), _tok(outcome),
        "-" if measured_u is None else int(measured_u), _tok(src), _text(detail))


# ---------------------------------------------------------------------------------------------------------------
# parser (ONE parser for nf_scharf, logindex and the S4 checker)
# ---------------------------------------------------------------------------------------------------------------

_MEMBER_RE = re.compile(r"^(?P<rid>[^:,\s]+):(?P<u>-?\d+)/(?P<s_p>-?\d+)@(?P<src>[^/,\s]+)/(?P<age>-?\d+)$")
_KV_RE = re.compile(r"(?P<k>[A-Za-z_][A-Za-z0-9_]*)=(?P<v>\S*)")


def parse_members(text: str) -> List[Member]:
    out: List[Member] = []
    if not text or text == "-":
        return out
    for part in text.split(","):
        m = _MEMBER_RE.match(part)
        if m is None:
            raise ValueError("bad DECISION member %r" % part)
        out.append(Member(m["rid"], int(m["u"]), int(m["s_p"]), m["src"], int(m["age"])))
    return out


def parse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a ``WEG2 DECISION`` / ``DECISION-DONE`` / ``DECISION-SUM`` line to a dict (``None``: not one of them).
    ``why`` / ``detail`` take the rest of the line. Raises ``ValueError`` on a malformed line of these kinds."""
    i = line.find("WEG2 DECISION")
    if i < 0:
        return None
    s = line[i:]
    head, _, _ = s.partition(" ")
    # the head is "WEG2" -- the tag is the second token
    toks = s.split(" ", 2)
    if len(toks) < 3:
        raise ValueError("short DECISION line")
    tag = toks[1]
    rest = toks[2]
    if tag not in ("DECISION", "DECISION-DONE", "DECISION-SUM"):
        return None
    tail_key = {"DECISION": "why", "DECISION-DONE": "detail", "DECISION-SUM": None}[tag]
    tail = ""
    if tail_key is not None:
        j = rest.find(" %s=" % tail_key)
        if j < 0:
            raise ValueError("DECISION line without %s=" % tail_key)
        tail = rest[j + len(tail_key) + 2:]
        rest = rest[:j]
    out: Dict[str, Any] = {"tag": tag}
    for m in _KV_RE.finditer(rest):
        out[m["k"]] = m["v"]
    if tail_key is not None:
        out[tail_key] = tail
    if tag == "DECISION":
        for k in ("seq", "epoch", "S", "S_P", "X", "halfX", "r6_rest", "unstarted", "numbers_age_ms_max",
                  "t_arrive_to_decide_ms", "d_decoding"):
            if k not in out:
                raise ValueError("DECISION line without %s=" % k)
            out[k] = int(out[k])
        out["members"] = parse_members(out.get("members", "-"))
        out["classes"] = [] if out.get("classes", "-") == "-" else out["classes"].split(",")
        out["t_half_s"] = float(out["t_half_s"])
        out["waited_s"] = float(out["waited_s"])
        sf = re.match(r"^(-?\d+)@(\S+)$", out.get("seats_free", ""))
        if sf is None:
            raise ValueError("bad seats_free")
        out["seats_free"], out["seats_src"] = int(sf[1]), sf[2]
        kv = re.match(r"^(-|-?\d+)@([^/\s]+)/(-|-?\d+)$", out.get("kv_free", ""))
        if kv is None:
            raise ValueError("bad kv_free")
        out["kv_free"] = None if kv[1] == "-" else int(kv[1])
        out["kv_src"] = kv[2]
        out["kv_age_ms"] = None if kv[3] == "-" else int(kv[3])
        out["set"] = None if out.get("set", "-") == "-" else int(out["set"])
    elif tag == "DECISION-DONE":
        out["seq"] = int(out["seq"])
        out["measured_u"] = None if out.get("measured_u", "-") == "-" else int(out["measured_u"])
    else:
        for k in ("window_s", "sites", "n", "lines"):
            out[k] = int(out[k])
        by: Dict[str, Tuple[int, int]] = {}
        for part in filter(None, out.get("by", "").split(";")):
            key, _, nums = part.rpartition("=")
            a, _, b = nums.partition("/")
            by[key] = (int(a), int(b))
        out["by"] = by
    return out


# ---------------------------------------------------------------------------------------------------------------
# SET-VERDICT (SOLL section 4), derived -- byte-identical to the H91 line
# ---------------------------------------------------------------------------------------------------------------

def set_verdict_line(*, epoch: int, set_id: int, members: Sequence[Tuple[str, int]], s_tokens: int, x_tok: int,
                     half_x: int, classes: Sequence[str], d_decoding: bool, seats_free: int, t_half: float,
                     waited: float, verdict: str, why: str) -> str:
    """The ``PDFLIP SET-VERDICT`` line of SOLL section 4, the exact format. ``members`` = ``(rid, own u)``."""
    return ("PDFLIP SET-VERDICT epoch=%d set=%d n=%d members=%s S=%d X=%d halfX=%d classes=%s d_decoding=%d "
            "seats_free=%d t_half_s=%.1f waited_s=%.1f verdict=%s why=%s" % (
                epoch, set_id, len(members), ",".join("%s:%d" % (r, u) for r, u in members), s_tokens, x_tok,
                half_x, ",".join(classes), int(d_decoding), seats_free, t_half, waited, verdict, why))


_RULE_IN_WHY = re.compile(r"\b(R[1-9]a?|R1[01]|R2a)\b")


def rule_of_set_why(why: str, verdict: str) -> str:
    """The rule tag of a SET-VERDICT's ``why`` (``R2``, ``R4`` ...); a verdict whose why names none (``yield_or_gate``,
    ``d_budget``: the front's own words) keeps the front's word as an ALT-free note ``R?``."""
    if (why or "").startswith("NS:S49"):
        return "NS:S49"          # EZ-S2a plan 2.6: the granted-not-started hold carries its own tag
    m = _RULE_IN_WHY.search(why or "")
    if m:
        return m.group(1)
    return "R2" if verdict == "D_NOW" else "NS:set-" + _tok(why.split(" ")[0] if why else verdict).lower()


# ---------------------------------------------------------------------------------------------------------------
# the recorder
# ---------------------------------------------------------------------------------------------------------------

class Recorder:
    """Writes DECISION / DECISION-DONE / DECISION-SUM, edge-triggers repeating sites, counts everything.

    One per front. ``log`` is a ``logging.Logger``; ``epoch_of`` gives the current epoch; ``clock`` is the monotonic
    clock (injectable for tests). Never raises into its caller."""

    EDGE_KEPT = 8192

    def __init__(self, log: logging.Logger, epoch_of: Callable[[], int] = lambda: 0,
                 clock: Callable[[], float] = time.monotonic, sum_window_s: float = SUM_WINDOW_S):
        self.log = log
        self.epoch_of = epoch_of
        self.clock = clock
        self.sum_window_s = float(sum_window_s)
        self.seq = 0
        self._edge: "collections.OrderedDict[Tuple[str, str, str], Tuple[str, str]]" = collections.OrderedDict()
        self._n: Dict[str, int] = collections.defaultdict(int)       # by-key: every call
        self._lines: Dict[str, int] = collections.defaultdict(int)   # by-key: lines written
        self._t_sum = clock()
        self.lines_total = 0
        self.suppressed_total = 0

    # -- DECISION --------------------------------------------------------------------------------------------------
    def record(self, d: Decision, *, edge: bool = False) -> int:
        """Write ``d`` (its ``seq`` / ``epoch`` are assigned here). ``edge=True``: only when ``(action, members)`` of
        ``(kind, rule, rid)`` changed against the last line of that site. Returns the seq (0: suppressed)."""
        try:
            key = "%s|%s|%s" % (d.kind, d.rule, d.action)
            self._n[key] += 1
            if edge:
                site = (d.kind, d.rule, d.rid or "-")
                sig = (d.action, ",".join(m.fmt() for m in d.members))
                last = self._edge.get(site)
                if last == sig:
                    self._edge.move_to_end(site)
                    self.suppressed_total += 1
                    self._tick()
                    return 0
                self._edge[site] = sig
                self._edge.move_to_end(site)
                while len(self._edge) > self.EDGE_KEPT:
                    self._edge.popitem(last=False)
            self.seq += 1
            line = Decision(**{**d.__dict__, "seq": self.seq, "epoch": int(self.epoch_of())}).line()
            self._lines[key] += 1
            self.lines_total += 1
            self.log.info("%s", line)
            self._tick()
            return self.seq
        except Exception:  # noqa: BLE001 -- an instrument never breaks a decision
            return 0

    def alt(self, marker: str, *, kind: str, action: str, rid: Optional[str] = None, why: str = "",
            members: Sequence[Member] = (), edge: bool = True, **fields: Any) -> int:
        """A shadow record of a decision the OLD code took (``rule=ALT:<marker>``)."""
        return self.record(Decision(kind=kind, action=_tok(action), rule=RULE_ALT_PREFIX + marker, why=why, rid=rid,
                                    members=tuple(members), **fields), edge=edge)

    # -- DECISION-DONE ---------------------------------------------------------------------------------------------
    def done(self, seq: int, rid: Optional[str], executor: str, outcome: str, *, measured_u: Optional[int] = None,
             src: str = "-", detail: str = "") -> None:
        try:
            self._n["DONE|%s|%s" % (executor, outcome)] += 1
            self._lines["DONE|%s|%s" % (executor, outcome)] += 1
            self.log.info("%s", done_line(seq, rid, executor, outcome, measured_u, src, detail))
        except Exception:  # noqa: BLE001
            pass

    # -- DECISION-SUM ----------------------------------------------------------------------------------------------
    def _tick(self) -> None:
        now = self.clock()
        if now - self._t_sum >= self.sum_window_s:
            self.flush_sum(now)

    def tick(self) -> None:
        """Called by the controller: the window closes even when no decision was recorded."""
        try:
            self._tick()
        except Exception:  # noqa: BLE001
            pass

    def flush_sum(self, now: Optional[float] = None) -> Optional[str]:
        """Write the window's summary (one line) and reset; ``None`` when nothing was counted."""
        now = self.clock() if now is None else now
        window = max(1, int(round(now - self._t_sum)))
        self._t_sum = now
        if not self._n:
            return None
        by = ";".join("%s=%d/%d" % (k, self._n[k], self._lines.get(k, 0)) for k in sorted(self._n))
        line = "%s window_s=%d sites=%d n=%d lines=%d by=%s" % (
            SUM_PREFIX, window, len(self._n), sum(self._n.values()), sum(self._lines.values()), by)
        self._n = collections.defaultdict(int)
        self._lines = collections.defaultdict(int)
        self.log.info("%s", line)
        return line


# ---------------------------------------------------------------------------------------------------------------
# ALT shadow installation (the old sites of the front, one table, no behaviour change)
# ---------------------------------------------------------------------------------------------------------------

def summarize(result: Any) -> str:
    """A whitespace-free token of what an old site returned (the shadow's ``action``)."""
    if result is None:
        return "none"
    if isinstance(result, bool):
        return "true" if result else "false"
    if isinstance(result, (int, float)):
        return str(result)
    if isinstance(result, str):
        return _tok(result)[:48]
    if isinstance(result, tuple) and result and isinstance(result[0], bool):
        # EZ-S1 review B7: ``(bool, why, ...)`` is a yes/no DECISION -- the shadow shows the bool, so the flank fires on a
        # changed decision, not on every call
        return "true" if result[0] else "false"
    v = getattr(result, "verdict", None)
    if isinstance(v, str) and v:
        return _tok(v)[:48]  # a ``phase_policy.SetVerdict`` (and any verdict object): the decision it carries
    if isinstance(result, (tuple, list, set, frozenset, dict)):
        return "n%d" % len(result)
    return type(result).__name__


def install_alt_shadow(cls: type, table: Mapping[str, Tuple[str, str]],
                       recorder_of: Callable[[Any], Optional[Recorder]]) -> List[str]:
    """Wrap each method of ``table = {marker: (method_name, kind)}`` of ``cls`` so that every call writes one
    ``ALT:<marker>`` record AFTER the old code returned. Sync and async methods; the result and the exceptions pass
    through untouched. Returns the markers that could NOT be installed (missing / not a plain function): the test of
    the table demands ``[]``."""
    missing: List[str] = []
    for marker, (name, kind) in table.items():
        orig = inspect.getattr_static(cls, name, None)
        if orig is None or not inspect.isfunction(orig) or getattr(orig, "_ez_alt_marker", None) is not None:
            if orig is None or not inspect.isfunction(orig):
                missing.append(marker)
            continue
        try:
            params = list(inspect.signature(orig).parameters)
        except (TypeError, ValueError):
            params = []
        # the request a site is about: ``rid`` / ``head_rid`` as a string, ``p`` / ``pending`` as a Pending (its .rid)
        rid_pos = next(((params.index(n), n) for n in ("rid", "head_rid", "p", "pending") if n in params), None)

        def _rid_of(args: tuple, kwargs: dict, _p=rid_pos) -> Optional[str]:
            if _p is None:
                return None
            pos, name = _p
            v = kwargs.get(name, args[pos - 1] if 0 < pos <= len(args) else None)
            if not isinstance(v, str):
                v = getattr(v, "rid", None)
            return v if isinstance(v, str) else None

        if inspect.iscoroutinefunction(orig):
            @functools.wraps(orig)
            async def _aw(self, *a, __o=orig, __m=marker, __k=kind, __r=_rid_of, **kw):
                res = await __o(self, *a, **kw)
                _shadow(self, __m, __k, res, __r(a, kw), recorder_of)
                return res
            _aw._ez_alt_marker = marker  # type: ignore[attr-defined]
            setattr(cls, name, _aw)
        else:
            @functools.wraps(orig)
            def _sw(self, *a, __o=orig, __m=marker, __k=kind, __r=_rid_of, **kw):
                res = __o(self, *a, **kw)
                _shadow(self, __m, __k, res, __r(a, kw), recorder_of)
                return res
            _sw._ez_alt_marker = marker  # type: ignore[attr-defined]
            setattr(cls, name, _sw)
    return missing


def _shadow(obj: Any, marker: str, kind: str, result: Any, rid: Optional[str],
            recorder_of: Callable[[Any], Optional[Recorder]]) -> None:
    try:
        rec = recorder_of(obj)
        if rec is not None:
            rec.alt(marker, kind=kind, action=summarize(result), rid=rid)
    except Exception:  # noqa: BLE001 -- the shadow never breaks the old decision
        pass
