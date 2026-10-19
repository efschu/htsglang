"""S64 S-1400 (b): PP0 RAISES the told to the depth every follower proved.

METAL (NF rc15 ``boot_weg2_...dauer10190714_489909d138_1019_071429``, P log,
pdflip-72-525, 08:31:42-08:31:51):

* PP1/PP2 early reads ``matched=98368 loaded=192`` -> ``completed=98560``;
* PP0 ``#915 PREFETCH REFUSED reason=too_short need=192 threshold=256`` (intake
  and ``P-HELD-REREAD ... why=wake_probe hint=98560`` after ``P-WAKE-PROBE
  store_now=98560``) -> PP0 told 98368, its match;
* both followers ``#1400 OVER-UNRESUMABLE told=98368 own=98560
  resumable_at_told=0`` (KV past told, the recurrent state only at 98560);
* ``PF TOLD-FALLBACK told=98368 -> 0 reason=mismatch acks={1: 98560, 2:
  98560}`` -> ``#1036 ... lost=98368``, P recomputed 98592 tokens.

S-1400 (a) (``pdflip_store_told.issue_completion_read``) makes PP0's own
completion re-read reach 98560 before its told, so this exact sequence no
longer reaches the PF verdict. (b) is the verdict-side rule for every case
where PP0's told still lands below what the followers hold: every follower
acked the SAME depth ``m`` above told (each ack is that stage's own read --
its ``own_prefix``), PP0's store holds ``m`` with its anchor (a fresh probe,
the #1416c/d full-span question with the fetch's key form), PP0 re-reads to
``m`` as a completion read (minimum 1), and once its own tree resumes at ``m``
(``pdflip_told_fidelity.pp0_admissible``, no side effect) PP0 admits ``m``
for EVERY rank. Differing acks, a store that does not hold ``m``, a re-read
that does not land, PP0's tree not resuming at ``m``, or the bound: the
verdict falls through to the unchanged PF rule (told=0 / floor) -- never a
split.

RANG-EINIGKEIT (SOLL R11 / #1004): PP0 decides alone; the followers move
nothing and follow the wire. A plain ``Admit(m)`` is the whole message: a
follower's admission compares its own prefix with the admitted told, and its
ack was that very prefix (``own_prefix`` is computed exactly as the admission
compares it). The wire marker ``raised`` only names the case in the
follower's log; nothing branches on it except the "differs from the
read-ahead" warning.

BOUND: one raise attempt per rid; while PP0's raise read is in flight the
verdict waits, at most ``ENV_WAIT_S`` seconds from the raise (then the PF rule
decides as before). No arena room is reserved, no other request waits.

Switch ``FLLIPER_PDFLIP_S64_TOLD_RAISE`` (default ON; 0 = the PF rule as before).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_RAISE = "FLLIPER_PDFLIP_S64_TOLD_RAISE"
ENV_WAIT_S = "FLLIPER_PDFLIP_S64_TOLD_RAISE_WAIT_S"
WAIT_S_DEFAULT = 10.0
REASON_RAISE = "raise"
WIRE_RAISE = "raised"
WAIT = "wait"
ADMIT = "admit"
_ATTR = "_pdflip_s64_raise"


def raise_on(env=None) -> bool:
    v = (env if env is not None else os.environ).get(ENV_RAISE, "1")
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def wait_s() -> float:
    try:
        return max(0.0, float(os.environ.get(ENV_WAIT_S, "") or WAIT_S_DEFAULT))
    except ValueError:
        return WAIT_S_DEFAULT


@dataclass
class _Raise:
    target: int
    since: float
    state: str  # "issued" | "done" | "failed"
    why: str = ""


def _raises(scheduler) -> Dict[str, _Raise]:
    d = getattr(scheduler, _ATTR, None)
    if d is None:
        d = {}
        setattr(scheduler, _ATTR, d)
    return d


def forget(scheduler, rid: str) -> None:
    d = getattr(scheduler, _ATTR, None)
    if d:
        d.pop(str(rid), None)


def raise_candidate(acks: Dict[int, int], told: int, pp_size: int) -> Optional[int]:
    """The depth every follower acked, when ALL followers acked, all acks are
    EQUAL and lie ABOVE told; else None. Pure."""
    followers = range(1, int(pp_size))
    if not followers or any(r not in acks for r in followers):
        return None
    vals = {int(acks[r]) for r in followers}
    if len(vals) != 1:
        return None
    m = vals.pop()
    return m if m > max(0, int(told)) else None


def store_anchored_depth(scheduler, req) -> Optional[int]:
    """PP0's store: the anchored depth of ``req``'s prompt NOW, in tokens from
    0 -- the same question as ``P-WAKE-PROBE`` (#1416c/d full span, the
    fetch's key form, P's hand-off chain), asked fresh. None = not askable."""
    from flliper.srt.managers import pdflip_store_told as _st

    try:
        cc = getattr(scheduler, "cache_controller", None) or getattr(
            getattr(scheduler, "tree_cache", None), "cache_controller", None)
        ids = getattr(req, "origin_input_ids", None)
        if cc is None or not callable(getattr(cc, "get_hash_str", None)) or not ids:
            return None
        page_size = int(getattr(cc, "page_size", 1) or 1)
        if _st._tree_key_probe_armed():
            key, _bigram = _st._probe_key(scheduler, req, ids, len(ids))
            pages = _st._anchored_pages_full_span(cc, key, page_size, handoff_keys=_st._handoff_chain(req))
        else:
            pages = _st._anchored_pages_full_span(cc, list(ids), page_size,
                                                  extra_key=getattr(req, "extra_key", None))
        return None if pages is None else int(pages) * page_size
    except Exception as exc:  # noqa: BLE001 - a probe never stops the verdict
        logger.warning("S64 TOLD-RAISE store probe rid=%s unavailable: %r", getattr(req, "rid", "?"), exc)
        return None


def _pp0_resumes_at(scheduler, req, m: int, absolute: bool) -> bool:
    """PP0's own tree resumes at ``m`` (TF probe, no side effect). No probe =
    no proof = no raise."""
    from flliper.srt.managers import pdflip_told_fidelity as _tf

    try:
        if not _tf.enabled():
            return False
        depth = _tf.absolute_depth(req, int(m), bool(absolute))
        own = _tf.pp0_admissible(scheduler, req, depth)
        return own is not None and int(own) >= int(depth)
    except Exception:  # noqa: BLE001
        return False


def _fail(scheduler, rid: str, st: _Raise, why: str) -> None:
    st.state, st.why = "failed", why
    n = getattr(scheduler, "_s64_raise_fail_n", 0) + 1
    scheduler._s64_raise_fail_n = n
    if n <= 32 or n % 128 == 0:
        logger.warning("S64 TOLD-RAISE FAILED rid=%s target=%d why=%s (n=%d): the PF rule decides as before",
                       rid, st.target, why, n)


def _partial_candidate(acks: Dict[int, int], told: int, pp_size: int) -> Optional[int]:
    """Some follower acks are still missing and every ack so far names the
    same depth above told: a raise is still possible. Pure."""
    followers = list(range(1, int(pp_size)))
    present = [int(acks[r]) for r in followers if r in acks]
    if not present or len(present) == len(followers) or len(set(present)) != 1:
        return None
    m = present[0]
    return m if m > max(0, int(told)) else None


def pp0_raise_step(scheduler, rid: str, req, told: int, acks: Dict[int, int], absolute: bool,
                   now: Optional[float] = None, pass_n: Optional[int] = None) -> Optional[Tuple[str, int]]:
    """PP0, before the PF verdict of a paced rid. ``None`` = no raise (the PF
    rule decides); ``(WAIT, m)`` = a raise is pending (the missing acks within
    the PF Frist, or PP0's raise read in flight): keep the rid open this pass;
    ``(ADMIT, m)`` = admit ``m`` for every rank."""
    rid = str(rid)
    if not raise_on() or not absolute:
        return None
    now = time.monotonic() if now is None else float(now)
    raises = _raises(scheduler)
    st = raises.get(rid)
    if st is None:
        m = raise_candidate(acks, told, int(scheduler.ps.pp_size))
        if m is None:
            # H64's rule for the floor, mirrored: while the acks so far agree on
            # a depth above told, wait for the missing ones -- bounded by the
            # unchanged PF Frist (T2: by progress); past it the PF rule decides.
            mp = _partial_candidate(acks, told, int(scheduler.ps.pp_size))
            if mp is None:
                return None
            from flliper.srt.managers import pdflip_told_fallback as _fb

            o = _fb._pp0_open_map(scheduler).get(rid)
            if o is not None and _fb._frist_verdict(scheduler, rid, o, now, pass_n) is None:
                return WAIT, int(mp)
            return None
        st = raises[rid] = _Raise(target=int(m), since=now, state="issued")
        store = store_anchored_depth(scheduler, req)
        if store is None or int(store) < int(m):
            _fail(scheduler, rid, st, f"store={store}<{m}")
            return None
        fn = getattr(scheduler, "_prefetch_kvcache", None)
        if not callable(fn):
            _fail(scheduler, rid, st, "no_reader")
            return None
        from flliper.srt.managers import pdflip_store_told as _st

        try:
            verdict = _st.issue_completion_read(fn, req)
        except Exception as exc:  # noqa: BLE001 - a re-read never takes the pass down
            _fail(scheduler, rid, st, f"raised:{type(exc).__name__}")
            return None
        if not verdict.startswith("issued"):
            _fail(scheduler, rid, st, f"read:{verdict}")
            return None
        n = getattr(scheduler, "_s64_raise_issued_n", 0) + 1
        scheduler._s64_raise_issued_n = n
        if n <= 32 or n % 128 == 0:
            logger.warning(
                "S64 TOLD-RAISE ISSUED rid=%s told=%d target=%d store=%d acks=%s (n=%d): every follower holds the "
                "same depth above told and the store holds it with its anchor -- PP0 re-reads to it (completion "
                "read, minimum 1) instead of told=0", rid, int(told), int(m), int(store), dict(acks), n)
        return WAIT, int(m)
    if st.state != "issued":
        return None
    tree = getattr(scheduler, "tree_cache", None)
    if tree is None or not tree.check_prefetch_progress(rid):
        if now - st.since > wait_s():
            _fail(scheduler, rid, st, "bound")
            return None
        return WAIT, st.target
    if not _pp0_resumes_at(scheduler, req, st.target, absolute):
        _fail(scheduler, rid, st, "pp0_not_resumable")
        return None
    st.state = "done"
    n = getattr(scheduler, "_s64_raise_admit_n", 0) + 1
    scheduler._s64_raise_admit_n = n
    logger.warning(
        "S64 TOLD-RAISE ADMIT rid=%s told=%d -> %d after=%.2fs (n=%d): PP0's re-read landed and its tree resumes "
        "there -- PP0 admits that depth for EVERY rank (each follower acked it)",
        rid, int(told), st.target, now - st.since, n)
    return ADMIT, st.target
