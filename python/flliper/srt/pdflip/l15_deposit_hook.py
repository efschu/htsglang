"""L15-14d: the P-stage side of the phase-2 deposit, wired.

Lifecycle per (P stage, rid), opt-in FLLIPER_PDFLIP_L15_DEPOSIT=1 (plus the hold
share: FLLIPER_PDFLIP_L15_HOT_SHARE=1, FLLIPER_PDFLIP_VMM_EXPORTABLE=1 at boot):

1. admission (:func:`open_for_sched`, scheduler ``_add_request_to_queue``):
   the front's ``dep.<rid>.json`` (l15_deposit.FrontDeposits) names
   ``e_start, n, anchor_row`` for D's published epoch; the stage checks the
   epoch against D rank 0's descriptor and the prompt length against ``n``,
   fetches every D rank's share through ONE HoldMapper and keeps a session;
2. every finished chunk (:func:`on_chunk`, tree ``cache_unfinished_req``,
   after the L2 publish): the tokens ``[upto, b)`` computed so far go from the
   request's P rows into D's owner rows (l15_deposit_put.put_kv; cap-0 ranks
   skipped by name -- they refill from L2 at D's wake);
3. the end (:func:`on_chunk` with ``final``, tree ``cache_finished_req``
   before the rows can be freed): the rest of the tokens, then the END anchor
   from the request's mamba slot (put_anchor), then the stage's record
   ``depdone.<rid>.<stage>.json`` and the session is closed (stream sync,
   every mapping and fd released).

Any refusal or error closes the session and writes the record with
``failed`` = the named reason: D adopts only complete records (every D
attention layer and every linear layer covered, ``upto == n``, anchor done,
same epoch) -- everything else goes today's way (L2 load). Nothing here ever
raises into the scheduler.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DEPOSIT_ENV = "FLLIPER_PDFLIP_L15_DEPOSIT"
#: sessions beyond this are closed oldest-first (an aborted request that
#: never reaches cache_finished_req must not hold D's mappings forever)
MAX_SESSIONS = 64


def deposit_on(env) -> bool:
    return str(env.get(DEPOSIT_ENV, "0")).strip() == "1"


@dataclass
class DepositSession:
    rid: str
    epoch: int
    e_start: int
    n: int
    anchor_row: int
    skip_ranks: Tuple[int, ...]
    geom: object
    shares: Dict[int, tuple]
    mapper: object
    directory: str
    stage_key: str
    upto: int = 0
    cells: int = 0
    log: object = None
    chunks: int = 0
    # L15-DEPOSIT-SALT (300): the P request's own extra_key (cache_salt / mm /
    # lora namespace) -- the record carries it to D's adopt; None = unsalted
    extra_key: Optional[str] = None


_SESSIONS: Dict[str, DepositSession] = {}


def active() -> bool:
    """Cheap guard for the tree hooks."""
    return bool(_SESSIONS)


def record_path(directory: str, rid: str, stage_key: str) -> str:
    return os.path.join(directory, "depdone.%s.%s.json" % (rid, stage_key))


def write_record(sess: DepositSession, *, anchor_bytes: int = 0,
                 failed: Optional[str] = None) -> None:
    g = sess.geom
    rec = {
        "epoch": int(sess.epoch), "rid": sess.rid, "e_start": int(sess.e_start),
        "n": int(sess.n), "anchor_row": int(sess.anchor_row),
        "upto": int(sess.upto), "cells": int(sess.cells),
        "anchor_bytes": int(anchor_bytes),
        "att_layers": sorted(int(x) for x in getattr(g, "p_buffers", {}) or {}),
        "linear": list(getattr(g, "stage_linear", (0, 0))),
        "skip_ranks": list(sess.skip_ranks),
        "failed": failed,
    }
    if sess.extra_key is not None:
        # absent = unsalted ONLY (an old record), never "any"
        rec["extra_key"] = str(sess.extra_key)
    path = record_path(sess.directory, sess.rid, sess.stage_key)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def _close(sess: DepositSession, *, anchor_bytes: int = 0,
           failed: Optional[str] = None) -> None:
    _SESSIONS.pop(sess.rid, None)
    try:
        sess.mapper.close()
    finally:
        write_record(sess, anchor_bytes=anchor_bytes, failed=failed)
        if sess.log is not None:
            if failed:
                sess.log("L15-DEPOSIT-OFF rid=%s stage=%s upto=%d/%d reason=%s"
                         % (sess.rid, sess.stage_key, sess.upto, sess.n, failed))
            else:
                sess.log("L15-DEPOSIT-DONE rid=%s stage=%s n=%d cells=%d "
                         "anchor_bytes=%d chunks=%d"
                         % (sess.rid, sess.stage_key, sess.n, sess.cells,
                            anchor_bytes, sess.chunks))


def open_session(*, rid: str, prompt_len: int, hint: dict, d0: dict, geom,
                 fetch, n_d_ranks: int, mapper, directory: str,
                 log, extra_key: Optional[str] = None) -> Optional[str]:
    """Register the session; None on success, else the named reason (the
    mapper is closed again on a refusal)."""
    from flliper.srt.pdflip.l15_hold_share import L15ShareError

    dep = d0.get("deposit")
    if not dep:
        mapper.close()
        return "D published no deposit region"
    if int(d0.get("epoch", -1)) != int(hint.get("epoch", -2)):
        mapper.close()
        return "hint epoch %s != D epoch %s" % (hint.get("epoch"), d0.get("epoch"))
    n = int(hint["n"])
    if int(prompt_len) != n:
        mapper.close()
        return "prompt %d tokens != deposit n %d" % (prompt_len, n)
    e_start = int(hint["e_start"])
    if e_start < int(dep["e0"]) or e_start + n > int(dep["e1"]):
        mapper.close()
        return "slots [%d,%d) outside the region [%s,%s)" % (
            e_start, e_start + n, dep["e0"], dep["e1"])
    a = int(hint["anchor_row"])
    if not int(dep["a0"]) <= a < int(dep["a1"]):
        mapper.close()
        return "anchor row %d outside [%s,%s)" % (a, dep["a0"], dep["a1"])
    skip = sorted({int(x) for x in dep.get("skip_ranks", ())}
                  | {int(x) for x in d0.get("cap0", ())})
    try:
        # the cap-0 ranks publish no share: nothing is deposited there
        shares = {r: fetch(r) for r in range(int(n_d_ranks)) if r not in skip}
    except L15ShareError as exc:
        mapper.close()
        return "share: %s" % exc
    lo, hi = geom.stage_linear
    stage_key = "L%d-%d" % (lo, hi)
    while len(_SESSIONS) >= MAX_SESSIONS:
        _close(next(iter(_SESSIONS.values())), failed="session cap")
    old = _SESSIONS.get(rid)
    if old is not None:
        _close(old, failed="re-admitted")
    _SESSIONS[rid] = DepositSession(
        rid=rid, epoch=int(d0["epoch"]), e_start=e_start, n=n, anchor_row=a,
        skip_ranks=tuple(skip),
        geom=geom, shares=shares, mapper=mapper, directory=directory,
        stage_key=stage_key, log=log, extra_key=extra_key)
    log("L15-DEPOSIT-OPEN rid=%s stage=%s e_start=%d n=%d anchor_row=%d skip=%s"
        % (rid, stage_key, e_start, n, a, list(dep.get("skip_ranks", ()))))
    return None


def open_for_sched(sched, req, env, log) -> Optional[str]:
    """Scheduler entry at P admission: None when nothing was to do or the
    session is open, else the named reason."""
    from flliper.srt.pdflip import l15_deposit, l15_share_publish
    from flliper.srt.pdflip.l15_hold_share import HoldMapper, L15ShareError
    from flliper.srt.pdflip.l15_share_admit import stage_geometry

    directory = l15_share_publish.share_dir(env)
    rid = str(getattr(req, "rid", ""))
    hint = l15_deposit.read_deposit_hint(directory, rid)
    if hint is None:
        return None
    from flliper.srt.pdflip.l15_share_admit import _first_share

    try:
        d0 = _first_share(directory, l15_share_publish.fetch_share)
    except L15ShareError as exc:
        return "share: %s" % exc
    geom = stage_geometry(sched, d0)
    if isinstance(geom, str):
        return geom
    mapper = HoldMapper(geom.dev)
    return open_session(
        rid=rid, prompt_len=len(getattr(req, "origin_input_ids", ()) or ()),
        hint=hint, d0=d0, geom=geom,
        fetch=lambda r: mapper.fetch(
            lambda q: l15_share_publish.fetch_share(directory, q), r),
        n_d_ranks=geom.n_d, mapper=mapper, directory=directory, log=log,
        # L15-DEPOSIT-SALT (300): the request's namespace, as its own tree uses it
        extra_key=getattr(req, "extra_key", None))


def _p_rows(sess: DepositSession, req, a: int, b: int) -> List[int]:
    rtp = sess.geom.req_to_token_pool
    idx = getattr(req, "req_pool_idx", None)
    if idx is None:
        raise ValueError("request has no req_pool_idx")
    return [int(x) for x in rtp.req_to_token[int(idx), a:b].tolist()]


def on_chunk(req, *, final: bool, filled: Optional[int] = None) -> None:
    """Tree hook. ``filled`` = tokens whose KV is computed (len(fill_ids) at
    cache_unfinished_req); ``final`` deposits up to ``n`` plus the anchor and
    closes. Never raises."""
    sess = _SESSIONS.get(str(getattr(req, "rid", "")))
    if sess is None:
        return
    try:
        from flliper.srt.pdflip.l15_deposit_put import put_anchor, put_kv

        b = sess.n if final else min(sess.n, int(filled or 0))
        g = sess.geom
        if b > sess.upto:
            sess.cells += put_kv(
                sess.shares, e_start=sess.e_start, a=sess.upto, b=b,
                stage_layers=sorted(g.p_buffers), p_buffers=g.p_buffers,
                p_rows=_p_rows(sess, req, sess.upto, b),
                skip_ranks=sess.skip_ranks, map_extent=sess.mapper)
            sess.upto = b
            sess.chunks += 1
        if not final:
            return
        slot = getattr(req, "mamba_pool_idx", None)
        if slot is None:
            _close(sess, failed="request has no mamba slot")
            return
        if int(g.stage_linear[0]) == 0:
            # L15-14e: the stage owning linear layer 0 leaves the prompt's
            # token ids for D's adopt (BEFORE its record: D trusts records)
            from flliper.srt.pdflip.l15_deposit_adopt import write_tokens

            ids = list(getattr(req, "origin_input_ids", ()) or ())[: sess.n]
            if len(ids) != sess.n:
                _close(sess, failed="prompt has %d ids < n %d" % (len(ids), sess.n))
                return
            write_tokens(sess.directory, sess.rid, ids)
        ratios = _ratios(sess.shares)
        nbytes = put_anchor(
            sess.shares, spec=g.spec, ratios=ratios, stage=g.stage_linear,
            p_temporal=g.p_temporal, p_conv=g.p_conv, p_slot=int(slot),
            anchor_row=sess.anchor_row, skip_ranks=sess.skip_ranks,
            map_extent=sess.mapper)
        _close(sess, anchor_bytes=nbytes)
    except Exception as exc:  # noqa: BLE001 -- D adopts only complete records
        _close(sess, failed="%s: %s" % (type(exc).__name__, exc))


def abort(rid: str, reason: str) -> None:
    sess = _SESSIONS.get(str(rid))
    if sess is not None:
        _close(sess, failed=reason)


def close_all(reason: str) -> int:
    """P sleep / shutdown: release every open session (named)."""
    n = 0
    for sess in list(_SESSIONS.values()):
        _close(sess, failed=reason)
        n += 1
    return n


def _ratios(shares) -> list:
    d = next(iter(shares.values()))[0]
    prefix = [int(x) for x in d["prefix"]]
    return [prefix[i + 1] - prefix[i] for i in range(len(prefix) - 1)]
