"""L15-14e: D adopts the complete phase-2 deposits at its wake.

The P stages wrote a request's prompt KV into D's held deposit rows (slots
``[e_start, e_start + n)``) and its END anchor into mamba row ``anchor_row``
(l15_deposit_hook); every stage left ``depdone.<rid>.L<lo>-<hi>.json`` and
the stage owning linear layer 0 the prompt's token ids
(``deptok.<rid>.bin``). At D's wake, after the group verdict "hold" and its
act (the deposit region lives inside the kept hold extents of that epoch):

1. every rank reads the same records (shared directory) and keeps the rids
   whose deposit is COMPLETE: no ``failed``, one epoch == the hold's, one
   (e_start, n, anchor_row), every stage ``upto == n``, the stages' attention
   layers covering D's and their linear ranges tiling ``[0, n_linear)``;
2. the token ids give the request's chain in D's tree (the dormant read put
   P's published prompt there as host nodes) -> per token the L2 page slot,
   generation and lane, plus the end node's anchor L2 slot: a HoldSpan;
3. a rank the deposit skipped (cap 0: P wrote nothing there) refills its
   owned rows and its anchor share from L2 -- the same rid-tagged plan,
   generation check and bulk loads as the held spans' refill; every rank
   checks the slots and the anchor row are still free;
4. ONE all_gather_object of each rank's ok-rid list; the intersection is
   adopted on every rank (l15_p_adopt.adopt: reserve, insert as device nodes
   with the anchor) -- ranks never disagree about the tree;
5. the files of this epoch are removed (the next epoch starts clean).

A deposit that is not adopted costs nothing: its rows stay free (garbage
until reused), the request loads from L2 as today. Opt-in
SGLANG_WEG2_L15_DEPOSIT=1; the collective runs on every rank whenever the
gate is open (verdict "hold", group ok, switch on), even with no deposits.
"""

from __future__ import annotations

from sglang.srt.weg2.l15_shadow import kv_pool_of as _kvp  # L15-FIX-REFILL-POOL

import json
import os
from array import array
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Deposit:
    rid: str
    epoch: int
    e_start: int
    n: int
    anchor_row: int
    skip_ranks: Tuple[int, ...]


# -- token ids (written by the P stage owning linear layer 0) --------------

def tokens_path(directory: str, rid: str) -> str:
    return os.path.join(directory, "deptok.%s.bin" % rid)


def write_tokens(directory: str, rid: str, token_ids: Sequence[int]) -> None:
    path = tokens_path(directory, rid)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(array("q", [int(t) for t in token_ids]).tobytes())
    os.replace(tmp, path)


def read_tokens(directory: str, rid: str) -> Optional[List[int]]:
    try:
        with open(tokens_path(directory, rid), "rb") as fh:
            a = array("q")
            a.frombytes(fh.read())
    except (FileNotFoundError, ValueError):
        return None
    return list(a)


# -- completeness ------------------------------------------------------------

def load_deposits(directory: str, epoch: int, att_layers: Sequence[int],
                  n_linear: int, listdir: Callable = os.listdir
                  ) -> Tuple[List[Deposit], Dict[str, str]]:
    """(complete deposits sorted by rid, {rid: reason} for the rest)."""
    by_rid: Dict[str, List[dict]] = {}
    try:
        names = listdir(directory)
    except FileNotFoundError:
        return [], {}
    for name in names:
        if not (name.startswith("depdone.") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(directory, name)) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        by_rid.setdefault(str(rec.get("rid", "")), []).append(rec)
    want_att = sorted(int(x) for x in att_layers)
    ok: List[Deposit] = []
    bad: Dict[str, str] = {}
    for rid in sorted(by_rid):
        why = _incomplete(by_rid[rid], int(epoch), want_att, int(n_linear))
        if why is not None:
            bad[rid] = why
            continue
        r0 = by_rid[rid][0]
        ok.append(Deposit(rid, int(r0["epoch"]), int(r0["e_start"]), int(r0["n"]),
                          int(r0["anchor_row"]),
                          tuple(int(x) for x in r0.get("skip_ranks", ()))))
    return ok, bad


def _incomplete(recs: List[dict], epoch: int, want_att: List[int],
                n_linear: int) -> Optional[str]:
    for r in recs:
        if r.get("failed"):
            return "stage %s failed: %s" % (r.get("linear"), r["failed"])
        if int(r.get("epoch", -1)) != epoch:
            return "epoch %s != hold epoch %d" % (r.get("epoch"), epoch)
    keys = {(int(r["e_start"]), int(r["n"]), int(r["anchor_row"]),
             tuple(r.get("skip_ranks", ()))) for r in recs}
    if len(keys) != 1:
        return "stages disagree on the slot range/anchor row"
    n = int(recs[0]["n"])
    short = [r.get("linear") for r in recs if int(r.get("upto", -1)) != n]
    if short:
        return "stage(s) %s deposited fewer than %d tokens" % (short, n)
    att = sorted(int(x) for r in recs for x in r.get("att_layers", ()))
    if att != want_att:
        return "attention layers %s != D's %s" % (att[:6], want_att[:6])
    lin = sorted(tuple(int(x) for x in r.get("linear", (0, 0))) for r in recs)
    cur = 0
    for lo, hi in lin:
        if lo != cur or hi <= lo:
            return "linear ranges %s do not tile [0,%d)" % (lin, n_linear)
        cur = hi
    if cur != n_linear:
        return "linear ranges %s do not tile [0,%d)" % (lin, n_linear)
    no_anchor = [r.get("linear") for r in recs if int(r.get("anchor_bytes", 0)) <= 0]
    if no_anchor:
        return "stage(s) %s wrote no END anchor" % (no_anchor,)
    return None


# -- the span from D's tree --------------------------------------------------

def host_rows_to_l2(pool, rows: Sequence[int], log) -> Tuple[tuple, tuple, tuple]:
    """Host row -> (arena page slot, generation, lane), the l15_bind rule:
    rows below ``staging_rows`` have no L2 copy (-1); slot = (row - S) // P,
    lane = (row - S) % P; a ``row_slot`` map (draft role) is not
    page-addressed. Generations from ONE census."""
    from sglang.srt.weg2.l15_bind import _slot_gens_or_minus_one

    s0 = int(getattr(pool, "staging_rows", 0))
    p = max(1, int(getattr(pool, "_arena_page_tokens", 1)))
    row_slot = getattr(pool, "row_slot", None)
    slots, lanes = [], []
    for r in rows:
        r = int(r)
        if r < s0:
            slots.append(-1)
            lanes.append(-1)
        elif row_slot is not None:
            slots.append(int(row_slot.get(r, -1)))
            lanes.append(0 if p == 1 else -1)
        else:
            slots.append((r - s0) // p)
            lanes.append((r - s0) % p)
    uniq = sorted({s for s in slots if s >= 0})
    gen = {int(s): int(g) for s, g in
           zip(uniq, _slot_gens_or_minus_one(pool, uniq, log, "kv"))} if uniq else {}
    return tuple(slots), tuple(gen.get(s, -1) for s in slots), tuple(lanes)


def end_anchor_l2(node, mamba_pool, log) -> Tuple[int, int]:
    """The END node's mamba host row -> (arena slot, generation); (-1, -1)
    when the node carries no L2 state."""
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        ComponentType,
    )
    from sglang.srt.weg2.l15_bind import _slot_gens_or_minus_one

    try:
        cd = node.component_data[ComponentType.MAMBA]
    except (AttributeError, KeyError, IndexError, TypeError):
        return (-1, -1)
    hv = getattr(cd, "host_value", None)
    if hv is None or not len(hv) or mamba_pool is None:
        return (-1, -1)
    row = int(hv.tolist()[0] if hasattr(hv, "tolist") else hv[0])
    s = row - int(getattr(mamba_pool, "staging_rows", 0))
    if s < 0:
        return (-1, -1)
    g = _slot_gens_or_minus_one(mamba_pool, [s], log, "mamba")[0]
    return (s, int(g))


def span_for(dep: Deposit, token_ids: Sequence[int], match, kv_pool,
             mamba_pool, log):
    """HoldSpan of ``dep`` from D's tree, or the named reason (str).
    ``match(token_ids) -> (node, matched_len)``: the deepest node (device or
    host) of the chain."""
    from sglang.srt.weg2.l15_bind import chain_host_rows
    from sglang.srt.weg2.l15_manifest import HoldSpan

    if len(token_ids) != dep.n:
        return "token ids %d != n %d" % (len(token_ids), dep.n)
    node, got = match(list(token_ids))
    if node is None or int(got) != dep.n:
        return "D's tree covers %d of %d tokens" % (int(got or 0), dep.n)
    rows = chain_host_rows(node)
    if len(rows) != dep.n:
        return "chain carries %d host rows for %d tokens" % (len(rows), dep.n)
    if kv_pool is None:
        return "no L2 host pool"
    slots, gens, lanes = host_rows_to_l2(kv_pool, rows, log)
    a_slot, a_gen = end_anchor_l2(node, mamba_pool, log)
    return HoldSpan(rid=dep.rid, depth=dep.n,
                    slots=tuple(range(dep.e_start, dep.e_start + dep.n)),
                    anchor_slot=dep.anchor_row, l2_slots=slots, l2_gens=gens,
                    anchor_l2_slot=a_slot, anchor_l2_gen=a_gen, l2_lanes=lanes)


# -- the wake --------------------------------------------------------------

def refill_skipped(spans: Sequence, rank: int, prefix: Sequence[int], *,
                   host_pool, device_pool, host_mamba, dev_mamba, log
                   ) -> Dict[str, str]:
    """This rank's L2 refill for the deposits that skipped it: KV rows (one
    generation-checked bulk load) + anchor shares. {rid: reason} of the
    spans that could not be refilled (the rest landed)."""
    import torch

    from sglang.srt.weg2 import l15_refill, l15_restore
    from sglang.srt.weg2.l15_manifest import Manifest

    failed: Dict[str, str] = {}
    live = []
    for sp in spans:
        if any(int(s) < 0 for s in sp.l2_slots):
            failed[sp.rid] = "token(s) without an L2 page"
        elif int(sp.anchor_l2_slot) < 0:
            failed[sp.rid] = "END anchor without an L2 state"
        else:
            live.append(sp)
    if live and (host_mamba is None or dev_mamba is None
                 or getattr(host_mamba, "slot_gens", None) is None):
        for sp in live:
            failed[sp.rid] = "no mamba host/device pool for the anchor"
        live = []
    if live:
        gens = [int(g) for g in host_mamba.slot_gens([int(sp.anchor_l2_slot)
                                                       for sp in live])]
        keep = []
        for sp, g in zip(live, gens):
            if g != int(sp.anchor_l2_gen):
                failed[sp.rid] = "anchor generation %d != %d" % (g, sp.anchor_l2_gen)
            else:
                keep.append(sp)
        live = keep
    if not live:
        return failed
    m = Manifest(epoch=0, pid=0, spans=tuple(live), rows_by_rank=(),
                 anchor_slots=0)
    plan = l15_restore.rid_tagged_plan(m, rank, prefix)
    ok, bad = l15_refill.gen_check(plan, host_pool)
    bad_rids = {str(x) for b in bad for x in (b if isinstance(b, tuple) else (b,))}
    for sp in live:
        if sp.rid in bad_rids:
            failed[sp.rid] = "KV generation mismatch"
    live = [sp for sp in live if sp.rid not in failed]
    ok = [e for e in ok if not (set(map(str, e[0])) & set(failed))]
    if not live:
        return failed
    page_tokens = max(1, int(getattr(host_pool, "_arena_page_tokens", 1)))
    try:
        if ok:
            l15_refill.refill(ok, host_pool, device_pool, page_tokens)
        host_mamba._load_states_all_layers(
            dev_mamba,
            torch.tensor([int(sp.anchor_l2_slot) for sp in live], dtype=torch.int64),
            torch.tensor([int(sp.anchor_slot) for sp in live], dtype=torch.int64))
    except Exception as exc:  # noqa: BLE001 -- none of this batch is adopted
        for sp in live:
            failed[sp.rid] = "L2 load failed: %s" % (exc,)
    return failed


def rows_free(dep: Deposit, kv_alloc, mamba_alloc) -> Optional[str]:
    free_kv = set(int(x) for x in kv_alloc.free_pages.tolist())
    miss = [s for s in range(dep.e_start, dep.e_start + dep.n) if s not in free_kv]
    if miss:
        return "slot(s) %s not free" % (miss[:4],)
    if int(dep.anchor_row) not in set(int(x) for x in mamba_alloc.free_slots.tolist()):
        return "anchor row %d not free" % dep.anchor_row
    return None


def agree(my_ok: Sequence[str], gather: Callable[[list], list]) -> List[str]:
    """The rids every rank can adopt, in one order."""
    votes = gather(sorted(set(my_ok)))
    common = set(votes[0]) if votes else set()
    for v in votes[1:]:
        common &= set(v or ())
    return sorted(common)


def adopt_deposits(*, directory: str, epoch: int, rank: int,
                   prefix: Sequence[int], att_layers: Sequence[int],
                   n_linear: int, match, tree_cache, kv_alloc, mamba_alloc,
                   host_pool, device_pool, host_mamba, dev_mamba,
                   gather: Callable[[list], list], log) -> List[str]:
    """Steps 1-4 of the module docstring; returns the adopted rids. Every
    rank must call it (the gather is unconditional)."""
    from sglang.srt.weg2 import l15_p_adopt

    try:
        deps, refused = load_deposits(directory, epoch, att_layers, n_linear)
    except Exception as exc:  # noqa: BLE001 -- vote empty, still gather
        log("L15-DEPOSIT-ADOPT records unreadable (%s: %s)" % (type(exc).__name__, exc))
        deps, refused = [], {}
    for rid, why in refused.items():
        log("L15-DEPOSIT-REFUSED rid=%s reason=%s" % (rid, why))
    spans, toks, my_ok = {}, {}, []
    for dep in deps:
        try:
            t = read_tokens(directory, dep.rid)
            why = "no token ids" if t is None else None
            if why is None:
                sp = span_for(dep, t, match, host_pool, host_mamba, log)
                if isinstance(sp, str):
                    why = sp
                else:
                    spans[dep.rid], toks[dep.rid] = sp, t
            if why is None:
                why = rows_free(dep, kv_alloc, mamba_alloc)
        except Exception as exc:  # noqa: BLE001 -- this rid only; the gather must run
            why = "%s: %s" % (type(exc).__name__, exc)
        if why is not None:
            log("L15-DEPOSIT-REFUSED rid=%s rank=%d reason=%s" % (dep.rid, rank, why))
            spans.pop(dep.rid, None)
    mine = [spans[d.rid] for d in deps if d.rid in spans and rank in d.skip_ranks]
    try:
        failed = refill_skipped(mine, rank, prefix, host_pool=host_pool,
                                device_pool=device_pool, host_mamba=host_mamba,
                                dev_mamba=dev_mamba, log=log) if mine else {}
    except Exception as exc:  # noqa: BLE001 -- nothing of this rank's adopted
        failed = {sp.rid: "%s: %s" % (type(exc).__name__, exc) for sp in mine}
    for rid, why in failed.items():
        log("L15-DEPOSIT-REFUSED rid=%s rank=%d reason=refill: %s" % (rid, rank, why))
    my_ok = [rid for rid in spans if rid not in failed]
    agreed = agree(my_ok, gather)
    by_rid = {d.rid: d for d in deps}
    adopted = []
    for rid in agreed:
        d = by_rid[rid]
        try:
            l15_p_adopt.adopt(tree_cache, kv_alloc, mamba_alloc,
                              token_ids=toks[rid],
                              rows=list(range(d.e_start, d.e_start + d.n)),
                              anchor_row=d.anchor_row)
        except l15_p_adopt.L15AdoptRefused as exc:
            # checked free before the vote on every rank: a refusal here
            # means ranks diverged -- loud, never silent
            raise RuntimeError("L15-DEPOSIT adopt of agreed rid %s refused on "
                               "rank %d: %s" % (rid, rank, exc)) from exc
        adopted.append(rid)
    log("L15-DEPOSIT-ADOPT epoch=%d rank=%d complete=%d mine_ok=%d agreed=%d "
        "adopted=%d tokens=%d" % (epoch, rank, len(deps), len(my_ok), len(agreed),
                                  len(adopted),
                                  sum(by_rid[r].n for r in adopted)))
    return adopted


def clear_epoch_files(directory: str, listdir: Callable = os.listdir) -> int:
    """Remove the deposit records/token files/hints after the adopt."""
    n = 0
    try:
        names = listdir(directory)
    except FileNotFoundError:
        return 0
    for name in names:
        if name.startswith(("depdone.", "deptok.", "dep.")):
            try:
                os.unlink(os.path.join(directory, name))
                n += 1
            except FileNotFoundError:
                pass
    return n


def adopt_for_sched(sched, env, log, *, epoch: int, gather) -> List[str]:
    """D wake entry (every D rank, behind the group verdict "hold"): the live
    geometry, then :func:`adopt_deposits`, then this epoch's files go. Never
    raises before the gather (a failure votes an empty set); a refused adopt
    of an AGREED rid is the one loud error (ranks diverged)."""
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.weg2 import l15_bind, l15_share_publish

    directory = l15_share_publish.share_dir(env)
    try:
        tree = getattr(sched, "tree_cache", None)
        mr = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
        wrapper = getattr(mr, "token_to_kv_pool", None)
        amap = getattr(wrapper, "full_attention_layer_id_mapping", None) or {}
        rtp = getattr(sched, "req_to_token_pool", None)
        n_linear = len(getattr(rtp, "mamba_map", None) or {})
        from sglang.srt.distributed.utils import get_cp_token_ratios

        ratios = get_cp_token_ratios() or [1]
        prefix = [0]
        for x in ratios:
            prefix.append(prefix[-1] + int(x))
        rank = int(getattr(getattr(sched, "ps", None), "tp_rank", 0) or 0)
        host_pool, host_mamba = l15_bind.live_host_pools(tree)

        def match(ids):
            res = tree.match_prefix(MatchPrefixParams(key=RadixKey(
                token_ids=list(ids), extra_key=None,
                is_bigram=getattr(tree, "is_eagle", False))))
            got = len(res.device_indices) + int(res.host_hit_length or 0)
            return res.best_match_node, got

        ready = True
    except Exception as exc:  # noqa: BLE001 -- vote empty, still gather
        log("L15-DEPOSIT-ADOPT geometry unavailable (%s: %s) -- votes none"
            % (type(exc).__name__, exc))
        ready = False
    if not ready:
        agree([], gather)
        return []
    try:
        adopted = adopt_deposits(
            directory=directory, epoch=int(epoch), rank=rank, prefix=prefix,
            att_layers=sorted(int(g) for g in amap), n_linear=n_linear,
            match=match, tree_cache=tree,
            kv_alloc=getattr(sched, "token_to_kv_pool_allocator", None),
            mamba_alloc=getattr(rtp, "mamba_allocator", None),
            host_pool=host_pool, device_pool=_kvp(wrapper), host_mamba=host_mamba,
            dev_mamba=getattr(rtp, "mamba_pool", None), gather=gather, log=log)
    finally:
        clear_epoch_files(directory)
    return adopted
