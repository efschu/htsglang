"""TOLD FIDELITY (27B rc12k27 b1, 27.09. 09:45:58, rid weg2-10-95): PP0's Admit
names exactly the prefix PP0 itself can admit, or it names 0 for every rank.

THE BREAK. The paced #1416e form puts ``Weg2StoreAdmit(told)`` on the wire at the
top of PP0's pass, and PP0 admits in the same pass; the followers adopt the told
one pass later (their radix match capped to it, #1419). An ABSOLUTE told (TK
ABS-TOLD, on for the 27B profile; a fork TWIN always) is head + span: the head
is what PP0's registration matched in ITS OWN TREE (16383 here, span 0), the
span its store read. Nothing holds that head between the registration and the
admission, and PP0's own head was no longer resumable when it admitted:

  PP0 P-CHUNK-BUDGET head=weg2-10-95 pos=16383 pos_src=told local_prefix=16383
  PP0 [#928 anchor] REFUSING resume: node carries no recurrent state on device
      or host ... match_tokens=69 best_value_len=0 kv_host_hit=0
  PP0 P-CHUNK-POLICY plan key=weg2-10-95 start=512 ... pick=mid=1024
  PP1/PP2 #988 LOADBACK rid=weg2-10-95 prefix moved to 16383 ... mamba_restored=33
  PP1 W27: 1024 rows for 512 tokens, sender_geom=(512, 1536)
      receiver_geom=(16895, 17407)

The followers read 16383 tokens WITH the anchor from the store; PP0 looked only
at its own tree, where the head's rows and state were gone: at 09:45:55 PP0
loaded weg2-10-93 back (``WEG2-ARENA-LOAD rows=151514 ... dst=[1,265705] of
265706`` -- the whole device KV pool), which evicts every unlocked device node,
the registered-but-not-admitted head included (the #1469 trail was exhausted at
09:40, so the evicting call itself is not on this boot's record). PP0 -- the authority
-- admitted a different prefix than the one it had told: a start split, and the
width split followed from the different start.

THE RULE (switch ``SGLANG_WEG2_TOLD_FIDELITY``, default on; ``0`` = the old Admit
byte for byte): right before PP0 puts an Admit with ``told > 0`` on the wire it
asks its own tree what its admission will resume from (:func:`pp0_admissible`,
the #928 rule read-only: the matched depth, 0 when the deepest node carries no
recurrent state on device or host). Less than told -> the Admit carries told=0
with the PF fallback marker: PP0 releases its own read, every follower releases
its read (``follower_release``) and every rank admits at 0 -- the same prefill
on every stage, before any follower adopted anything. A probe that cannot be
asked gives no verdict (the Admit stays as it was).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_TOLD_FIDELITY"


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _node_has_state(node) -> bool:
    """#928 (a) read-only: False only when the node carries a MAMBA component
    with neither a device value nor a host value. A tree without a recurrent
    component (dense model) always resumes from its KV."""
    data = getattr(node, "component_data", None)
    if data is None:
        return True
    try:
        from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        comp = data[ComponentType.MAMBA]
    except (IndexError, KeyError, TypeError, ImportError):
        return True
    if comp is None:
        return True
    return getattr(comp, "value", None) is not None or getattr(comp, "host_value", None) is not None


def pp0_admissible(scheduler, req, told: int) -> Optional[int]:
    """The prefix PP0's own tree can resume ``req`` from, capped at ``told``
    keys; None = the probe could not be asked (no verdict)."""
    try:
        tree = getattr(scheduler, "tree_cache", None)
        match = getattr(tree, "match_prefix", None)
        # R5b: no truth test on a sequence that may be a tensor (see below).
        ids = getattr(req, "full_untruncated_fill_ids", None)
        if ids is None or len(ids) == 0:
            ids = getattr(req, "origin_input_ids", None)
        if tree is None or not callable(match) or ids is None or len(ids) == 0:
            return None  # decided before any import: a tree without a match costs nothing
        from sglang.srt.managers.weg2_store_told import _probe_key
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

        key, _bigram = _probe_key(scheduler, req, ids, int(told))
        mr = match(MatchPrefixParams(key=key))
        # R5b (rc12z3 7055ec73f3, PP0 04:32-04:34, 11 of 11 probes): the
        # tree's device_indices is a torch tensor, so ``x or ()`` asked its
        # truth value -- 'Boolean value of Tensor with more than one value /
        # with no values is ambiguous' -- and every probe gave no verdict.
        _di = getattr(mr, "device_indices", None)
        dev = 0 if _di is None else int(_di.numel() if hasattr(_di, "numel") else len(_di))
        host = int(getattr(mr, "host_hit_length", 0) or 0)
        depth = dev + host
        if depth <= 0:
            return 0
        node = getattr(mr, "last_host_node", None) if host > 0 else getattr(mr, "last_device_node", None)
        if node is not None and not _node_has_state(node):
            return 0  # the #928 (a) refusal zeroes the whole match
        return min(int(depth), int(told))
    except Exception as exc:  # noqa: BLE001 - a probe never breaks publication
        logger.warning("#TF told-fidelity probe skipped for rid=%s: %r",
                       str(getattr(req, "rid", "?"))[:12], exc)
        return None


def absolute_depth(req, told: int, absolute: bool) -> int:
    """The prefix depth (keys) PP0's admission reaches with ``told``: the told
    itself when it is absolute (TW twin / TK ABS-TOLD), else the head the
    registration matched plus the span-relative told (#1400 record)."""
    if absolute:
        return int(told)
    return int(getattr(req, "_prefetch_registered_prefix_len", 0) or 0) + int(told)


def pp0_verdict(scheduler, req, told: int, absolute: bool = True) -> tuple:
    """(told_final, own): ``told`` unchanged, or 0 when PP0's own tree cannot
    admit it. ``own`` is the probe's answer in keys from 0 (None = no probe)."""
    if int(told) <= 0 or not enabled():
        return int(told), None
    depth = absolute_depth(req, told, absolute)
    own = pp0_admissible(scheduler, req, depth)
    if own is None or own >= depth:
        return int(told), own
    n = getattr(scheduler, "_tf_retold_n", 0) + 1
    try:
        scheduler._tf_retold_n = n
    except Exception:  # noqa: BLE001
        pass
    logger.warning(
        "#TF TOLD-FIDELITY rid=%s told=%d depth=%d pp0_admissible=%d (n=%d): PP0's own tree "
        "cannot resume at its told (head rows or anchor state gone since the read) -- "
        "the Admit carries told=0 for EVERY rank (PF fallback: every read released, "
        "every stage prefills from 0) instead of PP0 admitting a different prefix than "
        "the followers adopt (rc12k27 b1: W27 start split 512 vs 16895)",
        str(getattr(req, "rid", "?"))[:16], int(told), int(depth), int(own), n,
    )
    return 0, own
