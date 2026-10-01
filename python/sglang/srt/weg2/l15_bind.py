"""AP L15-11c: pure bindings from scheduler requests to retain_at_sleep.

Duck-typed adapter layer (getattr with defaults, no scheduler import, no
torch.cuda): translates Req-shaped objects into the callables that
l15_retain.retain_at_sleep expects, so tests can use SimpleNamespace fakes.

OPEN (L15-11c part 2): l2_of returns ((), ()) -- the L2 eviction ring is
not wired yet; the manifest's l2 columns stay empty until that lands.
"""

from typing import Callable, Dict, Iterable, Tuple

from sglang.srt.weg2.l15_shadow import candidates_from, rows_split


def _seq_len(req) -> int:
    """Token span of a running req: prompt length + generated length."""
    prompt = getattr(req, "origin_input_ids", None) or ()
    out = getattr(req, "output_ids", None) or ()
    return len(prompt) + len(out)


def slots_of_req(req, req_to_token) -> Tuple[int, ...]:
    """The req's KV slots in token order, as plain Python ints.

    The held span is seqlen - 1: the last generated token has not been
    forwarded yet, so its slot is unwritten (schedule_batch.py:2821,
    offload_kv_cache, reads the same span). A 0 INSIDE the span is the
    padding slot (freed/never-written row): raise.
    """
    row = int(req.req_pool_idx)
    n = max(_seq_len(req) - 1, 0)
    span = req_to_token[row, :n]
    if hasattr(span, "tolist"):
        slots = tuple(int(x) for x in span.tolist())
    else:
        slots = tuple(int(x) for x in span)
    if 0 in slots:
        raise ValueError(
            "padding slot 0 inside the token span of req "
            f"{getattr(req, 'rid', row)!r} at row {row}"
        )
    return slots


def anchor_slot_of_req(req) -> int:
    """The req's mamba anchor slot; a missing (None) or padding (0) anchor
    raises -- one req without a holdable anchor skips the whole retain round
    (benign: this runs before step 3, nothing is touched yet)."""
    idx = getattr(req, "mamba_pool_idx", None)
    rid = getattr(req, "rid", "?")
    if idx is None:
        raise ValueError(f"req {rid!r} has no mamba_pool_idx")
    try:
        anchor = int(idx)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"req {rid!r} has a non-scalar mamba_pool_idx: {idx!r}"
        ) from exc
    if anchor == 0:
        raise ValueError(f"req {rid!r} anchor slot is padding slot 0")
    return anchor


def node_of_req(req):
    """The req's radix-tree node; a missing (None) node raises -- the retain
    round skips (benign, pre-step-3: nothing is touched yet)."""
    node = getattr(req, "last_node", None)
    if node is None:
        raise ValueError(f"req {getattr(req, 'rid', '?')!r} has no last_node")
    return node


def build_retain_kwargs(
    reqs: Iterable,
    req_to_token,
    *,
    caps_rows_by_rank,
    cap_anchor_slots: int,
    prefix,
    rank: int,
    epoch: int,
    pid: int,
    kv_buffers,
    mamba_buffers,
    allocator,
    reset_keep: Callable[[list], None],
    set_keep: Callable[[object, Tuple], None],
    manifest_path: str,
    log: Callable[[str], None],
    mamba_allocator=None,
) -> Dict:
    """Assemble the whole retain_at_sleep keyword set from live reqs.

    Geometry mirrors the L15 shadow hook: rows split over n_ranks =
    len(prefix) - 1, anchor_depth = kv_depth = seqlen - 1 (the last token
    has no KV yet); kind/last_active are duck-typed (l15_kind /
    l15_last_active) until the hook lands.
    """
    n_ranks = max(int(len(prefix)) - 1, 1)
    # Rows follow the owner-weighted token vector (the prefix widths), not an
    # even split: mirroring the L15 shadow hook, which passes
    # get_cp_token_ratios(). An even split under-prices a 7/16-owned rank
    # against its cap and over-admits past the compact keep window (OOM class).
    # A malformed/empty vector falls back to the even split inside rows_split.
    ratios = [int(prefix[i + 1]) - int(prefix[i]) for i in range(len(prefix) - 1)]
    by_rid = {}
    entries = []
    for req in reqs:
        rid = str(req.rid)
        by_rid[rid] = req
        seq_len = _seq_len(req)
        # KV exists for seqlen - 1 tokens only (schedule_batch.py:2821).
        span = max(seq_len - 1, 0)
        entries.append(
            {
                "rid": rid,
                "kind": getattr(req, "l15_kind", "served") or "served",
                "last_active": float(
                    getattr(req, "l15_last_active", 0.0) or 0.0
                ),
                "rows_by_rank": rows_split(span, n_ranks, ratios),
                "anchor_depth": span,
                "kv_depth": span,
            }
        )
    candidates = candidates_from(entries)

    def node_of(rid: str):
        return node_of_req(by_rid[rid])

    def slots_of(rid: str) -> Tuple[int, ...]:
        return slots_of_req(by_rid[rid], req_to_token)

    def anchor_slot_of(rid: str) -> int:
        return anchor_slot_of_req(by_rid[rid])

    def l2_of(rid: str) -> Tuple[Tuple, Tuple]:
        # OPEN (L15-11c part 2): wire to the L2 eviction ring.
        return ((), ())

    return {
        "candidates": candidates,
        "node_of": node_of,
        "slots_of": slots_of,
        "anchor_slot_of": anchor_slot_of,
        "l2_of": l2_of,
        "caps_rows_by_rank": caps_rows_by_rank,
        "cap_anchor_slots": cap_anchor_slots,
        "prefix": prefix,
        "rank": rank,
        "epoch": epoch,
        "pid": pid,
        "kv_buffers": kv_buffers,
        "mamba_buffers": mamba_buffers,
        "allocator": allocator,
        "mamba_allocator": mamba_allocator,
        "reset_keep": reset_keep,
        "set_keep": set_keep,
        "manifest_path": manifest_path,
        "log": log,
    }
