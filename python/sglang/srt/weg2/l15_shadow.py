"""L15-02b: the L1.5 SHADOW instrument -- log lines only, no behaviour change.

Pure and stdlib-only, so the selection logic is unit-testable without a
scheduler. The three hooks (scheduler sleep flush, unified_radix_cache
load-back, front D->P flip) call into this module only when
``SGLANG_WEG2_L15_SHADOW`` is on; with the switch off the hooks produce
byte-identical behaviour (no new line, no new work).

The ledger carries one sleep's shadow hold-set rids to the next wake's
load-back lines, so an ``at=load`` line prices exactly the requests the
shadow policy *would have* kept resident (N1's price of the L1.5 hold).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.weg2.l15_plan import parse_l15_mib
from sglang.srt.weg2.l15_policy import Candidate

SHADOW_ENV = "SGLANG_WEG2_L15_SHADOW"
L15_MIB_ENV = "SGLANG_WEG2_L15_MIB"
_MIB = 2**20
_VALID_KINDS = ("seat", "parked", "served")


def shadow_on(env: Mapping[str, str]) -> bool:
    """True when the SHADOW instrument is switched on in *env*."""
    return str(env.get(SHADOW_ENV, "")).strip().lower() in ("1", "true", "on")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def candidates_from(entries: Sequence[Mapping[str, Any]]) -> List[Candidate]:
    """Build Candidates from plain dicts; invalid entries are skipped.

    Expected keys: rid, kind, last_active, rows_by_rank, anchor_depth,
    kv_depth. An entry that is not a mapping, carries an unknown kind, or
    has a malformed numeric field is skipped, never raised on: a shadow
    instrument must not break the path it observes.
    """
    out: List[Candidate] = []
    for e in entries:
        try:
            if not isinstance(e, dict):
                continue
            rid = e["rid"]
            kind = e["kind"]
            last_active = e["last_active"]
            rows = e["rows_by_rank"]
            anchor = e["anchor_depth"]
            kv = e["kv_depth"]
            if not isinstance(rid, str) or rid == "":
                continue
            if kind not in _VALID_KINDS:
                continue
            if not isinstance(last_active, (int, float)) or isinstance(
                last_active, bool
            ):
                continue
            if isinstance(rows, (str, bytes)) or not hasattr(rows, "__iter__"):
                continue
            rows_t = tuple(rows)
            if not rows_t or not all(_is_int(v) and v >= 0 for v in rows_t):
                continue
            if not _is_int(anchor) or anchor < 0 or not _is_int(kv) or kv < 0:
                continue
            out.append(
                Candidate(
                    rid=rid,
                    kind=kind,
                    last_active=float(last_active),
                    rows_by_rank=rows_t,
                    anchor_depth=anchor,
                    kv_depth=kv,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def caps_from_env(
    env: Mapping[str, str],
    n_ranks: int,
    cell_bytes_by_rank: Sequence[int],
    card_of_rank: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Per-rank row caps from ``SGLANG_WEG2_L15_MIB`` overrides.

    rows_r = floor(mib_c * 2**20 / cell_bytes_r) where c is the physical card
    rank r sits on (``card_of_rank``; None = identity, i.e. rank index is the
    card index). ``c`` is the budget ordinal: the ranks only ever see the
    ordinal form -- card-identity keys are rewritten into it by the launcher
    (``l15_plan.resolve_l15_mib``). A rank whose card is not named (or the
    whole env is auto/absent) gets cap 0 ("not held here", per l15_policy);
    zero, one or several ranks may be cap 0 (``cap0_ranks``). Caps come back
    in the rank order the caller gave (index = position in
    *cell_bytes_by_rank*).
    """
    mode, mib_by_card = parse_l15_mib(env.get(L15_MIB_ENV))
    caps: List[int] = []
    for rank in range(n_ranks):
        card = rank if card_of_rank is None else int(card_of_rank[rank])
        mib = mib_by_card.get(card, 0) if mode == "override" else 0
        cell = int(cell_bytes_by_rank[rank])
        caps.append((mib * _MIB) // cell if cell > 0 and mib > 0 else 0)
    return tuple(caps)


def cap0_ranks(
    env: Mapping[str, str],
    n_ranks: int,
    card_of_rank: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """The ranks whose cap ``caps_from_env`` makes 0 -- "not held here",
    refilled from L2 at the wake -- for any positive cell size (a MiB figure
    > 0 always prices >= 1 row at one byte per row). HW-GENERIC 1002: which
    ranks those are follows from the per-card MiB and the rank->card map
    alone, so another inventory may yield none or several."""
    caps = caps_from_env(env, n_ranks, [1] * int(n_ranks), card_of_rank)
    return tuple(r for r, c in enumerate(caps) if c == 0)


def cap0_line(ranks: Sequence[int]) -> Optional[str]:
    """The launcher warning when the cap-0 rank count is not exactly one:
    the code treats every cap-0 rank as a refill rank, but metal only ever
    ran exactly one. None for exactly one (the proven shape)."""
    ranks = tuple(int(r) for r in ranks)
    if len(ranks) == 1:
        return None
    return (f"L15-CAP0 ranks={','.join(str(r) for r in ranks) or '-'} "
            "(metal-proven only for exactly one cap-0 rank)")


def kv_pool_of(pool: Any) -> Any:
    """The pool that owns the KV ``k_buffer``/``v_buffer``: a hybrid wrapper
    (``HybridLinearKVPool``, the 27B FA+GDN pool) keeps them on
    ``.full_kv_pool`` and has no ``k_buffer`` of its own (N1 boot
    dkr27browauthoritybar1fs10011036: every L15-SHADOW line cap=0,0,0 although
    SGLANG_WEG2_L15_MIB=c1=7616,c2=1792 reached the D ranks). Same unwrap as
    dual_p_kv_stage.stage_pools."""
    inner = getattr(pool, "full_kv_pool", None)
    return inner if inner is not None else pool


def kv_buffers_of(pool: Any) -> List[Any]:
    """Every per-layer K and V tensor of the (unwrapped) pool, K first; [] when
    the pool has none (the L15 retain then must not plan a KV move)."""
    p = kv_pool_of(pool)
    out: List[Any] = []
    for lst in (getattr(p, "k_buffer", None), getattr(p, "v_buffer", None)):
        for t in (lst or ()):
            if hasattr(t, "data_ptr") and hasattr(t, "numel"):
                out.append(t)
    return out


def cell_bytes_from(pool: Any) -> int:
    """Bytes of KV held per one token row on this rank (0 when unknowable).

    ``k_buffer``/``v_buffer`` are per-layer tensor lists of shape (rows,
    heads, head_dim); one token's K+V bytes = sum over the layers of one ROW
    of K and of V. On paged layouts a row holds ``_kv_tokens_per_row`` token
    slots, so divide. A hybrid pool is unwrapped (``kv_pool_of``). Any failure
    means "cannot price" -> 0 (the shadow caps to 0, i.e. "not held here"),
    never a raised exception on the hot path.

    (N1 fix: the earlier form multiplied the WHOLE layer tensor's numel, i.e.
    the full pool's bytes, so even an unwrapped pool priced a cell of
    gigabytes and every cap came out 0.)
    """
    try:
        p = kv_pool_of(pool)
        ks = getattr(p, "k_buffer", None)
        vs = getattr(p, "v_buffer", None)
        if not ks or not vs or len(ks) != len(vs):
            return 0
        total = 0
        for t in list(ks) + list(vs):
            if not hasattr(t, "numel") or not getattr(t, "shape", None):
                return 0
            rows = int(t.shape[0])
            if rows <= 0:
                return 0
            total += (int(t.numel()) // rows) * int(t.element_size())
        tpr = int(getattr(p, "_kv_tokens_per_row", 1) or 1)
        return int(total // tpr)
    except Exception:  # noqa: BLE001 - estimate only, never break the flush
        return 0


def rows_split(
    tokens: int, n_ranks: int, ratios: Optional[Sequence[int]] = None
) -> Tuple[int, ...]:
    """Split *tokens* rows over *n_ranks* ranks; the parts sum to *tokens*.

    With *ratios* (the uneven-DCP token vector from
    ``sglang.srt.distributed.utils.get_cp_token_ratios``): proportional
    floor per rank, the remainder handed to the ranks in index order.
    Without ratios (or with a malformed vector): an even split, remainder
    to the leading ranks in order. Never raises -- a shadow instrument must
    not break the path it observes; a negative *tokens* clamps to 0.
    """
    n = int(n_ranks)
    if n <= 0:
        return ()
    t = int(tokens)
    if t < 0:
        t = 0
    vec: Optional[List[int]] = None
    if ratios is not None:
        try:
            vec = [int(x) for x in ratios]
        except (TypeError, ValueError):
            vec = None
        if vec is not None and (
            len(vec) != n or any(v < 0 for v in vec) or sum(vec) <= 0
        ):
            vec = None  # malformed vector -> even split
    if vec is None:
        base, rem = divmod(t, n)
        return tuple(base + (1 if i < rem else 0) for i in range(n))
    total = sum(vec)
    base = [(t * v) // total for v in vec]
    # remainder < #{ranks with share > 0}: a zero-share rank (the NF form's
    # rank 0) owns no slot under the owner rule, so the remainder must skip
    # it -- handing it a row would price a hold on a rank that cannot hold.
    for i in [j for j in range(n) if vec[j] > 0][: t - sum(base)]:
        base[i] += 1
    return tuple(base)


class ShadowLedger:
    """Carries the last sleep's shadow hold-set rids to the next wake.

    Reset per sleep by :meth:`note_sleep` (accepts a HoldSet or any rid
    sequence). :meth:`note_load` emits the ``at=load`` line only for a rid
    that was in the last shadow hold set -- that is the pair the shadow
    boot prices (kept resident, then still loaded back).
    """

    def __init__(self) -> None:
        self._held: set = set()

    def note_sleep(self, hold) -> None:
        rids = hold.rids if hasattr(hold, "rids") else hold
        self._held = set(rids)

    def note_load(self, rid: str, ms: Any, read_ms: Any) -> Optional[str]:
        if rid in self._held:
            return f"L15-SHADOW at=load rid={rid} ms={ms} read_ms={read_ms}"
        return None


# Process-wide singleton: the scheduler sleep hook writes, the
# unified_radix_cache load hook reads; both in the same worker process.
LEDGER = ShadowLedger()
