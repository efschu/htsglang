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
    card index). A rank whose card is not named (or the whole env is
    auto/absent) gets cap 0 ("not held here", per l15_policy). Caps come back
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


def cell_bytes_from(pool: Any) -> int:
    """Bytes of KV held per one token row on this rank (0 when unknowable).

    ``k_buffer``/``v_buffer`` are per-layer tensor lists; on paged layouts a
    row holds ``_kv_tokens_per_row`` token slots, so divide. Any failure means
    "cannot price" -> 0 (the shadow caps to 0, i.e. "not held here"), never a
    raised exception on the hot path.
    """
    try:
        ks = getattr(pool, "k_buffer", None)
        vs = getattr(pool, "v_buffer", None)
        if not ks or not vs or len(ks) != len(vs):
            return 0
        t = ks[0]
        if not hasattr(t, "numel"):
            return 0
        per_token = 2 * len(ks) * int(t.numel()) * int(t.element_size())
        tpr = int(getattr(pool, "_kv_tokens_per_row", 1) or 1)
        return int(per_token // tpr)
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
    for i in range(t - sum(base)):  # remainder < n, ranks in order
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
