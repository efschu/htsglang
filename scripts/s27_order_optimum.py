"""#631: the JOINT optimal wave order, priced with row blocking in it.

WHY THIS EXISTS. ``s27_seam_ceiling_sweep`` shows the backing transient
saturating around 1322 MiB however fine the row blocking gets -- a 28%
cut, not the 1/B the section 2.1 note promised. The reason is not the
blocking, it is the ORDER it runs under.

Blocking can only shrink a wave that has something to release. In
``pp_to_tp`` rank r commits every one of the 16 TP layers but releases
only its own PP stage, so its own layers are the BLOCKABLE waves and every
other layer is a bare commit that no fine-graining can help. The order
therefore decides almost everything, and the order that is best at B=1 is
not the order that is best at B=32: at B=1 you want releases spread to cap
each prefix, at large B you want every release as EARLY as possible
because the commits they pay for have become nearly free.

THE SEARCH IS EXACT, not a heuristic. Layers within one PP stage are
interchangeable for this objective -- all that matters is which stage each
position belongs to -- so an order is a multiset permutation of stage
labels, and the state (a, b, c) = labels already placed determines every
rank's running balance exactly. That makes a DP over at most
(n0+1)(n1+1)(n2+1) states exact, where a brute force over 1.4M
permutations would be merely slow.

Objective is the MAX OVER RANKS of the peak, because one order serves all
three and the binding card is whichever rank peaks highest.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache

MIB = 1024 * 1024
RIG_MAP_SIZES = (7, 5, 4)
RIG_VEC = (14, 10, 8)


def legs(pool_rows: int, row_bytes: int, direction: str):
    """(commit_per_wave, release_per_wave_if_owned) per rank, in bytes.

    pp_to_tp: commit a TP layer (this rank's head share, whole pool) on
    EVERY wave; release a PP layer (all heads, whole pool) only on waves
    for layers this rank owns.

    tp_to_pp is the mirror: release a TP layer every wave, commit a PP
    layer only on owned waves.
    """
    total = sum(RIG_VEC)
    pp_span = row_bytes * pool_rows
    out = []
    for r in range(3):
        tp_span = int(row_bytes * RIG_VEC[r] / total) * pool_rows
        if direction == "pp_to_tp":
            out.append((tp_span, pp_span, True))  # commit always, release if owned
        else:
            out.append((pp_span, tp_span, False))  # commit if owned, release always
    return out


def _inner(com: int, rel: int, blocks: int) -> int:
    """Peak inside one wave -- the same formula the runtime gate uses."""
    if blocks <= 1:
        return com
    num = max(com, blocks * com - (blocks - 1) * rel)
    return -(-num // blocks)


def solve(pool_rows: int, row_bytes: int, blocks: int, direction: str):
    L = legs(pool_rows, row_bytes, direction)
    n = RIG_MAP_SIZES

    @lru_cache(maxsize=None)
    def best(a: int, b: int, c: int):
        """Min achievable peak over the REMAINING placements, given prefix."""
        placed = (a, b, c)
        if placed == n:
            return 0, ()
        base = []
        for r in range(3):
            com, rel, commit_always = L[r]
            k = sum(placed)
            owned = placed[r]
            if commit_always:
                base.append(k * com - owned * rel)
            else:
                base.append(owned * com - k * rel)
        out = (None, ())
        for s in range(3):
            if placed[s] >= n[s]:
                continue
            step = 0
            for r in range(3):
                com, rel, commit_always = L[r]
                if commit_always:
                    c_w, r_w = com, (rel if s == r else 0)
                else:
                    c_w, r_w = (com if s == r else 0), rel
                step = max(step, base[r] + _inner(c_w, r_w, blocks))
            nxt = list(placed)
            nxt[s] += 1
            sub, tail = best(*nxt)
            peak = max(step, sub)
            if out[0] is None or peak < out[0]:
                out = (peak, (s,) + tail)
        return out

    peak, order = best(0, 0, 0)
    best.cache_clear()
    return peak, order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-bytes", type=int, default=2048)
    ap.add_argument("--at-pool", type=int, default=600000)
    ap.add_argument("--payload-mib", type=float, default=687.0)
    ap.add_argument("--budget-mib", type=float, default=753.0)
    ap.add_argument("--blocks", type=int, nargs="*",
                    default=[1, 2, 4, 8, 16, 32, 64])
    a = ap.parse_args()

    print(f"pool={a.at_pool} row_bytes={a.row_bytes} "
          f"budget={a.budget_mib} MiB payload={a.payload_mib} MiB")
    print(f"{'B':>5} {'pp_to_tp':>10} {'tp_to_pp':>10} {'worst':>10} "
          f"{'+payload':>10} {'fits?':>7}   optimal pp_to_tp stage order")
    for b in a.blocks:
        res = {}
        for d in ("pp_to_tp", "tp_to_pp"):
            peak, order = solve(a.at_pool, a.row_bytes, b, d)
            res[d] = (peak / MIB, order)
        worst = max(v[0] for v in res.values())
        total = worst + a.payload_mib
        seq = "".join("ABC"[s] for s in res["pp_to_tp"][1])
        print(f"{b:>5} {res['pp_to_tp'][0]:>10.1f} {res['tp_to_pp'][0]:>10.1f} "
              f"{worst:>10.1f} {total:>10.1f} "
              f"{'YES' if total <= a.budget_mib else 'no':>7}   {seq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
