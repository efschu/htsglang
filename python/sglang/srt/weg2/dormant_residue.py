# SPDX-License-Identifier: Apache-2.0
"""WEG2-DORMANT-SERVED: what a sleeping group leaves on its cards AFTER it served.

THE DEFECT (rc12 Dauerlauf dkrnfh91bar1dauer09262249, 530fb713ce, 26.09.
22:59:07Z). D TP0 on the 5090 died in an extend with torch.OutOfMemoryError
("80 MiB, 50.75 MiB free ... Process 477 has 1.69 GiB"). Process 477 is P's
PP0, asleep on the same card. The D budget charged it

    budget D ordinal=0 5090: 29624 = 32607 - corridor 1171
                                     - dormant_other 1320 - awake_overshoot 489

and the 1320 is the launcher's ONE measurement, taken at P's FIRST sleep --
before P ever ran a prefill (P.log:3476, nvml_proc 1320, allocated - paused =
96 MiB). After serving, the same sleeping rank holds 1730-1802 MiB (P.log
15412/21377: allocated - paused = 479 MiB of LIVE untagged tensors). Nobody
booked the +482 MiB, so D's awake peak met the card edge. rc11b
(dkrnfh91bar1dauer09262231, 09262218) shows the same growth on all three P
ranks; it never reached the extend peak while P sat at its high.

THE MEASUREMENT (generic, any model, any group): every sleep prints one
``WEG2-DC-BREAKDOWN stage=release tags=[...] nvml_proc=N MiB`` per rank. The
line whose tags carry a weight tag closes that sleep; its ``nvml_proc`` is what
the rank leaves on the card while dormant. Per rank the FIRST such line is the
fresh-boot residue (the launcher's own measurement), and the MAX over the later
ones is the served residue. The growth is their difference, placed on the card
the rank sits on (``WEG2-XCHG-MANIFEST-WRITE group=G rank=R card=C``).

:func:`served_growth` reads logs; ``python -m sglang.srt.weg2.dormant_residue
<P.log> ...`` prints the record row (``P_DORMANT_SERVED_GROWTH_MIB``) for
weg2/profile_records_data/<profile>.json -- max over the boots given, so a
profile's record only grows with evidence. The launcher charges it on top of
the measured dormant_other of the D budget (``budgets_from_dc``).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

RECORD_NAME = "P_DORMANT_SERVED_GROWTH_MIB"
MARKER = "WEG2-DORMANT-SERVED"

_RELEASE = re.compile(
    r"\[[^\]]*? (?P<rank>(?:PP|TP)\d+)\] WEG2-DC-BREAKDOWN stage=release "
    r"tags=\[(?P<tags>[^\]]*)\] nvml_proc=(?P<mib>\d+) MiB"
)
_CARD = re.compile(
    r"\[[^\]]*? (?P<rank>(?:PP|TP)\d+)\] WEG2-XCHG-MANIFEST-WRITE group=\w+ "
    r"rank=\d+ card=(?P<card>\d+)"
)


class DormantResidueError(ValueError):
    """A log that cannot price the growth -- refused, never guessed."""


@dataclass(frozen=True)
class RankResidue:
    rank: str
    card: int
    fresh_mib: int
    served_max_mib: int
    sleeps: int

    @property
    def growth_mib(self) -> int:
        return max(0, self.served_max_mib - self.fresh_mib)


def _closes_a_sleep(tags: str) -> bool:
    return any(t.strip().strip("'\"").startswith("weights") for t in tags.split(","))


def rank_residues(lines: Iterable[str]) -> List[RankResidue]:
    """Per rank: fresh (first) and served (max of the later) dormant residue."""
    seen: Dict[str, List[int]] = {}
    cards: Dict[str, int] = {}
    for line in lines:
        m = _RELEASE.search(line)
        if m is not None:
            if _closes_a_sleep(m.group("tags")):
                seen.setdefault(m.group("rank"), []).append(int(m.group("mib")))
            continue
        c = _CARD.search(line)
        if c is not None:
            cards.setdefault(c.group("rank"), int(c.group("card")))
    out = []
    for rank in sorted(seen):
        vals = seen[rank]
        if rank not in cards:
            raise DormantResidueError(
                f"{rank}: dormant residue measured but no WEG2-XCHG-MANIFEST-WRITE "
                f"names its card; the growth cannot be placed")
        served = vals[1:]
        out.append(RankResidue(rank=rank, card=cards[rank], fresh_mib=vals[0],
                               served_max_mib=max(served) if served else vals[0],
                               sleeps=len(vals)))
    return out


def served_growth(paths: Sequence[str], n_cards: int) -> Tuple[List[int], List[RankResidue]]:
    """Per card ordinal: max growth over the logs. A card no rank sits on is 0."""
    growth = [0] * n_cards
    ranks: List[RankResidue] = []
    for p in paths:
        with open(p, "r", errors="replace") as f:
            rr = rank_residues(f)
        if not any(r.sleeps > 1 for r in rr):
            raise DormantResidueError(f"{p}: no rank slept after serving; nothing measured")
        for r in rr:
            if not 0 <= r.card < n_cards:
                raise DormantResidueError(f"{p}: {r.rank} on card {r.card} outside 0..{n_cards - 1}")
            growth[r.card] = max(growth[r.card], r.growth_mib)
        ranks.extend(rr)
    return growth, ranks


def format_line(path: str, rr: Sequence[RankResidue]) -> str:
    parts = " ".join(
        f"{r.rank}@card{r.card}:fresh={r.fresh_mib},served_max={r.served_max_mib},"
        f"growth={r.growth_mib},sleeps={r.sleeps}" for r in rr)
    return f"{MARKER} {path}: {parts}"


def record_row(growth: Sequence[int], boots: Sequence[str], provenance: str) -> dict:
    return {
        "name": RECORD_NAME,
        "value": [int(g) for g in growth],
        "provenance": provenance,
        "boots": list(boots),
        "kind": "memory",
        "power_limit_w": None,
        "power_limit_source": "not a power-dependent quantity (bytes a sleeping rank keeps)",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    n_cards = 3
    if args[:1] == ["--cards"]:
        n_cards, args = int(args[1]), args[2:]
    if not args:
        print(f"usage: python -m sglang.srt.weg2.dormant_residue [--cards N] <P.log> ...",
              file=sys.stderr)
        return 2
    growth, _ = served_growth(args, n_cards)
    for p in args:
        with open(p, "r", errors="replace") as f:
            print(format_line(p, rank_residues(f)))
    boots = [re.sub(r"^boot_weg2_([^_]+)_.*$", r"\1", p.rsplit("/", 1)[-1]) for p in args]
    print(json.dumps(record_row(growth, boots, f"{MARKER} over {len(args)} P log(s)"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
