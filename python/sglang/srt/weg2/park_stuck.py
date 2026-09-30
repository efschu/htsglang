"""#287 NEED0 (c): requests parked over N D phases without progress, as an
IPC field of the front's ``/weg2/state`` (state.json ``front.d_park_stuck``).

NF y4k (dff1a7fed4, 09301110): ``weg2-0-4`` (25-token prompt, SHORT to D)
was parked at the first D->P flip 0.65 s after its admission and then
PARK-RESUMEd / PARK-RUNNING twelve times -- one D seat held through the whole
dmatrix bench, bs6 ran as bs5 -- until ``WEG2-SERVED ... wall=525.55s`` for
two tokens. Nothing counted it: every park line was a new line, no field held
the streak.

A STREAK is the number of consecutive parks of one rid with no output of that
rid in between (streamed chunks the front forwarded; a non-streamed request
shows none until it ends, so every park of it counts). It ends when the rid
leaves D (its seat is released). The front publishes the rids whose streak
reached ``SGLANG_WEG2_PARK_STUCK_PHASES`` (default 3).
"""

from __future__ import annotations

from typing import Dict, Iterable


class ParkStuck:
    def __init__(self) -> None:
        self.streak: Dict[str, int] = {}
        self._chunks: Dict[str, int] = {}
        self._at_park: Dict[str, int] = {}
        self.max_seen = 0

    def note_output(self, rid) -> None:
        r = str(rid)
        self._chunks[r] = self._chunks.get(r, 0) + 1

    def note_park(self, rids: Iterable) -> None:
        for rid in rids:
            r = str(rid)
            c = self._chunks.get(r, 0)
            if r in self.streak and self._at_park.get(r) == c:
                self.streak[r] += 1
            else:
                self.streak[r] = 1
            self._at_park[r] = c
            self.max_seen = max(self.max_seen, self.streak[r])

    def done(self, rid) -> None:
        r = str(rid)
        self.streak.pop(r, None)
        self._chunks.pop(r, None)
        self._at_park.pop(r, None)

    def block(self, min_phases: int) -> dict:
        n = max(1, int(min_phases))
        stuck = sorted((r for r, s in self.streak.items() if s >= n),
                       key=lambda r: -self.streak[r])
        return {
            "min_phases": n,
            "stuck": len(stuck),
            "max_streak": max(self.streak.values(), default=0),
            "max_streak_boot": int(self.max_seen),
            "rids": {r: self.streak[r] for r in stuck[:8]},
        }
