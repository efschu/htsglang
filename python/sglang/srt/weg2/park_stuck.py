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

y5c (30.09., weg2-0-2, 113k NON-stream): parked 6x, 1650 tokens decoded in
between -- the front forwards a non-stream answer whole at its end, so it
has no output of such a rid to count and no per-rid IPC source for one. Its
parks are not a streak of "no progress" but a blindness: those rids are
published apart (``blind_nonstream``: rid -> parks), never under ``stuck``.
"""

from __future__ import annotations

from typing import Dict, Iterable


class ParkStuck:
    def __init__(self) -> None:
        self.streak: Dict[str, int] = {}
        self._chunks: Dict[str, int] = {}
        self._at_park: Dict[str, int] = {}
        self.max_seen = 0
        self.nonstream: set = set()

    def note_stream(self, rid, is_stream: bool) -> None:
        if is_stream:
            self.nonstream.discard(str(rid))
        else:
            self.nonstream.add(str(rid))

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
            if r not in self.nonstream:
                self.max_seen = max(self.max_seen, self.streak[r])

    def done(self, rid) -> None:
        r = str(rid)
        self.streak.pop(r, None)
        self._chunks.pop(r, None)
        self._at_park.pop(r, None)
        self.nonstream.discard(r)

    def block(self, min_phases: int) -> dict:
        n = max(1, int(min_phases))
        seen = {r: s for r, s in self.streak.items() if r not in self.nonstream}
        blind = {r: s for r, s in self.streak.items() if r in self.nonstream}
        stuck = sorted((r for r, s in seen.items() if s >= n), key=lambda r: -seen[r])
        return {
            "min_phases": n,
            "stuck": len(stuck),
            "max_streak": max(seen.values(), default=0),
            "max_streak_boot": int(self.max_seen),
            "rids": {r: seen[r] for r in stuck[:8]},
            # non-stream rids: parks counted, progress not visible to the front
            "blind_nonstream": {r: blind[r] for r in sorted(blind, key=lambda r: -blind[r])[:8]},
        }
