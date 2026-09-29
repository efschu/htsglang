"""The host cushion over ONE flip -- an instrument, never a verdict.

Why it exists (z30u, 2026-09-29): W98 fired at 07:14:10 on ``cushion=1.33 <
1.50`` and nothing before it had said where the cushion stood at the flips
that went well. The only per-boot number was ``cushion_min_gib`` in the
FLIP-RATCHET record (once, at done epoch=2). This window follows every flip
from ``begin`` to ``done`` on the SAME readings the rate latch grades
(``host_ledger.read_cgroup_pressure``, fed from the latch loop's 0.5 s tick),
so the course is visible without a latch having to fire.

``cg_room`` = ``memory.max - memory.current``: inside the Docker form the
cgroup's own ceiling (``--memory 84g``) bounds the page cache the cushion is
made of -- z30u ran nonreclaim 79.7-80.8 GiB under 84 GiB, i.e. the cushion
could never exceed ~3-4 GiB whatever held the cache. ``absorb`` = cushion +
the smaller of the two free pools a write can land in (host MemFree, cgroup
room): a DERIVED figure, printed as such, graded by nothing.
"""
from __future__ import annotations

from typing import Dict, List, Optional

GIB = 1024 ** 3


def _cushion(pr: Dict[str, Optional[float]]) -> Optional[float]:
    f, sh = pr.get("file_gib"), pr.get("shmem_gib")
    return None if f is None or sh is None else float(f) - float(sh)


def _room(pr: Dict[str, Optional[float]], cg_max_gib: Optional[float]) -> Optional[float]:
    cur = pr.get("current_gib")
    return None if cg_max_gib is None or cur is None else float(cg_max_gib) - float(cur)


def _min(xs: List[Optional[float]]) -> Optional[float]:
    v = [x for x in xs if x is not None]
    return min(v) if v else None


def _max(xs: List[Optional[float]]) -> Optional[float]:
    v = [x for x in xs if x is not None]
    return max(v) if v else None


def absorb_gib(cushion: Optional[float], memfree: Optional[float],
               room: Optional[float]) -> Optional[float]:
    """Cushion plus the SMALLER free pool; None when the cushion or both pools
    are unreadable (never a zero for an absent term)."""
    pools = [x for x in (memfree, room) if x is not None]
    if cushion is None or not pools:
        return None
    return float(cushion) + max(0.0, min(pools))


class FlipCushionWindow:
    """``open`` at flip begin, ``note`` on every latch tick, ``close`` at done."""

    def __init__(self, cg_max_gib: Optional[float], floor_gib: float) -> None:
        self.cg_max_gib = cg_max_gib
        self.floor_gib = float(floor_gib)
        self._open: Optional[dict] = None
        self._rows: List[dict] = []

    @property
    def is_open(self) -> bool:
        return self._open is not None

    def _row(self, pr: Dict[str, Optional[float]]) -> dict:
        c = _cushion(pr)
        room = _room(pr, self.cg_max_gib)
        mf = pr.get("memfree_gib")
        return {"cushion": c, "memfree": mf, "room": room,
                "nonreclaim": pr.get("nonreclaim_gib"), "shmem": pr.get("shmem_gib"),
                "absorb": absorb_gib(c, mf, room)}

    def open(self, *, epoch: int, src: str, dst: str, pr: Dict[str, Optional[float]]) -> None:
        self._open = {"epoch": int(epoch), "sleep": src, "wake": dst}
        self._rows = [self._row(pr)]

    def note(self, pr: Dict[str, Optional[float]]) -> None:
        if self._open is not None:
            self._rows.append(self._row(pr))

    def close(self, pr: Dict[str, Optional[float]]) -> Optional[dict]:
        if self._open is None:
            return None
        rows = self._rows + [self._row(pr)]
        head, self._open, self._rows = self._open, None, []
        cush = [r["cushion"] for r in rows]
        sh0, sh1 = rows[0]["shmem"], rows[-1]["shmem"]
        rec = dict(head)
        rec.update({
            "samples": len(rows),
            "cushion_begin_gib": cush[0], "cushion_end_gib": cush[-1],
            "cushion_min_gib": _min(cush),
            "below_floor_samples": sum(1 for c in cush if c is not None and c < self.floor_gib),
            "floor_gib": self.floor_gib,
            "memfree_min_gib": _min([r["memfree"] for r in rows]),
            "cg_room_min_gib": _min([r["room"] for r in rows]),
            "cg_max_gib": self.cg_max_gib,
            "nonreclaim_max_gib": _max([r["nonreclaim"] for r in rows]),
            "shmem_delta_gib": None if sh0 is None or sh1 is None else float(sh1) - float(sh0),
            "absorb_min_gib": _min([r["absorb"] for r in rows]),
        })
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}


def _g(v: Optional[float], fmt: str = "%.2f") -> str:
    return "unreadable" if v is None else fmt % v


def line(rec: dict) -> str:
    return (
        f"WEG2-FLIP-CUSHION epoch={rec['epoch']} slept={rec['sleep']} woke={rec['wake']} "
        f"samples={rec['samples']} cushion min={_g(rec['cushion_min_gib'])} "
        f"begin={_g(rec['cushion_begin_gib'])} end={_g(rec['cushion_end_gib'])} GiB "
        f"floor={rec['floor_gib']:.2f} below_floor={rec['below_floor_samples']} "
        f"memfree_min={_g(rec['memfree_min_gib'])} cg_room_min={_g(rec['cg_room_min_gib'])} "
        f"(memory.max {_g(rec['cg_max_gib'])}) nonreclaim_max={_g(rec['nonreclaim_max_gib'])} "
        f"shmem_delta={_g(rec['shmem_delta_gib'], '%+.2f')} absorb_min={_g(rec['absorb_min_gib'])} GiB "
        "(instrument, no verdict: cushion = file - shmem, the latch's own reading; "
        "absorb = cushion + min(MemFree, memory.max - memory.current), DERIVED)"
    )


def cg_max_gib_of(cgroup: Dict[str, Optional[int]]) -> Optional[float]:
    m = cgroup.get("max")
    return None if m is None else int(m) / GIB
