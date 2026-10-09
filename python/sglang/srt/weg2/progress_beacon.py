"""FP FORWARD-PROGRESS BEACON (NF rc12p, 27.09. 14:13:39): "the group computes" is never "the group
is dead".

THE GAP. FH (weg2/front_health.py) counts an http_ok=False streak >= 2 as unhealthy (503; W17 off a
flip). NF measured one /health failure while D ran a 91k extend (200 again 7 s later); an extend of
91k-262k tokens is one 10-60 s forward, so two slow probes in a row read as a dead group.

THE BEACON (switch ``SGLANG_WEG2_PROGRESS_BEACON``, default on; ``0`` = no file, no reading -- the
FH rule byte for byte). Every scheduler rank of a Weg-2 group (``SGLANG_WEG2_GROUP``) keeps one
32-byte mmap file ``<arena>/progress/<GROUP>-pid<pid>.bin`` = (forward_ct, t_start_ns, t_done_ns,
pid): ``beat_start`` at the top of every forward (``Scheduler._run_batch_forward``), ``beat_done``
after its result is processed (``process_batch_result``). One ``struct.pack_into`` each -- no
syscall, no collective, no new endpoint (the /health path and ``/get_server_info`` both wait on the
busy process; the file does not).

THE READING (front, every FH poll): a group shows PROGRESS when any of its live ranks' files
(pid in the group's session) moved ``forward_ct`` since the previous poll, or is inside a forward
(t_start > t_done) begun less than ``BUSY_BOUND_S`` (120 s) ago. An http_ok=False with progress is
logged ``WEG2-HEALTH-BUSY`` and does not count toward the streak. A hold and process_alive=False
stay fatal at once, unchanged.
"""
from __future__ import annotations

import glob
import mmap
import os
import struct
import time
from typing import Callable, Dict, Optional, Tuple

ENV = "SGLANG_WEG2_PROGRESS_BEACON"
SUBDIR = "progress"
_FMT = "<qqqq"
_SIZE = struct.calcsize(_FMT)
#: PRIORITY LANES 1008 (L1, weg2/lanes.py): with SGLANG_WEG2_LANES=1 the file is 16 bytes longer --
#: ``(lane_floor, lane_epoch)`` the rank last saw, after the 32 bytes above. Every reader of the 32 bytes
#: reads on unchanged; with the switch off the file stays 32 bytes.
_LANE_FMT = "<qq"
_LANE_SIZE = struct.calcsize(_LANE_FMT)
#: a forward running longer than this is not progress (the scheduler
#: watchdog's own bound is 300 s; a 262k extend measured <= 60 s).
BUSY_BOUND_S = 120.0


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def beacon_dir(tag: str = "", env=None) -> str:
    from sglang.srt.weg2.resume_via_p import arena_dir

    base = arena_dir(tag, env)
    return os.path.join(base, SUBDIR) if base else ""


def _lanes_on() -> bool:
    """SGLANG_WEG2_LANES (weg2/lanes.py); a beacon never fails on it."""
    try:
        from sglang.srt.weg2 import lanes

        return bool(lanes.enabled())
    except Exception:  # noqa: BLE001
        return False


class _Writer:
    def __init__(self):
        self.mm = None
        self.dead = False

    def _open(self):
        if self.dead or self.mm is not None:
            return self.mm
        try:
            if not enabled():
                self.dead = True
                return None
            group = (os.environ.get("SGLANG_WEG2_GROUP", "") or "").strip().upper()
            d = beacon_dir()
            if not group or not d:
                self.dead = True
                return None
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{group}-pid{os.getpid()}.bin")
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            size = _SIZE + (_LANE_SIZE if _lanes_on() else 0)  # PRIORITY LANES 1008 (L1): +16 bytes only with the switch on
            try:
                os.ftruncate(fd, size)
                self.mm = mmap.mmap(fd, size)
            finally:
                os.close(fd)
            struct.pack_into(_FMT, self.mm, 0, 0, 0, 0, os.getpid())
        except Exception:  # noqa: BLE001 -- a beacon never stops a forward
            self.dead = True
            self.mm = None
        return self.mm

    def beat(self, forward_ct: int, start: bool) -> None:
        mm = self._open()
        if mm is None:
            return
        try:
            ct, ts, td, pid = struct.unpack_from(_FMT, mm, 0)
            now = time.time_ns()
            if start:
                struct.pack_into(_FMT, mm, 0, int(forward_ct), now, td, pid)
            else:
                struct.pack_into(_FMT, mm, 0, int(forward_ct), ts, now, pid)
        except Exception:  # noqa: BLE001
            pass


    def beat_lane(self, floor: int, epoch: int) -> None:
        """PRIORITY LANES 1008 (L1): the lane floor/epoch this rank last saw (trailer after the 32 bytes).
        No trailer (switch off, file of 32 bytes) = nothing written."""
        mm = self._open()
        if mm is None or len(mm) < _SIZE + _LANE_SIZE:
            return
        try:
            struct.pack_into(_LANE_FMT, mm, _SIZE, int(floor), int(epoch))
        except Exception:  # noqa: BLE001
            pass


_W = _Writer()


def beat_start(forward_ct: int) -> None:
    _W.beat(forward_ct, True)


def beat_done(forward_ct: int) -> None:
    _W.beat(forward_ct, False)


def beat_lane(floor: int, epoch: int) -> None:
    """PRIORITY LANES 1008 (L1): record the lane floor/epoch the rank saw (parts L2 / L3 call it where the
    rank takes a new floor). A no-op unless SGLANG_WEG2_LANES=1."""
    _W.beat_lane(floor, epoch)


def read_group(directory: str, group: str, sid: int,
               session_of: Optional[Callable[[int], Optional[int]]] = None) -> Dict[int, Tuple[int, int, int]]:
    """{pid: (forward_ct, t_start_ns, t_done_ns)} of the group's LIVE ranks."""
    if not directory or not sid:
        return {}
    if session_of is None:
        from sglang.srt.weg2.front_health import pid_session as session_of
    out: Dict[int, Tuple[int, int, int]] = {}
    for path in glob.glob(os.path.join(directory, f"{group}-pid*.bin")):
        try:
            with open(path, "rb") as fh:
                raw = fh.read(_SIZE)
            ct, ts, td, pid = struct.unpack(_FMT, raw)
        except Exception:  # noqa: BLE001
            continue
        if session_of(int(pid)) != int(sid):
            continue
        out[int(pid)] = (int(ct), int(ts), int(td))
    return out


def read_group_lane(directory: str, group: str, sid: int,
                    session_of: Optional[Callable[[int], Optional[int]]] = None) -> Dict[int, Tuple[int, int]]:
    """PRIORITY LANES 1008 (L1): ``{pid: (lane_floor, lane_epoch)}`` of the group's LIVE ranks whose beacon
    file carries the lane trailer (switch on); a rank with a 32-byte file is absent."""
    if not directory or not sid:
        return {}
    if session_of is None:
        from sglang.srt.weg2.front_health import pid_session as session_of
    out: Dict[int, Tuple[int, int]] = {}
    for path in glob.glob(os.path.join(directory, f"{group}-pid*.bin")):
        try:
            with open(path, "rb") as fh:
                raw = fh.read(_SIZE + _LANE_SIZE)
            if len(raw) < _SIZE + _LANE_SIZE:
                continue
            pid = struct.unpack_from(_FMT, raw, 0)[3]
            floor, epoch = struct.unpack_from(_LANE_FMT, raw, _SIZE)
        except Exception:  # noqa: BLE001
            continue
        if session_of(int(pid)) != int(sid):
            continue
        out[int(pid)] = (int(floor), int(epoch))
    return out


def progress(prev: Dict[int, Tuple[int, int, int]], cur: Dict[int, Tuple[int, int, int]],
             now_ns: Optional[int] = None, bound_s: float = BUSY_BOUND_S) -> Optional[str]:
    """Why the group shows progress (a short reason), or None."""
    now_ns = time.time_ns() if now_ns is None else now_ns
    for pid, (ct, ts, td) in cur.items():
        p = prev.get(pid)
        if p is not None and ct != p[0]:
            return f"forward_ct {p[0]}->{ct} (pid {pid})"
        if ts > td and (now_ns - ts) < bound_s * 1e9:
            return f"in forward {ct} for {(now_ns - ts) / 1e9:.1f}s (pid {pid})"
    return None


def lane_hold(front_floor: int, front_epoch: int,
              group_lane: Optional[Dict[int, Tuple[int, int]]] = None,
              held: bool = False, acked: bool = False) -> Optional[str]:
    """PRIORITY LANES 1008 (L4): why a group that shows NO forward progress is on a LANE HOLD and not stalled.

    A group whose requests all wait behind a higher lane runs no forward: ``forward_ct`` stands, nothing is in
    forward -- the picture a watchdog (W17 / H86 / the front's leg-1 stall) reads as a stall. It is a hold when
    the front keeps the lane floor above 0, the front holds requests of this group below it (``held``: parked on
    D, in flight on P, or waiting for it) AND the group knows the floor: either a rank's beacon trailer (written
    by L2 / L3 with :func:`beat_lane`) shows the front's epoch, or the front's floor RPC to the group was
    acknowledged (``acked``; a group without the trailer, 32-byte files, is read through the RPC alone).
    Returns a short reason or None (None = no hold: the stall reading stands)."""
    if not held or int(front_floor) <= 0:
        return None
    for pid, (floor, epoch) in sorted((group_lane or {}).items()):
        if int(epoch) == int(front_epoch) and int(floor) == int(front_floor):
            return f"lane-hold floor={int(floor)} epoch={int(epoch)} (pid {pid})"
    if acked:
        return f"lane-hold floor={int(front_floor)} epoch={int(front_epoch)} (floor RPC acknowledged)"
    return None


# ---------------------------------------------------------------------------
# FLIPZEIT D>P END (user order 02.10., NF + 27B identical names): FLIPZEIT runs
# from the last token of the outgoing phase to the first token of the incoming
# one ("... zu erstes Token Decode oder PREFILL BATCH BEGINN"). For D>P the end
# is the BEGIN OF THE FIRST FORWARD ON THE FIRST P PIPELINE STAGE (PP0) after
# the wake -- not the leg-1 dispatch, and not the last stage either: PP0 ->
# PP1 -> PP2 of the first chunk is pipeline fill, i.e. prefill compute (y7l:
# ~6.5 s from PP0's start to PP2's), not flip. Source: this beacon's
# ``t_start_ns`` of the PP0 rank at its first ``forward_ct`` rise after
# ``flip_done`` (``prefill_start_source="pp_first_forward"``). The PP-last
# rank's first rise is read alongside (``pp_last_start_ts``) so the pipeline
# fill can be shown as prefill. Unavailable -> ``"missing"`` and no timestamp.
# There is NO fallback (never the leg-1 dispatch, never leg-1 end minus P's
# own prefill time).
# ---------------------------------------------------------------------------

SOURCE_PP_FIRST = "pp_first_forward"
SOURCE_MISSING = "missing"
#: the probe stops polling after this long without a rise (an idle-layout flip
#: waits for its first request; 30 min is far past any phase dwell)
PP_PROBE_TIMEOUT_S = 1800.0
#: poll period of the probe thread. A P stage's forward of a prefill chunk
#: lasts tens of ms to seconds; a reading that comes later than ONE forward
#: shows as a forward_ct jump > 1 and is reported ``missing`` (late_read),
#: never as a value.
PP_PROBE_POLL_S = 0.002
#: the two stages the probe reads
STAGES = ("first", "last")


def pp_rank_of(pid: int) -> Optional[int]:
    """The pipeline stage of a scheduler rank from its process title
    (``sglang::scheduler[_DPd][_PPd][_TPd]...``, scheduler.run_scheduler_process
    sets it): the ``PP`` index, ``0`` when the title carries none (pp_size 1),
    None when the pid is no scheduler or its title is unreadable."""
    import re

    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as fh:
            raw = fh.read(4096).replace(b"\0", b" ").decode("utf-8", "replace")
    except (OSError, ValueError):
        return None
    m = re.search(r"sglang::scheduler(\S*)", raw)
    if m is None:
        return None
    pp = re.search(r"_PP(\d+)", m.group(1))
    return int(pp.group(1)) if pp else 0


def pp_stage_pids(beacons: Dict[int, Tuple[int, int, int]],
                  pp_of: Callable[[int], Optional[int]] = pp_rank_of
                  ) -> Tuple[Dict[str, Dict[int, int]], Optional[str]]:
    """``({"first": {pid: pp}, "last": {pid: pp}}, None)`` -- the ranks (all TP
    ranks) of the group's FIRST and LAST pipeline stage (the same with
    pp_size 1) -- else ``({}, reason)``. One rank of unknown stage makes both
    unknown: that rank could be exactly the first or the last stage."""
    if not beacons:
        return {}, "no_beacon"
    stages: Dict[int, int] = {}
    for pid in beacons:
        r = pp_of(int(pid))
        if r is None:
            return {}, f"pp_rank_unknown pid={pid}"
        stages[int(pid)] = int(r)
    lo, hi = min(stages.values()), max(stages.values())
    return {"first": {pid: r for pid, r in stages.items() if r == lo},
            "last": {pid: r for pid, r in stages.items() if r == hi}}, None


def first_rise(baseline: Dict[int, int], cur: Dict[int, Tuple[int, int, int]]) -> Optional[dict]:
    """One stage's first forward after the baseline: ``{ts, pid, ct}``
    (``ts`` = that rank's ``t_start_ns`` in seconds), ``{missing: reason}`` when
    the reading came after a later forward already overwrote the first one's
    start (forward_ct rose by more than one), None while no rank rose."""
    best = None
    late = None
    for pid, base in baseline.items():
        c = cur.get(pid)
        if c is None or c[0] <= base:
            continue
        ct, ts, _td = c
        if ct - base > 1:
            late = late or f"late_read pid={pid} forward_ct {base}->{ct}"
            continue
        if best is None or ts < best["ts_ns"]:
            best = {"ts_ns": int(ts), "pid": int(pid), "ct": int(ct)}
    if best is not None:
        return {"ts": best["ts_ns"] / 1e9, "pid": best["pid"], "ct": best["ct"]}
    if late is not None:
        return {"missing": late}
    return None


def _read_one(path: str) -> Optional[Tuple[int, int, int, int]]:
    try:
        with open(path, "rb") as fh:
            raw = fh.read(_SIZE)
        return struct.unpack(_FMT, raw)
    except Exception:  # noqa: BLE001
        return None


class PpForwardProbe:
    """Built at a D->P ``flip_done``: the baseline ``forward_ct`` of the woken
    group's FIRST and LAST pipeline stage. :meth:`start` runs a daemon thread
    that reads their beacon files every :data:`PP_PROBE_POLL_S` until both
    stages rose (or :meth:`stop`, or the timeout) and calls
    ``on_result(stage, result)`` once per stage -- ``{ts, pid, ct, pp_rank}``
    or ``{missing: reason}``. ``results["first"]`` is the D->P flip's end.
    :meth:`poll` reads once synchronously. Never raises."""

    def __init__(self, directory: str, group: str, sid: int, *,
                 session_of: Optional[Callable[[int], Optional[int]]] = None,
                 pp_of: Callable[[int], Optional[int]] = pp_rank_of,
                 poll_s: float = PP_PROBE_POLL_S, timeout_s: float = PP_PROBE_TIMEOUT_S) -> None:
        import threading

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.results: Dict[str, Optional[dict]] = {s: None for s in STAGES}
        self.on_result: Optional[Callable[[str, dict], None]] = None
        self.baseline: Dict[str, Dict[int, int]] = {s: {} for s in STAGES}
        self.stages: Dict[str, Dict[int, int]] = {s: {} for s in STAGES}
        self.paths: Dict[int, str] = {}
        self.poll_s, self.timeout_s = float(poll_s), float(timeout_s)
        self._thread = None
        #: PDFLIP-E3: (directory, group, sid, session_of, pp_of) while the done
        #: reading was EMPTY -- each poll rescans for the ranks' first files
        self._rescan = None
        try:
            if not enabled():
                self._set_all({"missing": f"beacon_off {ENV}=0"})
                return
            if not directory:
                self._set_all({"missing": "no_beacon_dir"})
                return
            if not sid:   # no session to read the ranks of: unreadable, not an empty reading
                self._set_all({"missing": "no_beacon"})
                return
            cur = read_group(directory, group, sid, session_of)
            if not cur:
                # PDFLIP-E3 (NF y7n epoch=1 reason=no_beacon; 27B N5f 13:30:08): before
                # P's first wake no P rank has run a forward, so no beacon file exists
                # at done. An empty reading is a baseline of forward_ct 0 -- every PP0
                # file that appears is that rank's first forward (the end). The last
                # stage needs the known rank set: on an empty baseline it is missing.
                self._rescan = (directory, group, sid, session_of, pp_of)
                self._set("last", {"missing": "empty_baseline"})
                return
            stages, why = pp_stage_pids(cur, pp_of)
            if why is not None:
                self._set_all({"missing": why})
                return
            self.stages = stages
            for s in STAGES:
                self.baseline[s] = {pid: cur[pid][0] for pid in stages[s]}
                for pid in stages[s]:
                    self.paths[pid] = os.path.join(directory, f"{group}-pid{pid}.bin")
            #: the baseline reading itself (ct, t_start_ns, t_done_ns) per pid, kept for
            #: :meth:`resolve_started_after` (Y8P-PPFWD-ARM-RACE)
            self.base_reading = dict(cur)
        except Exception as e:  # noqa: BLE001 -- an instrument never breaks a flip
            self._set_all({"missing": f"probe_error {type(e).__name__}"})

    def resolve_started_after(self, floor_ns: int) -> None:
        """Y8P-PPFWD-ARM-RACE (NF y8p 03.10. 08:52:05 / 08:53:41 / 08:57:00): the probe is armed
        at ``flip_done``, but P's first forward can start BEFORE the front logs ``done`` (the wake
        RPC returns, P admits at once; 0-70 ms ahead of the front). That forward is then already
        inside the baseline (``forward_ct`` rose before the arm), the probe waited for the NEXT rise
        and reported the SECOND chunk's start -- a false Nachlauf of one whole chunk (2.5-2.9 s) in
        3 of 7 flips, i.e. D>P total 5.2 s where the flip was 2.2 s. P sleeps for the whole D phase,
        so a forward whose ``t_start_ns`` lies at or after ``floor_ns`` (the flip's begin) is the
        woken group's first forward. Read from the baseline reading, per stage; never raises."""
        try:
            base = getattr(self, "base_reading", None) or {}
            for s in STAGES:
                if self.results[s] is not None:
                    continue
                best = None
                for pid in self.stages[s]:
                    r = base.get(pid)
                    if r is None:
                        continue
                    ct, ts, _td = r
                    if ct > 0 and ts >= int(floor_ns) and (best is None or ts < best[0]):
                        best = (int(ts), int(pid), int(ct))
                if best is not None:
                    self._set(s, {"ts": best[0] / 1e9, "pid": best[1], "ct": best[2],
                                 "armed_in_forward": True})
        except Exception:  # noqa: BLE001 -- an instrument never breaks a flip
            pass

    @property
    def result(self) -> Optional[dict]:
        """The D->P end: the FIRST stage's reading."""
        return self.results["first"]

    def done(self) -> bool:
        return all(self.results[s] is not None for s in STAGES)

    def _set(self, stage: str, res: dict) -> bool:
        with self._lock:
            if self.results[stage] is not None:
                return False
            if "pid" in res:
                res = dict(res, pp_rank=self.stages[stage].get(res["pid"]))
            self.results[stage] = res
            cb = self.on_result
        if cb is not None:
            try:
                cb(stage, res)
            except Exception:  # noqa: BLE001
                pass
        return True

    def _set_all(self, res: dict) -> None:
        for s in STAGES:
            self._set(s, dict(res))

    def poll(self) -> Dict[str, Optional[dict]]:
        """One reading now; the results so far."""
        if not self.done() and self._rescan is not None:
            self._poll_rescan()
        elif not self.done():
            cur: Dict[int, Tuple[int, int, int]] = {}
            for pid, path in self.paths.items():
                r = _read_one(path)
                if r is not None and int(r[3]) == int(pid):
                    cur[pid] = (int(r[0]), int(r[1]), int(r[2]))
            for s in STAGES:
                if self.results[s] is None:
                    res = first_rise(self.baseline[s], cur)
                    if res is not None:
                        self._set(s, res)
        return dict(self.results)

    def _poll_rescan(self) -> None:
        """PDFLIP-E3: the empty-baseline reading -- the group's files now; each
        new rank of pipeline stage 0 joins the first stage at forward_ct 0."""
        directory, group, sid, session_of, pp_of = self._rescan
        try:
            cur = read_group(directory, group, sid, session_of)
        except Exception:  # noqa: BLE001
            return
        first = dict(self.stages["first"])
        other = self.__dict__.setdefault("_rescan_other", set())   # ranks of a later stage
        for pid in cur:
            pid = int(pid)
            if pid in first or pid in other:
                continue
            r = pp_of(pid)
            if r == 0:
                first[pid] = 0
            elif r is not None:   # unknown (None): read again on the next poll
                other.add(pid)
        if len(first) != len(self.stages["first"]):
            # new dicts, never mutated in place (poll() also runs on the front's loop)
            self.stages = dict(self.stages, first=first)
            self.baseline = dict(self.baseline, first={pid: 0 for pid in first})
        if self.results["first"] is None and first:
            res = first_rise(self.baseline["first"], cur)
            if res is not None:
                self._set("first", res)

    def start(self, on_result: Optional[Callable[[str, dict], None]] = None) -> None:
        import threading

        with self._lock:
            self.on_result = on_result
            have = {s: r for s, r in self.results.items() if r is not None}
        if on_result is not None:
            for s in STAGES:  # already resolved (missing at once): report here, in order
                if s in have:
                    try:
                        on_result(s, have[s])
                    except Exception:  # noqa: BLE001
                        pass
        if self.done():
            return
        self._thread = threading.Thread(target=self._run, name="weg2-pp-forward", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        t_end = time.monotonic() + self.timeout_s
        while not self._stop.is_set():
            self.poll()
            if self.done():
                return
            if time.monotonic() >= t_end:
                self._set_all({"missing": f"timeout {self.timeout_s:.0f}s"})
                return
            self._stop.wait(self.poll_s)

    def stop(self, reason: str) -> None:
        """No more polling; a stage without a reading resolves ``missing``."""
        self._stop.set()
        self._set_all({"missing": reason})
