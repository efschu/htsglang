"""DUAL-TP3PP3 unified KV per card (U1): the ledger two PROCESSES share.

User order 30.09. ~07:10Z (verbatim): "der unified kv soll doch geshared
werden. die prozesse müssen sich gegenseitig abstimmen wer grad wie viel
bekommt. das anfangs beim boot festzulegen ist falsch".

In the dual layout a card carries one P stage process and one D rank process.
The boot fixes weights, contexts, mamba, graph pools and activation transients;
the REST of the card is ONE KV budget ``K_c`` in bytes, with no split. Each
process backs its KV pool physically in chunks on a VA-stable arena
(``mem_cache/kv_vmm_backing.py``: commit_span / decommit_span; addresses stay,
captured graphs keep replaying) and asks THIS ledger before every commit.

Invariant I1 (checked under the lock on every commit):
    committed[P] + committed[D] <= budget
No quota and no reserve: a request is granted from what is free. D has
priority: when D is short, it raises PRESSURE on P, which PAUSES its request at
the next chunk boundary (finished chunks are in L2 already), releases its whole
context and later resumes by reading them back; when P is short, it waits.
``arbitrate`` is the policy as a pure function; the ledger is the shared state
it runs on.

Shared state: one small record per card in /dev/shm, updated under
``fcntl.flock`` (the lease/flock pattern of ``registry/ledger.py``, without its
JSON file I/O -- a chunk grant must cost microseconds). A process that died
holds no physical memory any more (the driver freed it), so its committed
bytes are reaped when its pid is gone.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import mmap
import os
import struct
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple

GROUPS = ("P", "D")
_MAGIC = 0x4B564C47  # "KVLG"
_VERSION = 2
# magic, version, budget, epoch, then per group: pid, committed, demand, pressure, contribution
_FMT = "<IIqq" + "qqqqq" * len(GROUPS)
_SIZE = struct.calcsize(_FMT)


def _gi(group: str) -> int:
    try:
        return GROUPS.index(group)
    except ValueError:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}") from None


def card_key(card_uuid: str) -> str:
    """One spelling per card. NVML (the launcher, the front) says
    "GPU-31d7ef41-...", torch's device uuid (the ranks) says "31d7ef41-...".
    Metal dual15: the front read a ledger nobody wrote, so P never saw
    pressure and never paused."""
    u = str(card_uuid or "").strip().lower()
    for pre in ("gpu-", "mig-"):
        if u.startswith(pre):
            u = u[len(pre):]
    return u


def ledger_path(tag: str, card_uuid: str, root: str = "/dev/shm") -> str:
    h = hashlib.sha1(f"{tag}|{card_key(card_uuid)}".encode()).hexdigest()[:12]
    return os.path.join(root, f"wkv-{h}")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclasses.dataclass
class LedgerState:
    budget: int
    epoch: int
    pid: Dict[str, int]
    committed: Dict[str, int]
    demand: Dict[str, int]
    pressure: Dict[str, int]
    contrib: Dict[str, int] = dataclasses.field(default_factory=lambda: {g: 0 for g in GROUPS})

    @property
    def free(self) -> int:
        return self.budget - sum(self.committed.values())


def arbitrate(need: int, free: int, other_evictable: int, *, requester: str = "D",
              other_committed: int = 0) -> Tuple[int, int]:
    """The policy (user order 30.09. 07:25Z, verbatim: "wird vram kv knapp,
    pausiert P und gibt den context frei und das erarbeitete in den L2 zur
    späteren weiterverwendung wenn vram kv wieder frei wird"). ``need``: bytes
    the requester must commit now. Returns ``(grant, pressure)``; ``grant`` <=
    free is taken at once. No quota, no reserve.

    * D asks: D has priority. The shortfall becomes pressure on P up to ALL of
      P's committed KV -- P answers by PAUSING at its next chunk boundary (its
      finished chunks are already in L2 by the per-chunk write-through), then
      releasing its whole context. ``other_evictable`` is not a cap here.
    * P asks: P never presses D. A shortfall waits (backpressure) until D's
      demand drops; the paused request then resumes by READING its chunks
      from L2 ("Lesen statt Rechnen")."""
    need, free = max(0, int(need)), max(0, int(free))
    grant = min(need, free)
    if requester == "D":
        pressure = min(need - grant, max(0, int(other_committed)))
    else:
        pressure = 0
    return grant, pressure


class CardKvLedger:
    """One card's shared KV budget between the P and the D process."""

    def __init__(self, path: str, group: str, *, pid_alive=_pid_alive):
        self.path = path
        self.group = group
        self._gi = _gi(group)
        self._pid_alive = pid_alive
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.fstat(fd).st_size < _SIZE:
                os.ftruncate(fd, _SIZE)
            self._mm = mmap.mmap(fd, _SIZE)
        finally:
            os.close(fd)
        self._lock_fd = os.open(path + ".lock", os.O_RDWR | os.O_CREAT, 0o600)

    # -- record ---------------------------------------------------------------
    def _read(self) -> LedgerState:
        vals = struct.unpack_from(_FMT, self._mm, 0)
        magic, ver, budget, epoch = vals[:4]
        if magic != _MAGIC:
            budget, epoch = 0, 0
            rest = [0] * (5 * len(GROUPS))
        else:
            if ver != _VERSION:
                raise RuntimeError(f"card KV ledger {self.path}: version {ver}, expected {_VERSION}")
            rest = list(vals[4:])
        st = LedgerState(budget, epoch, {}, {}, {}, {}, {})
        for i, g in enumerate(GROUPS):
            (st.pid[g], st.committed[g], st.demand[g], st.pressure[g],
             st.contrib[g]) = rest[5 * i:5 * i + 5]
        return st

    def _write(self, st: LedgerState) -> None:
        vals = [_MAGIC, _VERSION, int(st.budget), int(st.epoch)]
        for g in GROUPS:
            vals += [int(st.pid[g]), int(st.committed[g]), int(st.demand[g]), int(st.pressure[g]),
                     int(st.contrib[g])]
        struct.pack_into(_FMT, self._mm, 0, *vals)

    @contextmanager
    def _locked(self) -> Iterator[LedgerState]:
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            st = self._read()
            self._reap(st)
            yield st
            st.epoch += 1
            self._write(st)
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _reap(self, st: LedgerState) -> None:
        for g in GROUPS:
            if st.pid[g] and not self._pid_alive(st.pid[g]):
                # a dead process holds no device memory: the driver freed it
                st.pid[g] = 0
                st.committed[g] = 0
                st.demand[g] = 0
                st.pressure[g] = 0
                # its KV bytes stay on the card for the survivor? No: a dead
                # process's pool is gone with it; its contribution leaves too.
                st.budget -= st.contrib[g]
                st.contrib[g] = 0

    # -- API ------------------------------------------------------------------
    def join(self, budget: int, pid: Optional[int] = None) -> LedgerState:
        """Register this process; the first one sets the card budget, a later
        one must state the SAME budget (both derive it from the same boot
        numbers; disagreeing means two different card plans -- refused)."""
        pid = os.getpid() if pid is None else int(pid)
        with self._locked() as st:
            if st.budget and int(budget) != st.budget and any(st.pid[g] for g in GROUPS if g != self.group):
                raise RuntimeError(
                    f"card KV ledger {self.path}: budget {int(budget)} from {self.group} disagrees with "
                    f"{st.budget} already set by the other process -- two different card plans")
            st.budget = int(budget)
            if st.pid[self.group] and st.pid[self.group] != pid and self._pid_alive(st.pid[self.group]):
                raise RuntimeError(f"card KV ledger {self.path}: group {self.group} already held by pid "
                                   f"{st.pid[self.group]}")
            st.pid[self.group] = pid
            return dataclasses.replace(st)

    def contribute(self, kv_bytes: int, committed: int = 0, pid: Optional[int] = None) -> LedgerState:
        """Put this process's boot KV into the card's ONE pool: the budget is
        the SUM of both processes' KV sizings (each sized its pool exactly as
        before -- nothing new to plan), none of it is anyone's. ``committed``:
        what this process keeps mapped right now (D: its boot stage; P: 0).
        Idempotent per process (a second call replaces its own share)."""
        pid = os.getpid() if pid is None else int(pid)
        with self._locked() as st:
            if st.pid[self.group] and st.pid[self.group] != pid and self._pid_alive(st.pid[self.group]):
                raise RuntimeError(f"card KV ledger {self.path}: group {self.group} already held by pid "
                                   f"{st.pid[self.group]}")
            st.pid[self.group] = pid
            # metal dual14 (bdfefe): D unmapped its boot pool (2415919104 B on
            # the 5090) in the second P began sizing, and P's KV sizing counted
            # it as free (4475322368 B vs 2122317824 in dual13) -- the budget
            # held the same bytes twice and D's grow hit cuMemCreate OOM. What
            # the other group contributed but has RELEASED was free memory to
            # this sizing: it is not contributed a second time.
            other = GROUPS[1 - self._gi]
            released = max(0, st.contrib[other] - st.committed[other]) if st.pid[other] else 0
            counted = max(0, int(kv_bytes) - released)
            st.budget += counted - st.contrib[self.group]
            st.contrib[self.group] = counted
            st.committed[self.group] = max(0, int(committed))
            if sum(st.committed.values()) > st.budget:
                raise RuntimeError(
                    f"card KV ledger {self.path}: committed {dict(st.committed)} exceed the pool "
                    f"{st.budget} after {self.group}'s contribution -- I1 cannot hold")
            return dataclasses.replace(st, pid=dict(st.pid), committed=dict(st.committed),
                                       demand=dict(st.demand), pressure=dict(st.pressure),
                                       contrib=dict(st.contrib))

    def state(self) -> LedgerState:
        with self._locked() as st:
            return dataclasses.replace(st, pid=dict(st.pid), committed=dict(st.committed),
                                       demand=dict(st.demand), pressure=dict(st.pressure),
                                       contrib=dict(st.contrib))

    def request(self, need: int, other_evictable: int = 0) -> Tuple[int, int]:
        """Commit up to ``need`` bytes for this group NOW (I1 under the lock).
        Returns ``(granted, pressure_raised)``. The caller backs exactly
        ``granted`` bytes; if ``granted < need`` the rest waits."""
        other = GROUPS[1 - self._gi]
        with self._locked() as st:
            grant, pressure = arbitrate(need, st.free, other_evictable, requester=self.group,
                                        other_committed=st.committed[other])
            st.committed[self.group] += grant
            st.demand[self.group] = max(0, int(need) - grant)
            # the CURRENT shortfall, not a high-water mark (metal dual20: the max kept
            # 855638016 on a 3080 for 4 min after P held nothing and D's seats were gone;
            # no P pass started, a user's LONG request starved). 0 when granted in full.
            st.pressure[other] = int(pressure)
            assert sum(st.committed.values()) <= st.budget, "I1 violated"
            return grant, pressure

    def lend(self, nbytes: int) -> int:
        """D PRIORITY stage 2: this group's process freed ``nbytes`` of NON-KV
        device memory (P's weights parked in host RAM) -- they join the card's
        pool for as long as the process sleeps. Unlike ``contribute`` (a boot
        sizing, net of what the other group released) this is a plain loan."""
        n = max(0, int(nbytes))
        with self._locked() as st:
            st.budget += n
            st.contrib[self.group] += n
            return n

    def reclaim(self, nbytes: int) -> bool:
        """The loan back, before the process maps its weights again: only when
        the pool's free bytes cover it (nobody committed into the loan) --
        False otherwise, nothing changed (the caller waits or stops named)."""
        n = max(0, int(nbytes))
        with self._locked() as st:
            if st.free < n:
                return False
            st.budget -= n
            st.contrib[self.group] = max(0, st.contrib[self.group] - n)
            assert sum(st.committed.values()) <= st.budget, "I1 violated"
            return True

    def release(self, nbytes: int) -> int:
        """Return ``nbytes`` this group decommitted; clears pressure it answered."""
        with self._locked() as st:
            n = min(max(0, int(nbytes)), st.committed[self.group])
            st.committed[self.group] -= n
            st.pressure[self.group] = max(0, st.pressure[self.group] - n)
            return n

    def reconcile(self, phys_free: int, cap: Optional[int] = None) -> int:
        """The card said no (cuMemCreate OOM) although the ledger had room: the
        budget promised bytes the card does not have. Lower it until the
        ledger's free equals the physical free bytes; returns the correction.
        Taken from this group's contribution first (a leaving group takes its
        own share out of the budget)."""
        with self._locked() as st:
            over = int(st.free) - max(0, int(phys_free))
            if cap is not None:
                over = min(over, int(cap))
            if over <= 0:
                return 0
            st.budget -= over
            mine = min(over, st.contrib[self.group])
            st.contrib[self.group] -= mine
            other = GROUPS[1 - self._gi]
            st.contrib[other] = max(0, st.contrib[other] - (over - mine))
            return over

    def clear_pressure(self) -> None:
        """This group no longer needs what it asked the other one for (its demand
        fits what it maps): the pressure it put on the other group goes."""
        other = GROUPS[1 - self._gi]
        with self._locked() as st:
            st.pressure[other] = 0
            st.demand[self.group] = 0

    def pressure_on_me(self) -> int:
        """Bytes the other process asked this one to release from its cache."""
        with self._locked() as st:
            return int(st.pressure[self.group])

    def leave(self) -> None:
        with self._locked() as st:
            st.pid[self.group] = 0
            st.committed[self.group] = 0
            st.demand[self.group] = 0
            st.pressure[self.group] = 0
            st.budget -= st.contrib[self.group]
            st.contrib[self.group] = 0

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            os.close(self._lock_fd)


def peek(path: str) -> Optional[LedgerState]:
    """Read one card's record without joining or writing (the front's view).
    None when the ledger does not exist yet (no process joined)."""
    if not os.path.exists(path):
        return None
    led = CardKvLedger.__new__(CardKvLedger)
    led.path, led.group, led._gi, led._pid_alive = path, "P", 0, _pid_alive
    fd = os.open(path, os.O_RDONLY)
    try:
        if os.fstat(fd).st_size < _SIZE:
            return None
        mm = mmap.mmap(fd, _SIZE, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    lock_fd = os.open(path + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        led._mm = mm
        st = led._read()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        mm.close()
    return st if st.budget else None


def p_resume_ready(paths):
    """The front may send a PAUSED P request back only when EVERY card shows:
    no pressure on P, nothing committed by P (every P stage released -- the
    followers too), and no D demand left (D grew or stopped asking). Metal
    dual22: the resume went out while PP1/PP2 still held the old instance, and
    straight into a second pause (D still growing). Returns (ready, per-card
    [(pressure, p_committed, d_demand)]); a missing ledger is not ready."""
    out = []
    ready = bool(paths)
    for pth in paths or ():
        st = peek(pth)
        if st is None:
            out.append(None)
            ready = False
            continue
        row = (int(st.pressure["P"]), int(st.committed["P"]), int(st.demand["D"]))
        out.append(row)
        ready = ready and row == (0, 0, 0)
    return ready, out


def p_pressure(paths) -> int:
    """The largest pressure any card puts on P (D is short there) -- the
    front's pause trigger."""
    worst = 0
    for pth in paths or ():
        st = peek(pth)
        if st is not None:
            worst = max(worst, int(st.pressure["P"]))
    return worst

