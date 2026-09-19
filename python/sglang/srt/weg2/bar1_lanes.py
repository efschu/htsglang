"""BAR1 lanes for the flip legs (18.09.2026, user order "ueber Fenster, so wie
wir es mit allreduce/allgather machen" -- and "dann bau die bar1 lanes jetzt").

A cross lane ``p<k>`` (``CROSS_PAIRS[k]`` = (src card, dst card)) carries one
tag's slices from the sleeping rank on the source card to the waking rank on
the destination card. Until now both ends bounced through a pinned host
buffer (2.9-4.1 GB/s per 5090 lane; measured xsn364: the two 3080->5090
lanes spent 8.6 s in ``copy_sync`` for 0.86 GB each, the host DRAM ceiling
~38 GB/s shared by six streams). Here the DESTINATION rank owns a VMM window
on its own card, exports it as a dma-buf and serves the fd on an AF_UNIX
socket; the SOURCE rank attaches it (``/dev/dmabuf_holder``), maps the
window's BAR1 pages (``resource1_wc``) and writes straight into it with
``cudaMemcpyAsync`` -- posted DMA, measured 13-14 GB/s per gen4-x8 link and
6.6 GB/s on the x4 card, no host byte (bar1_lane_probe.py, 18.09.).

The window is a RING of ``ring`` slots of ``slot_bytes``. The pieces of a
tag are cut by :func:`weight_exchange_transport.batch_descs` at
``slot_bytes`` -- deterministic from the plan alone, identical on both
sides, FLAT pieces split by bytes and STRIDED2D by rows -- so batch ``g``
goes to slot ``g % ring``. The credit per batch is a pair of flag files
(``full.<seq>.<g>`` by the depositor, carrying the batch's record;
``free.<seq>.<g>`` by the collector), each consumed once and unlinked:
cross-process, no launcher semaphores, no host memory in the data path.

Sizing: a 3080 exposes 256 MiB of BAR1 with ~168 MiB taken by the group
windows (P 24 + PP_0 96, D world 16 + tp 32), so each of its receiving
processes gets 2 lanes x 2 x 8 MiB; the 5090's 32-GiB BAR takes 2 x 64 MiB
per lane. Both ends of a lane exist TWICE (P->D flip: PP<src> deposits into
TP<dst>; D->P flip: TP<src> into PP<dst>), so the window, socket and flags
are keyed by the RECEIVING GROUP as well as the lane.

Per tag the DEPOSITOR decides the path and writes ``mode.<seq>`` ("bar1" or
"host" with the reason); the collector waits for that file before it waits
on anything else, so the two sides never sit on different handshakes. The
user's rule (memory bar1-lanes-systemram-nur-wenn-noetig): a lane falls back
to the host path only with a NAMED reason, printed in its lane line.
"""
from __future__ import annotations

import ctypes
import json
import logging
import mmap
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Mapping as _Map, Optional

logger = logging.getLogger(__name__)

ENV_ON = "SGLANG_WEG2_BAR1_LANES"
ENV_RING = "SGLANG_WEG2_BAR1_RING_SLOTS"
ENV_SMALL_SLOT_MIB = "SGLANG_WEG2_BAR1_SMALL_SLOT_MIB"
ENV_BIG_SLOT_MIB = "SGLANG_WEG2_BAR1_BIG_SLOT_MIB"
ENV_CONNECT_S = "SGLANG_WEG2_BAR1_CONNECT_S"
ENV_SMALL_BAR_GROUPS = "SGLANG_WEG2_BAR1_SMALL_BAR_GROUPS"
#: #22 (xsn323): a credit wait that waits longer than this while the sleeper
#: co-located with it is itself blocked on a deposit this group's waker will
#: only collect AFTER its own credit is a CYCLE -- named after `grace` seconds
#: (W109) instead of after the 120 s credit budget (W35).
ENV_CYCLE_GRACE_S = "SGLANG_WEG2_BAR1_CYCLE_GRACE_S"
SLOW_WAIT_S = 0.5     # a credit wait longer than this posts a `blocked` flag
BIG_BAR_MIN = 4 << 30      # a BAR1 at least this large holds the big slot ring
MODE_BAR1 = "bar1"
MODE_HOST = "host"


# -- pure decisions (desk-testable) ------------------------------------------

def lanes_on(env: Optional[_Map[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV_ON, "1")).strip().lower() not in ("0", "false", "no", "off")


def ring_slots(env: Optional[_Map[str, str]] = None) -> int:
    env = os.environ if env is None else env
    try:
        r = int(env.get(ENV_RING, "4"))
    except ValueError:
        r = 4
    return max(3, min(16, r))       # 3 = the smallest ring the one-behind sync can pipeline


def slot_bytes_for(bar1_size: int, env: Optional[_Map[str, str]] = None) -> int:
    """The ring slot of a RECEIVER card: 32 MiB on a big BAR, 4 MiB on the
    256-MiB BAR of a 3080 -- measured xsn365: the group windows (P 24+96,
    D 16+32 MiB) leave room for exactly ONE 16-MiB hold per 3080, every
    further DMABUF_HOLDER_IOC_HOLD is ENOMEM. Ring 4 x 4 MiB = that one
    window, see :func:`small_bar_serves` for who gets it."""
    env = os.environ if env is None else env
    try:
        small = int(env.get(ENV_SMALL_SLOT_MIB, "4"))
        big = int(env.get(ENV_BIG_SLOT_MIB, "32"))
    except ValueError:
        small, big = 4, 32
    return (max(1, big) if int(bar1_size) >= BIG_BAR_MIN else max(1, small)) << 20


def lane_is_cross(lane_key: str) -> bool:
    return str(lane_key or "").startswith("p")


def lane_role(lane_key: str, rank: int, cross_pairs) -> Optional[str]:
    """'src' (depositor), 'dst' (collector) or None for this rank on lane p<k>."""
    if not lane_is_cross(lane_key):
        return None
    try:
        k = int(str(lane_key)[1:])
        s, d = cross_pairs[k]
    except (ValueError, IndexError, TypeError):
        return None
    if int(rank) == int(s):
        return "src"
    if int(rank) == int(d):
        return "dst"
    return None


def borrow_regions(windows, n_lanes: int):
    """The borrowable regions of one receiver process, OVERLAP-FREE and
    deterministic: the payload regions of its tp/dcp/pp group windows (never
    world), largest first; while there are fewer regions than lanes the
    largest one is halved (MiB-aligned). xsn366 (18.09.) had both D lanes of
    a 3080 on dcp:0 at offset 0 because tp:0 was not in the registry -- two
    depositors on the same slots."""
    regs = []
    for name, size in dict(windows).items():
        prefix = str(name).split(":")[0]
        if prefix in ("tp", "dcp", "pp") and int(size) >= (4 << 20):
            regs.append((str(name), 0, int(size)))
    regs.sort(key=lambda r: (-r[2], r[0]))
    while regs and len(regs) < int(n_lanes):
        name, off, size = regs.pop(0)
        half = (size // 2) & ~((1 << 20) - 1)
        if half < (4 << 20):
            regs.insert(0, (name, off, size))
            break
        regs.extend([(name, off, half), (name, off + half, half)])
        regs.sort(key=lambda r: (-r[2], r[0], r[1]))
    return regs


def lane_order(dst_lanes, cross_pairs, big_cards):
    """The receiver's lanes, the ones fed by a big-BAR card first (PP0 on
    the 5090 carries the most bytes and is the flip's critical chain)."""
    big = {int(c) for c in big_cards}

    def key(lk):
        try:
            src = int(cross_pairs[int(str(lk)[1:])][0])
        except (ValueError, IndexError, TypeError):
            src = -1
        return (0 if src in big else 1, str(lk))
    return sorted(dst_lanes, key=key)


def borrow_plan(group: str, lane_key: str, cross_pairs, big_cards, windows,
                ring: int = 4, dst_lanes=None):
    """On a SMALL-BAR receiver card the lane BORROWS the payload region of one
    of this process's own barlink group windows instead of allocating a new
    one (user 18.09.: "warum koennen wir das bar1 fenster in dieser phase
    nicht von den Gruppenfenstern befreien -- waehrend des flips passiert ja
    kein traffic von P oder D"). Measured: a 3080 lends 240 of its 256 MiB to
    holds and the group windows already take 208 of them; a second hold of
    the same dma-buf by another card costs nothing (bar1_hold_budget.py).
    The receiver's group is asleep or still waking while its lanes are
    collected, and a lane's writes into its window end before the collect
    returns, so no collective ever sees them.

    ``windows`` = {group name: payload bytes} of the live transports of this
    process; ``dst_lanes`` = every lane this process receives on. Returns
    (window name, offset, size, slot_bytes) or a reason. The regions come
    from :func:`borrow_regions` (overlap-free), handed out in
    :func:`lane_order`."""
    if not lane_is_cross(lane_key):
        return "small BAR1: not a cross lane"
    lanes = lane_order(list(dst_lanes or [lane_key]), cross_pairs, big_cards)
    if lane_key not in lanes:
        lanes = lane_order(lanes + [lane_key], cross_pairs, big_cards)
    regs = borrow_regions(windows, len(lanes))
    i = lanes.index(lane_key)
    if i >= len(regs):
        return (f"small BAR1: no group window left to borrow for lane {i + 1}/{len(lanes)} "
                f"(have {sorted(dict(windows))})")
    name, off, size = regs[i]
    return _slotted(name, off, size, ring)


def _slotted(name: str, offset: int, size: int, ring: int):
    slot = (int(size) // max(1, int(ring))) & ~((1 << 20) - 1)
    if slot < (2 << 20):
        return f"small BAR1: window {name} too small for a ring of {ring} ({size >> 20} MiB)"
    return (name, int(offset), slot * int(ring), slot)


def other_group(group: str) -> str:
    return "D" if str(group).upper() == "P" else "P"


def lane_dir(boot_nonce: str, lane_key: str, dst_group: str, root: str = "/dev/shm") -> str:
    """Socket and flags of ONE direction of a lane: the receiving group names it."""
    return os.path.join(root, f"weg2-bar1-{boot_nonce}", f"{lane_key}.{str(dst_group).upper()}")


def socket_path(boot_nonce: str, lane_key: str, dst_group: str, root: str = "/dev/shm") -> str:
    return os.path.join(lane_dir(boot_nonce, lane_key, dst_group, root), "window.sock")


def flag_dir(boot_nonce: str, lane_key: str, dst_group: str, root: str = "/dev/shm") -> str:
    return os.path.join(lane_dir(boot_nonce, lane_key, dst_group, root), "flags")


def ring_slot(g: int, ring: int) -> int:
    return int(g) % max(1, int(ring))


# -- flags (cross-process credits) -------------------------------------------

def _flag_path(d: str, kind: str, seq, g: int) -> str:
    """``seq`` names the tag instance: "<flip index>-<tag>" from the updater
    (deterministic on both sides, safe under concurrent collects), or the
    per-lane counter of the desk tests."""
    return os.path.join(d, f"{kind}.{seq}.{int(g)}")


def post_flag(d: str, kind: str, seq, g: int, payload: Optional[dict] = None) -> None:
    """Atomic: written to a tmp name, renamed into place (the reader unlinks it)."""
    os.makedirs(d, exist_ok=True)
    p = _flag_path(d, kind, seq, g)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        if payload is not None:
            json.dump(payload, fh)
    os.replace(tmp, p)


def take_flag(d: str, kind: str, seq, g: int, timeout_s: float,
              liveness=None, on_slow=None) -> Optional[dict]:
    """The flag's payload ({} when it carried none) once it exists -- consumed
    (unlinked); None on timeout or when ``liveness()`` says the peer is gone.
    Hot poll: a batch is 0.6-5 ms of DMA, a 200-us sleep would be 4-30 %."""
    p = _flag_path(d, kind, seq, g)
    t0 = time.perf_counter()
    n = 0
    while True:
        try:
            with open(p) as fh:
                raw = fh.read()
            os.unlink(p)
            return json.loads(raw) if raw.strip() else {}
        except FileNotFoundError:
            pass
        n += 1
        if time.perf_counter() - t0 > timeout_s:
            return None
        if on_slow is not None and time.perf_counter() - t0 > SLOW_WAIT_S:
            on_slow()
            on_slow = None
        if liveness is not None and (n & 1023) == 0 and not liveness():
            return None
        time.sleep(0.00002)


# -- credits: file flags (desk, fallback) or the lane's socket (metal) ----------

_FRAME = struct.Struct("<cI")


class FileCredits:
    def __init__(self, d: str, seq, liveness=None):
        self.d, self.seq, self.liveness = d, seq, liveness
        self.on_slow = None

    def send(self, kind: str, g: int = 0, payload=None) -> None:
        post_flag(self.d, kind, self.seq, g, payload)

    def recv(self, kind: str, g: int, timeout_s: float):
        return take_flag(self.d, kind, self.seq, g, timeout_s, liveness=self.liveness,
                         on_slow=self.on_slow)


class SocketCredits:
    """One 5-byte frame per credit on the lane's open connection; a plan
    frame carries a length-prefixed JSON. Both sides read in order, so the
    kinds are checked, never searched."""
    KINDS = {"plan": b"P", "full": b"F", "free": b"R", "done": b"D"}

    def __init__(self, sock, liveness=None):
        self.sock, self.liveness = sock, liveness
        self.on_slow = None

    def send(self, kind: str, g: int = 0, payload=None) -> None:
        body = b"" if payload is None else json.dumps(payload).encode()
        self.sock.sendall(_FRAME.pack(self.KINDS[kind], int(g)) + struct.pack("<I", len(body)) + body)

    def _read(self, n: int, timeout_s: float) -> Optional[bytes]:
        buf = b""
        deadline = time.monotonic() + float(timeout_s)
        k = 0
        while len(buf) < n:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            self.sock.settimeout(min(left, 0.5))
            try:
                chunk = self.sock.recv(n - len(buf))
            except socket.timeout:
                k += 1
                if k == 1 and self.on_slow is not None:
                    self.on_slow()
                if self.liveness is not None and not self.liveness():
                    return None
                continue
            if not chunk:
                return None          # peer closed
            buf += chunk
        return buf

    def recv(self, kind: str, g: int, timeout_s: float):
        head = self._read(_FRAME.size + 4, timeout_s)
        if head is None:
            return None
        k, gg = _FRAME.unpack_from(head, 0)
        (n,) = struct.unpack_from("<I", head, _FRAME.size)
        body = self._read(n, timeout_s) if n else b""
        if body is None:
            return None
        if k != self.KINDS[kind] or int(gg) != int(g):
            raise RuntimeError(f"bar1 credit out of order: got {k!r} {gg}, expected {kind} {g}")
        return json.loads(body.decode()) if body else {}


# -- #22: the credit-wait cycle, as numbers ------------------------------------

def tag_of_seq(seq) -> str:
    """"<flip>-<tag>" -> "<tag>" (a desk counter has no tag: itself)."""
    s = str(seq)
    return s.split("-", 1)[1] if "-" in s else s


def cycle_grace_s(env: Optional[_Map[str, str]] = None) -> float:
    env = os.environ if env is None else env
    try:
        return max(0.5, float(env.get(ENV_CYCLE_GRACE_S, "3")))
    except ValueError:
        return 3.0


def credit_cycle(me: int, waits: dict, blocked: dict, grace_s: float, now: float):
    """The chain of blocked edges that closes at waker ``me``, or None.

    ``waits``: waker rank -> {"tag", "since", "submitted": [tags]} (this
    group's ranks inside a credit wait). ``blocked``: (src, dst) -> {"seq",
    "since"} (the OTHER group's rank ``src`` -- co-located with waker
    ``src`` -- blocked depositing ``seq`` to waker ``dst``). Waker r's
    credit is funded by the pauses of sleeper r; sleeper r pauses a tag
    only after depositing it; the deposit drains only when waker dst
    collects it, which waker dst does only after ITS credit -- unless it
    already submitted that tag's collect (then the deposit drains anyway
    and the edge is not blocking). Every edge older than ``grace_s``.
    """
    def edges(r):
        out = []
        for (s, d), b in blocked.items():
            if int(s) != int(r) or now - float(b.get("since", now)) < grace_s:
                continue
            w = waits.get(int(d))
            if w is None or now - float(w.get("since", now)) < grace_s:
                continue
            if tag_of_seq(b.get("seq", "")) in set(w.get("submitted") or ()):
                continue
            out.append((int(s), int(d), tag_of_seq(b.get("seq", ""))))
        return out

    if int(me) not in waits or now - float(waits[int(me)].get("since", now)) < grace_s:
        return None
    stack = [(int(me), [])]
    seen = set()
    while stack:
        cur, chain = stack.pop()
        for e in edges(cur):
            nxt = e[1]
            if nxt == int(me):
                return chain + [e]
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append((nxt, chain + [e]))
    return None


# -- the window (receiver) and the peer mapping (depositor) -------------------

@dataclass
class RecvWindow:
    lane_key: str
    dptr: int
    handle: int
    size: int
    slot_bytes: int
    ring: int
    dmabuf_fd: int
    hold_fds: list
    bdf: str
    ordinal: int
    listener: object = None
    thread: object = None
    borrowed: str = ""     # the group window whose payload region this ring borrows
    conn: object = None    # the depositor's open connection = the credit channel


@dataclass
class PeerWindow:
    lane_key: str
    dev_ptr: int           # this card's device pointer to the peer's window
    size: int
    slot_bytes: int
    ring: int
    peer_bdf: str
    holder_handle: int
    mmap_obj: object
    reg_address: int
    sock: object = None    # the open connection to the receiver = the credit channel


class Bar1Lanes:
    """Per process: receiver windows for the lanes where this rank is the
    destination (served to the OTHER group), peer mappings for the lanes
    where it is the source (connected to the other group's windows)."""

    def __init__(self, boot_nonce: str, group: str, rank: int, device, cross_pairs,
                 log=None, root: str = "/dev/shm"):
        self.boot_nonce = str(boot_nonce)
        self.group = str(group).upper()
        self.rank = int(rank)
        self.device = device
        self.cross_pairs = list(cross_pairs)
        self.log = log or logger.info
        self.root = root
        self.recv: dict = {}
        self.peers: dict = {}
        self.refusals: dict = {}
        self.ready = False
        self._cuda = None
        self._holder = None
        self._bdf = None
        self._group_own: dict = {}
        self._dst_lanes: list = []
        self._windows_logged = False
        self.last_seq: dict = {}       # lane -> the seq this side ran last (the depositor's done-wait)
        self.turn: dict = {}           # lane -> {(flip, index), ...} pending tags, oldest runs first
        self.turn_cv = threading.Condition()

    # -- helpers ---------------------------------------------------------
    def _ordinal(self) -> int:
        d = self.device
        return int(d.index if hasattr(d, "index") and d.index is not None else d)

    def _cu(self):
        if self._cuda is None:
            from sglang.srt.distributed.device_communicators.barlink_bar1 import _Cuda
            self._cuda = _Cuda()
        return self._cuda

    def bdf(self) -> str:
        if self._bdf is None:
            from sglang.srt.distributed.device_communicators.barlink_matrix import bdf_of_card
            self._bdf = str(bdf_of_card(self.device))
        return self._bdf

    def group_windows(self) -> dict:
        """{group name: payload bytes} of this process's live barlink
        transports (abort_gate registry); remembers (dptr, handle, size)."""
        out = {}
        try:
            from sglang.srt.distributed.device_communicators import barlink_abort_gate as gate
            for t in gate.registered():
                own = getattr(t, "_own", None)
                name = str(getattr(t, "group", "") or "")
                if not own or not own[0] or not name:
                    continue
                geo = getattr(t, "_geo", None) or {}
                size = int(geo.get("region_bytes", own[2]) or own[2])
                out[name] = size
                self._group_own[name] = (int(own[0]), int(own[1]), int(own[2]))
        except Exception as exc:  # noqa: BLE001 -- no registry = nothing to borrow
            self.log(f"WEG2-BAR1 group_windows: registry unreadable: {exc!r}")
        return out

    def big_cards(self) -> list:
        """Card indices whose BAR1 is big (the 5090): rank n of either group
        runs on cards[n], so the ordinal IS the card index."""
        from sglang.srt.distributed.device_communicators.barlink_bar1 import bar1_window
        from sglang.srt.distributed.device_communicators.barlink_matrix import bdf_of_card
        out = []
        for c in sorted({int(x) for pair in self.cross_pairs for x in pair}):
            try:
                if int(bar1_window(str(bdf_of_card(c))).size) >= BIG_BAR_MIN:
                    out.append(c)
            except Exception as exc:  # noqa: BLE001 -- an unreadable card is not big
                self.log(f"WEG2-BAR1 big_cards: card {c} unreadable: {exc!r}")
        return out

    def role(self, lane_key: str) -> Optional[str]:
        return lane_role(lane_key, self.rank, self.cross_pairs)

    def flags(self, lane_key: str, role: str) -> str:
        dst_group = self.group if role == "dst" else other_group(self.group)
        return flag_dir(self.boot_nonce, lane_key, dst_group, self.root)

    # -- #22: credit-wait flags and the cycle reader --------------------
    def credit_dir(self) -> str:
        return os.path.join(self.root, f"weg2-bar1-{self.boot_nonce}", "credit")

    def post_credit_wait(self, tag: str, submitted=None) -> None:
        try:
            post_flag(self.credit_dir(), "wait", self.group, self.rank,
                      {"tag": str(tag), "since": time.time(),
                       "submitted": [str(t) for t in (submitted or ())]})
        except OSError:
            pass

    def clear_credit_wait(self) -> None:
        try:
            os.unlink(_flag_path(self.credit_dir(), "wait", self.group, self.rank))
        except OSError:
            pass

    def _read_json(self, p: str) -> Optional[dict]:
        try:
            with open(p) as fh:
                raw = fh.read()
            return json.loads(raw) if raw.strip() else {}
        except (OSError, ValueError):
            return None

    def read_credit_waits(self) -> dict:
        """waker rank -> payload, for THIS group's ranks in a credit wait."""
        out = {}
        d = self.credit_dir()
        for r in range(len({s for s, _ in self.cross_pairs} | {d_ for _, d_ in self.cross_pairs})):
            p = self._read_json(_flag_path(d, "wait", self.group, r))
            if p is not None:
                out[r] = p
        return out

    def read_blocked(self) -> dict:
        """(src, dst) -> payload for every deposit of the OTHER group into
        this group that is blocked on our collect (its `blocked` flag)."""
        out = {}
        for k, (s, d_) in enumerate(self.cross_pairs):
            fd = flag_dir(self.boot_nonce, f"p{k}", self.group, self.root)
            try:
                names = os.listdir(fd)
            except OSError:
                continue
            for n in names:
                if not n.startswith("blocked.") or n.endswith(".tmp"):
                    continue
                p = self._read_json(os.path.join(fd, n))
                if p is None:
                    continue
                parts = n.split(".")
                p.setdefault("seq", parts[1] if len(parts) >= 3 else "")
                prev = out.get((int(s), int(d_)))
                if prev is None or float(p.get("since", 0)) < float(prev.get("since", 0)):
                    out[(int(s), int(d_))] = p
        return out

    def credit_cycle(self, grace_s: Optional[float] = None):
        """The blocked chain closing at this rank, or None (see `credit_cycle`)."""
        g = cycle_grace_s() if grace_s is None else float(grace_s)
        return credit_cycle(self.rank, self.read_credit_waits(), self.read_blocked(), g, time.time())

    def channel(self, lane_key: str, role: str):
        """The lane's open AF_UNIX connection from this side, or None (file flags)."""
        if role == "dst":
            w = self.recv.get(lane_key)
            return getattr(w, "conn", None) if w is not None else None
        p = self.peers.get(lane_key)
        return getattr(p, "sock", None) if p is not None else None

    def register_turns(self, order_key) -> None:
        """Called by the wake loop's MAIN thread, in tag order, before a tag's
        collect is submitted: the tag is pending on every lane this process
        receives on. A lane runs its tags in the order they were registered;
        a tag that turns out not to use a lane releases it (release_turns)."""
        if order_key is None:
            return
        key = (int(order_key[0]), int(order_key[1]))
        with self.turn_cv:
            for lk in self.recv:
                self.turn.setdefault(lk, set()).add(key)
            self.turn_cv.notify_all()

    def release_turns(self, order_key, used=None) -> None:
        """Drop the tag from the lanes it does not use (``used`` given) or from
        every lane (the tag's collect is over, whatever happened)."""
        if order_key is None:
            return
        key = (int(order_key[0]), int(order_key[1]))
        with self.turn_cv:
            for lk, pend in self.turn.items():
                if used is None or lk not in set(used):
                    pend.discard(key)
            self.turn_cv.notify_all()

    def take_turn(self, lane_key: str, order_key, timeout_s: float) -> bool:
        """Wait until this tag is the OLDEST pending one on the lane: with two
        collects in flight one lane's tags must not interleave on its byte
        stream. A key that was never registered passes (the desk, a lane
        without registration)."""
        if order_key is None:
            return True
        key = (int(order_key[0]), int(order_key[1]))
        deadline = time.monotonic() + float(timeout_s)
        with self.turn_cv:
            while True:
                pend = self.turn.get(lane_key) or set()
                if key not in pend or min(pend) == key:
                    return True
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self.turn_cv.wait(min(left, 0.5))

    def leave_turn(self, lane_key: str, order_key) -> None:
        if order_key is None:
            return
        key = (int(order_key[0]), int(order_key[1]))
        with self.turn_cv:
            pend = self.turn.get(lane_key)
            if pend:
                pend.discard(key)
            self.turn_cv.notify_all()

    def window_for(self, lane_key: str, role: str):
        """(base device pointer, slot_bytes, ring) of this lane's window as seen
        from this side, or None when this side holds no BAR1 end of it."""
        if role == "dst":
            w = self.recv.get(lane_key)
            return (int(w.dptr), int(w.slot_bytes), int(w.ring)) if w is not None else None
        p = self.peers.get(lane_key)
        return (int(p.dev_ptr), int(p.slot_bytes), int(p.ring)) if p is not None else None

    # -- the per-tag agreement ------------------------------------------------
    def lane_mode(self, lane_key: str, role: str, seq, timeout_s: float = 120.0,
                  liveness=None) -> str:
        """The depositor DECIDES (its peer mapping exists or not) and writes
        ``mode.<seq>``; the collector WAITS for it. Both return MODE_BAR1 or
        MODE_HOST; a collector that sees no mode file within the budget runs
        the host path (what a pre-BAR1 depositor would have done)."""
        d = self.flags(lane_key, role)
        if role == "src":
            p = self.peers.get(lane_key)
            mode = MODE_BAR1 if p is not None else MODE_HOST
            why = "" if p is not None else self.refusals.get(lane_key, "no peer window")
            post_flag(d, "mode", seq, 0, {"mode": mode, "reason": why})
            return mode
        got = take_flag(d, "mode", seq, 0, timeout_s, liveness=liveness)
        if got is None:
            return MODE_HOST
        mode = str(got.get("mode", MODE_HOST))
        if mode == MODE_BAR1 and lane_key not in self.recv:
            # cannot happen (the depositor mapped THIS window) -- named anyway
            self.log(f"WEG2-BAR1 lane={lane_key} role=dst mode=bar1 but no window here -> host")
            return MODE_HOST
        return mode

    def _mark_no_window(self, lane_key: str, why: str) -> None:
        """Tell the would-be depositor at once that no window will be served
        (it would otherwise wait the whole connect budget on the socket)."""
        try:
            d = lane_dir(self.boot_nonce, lane_key, self.group, self.root)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "no-window"), "w") as fh:
                fh.write(str(why))
        except OSError as exc:
            self.log(f"WEG2-BAR1 lane={lane_key} no-window marker failed: {exc!r}")

    # -- receiver: window + serve --------------------------------------------
    def open_window(self, lane_key: str) -> Optional[RecvWindow]:
        from sglang.srt.distributed.device_communicators.barlink_bar1 import bar1_window
        try:
            cu = self._cu()
            bdf = self.bdf()
            win = bar1_window(bdf)
            ring = ring_slots()
            borrowed = ""
            offset = 0
            if int(win.size) < BIG_BAR_MIN:
                wins = self.group_windows()
                if not self._windows_logged:
                    self._windows_logged = True
                    self.log(f"WEG2-BAR1 group windows of this process: "
                             f"{ {k: v >> 20 for k, v in wins.items()} } MiB; dst lanes={self._dst_lanes}")
                plan = borrow_plan(self.group, lane_key, self.cross_pairs, self.big_cards(),
                                   wins, ring, dst_lanes=self._dst_lanes)
                if isinstance(plan, str):
                    self.refusals[lane_key] = plan
                    self.log(f"WEG2-BAR1 lane={lane_key} role=dst group={self.group} NO-WINDOW {plan} -> host lane")
                    self._mark_no_window(lane_key, plan)
                    return None
                borrowed, offset, size, slot = plan
                dptr, handle, full = self._group_own[borrowed]
            else:
                slot = slot_bytes_for(int(win.size))
                dptr, handle, full = cu.vmm_alloc(self._ordinal(), slot * ring)
                size = full
            fd, hold_fds, how = cu.dmabuf_fd(dptr, handle, full, self._ordinal())
        except Exception as exc:  # noqa: BLE001 -- a named refusal, the host lane stays
            self.refusals[lane_key] = f"window: {type(exc).__name__}: {exc}"
            self.log(f"WEG2-BAR1 lane={lane_key} role=dst REFUSED {self.refusals[lane_key]} -> host lane")
            self._mark_no_window(lane_key, self.refusals[lane_key])
            return None
        w = RecvWindow(lane_key, int(dptr) + int(offset), int(handle), int(size), slot, ring, int(fd),
                       list(hold_fds), bdf, self._ordinal(), borrowed=borrowed)
        path = socket_path(self.boot_nonce, lane_key, self.group, self.root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        ls.bind(path)
        ls.listen(4)
        meta = json.dumps({"lane": lane_key, "size": w.size, "slot_bytes": slot, "ring": ring,
                           "bdf": bdf, "how": how, "offset": int(offset), "full": int(full),
                           "borrowed": borrowed}).encode()

        def _serve():
            while True:
                try:
                    conn, _ = ls.accept()
                except OSError:
                    return
                try:
                    socket.send_fds(conn, [meta], [w.dmabuf_fd])
                except OSError as exc:
                    self.log(f"WEG2-BAR1 lane={lane_key} serve failed: {exc!r}")
                    conn.close()
                    continue
                # the connection STAYS OPEN: it is the lane's credit channel
                # (blocking recv on both sides, no file polling, no GIL spin)
                old = w.conn
                w.conn = conn
                if old is not None:
                    try:
                        old.close()
                    except OSError:
                        pass
        th = threading.Thread(target=_serve, name=f"bar1-serve-{lane_key}", daemon=True)
        th.start()
        w.listener, w.thread = ls, th
        self.recv[lane_key] = w
        self.log(f"WEG2-BAR1 lane={lane_key} role=dst group={self.group} window={size >> 20} MiB "
                 f"slot={slot >> 20} MiB ring={ring} bdf={bdf} export={how} "
                 f"borrowed={borrowed or '-'} offset={offset >> 20}MiB served={path}")
        return w

    # -- depositor: connect + map --------------------------------------------
    def connect_peer(self, lane_key: str, timeout_s: Optional[float] = None) -> Optional[PeerWindow]:
        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            Holder, bar1_window,
        )
        path = socket_path(self.boot_nonce, lane_key, other_group(self.group), self.root)
        if timeout_s is None:
            try:
                timeout_s = float(os.environ.get(ENV_CONNECT_S, "240"))
            except ValueError:
                timeout_s = 240.0
        t0 = time.perf_counter()
        meta, fds = None, []
        while True:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(10.0)
                s.connect(path)
                data, fds, _flags, _addr = socket.recv_fds(s, 4096, 4)
                meta = json.loads(data.decode())
                break
            except (OSError, ValueError):
                marker = os.path.join(os.path.dirname(path), "no-window")
                if os.path.exists(marker):
                    try:
                        with open(marker) as fh:
                            why = fh.read().strip()
                    except OSError:
                        why = "?"
                    self.refusals[lane_key] = f"peer serves no window: {why}"
                    self.log(f"WEG2-BAR1 lane={lane_key} role=src REFUSED {self.refusals[lane_key]} -> host lane")
                    return None
                if time.perf_counter() - t0 > timeout_s:
                    self.refusals[lane_key] = f"connect: no window served at {path} within {timeout_s:.0f} s"
                    self.log(f"WEG2-BAR1 lane={lane_key} role=src REFUSED {self.refusals[lane_key]} -> host lane")
                    return None
                time.sleep(0.25)
        if not fds:
            s.close()
            self.refusals[lane_key] = "connect: no fd in the message"
            self.log(f"WEG2-BAR1 lane={lane_key} role=src REFUSED {self.refusals[lane_key]} -> host lane")
            return None
        fd = int(fds[0])
        handle_ = None
        try:
            cu = self._cu()
            if self._holder is None:
                self._holder = Holder()
            peer_bdf = str(meta["bdf"])
            window = bar1_window(peer_bdf)
            handle_, sg, _total = self._holder.hold(fd, self.bdf())
            hits = sorted([e for e in sg if window.base <= e.dma_address < window.end],
                          key=lambda e: e.dma_address)
            if not hits:
                raise RuntimeError(f"none of {len(sg)} sg entries inside the peer's BAR1")
            start = hits[0].dma_address
            length, expected = 0, start
            for e in hits:
                if e.dma_address != expected:
                    break
                length += e.length
                expected += e.length
            m_off = int(meta.get("offset", 0) or 0)
            if length < m_off + int(meta["size"]):
                raise RuntimeError(f"contiguous BAR1 run {length} < offset {m_off} + window {meta['size']}")
            offset = start - window.base + m_off
            page = mmap.PAGESIZE
            m_offset = (offset // page) * page
            lead_in = offset - m_offset
            m_length = int(meta["size"]) + lead_in
            res_fd = os.open(f"/sys/bus/pci/devices/{peer_bdf}/resource1_wc", os.O_RDWR | os.O_SYNC)
            try:
                mapped = mmap.mmap(res_fd, m_length, mmap.MAP_SHARED,
                                   mmap.PROT_READ | mmap.PROT_WRITE, offset=m_offset)
            finally:
                os.close(res_fd)
            host = ctypes.addressof(ctypes.c_char.from_buffer(mapped)) + lead_in
            cu.register_io(host - lead_in, m_length)
            dev = cu.dev_ptr(host - lead_in) + lead_in
        except Exception as exc:  # noqa: BLE001 -- a named refusal, the host lane stays
            if handle_ is not None:
                try:
                    self._holder.release(handle_)
                except Exception:  # noqa: BLE001
                    pass
            self.refusals[lane_key] = f"map: {type(exc).__name__}: {exc}"
            self.log(f"WEG2-BAR1 lane={lane_key} role=src REFUSED {self.refusals[lane_key]} -> host lane")
            try:
                s.close()
            except OSError:
                pass
            return None
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        s.settimeout(None)
        p = PeerWindow(lane_key, int(dev), int(meta["size"]), int(meta["slot_bytes"]), int(meta["ring"]),
                       peer_bdf, int(handle_), mapped, host - lead_in, sock=s)
        self.peers[lane_key] = p
        self.log(f"WEG2-BAR1 lane={lane_key} role=src group={self.group} mapped peer={peer_bdf} "
                 f"window={p.size >> 20} MiB slot={p.slot_bytes >> 20} MiB ring={p.ring} "
                 f"borrowed={meta.get('borrowed') or '-'} "
                 f"dev_ptr={p.dev_ptr:#x} ms={(time.perf_counter() - t0) * 1000:.0f}")
        return p

    # -- setup for all cross lanes of this rank ---------------------------------
    def setup(self, lane_keys) -> dict:
        """Windows first (so the peers can connect), then the peer mappings.
        Runs on a helper thread at boot: the CUDA context is made current
        there before the first driver call."""
        if not lanes_on():
            return {}
        try:
            import torch
            torch.cuda.set_device(self._ordinal())
            torch.cuda.current_stream(self._ordinal())
        except Exception as exc:  # noqa: BLE001 -- the driver calls will name it
            self.log(f"WEG2-BAR1 setup: torch device not set on this thread: {exc!r}")
        roles = {lk: self.role(lk) for lk in lane_keys}
        self._dst_lanes = sorted(lk for lk, r in roles.items() if r == "dst")
        for lk, r in roles.items():
            if r == "dst":
                self.open_window(lk)
        for lk, r in roles.items():
            if r == "src":
                self.connect_peer(lk)
        self.ready = True
        self.log(f"WEG2-BAR1 setup group={self.group} rank={self.rank} "
                 f"windows={sorted(self.recv)} peers={sorted(self.peers)} "
                 f"refused={ {k: v[:60] for k, v in self.refusals.items()} }")
        return roles


# -- the transport: ONE tag over the ring -------------------------------------

def run_bar1_units(descs, ops, *, lanes: Bar1Lanes, lane_key: str, role: str,
                   seq, phase: str, no_write=None, liveness=None,
                   budget_s: float = 120.0, device: int = 0, log=None,
                   order_key=None) -> str:
    """The BAR1 form of ``run_sequential_units``: the tag's pieces cut at the
    ring slot (``tp.batch_descs``), batch g into slot g % ring.

    Credits ride the lane's open AF_UNIX connection when both ends have it
    (``SocketCredits``: blocking recv, no polling -- xsn372 measured the
    file-flag polling at ~9 ms of GIL starvation per batch on a receiver
    with two collects in flight), else file flags (``FileCredits``, the desk).
    DEPOSIT (role 'src'): send the plan, for g >= ring wait free(g-ring),
    queue the batch's copies into the PEER window, sync one behind, send
    full(g); at the end wait for the collector's done. COLLECT (role 'dst'):
    take the lane's turn (``order_key`` = (flip, index): with two collects in
    flight one lane's tags must not interleave), receive the plan and check
    it against the own cut once, per batch wait full(g), copy out of the OWN
    window, sync, send free(g); at the end send done. Returns "" or the
    refusal text.
    """
    from sglang.srt.weg2 import weight_exchange_transport as tp
    log = log or lanes.log
    win = lanes.window_for(lane_key, role)
    if win is None:
        return f"bar1 lane {lane_key}: no window on this side (role={role})"
    base, slot_bytes, ring = win
    batches = tp.batch_descs(list(descs), slot_bytes=int(slot_bytes), first_seq=0)
    if not batches:
        return ""
    if role == "dst" and not lanes.take_turn(lane_key, order_key, budget_s):
        return (f"bar1 collect lane={lane_key} seq={seq}: the lane's turn {order_key} did not "
                f"come within {budget_s:.0f} s (an earlier tag's collect is stuck; pending="
                f"{sorted(lanes.turn.get(lane_key) or ())})")
    try:
        return _run_bar1_tag(descs, ops, lanes, lane_key, role, seq, phase, no_write, liveness,
                             budget_s, device, log, base, slot_bytes, ring, batches, tp)
    finally:
        if role == "dst":
            lanes.leave_turn(lane_key, order_key)


def _recv_marked(cred, kind: str, g: int, budget_s: float, d: str, seq, lane_key: str,
                 rank: int, wait: str):
    """The depositor's credit wait; longer than SLOW_WAIT_S it posts a
    `blocked` flag (#22) the wakers' cycle reader sees, removed on return."""
    posted = []

    def slow():
        if not posted:
            posted.append(1)
            try:
                post_flag(d, "blocked", seq, g, {"since": time.time(), "wait": wait,
                                                 "lane": lane_key, "rank": int(rank)})
            except OSError:
                pass

    cred.on_slow = slow
    try:
        return cred.recv(kind, g, budget_s)
    finally:
        cred.on_slow = None
        if posted:
            try:
                os.unlink(_flag_path(d, "blocked", seq, g))
            except OSError:
                pass


def _run_bar1_tag(descs, ops, lanes, lane_key, role, seq, phase, no_write, liveness,
                  budget_s, device, log, base, slot_bytes, ring, batches, tp) -> str:
    d = lanes.flags(lane_key, role)
    chan = lanes.channel(lane_key, role)
    cred = SocketCredits(chan, liveness) if chan is not None else FileCredits(d, seq, liveness)
    via = "sock" if chan is not None else "file"
    total = sum(int(b.total_bytes) for b in batches)
    npieces = sum(len(b.pieces) for b in batches)
    streams = []
    for _ in range(int(ring)):
        try:
            streams.append(ops.create_stream(int(device)))
        except Exception:  # noqa: BLE001 -- the desk fakes carry no stream
            streams.append(0)
    _nw = no_write or ()
    t0 = time.perf_counter()
    t_wait = t_copy = 0.0
    log(f"WEG2-BAR1 mapped lane={lane_key} phase={phase} seq={seq} bytes={total} "
        f"units={npieces} batches={len(batches)} slot={slot_bytes >> 20}MiB ring={ring} credits={via}")
    nb = len(batches)
    lag = (1 if role == "src" else 0) if int(ring) >= 3 else 0
    plan = [[[str(getattr(descs[pc.desc_index], "param_name", "?")),
              str(getattr(descs[pc.desc_index], "tag", "") or ""), int(pc.nbytes), int(pc.slot_off)]
             for pc in b.pieces] for b in batches]
    lanes.last_seq[lane_key] = seq
    try:
        if role == "src":
            cred.send("plan", 0, {"batches": plan})
        else:
            tw = time.perf_counter()
            got = cred.recv("plan", 0, budget_s)
            t_wait += time.perf_counter() - tw
            if got is None:
                return (f"bar1 collect lane={lane_key} seq={seq}: no plan record within "
                        f"{budget_s:.0f} s (depositor gone or stuck)")
            if (got.get("batches") or []) != plan:
                return (f"bar1 collect lane={lane_key} seq={seq}: the deposit plan "
                        f"({len(got.get('batches') or [])} batches) differs from this side's cut "
                        f"({len(plan)}) -- the two plans disagree")

        def _finish(g):
            nonlocal t_copy
            tc = time.perf_counter()
            ops.synchronize(streams[ring_slot(g, ring)])
            t_copy += time.perf_counter() - tc
            cred.send("full" if role == "src" else "free", g)

        for g, batch in enumerate(batches):
            sbase = base + ring_slot(g, ring) * slot_bytes
            stream = streams[ring_slot(g, ring)]
            if role == "src":
                if g >= ring:
                    tw = time.perf_counter()
                    if _recv_marked(cred, "free", g - ring, budget_s, d, seq, lane_key,
                                    lanes.rank, "free") is None:
                        return (f"bar1 deposit lane={lane_key} seq={seq}: no 'free' for batch "
                                f"{g - ring} within {budget_s:.0f} s (collector gone or stuck)")
                    t_wait += time.perf_counter() - tw
                for piece in batch.pieces:
                    desc = descs[piece.desc_index]
                    if desc.src_ptr is None:
                        return (f"bar1 deposit lane={lane_key} batch {g}: desc "
                                f"{getattr(desc, 'param_name', '?')!r} carries no src_ptr")
                    src = int(desc.src_ptr) + int(piece.src_off)
                    dst = sbase + int(piece.slot_off)
                    if piece.kind == tp.FLAT:
                        ops.memcpy_async(dst, src, int(piece.nbytes), stream)
                    else:
                        ops.memcpy2d_async(dst, int(piece.run_bytes), src, int(piece.spitch),
                                           int(piece.run_bytes), int(piece.rows), stream)
                if g >= lag:
                    _finish(g - lag)
                continue
            # ---- COLLECT ----
            tw = time.perf_counter()
            got = cred.recv("full", g, budget_s)
            t_wait += time.perf_counter() - tw
            if got is None:
                return (f"bar1 collect lane={lane_key} seq={seq}: no 'full' for batch {g} "
                        f"within {budget_s:.0f} s (depositor gone or stuck)")
            for piece in batch.pieces:
                desc = descs[piece.desc_index]
                name = str(getattr(desc, "param_name", "?"))
                tag = str(getattr(desc, "tag", "") or "")
                if (tag, name) in _nw or name in _nw:
                    continue
                if desc.dst_ptr is None:
                    return f"bar1 collect lane={lane_key} batch {g}: desc {name!r} carries no dst_ptr"
                dst = int(desc.dst_ptr) + int(piece.dst_off)
                src = sbase + int(piece.slot_off)
                if piece.kind == tp.FLAT:
                    ops.memcpy_async(dst, src, int(piece.nbytes), stream)
                else:
                    ops.memcpy2d_async(dst, int(piece.dpitch), src, int(piece.run_bytes),
                                       int(piece.run_bytes), int(piece.rows), stream)
            if g >= lag:
                _finish(g - lag)
        for g in range(max(0, nb - lag), nb):
            _finish(g)
        if role == "dst":
            if via == "file":
                for g in range(max(0, nb - ring), nb):
                    try:
                        os.unlink(_flag_path(d, "free", seq, g))
                    except FileNotFoundError:
                        pass
            cred.send("done", 0)
        else:
            # the collector's done closes the tag: its last copies out of the
            # slots are through before this side's next tag writes them. On a
            # byte stream the trailing frees (nobody's credit) come first.
            tw = time.perf_counter()
            if via == "sock":
                for g in range(max(0, nb - ring), nb):
                    if _recv_marked(cred, "free", g, budget_s, d, seq, lane_key,
                                    lanes.rank, "trailing-free") is None:
                        return (f"bar1 deposit lane={lane_key} seq={seq}: no trailing 'free' for "
                                f"batch {g} within {budget_s:.0f} s")
            if _recv_marked(cred, "done", 0, budget_s, d, seq, lane_key,
                            lanes.rank, "done") is None:
                return (f"bar1 deposit lane={lane_key} seq={seq}: the collector never reported "
                        f"done within {budget_s:.0f} s")
            t_wait += time.perf_counter() - tw
    except RuntimeError as exc:
        return f"bar1 lane={lane_key} seq={seq}: {exc}"
    total_s = time.perf_counter() - t0
    log(f"WEG2-BAR1 lane-time lane={lane_key} phase={phase} seq={seq} units={npieces} "
        f"batches={nb} bytes={total} total_ms={total_s * 1000:.0f} "
        f"wait_ms={t_wait * 1000:.0f} copy_sync_ms={t_copy * 1000:.0f} "
        f"rate_GBs={(total / max(total_s, 1e-9)) / 1e9:.1f} via=bar1 credits={via}")
    return ""
