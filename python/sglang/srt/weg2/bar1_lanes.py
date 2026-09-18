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

def _flag_path(d: str, kind: str, seq: int, g: int) -> str:
    return os.path.join(d, f"{kind}.{int(seq)}.{int(g)}")


def post_flag(d: str, kind: str, seq: int, g: int, payload: Optional[dict] = None) -> None:
    """Atomic: written to a tmp name, renamed into place (the reader unlinks it)."""
    os.makedirs(d, exist_ok=True)
    p = _flag_path(d, kind, seq, g)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        if payload is not None:
            json.dump(payload, fh)
    os.replace(tmp, p)


def take_flag(d: str, kind: str, seq: int, g: int, timeout_s: float,
              liveness=None) -> Optional[dict]:
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
        if liveness is not None and (n & 1023) == 0 and not liveness():
            return None
        time.sleep(0.00002)


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

    def window_for(self, lane_key: str, role: str):
        """(base device pointer, slot_bytes, ring) of this lane's window as seen
        from this side, or None when this side holds no BAR1 end of it."""
        if role == "dst":
            w = self.recv.get(lane_key)
            return (int(w.dptr), int(w.slot_bytes), int(w.ring)) if w is not None else None
        p = self.peers.get(lane_key)
        return (int(p.dev_ptr), int(p.slot_bytes), int(p.ring)) if p is not None else None

    # -- the per-tag agreement ------------------------------------------------
    def lane_mode(self, lane_key: str, role: str, seq: int, timeout_s: float = 120.0,
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
                finally:
                    conn.close()
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
                s.close()
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
            return None
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        p = PeerWindow(lane_key, int(dev), int(meta["size"]), int(meta["slot_bytes"]), int(meta["ring"]),
                       peer_bdf, int(handle_), mapped, host - lead_in)
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
                   seq: int, phase: str, no_write=None, liveness=None,
                   budget_s: float = 120.0, device: int = 0, log=None) -> str:
    """The BAR1 form of ``run_sequential_units``: the tag's pieces cut at the
    ring slot (``tp.batch_descs``), batch g into slot g % ring.

    DEPOSIT (role 'src'): for g >= ring wait ``free.<seq>.<g-ring>`` (the
    collector drained the slot), queue every piece's copy into the PEER window
    (posted DMA over BAR1), ONE sync, then post ``full.<seq>.<g>`` with the
    batch's record (names, tags, sizes). COLLECT (role 'dst'): wait
    ``full.<seq>.<g>``, check the record's identity BY NAME per piece, queue
    the copies out of the OWN window (D2D on this card), ONE sync, post
    ``free.<seq>.<g>``. Returns "" or the refusal text.
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
    d = lanes.flags(lane_key, role)
    total = sum(int(b.total_bytes) for b in batches)
    npieces = sum(len(b.pieces) for b in batches)
    # ONE STREAM PER SLOT, and the sync runs ONE BATCH BEHIND: batch g is
    # queued on stream g % ring, then batch g-1 (already done while g was
    # being issued) is synced and its flag posted. The copy engine never
    # waits for Python between batches -- xsn365 measured the per-batch form
    # (sync, flag, next issue) at ~1 ms of bubble per 8-MiB batch = half the
    # link on the 5090's lanes.
    streams = []
    for _ in range(int(ring)):
        try:
            streams.append(ops.create_stream(int(device)))
        except Exception:  # noqa: BLE001 -- the desk fakes carry no stream
            streams.append(0)
    _nw = no_write or ()
    t0 = time.perf_counter()
    t_wait = t_copy = 0.0
    log(f"WEG2-BAR1 mapped lane={lane_key} phase={phase} seq={int(seq)} bytes={total} "
        f"units={npieces} batches={len(batches)} slot={slot_bytes >> 20}MiB ring={ring}")
    nb = len(batches)

    def _finish(g):
        """sync batch g's stream and post its flag (full for the depositor,
        free for the collector)."""
        nonlocal t_copy
        tc = time.perf_counter()
        ops.synchronize(streams[ring_slot(g, ring)])
        t_copy += time.perf_counter() - tc
        post_flag(d, "full" if role == "src" else "free", seq, g)

    # LAG: the depositor syncs one batch behind (its copy engine never idles
    # between batches); the collector syncs at once (a D2D copy of one slot
    # is microseconds). That leaves ring-1 batches of SLACK between the two
    # sides -- xsn367 measured lag 2/1 (one batch of slack) at ~2 ms per
    # 9-MiB batch: every flag hop through the scheduler process's GIL costs
    # ~1 ms, and with one batch of slack the two sides waited for each other
    # hop by hop (PP0 4.7 GB/s per lane, single-lane probe 10.6).
    lag = (1 if role == "src" else 0) if int(ring) >= 3 else 0
    # ONE plan record per tag (the batches' pieces by name, tag, size and
    # slot offset) instead of a JSON per batch: the collector checks its own
    # deterministic cut against it once, the per-batch flags stay empty.
    if role == "src":
        post_flag(d, "plan", seq, 0, {"batches": [
            [[str(getattr(descs[pc.desc_index], "param_name", "?")),
              str(getattr(descs[pc.desc_index], "tag", "") or ""), int(pc.nbytes), int(pc.slot_off)]
             for pc in b.pieces] for b in batches]})
    else:
        tw = time.perf_counter()
        got = take_flag(d, "plan", seq, 0, budget_s, liveness=liveness)
        t_wait += time.perf_counter() - tw
        if got is None:
            return (f"bar1 collect lane={lane_key} seq={seq}: no plan record within "
                    f"{budget_s:.0f} s (depositor gone or stuck)")
        theirs = got.get("batches") or []
        mine = [[[str(getattr(descs[pc.desc_index], "param_name", "?")),
                  str(getattr(descs[pc.desc_index], "tag", "") or ""), int(pc.nbytes), int(pc.slot_off)]
                 for pc in b.pieces] for b in batches]
        if theirs != mine:
            return (f"bar1 collect lane={lane_key} seq={seq}: the deposit plan ({len(theirs)} batches) "
                    f"differs from this side's cut ({len(mine)}) -- the two plans disagree")
    for g, batch in enumerate(batches):
        sbase = base + ring_slot(g, ring) * slot_bytes
        stream = streams[ring_slot(g, ring)]
        if role == "src":
            if g >= ring:
                tw = time.perf_counter()
                if take_flag(d, "free", seq, g - ring, budget_s, liveness=liveness) is None:
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
        got = take_flag(d, "full", seq, g, budget_s, liveness=liveness)
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
    # the tail: every batch still in flight, in order (lag of them)
    for g in range(max(0, nb - lag), nb):
        _finish(g)
    if role == "dst":
        # the ring's last free flags have no taker: leave none behind
        for g in range(max(0, len(batches) - ring), len(batches)):
            try:
                os.unlink(_flag_path(d, "free", seq, g))
            except FileNotFoundError:
                pass
    total_s = time.perf_counter() - t0
    log(f"WEG2-BAR1 lane-time lane={lane_key} phase={phase} seq={int(seq)} units={npieces} "
        f"batches={len(batches)} bytes={total} total_ms={total_s * 1000:.0f} "
        f"wait_ms={t_wait * 1000:.0f} copy_sync_ms={t_copy * 1000:.0f} "
        f"rate_GBs={(total / max(total_s, 1e-9)) / 1e9:.1f} via=bar1")
    return ""
