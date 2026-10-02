"""L15-10 S4n-a: D publishes its held KV for the waking P (hot handover).

After a successful keep arm (the hold extents survive the kv pause), each D
rank writes ONE descriptor and serves the extents' fds:

* ``<dir>/D.<rank>.json`` -- epoch, rank, pid, the owner vector, and per
  hold base: its role (``k``/``v`` of attention layer ``layer``, or
  ``mamba``), the view offset and row unit inside the base, and the hold
  extents ``[[offset, size], ...]`` in fd order; plus the held spans
  (rid, depth, global D slots, anchor slot) from the published manifest.
* ``<dir>/D.<rank>.sock`` -- a unix socket; every connection gets the JSON
  header and ALL fds (l15_hold_share.send_hold), in the descriptor's order.

The fds stay open (D owns them) until :meth:`SharePublisher.close` at the
next wake -- the extents are D's and live exactly as long as the hold.
Gated by SGLANG_WEG2_L15_HOT_SHARE=1 at the call site; any failure here is a
named log line and no share (the hold itself is untouched).
"""

from __future__ import annotations

import json
import os
import socket
import threading
from typing import Dict, List, Optional, Sequence

DEFAULT_DIR = "/dev/shm/weg2-l15-share"


def share_dir(env) -> str:
    return str(env.get("SGLANG_WEG2_L15_SHARE_DIR", DEFAULT_DIR) or DEFAULT_DIR)


def build_descriptor(*, epoch: int, rank: int, prefix: Sequence[int],
                     bases: Sequence[dict], spans) -> dict:
    """``bases``: [{role, layer, view_off, unit, extents: [[off, size]...]}]
    in the order their fds are sent; ``spans``: manifest HoldSpans."""
    n_fds = sum(len(b["extents"]) for b in bases)
    return {
        "epoch": int(epoch), "rank": int(rank), "pid": os.getpid(),
        "prefix": [int(x) for x in prefix],
        "bases": [dict(b) for b in bases],
        "spans": [{"rid": str(s.rid), "depth": int(s.depth),
                   "slots": [int(x) for x in s.slots],
                   "anchor_slot": int(s.anchor_slot)} for s in spans],
        "n_fds": int(n_fds),
    }


class SharePublisher:
    """Writes the descriptor and serves header + fds to every connection."""

    def __init__(self, directory: str, rank: int, descriptor: dict,
                 fds: Sequence[int]):
        self.directory = directory
        self.rank = int(rank)
        self.descriptor = descriptor
        self.fds = list(fds)
        self.json_path = os.path.join(directory, "D.%d.json" % self.rank)
        self.sock_path = os.path.join(directory, "D.%d.sock" % self.rank)
        self._srv: Optional[socket.socket] = None
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        os.makedirs(self.directory, exist_ok=True)
        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.sock_path)
        srv.listen(8)
        srv.settimeout(0.2)
        self._srv = srv
        tmp = self.json_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.descriptor, fh, sort_keys=True)
        os.replace(tmp, self.json_path)
        self._thr = threading.Thread(target=self._serve, daemon=True,
                                     name="l15-share-D%d" % self.rank)
        self._thr.start()

    def _serve(self) -> None:
        from sglang.srt.weg2.l15_hold_share import send_hold

        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                send_hold(conn, self.descriptor, self.fds)
            except Exception:  # noqa: BLE001 -- one bad peer must not stop the server
                pass
            finally:
                conn.close()

    def close(self) -> None:
        """At D's wake: stop serving, remove the descriptor, close the fds
        (P's imports hold their own references)."""
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
        if self._thr is not None:
            self._thr.join(timeout=2.0)
        for p in (self.json_path, self.sock_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        for f in self.fds:
            try:
                os.close(f)
            except OSError:
                pass
        self.fds = []


def fetch_share(directory: str, rank: int, timeout_s: float = 5.0):
    """P side: (descriptor, fds) of D rank ``rank`` (raises
    l15_hold_share.L15ShareError when nothing is published)."""
    from sglang.srt.weg2.l15_hold_share import L15ShareError, recv_hold

    path = os.path.join(directory, "D.%d.sock" % int(rank))
    if not os.path.exists(path):
        raise L15ShareError("no hold share published for D rank %d" % rank)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    try:
        s.connect(path)
        return recv_hold(s)
    except OSError as exc:
        raise L15ShareError("hold share of D rank %d: %r" % (rank, exc)) from exc
    finally:
        s.close()
