"""The sampler in its own process (Nutzer 30.09. ~21:40Z: "der probenehmer sollte doch nicht an zu viel
last scheitern? der sollte das doch irgendwie parallel davon tun können?").

Until then the 1-s readers (IPC ring, NVML, host, history) were threads in the web server's process:
under /api/live load the GIL made a poll take > 1 s, and the reading itself was the moment.  Now:

  * ``python -m rigdash.sampler`` -- IpcBoots (rank counters + front mirror, 1 s on the tick),
    history.Recorder (NVML 1 s with power from the energy COUNTER, host 5 s, model rows 5 s,
    compaction) -- writes ``history.sqlite`` and the live ring ``ring.sqlite`` in the state dir;
  * the web server only reads both (``RingStore`` incrementally into its in-memory ring) and starts,
    watches and restarts the sampler (``Supervisor``); a dead or silent sampler is shown on the page and
    in /api/health.  The child dies with the service (same cgroup, and it exits when its parent is gone),
    so a rollback to an older release never leaves a second writer behind.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

RING_FILE = "ring.sqlite"
BEAT_STALE_S = 5.0


class RingStore:
    """The live ring shared between the sampler (writer) and the web server (reader): one row per boot
    and sample, the same compact record the in-process ring held."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS ring(key TEXT, t REAL, j TEXT, PRIMARY KEY(key, t)) WITHOUT ROWID")
        self.db.execute("CREATE INDEX IF NOT EXISTS ring_t ON ring(t)")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")

    def append(self, rows: List[tuple], keep: Dict[str, float]) -> None:
        """rows = (key, t, sample); keep = {key: oldest t still in the writer's ring} -- everything older,
        and every key not in ``keep``, goes."""
        with self.lock:
            self.db.execute("BEGIN")
            try:
                self.db.executemany("INSERT OR REPLACE INTO ring(key, t, j) VALUES (?, ?, ?)",
                                    [(k, t, json.dumps(s, separators=(",", ":"))) for k, t, s in rows])
                for k, t0 in keep.items():
                    self.db.execute("DELETE FROM ring WHERE key=? AND t<?", (k, t0))
                if keep:
                    q = ",".join("?" * len(keep))
                    self.db.execute("DELETE FROM ring WHERE key NOT IN (%s)" % q, tuple(keep))
                else:
                    self.db.execute("DELETE FROM ring")
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise

    def since(self, t: float) -> List[tuple]:
        with self.lock:
            return [(k, tt, json.loads(j)) for k, tt, j in
                    self.db.execute("SELECT key, t, j FROM ring WHERE t > ? ORDER BY t", (t,))]

    def extent(self) -> Dict[str, float]:
        with self.lock:
            return dict(self.db.execute("SELECT key, MIN(t) FROM ring GROUP BY key").fetchall())

    def set(self, k: str, v) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", (k, json.dumps(v)))

    def get(self, k: str, default=None):
        with self.lock:
            r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default


class Supervisor:
    """Starts the sampler process, restarts it when it dies, and says so."""

    def __init__(self, cmd: List[str], env: Optional[dict] = None, store: Optional[RingStore] = None):
        self.cmd, self.env, self.store = cmd, env, store
        self.proc: Optional[subprocess.Popen] = None
        self.restarts = 0
        self.last_exit: Optional[dict] = None
        self.started_t: Optional[float] = None
        self.lock = threading.Lock()

    def spawn(self) -> None:
        with self.lock:
            self.proc = subprocess.Popen(self.cmd, env=self.env, stdin=subprocess.DEVNULL)
            self.started_t = time.time()

    def run_forever(self, stop: threading.Event) -> None:
        self.spawn()
        while not stop.is_set():
            stop.wait(1.0)
            p = self.proc
            if p is not None and p.poll() is not None and not stop.is_set():
                self.last_exit = {"t": time.time(), "code": p.returncode, "pid": p.pid}
                self.restarts += 1
                stop.wait(min(30.0, 2.0 * self.restarts))        # back off a crash loop
                if not stop.is_set():
                    self.spawn()
        self.terminate()

    def terminate(self) -> None:
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()

    def status(self, now: Optional[float] = None) -> dict:
        now = now or time.time()
        p = self.proc
        beat = (self.store.get("beat") if self.store else None) or {}
        alive = p is not None and p.poll() is None
        age = (now - beat["t"]) if beat.get("t") else None
        ok = alive and age is not None and age <= BEAT_STALE_S and beat.get("pid") == p.pid
        return {"mode": "prozess", "ok": ok, "alive": alive, "pid": p.pid if p else None, "server_pid": os.getpid(),
                "restarts": self.restarts, "last_exit": self.last_exit, "beat_age_s": round(age, 2) if age is not None else None,
                "held": beat.get("held"), "errors": beat.get("errors") or {},
                "why": None if ok else ("Probennehmer-Prozess läuft nicht" if not alive else
                                        "Probennehmer meldet sich nicht (Herzschlag %s s alt)" % (round(age, 1) if age is not None else "–"))}


def main(argv=None) -> int:
    from . import history, ipcboot
    ap = argparse.ArgumentParser(description="rigdash sampler (own process; the web server only reads)")
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--docker-ssh", default="")
    ap.add_argument("--parent-pid", type=int, default=0)
    args = ap.parse_args(argv)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    signal.signal(signal.SIGINT, lambda *a: stop.set())
    store = RingStore(os.path.join(args.state_dir, RING_FILE))
    db = history.HistoryDB(os.path.join(args.state_dir, "history.sqlite"))
    boots = ipcboot.IpcBoots(store=store, role="sampler")
    rec = history.Recorder(db, boots, shlex.split(args.docker_ssh) if args.docker_ssh else [])
    threading.Thread(target=boots.run_forever, args=(stop,), name="sampler-ipc", daemon=True).start()
    threading.Thread(target=rec.run_forever, args=(stop,), name="sampler-hist", daemon=True).start()
    print("rigdash sampler pid %d, state %s" % (os.getpid(), args.state_dir), flush=True)
    while not stop.is_set():
        if args.parent_pid and os.getppid() != args.parent_pid:
            break                                   # the web server is gone: never a second writer
        try:
            store.set("beat", {"t": time.time(), "pid": os.getpid(), "held": rec.held_view(),
                               "errors": dict(rec.errors, **({"ipc": boots.last_error} if boots.last_error else {}))})
        except Exception as e:  # noqa: BLE001 -- the beat must not kill the sampler
            print("beat: %s" % e, file=sys.stderr, flush=True)
        stop.wait(1.0)
    stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
