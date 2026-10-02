"""Replay a frozen real boot through rigdash exactly as the server builds its card (Nutzer 02.10.: "warum faehrst
du nicht die test zum boot?").  A fixture directory holds what the server reads for one boot:

* ``state.json``    -- the launcher's state file (/spinning/docker-acceptance/<line>/state/<boot_id>/)
* ``events.jsonl``  -- its event log, same directory
* ``ring.jsonl.gz`` -- the boot's rows of /var/lib/rigdash/ring.sqlite (table ring, key = boot_id), one JSON per
  line, oldest first; the ring keeps only ~16 min, so freeze while the boot is still in it

``replay_boot(dir)`` returns the card (``ipcboot.build_view``), the Model and the IPC view at the time of the
newest ring row -- what the dashboard showed at that moment.  Freeze a new boot with ``freeze_boot``.
"""

import gzip
import json
import os
import shutil
import sqlite3
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rigdash import ipcboot, ipcstate  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
RING_DB = "/var/lib/rigdash/ring.sqlite"


def replay_boot(d: str, now: Optional[float] = None) -> dict:
    if not os.path.isabs(d):
        d = os.path.join(FIXTURES, d)
    with open(os.path.join(d, "state.json")) as fh:
        st = json.load(fh)
    ev = ipcstate._Events(os.path.join(d, "events.jsonl"))
    ev.poll()
    with gzip.open(os.path.join(d, "ring.jsonl.gz"), "rt") as fh:
        ring = [json.loads(line) for line in fh if line.strip()]
    now = now if now is not None else ring[-1]["t"]
    ipc = ipcstate.boot_view(d, st, ev, now)
    view = ipcboot.build_view(ipc, ring, {"rankstats": {}, "rankstate": {}}, None, now)
    return {"ipc": ipc, "ring": ring, "now": now, "model": ipcboot.model_of(ipc, ring), "view": view,
            "segs": view["timeline"]["segs"]}


def freeze_boot(state_dir: str, dst: str, ring_db: str = RING_DB) -> int:
    """Copy one boot's state.json + events.jsonl and dump its ring rows (read-only) into ``dst``."""
    os.makedirs(dst, exist_ok=True)
    for f in ("state.json", "events.jsonl"):
        shutil.copy(os.path.join(state_dir, f), os.path.join(dst, f))
    with open(os.path.join(state_dir, "state.json")) as fh:
        key = json.load(fh).get("boot_id") or state_dir
    db = sqlite3.connect("file:%s?mode=ro" % ring_db, uri=True)
    n = 0
    with gzip.open(os.path.join(dst, "ring.jsonl.gz"), "wt") as fh:
        for (j,) in db.execute("SELECT j FROM ring WHERE key=? ORDER BY t", (key,)):
            fh.write(j.strip() + "\n")
            n += 1
    return n


if __name__ == "__main__":
    print(freeze_boot(sys.argv[1], sys.argv[2]), "ring rows")
