"""RIG-RAM GUARD (02.10.): no uncapped test run while a rig boot is live.

THE DEATH THIS CLOSES. NF boot y7e (02ddfc4906, 09:26:59Z) stopped by name at
09:45:21Z (W3 PdFlipDrainWitnessUnreachable) after D's TP0 froze inside a D-direct
prefill at 09:43:31. Not a hung forward and not a lock: the HOST thrashed.
VictoriaMetrics (node_exporter, host 'proxmox'): MemAvailable 2.2-9.8 GiB for
the whole boot (rig shmem 61-64 GiB + 47 GiB mlocked expert store + anon), then
the agent LXC 'ct999' grew its anon 11.96 -> 21.49 GiB between 09:42:00 and
09:43:30 (uncapped agent pytest runs: `ps` shows them in
/system.slice/claude.service, memory.max=max) -- host anon 39.8 -> 50.6 GiB,
MemAvailable 2.2 GiB, swap-out 11k pages/s, major faults 27k/s. Every process
on the box stalled: TP0's L3 write-behind passes of ZERO pages took 4-45 s wall
at 0.2-1.7 s CPU, the front's host reader went blind 7-47 s (RATE-GAP), the
idle-clock daemon timed out, node_exporter itself has no sample 09:43:30-09:47:30.
The park RPC timed out (71 s), the drain witness was unreachable, W3.

THE RULE (user 24.09., agenten-tests-toeten-boot-am-host-ram): agent tests during
a boot only under a RAM cap (incg.sh: a cgroup with memory.max). This guard makes
the rule structural: while a rig boot is live, a pytest whose cgroup has no
memory.max at or below the cap refuses to start, by name. Off the rig (no state
tree) it does nothing. Explicit override: RIG_RAM_GUARD=off.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterable, List, Optional, Tuple

#: the state trees the boots publish (launcher state/current -> state.json)
STATE_ROOTS = ("/spinning/docker-acceptance/nf/state", "/spinning/docker-acceptance/27b/state")
#: lifecycle states in which a boot holds the host (anything not ended)
ENDED = {"stopped_clean", "stopped", "failed", "refused", "dead", "aborted", "torn_down"}
#: a boot whose newest heartbeat is older than this is not live
HEARTBEAT_S = 180.0
#: the largest per-run cap that counts as "capped" (incg.sh: 4 GiB per slot)
CAP_BYTES = 4 << 30


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def live_boots(roots: Iterable[str] = STATE_ROOTS, now: Optional[float] = None,
               read: Callable[[str], Optional[str]] = _read) -> List[str]:
    """The boot ids under ``roots`` whose ``current`` state is not ended and
    whose newest heartbeat is fresh."""
    now = time.time() if now is None else float(now)
    out = []
    for root in roots:
        txt = read(os.path.join(root, "current", "state.json"))
        if not txt:
            continue
        try:
            st = json.loads(txt)
        except ValueError:
            continue
        lc = st.get("lifecycle") or {}
        state = lc.get("state") if isinstance(lc, dict) else lc
        if state in ENDED:
            continue
        beats = [float(v.get("ts") or 0) for v in (st.get("heartbeat") or {}).values()
                 if isinstance(v, dict)]
        if beats and now - max(beats) > HEARTBEAT_S:
            continue
        out.append(str(st.get("boot_id") or root))
    return out


def own_memory_max(read: Callable[[str], Optional[str]] = _read,
                   pid: str = "self") -> Tuple[Optional[int], str]:
    """(memory.max in bytes or None for 'max'/unreadable, the cgroup path)."""
    cg = read(f"/proc/{pid}/cgroup") or ""
    path = ""
    for line in cg.splitlines():
        if line.startswith("0::"):
            path = line[3:].strip()
    if not path:
        return None, ""
    # the tightest memory.max on the path from this cgroup up to the root
    best = None
    p = path
    while True:
        v = read("/sys/fs/cgroup" + (p if p != "/" else "") + "/memory.max")
        if v is not None and v.strip() not in ("", "max"):
            try:
                n = int(v.strip())
                best = n if best is None else min(best, n)
            except ValueError:
                pass
        if p in ("", "/"):
            break
        p = os.path.dirname(p) or "/"
    return best, path


def verdict(env=None, read: Callable[[str], Optional[str]] = _read,
            now: Optional[float] = None, roots: Iterable[str] = STATE_ROOTS) -> Optional[str]:
    """None = may run; else the refusal text."""
    env = os.environ if env is None else env
    if str(env.get("RIG_RAM_GUARD", "")).lower() in ("off", "0", "false"):
        return None
    boots = live_boots(roots, now=now, read=read)
    if not boots:
        return None
    cap, path = own_memory_max(read)
    if cap is not None and cap <= CAP_BYTES:
        return None
    return ("RIG-RAM GUARD: rig boot(s) %s live and this pytest runs uncapped (cgroup %s, "
            "memory.max=%s > %d GiB). y7e died at 09:45Z of exactly this (agent tests pushed "
            "the host to 2.2 GiB MemAvailable). Run it under incg.sh (4 GiB cgroup) or set "
            "RIG_RAM_GUARD=off deliberately." % (
                ",".join(boots), path or "?", "max" if cap is None else cap, CAP_BYTES >> 30))
