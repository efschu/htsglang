"""#1275: a per-boot admin key, so the live levers stop being restart-only.

WHY THIS EXISTS. ``POST /hicache/storage-backend/resize`` shrinks the page
store in flight, by inline LRU eviction, and is explicitly documented not to
require an idle scheduler -- exactly the lever boot weg2sb4 needed when the
ledger's ``store = min(leftover, reap-bound)`` took the leftover branch and put
the boot over the host reap watermark. The endpoint was THERE and the
capability was NOT: every ``/hicache/storage-backend*`` route is
``@auth_level(ADMIN_OPTIONAL)`` and self-gates on
``server_args.admin_api_key`` (``http_server.py``, ``_admin_api_key_missing_response``),
which the launcher never passed. One flag on each group's argv turns a
restart-only lever into a live one.

THE TRAP THAT MAKES THIS MORE THAN ONE FLAG, and it is a boot-killer if
missed. ``ADMIN_OPTIONAL`` does not mean "optional to authenticate". Read
``utils/auth.py``'s decision function::

    ADMIN_OPTIONAL:
      only api_key        -> require api_key
      only admin_api_key  -> require ADMIN_API_KEY
      both                -> require admin_api_key (api_key is NOT accepted)
      neither             -> ALLOWED

So the routes are open today ONLY because no key is set. And these are
``ADMIN_OPTIONAL`` (verified on this tip, not assumed)::

    /flush_cache                 /release_memory_occupation
    /abort_request               /resume_memory_occupation
    /hicache/storage-backend*    (attach / detach / clear / resize)

The first four are precisely what ``weg2/front.py`` drives every flip with --
quiesce is ``POST /flush_cache``, the sleep/wake pair is
``release_memory_occupation`` / ``resume_memory_occupation``. ``Front.rpc``
sends a bare ``session.post`` with NO ``Authorization`` header. Setting
``--admin-api-key`` without teaching the front to authenticate therefore does
not "add security": it makes the next quiesce return 401 and kills the flip.
That is why this module exists as a shared thing rather than as one line in
the launcher -- the key has TWO consumers, and the second one is easy to
forget.

WHAT IS DELIBERATELY NOT CLAIMED. This is not a security boundary. The server
binds loopback, and ``--admin-api-key`` travels on each group's argv, where it
is world-readable in ``/proc/<pid>/cmdline`` to any user on this box (0444).
``server_args`` offers no env or file form for it, so argv is the only
supported channel and that exposure is inherent, not a choice made here. What
the key buys is a CAPABILITY (the admin routes answer at all) and an accident
barrier, not confidentiality against a local user. The 0600 file is how the
FRONT and the operator learn the key without it appearing a second time in
another process's argv; it is not what protects it.
"""

from __future__ import annotations

import os
import secrets
from typing import Dict, Optional

#: The admin routes the front itself drives. Kept here so the launcher, the
#: front and the test all read ONE list; the test cross-checks it against the
#: decorators in ``http_server.py`` so it cannot silently drift.
FRONT_ADMIN_ROUTES = (
    "/flush_cache",
    "/release_memory_occupation",
    "/resume_memory_occupation",
    "/abort_request",
)


def mint() -> str:
    """A fresh key for THIS boot. Never reused, never derived from the tag.

    Per boot rather than per rig: a key that outlives the boot would have to be
    stored somewhere durable and rotated by hand, and nothing here needs it to
    survive a restart -- the front is started by the same launcher run that
    mints it.
    """
    return secrets.token_urlsafe(32)


def key_path(gpu_arb: str, tag: str) -> str:
    """Beside the boot's own state json, named for the boot."""
    return f"{gpu_arb}/weg2/boot_{tag}.adminkey"


def write(path: str, key: str) -> str:
    """Write 0600, creating with the mode rather than fixing it afterwards.

    ``os.open`` with the mode in the ``open`` call closes the window in which a
    file created 0644 is briefly readable; a later ``chmod`` cannot. The
    ``0o600`` is also re-applied for the case where the file already existed
    with a wider mode (``O_CREAT`` does not change an existing file's mode).
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (key + "\n").encode())
    finally:
        os.close(fd)
    os.chmod(path, 0o600)
    return path


def read(path: str) -> Optional[str]:
    """The key, or None when there is no file. Never raises on absence.

    None is a legitimate state: a boot launched without the flag, or the front
    started by hand. Its consequence is that the front sends no header, which
    is correct exactly when the groups were also started without a key.
    """
    try:
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
    except OSError:
        return None
    return key or None


def auth_headers(key: Optional[str]) -> Dict[str, str]:
    """The header the front adds to every RPC, or nothing when unkeyed.

    Unconditional on the path ON PURPOSE. Sending a bearer token to a route
    that does not require one is harmless (``NORMAL`` routes ignore it when no
    ``api_key`` is set), whereas a per-path allowlist here would be a second
    copy of ``http_server.py``'s decorators that drifts the first time a route
    changes level -- and the failure mode of that drift is a dead flip, found
    on metal.
    """
    return {"Authorization": f"Bearer {key}"} if key else {}


def redact_argv(argv) -> list:
    """``argv`` with the admin key's VALUE replaced, for logging only.

    THE SHIPPED ARGV IS NEVER TOUCHED -- only the copy that goes into a log
    line. The distinction matters and was found the honest way: the launcher
    logs each group's full argv twice (its own log and that group's log
    header), so simply "not interpolating the key into a log call" was not
    enough and this module's own docstring promise ("never log ``key``") was
    false for two lines. /proc exposure is inherent and local; a log file is
    durable and gets pasted into records and tickets, which is a strictly worse
    channel, so it is the one worth closing.
    """
    out = list(argv)
    for i, tok in enumerate(out):
        if tok == "--admin-api-key" and i + 1 < len(out):
            out[i + 1] = "<redacted>"
    return out


def redact(key: Optional[str]) -> str:
    """What a log line may say about the key: that there is one, not which.

    The launcher logs the PATH; if a value must appear at all it appears like
    this. Never log ``key`` itself -- the front log is world-readable and is
    routinely pasted into records and tickets.
    """
    if not key:
        return "(none)"
    return f"(set, {len(key)} chars, ...{key[-4:]})"


# ---------------------------------------------------------------- stale sweep
#: #1303 (rescoped 2026-09-10): the residue this sweep exists for. Measured on
#: the box at 10:4xZ: `boot_weg2sn5pre.adminkey` and `boot_weg2xsn5.adminkey`,
#: both from Sep 9, both with ZERO live processes -- per-boot secrets outliving
#: their boots in a shared directory, which is precisely what #1275's
#: `drop_admin_key_file` exists to prevent. They survived because those boots
#: were never torn down (the same two tags own the orphaned `mem_timeseries.sh`
#: samplers the 08:17Z shm ticket lists), so the teardown path that would have
#: removed them never ran. A sweep at LAUNCH is the second half #1275 needs:
#: the boot that starts is the one process guaranteed to run.
_PROBE_MARKERS = ("pgrep", "pkill", "ADMINKEY SWEEP")


def tag_has_live_holder(
    tag: str,
    procs,
    own_pids=(),
) -> bool:
    """Does any LIVE process belong to boot ``tag``?

    PURE, taking ``procs`` as an iterable of ``(pid, cmdline)``, because the
    reader is the part that cannot be tested and the RULE is the part that
    must be. ``own_pids`` are excluded outright.

    THE SELF-MATCH TRAP, which this function exists to not fall into and which
    cost a wrong reading before it existed. The obvious probe is
    ``pgrep -f <tag>``, and it MATCHES ITS OWN COMMAND LINE: asked whether
    `weg2sn5pre` was alive, `pgrep -fc weg2sn5pre` answered **2** for a tag
    with zero real processes, because the shell running the probe and the
    pgrep itself both carry the tag in their argv. Reading that as "alive"
    is harmless; reading the inverse as "dead" is how a sweep deletes a live
    boot's secret. So two exclusions, not one:

    * ``own_pids`` -- this process and anything it was asked to ignore;
    * any cmdline that is itself a PROBE (``pgrep``/``pkill``, or one carrying
      this sweep's own log marker). A process whose whole job is to ask about
      the tag is not a holder of it.

    A cmdline that cannot be read is NOT a holder: an unreadable ``/proc``
    entry is a race with an exiting process, and the only safe reading of a
    race is "gone". The DANGEROUS direction is the other one, and it is
    covered by the caller keeping the live tag explicitly.
    """
    if not tag:
        return False
    own = {int(p) for p in own_pids}
    for pid, cmdline in procs:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        if pid in own:
            continue
        if not cmdline:
            continue
        if any(m in cmdline for m in _PROBE_MARKERS):
            continue
        if tag in cmdline:
            return True
    return False


def read_procs(proc_root: str = "/proc"):
    """``(pid, cmdline)`` for every readable process. The untestable half.

    Reads ``/proc`` directly rather than shelling out to ``pgrep``: a
    subprocess would put the tag into ANOTHER command line and re-create the
    self-match it is being called to avoid.
    """
    out = []
    try:
        names = os.listdir(proc_root)
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"{proc_root}/{name}/cmdline", "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        out.append((int(name), raw.replace(b"\0", b" ").decode("utf-8", "replace")))
    return out


def stale_key_tags(gpu_arb: str, keep_tags=(), procs=None, own_pids=()):
    """Split this directory's key files into (removable, kept) by TAG.

    ``keep_tags`` is unconditional and is the caller's safety belt: the
    current boot's own tag goes in it, so even a wrong liveness verdict cannot
    delete the key of the boot doing the sweeping.
    """
    import glob

    keep = {t for t in keep_tags if t}
    procs = read_procs() if procs is None else list(procs)
    removable, kept = [], []
    prefix = f"{gpu_arb}/weg2/boot_"
    for path in sorted(glob.glob(f"{gpu_arb}/weg2/boot_*.adminkey")):
        tag = path[len(prefix):-len(".adminkey")] if path.startswith(prefix) else ""
        if not tag:
            continue
        if tag in keep or tag_has_live_holder(tag, procs, own_pids):
            kept.append(tag)
        else:
            removable.append(tag)
    return removable, kept


def sweep_stale_keys(gpu_arb: str, keep_tags=(), dry: bool = False) -> str:
    """Remove key files whose boot is provably gone. Returns the log line.

    NEVER RAISES and never removes a kept tag. One line, both lists, so a
    reader can see what was spared as well as what went -- a sweep that
    printed only its removals would be unauditable in exactly the direction
    that matters.
    """
    try:
        removable, kept = stale_key_tags(
            gpu_arb, keep_tags=keep_tags, own_pids=(os.getpid(),)
        )
    except Exception as e:  # noqa: BLE001 - a launch-time sweep never blocks a launch
        return f"WEG2-LAUNCH ADMINKEY SWEEP failed: {type(e).__name__}: {e}"
    gone = []
    for tag in removable:
        path = key_path(gpu_arb, tag)
        if dry:
            gone.append(tag)
            continue
        try:
            os.unlink(path)
            gone.append(tag)
        except FileNotFoundError:
            gone.append(tag)
        except OSError:
            kept.append(tag)
    return (
        f"WEG2-LAUNCH ADMINKEY SWEEP removed={','.join(gone) or 'none'} "
        f"kept={','.join(sorted(set(kept))) or 'none'}"
        + (" (DRY)" if dry else "")
        + " (a per-boot secret must not outlive its boot, #1275; a tag with any "
        "live holder is kept, and the current boot's tag is kept "
        "unconditionally)"
    )
