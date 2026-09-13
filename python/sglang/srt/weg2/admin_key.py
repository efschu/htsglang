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

    #1361 [22-fix4] NO LEADING '-', AND THIS IS A BOOT KILLER THAT ALREADY FIRED.
    ``token_urlsafe`` draws from base64url -- A-Za-z0-9 plus ``-`` and ``_`` --
    so 1 key in 64 starts with a hyphen (the boot seat measured 3076 of 200000
    mints = 1.54 %, which is 1/64 to two digits). argparse then reads the VALUE
    as an option and the launch dies with

        argument --admin-api-key: expected one argument

    Boot weg2xsn25's first launch (065608) died exactly there: a random dud
    roughly every 65th boot, with a message that names the flag and not the
    cause, and which no dry run reproduces because the next mint is fine.
    A key starting with TWO hyphens is worse and rarer (1/4096): `_flag_pairs`
    would read it as a flag of its own and MOVE THE P FORM KEY, invalidating
    every ring table for that boot.

    Fixed at the mint rather than only at the one emitter we know about, because
    the value travels into argv, /proc, a file and a header, and a rule that
    holds at one of those is not a rule. The emitter also switched to the
    single-token ``--flag=value`` form (`launcher.admin_key_flag`); belt AND
    braces, since an operator-supplied key never passes through here.

    Entropy cost of the rejection: log2(63/64) = 0.023 bits out of 256.
    """
    while True:
        tok = secrets.token_urlsafe(32)
        if not tok.startswith("-"):
            return tok


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
        # #1361 [22-fix4]: BOTH SPELLINGS. The emitter now ships
        # `--admin-api-key=<value>` as one token; a redactor that only knew the
        # two-token form would have gone on returning "no match" and printed
        # the key into a log that gets pasted into records and tickets. A
        # redactor that silently stops matching is worse than none, because the
        # call site still believes it redacted.
        if tok.startswith("--admin-api-key="):
            out[i] = "--admin-api-key=<redacted>"
        elif tok == "--admin-api-key" and i + 1 < len(out):
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
