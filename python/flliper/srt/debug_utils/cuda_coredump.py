"""CUDA coredump helpers.

When FLLIPER_CUDA_COREDUMP=1, this module injects CUDA coredump environment
variables into the current process so that GPU exceptions (e.g. illegal
memory access) produce lightweight coredump files for post-mortem analysis
with cuda-gdb.

The injection happens at module import time via _inject_env() on a
best-effort basis.  If any CUDA_* variable is already present in the
environment (e.g. set by the user in the shell), injection is skipped for
that variable and a warning is printed.  For strict guarantees, set the
CUDA_* env vars in the shell before launching Python.
"""

import ctypes
import glob
import logging
import os
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

from flliper.srt.environ import envs

_CUDA_COREDUMP_FLAGS = (
    "skip_nonrelocated_elf_images,skip_global_memory,"
    "skip_shared_memory,skip_local_memory,skip_constbank_memory"
)
_SKIP_ABORT = "skip_abort"


def generation_flags(flags: Optional[str] = None) -> str:
    """The generation flags a dump in this fork is written with: the lightweight
    set (or ``flags`` as given) plus ``skip_abort`` while
    FLLIPER_CUDA_COREDUMP_SKIP_ABORT is on (default). skip_abort: the driver
    writes the GPU dump and then returns the error to the application instead of
    calling abort() -- the rank dies down its logged exception path and no CPU
    core of the whole rank is written (NF #162)."""
    parts = [p.strip() for p in (_CUDA_COREDUMP_FLAGS if flags is None else flags).split(",") if p.strip()]
    if envs.FLLIPER_CUDA_COREDUMP_SKIP_ABORT.get() and _SKIP_ABORT not in parts:
        parts.append(_SKIP_ABORT)
    return ",".join(parts)


def is_enabled() -> bool:
    return envs.FLLIPER_CUDA_COREDUMP.get()


def get_dump_dir() -> str:
    # Resolve the base dir the same way as the uploader
    # (.github/actions/upload-cuda-coredumps/action.yml) so they agree; an empty
    # FLLIPER_CUDA_COREDUMP_DIR counts as unset, like the action's `[ -n ... ]`.
    explicit = envs.FLLIPER_CUDA_COREDUMP_DIR.get()
    runner_temp = os.getenv("RUNNER_TEMP")
    if explicit:
        base = explicit
    elif runner_temp:
        base = os.path.join(runner_temp, "flliper_cuda_coredumps")
    else:
        base = "/tmp/flliper_cuda_coredumps"
    # Isolate dumps per (run, attempt): on a shared self-hosted runner a leftover
    # dump from one job must not be picked up and mis-attributed by a later one.
    run_id = os.getenv("GITHUB_RUN_ID")
    if run_id:
        attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")
        return os.path.join(base, f"{run_id}-{attempt}")
    return base


def _inject_env():
    """Inject CUDA coredump environment variables into the current process.
    If a CUDA_* variable is already present, skip it and log a warning."""
    dump_dir = get_dump_dir()
    os.makedirs(dump_dir, exist_ok=True)

    env_vars = {
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
        "CUDA_COREDUMP_SHOW_PROGRESS": "1",
        "CUDA_COREDUMP_GENERATION_FLAGS": generation_flags(),
        "CUDA_COREDUMP_FILE": f"{dump_dir}/cuda_coredump_%h.%p.%t",
    }
    for key, value in env_vars.items():
        if key in os.environ:
            warnings.warn(
                f"CUDA coredump env var {key} is already set to "
                f"'{os.environ[key]}', skipping injection of '{value}'.",
                stacklevel=2,
            )
        else:
            os.environ[key] = value


def cleanup_dump_dir():
    """Remove stale coredump files from the dump directory."""
    dump_dir = get_dump_dir()
    for f in glob.glob(os.path.join(dump_dir, "cuda_coredump_*")):
        os.remove(f)


def report():
    """Log any CUDA coredump files found after a test failure."""
    dump_dir = get_dump_dir()
    coredump_files = glob.glob(os.path.join(dump_dir, "cuda_coredump_*"))
    if not coredump_files:
        return

    print(f"\n{'='*60}")
    print(f"CUDA coredump(s) detected ({len(coredump_files)} file(s)):")
    for f in coredump_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {f} ({size_mb:.1f} MB)")
    print("Use cuda-gdb to analyze: cuda-gdb -c <coredump_file>")

    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        repo = os.environ.get("GITHUB_REPOSITORY", "sgl-project/sglang")
        print(f"Download from CI: gh run download {run_id} --repo {repo}")

    print(f"{'='*60}\n")


# --------------------------------------------------------------------------
# fork (NF #162, 23.10.): the scheduler ranks' environment, and the CUDA driver
# naming what a GPU exception in a rank will do.
#
# WHEN the driver takes the coredump settings (measured 23.10., libcuda 13020,
# test_nf_cudacore_rank_arming_1023.py): at the first CUDA initialisation of a
# process, and in a scheduler rank that is NOT run_scheduler_process -- it is
# the spawn child's unpickling import of flliper (package init -> srt.utils ->
# srt.platforms -> torch.cuda.is_available()). Env values set before that take
# permanent precedence over the driver API (driver docs, cuCoredumpSetAttribute).
# So a rank's coredump settings are exactly the environment it was SPAWNED with:
# whatever has to be in it must be in the launch_server group process's
# os.environ before the schedulers are spawned. ``_ensure_env`` (run at import in
# every process, sentinel or not) does that; changing the env inside the rank is
# too late and is not attempted.
#
# The import-time injection above prints its line only in the first process (the
# sentinel is inherited), so the boot log showed "Injecting CUDA coredump env
# vars" for launch_server alone and /proc/<scheduler>/environ cannot show the
# rest (setproctitle overwrites that block; desk/nf-rc32-cudacore-1022). Each
# rank therefore proves it on its own log lines:
#
#   rank_env_record() / log_rank_env()  after the rank's logger is set up: the
#       coredump variables as libc getenv returns them (= what the driver read),
#       and whether the dump directory is writable FROM THIS PROCESS. Pure env +
#       file system, never touches CUDA, never raises.
#   driver_report()  after the Scheduler is constructed (CUDA is up): the CUDA
#       driver itself (cuCoredumpGetAttribute[Global], CTK 12.1+) -- does a GPU
#       exception in this rank write a dump, to which (expanded) file, with which
#       flags. A setting the env did NOT carry is set by the API and read back
#       (works only for what the env left unset; the rest is reported as is).
# --------------------------------------------------------------------------
_logger = logging.getLogger(__name__)

RANK_KEYS = (
    "CUDA_ENABLE_COREDUMP_ON_EXCEPTION",
    "CUDA_COREDUMP_SHOW_PROGRESS",
    "CUDA_COREDUMP_GENERATION_FLAGS",
    "CUDA_COREDUMP_FILE",
)

# cuda.h (CTK 13.0, nvidia/cu13/include/cuda.h): CUcoredumpSettings / CUCoredumpGenerationFlags
CU_COREDUMP_ENABLE_ON_EXCEPTION = 1
CU_COREDUMP_TRIGGER_HOST = 2
CU_COREDUMP_FILE = 5
CU_COREDUMP_GENERATION_FLAGS = 7
_FLAG_BITS = (
    (1 << 0, "skip_nonrelocated_elf_images"),
    (1 << 1, "skip_global_memory"),
    (1 << 2, "skip_shared_memory"),
    (1 << 3, "skip_local_memory"),
    (1 << 4, "skip_abort"),
    (1 << 5, "skip_constbank_memory"),
    (1 << 6, "gzip_compress"),
    (1 << 7, "faulted_contexts_only"),
    (1 << 8, "no_errbar_at_exit"),
    (1 << 9, "log_only"),
)
CUDA_SUCCESS = 0
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_INVALID_CONTEXT = 201


def decode_flags(value: int) -> str:
    names = [n for bit, n in _FLAG_BITS if value & bit]
    rest = value & ~sum(b for b, _ in _FLAG_BITS)
    if rest:
        names.append(hex(rest))
    return ",".join(names) or "default"


def encode_flags(flags: str) -> int:
    by_name = {n: b for b, n in _FLAG_BITS}
    return sum(by_name.get(p.strip(), 0) for p in flags.split(",") if p.strip())


def _ensure_env() -> Dict[str, Any]:
    """Every process that imports this module with FLLIPER_CUDA_COREDUMP=1: fill a
    coredump variable that is missing and add skip_abort (FLLIPER_CUDA_COREDUMP_
    SKIP_ABORT, default on) to whatever generation flags are set -- the live NF
    profile appends its own flags without it through --env-p/--env-d, which the
    launcher applies last, and the import-time injection skips a key that is
    already set. Runs in the launch_server group process before it spawns the
    schedulers, which is the last moment it counts for them (see above)."""
    filled, skip_abort = [], "off"
    expected = {
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
        "CUDA_COREDUMP_SHOW_PROGRESS": "1",
        "CUDA_COREDUMP_GENERATION_FLAGS": generation_flags(),
        "CUDA_COREDUMP_FILE": f"{get_dump_dir()}/cuda_coredump_%h.%p.%t",
    }
    for key, value in expected.items():
        if not os.environ.get(key):
            os.environ[key] = value  # putenv: the libc environment a spawned child inherits
            filled.append(key)
    if envs.FLLIPER_CUDA_COREDUMP_SKIP_ABORT.get():
        before = os.environ.get("CUDA_COREDUMP_GENERATION_FLAGS", "")
        after = generation_flags(before)
        skip_abort = "present" if after == before else "added"
        if after != before:
            os.environ["CUDA_COREDUMP_GENERATION_FLAGS"] = after
    return {"filled": filled, "skip_abort": skip_abort}


def libc_getenv(key: str) -> Optional[str]:
    """The value the CUDA driver sees: libc getenv, not os.environ and never
    /proc/<pid>/environ (setproctitle overwrites that block)."""
    try:
        libc = ctypes.CDLL(None)
        libc.getenv.restype = ctypes.c_char_p
        libc.getenv.argtypes = [ctypes.c_char_p]
        raw = libc.getenv(key.encode())
    except Exception:  # noqa: BLE001 - a witness never stops a rank
        return os.environ.get(key)
    return raw.decode(errors="replace") if raw is not None else None


def _probe_dir(path: str) -> Tuple[bool, str]:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, f".arm_probe.{os.getpid()}")
        with open(probe, "wb") as fh:
            fh.write(b"x")
        os.remove(probe)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def rank_env_record(label: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """What this rank was spawned with (read-only). None when the switch is off."""
    try:
        if not is_enabled():
            return None
        seen = dict(_IMPORT_ENV) if _IMPORT_ENV is not None else {k: libc_getenv(k) for k in RANK_KEYS}
        target_dir = os.path.dirname(seen["CUDA_COREDUMP_FILE"] or "") or "."
        writable, why = _probe_dir(target_dir)
        flags = seen["CUDA_COREDUMP_GENERATION_FLAGS"] or ""
        skip_abort_ok = (not envs.FLLIPER_CUDA_COREDUMP_SKIP_ABORT.get()) or _SKIP_ABORT in flags.split(",")
        armed = (seen["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] == "1" and bool(seen["CUDA_COREDUMP_FILE"])
                 and writable and skip_abort_ok)
        return {"label": label, "pid": os.getpid(), "armed": armed, "getenv": seen, "dir": target_dir,
                "writable": writable, "why": why, "skip_abort_ok": skip_abort_ok}
    except Exception as exc:  # noqa: BLE001 - a witness never stops a rank
        return {"label": label, "pid": os.getpid(), "armed": False, "error": f"{type(exc).__name__}: {exc}"}


def log_rank_env(label: Optional[str] = None, log: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    rec = rank_env_record(label)
    if rec is None:
        return None
    log = log or _logger
    if "error" in rec:
        log.error("CUDA-COREDUMP RANK-ENV NOT ARMED rank=%s pid=%s: the check raised %s",
                  rec["label"], rec["pid"], rec["error"])
        return rec
    g = rec["getenv"]
    line = (
        "rank=%s pid=%s getenv: CUDA_ENABLE_COREDUMP_ON_EXCEPTION=%s CUDA_COREDUMP_FILE=%s "
        "CUDA_COREDUMP_GENERATION_FLAGS=%s dir=%s writable=%d%s (libc getenv = the environment this rank was spawned "
        "with and the CUDA driver latched at its first init; /proc/<pid>/environ of a rank is overwritten by "
        "setproctitle)"
    ) % (rec["label"], rec["pid"], g["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"], g["CUDA_COREDUMP_FILE"],
         g["CUDA_COREDUMP_GENERATION_FLAGS"], rec["dir"], int(rec["writable"]),
         (" why=" + rec["why"]) if rec["why"] else "")
    if rec["armed"]:
        log.info("CUDA-COREDUMP RANK-ENV " + line)
    else:
        log.error("CUDA-COREDUMP RANK-ENV NOT ARMED " + line)
    return rec


class CudaDriver:
    """ctypes view of the four coredump attribute calls of libcuda (CTK 12.1+)."""

    def __init__(self, lib: Any):
        self._lib = lib

    @classmethod
    def load(cls) -> "CudaDriver":
        return cls(ctypes.CDLL("libcuda.so.1"))

    def init(self) -> int:
        # A no-op returning 0 in a rank whose CUDA is up; elsewhere it latches the
        # coredump env as it stands NOW (that is the point: read what the driver took).
        return int(self._lib.cuInit(0))

    def get(self, attrib: int, scope: str) -> Tuple[int, bytes]:
        fn = self._lib.cuCoredumpGetAttribute if scope == "context" else self._lib.cuCoredumpGetAttributeGlobal
        size = ctypes.c_size_t(0)
        rc = int(fn(attrib, None, ctypes.byref(size)))
        if rc != CUDA_SUCCESS:
            return rc, b""
        buf = ctypes.create_string_buffer(size.value)  # the exact size: a larger one is CUDA_ERROR_INVALID_VALUE
        rc = int(fn(attrib, buf, ctypes.byref(size)))
        return rc, buf.raw

    def set(self, attrib: int, raw: bytes, scope: str) -> int:
        fn = self._lib.cuCoredumpSetAttribute if scope == "context" else self._lib.cuCoredumpSetAttributeGlobal
        buf = ctypes.create_string_buffer(raw, len(raw))
        size = ctypes.c_size_t(len(raw))
        return int(fn(attrib, buf, ctypes.byref(size)))


def _read_driver(drv: Any, scope: str) -> Tuple[int, Dict[str, Any]]:
    rc, en = drv.get(CU_COREDUMP_ENABLE_ON_EXCEPTION, scope)
    if rc != CUDA_SUCCESS:
        return rc, {}
    rc, fl = drv.get(CU_COREDUMP_GENERATION_FLAGS, scope)
    if rc != CUDA_SUCCESS:
        return rc, {}
    rc, fi = drv.get(CU_COREDUMP_FILE, scope)
    if rc != CUDA_SUCCESS:
        return rc, {}
    flags = int.from_bytes(fl[:4], "little") if fl else 0
    return CUDA_SUCCESS, {
        "enable": bool(en and en[0]),
        "flags": flags,
        "file": fi.split(b"\0", 1)[0].decode(errors="replace"),
    }


def driver_report(
    label: Optional[str] = None,
    log: Optional[logging.Logger] = None,
    driver_factory: Optional[Callable[[], Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Ask the CUDA driver what a GPU exception in this process does (see the
    block comment above); logs one CUDA-COREDUMP DRIVER line. Never raises."""
    log = log or _logger
    try:
        if not is_enabled():
            return None
        drv = (driver_factory or CudaDriver.load)()
        init_rc = drv.init()
        scope = "context"
        rc, st = _read_driver(drv, scope)
        if rc in (CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_NOT_INITIALIZED):
            scope = "global"
            rc, st = _read_driver(drv, scope)
        if rc != CUDA_SUCCESS:
            log.error("CUDA-COREDUMP DRIVER rank=%s pid=%s UNREADABLE (CUresult %s, cuInit %s): the driver cannot "
                      "say whether this rank dumps", label, os.getpid(), rc, init_rc)
            return {"label": label, "rc": rc, "init_rc": init_rc}
        source = "env"
        want_flags = generation_flags(os.environ.get("CUDA_COREDUMP_GENERATION_FLAGS") or None)
        want_bits = encode_flags(want_flags)
        need_set = (not st["enable"]) or (
            envs.FLLIPER_CUDA_COREDUMP_SKIP_ABORT.get() and not st["flags"] & encode_flags(_SKIP_ABORT)
        )
        if need_set:
            # The env reached this rank too late (after cuInit) or without skip_abort: set it by API.
            want_file = os.environ.get("CUDA_COREDUMP_FILE") or f"{get_dump_dir()}/cuda_coredump_%h.%p.%t"
            set_rc = []
            for sc in ("global", scope) if scope == "context" else ("global",):
                set_rc.append(drv.set(CU_COREDUMP_FILE, want_file.encode() + b"\0", sc))
                set_rc.append(drv.set(CU_COREDUMP_GENERATION_FLAGS, int(want_bits).to_bytes(4, "little"), sc))
                set_rc.append(drv.set(CU_COREDUMP_ENABLE_ON_EXCEPTION, b"\x01", sc))
            rc, st = _read_driver(drv, scope)
            source = "api(set_rc=%s)" % ",".join(str(r) for r in set_rc)
            if rc != CUDA_SUCCESS:
                log.error("CUDA-COREDUMP DRIVER rank=%s pid=%s UNREADABLE after the API set (CUresult %s)",
                          label, os.getpid(), rc)
                return {"label": label, "rc": rc, "init_rc": init_rc}
        rec = {
            "label": label, "pid": os.getpid(), "rc": CUDA_SUCCESS, "init_rc": init_rc, "scope": scope,
            "source": source, "enable": st["enable"], "flags": st["flags"], "flags_names": decode_flags(st["flags"]),
            "file": st["file"],
        }
        skip_abort_ok = (not envs.FLLIPER_CUDA_COREDUMP_SKIP_ABORT.get()) or bool(st["flags"] & encode_flags(_SKIP_ABORT))
        dump_dir_ok = bool(st["file"]) and os.path.isdir(os.path.dirname(st["file"]) or ".")
        rec["ok"] = st["enable"] and dump_dir_ok
        rec["skip_abort_ok"] = skip_abort_ok
        line = ("rank=%s pid=%s scope=%s source=%s enable_on_exception=%d file=%s flags=0x%x(%s) cuInit=%s -- "
                "a GPU exception in this rank writes that file, then %s") % (
            label, rec["pid"], scope, source, int(st["enable"]), st["file"], st["flags"], rec["flags_names"], init_rc,
            "returns the error to the rank (skip_abort)" if st["flags"] & encode_flags(_SKIP_ABORT)
            else "calls abort() (CPU core, no exception path)",
        )
        if rec["ok"] and skip_abort_ok:
            log.info("CUDA-COREDUMP DRIVER " + line)
        elif rec["ok"]:
            log.warning("CUDA-COREDUMP DRIVER WITHOUT-SKIP-ABORT " + line + " (the env this rank was spawned with "
                        "carried the flags without skip_abort; the driver keeps env values over the API)")
        else:
            log.error("CUDA-COREDUMP DRIVER NOT ARMED " + line)
        return rec
    except Exception as exc:  # noqa: BLE001 - a witness never stops a rank
        log.error("CUDA-COREDUMP DRIVER rank=%s pid=%s raised %s: %s", label, os.getpid(), type(exc).__name__, exc)
        return {"label": label, "error": f"{type(exc).__name__}: {exc}"}


# Auto-inject CUDA coredump env vars at import time.
# The sentinel env var is inherited by child processes, so injection only
# happens once in the top-level process.
_SENTINEL = "_FLLIPER_CUDA_COREDUMP_INJECTED"

if is_enabled() and _SENTINEL not in os.environ:
    os.environ[_SENTINEL] = "1"
    print(f"Injecting CUDA coredump env vars (pid={os.getpid()})")
    _inject_env()

# fork (NF #162): every process, sentinel or not -- the launch_server group
# process carries the sentinel from the launcher and must still put skip_abort
# into the environment its schedulers are spawned with.
# _IMPORT_ENV: this process's coredump variables as it was spawned with them,
# read BEFORE _ensure_env can change them -- in a rank that is what the driver
# latched; the RANK-ENV line reports it, not the (later) live value.
_IMPORT_ENV = {k: libc_getenv(k) for k in RANK_KEYS} if is_enabled() else None
_ENSURED = _ensure_env() if is_enabled() else None
