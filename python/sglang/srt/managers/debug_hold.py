"""#1223 HOLD AT THE WALL -- inspect a dying rank instead of re-booting it.

THE DEFECT THIS CLOSES. At a wall whose traceback does not show the root --
the canonical shape is a group STOP on a MIN-reduced slot, where the rank that
voted 0 and the reason it voted 0 live only in that rank's state seconds
earlier -- today the rank raises, prints the traceback, and exits. The cards go
free, the state is gone, and the only way to learn one more fact is another
boot with one more log line. That loop costs a boot per fact.

In hold mode the same boot, with one env flag, does three things at the fatal
exception instead of exiting:

  1. dumps the locals of every frame of the exception's traceback AND the
     current stack of every other Python thread to a file,
  2. opens a localhost-only TCP port with a stdlib ``pdb`` behind it, so the
     operator gets the normal console (``p`` / ``pp`` / ``w`` / ``up`` /
     ``down`` / ``l``) on every rank independently, and
  3. WAITS -- bounded -- instead of unwinding into the SIGQUIT that ends the
     group.

WHAT THIS IS NOT. It does not single-step the scheduler, it is not cuda-gdb,
and ``c`` does not resume serving: by the time this runs the exception has
already unwound the event loop, so continuing returns into ``run_scheduler_
process``'s except block, not into the loop. ``c`` and ``q`` are therefore
both "release the hold"; the distinction is only recorded in the release line.

DEFAULT OFF, AND OFF MEANS OFF. ``maybe_hold`` reads exactly one environment
variable and returns on the very first statement when it is unset -- no
socket, no file, no import of anything below, no log line. The off-path is
pinned by test, not by inspection.

THE HOLD IS BOUNDED BY CONSTRUCTION. ``SGLANG_DEBUG_HOLD_S`` (default 1800)
is a deadline, not a hint: when it passes, ``maybe_hold`` returns and the rank
continues down exactly the path it would have taken without the flag -- the
census emitters, the allocation dump, the SIGQUIT to the parent. A forgotten
hold ends by itself; it can never turn a debug boot into a permanently wedged
rig.

WHAT A HELD RANK BREAKS -- read before using it on a boot that measures
anything. A held rank answers no collective, feeds no watchdog and serves no
request. Every timing figure in a hold-mode boot is meaningless, and the
scheduler watchdog (``--watchdog-timeout``, default 300 s) will SIGQUIT the
PARENT and take the group down while a rank is held unless it is raised past
the hold. ``DEBUG_HOLD=1`` in the launcher does that lifting and prints the
list; see ``devtools/DEBUG-HOLD-1223.md``.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Env contract. Names are read HERE and nowhere else so the launcher and the
# docs have exactly one place to agree with.
# ---------------------------------------------------------------------------
HOLD_ENV = "SGLANG_DEBUG_HOLD"
HOLD_S_ENV = "SGLANG_DEBUG_HOLD_S"
HOLD_PORT_BASE_ENV = "SGLANG_DEBUG_HOLD_PORT_BASE"
HOLD_INJECT_ENV = "SGLANG_DEBUG_HOLD_INJECT"
HOLD_DIR_ENV = "SGLANG_DEBUG_HOLD_DIR"
TAG_ENV = "SGLANG_DEBUG_HOLD_TAG"

DEFAULT_HOLD_S = 1800.0
DEFAULT_PORT_BASE = 5000
DEFAULT_DUMP_DIR = "/spinning/gpu-arb/debug_hold"

#: Per-value repr cap in the dump. A frame local can be a 200k-entry list or a
#: whole batch object; an uncapped repr turns the dump into the thing nobody
#: reads. The cap is per VALUE, so a frame with fifty locals still shows all
#: fifty names -- names are the half you cannot reconstruct later.
MAX_VALUE_CHARS = 2000
#: Elements of a small tensor printed inline. A tensor is summarised as
#: shape/dtype/device ALWAYS; the values are a courtesy for scalars and tiny
#: vectors (the MIN-reduced slot vote is exactly this shape) and are never
#: printed for anything larger.
MAX_TENSOR_ELEMS = 8

MARKER = "#1223 DEBUG-HOLD"
INJECT_MESSAGE = "#1223 INJECTED WALL"


def hold_enabled() -> bool:
    """The one gate. Everything else in this module is downstream of it."""
    return os.environ.get(HOLD_ENV) == "1"


def _hold_seconds() -> float:
    raw = os.environ.get(HOLD_S_ENV)
    if not raw:
        return DEFAULT_HOLD_S
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_HOLD_S


def _port_base() -> int:
    raw = os.environ.get(HOLD_PORT_BASE_ENV)
    if not raw:
        return DEFAULT_PORT_BASE
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_PORT_BASE


def resolve_rank(scheduler, fallback_pp=None, fallback_tp=None) -> int:
    """The rank this hold is for, read the way the Scheduler actually stores it.

    ``scheduler.pp_rank`` DOES NOT EXIST -- the Scheduler keeps its ranks on
    the ParallelState namespace (``scheduler.ps.pp_rank`` /
    ``scheduler.ps.tp_rank``). Commit ba2e88fe is the precedent: a debug
    instrument read the flat attribute, got ``rank=-1`` on every line, and the
    three ranks' dumps were indistinguishable in a merged log. Under this
    topology (pp_size=3, tp_size=1) the PP rank is the one that separates the
    three processes, so it is read first.

    ``scheduler`` is ``None`` whenever the exception fired before the Scheduler
    was built, which is exactly the early-boot case; the caller's own
    ``pp_rank``/``tp_rank`` locals carry the answer then.
    """
    ps = getattr(scheduler, "ps", None)
    for attr in ("pp_rank", "tp_rank"):
        value = getattr(ps, attr, None)
        if isinstance(value, int):
            return value
    for value in (fallback_pp, fallback_tp):
        if isinstance(value, int):
            return value
    return 0


def _summarise(value) -> str:
    """One line for one local, with tensors summarised and never dumped.

    A full tensor repr is both useless (thousands of numbers) and dangerous
    (a CUDA tensor's repr copies it to host, which can itself fail on the rank
    that just died of OOM). Shape/dtype/device is the part that answers
    questions; the values are added only when there are few enough to read.
    """
    try:
        if _is_tensor(value):
            head = (
                f"<tensor shape={tuple(value.shape)} dtype={value.dtype} "
                f"device={value.device}"
            )
            try:
                if value.numel() <= MAX_TENSOR_ELEMS:
                    head += f" values={value.detach().cpu().tolist()}"
            except Exception as exc:  # noqa: BLE001 - a dump may not raise
                head += f" values=<unreadable: {type(exc).__name__}>"
            return head + ">"
        text = repr(value)
    except Exception as exc:  # noqa: BLE001 - a __repr__ may itself be broken
        return f"<repr failed: {type(exc).__name__}: {exc}>"
    if len(text) > MAX_VALUE_CHARS:
        return text[:MAX_VALUE_CHARS] + f"... <truncated, {len(text)} chars>"
    return text


def _is_tensor(value) -> bool:
    """Duck-typed on purpose: importing torch here would be an import in a
    crash handler, and this module must work in a hermetic test with no torch."""
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "device")
        and hasattr(value, "numel")
    )


def _frame_locals_block(frame, out) -> None:
    for name, value in sorted(frame.f_locals.items()):
        out.append(f"      {name} = {_summarise(value)}")


def write_dump(exc: BaseException, rank: int, path: str) -> str:
    """Traceback + every frame's locals + every other thread's stack, to a file.

    The other threads are in here because the interesting ones are not on the
    raising stack: the watchdog thread, the HiCache write-back thread and the
    transfer threads all hold state that explains a group STOP, and none of
    them appears in ``exc.__traceback__``.
    """
    import datetime
    import sys
    import threading
    import traceback

    out = []
    out.append(f"{MARKER} rank={rank}")
    out.append(f"utc: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    out.append(f"pid: {os.getpid()}")
    out.append(f"exception: {type(exc).__name__}: {exc}")
    out.append("")
    out.append("=== TRACEBACK ===")
    out.extend(
        "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ).splitlines()
    )
    out.append("")
    out.append("=== FRAMES OF THE RAISING STACK (locals) ===")
    tb = exc.__traceback__
    depth = 0
    while tb is not None:
        frame = tb.tb_frame
        code = frame.f_code
        out.append(f"  [{depth}] {code.co_name} at {code.co_filename}:{tb.tb_lineno}")
        _frame_locals_block(frame, out)
        tb = tb.tb_next
        depth += 1

    out.append("")
    out.append("=== OTHER PYTHON THREADS (current stacks) ===")
    try:
        names = {t.ident: t.name for t in threading.enumerate()}
        this_thread = threading.get_ident()
        for ident, frame in sys._current_frames().items():
            if ident == this_thread:
                continue
            out.append(f"  --- thread {names.get(ident, '?')} (id={ident}) ---")
            for entry in traceback.format_stack(frame):
                out.extend("    " + ln for ln in entry.rstrip().splitlines())
            out.append("    locals of innermost frame:")
            _frame_locals_block(frame, out)
    except Exception as exc2:  # noqa: BLE001 - a dump may not raise
        out.append(f"  <thread walk failed: {type(exc2).__name__}: {exc2}>")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def dump_path_for(rank: int) -> str:
    import datetime

    tag = os.environ.get(TAG_ENV) or f"pid{os.getpid()}"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = os.environ.get(HOLD_DIR_ENV) or DEFAULT_DUMP_DIR
    return os.path.join(directory, f"{tag}_rank{rank}_{stamp}.txt")


def _serve_pdb(sock, exc: BaseException) -> None:
    """One pdb console on one accepted connection, over the socket.

    ``pdb.Pdb(stdin=..., stdout=...)`` is the stdlib way to put a console on a
    file-like object; this venv is Python 3.12, which has no ``pdb -p``
    attach mode, and adding a pip dependency for a debug path is not on the
    table. ``interaction(None, tb)`` is post-mortem: the console starts at the
    frame that raised, so ``w`` / ``up`` / ``down`` walk the real stack rather
    than this helper's.
    """
    import pdb

    conn_in = sock.makefile("r")
    conn_out = sock.makefile("w")
    try:
        debugger = pdb.Pdb(stdin=conn_in, stdout=conn_out)
        debugger.use_rawinput = False
        debugger.prompt = "(hold-pdb) "
        conn_out.write(
            f"{MARKER}: post-mortem console on {type(exc).__name__}: {exc}\n"
            "  w / up / down / l / p <expr> / pp <expr> ; q releases the hold\n"
        )
        conn_out.flush()
        debugger.reset()
        debugger.interaction(None, exc.__traceback__)
    finally:
        for handle in (conn_in, conn_out):
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - teardown may not raise
                pass


def maybe_hold(exc: BaseException, scheduler=None, pp_rank=None, tp_rank=None) -> bool:
    """Hold this rank at the wall. Returns True if a hold actually ran.

    OFF-PATH CONTRACT: when ``SGLANG_DEBUG_HOLD`` is not exactly "1" this
    returns False on the first statement, having imported nothing, opened
    nothing and logged nothing. That is the property the off-path test pins.
    """
    if not hold_enabled():
        return False
    return _hold(exc, scheduler, pp_rank, tp_rank)


def _hold(exc: BaseException, scheduler, pp_rank, tp_rank) -> bool:
    """The whole hold, behind the gate. Never raises: a debug aid that can
    replace the exception it is diagnosing is worse than no debug aid."""
    import logging
    import socket
    import time

    logger = logging.getLogger(__name__)
    rank = resolve_rank(scheduler, pp_rank, tp_rank)
    hold_s = _hold_seconds()
    port = _port_base() + rank

    try:
        path = write_dump(exc, rank, dump_path_for(rank))
    except Exception as dump_exc:  # noqa: BLE001 - the hold matters more
        path = f"<dump failed: {type(dump_exc).__name__}: {dump_exc}>"

    listener = None
    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 127.0.0.1 ONLY. This is an unauthenticated Python console on a
        # process holding the model; it must never be reachable off-box.
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
    except Exception as bind_exc:  # noqa: BLE001
        logger.error(
            "%s rank=%d could not open the debug port %d (%s: %s) -- NOT holding, "
            "the rank dies as it would without the flag; dump: %s",
            MARKER,
            rank,
            port,
            type(bind_exc).__name__,
            bind_exc,
            path,
        )
        if listener is not None:
            try:
                listener.close()
            except Exception:  # noqa: BLE001
                pass
        return False

    message = str(exc)[:120]
    logger.error(
        "%s rank=%d holding on %s: %s — attach: nc 127.0.0.1 %d ; dump: %s ; "
        "hold ends in %.0f s",
        MARKER,
        rank,
        type(exc).__name__,
        message,
        port,
        path,
        hold_s,
    )

    deadline = time.monotonic() + hold_s
    outcome = "timeout"
    attached = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # A bounded accept, not a blocking one: the deadline must be able
            # to end the hold even when nobody ever attaches. This is the
            # clock that makes a forgotten hold self-limiting.
            listener.settimeout(min(remaining, 5.0))
            try:
                conn, _addr = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            attached = True
            try:
                _serve_pdb(conn, exc)
                # A pdb session that returns normally ended with q / c / EOF.
                # None of the three can resume the event loop (it is already
                # unwound), so all three mean the same thing here: release.
                outcome = "quit"
                break
            except Exception as session_exc:  # noqa: BLE001
                logger.warning(
                    "%s rank=%d pdb session ended abnormally (%s: %s)",
                    MARKER,
                    rank,
                    type(session_exc).__name__,
                    session_exc,
                )
                outcome = "attached"
                break
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
    finally:
        try:
            listener.close()
        except Exception:  # noqa: BLE001
            pass

    if outcome == "timeout" and attached:
        outcome = "attached"
    logger.error("%s rank=%d released: %s", MARKER, rank, outcome)
    return True


def maybe_inject(marker: str) -> None:
    """Raise the injected wall at a named phase point, for the metal proof.

    REFUSES WITHOUT THE HOLD FLAG. An inject env that fires on a boot with no
    hold behind it is a boot-killer wearing a debug label: it would take the
    group down and leave nothing to attach to. The refusal is loud, not
    silent, so a mistyped launcher does not read as "the inject site was never
    reached".
    """
    wanted = os.environ.get(HOLD_INJECT_ENV)
    if not wanted or wanted != marker:
        return
    if not hold_enabled():
        import logging

        logging.getLogger(__name__).error(
            "%s REFUSED: %s=%s is set but %s is not 1. The inject exists only to "
            "prove the hold; firing it without a hold would kill the group and "
            "leave nothing to attach to. Not raising.",
            MARKER,
            HOLD_INJECT_ENV,
            wanted,
            HOLD_ENV,
        )
        return
    raise RuntimeError(f"{INJECT_MESSAGE} (marker={marker})")
