"""S7 D-SURVIVES-P (24.10.): an MPS client rank leaves its card with no GPU work outstanding.

THE SPECIMEN. 27B NVFP4 dual rc39a (b9cddabda1), boot dkr27bnvfp4dualbar1drei10240856, cards 0,1,2,
P = PP0/PP1/PP2 and D = TP0/TP1/TP2 one per card, both groups clients of ONE MPS control daemon
(``--dual-mps on``, ``launcher.start_dual_mps``). 09:09:46 P PP2 dies in Python (#996 SECOND CHUNKED
MINT). 09:09:47 P PP1 (card 1) dies on the closed gloo pair while its last prefill chunk (launched
09:09:46, ~550 gpu-ms) and its PP traffic to the dead PP2 are still on the card -- the process returns
from ``run_scheduler_process`` and exits WITHOUT a device synchronize. Host kernel log, the same second:
``11:09:47.484 CEST NVRM: Xid (PCI:0000:18:00): 31, pid=<D TP1> ... MMU Fault ... FAULT_PDE
ACCESS_TYPE_VIRT_READ`` on card 1, and D TP1 raises ``illegal memory access`` at its next sync
(W113 status copy, then ``dual_d_kv_stage.tick`` -> ``max_live_id``). Cards 2 and 0 follow at 09:09:59
and 09:10:08 (Xid 31 under the MPS server's pid; P PP0's NCCL watchdog reports the IMA at 09:10:08).
Three other P deaths in the dual layout (10210031, 10220215, 10240650) happened at a host-side
scheduling point with nothing on the card: no Xid (10240650 is inside the host journal), D kept
decoding.

THE RULE IT BREAKS. NVIDIA MPS, "When to Use MPS" -> Client Early Termination: "Terminating an MPS
client via CTRL-C or signals is not supported and will lead to undefined behavior"; a client must be
idle (``cudaDeviceSynchronize`` / every stream synchronized) before it ends, otherwise "the affected
MPS server and all its clients must be restarted". The supported way to end a client that is not idle
is ``terminate_client <server PID> <client PID>`` -- "safe to invoke without affecting the MPS server
or its other MPS clients". And Memory Protection and Error Containment: a fatal fault is reported to
all clients on the GPUs it is contained in -- which on this layout is D.

THE FIX (two call sites, one module):
  * ``quiesce_self`` -- the dying rank, in ``run_scheduler_process``'s exception branch BEFORE the
    SIGQUIT to the parent: trip this process's barlink abort windows (its spin kernels end), then a
    BOUNDED device synchronize in a daemon thread; work that does not drain within the bound is ended
    through ``terminate_client`` on this process's own pid. Only then does the rank signal and exit.
  * ``terminate_clients`` -- ``kill_process_tree`` (the parent's teardown after a child's SIGQUIT,
    the launch-phase SIGTERM): every child that is a client of this daemon is terminated through the
    daemon BEFORE it gets SIGKILL. A SIGKILL is exactly the "signals" case above.

TIME BOUND (review SOLL 1). ``quiesce_self`` lasts at most ``bound_s`` (the sync) + ``TERMINATE_BUDGET_S``
(every daemon call of the fallback together); ``terminate_clients`` -- also before ``kill_process_tree``'s SIGKILL --
at most ``TERMINATE_BUDGET_S``: each ``nvidia-cuda-mps-control`` call gets ``min(CONTROL_TIMEOUT_S, rest)`` and
nothing is asked once the budget is spent (a hung daemon, NVIDIA forum 382783 "never returns", costs the budget,
not 3 x 10 s or (2 + N) x 10 s). The two scheduler call sites (``except Exception`` and
``except BaseException``) are siblings of one ``try``, so a rank quiesces at most once.

Armed only for an MPS client process (``CUDA_MPS_PIPE_DIRECTORY`` set -- the dual layout's
``--dual-mps on`` is the only producer on this line) and switch ``FLLIPER_PDFLIP_MPS_EXIT_QUIESCE``
(default ON, user order 14.10.; ``0`` = the pre-S7 exit). Bound ``FLLIPER_PDFLIP_MPS_EXIT_QUIESCE_S``
(default 5 s). Without MPS nothing here runs: a non-MPS context is torn down by the driver alone.

NOT covered (named): a rank killed from outside its own tree (OOM killer, ``docker kill``) -- no code
runs before that SIGKILL; the front's stop-everything on a P death (lifecycle ``dead``) and the P
restart S7 expects are the front's part, not the card's. ``terminate_client`` itself has a field
report of hitting untouched clients (NVIDIA forum 382783) -- that is why it is the FALLBACK after the
drain, never the first step.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

MARK = "S7 MPS-EXIT-QUIESCE"
ENV = "FLLIPER_PDFLIP_MPS_EXIT_QUIESCE"
BOUND_ENV = "FLLIPER_PDFLIP_MPS_EXIT_QUIESCE_S"
DEFAULT_BOUND_S = 5.0
CONTROL = "nvidia-cuda-mps-control"
CONTROL_TIMEOUT_S = 10.0
TERMINATE_BUDGET_S = 12.0
_FAIL_RX = re.compile(r"error|fail|invalid|unknown|not found|no such", re.IGNORECASE)


def armed(env=None) -> bool:
    """An MPS client process (``CUDA_MPS_PIPE_DIRECTORY`` set) with the switch on (default)."""
    env = os.environ if env is None else env
    if not str(env.get("CUDA_MPS_PIPE_DIRECTORY", "") or "").strip():
        return False
    return str(env.get(ENV, "1")).strip().lower() not in ("0", "false", "off", "no")


def bound_s(env=None) -> float:
    env = os.environ if env is None else env
    raw = str(env.get(BOUND_ENV, "") or "").strip()
    if not raw:
        return DEFAULT_BOUND_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("%s %s=%r is not a number -- %.1f s", MARK, BOUND_ENV, raw, DEFAULT_BOUND_S)
        return DEFAULT_BOUND_S


def control(cmd: str, *, env=None, timeout_s: float = CONTROL_TIMEOUT_S, run=subprocess.run) -> Optional[str]:
    """One command to this boot's MPS control daemon (its pipe dir rides in the env); None on failure.
    Never raises."""
    try:
        r = run([CONTROL], input=cmd + "\n", capture_output=True, text=True, timeout=timeout_s,
                env=dict(os.environ if env is None else env))
    except Exception as e:  # noqa: BLE001 - a teardown helper never raises
        logger.warning("%s control %r failed: %s: %s", MARK, cmd, type(e).__name__, e)
        return None
    out = (getattr(r, "stdout", "") or "") + (getattr(r, "stderr", "") or "")
    if getattr(r, "returncode", 1) != 0 or _FAIL_RX.search(out):
        logger.warning("%s control %r refused (rc=%s): %s", MARK, cmd, getattr(r, "returncode", None),
                       out.strip()[:200])
        return None
    return getattr(r, "stdout", "") or ""


def _pids(text: Optional[str]) -> List[int]:
    return [int(t) for t in re.findall(r"\b\d+\b", text or "")]


def mps_clients(*, ctl: Callable[[str], Optional[str]] = control) -> Dict[int, List[int]]:
    """server pid -> client pids, as this daemon reports them."""
    return {s: _pids(ctl(f"get_client_list {s}")) for s in _pids(ctl("get_server_list"))}


class _BudgetSpent(Exception):
    pass


def terminate_clients(pids: Iterable[int], *, why: str = "",
                      ctl: Optional[Callable[[str], Optional[str]]] = None,
                      budget_s: float = TERMINATE_BUDGET_S,
                      now: Callable[[], float] = time.monotonic) -> List[int]:
    """``terminate_client <server> <pid>`` for every pid that is a client of a server of this daemon.
    Returns the pids the daemon accepted. All daemon calls together stay inside ``budget_s`` (the real
    ``control`` gets ``min(CONTROL_TIMEOUT_S, rest)`` as its timeout). Never raises."""
    want = {int(p) for p in pids}
    done: List[int] = []
    if not want:
        return done
    deadline = now() + max(0.0, float(budget_s))

    def ask(cmd: str) -> Optional[str]:
        rest = deadline - now()
        if rest <= 0:
            raise _BudgetSpent(cmd)
        if ctl is None:
            return control(cmd, timeout_s=min(CONTROL_TIMEOUT_S, rest))
        return ctl(cmd)

    clients: Dict[int, List[int]] = {}
    try:
        clients = mps_clients(ctl=ask)
        for server, cl in clients.items():
            for c in cl:
                if c in want and ask(f"terminate_client {server} {c}") is not None:
                    done.append(c)
        logger.warning("%s TERMINATE why=%s asked=%s clients=%s terminated=%s (MPS terminate_client: the client's "
                       "outstanding GPU work ends without a fault for the other clients on its card)",
                       MARK, why or "-", sorted(want), {s: sorted(c) for s, c in clients.items()}, sorted(done))
    except _BudgetSpent as e:
        logger.warning("%s TERMINATE why=%s BUDGET SPENT after %.1f s before %r: asked=%s terminated=%s (the "
                       "rest gets the caller's next step unterminated)", MARK, why or "-", float(budget_s), str(e),
                       sorted(want), sorted(done))
    except Exception as e:  # noqa: BLE001 - a teardown helper never raises
        logger.warning("%s TERMINATE why=%s raised %s: %s", MARK, why or "-", type(e).__name__, e)
    return done


def _trip_barlink(reason: str) -> None:
    from flliper.srt.distributed.device_communicators import barlink_liveness

    barlink_liveness.trip_all_abort_windows(reason)


def _device_sync() -> None:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def quiesce_self(reason: str, *, env=None, trip: Optional[Callable[[str], None]] = None,
                 sync: Optional[Callable[[], None]] = None,
                 terminate: Optional[Callable[..., List[int]]] = None,
                 pid: Optional[int] = None, bound: Optional[float] = None) -> str:
    """The dying rank's last GPU act: nothing of its own may still run on the card when it exits.

    Returns ``off`` (not an MPS client / switch off), ``drained`` (the device synchronized inside the
    bound; ``drained-error`` when the synchronize itself raised -- the context was already broken),
    ``terminated`` (the daemon ended the rest) or ``undrained`` (neither -- named, the exit goes on).
    Never raises."""
    if not armed(env):
        return "off"
    t0 = time.monotonic()
    try:
        (trip or _trip_barlink)(f"S7 rank exit: {reason}")
    except Exception as e:  # noqa: BLE001
        logger.warning("%s barlink abort trip raised %s: %s", MARK, type(e).__name__, e)
    done = threading.Event()
    err: List[BaseException] = []

    def _run() -> None:
        try:
            (sync or _device_sync)()
        except BaseException as e:  # noqa: BLE001 - recorded, the verdict names it
            err.append(e)
        finally:
            done.set()

    threading.Thread(target=_run, name="s7-mps-exit-sync", daemon=True).start()
    b = bound_s(env) if bound is None else float(bound)
    me = os.getpid() if pid is None else int(pid)
    if done.wait(b):
        verdict = "drained-error" if err else "drained"
    else:
        killed = (terminate or terminate_clients)([me], why=f"self after {b:.1f}s: {reason}")
        verdict = "terminated" if me in killed else "undrained"
    logger.error("%s verdict=%s pid=%d reason=%s waited_ms=%d bound_s=%.1f%s (an MPS client that exits with work "
                 "on the card is undefined behaviour for every client of that card -- boot 10240856 09:09:47 card 1)",
                 MARK, verdict, me, reason, int(1000 * (time.monotonic() - t0)), b,
                 (" sync_error=%s: %s" % (type(err[0]).__name__, err[0])) if err else "")
    return verdict
