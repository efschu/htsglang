"""#673: destroy the process groups before the interpreter does.

THE DEFECT. A scheduler leaves its event loop on ``ShutdownReq`` and the
``finally`` in ``run_scheduler_process`` releases the trace, the FPM publisher
and the host resources -- and nothing else. The distributed environment is
never torn down: ``cleanup_dist_env_and_memory`` /
``destroy_distributed_environment`` are DEFINED in
``distributed/parallel_state.py`` and, in the whole of ``python/sglang/srt``,
CALLED BY NOBODY. torch says so itself in the shutdown log of every boot:

    ProcessGroupNCCL.cpp:1575  WARNING: destroy_process_group() was not called
                               before program exit

WHY THAT ABORTS. ``ProcessGroupNCCL`` runs C++ threads of its own -- the
watchdog and ``HeartbeatMonitor::runLoop()``, both visible in the #673
specimens. They are joined by the process group's DESTRUCTOR. With the group
never destroyed, those ``std::thread`` objects are still joinable when the
process tears down, and destroying a joinable ``std::thread`` calls
``std::terminate`` -- whose message is exactly the one in the specimens,
"terminate called without an active exception", with no active exception
because there never was one. It fires after a CLEAN drain because it is not a
work failure at all: it is the exit itself.

The specimens agree on the shape. The abort thread has ``<no Python frame>``
(C++, not Python); in two of the three the Python main thread is provably
elsewhere, asleep in the SIGQUIT handler's 5-second settle; and the sibling
ranks only notice afterwards, reporting TCPStore reads that return 0 bytes
because the store owner vanished mid-handshake.

WHY THIS IS OFF BY DEFAULT, and it is not timidity. ``GroupCoordinator.destroy``
closes ``barlink_comm`` before it destroys the groups -- barlink owns a POSIX
shm segment and the device-mapped abort word that spinning kernels read. Task
#722 (barlink abort-poll x flip) is live on exactly that machinery and belongs
to another lane. Turning this on changes WHEN that memory is unlinked relative
to those kernels, so it ships gated: the default path stays byte-identical, and
the lane that owns barlink decides when to arm it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#673 teardown]"


def distributed_teardown_enabled(server_args: Any) -> bool:
    """True when the scheduler should destroy its groups on the way out.

    Default False. See the module docstring: the destroy path runs barlink's
    ``close()``, which is #722's machinery.
    """
    return bool(getattr(server_args, "scheduler_distributed_teardown", False))


def release_distributed(scheduler: Any, *, graceful: bool) -> Optional[str]:
    """Destroy this process's collectives. Returns what it did, or None.

    Three properties, each load-bearing:

    * GRACEFUL PATH ONLY. On the exception path the GPU may already be wedged,
      and destroying a process group synchronises -- the same reason
      ``release_host_resources`` is already guarded that way. A teardown that
      hangs is worse than the abort it prevents, because the abort at least
      ends the process.
    * NEVER RAISES. This runs in a ``finally`` during shutdown; an exception
      here would replace a clean exit with a traceback, and on the exception
      path it would mask the original failure.
    * IDEMPOTENT. Called twice (a retry, a test, a future second call site) it
      must be a no-op the second time rather than an error about groups that
      are already gone.
    """
    if not graceful:
        return None
    server_args = getattr(scheduler, "server_args", None)
    if not distributed_teardown_enabled(server_args):
        return None
    try:
        from sglang.srt.distributed import parallel_state
    except Exception as e:  # pragma: no cover - import shape, not behaviour
        logger.warning("%s parallel_state unavailable: %s", LOG_PREFIX, e)
        return None

    done = []
    try:
        parallel_state.destroy_model_parallel()
        done.append("model_parallel")
    except Exception as e:
        logger.warning("%s destroy_model_parallel failed: %s", LOG_PREFIX, e)
    try:
        parallel_state.destroy_distributed_environment()
        done.append("distributed_environment")
    except Exception as e:
        logger.warning("%s destroy_distributed_environment failed: %s", LOG_PREFIX, e)

    if done:
        logger.info(
            "%s destroyed %s before exit; the NCCL watchdog and heartbeat "
            "threads are joined by that destructor, so they are no longer "
            "joinable when the process tears down.",
            LOG_PREFIX,
            " + ".join(done),
        )
    return " + ".join(done) if done else None
