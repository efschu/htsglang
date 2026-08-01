"""Child process for the #363 shutdown-path tests. Drives the REAL paths."""

import os
import sys
import time

sys.path.insert(0, os.environ["REGIME_PY"])

from sglang.srt.managers.regime_runtime import (  # noqa: E402
    MODE_OBSERVE,
    RegimeObserver,
    close_regime_trace,
    install_trace_shutdown_hook,
)

path = sys.argv[1]
how = sys.argv[2]

obs = RegimeObserver(
    consensus_interval=2, tp_size=1, mode=MODE_OBSERVE, trace_path=path
)
install_trace_shutdown_hook(obs)
for _ in range(6):
    obs.on_round(phase="prefill", held_tokens=0, capacity_tokens=453_632, running_bs=1)

if how == "sigterm":
    # Signal ourselves the way a supervisor would. Nothing here closes the
    # trace: only the installed handler can.
    os.kill(os.getpid(), 15)
    time.sleep(10)  # if the handler does not chain to death, the test times out
elif how == "exit":
    sys.exit(0)  # atexit path
elif how == "finally":
    # The shape of run_scheduler_process: an exception, and a finally that
    # calls the real helper against a duck-typed scheduler.
    class _Sched:
        regime_observer = obs

    try:
        raise RuntimeError("scheduler hit an exception")
    except Exception:
        pass
    finally:
        close_regime_trace(_Sched())
    os._exit(0)
elif how == "keyboardinterrupt":

    class _Sched2:
        regime_observer = obs

    try:
        raise KeyboardInterrupt
    except BaseException:
        pass
    finally:
        close_regime_trace(_Sched2())
    os._exit(0)
