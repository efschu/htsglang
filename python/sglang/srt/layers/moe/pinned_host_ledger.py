# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cross-process publication of the pinned HOST bytes each rank is holding.

Why a shared file and not just the in-process ledger (#534): the reader is the
GGUF stream trim (``model_loader/gguf_shards.ProgressCoupledTrim``), and the
quantity it has to compare against is ``memory.current`` of the **cgroup** --
which spans every rank process, not just this one. Window 2026-08-04 measured
the offload ledger reporting 20.78 + 14.44 + 14.44 = 49.66 GiB of pinned pool
across a TP=3 boot; a rank that only knew its own 14.44 GiB would correct the
trim's budget by less than a third of the real unreclaimable floor and the
defect would survive the fix. See
``docs/dev/ANALYSE_478_RESULT_q3kxl_refused.md`` §"Why the stream-trim could
not save it".

Contract, deliberately narrow:

* one small file per publishing PID, holding a single decimal byte count;
* the reader sums the files whose PID is still alive and skips (and tries to
  unlink) the rest, so a crashed rank cannot inflate the floor forever;
* every operation is best-effort. This feeds an OPTIMISATION -- a load must
  never fail because ``/dev/shm`` is full, read-only or absent -- so a failure
  returns ``None``/does nothing and the caller keeps its previous behaviour.

``None`` from :func:`total_pinned_bytes` means "no publisher has been seen",
which is a different statement from 0 bytes and the caller keeps the
difference (the #218 provenance rule, same as ``observability/spill_tiers``).

Note on the unit: the figure published here is the *cumulative* staged pinned
byte count from ``StreamingStagingLedger.pinned_bytes``. That is "bytes ever
staged", not "bytes held", and the two differ once a pool is released --
but the only reader is the trim, which runs exclusively DURING weight load,
where nothing has been released yet and the two are the same number. Any
future reader outside the load window must use
``observability/spill_tiers.expert_host_bytes`` instead.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ledger_dir",
    "publish_pinned_bytes",
    "total_pinned_bytes",
    "clear_pinned_bytes",
]

_DEFAULT_DIR = "/dev/shm/sglang-pinned-host"
_ENV_DIR = "SGLANG_PINNED_HOST_LEDGER_DIR"

_WARNED = False


def _warn_once(what: str, err: BaseException) -> None:
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    logger.debug("pinned-host ledger: %s failed (%r); continuing without it", what, err)


def ledger_dir() -> str:
    """Directory holding one file per publishing process."""
    return os.environ.get(_ENV_DIR) or _DEFAULT_DIR


def _own_path() -> str:
    return os.path.join(ledger_dir(), str(os.getpid()))


def publish_pinned_bytes(nbytes: int) -> None:
    """Publish this process's pinned host byte count. Best-effort, never raises.

    Called at MoE layer boundaries during weight load, which is both the
    cadence at which the figure changes and the cadence the trim reads at.
    One ``write`` of at most 20 bytes to tmpfs per layer.
    """
    path = _own_path()
    try:
        os.makedirs(ledger_dir(), exist_ok=True)
        # Write-then-rename would be tidier, but the payload is a single
        # short decimal integer written with one write(2) -- a concurrent
        # reader either sees the old contents or the new ones, never a mix.
        with open(path, "w") as fh:
            fh.write(str(int(nbytes)))
    except OSError as err:
        _warn_once(f"publish to {path}", err)


def _pid_alive(pid: int) -> bool:
    return os.path.isdir(f"/proc/{pid}")


def total_pinned_bytes() -> Optional[int]:
    """Summed pinned host bytes over every LIVE publisher, or ``None``.

    ``None`` = nothing has ever published here, i.e. this host has no expert
    offload pinned pool that this mechanism knows about. The caller must not
    read that as zero: it is the state in which the caller's pre-existing
    behaviour is the right behaviour.
    """
    directory = ledger_dir()
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    total = 0
    seen = False
    for name in names:
        try:
            pid = int(name)
        except ValueError:
            continue
        path = os.path.join(directory, name)
        if not _pid_alive(pid):
            # A rank that died mid-load must not keep inflating the floor.
            try:
                os.unlink(path)
            except OSError:
                pass
            continue
        try:
            with open(path) as fh:
                total += int(fh.read().strip() or 0)
            seen = True
        except (OSError, ValueError):
            continue
    return total if seen else None


def clear_pinned_bytes() -> None:
    """Drop this process's entry (tests, and a second model load in one process)."""
    try:
        os.unlink(_own_path())
    except OSError:
        pass
