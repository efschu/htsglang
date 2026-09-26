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
"""H101: the local-memory census of every Triton kernel this process loads.

THE DEATH IT ANSWERS (rc9p, 26.09., D-TP0 on the 5090). A kernel's per-thread
stack (register spills, local arrays) is backed by ONE context-wide device
allocation of ``stack x SMs x threads per SM``. When a launch needs more than
the context holds, ``cuLaunchKernel`` grows it -- an allocation at launch time
that no planner post, no torch counter and no #1028c bound sees coming. rc9p's
first launch of the QSA rows form (16, 1, 2) -- REG 128 / STACK 2320 B under
the CUDA-13 ptxas -- had to grow D-TP0 from 1024 B (255 MiB) to 2320 B
(578 MiB) inside the X-direct extend of 3585 rows, and died there:
``RuntimeError: Triton Error [CUDA]: out of memory``.

WHAT THIS DOES. The #1056 loader chokepoint (utils/triton_loader_window.py)
calls :func:`on_module_loaded` right after a cold module load, BEFORE the
kernel's first launch. The loaded function's ``CU_FUNC_ATTRIBUTE_LOCAL_SIZE_
BYTES`` is on the CompiledKernel already (Triton stores it as ``n_spills`` =
LOCAL_SIZE / 4, backends/nvidia/driver.c). Then:

1. CENSUS: the kernel's local bytes are recorded (largest per kernel name);
   :func:`census_max` is the largest stack any kernel of this process needs.
   The H15 wake (weg2/sleep_lmem.py) restores at least that -- the need stays
   booked across every sleep instead of being regrown mid-forward.
2. PRE-GROW: when the kernel needs more than the context holds, the stack
   limit is raised NOW, with ``cuCtxSetLimit(CU_LIMIT_STACK_SIZE, local)``,
   inside the cold-build window -- not later inside the launch. If the driver
   refuses (out of memory), torch's cached-but-free blocks are handed back
   (``empty_cache``) and the raise is tried once more; a second refusal is a
   NAMED line ``H101 LMEM-GROW REFUSED`` with the MiB it needed and the MiB
   the card had, so a following launch OOM is never anonymous again.
   Under stream capture nothing is set (cuCtxSetLimit is not capturable); the
   census entry is still made and the line says ``deferred=capturing``.

Every line is ``H101 LMEM-...``. Nothing here raises into the load: a failed
instrument leaves the stock behaviour (driver growth at the launch).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Optional, Tuple

import msgspec

logger = logging.getLogger(__name__)

__all__ = [
    "GrowVerdict",
    "census_max",
    "census_snapshot",
    "ensure_stack",
    "kernel_local_bytes",
    "on_module_loaded",
    "record",
]

_MIB = 1 << 20
_lock = threading.Lock()
#: kernel name -> the largest LOCAL_SIZE (bytes per thread) loaded under it.
_census: Dict[str, int] = {}


def kernel_local_bytes(kernel) -> Optional[int]:
    """LOCAL_SIZE_BYTES of a loaded Triton CompiledKernel, or None when the
    build does not expose it (``n_spills`` is LOCAL_SIZE / 4 in Triton 3.x)."""
    n = getattr(kernel, "n_spills", None)
    if n is None:
        return None
    try:
        return max(0, int(n)) * 4
    except (TypeError, ValueError):
        return None


def record(name: str, local_bytes: int) -> None:
    with _lock:
        if int(local_bytes) > _census.get(name, -1):
            _census[name] = int(local_bytes)


def census_max() -> Tuple[int, str]:
    """(largest per-thread stack of any loaded kernel, its name); (0, "") if none."""
    with _lock:
        if not _census:
            return 0, ""
        name = max(_census, key=lambda k: (_census[k], k))
        return _census[name], name


def census_snapshot() -> Dict[str, int]:
    with _lock:
        return dict(_census)


def _reset_for_test() -> None:
    with _lock:
        _census.clear()


class GrowVerdict(msgspec.Struct, frozen=True, kw_only=True):
    """What :func:`ensure_stack` did for one loaded kernel."""

    kernel: str
    local_bytes: int
    found_stack_bytes: int
    #: the context's stack after the call (== found when nothing was set)
    stack_bytes: int
    threads: int
    #: "" (covered, nothing to do) | "grown" | "grown-after-empty-cache" |
    #: "refused" | "deferred-capturing" | "unreadable"
    action: str
    free_mib: Optional[float] = None
    detail: str = ""

    def grow_mib(self) -> float:
        return max(0, self.local_bytes - self.found_stack_bytes) * self.threads / _MIB

    def line(self) -> str:
        total = self.local_bytes * self.threads / _MIB
        free = "n/a" if self.free_mib is None else f"{self.free_mib:.0f}"
        head = (
            f"H101 LMEM-GROW kernel={self.kernel} local={self.local_bytes} B "
            f"ctx_stack={self.found_stack_bytes}->{self.stack_bytes} B x {self.threads} threads "
            f"grow=+{self.grow_mib():.0f} MiB total={total:.0f} MiB driver_free={free} MiB"
        )
        if self.action == "refused":
            return head.replace("LMEM-GROW", "LMEM-GROW REFUSED", 1) + (
                f" ({self.detail}) -- the first launch of this kernel will ask the driver "
                "for the same growth inside cuLaunchKernel; an OOM there is THIS post"
            )
        if self.action == "deferred-capturing":
            return head + " deferred=capturing (cuCtxSetLimit is not capturable; the launch grows it)"
        return head + f" action={self.action} (booked before the first launch, restored at every wake)"


def ensure_stack(
    *,
    kernel: str,
    local_bytes: int,
    driver,
    threads: int,
    free_bytes: Callable[[], Optional[int]],
    empty_cache: Callable[[], None],
    capturing: Callable[[], bool],
) -> GrowVerdict:
    """Raise the context stack to ``local_bytes`` if it holds less. Never raises."""

    def _free() -> Optional[float]:
        try:
            b = free_bytes()
            return None if b is None else b / _MIB
        except Exception:  # noqa: BLE001
            return None

    try:
        found = int(driver.get_stack_bytes())
    except Exception as exc:  # noqa: BLE001
        return GrowVerdict(kernel=kernel, local_bytes=int(local_bytes), found_stack_bytes=0,
                           stack_bytes=0, threads=int(threads), action="unreadable",
                           detail=f"{type(exc).__name__}: {exc}")
    base = dict(kernel=kernel, local_bytes=int(local_bytes), found_stack_bytes=found,
                threads=int(threads))
    if int(local_bytes) <= found:
        return GrowVerdict(stack_bytes=found, action="", **base)
    try:
        if capturing():
            return GrowVerdict(stack_bytes=found, action="deferred-capturing",
                               free_mib=_free(), **base)
    except Exception:  # noqa: BLE001 -- unknown capture state: try the set
        pass
    errors = []
    for attempt, action in ((0, "grown"), (1, "grown-after-empty-cache")):
        if attempt:
            try:
                empty_cache()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"empty_cache: {type(exc).__name__}: {exc}")
        try:
            driver.set_stack_bytes(int(local_bytes))
            now = int(driver.get_stack_bytes())
            return GrowVerdict(stack_bytes=now, action=action, free_mib=_free(), **base)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
    return GrowVerdict(stack_bytes=found, action="refused", free_mib=_free(),
                       detail="; ".join(errors), **base)


def _device_threads() -> int:
    import torch

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(props.multi_processor_count) * int(props.max_threads_per_multi_processor)


def on_module_loaded(kernel) -> Optional[GrowVerdict]:
    """The chokepoint hook: census + pre-grow for one freshly loaded kernel.
    Returns the verdict when the kernel needs local memory, else None. Never
    raises."""
    try:
        local = kernel_local_bytes(kernel)
        if not local:
            return None
        name = str(getattr(kernel, "name", "?"))
        record(name, local)
        import torch

        from sglang.srt.weg2.sleep_lmem import CudaDriverStackLimit

        verdict = ensure_stack(
            kernel=name,
            local_bytes=local,
            driver=CudaDriverStackLimit(),
            threads=_device_threads(),
            free_bytes=lambda: torch.cuda.mem_get_info()[0],
            empty_cache=torch.cuda.empty_cache,
            capturing=torch.cuda.is_current_stream_capturing,
        )
        if verdict.action == "refused":
            logger.error("%s", verdict.line())
        elif verdict.action:
            logger.warning("%s", verdict.line())
        return verdict
    except Exception as exc:  # noqa: BLE001 -- an instrument never breaks a load
        logger.info("H101 LMEM-CENSUS n/a (%s: %s)", type(exc).__name__, str(exc)[:160])
        return None
