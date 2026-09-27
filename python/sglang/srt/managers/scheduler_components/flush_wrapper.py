import logging
import os
import time
from typing import Callable, Optional, Tuple

from sglang.srt.managers.io_struct import FlushCacheReqInput, FlushCacheReqOutput
from sglang.srt.managers.scheduler_components.ipc_channels import (
    SchedulerIpcChannels,
)


#: FD (27B rc12o27 b1, 13:11:55Z): a /flush_cache that reaches a group whose
#: kv_cache is released (``Scheduler.weg2_dormant``, set right after the pause
#: on every rank) is refused by name before it touches a pool: flush_cache ->
#: ReqToTokenPool.clear -> ``req_to_token.zero_()`` / MambaPool.reset_state
#: write into unmapped pages -> illegal memory access (PP1/PP2 died so). The
#: flag is set by the release RPC, which every rank runs in the same order as
#: the flush, so the refusal is the same on every rank. The sleep leg's own
#: flush (before the pause) and the wake-side restore flush (after the resume)
#: call flush_cache directly and never pass here. ``0`` = no refusal.
ENV_DORMANT_REFUSE = "SGLANG_WEG2_FLUSH_DORMANT_REFUSE"


def dormant_refuse_enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV_DORMANT_REFUSE, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


class SchedulerFlushWrapper:
    def __init__(
        self,
        *,
        flush_cache: Callable[..., bool],
        is_fully_idle: Callable[[], bool],
        ipc_channels: SchedulerIpcChannels,
        is_dormant: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._flush_cache = flush_cache
        self._is_fully_idle = is_fully_idle
        self._ipc_channels = ipc_channels
        self._is_dormant = is_dormant
        self._pending: Optional[Tuple[FlushCacheReqInput, float]] = None
        self._dormant_refusals = 0

    def _dormant_refusal(self) -> Optional[FlushCacheReqOutput]:
        """FD: the named refusal on a released group, or None."""
        if self._is_dormant is None or not dormant_refuse_enabled():
            return None
        try:
            dormant = bool(self._is_dormant())
        except Exception:  # noqa: BLE001 -- an unreadable flag refuses nothing
            return None
        if not dormant:
            return None
        self._dormant_refusals += 1
        n = self._dormant_refusals
        if n <= 8 or (n & (n - 1)) == 0:
            logging.warning(
                "WEG2-FLUSH-DORMANT refused (n=%d): /flush_cache reached a group whose kv_cache "
                "is released (weg2_dormant); a flush now would zero unmapped pools "
                "(req_to_token / mamba state) -> illegal memory access (rc12o27 b1). Nothing touched.",
                n,
            )
        return FlushCacheReqOutput(
            success=False,
            message="W25 Weg2DormantRefused: flush_cache on a released (dormant) group",
        )

    def handle(self, recv_req: FlushCacheReqInput) -> Optional[FlushCacheReqOutput]:
        refused = self._dormant_refusal()
        if refused is not None:
            return refused
        if self._pending is not None:
            return FlushCacheReqOutput(
                success=False,
                message="Another flush_cache is already in progress.",
            )

        timeout_s = float(recv_req.timeout_s or 0.0)
        if timeout_s <= 0.0:
            # fnFL2x105: the immediate RPC reaches every TP rank in the same
            # pass, so its idle verdict may be (and is) a TP-group reduction.
            # The deferred path below flushes on a rank-local idle test and
            # must stay rank-local.
            return FlushCacheReqOutput(
                success=self._flush_cache(tp_group_verdict=True)
            )

        if self._is_fully_idle():
            return FlushCacheReqOutput(success=self._flush_cache())

        self._pending = (recv_req, time.monotonic() + timeout_s)
        return None

    def check_pending(self) -> None:
        if self._pending is None:
            return

        pending_req, deadline = self._pending

        refused = self._dormant_refusal()
        if refused is not None:
            self._pending = None
            self._ipc_channels.send_to_tokenizer.send_output(refused, pending_req)
            return

        if self._is_fully_idle():
            success = self._flush_cache()
            self._pending = None
            self._ipc_channels.send_to_tokenizer.send_output(
                FlushCacheReqOutput(success=success), pending_req
            )
            return

        if time.monotonic() >= deadline:
            logging.warning(
                "Deferred flush_cache timed out while waiting for idle state."
            )
            self._pending = None
            self._ipc_channels.send_to_tokenizer.send_output(
                FlushCacheReqOutput(
                    success=False, message="Timed out waiting for idle state."
                ),
                pending_req,
            )
