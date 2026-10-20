# SPDX-License-Identifier: Apache-2.0
"""D-LEAD-MS-1007: when a request reached this rank's scheduler, and when its
first extend began -- in milliseconds, on this rank's own wall clock.

Voranalyse 07.10. (NF): a 150k-token agent turn spends 2.1-2.4 s between its
arrival on D and its extend; 0.42-0.55 s of it in the scheduler. The D log
stamps lines to the second, so that share was read off a py-spy, never off
the log. With ``FLLIPER_LOG_PDFLIP_D_LEAD_MS`` every rank of group D prints, at a
request's first extend, one line::

    PDFLIP D-LEAD-MS rid=<rid> recv_ms=<epoch ms> extend_ms=<epoch ms>
        recv_to_extend_ms=<ms> skip=<0|1>

``recv_ms`` is the start of ``handle_generate_request`` on the rank,
``extend_ms`` the first ``_run_batch_forward`` of an extend batch carrying
the rid (``skip=1``: a batch that adopts a tail and runs no forward).
Joined with the front's ``D-ADMIT`` line by rid and wall clock it splits the
lead into front->scheduler and scheduler->extend.

Only a log line: no collective, no state any decision reads. Off, or on any
group but D, nothing is built and no line is printed.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Callable, Iterable, Optional

from flliper.srt.environ import envs

logger = logging.getLogger(__name__)

#: rids awaiting their first extend; a rid refused or aborted before it is
#: dropped oldest-first
PENDING_CAP = 4096


def build_d_lead_probe(*, group: str) -> Optional["DLeadProbe"]:
    if not envs.FLLIPER_LOG_PDFLIP_D_LEAD_MS.get() or group.strip().upper() != "D":
        return None
    return DLeadProbe()


class DLeadProbe:
    def __init__(self, *, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._recv_s: "collections.OrderedDict[str, float]" = collections.OrderedDict()

    def note_recv(self, *, rid: str) -> None:
        self._recv_s.pop(rid, None)
        self._recv_s[rid] = self._clock()
        while len(self._recv_s) > PENDING_CAP:
            self._recv_s.popitem(last=False)

    def note_extend(self, *, rids: Iterable[str], skip: bool) -> None:
        now = self._clock()
        for rid in rids:
            recv = self._recv_s.pop(rid, None)
            if recv is None:
                continue
            logger.info(
                "PDFLIP D-LEAD-MS rid=%s recv_ms=%d extend_ms=%d recv_to_extend_ms=%.1f skip=%d",
                rid, int(recv * 1000), int(now * 1000), (now - recv) * 1000.0, int(skip),
            )
