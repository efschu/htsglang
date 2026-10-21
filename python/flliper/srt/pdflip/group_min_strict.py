# SPDX-License-Identifier: Apache-2.0
"""NF-RANG-AUDIT 1014, Fix 3 (B2): a group verdict that cannot be taken is a named stop.

``Scheduler._pdflip_group_min_flags`` / ``_pdflip_group_min_ints`` are the ONE verdict of
every park / settle / displace / capacity decision on group D (#1471e, #1479b): each
rank proposes its local flags, the element-wise MIN over the TP cpu group is what
every rank acts on. Their error arm answered an exception from the collective with
the LOCAL flags ("local flags used", INFO) -- the rank then acts on its own view while
its peers act on theirs, which is the forbidden silent alignment (raenge-nie-uneins:
"nie stilles Angleichen"), and nothing names it. A collective that fails (gloo
timeout, a vector length that differs because the replicated list already split) is
detected divergence; the only legal answer is a named stop of the group.

Switch ``FLLIPER_PDFLIP_GROUP_MIN_STRICT`` (default on); ``0`` = the local-flags fallback
of before. The no-group case (tp_size 1, no cpu group, torch.distributed not
initialised) is unchanged: there is nobody to disagree with, the local flags ARE the
verdict.
"""
from __future__ import annotations

import os

ENV = "FLLIPER_PDFLIP_GROUP_MIN_STRICT"
MARK = "PDFLIP GROUP-VERDICT FAILED"


# 27B DUAL (port fix 21.10., metal 00:39:28Z dual NVFP4 PP1 AgreedRoomShort "LB-VERDICT ROW-ROOM SHORT"): an NF-only P-group
# PP protocol; the 27B dual layout ran without it. Under FLLIPER_PDFLIP_DUAL_LAYOUT=1 (rank-uniform env) an UNSET switch is OFF.
def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    if not raw:
        return (e.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() != "1"
    return raw not in ("0", "false", "no", "off")


class PdFlipGroupVerdictFailed(RuntimeError):
    """The TP group's MIN verdict could not be taken on this rank: named stop."""


def failed(what: str, n: int, exc: BaseException) -> PdFlipGroupVerdictFailed:
    return PdFlipGroupVerdictFailed(
        f"{MARK} {what} n={n}: the group MIN over the TP cpu group raised {type(exc).__name__}: {exc} -- "
        "acting on this rank's local flags would put it in a state its peers are not in (the forbidden silent "
        "alignment); the group stops here by name (raenge-nie-uneins)"
    )
