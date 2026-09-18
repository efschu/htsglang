"""xsn338 (18.09.2026): P publishes a finished request's nodes AT ITS FINISH.

The store publish ran only in the PP loop's bubbles (weg2_bubble_publish) and
in the flush; a phase of back-to-back prefills has no bubble, so the pages of a
request that finished at 15:31:35 reached the arena at 15:32:12 -- with the
phase's last requests only at the sleep flush. D's dormant re-reads asked in
that gap and answered zero, and the wake paid the whole look-up burst.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional


def publish_at_retain_on(env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    if str(env.get("SGLANG_WEG2_PUBLISH_AT_RETAIN", "1")).strip().lower() in ("0", "false", "no", "off"):
        return False
    return str(env.get("SGLANG_WEG2_GROUP", "")).strip().upper() == "P"


def max_issue(env: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if env is None else env
    try:
        return max(1, int(env.get("SGLANG_WEG2_PUBLISH_AT_RETAIN_MAX", "64")))
    except ValueError:
        return 64
