# SPDX-License-Identifier: Apache-2.0
"""FRONT-SEND-IDS-1007 shadow comparison: the ids the front counted for a
text-only request against the ids group D hands its scheduler.

The front's X-EXACT check compares token COUNTS (``X-EXACT-ERR``), never the
ids. Before the D leg may carry the front's ids (``FLLIPER_ENABLE_PDFLIP_FRONT_
SEND_IDS``), the metal has to show they are D's ids. With
``FLLIPER_LOG_PDFLIP_PROMPT_IDS_DIGEST`` the front prints ``PDFLIP FRONT-IDS`` and
group D's tokenizer manager prints, per request it tokenized::

    PDFLIP D-IDS rid=<rid> n=<tokens> sha=<16 hex> src=<text|ids|handoff>

Joined by rid: same ``n`` and ``sha`` = same ids. Log only.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

import numpy as np

from flliper.srt.environ import envs

logger = logging.getLogger(__name__)


def ids_digest(ids: Sequence[int]) -> str:
    """sha1 of the ids as int32, 16 hex digits."""
    return hashlib.sha1(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()[:16]


def d_digest_armed(*, group: str) -> bool:
    return envs.FLLIPER_LOG_PDFLIP_PROMPT_IDS_DIGEST.get() and group.strip().upper() == "D"


def log_d_ids(*, rid: str, ids: Sequence[int], src: str) -> None:
    logger.info("PDFLIP D-IDS rid=%s n=%d sha=%s src=%s", rid, len(ids), ids_digest(ids), src)
