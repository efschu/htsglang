"""Punkt 3 of the user order of 18.09.2026 ("punkt 1 - 3 sofort umsetzen,
verdrahten und als standard setzen"): on Weg 2 group D a memory-pressure
retract RETAINS the request's computed span in the tree instead of throwing
it away.

Upstream's ``retract_decode`` releases with ``is_insert=False`` -- a
deliberate discard, the request re-prefills from zero. On group D that
discard costs a whole 100k prefill on P plus a flip. With retention the
span (KV rows, the Mamba/GDN checkpoint of the node, the DFlash draft rows
carried with the KV) is inserted into the tree as an EVICTABLE path: the
next decode allocation demotes it to the host arena (write-back), and the
re-admission finds it by prefix match and loads it back -- the same
loadback path the post-flip re-admission uses. No re-prefill.

Everywhere else (group P, non-Weg-2 boots) upstream's discard stays.
Standard on D; ``SGLANG_WEG2_RETRACT_RETAIN=0`` turns it off for an A/B.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional

ENV = "SGLANG_WEG2_RETRACT_RETAIN"
#: mirrors ``sglang.srt.managers.corridor_guard.GROUP_ENV`` (asserted by the
#: unit test) -- spelled here so this module imports nothing heavy.
GROUP_ENV = "SGLANG_WEG2_GROUP"
_OFF = ("0", "false", "no", "off")


def retract_retains(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when a memory-pressure retract must keep the span in the tree."""
    env = os.environ if env is None else env
    if str(env.get(ENV, "1")).strip().lower() in _OFF:
        return False
    return str(env.get(GROUP_ENV, "")).strip().upper() == "D"
