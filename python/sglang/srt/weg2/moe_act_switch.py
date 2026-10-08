"""H88-D: the ONE predicate "is the W4A8 switch of the MoE experts on?", shared by the launcher identity
(``launcher.l3_persist_identity``) and the rank identity (``hicache_storage.compute_model_identity_hash``).

It is stdlib-only on purpose (the launcher must stay torch-free) and it mirrors the RUNTIME reading of the switch, spelling by
spelling, because the dangerous direction of any divergence is "runtime W4A8, identity default": W4A8 pages would land in the
store of the W4A16 identity and nothing (W57 / W165) would notice.

* env ``SGLANG_MOE_ACT_INT8``  -> ``environ.EnvBool.parse``: ``true``/``1``/``yes``/``y`` in any case, nothing else (no strip;
  an unparsable value falls back to the default OFF at runtime, so it is OFF here too).
* flag ``--moe-act-int8``      -> ``ServerArgs.moe_act_int8`` is ``Literal["on", "off"]``; the runtime reads ``== "on"``.

A Python ``True`` (a caller handing over an already-parsed boolean) is on. Everything else, ``None`` included, is off.
"""
from __future__ import annotations

from typing import Any

MOE_ACT_INT8_ENV = "SGLANG_MOE_ACT_INT8"
MOE_ACT_INT8_FLAG = "--moe-act-int8"

#: byte for byte the true-set of ``sglang.srt.environ.EnvBool.parse`` (a test pins the two together)
ENV_TRUE = ("true", "1", "yes", "y")
FLAG_ON = "on"


def env_value_on(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.lower() in ENV_TRUE


def flag_value_on(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.lower() == FLAG_ON
