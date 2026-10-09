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

#: H88 (planer decision 1008): the value the switch writes into BOTH L3 identities -- the launcher field
#: ``moe_act`` (``launcher.l3_persist_identity``) and the rank part ``moe_act=<value>``
#: (``hicache_storage.compute_model_identity_hash``). It names the activation precision AND the scale
#: encoding of the W4A8 experts (``s16``: the int16 band-encoded weight scales of the SCALEFIX layout), so
#: a store written by the earlier ``moe_act=int8`` form (rank identity 40676341cb5764ad, other scale
#: encoding) is never read back as this one. ONE constant for both sides: the two identities cannot drift
#: apart. Only ever used while the switch is on (H88-D rule b: off adds nothing, the default identity stays).
IDENTITY_VALUE = "int8;s16"

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
