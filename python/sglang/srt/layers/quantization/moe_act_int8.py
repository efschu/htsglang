"""H88 (PLAN-H88-W4A8-1007): the ``--moe-act-int8`` / ``SGLANG_MOE_ACT_INT8`` switch of the compressed-tensors int4 MoE experts.

The switch says: the WNA16 MoE experts take int8 activations (W4A8) instead of 16-bit ones (W4A16).  Two spellings, either one
switches it on: the ServerArgs flag ``--moe-act-int8 on`` and the environment variable ``SGLANG_MOE_ACT_INT8=1`` (the
group environment ``--env-p`` / ``--env-d`` of a weg2 profile carries it to both groups).  Default off = nothing changes.

This module only READS the switch (H88-B reads it to pick the W4A8 scheme) and names the failure of a tree that has no W4A8
MoE scheme: :func:`require_w4a8_moe_scheme` is called by the MoE scheme dispatch and raises ``RuntimeError`` when the switch is
on, so a boot with the switch and without the scheme stops at the first MoE layer instead of silently running W4A16.
"""

from __future__ import annotations

from typing import Any, Optional

#: the message of the RuntimeError (a test and the boot-log reader match this text)
NO_SCHEME_MESSAGE = "MOE-ACT-INT8 requested but no W4A8 MoE scheme in this tree"

#: whether this tree has a W4A8 MoE scheme behind the switch.  H88-B (CompressedTensorsWNA16A8MoE) sets this True.
HAS_W4A8_MOE_SCHEME = True


def moe_act_int8_requested(server_args: Optional[Any] = None) -> bool:
    """True when the flag (``server_args.moe_act_int8 == "on"``) or the env (``SGLANG_MOE_ACT_INT8``) asks for int8 MoE
    activations.  ``server_args`` defaults to the process's own (``runtime_context.get_server_args``); a process without
    one (unit test, tool) reads the env alone."""
    from sglang.srt.environ import envs

    if envs.SGLANG_MOE_ACT_INT8.get():
        return True
    if server_args is None:
        try:
            from sglang.srt.runtime_context import get_server_args

            server_args = get_server_args()
        except Exception:  # noqa: BLE001 - no context in this process: the env was the only source
            server_args = None
    return str(getattr(server_args, "moe_act_int8", "off") or "off").lower() == "on"


def require_w4a8_moe_scheme(server_args: Optional[Any] = None) -> None:
    """Called where the MoE scheme is chosen for an int4 (WNA16) expert layer: raises ``RuntimeError(NO_SCHEME_MESSAGE)`` when
    the switch is on and this tree has no W4A8 MoE scheme (:data:`HAS_W4A8_MOE_SCHEME`).  Switch off: returns, no effect."""
    if moe_act_int8_requested(server_args) and not HAS_W4A8_MOE_SCHEME:
        raise RuntimeError(NO_SCHEME_MESSAGE)
