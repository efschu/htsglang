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
"""How large the flashinfer float workspace ends up, in ONE place.

The size is not simply ``SGLANG_FLASHINFER_WORKSPACE_SIZE``. The backend
REWRITES that variable during ``__init__``, twice and in a fixed order, so the
value a reader sees depends on whether it read before or after the backend was
built:

1. a listed Qwen/MiMo architecture raises it to 512 MiB;
2. ``--enable-deterministic-inference`` then raises it to 2048 MiB, overriding
   step 1 -- the deterministic kernels need the larger scratch and they do not
   care which architecture asked for 512.

Anything that must know the size BEFORE the backend exists -- the VRAM ledger
sizing a card, a planner predicting capacity -- cannot read the env var and be
right. It has to apply the same rule. This module is that rule, and the backend
calls it too, so there is one implementation rather than two that agree until
someone edits one of them.

THE BUG THIS MODULE CLOSES. The ledger's attention-workspace term read the raw
env var at build time and therefore did not move when
``enable_deterministic_inference`` moved -- it charged 384 MiB for a
configuration that allocates 2048 MiB. On the reference deterministic boot that
is 1664 MiB the ledger never charged, against 1649 MiB measured as actually
consumed between pool end and capture begin. A MODELED term that does not
respond to a configuration input it depends on is precisely what this project's
ledger contract forbids.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

__all__ = [
    "WORKSPACE_ARCH_MIB",
    "WORKSPACE_DETERMINISTIC_MIB",
    "HIGH_WORKSPACE_ARCHITECTURES",
    "resolve_flashinfer_workspace_bytes",
    "resolve_flashinfer_workspace_mib",
    "describe_flashinfer_workspace",
]

#: Architectures the backend raises the workspace to 512 MiB for. Kept as a
#: frozenset so a membership test cannot silently become a substring match --
#: "Qwen3ForCausalLM" must not match "Qwen3_5ForConditionalGeneration", and an
#: `in` over a string would have done exactly that.
HIGH_WORKSPACE_ARCHITECTURES = frozenset(
    {
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "MiMoForCausalLM",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
    }
)

WORKSPACE_ARCH_MIB = 512
WORKSPACE_DETERMINISTIC_MIB = 2048


def _default_bytes() -> int:
    from sglang.srt.environ import envs

    return int(envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get())


def resolve_flashinfer_workspace_bytes(
    *,
    enable_deterministic_inference: bool = False,
    architectures: Optional[Iterable[str]] = None,
    default_bytes: Optional[int] = None,
) -> int:
    """The workspace size the backend will end up with, in bytes.

    Precedence follows the backend's assignment ORDER, which is the part that
    is easy to get backwards: the architecture bump happens first and the
    deterministic bump happens second, so deterministic WINS over a listed
    architecture rather than the other way round.

    ``default_bytes`` overrides the env default; it exists so a test can state
    the default it is reasoning about instead of depending on the ambient
    environment.
    """
    if enable_deterministic_inference:
        return WORKSPACE_DETERMINISTIC_MIB * 1024 * 1024
    archs = set(architectures or ())
    if archs & HIGH_WORKSPACE_ARCHITECTURES:
        return WORKSPACE_ARCH_MIB * 1024 * 1024
    return _default_bytes() if default_bytes is None else int(default_bytes)


def resolve_flashinfer_workspace_mib(
    *,
    enable_deterministic_inference: bool = False,
    architectures: Optional[Iterable[str]] = None,
    default_bytes: Optional[int] = None,
) -> int:
    return resolve_flashinfer_workspace_bytes(
        enable_deterministic_inference=enable_deterministic_inference,
        architectures=architectures,
        default_bytes=default_bytes,
    ) // (1 << 20)


def describe_flashinfer_workspace(
    *,
    enable_deterministic_inference: bool = False,
    architectures: Optional[Sequence[str]] = None,
    default_bytes: Optional[int] = None,
) -> str:
    """Why the workspace is the size it is, for a ledger row or a boot log."""
    mib = resolve_flashinfer_workspace_mib(
        enable_deterministic_inference=enable_deterministic_inference,
        architectures=architectures,
        default_bytes=default_bytes,
    )
    if enable_deterministic_inference:
        why = (
            "raised by --enable-deterministic-inference, which overrides the "
            "architecture bump below"
        )
    elif set(architectures or ()) & HIGH_WORKSPACE_ARCHITECTURES:
        matched = sorted(set(architectures or ()) & HIGH_WORKSPACE_ARCHITECTURES)
        why = f"raised for architecture(s) {matched}"
    else:
        why = (
            "SGLANG_FLASHINFER_WORKSPACE_SIZE default; this architecture is "
            "not on the high-workspace list and deterministic inference is off"
        )
    return f"{mib} MiB/rank ({why})"
