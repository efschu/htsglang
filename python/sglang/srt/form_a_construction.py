# SPDX-License-Identifier: Apache-2.0
"""Form A slice 4a: what a WORKER rank builds, and what it must not.

Slice 5 stopped a worker READING the dense tensors (the loader veto in
`models/qwen4_exp.py weight_name_needed`). That saved checkpoint I/O and no
VRAM at all, because `create_weights` had already allocated the parameters
at CONSTRUCTION time. This module is the other half: the decision of which
submodules a worker constructs, and a placeholder for the ones it does not.

TWO THINGS, deliberately separated:

  * `worker_builds(kind)` -- the DECISION, a pure predicate over a small
    closed vocabulary of module kinds. It is the thing the census test can
    check without a GPU, a model, or a checkpoint.
  * `HostOnlyModule` -- the PLACEHOLDER that stands where the skipped module
    was. It holds no parameters, so the VRAM is genuinely gone, and its
    forward RAISES by name.

Why a raising placeholder rather than `None`: today every rank still runs
the same model forward, because the host-centric decode path (the worker
computing only its experts) is a later slice. So a worker that skips
construction will reach a dense module it does not have. With `None` that is
an `AttributeError` on line whatever, naming nothing. With this it is
`FormAHostOnlyModuleUsed`, naming the module, the rank and the slice that
has to land next. The placeholder is not a workaround for that ordering --
it is how the ordering announces itself.
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from sglang.srt.rank_role import RankRoleError, this_rank_is_form_a_worker

__all__ = [
    "FormAHostOnlyModuleUsed",
    "HOST_ONLY_KINDS",
    "WORKER_KINDS",
    "MODULE_KINDS",
    "HostOnlyModule",
    "worker_builds",
    "expected_census_categories",
    "skip_on_worker",
]


class FormAHostOnlyModuleUsed(RuntimeError):
    """A worker rank's forward reached a module only the host builds."""


#: Everything the attention host owns alone. These names are the module
#: KINDS, not class names, so the list survives a refactor of the model.
HOST_ONLY_KINDS: Tuple[str, ...] = (
    "self_attn",  # incl. the QSA indexer
    "linear_attn",  # GDN
    "o_proj",
    "hyper_connection",  # the mixer -- NOT sharded, so a worker pays it in full
    "embed_tokens",
    "lm_head",
    "ple",
    "norm",
    "moe_gate",  # the router: the host picks the experts and sends the ids
    "shared_expert",  # dense, runs for every token
    "draft",  # the MTP draft is unsharded on the host
    "vision",
)

#: What a worker builds. Exactly one thing.
WORKER_KINDS: Tuple[str, ...] = ("experts",)

MODULE_KINDS: Tuple[str, ...] = WORKER_KINDS + HOST_ONLY_KINDS


def worker_builds(kind: str) -> bool:
    """Does a Form A WORKER construct this kind of module?

    Refuses an unknown kind rather than defaulting: a typo that silently
    answered "host-only" would delete a module from the host, and a typo
    that silently answered "build" would put one back on a worker. Neither
    failure announces itself.
    """
    if kind not in MODULE_KINDS:
        raise RankRoleError(
            f"unknown module kind {kind!r}; Form A knows {list(MODULE_KINDS)}. "
            "Add it to WORKER_KINDS or HOST_ONLY_KINDS -- do not let an "
            "unlisted kind take a default."
        )
    return kind in WORKER_KINDS


def expected_census_categories(role: str) -> Tuple[str, ...]:
    """The `[vram-census]` keys a rank of this role may report.

    This is the ACCEPTANCE CRITERION of slice 4a, written where a test can
    read it: after the construction skip, a worker's census line must carry
    'experts' and nothing else. Today it also carries hyper_connection,
    linear_attn, embed_tokens, lm_head, moe_gate, shared_expert and ple --
    measured on boot fn8ah, 1.84 GiB per worker.
    """
    if role == "worker":
        return ("experts",)
    if role == "host":
        return ("experts",) + HOST_ONLY_KINDS
    raise RankRoleError(f"unknown role {role!r}")


class HostOnlyModule(nn.Module):
    """Stands where a host-only module would be on a worker rank.

    Holds no parameters and no buffers -- that is the entire point, and the
    census test checks it. Any use raises.
    """

    def __init__(self, kind: str, prefix: str = "") -> None:
        super().__init__()
        self.kind = kind
        self.prefix = prefix

    def _refuse(self, how: str):
        raise FormAHostOnlyModuleUsed(
            f"a Form A WORKER rank reached {how} of the host-only module "
            f"{self.kind!r} ({self.prefix or 'unnamed'}). A worker holds "
            "experts and nothing else, so this module was never built. "
            "Reaching it means the worker is still running the DENSE "
            "forward -- the host-centric decode path (the worker computing "
            "only its own experts, driven by the MoE broadcast/reduce) is "
            "the slice that has to land next."
        )

    def forward(self, *args, **kwargs):
        self._refuse("forward()")

    def __getattr__(self, item):
        # nn.Module resolves real attributes first; this only fires for the
        # ones a caller expected the skipped module to have.
        if item.startswith("_") or item in ("kind", "prefix"):
            raise AttributeError(item)
        self._refuse(f"attribute {item!r}")


def skip_on_worker(kind: str, prefix: str = ""):
    """`None` when this rank builds the module, a placeholder when it does
    not. The one-line call site:

        if (ph := skip_on_worker("ple", prefix)) is not None:
            self.ple = ph
        else:
            self.ple = Qwen4ExpPLELayer(...)

    Inert on every classic boot: no role plan installed means
    `this_rank_is_form_a_worker()` is False and this returns None.
    """
    if not this_rank_is_form_a_worker():
        return None
    if worker_builds(kind):
        return None
    return HostOnlyModule(kind, prefix)
