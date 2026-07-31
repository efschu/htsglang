# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``Class2DiffusionAdapter``: estimate-only, by design (#333 §10 M1).

M1 ships the Class-2 adapter as an estimator and nothing else. It can be
registered, it can be planned against, it takes part in "does this rig fit
these engines", and it refuses to launch. Promoting it to a launching adapter
is M3, which needs process-group management, a ZMQ client and the two
``multimodal_gen`` seam changes of §5.4 -- none of which belong in a registry
milestone.

An estimator is not a placeholder. The whole value of §7.4's "validate without
booting" is that a spec can be judged before a GPU window is booked, and a
diffusion tenant is precisely the one whose footprint an operator cannot guess.
The estimate follows §5.2's posts.
"""

from __future__ import annotations

from typing import Any, Mapping

from sglang.srt.registry.adapter import (
    AdapterContext,
    AdapterError,
    EstimateError,
    Health,
    register_adapter,
)
from sglang.srt.registry.ledger import MIB
from sglang.srt.registry.spec import (
    EngineClass,
    EngineSpec,
    ResidencyState,
    ResourceProfile,
)

ADAPTER_NAME = "class2_diffusion"

#: §5.2 memory posts for a diffusion tenant. Every one of them must be
#: declared: an activation peak that scales with resolution cannot be guessed
#: by the registry, and guessing it is how a card goes over the corridor at the
#: first full-resolution request rather than at plan time.
REQUIRED_POSTS = (
    "weights_bytes",
    "activation_peak_bytes",
    "latent_bytes",
    "text_encoder_bytes",
    "vae_bytes",
    "ctx_overhead_bytes",
)


class Class2DiffusionAdapter:
    """Costs a diffusion tenant. Does not run one."""

    klass = int(EngineClass.DIFFUSION)

    def __init__(self, spec: EngineSpec, context: AdapterContext) -> None:
        self.spec = spec
        self.context = context
        self.launch: Mapping[str, Any] = dict(spec.launch)
        posts = self.launch.get("posts_mib")
        if not posts:
            raise EstimateError(
                f"engine {spec.engine_id!r}: a Class-2 spec must declare "
                "launch.posts_mib with the §5.2 posts "
                f"{list(REQUIRED_POSTS)}; the registry does not model diffusion "
                "and will not invent an activation peak"
            )
        missing = [p for p in REQUIRED_POSTS if p not in posts]
        if missing:
            raise EstimateError(
                f"engine {spec.engine_id!r}: launch.posts_mib is missing "
                f"{missing}. Declare 0 for a post this configuration genuinely "
                "does not have, so that the zero is a statement rather than an "
                "omission."
            )
        self._posts = {str(k): int(v) * MIB for k, v in posts.items()}
        self._cards: tuple[str, ...] = ()

    def estimate(self, spec: EngineSpec, cards: tuple[str, ...]) -> ResourceProfile:
        if len(cards) != 1:
            raise EstimateError(
                f"engine {spec.engine_id!r}: the M1 Class-2 estimator covers a "
                f"single-card tenant; got {len(cards)} cards. Multi-card "
                "diffusion (sequence parallel) is M4."
            )
        card = cards[0]
        peak = sum(self._posts.values())
        # Steady is the peak minus the activation peak: between steps the
        # tenant holds weights, latents and encoders, not the transient. The
        # reservation is still the peak (§3.8) and the difference is the
        # declared slack a #330 report names.
        steady = peak - self._posts.get("activation_peak_bytes", 0)
        return ResourceProfile(
            posts={card: dict(self._posts)},
            peak_bytes={card: peak},
            steady_bytes={card: steady},
            notes=(
                "estimate only: the M1 Class-2 adapter does not launch "
                "(§10 M1). Registering it reserves nothing until it is promoted, "
                "and promotion is refused.",
            ),
        )

    def bind(self, cards: tuple[str, ...]) -> None:
        self._cards = tuple(cards)

    def state(self) -> ResidencyState:
        return ResidencyState.COLD

    def pids(self) -> tuple[int, ...]:
        return ()

    def promote(self, target: ResidencyState) -> None:
        raise AdapterError(
            f"engine {self.spec.engine_id!r}: the Class-2 adapter estimates and does "
            "not launch in M1. Promoting a diffusion tenant is M3 (process group, "
            "ZMQ client, the two multimodal_gen seams of §5.4)."
        )

    def demote(self, target: ResidencyState) -> None:
        if target != ResidencyState.COLD:
            raise AdapterError(
                f"engine {self.spec.engine_id!r}: the Class-2 adapter is always COLD "
                "in M1"
            )

    def measured(self) -> Mapping[str, int]:
        return {}

    def health(self) -> Health:
        return Health(ok=True, detail="estimate-only adapter (M1)")


register_adapter(ADAPTER_NAME, Class2DiffusionAdapter)
