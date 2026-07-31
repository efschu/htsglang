# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Universal client liveness: one core, one class table, one grace registry.

#344b. Every long-lived client attachment in the server -- token streams,
video streams, training event taps, image and speech lane calls, registry
lease holders -- is watched by the same component with a per-endpoint-class
timeout, and everything a dead-suspect client holds is published to a
process-wide registry so the reclamation ladder can see it.

Placement note: this sits in ``srt/liveness`` rather than in the
``video_enhance`` tenant where it was born, because ``srt/entrypoints`` and
``srt/registry`` must not import from a tenant package.
``sglang.srt.video_enhance.liveness`` re-exports from here, unchanged.

Not this module's layer: #312's bounded peer liveness inside collectives.
That watches *ranks* on NCCL/HTCCL and its failure mode is a hung all-reduce.
This watches *clients* on a socket. The two never interact and share no code.
"""

from sglang.srt.liveness.classes import (
    DEFAULT_TIMEOUT_RATIONALE,
    DEFAULT_TIMEOUTS_S,
    EndpointClass,
)
from sglang.srt.liveness.grace import (
    Attachment,
    AttachmentPhase,
    AttachmentRegistry,
    ClaimKind,
    ResourceClaim,
    global_attachment_registry,
)
from sglang.srt.liveness.ledger_bridge import (
    LedgerGraceBridge,
    attach_ledger_grace_bridge,
)
from sglang.srt.liveness.stream import (
    await_with_liveness,
    claims_for_rids,
    guard_generate_stream,
    guard_streaming_response,
    guarded_stream,
)
from sglang.srt.liveness.watchdog import (
    DEFAULT_GRACE_FRACTION,
    ConsumerGone,
    ConsumerWatchdog,
    LivenessConfig,
    LivenessPolicy,
    LivenessState,
    global_liveness_config,
    set_global_liveness_config,
)

__all__ = [
    "Attachment",
    "AttachmentPhase",
    "AttachmentRegistry",
    "ClaimKind",
    "ConsumerGone",
    "ConsumerWatchdog",
    "DEFAULT_GRACE_FRACTION",
    "DEFAULT_TIMEOUTS_S",
    "DEFAULT_TIMEOUT_RATIONALE",
    "EndpointClass",
    "LedgerGraceBridge",
    "LivenessConfig",
    "LivenessPolicy",
    "LivenessState",
    "ResourceClaim",
    "attach_ledger_grace_bridge",
    "await_with_liveness",
    "claims_for_rids",
    "global_attachment_registry",
    "global_liveness_config",
    "guard_generate_stream",
    "guard_streaming_response",
    "guarded_stream",
    "set_global_liveness_config",
]
