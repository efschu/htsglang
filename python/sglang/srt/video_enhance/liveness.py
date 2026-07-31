# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Compatibility re-export of the shared liveness core.

This module was where consumer-liveness detection was first built (#344a,
TASK_333 §8.2). #344b generalized it to every endpoint class in the server
and moved it to :mod:`sglang.srt.liveness`, because ``srt/entrypoints`` and
``srt/registry`` also need it and neither may import from a tenant package.

The names below are the ones this module exported before the move and they
behave identically -- ``EndpointClass`` gained members, and no existing
member, default or signature changed. Import from
:mod:`sglang.srt.liveness` in new code.
"""

from sglang.srt.liveness.classes import DEFAULT_TIMEOUTS_S, EndpointClass
from sglang.srt.liveness.watchdog import (
    ConsumerGone,
    ConsumerWatchdog,
    LivenessConfig,
    LivenessPolicy,
    LivenessState,
)

__all__ = [
    "ConsumerGone",
    "ConsumerWatchdog",
    "DEFAULT_TIMEOUTS_S",
    "EndpointClass",
    "LivenessConfig",
    "LivenessPolicy",
    "LivenessState",
]
