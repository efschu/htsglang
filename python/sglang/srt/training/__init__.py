# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Training and finetuning as an idle tenant of the rig (DESIGN #341).

The rig trains whenever it is not inferencing. Four pieces, one per module:

``store``
    Uploaded files and job records, in OpenAI's own shapes and state machine.
``feasibility``
    Does this job fit on *this* machine? A formula over NVML, the VRAM
    ledger, ``/proc/meminfo`` and the model's own config -- never a constant
    taken from the development rig (D2).
``backends``
    Existing training suites, wrapped as subprocesses and never vendored
    (D1). LLaMA-Factory for LLMs in M1; kohya and Unsloth slot in at M2.
``tenant``
    Idle detection, the VRAM lease, and checkpoint-and-release preemption
    (D4).

``service`` assembles them. Nothing in this package imports torch, so the
whole surface can be exercised on a host with no card.
"""

from sglang.srt.training.activity import note_serving_activity

__all__ = ["note_serving_activity"]
