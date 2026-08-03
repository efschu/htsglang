"""Live interview copilot (#502).

A browser app that listens to both sides of an online conversation and shows
the user keywords and short explanations to READ while the conversation runs.

The app process owns no model, no VRAM and no CUDA context. It reaches the
htsglang runtime over the runtime's own public surface: ``/v1/realtime`` for
streaming ASR and ``/v1/chat/completions`` for hints. See
``docs/dev/DESIGN_502_interview_copilot.md``.
"""

from sglang.srt.copilot.config import CopilotConfig

__all__ = ["CopilotConfig"]
