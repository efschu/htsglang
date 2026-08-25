"""#861h: this suite REQUIRES an accelerator. Declared, not discovered.

MEASURED, hermetically, 2026-08-25. Every test here calls
`ScriptedHttpServer.start`, which spawns a real server child, and that child
dies at construction under `CUDA_VISIBLE_DEVICES=""`::

    ServerArgs.__post_init__ -> _handle_missing_default_values -> get_device()
    RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) or platform
    plugin is available.

Before #861h that death was an unbounded WAIT in `_await_handshake` -- the
parent polled for a handshake from a corpse until its deadline. The fix turned
it into a raise in ~15 s, which is correct but still wrong as a VERDICT: a
device-requiring suite reporting FAILURES on a card-free box is a red that
says nothing about the tree.

Per the CUDA test discipline, a test that genuinely needs a device is excluded
BY NAME and reported, never accommodated by loosening a guard. So the suite
declares its requirement here and SKIPS when no accelerator is present. On a
box with cards nothing changes.

Deliberately NOT a `try: import torch; torch.cuda.is_available()` sniff: the
authority is the same `get_device()` the child itself calls, so the skip
condition and the failure condition cannot drift apart.
"""

import pytest


def _accelerator_available() -> bool:
    try:
        from sglang.srt.utils.common import get_device

        get_device()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _accelerator_available():
        return
    skip = pytest.mark.skip(
        reason=(
            "#861h: scripted_runtime requires an accelerator -- its spawned "
            "server child raises 'No accelerator ... available' from "
            "ServerArgs.__post_init__ under CUDA_VISIBLE_DEVICES=''. Declared "
            "requirement, not a silent failure."
        )
    )
    for item in items:
        item.add_marker(skip)
