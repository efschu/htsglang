"""#673/#237: patch triton WHEN it is imported, not by importing it.

``utils/common.py`` used to do this at module scope::

    import triton
    ...
    setattr(triton, "next_power_of_2", next_power_of_2)

Two lines, and the first one is why every process on this box loaded triton.
``utils/common`` is reached by the package root, so the tokenizer manager and
the detokenizer -- processes that must never touch CUDA (#237/#403) -- imported
a GPU kernel compiler to install a two-line integer helper they would never
call.

Deleting the patch is not on the table: 235 call sites across the tree read
``triton.next_power_of_2``, and the override exists because triton's own
implementation has been a loop rather than a bit-length shift. Moving the patch
"next to the first consumer" is not on the table either -- with 235 readers
there is no first consumer, and a patch that races its readers is a silent
numerics-and-perf landmine rather than a crash.

So the patch is applied BY THE IMPORT ITSELF. A meta-path finder claims
``triton``, resolves the real spec through the rest of the path, and wraps its
loader so the attribute is set immediately after the module body finishes and
BEFORE ``import triton`` returns to whoever asked. That is the ordering
property this module exists to guarantee, and it holds by construction rather
than by discipline:

    a reader must import triton to read the attribute, and the patch is
    installed inside that import.

Cost when nothing imports triton: one object on ``sys.meta_path``, and no
triton.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.utils import post_import_hook

_MODULE = "triton"


def next_power_of_2(n: int) -> int:
    """The override itself: a bit-length shift instead of a loop."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


def apply_patch(module: Any) -> None:
    """Install the override onto an already-executed triton module."""
    setattr(module, "next_power_of_2", next_power_of_2)


def install() -> None:
    """Arm the patch. Idempotent, and cheap: it does not import triton."""
    post_import_hook.install(_MODULE, apply_patch)


def uninstall() -> None:
    """Test hook: remove the finder."""
    post_import_hook.uninstall(_MODULE)


def is_armed() -> bool:
    return post_import_hook.is_armed(_MODULE)
