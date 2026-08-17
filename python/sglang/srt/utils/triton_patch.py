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

import importlib.util
import logging
import sys
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_FINDER: Optional[_TritonPatchFinder] = None


def next_power_of_2(n: int) -> int:
    """The override itself: a bit-length shift instead of a loop."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


def apply_patch(module: Any) -> None:
    """Install the override onto an already-executed triton module."""
    setattr(module, "next_power_of_2", next_power_of_2)


class _PatchingLoader:
    """Wraps triton's real loader and patches the module after it executes.

    Delegates everything else, so triton loads exactly as it would have; the
    only difference is one attribute assignment appended to ``exec_module``.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        apply_patch(module)

    def __getattr__(self, item):
        return getattr(self._inner, item)


class _TritonPatchFinder:
    """Claims ``triton`` once, to hand back a spec with a patching loader."""

    def find_module(self, fullname, path=None):  # pragma: no cover - legacy API
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "triton":
            return None
        # Resolve the REAL spec with this finder removed, or find_spec would
        # ask us again and recurse. Re-inserted in the finally, so a failed
        # resolution does not silently disarm the patch for the rest of the
        # process.
        with _LOCK:
            try:
                sys.meta_path.remove(self)
            except ValueError:  # pragma: no cover - concurrent removal
                return None
            try:
                spec = importlib.util.find_spec("triton")
            except Exception as e:  # pragma: no cover - broken install
                logger.warning("triton spec lookup failed: %s", e)
                spec = None
            finally:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader)
        return spec


def install() -> None:
    """Arm the patch. Idempotent, and cheap: it does not import triton.

    If triton is ALREADY imported (another module got there first, or this is
    a re-install), the patch is applied directly -- the guarantee is "patched
    before any read", and a module already in ``sys.modules`` cannot be
    patched by a future import.
    """
    global _FINDER
    with _LOCK:
        already = sys.modules.get("triton")
        if already is not None:
            apply_patch(already)
        if _FINDER is None:
            _FINDER = _TritonPatchFinder()
            sys.meta_path.insert(0, _FINDER)


def uninstall() -> None:
    """Test hook: remove the finder."""
    global _FINDER
    with _LOCK:
        if _FINDER is not None:
            try:
                sys.meta_path.remove(_FINDER)
            except ValueError:
                pass
            _FINDER = None


def is_armed() -> bool:
    return _FINDER is not None and _FINDER in sys.meta_path
