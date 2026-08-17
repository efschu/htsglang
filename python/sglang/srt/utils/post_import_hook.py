"""Run a callback the moment a module finishes importing -- without importing it.

Two patches in this tree need the same guarantee: "apply this to package X
before anyone can use X, but do not import X to do it". The triton override
(#673) and the transformers compatibility patches (#237 root ticket) both had
the shape "import the heavy package at module scope purely to patch it", which
is exactly what made text-only processes -- the tokenizer manager, the
detokenizer, router-class processes -- load GPU and model machinery they never
call.

The mechanism: a meta-path finder claims the target name, resolves the REAL
spec through the rest of ``sys.meta_path``, and wraps its loader so the
callback runs after the module body executes and BEFORE the ``import``
statement returns to whoever asked. The ordering therefore holds by
construction rather than by discipline:

    a user must import the module to use it, and the callback runs inside that
    import.

Arming costs one object on ``sys.meta_path`` and no import of the target.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_FINDERS: Dict[str, _PostImportFinder] = {}


class _PatchingLoader:
    """Wraps the real loader; appends the callback to ``exec_module``.

    Everything else is delegated, so the target loads exactly as it would
    have -- the only difference is the callback appended to its execution.
    """

    def __init__(self, inner: Any, callback: Callable[[Any], None], name: str) -> None:
        self._inner = inner
        self._callback = callback
        self._name = name

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        try:
            self._callback(module)
        except Exception:
            # A failing patch must not turn a working import into an
            # ImportError: the caller asked for the module, and half the point
            # of these patches is compatibility shims that are allowed to be
            # inapplicable. Loud, and the module is still returned.
            logger.exception("post-import hook for %r failed", self._name)

    def __getattr__(self, item):
        return getattr(self._inner, item)


class _PostImportFinder:
    """Claims one module name, to hand back a spec with a patching loader."""

    def __init__(self, name: str, callback: Callable[[Any], None]) -> None:
        self.name = name
        self.callback = callback

    def find_module(self, fullname, path=None):  # pragma: no cover - legacy API
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        # Resolve the REAL spec with this finder removed, or find_spec would
        # ask us again and recurse. Re-inserted in the finally, so a failed
        # resolution cannot silently disarm the hook for the rest of the run.
        with _LOCK:
            try:
                sys.meta_path.remove(self)
            except ValueError:  # pragma: no cover - concurrent removal
                return None
            try:
                spec = importlib.util.find_spec(self.name)
            except Exception as e:  # pragma: no cover - broken install
                logger.warning("spec lookup for %r failed: %s", self.name, e)
                spec = None
            finally:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, self.callback, self.name)
        return spec


def install(name: str, callback: Callable[[Any], None]) -> None:
    """Arm ``callback`` for when ``name`` is imported. Idempotent, and cheap.

    If ``name`` is ALREADY imported, the callback runs immediately: a module
    in ``sys.modules`` cannot be patched by a future import, and the guarantee
    is "patched before any use", not "patched by the hook".
    """
    with _LOCK:
        already = sys.modules.get(name)
        if already is not None:
            callback(already)
        if name not in _FINDERS:
            finder = _PostImportFinder(name, callback)
            _FINDERS[name] = finder
            sys.meta_path.insert(0, finder)


def uninstall(name: str) -> None:
    """Test hook: remove the finder for ``name``."""
    with _LOCK:
        finder = _FINDERS.pop(name, None)
        if finder is not None:
            try:
                sys.meta_path.remove(finder)
            except ValueError:
                pass


def is_armed(name: str) -> bool:
    finder = _FINDERS.get(name)
    return finder is not None and finder in sys.meta_path


def armed_names() -> tuple:
    return tuple(sorted(_FINDERS))


def _reset_for_test() -> None:
    for name in list(_FINDERS):
        uninstall(name)
