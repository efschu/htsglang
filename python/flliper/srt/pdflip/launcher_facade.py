"""Facade for the split launcher: a ``setattr`` on ``flliper.srt.pdflip.launcher`` reaches the ``launcher_*`` modules too.

``launcher.py`` was split behaviour-neutrally (refactor, 28.10.): its functions and constants live in ``launcher_*.py``
and are re-exported by ``launcher``. A function reads its constants and helpers from the module it now lives in, so a
caller that rebinds ``launcher.<name>`` (tests, ``propose_oracle``, ``hw_sim``: ``mock.patch.object(launcher, "STORE_ROOT", x)``,
``launcher._CARD_SELECTION = ...``) must reach every module that carries that name -- otherwise the patch would only
change what the staying code sees. The facade mirrors the rebinding (and the restore) onto each ``launcher_*`` module that
defines or imports the same name. Reading is untouched, and names a function rebinds itself with ``global`` stay in one
module (the mover keeps writer and readers together).
"""

import sys
import types

_SPLIT_PREFIX = "flliper.srt.pdflip.launcher_"


def _split_modules():
    return [m for name, m in list(sys.modules.items()) if name.startswith(_SPLIT_PREFIX) and m is not None]


class _LauncherFacade(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for mod in _split_modules():
            if name in mod.__dict__:
                mod.__dict__[name] = value

    def __delattr__(self, name):
        super().__delattr__(name)
        for mod in _split_modules():
            mod.__dict__.pop(name, None)


def install(module_name: str) -> None:
    """Make ``sys.modules[module_name]`` mirror attribute rebinding onto the ``launcher_*`` modules."""
    sys.modules[module_name].__class__ = _LauncherFacade
