"""First statement of the package: environment bridge and rig-state directory (RENAME_PLAN 8.7 step 2).

Runs before any module reads the environment, so every reader sees the canonical spelling and
child processes inherit it. The bridge itself is ``srt/name_compat.py`` (owned by the NF seat,
also called by the launcher's ``build_env`` and the container entrypoint): ONE helper, not two.
A missing helper is an ImportError on purpose -- a silently skipped bridge would let profiles
that set the old spelling fall back to defaults, the most dangerous class of the rename.
"""

import os


def bridge_environ(environ=None) -> None:
    from .srt.name_compat import canonical_env

    canonical_env(os.environ if environ is None else environ)


def link_state_dir() -> None:
    """Called by the server entry points (launch_server, the flip launcher), never at import."""
    from .srt.compat_shims import link_legacy_cache_dir

    link_legacy_cache_dir()
