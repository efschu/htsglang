# SPDX-License-Identifier: Apache-2.0
"""Deprecated-name compatibility for HTCCL environment variables.

HTCCL's env vars were renamed from German to English identifiers (task
#295). To avoid silently breaking anyone still exporting an old name, every
renamed var is resolved through this module: if only the OLD name is set,
its value is copied onto the NEW name. The NEW name always wins when both
are set. A ``DeprecationWarning`` is emitted the first time an old var is
observed to be set, then never again for that var in this process.

Resolution runs at IMPORT time (a module-level call below) for every
htccl_*.py module that reads a renamed var as a module-level constant
(``from sglang.srt.distributed.device_communicators import
htccl_env_compat  # noqa: F401``) -- that covers the common case, since
those constants are only ever read once, at their own module's import time.

It also has to run again on every CALL for the few readers that
deliberately re-read ``os.environ`` live on every invocation rather than
caching a module-level constant (e.g. ``parallel_state.graph_freigabe_gesetzt``
and ``htccl_bar1.graph_grid_default``) -- otherwise a test or caller that
sets the OLD name only after those modules have already been imported would
see it silently ignored, since the module-import-time resolution has
already run and won't run again on its own. Those call sites call
``resolve_env_aliases()`` explicitly, right before their own
``os.environ.get(...)`` read.
"""

import os
import warnings

# OLD (deprecated, German) -> NEW (current, English) env var name.
# Vars whose name was already English are intentionally absent here -- they
# never had an old name to alias.
DEPRECATED_ALIASES = {
    "SGLANG_HTCCL_AUFTEILUNG": "SGLANG_HTCCL_SPLIT",
    "SGLANG_HTCCL_BAR1_A2A_MAX_RUNDEN": "SGLANG_HTCCL_BAR1_A2A_MAX_ROUNDS",
    "SGLANG_HTCCL_BAR1_AG_MAX_RUNDEN": "SGLANG_HTCCL_BAR1_AG_MAX_ROUNDS",
    "SGLANG_HTCCL_BAR1_AR_MAX_RUNDEN": "SGLANG_HTCCL_BAR1_AR_MAX_ROUNDS",
    "SGLANG_HTCCL_BAR1_BC_MAX_RUNDEN": "SGLANG_HTCCL_BAR1_BC_MAX_ROUNDS",
    "SGLANG_HTCCL_BAR1_DECKEL_ZYKLEN": "SGLANG_HTCCL_BAR1_CAP_CYCLES",
    "SGLANG_HTCCL_BAR1_FENSTER_MIB": "SGLANG_HTCCL_BAR1_WINDOW_MIB",
    "SGLANG_HTCCL_BAR1_FENSTER_MIB_DCP": "SGLANG_HTCCL_BAR1_WINDOW_MIB_DCP",
    "SGLANG_HTCCL_BAR1_FLUSS": "SGLANG_HTCCL_BAR1_FLOW",
    "SGLANG_HTCCL_BAR1_GITTER_AB": "SGLANG_HTCCL_BAR1_GRID_THRESHOLD",
    "SGLANG_HTCCL_BAR1_GRAPH_GITTER": "SGLANG_HTCCL_BAR1_GRAPH_GRID",
    "SGLANG_HTCCL_BAR1_HALTER": "SGLANG_HTCCL_BAR1_HOLDER",
    "SGLANG_HTCCL_BAR1_LADEFORM": "SGLANG_HTCCL_BAR1_LOAD_SHAPE",
    "SGLANG_HTCCL_BAR1_NV_QUELLE": "SGLANG_HTCCL_BAR1_NV_SOURCE",
    "SGLANG_HTCCL_BAR1_PIPE_AB": "SGLANG_HTCCL_BAR1_PIPE_THRESHOLD",
    "SGLANG_HTCCL_BAR1_PIPE_DIREKT": "SGLANG_HTCCL_BAR1_PIPE_DIRECT",
    "SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH": "SGLANG_HTCCL_BAR1_PIPE_DIRECT_GRAPH",
    "SGLANG_HTCCL_BAR1_PIPE_ERG_EAGER": "SGLANG_HTCCL_BAR1_PIPE_RESULT_EAGER",
    "SGLANG_HTCCL_BAR1_PIPE_ERG_RING": "SGLANG_HTCCL_BAR1_PIPE_RESULT_RING",
    "SGLANG_HTCCL_BAR1_PIPE_GITTER_AB": "SGLANG_HTCCL_BAR1_PIPE_GRID_THRESHOLD",
    "SGLANG_HTCCL_BAR1_PIPE_QUITTUNG": "SGLANG_HTCCL_BAR1_PIPE_ACK",
    "SGLANG_HTCCL_BAR1_PIPE_SCHLITZ_KIB": "SGLANG_HTCCL_BAR1_PIPE_SLOT_KIB",
    "SGLANG_HTCCL_BAR1_PIPE_VORLAUF": "SGLANG_HTCCL_BAR1_PIPE_LEAD",
    "SGLANG_HTCCL_BAR1_RING_AB": "SGLANG_HTCCL_BAR1_RING_THRESHOLD",
    "SGLANG_HTCCL_GRAPH_FREIGABE": "SGLANG_HTCCL_GRAPH_ENABLE",
    "SGLANG_HTCCL_KONFIG": "SGLANG_HTCCL_CONFIG",
    "SGLANG_HTCCL_MATRIX_CACHE_AUS": "SGLANG_HTCCL_MATRIX_CACHE_OFF",
    "SGLANG_HTCCL_MESS_BUDGET_MS": "SGLANG_HTCCL_MEASURE_BUDGET_MS",
    "SGLANG_HTCCL_MESS_DUPLEX": "SGLANG_HTCCL_MEASURE_DUPLEX",
    "SGLANG_HTCCL_MESS_FANIN": "SGLANG_HTCCL_MEASURE_FANIN",
    "SGLANG_HTCCL_NETZ_FAKTOR": "SGLANG_HTCCL_MESH_FACTOR",
    "SGLANG_HTCCL_ROLLEN": "SGLANG_HTCCL_ROLES",
    "SGLANG_HTCCL_SCHRITT_US": "SGLANG_HTCCL_STEP_US",
}


#: Old names that have already emitted their one-time DeprecationWarning in
#: this process. A set, not a bool, because each old var warns independently
#: the first time IT is seen set, regardless of the others.
_WARNED: set = set()

#: new_name -> the value THIS module last wrote into os.environ[new_name]
#: while syncing it from the old name. Needed to tell "the user explicitly
#: set the new name" apart from "this is just what we synced onto it last
#: time" -- without it, syncing once would permanently freeze the new name
#: at that value even after the caller changes the OLD name again (e.g. a
#: test that flips the old name from "0" to "1" mid-test and expects the
#: new name to follow).
_LAST_SYNCED: dict = {}


def resolve_env_aliases() -> None:
    """Copy every OLD env var still set onto its NEW name (new wins if both
    are set). Safe to call any number of times: re-syncs the value on every
    call, but each old name only ever warns once per process."""
    for old_name, new_name in DEPRECATED_ALIASES.items():
        if old_name not in os.environ:
            continue
        if old_name not in _WARNED:
            _WARNED.add(old_name)
            warnings.warn(
                f"{old_name} is deprecated and will be removed in a future "
                f"release; use {new_name} instead. Both may be set during a "
                f"migration window -- {new_name} always wins.",
                DeprecationWarning,
                stacklevel=2,
            )
        current_new = os.environ.get(new_name)
        # Sync when the new name is unset, OR when its current value is
        # exactly what we ourselves last synced onto it (so it hasn't been
        # explicitly overridden by anyone since) -- in both cases the old
        # name's CURRENT value is free to take over. If the new name holds
        # anything else, someone set it deliberately: it wins, untouched.
        if current_new is None or current_new == _LAST_SYNCED.get(new_name):
            new_value = os.environ[old_name]
            os.environ[new_name] = new_value
            _LAST_SYNCED[new_name] = new_value


resolve_env_aliases()
