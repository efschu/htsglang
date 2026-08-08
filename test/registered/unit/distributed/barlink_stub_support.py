# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#624: anti-drift support for ``__new__``-style BAR1 transport stubs.

THE DRIFT (why this module exists). The abort tests build a
``BarlinkBar1Transport`` via ``__new__`` and hand-set the attributes the
methods under test read. That list silently diverges from the real
``__init__`` every time the transport gains state: #622's ack barrier alone
added three fields the stubs had to learn by failing (historically recorded
as the 7-fail builder drift of the deferred-517 file; the ack-barrier
adaptation was a repeat of the same mechanism, hand-patched again). At
audit time the real ``__init__`` assigns ~103 attributes and the stub sets
~30 — every one of the other ~70 is a latent drift.

THE FIX SHAPE. Not a bigger stub — an AUDIT: every attribute the real
``__init__`` assigns must be either (a) set by the stub builder, or (b)
listed here with a reason. A new transport attribute then turns the audit
test RED with its name in the message, forcing a conscious decision instead
of a silent drift; a removed attribute turns a stale exclusion RED the same
way (the comparison is exact, both directions).
"""

from __future__ import annotations

import ast
import inspect
from typing import Set


def init_assigned_attrs(cls) -> Set[str]:
    """Names of every ``self.<attr>`` assigned in ``cls.__init__`` (AST)."""
    src = inspect.getsource(cls.__init__)
    tree = ast.parse(
        "class _X:\n" + "\n".join("    " + line for line in src.splitlines())
    )
    names: Set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            names.add(node.attr)
    return names


#: Attributes the real ``__init__`` assigns that the abort-path stubs
#: deliberately do NOT set, each with the reason. Reviewed 2026-08-08
#: against barlink_bar1.py @ f2154baa9b. Grouping mirrors the transport's
#: own lifecycle: none of these are read by ``check_aborted`` /
#: ``_read_status_for_check`` / ``poll_status_word`` / the raise path,
#: which is the entire surface the abort tests exercise.
def _group(reason: str, *names: str) -> dict:
    return {n: reason for n in names}


ABORT_STUB_EXCLUSIONS = {
    # -- bring-up / aperture plumbing: touched by _build_up and close() only
    **_group(
        "bring-up/teardown plumbing; abort paths read tensors already built",
        "cpu_group", "device", "window_bytes", "_holder", "_cuda", "_ext",
        "_pipe_ext", "_dmabuf_fds", "_foreign_fds", "_hold_fds",
        "_peer_table", "_window_minimum", "ordinal",
    ),
    # -- byte-proof / sensor machinery: bring-up self-checks, never read by
    #    the abort or status paths
    **_group(
        "bring-up byte-proof/sensor state",
        "_a2a_proof", "_bc_proof", "_pipe_proof", "_own_sensor",
        "_proofs_hold",
    ),
    # -- launch planner & thresholds: consulted when DISPATCHING collectives;
    #    the abort tests drive only the status/abort readers
    **_group(
        "dispatch-path threshold/planner state",
        "_plan", "max_bytes", "min_bytes", "ring_from", "a2a_max_rounds",
        "a2a_min_bytes", "a2a_on", "ag_max_rounds", "ag_min_bytes", "ag_on",
        "ar_max_rounds", "bc_max_rounds", "bc_min_bytes", "bc_on",
        "pipe_ack", "pipe_chunk_bytes", "pipe_direct", "pipe_direct_graph",
        "pipe_from", "pipe_grid_from", "pipe_k", "pipe_k_max", "pipe_lead",
        "pipe_on", "pipe_result_eager", "pipe_result_ring", "pipe_slot",
        "pipe_slot_kib", "pipe_t",
    ),
    # -- result relay (#292-family device-resident slots): launch/replay
    #    path state; the abort raise only formats already-host fields
    **_group(
        "result-relay slot machinery, launch/replay path",
        "_result_alive", "_result_counter", "_result_eager_full",
        "_result_eager_full_reported", "_result_eager_slots",
        "_result_gen_dev", "_result_graph_assigned",
        "_result_graph_empty_reported", "_result_graph_slots", "_result_i",
        "_result_last", "_step_dev",
    ),
    # -- one-shot reporting flags: cosmetics, guarded by getattr defaults
    **_group(
        "one-shot log-dedupe flag, getattr-guarded",
        "_direct_graph_reported", "_expiry_census_fired",
    ),
    # -- watchdog poll mirror (#622): poll_status_word guards it with an
    #    explicit None check, and the stubs exercise the staged-read path,
    #    not the watchdog poll allocation
    "_round_mirror": "None-guarded pinned mirror; watchdog-poll alloc only",
}
