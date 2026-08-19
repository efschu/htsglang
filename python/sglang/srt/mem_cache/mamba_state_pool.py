# SPDX-License-Identifier: Apache-2.0
"""Phase-aware mamba STATE pool resolution (#767 residual).

Under a phase-flip build the radix cache's ``req_to_token_pool`` is bound
once, to the PRIMARY PP stack's pool, and never rebound (slot NUMBERING is
deliberately single-authority: one allocator, aliased mapping tensors).
The mamba STATE tensors, however, are per-stack -- a request computing in
the TP phase writes its conv/ssm bytes into the TP stack's pool. Any
tree-side operation that touches state BYTES through the bound pool
therefore reads stale storage whenever the other stack is computing:
measured 2026-08-19 as checkpoint slots filled with a previous PP-phase
occupant's state (a kite prompt answered with a foreign river essay).

``active_mamba_state_pool`` is the one seam: bookkeeping stays on the
bound pool, state-byte operations resolve the pool of the phase that is
actually computing. Non-flip builds have no resolver installed and fall
back to the bound pool -- bit-for-bit the old behavior.
"""

from __future__ import annotations


def active_mamba_state_pool(cache):
    """The mamba pool whose TENSORS hold the currently-computing phase's
    state bytes.

    ``cache`` is a radix cache. A phase-flip boot installs
    ``cache.phase_active_mamba_pool`` (see
    gdn_flip_mover.install_phase_aware_mamba_state_pool); without it the
    bound pool is the only pool and is returned unchanged.
    """
    resolver = getattr(cache, "phase_active_mamba_pool", None)
    if resolver is not None:
        return resolver()
    return cache.req_to_token_pool.mamba_pool
