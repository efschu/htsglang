# SPDX-License-Identifier: Apache-2.0
"""W2-APPLY (20.10.): FILTERED LANES -- the upstream EP mechanism, taken over for the pool's EM waves.

``run_pool_em`` (S51 EAGER-DEVICE-EM) computes an extend in waves; a wave serves only the lanes whose expert it
holds. Before, every other lane pointed at the zero row with weight 0 -- and still went through the GEMMs: every
wave ran the apply over ALL T x K lanes (rc27 D TP0 15:36:48Z: MIXED-D T=1922, em_ms 2134.6 of run_ms 2325,
copy_ms 371.8; experts=193 waves=2 at C=105; the S106 instrument showed wave 2 serving 0 lanes and its apply
running anyway).

Upstream already has the skip: in EP, a filtered expert's id is -1; ``moe_align_block_size`` (num_experts + 1
buckets, ids shifted by one) puts such lanes into blocks with expert id -1, and the Marlin MoE GEMMs with
``is_ep`` skip exactly those blocks (``marlin_template.h``: ``if (is_ep) ... expert_ids_ptr[i] == -1``;
``moe_wna16_marlin.cuh``: ``if (tmp_expert_id == -1) continue``), the second GEMM's output being zeroed first.
Here a wave marks its unserved lanes -1 and runs the apply inside :func:`active`, which makes
``fused_marlin_moe`` take that EP form without an expert map. A runner that does not have the skip is never
given a -1 lane: the capability is the MoE scheme's ``MOE_FILTERED_LANES`` (the compressed-tensors WNA16 Marlin
scheme), else the zero-row form stays. Thread-local, rank-local, no host read, no collective.
"""
from __future__ import annotations

import contextlib
import threading

_STATE = threading.local()


def is_active() -> bool:
    """Inside a pool EM wave's apply whose unserved lanes carry expert id -1."""
    return bool(getattr(_STATE, "on", False))


@contextlib.contextmanager
def active():
    prev = is_active()
    _STATE.on = True
    try:
        yield
    finally:
        _STATE.on = prev


def ep_form(expert_map) -> bool:
    """``fused_marlin_moe``'s EP form: an expert map (EP), or a pool EM wave's filtered lanes (W2-APPLY)."""
    return expert_map is not None or is_active()


def layer_supports(layer) -> bool:
    """The layer's MoE runner skips -1 lanes (the scheme says so; nothing is guessed)."""
    scheme = getattr(layer, "scheme", None)
    return bool(getattr(scheme, "MOE_FILTERED_LANES", False))
