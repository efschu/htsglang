# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#603b: build the sampling backend's JIT kernels at BOOT, not mid-forward.

THE DEFECT
----------
``flashinfer.sampling.top_k_top_p_sampling_from_probs`` resolves its CUDA
module lazily, on first call, through ``get_sampling_module`` ->
``build_and_load`` -> ninja -> nvcc. On a cold kernel cache that is a 60-90 s
compile, and the first call lands in ``Sampler.forward`` -- i.e. INSIDE the
serving forward, on a rank that its peers are waiting for.

Seven production crashes on 2026-08-06 ended as
``Bar1CollectiveAborted ... a peer did not arrive``. py-spy caught all three
ranks mid-wedge with the answer: two in ``run_ninja``, one blocked on the
build's ``FileLock``. The ``.o``/``.so`` mtimes in both arch caches sit inside
that same window, so it was a real compile, not a re-validation.

WHY THIS RIG AND NOT A UNIFORM ONE
----------------------------------
flashinfer keys its JIT cache by ARCH (``.cache/flashinfer/<ver>/<arch>/``),
and each rank sees only its own GPU. On a heterogeneous rig the 5090 rank
resolves to one arch dir and the two 3080 ranks resolve to a SHARED one --
same directory, same ``FileLock`` -- so those two SERIALISE against each other
while the odd rank builds alone. The ranks therefore leave the stall at
materially different times. Whichever leaves first walks into the next
forward's first collective (the vocab-parallel embedding all_reduce) and waits
on a peer still inside nvcc until the spin deadline fires.

Nothing is SKIPPED in this failure, which is why the collective census reports
byte-identical counts and cannot see it: it is a pure rank-local STALL, not a
count divergence.

WHY THE FIX IS PLACEMENT, NOT A LONGER DEADLINE
-----------------------------------------------
Wrapping the lazy build in ``cold_build_window`` does NOT work, and the reason
is worth stating so it is not "fixed" that way later: the window is
process-local, and the rank that aborts is the one NOT building -- its
multiplier is closed. Raising the deadline only pads the race. Moving the
build to a point where NO rank can be in a collective is what removes it.

  STALE AS OF 2026-08-07 -- READ THE PARAGRAPH ABOVE WITH ITS DATE (#1033c).
  The "process-local" premise was true when this was written (8bddb93d16,
  2026-08-06) and #615 falsified it ONE DAY LATER (38ec4fb348, 2026-08-07):
  ``cold_build_window`` now PUBLISHES a marker that peers read without any
  cooperation from the builder, and they extend their deadlines under a 900 s
  cap -- "every existing call site therefore becomes group-visible without
  moving". Both commits are ancestors of today's pin; checked with
  ``git log -S``, not assumed.

  This correction is written here rather than by deleting the paragraph
  because the paragraph is still RIGHT about this module: warming at boot
  plus a barrier REMOVES the race, while a window only EXTENDS it under a cap
  -- so #603b's own placement stands unchanged. What is no longer true is the
  general claim that wrapping a lazy build "does NOT work", and that claim,
  read at face value in 2026-08-31, nearly refuted the correct fix for the
  boot-22 wedge (#1033c, ``Scheduler.run_batch``) before it was built.
  A determination carries its date into its verdict; this one did not, and
  the cost was one near-miss.

So: build here, at boot, on every rank, inside the cold-build window, and then
BARRIER. The barrier is the load-bearing half -- it is what guarantees no rank
leaves for the serving loop while a peer is still in nvcc. It runs on the CPU
(gloo) group, which carries no deadline, so it can wait out an arbitrarily
long build.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["warm_sampling_backend_kernels"]

#: Width of the probe distribution. Only the KERNEL needs to be resident; the
#: vocab size is irrelevant to which module gets built, so this stays tiny
#: rather than allocating a real vocab-width tensor during boot.
_PROBE_VOCAB = 32


def sampling_backend_needs_jit_warmup(backend: str) -> bool:
    """Whether ``backend`` reaches a lazily-built JIT module.

    Keyed ONLY on the replicated server-args value. It must never consult a
    per-rank capability probe (e.g. whether sgl-kernel carries code for this
    card): the caller turns this answer into a collective, so a rank-local
    input here would put some ranks in the barrier and leave others out --
    which is the very failure class this module exists to close.
    """
    return backend == "flashinfer"


def warm_sampling_backend_kernels(
    backend: str,
    device: Any,
    tp_group: Any = None,
) -> str:
    """Force the sampling JIT module resident, then rendezvous.

    Returns a short status string (for the boot log and for tests).

    Ordering contract, and the reason for each step:

    1. The build runs inside ``cold_build_window`` so that any deadline-bearing
       device collective issued by OTHER machinery during this stretch gets the
       relaxed deadline.
    2. A build failure is logged, NOT raised. The lazy path still works, so
       failing the boot would trade a latent hang for a certain outage. The log
       line names the consequence explicitly so a silent regression cannot hide.
    3. The barrier runs UNCONDITIONALLY on the config-derived branch --
       including after a failed build. Every rank computes ``backend`` from the
       same replicated server args, so either all ranks reach the barrier or
       none do. Skipping it on the error path would desync the barrier itself.
    """
    if not sampling_backend_needs_jit_warmup(backend):
        return f"skipped: sampling_backend={backend!r} has no lazy JIT module"

    from sglang.srt.utils.jit_cold_build import cold_build_window

    status = "ok"
    with cold_build_window("flashinfer sampling JIT warmup (#603b)"):
        try:
            _run_probe_sampling(device)
        except Exception as exc:  # noqa: BLE001 - see contract note 2 above
            status = f"failed: {type(exc).__name__}: {exc}"
            logger.error(
                "sampling-backend JIT warmup FAILED (%s: %s). The kernels were "
                "NOT made resident, so the build will now happen lazily inside "
                "the first serving forward -- which is the #603b wedge: peers "
                "spin on a deadline-bearing collective while this rank sits in "
                "nvcc for 60-90 s, and the group aborts with "
                "'a peer did not arrive'. Boot continues because the lazy path "
                "still produces correct output.",
                type(exc).__name__,
                exc,
            )

    if tp_group is not None:
        # THE load-bearing step: no rank may enter the serving loop while a
        # peer is still building. CPU/gloo group, so it carries no deadline.
        tp_group.barrier()

    return status


def _run_probe_sampling(device: Any) -> None:
    """One tiny real call per lazily-built entry point.

    Deliberately the PUBLIC API rather than ``get_sampling_module``: the public
    call is what a request takes, so this proves the exact path is resident
    instead of asserting it from a private helper that could stop being the one
    the sampler reaches. Both entry points are exercised because the sampler
    dispatches to either depending on ``need_min_p_sampling``; they share one
    JIT module today, so the second call is nearly free, and if that ever stops
    being true this still covers both.

    Carries no collective, so it is safe to run before the barrier.
    """
    import torch
    from flashinfer.sampling import (
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )

    probs = torch.full((1, _PROBE_VOCAB), 1.0 / _PROBE_VOCAB, device=device)
    top_ks = torch.ones(1, dtype=torch.int32, device=device)
    top_ps = torch.ones(1, device=device)
    min_ps = torch.zeros(1, device=device)

    top_k_top_p_sampling_from_probs(
        probs.contiguous(), top_ks, top_ps, filter_apply_order="joint"
    )
    min_p_sampling_from_probs(probs.contiguous(), min_ps)
