"""Per-rank CuTe-DSL compile-target alignment for mixed-architecture rigs.

THE DEFECT (bug #208)
--------------------
``nvidia-cutlass-dsl`` decides, ONCE PER PROCESS, which GPU architecture it
compiles for, and it reads that from **driver device 0** -- not from the
device the process actually runs on::

    base_dsl/env_manager.py :: detect_gpu_arch()
        -> base_dsl/runtime/cuda.py :: get_compute_capability_major_minor(
               device_id=0)          # <-- hardcoded 0
        -> stored as CuTeDSL.envar.arch, e.g. "sm_86" / "sm_120a"

``envar.arch`` is then used for two different things:

* it is the codegen target of every ``cute.compile``
  (``base_dsl/dsl.py :: compile_and_cache`` -> ``preprocess_pipeline``), and
* it is part of the JIT cache key -- ``base_dsl/dsl.py :: get_module_hash``
  hashes every ``envar`` attribute -- which keys both the in-memory cache and
  the on-disk one under ``$CUTE_DSL_CACHE_DIR`` (default
  ``$TMPDIR/<user>/cutlass_python_cache``).

sglang's TP ranks are separate processes, but they only get ONE visible GPU
each when ``--rank-gpu-id`` is passed: that flag forces
``SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1`` and ``maybe_reindex_device_id()``
pins ``CUDA_VISIBLE_DEVICES`` to a single card per worker.  Without it every
rank sees every GPU, so driver device 0 is the SAME physical card in every
rank process.

On a mixed-architecture rig that is fatal: every rank compiles for device 0's
architecture.  The rank that does not own device 0 launches a kernel image its
card cannot execute and dies with ``cudaErrorNoKernelImageForDevice`` at the
first CuTe kernel -- observed in ``flashinfer.norm.rmsnorm_cute`` on a
3080 (sm86) rank while device 0 was a 5090 (sm120).  The shared on-disk cache
makes it sticky: the entry is filed under the wrong arch, so it is reused.

THE FIX
-------
Tell the DSL the architecture of the device THIS process will use, as early as
the rank's ``gpu_id`` is known.  Two mechanisms, because either alone is
insufficient:

* ``CUTE_DSL_ARCH`` in the environment -- honoured by ``detect_gpu_arch``'s
  ``get_str_env_var`` and inherited by any child process; but it is read when
  the ``CuTeDSL`` singleton is constructed, which already happened if
  ``flashinfer.norm`` was imported (it constructs the singleton at import).
* patching ``envar.arch`` on an already-constructed singleton -- every
  consumer (``compile_and_cache``, ``get_module_hash``, ``get_arch_enum``)
  reads the attribute at call time, so a late assignment is picked up.

On a homogeneous rig the value written is identical to what cutlass would have
detected by itself, so this is a no-op there.  A user-supplied
``CUTE_DSL_ARCH`` always wins and is never overwritten.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_ENV_VAR = "CUTE_DSL_ARCH"


def cute_dsl_arch_for_device(gpu_id: int) -> Optional[str]:
    """Return the ``CUTE_DSL_ARCH`` string for CUDA device ``gpu_id``.

    The format is replicated from cutlass's own ``detect_gpu_arch()``:
    ``sm_<major><minor>`` plus an ``a`` suffix from Hopper on (major >= 9).
    It must match character for character -- the string is both a codegen
    target and a cache-key component.

    ``torch`` is deliberately the source of the compute capability, not the
    CUDA driver API cutlass uses: what matters is the card this rank will
    actually run on, which is the one ``torch.cuda.set_device(gpu_id)``
    selects, and torch's device order can differ from the driver's.

    Returns ``None`` when no CUDA device can be inspected (ROCm/NPU/CPU
    builds, no driver, bad index) -- the caller then leaves cutlass alone.
    """
    try:
        import torch

        if not torch.cuda.is_available() or torch.version.cuda is None:
            return None
        major, minor = torch.cuda.get_device_capability(gpu_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not read compute capability of GPU %s: %s", gpu_id, exc)
        return None

    suffix = "a" if major >= 9 else ""
    return f"sm_{major}{minor}{suffix}"


def _patch_live_singleton(arch: str) -> bool:
    """Retarget an already-constructed ``CuTeDSL`` singleton to ``arch``.

    Returns True if a live instance was found and patched.  Importing cutlass
    here would defeat the purpose (a fresh singleton would read the env var we
    just set), so only ``sys.modules`` is consulted.
    """
    dsl_module = sys.modules.get("cutlass.base_dsl.dsl")
    if dsl_module is None:
        return False
    meta = getattr(dsl_module, "DSLSingletonMeta", None)
    instances = getattr(meta, "_instances", None) if meta is not None else None
    if not instances:
        return False

    patched = False
    for instance in list(instances.values()):
        envar = getattr(instance, "envar", None)
        if envar is None or getattr(envar, "arch", None) == arch:
            continue
        logger.debug(
            "Retargeting live %s DSL singleton: arch %s -> %s",
            getattr(instance, "name", "?"),
            envar.arch,
            arch,
        )
        envar.arch = arch
        patched = True
    return patched


def align_cute_dsl_arch(gpu_id: int) -> Optional[str]:
    """Point the CuTe DSL at ``gpu_id``'s architecture for this process.

    Call once per worker process, as soon as its ``gpu_id`` is known and
    before any CuTe kernel is compiled.  Returns the architecture string that
    is now in force, or ``None`` if nothing was done.
    """
    user_value = os.environ.get(_ENV_VAR)
    if user_value:
        # Explicit user intent wins -- but a stale singleton would still hold
        # the auto-detected value, so honour the override there too.
        _patch_live_singleton(user_value)
        return user_value

    arch = cute_dsl_arch_for_device(gpu_id)
    if arch is None:
        return None

    os.environ[_ENV_VAR] = arch
    patched = _patch_live_singleton(arch)
    logger.debug(
        "CUTE_DSL_ARCH=%s for GPU %s (live singleton %s)",
        arch,
        gpu_id,
        "retargeted" if patched else "not yet constructed",
    )
    return arch
