"""Hermetic mem_cache runs execute on CPU tensors; only the truly device-bound
tests skip.

History, and why this file changed shape twice
----------------------------------------------

Round 1 (#710). A hermetic run (``CUDA_VISIBLE_DEVICES=""``) ended **944 failed
/ 777 passed**, and the triage found all 944 share ONE root: ``get_device()``
raising ``RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) or
platform plugin is available``.  There were zero stub-drift failures, zero
genuinely broken paths and zero obsolete tests in that set -- red meant "no GPU
here", not "broken".  A permanently-red suite hides every new regression, which
is the #380/#585 test-honesty class, so ``get_device`` was patched at its source
to raise ``SkipTest`` instead.  ``retry`` (which ``CustomTestCase`` wraps every
test in) honours ``SkipTest`` and never retries it, so that fixed the masking
and the retry storm together.

Round 2 (#862) -- this file.  Turning red into skip cured the dishonesty but
left the suite DARK: measured on 4db09b72f5,
``test_unified_radix_cache_unittest.py`` reported **16 passed / 1506 skipped**
hermetically.  Nothing on the shipped ``UnifiedRadixCache`` class gated at the
desk; the skips read like coverage.

The skip was one step too wide.  ``get_device()`` is asked for a *device
string*; the pools then allocate ordinary ``torch`` tensors on it.  The radix
tree, its prefix matching, node splitting, eviction ordering and index
bookkeeping are device-agnostic -- they compute on Python objects and on tensor
*indices*, not on accelerator kernels.  Handing those tests ``"cpu"`` runs the
SAME assertions against the SAME code path; it does not substitute a weaker
check.  Measured: 642 passed / 212 failed / 648 skipped.

Two properties make that safe rather than optimistic:

* A test that genuinely needs device semantics FAILS on CPU -- it cannot pass
  vacuously.  There is not one ``if device == "cuda"``-guarded assertion in the
  suite (grep: zero hits for ``torch.cuda`` / ``is_cuda`` in the test file), so
  no assertion can be silently stepped over.
* The 648 that still skip are config-matrix pruning (``requires SWA``,
  ``requires page_size=1 Full+Mamba``, ...) raised from inside the test bodies
  against the parametrised fixture.  They skip identically ON a GPU; they are
  not darkness.

The 212 that fail on CPU are ONE cluster, not a spread: every one of them drives
a HiCache host<->device KV transfer and lands on ``sgl_kernel.kvcacheio``.  Those
are real CUDA kernels; there is no CPU equivalent and pretending otherwise would
be exactly the substitution this file refuses to make.  They are converted into
a precise environment skip below, naming the kernel that is missing, so the
window list says *which* kernel a GPU run would exercise.

A production defect this uncovered -- NOT fixed here, it is not a test change
--------------------------------------------------------------------------

``memory_pool_host.py:35``, ``pool_host/mha.py:50``, ``pool_host/mla.py:37``
each read::

    if _is_cuda or _is_hip:
        try:
            from sgl_kernel.kvcacheio import transfer_kv_all_layer_direct_lf_pf, ...
            _has_sgl_kvcacheio = True
        except ImportError:
            transfer_kv_all_layer_direct_lf_pf = None
            ...

The ``except ImportError`` branch defines the names as ``None`` so call sites can
guard on them -- but there is no ``else`` branch.  On any platform where
``_is_cuda`` and ``_is_hip`` are both False (NPU, XPU, MPS, CPU, or simply
``CUDA_VISIBLE_DEVICES=""``) the names are never bound at all, and the call sites
at ``memory_pool_host.py:581/1113/1489/2148``, ``pool_host/mla.py:450`` and
``pool_host/mha.py:473`` raise ``NameError: name
'transfer_kv_all_layer_direct_lf_pf' is not defined`` instead of the intended
``None`` guard.  Same shape in three files -- a class, not an instance.  The fix
belongs in those modules (bind the seven names to ``None`` unconditionally, or
hoist the fallback out of the ``if``); this conftest only makes the gap legible
as a skip so the test suite is not held hostage to it.
"""

import importlib
import unittest

import sglang.srt.utils as _utils
from sglang.srt.utils import common as _common

_REAL_GET_DEVICE = _common.get_device

# The seven names imported together at the top of each pool_host module.
_KVCACHEIO_SYMBOLS = (
    "transfer_kv_all_layer_direct_lf_pf",
    "transfer_kv_all_layer_mla",
    "transfer_kv_all_layer_mla_lf_pf",
    "transfer_kv_direct",
    "transfer_kv_per_layer_direct_pf_lf",
    "transfer_kv_per_layer_mla",
    "transfer_kv_per_layer_mla_pf_lf",
)

_KVCACHEIO_MODULES = (
    "sglang.srt.mem_cache.memory_pool_host",
    "sglang.srt.mem_cache.pool_host.mha",
    "sglang.srt.mem_cache.pool_host.mla",
)

_KVCACHEIO_SKIP_REASON = (
    "requires the sgl_kernel.kvcacheio CUDA transfer kernel {name}(): this test "
    "drives a HiCache host<->device KV transfer, which has no CPU equivalent. "
    'The hermetic run (CUDA_VISIBLE_DEVICES="") reaches the call site with the '
    "kernel absent, so this is an environment skip, not a code failure. Run on a "
    "GPU to exercise it."
)


def _accelerator_present() -> bool:
    try:
        _REAL_GET_DEVICE()
    except Exception:
        return False
    return True


def _kvcacheio_stub(name: str):
    def _raise_skip(*args, **kwargs):
        # retry() re-raises SkipTest immediately and never retries it, so this
        # reaches unittest as a skip even from inside CustomTestCase's wrapper.
        raise unittest.SkipTest(_KVCACHEIO_SKIP_REASON.format(name=name))

    _raise_skip.__name__ = name
    return _raise_skip


# Scope, deliberately narrow: installed only when there is genuinely no
# accelerator, so a GPU run behaves exactly as before and any real failure stays
# red.
if not _accelerator_present():

    def _get_device_cpu(*args, **kwargs):
        return "cpu"

    # Patched on BOTH the defining module and the package namespace: the test
    # modules bind the name at import time (``from sglang.srt.utils import
    # get_device``), and conftest is imported before them, so both spellings
    # must already point at the CPU-returning version.
    _common.get_device = _get_device_cpu
    _utils.get_device = _get_device_cpu

    for _modname in _KVCACHEIO_MODULES:
        try:
            _mod = importlib.import_module(_modname)
        except Exception:  # pragma: no cover - import guarded elsewhere
            continue
        for _sym in _KVCACHEIO_SYMBOLS:
            # Covers both halves of the production gap: the name missing
            # entirely (no else-branch) and the name bound to None (the
            # ImportError branch).
            if getattr(_mod, _sym, None) is None:
                setattr(_mod, _sym, _kvcacheio_stub(_sym))
