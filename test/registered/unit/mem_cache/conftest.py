"""Hermetic mem_cache runs SKIP the device-bound tests; they do not FAIL.

A permanently-red suite hides every new regression, which is the #380/#585
test-honesty class. Before this file a hermetic run
(``CUDA_VISIBLE_DEVICES=""``) ended **944 failed / 777 passed**, and the triage
found all 944 share ONE root: ``get_device()`` raising
``RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) or platform
plugin is available``. There were zero stub-drift failures, zero genuinely
broken paths and zero obsolete tests in that set -- red meant "no GPU here",
not "broken".

Two things made that worse than it needed to be:

* ``CustomTestCase._callTestMethod`` wraps every test in
  ``sglang.srt.utils.common.retry``, which re-raises a bare
  ``Exception("retry() exceed maximum number of retries.")`` WITHOUT chaining
  the cause. So the real error was discarded and 841 of the failures reported
  an opaque retry message instead of the one-line environmental reason.
* the same wrapper retried a deterministic environment failure several times
  each, which is most of the suite's ~4 minute hermetic runtime.

``retry`` does honour one exception: ``SkipTest`` is re-raised immediately and
never retried. So converting the environmental error into a skip AT ITS SOURCE
fixes the masking and the retry storm together, and needs no edit to any test.

Scope, deliberately narrow: the patch is installed only when there is genuinely
no accelerator, so a GPU run behaves exactly as before and any real failure
stays red. It fires only for tests that actually call ``get_device()`` -- the
777 that already pass never touch it and keep running.
"""

import unittest

import sglang.srt.utils as _utils
from sglang.srt.utils import common as _common

_REAL_GET_DEVICE = _common.get_device

_SKIP_REASON = (
    "requires a real accelerator: this mem_cache test constructs KV/mamba "
    "pools, which resolve a device through get_device(). The hermetic run "
    '(CUDA_VISIBLE_DEVICES="") has none, so this is an environment skip, not '
    "a code failure. Run on a GPU to exercise it."
)


def _accelerator_present() -> bool:
    try:
        _REAL_GET_DEVICE()
    except Exception:
        return False
    return True


if not _accelerator_present():

    def _get_device_or_skip(*args, **kwargs):
        raise unittest.SkipTest(_SKIP_REASON)

    # Patched on BOTH the defining module and the package namespace: the test
    # modules bind the name at import time (``from sglang.srt.utils import
    # get_device``), and conftest is imported before them, so both spellings
    # must already point at the skipping version.
    _common.get_device = _get_device_or_skip
    _utils.get_device = _get_device_or_skip
