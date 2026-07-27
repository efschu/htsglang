"""Tree-wide pytest guards.

Currently: neutralize process-wide torch state leaked during collection.
"""

import sys


def pytest_collection_finish(session):
    """Reset the torch default device before any test runs.

    test/registered/unit/batch_invariant_ops/test_batch_invariant_ops.py
    calls ``torch.set_default_device(<accelerator>)`` at module scope, so it
    executes during pytest collection and stays set for the whole process.
    On a machine without a usable accelerator (e.g. CUDA build with
    ``CUDA_VISIBLE_DEVICES`` pointing at no device) every later tensor
    construction through the default device then fails in whichever test
    modules happen to be collected afterwards -- visible as exhausted
    ``retry()`` loops far from the root cause, with the victim set changing
    whenever collection order changes (task #249: 12 phantom regressions in
    a merge validation run, all green standalone).

    ``torch.set_default_device(None)`` restores the pristine state, so the
    reset is safe when nothing leaked.  ``torch.get_default_device()`` is
    deliberately not consulted first: it constructs a tensor on the default
    device and would itself crash on the polluted no-GPU path.

    Runtime leaks (``setUpClass`` setting the default device without a
    teardown reset) are outside this hook's reach; they only bite GPU runs
    and are tracked separately.
    """
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "set_default_device"):
        torch.set_default_device(None)


def pytest_collectstart(collector):
    """Reset the torch default device between collection steps as well.

    The module-scope ``set_default_device`` described above pollutes not only
    the tests that run later but also the *collection imports* of every file
    collected after it: a module whose import chain constructs a tensor (e.g.
    ``compressed_tensors``' nvfp4 helpers, reached through
    ``sglang.srt.layers.quantization``) dies with ``No CUDA GPUs are
    available`` before ``pytest_collection_finish`` ever runs.  Which files
    are hit is a function of import-cache state, so adding a test file or
    making an import lazy can create new victims.  Resetting at every
    collector keeps collection imports on pristine state; the reset is a
    no-op when nothing leaked.
    """
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "set_default_device"):
        torch.set_default_device(None)
