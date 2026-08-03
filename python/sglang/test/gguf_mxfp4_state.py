# SPDX-License-Identifier: Apache-2.0
"""Force the GGUF MXFP4 dispatch state a test needs, on any wheel.

#398 gave the wheel native MXFP4 (ggml type 39) kernels, and
``sglang.srt.layers.quantization.gguf`` evaluates their presence ONCE at import
into ``MXFP4_NATIVE``. Everything downstream follows that flag: the three GGUF
type sets, the DeepSeek-4 executability gate, and
``gguf_mxfp4_repack``, which becomes the identity when the kernels are native.

That leaves two distinct production paths, and a test has to say which one it
means:

* the **native path** -- what this rig serves today. MXFP4 tensors are executed
  as they lie on disk and the repack never runs.
* the **repack path** -- MXFP4 rewritten to Q5_0 at load time. Still shipped and
  still reachable: the wheel is pinned separately from the source
  (rig-runbook 2.1), so a tree carrying #398 can run an older wheel, and
  ``SGLANG_GGUF_MXFP4_NATIVE=0`` is the standing A/B lever against the native
  path.

Both states are produced IN-PROCESS here, by the env lever plus a module
reload, so a test of either path is deterministic on every wheel. That is
deliberately not a capability skip: a skip keyed on "this wheel has no native
export" would sleep forever on every machine that has one, and the repack path
would silently lose its coverage on exactly the builds that still ship it
(test honesty, audit #506 axis 4).
"""

from __future__ import annotations

import contextlib
import importlib
import os
from typing import Iterator

import torch

#: The lever `_mxfp4_kernels_present` reads (``gguf.py``); first character only.
_ENV = "SGLANG_GGUF_MXFP4_NATIVE"


def _reload_gguf():
    """Re-evaluate ``MXFP4_NATIVE`` and the type sets derived from it."""
    import sglang.srt.layers.quantization.gguf as g

    return importlib.reload(g)


def wheel_exports_native_mxfp4() -> bool:
    """Whether the INSTALLED wheel carries the #398 kernels.

    Asked of the operator table rather than of ``MXFP4_NATIVE``, so the answer
    is the wheel's property and not the current state of the env lever. The
    import is load-bearing: ``torch.ops.sgl_kernel`` resolves lazily, so the
    namespace has to be populated before ``hasattr`` means anything.
    """
    import sglang.srt.layers.quantization.gguf  # noqa: F401

    try:
        return hasattr(torch.ops.sgl_kernel, "ggml_mxfp4_native")
    except Exception:
        return False


@contextlib.contextmanager
def _forced(value: str) -> Iterator[None]:
    previous = os.environ.get(_ENV)
    os.environ[_ENV] = value
    try:
        _reload_gguf()
        yield
    finally:
        if previous is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = previous
        _reload_gguf()


def repack_path() -> contextlib.AbstractContextManager[None]:
    """Run the block with MXFP4 unexecutable, i.e. the load-time repack live.

    Works on a native wheel too -- the env lever wins over a present op, which
    is the whole point of it being the A/B lever.
    """
    return _forced("0")


@contextlib.contextmanager
def native_path() -> Iterator[None]:
    """Run the block with MXFP4 natively executable, i.e. the repack a no-op.

    On a wheel that predates #398 the marker op is registered for the duration
    of the block. The probe only asks whether the op EXISTS, so registering an
    equivalent schema reproduces a post-#398 wheel for everything the python
    dispatch decides.
    """
    lib = None
    if not wheel_exports_native_mxfp4():
        lib = torch.library.Library("sgl_kernel", "FRAGMENT")
        lib.define("ggml_mxfp4_native() -> int")
        lib.impl("ggml_mxfp4_native", lambda: 1, "CUDA")
    try:
        with _forced("1"):
            yield
    finally:
        if lib is not None:
            lib._destroy()


class ForcesRepackPath:
    """unittest mixin: every test in the case runs with the repack path live.

    Mix in BEFORE ``TestCase`` so the ``setUp``/``tearDown`` chain reaches it.
    """

    def setUp(self):
        super().setUp()
        self._mxfp4_path_state = repack_path()
        self._mxfp4_path_state.__enter__()

    def tearDown(self):
        self._mxfp4_path_state.__exit__(None, None, None)
        super().tearDown()


class ForcesNativePath:
    """unittest mixin: every test in the case runs with MXFP4 native."""

    def setUp(self):
        super().setUp()
        self._mxfp4_path_state = native_path()
        self._mxfp4_path_state.__enter__()

    def tearDown(self):
        self._mxfp4_path_state.__exit__(None, None, None)
        super().tearDown()
