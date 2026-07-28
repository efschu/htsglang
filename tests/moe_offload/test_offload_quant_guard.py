# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the #268 fail-fast guard: the MoE expert-offload
installer must hard-reject GGUF-MoE / MoeWNA16 quant methods instead of
running MoEExpertOffloadCache.install() against a quant path with no
load-time offload half (GGUFUninitializedParameter only materializes in the
loader postprocess, #123; MoeWNA16's per-expert layout was never validated
against the offload cache's slice/fetch path).

No CUDA required -- ``assert_expert_offload_quant_supported`` is pure Python
(class-name dispatch, no tensor access), so it is exercised directly against
lightweight stand-ins for the real quant-method classes.

Run:  python -m pytest tests/moe_offload/test_offload_quant_guard.py -q
"""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "python"),
)

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    assert_expert_offload_quant_supported,
)


class _FakeGGUFMoEMethod:
    """Stand-in for sglang.srt.layers.quantization.gguf.GGUFMoEMethod.

    Matched by class name in the guard (to keep expert_offload.py import-
    light), so a same-named fake exercises the exact match path without
    pulling in the real GGUF quant module (which needs the optional
    ``gguf`` package installed) or CUDA.
    """


_FakeGGUFMoEMethod.__name__ = "GGUFMoEMethod"


class _FakeGGUFMoEAscendMethod:
    pass


_FakeGGUFMoEAscendMethod.__name__ = "GGUFMoEAscendMethod"


class _FakeMoeWNA16Method:
    pass


_FakeMoeWNA16Method.__name__ = "MoeWNA16Method"


class _FakeFp8MoEMethod:
    pass


_FakeFp8MoEMethod.__name__ = "Fp8MoEMethod"


class _FakeGPTQMarlinMoEScheme:
    pass


_FakeGPTQMarlinMoEScheme.__name__ = "GPTQMarlinMoEScheme"


class _FakeAWQMoEScheme:
    pass


_FakeAWQMoEScheme.__name__ = "AWQMoEScheme"


class _FakeUnquantizedFusedMoEMethod:
    pass


_FakeUnquantizedFusedMoEMethod.__name__ = "UnquantizedFusedMoEMethod"


@pytest.mark.parametrize(
    "unsupported_cls",
    [_FakeGGUFMoEMethod, _FakeGGUFMoEAscendMethod, _FakeMoeWNA16Method],
)
def test_rejects_unsupported_quant_methods(unsupported_cls):
    with pytest.raises(RuntimeError) as excinfo:
        assert_expert_offload_quant_supported(unsupported_cls(), layer_id=3)
    msg = str(excinfo.value)
    # Names the offending quant path.
    assert unsupported_cls.__name__ in msg
    # Names the supported alternatives, per #268 spec.
    assert "fp8" in msg
    assert "GPTQ" in msg
    assert "AWQ" in msg
    # Names the layer for a fail-fast message that is actionable.
    assert "layer_id=3" in msg


def test_rejects_unsupported_quant_method_without_layer_id():
    # layer_id is optional -- caller sites that don't have one yet must not
    # crash the guard itself.
    with pytest.raises(RuntimeError):
        assert_expert_offload_quant_supported(_FakeMoeWNA16Method())


@pytest.mark.parametrize(
    "supported_cls",
    [
        _FakeFp8MoEMethod,
        _FakeGPTQMarlinMoEScheme,
        _FakeAWQMoEScheme,
        _FakeUnquantizedFusedMoEMethod,
    ],
)
def test_allows_supported_quant_methods(supported_cls):
    # Must be a pure no-op (no raise) for every supported quant path.
    assert assert_expert_offload_quant_supported(supported_cls(), layer_id=0) is None


def test_allows_unknown_quant_method_by_default():
    # Deny-list semantics (#268): a quant method this guard has never heard of
    # is NOT rejected here -- it is presumed supported unless proven otherwise.
    # This matches the installer's existing self-guarding try/except, which
    # still catches genuine install failures for any quant path.
    class _SomeFutureQuantMethod:
        pass

    assert (
        assert_expert_offload_quant_supported(_SomeFutureQuantMethod(), layer_id=1)
        is None
    )
