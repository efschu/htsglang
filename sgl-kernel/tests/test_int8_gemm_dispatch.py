"""Hermetic guards for the INT8 GEMM architecture dispatch.

These run without a GPU and without a compiled ``sgl_kernel``: they read the
dispatch chain in ``csrc/gemm/int8_gemm_kernel.cu`` directly. The point is not
to re-test CUTLASS, it is to keep the arch coverage from silently regressing
when the file is rebased onto upstream -- the sm120 arm is fork-local, and its
absence shows up only as a runtime ``TORCH_CHECK_NOT_IMPLEMENTED`` on a 5090.
"""

import re
import sys
from pathlib import Path

import pytest

KERNEL_SOURCE = (
    Path(__file__).resolve().parent.parent / "csrc" / "gemm" / "int8_gemm_kernel.cu"
)


@pytest.fixture(scope="module")
def dispatch_body() -> str:
    """The body of ``int8_scaled_mm``, i.e. the runtime arch dispatch chain."""
    source = KERNEL_SOURCE.read_text()
    start = source.index("torch::Tensor int8_scaled_mm(")
    return source[start:]


def _branch_conditions(body: str) -> list[str]:
    return re.findall(r"(?:else )?if \((sm_version[^)]*)\)", body)


def test_source_exists():
    assert KERNEL_SOURCE.is_file(), KERNEL_SOURCE


def test_sm120_is_dispatched(dispatch_body):
    conditions = _branch_conditions(dispatch_body)
    assert any(">= 120" in condition for condition in conditions), (
        f"no sm120 arm in the int8 dispatch chain, only: {conditions}"
    )


def test_sm120_uses_the_ada_tile_table(dispatch_body):
    """sm120 must route through the CUTLASS 2.x IMMA path, not the sm90 c3x one.

    vLLM has no INT8 kernel for any Blackwell arch, so there is no c3x config to
    inherit; the sm89 table is the one that fits sm120's 100 KB shared-memory
    budget.
    """
    sm120_arm = dispatch_body[dispatch_body.index("sm_version >= 120") :]
    sm120_arm = sm120_arm[: sm120_arm.index("TORCH_CHECK_NOT_IMPLEMENTED")]

    assert "sm120_dispatch_shape" in sm120_arm
    assert "cutlass::arch::Sm80" in sm120_arm
    assert "cutlass::gemm::GemmShape<16, 8, 32>" in sm120_arm
    assert "sm90_dispatch_shape" not in sm120_arm

    full_source = KERNEL_SOURCE.read_text()
    forwarding = full_source[full_source.index("void sm120_dispatch_shape(") :]
    forwarding = forwarding[: forwarding.index("\n}\n")]
    assert "sm89_dispatch_shape" in forwarding


def test_both_output_dtypes_reach_sm120(dispatch_body):
    sm120_arm = dispatch_body[dispatch_body.index("sm_version >= 120") :]
    sm120_arm = sm120_arm[: sm120_arm.index("TORCH_CHECK_NOT_IMPLEMENTED")]
    assert "cutlass::bfloat16_t" in sm120_arm
    assert "cutlass::half_t" in sm120_arm


def test_datacenter_blackwell_stays_unimplemented(dispatch_body):
    """sm100/sm103 must keep falling through to the not-implemented error.

    Those parts have no classic IMMA path, so a broadened ``>= 100`` arm would
    compile and then fail at runtime instead of at dispatch time.
    """
    conditions = _branch_conditions(dispatch_body)
    assert not any(
        ">= 100" in condition or "== 100" in condition for condition in conditions
    ), f"sm100 must not be routed into an INT8 kernel: {conditions}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
