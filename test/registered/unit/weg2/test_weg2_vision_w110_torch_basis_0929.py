"""W110 judges the stage's own torch-allocated residue (29.09., 27B z30j).

z30j run=1 logged ``vram_residue_mib=+44.0`` and W110 although every tower
tensor was released: the NVML process delta also moves when the caching
allocator keeps a segment or cuBLAS/cuDNN build a workspace on the first
encode. The verdict now reads ``torch.cuda.memory_allocated`` after minus
before; the NVML figure stays in the line as information.
"""
import logging

from sglang.srt.weg2 import vision_rank_runner as vrr
from sglang.srt.weg2 import vision_rank_stage as vrs

MIB = vrs.MIB


def _out(nvml=None, torch_=None):
    o = vrr.StageOutcome()
    o.residue_bytes = nvml
    o.torch_residue_bytes = torch_
    return o


def test_torch_delta_is_the_basis_when_measured():
    leak, basis = vrr.teardown_leak_bytes(_out(nvml=44 * MIB, torch_=0))
    assert leak == 0 and "torch" in basis


def test_nvml_is_the_fallback_without_torch_accounting():
    leak, basis = vrr.teardown_leak_bytes(_out(nvml=44 * MIB, torch_=None))
    assert leak == 44 * MIB and "NVML" in basis


def test_unmeasured_is_none():
    assert vrr.teardown_leak_bytes(_out())[0] is None


def test_z30j_shape_no_W110_but_residue_still_reported(caplog):
    # a released tower with an allocator segment kept: NVML +44, torch 0
    caplog.set_level(logging.INFO)
    vrr.log_outcome(_out(nvml=44 * MIB, torch_=0), ["weg2-1-1"], 1)
    assert vrr.W_TEARDOWN not in caplog.text
    assert "vram_residue_mib=+44.0" in caplog.text and "torch_residue_mib=+0.0" in caplog.text


def test_a_real_leak_logs_W110_with_its_basis(caplog):
    caplog.set_level(logging.INFO)
    vrr.log_outcome(_out(nvml=60 * MIB, torch_=50 * MIB), ["weg2-1-1"], 2)
    assert vrr.W_TEARDOWN in caplog.text and "torch allocated delta" in caplog.text


def test_a_real_leak_still_fires():
    # a tower tensor kept alive: torch +50 MiB
    leak, _ = vrr.teardown_leak_bytes(_out(nvml=60 * MIB, torch_=50 * MIB))
    assert leak > vrr.RESIDUE_TOLERANCE_BYTES


def test_own_torch_bytes_is_none_off_cuda():
    import torch
    assert vrr.own_torch_bytes(torch.device("cpu")) is None
