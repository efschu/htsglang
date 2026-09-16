"""Minachist INT4/INT6 (AutoRound -> compressed-tensors pack-quantized): the
6-bit groups are stored as a dense little-endian bitstream (32/6 values per
int32; measured [10240, 480] for 2560 inputs) and are widened losslessly to
the 8-bit packing the Marlin uint8b128 path computes. Reference packer and
roundtrips here, plus the loader's packed-extent arithmetic with a Fraction
pack factor."""

from fractions import Fraction

import pytest
import torch

from sglang.srt.layers.parameter import _exact_div
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    unpack_dense_subbyte,
    widen_dense_packed_to_8bit,
)


def _pack_dense_reference(raw: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-serial reference: element e -> bits [e*bits, e*bits+bits) of the row."""
    rows, n = raw.shape
    words = -(-(n * bits) // 32)
    out = torch.zeros(rows, words, dtype=torch.int64)
    for e in range(n):
        v = raw[:, e].to(torch.int64)
        for b in range(bits):
            bit = (v >> b) & 1
            pos = e * bits + b
            out[:, pos // 32] |= bit << (pos % 32)
    # reinterpret as int32 (two's complement for the high bit)
    return torch.where(out >= 1 << 31, out - (1 << 32), out).to(torch.int32)


@pytest.mark.parametrize("bits,n", [(6, 64), (6, 2560), (5, 96), (3, 32), (4, 64), (8, 16)])
def test_unpack_dense_roundtrips_the_reference_packer(bits, n):
    g = torch.Generator().manual_seed(bits * 1000 + n)
    raw = torch.randint(0, 1 << bits, (3, n), generator=g)
    packed = _pack_dense_reference(raw, bits)
    assert packed.shape == (3, -(-(n * bits) // 32))
    got = unpack_dense_subbyte(packed, bits, n)
    assert torch.equal(got, raw.to(torch.int32))


def test_widen_6bit_to_8bit_keeps_the_signed_values():
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

    g = torch.Generator().manual_seed(7)
    raw = torch.randint(0, 64, (4, 256), generator=g)  # unsigned 6-bit
    packed6 = _pack_dense_reference(raw, 6)
    packed8 = widen_dense_packed_to_8bit(packed6, src_bits=6, in_features=256)
    assert packed8.shape == (4, 64) and packed8.dtype == torch.int32
    # compressed-tensors' own 8-bit unpacker (offset 128) must give q = raw - 32
    q8 = unpack_from_int32(packed8, 8, torch.Size((4, 256)), packed_dim=1)
    assert torch.equal(q8.to(torch.int64), raw - 32)


def test_widening_is_the_same_dequant_under_the_same_scale():
    g = torch.Generator().manual_seed(11)
    raw = torch.randint(0, 64, (2, 128), generator=g)
    scale = torch.rand(2, 2) * 0.01 + 0.001  # two groups of 64
    ref = (raw - 32).float().view(2, 2, 64) * scale.unsqueeze(-1)
    packed8 = widen_dense_packed_to_8bit(_pack_dense_reference(raw, 6), 6, 128)
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

    q8 = unpack_from_int32(packed8, 8, torch.Size((2, 128)), packed_dim=1).float()
    got = q8.view(2, 2, 64) * scale.unsqueeze(-1)
    assert torch.allclose(got, ref)


def test_exact_div_with_fraction_pack_factor():
    assert _exact_div(2560, Fraction(32, 6)) == 480
    assert _exact_div(384, Fraction(32, 6)) == 72
    assert _exact_div(2560, 8) == 320
    with pytest.raises(ValueError, match="not integral"):
        _exact_div(2568, Fraction(32, 6))


def test_scheme_widens_six_bits_to_the_8bit_kernel_type():
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
        CompressedTensorsWNA16,
    )

    s = CompressedTensorsWNA16(strategy="group", num_bits=6, group_size=64, symmetric=True)
    assert s.src_num_bits == 6 and s.pack_factor == Fraction(32, 6)
    assert s.quant_type.size_bits == 8
    with pytest.raises(ValueError, match="symmetric"):
        CompressedTensorsWNA16(strategy="group", num_bits=6, group_size=64, symmetric=False)
