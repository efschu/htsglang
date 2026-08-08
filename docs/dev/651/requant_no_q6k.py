#!/usr/bin/env python
"""#651: derive a Q6_K-free variant of a GGUF checkpoint (gfx1103 containment).

On gfx1103 (Radeon 780M) every GPU op that touches Q6_K except linear MMQ is
nondeterministically wrong (dequantize, MMVQ, ggml_moe_a8, ggml_moe_a8_vec --
8-run harnesses, 2026-08-08). Qwen3.6-35B-A3B-UD-Q4_K_M carries exactly four
Q6_K tensors: the lm_head (contained via the MMQ pin) and three MoE expert
down-projection stacks (blk.34/38/39) whose kernels have no clean Q6_K path at
all. Requantizing those four to Q8_0 -- a type validated byte-identical across
runs on this GPU, and already the file's dominant expert type -- removes the
defective type from the GPU entirely.

Quality: the requantization error (Q8_0 on real weight scales, ~5e-4 absmax)
is two to three orders below the noise it removes (up to 1.1e-1 run spread).
Size cost: +~250 MiB (Q8_0 8.5 bpw vs Q6_K 6.5625 bpw on ~854 MiB packed).

Usage:
    python requant_no_q6k.py <src.gguf> <dst.gguf>
"""

import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, Keys, quants
from gguf.constants import GGMLQuantizationType as QT

TARGET_TYPE = QT.Q8_0
BROKEN_TYPE = QT.Q6_K


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    reader = GGUFReader(src)

    arch = reader.fields[Keys.General.ARCHITECTURE].contents()
    writer = GGUFWriter(dst, arch=arch, endianess=reader.endianess)

    for field in reader.fields.values():
        if field.name == Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    targets = [t.name for t in reader.tensors if t.tensor_type == BROKEN_TYPE]
    print(f"requantizing {len(targets)} {BROKEN_TYPE.name} tensors -> "
          f"{TARGET_TYPE.name}: {targets}")

    # Requantize the targets up front (one at a time; the fp32 intermediate of
    # the largest is ~2 GiB).
    requantized: dict[str, np.ndarray] = {}
    for t in reader.tensors:
        if t.name not in targets:
            continue
        dense = quants.dequantize(t.data, BROKEN_TYPE)
        requantized[t.name] = quants.quantize(dense, TARGET_TYPE)
        rt = np.abs(quants.dequantize(requantized[t.name], TARGET_TYPE) - dense).max()
        print(f"  {t.name}: {t.data.nbytes / 2**20:.0f} MiB -> "
              f"{requantized[t.name].nbytes / 2**20:.0f} MiB, "
              f"roundtrip max|d| {rt:.3e}")
        del dense

    for t in reader.tensors:
        if t.name in requantized:
            q = requantized[t.name]
            writer.add_tensor_info(t.name, q.shape, q.dtype, q.nbytes, TARGET_TYPE)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype,
                                   t.data.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    done = 0
    for t in reader.tensors:
        data = requantized.get(t.name, t.data)
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
        done += data.nbytes
        if done % (1 << 31) < data.nbytes:
            print(f"  written {done / 2**30:.1f} GiB")

    writer.close()
    print(f"done: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
