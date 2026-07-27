"""Build a synthetic TP=1 HiCache 'file' store at a REAL model geometry.

Offline round-trip harness for the handover umsharder (#261): the byte gate
does not need a GPU or a checkpoint, only files of exactly the sizes a real
boot writes. Both geometries are the ones measured on the rig.
"""

import os
import random
import sys

GEOMETRIES = {
    # (linear layers, value heads, head_dim, state, key_dim, value_dim,
    #  conv_width, kv page bytes, n kv pages)
    "27b": dict(
        layers=48,
        heads=48,
        head_dim=128,
        state=128,
        key_dim=2048,
        value_dim=6144,
        conv_width=3,
        kv_page=32768,
        n_kv=287,
        suffix="Qwen3.6-27B_0123456789abcdef",
    ),
    "moe": dict(
        layers=30,
        heads=32,
        head_dim=128,
        state=128,
        key_dim=2048,
        value_dim=4096,
        conv_width=3,
        kv_page=10240,
        n_kv=192,
        suffix="Qwen3.6-35B-A3B_fedcba9876543210",
    ),
}


def main(which: str, directory: str) -> int:
    g = GEOMETRIES[which]
    os.makedirs(directory, exist_ok=True)
    conv_dim = g["key_dim"] * 2 + g["value_dim"]
    blob = g["layers"] * (
        g["heads"] * g["head_dim"] * g["state"] * 2 + conv_dim * g["conv_width"] * 2
    )
    rnd = random.Random(1234)
    chunk = bytes(rnd.getrandbits(8) for _ in range(1 << 20))
    with open(
        os.path.join(directory, f"deadbeef.mamba_{g['suffix']}_0_1.bin"), "wb"
    ) as f:
        left = blob
        while left:
            n = min(len(chunk), left)
            f.write(chunk[:n])
            left -= n
    for i in range(g["n_kv"]):
        page = bytes((i * 13 + j) % 256 for j in range(g["kv_page"]))
        with open(
            os.path.join(directory, f"{i:016x}_{g['suffix']}_0_1.bin"), "wb"
        ) as f:
            f.write(page)
    print(
        f"{which}: {g['n_kv']} KV pages of {g['kv_page']} B + 1 mamba blob of {blob} B"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
