# `306_ratio_probe` — lossless cold-tier compression, ratio falsification

Desk probe for task #306, WAVE 1 item 3 of
`docs/dev/ROADMAP_456_matrix_execution.md`. No GPU, no build: it measures
whether lossless compression is worth building for the #407 cold tiers at all,
on real asset samples from this box. Verdicts and raw numbers:
`docs/dev/ANALYSE_306_lossless_ratio.md`.

## Contents

| file | what it does |
|---|---|
| `blocks.py` | ggml block-struct layouts (cited to `sgl-kernel/csrc/quantization/gguf/ggml-common.h` line-by-line) and the three byte permutations: semantic `plane`, mechanical `stride`, sub-byte `nibble`. Each has an exact inverse. |
| `sample_extract.py` | seed-fixed extraction of real asset slices into a scratch dir plus a provenance manifest. Reads RAW QUANTISED BYTES at the tensor's file offset — never dequantises. |
| `ratio_probe.py` | the `codec x layout` matrix: ratio, compress MB/s, decompress MB/s (with and without the inverse permutation). Every round trip is verified byte-identical. |
| `entropy_ceiling.py` | order-0 byte entropy per payload and per plane (the information-theoretic ceiling for any memoryless coder), plus `zstd --ultra -22 --long` and `xz -9e` as the maximum-effort arm. |
| `parallel_decomp.py` | decompress rate vs worker count on independent 4 MiB frames. Establishes `D_max`, which fixes the serial break-even ratio for every link — the 8-worker figure in the main sweep is not an arbitrary tuning point. |
| `image_whole.py` | whole-file streaming measurement for the disk-image class, separating the zero-page (sparse-write) share from the residual codec gain. A chunk sample cannot answer this: the hibernate image is not homogeneous. |
| `fs_ratio.sh` | what the FILESYSTEM is already compressing, per file, from `du`/`stat` block accounting. Run this before pricing a codec for any disk tier. |
| `report.py` | aggregates `results.jsonl` into the markdown verdict tables, applying the break-even formulas against the tree's measured link rates. |
| `results/` | the committed raw data: `samples.json` (provenance + SHA-256 per sample), `results.jsonl` (2 064 rows), `ceiling.jsonl`, `parallel_decomp.json`, `image_whole.json`, `report_generated.md`. |

## Running

Any interpreter with `numpy`, `zstandard` and `gguf`. The probe was run with
`/spinning/htsglang-gpu/.venv/bin/python` (read-only use; nothing installed
into it).

```sh
V=/spinning/htsglang-gpu/.venv/bin/python
D=/spinning/wt-306-ratio/.probe-data

$V sample_extract.py  --out  "$D"            # 112 samples, 16 MiB each, seed 306
$V ratio_probe.py     --data "$D"            # ~52 min single-threaded, writes results.jsonl
$V entropy_ceiling.py --data "$D"            # ~10 min, writes ceiling.jsonl
$V parallel_decomp.py --data "$D"            # ~5 min, writes parallel_decomp.json
$V image_whole.py                            # ~15 min over the 7.18 GB #89 image
$V report.py          --data "$D"            # markdown tables to stdout
sh fs_ratio.sh <any asset file> ...          # what the filesystem already compresses
```

Run them one at a time: the throughput numbers are the load-bearing half of the
verdict, and a concurrent arm would contaminate them.

The scratch dir holds ~1.8 GiB of extracted samples and is deliberately outside
git. `samples.json` carries every sample's source path, tensor name, byte
offset and SHA-256, so any row can be re-derived without re-running the sweep.

## Discipline notes

* Rates are MB/s = 1e6 B/s, matching the GB/s link figures the verdicts compare
  against.
* `decomp_mbs` includes the inverse byte permutation for non-`raw` layouts —
  a de-interleaved asset is not usable until it is re-interleaved, so that cost
  belongs to the decompress side. `decomp_codec_mbs` is the codec alone.
* A layout is a permutation compressed as ONE frame, not one frame per plane:
  separate frames would forfeit the shared window, which is the entire point of
  grouping like-entropy bytes.
* Every `(sample, layout, codec)` triple asserts `restored == original`; a
  mismatch aborts the run rather than being reported.
