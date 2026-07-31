# Task 333 -- Prior-art reading: styler00dollar/VSGAN-tensorrt-docker

Recon-only task (CPU-side, no GPU window spent), source for the #333 Class-3
utility-lane design (video-enhance stream server on the htsglang fork).
Written to stand on its own for the #333 design team -- no assumption that
the reader has this conversation's context.

Sources: `styler00dollar/VSGAN-tensorrt-docker` README (main branch, fetched
2026-07-31), its `releases/tag/models` asset list (GitHub API, 328 assets),
`AmusementClub/vs-mlrt` README, `HolyWu/vs-rife` `vsrife/__init__.py`,
`xinntao/Real-ESRGAN` README, and one open GitHub issue. No wiki exists for
VSGAN-tensorrt-docker; no Discussions tab content was surfaced.

## 0. What the project actually is

VSGAN-tensorrt-docker is **not its own inference engine**. It is a
VapourSynth filter-chain repo -- Python scripts (`inference.py`, `main.py`,
`inference_config.py`) plus a curated Docker image -- that wires together
other people's VapourSynth plugins:

- `AmusementClub/vs-mlrt` (`vstrt`/`vsort`/`vsov`/`vsncnn`/`vsmigx`) for raw
  TensorRT / ONNX Runtime / OpenVINO / ncnn / MIGraphX execution,
- `HolyWu/vs-realesrgan`, `HolyWu/vs-rife`, `HolyWu/vs-animesr`,
  `TNTwise/vs-spandrel`, `routineLife1/VS-DRBA`, `Mr-Z-2697/ddfi-rife` for
  model-specific wrappers (some of these use PyTorch + `torch_tensorrt`
  directly instead of vs-mlrt),
- `AkarinVS/L-SMASH-Works` / `vapoursynth/bestsource` for decode,
- `pifroggi/vs_temporalfix`, `pifroggi/vs_colorfix` for post-filters.

Its own contribution is: a curated model zoo (own GitHub Release, see 1
below), example glue scripts per model, `trtexec` command recipes, and a
Docker image bundling the whole plugin set with matched library versions.
This framing matters for the reuse verdict in section 5.

## 1. Model / engine catalog

**Catalog source of record**: the `models` release
(`github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models`, 328
assets, commit `f235b5b`, tag body: "Just a place to store models. Sources
are in the README."). It is an uncurated flat dump -- `.pth` (PyTorch
source) and `.onnx` (exported) side by side, no manifest, no per-file
description beyond the filename convention.

**Filename convention** (self-consistent, no separate schema doc): `opNN` =
ONNX opset version (16-20 seen across the set), `fp16` suffix = exported at
half precision (absence = fp32), `clamp` = output clamped to [0,1] in-graph,
`onnxslim`/`sim` = post-export graph-simplification pass applied,
resolution suffixes (`720p`/`1080p`) on a few models (SAFMN, SCUNet) mean
the ONNX was exported with that shape baked in (static), not a runtime
option.

**Model families present** (by architecture, source repo in README):

| family | examples in release | arch source |
|---|---|---|
| SRVGGNetCompact (ESRGAN-compact) | `RealESRGAN_x4plus_anime_6B`, `RealESRGANv2-animevideo-xsx{2,4}`, `realesr-animevideov3`, `realesr-general-wdn-x4v3`, `AnimeJaNai V2/V3/HD` (Compact/UltraCompact/SuperUltraCompact/ESRGAN variants), `sudo_UltraCompact`, `sudo_shuffle_cugan`, `1x_Anime1080Fixer` | `xinntao/Real-ESRGAN` |
| SPAN | `ModernSpanimationV1/V1.5/V2`, `2x_AniSD_*_SPAN_*` | `hongyuanyu/SPAN` |
| Real-CUGAN | `cugan_pro-*`, `cugan_up{2,3,4}x-latest-*` (conservative/denoise/no-denoise variants) | `bilibili/ailab/Real-CUGAN` |
| waifu2x | `waifu2x_{anime_style_art,cunet,photo,upconv_7,upresnet10}_*` | mirrored from `AmusementClub/vs-mlrt` model releases, not re-trained here |
| DPIR (denoise/deblock) | `dpir_drunet_{color,gray}`, `dpir_drunet_deblocking_{color,grayscale}` | `cszn/DPIR`, 4-channel input (RGB + noise-level map), fp32 only (`op9`) |
| SAFMN | `SAFMN_L_Real_LSDIR_x{2,4}-v2` | `sunny2109/SAFMN` |
| APISR | `apisr_grl_4x` | `kiteretsu77/apisr` |
| Adore | `2x_Adore_renarchi` | `renarchi/Re-SISR` |
| AniScale2 | `2x_AniScale2_Omni_i16_40K` | `Sirosky/Upscale-Hub` |
| Frame interpolation (RIFE family) | via `HolyWu/vs-rife`, not bundled as static ONNX in this release (RIFE ships as `.pkl`/PyTorch state through the vs-rife plugin, not exported ONNX in this catalog) | `hzwer/Practical-RIFE` |
| Frame interpolation (GMFSS/GMFupSS) | `GMFSS_fortuna_*`, `GMFSS_union_*`, `GMFupSS_*` (`.pkl`, PyTorch) | — |
| Shot-boundary / scene-cut classifiers | `sc_efficientnetv2b0*`, `sc_efficientformerv2*`, `sc_swinv2_small*`, `sc_maxvit*`, `sc_davit_small*`, `sc_mobilevitv2*`, `sc_autoshot`, `sc_dists*`, `sc_shift_lpips_*` -- a large, somewhat idiosyncratic bench of classifier/metric backbones for scene-change and quality-metric use, not core to #333 | mixed sources, see README credits |

**Precision**: every family that has an `_fp16` variant is exported as
**two separate static-precision ONNX graphs** (fp16 and fp32), not one
graph with a runtime dtype switch. There is **no int8 or fp8 ONNX asset
anywhere in the 328-file catalog**, and grepping the full README (615
lines) for `int8` returns zero matches. The project's own `trtexec` recipe
(`--bf16 --fp16 ... --builderOptimizationLevel=5`) never passes `--int8` or
a calibration cache flag. TensorRT's own PTQ int8 calibration path is not
exercised anywhere in the repo. This is itself the finding for the
INT8-regression question -- see section 3.

**Engine-build recipe** (the concrete reusable artifact): a single
`trtexec` command template, applied per model with only the shape triplet
changed:

```
trtexec --bf16 --fp16 --onnx=model.onnx \
  --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 \
  --maxShapes=input:1x3x1080x1920 --saveEngine=model.engine \
  --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference \
  --useCudaGraph --noDataTransfers --builderOptimizationLevel=5 --infStreams=N
```

Engines are explicitly declared non-portable: "Engines are system specific,
don't use across multiple systems" -- i.e. rebuild-per-host is the
documented expectation, not an oversight to work around.

## 2. Pipeline / parallelization approach -- and the Regime-A/B gap

**Chain**: decode (`lsmash`/`bestsource`/`ffms2`) -> `resize.Bicubic` to
`RGBH`/`RGBS` -> one or more `core.trt.Model()` / `core.ort.Model()` /
wrapper (`realesrgan()`, `rife()`, ...) calls in sequence -> `resize.Bicubic`
back to `YUV420P{8,10}` -> piped to `ffmpeg` via `vspipe -c y4m inference.py
- | ffmpeg -i pipe: out.mkv`. This is a single-process, single-pass
VapourSynth graph; concurrency inside one GPU comes from `num_streams`
(CUDA streams for pipelined overlap of multiple in-flight frames on the
*same* engine/device context) and `batch_size` (multiple frames per
inference call) -- both tuning knobs on a single fixed device, not a
distribution mechanism.

**Multi-GPU**: exactly one pattern, credited to a community contributor
("Thanks to tepete who figured it out"), reproduced here in full because it
is the entire mechanism:

```python
stream0 = core.std.SelectEvery(core.trt.Model(clip, engine_path=..., num_streams=2, device_id=0), cycle=3, offsets=0)
stream1 = core.std.SelectEvery(core.trt.Model(clip, engine_path=..., num_streams=2, device_id=1), cycle=3, offsets=1)
stream2 = core.std.SelectEvery(core.trt.Model(clip, engine_path=..., num_streams=2, device_id=2), cycle=3, offsets=2)
clip = core.std.Interleave([stream0, stream1, stream2])
```

This is a **static, equal-share, modulo-N frame round-robin** across N
devices, set up once by hand in the script (`cycle=N` hard-codes the GPU
count and split ratio). There is:

- no capacity weighting -- a 5090 and a 3080 given `cycle=3` each get
  exactly 1/3 of frames, full stop; the user manually picks a different
  cycle/offset pattern by hand if they want an uneven split, and there is
  no guidance or mechanism in the repo for *how* to pick it,
- no notion of GPU-pair proximity/interconnect cost (no PCIe topology or
  NVLink awareness at all -- frames assigned round-robin regardless of
  which device is "close" to which decode/encode stage),
- no dynamic rebalancing -- if one device stalls or a frame is
  disproportionately expensive (this project's frames are fixed-size
  images, so that specific risk doesn't apply here, but the general
  mechanism has no feedback loop),
- device selection is per-node (`device_id=N` baked into each branch of a
  hand-written Python script), not a config-driven placement decision.

**This is precisely the gap our #333 planner is designed to fill.** Every
one of the four items above (capacity-weighted distribution instead of
equal modulo split, pair/topology-aware cost instead of blind round-robin,
config-driven rather than hand-edited-per-topology placement, and a
principled way to pick the split ratio at all) is absent from VSGAN's
model. VSGAN solves "make N identical devices cooperate on interleaved
frames" with the simplest correct mechanism (equal round-robin,
frame-granular parallelism -- reasonable given SR/interpolation frames are
independent, embarrassingly-parallel units); it does not solve "make
heterogeneous devices cooperate proportionally to their throughput and
their position in the pipeline," which is the actual #333 rig shape (mixed
5090/3080 class, and a stream server has to make this decision
automatically, not via a hand-edited script per deployment). Net: VSGAN
confirms frame-level parallelism is the right granularity for this
workload class, but contributes nothing to the actual placement/pricing
logic our planner needs -- that is genuinely novel work here, not
something to port.

## 3. Format chain -- colorspace, precision, and the INT8 question

**YUV<->RGB and precision**: handled entirely by VapourSynth's own
`resize.Bicubic`/zimg plugin, *outside* the TensorRT engine -- there is no
fusion of colorspace conversion into the TRT graph. The convention, quoted
directly from the README:

> "If you use the FP16 onnx you need to use `RGBH` colorspace, if you use
> FP32 onnx you need to use `RGBS` colorspace in `inference_config.py`."

i.e. `clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")`
before the model call, and `resize.Bicubic(clip, format=vs.YUV420P8,
matrix_s="709")` after, on the CPU/zimg path (GPU-resident conversion is not
what this call does by default). Precision selection is therefore a
**file-selection decision** (pick the fp16 or fp32 ONNX variant, matched to
the matching colorspace call), not a runtime flag on a single graph.

**INT8**: **no evidence of int8 use in this project**, confirmed three
independent ways -- (a) zero `int8` hits across the full 615-line README,
(b) zero int8/int4/quantized assets in the 328-file model release, (c) the
canonical `trtexec` recipe passes only `--bf16 --fp16`, never `--int8`
plus a calibration cache. The project is fp16/fp32/bf16-only across its
entire catalog. This reads as an implicit, undocumented avoidance rather
than a tested-and-rejected path -- there is no issue, PR, or README note
that says "we tried int8 and quality regressed," because int8 was
apparently never attempted here at all.

Because the user's own recollection of an int8 quality regression could not
be corroborated *in this specific repo*, it was checked against the
broader literature instead: INT8 post-training quantization of
single-image super-resolution networks is a well-documented, currently
active research problem precisely because naive PTQ measurably degrades SR
output (visible texture loss and edge artifacts from per-tensor/per-channel
quantization error, worse than for classification nets because SR has no
softmax/argmax step to absorb small activation errors) -- see e.g. "Efficient
INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and
Teacher-Guided Training" (arXiv, 2026) and "QuantVSR: Low-Bit Post-Training
Quantization for Real-World Video Super-Resolution" (arXiv, 2508.04485),
both of which exist specifically because plain PTQ int8 on SR nets is a
known, non-trivial problem requiring dedicated quantization-aware or
deployment-aware training to recover quality -- not something you get by
passing `--int8` to `trtexec` the way this repo does for fp16. **Verdict for
#333**: treat int8 for SR/interpolation as requiring its own calibration
and validation work if pursued later (consistent with this project's own
"lossy features last" prioritization already in place); it is not a drop-in
`trtexec` flag the way fp16 is, and VSGAN-tensorrt-docker provides no
recipe, calibration data, or quality data point to inherit here -- that
work would be ours from scratch, further evidence it should stay
low-priority.

## 4. Server / streaming capability

**None.** VSGAN-tensorrt-docker is a batch/pipe-oriented VapourSynth
script, invoked as `vspipe -c y4m inference.py - | ffmpeg -i pipe: -
out.mkv`. There is no HTTP/gRPC server, no API, no persistent
process serving multiple requests, no session concept. Each invocation
processes one input video end-to-end through a hand-edited
`inference_config.py`/`inference.py` and exits. This is confirmed by the
absence of any server/API/Flask/FastAPI reference in the README and by the
project's own framing ("Usage" -> single-shot `vspipe | ffmpeg` command).

**Docker packaging** is the one transferable pattern for #333's container
work: three image variants (`latest` full suite ~13.1GB, `latest_no_avx512`
for CPUs without AVX-512 -- documented as a real failure mode, "Illegal
instruction (core dumped)", issue #48 -- and `minimal` ~6.1GB with just
`ffmpeg`+`mlrt`+`ffms2`+`lsmash`+`bestsource`), driven by `compose.yaml`
with GPU passthrough (`--gpus all` / deploy.devices). The AVX-512 split in
particular is a concrete, previously-hit pitfall worth carrying into our
own image build (host CPU feature detection before picking a base image),
independent of the "no server" gap.

## 5. Reuse verdict

**(a) As a dependency**: no. VSGAN-tensorrt-docker is not a library or
importable package; it is a set of example scripts meant to be copy-edited
per deployment. There is nothing to `pip install` or link against.

**(b) Portable pieces**:
- The **model catalog** (section 1) is directly useful as a sourcing map --
  it tells us which architectures exist, where their weights/ONNX live,
  and which precision variants are pre-exported, saving a from-scratch
  survey of the RealESRGAN/RIFE/SPAN/CUGAN ecosystem.
- The **`trtexec` engine-build recipe** (flags, tactic sources, shape
  triplet convention) is a solid, battle-tested starting template for our
  own engine-build step.
- The **RGBH/RGBS fp16-fp32 colorspace convention** and the general
  decode->resize->infer->resize->encode shape of the chain is worth
  copying as-is; it is the obvious correct structure and there is no
  reason to redesign it.
- The **Docker AVX-512 pitfall** (section 4) is a concrete lesson to carry
  into our own image, cheaply.

**(c) Genuine eigenbau (nothing here to inherit)**:
- The #333 **planner** itself -- capacity-weighted distribution and
  pair-matrix hop-pricing have no counterpart in this project at all
  (section 2). This is the core of #333 and stays 100% our own design.
- The **stream/API server** -- VSGAN has none; building request handling,
  session/queue management, and multi-client scheduling is entirely new
  work.
- Any **int8/fp8 calibration path**, if pursued later -- not modeled here
  either (section 3).

**Relationship to `efschu/vs-mlrt`**: complementary, not competing, and at
a different layer than VSGAN-tensorrt-docker entirely. `AmusementClub/vs-mlrt`
is the upstream TensorRT/ONNX-RT/ncnn/OpenVINO/MIGraphX *execution* plugin
suite that VSGAN-tensorrt-docker itself depends on for its `vstrt`/`core.trt`
calls (see section 0) -- VSGAN does not reimplement inference, it only
wraps vs-mlrt (and a few HolyWu/TNTwise plugins that go through PyTorch +
`torch_tensorrt` instead). `efschu/vs-mlrt` (GitHub: "VapourSynth TensorRT
plugin with multi-engine pipeline, RTX support, and Lanczos-3 resize") is
the user's own independent engine-layer work, already at the same layer as
upstream vs-mlrt, not a fork of VSGAN-tensorrt-docker's scripting layer.
Net: VSGAN-tensorrt-docker's overlap with our existing work is at the
model-zoo/pipeline-recipe layer (section 1/3), not at the execution-engine
layer where `efschu/vs-mlrt` already sits -- the two efforts don't
collide.

## 6. Pinned reference artifacts for the #333 first build-out chain

Per explicit user direction, the following three artifacts are the
concrete first-build chain for the #333 Class-3 stream server and are
pinned here as reference, not left generic.

### 6.1 `realesr-general-wdn-x4v3_opset16.onnx` (the ESRGAN stage)

- **Location**: `styler00dollar/VSGAN-tensorrt-docker` release `models`
  asset, filename exactly `realesr-general-wdn-x4v3_opset16.onnx`, **size
  4,864,534 bytes (~4.64 MiB)**, ONNX opset **16**.
- **Not otherwise referenced anywhere in VSGAN's own README or example
  scripts** -- it sits in the release asset dump but has no worked
  `trtexec`/usage example in the benchmark table or usage sections (which
  only cover Compact/SPAN/CUGAN/DPIR rows). It is effectively a convenience
  mirror of the upstream `xinntao/Real-ESRGAN` release asset, not a
  VSGAN-curated/benchmarked model. Treat any VSGAN-side performance/VRAM
  numbers as not applicable to this specific file -- none exist.
- **Architecture**: SRVGGNetCompact ("realesr-general-x4v3 -- a tiny small
  model for general scenes", per upstream `xinntao/Real-ESRGAN` README), 4x
  scale. The small file size (4.64 MiB vs. 17-67 MiB for the
  ESRGAN/6-block variants in the same release) confirms this is the
  lightweight compact architecture, not a full RRDB/ESRGAN backbone --
  consistent with a "utility lane" (throughput-over-quality) tier.
- **"wdn" (with-denoise)**: upstream's `-dn` / denoise-strength feature is,
  in the original PyTorch inference script, a *runtime* linear
  interpolation between two checkpoints (the plain `x4v3` weights and a
  `wdn` "with denoise" counterpart) controlled by a blend scalar, used "to
  balance the noise (avoiding over-smooth results)". **The exported ONNX in
  this release bakes in one fixed point on that blend** (the filename
  encodes only `wdn`, no blend-ratio suffix) -- ONNX export requires a
  static graph, so the runtime-interpolation knob from the PyTorch script
  is not preserved. Implication for #333: if variable denoise strength is
  ever wanted as a request-time parameter, that requires either (i)
  exporting multiple fixed-blend ONNX variants and letting the server pick
  one, or (ii) re-implementing the two-checkpoint blend as a small
  pre-inference weight-interpolation step ourselves before ONNX export --
  it does not come for free from this artifact as pinned.
- **Precision / input layout**: no `_fp16` sibling exists for this specific
  filename in the release (unlike most other catalog entries, which ship
  matched fp16+fp32 pairs) -- **this asset is fp32-only as distributed**.
  Per VSGAN's own convention (section 3), that means `RGBS` colorspace at
  the VapourSynth boundary; obtaining an fp16 engine requires either
  passing `--fp16` to `trtexec` against this fp32 ONNX (TensorRT will
  downcast weights at build time -- this is what the project's own recipe
  does for every model, fp16-source or not, via `--bf16 --fp16` regardless
  of input ONNX precision) or re-exporting from the PyTorch source with
  half precision baked in, mirroring how VSGAN itself produces its `_fp16`
  variants for other models. Standard 4D NCHW RGB tensor input/output,
  consistent with the rest of the SRVGGNetCompact family in this catalog.
- **INT8/FP8**: no int8 or fp8 variant exists for this model in the
  release or anywhere in VSGAN's documented recipes (section 3 applies in
  full) -- calibration would be fully our own work if ever pursued.

### 6.2 `HolyWu/vs-rife` (the RIFE stage)

- **Fork target**: user plans `efschu/vs-rife` as a fork of this repo.
- **Model versions supported** (full enum from `vsrife/__init__.py`):
  `4.0` through `4.26`, including `.lite` and `.heavy` variants at several
  points: `4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 4.11,
  4.12, 4.12.lite, 4.13, 4.13.lite, 4.14, 4.14.lite, 4.15, 4.15.lite,
  4.16.lite, 4.17, 4.17.lite, 4.18, 4.19, 4.20, 4.21, 4.22, 4.22.lite,
  4.23, 4.24, 4.25, 4.25.lite, 4.25.heavy, 4.26, 4.26.heavy`. VSGAN's own
  benchmark table only exercises 4.6, 4.6-drba, 4.7, and 4.18 -- the rest
  of the enum is unverified by VSGAN itself, upstream-only.
- **Backends**: PyTorch (eager) is the primary path; TensorRT via
  `torch_tensorrt` (`trt=True`) is the accelerated path, requiring PyTorch
  2.10.0+, VapourSynth R69+, and TensorRT 10.14.1+ (i.e. this plugin
  tracks current, not legacy, TRT/torch releases -- version pinning will
  need active maintenance on our fork).
- **Precision**: input-format-driven like the ESRGAN stage --
  `RGBH`/`torch.half` vs `RGBS`/`torch.float`, no int8/fp8 path in this
  plugin either.
- **Resolution ceiling -- answer to the user's recalled constraint**: **it
  is a config default, not a hard architectural limit.** Two independent
  mechanisms interact:
  1. **Modulo padding**: every RIFE version pads input to a multiple of
     32 (64 for `4.25`/`4.25.heavy`/`4.26`, 128 for `4.25.lite`) --
     handled automatically, not a ceiling.
  2. **TensorRT shape range**: `trt_static_shape=True` locks the engine to
     one exact resolution (matching the padded input) -- any other input
     size requires a *different* engine, rebuilt for that size. In dynamic
     mode, `trt_min_shape`/`trt_opt_shape`/`trt_max_shape` default to
     `[128,128]` / `[1920,1080]` / `[1920,1080]` -- **i.e. out of the box,
     the dynamic-shape engine is only built to accept up to 1080p**;
     feeding 4K through it without explicitly raising `trt_max_shape`
     (and rebuilding the engine at that larger max) would fail or fall
     back. This is very likely the exact constraint the user hit: not
     "RIFE cannot do 4K," but "the default engine wasn't built for 4K."
  3. **No tiling exists in this plugin** -- full frames are processed in
     one pass, so whatever max shape you build for must fit in VRAM in a
     single forward pass; there is no chunk/spill mechanism to fall back
     on. This makes it a **combined config + VRAM question**, not a code
     ceiling: raising `trt_max_shape` and rebuilding the engine is
     sufficient *if* VRAM allows, and if it doesn't, there is no tiling
     escape hatch in this plugin to fall back to.
  4. The maintainer's own documented mitigation for large input is not
     "always pre-resize before RIFE" but the `scale` parameter (RIFE's own
     optical-flow-scale knob, valid values `[0.25, 0.5, 1.0, 2.0, 4.0]`,
     with "try scale=0.5 for 4K video" as the explicit recommendation) --
     a legitimate RIFE-native quality/speed trade-off distinct from a
     pre-resize workaround. For #333, this suggests: prefer raising
     `trt_max_shape` + `scale` tuning over blanket pre-resizing, and only
     fall back to pre-resize when VRAM genuinely can't fit the target
     resolution in one pass (no tiling means that ceiling is real, just
     VRAM-bound rather than code-bound).
