# htsglang-qwen35-gguf — Qwen3.5/3.6 GGUF + MTP overlay on the uneven-TP base.
#
# This is a THIN overlay on the already-built htsglang runtime image (the
# uneven-TP base with torch 2.11 / sgl-kernel 0.3.21 / flashinfer / NCCL 2.30.7
# / HiCache already installed). It only adds:
#   - ffmpeg  (GGUF repos that ship a video_preprocessor, e.g. huihui Q8_0,
#              need libavutil/torchcodec at config-load time)
#   - the GGUF fork source overlay (Qwen35 GGUF adapter, hybrid-GDN loader,
#     uneven-TP quant sharding, device-adaptive spec-decode crossover)
#   - the ENV-driven entrypoint with the GGUF flags exposed
#
# No torch / kernel rebuild — seconds, not minutes.
ARG BASE=htsglang:cu130-nccl2307
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# The base editable-installs sglang from /sgl-workspace/sglang/python, so
# overlaying the source there makes the GGUF code live with no reinstall.
COPY python /sgl-workspace/sglang/python

COPY --chmod=0755 docker/htsglang-entrypoint.sh /usr/local/bin/htsglang-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/htsglang-entrypoint.sh"]
