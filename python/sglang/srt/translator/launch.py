# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Start the translator tenant.

    python -m sglang.srt.translator.launch --asr faster-whisper --tts inprocess

The process pins itself to one physical card by NVML UUID before importing any
backend, so ``cuda:0`` is unambiguous inside it -- the same process-level
isolation the rest of the project uses instead of in-process device maps.

``--asr fake`` / ``--tts fake`` boot the whole surface with the hermetic
backends and no GPU at all. That is the mode the PWA, the tunnel and the
reconnect logic are developed against, and it is also the fastest way to check
that a deployment's networking works before any model is downloaded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

logger = logging.getLogger("translator.launch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sglang.srt.translator.launch",
        description="Live speech-to-speech translation tenant (#466).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30800)

    parser.add_argument(
        "--asr",
        default="fake",
        choices=("fake", "faster-whisper", "nemo"),
        help="recognizer backend; 'fake' needs no GPU and no model",
    )
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--asr-compute-type", default="int8_float16")
    parser.add_argument("--asr-budget-mib", type=int, default=3000)
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument(
        "--asr-cache",
        type=Path,
        default=Path("/spinning/llm_stuff/translator-models/asr-models"),
        help="shared with the gate script; a second root re-downloads 1.5 GB",
    )
    parser.add_argument(
        "--asr-lib",
        type=Path,
        default=Path("/spinning/llm_stuff/translator-models/asr-lib"),
        help=(
            "tree holding faster-whisper. APPENDED to sys.path, never "
            "prepended: the shared serving venv must keep priority for every "
            "package it already provides, and must not be modified at all"
        ),
    )

    parser.add_argument(
        "--tts", default="fake", choices=("fake", "inprocess"),
        help=(
            "synthesizer backend. 'inprocess' loads Qwen3-TTS as an "
            "nn.Module in this process, its weights registered as a ledgered "
            "asset class -- one runtime, no second engine. 'fake' needs no "
            "GPU and no model and is what the hermetic suite uses."
        ),
    )
    parser.add_argument(
        "--tts-model-dir", type=Path,
        default=Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base"),
        help="checkpoint for --tts inprocess",
    )
    parser.add_argument(
        "--tts-device", default="cuda:0",
        help="'cpu' works and is slow; the seam and the audio are identical",
    )
    parser.add_argument("--tts-dtype", default="bfloat16")
    parser.add_argument("--tts-budget-mib", type=int, default=4000)
    parser.add_argument("--tts-sample-rate", type=int, default=24000)
    parser.add_argument(
        "--min-reference-seconds", type=float, default=3.0,
        help="own-voice audio required before a speaker's clone is trusted",
    )
    parser.add_argument(
        "--voice-mode", default="clone", choices=("clone", "preset"),
        help="default output voice mode; switchable per session at runtime",
    )
    parser.add_argument(
        "--preset-voice-dir", type=Path, default=None,
        help="pool of preset voices, <class>/<voice_id>.<language>.wav",
    )

    parser.add_argument(
        "--embedder", default="fake", choices=("fake", "onnx"),
        help="speaker embedding backend",
    )
    parser.add_argument("--embedder-model", type=Path, default=None)

    parser.add_argument(
        "--mt-base-url", default="http://127.0.0.1:30000/v1",
        help="our own OpenAI-compatible endpoint; the dogfood hop",
    )
    parser.add_argument("--mt-model", default="default")
    parser.add_argument(
        "--warmup", dest="warmup", action="store_true", default=True,
        help="synthesize one throwaway sentence before opening the port, so "
             "the first real turn does not pay the talker's one-off "
             "initialisation (~15 s measured). On by default",
    )
    parser.add_argument(
        "--no-warmup", dest="warmup", action="store_false",
        help="open the port immediately and let turn 1 pay the cold start",
    )
    parser.add_argument(
        "--mt-lane", default="fast",
        help="server-side priority lane for translator traffic. 'fast' by default because every call here has a person waiting "
             "on it; empty string sends no lane, for a backend that does not know the field",
    )
    parser.add_argument(
        "--mt-timeout-s", type=float, default=None,
        help="per-request MT budget. The default (20 s) is generous "
             "for an idle backend and NOT generous when the MT server "
             "shares a card with the talker: synthesis starves the "
             "forward pass, the first token misses the budget, and the "
             "turn fails with a read timeout while the recording is "
             "still running. Raise it where the cards are shared",
    )
    parser.add_argument(
        "--mt-languages", default="",
        help="comma-separated; empty means the LLM claims no restriction",
    )
    parser.add_argument(
        "--mt-thinking", action="store_true",
        help=(
            "let a reasoning-capable MT model think out loud. OFF by default, "
            "and measured: Qwen3.6-27B-FP8 answered a German->Spanish request "
            "with \"Here's a thinking process: 1. Analyze User Input...\" and "
            "ran into the token limit before producing a translation. The "
            "chain of thought is not separable downstream -- it would be read "
            "aloud in the target voice."
        ),
    )
    parser.add_argument(
        "--mt-extra-body", default="",
        help=(
            "JSON merged into every MT request body, for options this "
            "launcher does not name (sampling, adapters, routing hints)."
        ),
    )

    parser.add_argument(
        "--participants", default="de,es",
        help=(
            "default conversation languages. A DEFAULT only: the client sets "
            "its own, and nothing in the pipeline is specialised to this pair."
        ),
    )
    parser.add_argument(
        "--card-uuid", default=None,
        help="NVML UUID of the physical card this tenant owns",
    )
    parser.add_argument("--vad", default="energy", choices=("energy", "silero"))
    parser.add_argument("--silero-model", type=Path, default=None)
    parser.add_argument("--hangover-ms", type=int, default=550)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--enable-metrics", action="store_true",
        help="serve Prometheus metrics on /metrics: per-stage turn latencies "
             "(ASR, MT, tts_wait), plus the live queue depths and talker "
             "state. Carried by every boot in this repo -- a server without "
             "it cannot say WHERE a slow turn was slow",
    )
    return parser


def require_websocket_library() -> str:
    """Refuse to start without a WebSocket implementation.

    Found the hard way: uvicorn serves the REST surface perfectly without one
    and answers **404** on the WebSocket route, because it never upgrades the
    connection. Every hermetic test passed regardless -- Starlette's TestClient
    implements the protocol in-process and never touches uvicorn's stack. So
    the entire desk suite was green while the live server could not accept a
    single conversation.

    That is the exact shape of failure this project keeps rediscovering: a
    green suite over a path the tests do not actually traverse. A startup
    refusal is the cheap fix, because the alternative is discovering it from a
    phone on a foreign network.
    """
    import importlib.util

    for module in ("websockets", "wsproto"):
        if importlib.util.find_spec(module) is not None:
            return module
    raise SystemExit(
        "no WebSocket library is installed, so the audio stream endpoint "
        "would answer 404 while every other endpoint worked.\n"
        "  fix: pip install websockets\n"
        "This is refused at startup on purpose: the failure is invisible to "
        "the REST surface and to the hermetic test suite."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    websocket_library = require_websocket_library()
    if args.enable_metrics:
        # Before the service is built: the depth gauges register at app build
        # time, and a flag flipped after that would leave a /metrics endpoint
        # that answers without them.
        from sglang.srt.translator import metrics as _metrics

        _metrics.enable()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Pin BEFORE importing anything that may create a CUDA context.
    if args.card_uuid:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.card_uuid)

    from sglang.srt.translator.backends import FakeAsr, FakeEmbedder, FakeTts
    from sglang.srt.translator.config import (
        AsrConfig,
        DiarizationConfig,
        TranslatorConfig,
        TtsConfig,
    )
    from sglang.srt.translator.mt import MtConfig, OpenAiMt
    from sglang.srt.translator.server import Stack, TranslatorService, build_app

    participants = tuple(p.strip() for p in args.participants.split(",") if p.strip())
    mt_languages = tuple(
        c.strip() for c in args.mt_languages.split(",") if c.strip()
    ) or None

    config = TranslatorConfig(
        asr=AsrConfig(
            backend=args.asr,
            model=args.asr_model,
            card_uuid=args.card_uuid,
            budget_mib=args.asr_budget_mib,
            compute_type=args.asr_compute_type,
        ),
        tts=TtsConfig(
            backend=args.tts,
            card_uuid=args.card_uuid,
            budget_mib=args.tts_budget_mib,
            output_sample_rate=args.tts_sample_rate,
            min_reference_seconds=args.min_reference_seconds,
            voice_mode=args.voice_mode,
            preset_voice_dir=args.preset_voice_dir,
        ),
        diarization=DiarizationConfig(backend=args.embedder, card_uuid=args.card_uuid),
        default_participants=participants,
        host=args.host,
        port=args.port,
    )

    # ASR
    if args.asr == "faster-whisper":
        # faster-whisper is not installed into the serving venv and must not
        # be: two long-running services map that venv, and the wheel's
        # dependency closure would upgrade numpy and tokenizers underneath
        # them. It lives in its own tree, APPENDED to sys.path so the venv
        # keeps priority for everything it already provides -- only the
        # genuinely missing modules resolve from the tree.
        if args.asr_lib and args.asr_lib.exists():
            path = str(args.asr_lib)
            if path not in sys.path:
                sys.path.append(path)
                logger.info("appended %s to sys.path for the recognizer", path)
        elif args.asr_lib:
            raise SystemExit(
                f"--asr faster-whisper needs its library tree at {args.asr_lib}; "
                "install it with\n"
                f"  pip install --target {args.asr_lib} faster-whisper\n"
                "and do NOT install it into the serving venv"
            )

        from sglang.srt.translator.asr_backends import FasterWhisperAsr

        asr = FasterWhisperAsr(
            model=args.asr_model,
            device=args.asr_device,
            compute_type=args.asr_compute_type,
            # The gate script and the tenant must share one cache: a second
            # root silently re-downloads 1.5 GB and, worse, lets the two
            # disagree about which revision is being measured.
            download_root=args.asr_cache,
        )
    elif args.asr == "nemo":
        from sglang.srt.translator.asr_backends import NemoStreamingAsr

        asr = NemoStreamingAsr(model=args.asr_model)
    else:
        asr = FakeAsr(languages=participants, pitch_map=[(150.0, participants[0])])

    # TTS first: the talker embedder borrows this backend's speaker
    # encoder, so the order is a real dependency, not a preference.
    if args.tts == "inprocess":
        from sglang.srt.translator.inprocess_tts import (
            InProcessQwen3Tts,
            InProcessTtsConfig,
        )

        tts = InProcessQwen3Tts(
            InProcessTtsConfig(
                model_dir=args.tts_model_dir,
                device=args.tts_device,
                dtype=args.tts_dtype,
                sample_rate=args.tts_sample_rate,
                min_reference_seconds=args.min_reference_seconds,
            )
        )
        # Load here rather than lazily on the first turn: the checkpoint takes
        # seconds and, more importantly, the weight verification and the two
        # transformers-5.x shims must have run and been SEEN to run before the
        # server advertises itself as ready. A first turn that discovers a
        # broken load is a first turn the user has already spoken into.
        tts.load()
    else:
        tts = FakeTts(
            languages=participants,
            sample_rate=args.tts_sample_rate,
            min_reference_seconds=args.min_reference_seconds,
        )

    # Speaker embedding
    if args.embedder == "onnx":
        from sglang.srt.translator.asr_backends import OnnxSpeakerEmbedder

        if args.embedder_model is None:
            raise SystemExit("--embedder onnx requires --embedder-model")
        embedder = OnnxSpeakerEmbedder(args.embedder_model)
    else:
        embedder = FakeEmbedder()

    # Thinking is suppressed through the chat template rather than filtered out
    # of the answer: a reasoning model's output has no reliable marker the
    # translator could strip, and anything left in the string is spoken.
    extra_body: Dict[str, object] = {}
    if not args.mt_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    # EVERY translator call is latency-critical: a person is waiting mid
    # sentence. The server's priority lanes (protocol.py) put this traffic
    # ahead of bulk co-tenants -- eval prefills and the like -- which is the
    # structural answer to the outages that cost three turns on 2026-08-03.
    # Measured without a lane: MT time-to-first-token 19.7 s behind a 46k
    # prefill, against a ~0.24 s median on an idle backend. The retry in
    # session.py survives such a window; the lane is what stops it happening.
    # Anti-starvation lives in the server, so this cannot lock anyone out.
    if args.mt_lane:
        extra_body["lane"] = args.mt_lane
    if args.mt_extra_body:
        try:
            extra_body.update(json.loads(args.mt_extra_body))
        except (ValueError, AttributeError) as exc:
            raise SystemExit(f"--mt-extra-body is not a JSON object: {exc}") from exc

    mt = OpenAiMt(
        MtConfig(
            base_url=args.mt_base_url,
            model=args.mt_model,
            languages=mt_languages,
            extra_body=extra_body,
            **({} if args.mt_timeout_s is None
               else {"timeout_s": args.mt_timeout_s}),
        )
    )

    service = TranslatorService(config, Stack(asr=asr, embedder=embedder, mt=mt, tts=tts))
    app = build_app(service)

    languages = service.languages()
    logger.info(
        "translator ready on %s:%d | asr=%s tts=%s mt=%s | %d routable pairs | "
        "voice=%s presets=%d | ws=%s",
        args.host,
        args.port,
        getattr(asr, "name", "?"),
        getattr(tts, "name", "?"),
        getattr(mt, "name", "?"),
        languages["pair_count"],
        service.voice_mode.value,
        len(service.voice_pool_template) if service.voice_pool_template else 0,
        websocket_library,
    )
    if not languages["default_participants_supported"]:
        logger.error(
            "the default conversation cannot run here: %s",
            languages["default_participants_error"],
        )

    # BEFORE the port opens, not behind a ready flag: there must be no window
    # in which health says ok and a turn would still pay the cold start. See
    # `TranslatorService.warmup` for the ~15 s this buys back on turn 1.
    if args.warmup:
        report = asyncio.run(service.warmup())
        if report["ran"]:
            logger.info(
                "talker warm after %.2fs (%d chunks, %s, voice %s) -- turn 1 "
                "no longer pays the cold start",
                report["seconds"], report["chunks"], report["language"],
                report["voice_id"],
            )
        else:
            # Never fatal, and never silent: a boot that quietly skipped the
            # warmup looks identical to one that did it, right up to the 15 s
            # the user waits for their first sentence.
            logger.warning(
                "talker warmup did not run (%s) -- turn 1 will pay the cold "
                "start", report["reason"],
            )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
