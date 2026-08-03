# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Start the translator tenant.

    python -m sglang.srt.translator.launch --asr faster-whisper --tts fake

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
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

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

    parser.add_argument(
        "--tts", default="fake", choices=("fake",),
        help=(
            "synthesizer backend. Only the hermetic fake is wired today: the "
            "chosen real backend (Qwen3-TTS via vLLM-Omni) runs in its own "
            "venv as a separate process and is reached over its OpenAI-"
            "compatible /v1/audio/speech endpoint, which is a client, not an "
            "in-process backend. See DESIGN_466 for why."
        ),
    )
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
        "--mt-languages", default="",
        help="comma-separated; empty means the LLM claims no restriction",
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
        from sglang.srt.translator.asr_backends import FasterWhisperAsr

        asr = FasterWhisperAsr(
            model=args.asr_model,
            compute_type=args.asr_compute_type,
            download_root=config.model_root / "asr",
        )
    elif args.asr == "nemo":
        from sglang.srt.translator.asr_backends import NemoStreamingAsr

        asr = NemoStreamingAsr(model=args.asr_model)
    else:
        asr = FakeAsr(languages=participants, pitch_map=[(150.0, participants[0])])

    # Speaker embedding
    if args.embedder == "onnx":
        from sglang.srt.translator.asr_backends import OnnxSpeakerEmbedder

        if args.embedder_model is None:
            raise SystemExit("--embedder onnx requires --embedder-model")
        embedder = OnnxSpeakerEmbedder(args.embedder_model)
    else:
        embedder = FakeEmbedder()

    tts = FakeTts(
        languages=participants,
        sample_rate=args.tts_sample_rate,
        min_reference_seconds=args.min_reference_seconds,
    )

    mt = OpenAiMt(
        MtConfig(
            base_url=args.mt_base_url,
            model=args.mt_model,
            languages=mt_languages,
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

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
