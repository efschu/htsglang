"""Entry point: ``python -m sglang.srt.copilot.launch``.

This process holds no GPU. It talks to an already-running htsglang runtime over
that runtime's own OpenAI-compatible surface. If the runtime is not up, every
hint request fails loudly at call time; the app deliberately does NOT probe on
boot, because a copilot that refuses to start is worse than one that shows the
transcript while the model is being restarted.
"""

from __future__ import annotations

import argparse
import logging

from sglang.srt.copilot.briefing import load_briefing
from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.hints import ChatHintBackend, DeskFakeHints
from sglang.srt.copilot.server import CopilotService, build_app

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sglang.srt.copilot.launch")
    parser.add_argument("--host", default=CopilotConfig.host)
    parser.add_argument("--port", type=int, default=CopilotConfig.port)
    parser.add_argument(
        "--runtime-base-url",
        default=CopilotConfig.runtime_base_url,
        help="Base URL of the htsglang runtime serving /v1/chat/completions "
        "and /v1/realtime.",
    )
    parser.add_argument("--model", default=CopilotConfig.model)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--briefing",
        default=None,
        help="Path to a Markdown briefing loaded as the default for new sessions.",
    )
    parser.add_argument(
        "--hint-lane",
        default=CopilotConfig.hint_lane,
        help="Scheduling lane for live hint requests. Effective only when the "
        "runtime runs with --enable-fast-lane; inert otherwise.",
    )
    parser.add_argument(
        "--desk-fake-backend",
        action="store_true",
        help="Use the desk-fake hint backend instead of the runtime. For "
        "protocol work without a booted model. Every frame it produces is "
        "marked desk_fake=true and its transcripts carry a marker.",
    )
    return parser


def build_service(args: argparse.Namespace) -> CopilotService:
    config = CopilotConfig(
        runtime_base_url=args.runtime_base_url,
        model=args.model,
        api_key=args.api_key,
        hint_lane=args.hint_lane,
        host=args.host,
        port=args.port,
    )
    backend = (
        DeskFakeHints(config=config)
        if args.desk_fake_backend
        else ChatHintBackend(config)
    )
    briefing = load_briefing(args.briefing) if args.briefing else None
    return CopilotService(
        config=config, hint_backend=backend, default_briefing=briefing
    )


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [copilot] %(message)s",
    )
    args = build_parser().parse_args(argv)
    service = build_service(args)
    app = build_app(service)

    import uvicorn

    logger.info(
        "serving on http://%s:%d, runtime at %s",
        args.host,
        args.port,
        args.runtime_base_url,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
