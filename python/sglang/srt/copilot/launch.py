"""Entry point: ``python -m sglang.srt.copilot.launch``.

This process holds no GPU. Under ``--backend stub`` (the default) it holds no
dependency on a runtime either: the whole app -- capture, attribution,
transcript, hints, prepared topic contexts, background briefing expansion --
runs against the stub set. That is deliberate and it is the development
contract: the rig is a config switch, never a prerequisite.

``--backend rig`` refuses at launch while the ``/v1/realtime`` transport is
unbuilt, and says which class is missing. It does not boot a half app.
"""

from __future__ import annotations

import argparse
import logging

from sglang.srt.copilot.backends import build_backend_set
from sglang.srt.copilot.briefing import load_briefing, parse_briefing
from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.server import CopilotService, build_app

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sglang.srt.copilot.launch")
    parser.add_argument("--host", default=CopilotConfig.host)
    parser.add_argument("--port", type=int, default=CopilotConfig.port)
    parser.add_argument(
        "--backend",
        default=CopilotConfig.backend,
        choices=("stub", "rig"),
        help="Which backend set to run against. 'stub' needs nothing else "
        "running; 'rig' talks to a booted htsglang runtime.",
    )
    parser.add_argument(
        "--runtime-base-url",
        default=CopilotConfig.runtime_base_url,
        help="Base URL of the htsglang runtime serving /v1/chat/completions "
        "and /v1/realtime. Read only under --backend rig.",
    )
    parser.add_argument("--model", default=CopilotConfig.model)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--briefing",
        default=None,
        help="Path to a Markdown briefing loaded as the default for new "
        "sessions. Under --backend stub, defaults to the briefing the stub "
        "script is a conversation about.",
    )
    parser.add_argument(
        "--hint-lane",
        default=CopilotConfig.hint_lane,
        help="Scheduling lane for live hint requests. Effective only when the "
        "runtime runs with --enable-fast-lane; inert otherwise.",
    )
    parser.add_argument(
        "--stub-script",
        default=CopilotConfig.stub_script,
        help="Which canned conversation the stub ASR speaks.",
    )
    parser.add_argument(
        "--stub-hint-latency-ms",
        type=float,
        default=CopilotConfig.stub_hint_latency_ms,
        help="Stub hint backend latency for a PREPARED topic. A cold topic "
        "additionally pays --stub-cold-penalty-ms.",
    )
    parser.add_argument(
        "--stub-cold-penalty-ms",
        type=float,
        default=CopilotConfig.stub_cold_penalty_ms,
        help="Extra stub hint latency when the topic context is not prepared.",
    )
    parser.add_argument(
        "--stub-prepared-capacity",
        type=int,
        default=CopilotConfig.stub_prepared_capacity,
        help="How many topic contexts the stub session-prep holds at once.",
    )
    parser.add_argument(
        "--min-hint-interval-s",
        type=float,
        default=CopilotConfig.min_hint_interval_s,
        help="Floor between two hint decodes for one conversation.",
    )
    parser.add_argument(
        "--expander-interval-s",
        type=float,
        default=CopilotConfig.expander_interval_s,
        help="Cadence of the background briefing expander.",
    )
    return parser


def build_service(args: argparse.Namespace) -> CopilotService:
    config = CopilotConfig(
        backend=args.backend,
        runtime_base_url=args.runtime_base_url,
        model=args.model,
        api_key=args.api_key,
        hint_lane=args.hint_lane,
        host=args.host,
        port=args.port,
        stub_script=args.stub_script,
        stub_hint_latency_ms=args.stub_hint_latency_ms,
        stub_cold_penalty_ms=args.stub_cold_penalty_ms,
        stub_prepared_capacity=args.stub_prepared_capacity,
        min_hint_interval_s=args.min_hint_interval_s,
        expander_interval_s=args.expander_interval_s,
    )
    backends = build_backend_set(config)
    if args.briefing:
        briefing = load_briefing(args.briefing)
    elif config.backend == "stub":
        from sglang.srt.copilot.stubs import stub_briefing_text

        briefing = parse_briefing(stub_briefing_text(), source="stub")
    else:
        briefing = None
    return CopilotService(config=config, backends=backends, default_briefing=briefing)


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
        "serving on http://%s:%d, backend=%s (stub=%s)",
        args.host,
        args.port,
        service.backends.name,
        service.backends.stub,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
