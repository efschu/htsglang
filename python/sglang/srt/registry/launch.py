# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Run the registry control plane.

::

    python -m sglang.srt.registry --port 8500 --engines engines.json

``--engines`` is a JSON file with the resting configuration::

    {
      "default_hot": ["qwen-4b"],
      "max_hot": null,
      "engines": [ {EngineSpec}, ... ]
    }

Specs are registered, not booted. Reaching the resting state is
``POST /registry/idle {"force": true}`` or an explicit state request, so that
starting the control plane never takes a GPU by surprise.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from sglang.srt.registry.arbiter import (
    EngineRegistry,
    RegistryError,
    card_totals_from_nvml,
    free_bytes_from_nvml,
)
from sglang.srt.registry.ledger import DEFAULT_CORRIDOR_BYTES, MIB, ReservationStore
from sglang.srt.registry.spec import EngineSpec

logger = logging.getLogger(__name__)


def build_registry_from_args(args: argparse.Namespace) -> EngineRegistry:
    totals = card_totals_from_nvml()
    store = ReservationStore(
        args.ledger_root, corridor_bytes=int(args.corridor_mib) * MIB
    )
    registry = EngineRegistry(
        store=store,
        card_totals=totals,
        max_hot=args.registry_max_hot,
        idle_after_s=args.idle_after_s,
        work_dir=args.work_dir,
        free_bytes_resolver=free_bytes_from_nvml,
    )
    if args.engines:
        config = json.loads(Path(args.engines).read_text())
        for item in config.get("engines", []):
            registry.register(EngineSpec.from_json(item))
        if config.get("default_hot"):
            registry.set_default_hot(config["default_hot"])
        if config.get("max_hot") is not None and args.registry_max_hot is None:
            registry.max_hot = int(config["max_hot"])
    return registry


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sglang.srt.registry")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8500)
    parser.add_argument("--engines", help="JSON file of specs to register at start")
    # #510: the registry's POST routes reach subprocess.Popen through
    # launch.argv, so give it the same two keys the runtime has. Unset keeps
    # the previous behaviour (open on whatever --host is bound to).
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SGLANG_REGISTRY_API_KEY") or None,
        help=(
            "Bearer token required by every registry route. Defaults to "
            "$SGLANG_REGISTRY_API_KEY. Unset means no authentication."
        ),
    )
    parser.add_argument(
        "--admin-api-key",
        default=os.environ.get("SGLANG_REGISTRY_ADMIN_API_KEY") or None,
        help=(
            "Bearer token required by the state-changing registry routes "
            "(register / deregister / state / pin / default_hot / idle). "
            "Defaults to $SGLANG_REGISTRY_ADMIN_API_KEY."
        ),
    )
    parser.add_argument(
        "--registry-max-hot",
        type=int,
        default=None,
        help=(
            "Blunt safety valve bounding the number of simultaneously hot "
            "engines. NOT the primary mechanism: how many fit is derived from "
            "the VRAM ledger (§7.2), because a count-based cap either wastes "
            "capacity or fails at runtime on a rig with unlike cards. Leave it "
            "unset unless you need to bound process count."
        ),
    )
    parser.add_argument(
        "--corridor-mib",
        type=int,
        default=DEFAULT_CORRIDOR_BYTES // MIB,
        help=(
            "Absolutely free bytes kept on every card (#330). Belongs to the "
            "card, not to any tenant, so no tenant's budget accounts for it."
        ),
    )
    parser.add_argument("--ledger-root", default=None)
    parser.add_argument("--work-dir", default=None, help="where tenant logs go")
    parser.add_argument("--idle-after-s", type=float, default=900.0)
    parser.add_argument(
        "--tick-interval-s",
        type=float,
        default=None,
        help=(
            "#305: seconds between control-tick evaluations, which step idle "
            "tenants one rung down the residency ladder. DEFAULT OFF (0): "
            "without it, idle demotion stays operator-driven via "
            "POST /registry/idle and nothing about an existing deployment "
            "changes. Falls back to $SGLANG_REGISTRY_TICK_INTERVAL_S. The tick "
            "only demotes -- promotion belongs to the request path."
        ),
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="register, print the plan and the derived M, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    args = make_parser().parse_args(argv)
    try:
        registry = build_registry_from_args(args)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.print_plan:
        print(json.dumps(registry.snapshot(), indent=2, default=str))
        return 0
    import uvicorn  # noqa: PLC0415

    from sglang.srt.registry.http_api import build_app  # noqa: PLC0415
    from sglang.srt.registry.tick import build_tick  # noqa: PLC0415

    tick = build_tick(
        registry, interval_s=args.tick_interval_s, idle_after_s=args.idle_after_s
    )
    tick.start()  # a no-op that logs, unless an interval was configured
    app = build_app(
        registry,
        api_key=args.api_key,
        admin_api_key=args.admin_api_key,
        control_tick=tick,
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        tick.stop()
        registry.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
