# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""``python -m sglang.srt.rigmon`` — run a collector, an aggregator, or a probe.

Typical two-rig setup::

    # rig 1 (the one with the browser): aggregator + local collector
    python -m sglang.srt.rigmon serve --port 8770 --host 0.0.0.0 \\
        --engine http://127.0.0.1:30000 --resolution live:1s:10m \\
        --resolution session:15s:6h --resolution history:5m:7d

    # rig 1: mint a pairing token (loopback only)
    python -m sglang.srt.rigmon join-token

    # rig 2: join once, then collect and push outbound
    python -m sglang.srt.rigmon collect --aggregator http://rig1:8770 \\
        --join-token <token> --engine http://127.0.0.1:30000

Rig 2 needs no inbound firewall rule: it opens the connection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from typing import List, Optional

from sglang.srt.rigmon.aggregator import Aggregator, serve
from sglang.srt.rigmon.capabilities import ProbeEnv, probe_all
from sglang.srt.rigmon.collector import Collector, PushClient, load_cached_profiles
from sglang.srt.rigmon.config import AggregatorConfig, CollectorConfig, default_node_id
from sglang.srt.rigmon.facilities import detect_host_environment, facilities
from sglang.srt.rigmon.series import DEFAULT_TIERS, parse_tier_spec


def _add_resolution_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--resolution",
        action="append",
        default=None,
        metavar="NAME:PERIOD:RETAIN",
        help=(
            "one resolution tier, fine to coarse; repeatable. "
            "Default: live:1s:10m, session:15s:6h, history:5m:7d"
        ),
    )


def _tiers(args):
    if not args.resolution:
        return DEFAULT_TIERS
    return tuple(parse_tier_spec(r) for r in args.resolution)


def _join(aggregator_url: str, join_token: str, node_id: str) -> str:
    """Redeem a pairing token for this node's long-lived push token."""
    body = json.dumps({"join_token": join_token, "node_id": node_id}).encode()
    req = urllib.request.Request(
        aggregator_url.rstrip("/") + "/api/join",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["push_token"]


def cmd_collect(args) -> int:
    token = args.push_token
    if args.join_token:
        token = _join(args.aggregator, args.join_token, args.node_id)
        print(f"joined {args.aggregator} as {args.node_id}")
        if args.print_token:
            print(f"push token: {token}")
    cfg = CollectorConfig(
        node_id=args.node_id,
        interval_s=args.interval,
        profile_every=args.profile_every,
        tiers=_tiers(args),
        engine_url=args.engine,
        aggregator_url=args.aggregator,
        push_token=token or "",
        push_every_s=args.push_every,
        boot_log=args.boot_log or "",
    )
    col = Collector(cfg)
    print(f"rigmon collector: node={cfg.node_id} interval={cfg.interval_s}s")
    print(f"  device backend : {col.sampler.backend.name}")
    print(f"  engine         : {cfg.engine_url or '(none)'}")
    print(f"  push to        : {cfg.aggregator_url or '(local only)'}")
    print(f"  resolutions    : " + ", ".join(
        f"{t.spec.name}={t.spec.period_s}s/{t.spec.capacity}" for t in col.series.tiers
    ))
    try:
        col.run_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_serve(args) -> int:
    cfg = AggregatorConfig(
        host=args.host, port=args.port, token=args.token or "", tiers=_tiers(args)
    )
    errs = cfg.validate()
    if errs:
        for e in errs:
            print(f"error: {e}", file=sys.stderr)
        return 2
    agg = Aggregator(cfg)
    col = None
    if not args.no_local:
        col = Collector(
            CollectorConfig(
                node_id=args.node_id,
                interval_s=args.interval,
                profile_every=args.profile_every,
                tiers=cfg.tiers,
                engine_url=args.engine,
                boot_log=args.boot_log or "",
            )
        )
        agg.attach_local(col)
        col.start()
    srv = serve(agg, static_dir=args.ui_dir)
    print(f"rigmon aggregator on http://{cfg.host}:{cfg.port}/")
    print(f"  local node : {args.node_id if col else '(disabled)'}")
    print(f"  push auth  : {'token set' if cfg.token else 'loopback, no token'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if col:
            col.stop()
    return 0


def cmd_join_token(args) -> int:
    url = args.aggregator.rstrip("/") + "/api/join_token"
    with urllib.request.urlopen(url, timeout=10) as r:
        out = json.loads(r.read())
    print(out["join_token"])
    print(
        f"# valid for {out['expires_in_s']:.0f}s, single use. On the other node:\n"
        f"#   python -m sglang.srt.rigmon collect "
        f"--aggregator {args.aggregator} --join-token {out['join_token']}",
        file=sys.stderr,
    )
    return 0


def cmd_capabilities(args) -> int:
    hw, _ = load_cached_profiles()
    report = probe_all(ProbeEnv(hw_profile=hw))
    if args.json:
        print(json.dumps(report.to_json(), indent=1))
        return 0
    if not report.engine_seen:
        print(report.to_json()["note"])
        print()
    for c in report.capabilities:
        print(f"{c.state.upper():<12} {c.label}")
        if c.reason:
            print(f"             {c.reason}")
    return 0


def cmd_facilities(args) -> int:
    env = detect_host_environment()
    # Ask the device backend directly whether the profiling counters exist, so
    # the facility carries the driver's own reason rather than a generic one.
    from sglang.srt.rigmon.sources import select_backend

    backend = select_backend()
    gpm = [f for f in backend.fields() if f.key == "sm_active"]
    backend.close()
    facs = facilities(
        env,
        gpm_supported=(gpm[0].available if gpm else None),
        gpm_reason=(gpm[0].reason if gpm else None),
    )
    if args.json:
        print(
            json.dumps(
                {"host": env.to_json(), "facilities": [f.to_json() for f in facs]},
                indent=1,
            )
        )
        return 0
    print(
        f"host: {'container=' + env.container if env.in_container else 'bare metal'}"
        f"  root={env.is_root}  driver={env.driver_version or '?'}"
    )
    print()
    for f in facs:
        mark = "OK  " if f.available else "NO  "
        print(f"{mark}{f.kind:<8} {f.label}")
        if not f.available:
            print(f"        why    : {f.reason}")
            for r in f.remedy:
                print(f"        remedy : {r}")
            if f.impossible_in_container:
                print("        note   : not achievable from a container at all")
    return 0


def cmd_snapshot(args) -> int:
    col = Collector(
        CollectorConfig(
            node_id=args.node_id, engine_url=args.engine, profile_every=0
        )
    )
    col.tick()
    time.sleep(1.0)
    col.tick()
    print(json.dumps(col.snapshot(), indent=1))
    return 0


def cmd_boot(args) -> int:
    from sglang.srt.rigmon.bootwatch import classify_boot, read_boot_log

    try:
        markers, tail = read_boot_log(args.log)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    d = classify_boot(markers, tail)
    if args.json:
        print(json.dumps(d.to_json(), indent=1))
        return 0
    print(f"stage   : {d.stage_label} ({'+'.join(d.reached) or 'nothing reached'})")
    print(f"serving : {'yes' if d.ready else 'no'}")
    if d.diagnosis:
        print(f"cause   : {d.diagnosis['label']}")
        print(f"means   : {d.diagnosis['meaning']}")
        print(f"do      : {d.diagnosis['remedy']}")
        if d.other_matches:
            print(f"also    : {', '.join(m['label'] for m in d.other_matches)}")
    elif not d.ready:
        print("cause   : no known signature matched; see the excerpt below")
    for line in d.excerpt:
        print(f"  | {line[:160]}")
    return 0 if d.ready else 1


def cmd_kv_budget(args) -> int:
    from sglang.srt.rigmon.kvbudget import (
        describe_budget,
        list_budget_files,
        reset_budget,
    )

    files = list_budget_files(args.cache_dir)
    if not files:
        print(f"no measured KV budget files in {args.cache_dir}")
        return 0
    if args.reset:
        for path in files:
            if args.config_hash and args.config_hash not in path:
                continue
            print(json.dumps(reset_budget(path, backup=not args.no_backup), indent=1))
        return 0
    out = []
    for path in files:
        try:
            out.append(describe_budget(path))
        except (OSError, ValueError) as e:
            print(f"warning: {path}: {e}", file=sys.stderr)
    if args.json:
        print(json.dumps([b.to_json() for b in out], indent=1))
        return 0
    for b in out:
        print(f"{b.config_hash}  max_total_num_tokens={b.max_total_num_tokens}  "
              f"ranks={b.ranks}  age={b.age_s / 3600:.1f}h")
        for c in b.components:
            print(
                f"  rank {c['rank']}: total={c['device_total_mib']} MiB "
                f"free_at_measure={c['free_at_measure_mib']} MiB "
                f"kv_tokens={c['kv_pool_tokens']} "
                f"foreign_resident={c['foreign_residency_mib']} MiB"
            )
        for w in b.warnings:
            print(f"  WARNING: {w}")
    return 0


def cmd_levers(args) -> int:
    from sglang.srt.planner.levers import render_levers_text, suggest_levers

    hw, _ = load_cached_profiles()
    facs = [f.key for f in facilities(detect_host_environment()) if f.available]
    out = suggest_levers(
        heterogeneous=not args.homogeneous,
        probe=hw,
        node_count=args.nodes,
        facility_keys_available=facs,
        keys=(args.lever.split(",") if args.lever else None),
    )
    if args.json:
        print(json.dumps([s.to_json() for s in out], indent=1))
        return 0
    print(render_levers_text(out))
    return 0


def cmd_scenario(args) -> int:
    from sglang.srt.planner.scenarios import (
        SCENARIOS,
        build_harness_command,
        plan_scenario,
        render_scenario_text,
        suggest,
    )

    if args.list:
        for key, s in SCENARIOS.items():
            print(f"{key:34s} {s.question}")
        return 0
    scenario = None
    if args.key:
        scenario = SCENARIOS.get(args.key)
        if scenario is None:
            print(f"error: no scenario {args.key!r}", file=sys.stderr)
            return 2
    elif args.question:
        scenario = suggest(args.question)
        if scenario is None:
            print(
                "No scenario matches that question. Use --list to see the "
                "known ones, or add your own with a scenario JSON file.",
                file=sys.stderr,
            )
            return 1
    else:
        print("error: give --key, --question or --list", file=sys.stderr)
        return 2
    plan = plan_scenario(scenario, facilities(detect_host_environment()))
    if args.json:
        print(json.dumps(plan.to_json(), indent=1))
        return 0
    print(render_scenario_text(plan))
    if scenario.harness:
        first = {a.key: (a.values[0] if a.values else None) for a in scenario.axes}
        cmd = build_harness_command(scenario, first, base_url=args.engine)
        print()
        print("First point drives the existing harness:")
        print(f"  {cmd['command']}")
        for e in cmd["external"]:
            print(f"  before it: set {e['label']} = {e['value']} ({e['apply']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sglang.srt.rigmon",
        description="Host-side rig telemetry: collector, aggregator, probes.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="run a node collector (optionally pushing)")
    c.add_argument("--node-id", default=default_node_id())
    c.add_argument("--interval", type=float, default=1.0,
                   help="base sample interval in seconds (default 1.0)")
    c.add_argument("--profile-every", type=int, default=10,
                   help="read profiling counters every Nth tick (0 = never)")
    c.add_argument("--engine", default="http://127.0.0.1:30000",
                   help="local engine base URL (blank to disable)")
    c.add_argument("--aggregator", default="",
                   help="aggregator base URL to push to")
    c.add_argument("--join-token", default="",
                   help="single-use pairing token from `join-token`")
    c.add_argument("--push-token", default="", help="an already-issued push token")
    c.add_argument("--push-every", type=float, default=5.0)
    c.add_argument("--print-token", action="store_true",
                   help="print the issued push token (it is a secret)")
    c.add_argument("--boot-log", default="")
    _add_resolution_arg(c)
    c.set_defaults(func=cmd_collect)

    s = sub.add_parser("serve", help="run the aggregator (and a local collector)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8770)
    s.add_argument("--token", default="",
                   help="shared push token; required when not on loopback")
    s.add_argument("--node-id", default=default_node_id())
    s.add_argument("--engine", default="http://127.0.0.1:30000")
    s.add_argument("--interval", type=float, default=1.0)
    s.add_argument("--profile-every", type=int, default=10)
    s.add_argument("--no-local", action="store_true",
                   help="aggregator only; do not sample this host")
    s.add_argument("--ui-dir", default=None, help="directory holding index.html")
    s.add_argument("--boot-log", default="")
    _add_resolution_arg(s)
    s.set_defaults(func=cmd_serve)

    j = sub.add_parser("join-token", help="mint a pairing token (loopback only)")
    j.add_argument("--aggregator", default="http://127.0.0.1:8770")
    j.set_defaults(func=cmd_join_token)

    k = sub.add_parser("capabilities", help="probe the capability table")
    k.add_argument("--json", action="store_true")
    k.set_defaults(func=cmd_capabilities)

    f = sub.add_parser(
        "facilities", help="what this host can measure and control, and why not"
    )
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_facilities)

    n = sub.add_parser("snapshot", help="one sample, as JSON")
    n.add_argument("--node-id", default=default_node_id())
    n.add_argument("--engine", default="http://127.0.0.1:30000")
    n.set_defaults(func=cmd_snapshot)

    b = sub.add_parser("boot", help="classify a boot log")
    b.add_argument("log", help="path to the server boot log")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_boot)

    kv = sub.add_parser(
        "kv-budget", help="show or reset the persisted measured KV budget"
    )
    kv.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/sglang"))
    kv.add_argument("--reset", action="store_true")
    kv.add_argument("--config-hash", default="", help="limit --reset to one file")
    kv.add_argument("--no-backup", action="store_true")
    kv.add_argument("--json", action="store_true")
    kv.set_defaults(func=cmd_kv_budget)

    lv = sub.add_parser("levers", help="the five lever profiles for this rig")
    lv.add_argument("--homogeneous", action="store_true",
                    help="treat the rig as uniform (skip heterogeneous-only flags)")
    lv.add_argument("--nodes", type=int, default=1)
    lv.add_argument("--lever", default="", help="comma-separated subset")
    lv.add_argument("--json", action="store_true")
    lv.set_defaults(func=cmd_levers)

    sc = sub.add_parser("scenario", help="benchmark scenarios")
    sc.add_argument("--key", default="")
    sc.add_argument("--question", default="")
    sc.add_argument("--list", action="store_true")
    sc.add_argument("--engine", default="http://127.0.0.1:30000")
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(func=cmd_scenario)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
