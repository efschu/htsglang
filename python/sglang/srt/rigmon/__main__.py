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

This CLI is host telemetry only. The five levers and the measurement
scenarios are planner data and live in the planner CLI::

    python -m sglang.planner --levers
    python -m sglang.planner --list-scenarios

They were reachable from here as well, which meant one concept behind two
front doors in two argument styles.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
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


def _join(
    aggregator_url: str,
    join_token: str,
    node_id: str,
    model_paths: Optional[List[str]] = None,
    force: bool = False,
) -> str:
    """Redeem a pairing token, presenting this node's identity to the gate.

    A refusal comes back as HTTP 409 with the full report; it is printed in
    full, because the reasons are the whole point of running the gate before
    the join rather than discovering the mismatch in a worker later.
    """
    from sglang.srt.rigmon.compat import local_identity

    identity = local_identity(node_id, model_paths or [])
    body = json.dumps(
        {
            "join_token": join_token,
            "node_id": node_id,
            "identity": identity.to_json(),
            "force": force,
        }
    ).encode()
    req = urllib.request.Request(
        aggregator_url.rstrip("/") + "/api/join",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 409:
            detail = json.loads(e.read())
            print("compatibility gate refused the join:", file=sys.stderr)
            for c in detail["compatibility"]["checks"]:
                if c["verdict"] == "block":
                    print(f"  BLOCK {c['label']}: {c['detail']}", file=sys.stderr)
                    if c.get("remedy"):
                        print(f"        {c['remedy']}", file=sys.stderr)
            print(f"  {detail['hint']}", file=sys.stderr)
            raise SystemExit(3)
        raise
    report = out.get("compatibility")
    if report:
        for c in report["checks"]:
            if c["verdict"] == "warn":
                print(f"  warning: {c['label']}: {c['detail']}", file=sys.stderr)
        if out.get("forced"):
            print(
                "  joined despite blocking checks (--force)", file=sys.stderr
            )
    return out["push_token"]


def cmd_collect(args) -> int:
    token = args.push_token
    if args.join_token:
        token = _join(
            args.aggregator,
            args.join_token,
            args.node_id,
            model_paths=args.model or [],
            force=args.force,
        )
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
    from sglang.srt.rigmon.compat import local_identity

    agg.local_identity = local_identity(args.node_id, args.model or [])
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


def cmd_probe(args) -> int:
    """The short probe, step 3 of the join sequence.

    Reads the cached stage-0 micro-probe rather than running one: a probe
    allocates a CUDA context on every card, and this CLI also runs next to a
    live server. `--reprobe` is the explicit opt-in that measures afresh.
    """
    from sglang.srt.rigmon.probe import (
        CardState,
        estimate_budget,
        from_hardware_profile,
        measure_host_link,
    )
    from sglang.srt.rigmon.sources import select_backend
    from sglang.srt.rigmon.transport import choose_all_pairs

    if args.reprobe:
        from sglang.srt.uneven_perf import get_hardware_profile

        hw, source, _ = get_hardware_profile()
        print(f"probe source: {source}", file=sys.stderr)
    else:
        hw, _ = load_cached_profiles()

    # Card state at read time: clock, temperature, throttling. A point taken
    # under throttling is kept and marked, never dropped.
    states = {}
    try:
        backend = select_backend()
        for c in backend.sample(with_profiling=False):
            if c.uuid:
                states[c.uuid] = CardState(
                    sm_clock_mhz=c.sm_clock_mhz,
                    sm_clock_max_mhz=c.sm_clock_max_mhz,
                    temp_c=c.temp_c,
                    throttle_reasons=tuple(c.performance_throttles()),
                    sampled_at=time.time(),
                )
        backend.close()
    except Exception as e:
        print(f"warning: no card state ({e})", file=sys.stderr)

    result = from_hardware_profile(hw, args.node_id, card_states=states)

    if args.peer:
        host, _, port = args.peer.rpartition(":")
        link = measure_host_link(host or args.peer, int(port or 8770))
        result.links.append(link)
        result.notes.append(link.note)

    facs = [f.key for f in facilities(detect_host_environment()) if f.available]
    choices = choose_all_pairs(result, available_facilities=facs)

    if args.json:
        print(
            json.dumps(
                {
                    "probe": result.to_json(),
                    "budget": estimate_budget(len(result.cards)).to_json(),
                    "transport": [c.to_json() for c in choices],
                },
                indent=1,
            )
        )
        return 0

    budget = estimate_budget(len(result.cards))
    print(
        f"cards: {len(result.cards)}  ordered pairs: {budget.ordered_pairs}  "
        f"estimated probe cost: {budget.estimate_s:.0f} s of a "
        f"{budget.budget_s:.0f} s budget "
        f"({'fits' if budget.fits else 'OVER BUDGET'})"
    )
    print()
    for c in result.cards:
        state = ""
        if c.state:
            bits = []
            if c.state.temp_c is not None:
                bits.append(f"{c.state.temp_c:.0f} C")
            if c.state.clock_ratio is not None:
                bits.append(f"{c.state.clock_ratio * 100:.0f} % clock")
            if c.state.throttled:
                bits.append("THROTTLED: " + ",".join(c.state.throttle_reasons))
            state = "  [" + ", ".join(bits) + "]" if bits else ""
        print(
            f"{c.id:44s} {c.total_mib or 0:>7} MiB  "
            f"{c.gemm_tflops or 0:>7.1f} TFLOPS ({c.gemm_dtype})  "
            f"{c.membw_gbs or 0:>7.0f} GB/s{state}"
        )
    for g in result.group_latencies:
        print(
            f"\ngroup all-reduce on {g.node_id}: "
            f"{g.allreduce_10kb_us or 0:.1f} us at 10 kB, "
            f"{g.allreduce_1mb_us or 0:.1f} us at 1 MB "
            "(a group figure, not a per-pair latency)"
        )
    print("\nordered pairs and the transport that falls out of them:")
    for ch in choices:
        chosen = ch.chosen
        m = ch.measurement or {}
        bw = m.get("bandwidth_gbs")
        lat = m.get("latency_us")
        meas = []
        if bw is not None:
            meas.append(f"{bw:.2f} GB/s")
        if lat is not None:
            meas.append(f"{lat:.1f} us")
        direction = m.get("direction", "")
        mark = "" if direction.startswith("measured") else "  (mirrored)"
        print(
            f"  {ch.src} -> {ch.dst}: "
            + (", ".join(meas) if meas else "not measured")
            + f" => {chosen.key if chosen else 'unknown'}{mark}"
        )
        if chosen:
            print(f"      {chosen.reason}")
        for o in ch.options:
            if o.verdict == "unavailable":
                print(f"      unavailable: {o.label} ({', '.join(o.missing_facilities)})")
    warnings = result.state_warnings()
    if warnings:
        print("\nhow to read this:")
        for w in warnings:
            print(f"  - {w}")
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
    c.add_argument("--model", action="append", default=None,
                   help="model path to present to the compatibility gate; "
                        "repeatable")
    c.add_argument("--force", action="store_true",
                   help="join even when the compatibility gate blocks")
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
    s.add_argument("--model", action="append", default=None,
                   help="model path this node serves; the compatibility "
                        "gate compares it against a joining node. Repeatable.")
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

    pr = sub.add_parser(
        "probe",
        help="the short probe: card rates + the ordered pair matrix, and the "
        "transport that falls out of it",
    )
    pr.add_argument("--node-id", default=default_node_id())
    pr.add_argument(
        "--reprobe",
        action="store_true",
        help="measure afresh instead of reading the cache. Allocates a CUDA "
        "context on every card, so not the default while a server runs.",
    )
    pr.add_argument(
        "--peer",
        default="",
        metavar="HOST:PORT",
        help="also measure the host-to-host leg to another node. It BOUNDS "
        "the device-to-device path across the rig boundary; it is not it.",
    )
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_probe)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
