# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""``python -m sglang.srt.turnkey`` -- the one command the units call.

Subcommands are deliberately few and each maps to one systemd hook:

===============  ==========================================================
``preflight``    ``ExecStartPre`` of the target. Named refusals, exit 3.
``boot``         ``ExecStart`` of ``htsglang-serving@``. Becomes the server.
``watch``        ``ExecStart`` of ``htsglang-watchdog@``. Detect, never spawn.
``probe``        Ad-hoc: is this lane actually generating right now?
``plan-pin``     Record the current world as the pinned plan.
``orphans``      List (and, only when asked, reap) stale pids on our cards.
===============  ==========================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from sglang.srt.turnkey import orchestrator, plan as PL, preflight as PF
from sglang.srt.turnkey import probe as P, runner as R, watchdog as W
from sglang.srt.turnkey import config as C
from sglang.srt.turnkey.refusal import RefusalError

DEFAULT_CONFIG = "/etc/htsglang/stack.toml"


def _load(path: str):
    try:
        return C.load(path), None
    except RefusalError as e:
        return None, e.refusal


def _policy(cfg) -> W.Policy:
    w = cfg.watchdog
    return W.Policy(poll_s=w.poll_s, generation_probe_s=w.generation_probe_s,
                    generation_probe_enabled=w.generation_probe_enabled,
                    wedge_signal_enabled=w.wedge_signal_enabled,
                    wedge_confirmations=w.wedge_confirmations,
                    backoff_s=tuple(w.backoff_s), max_restarts=w.max_restarts,
                    restart_window_s=w.restart_window_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sglang.srt.turnkey",
                                 description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight", help="run every named check")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("boot", help="preflight, resolve the plan, exec")
    p.add_argument("lane")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--setsid", action="store_true",
                   help="interactive use only; a systemd unit must not")

    p = sub.add_parser("watch", help="supervise a lane (detector only)")
    p.add_argument("lane")
    p.add_argument("--unit", default="",
                   help="serving unit to restart; default htsglang-serving@<lane>")
    p.add_argument("--ticks", type=int, default=0, help="0 = forever")

    p = sub.add_parser("probe", help="one real generation probe")
    p.add_argument("lane")
    p.add_argument("--timeout", type=float, default=60.0)

    p = sub.add_parser("plan-pin", help="write the current world as the pin")
    p.add_argument("lane")
    p.add_argument("--note", default="")
    p.add_argument("--flag", action="append", default=[],
                   help="a launch flag token the plan contributes")

    p = sub.add_parser("orphans", help="stale pids holding our cards")
    p.add_argument("--reap", action="store_true",
                   help="actually signal them (SIGTERM, then SIGKILL)")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg, r = _load(a.config)
    if r:
        print(r.line())
        return 3

    if a.cmd == "preflight":
        return _cmd_preflight(cfg, a)
    if a.cmd == "boot":
        return orchestrator.boot(cfg, a.lane, dry_run=a.dry_run,
                                 use_setsid=a.setsid)
    if a.cmd == "watch":
        return _cmd_watch(cfg, a)
    if a.cmd == "probe":
        return _cmd_probe(cfg, a)
    if a.cmd == "plan-pin":
        return _cmd_pin(cfg, a)
    if a.cmd == "orphans":
        return _cmd_orphans(cfg, a)
    return 2


def _cmd_preflight(cfg, a) -> int:
    refusals = orchestrator.run_preflight(cfg)
    if a.json:
        import json
        print(json.dumps([x.to_json() for x in refusals], indent=2))
    else:
        for x in refusals:
            print(x.line())
        print(f"turnkey preflight: {len(refusals)} refusal(s)")
    return 3 if refusals else 0


def _cmd_watch(cfg, a) -> int:
    lane = cfg.lane(a.lane)
    if lane is None:
        print(f"no such lane: {a.lane}")
        return 2
    if not cfg.watchdog.enabled:
        # This key was parsed and never read: 'enabled = false' silently did
        # nothing. Honouring it is part of retiring the prober -- a config
        # that says off must BE off.
        print("watchdog disabled by config ([watchdog].enabled = false)")
        return 0
    unit = a.unit or f"htsglang-serving@{lane.name}.service"
    pol = _policy(cfg)
    pol = W.Policy(**{**pol.__dict__, "boot_grace_s": float(lane.ready_timeout_s)})
    run = R.WatchdogRunner(
        unit=unit, base_url=f"http://127.0.0.1:{lane.port}", policy=pol,
        generation_timeout_s=cfg.watchdog.generation_timeout_s,
        # #799: both halves must name the SAME directory. Empty means "use
        # wedge_status.DEFAULT_STATUS_DIR", which is also what the publishing
        # scheduler falls back to, so the default configuration agrees by
        # construction rather than by two matching literals.
        wedge_status_dir=cfg.watchdog.wedge_status_dir or None,
        # #799: what a restart WOULD boot, and what the lane last DID boot.
        # Without both the drift veto is inert, so they are wired here rather
        # than left to a caller that might not pass them.
        lane_argv=lane.argv,
        boot_log=lane.boot_log)
    run.run(max_ticks=a.ticks or None)
    return 0


def _cmd_probe(cfg, a) -> int:
    lane = cfg.lane(a.lane)
    if lane is None:
        print(f"no such lane: {a.lane}")
        return 2
    url = f"http://127.0.0.1:{lane.port}"
    api = P.api_ok(url)
    print(f"api: ok={api.ok} {api.detail}")
    g = P.generation_ok(url, timeout=a.timeout)
    print(f"generation: ok={g.ok} {g.detail}")
    if api.ok and not g.ok:
        print(f"{R.ALARM} WEDGE SIGNATURE: API answers, generation does not")
        return 1
    return 0 if g.ok else 1


def _cmd_pin(cfg, a) -> int:
    lane = cfg.lane(a.lane)
    if lane is None:
        print(f"no such lane: {a.lane}")
        return 2
    p = PF.default_probes()
    observed = {c.uuid: c for c in p.cards()}
    try:
        cards = [(cfg.cards[i].uuid,
                  observed[cfg.cards[i].uuid].total_bytes // PF.MIB)
                 for i in lane.cards]
    except KeyError as e:
        print(f"card not present: {e}")
        return 3
    wheel_version = ""
    if cfg.wheel.must_import:
        try:
            wheel_version = p.probe_import(cfg.wheel.must_import[0],
                                           "int8_scaled_mm").version
        except ImportError:
            pass
    fp = PL.fingerprint_of(cards, lane.argv,
                           model_path=orchestrator._model_path_of(lane),
                           wheel_version=wheel_version)
    pin = PL.PinnedPlan(fingerprint=fp, launch_flags=tuple(a.flag),
                        solved_at=time.time(), solver="turnkey plan-pin",
                        note=a.note)
    PL.save_pinned(cfg.plan.path, pin)
    print(f"pinned plan written: {cfg.plan.path}")
    print(f"  cards: {fp.cards}")
    print(f"  argv digest: {fp.argv_digest}")
    return 0


def _cmd_orphans(cfg, a) -> int:
    uuids = [c.uuid for c in cfg.cards]
    found = R.orphan_pids(uuids, protect=cfg.preflight.protected_ports and ())
    for line in R.reap_orphans(found, dry_run=not a.reap):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
