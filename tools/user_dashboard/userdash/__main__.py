"""python -m userdash -- the user dashboard (Nutzer-Order 01.10. ~12:55Z: "in den docker gehört auch noch ein
'userdashboard' ohne die entwicklungssachen").

Every flag has an env twin; the entrypoint of the image passes the port and the front explicitly.
USERDASH_ENABLE=0 makes the process exit 0 at once (the entrypoint does not even start it with
HTSGLANG_USERDASH=0 / FLLIPER_USERDASH=0).
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse

from .collect import Collector, Config
from .server import serve

DEFAULT_PORT = 30080
#: ports this dashboard must never take: the front (30030), P/D group servers (30031/30032), the rig's
#: lifeline routers (30097/30099), rigdash (8890), VictoriaMetrics (8428)
RESERVED_PORTS = (30030, 30031, 30032, 30097, 30099, 8890, 8428)


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v.strip()


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("USERDASH_ENABLE", "1")).strip().lower() not in ("0", "false", "no", "off")


def port_problem(port: int, front: str) -> str:
    if not 1 <= port <= 65535:
        return "Port %d liegt ausserhalb 1..65535" % port
    if port in RESERVED_PORTS:
        return "Port %d ist belegt/reserviert (%s)" % (port, ", ".join(map(str, RESERVED_PORTS)))
    fp = urllib.parse.urlsplit(front).port
    if fp is not None and fp == port:
        return "Port %d ist der Port des Servers selbst (%s)" % (port, front)
    return ""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="userdash", description="Nutzer-Dashboard eines fLLiper/htsglang-Servers")
    ap.add_argument("--port", type=int, default=int(_env("USERDASH_PORT", str(DEFAULT_PORT))))
    ap.add_argument("--bind", default=_env("USERDASH_BIND", "0.0.0.0"))
    ap.add_argument("--front", default=_env("USERDASH_FRONT", "http://127.0.0.1:30030"),
                    help="Basis-URL des Servers (Front bzw. D-only-Server)")
    ap.add_argument("--group-metrics", default=os.environ.get("USERDASH_GROUP_METRICS", "auto"),
                    help="auto | '' (nie) | Komma-Liste von Basis-URLs (nur fuer eine Front ohne Gruppen-Label)")
    ap.add_argument("--active-s", type=float, default=float(_env("USERDASH_ACTIVE_S", "5")))
    ap.add_argument("--idle-s", type=float, default=float(_env("USERDASH_IDLE_S", "30")),
                    help="Abtastung ohne offenen Browser; 0 = gar nicht")
    ap.add_argument("--history-s", type=float, default=float(_env("USERDASH_HISTORY_S", "3600")))
    ap.add_argument("--no-gpu", action="store_true", default=_env("USERDASH_GPU", "1") == "0")
    return ap


def main(argv=None) -> int:
    if not enabled():
        print("userdash: USERDASH_ENABLE=0 -- nicht gestartet", file=sys.stderr)
        return 0
    a = build_parser().parse_args(argv)
    why = port_problem(a.port, a.front)
    if why:
        print("userdash: VERWEIGERT: " + why, file=sys.stderr)
        return 2
    cfg = Config(front=a.front, group_metrics=a.group_metrics, active_s=max(1.0, a.active_s),
                 idle_s=max(0.0, a.idle_s), history_s=max(300.0, a.history_s), gpu=not a.no_gpu)
    col = Collector(cfg)
    httpd = serve(col, a.bind, a.port)
    col.start()
    print("userdash: http://%s:%d/ -> Server %s (live %gs, ohne Browser %s)"
          % (a.bind, a.port, a.front, cfg.active_s, ("%gs" % cfg.idle_s) if cfg.idle_s else "aus"),
          file=sys.stderr, flush=True)
    try:
        httpd.serve_forever(poll_interval=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        col.stop.set()
        col.wake.set()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
