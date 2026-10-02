"""HTTP front of the rig dashboard (stdlib only, no CUDA, no sglang import).

Routes
  GET /               the live page
  GET /api/live       one JSON snapshot: boots (IPC: state.json, events.jsonl, rankstats -- no log), GPUs,
                      containers, gpuq plan, every source's age and error,
                      the running image's changes per seat (image_changes.json)
  GET /api/launch     Startflags + ENV je Modell (Container/Front/P/D) aus state.json,
                      P<->D-Vergleich; ?ver=<ver> antwortet {same: true}, solange gleich
  GET /api/history    Verlauf (history.py): ?model=27B|NF&range=15m|1h|6h|24h|7d
  GET /api/health     liveness of the dashboard itself
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shlex
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from . import energy, features, health, history, imagechanges, ipcboot, launchview, live, redact, sampler, sources, vmpush, weg2line

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
VERSION_FILE = os.path.join(HERE, "VERSION")
#: vendored, served locally (the rig is LAN-only, no CDN): uPlot 1.6.32 (MIT, static/uplot.LICENSE)
STATIC_FILES = {
    "/uplot.iife.min.js": ("uplot.iife.min.js", "application/javascript; charset=utf-8"),
    "/uplot.min.css": ("uplot.min.css", "text/css; charset=utf-8"),
    "/uplot.LICENSE": ("uplot.LICENSE", "text/plain; charset=utf-8"),
    "/grafik.js": ("grafik.js", "application/javascript; charset=utf-8"),
    "/zoom.js": ("zoom.js", "application/javascript; charset=utf-8"),
}


def _version():
    try:
        with open(VERSION_FILE) as fh:
            return fh.read().strip()
    except OSError:
        return "dev"


def _container_token(name: str) -> str:
    n = re.sub(r"^htsglang-(acc-)?", "", name or "")
    return n.replace("-", "").replace("_", "")


def parse_zoom(path: str):
    """``?zoom=<t0>,<t1>`` (unix s) of the page's click-and-drag zoom: the boot cards' curves for that
    stretch at 1/2/5-s buckets; anything unreadable is no zoom."""
    from urllib.parse import parse_qs, urlsplit
    z = (parse_qs(urlsplit(path).query).get("zoom") or [""])[0]
    try:
        a, b = (float(x) for x in z.split(","))
    except ValueError:
        return None
    return (a, b) if b - a >= 5 else None


def finish_series(b: dict, gpu_series: Optional[dict], now: float, bucket_s: float, key: str = "series") -> None:
    """Cut a series where the model really failed, and add tok/s per watt.

    Gap = GRUPPE TOT (from the first dead reason on) or the boot has ended
    (no log written and no running container) -- marked by ``gap_from`` and
    ``gap_reason``.  Power = sum of nvidia-smi power.draw over ALL cards in the
    bucket (mean of the samples); the idle draw of the sleeping group counts,
    because it is really spent.
    """
    ser = b.get(key)
    if not ser or not ser.get("t"):
        return
    ts = ser["t"]
    bucket_s = ser.get("step") or bucket_s
    gap_from, reason = None, None
    a = b.get("alarm") or {}
    if a.get("state") == "TOT":
        dead = [r["t"] for r in a.get("reasons") or [] if r.get("level") == "dead" and r.get("t")]
        if dead:
            gap_from, reason = min(dead), "GRUPPE TOT"
    c = b.get("container") or {}
    ended = not b.get("live") and c.get("State") != "running" and b.get("last_log_t")
    ps = a.get("planned_stop") or {}
    kind = "dead" if gap_from is not None else None
    if ps.get("stopping") and gap_from is None:
        # planned stop (stops.py): grey, from the first teardown sign or the log's end
        gap_from = ps.get("teardown_t") or (b["last_log_t"] if ended else None)
        if gap_from is not None:
            reason, kind = "gestoppt (geplant)", "planned"
    elif ended:
        if gap_from is None or b["last_log_t"] < gap_from:
            gap_from, reason, kind = b["last_log_t"], "Boot beendet / Container weg", "ended"
            death = (b.get("end") or {}).get("death")
            if death:
                # the harness named it a death (deadman verdict / hold end "Container-tot")
                reason = "tot (%s)" % ({"deadman": "Deadman", "state.json": death.get("text") or "state.json"}
                                       .get(death.get("src"), "Container-tot"))
    keys = [k for k in ser if k.endswith("_tps")]
    if gap_from is not None:
        for k in keys + [k for k in ser if k.endswith(("_rate", "_seats", "_stream"))]:
            ser[k] = [None if t + bucket_s > gap_from else v for t, v in zip(ts, ser[k])]
        ser["gap_from"], ser["gap_reason"], ser["gap_kind"] = gap_from, reason, kind
        tl = b.get("timeline")
        if tl:
            tl["segs"] = [dict(x, e=min(x["e"], gap_from)) for x in tl["segs"] if x["s"] < gap_from]
            tl["cut_at"], tl["cut_reason"], tl["cut_kind"] = gap_from, reason, kind
    power = [None] * len(ts)
    if gpu_series and gpu_series.get("t"):
        acc = [[0.0, 0] for _ in ts]
        t0 = ts[0]
        for t, row in zip(gpu_series["t"], gpu_series["power"]):
            i = int((t - t0) // bucket_s)
            if 0 <= i < len(ts) and row and all(x is not None for x in row):
                acc[i][0] += sum(row)
                acc[i][1] += 1
        power = [(a0 / n) if n else None for a0, n in acc]
        # nvidia-smi samples every ~2 s: at 1-/2-s buckets (zoom) a bucket without its own sample holds
        # the previous one for up to 3 s (a level, not a missing value)
        if bucket_s < 5:
            last, age = None, 0.0
            for i, v in enumerate(power):
                if v is not None:
                    last, age = v, 0.0
                elif last is not None and age + bucket_s <= 3.0:
                    age += bucket_s
                    power[i] = last
    ser["power_sum_w"] = power
    ser["per_w_60s"] = {}
    for k in keys:
        ser[k + "_per_w"] = [(v / p) if (v is not None and p) else None for v, p in zip(ser[k], power)]
        # mean of the tok/s/W curve over the last 60 s; rest intervals count as 0
        last = [x for x in ser[k + "_per_w"][-max(1, int(60 / bucket_s)):] if x is not None]
        ser["per_w_60s"][k] = (sum(last) / len(last)) if last else None


def attach_containers_ipc(boots, containers):
    """The boot's container by the name its state.json carries (``container``), no log dir involved."""
    by_name = {c.get("Names"): c for c in containers or []}
    claimed = set()
    # one container name serves boot after boot of a line: it belongs to the newest one only (boots come
    # live first, then newest first), an older boot of the same name must not look running
    order = sorted(boots, key=lambda b: (not b.get("live"), -(b.get("first_t") or 0)))
    for b in order:
        name = (b.get("ipc") or {}).get("container")
        hit = by_name.get(name)
        if hit is not None and name not in claimed:
            claimed.add(name)
            b["container"] = {k: hit.get(k) for k in ("Names", "Image", "Status", "State", "Ports", "RunningFor",
                                                      "health_output")}
    return boots


def attach_containers(boots, containers):
    """Give every boot the container that writes its logs, if any."""
    containers = containers or []
    by_dir = {}
    for c in containers:
        d = c.get("evidence_dir")
        if d:
            by_dir.setdefault(os.path.normpath(d), []).append(c)
    claimed = set()
    for b in boots:
        cands = by_dir.get(os.path.normpath(b["meta"].get("dir", "")), [])
        tag = b["meta"].get("tag") or b["stem"]
        hit = None
        for c in cands:
            tok = _container_token(c.get("Names", ""))
            if tok and tok in tag:
                hit = c
                break
        if hit is None and cands and b["live"]:
            hit = cands[0]
        if hit is not None and hit.get("Names") not in claimed:
            claimed.add(hit.get("Names"))
            b["container"] = {k: hit.get(k) for k in ("Names", "Image", "Status", "State", "Ports", "RunningFor", "health_output")}
    return boots


class App:
    def __init__(self, args):
        self.args = args
        # the boot cards from IPC only (NF-Operator 30.09.: keine Anzeige liest mehr ein Boot-Log);
        # live.LiveLogs, the old log reader, is no longer started.
        # Sampler in its own process (Nutzer 30.09. ~21:40Z): with a state dir the readings (IPC ring,
        # NVML, host, history) run in ``python -m rigdash.sampler``, started and watched here; this
        # process only reads ring.sqlite / history.sqlite.  Without a state dir (dev) all in one process.
        self.sampler_mode = getattr(args, "sampler", None) or ("prozess" if args.state_dir else "thread")
        self.ring_store = None
        self.sup = None
        if self.sampler_mode == "prozess":
            self.ring_store = sampler.RingStore(os.path.join(args.state_dir, sampler.RING_FILE))
            self.boots = ipcboot.IpcBoots(store=self.ring_store, role="reader")
            env = dict(os.environ)
            pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            cmd = [sys.executable, "-m", "rigdash.sampler", "--state-dir", args.state_dir,
                   "--docker-ssh", args.docker_ssh or "", "--parent-pid", str(os.getpid()),
                   "--docker-host-prefix", args.docker_host_prefix or "", "--gpuq", args.gpuq]
            for ep in args.front or []:
                cmd += ["--front", ep]
            if getattr(args, "vm_url", ""):
                cmd += ["--vm-url", args.vm_url]
            self.sup = sampler.Supervisor(cmd, env=env, store=self.ring_store)
        else:
            self.boots = ipcboot.IpcBoots()
        cfg = {
            "gpu_period": 2.0,
            "docker_ssh": shlex.split(args.docker_ssh) if args.docker_ssh else [],
            "docker_host_prefix": args.docker_host_prefix,
            "weg2_fronts": args.front or [],
            "gpuq": args.gpuq,
            "state_dir": args.state_dir or None,
        }
        # the cards (NVML), docker, gpuq, fronts and the energy book are the sampler's too (30.09. ~22Z):
        # this process reads what it published and measures nothing
        self.src = sources.SourcesReader(self.ring_store) if self.sup is not None else sources.Sources(cfg)
        self.weg2 = weg2line.Weg2Lines(cfg["docker_ssh"], args.release_profile or [])
        self.energy = (energy.EnergyReader(self.ring_store, live.BUCKET_S) if self.sup is not None
                       else energy.EnergyBook(args.state_dir or None, live.BUCKET_S))
        self.imgchg = imagechanges.ImageChanges(args.image_changes)
        self.imgchg_lock = threading.Lock()
        self.features = features.Features(args.features, args.features_repo)
        self.stop = threading.Event()
        # DASHBOARD-GRAFIKEN: the persistent history of the Grafana-style panels (history.py)
        self.hist = history.HistoryDB(os.path.join(args.state_dir, "history.sqlite") if args.state_dir else None)
        self.hist_rec = history.Recorder(self.hist, self.boots, cfg["docker_ssh"]) if self.sup is None else None
        self.view_held = {"total": 0, "last": 0}
        self.t0 = time.time()
        self.version = _version()
        self.edition = getattr(args, "edition", "rig") or "rig"
        # Nutzer-Order 01.10. ~07:40Z: rigdash liest die Zeitreihen per PromQL aus VictoriaMetrics
        self.vm = vmpush.VmClient(args.vm_url) if getattr(args, "vm_url", "") else None
        self.vm_boot_cache: dict = {}         # stem -> (fetched_t, vmpush.boot_rates) of finished boots
        self.live_cache = LiveCache(lambda: self.snapshot(True, None))

    def vm_boot(self, b: dict, now: float) -> Optional[dict]:
        """A finished boot's whole-boot prefill / decode out of VictoriaMetrics (Nutzer 02.10.: "Letzte Boots" showed no
        tok/s once the boot left the 16-min ring).  Refetched each minute for 30 min after the end (the sampler's
        last sums land seconds after it), then kept."""
        if self.vm is None or b.get("live"):
            return None
        hit = self.vm_boot_cache.get(b["stem"])
        if hit is not None and (now - hit[0] < 60.0 or (b.get("age_s") or 0) > 1800.0):
            return hit[1]
        try:
            v = vmpush.boot_rates(self.vm, b["stem"])
        except Exception as e:  # noqa: BLE001 -- VM down must not empty the page
            v = {"prefill": {}, "decode": None, "error": "%s: %s" % (type(e).__name__, e)}
        self.vm_boot_cache[b["stem"]] = (now, v)
        return v

    def energy_loop(self, stop: threading.Event):
        """Every 5 s: account the closed 5-s intervals of every live boot (energy.py); which class
        computed comes from the rank counters' deltas (ipcboot.IpcBoots.activity), not from a log."""
        while not stop.is_set():
            try:
                now = time.time()
                gs = self.src.gpu_series()
                for b in self.boots.snapshot(now):
                    if not b.get("live") or b.get("first_t") is None:
                        continue
                    self.energy.update(b["stem"], b["first_t"],
                                       lambda start, n, bs, k=b["stem"]: self.boots.activity(k, start, n, bs),
                                       lambda start, n, bs: energy.power_buckets(gs, start, n, bs), now)
                self.energy.save(now)
            except Exception as e:  # keep accounting alive; visible in /api/live
                self.energy_error = "%s: %s" % (type(e).__name__, e)
            stop.wait(5.0)

    def start(self):
        loops = [(self.boots.run_forever, "rigdash-ipcboots")]
        if self.sup is None:
            loops += [(self.src.run_forever, "rigdash-sources"), (self.energy_loop, "rigdash-energy")]
        loops.append((self.sup.run_forever, "rigdash-sampler-supervisor") if self.sup is not None
                     else (self.hist_rec.run_forever, "rigdash-history"))
        for target, name in loops:
            threading.Thread(target=target, args=(self.stop,), name=name, daemon=True).start()

    def sampler_status(self) -> dict:
        if self.sup is not None:
            return self.sup.status()
        r = self.hist_rec
        return {"mode": "thread", "ok": True, "pid": os.getpid(), "server_pid": os.getpid(),
                "held": r.held_view() if r else None, "why": None}

    def history_view(self, model, rng, lo_hi=None) -> dict:
        v = history.view(self.hist, self.hist_rec, model, rng, lo_hi=lo_hi)
        if self.vm is not None:
            # Nutzer 01.10.: TTFT mit Verlauf -- aus VictoriaMetrics (PromQL), nicht aus history.sqlite
            v["ttft"] = vmpush.ttft_series(self.vm, model, v.get("t") or [], int(v.get("step") or 5))
            v["pcie"] = vmpush.pcie_series(self.vm, v.get("t") or [], int(v.get("step") or 5))
        n = (v.get("held") or {}).get("view_filled") or 0
        self.view_held["total"] += n
        self.view_held["last"] = n
        return v

    def snapshot(self, with_series=True, zoom=None) -> dict:
        now = time.time()
        sv = self.src.view()
        boots = self.boots.snapshot(now, zoom=zoom)
        docker = sv.get("docker", {}).get("value")
        attach_containers_ipc(boots, docker)
        fronts = {k: v for k, v in sv.items() if k.startswith("front:")}
        for b in boots:
            b["front"] = sources.front_for_boot(fronts, b["meta"].get("tag"))
            ipc = b.get("ipc") or {}
            if b["front"] is None and ipc.get("front") and not ipc.get("terminal"):
                # the front's own /weg2/state is unreachable: the host's mirror in state.json, not a log line
                b["front"] = dict(ipc["front"], src="state.json front (Host-Spiegel)")
            b["alarm"] = health.assess(b, now)
        gser = self.src.gpu_series() if with_series else None
        for b in boots:
            finish_series(b, gser, now, live.BUCKET_S)
            if b.get("series_zoom"):
                finish_series(b, gser, now, live.BUCKET_S, key="series_zoom")
            b["energy"] = self.energy.view(b["stem"], (b.get("totals") or {}).get("boot_wall_s"))
            b["vm_boot"] = self.vm_boot(b, now)
        with self.imgchg_lock:
            images, img_err = self.imgchg.load()
        return {
            "t": now,
            "version": self.version,
            "uptime_s": round(now - self.t0, 1),
            "boots": boots,
            "gpus": sv.get("gpus"),
            "pcie": sv.get("pcie"),
            "gpu_series": gser,
            "docker": sv.get("docker"),
            "gpuq": sv.get("gpuq"),
            "fronts": {k: {kk: vv for kk, vv in v.items() if kk != "value"} for k, v in fronts.items()},
            "collector_error": self.boots.last_error,
            "sampler": self.sampler_status(),
            "energy_error": getattr(self, "energy_error", None),
            "image_changes": imagechanges.view(boots, images, img_err, self.imgchg.path),
            "features": features.attach_current(self.features.view(), boots, sv.get("gpus")),
            "vm": vmpush.tiles(self.vm) if self.vm is not None else None,
            "windows": {"rate_s": live.WINDOW_S, "bucket_s": live.BUCKET_S, "history_s": live.HISTORY_S,
                        "live_s": live.LIVE_S},
        }


#: Nutzer 01.10. ("vollkommen buggy", api/live 5,2 s nach 8,5 h): /api/live was computed per request -- every
#: open page (and the external proxy) recomputed all boots every 2 s, 0,4-1 s CPU each under the GIL, so the
#: server ran at 100 % CPU with 3 pages open.  Now one computation per LIVE_TTL_S is shared by all readers.
LIVE_TTL_S = 1.0
#: what the "Letzte Boots" table reads of a boot that is not shown as a card (lean page payload)
LEAN_KEEP = ("stem", "meta", "age_s", "live", "primary", "first_t", "last_log_t", "flip_count", "totals",
             "alarm", "container", "end", "stop_count", "error_count", "boot_s", "dur_s", "vm_boot")


def lean_boot(b: dict) -> dict:
    """A boot the page lists only as a table row: no curves, no fields, no timeline (the bulk of /api/live)."""
    out = {k: b.get(k) for k in LEAN_KEEP if k in b}
    out["prefill"] = {g: {"last_burst": {"tps": ((v or {}).get("last_burst") or {}).get("tps")}}
                      for g, v in (b.get("prefill") or {}).items()}
    out["decode"] = {g: {k: (v or {}).get(k) for k in ("gen_tps_last", "gen_tps_boot", "seats_boot", "boot_decode_s")}
                     for g, v in (b.get("decode") or {}).items()}
    ipc = b.get("ipc") or {}
    out["ipc"] = {k: ipc.get(k) for k in ("lifecycle", "terminal", "model", "tag", "boot_id", "cause") if k in ipc}
    # Nutzer 02.10.: "Letzte Boots" shows the Flipzeit p50/p90/max per direction (and "fehlt (Feld X)")
    out["flip_last"] = {d: {k: (x or {}).get(k) for k in ("n", "median", "p90", "max")}
                        | {"newest": {k: ((x or {}).get("newest") or {}).get(k) for k in ("kind", "missing")}}
                        for d, x in (b.get("flip_last") or {}).items()}
    out["lean"] = True
    return out


def shown_boots(boots: list) -> list:
    """Same rule as the page: live (or container running), else the primary boot."""
    live = [b for b in boots if b.get("live") or ((b.get("container") or {}).get("State") == "running")]
    return live or [b for b in boots if b.get("primary")][:1]


def lean_snapshot(snap: dict, with_dev: bool) -> dict:
    out = dict(snap)
    shown = {id(b) for b in shown_boots(snap.get("boots") or [])}
    out["boots"] = [b if id(b) in shown else lean_boot(b) for b in snap.get("boots") or []]
    if not with_dev:
        for k in ("features", "image_changes"):
            out.pop(k, None)
    out["lean"] = True
    return out


class LiveCache:
    """One /api/live computation per LIVE_TTL_S for every reader; the encoded variants are cached too."""

    def __init__(self, compute, ttl: float = LIVE_TTL_S):
        self.compute, self.ttl = compute, ttl
        self.lock = threading.Lock()
        self.t = 0.0
        self.snap = None
        self.enc: dict = {}
        self.stats = {"computed": 0, "served": 0, "last_ms": None}

    def get(self, variant: str, make, gz: bool = False) -> bytes:
        with self.lock:                    # one computes, the others wait for its result instead of computing too
            now = time.time()
            if self.snap is None or now - self.t >= self.ttl:
                t0 = time.time()
                self.snap = self.compute()
                self.t = time.time()
                self.enc = {}
                self.stats["computed"] += 1
                self.stats["last_ms"] = round(1000 * (self.t - t0), 1)
            body = self.enc.get(variant)
            if body is None:
                body = self.enc[variant] = make(self.snap)
            if gz:
                z = self.enc.get(variant + "z")
                if z is None:
                    z = self.enc[variant + "z"] = gzip.compress(body, 5)
                body = z
            self.stats["served"] += 1
            return body


EDITIONS = ("rig", "release")
DEV_BEGIN, DEV_END = "<!--DEV:BEGIN-->", "<!--DEV:END-->"
#: what the published fLLiper edition does not answer at all (the rig's own development state)
RELEASE_DROP_KEYS = ("features", "image_changes", "gpuq")


def edition_page(html: str, edition: str) -> str:
    """The page for one edition (Nutzer 30.09.: "im veröffentlichten fLLiper dashboard soll natürlich
    der untere entwicklungsstanddashboard leer sein oder fehlen").  ``rig`` = unchanged.  ``release``
    = every ``<!--DEV:BEGIN-->…<!--DEV:END-->`` block cut out -- no empty shell, the ids are not in
    the page -- and the title says fLLiper.  One page, one code path; the edition is a switch."""
    if edition != "release":
        return html
    out, i = [], 0
    while True:
        a = html.find(DEV_BEGIN, i)
        if a < 0:
            out.append(html[i:])
            break
        b = html.find(DEV_END, a)
        if b < 0:
            raise ValueError("index.html: DEV:BEGIN ohne DEV:END")
        out.append(html[i:a])
        i = b + len(DEV_END)
    page = "".join(out)
    return (page.replace('<html lang="de">', '<html lang="de" data-edition="release">', 1)
                .replace("<title>Rig-Dashboard</title>", "<title>fLLiper Dashboard</title>", 1)
                .replace('<h1 id="title">Rig-Dashboard</h1>', '<h1 id="title">fLLiper Dashboard</h1>', 1))


#: per boot, what names the image, branch, profile or env (NF-Operator 30.09.: Startform, Image-SHAs,
#: Zweig-, Profilnamen und Env-Schalter gehören zum Entwicklungsstand, im Release fehlen sie)
RELEASE_DROP_IPC = ("launch", "rev", "profile", "image", "tag", "boot_id", "dir", "line", "container", "gpuq_id")
RELEASE_DROP_META = ("launch", "tag", "sha", "profile", "image", "rev")


def edition_snapshot(snap: dict, edition: str) -> dict:
    """/api/live of the release edition: the development state is not answered either."""
    if edition == "release":
        for k in RELEASE_DROP_KEYS:
            snap.pop(k, None)
        for b in snap.get("boots") or []:
            for k in RELEASE_DROP_IPC:
                (b.get("ipc") or {}).pop(k, None)
            for k in RELEASE_DROP_META:
                (b.get("meta") or {}).pop(k, None)
            if isinstance(b.get("container"), dict):
                b["container"] = {k: v for k, v in b["container"].items() if k in ("State", "Status")}
        if isinstance(snap.get("docker"), dict):     # the container table (names, images) is dev; the chip keeps age/error
            snap["docker"] = {k: v for k, v in snap["docker"].items() if k != "value"}
        snap["edition"] = "release"
    return snap


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "rigdash/" + app.version

        def log_message(self, fmt, *a):  # quiet; journald gets errors only
            pass

        def _send(self, code, body, ctype, gzip_=False):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if gzip_:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, redact.guard(json.dumps(obj, default=str)), "application/json")

        def _via_proxy(self) -> bool:
            # the public reverse proxy (LXC 208 nginx, https://efeu.ddnss.de/rigdash/) sets these
            return bool(self.headers.get("X-Forwarded-Prefix") or self.headers.get("X-Forwarded-For"))

        def _weg2(self, path):
            if self._via_proxy():
                # the dry run executes a check script on the Proxmox host: LAN only
                return self._json({"ok": False, "error": "Startzeile und Trockenlauf nur im LAN (http://192.168.0.88:8890/weg2)"}, 403)
            from urllib.parse import parse_qs, urlsplit

            q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
            if not app.weg2.release_profiles:
                return self._json({"ok": False, "error": "keine Release-Profile konfiguriert (--release-profile)"}, 400)
            if path == "/api/weg2/options":
                return self._json(dict(app.weg2.options(), ok=True))
            built = app.weg2.build(q.get("profile", ""), q.get("image", ""), q.get("transport", "bar1"),
                                   q.get("house_guard", "memlimit"))
            if path == "/api/weg2/line":
                return self._json(dict(built, ok=True))
            if path == "/api/weg2/dry":
                return self._json(dict(app.weg2.dry_run(built), ok=True, line=built))
            return self._send(404, "not found", "text/plain")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
                        return self._send(200, edition_page(fh.read(), app.edition), "text/html; charset=utf-8")
                if path == "/api/live":
                    series = "noseries" not in self.path
                    zoom = parse_zoom(self.path)
                    if zoom is not None or not series:
                        snap = app.snapshot(series, zoom)          # a zoomed stretch: its own computation
                        snap["via_proxy"] = self._via_proxy()
                        return self._json(edition_snapshot(snap, app.edition))
                    from urllib.parse import parse_qs, urlsplit
                    q = parse_qs(urlsplit(self.path).query)
                    lean = (q.get("lean") or ["0"])[0] == "1"
                    dev = (q.get("dev") or ["1"])[0] == "1"
                    via = self._via_proxy()
                    variant = "%d%d%d" % (lean, dev, via)

                    def make(snap, lean=lean, dev=dev, via=via):
                        x = dict(lean_snapshot(snap, dev) if lean else snap, via_proxy=via)
                        return redact.guard(json.dumps(edition_snapshot(x, app.edition), default=str)).encode("utf-8")
                    gz = "gzip" in (self.headers.get("Accept-Encoding") or "")
                    body = app.live_cache.get(variant, make, gz)
                    return self._send(200, body, "application/json", gzip_=gz)
                if path == "/api/history":
                    # DASHBOARD-GRAFIKEN: tiles + series + marks of one model over one range
                    from urllib.parse import parse_qs, urlsplit

                    q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                    model = q.get("model", "27B")
                    if model not in history.MODELS:
                        raise ValueError("model muss 27B oder NF sein")
                    lo_hi = None
                    if q.get("from") and q.get("to"):
                        lo_hi = (float(q["from"]), float(q["to"]))     # a zoomed stretch (Klicken und Ziehen)
                    return self._json(app.history_view(model, q.get("range", "1h"), lo_hi=lo_hi))
                if path in STATIC_FILES:
                    name, ctype = STATIC_FILES[path]
                    with open(os.path.join(STATIC, name), "rb") as fh:
                        return self._send(200, fh.read(), ctype)
                if path in ("/logo.svg", "/logo-dark.svg", "/mark.svg", "/favicon.svg"):
                    name = "mark.svg" if path == "/favicon.svg" else path[1:]
                    with open(os.path.join(STATIC, name), "rb") as fh:
                        return self._send(200, fh.read(), "image/svg+xml")
                if app.edition == "release" and (path in ("/weg2", "/weg2.html", "/api/launch")
                                                 or path.startswith("/api/weg2/")):
                    return self._send(404, "not found", "text/plain")
                if path in ("/weg2", "/weg2.html") and self._via_proxy():
                    return self._send(403, "Startzeile nur im LAN: http://192.168.0.88:8890/weg2", "text/plain; charset=utf-8")
                if path in ("/weg2", "/weg2.html"):
                    with open(os.path.join(STATIC, "weg2.html"), "rb") as fh:
                        return self._send(200, fh.read(), "text/html; charset=utf-8")
                if path.startswith("/api/weg2/"):
                    return self._weg2(path)
                if path == "/api/launch":
                    # Startflags + ENV je Modell aus state.json (launchview); ?ver= spart den Körper,
                    # solange sich das Gezeigte nicht geändert hat
                    from urllib.parse import parse_qs, urlsplit

                    have = (parse_qs(urlsplit(self.path).query).get("ver") or [""])[0]
                    snap = launchview.snapshot()
                    return self._json({"ver": snap["ver"], "same": True} if have == snap["ver"] else snap)
                if path == "/api/health":
                    smp = app.sampler_status()
                    return self._json({"ok": True, "version": app.version,
                                       "uptime_s": round(time.time() - app.t0, 1),
                                       "live_cache": app.live_cache.stats,
                                       # the sampler's own process, and the emergency fills ("held", target 0)
                                       "sampler": smp, "held": {"sampler": smp.get("held"), "view_filled": app.view_held}})
                return self._send(404, "not found", "text/plain")
            except BrokenPipeError:
                return None
            except ValueError as e:
                return self._json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                return self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rigdash", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--docker-ssh", default="ssh -o BatchMode=yes -o ConnectTimeout=5 proxmox",
                    help="command prefix that reaches the Docker host ('' disables)")
    ap.add_argument("--docker-host-prefix", default="/spinning/subvol-999-disk-0",
                    help="where the Docker host sees this machine's filesystem")
    ap.add_argument("--front", action="append", default=[],
                    help="weg2 front base URL to read /weg2/state from (repeatable)")
    ap.add_argument("--gpuq", default="http://127.0.0.1:8770")
    ap.add_argument("--vm-url", default=os.environ.get("RIGDASH_VM_URL", vmpush.DEFAULT_URL),
                    help="VictoriaMetrics (PromQL lesen; der Probennehmer schreibt die IPC dorthin); '' = aus")
    ap.add_argument("--sampler", choices=("prozess", "thread"), default=None,
                    help="prozess = the readings in their own process (default with --state-dir); thread = in this one")
    ap.add_argument("--state-dir", default="", help="keeps the card history and history.sqlite (the panels' days)")
    ap.add_argument("--image-changes", default=imagechanges.DEFAULT_PATH,
                    help="the operator's per-image change list (rev -> fixes, expected gain, metal status)")
    ap.add_argument("--features", default=features.DEFAULT_PATH,
                    help="the feature list (built / in image / active / gain; im Image and aktiv are computed here)")
    ap.add_argument("--features-repo", default=features.DEFAULT_REPO,
                    help="git repo holding the image revs and feature commits")
    ap.add_argument("--release-profile", action="append", default=[],
                    help="profile name offered by the start-line wizard (repeatable; the unit names the release ones)")
    ap.add_argument("--edition", choices=EDITIONS, default=os.environ.get("RIGDASH_EDITION", "rig"),
                    help="rig = with the development state (Soll/Ist, Bausteine, Startflags, Sitze ...); "
                         "release = the published fLLiper edition: speed, efficiency, statistics only "
                         "(env RIGDASH_EDITION)")
    args = ap.parse_args(argv)
    app = App(args)
    app.start()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    srv.daemon_threads = True
    print("rigdash %s listening on %s:%d" % (app.version, args.host, args.port), flush=True)
    try:
        srv.serve_forever()
    finally:
        app.stop.set()
        if app.sup is not None:
            app.sup.terminate()
    return 0
