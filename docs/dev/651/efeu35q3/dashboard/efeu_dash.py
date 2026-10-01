#!/usr/bin/env python3
"""efeu-TP14 mini dashboard for the local model service.

Design rule (user, 01.10.): on a laptop nothing may spin while idle. This server
does NO background work at all: it sleeps in accept() and only reads sensors
when a browser asks. The page polls every 2 s while it is VISIBLE and stops
when the tab is hidden or closed. stdlib only.

Reads, per request:
  * front door  127.0.0.1:31651/ondemand/status   (never wakes the model)
  * backend     127.0.0.1:31661/metrics           (only if the model is up)
  * RAPL package energy (intel-rapl:0, AMD RAPL) -> W between two polls
  * amdgpu socket power / iGPU temp / busy %, k10temp CPU temp
  * /proc/meminfo (RAM, swap), /proc/stat (CPU busy %)
Rates (decode / prefill tok/s, TTFT of the last interval) come from the
sglang Prometheus counters, differenced between polls.

    python3 efeu_dash.py [--port 31680] [--host 0.0.0.0]
"""

import argparse
import glob
import json
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FRONT = "http://127.0.0.1:31651"
BACK = "http://127.0.0.1:31661"
_STAGE = "sglang:per_stage_req_latency_seconds"
_lock = threading.Lock()
_last = {}


def _read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _hwmon(name):
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        if _read(f"{d}/name") == name:
            return d
    return None


def _get(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    except Exception:  # noqa: BLE001
        return None


def _metrics(text):
    out = {}
    for ln in (text or "").splitlines():
        if not ln or ln[0] == "#":
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+naNinf]+)$", ln)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        if name.endswith("_bucket"):
            continue
        sel = re.search(r'(?:mode|stage)="([^"]+)"', labels)
        if sel:
            name = f"{name}|{sel.group(1)}"
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


def sample():
    now = time.time()
    s = {"t": now}
    st = _get(f"{FRONT}/ondemand/status")
    s["service"] = json.loads(st) if st else {"state": "front door down"}
    mi = {}
    for ln in (_read("/proc/meminfo", "") or "").splitlines():
        k, v = ln.split(":", 1)
        mi[k] = int(v.split()[0]) / 1048576
    s["ram"] = {"total_gb": mi.get("MemTotal"), "avail_gb": mi.get("MemAvailable"),
                "swap_used_gb": (mi.get("SwapTotal", 0) - mi.get("SwapFree", 0))}
    cpu = [float(x) for x in (_read("/proc/stat", "cpu 0") or "cpu 0").splitlines()[0].split()[1:]]
    busy, idle = sum(cpu) - cpu[3] - (cpu[4] if len(cpu) > 4 else 0), cpu[3] + (cpu[4] if len(cpu) > 4 else 0)
    energy = _read("/sys/class/powercap/intel-rapl:0/energy_uj")
    gpu = _hwmon("amdgpu")
    k10 = _hwmon("k10temp")
    s["gpu"] = {
        "socket_w": float(_read(f"{gpu}/power1_average", _read(f"{gpu}/power1_input", "0")) or 0) / 1e6 if gpu else None,
        "temp_c": float(_read(f"{gpu}/temp1_input", "0")) / 1000 if gpu else None,
        "busy_pct": float(_read("/sys/class/drm/card1/device/gpu_busy_percent", "0") or 0),
        "sclk": next((ln.split(":")[1].strip().rstrip("*").strip() for ln in (_read("/sys/class/drm/card1/device/pp_dpm_sclk", "") or "").splitlines() if ln.endswith("*")), None),
    }
    s["cpu"] = {"temp_c": float(_read(f"{k10}/temp1_input", "0")) / 1000 if k10 else None}
    s["profile"] = _read("/sys/firmware/acpi/platform_profile")
    met = _metrics(_get(f"{BACK}/metrics")) if s["service"].get("state") == "up" else {}
    with _lock:
        prev = dict(_last)
        _last.update(t=now, busy=busy, idle=idle, energy=energy, met=met)
    if prev:
        dt = max(1e-3, now - prev["t"])
        db, di = busy - prev["busy"], idle - prev["idle"]
        s["cpu"]["busy_pct"] = round(100 * db / max(1e-9, db + di), 1)
        if energy and prev.get("energy"):
            de = int(energy) - int(prev["energy"])
            s["package_w"] = round(de / 1e6 / dt, 2) if de >= 0 else None
        pm = prev.get("met") or {}

        def rate(k):
            return (met.get(k, 0) - pm.get(k, 0)) / dt if k in met and k in pm else None

        s["rates"] = {
            "decode_tok_s": rate("sglang:realtime_tokens_total|decode"),
            "prefill_tok_s": rate("sglang:realtime_tokens_total|prefill_compute"),
        }
        # No tokenizer-side TTFT metric on this build: TTFT (server side,
        # without queueing) = request_process + chunked_prefill +
        # prefill_forward stage time per finished prefill.
        stages = ("request_process", "chunked_prefill", "prefill_forward")
        n = met.get(f"{_STAGE}_count|prefill_forward", 0) - pm.get(f"{_STAGE}_count|prefill_forward", 0)
        if n > 0:
            s["rates"]["ttft_s_interval"] = sum(
                met.get(f"{_STAGE}_sum|{x}", 0) - pm.get(f"{_STAGE}_sum|{x}", 0) for x in stages) / n
    if met:
        c = met.get(f"{_STAGE}_count|prefill_forward", 0)
        s["ttft_mean_s"] = (sum(met.get(f"{_STAGE}_sum|{x}", 0) for x in
                                ("request_process", "chunked_prefill", "prefill_forward")) / c) if c else None
        s["kv"] = {"token_usage": met.get("sglang:num_used_tokens", 0) / max(1.0, met.get("sglang:max_total_num_tokens", 1)),
                   "running": met.get("sglang:num_running_reqs"),
                   "queued": met.get("sglang:num_queue_reqs"), "cache_hit_rate": met.get("sglang:cache_hit_rate"),
                   "gen_throughput": met.get("sglang:gen_throughput")}
    return s


PAGE = """<!doctype html><html lang=de><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>efeu Modell</title>
<style>
:root{--bg:#f6f7f9;--fg:#1d2330;--mut:#5b6475;--card:#fff;--ok:#1a7f37;--warn:#9a6700;--bad:#cf222e;--line:#d8dde5}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e6e9ef;--mut:#9aa3b2;--card:#1d2129;--line:#2c323d}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
.c{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.k{color:var(--mut);font-size:12px}.v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.s{font-size:12px;color:var(--mut)}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
</style></head><body><main>
<h1>Lokales Modell <span id=name class=s></span></h1><div class=grid id=g></div>
<p class=s id=foot></p></main><script>
const f=(x,d=1)=>x==null||isNaN(x)?"–":Number(x).toFixed(d);
function card(k,v,s,cls){return `<div class=c><div class=k>${k}</div><div class="v ${cls||""}">${v}</div><div class=s>${s||""}</div></div>`}
let timer=null;
async function tick(){try{const s=await (await fetch("api")).json();const sv=s.service||{};const r=s.rates||{};const kv=s.kv||{};
const st=sv.state||"?";const cls=st=="up"?"ok":st=="parked"?"warn":"bad";
document.getElementById("g").innerHTML=
card("Dienst",st,`Ladevorgänge ${sv.loads??"–"} · Leerlauf ${f(sv.idle_seconds,0)} s / ${f(sv.idle_park_seconds,0)} s · laufend ${sv.inflight??"–"}`,cls)+
card("Decode",f(r.decode_tok_s)+" tok/s",`gen_throughput ${f(kv.gen_throughput)}`)+
card("Prefill",f(r.prefill_tok_s)+" tok/s","Prompt-Token je s (letztes Intervall)")+
card("TTFT",f(r.ttft_s_interval??s.ttft_mean_s,2)+" s",(r.ttft_s_interval!=null?"letztes Intervall":"Mittel seit Laden")+" · Server-seitig, ohne Warteschlange")+
card("KV-Belegung",kv.token_usage==null?"–":f(100*kv.token_usage)+" %",`KV-Plätze ${sv.kv_tokens??"–"} · Cache-Treffer ${kv.cache_hit_rate==null?"–":f(100*kv.cache_hit_rate)+" %"}`)+
card("Paket (CPU+iGPU)",f(s.package_w)+" W",`amdgpu Socket ${f(s.gpu?.socket_w)} W · Profil ${s.profile}`)+
card("iGPU",f(s.gpu?.busy_pct,0)+" %",`${f(s.gpu?.temp_c,0)} °C · ${s.gpu?.sclk||"–"}`)+
card("CPU",f(s.cpu?.busy_pct)+" %",`${f(s.cpu?.temp_c,0)} °C`)+
card("RAM frei",f(s.ram?.avail_gb)+" GB",`von ${f(s.ram?.total_gb)} GB · Swap belegt ${f(s.ram?.swap_used_gb,2)} GB`);
document.getElementById("foot").textContent="Stand "+new Date(s.t*1000).toLocaleTimeString()+" · Abfrage nur bei sichtbarem Tab";
}catch(e){document.getElementById("foot").textContent="keine Verbindung"}}
function start(){if(!timer){tick();timer=setInterval(tick,2000)}}function stop(){clearInterval(timer);timer=null}
document.addEventListener("visibilitychange",()=>document.hidden?stop():start());start();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api"):
            body = json.dumps(sample()).encode()
            ctype = "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31680)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    # poll_interval only matters for shutdown(); a long one means the idle
    # server wakes once an hour instead of twice a second.
    ThreadingHTTPServer((a.host, a.port), H).serve_forever(poll_interval=3600)


if __name__ == "__main__":
    main()
