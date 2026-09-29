"""Headless check (Nutzer 29.09.: Dashboard darf beim Refresh nicht springen).

Opens rigdash, opens a <details> in the feature table, scrolls into the middle of the feature
table, waits past the 10-s reading pause, then samples every 2 s: scrollY, the viewport top of the
element that sat at reading height, whether it is still the same node, and whether the details is
still open. With --stress the boot card above grows/shrinks by a random height on every refresh.
"""
import asyncio
import glob
import json
import subprocess
import sys
import time
import urllib.request

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8891/"
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 60
STRESS = "--stress" in sys.argv
PORT = int(__import__("os").environ.get("CDP_PORT", "9337"))


async def main():
    b = glob.glob("/root/.cache/ms-playwright/chromium_headless_shell-1234/*/chrome-headless-shell")[0]
    pr = subprocess.Popen([b, "--no-sandbox", "--disable-gpu", f"--remote-debugging-port={PORT}",
                           "--window-size=1400,900", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=2))
                break
            except Exception:
                time.sleep(0.2)
        ws_url = [t for t in tabs if t.get("type") == "page"][0]["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=2**26) as ws:
            mid = 0

            async def cmd(method, **params):
                nonlocal mid
                mid += 1
                my = mid
                await ws.send(json.dumps({"id": my, "method": method, "params": params}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == my:
                        return m.get("result", m.get("error"))

            async def ev(expr):
                r = await cmd("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
                return (r.get("result") or {}).get("value", r)

            await cmd("Emulation.setDeviceMetricsOverride", width=1400, height=900, deviceScaleFactor=1, mobile=False)
            await cmd("Page.navigate", url=URL)
            for _ in range(60):
                await asyncio.sleep(1)
                if await ev("document.querySelectorAll('#prod tr').length > 5"):
                    break
            if STRESS:
                await ev("""(() => { const _bc = bootCard; bootCard = (...a) => _bc(...a)
                    + '<div style="height:' + (Math.random() * 400 | 0) + 'px;background:#f003">stress</div>'; return 1; })()""")
            # open one details in the middle of the feature table, then scroll into the middle of it
            print("setup", await ev("""(() => {
              const rows = [...document.querySelectorAll('#prod tr')];
              const det = document.querySelector('#prod details');
              if (det) det.open = true;
              const target = rows[Math.floor(rows.length / 2)];
              const y = target.getBoundingClientRect().top + window.scrollY - 300;
              window.scrollTo(0, y);
              window.__det = det;
              return {rows: rows.length, det: det && det.id, y, scrollH: document.documentElement.scrollHeight};
            })()"""))
            await asyncio.sleep(12)          # past the reading pause (the scrollTo counts as reading)
            await ev("""(() => { window.__ref = document.elementFromPoint(700, 300);
                window.__refTop = window.__ref.getBoundingClientRect().top; window.__y0 = window.scrollY; return 1; })()""")
            t0 = time.time()
            worst = 0.0
            samples = []
            subs = set()
            while time.time() - t0 < DURATION:
                await asyncio.sleep(2)
                s = await ev("""(() => ({y: window.scrollY, y0: window.__y0,
                    dTop: window.__ref.getBoundingClientRect().top - window.__refTop,
                    same: window.__ref.isConnected, open: window.__det ? window.__det.open : null,
                    sub: document.getElementById('sub').textContent, pause: document.getElementById('pausenote').textContent,
                    shift: window.__lastShift, H: document.documentElement.scrollHeight}))()""")
                subs.add(s["sub"])
                worst = max(worst, abs(s["dTop"]))
                samples.append(s)
            print("samples", len(samples), "distinct sub", len(subs), "last", json.dumps(samples[-1], ensure_ascii=False))
            print("max |dTop| px", round(worst, 1), "all connected", all(x["same"] for x in samples),
                  "details open", all(x["open"] in (True, None) for x in samples),
                  "anchor corrections", sorted({round(x["shift"] or 0) for x in samples})[:12])
            # (4) pause: a wheel event stops redraws for ~10 s
            await ev("window.dispatchEvent(new WheelEvent('wheel', {deltaY: 0})), 1")
            await asyncio.sleep(2.5)            # the note is written on the next 2-s tick
            print("pause after wheel:", await ev("document.getElementById('pausenote').textContent"))
            ok = worst < 1 and all(x["same"] for x in samples) and all(x["open"] in (True, None) for x in samples)
            print("RESULT", "OK" if ok else "FAIL")
    finally:
        pr.kill()


asyncio.run(main())
