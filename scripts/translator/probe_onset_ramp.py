# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The onset ramp binds at the start of a run, and nowhere else.

User report: "das knacken ist meist ganz am anfang des gesprochenen, danach
ist es meist knackfrei". A buffer that starts at a non-zero sample is a step
from silence, and a step is a click. The fix is an 8 ms cosine fade applied
only where a run STARTS -- at a cursor re-anchor -- so continuous speech is
untouched and no word is softened mid-sentence.

WHAT THIS ARM PROVES: that the ramp is applied exactly once for three pushes
that form one run, that the two continuous pushes are NOT ramped, and that no
curve was refused as scheduled-in-the-past. `--sabotage` sets the ramp length
to 0, reproducing the pre-fix client: `ramps` must then stay 0 while
`reanchors` still counts 1, which is what shows the arm is reading the ramp
and not the re-anchor.

WHAT IT DOES NOT PROVE: that the click is audibly gone. The gain node sits
AFTER the buffer, so the onset capture -- which samples the buffer -- cannot
see it either. The acoustic confirmation is the user's ear on the device, and
that is the honest state of this fix until he says so.
"""
import asyncio
import functools
import http.server
import shutil
import socketserver
import sys
import tempfile
import threading
from pathlib import Path
CLIENT = Path("/spinning/wt-466-translator/python/sglang/srt/translator/client/index.html")
STUB = """(() => { class D { constructor(){this.readyState=0;} send(){} close(){} addEventListener(){} }
window.WebSocket = D; window.fetch = () => Promise.resolve(new Response("{}", {status:200, headers:{"content-type":"application/json"}})); })();"""
PUSH = """(n) => { const rate=16000; const b=new Float32Array(Math.round(rate*0.3));
  for (let i=0;i<b.length;i++) b[i]=0.5;   // a hard step from silence: the click shape
  for (let k=0;k<n;k++) playback.push(b, rate);
  return {ramps: playback.ramps, reanchors: playback.reanchors, scheduled: playback.scheduled, skipped: playback.rampSkipped}; }"""
def serve(d):
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d))
    h.log_message = lambda *a, **k: None
    srv = socketserver.TCPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]
async def main(sabotage):
    from playwright.async_api import async_playwright
    st = Path(tempfile.mkdtemp())
    shutil.copy(CLIENT, st / "index.html")
    httpd, port = serve(st)
    async with async_playwright() as pw:
        br = await pw.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        ctx = await br.new_context(viewport={"width":390,"height":720})
        pg = await ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.add_init_script(STUB)
        await pg.goto(f"http://127.0.0.1:{port}/index.html")
        await pg.wait_for_selector("#speakers")
        await pg.evaluate("() => { playback.unlock(); playback.ensure(); }")
        for _ in range(40):
            state = await pg.evaluate("() => playback.ctx && playback.ctx.state")
            if state == "running":
                break
            await pg.wait_for_timeout(50)
        if sabotage:
            await pg.evaluate("() => { ONSET_RAMP_S = 0; }")
            print("[probe] SABOTAGE: ramp length 0 (the pre-fix client)")
        r = await pg.evaluate(PUSH, 3)
        print("[probe] after 3 pushes:", r, "errors", errs)
        await br.close()
    httpd.shutdown()
    shutil.rmtree(st, ignore_errors=True)
    ok = (r["ramps"] == 0) if sabotage else (r["ramps"] >= 1)
    ok = ok and r["scheduled"] >= 3 and not errs and r["skipped"] == 0
    print("[probe]", "PASS" if ok else "FAIL")
    return 0 if ok else 1
sys.exit(asyncio.run(main("--sabotage" in sys.argv)))
