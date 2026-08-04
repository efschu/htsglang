# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The onset ramp binds at both seams, and nowhere else.

User report: "das knacken ist meist ganz am anfang des gesprochenen, danach
ist es meist knackfrei". A buffer that starts at a non-zero sample is a step
from silence, and a step is a click. The fix is an 8 ms cosine fade applied
only where a run STARTS, so continuous speech is untouched and no word is
softened mid-sentence.

THERE ARE TWO SEAMS, and shipping only the first did not stop the report.

  * a CURSOR RE-ANCHOR -- the output clock drained and this buffer starts a
    new run. Covered since the seventh session.
  * a TURN START -- the first buffer of a new turn. A turn that begins while
    the previous turn's tail is still scheduled never drains the clock, so no
    re-anchor fires, and a freshly synthesized waveform is joined directly to
    the last sample of a different one. That is the seam the user kept
    hearing, and `--turns` is the arm for it.

ARMS:

  (none)        three pushes, one run, no turn identity: `ramps` == 1.
  --turns       three pushes, three DIFFERENT turns, pushed back-to-back so
                the clock never drains: `reanchors` stays 1 while
                `turn_starts` and `ramps` reach 3. The pre-fix client scores
                1 here, which is what makes this arm falsifiable.
  --same-turn   THE CONTROL for `--turns`: the same three pushes under ONE
                turn id must still ramp once. If this reached 3 the fix would
                be fading every buffer mid-word.
  --sabotage    sets the ramp length to 0, reproducing the pre-fix client:
                `ramps` must stay 0 while `reanchors` and `turn_starts` still
                count, which is what shows the arm reads the RAMP and not the
                seam. Combines with the arms above.

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
# `turns` selects the turn identity per push: null (no identity, the original
# arm), "one" (all three under one turn), or a distinct id per push. The
# pushes are back-to-back on purpose -- 0.3 s of audio each against a clock
# that advances microseconds -- so the cursor stays ahead and only the FIRST
# push can ever re-anchor. Anything the later two ramp is a turn seam.
PUSH = """([n, turns]) => { const rate=16000; const b=new Float32Array(Math.round(rate*0.3));
  for (let i=0;i<b.length;i++) b[i]=0.5;   // a hard step from silence: the click shape
  for (let k=0;k<n;k++) {
    if (turns === "each") playback.currentTurn = "turn-" + k;
    else if (turns === "one") playback.currentTurn = "turn-only";
    playback.push(b, rate);
  }
  return {ramps: playback.ramps, reanchors: playback.reanchors, turn_starts: playback.turnStarts,
          scheduled: playback.scheduled, skipped: playback.rampSkipped}; }"""
def serve(d):
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d))
    h.log_message = lambda *a, **k: None
    srv = socketserver.TCPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]
async def main(sabotage, turns):
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
        r = await pg.evaluate(PUSH, [3, turns])
        print(f"[probe] after 3 pushes (turns={turns}):", r, "errors", errs)
        await br.close()
    httpd.shutdown()
    shutil.rmtree(st, ignore_errors=True)
    # One re-anchor in every arm: the clock is cold at the first push and
    # never drains afterwards. If this moves, the arm stopped isolating the
    # turn seam and its ramp count means nothing.
    expected_ramps = {None: 1, "one": 1, "each": 3}[turns]
    expected_starts = {None: 0, "one": 1, "each": 3}[turns]
    ok = (r["ramps"] == 0) if sabotage else (r["ramps"] == expected_ramps)
    ok = ok and r["reanchors"] == 1 and r["turn_starts"] == expected_starts
    ok = ok and r["scheduled"] >= 3 and not errs and r["skipped"] == 0
    print(f"[probe] expected ramps {0 if sabotage else expected_ramps}, "
          f"turn_starts {expected_starts}")
    print("[probe]", "PASS" if ok else "FAIL")
    return 0 if ok else 1
_turns = "each" if "--turns" in sys.argv else ("one" if "--same-turn" in sys.argv else None)
sys.exit(asyncio.run(main("--sabotage" in sys.argv, _turns)))
