# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The log button, proved by PRESSING IT.

WHY THIS EXISTS, and it is not a style preference. The telemetry button
shipped after a gate that proved the page loads with no console errors. The
user pressed it and got `log upload failed: playback.debug is not a function`
-- the handler had never once been executed. "The page parses" and "the
feature works" are different claims, and only the second one matters to
somebody holding a phone. So the smoke test here IS the user's gesture: a real
click on the real element, and the proof is a FILE ON THE SERVER plus the
success text the user would read.

HOW IT REACHES A REAL SERVER WITHOUT DEPLOYING FIRST. The client file in this
worktree IS the deployment -- whatever sits at `client/index.html` is being
served to the phone right now. So a candidate must be provable BEFORE it is
flipped into place. This harness serves the candidate from a temporary
directory and proxies `POST /api/translator/client-logs` to the live tenant,
so the click travels the whole way to the store and back while the served file
is untouched. After the gate passes, the flip is one copy (§17.8.23).

    /spinning/htsglang-gpu/.venv/bin/python scripts/translator/probe_log_upload.py \\
      --client python/sglang/srt/translator/client/index.dev.html

`--sabotage` points the proxy at a dead path: the click must then produce a
VISIBLE failure. A button that fails silently is the defect this whole probe
exists to prevent, so the arm that proves the failure path is not optional.
"""
from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import shutil
import socketserver
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_CLIENT = Path(
    "/spinning/wt-466-translator/python/sglang/srt/translator/client/index.dev.html"
)
DEFAULT_TENANT = "http://192.168.0.101:30800"


def make_server(directory: Path, tenant: str, sabotage: bool):
    """Static files, plus one proxied POST so the click reaches the store."""
    seen = {}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, *a, **kw):
            pass

        def do_POST(self):  # noqa: N802 - stdlib naming
            length = int(self.headers.get("content-length", 0))
            body = self.rfile.read(length)
            target = tenant + (
                "/api/translator/no-such-endpoint" if sabotage
                else "/api/translator/client-logs"
            )
            request = urllib.request.Request(
                target, data=body,
                headers={"content-type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read()
                    status = response.status
            except urllib.error.HTTPError as exc:
                payload, status = exc.read(), exc.code
            except Exception as exc:  # noqa: BLE001 - reported to the page
                payload, status = json.dumps({"error": str(exc)}).encode(), 502
            seen["status"] = status
            seen["body"] = payload.decode("utf-8", "replace")
            seen["sent_bytes"] = len(body)
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1], seen


async def main(client: Path, tenant: str, sabotage: bool, shot: Path) -> int:
    from playwright.async_api import async_playwright

    staging = Path(tempfile.mkdtemp())
    shutil.copy(client, staging / "index.html")
    httpd, port, seen = make_server(staging, tenant, sabotage)
    errors, ok = [], False
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                args=["--autoplay-policy=no-user-gesture-required"]
            )
            context = await browser.new_context(viewport={"width": 390, "height": 780})
            page = await context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            await page.goto(f"http://127.0.0.1:{port}/index.html")
            await page.wait_for_selector("#sendlogs")

            # THE GESTURE. Not `sendLogs()` called from evaluate -- a real
            # click on the real element, which is what the user does and what
            # the previous gate never did.
            await page.click("#sendlogs")

            # The toast is the user-visible half of the contract.
            # ALL toasts, not the first. The page raises unrelated toasts of
            # its own (a language fetch that this harness does not serve), and
            # reading only `querySelector` picked one of those up and called a
            # working upload a failure.
            toast = ""
            for _ in range(80):
                await page.wait_for_timeout(100)
                toasts = await page.evaluate(
                    "() => Array.from(document.querySelectorAll('#toasts .text'))"
                    ".map(n => n.textContent)"
                )
                hit = [t for t in toasts if "upload" in t.lower()]
                if hit:
                    toast = hit[0]
                    break
                toast = " | ".join(toasts)
            label = await page.evaluate(
                "() => document.getElementById('sendlogs').textContent"
            )
            await page.screenshot(path=str(shot), full_page=False)
            print(f"[probe] toast   : {toast!r}")
            print(f"[probe] button  : {label!r}")
            print(f"[probe] proxied : status={seen.get('status')} "
                  f"sent={seen.get('sent_bytes')} bytes")
            await browser.close()
    finally:
        httpd.shutdown()
        shutil.rmtree(staging, ignore_errors=True)

    if sabotage:
        # The failure must be VISIBLE. Silence here is the defect.
        ok = ("failed" in toast.lower()) and not errors
        print("[probe] expected a visible failure")
    else:
        answer = {}
        try:
            answer = json.loads(seen.get("body") or "{}")
        except ValueError:
            pass
        stored = Path(answer.get("path", "/nonexistent"))
        ok = (
            seen.get("status") == 200
            and answer.get("stored") is True
            and stored.exists()
            and answer.get("entries", 0) >= 1
            and "uploaded" in toast.lower()
            and not errors
        )
        print(f"[probe] stored at: {stored} exists={stored.exists()}")
        print(f"[probe] entries  : {answer.get('entries')}")
    if errors:
        print(f"[probe] PAGE ERRORS: {errors[:3]}")
    print(f"[probe] screenshot: {shot}")
    print("[probe]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=Path, default=DEFAULT_CLIENT)
    parser.add_argument("--tenant", default=DEFAULT_TENANT)
    parser.add_argument("--sabotage", action="store_true")
    parser.add_argument("--screenshot", type=Path,
                        default=Path("/tmp/log_upload_click.png"))
    args = parser.parse_args()
    sys.exit(asyncio.run(
        main(args.client, args.tenant, args.sabotage, args.screenshot)
    ))
