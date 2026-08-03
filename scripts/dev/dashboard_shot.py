"""Headless screenshot + state probe of the planner dashboard (#533).

Standing user rule 2026-08-03: a UI change is accepted only against the REAL
RENDERED browser view. API probes are not acceptance evidence -- the #533
defect was precisely a page that stayed on a placeholder while every API call
answered 200.

Reuses the playwright pattern of scripts/translator/client_gate.py (chromium
launch, console capture, goto) rather than introducing a second harness.

Usage:
  dashboard_shot.py --url http://127.0.0.1:8780 --out DIR --label before
  dashboard_shot.py ... --settle-s 20   # let the poll loop produce deltas
"""
import argparse
import asyncio
import json
import re
from pathlib import Path


async def run(args):
    from playwright.async_api import async_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    console: list[str] = []
    result: dict = {"label": args.label, "url": args.url}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        context = await browser.new_context(viewport={"width": 1600, "height": 1200})
        page = await context.new_page()
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
        # Record every XHR the PAGE makes -- the user view, not our guess at it.
        xhr: list[str] = []
        page.on("request", lambda r: xhr.append(f"{r.method} {r.url}")
                if "/api/" in r.url else None)
        failed: list[str] = []
        page.on("requestfailed", lambda r: failed.append(
            f"{r.method} {r.url} :: {r.failure}"))

        await page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        # Let the page's own poll loop run: rates need at least two polls.
        await page.wait_for_timeout(int(args.settle_s * 1000))

        shot = out / f"{args.label}_full.png"
        await page.screenshot(path=str(shot), full_page=True)
        result["screenshot"] = str(shot)

        body = await page.inner_text("body")
        result["body_chars"] = len(body)
        # Acceptance signal: numbers in the page, not a placeholder.
        placeholders = [p for p in ("connect to", "reading the running server",
                                    "no server", "not running", "Find a server")
                        if p.lower() in body.lower()]
        numbers = re.findall(r"\d+\.\d+|\b\d{2,}\b", body)
        result["placeholders_present"] = placeholders
        result["numeric_tokens"] = len(numbers)
        result["sample_numbers"] = numbers[:15]
        result["xhr_api_calls"] = sorted(set(xhr))[:20]
        result["requests_failed"] = failed[:10]
        result["console"] = console[:20]
        result["body_head"] = body[:600]

        await browser.close()

    (out / f"{args.label}_state.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "body_head"}, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8780")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--settle-s", type=float, default=20.0)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
