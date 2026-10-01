#!/usr/bin/env python3
"""#655 needle-in-haystack probe against the on-demand service.

Purpose: prove the KV pool is not merely ALLOCATED but written and read back.
A pool-size log line cannot do that. So three distinct facts are planted at the
start, the middle and the end of a long prompt, and the model is asked for all
three in one reply. A truncated or silently-capped context loses the early
needle first, which is exactly the failure this is here to catch.

Usage: needle655.py <target_prompt_tokens> [outfile]

Reports prefill wall time and decode tok/s at the achieved depth, taken from
the server's own usage counters rather than from an estimate.
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:31651/v1/chat/completions"
MODEL = "qwen36-35b-a3b"

# Needles are deliberately arbitrary: no amount of world knowledge lets the
# model guess them, so a correct answer can only come from the context.
NEEDLES = [
    ("START", "The calibration code for the north sensor array is VIOLET-7734."),
    ("MIDDLE", "The reserve pump in bay four is rated at 412 litres per minute."),
    ("END", "The maintenance window for the west turbine opens at 03:47 UTC."),
]

FILLER = (
    "Routine operations continue as scheduled across all monitored sectors. "
    "Ambient readings remain within nominal tolerances and no action is required. "
    "The duty log records no exceptions for this interval. "
)


def build_prompt(target_tokens: int) -> str:
    # ~0.75 words per token for prose of this kind; the exact ratio does not
    # matter because the achieved depth is read back from the server's usage.
    target_words = int(target_tokens * 0.75)
    filler_words = FILLER.split()
    body, count = [], 0
    while count < target_words:
        body.append(FILLER)
        count += len(filler_words)
    third = len(body) // 3
    body.insert(0, NEEDLES[0][1] + " ")
    body.insert(third, NEEDLES[1][1] + " ")
    body.append(NEEDLES[2][1] + " ")
    return "".join(body)


def main() -> int:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    outfile = sys.argv[2] if len(sys.argv) > 2 else None

    haystack = build_prompt(target)
    question = (
        "\n\nThree specific facts are stated somewhere in the text above. "
        "Answer with exactly three lines, nothing else:\n"
        "1. The calibration code for the north sensor array\n"
        "2. The litres-per-minute rating of the reserve pump in bay four\n"
        "3. The UTC time the maintenance window for the west turbine opens\n"
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": haystack + question}],
        "max_tokens": 120,
        "temperature": 0,
        # Streamed so that prefill and decode are MEASURED separately rather
        # than inferred: time to the first token is the prefill wall, and the
        # remaining tokens give the decode rate at this depth. Solving for the
        # two from a single non-streamed wall time is not possible once prefix
        # caching makes a second request's prefill nearly free.
        "stream": True,
        "stream_options": {"include_usage": True},
        # Thinking off: this checkpoint otherwise spends the reply budget on a
        # reasoning preamble, which here is latency without benefit.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"}
    )

    t0 = time.time()
    ttft = None
    chunks = []
    usage = {}
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    chunks.append(piece)
    total_s = time.time() - t0

    text = "".join(chunks)
    ptok = usage.get("prompt_tokens", 0)
    ctok = usage.get("completion_tokens", len(chunks))

    decode_s = total_s - (ttft or 0.0)
    decode_tok_s = (ctok - 1) / decode_s if ctok > 1 and decode_s > 0 else 0.0
    prefill_tok_s = ptok / ttft if ttft else 0.0

    lines = [
        f"=== needle probe {time.strftime('%Y-%m-%dT%H:%M:%S')} ===",
        f"target_prompt_tokens={target}",
        f"achieved_prompt_tokens={ptok}",
        f"completion_tokens={ctok}",
        f"prefill_wall_s={ttft:.2f}" if ttft else "prefill_wall_s=NA",
        f"prefill_tok_s={prefill_tok_s:.1f}",
        f"decode_s={decode_s:.2f}",
        f"decode_tok_s={decode_tok_s:.2f}",
        f"total_wall_s={total_s:.2f}",
    ]
    lines.append("--- reply ---")
    lines.append(text)
    lines.append("--- needle check ---")

    ok = True
    for label, expect_key in (
        ("START", "VIOLET-7734"),
        ("MIDDLE", "412"),
        ("END", "03:47"),
    ):
        hit = expect_key in text
        ok = ok and hit
        lines.append(f"{label} {expect_key}: {'FOUND' if hit else 'MISSING'}")
    lines.append(f"NEEDLE-PROBE: {'PASS' if ok else 'FAIL'}")

    out = "\n".join(lines)
    print(out)
    if outfile:
        with open(outfile, "w") as fh:
            fh.write(out + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
