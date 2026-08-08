#!/usr/bin/env python3
"""#651: a coding agent small enough to run on this laptop's GPU.

WHY NOT AN OFF-THE-SHELF AGENT. Two were tried. oh-my-pi installs and is
correctly wired, but its system prompt plus 32 built-in tools measures 17029
tokens, and an agent prompt of that size is thousands of tokens of SUSTAINED
PREFILL -- the one regime this gfx1103 iGPU fails in, with `Triton HIP 719` in
userspace and `GPU reset ... device wedged` in dmesg. aider does not install at
all: this machine's Python is 3.14 and numpy has no wheel for it, so the build
fails.

So the tool surface is the problem, and the fix is to have almost none. This
agent's entire system prompt is a few hundred tokens, which keeps prefill in
the range the GPU demonstrably survives. That is the whole design rationale:
every feature below was weighed against the prefill it costs.

WHY NOT JSON TOOL-CALLING. The usual function-calling protocol spends hundreds
of tokens on schemas before the model has said anything, and a heavily
quantized model emits malformed JSON often enough that the retries cost more
prefill again. The protocol here is line-oriented and needs no schema: the
model answers with exactly one action, and a mis-formatted action costs one
short retry rather than a parse-and-repair loop.

SAFETY. Commands run as the invoking user with no elevation, inside a working
directory the user chose, and every command is echoed before it runs. There is
no sandbox: this is a local assistant for a single trusted user on their own
laptop, and pretending otherwise with a half-sandbox would be worse than being
explicit about it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = os.environ.get("EFEU_CODE_ENDPOINT", "http://127.0.0.1:31651/v1")
DEFAULT_MODEL = os.environ.get("EFEU_CODE_MODEL", "qwen36-35b-a3b")

#: Deliberately short. Every token here is paid as prefill on EVERY turn, and
#: prefill is the failure mode on this hardware.
SYSTEM_PROMPT = """You are a coding assistant on a Linux machine.
Reply with EXACTLY ONE action per message, in one of these forms:

WRITE <path>
```
<full file content>
```

RUN <shell command>

DONE <one line summary>

Rules: no explanations outside the action. Use WRITE to create or replace a
file. Use RUN to execute a command and see its output. Use DONE when the task
is finished."""

MAX_OUTPUT_CHARS = 4000


def call_model(endpoint: str, model: str, messages: list, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            # This checkpoint reasons out loud by default, which would spend the
            # reply budget before reaching the action line.
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    # No timeout ceiling by default: the endpoint is an on-demand service and a
    # cold model load legitimately takes minutes. A client timeout here would
    # look like a broken agent.
    with urllib.request.urlopen(req, timeout=1800) as resp:
        payload = json.load(resp)
    return payload["choices"][0]["message"]["content"]


def parse_action(text: str):
    """Return ``(kind, argument, body)`` for the first action in ``text``."""
    lines = text.strip().splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("DONE"):
            return "DONE", line[4:].strip(), None
        if line.startswith("RUN "):
            return "RUN", line[4:].strip(), None
        if line.startswith("WRITE "):
            path = line[6:].strip()
            # Take the fenced block that follows; tolerate a language tag.
            rest = lines[i + 1 :]
            start = None
            for j, r in enumerate(rest):
                if r.strip().startswith("```"):
                    start = j + 1
                    break
            if start is None:
                return None, None, None
            body = []
            for r in rest[start:]:
                if r.strip().startswith("```"):
                    break
                body.append(r)
            return "WRITE", path, "\n".join(body) + "\n"
    return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal local coding assistant")
    ap.add_argument("task", nargs="+", help="what to do")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--cwd", default=".")
    args = ap.parse_args()

    workdir = os.path.abspath(args.cwd)
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": " ".join(args.task)},
    ]

    print(f"[efeu-code] endpoint={args.endpoint} model={args.model}")
    print(f"[efeu-code] working directory: {workdir}")
    print("[efeu-code] first reply may take ~2.5 min if the model is parked\n")

    for step in range(1, args.max_steps + 1):
        try:
            reply = call_model(args.endpoint, args.model, messages, args.max_tokens)
        except urllib.error.HTTPError as exc:
            print(f"[efeu-code] endpoint error {exc.code}: {exc.read()[:300]!r}")
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"[efeu-code] endpoint unreachable: {exc}")
            return 2

        kind, arg, body = parse_action(reply)
        if kind is None:
            print(f"[efeu-code] step {step}: no action found, asking again")
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": "Reply with exactly one action: WRITE, RUN or DONE.",
                }
            )
            continue

        messages.append({"role": "assistant", "content": reply})

        if kind == "DONE":
            print(f"[efeu-code] done: {arg}")
            return 0

        if kind == "WRITE":
            target = os.path.abspath(os.path.join(workdir, arg))
            if not target.startswith(workdir + os.sep) and target != workdir:
                # Keep the agent inside the directory the user pointed it at;
                # a path like ../../.bashrc is a mistake worth refusing.
                print(f"[efeu-code] refusing write outside workdir: {arg}")
                messages.append(
                    {"role": "user", "content": "Refused: path outside working dir."}
                )
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w") as fh:
                fh.write(body or "")
            print(f"[efeu-code] wrote {arg} ({len(body or '')} bytes)")
            messages.append({"role": "user", "content": f"Wrote {arg}."})
            continue

        if kind == "RUN":
            print(f"[efeu-code] run: {arg}")
            proc = subprocess.run(
                arg, shell=True, capture_output=True, text=True, timeout=300
            )
            out = (proc.stdout + proc.stderr)[:MAX_OUTPUT_CHARS]
            print(out.rstrip())
            messages.append(
                {
                    "role": "user",
                    "content": f"Exit {proc.returncode}. Output:\n{out}",
                }
            )
            continue

    print("[efeu-code] step limit reached")
    return 1


if __name__ == "__main__":
    sys.exit(main())
