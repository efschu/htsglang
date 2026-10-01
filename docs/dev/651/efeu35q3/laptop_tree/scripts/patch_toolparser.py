#!/usr/bin/env python3
"""Add a gated --tool-call-parser to boot_ondemand.sh.

Why this is needed. The model already emits correct tool calls -- captured
verbatim from a real agent turn:

    <tool_call>
    <function=write>
    <parameter=path>/home/efeu/.../fib.py</parameter>
    ...

That is Qwen's tool-call syntax. sglang AUTO-DETECTS it at load
("Auto-detected template features: ... tool_call_parser=qwen3_coder") but
detection is not activation: server_args showed `tool_call_parser=None`, so the
server never converted it into an OpenAI `tool_calls` object and handed the raw
XML back as assistant text. The agent, quite correctly, printed it and did
nothing -- no file was written. So the agent loop was never broken by the GPU
or the context; the server was simply not parsing tool calls.

Gated behind an env var so the default command line is unchanged when unset.
"""
import re
import sys

PATH = "/root/651-p2/scripts/boot_ondemand.sh"

BLOCK = '''
# TOOL CALL PARSING. Unset = appends NOTHING, so the established command line is
# unchanged. sglang auto-DETECTS this checkpoint's parser at load and logs it,
# but detection does not activate it: without --tool-call-parser the server
# returns the model's raw <tool_call><function=...> XML as assistant text, and
# every OpenAI-protocol coding agent sees a chatty message with no tool_calls
# field and does nothing. qwen3_coder is the value the server's own detection
# reports for this checkpoint.
TOOLPARSER=${TOOLPARSER:-}
TOOLPARSER_ARGS=()
if [ -n "$TOOLPARSER" ]; then
  TOOLPARSER_ARGS=(--tool-call-parser "$TOOLPARSER")
fi

'''

def main() -> int:
    src = open(PATH).read()
    if "TOOLPARSER_ARGS" in src:
        print("already patched")
        return 0

    marker = "exec python -m sglang.launch_server"
    if marker not in src:
        print("ERROR: launch line not found", file=sys.stderr)
        return 1
    src = src.replace(marker, BLOCK.lstrip("\n") + marker, 1)

    # Insert the array into the argument list, just before --enable-metrics.
    if '  --enable-metrics \\\n' not in src:
        print("ERROR: --enable-metrics anchor not found", file=sys.stderr)
        return 1
    src = src.replace(
        '  --enable-metrics \\\n',
        '  "${TOOLPARSER_ARGS[@]}" \\\n  --enable-metrics \\\n',
        1,
    )

    open(PATH, "w").write(src)
    print("patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
