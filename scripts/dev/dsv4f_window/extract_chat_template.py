#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract, verify and self-test the DeepSeek-V4-Flash-0731 chat template.

WHY THIS EXISTS -- and a correction to the window briefing
----------------------------------------------------------
The briefing said the GGUF checkpoints "carry NO chat_template" and instructed
me to PRODUCE one (a "DeepSeek-V3/V4-style jinja template"). That is half
right and the half that is wrong matters.

* TRUE: neither quant directory's ``tokenizer_config.json`` has a
  ``chat_template`` key, and neither has any ``added_tokens_decoder`` entries.
  So the tokenizer sglang loads from the sidecar HF directory carries no
  template, and ``--chat-template`` really is required.
* FALSE: the GGUF files themselves DO carry ``tokenizer.chat_template`` in
  their metadata KV block -- 13698 bytes of Unsloth-fixed DeepSeek-V4 jinja,
  sha256 e643c31fcec17f342f72296e02c46d35846bf4c70f6a0271f23bad73fd4eb645,
  BYTE-IDENTICAL between UD-IQ3_XXS and UD-Q3_K_XL.

So the correct action is to EXTRACT the authoritative template, not to invent
a plausible one. A desk-written template would have been an unvalidated guess
about token markers (``<|User|>`` / ``<|Assistant|>`` in DeepSeek's fullwidth
form, ``<think>`` handling, the ``<|DSML|>`` tool-call block) with no way to
tell a wrong guess from a right one.

WHAT ELSE THIS CHECKS
---------------------
``TemplateManager._load_jinja_template`` (parser/template_manager.py:264-271)
does NOT load a .jinja file verbatim. It runs::

    chat_template = "".join(f.readlines()).strip("\\n")
    tokenizer.chat_template = chat_template.replace("\\\\n", "\\n")

i.e. it rewrites every literal backslash-n in the FILE into a real newline.
The extracted template contains many literal ``\\n`` sequences inside jinja
string literals (``'...\\n\\n' + tools_header + ...``). Jinja string literals
may span real newlines, so the rewrite should be semantically neutral -- but
"should be" is a testable claim, not a fact. ``--selftest`` renders both the
raw template and the mangled one against fixed message sets and asserts the
outputs are identical, with an executed can-fail arm proving the comparison
can actually detect a difference.

Usage
-----
  python3 extract_chat_template.py --write          # (re)generate the .jinja
  python3 extract_chat_template.py --verify         # file == GGUF metadata?
  python3 extract_chat_template.py --selftest       # hermetic, no GPU, no server
  python3 extract_chat_template.py --render         # print a rendered example
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys

DEFAULT_MODEL_ROOT = "/spinning/llm_stuff/club-3090/models-cache"
GGUF_ROOT = os.path.join(DEFAULT_MODEL_ROOT, "DeepSeek-V4-Flash-0731-GGUF")
SHARDS = {
    "iq3xxs": os.path.join(
        GGUF_ROOT, "UD-IQ3_XXS", "DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
    ),
    "q3kxl": os.path.join(
        GGUF_ROOT, "UD-Q3_K_XL", "DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf"
    ),
}
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEMPLATE = os.path.join(HERE, "dsv4f_chat_template.jinja")

# The sha256 measured on 2026-08-03 from BOTH quants' first shard. Pinned so a
# silent upstream re-quant cannot change the template under the window.
EXPECTED_SHA256 = "e643c31fcec17f342f72296e02c46d35846bf4c70f6a0271f23bad73fd4eb645"

_SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_SCALAR_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_TYPE_STRING = 8
_TYPE_ARRAY = 9


def read_gguf_kv(path: str, want: str) -> str | None:
    """One string-valued metadata key out of a GGUF header. Header bytes only.

    Pure file read -- no CUDA, no model load. Safe to run at desk time on a
    120 GiB checkpoint because it never touches the tensor data section.
    """
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file (magic {magic!r})")
        (version,) = struct.unpack("<I", fh.read(4))
        if version != 3:
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        fh.read(8)  # tensor count
        (n_kv,) = struct.unpack("<Q", fh.read(8))

        def read_str() -> str:
            (n,) = struct.unpack("<Q", fh.read(8))
            return fh.read(n).decode("utf-8")

        for _ in range(n_kv):
            key = read_str()
            (kind,) = struct.unpack("<I", fh.read(4))
            if kind == _TYPE_STRING:
                value = read_str()
                if key == want:
                    return value
            elif kind == _TYPE_ARRAY:
                (elem_kind,) = struct.unpack("<I", fh.read(4))
                (length,) = struct.unpack("<Q", fh.read(8))
                if elem_kind == _TYPE_STRING:
                    for _i in range(length):
                        (slen,) = struct.unpack("<Q", fh.read(8))
                        fh.seek(slen, 1)
                else:
                    fh.seek(_SCALAR_SIZE[elem_kind] * length, 1)
            else:
                fh.read(_SCALAR_SIZE[kind])
    return None


def extract() -> str:
    """The template, cross-checked byte-for-byte across both quants."""
    got: dict[str, str] = {}
    for name, path in SHARDS.items():
        if not os.path.exists(path):
            raise SystemExit(f"missing shard for {name}: {path}")
        value = read_gguf_kv(path, "tokenizer.chat_template")
        if value is None:
            raise SystemExit(f"{path} carries no tokenizer.chat_template")
        got[name] = value
    quants = sorted(got)
    first = got[quants[0]]
    for name in quants[1:]:
        if got[name] != first:
            raise SystemExit(
                f"chat_template differs between {quants[0]} and {name}; the "
                "#478 quant swap would then also be a prompt-format swap and "
                "the arms would not be comparable"
            )
    digest = hashlib.sha256(first.encode("utf-8")).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(
            f"chat_template sha256 is {digest}, pinned value is "
            f"{EXPECTED_SHA256}. The checkpoint changed under this window."
        )
    return first


# ---------------------------------------------------------------------------
# Rendering / selftest
# ---------------------------------------------------------------------------
PROBE_MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "First question."},
    {"role": "assistant", "content": "First answer."},
    {"role": "user", "content": "Second question."},
]

# Markers the template MUST emit for a chat request. Fullwidth forms, taken
# from the extracted source, not from memory.
REQUIRED_MARKERS = ["<｜User｜>", "<｜Assistant｜>", "</think>"]


def _render(template_src: str, messages, **kwargs) -> str:
    """Render through the EXACT environment the server will use.

    ``sglang``'s chat path calls the tokenizer's ``apply_chat_template``, which
    compiles the template with
    ``transformers.utils.chat_template_utils._compile_jinja_template``:
    an ``ImmutableSandboxedEnvironment`` with ``trim_blocks=True``,
    ``lstrip_blocks=True``, the ``loopcontrols`` extension, a non-strict
    ``Undefined``, and exactly three additions (``tojson`` filter,
    ``raise_exception`` and ``strftime_now`` globals)
    -- chat_template_utils.py:480-494.

    Borrowing it rather than approximating it is the point: a hand-rolled
    ``jinja2.Environment`` differs in block trimming and in undefined
    handling, so it would answer a question about a template the server never
    compiles. (A first draft of this file used ``StrictUndefined`` and failed
    on ``message['tool_calls']`` -- a message the server renders fine.)

    KNOWN GAP, recorded not papered over: the template uses ``| from_json``
    (an llama.cpp/minja filter) on assistant ``tool_calls`` whose ``arguments``
    are a string. Transformers defines no ``from_json`` filter, so that ONE
    branch would raise at render time. It is not on any path this window
    exercises (no tool-call replay in the probes), and it is a defect in the
    upstream Unsloth template, not in this extraction.
    """
    from transformers.utils.chat_template_utils import _compile_jinja_template

    tmpl = _compile_jinja_template(template_src)
    return tmpl.render(messages=messages, bos_token="<｜begin▁of▁sentence｜>", **kwargs)


def render_example(template_src: str) -> str:
    return _render(template_src, PROBE_MESSAGES, add_generation_prompt=True)


def selftest(template_src: str) -> int:
    """Hermetic. No GPU, no server, no network. Every arm has a can-fail twin."""
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    print("extract_chat_template selftest")

    # 1 -- the template renders at all, and emits the role markers.
    rendered = render_example(template_src)
    for marker in REQUIRED_MARKERS:
        check(f"renders marker {marker!r}", marker in rendered)
    check(
        "ends with the generation prompt",
        rendered.endswith("<｜Assistant｜></think>"),
        repr(rendered[-40:]),
    )

    # 2 -- CAN-FAIL ARM for check 1: a template that omits the markers must be
    # caught. Without this, "markers present" proves nothing about the check.
    broken = "{{- bos_token -}}{% for m in messages %}{{ m['content'] }}{% endfor %}"
    broken_out = _render(broken, PROBE_MESSAGES, add_generation_prompt=True)
    check(
        "can-fail: a marker-less template is rejected",
        all(marker not in broken_out for marker in REQUIRED_MARKERS),
        "the raw-concatenation control emits no role markers, as required",
    )

    # 3 -- TemplateManager mangling equivalence.
    # parser/template_manager.py:269-270 loads the file as
    #   "".join(readlines()).strip("\n") then .replace("\\n", "\n")
    # Assert that round trip is semantically neutral for THIS template.
    mangled = template_src.strip("\n").replace("\\n", "\n")
    check(
        "the mangling is not a no-op (so the check is not vacuous)",
        mangled != template_src,
        f"{template_src.count(chr(92) + 'n')} literal backslash-n sequences rewritten",
    )
    for label, msgs, kw in (
        ("plain multi-turn", PROBE_MESSAGES, {"add_generation_prompt": True}),
        ("no generation prompt", PROBE_MESSAGES, {"add_generation_prompt": False}),
        ("thinking on", PROBE_MESSAGES, {"add_generation_prompt": True, "thinking": True}),
        (
            "with tools",
            PROBE_MESSAGES,
            {
                "add_generation_prompt": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                        },
                    }
                ],
            },
        ),
    ):
        raw_out = _render(template_src, msgs, **kw)
        man_out = _render(mangled, msgs, **kw)
        check(
            f"file-load mangling is neutral: {label}",
            raw_out == man_out,
            "" if raw_out == man_out else f"len {len(raw_out)} vs {len(man_out)}",
        )

    # 4 -- CAN-FAIL ARM for check 3: the comparison must be able to SEE a
    # difference. Feed it two templates that genuinely differ.
    differing = template_src.replace("</think>", "</thinking>")
    check(
        "can-fail: the equivalence comparison detects a real difference",
        _render(template_src, PROBE_MESSAGES, add_generation_prompt=True)
        != _render(differing, PROBE_MESSAGES, add_generation_prompt=True),
    )

    # 5 -- content-format detection must resolve to 'string'; this template
    # takes message['content'] as a plain string, and 'openai' would make
    # serving_chat wrap it in a list of parts.
    try:
        sys.path.insert(0, os.path.join("/spinning/wt-dsv4f-window", "python"))
        from sglang.srt.parser.jinja_template_utils import (  # noqa: PLC0415
            detect_jinja_template_content_format,
        )

        fmt = detect_jinja_template_content_format(template_src)
        check("content format detects as 'string'", fmt == "string", f"got {fmt!r}")
    except Exception as exc:  # noqa: BLE001 - reported, never masked
        check("content format detection importable", False, str(exc))

    print(f"\n{'SELFTEST PASSED' if not failures else 'SELFTEST FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, help="the .jinja file to write/verify")
    ap.add_argument("--write", action="store_true", help="(re)generate the .jinja from the GGUF metadata")
    ap.add_argument("--verify", action="store_true", help="assert the file matches the GGUF metadata")
    ap.add_argument("--selftest", action="store_true", help="hermetic checks, no GPU and no server")
    ap.add_argument("--render", action="store_true", help="print a rendered example")
    args = ap.parse_args(argv)

    if not any((args.write, args.verify, args.selftest, args.render)):
        ap.error("pick one of --write / --verify / --selftest / --render")

    if args.selftest and not (args.write or args.verify or args.render):
        # Selftest prefers the checked-in file so it exercises what the boot
        # scripts actually pass; falls back to the GGUF when it is absent.
        if os.path.exists(args.template):
            with open(args.template, encoding="utf-8") as fh:
                src = fh.read()
        else:
            src = extract()
        return selftest(src)

    src = extract()

    if args.write:
        with open(args.template, "w", encoding="utf-8") as fh:
            fh.write(src)
        print(f"wrote {len(src)} bytes -> {args.template}")
        print(f"sha256 {hashlib.sha256(src.encode()).hexdigest()}")

    if args.verify:
        if not os.path.exists(args.template):
            print(f"missing template file {args.template}", file=sys.stderr)
            return 2
        with open(args.template, encoding="utf-8") as fh:
            have = fh.read()
        if have != src:
            print(
                f"{args.template} does NOT match the GGUF metadata "
                f"(file sha256 {hashlib.sha256(have.encode()).hexdigest()}, "
                f"gguf sha256 {hashlib.sha256(src.encode()).hexdigest()})",
                file=sys.stderr,
            )
            return 3
        print(f"verified: {args.template} == GGUF tokenizer.chat_template ({EXPECTED_SHA256[:16]}...)")

    if args.render:
        print(render_example(src))

    if args.selftest:
        return selftest(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
