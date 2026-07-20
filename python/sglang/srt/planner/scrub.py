# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Privacy scrubbing for the issue-text generators (design §4.5).

Applied to EVERYTHING auto-collected before it reaches a copy-paste block or
a prefilled URL: filesystem paths -> basename, tokens/secrets dropped,
hostnames / usernames / IPs removed, GPU UUIDs dropped. A submission is thus
anonymous — hardware is identified by card model + count, never by machine
identity. Nothing here is best-effort-optional: the generators route all free
text through these functions.
"""

from __future__ import annotations

import os
import re
from typing import List

__all__ = [
    "scrub_path",
    "scrub_text",
    "scrub_launch_flags",
    "scrub_log_excerpt",
    "error_window",
]

#: Secrets that KEEP a prefix and redact the value (``KEY=<redacted>``).
_SECRET_KV_PATTERNS = [
    re.compile(r"((?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\s*[=:]\s*)\S+"),
    re.compile(r"([A-Za-z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)\s*[=:]\s*)\S+", re.I),
    re.compile(r"(Authorization:\s*)\S+", re.I),
]
#: Bare credential strings dropped wholesale (design §4.5).
_SECRET_BARE_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
]

#: GPU UUIDs (fingerprintable — design §4.5 drops them, keeps name/VRAM/arch).
_UUID_PATTERNS = [
    re.compile(r"GPU-[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"),
    re.compile(r"MIG-[0-9a-fA-F-]{20,}"),
    re.compile(
        r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"
    ),
]

#: Non-loopback IPv4 (127.0.0.1 is kept).
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

#: A unix path (abs or ~-home). Reduced to its basename so a model / log /
#: hotset path leaks no directory structure or usernames.
_PATH_PATTERN = re.compile(r"(?<![\w])(?:~|/)[\w./+\-]+")

#: Long flags whose VALUE is a path — the value is basenamed in place.
_PATH_FLAGS = (
    "--model",
    "--model-path",
    "--tokenizer",
    "--tokenizer-path",
    "--boot-log",
    "--served-model-name",
    "--speculative-draft-model-path",
    "--load-format",
    "--download-dir",
)
_PATH_ENVS = ("SGLANG_MOE_HOTSET_FILE",)


def scrub_path(value: str) -> str:
    """Reduce a filesystem path to its basename (design §4.5). A trailing
    slash directory keeps its last component; an empty result falls back to
    the original stripped of leading path separators."""
    if not value:
        return value
    v = value.rstrip("/")
    base = os.path.basename(v)
    return base or v.lstrip("/")


def _scrub_ips(text: str) -> str:
    def repl(m):
        ip = m.group(0)
        return ip if ip == "127.0.0.1" else "<ip>"

    return _IP_PATTERN.sub(repl, text)


def scrub_text(text: str) -> str:
    """Full scrub of one blob of collected text: secrets, UUIDs, non-loopback
    IPs, ``$USER``/home expansion, then paths -> basename."""
    if not text:
        return text
    out = text
    for pat in _SECRET_KV_PATTERNS:
        out = pat.sub(lambda m: m.group(1) + "<redacted>", out)
    for pat in _SECRET_BARE_PATTERNS:
        out = pat.sub("<redacted>", out)
    for pat in _UUID_PATTERNS:
        out = pat.sub("<uuid>", out)
    out = _scrub_ips(out)

    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if user and len(user) >= 3:
        out = re.sub(rf"\b{re.escape(user)}\b", "<user>", out)
    home = os.path.expanduser("~")
    if home and home != "~":
        out = out.replace(home, "~")

    # Path -> basename, but never mangle URLs (keep scheme://host/path intact
    # only up to being a path; issue URLs are generated separately).
    out = _PATH_PATTERN.sub(lambda m: scrub_path(m.group(0)), out)
    return out


def scrub_launch_flags(flags: List[str]) -> List[str]:
    """Scrub a list of launch-flag tokens: path-valued flags/envs get their
    value basenamed; everything else passes through ``scrub_text``."""
    out: List[str] = []
    for tok in flags:
        scrubbed = tok
        for flag in _PATH_FLAGS:
            if tok.startswith(flag + " "):
                _, _, val = tok.partition(" ")
                scrubbed = f"{flag} {scrub_path(val)}"
                break
        else:
            for env in _PATH_ENVS:
                if tok.startswith(env + "="):
                    _, _, val = tok.partition("=")
                    scrubbed = f"{env}={scrub_path(val)}"
                    break
            else:
                scrubbed = scrub_text(tok)
        out.append(scrubbed)
    return out


def error_window(log_text: str, radius: int = 30) -> str:
    """Extract the window around the first error/traceback line (±``radius``
    lines), so a bug report carries the relevant excerpt and not the whole
    log (design §4.3/§4.5). Falls back to the last ``2*radius`` lines when no
    error marker is found."""
    lines = log_text.splitlines()
    if not lines:
        return ""
    markers = (
        "Traceback (most recent call last)",
        "Error",
        "error",
        "Exception",
        "assert",
        "CUDA out of memory",
        "OOM",
        "NCCL",
        "FAILED",
    )
    hit = None
    for i, line in enumerate(lines):
        if any(m in line for m in markers):
            hit = i
            break
    if hit is None:
        window = lines[-2 * radius :]
    else:
        window = lines[max(0, hit - radius) : hit + radius + 1]
    return "\n".join(window)


def scrub_log_excerpt(log_text: str, radius: int = 30, max_chars: int = 4000) -> str:
    """The error window, scrubbed and length-capped (design §4.5)."""
    excerpt = error_window(log_text, radius)
    excerpt = scrub_text(excerpt)
    if len(excerpt) > max_chars:
        excerpt = excerpt[-max_chars:]
        excerpt = "…(truncated)\n" + excerpt[excerpt.index("\n") + 1 :]
    return excerpt
