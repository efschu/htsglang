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
"""The shareable rig artifact: one schema, one scrub, one share path (#271).

ONE MODULE, SEVERAL SOURCES
---------------------------
Two things are worth sending to the project today -- a comm-suite run
(:mod:`sglang.srt.planner.comm_suite`) and the compact hardware profile the
rig has already accumulated (:mod:`sglang.srt.planner.rig_profile_source`) --
and more will follow when the wizard's full profile lands. None of them owns
the schema, the scrub, the curation or the GitHub binding: they all produce
:class:`Measurement` / :class:`Capability` / :class:`ErrorSignature` rows and
hand them here. A second source must not be a second copy of this file.

WHAT GETS SHARED: A CURATED DIGEST, NEVER EVERYTHING
----------------------------------------------------
Everything floods. The user clicks once; this module decides what is worth
sending, with no hand-curation step anywhere in the flow:

* **only load-bearing rows.** A measurement is an AGGREGATE (value + spread +
  n) with its DATE and its CONTEXT. Raw logs, per-iteration samples and
  free-text output never enter the schema, and :func:`curate` strips
  log-shaped keys defensively in case a future source forgets.
* **no duplicates.** Rows are identified by a stable ``id``; the newest
  ``taken_at`` wins, the older ones are counted, not shipped.
* **errors as signatures.** A failure is data, but a hundred instances of one
  failure is one finding with a count. Error text is normalized (numbers,
  hex, addresses, sizes folded out) into a signature that aggregates.
* **delta on re-share.** The previous digest is remembered locally; a second
  share carries what is new or changed and a count of what was carried over
  unchanged. The one-issue-update-in-place format of #152 makes that the
  natural shape -- the issue always shows the current state of the rig.
* **a hard ceiling of ~100 KB.** Over it, the digest AGGREGATES HARDER --
  never truncates. :data:`AGGREGATION_LADDER` is the fixed sequence of steps,
  and the artifact records which step it stopped at, so a reader can tell a
  compact rig from a compacted one.

WHAT NEVER LEAVES
-----------------
Hostnames, IPs, filesystem paths, GPU UUIDs, usernames, rig-env values,
tokens. :func:`scrub_tree` removes them and :func:`assert_anonymized` refuses
to hand out an artifact that still contains any -- and the gate runs inside
:func:`build_digest`, so the preview the user approves has already passed it.
Card MODELS, counts, VRAM, driver/CUDA/NCCL/UCX versions and every measured
number stay exact: that is the payload.

THE SHARE PATH IS #152, UNCHANGED
---------------------------------
Preview (:func:`build_report`, pure, no network) -> explicit confirmation ->
:func:`submit` -> ``github_share.submit(..., confirmed=True)``. One issue per
user, updated in place. The only additions are this artifact's own marker and
title (so a rig report and a throughput report do not overwrite each other)
and an OPTIONAL, opt-in token store so the second and later shares are one
click instead of two.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from sglang.srt.planner import github_share, scrub

__all__ = [
    "ARTIFACT_SCHEMA",
    "AGGREGATION_LADDER",
    "MAX_ARTIFACT_BYTES",
    "SHARE_MARKER",
    "SHARE_TITLE",
    "Capability",
    "ErrorSignature",
    "Measurement",
    "SourceSections",
    "assert_anonymized",
    "build_digest",
    "build_index_body",
    "build_report",
    "comment_marker",
    "compound_fingerprint",
    "error_signature",
    "curate",
    "digest_from_comment",
    "forget_token",
    "have_token",
    "load_last_digest",
    "load_token",
    "machine_tag",
    "merge_digests",
    "rig_fingerprint",
    "save_last_digest",
    "save_token",
    "scrub_tree",
    "submit",
]

ARTIFACT_SCHEMA = "htsglang-rig-artifact/v1"

#: Own marker and title. The MECHANISM is #152's; the ISSUE is separate,
#: because a rig profile and a measured serving result have different
#: lifetimes -- re-running the suite must not overwrite a throughput report.
SHARE_MARKER = "<!-- htsglang-rig-artifact v1 -->"
SHARE_TITLE = "htsglang rig report (comm suite + hardware profile)"

#: Hard ceiling for the serialized artifact. Exceeding it aggregates harder;
#: it never truncates, because a cut-off digest is a digest whose missing
#: rows look like rows that were never measured.
MAX_ARTIFACT_BYTES = 100_000

#: The fixed sequence :func:`curate` walks while the artifact is over the
#: ceiling. Each step removes a whole CLASS of detail from every row, so the
#: digest stays uniform -- a ladder that dropped individual rows would make
#: the surviving ones look like the whole story.
AGGREGATION_LADDER = (
    "full",
    "drop_distribution",     # p5/p95/min go; value + spread + n stay
    "trim_context",          # context reduced to the keys a comparison needs
    "drop_notes",            # per-row prose goes; status and value stay
    "group_measurements",    # rows collapse to per-group aggregates
    "capabilities_only",     # capability table + error signatures + counts
)

#: Context keys kept at the ``trim_context`` rung: the ones without which a
#: number cannot be compared at all.
_CORE_CONTEXT_KEYS = frozenset({
    "op", "size_kib", "world", "backend", "transport", "tp_size",
    "kv_cache_dtype", "quant", "device", "direction", "pair", "model_family",
})

#: Keys that would carry a raw log. No source emits them; stripped anyway,
#: because "a future source forgets" is the failure mode this guards.
_LOG_KEYS = frozenset({
    "log", "logs", "stdout", "stderr", "raw", "output", "boot_log",
    "output_head", "samples", "trace", "excerpt", "body",
})


# ===========================================================================
# Schema
# ===========================================================================
@dataclass
class Measurement:
    """One number worth sharing: an aggregate, its spread, its date, its
    context. Never a sample series, never a log."""

    id: str
    label: str
    source: str
    unit: str
    value: Optional[float] = None
    spread_pct: Optional[float] = None
    n: Optional[int] = None
    p5: Optional[float] = None
    p95: Optional[float] = None
    taken_at: Optional[str] = None
    status: str = "ok"                       # ok | warn | error | absent
    context: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_json(self) -> dict:
        return {
            "id": self.id, "label": self.label, "source": self.source,
            "unit": self.unit, "value": self.value,
            "spread_pct": self.spread_pct, "n": self.n,
            "p5": self.p5, "p95": self.p95, "taken_at": self.taken_at,
            "status": self.status, "context": dict(self.context),
            "note": self.note,
        }

    def fingerprint(self) -> str:
        """What "changed since the last share" compares. Deliberately coarse:
        a value that moved inside its own noise is not a change worth
        re-sending.

        Quantized to two significant digits on a grid fixed by the DECADE,
        not by the value: a tolerance computed from each value separately
        gives two neighbouring numbers two different grids, so nothing would
        ever land in the same bucket and every share would be a full share.
        """
        import math

        v = self.value
        if isinstance(v, (int, float)) and v and math.isfinite(v):
            step = 10.0 ** (math.floor(math.log10(abs(v))) - 1)
            v = round(round(v / step) * step, 12)
        return f"{self.status}|{v}"


@dataclass
class Capability:
    """One line of the capability table: what this rig CAN do, with the
    provenance vocabulary the rest of the dashboard uses."""

    name: str
    value: Any
    provenance: str = "measured"             # measured | estimate | absent
    note: str = ""

    def to_json(self) -> dict:
        return {"name": self.name, "value": self.value,
                "provenance": self.provenance, "note": self.note}


@dataclass
class ErrorSignature:
    """A failure class with a count, not a hundred copies of one failure."""

    signature: str
    count: int = 1
    where: str = ""
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None

    def to_json(self) -> dict:
        return {"signature": self.signature, "count": self.count,
                "where": self.where, "first_seen": self.first_seen,
                "last_seen": self.last_seen}


@dataclass
class SourceSections:
    """What one source contributes. The only thing a source must produce."""

    source: str
    rig: Dict[str, Any] = field(default_factory=dict)
    measurements: List[Measurement] = field(default_factory=list)
    capabilities: List[Capability] = field(default_factory=list)
    errors: List[ErrorSignature] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ===========================================================================
# Rig fingerprint
# ===========================================================================
#: Bumping this invalidates every fingerprint, i.e. splits every rig's
#: history into before and after. Only bump when the SIGNATURE changes shape.
RIG_FP_VERSION = 1

#: RAM buckets in GiB. A class, not a figure: "how much memory does this rig
#: have" separates a 64 GB box from a 256 GB box, and the exact 251 vs 257
#: GiB a kernel reports is noise that would split one rig profile into two.
_RAM_CLASSES = (16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)


def _ram_class_gib() -> Optional[int]:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    gib = pages * size / (1024 ** 3)
    for c in _RAM_CLASSES:
        if gib <= c * 1.08:
            return c
    return _RAM_CLASSES[-1]


def _cpu_model() -> Optional[str]:
    """CPU model name and core count -- a MODEL, like the card models. No
    serial, no microcode, no machine id."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    name = re.sub(r"\s+", " ", name)
                    return f"{name} x{os.cpu_count()}"
    except Exception:
        pass
    return f"unknown x{os.cpu_count()}" if os.cpu_count() else None


def _nic_types() -> List[str]:
    """The DRIVER behind each network interface, sorted and deduped.

    A driver name (``mlx5_core``, ``e1000e``, ``igb``) says what class of
    link this rig has, which is exactly what a cross-rig figure needs to be
    readable. Interface NAMES are not collected: on this rig they come out of
    the rig-env file and are treated as identifying.
    """
    drivers: set = set()
    base = "/sys/class/net"
    try:
        for iface in os.listdir(base):
            if iface == "lo":
                continue
            try:
                drv = os.path.basename(
                    os.path.realpath(os.path.join(base, iface, "device",
                                                  "driver")))
                if drv and drv != "driver":
                    drivers.add(drv)
            except OSError:
                continue
    except OSError:
        return []
    return sorted(drivers)


def _os_family() -> str:
    import platform

    rel = platform.release() or ""
    major_minor = ".".join(rel.split(".")[:2]) if rel else "?"
    return f"{platform.system() or '?'} {major_minor}"


def _major(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    parts = str(v).split(".")
    return parts[0] if parts else None


def _stable_hash(payload: Any, prefix: str = "") -> str:
    import hashlib

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def rig_fingerprint(rig: Dict[str, Any]) -> Dict[str, Any]:
    """The stable identity of a HARDWARE PROFILE -- never of a machine.

    Both sources derive it from this one function, so a comm-suite run and a
    hardware-profile share from the same box land on the same posting. The
    signature is everything that makes two rigs comparable and nothing that
    makes one of them findable:

    IN: card models with VRAM and count, CPU model + core count, RAM class,
    NIC driver types, driver major, CUDA major, OS family, schema version.

    OUT: hostname, serial, GPU UUID, MAC, IP, PCI address, path, username.

    Two physically distinct but identical machines therefore share a
    fingerprint on purpose -- that is what makes "N machines of this profile"
    a sample rather than N duplicate reports.
    """
    cards: Dict[str, int] = {}
    for c in rig.get("cards") or []:
        name = re.sub(r"^NVIDIA\s+GeForce\s+", "",
                      str(c.get("name") or "?")).strip()
        vram = c.get("vram_mib")
        # VRAM rounded to the nearest GiB: two cards of the same model report
        # totals a few MiB apart depending on ECC/driver state, and that must
        # not split one profile in two.
        key = f"{name}:{round((vram or 0) / 1024)}GiB"
        cards[key] = cards.get(key, 0) + 1
    signature = {
        "v": RIG_FP_VERSION,
        "cards": [f"{n}x {k}" for k, n in sorted(cards.items())],
        "cpu": _cpu_model(),
        "ram_class_gib": _ram_class_gib(),
        "nic_types": _nic_types(),
        "driver_major": _major(rig.get("driver")),
        "cuda_major": _major(rig.get("cuda")),
        "os_family": _os_family(),
    }
    label = " / ".join(filter(None, [
        " + ".join(signature["cards"]) or "no cards",
        f"{signature['ram_class_gib']} GiB RAM"
        if signature["ram_class_gib"] else None,
        f"CUDA {signature['cuda_major']}" if signature["cuda_major"] else None,
    ]))
    return {
        "id": _stable_hash(signature, "rig-"),
        "label": label,
        "signature": signature,
        "kind": "single",
    }


def compound_fingerprint(member_ids: Sequence[str], link_type: str
                         ) -> Dict[str, Any]:
    """The identity of a COMBINATION of rigs plus the line between them.

    A cross-rig number (link rate, a collective over the wire, the PP stage
    boundary) is not a property of either rig -- it is a property of the
    pair and the link. Giving it its own fingerprint keeps it out of both
    single-rig postings, where it would be read as something the single rig
    can do on its own.
    """
    members = sorted(set(m for m in member_ids if m))
    payload = {"v": RIG_FP_VERSION, "members": members,
               "link_type": link_type or "unknown"}
    return {
        "id": _stable_hash(payload, "link-"),
        "label": f"{len(members)} rigs over {link_type or 'unknown link'}",
        "signature": payload,
        "kind": "compound",
    }


def machine_tag() -> str:
    """An anonymous, LOCAL tag for this machine, so several identical rigs
    can be counted as a sample instead of overwriting each other.

    A random salt generated once on this machine, hashed. It identifies
    nothing outside this machine (the salt never leaves it, and the tag
    cannot be reversed to anything) -- its only job is to answer "is this the
    same box that posted last time, or a second one of the same model".
    """
    path = os.path.join(_state_dir(), "machine_salt")
    salt = None
    try:
        with open(path) as f:
            salt = f.read().strip()
    except Exception:
        salt = None
    if not salt:
        salt = os.urandom(16).hex()
        try:
            os.makedirs(_state_dir(), mode=0o700, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(salt)
        except Exception:
            pass
    return _stable_hash({"salt": salt}, "m-")[:10]


def comment_marker(fingerprint_id: str, label_suffix: str = "") -> str:
    """The per-fingerprint comment marker: one comment per rig profile.

    ``label_suffix`` is the escape hatch for a user who runs several
    identical machines and wants them apart rather than pooled. Default is
    pooled, because pooled is the more useful sample.
    """
    suffix = re.sub(r"[^A-Za-z0-9_\-]", "", label_suffix or "")[:24]
    tail = f" tag={suffix}" if suffix else ""
    return f"<!-- htsglang-rig-artifact v1 fp={fingerprint_id}{tail} -->"


_ERR_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_ERR_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")


def error_signature(text: str, where: str = "") -> ErrorSignature:
    """Fold one error message into a class.

    Numbers, hex and quoted fragments are what differ between two instances
    of the SAME failure (a size, a pid, a rank, a duration), so they are
    folded out; what remains is the shape of the failure, which is what
    aggregates across rigs.
    """
    body = (text or "").strip().splitlines()
    line = body[0] if body else ""
    line = _ERR_HEX_RE.sub("<hex>", line)
    line = _ERR_NUM_RE.sub("<n>", line)
    line = re.sub(r"'[^']{0,80}'", "<q>", line)
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) > 200:
        line = line[:197] + "..."
    now = time.strftime("%Y-%m-%d", time.gmtime())
    return ErrorSignature(signature=line or "unspecified failure",
                          count=1, where=where, first_seen=now, last_seen=now)


# ===========================================================================
# Anonymization
# ===========================================================================
_IDENTITY_KEYS = frozenset({
    "host", "hostname", "fqdn", "node", "node_id", "nodename",
    "ip", "addr", "address", "url", "endpoint", "peer", "target",
    "uuid", "uuids", "serial", "path", "paths", "model_path", "out",
    "cache_path", "comm_dir", "user", "username", "login", "master_addr",
    "cwd", "home", "ssh", "key", "token", "sn",
})

#: Env-var name prefixes whose VALUES identify this rig (the rig-env file:
#: RIG1_HOST, RIG2_VENV, RDMA_R1, PVE_MINIFORGE, ...). Every such value is
#: removed by LITERAL match: a regex-only scrub would miss an interface name
#: or a bespoke directory that reads as ordinary text.
_RIG_ENV_PREFIXES = ("RIG1_", "RIG2_", "RDMA_", "PVE_", "R3VAL_", "PVEKEY",
                     "MASTER_ADDR", "GLOO_SOCKET_IFNAME", "UCX_NET_DEVICES")

_ABS_PATH_RE = re.compile(r"(?<![\w])(?:~|/)[\w./+\-]{4,}")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_UUID_RE = re.compile(
    r"\b(?:GPU-)?[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b")


def _rig_env_literals() -> List[str]:
    """Every rig-identifying value in this process's env, longest first.

    Longest-first matters: removing ``RIG1_REPO_ROOT`` before ``RIG1_HOST``
    keeps a path that CONTAINS the host from leaving a bare host behind.
    """
    vals: List[str] = []
    for name, val in os.environ.items():
        if not val or len(val) < 4:
            continue
        if any(name.startswith(p) for p in _RIG_ENV_PREFIXES):
            vals.append(val)
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = ""
    if hostname and len(hostname) >= 3:
        vals.append(hostname)
        short = hostname.split(".")[0]
        if len(short) >= 3:
            vals.append(short)
    return sorted(set(vals), key=len, reverse=True)


def _scrub_string(text: str, literals: Sequence[str]) -> str:
    for lit in literals:
        if lit in text:
            text = text.replace(lit, "<rig-env>")
    # scrub.scrub_text is the shared rule (secrets, UUIDs, IPs, $USER, home,
    # paths -> basename). Reused rather than re-expressed so this repo has
    # ONE definition of "anonymous".
    text = scrub.scrub_text(text)
    # Loopback survives scrub.scrub_text by design (it identifies nobody),
    # but in a shared artifact it is noise.
    return text.replace("127.0.0.1", "<local>")


def scrub_tree(obj: Any, literals: Optional[Sequence[str]] = None) -> Any:
    """Drop identity keys and log keys, scrub every string leaf, recursively.

    Applied to the WHOLE artifact rather than to a hand-picked field list: a
    scrub that must be remembered per new field is a scrub that will be
    forgotten at the next field.
    """
    literal_list = list(literals) if literals is not None else _rig_env_literals()
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            tail = str(k).rsplit(".", 1)[-1].lower()
            if tail in _IDENTITY_KEYS or tail in _LOG_KEYS:
                continue
            out[str(k)] = scrub_tree(v, literal_list)
        return out
    if isinstance(obj, (list, tuple)):
        return [scrub_tree(v, literal_list) for v in obj]
    if isinstance(obj, str):
        return _scrub_string(obj, literal_list)
    return obj


def assert_anonymized(artifact: dict) -> None:
    """Raise if anything identifying survived. The gate, not a lint.

    Checks the SERIALIZED form, because a leak in a key is as bad as one in
    a value, and runs inside :func:`build_digest` so nothing that reaches a
    preview can have skipped it.
    """
    blob = json.dumps(artifact)
    problems: List[str] = []
    for m in _IP_RE.finditer(blob):
        problems.append(f"IP address {m.group(0)!r}")
    for m in _UUID_RE.finditer(blob):
        problems.append(f"UUID {m.group(0)!r}")
    for m in _ABS_PATH_RE.finditer(blob):
        problems.append(f"filesystem path {m.group(0)!r}")
    try:
        host = socket.gethostname()
    except Exception:
        host = ""
    if host and len(host) >= 3 and host in blob:
        problems.append("hostname")
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if user and len(user) >= 3 and re.search(rf"\b{re.escape(user)}\b", blob):
        problems.append("username")
    for lit in _rig_env_literals():
        if lit in blob:
            problems.append(f"rig-env value {lit[:12]!r}...")
    if problems:
        raise ValueError(
            "the rig artifact is not anonymous, refusing to hand it out: "
            + "; ".join(sorted(set(problems))[:8]))


# ===========================================================================
# Curation
# ===========================================================================
def _dedupe(measurements: Sequence[Measurement]) -> tuple:
    """Newest row per ``id`` wins. Returns (kept, dropped_count)."""
    best: Dict[str, Measurement] = {}
    dropped = 0
    for m in measurements:
        prev = best.get(m.id)
        if prev is None:
            best[m.id] = m
            continue
        dropped += 1
        if (m.taken_at or "") >= (prev.taken_at or ""):
            best[m.id] = m
    return sorted(best.values(), key=lambda x: x.id), dropped


def _fold_errors(errors: Sequence[ErrorSignature]) -> List[ErrorSignature]:
    folded: Dict[str, ErrorSignature] = {}
    for e in errors:
        cur = folded.get(e.signature)
        if cur is None:
            folded[e.signature] = ErrorSignature(
                signature=e.signature, count=e.count, where=e.where,
                first_seen=e.first_seen, last_seen=e.last_seen)
            continue
        cur.count += e.count
        if e.first_seen and (not cur.first_seen or e.first_seen < cur.first_seen):
            cur.first_seen = e.first_seen
        if e.last_seen and (not cur.last_seen or e.last_seen > cur.last_seen):
            cur.last_seen = e.last_seen
        if e.where and e.where not in cur.where:
            cur.where = (cur.where + ", " + e.where).strip(", ")
    return sorted(folded.values(), key=lambda x: (-x.count, x.signature))


def _group_of(m: Measurement) -> str:
    """``comm/barlink_ucx/all_reduce/20KiB`` -> ``comm/barlink_ucx``. The unit the
    ``group_measurements`` rung collapses to."""
    parts = m.id.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else m.id


def _apply_rung(rows: List[dict], rung: str) -> List[dict]:
    if rung == "drop_distribution":
        for r in rows:
            r.pop("p5", None)
            r.pop("p95", None)
        return rows
    if rung == "trim_context":
        for r in rows:
            ctx = r.get("context") or {}
            r["context"] = {k: v for k, v in ctx.items()
                            if k in _CORE_CONTEXT_KEYS}
        return rows
    if rung == "drop_notes":
        for r in rows:
            r.pop("note", None)
        return rows
    if rung == "group_measurements":
        groups: Dict[str, dict] = {}
        for r in rows:
            g = "/".join(r["id"].split("/")[:2]) if r["id"].count("/") > 1 \
                else r["id"]
            entry = groups.setdefault(g, {
                "id": g, "label": g, "source": r.get("source"),
                "unit": r.get("unit"), "aggregated_rows": 0,
                "values": [], "statuses": set(),
                "taken_at": r.get("taken_at"),
            })
            entry["aggregated_rows"] += 1
            if isinstance(r.get("value"), (int, float)):
                entry["values"].append(r["value"])
            entry["statuses"].add(r.get("status"))
            if (r.get("taken_at") or "") > (entry["taken_at"] or ""):
                entry["taken_at"] = r.get("taken_at")
        out = []
        for g, e in sorted(groups.items()):
            vals = sorted(e.pop("values"))
            statuses = e.pop("statuses")
            e["value"] = vals[len(vals) // 2] if vals else None
            e["min"] = vals[0] if vals else None
            e["max"] = vals[-1] if vals else None
            e["status"] = ("error" if "error" in statuses
                           else "warn" if "warn" in statuses else "ok")
            out.append(e)
        return out
    if rung == "capabilities_only":
        return []
    return rows


def curate(
    sections: Sequence[SourceSections],
    previous: Optional[dict] = None,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> dict:
    """Turn raw source sections into the digest that will actually be shared.

    The four rules of the curation, in the order they are applied:

    1. **dedupe** by ``id``, newest ``taken_at`` wins;
    2. **fold errors** into signatures with counts;
    3. **delta** against ``previous`` -- rows whose fingerprint is unchanged
       are counted, not shipped;
    4. **fit the ceiling** by walking :data:`AGGREGATION_LADDER`, never by
       cutting rows off the end.

    The result still has to pass :func:`scrub_tree` + :func:`assert_anonymized`
    -- that is :func:`build_digest`'s job, and no caller should skip it.
    """
    rig: Dict[str, Any] = {}
    measurements: List[Measurement] = []
    capabilities: List[Capability] = []
    errors: List[ErrorSignature] = []
    notes: List[str] = []
    source_names: List[str] = []
    for s in sections:
        source_names.append(s.source)
        for k, v in (s.rig or {}).items():
            rig.setdefault(k, v)
        measurements.extend(s.measurements)
        capabilities.extend(s.capabilities)
        errors.extend(s.errors)
        notes.extend(s.notes)

    measurements, dup_dropped = _dedupe(measurements)
    errors = _fold_errors(errors)

    cap_by_name: Dict[str, Capability] = {}
    for c in capabilities:
        cap_by_name.setdefault(c.name, c)
    capabilities = sorted(cap_by_name.values(), key=lambda c: c.name)

    carried_over = 0
    delta_against = None
    if previous:
        delta_against = previous.get("generated_at")
        prev_fp = {m.get("id"): m.get("fingerprint")
                   for m in (previous.get("measurements") or [])}
        kept: List[Measurement] = []
        for m in measurements:
            if prev_fp.get(m.id) == m.fingerprint():
                carried_over += 1
                continue
            kept.append(m)
        measurements = kept

    rows = [dict(m.to_json(), fingerprint=m.fingerprint())
            for m in measurements]

    fingerprint = rig_fingerprint(rig)
    tag = machine_tag()
    for r in rows:
        # Every row remembers which machine of this profile produced it, so a
        # merge across identical rigs can count samples instead of guessing.
        r.setdefault("machines", [tag])
        r.setdefault("samples", 1)

    digest: Dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": sorted(set(source_names)),
        "fingerprint": fingerprint,
        "machines": [tag],
        "sample_count": 1,
        "rig": rig,
        "capabilities": [c.to_json() for c in capabilities],
        "measurements": rows,
        "errors": [e.to_json() for e in errors],
        "notes": sorted(set(n for n in notes if n)),
        "curation": {
            "aggregation_level": AGGREGATION_LADDER[0],
            "duplicates_dropped": dup_dropped,
            "carried_over_unchanged": carried_over,
            "delta_against": delta_against,
            "max_bytes": max_bytes,
        },
    }

    for rung in AGGREGATION_LADDER[1:]:
        if len(json.dumps(digest)) <= max_bytes:
            break
        digest["measurements"] = _apply_rung(digest["measurements"], rung)
        if rung == "capabilities_only":
            digest["notes"] = digest["notes"][:3]
        digest["curation"]["aggregation_level"] = rung
    digest["curation"]["bytes"] = len(json.dumps(digest))
    digest["curation"]["measurement_rows"] = len(digest["measurements"])
    return digest


def build_digest(
    sections: Sequence[SourceSections],
    previous: Optional[dict] = None,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> dict:
    """:func:`curate`, then scrub, then the anonymity gate. The only entry
    point a UI may use -- the three steps are not separable in practice, and
    making them separable is how one of them gets skipped."""
    digest = curate(sections, previous=previous, max_bytes=max_bytes)
    digest = scrub_tree(digest)
    assert_anonymized(digest)
    return digest


# ===========================================================================
# Merging identical rigs (same fingerprint, several machines)
# ===========================================================================
def _merge_row(old: dict, new: dict) -> dict:
    """One measurement point seen on two machines of the SAME profile.

    The merged row keeps the newest value as the headline and reports the
    range across machines next to it. Averaging them would hide exactly the
    thing a multi-machine sample is for: whether two identical rigs actually
    behave identically.
    """
    machines = sorted(set((old.get("machines") or [])
                          + (new.get("machines") or [])))
    vals = [v for v in (old.get("value"), new.get("value"))
            if isinstance(v, (int, float))]
    lo = [v for v in ([old.get("min")] if old.get("min") is not None else [])
          + vals]
    hi = [v for v in ([old.get("max")] if old.get("max") is not None else [])
          + vals]
    merged = dict(new)
    merged["machines"] = machines
    merged["samples"] = len(machines)
    if vals:
        merged["min"] = min(lo) if lo else None
        merged["max"] = max(hi) if hi else None
        if merged["min"] is not None and merged["max"] is not None \
                and merged["max"] > 0:
            merged["across_machines_pct"] = round(
                100.0 * (merged["max"] - merged["min"]) / merged["max"], 1)
    return merged


def merge_digests(existing: Optional[dict], new: dict) -> dict:
    """Fold ``new`` into the digest already published for this fingerprint.

    Three things happen, and all three are what makes the one-comment-per-
    fingerprint format work:

    * **rows the new run did not repeat are KEPT.** Delta sharing drops
      unchanged rows from ``new``; the published comment must still show
      them, or a delta share would look like a regression to nothing.
    * **rows seen on both are merged** into a sample with a machine count and
      the range across machines.
    * **the sample count is the number of distinct machines**, not the number
      of shares -- re-sharing from one box does not inflate it.
    """
    if not existing:
        return new
    merged = dict(new)
    old_rows = {r.get("id"): r for r in (existing.get("measurements") or [])}
    new_rows = {r.get("id"): r for r in (new.get("measurements") or [])}
    out_rows: List[dict] = []
    for rid, row in old_rows.items():
        if rid in new_rows:
            out_rows.append(_merge_row(row, new_rows[rid]))
        else:
            carried = dict(row)
            carried["carried_over"] = True
            out_rows.append(carried)
    for rid, row in new_rows.items():
        if rid not in old_rows:
            out_rows.append(row)
    merged["measurements"] = sorted(out_rows, key=lambda r: r.get("id") or "")

    machines = sorted(set((existing.get("machines") or [])
                          + (new.get("machines") or [])))
    merged["machines"] = machines
    merged["sample_count"] = len(machines) or 1

    caps = {c.get("name"): c for c in (existing.get("capabilities") or [])}
    caps.update({c.get("name"): c for c in (new.get("capabilities") or [])})
    merged["capabilities"] = sorted(caps.values(),
                                    key=lambda c: c.get("name") or "")

    errs = _fold_errors(
        [ErrorSignature(**{k: v for k, v in e.items()
                           if k in ("signature", "count", "where",
                                    "first_seen", "last_seen")})
         for e in (existing.get("errors") or []) + (new.get("errors") or [])])
    merged["errors"] = [e.to_json() for e in errs]
    merged["notes"] = sorted(set((existing.get("notes") or [])
                                 + (new.get("notes") or [])))

    cur = dict(merged.get("curation") or {})
    cur["merged_with"] = existing.get("generated_at")
    cur["measurement_rows"] = len(merged["measurements"])
    cur["bytes"] = len(json.dumps(merged))
    merged["curation"] = cur

    # The merge can push the digest back over the ceiling; walk the same
    # ladder rather than letting a merged comment grow without bound.
    for rung in AGGREGATION_LADDER[1:]:
        if len(json.dumps(merged)) <= (cur.get("max_bytes")
                                       or MAX_ARTIFACT_BYTES):
            break
        merged["measurements"] = _apply_rung(merged["measurements"], rung)
        merged["curation"]["aggregation_level"] = rung
    merged["curation"]["bytes"] = len(json.dumps(merged))
    merged["curation"]["measurement_rows"] = len(merged["measurements"])
    return merged


_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.S)


def digest_from_comment(body: str) -> Optional[dict]:
    """Read back the machine-readable digest a published comment carries.

    The comment is the authoritative copy of what this fingerprint has
    already reported, which is what makes both the delta and the multi-
    machine merge possible without any server-side state.
    """
    if not body:
        return None
    for m in _JSON_BLOCK_RE.finditer(body):
        try:
            data = json.loads(m.group(1))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("schema") == ARTIFACT_SCHEMA:
            return data
    return None


def build_index_body(entries: Sequence[dict]) -> str:
    """The ISSUE body: an index of the rig profiles this user has shared.

    The numbers live in the comments, one per fingerprint. The body stays a
    table of contents so the issue is readable at a glance no matter how many
    rigs it accumulates, and so a re-share of rig B never rewrites rig A.
    """
    md: List[str] = []
    md.append("# htsglang rig reports")
    md.append("")
    md.append("Anonymized rig profiles shared from the dashboard's *Rig data "
              "share* tab, so the fork can be optimized against hardware its "
              "authors do not own. No hostnames, IPs, paths or GPU UUIDs -- "
              "card models, versions and measured numbers only.")
    md.append("")
    md.append("One comment per rig profile below; each is updated in place. "
              "Identical machines share a profile and are reported as one "
              "sample with a machine count.")
    md.append("")
    md.append("| Rig profile | Fingerprint | Machines | Sources | Updated |")
    md.append("|---|---|---|---|---|")
    for e in sorted(entries, key=lambda x: str(x.get("id") or "")):
        md.append(f"| {e.get('label', '-')} | `{e.get('id', '-')}` "
                  f"| {e.get('sample_count', 1)} "
                  f"| {', '.join(e.get('sources') or [])} "
                  f"| {e.get('updated', '-')} |")
    md.append("")
    md.append(SHARE_MARKER)
    return "\n".join(md)


_INDEX_ROW_RE = re.compile(
    r"^\|\s*(?P<label>[^|]*?)\s*\|\s*`(?P<id>[^`]+)`\s*\|"
    r"\s*(?P<samples>\d+)\s*\|\s*(?P<sources>[^|]*?)\s*\|"
    r"\s*(?P<updated>[^|]*?)\s*\|\s*$", re.M)


def parse_index_body(body: str) -> List[dict]:
    """The index rows already in the issue body, so a new rig is ADDED to
    the table instead of replacing it."""
    out: List[dict] = []
    for m in _INDEX_ROW_RE.finditer(body or ""):
        out.append({
            "label": m.group("label"),
            "id": m.group("id"),
            "sample_count": int(m.group("samples")),
            "sources": [s.strip() for s in m.group("sources").split(",")
                        if s.strip()],
            "updated": m.group("updated"),
        })
    return out


# ===========================================================================
# Preview
# ===========================================================================
def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def build_report(digest: dict) -> str:
    """The EXACT markdown that would be posted. Pure -- sends nothing.

    This is what the preview shows, verbatim. The user confirms this text or
    nothing gets posted; there is no step in which the page renders one thing
    and the transport sends another.
    """
    rig = digest.get("rig") or {}
    fp = digest.get("fingerprint") or {}
    md: List[str] = []
    md.append(f"### Rig profile `{fp.get('id', '?')}` -- "
              f"{fp.get('label') or rig.get('card_summary', 'unknown rig')}")
    md.append("")
    md.append("Anonymized rig digest from the dashboard's *Rig data share* "
              "tab. Card models, counts, versions and measured numbers only: "
              "no hostname, no IP, no path, no GPU UUID. Shared voluntarily "
              "so the fork can be optimized against real hardware it does "
              "not own.")
    md.append("")
    samples = digest.get("sample_count") or 1
    if samples > 1:
        md.append(f"**{samples} machines of this profile** report here. Each "
                  f"row carries its range across them (`across_machines_pct`) "
                  f"-- identical hardware that behaves differently is itself "
                  f"a finding.")
        md.append("")
    if fp.get("signature"):
        md.append("<details><summary>What the fingerprint is made of</summary>")
        md.append("")
        for k, v in sorted((fp.get("signature") or {}).items()):
            md.append(f"- `{k}`: {v}")
        md.append("")
        md.append("</details>")
        md.append("")

    cur = digest.get("curation") or {}
    md.append(f"*Sources: {', '.join(digest.get('sources') or ['-'])} "
              f"| generated {digest.get('generated_at')} "
              f"| aggregation `{cur.get('aggregation_level')}` "
              f"| {cur.get('measurement_rows')} rows, "
              f"{cur.get('bytes')} bytes*")
    if cur.get("carried_over_unchanged"):
        md.append("")
        md.append(f"*Delta share: {cur['carried_over_unchanged']} rows were "
                  f"unchanged since {cur.get('delta_against')} and are not "
                  f"repeated here; the issue above them still holds.*")
    md.append("")

    md.append("### Rig")
    for key, label in (("card_summary", "Cards"), ("driver", "Driver"),
                       ("cuda", "CUDA"), ("torch", "torch"),
                       ("nccl", "NCCL"), ("ucx", "UCX"),
                       ("commit", "htsglang commit"),
                       ("cpu_count", "CPU threads"),
                       ("interconnect", "Interconnect")):
        if rig.get(key) is not None:
            md.append(f"- {label}: {rig[key]}")
    md.append("")

    caps = digest.get("capabilities") or []
    if caps:
        md.append("### Capabilities")
        md.append("| Capability | Value | Provenance | Note |")
        md.append("|---|---|---|---|")
        for c in caps:
            note = str(c.get("note") or "").replace("|", "/")
            md.append(f"| {c.get('name')} | {_fmt(c.get('value'))} "
                      f"| {c.get('provenance')} | {note} |")
        md.append("")

    rows = digest.get("measurements") or []
    if rows:
        md.append("### Measurements")
        md.append("| Point | Value | Unit | Spread % | n | Status | Context |")
        md.append("|---|---|---|---|---|---|---|")
        for r in rows:
            ctx = ", ".join(f"{k}={v}" for k, v in
                            sorted((r.get("context") or {}).items()))
            if len(ctx) > 90:
                ctx = ctx[:87] + "..."
            md.append(
                f"| {r.get('id')} | {_fmt(r.get('value'))} "
                f"| {r.get('unit', '')} | {_fmt(r.get('spread_pct'))} "
                f"| {_fmt(r.get('n'))} | {r.get('status')} | {ctx} |")
        md.append("")

    errs = digest.get("errors") or []
    if errs:
        md.append("### Error signatures")
        md.append("These are findings, not noise: a failure on real hardware "
                  "is the most useful thing this report carries.")
        md.append("")
        md.append("| Count | Where | Signature |")
        md.append("|---|---|---|")
        for e in errs:
            sig = str(e.get("signature") or "").replace("|", "/")
            md.append(f"| {e.get('count')} | {e.get('where', '')} | {sig} |")
        md.append("")

    notes = digest.get("notes") or []
    if notes:
        md.append("### Reservations")
        for n in notes:
            md.append(f"- {n}")
        md.append("")

    md.append("<details><summary>Machine-readable digest (JSON)</summary>")
    md.append("")
    md.append("```json")
    md.append(json.dumps(digest, indent=1, sort_keys=True))
    md.append("```")
    md.append("")
    md.append("</details>")
    md.append("")
    md.append(comment_marker(fp.get("id", "unknown"),
                             digest.get("label_suffix", "")))
    return "\n".join(md)


# ===========================================================================
# Token store + last-share memory
# ===========================================================================
def _state_dir() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "htsglang")


def _token_path() -> str:
    return os.path.join(_state_dir(), "share_token")


def _last_digest_path() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "sglang", "rig_artifact_last_share.json")


def save_token(token: str) -> str:
    """Remember the PAT so later shares are ONE click.

    OPT-IN, and a deliberate, named deviation from #152's "never on disk":
    the user asked for one-click re-sharing, and the alternative -- typing a
    PAT into a field on every share -- is what makes people paste a
    longer-lived token. Written 0600 in a 0700 directory, never echoed back
    to the page (:func:`have_token` answers yes/no), and removable with one
    call (:func:`forget_token`).
    """
    if not token or not token.strip():
        raise ValueError("refusing to store an empty token")
    d = _state_dir()
    os.makedirs(d, mode=0o700, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    path = _token_path()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token.strip())
    return path


def load_token() -> Optional[str]:
    try:
        with open(_token_path()) as f:
            return f.read().strip() or None
    except Exception:
        return None


def have_token() -> bool:
    """Whether a stored token exists. The page asks THIS, never for the
    value: a token that can be read back into a browser is a token that can
    be read back by anything else that reaches the endpoint."""
    return bool(load_token())


def forget_token() -> bool:
    try:
        os.unlink(_token_path())
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def save_last_digest(digest: dict) -> str:
    """Remember what was shared, so the next share can be a delta.

    The authoritative copy is the issue body; this is the local shortcut that
    lets the PREVIEW be computed without a network round trip and without a
    token.
    """
    path = _last_digest_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    slim = {
        "generated_at": digest.get("generated_at"),
        "measurements": [{"id": m.get("id"), "fingerprint": m.get("fingerprint")}
                         for m in (digest.get("measurements") or [])],
    }
    with open(path, "w") as f:
        json.dump(slim, f)
    return path


def load_last_digest() -> Optional[dict]:
    try:
        with open(_last_digest_path()) as f:
            return json.load(f)
    except Exception:
        return None


# ===========================================================================
# Share (the #152 path, unchanged)
# ===========================================================================
def fetch_published(
    token: str,
    fingerprint_id: str,
    repo: str = github_share.DEFAULT_REPO,
    label_suffix: str = "",
    api=None,
) -> Optional[dict]:
    """The digest already published for this fingerprint, or None.

    READ ONLY -- this is what lets the PREVIEW show the finished, merged
    text rather than a local view that the submit step would then change.
    A preview that does not match what gets posted is not a preview.
    """
    issue = github_share.find_existing_issue(token, repo, marker=SHARE_MARKER,
                                             api=api)
    if issue is None:
        return None
    comment = github_share.find_existing_comment(
        token, repo, issue.get("number"),
        comment_marker(fingerprint_id, label_suffix), api=api)
    if comment is None:
        return None
    return digest_from_comment(comment.get("body") or "")


def submit(
    report: str,
    token: str,
    digest: dict,
    repo: str = github_share.DEFAULT_REPO,
    *,
    confirmed: bool = False,
    label_suffix: str = "",
    api=None,
) -> dict:
    """Publish the previewed report: index in the body, rig in a comment.

    POSTING SENDS DATA TO AN EXTERNAL SERVICE. Without ``confirmed=True``
    this performs NO network call at all -- the check happens before the
    first request, and :func:`github_share.submit` enforces it a second time
    underneath.

    The shape, and why:

    * **one issue per user**, found by :data:`SHARE_MARKER` (the #152
      mechanism, unchanged);
    * **the body is an INDEX** of that user's rig profiles, so sharing rig B
      never rewrites rig A;
    * **one comment per fingerprint**, updated in place, carrying exactly
      ``report`` -- the text the user approved, byte for byte.
    """
    if not confirmed:
        raise github_share.GitHubShareError(
            "refusing to submit: the caller did not confirm. Posting shares "
            "data with an external service (GitHub); show the digest "
            "preview, get the user's explicit approval, then call "
            "submit(..., confirmed=True).")
    if not token:
        raise github_share.GitHubShareError("no GitHub token given")
    fp = digest.get("fingerprint") or {}
    fp_id = fp.get("id") or "unknown"

    existing = github_share.find_existing_issue(
        token, repo, marker=SHARE_MARKER, api=api)
    entries = parse_index_body(existing.get("body") or "") if existing else []
    entries = [e for e in entries if e.get("id") != fp_id]
    entries.append({
        "id": fp_id,
        "label": fp.get("label") or "unknown rig",
        "sample_count": digest.get("sample_count", 1),
        "sources": digest.get("sources") or [],
        "updated": digest.get("generated_at") or "",
    })
    issue = github_share.submit(
        build_index_body(entries), token, repo=repo,
        existing_issue=existing.get("number") if existing else None,
        confirmed=True, title=SHARE_TITLE, marker=SHARE_MARKER, api=api)

    comment = github_share.upsert_comment(
        token, repo, issue.get("number"),
        comment_marker(fp_id, label_suffix), report, api=api)
    return {
        "action": comment.get("action"),
        "issue": issue.get("number"),
        "issue_url": issue.get("url"),
        "comment_url": comment.get("url"),
        "fingerprint": fp_id,
        "sample_count": digest.get("sample_count", 1),
    }
