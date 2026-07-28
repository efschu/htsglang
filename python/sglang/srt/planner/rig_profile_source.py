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
"""The second source of the shared rig artifact: what this rig already knows.

A rig accumulates measurements long before anybody presses the comm-suite
button -- the card probe's rates and ordered pair matrix, the split-probe
rows behind the Benchmark tab's tipping-point table, the stage-0 hardware
profile the boot path calibrates against. All of it is on disk already. This
module reads those caches and produces the SAME
:class:`~sglang.srt.planner.rig_artifact.SourceSections` rows the comm suite
produces, so the share path does not care which one filled it.

READ ONLY, and measures NOTHING. Opening the share tab must never start a
probe or take a card: a source that measured on read would make "look at
what I would share" an action with side effects. Anything absent stays
absent, with the study that would produce it named.

The identity problem this source has and the comm suite does not: these
caches are keyed by GPU UUID and carry model PATHS. Neither may be shared.
Cards are re-keyed to ``model#ordinal`` (``RTX 3080#1``) -- a model plus a
stable position, which is what a comparison needs and is not a machine
identity -- and model paths are reduced to a family label before they ever
reach a row. The artifact's own scrub is the backstop, not the plan.

WHEN THE FULL PROFILE ARRIVES: the wizard's v2 profile is expected to
supersede the split-probe rows read here. It will be another function in
this file feeding the same rows, not another share path.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.planner import rig_artifact

__all__ = [
    "SOURCE_NAME",
    "available",
    "to_sections",
]

#: How this source names itself in the shared artifact.
SOURCE_NAME = "hardware_profile"

#: Known model families, matched against the basename of a model path. A
#: FAMILY is shareable and useful ("this figure came from a 27B MoE"); a path
#: is neither.
_FAMILY_PATTERNS = (
    (re.compile(r"qwen3\.?6", re.I), "Qwen3.6"),
    (re.compile(r"qwen3\.?5", re.I), "Qwen3.5"),
    (re.compile(r"qwen", re.I), "Qwen"),
    (re.compile(r"deepseek", re.I), "DeepSeek"),
    (re.compile(r"llama", re.I), "Llama"),
    (re.compile(r"gemma", re.I), "Gemma"),
    (re.compile(r"mistral|mixtral", re.I), "Mistral"),
    (re.compile(r"gpt-?oss", re.I), "gpt-oss"),
)
_SIZE_RE = re.compile(r"(\d+)\s*[bB]\b")
_QUANT_RE = re.compile(r"(fp8|awq|gptq|q\d_k(?:_[a-z]+)?|int4|bf16|fp16)", re.I)


def model_family(model_path: str) -> str:
    """``/some/dir/Qwen3.6-27B-FP8`` -> ``Qwen3.6 27B FP8``.

    Family, parameter count and quantization survive because they are what
    makes a throughput figure comparable across rigs. The directory does not.
    """
    base = os.path.basename((model_path or "").rstrip("/"))
    if not base:
        return "unknown model"
    parts: List[str] = []
    for pat, name in _FAMILY_PATTERNS:
        if pat.search(base):
            parts.append(name)
            break
    else:
        parts.append("other")
    m = _SIZE_RE.search(base)
    if m:
        parts.append(f"{m.group(1)}B")
    q = _QUANT_RE.search(base)
    if q:
        parts.append(q.group(1).upper())
    return " ".join(parts)


def _cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "sglang")


def _card_key_map(cards: List[dict]) -> Dict[str, str]:
    """UUID -> ``model#ordinal``.

    Ordinal, not CUDA index: two RTX 3080s have to stay apart in a pair
    matrix (one of them sits on the x4 slot and really is slower), and the
    CUDA index is not stable across CUDA_DEVICE_ORDER settings. Ordinal is
    assigned by sorted UUID, so it is stable on THIS machine and meaningless
    off it.
    """
    seen: Dict[str, int] = {}
    out: Dict[str, str] = {}
    for c in sorted(cards, key=lambda x: str(x.get("uuid") or "")):
        model = re.sub(r"^NVIDIA\s+GeForce\s+", "",
                       str(c.get("name") or "?")).strip()
        n = seen.get(model, 0)
        seen[model] = n + 1
        out[str(c.get("uuid"))] = f"{model}#{n}" if n or \
            sum(1 for x in cards
                if re.sub(r"^NVIDIA\s+GeForce\s+", "",
                          str(x.get("name") or "?")).strip() == model) > 1 \
            else model
    return out


def _latest_card_probe() -> Optional[dict]:
    d = _cache_dir()
    try:
        files = [os.path.join(d, f) for f in os.listdir(d)
                 if f.startswith("card_probe-") and f.endswith(".json")]
    except OSError:
        return None
    if not files:
        return None
    newest = max(files, key=lambda p: os.path.getmtime(p))
    try:
        with open(newest) as f:
            return json.load(f)
    except Exception:
        return None


def _split_probe_rows() -> List[dict]:
    path = os.path.join(_cache_dir(), "split_probe.jsonl")
    rows: List[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return []
    return rows


def available() -> Dict[str, Any]:
    """What this source could contribute, WITHOUT reading the whole thing.

    The tab calls this to decide whether to offer the profile share at all,
    and to say what it would contain. Cheap on purpose: a stat and a line
    count, no parsing of every row.
    """
    probe = _latest_card_probe()
    rows = _split_probe_rows()
    return {
        "card_probe": {
            "present": probe is not None,
            "cards": len(probe.get("cards") or []) if probe else 0,
            "pairs": len(probe.get("pairs") or []) if probe else 0,
            "created": probe.get("created_str") if probe else None,
        },
        "split_probe": {
            "present": bool(rows),
            "rows": len(rows),
            "models": sorted({model_family(r.get("model_path", ""))
                              for r in rows}),
        },
        "any": probe is not None or bool(rows),
    }


def _date(ts: Optional[float]) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts or time.time()))


def _card_probe_sections(probe: dict) -> Tuple[
        Dict[str, Any], List[rig_artifact.Measurement],
        List[rig_artifact.Capability], List[str]]:
    cards = probe.get("cards") or []
    keys = _card_key_map(cards)
    taken = _date(probe.get("created"))
    measurements: List[rig_artifact.Measurement] = []
    capabilities: List[rig_artifact.Capability] = []
    notes: List[str] = []

    rig: Dict[str, Any] = {
        "cards": [{"index": c.get("cuda_index"), "name": c.get("name"),
                   "vram_mib": c.get("total_mib"),
                   "sm_clock_mhz": c.get("sm_clock_mhz"),
                   "sm_clock_max_mhz": c.get("sm_clock_max_mhz")}
                  for c in cards],
        "driver": probe.get("driver"),
        "torch": probe.get("torch_version"),
        "cuda": probe.get("cuda_version"),
        "card_count": len(cards),
    }
    counts: Dict[str, int] = {}
    for c in cards:
        name = re.sub(r"^NVIDIA\s+GeForce\s+", "",
                      str(c.get("name") or "?")).strip()
        counts[name] = counts.get(name, 0) + 1
    rig["card_summary"] = " + ".join(f"{n}x {k}"
                                     for k, n in sorted(counts.items()))

    for c in cards:
        key = keys.get(str(c.get("uuid")), "?")
        for field_name, unit, label in (
            ("membw_read_gbs", "GB/s", "memory read"),
            ("membw_copy_gbs", "GB/s", "memory copy"),
            ("membw_gemv_gbs", "GB/s", "GEMV rate"),
            ("gemm_bf16_tflops", "TFLOP/s", "bf16 GEMM"),
            ("gemm_fp8_tflops", "TFLOP/s", "fp8 GEMM"),
            ("h2d_gbs", "GB/s", "host to device"),
            ("d2h_gbs", "GB/s", "device to host"),
        ):
            if c.get(field_name) is None:
                continue
            measurements.append(rig_artifact.Measurement(
                id=f"card/{key}/{field_name}",
                label=f"{key} {label}",
                source=SOURCE_NAME, unit=unit, value=c.get(field_name),
                taken_at=taken, status="ok",
                context={"device": key,
                         "vram_mib": c.get("total_mib")},
            ))
        if c.get("gemm_fp8_tflops") is None and c.get("fp8_note"):
            capabilities.append(rig_artifact.Capability(
                name=f"card/{key}/fp8_gemm", value=None, provenance="absent",
                note=str(c.get("fp8_note"))[:160]))
        else:
            capabilities.append(rig_artifact.Capability(
                name=f"card/{key}/fp8_gemm",
                value=c.get("gemm_fp8_tflops"), provenance="measured",
                note="torch._scaled_mm path"))
        if c.get("throttled"):
            notes.append(
                f"{key} was below its clock ceiling when probed "
                f"(ratio {c.get('clock_ratio')}); its rates are a floor")

    for p in probe.get("pairs") or []:
        src = keys.get(str(p.get("src_uuid")), "?")
        dst = keys.get(str(p.get("dst_uuid")), "?")
        ctx = {"pair": f"{src}->{dst}", "direction": "ordered",
               "transport": p.get("transport"),
               "peer_access": p.get("peer_access")}
        if p.get("bandwidth_gbs") is not None:
            measurements.append(rig_artifact.Measurement(
                id=f"pair/{src}->{dst}/bandwidth",
                label=f"{src} -> {dst} bandwidth",
                source=SOURCE_NAME, unit="GB/s",
                value=p.get("bandwidth_gbs"), taken_at=taken, status="ok",
                context=ctx))
        if p.get("latency_us") is not None:
            measurements.append(rig_artifact.Measurement(
                id=f"pair/{src}->{dst}/latency",
                label=f"{src} -> {dst} latency",
                source=SOURCE_NAME, unit="us",
                value=p.get("latency_us"), taken_at=taken, status="ok",
                context=ctx))

    transports = {p.get("transport") for p in (probe.get("pairs") or [])}
    capabilities.append(rig_artifact.Capability(
        name="interconnect/p2p",
        value=any("p2p" in str(t) for t in transports),
        provenance="measured",
        note="; ".join(sorted(str(t) for t in transports if t))[:160]))
    rig["interconnect"] = ", ".join(sorted(str(t) for t in transports if t))
    for cav in probe.get("caveats") or []:
        notes.append(str(cav))
    return rig, measurements, capabilities, notes


def _split_probe_sections(rows: List[dict]) -> Tuple[
        List[rig_artifact.Measurement], List[rig_artifact.Capability],
        List[rig_artifact.ErrorSignature], List[str]]:
    """Serving figures from the split-probe store: throughput per topology.

    The row's own output verdict rides along in the context. Throughput on
    this class of rig follows the output CONTENT, so a decode figure whose
    text was degenerate is not comparable with one that was prose -- sharing
    the number without that judgement would export a known measurement trap.
    """
    measurements: List[rig_artifact.Measurement] = []
    capabilities: List[rig_artifact.Capability] = []
    errors: List[rig_artifact.ErrorSignature] = []
    notes: List[str] = []
    for r in rows:
        family = model_family(r.get("model_path", ""))
        cand = r.get("candidate") or "auto"
        taken = _date(r.get("timestamp"))
        if r.get("unbootable"):
            # A candidate that cannot boot is a finding, not a hole.
            errors.append(rig_artifact.error_signature(
                str(r["unbootable"]),
                where=f"split_probe/{family}/{cand}"))
            capabilities.append(rig_artifact.Capability(
                name=f"topology/{family}/{cand}", value=False,
                provenance="measured",
                note=str(r["unbootable"])[:160]))
            continue
        ctx = {
            "model_family": family,
            "tp_size": r.get("tp_size"),
            "candidate": cand,
            "vector": r.get("chosen_vector"),
            "output_verdict": (r.get("output_verdict") or "")[:80],
        }
        for field_name, unit, label in (
            ("decode_tok_s", "tok/s", "decode"),
            ("prefill_tok_s", "tok/s", "prefill"),
            ("ms_per_verify", "ms", "per verify round"),
            ("accept_length", "tokens", "accept length"),
            ("max_total_num_tokens", "tokens", "KV pool"),
        ):
            if r.get(field_name) is None:
                continue
            measurements.append(rig_artifact.Measurement(
                id=f"serving/{family}/{cand}/{field_name}",
                label=f"{family} {cand} {label}",
                source=SOURCE_NAME, unit=unit, value=r.get(field_name),
                taken_at=taken, status="ok", context=ctx,
                note="ms/round is the comparable axis; tok/s follows the "
                     "output content" if field_name == "decode_tok_s" else ""))
        capabilities.append(rig_artifact.Capability(
            name=f"topology/{family}/{cand}", value=True,
            provenance="measured",
            note=f"vector {r.get('chosen_vector')}, "
                 f"tp={r.get('tp_size')}"))
        # Per-rank compute/wait: the split that says WHICH rank is the clock.
        for rank in r.get("rank_compute_wait") or []:
            if not isinstance(rank, dict) or rank.get("wait_ms") is None:
                continue
            measurements.append(rig_artifact.Measurement(
                id=f"serving/{family}/{cand}/rank{rank.get('rank')}/wait_ms",
                label=f"{family} {cand} rank {rank.get('rank')} wait",
                source=SOURCE_NAME, unit="ms", value=rank.get("wait_ms"),
                taken_at=taken, status="ok",
                context=dict(ctx, rank=rank.get("rank"),
                             compute_ms=rank.get("compute_ms")),
                note="the rank with the smallest wait carries the largest "
                     "shard"))
    if rows:
        notes.append(
            "Serving figures come from the split-probe store: one boot per "
            "row, this rig's own topology. They are a floor for the hardware, "
            "not a verdict on the feature.")
    return measurements, capabilities, errors, notes


def to_sections() -> rig_artifact.SourceSections:
    """Everything this rig already knows, as shareable rows. Reads only."""
    rig: Dict[str, Any] = {}
    measurements: List[rig_artifact.Measurement] = []
    capabilities: List[rig_artifact.Capability] = []
    errors: List[rig_artifact.ErrorSignature] = []
    notes: List[str] = []

    probe = _latest_card_probe()
    if probe:
        rig, m, c, n = _card_probe_sections(probe)
        measurements += m
        capabilities += c
        notes += n
    else:
        notes.append(
            "no card probe on disk: the per-card rates and the ordered pair "
            "matrix are absent until the short probe is run once")

    rows = _split_probe_rows()
    if rows:
        m, c, e, n = _split_probe_sections(rows)
        measurements += m
        capabilities += c
        errors += e
        notes += n

    try:
        import torch

        rig.setdefault("torch", torch.__version__)
        rig.setdefault("cuda", torch.version.cuda)
        try:
            rig.setdefault("nccl", ".".join(
                str(x) for x in torch.cuda.nccl.version()))
        except Exception:
            pass
    except Exception:
        pass
    rig.setdefault("cpu_count", os.cpu_count())

    return rig_artifact.SourceSections(
        source=SOURCE_NAME, rig=rig, measurements=measurements,
        capabilities=capabilities, errors=errors, notes=notes)
