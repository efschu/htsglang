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
"""GitHub issue-text generators (design §4 + §5B.2), stage S2 — no backend.

Two zero-backend generators that turn the S1 planner output (+ optionally a
parsed boot log) into copy-paste markdown AND a prefilled GitHub-issue URL,
so crowdsourcing "what runs how on what hardware" needs no database:

  * RESULTS  (``results_issue``)  — the running/planned config block +
    anonymous hardware fingerprint (card model + count), with OPTIONAL
    measured benchmark/energy fields (§5B.2).
  * BUG      (``bug_issue``)      — config + hardware + a scrubbed error
    excerpt; the "planner expected vs actual" diff (§4.3).

Everything is opt-in: these functions only RETURN text/URL — nothing is ever
sent anywhere. All collected free text is scrubbed (``scrub.py``, §4.5):
paths -> basename, secrets/UUIDs/hostnames/IPs dropped; the rig is a
hardware CLASS (``1x RTX 5090, 2x RTX 3080``), never a machine identity.

HONESTY (structural, mirrors §3.4 / §5A.5): the benchmark and energy fields
are MEASURED-ONLY. The energy module (S2.5) does not exist yet, so those
fields are simply ABSENT until a caller passes measured values — there is no
code path that fabricates a tok/s or a Joule number. ``BenchmarkFields`` /
``EnergyFields`` carry only measured inputs and render nothing when omitted.
"""

from __future__ import annotations

import dataclasses
import subprocess
import urllib.parse
from collections import Counter
from typing import List, Optional, Sequence

from sglang.srt.planner.hardware import HardwareSpec
from sglang.srt.planner.scrub import (
    scrub_launch_flags,
    scrub_log_excerpt,
    scrub_path,
    scrub_text,
)

__all__ = [
    "HardwareFingerprint",
    "Versions",
    "BenchmarkFields",
    "EnergyFields",
    "IssueText",
    "hardware_fingerprint_from_spec",
    "hardware_fingerprint_from_nvml",
    "collect_versions",
    "results_issue",
    "bug_issue",
    "results_from_boot_log",
    "results_from_plan",
    "bug_from_plan",
]

#: Prefilled-URL byte ceiling (design §4.4). GitHub/browsers cap the URL
#: near 8 KB; below ~6 KB we offer the one-click URL, above it the caller
#: falls back to copy-paste (log excerpts blow the budget fast).
URL_BODY_BUDGET = 6 * 1024

DEFAULT_OWNER_REPO = "efschu/htsglang"


# ===========================================================================
# Anonymous hardware fingerprint (design §4.1) — card model + count only.
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class HardwareFingerprint:
    """A submission's hardware identity: card MODEL + COUNT, host RAM, driver
    — deliberately no UUIDs, no hostname, no machine identity (design §4.5).
    Doubles as a hardware profile seed for the future library (§2.7)."""

    #: [(count, name, total_mib), ...] grouped by (name, total_mib).
    cards: Sequence[tuple]
    host_ram_mib: Optional[int] = None
    driver: Optional[str] = None
    nvlink: Optional[bool] = None
    #: e.g. "PCIe 4.0 x16/x4" — coarse, non-identifying.
    interconnect: Optional[str] = None

    @property
    def gpu_count(self) -> int:
        return sum(c[0] for c in self.cards)

    def summary(self) -> str:
        """e.g. ``1x RTX 5090 32GB, 2x RTX 3080 20GB``."""
        parts = []
        for count, name, total_mib in self.cards:
            gb = round(total_mib / 1024)
            parts.append(f"{count}x {scrub_text(name)} {gb}GB")
        return ", ".join(parts)


def _group_cards(items) -> List[tuple]:
    """Group (name, total_mib) pairs into [(count, name, total_mib), ...],
    preserving first-seen order."""
    counter = Counter(items)
    seen = []
    for key in items:
        if key not in seen:
            seen.append(key)
    return [(counter[k], k[0], k[1]) for k in seen]


def hardware_fingerprint_from_spec(hardware: HardwareSpec) -> HardwareFingerprint:
    """Fingerprint from an S1 ``HardwareSpec`` (any source, incl. manual)."""
    items = [(g.name, g.total_mib) for g in hardware.gpus]
    # PCIe hint from the first card that reports it (coarse, non-identifying).
    interconnect = None
    for g in hardware.gpus:
        if g.pcie_gen:
            width = f" x{g.pcie_width}" if g.pcie_width else ""
            interconnect = f"PCIe {g.pcie_gen}.0{width}"
            break
    return HardwareFingerprint(
        cards=_group_cards(items),
        host_ram_mib=hardware.host_ram_mib,
        driver=scrub_text(hardware.driver) if hardware.driver else None,
        interconnect=interconnect,
    )


def hardware_fingerprint_from_nvml() -> HardwareFingerprint:
    """Fingerprint from a live NVML/nvidia-smi sample (reuses the S1 hardware
    source, which itself reuses the rig-dashboard ``sample_nvml``)."""
    from sglang.srt.planner.hardware import hardware_from_nvml

    return hardware_fingerprint_from_spec(hardware_from_nvml())


# ===========================================================================
# Versions (design §4.1) — best-effort, scrubbed.
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class Versions:
    torch: Optional[str] = None
    cuda: Optional[str] = None
    driver: Optional[str] = None
    htsglang_commit: Optional[str] = None

    def line(self) -> str:
        parts = []
        if self.driver:
            parts.append(f"driver {self.driver}")
        if self.cuda:
            parts.append(f"CUDA {self.cuda}")
        if self.torch:
            parts.append(f"torch {self.torch}")
        if self.htsglang_commit:
            parts.append(f"htsglang {self.htsglang_commit}")
        return " · ".join(parts) if parts else "(versions unavailable)"


def collect_versions(repo_dir: Optional[str] = None) -> Versions:
    """Collect torch / CUDA (build) / htsglang commit — CPU-only, no GPU
    query. Driver is left to the hardware fingerprint (NVML) path."""
    torch_v = cuda_v = None
    try:
        import torch

        torch_v = scrub_text(str(torch.__version__))
        cuda_v = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    if repo_dir is None:
        # Default to the sglang source tree so the commit resolves even when
        # the CLI is invoked from an unrelated cwd.
        import os

        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    commit = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = None
    return Versions(
        torch=torch_v, cuda=cuda_v, htsglang_commit=commit
    )


# ===========================================================================
# OPTIONAL measured fields (design §5B.2) — measured-only, absent by default.
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class BenchmarkFields:
    """MEASURED throughput of a real run (design §5B.2). There is no planner
    path that constructs this — it exists only to carry numbers a caller
    measured (from ``/metrics`` + a phase-tagged batch stream). Omitted =>
    the RESULTS issue simply has no benchmark section (never an estimate)."""

    prefill_tok_s: Optional[float] = None
    decode_tok_s: Optional[float] = None
    batch: Optional[int] = None
    concurrency: Optional[int] = None

    def has_data(self) -> bool:
        return self.prefill_tok_s is not None or self.decode_tok_s is not None


@dataclasses.dataclass(frozen=True)
class EnergyFields:
    """MEASURED energy of a real run (design §5A/§5B.2). The energy module
    (S2.5) is NOT built yet, so in S2 this is always supplied externally or
    absent — never synthesized. Per-token figures are per-batch-size-bucket
    in the real module; here they are opaque measured strings so this
    generator makes no numeric claim of its own."""

    j_per_prefill_token: Optional[str] = None
    j_per_decode_token: Optional[str] = None
    per_card_efficiency: Optional[str] = None
    kwh_saved: Optional[str] = None
    prefill_hours_saved: Optional[str] = None
    conditions: Optional[str] = None

    def has_data(self) -> bool:
        return any(
            v is not None
            for v in (
                self.j_per_prefill_token,
                self.j_per_decode_token,
                self.kwh_saved,
            )
        )


# ===========================================================================
# The rendered issue (design §4.4): markdown + prefilled URL.
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class IssueText:
    title: str
    markdown: str
    url: str
    #: True when the encoded body fits ``URL_BODY_BUDGET`` -> offer the
    #: one-click URL; else the caller uses the copy-paste markdown (which is
    #: always produced regardless).
    url_within_budget: bool


def _issue_url(title: str, body: str, owner_repo: str) -> tuple:
    enc_body = urllib.parse.quote(body)
    within = len(enc_body) <= URL_BODY_BUDGET
    url = (
        f"https://github.com/{owner_repo}/issues/new?"
        + urllib.parse.urlencode({"title": title, "body": body})
    )
    return url, within


# ===========================================================================
# Shared header block (hardware / versions / model / config).
# ===========================================================================


def _model_line(model_name: str, quant: Optional[str], group_size=None) -> str:
    line = scrub_path(model_name)
    if quant:
        line += f" · quant {scrub_text(str(quant))}"
        if group_size:
            line += f" g{group_size}"
    return line


def _hardware_block(fp: HardwareFingerprint) -> List[str]:
    lines = [f"- GPUs: {fp.summary()}"]
    extras = []
    if fp.nvlink is not None:
        extras.append("NVLink" if fp.nvlink else "no NVLink")
    if fp.interconnect:
        extras.append(fp.interconnect)
    if extras:
        lines[0] += "; " + "; ".join(extras)
    if fp.host_ram_mib:
        lines.append(f"- Host RAM: {round(fp.host_ram_mib / 1024)} GB")
    return lines


# ===========================================================================
# RESULTS generator (design §4.2 + §5B.2).
# ===========================================================================


def results_issue(
    *,
    model_name: str,
    hardware: HardwareFingerprint,
    launch_flags: Sequence[str],
    fits: bool,
    stock_runs: Optional[bool] = None,
    stock_reason: Optional[str] = None,
    capacity_pct_range=None,
    max_context_tokens: Optional[float] = None,
    max_total_num_tokens: Optional[int] = None,
    per_rank_gpu_gib: Optional[Sequence[float]] = None,
    quant: Optional[str] = None,
    group_size=None,
    env_flags: Optional[Sequence[str]] = None,
    versions: Optional[Versions] = None,
    measured: bool = False,
    benchmark: Optional[BenchmarkFields] = None,
    energy: Optional[EnergyFields] = None,
    owner_repo: str = DEFAULT_OWNER_REPO,
) -> IssueText:
    """Render a RESULTS submission (design §4.2).

    ``measured=False`` (default) labels the Result numbers "planned/estimate"
    (they came from the S1 planner); ``measured=True`` labels them measured
    (they came from a parsed boot log, ``results_from_boot_log``). The
    ``benchmark`` / ``energy`` blocks render ONLY when supplied with measured
    data — otherwise absent (no fabricated numbers; §5B.2 honesty).
    """
    versions = versions or collect_versions()
    driver = hardware.driver
    ver_line = versions.line()
    if driver and "driver" not in ver_line:
        ver_line = f"driver {driver} · " + ver_line

    verdict = "✅ runs" if fits else "❌ does not fit (planner)"
    stock_txt = ""
    if stock_runs is not None:
        stock_txt = (
            "  (stock even-TP: "
            + (
                "✅ runs"
                if stock_runs
                else f"❌ {scrub_text(stock_reason or 'cannot run')}"
            )
            + ")"
        )

    label = "measured" if measured else "planner estimate"
    md: List[str] = []
    md.append(
        f"## htsglang result — {scrub_path(model_name)} on {hardware.summary()}"
    )
    md.append("")
    md.append(f"**Verdict:** {verdict}{stock_txt}")
    md.append("")
    md.append("### Hardware")
    md += _hardware_block(hardware)
    md.append(f"- {ver_line}")
    md.append("")
    md.append("### Model")
    md.append(f"- {_model_line(model_name, quant, group_size)}")
    md.append("")
    md.append("### Config that ran")
    md.append("```")
    md += scrub_launch_flags(list(launch_flags))
    if env_flags:
        md += scrub_launch_flags(list(env_flags))
    md.append("```")
    md.append("")
    md.append("### Result")
    if per_rank_gpu_gib:
        md.append(
            "- Per-rank GPU: "
            + " / ".join(f"{g:.1f}" for g in per_rank_gpu_gib)
            + " GB"
        )
    if max_total_num_tokens is not None:
        md.append(f"- max_total_num_tokens: {max_total_num_tokens} ({label})")
    if max_context_tokens is not None:
        md.append(
            f"- max context: ~{int(max_context_tokens)} tokens ({label})"
        )
    if capacity_pct_range is not None:
        lo, hi = capacity_pct_range
        md.append(
            f"- Capacity vs stock even-TP: **{lo:+d}% .. {hi:+d}%** "
            f"KV/context ({label})"
        )

    # --- OPTIONAL measured benchmark (design §5B.2) --------------------------
    if benchmark is not None and benchmark.has_data():
        md.append("")
        md.append("### Benchmark (measured, opt-in)")
        bp = []
        if benchmark.prefill_tok_s is not None:
            bp.append(f"Prefill: {benchmark.prefill_tok_s:g} tok/s")
        if benchmark.decode_tok_s is not None:
            bp.append(f"Decode: {benchmark.decode_tok_s:g} tok/s")
        cond = []
        if benchmark.batch is not None:
            cond.append(f"batch {benchmark.batch}")
        if benchmark.concurrency is not None:
            cond.append(f"concurrency {benchmark.concurrency}")
        line = "- " + "  ·  ".join(bp)
        if cond:
            line += f"      ({', '.join(cond)})"
        md.append(line)
    # --- OPTIONAL measured energy (design §5A/§5B.2) — S2.5 supplies this ---
    if energy is not None and energy.has_data():
        md.append("")
        md.append("### Energy (measured, opt-in)")
        if energy.j_per_prefill_token or energy.j_per_decode_token:
            md.append(
                "- "
                + " · ".join(
                    x
                    for x in (
                        f"{energy.j_per_prefill_token} J/prefill-token"
                        if energy.j_per_prefill_token
                        else None,
                        f"{energy.j_per_decode_token} J/decode-token"
                        if energy.j_per_decode_token
                        else None,
                    )
                    if x
                )
            )
        if energy.per_card_efficiency:
            md.append(f"- Per-card efficiency: {scrub_text(energy.per_card_efficiency)}")
        if energy.kwh_saved or energy.prefill_hours_saved:
            saved = []
            if energy.kwh_saved:
                saved.append(f"~{energy.kwh_saved} kWh")
            if energy.prefill_hours_saved:
                saved.append(f"~{energy.prefill_hours_saved} h prefill compute")
            md.append("- Cumulative saved by caching: " + " · ".join(saved))
        if energy.conditions:
            md.append(f"- Measurement conditions: {scrub_text(energy.conditions)}")

    body = "\n".join(md)
    title = f"htsglang result: {scrub_path(model_name)} on {hardware.summary()}"
    url, within = _issue_url(title, body, owner_repo)
    return IssueText(
        title=title, markdown=body, url=url, url_within_budget=within
    )


# ===========================================================================
# BUG generator (design §4.3).
# ===========================================================================


def bug_issue(
    *,
    model_name: str,
    hardware: HardwareFingerprint,
    launch_flags: Sequence[str],
    symptom: str,
    planner_expected: Optional[str] = None,
    planner_max_context: Optional[float] = None,
    log_text: Optional[str] = None,
    quant: Optional[str] = None,
    group_size=None,
    env_flags: Optional[Sequence[str]] = None,
    versions: Optional[Versions] = None,
    owner_repo: str = DEFAULT_OWNER_REPO,
) -> IssueText:
    """Render a BUG report (design §4.3). The "planner expected vs actual"
    line turns a bug report into a divergence report (feeds §5 drift
    monitoring). The log excerpt is windowed around the error and scrubbed."""
    versions = versions or collect_versions()
    ver_line = versions.line()
    if hardware.driver and "driver" not in ver_line:
        ver_line = f"driver {hardware.driver} · " + ver_line

    md: List[str] = []
    md.append(f"## htsglang bug — {scrub_path(model_name)} on {hardware.summary()}")
    md.append("")
    md.append("### Hardware")
    md += _hardware_block(hardware)
    md.append(f"- {ver_line}")
    md.append("")
    md.append("### Model")
    md.append(f"- {_model_line(model_name, quant, group_size)}")
    md.append("")
    md.append("### Expected (planner)")
    if planner_expected:
        md.append(f"- {scrub_text(planner_expected)}")
    if planner_max_context is not None:
        md.append(f"- expected max context ~{int(planner_max_context)} tokens")
    if not planner_expected and planner_max_context is None:
        md.append("- (planner expectation not recorded)")
    md.append("")
    md.append("### Actual")
    md.append(f"- {scrub_text(symptom)}")
    md.append("")
    md.append("### Launch command")
    md.append("```")
    md += scrub_launch_flags(list(launch_flags))
    if env_flags:
        md += scrub_launch_flags(list(env_flags))
    md.append("```")
    if log_text:
        md.append("")
        md.append("### Log excerpt (scrubbed, around the error)")
        md.append("```")
        md.append(scrub_log_excerpt(log_text))
        md.append("```")
    md.append("")
    md.append("### Planner expectation vs outcome")
    exp = scrub_text(planner_expected) if planner_expected else "(not recorded)"
    md.append(f"- planner: {exp}   → actual: {scrub_text(symptom)}")

    body = "\n".join(md)
    title = f"htsglang bug: {scrub_path(model_name)} on {hardware.summary()}"
    url, within = _issue_url(title, body, owner_repo)
    return IssueText(
        title=title, markdown=body, url=url, url_within_budget=within
    )


# ===========================================================================
# Bridges from an S1 PlanResult (the PLANNED / estimate path).
# ===========================================================================


def _split_flags_envs(launch_flags):
    """Separate ``--flag ...`` tokens from ``ENV=value`` tokens (the S1
    launch_flags list mixes both)."""
    flags, envs = [], []
    for tok in launch_flags:
        (envs if ("=" in tok and not tok.startswith("--")) else flags).append(tok)
    return flags, envs


def _plan_common(result):
    """Pull the shared header fields out of an S1 PlanResult."""
    inputs = result.inputs
    fp = hardware_fingerprint_from_spec(result.hardware)
    flags, env_flags = _split_flags_envs(result.launch_flags)
    per_rank_gib = None
    if result.capacity is not None:
        per_rank_gib = [
            rc.weight_gib + rc.mamba_gib for rc in result.capacity.per_rank
        ]
    return inputs, fp, flags, env_flags, per_rank_gib


def results_from_plan(
    result,
    *,
    quant: Optional[str] = None,
    group_size=None,
    versions: Optional[Versions] = None,
    benchmark: Optional[BenchmarkFields] = None,
    energy: Optional[EnergyFields] = None,
    owner_repo: str = DEFAULT_OWNER_REPO,
) -> IssueText:
    """RESULTS issue from an S1 ``PlanResult`` — the PLANNED path (numbers
    labelled "planner estimate"). ``benchmark``/``energy`` stay absent unless
    a caller supplies measured values (§5B.2 honesty)."""
    inputs, fp, flags, env_flags, per_rank_gib = _plan_common(result)
    adv = result.advantage
    return results_issue(
        model_name=inputs.model_path,
        hardware=fp,
        launch_flags=flags,
        env_flags=env_flags,
        fits=result.fits,
        stock_runs=(adv.stock.runs if adv is not None else None),
        stock_reason=(
            "; ".join(adv.stock.reasons)
            if adv is not None and adv.stock.reasons
            else None
        ),
        capacity_pct_range=(
            adv.capacity_pct_range if adv is not None else None
        ),
        max_context_tokens=(
            result.capacity.max_context_tokens
            if result.capacity is not None
            else None
        ),
        per_rank_gpu_gib=per_rank_gib,
        quant=quant,
        group_size=group_size,
        versions=versions,
        measured=False,
        benchmark=benchmark,
        energy=energy,
        owner_repo=owner_repo,
    )


def bug_from_plan(
    result,
    *,
    symptom: str,
    log_text: Optional[str] = None,
    quant: Optional[str] = None,
    group_size=None,
    versions: Optional[Versions] = None,
    owner_repo: str = DEFAULT_OWNER_REPO,
) -> IssueText:
    """BUG report from an S1 ``PlanResult`` + an observed ``symptom`` (and an
    optional raw log). The planner's own verdict becomes the "expected" side
    of the divergence line (design §4.3)."""
    inputs, fp, flags, env_flags, _ = _plan_common(result)
    if result.fits and result.capacity is not None:
        expected = "planner said: fits"
        max_ctx = result.capacity.max_context_tokens
    else:
        expected = "planner said: does NOT fit — " + " ; ".join(
            result.infeasible_reasons
        )
        max_ctx = None
    return bug_issue(
        model_name=inputs.model_path,
        hardware=fp,
        launch_flags=flags,
        env_flags=env_flags,
        symptom=symptom,
        planner_expected=expected,
        planner_max_context=max_ctx,
        log_text=log_text,
        quant=quant,
        group_size=group_size,
        versions=versions,
        owner_repo=owner_repo,
    )


# ===========================================================================
# Convenience: RESULTS straight from a boot log (the MEASURED path).
# ===========================================================================


def results_from_boot_log(
    boot_log_path: str,
    hardware: Optional[HardwareFingerprint] = None,
    owner_repo: str = DEFAULT_OWNER_REPO,
    **overrides,
) -> IssueText:
    """Build a MEASURED RESULTS issue from a real sglang boot log, reusing
    the rig-dashboard ``parse_plan_file`` (design §4.2 "measured" path).

    ``hardware`` is optional — when omitted, the fingerprint is reconstructed
    from the boot log's own per-rank GPU names (card model + count), so the
    submission stays anonymous and self-contained.
    """
    import importlib.util
    import os

    here = os.path.abspath(__file__)
    repo = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    )
    pp_path = os.path.join(repo, "tools", "rig_dashboard", "plan_parser.py")
    spec = importlib.util.spec_from_file_location("_rig_plan_parser", pp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = mod.parse_plan_file(boot_log_path)

    # Reconstruct launch flags from the parsed plan.
    flags: List[str] = []
    if plan.get("tp_size"):
        flags.append(f"--tp-size {plan['tp_size']}")
    if plan.get("rank_gpu_id"):
        flags.append("--rank-gpu-id " + ",".join(map(str, plan["rank_gpu_id"])))
    if plan.get("memory_budgets_mib"):
        flags.append(
            "--rank-gpu-memory-mib "
            + ",".join(map(str, plan["memory_budgets_mib"]))
        )
    if plan.get("rank_tp_ratio"):
        flags.append("--rank-tp-ratio " + ",".join(map(str, plan["rank_tp_ratio"])))
    if plan.get("dcp_size"):
        flags.append(f"--dcp-size {plan['dcp_size']}")

    # Fingerprint from the log's own GPU inventory when none was supplied.
    if hardware is None:
        gpus = plan.get("gpus") or []
        if gpus:
            # No per-card total in the log -> group by name, unknown VRAM.
            items = [(g["name"], 0) for g in gpus]
            hardware = HardwareFingerprint(cards=_group_cards(items))
        else:
            hardware = HardwareFingerprint(cards=[])

    ranks = plan.get("ranks") or {}
    per_rank_gib = None
    if ranks:
        per_rank_gib = [
            (ranks[r].get("weight_gb") or 0.0)
            + (ranks[r].get("kv_gb") or 0.0)
            + (ranks[r].get("mamba_gb") or 0.0)
            for r in sorted(ranks)
        ]

    kwargs = dict(
        model_name=plan.get("model", "unknown-model"),
        hardware=hardware,
        launch_flags=flags,
        fits=True,  # it booted
        quant=plan.get("quant"),
        max_total_num_tokens=plan.get("max_total_num_tokens"),
        per_rank_gpu_gib=per_rank_gib,
        measured=True,
        owner_repo=owner_repo,
    )
    kwargs.update(overrides)
    return results_issue(**kwargs)
