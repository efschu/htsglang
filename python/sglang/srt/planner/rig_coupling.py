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
"""Coupling a second rig: the gate, the transport per message class, the pool.

Task #214, the desk half. :mod:`sglang.srt.rigmon.pairing` already sequences a
pairing -- reach the far rig, hold the identities against each other, rank the
transports for a pair, emit a configuration. This module is the layer that
pairing deliberately does not have: it decides **whether a coupling is worth
attempting at all**, on the grounds this project has actually measured, and it
says which of the remaining steps this process may take and which only the
host can.

Three properties, and each one exists because its absence produced a bug here
before:

**Nothing in this module touches the network.** Every function takes facts and
returns a report. A coupling recommendation that quietly opened a socket would
make "run the plan" and "contact the far rig" the same act, and the dashboard
runs in a container that cannot reach the fast line at all (runbook §1.1): a
figure it collected itself would describe the 1 GbE LAN while claiming to
describe a 40G RoCE link. Steps that only the host can take come back as
:class:`HostStep` -- a copyable command in the runbook §8 shape, with
``${VAR:-<placeholder>}`` throughout, never a hidden call.

**A refusal names its evidence.** Every :class:`GateRow` carries where its
verdict comes from: a row of the machine-readable rejected register
(:mod:`sglang.srt.planner.rejected`), a runbook section, or a measurement id.
A gate that only says "no" is a gate nobody can act on, and one that says "no"
from an assumption is worse than no gate.

**A number that was never measured says so.** The transport choice per message
class carries the ``measured`` / ``estimate`` / ``absent`` vocabulary the rest
of the dashboard uses. Having measured *this* rig says nothing about the wire
to the other one, so an absent row carries the command that would measure it
rather than a stand-in figure.

DUAL-GROUP SHAPE
----------------
A coupling result describes a **pool of cards**, not one compound machine. Two
rigs coupled today may be run as one lane spanning both, as two independent
lanes, or as a lane plus a spare -- and the dual-group runtime makes that a
runtime decision rather than a coupling-time one. So the report carries
``pool.cards`` and ``pool.lane_candidates`` (each lane a LIST of cards), and
never a single implied verbund.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sglang.srt.planner import rejected as rejectedmod

__all__ = [
    "OK",
    "WARN",
    "BLOCK",
    "MEASURED",
    "ESTIMATE",
    "ABSENT",
    "HERE",
    "HOST",
    "FAR",
    "CardFacts",
    "RigFacts",
    "GateRow",
    "TransportChoice",
    "HostStep",
    "LaneCandidate",
    "CouplingReport",
    "MESSAGE_CLASSES",
    "COLOCATION_NCCL_MIN",
    "DMABUF_RDMA_CHECKS",
    "arch_of_card_name",
    "gate",
    "transport_plan",
    "card_pool",
    "host_steps",
    "couple",
]

#: The verdict vocabulary of :mod:`sglang.srt.rigmon.compat`, repeated here as
#: literals rather than imported: this module is pure planner-side and must
#: stay importable on a machine where rigmon's probes cannot run. The strings
#: are the contract; a test pins them against compat so the two cannot drift.
OK = "ok"
WARN = "warn"
BLOCK = "block"

#: The provenance vocabulary every dashboard surface already uses (#218).
MEASURED = "measured"
ESTIMATE = "estimate"
ABSENT = "absent"

#: Where a step can run. The distinction is not cosmetic: the development
#: container has no interface on the cross-rig subnet and no /dev/infiniband
#: (runbook §1.1), so a step marked HOST cannot be made to work from here by
#: trying harder.
HERE = "dashboard"
HOST = "pve-host"
FAR = "far-rig"

#: Several ranks on one physical GPU need an NCCL at or above this. The rig's
#: venvs pin 2.28.9 (docker/htsglang-constraints.txt), which is BELOW it --
#: co-location is refused there and exists only in the Docker image that
#: carries 2.30.7 (runbook §6.2). Encoded as the threshold rather than as the
#: installed version so a rig with a newer NCCL passes without a code change.
COLOCATION_NCCL_MIN: Tuple[int, int] = (2, 30)

#: The four message classes a cross-rig run actually has, and which control
#: selects each (runbook §4.3.1). ``separable_from`` records the honest part:
#: (a) and (b) ride ONE UcpWorker context per rank, so naming them separately
#: describes the classes, not two independently routable lines.
MESSAGE_CLASSES: Tuple[Dict[str, Any], ...] = (
    {
        "key": "tp_small",
        "label": "TP collectives, small (decode / verify all-reduce, gather)",
        "carrier": "HTCCL UCX collective context",
        "flag": "--collective-net-small",
        "env": "SGLANG_COLLECTIVE_NET_SMALL",
        "wants": "latency",
        "separable_from": None,
    },
    {
        "key": "tp_bulk",
        "label": "TP collectives, large (prefill chunks)",
        "carrier": "HTCCL UCX collective context (the same one)",
        "flag": "--collective-net-small",
        "env": "SGLANG_COLLECTIVE_NET_SMALL",
        "wants": "bandwidth",
        "separable_from": "tp_small",
    },
    {
        "key": "kv_bulk",
        "label": "PD-KV / HiCache bulk",
        "carrier": "mooncake / nixl transfer engine",
        "flag": "--collective-net-bulk",
        "env": "SGLANG_COLLECTIVE_NET_BULK",
        "wants": "bandwidth",
        "separable_from": None,
    },
    {
        "key": "control",
        "label": "Rendezvous / control plane",
        "carrier": "gloo process group",
        "flag": "--dist-init-addr",
        "env": "GLOO_SOCKET_IFNAME",
        "wants": "reachability",
        "separable_from": None,
    },
)

#: The four preconditions the #214 desk evaluation asks for before a
#: `dmabuf_rdma` HTCCL transport is even worth prototyping
#: (``docs/EVAL_gdr_uebernahme.md`` §6.2, §9 P1). Each is read from
#: :attr:`RigFacts.capabilities` -- never assumed -- because the host-side
#: probe that answers most of these (``read_dmabuf_flag.sh`` needs `gdb` on
#: `/proc/kcore` as root) cannot run inside this process, the same reason
#: every other capability-backed row in this module reports ABSENT rather
#: than inferring. Deliberately NOT included here: "GPU works as RDMA
#: source" and "target BAR resizable". Both are real preconditions in the
#: evaluation's own table, but the first is exactly the row the evaluation
#: warns must not be inferred from PCIe topology (§1.2), and the second is
#: about which physical card is targeted, not about whether the chain is
#: installed -- neither belongs in a fixed four-item capability list.
DMABUF_RDMA_CHECKS: Tuple[Tuple[str, str], ...] = (
    ("dmabuf_open_kernel_module", "NVIDIA open kernel modules"),
    ("dmabuf_rdma_core", "rdma-core with ibv_reg_dmabuf_mr (>= rdma-core 34)"),
    ("dmabuf_mlx5_path", "mlx5_ib dmabuf path (kernel >= 5.12)"),
    ("dmabuf_vmm_export", "VMM export requests POSIX_FD (vmm_utils.py)"),
)

#: Capability values that count as "this precondition holds". A rig reports
#: its own vocabulary (bool, "ok", "open", ...); this is intentionally
#: permissive about the spelling and strict about everything else -- a typo'd
#: value reads as not-yet-confirmed rather than silently passing.
_DMABUF_OK_VALUES = {"ok", "open", "available", "present", "yes", "true"}


def _dmabuf_passed(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _DMABUF_OK_VALUES
    return bool(value) if value is not None else False


# ===========================================================================
# Facts about one rig
# ===========================================================================
#: Card-name fragments that decide an architecture wall. Only the walls this
#: project has actually hit are listed: a table of every GPU ever made would
#: be a maintenance burden whose wrong rows would read as facts. A card that
#: matches nothing keeps ``arch=None``, and every gate row that needs an arch
#: reports ABSENT for it rather than assuming.
_ARCH_PATTERNS: Tuple[Tuple[str, Optional[str], str], ...] = (
    # (regex over the lowercased card name, arch, vendor)
    (r"\b(2080|2070|2060|titan rtx)\b", "sm75", "nvidia"),
    (r"\bt4\b", "sm75", "nvidia"),
    (r"\b(3090|3080|3070|3060|a6000|a5000|a40)\b", "sm86", "nvidia"),
    (r"\b(4090|4080|4070|l40|l4)\b", "sm89", "nvidia"),
    (r"\b(5090|5080|5070)\b", "sm120", "nvidia"),
    (r"\b(h100|h200|h800)\b", "sm90", "nvidia"),
    (r"\b(a100|a30|a800)\b", "sm80", "nvidia"),
    (r"\bvega\b|\bgfx900\b", "gfx900", "amd"),
    (r"\bmi300\b|\bgfx942\b", "gfx942", "amd"),
    (r"\bradeon\b|\binstinct\b", None, "amd"),
)


def arch_of_card_name(name: str) -> Tuple[Optional[str], str]:
    """``(arch, vendor)`` inferred from a card MODEL name.

    Inference, and labelled as such wherever it is used: NVML reports a model
    string, not a compute capability, so "RTX 2080 Ti is Turing" is knowledge
    this table holds rather than something the far rig said. A caller that has
    the real value (a ``arch`` / ``sm_arch`` / ``compute_cap`` field on the
    card) must pass it instead; :meth:`CardFacts.from_json` prefers it.
    """
    low = " " + str(name or "").lower().replace("-", " ") + " "
    for pattern, arch, vendor in _ARCH_PATTERNS:
        if re.search(pattern, low):
            return arch, vendor
    return None, "unknown"


@dataclasses.dataclass
class CardFacts:
    """One card of one rig, in the terms a coupling decision needs.

    No UUID, no PCI address, no host: a coupling report is a thing people
    paste into an issue. ``label`` is positional (``rig-b/1``) and means
    nothing outside this report.
    """

    label: str
    name: str
    vram_mib: Optional[int] = None
    arch: Optional[str] = None
    #: "declared" (the far rig said so) | "inferred" (from the model name) |
    #: "unknown". A wall that fires on an inferred arch says so in its row.
    arch_source: str = "unknown"
    vendor: str = "unknown"
    index: Optional[int] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict, label: str) -> "CardFacts":
        d = dict(d or {})
        name = str(d.get("name") or "?")
        declared = d.get("arch") or d.get("sm_arch") or d.get("compute_cap")
        inferred_arch, inferred_vendor = arch_of_card_name(name)
        arch: Optional[str]
        if declared:
            arch, source = str(declared), "declared"
        else:
            arch, source = inferred_arch, ("inferred" if inferred_arch else "unknown")
        vram = d.get("vram_mib") or d.get("total_mib") or d.get("memory_total_mib")
        return cls(
            label=label,
            name=name,
            vram_mib=int(vram) if vram else None,
            arch=arch,
            arch_source=source,
            vendor=str(d.get("vendor") or inferred_vendor),
            index=d.get("index"),
        )


@dataclasses.dataclass
class RigFacts:
    """What one rig is, as far as a coupling decision is concerned.

    Every field is optional on purpose. A rig that has not run the comm suite
    reports no NCCL version, and the gate row that needs one says ABSENT with
    the command that would produce it -- which is a different statement from
    "this rig's NCCL is too old", and the two must never be collapsed.
    """

    rig_id: str
    label: str = ""
    #: "rig-profile" | "artifact" | "pairing-reach" | "manual"
    source: str = "manual"
    cards: List[CardFacts] = dataclasses.field(default_factory=list)
    nccl: Optional[str] = None
    torch: Optional[str] = None
    cuda: Optional[str] = None
    ucx: Optional[str] = None
    commit: Optional[str] = None
    driver: Optional[str] = None
    #: name -> value, in the shape :class:`rig_artifact.Capability` shares.
    capabilities: Dict[str, Any] = dataclasses.field(default_factory=dict)
    notes: List[str] = dataclasses.field(default_factory=list)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["cards"] = [c.to_json() for c in self.cards]
        return d

    # -- derived ---------------------------------------------------------
    def archs(self) -> List[str]:
        return sorted({c.arch for c in self.cards if c.arch})

    def vendors(self) -> List[str]:
        return sorted({c.vendor for c in self.cards if c.vendor != "unknown"})

    def cards_with_arch(self, arch: str) -> List[CardFacts]:
        return [c for c in self.cards if c.arch == arch]

    def nccl_tuple(self) -> Optional[Tuple[int, ...]]:
        return _version_tuple(self.nccl)

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_rig_profile(
        cls, profile: dict, rig_id: str, *, label: str = "", source: str = "rig-profile"
    ) -> "RigFacts":
        """From :func:`comm_suite.rig_profile` output (or the ``rig`` block of
        a shared artifact -- they are the same shape by construction)."""
        p = dict(profile or {})
        cards = [
            CardFacts.from_json(c, f"{rig_id}/{i}")
            for i, c in enumerate(p.get("cards") or [])
        ]
        return cls(
            rig_id=rig_id,
            label=label or str(p.get("card_summary") or rig_id),
            source=source,
            cards=cards,
            nccl=p.get("nccl"),
            torch=p.get("torch"),
            cuda=p.get("cuda"),
            ucx=p.get("ucx"),
            commit=p.get("commit"),
            driver=p.get("driver"),
        )

    @classmethod
    def from_artifact(cls, digest: dict, rig_id: str, *, label: str = "") -> "RigFacts":
        """From a shared rig artifact (``htsglang-rig-artifact/v1``).

        The artifact is the import path for a rig this process cannot reach:
        the far side runs the comm suite, the digest is copied over, and the
        coupling reasons about it exactly as it would about a live answer --
        with ``source`` recording which it was.
        """
        d = dict(digest or {})
        facts = cls.from_rig_profile(
            d.get("rig") or {}, rig_id, label=label, source="artifact"
        )
        for c in d.get("capabilities") or []:
            name = c.get("name")
            if name:
                facts.capabilities[str(name)] = {
                    "value": c.get("value"),
                    "provenance": c.get("provenance") or ABSENT,
                    "note": c.get("note") or "",
                }
        fp = (d.get("fingerprint") or {}).get("id")
        if fp:
            facts.notes.append(f"artifact fingerprint {fp}")
        return facts

    @classmethod
    def from_pairing_reach(cls, detail: dict, rig_id: str = "rig-b") -> "RigFacts":
        """From the ``reach`` step of a :mod:`rigmon.pairing` session.

        Reads only what the far rig already published to its own aggregator.
        This constructor performs no I/O -- the session detail was fetched by
        the pairing flow, which is the one place allowed to talk to a peer.
        """
        d = dict(detail or {})
        hw = d.get("hw_profile") or {}
        facts = cls.from_rig_profile(
            hw, rig_id, label=str(d.get("url") or rig_id), source="pairing-reach"
        )
        ident = d.get("identity") or {}
        facts.commit = facts.commit or ident.get("commit")
        facts.torch = facts.torch or ident.get("torch")
        facts.cuda = facts.cuda or ident.get("cuda")
        facts.driver = facts.driver or ident.get("driver")
        for c in (d.get("capabilities") or {}).get("capabilities") or []:
            key = c.get("key")
            if key:
                facts.capabilities[str(key)] = {
                    "value": c.get("state"),
                    "provenance": MEASURED,
                    "note": c.get("reason") or "",
                }
        return facts


def _version_tuple(v: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not v:
        return None
    parts = re.findall(r"\d+", str(v))
    if not parts:
        return None
    return tuple(int(p) for p in parts[:3])


# ===========================================================================
# The compatibility gate
# ===========================================================================
@dataclasses.dataclass
class GateRow:
    """One precondition, its verdict, and where the verdict comes from.

    ``evidence`` is the load-bearing field. ``register:gguf_on_sm75`` means a
    settled measurement in :mod:`planner.rejected` decided this; ``runbook
    §6.2`` means a documented hardware fact did; ``absent`` means nothing did
    and the row is a question rather than an answer.
    """

    key: str
    label: str
    verdict: str
    reason: str = ""
    remedy: str = ""
    local: str = ""
    remote: str = ""
    provenance: str = MEASURED
    evidence: str = ""
    register_key: Optional[str] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def _cap_value(facts: RigFacts, key: str) -> Optional[str]:
    entry = facts.capabilities.get(key)
    if isinstance(entry, dict):
        return entry.get("value")
    return entry


def _row_nccl_colocation(
    local: RigFacts, remote: RigFacts, colocation_wanted: bool
) -> GateRow:
    lv, rv = local.nccl_tuple(), remote.nccl_tuple()
    want = ".".join(str(x) for x in COLOCATION_NCCL_MIN)
    shown_l = local.nccl or "not reported"
    shown_r = remote.nccl or "not reported"
    if lv is None or rv is None:
        return GateRow(
            key="nccl_colocation",
            label=f"NCCL >= {want} (several ranks per GPU)",
            verdict=WARN if colocation_wanted else OK,
            local=shown_l,
            remote=shown_r,
            reason="At least one rig has not reported its NCCL version, so "
            "the co-location threshold cannot be checked.",
            remedy="Run the comm suite on the rig that reports nothing "
            "(Rig-data tab, or POST /api/commsuite/run) and import its "
            "artifact here.",
            provenance=ABSENT,
            evidence="absent",
        )
    too_old = [
        name
        for name, ver in (("this rig", lv), ("the far rig", rv))
        if ver[:2] < COLOCATION_NCCL_MIN
    ]
    if not too_old:
        return GateRow(
            key="nccl_colocation",
            label=f"NCCL >= {want} (several ranks per GPU)",
            verdict=OK,
            local=shown_l,
            remote=shown_r,
            reason="Both rigs can host several ranks on one physical GPU.",
            provenance=MEASURED,
            evidence="runbook §6.2",
        )
    return GateRow(
        key="nccl_colocation",
        label=f"NCCL >= {want} (several ranks per GPU)",
        # A rig below the threshold is not a broken coupling -- it is a
        # coupling that must put one rank per card. Only a plan that asks for
        # co-location is blocked by it.
        verdict=BLOCK if colocation_wanted else WARN,
        local=shown_l,
        remote=shown_r,
        reason=f"Below {want} on {' and '.join(too_old)}; co-location is "
        "refused there. One rank per card still works.",
        remedy="Run those ranks from the co-location image "
        "(docker/htsglang.Dockerfile carries NCCL 2.30.7), or keep the plan "
        "at one rank per card.",
        provenance=MEASURED,
        evidence="runbook §6.2",
    )


def _row_tree_commit(local: RigFacts, remote: RigFacts) -> GateRow:
    if not local.commit or not remote.commit:
        return GateRow(
            key="tree_commit",
            label="Same sglang tree on both rigs",
            verdict=WARN,
            local=local.commit or "not reported",
            remote=remote.commit or "not reported",
            reason="One side does not report its commit, so the trees cannot "
            "be compared. Requests are msgspec structs broadcast rank to "
            "rank: a field one side does not know kills the other at "
            "deserialization, minutes into an otherwise healthy boot.",
            remedy="Read <RIG2_SGLANG_SRC>/SYNCED_COMMIT.txt on the far rig, "
            "or re-sync the whole python/sglang tree from the host.",
            provenance=ABSENT,
            evidence="runbook §4.3",
        )
    same = local.commit[:12] == remote.commit[:12]
    return GateRow(
        key="tree_commit",
        label="Same sglang tree on both rigs",
        verdict=OK if same else BLOCK,
        local=local.commit[:12],
        remote=remote.commit[:12],
        reason=(
            "Both rigs run the same tree."
            if same
            else "The two rigs run different trees. Requests are msgspec "
            "structs broadcast rank to rank, so the older side dies at "
            "deserialization on a field the newer side added."
        ),
        remedy=(
            ""
            if same
            else "Sync the whole python/sglang tree (not just the transport "
            "file) from the PVE host and update SYNCED_COMMIT.txt."
        ),
        provenance=MEASURED,
        evidence="runbook §4.3",
    )


def _arch_rows(
    local: RigFacts, remote: RigFacts, checkpoint_format: Optional[str]
) -> List[GateRow]:
    """The architecture walls, one row each, only for archs actually present.

    A wall is reported against the rig that has the card, and it says whether
    the architecture was declared by that rig or inferred from the card model
    here -- an inferred arch that blocks a coupling has to be checkable.
    """
    rows: List[GateRow] = []
    turing: List[Tuple[str, CardFacts]] = []
    for facts, side in ((local, "this rig"), (remote, "the far rig")):
        for c in facts.cards_with_arch("sm75"):
            turing.append((side, c))
    if turing:
        where = ", ".join(f"{c.name} ({side})" for side, c in turing)
        inferred = any(c.arch_source != "declared" for _s, c in turing)
        prov = ESTIMATE if inferred else MEASURED
        entry = rejectedmod.by_key("gguf_on_sm75")
        if (checkpoint_format or "").lower() == "gguf":
            rows.append(
                GateRow(
                    key="arch_gguf_sm75",
                    label="GGUF checkpoint on a Turing rank",
                    verdict=BLOCK,
                    reason=(entry.verdict if entry else "GGUF does not run on sm75.")
                    + f" Turing cards in this coupling: {where}.",
                    remedy="Take a safetensors checkpoint for the lane that "
                    "includes the Turing card, or keep that card out of the "
                    "lane that loads GGUF.",
                    remote=where,
                    provenance=prov,
                    evidence=f"register:{entry.key}" if entry else "register",
                    register_key=entry.key if entry else None,
                )
            )
        else:
            rows.append(
                GateRow(
                    key="arch_gguf_sm75",
                    label="GGUF checkpoint on a Turing rank",
                    verdict=OK,
                    reason="No GGUF checkpoint is named for this coupling, so "
                    f"the sm75 wall does not apply. Turing cards: {where}.",
                    provenance=prov,
                    evidence=f"register:{entry.key}" if entry else "register",
                    register_key=entry.key if entry else None,
                )
            )
        rows.append(
            GateRow(
                key="arch_bf16_turing",
                label="bf16 on Turing",
                verdict=WARN,
                reason=f"Turing has no bf16 ({where}), so a run that spans it "
                "must be float16 on EVERY stage -- a mixed-dtype group is not "
                "a configuration, it is a silent numerics change.",
                remedy="Pass --dtype float16 on all stages of the lane that "
                "includes the Turing card.",
                remote=where,
                provenance=prov,
                evidence="runbook §4.9",
            )
        )
        rows.append(
            GateRow(
                key="arch_flashinfer_turing",
                label="flashinfer prefill on Turing",
                verdict=WARN,
                reason="flashinfer's prefill asks 65616 B of shared memory "
                "against Turing's 65536 at head_dim 256, which every Qwen3.5 "
                "size has.",
                remedy="Pass --attention-backend triton on the Turing stage. "
                "The backend is a per-process choice and the stages share no "
                "KV pool, so only that stage has to change.",
                remote=where,
                provenance=prov,
                evidence="runbook §4.9",
            )
        )
    amd = [c for f in (local, remote) for c in f.cards if c.vendor == "amd"]
    if amd:
        rows.append(
            GateRow(
                key="vendor_mixed",
                label="Mixed vendors in one lane",
                verdict=WARN,
                reason="An AMD card is in the pool ("
                + ", ".join(c.name for c in amd)
                + "). NCCL cannot span vendors; only the HTCCL data plane can.",
                remedy="Set SGLANG_HTCCL=1 on every rank of a lane that mixes "
                "vendors, and keep the value identical on all of them.",
                provenance=ESTIMATE,
                evidence="runbook §2",
            )
        )
    return rows


def _row_transport_available(local: RigFacts, remote: RigFacts) -> GateRow:
    """Is there any cross-rig carrier at all, and which one is ruled out.

    The verbs row is the reason this check is not a formality: NCCL's ibverbs
    path is broken on this fabric (``IBV_WC_REM_INV_REQ_ERR`` on the first
    proxy tensor) while UCX drives the same two HCAs. A gate that checked
    "RDMA present" would pass and the boot would still die.
    """
    have_ucx = [f.rig_id for f in (local, remote) if f.ucx]
    if len(have_ucx) == 2:
        return GateRow(
            key="transport_available",
            label="A cross-rig carrier exists",
            verdict=OK,
            local=local.ucx or "",
            remote=remote.ucx or "",
            reason="Both rigs report UCX, so HTCCL/ucx can carry the TP "
            "collectives. NCCL's verbs path is broken on this RoCE fabric; "
            "NCCL over sockets on the same HCA works and is the fallback.",
            remedy="Both hosts must load the SAME UCX release "
            "(SGLANG_HTCCL_UCX_LIB); mixed releases are refused at rendezvous.",
            provenance=MEASURED,
            evidence="runbook §8.2 / FEATURES §21 fabric finding",
        )
    missing = "this rig" if local.ucx is None else "the far rig"
    if not have_ucx:
        missing = "either rig"
    return GateRow(
        key="transport_available",
        label="A cross-rig carrier exists",
        verdict=WARN,
        local=local.ucx or "not reported",
        remote=remote.ucx or "not reported",
        reason=f"UCX is not reported on {missing}, so the HTCCL/ucx rung "
        "cannot be confirmed. gloo over TCP always works and NCCL over "
        "sockets works on this fabric; NCCL over verbs does not.",
        remedy="Run the comm suite on the rig that reports nothing, or "
        "install UCX there and set SGLANG_HTCCL_UCX_LIB to the same release "
        "on both hosts.",
        provenance=ABSENT if not have_ucx else ESTIMATE,
        evidence="runbook §8.2 / FEATURES §21 fabric finding",
    )


def _row_dmabuf_rdma(local: RigFacts, remote: RigFacts) -> GateRow:
    """The dmabuf GPU-RDMA precondition chain (#214, EVAL_gdr_uebernahme.md
    §6.2 / §9 P1) -- a CAPABILITY row, not a build recommendation.

    Four checks decide whether a `dmabuf_rdma` HTCCL transport (the
    evaluation's P4, explicitly NOT built now) even has a floor to stand on:
    open kernel modules, rdma-core with `ibv_reg_dmabuf_mr`, the mlx5 dmabuf
    kernel path, and this fork's VMM export path. This row never blocks a
    coupling -- dmabuf_rdma is not a transport either rig plan requires, so
    an unmet precondition here is exactly the same kind of fact as
    `transport_available` reporting on UCX: visible and actionable, not a
    reason to refuse the coupling.

    What this row deliberately does NOT do: treat "same PCIe switch as the
    NIC" as a precondition for a card acting as an RDMA source. The
    evaluation's own counter-datum is that on the reference rig the 5090
    reaches the NIC through the root complex -- not the NIC's switch -- and
    still works as a source; the 2080 Ti failure the handover blamed on
    topology therefore needs a different explanation, and baking the
    hypothesis into a BLOCK would gate a working card on unproven grounds.
    That reasoning is carried as a WARN note on every row, not as a fifth
    check that could fail.
    """
    topology_note = (
        " Not gated here: the handover's own hypothesis that an RDMA-source "
        "failure follows from sharing (or not sharing) the NIC's PCIe "
        "switch. On the reference rig the 5090 sits on the root complex, not "
        "the NIC's switch, and still works as an RDMA source -- a "
        "counter-datum to the hypothesis, not a confirmation of it "
        "(EVAL_gdr_uebernahme.md §1.2). Only a per-card source probe may "
        "ever decide that, never PCIe placement."
    )

    def _side(facts: RigFacts) -> Tuple[str, List[bool], List[bool]]:
        parts: List[str] = []
        reported: List[bool] = []
        passed: List[bool] = []
        for key, label in DMABUF_RDMA_CHECKS:
            value = _cap_value(facts, key)
            reported.append(value is not None)
            passed.append(value is not None and _dmabuf_passed(value))
            parts.append(f"{label}: {value if value is not None else 'not reported'}")
        return "; ".join(parts), reported, passed

    local_str, local_reported, local_passed = _side(local)
    remote_str, remote_reported, remote_passed = _side(remote)
    all_reported = all(local_reported) and all(remote_reported)
    any_reported = any(local_reported) or any(remote_reported)
    all_passed = all(local_passed) and all(remote_passed)

    if not any_reported:
        return GateRow(
            key="dmabuf_rdma",
            label="dmabuf GPU-RDMA precondition chain",
            verdict=WARN,
            reason=(
                "None of the four dmabuf_rdma preconditions (open kernel "
                "modules, rdma-core with ibv_reg_dmabuf_mr, mlx5 dmabuf "
                "path, VMM export) have been probed on either rig."
                + topology_note
            ),
            remedy="Run the host-side checks named in "
            "EVAL_gdr_uebernahme.md §6.2 (uname -r / lsmod, dpkg -l "
            "libibverbs1, /sys/class/infiniband, read_dmabuf_flag.sh) and "
            "record the results as capabilities on each rig's profile.",
            provenance=ABSENT,
            evidence="absent",
        )

    if all_reported and all_passed:
        return GateRow(
            key="dmabuf_rdma",
            label="dmabuf GPU-RDMA precondition chain",
            verdict=OK,
            local=local_str,
            remote=remote_str,
            reason=(
                "Every precondition for a dmabuf_rdma transport is reported "
                "met on both rigs. This is a capability check, not a build "
                "recommendation: the #214 desk evaluation's verdict "
                "(EVAL_gdr_uebernahme.md §0) is that the adoption case does "
                "not hold at this fork's message sizes -- do not build the "
                "transport on the strength of this row alone."
                + topology_note
            ),
            provenance=MEASURED,
            evidence="EVAL_gdr_uebernahme.md §6.2 / §9 P1",
        )

    missing = sorted(
        {
            label
            for (key, label), ok in zip(DMABUF_RDMA_CHECKS, local_passed)
            if not ok
        }
        | {
            label
            for (key, label), ok in zip(DMABUF_RDMA_CHECKS, remote_passed)
            if not ok
        }
    )
    return GateRow(
        key="dmabuf_rdma",
        label="dmabuf GPU-RDMA precondition chain",
        verdict=WARN,
        local=local_str,
        remote=remote_str,
        reason=(
            "Not every dmabuf_rdma precondition is confirmed on both rigs. "
            "Missing or unconfirmed: " + ", ".join(missing) + "."
            + topology_note
        ),
        remedy="Run the host-side checks named in EVAL_gdr_uebernahme.md "
        "§6.2 on whichever rig is unreported, or close the reported gap "
        "(e.g. KvVmmArena._prop does not yet set requestedHandleTypes for "
        "the POSIX_FD path, EVAL_gdr_uebernahme.md §4.3).",
        provenance=MEASURED if all_reported else ESTIMATE,
        evidence="EVAL_gdr_uebernahme.md §6.2 / §9 P1",
    )


def _row_cuda_graph(transports: Sequence["TransportChoice"]) -> GateRow:
    """Host-staged transports and CUDA graphs cannot both be on.

    Only the DATA plane counts. The control plane is a gloo process group for
    rendezvous; it carries no collective inside a capture, so letting it drag
    ``--disable-cuda-graph`` into every plan would be a warning that fires on
    everything and therefore says nothing.
    """
    data_plane = [t for t in transports if t.message_class in ("tp_small", "tp_bulk")]
    staged = [t for t in data_plane if t.chosen in ("htccl-ucx", "gloo-tcp", "shm")]
    if not staged:
        return GateRow(
            key="cuda_graph",
            label="CUDA graphs vs the chosen transport",
            verdict=OK,
            reason="No host-staged transport has been chosen yet, so nothing "
            "constrains graph capture.",
            provenance=ABSENT,
            evidence="runbook §6.3",
        )
    names = sorted({t.chosen for t in staged if t.chosen})
    return GateRow(
        key="cuda_graph",
        label="CUDA graphs vs the chosen transport",
        verdict=WARN,
        reason="The chosen carrier(s) " + ", ".join(names) + " host-stage "
        "every collective. Only the `device` HTCCL transport may run inside a "
        "capture; a graph-enabled boot with these is rejected at startup.",
        remedy="Add --disable-cuda-graph to the cross-rig lane, and never "
        "compare its numbers against a graph-enabled intra-rig run without "
        "saying so.",
        provenance=MEASURED,
        evidence="runbook §6.3",
    )


def _row_model_fit(model_fit: Optional[dict]) -> GateRow:
    """The planner's own arithmetic about the checkpoint on the pooled cards.

    Absent is the normal answer here: a coupling can be planned before a
    checkpoint is chosen, and inventing a fit for an unnamed model would be
    the worst kind of stand-in -- one that decides whether a rig is worth
    coupling at all.
    """
    fit = dict(model_fit or {})
    state = fit.get("state")
    if state is None:
        return GateRow(
            key="model_fit",
            label="The checkpoint fits the pooled cards",
            verdict=WARN,
            reason="No checkpoint was named, so no fit was computed.",
            remedy="Name a model on the Guide tab, or pass model_path to this "
            "endpoint, to have the planner size it against the pooled cards.",
            provenance=ABSENT,
            evidence="absent",
        )
    if state == "ok":
        return GateRow(
            key="model_fit",
            label="The checkpoint fits the pooled cards",
            verdict=OK,
            reason=fit.get("reason") or "The planner sizes this checkpoint "
            "onto the pooled cards.",
            local=str(fit.get("rig") or ""),
            provenance=fit.get("provenance") or ESTIMATE,
            evidence=fit.get("evidence") or "planner capacity arithmetic",
        )
    return GateRow(
        key="model_fit",
        label="The checkpoint fits the pooled cards",
        verdict=BLOCK,
        reason=fit.get("reason") or "The planner does not size this "
        "checkpoint onto the pooled cards.",
        remedy=fit.get("remedy")
        or "Pick a smaller checkpoint, a stronger quantisation, or a lane "
        "with more cards; the Guide tab shows what each buys.",
        local=str(fit.get("rig") or ""),
        provenance=fit.get("provenance") or ESTIMATE,
        evidence=fit.get("evidence") or "planner capacity arithmetic",
    )


def _register_rows(tags: Sequence[str]) -> List[GateRow]:
    """Everything the rejected register already settled about these tags.

    The register is the project's memory of what was tried. Rendering it here
    as data rather than as prose is what keeps a coupling from proposing a
    combination that was measured and put down -- and every row carries the
    counter-number, because a rejection without its number is an opinion.
    """
    rows: List[GateRow] = []
    for entry in rejectedmod.check_combination(list(tags)):
        rows.append(
            GateRow(
                key=f"register:{entry.key}",
                label=entry.what,
                verdict=BLOCK if entry.level == rejectedmod.BLOCKED else WARN,
                reason=entry.verdict,
                remedy=(
                    "Never offered; the register entry is the decision."
                    if entry.level == rejectedmod.BLOCKED
                    else "Available on request, with the measurement attached."
                ),
                provenance=MEASURED,
                evidence=entry.evidence
                + (" [verdict holds for this rig only]" if entry.scope == "rig" else ""),
                register_key=entry.key,
            )
        )
    return rows


def gate(
    local: RigFacts,
    remote: RigFacts,
    *,
    checkpoint_format: Optional[str] = None,
    model_fit: Optional[dict] = None,
    colocation_wanted: bool = False,
    transports: Sequence["TransportChoice"] = (),
    extra_tags: Sequence[str] = (),
) -> List[GateRow]:
    """Every precondition of a coupling, each with its verdict and evidence.

    The order is the order a reader should think in: can the two talk at all
    (tree, NCCL), can the cards run the same thing (arch walls), is there a
    wire (transport), does the workload fit (model), and what has the project
    already settled about this shape (register).
    """
    rows = [
        _row_tree_commit(local, remote),
        _row_nccl_colocation(local, remote, colocation_wanted),
    ]
    rows.extend(_arch_rows(local, remote, checkpoint_format))
    rows.append(_row_transport_available(local, remote))
    rows.append(_row_dmabuf_rdma(local, remote))
    rows.append(_row_cuda_graph(transports))
    rows.append(_row_model_fit(model_fit))
    tags = set(extra_tags) | {"crossrig-tp-push"}
    if (checkpoint_format or "").lower() == "gguf":
        tags.add("gguf")
    if any(c.arch == "sm75" for f in (local, remote) for c in f.cards):
        tags.add("sm75")
    rows.extend(_register_rows(sorted(tags)))
    return rows


# ===========================================================================
# Transport per message class
# ===========================================================================
@dataclasses.dataclass
class TransportChoice:
    """One message class, the carrier proposed for it, and why.

    ``provenance`` is about THIS choice, not about the carrier in general: a
    carrier picked because it is the only one available is an ``estimate``
    even when the carrier itself has been measured elsewhere.
    """

    message_class: str
    label: str
    carrier: str
    flag: str
    env: str
    wants: str
    chosen: Optional[str] = None
    provenance: str = ABSENT
    reason: str = ""
    candidates: List[dict] = dataclasses.field(default_factory=list)
    evidence: List[dict] = dataclasses.field(default_factory=list)
    how_to_measure: str = ""
    note: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


#: Cross-rig carriers, with the standing facts about each. Speed does not
#: appear here -- speed comes from measurements or not at all.
_CARRIERS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "htccl-ucx",
        "label": "HTCCL over UCX",
        "classes": ("tp_small", "tp_bulk"),
        "why": "Host-staged, vendor-neutral, and the transport the measured "
        "cross-rig TP=4 boot actually used.",
        "needs": "UCX on both hosts, the same release on each.",
    },
    {
        "key": "nccl-sockets",
        "label": "NCCL over sockets",
        "classes": ("tp_small", "tp_bulk"),
        "why": "Works on this fabric and reaches the 40G line when "
        "NCCL_SOCKET_IFNAME is pinned to the RoCE interface.",
        "needs": "NCCL_IB_DISABLE=1 plus NCCL_SOCKET_IFNAME on the fast "
        "interface; unpinned it silently takes the 1 GbE route.",
    },
    {
        "key": "nccl-verbs",
        "label": "NCCL over ibverbs",
        "classes": ("tp_small", "tp_bulk"),
        "why": "Would buy latency on paper.",
        "needs": "verbs. BROKEN on this RoCE fabric: the first 5120-byte "
        "proxy tensor returns IBV_WC_REM_INV_REQ_ERR(9) and the communicator "
        "dies. NCCL_IB=1 re-arms it for whoever wants to chase it.",
        "broken": True,
    },
    {
        "key": "gloo-tcp",
        "label": "gloo over TCP",
        "classes": ("tp_small", "tp_bulk", "control"),
        "why": "No special hardware; the rung that works wherever the pairing "
        "connection already works.",
        "needs": "nothing beyond a route.",
    },
    {
        "key": "mooncake",
        "label": "mooncake / nixl transfer engine",
        "classes": ("kv_bulk",),
        "why": "The only carrier for PD-KV and HiCache bulk extents.",
        "needs": "the mooncake wheel plus libibverbs1/librdmacm1 on both "
        "hosts; --collective-net-bulk seeds --disaggregation-ib-device.",
    },
)

#: The one command that turns an absent link row into a measured one. Kept as
#: text, not as an action: this module never starts a measurement.
_HOW_TO_MEASURE_LINK = (
    "Run the comm suite on the far rig and import its artifact "
    "(Rig-data tab there, or POST /api/commsuite/run + GET "
    "/api/commsuite/status), and run the cross-rig arm from the PVE host -- "
    "the dev container has no route to the fast line, so a figure taken here "
    "would describe the 1 GbE LAN. See the host steps in this report."
)


def _wire_rows(*digests: Optional[dict]) -> List[dict]:
    """Measurement rows that describe a WIRE between two rigs.

    A row qualifies only if it was taken by the cross-rig arm or explicitly
    tagged ``pair: cross-rig``. Everything else in an artifact was measured
    over loopback or inside one box, and a loopback number wearing a wire's
    label is precisely the dishonesty the whole provenance vocabulary exists
    to prevent.
    """
    out: List[dict] = []
    for d in digests:
        for m in (d or {}).get("measurements") or []:
            mid = str(m.get("id") or "")
            ctx = m.get("context") or {}
            if mid.startswith("comm/cross_rig/") or ctx.get("pair") == "cross-rig":
                if m.get("value") is not None:
                    out.append(m)
    return out


def _best_row(rows: Sequence[dict], wants: str) -> Optional[dict]:
    """Lowest latency, or highest bandwidth, from rows that state their unit."""
    if wants == "latency":
        cand = [r for r in rows if str(r.get("unit") or "").lower() in ("us", "ms")]
        return min(cand, key=lambda r: r["value"]) if cand else None
    cand = [
        r
        for r in rows
        if "b/s" in str(r.get("unit") or "").lower()
        or "bit" in str(r.get("unit") or "").lower()
    ]
    return max(cand, key=lambda r: r["value"]) if cand else None


def transport_plan(
    local: RigFacts,
    remote: RigFacts,
    *,
    local_digest: Optional[dict] = None,
    remote_digest: Optional[dict] = None,
    pair_matrix: Optional[Sequence[dict]] = None,
) -> List[TransportChoice]:
    """One row per message class: carrier, control, provenance, evidence.

    The rule (DESIGN_216) is that the choice falls out of the measurements,
    not out of the configuration. Where the measurements are silent the row
    says ``absent`` and carries the command that would fill it -- it does not
    fall back to a default that reads like a decision.

    ``pair_matrix`` is the #213 ordered pair matrix. It describes cards INSIDE
    one rig, so it never decides a cross-rig class; it is carried as context
    for the intra-rig legs of a lane, and its absence is reported the same way.
    """
    wire = _wire_rows(local_digest, remote_digest)
    ucx_both = bool(local.ucx and remote.ucx)
    rows: List[TransportChoice] = []
    for spec in MESSAGE_CLASSES:
        cls = spec["key"]
        candidates: List[dict] = []
        for c in _CARRIERS:
            if cls not in c["classes"]:
                continue
            entry = {
                "key": c["key"],
                "label": c["label"],
                "why": c["why"],
                "needs": c["needs"],
            }
            if c.get("broken"):
                entry["verdict"] = "unavailable"
                entry["reason"] = c["needs"]
            elif c["key"] == "htccl-ucx" and not ucx_both:
                entry["verdict"] = "unknown"
                entry["reason"] = "UCX is not reported on both rigs."
            else:
                entry["verdict"] = "usable"
                entry["reason"] = c["why"]
            candidates.append(entry)

        usable = [c for c in candidates if c["verdict"] == "usable"]
        best = _best_row(wire, spec["wants"]) if spec["wants"] != "reachability" else None
        chosen: Optional[str] = None
        provenance = ABSENT
        reason = ""
        evidence: List[dict] = []

        if cls == "control":
            # Deliberately left on the slow LAN: the control plane takes
            # interface names rather than RDMA device names, and the reference
            # bring-up keeps it off the fast line on purpose.
            chosen = "gloo-tcp"
            provenance = MEASURED
            reason = (
                "The control plane stays on the 1 GbE LAN by design: it "
                "carries rendezvous, not payload, and pinning it to the fast "
                "line buys nothing while adding a way to misconfigure it."
            )
        elif best is not None:
            backend = str((best.get("context") or {}).get("backend") or "")
            match = [c for c in usable if backend and backend in c["key"]]
            chosen = (match or usable or [{"key": None}])[0]["key"]
            provenance = MEASURED
            evidence = [
                {
                    "id": best.get("id"),
                    "label": best.get("label"),
                    "value": best.get("value"),
                    "unit": best.get("unit"),
                    "taken_at": best.get("taken_at"),
                }
            ]
            reason = (
                f"Ranked from a measured cross-rig row ({best.get('id')}): "
                f"{best.get('value')} {best.get('unit')}."
            )
        elif len(usable) == 1:
            chosen = usable[0]["key"]
            provenance = ESTIMATE
            reason = (
                f"{usable[0]['label']} is the only carrier available for this "
                "class on these two rigs; the choice is trivial rather than "
                "measured."
            )
        elif usable:
            provenance = ABSENT
            reason = (
                f"{len(usable)} carriers could take this class and nothing has "
                "been measured across the rig boundary, so none is proposed."
            )
        else:
            provenance = ABSENT
            reason = "No carrier for this class is available on both rigs."

        note = ""
        if spec["separable_from"]:
            note = (
                "Not separable from "
                + spec["separable_from"]
                + " today: one UcpWorker context per rank carries both, and "
                "the flag name should not be read as claiming otherwise."
            )
        rows.append(
            TransportChoice(
                message_class=cls,
                label=spec["label"],
                carrier=spec["carrier"],
                flag=spec["flag"],
                env=spec["env"],
                wants=spec["wants"],
                chosen=chosen,
                provenance=provenance,
                reason=reason,
                candidates=candidates,
                evidence=evidence,
                how_to_measure="" if provenance == MEASURED else _HOW_TO_MEASURE_LINK,
                note=note,
            )
        )

    if pair_matrix is not None:
        # Carried, not consumed: the ordered pair matrix is an intra-rig
        # measurement and says nothing about the boundary.
        for r in rows:
            r.candidates.append(
                {
                    "key": "intra-rig-pairs",
                    "label": "#213 ordered pair matrix",
                    "verdict": "context",
                    "reason": f"{len(list(pair_matrix))} ordered intra-rig "
                    "pair(s) measured; they size the legs inside each rig, "
                    "not the wire between them.",
                }
            )
    return rows


# ===========================================================================
# The card pool and the lanes that can be formed from it
# ===========================================================================
@dataclasses.dataclass
class LaneCandidate:
    """One lane that COULD be formed from the pool.

    A lane is a list of cards, and a coupling offers several -- the dual-group
    runtime decides at run time which ones exist simultaneously. Nothing here
    reserves a card; ``exclusive`` records the one hard rule, that a card
    carries at most one lane at a time.
    """

    key: str
    label: str
    scope: str  # "intra" | "cross"
    cards: List[str] = dataclasses.field(default_factory=list)
    blocked_by: List[str] = dataclasses.field(default_factory=list)
    note: str = ""
    exclusive: bool = True

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def card_pool(
    local: RigFacts, remote: RigFacts, gate_rows: Sequence[GateRow] = ()
) -> dict:
    """The coupled cards as a POOL, plus the lanes that can be cut from it.

    Deliberately not "the coupled rig": two rigs coupled today may be run as
    one lane spanning both, as two independent lanes, or as a lane plus a
    spare. Returning a single implied verbund here would push that decision
    into the coupling, where nothing knows the workload yet.
    """
    cards = [dict(c.to_json(), rig=local.rig_id) for c in local.cards]
    cards += [dict(c.to_json(), rig=remote.rig_id) for c in remote.cards]
    blocking = [r.key for r in gate_rows if r.verdict == BLOCK]

    lanes: List[LaneCandidate] = []
    for facts in (local, remote):
        if not facts.cards:
            continue
        lanes.append(
            LaneCandidate(
                key=f"lane-{facts.rig_id}",
                label=f"all cards of {facts.label or facts.rig_id}",
                scope="intra",
                cards=[c.label for c in facts.cards],
                note="Stays inside one rig: no cross-rig transport, no "
                "host-staging cost, and the intra-rig collectives are the "
                "ones the #213 pair matrix measured.",
            )
        )
    if local.cards and remote.cards:
        lanes.append(
            LaneCandidate(
                key="lane-cross",
                label="one lane spanning both rigs",
                scope="cross",
                cards=[c.label for c in local.cards] + [c.label for c in remote.cards],
                blocked_by=blocking,
                note="Every blocking gate row above applies to this lane and "
                "to no other; a blocked cross lane still leaves the intra-rig "
                "lanes usable.",
            )
        )
    archs = sorted({c.arch for c in local.cards + remote.cards if c.arch})
    return {
        "cards": cards,
        "card_count": len(cards),
        "rigs": [local.rig_id, remote.rig_id],
        "archs": archs,
        "vendors": sorted(set(local.vendors()) | set(remote.vendors())),
        "lane_candidates": [ln.to_json() for ln in lanes],
        "note": "A coupling produces a POOL of cards. Lanes are cut from it "
        "at run time and a card carries at most one lane at a time; this "
        "report never assumes a single verbund.",
    }


# ===========================================================================
# What only the host can do
# ===========================================================================
@dataclasses.dataclass
class HostStep:
    """A step this process may not take, with the command that takes it.

    The command is text. Nothing here executes it, and nothing here fills in a
    real address: every value is ``${VAR:-<placeholder>}`` against the
    /root/rig-env.sh convention, so the whole block is safe to paste into a
    repository, an issue or a chat.
    """

    key: str
    title: str
    where: str
    why_not_here: str
    command: str
    runbook: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


_ENV_PREAMBLE = "source /root/rig-env.sh 2>/dev/null || true"

_NO_ROUTE = (
    "The development container has no interface on the cross-rig subnet and "
    "no /dev/infiniband (runbook §1.1), so anything that touches the fast "
    "line runs from the Proxmox host. A measurement taken here would ride "
    "the 1 GbE LAN and describe the wrong wire."
)


def host_steps(
    target: str = "", *, dashboard_url: str = "${UI:-http://127.0.0.1:8791}"
) -> List[HostStep]:
    """The host-only half of a coupling, as copyable commands.

    Every entry answers the same two questions in the same order: why this
    process may not do it, and what to paste where instead.
    """
    peer = "${PEER_RIGMON:-<far-rig-host:port>}" if not target else target
    return [
        HostStep(
            key="far_rig_artifact",
            title="Measure the far rig and bring its artifact back",
            where=FAR,
            why_not_here="This process can read a far rigmon over the LAN, "
            "but it cannot make the far rig measure itself -- the comm suite "
            "runs where the cards are.",
            command="\n".join(
                [
                    _ENV_PREAMBLE,
                    'ssh -i "${RIG2_KEY:-<rig2-key>}" root@"${RIG2_HOST:-<rig2-host>}" \\',
                    '  "PYTHONPATH=${RIG2_SGLANG_SRC:-<rig2-sglang-src>} \\',
                    '   ${RIG2_VENV:-<rig2-venv>}/bin/python -m sglang.planner \\',
                    '   --serve --host 0.0.0.0 --port 8791" &',
                    "",
                    "# one button, ~10-12 s; then take the curated digest",
                    "curl -s -X POST http://%s/api/commsuite/run -d '{}'" % peer,
                    "curl -s http://%s/api/commsuite/status \\" % peer,
                    "  > /tmp/far-rig-artifact.json",
                ]
            ),
            runbook="§1.3, §8.1",
        ),
        HostStep(
            key="import_artifact",
            title="Import that artifact into this coupling",
            where=HERE,
            why_not_here="Nothing: this step runs here. It is listed so the "
            "sequence is complete when read as a script.",
            command="\n".join(
                [
                    "UI=%s" % dashboard_url,
                    "python3 - <<'PY' > /tmp/coupling-body.json",
                    "import json",
                    "art=json.load(open('/tmp/far-rig-artifact.json'))",
                    "job=art.get('job') or {}",
                    "print(json.dumps({'remote_artifact': job.get('artifact') or art}))",
                    "PY",
                    "curl -s -X POST $UI/api/rig_coupling/plan \\",
                    "  -d @/tmp/coupling-body.json | python3 -m json.tool | head -60",
                ]
            ),
            runbook="§8",
        ),
        HostStep(
            key="link_cost",
            title="Measure the wire itself (per-collective cost on the link)",
            where=HOST,
            why_not_here=_NO_ROUTE,
            command="\n".join(
                [
                    _ENV_PREAMBLE,
                    "# rank 0 on the PVE host, rank 1 on the far rig; the comm",
                    "# dir is the rendezvous both sides read.",
                    'ssh -i "${RIG1_KEY:-<rig1-key>}" root@"${RIG1_HOST:-<rig1-host>}" \\',
                    '  "python3 ${REPO_ROOT:-<repo-root>}/scripts/r3val/link_collective_cost.py \\',
                    '   --rank 0 --world 2 --comm-dir ${COMM_DIR:-<shared-comm-dir>} \\',
                    '   --op all_reduce --out /tmp/link-rank0.json"',
                    'ssh -i "${RIG2_KEY:-<rig2-key>}" root@"${RIG2_HOST:-<rig2-host>}" \\',
                    '  "python3 ${RIG2_SGLANG_SRC:-<rig2-sglang-src>}/../scripts/r3val/link_collective_cost.py \\',
                    '   --rank 1 --world 2 --comm-dir ${COMM_DIR:-<shared-comm-dir>} \\',
                    '   --op all_reduce --out /tmp/link-rank1.json"',
                ]
            ),
            runbook="§4.3, §8.2",
        ),
        HostStep(
            key="tree_sync",
            title="Put the same tree on the far rig",
            where=HOST,
            why_not_here="An rsync started in the container silently rides "
            "the 1 GbE LAN and saturates at ~105-118 MB/s. Hopping via the "
            "host takes the 40G line instead -- roughly ten times faster, and "
            "the transfer log is where that mistake shows up.",
            command="\n".join(
                [
                    _ENV_PREAMBLE,
                    'ssh -i "${RIG1_KEY:-<rig1-key>}" root@"${RIG1_HOST:-<rig1-host>}" \\',
                    '  "rsync -a --delete --exclude=__pycache__ \\',
                    '   ${WORKTREE:-<worktree>}/python/sglang/ \\',
                    '   root@${RDMA_R2:-<far-rig-fast-ip>}:${RIG2_SGLANG_SRC:-<rig2-sglang-src>}/sglang/"',
                    "# then update <RIG2_SGLANG_SRC>/SYNCED_COMMIT.txt",
                ]
            ),
            runbook="§4.3",
        ),
    ]


def model_fit_report(cards: Sequence[CardFacts], model_path: str, **plan_kwargs) -> dict:
    """Ask the planner whether ``model_path`` sizes onto the pooled cards.

    The one function here that reads something off disk (the checkpoint's
    config), which is why it is a separate call rather than part of
    :func:`couple`: a coupling must be plannable before a model is chosen, and
    a fit computed from a guessed checkpoint would be the worst stand-in this
    report could carry.

    The arithmetic itself is the planner's -- ``feasibility.plan`` via the
    explorer's single-cell path, the same code the Rigs tab draws. Nothing is
    re-derived here, so the coupling and the Guide cannot disagree about the
    same model on the same cards.
    """
    from sglang.srt.planner.explorer import _plan_one
    from sglang.srt.planner.hardware import GpuDescriptor, HardwareSpec

    gpus = tuple(
        GpuDescriptor(index=i, name=c.name, total_mib=int(c.vram_mib or 0))
        for i, c in enumerate(cards)
        if c.vram_mib
    )
    if not gpus:
        return {
            "state": None,
            "reason": "No card in the pool reports its VRAM total, so nothing "
            "can be sized.",
            "provenance": ABSENT,
        }
    # source="manual": a pool assembled from two rigs has no live free-VRAM
    # reading and no measured interconnect, and the planner's own honesty
    # rules key off exactly this field.
    spec = HardwareSpec(gpus=gpus, source="manual")
    cell = _plan_one(model_path, spec, **plan_kwargs)
    return {
        "state": "ok" if cell.fits else "block",
        "reason": cell.reason
        or (
            "The planner sizes this checkpoint onto the pooled cards "
            f"({cell.rig})."
        ),
        "rig": cell.rig,
        "provenance": ESTIMATE,
        "evidence": "planner feasibility.plan over the pooled cards "
        "(declared totals, no live free-VRAM reading)",
        "max_context_tokens": cell.max_context_tokens,
        "launch_flags": list(cell.launch_flags),
    }


# ===========================================================================
# The whole report
# ===========================================================================
@dataclasses.dataclass
class CouplingReport:
    target: str
    local: RigFacts
    remote: RigFacts
    gate_rows: List[GateRow]
    transports: List[TransportChoice]
    pool: dict
    host_steps: List[HostStep]

    @property
    def verdict(self) -> str:
        if any(r.verdict == BLOCK for r in self.gate_rows):
            return BLOCK
        if any(r.verdict == WARN for r in self.gate_rows):
            return WARN
        return OK

    @property
    def summary(self) -> str:
        blocked = [r for r in self.gate_rows if r.verdict == BLOCK]
        if blocked:
            return (
                f"{len(blocked)} gate row(s) block a lane spanning both rigs: "
                + ", ".join(r.label for r in blocked)
                + ". The intra-rig lanes are unaffected."
            )
        warns = [r for r in self.gate_rows if r.verdict == WARN]
        measured = [t for t in self.transports if t.provenance == MEASURED]
        return (
            f"No gate row blocks this coupling ({len(warns)} carry a "
            f"condition). {len(measured)} of {len(self.transports)} message "
            "classes have a measured basis."
        )

    def to_json(self) -> dict:
        return {
            "ok": True,
            "target": self.target,
            "verdict": self.verdict,
            "summary": self.summary,
            "local": self.local.to_json(),
            "remote": self.remote.to_json(),
            "gate": [r.to_json() for r in self.gate_rows],
            "transports": [t.to_json() for t in self.transports],
            "pool": self.pool,
            "host_steps": [h.to_json() for h in self.host_steps],
            "boots_nothing": True,
            "contacts_nothing": True,
        }


def couple(
    local: RigFacts,
    remote: RigFacts,
    *,
    target: str = "",
    checkpoint_format: Optional[str] = None,
    model_fit: Optional[dict] = None,
    colocation_wanted: bool = False,
    local_digest: Optional[dict] = None,
    remote_digest: Optional[dict] = None,
    pair_matrix: Optional[Sequence[dict]] = None,
    extra_tags: Sequence[str] = (),
) -> CouplingReport:
    """Gate, transport plan, card pool and host steps, from facts alone.

    Pure: no socket, no subprocess, no file. Everything this returns was
    computed from the two :class:`RigFacts` and the artifacts handed in, which
    is what makes ``contacts_nothing`` in the payload a property rather than a
    promise.
    """
    transports = transport_plan(
        local,
        remote,
        local_digest=local_digest,
        remote_digest=remote_digest,
        pair_matrix=pair_matrix,
    )
    rows = gate(
        local,
        remote,
        checkpoint_format=checkpoint_format,
        model_fit=model_fit,
        colocation_wanted=colocation_wanted,
        transports=transports,
        extra_tags=extra_tags,
    )
    return CouplingReport(
        target=target,
        local=local,
        remote=remote,
        gate_rows=rows,
        transports=transports,
        pool=card_pool(local, remote, rows),
        host_steps=host_steps(target),
    )
