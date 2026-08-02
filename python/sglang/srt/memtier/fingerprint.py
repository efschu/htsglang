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
"""Which machine is this, and what may a stored profile say about it (#407).

This module exists because of one defect in cut 1: ``bundled_profile()`` was
the default of :meth:`TierRegistry.from_profile`, so **any** machine that built
a registry without arguments received rig-1's host RAM at an estimated
38 GB/s, rig-1's ZFS pool at a measured 1.8 GB/s and rig-1's remote peer over a
40G line -- none of which it has. A measured figure from one rig read as a fact
about all of them is exactly the failure ``profile.py``'s own docstring says it
exists to prevent, and the default undid it.

The fix is not "ship no profile". It is **a profile may only speak about
hardware it matches**, and the match is computed from the identity canon
(#397): NVML UUID primary, PCI BDF as the readable secondary.

Two keys, because two different claims have two different scopes
----------------------------------------------------------------

``hardware_key``
    A digest over the *enumerated cards themselves*: sorted
    ``(uuid, model, total_bytes)``. Two physically distinct machines never
    share it, because a UUID is per card. A profile carrying this key describes
    THIS box, so an exact match licenses everything in it: the host tier, the
    filesystem tiers, the remote tiers. Host RAM size and the mount set are
    deliberately NOT in the key -- adding a DIMM or mounting a disk does not
    make a machine a different machine, and it must not orphan the profile that
    holds its card measurements. A host tier whose size changed is re-read from
    live facts on every boot anyway (``apply_local_facts``).

``model_key``
    A digest over the card *model multiset* only -- ``(model, GiB)`` counted,
    no UUID, no BDF, no host, no mount. Two machines with the same cards share
    it on purpose. A membw figure is a property of a 5090 and not of one
    particular 5090, so a model match licenses **device model templates and
    nothing else**. It does not license a host tier: two boxes with the same
    cards can have different RAM, different disks and a different wire.

The scope rule is enforced by :func:`licensed_document`, not by convention,
and the falsifier is a synthetic foreign rig loading rig-1's profile and
receiving zero of its host/filesystem/remote rows.

Why not ``rig_artifact.rig_fingerprint``
----------------------------------------

That function is the right identity for *sharing* a measurement -- it is
deliberately anonymised (no UUID, no hostname, no PCI) so several identical
machines pool into one sample. It is the wrong identity for *selecting a local
profile*, for two reasons, both structural rather than stylistic:

* it excludes the UUID, so it cannot express "this box", only "a box like
  this" -- and "a box like this" must not license a host or disk row;
* it is not injectable. It reads ``/proc/cpuinfo``, ``/sys/class/net`` and
  ``platform.release()`` inside itself (``rig_artifact.py:273``, ``:288``,
  ``:315``), so a hermetic test cannot compute the key of a synthetic FOREIGN
  rig -- the local machine's CPU, NICs and kernel would leak into it. Every
  function here is pure over its arguments for exactly that reason.

The two are siblings, not rivals: ``model_key`` uses the same VRAM-rounded
``model:NGiB`` spelling ``rig_fingerprint`` uses, so the two vocabularies read
the same on a dashboard.
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import msgspec

from sglang.srt.memtier.profile import CardFact, LocalFacts

__all__ = [
    "FINGERPRINT_VERSION",
    "HardwareFingerprint",
    "MatchScope",
    "ProfileMatch",
    "card_signature",
    "fingerprint_from_facts",
    "hardware_block",
    "hardware_key_for",
    "licensed_document",
    "match_profile",
    "model_key_for",
    "model_signature",
]

#: Bumped when the signature composition changes, so an old stored profile is
#: refused by name rather than matched under new rules.
FINGERPRINT_VERSION = 1

#: Host RAM is classed rather than exact: MemTotal moves by tens of MiB across
#: boots (hugepages, crashkernel, a different cgroup limit) and that must not
#: split one machine into two identities. The ladder is a pure function of the
#: number, so it introduces no rig constant -- a 1 TiB box lands on 1024.
_RAM_CLASS_GIB: Tuple[int, ...] = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)

_GIB = 1024**3
_VENDOR_PREFIX_RE = re.compile(r"^(?:NVIDIA|AMD|Intel)\s+(?:GeForce|Radeon|Arc)?\s*")


def _digest(payload: Any, prefix: str) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _ram_class_gib(total_bytes: Optional[int]) -> Optional[int]:
    if not total_bytes:
        return None
    gib = total_bytes / _GIB
    for rung in _RAM_CLASS_GIB:
        if gib <= rung * 1.08:
            return rung
    # Above the ladder, round to the next 128 GiB so the class stays a class.
    return int(-(-gib // 128) * 128)


def model_signature(cards: Sequence[CardFact]) -> Tuple[str, ...]:
    """``("1x RTX 5090:32GiB", "2x RTX 3080:20GiB")`` -- the model multiset.

    VRAM is rounded to the nearest GiB, the ``rig_artifact.rig_fingerprint``
    rule: two cards of one model report totals a few MiB apart depending on
    ECC and driver state, and that must not split one profile in two.
    """
    counted: Dict[str, int] = {}
    for card in cards:
        model = _VENDOR_PREFIX_RE.sub("", str(card.model or "?")).strip() or "?"
        gib = round((card.total_bytes or 0) / _GIB)
        key = f"{model}:{gib}GiB"
        counted[key] = counted.get(key, 0) + 1
    return tuple(f"{n}x {k}" for k, n in sorted(counted.items()))


def card_signature(cards: Sequence[CardFact]) -> Tuple[Tuple[str, str, int], ...]:
    """Sorted ``(uuid, model, total_bytes)`` -- the exact card set.

    The BDF is deliberately NOT in here even though it is the #397 readable
    secondary: a card that moves slots keeps its UUID and changes its BDF, and
    a re-slotted card is the same card. The BDF stays in the record
    (:class:`~sglang.srt.memtier.profile.CardFact` carries it when the caller
    supplies one) for the artifact adapters, which key on it.
    """
    return tuple(sorted((str(c.uuid), str(c.model), int(c.total_bytes)) for c in cards))


def hardware_key_for(cards: Sequence[Tuple[str, str, int]]) -> str:
    """The exact key over a card signature. Empty for an empty card set."""
    signature = tuple(sorted(tuple(c) for c in cards))
    if not signature:
        return ""
    return _digest(
        {"v": FINGERPRINT_VERSION, "cards": [list(c) for c in signature]}, "hw-"
    )


def model_key_for(models: Sequence[str]) -> str:
    """The portable key over a model multiset. Empty for an empty one."""
    entries = tuple(sorted(str(m) for m in models))
    if not entries:
        return ""
    return _digest({"v": FINGERPRINT_VERSION, "models": list(entries)}, "model-")


class HardwareFingerprint(msgspec.Struct, frozen=True, kw_only=True):
    """What this machine is, computed purely from facts handed in."""

    #: Exact: this box and no other. Empty when no card was enumerated.
    hardware_key: str
    #: Portable: any box with these card models. Empty when no card was seen.
    model_key: str
    cards: Tuple[Tuple[str, str, int], ...] = ()
    models: Tuple[str, ...] = ()
    ram_class_gib: Optional[int] = None
    mounts: Tuple[str, ...] = ()
    #: The logical host name the caller uses for tier ids on this machine.
    host: str = ""

    @property
    def has_cards(self) -> bool:
        return bool(self.cards)

    def label(self) -> str:
        cards = " + ".join(self.models) or "no cards enumerated"
        ram = f", {self.ram_class_gib} GiB RAM class" if self.ram_class_gib else ""
        return f"{cards}{ram}"

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": FINGERPRINT_VERSION,
            "hardware_key": self.hardware_key,
            "model_key": self.model_key,
            "label": self.label(),
            "cards": [list(c) for c in self.cards],
            "models": list(self.models),
            "ram_class_gib": self.ram_class_gib,
            "mounts": list(self.mounts),
            "host": self.host,
        }


def fingerprint_from_facts(facts: LocalFacts, *, host: str = "") -> HardwareFingerprint:
    """The fingerprint of the machine ``facts`` describes. Pure; no I/O.

    A machine with no enumerated card still gets a fingerprint -- with both
    keys empty. That is not a degenerate case to paper over: it is the CPU-only
    or masked-device state, and it must never match a profile that names cards.
    """
    cards = card_signature(facts.cards)
    models = model_signature(facts.cards)
    ram_class = _ram_class_gib(facts.host_total_bytes)
    mounts = tuple(sorted(f.mount for f in facts.filesystems))
    return HardwareFingerprint(
        hardware_key=hardware_key_for(cards),
        model_key=model_key_for(models),
        cards=cards,
        models=models,
        ram_class_gib=ram_class,
        mounts=mounts,
        host=host,
    )


class MatchScope(str, enum.Enum):
    """How much of a stored profile the live hardware licenses."""

    #: Same box. Every tier and every template applies.
    EXACT = "exact"
    #: Same card models, different box. Device model TEMPLATES only.
    MODEL = "model"
    #: Nothing applies. The registry bootstraps from live facts instead.
    NONE = "none"


class ProfileMatch(msgspec.Struct, frozen=True, kw_only=True):
    """The verdict, and the sentence that explains it in a log or a refusal."""

    scope: MatchScope
    profile_id: str
    reason: str
    #: What the profile is allowed to contribute under this scope.
    licenses_tiers: bool = False
    licenses_device_models: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "scope": self.scope.value,
            "profile_id": self.profile_id,
            "reason": self.reason,
            "licenses_tiers": self.licenses_tiers,
            "licenses_device_models": self.licenses_device_models,
        }


def match_profile(
    document: Mapping[str, Any], fingerprint: HardwareFingerprint
) -> ProfileMatch:
    """How much of ``document`` this machine licenses, and why.

    ``document`` is the *decoded JSON*, not a :class:`RigProfile`, because the
    licensing decision has to happen before the tiers are built -- a profile
    that does not match must not have its host and filesystem rows constructed
    at all, not even to be filtered out afterwards.

    A document with no ``hardware`` block matches NOTHING. That is the
    generality rule with no escape hatch: a profile that does not say which
    hardware it was measured on cannot be checked against any, and an
    unverifiable claim is refused rather than trusted. The pre-#434 rig-1
    profile is in exactly that state until its block is filled in, which is
    the intended pressure.
    """
    profile_id = str(document.get("profile_id", "") or "<unnamed>")
    hardware = document.get("hardware")
    if not isinstance(hardware, Mapping):
        return ProfileMatch(
            scope=MatchScope.NONE,
            profile_id=profile_id,
            reason=(
                f"profile {profile_id!r} carries no 'hardware' block, so there "
                "is no way to check which machine its numbers were measured on. "
                "An unverifiable profile is refused, not trusted: that is the "
                "defect #434 names -- one rig's measurements read as a fact "
                "about all of them."
            ),
        )
    version = int(hardware.get("version", FINGERPRINT_VERSION))
    if version != FINGERPRINT_VERSION:
        return ProfileMatch(
            scope=MatchScope.NONE,
            profile_id=profile_id,
            reason=(
                f"profile {profile_id!r} was keyed with fingerprint version "
                f"{version}; this build computes version {FINGERPRINT_VERSION}. "
                "The keys are not comparable, so the profile is not applied."
            ),
        )
    stored_hardware, stored_model, defect = _keys_of(hardware)
    if defect:
        return ProfileMatch(scope=MatchScope.NONE, profile_id=profile_id, reason=defect)
    if stored_hardware and stored_hardware == fingerprint.hardware_key:
        return ProfileMatch(
            scope=MatchScope.EXACT,
            profile_id=profile_id,
            reason=(
                f"hardware key {stored_hardware} matches the enumerated cards "
                f"({fingerprint.label()}); every tier in the profile describes "
                "this machine"
            ),
            licenses_tiers=True,
            licenses_device_models=True,
        )
    if stored_model and stored_model == fingerprint.model_key:
        return ProfileMatch(
            scope=MatchScope.MODEL,
            profile_id=profile_id,
            reason=(
                f"model key {stored_model} matches ({fingerprint.label()}) but "
                f"hardware key does not ({stored_hardware or 'not recorded'} != "
                f"{fingerprint.hardware_key or 'no cards enumerated'}). Card "
                "MODEL templates apply -- a membw figure is a property of the "
                "model. Host, filesystem and remote tiers do not: two machines "
                "with the same cards can have different RAM, disks and wire."
            ),
            licenses_device_models=True,
        )
    return ProfileMatch(
        scope=MatchScope.NONE,
        profile_id=profile_id,
        reason=(
            f"neither key matches for profile {profile_id!r}: it has hardware "
            f"{stored_hardware or 'unset'} / model {stored_model or 'unset'}, "
            f"this machine is {fingerprint.hardware_key or 'cardless'} / "
            f"{fingerprint.model_key or 'cardless'} ({fingerprint.label()}). "
            "Nothing from this profile is applied."
        ),
    )


def _keys_of(hardware: Mapping[str, Any]) -> Tuple[str, str, str]:
    """``(hardware_key, model_key, defect)`` for a profile's hardware block.

    The keys are DERIVED from the block's readable inputs rather than read out
    of it as opaque digests. That is not a convenience: a hand-edited profile
    carrying a stale digest would never match and would fail *silently*, since
    "no profile matched" is the normal path. Deriving them means the file
    states facts a human can check and the code does the hashing, so the two
    can never disagree.

    ``cards`` (a list of ``{uuid, model, total_bytes}``) yields the exact key.
    ``models`` (a list of ``"Nx MODEL:NGiB"``) yields the portable one, and is
    derived from ``cards`` when it is not stated separately.
    """
    rows = hardware.get("cards")
    cards: list = []
    if rows is not None:
        if not isinstance(rows, (list, tuple)):
            return "", "", "hardware.cards is not a list"
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                return "", "", f"hardware.cards[{index}] is not an object"
            uuid = str(row.get("uuid", "") or "")
            model = str(row.get("model", "") or "")
            total = row.get("total_bytes")
            if not uuid or not model or total is None:
                return (
                    "",
                    "",
                    f"hardware.cards[{index}] needs uuid, model and "
                    "total_bytes to key a profile; a partial card row cannot "
                    "identify hardware",
                )
            cards.append(CardFact(uuid=uuid, model=model, total_bytes=int(total)))
    stated_models = hardware.get("models")
    if stated_models is not None and not isinstance(stated_models, (list, tuple)):
        return "", "", "hardware.models is not a list"
    models = (
        tuple(str(m) for m in stated_models)
        if stated_models is not None
        else model_signature(cards)
    )
    if not cards and not models:
        return (
            "",
            "",
            "the hardware block names neither cards nor models, so it makes no "
            "checkable claim about which machine produced these numbers",
        )
    return hardware_key_for(card_signature(cards)), model_key_for(models), ""


def licensed_document(
    document: Mapping[str, Any], match: ProfileMatch
) -> Dict[str, Any]:
    """``document`` reduced to what ``match`` allows it to say.

    The single choke point. A caller cannot accidentally read a tier row out
    of a model-matched profile, because under ``MatchScope.MODEL`` the ``tiers``
    list is not present in what comes back -- it is removed here rather than
    ignored later.
    """
    if match.scope is MatchScope.EXACT:
        return dict(document)
    reduced = {k: v for k, v in document.items() if k not in ("tiers", "device_models")}
    if match.licenses_device_models and "device_models" in document:
        reduced["device_models"] = document["device_models"]
    reduced["tiers"] = []
    return reduced


def hardware_block(fingerprint: HardwareFingerprint) -> Dict[str, Any]:
    """The ``hardware`` block to write when persisting a profile.

    Writes the INPUTS -- the card rows and the model multiset -- not the
    digests, so the reader recomputes what the writer computed and the two can
    never disagree. Kept next to the matcher for exactly that reason.
    """
    return {
        "version": FINGERPRINT_VERSION,
        "label": fingerprint.label(),
        "cards": [
            {"uuid": uuid, "model": model, "total_bytes": total}
            for uuid, model, total in fingerprint.cards
        ],
        "models": list(fingerprint.models),
    }
