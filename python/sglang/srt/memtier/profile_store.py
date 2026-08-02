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
"""Stored profiles, selected by hardware fingerprint (#407 / directive #434).

A profile is a cache of measurements, not a configuration file. It is written
once the probes have run and read back on the next boot so a rig does not pay
for the same measurements twice -- and it is keyed by the hardware it was taken
on, so it can never be read on a machine it does not describe.

Selection, in one sentence: **every candidate is matched, the best match wins,
and a match of ``NONE`` contributes nothing at all.** There is no
"nearest profile", no "default profile" and no fallback to the bundled one.
What an unmatched machine gets is :mod:`sglang.srt.memtier.bootstrap`'s
live-facts registry, where every cap is ABSENT and names the probe that would
fill it.

Where profiles live, in search order:

1. ``$SGLANG_MEMTIER_PROFILE`` -- one explicit file. Still matched: an operator
   pointing at the wrong file gets a named refusal rather than another rig's
   numbers. ``$SGLANG_MEMTIER_PROFILE_TRUST=1`` is the deliberate override for
   the one legitimate case (a profile just measured on hardware whose keys have
   not been written back yet), and it logs what it is overriding.
2. ``$SGLANG_MEMTIER_PROFILE_DIR``, else ``$XDG_CACHE_HOME/sglang/memtier``,
   else ``~/.cache/sglang/memtier`` -- where :func:`save_profile` writes, one
   file per hardware key.
3. ``memtier/profiles/`` in the package -- profiles that ship with the fork.
   These are matched exactly like any other; shipping one gives it no standing.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import msgspec

from sglang.srt.memtier.fingerprint import (
    HardwareFingerprint,
    MatchScope,
    ProfileMatch,
    hardware_block,
    licensed_document,
    match_profile,
)
from sglang.srt.memtier.profile import (
    BUNDLED_PROFILE_PATH,
    ProfileError,
    RigProfile,
    profile_from_json,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PROFILE_DIR_ENV",
    "PROFILE_PATH_ENV",
    "PROFILE_TRUST_ENV",
    "ProfileSelection",
    "candidate_paths",
    "profile_store_dir",
    "save_profile",
    "select_profile",
]

PROFILE_PATH_ENV = "SGLANG_MEMTIER_PROFILE"
PROFILE_DIR_ENV = "SGLANG_MEMTIER_PROFILE_DIR"
PROFILE_TRUST_ENV = "SGLANG_MEMTIER_PROFILE_TRUST"

_BUNDLED_DIR = BUNDLED_PROFILE_PATH.parent

#: Scope order, best first. Used to pick between candidates, never to relax a
#: verdict: a MODEL match stays a MODEL match even when it is the only one.
_SCOPE_RANK = {MatchScope.EXACT: 0, MatchScope.MODEL: 1, MatchScope.NONE: 2}


class ProfileSelection(msgspec.Struct, frozen=True, kw_only=True):
    """Which profile was applied, how much of it, and what was passed over.

    ``rejected`` is the load-bearing field. A registry that silently found no
    profile and a registry that found four and matched none look identical from
    the outside, and only one of them means the operator has a typo.
    """

    #: ``None`` when nothing matched -- the caller bootstraps from live facts.
    profile: Optional[RigProfile] = None
    match: Optional[ProfileMatch] = None
    path: str = ""
    #: ``(path, ProfileMatch)`` for every candidate that did not win.
    rejected: Tuple[Tuple[str, ProfileMatch], ...] = ()
    #: Candidates that could not be read at all, with the parser's reason.
    unreadable: Tuple[Tuple[str, str], ...] = ()

    @property
    def scope(self) -> MatchScope:
        return self.match.scope if self.match is not None else MatchScope.NONE

    def render(self) -> str:
        """The boot-log paragraph. Says what was used AND what was not."""
        if self.profile is None or self.match is None:
            head = (
                "memtier: no stored profile matches this hardware; every cap "
                "starts ABSENT and names the probe that fills it"
            )
        else:
            head = (
                f"memtier: profile {self.profile.profile_id!r} from {self.path} "
                f"applied at scope {self.scope.value} -- {self.match.reason}"
            )
        lines = [head]
        for path, match in self.rejected:
            lines.append(f"  passed over {path}: {match.reason}")
        for path, reason in self.unreadable:
            lines.append(f"  unreadable {path}: {reason}")
        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        return {
            "profile_id": None if self.profile is None else self.profile.profile_id,
            "path": self.path,
            "scope": self.scope.value,
            "match": None if self.match is None else self.match.to_json(),
            "rejected": [{"path": p, "match": m.to_json()} for p, m in self.rejected],
            "unreadable": [{"path": p, "reason": r} for p, r in self.unreadable],
        }


def profile_store_dir() -> Path:
    """Where :func:`save_profile` writes. Created lazily, never at import."""
    override = os.environ.get(PROFILE_DIR_ENV, "").strip()
    if override:
        return Path(override)
    cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(cache) if cache else Path.home() / ".cache"
    return base / "sglang" / "memtier"


def candidate_paths(*, include_bundled: bool = True) -> List[Path]:
    """Every profile file worth matching, in search order, deduplicated."""
    seen: Dict[Path, None] = {}
    explicit = os.environ.get(PROFILE_PATH_ENV, "").strip()
    if explicit:
        seen[Path(explicit)] = None
    for directory in (profile_store_dir(),) + (
        (_BUNDLED_DIR,) if include_bundled else ()
    ):
        try:
            entries = sorted(directory.glob("*.json"))
        except OSError:
            continue
        for entry in entries:
            seen.setdefault(entry, None)
    return list(seen)


def _read_document(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def select_profile(
    fingerprint: HardwareFingerprint,
    *,
    paths: Optional[Sequence[Path]] = None,
    trust_explicit: Optional[bool] = None,
) -> ProfileSelection:
    """Match every candidate against ``fingerprint`` and apply the best one.

    ``paths`` is injectable so a hermetic test can present a synthetic set of
    foreign profiles without an environment variable and without touching the
    user's cache directory.
    """
    explicit = os.environ.get(PROFILE_PATH_ENV, "").strip()
    if trust_explicit is None:
        trust_explicit = os.environ.get(PROFILE_TRUST_ENV, "").strip() in (
            "1",
            "true",
            "yes",
        )
    candidates = list(paths) if paths is not None else candidate_paths()
    best: Optional[Tuple[int, Path, Mapping[str, Any], ProfileMatch]] = None
    rejected: List[Tuple[str, ProfileMatch]] = []
    unreadable: List[Tuple[str, str]] = []
    for path in candidates:
        try:
            document = _read_document(Path(path))
        except (OSError, json.JSONDecodeError) as exc:
            unreadable.append((str(path), f"{type(exc).__name__}: {exc}"))
            continue
        if not isinstance(document, Mapping):
            unreadable.append((str(path), "not a JSON object"))
            continue
        match = match_profile(document, fingerprint)
        if (
            trust_explicit
            and explicit
            and Path(explicit) == Path(path)
            and match.scope is MatchScope.NONE
        ):
            match = ProfileMatch(
                scope=MatchScope.EXACT,
                profile_id=match.profile_id,
                reason=(
                    f"{PROFILE_TRUST_ENV} is set, so the explicit profile is "
                    f"applied in full despite the fingerprint verdict: "
                    f"{match.reason}"
                ),
                licenses_tiers=True,
                licenses_device_models=True,
            )
            logger.warning(
                "memtier: %s overrides the hardware match for %s -- the "
                "profile's numbers are being read on hardware that did not "
                "produce them",
                PROFILE_TRUST_ENV,
                path,
            )
        rank = _SCOPE_RANK[match.scope]
        if match.scope is MatchScope.NONE:
            rejected.append((str(path), match))
            continue
        if best is None or rank < best[0]:
            if best is not None:
                rejected.append((str(best[1]), best[3]))
            best = (rank, Path(path), document, match)
        else:
            rejected.append((str(path), match))
    if best is None:
        return ProfileSelection(rejected=tuple(rejected), unreadable=tuple(unreadable))
    _, path, document, match = best
    try:
        profile = profile_from_json(licensed_document(document, match), path=str(path))
    except ProfileError as exc:
        return ProfileSelection(
            rejected=tuple(rejected),
            unreadable=tuple(unreadable) + ((str(path), str(exc)),),
        )
    return ProfileSelection(
        profile=profile,
        match=match,
        path=str(path),
        rejected=tuple(rejected),
        unreadable=tuple(unreadable),
    )


def save_profile(
    document: Mapping[str, Any],
    fingerprint: HardwareFingerprint,
    *,
    directory: Optional[Path] = None,
) -> Path:
    """Persist ``document`` under this machine's hardware key.

    The keys are written from ``fingerprint`` rather than taken from the
    document: a profile is keyed by the hardware that PRODUCED it, and letting
    a caller supply the key would make the one field that prevents cross-rig
    leakage the one field a caller can get wrong.

    Refuses when the machine has no enumerated card. A profile with no exact
    key can never match at ``EXACT`` scope, so writing one would create a file
    that is guaranteed to be passed over -- a silent no-op dressed as a save.
    """
    if not fingerprint.hardware_key:
        raise ProfileError(
            "cannot store a memtier profile for a machine with no enumerated "
            "card: there is no hardware key to file it under, and a keyless "
            "profile is refused by the matcher on every future boot"
        )
    target_dir = Path(directory) if directory is not None else profile_store_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(document)
    payload["hardware"] = hardware_block(fingerprint)
    target = target_dir / f"{fingerprint.hardware_key}.json"
    handle = tempfile.NamedTemporaryFile(
        "w", dir=str(target_dir), suffix=".tmp", delete=False, encoding="utf-8"
    )
    try:
        with handle as out:
            json.dump(payload, out, indent=2, sort_keys=True)
        os.replace(handle.name, target)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return target
