# SPDX-License-Identifier: Apache-2.0
"""The ENVIRONMENT a measured card rate was measured in, and how to date it.

WHY THIS MODULE EXISTS.

#584 measured this rig's three cards and compared the result against the rates
a previous shift had persisted (``evidence-631/s50/gate_check.py``):

    5090  membw  1533.8 -> 1661.7 GB/s   (+8.3 %)
    3080  membw   717.4 ->  717.4 GB/s   ( 0.0 %)
    5090  gemm    231.97 ->  203.57 TFLOPS  (-12.2 %)
    3080  gemm     65.57 ->   50.81 TFLOPS  (-22.5 %)

Bandwidth reproduced to the decimal; GEMM did not. Memory bandwidth is set by
the memory clock and is barely power-sensitive; sustained GEMM is a power-bound
quantity, and on 2026-08-05 this rig's power targets were cut (3080s 320 W ->
200 W, 5090 525 W -> 400 W). The rates were not wrong when they were taken.
They stopped describing the rig, and **nothing in the artifact recorded the one
thing that had changed**, so nothing could notice.

A rate is therefore not a number. It is a number PLUS the environment it holds
in, and the environment has to travel with it. The minimum that distinguishes
the failure above is:

  * the card's **enforced power management limit** (NVML, mW) -- the term that
    moved, and the term a GEMM rate is a direct function of;
  * the **driver version** -- a kernel-scheduling or clock-governor change
    lands here and moves rates without anyone touching a power target.

WHAT THIS DELIBERATELY DOES NOT DO.

It does not re-measure, predict, or correct. A rate measured under a different
power limit is not scaled to the current one -- the relationship is not linear,
it is not the same across kernels, and a corrected number is a fabricated
number wearing a measurement's clothes. The verdict is three-valued and the
consumer decides:

  ``fresh``    -- positively verified: same driver, same power limit.
  ``stale``    -- positive evidence of a change. Refuse; re-measure.
  ``unknown``  -- no fingerprint stored (an artifact older than this module),
                  or NVML cannot be read to compare against. Never silently
                  consumed: the consumer says so, loudly, every time.

``unknown`` is not ``fresh``. It is the state the s50 rates were in for the
whole time they were being trusted.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

__all__ = [
    "RATE_ENV_VERSION",
    "POWER_LIMIT_TOLERANCE_MW",
    "RateEnv",
    "FreshnessVerdict",
    "capture_rate_envs",
    "current_envs_by_name",
    "check_rate_freshness",
    "check_card_rate_freshness",
]

#: Bumped when the token grammar changes. A token from another version is not
#: comparable and reads as ``unknown`` rather than as a mismatch -- an
#: unreadable fingerprint is an absent one, not evidence of a changed rig.
RATE_ENV_VERSION = 1

#: Power limits are set in whole watts and NVML reports mW, but a driver may
#: round the enforced limit by a few mW. One watt of slack absorbs that and is
#: three orders of magnitude below the 120 W cut this guard exists to catch.
POWER_LIMIT_TOLERANCE_MW = 1000


@dataclasses.dataclass(frozen=True)
class RateEnv:
    """The environment terms a measured GEMM/bandwidth rate depends on.

    Serialised as a single flat token rather than a nested object so it can sit
    in a ``CardSpec`` field: the card library's JSON is a list of flat dicts,
    and a string keeps the dataclass hashable and cheaply comparable.
    """

    driver_version: Optional[str] = None
    power_limit_mw: Optional[int] = None

    @property
    def token(self) -> str:
        drv = self.driver_version or "?"
        plim = "?" if self.power_limit_mw is None else str(int(self.power_limit_mw))
        return f"v{RATE_ENV_VERSION};drv={drv};plimit_mw={plim}"

    @property
    def power_limit_w(self) -> Optional[float]:
        return None if self.power_limit_mw is None else self.power_limit_mw / 1000.0

    @classmethod
    def parse(cls, token: Optional[str]) -> Optional["RateEnv"]:
        """Parse a token, or None when it is absent or not this version."""
        if not token:
            return None
        parts = str(token).split(";")
        if not parts or parts[0] != f"v{RATE_ENV_VERSION}":
            return None
        fields: Dict[str, str] = {}
        for part in parts[1:]:
            if "=" in part:
                key, _, value = part.partition("=")
                fields[key.strip()] = value.strip()
        raw_plim = fields.get("plimit_mw")
        try:
            plim = None if raw_plim in (None, "", "?") else int(raw_plim)
        except ValueError:
            plim = None
        drv = fields.get("drv")
        return cls(
            driver_version=None if drv in (None, "", "?") else drv,
            power_limit_mw=plim,
        )

    @property
    def complete(self) -> bool:
        """Both terms present. A half-fingerprint cannot date a rate."""
        return bool(self.driver_version) and self.power_limit_mw is not None

    def describe(self) -> str:
        drv = self.driver_version or "unknown driver"
        if self.power_limit_mw is None:
            return f"driver {drv}, unknown power limit"
        return f"driver {drv}, power limit {self.power_limit_mw / 1000.0:.0f} W"

    def matches(self, other: "RateEnv") -> bool:
        if not (self.complete and other.complete):
            return False
        if self.driver_version != other.driver_version:
            return False
        return (
            abs(int(self.power_limit_mw) - int(other.power_limit_mw))
            <= POWER_LIMIT_TOLERANCE_MW
        )


@dataclasses.dataclass(frozen=True)
class FreshnessVerdict:
    """Three-valued, because "cannot tell" is not "fine"."""

    state: str
    reason: str = ""

    @property
    def fresh(self) -> bool:
        return self.state == "fresh"

    @property
    def stale(self) -> bool:
        return self.state == "stale"

    @property
    def unknown(self) -> bool:
        return self.state == "unknown"


def capture_rate_envs() -> Dict[str, RateEnv]:
    """``{uuid: RateEnv}`` for every card NVML can see, or ``{}``.

    Best-effort by design and never raises: a rig whose NVML cannot be read
    still writes a usable rate artifact, it just writes one that later reads as
    ``unknown``. Fabricating a fingerprint would be worse than carrying none --
    a wrong fingerprint reads as ``fresh``.
    """
    out: Dict[str, RateEnv] = {}
    try:
        from sglang.srt.registry.nvml import _decode, nvml_session

        with nvml_session() as pynvml:
            try:
                driver = _decode(pynvml.nvmlSystemGetDriverVersion())
            except Exception:
                driver = None
            count = pynvml.nvmlDeviceGetCount()
            for index in range(count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
                except Exception:
                    continue
                try:
                    # The ENFORCED limit, which is what the card actually runs
                    # under -- not the constraint range, and not the default.
                    plim = int(pynvml.nvmlDeviceGetPowerManagementLimit(handle))
                except Exception:
                    plim = None
                out[uuid] = RateEnv(driver_version=driver, power_limit_mw=plim)
    except Exception:
        return {}
    return out


def _names_by_uuid() -> Dict[str, str]:
    try:
        from sglang.srt.registry.nvml import identity_map

        return {c.uuid: c.name for c in identity_map().cards}
    except Exception:
        return {}


def current_envs_by_name() -> Dict[str, List[RateEnv]]:
    """``{canonical card name: [RateEnv, ...]}`` for the cards present now.

    Keyed by NAME because that is what the consumer has. ``--pp-solve-cut``
    reads the card name from the residency census (deliberately -- an
    index-based lookup walks into the torch-vs-NVML device-order trap), and the
    card library is name-keyed. Two cards of one model may sit at different
    power limits, so the value is a LIST and a stored rate is fresh if it
    matches ANY of them: the rate was taken on one of these cards, and which
    one is a question the name cannot answer.
    """
    from sglang.srt.planner.card_library import _canonical

    envs = capture_rate_envs()
    if not envs:
        return {}
    names = _names_by_uuid()
    out: Dict[str, List[RateEnv]] = {}
    for uuid, env in envs.items():
        name = names.get(uuid)
        if not name:
            continue
        out.setdefault(_canonical(name), []).append(env)
    return out


def check_rate_freshness(
    stored_token: Optional[str], current: Sequence[RateEnv]
) -> FreshnessVerdict:
    """Date a persisted rate against the environment running now."""
    stored = RateEnv.parse(stored_token)
    if stored is None or not stored.complete:
        return FreshnessVerdict(
            "unknown",
            "the rate carries no environment fingerprint, so it cannot be shown "
            "to have been measured under this rig's current power limit and "
            "driver -- it predates rate fingerprinting",
        )
    live = [e for e in current if e is not None and e.complete]
    if not live:
        return FreshnessVerdict(
            "unknown",
            f"the rate was measured at {stored.describe()}, and NVML reports no "
            f"current environment for this card to compare against",
        )
    if any(stored.matches(e) for e in live):
        return FreshnessVerdict("fresh", f"measured at {stored.describe()}")
    now = "; ".join(sorted({e.describe() for e in live}))
    return FreshnessVerdict(
        "stale",
        f"the rate was measured at {stored.describe()}, and this rig now "
        f"reports {now}",
    )


def check_card_rate_freshness(
    name: str,
    stored_token: Optional[str],
    by_name: Optional[Dict[str, List[RateEnv]]] = None,
) -> FreshnessVerdict:
    """:func:`check_rate_freshness` with the current environment looked up by
    card name. ``by_name`` is injectable so a caller checking several stages
    reads NVML once (and so tests stay hermetic)."""
    from sglang.srt.planner.card_library import _canonical

    table = current_envs_by_name() if by_name is None else by_name
    return check_rate_freshness(stored_token, _live_for(name, table))


def _live_for(name: str, table: Dict[str, List[RateEnv]]) -> List[RateEnv]:
    """Live environments for ``name``, matching VARIANTS as well as the name.

    An exact-key lookup cannot date a capacity-disambiguated profile, and this
    was found on metal (R13 act window): `card_rate_pass --run` measured all
    three cards, and `--show` then reported the 5090 FRESH and the 3080
    permanently UNKNOWN -- "NVML reports no current environment for this card"
    -- for cards NVML could see, seconds apart, in one pass.

    The two names are for one card. `#584`'s capacity resolution names this
    rig's profile ``RTX 3080 20GB``, because the driver calls both the 10 GB
    and the 20 GB card ``NVIDIA GeForce RTX 3080`` and the 20 GB cards were
    otherwise resolving onto the 10240 MiB seed entry. The live table here is
    keyed by the raw NVML name, ``RTX 3080``. Equality never holds, and no
    number of re-runs could have fixed it. The 5090 escaped only because
    nothing collides with it.

    The relation is NOT invented here: it is the one
    :meth:`CardLibrary.variants` already states -- a key matches when it
    equals the request, EXTENDS it, or is extended BY it. Held in one place, at
    a TOKEN boundary, so ``RTX 3080`` does not match ``RTX 3090`` and
    ``RTX 308`` does not match ``RTX 3080``.

    Loosening the LOOKUP does not loosen the VERDICT: what is found is still
    compared term by term, so a rate taken at a power limit the rig no longer
    runs comes back STALE through this path rather than FRESH.
    """
    from sglang.srt.planner.card_library import _canonical

    key = _canonical(name)
    out: List[RateEnv] = []
    for entry_key, envs in table.items():
        if (
            entry_key == key
            or entry_key.startswith(key + " ")
            or key.startswith(entry_key + " ")
        ):
            out.extend(envs)
    return out
