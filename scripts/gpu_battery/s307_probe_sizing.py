#!/usr/bin/env python3
"""Pure sizing helpers for the #307 arm B raise probe (s307_raise_probe.py).

Split out of the probe so the pressure-load arithmetic can be tested without
a server, a thread, or a socket -- see the TestArmBPressureSizing tests in
test/registered/unit/model_executor/test_ceiling_mamba_fit.py.

Why this exists: the pressure phase must occupy enough of the mamba pool to
cross --admission-throttle-high (default 0.30) with headroom. A FIXED request
count calibrated against a predicted pool size stops proving anything the
moment the fitted pool comes out a different size. The #307-Beleg card run
(2026-07-31, see docs/dev/INTEGRATION_R3_VALIDATION.md) hit exactly this: 24
requests were sized against the ~70-slot pool the fit's arithmetic predicted,
but the pool that actually got fitted was 90-94 slots, and 0.30 * 90 = ~27
occupied slots was never reached by 24 concurrent requests. The fix is to
read the pool the server itself reports and size the load as a FRACTION of
it, not as an absolute number.
"""

import math

PRESSURE_FRACTION = 0.4


def pool_from_info(info):
    """The fitted mamba pool size (``max_mamba_cache_size``), wherever the
    /get_server_info payload puts it -- same lookup shape as the admission
    limiter in s307_ceiling_fit.py's ``_limiter``."""
    if not isinstance(info, dict):
        return None
    for st in info.get("internal_states") or []:
        if isinstance(st, dict) and isinstance(st.get("max_mamba_cache_size"), int):
            return st["max_mamba_cache_size"]
    size = info.get("max_mamba_cache_size")
    return size if isinstance(size, int) else None


def default_concurrency(pool, fraction=PRESSURE_FRACTION, fallback=24):
    """ceil(fraction * pool), comfortably above --admission-throttle-high.

    Falls back to the historical fixed count only when no pool was reported
    at all (e.g. queried before any mamba-fit boot, so the field is absent).
    That fallback is not a calibration -- it is just "do not crash" when the
    server has nothing to size against.
    """
    if not isinstance(pool, int) or pool <= 0:
        return fallback
    return max(1, math.ceil(pool * fraction))
