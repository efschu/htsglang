"""23.09. (fnFL2x42): Next Flash's D group runs bs=1 -- max_running_requests=1,
graphs captured for bs=1 only. The eager trace showed 48/72-byte broadcasts
in its first decode round, which were read as bs=2 payloads for a batch that
held ONE request.

REFUTED by this probe on x43: ``seq_lens`` has one row at every site below,
and the trace's size field is the bytes moved to BOTH peers (payload x 2 at
R=3), not the payload -- the surviving 1-token health decode shows the same
48/72 bytes. The probe stays as the instrument for a batch's row count.

``SGLANG_SPEC_BATCH_PROBE=N`` logs, per site, the first N looks at the batch
shape where a second row can enter: the extend's hand-off, the overlap
resolve of ``seq_lens`` from ``future_indices``, and the decode entry. The
line names the requests and every per-row tensor's length, so the site that
first shows two rows is the one that made them. Off by default; never raises.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

ENV = "SGLANG_SPEC_BATCH_PROBE"
_LEFT: Dict[str, int] = {}


def _budget() -> int:
    try:
        return max(0, int(os.environ.get(ENV, "0") or 0))
    except ValueError:
        return 0


def _rows(t) -> str:
    if t is None:
        return "None"
    try:
        n = int(t.shape[0]) if getattr(t, "dim", lambda: 1)() > 0 else 1
        vals = t.reshape(-1)[:8].tolist()
        return f"{n}{vals}"
    except Exception as exc:  # noqa: BLE001 - probe must not raise
        return f"?({type(exc).__name__})"


def probe(site: str, batch, **extra) -> None:
    """One line about ``batch``'s row count at ``site``, while budget lasts."""
    try:
        if site not in _LEFT:
            _LEFT[site] = _budget()
        if _LEFT[site] <= 0:
            return
        _LEFT[site] -= 1
        reqs = getattr(batch, "reqs", None)
        spec = getattr(batch, "spec_info", None)
        fields = {
            "mode": getattr(getattr(batch, "forward_mode", None), "name", "?"),
            "reqs": (len(reqs) if reqs is not None else "None"),
            "rids": ",".join(str(getattr(r, "rid", "?"))[:10] for r in (reqs or [])[:4]),
            "req_pool_indices": _rows(getattr(batch, "req_pool_indices", None)),
            "seq_lens": _rows(getattr(batch, "seq_lens", None)),
            "spec": type(spec).__name__ if spec is not None else "None",
            "future_indices": _rows(getattr(spec, "future_indices", None)),
        }
        fields.update({k: (_rows(v) if hasattr(v, "shape") else v) for k, v in extra.items()})
        logger.info(
            "SPEC-BATCH-PROBE site=%s %s", site,
            " ".join(f"{k}={v}" for k, v in fields.items()),
        )
    except Exception:  # noqa: BLE001 - probe must not raise
        pass
