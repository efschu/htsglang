"""TSDB (user order 01.10., spec docs/TSDB-DELTA-27B-1001.md, part 2): rank
gauges and the decode-round histogram on the group servers' existing
``/metrics`` (``--enable-metrics``, prometheus multiprocess). No new env, no
new port.

* ``weg2_rank_vram_used_bytes`` / ``weg2_rank_kv_usage_ratio`` /
  ``weg2_rank_running`` {tp_rank, pp_rank}, ``multiprocess_mode="livemax"`` --
  set from the rankstats timer thread (pdflip/rankstats.py), never the round path;
* ``weg2_decode_round_seconds{bs}`` -- one observation per decode round, at the
  round's flush (DecodeRoundLog._emit, a boundary, never between two forwards);
* ``weg2_decode_tokens_total`` -- the generated-token delta per rankstats tick.

Armed only where the server has metrics on (``PROMETHEUS_MULTIPROC_DIR`` set,
which ``--enable-metrics`` does before the schedulers start). Every failure is
counted in ``ERRORS`` and switches nothing else; nothing here ever raises.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

_ROUND_BUCKETS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0)

ERRORS: Dict[str, int] = {}
_lock = threading.Lock()
_M: Optional[Dict[str, Any]] = None
_DISABLED = False
_last_gen_tokens: Optional[int] = None


def _err(where: str) -> None:
    ERRORS[where] = ERRORS.get(where, 0) + 1


def armed() -> bool:
    return bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))


def _metrics() -> Optional[Dict[str, Any]]:
    global _M, _DISABLED
    if _M is not None or _DISABLED:
        return _M
    with _lock:
        if _M is not None or _DISABLED:
            return _M
        if not armed():
            _DISABLED = True
            return None
        try:
            from prometheus_client import Counter, Gauge, Histogram

            lbl = ["tp_rank", "pp_rank"]
            _M = {
                "vram": Gauge("weg2_rank_vram_used_bytes", "CUDA memory reserved by this rank",
                              lbl, multiprocess_mode="livemax"),
                "kv": Gauge("weg2_rank_kv_usage_ratio", "Full-token KV usage of this rank",
                            lbl, multiprocess_mode="livemax"),
                "running": Gauge("weg2_rank_running", "Running requests on this rank",
                                 lbl, multiprocess_mode="livemax"),
                "round": Histogram("weg2_decode_round_seconds", "Decode round wall (GPU) time",
                                   ["bs"], buckets=_ROUND_BUCKETS),
                "tokens": Counter("weg2_decode_tokens", "Generated tokens (rankstats delta)"),
            }
        except Exception:  # noqa: BLE001 - no client / a registry clash: off, counted
            _err("init")
            _DISABLED = True
            _M = None
        return _M


def observe_decode_round(bs: int, seconds: float) -> None:
    m = _metrics()
    if m is None:
        return
    try:
        m["round"].labels(bs=str(int(bs))).observe(max(0.0, float(seconds)))
    except Exception:  # noqa: BLE001
        _err("round")


def _vram_reserved() -> Optional[int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_reserved())
    except Exception:  # noqa: BLE001
        _err("vram")
        return None


def on_rankstats(rec: Dict[str, Any], tp_rank: int, pp_rank: int,
                 vram_bytes: Optional[int] = None) -> None:
    """From the rankstats timer thread, after a record was built."""
    global _last_gen_tokens
    m = _metrics()
    if m is None:
        return
    try:
        lv = {"tp_rank": str(int(tp_rank)), "pp_rank": str(int(pp_rank))}
        sched = rec.get("sched") or {}
        usage = sched.get("full_token_usage")
        if usage is not None:
            m["kv"].labels(**lv).set(float(usage))
        running = sched.get("running")
        if running is not None:
            m["running"].labels(**lv).set(int(running))
        v = vram_bytes if vram_bytes is not None else _vram_reserved()
        if v is not None:
            m["vram"].labels(**lv).set(int(v))
        gen = int(((rec.get("tokens") or {}).get("decode_total")) or 0)
        if _last_gen_tokens is not None and gen > _last_gen_tokens:
            m["tokens"].inc(gen - _last_gen_tokens)
        _last_gen_tokens = gen
    except Exception:  # noqa: BLE001
        _err("rankstats")


def _reset_for_tests() -> None:
    global _M, _DISABLED, _last_gen_tokens
    _M = None
    _DISABLED = False
    _last_gen_tokens = None
    ERRORS.clear()
