"""NF #162 (23.10.): the device VA map of a rank at its death, so the fault address of an illegal memory access can be
given an owner.

rc31b (21.10. 23:58:39Z, deskq/done/nf-rc31b-ima-1022.md + nf-cudacore-1023.md): D-TP1 died on an SM read at VA
0x5_4a4e0000 with FAULT_PDE (the whole 2 MiB region unmapped). Two kinds of owner give exactly that and the fault data
cannot tell them apart: a TMS-managed tensor whose VA stays reserved while a span is unmapped (the expert bank rows of
the D-TRANSIENT-LEND / LIVE-SPANS moves, the KV stage tensors) and a torch segment that empty_cache gave back with
cudaFree. The boot log printed the base VA of none of them.

``write_va_map`` writes, on the rank's death path, ONE JSON file next to the CUDA coredump (the dump directory on the
evidence volume) and logs one line naming it:

* every page-controller tensor of this rank (d_seat_vram ``SeatVram``: bank rows, mamba slots, KV stage tensors) with
  base VA, reserved size and the spans mapped at this moment (``spans_by_ptr``, offsets relative to the base);
* the torch caching allocator's segments (``torch.cuda.memory_snapshot``: address, size, stream) -- host bookkeeping,
  no device call;
* the address the kernel names can then be looked up offline: ``va_lookup(map, addr)``.

Host-only, bounded (a daemon thread, 5 s), never raises -- it runs between the logged traceback and the SIGQUIT.
"""
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

CTL_ATTR = "_pdflip_d_seat_vram"  # d_seat_vram.CTL_ATTR (not imported: the death path may not import heavy modules)


def _geom_name(geom: Any) -> str:
    name = getattr(geom, "name", None)
    if name is None:
        name = getattr(getattr(geom, "geom", None), "name", None)
    return str(name if name is not None else type(geom).__name__)


def _geom_size(geom: Any) -> Optional[int]:
    for obj in (geom, getattr(geom, "geom", None)):
        if obj is None:
            continue
        for attr in ("alloc_bytes", "size_bytes", "size"):
            v = getattr(obj, attr, None)
            if isinstance(v, int) and v > 0:
                return int(v)
    return None


def managed_tensors(sched: Any) -> List[Dict[str, Any]]:
    ctl = getattr(sched, CTL_ATTR, None) if sched is not None else None
    if not ctl:
        return []
    spans_by_ptr = dict(getattr(ctl, "spans_by_ptr", {}) or {})
    out = []
    for kind in ("row_tensors", "slot_tensors", "kv_tensors"):
        for m in list(getattr(ctl, kind, []) or []):
            ptr = int(getattr(m, "ptr", 0) or 0)
            geom = getattr(m, "geom", None)
            spans = spans_by_ptr.get(ptr)
            out.append({
                "kind": kind,
                "name": _geom_name(geom),
                "base": ptr,
                "size": _geom_size(geom),
                "rows_boot": getattr(geom, "rows_boot", None),
                "rows_max": getattr(geom, "rows_max", None),
                "row_bytes": getattr(geom, "row_bytes", None),
                "mapped": [list(s) for s in spans] if spans is not None else None,
            })
    return out


def torch_segments() -> List[Dict[str, Any]]:
    import torch  # noqa: PLC0415 - already imported in a rank

    segs = []
    for seg in torch.cuda.memory_snapshot():
        segs.append({
            "base": int(seg.get("address", 0)),
            "size": int(seg.get("total_size", 0)),
            "allocated": int(seg.get("allocated_size", 0)),
            "stream": int(seg.get("stream", 0) or 0),
            "pool": str(seg.get("segment_pool_id", "")),
            "device": int(seg.get("device", 0) or 0),
        })
    return segs


def va_lookup(va_map: Dict[str, Any], addr: int) -> List[str]:
    """Who owns ``addr`` in a written map: one line per hit, '' list when nobody does (= neither a managed tensor nor
    a live torch segment -- a freed segment or foreign VA)."""
    hits = []
    for t in va_map.get("managed", []):
        base, size = t["base"], t.get("size") or 0
        if base <= addr < base + size:
            off = addr - base
            mapped = t.get("mapped")
            state = ("mapped" if any(lo <= off < hi for lo, hi in mapped) else "UNMAPPED") if mapped is not None \
                else "spans unknown"
            row = (" row=%d" % (off // t["row_bytes"])) if t.get("row_bytes") else ""
            hits.append("managed %s %s offset=0x%x%s %s" % (t["kind"], t["name"], off, row, state))
    for s in va_map.get("segments", []):
        if s["base"] <= addr < s["base"] + s["size"]:
            hits.append("torch segment base=0x%x size=%d stream=%d pool=%s" % (s["base"], s["size"], s["stream"],
                                                                              s["pool"]))
    return hits


def _collect(sched: Any, out: Dict[str, Any]) -> None:
    try:
        out["managed"] = managed_tensors(sched)
    except Exception as exc:  # noqa: BLE001
        out["managed_error"] = "%s: %s" % (type(exc).__name__, exc)
    try:
        out["segments"] = torch_segments()
    except Exception as exc:  # noqa: BLE001
        out["segments_error"] = "%s: %s" % (type(exc).__name__, exc)


def write_va_map(sched: Any, label: str, reason: str, *, dump_dir: Optional[str] = None,
                 log: Optional[logging.Logger] = None, timeout_s: float = 5.0) -> Optional[str]:
    log = log or _logger
    try:
        if dump_dir is None:
            from flliper.srt.debug_utils import cuda_coredump as _cc  # noqa: PLC0415

            if not _cc.is_enabled():
                return None
            dump_dir = os.path.dirname(os.environ.get("CUDA_COREDUMP_FILE", "") or "") or _cc.get_dump_dir()
        data: Dict[str, Any] = {"label": label, "pid": os.getpid(), "reason": reason, "t_unix": time.time()}
        th = threading.Thread(target=_collect, args=(sched, data), name="va-witness", daemon=True)
        th.start()
        th.join(timeout_s)
        if th.is_alive():
            data["timeout_s"] = timeout_s
        os.makedirs(dump_dir, exist_ok=True)
        tag = "".join(c if c.isalnum() else "_" for c in label)
        path = os.path.join(dump_dir, "va_map_%s_%d_%d.json" % (tag, os.getpid(), int(data["t_unix"])))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        managed = data.get("managed") or []
        segs = data.get("segments") or []

        def _span(items):
            if not items:
                return "-"
            lo = min(i["base"] for i in items)
            hi = max(i["base"] + (i.get("size") or 0) for i in items)
            return "[0x%x,0x%x)" % (lo, hi)

        log.error("VA-MAP rank=%s pid=%d reason=%s file=%s managed=%d %s segments=%d %s%s -- look the fault VA up "
                  "with va_witness.va_lookup (the Xid line or cuda-gdb names it)", label, os.getpid(), reason, path,
                  len(managed), _span(managed), len(segs), _span(segs),
                  " TIMEOUT" if "timeout_s" in data else "")
        return path
    except Exception as exc:  # noqa: BLE001 - a witness never masks the death
        log.error("VA-MAP rank=%s not written: %s: %s", label, type(exc).__name__, exc)
        return None
