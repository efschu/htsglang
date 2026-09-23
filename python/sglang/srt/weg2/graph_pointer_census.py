"""#105 (23.09.): the 27B rule for the NF flip, as a check instead of a hope.

The 27B flip survives a wake because every address a captured D graph reads is,
after the wake, still mapped and holds content that was exchanged, restored
from a backup or re-initialized in place. For Next Flash nothing established
that rule; the wake rebuilt pool state piece by piece ("rearm"), and every
missed piece would surface as an illegal address in the first replay -- with
no name attached.

This module names them. At capture it walks every node of a captured graph
and keeps each pointer-shaped kernel parameter word that the driver resolves
to a live allocation, plus the source and destination of every memcpy and the
destination of every memset. After a wake, before the first replay,
``verify`` resolves every kept address again: an address the driver no longer
knows, or whose allocation range changed, is a finding -- by kernel name,
parameter index and address.

v1 checks MAPPING (address still backed, same range). Content -- whether a
mapped range was exchanged, restored or re-set -- is the second half of #105
and classifies the same records by memory-saver tag.

Off unless ``SGLANG_WEG2_GRAPH_PTR_CENSUS=1``; never raises.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import msgspec

logger = logging.getLogger(__name__)

ENV_ENABLE = "SGLANG_WEG2_GRAPH_PTR_CENSUS"

#: Canonical user-space and GPU VAs on this platform sit below 2^49; anything
#: outside [PTR_MIN, PTR_MAX) cannot be an address and is not asked about.
PTR_MIN = 1 << 16
PTR_MAX = 1 << 49


class PtrRecord(msgspec.Struct, frozen=True):
    """One address a captured node reads or writes, as seen at capture."""

    graph: str
    node: int
    kind: str  # "kernel", "memcpy.src", "memcpy.dst", "memset.dst"
    name: str  # kernel name, or the node kind for copies
    param: int  # kernel parameter index (-1 for copies)
    word: int  # byte offset of the word inside the parameter
    addr: int
    range_start: int
    range_size: int


class Finding(msgspec.Struct, frozen=True):
    record: PtrRecord
    verdict: str  # "unmapped" or "range-changed"
    now_start: int = 0
    now_size: int = 0


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0") in ("1", "true", "True")


def pointer_words(raw: bytes) -> List[Tuple[int, int]]:
    """``(byte offset, value)`` of every 8-byte-aligned word that could be an
    address. A struct passed by value (the BAR1 transport's argument block)
    carries its pointers inside; scanning words finds them without knowing the
    struct."""
    out = []
    for off in range(0, len(raw) - 7, 8):
        val = int.from_bytes(raw[off:off + 8], "little")
        if PTR_MIN <= val < PTR_MAX:
            out.append((off, val))
    return out


def _driver():
    from cuda.bindings import driver as cu

    return cu


def _ok(res) -> Tuple[bool, tuple]:
    err, rest = res[0], res[1:]
    return int(err) == 0, rest


def resolve_range(addr: int) -> Optional[Tuple[int, int]]:
    """``(start, size)`` of the allocation the driver maps at ``addr``, or
    ``None`` when it knows none (unmapped, freed, or not an address)."""
    try:
        cu = _driver()
        attr = cu.CUpointer_attribute
        ok, start = _ok(cu.cuPointerGetAttribute(
            attr.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, addr))
        if not ok:
            return None
        ok, size = _ok(cu.cuPointerGetAttribute(
            attr.CU_POINTER_ATTRIBUTE_RANGE_SIZE, addr))
        if not ok:
            return None
        return int(start[0]), int(size[0])
    except Exception:  # noqa: BLE001 - instrument must not raise
        return None


def _kernel_name(cu, params) -> str:
    for handle, getter in ((getattr(params, "kern", None), cu.cuKernelGetName),
                           (getattr(params, "func", None), cu.cuFuncGetName)):
        if handle is None or int(handle) == 0:
            continue
        ok, rest = _ok(getter(handle))
        if ok and rest and rest[0]:
            name = rest[0]
            return name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)
    return "?"


def _kernel_records(cu, graph: str, idx: int, node) -> Iterable[PtrRecord]:
    ok, rest = _ok(cu.cuGraphKernelNodeGetParams(node))
    if not ok:
        return []
    params = rest[0]
    name = _kernel_name(cu, params)
    func = params.func
    base = int(params.kernelParams) if params.kernelParams else 0
    if not base or func is None or int(func) == 0:
        return []
    out = []
    pidx = 0
    while pidx < 4096:
        ok, rest = _ok(cu.cuFuncGetParamInfo(func, pidx))
        if not ok:
            break
        size = int(rest[1])
        argp = ctypes.c_void_p.from_address(base + 8 * pidx).value
        if argp and size:
            for off, val in pointer_words(ctypes.string_at(argp, size)):
                rng = resolve_range(val)
                if rng is not None:
                    out.append(PtrRecord(graph, idx, "kernel", name, pidx, off,
                                         val, rng[0], rng[1]))
        pidx += 1
    return out


def _copy_records(cu, graph: str, idx: int, node, ntype) -> Iterable[PtrRecord]:
    out = []
    kinds = cu.CUgraphNodeType
    if ntype == kinds.CU_GRAPH_NODE_TYPE_MEMCPY:
        ok, rest = _ok(cu.cuGraphMemcpyNodeGetParams(node))
        if not ok:
            return out
        p = rest[0]
        for kind, dev, host in (("memcpy.src", p.srcDevice, p.srcHost),
                                ("memcpy.dst", p.dstDevice, p.dstHost)):
            for val in (int(dev) if dev else 0, int(host) if host else 0):
                rng = resolve_range(val) if PTR_MIN <= val < PTR_MAX else None
                if rng is not None:
                    out.append(PtrRecord(graph, idx, kind, kind, -1, 0, val,
                                         rng[0], rng[1]))
    elif ntype == kinds.CU_GRAPH_NODE_TYPE_MEMSET:
        ok, rest = _ok(cu.cuGraphMemsetNodeGetParams(node))
        if ok:
            val = int(rest[0].dst)
            rng = resolve_range(val) if PTR_MIN <= val < PTR_MAX else None
            if rng is not None:
                out.append(PtrRecord(graph, idx, "memset.dst", "memset.dst",
                                     -1, 0, val, rng[0], rng[1]))
    return out


def census_graph(raw_graph: int, graph: str) -> List[PtrRecord]:
    """Every live address the nodes of one captured graph carry."""
    try:
        cu = _driver()
        ok, rest = _ok(cu.cuGraphGetNodes(raw_graph, 0))
        if not ok:
            return []
        count = int(rest[1])
        ok, rest = _ok(cu.cuGraphGetNodes(raw_graph, count))
        if not ok:
            return []
        records: List[PtrRecord] = []
        kinds = cu.CUgraphNodeType
        for idx, node in enumerate(rest[0]):
            ok, t = _ok(cu.cuGraphNodeGetType(node))
            if not ok:
                continue
            ntype = t[0]
            if ntype == kinds.CU_GRAPH_NODE_TYPE_KERNEL:
                records.extend(_kernel_records(cu, graph, idx, node))
            elif ntype in (kinds.CU_GRAPH_NODE_TYPE_MEMCPY,
                           kinds.CU_GRAPH_NODE_TYPE_MEMSET):
                records.extend(_copy_records(cu, graph, idx, node, ntype))
            elif ntype == kinds.CU_GRAPH_NODE_TYPE_GRAPH:
                ok, child = _ok(cu.cuGraphChildGraphNodeGetGraph(node))
                if ok:
                    records.extend(census_graph(child[0], f"{graph}/child{idx}"))
        return records
    except Exception as exc:  # noqa: BLE001 - instrument must not raise
        logger.warning("WEG2-GRAPH-PTR census of %s unavailable: %s: %s",
                       graph, type(exc).__name__, exc)
        return []


def verify(records: Iterable[PtrRecord]) -> List[Finding]:
    """Resolve every recorded address again; name what the wake broke."""
    findings = []
    for rec in records:
        rng = resolve_range(rec.addr)
        if rng is None:
            findings.append(Finding(rec, "unmapped"))
        elif rng != (rec.range_start, rec.range_size):
            findings.append(Finding(rec, "range-changed", rng[0], rng[1]))
    return findings


def summarize(findings: List[Finding], limit: int = 12) -> List[str]:
    """One line per (kernel, verdict), counted, the first address named."""
    groups: Dict[Tuple[str, str, str], List[Finding]] = {}
    for f in findings:
        groups.setdefault((f.record.graph, f.record.name, f.verdict), []).append(f)
    lines = []
    for (graph, name, verdict), fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        r = fs[0].record
        lines.append(
            f"graph={graph} {verdict} n={len(fs)} kernel={name} "
            f"node={r.node} param={r.param}+{r.word} addr=0x{r.addr:x} "
            f"was=[0x{r.range_start:x}+{r.range_size}]"
        )
    return lines[:limit]


#: graph label -> records, filled at capture, read at every wake.
_REGISTRY: Dict[str, List[PtrRecord]] = {}


def record_capture(raw_graph: int, graph: str) -> None:
    if not enabled():
        return
    records = census_graph(raw_graph, graph)
    _REGISTRY[graph] = records
    logger.info("WEG2-GRAPH-PTR captured graph=%s records=%d ranges=%d",
                graph, len(records), len({(r.range_start, r.range_size) for r in records}))


def verify_registry(where: str) -> int:
    """Check every captured graph; log findings by name. Returns their count."""
    if not enabled() or not _REGISTRY:
        return 0
    total = 0
    for graph, records in _REGISTRY.items():
        findings = verify(records)
        total += len(findings)
        if not findings:
            logger.info("WEG2-GRAPH-PTR %s graph=%s ok records=%d", where, graph, len(records))
            continue
        for line in summarize(findings):
            logger.error("WEG2-GRAPH-PTR %s FINDING %s", where, line)
    return total
