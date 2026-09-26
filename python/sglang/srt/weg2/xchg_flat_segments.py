"""Weg2 #1273 G1 -- the FLAT-SEGMENT container, generically (GGUF is the first user).

WHAT IT IS.  A quantization method may store a FUSED (merged column-parallel) module
as ONE 1-D byte buffer per rank: the loader's shards sit back to back, each at an
ALIGNED byte offset (the gap between two is pad and never read), each a
``[rows x row_bytes]`` block with its OWN row width -- one fused module may mix
quant types (a Q6_K shard beside a Q8_0 one) and dtypes (a dense shard cast to the
params dtype beside packed ones).  ``layers/quantization/gguf.py
_create_flat_weight_param`` is the first such container.  A segment may itself FUSE
several components: a checkpoint tensor that packs q|k|v arrives as ONE shard, the
loader splits it per component and re-fuses this rank's parts, so a TP rank's
segment rows are ``[q_r | k_r | v_r]`` -- three row ranges, not one.

WHY IT HAS ITS OWN GEOMETRY.  The storage is ``[total_bytes]``: the manifest join
sees one row per rank and byte totals that do not add up (every rank pads for
itself) -- or, worse, that add up by coincidence and classify as a plain column
cut, which copies rank 0's ``[seg0_r0 | seg1_r0 | ...]`` over the whole's first
bytes with no refusal anywhere.  No outer arithmetic can see the segments, so the
LOADER declares them where it places the bytes (:class:`FlatTable`) and this module
only joins declarations and cuts byte ranges out of them.  Nothing here reads a
shape, a quant type or a model constant.

THE LAWS, and nothing else:

* a component is split by WHOLE ROWS -- a rank holds rows ``[row_start, row_start +
  rows)`` of the component's checkpoint rows (the quantization packs along the
  input dim, so a column-parallel cut never splits a row) -- or it is held WHOLE on
  every rank (``REPLICATED``, e.g. k/v under replicated KV);
* a segment's row width is the same on every rank (the input dim is not cut here);
* bytes outside every segment are PAD: zero-filled on a destination, never a source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: The resolved axis of one component in a join.
COMPONENT_ROWS = "rows"
COMPONENT_REPLICATED = "replicated"


def _refuse(text: str) -> Exception:
    from sglang.srt.weg2.weight_exchange import Weg2XchgPlanDisagree

    return Weg2XchgPlanDisagree(f"W68 Weg2XchgPlanDisagree: {text}")


def _missing(text: str) -> Exception:
    from sglang.srt.weg2.weight_exchange import Weg2XchgSourceMissing

    return Weg2XchgSourceMissing(f"W74 Weg2XchgSourceMissing: {text}")


def shard_key(shard_id) -> str:
    """The loader's shard id as a key both groups spell the same way: ``0``,
    ``"q"``, and a fused multi-slot id ``(0, 1, 2)`` as ``"0+1+2"``."""
    if isinstance(shard_id, (tuple, list)):
        return "+".join(str(s) for s in shard_id)
    return str(shard_id)


@dataclass(frozen=True)
class FlatComponent:
    """Rows ``[row_start, row_start + rows)`` of a component whose checkpoint
    extent is ``full_rows`` -- what THIS rank's loader copied, in the order the
    rows sit inside the segment."""

    key: str
    full_rows: int
    row_start: int
    rows: int


@dataclass(frozen=True)
class FlatSegment:
    """One shard of the container: ``components`` stacked row-wise at byte
    ``offset`` of this rank's buffer, every row ``row_bytes`` wide."""

    key: str
    offset: int
    row_bytes: int
    components: Tuple[FlatComponent, ...]

    @property
    def rows(self) -> int:
        return sum(int(c.rows) for c in self.components)

    @property
    def nbytes(self) -> int:
        return self.rows * int(self.row_bytes)

    def component_offset(self, key: str) -> int:
        """Byte offset of component ``key`` in the buffer (its segment's offset
        plus the rows stacked before it)."""
        rows = 0
        for c in self.components:
            if c.key == key:
                return int(self.offset) + rows * int(self.row_bytes)
            rows += int(c.rows)
        raise KeyError(key)


@dataclass(frozen=True)
class FlatTable:
    """One rank's declared container: its whole buffer (``nbytes``, pad
    included) and its segments in buffer order."""

    nbytes: int
    segments: Tuple[FlatSegment, ...]

    def by_key(self) -> Dict[str, FlatSegment]:
        return {s.key: s for s in self.segments}

    def validate(self, name: str) -> None:
        """Refuse a declaration that could only be planned by guessing."""
        if int(self.nbytes) <= 0 or not self.segments:
            raise _refuse(
                f"{name}: a flat container of {self.nbytes} bytes with "
                f"{len(self.segments)} segments describes nothing."
            )
        keys = [s.key for s in self.segments]
        if len(set(keys)) != len(keys):
            raise _refuse(
                f"{name}: segment keys {keys} repeat -- two shards "
                f"would claim one identity."
            )
        end = 0
        for s in sorted(self.segments, key=lambda s: int(s.offset)):
            if int(s.row_bytes) <= 0 or int(s.offset) < end:
                raise _refuse(
                    f"{name}: segment {s.key!r} at byte {s.offset} (row width "
                    f"{s.row_bytes}) overlaps the previous one ending at {end} "
                    f"or has no row width."
                )
            ckeys = [c.key for c in s.components]
            if not ckeys or len(set(ckeys)) != len(ckeys):
                raise _refuse(
                    f"{name}: segment {s.key!r} declares components " f"{ckeys}."
                )
            for c in s.components:
                if (
                    int(c.rows) < 0
                    or int(c.row_start) < 0
                    or int(c.row_start) + int(c.rows) > int(c.full_rows)
                ):
                    raise _refuse(
                        f"{name}: segment {s.key!r} component {c.key!r} holds rows "
                        f"[{c.row_start}, {int(c.row_start) + int(c.rows)}) of "
                        f"{c.full_rows} -- outside its own checkpoint extent."
                    )
            end = int(s.offset) + s.nbytes
        if end > int(self.nbytes):
            raise _refuse(
                f"{name}: the segments end at byte {end}, past the "
                f"buffer's {self.nbytes}."
            )

    def pad_ranges(self) -> List[Tuple[int, int]]:
        """``[(offset, nbytes)]`` of every byte no segment holds."""
        out, cursor = [], 0
        for s in sorted(self.segments, key=lambda s: int(s.offset)):
            if int(s.offset) > cursor:
                out.append((cursor, int(s.offset) - cursor))
            cursor = max(cursor, int(s.offset) + s.nbytes)
        if cursor < int(self.nbytes):
            out.append((cursor, int(self.nbytes) - cursor))
        return out

    def as_json(self) -> Dict[str, object]:
        return {
            "nbytes": int(self.nbytes),
            "segments": [
                {
                    "key": s.key,
                    "offset": int(s.offset),
                    "row_bytes": int(s.row_bytes),
                    "components": [
                        [c.key, int(c.full_rows), int(c.row_start), int(c.rows)]
                        for c in s.components
                    ],
                }
                for s in self.segments
            ],
        }

    @classmethod
    def from_json(cls, raw) -> FlatTable:
        if isinstance(raw, str):
            raw = json.loads(raw)
        return cls(
            nbytes=int(raw["nbytes"]),
            segments=tuple(
                FlatSegment(
                    key=str(s["key"]),
                    offset=int(s["offset"]),
                    row_bytes=int(s["row_bytes"]),
                    components=tuple(
                        FlatComponent(str(k), int(f), int(a), int(n))
                        for k, f, a, n in s["components"]
                    ),
                )
                for s in raw["segments"]
            ),
        )


@dataclass(frozen=True)
class FlatJoin:
    """Both groups' declarations of one container, checked against each other:
    ``whole`` is the holder that has every component entire (the PP form),
    ``ranks[r]`` TP rank ``r``'s table, ``axes[(segment, component)]`` how that
    component is cut across the TP ranks."""

    whole: FlatTable
    ranks: Tuple[FlatTable, ...]
    axes: Tuple[Tuple[Tuple[str, str], str], ...]

    def axis_of(self, segment: str, component: str) -> str:
        return dict(self.axes)[(segment, component)]


def _shape_of(table: FlatTable) -> Dict[str, Tuple[int, Tuple[Tuple[str, int], ...]]]:
    """Per segment: its row width and its (component, full rows) sequence --
    the part every rank must declare identically."""
    return {
        s.key: (
            int(s.row_bytes),
            tuple((c.key, int(c.full_rows)) for c in s.components),
        )
        for s in table.segments
    }


def join_tables(name: str, whole: FlatTable, cut: Sequence[FlatTable]) -> FlatJoin:
    """Join the whole holder's declaration with the TP ranks'.  Every rank must
    declare the same segments, row widths and component sequence; the whole
    must hold every component entire; each component's rank rows must tile its
    checkpoint rows exactly (``rows``) or be the whole on every rank
    (``replicated``).  Anything else is a W68 -- never a guessed cut."""
    whole.validate(name)
    for t in cut:
        t.validate(name)
    if not cut:
        raise _refuse(f"{name}: a flat container joined against no TP rank.")
    shape = _shape_of(whole)
    for r, t in enumerate(cut):
        if _shape_of(t) != shape:
            raise _refuse(
                f"{name}: TP rank {r} declares segments {_shape_of(t)} and the "
                f"whole {shape} -- not the same container."
            )
    axes: List[Tuple[Tuple[str, str], str]] = []
    for seg in whole.segments:
        for comp in seg.components:
            if int(comp.row_start) != 0 or int(comp.rows) != int(comp.full_rows):
                raise _refuse(
                    f"{name}: the whole holds rows [{comp.row_start}, "
                    f"{int(comp.row_start) + int(comp.rows)}) of {seg.key!r}/"
                    f"{comp.key!r}, not all {comp.full_rows}."
                )
            spans = []
            for t in cut:
                c = [x for x in t.by_key()[seg.key].components if x.key == comp.key][0]
                spans.append((int(c.row_start), int(c.rows)))
            full = int(comp.full_rows)
            if all(s == (0, full) for s in spans) and len(spans) > 1:
                axes.append(((seg.key, comp.key), COMPONENT_REPLICATED))
                continue
            cursor = 0
            for start, rows in sorted(s for s in spans if s[1] > 0):
                if start != cursor:
                    raise _refuse(
                        f"{name}: {seg.key!r}/{comp.key!r} rank rows {spans} do "
                        f"not tile [0, {full}) -- a gap or an overlap at row "
                        f"{cursor}, and neither is a replica."
                    )
                cursor += rows
            if cursor != full:
                raise _refuse(
                    f"{name}: {seg.key!r}/{comp.key!r} rank rows {spans} cover "
                    f"{cursor} of {full} rows."
                )
            axes.append(((seg.key, comp.key), COMPONENT_ROWS))
    return FlatJoin(whole=whole, ranks=tuple(cut), axes=tuple(axes))


@dataclass(frozen=True)
class FlatCopy:
    """``nbytes`` from ``src_rank``'s buffer at ``src_off`` to ``dst_rank``'s at
    ``dst_off`` -- one contiguous run of whole rows."""

    src_rank: int
    dst_rank: int
    src_off: int
    dst_off: int
    nbytes: int


@dataclass(frozen=True)
class FlatFill:
    """``nbytes`` of pad at ``dst_off`` in ``dst_rank``'s buffer: zeroed, no source."""

    dst_rank: int
    dst_off: int
    nbytes: int


def _pick(cands: Sequence[Tuple[int, int, int]], d_rank: int) -> Tuple[int, int, int]:
    """The co-located source first (same index = same card), else the lowest rank."""
    for c in cands:
        if c[0] == d_rank:
            return c
    return min(cands)


def copy_plan(
    name: str,
    src: Sequence[Optional[FlatTable]],
    dst: Sequence[Optional[FlatTable]],
) -> Tuple[List[FlatCopy], List[FlatFill]]:
    """Every byte of every destination rank that holds the container, as row
    copies out of the source ranks' declarations plus pad fills.  ``None`` = the
    rank holds none of it.  A destination row with no source is W74, a byte
    covered twice or not at all is W68 -- the plan never passes by being short."""
    copies: List[FlatCopy] = []
    fills: List[FlatFill] = []
    for d_rank, dt in enumerate(dst):
        if dt is None:
            continue
        covered: List[Tuple[int, int]] = []
        for seg in dt.segments:
            for comp in seg.components:
                if int(comp.rows) <= 0:
                    continue
                d_base = seg.component_offset(comp.key)
                row, stop = int(comp.row_start), int(comp.row_start) + int(comp.rows)
                while row < stop:
                    cands = []
                    for s_rank, st in enumerate(src):
                        if st is None or seg.key not in st.by_key():
                            continue
                        sseg = st.by_key()[seg.key]
                        for sc in sseg.components:
                            if sc.key == comp.key and int(sc.row_start) <= row < int(
                                sc.row_start
                            ) + int(sc.rows):
                                if int(sseg.row_bytes) != int(seg.row_bytes):
                                    raise _refuse(
                                        f"{name}: {seg.key!r} is {sseg.row_bytes} B "
                                        f"per row on source rank {s_rank} and "
                                        f"{seg.row_bytes} on destination rank {d_rank}."
                                    )
                                cands.append(
                                    (
                                        s_rank,
                                        int(sc.row_start) + int(sc.rows),
                                        sseg.component_offset(sc.key)
                                        + (row - int(sc.row_start))
                                        * int(sseg.row_bytes),
                                    )
                                )
                    if not cands:
                        raise _missing(
                            f"{name}: destination rank {d_rank} needs row {row} of "
                            f"{seg.key!r}/{comp.key!r} and no source rank holds it."
                        )
                    s_rank, s_end, s_off = _pick(cands, d_rank)
                    end = min(stop, s_end)
                    n = (end - row) * int(seg.row_bytes)
                    d_off = d_base + (row - int(comp.row_start)) * int(seg.row_bytes)
                    copies.append(FlatCopy(s_rank, d_rank, s_off, d_off, n))
                    covered.append((d_off, d_off + n))
                    row = end
        for off, n in dt.pad_ranges():
            fills.append(FlatFill(d_rank, off, n))
            covered.append((off, off + n))
        cursor = 0
        for lo, hi in sorted(covered):
            if lo != cursor:
                raise _refuse(
                    f"{name}: destination rank {d_rank} is covered up to byte "
                    f"{cursor} and the next piece starts at {lo} -- "
                    f"{'an overlap' if lo < cursor else 'a gap'} in a "
                    f"{dt.nbytes}-byte container."
                )
            cursor = hi
        if cursor != int(dt.nbytes):
            raise _refuse(
                f"{name}: destination rank {d_rank} is covered to byte "
                f"{cursor} of {dt.nbytes}."
            )
    return copies, fills


def tables_from_json(raws: Iterable) -> Tuple[FlatTable, ...]:
    return tuple(FlatTable.from_json(r) for r in raws)
