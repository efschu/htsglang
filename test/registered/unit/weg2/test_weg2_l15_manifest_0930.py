"""AP L15-04: L1.5 manifest record + group agreement (pure, CPU-only).

Covers the five acceptance points of the task:
  1. write/read roundtrip equals the original manifest
  2. a record whose pid is dead is reaped (read returns None, file gone)
  3. fingerprint is stable under span permutation and reacts to a changed
     l2_gen, a changed slot and a changed rows_by_rank
  4. agree/decide: equal fingerprints -> "hold", unequal -> "fallback"
  5. from_json of a record missing "spans" raises ValueError naming "spans"
  6. from_json of a non-integer field value raises ValueError naming the
     field ("epoch", "slots[0]")
"""

import dataclasses
import os

import pytest

from sglang.srt.weg2.l15_manifest import (
    HoldSpan,
    Manifest,
    agree,
    decide,
    fingerprint,
    from_json,
    read,
    to_json,
    write,
)


def _manifest() -> Manifest:
    """Spans are listed in rid-sorted order: to_json canonicalizes to that
    order, so this manifest is a fixed point of the write/read roundtrip."""
    return Manifest(
        epoch=7,
        pid=os.getpid(),
        spans=(
            HoldSpan("a", 2, (20, 21), 19, (200, 201), (5, 6)),
            HoldSpan("b", 4, (10, 11, 12), 9, (100,), (3,)),
        ),
        rows_by_rank=(128, 128, 128),
        anchor_slots=8,
    )


def test_roundtrip_write_read(tmp_path) -> None:
    path = str(tmp_path / "l15_manifest.json")
    m = _manifest()
    write(path, m)
    assert read(path) == m


def test_read_absent_file_returns_none(tmp_path) -> None:
    assert read(str(tmp_path / "nope.json")) is None


def test_dead_pid_is_reaped(tmp_path) -> None:
    path = str(tmp_path / "l15_manifest.json")
    m = _manifest()
    write(path, m)
    assert read(path, pid_alive=lambda pid: False) is None
    assert not os.path.exists(path)


def test_fingerprint_permutation_and_changes() -> None:
    m = _manifest()
    permuted = dataclasses.replace(m, spans=tuple(reversed(m.spans)))
    assert fingerprint(m) == fingerprint(permuted)

    changed_l2_gen = dataclasses.replace(
        m,
        spans=tuple(
            dataclasses.replace(s, l2_gens=tuple(list(s.l2_gens)[:-1] + [s.l2_gens[-1] + 1]))
            for s in m.spans
        ),
    )
    assert fingerprint(changed_l2_gen) != fingerprint(m)

    changed_slot = dataclasses.replace(
        m,
        spans=(m.spans[0], dataclasses.replace(m.spans[1], slots=(10, 11, 99))),
    )
    assert fingerprint(changed_slot) != fingerprint(m)

    changed_rows = dataclasses.replace(m, rows_by_rank=(128, 128, 129))
    assert fingerprint(changed_rows) != fingerprint(m)


def test_agree_and_decide() -> None:
    assert agree(42, 42) is True
    assert decide(42, 42) == "hold"
    assert agree(42, 43) is False
    assert decide(42, 43) == "fallback"


def test_from_json_missing_spans_names_field() -> None:
    broken = '{"anchor_slots": 8, "epoch": 7, "pid": 1, "rows_by_rank": []}'
    with pytest.raises(ValueError, match="spans"):
        from_json(broken)


def test_from_json_bad_epoch_names_field() -> None:
    broken = (
        '{"anchor_slots": 8, "epoch": "abc", "pid": 1, '
        '"rows_by_rank": [128], "spans": []}'
    )
    with pytest.raises(ValueError, match="epoch"):
        from_json(broken)


def test_from_json_bad_span_slot_names_field() -> None:
    broken = (
        '{"anchor_slots": 8, "epoch": 7, "pid": 1, "rows_by_rank": [128], '
        '"spans": [{"rid": "a", "depth": 1, "slots": ["x"], "anchor_slot": 0, '
        '"l2_slots": [], "l2_gens": []}]}'
    )
    with pytest.raises(ValueError, match="slots"):
        from_json(broken)
