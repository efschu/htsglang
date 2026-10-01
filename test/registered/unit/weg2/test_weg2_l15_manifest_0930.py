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
    manifest_path,
    read,
    read_and_clear,
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


# --- L15-12c-C: per-(group, rank) manifest path + read-and-clear ---------

def test_manifest_path_two_ranks_two_files() -> None:
    # The defect: ONE file for all ranks made co-located ranks overwrite
    # each other at sleep and read the same record at wake -> identical
    # fingerprint -> a FALSE "hold" agreement. Two ranks -> two distinct
    # paths (and distinct groups too).
    p0 = manifest_path("D", 0, {})
    p1 = manifest_path("D", 1, {})
    assert p0 == "/tmp/weg2_l15_manifest.D.0.json"
    assert p1 == "/tmp/weg2_l15_manifest.D.1.json"
    assert p0 != p1
    assert manifest_path("P", 0, {}) == "/tmp/weg2_l15_manifest.P.0.json"
    assert manifest_path("P", 0, {}) != p0
    # None/absent env both fall back to the per-rank default.
    assert manifest_path("D", 2, None) == "/tmp/weg2_l15_manifest.D.2.json"
    assert manifest_path("D", 2, {}) == "/tmp/weg2_l15_manifest.D.2.json"


def test_manifest_path_env_prefix_and_directory() -> None:
    # SGLANG_WEG2_L15_MANIFEST, when set, is a prefix (or a directory when
    # it ends with a separator); the per-rank suffix is still appended in
    # both forms, so ranks never collide under the override either.
    assert manifest_path(
        "D", 2, {"SGLANG_WEG2_L15_MANIFEST": "/tmp/manual"}
    ) == "/tmp/manual.D.2.json"
    assert manifest_path(
        "D", 2, {"SGLANG_WEG2_L15_MANIFEST": "/tmp/manual/"}
    ) == "/tmp/manual/weg2_l15_manifest.D.2.json"
    assert manifest_path(
        "P", 0, {"SGLANG_WEG2_L15_MANIFEST": "/tmp/manual"}
    ) == "/tmp/manual.P.0.json"


def test_read_and_clear_consumes_record(tmp_path) -> None:
    # The wake is the only reader and it must CONSUME: a stale manifest
    # from an earlier sleep must not be able to re-vote at the next wake.
    path = str(tmp_path / "m.json")
    m = _manifest()
    write(path, m)
    # First read returns the manifest and removes the file...
    assert read_and_clear(path) == m
    assert not os.path.exists(path), "consumed record must be unlinked"
    # ...so a second read (e.g. a duplicate wake) finds nothing -> None.
    assert read_and_clear(path) is None, "second read must give None"


def test_read_and_clear_keeps_pid_reap(tmp_path) -> None:
    # A dead owner's record is still reaped (removed + None) on the first
    # read -- read-and-clear must not regress the pid-reap behaviour.
    path = str(tmp_path / "m.json")
    write(path, _manifest())
    assert read_and_clear(path, pid_alive=lambda pid: False) is None
    assert not os.path.exists(path)


def test_load_for_wake_reads_and_clears(tmp_path) -> None:
    # DEFECT 2 acceptance: the wake-side loader consumes the record, so a
    # stale manifest from an earlier sleep cannot vote again.
    from sglang.srt.weg2 import l15_restore

    path = str(tmp_path / "m.json")
    write(path, _manifest())
    m = l15_restore.load_for_wake(path)
    assert m is not None and m.epoch == 7
    assert not os.path.exists(path)
    assert l15_restore.load_for_wake(path) is None


def test_source_both_sites_use_manifest_path() -> None:
    # Source check: the SLEEP site (scheduler retain hook) and the WAKE site
    # (weight_updater resume) both derive the manifest path through
    # l15_manifest.manifest_path -- the old one-file-for-all-ranks default
    # (and the misspelled "l115" variant of it) is gone from both.
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[4]
    sched = (repo / "python" / "sglang" / "srt" / "managers"
             / "scheduler.py").read_text()
    wu = (repo / "python" / "sglang" / "srt" / "managers"
          / "scheduler_components" / "weight_updater.py").read_text()
    assert "l15_manifest.manifest_path(" in sched, (
        "the sleep site does not derive its manifest path via "
        "l15_manifest.manifest_path")
    assert "l15_manifest.manifest_path(" in wu, (
        "the wake site does not derive its manifest path via "
        "l15_manifest.manifest_path")
    for name, text in (("scheduler.py", sched), ("weight_updater.py", wu)):
        assert '"/tmp/weg2_l15_manifest.json"' not in text, (
            f"{name} still carries the one-file-for-all-ranks default")
        assert '"/tmp/weg2_l115_manifest.json"' not in text, (
            f"{name} still carries the misspelled default")

