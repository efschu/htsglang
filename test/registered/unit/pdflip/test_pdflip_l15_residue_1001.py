"""L15-13c: the D dormant-residue record excludes the L1.5 hold.

The P budget subtracts D's measured dormant residue per card
("dormant_other", launcher.dormant_max_from_records) AND the planner also
subtracts the L1.5 hold as its own ``l15`` post (L15-01b,
launcher.budgets_from_dc).  With the hold on, D's kv_cache keeps its held
rows mapped at sleep, so the residue measured at D's first sleep already
CONTAINS the hold; a record that keeps it would be charged twice on the
next boot.  The writer therefore strips the held bytes (kv_cache bytes
still mapped at sleep, via the adapter's ``tag_mapped_bytes``) from the
record; this test pins the pure function and the writer's call to it.

Plain pytest functions -- CustomTestCase.retry() hides failures.
"""
from __future__ import annotations

from pathlib import Path

from flliper.srt.pdflip import l15_plan as LP

FRONT_PY = (
    Path(__file__).resolve().parents[4]
    / "python" / "flliper" / "srt" / "pdflip" / "front.py"
)


def test_residue_without_hold_subtracts():
    assert LP.residue_without_hold(100, 30) == 70


def test_residue_without_hold_none_unchanged():
    assert LP.residue_without_hold(100, None) == 100


def test_residue_without_hold_floors_at_zero():
    assert LP.residue_without_hold(10, 30) == 0


def test_writer_calls_residue_without_hold():
    src = FRONT_PY.read_text()
    start = src.index("def sample_dormant_image")
    end = src.index("\n    def ", start + 1)
    body = src[start:end]
    assert "residue_without_hold" in body
