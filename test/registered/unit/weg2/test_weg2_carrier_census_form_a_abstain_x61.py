"""fnFL2x61: the carrier census lets Form A expert workers abstain.

Bug regression.  Group D of the Next Flash Form A boot is one attention host
(TP0, the 5090) and two expert workers (TP1/TP2: no attention layer, null
storage tier).  Under the arena host form (Task #107) TP0's ``#915 PREFETCH
LIMIT`` prints the arena's capacity (now=157248) while the workers' plain pool
is synced to TP0's 4096-row staging ring and prints 0.9 x that (now=3686).
The census demanded agreement over all three and refused the boot with

    W45 Weg2CarrierCensusRefused (disagree): the 3 TP ranks do not agree on
    'now': '157248' on TP [0]; '3686' on TP [1, 2]

although the group's carrier is TP0's alone.  The workers now abstain, named
by the marker line they print instead of ``#706 canonical KV page active``.
Hermetic: no CUDA, no server, no boot.
"""

from sglang.srt.weg2 import carrier_census as cc

LIMIT_TPL = (
    "[2026-09-23 15:44:20 TP{rank}] #915 PREFETCH LIMIT now={now} "
    "(fraction=0.9 x host size {size}) role=staging pool_id={pool} "
    "phase=pp generation=0 site=init_hicache"
)
WORKER_TPL = "[2026-09-23 15:44:19 TP{rank}] #706 canonical KV page: " + cc.FORM_A_WORKER_MARKER
FLOOR = 5120


def _x61_log(tmp_path, *, workers=(1, 2)):
    lines = [WORKER_TPL.format(rank=r) for r in workers]
    lines.append(LIMIT_TPL.format(rank=0, now=157248, size=4096, pool=127892673399120))
    lines.append(LIMIT_TPL.format(rank=1, now=3686, size=4096, pool=138470519137312))
    lines.append(LIMIT_TPL.format(rank=2, now=3686, size=4096, pool=139319074091280))
    p = tmp_path / "D.log"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_the_worker_ranks_are_read_from_their_own_marker_line(tmp_path):
    log = _x61_log(tmp_path)
    assert cc.form_a_worker_ranks(log) == (1, 2)


def test_without_abstention_the_x61_log_is_the_recorded_refusal(tmp_path):
    """The pre-fix shape, pinned so the abstention test cannot pass vacuously."""
    cen = cc.census(_x61_log(tmp_path), expected_ranks=3, floor=FLOOR)
    assert cen.verdict == "disagree"
    assert "'157248' on TP [0]; '3686' on TP [1, 2]" in cen.detail


def test_abstaining_workers_leave_the_attention_host_as_the_bound(tmp_path):
    log = _x61_log(tmp_path)
    cen = cc.census(log, expected_ranks=3, floor=FLOOR,
                    abstain_ranks=cc.form_a_worker_ranks(log))
    assert cen.verdict == "ok", cen.detail
    assert cen.bound == 157248
    assert cen.measured == 157248
    assert cen.per_rank == {0: 157248}
    assert cen.abstained == (1, 2)
    assert "TP [1, 2] abstain" in cen.detail
    # the workers' lines are not the census's evidence either
    assert all("TP0]" in ln for ln in cen.lines)


def test_an_all_abstaining_group_has_no_carrier_and_is_missing(tmp_path):
    log = _x61_log(tmp_path, workers=(0, 1, 2))
    cen = cc.census(log, expected_ranks=3, floor=FLOOR, abstain_ranks=(0, 1, 2))
    assert cen.verdict == "missing"
    assert cen.measured is None


def test_a_missing_attention_host_is_still_missing_with_abstaining_workers(tmp_path):
    lines = [WORKER_TPL.format(rank=r) for r in (1, 2)]
    lines.append(LIMIT_TPL.format(rank=1, now=3686, size=4096, pool=1))
    lines.append(LIMIT_TPL.format(rank=2, now=3686, size=4096, pool=2))
    p = tmp_path / "D.log"
    p.write_text("\n".join(lines) + "\n")
    cen = cc.census(str(p), expected_ranks=3, floor=FLOOR, abstain_ranks=(1, 2))
    assert cen.verdict == "missing"
    assert cen.measured is None
    assert cen.per_rank == {}  # the workers' lines never became evidence
    assert "TP [1, 2] abstain" in cen.detail


def test_a_log_without_the_marker_abstains_nobody(tmp_path):
    p = tmp_path / "D.log"
    p.write_text(LIMIT_TPL.format(rank=0, now=1, size=1, pool=1) + "\n")
    assert cc.form_a_worker_ranks(str(p)) == ()
    assert cc.form_a_worker_ranks(str(tmp_path / "absent.log")) == ()
