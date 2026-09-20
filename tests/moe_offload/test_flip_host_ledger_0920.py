# SPDX-License-Identifier: Apache-2.0
"""Slice 2: the ``[flip-host-ledger]`` line and its W114 verdict.

Hermetic: no torch, no device, no /proc read that is not explicitly pointed
at a fixture file. The measured six-process posts come from the two boots the
design cites by name.
"""
from __future__ import annotations

import pytest

from sglang.srt.flip_host_ledger import (
    LEDGER_PREFIX,
    ProcessHostPost,
    format_host_ledger_line,
    host_post_from_rss,
    ledger_from_lines,
    parse_host_ledger_line,
    read_rss_fields,
)
from sglang.srt.flip_nextflash_plan import (
    FORM_A_PINNED_GIB,
    HOST_MARK_GIB,
    PP3_PINNED_GIB,
    Weg2FlipHostPoolDoubled,
)

_GIB = 1024**3


def _post(layout, rank, pinned_gib, anon_gib, shm_gib=0.0, shared=False, pid=1000):
    return ProcessHostPost(
        layout=layout,
        rank=rank,
        pid=pid,
        pinned_bytes=int(pinned_gib * _GIB),
        anon_bytes=int(anon_gib * _GIB),
        shm_bytes=int(shm_gib * _GIB),
        shared=shared,
    )


def _six_lines(anon_gib, shared=False, shm_gib=0.0):
    lines = []
    pid = 1000
    for rank, gib in enumerate(PP3_PINNED_GIB):
        lines.append(
            format_host_ledger_line(
                _post("P", rank, gib, anon_gib, shm_gib, shared, pid + rank)
            )
        )
    for rank, gib in enumerate(FORM_A_PINNED_GIB):
        lines.append(
            format_host_ledger_line(
                _post("D", rank, gib, anon_gib, shm_gib, shared, pid + 10 + rank)
            )
        )
    return lines


# --------------------------------------------------------------------------
# The decomposition: the private pool must not be counted twice
# --------------------------------------------------------------------------
def test_a_private_pool_is_subtracted_from_rssanon():
    """``pinned_exact_empty`` maps MAP_PRIVATE|MAP_ANONYMOUS, so the pool is
    already inside RssAnon. Adding it would count 22.73 GiB twice."""
    rss = {"RssAnon": 30 * _GIB, "RssShmem": 0}
    post = host_post_from_rss("P", 0, pinned_bytes=22 * _GIB, shared=False, rss=rss, pid=7)
    assert post.pinned_gib == pytest.approx(22.0)
    assert post.anon_gib == pytest.approx(8.0)  # 30 - 22, not 30
    assert post.shm_gib == pytest.approx(0.0)


def test_a_shared_pool_lands_in_shmem_and_leaves_rssanon_alone():
    """With the cold tier on, the pool is a MAP_SHARED tmpfs mapping
    (shared_pinned.py:35-39): RssAnon is then the honest anonymous post."""
    rss = {"RssAnon": 8 * _GIB, "RssShmem": 22 * _GIB}
    post = host_post_from_rss("D", 0, pinned_bytes=22 * _GIB, shared=True, rss=rss, pid=7)
    assert post.anon_gib == pytest.approx(8.0)
    assert post.shm_gib == pytest.approx(22.0)


def test_a_stale_pinned_reading_clamps_instead_of_going_negative():
    rss = {"RssAnon": 1 * _GIB, "RssShmem": 0}
    post = host_post_from_rss("P", 1, pinned_bytes=5 * _GIB, shared=False, rss=rss, pid=7)
    assert post.anon_gib == 0.0
    assert post.pinned_gib == pytest.approx(5.0)  # still visible beside it


def test_a_missing_rss_field_does_not_read_as_zero_memory(tmp_path):
    """A kernel that does not report RssAnon must not make a process look
    empty -- read_rss_fields returns the field ABSENT, not 0."""
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmRSS:\t   1024 kB\n")
    fields = read_rss_fields(str(status))
    assert "RssAnon" not in fields
    assert fields["VmRSS"] == 1024 * 1024


def test_read_rss_fields_parses_a_real_shaped_status(tmp_path):
    status = tmp_path / "status"
    status.write_text(
        "Name:\tpython\n"
        "VmRSS:\t31457280 kB\n"
        "RssAnon:\t23068672 kB\n"
        "RssFile:\t  262144 kB\n"
        "RssShmem:\t 8126464 kB\n"
        "VmLck:\t23068672 kB\n"
    )
    f = read_rss_fields(str(status))
    assert f["RssAnon"] == 23068672 * 1024
    assert f["RssShmem"] == 8126464 * 1024
    assert f["VmLck"] == 23068672 * 1024


def test_a_missing_status_file_is_empty_not_an_exception():
    assert read_rss_fields("/definitely/not/a/status/file") == {}


# --------------------------------------------------------------------------
# The line: format and parse must round-trip
# --------------------------------------------------------------------------
def test_the_line_round_trips():
    post = _post("D", 2, 10.88, 6.5, 0.0, False, pid=4242)
    line = format_host_ledger_line(post)
    assert line.startswith(LEDGER_PREFIX)
    back = parse_host_ledger_line(line)
    assert back.layout == "D"
    assert back.rank == 2
    assert back.pid == 4242
    assert back.pinned_gib == pytest.approx(10.88, abs=0.005)
    assert back.anon_gib == pytest.approx(6.5, abs=0.005)
    assert back.shared is False


def test_the_line_survives_a_boot_log_prefix():
    post = _post("P", 0, 22.73, 6.5)
    raw = "[2026-09-20 18:00:00 TP0] INFO " + format_host_ledger_line(post)
    back = parse_host_ledger_line(raw)
    assert back.pinned_gib == pytest.approx(22.73, abs=0.005)


def test_a_non_ledger_line_is_refused():
    with pytest.raises(Weg2FlipHostPoolDoubled):
        parse_host_ledger_line("[offload-kv-regain] rank 0: ...")


def test_a_truncated_line_is_refused_not_defaulted():
    """A missing post is not a zero post."""
    with pytest.raises(Weg2FlipHostPoolDoubled) as exc:
        parse_host_ledger_line(f"{LEDGER_PREFIX} layout=P rank=0 pinned=22.73")
    assert "missing" in str(exc.value)


# --------------------------------------------------------------------------
# The verdict: the measured six processes against the 88 GiB mark
# --------------------------------------------------------------------------
def test_the_measured_six_process_form_unshared_blows_the_mark():
    """31.88 + 38.86 = 70.74 GiB pinned, plus the anonymous term. At the
    design's assumed 6.5 GiB anon per rank that is 70.74 + 39 = 109.7."""
    with pytest.raises(Weg2FlipHostPoolDoubled) as exc:
        ledger_from_lines(_six_lines(anon_gib=6.5, shared=False))
    msg = str(exc.value)
    assert "W114 Weg2FlipHostPoolDoubled" in msg
    assert "70.74" in msg  # the pinned sum
    assert "PER-PROCESS" in msg or "Sharing it would take" in msg


def test_sharing_takes_the_pinned_term_to_the_per_rank_maximum():
    """max(22.73;20.62) + max(5.30;7.36) + max(3.85;10.88) = 40.97 GiB."""
    ledger = ledger_from_lines(_six_lines(anon_gib=3.0, shared=True, shm_gib=0.0))
    assert ledger.shared is True
    assert ledger.pinned_gib == pytest.approx(40.97, abs=0.01)
    assert ledger.saved_gib == pytest.approx(70.74 - 40.97, abs=0.01)


def test_sharing_alone_is_not_enough_at_the_designs_anon_term():
    """Design §5.2: sharing takes the six-process form from ~126 to ~96 GiB
    -- closer, still over. The refusal must say ALREADY shared so nobody
    proposes sharing a second time."""
    with pytest.raises(Weg2FlipHostPoolDoubled) as exc:
        ledger_from_lines(_six_lines(anon_gib=9.2, shared=True))
    assert "ALREADY" in str(exc.value)


def test_the_shared_segment_is_counted_once_not_once_per_reader():
    """Two readers of a 20 GiB segment cost 20 GiB, not 40. If this ever
    sums, the shared pool looks like a failure of sharing rather than of
    counting."""
    lines = [
        format_host_ledger_line(_post("P", 0, 0.0, 2.0, shm_gib=20.0, shared=True, pid=1)),
        format_host_ledger_line(_post("D", 0, 0.0, 2.0, shm_gib=20.0, shared=True, pid=2)),
    ]
    ledger = ledger_from_lines(lines, mark_gib=25.0)
    # pinned 0 + anon 4 + shm counted once (20) = 24 < 25 -> fits.
    assert ledger.anon_gib == pytest.approx(4.0)
    # and one GiB more of anon per process pushes it over:
    over = [
        format_host_ledger_line(_post("P", 0, 0.0, 3.0, shm_gib=20.0, shared=True, pid=1)),
        format_host_ledger_line(_post("D", 0, 0.0, 3.0, shm_gib=20.0, shared=True, pid=2)),
    ]
    with pytest.raises(Weg2FlipHostPoolDoubled) as exc:
        ledger_from_lines(over, mark_gib=25.0)
    assert "counted ONCE" in str(exc.value)


def test_a_half_shared_boot_is_refused_rather_than_totalled():
    """Half the rig on the segment and half on private copies double-counts
    exactly the processes that did not attach."""
    lines = _six_lines(anon_gib=1.0, shared=True)[:3] + _six_lines(anon_gib=1.0)[3:]
    with pytest.raises(Weg2FlipHostPoolDoubled) as exc:
        ledger_from_lines(lines)
    msg = str(exc.value)
    assert "HALF shared" in msg
    assert "SGLANG_MOE_COLD_TIER_INSTANCE" in msg


def test_no_ledger_lines_at_all_is_not_a_pass():
    """An unmeasured host is not a host under the mark."""
    with pytest.raises(Weg2FlipHostPoolDoubled) as exc:
        ledger_from_lines(["[offload-kv-regain] rank 0: ..."])
    assert "no [flip-host-ledger] lines" in str(exc.value)


def test_a_one_sided_boot_fits_and_answers_risk_r1():
    """Slice 2 is boot-able one-sided. A Form-A-only boot at the measured
    38.86 GiB pinned plus a REAL anonymous term must fit under 88 -- that is
    the measurement that retires risk R1."""
    lines = [
        format_host_ledger_line(_post("D", r, g, 6.5))
        for r, g in enumerate(FORM_A_PINNED_GIB)
    ]
    ledger = ledger_from_lines(lines)
    assert ledger.pinned_gib == pytest.approx(38.86, abs=0.01)
    assert ledger.total_gib < HOST_MARK_GIB


def test_the_emitter_never_raises_on_a_broken_logger():
    """An instrument that kills the boot it measures is not an instrument."""
    from sglang.srt.flip_host_ledger import emit_host_ledger_line

    class Boom:
        def info(self, *a, **k):
            raise RuntimeError("log is gone")

    assert emit_host_ledger_line(Boom(), "P", 0, 1 << 30, False) is None
