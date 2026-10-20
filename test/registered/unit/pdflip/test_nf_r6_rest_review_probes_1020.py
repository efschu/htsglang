"""Review-Proben NF-R6-REST-EXAKT / R6k (Pruefer, done/nf-review-r6-rest-exakt-1020.md), uebernommen 20.10. mit den Auflagen
A1 (Start-Record = Start-Rest des FIFO-Kopfs) und K1 (Kurz-Klasse gegen D's exakten Rest). SOLL R6, R6k, R11."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from flliper.srt.pdflip import d_extend_beacon as dxb

SID = 7
R, B = "pdflip-r", "pdflip-b"


class _Ev:
    def __init__(self, done=False):
        self.done = done

    def query(self):
        return self.done


def _read(d, rid=R):
    got = dxb.read_group_full(str(d), "D", SID, session_of=lambda pid: SID)
    return None if got is None or dxb.rid_key(rid) not in got else got[dxb.rid_key(rid)][1]


def _w(tmp_path, n=1, rids=(R,)):
    ws = [dxb.DExtendBeacon(str(tmp_path / ("D-pid%d.bin" % (i + 1))), clock_ns=lambda: 10**9) for i in range(n)]
    for w in ws:
        for r in rids:
            w.note_recv(rid=r)
    return ws


def test_P1_launch_ahead_k_plus_1_start_record_hides_running_chunk_k(tmp_path):
    """Overlap: chunk k+1 wird gestartet (host), waehrend chunk k auf dem Geraet noch laeuft. Geraete-Wahrheit: 3000 nicht
    gerechnet. Vor A1 meldete der Start-Record von k+1 2000 (Unterzaehlung um den laufenden Chunk k, seit v2); mit A1 3000."""
    (w,) = _w(tmp_path)
    ev1 = _Ev(False)
    w.note_extend(rids=[R], rests=[3000], extend_lens=[1000])
    if hasattr(w, "arm_done"):
        w.arm_done(key=1, rids=[R], ready=ev1.query)
    w.note_extend(rids=[R], rests=[2000], extend_lens=[1000])   # k+1 launch, k still running
    assert _read(tmp_path) == 3000   # review A1: the device truth (chunk k still runs)


def test_P2_two_rids_concurrent_extends_fifo_and_no_double(tmp_path):
    (w,) = _w(tmp_path, rids=(R, B))
    e1, e2 = _Ev(), _Ev()
    w.note_extend(rids=[R], rests=[3000], extend_lens=[1000]); w.arm_done(key=10, rids=[R], ready=e1.query)
    w.note_extend(rids=[R, B], rests=[2000, 500], extend_lens=[1000, 500]); w.arm_done(key=11, rids=[R, B], ready=e2.query)
    assert _read(tmp_path, R) == 3000 and _read(tmp_path, B) == 500   # review A1: R's chunk 1 still runs
    e1.done = True
    assert w.poll_done() == 1
    assert _read(tmp_path, R) == 2000 and _read(tmp_path, B) == 500
    w.note_extend_done(rids=[R], key=10)          # result k after poll: nothing twice
    assert _read(tmp_path, R) == 2000 and _read(tmp_path, B) == 500
    e2.done = True
    assert w.poll_done() == 1
    assert _read(tmp_path, R) == 1000 and _read(tmp_path, B) == 0
    w.note_extend_done(rids=[R, B], key=11)
    assert _read(tmp_path, R) == 1000 and _read(tmp_path, B) == 0
    assert not w._pending and not w._polled


def test_P3_result_with_key_none_after_poll_double_writes(tmp_path):
    """Ohne key (forward_iter fehlt) schreibt der Ergebnispfad doppelt: Rest zu klein. Real: forward_iter wird in
    _run_batch_forward immer gesetzt und von ScheduleBatch.copy() mitgenommen -> nur dokumentiert."""
    (w,) = _w(tmp_path)
    e1, e2 = _Ev(True), _Ev()
    w.note_extend(rids=[R], rests=[3000], extend_lens=[1000]); w.arm_done(key=1, rids=[R], ready=e1.query)
    w.poll_done()
    w.note_extend(rids=[R], rests=[2000], extend_lens=[1000]); w.arm_done(key=2, rids=[R], ready=e2.query)
    w.note_extend_done(rids=[R], key=None)
    assert _read(tmp_path) == 1000   # undercount while k+1 runs (2000 true)


def test_P4_bounded_state_after_many_batches_without_results(tmp_path):
    (w,) = _w(tmp_path)
    for k in range(1, 2000):
        w.note_extend(rids=[R], rests=[10**6 - k], extend_lens=[1])
        w.arm_done(key=k, rids=[R], ready=_Ev(True).query)
        w.poll_done()
    assert len(w._pending) == 0 and len(w._polled) <= dxb.PENDING_CAP
    for k in range(5000, 5600):
        w.note_extend(rids=[R], rests=[5], extend_lens=[1])
        w.arm_done(key=k, rids=[R], ready=_Ev(False).query)
    assert len(w._pending) <= dxb.PENDING_CAP


def test_P5_mm_none_no_state(tmp_path):
    w = dxb.DExtendBeacon(str(tmp_path / "x" / "D-pid1.bin"))
    w.mm = None
    w.arm_done(key=1, rids=[R], ready=_Ev(True).query)
    assert w.poll_done() == 0 and not w._pending


# ---- R6k probes (review K1 applied)
import collections  # noqa: E402

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy  # noqa: E402


class _S:
    def __init__(self):
        self.counters = collections.Counter()
        self.epoch = 1


def test_K1_kurz_class_with_d_rest_over_half_x_is_contradicted():
    s = _S()
    F.Front._r6k_note_class(s, "r", phase_policy.CLASS_KURZ, "arrival", 3508)
    # X=7016, halbX=3508; D meldet exakt 5000 Rest (Praefix seit der Ankunft verdraengt) -> nicht kurz, R6 wie bisher
    assert F.Front._r6k_kurz_stays(s, "r", 5000, 1394) is False
    assert s.counters["r6k_kurz_contradicted"] == 1 and s.counters["r6k_kurz_stays_on_d"] == 0
    assert F.Front._r6k_kurz_stays(s, "r", 3507, 1394) is True       # under halbX: kurz stands


def test_K2_unknown_class_counts_as_before():
    s = _S()
    assert F.Front._r6k_kurz_stays(s, "x", 900, 900) is False
    assert s.counters["r6k_class_unknown"] == 1
