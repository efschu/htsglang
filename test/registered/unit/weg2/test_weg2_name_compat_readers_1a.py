# SPDX-License-Identifier: Apache-2.0
"""Rename step 1a: every in-tree reader of boot logs accepts the old AND the new name.

The rename turns the subsystem token into ``PDFLIP`` (line markers
``PDFLIP-GRAPH-POOL``, logger ``pdflip.front``, log stems ``boot_pdflip_*``).
Evidence written before it keeps the old spelling, and the planners read such
logs at every boot (wake-credit, P-card, D-card, graph-pool references; the
ring/form identity scans the evidence directory). Each test here feeds one
parser a REAL old log excerpt and the same excerpt in the new spelling and
requires the same, non-empty answer.

The old excerpts are fixtures (``fixtures/*``, verbatim boot-log lines, never
renamed: they are evidence). The new spelling is produced here, by
:func:`to_new`, which is written without the old token in one piece so the
mechanical rename cannot turn it into a no-op.
"""

import importlib.util
import os
import re
import shutil

import pytest

from sglang.srt import name_compat as nc
from sglang.srt.planner import expert_residency as er
from sglang.srt.planner import graph_pool_ledger as gpl
from sglang.srt.planner import p_card_chunk as pc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
NEW_FIX = os.path.join(FIX, "name_compat_1a")

OLD_U, NEW_U = "WE" "G2", "PDFLIP"
OLD_L, NEW_L = "we" "g2", "pdflip"
_RX_UP = re.compile(r"(?<![A-Za-z0-9_])%s(?=[- ])" % OLD_U)
_RX_LOGGER = re.compile(r"(?<![A-Za-z0-9_.])%s(?=\.front)" % OLD_L)
_RX_STEM = re.compile(r"(?<=boot_)%s(?=_)" % OLD_L)


def to_new(text: str) -> str:
    """The same log text as the renamed tree writes it."""
    out = _RX_STEM.sub(NEW_L, _RX_LOGGER.sub(NEW_L, _RX_UP.sub(NEW_U, text)))
    assert out != text, "excerpt carries no old marker -- the pair would prove nothing"
    assert not _RX_UP.search(out)
    return out


def _read(*parts: str) -> str:
    with open(os.path.join(FIX, *parts), errors="replace") as fh:
        return fh.read()


def _write(tmp_path, name: str, text: str) -> str:
    p = os.path.join(str(tmp_path), name)
    with open(p, "w") as fh:
        fh.write(text)
    return p


def _both(fn, text: str):
    """fn(old) and fn(new); the old answer must be non-empty."""
    old, new = fn(text), fn(to_new(text))
    assert old, "the parser found nothing in the OLD excerpt -- fixture or parser drifted"
    return old, new


# --------------------------------------------------------------------------
# the helper
# --------------------------------------------------------------------------


class TestHelper:
    def test_both_spellings_give_one_pattern(self):
        for old in (OLD_U + r"-CORRIDOR\s+phase=", OLD_U + " P-DRAIN epoch=",
                    r"^boot_" + OLD_L + r"_(?P<tag>.+)_x", r"\] INFO " + OLD_L + r"\.front: " + OLD_U + "-FLIP"):
            a, b = nc.tolerant_rx(old), nc.tolerant_rx(to_new(old))
            assert a == b and a != old
            assert nc.tolerant_rx(a) == a  # idempotent
        # re.escape()d markers (graph_pool_ledger, form, ring_table build their patterns so)
        a = nc.tolerant_rx(re.escape(OLD_U + "-GRAPH-POOL"))
        assert a == nc.tolerant_rx(re.escape(NEW_U + "-GRAPH-POOL")) != re.escape(OLD_U + "-GRAPH-POOL")
        assert re.match(a, OLD_U + "-GRAPH-POOL") and re.match(a, NEW_U + "-GRAPH-POOL")

    def test_names_that_are_not_markers_stay(self):
        # env/constant names, boot TAGS (evidence names), dotted module paths
        for s in ("SGLANG_" + OLD_U + "_GROUP", OLD_U + "_DESIGN_SPEC",
                  OLD_L + "xsn412", "boot_" + OLD_L + "_" + OLD_L + "xsn412_ab",
                  r"srt\." + OLD_L + r"\.launcher", "/spinning/gpu-arb/" + OLD_L + "/boot_x.json"):
            got = nc.tolerant_rx(s)
            if s.startswith("boot_"):
                assert got == "boot_(?:%s|%s)_%sxsn412_ab" % (OLD_L, NEW_L, OLD_L)
            else:
                assert got == s

    def test_variants_and_lookups(self):
        assert nc.marker_variants(OLD_U + "-FLIP begin") == (OLD_U + "-FLIP begin", NEW_U + "-FLIP begin")
        assert nc.marker_variants(NEW_U + "-FLIP begin") == (OLD_U + "-FLIP begin", NEW_U + "-FLIP begin")
        assert nc.marker_variants("PP-CUT ACTIVATION") == ("PP-CUT ACTIVATION",)
        assert nc.has_marker("x %s-DC group=D" % NEW_U, OLD_U + "-DC")
        assert nc.has_marker("x %s-DC group=D" % OLD_U, NEW_U + "-DC")
        assert not nc.has_marker("x DC group=D", OLD_U + "-DC")
        assert nc.marker_tail("[t] %s-VRAM-PEAK rank=1" % NEW_U, OLD_U + "-VRAM-PEAK ") == "rank=1"
        assert nc.marker_tail("nothing", OLD_U + "-VRAM-PEAK ") is None

    def test_helper_spells_no_old_token_in_one_piece(self):
        """The hermetic half of the rename check below: the rename rewrites
        exactly these tokens, so a file without them comes out unchanged."""
        src = open(nc.__file__).read()
        assert not re.search(OLD_L + "|" + OLD_U + "|W" "eg2", src)
        assert not re.search("sg" "lang|SG" "LANG|SG" "Lang|Sg" "lang|SG" "lang|sG" "Lang", src)


def _rename_tool():
    p = "/spinning/flliper/tools/rename_to_flliper.py"
    if not os.path.exists(p):
        return None
    spec = importlib.util.spec_from_file_location("_rename_to_flliper", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_name_compat_survives_rename():
    """The mechanical rename (all rule sets on) leaves the helper byte-identical,
    so its both-spellings answer is the same before and after the rename."""
    tool = _rename_tool()
    if tool is None:
        pytest.skip("rename tool not on this box")
    src = open(nc.__file__).read()
    out, rep, _skip = tool.rewrite_all(src, True, True, {})
    assert out == src, rep


# --------------------------------------------------------------------------
# planner references (read at every boot)
# --------------------------------------------------------------------------


def test_wake_credit_reference_d_to_p():
    from sglang.srt.weg2 import wake_credit as wc

    def ref(p, d, f):
        return wc.reference_from_logs(p, d, f, source="fnFL2x104", p_card=(1, 0, 2))

    p, d, f = (_read("wake_credit_h14", "fnFL2x104.%s.lines" % k) for k in ("P", "D", "front"))
    old = ref(p, d, f)
    assert old is not None
    assert ref(to_new(p), to_new(d), to_new(f)) == old


def test_wake_credit_reference_p_to_d():
    from sglang.srt.weg2 import wake_credit_pd as wpd

    def ref(p, d, f):
        return wpd.pd_reference_from_logs(p, d, f, source="fnFL2x141/0", p_card=(1, 0, 2), flip=0)

    p, d, f = (_read("wake_credit_pd_h34", "fnFL2x141.%s.lines" % k) for k in ("P", "D", "front"))
    old = ref(p, d, f)
    assert old is not None
    assert ref(to_new(p), to_new(d), to_new(f)) == old


def test_d_residency_h39_state():
    old, new = _both(er.boot_dense_repack_outside_pool, _read("d_h39_h50", "fnFL2x151.D.lines"))
    assert old is True and new is True


def test_graph_pool_ledger_samples():
    old, new = _both(gpl.samples_from_log, _read("d_h39_h50", "fnFL2x150.D.lines"))
    assert new == old


@pytest.mark.parametrize("boot", ["fnFL2x149", "fnFL2x164"])
def test_p_card_death_from_log(boot):
    text = _read("p_card_h41", boot + ".P.lines")
    old = pc.death_from_log(boot, text)
    assert old is not None
    assert pc.death_from_log(boot, to_new(text)) == old


@pytest.mark.parametrize("boot,rx,key", [
    ("fnFL2x101", "_RX_RUNWRITE", "-ARENA-WRITE"),
    ("fnFL2x130", "_RX_SLEEP_LMEM", "-SLEEP-LMEM lmem"),
])
def test_p_card_death_markers(boot, rx, key):
    """The two death_from_log markers the boots above do not reach."""
    lines = [l for l in _read("p_card_h41", boot + ".P.lines").splitlines() if key in l]
    rx = getattr(pc, rx)
    hits = [(rx.search(l), rx.search(to_new(l))) for l in lines]
    assert any(o for o, _n in hits)
    for o, n in hits:
        assert (o is None) == (n is None) and (o is None or o.groups() == n.groups())


def test_p_card_chunk_windows_and_pools():
    text = _read("p_card_h41", "fnFL2x164.P.lines")
    old, new = _both(pc.chunk_windows, text)
    assert new == old
    old, new = _both(pc.pool_samples, text)
    assert new == old


def test_p_card_co_tenant():
    old, new = _both(pc.co_tenant_by_flip, _read("p_card_h41", "fnFL2x164.D.lines"))
    assert new == old


# --------------------------------------------------------------------------
# evidence-directory readers: ring table, form identity, census
# --------------------------------------------------------------------------

RING_EVIDENCE = os.path.join(FIX, "xchg_launch_replay_0911", "ring_evidence")
RING_STEM = "boot_" + OLD_L + "_" + OLD_L + "sn5b_d8ea6261f7_0909_185632"


def _renamed_evidence(tmp_path) -> str:
    """The ring evidence dir as the renamed tree would have written it: file
    stems ``boot_<new>_...`` and new markers inside. The measured-record
    sidecar keeps its name (its reader is not a log parser)."""
    d = os.path.join(str(tmp_path), "ev_new")
    os.makedirs(d)
    for n in os.listdir(RING_EVIDENCE):
        src = os.path.join(RING_EVIDENCE, n)
        if n.startswith("boot_"):
            with open(src, errors="replace") as fh:
                text = fh.read()
            with open(os.path.join(d, to_new(n)), "w") as fh:
                fh.write(to_new(text))
        else:
            shutil.copy(src, os.path.join(d, n))
    return d


def test_ring_stem_and_group_logs(tmp_path):
    from sglang.srt.weg2 import ring_table as rt

    new_dir = _renamed_evidence(tmp_path)
    new_stem = to_new(RING_STEM)
    assert rt.boot_tag_of_stem(RING_STEM) == OLD_L + "sn5b"
    assert rt.boot_tag_of_stem(new_stem) == OLD_L + "sn5b"  # the TAG is evidence, never renamed
    for kind in ("P", "D"):
        old = rt.parse_group_log(os.path.join(RING_EVIDENCE, RING_STEM + ".%s.log" % kind))
        new = rt.parse_group_log(os.path.join(new_dir, new_stem + ".%s.log" % kind))
        assert old.lines_read if hasattr(old, "lines_read") else True
        assert repr(new) == repr(old)
    f_old = os.path.join(RING_EVIDENCE, RING_STEM + ".front.log")
    f_new = os.path.join(new_dir, new_stem + ".front.log")
    old, new = rt.parse_front_corridor(f_old), rt.parse_front_corridor(f_new)
    assert old and new == old
    assert rt.front_corridor_instrument(f_new) == rt.front_corridor_instrument(f_old)
    assert repr(rt.parse_front_corridor_floors(f_new)) == repr(rt.parse_front_corridor_floors(f_old))


def test_ring_solve_over_renamed_evidence(tmp_path):
    """The whole ring solve (stem discovery, group logs, sidecar join by tag)
    finds the same table in an evidence dir written under the new name."""
    from sglang.srt.weg2 import launcher as L
    from sglang.srt.weg2 import ring_table as rt
    import json

    with open(os.path.join(FIX, "xchg_launch_replay_0911", "recorded.json")) as fh:
        rec = json.load(fh)
    cards = [L.Card(nvml_index=int(c["nvml_index"]), uuid=c["uuid"], name=c["name"],
                    total_mib=int(c["total_mib"])) for c in rec["cards_ordinal_order"]]
    old_t, old_why = rt.solve(cards, RING_EVIDENCE, RING_STEM, p_argv=None)
    new_stem = to_new(RING_STEM)
    new_t, new_why = rt.solve(cards, _renamed_evidence(tmp_path), new_stem, p_argv=None)
    assert old_t is not None, old_why
    assert new_t is not None, new_why
    assert repr(new_t).replace(new_stem, RING_STEM) == repr(old_t)


def test_ring_xchg_form_event_line():
    from sglang.srt.weg2 import ring_table as rt

    line = next(l for l in _read("name_compat_1a", "h91v1.front.lines").splitlines()
                if "-XCHG-REGION epoch=" in l)
    assert rt._XCHG_FORM_EVENT_RE.search(line)
    assert rt._XCHG_FORM_EVENT_RE.search(to_new(line))


def test_form_log_stems_and_identity(tmp_path):
    from sglang.srt.weg2 import form

    for n in os.listdir(RING_EVIDENCE):
        if not n.startswith("boot_"):
            continue
        m_old = form._BOOT_LOG_RE.match(n)
        m_new = form._BOOT_LOG_RE.match(to_new(n))
        assert m_old and m_new and m_new.groupdict() == m_old.groupdict()
        if n.endswith(".front.log"):
            assert form._FRONT_LOG_RE.match(to_new(n)).groupdict() == form._FRONT_LOG_RE.match(n).groupdict()
    text = _read("name_compat_1a", "h91v1.front.lines")
    old = form.log_identity(_write(tmp_path, "old.front.log", text))
    new = form.log_identity(_write(tmp_path, "new.front.log", to_new(text)))
    assert old.form is not None and old.model
    assert new == old


def test_xchg_census_front_readers(tmp_path):
    from sglang.srt.weg2 import xchg_census as xc

    text = _read("name_compat_1a", "h91v1.front.lines")
    f_old, f_new = _write(tmp_path, "o.front.log", text), _write(tmp_path, "n.front.log", to_new(text))
    for fn in (xc.dormant_readings, xc.family_from_front, xc.realized_split):
        old = fn(f_old)
        assert old, fn.__name__
        assert fn(f_new) == old, fn.__name__
    assert xc.wave_map_from_front(f_new)[0] == xc.wave_map_from_front(f_old)[0]
    assert xc.wave_map_from_front(f_old)[0]
    src = _read("name_compat_1a", "sb1.front.lines")
    assert xc._SOURCE_RE.search(src).group(1) == xc._SOURCE_RE.search(to_new(src)).group(1).replace(
        "boot_" + NEW_L + "_", "boot_" + OLD_L + "_")


# --------------------------------------------------------------------------
# launcher, arms and report tools
# --------------------------------------------------------------------------


def test_launcher_break_even_patterns(tmp_path):
    """measure_x_inputs' three front-log patterns (leg 1, P-drain, leg 2) on the
    real lines; the function itself needs a single_prefill verdict this
    excerpt does not carry, so it is checked for equal answers only."""
    from sglang.srt.weg2 import launcher as L

    text = _read("name_compat_1a", "h91v1.front.lines")
    for rx, key in ((L._RE_LEG1, "group=P leg=1"), (L._RE_DRAIN, " P-DRAIN epoch="), (L._RE_LEG2, "group=D leg=2")):
        line = next(l for l in text.splitlines() if key in l)
        m_old, m_new = rx.search(line), rx.search(to_new(line))
        assert m_old and m_new and m_new.groups() == m_old.groups(), key
    assert L.measure_x_inputs(_write(tmp_path, "n.log", to_new(text)), 0) == \
        L.measure_x_inputs(_write(tmp_path, "o.log", text), 0)


def test_launcher_draft_kv_producer_line(tmp_path):
    from sglang.srt.weg2 import launcher as L

    text = _read("name_compat_1a", "rc7c.P.lines")
    old = L.check_draft_resident(_write(tmp_path, "o.P.log", text))
    new = L.check_draft_resident(_write(tmp_path, "n.P.log", to_new(text)))
    assert old.get("resident_mib") not in (None, -1)
    assert new == old


def test_dormant_arm_released_and_degrades():
    from sglang.srt.weg2 import dormant_arm as da

    d = _read("name_compat_1a", "h91v1.D.lines")
    old, new = _both(lambda t: da.parse_released(t, "D"), d)
    assert new == old
    p = _read("name_compat_1a", "xsn412.P.lines")
    old, new = _both(lambda t: da.find_degrades(t, "P"), p)
    assert [to_new(x) for x in old] == new


def test_corridor_arm_counts_samples(tmp_path):
    from sglang.srt.weg2 import corridor_arm as ca

    text = _read("name_compat_1a", "h91v1.front.lines")
    old = ca.arm_report(_write(tmp_path, "o.front.log", text))
    new = ca.arm_report(_write(tmp_path, "n.front.log", to_new(text)))
    assert old.samples > 0
    assert (new.samples, new.prose_mentions) == (old.samples, old.prose_mentions)


def test_nf_burst_probe_patterns():
    from sglang.srt.weg2.tools import nf_burst_probe as bp

    lines = _read("name_compat_1a", "h91v1.front.lines").splitlines()
    for rx, key in ((bp.RE_SERVED_P, "group=P leg=1"), (bp.RE_SERVED_D, "group=D leg=2"), (bp.RE_FLIP, "-FLIP begin")):
        line = next(l for l in lines if key in l)
        m_old, m_new = rx.search(line), rx.search(to_new(line))
        assert m_old and m_new and m_new.groups() == m_old.groups()


def test_vram_hires_report_readers(tmp_path):
    from sglang.srt.weg2.tools import vram_hires_report as vr

    d = _read("name_compat_1a", "h91v1.D.lines")
    d_old, d_new = _write(tmp_path, "o.D.log", d), _write(tmp_path, "n.D.log", to_new(d))
    old = vr.peak_lines(d_old)
    assert old and vr.peak_lines(d_new) == old
    old = vr.private_free_by_rank(d_old)
    assert old and vr.private_free_by_rank(d_new) == old
    f = _read("name_compat_1a", "h91v1.front.lines")
    old = vr.flip_windows(_write(tmp_path, "o.front.log", f))
    new = vr.flip_windows(_write(tmp_path, "n.front.log", to_new(f)))
    assert old
    assert [vars(w) for w in new] == [vars(w) for w in old]
