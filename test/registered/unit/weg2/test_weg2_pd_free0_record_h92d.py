# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H92d -- free beim Flip-Start P->D aus der juengsten Messung derselben Form.

DER BEFUND (H92c-Agent, Metall): der P->D-Kreditplan (``plan_wake_credit_pd``)
rechnete free beim Flip-Start aus EINER Zeitreferenz (fnFL2x158/1, 24.09.) plus
Pufferregel. bb2 druckte ``card2 ... free 9013``, das Metall hatte am ersten
Wake P->D 8629 (x178: 8735); card0 7599 gegen 7343; card1 9777 gegen 10862.
Die engste gerechnete Luft auf card2 war 8 MiB -- eine Luft, die es nicht gab.

DER POSTEN: der schlafende D-Mitbewohner. ``other processes hold`` in P's
DC-BREAKDOWN am Flip-Start x178 -> bb2: +246/+102/+108 MiB (nvml1/0/2); D's
Schlafrest (``WEG2-SLEEP-RESIDUE sleep=1 nvml_proc_used``) 1698/768/766 ->
1944/872/874 = +246/+104/+108; fremde Kontexte 718/554/548 in beiden. Der
Sprung liegt zwischen x178 (D 1 Sitz, Verify-Graph bs=[1]) und h91v1 (D 6 Sitze
seit H91b, bs=[1..6]) und steht seitdem (bb1, bb2, bb3 identisch). Gegen x158/1
kommt P's Allokator-Rest dazu (PP2 torch_untagged 288 -> 628).

DER FIX: free0 ist eine MESSUNG je Form (FORM_KEYS) UND D-Sitzzahl, die
juengste gilt (Sidecar-RECORD oder BUILTIN ``FREE0_RECORDS``), ohne Messung
BENANNT die Zeitreferenz. Die Front schreibt den Record an ihren ersten zwei
Wakes P->D (nach dem Flip, im H78-Schreibthread).

``fixtures/pd_free0_h92d/`` sind woertliche Zeilen aus
/spinning/evidence-665-f1/boot_weg2_fnFL2{x158,x178,h91v1,h91bb1,h91bb2,h91bb3}_*.
"""

import inspect
import json
import os
import shutil
import types

import pytest

from sglang.srt.planner import expert_residency as er
from sglang.srt.weg2 import wake_credit as wc
from sglang.srt.weg2 import wake_credit_pd as pd
from sglang.srt.weg2 import wake_credit_pd_refs as refs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "pd_free0_h92d")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
P_CARD = (1, 0, 2)
SLOT_MIB = 1297637376 / 512 / (1 << 20)
SPLIT, CHUNK, N_LAYERS = (29, 11, 8), 3, 48
#: bb1/bb2/bb3: FR_P 0.324/0.637/0.733887 -> P-Zeilen 198/359/408, FR_D Kante K
FR_P_BB = (0.324, 0.637, 0.733887)
FR_D_K = (0.06, 0.51, 0.48)
E_D, S_D = (193, 145, 177), (100, 48, 48)
COMMIT = {"x158": "c01951e3e1", "x178": "b89592806a", "h91v1": "39fd662d9e",
          "h91bb1": "e17bd548b5", "h91bb2": "e17bd548b5", "h91bb3": "50cd2884ac"}
#: Launch-Zeitpunkte (Front-Log-Name) -- "was wusste der Planer dieses Boots"
#: die Basis kennt die Records nicht -- ihre Tests scheitern dann an den Zahlen, nicht am Import
RECS = list(getattr(refs, "FREE0_RECORDS", []))
LAUNCH = {"h91bb2": "2026-09-26 14:22:51", "h91bb3": "2026-09-26 15:38:31"}


def _txt(boot, kind):
    with open(os.path.join(FIX, "fnFL2%s.%s.lines" % (boot, kind))) as fh:
        return fh.read()


def _metal(boot, flip):
    """driver_free am Start des ``flip``-ten Wakes P->D, direkt aus der
    Front-Zeile (unabhaengig vom Code unter Test)."""
    import ast
    import re

    rows = re.findall(r"WEG2-FLIP-ORDER epoch=\d+ src=P driver_free=(\{[^}]*\})",
                      _txt(boot, "front"))
    return {int(k): float(v) for k, v in ast.literal_eval(rows[flip]).items()}


def _as_of(when):
    return [r for r in RECS if str(r["at"]) < when]


def _plan(**kw):
    """Der Plan der bb-Form (P-Zeilen 198/359/408, D Kante K). Kennt die Basis
    ein Argument nicht, faellt es weg -- die Basis rechnet dann, wie sie es
    immer tat, und die Zahlen-Assertions sagen, wo sie das Metall verfehlt."""
    p_rows = [er.buffer_rows(local_experts=512, fraction=f, scratch_rows=32) for f in FR_P_BB]
    d_rows = [er.buffer_rows(local_experts=e, fraction=f, scratch_rows=s)
              for e, f, s in zip(E_D, FR_D_K, S_D)]
    args = dict(
        model=MODEL, p_split=SPLIT, chunk_layers=CHUNK, n_layers=N_LAYERS, p_card=P_CARD,
        d_ratio="183,137,168", draft_on_p=False, p_rows=p_rows, d_rows=d_rows,
        slot_mib=SLOT_MIB, label="D", apply=False, dense_repack=True,
        p_resident=[er.resident_rows(512, f) for f in FR_P_BB],
        d_resident=[er.resident_rows(e, f) for e, f in zip(E_D, FR_D_K)], **kw)
    have = inspect.signature(pd.plan_wake_credit_pd).parameters
    return pd.plan_wake_credit_pd(**{k: v for k, v in args.items() if k in have})


def _main_free(plan):
    out = {}
    for ln in plan.lines:
        if " P->D D card" in ln and ": free " in ln:
            card = int(ln.split(" P->D D card")[1].split(" ")[0])
            out[card] = float(ln.split(": free ")[1].split(" ")[0])
    return out


def _first_free(plan):
    (ln,) = [x for x in plan.lines if "Vergleichsflip" in x and "gegeben" in x]
    out = {}
    for part in ln.split("Kreditwarten je Karte ")[1].split("), card"):
        card = int(part.split(" ")[0].replace("card", ""))
        out[card] = float(part.split("free ")[1].split(",")[0]) if "free " in part else None
    return out


# --------------------------------------------------------------------------- Messung


@pytest.mark.parametrize("i", range(max(1, len(RECS))))
def test_the_builtin_free0_records_are_the_logs_own_measurement(i):
    assert RECS, "keine eingebauten free0-Messungen"
    rec = RECS[i]
    boot, flip = rec["source"][len("fnFL2"):].split("/")
    got = pd.pd_free0_from_logs(_txt(boot, "P"), _txt(boot, "D"), _txt(boot, "front"),
                                source=rec["source"], flip=int(flip))
    assert dict(got, form_key="fnFL2x158", commit=COMMIT[boot]) == rec


def test_the_step_is_the_sleeping_d_co_tenant_and_it_came_with_the_seats():
    """x178 -> bb2 am ersten Wake P->D: was P als 'other processes' sieht, ist
    um genau den Schlafrest des D-Rangs derselben Karte gewachsen."""
    import re

    rx_o = re.compile(r"(PP\d)\] WEG2-DC-BREAKDOWN stage=release tags=\['kv_cache', "
                      r"'cuda_graph'\].*?other processes hold (\d+) MiB")
    rx_r = re.compile(r"(TP\d)\] WEG2-SLEEP-RESIDUE sleep=1 .*?nvml_proc_used=(\d+) MiB")

    def posts(boot):
        oth, res = {}, {}
        for m in rx_o.finditer(_txt(boot, "P")):
            oth.setdefault(int(m.group(1)[2:]), int(m.group(2)))
        for m in rx_r.finditer(_txt(boot, "D")):
            res.setdefault(int(m.group(1)[2:]), int(m.group(2)))
        return oth, res

    (o178, r178), (obb2, rbb2) = posts("x178"), posts("h91bb2")
    assert [obb2[s] - o178[s] for s in range(3)] == [246, 102, 108]
    assert [rbb2[s] - r178[s] for s in range(3)] == [246, 104, 108]
    assert all(abs((obb2[s] - o178[s]) - (rbb2[s] - r178[s])) <= 2 for s in range(3))
    # the rest (foreign contexts) did not move: 718/554/548 -> 718/552/548
    assert all(abs((o178[s] - r178[s]) - (obb2[s] - rbb2[s])) <= 2 for s in range(3))
    for boot in ("x158", "x178"):
        assert "bs=[1]," in _txt(boot, "D")
    for boot in ("h91v1", "h91bb1", "h91bb2", "h91bb3"):
        assert "bs=[1, 2, 3, 4, 5, 6]," in _txt(boot, "D")
        assert posts(boot)[1] == {0: 1944, 1: 872, 2: 874}


# --------------------------------------------------------------------------- Plan gegen Metall


def test_the_bb2_plan_meets_the_bb2_metal():
    """Was bb2's Planer beim Start wusste (Records vor 14:22:51): der erste Wake
    P->D (Vergleichsflip /0) aus bb1/0 -- dieselbe Form, dieselben Zeilen --
    ist bb2's Metall auf das MiB; die Hauptrechnung /1 aus h91v1/1 nimmt auf
    nvml2 8721 statt 9013."""
    plan = _plan(d_seats=6, free0_builtin=_as_of(LAUNCH["h91bb2"]))
    metal = _metal("h91bb2", 0)
    first = _first_free(plan)
    for card in (0, 1, 2):
        assert first[card] is not None and abs(first[card] - metal[card]) <= 2, (card, first, metal)
    main = _main_free(plan)
    assert main[2] == 8721 and main[2] < 9013
    assert any("H92d free0 /1 BUILTIN fnFL2h91v1/1" in ln for ln in plan.lines)
    assert any("free0 /0 BUILTIN fnFL2h91bb1/0" in ln for ln in plan.lines)


def test_the_bb3_plan_meets_the_bb3_metal_on_both_flips():
    """bb3 (dieser Baum, 50cd2884ac) vom Stand seines Starts aus geplant: /0 aus
    bb2/0, /1 aus h91v1/1 + Pufferregel. Die 3080er (nvml0/nvml2, die knappen)
    treffen beide Flips auf <= 12 MiB; die Basis lag 178/294 MiB darueber.
    nvml1 /1: die Pufferregel auf PP0 (h91v1 202 -> bb3 198 Zeilen = 280 MiB
    gerechnet, 232 gemessen) laesst 48 MiB -- ein eigener Posten, benannt."""
    plan = _plan(d_seats=6, free0_builtin=_as_of(LAUNCH["h91bb3"]))
    m0, m1 = _metal("h91bb3", 0), _metal("h91bb3", 1)
    first, main = _first_free(plan), _main_free(plan)
    for card in (0, 2):
        assert abs(main[card] - m1[card]) <= 12, (card, main, m1)
        assert first[card] is not None and abs(first[card] - m0[card]) <= 12, (card, first, m0)
    assert abs(first[1] - m0[1]) <= 12
    assert 0 < main[1] - m1[1] <= 64


def test_the_release_plan_reads_its_own_commit():
    """Mit allen eingebauten Records rechnet dieser Baum aus bb3 selbst."""
    plan = _plan(d_seats=6)
    assert _main_free(plan) == {1: 10802.0, 0: 7421.0, 2: 8719.0}
    assert _first_free(plan) == {1: 10860.0, 0: 7333.0, 2: 8627.0}
    assert any("H92d free0 /1 BUILTIN fnFL2h91bb3/1 (2026-09-26 15:45:45,961, 50cd2884ac"
               in ln for ln in plan.lines)


def test_the_sleepers_staging_comes_back_at_its_leg_end():
    """Mit ehrlichem free haette der Ring bis zum Simulationsende (zwei c2-IPC-
    Stagings 2 x 617 MiB auf nvml2) einen Stillstand gerechnet, den das Metall
    nie hatte: bb3 flippte viermal, D TP2 bekam weights_8 nach 187-280 ms am
    Leg-Ende von PP2 (H111d pollt durch genau diese Rueckgabe). Seit H92d gibt
    der Schlaefer sein Staging mit dem Leg-Ende zurueck: der Plan steht nicht,
    weights_8 auf nvml2 wartet bis nach PP2s letzter Veroeffentlichung."""
    plan = _plan(d_seats=6)
    assert plan.refusal is None
    (c2,) = [ln for ln in plan.lines if " P->D D card2 D TP2/P PP2: " in ln]
    assert "-> FERTIG" in c2
    first = [ln for ln in plan.lines if "Vergleichsflip" in ln and "gegeben" in ln][0]
    assert "STEHT" not in first
    main = pd.planned_reference_pd(
        pd.reference_from_dict(refs.REFERENCES["fnFL2x158/1"]),
        p_rows=[198, 359, 408], d_rows=[112, 122, 133], slot_mib=SLOT_MIB, p_split=SPLIT,
        chunk_layers=CHUNK, n_layers=N_LAYERS, p_resident=[166, 327, 376],
        d_resident=[12, 74, 85], free0=[r for r in RECS
                                        if r["source"] == "fnFL2h91bb3/1"][0])
    run = pd.simulate_pd(main)
    assert run.complete
    last_pp2 = max(ms for (s, _t), ms in run.publish_ms.items() if s == 2)
    g8 = run.tag(2, "weights_8")
    assert g8.waited_ms > 0 and g8.grant_ms >= last_pp2


# --------------------------------------------------------------------------- Regel


def test_no_seats_named_prices_as_before_and_says_nothing():
    plan = _plan()
    assert _main_free(plan)[2] == 9013
    assert not any("H92d" in ln for ln in plan.lines)


def test_an_unmeasured_seat_count_is_named_with_the_command_that_closes_it():
    plan = _plan(d_seats=3)
    assert _main_free(plan)[2] == 9013
    (head,) = [ln for ln in plan.lines if "H92d" in ln]
    assert "free0 /1 UNGEMESSEN fuer Form fnFL2x158 mit 3 D-Sitz(en)" in head
    assert "weg2.tools.pd_free0_record" in head


def _sidecar_rec(at, free2, model=MODEL, flip=1, seats=6):
    return {"kind": "pd_free0", "group": "PD_FREE0", "form_key": "fnFL2x158",
            "d_seats": seats, "flip": flip, "at": at, "source": "fnFL2hX/%d" % flip,
            "free": {"0": 7421.0, "1": 10802.0, "2": free2}, "p_rows": [198, 359, 408],
            "model": model, "commit": "deadbeef00", "boot_tag": "fnFL2hX"}


def test_a_newer_sidecar_record_wins_and_a_tie_goes_to_the_record(tmp_path):
    newer = _sidecar_rec("2026-09-27 00:00:00,000", 8600.0)
    plan = _plan(d_seats=6, free0_records=[newer])
    assert _main_free(plan)[2] == 8600
    assert any("free0 /1 RECORD fnFL2hX/1" in ln for ln in plan.lines)
    tie = _sidecar_rec("2026-09-26 15:45:45,961", 8700.0)
    assert _main_free(_plan(d_seats=6, free0_records=[tie]))[2] == 8700
    older = _sidecar_rec("2026-09-20 00:00:00,000", 8500.0)
    assert _main_free(_plan(d_seats=6, free0_records=[older]))[2] == 8719
    # sidecar round trip through the one writer the front uses
    from sglang.srt.weg2 import host_ledger

    path = str(tmp_path / "rec.json")
    host_ledger.append_measured_record(path, newer)
    host_ledger.append_measured_record(path, {"group": "D", "rss_shmem_gib": 1.0})
    got = pd.read_free0_records(path)
    assert len(got) == 1 and _main_free(_plan(d_seats=6, free0_records=got))[2] == 8600
    assert pd.read_free0_records(str(tmp_path / "missing.json")) == []
    # the other sidecar readers never see the entry
    assert "PD_FREE0" not in host_ledger.read_measured_record(path)


def test_a_record_of_another_checkpoint_is_not_this_forms_measurement(tmp_path):
    other = _sidecar_rec("2026-09-27 00:00:00,000", 8000.0, model="Some-Other-Model")
    plan = _plan(d_seats=6, free0_records=[other])
    assert _main_free(plan)[2] == 8719


# --------------------------------------------------------------------------- Schreiber


def _front(meta, record="/x/rec.json"):
    from sglang.srt.weg2 import front

    f = object.__new__(front.Front)
    f.wake_credit_plan = {"P->D-free0": meta} if meta else {}
    f.tag, f.commit, f.measured_record = "fnFL2hT", "abcdef0123", record
    sent = []
    f._sidecar_submit = lambda fn, *a: sent.append((fn, a))
    return f, sent


def test_the_front_records_the_first_two_p_to_d_wakes_after_the_flip():
    from sglang.srt.weg2 import host_ledger

    meta = {"form_key": "fnFL2x158", "d_seats": 6, "p_rows": [198, 359, 408], "model": MODEL}
    f, sent = _front(meta)
    free = {0: 7343, 1: 10862, 2: 8629}
    f._note_pd_free0("D", "P", {0: 3357, 1: 8246, 2: 2225})   # D->P: nothing
    f._note_pd_free0("P", "D", free)
    assert sent == []                                           # not on the flip it measures
    f._flush_pd_free0()
    ((fn, (path, rec)),) = sent
    assert fn is host_ledger.append_measured_record and path == "/x/rec.json"
    assert rec["flip"] == 0 and rec["free"] == {0: 7343.0, 1: 10862.0, 2: 8629.0}
    assert rec["d_seats"] == 6 and rec["form_key"] == "fnFL2x158" and rec["source"] == "fnFL2hT/0"
    f._note_pd_free0("P", "D", free)
    f._flush_pd_free0()
    f._note_pd_free0("P", "D", free)
    f._flush_pd_free0()
    assert [s[1][1]["flip"] for s in sent] == [0, 1]
    f2, sent2 = _front(None)
    f2._note_pd_free0("P", "D", free)
    f2._flush_pd_free0()
    assert sent2 == []


def test_the_plan_hands_the_front_its_form():
    plan = _plan(d_seats=6)
    assert plan.front_plan["P->D-free0"] == {
        "form_key": "fnFL2x158", "d_seats": 6, "p_rows": [198, 359, 408], "model": MODEL}
    assert "P->D-free0" not in (_plan().front_plan or {})


def test_the_tool_reads_a_boot_into_the_same_record(tmp_path):
    from sglang.srt.weg2.tools import pd_free0_record as tool

    stem = tmp_path / "boot_weg2_fnFL2h91bb2_e17bd548b5_0926_142251"
    for kind in ("front", "P", "D"):
        shutil.copy(os.path.join(FIX, "fnFL2h91bb2.%s.lines" % kind), "%s.%s.log" % (stem, kind))
    (rec,) = tool.records_from_boot(str(stem) + ".front.log")
    want = [r for r in RECS if r["source"] == "fnFL2h91bb2/0"][0]
    assert {k: rec[k] for k in want} == want
    assert rec["model"] == MODEL and rec["boot_tag"] == "fnFL2h91bb2"
    tool.main([str(stem) + ".front.log", "--append"])
    got = pd.read_free0_records(str(tmp_path / "weg2_measured_record.json"))
    assert len(got) == 1 and got[0]["source"] == "fnFL2h91bb2/0"


# --------------------------------------------------------------------------- Launcher


def test_the_launcher_names_the_seats_and_reads_the_sidecar(tmp_path, monkeypatch):
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import launcher

    monkeypatch.setattr(
        pp_cut, "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(expert_layer_weight_bytes=1297637376.0,
                                         num_experts=512, n_layers=48))
    sidecar = tmp_path / "weg2_measured_record.json"
    sidecar.write_text(json.dumps({"samples": [_sidecar_rec("2026-09-27 00:00:00,000", 8600.0)]}))
    monkeypatch.setattr(launcher, "measured_record_path", lambda: str(sidecar))
    cfg = {"text_config": {"num_hidden_layers": 48, "vocab_size": 248320, "hidden_size": 2560}}
    (tmp_path / MODEL).mkdir()
    (tmp_path / MODEL / "config.json").write_text(json.dumps(cfg))
    ns = types.SimpleNamespace(
        model=str(tmp_path / MODEL), d_bs=6,
        extra_d=("--rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
                 "--rank-moe-resident-fraction 0.06,0.51,0.48 --max-running-requests 6"),
        env_d=("SGLANG_MOE_POOL_STAGING=8;SGLANG_MOE_SCRATCH_SLOTS=82,48,48;"
               "SGLANG_UNEVEN_MOE_EXPERT_SHARD=1;SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL=1"),
        extra_p="--rank-moe-resident-fraction 0.324,0.637,0.733887",
        env_p="SGLANG_MOE_SCRATCH_SLOTS=32",
        pp_cut_expert_device_fraction="",
        d_foreign_context_mib="", d_nontorch_mib="", d_reserve_mib="",
        d_residency_reference_logs="", wake_credit_reference_logs="",
    )
    cards = [launcher.Card(1, "u1", "RTX 5090", 32607),
             launcher.Card(0, "u0", "RTX 3080", 20480),
             launcher.Card(2, "u2", "RTX 3080", 20480)]
    from sglang.srt.environ import envs

    lines = []
    with envs.SGLANG_WEG2_DRAFT_ON_P.override(False):   # H25-Form (bb2/bb3), kein Draft auf P
        launcher.log_d_rank_vram_solve(ns, cards, (29624, 18664, 18672), lines.append,
                                       "D(dry, expectation)", p_split=[29, 11, 8], chunk_layers=3)
    pdl = [ln for ln in lines if ln.startswith(wc.MARKER + " P->D ")]
    assert any("H92d free0 /1 RECORD fnFL2hX/1" in ln for ln in pdl), pdl[:1]
    assert any(" card2 D TP2/P PP2: free 8600 " in ln for ln in pdl), pdl
    assert launcher._WAKE_CREDIT_FRONT_PLAN["P->D-free0"]["d_seats"] == 6
