"""fnFL2 H55: VRAM at 5 ms instead of 1 s -- the NVML probe, its reader, and
the in-process WEG2-VRAM-PEAK windows.

User 24.09. 18:20Z: "kannst du keine sonde bauen die das wirklich mitschneidet
wie viel vram tatsaechlich belegt ist? ... so hochaufgeloest, dass wir daraus
auch wirklich schluesse ziehen koennen?" -- ``nvidia-smi dmon -d 1`` sees one
value per ~1.1 s; the planner's 3697 MiB chunk transient and x149's 338 MiB
headroom live between those reads.

All hermetic: a fake NVML (scripted fb-used per card and per pid) on a fake
monotonic clock, a fake /proc tree, a fake torch.cuda allocator. No GPU, no
NVML library, no CUDA.
"""

import json
import os
import re
import tempfile
import unittest

import pytest

try:
    from sglang.srt.environ import envs
    from sglang.srt.model_executor import vram_family_census as vfc
    from sglang.srt.model_executor import vram_peak_window as vpw
    from sglang.srt.weg2.tools import vram_hires as vh
    from sglang.srt.weg2.tools import vram_hires_report as vr
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

MIB = 1 << 20


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeClock(vh.Clock):
    """Monotonic ns that only moves when the probe sleeps or 'spends' time in
    a call; wall = t0 + mono."""

    def __init__(self, wall0_ms=1_790_270_400_000):
        self.ns = 0
        self.wall0_ms = wall0_ms

    def mono_ns(self):
        return self.ns

    def wall_ms(self):
        return self.wall0_ms + self.ns // 1_000_000

    def sleep(self, s):
        if s > 0:
            self.ns += int(s * 1e9)

    def spend_us(self, us):
        self.ns += int(us * 1000)


class FakeNvml:
    """``used(card, t_ms)`` scripted; every call costs ``call_us`` on the
    fake clock and is counted."""

    def __init__(self, clock, totals, used_fn, procs_fn=None, call_us=100.0, proc_call_us=300.0,
                 offset_mib=500):
        self.clock = clock
        self.totals = totals
        self.used_fn = used_fn
        self.procs_fn = procs_fn or (lambda card, t: [])
        self.call_us = call_us
        self.proc_call_us = proc_call_us
        self.offset = offset_mib
        self.mem_calls = 0
        self.proc_calls = 0

    def t_ms(self):
        return self.clock.ns // 1_000_000

    def name(self, i):
        return "NVIDIA GeForce RTX 5090" if i == 1 else "NVIDIA GeForce RTX 3080"

    def mem(self, i):
        self.clock.spend_us(self.call_us)
        self.mem_calls += 1
        used = self.used_fn(i, self.t_ms()) * MIB
        total = self.totals[i] * MIB
        return used, total - used - self.offset * MIB, total

    def procs(self, i):
        self.clock.spend_us(self.proc_call_us)
        self.proc_calls += 1
        return [(pid, mib * MIB if mib is not None else None) for pid, mib in self.procs_fn(i, self.t_ms())]


TOTALS = {0: 20480, 1: 32607, 2: 20480}


def _spike_used(card, t):
    """card 1: 28000 MiB, a 3700-MiB chunk transient for 15 ms at t=1234 ms
    and a 40-ms one at 2500 ms; the other cards flat."""
    if card == 1:
        if 1234 <= t < 1249:
            return 31700
        if 2500 <= t < 2540:
            return 31000
        return 28000
    return 12000


def _run_probe(tmp, used_fn=_spike_used, procs_fn=None, ticks=700, call_us=100.0, proc_call_us=300.0,
               period_ms=5.0, raw=True, resolver=None):
    clock = FakeClock()
    src = FakeNvml(clock, TOTALS, used_fn, procs_fn, call_us=call_us, proc_call_us=proc_call_us)
    out = os.path.join(tmp, "vram_hires_fnFL2xT.csv")
    probe = vh.Probe(src, [0, 1, 2], out, vh.raw_path_for(out) if raw else None, period_ms=period_ms,
                     resolver=resolver or vh.RoleResolver(proc_root=os.path.join(tmp, "noproc")),
                     clock=clock, calib_ticks=20)
    probe.run(max_ticks=ticks)
    return probe, src, out


def _read(p):
    with open(p) as f:
        return f.read()


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------


class TestProbeResolution(unittest.TestCase):
    def test_15ms_spike_is_in_the_raw_stream_and_the_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe, src, out = _run_probe(tmp)
            raw = _read(vh.raw_path_for(out))
            rows = [ln.split(",") for ln in raw.splitlines() if ln and not ln.startswith("#")][1:]
            c1 = [(int(t), int(v)) for t, k, v in rows if k == "c1"]
            self.assertEqual(max(v for _, v in c1), 31700)
            t_spike = [t for t, v in c1 if v == 31700]
            self.assertTrue(t_spike and 1230 <= t_spike[0] <= 1250, t_spike)
            # change-only: the 28000 plateau is not written 200x per second
            self.assertLess(len(c1), 40)
            buckets = [ln.split(",") for ln in _read(out).splitlines()
                       if ln and not ln.startswith("#") and not ln.startswith("utc")]
            b1 = [b for b in buckets if b[2] == "c1"]
            self.assertEqual(max(int(b[5]) for b in b1), 31700)
            # free_min = total - offset - max
            spike_bucket = [b for b in b1 if int(b[5]) == 31700][0]
            self.assertEqual(int(spike_bucket[7]), 32607 - 500 - 31700)
            self.assertGreater(int(spike_bucket[8]), 150)  # ~200 samples in that second

    def test_one_second_view_misses_what_the_probe_sees(self):
        """dmon's shape: one read per second on the integer second -- the
        15-ms and 40-ms transients are both invisible to it."""
        dmon_reads = [_spike_used(1, s * 1000) for s in range(4)]
        self.assertEqual(max(dmon_reads), 28000)
        with tempfile.TemporaryDirectory() as tmp:
            _, _, out = _run_probe(tmp)
            raw = _read(vh.raw_path_for(out))
            self.assertIn(",c1,31700", raw)
            self.assertIn(",c1,31000", raw)

    def test_header_names_rate_and_call_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, out = _run_probe(tmp, call_us=100.0)
            head = _read(out)
            m = re.search(r"# calib rate_hz=(\d+) period_ms=([\d.]+) .*mean_call_us=([\d.]+) p99_call_us=(\d+)", head)
            self.assertIsNotNone(m, head[:600])
            self.assertEqual(int(m.group(1)), 200)
            self.assertEqual(float(m.group(2)), 5.0)
            self.assertAlmostEqual(float(m.group(3)), 100.0, delta=1.0)
            self.assertIn("# t0_unix_ms=1790270400000", head)
            self.assertIn("# card idx=1 name=NVIDIA_GeForce_RTX_5090 total_mib=32607 offset_mib=500", head)
            self.assertRegex(head, r"# end t_ms=\d+ ticks=700 rate_hz=")

    def test_overhead_budget_calls_per_second(self):
        """3 cards at 5 ms = 600 MemoryInfo calls/s, processes every 10 ms =
        300 calls/s; at 100 us + 300 us per call the probe is busy
        (3*100 + 3*300/2) us per 5 ms = 15 % of one core, and the rate holds."""
        with tempfile.TemporaryDirectory() as tmp:
            probe, src, _ = _run_probe(tmp, ticks=400)
            secs = probe.ticks * probe.period_ms / 1000.0
            mem_per_s = (src.mem_calls - 20 * 3 - 3) / secs
            proc_per_s = (src.proc_calls - 3) / secs
            self.assertAlmostEqual(mem_per_s, 600, delta=10)
            self.assertAlmostEqual(proc_per_s, 300, delta=10)
            line = probe.stat_line(int(secs * 1000))
            busy = float(re.search(r"busy_pct=([\d.]+)", line).group(1))
            self.assertLess(busy, 20.0)
            self.assertEqual(probe.late, 0)
            self.assertIn("rate_hz=200", probe.header_lines[-1])

    def test_period_is_raised_when_nvml_is_slow_and_the_header_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe, _, out = _run_probe(tmp, call_us=2000.0, ticks=50)
            self.assertGreater(probe.period_ms, 5.0)
            self.assertIn("period_raised_from=5", _read(out))
            # the tick cost stays within the budget share of the new period
            self.assertLessEqual(3 * 2.0 / probe.period_ms, vh.DEFAULT_BUDGET + 1e-9)

    def test_no_raw_writes_buckets_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, out = _run_probe(tmp, raw=False, ticks=250)
            self.assertFalse(os.path.exists(vh.raw_path_for(out)))
            self.assertIn(",c1,card,", _read(out))

    def test_process_rows_and_other(self):
        def procs(card, t):
            if card == 1:
                return [(4242, 26000 if t < 2000 else 27500), (999, None)]
            return []

        with tempfile.TemporaryDirectory() as tmp:
            _, _, out = _run_probe(tmp, procs_fn=procs, ticks=500)
            raw = _read(vh.raw_path_for(out))
            self.assertIn("# proc t_ms=", raw)
            self.assertIn("key=c1:4242 pid=4242 card=1 role=?:nopid", raw)
            self.assertIn(",c1:4242,26000", raw)
            self.assertIn(",c1:4242,27500", raw)
            # other = card - attributed (pid 999 has no used value: not attributed)
            self.assertIn(",c1:other,2000", raw)
            self.assertIn(",c0:other,12000", raw)


class TestBuckets(unittest.TestCase):
    def test_min_max_last_n_per_unix_second(self):
        agg = vh.BucketAgg(lambda key, mx: 100 - mx if key == "c0" else None)
        rows = []
        for ms, v in ((1000, 5), (1400, 9), (1999, 7), (2000, 3)):
            rows += agg.add(1_790_000_000_000 + ms, "c0", v)
        self.assertEqual(len(rows), 1)
        utc, sec, key, role, mn, mx, last, free, n = rows[0].split(",")
        self.assertEqual((key, role, int(mn), int(mx), int(last), int(free), int(n)), ("c0", "card", 5, 9, 7, 91, 3))
        tail = agg.flush()
        self.assertEqual(tail[0].split(",")[4:7], ["3", "3", "3"])

    def test_change_writer_keepalive(self):
        w = vh.ChangeWriter(keepalive_ms=1000)
        got = [w.feed(t, "c1", v) for t, v in ((0, 5), (5, 5), (10, 6), (15, 6), (1010, 6), (1015, 6))]
        self.assertEqual(got, ["0,c1,5", None, "10,c1,6", None, "1010,c1,6", None])

    def test_call_stats_percentile(self):
        cs = vh.CallStats()
        for _ in range(99):
            cs.add(100.0)
        cs.add(5000.0)
        self.assertAlmostEqual(cs.mean_us, 149.0)
        self.assertEqual(cs.pct_us(0.5), 110.0)
        self.assertEqual(cs.max_us, 5000.0)


# --------------------------------------------------------------------------
# role attribution and lifetime
# --------------------------------------------------------------------------


def _mkproc(root, pid, cmdline, ppid, comm=None):
    d = os.path.join(root, str(pid))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cmdline"), "wb") as f:
        f.write(cmdline.replace(" ", "\0").encode())
    with open(os.path.join(d, "status"), "w") as f:
        f.write(f"Name:\t{comm or cmdline[:15]}\nPPid:\t{ppid}\n")
    with open(os.path.join(d, "comm"), "w") as f:
        f.write((comm or cmdline.split()[0])[:15] + "\n")


class TestRoles(unittest.TestCase):
    def _tree(self, tmp):
        proc = os.path.join(tmp, "proc")
        _mkproc(proc, 100, "python -m sglang.srt.weg2.launcher --tree /t --tag fnFL2x999 --profile nextflash", 1)
        _mkproc(proc, 200, "python -m sglang.launch_server --model-path /m", 100)
        _mkproc(proc, 201, "sglang::scheduler_PP0", 200)
        _mkproc(proc, 300, "python -m sglang.launch_server --model-path /m", 100)
        _mkproc(proc, 302, "sglang::scheduler_TP1", 300)
        _mkproc(proc, 400, "python other_tool.py", 1, comm="other_tool")
        with open(os.path.join(tmp, "boot_fnFL2x999.json"), "w") as f:
            json.dump({"tag": "fnFL2x999", "pids": {"P": 200, "D": 300}}, f)
        return proc

    def test_scheduler_titles_groups_and_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._tree(tmp)
            rr = vh.RoleResolver(proc_root=proc, boot_json_dir=tmp)
            self.assertEqual(rr.role(201), ("P:PP0", "fnFL2x999"))
            self.assertEqual(rr.role(302), ("D:TP1", "fnFL2x999"))
            self.assertEqual(rr.role(400)[0], "?:other_tool")
            self.assertEqual(rr.role(777)[0], "?:nopid")

    def test_boot_json_group_beats_the_title_guess(self):
        """A TP-titled rank under the P leader is P (the json names leaders)."""
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._tree(tmp)
            _mkproc(proc, 203, "sglang::scheduler_TP0", 200)
            rr = vh.RoleResolver(proc_root=proc, boot_json_dir=tmp)
            self.assertEqual(rr.role(203)[0], "P:TP0")

    def test_owner_alive_and_watch(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._tree(tmp)
            pat = vh.default_owner_pattern("fnFL2x999")
            self.assertTrue(vh.owner_alive(pat, proc, self_pid=-1))
            self.assertFalse(vh.owner_alive(vh.default_owner_pattern("fnFL2x99"), proc, self_pid=-1))
        state = {"alive": False}
        w = vh.OwnerWatch("x", poll_s=5, grace_s=900, gone_s=60, max_s=3600, alive=lambda: state["alive"])
        self.assertIsNone(w(0))
        self.assertIsNone(w(600_000))           # not yet seen, inside grace
        state["alive"] = True
        self.assertIsNone(w(610_000))
        state["alive"] = False                  # dry-run launcher gone ...
        self.assertIsNone(w(620_000))
        self.assertIsNone(w(650_000))
        state["alive"] = True                   # ... boot launcher up 40 s later
        self.assertIsNone(w(660_000))
        state["alive"] = False
        self.assertIsNone(w(700_000))
        self.assertEqual(w(720_000), "owner_gone")
        w2 = vh.OwnerWatch("x", grace_s=900, alive=lambda: False)
        self.assertEqual(w2(900_000), "owner_never_seen")
        self.assertEqual(vh.OwnerWatch(None, max_s=10)(10_000), "max_s")


# --------------------------------------------------------------------------
# the reader
# --------------------------------------------------------------------------

DAY = "2026-09-24"
T0 = 1_790_270_400_000  # 2026-09-24 17:20:00Z


def _hms(ms):
    import datetime as dt

    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%H:%M:%S")


def _write_boot(tmp, with_peak_lines=True):
    """A probe trace plus P/D/front/dry/dmon logs of one fictitious boot:
    two P chunks on PP0 (nvml1), a flip P->D, D decode."""
    raw = os.path.join(tmp, "vram_hires_fnFL2xR.raw.csv")
    lines = [
        "# vram_hires v1 (fnFL2 H55)",
        f"# t0_unix_ms={T0} t0_utc=x pid=1 argv=x",
        "# card idx=0 name=RTX_3080 total_mib=20480 offset_mib=425 (free = total - offset - used)",
        "# card idx=1 name=RTX_5090 total_mib=32607 offset_mib=518 (free = total - offset - used)",
        "# card idx=2 name=RTX_3080 total_mib=20480 offset_mib=425 (free = total - offset - used)",
        "# calib rate_hz=200 period_ms=5 proc_every_ms=10 cards=3 mean_call_us=120.0 p99_call_us=300",
        "t_ms,key,used_mib",
        "0,c0,12000", "0,c1,28000", "0,c2,12500",
        "# proc t_ms=0 key=c1:201 pid=201 card=1 role=P:PP0 tag=fnFL2xR",
        "0,c1:201,27500",
        # chunk 0: 1000..4000 ms, transient 3700 for 20 ms at 2000 ms
        "2000,c1,31700", "2000,c1:201,31200", "2020,c1,28000", "2020,c1:201,27500",
        # chunk 1: 4000..8000, transient 3000 at 5000 for 300 ms
        "5000,c1,31000", "5300,c1,28300",
        # flip 10000..12400: card 0 up to 19800 for 50 ms at 11000
        "11000,c0,19800", "11050,c0,16700", "12000,c0,12000",
        # D decode 12400..20000: card 1 flat 30300, card 2 17900
        "12400,c1,30300", "12400,c2,17900",
        "# end t_ms=20000 ticks=4000 rate_hz=200.0",
    ]
    with open(raw, "w") as f:
        f.write("\n".join(lines) + "\n")
    csv = os.path.join(tmp, "vram_hires_fnFL2xR.csv")
    open(csv, "w").close()
    p = os.path.join(tmp, "boot_weg2_fnFL2xR_abc_0924_172000.P.log")
    with open(p, "w") as f:
        for i, (a, b, tr) in enumerate(((1000, 4000, 3700), (4000, 8000, 3000))):
            f.write(f"[{DAY} {_hms(T0 + a)} PP0] Prefill batch, #new-seq: 1, #new-token: 16384, #cached-token: 0\n")
            if with_peak_lines:
                f.write(f"[{DAY} {_hms(T0 + b)} PP0] WEG2-VRAM-PEAK rank=0 phase=chunk rows=16384 n=1 "
                        f"t0_unix_ms={T0 + a} t_unix_ms={T0 + b} window_ms={b - a} peak_allocated_mib={24000 + tr} "
                        f"peak_reserved_mib={27000 + tr} start_allocated_mib=24000 transient_mib={tr} "
                        f"allocated_mib=24000 reserved_mib=27000 card_free_start_mib=3500 card_free_mib=3600 "
                        f"card_total_mib=32088\n")
    d = os.path.join(tmp, "boot_weg2_fnFL2xR_abc_0924_172000.D.log")
    with open(d, "w") as f:
        f.write(f"[{DAY} {_hms(T0 + 13000)} TP0] Decode batch, #running-req: 1\n")
    front = os.path.join(tmp, "boot_weg2_fnFL2xR_abc_0924_172000.front.log")
    with open(front, "w") as f:
        f.write(f"[{DAY} {_hms(T0 + 10000)},000] INFO WEG2-FLIP begin epoch=1 sleep=P wake=D\n")
        f.write(f"[{DAY} {_hms(T0 + 12000)},400] INFO WEG2-FLIP done epoch=1 slept=P woke=D\n")
        f.write(f"[{DAY} {_hms(T0 + 20000)},000] INFO WEG2-FLIP begin epoch=2 sleep=D wake=P\n")
        f.write(f"[{DAY} {_hms(T0 + 21000)},000] INFO WEG2-FLIP done epoch=2 slept=D woke=P\n")
    dry = os.path.join(tmp, "dry_fnFL2xR.log")
    with open(dry, "w") as f:
        f.write("[x] WEG2-LAUNCH PP-CUT P-KARTE stage0 (nvml1) prompt=262144: Referenz ..., Transiente 3697 (+0, "
                "gemessen), LMEM 0 -> Kopfraum 2211 MiB (near-OOM 400) -> PASST\n")
        f.write("[x] WEG2-LAUNCH D-RANK VRAM (#145) KARTE D(dry, expectation) rang0: Referenz 94 Zeilen, "
                "-> Kopfraum 767 MiB (near-OOM 400), Decode frei\n")
        f.write("[x] WEG2-LAUNCH D-RANK VRAM (#145) KARTE D(dry, expectation) rang2: Referenz 1 Zeilen, "
                "-> Kopfraum 1443 MiB (near-OOM 400), Decode frei\n")
    dmon = os.path.join(tmp, "vram_fnFL2xR.log")
    with open(dmon, "w") as f:
        f.write("#Time         gpu  rxpci  txpci     fb   bar1   ccpm \n")
        for s in range(0, 21):
            ms = T0 + s * 1000
            for c, v in ((0, 12000), (1, 28000 if s < 12 else 30300), (2, 12500 if s < 12 else 17900)):
                if s == 11 and c == 0:
                    v = 16700
                f.write(f" {_hms(ms)}  {c}  0  0  {v}  2  0\n")
    return csv, p, d, front, dry, dmon


class TestReport(unittest.TestCase):
    def _rows(self, tmp, with_peak_lines=True):
        csv, p, d, front, dry, dmon = _write_boot(tmp, with_peak_lines)
        tr = vr.load_trace(vr.raw_path_of(csv))
        peaks = vr.peak_lines(p)
        flips = vr.flip_windows(front)
        wins = vr.p_chunk_windows(p, peaks) + flips + vr.decode_windows(flips)
        plan = vr.planner_card(dry, (1, 0, 2))
        crow = vr.card_rows(tr, wins, plan, vr.dmon_series(dmon, DAY), (1, 0, 2))
        return tr, crow, vr.report(csv, p, d, front=front, dry=dry, dmon=dmon, rank_cards=(1, 0, 2))

    def test_true_max_rest_and_what_dmon_missed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, crow, text = self._rows(tmp)
            c0 = [r for r in crow if r["phase"].startswith("P chunk 0") and r["card"] == 1][0]
            self.assertEqual(c0["true_max"], 31700)
            self.assertEqual(c0["rest_min"], 32607 - 518 - 31700)  # 389
            self.assertAlmostEqual(c0["at_s"], 1.0)
            self.assertEqual(c0["dmon_max"], 28000)
            self.assertEqual(c0["missed"], 3700)
            self.assertEqual(c0["plan"], 2211)
            self.assertEqual(c0["plan_minus_rest"], 2211 - 389)
            self.assertEqual(c0["verdict"], "UNTER near-OOM")
            self.assertEqual(c0["proc"], ("P:PP0", 31200))
            # chunk windows carry the rank: only PP0's card is read
            self.assertEqual({r["card"] for r in crow if r["phase"].startswith("P chunk")}, {1})
            fl = [r for r in crow if r["phase"] == "flip 1 P->D" and r["card"] == 0][0]
            self.assertEqual((fl["true_max"], fl["dmon_max"], fl["missed"]), (19800, 16700, 3100))
            self.assertEqual(fl["rest_min"], 20480 - 425 - 19800)
            dec = [r for r in crow if r["phase"] == "D decode 1" and r["card"] == 2][0]
            self.assertEqual((dec["true_max"], dec["plan"], dec["verdict"]), (17900, 1443, "ok"))
            self.assertIn("P chunk 0 PP0 rows=16384", text)
            self.assertIn("UNTER near-OOM", text)
            self.assertIn("IN-PROZESS", text)

    def test_inproc_table_joins_nvml_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv, p, d, front, dry, dmon = _write_boot(tmp)
            tr = vr.load_trace(vr.raw_path_of(csv))
            rows = vr.inproc_rows(tr, {"P": vr.peak_lines(p), "D": []}, vr.planner_card(dry, (1, 0, 2)), (1, 0, 2))
            r = [x for x in rows if x["phase"] == "chunk"][0]
            self.assertEqual((r["n"], r["peak_reserved"], r["transient"], r["plan_transient"]), (2, 30700, 3700, 3697))
            self.assertEqual(r["nvml_proc_max"], 31200)
            self.assertEqual(r["ausser_torch"], 31200 - 30700)
            self.assertEqual(r["card_free_min"], 3500)

    def test_h59_planner_definition_persist_and_torch_headroom(self):
        # fnFL2x164 PP0 chunk 0, verbatim numbers: H55 transient 4074 is the
        # planner's 3663 (peak - allocated AFTER) plus the 411 MiB the chunk
        # leaves behind; torch headroom = 105 + 28780 - 27307 - 216 = 1362
        # (= WEG2-GRAPH-POOL headroom_mib of the same chunk).
        line = ("[2026-09-24 19:15:41 PP0] WEG2-VRAM-PEAK rank=0 phase=chunk rows=16384 n=1 "
                "t0_unix_ms=1790277332153 t_unix_ms=1790277341963 window_ms=9811 "
                "peak_allocated_mib=27307 peak_reserved_mib=28868 start_allocated_mib=23233 "
                "transient_mib=4074 allocated_mib=23644 reserved_mib=28780 card_free_start_mib=5327 "
                "card_free_mib=105 card_total_mib=32088\n")
        pool = ("[2026-09-24 19:15:41 PP0] WEG2-GRAPH-POOL rank=0 phase=extend captured_mib=23156 "
                "private_free_mib=216 reserved_after_mib=28780 allocated_mib=23644 peak_mib=27307 "
                "card_free_mib=105 card_total_mib=32088 cap_mib=28885 headroom_mib=1362 pools=x\n")
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "boot_weg2_fnFL2x164_x_0924_191021.P.log")
            with open(p, "w") as f:
                f.write(pool + line)
            self.assertEqual(vr.private_free_by_rank(p), {"0": 216})
            rows = vr.inproc_rows(None, {"P": vr.peak_lines(p)}, {}, (1, 0, 2),
                                  private_free={"P": vr.private_free_by_rank(p)})
            r = [x for x in rows if x["phase"] == "chunk"][0]
            self.assertEqual((r["transient"], r["tr_plan"], r["persist"]), (4074, 3663, 411))
            self.assertEqual(r["kopf_torch_min"], 1362)
            # without the private-free term the headroom is not claimed
            bare = vr.inproc_rows(None, {"P": vr.peak_lines(p)}, {}, (1, 0, 2))
            self.assertIsNone([x for x in bare if x["phase"] == "chunk"][0]["kopf_torch_min"])

    def test_without_peak_lines_windows_fall_back_to_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, crow, _ = self._rows(tmp, with_peak_lines=False)
            ch = [r for r in crow if r["phase"].startswith("P chunk 0 ~s") and r["card"] == 1]
            self.assertEqual(ch[0]["true_max"], 31700)

    def test_sync_anchor_moves_the_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "vram_hires_x.raw.csv")
            with open(raw, "w") as f:
                f.write(f"# t0_unix_ms={T0} t0_utc=x\n# card idx=0 name=a total_mib=100 offset_mib=0 (x)\n"
                        f"t_ms,key,used_mib\n0,c0,10\n# sync t_ms=60000 unix_ms={T0 + 60250}\n61000,c0,90\n")
            tr = vr.load_trace(raw)
            self.assertEqual(tr.window_max("c0", T0 + 61250, T0 + 61250), (90, T0 + 61250))
            self.assertEqual(tr.window_max("c0", T0 + 61000, T0 + 61100)[0], 10)


# --------------------------------------------------------------------------
# in-process windows (WEG2-VRAM-PEAK)
# --------------------------------------------------------------------------


class FakeCuda:
    """Caching-allocator counters: current/peak for allocated and reserved;
    ``reset_peak_memory_stats`` re-bases the peaks to current (torch's rule)."""

    def __init__(self, total=32 << 30):
        self.a = self.r = self.pa = self.pr = 0
        self.total = total
        self.resets = 0

    def alloc(self, n):
        self.a += n
        self.r = max(self.r, self.a)
        self.pa = max(self.pa, self.a)
        self.pr = max(self.pr, self.r)

    def free(self, n):
        self.a -= n

    def memory_stats(self):
        return {"allocated_bytes.all.peak": self.pa, "reserved_bytes.all.peak": self.pr,
                "allocated_bytes.all.current": self.a, "reserved_bytes.all.current": self.r}

    def max_memory_allocated(self):
        return self.pa

    def reset_peak_memory_stats(self):
        self.resets += 1
        self.pa, self.pr = self.a, self.r

    def mem_get_info(self):
        return self.total - self.r - (500 << 20), self.total

    def is_current_stream_capturing(self):
        return False

    def memory_allocated(self):
        return self.a

    def memory_reserved(self):
        return self.r

    def memory_snapshot(self):
        raise RuntimeError("no snapshot in the fake")


class _Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_extend(self):
        return self.kind in ("extend", "verify")

    def is_decode(self):
        return self.kind == "decode"

    def is_target_verify(self):
        return self.kind == "verify"


class _Ids:
    def __init__(self, n):
        self.shape = (n,)


class _FB:
    def __init__(self, kind, n=1):
        self.forward_mode = _Mode(kind)
        self.input_ids = _Ids(n)


class _Runner:
    tp_rank = 0
    is_draft_model_runner = False


class TestPeakWindows(unittest.TestCase):
    def setUp(self):
        vpw.reset_since_pools()
        self.cuda = FakeCuda()
        self.cuda.alloc(20 << 30)
        self.cuda.reset_peak_memory_stats()  # "after pools"
        vpw.reset_since_pools()

    def _chunk(self, transient_mib):
        self.cuda.alloc(transient_mib << 20)
        self.cuda.free(transient_mib << 20)
        with self.assertLogs(vpw.logger, "INFO") as cm:
            line = vpw.on_forward_end(_Runner(), _FB("extend", 16384), self.cuda)
        self.assertIn(line, cm.output[-1])
        return dict(re.findall(r"(\w+)=(\S+)", line.split("WEG2-VRAM-PEAK ", 1)[1]))

    def test_every_chunk_gets_its_own_peak(self):
        a = self._chunk(3700)
        b = self._chunk(1200)
        self.assertEqual((a["phase"], a["rows"], a["peak_allocated_mib"]), ("chunk", "16384", str(20480 + 3700)))
        self.assertEqual(a["transient_mib"], "na")  # the first window has no start
        self.assertEqual(b["peak_allocated_mib"], str(20480 + 1200))  # NOT the 3700 of chunk 0
        self.assertEqual(b["transient_mib"], "1200")
        self.assertEqual(b["start_allocated_mib"], "20480")
        self.assertRegex(b["t_unix_ms"], r"^\d{13}$")
        self.assertNotEqual(b["window_ms"], "na")

    def test_since_pools_peak_survives_the_rebase(self):
        """[vram-peak]/WEG2-GRAPH-POOL read peak_mib 'since the pools'; the
        per-window re-base must not shrink it."""
        self._chunk(3700)
        self._chunk(1200)
        self.assertEqual(self.cuda.max_memory_allocated(), 20480 << 20)  # counter re-based
        self.assertEqual(vpw.cum_peak_allocated(self.cuda), (20480 + 3700) << 20)
        self.assertEqual(vfc._cum_peak_allocated(self.cuda), (20480 + 3700) << 20)
        # the planner's input line: WEG2-GRAPH-POOL peak_mib stays since-pools
        with self.assertLogs(vfc.logger, "INFO") as cm:
            vfc.log_graph_pool(_Runner(), "extend", cuda=self.cuda)
        self.assertIn(f"peak_mib={20480 + 3700} ", cm.output[-1])
        vpw.reset_since_pools()
        self.assertEqual(vpw.cum_peak_allocated(self.cuda), 20480 << 20)

    def test_rounds_close_every_n_and_draft_never_closes(self):
        with envs.SGLANG_WEG2_VRAM_PEAK_ROUNDS.override(4):
            out = []
            for i in range(9):
                self.cuda.alloc(10 << 20)
                self.cuda.free(10 << 20)
                draft = _Runner()
                draft.is_draft_model_runner = True
                self.assertIsNone(vpw.on_forward_end(draft, _FB("decode"), self.cuda))
                out.append(vpw.on_forward_end(_Runner(), _FB("verify" if i % 2 else "decode"), self.cuda))
            closed = [x for x in out if x]
            self.assertEqual(len(closed), 2)
            self.assertIn("phase=round n=4 ", closed[0])
            self.assertEqual(self.cuda.resets, 1 + 2)

    def test_open_rounds_are_closed_before_a_chunk(self):
        vpw.on_forward_end(_Runner(), _FB("decode"), self.cuda)
        with self.assertLogs(vpw.logger, "INFO") as cm:
            vpw.on_forward_end(_Runner(), _FB("extend", 3), self.cuda)
        self.assertIn("phase=round n=1 ", cm.output[0])
        self.assertIn("phase=chunk rows=3 n=1 ", cm.output[1])

    def test_flip_leg_lines_and_reraise(self):
        vpw._STATE["cuda_override"] = self.cuda
        try:
            class Mgr:
                scheduler = None

                @vpw.flip_leg("resume")
                def resume(self, x):
                    self_cuda.alloc(900 << 20)
                    self_cuda.free(900 << 20)
                    if x == "boom":
                        raise ValueError("leg died")
                    return "ok"

            self_cuda = self.cuda
            with self.assertLogs(vpw.logger, "INFO") as cm:
                self.assertEqual(Mgr().resume(1), "ok")
            self.assertIn("phase=idle", cm.output[0])
            self.assertRegex(cm.output[1], r"phase=flip leg=resume rpc_ms=\d+ rpc=ok n=0 ")
            self.assertIn(f"peak_allocated_mib={20480 + 900}", cm.output[1])
            with self.assertLogs(vpw.logger, "INFO") as cm:
                with self.assertRaises(ValueError):
                    Mgr().resume("boom")
            self.assertIn("rpc=raised", cm.output[-1])
        finally:
            vpw._STATE.pop("cuda_override", None)

    def test_off_means_no_line_and_no_rebase(self):
        with envs.SGLANG_WEG2_VRAM_PEAK.override(False):
            self.cuda.alloc(1 << 30)
            self.assertIsNone(vpw.on_forward_end(_Runner(), _FB("extend", 16384), self.cuda))
            self.assertEqual(self.cuda.resets, 1)

    def test_leg_without_cuda_is_a_plain_call(self):
        class Mgr:
            @vpw.flip_leg("release")
            def release(self):
                return 7

        self.assertEqual(Mgr().release(), 7)  # CPU-only test process: torch.cuda not initialized

    def test_scheduler_legs_carry_the_decorator(self):
        import ast
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        tree = ast.parse(inspect.getsource(wu))
        found = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ("release_memory_occupation",
                                                                   "resume_memory_occupation"):
                found[node.name] = [ast.unparse(d) for d in node.decorator_list]
        self.assertEqual(found["release_memory_occupation"][-1], "_vram_peak_leg('release')")
        self.assertEqual(found["resume_memory_occupation"][-1], "_vram_peak_leg('resume')")
        # the group-stop vote stays OUTSIDE: it sees the leg's exception after our line
        self.assertEqual(found["release_memory_occupation"][0], "_weg2_group_stop_on_leg_failure")


if __name__ == "__main__":
    unittest.main()
