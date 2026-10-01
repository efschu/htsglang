"""userdash without a GPU and without a server: fake expositions, fake clock, fake cards.

Covers the 27B builder's three asks (01.10.): default port + port env without collisions, /healthz,
the switch-off env -- plus parser, ring buffer, rendering and the absence of development fields.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG_ROOT)

from userdash import __main__ as cli  # noqa: E402
from userdash import prom  # noqa: E402
from userdash.collect import Collector, Config, parse_nvsmi  # noqa: E402
from userdash.server import STATIC, serve  # noqa: E402

#: words of the developer view that must never reach the user's page or JSON
DEV_WORDS = ("weg2", "pdflip", "flip", "awake", "epoch", "corridor", "x_tokens", "gpuq", "ipc", "boot_tag",
             "profile", "launcher", "served", "outstanding", "d_parked", "park", "seat", "victoria", "8428",
             "weg2_group", "P/D", "Expert", "Wächter", "deadman", "rank")


def front_metrics(gen_d=0.0, pre_p=0.0, cache_p=0.0, ttft=(), running=2, queue=1, kv_d=0.4, kv_p=0.9,
                  restart=False):
    """A front /metrics in the TSDB-1001 shape: own weg2_* + P and D relabelled weg2_group."""
    buckets = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    lines = ["# HELP weg2_outstanding Open requests of the front", "# TYPE weg2_outstanding gauge",
             "weg2_outstanding %d" % running, "weg2_queue_len %d" % queue,
             'weg2_awake{group="P"} 0', 'weg2_awake{group="D"} 1']
    for via in ("d_direct", "after_p"):
        obs = [v for (w, v) in ttft if w == via]
        for le in buckets:
            lines.append('weg2_ttft_seconds_bucket{le="%s",via="%s"} %d' % (le, via, sum(1 for v in obs if v <= le)))
        lines.append('weg2_ttft_seconds_bucket{le="+Inf",via="%s"} %d' % (via, len(obs)))
        lines.append('weg2_ttft_seconds_count{via="%s"} %d' % (via, len(obs)))
        lines.append('weg2_ttft_seconds_sum{via="%s"} %f' % (via, sum(obs)))
    lines.append('weg2_flip_seconds_count{dir="P2D",kind="layer"} 7')
    for g, kv in (("P", kv_p), ("D", kv_d)):
        for pp in ("0", "1"):   # PP=2: both stages export the same figures
            lab = 'engine_type="unified",model_name="Qwen3.8-27B",moe_ep_rank="0",pp_rank="%s",tp_rank="0",weg2_group="%s"' % (pp, g)
            lines.append("sglang:num_running_reqs{%s} %d" % (lab, 1))
            lines.append("sglang:token_usage{%s} %s" % (lab, kv))
            lines.append("sglang:max_total_num_tokens{%s} 262144" % lab)
            lines.append("sglang:kv_used_tokens{%s} %d" % (lab, int(kv * 262144)))
            dec = gen_d if g == "D" else 1.0
            pre = pre_p if g == "P" else 0.0
            cac = cache_p if g == "P" else 0.0
            lines.append('sglang:realtime_tokens_total{%s,mode="decode"} %s' % (lab, dec))
            lines.append('sglang:realtime_tokens_total{%s,mode="prefill_compute"} %s' % (lab, pre))
            lines.append('sglang:realtime_tokens_total{%s,mode="prefill_cache"} %s' % (lab, cac))
    return "\n".join(lines) + "\n"


MODELS = json.dumps({"object": "list", "data": [{"id": "Qwen3.8-27B", "object": "model",
                                                  "max_model_len": 262144}]}).encode()


class FakeServer:
    """fetch() stand-in: path -> (status, body) or an exception; records every URL asked."""

    def __init__(self):
        self.routes = {}
        self.asked = []
        self.down = False

    def __call__(self, url, timeout):
        self.asked.append(url)
        if self.down:
            raise ConnectionRefusedError("down")
        path = "/" + url.split("/", 3)[3] if url.count("/") >= 3 else "/"
        base = url[: len(url) - len(path)]
        r = self.routes.get((base, path), self.routes.get(path))
        if r is None:
            return 404, b"not found"
        if isinstance(r, Exception):
            raise r
        return r


class FakeGpus:
    error = None

    def read(self):
        return [{"index": 0, "name": "NVIDIA GeForce RTX 3080", "mem_used_mib": 18000.0, "mem_total_mib": 20480.0,
                 "util_pct": 97.0, "power_w": 220.5, "power_limit_w": 230.0, "temp_c": 61.0},
                {"index": 1, "name": "NVIDIA GeForce RTX 5090", "mem_used_mib": 30000.0, "mem_total_mib": 32607.0,
                 "util_pct": 88.0, "power_w": 380.0, "power_limit_w": 400.0, "temp_c": 58.0}]


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(clock=None, cfg=None, srv=None):
    clock = clock or Clock()
    srv = srv or FakeServer()
    col = Collector(cfg or Config(front="http://127.0.0.1:30030"), fetch=srv, gpu_reader=FakeGpus(), clock=clock)
    return col, srv, clock


class TestParser(unittest.TestCase):
    def test_parse_labels_escapes_and_bad_lines(self):
        s = prom.parse('a{x="1",y="q\\"uote"} 2\n# c\nbroken line here\nb 3 1700000000000\nc{le="+Inf"} +Inf\nd NaN\n')
        self.assertEqual(s[0], ("a", (("x", "1"), ("y", 'q"uote')), 2.0))
        self.assertEqual(s[1], ("b", (), 3.0))
        self.assertEqual(s[2][2], float("inf"))
        self.assertEqual(len(s), 3)  # NaN and the broken line are dropped

    def test_collapse_pp_stages_not_double_counted(self):
        s = prom.parse(front_metrics())
        by = prom.collapse(prom.select(s, "sglang:num_running_reqs"))
        self.assertEqual(by, {"P": 1.0, "D": 1.0})

    def test_collapse_tp_duplicates_max_dp_sum(self):
        rows = prom.select(prom.parse(
            'm{weg2_group="D",pp_rank="0",tp_rank="0",dp_rank="0"} 5\n'
            'm{weg2_group="D",pp_rank="0",tp_rank="1",dp_rank="0"} 5\n'
            'm{weg2_group="D",pp_rank="0",tp_rank="0",dp_rank="1"} 7\n'), "m")
        self.assertEqual(prom.collapse(rows), {"D": 12.0})

    def test_live_front_fixture(self):
        """The real exposition of the 27B front (01.10., trimmed): P on 3 PP stages, D, weg2_* own families."""
        with open(os.path.join(HERE, "fixtures", "front_metrics_live_27b_1001.txt"), encoding="utf-8") as fh:
            text = fh.read()
        s = prom.parse(text)
        rt = prom.select(s, "sglang:realtime_tokens_total", mode="prefill_compute")
        self.assertEqual(prom.collapse(rt), {"P": 38128.0, "D": 40335.0})   # P not tripled by its 3 stages
        col, srv, clk = make()
        srv.routes["/health"] = (200, b"")
        srv.routes["/metrics"] = (200, text.encode())
        col.sample_once()
        v = col.now_view()
        self.assertEqual(v["requests"], {"running": 2.0, "waiting": 1.0})
        self.assertAlmostEqual(v["kv"]["used_pct"], 21.0)                     # D is awake
        self.assertEqual(col._ttft_family, "weg2_ttft_seconds")
        cum = prom.histogram(s, "weg2_ttft_seconds")
        self.assertEqual(prom.count_of(cum), 26.0)                            # 5 after_p + 2 d_single + 19 d_direct
        self.assertIsNotNone(prom.quantile(0.9, cum))

    def test_quantile_matches_prometheus(self):
        b = {0.1: 10.0, 0.5: 60.0, 1.0: 90.0, float("inf"): 100.0}
        self.assertAlmostEqual(prom.quantile(0.5, b), 0.1 + 0.4 * (50 - 10) / 50)
        self.assertAlmostEqual(prom.quantile(0.9, b), 1.0)
        self.assertEqual(prom.quantile(0.99, b), 1.0)   # +Inf bucket -> largest finite bound
        self.assertIsNone(prom.quantile(0.5, {0.1: 0.0, float("inf"): 0.0}))

    def test_bucket_delta_restart(self):
        self.assertEqual(prom.bucket_delta({1.0: 2.0}, {1.0: 5.0}), {1.0: 2.0})
        self.assertEqual(prom.bucket_delta({1.0: 7.0}, {1.0: 5.0}), {1.0: 2.0})

    def test_nvsmi_csv(self):
        g = parse_nvsmi("0, NVIDIA GeForce RTX 3080, 1000, 20480, 5, 30.12, 230.00, 40\n1, X, [N/A], 1, 1, 1, 1, 1\n")
        self.assertEqual(g[0]["mem_total_mib"], 20480.0)
        self.assertIsNone(g[1]["mem_used_mib"])


class TestCollector(unittest.TestCase):
    def ready(self, srv, **kw):
        srv.routes["/health"] = (200, b"")
        srv.routes["/v1/models"] = (200, MODELS)
        srv.routes["/metrics"] = (200, front_metrics(**kw).encode())

    def test_status_boot_ready_dead(self):
        col, srv, clk = make()
        srv.down = True
        col.sample_once()
        self.assertEqual(col.now_view()["server"]["status"], "bootet")
        srv.down = False
        srv.routes["/health"] = (503, b"")
        clk.t += 5
        col.sample_once()
        self.assertEqual(col.status, "bootet")          # HTTP up, model still loading
        self.ready(srv)
        clk.t += 5
        col.sample_once()
        v = col.now_view()["server"]
        self.assertEqual(v["status"], "bereit")
        self.assertEqual(v["model"], "Qwen3.8-27B")
        self.assertEqual(v["context_len"], 262144)
        srv.routes["/health"] = (503, b"")
        clk.t += 5
        col.sample_once()
        self.assertEqual(col.status, "gestoert")
        srv.down = True
        clk.t += 5
        col.sample_once()
        self.assertEqual(col.status, "gestoert")        # one miss is not death
        clk.t += 5
        col.sample_once()
        self.assertEqual(col.status, "tot")

    def test_rates_kv_requests_ttft(self):
        col, srv, clk = make()
        self.ready(srv, gen_d=1000, pre_p=5000, cache_p=5000)
        col.sample_once()
        clk.t += 10
        ttft = [("d_direct", 0.08)] * 6 + [("after_p", 3.0)] * 4
        self.ready(srv, gen_d=1500, pre_p=25000, cache_p=10000, ttft=ttft)
        col.sample_once()
        v = col.now_view()
        self.assertAlmostEqual(v["throughput"]["decode_tps"], 50.0)        # (1500-1000)/10, P's 1 token flat
        self.assertAlmostEqual(v["throughput"]["prefill_tps"], 2000.0)     # PP=2 not double counted
        self.assertAlmostEqual(v["throughput"]["cache_hit_pct"], 20.0)
        self.assertEqual(v["requests"], {"running": 2.0, "waiting": 1.0})
        self.assertAlmostEqual(v["kv"]["used_pct"], 40.0)                  # the awake server, not the fuller P
        self.assertEqual(v["kv"]["total_tokens"], 262144)
        t = v["ttft"]
        self.assertEqual(t["n_5m"], 10)
        self.assertTrue(0.05 < t["p50_5m_s"] <= 0.1, t)
        self.assertTrue(2.0 < t["p90_5m_s"] <= 5.0, t)
        self.assertIn("ersten Token", t["basis"])
        self.assertEqual(len(v["gpus"]), 2)

    def test_counter_restart_never_negative(self):
        col, srv, clk = make()
        self.ready(srv, gen_d=100000)
        col.sample_once()
        clk.t += 5
        self.ready(srv, gen_d=250)              # server restarted: counter back near zero
        col.sample_once()
        self.assertAlmostEqual(col.now_view()["throughput"]["decode_tps"], 250 / 5.0)

    def test_rate_over_15s_window(self):
        col, srv, clk = make()
        for dec in (0, 0, 1000, 1000, 3000):
            self.ready(srv, gen_d=dec)
            col.sample_once()
            clk.t += 5
        # last sample at +20 s; window start = the oldest sample not older than 15 s (+5 s, 0 tokens)
        self.assertAlmostEqual(col.now_view()["throughput"]["decode_tps"], 3000 / 15.0)

    def test_ring_buffer_keeps_one_hour(self):
        col, srv, clk = make(cfg=Config(history_s=3600))
        self.ready(srv)
        for _ in range(800):
            col.sample_once()
            clk.t += 5
        h = col.history(3600)
        self.assertLessEqual(clk.t - h["t"][0], 3600 + 5)
        self.assertGreaterEqual(len(h["t"]), 700)
        self.assertEqual(len(h["gpus"]["util_pct"]), 2)
        self.assertEqual(len(h["gpus"]["util_pct"][0]), len(h["t"]))
        self.assertEqual(len(col.history(900)["t"]), 180)

    def test_cadence_browser_vs_idle(self):
        col, _, clk = make(cfg=Config(active_s=5, idle_s=30, hold_s=90))
        self.assertEqual(col.period(), 30)
        col.touch()
        self.assertEqual(col.period(), 5)
        clk.t += 91
        self.assertEqual(col.period(), 30)
        col2, _, _ = make(cfg=Config(idle_s=0))
        self.assertIsNone(col2.period())                 # no browser, no sampling

    def test_only_get_of_allowed_paths(self):
        col, srv, clk = make()
        self.ready(srv)
        for _ in range(3):
            col.sample_once()
            clk.t += 61
        paths = {u.split("30030", 1)[1] for u in srv.asked}
        self.assertTrue(paths <= {"/health", "/metrics", "/v1/models"}, paths)

    def test_legacy_passthrough_front_reads_groups(self):
        cfg = Config(front="http://127.0.0.1:30030", auto_group_urls=("http://127.0.0.1:30031", "http://127.0.0.1:30032"))
        col, srv, clk = make(cfg=cfg)
        plain = front_metrics(gen_d=10).replace(',weg2_group="P"', "").replace(',weg2_group="D"', "")
        srv.routes["/health"] = (200, b"")
        srv.routes["/metrics"] = (200, plain.encode())
        srv.routes["/weg2/state"] = (200, json.dumps({"state": "serving", "awake": "D"}).encode())
        srv.routes[("http://127.0.0.1:30031", "/metrics")] = (200, b'sglang:realtime_tokens_total{mode="decode"} 5\n')
        srv.routes[("http://127.0.0.1:30032", "/metrics")] = (200, b'sglang:realtime_tokens_total{mode="decode"} 100\n')
        col.sample_once()
        self.assertEqual(col.legacy_groups, list(cfg.auto_group_urls))
        clk.t += 10
        srv.routes[("http://127.0.0.1:30032", "/metrics")] = (200, b'sglang:realtime_tokens_total{mode="decode"} 400\n')
        col.sample_once()
        self.assertAlmostEqual(col.now_view()["throughput"]["decode_tps"], 30.0)

    def test_plain_server_without_front(self):
        col, srv, clk = make()
        srv.routes["/health"] = (200, b"")
        srv.routes["/metrics"] = (200, b"sglang:num_running_reqs{pp_rank=\"0\"} 3\nsglang:num_queue_reqs{pp_rank=\"0\"} 0\n"
                                         b"sglang:generation_tokens_total 10\nsglang:prompt_tokens_total 100\n")
        col.sample_once()
        self.assertEqual(col.now_view()["requests"], {"running": 3.0, "waiting": 0.0})
        self.assertIsNone(col.legacy_groups)              # 404 on the state routes -> plain server


class TestNoDevelopmentFields(unittest.TestCase):
    def test_json_and_page_carry_no_dev_words(self):
        col, srv, clk = make()
        srv.routes["/health"] = (200, b"")
        srv.routes["/v1/models"] = (200, MODELS)
        srv.routes["/metrics"] = (200, front_metrics(ttft=[("after_p", 1.0)]).encode())
        col.sample_once()
        clk.t += 5
        col.sample_once()
        blob = json.dumps(col.now_view()) + json.dumps(col.history(3600)) + json.dumps(col.health()[1])
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
            page = fh.read()
        for w in DEV_WORDS:
            self.assertNotIn(w.lower(), blob.lower(), "JSON: " + w)
            self.assertNotIn(w.lower(), page.lower(), "Seite: " + w)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class TestHttp(unittest.TestCase):
    def setUp(self):
        self.col, self.srv, self.clk = make()
        self.srv.routes["/health"] = (200, b"")
        self.srv.routes["/v1/models"] = (200, MODELS)
        self.srv.routes["/metrics"] = (200, front_metrics().encode())
        self.col.sample_once()
        self.port = free_port()
        self.httpd = serve(self.col, "127.0.0.1", self.port)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def get(self, path):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d%s" % (self.port, path), timeout=5) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), e.read()

    def test_page_and_api(self):
        code, ctype, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", ctype)
        html = body.decode()
        for anchor in ('id="ttft-p50"', 'id="ttft-p90"', 'id="dec"', 'id="pre"', 'id="req"', 'id="kv"',
                       'id="gpus"', 'id="charts"', 'id="pill"', "Grafikkarten", "Verlauf"):
            self.assertIn(anchor, html)
        code, _, body = self.get("/api/now")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["server"]["status"], "bereit")
        code, _, body = self.get("/api/history?s=900")
        self.assertEqual(code, 200)
        self.assertIn("decode_tps", json.loads(body))
        self.assertEqual(self.get("/mark.svg")[0], 200)
        self.assertEqual(self.get("/weg2/state")[0], 404)

    def test_healthz_does_not_raise_cadence(self):
        before = self.col.active()
        code, ctype, body = self.get("/healthz")
        self.assertEqual(code, 200)
        self.assertIn("application/json", ctype)
        d = json.loads(body)
        self.assertTrue(d["ok"])
        self.assertEqual(d["dienst"], "userdash")
        self.assertEqual(d["server_status"], "bereit")
        self.assertEqual(self.col.active(), before)
        self.get("/api/now")
        self.assertTrue(self.col.active())

    def test_healthz_503_when_sampler_died(self):
        self.col.thread = threading.Thread(target=lambda: None)
        self.col.thread.start()
        self.col.thread.join()
        self.assertEqual(self.get("/healthz")[0], 503)


class TestCli(unittest.TestCase):
    def test_default_port_is_free_of_rig_ports(self):
        self.assertEqual(cli.DEFAULT_PORT, 30080)
        for p in (30030, 30031, 30032, 30097, 30099, 8890, 8428):
            self.assertIn(p, cli.RESERVED_PORTS)
            self.assertTrue(cli.port_problem(p, "http://127.0.0.1:30030"))
        self.assertEqual(cli.port_problem(cli.DEFAULT_PORT, "http://127.0.0.1:30030"), "")
        self.assertTrue(cli.port_problem(30090, "http://127.0.0.1:30090"))   # never the server's own port

    def test_port_env(self):
        env = dict(os.environ, USERDASH_PORT="30085")
        out = subprocess.run([sys.executable, "-c", "from userdash import __main__ as m; print(m.build_parser().parse_args([]).port)"],
                             cwd=PKG_ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(out.stdout.strip(), "30085")

    def test_disable_env_exits_zero_without_listening(self):
        env = dict(os.environ, USERDASH_ENABLE="0", USERDASH_PORT=str(free_port()))
        out = subprocess.run([sys.executable, "-m", "userdash"], cwd=PKG_ROOT, env=env, capture_output=True,
                             text=True, timeout=30)
        self.assertEqual(out.returncode, 0)
        self.assertIn("USERDASH_ENABLE=0", out.stderr)

    def test_reserved_port_refused(self):
        env = dict(os.environ, USERDASH_PORT="30030")
        out = subprocess.run([sys.executable, "-m", "userdash"], cwd=PKG_ROOT, env=env, capture_output=True,
                             text=True, timeout=30)
        self.assertEqual(out.returncode, 2)
        self.assertIn("VERWEIGERT", out.stderr)

    def test_process_serves_healthz(self):
        port = free_port()
        env = dict(os.environ, USERDASH_PORT=str(port), USERDASH_BIND="127.0.0.1",
                   USERDASH_FRONT="http://127.0.0.1:%d" % free_port(), USERDASH_GPU="0")
        p = subprocess.Popen([sys.executable, "-m", "userdash"], cwd=PKG_ROOT, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            d = None
            for _ in range(50):
                try:
                    with urllib.request.urlopen("http://127.0.0.1:%d/healthz" % port, timeout=2) as r:
                        d = json.loads(r.read())
                        break
                except OSError:
                    time.sleep(0.1)
            self.assertIsNotNone(d)
            self.assertTrue(d["ok"])
            self.assertTrue(d["sampler_alive"])
            self.assertEqual(d["server_status"], "bootet")   # nothing listens on the front port
        finally:
            p.terminate()
            p.wait(timeout=10)


EP_SNIPPET = os.path.join(os.path.dirname(PKG_ROOT), "user_dashboard", "entrypoint_userdash.sh")


def run_entry(env_extra, front_port="30030", wait_health_port=None):
    """Source the entrypoint snippet as the image's entrypoint does (say/refuse stubs like entrypoint.sh)."""
    import tempfile
    logdir = tempfile.mkdtemp(prefix="ud_ep_")
    script = (
        'say() { printf "[ep] %%s\\n" "$*" >&2; }\n'
        'refuse() { say "REFUSED $1: $2"; exit 3; }\n'
        'FRONT_PORT=%s; PY=%s; LOGCOPY=%s; USERDASH_DIR=%s\n'
        '. %s && userdash_start\n'
        'echo "PID=$USERDASH_PID"\n' % (front_port, sys.executable, logdir, PKG_ROOT, EP_SNIPPET))
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(("HTSGLANG_USERDASH", "USERDASH_")):
            del env[k]
    env.update(env_extra)
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)
    pid = None
    for line in out.stdout.splitlines():
        if line.startswith("PID=") and line[4:].strip():
            pid = int(line[4:])
    return out, pid, logdir


class TestEntrypointSnippet(unittest.TestCase):
    def test_off_starts_nothing(self):
        out, pid, _ = run_entry({"HTSGLANG_USERDASH": "0"})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIsNone(pid)
        self.assertIn("USERDASH aus", out.stderr)

    def test_bad_values_refused(self):
        for env, front in (({"HTSGLANG_USERDASH": "ja"}, "30030"), ({"HTSGLANG_USERDASH_PORT": "30030"}, "30030"),
                           ({"HTSGLANG_USERDASH_PORT": "30099"}, "30030"), ({"HTSGLANG_USERDASH_PORT": "30090"}, "30090"),
                           ({"HTSGLANG_USERDASH_PORT": "x1"}, "30030")):
            out, pid, _ = run_entry(env, front)
            self.assertEqual(out.returncode, 3, (env, out.stderr))
            self.assertIn("REFUSED USERDASH", out.stderr)
            self.assertIsNone(pid)

    def test_on_by_default_serves_healthz_on_port_env(self):
        port = free_port()
        out, pid, logdir = run_entry({"HTSGLANG_USERDASH_PORT": str(port), "HTSGLANG_USERDASH_BIND": "127.0.0.1"},
                                     str(free_port()))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIsNotNone(pid)
        try:
            d = None
            for _ in range(80):
                try:
                    with urllib.request.urlopen("http://127.0.0.1:%d/healthz" % port, timeout=2) as r:
                        self.assertEqual(r.status, 200)
                        d = json.loads(r.read())
                        break
                except OSError:
                    time.sleep(0.1)
            self.assertIsNotNone(d, open(os.path.join(logdir, "userdash.log")).read())
            self.assertTrue(d["ok"])
            self.assertIn("USERDASH :%d" % port, out.stderr)
        finally:
            os.kill(pid, 15)

    def test_default_port_30080(self):
        with open(EP_SNIPPET, encoding="utf-8") as fh:
            self.assertIn(': "${HTSGLANG_USERDASH_PORT:=30080}"', fh.read())


if __name__ == "__main__":
    unittest.main()
