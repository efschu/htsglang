"""Rule (DASHBOARD-AUS-IPC, user order 29.09. via 27B): no NEW log parser in rigdash.

"das dashboard soll auch aus der inter prozess kommunikation gespeist werden, nicht aus logs".
Every regex literal the package compiles or matches today is frozen below (file + sha1 of the
pattern).  A new one fails this test: a new display reads its IPC source (state.json,
events.jsonl, rankstate/, /weg2/state -- see /spinning/gpu-arb/docs/DASHBOARD-AUS-IPC-INVENTAR-0929.md);
if the source does not exist yet, the writer gets it first (IPC-STATE-PLAN §2.2: the source
writes, then the reader moves, then the parser falls).  Removing a frozen pattern is always
allowed -- that is the point.  A regex that parses no log at all (a profile, a git subject)
goes into NON_LOG_ALLOWED with the reason, never into FROZEN.
"""

import ast
import hashlib
import os
import unittest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNCS = {"compile", "search", "match", "fullmatch", "findall", "finditer", "sub", "subn", "split"}

FROZEN = {
    ("features.py", "2e323b1c186c"),  # '[0-9a-f]{7,40}'
    ("features.py", "2fd3e71ae83a"),  # '^PROFILE_FORMAT=([^\\s#]+)'
    ("features.py", "7777fee0e4cd"),  # '^\\s*(?:source|\\.)\\s+"\\$\\(dirname "\\$\\{BASH_SOURC'
    ("features.py", "a6a5d065eea5"),  # '--max-kv-per-request[ =](\\d+)'
    ("features.py", "c2d5149d091b"),  # '27b'
    ("features.py", "c3b19bd43910"),  # '-(\\d{8}T\\d{6})Z-'
    ("features.py", "d4c043e9a5a2"),  # '-(rc[0-9][0-9a-z.]*)-'
    ("features.py", "e1df3afac94c"),  # '(^|\\s)#.*$'
    ("features_update.py", "0cf0ac9f21dd"),  # '(\\w+) D-only \\(([^)]*)\\)'
    ("features_update.py", "2b99a51b5fd4"),  # '\\*\\*|`'
    ("features_update.py", "485bb1ddf6c4"),  # '(?<!\\\\)\\|'
    ("features_update.py", "524f95e33de6"),  # 'Gemessene Runde \\(gpu-ms\\):([^\\n]*)'
    ("features_update.py", "6d8db6b44e20"),  # 'Stand (\\d\\d)\\.(\\d\\d)\\.(\\d{4}) ~?(\\d\\d):(\\d\\d)Z'
    ("features_update.py", "967febb02680"),  # 'bs(\\d) ([0-9,]+) \\(n=(\\d+)\\)'
    ("live.py", "91cecd3b000b"),  # '^(?P<stem>.+?)\\.(?P<group>P|D|front)\\.log$'
    ("live.py", "c2d5149d091b"),  # '27b'
    ("parse.py", "009c7604b034"),  # 'FWD-TIMING-PREFILL forward=(\\d+) tokens=(\\d+) .*'
    ("parse.py", "054d4bf381c0"),  # 'WEG2-HEALTH group=(\\w+) http_ok=(\\w+) process_al'
    ("parse.py", "0e036f00de7e"),  # '\\bsleep=(\\d+(?:\\.\\d+)?) ms'
    ("parse.py", "1230919b6c23"),  # '#running-req: (\\d+)'
    ("parse.py", "1327917382dc"),  # 'accept len: ([\\d.]+)'
    ("parse.py", "16d78071c6ff"),  # 'WEG2-CORRIDOR phase=(\\w)\\((\\w+)\\)'
    ("parse.py", "2f20bd0593be"),  # 'WEG2-FLIP done epoch=(\\d+) slept=(\\w+) woke=(\\w+'
    ("parse.py", "3325f8ed0925"),  # '\\bprompt_tokens=(\\d+)'
    ("parse.py", "35d76907b67a"),  # 'WEG2 D-PHASE-SEATS \\(H95\\) epoch=\\S+ handoff_n=('
    ("parse.py", "37155ad88cff"),  # '(?:^|\\s)([A-Za-z_][A-Za-z0-9_.]*(?:Error|Excepti'
    ("parse.py", "39bbb9a7c7f6"),  # '#new-token: (\\d+)'
    ("parse.py", "3a476bb8c4f0"),  # 'drain\\+quiesce=(\\d+(?:\\.\\d+)?) ms'
    ("parse.py", "43cfde3a361a"),  # '\\b\\w+Refused: #\\d+|#791b|SPLIT refused|ADMISSION'
    ("parse.py", "43e55a00afa8"),  # '^MAMBA-HOST-RESUME n=\\d+: anchor accepted at dep'
    ("parse.py", "473cd316245f"),  # 'full token usage: ([\\d.]+)'
    ("parse.py", "4c680bef4cc1"),  # '\\bwake=(\\d+(?:\\.\\d+)?) ms'
    ("parse.py", "4d5178a6e62e"),  # 'gpu-ms: ([\\d.]+) \\(compute ([\\d.]+), wait ([\\d.]'
    ("parse.py", "4eba07a97419"),  # '^#988 LOADBACK rid=(\\S+)'
    ("parse.py", "5b27b48b9199"),  # '#queue-req: (\\d+)'
    ("parse.py", "5b3bb06e4856"),  # '\\banchor_depth=(\\d+)'
    ("parse.py", "5c339fc08969"),  # 'WEG2-POST-WAKE-PASS n=0 mode=(\\w+) .*?\\bschedule'
    ("parse.py", "5d4a8b62b5d7"),  # "served_model_name='([^']*)'"
    ("parse.py", "6c44e7aed221"),  # '#new-seq: (\\d+)'
    ("parse.py", "6f5519dd5fca"),  # 'flip_total=(\\d+(?:\\.\\d+)?) ms'
    ("parse.py", "6f7245404f4b"),  # "model_path='([^']*)'"
    ("parse.py", "71b94360877f"),  # '#pending-token: (\\d+)'
    ("parse.py", "794a8d875f7b"),  # 'WEG2-SERVED group=(\\w+) leg=(\\d+) rid=(\\S+)'
    ("parse.py", "7cc6ca438d70"),  # 'WEG2 BOOT tag=(\\S+) tree=(\\S+) @ (\\w+)'
    ("parse.py", "8281b2005948"),  # 'accept rate: ([\\d.]+)'
    ("parse.py", "857b4b2f30b0"),  # 'bubble_ms=([\\d.]+)'
    ("parse.py", "85b2958c0db7"),  # 'WEG2-ROUTE .*?\\(awake=(\\w+)\\b.*?queue=(\\d+)'
    ("parse.py", "8db93d227ba4"),  # '\\bcached_tokens=(\\d+)'
    ("parse.py", "95eda8eafa36"),  # 'WEG2-FORM (.*?) \\(sources:'
    ("parse.py", "9ff3008b9834"),  # '#cached-token: (\\d+)'
    ("parse.py", "a7daeb65d2c6"),  # 'input throughput \\(token/s\\): ([\\d.]+)'
    ("parse.py", "aaba130354a5"),  # '\\bcompletion_tokens=(\\d+)'
    ("parse.py", "b1a067a5ba8f"),  # 'TIMING-FLUSH-WAIT .*?\\bforward=(\\d+) .*?\\bt_unix'
    ("parse.py", "b1e35b780087"),  # '#round: (\\d+)'
    ("parse.py", "bc554e2b7781"),  # 'gen throughput \\(token/s\\): ([\\d.]+)'
    ("parse.py", "beb8ed2d9a8c"),  # '\\bkv_applied=(\\d+)'
    ("parse.py", "bf3fed22783b"),  # 'cuda graph: (\\w+)'
    ("parse.py", "c0d93e39a648"),  # ', t: (\\d+\\.\\d+),'
    ("parse.py", "c362bae3b899"),  # '\\bpp_size=(\\d+)'
    ("parse.py", "c5c364cd0b6a"),  # '#full token: (\\d+)'
    ("parse.py", "c97f11d7dbcb"),  # 'WEG2-FORM .*?\\bmodel=(\\S+)'
    ("parse.py", "c9db86cc0a57"),  # 'gpu-ms: ([\\d.]+)'
    ("parse.py", "ce879329bfd3"),  # '^\\s*(ERROR|CRITICAL)\\b'
    ("parse.py", "d2b5f73b70ff"),  # '^max_total_num_tokens=(\\d+),.*\\bmax_running_requ'
    ("parse.py", "d3dc7505fac3"),  # '^\\[(\\d{4})-(\\d{2})-(\\d{2})[ T](\\d{2}):(\\d{2}):(\\'
    ("parse.py", "dab7a8287157"),  # 'WEG2-FLIP begin epoch=(\\d+) sleep=(\\w+) wake=(\\w'
    ("parse.py", "daf24911d6db"),  # '^#\\d+ PREFETCH (LANDED|REFUSED|DEFERRED|TIMEOUT)'
    ("parse.py", "ed220b25da90"),  # '^#\\d+ STORE READ INCOMPLETE rid=(\\S+) delivered='
    ("parse.py", "f1b24eb4784e"),  # '\\bbs: (\\d+)'
    ("parse.py", "f6014db7c442"),  # '\\btp_size=(\\d+)'
    ("parse.py", "fc1645a8ccdc"),  # 'HOST-ANON-PASS pass=\\d+ phase=EXTEND .*?\\bwall_m'
    ("redact.py", "22ff14d522f4"),  # 'ADMIN-KEY|admin-api-key|\\.adminkey\\b|auth=bearer'
    ("redact.py", "5f54e3b3716a"),  # '(?i)[\\w./-]*(\\.adminkey|\\.pem|\\.key|_PAT)\\b'
    ("redact.py", "5fd4f3cf6518"),  # '(?i)\\b(api[_-]?key|admin[_-]?key|access[_-]?key|'
    ("redact.py", "a2a35f80eeb4"),  # 'ADMIN-KEY|admin-api-key|\\.adminkey|auth=bearer|G'
    ("server.py", "bfaff801a873"),  # '^htsglang-(acc-)?'
    ("stops.py", "b9493c0e6584"),  # 'DEADMAN-VERDICT deadman_(\\S+?)_(P|D|front): DEAD'
    ("stops.py", "c5f61547d2a6"),  # '^\\[host-acc (\\d\\d):(\\d\\d):(\\d\\d)Z\\] AGENT-HOLD \\'
    ("stops.py", "f171b2129f1c"),  # '^\\[nf-dauer (\\d{4})-(\\d\\d)-(\\d\\d)T(\\d\\d):(\\d\\d):'
    ("weg2line.py", "5268c821ce44"),  # '^(PROFILE_[A-Z_]+)=(\\"[^\\"]*\\"|\'[^\']*\'|[^\\s#]*)'
    ("weg2line.py", "93beb1e2ce98"),  # '^htsglang:(cu\\d+)-weg2-(.+?)-27b-nf(-flat)?$'
    ("weg2line.py", "e2973332316e"),  # '^\\s*(trap |mkdir |docker |rm |pct |systemctl )'
}

# (file, sha1[:12]) -> why this regex reads no log line
NON_LOG_ALLOWED = {}


def regex_literals():
    out = set()
    for fn in sorted(os.listdir(PKG)):
        if not fn.endswith(".py"):
            continue
        with open(os.path.join(PKG, fn)) as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            base = f.value.id if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else None
            if (name in FUNCS and base == "re") or name == "_f":
                p = node.args[0].value
                out.add((fn, hashlib.sha1(p.encode()).hexdigest()[:12], p))
    return out


class TestNoNewLogParsers(unittest.TestCase):
    def test_no_regex_beyond_the_frozen_set(self):
        new = sorted((fn, h, p) for fn, h, p in regex_literals()
                     if (fn, h) not in FROZEN and (fn, h) not in NON_LOG_ALLOWED)
        self.assertEqual(new, [], "neuer Log-Parser im rigdash verboten (DASHBOARD-AUS-IPC): %s -- "
                         "IPC-Quelle lesen (state.json/events.jsonl/rankstate//weg2/state), fehlt sie, "
                         "zuerst beim Schreiber anlegen; siehe DASHBOARD-AUS-IPC-INVENTAR-0929.md" % new)

    def test_ipc_reader_parses_no_log(self):
        # the IPC reader reads JSON records only
        self.assertEqual([x for x in regex_literals() if x[0] == "ipcstate.py"], [])
