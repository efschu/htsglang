#!/usr/bin/env python3
"""SET-X-Wache NF (SCHEDULER-SOLL v2, h16u 1017): ersetzt die pro-Request-Wache prefill_x_wache.py. Nur Lesen.

Misst JE SET (= je Entscheid) aus den Front-Logzeilen `WEG2 SET-VERDICT ...` (SOLL §4) die Verstoesse:
  R7 : verdict=FLIP_P (oder REROUTE_P) mit S < X, ausser ein Mitglied ist p_only (R9: p_only = gross)
  R3 : verdict=D_AFTER_TIMER mit S >= X (das Set haette nach P gemusst)
  R6 : verdict=D_NOW (D-Prefill) mit S > X
  R2 : ein kurz-Request wartet (classes nur kurz und waited_s > 0.5) oder bekommt statt D_NOW einen anderen Entscheid
  FORMAT : eine Zeile mit `WEG2 SET-VERDICT`, die nicht dem Format §4 entspricht (S1 hat vom Format abweichend geloggt)
COLLECT-Zeilen sind kein Entscheid und zaehlen nur in der Stundenbilanz.

Ausgabe: stdout nur Verstoesse (gedrosselt: 1 Zeile je Art je THROTTLE s, <=200 Zeichen); dieselben Zeilen nach set_x.log;
Stundenbilanz nach set_x_stunde.txt. Solange noch KEIN SET-VERDICT im Log steht (alter Boot): je Stunde die Zeile
"SET-VERDICT fehlt (alter Stand)" in die Stundendatei. Bei Boot-Wechsel (neuestes *.front.log) haengt sie automatisch um.
Aufruf live:  set_x_wache.py      Replay:  set_x_wache.py --replay [HH:MM [HH:MM]] [--file <front.log>]
"""
import glob, os, re, sys, time, datetime

EV = os.environ.get("SX_EVIDENCE", "/spinning/docker-acceptance/nf/evidence")
TMP = os.environ.get("SX_OUT", "/spinning/gpu-arb/deskq/wacht")
LOG = TMP + "/set_x.log"
STUNDE = TMP + "/set_x_stunde.txt"
THROTTLE = float(os.environ.get("SX_THROTTLE", "120"))
KURZ_WAIT_S = float(os.environ.get("SX_KURZ_WAIT_S", "0.5"))

RE_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d) (\d\d):(\d\d):(\d\d),\d+\]")
# Format SOLL §4 (exakt; why = Rest der Zeile)
RE_VERDICT = re.compile(
    r"WEG2 SET-VERDICT epoch=(?P<epoch>\d+) set=(?P<set>\S+) n=(?P<n>\d+) members=(?P<members>\S*) "
    r"S=(?P<S>\d+) X=(?P<X>\d+) halfX=(?P<halfX>\d+) classes=(?P<classes>\S*) "
    r"d_decoding=(?P<dd>[01]) seats_free=(?P<seats>-?\d+) t_half_s=(?P<th>[\d.]+) waited_s=(?P<w>[\d.]+) "
    r"verdict=(?P<verdict>[A-Z_]+) why=(?P<why>.*)$")
CLASSES = {"kurz", "halb", "gross", "p_only"}
VERDICTS = {"COLLECT", "FLIP_P", "D_AFTER_TIMER", "PARK_NO_SEAT", "REROUTE_P", "D_NOW", "NOT_D"}
# NOT_D (A6, 17.10.): a kurz request the front did NOT admit on D (why=yield_or_gate|d_budget) -- always an R2 finding
ARTS = ("R2", "R3", "R6", "R7", "FORMAT")


def parse_verdict(line):
    """-> dict | None (keine SET-VERDICT-Zeile) | 'FORMAT' (SET-VERDICT, aber nicht §4)."""
    if "WEG2 SET-VERDICT" not in line:
        return None
    m = RE_VERDICT.search(line)
    if not m:
        return "FORMAT"
    d = m.groupdict()
    cls = d["classes"].split(",") if d["classes"] else []
    if d["verdict"] not in VERDICTS or not cls or any(c not in CLASSES for c in cls):
        return "FORMAT"
    return {"epoch": int(d["epoch"]), "set": d["set"], "n": int(d["n"]), "S": int(d["S"]), "X": int(d["X"]),
            "halfX": int(d["halfX"]), "classes": cls, "d_decoding": d["dd"] == "1", "seats_free": int(d["seats"]),
            "t_half_s": float(d["th"]), "waited_s": float(d["w"]), "verdict": d["verdict"], "why": d["why"]}


def check(v):
    """Verstoss-Art eines geparsten Entscheids oder None. Reine Funktion."""
    verdict, S, X, cls = v["verdict"], v["S"], v["X"], v["classes"]
    if verdict == "COLLECT":
        return "R2" if cls == ["kurz"] else None
    if cls == ["kurz"] or all(c == "kurz" for c in cls):
        # R5 (corrected 17.10.): waiting for a D seat (why=seat_wait) or a refusal for budget / a decided flip
        # (NOT_D why=d_budget|yield_or_gate) is allowed; a kurz request that waits WITHOUT such a reason is R2
        if verdict == "NOT_D":
            if v["why"].split()[0] not in ("d_budget", "yield_or_gate"):
                return "R2"
        elif verdict != "D_NOW" or (v["waited_s"] > KURZ_WAIT_S and v["why"].split()[0] != "seat_wait"):
            return "R2"
        return "R6" if S > X else None      # D-Prefill ueber X (S schliesst einen laufenden D-Prefill ein)
    if verdict in ("FLIP_P", "REROUTE_P"):
        if S < X and "p_only" not in cls:
            return "R7"
        return None
    if verdict == "D_AFTER_TIMER":
        return "R3" if S >= X else None
    if verdict == "D_NOW":
        return "R6" if S > X else None
    return None


class State:
    def __init__(self, clock=time.time, log=LOG, stunde=STUNDE):
        self.clock = clock
        self.log = log
        self.stunde_path = stunde
        self.last_emit = {}
        self.pending = {}        # art -> (n seit letzter Meldung, schlimmster Fall)
        self.seen_ever = False
        self.hour = None
        self.stunde_lines = []
        self.reset_stat()

    def reset_stat(self):
        self.stat = {"sets": 0, "COLLECT": 0, "FLIP_P": 0, "D_AFTER_TIMER": 0, "PARK_NO_SEAT": 0, "REROUTE_P": 0, "D_NOW": 0, "NOT_D": 0,
                     "viol": {a: 0 for a in ARTS}}

    # --- Eingabe -------------------------------------------------------------------------------------------
    def feed(self, line, emit):
        m = RE_TS.match(line)
        if not m:
            return
        self.roll("%s %s" % (m.group(1), m.group(2)))
        hhmmss = "%s:%s:%s" % (m.group(2), m.group(3), m.group(4))
        v = parse_verdict(line)
        if v is None:
            return
        self.seen_ever = True
        if v == "FORMAT":
            self.stat["viol"]["FORMAT"] += 1
            self.fall("FORMAT", hhmmss, "-", 0, 0, line.strip()[-120:], emit)
            return
        if v["verdict"] != "COLLECT":
            self.stat["sets"] += 1
        self.stat[v["verdict"]] += 1
        art = check(v)
        if art:
            self.stat["viol"][art] += 1
            self.fall(art, hhmmss, v["set"], v["S"], v["X"], "verdict=%s classes=%s waited_s=%.2f why=%s" % (
                v["verdict"], ",".join(v["classes"]), v["waited_s"], v["why"][:40]), emit)

    def roll(self, hour):
        if self.hour is None:
            self.hour = hour
        elif hour != self.hour:
            self.stunde_ausgeben(self.hour)
            self.hour = hour

    # --- Ausgabe -------------------------------------------------------------------------------------------
    def _append(self, path, zeile):
        try:
            with open(path, "a") as f:
                f.write(zeile + "\n")
        except OSError:
            pass

    def fall(self, art, hhmmss, set_id, S, X, detail, emit):
        n, schlimm = self.pending.get(art, (0, None))
        n += 1
        score = abs(S - X)
        if schlimm is None or score > schlimm[0]:
            schlimm = (score, hhmmss, set_id, S, X, detail)
        self.pending[art] = (n, schlimm)
        now = self.clock()
        if now - self.last_emit.get(art, -1e18) >= THROTTLE:
            self.last_emit[art] = now
            n, (_, t, sid, s, x, det) = self.pending.pop(art)
            zeile = "%s %s n=%d seit letzter Meldung; schlimmster set=%s S/X=%d/%d %s" % (t, art, n, sid, s, x, det)
            zeile = zeile[:200]
            self._append(self.log, zeile)
            emit(zeile)

    def stunde_ausgeben(self, h):
        s = self.stat
        if not self.seen_ever:
            zeile = "%sZ SET-VERDICT fehlt (alter Stand)" % h
        else:
            v = s["viol"]
            zeile = ("%sZ Sets %d (FLIP_P %d, D_AFTER_TIMER %d, PARK_NO_SEAT %d, REROUTE_P %d, D_NOW %d; COLLECT-Zeilen %d); "
                     "Verstoesse R7 %d, R3 %d, R6 %d, R2 %d, FORMAT %d") % (
                h, s["sets"], s["FLIP_P"], s["D_AFTER_TIMER"], s["PARK_NO_SEAT"], s["REROUTE_P"], s["D_NOW"], s["COLLECT"],
                v["R7"], v["R3"], v["R6"], v["R2"], v["FORMAT"])
        self._append(self.stunde_path, zeile)
        self.stunde_lines.append(zeile)
        self.reset_stat()
        return zeile


def newest_front():
    fs = sorted(glob.glob(EV + "/boot_weg2_*.front.log"), key=os.path.getmtime, reverse=True)
    return fs[0] if fs else None


def emit_live(msg):
    print(msg[:200], flush=True)


def replay(path, a=None, b=None):
    st = State(log=os.devnull, stunde=os.devnull)   # Replay schreibt NIE in die Live-Dateien
    t = [0.0]
    st.clock = lambda: t.__setitem__(0, t[0] + THROTTLE + 1) or t[0]
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_TS.match(line)
            if not m:
                continue
            hm = "%s:%s" % (m.group(2), m.group(3))
            if a and hm < a:
                continue
            if b and hm > b:
                break
            st.feed(line, out.append)
    st.stunde_ausgeben(st.hour or "??")
    print("\n".join(out + ["STUNDE " + z for z in st.stunde_lines]))


def live():
    st = State()
    cur = fh = None
    while True:
        nf = newest_front()
        if nf and nf != cur:
            if fh:
                fh.close()
            cur = nf
            fh = open(cur, errors="replace")
            fh.seek(0, 2)
            st = State()
            emit_live("SET-X-WACHE haengt um auf %s" % os.path.basename(cur)[-40:])
        got = False
        while fh:
            line = fh.readline()
            if not line:
                break
            got = True
            st.feed(line, emit_live)
        # Leerlauf: die Stunde laeuft auch ohne Logzeilen weiter (alter Stand schreibt trotzdem je Stunde)
        st.roll(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H"))
        time.sleep(0.5 if got else 1.0)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--replay":
        rest = args[1:]
        fpath = None
        if "--file" in rest:
            i = rest.index("--file")
            fpath = rest[i + 1]
            rest = rest[:i] + rest[i + 2:]
        replay(fpath or newest_front(), rest[0] if rest else None, rest[1] if len(rest) > 1 else None)
    else:
        live()
