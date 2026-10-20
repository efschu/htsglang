#!/usr/bin/env python3
# nf_collect_wait_monitor.py (14.10. NF-Sitz, Nutzerorder: "wenn kein Decode mehr ansteht, wird sofort geprefillt -- Monitor,
# der das ueberwacht und dir mitteilt, 14 Tage lang"; Planer IMMER informieren).
#
# Befund-Klasse (Planer 14.10.): DECODE-COLLECT hielt Prefills 7,5 s im Sammelfenster, obwohl D nichts rechnete
# (_d_pf_pending_tokens zaehlte lane-geparkte rids als "D prefillt"; Fix 95914428c2 in int24f8h4).
# Der Monitor paart jede Front-Zeile 'PDFLIP DECODE-COLLECT release ... waited_s=W' (W >= WAIT_MIN) mit dem D-Log des Boots
# (TP0 'Decode batch, #running-req: N' / 'Prefill batch, #new-seq: N'): lief im Fenster [release-W, release] auf D weder ein
# Decode-Batch mit N>=1 noch ein Prefill-Batch, war D idle => Wartezeit ohne Grund => BEFUND an den Planer.
# Zusatz (Nutzerorder 14.10. "er muss auch pruefen ob korrekt gesammelt wird fuer D und fuer P"): jedes Release wird gegen die Regeln
# des Codes (front.py _decode_collect) geprueft:
#   R1 zu frueh   : stage=window vor window_s bzw. stage=dcheck vor d_check_s freigegeben
#   R2 zu spaet   : Wartezeit > window_s + 2 s (Fenster haelt laenger als gesetzt)
#   R3 D falsch   : route=D obwohl p_bound=True oder tokens > X (Summe passt nicht unter X)
#   R4 P falsch   : route=P obwohl p_bound=False, tokens <= X und stage != window (kleine Menge haette nach D gehoert)
#   F-D           : nach route=D innerhalb 45 s weder D-Prefill-Batch (TP0) noch 'D-ADMIT' => Menge kommt auf D nicht an
#   F-P           : nach route=P innerhalb 150 s (Seat-Gate-Hold: 240 s) kein P-Prefill-Batch (PP0) => Menge wird auf P nicht prefillt
# Laeuft DAYS Tage, folgt Boot-Wechseln, schreibt Digest alle 30 min. Nur lesen + Zeile anhaengen.
#
# 15.10. (Auftrag nf-core-collect-1015, Aufgabe #119): D-BUSY-ZEUGEN. Der Befund 'D war IDLE' vom 15.10. 13:13:02Z (release
# waited_s=7.5 n=1 tokens=5302 X=6057 route=D stage=dcheck) war ein FEHLALARM des Monitors, nicht der Front: D dekodierte in dem
# Fenster mit bs=5-6 (Decode rank batch, 9034-9061) und prefillte danach pdflip-40-128 in zwei Chunks (13:12:56-13:13:04). Der Monitor
# kannte nur 'Decode batch, #running-req' (eine Zeile je ~64 Runden, fehlt, solange ein Prefill-Chunk die Runden anhaelt) und
# 'Prefill batch, #new-seq' (wird am ENDE des Chunks gestempelt, hier 13:13:04 = ausserhalb des Fensters). Jetzt zaehlen zusaetzlich:
#   - 'Decode rank batch ... t: <epoch> ... bs: N' (N>=1)  jede Runde, mit exakter Epoche
#   - '#969 EXTENT'                                         Beginn jedes Extend/Prefill-Chunks (Zeilenstempel = Start)
#   - 'PDFLIP-VRAM-PEAK ... phase=chunk ... t_unix_ms'            das Chunk-ENDE (exakt), Beginn aus dem EIGENEN EXTENT (siehe unten)
#
# 15.10. RUNDE 2 (Review NEIN, deskq/done/nf-core-collect-1015-review.md): die Runde-1-Fassung war blind fuer die echte Klasse (route=D, D leer,
# 14.10. 10:29:45Z / 10:30:14Z / 10:54:41Z): sie zaehlte das EXTENT der FREIGEGEBENEN Menge selbst (+1 s nach dem Release) und las t0_unix_ms
# als Chunk-Beginn. Regeln jetzt (Gesetz: D-Arbeit zaehlt nur, wenn sie BEGONNEN hat, BEVOR der Release fiel, und nicht zur freigegebenen Menge gehoert):
#   - Stempel der D-Zeilen sind auf die SEKUNDE ABGERUNDET (wirkliche Zeit in [s, s+1)). Ein Punktzeuge zaehlt nur, wenn s + 1 <= release
#     (sicher VOR dem Release) UND s >= Fensterbeginn (sicher im Fenster, RUNDE 3: SLACK_S = 0.0); nach dem Release keine Toleranz. Nur 'Decode rank batch ... t:' hat eine exakte Epoche
#     (Rundenbeginn): zaehlt in [release-waited, release].
#   - 't0_unix_ms' ist NICHT der Chunk-Beginn, sondern das Ende des vorigen Messfensters des VRAM-Instruments (schliesst die Leerlaufzeit vor dem
#     Chunk ein). Der Chunk-Beginn kommt aus dem EIGENEN '#969 EXTENT' derselben Menge: die chunk-Zeile folgt im D-Log ihrem EXTENT, Zuordnung
#     ueber die Zeilenzahl (rows == Summe der EXTENT-Zeilen, an 419 von 419 Paaren zweier Boots gemessen). Ohne Paar kein Intervall.
#   - Zeugen, deren rid zur freigegebenen Menge gehoert (D-ADMIT dieser rid [release, release+2 s] im Front-Log), zaehlen nie.
# 15.10. RUNDE 3 (Review r2 JA_MIT_AUFLAGE, deskq/done/nf-core-collect-1015-review-r2.md): die Vorwaerts-Toleranz SLACK_S=1.0 am Fensterbeginn liess am
# ENDE gestempelte Zeugen ('Prefill batch', 'Decode batch') aus [w0-1, w0) als D-Arbeit im Fenster gelten (26 echte Faelle der Klasse verschluckt,
# 14.10. bc35462240 10:23:07Z, a4856f6e5d 09:42:21Z). Jetzt "sicher im Fenster" auf BEIDEN Seiten. Zweites Kriterium (Review Abschnitt 3):
# 'D wurde im Fenster leer' bei stage in (dcheck, window): letzte sichere D-Arbeit > D_IDLE_TAIL_S vor dem Release und kein Chunk ohne Beginn,
# der ueber den Release laeuft.
# Gemessen: siehe deskq/done/nf-core-collect-1015.md, Abschnitt Runde 2 und 3.
import argparse, calendar, collections, glob, os, re, sys, time

EVID = "/spinning/docker-acceptance/nf/evidence"
EINGANG = "/spinning/gpu-arb/deskq/an-planer/eingang.log"
LOG = "/spinning/gpu-arb/deskq/wacht/nf_collect_wait_monitor.log"
PID = "/spinning/gpu-arb/deskq/wacht/nf_collect_wait_monitor.pid"
DAYS = 14
STOPFILE = "/spinning/docker-acceptance/nf/nf_dauer_stop"   # Planer 14.10.: ab dieser Datei ist D fuer Neuaufnahmen zu (admit_d=False) => Releases nicht pruefen
WAIT_MIN = 2.0            # ab so vielen Sekunden Sammelwartezeit wird geprueft
SLACK_S = 0.0             # RUNDE 3 (Review r2, HOCH): KEINE Toleranz am Fensterbeginn. Ein Stempel s liegt in [s, s+1); ein End-Stempel
                          # ('Prefill batch', 'Decode batch') mit s < w0 kann ganz VOR dem Fenster liegen => zaehlt nur bei s >= w0 (sicher im Fenster).
STAMP_S = 1.0             # Stempel s bedeutet wirkliche Zeit in [s, s+1): ein Punktzeuge liegt nur dann SICHER vor dem Release, wenn s + STAMP_S <= release
D_IDLE_TAIL_S = 3.0       # RUNDE 3, zweites Kriterium: so lange darf D vor dem Release schon ruhen (d-idle p99 der Front 1,28 s + Stempelgenauigkeit 1 s)
CHUNK_RUN_MAX_S = 15.0    # a chunk END without a paired begin counts as 'running over the release' only if it falls within this long after it (longest chunk seen ~11 s)
ADMIT_AFTER_S = 2.0       # D-ADMIT einer rid bis so lange nach dem Release = die rid gehoert zur freigegebenen Menge
CHUNK_MIN_OVERLAP_S = 0.5  # Chunk-Intervall muss das Wartefenster mind. so lange ueberlappen, um als D-Arbeit zu zaehlen
SETTLE_S = 6.0            # so lange auf D-Log-Zeilen nach dem Release warten, bevor entschieden wird
DIGEST_S = 1800
MAX_LINES_PER_10MIN = 20

RE_REL = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})\] .*PDFLIP DECODE-COLLECT release n=(\d+) tokens=(\d+) X=(\d+) "
                    r"waited_s=([\d.]+) route=(\S+) stage=(\S+) p_bound=(\S+) epoch=(\d+)")
RE_HOLD = re.compile(r"PDFLIP DECODE-COLLECT hold n=\d+ tokens=\d+ X=\d+ waited_s=[\d.]+ window_s=([\d.]+) d_check_s=([\d.]+) epoch=(\d+)")
RE_SEAT = re.compile(r"PDFLIP DECODE-COLLECT seat-gate hold taken=\d+ n=\d+ epoch=(\d+)")
RE_LHOLD = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})\] .*PDFLIP LANE-ARRIVE rid=\S+ .*action=hold")
RE_LRESUME = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})\] .*PDFLIP LANE-RESUME floor=")
RE_WAITSEAT = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})\] .*PDFLIP ARRIVAL-SEAT rid=\S+ verdict=wait_seat")
RE_ADMIT = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})\] .*PDFLIP D-ADMIT rid=(\S+)")
RE_PPRE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) PP0\] Prefill batch, #new-seq: (\d+)")
RE_DEC = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] Decode batch, #running-req: (\d+)")
RE_PRE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] Prefill batch, #new-seq: (\d+)")
# D-BUSY-ZEUGEN (15.10.): per-round / per-chunk instruments of TP0 (see header)
RE_RDEC = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] Decode rank batch, rank: 0, #round: \d+, t: ([\d.]+), bs: (\d+)")
RE_EXT = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] #969 EXTENT n=\d+ fwd=\d+ reqs=(\[.*\])")
RE_EXT_REQ = re.compile(r"\('([^']*)',\s*\d+,\s*\d+,\s*\d+,\s*(\d+)\)")   # (rid, cached, total, cached, rows)
RE_RANKPF = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] Prefill rank batch, #new-token: (\d+), #cached-token: \d+, #chunks: \d+, gpu-ms: ([\d.]+)")
# t0_unix_ms is read by nobody on purpose (it is the START of the instrument's window, not of the chunk; may be 'na')
RE_CHUNK = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=(\d+) .*?t_unix_ms=(\d+)")


def now_z():
    return time.strftime("%H:%M:%SZ", time.gmtime())


def ts(s, ms=0):
    return calendar.timegm(time.strptime(s, "%Y-%m-%d %H:%M:%S")) + ms / 1000.0


class Mon:
    def __init__(self, dry):
        self.dry = dry
        self.history_s = 7200.0              # so lange (im Takt des D-Logs) bleiben D-Zeugen im Speicher; --replay: unbegrenzt (Runde 2: ein Replay
                                             # las das D-Log vorab komplett und beschnitt dann alles, was vor dem Log-ENDE minus 2 h lag)
        self.busy = collections.deque()      # (epoch_s, kind, rids) D war beschaeftigt (Decode N>=1 / Prefill N>=1); rids nur beim EXTENT
        self.busy_iv = collections.deque()   # (extent_stempel, chunk_ende, rids) Chunk-Intervalle (Beginn aus dem eigenen EXTENT, Runde 2)
        self.last_ext = None                 # (stempel, rows_summe, rids) des juengsten EXTENT, wartet auf seine chunk-Zeile
        self.last_chunk = None               # (intervall-eintrag, rows) der juengsten chunk-Zeile, wartet auf ihre 'Prefill rank batch'-Zeile
        self.admits = collections.deque()    # (epoch_s, rid) D-ADMIT aus dem Front-Log (Zugehoerigkeit zur freigegebenen Menge)
        self.d_last = 0.0                    # juengster D-Log-Zeitstempel
        self.win, self.dchk = 15.0, 7.5
        self.stop_t = None                   # epoch der Stop-Datei (mtime), solange sie existiert
        self.seat_hold_epochs = set()
        self.lane_holds = []                 # Ankunftszeiten (epoch_s) an der Lane-Gate gehaltener Anfragen (Planer 14.10. 12:40Z)
        self.lane_credit = (0.0, 0.0)        # (LANE-RESUME-Zeit, Haltezeit) des juengsten Resume
        self.wait_seat = collections.deque()  # Zeitpunkte 'ARRIVAL-SEAT verdict=wait_seat' (Planer 14.10.: Sitzwarte, kein Verlust)
        self.seat_open = []                  # (release_t, f, src): route=D, Sitzwarte, D-ADMIT noch offen
        self.r2_pending = []                 # zurueckgestellte R2-Kandidaten (t, f, src, wall)
        self.p_pf =collections.deque()      # PP0-Prefill-Batches (epoch_s)
        self.d_pf = collections.deque()      # D-Prefill-Batches TP0 + D-ADMIT (epoch_s)
        self.follow = []                     # (t, route, f, src, wall)
        self.pending = []                    # (release_epoch, waited, fields, src, wall)
        self.stats = collections.Counter()
        self.waste_s = 0.0
        self.sent = []
        self.suppressed = 0
        self.last_digest = time.time()
        self.starts = time.time()

    def say(self, text, planer=True):
        line = "[%s] NF-MONITOR COLLECT-WARTEZEIT: %s" % (now_z(), text)
        if self.dry:
            print(line)
            return
        with open(LOG, "a") as f:
            f.write(line + "\n")
        if not planer:
            return
        t = time.time()
        self.sent = [s for s in self.sent if t - s < 600]
        if len(self.sent) >= MAX_LINES_PER_10MIN:
            self.suppressed += 1
            return
        self.sent.append(t)
        with open(EINGANG, "a") as f:
            f.write(line + "\n")

    def feed_d(self, line):
        m = RE_RDEC.match(line)
        if m:
            t = float(m.group(2))
            self.d_last = max(self.d_last, t)
            if int(m.group(3)) >= 1:
                self.busy.append((t, "decode-round", ()))
            return
        m = RE_EXT.match(line)
        if m:
            t = ts(m.group(1))
            self.d_last = max(self.d_last, t)
            reqs = RE_EXT_REQ.findall(m.group(2))
            rids = tuple(r for r, _ in reqs)
            self.busy.append((t, "extend", rids))
            self.last_ext = (t, sum(int(n) for _, n in reqs), rids)
            return
        m = RE_CHUNK.match(line)
        if m:
            self.d_last = max(self.d_last, ts(m.group(1)))
            b, ext, rows, self.last_ext = int(m.group(3)) / 1000.0, self.last_ext, int(m.group(2)), None
            # (1) the chunk END is exact (t_unix_ms): D worked up to b -> a point witness (exact epoch, like a decode round).
            # (2) an interval needs a START, and every start here is an UPPER bound of the true start (so [S, b] lies inside the real busy
            #     time, never beyond it): the stamp of the chunk's OWN #969 EXTENT + 1 s (the stamp is floored; rows == sum of the extent's
            #     rows pairs them) and/or b - gpu-ms of its 'Prefill rank batch' line (below). Neither available -> no interval, the end
            #     point still counts. NEVER t0_unix_ms (= start of the instrument's window, may precede the chunk by minutes or be 'na').
            #     The #969 probe is throttled in the code (schedule_batch.py: n <= 400 or n % 64 == 0), so after ~400 extends of a boot
            #     most chunks have no EXTENT line at all -- the rank-batch duration is the start source that does not run out.
            rids = ext[2] if ext is not None and ext[1] == rows else ()
            self.busy.append((b, "chunk-end", rids))
            entry = [ext[0] + STAMP_S if rids and b > ext[0] else None, b, rids]
            self.busy_iv.append(entry)
            self.last_chunk = (entry, rows)
            return
        m = RE_RANKPF.match(line)
        if m:
            t = ts(m.group(1))
            self.d_last = max(self.d_last, t)
            lc, self.last_chunk = self.last_chunk, None
            if lc is not None and lc[1] == int(m.group(2)):
                entry = lc[0]
                s2 = entry[1] - float(m.group(3)) / 1000.0
                entry[0] = s2 if entry[0] is None else min(entry[0], s2)
            # the rank-batch line itself is no witness (its end-stamp is the legacy 'Prefill batch' line's job); it only dates the chunk start
            return
        m = RE_DEC.match(line)
        if m:
            t = ts(m.group(1))
            self.d_last = max(self.d_last, t)
            if int(m.group(2)) >= 1:
                self.busy.append((t, "decode", ()))
            return
        m = RE_PRE.match(line)
        if m:
            t = ts(m.group(1))
            self.d_last = max(self.d_last, t)
            if int(m.group(2)) >= 1:
                self.busy.append((t, "prefill", ()))
                self.d_pf.append(t)

    def d_work(self, t, waited):
        """D work that BEGAN before the release at `t` (and is not the released set itself), as a Counter of witness kinds.

        Window = [t-waited, t]. D-log stamps are floored to the second (true time in [s, s+1)), so a point witness counts only when
        s + STAMP_S <= t (certainly before the release) and s >= t-waited-SLACK_S (SLACK_S = 0: certainly inside the window, round 3). Exact epochs ('Decode rank batch' t: = round start,
        chunk end t_unix_ms) count in [t-waited, t] (a chunk end needs CHUNK_MIN_OVERLAP_S inside the window). A chunk interval [S, end]
        (S = an upper bound of its true start, see feed_d) must overlap the window by CHUNK_MIN_OVERLAP_S. Witnesses of rids that were
        D-ADMITted in [t, t+ADMIT_AFTER_S] (the released set) never count.
        """
        w0 = t - waited
        released = {r for ta, r in self.admits if t <= ta <= t + ADMIT_AFTER_S}
        kinds = collections.Counter()
        for bt, k, rids in self.busy:
            if rids and released and set(rids) <= released:
                continue
            if k == "decode-round":
                ok = w0 <= bt <= t
            elif k == "chunk-end":
                ok = w0 + CHUNK_MIN_OVERLAP_S <= bt <= t
            else:
                ok = w0 - SLACK_S <= bt and bt + STAMP_S <= t
            if ok:
                kinds[k] += 1
        for a, b, rids in self.busy_iv:
            if a is None or (rids and released and set(rids) <= released):
                continue
            if min(b, t) - max(a, w0) >= CHUNK_MIN_OVERLAP_S:
                kinds["chunk"] += 1
        return kinds

    def d_tail(self, t):
        """RUNDE 3, second criterion ('D wurde im Fenster leer, Front hielt weiter'): (last, running) for a release at `t`.

        last = an UPPER bound of the time of the last D work that certainly happened before the release (released set excluded):
        a decode round / a chunk end at its exact epoch, a floored point witness at s + STAMP_S (its true time is in [s, s+1)), a
        chunk interval at min(end, t). running = a chunk that is not over at the release: one whose own begin is known to lie
        before the release and whose end lies after it, a chunk END after the release WITHOUT a paired begin (it may well have begun
        before the release -- the doubt always goes to 'D worked', never to an alarm), or the open EXTENT of a chunk whose line has
        not been written yet (live, the chunk line is end-stamped). The front answers a D gone idle with stage=d-idle after
        p50 0.13 s / p99 1.28 s; a tail beyond D_IDLE_TAIL_S with nothing running is a hold on an idle D."""
        released = {r for ta, r in self.admits if t <= ta <= t + ADMIT_AFTER_S}
        last, running = None, False

        def upd(x):
            nonlocal last
            last = x if last is None else max(last, x)

        for bt, k, rids in self.busy:
            if rids and released and set(rids) <= released:
                continue
            if k in ("decode-round", "chunk-end"):
                if bt <= t:
                    upd(bt)
            elif bt + STAMP_S <= t:
                upd(bt + STAMP_S)
        for a, b, rids in self.busy_iv:
            if rids and released and set(rids) <= released:
                continue
            if a is None:
                running = running or t < b <= t + CHUNK_RUN_MAX_S
            elif a <= t:
                if b > t:
                    running = running or b <= t + CHUNK_RUN_MAX_S
                else:
                    upd(b)
        le = self.last_ext
        if le is not None and t - 30.0 <= le[0] and le[0] + STAMP_S <= t and not (le[2] and released and set(le[2]) <= released):
            running = True
        return last, running

    def feed_p(self, line):
        m = RE_PPRE.match(line)
        if m and int(m.group(2)) >= 1:
            self.p_pf.append(ts(m.group(1)))

    def feed_front(self, line, src):
        m = RE_HOLD.search(line)
        if m:
            self.win, self.dchk = float(m.group(1)), float(m.group(2))
            return
        m = RE_SEAT.search(line)
        if m:
            self.seat_hold_epochs.add(int(m.group(1)))
            return
        m = RE_WAITSEAT.match(line)
        if m:
            self.wait_seat.append(ts(m.group(1), int(m.group(2))))
            return
        m = RE_LHOLD.match(line)
        if m:
            self.lane_holds.append(ts(m.group(1), int(m.group(2))))
            return
        m = RE_LRESUME.match(line)
        if m:
            tr = ts(m.group(1), int(m.group(2)))
            if self.lane_holds:
                self.lane_credit = (tr, tr - min(self.lane_holds))
                self.lane_holds = []
                self.r2_flush(resume_t=tr)
            return
        m = RE_ADMIT.match(line)
        if m:
            ta = ts(m.group(1), int(m.group(2)))
            self.d_pf.append(ta)
            self.admits.append((ta, m.group(3)))
            return
        m = RE_REL.match(line)
        if not m:
            return
        t = ts(m.group(1), int(m.group(2)))
        n, tok, x, waited = int(m.group(3)), int(m.group(4)), int(m.group(5)), float(m.group(6))
        route, stage, pb, ep = m.group(7), m.group(8), m.group(9) == "True", int(m.group(10))
        f = dict(n=n, tokens=tok, X=x, route=route, stage=stage, p_bound=pb, epoch=ep, waited=waited)
        if self.stop_t is not None and t >= self.stop_t - 0.5:
            self.stats["stopp_ausgenommen"] += 1    # Stopp laeuft: D nimmt nichts Neues auf, kein Befund
            return
        self.stats["releases"] += 1
        self.stats["wait_total_s"] += waited
        self.stats["route_" + route] += 1
        if n == 0:
            self.stats["fenster_ohne_inhalt"] += 1      # Planer 14.10.: gesondert zaehlen, F-D/F-P nur fuer n>0
            self.stats["fenster_ohne_inhalt_s"] += waited
        else:
            self._rules(t, f, src)
            self.follow.append((t, route, f, src, time.time()))
        if waited < WAIT_MIN:
            return
        self.stats["checked"] += 1
        self.pending.append((t, waited, f, src, time.time()))

    def _rule_hit(self, code, t, f, src, why):
        self.stats["rule_" + code] += 1
        self.say("BEFUND %s: release um %s n=%d tokens=%d X=%d waited_s=%.1f route=%s stage=%s p_bound=%s epoch=%d window_s=%.1f d_check_s=%.1f -- %s; "
                 "Quelle %s (Front-Zeile 'DECODE-COLLECT release')"
                 % (code, time.strftime("%H:%M:%SZ", time.gmtime(t)), f["n"], f["tokens"], f["X"], f["waited"], f["route"], f["stage"],
                    f["p_bound"], f["epoch"], self.win, self.dchk, why, src))

    def _rules(self, t, f, src):
        w, st, rt = f["waited"], f["stage"], f["route"]
        if st == "window" and w < self.win - 0.3:
            self._rule_hit("R1-ZU-FRUEH", t, f, src, "stage=window, aber nur %.1f s gesammelt (Fenster %.1f s)" % (w, self.win))
        if st == "dcheck" and w < self.dchk - 0.3:
            self._rule_hit("R1-ZU-FRUEH", t, f, src, "stage=dcheck, aber nur %.1f s gesammelt (D-Check %.1f s)" % (w, self.dchk))
        if w > self.win + 2.0:
            # Planer 14.10.: waited_s zaehlt ab Ankunft und enthaelt die Lane-Haltezeit. LANE-RESUME steht im Front-Log meist
            # wenige ms NACH dem Release -> Entscheidung zurueckstellen (r2_flush), bis das Resume kommt oder 3 s um sind.
            self.r2_pending.append((t, f, src, time.time()))
        if rt == "D" and f["p_bound"]:
            self._rule_hit("R3-D-FALSCH", t, f, src, "route=D, obwohl p_bound=True (P-gebundene Anfrage darf nicht nach D)")
        if rt == "D" and f["tokens"] > f["X"]:
            self._rule_hit("R3-D-FALSCH", t, f, src, "route=D, obwohl Summe %d > X=%d" % (f["tokens"], f["X"]))
        if rt == "P" and not f["p_bound"] and f["tokens"] <= f["X"] and st != "window":
            self._rule_hit("R4-P-FALSCH", t, f, src, "route=P fuer eine kleine Menge (%d <= X=%d, nicht P-gebunden, stage=%s): haette nach D gehoert"
                           % (f["tokens"], f["X"], st))

    def r2_flush(self, resume_t=None, force=False):
        keep = []
        now = time.time()
        for t, f, src, wall in self.r2_pending:
            tr, held = self.lane_credit
            if resume_t is not None and abs(resume_t - t) <= 5.0 and f["waited"] - held <= self.win + 2.0:
                self.stats["lane_gehalten"] += 1
                self.stats["lane_gehalten_s"] += held
            elif resume_t is None and not force and now - wall < 3.0 and not (0.0 <= t - tr <= 5.0):
                keep.append((t, f, src, wall))
            elif resume_t is None and 0.0 <= t - tr <= 5.0 and f["waited"] - held <= self.win + 2.0:
                self.stats["lane_gehalten"] += 1
                self.stats["lane_gehalten_s"] += held
            elif resume_t is not None and abs(resume_t - t) > 5.0:
                keep.append((t, f, src, wall))
            else:
                h = held if abs(tr - t) <= 5.0 else 0.0
                self._rule_hit("R2-ZU-SPAET", t, f, src, "gesammelt %.1f s > Fenster %.1f s + 2 s (Lane-Haltezeit abgezogen: %.1f s)"
                               % (f["waited"], self.win, h))
        self.r2_pending = keep

    def follow_up(self, force=False):
        keep = []
        now = time.time()
        for t, route, f, src, wall in self.follow:
            hz = 45.0 if route == "D" else (240.0 if f["epoch"] in self.seat_hold_epochs else 150.0)
            if not force and now - wall < hz:
                keep.append((t, route, f, src, wall))
                continue
            dq = self.d_pf if route == "D" else self.p_pf
            seen = any(t - 1.0 <= x <= t + hz for x in dq)
            self.stats["follow_" + route] += 1
            if seen:
                self.stats["follow_ok_" + route] += 1
                continue
            if route == "D" and any(t - 0.5 <= x <= t + hz for x in self.wait_seat):
                self.stats["sitzwarte_D"] += 1          # Planer 14.10.: wait_seat nach dem Release = Sitzwarte, kein F-D
                self.seat_open.append((t, f, src))
                continue
            self.stats["follow_miss_" + route] += 1
            if route == "D":
                self._rule_hit("F-D-NICHT-ANGEKOMMEN", t, f, src,
                               "nach route=D innerhalb %.0f s weder D-Prefill-Batch (TP0) noch 'D-ADMIT' im Front-Log -- die gesammelte Menge kam auf D nicht an" % hz)
            else:
                self._rule_hit("F-P-NICHT-PREFILLT", t, f, src,
                               "nach route=P innerhalb %.0f s kein P-Prefill-Batch (PP0) -- die gesammelte Menge wird auf P nicht prefillt%s"
                               % (hz, " (Seat-Gate-Hold war in dieser Epoche)" if f["epoch"] in self.seat_hold_epochs else ""))
        self.follow = keep
        still = []
        for t0, f0, src0 in self.seat_open:
            adm = [x for x in self.d_pf if x >= t0]
            if adm:
                d = min(adm) - t0
                self.stats["sitzwarte_aufgeloest"] += 1
                self.stats["sitzwarte_s"] += d
                self.stats["sitzwarte_max_s"] = max(self.stats["sitzwarte_max_s"], d)
            elif time.time() - t0 > 300.0 and not force:
                self._rule_hit("F-D-SITZWARTE-LANG", t0, f0, src0, "Sitzwarte (wait_seat) seit >300 s ohne D-ADMIT")
            else:
                still.append((t0, f0, src0))
        self.seat_open = still
        while self.wait_seat and self.wait_seat[0] < time.time() - 7200:
            self.wait_seat.popleft()
        while self.p_pf and self.p_pf[0] < time.time() - 7200:
            self.p_pf.popleft()
        while self.d_pf and self.d_pf[0] < time.time() - 7200:
            self.d_pf.popleft()

    def settle(self, force=False):
        keep = []
        ref = self.d_last or time.time()          # the D log's own clock (live: ~now; replay of an old log: its end), never the wall clock
        while self.busy and self.busy[0][0] < ref - self.history_s:   # nur 2 h Rueckblick halten
            self.busy.popleft()
        while self.busy_iv and self.busy_iv[0][1] < ref - self.history_s:
            self.busy_iv.popleft()
        while self.admits and self.admits[0][0] < ref - self.history_s:
            self.admits.popleft()
        for t, waited, f, src, wall in self.pending:
            ready = force or self.d_last >= t + 1.0 or time.time() - wall >= SETTLE_S
            if not ready:
                keep.append((t, waited, f, src, wall))
                continue
            lo, hi = t - waited - SLACK_S, t
            kinds = self.d_work(t, waited)
            if kinds:
                self.stats["justified"] += 1
                self.stats["justified_s"] += waited
                if f["stage"] in ("dcheck", "window"):
                    self._d_went_idle(t, waited, f, src)
                continue
            self.stats["idle_waits"] += 1
            self.stats["idle_wait_s"] += waited
            self.waste_s += waited
            self.say("BEFUND D war IDLE waehrend der Sammelwartezeit: release um %s waited_s=%.1f n=%d tokens=%d X=%d route=%s stage=%s "
                     "p_bound=%s epoch=%d -- im D-Log (TP0) in [%s .. %s] kein Decode (Batch #running-req>=1 / Rank-Runde bs>=1), kein Extend (#969 EXTENT), kein Chunk-Intervall, kein Prefill-Batch; "
                     "Quelle %s (Front-Zeile 'DECODE-COLLECT release', D.log desselben Boots)"
                     % (time.strftime("%H:%M:%SZ", time.gmtime(t)), waited, f["n"], f["tokens"], f["X"], f["route"], f["stage"],
                        f["p_bound"], f["epoch"], time.strftime("%H:%M:%S", time.gmtime(lo)), time.strftime("%H:%M:%S", time.gmtime(hi)), src))
        self.pending = keep

    def _d_went_idle(self, t, waited, f, src):
        """Second criterion (round 3): D worked in the window but fell idle D_IDLE_TAIL_S or more before the release, nothing running."""
        last, running = self.d_tail(t)
        if last is None or running or t - last <= D_IDLE_TAIL_S:
            return
        tail = t - last
        if f["p_bound"]:
            # a P-bound set collects for P by design (review r2, 15.10. 11:52:08Z: route=P stage=window p_bound=True): counted, no message to the Planer
            self.stats["d_leer_p_bound"] += 1
            self.say("D wurde im Fenster leer (p_bound=True, nur gezaehlt): release um %s waited_s=%.1f, letzte sichere D-Arbeit %.1f s vor dem Release; Quelle %s"
                     % (time.strftime("%H:%M:%SZ", time.gmtime(t)), waited, tail, src), planer=False)
            return
        self.stats["d_leer"] += 1
        self.stats["d_leer_s"] += tail
        self.say("BEFUND D wurde im Fenster leer, Front hielt weiter: release um %s waited_s=%.1f n=%d tokens=%d X=%d route=%s stage=%s p_bound=%s epoch=%d -- "
                 "die letzte sichere D-Arbeit (Runde / Chunk-Ende / Stempel+1 s) liegt %.1f s vor dem Release (Schwelle %.1f s), kein Chunk laeuft ueber den Release; "
                 "die Front haette bei leerem D mit stage=d-idle sofort freigeben muessen (p99 1,3 s); Quelle %s (Front-Zeile 'DECODE-COLLECT release', D.log desselben Boots)"
                 % (time.strftime("%H:%M:%SZ", time.gmtime(t)), waited, f["n"], f["tokens"], f["X"], f["route"], f["stage"], f["p_bound"],
                    f["epoch"], tail, D_IDLE_TAIL_S, src))

    def digest(self, final=False):
        s = self.stats
        txt = ("%s: Releases=%d, davon Wartezeit>=%.1fs=%d | begruendet (D rechnete)=%d (%.0f s) | OHNE GRUND (D idle)=%d (%.0f s verschenkt) | "
               "Gesamtwartezeit aller Releases=%.0f s | route D=%d P=%d | Regelverstoesse R1 zu frueh=%d R2 zu spaet=%d R3 D falsch=%d R4 P falsch=%d | "
               "Ankunft D: ok %d / verfehlt %d | Ankunft P: ok %d / verfehlt %d | Fenster ohne Inhalt (n=0)=%d (%.0f s) | D im Fenster leer geworden (Front hielt weiter)=%d (%.0f s; p_bound nur gezaehlt: %d) | Lane gehalten (kein R2)=%d (%.0f s) | Sitzwarte D (kein F-D)=%d, aufgeloest %d, Summe %.0f s, max %.0f s | im Stopp ausgenommen=%d | unterdrueckte Einzelzeilen=%d"
               % ("ABSCHLUSS" if final else "DIGEST", s["releases"], WAIT_MIN, s["checked"], s["justified"], s["justified_s"],
                  s["idle_waits"], s["idle_wait_s"], s["wait_total_s"], s["route_D"], s["route_P"], s["rule_R1-ZU-FRUEH"], s["rule_R2-ZU-SPAET"],
                  s["rule_R3-D-FALSCH"], s["rule_R4-P-FALSCH"], s["follow_ok_D"], s["follow_miss_D"], s["follow_ok_P"], s["follow_miss_P"],
                  s["fenster_ohne_inhalt"], s["fenster_ohne_inhalt_s"], s["d_leer"], s["d_leer_s"], s["d_leer_p_bound"], s["lane_gehalten"], s["lane_gehalten_s"], s["sitzwarte_D"], s["sitzwarte_aufgeloest"], s["sitzwarte_s"], s["sitzwarte_max_s"], s["stopp_ausgenommen"], self.suppressed))
        self.suppressed = 0
        line = "[%s] NF-MONITOR COLLECT-WARTEZEIT: %s" % (now_z(), txt)
        if self.dry:
            print(line)
        else:
            with open(LOG, "a") as f:
                f.write(line + "\n")
            with open(EINGANG, "a") as f:
                f.write(line + "\n")


def newest_front():
    fs = glob.glob(os.path.join(EVID, "*.front.log"))
    return max(fs, key=os.path.getmtime) if fs else None


def read_new(fh, fn, limit=200000):
    n = 0
    for line in fh:
        fn(line)
        n += 1
        if n > limit:
            break


def run_live():
    if os.path.exists(PID):
        try:
            old = int(open(PID).read().strip())
            os.kill(old, 0)
            print("already running (pid %d)" % old)
            return 0
        except Exception:
            pass
    open(PID, "w").write(str(os.getpid()))
    deadline = time.time() + DAYS * 86400
    mon = Mon(dry=False)
    mon.say("GESTARTET fuer %d Tage bis %s. Prueft jede Front-Zeile 'DECODE-COLLECT release' mit waited_s>=%.1f gegen das D-Log (TP0): "
            "kein Decode/Prefill im Fenster => D war idle => BEFUND (Klasse: Sammelfenster zaehlt geparkte rids als Decode, Planer-Fix 95914428c2 in int24f8h4). "
            "ZUSATZ (Nutzer: korrekt sammeln fuer D und P): jedes Release gegen die Code-Regeln R1 zu frueh / R2 zu spaet / R3 D falsch / R4 P falsch, "
            "danach F-D (Menge kommt auf D an: D-Prefill-Batch/D-ADMIT binnen 45 s) und F-P (P-Prefill-Batch PP0 binnen 150 s, mit Seat-Gate-Hold 240 s). "
            "Digest alle %d min. Quelle: neuestes %s/*.front.log + zugehoerige *.D.log und *.P.log."
            % (DAYS, time.strftime("%Y-%m-%d %H:%MZ", time.gmtime(deadline)), WAIT_MIN, DIGEST_S // 60, EVID))
    cur, fh_f, fh_d, fh_p = None, None, None, None
    first = True
    while time.time() < deadline:
        nf = newest_front()
        if nf and nf != cur:
            for h in (fh_f, fh_d, fh_p):
                if h:
                    h.close()
            if cur is not None:
                mon.stats["boot_wechsel_offen"] += len(mon.follow)
                mon.follow = []
                mon.pending = []
            cur = nf
            fh_f = open(cur, "r", errors="replace")
            dpath = cur[:-len("front.log")] + "D.log"
            fh_d = open(dpath, "r", errors="replace") if os.path.exists(dpath) else None
            ppath = cur[:-len("front.log")] + "P.log"
            fh_p = open(ppath, "r", errors="replace") if os.path.exists(ppath) else None
            if first:      # allererster Start: ab Ende, Altbestand steht im Replay
                fh_f.seek(0, 2)
                if fh_d:
                    # 10 min D-Log zurueck lesen, damit ein Release kurz nach dem Start nicht ohne Gegenstelle bleibt
                    size = fh_d.seek(0, 2)
                    fh_d.seek(max(0, size - 4_000_000))
                    fh_d.readline()
                if fh_p:
                    size = fh_p.seek(0, 2)
                    fh_p.seek(max(0, size - 4_000_000))
                    fh_p.readline()
                first = False
            mon.say("LOG %s" % os.path.basename(cur), planer=False)
        if cur and fh_d is None:      # D.log/P.log entstehen nach front.log: immer nachholen, sonst fehlt die Gegenstelle fuer immer
            dpath = cur[:-len("front.log")] + "D.log"
            if os.path.exists(dpath):
                fh_d = open(dpath, "r", errors="replace")
        if cur and fh_p is None:
            ppath = cur[:-len("front.log")] + "P.log"
            if os.path.exists(ppath):
                fh_p = open(ppath, "r", errors="replace")
        if fh_d:
            read_new(fh_d, mon.feed_d)
        try:
            mon.stop_t = os.path.getmtime(STOPFILE)
        except OSError:
            mon.stop_t = None
        if fh_p:
            read_new(fh_p, mon.feed_p)
        if fh_f:
            read_new(fh_f, lambda l: mon.feed_front(l, os.path.basename(cur)[-24:]))
        mon.settle()
        mon.r2_flush()
        mon.follow_up()
        if time.time() - mon.last_digest >= DIGEST_S:
            mon.digest()
            mon.last_digest = time.time()
        time.sleep(1)
    mon.settle(force=True)
    mon.r2_flush(force=True)
    mon.follow_up(force=True)
    mon.digest(final=True)
    mon.say("14 Tage um, Monitor beendet sich. Verlaengerung nur auf Nutzerwort.")
    try:
        os.remove(PID)
    except OSError:
        pass
    return 0


def run_replay(front):
    mon = Mon(dry=True)
    mon.history_s = float("inf")
    dpath = front[:-len("front.log")] + "D.log"
    with open(dpath, "r", errors="replace") as f:
        read_new(f, mon.feed_d, limit=10**9)
    ppath = front[:-len("front.log")] + "P.log"
    if os.path.exists(ppath):
        with open(ppath, "r", errors="replace") as f:
            read_new(f, mon.feed_p, limit=10**9)
    with open(front, "r", errors="replace") as f:
        read_new(f, lambda l: mon.feed_front(l, os.path.basename(front)[-24:]), limit=10**9)
    mon.settle(force=True)
    mon.r2_flush(force=True)
    mon.follow_up(force=True)
    mon.digest()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="check front.log once against its D.log (no report to the planner)")
    a = ap.parse_args()
    sys.exit(run_replay(a.replay) if a.replay else run_live())
