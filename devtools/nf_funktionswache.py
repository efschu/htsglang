#!/usr/bin/env python3
"""nf_funktionswache.py -- ereignisgetriebene Funktionswache fuer den NF-Dauerserver.

NUR LESEN: kein docker, kein stop/boot, kein kill. Laeuft mit nice 19, liest die Logs
des neuesten *dauer*-Boots inkrementell (Offset/seek) und schreibt auf stdout NUR bei
Regelverstoessen je eine Zeile (max. 300 Zeichen):
    <UTC-Zeit> <ALARM|WARN> <Regel> <Text>
Alle 60 s kommt der Lebensbeweis als JSON nach
/root/.claude/jobs/aef87d47/tmp/funktionswache.status.json (nie auf stdout).

Betriebsarten
  --live (Standard)   Offset-Polling alle 1 s; startet am Dateiende. --from-start liest
                      nur den Front-Log einmal von vorn (D/P ab Ende). Alle 30 s Pruefung
                      auf neueren Boot -> Zeile "BOOT-WECHSEL <name>".
  --replay FRONT.log --from "YYYY-MM-DD HH:MM:SS" [--to "..."] [--rules V1,V5]
                      wertet Front + gleichnamige D/P-Logs nur im Zeitfenster aus (Binaer-
                      suche auf den Zeitstempel, Streaming, nichts komplett im RAM), druckt
                      die Ereignisse und am Ende eine Zaehlung.
  --selftest          synthetische Logzeilen fuer V1..V7, Exit-Code != 0 bei Fehlschlag.

Regeln (Quelle = verifizierte Logzeilen des Boots dauer10121858_09235ed28a)
  V1 SUMME-ROUTE (ALARM)   "WEG2 DECODE-COLLECT release n= tokens=T X= waited_s= route=D" mit T > X.
  V2 P-ENTSCHEID-GEKAPERT (ALARM)
        Nach "release ... route=P epoch=E" folgen "WEG2 SHORT-KEPT rid= uncached= epoch=E":
        die Summenregel schickte den Satz nach P, die Requests blieben aber fuer D.
        Mitglieder des Satzes = die ersten <= n SHORT-KEPT-Zeilen derselben Epoche innerhalb
        1,0 s nach der Freigabe, deren uncached-Summe <= tokens bleibt (SHORT-KEPT spaeterer
        Neuankuenfte in derselben Epoche sind KEINE Satzmitglieder, sonst Fehlalarm: im
        D-Phasenbetrieb ist SHORT-KEPT sonst alltaeglich).
        Gegenprobe V2-OK: "COLLECT-P-QUEUED rid=" (geplanter Fix; Zeilenform NICHT
        verifizierbar, da im Boot noch nicht vorhanden -> reine Substring-Suche) oder, als
        V2-OK-handback, "PARK-HANDBACK rid=" / "P-BATCH-TAKES rid=" des Mitglieds binnen 5 s
        (dann ist der Request doch bei P gelandet, z. B. 21:00:05 Epoche 88). Erst wenn nach 5 s
        ein Mitglied weder noch hat, kommt der ALARM (mit rids + Summe uncached).
  V3 D-PREFILL-UEBER-X (ALARM) / V3 D-PREFILL-FEHLPREIS (WARN)   -- RID-GENAU, seit 15.10. (nf-core-dx-r2)
        Die alte V3 summierte alle D-TP0-Prefill-Batches im gleitenden 60-s-Fenster gegen das
        zuletzt gesehene X-ROUTE-X. Das ist nicht das Gesetz: X ist ein Preis je Entscheidung
        (Einzelrequest / DECODE-COLLECT-Satz / X-SUM-PRICE-Satz), je Ankunft verschieden (X_k),
        queued Shorts werden mit X_busy/X_1 neu entschieden, und die Chunks EINES Requests sind
        mehrere Batches. Boot 124245 13:11-13:13Z: 8 Fehlalarme bei 0 Verstoessen.
        Jetzt, je rid und nur fuer die ERSTE Aufnahme ("WEG2 X-GATE rid= ... verdict=admit", D TP0;
        spaetere admit-Zeilen derselben rid sind D-Park/Resume und liegen unter dem statischen X):
          ALARM  uncached > X_Entscheidung + TOL (TOL = 64 = eine Seite, wie das D-Gate), verdict=admit.
                 X_Entscheidung = "front_x=" der X-GATE-Zeile selbst (ab FRONT-X-MIT, Quelle X-GATE),
                 sonst das LETZTE Entscheidungs-X der Front fuer diese rid mit Zeitstempel <= Gate-Zeile
                 (X-ROUTE X=, ARRIVAL-SEAT QUEUED-SHORT X=, D-SHORT-DRAIN X=, X-IDLE-REGRANT X_idle=,
                 DECODE-COLLECT to-d X=; Quelle Front). verdict=W31 ist KEIN Alarm (benannte Verweigerung).
          WARN   FEHLPREIS: uncached - gepreist > 1024 (gepreist = "front_priced=" der Zeile, sonst der letzte
                 Preis der Front: X-ROUTE est_uncached / X-EXACT-REPRICE -> / QUEUED-SHORT uncached).
                 Unter X ist das kein Gesetzesbruch, aber der Rand der Absicherung (weg2-40-127: 237 -> 1517).
        Reihenfolge-Schutz: sieht die Wache die D-Zeile vor der Front-Entscheidung, wartet sie bis 5 s
        Log-Zeit; ohne jede X-Quelle zaehlt "V3-ohne-X" (nie ein Alarm aus Unwissen).
  V4 FLIP-LATENZ (WARN)  Freigabe route=P bis zum naechsten Flip-Beginn der Front > 20 s.
        GEWAEHLTE FLIP-MARKE: "WEG2-FLIP begin epoch=N sleep=X wake=Y ..." (geprueft:
        21:00:05.437 release -> 21:00:06.800 begin; DP-WAIT erklaert hold als "before
        WEG2-FLIP begin"). Steht dazwischen "DECODE-COLLECT seat-gate hold", wird
        "(Sitz-Sperre)" angehaengt. Meldung schon bei +20 s ohne Flip (Laufzeit) und bei
        Flip-Beginn mit der echten Latenz, falls > 20 s.
  V5 FENSTER-UEBERZUG (WARN)  release route=P mit waited_s > 16.5 (Fenster 15 s).
  V6 FEHLER (ALARM)  front: Level ERROR/CRITICAL, Traceback, RANK_EXCEPTION, memory leak
        detected, CUDA out of memory/OutOfMemoryError, PROGRESS-STALL, DEADMAN; P/D: dieselben
        Marken ohne Level (P/D-Zeilen haben keinen Level) plus Wort OOM/ERROR (ausgenommen
        Prosa: WEG2-LAUNCH, CORRIDOR-ADMISSION, PP-CUT, "D-OOM"-Zaehlerzeilen). Dedupe nach
        den ersten 80 Zeichen der Meldung (Ziffernfolgen maskiert), Wiederholungen nur als
        "(xN)" alle 5 min.
  V7 FLIP-ZEIT (WARN)  "WEG2-FLIP-TIMELINE epoch= slept= woke= ... done@MS" (dieselbe Zeile wie
        beim Waechter nf_lebenszeichen.py) mit done > 8000 ms.

Dedupe: dieselbe Regel + Epoche hoechstens alle 30 s (Zaehler zaehlen trotzdem).
Zeitbasis: Log-Stempel (UTC), nicht Wanduhr; "jetzt" = juengster gelesener Stempel.
"""
import argparse
import calendar
import collections
import glob
import heapq
import json
import os
import re
import resource
import sys
import time

EVDIR = "/spinning/docker-acceptance/nf/evidence"
STATUS = "/root/.claude/jobs/aef87d47/tmp/funktionswache.status.json"
MAXLEN = 300
DEDUPE_S = 30.0
V3_TOL = 64            # eine Seite (server_args.page_size der NF-D): die systematische Rundung des Praefixes
V3_MISPRICE = 1024     # realised - gepreist, ab dem der Fehlpreis gemeldet wird (16 Seiten, >10x die Rundung)
V3_GRACE_S = 5.0       # so lange wartet eine D-Zeile auf die Front-Entscheidung (Lesereihenfolge)
V3_KEEP_S = 1800.0     # so lange bleibt eine Entscheidung im Speicher
V4_MAX = 20.0
V5_MAX = 16.5
V7_MAX_MS = 8000
V2_MEMBER_S = 1.0
V2_GRACE_S = 5.0
V6_REPEAT_S = 300.0
RULES = ("V1", "V2", "V3", "V4", "V5", "V6", "V7")

KV = re.compile(r"(\w+)=([^\s:,;)\]'\"]+)")
RX_FRONT_ERR = re.compile(
    r"^\[[^\]]+\] (?:ERROR|CRITICAL) |Traceback \(most recent|RANK_EXCEPTION|memory leak detected"
    r"|CUDA out of memory|OutOfMemoryError|PROGRESS-STALL|DEADMAN")
RX_DP_ERR = re.compile(
    r"Traceback \(most recent|RANK_EXCEPTION|memory leak detected|CUDA out of memory|OutOfMemoryError"
    r"|PROGRESS-STALL|DEADMAN|hit an exception|(?<![\w-])OOM(?![\w-])|\bERROR\b")
DP_PROSE = ("WEG2-LAUNCH", "CORRIDOR-ADMISSION", "PP-CUT", "D-OOM")
DP_KEYS = ("WEG2 X-GATE rid=", "Traceback (most recent", "RANK_EXCEPTION", "memory leak detected",
           "CUDA out of memory", "OutOfMemoryError", "PROGRESS-STALL", "DEADMAN", "hit an exception",
           "OOM", "ERROR")
RX_REPRICE = re.compile(r"est_uncached \d+ -> (\d+)")
RX_RIDS = re.compile(r"weg2-\d+-\d+")
RX_GATE = re.compile(r"WEG2 X-GATE rid=(\S+) uncached=(\d+) X=(\d+) .*?verdict=(\S+)")
RX_TL = re.compile(r"WEG2-FLIP-TIMELINE epoch=(\d+) slept=(\w) woke=(\w) ms-from-begin: (.*?) \(stage@t")
RX_STAGE = re.compile(r"([a-z\-]+)@(\d+)")

_tc = {}


def parse_ts(s, ms=0):
    v = _tc.get(s)
    if v is None:
        v = calendar.timegm(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
        if len(_tc) > 20000:
            _tc.clear()
        _tc[s] = v
    return v + ms / 1000.0


def line_t(l):
    """Zeitstempel einer Logzeile ('[YYYY-MM-DD HH:MM:SS(,mmm)] ...') oder None."""
    if len(l) < 20 or l[0] != "[" or l[11] != " " or l[5] != "-":
        return None
    try:
        ms = int(l[21:24]) if l[20] == "," else 0
        return parse_ts(l[1:20], ms)
    except (ValueError, IndexError):
        return None


def fmt_t(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def hms(t):
    return time.strftime("%H:%M:%S", time.gmtime(t))


def kv(l):
    return dict(KV.findall(l))


def rid_num(r):
    try:
        return tuple(int(x) for x in r.split("-")[1:])
    except ValueError:
        return (0,)


class Wache:
    def __init__(self, out, rules=RULES):
        self.out = out
        self.rules = set(rules)
        self.reset()
        self.cnt = collections.Counter()   # erkannt (vor Dedupe)
        self.em = collections.Counter()    # ausgegeben
        self.last_event = {}
        self.parse_err = collections.Counter()
        self.last_stamp = {"front": None, "D": None, "P": None}
        self.last_emit_t = None

    def reset(self):
        self.now = 0.0
        self.X = None
        self.epoch = None
        self.ded = {}
        self.rel = None
        self.p_ok = {}
        self.v4 = None
        self.xdec = {}                     # rid -> [(t, quelle, X)]  Entscheidungs-X der Front (V3)
        self.xpriced = {}                  # rid -> [(t, uncached)]   Preis der Front (V3)
        self.v3_seen = {}                  # rid -> t der ersten admit-Zeile (nur die erste zaehlt)
        self.v3_pend = collections.deque()  # (t, rec) D-Zeilen, die auf die Front-Entscheidung warten
        self.seen_tl = set()
        self.v6 = {}

    # ---------------------------------------------------------------- Ausgabe
    def emit(self, rule, level, t, text, dkey=None, count=True):
        if count:
            self.cnt[rule] += 1
        self.last_event[rule] = fmt_t(t)
        if dkey is not False:
            k = (rule, dkey if dkey is not None else self.epoch)
            last = self.ded.get(k)
            if last is not None and t - last < DEDUPE_S:
                return False
            self.ded[k] = t
            if len(self.ded) > 5000:
                self.ded = {a: b for a, b in self.ded.items() if t - b < DEDUPE_S}
        if rule not in self.rules:
            return False
        line = ("%s %s %s %s" % (fmt_t(t), level, rule, text))
        if len(line) > MAXLEN:
            line = line[:MAXLEN - 1] + "~"
        self.em[rule] += 1
        self.last_emit_t = fmt_t(t)
        self.out(line)
        return True

    # ------------------------------------------------------------------ Zeit
    def advance(self, t):
        if t > self.now:
            self.now = t
        now = self.now
        r = self.rel
        if r and not r["done"] and now >= r["t"] + V2_GRACE_S:
            self.finish_rel()
        v4 = self.v4
        if v4 and not v4["warn"] and now - v4["t"] > V4_MAX:
            v4["warn"] = True
            self.cnt_v4(v4)
            self.emit("V4", "WARN", now, "FLIP-LATENZ epoch=%s seit release route=P (%s) kein Flip-Beginn nach %.0f s%s"
                      % (v4["epoch"], hms(v4["t"]), now - v4["t"], " (Sitz-Sperre)" if v4["hold"] else ""),
                      dkey=(v4["epoch"], "lauf"))
        while self.v3_pend and now >= self.v3_pend[0][0] + V3_GRACE_S:
            t0, rec = self.v3_pend.popleft()
            self.v3_judge(t0, rec, final=True)
        if len(self.xdec) > 4000 or len(self.v3_seen) > 8000:
            self.v3_prune(now)
        for k in [k for k, v in self.v6.items() if v["n"] > v["n_emit"] and now - v["t_emit"] >= V6_REPEAT_S]:
            v = self.v6[k]
            v["n_emit"] = v["n"]
            v["t_emit"] = now
            self.emit("V6", "ALARM", now, "FEHLER %s (x%d) %s" % (v["src"], v["n"], v["msg"][:200]), dkey=False, count=False)

    def cnt_v4(self, v4):
        if not v4["counted"]:
            v4["counted"] = True

    # ---------------------------------------------------------------- Front
    def front_line(self, l):
        t = line_t(l)
        if t is None:
            if "Traceback (most recent" in l:
                self.on_err("front", self.now, l)
            return
        self.last_stamp["front"] = l[1:20]
        self.advance(t)
        if "DECODE-COLLECT" in l:
            if " release " in l:
                self.on_release(t, l)
            elif "seat-gate hold" in l and self.v4:
                self.v4["hold"] = True
        elif "SHORT-KEPT" in l:
            self.on_short_kept(t, l)
        elif "COLLECT-P-QUEUED" in l:
            d = kv(l)
            if d.get("rid"):
                self.p_ok[d["rid"]] = t
            self.cnt["V2-OK-queued"] += 1
        elif "PARK-HANDBACK" in l or "P-BATCH-TAKES" in l:
            d = kv(l)
            if d.get("rid"):
                self.p_ok[d["rid"]] = t
        elif "WEG2 X-ROUTE " in l:
            d = kv(l)
            x = d.get("X")
            if x and x.isdigit():
                self.X = int(x)
                self.v3_note_decision(t, d.get("rid"), "X-ROUTE", int(x), d.get("est_uncached"))
        elif "QUEUED-SHORT" in l and "ARRIVAL-SEAT" in l:
            d = kv(l)
            if d.get("X", "").isdigit():
                self.v3_note_decision(t, d.get("rid"), "QUEUED-SHORT", int(d["X"]), d.get("uncached"))
        elif "X-EXACT-REPRICE" in l:
            m = RX_REPRICE.search(l)
            rid = kv(l).get("rid")
            if m and rid:
                self.v3_note_price(t, rid, int(m[1]))
        elif "D-SHORT-DRAIN" in l or "X-IDLE-REGRANT" in l or "DECODE-COLLECT to-d" in l:
            d = kv(l)
            x = d.get("X") or d.get("X_idle")
            if x and x.isdigit():
                src = "D-SHORT-DRAIN" if "D-SHORT-DRAIN" in l else "X-IDLE-REGRANT" if "X-IDLE-REGRANT" in l else "COLLECT-to-d"
                for rid in RX_RIDS.findall(l[l.find("rids="):] if "rids=" in l else ""):
                    self.v3_note_decision(t, rid, src, int(x), None)
        elif "WEG2-FLIP begin" in l:
            self.on_flip_begin(t, l)
        elif "WEG2-FLIP-TIMELINE" in l:
            self.on_timeline(t, l)
        if "ERROR" in l or "Trace" in l or "RANK_" in l or "leak" in l or "memory" in l or "STALL" in l or "DEADMAN" in l:
            if RX_FRONT_ERR.search(l):
                self.on_err("front", t, l)

    def on_release(self, t, l):
        d = kv(l)
        try:
            n = int(d["n"])
            tok = int(d["tokens"])
            X = int(d["X"])
            w = float(d["waited_s"])
            route = d["route"]
            ep = d.get("epoch")
        except (KeyError, ValueError):
            self.parse_err["release"] += 1
            return
        self.epoch = ep
        self.X = X
        if route == "D" and tok > X:
            self.emit("V1", "ALARM", t, "SUMME-ROUTE epoch=%s release n=%d tokens=%d > X=%d route=D waited_s=%.1f -- Summe ueber X, trotzdem D"
                      % (ep, n, tok, X, w))
        if route == "P":
            if w > V5_MAX:
                self.emit("V5", "WARN", t, "FENSTER-UEBERZUG epoch=%s release n=%d tokens=%d X=%d route=P waited_s=%.1f > %.1f (Fenster 15 s)"
                          % (ep, n, tok, X, w, V5_MAX))
            if self.rel and not self.rel["done"]:
                self.finish_rel()
            self.rel = {"t": t, "epoch": ep, "n": n, "tokens": tok, "X": X, "members": {}, "usum": 0, "done": False}
            if self.v4 is None:
                self.v4 = {"t": t, "epoch": ep, "hold": False, "warn": False, "counted": False}

    def on_short_kept(self, t, l):
        r = self.rel
        if not r or r["done"]:
            return
        d = kv(l)
        rid = d.get("rid")
        try:
            u = int(d["uncached"])
        except (KeyError, ValueError):
            self.parse_err["short-kept"] += 1
            return
        if (rid and d.get("epoch") == r["epoch"] and t - r["t"] <= V2_MEMBER_S and len(r["members"]) < r["n"]
                and rid not in r["members"] and r["usum"] + u <= r["tokens"]):
            r["members"][rid] = u
            r["usum"] += u

    def finish_rel(self):
        r = self.rel
        r["done"] = True
        bad = {}
        for rid, u in r["members"].items():
            ok_t = self.p_ok.get(rid)
            if ok_t is not None and ok_t >= r["t"]:
                self.cnt["V2-OK-handback"] += 1
            else:
                bad[rid] = u
        for k in [k for k, v in self.p_ok.items() if v < self.now - 120]:
            del self.p_ok[k]
        if bad:
            rids = " ".join("%s:%d" % (k, bad[k]) for k in sorted(bad, key=rid_num))
            self.emit("V2", "ALARM", r["t"], "P-ENTSCHEID-GEKAPERT epoch=%s Ausloeser release n=%d tokens=%d X=%d route=P; "
                      "fuer D behalten statt P: %s uncached_sum=%d (von tokens=%d)" % (r["epoch"], r["n"], r["tokens"], r["X"], rids, sum(bad.values()), r["tokens"]),
                      dkey=r["epoch"])

    def on_flip_begin(self, t, l):
        v4 = self.v4
        if not v4:
            return
        lat = t - v4["t"]
        if lat > V4_MAX:
            self.emit("V4", "WARN", t, "FLIP-LATENZ epoch=%s release route=P (%s) -> WEG2-FLIP begin nach %.1f s > %.0f s%s"
                      % (v4["epoch"], hms(v4["t"]), lat, V4_MAX, " (Sitz-Sperre)" if v4["hold"] else ""),
                      dkey=(v4["epoch"], "ende"))
        self.v4 = None

    def on_timeline(self, t, l):
        m = RX_TL.search(l)
        if not m:
            self.parse_err["flip-timeline"] += 1
            return
        if m[1] in self.seen_tl:
            return
        self.seen_tl.add(m[1])
        stages = {k: int(v) for k, v in RX_STAGE.findall(m[4])}
        done = stages.get("done")
        if done is None:
            self.parse_err["flip-timeline"] += 1
            return
        if done > V7_MAX_MS:
            order = sorted(stages.items(), key=lambda kv_: kv_[1])
            gap, name = 0, "-"
            for (a, ta), (b, tb) in zip(order, order[1:]):
                if tb - ta > gap:
                    gap, name = tb - ta, "%s>%s" % (a, b)
            self.emit("V7", "WARN", t, "FLIP-ZEIT epoch=%s %s>%s done=%d ms > %d ms, groesste Luecke %s %d ms"
                      % (m[1], m[2], m[3], done, V7_MAX_MS, name, gap), dkey=m[1])

    # ----------------------------------------------------------------- D/P
    def dp_line(self, src, l):
        t = line_t(l)
        if t is None:
            t = self.now
        else:
            self.last_stamp[src] = l[1:20]
            self.advance(t)
        if "WEG2 X-GATE rid=" in l:
            if src == "D" and " TP0]" in l[:40]:
                self.on_x_gate(t, l)
            return
        if RX_DP_ERR.search(l) and not any(p in l for p in DP_PROSE):
            self.on_err(src, t, l)

    # ------------------------------------------------------------------ V3 (rid-genau)
    def v3_note_decision(self, t, rid, quelle, x, priced):
        if not rid:
            return
        self.xdec.setdefault(rid, []).append((t, quelle, x))
        if priced is not None and str(priced).isdigit():
            self.v3_note_price(t, rid, int(priced))
        self.v3_release(rid)

    def v3_note_price(self, t, rid, u):
        self.xpriced.setdefault(rid, []).append((t, u))

    def v3_release(self, rid):
        """Eine Front-Entscheidung ist da: wartende D-Zeilen dieser rid jetzt urteilen."""
        if not self.v3_pend:
            return
        keep = collections.deque()
        for t0, rec in self.v3_pend:
            if rec["rid"] == rid:
                self.v3_judge(t0, rec, final=False)
                if rec.get("wait"):
                    keep.append((t0, rec))
            else:
                keep.append((t0, rec))
        self.v3_pend = keep

    def v3_prune(self, now):
        for k in [k for k, v in self.xdec.items() if v and v[-1][0] < now - V3_KEEP_S]:
            self.xdec.pop(k, None)
            self.xpriced.pop(k, None)
        for k in [k for k, v in self.v3_seen.items() if v < now - V3_KEEP_S]:
            self.v3_seen.pop(k, None)

    def on_x_gate(self, t, l):
        m = RX_GATE.search(l)
        if not m:
            self.parse_err["x-gate"] += 1
            return
        rid, unc, xs, verdict = m[1], int(m[2]), int(m[3]), m[4]
        self.cnt["V3-gate-zeilen"] += 1
        if verdict != "admit":
            self.cnt["V3-W31-benannt"] += 1     # D verweigert benannt (W31/W50): nie ein Alarm
            return
        if rid in self.v3_seen:
            self.cnt["V3-spaetere-admit"] += 1  # D-Park/Resume: gilt das statische X, kein Entscheidungs-X
            return
        self.v3_seen[rid] = t
        d = kv(l)
        fx = d.get("front_x")
        fp = d.get("front_priced")
        rec = {"rid": rid, "unc": unc, "static": xs, "t": t,
               "fx": int(fx) if fx and fx.isdigit() else None,
               "fp": int(fp) if fp and fp.isdigit() else None, "wait": False}
        self.v3_judge(t, rec, final=False)
        if rec.get("wait"):
            self.v3_pend.append((t, rec))

    def v3_judge(self, t, rec, final):
        rid, unc = rec["rid"], rec["unc"]
        if rec["fx"] is not None:
            x, quelle = rec["fx"], "X-GATE"
        else:
            cand = [e for e in self.xdec.get(rid, ()) if e[0] <= t + 2.0]
            if not cand:
                if final:
                    self.cnt["V3-ohne-X"] += 1
                else:
                    rec["wait"] = True
                return
            _, quelle, x = cand[-1]
            quelle = "Front " + quelle
        rec["wait"] = False
        self.cnt["V3-geprueft"] += 1
        if unc > x + V3_TOL:
            self.emit("V3", "ALARM", t, "D-PREFILL-UEBER-X rid=%s uncached=%d > X=%d (+%d Seite) Quelle=%s verdict=admit statisches X=%d -- "
                      "D hat ueber dem Entscheidungs-X gerechnet, ohne zu verweigern" % (rid, unc, x, V3_TOL, quelle, rec["static"]),
                      dkey=("x", rid))
        priced = rec["fp"]
        if priced is None:
            pr = [e for e in self.xpriced.get(rid, ()) if e[0] <= t + 2.0]
            priced = pr[-1][1] if pr else None
        if priced is not None and unc - priced > V3_MISPRICE:
            self.emit("V3", "WARN", t, "D-PREFILL-FEHLPREIS rid=%s uncached=%d gepreist=%d (+%d > %d) X=%d Quelle=%s -- unter X, kein Gesetzesbruch; "
                      "Front-Kredit tiefer als D's Resume-Anker" % (rid, unc, priced, unc - priced, V3_MISPRICE, x, quelle),
                      dkey=("fp", rid))

    # ------------------------------------------------------------------ V6
    def on_err(self, src, t, l):
        msg = l.rstrip("\n")
        if line_t(msg) is not None:
            i = msg.find("] ")
            msg = msg[i + 2:] if i >= 0 else msg
        key = re.sub(r"\d+", "#", msg[:80])
        v = self.v6.get(key)
        if v is None:
            self.v6[key] = {"n": 1, "n_emit": 1, "t_emit": t, "src": src, "msg": msg}
            self.emit("V6", "ALARM", t, "FEHLER %s %s" % (src, msg[:230]), dkey=False)
        else:
            v["n"] += 1
            self.cnt["V6-Wiederholung"] += 1

    # ---------------------------------------------------------------- Ende
    def finish(self, t_end=None):
        if t_end is not None:
            self.advance(t_end)
        if self.rel and not self.rel["done"]:
            self.finish_rel()
        while self.v3_pend:
            t0, rec = self.v3_pend.popleft()
            self.v3_judge(t0, rec, final=True)
        for v in self.v6.values():
            if v["n"] > v["n_emit"]:
                self.emit("V6", "ALARM", self.now, "FEHLER %s (x%d) %s" % (v["src"], v["n"], v["msg"][:200]), dkey=False, count=False)
                v["n_emit"] = v["n"]

    def summary(self):
        keys = list(RULES) + ["V2-OK-queued", "V2-OK-handback", "V6-Wiederholung", "V3-gate-zeilen", "V3-geprueft",
                              "V3-W31-benannt", "V3-spaetere-admit", "V3-ohne-X"]
        return {k: {"erkannt": self.cnt.get(k, 0), "ausgegeben": self.em.get(k, 0), "letzter": self.last_event.get(k)}
                for k in keys}


# ===================================================================== Dateien
def newest_boot(evdir):
    best, bt = None, -1
    for p in glob.glob(os.path.join(evdir, "*dauer*.front.log")):
        try:
            mt = os.stat(p).st_mtime
        except OSError:
            continue
        if mt > bt:
            best, bt = p, mt
    return best


def sibling(front, src):
    return front[:-len(".front.log")] + "." + src + ".log"


def dp_hits(text):
    """Interessante Zeilen eines Textblocks (nur ganze Zeilen) per C-schnellem find()."""
    starts = {}
    for key in DP_KEYS:
        i = text.find(key)
        while i >= 0:
            s = text.rfind("\n", 0, i) + 1
            e = text.find("\n", i)
            if e < 0:
                e = len(text)
            if s not in starts:
                starts[s] = text[s:e]
            i = text.find(key, e)
    return [starts[k] for k in sorted(starts)]


class Tailer:
    def __init__(self, path, at_end=True):
        self.path = path
        self.pos = None
        self.at_end = at_end
        self.carry = b""

    def read(self, maxbytes=32 << 20):
        try:
            size = os.stat(self.path).st_size
        except OSError:
            return b""
        if self.pos is None:
            self.pos = size if self.at_end else 0
        if size < self.pos:
            self.pos, self.carry = 0, b""
        if size == self.pos:
            return b""
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read(min(maxbytes, size - self.pos))
        self.pos += len(data)
        data = self.carry + data
        i = data.rfind(b"\n")
        if i < 0:
            self.carry = data
            return b""
        self.carry = data[i + 1:]
        return data[:i + 1]


def find_offset(path, target):
    """Byte-Offset der ersten Zeile mit Stempel >= target ('YYYY-MM-DD HH:MM:SS'); Binaersuche."""
    size = os.path.getsize(path)
    lo, hi = 0, size
    with open(path, "rb") as f:
        def stamp_at(off):
            f.seek(off)
            if off:
                f.readline()
            for _ in range(2000):
                pos = f.tell()
                l = f.readline()
                if not l:
                    return None, size
                s = l[1:20].decode("ascii", "replace")
                if l[:1] == b"[" and len(s) == 19 and s[4] == "-" and s[10] == " ":
                    return s, pos
            return None, size
        while hi - lo > 1 << 20:
            mid = (lo + hi) // 2
            s, pos = stamp_at(mid)
            if s is None or s >= target:
                hi = mid
            else:
                lo = mid
    return lo


def replay_front(path, t0, t1, s0):
    off = find_offset(path, s0)
    lastt = t0
    with open(path, "rb") as f:
        f.seek(off)
        for raw in f:
            l = raw.decode("utf-8", "replace")
            t = line_t(l)
            if t is not None:
                if t < t0:
                    continue
                if t > t1:
                    break
            if t is None:
                t = lastt
            lastt = t
            yield (t, 0, "front", l)


def replay_dp(path, src, t0, t1, s0):
    if not os.path.exists(path):
        return
    off = find_offset(path, s0)
    size = os.path.getsize(path)
    carry = b""
    with open(path, "rb") as f:
        f.seek(off)
        while True:
            data = f.read(8 << 20)
            if not data:
                break
            data = carry + data
            i = data.rfind(b"\n")
            if i < 0:
                carry = data
                continue
            carry = data[i + 1:]
            text = data[:i + 1].decode("utf-8", "replace")
            stop = False
            for l in dp_hits(text):
                t = line_t(l)
                if t is None:
                    t = t0
                elif t < t0:
                    continue
                elif t > t1:
                    stop = True
                    break
                yield (t, 1 if src == "D" else 2, src, l)
            if stop:
                return
            if f.tell() >= size and not carry:
                break


def run_replay(args):
    front = args.replay
    s0 = args.frm
    s1 = args.to or "2099-01-01 00:00:00"
    t0, t1 = parse_ts(s0), parse_ts(s1)
    w = Wache(lambda s: print(s, flush=True), rules=args.rules.split(","))
    streams = [replay_front(front, t0, t1, s0)]
    if args.rules_dp:
        streams.append(replay_dp(sibling(front, "D"), "D", t0, t1, s0))
        streams.append(replay_dp(sibling(front, "P"), "P", t0, t1, s0))
    for t, _, src, l in heapq.merge(*streams, key=lambda e: (e[0], e[1])):
        if src == "front":
            w.front_line(l)
        else:
            w.dp_line(src, l)
    w.finish(w.now)
    print("ZAEHLUNG %s" % json.dumps(w.summary(), ensure_ascii=False))
    ru = resource.getrusage(resource.RUSAGE_SELF)
    print("CPU user=%.2fs sys=%.2fs maxrss=%dMB parse_fehler=%s" % (ru.ru_utime, ru.ru_stime, ru.ru_maxrss // 1024, dict(w.parse_err) or "-"))


# ====================================================================== Live
def tail_stamp(path):
    """Stempel der letzten Logzeile (liest nur die letzten 64 KB)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            for raw in reversed(f.read().decode("utf-8", "replace").splitlines()):
                if line_t(raw) is not None:
                    return raw[1:20]
    except OSError:
        pass
    return None


def write_status(path, w, boot, started):
    d = {"zeit_utc": fmt_t(time.time()), "gestartet_utc": fmt_t(started), "pid": os.getpid(), "boot": boot,
         "letzter_logstempel": dict(w.last_stamp), "letztes_event": w.last_emit_t,
         "regeln": w.summary(), "parse_fehler": dict(w.parse_err), "jetzt_log": fmt_t(w.now) if w.now else None}
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def run_live(args):
    try:
        os.nice(19)
    except OSError:
        pass
    started = time.time()
    w = Wache(lambda s: print(s, flush=True))
    front = newest_boot(args.dir)
    boot = os.path.basename(front)[:-len(".front.log")] if front else None
    tails = {}

    def open_tails(fr, first):
        t = {"front": Tailer(fr, at_end=not (args.from_start and first))}
        t["D"] = Tailer(sibling(fr, "D"), at_end=True)
        t["P"] = Tailer(sibling(fr, "P"), at_end=True)
        return t

    if front:
        tails = open_tails(front, True)
        for src, tl in (("front", tails["front"]), ("D", tails["D"]), ("P", tails["P"])):
            w.last_stamp[src] = tail_stamp(tl.path)
    last_check = last_status = 0.0
    while True:
        now = time.time()
        try:
            if now - last_check >= 30:
                last_check = now
                nf = newest_boot(args.dir)
                if nf and nf != front:
                    front = nf
                    boot = os.path.basename(front)[:-len(".front.log")]
                    w.reset()
                    big = os.path.getsize(front) > (8 << 20)
                    tails = {"front": Tailer(front, at_end=big), "D": Tailer(sibling(front, "D"), at_end=big),
                             "P": Tailer(sibling(front, "P"), at_end=big)}
                    w.out(fmt_t(now) + " WARN BOOT-WECHSEL " + boot)
            if tails:
                data = tails["front"].read()
                if data:
                    for l in data.decode("utf-8", "replace").splitlines():
                        w.front_line(l)
                for src in ("D", "P"):
                    data = tails[src].read()
                    if data:
                        for l in dp_hits(data.decode("utf-8", "replace")):
                            w.dp_line(src, l)
                w.advance(w.now)
            if now - last_status >= 60 or last_status == 0.0 and now - started >= 2:
                last_status = now
                write_status(args.status, w, boot, started)
        except Exception as e:  # die Wache darf nie sterben
            print("funktionswache interner Fehler: %r" % (e,), file=sys.stderr, flush=True)
        time.sleep(1.0)


# =================================================================== Selbsttest
def selftest():
    out = []
    w = Wache(out.append)
    base = calendar.timegm((2026, 10, 12, 12, 0, 0))

    def L(dt, body, lvl="INFO"):
        t = base + dt
        return "[%s,%03d] %s weg2.front: %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)), int((t % 1) * 1000), lvl, body)

    def feed(dt, body, lvl="INFO"):
        w.front_line(L(dt, body, lvl))

    def D(dt, n):
        t = base + dt
        w.dp_line("D", "[%s TP0] Prefill batch, #new-seq: 1, #new-token: %d, #cached-token: 0, full token usage: 0.1"
                  % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)), n))

    fails = []

    def check(name, cond):
        if not cond:
            fails.append(name)
        print("%-34s %s" % (name, "ok" if cond else "FEHLSCHLAG"))

    # V1
    feed(0, "WEG2 DECODE-COLLECT release n=2 tokens=7000 X=8000 waited_s=7.5 route=D stage=dcheck p_bound=False epoch=1")
    check("V1 kein Alarm bei T<=X", not out)
    feed(1, "WEG2 DECODE-COLLECT release n=2 tokens=9000 X=8000 waited_s=7.5 route=D stage=dcheck p_bound=False epoch=1")
    check("V1 Alarm bei T>X", len(out) == 1 and " ALARM V1 " in out[0])
    feed(2, "WEG2 DECODE-COLLECT release n=2 tokens=9001 X=8000 waited_s=7.5 route=D stage=dcheck p_bound=False epoch=1")
    check("Dedupe gleiche Regel+Epoche 30 s", len(out) == 1 and w.cnt["V1"] == 2)
    # V5
    out.clear()
    feed(100, "WEG2 DECODE-COLLECT release n=1 tokens=10 X=8000 waited_s=15.0 route=P stage=window p_bound=False epoch=2")
    check("V5 15.0 s kein Treffer", not any(" V5 " in o for o in out))
    feed(101, "WEG2-FLIP begin epoch=2 sleep=D wake=P outstanding=0 queue=1")
    feed(200, "WEG2 DECODE-COLLECT release n=1 tokens=10 X=8000 waited_s=17.2 route=P stage=window p_bound=False epoch=3")
    check("V5 17.2 s WARN", any(" WARN V5 " in o for o in out))
    feed(201, "WEG2-FLIP begin epoch=3 sleep=D wake=P outstanding=0 queue=1")
    # V2 Alarm
    out.clear()
    feed(300, "WEG2 DECODE-COLLECT release n=2 tokens=300 X=200 waited_s=15.0 route=P stage=window p_bound=False epoch=5")
    feed(300.01, "WEG2 SHORT-KEPT rid=weg2-5-11 uncached=100 presence=1 epoch=5: the SHORT fell through")
    feed(300.02, "WEG2 SHORT-KEPT rid=weg2-5-10 uncached=200 presence=1 epoch=5: the SHORT fell through")
    feed(301, "WEG2-FLIP begin epoch=5 sleep=D wake=P outstanding=0 queue=1")
    feed(310, "WEG2 irgendwas")
    v2 = [o for o in out if " V2 " in o]
    check("V2 Alarm, rids sortiert + Summe", len(v2) == 1 and "ALARM" in v2[0] and "weg2-5-10:200 weg2-5-11:100" in v2[0] and "uncached_sum=300" in v2[0])
    # V2 OK per Handback
    out.clear()
    feed(400, "WEG2 DECODE-COLLECT release n=2 tokens=300 X=200 waited_s=15.0 route=P stage=window p_bound=False epoch=6")
    feed(400.01, "WEG2 SHORT-KEPT rid=weg2-6-1 uncached=100 presence=1 epoch=6: x")
    feed(401, "WEG2 PARK-HANDBACK rid=weg2-6-1 epoch=6 (D held)")
    feed(401.1, "WEG2-FLIP begin epoch=6 sleep=D wake=P outstanding=0 queue=1")
    feed(410, "WEG2 irgendwas")
    check("V2 Handback = OK", not any(" V2 " in o for o in out) and w.cnt["V2-OK-handback"] == 1)
    # V2 Gegenprobe COLLECT-P-QUEUED, spaete SHORT-KEPT zaehlen nicht
    out.clear()
    feed(500, "WEG2 DECODE-COLLECT release n=1 tokens=100 X=50 waited_s=15.0 route=P stage=window p_bound=False epoch=7")
    feed(500.01, "WEG2 COLLECT-P-QUEUED rid=weg2-7-1 uncached=100 epoch=7")
    feed(502, "WEG2 SHORT-KEPT rid=weg2-7-9 uncached=10 presence=1 epoch=7: neue Ankunft")
    feed(501, "WEG2-FLIP begin epoch=7 sleep=D wake=P outstanding=0 queue=1")
    feed(510, "WEG2 irgendwas")
    check("V2 COLLECT-P-QUEUED/Spaeteingang ok", not any(" V2 " in o for o in out) and w.cnt["V2-OK-queued"] == 1)
    # V4
    out.clear()
    feed(600, "WEG2 DECODE-COLLECT release n=1 tokens=100 X=50 waited_s=15.0 route=P stage=window p_bound=False epoch=8")
    feed(602, "WEG2 DECODE-COLLECT seat-gate hold taken=6 n=6 epoch=8 -- route P")
    feed(625, "WEG2 irgendwas")
    check("V4 Lauf-Warnung + Sitz-Sperre", any(" WARN V4 " in o and "(Sitz-Sperre)" in o for o in out))
    feed(640, "WEG2-FLIP begin epoch=8 sleep=D wake=P outstanding=0 queue=1")
    check("V4 Schlusszeile mit Latenz", any("nach 40.0 s" in o for o in out))
    # V3 (rid-genau, seit 15.10.): D-Zeile vs. das Entscheidungs-X DIESER rid
    out.clear()

    def G(dt, rid, unc, verdict="admit", extra="", static=12288):
        t = base + dt
        w.dp_line("D", "[%s TP0] WEG2 X-GATE rid=%s uncached=%d X=%d replicated_term=group verdict=%s%s"
                  % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)), rid, unc, static, verdict, extra))

    # (a) der alte Fehlalarm: viele D-Prefills im selben Fenster, jeder fuer sich <= X -> KEIN Alarm
    feed(700, "WEG2 X-ROUTE rid=weg2-9-1 est_uncached=10 X=100 (EXACT)")
    for k in range(8):
        feed(700.1 + k, "WEG2 X-ROUTE rid=weg2-9-1%d est_uncached=90 X=100 (EXACT)" % k)
        G(700.5 + k, "weg2-9-1%d" % k, 90)
    check("V3 Summe ueber X, je rid <= X: kein Alarm", not any(" V3 " in o for o in out) and w.cnt["V3-geprueft"] >= 8)
    # (b) Gesetzesbruch: admit ueber dem Entscheidungs-X (+ eine Seite)
    feed(720, "WEG2 X-ROUTE rid=weg2-9-30 est_uncached=100 X=1010 (EXACT)")
    G(721, "weg2-9-30", 1010 + 64)  # eine Seite, woertlich: die Konstante darf nicht mitwandern
    check("V3 genau X+Seite: kein Alarm", not any(" ALARM V3 " in o for o in out))
    feed(722, "WEG2 X-ROUTE rid=weg2-9-31 est_uncached=100 X=1010 (EXACT)")
    G(723, "weg2-9-31", 1010 + 64 + 1)
    check("V3 X+Seite+1: ALARM mit rid, X, Quelle", any(" ALARM V3 " in o and "weg2-9-31" in o and "X=1010" in o and "Quelle=Front X-ROUTE" in o for o in out))
    # (c) das spaetere Entscheidungs-X gilt (QUEUED-SHORT 6057 nach Ankunfts-X 1010) -- der 40-128-Fall
    out.clear()
    feed(740, "WEG2 X-ROUTE rid=weg2-9-40 est_uncached=737 X=1010 (EXACT)")
    feed(741, "WEG2 X-EXACT-REPRICE rid=weg2-9-40 why=x est_uncached 737 -> 5089 X=6733 crossed=no src=d_seq_anchor epoch=1 (q)")
    feed(745, "WEG2 ARRIVAL-SEAT rid=weg2-9-40 verdict=d_prefill QUEUED-SHORT uncached=5089 X=6057 waited=5.0 s -- q")
    G(746, "weg2-9-40", 5089)
    check("V3 QUEUED-SHORT-X 6057 gilt, 5089 kein Alarm/Fehlpreis", not any(" V3 " in o for o in out))
    # (d) FEHLPREIS: unter X, aber > 1024 ueber dem Preis (weg2-40-127: 237 -> 1517)
    out.clear()
    feed(760, "WEG2 X-ROUTE rid=weg2-9-50 est_uncached=237 X=1010 (EXACT)")
    feed(765, "WEG2 ARRIVAL-SEAT rid=weg2-9-50 verdict=d_prefill QUEUED-SHORT uncached=237 X=6057 waited=5.0 s -- q")
    G(766, "weg2-9-50", 1517)
    check("V3 Fehlpreis 237->1517 unter X: WARN, kein ALARM",
          any(" WARN V3 " in o and "FEHLPREIS" in o and "weg2-9-50" in o and "gepreist=237" in o for o in out)
          and not any(" ALARM V3 " in o for o in out))
    # (e) verdict=W31 ist benannt: kein Alarm, auch weit ueber X
    out.clear()
    feed(780, "WEG2 X-ROUTE rid=weg2-9-60 est_uncached=90 X=100 (EXACT)")
    G(781, "weg2-9-60", 91533, verdict="W31")
    check("V3 W31 benannt: kein Alarm", not out and w.cnt["V3-W31-benannt"] >= 1)
    # (f) nur die ERSTE admit-Zeile zaehlt (D-Park/Resume: statisches X)
    feed(800, "WEG2 X-ROUTE rid=weg2-9-70 est_uncached=90 X=100 (EXACT)")
    G(801, "weg2-9-70", 90)
    G(802, "weg2-9-70", 5000)
    check("V3 spaetere admit-Zeile ignoriert", not out and w.cnt["V3-spaetere-admit"] >= 1)
    # (g) ab FRONT-X-MIT traegt die X-GATE-Zeile selbst front_x: Quelle X-GATE, sie geht vor dem Front-Log
    out.clear()
    G(820, "weg2-9-80", 2000, extra=" front_x=1010 front_priced=900 limit=1074")
    check("V3 front_x in der Zeile: Quelle X-GATE, ALARM", any(" ALARM V3 " in o and "Quelle=X-GATE" in o for o in out))
    # (h) D-Zeile vor der Front-Entscheidung: wartet, urteilt bei deren Ankunft
    out.clear()
    G(840, "weg2-9-90", 2000)
    check("V3 wartet auf die Front-Entscheidung", not out and len(w.v3_pend) == 1)
    feed(840.5, "WEG2 X-ROUTE rid=weg2-9-90 est_uncached=100 X=1010 (EXACT)")
    check("V3 Reihenfolge-Schutz: ALARM nach Ankunft der Entscheidung", any(" ALARM V3 " in o and "weg2-9-90" in o for o in out))
    # (j) die echten Zahlen von weg2-40-127 (Boot 124245, 13:12:08-13:12:49Z): gepreist 237, Ankunfts-X 1010, D rechnete 1517.
    #     (j1) wie im Boot: QUEUED-SHORT entschied mit X=6057 -> 1517 <= 6057, KEIN Alarm, aber der Fehlpreis (+1280) wird benannt.
    out.clear()
    feed(900, "WEG2 X-ROUTE rid=weg2-40-127 est_uncached=237 X=1010 (EXACT, front tokenizer, see X-EXACT-PRICE)")
    feed(915, "WEG2 ARRIVAL-SEAT rid=weg2-40-127 verdict=d_prefill QUEUED-SHORT uncached=237 X=6057 waited=15.4 s -- q")
    G(937, "weg2-40-127", 1517)
    check("V3 40-127 real: Fehlpreis WARN, kein Alarm", any("FEHLPREIS" in o and "uncached=1517" in o and "gepreist=237" in o for o in out)
          and not any(" ALARM V3 " in o for o in out))
    #     (j2) waere es mit dem Ankunfts-X 1010 direkt nach D gegangen (kein QUEUED-SHORT): 1517 > 1010 + Seite -> ALARM bleibt.
    out.clear()
    feed(950, "WEG2 X-ROUTE rid=weg2-40-227 est_uncached=237 X=1010 (EXACT, front tokenizer, see X-EXACT-PRICE)")
    G(970, "weg2-40-227", 1517)
    check("V3 40-127-Zahlen mit Ankunfts-X: ALARM bleibt", any(" ALARM V3 " in o and "uncached=1517 > X=1010" in o for o in out))
    # (i) ohne jede X-Quelle: nie ein Alarm aus Unwissen
    out.clear()
    G(860, "weg2-9-95", 99999)
    feed(870, "WEG2 irgendwas")
    check("V3 ohne X-Quelle: zaehlt, kein Alarm", not out and w.cnt["V3-ohne-X"] == 1)
    # V6
    out.clear()
    feed(800, "WEG2 leg2 rid=weg2-9-5 failed: Boom", lvl="ERROR")
    feed(801, "WEG2 leg2 rid=weg2-9-6 failed: Boom", lvl="ERROR")
    check("V6 erste Zeile, Wiederholung stumm", len([o for o in out if " V6 " in o]) == 1 and w.cnt["V6-Wiederholung"] == 1)
    feed(1200, "WEG2 irgendwas")
    check("V6 (xN) nach 5 min", any("(x2)" in o for o in out))
    # V7
    out.clear()
    feed(1300, "WEG2-FLIP-TIMELINE epoch=11 slept=D woke=P ms-from-begin: drain@0 quiesce@1 sleep-kv@33 gathered-legs@162 wake-kv@9134 dc@9134 done@9134 (stage@t = x)")
    check("V7 done 9134 ms", any(" WARN V7 " in o and "done=9134" in o for o in out))
    w.finish(w.now)
    print("SELFTEST", "FEHLER: " + ", ".join(fails) if fails else "OK")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--from-start", action="store_true")
    ap.add_argument("--replay")
    ap.add_argument("--from", dest="frm")
    ap.add_argument("--to")
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir", default=EVDIR)
    ap.add_argument("--status", default=STATUS)
    ap.add_argument("--no-nice", action="store_true")
    args = ap.parse_args()
    if not args.no_nice:
        try:
            os.nice(19)
        except OSError:
            pass
    if args.selftest:
        sys.exit(selftest())
    if args.replay:
        if not args.frm:
            ap.error("--replay braucht --from")
        # D/P nur lesen, wenn V3 oder V6 gefragt sind (V1/V2/V4/V5/V7 brauchen nur die Front)
        args.rules_dp = bool(set(args.rules.split(",")) & {"V3", "V6"})
        run_replay(args)
        return
    run_live(args)


if __name__ == "__main__":
    main()
