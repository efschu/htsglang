# ANALYSE #997 — Boot 16: #631 PROXY LEFTOVER, Wurzel

Datum 2026-08-28. Baum `/spinning/wt-943boot` @ `merge/flip-window-0828`.
Specimen: `/spinning/evidence-665-f1/boot_943bx_996fbf4aca_0828_221614.log`
(Boot-Pin `996fbf4aca`). READ-ONLY, kein Metall.
Beleg-Stufe **DESK-BEWIESEN** (Log + Code + AST). Nichts hier ist "gefixt".

Fortsetzung von `ANALYSE_996_reset_klasse_zensus.md`: dies ist eine Instanz der
dort benannten Familie B — ein **Halter, den der Reset nicht kennt**, hier der
Microbatch-Slot-Ring.

---

## 1. Praemissen-Korrektur zum Auftrag

Der Auftrag nannte "ein voller Flip-Roundtrip (3x pp_to_tp epoch 1, 3x tp_to_pp
epoch 2) vollzogen". **Das Log sagt etwas anderes, und der Unterschied ist die
halbe Diagnose:**

```
:2099  [22:21:47 PP0] PHASE-FLIP FLIP ABANDONED (no quorum): pp_to_tp waited
       60.0s for epoch 2 and rank(s) [1, 2] did not announce presence.
       NOTHING was entered and no request was touched.
```

Der Epoche-2-Flip wurde **ABGEBROCHEN, nicht vollzogen**. Es gab keinen Cutover,
kein Ring-Rebuild. Genau deshalb greift der Erste der beiden Guard-Zweige
(gleiche Epoche) und nicht der Cross-Epoch-Zweig — das war korrekt beobachtet,
aber aus dem falschen Grund.

Warum kein Quorum, ebenfalls im Log:

```
:2076  [PP0] armed (pp_to_tp) but NOT QUIESCENT: PP microbatches still in
       flight (mb slots [1, 2])
:2090  [PP1] armed (pp_to_tp) but NOT QUIESCENT: a chunked prefill is incomplete
:2094  [PP2] armed (pp_to_tp) but NOT QUIESCENT: a chunked prefill is incomplete
```

**PP1 und PP2 hielten einen unvollstaendigen chunked prefill, PP0 nicht.** Diese
Asymmetrie ist die Wurzel.

---

## 2. Die Wurzel, mit file:line

### D1 — ROOT: der Slot-Hold ist rein rang-lokal und seine eigene Korrektheits-Voraussetzung faellt damit

`_pp_flip_hold_slot` (`scheduler_pp_mixin.py:5241-5335`) ist der Aktuator, der
verhindern soll, dass Raenge waehrend eines armed window auf verschiedenen Slots
landen. Er endet mit:

```python
5330        if getattr(self, "chunked_req", None) is not None:
5331            return False          # <-- rang-lokale Ausnahme
5332        mbs = getattr(self, "mbs", None)
5333        if not mbs: return False
5335        return all(mb is None for mb in mbs)
```

AST-geprueft: die Funktion liest **ausschliesslich** `chunked_req` und `mbs`,
beide rang-lokal; kein `counters`, kein `all_gather`, kein Barrier. (Das Wort
"peer" kommt nur im Docstring vor.)

Der Docstring benennt seine eigene Korrektheits-Voraussetzung woertlich
(`:5292-5302`):

> "WHY THE HOLD IS REACHED ON THE SAME SLOT ON EVERY RANK, **which is the only
> property that makes this correct**. … From that shared slot every rank needs
> exactly `pp_loop_size` parked iterations to null every slot, so
> `all(mb is None)` first holds at the same slot index on every rank."

**Dieses Argument setzt voraus, dass JEDER Rang den Hold ueberhaupt betritt.**
Ein Rang, den `:5330-5331` ausnimmt, betritt ihn nie — und die als "einzige
Eigenschaft, die das korrekt macht" bezeichnete Slot-Uniformitaet ist damit
aufgehoben, sobald die Raenge uneinig sind, ob sie einen `chunked_req` halten.

`scheduler.chunked_req` ist ein einzelnes rang-lokales Feld. Uneinigkeit ist
nicht der Ausnahmefall, sondern der Normalfall unter Last — Boot 16 ist die
Messung: PP0 ohne Chunk haelt, PP1/PP2 mit Chunk halten nicht.

Der Docstring rechtfertigt die Ausnahme mit "Holding there would stop a rank the
pipeline is still driving" (`:5304-5308`). Das ist richtig **solange der Chunk
gefahren wird** und falsch **ueber einen Abbruch hinweg**: nach dem Abbruch faehrt
niemand mehr, und der Rang steht auf einem anderen Slot als die Gruppe.

### Die Folge, vom Instrument des Systems selbst gemessen

```
:2105  [22:21:48 PP0] PHASE-FLIP PASS-CLOCK across the armed window: rank 0 ran 0
       slot iteration(s) (armed at mb_id=1, disarmed at mb_id=2); group passes
       [0, 1, 1], SPREAD 1; group RESUME SLOTS [1, 2, 2] -- DIVERGED, so every
       later proxy on this instance is mispaired and the slot hold did not do
       its job.
```

`group passes [0, 1, 1]`: PP0 lief 0 Iterationen (Hold griff), PP1/PP2 je 1
(Hold griff nicht). `RESUME SLOTS [1, 2, 2]` ist die publizierte Gauge
(`CHAN_SLOT`, publiziert bei `:5025` mit dem laufenden `mb_id` je armed tick,
gelesen bei `:5204`).

Eine Sekunde spaeter, exakt passend:

```
:2147  RuntimeError: #631 PROXY LEFTOVER REFUSED: a proxy stamped mb_id=1 seq=17
       rows=4096 epoch=2 arrived while this rank is on mb_id=2 in flip epoch 2.
```

PP0 sendet auf Slot 1, PP1 steht auf Slot 2, gleiche Epoche 2. **Der Crasher ist
PP1** (`:2129`). Die Zahlen des Refusals sind eine 1:1-Ableitung aus
`RESUME SLOTS [1, 2, 2]`.

### D2 — Der Backstop ist ebenfalls rang-lokal konditioniert

Der SLOT-RESTORE, gedacht als Auffangnetz fuer genau diesen Fall
(`scheduler_pp_mixin.py:5121`):

```python
would_restore = passes == 0 and arm_mb is not None and int(arm_mb) != int(mb_id)
```

`passes` ist **dieses Rangs eigene** Slot-Iterationszahl. Er feuert also genau
auf den Raengen, die gehalten haben (PP0, passes=0), und nie auf denen, die
gedriftet sind (PP1/PP2, passes=1). Ein Auffangnetz fuer eine Gruppen-Invariante,
dessen Ausloeser rang-lokal ist, kann die Invariante nicht herstellen.

**Ehrliche Einschraenkung, weil ich es zuerst schaerfer formuliert hatte:** ich
behaupte NICHT, der Restore habe die Divergenz erzeugt. Die publizierten Gauges
standen bereits auf [1,2,2], bevor der Restore lief — die Divergenz entsteht in
D1. Der Restore haelt PP0 auf dem Minderheits-Slot, konsistent mit dem, was PP0
publiziert hatte. Ob "ohne Restore waeren alle drei auf 2 gewesen" gilt, laesst
sich aus dem Log allein nicht entscheiden, weil PP0s Gauge dann seiner realen
Position widersprochen haette. Das ist offen.

### D3 — Das Urteil existiert und niemand handelt darauf

`:5215` berechnet `agreed = len(set(slots)) <= 1` und `:5216-5239` **loggt** es,
auf ERROR wenn diverged. Es gibt keinen Aktuator. Die Gruppe weiss, dass sie
divergiert ist, und faehrt weiter — bis der Proxy-Guard eine Sekunde spaeter
abbricht. Dieselbe #719-Form wie der tote Ratchet aus #996: Instrument belegt,
Konsument fehlt.

---

## 3. Der Hinweis des Guards ist FALSCH

Der Refusal-Text schliesst mit:

> "The drains (pp_flip_drain_tensor_dicts while armed, pp_flip_drain_leftover_dicts
> at disarm) are what is supposed to prevent this from ever being reached -- if
> you are reading this, they did not, and THAT is the defect to chase."

**Fuer diese Instanz ist das eine Fehlleitung, aus Ordnungsgruenden.** Die
Nachricht, die PP1 toetet, ist keine Leiche: sie ist eine korrekt geformte,
aktuell-epochige Nachricht von einem Rang, der tatsaechlich auf Slot 1 steht, und
sie **entsteht erst nach dem Disarm**. Kein Drain kann eine Nachricht entfernen,
die zum Drain-Zeitpunkt noch nicht existiert. Der Drain-Pfad ist hier gesund; die
Slot-Zuordnung ist krank.

Vierter nachweislich falsche Kommentar dieser Kampagne. Wer dem Guard folgt,
verliert ein Fenster an den Drains.

### Und strukturell koennen beide Drains diese Nachricht ohnehin nicht sehen

Unabhaengig nachgetracet, von mir an den Verzweigungen gegengeprueft:

| Drain | Wann aktiv | file:line |
|---|---|---|
| `pp_flip_drain_tensor_dicts` | **nur waehrend dieser Rang ARMED ist** — einziger Aufruf `:5832` in `pp_flip_service`, dessen Aufrufer `:4730` unter `if self.pp_phase_flip_armed():` (`:4703`) steht | `:5483-5598` |
| `pp_flip_drain_leftover_dicts` | **genau EINMAL**, an der fallenden Flanke des eigenen Disarms (`:5196`); danach nie wieder, weil `_pp_flip_armed_passes = None` (`:5036`) jeden spaeteren Tick bei `:5034-5035` vorher zurueckkehren laesst. Plus ein Settle-Fenster von hoechstens 0,75 s | `:5600-5756` |

Der toedliche Proxy entsteht **nach** PP1s Disarm — PP0 bricht auf seiner eigenen
Uhr ab und nimmt danach den Betrieb wieder auf. Zu diesem Zeitpunkt ist PP1 nicht
mehr armed (erster Drain laeuft nicht mehr) und hat seinen Einmal-Sweep laengst
verbraucht (zweiter auch nicht). **Beide Drains sind im Moment des Einschlags
konstruktiv abgeschaltet.** Der Guard beschuldigt zwei Mechanismen, die per
Kontrollfluss gar nicht zustaendig sein koennen.

**Fuenfter falscher Kommentar, im selben Umfeld:** der Docstring von
`pp_flip_drain_tensor_dicts` (`:5521-5530`) sagt woertlich "DISABLED -- CORPSE S
… It is left here, **uncalled**, so the next reader inherits the measurement".
Die Funktion wird bei `:5832` aufgerufen, und der Kommentar am Aufrufer sagt
"#757: RE-ENABLED". Wer dem Docstring glaubt, haelt einen aktiven Drain fuer
toten Code.

---

## 4. Die vom Auftrag gestellten Fragen, beantwortet

**Wird der Ring am Cutover rang-vereinbart neu aufgesetzt oder laeuft er
rang-lokal weiter?** Bei einem **vollzogenen** Cutover wird er neu gebaut
(`init_pp_loop_state`, `scheduler_pp_mixin.py:6233-6401`, gerufen aus `_cutover`
`phase_flip_runtime.py:3231`), und `pp_flip_forget_ring_scoped_slots`
(`:1315-1317`) verwirft die ring-skalierten Slots. Bei einem **Abbruch** — Boot
16 — passiert nichts davon: `_abandon_no_quorum` (`phase_flip_runtime.py:3885`)
und `_abandon_unjoined_flip` (`:3949`) sind laut dem #757-Kommentar
(`scheduler_pp_mixin.py:5183-5194`) "purely rank-local … no collective and no
channel re-check". Der Ring laeuft rang-lokal weiter, mit dem Slot, auf dem jeder
Rang zufaellig stand. **Es gibt keinen Punkt, an dem die Gruppe sich auf einen
Resume-Slot einigt.**

**Crash oder stille Fehlpaarung ohne den Guard?** **Stille Fehlpaarung**, und
zwar die schlimmere Sorte. Der Guard-Text sagt es und die Struktur bestaetigt es:
die Hidden States eines Microbatches wuerden mit den Metadaten eines anderen
gepaart. Der Breitencheck in `model_runner.forward` sieht das nicht, weil
chunked prefill jeden Chunk auf dieselbe Groesse deckelt (`rows=4096` im
Specimen) — gleiche Breite, falscher Inhalt. Der Guard bei `:8909` ist das
Einzige, was daraus einen lauten Fehler macht. **Er gehoert nicht entschaerft.**

---

## 5. Fix-Shape

Nach Aufwand/Ertrag, kleinste zuerst. Ich habe kein Metall; das sind Vorschlaege,
keine Verifikate.

**F1 (kleinste, deckt D2/D3, keine neue Kollektive):** Die Restore-Entscheidung
gegen die **bereits publizierten** Gruppen-Gauges treffen statt gegen das lokale
`passes`. `slots = [counters.sent(CHAN_SLOT, r) …]` wird in derselben Funktion
ohnehin gelesen (`:5204`) — nur eben **nach** der Restore-Entscheidung
(`:5151-5181`). Den Read vorziehen und nur restaurieren, wenn das die
Slot-Einigkeit nicht bricht. Nutzt einen vorhandenen Mechanismus, fuegt keinen
Synchronisationspunkt hinzu und respektiert damit das ausgeschriebene Designgesetz
"NO LAUNCH TIMING MOVES" (`:5317-5321`). Macht D3 vom Log zum Aktuator.

**F2 (Wurzel, groesser):** Die `chunked_req`-Ausnahme in `_pp_flip_hold_slot`
(`:5330-5331`) darf die Gruppen-Uniformitaet nicht brechen. Zwei Wege:
(a) der Hold greift auch mit `chunked_req`, sobald das armed window die Pipeline
trockengelaufen hat — dann ist die Ausnahme auf "der Chunk wird noch gefahren"
verengt statt auf "ein Chunk existiert"; oder
(b) ein armed Rang mit unvollstaendigem Chunk **voidet** ihn, statt ausgenommen zu
werden — was die Quieszenz-Regel ohnehin will ("strict: the flip would discard
it, #856 removed the carry", Log `:2090`) und was zur Nutzer-Doktrin
Cutover = Re-Entry passt.
(b) ist die ehrlichere, beruehrt aber den Kein-Doppel-Prefill-Pfad und gehoert
damit nicht in einen Boot-Fix-Zyklus ohne eigene Abwaegung.

**Nicht tun:** den Guard bei `:8909` aufweichen, und den Drains nachgehen.

---

## 6. Ko-Timing FILL-ADOPT — geprueft, nicht angenommen

Zwei `#987 FILL-ADOPT` in derselben Sekunde 22:21:48, PP1 (`:2120`) und PP2
(`:2126`), gleicher rid `da614e20…`, `local=8446 -> upstream=8447 appended=1
tail=[271]`, beide mit "seam #631".

**Verdikt: mechanistisch UNABHAENGIG, aber gemeinsame Wurzel.**

- *Unabhaengig:* FILL-ADOPT bewegt die **Fill-Laenge** (Token-Ids); der Refusal
  entsteht aus dem **Slot-Index**. Die Slot-Divergenz stand bereits in den
  publizierten Gauges (`RESUME SLOTS [1,2,2]`, `:2105`) und ist eine reine
  Funktion der Iterationszahlen waehrend des armed window — von Fill-Laengen
  haengt sie nirgends ab. Keine Kausalitaet in beide Richtungen.
- *Gemeinsame Wurzel:* `da614e20…` ist genau der chunked request
  (`:2127` `#788 PP-ADMISSION … rids=da614e20… chunked=1`), dessen
  Unvollstaendigkeit (i) PP1/PP2 per `:5330-5331` vom Hold ausnahm — die
  Slot-Divergenz — und (ii) den Carry ausloeste, den FILL-ADOPT materialisiert.
  Ein Request, zwei Symptome, ein Grund.

Die Gleichzeitigkeit ist daher echt und nicht zufaellig, aber sie ist
Ko-Symptomatik, keine Kette. Wer FILL-ADOPT abschaltet, behebt den Crash nicht.

---

## 7. Was ich NICHT pruefen konnte

- Ob ohne den Restore alle drei Raenge auf Slot 2 zusammengefallen waeren (§2 D2).
- Ob F1 unter Last haelt — kein Metall, keine Messung.
- (ERLEDIGT, siehe §3) Die Drain-Koerper sind jetzt getracet.
  gesund fuer DIESE Instanz aus Ordnungsgruenden abgeleitet (die Nachricht
  entsteht nach dem Disarm), nicht durch vollstaendiges Tracing ihrer Koerper.
- Die Frage, ob `_abandon_unjoined_flip` (`phase_flip_runtime.py:3949`) dieselbe
  Slot-Asymmetrie erzeugt wie `_abandon_no_quorum` — plausibel, ungeprueft.

---

## 8. NACHTRAG — konkurrierende Ursache (Arithmetik-Achse), per ORDNUNG entschieden

Der Arithmetik-Sweep schlaegt fuer dasselbe Specimen eine andere Wurzel vor: eine
falsch abgeleitete `last_chunk`-Entscheidung habe eine Continuation gemintet, die
die Decision nie nannte, und DEREN Proxy sei der Leftover. Cut C dagegen ist
committet (`26ccc1065f`).

**Beide Mechanismen sind in diesem Specimen vorhanden. Die Slot-Divergenz ist die
FRUEHERE und steht unabhaengig, und das entscheidet die Zeitachse:**

| Zeit | Ereignis | Beleg |
|---|---|---|
| 22:20:47 | armed window beginnt, alle drei Raenge | `:2061`, `:2075`, `:2088` |
| 22:20:47–22:21:47 | **das armed window**: PP0 haelt (kein Chunk, passes=0), PP1/PP2 halten nicht (Chunk, passes=1 je). Die Slot-Gauges divergieren HIER, publiziert je armed tick bei `:5025` | `:2090`, `:2094` |
| 22:21:47 | ABANDON nach 60 s | `:2099` |
| 22:21:48 | PASS-CLOCK meldet `RESUME SLOTS [1, 2, 2] -- DIVERGED` | `:2105` |
| 22:21:48 | FILL-ADOPT auf PP1/PP2; der behauptete Fehl-Mint kann fruehestens JETZT stattfinden | `:2120`, `:2126` |
| 22:21:48 | Refusal auf PP1 | `:2147` |

Die Divergenz entsteht waehrend des 60-Sekunden-Fensters und ist gemessen, bevor
irgendein Post-Abandon-Mint existieren kann. **Ein Ereignis um 22:21:48 kann eine
Zustandsdivergenz nicht verursacht haben, die zwischen 22:20:47 und 22:21:47
entstanden und um 22:21:48 bereits ausgedruckt ist.**

Hinzu kommt: der beobachtete Stempel `mb_id=1` ist genau PP0s publizierter
Resume-Slot. PP0 nimmt nach dem Abbruch den normalen Betrieb auf Slot 1 wieder
auf und sendet an PP1, das auf Slot 2 steht — dafuer braucht es **gar keinen
Extra-Mint**. Die einfachste Erklaerung des konkreten Stempels ist der
Normalbetrieb ueber einen divergenten Ring.

**Was daraus NICHT folgt:** dass der Fehl-Mint kein echter Defekt ist. Er kann
ein zweiter, unabhaengiger Beitrag sein, und Cut C ist unabhaengig davon
gerechtfertigt. Die vom Sweep vorgeschlagene Probe ist gut und ich nehme sie an:
verschwindet der Leftover nach Cut C, war der Mint hinreichend; bleibt er, ist
die Ring-Divergenz die Wurzel. Meine Vorhersage, damit sie falsifizierbar
protokolliert ist: **der Leftover bleibt**, solange `_pp_flip_hold_slot`
rang-lokal auf `chunked_req` ausnimmt — denn `passes [0,1,1]` haengt an der
Chunk-Asymmetrie und nicht an der Chunk-Laenge.

**Korrektur einer Zuschreibung:** meine These ist NICHT "wer haelt den Slot ueber
den Reset". In diesem Specimen gab es keinen Reset — der Flip wurde abgebrochen
(§1). Die These ist enger: *der Aktuator, der die Raenge waehrend eines armed
window slot-uniform halten soll, ist rang-lokal konditioniert und hebt damit
seine eigene, im Docstring ausgeschriebene Korrektheits-Voraussetzung auf.*
