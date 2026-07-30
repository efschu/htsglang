# GPU-Phase des BAR1-Direktmodus (#292)

Die CPU-Phase ist abgeschlossen (`CUDA_VISIBLE_DEVICES=99` durchgehend, keine
Karte angefasst). Was hier steht, ist die Kommandoliste fuer die Karten --
wortwoertlich ausfuehrbar, in der Reihenfolge, in der sie laufen muss.

Die Begruendung je Punkt steht in `docs/dev/INTEGRATION_R3_VALIDATION.md`,
Abschnitt "BAR1-Direktmodus graphfest (#292)". Dieses Blatt wiederholt sie
nicht, es fuehrt aus.

**Reihenfolge ist Absicht: erst Byte-Belege, dann Zahlen. Keine Zeitmessung
vor einem bestandenen Byte-Beleg.**

---

## 0. Wo das laeuft

Die BAR1-Arbeit laeuft **nicht** dort, wo die Batterie laeuft. Der gepatchte
Treiber, `dmabuf_holder` und `/dev/dmabuf_holder` liegen auf dem PVE-Host;
CT999 kann das Holder-Geraet nicht oeffnen (Major 10 steht nicht in der
Device-Allowlist des Containers). Also: Kommandos ueber ssh auf dem Host,
Artefakte im Container.

    Host            192.168.0.1
    Schluessel      /root/.ssh/id_root@proxmox
    Pfadabbildung   <Container-Pfad>  ->  /spinning/subvol-999-disk-0<Container-Pfad>

Jeder ssh-Aufruf bekommt ein **Timeout**. Ein unbegrenzter ssh in einem
einzelnen Bash-Aufruf macht den Agenten unerreichbar, ohne dass jemand es
sieht (`scripts/gpu_battery/battery_host.sh`, `host_ssh_for`).

### Voraussetzungen auf dem Host (04_BETRIEB.md der P2P-Uebergabe)

Pruefen, nicht annehmen:

    ssh -i /root/.ssh/id_root@proxmox root@192.168.0.1 \
      'grep -i "^RegistryDwords:" /proc/driver/nvidia/params; \
       ls -l /dev/dmabuf_holder; lsmod | grep -c "^dmabuf_holder"'

Leere `RegistryDwords` heisst Serientreiber -- dann faehrt der Direktmodus
gar nicht, und jede Zahl aus so einem Lauf misst etwas anderes. Laden nach
`04_BETRIEB.md`, "Treiber laden" (`nvidia_modeset` muss mit entladen werden,
sonst scheitert `rmmod nvidia` mit "File exists").

### Umgebung fuer jeden Host-Lauf

Wortwoertlich aus `scripts/gpu_battery/s11_bar1_e2e.sh`, nur mit dem
Arbeitsbaum dieses Zweigs:

    V=/spinning/subvol-999-disk-0/spinning/htsglang-gpu/.venv
    W=/spinning/subvol-999-disk-0/spinning/wt-direkt-graph
    N=/spinning/subvol-999-disk-0/spinning/nvidia-open-595
    P=/spinning/miniforge3_local_install/bin/python3.12

    cd $W
    PYTHONPATH=$W/python:$V/lib/python3.12/site-packages \
    LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/nvidia/cu13/lib \
    CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 \
    SGLANG_HTCCL_BAR1_NV_QUELLE=$N \
    TORCH_EXTENSIONS_DIR=/spinning/subvol-999-disk-0/spinning/htccl_extcache_host \
    TORCH_CUDA_ARCH_LIST="8.6;12.0" MAX_JOBS=4 \
      $P <Kommando>

`CUDA_HOME` ist zwingend, sonst scheitert der JIT-Bau an `ninja`.
`MAX_JOBS=4`, weil die Kiste swaplos ist.

---

## 1. Sperren

Zwei `/tmp`-Namensraeume, und sie sehen einander nicht: `/tmp/gpu-card-N.lock`
in CT999 und der gleichnamige Pfad auf dem Host sind **verschiedene** Sperren.
Wer die Karten anfasst, nimmt **beide**.

    # Container-Seite (eine je Karte)
    for i in 0 1 2; do mkdir /tmp/gpu-card-$i.lock || echo "BELEGT: $i"; done
    printf 'holder=direktmodus_gpu_phase\nstep=292\nacquired=%s\nheartbeat=%s\n' \
      "$(date -Is)" "$(date -Is)" | tee /tmp/gpu-card-{0,1,2}.lock/info

    # Host-Seite, identisch, ueber ssh (siehe host_locks_acquire)

`mkdir` ist der atomare Erwerb, die `info`-Datei traegt Halter und Herzschlag.
**Eine fremde Sperre wird nie gebrochen** -- Halter aus `info` lesen, Operator
fragen, abbrechen. Herzschlag alle 60 s nachziehen, sonst raeumt der Reaper
(`/spinning/gpu-arb/arb-reaper.sh`) die Sperre unter dem laufenden Versuch weg.

Freigeben in **jedem** Ausgang, auch im Fehlerfall.

---

## 2. Der Graph-Beleg -- neun Gate-Faelle

    <Umgebung aus 0> benchmark/bar1_graph_check.py 0,1,2

Erwartet: **"Alle Gate-Faelle bestanden."** Neun Gate-Faelle stehen in der
Zusammenfassung, dazu `gitter` als Info-Fall (kein Gate):

    [Gate]  1blk-klein            [Gate]  pipe
    [Gate]  1blk-gross            [Gate]  pipe-direkt              <- neu (#292)
    [Info]  gitter                [Gate]  pipe-direkt-vorrat-leer  <- neu (#292)
    [Gate]  vorbehalt             [Gate]  broadcast
    [Gate]  zwei-graphen          [Gate]  broadcast-zwei-graphen

Nur die beiden neuen:

    <Umgebung> benchmark/bar1_graph_check.py 0,1,2 29593 pipe-direkt,pipe-direkt-vorrat-leer

`pipe-direkt` prueft zweierlei, was kein anderer Fall prueft:

* der Ergebnistensor muss **wirklich im BAR1-Fenster liegen** (`erg_fenster()`).
  Faellt das weg, hat der Fall den `direkt=0`-Kontrollpfad gemessen und
  bestanden, ohne die Frage zu beantworten. Der Fall bricht dann mit dem
  Hinweis auf `SGLANG_HTCCL_BAR1_PIPE_ERG_RING` ab, statt gruen durchzugehen;
* zurueckgelesen wird zusaetzlich ueber den **Host** statt ueber den L2 der
  Empfaengerkarte. Der L2 ist mit eingehenden PCIe-Schreibvorgaengen nicht
  kohaerent (`BEFUND_L2_NICHT_KOHAERENT.md`), also ist der zweite Lesepfad
  hier keine Formalie (Messdisziplin, Regel 3).

`pipe-direkt-vorrat-leer` ist die Negativkontrolle: mit `ERG_RING=2` gibt es
keine Graph-Plaetze, **jeder** aufgezeichnete Aufruf muss auf `direkt=0`
zurueckfallen und trotzdem die richtigen Bytes liefern. Ein Fall, der hier
einen Platz im Fenster findet, meldet eine kaputte Ringaufteilung.

---

## 3. Byte-Beleg eager, mit eingeschaltetem Handschlag

Vor dem Graphen, weil ein gefallener eager-Beleg jede Graph-Zahl entwertet.

    <Umgebung> env \
      SGLANG_HTCCL_BAR1_PIPE=1 \
      SGLANG_HTCCL_BAR1_PIPE_DIREKT=1 \
      SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH=1 \
      SGLANG_HTCCL_BAR1_PIPE_ERG_RING=5 \
      $P benchmark/bar1_diag.py 0,1,2

Damit laeuft der Freigabe-Handschlag auch eager mit (`ergSlack = 2`) und die
Flaggenfamilie 4 wird zum ersten Mal auf echter Hardware beschrieben. Bis
hierher hat sie nur eine Python-Simulation gesehen.

Nach **jedem** Lauf der Deckel:

    grep -c "Zeitlimit" <log>      # erwartet: 0

Der Handschlag ist eine neue Wartebedingung. Ein gerissener Zeitdeckel dort
ist der erste Verdacht, wenn etwas haengt, und er entwertet jede Zahl aus
diesem Lauf.

---

## 4. Registerzahl auf sm_120, am JIT-Objekt

Offline gemessen (`scripts/probe/bar1_pipe_spill.sh`) steigt die gitter-
Variante auf sm_120 von REG 40 auf 48, STACK bleibt 0. Ob das die Belegung
messbar drueckt, entscheidet die Karte. Gemessen wird am **gebauten** Objekt,
nicht am offline uebersetzten:

    ssh -i /root/.ssh/id_root@proxmox root@192.168.0.1 \
      'ls -l /spinning/htccl_extcache_host/htccl_bar1_pipe_ext_cuda_86_120/'

    ssh ... '/usr/local/cuda/bin/cuobjdump -res-usage \
      /spinning/htccl_extcache_host/htccl_bar1_pipe_ext_cuda_86_120/htccl_bar1_pipe_ext_cuda_86_120.so \
      | grep -E "Function|REG|STACK"'

**Erst die Zeitstempel lesen.** Der Extension-Cache ist ueber Boots hinweg
geteilt (ein Kaltbau kostet Minuten). Ist die `.so` aelter als
`python/sglang/srt/distributed/device_communicators/htccl_bar1_pipe_ext.py`,
misst `cuobjdump` den **alten** Kern -- dann ist die Zahl wertlos und das
Verzeichnis muss weg, bevor irgendetwas anderes laeuft.

---

## 5. A/B: `DIREKT=0` gegen `DIREKT=1`

Der Direktmodus ist bis heute uebersetzt und **ungemessen**.

Regeln fuer diese Messung, alle drei nicht verhandelbar:

* **Vorlauf**, sonst ist die Zahl falsch (P-State-Rampe: 726 us ohne, 95 us
  mit). Den Arbeitspunkt vorher anfahren, nicht in ihn hineinwachsen.
* **Verschraenkt** messen, nicht Arm nach Arm: A,B,A,B,... im selben Prozess,
  gleiche Groessen, gleiche Reihenfolge. Zwei getrennte Laeufe messen die
  Takt- und Temperaturlage mit.
* **Rauschboden zuerst**: ein A-gegen-A-Durchgang, bevor A gegen B berichtet
  wird. Was unter dem Rauschboden liegt, wird nicht berichtet.

    <Umgebung> env SGLANG_HTCCL_BAR1_PIPE=1 SGLANG_HTCCL_BAR1_PIPE_DIREKT=0 \
      $P benchmark/bar1_diag.py 0,1,2
    <Umgebung> env SGLANG_HTCCL_BAR1_PIPE=1 SGLANG_HTCCL_BAR1_PIPE_DIREKT=1 \
      $P benchmark/bar1_diag.py 0,1,2

Und dasselbe fuer `SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH=0/1` bei
`DIREKT=1` -- das ist die Frage nach den 8 Registern aus Punkt 4.

Erwartet wird ein gesparter VRAM-Durchgang beim Empfaenger. Gegen den
PCIe-Engpass dieses Rigs ist das wenig; **ein Nullbefund ist ein moegliches
und berichtenswertes Ergebnis**, kein Fehlschlag. Berichtet wird in
ms/Runde, nicht in tok/s.

---

## 6. Standardlauf e2e

Mit `SGLANG_HTCCL_GRAPH_FREIGABE=1` und dem graphfesten Direktmodus, sonst
wie `scripts/gpu_battery/s11_bar1_e2e.sh`.

Zu erwarten ist, dass der Graph-Vorrat bei einem echten Modell (viele
Aufrufstellen je Graph, viele Graphen) schnell leer ist und der Rest
`direkt=0` faehrt. Die Meldung dazu erscheint einmal je Rang und nennt die
Zahlen. Das ist der ehrliche Rahmen des Features, nicht sein Scheitern: der
graphfeste Direktmodus traegt eine BESCHRAENKTE Zahl aufgezeichneter
Aufrufstellen, und jeder Platz kostet `roundup(max_bytes, 4096)` Byte im
BAR1-Fenster.

---

## Abbruchkriterien

Abbrechen, Sperren freigeben, Befund melden -- nicht weiterfahren und nicht
umkonfigurieren, um das Hindernis zu umgehen. Eine Konfiguration zu aendern,
um ein Hindernis zu umgehen, tauscht die Frage aus.

1. **Fremde Sperre** auf einer der drei Karten (Container- oder Hostseite).
   Halter aus `info` melden, Operator fragen.
2. **`RegistryDwords` leer** oder `/dev/dmabuf_holder` fehlt -- Serientreiber,
   der Direktmodus faehrt nicht.
3. **Ein Gate-Fall aus Punkt 2 faellt.** `SGLANG_HTCCL_GRAPH_FREIGABE` bleibt
   aus, Punkt 5 und 6 laufen nicht. Der Info-Fall `gitter` darf fallen.
4. **`pipe-direkt` meldet "Ergebnistensor NICHT im BAR1-Fenster".** Dann hat
   der Lauf den Kontrollpfad gemessen; die Zahl daraus ist keine Aussage
   ueber den Direktmodus.
5. **Zeitdeckel gerissen** (`Zeitlimit` im Log, `htccl.status()` ungleich 0).
   Jede Zahl aus diesem Lauf ist entwertet.
6. **JIT-Objekt aelter als die Quelle** (Punkt 4). Cache weg, neu bauen,
   von vorn.
7. **VRAM-Korridor verletzt**: frei < 400 MiB absolut, oder > 1,5 GiB netto
   verschwendet. Nicht auf rot messen.
8. **Ein Haenger.** `py-spy dump` VOR jedem Kill, und nur die eigenen PIDs
   killen -- nie ein breites `pkill`, auf dieser Kiste laufen fremde Server.
