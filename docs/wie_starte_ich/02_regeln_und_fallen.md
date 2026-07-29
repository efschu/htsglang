# 02 — Regeln und Fallen (PFLICHTLEKTUERE, jede hat schon einen Lauf gekostet)

## Arbitrierung & Umfeld
1. **Locks sind VERZEICHNISSE**: `/tmp/gpu-card-N.lock` per `mkdir` (atomar)
   nehmen, `info`-Datei mit holder+Zeitstempel hineinschreiben, gelegentlich
   touchen. Einfache Dateien kollidieren ("gehalten von unknown").
   Fremde Locks NIE brechen — zurueckmelden statt uebernehmen.
2. **VRAM-Korridor**: vor jedem Boot muss jede Karte >= 400 MiB absolut frei
   sein bzw. praktisch leer fuer den Standardlauf. Kein Test auf rot.
3. **Karten koennen dem Nutzer gehoeren** (gerade: Treiber-Update-Phase).
   Boot nur, wenn der Auftrag es ausdruecklich erlaubt.
4. Mehrere Agenten teilen sich diese Kiste (swaplos!): GPU-Boots
   serialisieren, keine Parallelboots, keine dicken RAM-Jobs nebenher.
   CUDA-Builds immer mit MAX_JOBS=4 / -j4.

## Prozess-Hygiene
5. **NIE breites `pkill -f`** — es killt fremde Server und ggf. die eigene
   Shell. Nur eigene, gemerkte PIDs beenden (`kill <pid>`), notfalls
   `pkill -x <exaktername>`.
6. **`py-spy dump --pid <pid>` VOR jedem Kill** eines haengenden Prozesses.
   Der Stack ist der Befund; ohne ihn ist der Haenger unerklaerbar weg.
7. **Nie unbegrenzt warten**: jeder curl mit `-m`, jedes Warten als Poll-
   Schleife mit Frist. Ein unbegrenztes Wait in einem Bash-Aufruf blockiert
   auch die Nachrichten-Zustellung an dich (Wedge-Falle).
8. **Nie Serverlogs in den Kontext**: Log in Datei, gezielt greppen,
   im Fehlerfall maximal `tail -30`.

## Konfigurations-Fallen
9. **Device-Order**: torch-Reihenfolge != NVML/nvidia-smi-Reihenfolge.
   `cuda:0` ist hier die 5090. Karten im Zweifel ueber PCI-Adresse/Namen
   aufloesen, nie ueber feste Indizes aus anderer Quelle.
10. **Reserve-Falle**: `--rank-auto-reserve-mib` 2200 auf den 3080ern kippt
    im ersten echten Prefill (Warmup ueberlebt!). 2700 ist die Grenze.
11. **NCCL 2.28.9 auf dem Host**: mehrere Raenge je physischer GPU
    (Co-Location) brauchen >= 2.30 -> nur im Container. Der Standardlauf
    (1 Rang je Karte) ist davon NICHT betroffen.
12. **Schreibtisch-Arbeit ohne Karten**: `CUDA_VISIBLE_DEVICES=99` setzen.
    Leer oder `-1` laesst sglang beim Import abstuerzen.
13. **Eigene Temp-Dateien** nach `$CLAUDE_JOB_DIR/tmp` bzw. ein eigenes
    Verzeichnis — parallele Jobs ueberschreiben sich sonst in /tmp.
14. **Repo-Skripte tragen nur Platzhalter** (IPs/Pfade): ohne
    `source /root/rig-env.sh` scheitern sie absichtlich. Niemals echte
    Werte ins Repo schreiben; PATs nie in Push-URLs einbetten.

## Validierungs-Grundsaetze
15. Immer mit CUDA-Graphen + Spekulation validieren, NIE eager — eager
    versteckt genau die Graph-Replay-Bugklasse, die hier mehrfach zuschlug.
16. Nach ZEIT begrenzen (10-20 s Messfenster Default, kein Punkt > 60 s).
17. Boots einzeln: eine Aenderung je Boot, Ergebnis notieren, dann weiter.
