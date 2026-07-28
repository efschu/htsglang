# GPU-zu-GPU direkt über die BARs — Mechanismus, Grenzen, Bauplan

Stand 2026-07-28. Quelle für alle Treiber-Aussagen: `open-gpu-kernel-modules`
Tag **595.58.03**, geklont nach `/spinning/nvidia-open-595` (das ist exakt die
Version, die auf dem Rig-1-Host läuft: `NVIDIA UNIX Open Kernel Module
595.58.03`). Zeilenangaben beziehen sich auf diesen Baum.

Ziel dieses Dokuments: den Weg beschreiben, auf dem zwei Karten dieses Rigs
**direkt über PCIe** miteinander reden — ohne NIC, ohne System-RAM. Es geht um
den Mechanismus, nicht um die Frage, welche Kollektiv-Topologie am Ende
schneller ist.

---

## 1. Der Mechanismus existiert im Treiber bereits — und zwar ab Turing

Der entscheidende Fund: NVIDIAs eigener BAR1-P2P-Pfad ist kein
Hopper-Sonderweg im Sinne von "nur auf Hopper-Silizium gebaut". Die
Setup-Funktion ist ein **Turing**-HAL:

`src/nvidia/src/kernel/gpu/bus/arch/turing/kern_bus_tu102.c:525`
```
kbusEnableStaticBar1Mapping_TU102(pGpu, pKernelBus, gfid, bar1Offset)
```

TU102 ist die Turing-Basis; alle drei Karten dieses Verbunds (2080 Ti = TU102,
3080 = GA102, 5090 = Blackwell) erben diesen HAL. Das Silizium kann es also.
Was fehlt, ist ausschließlich die **Freischaltung** und eine
**Größenannahme** — beides Software.

### Was die Funktion tut (das ist die "Memory-Table" aus der Nutzerfrage)

1. Sie beschreibt einen FB-Bereich der eigenen Karte als Memdesc.
2. Sie mappt ihn per `kbusMapFbApertureSingle()` in das **eigene BAR1**.
   Genau hier liegt die Tabelle: BAR1 ist **kein** starres 1:1-Fenster auf
   das VRAM, sondern eine **Apertur mit Seitentabellen**. Welche VRAM-Seiten
   an welchem BAR1-Offset erscheinen, programmiert der Treiber.
3. Sie bildet die entstandene Busadresse
   (`gpumgrGetGpuPhysFbAddr(pGpu) + bar1Offset`) als Memdesc mit dem
   Aperture-Typ **`ADDR_SYSMEM`** ab und legt sie unter
   `pKernelBus->bar1[gfid].staticBar1.pDmaMemDesc` ab.

Punkt 3 ist der Kern. Der Peer bekommt das Ziel als "System-Speicher"
vorgesetzt — aber die physische Adresse dahinter ist die **MMIO-Apertur der
Zielkarte**, nicht der Hauptspeicher. Der Peer schreibt also mit ganz
gewöhnlichen DMA-Schreibvorgängen, und die PCIe-Fabric leitet sie an die
Zielkarte statt an den RAM. Kein Byte berührt den System-RAM.

`kbusGetBar1P2PDmaInfo_GH100` (`kern_bus_gh100.c:1655`) reicht genau diese
Adresse und Größe an die Quellkarte weiter.

### Warum der tinygrad-Patch funktioniert

Der tinygrad-Patch (Branch `570.148.08-p2p`, gegen NVIDIA 570.148.08 geprüft)
baut denselben Effekt von Hand nach, statt den offiziellen Pfad zu benutzen.
Er schreibt in `nv_gpu_ops.c` die Peer-PTEs um:

- `GMMU_APERTURE_PEER` wird zu `GMMU_APERTURE_SYS_COH` / `SYS_NONCOH`,
- das Adressfeld wechselt von `fldAddrPeer` auf `fldAddrSysmem`
  (`gmmu_fmt.c`),
- `fabricBaseAddress` wird auf `gpumgrGetGpuPhysFbAddr(peer)` gesetzt.

Das ist inhaltlich dieselbe Aussage wie Punkt 3 oben — "adressiere den Peer
als Systemspeicher an seiner BAR1-Busadresse" —, nur ohne die
Verwaltungsstruktur des offiziellen Pfades. Dazu schaltet er die
BAR1-P2P-HAL-Zeiger von den Stubs (`_395e98`, `_d69453`) auf die echten
`_GH100`-Implementierungen und setzt `p2pOverride`/`forceP2PType` auf BAR1.

**Konsequenz für uns:** Der tinygrad-Weg ist eine gültige Abkürzung, aber der
offizielle Weg ist im 595er-Baum vollständig vorhanden und sauberer. Alle
Angriffspunkte des Patches existieren in 595.58.03 unverändert (geprüft:
`kern_bus_gp100.c`, `nv_gpu_ops.c`, `gmmu_fmt.c`, `kernel_bif.c`,
`g_kern_bus_nvoc.c`, `nv-pci.c`).

---

## 2. Die drei Bedingungen, die heute NEIN sagen

`kbusIsPcieBar1P2PMappingSupported_GH100` (`kern_bus_gh100.c:1447 ff.`)
verlangt:

1. **Freischaltung**: `pKernelBif->pcieP2PType == NV_REG_STR_RM_PCIEP2P_TYPE_BAR1`
   **oder** die Property `PDB_PROP_KBUS_SUPPORT_BAR1_P2P_BY_DEFAULT`.
2. **Beide** Karten müssen `kbusIsStaticBar1Enabled()` erfüllen.
3. Es darf noch keine Mailbox-P2P-Verbindung zwischen dem Paar bestehen.

Bedingung 1 ist billig: `NV_REG_STR_RM_PCIEP2P_TYPE` heißt als Regkey
**`"RMPcieP2PType"`** (`src/nvidia/interface/nvrm_registry.h:1118`), Wert
`1` = BAR1. Regkeys nimmt das Modul über den Parameter
`NVreg_RegistryDwords` entgegen — **dafür ist kein Patch nötig**, nur ein
Modul-Neuladen.

Bedingung 2 ist die eigentliche Wand, und sie hat zwei getrennte Ursachen.

---

## 3. Die Größenwand — und warum ReBAR sie NICHT einreißt

`kbusEnableStaticBar1Mapping_TU102` mappt **die gesamte client-sichtbare FB**:

```c
bar1MapSize = RM_ALIGN_DOWN(memmgrGetClientFbAddrSpaceSize(pGpu, pMemoryManager),
                            RM_PAGE_SIZE_2M);
```

und die Aufrufstelle (`kern_bus_gm107.c:1163`) legt den Startoffset auf

```c
NvU64 bar1Offset = NV_ALIGN_UP(consoleSize + p2pPcie.writeMailboxTotalSize,
                               RM_PAGE_SIZE_512M);
```

Für die 5090 (32-GiB-BAR1, ~32 GB VRAM) ist das erfüllbar. Für die 3080er
scheitert es **doppelt**: 20 GB FB passen nicht in 256 MB BAR1, und schon die
512-MiB-Aufrundung des Startoffsets ist größer als deren ganzes BAR.

### Der falsifizierte Altbefund

Die Übergabe (`04_OFFEN.md` §3) nimmt an, ReBAR im Rig-1-UEFI verschaffe auch
den 3080ern volle BARs und damit ein volles P2P-Mesh. **Das ist falsch.**
Gemessen per `lspci -vv` auf dem Rig-1-Host:

| Karte | BAR1 aktuell | BAR1 unterstützt (Gerätekapazität) |
|---|---|---|
| 3080 `05:00.0` | 256 MB | 64 MB / 128 MB / **256 MB** |
| 3080 `0b:00.0` | 256 MB | 64 MB / 128 MB / **256 MB** |
| 5090 `0a:00.0` | 32 GB | bis **32 GB** |

256 MB ist das **Maximum, das die 3080er überhaupt anbieten**. ReBAR kann
nur unter den angebotenen Größen wählen, nicht neue erfinden. Ein
BIOS-Update ändert daran nichts — die Liste kommt vom VBIOS der Karte.

Damit ist "erst ReBAR, dann volles Mesh" als Weg **erledigt**. Wer die
3080er einbinden will, muss mit 256 MB auskommen.

---

## 4. Der Ausweg: Fenster statt Vollabbildung

Die Größenannahme steckt ausschließlich in den zwei Zeilen oben — nicht in
der Hardware und nicht im Protokoll. `kbusEnableStaticBar1Mapping_TU102`
nimmt `bar1Offset` bereits als Parameter, und `memdescDescribe` kann jeden
FB-Ausschnitt beschreiben. Ein **Fenster-Mapping** ist derselbe Codepfad mit
anderen Zahlen:

- statt der ganzen FB einen reservierten **Staging-Bereich** beschreiben
  (Größenordnung 128–224 MB, muss in 256 MB minus Konsolenpuffer passen),
- den Startoffset nicht auf 512 MiB aufrunden, sondern auf 2 MiB,
- `staticBar1.size` entsprechend klein setzen.

Danach ist `kbusIsStaticBar1Enabled()` auch auf den 3080ern wahr, Bedingung 2
ist erfüllt, und der **offizielle** BAR1-P2P-Pfad trägt alle drei Karten.

Der Preis ist eine Indirektion auf der Zielkarte: der Peer schreibt in das
Fenster, nicht an die endgültige Stelle. Ein lokaler Kernel auf der
Zielkarte kopiert Fenster → Ziel. Diese Kopie läuft im eigenen VRAM mit
mehreren hundert GB/s; für eine 20-KiB-Nachricht sind das Bruchteile einer
Mikrosekunde, gegen ~6 µs Drahtzeit auf der Strecke. Sie fällt nicht ins
Gewicht, und sie berührt **keinen System-RAM**.

Für Kollektive ist das Fenster reichlich bemessen: bei Chunk-Größen von
20–80 KiB fasst ein 128-MB-Fenster Tausende Chunks gleichzeitig.

### Alternative ohne Fenster, falls nur eine Richtung gebraucht wird

Für "3080 schreibt in die 5090" wird streng genommen **nur das BAR der
Zielkarte** gebraucht — die Quellkarte muss lediglich den fremden
SYSMEM-Memdesc in ihre GMMU mappen, was ihr eigenes BAR nicht berührt. Die
symmetrische Bedingung 2 ist an dieser Stelle konservativer als nötig.
Lockert man sie auf "die **Zielkarte** braucht Static BAR1", funktioniert
Hub-Verkehr in die 5090 ohne jede Small-BAR-Arbeit.

Die Rückrichtung ginge dann als **Pull**: die 3080 *liest* aus der
5090-BAR. Das kostet allerdings — Befund `02_BEFUNDE.md` §3 zeigt, dass
non-posted Reads aus einer BAR deutlich teurer sind als posted Writes
hinein (2080 Ti: 3254 MB/s schreiben gegen 1132 MB/s lesen). Push in ein
3080-Fenster ist der schnellere Weg, Pull der billiger zu bauende.

---

## 5. Reihenfolge des Vorgehens

Aufsteigend nach Eingriffstiefe. Jede Stufe ist für sich prüfbar.

**Stufe 0 — ohne Patch.** `NVreg_RegistryDwords="RMPcieP2PType=1"` setzen und
prüfen, ob auf der 5090 Static BAR1 überhaupt hochkommt (`dmesg`:
"Static bar1 mapped offset ... size ..."). Erwartung: Bedingung 2 scheitert,
weil die 3080er kein Static BAR1 haben — aber die Meldung sagt uns, ob die
5090-Seite trägt. Braucht ein Modul-Neuladen, also Nutzer-Freigabe.

**Stufe 1 — Big-BAR-Patch (tinygrad-Klasse), 5090 als Ziel.** Portierung auf
595.58.03; alle Angriffspunkte sind vorhanden. Liefert 3080 → 5090.

**Stufe 2 — Fenster-Static-BAR1 für die 3080er.** Die eigentliche
Eigenleistung. Liefert das volle Mesh in beide Richtungen.

**Stufe 3 — Nachweis und Zahlen.** Erst Byte-Beleg (Muster schreiben, auf der
Zielkarte zurücklesen, `bad_bytes=0`), dann dieselbe Größenleiter wie
`gpurdma_04_bench`, damit die Zahlen neben den vorhandenen stehen:

| Größe | NCCL send/recv (heute) | NIC-Relay direkt | P2P über BAR |
|---|---|---|---|
| 20 KiB | 37,41 µs | 7,37 µs | zu messen |
| 80 KiB | 44,27 µs | 16,56 µs | zu messen |
| 1 MiB | 220,81 µs | 169,88 µs | zu messen |

Erst messen, dann behaupten.

---

## 6. Warum der NIC-Umweg heute überhaupt gewinnt

Zur Einordnung, weil die Frage berechtigt ist: eine Gen3-x4-NIC kann keine
x16-Direktverbindung schlagen. Der Vergleich in `02_BEFUNDE.md` §7 misst auch
nicht das — er misst **DMA-Engine mit fertigem Deskriptor gegen ein
Software-Kollektiv**.

20 KiB brauchen auf Gen3 x4 rund 5,9 µs reine Drahtzeit. Die 7,37 µs des
NIC-Arms sind also fast vollständig Draht — dieser Pfad ist praktisch
optimal. Dieselben 20 KiB sind durch den System-RAM in unter 2 µs kopiert;
die 37,41 µs von NCCL können damit gar keine Transportkosten sein, sondern
sind Protokoll-, Start- und Synchronisationskosten.

Der NIC-Vorsprung ist also ein **Software**-Vorsprung, kein Bandbreiten-
vorsprung — und genau deshalb ist zu erwarten, dass ein direkter BAR-Pfad
über x16 ohne Zwischenstation beide schlägt. Erwartet, nicht gemessen.

Die Sonde `ramrelay.cu` (Modi `ce`/`zc`) in diesem Verzeichnis beziffert die
Latte: sie misst denselben RAM-Pfad ohne NCCL-Protokoll. Sie ist gebaut, aber
noch nicht gefahren — die Karten tragen laufenden sglang-Betrieb.

---

## 7. Was NICHT gilt

- **ReBAR als Weg zu großen 3080-BARs**: widerlegt, §3.
- **`cudaHostRegisterIoMemory` auf NVIDIA-BARs**: bleibt zu, benannter
  Treiber-Guard `osCheckGpuBarsOverlapAddrRange` (`02_BEFUNDE.md` §8). Der
  hier beschriebene Weg umgeht ihn nicht, sondern ersetzt ihn: die
  Adressierung passiert im Kernel über Memdescs, nicht über eine
  Userspace-Registrierung.
- **System-RAM als Zwischenstation**: nach Nutzer-Vorgabe kein Ziel mehr.
  `ramrelay` bleibt reine Vergleichslatte.
