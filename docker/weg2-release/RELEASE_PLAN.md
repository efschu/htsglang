# Docker-Release htsglang: Upgrade des August-Containers (27B + NF, barlink BAR1 und NCCL, Modellformate)

**Stand:** 2026-09-25 ~01:55Z · **Verfasser:** 27B-Sitz, Agent R · **Status:** ENTWURF, Kontext für RC2-final erzeugt.
Nichts veröffentlicht, kein Image gebaut, keine GPU, kein Boot. Auf dem Proxmox-Host nur lesend (SSH als root mit dem Schlüssel aus `/root/.ssh/`, vom Nutzer erlaubt).
**Auftrag (Nutzer 24.09. ~20:40Z):** „du und der nf agent solltet solangsam auch das docker release planen und vorbereiten, es soll mit barlink und nccl funktionieren“
**Nutzer-Entscheide (über den Operator, wörtlich):** „docker auf host, den von august weiterverwenden und aktualisieren upgraden / 1 ja / 2 auf host / 3 veröffentlichung geplant als upgrade zu unserem bisher angebotenen docker container auf meinem github / 4 ja / 5 ja / 6 ja muss aber optional sein“; „nein die modellformate müssen schon mit dem docker funktionieren“.
**NF-Teil:** [`NF_PROFILE.md`](NF_PROFILE.md) (NF-Sitz, fertig, 1166 Z.). Das NF-Profil `profiles/nf.env` ist daraus §11.1 **wörtlich** übernommen.

**Entscheide des Operators (24.09. ~22:45Z) und wo sie umgesetzt sind:**

| # | Entscheid | Umsetzung |
|---|---|---|
| F7 | Default `bar1`, `nccl` nur ausdrücklich, beide müssen funktionieren | `entrypoint.sh`: `auto` gestrichen (verweigert mit Grund); Abnahme bootet beide (`serve bar1`, `serve nccl`) |
| F8 | Instrument-Env im Release aus, schaltbar | `HTSGLANG_INSTRUMENTS` Default 0, vor dem Profil exportiert (nf.env liest es beim Sourcen); Abnahme `serve` mit 1 (Parität), Schritt `release` mit 0 |
| F9 | #599 nicht besetzt | — (die Arbeit soll konvergieren) |
| F10 | NCCL-Version der Linie benennen, mit Beleg | **2.28.9+cuda13.0** (§4, Beleg); Image erzwingt cu13 2.28.9 zuletzt, prüft Banner und sha256 gegen die Rig-Datei. **Folgefrage F10b** (Co-Location) §15 |
| F11 | erledigt (Snapshots gelöscht, `/` 740G frei) | R-9 geschlossen |
| F12 | Host-Images nur beim Bau, nie `cu130-nccl2307` vor der Ablösung | `host_acceptance.sh build`: Liste, Löschen nur mit `PRUNE_HOST_IMAGES=1`; geschützt: alles `ghcr.io/efschu/*`, `*cu130-nccl2307*`, `nvidia/cuda:*`, jedes von einem Container genutzte Image |
| F13 | offen beim Nutzer | `host_acceptance.sh`: Schalter `HOUSE_GUARD=memlimit\|ct999-ruht`, Pflicht für GPU-Schritte (§8) |
| F14 | neuer versionierter Tag; öffentlichen Tag nur mit Go ersetzen | Tag `htsglang:cu129-weg2-<linie>-<release>-<sha10>` (z.B. `…-27b-rc2-…`), Build verweigert einen vorhandenen Tag; kein Push im Skript |
| F15 | README-Neufassung vorbereiten, nicht veröffentlichen | `README_RELEASE_DRAFT.md` |
| F16 | Downloads laufen | Stand 22:50Z: 27B-FP8, 27B-NVFP4-RadixArk, 27B-GGUF (IQ4_XS + Q8_K_XL) und NF-NVFP4 liegen vollständig; NF-GGUF lädt (§5) |

**RC2-final (Operator 25.09.): INT8-Freeze abgenommen, Build-Kontext erzeugt.**

| Punkt | Stand |
|---|---|
| Baum | `desk/27b-release-rc2-0924 @ b5c7d01614` (gepusht). Stufe B steckt als Pick `68bc631d8c` darin (patch-id = bae049a3b4); `prepare_context.sh` prüft jetzt Vorfahr ODER Pick und die fünf Nähte inhaltlich |
| Profil `27b` | RC2-final INT8 = `arm_rc2final.sh` (FORMAT=int8): RC2 + `--p-prefill-graph-split 7` (P_GRAPH_SPLIT=1). Gate: Env 34/34 gleich, geparster Launcher-Namespace (Parser von b5c7d01614) gleich |
| Profil `27b-fp8` | **abgenommen** (Operator 25.09.: weg2rc2f8 FERTIG 6/6, GEN/Nadeln MATCH, 0× KEPT, Decode INT8-Niveau, P-Leiter 3054/3899/3806 tok/s). Baut auf `27b.env` auf + `--model …-FP8 --fp8-uniform-marlin --weg2-xchg-census-foreign` = `arm_rc2final_fp8.sh` (FORMAT=fp8); Gate wie INT8 gleich. P-Prefill 44–49 % von INT8 (Leiter weg2rc2f 6600/8809/7711), nicht 55–65 % wie Ks Notiz |
| Kontext | `ctx/27b-b5c7d01614`, 530 MiB scheinbar; `MANIFEST.sha256` = **`1fde09c8983603254fd9a61a029b66b094752f980143e4676094d5fec83c7b3c`** (FP8 freigeschaltet per `--refresh-tools`; Vorgänger `cde24e5b…` und `7fd2d42a…` in `MANIFEST.history`) |
| Host-Trockenlauf | 25.09. 01:56Z, nur lesend: Manifest ok, `src/` sauber; Bau würde verweigert, solange CT999 bootet (MemAvailable 23 GiB < 32, Boot-Prozesse in CT999); `spinning/docker` 824 GiB frei; Builder-Plan 24g/24g, 12 CPUs, shares 256; Basis `nvidia/cuda@sha256:020bc241…`; F12: 78 Kandidaten nur gelistet |
| JIT-Vorbau | zusätzlich `hicache_hash_cpp_avx2` (CPU-Ext, am FP8-Boot belegt, 109 s) |
| Bau | `host_build.sh`: eigener Builder mit Deckel (24 GiB, 12 CPUs, shares 256, oom_score_adj 500), Manifest-Prüfung, Vorbedingungen; Schätzung 1–2 h, ~30 GiB Image, Spitze ~70–80 GiB scheinbar |
| Bekannt | `KNOWN_ISSUES_RC2.md`: Ks FP8-Notiz, erster D→P-Flip 8–16 s, kosmetischer 503 |
| **Nutzerentscheid 25.09. ~05:30Z: ZWEI Images** | 27B aus RC5 und NF aus der NF-Linie; die Vereinigung der Linien ist ein Folgeprojekt. Bau 27B im bootfreien Fenster (~06:00–07:00Z) auf Go des Operators, NF direkt danach; Veröffentlichung weiter nur mit Go des Nutzers |
| **Kontexte 05:40Z** | **27B:** `ctx/27b-5f13f1aad9`, Manifest **`bbdaf06c254c8fc5012019d32a14f2aee5fbafe42b4379faf06555e3452951f1`**, Tag `htsglang:cu129-weg2-27b-rc5-5f13f1aad9`, push_state `pushed`. **NF:** `ctx/nf-d93a17316b` (`desk/nf-platztausch-0922 @ d93a17316b`, aus dem gemeinsamen Objektspeicher geklont, NFs Worktree nur gelesen, 1860 Commits), Manifest **`716fb2f314b4f2432c48d87b47e7fa9335531a4ad1deef5e720391426ebfe11c`**, Tag `htsglang:cu129-weg2-nf-rc1-d93a17316b`, push_state **`UNPUSHED, Veroeffentlichung erst nach Push durch den Nutzer`** (benannte Ausnahme `--allow-unpushed`; BUILD_INFO und Image-Label `htsglang.push_state`; die Sperre sitzt in `host_publish.sh`: Revision muss auf einem Remote-Branch liegen, geprüft am Git-Stand). Stufe B im NF-Baum als Pick `b6bd57ae6f` (patch-id = bae049a3b4, Nähte inhaltlich in allen fünf Dateien, Stufe-B-Tests 6/6 grün auf einem NF-Abbild) |
| **Gemeinsame Vorbau-Schicht** | Lock, FlashInfer-Modulliste, `prebuild_jit.py`, Dockerfile und Kernel-Wheel beider Kontexte byte-gleich, NCCL-sha gleich; `ARG HTSGLANG_LINE` und `ARG PUSH_STATE` stehen jetzt erst am Ende des Dockerfiles (sonst hätte die Linie alle RUN-Schritte getrennt). ⇒ Das NF-Image nutzt die Schichten 1–3b des 27B-Baus (gleicher Builder, nicht dazwischen aufräumen): ~15 min |
| **Profile** | 27B: int8/fp8/nvfp4 abgenommen, gguf vorbereitet (RC5-GGUF-Boot). NF: `nf` **vorbereitet**, bis NF die x177-Nachführung liefert (die Datei trägt noch x163; abgenommen ist INT4-Mixed x177), `nf-nvfp4` **experimentell** (läuft nur mit `HTSGLANG_ALLOW_EXPERIMENTAL=1`, Form fehlt noch), `nf-gguf` **geplant**. Neuer Profil-Stand im Entrypoint: abgenommen / experimentell / vorbereitet / geplant; Linie wird vor dem Stand geprüft. NF-Profildaten über `profiles/nf.assets` (x177: drei Wake-Kredit-Referenzlogs von fnFL2x162, 16 MiB, ohne Schlüssel) |
| **NF-Container-Bedarf (Abnahme)** | tmpfs `/mnt/nf-experts` 72 GiB (INT4 ~39, NVFP4 ~46 GiB belegt), /dev/shm 16g (Profil-Minimum), BAR1-P2P aller Karten wie 27B, Host-RAM-Spitze 83 GiB (INT4) bzw. 91 GiB (NVFP4) ⇒ `host_acceptance.sh`: MemAvailable-Minimum je Profil 90 (nf) bzw. 98 (nf-nvfp4, praktisch nur mit `HOUSE_GUARD=ct999-ruht`) |
| **RC5-Kontext (25.09. 05:26Z) = Release-Stand** | `ctx/27b-5f13f1aad9` für `desk/27b-release-rc5-0925 @ 5f13f1aad9` (RC4 + K `b857a22cb5` Reload-load_config je Runner + S `1f8c24d338` Draft-Ladeformat-Resolver; gepusht; Stufe B als Pick `68bc631d8c`), **Manifest `27b2a86b4a1d6af5328714bba69959064cb0a350819453295e49f18579c68e14`**, 531 MiB, 13 022 Dateien, neue Schichtfolge (§7.1). Lock, FlashInfer-Modulliste (52, ohne Zeitstempel) und `prebuild_jit.py` byte-gleich zum RC4-Kontext. Profile: `27b`, `27b-fp8`, `27b-nvfp4` **abgenommen** (INT8/FP8/NVFP4-LoadConfig per sha256 = RC4; NVFP4: RC4-Boot weg2rc4n4, Urteil L JA), `27b-gguf` **vorbereitet** bis zum RC5-GGUF-Boot (weg2rc5gg, seit 05:18Z). Alle vier gegen `arm_rc5.sh` gegatet (Argv und Env byte-gleich zu `arm_rc4.sh`, Namespace des RC5-Launchers gleich). `--refresh-jit`/`--refresh-tools` direkt danach: nichts geändert (beide Modi sind jetzt ohne Änderung wirkungslos, statt den Digest zu verschieben). Der erste Bau mit der neuen Folge läuft voll (die ARG-Umstellung ändert die Umgebung aller RUN-Schritte, die alten Cache-Schichten passen nicht mehr) |
| **RC4-Kontext (25.09. 04:20Z)** | `ctx/27b-9738626129` für `desk/27b-release-rc4-0925 @ 9738626129` (gepusht; Stufe B als Pick `68bc631d8c`), **Manifest `11784ab15c26c05bca1d9c1626cc71ff46d51aa5b7cb13ecf16283bf20139cb5`**, 531 MiB scheinbar, 13 020 Dateien; Lock byte-gleich zu RC2-final; FlashInfer 52 Module, tvm-ffi-Saat 194 (+2 seit dem NVFP4-Boot); mit Schicht 6b und roter Selbstprüfung. Profile: `27b` und `27b-fp8` abgenommen (Form in RC4 unverändert), `27b-nvfp4` und `27b-gguf` **vorbereitet** -- alle vier gegen `arm_rc4.sh` gegatet (Env 34/34 bzw. 35 mit G3-Instrument, geparster Namespace des RC4-Launchers gleich). Nach den RC4-Formatboots: `prepare_context.sh --refresh-jit` (neue JIT-Artefakte), nach Freigaben `--refresh-tools`. Erzeugt in einer eigenen 2-GiB-cgroup (cpu.idle), weil die gemeinsame Test-Sperre `incg.lock` 30 min von einem Gate-Lauf belegt war; Spitze 0,5 GiB |
| Host-Bau Versuch 3 (02:44:07Z) | **Bau grün, Image nicht bootfähig.** `htsglang:cu129-weg2-27b-rc2final-b5c7d01614`, ID `sha256:d43cbd3fd7bd…`, 30,0 GB, 53 min (Vorbau 38,7 min: 86 21/21, 120f 31/31, gencode gleich Rig; barlink 2 gebaut, dmabuf ohne NV-Header abgelehnt; hicache_hash gebaut). Builder-Spitze 20,6 GiB (memory.peak, kein Neustart), Host-MemAvailable-Minimum 68,2 GiB. **JIT verdict INCOMPLETE: TMS-Preload `ld: cannot find -lcudart`** -- der Rig-venv trägt zehn von Hand angelegte Links unter `nvidia/cu13/lib` (in keinem RECORD), das Image nicht; der Launcher baut den Hook bei jedem Boot und verweigert dann (`Weg2LaunchRefused`). Fix: Schicht 6b nach dem Vorbau (Links wie am Rig, Hook bauen, Bericht nachziehen), Selbstprüfung scheitert ohne Hook; Kontext `1f23d696…`; inkrementeller Neubau unter neuem Tag `…-r2` mit unveränderter SCM-Version (Cache bis Schritt 26 bleibt) |
| Host-Bau Versuch 2 (02:04:40Z) | **gescheitert 02:36:53Z** im JIT-Vorbau (Schritt 26/31): (1) `[prebuild] flashinfer 86: 0 built, 21 errors` -- Parser-Fehler in `prebuild_jit.py`: die gencode-Erwartung wurde am Komma zerlegt, obwohl jedes Token selbst eines enthält -> jedes fertig gebaute Modul als Fehler gemeldet; (2) memcg-OOM des Builders bei 24 GiB in `gemm_sm120` (MAX_JOBS=4, anon 23,2 GiB), der OOM-Killer traf wegen oom_score_adj 500 **docker-init** (PID 1) statt `cicc` -> Builder neu gestartet, `failed to receive status: ... EOF`. Builder `htsglang-build` samt State-Volume entfernt, `ikbuilder` unberührt, kein Image entstanden. Fixes: Regex-Parser, Flag-Kontext je Modul wie am Rig (write_ninja-Probe: 52/52 gencode gleich), Vorbau-Fehler laut statt fatal (`PREBUILD_STRICT=0`), kein oom_score_adj am Init, Deckel 32g + MAX_JOBS=3; Kontext-Digest jetzt `9c296031…` |
| Host-Bau | GO Operator 25.09., bootfreies Fenster md55m8 (02:00:44–04:00:44Z). Erster Start 02:02:55Z verweigert: das Boot-Muster traf einen Agenten-Befehl in CT999 mit dem Pfad `python/sglang/srt/weg2/launcher.py` (`.` = beliebiges Zeichen) -- Muster geschärft (`sglang::schedule[r]`, `-m sglang\.srt\.weg2\.launche[r]`, `-m sglang\.launch_serve[r]`), auch in `host_acceptance.sh`. Zweiter Start 02:04:40Z abgekoppelt über `host_build_detached.sh` (Speicher-Sampler 5 s); Builder mit Deckel 24 GiB / 12 CPUs / shares 256 / oom_score_adj 500 bestätigt |

Belege: `L:` = `python/sglang/srt/weg2/launcher.py` der 27B-Linie (Worktree `/spinning/wt-27b-docker-0924`, Basis 5a3f533f0f). Rig-Lesungen „24.09.“ sind Datei-, `/proc`-, `docker`-, `zfs`- und NVML-Abfragen ohne CUDA-Kontext.

---

## 0. Kurzfassung

1. **Was aktualisiert wird:** Öffentlich ist `ghcr.io/efschu/htsglang:cu130-nccl2307` vom 14.07. Die #384-Probe vom 14.08. hat es als „SHADOWED“ markiert: zwei Distributionen liefern `sgl_kernel`. Der August-Stand ist `docker/htsglang.Dockerfile` auf `chore/release-chain-prep-r3 @ ab4a42d392`. Daraus stammen die Builds `htsglang:r2-99a4b0a4[-gated]` vom 14.08. mit Gate „verdict=ARMED“; gepusht wurden sie nie. Das Upgrade baut auf diesem Stand auf und erscheint unter einem **neuen versionierten Tag** (F14); den öffentlichen Tag ersetzt es erst mit Go des Nutzers (RELEASE_CHECKLIST §7.0).
2. **Machbarkeit:** Der Host kann es.
   - CT999 (LXC) hat Docker, aber keine NVIDIA-Runtime.
   - Der **Proxmox-Host** hat Docker 29.5.3 mit `nvidia`-Runtime (Toolkit 1.13.5).
   - Die BAR1-Kette ist dort vollständig: Treiber 595.58.03 open/smallbar, `RegistryDwords "RMSmallBarP2PPeerBar1=1;PeerMappingOverride=1"`, `dmabuf_holder` geladen, `/dev/dmabuf_holder` 0666, `resource1_wc` 666.
   - Im August lief dort BAR1 im Image (#369: 10/10).
   - **Engpass Host-RAM:** 125,7 GiB gesamt. Außerhalb CT999 liegen ~20 GiB (system.slice mit den Haus-Containern 9,5, SUnreclaim inkl. ZFS-ARC 8,2 bei ARC-Deckel 5, andere LXC 1,8; gelesen 24.09. ~22:55Z). Für CT999 + Container bleiben höchstens ~105 GiB; ein Container-Boot geht nur ohne CT999-Boot, mit Speicherdecke und OOM-Vorrang (F13-Schalter, §8).
3. **Stufe B umgesetzt, Commit `bae049a3b4`** (gepusht auf `origin/desk/27b-docker-0924`, pickbar für NF):
   - Die Rig-Pfade des weg2-Launchers kommen aus `SGLANG_WEG2_*`; ohne Variable bleibt alles byte-identisch.
   - Tests: 6/6 neu, betroffene Tests ohne neues Rot.
   - Probe-Cherry-Pick auf den NF-Tip `c286615929` ohne Konflikt.
4. **Transport:** barlink bar1 ist Default. `nccl` schaltet barlink **komplett** ab: Der Launcher streicht mit `--transport nccl` die barlink-Flags, der Entrypoint entfernt `SGLANG_BARLINK*`. Ein stilles Umschalten gibt es nicht. Hintergrund (NF §5.3): Mit eingeschaltetem barlink ohne BAR1 bricht D beim Graph-Capture ab.
   - **NCCL ist für beide Profile unbelegt.** Die einzige weg2-NCCL-Messung (weg2ab, 07.09., andere Form) ergab P-Prefill 2,06× langsamer und `tp.all_reduce` +26 %.
   - Kein `auto` (F7). Beide Transporte bootet die Abnahme.
5. **Toolchain des Stands:** torch läuft auf cu13 (2.11.0+cu130). Jeder JIT-Kern des Rigs ist dagegen mit System-nvcc **12.9.86** gebaut und linkt `libcudart.so.12`. Basis deshalb `nvidia/cuda:12.9.1-devel-ubuntu24.04` (F5 „ja“).
6. **NV-Header optional** (F6): `WITH_NV_HEADERS=0` ist Default. Ohne Header verweigert `bar1` mit Grund; alternativ werden die Header des laufenden Treibers gemountet.
7. **Modellformate** (Nutzer):
   - 27B: INT8 (heute), FP8, NVFP4 (RadixArk), GGUF.
   - NF: INT4-Mixed (heute), NVFP4 (nvidia), GGUF (unsloth).
   - Je Format ein Profil. Die Formaterkennung im Entrypoint ist am Rig gegen alle vorhandenen Checkpoints geprüft.
   - JIT-Vorbau je Format und Arch nach §7: K baut 27B-FP8, NF die NVFP4-Basis, GGUF folgt.
   - Downloads (F16, 22:50Z): 27B-FP8, 27B-NVFP4-RadixArk, 27B-GGUF (UD-IQ4_XS + UD-Q8_K_XL), NF-NVFP4 vollständig; NF-GGUF (UD-IQ4_XS, 3 Teile, MTP als Q8_0-GGUF) lädt.
8. **Aufräumen CT999 erledigt:** 15 ungenutzte Images und der Build-Cache (191 Einträge, 98,9 GB) sind weg; zoekt-web und sein Image blieben. Die Snapshots, die den Platz hielten, hat der Nutzer gelöscht (F11): `/` hat jetzt ~740G frei.
9. **27B-Release-Form RC2** (Operator 24.09.): Baum `desk/27b-release-rc2-0924 @ 4aded781ab` (Worktree `/spinning/wt-27b-line-rc2`, ungepusht), RC1-Form + Idle-Politik (`--idle-layout pp --d-hold-s 10 --d-short-drain-tokens 4096`) + fünf Schalter. `profiles/27b.env` ist gegen die Quelle `arm_rc2.sh` (Agent L, 23:00Z) gegatet: Env 34/34 gleich (nur der Pfad von `SGLANG_DFLASH_PHASE_TIMING` folgt Stufe B), die geparsten Launcher-Namespaces von Arm-Argv und Profil-Argv sind gleich (RC2-Parser); P-Trim ist in RC2 **an**. **Der RC2-Baum trägt bae049a3b4 nicht** — vor dem Kontext-Bau picken.

---

## 1. Welcher Container wird aktualisiert (Belege)

| Stand | Wo | Befund |
|---|---|---|
| **veröffentlicht** | `ghcr.io/efschu/htsglang:cu130-nccl2307`, gepusht 14.07., Digest `sha256:5ec17442…`, am 14.07. **öffentlich** gestellt (Transkript 14.07. 19:22) | CUDA 13.0.1, System-pip, NCCL 2.30.7. Probe 14.08.: `verdict=SHADOWED` (sgl-kernel 0.3.21 **und** sglang-kernel 0.4.4; der INT8-Arm ist da, aber instabil). Liegt auf dem Host (22,1 GB) |
| daneben | `ghcr.io/efschu/htsglang-qwen35-gguf:cu130` (15.07.) | GGUF-Variante, auf dem Host |
| **August-Stand** | `chore/release-chain-prep-r3 @ ab4a42d392`; Builds `htsglang:r2-99a4b0a4` und `-gated` (14.08., `/spinning/wt-release3`) | `INSTALL_SGL_KERNEL=0` als Default (pyproject pinnt `sglang-kernel==0.4.4`), Provenienz-Gate **unbedingt** (d2c58d3219, f0320af573), Checkliste §0.0/§4.0/§7.0/§7.1. Nie gepusht und nicht auf dem Host |
| 27B-Linie | `docker/` @ 5a3f533f0f = Stand 99a4b0a4 | Die vier r3-Commits für docker/ und Checkliste fehlen der Linie. Beim Release mit diesem Upgrade in `docker/` übernehmen |

**Pflicht aus dem August (§7.0):** Das Release-Image **ersetzt** `cu130-nccl2307`, entweder unter demselben Tag mit neuem Digest oder unter neuem Tag, dann wird der alte zurückgezogen. Danach folgt die Verifikation **per Digest** im #416-Muster: Pull als Fremder, Gate, `--print-argv-only`, NCCL-Banner. Erst danach wird angekündigt.

---

## 2. Rig-Befund (24.09., lesend)

| | CT999 (LXC, unprivilegiert) | Proxmox-Host `proxmox` |
|---|---|---|
| Docker | `docker.io 29.1.3`, overlayfs/containerd, cgroup v2 | **Docker 29.5.3**, Storage-Driver **zfs** (`spinning/docker`, 927 GiB belegt), cgroup v2 |
| NVIDIA-Runtime | **keine** (`Runtimes: runc`, kein Toolkit, kein CDI) | **`nvidia`** in `daemon.json`, Toolkit 1.13.5 (alt), `/etc/cdi/nvidia.yaml` vom 30.04. (veraltet, ungenutzt) |
| Treiber | 595.58.03 open (Userspace im LXC) | **595.58.03 open**, gebaut „root@proxmox“ 29.07., Host oben seit 05.08. 21:02 |
| BAR1-Kette | RegistryDwords ok, Holder 0666, `resource1_wc` 666 | **dieselbe**, `dmabuf_holder` geladen (18 Nutzer) |
| RAM | lxcfs-Sicht 118 GiB | **125 GiB gesamt**; beim laufenden CT999-Boot 9 GiB verfügbar |
| Platz | `/` 180 GiB frei (Pool `spinning`) | derselbe Pool: 180 GiB frei; Host-Wurzel `rpool` 59 GiB frei |
| memlock | hart 8 MiB (nicht hebbar, reicht) | root 8 MiB, `--ulimit memlock=-1` möglich |
| Werkzeuge | — | python3 3.12.8, curl, jq, `pct` |
| Sonstiges | IP 192.168.0.88, gpuq auf 0.0.0.0:8770 | Haus-Container laufen (Vaultwarden, Home Assistant, Nextcloud, ioBroker, …) → **nie** `docker system prune` o.ä. auf dem Host ohne Nutzer |

Die frühere Annahme aus dem ersten Entwurf ist damit überholt: Die Abnahme läuft auf dem **Host** (Nutzer F2), nicht verschachtelt in CT999.

---

## 3. Was barlink BAR1 vom Host braucht

| # | Voraussetzung | Host heute | Im Container |
|---|---|---|---|
| H1 | gepatchter open-Treiber 595.58.03 (smallbar) | ja | nicht lieferbar; Kernel-Update ⇒ Patch neu bauen (Memory `barlink-standard.md`) |
| H2 | `RMSmallBarP2PPeerBar1=1;PeerMappingOverride=1` | ja | Preflight prüft; mit PeerMappingOverride **kein** `CAP_SYS_ADMIN` (`barlink_bar1.py:3075-3100`) |
| H3 | `dmabuf_holder`, `/dev/dmabuf_holder` 10:262 0666 | ja | `--device /dev/dmabuf_holder` |
| H4 | `resource1_wc` beschreibbar | 666 | `-v /sys:/sys` (Docker mountet `/sys` sonst ro), `apparmor=unconfined` |
| H5 | BAR1 5090 32 GiB, 3080 je 256 MiB | ja | Fenster löst der Launcher |
| H6 | `iommu=pt`, ACS-Override | ja (Kernel-cmdline) | — |
| H7 | NV-Header des laufenden Treibers (dma-buf-Ext, UAPI versionsgebunden) | `/spinning/nvidia-open-595` (2 Header gepatcht) | **optional im Image** (`WITH_NV_HEADERS=1`) oder `-v …:/opt/nvidia-open-595:ro` |
| H8 | alle Ränge in einem Container (SCM_RIGHTS, CUDA-IPC, gemeinsames shm) | — | ein Container je Profil |

---

## 4. NCCL als zweiter Transport

- **Wahl:** `launcher --transport {bar1,nccl}` (L:11241, Default bar1). Unter `nccl` streicht der Launcher die barlink-Flags beider Gruppen (L:399-451) und `SGLANG_BARLINK_BUILD_WINDOW_CAP_S`, und die D-Ruhereserve steigt 64 → 192 MiB (L:252, L:415-426). Ohne `--barlink` baut sglang PyNccl (`parallel_state.py:1030-1043`).
- **Kein NCCL-Rückfall innerhalb von barlink:** Scheitert bar1, fällt die Gruppe auf gloo. Unter CUDA-Graphen bricht D dann ab (NF §5.3). Daraus folgt die Entrypoint-Regel:
  - `bar1` nur mit vollständiger Kette.
  - `nccl` = barlink ganz aus. Der Entrypoint entfernt zusätzlich `SGLANG_BARLINK`, `SGLANG_BARLINK_TRANSPORT` und `SGLANG_BARLINK_PP_TRANSPORT`, damit keine Rest-Variable barlink ohne Flags doch baut.
  - Kein `auto` (F7): `HTSGLANG_TRANSPORT=auto` verweigert mit Grund.
- **GeForce ohne NVLink:** Kein P2P (NS/CNS), NCCL nimmt selbst den SHM-Transport. Nötig sind genug privates `/dev/shm` und alle Ränge in einem Container. Nicht ausliefern: `NCCL_P2P_DISABLE` (Klasse [RIG]). Profil-Env für beide Transporte: `NCCL_BUFFSIZE=1048576 NCCL_MAX_NCHANNELS=8`.
- **Auch unter bar1 lebt NCCL:** PP-send/recv zwischen den P-Stufen (NF x163).
- **Leistungsfolgen** (`BOOT_weg2ab_0907.md`, 27B-Linie, andere Form):
  - D-Prefill −25 %
  - `tp.all_reduce` +26 % gegenüber bar1
  - `dcp.all_reduce` +50 %
  - P-Leg-1-Prefill 2,06× langsamer
  - Flip D→P +12 %
  - +80 MiB Ruhe-Residuum
- **#599** (NCCL-Tuning, „unmittelbar vor Docker-Release“): nicht besetzt (F9).

### F10: NCCL-Version der Release-Linie (Beleg)

Die Linie lädt am Rig **NCCL 2.28.9+cuda13.0** aus `nvidia-nccl-cu13==2.28.9`:
- **Laufende Ränge:** Die Scheduler-Prozesse eines laufenden Boots (24.09., pids 3890777–79, `sglang::scheduler_TP0..2`) mappen `/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2` (`/proc/<pid>/maps`); ebenso am Vormittag `scheduler_TP0` pid 3681788.
- **Die Datei:** Banner `NCCL version 2.28.9+cuda13.0`, sha256 `1c8618b866734cbdd5401715d6178be763ece283b7f808ecf86dedab211162c1`, 217 995 896 Byte, mtime 2026-07-13 08:52:14.
- **Schatten wie #384:** Zwei RECORDs beanspruchen dieselbe Datei. `nvidia_nccl_cu13-2.28.9` nennt `sha256=HIYYuGZzTL3VQBcV1heL52Ps4oO3-Ajs-G3tqyERYsE,217995896` = der Inhalt auf der Platte; `nvidia_nccl_cu12-2.29.7` nennt `sha256=GdmFG2X-…,400842456` = nicht der Inhalt. cu13 gewann, weil es zuletzt installiert wurde. cu12 bringt zusätzlich 9 Device-Header/`libnccl_device.bc`, die cu13 nicht hat; kein Paket verlangt cu12 (`Requires-Dist`), torch verlangt cu13.
- **Log:** Der einzige weg2-Boot auf dem NCCL-Transport (`weg2ab0b`, 07.09., D-Log Z. 54–55) meldet `sglang is using nccl==2.28.9`. Unter bar1 schreibt sglang diese Zeile nicht.
- **Image:** Lock mit beiden Paketen wie am Rig, danach `pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.28.9` (Gewinner festgelegt statt Installationsreihenfolge), Build-Abbruch bei anderem Banner oder anderem sha256 (`NCCL_SHA256` aus `BUILD_INFO.json .nccl`, von `prepare_context.sh` am Referenz-venv gemessen). Nie `pip uninstall nvidia-nccl-cu12` im Image: es löschte die geteilte Datei mit.
- **Folge (F10b):** Duplikate in `--rank-gpu-id` verlangen NCCL ≥ 2.30 (rig-runbook §4.2, §6.2); mit 2.28.9 verweigert der Launch-Pfad die Co-Location. Das veröffentlichte Image pinnte 2.30.7 genau dafür. Siehe §15.

---

## 5. Ziel und Umfang

Ein Image je Linie (Memory `27b-strikt-getrennt-von-nf.md`), darin mehrere **Profile**, eines je Modellformat. Jedes Profil läuft mit `bar1` **und** `nccl`.

| Profil | Modell (Pfad am Rig, read-only unter demselben Pfad gemountet) | Format (erkannt) | Form geliefert von | Stand |
|---|---|---|---|---|
| `27b` | `Qwen3.8-27B-INT8-gdncov-vocabembed` + DFlash2-W8 | `int8` | 27B-Sitz (Form = `arm_rc2.sh`, gegatet) | Profil fertig (RC2) |
| `27b-fp8` | `Qwen3.8-27B-FP8` (29 GB, fp8) | `fp8` | **Agent K** | Platzhalter |
| `27b-nvfp4` | `Qwen3.8-27B-NVFP4-RadixArk` (21 GiB, modelopt MIXED_PRECISION float4+float8) | `nvfp4-modelopt` | 27B-Sitz nach der NVFP4-Basis | Platzhalter |
| `27b-gguf` | `Qwen3.8-27B-GGUF-unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf` (oder `UD-Q8_K_XL`) | `gguf` | offen | Platzhalter, Modell da |
| `nf` | `Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist` + MTP-INT4 | `int4-mixed` | NF-Sitz (x163) | Profil fertig (NF §11.1) |
| `nf-nvfp4` | `Qwen3.8-Flash-Next-NVFP4-nvidia` (124 GiB, modelopt float4+float8) | `nvfp4-modelopt` | **NF-Sitz** (NVFP4-Basis) | Platzhalter |
| `nf-gguf` | `Qwen3.8-Flash-Next-GGUF-unsloth/UD-IQ4_XS` (+ `MTP/…Q8_0.gguf`) | `gguf` | offen | Platzhalter, Download läuft |

**Nicht-Ziele dieser Runde:** Portabilität auf fremde Hardware; beide Linien sind Rig-Profile: `--user-reserve-mib`, Host-Riegel, Census und W19-Kartenmodelle sind Messungen dieses Rigs. Ebenfalls nicht: Multi-Node, Veröffentlichung ohne Go.

---

## 6. Image (Upgrade von `docker/htsglang.Dockerfile`)

| Punkt | August (r3) | Upgrade |
|---|---|---|
| Basis | `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` | `nvidia/cuda:12.9.1-devel-ubuntu24.04` (Digest beim Pull festschreiben) |
| Python | System-pip | venv `/opt/venv` (der Launcher braucht `{venv}/lib/python3.12/site-packages/nvidia/cu13`) |
| Pakete | Constraints + Einzelpins | exakt das Referenz-venv (`pip freeze`, 314 Pakete, 15 cu12-Reste bleiben: JIT-Kerne linken `libcudart.so.12`) |
| sgl-kernel | Gate unbedingt, Fork-Wheel optional | Fork-Wheel `sglang_kernel-0.4.4` (sha256 `67f03cfa…`) **Pflicht**, `REQUIRE_INT8_ARM=1`, Gate bleibt |
| NCCL | 2.30.7 erzwungen (Co-Location) | **2.28.9+cuda13.0** = was die Linie am Rig lädt (F10, §4): cu13 zuletzt erzwungen, Banner + sha256 geprüft; `NCCL_PIN=2.30.7` (+`NCCL_BANNER`, `NCCL_SHA256`) stellt den August-Stand her (F10b) |
| Modi | `MODE=server|planner` | unverändert (August-Entrypoint), **neu** `MODE=weg2` |
| barlink BAR1 | „Image ohne BAR1“ (§2.2) | NV-Header **optional** (`WITH_NV_HEADERS`, Default 0) |
| JIT | Laufzeit | Vorbau im Build + tvm-ffi-Saat (§7) |
| Rig-Pfade | — | Stufe B: `SGLANG_WEG2_EVIDENCE_DIR=/var/lib/htsglang/evidence`, `…_GPU_ARB=/var/lib/htsglang/arb`, `…_DEVTOOLS_DIR=/opt/htsglang/devtools`, `…_STORE_ROOT=/var/lib/htsglang/hicache-weg2`, `…_VENV=/opt/venv`, `…_TMS_OUT_DIR=/opt/htsglang/tms` |
| ENV | `SGLANG_BARLINK_LAUNCH_DUMP=0`, `TORCH_CUDA_ARCH_LIST` 7 Archs, Planner-Pfade | unverändert; `MODE=weg2` entfernt `TORCH_CUDA_ARCH_LIST` (sonst passen die vorgebauten barlink-Namen nicht) |

**Kontext** (`prepare_context.sh <linie> <sha> <branch> [--with-nv-headers] [--no-tvm-ffi-seed]`):
- flacher git-Klon mit Historie, weil der Launcher stempelt und die Kalibrier-Identität per `git rev-list` prüft
- Lock, Kernel-Wheel
- optional die NV-Header
- Rig-Werkzeuge (3 devtools)
- ARB-Saat (`PROBE_RING_0907.md`, `calib/*.json`, `corridor_budget_sample*.json`)
- Profil-Daten (Census, NF-Korridor-Sample)
- FlashInfer-Modulliste **mit gencode je Modul**
- tvm-ffi-Saat
- Geheimnis-Scan

Verlangt ist, dass der Commit bae049a3b4 enthält. **Gebaut wird auf dem Host** aus dem LXC-Pfad (`/spinning/subvol-999-disk-0/spinning/gpu-arb/docker/ctx/…`). So entsteht keine Transfer-Kopie, das Image liegt einmal im Pool.

---

## 7. JIT-Vorbau je Format und Architektur

**Quelle der Wahrheit ist der Rig-Cache nach einem nativen Boot des Formats.** Erst bootet ein Format am Rig (K: 27B-FP8, NF: NVFP4), danach wird der Kontext neu erzeugt, und das Image trägt dessen Kerne.

| Komponente | Cache | Pfadabhängig? | Vorbau |
|---|---|---|---|
| FlashInfer 0.6.14 | `~/.cache/flashinfer/0.6.14/{120f,86}` | ja (build.ninja) | **im Build**, je Verzeichnis eigener Prozess in Server-Reihenfolge. Vor dem Import `12.0` bzw. `8.6` (Verzeichnis `120f`/`86`), danach `12.0a` bzw. `8.6`, gespiegelt aus `set_cuda_arch`, `model_runner.py:2494` → `utils/common.py:1547`. **Gemessen:** `120f` trägt ZWEI Flag-Sorten. `compute_120a` haben norm, sampling, topk, prefill und decode (frischer Kontext). `compute_120f` haben `gemm_sm120` (Import-Kontext `flashinfer/jit/core.py:138`) und `fp4_quantization_120f`. Der Vorbau erzeugt beides von selbst und prüft je Modul gegen die gencode-Menge der Rig-build.ninja |
| barlink | `~/.cache/torch_extensions/py312_cu130/barlink_*_cuda_86_120` | ja | **im Build**, gleicher Name und gleiche Flags über die Gruppen-Union 8.6+12.0 (`barlink_device.py:681-712`); dma-buf-Ext nur mit NV-Headern |
| TMS-Preload | `/opt/htsglang/tms/…_<sha12>.so` | nein | **im Build** (`build_tms_preload.sh`, Ziel über `SGLANG_WEG2_TMS_OUT_DIR`) |
| sglang jit_kernel (tvm-ffi): Marlin FP8/FP4/INT4, HiCache, Aktivierungen … | `~/.cache/tvm-ffi` | **nein**: inhaltsadressiert, Provenienz source/build-Hash + Arch, ohne Include-Pfade (`jit_kernel/utils.py:196-345, 600-700`) | **Saat aus dem Rig** (192 vollständige Einträge mit Provenienz, 338 MB). Was zum Baum passt, wird geladen |
| Triton | `~/.triton` | nein (inhaltsadressiert) | Saat per Bind-Verzeichnis beim Host-Lauf (6,4 GB), nicht im Image |
| DeepGEMM | — | — | auf sm_86/sm_120 aus |

| Format | sm_120 (5090) | sm_86 (3080) |
|---|---|---|
| INT8-W8A8 (27B heute) | sgl-kernel `int8_scaled_mm` (Fork-Wheel, Arm Pflicht) | wie links (sm_86-Cubins im Wheel) |
| FP8 (27B-FP8) | CUTLASS FP8: FlashInfer `gemm_sm120` bzw. sgl-kernel `fp8_scaled_mm` | kein FP8-Tensorkern → **Marlin-FP8** (W8A16) über tvm-ffi |
| NVFP4 (27B-NVFP4, NF-NVFP4) | CUTLASS-FP4 (`gemm_sm120`, `fp4_quantization_120f`) | kein FP4 → **Marlin-FP4** (Dequant) über tvm-ffi |
| INT4-Mixed (NF heute) | Marlin/MoE-WNA16 (tvm-ffi `moe_wna16_marlin`, `gptq_marlin`) | wie links |
| GGUF | k-quant-Kerne des Forks | wie links |

Die Formaterkennung im Entrypoint wurde am Rig gegen alle vorhandenen Checkpoints geprüft:

| Checkpoint | erkannt |
|---|---|
| 27B-INT8 (gdncov-vocabembed) und DFlash2-W8-Draft | `int8` |
| 27B-FP8 | `fp8` |
| 27B-NVFP4-RadixArk, NF-NVFP4-nvidia (modelopt, `hf_quant_config.json`) | `nvfp4-modelopt` |
| 27B-NVFP4 (unsloth, compressed-tensors) | `nvfp4-ct` (anderer Lader, nicht das bestellte RadixArk) |
| NF-INT4-Mixed (int4/6/8) | `int4-mixed` |
| NF-MTP-INT4-Draft | `int4` |
| 27B-GGUF-unsloth (Verzeichnis **mit** `config.json`) und eine `.gguf`-Datei daraus | `gguf` (vorher fälschlich `bf16`, korrigiert) |
| NF-GGUF-unsloth (gguf nur in Unterordnern) | `gguf` (vorher Absturz ohne `config.json`, korrigiert) |

---

### 7.1 Builder-Cache und Schichtfolge (UMGESETZT 25.09. ~04:30Z nach Operator-GO; RC4-Kontext `c19a6bd7…`)

**Was bei RC4 mit der heutigen Schichtfolge aus dem Cache kommt** (Builder `htsglang-build`, 47 GB Cache aus dem RC2-final-Bau):

| Schritt | RC4 | Warum |
|---|---|---|
| 1–10: Basis, apt, UCX, rustup, venv, `COPY lock/`, pip-Lock, NCCL, sgl-kernel-Wheel | **Cache** (~6 min gespart) | Dockerfile-Text davor unverändert; Lock byte-gleich (25.09. 03:4xZ gegen den RC2-final-Lock gedifft); Build-Args gleich |
| 11 `COPY src/` und alles danach | **neu** | neue SHA ⇒ neuer Inhalt; zusätzlich ändert `SGLANG_SCM_VERSION` (Build-Arg, enthält die SHA) Schritt 13 |
| 26 JIT-Vorbau (38,7 min) | **neu** | hängt hinter `COPY src/`, dazu hinter `COPY tools/profiles/` (19) und `COPY assets/tvm-ffi/` (22: 192 → 194 Einträge seit dem NVFP4-Boot) und `COPY assets/flashinfer_modules.txt` (25: die Datei trägt einen **Zeitstempel im Kopf** und ändert sich darum bei JEDER Kontext-Erzeugung) |
| 6b, 7, Export | neu | folgen |

⇒ RC4 kostet mit der heutigen Folge wieder ~50 min (sglang 0,8 + Vorbau 38,7 + 6b ~1 + Export 8,3). Schlimmer: schon das Umschalten eines Profils von „vorbereitet“ auf „abgenommen“ (`--refresh-tools`) invalidiert über Schritt 19 den ganzen Vorbau.

**Plan: FlashInfer-Vorbau vor den Quellcode ziehen.** Der teure Teil (FlashInfer 86 + 120f, ~36 der 38,7 min) braucht nur das venv (flashinfer, torch aus dem Lock), die CUDA-Toolchain der Basis, `prebuild_jit.py` und die Modulliste -- **kein** `src/`: `fi_child` importiert nur flashinfer, der Flag-Kontext (Import- vs. Server-Kontext) wird in `prebuild_jit.py` selbst nachgebildet (52/52 gencode gleich Rig, write_ninja-Probe 25.09.). Nur barlink, hicache-Hash und TMS-Hook (~3–4 min) brauchen Quellen aus `src/`. Neue Folge:

1. Basis, apt, rustup, venv, Lock, NCCL, sgl-kernel (wie heute)
2. `COPY tools/prebuild_jit.py` + `COPY assets/flashinfer_modules.txt` → `RUN prebuild_jit.py --only flashinfer` (Bericht nach `JIT_PREBUILD.json`)
3. `COPY src/` → Baum-Prüfung → `pip install` sglang (SCM-Version)
4. NV-Header (optional), dann die Rig-Links (6b) → `RUN prebuild_jit.py --only barlink,cpu_ext,tms` (ergänzt den vorhandenen Bericht statt ihn zu ersetzen)
5. `COPY` devtools, arb-seed, profiles, tvm-ffi-Saat, entrypoint, healthcheck, BUILD_INFO; ENV/LABEL

Dazu nötig: (a) **kein Zeitstempel in `flashinfer_modules.txt`** (Stand der Liste gehört nach `BUILD_INFO.json`), sonst trifft Schritt 2 nie den Cache; (b) `prebuild_jit.py` führt einen vorhandenen Bericht fort (Abschnitte ergänzen, verdict neu rechnen); (c) die tvm-ffi-Saat kommt NACH dem Vorbau (sie wird zur Laufzeit gelesen, nicht im Vorbau).

**Umgesetzt (25.09. ~04:30Z)**, mit einem Befund, den der Plan noch nicht hatte: `ARG HTSGLANG_REVISION` und `ARG SGLANG_SCM_VERSION` standen direkt hinter `FROM`. Jede ARG-Deklaration geht als Umgebung in alle folgenden RUN-Schritte und damit in deren Cache-Schlüssel ein -- eine neue SHA hätte also schon apt, pip und den Vorbau invalidiert, und die Schätzung „Schritte 1–10 aus dem Cache“ für RC4 wäre ohne Umbau falsch gewesen. Beide ARGs stehen jetzt erst hinter `COPY src/`. Prüfungen am Desk:
- statische Reihenfolge-Prüfung des Dockerfiles (12/12): FlashInfer-RUN vor `COPY src/` und ohne `src/`-Bezug, keine SHA-ARGs davor, Teil B nach src/ und NV-Headern mit den Rig-Links vor dem TMS-Bau und `--continue-report`, Profile/tvm-ffi/BUILD_INFO dahinter;
- 52/52 gencode gleich Rig über genau den Teil-A-Pfad (Kindprozess je Arch, cwd `/`, leerer PYTHONPATH, sglang nicht importiert);
- Berichts-Fortführung mit Attrappen (6/6): A+B grün ⇒ OK; TMS rot ⇒ INCOMPLETE mit `tms: rc=1`; B ohne A ⇒ fehlender Abschnitt benannt; Fehler aus A bleibt über B stehen; `--strict` bewertet nur den eigenen Teil;
- rote Selbstprüfung: ohne TMS-Hook rc=1 mit `SELFCHECK FEHLER`, mit Hook rc=0;
- Modulliste deterministisch (zwei Läufe byte-gleich), Stand in `BUILD_INFO.json .jit_snapshot`.

Wirkung (Schätzung aus den Phasenzeiten vom 25.09.): neue SHA ⇒ ~5 min Rechnen (sglang 0,8, barlink/hicache/TMS ~3, Rest Sekunden) + Export ~8 min ≈ **~15 min statt ~50**; nur Profile/Saaten geändert ⇒ ~1 min + Export ≈ **~9 min**. Der FlashInfer-Vorbau läuft nur noch neu, wenn sich Lock, Modulliste, `prebuild_jit.py` oder `MAX_JOBS` ändern -- genau dann, wenn er muss. Der Export (`--load` schiebt das ganze 30-GB-Image als Tarball in den Image-Store) bleibt der feste Sockel; der docker-Treiber würde ihn sparen, hat aber keinen Speicher-Deckel und fällt darum aus.

## 8. Laufzeit auf dem Host

Die Laufzeile steht in `host_acceptance.sh`, Funktion `run_args`.

- **GPU und Geräte:** `--gpus all`, `--device /dev/dmabuf_holder`, `-v /sys:/sys`, `--security-opt apparmor=unconfined`. **Kein** `CAP_SYS_ADMIN`, kein `--privileged`.
- **Speicher:** `--shm-size`: 27B 48g (Spitze 34,0–34,2 GiB), NF 16g (≤ 8,7 GiB). **Kein `--ipc=host`**, sonst sähe der #1217-Sweep fremde Halter. NF bekommt zusätzlich `--mount type=tmpfs,dst=/mnt/nf-experts,tmpfs-size=72g`.
- **Haus-Schutz (F13, Schalter `HOUSE_GUARD`, Pflicht für check/t0/serve/release/negative):**
  - `memlimit`: CT999 läuft weiter (Router 30099, gpuq, Agenten), aber ohne Boot; Container `--memory 104g --memory-swap 104g --oom-score-adj 500`. Wächst CT999 während des Boots, stirbt der Abnahme-Boot, nicht das Haus.
  - `ct999-ruht`: CT999 gestoppt oder eingefroren — **das tut der Nutzer selbst**, nie das Skript und nie ein Agent (Router, gpuq und alle Agenten leben in CT999). Container-Decke = CT999-LXC-Grenze aus `pct config` (120880 MiB, gleiche Decke wie die nativen Referenzboots), `--oom-score-adj 500`; gpuq ruht mit, bindend ist „Karten leer“.
  - Beide: Host-MemAvailable ≥ `MEMAVAIL_MIN_GIB` (Spitze + Luft: 27B 80 nach RC2-final 70,1/71,7 GiB, NF 90). Die frühere Bedingung „≥ 105 GiB“ war nie erfüllbar (außerhalb CT999 liegen ~20 GiB) und ist ersetzt.
  - Spitzen: 27B nativ 89,9–91,4 GiB, NF 83,7–85,1 GiB nonreclaim.
- **Zustand** (Bind-Verzeichnisse `/spinning/docker-acceptance/<linie>/…` im LXC, für logindex lesbar):
  - `evidence` (Boot-Logs und Kalibrierquellen)
  - `arb` (State, Admin-Key, Deadman-Ausgaben; die ARB-Saat ergänzt nur fehlende Dateien)
  - `store` (HiCache-L3)
  - `sglang` (card_probe, phase_footprint …)
  - `triton`
  - JIT-Caches als **benannte** Volumes je Image-Tag: Docker befüllt sie beim ersten Mount aus dem Image
  - **Getrennt von den nativen Rig-Verzeichnissen**, sonst würden Container-Boots Kalibrierquellen nativer Boots und umgekehrt
- **Saat für die Parität:**
  - die drei Logs des nativen Referenzboots derselben Linie (`REF_TAG`)
  - `weg2_measured_record.json`
  - `~/.cache/sglang/*.json`
  - Triton-Einträge
- **Modelle:** read-only unter **demselben** Pfad (`/spinning/llm_stuff/club-3090/models-cache`, das ganze Verzeichnis, wegen der absoluten Draft-Symlinks von NF).
- **Ports:** Front `0.0.0.0:30030` im Container, auf dem Host nur auf `127.0.0.1:31030` veröffentlicht (Proben laufen auf dem Host; die Front hat keine Authentifizierung). Gruppen 30031/30032 bleiben intern. Router 30099 liegt in CT999 und wird nicht berührt.
- **Geheimnisse:**
  - Kein Token im Image, der Admin-Key wird je Boot neu erzeugt.
  - `GPUQ_*` nie in den Container, weil `build_env` die ganze Umgebung an alle Ränge vererbt.
  - PAT-Dateien werden weder gelesen noch kopiert. Der Git-Push läuft über den konfigurierten Credential-Helper.

---

## 9. Einstiegspunkt (`entrypoint.sh`)

- **Dispatcher:** Ist `MODE` ungleich `weg2`, wird exakt das August-Entrypoint gestartet. Bestehende `docker run`-Zeilen bleiben gültig, der Test `test_entrypoint_empty_env_384.py` bleibt maßgeblich.
- **weg2-Untermodi:** `serve`, `dryrun`, `preflight`, `selfcheck`, `version`.
- **Profile:**
  - `HTSGLANG_PROFILE`, Default = Linie.
  - Die Profil-Linie muss der Image-Linie entsprechen.
  - Platzhalter verweigern mit Nennung des Eigentümers.
  - Form-Werte werden über `${X:-Bestform}` gesetzt, jede Abweichung wird laut gemeldet.
- **Formaterkennung:**
  - `config.json` bzw. `*.gguf` gegen `PROFILE_FORMAT`.
  - Ein Mismatch führt zur Verweigerung; `HTSGLANG_FORMAT_CHECK=0` schaltet die Prüfung ab.
- **Umgebung:**
  - `FLASHINFER_CUDA_ARCH_LIST` und `TORCH_CUDA_ARCH_LIST` werden entfernt.
  - `expandable_segments` wird verweigert (L:5515).
- **Preflight:**
  - Karteninventar über NVML gegen das Profil.
  - `/dev/shm`, MemAvailable und cgroup-Limit.
  - Baum: HEAD entspricht der Image-Revision und ist sauber.
  - Modell und Draft vorhanden.
  - Zustands-Dirs beschreibbar, ARB-Saat eingespielt.
  - NF-Store ist tmpfs.
  - BAR1-Kette: Regkeys, CapEff, Holder, `resource1_wc`, Header, und Treiber == Header-Stand, wenn die Header aus dem Image kommen.
- **Transport:** wie §4, Default `bar1`, `nccl` nur ausdrücklich, kein `auto` (F7).
- **Instrumente (F8):** `HTSGLANG_INSTRUMENTS` Default 0, vor dem Profil exportiert; 1 schaltet `profile_instr_env` (27B) bzw. `NF_ENV_*_INSTR` (NF) zu.
- **P-Trim (27B):** `HTSGLANG_P_TRIM=0|1` hängt `--p-trim-end-anchor` an; der Launcher setzt `SGLANG_WEG2_P_TRIM_END_ANCHOR=1` nur für Gruppe P. Abweichung von der Bestform wird laut gemeldet.
- **Lebenszyklus:**
  - Der Launcher kehrt nach `LAUNCHED` zurück (L:14068-14069). Der Entrypoint wartet signalfest, beaufsichtigt danach die Front und baut über `--teardown <state.json>` ab (L:12362-12367, 14307).
  - Laufzeit-Artefakte gehen nach `evidence/docker_<tag>/`, der Admin-Key nie.
  - `docker stop -t 180`.

---

## 10. Health / Readiness

| Zweck | Endpunkt | Semantik |
|---|---|---|
| Liveness (`healthcheck.sh`) | weg2: `:30030/health`; sonst `:${PORT:-30000}/health` | Die Front antwortet 200 nur, wenn beide Gruppen 200 liefern und `state != STOP` (`front.py:2756-2767`) |
| Readiness | `:30030/weg2/state` → `"state": "serving"` | wie die Arme |
| nicht als Healthcheck | `/health_generate` | 503 während eines Flips |
| Referenz nativ | 27B xsn430: P READY 66 s, D 72 s, Front +3 s; NF 296–331 s | |

---

## 11. Versionierung

- **Image-Tags (F14):** neuer versionierter Tag je Linie und Stand, lokal `htsglang:cu129-weg2-27b-rc2-<sha10>` bzw. `…-nf-<release>-<sha10>`, bei Freigabe `ghcr.io/efschu/htsglang:<derselbe>`. `cu130-nccl2307` wird erst mit Go des Nutzers ersetzt oder zurückgezogen; bis dahin bleibt es unberührt (auch lokal, F12).
- **OCI-Labels:** revision/source/line/driver.expected/flashinfer.
- **Im Image:** `BUILD_INFO.json` (Revision, Branch, Lock-sha256, Wheel-sha256, NV-Header ja/nein und Herkunft, tvm-ffi-Saat, Abstammung) und `JIT_PREBUILD.json`. Verifikation per **Digest**.
- **27B-Pin:**
  - Release-Baum RC2-final: `desk/27b-release-rc2-0924 @ b5c7d01614` (gepusht) = RC2 `4aded781ab` + Q-Review-Fixes + Stufe B als Pick `68bc631d8c` + K FP8 (`a9aedcd1fd`, Prewarm `b5c7d01614`) + L B/C + Q Blocker A.
  - Vor dem Build wird der Release-Commit eingefroren und gepusht, mit Fetch-Beleg, efschu ohne Trailer.
  - Die r3-Docker-Commits werden in `docker/` übernommen.
- **NF-Pin:** Kandidat `ce1dac1984` (x163) laut NF_PROFILE §11.3 N1. Der NF-Tip steht heute auf `c286615929`, bae049a3b4 muss gepickt werden.
- **Veröffentlichung:** nur auf ausdrückliches Go je Stand; bis dahin kein Push von Images, keine Registry.

---

## 12. Testplan (Skript `host_acceptance.sh`, Proben `probes.py`)

Alle GPU-Schritte laufen im gpuq-Fenster (bei `HOUSE_GUARD=ct999-ruht` gehören die Karten dem Nutzer): alle drei Karten, Holder und Herzschlag, der Herzschlag wird vor der Freigabe gestoppt, nach jedem Lauf wird freigegeben. Vergleichsbasis ist ein **nativer Boot derselben SHA** im selben Fenster. `serve` läuft mit `HTSGLANG_INSTRUMENTS=1` (Parität zu den nativen Boots), `release` mit 0 (was veröffentlicht wird).

| # | Schritt | Bestanden |
|---|---|---|
| check | nur lesen: Treiber, BAR1-Kette, Karten leer, Haus-Schutz nach `HOUSE_GUARD`, MemAvailable, Platz, Fenster | alles grün oder benannte Warnung |
| build + D1 | `host_build.sh` (EXPECT_MANIFEST, Deckel, Vorbedingungen), F12-Liste (Löschen nur mit `PRUNE_HOST_IMAGES=1`), Build unter neuem Tag, `selfcheck`, `version` | Build-Log: `bundled libnccl … NCCL version 2.28.9+cuda13.0 sha256=1c8618b8…`; torch 2.11.0/cu13, `torch.cuda.nccl.version()` (2,28,9), FlashInfer 120f/86 vorgebaut, `*_cuda_86_120`, TMS, tvm-ffi-Saat, Gate ARMED, Baum sauber, `JIT_PREBUILD verdict=OK` |
| seed | Bind-Verzeichnisse und Saat | Referenz-Logs und Zustand vorhanden |
| D2 | `dryrun` je Transport gegen den nativen Trockenlauf | P/D-argv gleich bis auf Pfade/Tag, `WEG2-P-FORM key=` gleich, keine neue W-Zeile. **Host-Ledger prüfen:** REAP MODEL mit Host-MemTotal 125 GiB und cgroup `memory.max` 104g statt LXC-Werten (R-2) |
| T0 | `preflight` und `bar1_graph_check.py 0,1,2` | BAR1-Kette vollständig, 10/10 (Referenz #369) |
| serve bar1 | Boot, Readiness, `gen`, `needle` (~100k, Nadel 10 %), `flip`, `gen`, `flip`, `gen`, `decode` code/prosa/thinking/code @10k 1024 Tok | `ACHIEVED=bar1` je Gruppe; Front-Zeile `WEG2-IDLE-POLICY idle_layout=P d_short_drain_tokens=4096 … d_hold_s=10.0` (27B); GEN ok; Nadel MATCH; Flips kehren nach serving zurück; Flipzeit (P-Ende → erstes D-Token), Decode und Boot-Zeit gegen nativ benannt; keine neuen `.so` im Cache-Volume (warm); Teardown: 0 MiB |
| serve nccl | wie oben | keine barlink-Gruppe, `sglang is using nccl==2.28.9`, W19 feuert nicht, Zahlen gegen bar1 **benannt** (Korrektheit Pflicht, Tempo Befund) |
| release | Boot bar1 mit `HTSGLANG_INSTRUMENTS=0`, `gen`, `needle` | wie serve bar1 für GEN/Nadel; keine Instrument-Dateien im Evidenz-Volume |
| negative | `bar1` ohne Holder | REFUSED, rc 3 |
| je Format | nach Lieferung der Form (K, NF): dieselbe Matrix mit dem Format-Profil | wie oben; dazu Formaterkennung = PROFILE_FORMAT |

---

## 13. Risiken

| # | Risiko | Gegenmittel |
|---|---|---|
| R-1 | **Host-RAM 125,7 GiB** geteilt mit Haus-Diensten (~20 GiB außerhalb CT999) und CT999; 27B RC2-final braucht ~72 GiB nonreclaim (INT8 70,1 / FP8 71,7, CT999-Sicht), ältere Formen bis ~91 | F13-Schalter (`memlimit` / `ct999-ruht`), MemAvailable ≥ Spitze + 5 GiB, Decke + `--oom-score-adj 500` |
| R-2 | **Host-Ledger im Container:** `/proc/meminfo` ist der Host (125 GiB), die Wasserlinie 95,90 GiB und andere Konstanten sind LXC-Messungen, `memory.max` ist jetzt endlich | D2 auf dem Host liest die Ledger-Zeilen; bei Abweichung Kalibrierung für den Host (Code, nicht Docker) |
| R-3 | Toolchain 12.9 (Stand) gegen 13.0 | 12.9 für Release 1 (F5) |
| R-4 | JIT-Fehltreffer (Reihenfolge, Env, venv-Pfad, ninja, gencode-Mischung in 120f) | Vorbau nach §7 mit gencode-Prüfung je Modul; serve misst neue `.so` |
| R-5 | Kalibrierquellen: git-Historie, Evidenz-Logs, calib/census | flacher Klon, ARB-Saat, Referenz-Logs; D2 leer gegen geseedet |
| R-6 | Toolkit 1.13.5 alt, CDI-Spec veraltet | #369 lief mit genau diesem Toolkit und Treiber; T0 bestätigt |
| R-7 | NV-Header/Treiber-Bindung (Kernel-Update legt nvidia lahm) | Header optional; Entrypoint vergleicht den Treiber |
| R-8 | NCCL unbelegt für beide Profile; barlink ohne BAR1 bricht D ab | Transport-Regel §4; serve nccl |
| R-9 | Platz | geschlossen: Snapshots gelöscht (F11), `/` ~740G frei; Host-Images nur beim Bau (F12) |
| R-10 | Linie bewegt sich stündlich → Profil-Drift | Pin einfrieren, Profil aus dem Arm ableiten, D2 als Diff-Gate |
| R-11 | Container-Boots als Kalibrierquelle nativer Boots | getrennte Bind-Verzeichnisse |
| R-12 | Haus-Container auf dem Host | Skript fasst nur `htsglang-acc-*` an, kein Prune |
| R-13 | GGUF-Modelle fehlen | Download (Nutzer/NF), dann Profil |
| R-14 | System-`libnccl` der devel-Basis | `libtorch_cuda.so` hat DT_RPATH auf das venv-NCCL; D1 prüft (2,28,9) |
| R-15 | **Co-Location-Rückschritt:** das Upgrade mit 2.28.9 verweigert Duplikate in `--rank-gpu-id`, die das öffentliche Image (2.30.7) konnte | F10b beim Nutzer; die Verweigerung kommt sauber beim Launch, kein Hänger |
| R-16 | Arm-Drift: L ändert `arm_rc2.sh` nach dem Gate | Gate wiederholen (Env-Diff + Namespace-Vergleich mit dem RC2-Parser, Befehle im Bericht vom 24.09.) |
| R-17 | NCCL-Schatten (zwei RECORDs, eine Datei) | Gewinner im Image erzwungen, Banner + sha256 im Build, nie `pip uninstall nvidia-nccl-cu12` |

---

## 14. Arbeitsteilung und Abgleich mit NF

| Wer | Liefert |
|---|---|
| 27B-Sitz (R) | Rahmen, Dockerfile, Entrypoint, Vorbau, Kontext, Host-Skript, Proben, `27b.env`, Stufe-B-Commit bae049a3b4 |
| Agent K (27B) | Form `27b-fp8` (Profil + nativer Boot, danach Rig-Cache für den Vorbau) |
| NF-Sitz | `nf.env` (übernommen), NVFP4-Basis und `nf-nvfp4`, NF-Pin, NF-Tests |
| Operator | Fenster, Build-Go, Host-Ausführung, Abstimmung |
| Nutzer | §15 |

**NF-Antworten (NF_PROFILE §11.3) eingearbeitet:**
- **N1:** gleiches venv, Pin-Kandidat `ce1dac1984`.
- **N2:** `nf.env` übernommen: `ARENA_GIB=4`, `MAMBA_SLOTS=32`, Instrumente getrennt.
- **N3:** FlashInfer-Liste aus dem Rig-Cache samt gencode; tvm-ffi über die Saat.
- **N4:** Saat im Kontext bzw. per Host-Skript:
  - Census `_graph` und `_computed`
  - Korridor-Sample
  - PROBE_RING
  - phase_footprint
  - `weg2_measured_record.json`
- **N5:** NCCL für NF unbelegt, daher die Transport-Regel aus §4.
- **N6:** Coredump-Env bleibt Diagnose und läuft nicht im Default.

NF-Hinweise umgesetzt: tmpfs-Store-Prüfung, privates shm, `--pid` bleibt privat. NF OP-08 (NVML-PIDs): In CT999 übersetzt der Treiber in den Namespace des Aufrufers; T0 bestätigt das für Docker.

---

## 15. Offene Fragen

F7–F16 sind entschieden (Tabelle oben). **Noch offen:**
- **F13** (beim Nutzer): `HOUSE_GUARD=memlimit` oder `ct999-ruht`. Beide sind im Skript; ohne Wahl verweigern die GPU-Schritte.
- **F10b** (neu, aus F10): Das Upgrade mit NCCL 2.28.9 verweigert Co-Location (Duplikate in `--rank-gpu-id`), die das öffentliche `cu130-nccl2307` mit 2.30.7 konnte. Möglich: (a) so veröffentlichen und die Verweigerung dokumentieren (README-Entwurf tut das); (b) eine zweite Build-Variante `NCCL_PIN=2.30.7` für den Server-Modus unter eigenem Tag; (c) die weg2-Linie auf 2.30.7 heben — das verlangt native Boots der Linie mit 2.30.7, bevor das Image es tut.
- **RC2-Quelle:** erledigt — `arm_rc2.sh` lag um 23:00Z vor, `profiles/27b.env` ist dagegen gegatet (P-Trim an).

**NF-Sitz:** bae049a3b4 picken (pickbar geprüft); `nf-nvfp4.env` liefern (Format jetzt `nvfp4-modelopt`); nach dem ersten nativen NVFP4-Boot den Rig-Cache für den Kontext freigeben. `nf.env` braucht keine Änderung für F8: der Entrypoint exportiert `HTSGLANG_INSTRUMENTS=0` vor dem Sourcen.
**Agent K:** `27b-fp8.env` nach der Schnittstelle von `27b.env`, nativer FP8-Boot vor dem Kontext.

---

## 16. Dateien (`/spinning/gpu-arb/docker/`)

| Datei | Inhalt | Status |
|---|---|---|
| `RELEASE_PLAN.md` | dieser Plan | Entwurf |
| `NF_PROFILE.md` | NF-Profil | NF-Sitz |
| `Dockerfile` | Upgrade von `docker/htsglang.Dockerfile` (August r3) | Entwurf, nicht gebaut |
| `.dockerignore` | für die Kontext-Wurzel | Entwurf |
| `entrypoint.sh` | Dispatcher (August-Modi) und weg2 | Entwurf, `bash -n` ok |
| `healthcheck.sh` | Liveness beider Modi | Entwurf |
| `prebuild_jit.py` | FlashInfer (Server-Reihenfolge, gencode je Modul, FP8/FP4/xqa-Mappings), barlink, TMS | Entwurf, `ast` ok |
| `prepare_context.sh` | Kontext, Lock, Wheel, optionale NV-Header, Werkzeuge, ARB-Saat, Profil-Daten, gencode-Liste, tvm-ffi-Saat, Geheimnis-Scan | Entwurf, `bash -n` ok |
| `host_acceptance.sh` | Host: check/build/seed/d2/t0/serve/release/negative; F13-Schalter `HOUSE_GUARD`; `build` ruft `host_build.sh` | Entwurf, `bash -n` ok, Schutzlogik mit Attrappen geprüft, **nicht ausgeführt** |
| `host_publish.sh` | Veröffentlichungsschritt mit Sperren: Nutzer-Go (USER_GO), Revision auf Remote-Branch (sonst „UNPUSHED …“), Alt-Tag `cu130-nccl2307` nur mit REPLACE_PUBLIC=1, kein Überschreiben vorhandener Tags; ohne `--push` nur Prüfung | Entwurf, Sperren mit Attrappen geprüft, **nie gelaufen** |
| `profiles/nf.assets` | zusätzliche NF-Profildaten (x162-Wake-Kredit-Logs) | 25.09. |
| `host_build.sh` | Host-Bau mit Deckel (eigener buildx-Builder docker-container: memory/memory-swap, cpu-quota, cpu-shares; oom_score_adj), Manifest- und Baum-Prüfung, F12-Liste, D1 | Entwurf, Trockenlauf in CT999 mit Attrappen geprüft, **nicht auf dem Host ausgeführt** |
| `KNOWN_ISSUES_RC2.md` | bekannte Punkte RC2/RC2-final | ergänzt 25.09. |
| `ctx/27b-b5c7d01614/` | Build-Kontext RC2-final (Manifest `7fd2d42a…`) | erzeugt 25.09. 01:47Z |
| `README_RELEASE_DRAFT.md` | README-Neufassung auf Basis #135 (weg2, Transporte, Formate mit Status, Idle-Politik, Profile, Pfad-Env, Host-Voraussetzungen) | Entwurf, **nicht veröffentlicht** (F15) |
| `probes.py` | gen/needle/decode/flip/state (stdlib) | Entwurf, `ast` ok |
| `profiles/27b.env` | 27B INT8, RC2-Form (= `arm_rc2.sh`: RC1 + P-Trim an + Idle-Politik pp/10/4096 + fünf Schalter) | Entwurf, gegen `arm_rc2.sh` gegatet (Env 34/34, Launcher-Namespace gleich) |
| `profiles/nf.env` | NF INT4-Mixed (NF_PROFILE §11.1, wörtlich) | NF-Inhalt |
| `profiles/27b-fp8.env`, `27b-nvfp4.env`, `27b-gguf.env`, `nf-nvfp4.env`, `nf-gguf.env` | Format-Platzhalter mit Eigentümer | Platzhalter |
