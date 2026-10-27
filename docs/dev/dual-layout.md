# Dual-Layout (27B NVFP4): Gewichte im D-Layout, P laedt die Differenz, MPS verteilt den GEMM-Anteil

Stand 27.10.2026 (Desk-Sitz 27B, Auftrag C). PFLICHT-Lektuere vor jeder Aenderung an Dual-Code und vor jeder Zahl, die das Dual
betrifft (Eintrag in `.claude/rules/modify-component-must-read.md`).

Kennzeichnung in diesem Text: **[Nutzer]** = Wortlaut oder Anweisung des Nutzers, **[gelesen]** = im Code/Profil/Log nachgelesen
(Datei:Zeile), **[gemessen]** = Zahl aus einem Boot-Log oder einer Messung (Quelle genannt), **[gerechnet]** = aus Budgetposten
abgeleitet, nicht gemessen, **[Lesart]** = meine Deutung, vom Nutzer zu pruefen. Zeilennummern gelten fuer den Worktree
`vision-weights-1009` am 27.10. 12:5x; `launcher.py` wird parallel bearbeitet, die Zeilen koennen driften, darum steht
jeweils ein Suchbegriff dabei.

Die Originalfassung (Nutzerworte + ausfuehrliche Lesart + offene Fragen) liegt ausserhalb des Repos:
`/spinning/gpu-arb/deskq/dual-layout-erklaerung-1027.md`. Memory: `dual-layout-erklaerung-nutzer-1027.md`.

## 1. Die Erklaerung des Nutzers (wortgetreu, nicht umdeuten)

Quelle: Nutzer, 27.10. ~12:30Z.

> gewichte werden im Decode layout geladen. dann wird die differenz eines prefill layouts die diffenz der layerbytes von P zu D auf
> die karten geladen, es gibt also pro karte layerbytes die nur P nutzt, layerbyte die beide layouts gemeinsam nutzen und layerbyte
> die nur D nutzt.
>
> dadurch fallen keine kolltkvkosten beim prefill an und decode und prefill können gemeinsam laufen. die GEMMS werden geteilt je nach
> last. bei wenig decode last bekommt p mehr gemm anteil, bei hoher decode last bekommt p weniger gemm anteil. läuft kein decode
> bekommt p den ganzen gemm anteil

Weitere Nutzerworte am selben Tag: „das layout stimmt so schon"; „du hast nur ninfer verglichen ohne unsere Fähigkeit, Prefill und
Decode gleichzeitig zu machen"; „Im Dual ist der Schnitt nach Decode-Layout und dann Prefill die Differenz davon."

Praezisierung des Nutzers: „kolltkvkosten" = das **Kollektiv**, also der **All-Reduce**, der beim Tensor-Split in jedem Layer
anfaellt. Von „KV-Uebergabe" war nie die Rede.

NEU, Nutzer 27.10. ~12:45Z: **MPS wird beim Dual genau dafuer gebraucht.** P und D laufen als **zwei MPS-Clients (doppelte
Raenge) auf jeder Karte**, und der GEMM-Anteil wird **ueber MPS** verteilt. Die anderen Formen (int8, NF Form B, Duo) brauchen kein
MPS.

## 2. Lesart (vom Nutzer zu pruefen)

1. **Eine Gewichtskopie je Karte, kein Doppelladen.** Geladen wird das D-Layout (TP3, ungleicher Shard, `--rank-tp-ratio`). Das
   P-Layout (PP3, Schnitt 31/17/16 Layer) laedt nur den Unterschied nach.
2. **Drei Byte-Klassen je Karte:** nur P, gemeinsam, nur D. Das ist im Code als reine Rechnung festgeschrieben:
   `python/flliper/srt/pdflip/dual_layout_plan.py` (Modul-Docstring, Z.1-25) nennt sie **SHARED** (D's TP-Shard der Layer, die
   P's Stufe auf *dieser* Karte rechnet, plus replizierte Tensoren, plus D's Vokabel-Shard), **PP_ONLY** (die Shards der
   anderen Raenge fuer die Layer dieser Stufe = „die Differenz") und **TP_ONLY** (D's Shard der Layer, die in anderen
   P-Stufen liegen). [gelesen] Das deckt sich mit den Nutzerworten. (Hinweis: der Docstring beschreibt auch eine aeltere
   Variante #2, die TP_ONLY bei leerem Decode in den Host-RAM parkt; das ist nicht der heutige Lauf, im heutigen Profil
   schlaeft P nur unter KV-Druck, `--dual-p-sleep on`.)
3. **Rechenfolge** [gerechnet, nicht gemessen]: Rechnet P einen Layer komplett auf Karte c, hat dieser dort die Groesse L; D
   haelt davon den Anteil r_c. Gemeinsam = r_c*L, nur P auf c = (1-r_c)*L. Jeder P-Layer, den eine Karte zusaetzlich
   uebernimmt, kostet dort also P-only-Bytes (1-r_c)*L. Ob die 5090 mehr P-Layer aufnehmen kann, haengt an ihrem freien VRAM
   nach D-Shard, Draft, geteiltem KV-Pool, Korridor.
4. **Kein All-Reduce beim Prefill.** Pipeline-Parallel (P, PP3) hat **keinen** All-Reduce, nur die Aktivierungs-Uebergabe
   zwischen den Stufen. Tensor-Parallel (D, TP3) zahlt den All-Reduce **in jedem Layer** (laeuft ueber BAR1, Logzeile
   `barlink-BAR1: oneshot all_reduce`). Beide Layouts haben ihre Gewichte resident, darum laufen P und D **gleichzeitig** auf
   denselben Karten. Zahlen zur Uebergabe (1024 Token x 5120 x 2 B ~ 10 MiB je Chunk und Stufenwechsel, rechnerisch ~0,4 ms
   bei 25 GB/s) sind [gerechnet]; der Planer meldet `crossing_ms=2.3` [gelesen, Boot-Log 27.10. 11:55, PP-CUT solver-Zeile]; die Rechnung vom 06.10. nennt
   10 MiB-Frames und 4,1 ms je Chunk [gelesen, `deskq/done/dual-schnitt-262k-1006.md`, dort als Preis aus Messpreisen].
5. **GEMM-Anteil nach Last** [Nutzer]: wenig Decode -> P mehr; viel Decode -> P weniger; kein Decode -> P alles.

## 3. Die Byte-Klassen je Karte (Kurzform)

| Klasse | Bedeutung | Wer haelt sie | Wo im Code |
|---|---|---|---|
| gemeinsam (SHARED) | D's TP-Shard der Layer, die P auf dieser Karte rechnet | P und D lesen dieselben Bytes | `dual_layout_plan.py` Z.1-25; Launcher: `--dual-share` (D bootet als Union-Besitzer, P bindet das Gemeinsame) |
| nur P (PP_ONLY) | die uebrigen Shard-Anteile (1-r_c)*L der P-Layer dieser Karte | nur P, zusaetzlich geladen („Differenz") | `dual_layout_plan.py`; P-Budget je Rang `--rank-gpu-memory-mib 6610,5050,5200` im Profil |
| nur D (TP_ONLY) | D's Shard der Layer anderer P-Stufen | nur D | `dual_layout_plan.py` |

Folge fuer Aenderungen am P-Schnitt: nie am Schnitt drehen, ohne die P-only-Bytes je Karte gerechnet zu haben.

## 4. Kollektiv-Unterschied PP gegen TP

| | P (PP3, Prefill) | D (TP3, Decode) |
|---|---|---|
| Kollektiv je Layer | **keins** (nur Stufen-Uebergabe der Aktivierung) | All-Reduce in jedem Layer (BAR1, `barlink-BAR1: oneshot all_reduce`) |
| Gewichte | D-Layout geladen + nur-P-Differenz | D-Layout |
| laeuft gleichzeitig | ja, beide resident | ja |

[gelesen] Im D- und P-Log erscheint beim MPS-Betrieb `barlink-BAR1: MPS client (CUDA_MPS_PIPE_DIRECTORY set) -- the cooperative
full-card grid variant is OFF` (Boot 27.10. 11:55, `.../docker-acceptance/27b/evidence/boot_weg2_dkr27bnvfp4dualbar1drei10271155_*.P.log`);
sie ist die Pruefzeile, dass die Raenge als MPS-Client laufen (`barlink_bar1.py:2331`).

## 5. MPS-Mechanismus mit Datei:Zeile

Zwei Prozessgruppen je Boot, also zwei Raenge je Karte: **P** (3 Stufen PP0-PP2, je eine Karte) und **D** (3 TP-Raenge, je eine
Karte). [gelesen] `launcher.py` ~Z.29417-29445: `spec_p`, danach `_sd` (D, Dual-Share-Pfad). Beide bekommen dieselbe MPS-Umgebung.

| Schritt | Stelle |
|---|---|
| Opt-In-Env `FLLIPER_PDFLIP_DUAL_MPS_OPT_IN` (Konstante, mit Begruendung der frueheren Sperre) | `launcher.py:16104-16112` (`DUAL_MPS_OPT_IN_ENV`) |
| Verweigerung ohne Opt-In, Verweigerung ohne `--dual-layout` | `launcher.py:16131-16152` (`resolve_dual_layout`) |
| Privater MPS-Daemon je Boot: `nvidia-cuda-mps-control -d`, Pipe-Verzeichnis `/tmp/pdflip-dual-mps-<tag>/pipe`; liefert die Client-Env `CUDA_MPS_PIPE_DIRECTORY`, `CUDA_MPS_LOG_DIRECTORY` | `launcher.py:16576-16596` (`start_dual_mps`) |
| Aufruf: Client-Env an P (`spec_p.env.update(ns._dual_mps_env)`) und an D (`_sd.env.update(...)`, im Nicht-Share-Pfad `spec_d.env.update(...)`) | `launcher.py:29417-29418`, `29441`, `29717` |
| Statischer SM-Anteil von P: `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` (nur mit `--dual-mps on` und `--dual-p-sm-pct` < 100) | `launcher.py:16446-16453` (`dual_p_sm_env`), Flag `launcher.py:24574` |
| Pausenzulassung ohne MPS: `--dual-p-duty` (Default 1.0 = pausenfrei) | `launcher.py:16432-16443` (`dual_duty_env`), Flag `launcher.py:24566`; Wirkung `pdflip/dual_duty.py`, P-Seite `managers/scheduler_pp_mixin.py:12523-12535` |
| P-Prioritaet unter MPS: `CUDA_MPS_CLIENT_PRIORITY=1` (`--dual-p-mps-low-prio`, Default off) | `pdflip/dual_share.py:940-952`; Aufruf `launcher.py:16503-16507` |
| Dynamischer SM-Anteil: Regler im Front, Stufen 100/75/50/25 % | `pdflip/dual_share.py` (`MODES` Z.63, `ShareConfig` Z.140-210), `pdflip/dual_green.py:1161` (`GreenController`), Front-Bau `dual_share.py:1021-1045` |
| Dynamischer SM-Anteil: P-Seite, Green-Context-Leiter je Forward | `pdflip/dual_green.py:435` (`GreenLadder`), `:753` (`GreenActuator`), `:977` (`maybe_arm`); Scheduler `managers/scheduler.py:2666-2672`; Forward-Grenze `managers/scheduler_pp_mixin.py:12542-12547` (`pick`) |
| Flags | `--dual-mps` `launcher.py:24639`, `--dual-priority`, `--dual-share-actuators`, `--dual-green-ladder` `launcher.py:24580-24608` |
| Profil `27b-nvfp4-dual.env`: `--dual-mps on`, `--dual-p-sm-pct 100`, `--dual-p-duty 1.0`, `--dual-priority dynamic`, `--dual-share-actuators green,duty`, `--dual-green-ladder on`; Opt-In `_form FLLIPER_PDFLIP_DUAL_MPS_OPT_IN 1` | `/spinning/gpu-arb/docker/drei/27b/fe572f3224/27b-nvfp4-dual.env` Z.144-165 und Z.194-196 (nach dem Kommentarkopf vom 27.10.; ohne ihn 9 Zeilen weniger); Repo-Kopie `docker/flliper/profiles_release/27b-nvfp4-dual.env` |

[Lesart, aus dem Code gelesen] Arbeitsteilung: **MPS selbst** sorgt dafuer, dass Kernel von P und D (zwei Prozesse, zwei
CUDA-Kontexte je Karte) **gleichzeitig** auf den SMs laufen statt vom Treiber zeitgeschnitten zu werden. Der **GEMM-Anteil nach
Last** wird im Standardprofil nicht ueber den MPS-Prozentwert, sondern ueber die **Green-Context-Leiter** in P (innerhalb des
MPS-Clients) eingestellt, gesteuert vom Regler im Front. Das passt zur Nutzeraussage, MPS werde dafuer gebraucht, ist aber **nicht**
dasselbe wie „MPS verteilt per Prozent": der statische MPS-Prozentwert ist im Profil bewusst 100. Das ist ein Punkt zur
Bestaetigung durch den Nutzer (Abschnitt 8, Punkt 1).

[gemessen, Boot 27.10. 11:55, Front-Log `...115529.front.log`] `PDFLIP-DUAL MPS daemon started: pipe /tmp/pdflip-dual-mps-... -- P and
D are MPS clients on every card`; `DUAL-SHARE GREEN ladder front armed (on)`; `P-STUFE bs=1/6 ... stage=0->1 f=0.75` (ein Decode
-> P 75 %), `P-STUFE bs=0/6 ... stage=1->0 f=1.00 reason=d_idle` (kein Decode -> P alles). P-Log: `DUAL-GREEN P ladder pp=0 ... RTX 5090
sm120 sm_total=170 granularity=8 rungs=[0.75:128,0.50:88,0.25:48]`. Auf PP2 (3080): `W-DUAL-SHARE-FALLBACK mech=green ... rung 1
f=0.75 mask does not bite: time x1.05 for 52/68 SM ... -> served by duty/chunk` (diese Stufe laeuft dort ueber den Rueckfall).

## 6. Messhinweise

- Eine Messung „nur Prefill" (Decode im Leerlauf) zeigt P mit dem **ganzen** GEMM-Anteil = reine P-Rate (Grenzwert). Eine Messung
  „nur Decode" zeigt D mit dem ganzen Anteil. Beides sind Grenzwerte, **kein Urteil ueber das Dual**.
- Ein Vergleich mit ninfer (macht Prefill ODER Decode) ist nur ueber die **Mischlast** fair; Gleichzeitigkeit ist die Faehigkeit, die
  ninfer nicht hat. Testsatz: Abschnitt `mixed` in `/spinning/gpu-arb/deskq/bl/b27_testsatz.py` (Kennzahlen `mix_*`,
  `mix_gewinn_<s>`).
- [gemessen 27.10., Dual NVFP4, Decode im Leerlauf, formb98v, je 1024-Token-Chunk, nur Rechenzeit] PP0 5090 31 Layer 243 ms; PP1 3080
  17 Layer 511 ms; PP2 3080 16 Layer 482 ms. Kernel 5090 W4A4 (FlashInfer-CUTLASS), 3080 W4A8 auf INT8-Kernen
  (`nvfp4_w4a8_int8_sm86`). Die 3080 sind in P die langsame Stufe.
- Unter Last prueft man die Stufe an `P-STUFE`-Zeilen im Front-Log (`stage=`, `f=`, `sm=`) und an `rung=` in den
  `Prefill rank batch`-Zeilen des P-Logs, nicht an einer Einzelzahl.
- Zahl zum Dual nennen heisst: gegen `/spinning/gpu-arb/deskq/formb-bestwerte.md` stellen und die Last (nur P / nur D / gemischt)
  dazuschreiben.

## 7. Testvarianten sind keine Formen

**Formen** des 27B sind nur: **int8**, **dual**, **duo** (und die int4ar-Zeile, sowie NF Form B fuer das MoE-Modell). Create-Args
aller Varianten liegen in `/spinning/gpu-arb/deskq/rotation/gen/` (`create_args_<name>.txt`, erzeugt von `mk_create_args.py`).

**Testvarianten** (gleiche Form, ein Schalter anders; nicht als Form fuehren, nicht als Ergebnis „des Dual" ausgeben) [gelesen,
Diff der Create-Args gegen `create_args_dual.txt`/`create_args_int8.txt`]:

| Variante | Unterschied zur Form |
|---|---|
| dualm | Env `FLLIPER_FP4_NATIVE_MIXED_SM8X=marlin`: 3080 mit W4A16 (Marlin, BF16-Kerne), native-mixed sonst |
| dualu | anderes Profil `27b-nvfp4-dual-w4a16.env` (uniform Marlin, alle Karten); das normale Profil haengt als `...-native.env` ein |
| dualv | Image `formb98v`, L1.5-Env aus (`FLLIPER_PDFLIP_L15=0` usw.) |
| int8v | wie int8, L1.5-Env aus |

Ein Ergebnis aus einer Testvariante traegt ihren Namen; die Form heisst erst dann so, wenn der Nutzer sie dazu erklaert.

## 8. Was ich (Agent) NICHT verstehe / offen

1. **Wo und ob der MPS-Thread-Anteil dynamisch nach Decode-Last umgestellt wird.** Befund [gelesen, Code und Log]:
   - `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` wird **nur statisch** gesetzt: `launcher.py:16446-16453` (`dual_p_sm_env`), ein
     Client liest ihn beim Erzeugen seines Kontexts (`dual_green.py:5`, `scripts/dual_group/green_ctx_probe.py:6`). Er kann zur
     Laufzeit nicht umgestellt werden. Das Profil setzt ihn auf 100 (= nicht gesetzt); `refuse_dual_priority`
     (`launcher.py:16565-16569`) verbietet die Kombination mit der Leiter ausdruecklich.
   - `set_default_active_thread_percentage` und ein Aufruf von `nvidia-cuda-mps-control` zur Laufzeit: **nicht gefunden**
     (einziger Aufruf: `launcher.py:16588`, `-d` beim Start; `mps_exit_quiesce.py` nur fuer den Abbau).
   - **Dynamisch** wird der Anteil durch die **Green-Context-Leiter** umgestellt, nicht durch MPS-Prozent: Regler im Front
     (`dual_green.py:1161`, `GreenController`; Eingaenge D-Batchgroesse bs, wartende P-Tokens tau, gemessene D-Rundenzeit),
     Ausgang eine Stufe 0..3 = P 100/75/50/25 %, geschrieben in eine Ctl-Datei; PP0 stempelt die Stufe auf die Request-Leitung,
     alle drei P-Stufen wenden sie am Forward-Rand an (`scheduler_pp_mixin.py:12542-12547`, `GreenActuator.pick`). Tabelle im
     Profil: `FLLIPER_PDFLIP_DUAL_SHARE_GREEN_TABLE "1:1:1;2:2:2;1000000000:3:3"` (bs1 -> 75 %, bs2 -> 50 %, bs3+ -> 25 %),
     kein Decode (`reason=d_idle`) -> P 100 %. Das entspricht der Nutzerregel (wenig Decode -> P mehr, viel -> weniger, keiner -> alles).
   - Die Leiter laeuft **innerhalb** des MPS-Clients P (Green Contexts unter MPS; Metall-Probe in der Docstring von
     `dual_green.py`). Ob die Nutzer-Aussage „GEMM-Anteil ueber MPS verteilt" genau diese Leiter meint oder den statischen
     MPS-Prozentwert, habe ich nicht festgestellt: **offen, Nutzer fragen**. Im Standardprofil ist nur die Leiter wirksam.
   - Mehr offen: Die 3080 (Granularitaet 2 SM) bedient Stufe 1 nicht (Maske beisst nicht, Rueckfall duty/chunk, s. Abschnitt 5).
     Wie sich das auf den GEMM-Anteil der langsamen Stufen auswirkt, ist nicht untersucht.
   - Widerspruch im Code, den ich nicht aufloese: `launcher.py:24639-24644` (Hilfetext `--dual-mps`) und Meldung
     `launcher.py:16145-16152` sagen „REFUSED ... the dual layout runs without MPS", das Release-Profil setzt aber
     `FLLIPER_PDFLIP_DUAL_MPS_OPT_IN=1` und das Dual laeuft damit (Boot 27.10. 11:55). Der Profil-Kopf nennt als Bedingung,
     dass der Baum `b855aecfd3` traegt (Wurzel des frueheren Wedge); `git merge-base --is-ancestor b855aecfd3 HEAD` ist wahr
     [gelesen]. Der Hilfetext ist wahrscheinlich veraltet [Lesart].
2. **Warum der P-Schnitt auf 31/17/16 gepinnt ist.** Befund:
   - Die Pinzeile: `PROFILE_ARGS+=(--pp-stage-ratio 31,17,16 --pp-attn-stage-ratio 7,5,4)` in `27b-nvfp4-dual.env` Z.99 (Rig-Datei; Repo-Kopien ohne Kopf:
     `docker/flliper/profiles_release/27b-nvfp4-dual.env` Z.90, `tools/release/profconv/27b-nvfp4-dual.env` Z.90). Der Launcher
     druckt dazu `PP-CUT solver: PINNED (user override) ...` (`planner/pp_cut_launch.py:656`, `launcher.py:20428`;
     Regel: `--pp-stage-ratio` schlaegt das Ziel `--pp-solve-objective`, `launcher.py:24131`). Im Boot-Log 27.10. 11:55: `PINNED
     (user override) layers=31,17,16 attn=7,5,4 ... pool_tokens=788255`; frei gewaehlt haette der Solver `45,9,10 / 11,2,3`
     (`pool_tokens=546732`, makespan 403,9 ms statt 696,8 ms) [gemessen/gelesen].
   - **Grund** [gelesen in `/spinning/gpu-arb/deskq/done/dual-schnitt-262k-1006.md`, Rechnung des 27B-Sitzes 06.10.; Anlass
     Nutzerauftrag 06.10. „Dual 262k, Weg 1" und „P muss den vollen Kontext tragen", vermerkt in `ZUSTAENDIG.md` und im
     Kommentar `27b-nvfp4-dual.env` ~Z.95]: Der Planer kennt das Dual-Poolmodell nicht (sein Pool ist das Flip-Modell, ganze Karte)
     und wuerde `44,10,10` waehlen; im Dual-Pool fehlen dort auf der 5090 ~2440 MiB. Bester statischer Schnitt fuer vollen Kontext
     (262144) bei leerem D und die 5090 nicht allein belastet: **31,17,16 / 7,5,4** (Marge bei D->0 +923/+2027/+2250 MiB je Karte,
     toleriert D bis ~153k Zeilen; FA-Sprung zwischen 32,16,16 und 31,17,16 = 520 MiB). Preis [dort als Vermutung gerechnet]: P -40 %.
   - **Git-Hinweis:** Im Repo ist die Zeile seit dem Umzugs-Commit `2575caae5e` (07.10., „F0-A: rename kit", tools/release/profconv)
     bzw. `67366ac820` (08.10., „F0-G (27B line): profiles in the new spelling", docker/flliper/profiles_release) enthalten; die
     eigentliche Aenderung 45,10,9 -> 31,17,16 geschah davor am Rig-Profil (`/spinning/gpu-arb/...` ist kein Git-Repo), also
     zwischen dem Vorschlag 06.10. ~10:41Z und dem Dual-262k-Boot (`ZUSTAENDIG.md` Z.226, 06.10. 10:51Z).
     `git log -S"--pp-stage-ratio 31,17,16" -- tools/release/profconv` findet nur den Umzugs-Commit.
   - **Stolperstein** [gelesen]: der Kommentar direkt ueber der Pinzeile (Rig Z.98, Repo Z.89) sagt noch „P-Schnitt Dual: 45,10,9 / attn 11,2,3
     (dual1h ...)", der Wert darunter ist 31,17,16. Der Kommentar ist veraltet, der Wert gilt.
   - **Offen:** ob der Nutzer 31,17,16 heute noch will. Die Pin-Rechnung stammt von der 262k-Vorgabe (Pool fuer P, nicht
     Geschwindigkeit); der 27B-Sitz hat am 27.10. gemessen, dass die 3080-Stufen die langsamen sind (511/482 ms gegen 243 ms),
     und die P-only-Bytes je Karte sind fuer einen neuen Schnitt noch zu rechnen (Abschnitt 2, Punkt 3).
3. Weiteres, das ich nicht weiss: ob `r_c` im Betrieb gleich dem Gewichtsvektor von `--rank-tp-ratio` ist (laut Dry-Run
   `[58,25,25]`, 06.10.) oder durch Vision/Draft verschoben wird. Und: ob der Vergleich mit ninfer ueber die Mischlast eine
   feste Lastmischung braucht (Testsatz `mixed` nennt eine, die Wahl ist nicht begruendet).

## 9. Wer das benutzt

- Vor Aenderungen an Dual-Code: diese Datei lesen (Regel in `.claude/rules/modify-component-must-read.md`).
- Dual-Code: `python/flliper/srt/pdflip/launcher.py` (Dual-Teile: `resolve_dual_layout`, `start_dual_mps`, `dual_p_sm_env`,
  `dual_duty_env`, `refuse_dual_priority`, Start P/D), `python/flliper/srt/planner/pp_cut_launch.py` (P-Schnitt),
  `python/flliper/srt/managers/scheduler_phase_layout_mixin.py`, `python/flliper/srt/managers/pdflip_memory_saver.py` (Dual-P),
  `python/flliper/srt/pdflip/dual_*.py`, `python/flliper/srt/managers/scheduler_pp_mixin.py` (P-Forward, Duty, Green).

## 10. Fremdprojekte

| gesucht | gefunden | Lizenz | uebernommen | warum |
|---|---|---|---|---|
| Dokumentation der CUDA-MPS-Mechanik (Prozent, Client-Prioritaet, Green Contexts) | NVIDIA MPS Documentation und CUDA Driver API (Green Contexts), `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` wirkt beim Erzeugen des Kontexts | proprietaer, oeffentliche Doku | nur als Quelle | der Code nutzt sie bereits; hier nichts nachgebaut |
| vorhandene Dual-/PD-Erklaerungen im Projekt | `docs/DESIGN_*`, `dual_layout_plan.py` (Byte-Klassen), `dual_green.py` (Leiter), `deskq/done/dual-schnitt-262k-1006.md` | projekteigen | ja, verwiesen | keine zweite Rechnung begonnen |

## 11. P-Chunkbreite, P-Schnitt und Mischlast im Dual: Messbefund und Optimum (Dual-Sitz, 27.10.)

Kennzeichnung wie oben: **[gemessen]** = Zahl aus einem Boot-Log (Quelle genannt), **[gefittet]** = Gerade durch gemessene Einzelwerte, **[gerechnet]** = Modell
aus gemessenen Termen, **[vermutet]** = nicht belegt. Quellen: Dual-Boots `dkr27bnvfp4dualbar1drei10271355` (13:55, dualm = 3080 Marlin W4A16, Image formb99w) und `...10271415`
(14:15, dualm), zum Vergleich `...10271155` (11:55, W4A8, formb98v); Dateien unter `/spinning/docker-acceptance/27b/evidence/boot_weg2_*.{P,front}.log`.
Werkzeug: `python/flliper/srt/planner/dual_stage_fit.py` (liest nur die `Prefill rank batch`-Zeilen; das Dual druckt kein `#PGAP`: `launcher.py` `_pgap = p_hostgap and not dual_layout`,
Nutzerauftrag 20.10.), Test `test/registered/unit/pdflip/test_dual_stage_fit_1027.py`.

### 11.1 Was die Chunkbreite heute begrenzt (A)

| Ebene | Wert | Stelle |
|---|---|---|
| P-Breite (Planer) | `--p-chunk-policy dynamic --p-chunk-max 1024`, Leiter 512/1024, Graph-Bucket 512 (1024 laeuft eager), Kurzprompt-Bypass bis 8192 ungecachte Token = fest 512 | `27b-nvfp4-dual.env` Z.97 (Repo-Kopie `docker/flliper/profiles_release/`), `launcher.py` `apply_p_chunk_policy` (Z.2017), `p_chunk_policy.py` `dynamic_min_tokens` (Z.640) |
| P-Stufenmodell des Planers | `27b-nvfp4.pchunk.json`: ein Flip-Modell (Schnitt 42/11/11), **nicht** das Dual: Planer sagt 16347 Token in 2427 ms voraus (`P-CHUNK-POLICY plan ... predicted_ms=2427.1`), gemessen 7,2-7,7 s TTFT | P-Log 14:20:54 [gemessen] |
| P-Speicherwache | `P-CHUNK-MEM-GUARD` wirkt nur ueber 2048 und nur auf PP0 (5090, Rate 0,2256 MiB/Token); die 3080-Stufen sieht sie nicht | `scheduler.py` `_p_chunk_mem_guard` (Z.10730), `p_chunk_policy.py` Z.771-795 |
| P-Transient im Karten-Ledger | `card_law`: der in DIESEM Boot gemessene groesste Transient je Karte (Max ueber beide Gruppen), kein Plan-Posten; waechst mit der Breite selbst | `dual_p_kv_stage.py` `card_law` (Z.455) |
| Verleih D -> P | `awake_lend` / `sleep_lend` verschieben **KV**-Bytes innerhalb des Karten-Pools K_c (P gibt, D nimmt, Q-660); fuer einen Chunk-Transient (kein KV) ist das nicht der Hebel | `dual_p_kv_stage.py` Z.1670 / Z.1602, `card_kv_ledger.py` `lend`/`reclaim` |
| D-Seite | D prefillt im Dual nichts: X = 1 + `--dual-d-prefill-tokens` (0); D-Breitenabstimmung `extend_trim.width_vote` und `D-POOL-LEND` (Gate: kein Dual) sind im Dual ohne Wirkung | `launcher.py` `dual_x_tokens` (Z.4283), `d_pool_lend.py` Z.113 |

**Transient je Zeile** [gemessen, `PDFLIP-VRAM-PEAK phase=chunk`, Median, beide dualm-Boots, n = 1489 Fenster zu 1024 Zeilen]: 1024 Zeilen 197 / 197 / 205 MiB (PP0 / PP1 / PP2, Maximum
217 / 221 / 225), 512 Zeilen ca. 90-103 MiB, also **0,19-0,20 MiB je Zeile auf allen drei Stufen**. Fuer 2048 / 4096 / 8192 Zeilen [gerechnet, linear] ca. 0,4 / 0,8 / 1,6 GiB.
Der Ledger-Wert `law = 905 MiB` auf der 5090 ist **kein Chunk-Transient**: die beiden 906 / 867 MiB-Fenster (`rows=512`, `window_ms` ca. 10 s) fallen in die Vision-Probe
(14:18:13, `W102 PdFlipVisionStage`, sdpa-Vision-Turm auf PP0) [gemessen, Zeitgleichheit; Zuordnung vermutet]. Kleinster freier Kartenspeicher waehrend P-Chunks im selben Lauf
[gemessen]: 2041 / 1792 / 1696 MiB bei 1024 Zeilen (PP2 1104 bei einem 512er-Fenster). **Folgerung** [gerechnet]: 2048 passt auf allen Karten (+200 MiB, der torch-Cache haelt je
Prozess 1,1-1,4 GiB frei reserviert), 4096 ist auf PP2 knapp (1104 - 600 = 504 > near-OOM 400), 8192 laeuft dort in den OOM. Die Wache sieht PP1/PP2 nicht.

### 11.2 Stufenzeiten (E, B)

**Gerade je Stufe** `T(x) = a + b*x`, 1024er Chunk, Rung 0 (Decode im Leerlauf), x = mittlere Tiefe in 1k [gefittet aus `Prefill rank batch gpu-ms`, ca. 370-960 Chunks je Stufe, sd 2-6 ms]:

| Stufe (Karte, Layer/FA) | a: 13:55 / 14:15 (W4A16) | a: 11:55 (W4A8) | b ms je 1k Tiefe (W4A16) |
|---|---|---|---|
| PP0 (5090, 31/7) | 103,8 / 99,6 ms | 99,9 | 1,43 / 1,49 |
| PP1 (3080, 17/5) | 322,7 / 325,6 ms | 271,3 | 2,56 |
| PP2 (3080, 16/4) | 335,8 / 336,2 ms | 280,5 | 2,19 / 2,22 |

Abgeleitet [gerechnet]: 3080 19,0-19,2 ms je Layer (Marlin W4A16; W4A8 15,96, also +20 % an den bindenden Stufen), 5090 3,2-3,3 ms je Layer, Attention-Anteil 0,51-0,55 ms je FA-Layer und 1k
Tiefe auf der 3080, 0,21 auf der 5090. **PP2 traegt zusaetzlich ca. 30 ms je 1024er Chunk** (16 Layer rechnen 336 ms, 16 x 19,06 = 305 ms): derselbe Betrag unter W4A8 (25 ms), er waechst mit
den Token (512er Chunk: ca. +12 bis +20 ms), ist also kernelunabhaengig. Ursache **offen** [vermutet: Last-Stufe, nicht lm_head auf der letzten Zeile]; ein Profiler-Lauf auf PP2 klaert es. Die 5090 ist
mit 100-300 ms (Tiefe 0-128k) zu 31-48 % ausgelastet, die zwei 3080 binden die Pipeline.

**512er Graph-Chunk gegen 1024er eager** [gemessen, Median je Tiefenband, 3080-Stufen, beide Boots]: bei Tiefe < 3k 2 x 132-140 ms (PP1) gegen 332-336 ms, bei 3-20k 2 x 138-158 gegen 346-353 ms,
also **512 ist 10-21 % billiger je Token** (PP2 ebenso 11-20 %); ab 100k Tiefe ungefaehr gleich (PP1 333 / 336 gegen 640 / 619 je 1024, nur n = 5 je Boot). Der Planer waehlt 1024 mittig, weil sein
Flip-Modell das so vorsagt (`gain=+4.1%`). **Mehr als 1024 bringt auf den bindenden 3080-Stufen keine Rate** [gerechnet aus: Marlin je Token bei 512 -> 1024 flach, kein Messwert darueber];
das int8-Flip-P ist von 2048 bis 8192 ebenfalls flach (4768 / 4601 / 4727 tok/s, Briefing 27.10.). Nur PP0 (nicht bindend) gewinnt mit der Breite (NVFP4-Modell -13 % je Token bei 2048).

**Schnitt** [gerechnet: Stufenmodell aus den Fits des 14:15-Boots, Kalibrierfaktor 1,065 auf die gemessenen TTFT 30,0 / 68,0 / 177,4 s des Ist-Schnitts; ohne Faktor unterschaetzt das Modell die
Zeit um 3-12 % (Pipelineluecken); `python3 -m flliper.srt.planner.dual_stage_fit <P.log>`]:

| Schnitt (FA) | tok/s 16k / 64k / 128k / 256k | gegen Ist | 5090-K0-Marge D leer | D neben 262k-Prompt | P-Budget `--rank-gpu-memory-mib` |
|---|---|---|---|---|---|
| 31,17,16 (7,5,4) Ist | 2528 / 2285 / 1909 / 1424 (gemessen 64k 2186, 128k 1899-1951, 256k 1456) | - | +1032 MiB | ca. 86-90k Zeilen | 6610,5050,5200 |
| 32,16,16 (8,4,4) | 2535 / 2333 / 2010 / 1561 | 0 / +2 / +5 / +10 % | +426 | ca. 38-44k | 6888,4659,5200 |
| **33,16,15 (8,4,4)** | 2668 / 2446 / 2093 / 1611 | **+6 / +7 / +10 / +13 %** | **+322** | ca. 30-36k | **6992,4659,4979** |
| 34,15,15 (8,4,4) | 2676 / 2448 / 2094 / 1612 | gleich wie 33,16,15 | +218 | ca. 21-27k | 7095,4437,4979 |
| 35,15,14 (8,4,4) | 2825 / 2573 / 2185 / 1665 | +12 / +13 / +15 / +17 % | +114 (unter Modellfehler +-100) | ca. 11k | 7200,4440,4760 |

Marge, D-Toleranz, Budgets: `deskq/dual-p-schnitt-rechnung-1027.md` (Formeln des Planers `propose_dual.pool_of`, am 11:55-Boot geeicht; die Zeilen 32,16,16 und 33,16,15 neu gerechnet mit
demselben Skript). 33,16,15 und 34,15,15 sind gleich schnell, weil PP2 (15 Layer + 30 ms) in beiden bindet: 33,16,15 behaelt 104 MiB mehr Marge. 36,14,14 und mehr kosten den
vollen 262k-P-Kontext (9 FA-Layer auf der 5090: Marge negativ). Der Schnitt bleibt **gepinnt**: der Nutzerauftrag 06.10. ("P muss den vollen Kontext tragen") ist der Grund fuer 31,17,16; 33,16,15
spart den 262k-Kontext nicht, nimmt aber der 5090 bei D-Last Pool (D-Toleranz 90k -> ca. 35k Zeilen waehrend ein 262k-Prompt prefillt). Die Entscheidung trifft der Nutzer.

### 11.3 Decode neben Prefill (C)

[gemessen, Front-Log `P-STUFE`-Zeilen mit `d_round_ms`, Median, zwei dualm-Boots, P wartet auf Token (`pending` > 0)]: D-Runde bei bs 1: 41 / 34 ms (Stufe 1, P 75 %), 29 / 28 (Stufe 2);
bs 2: 27-33 ms (Stufen 1-2), 41-48 ms (Stufe 3); bs 3: 28-45 ms; gegen die gelernte Solo-Runde `tsolo_ms` 20-31 ms. **D laeuft neben dem P-Prefill mit 1,0-1,5 x seiner Solo-Rundenzeit**
(Ausreisser bis 880 ms bei Stufenwechseln). Dagegen verlangsamt sich P unter Decode-Last [gemessen, Verhaeltnis zur Rung-0-Geraden bei gleicher Tiefe]: 14:15-Boot Rung 1 / 2: PP1 x2,2 / x2,7,
PP0 x2,9 / x3,7; 13:55-Boot nur x1,2. Die Spanne (1,2-3,7 x) haengt an der D-Last des jeweiligen Testsatzes und ist nicht aufgeloest. **Die Chunkbreite ist nicht der Decode-Hebel**: D wird
ueber die Green-Context-Leiter (SM-Anteil je Forward) geschuetzt, nicht ueber die Chunklaenge; auf den 3080 beisst Stufe 1 nicht (`W-DUAL-SHARE-FALLBACK mech=green`, Rueckfall duty/chunk), dort
bestimmt die Chunkzeit (512: 135-185 ms, 1024: 285-350 ms je Stufe) die Reaktionsgranularitaet der Leiter. Regel [vermutet, ungemessen]: unter D-Last ist 512 nie schlechter als 1024 (halbe
Forwardzeit, gleiche oder bessere Rate je Token bis ca. 20k Tiefe); Messung: Abschnitt 11.5, Block C.

### 11.4 X und Layout (D)

X bleibt 1: `launcher.py:4283` `dual_x_tokens` (X = 1 + `--dual-d-prefill-tokens`, Standard 0; Nutzerentscheid 01.10., Metall dual1m: eine 2185-Token-Extend auf D stoppte den Decode 2,5 s). Das Dual flippt nie, die
X-Kurven-Rechnung (D-Rate gegen P-Rate plus Flipzeit) gilt nur fuer die Flip-Formen; `resolve_x` verweigert die X-Schalter im Dual (W197). Weder 512 noch ein anderer Schnitt aendert daran etwas.
Auffaellig fuer spaeter [gemessen]: die P-Rate des Duals (ca. 2200 tok/s bei 16-64k, TTFT-Definition) liegt in der Naehe der D-Prefill-Rate der Flip-Form (2056-2184 tok/s, int8); ein D-Anteil am Prefill waere
nur ueber `--dual-d-prefill-tokens` > 0 moeglich und kostet D-Decode (siehe Metall-Befund oben), also nicht empfohlen.

### 11.5 Messplan fuer das naechste Fenster (Rolle `dualk`, Kette `deskq/bl/auto_w4a16_1027.sh`)

Overlay `docker/drei/27b/fe572f3224/27b-nvfp4-dual-pk.env` (Knoepfe leer = Dual-Profil byte-gleich), Rolle `deskq/rotation/gen/create_args_dualk.txt`, Treiber `deskq/bl/dual_pk_sweep_1027.py` (Rig-Dateien,
nicht im Repo). Fenster `cards [0,1,2]` exklusiv, 60-90 min; jeder Lauf gibt bei Fehlschlag das Fenster sofort zurueck, `KEEP_WINDOW=1` haelt es sonst fuer den naechsten Boot.

- **Block A (P-Breite, 1 Boot):** `ROLE=dualk ID=<ID> KEEP_WINDOW=1 PK_P_CHUNK_MAX=2048 PK_SWEEP=512,1024,2048 PK_SWEEP_DRIVER=1 PK_LEN="16k 64k" PK_REPEAT=1 FORM=dual-w4a16-pk bash /spinning/gpu-arb/deskq/bl/auto_w4a16_1027.sh`.
  Beleg: `$O/summary.txt` Abschnitt `### PK-SWEEP` (Breite je Request aus PP1-Zeilen, Median tok/s nach TTFT je Breite), `PDFLIP-VRAM-PEAK ... rows=2048 transient_mib=` (Soll ca. 400 je Stufe),
  `P-CHUNK-POLICY armed ... SWEEP=[512, 1024, 2048]`, `card_free_mib` Minimum je Rang, keine `OutOfMemory`/`RANK-DEATH`. Vorhersage [gerechnet]: 512 schneller als 1024 um 10-20 % bei 16k, bei 64k um ca. 4 %;
  2048 +-3 % gegen 1024 (flach).
- **Block A2 (Schalter-Test, wenn A 512 bestaetigt, 1 Boot):** `ROLE=dualk ID=<ID> KEEP_WINDOW=1 PK_P_CHUNK_MAX=default PK_SWEEP=default PK_DYN_MIN=32768 KURZ=1 KURZ_DEPTHS=1k,4k,16k,64k TS_EXTRA="--no-update" FORM=dual-w4a16-pk bash ...`
  (Requests bis 32768 ungecachte Token laufen fest mit 512). Beleg: Testsatz `prefill_16k_tok_s` gegen 2183-2259 (gemessen, dualm 1024), `P-CHUNK-POLICY armed ... dynamic_min_tokens=32768`.
- **Block B (Schnitt, 1 Boot):** `ROLE=dualk ID=<ID> KEEP_WINDOW=1 PK_P_CHUNK_MAX=default PK_SWEEP=default PK_DYN_MIN=default PK_CUT=33,16,15 PK_ATTN=8,4,4 PK_BUDGET=6992,4659,4979 KURZ=1 KURZ_S=420 KURZ_DEPTHS=16k,64k,128k,256k TS_EXTRA="--no-update" FORM=dual-w4a16-pk bash ...`.
  Beleg: `PP-CUT solver: PINNED (user override) layers=33,16,15 attn=8,4,4`, kein `PPCutRefused`/`REFUSED`; `LEDGER-PHYS ... budget=` je Karte gegen 4482 / ca. 4590 / ca. 4285 (Soll +-100; dualm-Pool der 3080 = Ist-Boot 4392 / 4064 plus Verschiebung der Rechnung, Marge dort > +2000);
  `GRANT-INFEASIBLE`/`LAW MAKES FULL CONTEXT INFEASIBLE` darf nicht auftreten; Abnahme = `prefill_256k` laeuft durch (Marge nur +322); Rate gegen die Vorhersage 2446 / 2093 / 1611 tok/s und gegen
  die Ist-Werte 2186 / 1899-1951 / 1456. Danach `PYTHONPATH=<worktree>/python python3 -m flliper.srt.planner.dual_stage_fit <P.log> --fit-cut 33,16,15 --cuts "33,16,15;34,15,15" --ttft <tokens:s,...>`.
- **Block C (Mischlast, optional, nur nach A):** je Variante ein Boot, `PK_P_CHUNK_MAX=512` gegen `default`: `... KEEP_WINDOW=1 TS_EXTRA="--no-update --skip vision,bs6,bs1,matrix --depths 1k --mix-stages 16k --budget-s 420"` (nicht erprobt, Flags aus
  `b27_testsatz.py`). Beleg: `mix_16k decode 3 Stroeme: VOR | WAEHREND | NACH`, `mix_prefill16k_tok_s`, `mix_gewinn_16k`, Front-Log `d_round_ms`.
- **Nachlauf:** den letzten Boot ohne `KEEP_WINDOW=1` fahren: die Kette stoppt den Container (Waechter-Stopp, MPS-Rest-Check) und gibt das Fenster zurueck. Knopfwerte bleiben in `create_args_dualk.txt`
  stehen (nur diese Rolle), `default` stellt zurueck.

### 11.6 Offene Punkte

1. PP2-Zusatzterm (ca. 30 ms je 1024er Chunk, kernelunabhaengig): Ursache offen; 5-6 % Rate lagen darin (PP1 bindet dann bei 306 ms statt PP2 bei 318, Modell).
2. 512er gegen 1024er ab 20k Tiefe: nur n = 4-7 Chunks je Band; Block A misst es.
3. Unter Decode-Last variiert P um das 1,2- bis 3,7-fache (Boots 13:55 gegen 14:15); Ursache nicht gelesen (D-Batchgroesse, Stufentabelle, MPS-Belegung der 3080).
4. `law` der 5090 (905 MiB) ist ein Vision-Spitzenwert, keine Chunkbreite; er haelt im Ledger auf der 5090 ca. 700 MiB mehr frei als P-Chunks brauchen. Wer K_c der 5090 anheben will (mehr P-Layer dort), sollte das
   Law nach Chunk- und Vision-Fenster trennen; nicht gebaut, weil ein Vision-Request den Transient wirklich braucht.
5. Der `P-CHUNK-MEM-GUARD` kennt nur PP0; fuer P-Breiten > 1024 im Dual braeuchte er die Karten PP1/PP2 (die Breite kommt von PP0, die Stufen muessten sich einigen). Nicht gebaut: ohne Gewinn durch Breite kein Bedarf.

### 11.7 Fremdprojekte

| gesucht | gefunden | Lizenz | uebernommen | warum |
|---|---|---|---|---|
| Stufen-/Pipeline-Partitionierer nach gemessener Stufenzeit | DeepSpeed `PipelineModule.partition_balanced` (Apache-2.0), Megatron-Core PP-Verteilung (BSD-3), Alpa/PipeDream (Apache-2.0/MIT) | siehe links | nein (Kenntnis, im Netz nicht neu gesucht) | alle balancieren Zeit oder Gewicht; keiner kennt den gemeinsamen Dual-KV-Pool (P und D teilen K_c) und die 262k-Marge je Karte; bei 3 Stufen reicht die Aufzaehlung |
| Fit der Stufenkosten aus Boot-Logs | `planner/pgap_stage_fit.py` (projekteigen, braucht `#PGAP`, im Dual aus) | projekteigen | Modell und Annahmen ja (Mittelstufe je Karte, Rest als Last-Stufen-Term), Parser neu | die Dual-Boots tragen keine `#PGAP`-Zeilen; `Prefill rank batch` ist die einzige Quelle |
| Chunk-Planer mit Schalter fuer feste kurze Breite | `p_chunk_policy.py` `dynamic_min_tokens` (`--p-chunk-dynamic-min-tokens`), `--p-chunk-sweep` | projekteigen | ja | fuer den Messplan reichen vorhandene Schalter, kein neuer Code im Server |
