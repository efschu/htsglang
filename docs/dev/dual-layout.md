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
