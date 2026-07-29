# 04 — Falls du messen sollst (Kurzfassung der Pflichten)

Vollstaendig: Benchmark-Pflichten des Projekts (der Operator briefed sie);
hier das Minimum, damit Zahlen nicht wertlos sind.

1. **Accept-Laenge richtig lesen**: `meta_info.spec_accept_length` aus der
   Response. NIEMALS `spec_ema_accept_len` aus den Logs — das ist NICHT die
   Accept-Laenge. Durchsatz-Deutung braucht `spec_verify_ct` + decode_s.
2. **Spec-Tests mit hohem K** (Produktions-k=3), nie nur k=1 — und je
   Messung die bekannte Inhalts-Referenz als Spalte danebenstellen
   (Code ~3,3 / Prosa ~2,7 auf FP8). Per-Positions-Kurve (p1,p2,p3)
   gehoert in jeden Spec-Bericht.
3. **Inhaltsachse trennen**: Code / Prosa / Misch einzeln ausweisen, nie
   mitteln — Durchsatz folgt dem Ausgabe-INHALT (r=0,90).
4. **Rauschboden zuerst**: A-vs-A desselben Arms vor jedem A/B; Effekte
   unter der so gemessenen Streuung heissen "unter der Nachweisgrenze",
   nie "kleiner Gewinn". Verschraenkt messen (A,B,A,B), greedy, identische
   Prompts.
5. **Output validieren**: je Arm die ersten ~300 Zeichen des generierten
   Texts zitieren + Ein-Satz-Urteil (kohaerent/repetitiv/Muell). Ein
   schneller Muell-Lauf sieht in der Tabelle gut aus.
6. **Zeitboxen**: 10-20 s je Messpunkt, kein Punkt >60 s, Rohdaten
   (Timestamps je Token) persistieren statt nur Endzahlen.
7. **ms/Runde als Messlatte** wo moeglich (ms/Verify, ms/Prefill-Chunk),
   nicht nur tok/s — pro Rang Rechen- vs Wartezeit unterscheiden.
