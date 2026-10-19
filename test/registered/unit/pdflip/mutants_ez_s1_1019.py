"""EZ-S1 mutants (plan nf-entscheidungszentrale-plan-1019 R.6: M1, M2, M3, M4 stage 1, M10 + the S1 mutants of its own
danger directions). Usage, from anywhere:

    python mutants_ez_s1_1019.py <worktree>

Each mutant is ONE exact text replacement in <worktree>; at least one of the three S1 test files must go red:
``test_decision_numbers_1019.py``, ``test_decision_record_1019.py``, ``test_measure_on_d_1019.py``.
STRICTLY SERIAL (rig rule 19.10. after the kernel OOM): one pytest at a time, in the FOREGROUND, each in its own
``systemd-run --scope -p MemoryMax=2G``, CUDA hidden -- no ``setsid -f``, no ``xargs -P``. The source file is restored byte
for byte after every mutant (also on Ctrl-C).

Danger directions:
  * M1  ``u_p >= halbX`` classifies half / big (an upper bound taken as a number: a flip under X, SOLL R7);
  * M2  UNKNOWN -> kurz (no number taken as a short request: D over X);
  * M3  a fresh exact ``u_d`` is skipped for ``u_p`` (too big a class, the flip for nothing);
  * M4  the measuring barrier is X instead of halbX-1 (a half request is prefilled on D without the front's decision);
  * M10 a decision committed after an await without the epoch check (a decision on an old phase);
  * KV1 a stale KV reading counts as a number; KV2 the refresh runs for a sleeping D; KV3 no read at the end of P->D;
  * N1  the media n is ``exact`` again (carrier_est from n_est); PB1 the price barrier reads chars/3;
  * R1  the edge trigger is gone (the log floods); R2 the ALT shadow swallows the old site's value;
  * R3  SET-VERDICT is no longer byte-identical to the H91 line (every monitor reads it);
  * R4  a measure request gets ``D_NOW`` with u=0; R5 no barrier header; R6 an under-halbX report is no defect;
  * R7  the context gate writes no decision; R8 the boot flip pair writes no decision;
  * review Nachbesserung (test_ez_s1_review_1019.py): MB1 a MEASURE returnee is p_only by STATE again (B1), MB1b the
    needs_p exemption of a measured kurz/halb entry is gone (B1), MB2 the ``_x_exact_price -> None`` fallback classifies
    on chars/3 again (B2), MB3 ``under`` is judged on today's halbX, not the barrier's (B3), MB3b the barrier is X at depth
    0 (B3, curve mode), MB4 the barrier is armed inside the instrument except (B4), MB5 a kurz-on-u_p request carries no
    barrier (B5), MB7 the ALT shadow loses the bool of a tuple (B7).
"""
import os
import subprocess
import sys

DN = "python/flliper/srt/pdflip/decision_numbers.py"
DR = "python/flliper/srt/pdflip/decision_record.py"
FR = "python/flliper/srt/pdflip/front.py"
PP = "python/flliper/srt/pdflip/phase_policy.py"
TESTS = ["test/registered/unit/pdflip/test_ez_s1_review_1019.py",
         "test/registered/unit/pdflip/test_decision_numbers_1019.py",
         "test/registered/unit/pdflip/test_decision_record_1019.py",
         "test/registered/unit/pdflip/test_measure_on_d_1019.py"]

MUTANTS = [
    ("M1 u_p >= halbX classifies half/big", DN,
     "    if u_p is not None and n_from_d_ids and int(u_p.value) < int(half_x):\n"
     "        return ClassBasis(KIND_P_UPPER, int(u_p.value), u_p.src, CLASS_KURZ,",
     "    if u_p is not None and n_from_d_ids:\n"
     "        return ClassBasis(KIND_P_UPPER, int(u_p.value), u_p.src, classify(u_p.value, x_tokens, half_x),"),
    ("M2 UNKNOWN -> kurz", DN,
     '    return ClassBasis(KIND_MEASURE, None, SRC_NONE, None, why + ": D measures")',
     '    return ClassBasis(KIND_P_UPPER, 0, SRC_NONE, CLASS_KURZ, why + ": D measures")'),
    ("M3 u_p instead of the fresh exact u_d", DN,
     "    if u_d is not None:\n        age = u_d.age_ms(now)",
     "    if False and u_d is not None:\n        age = u_d.age_ms(now)"),
    ("M4 the barrier is ~X instead of halbX-1", DN,
     "    return max(0, int(half_x) - 1)",
     "    return max(0, int(half_x) * 2 - 1)"),
    ("M10 commit without the epoch check", DN,
     '    if int(getattr(front, "epoch", 0)) != int(snap.epoch):\n        return False',
     '    if False:\n        return False'),
    ("KV1 a stale KV reading counts as a number", DN,
     "    if reading is None or age > int(fresh_ms):",
     "    if reading is None:"),
    ("KV2 the KV refresh runs for a sleeping D", DN,
     "    if refresh_ms <= 0 or not d_awake or fetch_pending:",
     "    if refresh_ms <= 0 or fetch_pending:"),
    ("KV3 no KV read at the end of P->D", FR,
     '        return Front._kv_fetch_kick(self, "flip_end") if dst == "D" else False',
     "        return False"),
    ("N1 the unpriced media n is exact again", FR,
     '            if (envs.FLLIPER_PDFLIP_EXACT_CLASS.get() and getattr(_xx, "unpriced", False)',
     '            if (False and getattr(_xx, "unpriced", False)'),
    ("PB1 the price barrier reads the chars/3 price", FR,
     '        if envs.FLLIPER_PDFLIP_EXACT_CLASS.get():\n'
     '            # EZ-S1 (plan S1, user 19.10. "nichts schaetzen"): the barrier reads NO chars/3 price any more',
     '        if False:\n'
     '            # EZ-S1 (plan S1, user 19.10. "nichts schaetzen"): the barrier reads NO chars/3 price any more'),
    ("R1 no edge trigger (the log floods)", DR,
     "                if last == sig:",
     "                if False and last == sig:"),
    ("R2 the ALT shadow swallows the old site's value", DR,
     "                _shadow(self, __m, __k, res, __r(a, kw), recorder_of)\n"
     "                return res\n"
     "            _sw._ez_alt_marker = marker",
     "                _shadow(self, __m, __k, res, __r(a, kw), recorder_of)\n"
     "                return None\n"
     "            _sw._ez_alt_marker = marker"),
    ("R3 SET-VERDICT no longer the H91 bytes", DR,
     '"seats_free=%d t_half_s=%.1f waited_s=%.1f verdict=%s why=%s" % (',
     '"seats_free=%d t_half_s=%.2f waited_s=%.1f verdict=%s why=%s" % ('),
    ("R4 a measure request gets D_NOW with u=0", FR,
     '        if (rid not in (self.__dict__.get("_measure_on_d") or ())\n'
     '                and phase_policy.classify_arrival',
     '        if (True\n'
     '                and phase_policy.classify_arrival'),
    ("R5 no barrier header", FR,
     '        if _meas is not None and not _after_p and envs.FLLIPER_PDFLIP_EXACT_CLASS.get():',
     '        if False:'),
    ("R6 an under-halbX report is no defect", FR,
     "            under = klass == _dn.CLASS_KURZ",
     "            under = False"),
    ("R7 the context gate writes no decision", FR,
     '        Front._dec_gate(self, "Gate-CTX", rid,',
     '        0 and Front._dec_gate(self, "Gate-CTX", rid,'),
    ("R8 the boot flip pair writes no decision", FR,
     '_b1 = Front._dec_flip(self, "boot", "FLIP_D", "NS:BOOT",',
     '_b1 = 0 and Front._dec_flip(self, "boot", "FLIP_D", "NS:BOOT",'),
    ("MB1 a MEASURE returnee is p_only by state", FR,
     "(u < xq and not _meas)",
     "(u < xq)"),
    ("MB1b no needs_p exemption for a measured kurz/halb entry", PP,
     '        if getattr(p, "measure_class", "") in ("kurz", "halb"):',
     "        if False:"),
    ("MB2 the x_exact fallback classifies on chars/3", FR,
     "            if _xx is None and envs.FLLIPER_PDFLIP_EXACT_CLASS.get():\n                # EZ-S1 review B2",
     "            if False and _xx is None and envs.FLLIPER_PDFLIP_EXACT_CLASS.get():\n                # EZ-S1 review B2"),
    ("MB3 under is judged on today's halbX", FR,
     "            defect = hx_bar is not None and u < hx_bar",
     "            defect = u < hx"),
    ("MB3b the barrier is X at depth 0", FR,
     "                x = min(x, Front._x_min_over_depths(self, int(n_prompt)))",
     "                x = x"),
    ("MB4 the barrier is armed inside the instrument except", FR,
     "        if measure or p_upper:\n            Front._measure_arm(",
     "        if False:\n            Front._measure_arm("),
    ("MB5 a kurz-on-u_p request carries no barrier", FR,
     "            p_upper = _cb0 is not None and _cb0.kind == _dn.KIND_P_UPPER",
     "            p_upper = False"),
    ("MB7 the ALT shadow loses the bool of a tuple", DR,
     "    if isinstance(result, tuple) and result and isinstance(result[0], bool):",
     "    if False:"),
]


def run_test(wt, rel):
    cmd = ["systemd-run", "--scope", "-q", "-p", "MemoryMax=2G",
           "-E", "PYTHONPATH=python:test/registered/unit/pdflip", "-E", "CUDA_VISIBLE_DEVICES=",
           "-E", "PYTHONDONTWRITEBYTECODE=1",
           "/spinning/shvllm/.venv/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--color=no", "-x", rel]
    p = subprocess.run(cmd, cwd=wt, capture_output=True, text=True, timeout=900)
    tail = [ln for ln in (p.stdout + p.stderr).splitlines() if " passed" in ln or " failed" in ln or " error" in ln]
    return p.returncode, (tail[-1].strip() if tail else "")


def main(wt):
    for rel in TESTS:
        rc, line = run_test(wt, rel)
        print(f"BASE {os.path.basename(rel)} rc={rc} {line}", flush=True)
        if rc != 0:
            print("base not green -- abort")
            return 1
    bad = 0
    for name, rel, old, new in MUTANTS:
        path = os.path.join(wt, rel)
        orig = open(path, encoding="utf-8").read()
        if orig.count(old) != 1:
            print(f"{name}: ANCHOR NOT UNIQUE ({orig.count(old)}) -- mutant not applied", flush=True)
            bad += 1
            continue
        try:
            open(path, "w", encoding="utf-8").write(orig.replace(old, new))
            killed_by = None
            for t in TESTS:
                rc, line = run_test(wt, t)
                if rc != 0:
                    killed_by = f"{os.path.basename(t)}: {line}"
                    break
            print(f"{name}: {'RED (killed) ' + killed_by if killed_by else 'SURVIVED'}", flush=True)
            if not killed_by:
                bad += 1
        finally:
            open(path, "w", encoding="utf-8").write(orig)
    print(f"SURVIVORS/BAD: {bad} of {len(MUTANTS)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.getcwd()))
