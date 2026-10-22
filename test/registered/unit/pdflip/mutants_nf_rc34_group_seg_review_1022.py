"""GROUP-SEG-1022 review mutants (#157). Usage:

    PDFLIP_TEST_NF_CKPT=<nf tokenizer dir> python mutants_nf_rc34_group_seg_review_1022.py <worktree>

Each mutant is exact text replacements in <worktree>; its named test in ``test_nf_rc34_group_seg_review_1022.py``
must go red ALONE (-k). STRICTLY SERIAL through /spinning/gpu-arb/pytest_gedeckelt.sh; sources restored byte for
byte after every mutant.

  R1 decode key hashed as int32 again (id x + 2**32 answered from x's cached text);
  R2 the structural overlap check removed (an added token across the cut arms the encode);
  R3 the decode cache's entry bound removed;  R4 the encode cache's entry bound removed;
  R5 the encode fallback removed (a failing segmented call raises);
  R6 a cut that is not the marker's (before every CR as well) with the load check off: the property test alone.
"""
import os
import subprocess
import sys

GS = "python/flliper/srt/pdflip/group_seg_tokens.py"
FTK = "python/flliper/srt/pdflip/front_tokens.py"
TEST = "test/registered/unit/pdflip/test_nf_rc34_group_seg_review_1022.py"

MUTANTS = [
    ("R1 decode key int32", "int32",
     [(GS, "np.ascontiguousarray(ids, dtype=np.int64).tobytes()", "np.ascontiguousarray(ids, dtype=np.int32).tobytes()")]),
    ("R2 overlap check removed", "overlapping",
     [(GS, "            why = _marker_overlap(self.tok, SEGMENT_MARKER)\n", "            why = None\n")]),
    ("R3 decode entry bound removed", "entry_bound",
     [(GS, "self.dec_tokens > self.dec_cap or len(self.dec_cache) > self.cap_entries", "self.dec_tokens > self.dec_cap")]),
    ("R4 encode entry bound removed", "entry_bound",
     [(FTK, "or (self.cap_entries is not None and len(self.cache) > self.cap_entries)", "or False")]),
    ("R5 encode fallback removed", "failing_segmented",
     [(GS, "            self._fell_back(\"encode\", e)\n            return self.tok.encode(text, **kw)\n",
       "            raise\n")]),
    ("R6 cut before CR too, load check off", "property",
     [(FTK, "        out.extend(self.marker + p for p in parts[1:])\n        return out\n",
       "        out.extend(self.marker + p for p in parts[1:])\n"
       "        return [q for o in out for q in __import__('re').split(r'(?=\\r)', o) if q]\n"),
      (GS, "            self.enc_ok, self.enc_why = self._check_encode()\n",
       "            self.enc_ok, self.enc_why = True, 'unchecked'\n")]),
]


def run_tests(wt, k=None):
    env = dict(os.environ, PYTEST_PYTHONPATH="python:test/registered/unit/weg2:test/registered/unit/pdflip",
               PYTHONDONTWRITEBYTECODE="1")
    cmd = ["/spinning/gpu-arb/pytest_gedeckelt.sh", "--color=no", "-x"] + (["-k", k] if k else []) + [TEST]
    p = subprocess.run(cmd, cwd=wt, capture_output=True, text=True, timeout=900, env=env)
    out = p.stdout + p.stderr
    tail = [ln for ln in out.splitlines() if " passed" in ln or " failed" in ln]
    first = [ln for ln in out.splitlines() if ln.startswith("FAILED")]
    return p.returncode, (tail[-1].strip() if tail else ""), (first[0][:200] if first else "")


def main(wt):
    rc, line, _ = run_tests(wt)
    print(f"BASE rc={rc} {line}", flush=True)
    if rc != 0:
        return 1
    bad = 0
    for name, k, edits in MUTANTS:
        saved = {}
        try:
            ok = True
            for rel, old, new in edits:
                path = os.path.join(wt, rel)
                saved.setdefault(rel, open(path, "rb").read())
                text = open(path, "rb").read().decode()
                if text.count(old) != 1:
                    print(f"{name}: ANCHOR count={text.count(old)} in {rel}", flush=True)
                    ok = False
                    break
                open(path, "w").write(text.replace(old, new, 1))
            if not ok:
                bad += 1
                continue
            rc, line, first = run_tests(wt, k)
        finally:
            for rel, src in saved.items():
                open(os.path.join(wt, rel), "wb").write(src)
        bad += 0 if rc != 0 else 1
        print(f"{name}: {'KILLED' if rc != 0 else 'SURVIVED'} rc={rc} {line} {first}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")))
