"""GROUP-SEG-1022 mutants (#157, danger directions of the segmented group tokenization). Usage:

    PDFLIP_TEST_NF_CKPT=<nf tokenizer dir> python mutants_nf_rc34_group_seg_1022.py <worktree>

Each mutant is one or more exact text replacements in <worktree>; ``test_nf_rc34_group_seg_1022.py`` must go red.
STRICTLY SERIAL: one pytest at a time through /spinning/gpu-arb/pytest_gedeckelt.sh, in the foreground. The
source files are restored byte for byte after every mutant (also on Ctrl-C).

Danger directions:
  * M1 falsche IDs bei Praefix-Abweichung: the piece cache keyed by the piece's HEAD (a message changed late
    in its text keeps its old ids);
  * M2 P/D unterschiedliche IDs: the decode cache keyed by the piece's LENGTH (a warm group returns another
    piece's text of the same length -- a cold group does not);
  * M3 the decode cut off the marker (fixed-width pieces) and its load check skipped: a multi-byte character
    split across a cut decodes differently;
  * M4 the decode cut off the marker with the load check in place: the check must refuse it (not armed);
  * M5 the tokenizer manager's encode not segmented (the base wiring: the re-encode stays whole);
  * M6 serving_chat's decode not segmented (the base wiring);
  * M7 serving_chat's encode not segmented (the base wiring).
"""
import os
import subprocess
import sys

GS = "python/flliper/srt/pdflip/group_seg_tokens.py"
FTK = "python/flliper/srt/pdflip/front_tokens.py"
TMG = "python/flliper/srt/managers/tokenizer_manager.py"
SC = "python/flliper/srt/entrypoints/openai/serving_chat.py"
TEST = "test/registered/unit/pdflip/test_nf_rc34_group_seg_1022.py"

_CUTS = "        return [0] + [p for p in pos if p > 0] + [int(ids.size)]\n"
_CUTS_FIXED = "        return [0] + list(range(5, int(ids.size), 5)) + [int(ids.size)]\n"

MUTANTS = [
    # M1 / M2 run against their danger test ALONE (-k): it must kill them by itself
    ("M1 piece cache keyed by the piece's head (prefix deviation)", "prefix_deviation",
     [(FTK, "                key = (_sha(seg), kwkey)\n", "                key = (_sha(seg[:64]), kwkey)\n")]),
    ("M2 decode cache keyed by the piece's length (P/D differ)", "p_and_d_get_the_same_ids",
     [(GS, "            key = (_sha_ids(piece), kwkey)\n", "            key = (str(int(piece.size)), kwkey)\n")]),
    ("M3 decode cut off the marker, load check skipped", None,
     [(GS, _CUTS, _CUTS_FIXED),
      (GS, "                self.dec_ok, self.dec_why = self._check_decode()\n",
       "                self.dec_ok, self.dec_why = True, \"unchecked\"\n")]),
    ("M4 decode cut off the marker (the check must refuse it)", None, [(GS, _CUTS, _CUTS_FIXED)]),
    ("M5 tokenizer manager encode whole (base wiring)", None,
     [(TMG, "                input_ids = [_gseg.seg_encode_call(self.tokenizer, tokenizer_input[0])]\n",
       "                input_ids = [list(self.tokenizer(tokenizer_input)[\"input_ids\"][0])]\n")]),
    ("M6 serving_chat decode whole (base wiring)", None,
     [(SC, "                prompt = pdflip_group_seg.seg_decode(\n                    self.tokenizer_manager.tokenizer, prompt_ids\n                )\n",
       "                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)\n")]),
    ("M7 serving_chat encode whole (base wiring)", None,
     [(SC, "                prompt_ids = pdflip_group_seg.seg_encode(\n                    self.tokenizer_manager.tokenizer, rendered_prompt, **encode_kwargs\n                )\n            except Exception:\n",
       "                prompt_ids = self.tokenizer_manager.tokenizer.encode(\n                    rendered_prompt, **encode_kwargs\n                )\n            except Exception:\n")]),
]


def run_tests(wt, k=None):
    env = dict(os.environ, PYTEST_PYTHONPATH="python:test/registered/unit/weg2:test/registered/unit/pdflip",
               PYTHONDONTWRITEBYTECODE="1")
    cmd = ["/spinning/gpu-arb/pytest_gedeckelt.sh", "--color=no", "-x", "-k", k or "not runtime", TEST]
    p = subprocess.run(cmd, cwd=wt, capture_output=True, text=True, timeout=900, env=env)
    out = p.stdout + p.stderr
    tail = [ln for ln in out.splitlines() if " passed" in ln or " failed" in ln]
    first = [ln for ln in out.splitlines() if ln.startswith("FAILED")]
    return p.returncode, (tail[-1].strip() if tail else ""), (first[0][:220] if first else "")


def main(wt):
    rc, line, _ = run_tests(wt)
    print(f"BASE rc={rc} {line}", flush=True)
    if rc != 0:
        print("base not green -- abort")
        return 1
    bad = 0
    for name, k, edits in MUTANTS:
        saved = {}
        try:
            ok = True
            for rel, old, new in edits:
                path = os.path.join(wt, rel)
                if rel not in saved:
                    saved[rel] = open(path, "rb").read()
                text = open(path, "rb").read().decode()
                if text.count(old) != 1:
                    print(f"{name}: ANCHOR count={text.count(old)} in {rel} -- not applied", flush=True)
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
        killed = rc != 0
        bad += 0 if killed else 1
        print(f"{name}: {'KILLED' if killed else 'SURVIVED'} rc={rc} {line} {first}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")))
