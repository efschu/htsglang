#!/usr/bin/env python3
"""planer_golden_regen.py <old tree> <renamed tree>  -- F0-D (08.10.2026): the dry-run goldens of the planner reference for the renamed tree.

planer_fixture_sync.py converts the goldens with the kit's name rule; that is not enough for the launcher DUMP goldens (`golden/plan_*.txt`):
(1) text the launcher copies out of evidence files (census/record texts: `(WEG2-DC per-process minus model tensors)`, boot tags in a
provenance clause) keeps its old spelling in the dump, but the rule renames it in the golden; (2) the translated launcher log lines (step 3
of the kit: `keine Experten-Karte` -> `no experts card`) differ from the German golden lines. The test (`test_planer_referenz_n3_1006`)
compares golden and dump line by line, so the goldens must say what the renamed tree prints.

Method, with the evidence it needs: both trees dump the reference profiles on the replayed inventory (the test's own helpers). The golden
of the OLD tree is aligned with the old dump (difflib): a line that is equal in both is STABLE (a line the reference proves); a line that is
not equal is LIVE-BOX state (measured records and cache paths of today's box, the 6 TestDryRunGolden tests are red on the old tree for exactly
these). The new golden takes, for every STABLE line, the renamed tree's dump line at the aligned position (alignment old dump -> new dump on
the lines with names and numbers masked; an unaligned line is reported and falls back to the rule rename) and, for every LIVE line, the rule
renamed old golden line unchanged (it stays as red as it was). So: new golden vs new dump differ in exactly the lines in which old golden
and old dump differed, and nowhere else.

usage: planer_golden_regen.py <old tree> <renamed tree> [--dry]     (HOME of the caller = the private test HOME; python of the caller)
Prints per golden: lines, stable, live, translated-or-renamed stable lines, unaligned; exit 3 on an unaligned stable line."""
import difflib, importlib, os, re, subprocess, sys, json

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
import rename_to_flliper as R

PROFILES = [("27b-base", "plan_27b_flip_n3.txt"), ("27b-nvfp4-dual", "plan_27b_dual_n3.txt"), ("nf-int4-h6-abl", "plan_nf_abl_n3.txt")]
MASK = re.compile(r"WEG2|PDFLIP|Weg2|PdFlip|SGLANG|FLLIPER|weg2|pdflip|sglang|flliper")


def masked(l):
    return re.sub(r"[0-9]+(\.[0-9]+)?", "N", MASK.sub("NAME", l))


def dumps(tree, sub):
    """child process: the test modules of `tree` dump the reference profiles; returns {golden name: (golden text, dump text)}"""
    code = (
        "import importlib,sys,json;t,s=sys.argv[1:3];sys.path.insert(0,t+'/python');sys.path.insert(0,t+'/test/registered/unit/'+s)\n"
        "m=importlib.import_module('test_planer_referenz_n3_1006');o={}\n"
        "for p,g in %r:\n"
        "    r=m._dump_of(p,m.O.read_replay(m.REPLAY_REF));o[g]=[m._read(m._golden_file(g)),r.result.dump()]\n"
        # the live Dual snapshot of AP-J (test_planer_abnahme_1006.TestLiveDualReference): the proposal's dry run, a named launcher refusal
        "ab=importlib.import_module('test_planer_abnahme_1006');ab.setUpModule();A=ab.APE;O=ab.O\n"
        "v=A._propose('ref3',profile=ab.LIVE);b=A._dual_profile(ab.LIVE)\n"
        "li=O.LaunchInput(v['argv'],v['env'],b.vars,[],'propose:'+v['basis'],b.instruments)\n"
        "res=O.run_profile('',A._ref_rows(),tree=A.TREE,force=False,launch_input=li).result\n"
        "o['plan_27b_dual_live1521_n3.txt']=[open(ab.GOLDEN_TXT,encoding='utf-8').read(),res.dump()]\n"
        "print('@@JSON@@'+json.dumps(o))\n" % (PROFILES,))
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code, tree, sub], capture_output=True, text=True, cwd=tree,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=""))
    line = [l for l in out.stdout.splitlines() if l.startswith("@@JSON@@")]
    if not line:
        sys.exit("dump failed in %s: %s" % (tree, out.stderr[-1500:]))
    return json.loads(line[0][8:])


old_tree, new_tree = sys.argv[1], sys.argv[2]
DRY = "--dry" in sys.argv
imap = R._load_imap(os.path.join(KIT, "data", "merged_0928.json"))
OLD = dumps(old_tree, "weg2")
NEW = dumps(new_tree, "pdflip")
bad = 0
for g in OLD:
    G, D = [x.split("\n") for x in OLD[g]]
    D2 = NEW[g][1].split("\n")
    rule = [R.rewrite_all(l, False, True, imap)[0] for l in G]
    # old golden vs old dump: stable = equal lines
    sm = difflib.SequenceMatcher(None, G, D, autojunk=False)
    # old dump -> new dump alignment on masked lines
    a2 = difflib.SequenceMatcher(None, [masked(x) for x in D], [masked(x) for x in D2], autojunk=False)
    jmap, tmap = {}, set()
    for tag, i1, i2, j1, j2 in a2.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):      # a replaced block of equal size = the same lines, translated
            for k in range(i2 - i1):
                jmap[i1 + k] = j1 + k
                if tag == "replace":
                    tmap.add(i1 + k)
    out, stable, live, changed, unaligned = [], 0, 0, 0, 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                j = j1 + k
                if j in jmap:
                    nl = D2[jmap[j]]
                    stable += 1
                    changed += (j in tmap)
                    out.append(nl)
                else:
                    unaligned += 1
                    out.append(rule[i1 + k])
        else:
            live += i2 - i1
            out.extend(rule[i1:i2])
    bad += unaligned
    path = os.path.join(new_tree, "test/registered/unit/pdflip/fixtures/planer_1006/golden", g)
    if not DRY:
        open(path, "w", encoding="utf-8").write("\n".join(out))
    print("%-24s lines=%d stable=%d live=%d stable-lines-translated-or-reworded=%d unaligned=%d" % (g, len(out), stable, live, changed, unaligned))
sys.exit(3 if bad else 0)
