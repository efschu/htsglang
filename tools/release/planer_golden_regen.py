#!/usr/bin/env python3
"""planer_golden_regen.py <old tree> <renamed tree>  -- F0-D (08.10.2026, fix round 1): the dry-run goldens of the planner reference for the renamed tree.

planer_fixture_sync.py converts the goldens with the kit's NAME rule.  That is not enough for the launcher DUMP goldens (`golden/plan_*.txt`):
the launcher prints text whose wording the kit changed in steps 2-3 (ident_fix words inside strings, the German -> English translation of the
log lines), and a name rule on the golden knows nothing about that.  `test_planer_referenz_n3_1006` compares golden and dump line by line, so the
golden must say what the renamed tree prints.

Reference state.  Both trees dump the reference profiles on the replayed inventory (the test's own helpers) with a private EMPTY `$HOME`
(the state S0: no card library, no corridor-floor pointer, no measured records).  The golden of the OLD tree was recorded on the live box of
2026-10-06; against S0 it is aligned with the old dump (difflib): a line equal in both is STABLE (a line the reference proves at S0), a line that
is not equal is LIVE-BOX state (card library, floor pointer, measured records of that box; they are what makes the 6 TestDryRunGolden tests red
on the old tree on any other box).

New golden, line by line.
  * STABLE line: the renamed tree's dump line at the aligned position (alignment old dump -> new dump on the lines with names and numbers masked;
    an unaligned stable line is reported and exits 3).  This is the renamed launcher's own output, including the recomputed `plan_id` hashes.
  * LIVE line: the old golden line with the SAME rewrite the kit applied to the code.  The kit's rewrite is read from the code itself: the
    string constants of every module of the old tree are paired with the constants at the same AST position of the renamed module (the passes
    keep the structure: ~3650 modules, a position mismatch is counted and reported); every constant that differs is a fragment (old text ->
    new text), %-/{}-placeholders split a format string into its static segments.  The golden line is rewritten with these fragments (longest
    first, one pass, word-bounded), then `/records/weg2` -> `/records/pdflip` (the one host-path spelling the renamed launcher builds new),
    then the kit's name rule over the rest.  What must NOT be touched is text the launcher copies out of evidence files (boot tags,
    `WEG2-DC` in a record's wording, the census tool's path): the renamed launcher still prints such tokens with an old name, so every token
    with an old name in the renamed dumps is lifted out of the name rule, and a fragment whose old text the renamed dumps still print
    verbatim is not applied (dump guard).  The method is VALIDATED on the STABLE lines, where the answer is known: for each of them the
    method's output (guards from the OTHER goldens only) is compared with the renamed dump line (printed as `check`; --verbose lists every
    disagreement).  Known limits, all in the live lines only: a recomputed `plan_id` hash cannot be derived by any rewrite, a sorted list
    (the env listing) can change its order with the names, a German word of a constant shorter than MIN_FRAGMENT stays German.

So: new golden vs new dump (at S0) differ in exactly the lines in which old golden and old dump differed, and nowhere else; and the live lines
carry the wording the renamed code can print.  A golden line that depends on state S0 does not have cannot be proven by any tool.

usage: planer_golden_regen.py <old tree> <renamed tree> [--dry] [--verbose] [--cache DIR]
       (--cache DIR keeps the two dump runs, the dumps are the slow part: ~2 min each)
Prints per golden: lines, stable, live, stable-lines-translated-or-reworded, unaligned, fragment-check agree/disagree; exit 3 on an unaligned stable line."""
import ast, collections, difflib, glob, json, os, re, subprocess, sys, tempfile

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
import rename_to_flliper as R

PROFILES = [("27b-base", "plan_27b_flip_n3.txt"), ("27b-nvfp4-dual", "plan_27b_dual_n3.txt"), ("nf-int4-h6-abl", "plan_nf_abl_n3.txt")]
MASK = re.compile(r"WEG2|PDFLIP|Weg2|PdFlip|SGLANG|FLLIPER|weg2|pdflip|sglang|flliper")
PLACEHOLDER = re.compile(r"%(?:\([^)]*\))?[-+ #0]*(?:\d+|\*)?(?:\.(?:\d+|\*))?[sdrfxXeEgGiuc]|%%|\{[^{}]*\}")
# Fragments shorter than this are never applied (measured on the 967 stable lines of the three goldens + the live Dual one, 28.08. cache): the
# kit also translated short constants (dict keys, one-word labels: `nicht`, `die`, `Spitze`), and the same German words stand in text the
# launcher copies out of evidence files, where they stay German.  Two guards, both measured on the 967 stable lines of the four goldens
# (agreement of the method with the renamed dump, the guard text taken from the OTHER goldens only): a minimum length (3 chars: 795/967,
# 8: 919, 14: 925, 24: 928) and the dump guard (a fragment whose old text the renamed launcher still prints verbatim is not applied:
# 3 chars 910, 8: 938, 14: 940).  What the guards leave German in a LIVE line is a miss on the side of "unchanged", never a corrupted word.
MIN_FRAGMENT = 14


def masked(l):
    return re.sub(r"[0-9]+(\.[0-9]+)?", "N", MASK.sub("NAME", l))


def dumps(tree, sub, home):
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
                         env=dict(os.environ, HOME=home, CUDA_VISIBLE_DEVICES="", PYTHONPATH=""))
    line = [l for l in out.stdout.splitlines() if l.startswith("@@JSON@@")]
    if not line:
        sys.exit("dump failed in %s: %s" % (tree, out.stderr[-1500:]))
    return json.loads(line[0][8:])


# ---------------------------------------------------------------- the kit's rewrite, read from the code

def _old_path(new_rel):
    """python/flliper/srt/pdflip/pdflip_x.py -> python/sglang/srt/weg2/weg2_x.py (the mechanical rule; German file stems of python/ did not move)"""
    p = new_rel.replace("python/flliper", "python/sglang", 1).replace("/pdflip/", "/weg2/")
    d, b = os.path.split(p)
    return os.path.join(d, "weg2_" + b[len("pdflip_"):]) if b.startswith("pdflip_") else p


def _pair(o, n, out, st):
    if type(o) is not type(n):
        st["mismatch"] += 1
        return
    if isinstance(o, ast.Constant):
        if isinstance(o.value, str) and isinstance(n.value, str) and o.value != n.value:
            out.append((o.value, n.value))
        return
    oc, nc = list(ast.iter_child_nodes(o)), list(ast.iter_child_nodes(n))
    if len(oc) != len(nc):
        st["mismatch"] += 1
        return
    for a, b in zip(oc, nc):
        _pair(a, b, out, st)


def _split(old, new, out):
    """a format string pair -> the pairs of its static segments (same placeholders in the same order), else the whole pair"""
    po, pn = PLACEHOLDER.findall(old), PLACEHOLDER.findall(new)
    if po and po == pn:
        for a, b in zip(PLACEHOLDER.split(old), PLACEHOLDER.split(new)):
            if a != b:
                out.append((a, b))
    else:
        out.append((old, new))


def fragment_table(old_tree, new_tree):
    """{old fragment: new fragment} from the string constants of the old tree against the renamed one; plus statistics"""
    table = collections.defaultdict(collections.Counter)
    st = collections.Counter()
    for p in sorted(glob.glob(new_tree + "/python/flliper/**/*.py", recursive=True)):
        rel = os.path.relpath(p, new_tree)
        op = os.path.join(old_tree, _old_path(rel))
        if not os.path.exists(op):
            st["no-old-file"] += 1
            continue
        try:
            to = ast.parse(open(op, encoding="utf-8").read())
            tn = ast.parse(open(p, encoding="utf-8").read())
        except (SyntaxError, ValueError):
            st["parse-error"] += 1
            continue
        pairs = []
        _pair(to, tn, pairs, st)
        st["modules"] += 1
        for a, b in pairs:
            segs = []
            _split(a, b, segs)
            for x, y in segs:
                if len(x.strip()) >= MIN_FRAGMENT and x != y:
                    table[x][y] += 1
    amb = {a: c for a, c in table.items() if len(c) > 1}
    st["fragments"] = len(table)
    st["ambiguous"] = len(amb)
    return {a: c.most_common(1)[0][0] for a, c in table.items() if len(c) == 1}, st


TOKEN = re.compile(r"[^\s,;:()\[\]'\"=<>{}]+")


def old_name_tokens(texts):
    """Tokens that still spell an OLD name in what the renamed launcher prints.  The renamed code prints no old name of its own, so every
    such token is text the launcher copied out of an evidence file (boot tags `boot_weg2_...`, `WEG2-DC` in a record's wording, host paths
    `/spinning/gpu-arb/weg2/...`, the census tool's path): a rewrite of the golden must not touch it."""
    out = set()
    for t in texts:
        for tok in TOKEN.findall(t):
            if MASK.search(tok):
                out.add(tok)
    return out


def dump_guard(table, dump_text):
    """the fragments whose OLD text is not printed verbatim (word-bounded) by the renamed launcher: text it still prints is copied from an
    evidence file or left untranslated, a rewrite of the golden must not turn it into the English of an unrelated constant"""
    def bound(k):
        s = re.escape(k)
        if re.match(r"\w", k[0]):
            s = r"(?<![A-Za-z0-9_])" + s
        if re.match(r"\w", k[-1]):
            s = s + r"(?![A-Za-z0-9_])"
        return s
    return {k: v for k, v in table.items() if not re.search(bound(k), dump_text)}


def protect_rx(tokens):
    toks = sorted(tokens, key=lambda k: (-len(k), k))
    return re.compile("|".join(re.escape(k) for k in toks)) if toks else None


class Rewriter:
    def __init__(self, table, imap):
        self.table, self.imap = table, imap
        keys = sorted(table, key=lambda k: (-len(k), k))

        def bound(k):                # a fragment that starts / ends with a word character never matches inside a longer word
            s = re.escape(k)
            if re.match(r"\w", k[0]):
                s = r"(?<![A-Za-z0-9_])" + s
            if re.match(r"\w", k[-1]):
                s = s + r"(?![A-Za-z0-9_])"
            return s
        self.rx = re.compile("|".join(bound(k) for k in keys)) if keys else None

    def line(self, text, protected=None):
        """fragments first, then the name rule over the rest; `protected` (a compiled alternation) is lifted out before the name rule"""
        t = self.rx.sub(lambda m: self.table[m.group(0)], text) if self.rx else text
        t = t.replace("/records/weg2", "/records/pdflip")      # the one host-path spelling the renamed launcher builds new (planer_fixture_sync.PINS)
        hold = {}
        if protected is not None:
            def lift(m):
                k = "\x00%d\x00" % len(hold)
                hold[k] = m.group(0)
                return k
            t = protected.sub(lift, t)
        t = R.rewrite_all(t, False, True, self.imap)[0]
        for k, v in hold.items():
            t = t.replace(k, v)
        return t


def main():
    old_tree, new_tree = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    DRY, VERBOSE = "--dry" in sys.argv, "--verbose" in sys.argv
    cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else None
    imap = R._load_imap(os.path.join(KIT, "data", "merged_0928.json"))
    table, st = fragment_table(old_tree, new_tree)
    print("fragments: %s" % dict(st))

    def get(tree, sub, tag):
        f = os.path.join(cache, tag + ".json") if cache else None
        if f and os.path.exists(f):
            return json.load(open(f))
        with tempfile.TemporaryDirectory(prefix="golden-regen-home-") as home:
            d = dumps(tree, sub, home)
        if f:
            os.makedirs(cache, exist_ok=True)
            json.dump(d, open(f, "w"))
        return d

    OLD = get(old_tree, "weg2", "old")
    NEW = get(new_tree, "pdflip", "new")
    per = {g: old_name_tokens([NEW[g][1]]) for g in NEW}
    rw_live = Rewriter(dump_guard(table, "\n".join(NEW[g][1] for g in NEW)), imap)
    print("fragments after the dump guard (live lines): %d of %d" % (len(rw_live.table), len(table)))
    all_tokens = set().union(*per.values())
    print("old-name tokens the renamed launcher still prints (evidence text): %d" % len(all_tokens))
    bad = 0
    for g in OLD:
        G, D = [x.split("\n") for x in OLD[g]]
        D2 = NEW[g][1].split("\n")
        # old dump -> new dump alignment on masked lines
        a2 = difflib.SequenceMatcher(None, [masked(x) for x in D], [masked(x) for x in D2], autojunk=False)
        jmap, tmap = {}, set()
        for tag, i1, i2, j1, j2 in a2.get_opcodes():
            if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):      # a replaced block of equal size = the same lines, translated
                for k in range(i2 - i1):
                    jmap[i1 + k] = j1 + k
                    if tag == "replace":
                        tmap.add(i1 + k)
        sm = difflib.SequenceMatcher(None, G, D, autojunk=False)       # old golden vs old dump: stable = equal lines
        out, stable, live, changed, unaligned, agree, disagree = [], 0, 0, 0, 0, 0, []
        # the check protects only the tokens of the OTHER goldens' dumps (an honest test: a golden's own dump is the answer it is graded on);
        # the live lines may use every token the renamed launcher prints
        others = [o for o in NEW if o != g]
        rw_check = Rewriter(dump_guard(table, "\n".join(NEW[o][1] for o in others)), imap)
        check_rx, live_rx = protect_rx(set().union(*[per[o] for o in others])), protect_rx(all_tokens)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    j = j1 + k
                    if j in jmap:
                        nl = D2[jmap[j]]
                        stable += 1
                        changed += (j in tmap)
                        out.append(nl)
                        if rw_check.line(G[i1 + k], check_rx) == nl:                  # the fragment method against the known answer
                            agree += 1
                        else:
                            disagree.append((G[i1 + k], rw_check.line(G[i1 + k], check_rx), nl))
                    else:
                        unaligned += 1
                        out.append(rw_live.line(G[i1 + k], live_rx))
            else:
                live += i2 - i1
                out.extend(rw_live.line(l, live_rx) for l in G[i1:i2])
        bad += unaligned
        path = os.path.join(new_tree, "test/registered/unit/pdflip/fixtures/planer_1006/golden", g)
        if not DRY:
            open(path, "w", encoding="utf-8").write("\n".join(out))
        print("%-34s lines=%d stable=%d live=%d stable-lines-translated-or-reworded=%d unaligned=%d check(fragment method vs renamed dump on stable lines)=%d/%d"
              % (g, len(out), stable, live, changed, unaligned, agree, stable))
        plan_ids = sum("plan_id=sha256:" in c for _, _, c in disagree)
        if disagree:
            print("   check-disagree classes: plan_id hash recomputed by the renamed launcher %d, other %d "
                  "(text copied from evidence files keeps its old spelling in the dump; --verbose lists them)" % (plan_ids, len(disagree) - plan_ids))
        for a, b, c in (disagree if VERBOSE else []):
            print("   check-disagree:\n     golden   %s\n     method   %s\n     renamed  %s" % (a[:200], b[:200], c[:200]))
    sys.exit(3 if bad else 0)


if __name__ == "__main__":
    main()
