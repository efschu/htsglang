"""Item 600: list EVERY identifier collision of the mechanical pass in one tree (the tool itself exits at the first).
usage: collision_survey.py <worktree root | ref:<git-ref>>   -- read only (ref: REPO=<git-repo>, no checkout).
Exemptions: COLLISION_OK_FILE (name rule, per line: data/collision_ok_1007_<line>.json) and IDENT_COLLISION_OK_FILE (ident map; default
data/ident_collision_ok_1007.json) -- the same two files the mechanical pass reads. Exit code 0 = no OPEN collision (the acceptance
"0 ungeklaerte Kollisionen"), 1 = open collisions listed, exempted ones are listed separately.
It covers BOTH collision guards of release_rename.sh: step 1 (name rule + merged_0928.json, per file) and step 2 (ident_fix.py with the
FIXMAP of the kit run: data/identfix_map.json + FIXMAP_EXTRA tables, collision rule IDENT_FIX_COLLISION, default `refined` as in
release_rename.sh). Step 2 runs on the tree AFTER step 1, so it is evaluated on the text the mechanical pass would produce.
F0-A fix round 2: without the step-2 section a survey of 0 could still stop the kit run at ident_fix (catalog.json vorlauf -> warmup)."""
import os, sys, json, re, collections
KITDIR = os.path.dirname(os.path.abspath(__file__))
_okf = os.path.join(KITDIR, "data", "ident_collision_ok_1007.json")
if not os.environ.get("IDENT_COLLISION_OK_FILE") and os.path.isfile(_okf):
    os.environ["IDENT_COLLISION_OK_FILE"] = _okf        # before the engine import: it reads the env at import time
sys.path.insert(0, KITDIR)
import rename_to_flliper as R
import ident_precheck as IP


def load_fixmap():
    """The table ident_fix.py gets in release_rename.sh: identfix_map.json, FIXMAP_EXTRA tables merged over it (later wins), `_comment` only dropped."""
    out = {}
    files = [os.path.join(KITDIR, "data", "identfix_map.json")] + os.environ.get("FIXMAP_EXTRA", "").split()
    for f in files:
        with open(f, encoding="utf-8") as fh:
            out.update({k: v for k, v in json.load(fh).items() if k != "_comment"})
    return out


def fix_names(path, text):
    """`have` set of ident_fix.py (same rule): py = NAME tokens, md = words inside backticks, other = every word."""
    if path.endswith(".py"):
        return IP.names_py(text)
    if path.endswith(".md"):
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(re.findall(r"`[^`\n]+`", text))))
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))


def iter_tree(root):
    """A worktree directory, or 'ref:<git-ref>' = the committed tree of that ref in $REPO (default /spinning/htsglang), no checkout (F0-A)."""
    if root.startswith("ref:"):
        return R.iter_ref(os.environ.get("REPO", "/spinning/htsglang"), root[4:])
    return R.iter_root(root)

root = sys.argv[1]
imap = R._load_imap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "merged_0928.json"))
FIXMAP = load_fixmap()
FIXRX = {k: re.compile(r"(?<![A-Za-z0-9_])" + re.escape(k) + r"(?![A-Za-z0-9_])") for k in FIXMAP}
FIXMODE = os.environ.get("IDENT_FIX_COLLISION", "refined")
fixcoll = {}
n_files = 0
coll = {}
imapcoll = {}
imapexempt = {}
for path, mode, data in iter_tree(root):
    if mode == "120000":
        continue
    text = R.as_text(data)
    ext = os.path.splitext(path)[1]
    if text is None or not R.in_scope(path) or ext in R.CXX_EXT:
        continue
    n_files += 1
    new_text, rep, skip = R.rewrite_all(text, path.endswith(".py"), True, imap)
    clash = R.file_collisions(text, R.rewrite_all(text, path.endswith(".py"), True, {})[0])
    ok = R.COLLISION_OK.get(path, frozenset())
    clash = {k: v for k, v in clash.items() if k not in ok}
    if clash:
        coll[path] = clash
    # step 2 (ident_fix.py) on the text after step 1: a target already present in a file the table touches
    have = None
    for k, v in FIXMAP.items():
        if FIXRX[k].search(new_text):
            if have is None:
                have = fix_names(path, new_text)
            if v in have:
                if FIXMODE == "refined":
                    names_r, exact_r = IP.refined_sets(path, new_text)
                    if not ((k in names_r and v in names_r) or (k in exact_r and v in exact_r)):
                        continue
                fixcoll.setdefault(path, []).append([k, v])
    if path.endswith(".py"):
        ic = R.ident_collisions(text, imap)
        okw = R.IDENT_COLLISION_OK.get(path, frozenset())
        if [c for c in ic if c[0] in okw]:
            imapexempt[path] = [c for c in ic if c[0] in okw]
        ic = [c for c in ic if c[0] not in okw]
        if ic:
            imapcoll[path] = ic
print("files scanned:", n_files)
print("name-rule collisions (files):", len(coll))
for p, c in sorted(coll.items()):
    print("  C", p, json.dumps(c, sort_keys=True))
print("ident-map collisions (files):", len(imapcoll))
for p, c in sorted(imapcoll.items()):
    print("  I", p, json.dumps(c, sort_keys=True, default=str))
print("ident-map collisions exempted by IDENT_COLLISION_OK_FILE (files):", len(imapexempt))
for p, c in sorted(imapexempt.items()):
    print("  E", p, json.dumps(c, sort_keys=True, default=str))
print("ident_fix (step 2, table identfix_map.json%s, rule %s) collisions (files):" % (" + FIXMAP_EXTRA" if os.environ.get("FIXMAP_EXTRA") else "", FIXMODE), len(fixcoll))
for p, c in sorted(fixcoll.items()):
    print("  F", p, json.dumps(c, sort_keys=True))
print("OPEN collisions:", len(coll) + len(imapcoll) + len(fixcoll))
sys.exit(1 if coll or imapcoll or fixcoll else 0)
