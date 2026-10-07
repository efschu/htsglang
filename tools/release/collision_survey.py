"""Item 600: list EVERY identifier collision of the mechanical pass in one tree (the tool itself exits at the first).
usage: collision_survey.py <worktree root | ref:<git-ref>>   -- read only (ref: REPO=<git-repo>, no checkout)."""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rename_to_flliper as R


def iter_tree(root):
    """A worktree directory, or 'ref:<git-ref>' = the committed tree of that ref in $REPO (default /spinning/htsglang), no checkout (F0-A)."""
    if root.startswith("ref:"):
        return R.iter_ref(os.environ.get("REPO", "/spinning/htsglang"), root[4:])
    return R.iter_root(root)

root = sys.argv[1]
imap = R._load_imap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "merged_0928.json"))
n_files = 0
coll = {}
imapcoll = {}
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
    if path.endswith(".py"):
        ic = R.ident_collisions(text, imap)
        if ic:
            imapcoll[path] = ic
print("files scanned:", n_files)
print("name-rule collisions (files):", len(coll))
for p, c in sorted(coll.items()):
    print("  C", p, json.dumps(c, sort_keys=True))
print("ident-map collisions (files):", len(imapcoll))
for p, c in sorted(imapcoll.items()):
    print("  I", p, json.dumps(c, sort_keys=True, default=str))
