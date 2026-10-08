"""F0-A (07.10.2026): the rename kit lives in the tree (tools/release/), not in a separate repo.

Hermetic checks, no GPU, no git history: the kit's engine passes its own selftest from the in-tree path, the kit directory is
content-locked against the rename it performs, the scripts carry no path of the old kit location, and the identifier tables
are well formed.  The words of the old name are built from pieces so that this file itself passes the mechanical rename
unchanged (the same rule as the name_compat tests)."""
import json
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
KIT = os.path.join(REPO, "tools", "release")
OLD_KIT_DIR = "/spinning/" + "fl" + "liper/tools"          # the pre-F0-A kit location (a separate repo)

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(KIT, "rename_to_flliper.py")), reason="kit not in this tree")


def _rev(word):
    """German key words are written reversed in this file: ident_fix.py rewrites whole German words inside string tokens."""
    return word[::-1]


def _engine():
    sys.path.insert(0, KIT)
    try:
        import importlib
        return importlib.import_module("rename_to_flliper")
    finally:
        sys.path.remove(KIT)


def test_engine_selftest_passes_from_the_tree():
    r = subprocess.run([sys.executable, os.path.join(KIT, "rename_to_flliper.py"), "selftest"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-400:]
    assert r.stdout.strip().splitlines()[-1] == "SELFTEST PASS"
    assert "FAIL" not in r.stdout


def test_kit_and_rename_entry_are_content_locked():
    """The kit is tool input (ident table keys, translation memory, old-name regexes): its bytes must survive the pass it performs,
    while the paths still move with their directories. The dashboard's boot recordings are sha256-stamped evidence (plan_id)."""
    R = _engine()
    for p in ("tools/release/data/merged_0928.json", "tools/release/ident_fix.py", "tools/release/profconv/27b.env",
              "docker/weg2-release/rename_rigdash.py", "docker/weg2-release/rename_rigdash_collision_ok.json",
              "tools/rig_dashboard/rigdash/kartenplan_data/nf-int4-abl.json"):
        assert R.in_path_scope(p), p
        assert not R.in_scope(p), p
    assert R.in_scope("python/" + "sg" + "lang/srt/server_args.py")      # the lock is not a blanket


def test_kit_scripts_do_not_point_at_the_old_kit_location():
    bad = []
    for name in sorted(os.listdir(KIT)):
        p = os.path.join(KIT, name)
        if os.path.isfile(p) and name.endswith((".py", ".sh")):
            txt = open(p, encoding="utf-8").read()
            for m in re.finditer(re.escape(OLD_KIT_DIR) + r"[^\s\"')]*", txt):
                bad.append("%s: %s" % (name, m.group(0)))
    assert not bad, bad


def test_kit_scripts_compile():
    import py_compile
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for name in sorted(os.listdir(KIT)):
            if name.endswith(".py"):
                py_compile.compile(os.path.join(KIT, name), cfile=os.path.join(d, name + "c"), doraise=True)
    for name in sorted(os.listdir(KIT)):
        if name.endswith(".sh"):
            r = subprocess.run(["bash", "-n", os.path.join(KIT, name)], capture_output=True, text=True)
            assert r.returncode == 0, name + ": " + r.stderr


def test_identifier_tables_are_well_formed():
    data = os.path.join(KIT, "data")
    tables = {}
    for name in ("merged_0928.json", "identfix_map.json", "decided_0928.json", "ident_map_1007.json"):
        p = os.path.join(data, name)
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as fh:
            tables[name] = {k: v for k, v in json.load(fh).items() if k != "_comment"}
    assert "merged_0928.json" in tables and "identfix_map.json" in tables
    ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for name, t in tables.items():
        for k, v in t.items():
            assert ident.match(k) and ident.match(v), (name, k, v)
            assert k != v, (name, k)
        # a table may not map one word to a word that is itself renamed by the same table (a second pass would move it again)
        assert not (set(t.values()) & set(t)), (name, sorted(set(t.values()) & set(t))[:5])


def test_the_planner_key_table_follows_the_planner_decision():
    """07.10. decision of the planner seat: the JSON keys of the four planner schemas (propose, ui, bars, verdict) become English."""
    p = os.path.join(KIT, "data", "ident_map_1007.json")
    if not os.path.isfile(p):
        pytest.skip("table not in this tree")
    with open(p, encoding="utf-8") as fh:
        t = {k: v for k, v in json.load(fh).items() if k != "_comment"}
    decided = {_rev(k): v for k, v in {
        "etrew": "values",
        "tfnukreh": "source",
        "tkidrev": "verdict",
        "dnatsuz": "state",
        "gnagsua": "outcome",
        "dnurg": "reason",
        "nemrof": "forms",
        "ettinhcsba": "sections",
        "eleiz": "goals",
        "gnar_ej": "per_rank",
        "nerotkev": "vectors",
        "tredneaeg": "changed",
        "tgelebnu": "unverified",
        "esiewnih": "notes",
        "negnealrotkev": "vector_lengths",
        "egeartnie": "entries",
        "trew": "value",
        "liforp_eiw_hcielg": "same_as_profile",
        "treitaruk": "curated",
        "trealkre": "explained",
        "tetnreeg": "harvested",
        "trealkrenu": "unexplained",
        "thcsuat": "trades",
        "thcuarb": "requires",
        "sua_tsseilhcs": "excludes",
        "nov_tetielegba": "derived_from",
        "tim_treilaks": "scales_with",
    }.items()}
    for k, v in decided.items():
        assert t.get(k) == v, (k, t.get(k), v)
    for kept in ("in_argv", "blocker"):
        assert kept not in t


def test_reviewed_ident_collision_is_exempted_and_listed(tmp_path):
    """IDENT_COLLISION_OK_FILE (F0-A): an ident-map collision between two names that live in different functions aborts the pass
    unless the (path, old word) pair is reviewed and listed; the pass then renames and reports the exemption."""
    pkg = "sg" + "lang"
    old, new = _rev("nednufeg"), "hit"
    src = "def f():\n    %s = 1\n    return %s\n\n\ndef g():\n    %s = 2\n    return %s\n" % (old, old, new, new)
    root = tmp_path / "repo"
    (root / "python" / pkg).mkdir(parents=True)
    (root / "python" / pkg / "m.py").write_text(src)
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, check=True)
    git("init", "-q", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    imap = tmp_path / "map.json"
    imap.write_text(json.dumps({old: new}))
    okf = tmp_path / "ok.json"
    okf.write_text(json.dumps({"python/%s/m.py" % pkg: [old]}))
    cmd = [sys.executable, os.path.join(KIT, "rename_to_flliper.py"), "apply", "--root", str(root), "--" + "we" + "g2", "--ident-map", str(imap),
           "--manifest", str(tmp_path / "man.json")]
    env = {k: v for k, v in os.environ.items() if k != "IDENT_COLLISION_OK_FILE"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode != 0 and "ident-map collision" in (r.stdout + r.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(env, IDENT_COLLISION_OK_FILE=str(okf)), timeout=120)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    man = json.loads((tmp_path / "man.json").read_text())
    assert man["ident_collisions_allowed"] == {"python/%s/m.py" % pkg: ["%s->%s" % (old, new)]}
    out = (root / "python" / "fl" "liper" / "m.py").read_text()
    assert old not in out and out.count(new) == 4


def test_translation_memory_is_tracked_not_ignored():
    """F0-A fix round 1: the blanket `*.jsonl` rule of .gitignore ate data/tm.jsonl, so a clean checkout carried no translation memory and
    a run would silently re-translate every prose unit. The file must be tracked (or at least not ignored) and be a well-formed memory."""
    tm = os.path.join(KIT, "data", "tm.jsonl")
    assert os.path.isfile(tm)
    rows = [json.loads(l) for l in open(tm, encoding="utf-8") if l.strip()]
    assert len(rows) > 100 and all("text" in r and "translation" in r for r in rows[:50])
    if not os.path.isdir(os.path.join(REPO, ".git")) and not os.path.isfile(os.path.join(REPO, ".git")):
        pytest.skip("no git metadata")
    rel = "tools/release/data/tm.jsonl"
    ign = subprocess.run(["git", "-C", REPO, "check-ignore", "-q", rel])
    assert ign.returncode == 1, "tm.jsonl is git-ignored"
    ls = subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", rel], capture_output=True, text=True)
    assert ls.returncode == 0, "tm.jsonl is not tracked: " + ls.stderr


def test_translate_shard_refuses_a_missing_translation_memory(tmp_path):
    units = tmp_path / "u.jsonl"
    units.write_text(json.dumps({"kind": "log", "text": "\"x\""}) + "\n")
    base = [sys.executable, os.path.join(KIT, "translate_shard.py"), str(units), str(tmp_path / "o.jsonl"), str(tmp_path / "t.jsonl"),
            "m", "log", str(tmp_path / "no_such_tm.jsonl")]
    env = {k: v for k, v in os.environ.items() if k != "KIT_TM_ALLOW_EMPTY"}
    r = subprocess.run(base, capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode != 0 and "translation memory" in (r.stdout + r.stderr) and "not found" in (r.stdout + r.stderr)


def test_entrypoint_is_not_exempted_from_the_collision_guard():
    """The entrypoint's colliding tokens are executable compat code (state-path fallback, cache volume detection, package-dir test):
    an exemption would collapse the old and new name onto one token without any error."""
    for line in ("27b", "nf"):
        p = os.path.join(KIT, "data", "collision_ok_1007_%s.json" % line)
        with open(p, encoding="utf-8") as fh:
            ok = json.load(fh)
        assert not [k for k in ok if k.endswith("weg2-release/entrypoint.sh")], (line, "entrypoint.sh exempted")
        for words in ok.values():
            assert isinstance(words, list) and all(isinstance(w, str) for w in words)


def test_entrypoint_content_is_locked_in_the_engine():
    """Fix round 2: the entrypoint is not exempted (test above) but content-locked: the mechanical pass leaves its bytes (F0-B/F0-G own it),
    so the survey has no name-rule collision to report for it; the file still moves with its directory."""
    sys.path.insert(0, KIT)
    import rename_to_flliper as R
    for p in ("docker/weg2-release/entrypoint.sh", "docker/pd" + "flip-release/entrypoint.sh"):
        assert R.in_path_scope(p) and not R.in_scope(p), p
    assert R.in_scope("docker/weg2-release/README_RELEASE_DRAFT.md")


def test_survey_covers_the_ident_fix_guard_of_release_rename():
    """Fix round 2: the survey's exit code is the acceptance of BOTH guards. A tiny repo with a German word whose target is already a
    NAME of the same file: strict rule = open, default (refined, as release_rename.sh) = open only where a merge is possible."""
    import tempfile
    d = tempfile.mkdtemp()
    def g(*a):
        subprocess.run(["git", "-C", d, *a], check=True, capture_output=True)
    g("init", "-q")
    os.makedirs(os.path.join(d, "python"))
    with open(os.path.join(d, "python", "a.py"), "w") as fh:
        german = "vor" "lauf"      # split: the kit pass over this very file must not turn the German test word into its English target
        fh.write("def f():\n    %s = 1\n    warmup = 2\n    return %s + warmup\n" % (german, german))
    with open(os.path.join(d, "python", "b.json"), "w") as fh:
        fh.write('{"note": "the %s phase is the warmup"}\n' % german)
    g("add", "-A")
    env = {k: v for k, v in os.environ.items() if k not in ("COLLISION_OK_FILE", "IDENT_COLLISION_OK_FILE", "FIXMAP_EXTRA", "IDENT_FIX_COLLISION")}
    r = subprocess.run([sys.executable, os.path.join(KIT, "collision_survey.py"), d], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "ident_fix (step 2" in r.stdout and "  F python/a.py" in r.stdout and "  F python/b.json" not in r.stdout, r.stdout
    env["IDENT_FIX_COLLISION"] = "strict"
    r = subprocess.run([sys.executable, os.path.join(KIT, "collision_survey.py"), d], capture_output=True, text=True, env=env, timeout=120)
    assert "  F python/b.json" in r.stdout, r.stdout


def test_ident_collision_ok_file_is_well_formed_and_scope_separate():
    """data/ident_collision_ok_1007.json lists only (path, old word) pairs that never meet in one function scope."""
    p = os.path.join(KIT, "data", "ident_collision_ok_1007.json")
    with open(p, encoding="utf-8") as fh:
        ok = {k: v for k, v in json.load(fh).items() if k != "_comment"}
    assert ok and all(isinstance(v, list) and v for v in ok.values())
    with open(os.path.join(KIT, "data", "merged_0928.json"), encoding="utf-8") as fh:
        imap = json.load(fh)
    import ast
    fn = (ast.FunctionDef, ast.AsyncFunctionDef)

    def names(node):
        out = set()
        for x in ast.walk(node):
            if isinstance(x, ast.Name):
                out.add(x.id)
            elif isinstance(x, ast.arg):
                out.add(x.arg)
            elif isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(x.name)
        return out

    for path, words in ok.items():
        src = os.path.join(REPO, path)
        if not os.path.isfile(src):
            continue                      # another line's tree may not carry the file
        tree = ast.parse(open(src, encoding="utf-8").read())
        for w in words:
            tgt = imap[w]
            for s in ast.walk(tree):
                if isinstance(s, fn):
                    n = names(s)
                    assert not (w in n and tgt in n), (path, w, tgt, s.name)
