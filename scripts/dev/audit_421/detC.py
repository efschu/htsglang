"""DETECTOR C -- "gate honoured asymmetrically".

An escape-hatch flag/env has REAL consumers -- so any consumer-count or grep
detector sees it as wired -- but a structurally parallel sibling path that the
flag's OWN documentation claims to cover never consults it. Half the feature
is live and half is silently dead.

The shape it is built for (#197, pre-fix ef6f8bc0a2^):

    gguf_dense_vocab()  docstring: "embed_tokens/lm_head are built with the
                                    GGUF quant_config"
    honoured by:  Qwen3_5ForCausalLM.__init__   -> self.embed_tokens = ...
    NOT honoured: Qwen3VLForConditionalGeneration.__init__ -> self.lm_head = ...
                  (reached by the 27B through inheritance)

Algorithm
    1. gate set G  = {ENV_NAME} + every function whose body reads ENV_NAME
       (the predicate helpers).
    2. doc text    = docstrings + surrounding comment block of every prod
       occurrence of a name in G, + the environ.py declaration block.
    3. entities    = identifiers in that text that also exist in the tree as an
       assigned attribute (`self.X = ...`), a def or a class. Prose-only words
       are dropped by that existence requirement.
    4. def sites   = every `self.X = <Call>` for each entity X.
    5. honoured    = the enclosing function of the def site references a name
       in G (directly, or via a same-module helper one hop away).
    6. scope       = "blast radius": modules that reference G, plus modules
       reachable in one import hop AND supplying a base class to a class in a
       G-referencing module (the inheritance link is what makes the sibling
       path structurally parallel rather than merely nearby).
    7. FIRE when at least one entity has an honoured def site in scope and at
       least one has an unhonoured def site in scope.

Usage:
    python3 /tmp/a421/detC.py <tree_root> --env SGLANG_FOO [--scope inherit|import|refonly]
    python3 /tmp/a421/detC.py <tree_root> --auto          # every fork env
"""

import argparse
import ast
import re
import sys

sys.path.insert(0, "/tmp/a421")
from astlib import Index, is_fork_file, is_test_path  # noqa: E402

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
STOP_WORDS = set(
    """
the and for with that this from into when what only default true false none
not but are was were has have had its it's they them their there here also
which while every each both same other than then just like such over under
above below because since after before during about between within without
per via out off out_of set unset flag env environment variable value values
path paths file files line lines code codes name names class classes module
modules function functions method methods param params arg args kwargs return
returns type types int str bool float list dict tuple optional sequence
gguf cuda nvml torch python sglang srt test tests note noqa pragma todo fixme
one two three four rank ranks world size sizes group groups layer layers
model models load loads loading weight weights tensor tensors quant config
configs enable enabled disable disabled use uses used using build built
builds make makes made keep keeps kept still same case cases path way ways
half halves side sides pair pairs whole full part parts step steps
""".split()
)


def env_reads(idx, env):
    """Function names whose body reads `env` -> the predicate helpers."""
    helpers = set()
    pat = re.compile(r"\b" + re.escape(env) + r"\b")
    for rel, tree in idx.trees.items():
        if is_test_path(rel):
            continue
        src = idx.files[rel]
        if env not in src:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    seg = ast.get_source_segment(src, node) or ""
                except Exception:
                    seg = ""
                if pat.search(seg):
                    # innermost function that mentions it, and it must be small
                    # enough to be a predicate rather than a whole feature
                    if len(seg.splitlines()) <= 40:
                        helpers.add(node.name)
    return helpers


def gate_doc_text(idx, gate_names):
    """Docstrings + comment blocks around every prod mention of a gate name."""
    chunks = []
    pat = re.compile(r"\b(" + "|".join(re.escape(g) for g in gate_names) + r")\b")
    for rel, tree in idx.trees.items():
        if is_test_path(rel):
            continue
        src = idx.files[rel]
        if not pat.search(src):
            continue
        lines = idx.lines[rel]
        for i, line in enumerate(lines):
            if not pat.search(line):
                continue
            # take the contiguous comment block / docstring neighbourhood
            lo, hi = max(0, i - 8), min(len(lines), i + 9)
            chunks.append("\n".join(lines[lo:hi]))
        # plus every docstring of a function/class that mentions a gate name
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
            ):
                d = ast.get_docstring(node)
                if d and pat.search(d):
                    chunks.append(d)
    return "\n".join(chunks)


def attribute_assign_sites(idx):
    """{attr_name: [(rel, lineno, enclosing_func_node)]} for `self.X = <Call>`."""
    out = {}
    for rel, tree in idx.trees.items():
        if is_test_path(rel):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    fn = idx.enclosing_func(node)
                    out.setdefault(t.attr, []).append((rel, node.lineno, fn))
    return out


def defined_symbols(idx):
    syms = set()
    for rel, tree in idx.trees.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                syms.add(node.name)
    return syms


def module_of(rel):
    return (
        rel[len("python/") : -3].replace("/", ".")
        if rel.startswith("python/")
        else None
    )


def import_graph(idx):
    """{rel: set(imported module strings)}"""
    g = {}
    for rel, tree in idx.trees.items():
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    mods.add(a.name)
        g[rel] = mods
    return g


def base_class_modules(idx, rel):
    """Modules that supply a base class to a class defined in `rel`."""
    tree = idx.trees[rel]
    bases = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            for b in n.bases:
                nm = (
                    b.id
                    if isinstance(b, ast.Name)
                    else (b.attr if isinstance(b, ast.Attribute) else None)
                )
                if nm:
                    bases.add(nm)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                if a.name in bases or (a.asname and a.asname in bases):
                    mods.add(n.module)
    return mods


def fn_references_gate(idx, rel, fn, gate_names, depth=1):
    """Does this function reach the gate directly or via a same-module helper?"""
    if fn is None:
        return False
    src = idx.files[rel]
    try:
        seg = ast.get_source_segment(src, fn) or ""
    except Exception:
        seg = ""
    pat = re.compile(r"\b(" + "|".join(re.escape(g) for g in gate_names) + r")\b")
    if pat.search(seg):
        return True
    if depth <= 0:
        return False
    # one hop through same-module helpers
    called = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            called.add(
                f.id
                if isinstance(f, ast.Name)
                else (f.attr if isinstance(f, ast.Attribute) else None)
            )
    for other in ast.walk(idx.trees[rel]):
        if (
            isinstance(other, (ast.FunctionDef, ast.AsyncFunctionDef))
            and other.name in called
        ):
            if fn_references_gate(idx, rel, other, gate_names, depth - 1):
                return True
    return False


def run_env(idx, env, scope="inherit", verbose=True):
    helpers = env_reads(idx, env)
    gate = {env} | helpers
    doc = gate_doc_text(idx, gate)
    syms = defined_symbols(idx)
    attrs = attribute_assign_sites(idx)
    words = {w for w in WORD_RE.findall(doc)}
    entities = sorted(
        w
        for w in words
        if w.lower() not in STOP_WORDS
        and w not in gate
        and (w in attrs or w in syms)
        and w in attrs
    )

    # ---- blast radius
    ref_mods = set()
    pat = re.compile(r"\b(" + "|".join(re.escape(g) for g in gate) + r")\b")
    for rel, src in idx.files.items():
        if is_test_path(rel):
            continue
        if pat.search(src):
            ref_mods.add(rel)
    scope_rels = set(ref_mods)
    if scope in ("inherit", "import"):
        ig = import_graph(idx)
        mod2rel = {module_of(r): r for r in idx.trees if module_of(r)}
        for r in list(ref_mods):
            want = base_class_modules(idx, r) if scope == "inherit" else ig[r]
            for m in want:
                if m in mod2rel:
                    scope_rels.add(mod2rel[m])

    honoured, unhonoured = [], []
    for e in entities:
        for rel, line, fn in attrs.get(e, []):
            if rel not in scope_rels:
                continue
            ok = fn_references_gate(idx, rel, fn, gate)
            rec = dict(entity=e, rel=rel, line=line, fn=(fn.name if fn else "<module>"))
            (honoured if ok else unhonoured).append(rec)

    # ---- RANKING: the signal is a DOC SIBLING PAIR.
    # Entity X is a STRONG candidate when (a) every def site of X in scope
    # ignores the gate, and (b) X is named on the same documentation LINE as an
    # entity Y that does honour it. "embed_tokens/lm_head are built with the
    # GGUF quant_config" is exactly that sentence; `pp_group` appearing eight
    # lines below an unrelated mention is not.
    hon_names = {h["entity"] for h in honoured}
    unh_names = {h["entity"] for h in unhonoured}
    fully_unhonoured = unh_names - hon_names
    siblings = {}
    for line in doc.splitlines():
        present = [w for w in set(WORD_RE.findall(line)) if w in entities]
        for x in present:
            for y in present:
                if x != y:
                    siblings.setdefault(x, set()).add(y)
    strong = sorted(x for x in fully_unhonoured if siblings.get(x, set()) & hon_names)
    strong_sites = [h for h in unhonoured if h["entity"] in strong]

    fired = bool(strong_sites)
    return dict(
        env=env,
        helpers=sorted(helpers),
        entities=entities,
        scope_files=sorted(scope_rels),
        ref_files=sorted(ref_mods),
        honoured=honoured,
        unhonoured=unhonoured,
        strong=strong,
        strong_sites=strong_sites,
        siblings={k: sorted(v) for k, v in siblings.items() if k in strong},
        fired=fired,
        weak_fired=bool(honoured) and bool(unhonoured),
    )


def fork_envs(idx):
    envs = set()
    for rel, src in idx.files.items():
        if is_test_path(rel) or not is_fork_file(rel):
            continue
        envs |= set(re.findall(r"\bSGLANG_[A-Z0-9_]{3,}\b", src))
    # plus envs added to inherited files by the fork: cheap approximation --
    # every SGLANG_ env whose *only* references are in fork files
    return sorted(envs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--env", default=None)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument(
        "--scope", default="inherit", choices=("inherit", "import", "refonly")
    )
    a = ap.parse_args()
    idx = Index(a.root)
    targets = [a.env] if a.env else fork_envs(idx)
    print(f"# DETECTOR C on {a.root} scope={a.scope} targets={len(targets)}")
    nfired = 0
    for env in targets:
        r = run_env(idx, env, a.scope)
        if not r["entities"]:
            continue
        if not r["fired"]:
            if a.env:
                print(
                    f"\n--- {env}: NO STRONG FIRE (honoured={len(r['honoured'])} "
                    f"unhonoured={len(r['unhonoured'])} weak_fired={r['weak_fired']})"
                )
                print(f"    entities={r['entities']}")
                for h in r["unhonoured"]:
                    print(
                        f"    weak-unhonoured: {h['entity']:20s} {h['rel']}:{h['line']} in {h['fn']}()"
                    )
            continue
        nfired += 1
        print(f"\n===== ASYMMETRIC GATE: {env} =====")
        print(f"  predicate helpers: {r['helpers']}")
        print(
            "  STRONG candidates (fully unhonoured + doc-sibling of an honoured entity):"
        )
        for h in r["strong_sites"]:
            print(
                f"     {h['entity']:20s} {h['rel']}:{h['line']}  in {h['fn']}()"
                f"   doc-siblings={r['siblings'].get(h['entity'], [])}"
            )
        print(f"  honoured sites ({len(r['honoured'])}):")
        for h in r["honoured"]:
            print(f"     {h['entity']:20s} {h['rel']}:{h['line']}  in {h['fn']}()")
        print(
            f"  other unhonoured ({len([h for h in r['unhonoured'] if h['entity'] not in r['strong']])}) [weak tier, not reported]"
        )
    print(f"\n# fired: {nfired}/{len(targets)}")


if __name__ == "__main__":
    main()
