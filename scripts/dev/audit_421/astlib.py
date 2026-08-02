"""Shared AST index for the #421 sub-job A detectors (Detector B / Detector C).

Pure static analysis: parses a *tree export* (see /tmp/a421/tree_*), never
imports sglang, never touches a GPU.
"""

import ast
import os
import re

UPSTREAM_FILES = set()
_uf = "/tmp/a421/upstream_files.txt"
if os.path.exists(_uf):
    UPSTREAM_FILES = {line.strip() for line in open(_uf) if line.strip()}

TEST_RE = re.compile(
    r"(^|/)(tests?|benchmark|bench|examples?|docs)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$"
)


def is_test_path(rel):
    return bool(TEST_RE.search("/" + rel))


def is_fork_file(rel):
    """True when the path does not exist in upstream/main -> fork-added."""
    return rel not in UPSTREAM_FILES


class Index:
    def __init__(self, root):
        self.root = root
        self.files = {}  # rel -> source
        self.trees = {}  # rel -> ast.Module
        self.lines = {}  # rel -> list[str]
        for base, dirs, fs in os.walk(root):
            dirs[:] = [
                d for d in dirs if d not in (".git", "node_modules", "__pycache__")
            ]
            for f in fs:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(base, f)
                rel = os.path.relpath(p, root)
                try:
                    src = open(p, errors="ignore").read()
                    self.trees[rel] = ast.parse(src)
                except Exception:
                    continue
                self.files[rel] = src
                self.lines[rel] = src.splitlines()
        self._link_parents()

    def _link_parents(self):
        for rel, tree in self.trees.items():
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child._parent = node
                    child._rel = rel
            tree._rel = rel

    def enclosing_func(self, node):
        n = getattr(node, "_parent", None)
        while n is not None:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return n
            n = getattr(n, "_parent", None)
        return None

    def enclosing_class(self, node):
        n = getattr(node, "_parent", None)
        while n is not None:
            if isinstance(n, ast.ClassDef):
                return n
            n = getattr(n, "_parent", None)
        return None


def call_name(node):
    """Simple callee name of a Call node: f(...) -> 'f', a.b.f(...) -> 'f'."""
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def is_none_literal(node):
    return isinstance(node, ast.Constant) and node.value is None


def src_seg(idx, rel, node):
    try:
        return ast.get_source_segment(idx.files[rel], node) or ""
    except Exception:
        return ""
