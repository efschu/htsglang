#!/bin/bash
# $1 = worktree, $2 = renamed(1)|old(0); maps the base test paths through the tool's own path rule
K=${RELEASE_KIT:-$(cd "$(dirname "$0")" && pwd)}; F=${RELEASE_WORK:-/spinning/flliper/work/release}; W=$1; R=$2; cd "$W"
T=$(${PY:-/spinning/htsglang-gpu/.venv/bin/python} -c "
import sys; sys.path.insert(0,'$K'); import rename_to_flliper as R, os
ps=[l.strip() for l in open('${TESTS_LIST:-$K/tests_all.txt}') if l.strip()]+['test/registered/unit/entrypoints/test_compat_shims.py']
ps=[R.rewrite_path(p, True) if '$R'=='1' else p for p in ps]
print(' '.join(p for p in ps if os.path.exists(p)))")
H=$F/home-$(basename $W); rm -rf "$H"; mkdir -p "$H/.cache"; cp -a /root/.cache/sglang "$H/.cache/sglang"
# renamed tree: the state-dir link the server entry points make at start (compat_shims.link_legacy_cache_dir)
[ "$R" = 1 ] && ln -s "$H/.cache/sglang" "$H/.cache/flliper"
HOME=$H PYTHONPATH=$W/python exec ${RIG_TEST_WRAP:-$K/capped_run.sh} ${PY:-/spinning/htsglang-gpu/.venv/bin/python} -m pytest -q -p no:cacheprovider ${PYTEST_EXTRA:-} --color=no --continue-on-collection-errors -rfEs $T
